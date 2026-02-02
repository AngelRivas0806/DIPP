"""
Script de evaluación para datasets ETH/UCY procesados
Open-loop testing (predicción sin replanning)
"""

import sys
import torch
import argparse
import numpy as np
import os
import logging
from tqdm import tqdm
from utils.train_utils import DrivingData, bicycle_model, select_future
from utils.visualization import plot_trajectory_prediction, plot_multiple_samples
from model.predictor import Predictor
from model.planner import MotionPlanner
from torch.utils.data import DataLoader


def compute_ade_fde(predictions, ground_truth):
    """
    Compute Average Displacement Error and Final Displacement Error
    
    Args:
        predictions: (batch, pred_len, 2) predicted trajectory
        ground_truth: (batch, pred_len, 2) ground truth trajectory
    
    Returns:
        ade, fde: scalars
    """
    # Calcular desplazamiento en cada timestep
    displacement = torch.norm(predictions - ground_truth, dim=-1)  # (batch, pred_len)
    
    # ADE: promedio de todos los desplazamientos
    ade = torch.mean(displacement)
    
    # FDE: desplazamiento en el punto final
    fde = torch.mean(displacement[:, -1])
    
    return ade.item(), fde.item()


def evaluate_model(model, data_loader, device, use_planning=False, planner=None, 
                  visualize=False, vis_path=None, num_vis_samples=10):
    """
    Evaluar el modelo en un conjunto de datos
    
    Args:
        visualize: Si generar visualizaciones
        vis_path: Directorio para guardar visualizaciones
        num_vis_samples: Número de muestras a visualizar
    """
    model.eval()
    
    # Métricas acumuladas
    ego_ades, ego_fdes = [], []
    neighbor_ades, neighbor_fdes = [], []
    
    # Datos para visualización
    vis_samples = []
    
    with torch.no_grad():
        for batch_idx, batch in enumerate(tqdm(data_loader, desc="Evaluating")):
            # Preparar datos
            ego          = batch[0].to(device)
            neighbors    = batch[1].to(device)
            ground_truth = batch[2].to(device)
            
            current_state= torch.cat([ego.unsqueeze(1), neighbors[..., :-1]], dim=1)[:, :, -1]

            print(neighbors.shape)
            # Predicción
            plans, predictions, scores, cost_function_weights = model(ego, neighbors)
            print(predictions.shape)
            # Generar trayectorias de planes
            plan_trajs = torch.stack([bicycle_model(plans[:, i], ego[:, -1])[:, :, :3] for i in range(3)], dim=1)
            
            # Seleccionar mejor futuro
            plan_traj, prediction = select_future(plan_trajs, predictions, scores)
            
            # Si usamos planning, refinar con el planner
            if use_planning and planner is not None:
                # Necesitamos los controles originales (antes de bicycle_model)
                # Seleccionar el mejor plan de control basado en scores
                best_mode_idx = torch.argmax(scores, dim=1)
                plan_control = plans[torch.arange(ego.shape[0]), best_mode_idx]  # (batch, 12, 2)
                
                # Create dummy ref_line_info
                ref_line_info = torch.zeros(ego.shape[0], 1200, 5).to(device)
                
                planner_inputs = {
                    "control_variables": plan_control.view(ego.shape[0], 24),  # 12 timesteps * 2 controls
                    "predictions": prediction,
                    "ref_line_info": ref_line_info,
                    "current_state": current_state
                }
                
                for i in range(cost_function_weights.shape[1]):
                    planner_inputs[f'cost_function_weight_{i+1}'] = cost_function_weights[:, i].unsqueeze(1)
                
                final_values, info = planner.layer.forward(planner_inputs)
                refined_control = final_values["control_variables"].view(-1, 12, 2)
                plan_traj = bicycle_model(refined_control, ego[:, -1])[:, :, :3]
            
            # Calcular métricas para ego
            ego_gt = ground_truth[:, 0, :, :2]  # (batch, pred_len, 2)
            ego_pred = plan_traj[:, :, :2]  # (batch, pred_len, 2)
            
            ade, fde = compute_ade_fde(ego_pred, ego_gt)
            ego_ades.append(ade)
            ego_fdes.append(fde)
            
            # Calcular métricas para vecinos
            # Filtrar vecinos válidos (que tienen ground truth)
            neighbor_gt = ground_truth[:, 1:, :, :2]  # (batch, num_neighbors, pred_len, 2)
            neighbor_pred = prediction[:, :, :, :2]  # (batch, num_neighbors, pred_len, 2)
            
            # Máscara de vecinos válidos
            valid_mask = torch.sum(torch.abs(neighbor_gt), dim=(2, 3)) > 0  # (batch, num_neighbors)
            
            for b in range(neighbor_gt.shape[0]):
                for n in range(neighbor_gt.shape[1]):
                    if valid_mask[b, n]:
                        ade, fde = compute_ade_fde(
                            neighbor_pred[b:b+1, n:n+1, :, :],
                            neighbor_gt[b:b+1, n:n+1, :, :]
                        )
                        neighbor_ades.append(ade)
                        neighbor_fdes.append(fde)
            
            # ========== VISUALIZACIÓN ==========
            if visualize and len(vis_samples) < num_vis_samples:
                # Guardar algunas muestras para visualizar
                for b in range(min(ego.shape[0], num_vis_samples - len(vis_samples))):
                    sample_data = {
                        'ego_hist': ego[b, :, :2].cpu().numpy(),  # (obs_len, 2)
                        'ego_pred': ego_pred[b, :, :].cpu().numpy(),  # (pred_len, 2)
                        'ego_gt': ego_gt[b, :, :].cpu().numpy(),  # (pred_len, 2)
                        'neighbors_hist': neighbors[b, :, :, :2].cpu().numpy(),  # (num_neighbors, obs_len, 2)
                        'neighbors_pred': neighbor_pred[b, :, :, :].cpu().numpy(),  # (num_neighbors, pred_len, 2)
                        'neighbors_gt': neighbor_gt[b, :, :, :].cpu().numpy(),  # (num_neighbors, pred_len, 2)
                    }
                    vis_samples.append(sample_data)
    import sys
    sys.exit()
    # Calcular promedios
    results = {
        'ego_ADE': np.mean(ego_ades),
        'ego_FDE': np.mean(ego_fdes),
        'neighbor_ADE': np.mean(neighbor_ades) if neighbor_ades else 0.0,
        'neighbor_FDE': np.mean(neighbor_fdes) if neighbor_fdes else 0.0,
        'num_samples': len(ego_ades),
        'num_neighbors_evaluated': len(neighbor_ades),
        'vis_samples': vis_samples
    }
    
    return results


def main():
    parser = argparse.ArgumentParser(description='Test ETH/UCY')
    parser.add_argument('--model_path', type=str, required=True, help='Path to trained model')
    parser.add_argument('--test_set', type=str, required=True, help='Path to test dataset')
    parser.add_argument('--name', type=str, default='test_result', help='Name for logging')
    parser.add_argument('--batch_size', type=int, default=32, help='Batch size for testing')
    parser.add_argument('--use_planning', action='store_true', help='Use planning module')
    parser.add_argument('--device', type=str, default='cuda', help='Device to use')
    parser.add_argument('--visualize', action='store_true', help='Generate visualizations')
    parser.add_argument('--num_vis_samples', type=int, default=10, help='Number of samples to visualize')
    
    args = parser.parse_args()
    
    # Setup logging
    log_path = f"./testing_log/{args.name}/"
    os.makedirs(log_path, exist_ok=True)
    logging.basicConfig(
        level=logging.INFO,
        format='[%(levelname)s %(asctime)s] %(message)s',
        handlers=[
            logging.FileHandler(f"{log_path}/test.log"),
            logging.StreamHandler()
        ]
    )
    
    logging.info(f"{'='*60}")
    logging.info(f"Testing: {args.name}")
    logging.info(f"Model: {args.model_path}")
    logging.info(f"Test set: {args.test_set}")
    logging.info(f"Use planning: {args.use_planning}")
    logging.info(f"Device: {args.device}")
    logging.info(f"{'='*60}")
    
    # Cargar modelo
    logging.info("Loading model...")
    predictor = Predictor(12).to(args.device)
    predictor.load_state_dict(torch.load(args.model_path, map_location=args.device))
    predictor.eval()
    logging.info("Model loaded successfully!")
    
    # Setup planner si es necesario
    planner = None
    if args.use_planning:
        logging.info("Setting up planner...")
        trajectory_len, feature_len = 12, 9
        planner = MotionPlanner(trajectory_len, feature_len, args.device)
        logging.info("Planner ready!")
    
    # Cargar datos
    logging.info("Loading test data...")
    test_set = DrivingData(args.test_set + '/*')
    test_loader = DataLoader(test_set, batch_size=args.batch_size, shuffle=False, num_workers=4)
    logging.info(f"Test set: {len(test_set)} samples")
    
    # Evaluar
    logging.info(f"\n{'='*60}")
    logging.info("Starting evaluation...")
    logging.info(f"{'='*60}\n")
    
    vis_path = f"{log_path}/visualizations" if args.visualize else None
    results = evaluate_model(predictor, test_loader, args.device, 
                            args.use_planning, planner,
                            visualize=args.visualize, 
                            vis_path=vis_path,
                            num_vis_samples=args.num_vis_samples)
    
    # Generar visualizaciones
    if args.visualize and len(results['vis_samples']) > 0:
        logging.info(f"\n{'='*60}")
        logging.info("Generating visualizations...")
        logging.info(f"{'='*60}\n")
        
        # Visualizaciones individuales
        for idx, sample in enumerate(results['vis_samples']):
            plot_trajectory_prediction(
                ego_hist=sample['ego_hist'],
                ego_pred=sample['ego_pred'],
                ego_gt=sample['ego_gt'],
                neighbors_hist=sample['neighbors_hist'],
                neighbors_pred=sample['neighbors_pred'],
                neighbors_gt=sample['neighbors_gt'],
                save_path=vis_path,
                show=False,
                sample_id=idx
            )
        
        # Cuadrícula con múltiples muestras
        plot_multiple_samples(results['vis_samples'], vis_path, max_samples=6)
        
        logging.info(f"Visualizations saved to: {vis_path}")
    
    # Mostrar resultados
    logging.info(f"\n{'='*60}")
    logging.info("RESULTS")
    logging.info(f"{'='*60}")
    logging.info(f"Ego Agent:")
    logging.info(f"  ADE: {results['ego_ADE']:.4f} m")
    logging.info(f"  FDE: {results['ego_FDE']:.4f} m")
    logging.info(f"\nNeighbor Agents:")
    logging.info(f"  ADE: {results['neighbor_ADE']:.4f} m")
    logging.info(f"  FDE: {results['neighbor_FDE']:.4f} m")
    logging.info(f"\nStatistics:")
    logging.info(f"  Total samples: {results['num_samples']}")
    logging.info(f"  Neighbors evaluated: {results['num_neighbors_evaluated']}")
    logging.info(f"{'='*60}\n")
    
    # Guardar resultados
    results_file = f"{log_path}/results.txt"
    with open(results_file, 'w') as f:
        f.write(f"Model: {args.model_path}\n")
        f.write(f"Test set: {args.test_set}\n")
        f.write(f"Use planning: {args.use_planning}\n")
        f.write(f"\n{'='*60}\n")
        f.write(f"Ego ADE: {results['ego_ADE']:.4f} m\n")
        f.write(f"Ego FDE: {results['ego_FDE']:.4f} m\n")
        f.write(f"Neighbor ADE: {results['neighbor_ADE']:.4f} m\n")
        f.write(f"Neighbor FDE: {results['neighbor_FDE']:.4f} m\n")
        f.write(f"\nTotal samples: {results['num_samples']}\n")
        f.write(f"Neighbors evaluated: {results['num_neighbors_evaluated']}\n")
    
    logging.info(f"Results saved to: {results_file}")


if __name__ == "__main__":
    main()
