import sys
import torch
import argparse
import numpy as np
import os
import logging
from tqdm import tqdm
from utils.train_utils import DrivingData, bicycle_model, select_future
from utils.visualization import plot_trajectory_prediction, plot_ego_top3_modes
from model.predictor import Predictor, NUM_MODES
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
                  visualize=False, vis_path=None, num_vis_samples=10, min_neighbors=0,
                  raw_dataset=None):
    """
    Evaluar el modelo en un conjunto de datos
    
    Args:
        visualize: Si generar visualizaciones
        vis_path: Directorio para guardar visualizaciones
        num_vis_samples: Número de muestras a visualizar
    """
    model.eval()
    
    # Accumulated metrics
    ego_ades, ego_fdes           = [], []
    neighbor_ades, neighbor_fdes = [], []
    
    # Precompute candidate indices for visualization (uniformly distributed)
    vis_candidate_indices = set()
    if visualize and raw_dataset is not None:
        neighbors_raw = raw_dataset.neighbors  # (N,K,obs,7)
        valid_counts = np.array([
            int(np.sum(np.sum(np.abs(neighbors_raw[i]), axis=(1, 2)) > 0))
            for i in range(len(raw_dataset))
        ])
        candidates = np.where(valid_counts >= min_neighbors)[0]
        if len(candidates) >= num_vis_samples:
            step = len(candidates) // num_vis_samples
            selected = candidates[::step][:num_vis_samples]
        else:
            selected = candidates
        vis_candidate_indices = set(int(x) for x in selected)

    # Datos para visualización
    vis_samples = []
    
    with torch.no_grad():
        for batch_idx, batch in enumerate(tqdm(data_loader, desc="Evaluating")):
            # Preparar datos
            ego          = batch[0].to(device)
            neighbors    = batch[1].to(device)
            ground_truth = batch[2].to(device)
            
            current_state= torch.cat([ego.unsqueeze(1), neighbors[..., :-1]], dim=1)[:, :, -1]

            # Predicción
            plans, predictions, scores, cost_function_weights = model(ego, neighbors)
            # Generar trayectorias de planes
            plan_trajs = torch.stack([bicycle_model(plans[:, i], ego[:, -1])[:, :, :3] for i in range(NUM_MODES)], dim=1)
            
            # Seleccionar mejor futuro
            plan_traj, prediction = select_future(current_state, plan_trajs, predictions, scores)
            
            # Si usamos planning, refinar con el planner
            if use_planning and planner is not None:
                # Necesitamos los controles originales (antes de bicycle_model)
                # Seleccionar el mejor plan de control basado en scores
                best_mode_idx = torch.argmax(scores, dim=1)
                plan_control = plans[torch.arange(ego.shape[0]), best_mode_idx]  # (batch, 12, 2)
                
                # Ground truth trajectory for trajectory following cost
                gt_trajectory = ground_truth[:, 0, :, :2]  # ego's ground truth future (x,y)
                
                planner_inputs = {
                    "control_variables": plan_control.view(ego.shape[0], 24),
                    "predictions": prediction,
                    "current_state": current_state,
                }
                
                w = cost_function_weights  # (B, 5)
                planner_inputs["w_acc"] = w[:, 0].unsqueeze(1)
                planner_inputs["w_jerk"] = w[:, 1].unsqueeze(1)
                planner_inputs["w_steer"] = w[:, 2].unsqueeze(1)
                planner_inputs["w_steer_change"] = w[:, 3].unsqueeze(1)
                planner_inputs["w_collision"] = w[:, 4].unsqueeze(1)
                
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
            
            # Mask for valid neighbors
            valid_mask = torch.sum(torch.abs(neighbor_gt), dim=(2, 3)) > 0  # (B,N)
            
            # For any data in the batch
            for b in range(neighbor_gt.shape[0]):
                # For all neighbors
                for n in range(neighbor_gt.shape[1]):
                    if valid_mask[b, n]:
                        # Aplanar a (1, pred_len, 2) para compute_ade_fde
                        ade, fde = compute_ade_fde(
                            neighbor_pred[b, n:n+1, :, :],  # (1, pred_len, 2)
                            neighbor_gt[b, n:n+1, :, :]     # (1, pred_len, 2)
                        )
                        neighbor_ades.append(ade)
                        neighbor_fdes.append(fde)
            
            # ========== VISUALIZACIÓN ==========
            if visualize and len(vis_samples) < num_vis_samples:
                batch_start = batch_idx * data_loader.batch_size
                for b in range(ego.shape[0]):
                    if len(vis_samples) >= num_vis_samples:
                        break
                    global_idx = batch_start + b
                    if global_idx not in vis_candidate_indices:
                        continue

                    nb_hist = neighbors[b, :, :, :2].cpu().numpy()
                    n_valid = int(np.sum([np.sum(np.abs(nb_hist[n])) > 0 for n in range(nb_hist.shape[0])]))
                    # Máscara de vecinos válidos usando el flag (col 6 del último frame)
                    neigh_valid = (neighbors[b, :, -1, 6] > 0).cpu().numpy()  # (N,) bool

                    # --- Top-3 modos del ego ---
                    top3_modes = torch.argsort(scores[b], descending=True)[:3]              # (3,) tensor
                    all_probs  = torch.softmax(scores[b], dim=0)                            # (20,) prob real sobre todos los modos
                    top3_scores = all_probs[top3_modes].cpu().numpy()                       # (3,) prob relativa a los 20 modos

                    if use_planning and planner is not None:
                        # Refinar cada uno de los 3 modos con el planner
                        top3_trajs_list   = []
                        top3_nb_pred_list = []
                        for k in range(3):
                            mode_k = top3_modes[k].item()
                            ctrl_k = plans[b:b+1, mode_k]           # (1, 12, 2)
                            pred_k = predictions[b:b+1, mode_k]     # (1, N, 12, 2)
                            planner_inputs_k = {
                                "control_variables": ctrl_k.view(1, 24),
                                "predictions":       pred_k,
                                "current_state":     current_state[b:b+1],
                                "w_acc":          cost_function_weights[b:b+1, 0].unsqueeze(1),
                                "w_jerk":         cost_function_weights[b:b+1, 1].unsqueeze(1),
                                "w_steer":        cost_function_weights[b:b+1, 2].unsqueeze(1),
                                "w_steer_change": cost_function_weights[b:b+1, 3].unsqueeze(1),
                                "w_collision":    cost_function_weights[b:b+1, 4].unsqueeze(1),
                            }
                            final_k, _ = planner.layer.forward(planner_inputs_k)
                            refined_ctrl_k = final_k["control_variables"].view(1, 12, 2)
                            traj_k = bicycle_model(refined_ctrl_k, ego[b:b+1, -1])[:, :, :2]  # (1,12,2)
                            top3_trajs_list.append(traj_k.squeeze(0).cpu().numpy())        # (12,2)
                            top3_nb_pred_list.append(pred_k.squeeze(0).cpu().numpy())      # (N,12,2)
                        top3_trajs_norm    = np.stack(top3_trajs_list,   axis=0)  # (3,12,2)
                        top3_nb_preds_norm = np.stack(top3_nb_pred_list, axis=0)  # (3,N,12,2)
                    else:
                        # Sin planner: usar plan_trajs ya calculadas
                        top3_trajs_norm    = plan_trajs[b][top3_modes][:, :, :2].cpu().numpy()  # (3,12,2)
                        top3_nb_preds_norm = predictions[b][top3_modes][:, :, :, :2].cpu().numpy()  # (3,N,12,2)

                    # --- Desnormalización ---
                    if raw_dataset is not None:
                        raw_idx = batch_start + b
                        raw_ego = raw_dataset.ego[raw_idx]          # (obs, 6) sin normalizar
                        ego_pos0 = raw_ego[-1, :2].copy()           # traslación
                        vx, vy = float(raw_ego[-1, 2]), float(raw_ego[-1, 3])
                        heading = float(np.arctan2(vy, vx)) if abs(vx) > 1e-2 or abs(vy) > 1e-2 else 0.0
                        c, s = np.cos(heading), np.sin(heading)     # inverso: +heading

                        def denorm_xy(xy):
                            # xy: (..., 2)  rot con +heading luego traslada
                            xr = xy[..., 0] * c - xy[..., 1] * s
                            yr = xy[..., 0] * s + xy[..., 1] * c
                            out = np.stack([xr, yr], axis=-1)
                            out = out + ego_pos0
                            return out

                        ego_hist_dn   = denorm_xy(ego[b, :, :2].cpu().numpy())
                        ego_pred_dn   = denorm_xy(ego_pred[b, :, :].cpu().numpy())
                        ego_gt_dn     = denorm_xy(ego_gt[b, :, :].cpu().numpy())
                        nb_hist_dn    = denorm_xy(nb_hist)
                        nb_pred_dn    = denorm_xy(neighbor_pred[b, :, :, :].cpu().numpy())
                        nb_gt_dn      = denorm_xy(neighbor_gt[b, :, :, :].cpu().numpy())
                        top3_trajs_dn    = np.stack([denorm_xy(top3_trajs_norm[k])    for k in range(3)], axis=0)
                        top3_nb_preds_dn = np.stack([denorm_xy(top3_nb_preds_norm[k]) for k in range(3)], axis=0)
                    else:
                        ego_hist_dn   = ego[b, :, :2].cpu().numpy()
                        ego_pred_dn   = ego_pred[b, :, :].cpu().numpy()
                        ego_gt_dn     = ego_gt[b, :, :].cpu().numpy()
                        nb_hist_dn    = nb_hist
                        nb_pred_dn    = neighbor_pred[b, :, :, :].cpu().numpy()
                        nb_gt_dn      = neighbor_gt[b, :, :, :].cpu().numpy()
                        top3_trajs_dn    = top3_trajs_norm
                        top3_nb_preds_dn = top3_nb_preds_norm

                    sample_data = {
                        'ego_hist': ego_hist_dn,
                        'ego_pred': ego_pred_dn,
                        'ego_gt':   ego_gt_dn,
                        'neighbors_hist': nb_hist_dn,
                        'neighbors_pred': nb_pred_dn,
                        'neighbors_gt':   nb_gt_dn,
                        'num_valid_neighbors': int(n_valid),
                        'neighbors_valid':     neigh_valid,         # (N,) bool — flag del dataset
                        'ego_top3_trajs':      top3_trajs_dn,       # (3, pred_len, 2)
                        'ego_top3_scores':     top3_scores,         # (3,) probabilidades softmax
                        'ego_top3_nb_preds':   top3_nb_preds_dn,   # (3, N, pred_len, 2)
                    }
                    vis_samples.append(sample_data)
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
    parser.add_argument('--num_vis_samples', type=int, default=20, help='Number of samples to visualize')
    parser.add_argument('--min_neighbors', type=int, default=0, help='Minimum valid neighbors to include a sample in visualization')
    
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
        planner = MotionPlanner(trajectory_len, args.device, test=True)
        logging.info("Planner ready!")
    
    # Cargar datos
    logging.info("Loading test data...")
    test_set = DrivingData(args.test_set)
    test_loader = DataLoader(test_set, batch_size=args.batch_size, shuffle=False, num_workers=4)
    logging.info(f"Test set: {len(test_set)} samples")
    
    # Evaluar
    logging.info(f"\n{'='*60}")
    logging.info("Starting evaluation...")
    logging.info(f"{'='*60}\n")
    
    vis_path = f"{log_path}/visualizations"
    results = evaluate_model(predictor, test_loader, args.device, 
                            args.use_planning, planner,
                            visualize=True, 
                            vis_path=vis_path,
                            num_vis_samples=args.num_vis_samples,
                            min_neighbors=args.min_neighbors,
                            raw_dataset=test_set)
    
    # Generar visualizaciones
    if len(results['vis_samples']) > 0:
        logging.info(f"\n{'='*60}")
        logging.info("Generating visualizations...")
        logging.info(f"{'='*60}\n")
        
        # Visualizaciones de top-3 modos (una imagen con 3 subplots por muestra)
        for idx, sample in enumerate(results['vis_samples']):
            if 'ego_top3_trajs' in sample:
                plot_ego_top3_modes(
                    ego_hist=sample['ego_hist'],
                    ego_gt=sample['ego_gt'],
                    ego_top3_trajs=sample['ego_top3_trajs'],
                    ego_top3_scores=sample.get('ego_top3_scores', None),
                    ego_top3_nb_preds=sample.get('ego_top3_nb_preds', None),
                    neighbors_hist=sample['neighbors_hist'],
                    neighbors_valid=sample.get('neighbors_valid', None),
                    neighbors_gt=sample['neighbors_gt'],
                    save_path=vis_path,
                    sample_id=idx
                )
        
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
