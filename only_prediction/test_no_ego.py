"""
Script de testing para PredictorNoEgo.
Evalúa el modelo entrenado en el conjunto de test.
"""

import argparse
import os
import sys
import numpy as np
import torch
from torch import nn
import time
import matplotlib.pyplot as plt
from pathlib import Path

# Añadir directorio padre al path para importar
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from predictor_no_ego import PredictorNoEgo
from visualize_test_scenes import generate_scene_visualizations


def test_model(model, dataloader, device, save_dir=None):
    """Evalúa el modelo en el conjunto de test."""
    model.eval()
    
    all_ade = []
    all_fde = []
    all_misses = []
    
    # Para guardar TODAS las predicciones para visualización de escenas
    all_predictions = []
    all_scores = []
    all_gt = []
    all_observed = []
    
    print("\n" + "="*60)
    print("Testing PredictorNoEgo")
    print("="*60)
    
    start_time = time.time()
    
    with torch.no_grad():
        for batch_idx, (observed, future, indices) in enumerate(dataloader):
            observed = observed.to(device)
            future = future.to(device)
            
            # Forward pass
            predictions, scores = model(observed)
            # predictions: (batch, num_modes, future_steps, 2)
            # scores: (batch, num_modes)
            
            # Compute metrics
            _, batch_ade = compute_ade(predictions, future)
            _, batch_fde = compute_fde(predictions, future)
            _, batch_misses = compute_miss_rate(predictions, future, threshold=2.0)
            
            all_ade.extend(batch_ade.cpu().numpy())
            all_fde.extend(batch_fde.cpu().numpy())
            all_misses.extend(batch_misses.cpu().numpy())
            
            # Guardar TODAS las predicciones para visualización de escenas
            all_observed.append(observed.cpu().numpy())
            all_gt.append(future.cpu().numpy())
            all_predictions.append(predictions.cpu().numpy())
            all_scores.append(scores.cpu().numpy())
            
            # Progress
            if (batch_idx + 1) % 20 == 0:
                elapsed = time.time() - start_time
                samples_processed = (batch_idx + 1) * dataloader.batch_size
                print(f"  Progress: [{samples_processed}/{len(dataloader.dataset)}] "
                      f"samples processed in {elapsed:.1f}s")
    
    elapsed = time.time() - start_time
    
    # Convertir a arrays
    all_ade = np.array(all_ade)
    all_fde = np.array(all_fde)
    all_misses = np.array(all_misses)
    
    # Concatenar todas las predicciones para visualización de escenas
    all_observed = np.concatenate(all_observed, axis=0)  # (N, obs_len, 2)
    all_gt = np.concatenate(all_gt, axis=0)  # (N, pred_len, 2)
    all_predictions = np.concatenate(all_predictions, axis=0)  # (N, num_modes, pred_len, 2)
    all_scores = np.concatenate(all_scores, axis=0)  # (N, num_modes)
    
    # Calcular estadísticas
    mean_ade = all_ade.mean()
    mean_fde = all_fde.mean()
    miss_rate = all_misses.mean() * 100
    
    print("\n" + "="*60)
    print("TEST RESULTS")
    print("="*60)
    print(f"Number of test samples: {len(all_ade)}")
    print(f"Total time: {elapsed:.2f}s")
    print(f"Samples/second: {len(all_ade)/elapsed:.1f}")
    print(f"\n{'Metric':<20} {'Value':<15}")
    print("-"*60)
    print(f"{'minADE (m)':<20} {mean_ade:<15.4f}")
    print(f"{'minFDE (m)':<20} {mean_fde:<15.4f}")
    print(f"{'Miss Rate @ 2.0m (%)':<20} {miss_rate:<15.2f}")
    print("="*60)
    
    # Guardar resultados
    if save_dir:
        os.makedirs(save_dir, exist_ok=True)
        
        # Guardar métricas
        results_file = os.path.join(save_dir, 'test_results.txt')
        with open(results_file, 'w') as f:
            f.write("="*60 + "\n")
            f.write("TEST RESULTS - PredictorNoEgo\n")
            f.write("="*60 + "\n")
            f.write(f"Number of test samples: {len(all_ade)}\n")
            f.write(f"Total time: {elapsed:.2f}s\n")
            f.write(f"Samples/second: {len(all_ade)/elapsed:.1f}\n\n")
            f.write(f"{'Metric':<20} {'Value':<15}\n")
            f.write("-"*60 + "\n")
            f.write(f"{'minADE (m)':<20} {mean_ade:<15.4f}\n")
            f.write(f"{'minFDE (m)':<20} {mean_fde:<15.4f}\n")
            f.write(f"{'Miss Rate @ 2.0m (%)':<20} {miss_rate:<15.2f}\n")
            f.write("="*60 + "\n")
        
        print(f"\n✓ Results saved to: {results_file}")
        
        # Guardar métricas individuales
        np.savez(os.path.join(save_dir, 'test_metrics.npz'),
                 ade=all_ade,
                 fde=all_fde,
                 miss=all_misses)
        
        print(f"✓ Individual metrics saved to: {os.path.join(save_dir, 'test_metrics.npz')}")
        
        # Generar visualizaciones de escenas automáticamente
        print("\n" + "="*60)
        print("Generating scene visualizations...")
        print("="*60)
        generate_scene_visualizations(
            observed=all_observed,
            future=all_gt, 
            predictions=all_predictions,
            scores=all_scores,
            save_dir=save_dir,
            num_scenes=15,
            samples_per_scene=10  # Aumentado de 6 a 10 peatones por escena
        )
    
    return {
        'ade': mean_ade,
        'fde': mean_fde,
        'miss_rate': miss_rate
    }


def main():
    parser = argparse.ArgumentParser(description='Test PredictorNoEgo')
    
    # Datos
    parser.add_argument('--test_set', type=str, required=True,
                       help='Path to test data .npz file')
    
    # Modelo
    parser.add_argument('--checkpoint', type=str, required=True,
                       help='Path to model checkpoint (.pth)')
    parser.add_argument('--obs_len', type=int, default=8)
    parser.add_argument('--future_steps', type=int, default=8)
    parser.add_argument('--num_modes', type=int, default=20)
    
    # Testing
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--device', type=str, default='cuda', 
                       choices=['cuda', 'cpu'])
    
    # Output
    parser.add_argument('--save_dir', type=str, 
                       default='only_prediction/test_results',
                       help='Directory to save test results')
    
    args = parser.parse_args()
    
    # Setup
    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    
    print(f"\n{'='*60}")
    print(f"PredictorNoEgo - Testing")
    print(f"{'='*60}")
    print(f"Device: {device}")
    print(f"Checkpoint: {args.checkpoint}")
    print(f"Test set: {args.test_set}")
    print(f"{'='*60}\n")
    
    # Cargar datos
    test_dataset = TrajectoryDataset(args.test_set)
    test_loader = DataLoader(
        test_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=4,
        pin_memory=True
    )
    
    # Crear modelo
    model = PredictorNoEgo(
        obs_len=args.obs_len,
        future_steps=args.future_steps,
        num_modes=args.num_modes
    ).to(device)
    
    # Cargar checkpoint
    print(f"Loading checkpoint from: {args.checkpoint}")
    checkpoint = torch.load(args.checkpoint, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    
    if 'epoch' in checkpoint:
        print(f"  - Trained for {checkpoint['epoch']} epochs")
    if 'val_ade' in checkpoint:
        print(f"  - Best validation ADE: {checkpoint['val_ade']:.4f}")
    if 'val_fde' in checkpoint:
        print(f"  - Best validation FDE: {checkpoint['val_fde']:.4f}")
    
    print(f"\nModel parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    # Crear directorio de resultados
    save_dir = os.path.join(args.save_dir, Path(args.checkpoint).parent.name)
    
    # Test
    results = test_model(model, test_loader, device, save_dir)
    
    print(f"\n{'='*60}")
    print("Testing completed successfully!")
    print(f"{'='*60}\n")


if __name__ == '__main__':
    main()
