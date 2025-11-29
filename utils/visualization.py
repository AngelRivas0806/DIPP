"""
Utilidades de visualización para predicción de trayectorias
"""

import matplotlib.pyplot as plt
import numpy as np
import os


def plot_trajectory_prediction(ego_hist, ego_pred, ego_gt, 
                               neighbors_hist=None, neighbors_pred=None, neighbors_gt=None,
                               save_path=None, show=False, sample_id=0):
    """
    Visualizar predicciones de trayectorias
    
    Args:
        ego_hist: (obs_len, 2) - Historia del ego
        ego_pred: (pred_len, 2) - Predicción del ego
        ego_gt: (pred_len, 2) - Ground truth del ego
        neighbors_hist: (num_neighbors, obs_len, 2) - Historia de vecinos
        neighbors_pred: (num_neighbors, pred_len, 2) - Predicciones de vecinos
        neighbors_gt: (num_neighbors, pred_len, 2) - Ground truth de vecinos
        save_path: Directorio donde guardar la imagen
        show: Si mostrar la gráfica
        sample_id: ID de la muestra para el nombre del archivo
    """
    fig, ax = plt.subplots(figsize=(12, 10))
    
    # Colores
    ego_color = '#FF0000'  # Rojo para ego
    neighbor_color = '#0000FF'  # Azul para vecinos
    ego_pred_color = '#FF8C00'  # Naranja para predicción del ego
    neighbor_pred_color = '#00FF00'  # Verde para predicciones de vecinos
    hist_alpha = 0.6
    pred_alpha = 0.8
    
    # ========== EGO ==========
    # Historia del ego
    ax.plot(ego_hist[:, 0], ego_hist[:, 1], 
            color=ego_color, linewidth=3, linestyle='-', alpha=hist_alpha,
            label='Ego History', marker='o', markersize=6)
    
    # Predicción del ego
    ax.plot(ego_pred[:, 0], ego_pred[:, 1], 
            color=ego_pred_color, linewidth=3, linestyle='--', alpha=pred_alpha,
            label='Ego Prediction', marker='s', markersize=6)
    
    # Ground truth del ego
    ax.plot(ego_gt[:, 0], ego_gt[:, 1], 
            color=ego_color, linewidth=2, linestyle=':', alpha=0.9,
            label='Ego Ground Truth', marker='x', markersize=8)
    
    # Conectar historia con predicción
    ax.plot([ego_hist[-1, 0], ego_pred[0, 0]], 
            [ego_hist[-1, 1], ego_pred[0, 1]], 
            color=ego_pred_color, linewidth=1, linestyle='--', alpha=0.5)
    
    # Marcar puntos importantes del ego
    # Punto de inicio de la historia (azul oscuro)
    ax.scatter(ego_hist[0, 0], ego_hist[0, 1], 
              color='darkblue', s=150, marker='o', edgecolors='black', 
              linewidths=2, zorder=10, label='Ego History Start')
    
    # Punto donde empieza la predicción/ground truth (rojo - último de historia)
    ax.scatter(ego_hist[-1, 0], ego_hist[-1, 1], 
              color=ego_color, s=200, marker='o', edgecolors='black', 
              linewidths=2, zorder=10, label='Ego Prediction Start')
    
    # Punto final predicho
    ax.scatter(ego_pred[-1, 0], ego_pred[-1, 1], 
              color=ego_pred_color, s=200, marker='*', edgecolors='black', 
              linewidths=2, zorder=10, label='Ego Predicted End')
    
    # ========== VECINOS ==========
    if neighbors_hist is not None and neighbors_pred is not None and neighbors_gt is not None:
        num_neighbors = neighbors_hist.shape[0]
        
        for i in range(num_neighbors):
            # Verificar si el vecino es válido (tiene datos)
            if np.sum(np.abs(neighbors_gt[i])) == 0:
                continue
            
            # Historia del vecino
            ax.plot(neighbors_hist[i, :, 0], neighbors_hist[i, :, 1], 
                   color=neighbor_color, linewidth=1.5, linestyle='-', alpha=hist_alpha,
                   marker='o', markersize=3)
            
            # Predicción del vecino
            ax.plot(neighbors_pred[i, :, 0], neighbors_pred[i, :, 1], 
                   color=neighbor_pred_color, linewidth=1.5, linestyle='--', alpha=pred_alpha,
                   marker='s', markersize=3)
            
            # Ground truth del vecino
            ax.plot(neighbors_gt[i, :, 0], neighbors_gt[i, :, 1], 
                   color=neighbor_color, linewidth=1, linestyle=':', alpha=0.7,
                   marker='x', markersize=4)
            
            # Conectar historia con predicción
            ax.plot([neighbors_hist[i, -1, 0], neighbors_pred[i, 0, 0]], 
                   [neighbors_hist[i, -1, 1], neighbors_pred[i, 0, 1]], 
                   color=neighbor_pred_color, linewidth=0.5, linestyle='--', alpha=0.3)
        
        # Agregar etiquetas solo una vez para los vecinos
        ax.plot([], [], color=neighbor_color, linewidth=1.5, linestyle='-', 
               alpha=hist_alpha, label='Neighbors History')
        ax.plot([], [], color=neighbor_pred_color, linewidth=1.5, linestyle='--', 
               alpha=pred_alpha, label='Neighbors Prediction')
        ax.plot([], [], color=neighbor_color, linewidth=1, linestyle=':', 
               alpha=0.7, label='Neighbors Ground Truth')
    
    # Configuración de la gráfica
    ax.set_xlabel('X (m)', fontsize=14, fontweight='bold')
    ax.set_ylabel('Y (m)', fontsize=14, fontweight='bold')
    ax.set_title(f'Trajectory Prediction - Sample {sample_id}', 
                fontsize=16, fontweight='bold', pad=20)
    ax.legend(loc='best', fontsize=11, framealpha=0.9)
    ax.grid(True, alpha=0.3, linestyle='--', linewidth=0.5)
    ax.set_aspect('equal', adjustable='box')
    
    # Ajustar límites con margen
    all_x = np.concatenate([ego_hist[:, 0], ego_pred[:, 0], ego_gt[:, 0]])
    all_y = np.concatenate([ego_hist[:, 1], ego_pred[:, 1], ego_gt[:, 1]])
    
    if neighbors_hist is not None:
        for i in range(neighbors_hist.shape[0]):
            if np.sum(np.abs(neighbors_gt[i])) > 0:
                all_x = np.concatenate([all_x, neighbors_hist[i, :, 0], 
                                       neighbors_pred[i, :, 0], neighbors_gt[i, :, 0]])
                all_y = np.concatenate([all_y, neighbors_hist[i, :, 1], 
                                       neighbors_pred[i, :, 1], neighbors_gt[i, :, 1]])
    
    margin = 0.1
    x_range = all_x.max() - all_x.min()
    y_range = all_y.max() - all_y.min()
    ax.set_xlim(all_x.min() - margin * x_range, all_x.max() + margin * x_range)
    ax.set_ylim(all_y.min() - margin * y_range, all_y.max() + margin * y_range)
    
    plt.tight_layout()
    
    # Guardar
    if save_path is not None:
        os.makedirs(save_path, exist_ok=True)
        filepath = os.path.join(save_path, f'prediction_sample_{sample_id:04d}.png')
        plt.savefig(filepath, dpi=150, bbox_inches='tight')
        print(f"Saved: {filepath}")
    
    # Mostrar
    if show:
        plt.show()
    else:
        plt.close()


def plot_multiple_samples(samples_data, save_path, max_samples=6):
    """
    Crear una cuadrícula con múltiples muestras
    
    Args:
        samples_data: Lista de diccionarios con las trayectorias
        save_path: Directorio donde guardar
        max_samples: Número máximo de muestras a plotear
    """
    num_samples = min(len(samples_data), max_samples)
    rows = 2
    cols = 3
    
    fig, axes = plt.subplots(rows, cols, figsize=(18, 12))
    axes = axes.flatten()
    
    for idx, (ax, data) in enumerate(zip(axes[:num_samples], samples_data[:num_samples])):
        # Colores
        ego_color = '#FF0000'
        neighbor_color = '#0000FF'
        ego_pred_color = '#FF8C00'  # Naranja
        neighbor_pred_color = '#00FF00'
        
        # Ego
        ax.plot(data['ego_hist'][:, 0], data['ego_hist'][:, 1], 
               color=ego_color, linewidth=2, alpha=0.6, marker='o', markersize=4)
        ax.plot(data['ego_pred'][:, 0], data['ego_pred'][:, 1], 
               color=ego_pred_color, linewidth=2, linestyle='--', alpha=0.8, marker='s', markersize=4)
        ax.plot(data['ego_gt'][:, 0], data['ego_gt'][:, 1], 
               color=ego_color, linewidth=1.5, linestyle=':', alpha=0.9, marker='x', markersize=5)
        
        # Vecinos
        if 'neighbors_hist' in data:
            for i in range(data['neighbors_hist'].shape[0]):
                if np.sum(np.abs(data['neighbors_gt'][i])) > 0:
                    ax.plot(data['neighbors_hist'][i, :, 0], data['neighbors_hist'][i, :, 1], 
                           color=neighbor_color, linewidth=1, alpha=0.5)
                    ax.plot(data['neighbors_pred'][i, :, 0], data['neighbors_pred'][i, :, 1], 
                           color=neighbor_pred_color, linewidth=1, linestyle='--', alpha=0.6)
        
        ax.set_title(f'Sample {idx}', fontsize=12, fontweight='bold')
        ax.grid(True, alpha=0.3)
        ax.set_aspect('equal', adjustable='box')
        ax.set_xlabel('X (m)', fontsize=10)
        ax.set_ylabel('Y (m)', fontsize=10)
    
    # Ocultar ejes vacíos
    for idx in range(num_samples, len(axes)):
        axes[idx].axis('off')
    
    plt.suptitle('Trajectory Predictions - Multiple Samples', 
                fontsize=16, fontweight='bold', y=0.995)
    plt.tight_layout()
    
    os.makedirs(save_path, exist_ok=True)
    filepath = os.path.join(save_path, 'predictions_grid.png')
    plt.savefig(filepath, dpi=150, bbox_inches='tight')
    print(f"Saved grid: {filepath}")
    plt.close()
