"""
Visualizador de escenas para modelo sin ego.
Genera visualizaciones simplificadas de múltiples peatones.
"""

import numpy as np
import matplotlib.pyplot as plt
import random
import os


def visualize_scene_predictions(observed, future, predictions, scores, 
                               sample_indices, save_path, num_samples=6):
    """
    Visualiza una escena con múltiples peatones.
    
    Args:
        observed: Array numpy (N, obs_len, 2)
        future: Array numpy (N, pred_len, 2)
        predictions: Array numpy (N, num_modes, pred_len, 2)
        scores: Array numpy (N, num_modes)
        sample_indices: Lista de índices de muestras a visualizar
        save_path: Ruta donde guardar la figura
        num_samples: Número de peatones a mostrar
    """
    fig, ax = plt.subplots(1, 1, figsize=(12, 10))
    
    for idx, sample_idx in enumerate(sample_indices[:num_samples]):
        obs = observed[sample_idx]  # (obs_len, 2)
        gt = future[sample_idx]  # (pred_len, 2)
        pred = predictions[sample_idx]  # (num_modes, pred_len, 2)
        score = scores[sample_idx]  # (num_modes,)
        
        # Encontrar el mejor modo (mayor score)
        best_mode_idx = np.argmax(score)
        best_pred = pred[best_mode_idx]
        
        # Plot trayectoria observada en azul con puntitos
        ax.plot(obs[:, 0], obs[:, 1], 
                'b-', linewidth=2.5,
                label='Observed' if idx == 0 else None,
                zorder=3, alpha=0.8)
        ax.plot(obs[:, 0], obs[:, 1], 
                'bo', markersize=3,
                zorder=3, alpha=0.8)
        
        # Plot ground truth en verde con puntitos
        ax.plot(gt[:, 0], gt[:, 1], 
                'g-', linewidth=2.5,
                label='Ground Truth' if idx == 0 else None,
                zorder=3, alpha=0.8)
        ax.plot(gt[:, 0], gt[:, 1], 
                'go', markersize=3,
                zorder=3, alpha=0.8)
        
        # Plot mejor predicción en rojo con puntitos
        ax.plot(best_pred[:, 0], best_pred[:, 1], 
                'r-', linewidth=2.5,
                label='Best Prediction' if idx == 0 else None,
                zorder=4, alpha=0.8)
        ax.plot(best_pred[:, 0], best_pred[:, 1], 
                'ro', markersize=3,
                zorder=4, alpha=0.8)
        
        # Plot solo 2 modos alternativos en amarillo
        alternative_modes_shown = 0
        for mode_idx in range(len(pred)):
            if mode_idx != best_mode_idx and alternative_modes_shown < 2:
                mode_traj = pred[mode_idx]
                ax.plot(mode_traj[:, 0], mode_traj[:, 1], 
                       'y-', linewidth=1.0,
                       alpha=0.4, zorder=2,
                       label='Alternative Modes' if idx == 0 and alternative_modes_shown == 0 else None)
                alternative_modes_shown += 1
        
        # Agregar círculo negro en la posición actual (última posición observada)
        current_pos = obs[-1]  # Última posición observada
        ax.plot(current_pos[0], current_pos[1], 
               'ko', markersize=8, zorder=10,
               markerfacecolor='black', markeredgecolor='black')
    
    # Configurar ejes
    ax.set_xlabel('X (m)', fontsize=12, fontweight='bold')
    ax.set_ylabel('Y (m)', fontsize=12, fontweight='bold')
    ax.set_title(f'Scene with {num_samples} Pedestrians', 
                fontsize=14, fontweight='bold', pad=15)
    ax.grid(True, alpha=0.3, linestyle='--', linewidth=0.5)
    ax.axis('equal')
    
    # Leyenda
    ax.legend(loc='upper right', fontsize=11, 
             framealpha=0.9, edgecolor='black')
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()


def select_diverse_scenes(total_samples, num_scenes, samples_per_scene):
    """
    Selecciona índices de muestras para crear escenas diversas.
    
    Args:
        total_samples: Número total de muestras disponibles
        num_scenes: Número de escenas a crear
        samples_per_scene: Número de peatones por escena
    
    Returns:
        Lista de listas con índices de muestras para cada escena
    """
    scenes = []
    samples_used = set()
    
    # Dividir el dataset en regiones
    region_size = total_samples // num_scenes
    
    for scene_idx in range(num_scenes):
        # Región de donde seleccionar muestras
        region_start = scene_idx * region_size
        region_end = min(region_start + region_size * 2, total_samples)
        
        # Seleccionar muestras aleatorias de esta región
        available = [i for i in range(region_start, region_end) if i not in samples_used]
        
        if len(available) < samples_per_scene:
            # Si no hay suficientes, tomar de todo el dataset
            available = [i for i in range(total_samples) if i not in samples_used]
        
        # Seleccionar muestras para esta escena (más de las necesarias para filtrar)
        num_to_sample = min(samples_per_scene * 3, len(available))
        scene_samples = random.sample(available, num_to_sample)
        scenes.append(scene_samples)
        samples_used.update(scene_samples)
    
    return scenes


def filter_active_pedestrians(observed, future, predictions, scene_samples, samples_per_scene):
    """
    Filtra peatones que tienen movimiento significativo.
    
    Returns:
        Lista de índices de peatones activos (máximo samples_per_scene)
    """
    active_pedestrians = []
    
    for idx in scene_samples:
        # Calcular distancia total recorrida en el futuro
        gt_displacement = np.linalg.norm(future[idx][-1] - future[idx][0])
        
        # Solo incluir peatones que se mueven al menos 0.8 metros
        if gt_displacement > 0.8:
            active_pedestrians.append(idx)
            
        if len(active_pedestrians) >= samples_per_scene:
            break
    
    # Si no hay suficientes activos, agregar algunos estáticos
    if len(active_pedestrians) < samples_per_scene:
        for idx in scene_samples:
            if idx not in active_pedestrians:
                active_pedestrians.append(idx)
                if len(active_pedestrians) >= samples_per_scene:
                    break
    
    return active_pedestrians


def generate_scene_visualizations(observed, future, predictions, scores, 
                                  save_dir, num_scenes=15, samples_per_scene=6):
    """
    Genera múltiples visualizaciones de escenas.
    
    Args:
        observed: Array numpy (N, obs_len, 2)
        future: Array numpy (N, pred_len, 2)
        predictions: Array numpy (N, num_modes, pred_len, 2)
        scores: Array numpy (N, num_modes)
        save_dir: Directorio donde guardar las imágenes
        num_scenes: Número de escenas a generar
        samples_per_scene: Número de peatones por escena
    """
    os.makedirs(save_dir, exist_ok=True)
    
    total_samples = len(observed)
    
    print(f"\n{'='*70}")
    print(f"Generating {num_scenes} scene visualizations...")
    print(f"Data shapes: obs={observed.shape}, gt={future.shape}, pred={predictions.shape}")
    print(f"{'='*70}")
    
    # Seleccionar escenas diversas
    scenes = select_diverse_scenes(total_samples, num_scenes, samples_per_scene)
    
    for scene_idx, scene_samples in enumerate(scenes, 1):
        # Filtrar peatones con movimiento significativo
        active_samples = filter_active_pedestrians(
            observed, future, predictions, scene_samples, samples_per_scene
        )
        
        save_path = os.path.join(save_dir, f'scene_{scene_idx:02d}.png')
        
        visualize_scene_predictions(
            observed, future, predictions, scores,
            active_samples, save_path, len(active_samples)
        )
        
        print(f"  ✓ Scene {scene_idx}/{num_scenes} saved: {save_path}")
    
    print(f"\n{'='*70}")
    print(f"All scene visualizations saved in: {save_dir}")
    print(f"{'='*70}\n")
