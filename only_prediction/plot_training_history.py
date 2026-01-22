"""
Script para graficar las pérdidas y métricas del entrenamiento.
Genera gráficas de Loss, ADE y FDE para train y validation.
"""

import argparse
import os
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path


def plot_training_history(history_path, save_dir=None):
    """
    Grafica el historial de entrenamiento.
    
    Args:
        history_path: ruta al archivo .npz con el historial
        save_dir: directorio donde guardar las gráficas
    """
    # Cargar historia
    print(f"Loading training history from: {history_path}")
    data = np.load(history_path)
    
    epochs = data['epochs']
    train_loss = data['train_loss']
    train_ade = data['train_ade']
    train_fde = data['train_fde']
    val_loss = data['val_loss']
    val_ade = data['val_ade']
    val_fde = data['val_fde']
    
    print(f"  - Epochs: {len(epochs)}")
    print(f"  - Train Loss: {train_loss[-1]:.4f} (final)")
    print(f"  - Val ADE: {val_ade[-1]:.4f} (final)")
    print(f"  - Best Val ADE: {val_ade.min():.4f} (epoch {epochs[val_ade.argmin()]})")
    
    # Crear figura con 3 subplots
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    
    # 1. Loss
    ax = axes[0]
    ax.plot(epochs, train_loss, 'b-o', linewidth=2, markersize=4, label='Train Loss')
    ax.plot(epochs, val_loss, 'r-s', linewidth=2, markersize=4, label='Val Loss')
    ax.set_xlabel('Epoch', fontsize=12)
    ax.set_ylabel('Loss', fontsize=12)
    ax.set_title('Training and Validation Loss', fontsize=14, fontweight='bold')
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)
    
    # Añadir valores finales
    final_train = train_loss[-1]
    final_val = val_loss[-1]
    ax.text(0.98, 0.98, f'Final Train: {final_train:.4f}\nFinal Val: {final_val:.4f}',
            transform=ax.transAxes, fontsize=10,
            verticalalignment='top', horizontalalignment='right',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    # 2. ADE (Average Displacement Error)
    ax = axes[1]
    ax.plot(epochs, train_ade, 'b-o', linewidth=2, markersize=4, label='Train ADE')
    ax.plot(epochs, val_ade, 'r-s', linewidth=2, markersize=4, label='Val ADE')
    ax.set_xlabel('Epoch', fontsize=12)
    ax.set_ylabel('ADE (meters)', fontsize=12)
    ax.set_title('Average Displacement Error (minADE)', fontsize=14, fontweight='bold')
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)
    
    # Marcar mejor valor
    best_epoch = epochs[val_ade.argmin()]
    best_val_ade = val_ade.min()
    ax.axhline(y=best_val_ade, color='g', linestyle='--', alpha=0.5)
    ax.plot(best_epoch, best_val_ade, 'g*', markersize=15, label=f'Best: {best_val_ade:.4f}')
    ax.legend(fontsize=11)
    
    # 3. FDE (Final Displacement Error)
    ax = axes[2]
    ax.plot(epochs, train_fde, 'b-o', linewidth=2, markersize=4, label='Train FDE')
    ax.plot(epochs, val_fde, 'r-s', linewidth=2, markersize=4, label='Val FDE')
    ax.set_xlabel('Epoch', fontsize=12)
    ax.set_ylabel('FDE (meters)', fontsize=12)
    ax.set_title('Final Displacement Error (minFDE)', fontsize=14, fontweight='bold')
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)
    
    # Marcar mejor valor
    best_epoch_fde = epochs[val_fde.argmin()]
    best_val_fde = val_fde.min()
    ax.axhline(y=best_val_fde, color='g', linestyle='--', alpha=0.5)
    ax.plot(best_epoch_fde, best_val_fde, 'g*', markersize=15, label=f'Best: {best_val_fde:.4f}')
    ax.legend(fontsize=11)
    
    plt.tight_layout()
    
    # Guardar figura
    if save_dir:
        os.makedirs(save_dir, exist_ok=True)
        save_path = os.path.join(save_dir, 'training_curves.png')
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"\n✓ Training curves saved to: {save_path}")
        
        # También en PDF para mejor calidad
        save_path_pdf = os.path.join(save_dir, 'training_curves.pdf')
        plt.savefig(save_path_pdf, bbox_inches='tight')
        print(f"✓ Training curves saved to: {save_path_pdf}")
    
    plt.show()
    
    # Crear segunda figura con comparación detallada
    fig2, axes2 = plt.subplots(2, 2, figsize=(14, 10))
    
    # Loss comparativo
    ax = axes2[0, 0]
    ax.plot(epochs, train_loss, 'b-o', linewidth=2, markersize=4, label='Train')
    ax.plot(epochs, val_loss, 'r-s', linewidth=2, markersize=4, label='Validation')
    ax.set_xlabel('Epoch', fontsize=11)
    ax.set_ylabel('Loss', fontsize=11)
    ax.set_title('Loss Progression', fontsize=13, fontweight='bold')
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    
    # ADE comparativo
    ax = axes2[0, 1]
    ax.plot(epochs, train_ade, 'b-o', linewidth=2, markersize=4, label='Train')
    ax.plot(epochs, val_ade, 'r-s', linewidth=2, markersize=4, label='Validation')
    ax.set_xlabel('Epoch', fontsize=11)
    ax.set_ylabel('ADE (m)', fontsize=11)
    ax.set_title('ADE Progression', fontsize=13, fontweight='bold')
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    
    # FDE comparativo
    ax = axes2[1, 0]
    ax.plot(epochs, train_fde, 'b-o', linewidth=2, markersize=4, label='Train')
    ax.plot(epochs, val_fde, 'r-s', linewidth=2, markersize=4, label='Validation')
    ax.set_xlabel('Epoch', fontsize=11)
    ax.set_ylabel('FDE (m)', fontsize=11)
    ax.set_title('FDE Progression', fontsize=13, fontweight='bold')
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    
    # Tabla de resumen
    ax = axes2[1, 1]
    ax.axis('off')
    
    summary_text = f"""
    Training Summary
    {'='*40}
    
    Total Epochs: {len(epochs)}
    
    Final Results:
      • Train Loss: {train_loss[-1]:.4f}
      • Val Loss:   {val_loss[-1]:.4f}
      
      • Train ADE:  {train_ade[-1]:.4f} m
      • Val ADE:    {val_ade[-1]:.4f} m
      
      • Train FDE:  {train_fde[-1]:.4f} m
      • Val FDE:    {val_fde[-1]:.4f} m
    
    Best Validation Results:
      • Best ADE:   {val_ade.min():.4f} m (epoch {epochs[val_ade.argmin()]})
      • Best FDE:   {val_fde.min():.4f} m (epoch {epochs[val_fde.argmin()]})
      • Best Loss:  {val_loss.min():.4f} (epoch {epochs[val_loss.argmin()]})
    
    Improvement:
      • ADE: {((val_ade[0] - val_ade[-1])/val_ade[0]*100):.1f}% reduction
      • FDE: {((val_fde[0] - val_fde[-1])/val_fde[0]*100):.1f}% reduction
    """
    
    ax.text(0.1, 0.9, summary_text, transform=ax.transAxes, 
            fontsize=11, verticalalignment='top',
            fontfamily='monospace',
            bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.3))
    
    plt.tight_layout()
    
    if save_dir:
        save_path2 = os.path.join(save_dir, 'training_summary.png')
        plt.savefig(save_path2, dpi=150, bbox_inches='tight')
        print(f"✓ Training summary saved to: {save_path2}")
    
    plt.show()
    
    print("\n" + "="*60)
    print("TRAINING ANALYSIS")
    print("="*60)
    print(f"Initial Val ADE: {val_ade[0]:.4f} m")
    print(f"Final Val ADE:   {val_ade[-1]:.4f} m")
    print(f"Best Val ADE:    {val_ade.min():.4f} m (epoch {epochs[val_ade.argmin()]})")
    print(f"Improvement:     {((val_ade[0] - val_ade[-1])/val_ade[0]*100):.1f}%")
    print("="*60)


def main():
    parser = argparse.ArgumentParser(description='Plot training history')
    
    parser.add_argument('--history', type=str, required=True,
                       help='Path to training_history.npz file')
    parser.add_argument('--save_dir', type=str, default=None,
                       help='Directory to save plots (default: same as history file)')
    
    args = parser.parse_args()
    
    # Si no se especifica save_dir, usar el mismo directorio que el archivo de historia
    if args.save_dir is None:
        args.save_dir = str(Path(args.history).parent)
    
    print(f"\n{'='*60}")
    print("Training History Plotter")
    print(f"{'='*60}\n")
    
    plot_training_history(args.history, args.save_dir)
    
    print(f"\n{'='*60}")
    print("Plotting completed!")
    print(f"{'='*60}\n")


if __name__ == '__main__':
    main()
