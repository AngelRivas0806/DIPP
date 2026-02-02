"""
Script para graficar métricas de entrenamiento
"""

import pandas as pd
import matplotlib.pyplot as plt
import argparse
import os


def plot_training_metrics(log_path, save_dir=None):
    """
    Grafica las métricas de entrenamiento desde el CSV de logs
    
    Args:
        log_path: Path al archivo train_log.csv
        save_dir: Directorio donde guardar las gráficas (opcional)
    """
    # Leer el CSV
    df = pd.read_csv(log_path)
    
    # Crear figura con subplots
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle('Training Metrics', fontsize=16, fontweight='bold')
    
    # 1. Loss (Train vs Validation)
    ax1 = axes[0, 0]
    ax1.plot(df['epoch'], df['loss'], 'b-', linewidth=2, marker='o', label='Train Loss')
    ax1.plot(df['epoch'], df['val-loss'], 'r-', linewidth=2, marker='s', label='Validation Loss')
    ax1.set_xlabel('Epoch', fontsize=12)
    ax1.set_ylabel('Loss', fontsize=12)
    ax1.set_title('Training and Validation Loss', fontsize=14, fontweight='bold')
    ax1.legend(fontsize=10)
    ax1.grid(True, alpha=0.3)
    
    # 2. Planner ADE (Train vs Validation)
    ax2 = axes[0, 1]
    ax2.plot(df['epoch'], df['train-plannerADE'], 'b-', linewidth=2, marker='o', label='Train Planner ADE')
    ax2.plot(df['epoch'], df['val-plannerADE'], 'r-', linewidth=2, marker='s', label='Val Planner ADE')
    ax2.set_xlabel('Epoch', fontsize=12)
    ax2.set_ylabel('ADE (m)', fontsize=12)
    ax2.set_title('Planner Average Displacement Error', fontsize=14, fontweight='bold')
    ax2.legend(fontsize=10)
    ax2.grid(True, alpha=0.3)
    
    # 3. Planner FDE (Train vs Validation)
    ax3 = axes[1, 0]
    ax3.plot(df['epoch'], df['train-plannerFDE'], 'b-', linewidth=2, marker='o', label='Train Planner FDE')
    ax3.plot(df['epoch'], df['val-plannerFDE'], 'r-', linewidth=2, marker='s', label='Val Planner FDE')
    ax3.set_xlabel('Epoch', fontsize=12)
    ax3.set_ylabel('FDE (m)', fontsize=12)
    ax3.set_title('Planner Final Displacement Error', fontsize=14, fontweight='bold')
    ax3.legend(fontsize=10)
    ax3.grid(True, alpha=0.3)
    
    # 4. Learning Rate
    ax4 = axes[1, 1]
    ax4.plot(df['epoch'], df['lr'], 'g-', linewidth=2, marker='^')
    ax4.set_xlabel('Epoch', fontsize=12)
    ax4.set_ylabel('Learning Rate', fontsize=12)
    ax4.set_title('Learning Rate Schedule', fontsize=14, fontweight='bold')
    ax4.set_yscale('log')
    ax4.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    # Guardar o mostrar
    if save_dir:
        os.makedirs(save_dir, exist_ok=True)
        save_path = os.path.join(save_dir, 'training_metrics.png')
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f" Gráfica guardada en: {save_path}")
    else:
        # Solo guardar en raíz si NO se especificó save_dir
        plt.savefig('training_metrics.png', dpi=300, bbox_inches='tight')
        print(f" Gráfica guardada en: training_metrics.png")
    
    # Mostrar estadísticas
    print("\n" + "="*60)
    print("ESTADÍSTICAS DE ENTRENAMIENTO")
    print("="*60)
    print(f"Total de épocas: {len(df)}")
    print(f"\nMejor Validation Planner ADE: {df['val-plannerADE'].min():.4f} m (Época {df.loc[df['val-plannerADE'].idxmin(), 'epoch']:.0f})")
    print(f"Mejor Validation Planner FDE: {df['val-plannerFDE'].min():.4f} m (Época {df.loc[df['val-plannerFDE'].idxmin(), 'epoch']:.0f})")
    print(f"Mejor Validation Loss: {df['val-loss'].min():.4f} (Época {df.loc[df['val-loss'].idxmin(), 'epoch']:.0f})")
    print(f"\nÚltima época:")
    print(f"  Train Loss: {df['loss'].iloc[-1]:.4f}")
    print(f"  Val Loss: {df['val-loss'].iloc[-1]:.4f}")
    print(f"  Planner ADE: {df['val-plannerADE'].iloc[-1]:.4f} m")
    print(f"  Planner FDE: {df['val-plannerFDE'].iloc[-1]:.4f} m")
    print("="*60 + "\n")
    
    # Crear gráfica adicional de Predictor ADE/FDE
    fig2, axes2 = plt.subplots(1, 2, figsize=(14, 5))
    fig2.suptitle('Predictor Metrics', fontsize=16, fontweight='bold')
    
    # Predictor ADE
    axes2[0].plot(df['epoch'], df['train-predictorADE'], 'b-', linewidth=2, marker='o', label='Train Predictor ADE')
    axes2[0].plot(df['epoch'], df['val-predictorADE'], 'r-', linewidth=2, marker='s', label='Val Predictor ADE')
    axes2[0].set_xlabel('Epoch', fontsize=12)
    axes2[0].set_ylabel('ADE (m)', fontsize=12)
    axes2[0].set_title('Predictor Average Displacement Error', fontsize=14, fontweight='bold')
    axes2[0].legend(fontsize=10)
    axes2[0].grid(True, alpha=0.3)
    
    # Predictor FDE
    axes2[1].plot(df['epoch'], df['train-predictorFDE'], 'b-', linewidth=2, marker='o', label='Train Predictor FDE')
    axes2[1].plot(df['epoch'], df['val-predictorFDE'], 'r-', linewidth=2, marker='s', label='Val Predictor FDE')
    axes2[1].set_xlabel('Epoch', fontsize=12)
    axes2[1].set_ylabel('FDE (m)', fontsize=12)
    axes2[1].set_title('Predictor Final Displacement Error', fontsize=14, fontweight='bold')
    axes2[1].legend(fontsize=10)
    axes2[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_dir:
        save_path2 = os.path.join(save_dir, 'predictor_metrics.png')
        plt.savefig(save_path2, dpi=300, bbox_inches='tight')
        print(f" Gráfica guardada en: {save_path2}")
    else:
        # Solo guardar en raíz si NO se especificó save_dir
        plt.savefig('predictor_metrics.png', dpi=300, bbox_inches='tight')
        print(f" Gráfica guardada en: predictor_metrics.png")


def main():
    parser = argparse.ArgumentParser(description='Plot training metrics')
    parser.add_argument('--log_path', type=str, required=True, help='Path to train_log.csv')
    parser.add_argument('--save_dir', type=str, default=None, help='Directory to save plots')
    
    args = parser.parse_args()
    
    if not os.path.exists(args.log_path):
        print(f"❌ Error: No se encontró el archivo {args.log_path}")
        return
    
    plot_training_metrics(args.log_path, args.save_dir)


if __name__ == '__main__':
    main()
