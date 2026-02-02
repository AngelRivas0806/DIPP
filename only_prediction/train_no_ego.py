"""
Script de entrenamiento para PredictorNoEgo.
Entrena un modelo de predicción de trayectorias sin agente ego.
"""

import argparse
import os
import sys
import numpy as np
import torch
from torch import nn, optim
from torch.utils.data import Dataset, DataLoader
import time
from datetime import datetime

# Añadir directorio padre al path para importar
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from predictor_no_ego import PredictorNoEgo
from eval_utils import compute_ade, compute_fde
from data_utils import TrajectoryDataset

if False:
    class TrajectoryDataset(Dataset):
        """Dataset para trayectorias sin ego."""
        
        def __init__(self, data_path):
            """
            Args:
                data_path: ruta al archivo .npz con datos preprocesados
            """
            print(f"Loading dataset from: {data_path}")
            data = np.load(data_path)
            
            self.observed = torch.FloatTensor(data['observed_trajectory'])
            self.future   = torch.FloatTensor(data['gt_future_trajectory'])
            
            print(f"  - Loaded {len(self.observed)} samples")
            print(f"  - Observed shape: {self.observed.shape}")
            print(f"  - Future shape: {self.future.shape}")
        
        def __len__(self):
            return len(self.observed)
        
        def __getitem__(self, idx):
            return self.observed[idx], self.future[idx]

    def compute_ade(pred_traj, gt_traj):
        """
        Average Displacement Error - minADE (best mode).
        Args:
            pred_traj: (batch, num_modes, future_steps, 2)
            gt_traj: (batch, future_steps, 2)
        Returns:
            ade: escalar, promedio de distancias para el mejor modo
        """
        gt_expanded = gt_traj.unsqueeze(1)  # (batch, 1, future_steps, 2)
        errors = torch.norm(pred_traj - gt_expanded, dim=-1)  # (batch, num_modes, future_steps)
        ade_per_mode = errors.mean(dim=-1)  # (batch, num_modes)
        min_ade = ade_per_mode.min(dim=-1)[0]  # (batch,)
        return min_ade.mean(), min_ade

    def compute_ade(pred_traj, gt_traj):
        """
        Average Displacement Error.
        Args:
            pred_traj: (batch, num_modes, future_steps, 2)
            gt_traj: (batch, future_steps, 2)
        Returns:
            ade: escalar, promedio de distancias entre predicción y ground truth
        """
        # Calcular error para cada modo
        gt_expanded = gt_traj.unsqueeze(1)  # (batch, 1, future_steps, 2)
        errors = torch.norm(pred_traj - gt_expanded, dim=-1)  # (batch, num_modes, future_steps)
        
        # Promedio sobre timesteps
        ade_per_mode = errors.mean(dim=-1)  # (batch, num_modes)
        
        # Tomar el mejor modo (menor error)
        min_ade = ade_per_mode.min(dim=-1)[0]  # (batch,)
        
        return min_ade.mean()


    def compute_fde(pred_traj, gt_traj):
        """
        Final Displacement Error.
        Args:
            pred_traj: (batch, num_modes, future_steps, 2)
            gt_traj: (batch, future_steps, 2)
        Returns:
            fde: escalar, error en el último timestep
        """
        # Tomar última posición
        pred_final = pred_traj[:, :, -1, :]  # (batch, num_modes, 2)
        gt_final = gt_traj[:, -1, :]  # (batch, 2)
        
        # Calcular error para cada modo
        gt_expanded = gt_final.unsqueeze(1)  # (batch, 1, 2)
        errors = torch.norm(pred_final - gt_expanded, dim=-1)  # (batch, num_modes)
        
        # Tomar el mejor modo
        min_fde = errors.min(dim=-1)[0]  # (batch,)
        
        return min_fde.mean()


def train_epoch(model, dataloader, optimizer, device, epoch, total_epochs):
    """Entrena una época."""
    model.train()
    
    total_loss = 0.0
    total_ade = 0.0
    total_fde = 0.0
    num_batches = 0
    
    start_time = time.time()
    
    for batch_idx, (observed, future, __) in enumerate(dataloader):
        observed = observed.to(device)
        future = future.to(device)
        
        # Forward pass
        predictions, scores = model(observed)
        
        # Loss: combinación de regresión y clasificación
        future_expanded = future.unsqueeze(1)
        regression_errors = torch.norm(predictions - future_expanded, dim=-1)
        mode_errors = regression_errors.mean(dim=-1)
        best_mode_idx = mode_errors.argmin(dim=-1)
        
        classification_loss = nn.CrossEntropyLoss()(scores, best_mode_idx)
        
        batch_size = predictions.shape[0]
        best_mode_predictions = predictions[torch.arange(batch_size), best_mode_idx]
        regression_loss = nn.MSELoss()(best_mode_predictions, future)
        loss = regression_loss + 0.1 * classification_loss
        
        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
        optimizer.step()
        
        # Métricas
        with torch.no_grad():
            ade = compute_ade(predictions, future)[0]
            fde = compute_fde(predictions, future)[0]
        
        total_loss += loss.item()
        total_ade += ade.item()
        total_fde += fde.item()
        num_batches += 1
        
        # Log progress
        if (batch_idx + 1) % 50 == 0:
            elapsed = time.time() - start_time
            samples_per_sec = (batch_idx + 1) * dataloader.batch_size / elapsed
            print(f"  [{batch_idx+1:4d}/{len(dataloader):4d}] "
                  f"Loss: {loss.item():.4f} | ADE: {ade.item():.4f} | "
                  f"FDE: {fde.item():.4f} | {samples_per_sec:.1f} samples/s")
    
    return total_loss / num_batches, total_ade / num_batches, total_fde / num_batches


def validate(model, dataloader, device):
    """Valida el modelo."""
    model.eval()
    
    total_loss = 0.0
    total_ade = 0.0
    total_fde = 0.0
    num_batches = 0
    
    with torch.no_grad():
        for observed, future, __ in dataloader:
            observed = observed.to(device)
            future = future.to(device)
            
            predictions, scores = model(observed)
            
            future_expanded = future.unsqueeze(1)
            regression_errors = torch.norm(predictions - future_expanded, dim=-1)
            mode_errors = regression_errors.mean(dim=-1)
            best_mode_idx = mode_errors.argmin(dim=-1)
            
            classification_loss = nn.CrossEntropyLoss()(scores, best_mode_idx)
            
            batch_size = predictions.shape[0]
            best_mode_predictions = predictions[torch.arange(batch_size), best_mode_idx]
            regression_loss = nn.MSELoss()(best_mode_predictions, future)
            loss = regression_loss + 0.1 * classification_loss
            
            ade = compute_ade(predictions, future)[0]
            fde = compute_fde(predictions, future)[0]
            
            total_loss += loss.item()
            total_ade += ade.item()
            total_fde += fde.item()
            num_batches += 1
    
    return total_loss / num_batches, total_ade / num_batches, total_fde / num_batches


def main():
    parser = argparse.ArgumentParser(description='Train PredictorNoEgo')
    
    # Datos
    parser.add_argument('--train_set', type=str, required=True)
    parser.add_argument('--valid_set', type=str, required=True)
    
    # Arquitectura
    parser.add_argument('--obs_len', type=int, default=8)
    parser.add_argument('--future_steps', type=int, default=12)
    parser.add_argument('--num_modes', type=int, default=20)
    
    # Entrenamiento
    parser.add_argument('--epochs', type=int, default=10)
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--learning_rate', type=float, default=2e-4)
    parser.add_argument('--device', type=str, default='cuda', choices=['cuda', 'cpu'])
    
    # Output
    parser.add_argument('--name', type=str, default='predictor_no_ego')
    parser.add_argument('--save_dir', type=str, default='only_prediction/checkpoints')
    
    args = parser.parse_args()
    
    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    
    print(f"\n{'='*60}")
    print(f"Training PredictorNoEgo - {args.name}")
    print(f"{'='*60}")
    print(f"Device: {device}")
    print(f"Batch size: {args.batch_size}")
    print(f"Learning rate: {args.learning_rate}")
    print(f"Epochs: {args.epochs}")
    print(f"Num modes: {args.num_modes}")
    print(f"{'='*60}\n")
    
    save_path = os.path.join(args.save_dir, args.name)
    os.makedirs(save_path, exist_ok=True)
    
    train_dataset = TrajectoryDataset(args.train_set)
    valid_dataset = TrajectoryDataset(args.valid_set)
    
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, 
                             shuffle=True, num_workers=4, pin_memory=True)
    valid_loader = DataLoader(valid_dataset, batch_size=args.batch_size,
                             shuffle=False, num_workers=4, pin_memory=True)
    
    print(f"Dataset: {len(train_dataset)} train, {len(valid_dataset)} val\n")
    
    model = PredictorNoEgo(obs_len=args.obs_len, future_steps=args.future_steps,
                          num_modes=args.num_modes).to(device)
    
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}\n")
    
    optimizer = optim.Adam(model.parameters(), lr=args.learning_rate)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.5)
    
    best_val_ade = float('inf')
    
    # Historia de entrenamiento
    history = {
        'train_loss': [],
        'train_ade': [],
        'train_fde': [],
        'val_loss': [],
        'val_ade': [],
        'val_fde': [],
        'epochs': []
    }
    
    for epoch in range(1, args.epochs + 1):
        print(f"\nEpoch {epoch}/{args.epochs}")
        print("-" * 60)
        
        train_loss, train_ade, train_fde = train_epoch(
            model, train_loader, optimizer, device, epoch, args.epochs)
        
        print(f"\nTrain - Loss: {train_loss:.4f} | ADE: {train_ade:.4f} | FDE: {train_fde:.4f}")
        
        val_loss, val_ade, val_fde = validate(model, valid_loader, device)
        print(f"Valid - Loss: {val_loss:.4f} | ADE: {val_ade:.4f} | FDE: {val_fde:.4f}")
        
        # Guardar en historia
        history['train_loss'].append(train_loss)
        history['train_ade'].append(train_ade)
        history['train_fde'].append(train_fde)
        history['val_loss'].append(val_loss)
        history['val_ade'].append(val_ade)
        history['val_fde'].append(val_fde)
        history['epochs'].append(epoch)
        
        scheduler.step()
        
        if val_ade < best_val_ade:
            best_val_ade = val_ade
            checkpoint_path = os.path.join(save_path, 'best_model.pth')
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_ade': val_ade,
                'val_fde': val_fde,
                'args': vars(args)
            }, checkpoint_path)
            print(f"✓ Best model saved (ADE: {val_ade:.4f})")
        
        last_checkpoint_path = os.path.join(save_path, 'last_model.pth')
        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'val_ade': val_ade,
            'val_fde': val_fde,
            'args': vars(args)
        }, last_checkpoint_path)
    
    # Guardar historia de entrenamiento
    history_path = os.path.join(save_path, 'training_history.npz')
    np.savez(history_path,
             train_loss=np.array(history['train_loss']),
             train_ade=np.array(history['train_ade']),
             train_fde=np.array(history['train_fde']),
             val_loss=np.array(history['val_loss']),
             val_ade=np.array(history['val_ade']),
             val_fde=np.array(history['val_fde']),
             epochs=np.array(history['epochs']))
    
    print(f"\n{'='*60}")
    print(f"Training completed! Best val ADE: {best_val_ade:.4f}")
    print(f"Models saved in: {save_path}")
    print(f"Training history saved to: {history_path}")
    print(f"{'='*60}\n")


if __name__ == '__main__':
    main()
