import torch
import sys
import csv
import time
import argparse
import logging
import os
import numpy as np
import subprocess
from torch import nn, optim
from utils.train_utils import *
from model.planner import MotionPlanner
from model.predictor import Predictor, NUM_MODES
from torch.utils.data import DataLoader


def train_epoch(data_loader, predictor, planner, optimizer, use_planning):
    epoch_loss = []
    epoch_metrics = []
    current = 0
    size = len(data_loader.dataset)
    predictor.train()
    start_time = time.time()

    for batch_idx, batch in enumerate(data_loader):
        # prepare data
        ego = batch[0].to(args.device)
        neighbors = batch[1].to(args.device)
        ground_truth = batch[2].to(args.device)
        
        # ========================================================
        if batch_idx == 0:
            print(f"\n=== Tensor Shapes en GPU ===")
            print(f"ego.shape: {ego.shape}")
            print(f"neighbors.shape: {neighbors.shape}")
            print(f"ground_truth.shape: {ground_truth.shape}")
            print(f"ego device: {ego.device}")
        # ========================================================
        
        current_state = torch.cat([ego.unsqueeze(1), neighbors[..., :-1]], dim=1)[:, :, -1]
        weights = torch.ne(ground_truth[:, 1:, :, :2], 0)

        # predict
        optimizer.zero_grad()
        plans, predictions, scores, cost_function_weights = predictor(ego, neighbors)
        if batch_idx == 0:
            weights_to_print = cost_function_weights[0].detach().cpu().numpy()
            np.set_printoptions(precision=4, suppress=True)
            
            print(f"\n{'='*50}")
            print(f"INSPECCIÓN DE PESOS PARA VER SI SE ESTAN APRENDIENDO (Inicio de la Época)")
            print(f"Cost Weights: {weights_to_print}")
            print(f"{'='*50}\n")

        plan_trajs = torch.stack([bicycle_model(plans[:, i], ego[:, -1])[:, :, :3] for i in range(NUM_MODES)], dim=1)
        loss = MFMA_loss(plan_trajs, predictions, scores, ground_truth, weights)
        
        # plan
        if use_planning:
            plan, prediction = select_future(plans, predictions, scores)
            
            # Ground truth trajectory for trajectory following cost - shape: [batch, 12, 3]
            gt_trajectory = ground_truth[:, 0, :, :3]  # ego's ground truth future

            planner_inputs = {
                "control_variables": plan.view(ego.shape[0], 24),
                "predictions": prediction,
                "current_state": current_state,
                "gt_trajectory": gt_trajectory
            }

            for i in range(cost_function_weights.shape[1]):
                planner_inputs[f'cost_function_weight_{i+1}'] = cost_function_weights[:, i].unsqueeze(1)

            final_values, info = planner.layer.forward(planner_inputs)
            plan = final_values["control_variables"].view(-1, 12, 2)
            plan = bicycle_model(plan, ego[:, -1])[:, :, :3]

            plan_cost = info.last_err.mean() if hasattr(info, 'last_err') else 0
            plan_loss = F.smooth_l1_loss(plan, ground_truth[:, 0, :, :3]) 
            plan_loss += F.smooth_l1_loss(plan[:, -1], ground_truth[:, 0, -1, :3])
            
            loss += plan_loss + 1e-3 * plan_cost
        else:
            plan, prediction = select_future(plan_trajs, predictions, scores)

        # loss backward
        loss.backward()
        nn.utils.clip_grad_norm_(predictor.parameters(), 1.0)
        optimizer.step()

        # compute metrics
        metrics = motion_metrics(plan, prediction, ground_truth, weights)
        epoch_metrics.append(metrics)
        epoch_loss.append(loss.item())

        # show loss
        current += batch[0].shape[0]
        sys.stdout.write(f"\rTrain Progress: [{current:>6d}/{size:>6d}]  Loss: {np.mean(epoch_loss):>.4f}  {(time.time()-start_time)/current:>.4f}s/sample")
        sys.stdout.flush()

    # show metrics
    epoch_metrics = np.array(epoch_metrics)
    plannerADE, plannerFDE = np.mean(epoch_metrics[:, 0]), np.mean(epoch_metrics[:, 1])
    predictorADE, predictorFDE = np.mean(epoch_metrics[:, 2]), np.mean(epoch_metrics[:, 3])
    epoch_metrics = [plannerADE, plannerFDE, predictorADE, predictorFDE]
    logging.info(f'\nplannerADE: {plannerADE:.4f}, plannerFDE: {plannerFDE:.4f}, predictorADE: {predictorADE:.4f}, predictorFDE: {predictorFDE:.4f}')
  
    return np.mean(epoch_loss), epoch_metrics

def valid_epoch(data_loader, predictor, planner, use_planning):
    epoch_loss = []
    epoch_metrics = []
    current = 0
    size = len(data_loader.dataset)
    predictor.eval()
    start_time = time.time()

    for batch in data_loader:
        # prepare data
        ego = batch[0].to(args.device)
        neighbors = batch[1].to(args.device)
        ground_truth = batch[2].to(args.device)
        current_state = torch.cat([ego.unsqueeze(1), neighbors[..., :-1]], dim=1)[:, :, -1]
        weights = torch.ne(ground_truth[:, 1:, :, :2], 0)

        # predict
        with torch.no_grad():
            # plans, predictions, scores, cost_function_weights = predictor(ego, neighbors, map_lanes, map_crosswalks)
            plans, predictions, scores, cost_function_weights = predictor(ego, neighbors)
            plan_trajs = torch.stack([bicycle_model(plans[:, i], ego[:, -1])[:, :, :3] for i in range(NUM_MODES)], dim=1)
            loss = MFMA_loss(plan_trajs, predictions, scores, ground_truth, weights) # multi-future multi-agent loss
            
        # plan 
        if use_planning:
            plan, prediction = select_future(plans, predictions, scores)
            
            # Ground truth trajectory for trajectory following cost - shape: [batch, 12, 3]
            gt_trajectory = ground_truth[:, 0, :, :3]  # ego's ground truth future

            planner_inputs = {
                "control_variables": plan.view(ego.shape[0], 24),
                "predictions": prediction,
                "current_state": current_state,
                "gt_trajectory": gt_trajectory
            }

            for i in range(cost_function_weights.shape[1]):
                planner_inputs[f'cost_function_weight_{i+1}'] = cost_function_weights[:, i].unsqueeze(1)

            with torch.no_grad():
                final_values, info = planner.layer.forward(planner_inputs)

            plan = final_values["control_variables"].view(-1, 12, 2)
            plan = bicycle_model(plan, ego[:, -1])[:, :, :3]

            plan_cost = info.last_err.mean() if hasattr(info, 'last_err') else 0
            plan_loss = F.smooth_l1_loss(plan, ground_truth[:, 0, :, :3])
            plan_loss += F.smooth_l1_loss(plan[:, -1], ground_truth[:, 0, -1, :3])
            loss += plan_loss + 1e-3 * plan_cost # planning loss
        else:
            plan, prediction = select_future(plan_trajs, predictions, scores)

        # compute metrics
        metrics = motion_metrics(plan, prediction, ground_truth, weights)
        epoch_metrics.append(metrics)
        epoch_loss.append(loss.item())

        # show progress
        current += batch[0].shape[0]
        sys.stdout.write(f"\rValid Progress: [{current:>6d}/{size:>6d}]  Loss: {np.mean(epoch_loss):>.4f}  {(time.time()-start_time)/current:>.4f}s/sample")
        sys.stdout.flush()
 
    epoch_metrics = np.array(epoch_metrics)
    plannerADE, plannerFDE = np.mean(epoch_metrics[:, 0]), np.mean(epoch_metrics[:, 1])
    predictorADE, predictorFDE = np.mean(epoch_metrics[:, 2]), np.mean(epoch_metrics[:, 3])
    epoch_metrics = [plannerADE, plannerFDE, predictorADE, predictorFDE]
    logging.info(f'\nval-plannerADE: {plannerADE:.4f}, val-plannerFDE: {plannerFDE:.4f}, val-predictorADE: {predictorADE:.4f}, val-predictorFDE: {predictorFDE:.4f}')

    return np.mean(epoch_loss), epoch_metrics

def model_training():
    # Logging
    log_path = f"./training_log/{args.name}/"
    os.makedirs(log_path, exist_ok=True)
    initLogging(log_file=log_path+'train.log')

    logging.info("------------- {} -------------".format(args.name))
    logging.info("Batch size: {}".format(args.batch_size))
    logging.info("Learning rate: {}".format(args.learning_rate))
    logging.info("Use integrated planning module: {}".format(args.use_planning))
    logging.info("Use device: {}".format(args.device))

    # set seed
    set_seed(args.seed)

    # set up predictor
    predictor = Predictor(12).to(args.device)
    
    # set up planner
    if args.use_planning:
        trajectory_len, feature_len = 12, 9
        planner = MotionPlanner(trajectory_len, feature_len, args.device)
    else:
        planner = None
    
    # set up optimizer
    optimizer = optim.Adam(predictor.parameters(), lr=args.learning_rate)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=1)

    # training parameters
    train_epochs = args.train_epochs
    batch_size = args.batch_size
    
    # set up data loaders
    # Check if train_set is a direct path to .npz file or a directory
    if args.train_set.endswith('.npz'):
        # New format: Direct path to consolidated .npz file
        train_set = DrivingData(args.train_set)
        valid_set = DrivingData(args.valid_set)
    else:
        # Old format: Directory with multiple files
        train_set = DrivingData(args.train_set+'/*')
        valid_set = DrivingData(args.valid_set+'/*')
    
    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True, num_workers=args.num_workers)
    valid_loader = DataLoader(valid_set, batch_size=batch_size, shuffle=False, num_workers=args.num_workers)
    logging.info("Dataset Prepared: {} train data, {} validation data\n".format(len(train_set), len(valid_set)))
    
    # begin training
    for epoch in range(train_epochs):
        logging.info(f"Epoch {epoch+1}/{train_epochs}")
        
        # train 
        if planner:
            if epoch < args.pretrain_epochs:
                args.use_planning = False
            else:
                args.use_planning = True         

        train_loss, train_metrics = train_epoch(train_loader, predictor, planner, optimizer, args.use_planning)
        val_loss, val_metrics = valid_epoch(valid_loader, predictor, planner, args.use_planning)

        # save to training log
        log = {'epoch': epoch+1, 'loss': train_loss, 'lr': optimizer.param_groups[0]['lr'], 'val-loss': val_loss, 
               'train-plannerADE': train_metrics[0], 'train-plannerFDE': train_metrics[1], 
               'train-predictorADE': train_metrics[2], 'train-predictorFDE': train_metrics[3],
               'val-plannerADE': val_metrics[0], 'val-plannerFDE': val_metrics[1], 
               'val-predictorADE': val_metrics[2], 'val-predictorFDE': val_metrics[3]}

        if epoch == 0:
            with open(f'./training_log/{args.name}/train_log.csv', 'w') as csv_file: 
                writer = csv.writer(csv_file) 
                writer.writerow(log.keys())
                writer.writerow(log.values())
        else:
            with open(f'./training_log/{args.name}/train_log.csv', 'a') as csv_file: 
                writer = csv.writer(csv_file)
                writer.writerow(log.values())

        # reduce learning rate
        scheduler.step()

        # save model at the end of epoch
        torch.save(predictor.state_dict(), f'training_log/{args.name}/model_{epoch+1}_{val_metrics[0]:.4f}.pth')
        logging.info(f"Model saved in training_log/{args.name}\n")
    
    # Generate graphs automatically
    logging.info("="*60)
    logging.info("Generating training graphs...")
    logging.info("="*60)
    
    try:
        log_csv_path = f'training_log/{args.name}/train_log.csv'
        result = subprocess.run(
            ['python', 'plot_training.py', '--log_path', log_csv_path, '--save_dir', f'training_log/{args.name}'],
            capture_output=True,
            text=True,
            check=True
        )

        logging.info(f"Graphs saved in: training_log/{args.name}/")
        if result.stdout:
            logging.info(result.stdout)
    except subprocess.CalledProcessError as e:
        logging.warning(f"Error al generar gráficas: {e}")
        if e.stderr:
            logging.warning(e.stderr)
    except FileNotFoundError:
        logging.warning("No se encontró plot_training.py")
    
    logging.info("="*60)
    logging.info("ENTRENAMIENTO COMPLETADO")
    logging.info("="*60)

if __name__ == "__main__":
    
    parser = argparse.ArgumentParser(description='Training')
    parser.add_argument('--name', type=str, help='log name (default: "Exp1")', default="Exp1")
    parser.add_argument('--train_set', type=str, help='path to train datasets')
    parser.add_argument('--valid_set', type=str, help='path to validation datasets')
    parser.add_argument('--seed', type=int, help='fix random seed', default=42)
    parser.add_argument("--num_workers", type=int, default=8, help="number of workers used for dataloader")
    parser.add_argument('--pretrain_epochs', type=int, help='epochs of pretraining predictor', default=7)
    parser.add_argument('--train_epochs', type=int, help='epochs of training', default=20)
    parser.add_argument('--batch_size', type=int, help='batch size (default: 32)', default=32)
    parser.add_argument('--learning_rate', type=float, help='learning rate (default: 2e-4)', default=2e-4)
    parser.add_argument('--use_planning', action="store_true", help='if use integrated planning module (default: False)', default=False)
    parser.add_argument('--device', type=str, help='run on which device (default: cuda)', default='cuda')
    args = parser.parse_args()

    
    model_training()
