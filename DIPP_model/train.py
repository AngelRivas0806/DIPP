import torch
import sys
import csv
import time
import argparse
import logging
import os
import numpy as np
import subprocess
import wandb
from torch import log, nn, optim
import torch.nn.functional as F  
from torch.utils.data import DataLoader
from model import predictor
from utils.train_utils import *
from model.planner import MotionPlanner
from model.predictor import Predictor
from model.predictorvae import PredictorVAE

# =========================
"""Train a single epoch"""
# =========================

def train_epoch(data_loader, predictor, planner, optimizer, use_planning):

    epoch_loss = []
    epoch_metrics = []
    current = 0
    size = len(data_loader.dataset)
    start_time = time.time()

    predictor.train()

    for batch_idx, batch in enumerate(data_loader):
        ego          = batch[0].to(args.device)
        neighbors    = batch[1].to(args.device)
        ground_truth = batch[2].to(args.device)
        map_segments = None
        map_mask     = None
        if len(batch) >= 5:
            map_segments = batch[3].to(args.device)  # (B,M,4)
            map_mask     = batch[4].to(args.device)  # (B,M)

        # if map_mask is not None:
        #     valid_counts = map_mask.bool().sum(dim=1)

        #     if batch_idx < 5:
        #         print("map valid counts:", valid_counts.detach().cpu().numpy())
        #         print("min valid:", int(valid_counts.min().item()))
        #         print("num empty:", int((valid_counts == 0).sum().item()))

        # current_state: (B, 11, feat) with the last observed state of ego and neighbors
        current_state = torch.cat([ego.unsqueeze(1), neighbors[..., :-1]], dim=1)[:, :, -1]

        # weights: máscara para vecinos presentes (no-0)
        weights = torch.ne(ground_truth[:, 1:, :, :2], 0)

        # ========================================================
        if batch_idx == 0:
            print(f"\n=== Tensor Shapes in GPU ===")
            print(f"ego.shape: {ego.shape}")
            print(f"neighbors.shape: {neighbors.shape}")
            print(f"ground_truth.shape: {ground_truth.shape}")
            print(f"ego device: {ego.device}")
        # ========================================================

        # --- DEBUG: imprime segmentos de la primera muestra del batch ---
        # if batch_idx == 0 and map_segments is not None:

        #     for i in range(10):
        #         mask0 = map_mask[i].bool()  # (M,)
        #         segs0 = map_segments[i][mask0]  # (n_valid,4)

        #         print("\n=== FIRST SAMPLE MAP SEGMENTS (ego-frame, clipped) ===")
        #         print("num_valid:", int(mask0.sum().item()))
        #         if segs0.numel() == 0:
        #             print("No valid segments in sample 0.")
        #         else:
        #             # imprime hasta 8 segmentos para no saturar
        #             n_show = min(8, segs0.shape[0])
        #             print(segs0[:n_show].detach().cpu().numpy())

        #             # extra: longitudes de esos segmentos
        #             a = segs0[:n_show, 0:2]
        #             b = segs0[:n_show, 2:4]
        #             lens = torch.norm(b - a, dim=1)
        #             print("lengths:", lens.detach().cpu().numpy())
        #             print("map_segments shape:", map_segments.shape)   # (B, M, 4)
        #             print("map_mask sum sample0:", int(map_mask[0].bool().sum().item()))  # n_valid
        #             print("map_max_segments (M):", map_segments.shape[1])
        # predict
        optimizer.zero_grad()

        # ======================================================================
        """ ACOORDING TO THE PREDICTOR YOU CHOSE (PREDICTORVAE OR PREDICTOR)"""
        # ======================================================================

        # ================
        """PredictorVAE"""
        # ================

        if args.predictor == "PredictorVAE":
            
            # ===================
            """if use_map"""
            # ===================
            if map_segments is None:
                plans, predictions, cost_function_weights, mu, logvar, priori_mu, priori_logvar = predictor(
                    ego, neighbors, ground_truth)
            else:
                plans, predictions, cost_function_weights, mu, logvar, priori_mu, priori_logvar = predictor(
                    ego, neighbors, ground_truth, map_segments, map_mask)

            plan_trajs = torch.stack([bicycle_model(plans[:, i], ego[:, -1])[:, :, :2] for i in range(1)],dim=1)  # (B,1,T,2)

            # ================================
            """learning prior - use_prior """
            # ================================
            if args.use_prior:
                _, pred_loss, kl_loss = VAE_loss(plan_trajs, predictions, ground_truth, weights, mu, logvar,
                    ego_in_pred_loss=not use_planning, beta=1.0,
                    prior_mu=priori_mu, prior_logvar=priori_logvar)
            else:
                _, pred_loss, kl_loss = VAE_loss(plan_trajs, predictions, ground_truth, weights, mu, logvar,
                    ego_in_pred_loss=not use_planning, beta=1.0)

            base_loss = (args.lambda_pred * pred_loss + args.lambda_score * kl_loss)

            # ======================================================================
            if batch_idx == 0:
                weights_to_print = cost_function_weights[0].detach().cpu().numpy()
                np.set_printoptions(precision=4, suppress=True)
                print(f"\n{'='*50}")
                print(f"Inicio de la Época")
                print(f"Cost Weights: {weights_to_print}")
                print(f"{'='*50}\n")
            # =======================================================================



            # ====================
            """With Planning"""
            # ====================

            if use_planning:

                plan = plans[:, 0]  # (B, T, 2)     # (B, T, 9) con controles del modo 0, que es el único modo en PredictorVAE
                prediction = predictions[:, 0] 

                w = torch.nan_to_num(cost_function_weights, nan=1.0, posinf=20.0, neginf=1e-2)
                w = torch.clamp(w, min=1e-2, max=20.0)

                gt_ego = ground_truth[:, 0, :, :2]
                planner_inputs = {
                    "control_variables": plan.view(ego.shape[0], 24),
                    "predictions": prediction,
                    "current_state": current_state,
                    "gt_trajectory": gt_ego,

                    "w_acc":          w[:, 0].unsqueeze(1),
                    "w_jerk":         w[:, 1].unsqueeze(1),
                    "w_steer":        w[:, 2].unsqueeze(1),
                    "w_steer_change": w[:, 3].unsqueeze(1),
                    "w_collision":    w[:, 4].unsqueeze(1),
                    "w_speed_limit":  w[:, 5].unsqueeze(1),
                    "w_endpoint_goal": w[:, 6].unsqueeze(1),}

                final_values, info = planner.layer.forward(planner_inputs)
                u_opt = final_values["control_variables"].view(-1, 12, 2)
                ego_traj = bicycle_model(u_opt, ego[:, -1], dt=0.4)

                plan = ego_traj[:, :, :3]   # x, y, theta
                speed = ego_traj[:, :, 3]   # v

                # Calcular cada residuo en PyTorch para que el gradiente llegue a cost_weights
                # dt = 0.4
                # w_acc          = w[:, 0].mean()
                # w_jerk         = w[:, 1].mean()
                # w_steer        = w[:, 2].mean()
                # w_steer_change = w[:, 3].mean()
                # w_col          = w[:, 4].mean()
                # w_speed        = w[:, 5].mean()

                # acc    = u_opt[:, :, 0]                            # (B, T)
                # jerk   = torch.diff(acc,   dim=1) / dt            # (B, T-1)
                # steer  = u_opt[:, :, 1]                           # (B, T)
                # ds     = torch.diff(steer, dim=1) / dt            # (B, T-1)

                
                # speed_violation = torch.clamp(speed - args.max_speed, min=0.0)
                # speed_reg = speed_violation.pow(2).mean()

                # imitacion sobre plan optimizado — suma
                imitation_loss = F.smooth_l1_loss(plan[:, :, :2], ground_truth[:, 0, :, :2], reduction='sum')

                plan_cost = info.last_err.mean() if hasattr(info, "last_err") else 0

                loss = (base_loss + args.lambda_imitation * imitation_loss + args.lambda_cost * plan_cost)

                # ====================
                """Without Planning"""
                # ====================

            else:
                # Use the most likely prediction for metrics (no planner)
                prediction = predictions[:, 0]  # (B, N, T, 2)
                plan = plan_trajs[:, 0]        # (B, T, 3)

                # Imitación sobre plan directo del VAE
                imitation_loss = F.smooth_l1_loss(plan[:, :, :2], ground_truth[:, 0, :, :2],  reduction='sum')

                loss = (base_loss + args.lambda_imitation * imitation_loss)

            # ==================
            """Predictor DIPP"""
            # ==================
        else:
        # ===================
            """if use_map"""
        # ===================
            if map_segments is None:
                plans, predictions, scores, cost_function_weights = predictor(ego, neighbors)
            else:
                plans, predictions, scores, cost_function_weights = predictor(ego, neighbors, map_segments, map_mask)


            plan_trajs = torch.stack([bicycle_model(plans[:, i], ego[:, -1])[:, :, :2] for i in range(NUM_MODES)], dim=1)

            # ego_in_pred_loss=False cuando use_planning: el ego se supervisa via imitation_loss sobre plan optimizado
            _, pred_loss, score_loss, best_mode = MFMA_loss(plan_trajs, predictions, scores, ground_truth, weights,
                                                            ego_in_pred_loss=not use_planning)
            
            # imitacion: ego del modo ganador vs gt — suma
            best_plan = plan_trajs[torch.arange(plan_trajs.shape[0], device=plan_trajs.device), best_mode]
            imitation_loss = F.smooth_l1_loss(best_plan[:, :, :2], ground_truth[:, 0, :, :2], reduction='sum')

            # sin planning: loss base, cost_weights NO se usan aqui
            base_loss = (args.lambda_pred * pred_loss + args.lambda_score * score_loss)
            # loss = args.lambda_pred * pred_loss + args.lambda_score * score_loss + args.lambda_imitation * imitation_loss

            # ======================================================================
            if batch_idx == 0:
                weights_to_print = cost_function_weights[0].detach().cpu().numpy()
                np.set_printoptions(precision=4, suppress=True)
                print(f"\n{'='*50}")
                print(f"Inicio de la Época")
                print(f"Cost Weights: {weights_to_print}")
                print(f"{'='*50}\n")
            # =======================================================================

            # ==================
            """With Planning"""
            # ==================
            if use_planning:

                plan, prediction = select_future(plans, predictions, scores, best_mode)

                w = torch.nan_to_num(
                cost_function_weights,
                nan=1.0,
                posinf=20.0,
                neginf=1e-2,
                )

                w = torch.clamp(w, min=1e-2, max=20.0)

                gt_ego = ground_truth[:, 0, :, :2]
                planner_inputs = {
                    "control_variables": plan.view(ego.shape[0], 24),
                    "predictions": prediction,
                    "current_state": current_state,
                    "gt_trajectory": gt_ego,

                    "w_acc":          w[:, 0].unsqueeze(1),
                    "w_jerk":         w[:, 1].unsqueeze(1),
                    "w_steer":        w[:, 2].unsqueeze(1),
                    "w_steer_change": w[:, 3].unsqueeze(1),
                    "w_collision":    w[:, 4].unsqueeze(1),
                    "w_speed_limit":  w[:, 5].unsqueeze(1),
                    "w_endpoint_goal": w[:, 6].unsqueeze(1),}

                final_values, info = planner.layer.forward(planner_inputs)
                u_opt = final_values["control_variables"].view(-1, 12, 2)
                ego_traj = bicycle_model(u_opt, ego[:, -1], dt=0.4)

                plan = ego_traj[:, :, :3]   # x, y, theta
                speed = ego_traj[:, :, 3]   # v

                # Calcular cada residuo en PyTorch para que el gradiente llegue a cost_weights
                # dt = 0.4
                # w_acc          = w[:, 0].mean()
                # w_jerk         = w[:, 1].mean()
                # w_steer        = w[:, 2].mean()
                # w_steer_change = w[:, 3].mean()
                # w_col          = w[:, 4].mean()
                # w_speed        = w[:, 5].mean()

                # acc    = u_opt[:, :, 0]                            # (B, T)
                # jerk   = torch.diff(acc,   dim=1) / dt             # (B, T-1)
                # steer  = u_opt[:, :, 1]                            # (B, T)
                # ds     = torch.diff(steer, dim=1) / dt             # (B, T-1)

                # speed_violation = torch.clamp(speed - args.max_speed, min=0.0)
                # speed_reg = speed_violation.pow(2).mean()

                # cost_reg = (
                #     w_acc * acc.pow(2).mean()
                #     + w_jerk * jerk.pow(2).mean()
                #     + w_steer * steer.pow(2).mean()
                #     + w_steer_change * ds.pow(2).mean()
                #     + w_speed * speed_reg
                # )

                imitation_loss = F.smooth_l1_loss(plan[:, :, :2], ground_truth[:, 0, :, :2], reduction='sum')

                if hasattr(info, "last_err") and info.last_err is not None:
                    plan_cost = info.last_err.mean()
                else:
                    plan_cost = torch.tensor(0.0, device=ego.device, dtype=plan.dtype)

                # loss = (
                #     args.lambda_pred * pred_loss
                #     + args.lambda_score * score_loss
                #     + args.lambda_imitation * imitation_loss
                #     + args.lambda_cost * (cost_reg + w_col * plan_cost)
                # )
                loss = ( base_loss
                    + args.lambda_imitation * imitation_loss
                    + args.lambda_cost * plan_cost)

            else:
                # ====================
                """Without Planning"""
                # ====================

                plan, prediction = select_future(plan_trajs, predictions, scores, best_mode)

                loss = (base_loss + args.lambda_imitation * imitation_loss)

        # Backpropagation
        loss.backward()
        nn.utils.clip_grad_norm_(predictor.parameters(), 1.0)
        optimizer.step()
    
        # metrics
        metrics = motion_metrics(plan, prediction, ground_truth, weights)
        epoch_metrics.append(metrics)
        epoch_loss.append(loss.item())

        # progress
        current += batch[0].shape[0]
        sys.stdout.write(
            f"\rTrain Progress: [{current:>6d}/{size:>6d}]  "
            f"Loss: {np.mean(epoch_loss):>.4f}  {(time.time()-start_time)/current:>.4f}s/sample"
        )
        sys.stdout.flush()

    epoch_metrics = np.array(epoch_metrics)
    plannerADE, plannerFDE = np.mean(epoch_metrics[:, 0]), np.mean(epoch_metrics[:, 1])
    predictorADE, predictorFDE = np.mean(epoch_metrics[:, 2]), np.mean(epoch_metrics[:, 3])
    epoch_metrics = [plannerADE, plannerFDE, predictorADE, predictorFDE]

    logging.info(
        f"\nplannerADE: {plannerADE:.4f}, plannerFDE: {plannerFDE:.4f}, "
        f"predictorADE: {predictorADE:.4f}, predictorFDE: {predictorFDE:.4f}"
    )


    return np.mean(epoch_loss), epoch_metrics


def valid_epoch(data_loader, predictor, planner, use_planning):
    epoch_loss = []
    epoch_metrics = []
    current = 0
    size = len(data_loader.dataset)
    predictor.eval()
    start_time = time.time()

    for batch in data_loader:

        ego = batch[0].to(args.device)
        neighbors = batch[1].to(args.device)
        ground_truth = batch[2].to(args.device)
        map_segments = None
        map_mask     = None
        if len(batch) >= 5:
            map_segments = batch[3].to(args.device)  # (B,M,4)
            map_mask     = batch[4].to(args.device)  # (B,M)

        current_state = torch.cat([ego.unsqueeze(1), neighbors[..., :-1]], dim=1)[:, :, -1]
        weights = torch.ne(ground_truth[:, 1:, :, :2], 0)

       
        with torch.no_grad():

             """ =========================
             PREDICTOR VAE VALIDATION
             ============================= """ 

             if args.predictor == "PredictorVAE":
                if map_segments is None:
                    plans, predictions, cost_function_weights, mu, logvar, priori_mu, priori_logvar = predictor(
                        ego, neighbors, ground_truth
                    )
                else:
                    plans, predictions, cost_function_weights, mu, logvar, priori_mu, priori_logvar = predictor(
                        ego, neighbors, ground_truth, map_segments, map_mask
    )

                plan_trajs = torch.stack([bicycle_model(plans[:, i], ego[:, -1])[:, :, :3] for i in range(1)], dim=1)
                
                if args.use_prior:
                     _, pred_loss, kl_loss = VAE_loss(plan_trajs, predictions, ground_truth, weights, mu, logvar, ego_in_pred_loss=not use_planning, beta=1.0, prior_mu=priori_mu, prior_logvar=priori_logvar)
                else:
                     _, pred_loss, kl_loss = VAE_loss(plan_trajs, predictions, ground_truth, weights, mu, logvar, ego_in_pred_loss=not use_planning, beta=1.0)

                best_plan = plan_trajs.squeeze(1)

                base_loss = (args.lambda_pred * pred_loss + args.lambda_score * kl_loss)


            
                """ =========================
                      WITH PLANNING
                ============================= """ 
                if use_planning:

                    plan = plans[:, 0]  # (B, T, 2)
                    prediction = predictions[:, 0]

                    w = torch.nan_to_num(
                        cost_function_weights,
                        nan=1.0,
                        posinf=20.0,
                        neginf=1e-2,
                    )

                    w = torch.clamp(w, min=1e-2, max=20.0)

                    gt_ego = ground_truth[:, 0, :, :2]
                    planner_inputs = {
                        "control_variables": plan.view(ego.shape[0], 24),
                        "predictions": prediction,
                        "current_state": current_state,
                        "gt_trajectory": gt_ego,

                        "w_acc":          w[:, 0].unsqueeze(1),
                        "w_jerk":         w[:, 1].unsqueeze(1),
                        "w_steer":        w[:, 2].unsqueeze(1),
                        "w_steer_change": w[:, 3].unsqueeze(1),
                        "w_collision":    w[:, 4].unsqueeze(1),
                        "w_speed_limit":  w[:, 5].unsqueeze(1),
                        "w_endpoint_goal": w[:, 6].unsqueeze(1),}

                    final_values, info = planner.layer.forward(planner_inputs)
                    u_opt = final_values["control_variables"].view(-1, 12, 2)
                    ego_traj = bicycle_model(u_opt, ego[:, -1], dt=0.4)

                    plan = ego_traj[:, :, :3]   # x, y, theta
                    speed = ego_traj[:, :, 3]   # v

                    # Calcular cada residuo en PyTorch para que el gradiente llegue a cost_weights
                    # dt = 0.4
                    # w_acc          = w[:, 0].mean()
                    # w_jerk         = w[:, 1].mean()
                    # w_steer        = w[:, 2].mean()
                    # w_steer_change = w[:, 3].mean()
                    # w_col          = w[:, 4].mean()
                    # w_speed        = w[:, 5].mean()

                    # acc    = u_opt[:, :, 0]                            # (B, T)
                    # jerk   = torch.diff(acc,   dim=1) / dt            # (B, T-1)
                    # steer  = u_opt[:, :, 1]                           # (B, T)
                    # ds     = torch.diff(steer, dim=1) / dt            # (B, T-1)

                    
                    # speed_violation = torch.clamp(speed - args.max_speed, min=0.0)
                    # speed_reg = speed_violation.pow(2).mean()

                    # cost_reg = (
                    #     w_acc * acc.pow(2).mean()
                    #     + w_jerk * jerk.pow(2).mean()
                    #     + w_steer * steer.pow(2).mean()
                    #     + w_steer_change * ds.pow(2).mean()
                    #     + w_speed * speed_reg
                    # )

                    if hasattr(info, "last_err") and info.last_err is not None:
                        plan_cost = info.last_err.mean()
                    else:
                        plan_cost = torch.tensor(0.0, device=ego.device, dtype=plan.dtype)

                    imitation_loss = F.smooth_l1_loss(plan[:, :, :2], ground_truth[:, 0, :, :2], reduction='sum')

                    # loss = (base_loss + args.lambda_imitation * imitation_loss + args.lambda_cost * (cost_reg + w_col * plan_cost))

                    loss = (base_loss + args.lambda_imitation * imitation_loss + args.lambda_cost * plan_cost)
                
                else:
                    """ =========================
                            WITHOUT PLANNING
                    ============================= """ 
                    plan = best_plan
                    prediction = predictions[:,0] 

                    imitation_loss = F.smooth_l1_loss(plan[:, :, :2], ground_truth[:, 0, :, :2],  reduction='sum')

                    loss = (base_loss + args.lambda_imitation * imitation_loss)
            
             else:
                """ =========================
                    PREDICTOR VALIDATION
                ============================= """ 

                if args.use_map:
                
                    plans, predictions, scores, cost_function_weights = predictor(ego, neighbors, map_segments=map_segments, map_mask=map_mask)
                else:
                    plans, predictions, scores, cost_function_weights = predictor(ego, neighbors)

                plan_trajs = torch.stack([bicycle_model(plans[:, i], ego[:, -1])[:, :, :2] for i in range(NUM_MODES)], dim=1)

                # ego_in_pred_loss=False cuando use_planning: el ego se supervisa via imitation_loss sobre plan optimizado
                _, pred_loss, score_loss, best_mode = MFMA_loss(plan_trajs, predictions, scores, ground_truth, weights, ego_in_pred_loss=not use_planning)

                best_plan = plan_trajs[torch.arange(plan_trajs.shape[0], device=plan_trajs.device), best_mode]

                base_loss = (args.lambda_pred * pred_loss + args.lambda_score * score_loss)
                


                """ =========================
                      WITH PLANNING
                ============================= """ 
                if use_planning:
                       # selecciona el modo ganador para iniciar el optimizador
                    plan, prediction = select_future(plans, predictions, scores, best_mode)

                    
                    w = torch.nan_to_num(
                    cost_function_weights,
                    nan=1.0,
                    posinf=20.0,
                    neginf=1e-2,
                    )

                    w = torch.clamp(w, min=1e-2, max=20.0)
                    gt_ego = ground_truth[:, 0, :, :2]
                    planner_inputs = {
                        "control_variables": plan.view(ego.shape[0], 24),
                        "predictions": prediction,
                        "current_state": current_state,
                        "gt_trajectory": gt_ego,

                        "w_acc":          w[:, 0].unsqueeze(1),
                        "w_jerk":         w[:, 1].unsqueeze(1),
                        "w_steer":        w[:, 2].unsqueeze(1),
                        "w_steer_change": w[:, 3].unsqueeze(1),
                        "w_collision":    w[:, 4].unsqueeze(1),
                        "w_speed_limit":  w[:, 5].unsqueeze(1),
                        "w_endpoint_goal": w[:, 6].unsqueeze(1),}

                    final_values, info = planner.layer.forward(planner_inputs)
                    u_opt = final_values["control_variables"].view(-1, 12, 2)
                    ego_traj = bicycle_model(u_opt, ego[:, -1], dt=0.4)

                    plan = ego_traj[:, :, :3]   # x, y, theta
                    speed = ego_traj[:, :, 3]   # v

                    # Calcular cada residuo en PyTorch para que el gradiente llegue a cost_weights
                    # dt = 0.4
                    # w_acc          = w[:, 0].mean()
                    # w_jerk         = w[:, 1].mean()
                    # w_steer        = w[:, 2].mean()
                    # w_steer_change = w[:, 3].mean()
                    # w_col          = w[:, 4].mean()
                    # w_speed        = w[:, 5].mean()

                    # acc    = u_opt[:, :, 0]                            # (B, T)
                    # jerk   = torch.diff(acc,   dim=1) / dt            # (B, T-1)
                    # steer  = u_opt[:, :, 1]                           # (B, T)
                    # ds     = torch.diff(steer, dim=1) / dt            # (B, T-1)
                    
                    # speed_violation = torch.clamp(speed - args.max_speed, min=0.0)
                    # speed_reg = speed_violation.pow(2).mean()

                    # cost_reg = (
                    #     w_acc * acc.pow(2).mean()
                    #     + w_jerk * jerk.pow(2).mean()
                    #     + w_steer * steer.pow(2).mean()
                    #     + w_steer_change * ds.pow(2).mean()
                    #     + w_speed * speed_reg
                    # )

                    plan_cost = info.last_err.mean() if hasattr(info, "last_err") else 0

                    imitation_loss = F.smooth_l1_loss(plan[:, :, :2], ground_truth[:, 0, :, :2], reduction='sum')

                    loss = (base_loss + args.lambda_imitation * imitation_loss + args.lambda_cost * plan_cost)
                else:
                    """ =========================
                              WITHOUT PLANNING
                    ============================= """ 

                    plan, prediction = select_future(plan_trajs, predictions, scores, best_mode)
                    imitation_loss = F.smooth_l1_loss(plan[:, :, :2], ground_truth[:, 0, :, :2],  reduction='sum')

                    loss = (base_loss + args.lambda_imitation * imitation_loss)

        metrics = motion_metrics(plan, prediction, ground_truth, weights)
        epoch_metrics.append(metrics)
        epoch_loss.append(loss.item())

        current += batch[0].shape[0]
        sys.stdout.write(
            f"\rValid Progress: [{current:>6d}/{size:>6d}]  "
            f"Loss: {np.mean(epoch_loss):>.4f}  {(time.time()-start_time)/current:>.4f}s/sample"
        )
        sys.stdout.flush()

    epoch_metrics = np.array(epoch_metrics)
    plannerADE, plannerFDE = np.mean(epoch_metrics[:, 0]), np.mean(epoch_metrics[:, 1])
    predictorADE, predictorFDE = np.mean(epoch_metrics[:, 2]), np.mean(epoch_metrics[:, 3])
    epoch_metrics = [plannerADE, plannerFDE, predictorADE, predictorFDE]

    logging.info(
        f"\nval-plannerADE: {plannerADE:.4f}, val-plannerFDE: {plannerFDE:.4f}, "
        f"val-predictorADE: {predictorADE:.4f}, val-predictorFDE: {predictorFDE:.4f}"
    )


    return np.mean(epoch_loss), epoch_metrics


def model_training():
    # Logging
    script_dir = os.path.dirname(os.path.abspath(__file__))
    log_path   = os.path.join(script_dir, f"training_log/{args.name}/")
    os.makedirs(log_path, exist_ok=True)

    initLogging(log_file=os.path.join(log_path, "train.log"))

    logging.info("------------- {} -------------".format(args.name))
    logging.info("--- Use device: {}".format(args.device))
    logging.info("--- Batch size: {}".format(args.batch_size))
    logging.info("--- Learning rate: {}".format(args.learning_rate))
    logging.info("--- Using {}".format(args.predictor))
    logging.info("--- Use integrated planning module: {}".format(args.use_planning))

    # Set seed
    set_seed(args.seed)

    # Predictor o PredictorVAE
    if args.predictor == "PredictorVAE":
        predictor = PredictorVAE(
            12,
            predict_positions=args.predict_positions,
            use_prior=args.use_prior,
            use_map=args.use_map,
        ).to(args.device)
    else:
        predictor = Predictor(
            12,
            predict_positions=args.predict_positions,
            use_map=args.use_map,
        ).to(args.device)

    # planner
    if args.use_planning:
        # NOTE (Review MotionPlanner initialization)
        trajectory_len, feature_len = 12, 9
        planner = MotionPlanner(trajectory_len=trajectory_len, device=args.device, dt = 0.4)
    else:
        planner = None

        # ============================
    # Fine-tuning / Resume weights
    # ============================
    if args.finetune_from is not None:
        logging.info(f"--- Loading pretrained weights from: {args.finetune_from}")

        state_dict = torch.load(args.finetune_from, map_location=args.device)

        # Por compatibilidad con checkpoints guardados de distintas formas
        if isinstance(state_dict, dict) and "state_dict" in state_dict:
            state_dict = state_dict["state_dict"]

        missing, unexpected = predictor.load_state_dict(state_dict, strict=False)

        logging.info(f"--- Fine-tune loaded.")
        logging.info(f"--- Missing keys: {missing}")
        logging.info(f"--- Unexpected keys: {unexpected}")

    # optimizer
    optimizer = optim.Adam(predictor.parameters(), lr=args.learning_rate)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=3, gamma=.99)

    # Hyperparameters
    train_epochs = args.train_epochs
    batch_size   = args.batch_size

    # Dataset loaders
    train_set = DrivingData(
        args.train_set,
        ego_frame=True,
        use_map=args.use_map,
        map_root=args.map_root,
        map_radius=args.map_radius,
        map_max_segments=args.map_max_segments,
        prefilter_margin=args.map_prefilter_margin,
        dataset=args.dataset,
    )

    valid_set = DrivingData(
        args.valid_set,
        ego_frame=True,
        use_map=args.use_map,
        map_root=args.map_root,
        map_radius=args.map_radius,
        map_max_segments=args.map_max_segments,
        prefilter_margin=args.map_prefilter_margin,
        dataset=args.dataset,
)

    # DataLoaders
    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True, num_workers=args.num_workers, pin_memory=True)
    valid_loader = DataLoader(valid_set, batch_size=batch_size, shuffle=False, num_workers=args.num_workers, pin_memory=True)

    logging.info("--- Dataset Prepared: {} train data, {} validation data\n".format(len(train_set), len(valid_set)))
    logging.info(f"--- Use map: {args.use_map}")
    if args.use_map:
        logging.info(f"--- map_root: {args.map_root}, R={args.map_radius}, max_segments={args.map_max_segments}")

    # train loop
    for epoch in range(train_epochs):

        logging.info(f"--- Epoch {epoch+1}/{train_epochs}")

        # With a planner, pretrain predictor sin planning
        use_planning_now = args.use_planning
        if planner is not None:
            use_planning_now = (epoch >= args.pretrain_epochs)

        train_loss, train_metrics = train_epoch(train_loader, predictor, planner, optimizer, use_planning_now)
        val_loss, val_metrics = valid_epoch(valid_loader, predictor, planner, use_planning_now)

        scheduler.step()

        """wandb"""

        log = {
            "epoch": epoch + 1,
            "train/loss": train_loss,
            "val/loss": val_loss,
            "train/planner_ADE": train_metrics[0],
            "train/planner_FDE": train_metrics[1],
            "train/predictor_ADE": train_metrics[2],
            "train/predictor_FDE": train_metrics[3],
            "val/planner_ADE": val_metrics[0],
            "val/planner_FDE": val_metrics[1],
            "val/predictor_ADE": val_metrics[2],
            "val/predictor_FDE": val_metrics[3],
            "lr": optimizer.param_groups[0]["lr"],
        }

        log.update(get_current_cost_weights(predictor))

        wandb.log(log, step=epoch + 1)

        csv_path = os.path.join(script_dir, f"training_log/{args.name}/train_log.csv")
        if epoch == 0:
            with open(csv_path, "w") as csv_file:
                writer = csv.writer(csv_file)
                writer.writerow(log.keys())
                writer.writerow(log.values())
        else:
            with open(csv_path, "a") as csv_file:
                writer = csv.writer(csv_file)
                writer.writerow(log.values())


        # save model
        torch.save(predictor.state_dict(), os.path.join(script_dir, f"training_log/{args.name}/model_{epoch+1}_{val_metrics[0]:.4f}.pth"))

        logging.info(f"Model saved in training_log/{args.name}\n")

    # plots
    # logging.info("=" * 60)
    # logging.info("Generating training graphs...")
    # logging.info("=" * 60)

    # try:
    #     log_csv_path = os.path.join(script_dir, f"training_log/{args.name}/train_log.csv")
    #     save_dir = os.path.join(script_dir, f"training_log/{args.name}")
    #     plot_script = os.path.join(script_dir, "plot_training.py")
    #     result = subprocess.run(
    #         ["python", plot_script, "--log_path", log_csv_path, "--save_dir", save_dir],
    #         capture_output=True,
    #         text=True,
    #         check=True,
    #         cwd=script_dir,
    #     )
    #     logging.info(f"Graphs saved in: training_log/{args.name}/")
    #     if result.stdout:
    #         logging.info(result.stdout)
    # except subprocess.CalledProcessError as e:
    #     logging.warning(f"Error al generar gráficas: {e}")
    #     if e.stderr:
    #         logging.warning(e.stderr)
    # except FileNotFoundError:
    #     logging.warning("No se encontró plot_training.py")

    logging.info("=" * 60)
    logging.info("ENTRENAMIENTO COMPLETADO")
    logging.info("=" * 60)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Training")

    
    parser.add_argument("--name", type=str, help='log name (default: "Exp1")', default="Exp1")
    parser.add_argument("--predictor", type=str, help="Predictor PredictorVAE", default="Predictor")
    parser.add_argument("--train_set", type=str, help="path to train datasets", required=True)
    parser.add_argument("--valid_set", type=str, help="path to validation datasets", required=True)
    parser.add_argument("--seed", type=int, help="fix random seed", default=42)
    parser.add_argument("--num_workers", type=int, default=8, help="number of workers used for dataloader")
    parser.add_argument("--pretrain_epochs", type=int, help="epochs of pretraining predictor", default=2)
    parser.add_argument("--train_epochs", type=int, help="epochs of training", default=10)
    parser.add_argument("--batch_size", type=int, help="batch size (default: 32)", default=32)
    parser.add_argument("--learning_rate", type=float, help="learning rate (default: 2e-4)", default=2e-4)
    parser.add_argument("--use_planning", action="store_true", help="if use integrated planning module (default: False)", default=False)
    parser.add_argument("--device", type=str, help="run on which device (default: cuda)", default="cuda")

    parser.add_argument(
    "--finetune_from",
    type=str,
    default=None,
    help="Checkpoint .pth desde el cual iniciar fine-tuning."
    )

    # Weights 
    parser.add_argument("--lambda_pred",       type=float, default=1,  help="peso perdida de prediccion")
    parser.add_argument("--lambda_score",      type=float, default=0.2,  help="peso perdida de score")
    parser.add_argument("--lambda_imitation",  type=float, default=1,  help="peso perdida de imitacion")
    parser.add_argument("--lambda_cost",       type=float, default=1e-3, help="peso del costo Theseus (solo con planning)")

    parser.add_argument("--max_speed", type=float, default=1.5, help="velocidad maxima permitida para el ego en m/s")
    parser.add_argument("--predict_positions", action="store_true", help="predecir posiciones (x,y) absolutas en vez de desplazamientos (dx,dy) para vecinos")
    parser.add_argument("--use_prior", action="store_true", help="Aprender distribución prior (default: False)", default=False)

    # MAPS
    parser.add_argument("--use_map", action="store_true", help="Activar contexto de mapa (segmentos locales).")
    parser.add_argument("--map_root", type=str, default="mapa", help="Raíz del mapa: mapa/<scene>/obstacles.txt")
    parser.add_argument("--map_radius", type=float, default=7.0, help="Radio del mapa local alrededor del ego.")
    parser.add_argument("--map_max_segments", type=int, default=10, help="Máximo # de segmentos locales (padding).")
    parser.add_argument("--map_prefilter_margin", type=float, default=1.0, help="Margen extra para el prefiltrado (metros).")
    parser.add_argument("--dataset", type=str, default="eth_ucy", choices=["eth_ucy", "thor_magni"],
    help="Dataset usado para mapear scene_id a nombres de carpetas de mapa.")
    args = parser.parse_args()

    if args.device.startswith("cuda") and torch.cuda.is_available():
        # fuerza el device global a 0 para que Theseus no intente cuda:1
        torch.cuda.set_device(0)
        # normaliza "cuda" -> "cuda:0"
        if args.device == "cuda":
            args.device = "cuda:0"
    
    if args.predictor == "PredictorVAE":
        NUM_MODES = 1
    else:
        NUM_MODES = 20

    """WANDB"""

    wandb.init(project="DIPP(primera prueba)", name=args.name)
    wandb.config.update(args)

    # Main training function
    model_training()

    wandb.finish()