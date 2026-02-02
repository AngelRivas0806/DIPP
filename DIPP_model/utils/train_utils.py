import random
import numpy as np
import torch
import logging
import glob
import os
from torch.utils.data import Dataset
from torch.nn import functional as F

def __getitem__(self, idx):
        data = np.load(self.data_list[idx])
        ego = data['ego']
        neighbors = data['neighbors']
        # ref_line = data['ref_line'] 
        # map_lanes = data['map_lanes']
        # map_crosswalks = data['map_crosswalks']
        gt_future_states = data['gt_future_states']

        # return ego, neighbors, map_lanes, map_crosswalks, ref_line, gt_future_states
        return ego, neighbors, gt_future_states

def initLogging(log_file: str, level: str = "INFO"):
    logging.basicConfig(filename=log_file, filemode='w',
                        level=getattr(logging, level, None),
                        format='[%(levelname)s %(asctime)s] %(message)s',
                        datefmt='%m-%d %H:%M:%S')
    logging.getLogger().addHandler(logging.StreamHandler())

def set_seed(CUR_SEED):
    random.seed(CUR_SEED)
    np.random.seed(CUR_SEED)
    torch.manual_seed(CUR_SEED)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


"""=== Estructura del archivo .npz ===
Keys disponibles: ['ego', 'neighbors', 'gt_future_states']

ego:
  - Shape: (8, 8)
  - Dtype: float32
  - Primeros valores:
[0.18889 0.38368]

neighbors:
  - Shape: (10, 8, 9)
  - Dtype: float32
  - Primeros valores:
[ 0.65      0.48958  -0.004028 -0.001041  0.      ]

gt_future_states:
  - Shape: (11, 50, 3)
  - Dtype: float32
  - Primeros valores:
[0.19167   0.38368   2.2462397]"""


class DrivingData(Dataset):
    def __init__(self, data_dir):
        """
        Args:
            data_dir: Path to the consolidated .npz file (e.g., 'data/train_combined/data.npz')
                      OR path pattern for multiple files (e.g., 'data/train_combined/*.npz')
        """
        # Check if it's a single .npz file or a directory pattern
        if data_dir.endswith('.npz') and os.path.isfile(data_dir):
            # New format: Single consolidated .npz file
            print(f"Loading consolidated dataset from: {data_dir}")
            data = np.load(data_dir)
            self.ego = data['ego']
            self.neighbors = data['neighbors']
            self.gt_future_states = data['gt_future_states']
            self.consolidated = True
            print(f"  - Loaded {len(self.ego)} samples")
        else:
            # Old format: Multiple .npz files
            print(f"Loading multiple files from: {data_dir}")
            self.data_list = glob.glob(data_dir)
            self.consolidated = False
            print(f"  - Found {len(self.data_list)} files")

    def __len__(self):
        if self.consolidated:
            return len(self.ego)
        else:
            return len(self.data_list)

    def __getitem__(self, idx):
        if self.consolidated:
            # Load from pre-loaded arrays
            ego = self.ego[idx].copy()
            neighbors = self.neighbors[idx].copy()
            gt_future_states = self.gt_future_states[idx].copy()
        else:
            # Load from individual files (old format)
            try:
                data = np.load(self.data_list[idx])
                ego = data['ego']
                neighbors = data['neighbors']
                gt_future_states = data['gt_future_states']
            except (ValueError, TypeError, KeyError):
                # Si hay problema con este archivo, intentar con el siguiente
                return self.__getitem__((idx + 1) % len(self.data_list))
        
        # ========== FILTRAR DATOS INVÁLIDOS ==========
        # Si el ego está completamente vacío (todo ceros), saltar este sample
        if np.all(ego == 0):
            if self.consolidated:
                # En modo consolidado, retornar el siguiente índice válido
                return self.__getitem__((idx + 1) % len(self))
            else:
                return self.__getitem__((idx + 1) % len(self.data_list))
        
        # ref_line = data['ref_line'] 
        # map_lanes = data['map_lanes']
        # map_crosswalks = data['map_crosswalks']

        # ========== NORMALIZACIÓN AL SISTEMA DE REFERENCIA DEL EGO ==========
        # ========== NORMALIZACIÓN AL SISTEMA DE REFERENCIA DEL EGO ==========
        # En process_eth_ucy.py el formato es: [x, y, vx, vy, ax, ay, 0, 0]
        # Por lo tanto, no tenemos theta explícito, debemos calcularlo.
        
        # Obtener velocidades del último frame
        vx_end = ego[-1, 2]
        vy_end = ego[-1, 3]
        
        # Calcular heading actual
        if np.abs(vx_end) < 1e-2 and np.abs(vy_end) < 1e-2:
            ego_current_heading = 0.0 # Si está quieto, asume 0
        else:
            ego_current_heading = np.arctan2(vy_end, vx_end)
            
        ego_current_pos = ego[-1, :2].copy()  # (x, y)
        
        cos_h = np.cos(-ego_current_heading)
        sin_h = np.sin(-ego_current_heading)
        
        # 1. Normalizar ego
        ego[:, :2] = ego[:, :2] - ego_current_pos  # trasladar
        
        # Rotar posiciones (x, y) -> indices 0, 1
        x_rot = ego[:, 0] * cos_h - ego[:, 1] * sin_h
        y_rot = ego[:, 0] * sin_h + ego[:, 1] * cos_h
        ego[:, 0] = x_rot
        ego[:, 1] = y_rot
        
        # Rotar velocidades (vx, vy) -> indices 2, 3
        vx_rot = ego[:, 2] * cos_h - ego[:, 3] * sin_h
        vy_rot = ego[:, 2] * sin_h + ego[:, 3] * cos_h
        ego[:, 2] = vx_rot
        ego[:, 3] = vy_rot
        
        # Rotar aceleraciones (ax, ay) -> indices 4, 5
        ax_rot = ego[:, 4] * cos_h - ego[:, 5] * sin_h
        ay_rot = ego[:, 4] * sin_h + ego[:, 5] * cos_h
        ego[:, 4] = ax_rot
        ego[:, 5] = ay_rot

        # 2. Normalizar vecinos
        for i in range(neighbors.shape[0]):
            if neighbors[i, -1, 8] != 0:  # si el vecino existe (flag en idx 8)
                neighbors[i, :, :2] = neighbors[i, :, :2] - ego_current_pos
                
                # Rotar pos
                x_rot = neighbors[i, :, 0] * cos_h - neighbors[i, :, 1] * sin_h
                y_rot = neighbors[i, :, 0] * sin_h + neighbors[i, :, 1] * cos_h
                neighbors[i, :, 0] = x_rot
                neighbors[i, :, 1] = y_rot
                
                # Rotar vel
                vx_rot = neighbors[i, :, 2] * cos_h - neighbors[i, :, 3] * sin_h
                vy_rot = neighbors[i, :, 2] * sin_h + neighbors[i, :, 3] * cos_h
                neighbors[i, :, 2] = vx_rot
                neighbors[i, :, 3] = vy_rot
                
                 # Rotar acc
                ax_rot = neighbors[i, :, 4] * cos_h - neighbors[i, :, 5] * sin_h
                ay_rot = neighbors[i, :, 4] * sin_h + neighbors[i, :, 5] * cos_h
                neighbors[i, :, 4] = ax_rot
                neighbors[i, :, 5] = ay_rot

        
        # 3. Normalizar ground truth
        for i in range(gt_future_states.shape[0]):
            # Verificar si la trayectoria es válida (no todo ceros)
            if np.abs(gt_future_states[i]).sum() > 0.001:
                gt_future_states[i, :, :2] = gt_future_states[i, :, :2] - ego_current_pos
                
                # Rotar pos
                x_rot = gt_future_states[i, :, 0] * cos_h - gt_future_states[i, :, 1] * sin_h
                y_rot = gt_future_states[i, :, 0] * sin_h + gt_future_states[i, :, 1] * cos_h
                gt_future_states[i, :, 0] = x_rot
                gt_future_states[i, :, 1] = y_rot
                
                # Rotar vel (si existen en GT)
                if gt_future_states.shape[2] > 2:
                    vx_rot = gt_future_states[i, :, 2] * cos_h - gt_future_states[i, :, 3] * sin_h
                    vy_rot = gt_future_states[i, :, 2] * sin_h + gt_future_states[i, :, 3] * cos_h
                    gt_future_states[i, :, 2] = vx_rot
                    gt_future_states[i, :, 3] = vy_rot
        # ========== FIN NORMALIZACIÓN ==========
        
        # Pad or truncate gt_future_states to 12 frames
        if gt_future_states.shape[1] < 12:
            pad_length = 12 - gt_future_states.shape[1]
            padding = np.zeros((gt_future_states.shape[0], pad_length, gt_future_states.shape[2]))
            gt_future_states = np.concatenate([gt_future_states, padding], axis=1)
        elif gt_future_states.shape[1] > 12:
            gt_future_states = gt_future_states[:, :12, :]

        # return ego, neighbors, map_lanes, map_crosswalks, ref_line, gt_future_states
        return ego, neighbors, gt_future_states

def MFMA_loss(plans, predictions, scores, ground_truth, weights):
    global best_mode

    predictions = predictions * weights.unsqueeze(1)
    # Cambiar 9::10 a usar más frames para 12 timesteps (usar frames 5 y 11)
    # predictions has shape [B, modes, neighbors, T, 2]
    # ground_truth has shape [B, agents, T, 8] -> we need agents 1 to N, and features :2 (x,y)
    
    prediction_distance = torch.norm(predictions[:, :, :, [5, 11], :2] - ground_truth[:, None, 1:, [5, 11], :2], dim=-1)
    plan_distance = torch.norm(plans[:, :, [5, 11], :2] - ground_truth[:, None, 0, [5, 11], :2], dim=-1)
    prediction_distance = prediction_distance.mean(-1).sum(-1)
    plan_distance = plan_distance.mean(-1)

    best_mode = torch.argmin(plan_distance+prediction_distance, dim=-1) 
    score_loss = F.cross_entropy(scores, best_mode)
    best_mode_plan = torch.stack([plans[i, m] for i, m in enumerate(best_mode)])
    best_mode_prediction = torch.stack([predictions[i, m] for i, m in enumerate(best_mode)])
    
    # Pad neighbor predictions with a dummy theta (0) to match ego plan's 3D shape [x, y, theta]
    # best_mode_prediction: [B, neighbors, T, 2] -> [B, neighbors, T, 3]
    dummy_theta = torch.zeros_like(best_mode_prediction[:, :, :, :1])
    best_mode_prediction_3d = torch.cat([best_mode_prediction, dummy_theta], dim=-1)
    
    # Now execute concatenation with matching shapes
    prediction = torch.cat([best_mode_plan.unsqueeze(1), best_mode_prediction_3d], dim=1)

    prediction_loss: torch.tensor = 0
    # Loop over all agents (0=ego, 1..N=neighbors)
    for i in range(prediction.shape[1]):
        if i == 0:
             # Ego plan has 3 dims (x, y, theta) or 9 dims from planner? 
             # bicycle_model outputs 4 dims: x,y,theta,v. We slice :3 in train.py usually.
             # Wait, prediction[0] comes from 'plans' which comes from bicycle_model in train.py (line 60, 139) -> [x, y, theta]
             # But prediction[1:] comes from 'predictions' (AgentDecoder) -> [x, y]
             
             # So for ego (i=0), we compare x,y,theta? Or just x,y?
             # Let's compare all 3 for ego since planner produces them.
             prediction_loss += F.smooth_l1_loss(prediction[:, i, :, :3], ground_truth[:, i, :, :3])
             prediction_loss += F.smooth_l1_loss(prediction[:, i, -1, :3], ground_truth[:, i, -1, :3])
        else:
             # For neighbors (i>0), we only have [x, y]
             prediction_loss += F.smooth_l1_loss(prediction[:, i, :, :2], ground_truth[:, i, :, :2])
             prediction_loss += F.smooth_l1_loss(prediction[:, i, -1, :2], ground_truth[:, i, -1, :2])
        
    return 0.5 * prediction_loss + score_loss

def select_future(plans, predictions, scores):
    """
    Seleccionar el mejor futuro basado en los scores
    
    Args:
        plans: (batch, num_modes, pred_len, features)
        predictions: (batch, num_modes, num_neighbors, pred_len, features)
        scores: (batch, num_modes)
    """
    # Si best_mode ya fue calculado por MFMA_loss, usarlo
    global best_mode
    if 'best_mode' in globals() and best_mode is not None:
        plan = torch.stack([plans[i, m] for i, m in enumerate(best_mode)])
        prediction = torch.stack([predictions[i, m] for i, m in enumerate(best_mode)])
    else:
        # En inferencia, seleccionar el modo con mayor score
        best_mode_infer = torch.argmax(scores, dim=-1)
        plan = torch.stack([plans[i, m] for i, m in enumerate(best_mode_infer)])
        prediction = torch.stack([predictions[i, m] for i, m in enumerate(best_mode_infer)])

    return plan, prediction

def motion_metrics(plan_trajectory, prediction_trajectories, ground_truth_trajectories, weights):
    prediction_trajectories = prediction_trajectories * weights
    # Plan is 3D (x,y,theta), GT is 3D+... compare positions :2
    plan_distance = torch.norm(plan_trajectory[:, :, :2] - ground_truth_trajectories[:, 0, :, :2], dim=-1)
    
    # Prediction is 2D (x,y), GT is 3D+... compare positions :2
    prediction_distance = torch.norm(prediction_trajectories[:, :, :, :2] - ground_truth_trajectories[:, 1:, :, :2], dim=-1)

    # planning
    plannerADE = torch.mean(plan_distance)
    plannerFDE = torch.mean(plan_distance[:, -1])
    
    # prediction
    predictorADE = torch.mean(prediction_distance, dim=-1)
    predictorADE = torch.masked_select(predictorADE, weights[:, :, 0, 0])
    predictorADE = torch.mean(predictorADE)
    predictorFDE = prediction_distance[:, :, -1]
    predictorFDE = torch.masked_select(predictorFDE, weights[:, :, 0, 0])
    predictorFDE = torch.mean(predictorFDE)

    return plannerADE.item(), plannerFDE.item(), predictorADE.item(), predictorFDE.item()

def project_to_frenet_frame(traj, ref_line):
    distance_to_ref = torch.cdist(traj[:, :, :2], ref_line[:, :, :2])
    k = torch.argmin(distance_to_ref, dim=-1).view(-1, traj.shape[1], 1).expand(-1, -1, 3)
    ref_points = torch.gather(ref_line, 1, k)
    x_r, y_r, theta_r = ref_points[:, :, 0], ref_points[:, :, 1], ref_points[:, :, 2] 
    x, y = traj[:, :, 0], traj[:, :, 1]
    s = 0.1 * (k[:, :, 0] - 200)
    l = torch.sign((y-y_r)*torch.cos(theta_r)-(x-x_r)*torch.sin(theta_r)) * torch.sqrt(torch.square(x-x_r)+torch.square(y-y_r))
    sl = torch.stack([s, l], dim=-1)

    return sl

def project_to_cartesian_frame(traj, ref_line):
    k = (10 * traj[:, :, 0] + 200).long()
    k = torch.clip(k, 0, 1200-1)
    ref_points = torch.gather(ref_line, 1, k.view(-1, traj.shape[1], 1).expand(-1, -1, 3))
    x_r, y_r, theta_r = ref_points[:, :, 0], ref_points[:, :, 1], ref_points[:, :, 2] 
    x = x_r - traj[:, :, 1] * torch.sin(theta_r)
    y = y_r + traj[:, :, 1] * torch.cos(theta_r)
    xy = torch.stack([x, y], dim=-1)

    return xy

def bicycle_model(control, current_state):
    """
    Modelo cinemático adaptado para peatones
    Para datos de peatones: current_state = [x, y, vx, vy, ax, ay, 0, 0]
    """
    dt = 0.4 # discrete time period [s] (para 2.5 fps = 1/2.5 = 0.4s)
    
    # Para peatones, usar modelo de punto de masa simple
    x_0 = current_state[:, 0]  # x inicial
    y_0 = current_state[:, 1]  # y inicial
    
    # Verificar si tenemos theta (vehículos) o velocidades (peatones)
    # Si current_state tiene 8 features y las posiciones 2-3 son velocidades
    if current_state.shape[1] >= 4:
        vx_0 = current_state[:, 2]  # velocidad x
        vy_0 = current_state[:, 3]  # velocidad y
        
        # Control para peatones: [ax, ay] (aceleraciones)
        ax = control[:, :, 0]  # aceleración en x
        ay = control[:, :, 1]  # aceleración en y
        
        # Integrar velocidades
        vx = vx_0.unsqueeze(1) + torch.cumsum(ax * dt, dim=1)
        vy = vy_0.unsqueeze(1) + torch.cumsum(ay * dt, dim=1)
        
        # Integrar posiciones
        x = x_0.unsqueeze(1) + torch.cumsum(vx * dt, dim=-1)
        y = y_0.unsqueeze(1) + torch.cumsum(vy * dt, dim=-1)
        
        # Calcular heading desde velocidades
        theta = torch.atan2(vy, vx)
        
        # Velocidad escalar
        v = torch.hypot(vx, vy)
        
        # output trajectory [x, y, theta, v]
        traj = torch.stack([x, y, theta, v], dim=-1)
    else:
        # Modo vehículo original
        max_delta = 0.6  # vehicle's steering limits [rad]
        max_a = 5  # vehicle's acceleration limits [m/s^2]
        
        theta_0 = current_state[:, 2]  # vehicle's heading [rad]
        v_0 = torch.hypot(current_state[:, 3], current_state[:, 4]) if current_state.shape[1] > 4 else current_state[:, 3]
        L = 3.089  # vehicle's wheelbase [m]
        a = control[:, :, 0].clamp(-max_a, max_a)
        delta = control[:, :, 1].clamp(-max_delta, max_delta)
        
        # speed
        v = v_0.unsqueeze(1) + torch.cumsum(a * dt, dim=1)
        v = torch.clamp(v, min=0)
        
        # angle
        d_theta = v * delta / L
        theta = theta_0.unsqueeze(1) + torch.cumsum(d_theta * dt, dim=-1)
        theta = torch.fmod(theta, 2*torch.pi)
        
        # x and y coordinate
        x = x_0.unsqueeze(1) + torch.cumsum(v * torch.cos(theta) * dt, dim=-1)
        y = y_0.unsqueeze(1) + torch.cumsum(v * torch.sin(theta) * dt, dim=-1)
        
        # output trajectory
        traj = torch.stack([x, y, theta, v], dim=-1)

    return traj

def physical_model(control, current_state, dt=0.1):
    dt = 0.1 # discrete time period [s]
    max_d_theta = 0.5 # vehicle's change of angle limits [rad/s]
    max_a = 5 # vehicle's accleration limits [m/s^2]

    x_0 = current_state[:, 0] # vehicle's x-coordinate
    y_0 = current_state[:, 1] # vehicle's y-coordinate
    theta_0 = current_state[:, 2] # vehicle's heading [rad]
    v_0 = torch.hypot(current_state[:, 3], current_state[:, 4]) # vehicle's velocity [m/s]
    a = control[:, :, 0].clamp(-max_a, max_a) # vehicle's accleration [m/s^2]
    d_theta = control[:, :, 1].clamp(-max_d_theta, max_d_theta) # vehicle's heading change rate [rad/s]

    # speed
    v = v_0.unsqueeze(1) + torch.cumsum(a * dt, dim=1)
    v = torch.clamp(v, min=0)

    # angle
    theta = theta_0.unsqueeze(1) + torch.cumsum(d_theta * dt, dim=-1)
    theta = torch.fmod(theta, 2*torch.pi)
    
    # x and y coordniate
    x = x_0.unsqueeze(1) + torch.cumsum(v * torch.cos(theta) * dt, dim=-1)
    y = y_0.unsqueeze(1) + torch.cumsum(v * torch.sin(theta) * dt, dim=-1)

    # output trajectory
    traj = torch.stack([x, y, theta, v], dim=-1)

    return traj
