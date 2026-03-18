import random
import numpy as np
import torch
import logging
import os
from torch.utils.data import Dataset
from torch.nn import functional as F
from typing import Tuple, Optional

# =========================================================
# Logging + Reproducibility
# =========================================================
def initLogging(log_file: str, level: str = "INFO"):
    logging.basicConfig(
        filename=log_file,
        filemode="w",
        level=getattr(logging, level, None),
        format="[%(levelname)s %(asctime)s] %(message)s",
        datefmt="%m-%d %H:%M:%S",
    )
    logging.getLogger().addHandler(logging.StreamHandler())


def set_seed(CUR_SEED: int):
    random.seed(CUR_SEED)
    np.random.seed(CUR_SEED)
    torch.manual_seed(CUR_SEED)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


# =========================================================
# Dataset
# =========================================================
class DrivingData(Dataset):
    """
    Dataset

    Espera un .npz con keys:
      - ego:             (N, obs_len, 6)          [x,y,vx,vy,ax,ay]
      - neighbors:       (N, K, obs_len, 7)       [x,y,vx,vy,ax,ay,flag]
      - gt_future_states:(N, K+1, pred_len, F)    F>=2 (idealmente 6)

    Devuelve:
      ego, neighbors, gt_future_states como torch.float32 y con marco de referencia del ego
    """

    def __init__(self, npz_path: str, ego_frame: bool = True):
        assert npz_path.endswith(".npz") and os.path.isfile(npz_path), f"Archivo inválido: {npz_path}"
        self.npz_path = npz_path
        self.ego_frame = ego_frame

        data = np.load(npz_path, allow_pickle=False)
        self.ego = data["ego"].astype(np.float32)             # (N, obs, 6)
        self.neighbors = data["neighbors"].astype(np.float32) # (N, K, obs, 7)
        self.gt = data["gt_future_states"].astype(np.float32) # (N, K+1, pred, F)

        assert self.ego.ndim == 3 and self.ego.shape[-1] == 6, f"ego debe ser (N,obs,6), got {self.ego.shape}"
        assert self.neighbors.ndim == 4 and self.neighbors.shape[-1] == 7, f"neighbors debe ser (N,K,obs,7), got {self.neighbors.shape}"
        assert self.gt.ndim == 4, f"gt_future_states debe ser (N,K+1,pred,F), got {self.gt.shape}"

    def __len__(self) -> int:
        return self.ego.shape[0]

    @staticmethod
    def _rot2d(xy: np.ndarray, c: float, s: float) -> Tuple[np.ndarray, np.ndarray]:
        """Rota componentes (x,y) con cos/sin ya calculados."""
        xr = xy[..., 0] * c - xy[..., 1] * s
        yr = xy[..., 0] * s + xy[..., 1] * c
        return xr, yr

    def _ego_transform(self, ego: np.ndarray, neighbors: np.ndarray, gt: np.ndarray):
        """
        ego:       (obs,6)
        neighbors: (K,obs,7) flag en idx -1
        gt:        (K+1,pred,F)
        """
        # Origen: última observación del ego
        ego_pos0 = ego[-1, 0:2].copy()

        # Heading por velocidad en el último frame observado
        vx_end, vy_end = float(ego[-1, 2]), float(ego[-1, 3])
        if abs(vx_end) < 1e-2 and abs(vy_end) < 1e-2:
            heading = 0.0
        else:
            heading = float(np.arctan2(vy_end, vx_end))

        c = float(np.cos(-heading))
        s = float(np.sin(-heading))

        # ===== 1) Ego: trasladar + rotar pos/vel/acc =====
        ego[:, 0:2] = ego[:, 0:2] - ego_pos0
        xr, yr = self._rot2d(ego[:, 0:2], c, s)
        ego[:, 0], ego[:, 1] = xr, yr

        xr, yr = self._rot2d(ego[:, 2:4], c, s)
        ego[:, 2], ego[:, 3] = xr, yr

        xr, yr = self._rot2d(ego[:, 4:6], c, s)
        ego[:, 4], ego[:, 5] = xr, yr

        # ===== 2) Neighbors: solo si flag==1 (tomamos flag del último obs frame) =====
        flag_idx = -1
        for k in range(neighbors.shape[0]):
            if neighbors[k, -1, flag_idx] <= 0.0:
                continue

            neighbors[k, :, 0:2] = neighbors[k, :, 0:2] - ego_pos0
            xr, yr = self._rot2d(neighbors[k, :, 0:2], c, s)
            neighbors[k, :, 0], neighbors[k, :, 1] = xr, yr

            xr, yr = self._rot2d(neighbors[k, :, 2:4], c, s)
            neighbors[k, :, 2], neighbors[k, :, 3] = xr, yr

            xr, yr = self._rot2d(neighbors[k, :, 4:6], c, s)
            neighbors[k, :, 4], neighbors[k, :, 5] = xr, yr

        # ===== 3) GT: rotar pos/vel/acc si existen (y si no es todo ceros) =====
        Fdim = gt.shape[-1]
        for a in range(gt.shape[0]):  # (K+1)
            if np.abs(gt[a]).sum() <= 1e-6:
                continue

            gt[a, :, 0:2] = gt[a, :, 0:2] - ego_pos0
            xr, yr = self._rot2d(gt[a, :, 0:2], c, s)
            gt[a, :, 0], gt[a, :, 1] = xr, yr

            if Fdim >= 4:
                xr, yr = self._rot2d(gt[a, :, 2:4], c, s)
                gt[a, :, 2], gt[a, :, 3] = xr, yr

            if Fdim >= 6:
                xr, yr = self._rot2d(gt[a, :, 4:6], c, s)
                gt[a, :, 4], gt[a, :, 5] = xr, yr

        return ego, neighbors, gt

    def __getitem__(self, idx: int):
        ego = self.ego[idx].copy()
        neighbors = self.neighbors[idx].copy()
        gt = self.gt[idx].copy()

        if np.all(ego == 0):
            raise ValueError(f"Sample {idx} tiene ego vacío (todo ceros). Revisa tu preprocessing.")

        if self.ego_frame:
            ego, neighbors, gt = self._ego_transform(ego, neighbors, gt)

        return (
            torch.from_numpy(ego).float(),
            torch.from_numpy(neighbors).float(),
            torch.from_numpy(gt).float(),
        )


# =========================================================
# Helpers for weights/masks (robusto)
# =========================================================
def _neighbor_presence_mask_from_weights(weights: torch.Tensor, N: int, T: int, device=None) -> torch.Tensor:
    """
    Devuelve máscara booleana (B, N, T, 1) indicando presencia del vecino por frame.
    - Si weights ya viene como (B,N,T,1) o (B,N,T,2) -> usa weights[...,0]
    - Si weights viene como (B,N,1,1) -> expande a T
    - Si weights viene como (B,N) -> expande a (B,N,T,1)
    """
    if device is None:
        device = weights.device

    if weights is None:
        # si no hay weights, asume todos presentes
        return torch.ones((1, N, T, 1), dtype=torch.bool, device=device)

    w = weights

    # Normaliza dims
    if w.ndim == 4:
        # (B,N,T,C)
        mask = w[..., 0:1] > 0.5
        # si T no coincide, intenta arreglar
        if mask.shape[2] != T:
            if mask.shape[2] == 1:
                mask = mask.expand(-1, -1, T, -1)
            else:
                mask = mask[:, :, :T, :]
        return mask
    elif w.ndim == 3:
        # (B,N,T)
        mask = w.unsqueeze(-1) > 0.5
        if mask.shape[2] != T:
            if mask.shape[2] == 1:
                mask = mask.expand(-1, -1, T, -1)
            else:
                mask = mask[:, :, :T, :]
        return mask
    elif w.ndim == 2:
        # (B,N)
        return (w > 0.5).unsqueeze(-1).unsqueeze(-1).expand(-1, -1, T, 1)
    else:
        raise ValueError(f"weights shape inesperado: {w.shape}")


# =========================================================
# Loss + Selection
# =========================================================
def MFMA_loss(plans, predictions, scores, ground_truth, weights, best_mode: Optional[torch.Tensor] = None, ego_in_pred_loss: bool = True):
    B, M, T, _ = plans.shape
    _, _, N, _, _ = predictions.shape

    # Máscara de vecinos presentes (B,N,T,1)
    neigh_mask = _neighbor_presence_mask_from_weights(weights, N=N, T=T, device=plans.device)
    neigh_any  = neigh_mask.any(dim=2, keepdim=True)  # (B,N,1,1)
    if False:
        predictions = current_state[:, 1:, :2].unsqueeze(1).unsqueeze(3) + delta_predictions.cumsum(dim=3)  # (B,M,N,T,2)
    else:
        predictions = delta_predictions
    predictions_masked = predictions * neigh_any[:, None].float()


    if best_mode is None:
        # ---- Selección best_mode: suma de errores por modo (sin promediar)

        # Ego: suma sobre T y xy → (B, M)
        ego_loss_per_mode = F.smooth_l1_loss(
            plans[:, :, :, :2],
            ground_truth[:, None, 0, :, :2].expand(-1, M, -1, -1),
            reduction='none'
        ).sum(dim=[-1, -2])  # (B, M)

        # Vecinos: suma sobre xy → (B, M, N, T), luego suma sobre N y T enmascarando ausentes → (B, M)
        nei_loss_per_mode = F.smooth_l1_loss(
            predictions_masked[:, :, :, :, :2],
            ground_truth[:, None, 1:, :, :2].expand(-1, M, -1, -1, -1),
            reduction='none'
        ).sum(dim=-1)  # (B, M, N, T)

        nei_loss_per_mode = (nei_loss_per_mode * neigh_mask[:, None, :, :, 0].float()).sum(dim=[-1, -2])  # (B, M)

        # Suma conjunta: ego + vecinos (sin normalizar)
        joint_loss_per_mode = ego_loss_per_mode + nei_loss_per_mode  # (B, M)
        best_mode = torch.argmin(joint_loss_per_mode, dim=-1)  # (B,)

    score_loss = F.cross_entropy(scores, best_mode)

    # Seleccionar modo ganador
    best_mode_plan = plans[torch.arange(B, device=plans.device), best_mode]               # (B,T,3)
    best_mode_pred = predictions_masked[torch.arange(B, device=plans.device), best_mode]  # (B,N,T,2)

    dummy_theta      = torch.zeros_like(best_mode_pred[:, :, :, :1])
    best_mode_pred_3d= torch.cat([best_mode_pred, dummy_theta], dim=-1)
    prediction_all   = torch.cat([best_mode_plan.unsqueeze(1), best_mode_pred_3d], dim=1)  # (B,1+N,T,3)

    # ---- pred_loss sobre modo ganador: suma de errores (sin promediar)

    # Ego: solo si ego_in_pred_loss=True (con planning se supervisa vía imitation_loss)
    if ego_in_pred_loss:
        ego_l = F.smooth_l1_loss(prediction_all[:, 0, :, :2], ground_truth[:, 0, :, :2], reduction='sum')
    else:
        ego_l = torch.tensor(0.0, device=plans.device)
    pred_loss = ego_l

    # Vecinos: suma sobre xy, T y vecinos presentes (sin dividir)
    gt_nei_xy   = ground_truth[:, 1:, :, :2]
    pred_nei_xy = prediction_all[:, 1:, :, :2]
    nei_l = F.smooth_l1_loss(pred_nei_xy, gt_nei_xy, reduction='none').sum(dim=-1, keepdim=True)  # (B,N,T,1)
    nei_l = (nei_l * neigh_mask.float()).sum()  # escalar: suma total
    pred_loss = pred_loss + nei_l

    total = 0.5 * pred_loss + score_loss
    return total, pred_loss, score_loss, best_mode


def select_future(current_state, plans, delta_predictions, scores, best_mode: Optional[torch.Tensor] = None):
    """
    Seleccionar el mejor futuro basado en los scores (inferencia) o best_mode (entrenamiento)
    Args:
        plans: (B, M, T, Fp)
        predictions: (B, M, N, T, Fq)
        scores: (B, M)
        best_mode: (B,) opcional
    """
    B = plans.shape[0]
    if best_mode is None:
        best_mode = torch.argmax(scores, dim=-1)

    plan = plans[torch.arange(B, device=plans.device), best_mode]
    if False:
        predictions = current_state[:, 1:, :2].unsqueeze(1).unsqueeze(3) + delta_predictions.cumsum(dim=3)  # (B,M,N,T,2)
    else:   
        predictions = delta_predictions
    prediction = predictions[torch.arange(B, device=plans.device), best_mode]
    return plan, prediction


def motion_metrics(plan_trajectory, prediction_trajectories, ground_truth_trajectories, weights):
    """
    plan_trajectory: (B, T, 3)
    prediction_trajectories: (B, N, T, 2)
    ground_truth_trajectories: (B, 1+N, T, F)
    weights: máscara/pesos para vecinos (idealmente 0/1)
    """
    B, T, _ = plan_trajectory.shape
    N = prediction_trajectories.shape[1]

    neigh_mask = _neighbor_presence_mask_from_weights(weights, N=N, T=T, device=plan_trajectory.device)  # (B,N,T,1)
    neigh_any = neigh_mask.any(dim=2, keepdim=True)  # (B,N,1,1)

    # planning
    plan_distance = torch.norm(plan_trajectory[:, :, :2] - ground_truth_trajectories[:, 0, :, :2], dim=-1)  # (B,T)
    plannerADE = torch.mean(plan_distance)
    plannerFDE = torch.mean(plan_distance[:, -1])

    # prediction
    pred_distance = torch.norm(
        prediction_trajectories[:, :, :, :2] - ground_truth_trajectories[:, 1:, :, :2],
        dim=-1,
        keepdim=True
    )  # (B,N,T,1)

    # ADE por vecino (promedio en T), luego promedio sobre vecinos presentes
    pred_ade_per_nei = pred_distance.mean(dim=2, keepdim=True)  # (B,N,1,1)
    predictorADE = (pred_ade_per_nei * neigh_any.float()).sum() / neigh_any.float().sum().clamp_min(1.0)

    pred_fde_per_nei = pred_distance[:, :, -1:, :]  # (B,N,1,1)
    predictorFDE = (pred_fde_per_nei * neigh_any.float()).sum() / neigh_any.float().sum().clamp_min(1.0)

    return plannerADE.item(), plannerFDE.item(), predictorADE.item(), predictorFDE.item()


# =========================================================
# Frame projections
# =========================================================
def project_to_frenet_frame(traj, ref_line):
    distance_to_ref = torch.cdist(traj[:, :, :2], ref_line[:, :, :2])
    k = torch.argmin(distance_to_ref, dim=-1).view(-1, traj.shape[1], 1).expand(-1, -1, 3)
    ref_points = torch.gather(ref_line, 1, k)
    x_r, y_r, theta_r = ref_points[:, :, 0], ref_points[:, :, 1], ref_points[:, :, 2]
    x, y = traj[:, :, 0], traj[:, :, 1]
    s = 0.1 * (k[:, :, 0] - 200)
    l = torch.sign((y - y_r) * torch.cos(theta_r) - (x - x_r) * torch.sin(theta_r)) * torch.sqrt(
        torch.square(x - x_r) + torch.square(y - y_r)
    )
    sl = torch.stack([s, l], dim=-1)
    return sl


def project_to_cartesian_frame(traj, ref_line):
    k = (10 * traj[:, :, 0] + 200).long()
    k = torch.clip(k, 0, 1200 - 1)
    ref_points = torch.gather(ref_line, 1, k.view(-1, traj.shape[1], 1).expand(-1, -1, 3))
    x_r, y_r, theta_r = ref_points[:, :, 0], ref_points[:, :, 1], ref_points[:, :, 2]
    x = x_r - traj[:, :, 1] * torch.sin(theta_r)
    y = y_r + traj[:, :, 1] * torch.cos(theta_r)
    xy = torch.stack([x, y], dim=-1)
    return xy


# =========================================================
# Motion models
# =========================================================
def bicycle_model(control, current_state):
    """
    Modelo cinemático de vehículo (bicycle model).
    current_state: (B, 6) = [x, y, vx, vy, ax, ay]  ← formato ego del dataset
                   (B, 4) = [x, y, theta, v]          ← formato compacto (fallback)
    control:       (B, T, 2) = [accel, steering]
    """
    dt = 0.4  # 0.4s por frame
    max_delta = 0.6
    max_a = 5.0
    L = 3.089  # wheelbase (robot/vehículo)

    x_0     = current_state[:, 0]
    y_0     = current_state[:, 1]

    if current_state.shape[1] >= 6:
        # Formato ego del dataset: [x, y, vx, vy, ax, ay]
        # En el frame ego-centrado: vx > 0, vy ≈ 0, theta ≈ 0
        vx_0    = current_state[:, 2]
        vy_0    = current_state[:, 3]
        theta_0 = torch.atan2(vy_0, vx_0)                          # (B,) heading inicial
        v_0     = torch.hypot(vx_0, vy_0)                          # (B,) velocidad inicial
    else:
        # Formato compacto: [x, y, theta, v]
        theta_0 = current_state[:, 2]
        v_0     = current_state[:, 3]

    a     = control[:, :, 0].clamp(-max_a, max_a)                  # (B, T)
    delta = control[:, :, 1].clamp(-max_delta, max_delta)          # (B, T)

    v = torch.clamp(v_0.unsqueeze(1) + torch.cumsum(a * dt, dim=1), min=0.0)  # (B, T)

    d_theta = v * delta / L                                         # (B, T)
    theta   = torch.fmod(theta_0.unsqueeze(1) + torch.cumsum(d_theta * dt, dim=1), 2 * torch.pi)  # (B, T)

    x = x_0.unsqueeze(1) + torch.cumsum(v * torch.cos(theta) * dt, dim=1)  # (B, T)
    y = y_0.unsqueeze(1) + torch.cumsum(v * torch.sin(theta) * dt, dim=1)  # (B, T)

    return torch.stack([x, y, theta, v], dim=-1)  # (B, T, 4)


# def physical_model(control, current_state, dt=0.1):
#     """
#     Modelo físico simple (vehículo).
#     FIX: respeta el dt que se pasa como argumento.
#     """
#     max_d_theta = 0.5
#     max_a = 5

#     x_0 = current_state[:, 0]
#     y_0 = current_state[:, 1]
#     theta_0 = current_state[:, 2]
#     v_0 = torch.hypot(current_state[:, 3], current_state[:, 4])

#     a = control[:, :, 0].clamp(-max_a, max_a)
#     d_theta = control[:, :, 1].clamp(-max_d_theta, max_d_theta)

#     v = torch.clamp(v_0.unsqueeze(1) + torch.cumsum(a * dt, dim=1), min=0.0)

#     theta = torch.fmod(theta_0.unsqueeze(1) + torch.cumsum(d_theta * dt, dim=1), 2 * torch.pi)

#     x = x_0.unsqueeze(1) + torch.cumsum(v * torch.cos(theta) * dt, dim=1)
#     y = y_0.unsqueeze(1) + torch.cumsum(v * torch.sin(theta) * dt, dim=1)

#     traj = torch.stack([x, y, theta, v], dim=-1)
#     return traj