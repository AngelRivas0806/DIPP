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
    Dataset simplificado para ego-only consolidado.

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
def MFMA_loss(plans, predictions, scores, ground_truth, weights, best_mode: Optional[torch.Tensor] = None):
    """
    Compatibilidad: si no pasas best_mode, se calcula aquí como antes.
    Cambios críticos:
      - No usa global best_mode.
      - Usa máscara booleana para vecinos.
      - Mantiene la idea original: elegir modo con dist en frames [5,11] y entrenar scores con CE.
    """
    # predictions: [B, modes, neighbors, T, 2]
    # plans:       [B, modes, T, 3]  (x,y,theta) (o al menos x,y)
    # ground_truth:[B, agents, T, F] agents=1+neighbors
    B, M, T, _ = plans.shape
    _, _, N, _, _ = predictions.shape

    # Máscara de vecinos presentes (B,N,T,1)
    neigh_mask = _neighbor_presence_mask_from_weights(weights, N=N, T=T, device=plans.device)
    # Para “existió alguna vez”
    neigh_any = neigh_mask.any(dim=2, keepdim=True)  # (B,N,1,1)

    # Aplica máscara a predicciones (solo para distancias y loss)
    predictions_masked = predictions * neigh_any[:, None].float()  # (B,1,N,1,1)

    # ---- Selección best_mode (como tu idea original)
    # usa frames 5 y 11 (requiere T>=12)
    idx = [5, 11] if T > 11 else [max(0, T//2), T-1]

    # dist vecinos en esos frames
    pred_dist = torch.norm(
        predictions_masked[:, :, :, idx, :2] - ground_truth[:, None, 1:, idx, :2],
        dim=-1
    )  # (B,M,N,len(idx))
    plan_dist = torch.norm(
        plans[:, :, idx, :2] - ground_truth[:, None, 0, idx, :2],
        dim=-1
    )  # (B,M,len(idx))

    pred_dist = pred_dist.mean(-1).sum(-1)  # (B,M)
    plan_dist = plan_dist.mean(-1)          # (B,M)

    if best_mode is None:
        best_mode = torch.argmin(plan_dist + pred_dist, dim=-1)  # (B,)

    score_loss = F.cross_entropy(scores, best_mode)

    # Seleccionar el modo
    best_mode_plan = plans[torch.arange(B, device=plans.device), best_mode]              # (B,T,3)
    best_mode_pred = predictions_masked[torch.arange(B, device=plans.device), best_mode] # (B,N,T,2)

    # Construir "prediction" concatenado (ego + vecinos) para mantener tu lógica
    dummy_theta = torch.zeros_like(best_mode_pred[:, :, :, :1])        # (B,N,T,1)
    best_mode_pred_3d = torch.cat([best_mode_pred, dummy_theta], dim=-1)  # (B,N,T,3)
    prediction_all = torch.cat([best_mode_plan.unsqueeze(1), best_mode_pred_3d], dim=1)  # (B,1+N,T,3)

    # ---- Loss: ego (3 si posible), vecinos (2)
    pred_loss = 0.0

    # Ego
    if ground_truth.shape[-1] >= 3:
        pred_loss = pred_loss + F.smooth_l1_loss(prediction_all[:, 0, :, :3], ground_truth[:, 0, :, :3])
        pred_loss = pred_loss + F.smooth_l1_loss(prediction_all[:, 0, -1, :3], ground_truth[:, 0, -1, :3])
    else:
        pred_loss = pred_loss + F.smooth_l1_loss(prediction_all[:, 0, :, :2], ground_truth[:, 0, :, :2])
        pred_loss = pred_loss + F.smooth_l1_loss(prediction_all[:, 0, -1, :2], ground_truth[:, 0, -1, :2])

    # Vecinos: enmascarar para ignorar vecinos ausentes
    # ground_truth vecinos: (B,N,T,2)
    gt_nei_xy = ground_truth[:, 1:, :, :2]
    pred_nei_xy = prediction_all[:, 1:, :, :2]  # (B,N,T,2)
    nei_l = F.smooth_l1_loss(pred_nei_xy, gt_nei_xy, reduction="none").mean(dim=-1, keepdim=True)  # (B,N,T,1)
    nei_l = (nei_l * neigh_mask.float()).sum() / (neigh_mask.float().sum().clamp_min(1.0))
    pred_loss = pred_loss + nei_l

    # último punto vecinos (también enmascarado por "existió")
    nei_last = F.smooth_l1_loss(pred_nei_xy[:, :, -1, :], gt_nei_xy[:, :, -1, :], reduction="none").mean(dim=-1, keepdim=True)  # (B,N,1)
    nei_last = (nei_last * neigh_any.squeeze(-1).float()).sum() / (neigh_any.squeeze(-1).float().sum().clamp_min(1.0))
    pred_loss = pred_loss + nei_last

    total = 0.5 * pred_loss + score_loss
    return total, best_mode


def select_future(plans, predictions, scores, best_mode: Optional[torch.Tensor] = None):
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
    Modelo cinemático adaptado para peatones
    Para datos de peatones: current_state = [x, y, vx, vy, ax, ay, 0, 0]
    """
    dt = 0.4  # 2.5 fps -> 0.4s

    x_0 = current_state[:, 0]
    y_0 = current_state[:, 1]

    # Si hay vx,vy: modo peatón (punto-masa)
    if current_state.shape[1] >= 4:
        vx_0 = current_state[:, 2]
        vy_0 = current_state[:, 3]

        # control peatón: (ax, ay)
        ax = control[:, :, 0]
        ay = control[:, :, 1]

        # integrar velocidades (T)
        vx = vx_0.unsqueeze(1) + torch.cumsum(ax * dt, dim=1)
        vy = vy_0.unsqueeze(1) + torch.cumsum(ay * dt, dim=1)

        # integrar posiciones (T)
        x = x_0.unsqueeze(1) + torch.cumsum(vx * dt, dim=1)
        y = y_0.unsqueeze(1) + torch.cumsum(vy * dt, dim=1)

        theta = torch.atan2(vy, vx)
        v = torch.hypot(vx, vy)

        traj = torch.stack([x, y, theta, v], dim=-1)

    else:
        # Modo vehículo (fallback)
        max_delta = 0.6
        max_a = 5
        L = 3.089

        theta_0 = current_state[:, 2]
        # Si solo trae v en idx 3:
        v_0 = current_state[:, 3]

        a = control[:, :, 0].clamp(-max_a, max_a)
        delta = control[:, :, 1].clamp(-max_delta, max_delta)

        v = torch.clamp(v_0.unsqueeze(1) + torch.cumsum(a * dt, dim=1), min=0.0)

        d_theta = v * delta / L
        theta = torch.fmod(theta_0.unsqueeze(1) + torch.cumsum(d_theta * dt, dim=1), 2 * torch.pi)

        x = x_0.unsqueeze(1) + torch.cumsum(v * torch.cos(theta) * dt, dim=1)
        y = y_0.unsqueeze(1) + torch.cumsum(v * torch.sin(theta) * dt, dim=1)

        traj = torch.stack([x, y, theta, v], dim=-1)

    return traj


def physical_model(control, current_state, dt=0.1):
    """
    Modelo físico simple (vehículo).
    FIX: respeta el dt que se pasa como argumento.
    """
    max_d_theta = 0.5
    max_a = 5

    x_0 = current_state[:, 0]
    y_0 = current_state[:, 1]
    theta_0 = current_state[:, 2]
    v_0 = torch.hypot(current_state[:, 3], current_state[:, 4])

    a = control[:, :, 0].clamp(-max_a, max_a)
    d_theta = control[:, :, 1].clamp(-max_d_theta, max_d_theta)

    v = torch.clamp(v_0.unsqueeze(1) + torch.cumsum(a * dt, dim=1), min=0.0)

    theta = torch.fmod(theta_0.unsqueeze(1) + torch.cumsum(d_theta * dt, dim=1), 2 * torch.pi)

    x = x_0.unsqueeze(1) + torch.cumsum(v * torch.cos(theta) * dt, dim=1)
    y = y_0.unsqueeze(1) + torch.cumsum(v * torch.sin(theta) * dt, dim=1)

    traj = torch.stack([x, y, theta, v], dim=-1)
    return traj