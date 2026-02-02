import torch

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


def compute_fde(pred_traj, gt_traj):
    """
    Final Displacement Error - minFDE (best mode).
    Args:
        pred_traj: (batch, num_modes, future_steps, 2)
        gt_traj: (batch, future_steps, 2)
    Returns:
        fde: escalar, error en el último timestep para el mejor modo
    """
    pred_final = pred_traj[:, :, -1, :]  # (batch, num_modes, 2)
    gt_final = gt_traj[:, -1, :]  # (batch, 2)
    gt_expanded = gt_final.unsqueeze(1)  # (batch, 1, 2)
    errors = torch.norm(pred_final - gt_expanded, dim=-1)  # (batch, num_modes)
    min_fde = errors.min(dim=-1)[0]  # (batch,)
    return min_fde.mean(), min_fde


def compute_miss_rate(pred_traj, gt_traj, threshold=2.0):
    """
    Miss Rate: porcentaje de predicciones donde el error final > threshold.
    Args:
        pred_traj: (batch, num_modes, future_steps, 2)
        gt_traj: (batch, future_steps, 2)
        threshold: umbral en metros (default: 2.0m)
    Returns:
        miss_rate: porcentaje de misses
    """
    pred_final = pred_traj[:, :, -1, :]  # (batch, num_modes, 2)
    gt_final = gt_traj[:, -1, :]  # (batch, 2)
    gt_expanded = gt_final.unsqueeze(1)  # (batch, 1, 2)
    errors = torch.norm(pred_final - gt_expanded, dim=-1)  # (batch, num_modes)
    min_errors = errors.min(dim=-1)[0]  # (batch,) - mejor modo
    misses = (min_errors > threshold).float()
    return misses.mean() * 100, misses  # porcentaje

