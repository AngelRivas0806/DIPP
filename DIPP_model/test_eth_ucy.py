from xml.parsers.expat import model
import torch
import argparse
import numpy as np
import os
import logging
from tqdm import tqdm
from torch.utils.data import DataLoader

from model.predictorvae import PredictorVAE
from model.predictor import Predictor
from model.planner import MotionPlanner

from utils.train_utils import DrivingData, bicycle_model, select_future
from utils.visualization import save_visualizations_from_samples
import warnings

warnings.filterwarnings(
    "ignore",
    message="The PyTorch API of nested tensors is in prototype stage.*",
    category=UserWarning
)

# =========================================================
# Utils
# =========================================================

def compute_ade_fde(predictions, ground_truth):
    """
    predictions: (B, T, 2)
    ground_truth: (B, T, 2)
    """
    displacement = torch.norm(predictions - ground_truth, dim=-1)
    ade = torch.mean(displacement)
    fde = torch.mean(displacement[:, -1])
    return ade.item(), fde.item()

def compute_efficiency_metrics(
    ego_traj,
    control,
    neighbor_pred,
    safety_distance=0.5,
    dt=0.4,
):
    """
    ego_traj:      (B, T, 3) o (B, T, 4), trayectoria del ego
    control:       (B, T, 2), [acceleration, steering]
    neighbor_pred: (B, N, T, 2), predicciones de vecinos

    Returns dict con promedios por batch.
    """

    acc = control[:, :, 0]      # (B, T)
    steer = control[:, :, 1]    # (B, T)

    jerk = torch.diff(acc, dim=1) / dt            # (B, T-1)
    steer_change = torch.diff(steer, dim=1) / dt  # (B, T-1)

    acc_metric = acc.abs().mean().item()
    jerk_metric = jerk.abs().mean().item()
    steer_metric = steer.abs().mean().item()
    steer_change_metric = steer_change.abs().mean().item()

    # ==========================
    # Collision / safety violation
    # ==========================
    ego_pos = ego_traj[:, :, :2]              # (B, T, 2)
    neighbors_pos = neighbor_pred[:, :, :, :2]  # (B, N, T, 2)

    # vecinos válidos: si todo es cero, es padding
    neighbor_valid_mask = (
        neighbors_pos.abs().sum(dim=-1).sum(dim=-1) > 0.01
    )  # (B, N)

    ego_pos_exp = ego_pos.unsqueeze(1)  # (B, 1, T, 2)

    distances = torch.norm(
        ego_pos_exp - neighbors_pos,
        dim=-1
    )  # (B, N, T)

    distances = torch.where(
        neighbor_valid_mask.unsqueeze(-1).expand_as(distances),
        distances,
        torch.ones_like(distances) * 1e4
    )

    min_distances, _ = torch.min(distances, dim=1)  # (B, T)

    collision_violation = torch.clamp(
        safety_distance - min_distances,
        min=0.0
    )  # (B, T)

    collision_metric = collision_violation.mean().item()

    # También útil: porcentaje de timesteps con violación
    collision_rate = (collision_violation > 0.0).float().mean().item()

    return {
        "acceleration": acc_metric,
        "jerk": jerk_metric,
        "steering": steer_metric,
        "steering_change": steer_change_metric,
        "collision": collision_metric,
        "collision_rate": collision_rate,
    }

def segs_ego_to_world(segs_ego, ego_center_world, heading_rad):
    # segs_ego: (M,4) [x1,y1,x2,y2] en ego-frame
    c, s = np.cos(heading_rad), np.sin(heading_rad)

    R = np.array([[c, -s],
                  [s,  c]], dtype=np.float32)

    a = segs_ego[:, 0:2] @ R.T + ego_center_world
    b = segs_ego[:, 2:4] @ R.T + ego_center_world

    return np.concatenate([a, b], axis=1)

def make_current_state(ego, neighbors):
    """
    Return: (B, 1+N, feat_dim_last)
    Usa el último timestep de ego y vecinos (sin flag).
    """
    # ego: (B, T_obs, 6)
    # neighbors: (B, N, T_obs, 7) -> quitamos flag
    return torch.cat([ego.unsqueeze(1), neighbors[..., :-1]], dim=1)[:, :, -1]  # (B, 1+N, 6)


def get_visualization_candidates(raw_dataset, num_vis_samples, min_neighbors):
    neighbors_raw = raw_dataset.neighbors
    valid_counts = np.array([
        int(np.sum(np.sum(np.abs(neighbors_raw[i]), axis=(1, 2)) > 0))
        for i in range(len(raw_dataset))
    ])

    candidates = np.where(valid_counts >= min_neighbors)[0]
    if len(candidates) >= num_vis_samples:
        step = max(1, len(candidates) // num_vis_samples)
        selected = candidates[::step][:num_vis_samples]
    else:
        selected = candidates

    return set(int(x) for x in selected)


def denorm_xy_factory(raw_dataset, raw_idx):
    raw_ego = raw_dataset.ego[raw_idx]
    ego_pos0 = raw_ego[-1, :2].copy()
    vx, vy = float(raw_ego[-1, 2]), float(raw_ego[-1, 3])
    heading = float(np.arctan2(vy, vx)) if abs(vx) > 1e-2 or abs(vy) > 1e-2 else 0.0
    c, s = np.cos(heading), np.sin(heading)

    def denorm_xy(xy):
        xr = xy[..., 0] * c - xy[..., 1] * s
        yr = xy[..., 0] * s + xy[..., 1] * c
        out = np.stack([xr, yr], axis=-1)
        out = out + ego_pos0
        return out

    return denorm_xy


def tensor_to_list(x):
    if isinstance(x, torch.Tensor):
        return x.detach().cpu().numpy().tolist()
    if isinstance(x, np.ndarray):
        return x.tolist()
    return x


def align_neighbor_predictions(pred_tensor, num_neighbors_expected):
    """
    Alinea tensor (B, A, T, 2) para que solo contenga vecinos.
    Si viene el ego incluido como primer agente, lo quita.
    """
    if pred_tensor.shape[1] == num_neighbors_expected:
        return pred_tensor
    if pred_tensor.shape[1] == num_neighbors_expected + 1:
        return pred_tensor[:, 1:]
    raise ValueError(
        f"Predicciones con número inesperado de agentes: "
        f"{pred_tensor.shape[1]} vs vecinos esperados {num_neighbors_expected}"
    )


def align_neighbor_predictions_all(pred_tensor, num_neighbors_expected):
    """
    Alinea tensor multimodo (B, M, A, T, 2) para que solo contenga vecinos.
    Si viene el ego incluido como primer agente, lo quita.
    """
    if pred_tensor.shape[2] == num_neighbors_expected:
        return pred_tensor
    if pred_tensor.shape[2] == num_neighbors_expected + 1:
        return pred_tensor[:, :, 1:]
    raise ValueError(
        f"Predicciones multimodo con número inesperado de agentes: "
        f"{pred_tensor.shape[2]} vs vecinos esperados {num_neighbors_expected}"
    )


def compute_joint_scene_error(plan_trajs, predictions_all, ground_truth):
    """
    plan_trajs:      (B, K, T, 3)
    predictions_all: (B, K, N, T, 2)
    ground_truth:    (B, 1+N, T, D)

    Returns:
        joint_err: (B, K)  -> error conjunto ego + vecinos válidos
    """
    ego_gt = ground_truth[:, 0, :, :2]      # (B, T, 2)
    nei_gt = ground_truth[:, 1:, :, :2]     # (B, N, T, 2)

    valid_mask = torch.sum(torch.abs(nei_gt), dim=(2, 3)) > 0  # (B, N)

    all_joint_err = []
    K = plan_trajs.shape[1]

    for k in range(K):
        ego_pred_k = plan_trajs[:, k, :, :2]                           # (B, T, 2)
        ego_err_k = torch.norm(ego_pred_k - ego_gt, dim=-1).mean(dim=-1)  # (B,)

        nei_pred_k = predictions_all[:, k, :, :, :2]                   # (B, N, T, 2)
        nei_disp_k = torch.norm(nei_pred_k - nei_gt, dim=-1)           # (B, N, T)
        nei_ade_k = nei_disp_k.mean(dim=-1)                            # (B, N)
        nei_ade_k = torch.where(valid_mask, nei_ade_k, torch.zeros_like(nei_ade_k))

        valid_counts = valid_mask.sum(dim=1).clamp(min=1)              # (B,)
        nei_err_k = nei_ade_k.sum(dim=1) / valid_counts                # (B,)

        joint_err_k = ego_err_k + nei_err_k
        all_joint_err.append(joint_err_k)

    joint_err = torch.stack(all_joint_err, dim=1)  # (B, K)
    return joint_err


# =========================================================
# Batch runners
# =========================================================

@torch.no_grad()
def run_batch_standard(model, ego, neighbors, ground_truth=None,
                       map_segments=None, map_mask=None, return_attn: bool = False):
    """
    Predictor normal (multi-modal):
      - sin mapa: model(ego, neighbors)
      - con mapa: model(ego, neighbors, map_segments, map_mask)
    Opcional: return_attn=True si tu Predictor.forward ya devuelve attn_w como 5to output.
    """
    attn_w = None

    if return_attn:
        if map_segments is None or map_mask is None:
            plans, predictions_all, scores, cost_function_weights, attn_w = model(
                ego, neighbors, return_attn=True
            )
        else:
            plans, predictions_all, scores, cost_function_weights, attn_w = model(
                ego, neighbors, map_segments, map_mask, return_attn=True
            )
    else:
        if map_segments is None or map_mask is None:
            plans, predictions_all, scores, cost_function_weights = model(ego, neighbors)
        else:
            plans, predictions_all, scores, cost_function_weights = model(ego, neighbors, map_segments, map_mask)

    num_modes = plans.shape[1]
    num_neighbors_expected = neighbors.shape[1]
    predictions_all = align_neighbor_predictions_all(predictions_all, num_neighbors_expected)

    plan_trajs = torch.stack(
        [bicycle_model(plans[:, k], ego[:, -1])[:, :, :3] for k in range(num_modes)],
        dim=1
    )  # (B,K,T,3)

    out = {
        "mode": "standard",
        "plans": plans,
        "predictions_all": predictions_all,
        "scores": scores,
        "cost_function_weights": cost_function_weights,
        "plan_trajs": plan_trajs,
        "num_modes": num_modes,
        "ground_truth": ground_truth,
    }
    if return_attn:
        out["attn_w"] = attn_w
    return out

def pack_vis_sample_standard_topk(
    b, batch_start, ego, neighbors, ground_truth,
    plan_trajs_all, predictions_all, scores,
    topk=3, raw_dataset=None
):
    ego_hist = ego[b, :, :2].cpu().numpy()
    ego_gt   = ground_truth[b, 0, :, :2].cpu().numpy()

    nb_hist = neighbors[b, :, :, :2].cpu().numpy()
    nb_gt   = ground_truth[b, 1:, :, :2].cpu().numpy()

    neigh_valid = (neighbors[b, :, -1, 6] > 0).cpu().numpy()
    n_valid = int(np.sum([np.sum(np.abs(nb_hist[n])) > 0 for n in range(nb_hist.shape[0])]))

    # top-k por prob/score
    k = min(topk, scores.shape[1])
    topk_idx = torch.topk(scores[b], k=k, dim=0).indices  # (k,)

    ego_samples_trajs = plan_trajs_all[b, topk_idx, :, :2].cpu().numpy()         # (k,T,2)
    neighbor_samples  = predictions_all[b, topk_idx, :, :, :2].cpu().numpy()     # (k,N,T,2)
    # scores: (B, M) logits/scores crudos
    scores_prob = torch.softmax(scores, dim=1)     # (B, M) en [0,1]
    topk_scores = scores_prob[b, topk_idx].detach().cpu().numpy()  # (K,)

    if raw_dataset is not None:
        raw_idx = batch_start + b
        scene_id = None
        if hasattr(raw_dataset, "scene_id") and raw_dataset.scene_id is not None:
            scene_id = int(raw_dataset.scene_id[raw_idx])

        denorm_xy = denorm_xy_factory(raw_dataset, raw_idx)
        ego_hist = denorm_xy(ego_hist)
        ego_gt   = denorm_xy(ego_gt)
        nb_hist  = denorm_xy(nb_hist)
        nb_gt    = denorm_xy(nb_gt)
        ego_samples_trajs = np.stack([denorm_xy(ego_samples_trajs[i]) for i in range(k)], axis=0)
        neighbor_samples  = np.stack([denorm_xy(neighbor_samples[i])  for i in range(k)], axis=0)

    out = {
        "kind": "standard_topk",
        "ego_hist": ego_hist,
        "ego_gt": ego_gt,
        "neighbors_hist": nb_hist,
        "neighbors_gt": nb_gt,
        "neighbors_valid": neigh_valid,
        "num_valid_neighbors": int(n_valid),

        "ego_topk_trajs": ego_samples_trajs,          # (K,T,2)
        "ego_topk_scores": topk_scores,               # (K,)
        "ego_topk_nb_preds": neighbor_samples,        # (K,N,T,2)
    }

    # centro del ego (en world si raw_dataset existe, si no, en el frame actual)
    out["ego_center"] = ego_hist[-1].copy()

    if raw_dataset is not None:
        out["scene_id"] = scene_id  # puede ser None si no existe

    return out

@torch.no_grad()
def run_batch_vae(
    model,
    ego,
    neighbors,
    ground_truth=None,
    num_samples=1,
    vae_prior="fixed",
    map_segments=None,
    map_mask=None,
    return_attn: bool = False,
):
    """
    PredictorVAE batch runner.

    - vae_prior="fixed"   -> z ~ N(0, I)
    - vae_prior="learned" -> z ~ p(z | past) (prior aprendida condicionada al pasado)

    Si return_attn=True y el modelo lo soporta, regresa también attn_w
    (esperado: (B, modes, heads, A, M) o similar dependiendo de tu implementación).
    """

    want_attn = bool(return_attn)

    # ---- Call inference (prefer keyword args, robust to older signatures) ----
    try:
        outputs = model.inference(
            ego,
            neighbors,
            map_segments=map_segments,
            map_mask=map_mask,
            num_samples=num_samples,
            prior=("learned" if vae_prior == "learned" else "fixed"),
            return_attn=want_attn,
        )
    # except TypeError:
    #     # Backward compatibility: model.inference may not accept some kwargs
    #     if vae_prior == "learned" and hasattr(model, "inference_learned_prior"):
    #         outputs = model.inference_learned_prior(ego, neighbors, num_samples=num_samples)
    #     else:
    #         # Try without map + return_attn + prior
    #         outputs = model.inference(ego, neighbors, num_samples=num_samples)
    except TypeError as e:
        print("\n[ERROR] model.inference no aceptó los argumentos con mapa/atención")
        print("return_attn:", want_attn)
        print("map_segments is None?", map_segments is None)
        print("map_mask is None?", map_mask is None)
        print("TypeError:", e)
        raise

    # ---- Unpack outputs ----
    z = None
    attn_w = None

    if isinstance(outputs, (list, tuple)):
        if len(outputs) == 5:
            plans, predictions_all, cost_function_weights, z, attn_w = outputs
        elif len(outputs) == 4:
            plans, predictions_all, cost_function_weights, z = outputs
        elif len(outputs) == 3:
            plans, predictions_all, cost_function_weights = outputs
        else:
            raise ValueError(f"Salida inesperada de PredictorVAE.inference(...): len={len(outputs)}")
    else:
        raise ValueError(f"Salida inesperada de PredictorVAE.inference(...): type={type(outputs)}")

    # ---- Align neighbor predictions (safety) ----
    num_modes = plans.shape[1]
    num_neighbors_expected = neighbors.shape[1]
    predictions_all = align_neighbor_predictions_all(predictions_all, num_neighbors_expected)

    # ---- Convert controls to trajectories ----
    plan_trajs = torch.stack(
        [bicycle_model(plans[:, k], ego[:, -1])[:, :, :3] for k in range(num_modes)],
        dim=1
    )  # (B,K,T,3)

    out = {
        "mode": "vae",
        "vae_prior": vae_prior,
        "plans": plans,
        "predictions_all": predictions_all,
        "cost_function_weights": cost_function_weights,
        "plan_trajs": plan_trajs,
        "z": z,
        "num_modes": num_modes,
        "ground_truth": ground_truth,
    }
    if want_attn:
        out["attn_w"] = attn_w

    return out


# =========================================================
# Planning (UNA sola optimización final)
# =========================================================

@torch.no_grad()
def apply_planning_single(
    plan_control_init,          # (B,T,2)
    prediction_init,            # (B,N,T,2)
    cost_function_weights_init, # (B,5)
    ego,
    current_state,
    planner,
    T=12
):
    """
    Corre Theseus UNA vez con una inicialización (plan_control_init) y un set de predicciones (prediction_init).
    Return refined_plan_traj: (B,T,3)
    """
    B = plan_control_init.shape[0]

    planner_inputs = {
        "control_variables": plan_control_init.view(B, -1),
        "predictions": prediction_init,
        "current_state": current_state,
        "w_acc": cost_function_weights_init[:, 0].unsqueeze(1),
        "w_jerk": cost_function_weights_init[:, 1].unsqueeze(1),
        "w_steer": cost_function_weights_init[:, 2].unsqueeze(1),
        "w_steer_change": cost_function_weights_init[:, 3].unsqueeze(1),
        "w_collision": cost_function_weights_init[:, 4].unsqueeze(1),
        "w_speed_limit":  cost_function_weights_init[:, 5].unsqueeze(1),
    }

    final_values, info = planner.layer.forward(planner_inputs)
    refined_control = final_values["control_variables"].view(B, T, 2)

    refined_plan_traj = bicycle_model(refined_control, ego[:, -1], dt=0.4)[:, :, :3]  # (B,T,3)
    return refined_plan_traj, refined_control


def select_vae_candidate_before_planning(
    model_out, ground_truth, selection_mode="best_of_k"
):
    """
    Selecciona UNA trayectoria/predicción a partir de las K muestras del VAE ANTES de planificar.
    Devuelve:
      plan_control_init: (B,T,2)
      prediction_init:   (B,N,T,2)
      weights_init:      (B,5)
      selected_idx:      (B,) o None
      aux_for_vis:       dict con info extra (por si quieres visualizar K muestras)
    """
    plans = model_out["plans"]                       # (B,K,T,2)
    predictions_all = model_out["predictions_all"]   # (B,K,N,T,2)
    plan_trajs = model_out["plan_trajs"]             # (B,K,T,3) sin planner
    w = model_out["cost_function_weights"]           # (B,5) o (B,K,5)

    B, K = plans.shape[0], plans.shape[1]
    batch_ids = torch.arange(B, device=plans.device)

    if selection_mode == "mean_of_k":
        plan_control_init = plans.mean(dim=1)             # (B,T,2)
        prediction_init = predictions_all.mean(dim=1)     # (B,N,T,2)
        weights_init = w.mean(dim=1) if w.dim() == 3 else w
        selected_idx = None

    elif selection_mode == "first":
        selected_idx = torch.zeros(B, dtype=torch.long, device=plans.device)
        plan_control_init = plans[:, 0]
        prediction_init = predictions_all[:, 0]
        weights_init = w[:, 0] if w.dim() == 3 else w

    else:  # "best_of_k" (oracle con GT)
        if ground_truth is None:
            raise ValueError("best_of_k requiere ground_truth para seleccionar k*.")

        joint_err = compute_joint_scene_error(plan_trajs, predictions_all, ground_truth)  # (B,K)
        selected_idx = torch.argmin(joint_err, dim=1)                                      # (B,)

        plan_control_init = plans[batch_ids, selected_idx]         # (B,T,2)
        prediction_init = predictions_all[batch_ids, selected_idx] # (B,N,T,2)
        weights_init = (w[batch_ids, selected_idx] if w.dim() == 3 else w)  # (B,5)

    aux_for_vis = {
        "plan_trajs_all": plan_trajs,            # (B,K,T,3)
        "predictions_all": predictions_all,      # (B,K,N,T,2)
        "selected_idx": selected_idx
    }
    return plan_control_init, prediction_init, weights_init, selected_idx, aux_for_vis


# =========================================================
# Visualization packers
# =========================================================

def pack_vis_sample_vae_final_only(
    b, batch_start, ego, neighbors, ground_truth,
    final_plan_traj, final_prediction,
    raw_dataset=None
):
    ego_hist = ego[b, :, :2].cpu().numpy()
    ego_gt = ground_truth[b, 0, :, :2].cpu().numpy()

    nb_hist = neighbors[b, :, :, :2].cpu().numpy()
    nb_gt = ground_truth[b, 1:, :, :2].cpu().numpy()

    neigh_valid = (neighbors[b, :, -1, 6] > 0).cpu().numpy()
    n_valid = int(np.sum([np.sum(np.abs(nb_hist[n])) > 0 for n in range(nb_hist.shape[0])]))

    ego_pred = final_plan_traj[b, :, :2].cpu().numpy()
    nb_pred = final_prediction[b, :, :, :2].cpu().numpy()

    scene_id = None
    if raw_dataset is not None:
        raw_idx = batch_start + b
        if hasattr(raw_dataset, "scene_id") and raw_dataset.scene_id is not None:
            scene_id = int(raw_dataset.scene_id[raw_idx])

        denorm_xy = denorm_xy_factory(raw_dataset, raw_idx)
        ego_hist = denorm_xy(ego_hist)
        ego_gt = denorm_xy(ego_gt)
        nb_hist = denorm_xy(nb_hist)
        nb_gt = denorm_xy(nb_gt)
        ego_pred = denorm_xy(ego_pred)
        nb_pred = denorm_xy(nb_pred)

    out = {
        "kind": "vae_final",
        "ego_hist": ego_hist,
        "ego_gt": ego_gt,
        "ego_pred": ego_pred,
        "neighbors_hist": nb_hist,
        "neighbors_gt": nb_gt,
        "neighbors_pred": nb_pred,
        "neighbors_valid": neigh_valid,
        "num_valid_neighbors": int(n_valid),
        "ego_center": ego_hist[-1].copy(),
    }

    if scene_id is not None:
        out["scene_id"] = scene_id

    return out


def pack_vis_sample_vae_multi_optional(
    b, batch_start, ego, neighbors, ground_truth,
    plan_trajs_all, predictions_all, selected_idx=None,
    raw_dataset=None
):
    """
    Guarda K muestras del VAE en una sola escena (solo visualización).
    """
    ego_hist = ego[b, :, :2].cpu().numpy()
    ego_gt = ground_truth[b, 0, :, :2].cpu().numpy()

    nb_hist = neighbors[b, :, :, :2].cpu().numpy()
    nb_gt = ground_truth[b, 1:, :, :2].cpu().numpy()

    neigh_valid = (neighbors[b, :, -1, 6] > 0).cpu().numpy()
    n_valid = int(np.sum([np.sum(np.abs(nb_hist[n])) > 0 for n in range(nb_hist.shape[0])]))

    ego_samples_trajs = plan_trajs_all[b, :, :, :2].cpu().numpy()            # (K,T,2)
    neighbor_samples_preds = predictions_all[b, :, :, :, :2].cpu().numpy()   # (K,N,T,2)

    scene_id = None
    if raw_dataset is not None:
        raw_idx = batch_start + b

        # escena para obstacles.txt
        if hasattr(raw_dataset, "scene_id") and raw_dataset.scene_id is not None:
            scene_id = int(raw_dataset.scene_id[raw_idx])

        denorm_xy = denorm_xy_factory(raw_dataset, raw_idx)

        ego_hist = denorm_xy(ego_hist)
        ego_gt = denorm_xy(ego_gt)
        nb_hist = denorm_xy(nb_hist)
        nb_gt = denorm_xy(nb_gt)

        ego_samples_trajs = np.stack(
            [denorm_xy(ego_samples_trajs[k]) for k in range(ego_samples_trajs.shape[0])], axis=0
        )
        neighbor_samples_preds = np.stack(
            [denorm_xy(neighbor_samples_preds[k]) for k in range(neighbor_samples_preds.shape[0])], axis=0
        )

    out = {
        "kind": "vae_multi",
        "ego_hist": ego_hist,
        "ego_gt": ego_gt,
        "neighbors_hist": nb_hist,
        "neighbors_gt": nb_gt,
        "neighbors_valid": neigh_valid,
        "num_valid_neighbors": int(n_valid),
        "ego_samples_trajs": ego_samples_trajs,
        "neighbor_samples_preds": neighbor_samples_preds,
        "ego_center": ego_hist[-1].copy(),   # para círculo en visualization.py
    }

    if scene_id is not None:
        out["scene_id"] = scene_id

    if selected_idx is not None:
        out["selected_idx"] = int(selected_idx[b].item())

    return out


# =========================================================
# Evaluators
# =========================================================

@torch.no_grad()
def evaluate_model_standard(
    model, data_loader, device,
    use_planning=False, planner=None,
    save_vis=False, num_vis_samples=10,
    min_neighbors=0, raw_dataset=None,
    save_attn: bool = False, 
):
    model.eval()

    eff_accs = []
    eff_jerks = []
    eff_steers = []
    eff_steer_changes = []
    eff_collisions = []
    eff_collision_rates = []
    ego_ades, ego_fdes = [], []
    neighbor_ades, neighbor_fdes = [], []
    vis_samples = []

    vis_candidate_indices = set()
    if save_vis and raw_dataset is not None:
        vis_candidate_indices = get_visualization_candidates(raw_dataset, num_vis_samples, min_neighbors)

    for batch_idx, batch in enumerate(tqdm(data_loader, desc="Evaluating standard predictor")):
        ego = batch[0].to(device)
        neighbors = batch[1].to(device)
        ground_truth = batch[2].to(device)
        map_segments = None
        map_mask = None
        ego_center_batch = None
        heading_batch = None

        if len(batch) >= 5:
            map_segments = batch[3].to(device)
            map_mask = batch[4].to(device)

        if len(batch) >= 7:
            ego_center_batch = batch[5].to(device)  # (B,2)
            heading_batch = batch[6].to(device)     # (B,1)

        current_state = make_current_state(ego, neighbors)

        model_out = run_batch_standard(
            model=model,
            ego=ego,
            neighbors=neighbors,
            ground_truth=ground_truth,
            map_segments=map_segments,
            map_mask=map_mask,
            return_attn=save_attn,
            )
        # Elegir modo final: argmax de scores
        batch_ids = torch.arange(ego.shape[0], device=device)
        best_mode_idx = torch.argmax(model_out["scores"], dim=1)  # (B,)

        # Trayectoria y predicción del modo seleccionado ANTES del planner
        final_plan_traj = model_out["plan_trajs"][batch_ids, best_mode_idx]          # (B,T,3)
        final_prediction = model_out["predictions_all"][batch_ids, best_mode_idx]    # (B,N,T,2)

        # Control inicial del modo seleccionado
        plan_control_init = model_out["plans"][batch_ids, best_mode_idx]             # (B,T,2)

        # Por defecto, si NO hay planning, el control final es el control inicial
        final_control = plan_control_init

        # Si hay planning, Theseus refina el control
        if use_planning and planner is not None:

            w = model_out["cost_function_weights"]

            if w.dim() == 3 and w.shape[1] == 1:
                w = w[:, 0]

            final_plan_traj, final_control = apply_planning_single(
                plan_control_init=plan_control_init,
                prediction_init=final_prediction,
                cost_function_weights_init=w,
                ego=ego,
                current_state=current_state,
                planner=planner,
                T=12
            )

        # =========================
        # Efficiency / safety metrics
        # =========================
        eff = compute_efficiency_metrics(
            ego_traj=final_plan_traj,
            control=final_control,
            neighbor_pred=final_prediction,
            safety_distance=0.5,
            dt=0.4,
        )

        eff_accs.append(eff["acceleration"])
        eff_jerks.append(eff["jerk"])
        eff_steers.append(eff["steering"])
        eff_steer_changes.append(eff["steering_change"])
        eff_collisions.append(eff["collision"])
        eff_collision_rates.append(eff["collision_rate"])

        # Metrics - ego
        ego_gt = ground_truth[:, 0, :, :2]
        ego_pred = final_plan_traj[:, :, :2]
        ade, fde = compute_ade_fde(ego_pred, ego_gt)
        ego_ades.append(ade)
        ego_fdes.append(fde)

        # Metrics - neighbors
        neighbor_gt = ground_truth[:, 1:, :, :2]
        neighbor_pred = final_prediction[:, :, :, :2]
        valid_mask = torch.sum(torch.abs(neighbor_gt), dim=(2, 3)) > 0

        for bb in range(neighbor_gt.shape[0]):
            for n in range(neighbor_gt.shape[1]):
                if valid_mask[bb, n]:
                    ade, fde = compute_ade_fde(
                        neighbor_pred[bb, n:n+1],
                        neighbor_gt[bb, n:n+1]
                    )
                    neighbor_ades.append(ade)
                    neighbor_fdes.append(fde)

        # Visualizations
        # (tu visual standard ya la tienes, no la toco aquí para no alargar)
        # Si quieres también te la ajusto a "final only", pero lo principal era VAE.
        if save_vis and len(vis_samples) < num_vis_samples:
            batch_start = batch_idx * data_loader.batch_size
            for bb in range(ego.shape[0]):
                if len(vis_samples) >= num_vis_samples:
                    break

                global_idx = batch_start + bb
                if raw_dataset is not None and global_idx not in vis_candidate_indices:
                    continue

                sample_multi = pack_vis_sample_standard_topk(
                    b=bb,
                    batch_start=batch_start,
                    ego=ego,
                    neighbors=neighbors,
                    ground_truth=ground_truth,
                    plan_trajs_all=model_out["plan_trajs"],
                    predictions_all=model_out["predictions_all"],
                    scores=model_out["scores"],
                    topk=3,
                    raw_dataset=raw_dataset
                )
                # ---- Adjuntar mapa local ----
                if map_segments is not None and map_mask is not None:
                    # ---- Adjuntar mapa local (segmentos) en WORLD ----
                    if map_segments is not None and map_mask is not None and raw_dataset is not None:
                        seg_ego = map_segments[bb].detach().cpu().numpy().astype(np.float32)

                        ego_center_world = ego_center_batch[bb].detach().cpu().numpy().astype(np.float32)
                        heading = float(heading_batch[bb].detach().cpu().item())

                        seg_world = segs_ego_to_world(seg_ego, ego_center_world, heading)

                        sample_multi["map_segments"] = seg_world
                        sample_multi["map_mask"]     = map_mask[bb].detach().cpu().numpy()
                    elif map_segments is not None and map_mask is not None:
                        # fallback: si no hay raw_dataset, dejamos en el frame actual
                        sample_multi["map_segments"] = map_segments[bb].detach().cpu().numpy()
                        sample_multi["map_mask"]     = map_mask[bb].detach().cpu().numpy()
                # ---- Adjuntar atención (solo ego) si existe ----
                attn_w = model_out.get("attn_w", None)   # esperado: (B, modes, heads, A, M)
                if attn_w is not None:
                    aw = attn_w[bb]               # (modes, heads, A, M)
                    aw = aw.mean(dim=0)           # promedio modos -> (heads, A, M)
                    aw = aw.mean(dim=0)           # promedio heads -> (A, M)
                    attn_ego = aw[0]              # ego idx 0 -> (M,)
                    attn_ego = attn_ego / (attn_ego.max() + 1e-8)
                    sample_multi["attn_ego"] = attn_ego.detach().cpu().numpy()
                vis_samples.append(sample_multi)
    return {
        "predictor_type": "standard",
        "ego_ADE": np.mean(ego_ades) if ego_ades else 0.0,
        "ego_FDE": np.mean(ego_fdes) if ego_fdes else 0.0,
        "neighbor_ADE": np.mean(neighbor_ades) if neighbor_ades else 0.0,
        "neighbor_FDE": np.mean(neighbor_fdes) if neighbor_fdes else 0.0,
        "acceleration": np.mean(eff_accs) if eff_accs else 0.0,
        "jerk": np.mean(eff_jerks) if eff_jerks else 0.0,
        "steering": np.mean(eff_steers) if eff_steers else 0.0,
        "steering_change": np.mean(eff_steer_changes) if eff_steer_changes else 0.0,
        "collision": np.mean(eff_collisions) if eff_collisions else 0.0,
        "collision_rate": np.mean(eff_collision_rates) if eff_collision_rates else 0.0,
        "num_samples": len(ego_ades),
        "num_neighbors_evaluated": len(neighbor_ades),
        "vis_samples": vis_samples,
    }


@torch.no_grad()
def evaluate_model_vae(
    model, data_loader, device,
    use_planning=False, planner=None,
    save_vis=False, num_vis_samples=10,
    min_neighbors=0, raw_dataset=None,
    vae_prior="fixed",
    vae_num_samples=15,
    vae_select_mode="best_of_k",      # best_of_k | mean_of_k | first
    vae_vis_mode="final",
    save_attn: bool = False,              # final | multi | both
):
    """
    Nuevo flujo VAE:
    1) sample K (plans, neighbors)
    2) selecciona UNA (best_of_k) o promedio (mean_of_k)
    3) planner optimiza UNA vez
    4) visualiza una escena final (y opcionalmente también las K muestras)
    """
    model.eval()

    eff_accs = []
    eff_jerks = []
    eff_steers = []
    eff_steer_changes = []
    eff_collisions = []
    eff_collision_rates = []
    ego_ades, ego_fdes = [], []
    neighbor_ades, neighbor_fdes = [], []
    vis_samples = []

    vis_candidate_indices = set()
    if save_vis and raw_dataset is not None:
        vis_candidate_indices = get_visualization_candidates(raw_dataset, num_vis_samples, min_neighbors)

    for batch_idx, batch in enumerate(tqdm(data_loader, desc="Evaluating VAE predictor")):
        ego = batch[0].to(device)
        neighbors = batch[1].to(device)
        ground_truth = batch[2].to(device)

        map_segments = None
        map_mask = None
        ego_center_batch = None
        heading_batch = None

        if len(batch) >= 5:
            map_segments = batch[3].to(device)
            map_mask = batch[4].to(device)

        if len(batch) >= 7:
            ego_center_batch = batch[5].to(device)  # (B,2)
            heading_batch = batch[6].to(device)     # (B,1)

        current_state = make_current_state(ego, neighbors)

        # 1) Genera K muestras
        model_out = run_batch_vae(
            model=model,
            ego=ego,
            neighbors=neighbors,
            ground_truth=ground_truth,
            num_samples=vae_num_samples,
            vae_prior=vae_prior,
            map_segments=map_segments,
            map_mask=map_mask,
            return_attn=save_attn,
            )

        # 2) Selección antes del planner
        plan_control_init, prediction_init, weights_init, selected_idx, aux_vis = \
            select_vae_candidate_before_planning(
                model_out=model_out,
                ground_truth=ground_truth,
                selection_mode=vae_select_mode
            )

        # 3) Planner: UNA sola optimización (si use_planning)
        if use_planning and planner is not None:
            final_plan_traj, final_control = apply_planning_single(
                plan_control_init=plan_control_init,
                prediction_init=prediction_init,
                cost_function_weights_init=weights_init,
                ego=ego,
                current_state=current_state,
                planner=planner,
                T=12
            )
        else:
            final_control = plan_control_init
            final_plan_traj = bicycle_model(final_control, ego[:, -1])[:, :, :3]

        final_prediction = prediction_init  # vecinos (seleccionados o promedio)

        eff = compute_efficiency_metrics(
            ego_traj=final_plan_traj,
            control=final_control,
            neighbor_pred=final_prediction,
            safety_distance=0.5,
            dt=0.4,
        )

        eff_accs.append(eff["acceleration"])
        eff_jerks.append(eff["jerk"])
        eff_steers.append(eff["steering"])
        eff_steer_changes.append(eff["steering_change"])
        eff_collisions.append(eff["collision"])
        eff_collision_rates.append(eff["collision_rate"])

        # 4) Metrics - ego
        ego_gt = ground_truth[:, 0, :, :2]
        ego_pred = final_plan_traj[:, :, :2]
        ade, fde = compute_ade_fde(ego_pred, ego_gt)
        ego_ades.append(ade)
        ego_fdes.append(fde)

        # 5) Metrics - neighbors
        neighbor_gt = ground_truth[:, 1:, :, :2]
        neighbor_pred = final_prediction[:, :, :, :2]
        valid_mask = torch.sum(torch.abs(neighbor_gt), dim=(2, 3)) > 0

        for bb in range(neighbor_gt.shape[0]):
            for n in range(neighbor_gt.shape[1]):
                if valid_mask[bb, n]:
                    ade, fde = compute_ade_fde(
                        neighbor_pred[bb, n:n+1],
                        neighbor_gt[bb, n:n+1]
                    )
                    neighbor_ades.append(ade)
                    neighbor_fdes.append(fde)

        # 6) Visualizations
        if save_vis and len(vis_samples) < num_vis_samples:
            batch_start = batch_idx * data_loader.batch_size
            for bb in range(ego.shape[0]):
                if len(vis_samples) >= num_vis_samples:
                    break

                global_idx = batch_start + bb
                if global_idx not in vis_candidate_indices:
                    continue

                if vae_vis_mode in ("final", "both"):
                    sample_final = pack_vis_sample_vae_final_only(
                        b=bb,
                        batch_start=batch_start,
                        ego=ego,
                        neighbors=neighbors,
                        ground_truth=ground_truth,
                        final_plan_traj=final_plan_traj,
                        final_prediction=final_prediction,
                        raw_dataset=raw_dataset
                    )
                    # útil para saber qué modo usaste
                    sample_final["vae_select_mode"] = vae_select_mode
                    if selected_idx is not None:
                        sample_final["selected_idx"] = int(selected_idx[bb].item())
                    # ---- Adjuntar mapa local (segmentos) ----
                    if (
                        map_segments is not None
                        and map_mask is not None
                        and raw_dataset is not None
                        and ego_center_batch is not None
                        and heading_batch is not None
                    ):
                        seg_ego = map_segments[bb].detach().cpu().numpy().astype(np.float32)

                        ego_center_world = ego_center_batch[bb].detach().cpu().numpy().astype(np.float32)
                        heading = float(heading_batch[bb].detach().cpu().item())

                        seg_world = segs_ego_to_world(seg_ego, ego_center_world, heading)

                        sample_final["map_segments"] = seg_world
                        sample_final["map_mask"] = map_mask[bb].detach().cpu().numpy()

                    elif map_segments is not None and map_mask is not None:
                        sample_final["map_segments"] = map_segments[bb].detach().cpu().numpy()
                        sample_final["map_mask"] = map_mask[bb].detach().cpu().numpy()
                    # ---- Adjuntar atención (solo ego) ----
                    attn_w = model_out.get("attn_w", None)  # esperado: (B, modes, heads, A, M)
                    if attn_w is not None:
                        aw = attn_w[bb]               # (modes, heads, A, M)
                        aw = aw.mean(dim=0)           # promedio modos -> (heads, A, M)
                        aw = aw.mean(dim=0)           # promedio heads -> (A, M)
                        attn_ego = aw[0]              # ego idx 0 -> (M,)
                        attn_ego = attn_ego / (attn_ego.max() + 1e-8)
                        sample_final["attn_ego"] = attn_ego.detach().cpu().numpy()

                        if batch_idx == 0 and bb == 0:
                            print("\n========== DEBUG HEATMAP ==========")
                            print("dataset:", getattr(raw_dataset, "dataset", None))
                            print("scene_id:", sample_final.get("scene_id", None))
                            print("sample keys:", list(sample_final.keys()))

                            print("map_segments in sample?", "map_segments" in sample_final)
                            print("map_mask in sample?", "map_mask" in sample_final)

                            if "map_segments" in sample_final:
                                ms = np.asarray(sample_final["map_segments"])
                                print("map_segments shape:", ms.shape)
                                print("map_segments min/max:", float(np.nanmin(ms)), float(np.nanmax(ms)))

                            if "map_mask" in sample_final:
                                mm = np.asarray(sample_final["map_mask"]).astype(bool)
                                print("map_mask shape:", mm.shape)
                                print("map_mask sum:", int(mm.sum()))

                            print("attn_w is None?", attn_w is None)

                            if attn_w is not None:
                                print("attn_w shape:", tuple(attn_w.shape))
                                print("attn_w min/max:", float(attn_w.min()), float(attn_w.max()))

                            print("attn_ego in sample?", "attn_ego" in sample_final)

                            if "attn_ego" in sample_final:
                                ae = np.asarray(sample_final["attn_ego"])
                                print("attn_ego shape:", ae.shape)
                                print("attn_ego min/max:", float(ae.min()), float(ae.max()))
                                print("attn_ego std:", float(ae.std()))

                            print("===================================\n")

                    vis_samples.append(sample_final)

                if vae_vis_mode in ("multi", "both") and len(vis_samples) < num_vis_samples:
                    sample_multi = pack_vis_sample_vae_multi_optional(
                        b=bb,
                        batch_start=batch_start,
                        ego=ego,
                        neighbors=neighbors,
                        ground_truth=ground_truth,
                        plan_trajs_all=aux_vis["plan_trajs_all"],
                        predictions_all=aux_vis["predictions_all"],
                        selected_idx=aux_vis["selected_idx"],
                        raw_dataset=raw_dataset
                    )
                    sample_multi["vae_select_mode"] = vae_select_mode

                    # ---- Adjuntar mapa local (segmentos) ----
                    if map_segments is not None and map_mask is not None:
                        if raw_dataset is not None and ego_center_batch is not None and heading_batch is not None:
                            seg_ego = map_segments[bb].detach().cpu().numpy().astype(np.float32)

                            ego_center_world = ego_center_batch[bb].detach().cpu().numpy().astype(np.float32)
                            heading = float(heading_batch[bb].detach().cpu().item())

                            seg_world = segs_ego_to_world(seg_ego, ego_center_world, heading)

                            sample_multi["map_segments"] = seg_world
                            sample_multi["map_mask"] = map_mask[bb].detach().cpu().numpy()

                        else:
                            sample_multi["map_segments"] = map_segments[bb].detach().cpu().numpy()
                            sample_multi["map_mask"] = map_mask[bb].detach().cpu().numpy()

                    # ---- Adjuntar atención (solo ego) ----
                    attn_w = model_out.get("attn_w", None)
                    if attn_w is not None:
                        aw = attn_w[bb].mean(dim=0).mean(dim=0)  # (A,M) promedio modos+heads
                        attn_ego = aw[0]
                        attn_ego = attn_ego / (attn_ego.max() + 1e-8)
                        sample_multi["attn_ego"] = attn_ego.detach().cpu().numpy()
                    vis_samples.append(sample_multi)

    return {
        "predictor_type": "vae",
        "ego_ADE": np.mean(ego_ades) if ego_ades else 0.0,
        "ego_FDE": np.mean(ego_fdes) if ego_fdes else 0.0,
        "neighbor_ADE": np.mean(neighbor_ades) if neighbor_ades else 0.0,
        "neighbor_FDE": np.mean(neighbor_fdes) if neighbor_fdes else 0.0,
        "acceleration": np.mean(eff_accs) if eff_accs else 0.0,
        "jerk": np.mean(eff_jerks) if eff_jerks else 0.0,
        "steering": np.mean(eff_steers) if eff_steers else 0.0,
        "steering_change": np.mean(eff_steer_changes) if eff_steer_changes else 0.0,
        "collision": np.mean(eff_collisions) if eff_collisions else 0.0,
        "collision_rate": np.mean(eff_collision_rates) if eff_collision_rates else 0.0,
        "num_samples": len(ego_ades),
        "num_neighbors_evaluated": len(neighbor_ades),
        "vis_samples": vis_samples,
    }


# =========================================================
# Main
# =========================================================

def main():
    parser = argparse.ArgumentParser(description="Test ETH/UCY")
    parser.add_argument("--model_path", type=str, required=True)
    parser.add_argument("--test_set", type=str, required=True)
    parser.add_argument("--name", type=str, default="test_result")
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--use_planning", action="store_true")
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--num_vis_samples", type=int, default=20)
    parser.add_argument("--min_neighbors", type=int, default=5)
    parser.add_argument("--predictor", type=str, default="Predictor", choices=["Predictor", "PredictorVAE"])
    parser.add_argument("--save_vis", action="store_true", help="Guardar imágenes de visualización")
    parser.add_argument("--max_speed", type=float, default=1, help="Velocidad maxima permitida para el ego en m/s")

    # VAE
    parser.add_argument("--vae_num_samples", type=int, default=5, help="Número de muestras z para PredictorVAE en inferencia")
    parser.add_argument("--vae_select_mode", type=str, default="best_of_k", choices=["first", "best_of_k", "mean_of_k"], help="tipo de selección")
    parser.add_argument("--vae_vis_mode", type=str, default="final", choices=["final", "multi", "both"], help="Qué guardar/visualizar: final=una escena, multi=K muestras, both=ambas")
    parser.add_argument("--vae_prior", type=str, default="fixed", choices=["fixed", "learned"], help="fixed: z~N(0,I). learned: z~p(z|past) aprendida.")
    
    # MAP
    parser.add_argument("--use_map", action="store_true", help="Usar mapa (segmentos locales)")
    parser.add_argument("--map_root", type=str, default="mapa")
    parser.add_argument("--map_radius", type=float, default=7.0)
    parser.add_argument("--map_max_segments", type=int, default=64)
    parser.add_argument("--map_prefilter_margin", type=float, default=1.0)

    parser.add_argument(
    "--dataset",
    type=str,
    default="eth_ucy",
    choices=["eth_ucy", "thor_magni"],
    help="Dataset usado para mapear scene_id a nombres de carpetas de mapa."
    )

    # Interpretation
    parser.add_argument("--save_attn", action="store_true", help="Guardar/usar pesos de atención agente->segmento (solo ego) para visualización.")
    args = parser.parse_args()


    log_path = f"./testing_log/{args.name}/"
    os.makedirs(log_path, exist_ok=True)

    logging.basicConfig(
        level=logging.INFO,
        format='[%(levelname)s %(asctime)s] %(message)s',
        handlers=[
            logging.FileHandler(f"{log_path}/test.log"),
            logging.StreamHandler()
        ]
    )

    logging.info("=" * 60)
    logging.info(f"Testing: {args.name}")
    logging.info(f"Model: {args.model_path}")
    logging.info(f"Test set: {args.test_set}")
    logging.info(f"Use planning: {args.use_planning}")
    logging.info(f"Predictor: {args.predictor}")
    logging.info(f"Device: {args.device}")
    if args.predictor == "PredictorVAE":
        logging.info(f"VAE num samples: {args.vae_num_samples}")
        logging.info(f"VAE select mode: {args.vae_select_mode}")
        logging.info(f"VAE vis mode: {args.vae_vis_mode}")
    logging.info("=" * 60)

    if args.predictor == "PredictorVAE":
        logging.info(f"VAE prior: {args.vae_prior}")
        logging.info(f"Use map in model: {args.use_map}")

        model = PredictorVAE(
            12,
            use_map=args.use_map,
            use_prior=(args.vae_prior == "learned")
        ).to(args.device)
    else:
        model = Predictor(
            12,
            use_map=args.use_map
        ).to(args.device)

    logging.info("Loading model...")
    state_dict = torch.load(
    args.model_path,
    map_location=args.device,
    weights_only=True
)

    model.load_state_dict(state_dict)
    model.eval()
    logging.info("Model loaded successfully!")

    planner = None
    if args.use_planning:
        logging.info("Setting up planner...")
        # IMPORTANT: usa dt=0.4 para ETH/UCY (tu dataset)
        planner = MotionPlanner(12, args.device, test=True, dt=0.4, safety_distance=0.5, max_speed=args.max_speed)
        logging.info("Planner ready!")

    logging.info("Loading test data...")
    test_set = DrivingData(
    args.test_set,
    ego_frame=True,
    use_map=args.use_map,
    map_root=args.map_root,
    map_radius=args.map_radius,
    map_max_segments=args.map_max_segments,
    prefilter_margin=args.map_prefilter_margin,
    dataset=args.dataset,
    )
    test_loader = DataLoader(test_set, batch_size=args.batch_size, shuffle=False, num_workers=4)
    logging.info(f"Test set: {len(test_set)} samples")

    logging.info("=" * 60)
    logging.info("Starting evaluation...")
    logging.info("=" * 60)

    if args.predictor == "PredictorVAE":
        results = evaluate_model_vae(
            model=model,
            data_loader=test_loader,
            device=args.device,
            use_planning=args.use_planning,
            planner=planner,
            save_vis=args.save_vis,
            num_vis_samples=args.num_vis_samples,
            min_neighbors=args.min_neighbors,
            raw_dataset=test_set,
            vae_prior=args.vae_prior,
            vae_num_samples=args.vae_num_samples,
            vae_select_mode=args.vae_select_mode,
            vae_vis_mode=args.vae_vis_mode,
            save_attn=args.save_attn,
        )
    else:
        results = evaluate_model_standard(
            model=model,
            data_loader=test_loader,
            device=args.device,
            use_planning=args.use_planning,
            planner=planner,
            save_vis=args.save_vis,
            num_vis_samples=args.num_vis_samples,
            min_neighbors=args.min_neighbors,
            raw_dataset=test_set,
            save_attn=args.save_attn,
        )

    logging.info("=" * 60)
    logging.info("RESULTS")
    logging.info("=" * 60)
    logging.info("Ego Agent:")
    logging.info(f"  ADE: {results['ego_ADE']:.4f} m")
    logging.info(f"  FDE: {results['ego_FDE']:.4f} m")
    logging.info("Neighbor Agents:")
    logging.info(f"  ADE: {results['neighbor_ADE']:.4f} m")
    logging.info(f"  FDE: {results['neighbor_FDE']:.4f} m")
    logging.info("Statistics:")
    logging.info(f"  Total samples: {results['num_samples']}")
    logging.info(f"  Neighbors evaluated: {results['num_neighbors_evaluated']}")
    logging.info("=" * 60)
    logging.info("Efficiency / Comfort / Safety:")
    logging.info(f"  Acceleration:     {results['acceleration']:.4f}")
    logging.info(f"  Jerk:             {results['jerk']:.4f}")
    logging.info(f"  Steering:         {results['steering']:.4f}")
    logging.info(f"  Steering Change:  {results['steering_change']:.4f}")
    logging.info(f"  Collision:        {results['collision']:.4f}")
    logging.info(f"  Collision Rate:   {results['collision_rate']:.4f}")

    results_file = f"{log_path}/results.txt"
    with open(results_file, "w") as f:
        f.write(f"Model: {args.model_path}\n")
        f.write(f"Test set: {args.test_set}\n")
        f.write(f"Use planning: {args.use_planning}\n")
        f.write(f"Predictor: {args.predictor}\n")
        f.write(f"Dataset: {args.dataset}\n")
        f.write(f"Use map: {args.use_map}\n")
        f.write(f"Map root: {args.map_root}\n")
        if args.predictor == "PredictorVAE":
            f.write(f"VAE prior: {args.vae_prior}\n")
            f.write(f"VAE num samples: {args.vae_num_samples}\n")
            f.write(f"VAE select mode: {args.vae_select_mode}\n")
            f.write(f"VAE vis mode: {args.vae_vis_mode}\n")
        f.write("\n")
        f.write(f"Ego ADE: {results['ego_ADE']:.4f} m\n")
        f.write(f"Ego FDE: {results['ego_FDE']:.4f} m\n")
        f.write(f"Neighbor ADE: {results['neighbor_ADE']:.4f} m\n")
        f.write(f"Neighbor FDE: {results['neighbor_FDE']:.4f} m\n")
        f.write(f"Total samples: {results['num_samples']}\n")
        f.write(f"Neighbors evaluated: {results['num_neighbors_evaluated']}\n")
        f.write("\n")
        f.write(f"Acceleration: {results['acceleration']:.4f}\n")
        f.write(f"Jerk: {results['jerk']:.4f}\n")
        f.write(f"Steering: {results['steering']:.4f}\n")
        f.write(f"Steering Change: {results['steering_change']:.4f}\n")
        f.write(f"Collision: {results['collision']:.4f}\n")
        f.write(f"Collision Rate: {results['collision_rate']:.4f}\n")

    logging.info(f"Results saved to: {results_file}")

    # Visualizations
    if args.save_vis and len(results["vis_samples"]) > 0:
        vis_path = f"{log_path}/visualizations"
        save_visualizations_from_samples(
            samples_data=results["vis_samples"],
            save_path=vis_path,
            predictor_type=results["predictor_type"],
            map_root=args.map_root,
            map_radius=args.map_radius,
            dataset=args.dataset,
)
        logging.info(f"Visualizations saved to: {vis_path}")


if __name__ == "__main__":
    main()