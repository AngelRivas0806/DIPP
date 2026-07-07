import os
from random import sample
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Circle


ETH_UCY_DATASETS = [
    "eth-hotel",
    "eth-univ",
    "ucy-zara01",
    "ucy-zara02",
    "ucy-univ",
]

THOR_MAGNI_DATASETS = [
    "THOR_MAGNI_120522_SC3",
    "THOR_MAGNI_130522_SC3",
    "THOR_MAGNI_170522_SC3",
    "THOR_MAGNI_180522_SC3",
]

def get_scene_names(dataset):
    dataset = dataset.lower()

    if dataset in ["eth_ucy", "ethucy", "eth-ucy"]:
        return ETH_UCY_DATASETS

    if dataset in ["thor", "thor_magni", "thor-magni"]:
        return THOR_MAGNI_DATASETS

    raise ValueError(f"Dataset no soportado: {dataset}")


def _ensure_dir(path: str):
    os.makedirs(path, exist_ok=True)


# =========================================================
# Obstacles + circle + segments overlays (opcional)
# =========================================================
def _load_obstacles_polys(map_root: str, scene_id: int, dataset: str = "eth_ucy"):
    """
    Carga obstáculos como lista de polilíneas desde:
        map_root/<scene_name>/obstacles.txt
    """
    if map_root is None:
        return []
    if scene_id is None:
        return []

    scene_names = get_scene_names(dataset)

    if scene_id < 0 or scene_id >= len(scene_names):
        return []

    scene_name = scene_names[int(scene_id)]
    path = os.path.join(map_root, scene_name, "obstacles.txt")

    if not os.path.exists(path):
        print(f"[WARN visualization] No existe mapa: {path}")
        return []

    polys = []
    cur = []

    def flush():
        nonlocal cur
        if len(cur) >= 3:
            polys.append(np.asarray(cur, dtype=np.float32))
        cur = []

    with open(path, "r") as f:
        for raw in f:
            s = raw.strip()
            if s == "":
                flush()
                continue

            parts = s.replace(",", " ").split()
            if len(parts) < 2:
                flush()
                continue

            try:
                x, y = float(parts[0]), float(parts[1])
            except ValueError:
                flush()
                continue

            cur.append([x, y])

    flush()
    return polys


def _draw_obstacles(ax, polys, fill_alpha=0.07, edge_alpha=0.22):
    """
    Dibuja obstáculos globales como polígonos negros suaves.
    Menor alpha = más transparencia.
    """
    for p in polys:
        pts = np.asarray(p, dtype=np.float32)

        if pts.shape[0] >= 3 and np.linalg.norm(pts[0] - pts[-1]) > 1e-9:
            pts = np.vstack([pts, pts[0]])

        ax.fill(
            pts[:, 0],
            pts[:, 1],
            color="black",
            alpha=fill_alpha,
            zorder=1,
        )

        ax.plot(
            pts[:, 0],
            pts[:, 1],
            color="black",
            lw=0.8,
            alpha=edge_alpha,
            zorder=2,
        )


def _draw_radius_circle(ax, center_xy, radius=7.0):
    """Círculo de alcance (world frame)."""
    if center_xy is None:
        return

    cx, cy = float(center_xy[0]), float(center_xy[1])

    c = Circle(
        (cx, cy),
        radius=float(radius),
        edgecolor="#00E5FF",   # cian futurista
        facecolor="#00E5FF",
        alpha=0.1,           # relleno muy tenue
        linewidth=1.2,
        linestyle="--",
        zorder=3,
    )
    ax.add_patch(c)

def _plot_traj_with_points(
    ax,
    traj,
    color,
    label=None,
    lw=1.5,
    alpha=0.8,
    marker="o",
    markersize=16,
    marker_alpha=0.75,
    linestyle="-",
    zorder=10,
):
    """
    Dibuja una trayectoria como línea + puntos temporales.
    Sirve para que se vean los pasos de pasado, futuro GT y predicción.
    """
    if traj is None:
        return

    traj = np.asarray(traj, dtype=np.float32)

    if traj.ndim != 2 or traj.shape[0] == 0 or traj.shape[1] < 2:
        return

    ax.plot(
        traj[:, 0],
        traj[:, 1],
        color=color,
        lw=lw,
        alpha=alpha,
        linestyle=linestyle,
        label=label,
        zorder=zorder,
    )

    ax.scatter(
        traj[:, 0],
        traj[:, 1],
        color=color,
        s=markersize,
        alpha=marker_alpha,
        marker=marker,
        edgecolors="white",
        linewidths=0.35,
        zorder=zorder + 1,
    )

def _draw_segments_heatmap(
    ax,
    map_segments,
    map_mask,
    attn_ego=None,
    cmap_name="turbo",
    add_colorbar=True,
):
    """
    attn_ego:     (M,) atención/importancia del ego sobre cada segmento
    """
    if map_segments is None or map_mask is None:
        return

    segs = np.asarray(map_segments, dtype=np.float32)
    mask = np.asarray(map_mask).astype(bool)

    if segs.ndim != 2 or segs.shape[1] != 4:
        return

    M = segs.shape[0]

    # Si no hay atención, dibuja segmentos normales pero bonitos
    if attn_ego is None:
        for i in range(M):
            if i >= len(mask) or not mask[i]:
                continue

            x1, y1, x2, y2 = segs[i]

            # glow suave
            ax.plot(
                [x1, x2], [y1, y2],
                color="#00E5FF",
                linewidth=5.0,
                alpha=0.16,
                solid_capstyle="round",
                zorder=7,
            )

            # línea principal
            ax.plot(
                [x1, x2], [y1, y2],
                color="#00E5FF",
                linewidth=2.0,
                alpha=0.88,
                solid_capstyle="round",
                zorder=8,
            )
        return

    # Atención
    attn = np.asarray(attn_ego, dtype=np.float32).reshape(-1)

    # Asegurar misma longitud
    M = min(segs.shape[0], mask.shape[0], attn.shape[0])
    segs = segs[:M]
    mask = mask[:M]
    attn = attn[:M]

    valid_attn = attn[mask]
    if valid_attn.size == 0:
        return

    # Normalización robusta usando solo segmentos válidos
    a_min = float(valid_attn.min())
    a_max = float(valid_attn.max())

    if abs(a_max - a_min) < 1e-8:
        attn_norm = np.zeros_like(attn, dtype=np.float32)
    else:
        attn_norm = (attn - a_min) / (a_max - a_min + 1e-8)

    cmap = plt.get_cmap(cmap_name)

    for i in range(M):
        if not mask[i]:
            continue

        score = float(attn_norm[i])
        color = cmap(score)

        x1, y1, x2, y2 = segs[i]

        # Grosor según importancia:
        # score = 0   -> segmento delgado
        # score = 1   -> segmento grueso
        main_lw = 1.4 + 4.6 * score
        glow_lw = main_lw + 5.5

        # Opacidad según importancia
        main_alpha = 0.50 + 0.50 * score
        glow_alpha = 0.10 + 0.25 * score

        # Glow externo, da efecto futurista
        ax.plot(
            [x1, x2],
            [y1, y2],
            color=color,
            linewidth=glow_lw,
            alpha=glow_alpha,
            solid_capstyle="round",
            zorder=7,
        )

        # Línea principal
        ax.plot(
            [x1, x2],
            [y1, y2],
            color=color,
            linewidth=main_lw,
            alpha=main_alpha,
            solid_capstyle="round",
            zorder=8,
        )

        # Núcleo brillante del segmento
        ax.plot(
            [x1, x2],
            [y1, y2],
            color="white",
            linewidth=max(0.5, main_lw * 0.18),
            alpha=0.20 + 0.35 * score,
            solid_capstyle="round",
            zorder=9,
        )

    if add_colorbar:
        import matplotlib as mpl

        norm = mpl.colors.Normalize(vmin=0.0, vmax=1.0)
        sm = mpl.cm.ScalarMappable(norm=norm, cmap=cmap)
        sm.set_array([])

        cbar = plt.colorbar(sm, ax=ax, fraction=0.035, pad=0.02)
        cbar.set_label("Importancia del segmento", fontsize=9)

def _maybe_draw_map_overlays(ax, sample: dict, map_root=None, map_radius=7.0, dataset="eth_ucy"):
    """
    Dibuja overlays del mapa si existen:
      - obstáculos globales desde obstacles.txt
      - círculo de radio local
      - segmentos del mapa con heatmap de atención si existe attn_ego
    """

    scene_id = sample.get("scene_id", None)

    if map_root is not None and scene_id is not None:
        polys = _load_obstacles_polys(map_root, int(scene_id), dataset=dataset)
        _draw_obstacles(ax, polys)

    ego_center = sample.get("ego_center", None)
    if ego_center is not None:
        _draw_radius_circle(ax, ego_center, radius=map_radius)

    if "map_segments" in sample and "map_mask" in sample:
        _draw_segments_heatmap(
            ax=ax,
            map_segments=sample.get("map_segments", None),
            map_mask=sample.get("map_mask", None),
            attn_ego=sample.get("attn_ego", None),
            cmap_name="inferno",
            add_colorbar=True,
        )
# =========================================================
# Standard: Top-K
# =========================================================
def plot_ego_topk_modes(
    ego_hist,
    ego_gt,
    ego_topk_trajs,
    ego_topk_scores=None,
    ego_topk_nb_preds=None,
    neighbors_hist=None,
    neighbors_valid=None,
    neighbors_gt=None,
    save_path=None,
    sample_id=0,
    predictor_type="standard",
    # overlays (opcionales, no rompen nada)
    sample_meta=None,
    map_root=None,
    map_radius=7.0,
    dataset="eth_ucy",
):
    """
    Predictor normal:
    - K subplots, uno por modo.
    - Si sample_meta trae scene_id/ego_center/map_segments/attn_ego, dibuja overlays.
    """
    K = ego_topk_trajs.shape[0]
    K = int(K)

    fig, axes = plt.subplots(1, K, figsize=(5.5 * K, 5.5), dpi=140)
    if K == 1:
        axes = [axes]

    for k in range(K):
        ax = axes[k]

        # Overlays (si hay)
        if sample_meta is not None:
            _maybe_draw_map_overlays(ax, sample_meta, map_root=map_root, map_radius=map_radius, dataset=dataset)

        # Neighbors
        if neighbors_hist is not None:
            for n in range(neighbors_hist.shape[0]):
                valid_n = True
                if neighbors_valid is not None:
                    valid_n = bool(neighbors_valid[n])
                else:
                    valid_n = (np.sum(np.abs(neighbors_hist[n])) > 0)

                if not valid_n:
                    continue

                ax.plot(neighbors_hist[n, :, 0], neighbors_hist[n, :, 1], color="gray", lw=1.0, alpha=0.35)
                if neighbors_gt is not None and np.sum(np.abs(neighbors_gt[n])) > 0:
                    ax.plot(neighbors_gt[n, :, 0], neighbors_gt[n, :, 1], color="navy", lw=1.0, alpha=0.35)

                # Neighbor predictions for this mode (optional)
                if ego_topk_nb_preds is not None:
                    pred_n = ego_topk_nb_preds[k, n]
                    ax.plot(pred_n[:, 0], pred_n[:, 1], color="green", lw=1.2, alpha=0.55)

        # Ego hist + GT
        ax.plot(ego_hist[:, 0], ego_hist[:, 1], color="red", lw=2.0, label="ego hist")
        ax.scatter(ego_gt[:, 0], ego_gt[:, 1], marker="x", color="darkred", s=18, label="ego gt")

        # Ego pred for mode k
        traj = ego_topk_trajs[k]
        ax.plot(traj[:, 0], traj[:, 1], color="orange", lw=2.0, alpha=0.95, label=f"mode {k+1}")

        # connect hist->pred
        ax.plot([ego_hist[-1, 0], traj[0, 0]], [ego_hist[-1, 1], traj[0, 1]], color="orange", lw=1.0, alpha=0.25)

        title = f"Top-{K} | Mode {k+1}"
        if ego_topk_scores is not None:
            try:
                title += f" | p={float(ego_topk_scores[k]):.3f}"
            except Exception:
                pass
        ax.set_title(title)
        ax.set_aspect("equal", adjustable="box")
        ax.grid(True, alpha=0.12)

    plt.tight_layout()
    if save_path is not None:
        _ensure_dir(save_path)
        out = os.path.join(save_path, f"{predictor_type}_standard_topk_sample_{sample_id:04d}.png")
        plt.savefig(out, bbox_inches="tight")
    plt.close(fig)


# =========================================================
# VAE: final
# =========================================================
def plot_vae_final_scenario(sample, save_path=None, sample_id=0, map_root=None, map_radius=7.0, dataset="eth_ucy"):
    """
    Dibuja una escena VAE final:
      - ego hist/gt/pred
      - neighbors hist/gt/pred
      - overlays mapa si existen en sample
    """
    ego_hist = sample["ego_hist"]
    ego_gt = sample["ego_gt"]
    ego_pred = sample["ego_pred"]
    nb_hist = sample["neighbors_hist"]
    nb_gt = sample["neighbors_gt"]
    nb_pred = sample["neighbors_pred"]
    nb_valid = sample.get("neighbors_valid", None)

    fig, ax = plt.subplots(1, 1, figsize=(7.0, 7.0), dpi=150)

    # overlays
    _maybe_draw_map_overlays(ax, sample, map_root=map_root, map_radius=map_radius, dataset=dataset)

    # neighbors
    for n in range(nb_hist.shape[0]):
        valid_n = True
        if nb_valid is not None:
            valid_n = bool(nb_valid[n])
        else:
            valid_n = (np.sum(np.abs(nb_hist[n])) > 0)

        if not valid_n:
            continue

        ax.plot(nb_hist[n, :, 0], nb_hist[n, :, 1], color="gray", lw=1.0, alpha=0.35)
        if np.sum(np.abs(nb_gt[n])) > 0:
            ax.plot(nb_gt[n, :, 0], nb_gt[n, :, 1], color="navy", lw=1.0, alpha=0.35)
        ax.plot(nb_pred[n, :, 0], nb_pred[n, :, 1], color="green", lw=1.2, alpha=0.55)

    # ego
    ax.plot(ego_hist[:, 0], ego_hist[:, 1], color="red", lw=2.0, label="ego hist")
    ax.scatter(ego_gt[:, 0], ego_gt[:, 1], marker="x", color="darkred", s=18, label="ego gt")
    ax.plot(ego_pred[:, 0], ego_pred[:, 1], color="orange", lw=2.0, label="ego pred")

    ax.plot([ego_hist[-1, 0], ego_pred[0, 0]], [ego_hist[-1, 1], ego_pred[0, 1]], color="orange", lw=1.0, alpha=0.25)

    ax.set_title("VAE final scenario")
    ax.set_aspect("equal", adjustable="box")
    ax.grid(True, alpha=0.12)
    ax.legend()

    plt.tight_layout()
    if save_path is not None:
        _ensure_dir(save_path)
        out = os.path.join(save_path, f"vae_final_sample_{sample_id:04d}.png")
        plt.savefig(out, bbox_inches="tight")
    plt.close(fig)


# =========================================================
# VAE: multi
# =========================================================
def plot_vae_multisample_scenarios(sample, save_path=None, sample_id=0, map_root=None, map_radius=7.0, dataset="eth_ucy"):
    """
    Dibuja K muestras del VAE:
      - ego hist/gt
      - ego_samples_trajs (K)
      - neighbor_samples_preds (K)
      - overlays mapa si existen
    """
    ego_hist = sample["ego_hist"]
    ego_gt = sample["ego_gt"]
    nb_hist = sample["neighbors_hist"]
    nb_gt = sample["neighbors_gt"]
    nb_valid = sample.get("neighbors_valid", None)

    ego_samples = sample["ego_samples_trajs"]           # (K,T,2)
    nb_samples = sample["neighbor_samples_preds"]       # (K,N,T,2)

    K = ego_samples.shape[0]

    cmap = plt.get_cmap("turbo")
    sample_colors = [cmap(k / max(K - 1, 1)) for k in range(K)]

    fig, ax = plt.subplots(1, 1, figsize=(7.0, 7.0), dpi=150)

    # overlays
    _maybe_draw_map_overlays(ax, sample, map_root=map_root, map_radius=map_radius, dataset=dataset)

    # neighbors hist/gt
    for n in range(nb_hist.shape[0]):
        valid_n = True
        if nb_valid is not None:
            valid_n = bool(nb_valid[n])
        else:
            valid_n = (np.sum(np.abs(nb_hist[n])) > 0)

        if not valid_n:
            continue

        ax.plot(nb_hist[n, :, 0], nb_hist[n, :, 1], color="gray", lw=1.0, alpha=0.30)
        if np.sum(np.abs(nb_gt[n])) > 0:
            ax.plot(nb_gt[n, :, 0], nb_gt[n, :, 1], color="navy", lw=1.0, alpha=0.30)

    # ego hist/gt
    ax.plot(ego_hist[:, 0], ego_hist[:, 1], color="red", lw=2.0, label="ego hist")
    ax.scatter(ego_gt[:, 0], ego_gt[:, 1], marker="x", color="darkred", s=18, label="ego gt")

    # samples (ego + neighbor preds)
    for k in range(K):
        color_k = sample_colors[k]

        # Ego predicho para la muestra k
        _plot_traj_with_points(
            ax,
            ego_samples[k],
            color=color_k,
            lw=1.4,
            alpha=0.55,
            markersize=9,
            marker_alpha=0.45,
            zorder=14,
        )

        # Vecinos predichos para la muestra k
        for n in range(nb_samples.shape[1]):
            pred_kn = nb_samples[k, n]

            if np.sum(np.abs(pred_kn)) <= 1e-6:
                continue

            _plot_traj_with_points(
                ax,
                pred_kn,
                color=color_k,
                lw=0.9,
                alpha=0.22,
                markersize=5,
                marker_alpha=0.20,
                zorder=13,
            )

    ax.set_title(f"VAE multi samples (K={K})")
    ax.set_aspect("equal", adjustable="box")
    ax.grid(True, alpha=0.12)
    ax.legend()

    plt.tight_layout()
    if save_path is not None:
        _ensure_dir(save_path)
        out = os.path.join(save_path, f"vae_multi_sample_{sample_id:04d}.png")
        plt.savefig(out, bbox_inches="tight")
    plt.close(fig)


# =========================================================
# Dispatcher
# =========================================================
def save_visualizations_from_samples(
    samples_data,
    save_path,
    predictor_type="standard",
    map_root=None,
    map_radius=7.0,
    dataset="eth_ucy",
):
    """
    samples_data: list of dicts.
      - standard_topk: keys ego_hist/ego_gt/ego_topk_trajs/...
      - vae_final: keys ego_hist/ego_gt/ego_pred/...
      - vae_multi: keys ego_samples_trajs/neighbor_samples_preds/...

    Extras opcionales por sample:
      - scene_id, ego_center, map_segments, map_mask, attn_ego
    """
    _ensure_dir(save_path)

    for i, data in enumerate(samples_data):
        kind = data.get("kind", None)

        if kind == "standard_topk":
            plot_ego_topk_modes(
                ego_hist=data["ego_hist"],
                ego_gt=data["ego_gt"],
                ego_topk_trajs=np.asarray(data["ego_topk_trajs"]),
                ego_topk_scores=np.asarray(data["ego_topk_scores"]) if "ego_topk_scores" in data else None,
                ego_topk_nb_preds=np.asarray(data["ego_topk_nb_preds"]) if "ego_topk_nb_preds" in data else None,
                neighbors_hist=np.asarray(data["neighbors_hist"]) if "neighbors_hist" in data else None,
                neighbors_valid=np.asarray(data["neighbors_valid"]) if "neighbors_valid" in data else None,
                neighbors_gt=np.asarray(data["neighbors_gt"]) if "neighbors_gt" in data else None,
                save_path=save_path,
                sample_id=i,
                predictor_type=predictor_type,
                sample_meta=data,
                map_root=map_root,
                map_radius=map_radius,
                dataset=dataset,
            )

        elif kind == "vae_final":
            plot_vae_final_scenario(
                sample=data,
                save_path=save_path,
                sample_id=i,
                map_root=map_root,
                map_radius=map_radius,
                dataset=dataset,
            )

        elif kind == "vae_multi":
            plot_vae_multisample_scenarios(
                sample=data,
                save_path=save_path,
                sample_id=i,
                map_root=map_root,
                map_radius=map_radius,
                dataset=dataset,
            )

        else:
            # Si llega algún sample desconocido, no rompemos.
            continue