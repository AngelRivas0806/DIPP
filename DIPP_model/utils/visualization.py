import matplotlib.pyplot as plt
import numpy as np
import os


def _ensure_dir(path):
    os.makedirs(path, exist_ok=True)


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
    predictor_type="standard"
):
    """
    Predictor normal:
    - Predicciones: líneas limpias (sin markers)
    - GT: con tachitas 'x' en cada paso
    - Conector hist->pred muy tenue
    """
    K = ego_topk_trajs.shape[0]

    ego_color = '#FF0000'
    ego_gt_color = '#8B0000'
    neighbor_color = '#0000FF'
    neighbor_pred_color = '#00AA00'
    neighbor_gt_color = '#000080'
    mode_colors = ['#FF8C00', '#FFA500', '#FFD700', '#ADFF2F', '#FF69B4']
    mode_labels = [f'Modo {k+1}' for k in range(K)]

    all_x = np.concatenate([ego_hist[:, 0], ego_gt[:, 0], ego_topk_trajs[:, :, 0].ravel()])
    all_y = np.concatenate([ego_hist[:, 1], ego_gt[:, 1], ego_topk_trajs[:, :, 1].ravel()])

    if neighbors_hist is not None:
        for i in range(neighbors_hist.shape[0]):
            valid_i = True
            if neighbors_valid is not None:
                valid_i = bool(neighbors_valid[i])
            elif np.sum(np.abs(neighbors_hist[i])) == 0:
                valid_i = False

            if valid_i:
                all_x = np.concatenate([all_x, neighbors_hist[i, :, 0]])
                all_y = np.concatenate([all_y, neighbors_hist[i, :, 1]])
                if neighbors_gt is not None and np.sum(np.abs(neighbors_gt[i])) > 0:
                    all_x = np.concatenate([all_x, neighbors_gt[i, :, 0]])
                    all_y = np.concatenate([all_y, neighbors_gt[i, :, 1]])

    margin = 0.12
    cx = (all_x.max() + all_x.min()) / 2.0
    cy = (all_y.max() + all_y.min()) / 2.0
    half = max((all_x.max() - all_x.min()), (all_y.max() - all_y.min())) / 2.0 * (1 + margin)
    half = half if half > 0 else 1.0
    xlim = (cx - half, cx + half)
    ylim = (cy - half, cy + half)

    fig, axes = plt.subplots(1, K, figsize=(8 * K, 8))
    if K == 1:
        axes = [axes]

    fig.suptitle(
        f'Top-{K} modos del ego con probabilidad — Sample {sample_id} ({predictor_type})',
        fontsize=16, fontweight='bold', y=1.01
    )

    for k, ax in enumerate(axes):
        nb_pred_k = ego_topk_nb_preds[k] if ego_topk_nb_preds is not None else None

        # Vecinos: hist / pred / gt
        if neighbors_hist is not None:
            for i in range(neighbors_hist.shape[0]):
                valid_i = True
                if neighbors_valid is not None:
                    valid_i = bool(neighbors_valid[i])
                elif np.sum(np.abs(neighbors_hist[i])) == 0:
                    valid_i = False
                if not valid_i:
                    continue

                ax.plot(neighbors_hist[i, :, 0], neighbors_hist[i, :, 1],
                        color=neighbor_color, linewidth=1.2, linestyle='-', alpha=0.45)

                if nb_pred_k is not None and i < nb_pred_k.shape[0]:
                    ax.plot(nb_pred_k[i, :, 0], nb_pred_k[i, :, 1],
                            color=neighbor_pred_color, linewidth=1.2, linestyle='--', alpha=0.55)

                if neighbors_gt is not None and i < neighbors_gt.shape[0] and np.sum(np.abs(neighbors_gt[i])) > 0:
                    ax.plot(neighbors_gt[i, :, 0], neighbors_gt[i, :, 1],
                            color=neighbor_gt_color, linewidth=1.0, linestyle=':',
                            alpha=0.85, marker='x', markersize=4, markevery=1)

            ax.plot([], [], color=neighbor_color, linewidth=1.2, alpha=0.45, label='Vecinos Hist')
            if nb_pred_k is not None:
                ax.plot([], [], color=neighbor_pred_color, linewidth=1.2, linestyle='--', alpha=0.55, label='Vecinos Pred')
            if neighbors_gt is not None:
                ax.plot([], [], color=neighbor_gt_color, linewidth=1.0, linestyle=':',
                        alpha=0.85, marker='x', markersize=4, label='Vecinos GT')

        # Ego hist
        ax.plot(ego_hist[:, 0], ego_hist[:, 1],
                color=ego_color, linewidth=2.5, linestyle='-', alpha=0.7, label='Ego History')

        # Ego GT (tachitas)
        ax.plot(ego_gt[:, 0], ego_gt[:, 1],
                color=ego_gt_color, linewidth=2.0, linestyle=':',
                alpha=0.95, marker='x', markersize=7, markevery=1, label='Ego GT')

        # Ego pred (línea limpia)
        traj = ego_topk_trajs[k]
        ax.plot(traj[:, 0], traj[:, 1],
                color=mode_colors[k % len(mode_colors)], linewidth=3, linestyle='--',
                alpha=0.95, label=mode_labels[k])

        # Conector tenue (opcional)
        ax.plot([ego_hist[-1, 0], traj[0, 0]], [ego_hist[-1, 1], traj[0, 1]],
                color=mode_colors[k % len(mode_colors)], linewidth=0.6, linestyle='--', alpha=0.12)

        ax.set_xlim(xlim)
        ax.set_ylim(ylim)
        ax.set_aspect('equal', adjustable='box')

        if ego_topk_scores is not None:
            title = f'{mode_labels[k]}\nProb: {ego_topk_scores[k]:.1%}'
        else:
            title = mode_labels[k]

        ax.set_title(title, fontsize=13, fontweight='bold')
        ax.set_xlabel('X (m)', fontsize=11)
        ax.set_ylabel('Y (m)', fontsize=11)
        ax.legend(loc='best', fontsize=9, framealpha=0.85)
        ax.grid(True, alpha=0.3, linestyle='--', linewidth=0.5)

    plt.tight_layout()

    if save_path is not None:
        _ensure_dir(save_path)
        filepath = os.path.join(save_path, f'topk_modes_sample_{sample_id:04d}.png')
        plt.savefig(filepath, dpi=150, bbox_inches='tight')
        print(f"Saved: {filepath}")

    plt.close()


# =========================================================
# VAE: FINAL ONLY (una sola escena)
# =========================================================
def plot_vae_final_scenario(
    ego_hist,
    ego_gt,
    ego_pred,
    neighbors_hist=None,
    neighbors_gt=None,
    neighbors_pred=None,
    neighbors_valid=None,
    selected_idx=None,
    save_path=None,
    sample_id=0,
    predictor_type="vae_final"
):
    """
    VAE (final-only):
    - Predicciones: líneas limpias
    - GT: tachitas 'x' en cada paso
    """
    fig, ax = plt.subplots(figsize=(12, 10))

    ego_hist_color = '#FF0000'
    ego_gt_color = '#8B0000'
    ego_pred_color = '#FF8C00'

    neighbor_hist_color = '#0000FF'
    neighbor_gt_color = '#000080'
    neighbor_pred_color = '#00AA00'

    ax.plot(ego_hist[:, 0], ego_hist[:, 1],
            color=ego_hist_color, linewidth=1.5, alpha=0.75, label="Ego Hist")

    ax.plot(ego_gt[:, 0], ego_gt[:, 1],
            color=ego_gt_color, linewidth=3, linestyle=':',
            alpha=0.95, marker='x', markersize=6, markevery=1, label="Ego GT")

    ax.plot(ego_pred[:, 0], ego_pred[:, 1],
            color=ego_pred_color, linewidth=2.5, linestyle='-',
            alpha=0.95, label="Ego Pred (final)")

    if neighbors_hist is not None:
        for i in range(neighbors_hist.shape[0]):
            valid_i = True
            if neighbors_valid is not None:
                valid_i = bool(neighbors_valid[i])
            elif np.sum(np.abs(neighbors_hist[i])) == 0:
                valid_i = False
            if not valid_i:
                continue

            ax.plot(neighbors_hist[i, :, 0], neighbors_hist[i, :, 1],
                    color=neighbor_hist_color, linewidth=1.0, alpha=0.45)

            if neighbors_pred is not None and i < neighbors_pred.shape[0]:
                ax.plot(neighbors_pred[i, :, 0], neighbors_pred[i, :, 1],
                        color=neighbor_pred_color, linewidth=1.0, linestyle='--', alpha=0.6)

            if neighbors_gt is not None and i < neighbors_gt.shape[0] and np.sum(np.abs(neighbors_gt[i])) > 0:
                ax.plot(neighbors_gt[i, :, 0], neighbors_gt[i, :, 1],
                        color=neighbor_gt_color, linewidth=1.0, linestyle=':',
                        alpha=0.85, marker='x', markersize=4, markevery=1)

        ax.plot([], [], color=neighbor_hist_color, linewidth=1.0, alpha=0.45, label="Vecinos Hist")
        if neighbors_pred is not None:
            ax.plot([], [], color=neighbor_pred_color, linewidth=1.0, linestyle='--', alpha=0.6, label="Vecinos Pred")
        if neighbors_gt is not None:
            ax.plot([], [], color=neighbor_gt_color, linewidth=1.0, linestyle=':', alpha=0.85,
                    marker='x', markersize=4, label="Vecinos GT")

    # Auto-límites
    all_x = [ego_hist[:, 0], ego_gt[:, 0], ego_pred[:, 0]]
    all_y = [ego_hist[:, 1], ego_gt[:, 1], ego_pred[:, 1]]

    if neighbors_hist is not None:
        for i in range(neighbors_hist.shape[0]):
            valid_i = True
            if neighbors_valid is not None:
                valid_i = bool(neighbors_valid[i])
            elif np.sum(np.abs(neighbors_hist[i])) == 0:
                valid_i = False
            if valid_i:
                all_x.append(neighbors_hist[i, :, 0])
                all_y.append(neighbors_hist[i, :, 1])

    if neighbors_gt is not None:
        for i in range(neighbors_gt.shape[0]):
            if np.sum(np.abs(neighbors_gt[i])) > 0:
                all_x.append(neighbors_gt[i, :, 0])
                all_y.append(neighbors_gt[i, :, 1])

    if neighbors_pred is not None:
        all_x.append(neighbors_pred[:, :, 0].ravel())
        all_y.append(neighbors_pred[:, :, 1].ravel())

    all_x = np.concatenate(all_x)
    all_y = np.concatenate(all_y)

    margin = 0.1
    x_range = max(all_x.max() - all_x.min(), 1e-6)
    y_range = max(all_y.max() - all_y.min(), 1e-6)
    ax.set_xlim(all_x.min() - margin * x_range, all_x.max() + margin * x_range)
    ax.set_ylim(all_y.min() - margin * y_range, all_y.max() + margin * y_range)

    title = f'VAE final scenario — Sample {sample_id}'
    if selected_idx is not None:
        title += f' | selected_k={selected_idx}'
    ax.set_title(title, fontsize=16, fontweight='bold', pad=20)

    ax.set_xlabel('X (m)', fontsize=14, fontweight='bold')
    ax.set_ylabel('Y (m)', fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3, linestyle='--', linewidth=0.5)
    ax.set_aspect('equal', adjustable='box')
    ax.legend(loc='best', fontsize=10, framealpha=0.85)

    plt.tight_layout()
    if save_path is not None:
        _ensure_dir(save_path)
        filepath = os.path.join(save_path, f'vae_final_scene_{sample_id:04d}.png')
        plt.savefig(filepath, dpi=150, bbox_inches='tight')
        print(f"Saved: {filepath}")
    plt.close()


# =========================================================
# VAE: MULTISAMPLE (K muestras)
# =========================================================
def plot_vae_multisample_scenarios(
    ego_hist,
    ego_gt,
    ego_samples_trajs,
    neighbors_hist=None,
    neighbor_samples_preds=None,
    neighbors_gt=None,
    neighbors_valid=None,
    z_values=None,
    selected_idx=None,
    save_path=None,
    sample_id=0,
    predictor_type="vae"
):
    """
    VAE multisample:
    - Predicciones: líneas limpias (ego y vecinos, sin markers)
    - GT: tachitas 'x' en cada paso (ego y vecinos)
    - Conector hist->pred muy tenue
    """
    K = ego_samples_trajs.shape[0]
    fig, ax = plt.subplots(figsize=(12, 10))

    ego_hist_color = '#FF0000'
    ego_gt_color = '#8B0000'
    neighbor_hist_color = '#0000FF'
    neighbor_gt_color = '#000080'
    cmap = plt.cm.get_cmap('tab10', max(K, 1))

    ax.plot(ego_hist[:, 0], ego_hist[:, 1],
            color=ego_hist_color, linewidth=1.5, alpha=0.75, label='Ego History')

    ax.plot(ego_gt[:, 0], ego_gt[:, 1],
            color=ego_gt_color, linewidth=3, linestyle=':',
            alpha=0.95, marker='x', markersize=6, markevery=1, label='Ego GT')

    if neighbors_hist is not None:
        for i in range(neighbors_hist.shape[0]):
            valid_i = True
            if neighbors_valid is not None:
                valid_i = bool(neighbors_valid[i])
            elif np.sum(np.abs(neighbors_hist[i])) == 0:
                valid_i = False
            if not valid_i:
                continue

            ax.plot(neighbors_hist[i, :, 0], neighbors_hist[i, :, 1],
                    color=neighbor_hist_color, linewidth=1.2, alpha=0.5)

            if neighbors_gt is not None and i < neighbors_gt.shape[0] and np.sum(np.abs(neighbors_gt[i])) > 0:
                ax.plot(neighbors_gt[i, :, 0], neighbors_gt[i, :, 1],
                        color=neighbor_gt_color, linewidth=1.2, linestyle=':',
                        alpha=0.85, marker='x', markersize=4, markevery=1)

        ax.plot([], [], color=neighbor_hist_color, linewidth=1.2, alpha=0.5, label='Vecinos Hist')
        if neighbors_gt is not None:
            ax.plot([], [], color=neighbor_gt_color, linewidth=1.2, linestyle=':',
                    alpha=0.85, marker='x', markersize=4, label='Vecinos GT')

    for k in range(K):
        color_k = cmap(k)
        traj_k = ego_samples_trajs[k]

        label_k = f'z sample {k}'
        if selected_idx is not None and int(selected_idx) == k:
            label_k = f'z sample {k} (selected)'

        ax.plot(traj_k[:, 0], traj_k[:, 1],
                color=color_k, linewidth=1.0, alpha=0.9, label=label_k)

        ax.plot([ego_hist[-1, 0], traj_k[0, 0]], [ego_hist[-1, 1], traj_k[0, 1]],
                color=color_k, linewidth=0.5, linestyle='--', alpha=0.12)

        if neighbor_samples_preds is not None:
            preds_k = neighbor_samples_preds[k]  # (N,T,2)
            for i in range(preds_k.shape[0]):
                valid_i = True
                if neighbors_valid is not None and i < len(neighbors_valid):
                    valid_i = bool(neighbors_valid[i])
                if not valid_i:
                    continue
                ax.plot(preds_k[i, :, 0], preds_k[i, :, 1],
                        color=color_k, linewidth=0.9, alpha=0.75)

    # límites
    all_x = [ego_hist[:, 0], ego_gt[:, 0], ego_samples_trajs[:, :, 0].ravel()]
    all_y = [ego_hist[:, 1], ego_gt[:, 1], ego_samples_trajs[:, :, 1].ravel()]
    if neighbors_hist is not None:
        for i in range(neighbors_hist.shape[0]):
            valid_i = True
            if neighbors_valid is not None:
                valid_i = bool(neighbors_valid[i])
            elif np.sum(np.abs(neighbors_hist[i])) == 0:
                valid_i = False
            if valid_i:
                all_x.append(neighbors_hist[i, :, 0])
                all_y.append(neighbors_hist[i, :, 1])
    if neighbors_gt is not None:
        for i in range(neighbors_gt.shape[0]):
            if np.sum(np.abs(neighbors_gt[i])) > 0:
                all_x.append(neighbors_gt[i, :, 0])
                all_y.append(neighbors_gt[i, :, 1])
    if neighbor_samples_preds is not None:
        all_x.append(neighbor_samples_preds[:, :, :, 0].ravel())
        all_y.append(neighbor_samples_preds[:, :, :, 1].ravel())

    all_x = np.concatenate(all_x)
    all_y = np.concatenate(all_y)
    margin = 0.1
    x_range = max(all_x.max() - all_x.min(), 1e-6)
    y_range = max(all_y.max() - all_y.min(), 1e-6)
    ax.set_xlim(all_x.min() - margin * x_range, all_x.max() + margin * x_range)
    ax.set_ylim(all_y.min() - margin * y_range, all_y.max() + margin * y_range)

    ax.set_title(f'VAE sampled scenarios — Sample {sample_id} ({predictor_type}) | K={K}',
                 fontsize=16, fontweight='bold', pad=20)

    ax.set_xlabel('X (m)', fontsize=14, fontweight='bold')
    ax.set_ylabel('Y (m)', fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3, linestyle='--', linewidth=0.5)
    ax.set_aspect('equal', adjustable='box')
    # ax.legend(...)  # opcional

    plt.tight_layout()
    if save_path is not None:
        _ensure_dir(save_path)
        filepath = os.path.join(save_path, f'vae_multisample_scene_{sample_id:04d}.png')
        plt.savefig(filepath, dpi=150, bbox_inches='tight')
        print(f"Saved: {filepath}")
    plt.close()


# =========================================================
# Dispatcher
# =========================================================
def save_visualizations_from_samples(samples_data, save_path, predictor_type="standard"):
    """
    Soporta:
    - standard: top-k
    - vae: samples con kind="vae_final" o kind="vae_multi"
    """
    _ensure_dir(save_path)

    if predictor_type == "standard":
        topk_dir = os.path.join(save_path, "topk_modes")
        _ensure_dir(topk_dir)

        for idx, data in enumerate(samples_data):
            plot_ego_topk_modes(
                ego_hist=np.asarray(data["ego_hist"]),
                ego_gt=np.asarray(data["ego_gt"]),
                ego_topk_trajs=np.asarray(data["ego_topk_trajs"]),
                ego_topk_scores=np.asarray(data["ego_topk_scores"]) if "ego_topk_scores" in data else None,
                ego_topk_nb_preds=np.asarray(data["ego_topk_nb_preds"]) if "ego_topk_nb_preds" in data else None,
                neighbors_hist=np.asarray(data["neighbors_hist"]) if "neighbors_hist" in data else None,
                neighbors_valid=np.asarray(data["neighbors_valid"]) if "neighbors_valid" in data else None,
                neighbors_gt=np.asarray(data["neighbors_gt"]) if "neighbors_gt" in data else None,
                save_path=topk_dir,
                sample_id=idx,
                predictor_type=predictor_type
            )

    elif predictor_type == "vae":
        vae_root = os.path.join(save_path, "vae")
        _ensure_dir(vae_root)

        for idx, data in enumerate(samples_data):
            kind = data.get("kind", "vae_multi")

            if kind == "vae_final":
                out_dir = os.path.join(vae_root, "vae_final")
                _ensure_dir(out_dir)
                plot_vae_final_scenario(
                    ego_hist=np.asarray(data["ego_hist"]),
                    ego_gt=np.asarray(data["ego_gt"]),
                    ego_pred=np.asarray(data["ego_pred"]),
                    neighbors_hist=np.asarray(data["neighbors_hist"]) if "neighbors_hist" in data else None,
                    neighbors_gt=np.asarray(data["neighbors_gt"]) if "neighbors_gt" in data else None,
                    neighbors_pred=np.asarray(data["neighbors_pred"]) if "neighbors_pred" in data else None,
                    neighbors_valid=np.asarray(data["neighbors_valid"]) if "neighbors_valid" in data else None,
                    selected_idx=data.get("selected_idx", None),
                    save_path=out_dir,
                    sample_id=idx,
                    predictor_type="vae_final"
                )

            elif kind == "vae_multi":
                out_dir = os.path.join(vae_root, "vae_multisample")
                _ensure_dir(out_dir)
                plot_vae_multisample_scenarios(
                    ego_hist=np.asarray(data["ego_hist"]),
                    ego_gt=np.asarray(data["ego_gt"]),
                    ego_samples_trajs=np.asarray(data["ego_samples_trajs"]),
                    neighbors_hist=np.asarray(data["neighbors_hist"]) if "neighbors_hist" in data else None,
                    neighbor_samples_preds=np.asarray(data["neighbor_samples_preds"]) if "neighbor_samples_preds" in data else None,
                    neighbors_gt=np.asarray(data["neighbors_gt"]) if "neighbors_gt" in data else None,
                    neighbors_valid=np.asarray(data["neighbors_valid"]) if "neighbors_valid" in data else None,
                    z_values=np.asarray(data["z"]) if "z" in data else None,
                    selected_idx=data.get("selected_idx", None),
                    save_path=out_dir,
                    sample_id=idx,
                    predictor_type="vae"
                )

            else:
                print(f"[WARN] Unknown VAE sample kind '{kind}', skipping idx={idx}")

    else:
        raise ValueError(f"Unknown predictor_type: {predictor_type}")