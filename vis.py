import os
import argparse
from typing import Optional, Dict, List, Any

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Circle

ETH_UCY_DATASETS = ["eth-hotel", "eth-univ", "ucy-zara01", "ucy-zara02", "ucy-univ"]


def print_npz_info(path: str):
    data = np.load(path, allow_pickle=False)
    print("Keys:", list(data.keys()))
    for k in data.files:
        arr = data[k]
        print(f"{k:20s} shape={arr.shape} dtype={arr.dtype}")
    return data


def rotation_matrix(theta: float) -> np.ndarray:
    c, s = np.cos(theta), np.sin(theta)
    return np.array([[c, -s],
                     [s,  c]], dtype=np.float32)


def to_ego_frame(xy: np.ndarray, origin: np.ndarray, R: Optional[np.ndarray]) -> np.ndarray:
    xy = np.asarray(xy, dtype=np.float32)
    origin = np.asarray(origin, dtype=np.float32)
    out = xy - origin
    if R is not None:
        out = out @ R.T
    return out


# ---------- Obstacles: exact parser + cache ----------

def load_obstacles_file(path: str) -> List[np.ndarray]:
    """
    Formato:
      - líneas: x y (float)
      - línea vacía separa obstáculos
    """
    if path is None or (not os.path.exists(path)):
        return []

    obstacles: List[np.ndarray] = []
    cur: List[List[float]] = []

    def flush():
        nonlocal cur
        if len(cur) >= 3:
            obstacles.append(np.asarray(cur, dtype=np.float32))
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
                x = float(parts[0])
                y = float(parts[1])
            except ValueError:
                flush()
                continue

            cur.append([x, y])

    flush()
    return obstacles


def build_obstacle_cache(map_root: str) -> Dict[int, Dict[str, Any]]:
    cache: Dict[int, Dict[str, Any]] = {}
    for sid, scene_name in enumerate(ETH_UCY_DATASETS):
        path = os.path.join(map_root, scene_name, "obstacles.txt")
        obs_list = load_obstacles_file(path)
        sizes = [o.shape[0] for o in obs_list]

        if len(obs_list) > 0:
            all_pts = np.concatenate(obs_list, axis=0).astype(np.float32)
            xmin, ymin = all_pts.min(axis=0)
            xmax, ymax = all_pts.max(axis=0)
            bbox = (float(xmin), float(xmax), float(ymin), float(ymax))
        else:
            bbox = None

        cache[sid] = {"obstacles": obs_list, "bbox": bbox}
        print(f"[obstacles] Scene {sid} ({scene_name}): obstacles={len(obs_list)} sizes={sizes} bbox={bbox}")

    return cache


def draw_obstacles(ax, obstacles_list: List[np.ndarray], origin, R, ego_frame: bool,
                   fill_black: bool = True, fill_alpha: float = 1.0):
    if not obstacles_list:
        return

    for poly in obstacles_list:
        pts = np.asarray(poly, dtype=np.float32)
        if ego_frame:
            pts = to_ego_frame(pts, origin, R)

        # cierre visual
        if pts.shape[0] >= 3 and np.linalg.norm(pts[0] - pts[-1]) > 1e-9:
            pts = np.vstack([pts, pts[0]])

        if fill_black and pts.shape[0] >= 3:
            ax.fill(pts[:, 0], pts[:, 1], color="black", alpha=fill_alpha, zorder=2)
            ax.plot(pts[:, 0], pts[:, 1], color="black", lw=1.2, alpha=1.0, zorder=2)
        else:
            ax.plot(pts[:, 0], pts[:, 1], color="black", lw=2.0, alpha=0.9, zorder=2)


# ---------- Plot sample ----------

def plot_sample(
    data,
    i: int,
    title: str = "",
    ego_frame: bool = True,
    rotate: bool = False,
    show_obstacles: bool = False,
    obstacle_cache: Optional[Dict[int, Dict[str, Any]]] = None,
    view_all_obstacles: bool = True,
    fig_w: float = 24.0,
    fig_h: float = 24.0,
    dpi: int = 160,
    save_path: Optional[str] = None,
):
    ego       = data["ego"][i]
    neighbors = data["neighbors"][i]
    gt        = data["gt_future_states"][i]

    K       = neighbors.shape[0]
    obs_len = ego.shape[0]
    c       = obs_len - 1

    ego_obs_xy = ego[:, 0:2].astype(np.float32)
    ego_gt_xy  = gt[0, :, 0:2].astype(np.float32)

    neigh_valid  = neighbors[:, :, 6].sum(axis=1) > 0
    neigh_obs_xy = neighbors[:, :, 0:2].astype(np.float32)
    neigh_gt_xy  = gt[1:, :, 0:2].astype(np.float32)

    origin = ego_obs_xy[c].copy()
    R = None

    if ego_frame and rotate:
        if c >= 1:
            v = ego_obs_xy[c] - ego_obs_xy[c - 1]
        else:
            v = ego_gt_xy[0] - ego_obs_xy[c]
        norm = np.linalg.norm(v)
        if norm > 1e-6:
            v = v / norm
            theta = -np.arctan2(v[1], v[0])
            R = rotation_matrix(theta)

    if ego_frame:
        ego_obs_xy   = to_ego_frame(ego_obs_xy, origin, R)
        ego_gt_xy    = to_ego_frame(ego_gt_xy, origin, R)
        neigh_obs_xy = to_ego_frame(neigh_obs_xy, origin, R)
        neigh_gt_xy  = to_ego_frame(neigh_gt_xy, origin, R)

    plt.figure(figsize=(fig_w, fig_h), dpi=dpi)
    ax = plt.gca()
    # Iterar sobre los 4 bordes (arriba, abajo, izquierda, derecha)
    for spine in ax.spines.values():
        spine.set_linewidth(5.0)  # Cambia 2.0 por el grosor deseado
    # Obstáculos
    if show_obstacles and obstacle_cache is not None and "scene_id" in data:
        sid = int(data["scene_id"][i])
        entry = obstacle_cache.get(sid, {"obstacles": [], "bbox": None})
        obs = entry["obstacles"]
        draw_obstacles(ax, obs, origin=origin, R=R, ego_frame=ego_frame)

        if view_all_obstacles:
            ax.relim()
            ax.autoscale_view()
            ax.margins(0.05)

    ax.set_title(title or f"Sample {i}" + (" (ego-frame)" if ego_frame else ""))

    s_pt = 6
    s_center = 60
    lw = 1.2
    alpha_neigh = 0.55

    # Radio
    radius_m = 7.0
    center_xy = np.array([0.0, 0.0], dtype=np.float32) if ego_frame else ego_obs_xy[c]
    circle = Circle((float(center_xy[0]), float(center_xy[1])),
                    radius=radius_m, edgecolor="yellow", facecolor="yellow",
                    alpha=0.3, linewidth=1.5, zorder = 1)
    ax.add_patch(circle)

    # Vecinos
    for k in range(K):
        if not neigh_valid[k]:
            continue
        xy_obs = neigh_obs_xy[k]
        xy_gt = neigh_gt_xy[k]

        ax.scatter(xy_obs[:, 0], xy_obs[:, 1], s=s_pt, c="gray", alpha=alpha_neigh, zorder=5)
        ax.scatter(xy_gt[:, 0],  xy_gt[:, 1],  s=s_pt, c="blue", alpha=alpha_neigh, zorder=5)

        ax.plot(xy_obs[:, 0], xy_obs[:, 1], c="gray", linewidth=lw, alpha=0.35, zorder=5)
        ax.plot(xy_gt[:, 0],  xy_gt[:, 1],  c="blue", linewidth=lw, alpha=0.35, zorder=5)

        ax.plot([xy_obs[-1, 0], xy_gt[0, 0]],
                [xy_obs[-1, 1], xy_gt[0, 1]],
                c="blue", linewidth=lw, alpha=0.4, zorder=5)

        ax.scatter([xy_obs[c, 0]], [xy_obs[c, 1]], s=s_center, c="gray", alpha=1.0, zorder=6)

    # Ego
    ax.scatter(ego_obs_xy[:, 0], ego_obs_xy[:, 1], s=s_pt, c="red", label="ego past", zorder=7)
    ax.scatter(ego_gt_xy[:, 0],  ego_gt_xy[:, 1],  s=s_pt, c="green", label="ego future", zorder=7)

    ax.plot(ego_obs_xy[:, 0], ego_obs_xy[:, 1], c="red", linewidth=lw, zorder=7)
    ax.plot(ego_gt_xy[:, 0],  ego_gt_xy[:, 1],  c="green", linewidth=lw, zorder=7)

    ax.plot([ego_obs_xy[-1, 0], ego_gt_xy[0, 0]],
            [ego_obs_xy[-1, 1], ego_gt_xy[0, 1]],
            c="green", linewidth=lw, zorder=7)

    ax.scatter([ego_obs_xy[c, 0]], [ego_obs_xy[c, 1]], s=s_center, c="red", alpha=1.0, zorder=8)

    if ego_frame:
        ax.scatter([0.0], [0.0], s=150, marker="*", label="center (ego)", zorder=9)

    if "scene_id" in data:
        sid = int(data["scene_id"][i])
        scene_name = ETH_UCY_DATASETS[sid] if 0 <= sid < len(ETH_UCY_DATASETS) else f"scene_id={sid}"
        ax.text(0.01, 0.99, f"scene: {scene_name} (id={sid})",
                transform=ax.transAxes, va="top", ha="left",
                fontsize=12, bbox=dict(facecolor="white", alpha=0.75, edgecolor="none"))

    ax.set_aspect("equal", adjustable="box")
    ax.grid(True, alpha=0.1)
    ax.legend()
    plt.tight_layout()

    if save_path is not None:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, bbox_inches="tight")
        plt.close()
        return
    else:
        plt.show()
        plt.close()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--npz", required=True)
    ap.add_argument("--seed", type=int, default=None)
    ap.add_argument("--num_save", type=int, default=20, help="Total de imágenes a guardar")
    ap.add_argument("--out_dir", type=str, default="vis_out", help="Carpeta destino")
    ap.add_argument("--per_scene", type=int, default=None,
                    help="Cuántas por escena. Si no lo pones, se reparte automáticamente.")

    ap.add_argument("--ego_frame", action="store_true")
    ap.add_argument("--rotate", action="store_true")

    ap.add_argument("--show_obstacles", action="store_true")
    ap.add_argument("--map_root", type=str, default=None)
    ap.add_argument("--view_all_obstacles", action="store_true")

    ap.add_argument("--fig_w", type=float, default=24.0)
    ap.add_argument("--fig_h", type=float, default=24.0)
    ap.add_argument("--dpi", type=int, default=160)

    args = ap.parse_args()

    data = print_npz_info(args.npz)
    if "ego" not in data or data["ego"].shape[0] == 0:
        print("No samples in this file.")
        return

    obstacle_cache = None
    if args.show_obstacles:
        if args.map_root is None:
            raise ValueError("Si usas --show_obstacles, debes pasar --map_root")
        obstacle_cache = build_obstacle_cache(args.map_root)

    N = data["ego"].shape[0]
    rng = np.random.default_rng(args.seed)

    # Agrupar índices por escena (si existe scene_id)
    if "scene_id" in data:
        scene_ids = data["scene_id"].astype(np.int64)
        scenes_present = np.unique(scene_ids)

        # decidir cuántas por escena
        if args.per_scene is None:
            # reparto automático: ceil(num_save / #scenes)
            per_scene = int(np.ceil(args.num_save / max(1, len(scenes_present))))
        else:
            per_scene = int(args.per_scene)

        saved = 0
        for sid in rng.permutation(scenes_present):
            idxs = np.where(scene_ids == sid)[0]
            if idxs.size == 0:
                continue
            take = min(per_scene, idxs.size, args.num_save - saved)
            chosen = rng.choice(idxs, size=take, replace=False)

            for idx in chosen:
                scene_name = ETH_UCY_DATASETS[int(sid)]
                out_path = os.path.join(args.out_dir, f"{scene_name}_idx{int(idx):06d}.png")
                plot_sample(
                    data, int(idx),
                    title=f"{scene_name} | idx={int(idx)}",
                    ego_frame=args.ego_frame,
                    rotate=args.rotate,
                    show_obstacles=args.show_obstacles,
                    obstacle_cache=obstacle_cache,
                    view_all_obstacles=args.view_all_obstacles,
                    fig_w=args.fig_w, fig_h=args.fig_h, dpi=args.dpi,
                    save_path=out_path
                )
                saved += 1
                if saved >= args.num_save:
                    break
            if saved >= args.num_save:
                break

        print(f"\n Guardadas {saved} imágenes en: {args.out_dir}")

    else:
        # sin scene_id: solo random global
        take = min(args.num_save, N)
        chosen = rng.choice(np.arange(N), size=take, replace=False)
        for idx in chosen:
            out_path = os.path.join(args.out_dir, f"idx{int(idx):06d}.png")
            plot_sample(
                data, int(idx),
                title=f"idx={int(idx)}",
                ego_frame=args.ego_frame,
                rotate=args.rotate,
                show_obstacles=args.show_obstacles,
                obstacle_cache=obstacle_cache,
                view_all_obstacles=args.view_all_obstacles,
                fig_w=args.fig_w, fig_h=args.fig_h, dpi=args.dpi,
                save_path=out_path
            )
        print(f"\n Guardadas {take} imágenes en: {args.out_dir}")


if __name__ == "__main__":
    main()
