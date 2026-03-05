import argparse
from typing import Optional

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Circle


def print_npz_info(path: str):
    data = np.load(path, allow_pickle=False)
    print("Keys:", list(data.keys()))
    for k in data.files:
        arr = data[k]
        print(f"{k:20s} shape={arr.shape} dtype={arr.dtype}")
    return data


def print_neighbor_count_stats(data, flag_idx: int = 6):
    """
    Imprime cuántas escenas (samples) tienen 0,1,2,...,K vecinos válidos.
    Asume neighbors shape = (N, K, obs_len, F) y el flag está en neighbors[..., flag_idx].
    """
    if "neighbors" not in data:
        print("[neighbor stats] No existe la key 'neighbors' en el npz.")
        return

    neighbors = data["neighbors"]
    if neighbors.ndim != 4:
        print(f"[neighbor stats] neighbors tiene ndim={neighbors.ndim}, esperado 4.")
        return

    N, K, obs_len, F = neighbors.shape
    if flag_idx < 0 or flag_idx >= F:
        print(f"[neighbor stats] flag_idx={flag_idx} fuera de rango. F={F}")
        return

    # flag_idx is the index of the feature that indicates if the neighbor is valid 
    # For each sample/neighbor, check if any of the history timesteps has a valid flag
    valid = neighbors[:, :, :, flag_idx].sum(axis=2) > 0  # (N,K)
    # Count how many valid neighbors each sample has
    counts = valid.sum(axis=1).astype(np.int64)           # (N,)
    hist = np.bincount(counts, minlength=K + 1)           # (K+1,)

    print("\n=== Neighbor count distribution ===")
    print(f"N samples = {N}, K max = {K}, flag_idx = {flag_idx}")
    for n_neigh in range(K + 1):
        print(f"  {n_neigh:2d} vecinos: {int(hist[n_neigh])}")
    print("==================================\n")


# Same as above, but with a histogram plot
def plot_neighbor_histogram(data, flag_idx: int = 6):
    """
    Grafica histograma de cuántos vecinos válidos hay por sample.
    neighbors: (N,K,obs_len,F), flag en neighbors[..., flag_idx]
    """
    if "neighbors" not in data:
        print("[neighbor hist] No existe la key 'neighbors' en el npz.")
        return

    neighbors = data["neighbors"]
    if neighbors.ndim != 4:
        print(f"[neighbor hist] neighbors tiene ndim={neighbors.ndim}, esperado 4.")
        return

    N, K, obs_len, F = neighbors.shape
    if flag_idx < 0 or flag_idx >= F:
        print(f"[neighbor hist] flag_idx={flag_idx} fuera de rango. F={F}")
        return

    valid = neighbors[:, :, :, flag_idx].sum(axis=2) > 0  # (N,K)
    counts = valid.sum(axis=1).astype(np.int64)           # (N,)

    # bins enteros: 0..K
    edges = np.arange(-0.5, K + 1.5, 1.0)

    plt.figure(figsize=(8, 4), dpi=130)
    plt.hist(counts, bins=edges, rwidth=0.9)
    plt.xticks(range(0, K + 1))
    plt.xlabel("Número de vecinos válidos (por sample)")
    plt.ylabel("Cantidad de samples")
    plt.title(f"Distribución de vecinos válidos (K={K})")
    plt.grid(True, alpha=0.25)
    plt.tight_layout()
    plt.show()


def rotation_matrix(theta: float) -> np.ndarray:
    c, s = np.cos(theta), np.sin(theta)
    return np.array([[c, -s],
                     [s,  c]], dtype=np.float32)

# Center the coordinates at 'origin' and optionally rotate by R (2x2 matrix)
def to_ego_frame(xy: np.ndarray, origin: np.ndarray, R: Optional[np.ndarray]) -> np.ndarray:
    out = xy - origin
    if R is not None:
        out = out @ R.T
    return out

# Plot sample i
def plot_sample(data, i: int, title: str = "", ego_frame: bool = True, rotate: bool = False):
    ego       = data["ego"][i]               # (8,6)
    neighbors = data["neighbors"][i]         # (K,8,7)
    # Ego is always the first "agent" in gt, neighbors are the next K agents.
    gt        = data["gt_future_states"][i]  # (K+1,12,6)

    K       = neighbors.shape[0]
    obs_len = ego.shape[0]
    c       = obs_len - 1

    # Observations of the ego (obs_len,2)
    ego_obs_xy = ego[:, 0:2]
    # GT future states of the ego (pred_len,2)
    ego_gt_xy  = gt[0,:, 0:2]

    # Valid neighbors: those that have at least one valid flag in their history
    neigh_valid  = neighbors[:, :, 6].sum(axis=1) > 0
    # Observations of the neighbors (K, obs_len, 2)
    neigh_obs_xy = neighbors[:, :, 0:2]
    # GT future states of the neighbors (K, pred_len, 2)
    neigh_gt_xy  = gt[1:, :, 0:2]

    origin = ego_obs_xy[c].copy()
    R      = None

    if ego_frame and rotate:
        if c >= 1:
            # Compute the velocity vector from the last two observed positions of the ego
            v = ego_obs_xy[c] - ego_obs_xy[c - 1]
        else:
            v = ego_gt_xy[0] - ego_obs_xy[c]

        norm = np.linalg.norm(v)
        if norm > 1e-6:
            v     = v / norm
            # Minus polar angle to rotate the velocity vector to align with +X axis
            theta = -np.arctan2(v[1], v[0])
            R     = rotation_matrix(theta)

    if ego_frame:
        # Transforms ego/neighbors (both obs and gt) to ego-centered coordinates.
        ego_obs_xy   = to_ego_frame(ego_obs_xy, origin, R)
        ego_gt_xy    = to_ego_frame(ego_gt_xy, origin, R)
        neigh_obs_xy = to_ego_frame(neigh_obs_xy, origin, R)
        neigh_gt_xy  = to_ego_frame(neigh_gt_xy, origin, R)

    plt.figure(figsize=(15, 15), dpi=120)
    ax = plt.gca()
    plt.title(title or f"Sample {i}" + (" (ego-frame)" if ego_frame else ""))

    s_pt = .5
    s_center = 30
    lw = 0.8
    alpha_neigh = 0.5

    # Radio alrededor del ego
    radius_m = 3.0
    center_xy = np.array([0.0, 0.0], dtype=np.float32) if ego_frame else ego_obs_xy[c]

    circle = Circle(
        (float(center_xy[0]), float(center_xy[1])),
        radius=radius_m,
        edgecolor="yellow",
        facecolor="yellow",
        alpha=0.12,
        linewidth=1.5
    )
    ax.add_patch(circle)

    # Plot neighbors
    n_plotted = 0
    for k in range(K):
        if not neigh_valid[k]:
            continue
        n_plotted += 1

        xy_obs = neigh_obs_xy[k]
        xy_gt = neigh_gt_xy[k]

        plt.scatter(xy_obs[:, 0], xy_obs[:, 1], s=s_pt, c="gray", alpha=alpha_neigh)
        plt.scatter(xy_gt[:, 0],  xy_gt[:, 1],  s=s_pt, c="blue", alpha=alpha_neigh)

        plt.plot(xy_obs[:, 0], xy_obs[:, 1], c="gray", linewidth=lw, alpha=0.3)
        plt.plot(xy_gt[:, 0],  xy_gt[:, 1],  c="blue", linewidth=lw, alpha=0.3)

        plt.plot([xy_obs[-1, 0], xy_gt[0, 0]],
                 [xy_obs[-1, 1], xy_gt[0, 1]],
                 c="blue", linewidth=lw, alpha=0.4)

        plt.scatter([xy_obs[c, 0]], [xy_obs[c, 1]], s=s_center, c="gray", alpha=1.0)

    # Ego
    plt.scatter(ego_obs_xy[:, 0], ego_obs_xy[:, 1], s=s_pt, c="red", label="ego past")
    plt.scatter(ego_gt_xy[:, 0],  ego_gt_xy[:, 1],  s=s_pt, c="green", label="ego future")

    plt.plot(ego_obs_xy[:, 0], ego_obs_xy[:, 1], c="red", linewidth=lw)
    plt.plot(ego_gt_xy[:, 0],  ego_gt_xy[:, 1],  c="green", linewidth=lw)

    plt.plot([ego_obs_xy[-1, 0], ego_gt_xy[0, 0]],
             [ego_obs_xy[-1, 1], ego_gt_xy[0, 1]],
             c="green", linewidth=lw)

    plt.scatter([ego_obs_xy[c, 0]], [ego_obs_xy[c, 1]], s=s_center, c="red", alpha=1.0)

    if ego_frame:
        plt.scatter([0.0], [0.0], s=120, marker="*", label="center (ego)")
    else:
        plt.scatter([ego_obs_xy[c, 0]], [ego_obs_xy[c, 1]], s=120, marker="*", label="center (last obs)")

    plt.axis("equal")
    plt.grid(True, alpha=0.1)
    plt.legend()
    plt.tight_layout()
    print(f"Plotted {n_plotted} valid neighbors out of K={K}")
    plt.show()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--npz", required=True, help="Path to data.npz (ego-only format)")
    ap.add_argument("--idx", type=int, default=0, help="Sample index")
    ap.add_argument("--random", action="store_true", help="Pick a random sample index")
    ap.add_argument("--ego_frame", action="store_true", help="Translate ego center to origin")
    ap.add_argument("--rotate", action="store_true", help="Rotate so ego heading points to +X (requires --ego_frame)")
    args = ap.parse_args()

    data = print_npz_info(args.npz)

    if "ego" not in data or data["ego"].shape[0] == 0:
        print("No samples in this file.")
        return

    # stats + hist
    print_neighbor_count_stats(data, flag_idx=6)
    plot_neighbor_histogram(data, flag_idx=6)

    N = data["ego"].shape[0]
    i = np.random.randint(0, N) if args.random else args.idx
    i = int(np.clip(i, 0, N - 1))

    plot_sample(
        data, i,
        title=f"{args.npz} | i={i}/{N-1}",
        ego_frame=args.ego_frame,
        rotate=args.rotate
    )


if __name__ == "__main__":
    main()
