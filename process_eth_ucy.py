import os
import argparse
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from tqdm import tqdm

ETH_UCY_DATASETS = ["eth-hotel", "eth-univ", "ucy-zara01", "ucy-zara02", "ucy-univ"]


@dataclass(frozen=True)
class SeqConfig:
    obs_len: int = 8
    pred_len: int = 12
    fps: float = 2.5
    frame_step: int = 10     # ETH/UCY sampleado cada 10 frames
    sample_step: int = 20    # stride del centro (en unidades del CSV)
    k_neighbors: int = 10
    neighbor_radius: float = 3.0  # metros

    @property
    def total_len(self) -> int:
        return self.obs_len + self.pred_len

    @property
    def dt(self) -> np.float32:
        return np.float32(1.0 / self.fps)


def load_csv(csv_path: str) -> pd.DataFrame:
    df = pd.read_csv(csv_path, header=None, names=["frame", "ped_id", "x", "y"])
    df = df.copy()
    df["frame"] = df["frame"].astype(np.int64)
    df["ped_id"] = df["ped_id"].astype(np.int64)
    df["x"] = df["x"].astype(np.float32)
    df["y"] = df["y"].astype(np.float32)
    return df


class TrackIndex:
    """Cache por peatón y por frame para queries rápidas."""
    def __init__(self, df: pd.DataFrame):
        self.ped_tracks: Dict[int, Tuple[np.ndarray, np.ndarray]] = {}
        self.frame_index: Dict[int, Tuple[np.ndarray, np.ndarray]] = {}

        for pid, g in df.groupby("ped_id", sort=False):
            g = g.sort_values("frame")
            frames = g["frame"].to_numpy(np.int64)
            xy = g[["x", "y"]].to_numpy(np.float32)
            self.ped_tracks[int(pid)] = (frames, xy)

        for f, g in df.groupby("frame", sort=False):
            ped_ids = g["ped_id"].to_numpy(np.int64)
            xy = g[["x", "y"]].to_numpy(np.float32)
            self.frame_index[int(f)] = (ped_ids, xy)

    def get_positions(self, ped_id: int, frames_expected: np.ndarray) -> Optional[np.ndarray]:
        data = self.ped_tracks.get(ped_id)
        if data is None:
            return None

        ped_frames, ped_xy = data
        idx = np.searchsorted(ped_frames, frames_expected)

        if np.any(idx >= ped_frames.size):
            return None
        if np.any(ped_frames[idx] != frames_expected):
            return None

        return ped_xy[idx]  # (T,2)


def compute_vel_acc(xy: np.ndarray, dt: np.float32) -> Tuple[np.ndarray, np.ndarray]:
    if xy.shape[0] < 2:
        v = np.zeros_like(xy, dtype=np.float32)
        a = np.zeros_like(xy, dtype=np.float32)
        return v, a

    v = np.zeros_like(xy, dtype=np.float32)
    v[1:] = (xy[1:] - xy[:-1]) / dt
    v[0] = v[1]

    a = np.zeros_like(v, dtype=np.float32)
    a[1:] = (v[1:] - v[:-1]) / dt
    a[0] = a[1]
    return v, a


def xy_to_feat(xy: np.ndarray, dt: np.float32) -> np.ndarray:
    # 6 features: x,y,vx,vy,ax,ay
    v, a = compute_vel_acc(xy, dt)
    feat = np.zeros((xy.shape[0], 6), dtype=np.float32)
    feat[:, 0:2] = xy
    feat[:, 2:4] = v
    feat[:, 4:6] = a
    return feat


def find_dataset_csvs(datasets_dir: str) -> List[Tuple[str, str]]:
    found = []
    for ds in ETH_UCY_DATASETS:
        p = os.path.join(datasets_dir, ds, "mundo", "mun_pos.csv")
        if os.path.exists(p):
            found.append((ds, p))
    return found


def process_csv_ego_only(csv_path: str, cfg: SeqConfig, scene_name: str) -> Dict[str, np.ndarray]:
    """
    Devuelve:
      ego:             (N, obs_len, 6)
      neighbors:       (N, K, obs_len, 7)  (6 feats + valid flag)
      gt_future_states:(N, K+1, pred_len, 6)

    + Metadatos:
      scene_id:        (N,)
      ego_pid:         (N,)
      center_frame:    (N,)
    """
    df = load_csv(csv_path)
    idx = TrackIndex(df)

    ego_samples: List[np.ndarray] = []
    neigh_samples: List[np.ndarray] = []
    gt_samples: List[np.ndarray] = []

    scene_id = ETH_UCY_DATASETS.index(scene_name)
    scene_ids: List[int] = []
    ego_pids: List[int] = []
    center_frames: List[int] = []

    T = cfg.total_len
    fs = cfg.frame_step
    stride = cfg.sample_step
    K = cfg.k_neighbors
    R = float(cfg.neighbor_radius)

    for pid, (ped_frames, _) in tqdm(idx.ped_tracks.items(),
                                     desc=f"ego_only {scene_name}",
                                     leave=False):

        min_center = int(ped_frames.min() + (cfg.obs_len - 1) * fs)
        max_center = int(ped_frames.max() - cfg.pred_len * fs)

        for center in range(min_center, max_center + 1, stride):
            j = np.searchsorted(ped_frames, center)
            if j >= ped_frames.size or ped_frames[j] != center:
                continue

            start = center - (cfg.obs_len - 1) * fs
            end = center + cfg.pred_len * fs
            frames_expected = np.arange(start, end + fs, fs, dtype=np.int64)
            if frames_expected.size != T:
                continue

            ego_xy = idx.get_positions(int(pid), frames_expected)
            if ego_xy is None:
                continue

            center_peds = idx.frame_index.get(int(center), None)
            if center_peds is None:
                continue

            ped_ids_f, xy_f = center_peds
            mask = ped_ids_f != pid
            cand_ids = ped_ids_f[mask]
            cand_xy = xy_f[mask]

            neigh = np.zeros((K, cfg.obs_len, 7), dtype=np.float32)
            gt = np.zeros((K + 1, cfg.pred_len, 6), dtype=np.float32)

            ego_feat = xy_to_feat(ego_xy, cfg.dt)   # (T,6)
            ego_obs = ego_feat[: cfg.obs_len]       # (8,6)
            ego_future = ego_feat[cfg.obs_len :]    # (12,6)
            gt[0] = ego_future

            if cand_ids.size > 0:
                ego_center_xy = ego_xy[cfg.obs_len - 1]
                d = np.linalg.norm(cand_xy - ego_center_xy, axis=1)

                in_r = d <= R
                if np.any(in_r):
                    cand_ids_r = cand_ids[in_r]
                    d_r = d[in_r]

                    order = np.argsort(d_r)
                    keep = order[: min(K, order.size)]
                    neigh_ids = cand_ids_r[keep]

                    for k, nid in enumerate(neigh_ids.tolist()):
                        n_xy = idx.get_positions(int(nid), frames_expected)
                        if n_xy is None:
                            continue
                        n_feat = xy_to_feat(n_xy, cfg.dt)  # (T,6)
                        neigh[k, :, :6] = n_feat[: cfg.obs_len]
                        neigh[k, :, 6] = 1.0
                        gt[k + 1] = n_feat[cfg.obs_len :]

            ego_samples.append(ego_obs)
            neigh_samples.append(neigh)
            gt_samples.append(gt)

            scene_ids.append(scene_id)
            ego_pids.append(int(pid))
            center_frames.append(int(center))

    if len(ego_samples) == 0:
        return {
            "ego": np.empty((0, cfg.obs_len, 6), dtype=np.float32),
            "neighbors": np.empty((0, cfg.k_neighbors, cfg.obs_len, 7), dtype=np.float32),
            "gt_future_states": np.empty((0, cfg.k_neighbors + 1, cfg.pred_len, 6), dtype=np.float32),
            "scene_id": np.empty((0,), dtype=np.int16),
            "ego_pid": np.empty((0,), dtype=np.int64),
            "center_frame": np.empty((0,), dtype=np.int64),
        }

    return {
        "ego": np.stack(ego_samples, axis=0),
        "neighbors": np.stack(neigh_samples, axis=0),
        "gt_future_states": np.stack(gt_samples, axis=0),
        "scene_id": np.asarray(scene_ids, dtype=np.int16),
        "ego_pid": np.asarray(ego_pids, dtype=np.int64),
        "center_frame": np.asarray(center_frames, dtype=np.int64),
    }


def concat_ego_dicts(dicts: List[Dict[str, np.ndarray]], cfg: SeqConfig) -> Dict[str, np.ndarray]:
    dicts = [d for d in dicts if d["ego"].shape[0] > 0]
    if not dicts:
        return {
            "ego": np.empty((0, cfg.obs_len, 6), np.float32),
            "neighbors": np.empty((0, cfg.k_neighbors, cfg.obs_len, 7), np.float32),
            "gt_future_states": np.empty((0, cfg.k_neighbors + 1, cfg.pred_len, 6), np.float32),
            "scene_id": np.empty((0,), dtype=np.int16),
            "ego_pid": np.empty((0,), dtype=np.int64),
            "center_frame": np.empty((0,), dtype=np.int64),
        }

    out = {
        "ego": np.concatenate([d["ego"] for d in dicts], axis=0),
        "neighbors": np.concatenate([d["neighbors"] for d in dicts], axis=0),
        "gt_future_states": np.concatenate([d["gt_future_states"] for d in dicts], axis=0),
        "scene_id": np.concatenate([d["scene_id"] for d in dicts], axis=0),
        "ego_pid": np.concatenate([d["ego_pid"] for d in dicts], axis=0),
        "center_frame": np.concatenate([d["center_frame"] for d in dicts], axis=0),
    }
    return out


def split_indices(n: int, train_ratio: float, val_ratio: float, seed: int):
    rng = np.random.default_rng(seed)
    idx = np.arange(n, dtype=np.int64)
    rng.shuffle(idx)

    r = train_ratio + val_ratio
    n_train = int(n * (train_ratio / r))
    return idx[:n_train], idx[n_train:]


def save_npz(out_path: str, data: Dict[str, np.ndarray]):
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    np.savez_compressed(out_path, **data)


def main():
    p = argparse.ArgumentParser("ETH/UCY ego_only preprocessing (leave-one-out, radius neighbors)")
    p.add_argument("--datasets_dir", type=str, default="datasets")
    p.add_argument("--out_dir", type=str, default="processed_ego_only")
    p.add_argument("--leave_out", type=str, choices=ETH_UCY_DATASETS, default=None)
    p.add_argument("--split", action="store_true")
    p.add_argument("--train_ratio", type=float, default=0.7)
    p.add_argument("--val_ratio", type=float, default=0.15)
    p.add_argument("--seed", type=int, default=42)

    p.add_argument("--obs_len", type=int, default=8)
    p.add_argument("--pred_len", type=int, default=12)
    p.add_argument("--fps", type=float, default=2.5)
    p.add_argument("--frame_step", type=int, default=10)
    p.add_argument("--sample_step", type=int, default=20)
    p.add_argument("--k_neighbors", type=int, default=10)
    p.add_argument("--neighbor_radius", type=float, default=3.0,
                   help="Radio (metros) para considerar vecinos alrededor del ego en frame central")

    args = p.parse_args()

    cfg = SeqConfig(
        obs_len=args.obs_len,
        pred_len=args.pred_len,
        fps=args.fps,
        frame_step=args.frame_step,
        sample_step=args.sample_step,
        k_neighbors=args.k_neighbors,
        neighbor_radius=args.neighbor_radius,
    )

    csvs = find_dataset_csvs(args.datasets_dir)
    if not csvs:
        raise FileNotFoundError("No se encontraron CSVs en datasets_dir.")

    train_csvs = [(name, pth) for (name, pth) in csvs if name != args.leave_out]
    test_csvs = [(name, pth) for (name, pth) in csvs if name == args.leave_out] if args.leave_out else []

    train_parts = [process_csv_ego_only(pth, cfg, scene_name=name) for (name, pth) in train_csvs]
    train_all = concat_ego_dicts(train_parts, cfg)

    if args.split:
        n = train_all["ego"].shape[0]
        tr, va = split_indices(n, args.train_ratio, args.val_ratio, args.seed)

        save_npz(os.path.join(args.out_dir, "train", "data.npz"),
                 {k: v[tr] for k, v in train_all.items()})
        save_npz(os.path.join(args.out_dir, "val", "data.npz"),
                 {k: v[va] for k, v in train_all.items()})
    else:
        save_npz(os.path.join(args.out_dir, "train_full", "data.npz"), train_all)

    if test_csvs:
        test_part = process_csv_ego_only(test_csvs[0][1], cfg, scene_name=test_csvs[0][0])
        save_npz(os.path.join(args.out_dir, "test", "data.npz"), test_part)


if __name__ == "__main__":
    main()