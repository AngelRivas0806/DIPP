#!/usr/bin/env python3
"""
Preprocesamiento THÖR-MAGNI para salida de thor-magni-tools -> formato DIPP/ETH-UCY.

Entrada esperada por CSV:
    Time, frame_id, x, y, z, ag_id, agent_type

Ejemplo:
    0.01,1,-1431.608,-1449.262,696.792,DARKO_Robot,Differential-Teleoperated
    0.41,41,-6968.9775,1581.532,1748.174,Helmet_6,Visitors-Group 3

Salida .npz:
    ego:              (N, obs_len, 6)
    neighbors:        (N, K, obs_len, 7)   # [x,y,vx,vy,ax,ay,valid]
    gt_future_states: (N, K+1, pred_len, 6)
    scene_id:         (N,)
    ego_pid:          (N,)                 # -1 = DARKO_Robot
    center_frame:     (N,)                 # índice temporal interno a 2.5 Hz

Notas:
    - DARKO_Robot se usa como ego.
    - Helmet_* se usan como peatones.
    - LO* se excluye.
    - x,y,z vienen en milímetros desde thor-magni-tools; se convierten a metros.
    - Se recomienda que thor-magni-tools ya haya hecho interpolación, resampling a 400ms
      y smoothing opcional. Por eso aquí normalmente se usa frame_step=1 y smooth_window=1.

Uso típico:
    python preprocess_thor_magni_tools_dipp.py \
      --thor_dir /home/rivas0806/Documentos/DIPP/THOR_MAGNI_processed_tools/Scenario_3 \
      --out_dir processed_thor_magni_sc3_tools \
      --split \
      --obs_len 8 \
      --pred_len 12 \
      --fps 2.5 \
      --frame_step 1 \
      --sample_step 1 \
      --k_neighbors 10 \
      --neighbor_radius 20.0 \
      --smooth_window 1 \
      --min_robot_motion 0.05 \
      --debug_counts
"""

import os
import re
import argparse
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple, Any

import numpy as np
import pandas as pd
from tqdm import tqdm


THOR_MAGNI_SCENE_ORDER = [
    "THOR_MAGNI_120522_SC3",
    "THOR_MAGNI_130522_SC3",
    "THOR_MAGNI_170522_SC3",
    "THOR_MAGNI_180522_SC3",
]


@dataclass(frozen=True)
class SeqConfig:
    obs_len: int = 8
    pred_len: int = 12
    fps: float = 2.5

    # Si la salida de thor-magni-tools ya está a 400ms, usa frame_step=1.
    frame_step: int = 1

    # sample_step=1 -> una ventana nueva cada 0.4s.
    # sample_step=2 -> una ventana nueva cada 0.8s.
    sample_step: int = 1

    k_neighbors: int = 10
    neighbor_radius: float = 20.0
    smooth_window: int = 1
    min_robot_motion: float = 0.05

    @property
    def total_len(self) -> int:
        return self.obs_len + self.pred_len

    @property
    def dt(self) -> np.float32:
        return np.float32(1.0 / self.fps)


def parse_scene_from_filename(path: str) -> str:
    """
    Intenta convertir nombres tipo:
        THOR-Magni_120522_SC3A_R1.csv -> THOR_MAGNI_120522_SC3
        THOR-Magni_130522_SC3B_R2.csv -> THOR_MAGNI_130522_SC3
    Si no encuentra patrón, usa el nombre base sin extensión.
    """
    base = os.path.splitext(os.path.basename(path))[0].upper()
    m = re.search(r"(\d{6})_(SC\d+)[A-Z]?", base)
    if m:
        return f"THOR_MAGNI_{m.group(1)}_{m.group(2)}"
    m = re.search(r"(SC\d+)[A-Z]?", base)
    if m:
        return f"THOR_MAGNI_{m.group(1)}"
    return base


def normalize_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Normaliza nombres de columnas frecuentes de la salida de thor-magni-tools."""
    rename = {}
    for c in df.columns:
        cc = str(c).strip()
        low = cc.lower()
        if low == "time":
            rename[c] = "Time"
        elif low in ["frame_id", "frame", "frameid"]:
            rename[c] = "frame_id"
        elif low == "x":
            rename[c] = "x"
        elif low == "y":
            rename[c] = "y"
        elif low == "z":
            rename[c] = "z"
        elif low in ["ag_id", "agent_id", "id"]:
            rename[c] = "ag_id"
        elif low in ["agent_type", "type", "role"]:
            rename[c] = "agent_type"
        else:
            rename[c] = cc
    return df.rename(columns=rename)


def read_tools_csv(csv_path: str) -> Tuple[pd.DataFrame, str, str]:
    """Lee CSV ya procesado por thor-magni-tools."""
    df = pd.read_csv(
        csv_path,
        na_values=["", "N/A", "NA", "nan", "NaN", "None"],
        low_memory=False,
    )
    df = df.loc[:, ~df.columns.astype(str).str.startswith("Unnamed")]
    df = normalize_columns(df)

    required = ["Time", "frame_id", "x", "y", "ag_id"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Faltan columnas {missing} en {csv_path}. Columnas={list(df.columns)}")

    if "agent_type" not in df.columns:
        df["agent_type"] = "unknown"

    scene_name = parse_scene_from_filename(csv_path)
    file_id = os.path.splitext(os.path.basename(csv_path))[0]
    return df, file_id, scene_name


def build_time_frame_index(df: pd.DataFrame, time_col: str = "Time") -> pd.Series:
    """
    Crea índice temporal consecutivo 0,1,2,... usando los tiempos únicos.
    Esto evita usar frame_id original 1,41,81,... cuando ya está resampleado a 400ms.
    """
    times = pd.to_numeric(df[time_col], errors="coerce")
    unique_times = np.sort(times.dropna().unique())
    # Redondeo para evitar problemas mínimos de float tipo 0.410000000001.
    unique_times_round = np.round(unique_times.astype(np.float64), 6)
    time_to_idx = {t: i for i, t in enumerate(unique_times_round)}
    return times.round(6).map(time_to_idx)


def tools_df_to_tracks(
    df: pd.DataFrame,
    robot_body: str = "DARKO_Robot",
    include_bodies_regex: str = r"^Helmet_",
    exclude_bodies_regex: str = r"^LO\d+",
    use_meters: bool = True,
    use_consecutive_frames: bool = True,
) -> Dict[str, pd.DataFrame]:
    """
    Convierte formato largo de thor-magni-tools a tracks por agente.

    Retorna dict:
        agent_id -> DataFrame[Frame, Time, agent_id, x, y]
    """
    df = df.copy()

    df["Time"] = pd.to_numeric(df["Time"], errors="coerce")
    df["frame_id"] = pd.to_numeric(df["frame_id"], errors="coerce")
    df["x"] = pd.to_numeric(df["x"], errors="coerce")
    df["y"] = pd.to_numeric(df["y"], errors="coerce")
    df["ag_id"] = df["ag_id"].astype(str)
    df["agent_type"] = df["agent_type"].astype(str)

    # Mantener solo filas con tiempo/frame/agente. x,y pueden ser NaN; se eliminan por track.
    df = df.dropna(subset=["Time", "frame_id", "ag_id"]).copy()

    if use_consecutive_frames:
        df["Frame"] = build_time_frame_index(df).astype("Int64")
    else:
        # Útil si quieres usar frame_id original 1,41,81,... con frame_step=40.
        df["Frame"] = df["frame_id"].astype("Int64")

    df = df.dropna(subset=["Frame"]).copy()
    df["Frame"] = df["Frame"].astype(np.int64)

    if use_meters:
        df["x"] = df["x"] / 1000.0
        df["y"] = df["y"] / 1000.0

    include_re = re.compile(include_bodies_regex) if include_bodies_regex else None
    exclude_re = re.compile(exclude_bodies_regex) if exclude_bodies_regex else None

    tracks: Dict[str, pd.DataFrame] = {}

    for aid, g in df.groupby("ag_id", sort=False):
        aid = str(aid)

        # Excluir objetos cargados u otros no-agentes.
        if exclude_re and exclude_re.search(aid):
            continue

        # Solo robot o peatones incluidos por regex.
        is_robot = aid == robot_body
        is_ped = include_re.search(aid) is not None if include_re else True
        if not is_robot and not is_ped:
            continue

        # Si thor-magni-tools dejó huecos mayores a max_nans_interpolate, aquí siguen como NaN.
        # No los inventamos: se eliminan y luego get_positions decide si la ventana es completa.
        g = g.dropna(subset=["x", "y"]).copy()
        if len(g) == 0:
            continue

        out = g[["Frame", "Time", "x", "y"]].copy()
        out = out.sort_values("Frame")
        out = out.drop_duplicates(subset="Frame", keep="first")
        out["agent_id"] = aid
        tracks[aid] = out[["Frame", "Time", "agent_id", "x", "y"]]

    return tracks


class BodyTrackIndex:
    def __init__(self, tracks: Dict[str, pd.DataFrame]):
        self.tracks: Dict[str, Tuple[np.ndarray, np.ndarray]] = {}
        self.frame_index: Dict[int, Tuple[np.ndarray, np.ndarray]] = {}

        all_rows = []
        for aid, g in tracks.items():
            g = g.sort_values("Frame")
            frames = g["Frame"].to_numpy(np.int64)
            xy = g[["x", "y"]].to_numpy(np.float32)
            self.tracks[aid] = (frames, xy)
            all_rows.append(g)

        if all_rows:
            all_df = pd.concat(all_rows, ignore_index=True)
            for f, g in all_df.groupby("Frame", sort=False):
                ids = g["agent_id"].astype(str).to_numpy()
                xy = g[["x", "y"]].to_numpy(np.float32)
                self.frame_index[int(f)] = (ids, xy)

    def get_positions(self, agent_id: str, frames_expected: np.ndarray) -> Optional[np.ndarray]:
        data = self.tracks.get(agent_id)
        if data is None:
            return None
        frames, xy = data
        idx = np.searchsorted(frames, frames_expected)
        if np.any(idx >= frames.size):
            return None
        if np.any(frames[idx] != frames_expected):
            return None
        return xy[idx]


def moving_average_xy(xy: np.ndarray, window: int) -> np.ndarray:
    if window is None or window <= 1 or xy.shape[0] < window:
        return xy.astype(np.float32)
    if window % 2 == 0:
        window += 1
    pad = window // 2
    padded = np.pad(xy, ((pad, pad), (0, 0)), mode="edge")
    kernel = np.ones(window, dtype=np.float32) / float(window)
    smoothed = np.zeros_like(xy, dtype=np.float32)
    for j in range(2):
        smoothed[:, j] = np.convolve(padded[:, j], kernel, mode="valid")
    return smoothed.astype(np.float32)


def compute_vel_acc(xy: np.ndarray, dt: np.float32) -> Tuple[np.ndarray, np.ndarray]:
    v = np.zeros_like(xy, dtype=np.float32)
    a = np.zeros_like(xy, dtype=np.float32)
    if xy.shape[0] < 2:
        return v, a
    v[1:] = (xy[1:] - xy[:-1]) / dt
    v[0] = v[1]
    a[1:] = (v[1:] - v[:-1]) / dt
    a[0] = a[1]
    return v, a


def xy_to_feat(xy: np.ndarray, dt: np.float32, smooth_window: int = 1) -> np.ndarray:
    xy = xy.astype(np.float32)
    xy_s = moving_average_xy(xy, smooth_window)
    v, a = compute_vel_acc(xy_s, dt)
    feat = np.zeros((xy.shape[0], 6), dtype=np.float32)
    feat[:, 0:2] = xy_s
    feat[:, 2:4] = v
    feat[:, 4:6] = a
    return feat


def process_tools_csv(
    csv_path: str,
    cfg: SeqConfig,
    scene_to_id: Dict[str, int],
    robot_body: str = "DARKO_Robot",
    include_bodies_regex: str = r"^Helmet_",
    exclude_bodies_regex: str = r"^LO\d+",
    use_consecutive_frames: bool = True,
    debug_counts: bool = False,
) -> Dict[str, np.ndarray]:
    df, file_id, scene_name = read_tools_csv(csv_path)
    scene_id = scene_to_id.setdefault(scene_name, len(scene_to_id))

    tracks = tools_df_to_tracks(
        df,
        robot_body=robot_body,
        include_bodies_regex=include_bodies_regex,
        exclude_bodies_regex=exclude_bodies_regex,
        use_meters=True,
        use_consecutive_frames=use_consecutive_frames,
    )

    if robot_body not in tracks:
        available = sorted(tracks.keys())
        raise ValueError(f"No encontré {robot_body} en {csv_path}. Agentes disponibles: {available[:30]}")

    idx = BodyTrackIndex(tracks)

    ego_samples: List[np.ndarray] = []
    neigh_samples: List[np.ndarray] = []
    gt_samples: List[np.ndarray] = []
    scene_ids: List[int] = []
    ego_pids: List[int] = []
    center_frames: List[int] = []

    T = cfg.total_len
    fs = cfg.frame_step
    stride = cfg.sample_step
    K = cfg.k_neighbors
    R = float(cfg.neighbor_radius)

    robot_frames, _ = idx.tracks[robot_body]
    min_center = int(robot_frames.min() + (cfg.obs_len - 1) * fs)
    max_center = int(robot_frames.max() - cfg.pred_len * fs)

    # Debug acumulado por CSV.
    dbg_center_candidates: List[int] = []
    dbg_in_radius: List[int] = []
    dbg_full_valid: List[int] = []
    dbg_saved: List[int] = []

    for center in tqdm(range(min_center, max_center + 1, stride), desc=f"{file_id}", leave=False):
        j = np.searchsorted(robot_frames, center)
        if j >= robot_frames.size or robot_frames[j] != center:
            continue

        start = center - (cfg.obs_len - 1) * fs
        end = center + cfg.pred_len * fs
        frames_expected = np.arange(start, end + fs, fs, dtype=np.int64)
        if frames_expected.size != T:
            continue

        ego_xy = idx.get_positions(robot_body, frames_expected)
        if ego_xy is None:
            continue

        if cfg.min_robot_motion > 0:
            dist_total = float(np.linalg.norm(ego_xy[-1] - ego_xy[0]))
            if dist_total < cfg.min_robot_motion:
                continue

        center_agents = idx.frame_index.get(int(center), None)
        if center_agents is None:
            continue

        agent_ids_f, xy_f = center_agents
        mask = agent_ids_f != robot_body
        cand_ids = agent_ids_f[mask]
        cand_xy = xy_f[mask]

        neigh = np.zeros((K, cfg.obs_len, 7), dtype=np.float32)
        gt = np.zeros((K + 1, cfg.pred_len, 6), dtype=np.float32)

        ego_feat = xy_to_feat(ego_xy, cfg.dt, cfg.smooth_window)
        ego_obs = ego_feat[: cfg.obs_len]
        ego_future = ego_feat[cfg.obs_len :]
        gt[0] = ego_future

        num_center = int(cand_ids.size)
        num_radius = 0
        num_full = 0
        num_saved = 0

        if cand_ids.size > 0:
            ego_center_xy = ego_xy[cfg.obs_len - 1]
            d = np.linalg.norm(cand_xy - ego_center_xy, axis=1)
            in_r = d <= R
            num_radius = int(np.sum(in_r))

            if np.any(in_r):
                cand_ids_r = cand_ids[in_r]
                d_r = d[in_r]
                order = np.argsort(d_r)

                out_k = 0
                for oi in order:
                    if out_k >= K:
                        break

                    nid = str(cand_ids_r[oi])
                    n_xy = idx.get_positions(nid, frames_expected)
                    if n_xy is None:
                        continue

                    num_full += 1
                    n_feat = xy_to_feat(n_xy, cfg.dt, cfg.smooth_window)
                    neigh[out_k, :, :6] = n_feat[: cfg.obs_len]
                    neigh[out_k, :, 6] = 1.0
                    gt[out_k + 1] = n_feat[cfg.obs_len :]
                    out_k += 1

                num_saved = out_k

        dbg_center_candidates.append(num_center)
        dbg_in_radius.append(num_radius)
        dbg_full_valid.append(num_full)
        dbg_saved.append(num_saved)

        ego_samples.append(ego_obs)
        neigh_samples.append(neigh)
        gt_samples.append(gt)
        scene_ids.append(scene_id)
        ego_pids.append(-1)
        center_frames.append(int(center))

    if debug_counts and len(dbg_saved) > 0:
        def stats(xs: List[int]) -> str:
            arr = np.asarray(xs, dtype=np.float32)
            return f"mean={arr.mean():.2f}, min={arr.min():.0f}, max={arr.max():.0f}"

        print(f"\n[DEBUG] {file_id} | scene={scene_name}")
        print(f"  windows saved:       {len(dbg_saved)}")
        print(f"  center candidates:   {stats(dbg_center_candidates)}")
        print(f"  in radius:           {stats(dbg_in_radius)}")
        print(f"  full-window valid:   {stats(dbg_full_valid)}")
        print(f"  saved neighbors:     {stats(dbg_saved)}")

    if len(ego_samples) == 0:
        return empty_output(cfg)

    return {
        "ego": np.stack(ego_samples, axis=0),
        "neighbors": np.stack(neigh_samples, axis=0),
        "gt_future_states": np.stack(gt_samples, axis=0),
        "scene_id": np.asarray(scene_ids, dtype=np.int16),
        "ego_pid": np.asarray(ego_pids, dtype=np.int64),
        "center_frame": np.asarray(center_frames, dtype=np.int64),
    }


def empty_output(cfg: SeqConfig) -> Dict[str, np.ndarray]:
    return {
        "ego": np.empty((0, cfg.obs_len, 6), dtype=np.float32),
        "neighbors": np.empty((0, cfg.k_neighbors, cfg.obs_len, 7), dtype=np.float32),
        "gt_future_states": np.empty((0, cfg.k_neighbors + 1, cfg.pred_len, 6), dtype=np.float32),
        "scene_id": np.empty((0,), dtype=np.int16),
        "ego_pid": np.empty((0,), dtype=np.int64),
        "center_frame": np.empty((0,), dtype=np.int64),
    }


def concat_dicts(dicts: List[Dict[str, np.ndarray]], cfg: SeqConfig) -> Dict[str, np.ndarray]:
    dicts = [d for d in dicts if d["ego"].shape[0] > 0]
    if not dicts:
        return empty_output(cfg)
    return {k: np.concatenate([d[k] for d in dicts], axis=0) for k in dicts[0].keys()}


def save_npz(out_path: str, data: Dict[str, np.ndarray]) -> None:
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    np.savez_compressed(out_path, **data)
    print(f"Saved {out_path}")
    print("  ego:", data["ego"].shape)
    print("  neighbors:", data["neighbors"].shape)
    print("  gt_future_states:", data["gt_future_states"].shape)


def find_csvs(root: str) -> List[str]:
    out = []
    for dirpath, _, filenames in os.walk(root):
        for fn in filenames:
            if fn.lower().endswith(".csv"):
                out.append(os.path.join(dirpath, fn))
    return sorted(out)


def split_indices(n: int, train_ratio: float, val_ratio: float, seed: int) -> Tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    idx = np.arange(n, dtype=np.int64)
    rng.shuffle(idx)
    r = train_ratio + val_ratio
    n_train = int(n * (train_ratio / r))
    return idx[:n_train], idx[n_train:]


def split_indices3(
    n: int,
    train_ratio: float,
    val_ratio: float,
    test_ratio: float,
    seed: int,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    if n <= 0:
        empty = np.empty((0,), dtype=np.int64)
        return empty, empty, empty

    ratios = np.asarray([train_ratio, val_ratio, test_ratio], dtype=np.float64)
    if np.any(ratios < 0) or ratios.sum() <= 0:
        raise ValueError("train_ratio, val_ratio y test_ratio deben ser no negativos y sumar > 0")
    ratios = ratios / ratios.sum()

    rng = np.random.default_rng(seed)
    idx = np.arange(n, dtype=np.int64)
    rng.shuffle(idx)

    n_train = int(round(n * ratios[0]))
    n_val = int(round(n * ratios[1]))
    n_train = min(n_train, n)
    n_val = min(n_val, n - n_train)

    tr = idx[:n_train]
    va = idx[n_train : n_train + n_val]
    te = idx[n_train + n_val :]
    return tr, va, te


def select_by_scene(data: Dict[str, np.ndarray], scene_ids: List[int]) -> Dict[str, np.ndarray]:
    if len(scene_ids) == 0:
        return {k: v[:0] for k, v in data.items()}
    mask = np.isin(data["scene_id"], np.asarray(scene_ids, dtype=np.int16))
    return {k: v[mask] for k, v in data.items()}


def select_not_by_scene(data: Dict[str, np.ndarray], scene_ids: List[int]) -> Dict[str, np.ndarray]:
    if len(scene_ids) == 0:
        return data
    mask = ~np.isin(data["scene_id"], np.asarray(scene_ids, dtype=np.int16))
    return {k: v[mask] for k, v in data.items()}


def main() -> None:
    p = argparse.ArgumentParser("THÖR-MAGNI tools-output preprocessing compatible with DIPP/ETH-UCY npz format")
    p.add_argument("--thor_dir", type=str, required=True, help="Carpeta raíz con CSVs procesados por thor-magni-tools")
    p.add_argument("--out_dir", type=str, default="processed_thor_magni_tools")

    p.add_argument("--robot_body", type=str, default="DARKO_Robot")
    p.add_argument("--include_bodies_regex", type=str, default=r"^Helmet_", help="Qué ag_id usar como peatones")
    p.add_argument("--exclude_bodies_regex", type=str, default=r"^LO\d+", help="Qué ag_id excluir")

    p.add_argument("--obs_len", type=int, default=8)
    p.add_argument("--pred_len", type=int, default=12)
    p.add_argument("--fps", type=float, default=2.5)
    p.add_argument("--frame_step", type=int, default=1, help="Si usas índice consecutivo a 2.5Hz, usar 1")
    p.add_argument("--sample_step", type=int, default=1)
    p.add_argument("--k_neighbors", type=int, default=10)
    p.add_argument("--neighbor_radius", type=float, default=20.0)
    p.add_argument("--smooth_window", type=int, default=1, help="Usar 1 si thor-magni-tools ya suavizó")
    p.add_argument("--min_robot_motion", type=float, default=0.05)

    p.add_argument("--use_original_frame_id", action="store_true", help="Usa frame_id original 1,41,81,... En ese caso usa frame_step=40")
    p.add_argument("--debug_counts", action="store_true")

    p.add_argument("--split", action="store_true")
    p.add_argument("--split_by_scene", action="store_true", help="Split por escenas/días, no por muestras")
    p.add_argument("--test_scenes", type=str, default="")
    p.add_argument("--val_scenes", type=str, default="")
    p.add_argument("--train_ratio", type=float, default=0.7)
    p.add_argument("--val_ratio", type=float, default=0.15)
    p.add_argument("--test_ratio", type=float, default=0.15)
    p.add_argument("--seed", type=int, default=42)

    args = p.parse_args()

    cfg = SeqConfig(
        obs_len=args.obs_len,
        pred_len=args.pred_len,
        fps=args.fps,
        frame_step=args.frame_step,
        sample_step=args.sample_step,
        k_neighbors=args.k_neighbors,
        neighbor_radius=args.neighbor_radius,
        smooth_window=args.smooth_window,
        min_robot_motion=args.min_robot_motion,
    )

    csvs = find_csvs(args.thor_dir)
    if not csvs:
        raise FileNotFoundError(f"No encontré CSVs en {args.thor_dir}")

    scene_to_id: Dict[str, int] = {name: i for i, name in enumerate(THOR_MAGNI_SCENE_ORDER)}
    parts = []
    for csv_path in csvs:
        try:
            part = process_tools_csv(
                csv_path=csv_path,
                cfg=cfg,
                scene_to_id=scene_to_id,
                robot_body=args.robot_body,
                include_bodies_regex=args.include_bodies_regex,
                exclude_bodies_regex=args.exclude_bodies_regex,
                use_consecutive_frames=not args.use_original_frame_id,
                debug_counts=args.debug_counts,
            )
            parts.append(part)
        except Exception as e:
            print(f"[WARN] Saltando {csv_path}: {e}")

    all_data = concat_dicts(parts, cfg)
    print("Scene mapping:", scene_to_id)
    print("Total samples:", all_data["ego"].shape[0])

    if not args.split:
        save_npz(os.path.join(args.out_dir, "train_full", "data.npz"), all_data)
        return

    if args.split_by_scene:
        inv = {v: k for k, v in scene_to_id.items()}
        test_names = [s.strip().upper() for s in args.test_scenes.split(",") if s.strip()]
        val_names = [s.strip().upper() for s in args.val_scenes.split(",") if s.strip()]

        name_to_id = {k.upper(): v for k, v in scene_to_id.items()}
        test_ids = [name_to_id[s] for s in test_names if s in name_to_id]
        val_ids = [name_to_id[s] for s in val_names if s in name_to_id]

        test_data = select_by_scene(all_data, test_ids)
        remaining = select_not_by_scene(all_data, test_ids)

        if val_ids:
            val_data = select_by_scene(remaining, val_ids)
            train_data = select_not_by_scene(remaining, val_ids)
        else:
            n = remaining["ego"].shape[0]
            tr, va = split_indices(n, args.train_ratio, args.val_ratio, args.seed)
            train_data = {k: v[tr] for k, v in remaining.items()}
            val_data = {k: v[va] for k, v in remaining.items()}

        save_npz(os.path.join(args.out_dir, "train", "data.npz"), train_data)
        save_npz(os.path.join(args.out_dir, "val", "data.npz"), val_data)
        if test_data["ego"].shape[0] > 0:
            save_npz(os.path.join(args.out_dir, "test", "data.npz"), test_data)
        else:
            print("[INFO] No se guardó test porque no hubo muestras o no coincidió test_scenes.")

        print("Scene ids usados:")
        for sid, sname in inv.items():
            print(f"  {sid}: {sname}")
        return

    n = all_data["ego"].shape[0]
    tr, va, te = split_indices3(n, args.train_ratio, args.val_ratio, args.test_ratio, args.seed)

    print(
        f"Random split samples: train={len(tr)}, val={len(va)}, test={len(te)} "
        f"(ratios={args.train_ratio}/{args.val_ratio}/{args.test_ratio})"
    )

    save_npz(os.path.join(args.out_dir, "train", "data.npz"), {k: v[tr] for k, v in all_data.items()})
    save_npz(os.path.join(args.out_dir, "val", "data.npz"), {k: v[va] for k, v in all_data.items()})
    save_npz(os.path.join(args.out_dir, "test", "data.npz"), {k: v[te] for k, v in all_data.items()})


if __name__ == "__main__":
    main()
