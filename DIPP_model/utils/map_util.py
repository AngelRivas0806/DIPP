import os
from typing import Dict, List, Tuple, Optional

import numpy as np

ETH_UCY_DATASETS = ["eth-hotel", "eth-univ", "ucy-zara01", "ucy-zara02", "ucy-univ"]
THOR_MAGNI_DATASETS = [
    "THOR_MAGNI_120522_SC3",
    "THOR_MAGNI_130522_SC3",
    "THOR_MAGNI_170522_SC3",
    "THOR_MAGNI_180522_SC3",
]

def load_obstacles_polylines(map_root: str, dataset: str = "eth_ucy") -> Dict[int, List[np.ndarray]]:
    """
    Lee mapa/<scene>/obstacles.txt y regresa polilíneas por scene_id.
    """

    if dataset.lower() in ["eth_ucy", "ethucy", "eth-ucy"]:
        scene_names = ETH_UCY_DATASETS
    elif dataset.lower() in ["thor", "thor_magni", "thor-magni"]:
        scene_names = THOR_MAGNI_DATASETS
    else:
        raise ValueError(f"Dataset no soportado: {dataset}")

    scene_polys: Dict[int, List[np.ndarray]] = {}

    def load_one(path: str) -> List[np.ndarray]:
        if not os.path.exists(path):
            print(f"[WARN] No existe mapa: {path}")
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

    for sid, name in enumerate(scene_names):
        path = os.path.join(map_root, name, "obstacles.txt")
        scene_polys[sid] = load_one(path)

    return scene_polys


def polylines_to_segments(polys: List[np.ndarray]) -> np.ndarray:
    """
    Convierte lista de polilíneas [(Ni,2), ...] a segmentos (S,4) [x1,y1,x2,y2].
    """
    segs = []
    for p in polys:
        p = np.asarray(p, dtype=np.float32)
        if p.shape[0] < 2:
            continue
        # segmentos consecutivos
        a = p[:-1]
        b = p[1:]
        s = np.concatenate([a, b], axis=1)  # (Ni-1,4)
        segs.append(s)
    if not segs:
        return np.zeros((0, 4), dtype=np.float32)
    return np.concatenate(segs, axis=0).astype(np.float32)


def rotation_matrix(theta: float) -> np.ndarray:
    c, s = np.cos(theta), np.sin(theta)
    return np.array([[c, -s],
                     [s,  c]], dtype=np.float32)


def estimate_heading_from_ego_history(ego_xy_hist: np.ndarray, k_back: int = 3, eps: float = 1e-6) -> float:
    """
    ego_xy_hist: (T_obs,2) en world
    heading estimado por dirección de movimiento.
    Si está casi estático, regresa 0.
    """
    T = ego_xy_hist.shape[0]
    j = T - 1
    i = max(0, j - k_back)
    v = ego_xy_hist[j] - ego_xy_hist[i]
    n = float(np.linalg.norm(v))
    if n < eps:
        return 0.0
    return float(np.arctan2(v[1], v[0]))


def point_segment_distance_sq(C: np.ndarray, A: np.ndarray, B: np.ndarray) -> np.ndarray:
    """
    Distancia^2 de punto C (2,) a muchos segmentos A,B (S,2).
    Devuelve (S,)
    """
    AB = B - A
    AC = C[None, :] - A
    ab2 = (AB * AB).sum(axis=1) + 1e-12
    t = (AC * AB).sum(axis=1) / ab2
    t = np.clip(t, 0.0, 1.0)
    P = A + AB * t[:, None]
    d2 = ((P - C[None, :]) ** 2).sum(axis=1)
    return d2


def clip_segment_to_circle(A: np.ndarray, B: np.ndarray, R: float, eps: float = 1e-12) -> Optional[Tuple[np.ndarray, np.ndarray]]:
    """
    Recorta un segmento A->B al círculo centrado en (0,0) de radio R.
    A,B: (2,) en el frame donde el círculo está centrado en 0.
    Regresa (A',B') dentro del círculo o None si no intersecta.
    """
    A = A.astype(np.float32)
    B = B.astype(np.float32)
    d = B - A

    a = float(np.dot(d, d))
    if a < eps:
        # segmento degenerado
        if float(np.dot(A, A)) <= R * R:
            return A, A
        return None

    b = 2.0 * float(np.dot(A, d))
    c = float(np.dot(A, A) - R * R)

    disc = b * b - 4.0 * a * c

    insideA = float(np.dot(A, A)) <= R * R
    insideB = float(np.dot(B, B)) <= R * R

    if disc < 0.0:
        # no intersección
        if insideA and insideB:
            return A, B
        return None

    sqrt_disc = float(np.sqrt(max(disc, 0.0)))
    t1 = (-b - sqrt_disc) / (2.0 * a)
    t2 = (-b + sqrt_disc) / (2.0 * a)
    tmin, tmax = (min(t1, t2), max(t1, t2))

    # intervalo de t dentro del círculo es [tmin,tmax]
    # intersectarlo con [0,1]
    u0 = max(0.0, tmin)
    u1 = min(1.0, tmax)
    if u0 > u1:
        # ambos endpoints dentro podría ocurrir (pero ya se capturó arriba)
        return None

    P0 = A + d * u0
    P1 = A + d * u1

    # Si uno de los endpoints estaba dentro, el intervalo válido podría ser [0,u1] o [u0,1].
    # Evitar segmentos casi cero:
    if float(np.linalg.norm(P1 - P0)) < 1e-6:
        return P0, P0

    return P0.astype(np.float32), P1.astype(np.float32)


def extract_local_segments(
    segs_world: np.ndarray,      # (S,4) en world frame
    ego_center_xy: np.ndarray,   # (2,) en world
    heading: float,              # yaw en rad
    radius: float = 7.0,
    max_segments: int = 128,
    prefilter_margin: float = 1.0
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Devuelve:
      seg_local: (max_segments,4) en ego-frame  [x1,y1,x2,y2]
      mask:      (max_segments,) bool
    Proceso:
      - prefiltra por distancia del midpoint al ego (rápido)
      - distancia punto-segmento para ranking (opcional)
      - clip contra círculo (en ego-frame centrado)
      - pad a max_segments
    """
    S = segs_world.shape[0]
    if S == 0:
        return np.zeros((max_segments, 4), np.float32), np.zeros((max_segments,), np.bool_)

    A = segs_world[:, 0:2]
    B = segs_world[:, 2:4]
    mid = 0.5 * (A + B)

    # prefilter: midpoint dentro de R+margin
    dmid2 = ((mid - ego_center_xy[None, :]) ** 2).sum(axis=1)
    keep = dmid2 <= float((radius + prefilter_margin) ** 2)
    A = A[keep]; B = B[keep]
    if A.shape[0] == 0:
        return np.zeros((max_segments, 4), np.float32), np.zeros((max_segments,), np.bool_)

    # ranking por distancia exacta a segmento
    d2 = point_segment_distance_sq(ego_center_xy.astype(np.float32), A.astype(np.float32), B.astype(np.float32))
    order = np.argsort(d2)

    # ego-frame transform (center + rotate so heading -> +X)
    theta = -heading
    Rm = rotation_matrix(theta)

    out = np.zeros((max_segments, 4), dtype=np.float32)
    mask = np.zeros((max_segments,), dtype=np.bool_)
    count = 0

    for idx in order:
        if count >= max_segments:
            break
        a = A[idx] - ego_center_xy
        b = B[idx] - ego_center_xy
        a = a @ Rm.T
        b = b @ Rm.T

        clipped = clip_segment_to_circle(a, b, radius)
        if clipped is None:
            continue
        a2, b2 = clipped
        out[count, :] = np.array([a2[0], a2[1], b2[0], b2[1]], dtype=np.float32)
        mask[count] = True
        count += 1

    return out, mask