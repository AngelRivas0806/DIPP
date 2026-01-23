import numpy as np
import pandas as pd
import argparse
import os
from tqdm import tqdm
import glob
import shutil


class ETHUCYProcessor:
    def __init__(self, obs_len=8, pred_len=12, fps=2.5, num_neighbors=10, use_ego=True):
        """
        Args:
            obs_len: Number of observation frames (history)
            pred_len: Number of prediction frames (future)
            fps: Frames per second of the dataset
            num_neighbors: Maximum number of neighbors to consider
            use_ego: If True, process with ego-centric representation. If False, only positions.
        """
        self.obs_len = obs_len
        self.pred_len = pred_len
        self.total_len = obs_len + pred_len
        self.fps = fps
        self.dt = np.float32(1.0 / fps)  # time between frames
        self.num_neighbors = num_neighbors
        self.use_ego = use_ego
        self._ped_tracks = None
        self._frame_index = None
        
    def load_csv(self, csv_path):
        """
        Load CSV with format [frame, ped_id, x, y]
        """
        df = pd.read_csv(csv_path, header=None, names=['frame', 'ped_id', 'x', 'y'])
        return df
    
    def _build_cache(self, df):
        """
        Build cache structures for fast trajectory access.
        Creates:
        - self._ped_tracks: dict {ped_id: (frames_array, xy_array)}
        - self._frame_index: dict {frame: (ped_ids_array, xy_array)}
        """
        self._ped_tracks = {}
        self._frame_index = {}
        
        # Build per-pedestrian index using groupby
        for ped_id, g in df.groupby('ped_id', sort=False):
            g = g.sort_values('frame')
            frames = g['frame'].values
            xy = g[['x', 'y']].values.astype(np.float32)
            self._ped_tracks[ped_id] = (frames, xy)
        
        # Build per-frame index using groupby
        for frame, g in df.groupby('frame', sort=False):
            ped_ids = g['ped_id'].values
            xy = g[['x', 'y']].values.astype(np.float32)
            self._frame_index[int(frame)] = (ped_ids, xy)
    
    def compute_velocity(self, positions):
        """Compute velocities from positions"""
        if len(positions) < 2:
            return np.zeros((len(positions), 2), dtype=positions.dtype)
        
        velocities = np.zeros_like(positions)
        velocities[1:] = (positions[1:] - positions[:-1]) / self.dt
        velocities[0] = velocities[1]  # use first valid velocity for the first frame
        return velocities
    
    def get_trajectory(self, df, ped_id, start_frame, end_frame, frame_step=10):
        """
        Extract trajectory of a pedestrian over a range of frames

        Args:
            frame_step: Step between frames (10 for ETH/UCY which are sampled every 10 frames)

        Returns:
            traj: array of shape (num_frames, 8) with [x, y, vx, vy, ax, ay, 0, 0]
                  or None if there is not enough data
        """
        frames = np.arange(start_frame, end_frame, frame_step)
        ped_data = df[(df['ped_id'] == ped_id) & (df['frame'].isin(frames))]

        # Check if we have all frames
        if len(ped_data) != len(frames):
            return None

        # Sort by frame
        ped_data = ped_data.sort_values('frame')

        # Extract positions
        positions = ped_data[['x', 'y']].values

        # Compute velocities
        velocities = self.compute_velocity(positions)

        # Compute accelerations
        accelerations = self.compute_velocity(velocities)

        # Build feature array [x, y, vx, vy, ax, ay, 0, 0]
        # The last 2 fields are for compatibility (can be heading, type, etc.)
        traj = np.zeros((len(frames), 8))
        traj[:, 0:2] = positions
        traj[:, 2:4] = velocities
        traj[:, 4:6] = accelerations
        # traj[:, 6] = 0  # reserved (can be heading)
        # traj[:, 7] = 0  # reserved (can be agent type)

        return traj
    
    def get_trajectory_fast(self, ped_id, start_frame, end_frame, frame_step=10):
        """
        Fast version of get_trajectory using cache.
        
        Args:
            ped_id: Pedestrian ID
            start_frame: Start frame
            end_frame: End frame (exclusive)
            frame_step: Step between frames
            
        Returns:
            traj: array of shape (num_frames, 8) or None if insufficient data
        """
        if self._ped_tracks is None or ped_id not in self._ped_tracks:
            return None
        
        frames_expected = np.arange(start_frame, end_frame, frame_step)
        ped_frames, ped_xy = self._ped_tracks[ped_id]
        
        # Find indices of expected frames using searchsorted
        indices = np.searchsorted(ped_frames, frames_expected)
        
        # Validate all frames exist
        if np.any(indices >= len(ped_frames)) or np.any(ped_frames[indices] != frames_expected):
            return None
        
        # Extract positions
        positions = ped_xy[indices]
        
        # Compute velocities
        velocities = self.compute_velocity(positions)
        
        # Compute accelerations
        accelerations = self.compute_velocity(velocities)
        
        # Build feature array with float32
        traj = np.zeros((len(frames_expected), 8), dtype=np.float32)
        traj[:, 0:2] = positions
        traj[:, 2:4] = velocities
        traj[:, 4:6] = accelerations
        
        return traj
    
    def get_neighbors(self, df, ego_id, center_frame, max_neighbors=10, frame_step=10):
        """
        Get the N closest neighbors to the ego in the center frame

        Returns:
            neighbors: array de shape (max_neighbors, total_len, 9) 
                      [x, y, vx, vy, ax, ay, 0, 0, valid_flag]
        """
        start_frame = center_frame - (self.obs_len - 1) * frame_step
        end_frame = center_frame + (self.pred_len + 1) * frame_step
        
        # Obtener posición del ego en el centro
        ego_data = df[(df['ped_id'] == ego_id) & (df['frame'] == center_frame)]
        if len(ego_data) == 0:
            return None
        
        ego_pos = ego_data[['x', 'y']].values[0]
        
        # Encontrar todos los peatones en el frame central
        frame_data = df[(df['frame'] == center_frame) & (df['ped_id'] != ego_id)]
        
        if len(frame_data) == 0:
            # No hay vecinos
            neighbors = np.zeros((max_neighbors, self.total_len, 9))
            return neighbors
        
        # Calcular distancias
        distances = np.linalg.norm(frame_data[['x', 'y']].values - ego_pos, axis=1)
        
        # Ordenar por distancia
        sorted_indices = np.argsort(distances)
        neighbor_ids = frame_data.iloc[sorted_indices]['ped_id'].values[:max_neighbors]
        
        # Extraer trayectorias de vecinos
        neighbors = np.zeros((max_neighbors, self.total_len, 9))
        
        for i, neighbor_id in enumerate(neighbor_ids):
            traj = self.get_trajectory(df, neighbor_id, start_frame, end_frame, frame_step)
            if traj is not None:
                neighbors[i, :, :8] = traj
                neighbors[i, :, 8] = 1  # flag de validez
        
        return neighbors
    
    def get_neighbors_fast(self, ego_id, center_frame, max_neighbors=10, frame_step=10):
        """
        Fast version of get_neighbors using cache.
        
        Returns:
            neighbors: array of shape (max_neighbors, total_len, 9) or None
        """
        start_frame = center_frame - (self.obs_len - 1) * frame_step
        end_frame = center_frame + (self.pred_len + 1) * frame_step
        
        # Get ego position at center frame
        if self._ped_tracks is None or ego_id not in self._ped_tracks:
            return None
        
        ego_frames, ego_xy = self._ped_tracks[ego_id]
        ego_idx = np.searchsorted(ego_frames, center_frame)
        if ego_idx >= len(ego_frames) or ego_frames[ego_idx] != center_frame:
            return None
        
        ego_pos = ego_xy[ego_idx]
        
        # Get all pedestrians in center frame
        if self._frame_index is None or center_frame not in self._frame_index:
            neighbors = np.zeros((max_neighbors, self.total_len, 9), dtype=np.float32)
            return neighbors
        
        frame_ped_ids, frame_xy = self._frame_index[center_frame]
        
        # Filter out ego
        mask = frame_ped_ids != ego_id
        neighbor_ids = frame_ped_ids[mask]
        neighbor_xy = frame_xy[mask]
        
        if len(neighbor_ids) == 0:
            neighbors = np.zeros((max_neighbors, self.total_len, 9), dtype=np.float32)
            return neighbors
        
        # Calculate distances and sort
        distances = np.linalg.norm(neighbor_xy - ego_pos, axis=1)
        sorted_indices = np.argsort(distances)
        neighbor_ids = neighbor_ids[sorted_indices[:max_neighbors]]
        
        # Extract trajectories
        neighbors = np.zeros((max_neighbors, self.total_len, 9), dtype=np.float32)
        
        for i, neighbor_id in enumerate(neighbor_ids):
            traj = self.get_trajectory_fast(neighbor_id, start_frame, end_frame, frame_step)
            if traj is not None:
                neighbors[i, :, :8] = traj
                neighbors[i, :, 8] = 1
        
        return neighbors
    
    def process_dataset_no_ego(self, csv_path, frame_step=10):
        """
        Process dataset without ego-centric representation (only positions for trajectory prediction)
        
        Returns:
            all_samples: List of dictionaries with 'observed' and 'future' trajectories
        """
        print(f"\n{'='*60}")
        print(f"Procesando (SIN EGO): {csv_path}")
        print(f"{'='*60}\n")
        
        # Load data
        df = self.load_csv(csv_path)
        
        print(f"Dataset cargado:")
        print(f"  - Total registros: {len(df)}")
        print(f"  - Frames únicos: {df['frame'].nunique()}")
        print(f"  - Peatones únicos: {df['ped_id'].nunique()}")
        print(f"  - Rango frames: {df['frame'].min()} - {df['frame'].max()}\n")
        
        all_ped_ids = df['ped_id'].unique()
        all_samples = []
        skipped_count = 0
        
        for ped_id in tqdm(all_ped_ids, desc="Procesando peatones"):
            ped_frames = df[df['ped_id'] == ped_id]['frame'].values
            
            min_frame = ped_frames.min() + (self.obs_len - 1) * frame_step
            max_frame = ped_frames.max() - self.pred_len * frame_step
            
            sampling_step = frame_step * 2
            
            for center_frame in range(min_frame, max_frame + 1, sampling_step):
                if center_frame not in ped_frames:
                    continue
                
                start_frame = center_frame - (self.obs_len - 1) * frame_step
                end_frame = center_frame + (self.pred_len + 1) * frame_step
                
                # Get full trajectory
                full_traj = self.get_trajectory(df, ped_id, start_frame, end_frame, frame_step)
                
                if full_traj is None:
                    skipped_count += 1
                    continue
                
                # Extract only positions (x, y)
                positions = full_traj[:, :2]  # shape: (20, 2)
                
                # Split into observed and future
                observed = positions[:self.obs_len]  # shape: (8, 2)
                future = positions[self.obs_len:]    # shape: (12, 2)
                
                all_samples.append({
                    'observed': observed.astype(np.float32),
                    'future': future.astype(np.float32)
                })
        
        print(f"\n{'='*60}")
        print(f"Procesamiento completado!")
        print(f"  - Muestras creadas: {len(all_samples)}")
        print(f"  - Muestras omitidas: {skipped_count}")
        print(f"{'='*60}\n")
        
        return all_samples
    
    def process_dataset_with_ego_consolidated(self, csv_path, frame_step=10):
        """
        Process dataset WITH ego-centric representation but return all samples in memory
        (to be saved later as a single consolidated .npz file)
        
        Returns:
            all_samples: List of dictionaries with 'ego', 'neighbors', 'gt_future_states'
        """
        print(f"\n{'='*60}")
        print(f"Procesando (CON EGO - Consolidado): {csv_path}")
        print(f"{'='*60}\n")
        
        # Load data
        df = self.load_csv(csv_path)
        
        # Build cache once
        self._build_cache(df)
        
        print(f"Dataset cargado:")
        print(f"  - Total registros: {len(df)}")
        print(f"  - Frames únicos: {df['frame'].nunique()}")
        print(f"  - Peatones únicos: {df['ped_id'].nunique()}")
        print(f"  - Rango frames: {df['frame'].min()} - {df['frame'].max()}\n")
        
        all_ped_ids = df['ped_id'].unique()
        all_samples = []
        skipped_count = 0
        
        for ped_id in tqdm(all_ped_ids, desc="Procesando peatones"):
            ped_frames, _ = self._ped_tracks[ped_id]
            
            min_frame = ped_frames.min() + (self.obs_len - 1) * frame_step
            max_frame = ped_frames.max() - self.pred_len * frame_step
            
            sampling_step = frame_step * 2
            
            for center_frame in range(min_frame, max_frame + 1, sampling_step):
                # Use searchsorted for O(log n) lookup
                idx = np.searchsorted(ped_frames, center_frame)
                if idx >= len(ped_frames) or ped_frames[idx] != center_frame:
                    continue
                
                start_frame = center_frame - (self.obs_len - 1) * frame_step
                end_frame = center_frame + (self.pred_len + 1) * frame_step
                
                ego_traj = self.get_trajectory_fast(ped_id, start_frame, end_frame, frame_step)
                
                if ego_traj is None:
                    skipped_count += 1
                    continue
                
                # Separar observación y futuro
                ego_obs = ego_traj[:self.obs_len]  # shape: (8, 8)
                ego_future = ego_traj[self.obs_len:]  # shape: (12, 8)
                
                # Obtener vecinos
                neighbors = self.get_neighbors_fast(ped_id, center_frame, self.num_neighbors, frame_step)
                
                if neighbors is None:
                    skipped_count += 1
                    continue
                
                # Separar vecinos en observación
                neighbors_obs = neighbors[:, :self.obs_len, :]  # shape: (10, 8, 9)
                
                # Crear ground truth futuro para ego y vecinos
                gt_future_states = np.zeros((self.num_neighbors + 1, self.pred_len, 8))
                gt_future_states[0] = ego_future
                
                # Agregar futuros de vecinos
                for i in range(self.num_neighbors):
                    if neighbors[i, self.obs_len:, 8].sum() > 0:  # si el vecino es válido
                        gt_future_states[i + 1] = neighbors[i, self.obs_len:, :8]
                
                all_samples.append({
                    'ego': ego_obs.astype(np.float32),
                    'neighbors': neighbors_obs.astype(np.float32),
                    'gt_future_states': gt_future_states.astype(np.float32)
                })
        
        print(f"\n{'='*60}")
        print(f"Procesamiento completado!")
        print(f"  - Muestras creadas: {len(all_samples)}")
        print(f"  - Muestras omitidas: {skipped_count}")
        print(f"{'='*60}\n")
        
        return all_samples
    
    def process_dataset(self, csv_path, output_dir, frame_step=10):
        """
        Procesar dataset completo y guardar archivos .npz
        
        Args:
            frame_step: Paso entre frames (10 para ETH/UCY)
        """
        print(f"\n{'='*60}")
        print(f"Procesando: {csv_path}")
        print(f"Output: {output_dir}")
        print(f"{'='*60}\n")
        
        # Crear directorio de salida
        os.makedirs(output_dir, exist_ok=True)
        
        # Cargar datos
        df = self.load_csv(csv_path)
        
        # Build cache once
        self._build_cache(df)
        
        print(f"Dataset cargado:")
        print(f"  - Total registros: {len(df)}")
        print(f"  - Frames únicos: {df['frame'].nunique()}")
        print(f"  - Peatones únicos: {df['ped_id'].nunique()}")
        print(f"  - Rango frames: {df['frame'].min()} - {df['frame'].max()}\n")
        
        # Obtener todos los peatones
        all_ped_ids = df['ped_id'].unique()
        
        # Procesar cada peatón como ego
        sample_count = 0
        skipped_count = 0
        
        for ped_id in tqdm(all_ped_ids, desc="Procesando peatones"):
            # Obtener frames donde aparece este peatón
            ped_frames, _ = self._ped_tracks[ped_id]
            
            # Deslizar ventana temporal
            # Necesitamos obs_len frames hacia atrás y pred_len hacia adelante
            min_frame = ped_frames.min() + (self.obs_len - 1) * frame_step
            max_frame = ped_frames.max() - self.pred_len * frame_step
            
            # Crear muestras cada ciertos frames (ajustable)
            sampling_step = frame_step * 2  # muestrear cada 2 intervalos (20 frames)
            
            for center_frame in range(min_frame, max_frame + 1, sampling_step):
                # Use searchsorted for O(log n) lookup
                idx = np.searchsorted(ped_frames, center_frame)
                if idx >= len(ped_frames) or ped_frames[idx] != center_frame:
                    continue
                
                # Extraer trayectoria del ego
                start_frame = center_frame - (self.obs_len - 1) * frame_step
                end_frame = center_frame + (self.pred_len + 1) * frame_step
                
                ego_traj = self.get_trajectory_fast(ped_id, start_frame, end_frame, frame_step)
                
                if ego_traj is None:
                    skipped_count += 1
                    continue
                
                # Separar observación y futuro
                ego_obs = ego_traj[:self.obs_len]  # shape: (8, 8)
                ego_future = ego_traj[self.obs_len:]  # shape: (12, 8)
                
                # Obtener vecinos
                neighbors = self.get_neighbors_fast(ped_id, center_frame, self.num_neighbors, frame_step)
                
                if neighbors is None:
                    skipped_count += 1
                    continue
                
                # Separar vecinos en observación
                neighbors_obs = neighbors[:, :self.obs_len, :]  # shape: (10, 8, 9)
                
                # Crear ground truth futuro para ego y vecinos
                # Shape: (num_agents, pred_len, 8)
                # El primer agente es el ego
                gt_future_states = np.zeros((self.num_neighbors + 1, self.pred_len, 8))
                gt_future_states[0] = ego_future
                
                # Agregar futuros de vecinos
                for i in range(self.num_neighbors):
                    if neighbors[i, self.obs_len:, 8].sum() > 0:  # si el vecino es válido
                        gt_future_states[i + 1] = neighbors[i, self.obs_len:, :8]
                
                # Guardar datos
                dataset_name = os.path.basename(csv_path).replace('.csv', '')
                filename = f"{output_dir}/{dataset_name}_{sample_count:05d}.npz"
                
                np.savez(
                    filename,
                    ego=ego_obs.astype(np.float32),
                    neighbors=neighbors_obs.astype(np.float32),
                    gt_future_states=gt_future_states.astype(np.float32)
                )
                
                sample_count += 1
        
        print(f"\n{'='*60}")
        print(f"Procesamiento completado!")
        print(f"  - Muestras creadas: {sample_count}")
        print(f"  - Muestras omitidas: {skipped_count}")
        print(f"  - Guardadas en: {output_dir}")
        print(f"{'='*60}\n")
        
        return sample_count


def process_all_datasets(datasets_dir, output_base_dir, leave_out=None, split=False, 
                         train_ratio=0.7, val_ratio=0.15, test_ratio=0.15, seed=42, use_ego=True,
                         obs_len=8, pred_len=12, fps=2.5, num_neighbors=10):
    """
    Process all found datasets (with option to exclude one for Leave-One-Out)

    Args:
        datasets_dir: Directory with datasets
        output_base_dir: Output directory
        leave_out: Dataset to exclude (for Leave-One-Out). Ej: 'zara01', 'eth-hotel'
        split: If it should split into train/val/test
        train_ratio, val_ratio, test_ratio: Proportions for split
        seed: Seed for reproducibility
        use_ego: If True, process with ego representation. If False, only positions (no ego)
        obs_len: Observation length
        pred_len: Prediction length
        fps: Frames per second
        num_neighbors: Number of neighbors
    """
    
    processor = ETHUCYProcessor(obs_len=obs_len, pred_len=pred_len, fps=fps, num_neighbors=num_neighbors, use_ego=use_ego)
    
    # All available datasets
    all_datasets = ['eth-hotel', 'eth-univ', 'ucy-zara01', 'ucy-zara02', 'ucy-univ']
    
    # Search for CSV files in world/mun_pos.csv
    all_csv_files = []
    for dataset in all_datasets:
        csv_path = f"{datasets_dir}/{dataset}/mundo/mun_pos.csv"
        if os.path.exists(csv_path):
            all_csv_files.append((dataset, csv_path))
    
    if not all_csv_files:
        print("No se encontraron archivos CSV")
        return
    
    # Separar datasets de entrenamiento y test
    train_csv_files = []
    test_csv_file = None
    
    if leave_out:
        print(f"\n{'='*60}")
        print(f"MODO: Leave-One-Out (excluyendo '{leave_out}' para TEST)")
        print(f"{'='*60}")
        
        for name, path in all_csv_files:
            if name == leave_out:
                test_csv_file = (name, path)
            else:
                train_csv_files.append((name, path))
        
        if len(train_csv_files) == 0:
            print(f"Error: No hay datasets para procesar después de excluir '{leave_out}'")
            return
    else:
        train_csv_files = all_csv_files
    
    print(f"\nDatasets de ENTRENAMIENTO: {len(train_csv_files)}")
    for name, path in train_csv_files:
        print(f"  ✓ {name}")
    
    if test_csv_file:
        print(f"\nDataset de TEST:")
        print(f"{test_csv_file[0]}")
    
    #=========================================================
    # BRANCH: Procesamiento SIN EGO (solo posiciones)
    #=========================================================
    if not use_ego:
        print(f"\n{'='*60}")
        print(f"MODO: SIN EGO (solo posiciones x,y para predicción)")
        print(f"{'='*60}\n")
        
        # Process training datasets and combine into single arrays
        print(f"{'='*60}")
        print(f"Procesando datasets de ENTRENAMIENTO...")
        print(f"{'='*60}\n")
        
        all_train_samples = []
        for dataset_name, csv_path in train_csv_files:
            print(f"\nProcesando: {dataset_name}")
            samples = processor.process_dataset_no_ego(csv_path)
            all_train_samples.extend(samples)
        
        print(f"\n{'='*60}")
        print(f"Total muestras de entrenamiento: {len(all_train_samples)}")
        print(f"{'='*60}\n")
        
        # Convert to numpy arrays
        train_observed = np.array([s['observed'] for s in all_train_samples], dtype=np.float32)
        train_future = np.array([s['future'] for s in all_train_samples], dtype=np.float32)
        
        # Split into train/val if requested
        if split:
            print(f"{'='*60}")
            print(f"Dividiendo en train/val...")
            print(f"{'='*60}\n")
            
            n_total = len(all_train_samples)
            np.random.seed(seed)
            indices = np.arange(n_total)
            np.random.shuffle(indices)
            
            total_ratio = train_ratio + val_ratio
            adjusted_train_ratio = train_ratio / total_ratio
            n_train = int(n_total * adjusted_train_ratio)
            
            train_indices = indices[:n_train]
            val_indices = indices[n_train:]
            
            print(f"División (seed={seed}):")
            print(f"  - Train: {len(train_indices)} muestras ({len(train_indices)/n_total*100:.1f}%)")
            print(f"  - Val:   {len(val_indices)} muestras ({len(val_indices)/n_total*100:.1f}%)\n")
            
            # Create output directories
            os.makedirs(f"{output_base_dir}/train_combined", exist_ok=True)
            os.makedirs(f"{output_base_dir}/val_combined", exist_ok=True)
            
            # Save train split
            print("Guardando train_combined.npz...")
            np.savez(
                f"{output_base_dir}/train_combined/data.npz",
                observed_trajectory=train_observed[train_indices],
                gt_future_trajectory=train_future[train_indices]
            )
            
            # Save val split
            print("Guardando val_combined.npz...")
            np.savez(
                f"{output_base_dir}/val_combined/data.npz",
                observed_trajectory=train_observed[val_indices],
                gt_future_trajectory=train_future[val_indices]
            )
            
            print(f"\n✓ Archivos guardados:")
            print(f"  - {output_base_dir}/train_combined/data.npz ({len(train_indices)} muestras)")
            print(f"  - {output_base_dir}/val_combined/data.npz ({len(val_indices)} muestras)")
        else:
            # Save all as combined
            os.makedirs(f"{output_base_dir}/combined", exist_ok=True)
            print("Guardando combined.npz...")
            np.savez(
                f"{output_base_dir}/combined/data.npz",
                observed_trajectory=train_observed,
                gt_future_trajectory=train_future
            )
            print(f"\n✓ Archivo guardado: {output_base_dir}/combined/data.npz ({len(all_train_samples)} muestras)")
        
        # Process test dataset if exists
        if test_csv_file:
            print(f"\n{'='*60}")
            print(f"Procesando dataset de TEST ({test_csv_file[0]})...")
            print(f"{'='*60}")
            
            test_samples = processor.process_dataset_no_ego(test_csv_file[1])
            test_observed = np.array([s['observed'] for s in test_samples], dtype=np.float32)
            test_future = np.array([s['future'] for s in test_samples], dtype=np.float32)
            
            os.makedirs(f"{output_base_dir}/test", exist_ok=True)
            print("Guardando test.npz...")
            np.savez(
                f"{output_base_dir}/test/data.npz",
                observed_trajectory=test_observed,
                gt_future_trajectory=test_future
            )
            print(f"\n✓ Archivo guardado: {output_base_dir}/test/data.npz ({len(test_samples)} muestras)")
        
        print(f"\n{'='*60}")
        print(f"✓ PROCESAMIENTO COMPLETADO (MODO SIN EGO)")
        print(f"{'='*60}\n")
        print(f"Estructura creada en: {output_base_dir}/")
        print(f"\nFormato de datos:")
        print(f"  - observed_trajectory: (N, 8, 2)  # 8 frames observados, (x,y)")
        print(f"  - gt_future_trajectory: (N, 12, 2) # 12 frames futuros, (x,y)")
        print(f"\nPróximo paso - Entrenar modelo de predicción:")
        print(f"  cd only_prediction")
        print(f"  python train_prediction_only.py \\")
        print(f"    --train_set ../{output_base_dir}/train_combined/data.npz \\")
        print(f"    --valid_set ../{output_base_dir}/val_combined/data.npz")
        print(f"{'='*60}\n")
        
        return train_csv_files
    
    #=========================================================
    # PASO 1: Procesar y combinar datasets de entrenamiento (CON EGO - Consolidado)
    #=========================================================
    print(f"\n{'='*60}")
    print(f"PASO 1/3: Procesando y combinando datasets de ENTRENAMIENTO (CON EGO)")
    print(f"{'='*60}\n")
    
    # Process all training datasets and collect samples in memory
    all_train_samples = []
    for dataset_name, csv_path in train_csv_files:
        print(f"\nProcesando: {dataset_name}")
        samples = processor.process_dataset_with_ego_consolidated(csv_path)
        all_train_samples.extend(samples)
    
    print(f"\n{'='*60}")
    print(f"Total muestras de entrenamiento: {len(all_train_samples)}")
    print(f"{'='*60}\n")
    
    # Convert lists to numpy arrays
    train_ego = np.array([s['ego'] for s in all_train_samples], dtype=np.float32)
    train_neighbors = np.array([s['neighbors'] for s in all_train_samples], dtype=np.float32)
    train_gt = np.array([s['gt_future_states'] for s in all_train_samples], dtype=np.float32)
    
    #=========================================================
    # PASO 2: Dividir en train/val si aplica
    #=========================================================
    if split:
        print(f"{'='*60}")
        print(f"PASO 2/3: Dividiendo datos en train/val")
        print(f"{'='*60}\n")
        
        n_total = len(all_train_samples)
        np.random.seed(seed)
        indices = np.arange(n_total)
        np.random.shuffle(indices)
        
        # Calcular división (sin test, solo train/val)
        total_ratio = train_ratio + val_ratio
        adjusted_train_ratio = train_ratio / total_ratio
        n_train = int(n_total * adjusted_train_ratio)
        
        train_indices = indices[:n_train]
        val_indices = indices[n_train:]
        
        print(f"Total de muestras: {n_total}")
        print(f"División (seed={seed}):")
        print(f"  - Train: {len(train_indices)} muestras ({len(train_indices)/n_total*100:.1f}%)")
        print(f"  - Val:   {len(val_indices)} muestras ({len(val_indices)/n_total*100:.1f}%)\n")
        
        # Create output directories
        os.makedirs(f"{output_base_dir}/train_combined", exist_ok=True)
        os.makedirs(f"{output_base_dir}/val_combined", exist_ok=True)
        
        # Save train split
        print("Guardando train_combined.npz...")
        np.savez(
            f"{output_base_dir}/train_combined/data.npz",
            ego=train_ego[train_indices],
            neighbors=train_neighbors[train_indices],
            gt_future_states=train_gt[train_indices]
        )
        
        # Save val split
        print("Guardando val_combined.npz...")
        np.savez(
            f"{output_base_dir}/val_combined/data.npz",
            ego=train_ego[val_indices],
            neighbors=train_neighbors[val_indices],
            gt_future_states=train_gt[val_indices]
        )
        
        print(f"\n✓ Archivos guardados:")
        print(f"  - {output_base_dir}/train_combined/data.npz ({len(train_indices)} muestras)")
        print(f"  - {output_base_dir}/val_combined/data.npz ({len(val_indices)} muestras)")
        
        train_counter = len(train_indices)
        val_counter = len(val_indices)
    else:
        # Save all as combined
        os.makedirs(f"{output_base_dir}/combined", exist_ok=True)
        print("Guardando combined.npz...")
        np.savez(
            f"{output_base_dir}/combined/data.npz",
            ego=train_ego,
            neighbors=train_neighbors,
            gt_future_states=train_gt
        )
        print(f"\n✓ Archivo guardado: {output_base_dir}/combined/data.npz ({len(all_train_samples)} muestras)")
        train_counter = len(all_train_samples)
        val_counter = 0
    
    #=========================================================
    # PASO 3: Procesar dataset de test (si aplica)
    #=========================================================
    if test_csv_file:
        print(f"\n{'='*60}")
        print(f"PASO 3/3: Procesando dataset de TEST ({test_csv_file[0]})")
        print(f"{'='*60}")
        
        test_dataset_name, test_csv_path = test_csv_file
        test_samples_list = processor.process_dataset_with_ego_consolidated(test_csv_path)
        
        # Convert to numpy arrays
        test_ego = np.array([s['ego'] for s in test_samples_list], dtype=np.float32)
        test_neighbors = np.array([s['neighbors'] for s in test_samples_list], dtype=np.float32)
        test_gt = np.array([s['gt_future_states'] for s in test_samples_list], dtype=np.float32)
        
        os.makedirs(f"{output_base_dir}/test", exist_ok=True)
        print("Guardando test.npz...")
        np.savez(
            f"{output_base_dir}/test/data.npz",
            ego=test_ego,
            neighbors=test_neighbors,
            gt_future_states=test_gt
        )
        
        print(f"\n✓ Archivo guardado: {output_base_dir}/test/data.npz ({len(test_samples_list)} muestras)")
        test_samples = len(test_samples_list)
    
    #=========================================================
    print(f"\n{'='*60}")
    print(f"✓ PROCESAMIENTO COMPLETADO (MODO CON EGO)")
    print(f"{'='*60}")
    print(f"\nEstructura creada en: {output_base_dir}/")
    print(f"\nFormato de datos:")
    print(f"  - ego: (N, 8, 8)              # Observaciones ego [x,y,vx,vy,ax,ay,0,0]")
    print(f"  - neighbors: (N, 10, 8, 9)    # Vecinos + flag de validez")
    print(f"  - gt_future_states: (N, 11, 12, 8)  # Futuros de ego + vecinos")
    print(f"\nPara ENTRENAMIENTO:")
    if split:
        print(f"  ├── train_combined/data.npz   ({train_counter} muestras)")
        print(f"  └── val_combined/data.npz     ({val_counter} muestras)")
    else:
        print(f"  └── combined/data.npz         ({train_counter} muestras)")
    
    if test_csv_file:
        print(f"\nPara TESTING:")
        print(f"  └── test/data.npz             ({test_samples} muestras)")
    
    print(f"\nPróximo paso - Entrenar modelo:")
    if split and test_csv_file:
        print(f"  python train.py \\")
        print(f"    --name leave_{test_dataset_name}_out \\")
        print(f"    --train_set {output_base_dir}/train_combined/data.npz \\")
        print(f"    --valid_set {output_base_dir}/val_combined/data.npz \\")
        print(f"    --use_planning")
    print(f"{'='*60}\n")
    
    return train_csv_files


def split_processed_data(input_dir, train_ratio=0.7, val_ratio=0.15, test_ratio=0.15, seed=42):
    """
    Divide datos procesados en train/val/test
    
    Args:
        input_dir: Directorio con archivos .npz procesados
        train_ratio: Proporción para train (default: 0.7)
        val_ratio: Proporción para validación (default: 0.15)
        test_ratio: Proporción para test (default: 0.15)
        seed: Semilla para reproducibilidad
    
    Returns:
        Tupla con (n_train, n_val, n_test)
    """
    import shutil
    
    print(f"\n{'='*60}")
    print(f"Dividiendo datos en train/val/test")
    print(f"{'='*60}")
    
    # Verificar ratios
    assert abs(train_ratio + val_ratio + test_ratio - 1.0) < 0.001, \
        "Los ratios deben sumar 1.0"
    
    # Obtener todos los archivos .npz
    npz_files = sorted(glob.glob(os.path.join(input_dir, "*.npz")))
    total_samples = len(npz_files)
    
    print(f"Total de muestras: {total_samples}")
    
    if total_samples == 0:
        print(f"Error: No se encontraron archivos .npz en {input_dir}")
        return 0, 0, 0
    
    # Mezclar con semilla fija
    np.random.seed(seed)
    indices = np.arange(total_samples)
    np.random.shuffle(indices)
    
    # Calcular divisiones
    n_train = int(total_samples * train_ratio)
    n_val = int(total_samples * val_ratio)
    
    train_indices = indices[:n_train]
    val_indices = indices[n_train:n_train + n_val]
    test_indices = indices[n_train + n_val:]
    
    print(f"\nDivisión (seed={seed}):")
    print(f"  - Train: {len(train_indices)} muestras ({len(train_indices)/total_samples*100:.1f}%)")
    print(f"  - Val:   {len(val_indices)} muestras ({len(val_indices)/total_samples*100:.1f}%)")
    print(f"  - Test:  {len(test_indices)} muestras ({len(test_indices)/total_samples*100:.1f}%)")
    
    # Crear directorios de salida
    splits = {
        'train': (train_indices, f"{input_dir}_train"),
        'val': (val_indices, f"{input_dir}_val"),
        'test': (test_indices, f"{input_dir}_test")
    }
    
    for split_name, (split_indices, output_dir) in splits.items():
        os.makedirs(output_dir, exist_ok=True)
        
        print(f"\nCopiando {split_name}...")
        for new_idx, original_idx in enumerate(tqdm(split_indices, desc=f"{split_name}")):
            src_file = npz_files[original_idx]
            dst_file = os.path.join(output_dir, f"sample_{new_idx:05d}.npz")
            shutil.copy2(src_file, dst_file)
    
    print(f"\n{'='*60}")
    print("Split finished!")
    print(f"{'='*60}")
    print(f"\nDirectorios creados:")
    print(f"  - {input_dir}_train/  ({len(train_indices)} muestras)")
    print(f"  - {input_dir}_val/    ({len(val_indices)} muestras)")
    print(f"  - {input_dir}_test/   ({len(test_indices)} muestras)")
    print(f"{'='*60}\n")
    
    return len(train_indices), len(val_indices), len(test_indices)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Procesar datasets ETH/UCY a formato .npz')
    parser.add_argument('--dataset', type=str, help='Ruta al archivo CSV específico')
    parser.add_argument('--datasets_dir', type=str, default='datasets', 
                        help='Directorio con todos los datasets')
    parser.add_argument('--output', type=str, default='processed_data_eth_ucy',
                        help='Directorio de salida')
    parser.add_argument('--obs_len', type=int, default=8, help='Longitud de observación')
    parser.add_argument('--pred_len', type=int, default=12, help='Longitud de predicción')
    parser.add_argument('--fps', type=float, default=2.5, help='Frames por segundo')
    parser.add_argument('--num_neighbors', type=int, default=10, help='Número de vecinos')
    parser.add_argument('--process_all', action='store_true', 
                        help='Procesar todos los datasets en datasets_dir')
    parser.add_argument('--leave_out', type=str, choices=['eth-hotel', 'eth-univ', 'ucy-zara01', 'ucy-zara02', 'ucy-univ'],
                        help='Dataset a EXCLUIR del procesamiento (Leave-One-Out). Ej: ucy-zara02')
    parser.add_argument('--split', action='store_true',
                        help='Dividir automáticamente en train/val/test (70/15/15)')
    parser.add_argument('--train_ratio', type=float, default=0.7,
                        help='Proporción para train (default: 0.7)')
    parser.add_argument('--val_ratio', type=float, default=0.15,
                        help='Proporción para validación (default: 0.15)')
    parser.add_argument('--test_ratio', type=float, default=0.15,
                        help='Proporción para test (default: 0.15)')
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed para split (default: 42)')
    parser.add_argument('--no_ego', action='store_true',
                        help='Procesar SIN ego (solo posiciones x,y para predicción de trayectorias)')
    
    args = parser.parse_args()
    
    if args.process_all:
        # Procesar todos los datasets (con opción de leave-one-out)
        process_all_datasets(
            args.datasets_dir, 
            args.output,
            leave_out=args.leave_out,
            split=args.split,
            train_ratio=args.train_ratio,
            val_ratio=args.val_ratio,
            test_ratio=args.test_ratio,
            seed=args.seed,
            use_ego=not args.no_ego  # Invertir el flag
        )
    elif args.dataset:
        # Procesar un dataset específico
        processor = ETHUCYProcessor(
            obs_len=args.obs_len,
            pred_len=args.pred_len,
            fps=args.fps,
            num_neighbors=args.num_neighbors
        )
        processor.process_dataset(args.dataset, args.output)
        
        # Dividir si se especificó --split
        if args.split:
            split_processed_data(args.output, args.train_ratio, args.val_ratio, 
                               args.test_ratio, args.seed)
    else:
        print("Error: Especifica --dataset o usa --process_all")
        print("\n" + "="*60)
        print("EJEMPLOS DE USO")
        print("="*60)
        print("\n=== MODO CON EGO (Para DIPP model con planning) ===")
        print("\n1. Procesar UN dataset específico (coordenadas mundo en metros):")
        print("   python process_eth_ucy.py --dataset datasets/ucy-zara01/mundo/mun_pos.csv --output DIPP_model/data")
        
        print("\n2. Procesar UN dataset Y dividir en train/val/test:")
        print("   python process_eth_ucy.py --dataset datasets/ucy-zara01/mundo/mun_pos.csv --output DIPP_model/data --split")
        
        print("\n3. Leave-One-Out CON EGO (zara02 para testing):")
        print("   python process_eth_ucy.py --process_all --leave_out ucy-zara02 --output DIPP_model/data --split")
        
        print("\n=== MODO SIN EGO (Para predicción de trayectorias) ===")
        print("\n4. Leave-One-Out SIN EGO (solo posiciones x,y):")
        print("   python process_eth_ucy.py --process_all --leave_out ucy-zara02 --output only_prediction/data --split --no_ego")
        
        print("\n5. Procesar TODOS sin ego:")
        print("   python process_eth_ucy.py --process_all --output only_prediction/data --split --no_ego")
        
        print("\n=== DIFERENCIAS ENTRE MODOS ===")
        print("CON EGO (default):")
        print("  - Un archivo .npz consolidado por split")
        print("  - Formato: ego(N,8,8), neighbors(N,10,8,9), gt_future_states(N,11,12,8)")
        print("  - Incluye: velocidades, aceleraciones, vecinos")
        print("  - Uso: DIPP_model/ (predicción + planning)")
        
        print("\nSIN EGO (--no_ego):")
        print("  - Un archivo .npz consolidado por split")
        print("  - Formato: observed_trajectory(N,8,2), gt_future_trajectory(N,12,2)")
        print("  - Solo posiciones (x, y)")
        print("  - Uso: only_prediction/ (solo predicción para comparación)")
        
        print("\nDatasets disponibles para --leave_out:")
        print("  - eth-hotel")
        print("  - eth-univ")
        print("  - ucy-zara01")
        print("  - ucy-zara02")
        print("  - ucy-univ")
        print("="*60 + "\n")
