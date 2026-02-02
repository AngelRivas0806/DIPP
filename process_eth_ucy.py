"""
Script para procesar datasets ETH/UCY desde CSV a formato .npz
Compatible con el modelo DIPP

Formato de entrada: CSV con columnas [frame, ped_id, x, y]
Formato de salida: .npz con keys [ego, neighbors, gt_future_states]

Uso:
    python process_eth_ucy.py --dataset datasets/ucy-zara01/pixel_pos.csv --output data/processed_data_zara01 --fps 2.5 --split
"""
"""=== Estructura del archivo .npz procesado ===
Keys disponibles: ['ego', 'neighbors', 'gt_future_states']

ego:
  - Shape: (8, 8)
  - Dtype: float32
  - Min/Max: -0.0847 / 0.8875
  - Primeras 2 posiciones:
[[ 0.8875    0.71354  -0.084725  0.      ]
 [ 0.85361   0.71354  -0.084725  0.      ]]

neighbors:
  - Shape: (10, 8, 9)
  - Dtype: float32
  - Min/Max: -0.0855 / 1.0000

gt_future_states:
  - Shape: (11, 12, 8)
  - Dtype: float32
  - Min/Max: -0.0903 / 0.9077"""

import numpy as np
import pandas as pd
import argparse
import os
from tqdm import tqdm
import glob


class ETHUCYProcessor:
    def __init__(self, obs_len=8, pred_len=12, fps=2.5, num_neighbors=10):
        """
        Args:
            obs_len: Número de frames de observación (historia)
            pred_len: Número de frames de predicción (futuro)
            fps: Frames por segundo del dataset
            num_neighbors: Número máximo de vecinos a considerar
        """
        self.obs_len = obs_len
        self.pred_len = pred_len
        self.total_len = obs_len + pred_len
        self.fps = fps
        self.dt = 1.0 / fps  # tiempo entre frames
        self.num_neighbors = num_neighbors
        
    def load_csv(self, csv_path):
        """Cargar CSV con formato [frame, ped_id, x, y]"""
        df = pd.read_csv(csv_path, header=None, names=['frame', 'ped_id', 'x', 'y'])
        return df
    
    def compute_velocity(self, positions):
        """Calcular velocidades a partir de posiciones"""
        if len(positions) < 2:
            return np.zeros((len(positions), 2))
        
        velocities = np.zeros_like(positions)
        velocities[1:] = (positions[1:] - positions[:-1]) / self.dt
        velocities[0] = velocities[1]  # usar la siguiente velocidad para el primer frame
        return velocities
    
    def get_trajectory(self, df, ped_id, start_frame, end_frame, frame_step=10):
        """
        Extraer trayectoria de un peatón en un rango de frames
        
        Args:
            frame_step: Paso entre frames (10 para ETH/UCY que están muestreados cada 10 frames)
        
        Returns:
            traj: array de shape (num_frames, 8) con [x, y, vx, vy, ax, ay, 0, 0]
                  o None si no hay suficientes datos
        """
        frames = np.arange(start_frame, end_frame, frame_step)
        ped_data = df[(df['ped_id'] == ped_id) & (df['frame'].isin(frames))]
        
        # Verificar que tenemos todos los frames
        if len(ped_data) != len(frames):
            return None
        
        # Ordenar por frame
        ped_data = ped_data.sort_values('frame')
        
        # Extraer posiciones
        positions = ped_data[['x', 'y']].values
        
        # Calcular velocidades
        velocities = self.compute_velocity(positions)
        
        # Calcular aceleraciones
        accelerations = self.compute_velocity(velocities)
        
        # Construir array de features [x, y, vx, vy, ax, ay, 0, 0]
        # Los últimos 2 campos son para compatibilidad (pueden ser heading, type, etc.)
        traj = np.zeros((len(frames), 8))
        traj[:, 0:2] = positions
        traj[:, 2:4] = velocities
        traj[:, 4:6] = accelerations
        # traj[:, 6] = 0  # reservado (puede ser heading)
        # traj[:, 7] = 0  # reservado (puede ser tipo de agente)
        
        return traj
    
    def get_neighbors(self, df, ego_id, center_frame, max_neighbors=10, frame_step=10):
        """
        Obtener los N vecinos más cercanos al ego en el frame central
        
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
                # La ultima columna es un flag
                neighbors[i, :, 8] = 1  # flag de validez
        
        return neighbors
    
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
            ped_frames = df[df['ped_id'] == ped_id]['frame'].values
            
            # Deslizar ventana temporal
            # Necesitamos obs_len frames hacia atrás y pred_len hacia adelante
            min_frame = ped_frames.min() + (self.obs_len - 1) * frame_step
            max_frame = ped_frames.max() - self.pred_len * frame_step
            
            # Crear muestras cada ciertos frames (ajustable)
            sampling_step = frame_step * 2  # muestrear cada 2 intervalos (20 frames)
            
            for center_frame in range(min_frame, max_frame + 1, sampling_step):
                if center_frame not in ped_frames:
                    continue
                
                # Extraer trayectoria del ego
                start_frame = center_frame - (self.obs_len - 1) * frame_step
                end_frame = center_frame + (self.pred_len + 1) * frame_step
                
                ego_traj = self.get_trajectory(df, ped_id, start_frame, end_frame, frame_step)
                
                if ego_traj is None:
                    skipped_count += 1
                    continue
                
                # Separar observación y futuro
                ego_obs = ego_traj[:self.obs_len]  # shape: (8, 8)
                ego_future = ego_traj[self.obs_len:]  # shape: (12, 8)
                
                # Obtener vecinos
                neighbors = self.get_neighbors(df, ped_id, center_frame, self.num_neighbors, frame_step)
                
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


def process_all_datasets(datasets_dir, output_base_dir):
    """Procesar todos los datasets encontrados"""
    
    processor = ETHUCYProcessor(obs_len=8, pred_len=12, fps=2.5, num_neighbors=10)
    
    # Buscar todos los CSV
    csv_files = []
    for dataset in ['eth-hotel', 'eth-univ', 'ucy-zara01', 'ucy-zara02', 'ucy-univ']:
        csv_path = f"{datasets_dir}/{dataset}/pixel_pos.csv"
        if os.path.exists(csv_path):
            csv_files.append((dataset, csv_path))
    
    if not csv_files:
        print("No se encontraron archivos CSV")
        return
    
    print(f"\nDatasets encontrados: {len(csv_files)}")
    for name, path in csv_files:
        print(f"  - {name}: {path}")
    
    # Procesar cada dataset
    total_samples = 0
    for dataset_name, csv_path in csv_files:
        output_dir = f"{output_base_dir}/{dataset_name}"
        samples = processor.process_dataset(csv_path, output_dir)
        total_samples += samples
    
    print(f"\n{'='*60}")
    print(f"RESUMEN FINAL")
    print(f"  - Total de muestras generadas: {total_samples}")
    print(f"  - Datasets procesados: {len(csv_files)}")
    print(f"{'='*60}\n")


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
    
    args = parser.parse_args()
    
    if args.process_all:
        # Procesar todos los datasets
        process_all_datasets(args.datasets_dir, args.output)
        if args.split:
            split_processed_data(args.output, args.train_ratio, args.val_ratio, 
                               args.test_ratio, args.seed)
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
        print("\nEjemplos de uso:")
        print("  # Procesar un dataset específico:")
        print("  python process_eth_ucy.py --dataset datasets/ucy-zara01/pixel_pos.csv --output processed_zara01")
        print("\n  # Procesar Y dividir automáticamente:")
        print("  python process_eth_ucy.py --dataset datasets/ucy-zara01/pixel_pos.csv --output processed_zara01 --split")
        print("\n  # Procesar todos los datasets:")
        print("  python process_eth_ucy.py --process_all --output processed_data_eth_ucy")
