"""
Script to process ETH/UCY datasets from CSV to .npz format


Input format: CSV with columns [frame, ped_id, x, y] in WORLD COORDINATES (meters)
Output format: .npz with keys [ego, neighbors, gt_future_states]

=== Structure of the processed .npz file ===
Available keys: ['ego', 'neighbors', 'gt_future_states']

ego:
  - Shape: (8, 8)
  - Dtype: float32
  - Min/Max: -0.0847 / 0.8875
  - First 2 positions:
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
import shutil


class ETHUCYProcessor:
    def __init__(self, obs_len=8, pred_len=12, fps=2.5, num_neighbors=10):
        """
        Args:
            obs_len: Number of observation frames (history)
            pred_len: NNumber of prediction frames (future)
            fps: Frames per second of the dataset
            num_neighbors: Maximum number of neighbors to consider
        """
        self.obs_len = obs_len
        self.pred_len = pred_len
        self.total_len = obs_len + pred_len
        self.fps = fps
        self.dt = 1.0 / fps  # time between frames
        self.num_neighbors = num_neighbors
        
    def load_csv(self, csv_path):
        """
        Load CSV with format [frame, ped_id, x, y]
        """
        df = pd.read_csv(csv_path, header=None, names=['frame', 'ped_id', 'x', 'y'])
        return df
    
    def compute_velocity(self, positions):
        """Compute velocities from positions"""
        if len(positions) < 2:
            return np.zeros((len(positions), 2))
        
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


def process_all_datasets(datasets_dir, output_base_dir, leave_out=None, split=False, 
                         train_ratio=0.7, val_ratio=0.15, test_ratio=0.15, seed=42):
    """
    Process all found datasets (with option to exclude one for Leave-One-Out)

    Args:
        datasets_dir: Directory with datasets
        output_base_dir: Output directory
        leave_out: Dataset to exclude (for Leave-One-Out). Ej: 'zara01', 'eth-hotel'
        split: If it should split into train/val/test
        train_ratio, val_ratio, test_ratio: Proportions for split
        seed: Seed for reproducibility
    """
    
    processor = ETHUCYProcessor(obs_len=8, pred_len=12, fps=2.5, num_neighbors=10)
    
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
    # PASO 1: Procesar y combinar datasets de entrenamiento
    #=========================================================
    print(f"\n{'='*60}")
    print(f"PASO 1/3: Procesando y combinando datasets de ENTRENAMIENTO")
    print(f"{'='*60}\n")
    
    # Crear directorio temporal combinado
    combined_temp_dir = f"{output_base_dir}/combined_temp"
    os.makedirs(combined_temp_dir, exist_ok=True)
    
    total_samples = 0
    sample_counter = 0
    
    for dataset_name, csv_path in train_csv_files:
        print(f"\nProcesando: {dataset_name}")
        
        # Procesar dataset temporalmente
        temp_output_dir = f"{output_base_dir}/{dataset_name}_temp"
        samples = processor.process_dataset(csv_path, temp_output_dir)
        
        # Copiar archivos al directorio combinado con numeración secuencial
        print(f"Combinando {samples} muestras de {dataset_name}...")
        temp_files = sorted(glob.glob(os.path.join(temp_output_dir, "*.npz")))
        for src_file in tqdm(temp_files, desc=f"  → combined_temp", leave=False):
            dst_file = os.path.join(combined_temp_dir, f"sample_{sample_counter:05d}.npz")
            shutil.move(src_file, dst_file)
            sample_counter += 1
        
        # Eliminar directorio temporal
        os.rmdir(temp_output_dir)
        total_samples += samples
    
    print(f"\n{'='*60}")
    print(f"Datasets combinados: {total_samples} muestras")
    print(f"{'='*60}\n")
    
    #=========================================================
    # PASO 2: Dividir en train/val/test si aplica
    #=========================================================
    if split:
        print(f"{'='*60}")
        print(f"PASO 2/3: Dividiendo datos combinados en train/val")
        print(f"{'='*60}\n")
        
        # Dividir el directorio combinado (solo train y val, sin test)
        train_combined_dir = f"{output_base_dir}/train_combined"
        val_combined_dir = f"{output_base_dir}/val_combined"
        
        os.makedirs(train_combined_dir, exist_ok=True)
        os.makedirs(val_combined_dir, exist_ok=True)
        
        # Obtener todos los archivos
        all_files = sorted(glob.glob(os.path.join(combined_temp_dir, "*.npz")))
        n_total = len(all_files)
        
        # Mezclar con semilla
        np.random.seed(seed)
        indices = np.arange(n_total)
        np.random.shuffle(indices)
        
        # Calcular división (ajustar proporciones ya que no hay test)
        # train_ratio y val_ratio se normalizan para sumar 1.0
        total_ratio = train_ratio + val_ratio
        adjusted_train_ratio = train_ratio / total_ratio
        
        n_train = int(n_total * adjusted_train_ratio)
        
        train_indices = indices[:n_train]
        val_indices = indices[n_train:]
        
        print(f"Total de muestras combinadas: {n_total}")
        print(f"División (seed={seed}):")
        print(f"  - Train: {len(train_indices)} muestras ({len(train_indices)/n_total*100:.1f}%)")
        print(f"  - Val:   {len(val_indices)} muestras ({len(val_indices)/n_total*100:.1f}%)\n")
        
        # Copiar archivos train
        print("Copiando train_combined...")
        for new_idx, original_idx in enumerate(tqdm(train_indices, desc="  train")):
            src_file = all_files[original_idx]
            dst_file = os.path.join(train_combined_dir, f"sample_{new_idx:05d}.npz")
            shutil.copy2(src_file, dst_file)
        
        # Copiar archivos val
        print("Copiando val_combined...")
        for new_idx, original_idx in enumerate(tqdm(val_indices, desc="  val")):
            src_file = all_files[original_idx]
            dst_file = os.path.join(val_combined_dir, f"sample_{new_idx:05d}.npz")
            shutil.copy2(src_file, dst_file)
        
        # Eliminar directorio temporal
        import shutil as sh
        sh.rmtree(combined_temp_dir)
        
        print(f"\n División completada:")
        print(f"  - train_combined/: {len(train_indices)} muestras")
        print(f"  - val_combined/: {len(val_indices)} muestras")
        
        train_counter = len(train_indices)
        val_counter = len(val_indices)
    else:
        # Si no se especifica split, solo renombrar el directorio
        train_combined_dir = f"{output_base_dir}/combined"
        shutil.move(combined_temp_dir, train_combined_dir)
        train_counter = total_samples
        val_counter = 0
    
    #=========================================================
    # PASO 3: Procesar dataset de test (si aplica)
     #=========================================================
    if test_csv_file:
        print(f"\n{'='*60}")
        print(f"PASO 3/3: Procesando dataset de TEST ({test_csv_file[0]})")
        print(f"{'='*60}")
        
        test_dataset_name, test_csv_path = test_csv_file
        test_output_dir = f"{output_base_dir}/test"
        
        test_samples = processor.process_dataset(test_csv_path, test_output_dir)
        
        print(f"\n Dataset de test procesado:")
        print(f"  - test/: {test_samples} muestras")
    
   #=========================================================
    print(f"\n{'='*60}")
    print(f" PROCESAMIENTO COMPLETADO")
    print(f"{'='*60}")
    print(f"\n Estructura creada en: {output_base_dir}/")
    print(f"\n Para ENTRENAMIENTO:")
    if split:
        print(f"  ├── train_combined/        ({train_counter} muestras)")
        print(f"  └── val_combined/          ({val_counter} muestras)")
    else:
        print(f"  └── combined/              ({train_counter} muestras)")
    
    if test_csv_file:
        print(f"\n Para TESTING:")
        print(f"  └── test/   ({test_samples} muestras)")
    
    print(f"\n Próximo paso - Entrenar modelo:")
    if split and test_csv_file:
        print(f"  python train.py \\")
        print(f"    --name leave_{test_dataset_name}_out \\")
        print(f"    --train_set {output_base_dir}/train_combined \\")
        print(f"    --valid_set {output_base_dir}/val_combined \\")
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
            seed=args.seed
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
        print("\n1. Procesar UN dataset específico (usando coordenadas del mundo en metros):")
        print("   python process_eth_ucy.py --dataset datasets/ucy-zara01/mundo/mun_pos.csv --output processed_zara01")
        
        print("\n2. Procesar UN dataset Y dividir en train/val/test:")
        print("   python process_eth_ucy.py --dataset datasets/ucy-zara01/mundo/mun_pos.csv --output processed_zara01 --split")
        
        print("\n3. Procesar TODOS los datasets (busca automáticamente mundo/mun_pos.csv):")
        print("   python process_eth_ucy.py --process_all --output processed_data --split")
        
        print("\n4. Leave-One-Out: Procesar TODOS EXCEPTO zara02 (para testing):")
        print("   python process_eth_ucy.py --process_all --leave_out ucy-zara02 --output processed_data --split")
        
        print("\n5. Leave-One-Out: Excluir eth-hotel:")
        print("   python process_eth_ucy.py --process_all --leave_out eth-hotel --output processed_data --split")
        
        print("\nDatasets disponibles para --leave_out:")
        print("  - eth-hotel")
        print("  - eth-univ")
        print("  - ucy-zara01")
        print("  - ucy-zara02")
        print("  - ucy-univ")
        print("="*60 + "\n")
