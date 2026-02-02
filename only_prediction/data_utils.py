import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np

class TrajectoryDataset(Dataset):
    """Dataset para trayectorias sin ego."""
    
    def __init__(self, data_path):
        print(f"Loading test dataset from: {data_path}")
        data = np.load(data_path)
        
        self.observed = torch.FloatTensor(data['observed_trajectory'])
        self.future = torch.FloatTensor(data['gt_future_trajectory'])
        
        print(f"  - Loaded {len(self.observed)} test samples")
        print(f"  - Observed shape: {self.observed.shape}")
        print(f"  - Future shape: {self.future.shape}")
    
    def __len__(self):
        return len(self.observed)
    
    def __getitem__(self, idx):
        return self.observed[idx], self.future[idx], idx


