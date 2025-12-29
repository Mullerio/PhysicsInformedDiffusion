import torch
from torch.utils.data import Dataset, DataLoader
from typing import Tuple
import numpy as np


class BaseDataset(Dataset):
    """Base dataset class for diffusion models. Override __len__ and __getitem__ in subclass."""
    
    def __init__(self, split: str = "train"):
        """
        Args:
            split: Dataset split ('train', 'val', 'test')
        """
        self.split = split
        # Generate simple 2D Gaussian mixture data
        self.data = self._generate_gaussian_mixture_data()
        
    def _generate_gaussian_mixture_data(self, num_samples: int = 1000000) -> torch.Tensor:
        """Generate 2D Gaussian mixture point samples.
        
        Args:
            num_samples: Number of points to generate
            
        Returns:
            Tensor of shape (num_samples, 2) - 2D points sampled from Gaussian mixture
        """
        points = []
        
        # Define Gaussian mixture centers
        centers = [
            np.array([-1.5, -1.5]),
            np.array([1.5, -1.5]),
            np.array([-1.5, 1.5]),
            np.array([1.5, 1.5]),
        ]
        
        points_per_component = num_samples // len(centers)
        
        for center in centers:
            # Sample from Gaussian with this center
            component_points = np.random.normal(center, 0.5, size=(points_per_component, 2))
            points.append(component_points)
        
        # Combine all points
        all_points = np.vstack(points)
        
        # Shuffle
        np.random.shuffle(all_points)
        
        # Convert to tensor
        return torch.tensor(all_points, dtype=torch.float32)
        
    def __len__(self) -> int:
        return len(self.data)
    
    def __getitem__(self, idx: int) -> torch.Tensor:
        return self.data[idx]

