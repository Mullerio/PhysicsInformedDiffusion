import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import datasets, transforms
from typing import Tuple, Optional
import numpy as np
import os


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


class MNISTDataset(Dataset):
    """MNIST dataset for diffusion models.
    
    Loads MNIST images and flattens them to 1D vectors (784-dimensional).
    Normalizes to [0, 1] range.
    """
    
    def __init__(self, split: str = "train", data_path: str = "data", val_split: float = 0.1):
        """
        Args:
            split: Dataset split ('train', 'val', 'test')
            data_path: Path to store/load MNIST data
            val_split: Fraction of training data to use for validation
        """
        self.split = split
        self.data_path = data_path
        os.makedirs(data_path, exist_ok=True)
        
        # Define transforms
        transform = transforms.Compose([
            transforms.ToTensor(),  # Converts to [0, 1] range
        ])
        
        # Load MNIST dataset
        if split == 'test':
            self.mnist = datasets.MNIST(root=data_path, train=False, transform=transform, download=True)
        else:
            # Load full training set
            full_mnist = datasets.MNIST(root=data_path, train=True, transform=transform, download=True)
            
            # Split into train and val
            num_samples = len(full_mnist)
            val_size = int(num_samples * val_split)
            train_size = num_samples - val_size
            
            train_indices = list(range(train_size))
            val_indices = list(range(train_size, num_samples))
            
            if split == 'train':
                self.indices = train_indices
                self.mnist = full_mnist
            elif split == 'val':
                self.indices = val_indices
                self.mnist = full_mnist
            else:
                raise ValueError(f"Unknown split: {split}")
    
    def __len__(self) -> int:
        if self.split == 'test':
            return len(self.mnist)
        else:
            return len(self.indices)
    
    def __getitem__(self, idx: int) -> torch.Tensor:
        if self.split == 'test':
            image, _ = self.mnist[idx]
        else:
            actual_idx = self.indices[idx]
            image, _ = self.mnist[actual_idx]
        
        # Flatten image from (1, 28, 28) to (784,)
        flattened = image.view(-1)
        
        return flattened

