"""Main script for training physics-informed diffusion models."""

import torch
from torch.utils.data import DataLoader
from pathlib import Path
from src.trainer import PIDMTrainer, load_config
from src.networks import NDimensionalMLP, SimpleUNet
from src.data_utils import BaseDataset, MNISTDataset, UnitSphereDataset
from src.residuals import unit_sphere_residual


def train_unit_sphere_2d():
    """Train diffusion model on 2D unit sphere data using MLP."""
    output_dir = Path('sphere2d_runs')
    output_dir.mkdir(exist_ok=True)

    config = load_config('configs/default_pidm_config.yaml')

    model = NDimensionalMLP(in_features=2, out_features=2, time_embed_dim=128)
    trainer = PIDMTrainer(model=model, args=config, output_dir=str(output_dir))

    train_dataset = UnitSphereDataset(num_samples=1000000, dim=2)
    train_loader = DataLoader(
        train_dataset,
        batch_size=config['training']['batch_size'],
        shuffle=True
    )

    val_dataset = UnitSphereDataset(num_samples=100000, dim=2)
    val_loader = DataLoader(
        val_dataset,
        batch_size=config['training']['batch_size'],
        shuffle=False
    )
    
    history = trainer.train_physics(
        train_loader=train_loader,
        residual_fn=unit_sphere_residual,
        val_loader=val_loader,
        num_epochs=config['training']['epochs'],
        c=1.0
    )
    model2 = NDimensionalMLP(in_features=2, out_features=2, time_embed_dim=128)

    trainer2 = PIDMTrainer(model=model2, args=config, output_dir=str(output_dir))

    history = trainer2.train_standard(
        train_loader=train_loader,
        val_loader=val_loader,
        num_epochs=config['training']['epochs']
    )


    print("Training complete!")
    print(f"Final train loss: {history['train'][-1]:.6f}")
    if history['val']:
        print(f"Final val loss: {history['val'][-1]:.6f}")
    print(f"Results saved to {output_dir}/")


def train_gaussian_mixture():
    """Train diffusion model on 2D Gaussian mixture data."""
    # Create output directory
    output_dir = Path('gaussian_runs')
    output_dir.mkdir(exist_ok=True)
    
    # Load configuration
    config = load_config('configs/default_pidm_config.yaml')
    
    # Initialize model for n-dimensional sampling (2D Gaussian mixture example)
    model = NDimensionalMLP(in_features=2, out_features=2, time_embed_dim=128)
    
    # Initialize trainer with output directory
    trainer = PIDMTrainer(model=model, args=config, output_dir=str(output_dir))
    
    # Create datasets and dataloaders
    train_dataset = BaseDataset(split='train')
    train_loader = DataLoader(
        train_dataset, 
        batch_size=config['training']['batch_size'], 
        shuffle=True
    )
    
    val_dataset = BaseDataset(split='val')
    val_loader = DataLoader(
        val_dataset, 
        batch_size=config['training']['batch_size'], 
        shuffle=False
    )
    
    # Train the diffusion model
    history = trainer.train(
        train_loader=train_loader,
        val_loader=val_loader,
        num_epochs=config['training']['epochs']
    )
    
    print("Training complete!")
    print(f"Final train loss: {history['train'][-1]:.6f}")
    if history['val']:
        print(f"Final val loss: {history['val'][-1]:.6f}")
    print(f"Results saved to {output_dir}/")


def train_mnist(
    batch_size: int = 128,
    num_epochs: int = 10,
    learning_rate: float = 1e-3,
    data_path: str = "data",
    pred_type: str = "eps",
    val_split: float = 0.1,
    device: str = "cuda" if torch.cuda.is_available() else "cpu",
):
    """Train U-Net diffusion model on MNIST dataset.
    
    Args:
        batch_size: Batch size for training
        num_epochs: Number of training epochs
        learning_rate: Learning rate for optimizer
        data_path: Path to store/load MNIST data
        val_split: Fraction of training data to use for validation
        device: Device to train on ('cuda' or 'cpu')
    """
    print(f"Training MNIST diffusion model (U-Net) on {device}")
    
    # Create output directory
    output_dir = Path('mnist_runs')
    output_dir.mkdir(exist_ok=True)
    
    # Load configuration
    config = load_config('configs/default_pidm_config.yaml')
    
    # Override with MNIST-specific settings
    config['training']['batch_size'] = batch_size
    config['training']['epochs'] = num_epochs
    config['training']['learning_rate'] = learning_rate
    config['model']['prediction_type'] = pred_type
    config['device']['use_cuda'] = device == 'cuda'
    
    # Initialize U-Net model for MNIST (1 channel input/output, 28x28 images)
    model = SimpleUNet(in_channels=1, out_channels=1, time_embed_dim=128)
    
    # Initialize trainer with output directory
    trainer = PIDMTrainer(model=model, args=config, output_dir=str(output_dir))
    
    # Create MNIST datasets and dataloaders
    print("Loading MNIST dataset...")
    train_dataset = MNISTDataset(split='train', data_path=data_path, val_split=val_split)
    train_loader = DataLoader(
        train_dataset, 
        batch_size=batch_size, 
        shuffle=True,
        num_workers=2,
        collate_fn=_mnist_collate_fn
    )
    
    val_dataset = MNISTDataset(split='val', data_path=data_path, val_split=val_split)
    val_loader = DataLoader(
        val_dataset, 
        batch_size=batch_size, 
        shuffle=False,
        num_workers=2,
        collate_fn=_mnist_collate_fn
    )
    
    print(f"Training set size: {len(train_dataset)}")
    print(f"Validation set size: {len(val_dataset)}")
    
    print("Starting training...")
    history = trainer.train(
        train_loader=train_loader,
        val_loader=val_loader,
        num_epochs=num_epochs
    )
    
    print("\nTraining complete!")
    print(f"Final train loss: {history['train'][-1]:.6f}")
    if history['val']:
        print(f"Final val loss: {history['val'][-1]:.6f}")
    print(f"Results saved to {output_dir}/")
    
    return trainer, history


def _mnist_collate_fn(batch):
    """Collate function to reshape flattened MNIST data to 4D tensors for U-Net."""
    # batch is a list of 1D tensors (784,)
    stacked = torch.stack(batch)  # (batch_size, 784)
    # Reshape to (batch_size, 1, 28, 28)
    reshaped = stacked.view(-1, 1, 28, 28)
    return reshaped


def main():
    """Main entry point - trains on Gaussian mixture by default."""
    train_gaussian_mixture()


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Train diffusion models")
    parser.add_argument('--dataset', type=str, default='gaussian', 
                       choices=['gaussian', 'mnist', 'sphere2d'],
                       help='Dataset to train on')
    parser.add_argument('--batch-size', type=int, default=128, 
                       help='Batch size for training')
    parser.add_argument('--num-epochs', type=int, default=10, 
                       help='Number of training epochs')
    parser.add_argument('--learning-rate', type=float, default=1e-3, 
                       help='Learning rate for optimizer')
    parser.add_argument('--data-path', type=str, default='data', 
                       help='Path to store/load datasets')
    parser.add_argument('--pred_type', type=str, default='eps',
                        choices=['eps', 'x0'])
    
    args = parser.parse_args()
    
    if args.dataset == 'mnist':
        train_mnist(
            batch_size=args.batch_size,
            num_epochs=args.num_epochs,
            learning_rate=args.learning_rate,
            data_path=args.data_path,
            pred_type=args.pred_type    
        )
    elif args.dataset == 'sphere2d':
        train_unit_sphere_2d()
    else:
        train_gaussian_mixture()


