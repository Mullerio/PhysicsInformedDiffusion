"""Main script for training physics-informed diffusion models."""

import torch
from torch.utils.data import DataLoader
from src.trainer import PIDMTrainer, load_config
from src.networks import NDimensionalMLP
from src.data_utils import BaseDataset


def main():
    # Load configuration
    config = load_config('configs/default_pidm_config.yaml')
    
    # Initialize model for n-dimensional sampling (2D Gaussian mixture example)
    model = NDimensionalMLP(in_features=2, out_features=2, time_embed_dim=128)
    
    # Initialize trainer (no loss function needed for diffusion)
    trainer = PIDMTrainer(model=model, args=config)
    
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


if __name__ == "__main__":
    main()

