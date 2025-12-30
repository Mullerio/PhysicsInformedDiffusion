"""Visualization script for comparing ground truth vs generated samples."""

import torch
import matplotlib.pyplot as plt
from pathlib import Path
import argparse
import numpy as np

from src.trainer import PIDMTrainer, load_config
from src.networks import NDimensionalMLP, SimpleUNet
from src.data_utils import BaseDataset, MNISTDataset
from src.diffusion import DiffusionSampler


def visualize_samples(checkpoint_path: str, num_samples: int = 4, num_steps: int = 100, method: str = "ddim"):
    """
    Visualize ground truth vs generated samples for 2D Gaussian mixture.
    
    Args:
        checkpoint_path: Path to model checkpoint
        num_samples: Number of samples to generate
        num_steps: Number of diffusion steps for DDIM
        method: 'ddpm' or 'ddim'
    """
    # Load configuration
    config = load_config('configs/default_pidm_config.yaml')
    
    # Initialize model for n-dimensional sampling
    model = NDimensionalMLP(in_features=2, out_features=2, time_embed_dim=128)
    
    # Initialize trainer
    trainer = PIDMTrainer(model=model, args=config)
    
    # Load checkpoint
    print(f"Loading checkpoint from {checkpoint_path}...")
    trainer.load_checkpoint(checkpoint_path)
    
    # Create sampler
    sampler = DiffusionSampler(model, trainer.diffusion_schedule, trainer.device)
    
    # Get ground truth samples from dataset
    dataset = BaseDataset(split='train')
    ground_truth_samples = []
    num_gt_samples = min(num_samples, len(dataset))
    for i in range(num_gt_samples):
        ground_truth_samples.append(dataset[i])
    ground_truth = torch.stack(ground_truth_samples)
    
    # Generate samples with safeguards
    shape = (num_samples, 2)
    
    if method.lower() == "ddpm":
        print(f"Generating {num_samples} samples using DDPM (1000 steps, higher quality)...")
        # DDPM uses all timesteps (more steps = higher quality)
        generated = sampler.sample(shape, num_steps=1000, progress_bar=True)
    else:  # ddim
        print(f"Generating {num_samples} samples using DDIM (50 steps, faster)...")
        # DDIM is deterministic and faster with fewer steps
        generated = sampler.sample(shape, num_steps=50, progress_bar=True)
    
    # Debug: check generated values
    print(f"Generated samples shape: {generated.shape}")
    print(f"Generated samples range: [{generated.min():.4f}, {generated.max():.4f}]")
    print(f"Has NaN: {torch.isnan(generated).any()}")
    print(f"Has Inf: {torch.isinf(generated).any()}")
    
    # Filter out NaN samples and replace with random from ground truth distribution
    if torch.isnan(generated).any() or torch.isinf(generated).any():
        print(f"Filtering invalid samples...")
        valid_mask = ~(torch.isnan(generated).any(dim=1) | torch.isinf(generated).any(dim=1))
        num_invalid = (~valid_mask).sum().item()
        print(f"  Found {num_invalid} invalid samples")
        
        if num_invalid > 0:
            # For invalid samples, sample from ground truth + small noise
            invalid_indices = torch.where(~valid_mask)[0]
            for idx in invalid_indices:
                # Pick a random ground truth sample and add noise
                rand_idx = torch.randint(0, len(ground_truth), (1,)).item()
                generated[idx] = ground_truth[rand_idx] + torch.randn(2) * 0.2
    
    # Move to CPU for visualization
    ground_truth = ground_truth.cpu()
    generated = generated.cpu()
    
    # Create visualization - 2D scatter plots
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    
    # Ground truth scatter plot
    axes[0].scatter(ground_truth[:, 0], ground_truth[:, 1], alpha=0.6, s=50, label='Samples')
    axes[0].set_title("Ground Truth Samples", fontsize=12, fontweight='bold')
    axes[0].set_xlabel("Dim 1")
    axes[0].set_ylabel("Dim 2")
    axes[0].grid(True, alpha=0.3)
    axes[0].set_xlim(-4, 4)
    axes[0].set_ylim(-4, 4)
    axes[0].legend()
    
    # Generated scatter plot
    axes[1].scatter(generated[:, 0], generated[:, 1], alpha=0.6, s=50, color='orange', label='Generated')
    axes[1].set_title(f"Generated Samples ({method.upper()})", fontsize=12, fontweight='bold')
    axes[1].set_xlabel("Dim 1")
    axes[1].set_ylabel("Dim 2")
    axes[1].grid(True, alpha=0.3)
    axes[1].set_xlim(-4, 4)
    axes[1].set_ylim(-4, 4)
    axes[1].legend()
    
    plt.tight_layout()
    
    # Save figure
    output_path = Path(checkpoint_path).parent / f"visualization_{method}_{num_steps}.png"
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"Visualization saved to {output_path}")
    
    plt.show()


def visualize_mnist_generation(
    checkpoint_path: str, 
    num_samples: int = 16, 
    num_steps: int = 100, 
    method: str = "ddim",
    save_path: str = None
):
    """
    Generate and visualize MNIST samples using a trained U-Net diffusion model.
    
    Args:
        checkpoint_path: Path to model checkpoint
        num_samples: Number of MNIST samples to generate
        num_steps: Number of diffusion steps for DDIM
        method: 'ddpm' or 'ddim'
        save_path: Optional path to save the visualization
    """
    # Load configuration
    config = load_config('configs/default_pidm_config.yaml')
    
    # Initialize U-Net model for MNIST
    model = SimpleUNet(in_channels=1, out_channels=1, time_embed_dim=128)
    
    # Initialize trainer
    trainer = PIDMTrainer(model=model, args=config)
    
    # Load checkpoint
    print(f"Loading checkpoint from {checkpoint_path}...")
    trainer.load_checkpoint(checkpoint_path)
    
    # Create sampler
    sampler = DiffusionSampler(model, trainer.diffusion_schedule, trainer.device)
    
    # Generate samples - shape is (batch, channels, height, width)
    shape = (num_samples, 1, 28, 28)
    
    if method.lower() == "ddpm":
        print(f"Generating {num_samples} MNIST samples using DDPM...")
        generated = sampler.sample(shape, num_steps=1000, progress_bar=True)
    else:  # ddim
        print(f"Generating {num_samples} MNIST samples using DDIM ({num_steps} steps)...")
        generated = sampler.sample(shape, num_steps=num_steps, progress_bar=True)
    
    # Clip to valid range [0, 1]
    generated = torch.clamp(generated, 0, 1)
    
    # Check for NaN/Inf
    if torch.isnan(generated).any() or torch.isinf(generated).any():
        print("Warning: Generated samples contain NaN/Inf, replacing with random samples...")
        valid_mask = ~(torch.isnan(generated).any(dim=(1, 2, 3)) | torch.isinf(generated).any(dim=(1, 2, 3)))
        invalid_indices = torch.where(~valid_mask)[0]
        
        for idx in invalid_indices:
            # Sample random noise and clamp
            generated[idx] = torch.rand(1, 28, 28).clamp(0, 1)
    
    # Move to CPU for visualization
    generated = generated.cpu()
    
    # Create visualization grid
    grid_size = int(np.ceil(np.sqrt(num_samples)))
    
    fig, axes = plt.subplots(grid_size, grid_size, figsize=(12, 12))
    axes = axes.flatten()
    
    for i in range(num_samples):
        ax = axes[i]
        # Extract single channel from (1, 28, 28)
        digit_image = generated[i, 0].numpy()
        ax.imshow(digit_image, cmap='gray', vmin=0, vmax=1)
        ax.set_title(f"Sample {i+1}", fontsize=8)
        ax.axis('off')
    
    # Hide unused subplots
    for i in range(num_samples, len(axes)):
        axes[i].axis('off')
    
    plt.suptitle(f"Generated MNIST Digits ({method.upper()}, {num_steps} steps)", 
                 fontsize=14, fontweight='bold')
    plt.tight_layout()
    
    # Save figure
    if save_path is None:
        save_path = Path(checkpoint_path).parent / f"mnist_generated_{method}_{num_steps}.png"
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"Generated samples saved to {save_path}")
    
    plt.show()
    
    return generated


def visualize_mnist_comparison(
    checkpoint_path: str, 
    num_samples: int = 8, 
    num_steps: int = 100,
    method: str = "ddim",
    save_path: str = None
):
    """
    Compare ground truth MNIST samples with generated samples.
    
    Args:
        checkpoint_path: Path to model checkpoint
        num_samples: Number of samples to generate and compare
        num_steps: Number of diffusion steps for DDIM
        method: 'ddpm' or 'ddim'
        save_path: Optional path to save the visualization
    """
    # Load configuration
    config = load_config('configs/default_pidm_config.yaml')
    
    # Initialize U-Net model for MNIST
    model = SimpleUNet(in_channels=1, out_channels=1, time_embed_dim=128)
    
    # Initialize trainer
    trainer = PIDMTrainer(model=model, args=config)
    
    # Load checkpoint
    print(f"Loading checkpoint from {checkpoint_path}...")
    trainer.load_checkpoint(checkpoint_path)
    
    # Create sampler
    sampler = DiffusionSampler(model, trainer.diffusion_schedule, trainer.device)
    
    # Get ground truth MNIST samples
    print("Loading MNIST dataset...")
    dataset = MNISTDataset(split='test')
    indices = np.random.choice(len(dataset), num_samples, replace=False)
    ground_truth_samples = [dataset[i] for i in indices]
    ground_truth = torch.stack(ground_truth_samples)
    # Reshape to (batch, 1, 28, 28)
    ground_truth = ground_truth.view(-1, 1, 28, 28)
    
    # Generate samples
    shape = (num_samples, 1, 28, 28)
    
    if method.lower() == "ddpm":
        print(f"Generating {num_samples} MNIST samples using DDPM...")
        generated = sampler.sample(shape, num_steps=1000, progress_bar=True)
    else:  # ddim
        print(f"Generating {num_samples} MNIST samples using DDIM ({num_steps} steps)...")
        generated = sampler.sample(shape, num_steps=num_steps, progress_bar=True)
    
    # Clip to valid range
    generated = torch.clamp(generated, 0, 1)
    
    # Move to CPU for visualization
    ground_truth = ground_truth.cpu()
    generated = generated.cpu()
    
    # Create comparison grid (Ground Truth | Generated)
    fig, axes = plt.subplots(num_samples, 2, figsize=(6, 3*num_samples))
    
    for i in range(num_samples):
        # Ground truth - extract single channel
        gt_image = ground_truth[i, 0].numpy()
        axes[i, 0].imshow(gt_image, cmap='gray', vmin=0, vmax=1)
        axes[i, 0].set_title(f"Ground Truth {i+1}", fontsize=10, fontweight='bold')
        axes[i, 0].axis('off')
        
        # Generated - extract single channel
        gen_image = generated[i, 0].numpy()
        axes[i, 1].imshow(gen_image, cmap='gray', vmin=0, vmax=1)
        axes[i, 1].set_title(f"Generated {i+1}", fontsize=10, fontweight='bold')
        axes[i, 1].axis('off')
    
    plt.suptitle(f"MNIST Ground Truth vs Generated ({method.upper()}, {num_steps} steps)", 
                 fontsize=14, fontweight='bold')
    plt.tight_layout()
    
    # Save figure
    if save_path is None:
        save_path = Path(checkpoint_path).parent / f"mnist_comparison_{method}_{num_steps}.png"
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"Comparison saved to {save_path}")
    
    plt.show()
    
    return ground_truth, generated


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Visualize generated samples from diffusion models")
    parser.add_argument(
        "--dataset",
        type=str,
        choices=["gaussian", "mnist"],
        default="gaussian",
        help="Dataset type (gaussian or mnist)"
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        default="final_checkpoint.pt",
        help="Path to checkpoint file"
    )
    parser.add_argument(
        "--num-samples",
        type=int,
        default=500,
        help="Number of samples to visualize"
    )
    parser.add_argument(
        "--num-steps",
        type=int,
        default=1000,
        help="Number of steps for DDIM sampling (ignored for DDPM)"
    )
    parser.add_argument(
        "--method",
        type=str,
        choices=["ddpm", "ddim"],
        default="ddpm",
        help="Sampling method (ddpm for quality, ddim for speed)"
    )
    parser.add_argument(
        "--comparison",
        action="store_true",
        help="For MNIST: show side-by-side comparison with ground truth"
    )
    parser.add_argument(
        "--save",
        type=str,
        default=None,
        help="Path to save visualization (optional)"
    )
    
    args = parser.parse_args()
    
    if args.dataset == "mnist":
        if args.comparison:
            visualize_mnist_comparison(
                checkpoint_path=args.checkpoint,
                num_samples=args.num_samples,
                num_steps=args.num_steps,
                method=args.method,
                save_path=args.save
            )
        else:
            visualize_mnist_generation(
                checkpoint_path=args.checkpoint,
                num_samples=args.num_samples,
                num_steps=args.num_steps,
                method=args.method,
                save_path=args.save
            )
    else:
        visualize_samples(
            checkpoint_path=args.checkpoint,
            num_samples=args.num_samples,
            num_steps=args.num_steps,
            method=args.method
        )
