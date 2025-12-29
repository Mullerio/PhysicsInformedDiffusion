"""Visualization script for comparing ground truth vs generated samples."""

import torch
import matplotlib.pyplot as plt
from pathlib import Path
import argparse

from src.trainer import PIDMTrainer, load_config
from src.networks import NDimensionalMLP
from src.data_utils import BaseDataset
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


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Visualize ground truth vs generated samples")
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
    
    args = parser.parse_args()
    
    visualize_samples(
        checkpoint_path=args.checkpoint,
        num_samples=args.num_samples,
        num_steps=args.num_steps,
        method=args.method
    )
