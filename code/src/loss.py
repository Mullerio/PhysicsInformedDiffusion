"""Loss functions for physics-informed diffusion models."""

import torch
import torch.nn as nn
from typing import Optional
from .networks import DiffusionSchedule


def train_step(
    model: nn.Module,
    diffusion_schedule: DiffusionSchedule,
    optimizer: torch.optim.Optimizer,
    x: torch.Tensor,
    device: torch.device
) -> float:
    """
    Execute single diffusion training step.
    
    Args:
        model: Diffusion model
        diffusion_schedule: Diffusion schedule
        optimizer: Optimizer
        x: Clean data samples - can be (batch, features) for 2D points or higher dimensional
        device: Device to run on
        
    Returns:
        Loss value
    """
    x = x.to(device)
    batch_size = x.shape[0]
    
    # Sample random timesteps
    t = torch.randint(0, diffusion_schedule.num_timesteps, (batch_size,), device=device)
    
    # Sample noise
    noise = torch.randn_like(x)
    
    # Get diffusion coefficients
    sqrt_alphas_cumprod = diffusion_schedule.get_sqrt_alphas_cumprod(t)
    sqrt_one_minus_alphas_cumprod = diffusion_schedule.get_sqrt_one_minus_alphas_cumprod(t)
    
    # Reshape for broadcasting
    # For 2D points: (batch, 2) -> reshape to (batch, 1) for broadcasting
    # For images: (batch, c, h, w) -> reshape to (batch, 1, 1, 1)
    shape_to_broadcast = [batch_size] + [1] * (x.dim() - 1)
    sqrt_alphas_cumprod = sqrt_alphas_cumprod.view(*shape_to_broadcast)
    sqrt_one_minus_alphas_cumprod = sqrt_one_minus_alphas_cumprod.view(*shape_to_broadcast)
    
    # Forward diffusion process: q(x_t | x_0)
    x_t = sqrt_alphas_cumprod * x + sqrt_one_minus_alphas_cumprod * noise
    
    # Model predicts noise
    optimizer.zero_grad()
    predicted_noise = model(x_t, t)
    
    # Noise prediction loss (L2 loss)
    loss = nn.functional.mse_loss(predicted_noise, noise)
    
    loss.backward()
    
    # Gradient clipping to prevent explosion
    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
    
    optimizer.step()
    
    return loss.item()


def val_step(
    model: nn.Module,
    diffusion_schedule: DiffusionSchedule,
    x: torch.Tensor,
    device: torch.device
) -> float:
    """
    Execute single diffusion validation step.
    
    Args:
        model: Diffusion model
        diffusion_schedule: Diffusion schedule
        x: Clean data samples - can be (batch, features) for 2D points or higher dimensional
        device: Device to run on
        
    Returns:
        Loss value
    """
    x = x.to(device)
    batch_size = x.shape[0]
    
    with torch.no_grad():
        # Sample random timesteps
        t = torch.randint(0, diffusion_schedule.num_timesteps, (batch_size,), device=device)
        
        # Sample noise
        noise = torch.randn_like(x)
        
        # Get diffusion coefficients
        sqrt_alphas_cumprod = diffusion_schedule.get_sqrt_alphas_cumprod(t)
        sqrt_one_minus_alphas_cumprod = diffusion_schedule.get_sqrt_one_minus_alphas_cumprod(t)
        
        # Reshape for broadcasting
        shape_to_broadcast = [batch_size] + [1] * (x.dim() - 1)
        sqrt_alphas_cumprod = sqrt_alphas_cumprod.view(*shape_to_broadcast)
        sqrt_one_minus_alphas_cumprod = sqrt_one_minus_alphas_cumprod.view(*shape_to_broadcast)
        
        # Forward diffusion process: q(x_t | x_0)
        x_t = sqrt_alphas_cumprod * x + sqrt_one_minus_alphas_cumprod * noise
        
        # Model predicts noise
        predicted_noise = model(x_t, t)
        
        # Noise prediction loss
        loss = nn.functional.mse_loss(predicted_noise, noise)
    
    return loss.item()

