"""Loss functions for physics-informed diffusion models."""

import torch
import torch.nn as nn
from typing import Optional
from .networks import DiffusionSchedule

class MinSNRWeighting:
    def __init__(self, gamma: float = 5.0):
        self.gamma = gamma

    def __call__(self, diffusion_schedule, t):
        alpha_bar = diffusion_schedule.get_alphas_cumprod(t)
        snr = alpha_bar / (1.0 - alpha_bar)
        weight = torch.minimum(snr, torch.tensor(self.gamma, device=snr.device))
        return weight


def train_step(
    model: nn.Module,
    diffusion_schedule: DiffusionSchedule,
    optimizer: torch.optim.Optimizer,
    x: torch.Tensor,
    device: torch.device,
    prediction_type: str = "eps",  # "eps" or "x0"
    weighting_fn=None
) -> float:

    x = x.to(device)
    batch_size = x.shape[0]

    # Sample timesteps
    t = torch.randint(
        0, diffusion_schedule.num_timesteps, (batch_size,), device=device
    )

    # Sample noise
    noise = torch.randn_like(x)

    # Diffusion coefficients
    sqrt_alpha_bar = diffusion_schedule.get_sqrt_alphas_cumprod(t)
    sqrt_one_minus_alpha_bar = diffusion_schedule.get_sqrt_one_minus_alphas_cumprod(t)

    # Broadcast
    shape = [batch_size] + [1] * (x.dim() - 1)
    sqrt_alpha_bar = sqrt_alpha_bar.view(*shape)
    sqrt_one_minus_alpha_bar = sqrt_one_minus_alpha_bar.view(*shape)

    # Forward diffusion
    x_t = sqrt_alpha_bar * x + sqrt_one_minus_alpha_bar * noise

    optimizer.zero_grad()

    # Model output
    model_out = model(x_t, t)

    if prediction_type == "eps":
        # Standard noise prediction
        loss = nn.functional.mse_loss(model_out, noise)

    elif prediction_type == "x0":
        # x0 prediction
        x0_pred = model_out

        if weighting_fn is not None:
            weights = weighting_fn(diffusion_schedule, t)
        else:
            #default wheighting under which both losses are equivalent
            alpha_bar = diffusion_schedule.get_sqrt_alphas_cumprod(t)**2
            weights = alpha_bar / (1 - alpha_bar)
    
        weights = weights.view(*shape)

        loss = (weights * (x0_pred - x) ** 2).mean()
            

    else:
        raise ValueError(f"Unknown prediction_type: {prediction_type}")

    loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
    optimizer.step()

    return loss.item()

def val_step(
    model: nn.Module,
    diffusion_schedule: DiffusionSchedule,
    x: torch.Tensor,
    device: torch.device,
    prediction_type: str = "eps",  # "eps" or "x0"
    weighting_fn=None
) -> float:
    """
    Execute single diffusion validation step.
    
    Args:
        model: Diffusion model
        diffusion_schedule: Diffusion schedule
        x: Clean data samples - can be (batch, features) for 2D points or higher dimensional
        device: Device to run on
        prediction_type: Type of prediction ("eps" for noise or "x0" for data)
        weighting_fn: Optional weighting function for x0 prediction
        
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
        
        # Model output
        model_out = model(x_t, t)
        
        if prediction_type == "eps":
            # Noise prediction loss
            loss = nn.functional.mse_loss(model_out, noise)
        elif prediction_type == "x0":
            # x0 prediction loss
            x0_pred = model_out
            
            if weighting_fn is not None:
                weights = weighting_fn(diffusion_schedule, t)
            else:
                # Default weighting under which both losses are equivalent
                alpha_bar = diffusion_schedule.get_sqrt_alphas_cumprod(t)**2
                weights = alpha_bar / (1 - alpha_bar)
            
            weights = weights.view(*shape_to_broadcast)
            loss = (weights * (x0_pred - x) ** 2).mean()
        else:
            raise ValueError(f"Unknown prediction_type: {prediction_type}")
    
    return loss.item()

