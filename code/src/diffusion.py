"""Sampling and inference utilities for diffusion models."""

import torch
import torch.nn as nn
from tqdm import tqdm
from .networks import DiffusionSchedule


class DiffusionSampler:
    """Sampler for generating samples from diffusion models."""
    
    def __init__(
        self,
        model: nn.Module,
        diffusion_schedule: DiffusionSchedule,
        device: torch.device,
    ):
        """
        Args:
            model: Trained diffusion model
            diffusion_schedule: Diffusion schedule used during training
            device: Device to run on
        """
        self.model = model
        self.diffusion_schedule = diffusion_schedule
        self.device = device
        self.model.eval()
    
    @torch.no_grad()
    def sample_ddpm(
        self,
        shape: tuple,
        progress_bar: bool = True,
    ) -> torch.Tensor:
        """
        Sample using DDPM (stochastic, uses all timesteps for best quality).
        
        Args:
            shape: Shape of samples to generate (batch_size, ...)
            progress_bar: Whether to show progress bar
            
        Returns:
            Generated samples
        """
        batch_size = shape[0]
        x_t = torch.randn(shape, device=self.device)
        
        # Use all timesteps for DDPM
        timesteps = torch.arange(self.diffusion_schedule.num_timesteps - 1, -1, -1)
        
        iterator = tqdm(timesteps, disable=not progress_bar)
        
        for i, t_idx_tensor in enumerate(iterator):
            t_idx = t_idx_tensor.item()
            t = torch.full((batch_size,), t_idx, dtype=torch.long, device=self.device)
            
            # Model prediction
            predicted_noise = self.model(x_t, t)
            predicted_noise = torch.clamp(predicted_noise, -10.0, 10.0)
            
            # Get alphas
            alpha_bar_t = self.diffusion_schedule.alphas_cumprod[t_idx].item()
            alpha_bar_t = max(alpha_bar_t, 1e-7)
            
            # DDPM: predict mean and add stochastic noise
            sqrt_alpha_bar_t = alpha_bar_t ** 0.5
            sqrt_one_minus_alpha_bar_t = (1 - alpha_bar_t) ** 0.5
            
            # Predict x_0
            x_0_pred = (x_t - sqrt_one_minus_alpha_bar_t * predicted_noise) / sqrt_alpha_bar_t
            x_0_pred = torch.clamp(x_0_pred, -20.0, 20.0)
            
            # Get posterior variance
            posterior_var = self.diffusion_schedule.get_posterior_var(torch.tensor(t_idx, device=self.device))
            
            # Mean of q(x_{t-1} | x_t, x_0)
            if t_idx > 0:
                alpha_t = self.diffusion_schedule.alphas[t_idx].item()
                alpha_bar_t_prev = self.diffusion_schedule.alphas_cumprod_prev[t_idx].item()
                
                mean_coeff_x_t = ((1 - alpha_bar_t_prev) / (1 - alpha_bar_t)) * alpha_t ** 0.5
                mean_coeff_x_0 = (alpha_bar_t_prev ** 0.5 * (1 - alpha_t)) / (1 - alpha_bar_t)
                
                mean = mean_coeff_x_t * x_t + mean_coeff_x_0 * x_0_pred
                
                # Add stochastic noise
                z = torch.randn_like(x_t)
                x_t = mean + posterior_var.sqrt() * z
            else:
                # Last step: just use the predicted x_0
                x_t = x_0_pred
            
            x_t = torch.clamp(x_t, -20.0, 20.0)
            
            # Check for NaN
            if torch.isnan(x_t).any():
                print(f"NaN at step {i}/{len(timesteps)}, replacing with previous estimate")
                x_t = x_0_pred
        
        return x_t
    
    @torch.no_grad()
    def sample_ddim(
        self,
        shape: tuple,
        num_steps: int = 50,
        eta: float = 0.0,
        progress_bar: bool = True,
    ) -> torch.Tensor:
        """
        Sample using DDIM as in https://arxiv.org/abs/2010.02502.
        
        Args:
            shape: Shape of samples to generate
            num_steps: Number of reverse steps (can be < num_timesteps)
            eta: Stochasticity parameter (0 = deterministic)
            progress_bar: Whether to show progress bar
            
        Returns:
            Generated samples
        """
        x_t = torch.randn(shape, device=self.device)
        
        # subsample timesteps for faster DDIM sampling
        timesteps = torch.linspace(self.diffusion_schedule.num_timesteps - 1, 0, num_steps, dtype=torch.long, device=self.device)
        
        iterator = tqdm(timesteps, disable=not progress_bar)
        
        for i, t_idx in enumerate(iterator):
            t = torch.full((shape[0],), t_idx.item(), dtype=torch.long, device=self.device)
            
            # Get previous timestep
            t_prev_idx = timesteps[i + 1] if i < len(timesteps) - 1 else torch.tensor(-1)
            
            # Model predicts noise
            predicted_noise = self.model(x_t, t)
            
            # Get coefficients
            alpha_bar_t = self.diffusion_schedule.alphas_cumprod[t_idx.item()]
            alpha_bar_t_prev = (
                self.diffusion_schedule.alphas_cumprod[t_prev_idx.item()] 
                if t_prev_idx >= 0 else torch.tensor(1.0, device=self.device)
            )
            
            # Predict x_0
            x_0_pred = (x_t - (1 - alpha_bar_t).sqrt() * predicted_noise) / alpha_bar_t.sqrt()
            
            # Direction pointing to x_t
            direction = (1 - alpha_bar_t_prev).sqrt() * predicted_noise
            
            # New x_t
            x_t = alpha_bar_t_prev.sqrt() * x_0_pred + direction
            
            # Add stochasticity
            if t_prev_idx >= 0:
                variance = (1 - alpha_bar_t_prev) / (1 - alpha_bar_t) * (1 - alpha_bar_t / alpha_bar_t_prev)
                if eta > 0:
                    noise = torch.randn_like(x_t)
                    x_t = x_t + eta * variance.sqrt() * noise
        
        return x_t
    
    @torch.no_grad()
    def sample(
        self,
        shape: tuple,
        num_steps: int = 1000,
        progress_bar: bool = True,
        method: str = "ddim",
    ) -> torch.Tensor:
        """
        Sample from the diffusion model.
        
        Args:
            shape: Shape of samples to generate
            num_steps: Number of reverse steps (used by DDIM; ignored for DDPM which uses all timesteps)
            progress_bar: Whether to show progress bar
            method: 'ddpm' (all steps, stochastic) or 'ddim' (faster, deterministic)
            
        Returns:
            Generated samples
        """
        if method.lower() == "ddpm":
            return self.sample_ddpm(shape, progress_bar)
        else:
            return self.sample_ddim(shape, num_steps, progress_bar=progress_bar)
