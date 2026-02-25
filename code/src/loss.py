"""Loss functions for physics-informed diffusion models."""

import torch
import torch.nn as nn
from typing import Optional
from .networks import DiffusionSchedule
from .diffusion import DiffusionSampler

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


def physics_loss_step(
    model: nn.Module,
    diffusion_schedule: DiffusionSchedule,
    optimizer: torch.optim.Optimizer,
    x: torch.Tensor,
    device: torch.device,
    residual_fn,  # Callable: computes R(x0_pred)
    prediction_type: str = "x0",
    weighting_fn=None,
    c: float = 1.0, #weight for physics term
    use_ddim: bool = False,  # Use DDIM sampling for x_t
    ddim_steps: int = 50,  # Number of DDIM steps for sampling
) -> float:
    """
    Physics-informed diffusion model training step.
    Args:
        model: Diffusion model
        diffusion_schedule: Diffusion schedule
        optimizer: Optimizer
        x: samples (used as reference for data loss in mean estimation, or for shape in sample estimation)
        device: Device to run on
        residual_fn: Callable that computes residual R(x0_pred)
        prediction_type: Type of prediction (should be 'x0' for physics-informed)
        weighting_fn: Optional weighting function for x0 prediction
        c: Weight for physics term
        use_ddim: If True, use DDIM sample estimation (sample x_0 then forward diffuse); 
                  if False, use mean estimation (standard forward diffusion from data)
        ddim_steps: Number of steps for DDIM sampling (only used if use_ddim=True)
    Returns:
        Loss value
    """
    x = x.to(device)
    batch_size = x.shape[0]

    # Sample timesteps
    t = torch.randint(0, diffusion_schedule.num_timesteps, (batch_size,), device=device)

    sqrt_alpha_bar = diffusion_schedule.get_sqrt_alphas_cumprod(t)
    sqrt_one_minus_alpha_bar = diffusion_schedule.get_sqrt_one_minus_alphas_cumprod(t)

    shape = [batch_size] + [1] * (x.dim() - 1)
    sqrt_alpha_bar = sqrt_alpha_bar.view(*shape)
    sqrt_one_minus_alpha_bar = sqrt_one_minus_alpha_bar.view(*shape)

    # Always do forward diffusion from actual data first
    noise = torch.randn_like(x)
    x_t = sqrt_alpha_bar * x + sqrt_one_minus_alpha_bar * noise

    x_0_sample = None
    if use_ddim:
        # Sample estimation: use DDIM to denoise from x_t back to x_0
        sampler = DiffusionSampler(model, diffusion_schedule, device, pred_type=prediction_type)
        with torch.no_grad():
            # Denoise from x_t at timestep t to x_0 using DDIM
            # Pass scalar timestep (t[0] since all elements are the same)
            x_0_sample = sampler.sample_ddim(x.shape, num_steps=ddim_steps, progress_bar=False, t=t[0], x_t=x_t)
    # else: do nothing, use mean estimation

    optimizer.zero_grad()

    model_out = model(x_t, t)

    if prediction_type == "x0":
        x0_pred = model_out
        if weighting_fn is not None:
            weights = weighting_fn(diffusion_schedule, t)
        else:
            alpha_bar = diffusion_schedule.get_sqrt_alphas_cumprod(t) ** 2
            weights = alpha_bar / (1 - alpha_bar)
        weights = weights.view(*shape)
        # Data term: always compare prediction to actual data
        data_loss = (weights * (x0_pred - x) ** 2).mean()
        
        # Physics residual term from paper  https://arxiv.org/abs/2403.14404
        # For sample estimation, evaluate residual on the DDIM-denoised sample
        if use_ddim and x_0_sample is not None:
            residual = residual_fn(x_0_sample)
        else:
            # Mean estimation: evaluate residual on current prediction
            residual = residual_fn(x0_pred)
        
        alpha_bar_t = diffusion_schedule.alphas_cumprod[t]  # (batch,)
        alpha_bar_tm1 = diffusion_schedule.alphas_cumprod_prev[t]  # (batch,)
        beta_t = diffusion_schedule.betas[t]  # (batch,)
        #sigma2_t = ((1 - alpha_bar_tm1) / (1 - alpha_bar_t)) * beta_t  # (batch,)
        #sigma2_t = sigma2_t.view(*shape) 
        # Physics loss: mean over batch
        #NEED TO CLAMP FOR STABILITY
        sigma2_t = diffusion_schedule.get_posterior_var(t)
        sigma2_t = sigma2_t.view(*shape).clamp(min=1e-8)

        physics_loss = (0.5 / sigma2_t * (residual ** 2)).mean()
        loss = data_loss + c * physics_loss
    else:
        raise ValueError("Physics-informed loss requires prediction_type='x0' per paper")

    loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
    optimizer.step()

    return loss.item()


def physics_val_step(
    model: nn.Module,
    diffusion_schedule: DiffusionSchedule,
    x: torch.Tensor,
    device: torch.device,
    residual_fn,  # Callable: computes R(x0_pred)
    prediction_type: str = "x0",
    weighting_fn=None,
    c: float = 1.0, #weight for physics term
    use_ddim: bool = False,  # Use DDIM sampling for x_t
    ddim_steps: int = 50,  # Number of DDIM steps for sampling
) -> float:
    """
    Physics-informed diffusion model validation step.
    Args:
        model: Diffusion model
        diffusion_schedule: Diffusion schedule
        x: samples (used as reference for data loss in mean estimation, or for shape in sample estimation)
        device: Device to run on
        residual_fn: Callable that computes residual R(x0_pred)
        prediction_type: Type of prediction (should be 'x0' for physics-informed)
        weighting_fn: Optional weighting function for x0 prediction
        c: Weight for physics term
        use_ddim: If True, use DDIM sample estimation (sample x_0 then forward diffuse); 
                  if False, use mean estimation (standard forward diffusion from data)
        ddim_steps: Number of steps for DDIM sampling (only used if use_ddim=True)
    Returns:
        Loss value
    """
    x = x.to(device)
    batch_size = x.shape[0]
    with torch.no_grad():
        # Sample timesteps
        t = torch.randint(0, diffusion_schedule.num_timesteps, (batch_size,), device=device)
        sqrt_alpha_bar = diffusion_schedule.get_sqrt_alphas_cumprod(t)
        sqrt_one_minus_alpha_bar = diffusion_schedule.get_sqrt_one_minus_alphas_cumprod(t)
        shape = [batch_size] + [1] * (x.dim() - 1)
        sqrt_alpha_bar = sqrt_alpha_bar.view(*shape)
        sqrt_one_minus_alpha_bar = sqrt_one_minus_alpha_bar.view(*shape)
        
        # Always do forward diffusion from actual data first
        noise = torch.randn_like(x)
        x_t = sqrt_alpha_bar * x + sqrt_one_minus_alpha_bar * noise
        
        x_0_sample = None
        if use_ddim:
            sampler = DiffusionSampler(model, diffusion_schedule, device, pred_type=prediction_type)
            # Pass scalar timestep (t[0] since all elements are the same)
            x_0_sample = sampler.sample_ddim(x.shape, num_steps=ddim_steps, progress_bar=False, t=t[0], x_t=x_t)
        
        model_out = model(x_t, t)
        if prediction_type == "x0":
            x0_pred = model_out
            if weighting_fn is not None:
                weights = weighting_fn(diffusion_schedule, t)
            else:
                alpha_bar = diffusion_schedule.get_sqrt_alphas_cumprod(t) ** 2
                weights = alpha_bar / (1 - alpha_bar)
            weights = weights.view(*shape)
            # Data term: always compare prediction to actual data
            data_loss = (weights * (x0_pred - x) ** 2).mean()
            
            if use_ddim and x_0_sample is not None:
                residual = residual_fn(x_0_sample)
            else:
                # Mean estimation: evaluate residual on current prediction
                residual = residual_fn(x0_pred)
            
            alpha_bar_t = diffusion_schedule.alphas_cumprod[t]
            alpha_bar_tm1 = diffusion_schedule.alphas_cumprod_prev[t]
            beta_t = diffusion_schedule.betas[t]
            sigma2_t = ((1 - alpha_bar_tm1) / (1 - alpha_bar_t)) * beta_t
            sigma2_t = sigma2_t.view(*shape)
            sigma2_t = sigma2_t.clamp(min=1e-8)

            physics_loss = (0.5 / sigma2_t * (residual ** 2)).mean()
            loss = data_loss + c * physics_loss
        else:
            raise ValueError("Physics-informed loss requires prediction_type='x0' per paper")
    return loss.item()