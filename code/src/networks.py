"""Neural network components for physics-informed diffusion models."""

import torch
import torch.nn as nn
import math
from typing import Optional


class TimeEmbedding(nn.Module):
    """Sinusoidal time embedding for diffusion timesteps."""
    
    def __init__(self, embed_dim: int):
        """
        Args:
            embed_dim: Embedding dimension
        """
        super().__init__()
        self.embed_dim = embed_dim
        self.linear = nn.Sequential(
            nn.Linear(embed_dim, embed_dim * 4),
            nn.SiLU(),
            nn.Linear(embed_dim * 4, embed_dim)
        )
    
    def forward(self, t: torch.Tensor) -> torch.Tensor:
        """
        Args:
            t: Timestep indices (batch_size,)
            
        Returns:
            Embeddings (batch_size, embed_dim)
        """
        # Sinusoidal positional encoding
        device = t.device
        half_dim = self.embed_dim // 2
        freqs = torch.exp(
            -math.log(10000) * torch.arange(start=0, end=half_dim, dtype=torch.float32, device=device) / half_dim
        )
        args = t[:, None].float() * freqs[None]
        embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
        return self.linear(embedding)


class DiffusionSchedule:
    """Diffusion schedules for forward/reverse processes."""
    
    def __init__(self, num_timesteps: int = 1000, schedule_type: str = "linear"):
        """
        Args:
            num_timesteps: Number of diffusion steps
            schedule_type: "linear", "quadratic", or "cosine"
        """
        self.num_timesteps = num_timesteps
        self.schedule_type = schedule_type
        
        # Compute betas (variance schedule)
        if schedule_type == "linear":
            self.betas = torch.linspace(0.0001, 0.02, num_timesteps)
        elif schedule_type == "quadratic":
            self.betas = torch.linspace(0.0001 ** 0.5, 0.02 ** 0.5, num_timesteps) ** 2
        elif schedule_type == "cosine":
            # Cosine schedule as in https://arxiv.org/abs/2102.09672
            s = 0.008
            steps = torch.arange(num_timesteps + 1)
            alphas_cumprod = torch.cos(((steps / num_timesteps) + s) / (1 + s) * math.pi * 0.5) ** 2
            alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
            self.betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
            self.betas = torch.clip(self.betas, 0.0001, 0.9999)
        else:
            raise ValueError(f"Unknown schedule type: {schedule_type}")
        
        # Pre-compute alphas and alpha_cumprod
        self.alphas = 1 - self.betas
        self.alphas_cumprod = torch.cumprod(self.alphas, dim=0)
        self.alphas_cumprod_prev = torch.cat([torch.ones(1), self.alphas_cumprod[:-1]])
        
        # Variance for reverse process
        self.posterior_var = self.betas * (1 - self.alphas_cumprod_prev) / (1 - self.alphas_cumprod)
        
    def to(self, device: torch.device):
        """Move schedule to device."""
        self.betas = self.betas.to(device)
        self.alphas = self.alphas.to(device)
        self.alphas_cumprod = self.alphas_cumprod.to(device)
        self.alphas_cumprod_prev = self.alphas_cumprod_prev.to(device)
        self.posterior_var = self.posterior_var.to(device)
        return self
    
    def get_sqrt_alphas_cumprod(self, t: torch.Tensor) -> torch.Tensor:
        """Get sqrt(alpha_cumprod) for timesteps t."""
        return torch.sqrt(self.alphas_cumprod[t])
    
    def get_sqrt_one_minus_alphas_cumprod(self, t: torch.Tensor) -> torch.Tensor:
        """Get sqrt(1 - alpha_cumprod) for timesteps t."""
        return torch.sqrt(1 - self.alphas_cumprod[t])
    
    def get_posterior_var(self, t: torch.Tensor) -> torch.Tensor:
        """Get posterior variance for reverse process."""
        return self.posterior_var[t]


class ResidualBlock(nn.Module):
    """Basic residual block with time embedding"""
    
    def __init__(self, channels: int, time_embed_dim: int, dropout: float = 0.1):
        """
        Args:
            channels: Number of channels
            time_embed_dim: Time embedding dimension
            dropout: Dropout rate
        """
        super().__init__()
        self.time_embed_proj = nn.Linear(time_embed_dim, channels)
        
        self.block = nn.Sequential(
            nn.GroupNorm(32, channels),
            nn.SiLU(),
            nn.Conv2d(channels, channels, kernel_size=3, padding=1),
            nn.GroupNorm(32, channels),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Conv2d(channels, channels, kernel_size=3, padding=1),
        )
    
    def forward(self, x: torch.Tensor, t_embed: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Input tensor (batch, channels, height, width)
            t_embed: Time embedding (batch, time_embed_dim)
            
        Returns:
            Output tensor (batch, channels, height, width)
        """
        t_proj = self.time_embed_proj(t_embed)
        # Reshape for broadcasting
        t_proj = t_proj.view(t_proj.shape[0], -1, 1, 1)
        
        out = self.block(x)
        return out + x + t_proj


class SimpleUNet(nn.Module):
    """Simple UNet architecture for diffusion models."""
    
    def __init__(self, in_channels: int = 1, out_channels: int = 1, time_embed_dim: int = 128):
        """
        Args:
            in_channels: Number of input channels
            out_channels: Number of output channels
            time_embed_dim: Time embedding dimension
        """
        super().__init__()
        self.time_embed_dim = time_embed_dim
        self.time_embedding = TimeEmbedding(time_embed_dim)
        
        # Encoder
        self.input_proj = nn.Conv2d(in_channels, 64, kernel_size=3, padding=1)
        self.enc1 = ResidualBlock(64, time_embed_dim)
        self.pool1 = nn.MaxPool2d(2)
        
        self.ch_up1 = nn.Conv2d(64, 128, kernel_size=1)
        self.enc2 = ResidualBlock(128, time_embed_dim)
        self.pool2 = nn.MaxPool2d(2)
        
        # Bottleneck
        self.ch_up2 = nn.Conv2d(128, 256, kernel_size=1)
        self.bottleneck = ResidualBlock(256, time_embed_dim)
        
        # Decoder
        self.up2 = nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2)
        self.dec2 = ResidualBlock(128, time_embed_dim)
        
        self.up1 = nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2)
        self.dec1 = ResidualBlock(64, time_embed_dim)
        
        self.output_proj = nn.Conv2d(64, out_channels, kernel_size=3, padding=1)
    
    def forward(self, x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Input tensor (batch, in_channels, height, width)
            t: Timestep indices (batch,)
            
        Returns:
            Output tensor (batch, out_channels, height, width)
        """
        # Time embedding
        t_embed = self.time_embedding(t)
        
        # Encoder
        x = self.input_proj(x)
        x1 = self.enc1(x, t_embed)
        x = self.pool1(x1)
        
        x = self.ch_up1(x)
        x2 = self.enc2(x, t_embed)
        x = self.pool2(x2)
        
        # Bottleneck
        x = self.ch_up2(x)
        x = self.bottleneck(x, t_embed)
        
        # Decoder with skip connections
        x = self.up2(x)
        x = self.dec2(x + x2, t_embed)
        
        x = self.up1(x)
        x = self.dec1(x + x1, t_embed)
        
        x = self.output_proj(x)
        return x


class NDimensionalMLP(nn.Module):
    """MLP for n-dimensional point sampling (e.g., Gaussian mixtures)."""
    
    def __init__(self, in_features: int, out_features: int, time_embed_dim: int = 128, hidden_dim: int = 256):
        """
        Args:
            in_features: Input dimension 
            out_features: Output dimension (typically same as in_features)
            time_embed_dim: Time embedding dimension
            hidden_dim: Hidden layer dimension
        """
        super().__init__()
        self.time_embed_dim = time_embed_dim
        self.time_embedding = TimeEmbedding(time_embed_dim)
        
        self.net = nn.Sequential(
            nn.Linear(in_features + time_embed_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, out_features)
        )
    
    def forward(self, x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Input tensor (batch, in_features)
            t: Timestep indices (batch,)
            
        Returns:
            Output tensor (batch, out_features)
        """
        # Time embedding
        t_embed = self.time_embedding(t)
        
        # Concatenate input with time embedding
        x_with_time = torch.cat([x, t_embed], dim=-1)
        
        # Pass through MLP
        return self.net(x_with_time)
