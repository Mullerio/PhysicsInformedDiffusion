import torch 
import torch.nn as nn
from typing import Optional


def create_optimizer(model: torch.nn.Module, config: dict) -> torch.optim.Optimizer:
    """
    Create optimizer from config.
    
    Args:
        model: Neural network
        config: Config dict with optimizer settings
        
    Returns:
        Optimizer instance
    """
    opt_name = config['optimizer']['name'].lower()
    lr = config['training']['learning_rate']
    wd = config['training'].get('weight_decay', 0.0)
    
    if opt_name == "adam":
        return torch.optim.Adam(model.parameters(), lr=lr, weight_decay=wd)
    elif opt_name == "adamw":
        return torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=wd)
    elif opt_name == "sgd":
        return torch.optim.SGD(model.parameters(), lr=lr, weight_decay=wd, momentum=0.9)
    else:
        raise ValueError(f"Unknown optimizer: {opt_name}")


def create_scheduler(optimizer: torch.optim.Optimizer, config: dict) -> Optional[torch.optim.lr_scheduler._LRScheduler]:
    """
    Create learning rate scheduler from config.
    
    Args:
        optimizer: Optimizer instance
        config: Config dict with scheduler settings
        
    Returns:
        Scheduler or None
    """
    sched_name = config['scheduler']['name'].lower()
    epochs = config['training']['epochs']
    
    if sched_name == "cosine":
        return torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    elif sched_name == "step":
        return torch.optim.lr_scheduler.StepLR(optimizer, step_size=epochs // 3, gamma=0.1)
    elif sched_name == "exponential":
        return torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.95)
    else:
        return None
