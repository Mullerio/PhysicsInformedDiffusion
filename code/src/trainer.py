"""Trainer for physics-informed diffusion models."""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from typing import Dict, Optional
from tqdm import tqdm
import yaml
import wandb
from pathlib import Path
from .utils import create_optimizer, create_scheduler
from .networks import DiffusionSchedule
from .loss import train_step, val_step


class PIDMTrainer:
    """Trainer for physics-informed diffusion models."""
    def __init__(
        self,
        model: nn.Module,
        args: Dict,
        device: Optional[torch.device] = None,
        output_dir: Optional[str] = None,
    ):
        """
        Args:
            model: Diffusion model
            args: Configuration dict    
            device: torch device (selected from config if None)
            output_dir: Directory to save checkpoints and results
        """
        if args['device']['use_cuda'] and torch.cuda.is_available():
            self.device = device or torch.device(f"cuda:{args['device'].get('device_id', 0)}")
        else:
            self.device = device or torch.device("cpu")
    
        self.model = model.to(self.device)
        self.pred_type = args.get('model', {}).get('prediction_type', 'eps')
        
        # Initialize diffusion schedule
        num_timesteps = args.get('diffusion', {}).get('num_timesteps', 1000)
        schedule_type = args.get('diffusion', {}).get('schedule', 'linear')
        self.diffusion_schedule = DiffusionSchedule(num_timesteps, schedule_type).to(self.device)
        
        self.args = args
        
        # Setup output directory
        self.output_dir = Path(output_dir) if output_dir else Path('.')
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        self.optimizer = create_optimizer(self.model, args)
        self.scheduler = create_scheduler(self.optimizer, args)
        
        if self.args['wandb']['enabled']:
            wandb.init(project=self.args['wandb']['project'], config=self.args)

        
    def train(
        self,
        train_loader: DataLoader,
        val_loader: Optional[DataLoader] = None,
        num_epochs: Optional[int] = None,
    ) -> Dict[str, list]:
        """Train the diffusion model."""
        num_epochs = num_epochs or self.args['training']['epochs']
        log_freq = self.args['logging'].get('log_freq', 100)
        save_freq = self.args['logging'].get('save_freq', 10)
        history = {'train': [], 'val': []}
        
        for epoch in range(num_epochs):
            self.model.train()
            train_loss = 0.0
            
            pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs} [Train]")
            for batch_idx, x in enumerate(pbar):
                loss = train_step(self.model, self.diffusion_schedule, self.optimizer, x, self.device, self.pred_type)
                train_loss += loss
                
                avg_loss = train_loss / (batch_idx + 1)
                pbar.set_postfix({'loss': avg_loss})
                
                if batch_idx % log_freq == 0:
                    continue  # TODO: Add needed logging, eval stuff wanted during training here.
                    
                    
            train_loss /= len(train_loader)
            history['train'].append(train_loss)
            
            # Validation
            val_loss = None
            if val_loader is not None:
                self.model.eval()
                val_loss = 0.0
                pbar = tqdm(val_loader, desc=f"Epoch {epoch+1}/{num_epochs} [Val]")
                for x in pbar:
                    loss = val_step(self.model, self.diffusion_schedule, x, self.device)
                    val_loss += loss
                    pbar.set_postfix({'loss': loss})
                
                val_loss /= len(val_loader)
                history['val'].append(val_loss)
            
            # Scheduler step
            if self.scheduler:
                self.scheduler.step()
            
            # Periodic checkpoint saving
            if (epoch + 1) % save_freq == 0:
                self.save_checkpoint(f'checkpoint_epoch_{epoch+1}.pt')
            
            # Logging
            log_dict = {'epoch': epoch + 1, 'train_loss': train_loss}
            if val_loss is not None:
                log_dict['val_loss'] = val_loss
            if self.scheduler:
                log_dict['lr'] = self.optimizer.param_groups[0]['lr']
            
            print(f"Epoch {epoch+1}/{num_epochs} - train_loss: {train_loss:.6f}", end="")
            if val_loss is not None:
                print(f", val_loss: {val_loss:.6f}", end="")
            print()
            
            if self.args['wandb']['enabled']:
                wandb.log(log_dict)
        
        if self.args['wandb']['enabled']:
            wandb.finish()
        
        
        self.save_checkpoint('final_checkpoint.pt')
        return history
    
    def save_checkpoint(self, path: str = 'checkpoint.pt') -> None:
        """Save model checkpoint with diffusion schedule."""
        checkpoint_path = self.output_dir / path
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'diffusion_config': {
                'num_timesteps': self.diffusion_schedule.num_timesteps,
                'schedule_type': self.diffusion_schedule.schedule_type,
            }
        }, checkpoint_path)
        print(f"Checkpoint saved to {checkpoint_path}")
    
    def load_checkpoint(self, path: str) -> None:
        """Load model checkpoint."""
        checkpoint = torch.load(path, map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
                
        
        
def load_config(path: str) -> Dict:
    """Load YAML config."""
    with open(path) as f:
        return yaml.safe_load(f)





