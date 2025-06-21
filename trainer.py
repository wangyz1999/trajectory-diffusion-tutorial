"""
Training Script for 2D Trajectory Diffusion Model
Implements DDPM training with experiment management.
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from diffusers import DDPMScheduler
import numpy as np
import matplotlib.pyplot as plt
import os
import json
import time
from tqdm import tqdm
from typing import Dict, Optional

from dataset import TrajectoryDataModule
from model import create_model
import utils


class TrajectoryDiffusionTrainer:
    """Trainer for trajectory diffusion model."""
    
    def __init__(
        self,
        model: nn.Module,
        scheduler: DDPMScheduler,
        data_module: TrajectoryDataModule,
        device: str = "cuda" if torch.cuda.is_available() else "cpu",
        learning_rate: float = 1e-4,
        weight_decay: float = 1e-6,
        exp_name: str = "trajectory_diffusion"
    ):
        """
        Initialize the trainer.
        
        Args:
            model: UNet1D model for trajectory diffusion
            scheduler: DDPM noise scheduler
            data_module: Data module with train/val/test splits
            device: Device to train on
            learning_rate: Learning rate for optimizer
            weight_decay: Weight decay for optimizer
            exp_name: Experiment name
        """
        self.device = device
        self.model = model.to(device)
        self.scheduler = scheduler
        self.data_module = data_module
        self.exp_name = exp_name
        
        # Optimizer
        self.optimizer = optim.AdamW(
            self.model.parameters(),
            lr=learning_rate,
            weight_decay=weight_decay
        )
        
        # Training state
        self.global_step = 0
        self.current_epoch = 0
        self.best_val_loss = float('inf')
        self.train_losses = []
        self.val_losses = []
        
        # Experiment directory
        self.exp_dir = utils.create_experiment_dir(exp_name)
        print(f"Experiment directory: {self.exp_dir}")
        
        # Save experiment config
        self._save_config()
    
    def _save_config(self):
        """Save experiment configuration."""
        config = {
            'model_params': {
                'in_channels': self.model.in_channels,
                'out_channels': self.model.out_channels,
                'time_emb_dim': self.model.time_emb_dim,
            },
            'scheduler_params': {
                'num_train_timesteps': self.scheduler.config.num_train_timesteps,
                'beta_start': self.scheduler.config.beta_start,
                'beta_end': self.scheduler.config.beta_end,
                'beta_schedule': self.scheduler.config.beta_schedule,
            },
            'optimizer_params': {
                'learning_rate': self.optimizer.param_groups[0]['lr'],
                'weight_decay': self.optimizer.param_groups[0]['weight_decay'],
            },
            'data_params': {
                'batch_size': self.data_module.batch_size,
                'sequence_length': self.data_module.sequence_length,
                'train_split': self.data_module.train_split,
                'val_split': self.data_module.val_split,
            },
            'device': self.device,
            'exp_name': self.exp_name
        }
        
        with open(os.path.join(self.exp_dir, 'config.json'), 'w') as f:
            json.dump(config, f, indent=2)
    
    def train_step(self, batch: Dict[str, torch.Tensor]) -> float:
        """Single training step."""
        self.model.train()
        
        trajectories = batch['trajectory'].to(self.device)  # [batch, seq_len, 2]
        batch_size = trajectories.shape[0]
        
        # Transpose to [batch, 2, seq_len] for conv1d
        trajectories = trajectories.transpose(1, 2)
        
        # Sample random timesteps
        timesteps = torch.randint(
            0, self.scheduler.config.num_train_timesteps,
            (batch_size,), device=self.device
        ).long()
        
        # Sample noise
        noise = torch.randn_like(trajectories)
        
        # Add noise to trajectories
        noisy_trajectories = self.scheduler.add_noise(trajectories, noise, timesteps)
        
        # Predict noise
        noise_pred = self.model(noisy_trajectories, timesteps)
        
        # Handle potential size mismatch due to model architecture
        if noise_pred.shape != noise.shape:
            # Crop noise to match prediction size
            seq_len_pred = noise_pred.shape[-1]
            noise = noise[..., :seq_len_pred]
            noisy_trajectories = noisy_trajectories[..., :seq_len_pred]
        
        # Compute loss
        loss = nn.functional.mse_loss(noise_pred, noise)
        
        # Backward pass
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        
        self.global_step += 1
        
        return loss.item()
    
    def validate(self) -> float:
        """Validation step."""
        self.model.eval()
        total_loss = 0.0
        num_batches = 0
        
        with torch.no_grad():
            for batch in self.data_module.val_dataloader():
                trajectories = batch['trajectory'].to(self.device)
                batch_size = trajectories.shape[0]
                
                # Transpose to [batch, 2, seq_len]
                trajectories = trajectories.transpose(1, 2)
                
                # Sample random timesteps
                timesteps = torch.randint(
                    0, self.scheduler.config.num_train_timesteps,
                    (batch_size,), device=self.device
                ).long()
                
                # Sample noise
                noise = torch.randn_like(trajectories)
                
                # Add noise
                noisy_trajectories = self.scheduler.add_noise(trajectories, noise, timesteps)
                
                # Predict noise
                noise_pred = self.model(noisy_trajectories, timesteps)
                
                # Handle potential size mismatch due to model architecture
                if noise_pred.shape != noise.shape:
                    # Crop noise to match prediction size
                    seq_len_pred = noise_pred.shape[-1]
                    noise = noise[..., :seq_len_pred]
                    noisy_trajectories = noisy_trajectories[..., :seq_len_pred]
                
                # Compute loss
                loss = nn.functional.mse_loss(noise_pred, noise)
                
                total_loss += loss.item()
                num_batches += 1
        
        return total_loss / num_batches if num_batches > 0 else 0.0
    
    def train(self, num_epochs: int, save_every: int = 10, validate_every: int = 1):
        """Main training loop."""
        print(f"Starting training for {num_epochs} epochs...")
        print(f"Device: {self.device}")
        print(f"Model parameters: {sum(p.numel() for p in self.model.parameters()):,}")
        
        start_time = time.time()
        
        for epoch in range(num_epochs):
            self.current_epoch = epoch
            epoch_losses = []
            
            # Training
            pbar = tqdm(self.data_module.train_dataloader(), desc=f"Epoch {epoch+1}/{num_epochs}")
            for batch in pbar:
                loss = self.train_step(batch)
                epoch_losses.append(loss)
                
                # Update progress bar
                pbar.set_postfix({'loss': f'{loss:.4f}'})
            
            # Average epoch loss
            avg_train_loss = np.mean(epoch_losses)
            self.train_losses.append(avg_train_loss)
            
            # Validation
            if (epoch + 1) % validate_every == 0:
                val_loss = self.validate()
                self.val_losses.append(val_loss)
                
                print(f"Epoch {epoch+1}: Train Loss = {avg_train_loss:.4f}, Val Loss = {val_loss:.4f}")
                
                # Save best model
                if val_loss < self.best_val_loss:
                    self.best_val_loss = val_loss
                    self.save_checkpoint('best_model.pth')
                    print(f"New best model saved with val loss: {val_loss:.4f}")
            else:
                print(f"Epoch {epoch+1}: Train Loss = {avg_train_loss:.4f}")
            
            # Save checkpoint
            if (epoch + 1) % save_every == 0:
                self.save_checkpoint(f'checkpoint_epoch_{epoch+1}.pth')
            
            # Plot training curves
            self.plot_training_curves()
        
        # Final save
        self.save_checkpoint('final_model.pth')
        
        total_time = time.time() - start_time
        print(f"Training completed in {total_time:.2f} seconds")
        print(f"Best validation loss: {self.best_val_loss:.4f}")
    
    def save_checkpoint(self, filename: str):
        """Save model checkpoint."""
        checkpoint = {
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_config': self.scheduler.config,
            'global_step': self.global_step,
            'current_epoch': self.current_epoch,
            'best_val_loss': self.best_val_loss,
            'train_losses': self.train_losses,
            'val_losses': self.val_losses,
        }
        
        filepath = os.path.join(self.exp_dir, filename)
        torch.save(checkpoint, filepath)
        print(f"Checkpoint saved: {filepath}")
    
    def load_checkpoint(self, filepath: str):
        """Load model checkpoint."""
        checkpoint = torch.load(filepath, map_location=self.device, weights_only=False)
        
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.global_step = checkpoint['global_step']
        self.current_epoch = checkpoint['current_epoch']
        self.best_val_loss = checkpoint['best_val_loss']
        self.train_losses = checkpoint['train_losses']
        self.val_losses = checkpoint['val_losses']
        
        print(f"Checkpoint loaded: {filepath}")
    
    def plot_training_curves(self):
        """Plot and save training curves."""
        plt.figure(figsize=(12, 4))
        
        # Loss curves
        plt.subplot(1, 2, 1)
        plt.plot(self.train_losses, label='Train Loss', alpha=0.7)
        if self.val_losses:
            # Create x-axis for validation losses (assuming validation every epoch)
            val_epochs = np.arange(0, len(self.val_losses)) + 1
            plt.plot(val_epochs, self.val_losses, label='Val Loss', alpha=0.7)
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.title('Training and Validation Loss')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        # Learning rate
        plt.subplot(1, 2, 2)
        lr = self.optimizer.param_groups[0]['lr']
        plt.axhline(y=lr, color='r', linestyle='--', label=f'Learning Rate: {lr:.2e}')
        plt.xlabel('Epoch')
        plt.ylabel('Learning Rate')
        plt.title('Learning Rate Schedule')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.exp_dir, 'training_curves.png'), dpi=150, bbox_inches='tight')
        plt.close()


def create_trainer(
    data_dir: str = "data",
    batch_size: int = 32,
    sequence_length: int = 100,
    learning_rate: float = 1e-4,
    num_train_timesteps: int = 1000,
    exp_name: str = "trajectory_diffusion"
) -> TrajectoryDiffusionTrainer:
    """Create a configured trainer."""
    
    # Create data module
    data_module = TrajectoryDataModule(
        data_dir=data_dir,
        batch_size=batch_size,
        sequence_length=sequence_length
    )
    
    # Create model
    model = create_model(sequence_length=sequence_length)
    
    # Create scheduler
    scheduler = DDPMScheduler(
        num_train_timesteps=num_train_timesteps,
        beta_start=0.0001,
        beta_end=0.02,
        beta_schedule="linear",
        clip_sample=True
    )
    
    # Create trainer
    trainer = TrajectoryDiffusionTrainer(
        model=model,
        scheduler=scheduler,
        data_module=data_module,
        learning_rate=learning_rate,
        exp_name=exp_name
    )
    
    return trainer


if __name__ == "__main__":
    # Create and run trainer
    trainer = create_trainer(
        data_dir="data",
        batch_size=32,
        sequence_length=100,
        learning_rate=1e-4,
        num_train_timesteps=1000,
        exp_name="trajectory_diffusion"
    )
    
    # Train the model
    trainer.train(
        num_epochs=50,
        save_every=10,
        validate_every=1
    ) 