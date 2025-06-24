"""
Training Script for Multi-Agent 2D Trajectory Diffusion Model
Implements DDPM training with multi-agent support and experiment management.
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

from dataset import MultiAgentTrajectoryDataModule, multi_agent_collate_fn
from model import create_multi_agent_model
import utils


class MultiAgentTrajectoryDiffusionTrainer:
    """Trainer for text-conditioned multi-agent trajectory diffusion model."""
    
    def __init__(
        self,
        model: nn.Module,
        scheduler: DDPMScheduler,
        data_module: MultiAgentTrajectoryDataModule,
        device: str = "cuda" if torch.cuda.is_available() else "cpu",
        learning_rate: float = 1e-4,
        weight_decay: float = 1e-6,
        exp_name: str = "multi_agent_trajectory_diffusion",
        max_agents: int = 5
    ):
        """
        Initialize the trainer.
        
        Args:
            model: MultiAgentUNet1D model for trajectory diffusion
            scheduler: DDPM noise scheduler
            data_module: Data module with train/val/test splits
            device: Device to train on
            learning_rate: Learning rate for optimizer
            weight_decay: Weight decay for optimizer
            exp_name: Experiment name
            max_agents: Maximum number of agents per sample
        """
        self.device = device
        self.model = model.to(device)
        self.scheduler = scheduler
        self.data_module = data_module
        self.exp_name = exp_name
        self.max_agents = max_agents
        
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
                'max_agents': self.model.max_agents,
                'in_channels_per_agent': self.model.in_channels_per_agent,
                'time_emb_dim': self.model.time_emb_dim,
                'text_emb_dim': self.model.text_emb_dim,
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
                'max_agents': self.max_agents,
                'multi_agent': True,
            },
            'device': self.device,
            'exp_name': self.exp_name
        }
        
        with open(os.path.join(self.exp_dir, 'config.json'), 'w') as f:
            json.dump(config, f, indent=2)
    
    def train_step(self, batch: Dict[str, torch.Tensor]) -> float:
        """Single training step with multi-agent text conditioning."""
        self.model.train()
        
        # Multi-agent trajectories: [batch, max_agents, seq_len, 2]
        trajectories = batch['trajectories'].to(self.device)
        # Agent mask: [batch, max_agents] - True where agents are active
        agent_mask = batch['agent_mask'].to(self.device)
        # Text embeddings: [batch, text_emb_dim]
        text_embeddings = batch['text_embedding'].to(self.device)
        
        batch_size = trajectories.shape[0]
        
        # Sample random timesteps
        timesteps = torch.randint(
            0, self.scheduler.config.num_train_timesteps,
            (batch_size,), device=self.device
        ).long()
        
        # Sample noise with same shape as trajectories
        noise = torch.randn_like(trajectories)
        
        # Add noise to trajectories
        noisy_trajectories = self.scheduler.add_noise(trajectories, noise, timesteps)
        
        # Predict noise with text conditioning and agent masking
        noise_pred = self.model(noisy_trajectories, timesteps, text_embeddings, agent_mask)
        
        # Compute loss only for active agents
        # Apply agent mask to both prediction and target
        mask_expanded = agent_mask.unsqueeze(-1).unsqueeze(-1).float()  # [batch, max_agents, 1, 1]
        
        # Masked predictions and targets
        noise_pred_masked = noise_pred * mask_expanded
        noise_masked = noise * mask_expanded
        
        # Compute MSE loss
        loss = nn.functional.mse_loss(noise_pred_masked, noise_masked, reduction='sum')
        
        # Normalize by number of active elements to get proper average
        num_active_elements = mask_expanded.sum()
        if num_active_elements > 0:
            loss = loss / num_active_elements
        
        # Backward pass
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        
        self.global_step += 1
        
        return loss.item()
    
    def validate(self) -> float:
        """Validation step with multi-agent text conditioning."""
        self.model.eval()
        total_loss = 0.0
        num_batches = 0
        
        with torch.no_grad():
            for batch in self.data_module.val_dataloader():
                # Multi-agent trajectories and masks
                trajectories = batch['trajectories'].to(self.device)
                agent_mask = batch['agent_mask'].to(self.device)
                text_embeddings = batch['text_embedding'].to(self.device)
                
                batch_size = trajectories.shape[0]
                
                # Sample random timesteps
                timesteps = torch.randint(
                    0, self.scheduler.config.num_train_timesteps,
                    (batch_size,), device=self.device
                ).long()
                
                # Sample noise
                noise = torch.randn_like(trajectories)
                
                # Add noise
                noisy_trajectories = self.scheduler.add_noise(trajectories, noise, timesteps)
                
                # Predict noise with text conditioning and agent masking
                noise_pred = self.model(noisy_trajectories, timesteps, text_embeddings, agent_mask)
                
                # Compute loss only for active agents
                mask_expanded = agent_mask.unsqueeze(-1).unsqueeze(-1).float()
                noise_pred_masked = noise_pred * mask_expanded
                noise_masked = noise * mask_expanded
                
                # Compute MSE loss
                loss = nn.functional.mse_loss(noise_pred_masked, noise_masked, reduction='sum')
                
                # Normalize by number of active elements
                num_active_elements = mask_expanded.sum()
                if num_active_elements > 0:
                    loss = loss / num_active_elements
                
                total_loss += loss.item()
                num_batches += 1
        
        return total_loss / num_batches if num_batches > 0 else 0.0
    
    def train(self, num_epochs: int, save_every: int = 10, validate_every: int = 1, plot_every: int = 5):
        """Main training loop for multi-agent text-conditioned trajectory diffusion."""
        print(f"🚀 Starting multi-agent trajectory diffusion training for {num_epochs} epochs...")
        print(f"📱 Device: {self.device}")
        print(f"🔢 Model parameters: {sum(p.numel() for p in self.model.parameters()):,}")
        print(f"🤖 Max agents: {self.max_agents}")
        print(f"📝 Text embedding dimension: {self.model.text_emb_dim}")
        print(f"📊 Data statistics:")
        sample_batch = self.data_module.get_sample_batch('train')
        print(f"   • Multi-agent trajectory shape: {sample_batch['trajectories'].shape}")
        print(f"   • Agent mask shape: {sample_batch['agent_mask'].shape}")
        print(f"   • Text embedding shape: {sample_batch['text_embedding'].shape}")
        print(f"   • Active agents in sample: {sample_batch['agent_mask'].float().mean(dim=1).mean():.2f}")
        
        # Generate initial test samples (epoch 0)
        print("🎨 Generating initial test samples...")
        self.plot_test_samples(0)
        
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
            
            # Generate test plots
            if (epoch + 1) % plot_every == 0:
                self.plot_test_samples(epoch + 1)
            
            # Plot training curves
            self.plot_training_curves()
        
        # Final save
        self.save_checkpoint('final_model.pth')
        
        # Generate final comprehensive test samples
        print("🎯 Generating final comprehensive test samples...")
        self.plot_final_results()
        
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
            'max_agents': self.max_agents,
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
    
    def generate_test_samples(self, n_samples: int = 6) -> tuple:
        """Generate multi-agent test samples during training to monitor progress."""
        self.model.eval()
        
        # Get a sample batch from validation set
        val_batch = self.data_module.get_sample_batch('val')
        
        # Select samples
        n_samples = min(n_samples, len(val_batch['trajectories']))
        real_trajectories = val_batch['trajectories'][:n_samples].numpy()  # [n_samples, max_agents, seq_len, 2]
        real_agent_masks = val_batch['agent_mask'][:n_samples].numpy()  # [n_samples, max_agents]
        text_embeddings = val_batch['text_embedding'][:n_samples].to(self.device)  # [n_samples, text_emb_dim]
        text_descriptions = val_batch['text_description'][:n_samples]
        agent_masks_tensor = val_batch['agent_mask'][:n_samples].to(self.device)  # [n_samples, max_agents]
        
        # Generate trajectories using the diffusion process
        with torch.no_grad():
            # Start from pure noise with same shape as real trajectories
            generated_trajectories = torch.randn(n_samples, self.max_agents, self.data_module.sequence_length, 2).to(self.device)
            
            # Reverse diffusion process
            for t in tqdm(reversed(range(self.scheduler.config.num_train_timesteps)), 
                         desc="Generating samples", leave=False):
                timesteps = torch.full((n_samples,), t, device=self.device).long()
                
                # Predict noise with agent masking
                noise_pred = self.model(generated_trajectories, timesteps, text_embeddings, agent_masks_tensor)
                
                # Remove predicted noise
                generated_trajectories = self.scheduler.step(
                    noise_pred, t, generated_trajectories
                ).prev_sample
        
        # Convert to numpy
        generated_trajectories = generated_trajectories.cpu().numpy()  # [n_samples, max_agents, seq_len, 2]
        
        return real_trajectories, generated_trajectories, real_agent_masks, text_descriptions
    
    def plot_test_samples(self, epoch: int):
        """Generate and plot multi-agent test samples to monitor training progress."""
        print(f"🎨 Generating multi-agent test samples for epoch {epoch}...")
        
        try:
            real_trajectories, generated_trajectories, agent_masks, text_descriptions = self.generate_test_samples(n_samples=6)
            
            # Plot training progress
            save_path = os.path.join(self.exp_dir, f'multi_agent_training_progress_epoch_{epoch:03d}.png')
            utils.plot_multi_agent_training_progress(
                real_trajectories, 
                generated_trajectories,
                agent_masks,
                text_descriptions,
                epoch,
                save_path=save_path,
                n_samples=6
            )
            
            print(f"✅ Multi-agent test samples plotted and saved for epoch {epoch}")
            
        except Exception as e:
            print(f"⚠️  Failed to generate test samples: {str(e)}")
        
        # Switch back to training mode
        self.model.train()
    
    def plot_final_results(self):
        """Generate comprehensive final results with diverse multi-agent text prompts."""
        print("🎨 Generating comprehensive final multi-agent results...")
        
        try:
            # Generate samples for different agent configurations
            sample_batch = self.data_module.get_sample_batch('val', 12)
            
            final_trajectories = []
            final_agent_masks = []
            final_descriptions = []
            
            real_trajectories = sample_batch['trajectories'][:12]
            real_agent_masks = sample_batch['agent_mask'][:12]
            text_embeddings = sample_batch['text_embedding'][:12].to(self.device)
            text_descriptions = sample_batch['text_description'][:12]
            agent_masks_tensor = sample_batch['agent_mask'][:12].to(self.device)
            
            # Generate trajectories
            with torch.no_grad():
                generated_batch = torch.randn(12, self.max_agents, self.data_module.sequence_length, 2).to(self.device)
                
                # Reverse diffusion process
                for t in tqdm(reversed(range(self.scheduler.config.num_train_timesteps)), desc="Final generation"):
                    timesteps = torch.full((12,), t, device=self.device).long()
                    noise_pred = self.model(generated_batch, timesteps, text_embeddings, agent_masks_tensor)
                    generated_batch = self.scheduler.step(noise_pred, t, generated_batch).prev_sample
                
                generated_trajectories_final = generated_batch.cpu().numpy()
            
            # Plot comprehensive results
            save_path = os.path.join(self.exp_dir, 'comprehensive_multi_agent_results.png')
            utils.plot_multi_agent_final_results(
                real_trajectories.numpy(),
                generated_trajectories_final,
                real_agent_masks.numpy(),
                text_descriptions,
                title="Final Multi-Agent Text-Conditioned Results",
                save_path=save_path,
                max_plots=12
            )
            
            print(f"✅ Comprehensive final multi-agent results saved")
            
        except Exception as e:
            print(f"⚠️  Failed to generate final results: {str(e)}")
        
        # Switch back to training mode
        self.model.train()
    
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
        plt.title('Multi-Agent Training and Validation Loss')
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
        plt.savefig(os.path.join(self.exp_dir, 'multi_agent_training_curves.png'), dpi=150, bbox_inches='tight')
        plt.close()


def create_multi_agent_trainer(
    data_dir: str = "data",
    batch_size: int = 32,
    sequence_length: int = 100,
    max_agents: int = 5,
    learning_rate: float = 1e-4,
    num_train_timesteps: int = 1000,
    text_emb_dim: int = 384,
    exp_name: str = "multi_agent_trajectory_diffusion"
) -> MultiAgentTrajectoryDiffusionTrainer:
    """Create a configured trainer for multi-agent text-conditioned trajectory diffusion."""
    
    # Create multi-agent data module
    data_module = MultiAgentTrajectoryDataModule(
        data_dir=data_dir,
        batch_size=batch_size,
        sequence_length=sequence_length
    )
    
    # Create multi-agent model
    model = create_multi_agent_model(
        max_agents=max_agents,
        sequence_length=sequence_length,
        text_emb_dim=text_emb_dim
    )
    
    # Create scheduler
    scheduler = DDPMScheduler(
        num_train_timesteps=num_train_timesteps,
        beta_start=0.0001,
        beta_end=0.02,
        beta_schedule="linear",
        clip_sample=True
    )
    
    # Create trainer
    trainer = MultiAgentTrajectoryDiffusionTrainer(
        model=model,
        scheduler=scheduler,
        data_module=data_module,
        learning_rate=learning_rate,
        exp_name=exp_name,
        max_agents=max_agents
    )
    
    return trainer


# Keep the original for backward compatibility
TrajectoryDiffusionTrainer = MultiAgentTrajectoryDiffusionTrainer
create_trainer = create_multi_agent_trainer


if __name__ == "__main__":
    # Create and run multi-agent trainer
    trainer = create_multi_agent_trainer(
        data_dir="data",
        batch_size=64,  # Smaller batch size for multi-agent due to increased memory usage
        sequence_length=100,
        max_agents=5,
        learning_rate=1e-4,
        num_train_timesteps=1000,
        text_emb_dim=384,
        exp_name="multi_agent_trajectory_diffusion"
    )
    
    # Train the multi-agent model
    trainer.train(
        num_epochs=50,
        save_every=1,
        validate_every=1,
        plot_every=1  # Generate test plots every 5 epochs
    ) 