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
from typing import Dict, Optional, List

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
        max_agents: int = 5,
        cfg_dropout_prob: float = 0.15  # Probability to drop text conditioning for CFG training
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
            cfg_dropout_prob: Probability to drop text conditioning during training for CFG
        """
        self.device = device
        self.model = model.to(device)
        self.scheduler = scheduler
        self.data_module = data_module
        self.exp_name = exp_name
        self.max_agents = max_agents
        self.cfg_dropout_prob = cfg_dropout_prob
        
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
            'training_params': {
                'cfg_dropout_prob': self.cfg_dropout_prob,
                'cfg_enabled': True,
            },
            'device': self.device,
            'exp_name': self.exp_name
        }
        
        with open(os.path.join(self.exp_dir, 'config.json'), 'w') as f:
            json.dump(config, f, indent=2)
    
    def train_step(self, batch: Dict[str, torch.Tensor]) -> float:
        """Single training step with multi-agent text conditioning and CFG training."""
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
        
        # Classifier-free guidance training: randomly drop text conditioning
        cfg_mask = torch.rand(batch_size, device=self.device) < self.cfg_dropout_prob
        text_embeddings_cfg = text_embeddings.clone()
        
        # Zero out text embeddings for CFG dropout samples
        if cfg_mask.any():
            text_embeddings_cfg[cfg_mask] = 0.0
        
        # Predict noise with text conditioning and agent masking
        noise_pred = self.model(noisy_trajectories, timesteps, text_embeddings_cfg, agent_mask)
        
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
    
    def train(self, num_epochs: int, save_every: int = 10, validate_every: int = 1, plot_every: int = 1):
        """Main training loop for multi-agent text-conditioned trajectory diffusion with CFG."""
        print(f"🚀 Starting multi-agent trajectory diffusion training with CFG for {num_epochs} epochs...")
        print(f"📱 Device: {self.device}")
        print(f"🔢 Model parameters: {sum(p.numel() for p in self.model.parameters()):,}")
        print(f"🤖 Max agents: {self.max_agents}")
        print(f"📝 Text embedding dimension: {self.model.text_emb_dim}")
        print(f"🎯 CFG dropout probability: {self.cfg_dropout_prob:.1%}")
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
                pbar.set_postfix({'loss': f'{loss:.4f}', 'cfg_drop': f'{self.cfg_dropout_prob:.1%}'})
            
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
            
            # Generate test plots every epoch
            if (epoch + 1) % plot_every == 0:
                self.plot_test_samples(epoch + 1)
            
            # Plot training curves
            self.plot_training_curves()
        
        # Final save
        self.save_checkpoint('final_model.pth')
        
        # Generate final comprehensive test samples
        print("🎯 Generating final comprehensive test samples with CFG...")
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
    
    def generate_test_samples_with_cfg(self, n_samples: int = 6, guidance_scales: List[float] = [0.0, 1.0, 4.0, 7.0, 10.0, 13.0, 15.0, 20.0]) -> Dict[str, any]:
        """Generate multi-agent test samples with different CFG guidance scales."""
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
        
        # Generate trajectories using different guidance scales
        generated_trajectories_dict = {}
        
        with torch.no_grad():
            for cfg_scale in guidance_scales:
                print(f"  🎯 Generating with CFG scale: {cfg_scale}")
                
                # Start from pure noise with same shape as real trajectories
                generated_trajectories = torch.randn(n_samples, self.max_agents, self.data_module.sequence_length, 2).to(self.device)
                
                # Reverse diffusion process with CFG
                for t in tqdm(reversed(range(self.scheduler.config.num_train_timesteps)), 
                             desc=f"CFG {cfg_scale}", leave=False):
                    timesteps = torch.full((n_samples,), t, device=self.device).long()
                    
                    # Use CFG for noise prediction
                    if cfg_scale == 1.0:
                        # Standard conditional generation
                        noise_pred = self.model(generated_trajectories, timesteps, text_embeddings, agent_masks_tensor)
                    else:
                        # CFG generation
                        noise_pred = self.model.forward_with_cfg(
                            generated_trajectories, timesteps, text_embeddings, 
                            agent_masks_tensor, guidance_scale=cfg_scale
                        )
                    
                    # Remove predicted noise
                    generated_trajectories = self.scheduler.step(
                        noise_pred, t, generated_trajectories
                    ).prev_sample
                
                # Convert to numpy and store
                generated_trajectories_dict[cfg_scale] = generated_trajectories.cpu().numpy()
        
        return {
            'real_trajectories': real_trajectories,
            'generated_trajectories': generated_trajectories_dict,
            'agent_masks': real_agent_masks,
            'text_descriptions': text_descriptions,
            'guidance_scales': guidance_scales
        }
    
    def generate_test_samples(self, n_samples: int = 6) -> tuple:
        """Generate multi-agent test samples during training to monitor progress (backward compatibility)."""
        # Use standard guidance scale of 1.0 for backward compatibility
        results = self.generate_test_samples_with_cfg(n_samples, guidance_scales=[1.0])
        
        real_trajectories = results['real_trajectories']
        generated_trajectories = results['generated_trajectories'][1.0]
        agent_masks = results['agent_masks']
        text_descriptions = results['text_descriptions']
        
        return real_trajectories, generated_trajectories, agent_masks, text_descriptions
    
    def plot_test_samples(self, epoch: int):
        """Generate and plot multi-agent test samples with CFG to monitor training progress."""
        print(f"🎨 Generating multi-agent test samples with CFG for epoch {epoch}...")
        
        try:
            # Generate three sets of plots with different random samples
            for version in range(1, 4):  # v1, v2, v3
                print(f"  📊 Generating version {version}/3...")
                
                # Generate samples with full CFG scale range
                results = self.generate_test_samples_with_cfg(n_samples=6, guidance_scales=[0.0, 1.0, 4.0, 7.0, 10.0, 13.0, 15.0, 20.0])
                
                # Plot standard training progress (with CFG scale 1.0)
                save_path = os.path.join(self.exp_dir, f'multi_agent_training_progress_epoch_{epoch:03d}_v{version}.png')
                utils.plot_multi_agent_training_progress(
                    results['real_trajectories'], 
                    results['generated_trajectories'][1.0],
                    results['agent_masks'],
                    results['text_descriptions'],
                    epoch,
                    save_path=save_path,
                    n_samples=6
                )
                
                # Plot CFG comparison with full range
                cfg_save_path = os.path.join(self.exp_dir, f'cfg_comparison_epoch_{epoch:03d}_v{version}.png')
                self.plot_cfg_comparison(results, epoch, cfg_save_path)
            
            print(f"✅ Multi-agent test samples plotted and saved for epoch {epoch} (3 versions)")
            
        except Exception as e:
            print(f"⚠️  Failed to generate test samples: {str(e)}")
        
        # Switch back to training mode
        self.model.train()
    
    def plot_cfg_comparison(self, results: Dict[str, any], epoch: int, save_path: str):
        """Plot comparison of different CFG guidance scales."""
        utils.plot_cfg_comparison(
            real_trajectories=results['real_trajectories'],
            generated_trajectories_dict=results['generated_trajectories'],
            agent_masks=results['agent_masks'],
            text_descriptions=results['text_descriptions'],
            guidance_scales=results['guidance_scales'],
            epoch=epoch,
            save_path=save_path
        )
    
    def plot_final_results(self):
        """Generate comprehensive final results with diverse multi-agent text prompts and CFG analysis."""
        print("🎨 Generating comprehensive final multi-agent results with CFG analysis...")
        
        try:
            # Generate samples with comprehensive CFG scale range
            cfg_scales = [0.0, 1.0, 4.0, 7.0, 10.0, 13.0, 15.0, 20.0]
            results = self.generate_test_samples_with_cfg(n_samples=12, guidance_scales=cfg_scales)
            
            # Plot comprehensive CFG comparison
            cfg_comparison_path = os.path.join(self.exp_dir, 'final_cfg_comprehensive_comparison.png')
            utils.plot_cfg_comparison(
                real_trajectories=results['real_trajectories'][:4],  # Show 4 samples
                generated_trajectories_dict=results['generated_trajectories'],
                agent_masks=results['agent_masks'][:4],
                text_descriptions=results['text_descriptions'][:4],
                guidance_scales=[0.0, 1.0, 7.0, 13.0, 20.0],  # Key scales for comparison
                epoch=self.current_epoch,
                save_path=cfg_comparison_path,
                n_samples=4
            )
            
            # Plot CFG ablation study
            ablation_path = os.path.join(self.exp_dir, 'final_cfg_ablation_study.png')
            utils.plot_cfg_ablation_study(
                trajectories_dict=results['generated_trajectories'],
                agent_masks=results['agent_masks'],
                text_descriptions=results['text_descriptions'],
                guidance_scales=cfg_scales,
                save_path=ablation_path
            )
            
            # Generate and save CFG analysis
            analysis = utils.analyze_cfg_effects(
                trajectories_dict=results['generated_trajectories'],
                agent_masks=results['agent_masks'],
                guidance_scales=cfg_scales
            )
            
            # Print analysis to console
            utils.print_cfg_analysis(analysis)
            
            # Save analysis to file
            analysis_path = os.path.join(self.exp_dir, 'cfg_analysis.json')
            with open(analysis_path, 'w') as f:
                json.dump(analysis, f, indent=2)
            print(f"CFG analysis saved to: {analysis_path}")
            
            # Plot standard final results (backward compatibility)
            save_path = os.path.join(self.exp_dir, 'comprehensive_multi_agent_results.png')
            utils.plot_multi_agent_final_results(
                results['real_trajectories'],
                results['generated_trajectories'][1.0],  # Use standard CFG scale
                results['agent_masks'],
                results['text_descriptions'],
                title="Final Multi-Agent Text-Conditioned Results (CFG=1.0)",
                save_path=save_path,
                max_plots=12
            )
            
            print(f"✅ Comprehensive final multi-agent results with CFG analysis saved")
            
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
    exp_name: str = "multi_agent_trajectory_diffusion",
    cfg_dropout_prob: float = 0.15
) -> MultiAgentTrajectoryDiffusionTrainer:
    """Create a configured trainer for multi-agent text-conditioned trajectory diffusion with CFG."""
    
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
        max_agents=max_agents,
        cfg_dropout_prob=cfg_dropout_prob
    )
    
    return trainer


# Keep the original for backward compatibility
TrajectoryDiffusionTrainer = MultiAgentTrajectoryDiffusionTrainer
create_trainer = create_multi_agent_trainer


if __name__ == "__main__":
    # Create and run multi-agent trainer with CFG
    trainer = create_multi_agent_trainer(
        data_dir="data",
        batch_size=64,  # Smaller batch size for multi-agent due to increased memory usage
        sequence_length=100,
        max_agents=5,
        learning_rate=1e-4,
        num_train_timesteps=1000,
        text_emb_dim=384,
        exp_name="multi_agent_trajectory_diffusion_cfg",
        cfg_dropout_prob=0.15  # 15% CFG dropout for training
    )
    
    # Train the multi-agent model with CFG
    print("🎯 Training multi-agent model with Classifier-Free Guidance")
    print(f"📊 CFG dropout probability: {trainer.cfg_dropout_prob:.1%}")
    print("🎨 Generating 3 plot versions every epoch with full CFG range")
    trainer.train(
        num_epochs=50,
        save_every=5,
        validate_every=1,
        plot_every=1  # Generate CFG test plots every epoch (3 versions each)
    ) 