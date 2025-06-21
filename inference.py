"""
Inference Script for 2D Trajectory Diffusion Model
Generates new trajectories using trained model.
"""

import torch
import numpy as np
import matplotlib.pyplot as plt
import os
import argparse
from typing import Optional, List

from model import create_model
from diffusers import DDPMScheduler
import utils


class TrajectoryGenerator:
    """Generator for trajectory diffusion model."""
    
    def __init__(self, exp_dir: str, device: str = "cuda" if torch.cuda.is_available() else "cpu"):
        """
        Initialize generator.
        
        Args:
            exp_dir: Experiment directory containing trained model
            device: Device to run generation on
        """
        self.exp_dir = exp_dir
        self.device = device
        
        # Load configuration
        self.config = utils.load_config(exp_dir)
        print(f"Loaded config from: {exp_dir}")
        
        # Create model
        self.model = create_model(
            sequence_length=self.config['data_params']['sequence_length'],
            in_channels=self.config['model_params']['in_channels'],
            time_emb_dim=self.config['model_params']['time_emb_dim']
        ).to(device)
        
        # Create scheduler
        self.scheduler = DDPMScheduler(
            num_train_timesteps=self.config['scheduler_params']['num_train_timesteps'],
            beta_start=self.config['scheduler_params']['beta_start'],
            beta_end=self.config['scheduler_params']['beta_end'],
            beta_schedule=self.config['scheduler_params']['beta_schedule']
        )
        
        # Load trained model
        self._load_model()
        
        print(f"Generator initialized with device: {device}")
    
    def _load_model(self):
        """Load the trained model."""
        model_path = os.path.join(self.exp_dir, 'best_model.pth')
        if not os.path.exists(model_path):
            model_path = os.path.join(self.exp_dir, 'final_model.pth')
            if not os.path.exists(model_path):
                raise FileNotFoundError(f"No trained model found in {self.exp_dir}")
        
        checkpoint = torch.load(model_path, map_location=self.device, weights_only=False)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.model.eval()
        
        print(f"Loaded model from: {model_path}")
    
    def generate(
        self,
        num_samples: int = 1,
        seed: Optional[int] = None
    ) -> np.ndarray:
        """
        Generate trajectory samples.
        
        Args:
            num_samples: Number of trajectories to generate
            seed: Random seed for reproducible generation
        
        Returns:
            Generated trajectories [num_samples, seq_len, 2]
        """
        if seed is not None:
            torch.manual_seed(seed)
            if torch.cuda.is_available():
                torch.cuda.manual_seed(seed)
        
        print(f"Generating {num_samples} trajectory samples...")
        
        self.model.eval()
        sequence_length = self.config['data_params']['sequence_length']
        
        # Start with pure noise
        shape = (num_samples, 2, sequence_length)
        trajectories = torch.randn(shape, device=self.device)
        
        # Denoising loop with progress
        timesteps = list(reversed(range(self.scheduler.config.num_train_timesteps)))
        
        for i, t in enumerate(timesteps):
            if i % 100 == 0:
                print(f"Denoising step {i+1}/{len(timesteps)}")
            
            t_tensor = torch.full((num_samples,), t, device=self.device, dtype=torch.long)
            
            with torch.no_grad():
                # Predict noise
                noise_pred = self.model(trajectories, t_tensor)
                
                # Handle potential size mismatch due to model architecture
                if noise_pred.shape != trajectories.shape:
                    seq_len_pred = noise_pred.shape[-1]
                    trajectories = trajectories[..., :seq_len_pred]
                
                # Remove noise
                trajectories = self.scheduler.step(noise_pred, t, trajectories).prev_sample
        
        # Convert to numpy and transpose back to [batch, seq_len, 2]
        trajectories = trajectories.cpu().numpy().transpose(0, 2, 1)
        
        print("Generation completed!")
        return trajectories
    
    def generate_and_save(
        self,
        num_samples: int = 10,
        save_dir: Optional[str] = None,
        seed: Optional[int] = None,
        create_plots: bool = True,
        create_animations: bool = False
    ) -> np.ndarray:
        """
        Generate trajectories and save results.
        
        Args:
            num_samples: Number of trajectories to generate
            save_dir: Directory to save results (defaults to exp_dir)
            seed: Random seed
            create_plots: Whether to create plots
            create_animations: Whether to create animations
        
        Returns:
            Generated trajectories
        """
        if save_dir is None:
            save_dir = self.exp_dir
        
        # Generate trajectories
        trajectories = self.generate(num_samples, seed)
        
        # Save trajectories as numpy array
        traj_path = os.path.join(save_dir, 'generated_trajectories.npy')
        np.save(traj_path, trajectories)
        print(f"Trajectories saved: {traj_path}")
        
        if create_plots:
            # Create visualization plots
            plot_path = os.path.join(save_dir, 'generated_trajectories_inference.png')
            utils.plot_trajectories(
                trajectories,
                title=f"Generated Trajectories (n={num_samples})",
                save_path=plot_path
            )
            
            # Create individual trajectory plots
            self._create_individual_plots(trajectories, save_dir)
        
        if create_animations:
            # Create animations for first few trajectories
            n_animations = min(5, len(trajectories))
            for i in range(n_animations):
                animation_path = os.path.join(save_dir, f'trajectory_{i}_animation.gif')
                utils.create_animation(trajectories[i], animation_path)
        
        return trajectories
    
    def _create_individual_plots(self, trajectories: np.ndarray, save_dir: str):
        """Create individual plots for each trajectory."""
        individual_dir = os.path.join(save_dir, 'individual_trajectories')
        os.makedirs(individual_dir, exist_ok=True)
        
        for i, traj in enumerate(trajectories):
            plt.figure(figsize=(8, 8))
            plt.plot(traj[:, 0], traj[:, 1], 'b-', linewidth=2, alpha=0.8)
            plt.scatter(traj[0, 0], traj[0, 1], color='green', s=100, marker='o', label='Start', zorder=5)
            plt.scatter(traj[-1, 0], traj[-1, 1], color='red', s=100, marker='s', label='End', zorder=5)
            
            plt.xlim(-1.1, 1.1)
            plt.ylim(-1.1, 1.1)
            plt.gca().set_aspect('equal')
            plt.grid(True, alpha=0.3)
            plt.title(f'Generated Trajectory {i}')
            plt.legend()
            
            save_path = os.path.join(individual_dir, f'trajectory_{i:03d}.png')
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            plt.close()
        
        print(f"Individual trajectory plots saved in: {individual_dir}")
    
    def interpolate_trajectories(
        self,
        start_noise: Optional[torch.Tensor] = None,
        end_noise: Optional[torch.Tensor] = None,
        num_interpolations: int = 10,
        save_dir: Optional[str] = None
    ) -> np.ndarray:
        """
        Generate trajectories by interpolating between two noise vectors.
        
        Args:
            start_noise: Starting noise vector [2, seq_len] 
            end_noise: Ending noise vector [2, seq_len]
            num_interpolations: Number of interpolation steps
            save_dir: Directory to save results
        
        Returns:
            Interpolated trajectories [num_interpolations, seq_len, 2]
        """
        if save_dir is None:
            save_dir = self.exp_dir
        
        sequence_length = self.config['data_params']['sequence_length']
        
        # Generate random noise vectors if not provided
        if start_noise is None:
            start_noise = torch.randn(2, sequence_length, device=self.device)
        else:
            start_noise = start_noise.to(self.device)
        
        if end_noise is None:
            end_noise = torch.randn(2, sequence_length, device=self.device)
        else:
            end_noise = end_noise.to(self.device)
        
        # Create interpolation weights
        alphas = torch.linspace(0, 1, num_interpolations, device=self.device)
        
        interpolated_trajectories = []
        
        for alpha in alphas:
            # Interpolate noise
            noise = (1 - alpha) * start_noise + alpha * end_noise
            noise = noise.unsqueeze(0)  # Add batch dimension
            
            # Denoise
            trajectories = noise.clone()
            
            for t in reversed(range(self.scheduler.config.num_train_timesteps)):
                t_tensor = torch.full((1,), t, device=self.device, dtype=torch.long)
                
                with torch.no_grad():
                    noise_pred = self.model(trajectories, t_tensor)
                    
                    # Handle potential size mismatch due to model architecture
                    if noise_pred.shape != trajectories.shape:
                        seq_len_pred = noise_pred.shape[-1]
                        trajectories = trajectories[..., :seq_len_pred]
                    
                    trajectories = self.scheduler.step(noise_pred, t, trajectories).prev_sample
            
            # Convert to numpy
            traj = trajectories.cpu().numpy().transpose(0, 2, 1)[0]  # Remove batch dim
            interpolated_trajectories.append(traj)
        
        interpolated_trajectories = np.array(interpolated_trajectories)
        
        # Save and visualize
        interp_path = os.path.join(save_dir, 'interpolated_trajectories.npy')
        np.save(interp_path, interpolated_trajectories)
        
        # Create interpolation plot
        plt.figure(figsize=(15, 3))
        for i, traj in enumerate(interpolated_trajectories):
            plt.subplot(1, num_interpolations, i + 1)
            plt.plot(traj[:, 0], traj[:, 1], 'b-', linewidth=2)
            plt.xlim(-1.1, 1.1)
            plt.ylim(-1.1, 1.1)
            plt.gca().set_aspect('equal')
            plt.title(f'α={alphas[i]:.2f}')
            plt.grid(True, alpha=0.3)
        
        plt.suptitle('Trajectory Interpolation')
        plt.tight_layout()
        
        interp_plot_path = os.path.join(save_dir, 'trajectory_interpolation.png')
        plt.savefig(interp_plot_path, dpi=150, bbox_inches='tight')
        plt.close()
        
        print(f"Interpolation results saved: {interp_path}")
        print(f"Interpolation plot saved: {interp_plot_path}")
        
        return interpolated_trajectories


def main():
    """Main inference function."""
    parser = argparse.ArgumentParser(description='Generate trajectories using trained diffusion model')
    parser.add_argument('--exp_idx', type=int, required=True, help='Experiment index')
    parser.add_argument('--exp_name', type=str, default='trajectory_diffusion', help='Experiment name')
    parser.add_argument('--num_samples', type=int, default=10, help='Number of trajectories to generate')
    parser.add_argument('--seed', type=int, default=None, help='Random seed for reproducible generation')
    parser.add_argument('--create_plots', action='store_true', help='Create visualization plots')
    parser.add_argument('--create_animations', action='store_true', help='Create trajectory animations')
    parser.add_argument('--interpolate', action='store_true', help='Generate trajectory interpolations')
    parser.add_argument('--num_interpolations', type=int, default=10, help='Number of interpolation steps')
    
    args = parser.parse_args()
    
    # Find experiment directory
    exp_dir = utils.find_experiment_dir(args.exp_idx, args.exp_name)
    
    # Create generator
    generator = TrajectoryGenerator(exp_dir)
    
    # Generate trajectories
    trajectories = generator.generate_and_save(
        num_samples=args.num_samples,
        seed=args.seed,
        create_plots=args.create_plots,
        create_animations=args.create_animations
    )
    
    print(f"Generated {len(trajectories)} trajectories")
    print(f"Trajectory shape: {trajectories.shape}")
    
    # Generate interpolations if requested
    if args.interpolate:
        interpolated = generator.interpolate_trajectories(
            num_interpolations=args.num_interpolations
        )
        print(f"Generated {len(interpolated)} interpolated trajectories")
    
    print(f"\nInference completed! Results saved in: {exp_dir}")


if __name__ == "__main__":
    main() 