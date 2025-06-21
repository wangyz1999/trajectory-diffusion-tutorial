"""
Evaluation Script for 2D Trajectory Diffusion Model
Loads trained model and evaluates on test data.
"""

import torch
import numpy as np
import matplotlib.pyplot as plt
import os
import json
import argparse
from typing import Dict, List

from dataset import TrajectoryDataModule
from model import create_model
from diffusers import DDPMScheduler
import utils


class TrajectoryDiffusionEvaluator:
    """Evaluator for trajectory diffusion model."""
    
    def __init__(self, exp_dir: str, device: str = "cuda" if torch.cuda.is_available() else "cpu"):
        """
        Initialize evaluator.
        
        Args:
            exp_dir: Experiment directory containing trained model
            device: Device to run evaluation on
        """
        self.exp_dir = exp_dir
        self.device = device
        
        # Load configuration
        self.config = utils.load_config(exp_dir)
        print(f"Loaded config from: {exp_dir}")
        
        # Load data module
        self.data_module = TrajectoryDataModule(
            data_dir=self.config['data_params']['data_dir'] if 'data_dir' in self.config['data_params'] else "data",
            batch_size=self.config['data_params']['batch_size'],
            sequence_length=self.config['data_params']['sequence_length'],
        )
        
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
        self._load_best_model()
        
        print(f"Evaluator initialized with device: {device}")
    
    def _load_best_model(self):
        """Load the best trained model."""
        model_path = os.path.join(self.exp_dir, 'best_model.pth')
        if not os.path.exists(model_path):
            # Try final model if best model doesn't exist
            model_path = os.path.join(self.exp_dir, 'final_model.pth')
            if not os.path.exists(model_path):
                raise FileNotFoundError(f"No trained model found in {self.exp_dir}")
        
        checkpoint = torch.load(model_path, map_location=self.device, weights_only=False)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.model.eval()
        
        print(f"Loaded model from: {model_path}")
        print(f"Best validation loss: {checkpoint.get('best_val_loss', 'N/A')}")
    
    def evaluate_reconstruction(self, num_samples: int = 100) -> Dict[str, float]:
        """
        Evaluate model's reconstruction capability.
        
        Args:
            num_samples: Number of test samples to evaluate
        
        Returns:
            Dictionary of reconstruction metrics
        """
        print("Evaluating reconstruction...")
        
        self.model.eval()
        reconstruction_losses = []
        
        with torch.no_grad():
            sample_count = 0
            for batch in self.data_module.test_dataloader():
                if sample_count >= num_samples:
                    break
                
                trajectories = batch['trajectory'].to(self.device)
                batch_size = trajectories.shape[0]
                
                # Transpose to [batch, 2, seq_len]
                trajectories = trajectories.transpose(1, 2)
                
                # Test reconstruction at different noise levels
                for t in [0, 100, 250, 500, 750, 999]:  # Different timesteps
                    timesteps = torch.full((batch_size,), t, device=self.device, dtype=torch.long)
                    
                    # Add noise
                    noise = torch.randn_like(trajectories)
                    noisy_trajectories = self.scheduler.add_noise(trajectories, noise, timesteps)
                    
                    # Predict noise
                    predicted_noise = self.model(noisy_trajectories, timesteps)
                    
                    # Handle potential size mismatch due to model architecture
                    if predicted_noise.shape != noise.shape:
                        # Crop noise to match prediction size
                        seq_len_pred = predicted_noise.shape[-1]
                        noise = noise[..., :seq_len_pred]
                        noisy_trajectories = noisy_trajectories[..., :seq_len_pred]
                    
                    # Compute loss
                    loss = torch.nn.functional.mse_loss(predicted_noise, noise)
                    reconstruction_losses.append(loss.item())
                
                sample_count += batch_size
        
        metrics = {
            'reconstruction_loss_mean': float(np.mean(reconstruction_losses)),
            'reconstruction_loss_std': float(np.std(reconstruction_losses)),
            'reconstruction_loss_min': float(np.min(reconstruction_losses)),
            'reconstruction_loss_max': float(np.max(reconstruction_losses))
        }
        
        return metrics
    
    def generate_samples(self, num_samples: int = 64) -> np.ndarray:
        """
        Generate trajectory samples using the trained model.
        
        Args:
            num_samples: Number of samples to generate
        
        Returns:
            Generated trajectories [num_samples, seq_len, 2]
        """
        print(f"Generating {num_samples} trajectory samples...")
        
        self.model.eval()
        sequence_length = self.config['data_params']['sequence_length']
        
        # Start with pure noise
        shape = (num_samples, 2, sequence_length)
        trajectories = torch.randn(shape, device=self.device)
        
        # Denoising loop
        for t in reversed(range(self.scheduler.config.num_train_timesteps)):
            timesteps = torch.full((num_samples,), t, device=self.device, dtype=torch.long)
            
            with torch.no_grad():
                # Predict noise
                noise_pred = self.model(trajectories, timesteps)
                
                # Handle potential size mismatch due to model architecture
                if noise_pred.shape != trajectories.shape:
                    seq_len_pred = noise_pred.shape[-1]
                    trajectories = trajectories[..., :seq_len_pred]
                
                # Remove noise
                trajectories = self.scheduler.step(noise_pred, t, trajectories).prev_sample
        
        # Convert to numpy and transpose back to [batch, seq_len, 2]
        trajectories = trajectories.cpu().numpy().transpose(0, 2, 1)
        
        return trajectories
    
    def evaluate_sample_quality(self, num_samples: int = 100) -> Dict[str, float]:
        """
        Evaluate quality of generated samples.
        
        Args:
            num_samples: Number of samples to generate and evaluate
        
        Returns:
            Dictionary of quality metrics
        """
        print("Evaluating sample quality...")
        
        # Generate samples
        generated_trajectories = self.generate_samples(num_samples)
        
        # Get real test samples
        real_trajectories = []
        sample_count = 0
        
        for batch in self.data_module.test_dataloader():
            if sample_count >= num_samples:
                break
            
            trajectories = batch['trajectory'].numpy()  # [batch, seq_len, 2]
            real_trajectories.append(trajectories)
            sample_count += len(trajectories)
        
        real_trajectories = np.concatenate(real_trajectories, axis=0)[:num_samples]
        
        # Compute metrics
        metrics = utils.compute_trajectory_metrics(real_trajectories, generated_trajectories)
        
        return metrics, real_trajectories, generated_trajectories
    
    def create_evaluation_plots(self, real_trajectories: np.ndarray, generated_trajectories: np.ndarray):
        """Create evaluation plots."""
        print("Creating evaluation plots...")
        
        # Comparison plot
        comparison_path = os.path.join(self.exp_dir, 'trajectory_comparison.png')
        utils.compare_trajectories(
            real_trajectories, generated_trajectories,
            save_path=comparison_path, n_samples=8
        )
        
        # Generated samples plot
        generated_path = os.path.join(self.exp_dir, 'generated_trajectories.png')
        utils.plot_trajectories(
            generated_trajectories[:25],
            title="Generated Trajectories",
            save_path=generated_path
        )
        
        # Real samples plot
        real_path = os.path.join(self.exp_dir, 'real_trajectories.png')
        utils.plot_trajectories(
            real_trajectories[:25],
            title="Real Test Trajectories",
            save_path=real_path
        )
        
        # Distribution comparison
        self._plot_distribution_comparison(real_trajectories, generated_trajectories)
    
    def _plot_distribution_comparison(self, real_trajectories: np.ndarray, generated_trajectories: np.ndarray):
        """Plot distribution comparison between real and generated trajectories."""
        fig, axes = plt.subplots(2, 3, figsize=(15, 10))
        
        # X coordinate distribution
        axes[0, 0].hist(real_trajectories[:, :, 0].flatten(), bins=50, alpha=0.7, label='Real', density=True)
        axes[0, 0].hist(generated_trajectories[:, :, 0].flatten(), bins=50, alpha=0.7, label='Generated', density=True)
        axes[0, 0].set_title('X Coordinate Distribution')
        axes[0, 0].legend()
        axes[0, 0].grid(True, alpha=0.3)
        
        # Y coordinate distribution
        axes[0, 1].hist(real_trajectories[:, :, 1].flatten(), bins=50, alpha=0.7, label='Real', density=True)
        axes[0, 1].hist(generated_trajectories[:, :, 1].flatten(), bins=50, alpha=0.7, label='Generated', density=True)
        axes[0, 1].set_title('Y Coordinate Distribution')
        axes[0, 1].legend()
        axes[0, 1].grid(True, alpha=0.3)
        
        # Trajectory length distribution
        real_lengths = np.sqrt(np.sum(np.diff(real_trajectories, axis=1) ** 2, axis=2)).sum(axis=1)
        gen_lengths = np.sqrt(np.sum(np.diff(generated_trajectories, axis=1) ** 2, axis=2)).sum(axis=1)
        
        axes[0, 2].hist(real_lengths, bins=30, alpha=0.7, label='Real', density=True)
        axes[0, 2].hist(gen_lengths, bins=30, alpha=0.7, label='Generated', density=True)
        axes[0, 2].set_title('Trajectory Length Distribution')
        axes[0, 2].legend()
        axes[0, 2].grid(True, alpha=0.3)
        
        # Start point distribution
        axes[1, 0].scatter(real_trajectories[:, 0, 0], real_trajectories[:, 0, 1], alpha=0.5, label='Real', s=10)
        axes[1, 0].scatter(generated_trajectories[:, 0, 0], generated_trajectories[:, 0, 1], alpha=0.5, label='Generated', s=10)
        axes[1, 0].set_title('Start Point Distribution')
        axes[1, 0].legend()
        axes[1, 0].grid(True, alpha=0.3)
        axes[1, 0].set_aspect('equal')
        
        # End point distribution
        axes[1, 1].scatter(real_trajectories[:, -1, 0], real_trajectories[:, -1, 1], alpha=0.5, label='Real', s=10)
        axes[1, 1].scatter(generated_trajectories[:, -1, 0], generated_trajectories[:, -1, 1], alpha=0.5, label='Generated', s=10)
        axes[1, 1].set_title('End Point Distribution')
        axes[1, 1].legend()
        axes[1, 1].grid(True, alpha=0.3)
        axes[1, 1].set_aspect('equal')
        
        # Velocity distribution
        real_velocities = np.linalg.norm(np.diff(real_trajectories, axis=1), axis=2).flatten()
        gen_velocities = np.linalg.norm(np.diff(generated_trajectories, axis=1), axis=2).flatten()
        
        axes[1, 2].hist(real_velocities, bins=50, alpha=0.7, label='Real', density=True)
        axes[1, 2].hist(gen_velocities, bins=50, alpha=0.7, label='Generated', density=True)
        axes[1, 2].set_title('Velocity Distribution')
        axes[1, 2].legend()
        axes[1, 2].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.exp_dir, 'distribution_comparison.png'), dpi=150, bbox_inches='tight')
        plt.close()
    
    def run_full_evaluation(self) -> Dict:
        """Run complete evaluation."""
        print("Starting full evaluation...")
        
        results = {}
        
        # Reconstruction evaluation
        reconstruction_metrics = self.evaluate_reconstruction()
        results['reconstruction'] = reconstruction_metrics
        
        # Sample quality evaluation
        quality_metrics, real_trajectories, generated_trajectories = self.evaluate_sample_quality()
        results['sample_quality'] = quality_metrics
        
        # Create plots
        self.create_evaluation_plots(real_trajectories, generated_trajectories)
        
        # Create animation of sample trajectory
        if len(generated_trajectories) > 0:
            animation_path = os.path.join(self.exp_dir, 'sample_trajectory_animation.gif')
            utils.create_animation(generated_trajectories[0], animation_path)
        
        # Save results
        utils.save_results(results, self.exp_dir, 'evaluation_results.json')
        
        # Print summary
        self._print_evaluation_summary(results)
        
        return results
    
    def _print_evaluation_summary(self, results: Dict):
        """Print evaluation summary."""
        print("\n" + "="*60)
        print("EVALUATION SUMMARY")
        print("="*60)
        
        # Reconstruction metrics
        recon = results['reconstruction']
        print(f"Reconstruction Loss: {recon['reconstruction_loss_mean']:.4f} ± {recon['reconstruction_loss_std']:.4f}")
        
        # Sample quality metrics
        quality = results['sample_quality']
        print(f"\nSample Quality Metrics:")
        print(f"  MSE: {quality['mse']:.4f}")
        print(f"  MAE: {quality['mae']:.4f}")
        print(f"  Frechet Distance: {quality['frechet_distance_mean']:.4f} ± {quality['frechet_distance_std']:.4f}")
        print(f"  Endpoint Distance: {quality['endpoint_distance_mean']:.4f} ± {quality['endpoint_distance_std']:.4f}")
        
        print(f"\nTrajectory Length:")
        print(f"  Real: {quality['real_length_mean']:.4f} ± {quality['real_length_std']:.4f}")
        print(f"  Generated: {quality['gen_length_mean']:.4f} ± {quality['gen_length_std']:.4f}")
        
        print("="*60)


def main():
    """Main evaluation function."""
    parser = argparse.ArgumentParser(description='Evaluate trajectory diffusion model')
    parser.add_argument('--exp_idx', type=int, required=True, help='Experiment index')
    parser.add_argument('--exp_name', type=str, default='trajectory_diffusion', help='Experiment name')
    parser.add_argument('--num_samples', type=int, default=100, help='Number of samples for evaluation')
    
    args = parser.parse_args()
    
    # Find experiment directory
    exp_dir = utils.find_experiment_dir(args.exp_idx, args.exp_name)
    
    # Create evaluator
    evaluator = TrajectoryDiffusionEvaluator(exp_dir)
    
    # Run evaluation
    results = evaluator.run_full_evaluation()
    
    print(f"\nEvaluation completed! Results saved in: {exp_dir}")


if __name__ == "__main__":
    main() 