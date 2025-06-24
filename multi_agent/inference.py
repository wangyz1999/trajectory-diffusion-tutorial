"""
Inference Script for Text-Conditioned 2D Trajectory Diffusion Model
Loads trained models and generates trajectories with visualizations.
"""

import os
import json
import torch
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from tqdm import tqdm
from typing import List, Optional, Tuple
import argparse

from single_agent.dataset import TrajectoryDataModule, TextEncoder
from model import create_model
from diffusers import DDPMScheduler
import utils


class TrajectoryDiffusionInference:
    """Inference class for text-conditioned trajectory generation."""
    
    def __init__(self, exp_dir: str, device: str = "cuda" if torch.cuda.is_available() else "cpu"):
        """
        Initialize inference from experiment directory.
        
        Args:
            exp_dir: Experiment directory containing model and config
            device: Device to run inference on
        """
        self.exp_dir = exp_dir
        self.device = device
        
        # Load configuration
        self.config = utils.load_config(exp_dir)
        print(f"📂 Loaded config from: {exp_dir}")
        
        # Create model
        self.model = create_model(
            sequence_length=self.config['data_params']['sequence_length'],
            text_emb_dim=self.config['model_params']['text_emb_dim']
        )
        
        # Load trained weights
        self.load_model()
        
        # Create scheduler
        scheduler_config = self.config['scheduler_params']
        self.scheduler = DDPMScheduler(
            num_train_timesteps=scheduler_config['num_train_timesteps'],
            beta_start=scheduler_config['beta_start'],
            beta_end=scheduler_config['beta_end'],
            beta_schedule=scheduler_config['beta_schedule'],
            clip_sample=True
        )
        
        # Create text encoder
        self.text_encoder = TextEncoder(device=device)
        
        print(f"✅ Model loaded successfully on {device}")
        print(f"🔢 Model parameters: {sum(p.numel() for p in self.model.parameters()):,}")
    
    def load_model(self, checkpoint_name: str = "best_model.pth"):
        """Load model checkpoint."""
        checkpoint_path = os.path.join(self.exp_dir, checkpoint_name)
        
        if not os.path.exists(checkpoint_path):
            raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
        
        checkpoint = torch.load(checkpoint_path, map_location=self.device, weights_only=False)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.model.to(self.device)
        self.model.eval()
        
        print(f"📦 Loaded checkpoint: {checkpoint_path}")
        if 'best_val_loss' in checkpoint:
            print(f"📊 Best validation loss: {checkpoint['best_val_loss']:.4f}")
    
    def generate_trajectory(
        self, 
        text_description: str, 
        save_steps: bool = False,
        num_inference_steps: Optional[int] = None
    ) -> Tuple[np.ndarray, Optional[List[np.ndarray]]]:
        """
        Generate a single trajectory from text description.
        
        Args:
            text_description: Text description of desired trajectory
            save_steps: Whether to save intermediate steps for animation
            num_inference_steps: Number of inference steps (default: use all)
            
        Returns:
            Generated trajectory [seq_len, 2] and optionally list of intermediate steps
        """
        # Encode text
        text_embedding = self.text_encoder.encode([text_description])
        text_embedding = text_embedding.to(self.device)
        
        # Get sequence length from config
        seq_len = self.config['data_params']['sequence_length']
        
        # Start from pure noise
        trajectory = torch.randn(1, 2, seq_len).to(self.device)
        
        # Determine steps to use
        total_steps = self.scheduler.config.num_train_timesteps
        if num_inference_steps is None:
            steps = list(reversed(range(total_steps)))
        else:
            # Use evenly spaced steps for faster inference
            step_size = total_steps // num_inference_steps
            steps = list(reversed(range(0, total_steps, step_size)))
        
        intermediate_steps = [] if save_steps else None
        
        with torch.no_grad():
            for i, t in enumerate(tqdm(steps, desc="🎨 Generating trajectory")):
                timesteps = torch.full((1,), t, device=self.device).long()
                
                # Predict noise
                noise_pred = self.model(trajectory, timesteps, text_embedding)
                
                # Remove predicted noise
                trajectory = self.scheduler.step(noise_pred, t, trajectory).prev_sample
                
                # Save intermediate step (every 30 frames + always save the last frame)
                if save_steps and (i % 30 == 0 or i == len(steps) - 1):
                    step_traj = trajectory.cpu().numpy().transpose(0, 2, 1)[0]  # [seq_len, 2]
                    intermediate_steps.append((step_traj.copy(), i, len(steps)))
        
        # Convert final result
        final_trajectory = trajectory.cpu().numpy().transpose(0, 2, 1)[0]  # [seq_len, 2]
        
        # Add multiple copies of the final frame to make it stay longer
        if save_steps and intermediate_steps:
            final_step_info = intermediate_steps[-1]
            # Add 15 more copies of the final frame (2 seconds at 8 fps)
            for _ in range(15):
                intermediate_steps.append(final_step_info)
        
        return final_trajectory, intermediate_steps
    
    def generate_batch_trajectories(
        self, 
        text_descriptions: List[str],
        num_inference_steps: Optional[int] = None
    ) -> np.ndarray:
        """
        Generate multiple trajectories from text descriptions.
        
        Args:
            text_descriptions: List of text descriptions
            num_inference_steps: Number of inference steps for faster generation
            
        Returns:
            Generated trajectories [n_samples, seq_len, 2]
        """
        # Encode all texts
        text_embeddings = self.text_encoder.encode(text_descriptions)
        text_embeddings = text_embeddings.to(self.device)
        
        batch_size = len(text_descriptions)
        seq_len = self.config['data_params']['sequence_length']
        
        # Start from pure noise
        trajectories = torch.randn(batch_size, 2, seq_len).to(self.device)
        
        # Determine steps to use
        total_steps = self.scheduler.config.num_train_timesteps
        if num_inference_steps is None:
            steps = list(reversed(range(total_steps)))
        else:
            step_size = total_steps // num_inference_steps
            steps = list(reversed(range(0, total_steps, step_size)))
        
        with torch.no_grad():
            for t in tqdm(steps, desc="🎨 Generating trajectories"):
                timesteps = torch.full((batch_size,), t, device=self.device).long()
                
                # Predict noise
                noise_pred = self.model(trajectories, timesteps, text_embeddings)
                
                # Remove predicted noise
                trajectories = self.scheduler.step(noise_pred, t, trajectories).prev_sample
        
        # Convert to numpy
        trajectories = trajectories.cpu().numpy().transpose(0, 2, 1)  # [batch_size, seq_len, 2]
        
        return trajectories
    
    def create_diffusion_animation(
        self, 
        text_description: str, 
        save_path: str,
        fps: int = 10,
        num_inference_steps: int = 50
    ):
        """
        Create animation showing the diffusion process.
        
        Args:
            text_description: Text description for trajectory
            save_path: Path to save animation GIF
            fps: Frames per second
            num_inference_steps: Number of steps to visualize
        """
        print(f"🎬 Creating diffusion animation for: '{text_description}'")
        
        # Generate trajectory with intermediate steps
        final_trajectory, intermediate_steps = self.generate_trajectory(
            text_description, 
            save_steps=True,
            num_inference_steps=num_inference_steps
        )
        
        if not intermediate_steps:
            print("⚠️  No intermediate steps saved")
            return
        
        # Create animation
        fig, ax = plt.subplots(figsize=(10, 10))
        ax.set_xlim(-1.5, 1.5)
        ax.set_ylim(-1.5, 1.5)
        ax.set_aspect('equal')
        ax.grid(True, alpha=0.3)
        ax.set_title(f'Diffusion Process: "{text_description}"', fontsize=14, wrap=True)
        
        line, = ax.plot([], [], 'b-', linewidth=2, alpha=0.8)
        points = ax.scatter([], [], c=[], s=30, cmap='viridis', alpha=0.8)
        progress_text = ax.text(0.02, 0.95, '', transform=ax.transAxes, 
                               fontsize=12, bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.8))
        
        def animate(frame):
            if frame < len(intermediate_steps):
                traj, step_idx, total_steps = intermediate_steps[frame]
                
                # Update line
                line.set_data(traj[:, 0], traj[:, 1])
                
                # Update points with color gradient
                colors = np.linspace(0, 1, len(traj))
                points.set_offsets(traj)
                points.set_array(colors)
                
                # Update progress text with actual step numbers
                progress = (step_idx + 1) / total_steps
                progress_text.set_text(f'Denoising Progress: {progress:.1%}\nStep {step_idx+1}/{total_steps}')
            
            return line, points, progress_text
        
        anim = animation.FuncAnimation(
            fig, animate, frames=len(intermediate_steps),
            interval=1000//fps, blit=True, repeat=True
        )
        
        # Save animation
        anim.save(save_path, writer='pillow', fps=fps)
        plt.close()
        
        print(f"🎬 Animation saved: {save_path}")
    
    def create_ground_truth_comparison(
        self, 
        output_dir: str, 
        num_samples: int = 12,
        num_inference_steps: int = 1000
    ):
        """
        Create side-by-side comparison plots of ground truth vs generated trajectories.
        
        Args:
            output_dir: Directory to save results
            num_samples: Number of samples to compare
            num_inference_steps: Number of inference steps for generation
        """
        print(f"📊 Creating ground truth comparison with {num_samples} samples...")
        print(f"🔍 Debug: num_samples parameter = {num_samples}")
        
        # Load dataset with larger test split to get more samples
        data_module = TrajectoryDataModule(
            data_dir="data",
            batch_size=32,  # Don't limit batch size by num_samples
            sequence_length=self.config['data_params']['sequence_length'],
            train_split=0.6,  # Reduce train split
            val_split=0.2,   # Increase val split  
            # Test split will be 0.2 (20%) - much larger
        )
        
        print(f"📊 Dataset splits: Train={len(data_module.train_dataset)}, Val={len(data_module.val_dataset)}, Test={len(data_module.test_dataset)}")
        
        # Get samples directly from the test dataset (not limited by batch size)
        test_dataset = data_module.test_dataset
        n_test_samples = len(test_dataset)
        print(f"🎯 Found {n_test_samples} test samples in dataset")
        
        # Select the number of samples we want
        n_samples = min(num_samples, n_test_samples)
        print(f"📝 Selecting {n_samples} samples for comparison")
        
        # Collect samples directly from dataset
        gt_trajectories = []
        text_descriptions = []
        pattern_labels = []
        
        # Get samples from different indices to ensure variety
        if n_test_samples >= num_samples:
            # Spread samples across the test set for variety
            indices = np.linspace(0, n_test_samples-1, num_samples, dtype=int)
        else:
            # Use all available test samples
            indices = list(range(n_test_samples))
        
        print(f"🎲 Using test sample indices: {indices}")
        
        for i, idx in enumerate(indices):
            sample = test_dataset[idx]
            gt_trajectories.append(sample['trajectory'].numpy())
            text_descriptions.append(sample['text_description'])
            pattern_labels.append(sample['label_name'])
            
            if i < 5:  # Show first few for debugging
                print(f"  Sample {i}: {sample['label_name']} - '{sample['text_description'][:50]}...'")
        
        gt_trajectories = np.array(gt_trajectories)  # [n_samples, seq_len, 2]
        split_used = "test"
        
        print(f"🎯 Using {n_samples} samples from {split_used} set for comparison...")
        
        # Generate trajectories using the same text descriptions
        generated_trajectories = self.generate_batch_trajectories(
            text_descriptions, 
            num_inference_steps=num_inference_steps
        )
        
        # Create detailed side-by-side comparison
        self._create_detailed_comparison_plot(
            gt_trajectories, 
            generated_trajectories, 
            text_descriptions,
            pattern_labels,
            output_dir,
            num_inference_steps
        )
        
        # Save comparison data
        comparison_data = {
            'ground_truth_trajectories': gt_trajectories.tolist(),
            'generated_trajectories': generated_trajectories.tolist(),
            'text_descriptions': text_descriptions,
            'pattern_labels': pattern_labels,
            'num_inference_steps': num_inference_steps,
            'split_used': split_used,
            'total_samples_available': n_test_samples,
            'selected_samples': n_samples
        }
        
        with open(os.path.join(output_dir, 'ground_truth_comparison.json'), 'w') as f:
            json.dump(comparison_data, f, indent=2)
        
        print(f"✅ Ground truth comparison completed and saved with {n_samples} samples from {split_used} set!")
    
    def _create_detailed_comparison_plot(
        self,
        gt_trajectories: np.ndarray,
        generated_trajectories: np.ndarray,
        text_descriptions: List[str],
        pattern_labels: List[str],
        output_dir: str,
        num_inference_steps: int
    ):
        """Create detailed side-by-side comparison plots."""
        n_samples = len(gt_trajectories)
        
        # Determine grid layout
        if n_samples <= 6:
            rows, cols = 2, 3
            figsize = (18, 12)
        elif n_samples <= 8:
            rows, cols = 2, 4
            figsize = (24, 12)
        elif n_samples <= 12:
            rows, cols = 3, 4
            figsize = (24, 18)
        else:
            rows, cols = 4, 4
            figsize = (24, 24)
        
        fig, axes = plt.subplots(rows, cols, figsize=figsize)
        if rows * cols == 1:
            axes = [axes]
        else:
            axes = axes.flatten()
        
        colors = ['blue', 'red']
        labels = ['Ground Truth', 'Generated']
        
        for i in range(min(n_samples, len(axes))):
            ax = axes[i]
            
            gt_traj = gt_trajectories[i]
            gen_traj = generated_trajectories[i]
            
            # Plot ground truth
            ax.plot(gt_traj[:, 0], gt_traj[:, 1], 
                   color=colors[0], linewidth=2.5, alpha=0.8, label=labels[0])
            ax.plot(gt_traj[0, 0], gt_traj[0, 1], 'go', markersize=8, label='Start')
            ax.plot(gt_traj[-1, 0], gt_traj[-1, 1], 'ko', markersize=8, label='End')
            
            # Plot generated trajectory
            ax.plot(gen_traj[:, 0], gen_traj[:, 1], 
                   color=colors[1], linewidth=2.5, alpha=0.8, label=labels[1], linestyle='--')
            ax.plot(gen_traj[0, 0], gen_traj[0, 1], 'g^', markersize=8)
            ax.plot(gen_traj[-1, 0], gen_traj[-1, 1], 'k^', markersize=8)
            
            ax.set_aspect('equal')
            ax.grid(True, alpha=0.3)
            ax.set_xlim(-1.2, 1.2)
            ax.set_ylim(-1.2, 1.2)
            
            # Title with pattern and text description
            title = f"{pattern_labels[i]}\n\"{text_descriptions[i][:40]}{'...' if len(text_descriptions[i]) > 40 else ''}\""
            ax.set_title(title, fontsize=10, wrap=True)
            
            if i == 0:  # Only show legend on first plot
                ax.legend(fontsize=9, loc='upper right')
        
        # Remove empty subplots
        for i in range(n_samples, len(axes)):
            fig.delaxes(axes[i])
        
        plt.suptitle(f'Ground Truth vs Generated Trajectories ({num_inference_steps} steps)', fontsize=16)
        plt.tight_layout()
        
        # Save comparison plot
        save_path = os.path.join(output_dir, f'ground_truth_vs_generated_comparison_{num_inference_steps}steps.png')
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()
        
        print(f"📊 Detailed comparison plot saved: {save_path}")
        
        # Also create a simpler 2-column layout for easier viewing
        self._create_simple_comparison_plot(
            gt_trajectories, 
            generated_trajectories, 
            text_descriptions,
            pattern_labels,
            output_dir,
            num_inference_steps
        )
    
    def _create_simple_comparison_plot(
        self,
        gt_trajectories: np.ndarray,
        generated_trajectories: np.ndarray,
        text_descriptions: List[str],
        pattern_labels: List[str],
        output_dir: str,
        num_inference_steps: int
    ):
        """Create 6x6 grid comparison plot showing up to 18 examples with text prompts as titles."""
        n_samples = min(len(gt_trajectories), 18)  # Show up to 18 examples
        
        # Create 6x6 grid
        fig, axes = plt.subplots(6, 6, figsize=(36, 36))
        
        # Define positions for up to 18 examples in a 6x6 grid
        # Layout: 3 examples per row, 6 rows total = 18 examples max
        # Each example takes 2 columns (GT + Generated)
        positions = []
        for row in range(6):  # All 6 rows
            for col in range(0, 6, 2):  # Cols 0, 2, 4 (3 pairs per row)
                positions.append((row, col))
        
        for i in range(n_samples):
            if i >= len(positions):
                break
                
            row, col = positions[i]
            gt_traj = gt_trajectories[i]
            gen_traj = generated_trajectories[i]
            
            # Ground truth plot
            ax_gt = axes[row, col]
            ax_gt.plot(gt_traj[:, 0], gt_traj[:, 1], 'b-', linewidth=3, alpha=0.8)
            ax_gt.plot(gt_traj[0, 0], gt_traj[0, 1], 'go', markersize=10)
            ax_gt.plot(gt_traj[-1, 0], gt_traj[-1, 1], 'ro', markersize=10)
            ax_gt.set_aspect('equal')
            ax_gt.grid(True, alpha=0.3)
            ax_gt.set_xlim(-1.2, 1.2)
            ax_gt.set_ylim(-1.2, 1.2)
            ax_gt.set_title('Ground Truth', fontsize=14, fontweight='bold')
            
            # Generated plot
            ax_gen = axes[row, col + 1]
            ax_gen.plot(gen_traj[:, 0], gen_traj[:, 1], 'r-', linewidth=3, alpha=0.8)
            ax_gen.plot(gen_traj[0, 0], gen_traj[0, 1], 'go', markersize=10)
            ax_gen.plot(gen_traj[-1, 0], gen_traj[-1, 1], 'ro', markersize=10)
            ax_gen.set_aspect('equal')
            ax_gen.grid(True, alpha=0.3)
            ax_gen.set_xlim(-1.2, 1.2)
            ax_gen.set_ylim(-1.2, 1.2)
            ax_gen.set_title('Generated', fontsize=14, fontweight='bold')
            
            # Add text prompt as subplot title (more compact)
            combined_title = f'"{text_descriptions[i][:30]}{"..." if len(text_descriptions[i]) > 30 else ""}"'
            
            # Create a text annotation above the pair
            fig.text(
                0.17 + (col // 2) * 0.33,  # X position based on column pair
                0.95 - row * 0.155,  # Y position based on row
                combined_title,
                ha='center', va='center', 
                fontsize=12, fontweight='bold', style='italic',
                bbox=dict(boxstyle="round,pad=0.5", facecolor="lightblue", alpha=0.8),
                wrap=True
            )
        
        # Turn off all unused subplots
        used_positions = set()
        for i in range(n_samples):
            if i < len(positions):
                row, col = positions[i]
                used_positions.add((row, col))
                used_positions.add((row, col + 1))
        
        for row in range(6):
            for col in range(6):
                if (row, col) not in used_positions:
                    axes[row, col].axis('off')
        
        plt.suptitle('Ground Truth vs Generated Trajectories - 6x6 Grid Comparison', 
                    fontsize=24, y=0.98, fontweight='bold')
        plt.tight_layout()
        plt.subplots_adjust(top=0.95, hspace=0.15, wspace=0.15)
        
        # Save simple comparison plot
        save_path = os.path.join(output_dir, f'simple_ground_truth_vs_generated_{num_inference_steps}steps.png')
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()
        
        print(f"📊 6x6 grid comparison plot with {n_samples} examples saved: {save_path}")
    
    def run_comprehensive_inference(self, output_dir: Optional[str] = None, num_samples: int = 12):
        """
        Run comprehensive inference with various visualizations.
        
        Args:
            output_dir: Directory to save results (default: exp_dir)
            num_samples: Number of samples to generate
        """
        if output_dir is None:
            output_dir = self.exp_dir
        
        print(f"🚀 Running comprehensive inference...")
        print(f"📁 Output directory: {output_dir}")
        
        # Define diverse text prompts
        text_prompts = [
            "A perfect circle",
            "A spiral shape",
            "A zigzag pattern", 
            "A heart shape",
            "A square trajectory",
            "A sine wave",
            "A star pattern",
            "An infinity symbol",
            "A triangle shape",
            "A random walk",
            "A sawtooth wave",
            "Multiple circles"
        ]
        
        text_prompts = text_prompts[:num_samples]
        
        # Save text prompts
        with open(os.path.join(output_dir, 'text_prompts.txt'), 'w') as f:
            for i, prompt in enumerate(text_prompts):
                f.write(f"{i}: {prompt}\n")
        
        print(f"📝 Generating {len(text_prompts)} trajectories...")
        
        # Generate trajectories
        trajectories = self.generate_batch_trajectories(text_prompts, num_inference_steps=100)
        
        # Save trajectories
        np.save(os.path.join(output_dir, 'generated_trajectories_text_conditioned.npy'), trajectories)
        
        # Create grid visualization
        print("🎨 Creating trajectory grid...")
        grid_save_path = os.path.join(output_dir, 'generated_trajectories_grid.png')
        utils.plot_text_conditioned_trajectories(
            trajectories, 
            text_prompts,
            title="Text-Conditioned Trajectory Generation Results",
            save_path=grid_save_path,
            max_plots=num_samples
        )
        
        # Create individual trajectory plots
        individual_dir = os.path.join(output_dir, 'individual_trajectories')
        os.makedirs(individual_dir, exist_ok=True)
        
        print("🖼️  Creating individual trajectory plots...")
        for i, (traj, prompt) in enumerate(zip(trajectories, text_prompts)):
            fig, ax = plt.subplots(figsize=(8, 8))
            ax.plot(traj[:, 0], traj[:, 1], 'b-', linewidth=3, alpha=0.8)
            ax.plot(traj[0, 0], traj[0, 1], 'go', markersize=10, label='Start')
            ax.plot(traj[-1, 0], traj[-1, 1], 'ro', markersize=10, label='End')
            ax.set_aspect('equal')
            ax.grid(True, alpha=0.3)
            ax.set_xlim(-1.2, 1.2)
            ax.set_ylim(-1.2, 1.2)
            ax.set_title(f'"{prompt}"', fontsize=14, wrap=True)
            ax.legend()
            
            save_path = os.path.join(individual_dir, f'trajectory_{i:03d}.png')
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            plt.close()
        
        # Create ground truth comparison with test data
        print("📊 Creating ground truth comparison...")
        self.create_ground_truth_comparison(
            output_dir, 
            num_samples=18,  # Always use 18 samples for 6x6 grid
            num_inference_steps=1000
        )
        
        # Create diffusion process animations
        print("🎬 Creating diffusion process animations...")
        animation_prompts = text_prompts[:3]  # First 3 for animations
        
        for i, prompt in enumerate(animation_prompts):
            animation_path = os.path.join(output_dir, f'diffusion_process_{i:03d}.gif')
            self.create_diffusion_animation(
                prompt, 
                animation_path, 
                fps=8, 
                num_inference_steps=1000
            )
        
        # Create comparison with different inference steps
        print("⚡ Testing different inference speeds...")
        test_prompt = "A perfect circle"
        step_counts = [10, 25, 50, 100, 250, 500, 1000]
        step_trajectories = []
        step_labels = []
        
        for steps in step_counts:
            traj = self.generate_batch_trajectories([test_prompt], num_inference_steps=steps)[0]
            step_trajectories.append(traj)
            step_labels.append(f"{steps} steps")
        
        # Plot speed comparison
        fig, axes = plt.subplots(1, len(step_counts), figsize=(28, 4))
        for i, (traj, label) in enumerate(zip(step_trajectories, step_labels)):
            axes[i].plot(traj[:, 0], traj[:, 1], 'b-', linewidth=2, alpha=0.8)
            axes[i].plot(traj[0, 0], traj[0, 1], 'go', markersize=8)
            axes[i].plot(traj[-1, 0], traj[-1, 1], 'ro', markersize=8)
            axes[i].set_aspect('equal')
            axes[i].grid(True, alpha=0.3)
            axes[i].set_xlim(-1.2, 1.2)
            axes[i].set_ylim(-1.2, 1.2)
            axes[i].set_title(f'{label}')
        
        plt.suptitle(f'Inference Speed Comparison: "{test_prompt}"', fontsize=16)
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'inference_speed_comparison.png'), dpi=150, bbox_inches='tight')
        plt.close()
        
        print(f"✅ Comprehensive inference completed!")
        print(f"📊 Results saved in: {output_dir}")
        print(f"📁 Generated files:")
        print(f"  • generated_trajectories_grid.png - Overview of all results")  
        print(f"  • individual_trajectories/ - Individual trajectory plots")
        print(f"  • ground_truth_vs_generated_comparison_1000steps.png - GT vs Generated comparison")
        print(f"  • simple_ground_truth_vs_generated_1000steps.png - Side-by-side comparison")
        print(f"  • diffusion_process_*.gif - Diffusion animations")
        print(f"  • inference_speed_comparison.png - Speed comparison")
        print(f"  • generated_trajectories_text_conditioned.npy - Raw trajectory data")
        print(f"  • ground_truth_comparison.json - Comparison data")


def main():
    """Main inference function."""
    parser = argparse.ArgumentParser(description='Text-Conditioned Trajectory Diffusion Inference')
    parser.add_argument('exp_folder', type=str, help='Experiment folder name (e.g., test_plotting_000)')
    parser.add_argument('--base-dir', type=str, default='output', help='Base directory containing experiments')
    parser.add_argument('--device', type=str, default='auto', help='Device to use (cuda/cpu/auto)')
    parser.add_argument('--num-samples', type=int, default=12, help='Number of samples to generate')
    parser.add_argument('--checkpoint', type=str, default='best_model.pth', help='Checkpoint to load')
    
    args = parser.parse_args()
    
    # Determine device
    if args.device == 'auto':
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
    else:
        device = args.device
    
    # Find experiment directory
    exp_dir = utils.find_experiment_dir(
        exp_idx=int(args.exp_folder.split('_')[-1]),
        exp_name='_'.join(args.exp_folder.split('_')[:-1]),
        base_dir=args.base_dir
    )
    
    print(f"🎯 Running inference on: {exp_dir}")
    print(f"📱 Using device: {device}")
    
    # Run inference
    inferencer = TrajectoryDiffusionInference(exp_dir, device=device)
    
    # Load specific checkpoint if requested
    if args.checkpoint != 'best_model.pth':
        inferencer.load_model(args.checkpoint)
    
    # Run comprehensive inference
    inferencer.run_comprehensive_inference(num_samples=args.num_samples)


if __name__ == "__main__":
    # If no command line args, run with default test case
    import sys
    
    if len(sys.argv) == 1:
        # Default test case
        exp_folder = "test_plotting_000"
        print(f"🧪 Running inference with default experiment: {exp_folder}")
        
        try:
            exp_dir = utils.find_experiment_dir(0, "test_plotting", "output")
            inferencer = TrajectoryDiffusionInference(exp_dir)
            inferencer.run_comprehensive_inference(num_samples=8)
        except FileNotFoundError as e:
            print(f"❌ {e}")
            print("💡 Please provide an experiment folder name:")
            print("   python inference.py <exp_folder>")
            print("   Example: python inference.py test_plotting_000")
    else:
        main() 