"""
Visualize Forward and Backward Diffusion Process
Shows how noise is added (forward) and removed (backward) during diffusion.
"""

import torch
import numpy as np
import matplotlib.pyplot as plt
import os
import argparse
from typing import List

from model import create_model
from dataset import TrajectoryDataModule
from diffusers import DDPMScheduler
import utils


def visualize_diffusion_process(exp_idx: int, save_path: str = None):
    """
    Visualize both forward (noising) and backward (denoising) diffusion processes.
    
    Args:
        exp_idx: Experiment index to load model from
        save_path: Path to save the visualization
    """
    # Find experiment directory
    exp_dir = utils.find_experiment_dir(exp_idx, "trajectory_diffusion")
    
    # Load configuration
    config = utils.load_config(exp_dir)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    print(f"Loading model from experiment {exp_idx}...")
    
    # Load data module to get a sample trajectory
    data_module = TrajectoryDataModule(
        data_dir=config['data_params'].get('data_dir', 'data'),
        batch_size=1,
        sequence_length=config['data_params']['sequence_length'],
    )
    
    # Create model
    model = create_model(
        sequence_length=config['data_params']['sequence_length'],
        in_channels=config['model_params']['in_channels'],
        time_emb_dim=config['model_params']['time_emb_dim']
    ).to(device)
    
    # Create scheduler
    scheduler = DDPMScheduler(
        num_train_timesteps=config['scheduler_params']['num_train_timesteps'],
        beta_start=config['scheduler_params']['beta_start'],
        beta_end=config['scheduler_params']['beta_end'],
        beta_schedule=config['scheduler_params']['beta_schedule']
    )
    
    # Load trained model
    model_path = os.path.join(exp_dir, 'best_model.pth')
    if not os.path.exists(model_path):
        model_path = os.path.join(exp_dir, 'final_model.pth')
    
    checkpoint = torch.load(model_path, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    
    print("Model loaded successfully!")
    
    # Get a sample trajectory
    sample_batch = data_module.get_sample_batch('test')
    original_trajectory = sample_batch['trajectory'][0].to(device)  # [seq_len, 2]
    trajectory_2d = original_trajectory.transpose(0, 1).unsqueeze(0)  # [1, 2, seq_len]
    
    print(f"Sample trajectory shape: {trajectory_2d.shape}")
    print(f"Pattern type: {sample_batch['label_name'][0]}")
    
    # Select timesteps to visualize
    num_steps = 8
    max_timesteps = scheduler.config.num_train_timesteps
    timesteps = np.linspace(0, max_timesteps - 1, num_steps).astype(int)
    
    print(f"Visualizing timesteps: {timesteps}")
    
    # Forward process: Add noise progressively
    print("Running forward process (adding noise)...")
    forward_trajectories = []
    
    for t in timesteps:
        t_tensor = torch.tensor([t], device=device)
        noise = torch.randn_like(trajectory_2d)
        noisy_trajectory = scheduler.add_noise(trajectory_2d, noise, t_tensor)
        forward_trajectories.append(noisy_trajectory.cpu().numpy()[0].transpose(1, 0))  # [seq_len, 2]
    
    # Backward process: Start from pure noise and denoise
    print("Running backward process (denoising)...")
    backward_trajectories = []
    
    # Start with pure noise
    current_trajectory = torch.randn_like(trajectory_2d)
    
    # Store trajectories at selected timesteps during denoising
    timestep_indices = set(max_timesteps - 1 - timesteps)  # Reverse timesteps for denoising
    step_counter = 0
    
    with torch.no_grad():
        for t in reversed(range(max_timesteps)):
            t_tensor = torch.tensor([t], device=device)
            
            # Predict noise
            noise_pred = model(current_trajectory, t_tensor)
            
            # Handle potential size mismatch
            if noise_pred.shape != current_trajectory.shape:
                seq_len_pred = noise_pred.shape[-1]
                current_trajectory = current_trajectory[..., :seq_len_pred]
            
            # Remove noise
            current_trajectory = scheduler.step(noise_pred, t, current_trajectory).prev_sample
            
            # Store trajectory if it's one of our selected timesteps
            if step_counter in timestep_indices:
                backward_trajectories.append(current_trajectory.cpu().numpy()[0].transpose(1, 0))
            
            step_counter += 1
    
    # Reverse backward trajectories to match forward order
    backward_trajectories = backward_trajectories[::-1]
    
    # Create visualization
    print("Creating visualization...")
    fig, axes = plt.subplots(2, num_steps, figsize=(20, 8))
    
    # Plot forward process (top row)
    for i, (t, traj) in enumerate(zip(timesteps, forward_trajectories)):
        ax = axes[0, i]
        ax.plot(traj[:, 0], traj[:, 1], 'b-', linewidth=2, alpha=0.8)
        ax.set_xlim(-2, 2)
        ax.set_ylim(-2, 2)
        ax.set_aspect('equal')
        ax.grid(True, alpha=0.3)
        ax.set_title(f't = {t}')
        
        if i == 0:
            ax.set_ylabel('Forward Process\n(Adding Noise)', fontsize=12, fontweight='bold')
    
    # Plot backward process (bottom row)
    for i, (t, traj) in enumerate(zip(timesteps, backward_trajectories)):
        ax = axes[1, i]
        ax.plot(traj[:, 0], traj[:, 1], 'r-', linewidth=2, alpha=0.8)
        ax.set_xlim(-2, 2)
        ax.set_ylim(-2, 2)
        ax.set_aspect('equal')
        ax.grid(True, alpha=0.3)
        ax.set_title(f't = {max_timesteps - 1 - t}')
        
        if i == 0:
            ax.set_ylabel('Backward Process\n(Removing Noise)', fontsize=12, fontweight='bold')
    
    # Add overall title and labels
    plt.suptitle(f'Diffusion Process Visualization - {sample_batch["label_name"][0].capitalize()} Pattern', 
                 fontsize=16, fontweight='bold')
    
    # Add timestep explanation
    fig.text(0.5, 0.02, 'Timestep →', ha='center', fontsize=12)
    fig.text(0.02, 0.75, 'Clean', va='center', rotation=90, fontsize=10, color='blue')
    fig.text(0.02, 0.65, '↓', va='center', fontsize=12, color='blue')
    fig.text(0.02, 0.55, 'Noisy', va='center', rotation=90, fontsize=10, color='blue')
    
    fig.text(0.02, 0.45, 'Noisy', va='center', rotation=90, fontsize=10, color='red')
    fig.text(0.02, 0.35, '↓', va='center', fontsize=12, color='red')
    fig.text(0.02, 0.25, 'Clean', va='center', rotation=90, fontsize=10, color='red')
    
    plt.tight_layout()
    plt.subplots_adjust(top=0.9, bottom=0.1, left=0.05)
    
    # Save plot
    if save_path is None:
        save_path = os.path.join(exp_dir, 'diffusion_process_visualization.png')
    
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"Visualization saved: {save_path}")
    
    # Also create a timeline plot showing noise level over time
    create_noise_timeline(timesteps, forward_trajectories, backward_trajectories, 
                         original_trajectory.cpu().numpy(), exp_dir)
    
    plt.close()


def create_noise_timeline(timesteps, forward_trajectories, backward_trajectories, 
                         original_trajectory, exp_dir):
    """Create a timeline plot showing noise levels over time."""
    
    # Calculate noise levels (distance from original)
    forward_noise_levels = []
    backward_noise_levels = []
    
    for traj in forward_trajectories:
        noise_level = np.mean(np.linalg.norm(traj - original_trajectory, axis=1))
        forward_noise_levels.append(noise_level)
    
    for traj in backward_trajectories:
        # Crop to match original length
        min_len = min(len(traj), len(original_trajectory))
        noise_level = np.mean(np.linalg.norm(traj[:min_len] - original_trajectory[:min_len], axis=1))
        backward_noise_levels.append(noise_level)
    
    # Create timeline plot
    plt.figure(figsize=(12, 6))
    
    plt.subplot(1, 2, 1)
    plt.plot(timesteps, forward_noise_levels, 'bo-', linewidth=2, markersize=8, label='Forward Process')
    plt.xlabel('Timestep')
    plt.ylabel('Noise Level (Distance from Original)')
    plt.title('Forward Process: Adding Noise')
    plt.grid(True, alpha=0.3)
    plt.legend()
    
    plt.subplot(1, 2, 2)
    reverse_timesteps = len(timesteps) - 1 - np.arange(len(timesteps))
    plt.plot(reverse_timesteps, backward_noise_levels, 'ro-', linewidth=2, markersize=8, label='Backward Process')
    plt.xlabel('Denoising Steps')
    plt.ylabel('Noise Level (Distance from Clean)')
    plt.title('Backward Process: Removing Noise')
    plt.grid(True, alpha=0.3)
    plt.legend()
    
    plt.tight_layout()
    plt.savefig(os.path.join(exp_dir, 'noise_timeline.png'), dpi=300, bbox_inches='tight')
    print(f"Noise timeline saved: {os.path.join(exp_dir, 'noise_timeline.png')}")
    plt.close()


def visualize_multi_pattern_diffusion(exp_idx: int, save_path: str = None):
    """
    Visualize diffusion process for multiple pattern types.
    
    Args:
        exp_idx: Experiment index to load model from
        save_path: Path to save the visualization
    """
    # Find experiment directory
    exp_dir = utils.find_experiment_dir(exp_idx, "trajectory_diffusion")
    
    # Load configuration
    config = utils.load_config(exp_dir)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    print(f"Loading model from experiment {exp_idx}...")
    
    # Load data module
    data_module = TrajectoryDataModule(
        data_dir=config['data_params'].get('data_dir', 'data'),
        batch_size=32,
        sequence_length=config['data_params']['sequence_length'],
    )
    
    # Create model
    model = create_model(
        sequence_length=config['data_params']['sequence_length'],
        in_channels=config['model_params']['in_channels'],
        time_emb_dim=config['model_params']['time_emb_dim']
    ).to(device)
    
    # Create scheduler
    scheduler = DDPMScheduler(
        num_train_timesteps=config['scheduler_params']['num_train_timesteps'],
        beta_start=config['scheduler_params']['beta_start'],
        beta_end=config['scheduler_params']['beta_end'],
        beta_schedule=config['scheduler_params']['beta_schedule']
    )
    
    # Load trained model
    model_path = os.path.join(exp_dir, 'best_model.pth')
    if not os.path.exists(model_path):
        model_path = os.path.join(exp_dir, 'final_model.pth')
    
    checkpoint = torch.load(model_path, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    
    print("Model loaded successfully!")
    
    # Get samples of different pattern types
    pattern_samples = {}
    pattern_types = ['sine', 'spiral', 'circle', 'lemniscate', 'cardioid']
    
    # Collect samples for each pattern type
    for batch in data_module.test_dataloader():
        for i, label_name in enumerate(batch['label_name']):
            if label_name in pattern_types and label_name not in pattern_samples:
                trajectory = batch['trajectory'][i].to(device)  # [seq_len, 2]
                trajectory_2d = trajectory.transpose(0, 1).unsqueeze(0)  # [1, 2, seq_len]
                pattern_samples[label_name] = trajectory_2d
        
        if len(pattern_samples) >= len(pattern_types):
            break
    
    # Select key timesteps
    max_timesteps = scheduler.config.num_train_timesteps
    key_timesteps = [0, max_timesteps // 4, max_timesteps // 2, 3 * max_timesteps // 4, max_timesteps - 1]
    
    print(f"Visualizing timesteps: {key_timesteps}")
    
    # Create visualization
    fig, axes = plt.subplots(len(pattern_types), len(key_timesteps), figsize=(15, 12))
    
    for row, (pattern_type, trajectory_2d) in enumerate(pattern_samples.items()):
        print(f"Processing {pattern_type} pattern...")
        
        for col, t in enumerate(key_timesteps):
            ax = axes[row, col]
            
            # Add noise at timestep t
            t_tensor = torch.tensor([t], device=device)
            noise = torch.randn_like(trajectory_2d)
            noisy_trajectory = scheduler.add_noise(trajectory_2d, noise, t_tensor)
            
            # Convert to numpy and plot
            traj_np = noisy_trajectory.cpu().numpy()[0].transpose(1, 0)  # [seq_len, 2]
            ax.plot(traj_np[:, 0], traj_np[:, 1], linewidth=2, alpha=0.8)
            ax.set_xlim(-2, 2)
            ax.set_ylim(-2, 2)
            ax.set_aspect('equal')
            ax.grid(True, alpha=0.3)
            
            # Labels
            if row == 0:
                ax.set_title(f't = {t}', fontsize=12)
            if col == 0:
                ax.set_ylabel(f'{pattern_type.capitalize()}', fontsize=12, fontweight='bold')
    
    plt.suptitle('Forward Diffusion Process - Multiple Patterns', fontsize=16, fontweight='bold')
    plt.tight_layout()
    
    # Save plot
    if save_path is None:
        save_path = os.path.join(exp_dir, 'multi_pattern_diffusion.png')
    
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"Multi-pattern visualization saved: {save_path}")
    plt.close()


def main():
    """Main function."""
    parser = argparse.ArgumentParser(description='Visualize diffusion process')
    parser.add_argument('--exp_idx', type=int, required=True, help='Experiment index')
    parser.add_argument('--save_path', type=str, default=None, help='Path to save visualization')
    parser.add_argument('--multi_pattern', action='store_true', help='Create multi-pattern visualization')
    
    args = parser.parse_args()
    
    if args.multi_pattern:
        visualize_multi_pattern_diffusion(args.exp_idx, args.save_path)
    else:
        visualize_diffusion_process(args.exp_idx, args.save_path)


if __name__ == "__main__":
    main() 