"""
Utility functions for trajectory diffusion project.
"""

import os
import json
import torch
import numpy as np
import matplotlib.pyplot as plt
from typing import Dict, List, Optional, Tuple
import glob


def create_experiment_dir(exp_name: str, base_dir: str = "output") -> str:
    """
    Create experiment directory with incremental index.
    
    Args:
        exp_name: Base experiment name
        base_dir: Base directory for experiments
    
    Returns:
        Path to experiment directory
    """
    os.makedirs(base_dir, exist_ok=True)
    
    # Find existing experiment directories
    existing_dirs = glob.glob(os.path.join(base_dir, f"{exp_name}_*"))
    
    # Extract indices
    indices = []
    for dir_path in existing_dirs:
        dirname = os.path.basename(dir_path)
        if dirname.startswith(f"{exp_name}_"):
            try:
                idx = int(dirname.split('_')[-1])
                indices.append(idx)
            except ValueError:
                continue
    
    # Determine next index
    next_idx = max(indices) + 1 if indices else 0
    
    # Create experiment directory
    exp_dir = os.path.join(base_dir, f"{exp_name}_{next_idx:03d}")
    os.makedirs(exp_dir, exist_ok=True)
    
    print(f"Created experiment directory: {exp_dir}")
    return exp_dir


def find_experiment_dir(exp_idx: int, exp_name: str = "trajectory_diffusion", base_dir: str = "output") -> str:
    """
    Find experiment directory by index.
    
    Args:
        exp_idx: Experiment index
        exp_name: Base experiment name
        base_dir: Base directory for experiments
    
    Returns:
        Path to experiment directory
    
    Raises:
        FileNotFoundError: If experiment directory doesn't exist
    """
    exp_dir = os.path.join(base_dir, f"{exp_name}_{exp_idx:03d}")
    
    if not os.path.exists(exp_dir):
        # List available experiments
        available_dirs = glob.glob(os.path.join(base_dir, f"{exp_name}_*"))
        available_indices = []
        for dir_path in available_dirs:
            dirname = os.path.basename(dir_path)
            try:
                idx = int(dirname.split('_')[-1])
                available_indices.append(idx)
            except ValueError:
                continue
        
        raise FileNotFoundError(
            f"Experiment directory not found: {exp_dir}\n"
            f"Available experiments: {sorted(available_indices)}"
        )
    
    return exp_dir


def load_config(exp_dir: str) -> Dict:
    """Load experiment configuration."""
    config_path = os.path.join(exp_dir, 'config.json')
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Config file not found: {config_path}")
    
    with open(config_path, 'r') as f:
        config = json.load(f)
    
    return config


def save_results(results: Dict, exp_dir: str, filename: str = "results.json"):
    """Save results to experiment directory."""
    filepath = os.path.join(exp_dir, filename)
    with open(filepath, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"Results saved: {filepath}")


def plot_trajectories(
    trajectories: np.ndarray,
    labels: Optional[List[str]] = None,
    title: str = "Trajectories",
    save_path: Optional[str] = None,
    figsize: Tuple[int, int] = (12, 8),
    alpha: float = 0.7,
    max_plots: int = 25
) -> None:
    """
    Plot multiple trajectories.
    
    Args:
        trajectories: Array of trajectories [N, seq_len, 2]
        labels: Optional labels for trajectories
        title: Plot title
        save_path: Path to save plot
        figsize: Figure size
        alpha: Line alpha
        max_plots: Maximum number of trajectories to plot
    """
    n_trajs = min(len(trajectories), max_plots)
    
    # Determine subplot layout
    if n_trajs <= 4:
        rows, cols = 2, 2
    elif n_trajs <= 9:
        rows, cols = 3, 3
    elif n_trajs <= 16:
        rows, cols = 4, 4
    else:
        rows, cols = 5, 5
    
    fig, axes = plt.subplots(rows, cols, figsize=figsize)
    if rows * cols == 1:
        axes = [axes]
    else:
        axes = axes.flatten()
    
    for i in range(min(n_trajs, len(axes))):
        traj = trajectories[i]
        ax = axes[i]
        
        ax.plot(traj[:, 0], traj[:, 1], alpha=alpha, linewidth=1.5)
        ax.set_aspect('equal')
        ax.grid(True, alpha=0.3)
        ax.set_xlim(-1.1, 1.1)
        ax.set_ylim(-1.1, 1.1)
        
        if labels is not None and i < len(labels):
            ax.set_title(f"Traj {i}: {labels[i]}")
        else:
            ax.set_title(f"Trajectory {i}")
    
    # Remove empty subplots
    for i in range(n_trajs, len(axes)):
        fig.delaxes(axes[i])
    
    plt.suptitle(title, fontsize=16)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Plot saved: {save_path}")
    
    plt.show()


def compare_trajectories(
    real_trajectories: np.ndarray,
    generated_trajectories: np.ndarray,
    save_path: Optional[str] = None,
    n_samples: int = 5
) -> None:
    """
    Compare real and generated trajectories side by side.
    
    Args:
        real_trajectories: Real trajectory samples [N, seq_len, 2]
        generated_trajectories: Generated trajectory samples [N, seq_len, 2]
        save_path: Path to save comparison plot
        n_samples: Number of samples to compare
    """
    fig, axes = plt.subplots(2, n_samples, figsize=(3*n_samples, 6))
    
    for i in range(n_samples):
        # Real trajectory
        if i < len(real_trajectories):
            real_traj = real_trajectories[i]
            axes[0, i].plot(real_traj[:, 0], real_traj[:, 1], 'b-', alpha=0.8, linewidth=2)
            axes[0, i].set_title(f'Real {i}')
        
        # Generated trajectory
        if i < len(generated_trajectories):
            gen_traj = generated_trajectories[i]
            axes[1, i].plot(gen_traj[:, 0], gen_traj[:, 1], 'r-', alpha=0.8, linewidth=2)
            axes[1, i].set_title(f'Generated {i}')
        
        # Set common properties
        for row in range(2):
            axes[row, i].set_aspect('equal')
            axes[row, i].grid(True, alpha=0.3)
            axes[row, i].set_xlim(-1.1, 1.1)
            axes[row, i].set_ylim(-1.1, 1.1)
    
    plt.suptitle('Real vs Generated Trajectories', fontsize=16)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Comparison plot saved: {save_path}")
    
    plt.show()


def compute_trajectory_metrics(
    real_trajectories: np.ndarray,
    generated_trajectories: np.ndarray
) -> Dict[str, float]:
    """
    Compute metrics comparing real and generated trajectories.
    
    Args:
        real_trajectories: Real trajectories [N, seq_len, 2]
        generated_trajectories: Generated trajectories [N, seq_len, 2]
    
    Returns:
        Dictionary of computed metrics
    """
    # Handle different sequence lengths by cropping to minimum
    min_len = min(real_trajectories.shape[1], generated_trajectories.shape[1])
    real_trajectories = real_trajectories[:, :min_len, :]
    generated_trajectories = generated_trajectories[:, :min_len, :]
    
    metrics = {}
    
    # Mean squared error
    mse = np.mean((real_trajectories - generated_trajectories) ** 2)
    metrics['mse'] = float(mse)
    
    # Mean absolute error
    mae = np.mean(np.abs(real_trajectories - generated_trajectories))
    metrics['mae'] = float(mae)
    
    # Trajectory length comparison
    real_lengths = np.sqrt(np.sum(np.diff(real_trajectories, axis=1) ** 2, axis=2)).sum(axis=1)
    gen_lengths = np.sqrt(np.sum(np.diff(generated_trajectories, axis=1) ** 2, axis=2)).sum(axis=1)
    
    metrics['real_length_mean'] = float(np.mean(real_lengths))
    metrics['real_length_std'] = float(np.std(real_lengths))
    metrics['gen_length_mean'] = float(np.mean(gen_lengths))
    metrics['gen_length_std'] = float(np.std(gen_lengths))
    
    # Endpoint distance
    endpoint_dist = np.linalg.norm(
        real_trajectories[:, -1, :] - generated_trajectories[:, -1, :], axis=1
    )
    metrics['endpoint_distance_mean'] = float(np.mean(endpoint_dist))
    metrics['endpoint_distance_std'] = float(np.std(endpoint_dist))
    
    # Frechet distance (simplified version)
    frechet_distances = []
    for i in range(min(len(real_trajectories), len(generated_trajectories))):
        real_traj = real_trajectories[i]
        gen_traj = generated_trajectories[i]
        
        # Simple approximation: average point-wise distance
        pointwise_dist = np.linalg.norm(real_traj - gen_traj, axis=1)
        frechet_distances.append(np.mean(pointwise_dist))
    
    metrics['frechet_distance_mean'] = float(np.mean(frechet_distances))
    metrics['frechet_distance_std'] = float(np.std(frechet_distances))
    
    return metrics


def plot_text_conditioned_trajectories(
    trajectories: np.ndarray,
    text_descriptions: List[str],
    title: str = "Text-Conditioned Trajectories",
    save_path: Optional[str] = None,
    figsize: Tuple[int, int] = (15, 10),
    max_plots: int = 12
) -> None:
    """
    Plot trajectories with their corresponding text descriptions.
    
    Args:
        trajectories: Array of trajectories [N, seq_len, 2]
        text_descriptions: List of text descriptions
        title: Plot title
        save_path: Path to save plot
        figsize: Figure size
        max_plots: Maximum number of trajectories to plot
    """
    n_trajs = min(len(trajectories), len(text_descriptions), max_plots)
    
    # Determine subplot layout
    if n_trajs <= 4:
        rows, cols = 2, 2
    elif n_trajs <= 6:
        rows, cols = 2, 3
    elif n_trajs <= 9:
        rows, cols = 3, 3
    else:
        rows, cols = 3, 4
    
    fig, axes = plt.subplots(rows, cols, figsize=figsize)
    if rows * cols == 1:
        axes = [axes]
    else:
        axes = axes.flatten()
    
    colors = plt.cm.tab10(np.linspace(0, 1, 10))
    
    for i in range(min(n_trajs, len(axes))):
        traj = trajectories[i]
        ax = axes[i]
        
        # Plot trajectory with gradient color
        ax.plot(traj[:, 0], traj[:, 1], color=colors[i % 10], alpha=0.8, linewidth=2)
        
        # Mark start and end points
        ax.plot(traj[0, 0], traj[0, 1], 'go', markersize=8, label='Start')
        ax.plot(traj[-1, 0], traj[-1, 1], 'ro', markersize=8, label='End')
        
        ax.set_aspect('equal')
        ax.grid(True, alpha=0.3)
        ax.set_xlim(-1.1, 1.1)
        ax.set_ylim(-1.1, 1.1)
        
        # Wrap text description for better display
        text = text_descriptions[i][:50] + "..." if len(text_descriptions[i]) > 50 else text_descriptions[i]
        ax.set_title(f"{text}", fontsize=10, wrap=True)
        
        if i == 0:  # Only show legend on first plot
            ax.legend(fontsize=8)
    
    # Remove empty subplots
    for i in range(n_trajs, len(axes)):
        fig.delaxes(axes[i])
    
    plt.suptitle(title, fontsize=16)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Text-conditioned plot saved: {save_path}")
        plt.close()  # Close to save memory
    else:
        plt.show()


def plot_training_progress(
    real_trajectories: np.ndarray,
    generated_trajectories: np.ndarray,
    text_descriptions: List[str],
    epoch: int,
    save_path: Optional[str] = None,
    n_samples: int = 6
) -> None:
    """
    Plot training progress showing real vs generated trajectories with text.
    
    Args:
        real_trajectories: Real trajectory samples [N, seq_len, 2]
        generated_trajectories: Generated trajectory samples [N, seq_len, 2]
        text_descriptions: Text descriptions for trajectories
        epoch: Current training epoch
        save_path: Path to save plot
        n_samples: Number of samples to show
    """
    fig, axes = plt.subplots(2, n_samples, figsize=(4*n_samples, 8))
    
    colors = ['blue', 'red']
    labels = ['Real', 'Generated']
    
    for i in range(n_samples):
        # Real trajectory
        if i < len(real_trajectories):
            real_traj = real_trajectories[i]
            axes[0, i].plot(real_traj[:, 0], real_traj[:, 1], 
                           color=colors[0], alpha=0.8, linewidth=2, label=labels[0])
            axes[0, i].plot(real_traj[0, 0], real_traj[0, 1], 'go', markersize=6)
            axes[0, i].plot(real_traj[-1, 0], real_traj[-1, 1], 'ro', markersize=6)
        
        # Generated trajectory
        if i < len(generated_trajectories):
            gen_traj = generated_trajectories[i]
            axes[1, i].plot(gen_traj[:, 0], gen_traj[:, 1], 
                           color=colors[1], alpha=0.8, linewidth=2, label=labels[1])
            axes[1, i].plot(gen_traj[0, 0], gen_traj[0, 1], 'go', markersize=6)
            axes[1, i].plot(gen_traj[-1, 0], gen_traj[-1, 1], 'ro', markersize=6)
        
        # Set properties for both rows
        for row in range(2):
            axes[row, i].set_aspect('equal')
            axes[row, i].grid(True, alpha=0.3)
            axes[row, i].set_xlim(-1.1, 1.1)
            axes[row, i].set_ylim(-1.1, 1.1)
            
            if i < len(text_descriptions):
                text = text_descriptions[i][:40] + "..." if len(text_descriptions[i]) > 40 else text_descriptions[i]
                axes[row, i].set_title(f"{labels[row]}: {text}", fontsize=9)
    
    plt.suptitle(f'Training Progress - Epoch {epoch}', fontsize=16)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Training progress plot saved: {save_path}")
    
    plt.close()  # Close to save memory during training


def create_animation(trajectory: np.ndarray, save_path: str, fps: int = 10) -> None:
    """
    Create animated visualization of trajectory.
    
    Args:
        trajectory: Single trajectory [seq_len, 2]
        save_path: Path to save animation
        fps: Frames per second
    """
    try:
        import matplotlib.animation as animation
    except ImportError:
        print("matplotlib.animation not available, skipping animation")
        return
    
    fig, ax = plt.subplots(figsize=(8, 8))
    ax.set_xlim(-1.1, 1.1)
    ax.set_ylim(-1.1, 1.1)
    ax.set_aspect('equal')
    ax.grid(True, alpha=0.3)
    ax.set_title('Trajectory Animation')
    
    line, = ax.plot([], [], 'b-', linewidth=2, alpha=0.8)
    point, = ax.plot([], [], 'ro', markersize=8)
    
    def animate(frame):
        if frame < len(trajectory):
            # Draw trajectory up to current frame
            line.set_data(trajectory[:frame+1, 0], trajectory[:frame+1, 1])
            # Draw current point
            point.set_data([trajectory[frame, 0]], [trajectory[frame, 1]])
        return line, point
    
    anim = animation.FuncAnimation(
        fig, animate, frames=len(trajectory),
        interval=1000//fps, blit=True, repeat=True
    )
    
    anim.save(save_path, writer='pillow', fps=fps)
    plt.close()
    print(f"Animation saved: {save_path}")


def print_model_summary(model: torch.nn.Module) -> None:
    """Print model summary."""
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    print("=" * 50)
    print("MODEL SUMMARY")
    print("=" * 50)
    print(f"Total parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")
    print(f"Model size: {total_params * 4 / 1024 / 1024:.2f} MB (float32)")
    print("=" * 50)


def seed_everything(seed: int = 42) -> None:
    """Set random seeds for reproducibility."""
    import random
    
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    
    print(f"Random seed set to {seed}")


if __name__ == "__main__":
    # Test utility functions
    print("Testing utility functions...")
    
    # Test experiment directory creation
    exp_dir = create_experiment_dir("test_exp", "test_output")
    print(f"Created: {exp_dir}")
    
    # Test trajectory plotting
    test_trajectories = np.random.randn(5, 100, 2)
    plot_trajectories(test_trajectories, title="Test Trajectories")
    
    # Clean up test directory
    import shutil
    if os.path.exists("test_output"):
        shutil.rmtree("test_output")
    
    print("Utility functions test completed!") 