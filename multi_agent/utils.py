"""
Utility functions for trajectory diffusion project.
"""

import os
import json
import torch
import numpy as np
import matplotlib.pyplot as plt
from typing import Dict, List, Optional, Tuple, Any
import glob
from datetime import datetime
import time
import shutil


def get_timestamp() -> str:
    """Get current timestamp for unique experiment naming."""
    return datetime.now().strftime("%Y%m%d_%H%M%S")


def create_experiment_dir(exp_name: str, base_dir: str = "output") -> str:
    """
    Create unique experiment directory.
    
    Args:
        exp_name: Base experiment name
        base_dir: Base directory for experiments
        
    Returns:
        Path to created experiment directory
    """
    timestamp = get_timestamp()
    exp_dir = os.path.join(base_dir, f"{exp_name}_{timestamp}")
    os.makedirs(exp_dir, exist_ok=True)
    
    # Create subdirectories
    for subdir in ['plots', 'checkpoints', 'logs']:
        os.makedirs(os.path.join(exp_dir, subdir), exist_ok=True)
    
    return exp_dir


def save_experiment_metadata(
    exp_dir: str,
    config: Dict[str, Any],
    metrics: Optional[Dict[str, Any]] = None
) -> None:
    """
    Save experiment metadata and configuration.
    
    Args:
        exp_dir: Experiment directory path
        config: Configuration dictionary
        metrics: Optional metrics dictionary
    """
    metadata = {
        'timestamp': get_timestamp(),
        'config': config,
        'metrics': metrics or {}
    }
    
    with open(os.path.join(exp_dir, 'metadata.json'), 'w') as f:
        json.dump(metadata, f, indent=2)


def backup_code(exp_dir: str, src_files: List[str]) -> None:
    """
    Backup source code files to experiment directory.
    
    Args:
        exp_dir: Experiment directory path
        src_files: List of source files to backup
    """
    code_backup_dir = os.path.join(exp_dir, 'code_backup')
    os.makedirs(code_backup_dir, exist_ok=True)
    
    for src_file in src_files:
        if os.path.exists(src_file):
            dst_file = os.path.join(code_backup_dir, os.path.basename(src_file))
            shutil.copy2(src_file, dst_file)


def time_function(func):
    """Decorator to time function execution."""
    def wrapper(*args, **kwargs):
        start_time = time.time()
        result = func(*args, **kwargs)
        end_time = time.time()
        print(f"{func.__name__} took {end_time - start_time:.2f} seconds")
        return result
    return wrapper


def print_model_summary(model, input_shape: Tuple[int, ...]) -> None:
    """
    Print model architecture summary.
    
    Args:
        model: PyTorch model
        input_shape: Input tensor shape (without batch dimension)
    """
    from torchsummary import summary
    try:
        summary(model, input_shape)
    except ImportError:
        print("torchsummary not available. Install with: pip install torchsummary")
        print(f"Model has {sum(p.numel() for p in model.parameters()):,} parameters")


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


def plot_multi_agent_trajectories(
    trajectories: np.ndarray,
    agent_masks: np.ndarray,
    pattern_types: Optional[List[List[str]]] = None,
    region_names: Optional[List[List[str]]] = None,
    text_descriptions: Optional[List[str]] = None,
    title: str = "Multi-Agent Trajectories",
    save_path: Optional[str] = None,
    figsize: Tuple[int, int] = (15, 10),
    max_plots: int = 12
) -> None:
    """
    Plot multi-agent trajectories with different colors for each agent.
    
    Args:
        trajectories: Array of multi-agent trajectories [N, max_agents, seq_len, 2]
        agent_masks: Boolean masks for active agents [N, max_agents]
        pattern_types: Optional pattern types for each agent
        region_names: Optional region names for each agent
        text_descriptions: Optional text descriptions
        title: Plot title
        save_path: Path to save plot
        figsize: Figure size
        max_plots: Maximum number of samples to plot
    """
    n_samples = min(len(trajectories), max_plots)
    
    # Determine subplot layout
    if n_samples <= 4:
        rows, cols = 2, 2
    elif n_samples <= 6:
        rows, cols = 2, 3
    elif n_samples <= 9:
        rows, cols = 3, 3
    else:
        rows, cols = 3, 4
    
    fig, axes = plt.subplots(rows, cols, figsize=figsize)
    if rows * cols == 1:
        axes = [axes]
    else:
        axes = axes.flatten()
    
    colors = plt.cm.tab10(np.linspace(0, 1, 10))
    
    for i in range(min(n_samples, len(axes))):
        ax = axes[i]
        sample_trajs = trajectories[i]
        sample_mask = agent_masks[i]
        
        # Plot each active agent's trajectory
        agent_count = 0
        for j, (traj, is_active) in enumerate(zip(sample_trajs, sample_mask)):
            if is_active:
                color = colors[j % len(colors)]
                ax.plot(traj[:, 0], traj[:, 1], color=color, linewidth=2, alpha=0.8, 
                       label=f'Agent {j+1}')
                
                # Mark start and end points
                ax.plot(traj[0, 0], traj[0, 1], 's', color=color, markersize=8, alpha=0.9)
                ax.plot(traj[-1, 0], traj[-1, 1], 'o', color=color, markersize=8, alpha=0.9)
                agent_count += 1
        
        ax.set_aspect('equal')
        ax.grid(True, alpha=0.3)
        ax.set_xlim(-1.1, 1.1)
        ax.set_ylim(-1.1, 1.1)
        
        # Create title
        if text_descriptions is not None and i < len(text_descriptions):
            text = text_descriptions[i][:60] + "..." if len(text_descriptions[i]) > 60 else text_descriptions[i]
            ax.set_title(f'Sample {i+1} ({agent_count} agents)\n"{text}"', fontsize=10)
        else:
            ax.set_title(f'Sample {i+1} ({agent_count} agents)', fontsize=12)
        
        if agent_count <= 3:  # Only show legend for simpler cases
            ax.legend(fontsize=8, loc='upper right')
    
    # Remove empty subplots
    for i in range(n_samples, len(axes)):
        fig.delaxes(axes[i])
    
    plt.suptitle(title, fontsize=16)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Multi-agent plot saved: {save_path}")
        plt.close()  # Close to save memory
    else:
        plt.show()


def plot_multi_agent_training_progress(
    real_trajectories: np.ndarray,
    generated_trajectories: np.ndarray,
    agent_masks: np.ndarray,
    text_descriptions: List[str],
    epoch: int,
    save_path: Optional[str] = None,
    n_samples: int = 6
) -> None:
    """
    Plot multi-agent training progress showing real vs generated trajectories with text.
    
    Args:
        real_trajectories: Real multi-agent trajectory samples [N, max_agents, seq_len, 2]
        generated_trajectories: Generated multi-agent trajectory samples [N, max_agents, seq_len, 2]
        agent_masks: Boolean masks for active agents [N, max_agents]
        text_descriptions: Text descriptions for trajectories
        epoch: Current training epoch
        save_path: Path to save plot
        n_samples: Number of samples to show
    """
    fig, axes = plt.subplots(2, n_samples, figsize=(4*n_samples, 8))
    
    colors = plt.cm.tab10(np.linspace(0, 1, 10))
    labels = ['Real', 'Generated']
    
    for i in range(n_samples):
        # Real trajectories
        if i < len(real_trajectories):
            real_sample = real_trajectories[i]
            real_mask = agent_masks[i]
            
            for j, (traj, is_active) in enumerate(zip(real_sample, real_mask)):
                if is_active:
                    color = colors[j % len(colors)]
                    axes[0, i].plot(traj[:, 0], traj[:, 1], color=color, alpha=0.8, 
                                  linewidth=2, label=f'Agent {j+1}' if i == 0 else "")
                    axes[0, i].plot(traj[0, 0], traj[0, 1], 's', color=color, markersize=6)
                    axes[0, i].plot(traj[-1, 0], traj[-1, 1], 'o', color=color, markersize=6)
        
        # Generated trajectories
        if i < len(generated_trajectories):
            gen_sample = generated_trajectories[i]
            gen_mask = agent_masks[i]  # Use same mask as real
            
            for j, (traj, is_active) in enumerate(zip(gen_sample, gen_mask)):
                if is_active:
                    color = colors[j % len(colors)]
                    axes[1, i].plot(traj[:, 0], traj[:, 1], color=color, alpha=0.8, 
                                  linewidth=2, linestyle='--')
                    axes[1, i].plot(traj[0, 0], traj[0, 1], 's', color=color, markersize=6)
                    axes[1, i].plot(traj[-1, 0], traj[-1, 1], 'o', color=color, markersize=6)
        
        # Set properties for both rows
        for row in range(2):
            axes[row, i].set_aspect('equal')
            axes[row, i].grid(True, alpha=0.3)
            axes[row, i].set_xlim(-1.1, 1.1)
            axes[row, i].set_ylim(-1.1, 1.1)
            
            if i < len(text_descriptions):
                text = text_descriptions[i][:40] + "..." if len(text_descriptions[i]) > 40 else text_descriptions[i]
                axes[row, i].set_title(f"{labels[row]}: {text}", fontsize=9)
    
    plt.suptitle(f'Multi-Agent Training Progress - Epoch {epoch}', fontsize=16)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Multi-agent training progress plot saved: {save_path}")
    
    plt.close()  # Close to save memory during training


def plot_multi_agent_final_results(
    real_trajectories: np.ndarray,
    generated_trajectories: np.ndarray,
    agent_masks: np.ndarray,
    text_descriptions: List[str],
    title: str = "Final Multi-Agent Results",
    save_path: Optional[str] = None,
    figsize: Tuple[int, int] = (16, 12),
    max_plots: int = 12
) -> None:
    """
    Plot comprehensive final results for multi-agent trajectories.
    
    Args:
        real_trajectories: Real multi-agent trajectories [N, max_agents, seq_len, 2]
        generated_trajectories: Generated multi-agent trajectories [N, max_agents, seq_len, 2]
        agent_masks: Boolean masks for active agents [N, max_agents]
        text_descriptions: Text descriptions
        title: Plot title
        save_path: Path to save plot
        figsize: Figure size
        max_plots: Maximum number of samples to plot
    """
    n_samples = min(len(real_trajectories), len(generated_trajectories), max_plots)
    
    # Create subplot layout - pairs of real/generated
    if n_samples <= 3:
        rows, cols = 2, 3  # 2 rows (real/generated), 3 columns
    elif n_samples <= 6:
        rows, cols = 2, 6
    else:
        rows, cols = 3, 4  # More compact layout
    
    fig, axes = plt.subplots(rows, cols, figsize=figsize)
    axes = axes.flatten()
    
    colors = plt.cm.tab10(np.linspace(0, 1, 10))
    
    for i in range(n_samples):
        # Real trajectories
        real_ax_idx = i
        if real_ax_idx < len(axes):
            ax_real = axes[real_ax_idx]
            real_sample = real_trajectories[i]
            real_mask = agent_masks[i]
            
            active_count = 0
            for j, (traj, is_active) in enumerate(zip(real_sample, real_mask)):
                if is_active:
                    color = colors[j % len(colors)]
                    ax_real.plot(traj[:, 0], traj[:, 1], color=color, alpha=0.8, linewidth=2.5)
                    ax_real.plot(traj[0, 0], traj[0, 1], 's', color=color, markersize=8)
                    ax_real.plot(traj[-1, 0], traj[-1, 1], 'o', color=color, markersize=8)
                    active_count += 1
            
            ax_real.set_aspect('equal')
            ax_real.grid(True, alpha=0.3)
            ax_real.set_xlim(-1.2, 1.2)
            ax_real.set_ylim(-1.2, 1.2)
            ax_real.set_title(f'Real #{i+1} ({active_count} agents)', fontsize=11, fontweight='bold')
        
        # Generated trajectories  
        gen_ax_idx = i + n_samples
        if gen_ax_idx < len(axes):
            ax_gen = axes[gen_ax_idx]
            gen_sample = generated_trajectories[i]
            gen_mask = agent_masks[i]  # Use same mask
            
            active_count = 0
            for j, (traj, is_active) in enumerate(zip(gen_sample, gen_mask)):
                if is_active:
                    color = colors[j % len(colors)]
                    ax_gen.plot(traj[:, 0], traj[:, 1], color=color, alpha=0.8, 
                              linewidth=2.5, linestyle='--')
                    ax_gen.plot(traj[0, 0], traj[0, 1], '^', color=color, markersize=8)
                    ax_gen.plot(traj[-1, 0], traj[-1, 1], 'v', color=color, markersize=8)
                    active_count += 1
            
            ax_gen.set_aspect('equal')
            ax_gen.grid(True, alpha=0.3)
            ax_gen.set_xlim(-1.2, 1.2)
            ax_gen.set_ylim(-1.2, 1.2)
            ax_gen.set_title(f'Generated #{i+1} ({active_count} agents)', fontsize=11, fontweight='bold')
        
        # Add text description as figure text
        if i < len(text_descriptions):
            text = text_descriptions[i][:50] + "..." if len(text_descriptions[i]) > 50 else text_descriptions[i]
            # Position text between real and generated plots
            fig.text(0.1 + (i % cols) * (0.8 / cols), 0.45, f'"{text}"', 
                    fontsize=10, ha='left', va='center', style='italic',
                    bbox=dict(boxstyle="round,pad=0.3", facecolor="lightblue", alpha=0.7))
    
    # Remove unused subplots
    for i in range(2 * n_samples, len(axes)):
        fig.delaxes(axes[i])
    
    plt.suptitle(title, fontsize=16, fontweight='bold')
    plt.tight_layout()
    plt.subplots_adjust(top=0.9)  # Make room for suptitle
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Multi-agent final results plot saved: {save_path}")
        plt.close()  # Close to save memory
    else:
        plt.show()


# Statistical analysis functions
def calculate_trajectory_metrics(trajectories: np.ndarray) -> Dict[str, float]:
    """
    Calculate various metrics for trajectory analysis.
    
    Args:
        trajectories: Array of trajectories [N, seq_len, 2]
        
    Returns:
        Dictionary of metrics
    """
    metrics = {}
    
    # Path length
    path_lengths = []
    for traj in trajectories:
        diffs = np.diff(traj, axis=0)
        lengths = np.sqrt(np.sum(diffs**2, axis=1))
        path_lengths.append(np.sum(lengths))
    
    metrics['mean_path_length'] = np.mean(path_lengths)
    metrics['std_path_length'] = np.std(path_lengths)
    
    # Displacement (start to end distance)
    displacements = []
    for traj in trajectories:
        disp = np.linalg.norm(traj[-1] - traj[0])
        displacements.append(disp)
    
    metrics['mean_displacement'] = np.mean(displacements)
    metrics['std_displacement'] = np.std(displacements)
    
    # Bounding box area
    areas = []
    for traj in trajectories:
        min_coords = np.min(traj, axis=0)
        max_coords = np.max(traj, axis=0)
        area = (max_coords[0] - min_coords[0]) * (max_coords[1] - min_coords[1])
        areas.append(area)
    
    metrics['mean_area'] = np.mean(areas)
    metrics['std_area'] = np.std(areas)
    
    return metrics


def print_metrics_comparison(real_metrics: Dict[str, float], generated_metrics: Dict[str, float]) -> None:
    """Print comparison of trajectory metrics."""
    print("\n📊 Trajectory Metrics Comparison:")
    print("=" * 50)
    
    for key in real_metrics:
        real_val = real_metrics[key]
        gen_val = generated_metrics[key]
        diff_pct = abs(real_val - gen_val) / real_val * 100 if real_val != 0 else 0
        
        print(f"{key:20s}: Real={real_val:.4f}, Gen={gen_val:.4f}, Diff={diff_pct:.1f}%")


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
    if os.path.exists("test_output"):
        shutil.rmtree("test_output")
    
    print("Utility functions test completed!") 