"""
Dataset Generation Script for 2D Trajectory Diffusion Model
Generates various mathematical patterns as 2D trajectories.
"""

import numpy as np
import matplotlib.pyplot as plt
import os
from typing import Tuple, List, Dict
import json


def generate_sine_wave(n_points: int = 100, amplitude: float = 1.0, frequency: float = 1.0, phase: float = 0.0) -> np.ndarray:
    """Generate a sine wave trajectory in 2D space."""
    t = np.linspace(0, 4*np.pi, n_points)
    x = t
    y = amplitude * np.sin(frequency * t + phase)
    return np.column_stack([x, y])


def generate_spiral(n_points: int = 100, a: float = 0.5, b: float = 0.2) -> np.ndarray:
    """Generate an Archimedean spiral trajectory."""
    t = np.linspace(0, 6*np.pi, n_points)
    r = a + b * t
    x = r * np.cos(t)
    y = r * np.sin(t)
    return np.column_stack([x, y])


def generate_circle(n_points: int = 100, radius: float = 1.0, center: Tuple[float, float] = (0, 0)) -> np.ndarray:
    """Generate a circular trajectory."""
    t = np.linspace(0, 2*np.pi, n_points)
    x = center[0] + radius * np.cos(t)
    y = center[1] + radius * np.sin(t)
    return np.column_stack([x, y])


def generate_lemniscate(n_points: int = 100, a: float = 1.0) -> np.ndarray:
    """Generate a figure-8 (lemniscate) trajectory."""
    t = np.linspace(0, 2*np.pi, n_points)
    x = a * np.cos(t) / (1 + np.sin(t)**2)
    y = a * np.sin(t) * np.cos(t) / (1 + np.sin(t)**2)
    return np.column_stack([x, y])


def generate_cardioid(n_points: int = 100, a: float = 1.0) -> np.ndarray:
    """Generate a cardioid trajectory."""
    t = np.linspace(0, 2*np.pi, n_points)
    x = a * (2 * np.cos(t) - np.cos(2*t))
    y = a * (2 * np.sin(t) - np.sin(2*t))
    return np.column_stack([x, y])


def normalize_trajectory(trajectory: np.ndarray) -> np.ndarray:
    """Normalize trajectory to [-1, 1] range."""
    min_vals = trajectory.min(axis=0)
    max_vals = trajectory.max(axis=0)
    range_vals = max_vals - min_vals
    range_vals[range_vals == 0] = 1  # Avoid division by zero
    
    normalized = 2 * (trajectory - min_vals) / range_vals - 1
    return normalized


def add_noise(trajectory: np.ndarray, noise_level: float = 0.05) -> np.ndarray:
    """Add Gaussian noise to trajectory."""
    noise = np.random.normal(0, noise_level, trajectory.shape)
    return trajectory + noise


def generate_dataset(
    n_samples_per_type: int = 1000,
    n_points: int = 100,
    output_dir: str = "data",
    add_noise_prob: float = 0.3,
    noise_level: float = 0.05
) -> None:
    """Generate complete dataset with various trajectory types."""
    
    os.makedirs(output_dir, exist_ok=True)
    
    trajectories = []
    labels = []
    
    pattern_generators = {
        'sine': lambda: generate_sine_wave(
            n_points, 
            amplitude=np.random.uniform(0.5, 2.0),
            frequency=np.random.uniform(0.5, 2.0),
            phase=np.random.uniform(0, 2*np.pi)
        ),
        'spiral': lambda: generate_spiral(
            n_points,
            a=np.random.uniform(0.1, 0.8),
            b=np.random.uniform(0.1, 0.5)
        ),
        'circle': lambda: generate_circle(
            n_points,
            radius=np.random.uniform(0.5, 2.0),
            center=(np.random.uniform(-0.5, 0.5), np.random.uniform(-0.5, 0.5))
        ),
        'lemniscate': lambda: generate_lemniscate(
            n_points,
            a=np.random.uniform(0.5, 2.0)
        ),
        'cardioid': lambda: generate_cardioid(
            n_points,
            a=np.random.uniform(0.5, 1.5)
        )
    }
    
    print(f"Generating {n_samples_per_type} samples for each pattern type...")
    
    for pattern_name, generator in pattern_generators.items():
        print(f"Generating {pattern_name} patterns...")
        
        for i in range(n_samples_per_type):
            # Generate base trajectory
            trajectory = generator()
            
            # Normalize to [-1, 1]
            trajectory = normalize_trajectory(trajectory)
            
            # Add noise with some probability
            if np.random.random() < add_noise_prob:
                trajectory = add_noise(trajectory, noise_level)
                trajectory = np.clip(trajectory, -1, 1)  # Ensure bounds after noise
            
            trajectories.append(trajectory)
            labels.append(pattern_name)
    
    # Convert to numpy arrays
    trajectories = np.array(trajectories)  # Shape: (N, n_points, 2)
    
    # Save dataset
    np.save(os.path.join(output_dir, 'trajectories.npy'), trajectories)
    
    with open(os.path.join(output_dir, 'labels.json'), 'w') as f:
        json.dump(labels, f)
    
    # Save metadata
    metadata = {
        'n_samples_per_type': n_samples_per_type,
        'n_points': n_points,
        'total_samples': len(trajectories),
        'pattern_types': list(pattern_generators.keys()),
        'noise_probability': add_noise_prob,
        'noise_level': noise_level
    }
    
    with open(os.path.join(output_dir, 'metadata.json'), 'w') as f:
        json.dump(metadata, f, indent=2)
    
    print(f"Dataset generated successfully!")
    print(f"Total samples: {len(trajectories)}")
    print(f"Shape: {trajectories.shape}")
    print(f"Saved to: {output_dir}/")
    
    # Create visualization
    visualize_samples(trajectories, labels, output_dir)


def visualize_samples(trajectories: np.ndarray, labels: List[str], output_dir: str) -> None:
    """Create visualization of sample trajectories."""
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    axes = axes.flatten()
    
    pattern_types = ['sine', 'spiral', 'circle', 'lemniscate', 'cardioid']
    
    for i, pattern_type in enumerate(pattern_types):
        # Find indices for this pattern type
        indices = [j for j, label in enumerate(labels) if label == pattern_type]
        
        # Plot first 5 samples of this type
        for j in range(min(5, len(indices))):
            idx = indices[j]
            traj = trajectories[idx]
            axes[i].plot(traj[:, 0], traj[:, 1], alpha=0.7, linewidth=1)
        
        axes[i].set_title(f'{pattern_type.capitalize()} Trajectories')
        axes[i].set_aspect('equal')
        axes[i].grid(True, alpha=0.3)
        axes[i].set_xlim(-1.1, 1.1)
        axes[i].set_ylim(-1.1, 1.1)
    
    # Remove empty subplot
    fig.delaxes(axes[5])
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'sample_trajectories.png'), dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"Sample visualization saved to: {output_dir}/sample_trajectories.png")


if __name__ == "__main__":
    # Generate dataset with default parameters
    generate_dataset(
        n_samples_per_type=1000,
        n_points=100,
        output_dir="data",
        add_noise_prob=0.3,
        noise_level=0.05
    ) 