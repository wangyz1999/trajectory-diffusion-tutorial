"""
Dataset Generation Script for Text-Conditioned 2D Trajectory Diffusion Model
Generates various mathematical patterns with creative textual descriptions.
"""

import numpy as np
import matplotlib.pyplot as plt
import os
from typing import Tuple, List, Dict
import json
import random
from tqdm import tqdm


def generate_circle(n_points: int = 100, radius: float = 1.0, center: Tuple[float, float] = (0, 0)) -> np.ndarray:
    """Generate a circular trajectory."""
    t = np.linspace(0, 2*np.pi, n_points)
    x = center[0] + radius * np.cos(t)
    y = center[1] + radius * np.sin(t)
    return np.column_stack([x, y])


def generate_spiral(n_points: int = 100, a: float = 0.5, b: float = 0.2) -> np.ndarray:
    """Generate an Archimedean spiral trajectory."""
    t = np.linspace(0, 6*np.pi, n_points)
    r = a + b * t
    x = r * np.cos(t)
    y = r * np.sin(t)
    return np.column_stack([x, y])


def generate_sine_wave(n_points: int = 100, amplitude: float = 1.0, frequency: float = 1.0, phase: float = 0.0) -> np.ndarray:
    """Generate a sine wave trajectory."""
    t = np.linspace(0, 4*np.pi, n_points)
    x = t
    y = amplitude * np.sin(frequency * t + phase)
    return np.column_stack([x, y])


def generate_square_wave(n_points: int = 100, amplitude: float = 1.0, frequency: float = 1.0) -> np.ndarray:
    """Generate a square wave trajectory."""
    t = np.linspace(0, 4*np.pi, n_points)
    x = t
    y = amplitude * np.sign(np.sin(frequency * t))
    return np.column_stack([x, y])


def generate_sawtooth_wave(n_points: int = 100, amplitude: float = 1.0, frequency: float = 1.0) -> np.ndarray:
    """Generate a sawtooth wave trajectory."""
    t = np.linspace(0, 4*np.pi, n_points)
    x = t
    y = amplitude * ((frequency * t) % (2*np.pi) - np.pi) / np.pi
    return np.column_stack([x, y])


def generate_triangle(n_points: int = 100, size: float = 1.0, center: Tuple[float, float] = (0, 0)) -> np.ndarray:
    """Generate a triangular trajectory."""
    angles = np.array([0, 2*np.pi/3, 4*np.pi/3, 0])  # Close the triangle
    vertices = np.array([[center[0] + size * np.cos(a), center[1] + size * np.sin(a)] for a in angles])
    
    # Interpolate uniformly around the triangle
    trajectory = []
    total_segments = len(vertices) - 1
    points_per_segment = n_points // total_segments
    
    for i in range(total_segments):
        if i == total_segments - 1:  # Last segment gets remaining points
            segment_points = n_points - len(trajectory)
        else:
            segment_points = points_per_segment
        
        t = np.linspace(0, 1, segment_points + (1 if i > 0 else 0))
        if i > 0:
            t = t[1:]  # Skip first point to avoid duplication
        
        seg = vertices[i] * (1 - t[:, None]) + vertices[i + 1] * t[:, None]
        trajectory.extend(seg)
    
    # Ensure exact length
    trajectory = np.array(trajectory)
    if len(trajectory) > n_points:
        trajectory = trajectory[:n_points]
    elif len(trajectory) < n_points:
        # Pad by repeating last points
        padding = np.tile(trajectory[-1:], (n_points - len(trajectory), 1))
        trajectory = np.vstack([trajectory, padding])
    
    return trajectory


def generate_square(n_points: int = 100, size: float = 1.0, center: Tuple[float, float] = (0, 0)) -> np.ndarray:
    """Generate a square trajectory."""
    half_size = size / 2
    vertices = np.array([
        [center[0] - half_size, center[1] - half_size],
        [center[0] + half_size, center[1] - half_size],
        [center[0] + half_size, center[1] + half_size],
        [center[0] - half_size, center[1] + half_size],
        [center[0] - half_size, center[1] - half_size]  # Close the square
    ])
    
    trajectory = []
    total_segments = len(vertices) - 1
    points_per_segment = n_points // total_segments
    
    for i in range(total_segments):
        if i == total_segments - 1:  # Last segment gets remaining points
            segment_points = n_points - len(trajectory)
        else:
            segment_points = points_per_segment
        
        t = np.linspace(0, 1, segment_points + (1 if i > 0 else 0))
        if i > 0:
            t = t[1:]  # Skip first point to avoid duplication
        
        seg = vertices[i] * (1 - t[:, None]) + vertices[i + 1] * t[:, None]
        trajectory.extend(seg)
    
    # Ensure exact length
    trajectory = np.array(trajectory)
    if len(trajectory) > n_points:
        trajectory = trajectory[:n_points]
    elif len(trajectory) < n_points:
        # Pad by repeating last points
        padding = np.tile(trajectory[-1:], (n_points - len(trajectory), 1))
        trajectory = np.vstack([trajectory, padding])
    
    return trajectory


def generate_star(n_points: int = 100, num_spikes: int = 5, outer_radius: float = 1.0, inner_radius: float = 0.4, center: Tuple[float, float] = (0, 0)) -> np.ndarray:
    """Generate a star trajectory."""
    angles = []
    for i in range(num_spikes):
        outer_angle = 2 * np.pi * i / num_spikes
        inner_angle = 2 * np.pi * (i + 0.5) / num_spikes
        angles.extend([outer_angle, inner_angle])
    angles.append(angles[0])  # Close the star
    
    vertices = []
    for i, angle in enumerate(angles[:-1]):
        radius = outer_radius if i % 2 == 0 else inner_radius
        x = center[0] + radius * np.cos(angle)
        y = center[1] + radius * np.sin(angle)
        vertices.append([x, y])
    vertices.append(vertices[0])  # Close
    
    trajectory = []
    total_segments = len(vertices) - 1
    points_per_segment = n_points // total_segments
    
    for i in range(total_segments):
        if i == total_segments - 1:  # Last segment gets remaining points
            segment_points = n_points - len(trajectory)
        else:
            segment_points = points_per_segment
        
        t = np.linspace(0, 1, segment_points + (1 if i > 0 else 0))
        if i > 0:
            t = t[1:]  # Skip first point to avoid duplication
            
        seg = np.array(vertices[i]) * (1 - t[:, None]) + np.array(vertices[i + 1]) * t[:, None]
        trajectory.extend(seg)
    
    # Ensure exact length
    trajectory = np.array(trajectory)
    if len(trajectory) > n_points:
        trajectory = trajectory[:n_points]
    elif len(trajectory) < n_points:
        # Pad by repeating last points
        padding = np.tile(trajectory[-1:], (n_points - len(trajectory), 1))
        trajectory = np.vstack([trajectory, padding])
    
    return trajectory


def generate_heart(n_points: int = 100, size: float = 1.0, center: Tuple[float, float] = (0, 0)) -> np.ndarray:
    """Generate a heart-shaped trajectory."""
    t = np.linspace(0, 2*np.pi, n_points)
    x = size * (16 * np.sin(t)**3)
    y = size * (13 * np.cos(t) - 5 * np.cos(2*t) - 2 * np.cos(3*t) - np.cos(4*t))
    
    # Scale and center
    x = x / 20 + center[0]
    y = y / 20 + center[1]
    return np.column_stack([x, y])


def generate_infinity(n_points: int = 100, size: float = 1.0, center: Tuple[float, float] = (0, 0)) -> np.ndarray:
    """Generate an infinity (figure-8) trajectory."""
    t = np.linspace(0, 2*np.pi, n_points)
    x = size * np.cos(t) / (1 + np.sin(t)**2)
    y = size * np.sin(t) * np.cos(t) / (1 + np.sin(t)**2)
    return np.column_stack([x + center[0], y + center[1]])


def generate_zigzag(n_points: int = 100, amplitude: float = 1.0, frequency: int = 5) -> np.ndarray:
    """Generate a zigzag trajectory."""
    x = np.linspace(-2, 2, n_points)
    y = amplitude * np.array([(-1)**int(frequency * xi / 4) for xi in x])
    # Smooth the transitions
    for i in range(1, len(y) - 1):
        if y[i] != y[i-1]:
            y[i] = (y[i-1] + y[i+1]) / 2
    return np.column_stack([x, y])


def generate_random_walk(n_points: int = 100, step_size: float = 0.1) -> np.ndarray:
    """Generate a random walk trajectory."""
    steps = np.random.randn(n_points, 2) * step_size
    trajectory = np.cumsum(steps, axis=0)
    return trajectory


def generate_line(n_points: int = 100, start: Tuple[float, float] = (-1, -1), end: Tuple[float, float] = (1, 1)) -> np.ndarray:
    """Generate a straight line trajectory."""
    t = np.linspace(0, 1, n_points)
    x = start[0] * (1 - t) + end[0] * t
    y = start[1] * (1 - t) + end[1] * t
    return np.column_stack([x, y])


def generate_multiple_circles(n_points: int = 100, num_circles: int = 2, radius: float = 0.5) -> np.ndarray:
    """Generate multiple circles trajectory."""
    points_per_circle = n_points // num_circles
    trajectory = []
    
    for i in range(num_circles):
        center_x = (i - (num_circles - 1) / 2) * 1.5
        center_y = 0
        
        if i == num_circles - 1:  # Last circle gets remaining points
            circle_points = n_points - len(trajectory)
        else:
            circle_points = points_per_circle
            
        circle = generate_circle(circle_points, radius, (center_x, center_y))
        trajectory.extend(circle)
    
    # Ensure exact length
    trajectory = np.array(trajectory)
    if len(trajectory) != n_points:
        # Resize to exact length
        if len(trajectory) > n_points:
            trajectory = trajectory[:n_points]
        else:
            # Interpolate to exact length
            t_old = np.linspace(0, 1, len(trajectory))
            t_new = np.linspace(0, 1, n_points)
            x_new = np.interp(t_new, t_old, trajectory[:, 0])
            y_new = np.interp(t_new, t_old, trajectory[:, 1])
            trajectory = np.column_stack([x_new, y_new])
    
    return trajectory


def generate_text_descriptions():
    """Generate diverse and creative text description templates."""
    
    # Basic shape descriptors
    shape_descriptors = {
        'circle': ['circle', 'circular shape', 'round shape', 'ring', 'loop'],
        'spiral': ['spiral', 'coil', 'helix', 'swirl', 'vortex'],
        'sine': ['sine wave', 'wavy line', 'smooth curve', 'wave pattern', 'oscillation'],
        'square_wave': ['square wave', 'step pattern', 'rectangular wave', 'digital signal'],
        'sawtooth': ['sawtooth wave', 'ramp pattern', 'triangular wave', 'zigzag pattern'],
        'triangle': ['triangle', 'triangular shape', 'three-sided shape', 'delta shape'],
        'square': ['square', 'rectangular shape', 'box shape', 'four-sided shape'],
        'star': ['star', 'star shape', 'pointed shape', 'asterisk'],
        'heart': ['heart', 'heart shape', 'love symbol', 'cardiac shape'],
        'infinity': ['infinity symbol', 'figure-eight', 'lemniscate', 'endless loop'],
        'zigzag': ['zigzag', 'jagged line', 'sawtooth pattern', 'angular path'],
        'random_walk': ['random path', 'chaotic trajectory', 'wandering line', 'erratic movement'],
        'line': ['straight line', 'linear path', 'direct route', 'simple line'],
        'multiple_circles': ['multiple circles', 'overlapping rings', 'chain of circles', 'circle pattern']
    }
    
    # Size modifiers
    size_modifiers = ['tiny', 'small', 'medium', 'large', 'huge', 'massive', 'perfect', 'precise']
    
    # Position modifiers
    position_modifiers = [
        'at the top', 'at the bottom', 'on the left', 'on the right', 'in the center',
        'at top left', 'at top right', 'at bottom left', 'at bottom right',
        'slightly offset', 'perfectly centered', 'off-center'
    ]
    
    # Style modifiers
    style_modifiers = [
        'smooth', 'rough', 'perfect', 'imperfect', 'clean', 'messy', 'elegant', 'simple',
        'complex', 'detailed', 'precise', 'loose', 'tight', 'flowing', 'rigid'
    ]
    
    # Creative descriptors
    creative_descriptors = [
        'beautiful', 'artistic', 'geometric', 'abstract', 'mathematical', 'organic',
        'mechanical', 'natural', 'synthetic', 'classic', 'modern', 'minimalist'
    ]
    
    return {
        'shapes': shape_descriptors,
        'sizes': size_modifiers,
        'positions': position_modifiers,
        'styles': style_modifiers,
        'creative': creative_descriptors
    }


def generate_text_for_pattern(pattern_type: str, pattern_params: dict) -> str:
    """Generate a creative text description for a pattern."""
    templates = generate_text_descriptions()
    
    # Get base shape description
    base_shapes = templates['shapes'].get(pattern_type, [pattern_type])
    base_shape = random.choice(base_shapes)
    
    # Randomly choose description complexity
    complexity = random.choice(['simple', 'medium', 'complex', 'creative'])
    
    if complexity == 'simple':
        # Simple: "A circle"
        article = 'An' if base_shape[0].lower() in 'aeiou' else 'A'
        return f"{article} {base_shape}"
    
    elif complexity == 'medium':
        # Medium: "A large circle" or "A circle at the center"
        if random.choice([True, False]):
            size = random.choice(templates['sizes'])
            article = 'An' if (size[0].lower() in 'aeiou' or base_shape[0].lower() in 'aeiou') else 'A'
            return f"{article} {size} {base_shape}"
        else:
            position = random.choice(templates['positions'])
            article = 'An' if base_shape[0].lower() in 'aeiou' else 'A'
            return f"{article} {base_shape} {position}"
    
    elif complexity == 'complex':
        # Complex: "A smooth large circle at the center"
        style = random.choice(templates['styles'])
        size = random.choice(templates['sizes'])
        article = 'An' if (style[0].lower() in 'aeiou' or size[0].lower() in 'aeiou' or base_shape[0].lower() in 'aeiou') else 'A'
        
        if random.choice([True, False]):
            position = random.choice(templates['positions'])
            return f"{article} {style} {size} {base_shape} {position}"
        else:
            return f"{article} {style} {size} {base_shape}"
    
    else:  # creative
        # Creative: "Beautiful geometric patterns" or parameter-based descriptions
        creative = random.choice(templates['creative'])
        
        # Parameter-specific descriptions
        if pattern_type == 'circle' and 'radius' in pattern_params:
            if pattern_params['radius'] > 1.5:
                return f"A {creative} huge circular path"
            elif pattern_params['radius'] < 0.5:
                return f"A tiny {creative} circle"
        
        if pattern_type == 'star' and 'num_spikes' in pattern_params:
            num_spikes = pattern_params['num_spikes']
            return f"A {creative} {num_spikes}-pointed star"
        
        if pattern_type == 'multiple_circles' and 'num_circles' in pattern_params:
            num_circles = pattern_params['num_circles']
            if num_circles == 2:
                return f"Two {creative} overlapping circles"
            else:
                return f"{num_circles} {creative} circles in a pattern"
        
        # Fallback creative description
        style = random.choice(templates['styles'])
        return f"{creative.capitalize()} {style} {base_shape}"


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
    n_samples_per_type: int = 200,
    n_points: int = 100,
    output_dir: str = "data",
    add_noise_prob: float = 0.3,
    noise_level: float = 0.05
) -> None:
    """Generate complete dataset with various trajectory types and text descriptions."""
    
    os.makedirs(output_dir, exist_ok=True)
    
    trajectories = []
    labels = []
    text_descriptions = []
    
    pattern_generators = {
        'circle': lambda: generate_circle(
            n_points, 
            radius=np.random.uniform(0.3, 1.8),
            center=(np.random.uniform(-0.3, 0.3), np.random.uniform(-0.3, 0.3))
        ),
        'spiral': lambda: generate_spiral(
            n_points,
            a=np.random.uniform(0.1, 0.8),
            b=np.random.uniform(0.05, 0.3)
        ),
        'sine': lambda: generate_sine_wave(
            n_points, 
            amplitude=np.random.uniform(0.5, 2.0),
            frequency=np.random.uniform(0.5, 2.0),
            phase=np.random.uniform(0, 2*np.pi)
        ),
        'square_wave': lambda: generate_square_wave(
            n_points,
            amplitude=np.random.uniform(0.5, 1.5),
            frequency=np.random.uniform(0.5, 2.0)
        ),
        'sawtooth': lambda: generate_sawtooth_wave(
            n_points,
            amplitude=np.random.uniform(0.5, 1.5),
            frequency=np.random.uniform(0.5, 2.0)
        ),
        'triangle': lambda: generate_triangle(
            n_points,
            size=np.random.uniform(0.5, 1.5),
            center=(np.random.uniform(-0.2, 0.2), np.random.uniform(-0.2, 0.2))
        ),
        'square': lambda: generate_square(
            n_points,
            size=np.random.uniform(0.8, 1.8),
            center=(np.random.uniform(-0.2, 0.2), np.random.uniform(-0.2, 0.2))
        ),
        'star': lambda: generate_star(
            n_points,
            num_spikes=np.random.choice([4, 5, 6, 8]),
            outer_radius=np.random.uniform(0.8, 1.5),
            inner_radius=np.random.uniform(0.3, 0.7),
            center=(np.random.uniform(-0.2, 0.2), np.random.uniform(-0.2, 0.2))
        ),
        'heart': lambda: generate_heart(
            n_points,
            size=np.random.uniform(0.8, 1.2),
            center=(np.random.uniform(-0.1, 0.1), np.random.uniform(-0.1, 0.1))
        ),
        'infinity': lambda: generate_infinity(
            n_points,
            size=np.random.uniform(0.8, 1.5),
            center=(np.random.uniform(-0.2, 0.2), np.random.uniform(-0.2, 0.2))
        ),
        'zigzag': lambda: generate_zigzag(
            n_points,
            amplitude=np.random.uniform(0.5, 1.5),
            frequency=np.random.randint(3, 8)
        ),
        'random_walk': lambda: generate_random_walk(
            n_points,
            step_size=np.random.uniform(0.05, 0.15)
        ),
        'line': lambda: generate_line(
            n_points,
            start=(np.random.uniform(-1.5, 0), np.random.uniform(-1.5, 1.5)),
            end=(np.random.uniform(0, 1.5), np.random.uniform(-1.5, 1.5))
        ),
        'multiple_circles': lambda: generate_multiple_circles(
            n_points,
            num_circles=np.random.choice([2, 3]),
            radius=np.random.uniform(0.3, 0.7)
        )
    }
    
    print(f"🎯 Generating {n_samples_per_type} samples for each pattern type...")
    print(f"📊 Total patterns: {len(pattern_generators)} types")
    print(f"🎨 Total samples: {n_samples_per_type * len(pattern_generators):,}")
    
    # Progress bar for pattern types
    pattern_pbar = tqdm(pattern_generators.items(), desc="Pattern Types", 
                       unit="pattern", colour='blue')
    
    for pattern_name, generator in pattern_pbar:
        pattern_pbar.set_description(f"Generating {pattern_name}")
        
        # Progress bar for samples within each pattern
        sample_pbar = tqdm(range(n_samples_per_type), desc=f"  {pattern_name} samples", 
                          leave=False, unit="sample", colour='green')
        
        for i in sample_pbar:
            # Store current random state for parameter extraction
            current_state = np.random.get_state()
            
            # Generate base trajectory
            trajectory = generator()
            
            # Extract parameters for text generation (approximate)
            pattern_params = {}
            if pattern_name == 'circle':
                pattern_params = {'radius': 1.0}  # Default values
            elif pattern_name == 'star':
                pattern_params = {'num_spikes': 5}
            elif pattern_name == 'multiple_circles':
                pattern_params = {'num_circles': 2}
            
            # Generate text description
            text_desc = generate_text_for_pattern(pattern_name, pattern_params)
            
            # Normalize to [-1, 1]
            trajectory = normalize_trajectory(trajectory)
            
            # Add noise with some probability
            if np.random.random() < add_noise_prob:
                trajectory = add_noise(trajectory, noise_level)
                trajectory = np.clip(trajectory, -1, 1)  # Ensure bounds after noise
            
            trajectories.append(trajectory)
            labels.append(pattern_name)
            text_descriptions.append(text_desc)
            
            # Update progress description
            sample_pbar.set_postfix({
                'noise': f'{add_noise_prob:.1%}',
                'total': len(trajectories)
            })
        
        sample_pbar.close()
    
    pattern_pbar.close()
    
    # Convert to numpy arrays
    trajectories = np.array(trajectories)  # Shape: (N, n_points, 2)
    
    print(f"\n💾 Saving dataset...")
    
    # Save dataset with progress
    save_tasks = [
        ("trajectories.npy", lambda: np.save(os.path.join(output_dir, 'trajectories.npy'), trajectories)),
        ("labels.json", lambda: json.dump(labels, open(os.path.join(output_dir, 'labels.json'), 'w'))),
        ("text_descriptions.json", lambda: json.dump(text_descriptions, open(os.path.join(output_dir, 'text_descriptions.json'), 'w'))),
        ("metadata.json", lambda: None)  # Will be handled separately
    ]
    
    for task_name, task_func in tqdm(save_tasks[:-1], desc="Saving files", unit="file"):
        task_func()
    
    # Save metadata
    metadata = {
        'n_samples_per_type': n_samples_per_type,
        'n_points': n_points,
        'total_samples': len(trajectories),
        'pattern_types': list(pattern_generators.keys()),
        'noise_probability': add_noise_prob,
        'noise_level': noise_level,
        'text_conditioned': True
    }
    
    with open(os.path.join(output_dir, 'metadata.json'), 'w') as f:
        json.dump(metadata, f, indent=2)
    
    print(f"\n✅ Text-conditioned dataset generated successfully!")
    print(f"📊 Total samples: {len(trajectories):,}")
    print(f"📐 Shape: {trajectories.shape}")
    print(f"🎨 Pattern types: {len(pattern_generators)}")
    print(f"💾 Saved to: {output_dir}/")
    
    # Create visualization
    print(f"\n🎨 Creating sample visualization...")
    visualize_samples_with_text(trajectories, labels, text_descriptions, output_dir)


def visualize_samples_with_text(trajectories: np.ndarray, labels: List[str], 
                               text_descriptions: List[str], output_dir: str) -> None:
    """Create visualization of sample trajectories with text descriptions."""
    
    # Get unique pattern types
    unique_patterns = list(set(labels))
    n_patterns = len(unique_patterns)
    
    # Create grid layout
    cols = 4
    rows = (n_patterns + cols - 1) // cols
    
    fig, axes = plt.subplots(rows, cols, figsize=(16, 4 * rows))
    if rows * cols == 1:
        axes = [axes]
    else:
        axes = axes.flatten()
    
    for i, pattern_type in enumerate(unique_patterns):
        if i >= len(axes):
            break
            
        ax = axes[i]
        
        # Find indices for this pattern type
        indices = [j for j, label in enumerate(labels) if label == pattern_type]
        
        # Plot first 3 samples of this type
        colors = ['blue', 'red', 'green']
        for j in range(min(3, len(indices))):
            idx = indices[j]
            traj = trajectories[idx]
            ax.plot(traj[:, 0], traj[:, 1], alpha=0.7, linewidth=2, 
                   color=colors[j], label=f'"{text_descriptions[idx]}"')
        
        ax.set_title(f'{pattern_type.capitalize()} Patterns', fontsize=12, fontweight='bold')
        ax.set_aspect('equal')
        ax.grid(True, alpha=0.3)
        ax.set_xlim(-1.2, 1.2)
        ax.set_ylim(-1.2, 1.2)
        ax.legend(fontsize=8, loc='upper right')
    
    # Remove empty subplots
    for i in range(len(unique_patterns), len(axes)):
        fig.delaxes(axes[i])
    
    plt.suptitle('Text-Conditioned Trajectory Samples', fontsize=16, fontweight='bold')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'sample_trajectories_with_text.png'), 
               dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"Sample visualization with text saved to: {output_dir}/sample_trajectories_with_text.png")


if __name__ == "__main__":
    # Generate text-conditioned dataset
    generate_dataset(
        n_samples_per_type=5000,  # More samples per type due to more types
        n_points=100,
        output_dir="data",
        add_noise_prob=0.3,
        noise_level=0.05
    ) 