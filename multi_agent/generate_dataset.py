"""
Multi-Agent Dataset Generation Script for Text-Conditioned 2D Trajectory Diffusion Model
Generates multiple mathematical patterns per sample with spatial location descriptions.
"""

import numpy as np
import matplotlib.pyplot as plt
import os
from typing import Tuple, List, Dict, Optional
import json
import random
from tqdm import tqdm


def generate_circle(n_points: int = 100, radius: float = 1.0, center: Tuple[float, float] = (0, 0)) -> np.ndarray:
    """Generate a circular trajectory."""
    t = np.linspace(0, 2*np.pi, n_points)
    x = center[0] + radius * np.cos(t)
    y = center[1] + radius * np.sin(t)
    return np.column_stack([x, y])


def generate_spiral(n_points: int = 100, a: float = 0.5, b: float = 0.2, center: Tuple[float, float] = (0, 0)) -> np.ndarray:
    """Generate an Archimedean spiral trajectory."""
    t = np.linspace(0, 6*np.pi, n_points)
    r = a + b * t
    x = center[0] + r * np.cos(t)
    y = center[1] + r * np.sin(t)
    return np.column_stack([x, y])


def generate_sine_wave(n_points: int = 100, amplitude: float = 1.0, frequency: float = 1.0, phase: float = 0.0, center: Tuple[float, float] = (0, 0)) -> np.ndarray:
    """Generate a sine wave trajectory."""
    t = np.linspace(0, 4*np.pi, n_points)
    x = center[0] + t
    y = center[1] + amplitude * np.sin(frequency * t + phase)
    return np.column_stack([x, y])


def generate_square_wave(n_points: int = 100, amplitude: float = 1.0, frequency: float = 1.0, center: Tuple[float, float] = (0, 0)) -> np.ndarray:
    """Generate a square wave trajectory."""
    t = np.linspace(0, 4*np.pi, n_points)
    x = center[0] + t
    y = center[1] + amplitude * np.sign(np.sin(frequency * t))
    return np.column_stack([x, y])


def generate_sawtooth_wave(n_points: int = 100, amplitude: float = 1.0, frequency: float = 1.0, center: Tuple[float, float] = (0, 0)) -> np.ndarray:
    """Generate a sawtooth wave trajectory."""
    t = np.linspace(0, 4*np.pi, n_points)
    x = center[0] + t
    y = center[1] + amplitude * ((frequency * t) % (2*np.pi) - np.pi) / np.pi
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


def generate_line(n_points: int = 100, start: Tuple[float, float] = (-1, -1), end: Tuple[float, float] = (1, 1)) -> np.ndarray:
    """Generate a straight line trajectory."""
    t = np.linspace(0, 1, n_points)
    x = start[0] * (1 - t) + end[0] * t
    y = start[1] * (1 - t) + end[1] * t
    return np.column_stack([x, y])


# Define spatial regions for multi-agent positioning
SPATIAL_REGIONS = {
    'center': (0.0, 0.0),
    'top': (0.0, 0.6),
    'bottom': (0.0, -0.6),
    'left': (-0.6, 0.0),
    'right': (0.6, 0.0),
    'top_left': (-0.5, 0.5),
    'top_right': (0.5, 0.5),
    'bottom_left': (-0.5, -0.5),
    'bottom_right': (0.5, -0.5),
}

REGION_NAMES = {
    'center': ['at the center', 'in the middle', 'centrally located'],
    'top': ['at the top', 'above', 'on top'],
    'bottom': ['at the bottom', 'below', 'underneath'],
    'left': ['on the left', 'to the left', 'left side'],
    'right': ['on the right', 'to the right', 'right side'],
    'top_left': ['at the top left', 'upper left', 'top-left corner'],
    'top_right': ['at the top right', 'upper right', 'top-right corner'],
    'bottom_left': ['at the bottom left', 'lower left', 'bottom-left corner'],
    'bottom_right': ['at the bottom right', 'lower right', 'bottom-right corner'],
}


def get_pattern_generators():
    """Get pattern generators that support center parameter for positioning."""
    return {
        'circle': lambda center, **kwargs: generate_circle(
            n_points=kwargs.get('n_points', 100),
            radius=kwargs.get('radius', np.random.uniform(0.2, 0.4)),
            center=center
        ),
        'spiral': lambda center, **kwargs: generate_spiral(
            n_points=kwargs.get('n_points', 100),
            a=kwargs.get('a', np.random.uniform(0.05, 0.15)),
            b=kwargs.get('b', np.random.uniform(0.02, 0.08)),
            center=center
        ),
        'triangle': lambda center, **kwargs: generate_triangle(
            n_points=kwargs.get('n_points', 100),
            size=kwargs.get('size', np.random.uniform(0.25, 0.4)),
            center=center
        ),
        'square': lambda center, **kwargs: generate_square(
            n_points=kwargs.get('n_points', 100),
            size=kwargs.get('size', np.random.uniform(0.3, 0.5)),
            center=center
        ),
        'star': lambda center, **kwargs: generate_star(
            n_points=kwargs.get('n_points', 100),
            num_spikes=kwargs.get('num_spikes', np.random.choice([4, 5, 6])),
            outer_radius=kwargs.get('outer_radius', np.random.uniform(0.2, 0.35)),
            inner_radius=kwargs.get('inner_radius', np.random.uniform(0.1, 0.2)),
            center=center
        ),
        'heart': lambda center, **kwargs: generate_heart(
            n_points=kwargs.get('n_points', 100),
            size=kwargs.get('size', np.random.uniform(0.15, 0.25)),
            center=center
        ),
        'infinity': lambda center, **kwargs: generate_infinity(
            n_points=kwargs.get('n_points', 100),
            size=kwargs.get('size', np.random.uniform(0.2, 0.35)),
            center=center
        ),
    }


def generate_multi_agent_sample(n_points: int = 100, min_agents: int = 1, max_agents: int = 5) -> Tuple[List[np.ndarray], List[str], List[str], str]:
    """Generate a multi-agent sample with spatial positioning and comprehensive description."""
    
    pattern_generators = get_pattern_generators()
    n_agents = np.random.randint(min_agents, max_agents + 1)
    
    # Select patterns and regions
    selected_patterns = np.random.choice(list(pattern_generators.keys()), size=n_agents, replace=True)
    available_regions = list(SPATIAL_REGIONS.keys())
    
    # For multiple agents, avoid center if possible and ensure good spacing
    if n_agents > 1:
        # Prioritize non-center regions for better visual separation
        non_center_regions = [r for r in available_regions if r != 'center']
        if len(non_center_regions) >= n_agents:
            selected_regions = np.random.choice(non_center_regions, size=n_agents, replace=False)
        else:
            # Include center if we need more regions
            selected_regions = np.random.choice(available_regions, size=n_agents, replace=False)
    else:
        selected_regions = np.random.choice(available_regions, size=n_agents, replace=False)
    
    trajectories = []
    pattern_types = []
    region_names = []
    
    # Generate each agent's trajectory
    for pattern_type, region in zip(selected_patterns, selected_regions):
        center = SPATIAL_REGIONS[region]
        generator = pattern_generators[pattern_type]
        
        # Generate trajectory at specified location
        trajectory = generator(center=center, n_points=n_points)
        
        trajectories.append(trajectory)
        pattern_types.append(pattern_type)
        region_names.append(region)
    
    # Generate comprehensive text description
    text_description = generate_multi_agent_description(pattern_types, region_names)
    
    return trajectories, pattern_types, region_names, text_description


def generate_multi_agent_description(pattern_types: List[str], region_names: List[str]) -> str:
    """Generate natural language description for multi-agent trajectory."""
    
    # Shape name mappings
    shape_names = {
        'circle': ['circle', 'circular shape', 'ring'],
        'spiral': ['spiral', 'coil', 'swirl'],
        'triangle': ['triangle', 'triangular shape'],
        'square': ['square', 'box', 'rectangular shape'],
        'star': ['star', 'star shape'],
        'heart': ['heart', 'heart shape'],
        'infinity': ['infinity symbol', 'figure-eight'],
    }
    
    if len(pattern_types) == 1:
        # Single agent description
        shape = random.choice(shape_names.get(pattern_types[0], [pattern_types[0]]))
        location = random.choice(REGION_NAMES[region_names[0]])
        
        templates = [
            f"A {shape} {location}",
            f"There is a {shape} {location}",
            f"A {shape} positioned {location}",
        ]
        return random.choice(templates)
    
    elif len(pattern_types) == 2:
        # Two agent descriptions with spatial relationships
        shape1 = random.choice(shape_names.get(pattern_types[0], [pattern_types[0]]))
        shape2 = random.choice(shape_names.get(pattern_types[1], [pattern_types[1]]))
        loc1 = random.choice(REGION_NAMES[region_names[0]])
        loc2 = random.choice(REGION_NAMES[region_names[1]])
        
        # Use "and" or "with" connectors
        templates = [
            f"A {shape1} {loc1} and a {shape2} {loc2}",
            f"A {shape1} {loc1} with a {shape2} {loc2}",
            f"There is a {shape1} {loc1} and a {shape2} {loc2}",
            f"A {shape2} {loc2} and a {shape1} {loc1}",  # Reverse order
        ]
        return random.choice(templates)
    
    else:
        # Multiple agents (3-5)
        descriptions = []
        for i, (pattern_type, region_name) in enumerate(zip(pattern_types, region_names)):
            shape = random.choice(shape_names.get(pattern_type, [pattern_type]))
            location = random.choice(REGION_NAMES[region_name])
            
            if i == 0:
                descriptions.append(f"a {shape} {location}")
            elif i == len(pattern_types) - 1:  # Last item
                descriptions.append(f"and a {shape} {location}")
            else:
                descriptions.append(f"a {shape} {location}")
        
        # Join with commas and proper grammar
        if len(descriptions) == 3:
            return f"There are {descriptions[0]}, {descriptions[1]}, {descriptions[2]}"
        else:
            first_part = ", ".join(descriptions[:-1])
            return f"There are {first_part}, {descriptions[-1]}"


def normalize_trajectory(trajectory: np.ndarray) -> np.ndarray:
    """Normalize trajectory to [-1, 1] range while preserving spatial relationships."""
    # For multi-agent, we want to preserve relative positions, so normalize globally
    min_vals = trajectory.min(axis=0)
    max_vals = trajectory.max(axis=0)
    range_vals = max_vals - min_vals
    range_vals[range_vals == 0] = 1  # Avoid division by zero
    
    # Scale to fit in [-1, 1] with some margin
    scale_factor = 1.8 / np.max(range_vals)  # Leave 10% margin
    center = (min_vals + max_vals) / 2
    
    normalized = (trajectory - center) * scale_factor
    return np.clip(normalized, -1, 1)


def add_noise(trajectory: np.ndarray, noise_level: float = 0.05) -> np.ndarray:
    """Add Gaussian noise to trajectory."""
    noise = np.random.normal(0, noise_level, trajectory.shape)
    return trajectory + noise


def generate_dataset(
    n_samples: int = 10000,
    n_points: int = 100,
    output_dir: str = "data",
    add_noise_prob: float = 0.3,
    noise_level: float = 0.05,
    min_agents: int = 1,
    max_agents: int = 5
) -> None:
    """Generate complete multi-agent dataset with spatial descriptions."""
    
    os.makedirs(output_dir, exist_ok=True)
    
    all_trajectories = []  # List of lists of trajectories (one list per sample)
    all_pattern_types = []  # List of lists of pattern types
    all_region_names = []  # List of lists of region names  
    text_descriptions = []
    
    print(f"🤖 Generating {n_samples:,} multi-agent samples...")
    print(f"👥 Agents per sample: {min_agents}-{max_agents}")
    print(f"📍 Spatial positioning enabled")
    
    # Progress bar for samples
    sample_pbar = tqdm(range(n_samples), desc="Generating samples", unit="sample", colour='blue')
    
    for i in sample_pbar:
        # Generate multi-agent sample
        trajectories, pattern_types, region_names, text_desc = generate_multi_agent_sample(
            n_points=n_points, 
            min_agents=min_agents, 
            max_agents=max_agents
        )
        
        # Combine all trajectories into single array for normalization
        combined_traj = np.vstack(trajectories)
        combined_traj = normalize_trajectory(combined_traj)
        
        # Add noise with some probability
        if np.random.random() < add_noise_prob:
            combined_traj = add_noise(combined_traj, noise_level)
            combined_traj = np.clip(combined_traj, -1, 1)
        
        # Split back into individual trajectories
        split_trajectories = []
        start_idx = 0
        for traj in trajectories:
            end_idx = start_idx + len(traj)
            split_trajectories.append(combined_traj[start_idx:end_idx])
            start_idx = end_idx
        
        all_trajectories.append(split_trajectories)
        all_pattern_types.append(pattern_types)
        all_region_names.append(region_names)
        text_descriptions.append(text_desc)
        
        # Update progress
        if i % 1000 == 0 and i > 0:
            avg_agents = np.mean([len(patterns) for patterns in all_pattern_types])
            sample_pbar.set_postfix({
                'avg_agents': f'{avg_agents:.1f}',
                'noise': f'{add_noise_prob:.1%}'
            })
    
    sample_pbar.close()
    
    print(f"\n💾 Saving multi-agent dataset...")
    
    # Convert to format suitable for saving
    # For trajectories: pad to max number of agents and create mask
    max_agents_actual = max(len(sample_trajs) for sample_trajs in all_trajectories)
    
    # Create padded trajectory array: (n_samples, max_agents, n_points, 2)
    padded_trajectories = np.zeros((n_samples, max_agents_actual, n_points, 2))
    agent_masks = np.zeros((n_samples, max_agents_actual), dtype=bool)  # True where agent exists
    
    for i, sample_trajs in enumerate(all_trajectories):
        for j, traj in enumerate(sample_trajs):
            padded_trajectories[i, j] = traj
            agent_masks[i, j] = True
    
    # Save dataset
    save_tasks = [
        ("trajectories.npy", lambda: np.save(os.path.join(output_dir, 'trajectories.npy'), padded_trajectories)),
        ("agent_masks.npy", lambda: np.save(os.path.join(output_dir, 'agent_masks.npy'), agent_masks)),
        ("pattern_types.json", lambda: json.dump(all_pattern_types, open(os.path.join(output_dir, 'pattern_types.json'), 'w'))),
        ("region_names.json", lambda: json.dump(all_region_names, open(os.path.join(output_dir, 'region_names.json'), 'w'))),
        ("text_descriptions.json", lambda: json.dump(text_descriptions, open(os.path.join(output_dir, 'text_descriptions.json'), 'w'))),
    ]
    
    for task_name, task_func in tqdm(save_tasks, desc="Saving files", unit="file"):
        task_func()
    
    # Save metadata
    agent_counts = [len(patterns) for patterns in all_pattern_types]
    pattern_generators = get_pattern_generators()
    
    metadata = {
        'n_samples': n_samples,
        'n_points': n_points,
        'min_agents': min_agents,
        'max_agents': max_agents,
        'max_agents_actual': max_agents_actual,
        'avg_agents_per_sample': np.mean(agent_counts),
        'pattern_types': list(pattern_generators.keys()),
        'spatial_regions': list(SPATIAL_REGIONS.keys()),
        'noise_probability': add_noise_prob,
        'noise_level': noise_level,
        'multi_agent': True,
        'spatial_conditioning': True,
        'trajectory_shape': list(padded_trajectories.shape),
        'agent_distribution': {
            f'{i}_agents': int(np.sum(np.array(agent_counts) == i))
            for i in range(min_agents, max_agents + 1)
        }
    }
    
    with open(os.path.join(output_dir, 'metadata.json'), 'w') as f:
        json.dump(metadata, f, indent=2)
    
    print(f"\n✅ Multi-agent dataset generated successfully!")
    print(f"📊 Total samples: {n_samples:,}")
    print(f"🤖 Average agents per sample: {np.mean(agent_counts):.1f}")
    print(f"📐 Trajectory shape: {padded_trajectories.shape}")
    print(f"🎨 Pattern types: {len(pattern_generators)}")
    print(f"📍 Spatial regions: {len(SPATIAL_REGIONS)}")
    print(f"💾 Saved to: {output_dir}/")
    
    # Create visualization
    print(f"\n🎨 Creating sample visualization...")
    visualize_multi_agent_samples(padded_trajectories, agent_masks, all_pattern_types, 
                                 all_region_names, text_descriptions, output_dir)


def visualize_multi_agent_samples(trajectories: np.ndarray, agent_masks: np.ndarray, 
                                 pattern_types: List[List[str]], region_names: List[List[str]],
                                 text_descriptions: List[str], output_dir: str, n_samples_to_show: int = 12) -> None:
    """Create visualization of multi-agent trajectory samples."""
    
    # Create grid layout
    cols = 4
    rows = 3
    
    fig, axes = plt.subplots(rows, cols, figsize=(16, 12))
    axes = axes.flatten()
    
    colors = ['blue', 'red', 'green', 'orange', 'purple', 'brown', 'pink', 'gray']
    
    for i in range(n_samples_to_show):
        if i >= len(trajectories):
            break
            
        ax = axes[i]
        sample_trajs = trajectories[i]
        sample_mask = agent_masks[i]
        
        # Plot each agent's trajectory
        for j, (traj, is_active) in enumerate(zip(sample_trajs, sample_mask)):
            if is_active:
                color = colors[j % len(colors)]
                pattern_type = pattern_types[i][j] if j < len(pattern_types[i]) else 'unknown'
                ax.plot(traj[:, 0], traj[:, 1], color=color, linewidth=2, 
                       alpha=0.8, label=f'Agent {j+1}: {pattern_type}')
        
        ax.set_title(f'Sample {i+1}\n"{text_descriptions[i]}"', fontsize=10, fontweight='bold')
        ax.set_aspect('equal')
        ax.grid(True, alpha=0.3)
        ax.set_xlim(-1.1, 1.1)
        ax.set_ylim(-1.1, 1.1)
        ax.legend(fontsize=8, loc='upper right')
    
    # Remove empty subplots
    for i in range(n_samples_to_show, len(axes)):
        fig.delaxes(axes[i])
    
    plt.suptitle('Multi-Agent Trajectory Samples with Spatial Descriptions', 
                fontsize=16, fontweight='bold')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'multi_agent_samples.png'), 
               dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"Multi-agent visualization saved to: {output_dir}/multi_agent_samples.png")


if __name__ == "__main__":
    # Generate multi-agent dataset
    # generate_dataset(
    #     n_samples=300000,
    #     n_points=100,
    #     output_dir="data",
    #     add_noise_prob=0.3,
    #     noise_level=0.05,
    #     min_agents=1,
    #     max_agents=5
    # ) 

    generate_dataset(
        n_samples=300000,
        n_points=100,
        output_dir="data_no_noise",
        add_noise_prob=0,
        noise_level=0.0,
        min_agents=1,
        max_agents=5
    ) 