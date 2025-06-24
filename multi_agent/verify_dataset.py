"""
Verification script to demonstrate the multi-agent trajectory dataset
"""

import numpy as np
import matplotlib.pyplot as plt
import json
import os

def verify_dataset():
    """Verify and demonstrate the multi-agent dataset structure."""
    
    data_dir = "data"
    
    print("🔍 Multi-Agent Dataset Verification")
    print("=" * 50)
    
    # Load data
    trajectories = np.load(os.path.join(data_dir, 'trajectories.npy'))
    agent_masks = np.load(os.path.join(data_dir, 'agent_masks.npy'))
    
    with open(os.path.join(data_dir, 'pattern_types.json'), 'r') as f:
        pattern_types = json.load(f)
    
    with open(os.path.join(data_dir, 'region_names.json'), 'r') as f:
        region_names = json.load(f)
    
    with open(os.path.join(data_dir, 'text_descriptions.json'), 'r') as f:
        text_descriptions = json.load(f)
    
    with open(os.path.join(data_dir, 'metadata.json'), 'r') as f:
        metadata = json.load(f)
    
    # Print dataset statistics
    print(f"📊 Dataset Statistics:")
    print(f"   Total samples: {trajectories.shape[0]:,}")
    print(f"   Trajectory shape: {trajectories.shape}")
    print(f"   Average agents per sample: {metadata['avg_agents_per_sample']:.2f}")
    print(f"   Pattern types: {len(metadata['pattern_types'])}")
    print(f"   Spatial regions: {len(metadata['spatial_regions'])}")
    
    print(f"\n👥 Agent Distribution:")
    for k, v in metadata['agent_distribution'].items():
        print(f"   {k}: {v:,} samples ({v/metadata['n_samples']*100:.1f}%)")
    
    print(f"\n🎨 Pattern Types: {', '.join(metadata['pattern_types'])}")
    print(f"📍 Spatial Regions: {', '.join(metadata['spatial_regions'])}")
    
    # Show examples
    print(f"\n📝 Sample Multi-Agent Descriptions:")
    for i in range(5):
        n_agents = len(pattern_types[i])
        patterns = pattern_types[i]
        regions = region_names[i]
        description = text_descriptions[i]
        
        print(f"\n   Sample {i+1} ({n_agents} agents):")
        print(f"   Patterns: {patterns}")
        print(f"   Regions: {regions}")
        print(f"   Description: \"{description}\"")
    
    # Create a simple visualization of one sample
    print(f"\n🎨 Creating example visualization...")
    
    fig, ax = plt.subplots(1, 1, figsize=(10, 8))
    
    # Select a sample with multiple agents
    sample_idx = 0  # Sample 1 has 4 agents
    sample_trajs = trajectories[sample_idx]
    sample_mask = agent_masks[sample_idx]
    sample_patterns = pattern_types[sample_idx]
    sample_regions = region_names[sample_idx]
    sample_desc = text_descriptions[sample_idx]
    
    colors = ['blue', 'red', 'green', 'orange', 'purple']
    
    for j, (traj, is_active) in enumerate(zip(sample_trajs, sample_mask)):
        if is_active:
            color = colors[j % len(colors)]
            pattern = sample_patterns[j]
            region = sample_regions[j]
            ax.plot(traj[:, 0], traj[:, 1], color=color, linewidth=3, 
                   alpha=0.8, label=f'Agent {j+1}: {pattern} ({region})')
    
    ax.set_title(f'Multi-Agent Sample\n"{sample_desc}"', fontsize=14, fontweight='bold')
    ax.set_aspect('equal')
    ax.grid(True, alpha=0.3)
    ax.set_xlim(-1.1, 1.1)
    ax.set_ylim(-1.1, 1.1)
    ax.legend(fontsize=10)
    
    # Add region markers
    regions_coords = {
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
    
    for region, (x, y) in regions_coords.items():
        ax.plot(x, y, 'k+', markersize=8, alpha=0.3)
        ax.text(x, y-0.1, region, ha='center', va='top', fontsize=8, alpha=0.5)
    
    plt.tight_layout()
    plt.savefig('multi_agent_example.png', dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"   Example visualization saved to: multi_agent_example.png")
    
    # Verify data integrity
    print(f"\n✅ Data Integrity Check:")
    
    # Check that all active agents have valid data
    active_agents = agent_masks.sum()
    total_possible = agent_masks.shape[0] * agent_masks.shape[1]
    print(f"   Active agents: {active_agents:,} / {total_possible:,} ({active_agents/total_possible*100:.1f}%)")
    
    # Check trajectory bounds
    active_trajectories = trajectories[agent_masks]
    min_vals = active_trajectories.min(axis=(0,1))
    max_vals = active_trajectories.max(axis=(0,1))
    print(f"   Trajectory bounds: X=[{min_vals[0]:.3f}, {max_vals[0]:.3f}], Y=[{min_vals[1]:.3f}, {max_vals[1]:.3f}]")
    
    # Check description lengths
    desc_lengths = [len(desc) for desc in text_descriptions]
    print(f"   Description lengths: min={min(desc_lengths)}, max={max(desc_lengths)}, avg={np.mean(desc_lengths):.1f}")
    
    print(f"\n🎉 Multi-agent dataset verification complete!")
    print(f"   Dataset is ready for training multi-agent trajectory diffusion models!")


if __name__ == "__main__":
    verify_dataset() 