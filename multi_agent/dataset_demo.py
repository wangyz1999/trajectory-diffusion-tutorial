"""
Multi-Agent Dataset Demo
Demonstrates how to use the multi-agent trajectory dataset
"""

import torch
import numpy as np
import matplotlib.pyplot as plt
from simple_dataset_test import SimpleMultiAgentDataset, multi_agent_collate_fn
from torch.utils.data import DataLoader


def visualize_multi_agent_batch(batch, batch_idx=0, save_path="multi_agent_batch_demo.png"):
    """Visualize a multi-agent batch sample."""
    
    # Extract data for the specified batch item
    trajectories = batch['trajectories'][batch_idx]  # [max_agents, seq_len, 2]
    agent_mask = batch['agent_mask'][batch_idx]      # [max_agents]
    pattern_names = batch['pattern_names'][batch_idx]
    region_names = batch['region_names'][batch_idx]
    text_description = batch['text_description'][batch_idx]
    n_agents = batch['n_agents'][batch_idx].item()
    
    # Create visualization
    fig, ax = plt.subplots(1, 1, figsize=(10, 8))
    
    colors = ['blue', 'red', 'green', 'orange', 'purple']
    
    # Plot each active agent's trajectory
    for agent_idx in range(len(agent_mask)):
        if agent_mask[agent_idx]:  # Agent is active
            traj = trajectories[agent_idx].numpy()
            color = colors[agent_idx % len(colors)]
            pattern = pattern_names[agent_idx] if agent_idx < len(pattern_names) else 'unknown'
            region = region_names[agent_idx] if agent_idx < len(region_names) else 'unknown'
            
            ax.plot(traj[:, 0], traj[:, 1], color=color, linewidth=3, alpha=0.8,
                   label=f'Agent {agent_idx+1}: {pattern} ({region})')
    
    # Add spatial region markers
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
        ax.plot(x, y, 'k+', markersize=10, alpha=0.4)
        ax.text(x, y-0.08, region, ha='center', va='top', fontsize=8, alpha=0.6)
    
    ax.set_title(f'Multi-Agent Sample (Batch {batch_idx})\n"{text_description}"', 
                fontsize=12, fontweight='bold')
    ax.set_aspect('equal')
    ax.grid(True, alpha=0.3)
    ax.set_xlim(-1.1, 1.1)
    ax.set_ylim(-1.1, 1.1)
    ax.legend(fontsize=10)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"Visualization saved to: {save_path}")


def analyze_batch_structure(batch):
    """Analyze the structure and content of a multi-agent batch."""
    
    print(f"📊 Multi-Agent Batch Analysis")
    print(f"=" * 50)
    
    batch_size = batch['trajectories'].shape[0]
    max_agents = batch['trajectories'].shape[1]
    seq_len = batch['trajectories'].shape[2]
    
    print(f"Batch dimensions:")
    print(f"  • Batch size: {batch_size}")
    print(f"  • Max agents: {max_agents}")
    print(f"  • Sequence length: {seq_len}")
    print(f"  • Coordinate dimensions: {batch['trajectories'].shape[3]}")
    
    print(f"\nTensor shapes:")
    for key, value in batch.items():
        if isinstance(value, torch.Tensor):
            print(f"  • {key}: {value.shape} ({value.dtype})")
        else:
            print(f"  • {key}: List[{type(value[0]).__name__}] (length: {len(value)})")
    
    print(f"\nAgent distribution in batch:")
    for i in range(batch_size):
        n_agents = batch['n_agents'][i].item()
        active_agents = batch['agent_mask'][i].sum().item()
        patterns = batch['pattern_names'][i]
        regions = batch['region_names'][i]
        
        print(f"  Sample {i+1}: {n_agents} agents ({active_agents} active)")
        print(f"    Patterns: {patterns}")
        print(f"    Regions: {regions}")
        print(f"    Description: \"{batch['text_description'][i][:60]}...\"")
    
    # Data quality checks
    print(f"\nData quality checks:")
    
    # Check trajectory bounds
    active_trajectories = batch['trajectories'][batch['agent_mask']]
    print(f"  • Trajectory bounds: [{active_trajectories.min():.3f}, {active_trajectories.max():.3f}]")
    
    # Check for NaN values
    has_nan = torch.isnan(active_trajectories).any()
    print(f"  • Contains NaN: {'Yes' if has_nan else 'No'}")
    
    # Check agent consistency
    consistent = True
    for i in range(batch_size):
        n_agents = batch['n_agents'][i].item()
        active_agents = batch['agent_mask'][i].sum().item()
        pattern_count = len(batch['pattern_names'][i])
        region_count = len(batch['region_names'][i])
        
        if not (n_agents == active_agents == pattern_count == region_count):
            consistent = False
            break
    
    print(f"  • Agent count consistency: {'✅ Consistent' if consistent else '❌ Inconsistent'}")


def demo_multi_agent_dataset():
    """Complete demo of multi-agent dataset functionality."""
    
    print("🎯 Multi-Agent Trajectory Dataset Demo")
    print("=" * 60)
    
    # 1. Load dataset
    print("\n1️⃣ Loading Dataset")
    dataset = SimpleMultiAgentDataset("data")
    
    # 2. Create data loader
    print("\n2️⃣ Creating DataLoader")
    dataloader = DataLoader(
        dataset, 
        batch_size=6, 
        shuffle=True,
        collate_fn=multi_agent_collate_fn
    )
    
    # 3. Get a batch
    print("\n3️⃣ Getting Sample Batch")
    batch = next(iter(dataloader))
    
    # 4. Analyze batch structure
    print("\n4️⃣ Analyzing Batch Structure")
    analyze_batch_structure(batch)
    
    # 5. Visualize samples
    print("\n5️⃣ Creating Visualizations")
    
    # Visualize first 3 samples in the batch
    for i in range(min(3, batch['trajectories'].shape[0])):
        visualize_multi_agent_batch(batch, batch_idx=i, 
                                   save_path=f"demo_sample_{i+1}.png")
    
    # 6. Dataset statistics
    print(f"\n6️⃣ Dataset Usage Examples")
    
    print(f"\nExample 1: Accessing individual samples")
    sample = dataset[42]  # Get sample 42
    print(f"  Sample 42 has {sample['n_agents'].item()} agents:")
    print(f"    Patterns: {sample['pattern_names']}")
    print(f"    Regions: {sample['region_names']}")
    print(f"    Description: \"{sample['text_description'][:50]}...\"")
    
    print(f"\nExample 2: Filtering by pattern")
    circle_samples = []
    for i in range(min(100, len(dataset))):  # Check first 100 samples
        sample = dataset[i]
        if 'circle' in sample['pattern_names']:
            circle_samples.append(i)
    
    print(f"  Found {len(circle_samples)} samples with circles in first 100 samples")
    if circle_samples:
        sample_idx = circle_samples[0]
        sample = dataset[sample_idx]
        print(f"    Sample {sample_idx}: {sample['pattern_names']} at {sample['region_names']}")
    
    print(f"\nExample 3: Batch processing")
    total_agents = 0
    pattern_counts = {}
    
    for i, batch in enumerate(dataloader):
        if i >= 5:  # Process first 5 batches
            break
            
        # Count total agents
        total_agents += batch['n_agents'].sum().item()
        
        # Count patterns
        for pattern_list in batch['pattern_names']:
            for pattern in pattern_list:
                pattern_counts[pattern] = pattern_counts.get(pattern, 0) + 1
    
    print(f"  Processed 5 batches:")
    print(f"    Total agents: {total_agents}")
    print(f"    Pattern distribution: {pattern_counts}")
    
    print(f"\n🎉 Multi-Agent Dataset Demo Completed!")
    print(f"✅ Dataset is ready for multi-agent trajectory diffusion model training!")


if __name__ == "__main__":
    demo_multi_agent_dataset() 