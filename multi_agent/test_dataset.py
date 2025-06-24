"""
Simple test script for multi-agent dataset loading
"""

import numpy as np
import json
import os

def test_multi_agent_data():
    """Test loading and inspecting the multi-agent dataset structure."""
    
    data_dir = "data"
    
    print("🔍 Testing Multi-Agent Dataset Structure")
    print("=" * 50)
    
    # Load data files
    try:
        print("Loading trajectory data...")
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
        
        print("✅ All data files loaded successfully!")
        
    except Exception as e:
        print(f"❌ Error loading data: {e}")
        return
    
    # Print data structure information
    print(f"\n📊 Data Structure Analysis:")
    print(f"  • Trajectories shape: {trajectories.shape}")
    print(f"  • Agent masks shape: {agent_masks.shape}")
    print(f"  • Number of samples: {len(pattern_types)}")
    print(f"  • Number of text descriptions: {len(text_descriptions)}")
    
    # Analyze first few samples
    print(f"\n🔍 Sample Analysis:")
    for i in range(min(5, len(pattern_types))):
        n_agents = len(pattern_types[i])
        active_agents = int(agent_masks[i].sum())
        patterns = pattern_types[i]
        regions = region_names[i]
        description = text_descriptions[i]
        
        print(f"\n  Sample {i+1}:")
        print(f"    Agents: {n_agents} (active: {active_agents})")
        print(f"    Patterns: {patterns}")
        print(f"    Regions: {regions}")
        print(f"    Description: \"{description[:100]}{'...' if len(description) > 100 else ''}\"")
        print(f"    Trajectory shape: {trajectories[i].shape}")
        print(f"    Active mask: {agent_masks[i]}")
    
    # Dataset statistics
    print(f"\n📈 Dataset Statistics:")
    print(f"  • Total samples: {len(trajectories)}")
    print(f"  • Max agents per sample: {trajectories.shape[1]}")
    print(f"  • Sequence length: {trajectories.shape[2]}")
    print(f"  • Coordinate dimensions: {trajectories.shape[3]}")
    print(f"  • Total active agents: {agent_masks.sum()}")
    print(f"  • Agent utilization: {agent_masks.sum() / agent_masks.size:.1%}")
    
    # Pattern and region statistics
    all_patterns = [pattern for sample in pattern_types for pattern in sample]
    all_regions = [region for sample in region_names for region in sample]
    
    unique_patterns = list(set(all_patterns))
    unique_regions = list(set(all_regions))
    
    print(f"\n🎨 Pattern & Region Statistics:")
    print(f"  • Unique patterns: {len(unique_patterns)} ({', '.join(unique_patterns)})")
    print(f"  • Unique regions: {len(unique_regions)} ({', '.join(unique_regions)})")
    
    # Agent distribution
    agents_per_sample = agent_masks.sum(axis=1)
    print(f"\n👥 Agent Distribution:")
    for n in range(1, trajectories.shape[1] + 1):
        count = int(np.sum(agents_per_sample == n))
        if count > 0:
            print(f"  • {n} agents: {count} samples ({count/len(trajectories)*100:.1f}%)")
    
    # Data quality checks
    print(f"\n✅ Data Quality Checks:")
    
    # Check trajectory bounds
    active_trajectories = trajectories[agent_masks]
    min_vals = active_trajectories.min()
    max_vals = active_trajectories.max()
    print(f"  • Trajectory bounds: [{min_vals:.3f}, {max_vals:.3f}]")
    
    # Check for NaN values
    has_nan = np.isnan(active_trajectories).any()
    print(f"  • Contains NaN values: {'Yes' if has_nan else 'No'}")
    
    # Check consistency between masks and pattern lists
    mask_agent_counts = agent_masks.sum(axis=1)
    pattern_agent_counts = np.array([len(patterns) for patterns in pattern_types])
    consistency_check = np.all(mask_agent_counts == pattern_agent_counts)
    print(f"  • Agent mask consistency: {'✅ Consistent' if consistency_check else '❌ Inconsistent'}")
    
    # Check description variety
    unique_descriptions = len(set(text_descriptions))
    print(f"  • Description variety: {unique_descriptions}/{len(text_descriptions)} unique ({unique_descriptions/len(text_descriptions)*100:.1f}%)")
    
    print(f"\n🎉 Multi-agent dataset structure test completed!")
    
    return {
        'trajectories': trajectories,
        'agent_masks': agent_masks,
        'pattern_types': pattern_types,
        'region_names': region_names,
        'text_descriptions': text_descriptions,
        'metadata': metadata
    }


if __name__ == "__main__":
    test_multi_agent_data() 