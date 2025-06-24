"""
Simplified Multi-Agent Dataset Test (without transformers dependency)
"""

import torch
import numpy as np
import json
import os
from torch.utils.data import Dataset, DataLoader


def multi_agent_collate_fn(batch):
    """Custom collate function for multi-agent data."""
    # Separate tensor and non-tensor data
    tensor_keys = ['trajectories', 'agent_mask', 'pattern_indices', 'region_indices', 'n_agents']
    
    collated = {}
    
    for key in tensor_keys:
        collated[key] = torch.stack([item[key] for item in batch])
    
    # Handle list data separately (variable length)
    collated['pattern_names'] = [item['pattern_names'] for item in batch]
    collated['region_names'] = [item['region_names'] for item in batch]  
    collated['text_description'] = [item['text_description'] for item in batch]
    
    return collated


class SimpleMultiAgentDataset(Dataset):
    """Simplified multi-agent dataset for testing without transformers."""
    
    def __init__(self, data_dir: str = "data", sequence_length: int = 100):
        """Initialize the dataset."""
        self.data_dir = data_dir
        self.sequence_length = sequence_length
        
        print("📂 Loading multi-agent data...")
        
        # Load data files
        self.trajectories = np.load(os.path.join(data_dir, 'trajectories.npy'))
        self.agent_masks = np.load(os.path.join(data_dir, 'agent_masks.npy'))
        
        with open(os.path.join(data_dir, 'pattern_types.json'), 'r') as f:
            self.pattern_types = json.load(f)
        
        with open(os.path.join(data_dir, 'region_names.json'), 'r') as f:
            self.region_names = json.load(f)
        
        with open(os.path.join(data_dir, 'text_descriptions.json'), 'r') as f:
            self.text_descriptions = json.load(f)
        
        # Extract dataset info
        self.n_samples, self.max_agents, self.seq_len, self.coord_dim = self.trajectories.shape
        
        # Create pattern and region mappings
        all_patterns = [p for sample in self.pattern_types for p in sample]
        all_regions = [r for sample in self.region_names for r in sample]
        
        self.unique_patterns = list(set(all_patterns))
        self.unique_regions = list(set(all_regions))
        
        self.pattern_to_idx = {p: i for i, p in enumerate(self.unique_patterns)}
        self.region_to_idx = {r: i for i, r in enumerate(self.unique_regions)}
        
        print(f"✅ Loaded multi-agent dataset:")
        print(f"  • Samples: {self.n_samples:,}")
        print(f"  • Max agents: {self.max_agents}")
        print(f"  • Sequence length: {self.seq_len}")
        print(f"  • Patterns: {len(self.unique_patterns)}")
        print(f"  • Regions: {len(self.unique_regions)}")
    
    def __len__(self):
        return len(self.trajectories)
    
    def __getitem__(self, idx):
        """Get a single multi-agent sample."""
        # Get data
        trajectories = self.trajectories[idx]  # [max_agents, seq_len, 2]
        agent_mask = self.agent_masks[idx]     # [max_agents]
        patterns = self.pattern_types[idx]
        regions = self.region_names[idx]
        description = self.text_descriptions[idx]
        
        # Create pattern/region indices
        pattern_indices = np.full(self.max_agents, -1, dtype=np.int64)
        region_indices = np.full(self.max_agents, -1, dtype=np.int64)
        
        for i, (pattern, region) in enumerate(zip(patterns, regions)):
            if i < self.max_agents:
                pattern_indices[i] = self.pattern_to_idx[pattern]
                region_indices[i] = self.region_to_idx[region]
        
        # Convert to torch tensors
        return {
            'trajectories': torch.FloatTensor(trajectories),
            'agent_mask': torch.BoolTensor(agent_mask),
            'pattern_indices': torch.LongTensor(pattern_indices),
            'region_indices': torch.LongTensor(region_indices),
            'pattern_names': patterns,
            'region_names': regions,
            'text_description': description,
            'n_agents': torch.LongTensor([int(agent_mask.sum())])[0]
        }


def test_dataset_functionality():
    """Test the dataset functionality."""
    print("🧪 Testing Multi-Agent Dataset Functionality")
    print("=" * 50)
    
    # Create dataset
    dataset = SimpleMultiAgentDataset("data")
    
    # Test __getitem__
    print(f"\n🔍 Testing sample retrieval...")
    sample = dataset[0]
    
    print(f"Sample 0 structure:")
    for key, value in sample.items():
        if isinstance(value, torch.Tensor):
            print(f"  • {key}: {value.shape} {value.dtype}")
        else:
            print(f"  • {key}: {type(value)} - {value}")
    
    # Test batch loading with custom collate function
    print(f"\n📦 Testing batch loading...")
    dataloader = DataLoader(dataset, batch_size=4, shuffle=False, collate_fn=multi_agent_collate_fn)
    batch = next(iter(dataloader))
    
    print(f"Batch structure:")
    for key, value in batch.items():
        if isinstance(value, torch.Tensor):
            print(f"  • {key}: {value.shape} {value.dtype}")
        else:
            print(f"  • {key}: {type(value)} (length: {len(value)})")
    
    # Show sample data
    print(f"\n📋 Sample batch data:")
    for i in range(2):  # First 2 samples in batch
        n_agents = batch['n_agents'][i].item()
        patterns = batch['pattern_names'][i]
        regions = batch['region_names'][i]
        description = batch['text_description'][i]
        
        print(f"\n  Batch item {i+1}:")
        print(f"    Agents: {n_agents}")
        print(f"    Patterns: {patterns}")
        print(f"    Regions: {regions}")
        print(f"    Description: \"{description[:80]}{'...' if len(description) > 80 else ''}\"")
        print(f"    Active mask: {batch['agent_mask'][i]}")
        print(f"    Pattern indices: {batch['pattern_indices'][i]}")
        print(f"    Region indices: {batch['region_indices'][i]}")
    
    # Test data consistency
    print(f"\n🔍 Testing data consistency...")
    for i in range(4):  # Check all 4 items in batch
        n_agents = batch['n_agents'][i].item()
        active_agents = batch['agent_mask'][i].sum().item()
        pattern_count = len(batch['pattern_names'][i])
        region_count = len(batch['region_names'][i])
        
        print(f"  Item {i+1}: n_agents={n_agents}, active={active_agents}, patterns={pattern_count}, regions={region_count}")
        
        # Check consistency
        if n_agents == active_agents == pattern_count == region_count:
            print(f"    ✅ Consistent")
        else:
            print(f"    ❌ Inconsistent!")
    
    print(f"\n✅ Dataset functionality test completed!")
    return dataset, batch


if __name__ == "__main__":
    test_dataset_functionality() 