"""
PyTorch Dataset for Multi-Agent Text-Conditioned 2D Trajectory Diffusion Model
"""

import torch
import numpy as np
import json
import os
from torch.utils.data import Dataset, DataLoader
from typing import Dict, List, Optional, Tuple
from transformers import AutoTokenizer, AutoModel
import torch.nn.functional as F
from tqdm import tqdm
from sentence_transformers import SentenceTransformer


def multi_agent_collate_fn(batch):
    """Custom collate function for multi-agent data with variable-length lists."""
    # Separate tensor and non-tensor data
    tensor_keys = ['trajectories', 'agent_mask', 'pattern_indices', 'region_indices', 'n_agents']
    
    collated = {}
    
    # Stack tensor data
    for key in tensor_keys:
        collated[key] = torch.stack([item[key] for item in batch])
    
    # Handle list data separately (variable length)
    collated['pattern_names'] = [item['pattern_names'] for item in batch]
    collated['region_names'] = [item['region_names'] for item in batch]  
    collated['text_description'] = [item['text_description'] for item in batch]
    
    # Text embeddings need special handling
    collated['text_embedding'] = torch.stack([item['text_embedding'] for item in batch])
    
    return collated


class TextEncoder:
    """Simple text encoder using pre-trained transformer model."""
    
    def __init__(self, model_name: str = "sentence-transformers/all-MiniLM-L6-v2", device: str = "cuda"):
        """
        Initialize text encoder.
        
        Args:
            model_name: Pre-trained model name for text encoding
            device: Device to run the model on
        """
        # Check if CUDA is available
        if device == "cuda" and not torch.cuda.is_available():
            print("⚠️  CUDA not available, falling back to CPU")
            device = "cpu"
        
        self.device = device
        self.encoder = SentenceTransformer(model_name)
        self.encoder.to(device)
        self.embedding_dim = self.encoder.get_sentence_embedding_dimension()
        print(f"✅ Using SentenceTransformer: {model_name}, embedding dim: {self.embedding_dim}, device: {device}")

    def encode(self, texts: List[str], batch_size: int = 32, show_progress: bool = False) -> torch.Tensor:
        """
        Encode list of texts into embeddings with optional progress bar.
        
        Args:
            texts: List of text descriptions
            batch_size: Batch size for encoding (only used with SentenceTransformer)
            show_progress: Whether to show progress bar
            
        Returns:
            Text embeddings tensor [batch_size, embedding_dim]
        """
        # Use SentenceTransformer with batched encoding
        if show_progress and len(texts) > 100:
            embeddings = []
            
            # Create progress bar
            pbar = tqdm(range(0, len(texts), batch_size), 
                        desc="🔤 Encoding text", unit="batch", colour='cyan')
            
            for i in pbar:
                batch_texts = texts[i:i + batch_size]
                batch_embeddings = self.encoder.encode(
                    batch_texts, 
                    convert_to_tensor=True, 
                    device=self.device,
                    show_progress_bar=False  # Disable internal progress bar
                )
                embeddings.append(batch_embeddings)
                
                # Update progress bar
                pbar.set_postfix({
                    'texts': f'{min(i + batch_size, len(texts))}/{len(texts)}',
                    'batch_size': len(batch_texts)
                })
            
            pbar.close()
            
            # Concatenate all embeddings
            embeddings = torch.cat(embeddings, dim=0)
        else:
            # Single batch encoding for smaller datasets
            embeddings = self.encoder.encode(texts, convert_to_tensor=True, device=self.device)
        
        return embeddings.to(self.device)
    
    def encode_for_dataset(self, texts: List[str], batch_size: int = 32, show_progress: bool = False) -> torch.Tensor:
        """
        Encode texts for dataset storage (returns CPU tensors for DataLoader compatibility).
        
        Args:
            texts: List of text descriptions
            batch_size: Batch size for encoding
            show_progress: Whether to show progress bar
            
        Returns:
            Text embeddings tensor on CPU [batch_size, embedding_dim]
        """
        embeddings = self.encode(texts, batch_size, show_progress)
        return embeddings.cpu()  # Always return CPU tensors for dataset storage
    

class MultiAgentTrajectoryDataset(Dataset):
    """Dataset for multi-agent text-conditioned 2D trajectory data."""
    
    def __init__(
        self,
        data_dir: str = "data",
        sequence_length: int = 100,
        transform: Optional[callable] = None,
        text_encoder: Optional[TextEncoder] = None
    ):
        """
        Initialize the multi-agent trajectory dataset.
        
        Args:
            data_dir: Directory containing the dataset files
            sequence_length: Length of trajectory sequences
            transform: Optional transform to apply to trajectories
            text_encoder: Text encoder for processing descriptions
        """
        self.data_dir = data_dir
        self.sequence_length = sequence_length
        self.transform = transform
        
        # Initialize text encoder
        if text_encoder is None:
            self.text_encoder = TextEncoder()
        else:
            self.text_encoder = text_encoder
        
        # Load multi-agent trajectories and masks
        print("📂 Loading multi-agent trajectories and masks...")
        self.trajectories = np.load(os.path.join(data_dir, 'trajectories.npy'))  # [N, max_agents, seq_len, 2]
        self.agent_masks = np.load(os.path.join(data_dir, 'agent_masks.npy'))    # [N, max_agents]
        
        # Load pattern types and regions (list of lists)
        with open(os.path.join(data_dir, 'pattern_types.json'), 'r') as f:
            self.pattern_types = json.load(f)  # List[List[str]]
        
        with open(os.path.join(data_dir, 'region_names.json'), 'r') as f:
            self.region_names = json.load(f)   # List[List[str]]
        
        # Load text descriptions
        text_file = os.path.join(data_dir, 'text_descriptions.json')
        if os.path.exists(text_file):
            print("📝 Loading multi-agent text descriptions...")
            with open(text_file, 'r') as f:
                self.text_descriptions = json.load(f)
            self.is_text_conditioned = True
            print(f"✅ Loaded text-conditioned dataset with {len(self.text_descriptions):,} descriptions")
        else:
            # Fallback for datasets without text
            print("⚠️  No text descriptions found, creating simple descriptions...")
            self.text_descriptions = [self._create_simple_description(i) for i in range(len(self.trajectories))]
            self.is_text_conditioned = False
            print("✅ Created simple descriptions")
        
        # Load metadata
        metadata_file = os.path.join(data_dir, 'metadata.json')
        if os.path.exists(metadata_file):
            with open(metadata_file, 'r') as f:
                self.metadata = json.load(f)
        else:
            self.metadata = {}
        
        # Extract dataset information
        self.n_samples, self.max_agents, self.seq_len, self.coord_dim = self.trajectories.shape
        
        # Create unique pattern and region mappings
        self.unique_patterns = list(set([pattern for sample_patterns in self.pattern_types for pattern in sample_patterns]))
        self.unique_regions = list(set([region for sample_regions in self.region_names for region in sample_regions]))
        
        self.pattern_to_idx = {pattern: idx for idx, pattern in enumerate(self.unique_patterns)}
        self.region_to_idx = {region: idx for idx, region in enumerate(self.unique_regions)}
        
        # Pre-encode all text descriptions for efficiency
        print(f"🔤 Encoding {len(self.text_descriptions):,} multi-agent text descriptions...")
        self.text_embeddings = self.text_encoder.encode_for_dataset(
            self.text_descriptions, 
            batch_size=1024,  # Reasonable batch size for encoding
            show_progress=True
        )
        print("📱 Multi-agent text embeddings stored on CPU for DataLoader compatibility")
        
        print(f"✅ Multi-agent dataset loaded successfully!")
        print(f"📊 Dataset statistics:")
        print(f"  • Trajectory samples: {len(self.trajectories):,}")
        print(f"  • Trajectory shape: {self.trajectories.shape}")
        print(f"  • Max agents per sample: {self.max_agents}")
        print(f"  • Active agents: {self.agent_masks.sum():,} / {self.agent_masks.size:,} ({self.agent_masks.sum()/self.agent_masks.size*100:.1f}%)")
        print(f"  • Average agents per sample: {self.agent_masks.sum(axis=1).mean():.2f}")
        print(f"  • Text embedding shape: {self.text_embeddings.shape}")
        print(f"  • Unique patterns: {len(self.unique_patterns)} ({', '.join(self.unique_patterns)})")
        print(f"  • Unique regions: {len(self.unique_regions)} ({', '.join(self.unique_regions)})")
    
    def _create_simple_description(self, idx: int) -> str:
        """Create a simple text description for samples without pre-existing descriptions."""
        patterns = self.pattern_types[idx]
        regions = self.region_names[idx]
        
        if len(patterns) == 1:
            return f"A {patterns[0]} at {regions[0]}"
        elif len(patterns) == 2:
            return f"A {patterns[0]} at {regions[0]} and a {patterns[1]} at {regions[1]}"
        else:
            pattern_region_pairs = [f"{p} at {r}" for p, r in zip(patterns, regions)]
            return f"Multiple patterns: {', '.join(pattern_region_pairs)}"
    
    def __len__(self) -> int:
        return len(self.trajectories)
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """
        Get a single multi-agent trajectory sample with text conditioning.
        
        Returns:
            Dictionary containing:
                - 'trajectories': Multi-agent trajectory data [max_agents, seq_len, 2]
                - 'agent_mask': Boolean mask for active agents [max_agents]
                - 'pattern_indices': Pattern type indices for each agent [max_agents]
                - 'region_indices': Region indices for each agent [max_agents]
                - 'pattern_names': Pattern type names for active agents
                - 'region_names': Region names for active agents
                - 'text_embedding': Text embedding [embedding_dim]
                - 'text_description': Original text description
                - 'n_agents': Number of active agents (scalar)
        """
        # Get multi-agent trajectory data
        trajectories = self.trajectories[idx]  # [max_agents, seq_len, 2]
        agent_mask = self.agent_masks[idx]     # [max_agents]
        
        # Get pattern and region information
        sample_patterns = self.pattern_types[idx]
        sample_regions = self.region_names[idx]
        text_description = self.text_descriptions[idx]
        text_embedding = self.text_embeddings[idx]  # Pre-computed embedding
        
        # Create padded arrays for pattern and region indices
        pattern_indices = np.full(self.max_agents, -1, dtype=np.int64)  # -1 for inactive agents
        region_indices = np.full(self.max_agents, -1, dtype=np.int64)
        
        # Fill indices for active agents
        for i, (pattern, region) in enumerate(zip(sample_patterns, sample_regions)):
            if i < self.max_agents:
                pattern_indices[i] = self.pattern_to_idx[pattern]
                region_indices[i] = self.region_to_idx[region]
        
        # Ensure correct sequence length for all agents
        if trajectories.shape[1] != self.sequence_length:
            # Interpolate each agent's trajectory
            new_trajectories = np.zeros((self.max_agents, self.sequence_length, 2))
            for agent_idx in range(self.max_agents):
                if agent_mask[agent_idx]:  # Only interpolate active agents
                    new_trajectories[agent_idx] = self._interpolate_trajectory(
                        trajectories[agent_idx], self.sequence_length
                    )
            trajectories = new_trajectories
        
        # Apply transform if provided
        if self.transform:
            # Apply transform to each agent's trajectory
            for agent_idx in range(self.max_agents):
                if agent_mask[agent_idx]:
                    trajectories[agent_idx] = self.transform(trajectories[agent_idx])
        
        # Convert to torch tensors
        trajectories = torch.FloatTensor(trajectories)  # [max_agents, seq_len, 2]
        agent_mask = torch.BoolTensor(agent_mask)       # [max_agents]
        pattern_indices = torch.LongTensor(pattern_indices)  # [max_agents]
        region_indices = torch.LongTensor(region_indices)    # [max_agents]
        n_agents = torch.LongTensor([int(agent_mask.sum())])[0]  # Scalar tensor
        
        return {
            'trajectories': trajectories,
            'agent_mask': agent_mask,
            'pattern_indices': pattern_indices,
            'region_indices': region_indices,
            'pattern_names': sample_patterns,
            'region_names': sample_regions,
            'text_embedding': text_embedding.clone(),  # Clone to avoid reference issues
            'text_description': text_description,
            'n_agents': n_agents
        }
    
    def _interpolate_trajectory(self, trajectory: np.ndarray, target_length: int) -> np.ndarray:
        """Interpolate trajectory to target length."""
        current_length = len(trajectory)
        if current_length == target_length:
            return trajectory
        
        # Create interpolation indices
        old_indices = np.linspace(0, current_length - 1, current_length)
        new_indices = np.linspace(0, current_length - 1, target_length)
        
        # Interpolate x and y coordinates separately
        x_new = np.interp(new_indices, old_indices, trajectory[:, 0])
        y_new = np.interp(new_indices, old_indices, trajectory[:, 1])
        
        return np.column_stack([x_new, y_new])
    
    def get_pattern_samples(self, pattern_type: str, n_samples: int = 5) -> List[Dict]:
        """Get sample trajectories for a specific pattern type."""
        matching_samples = []
        
        for i, sample_patterns in enumerate(self.pattern_types):
            if pattern_type in sample_patterns:
                # Find which agent has this pattern
                agent_idx = sample_patterns.index(pattern_type)
                if self.agent_masks[i][agent_idx]:  # Ensure agent is active
                    matching_samples.append({
                        'sample_idx': i,
                        'agent_idx': agent_idx,
                        'trajectory': self.trajectories[i, agent_idx],
                        'pattern': pattern_type,
                        'region': self.region_names[i][agent_idx],
                        'text_description': self.text_descriptions[i]
                    })
        
        # Randomly sample
        if len(matching_samples) > n_samples:
            indices = np.random.choice(len(matching_samples), n_samples, replace=False)
            matching_samples = [matching_samples[i] for i in indices]
        
        return matching_samples
    
    def get_text_samples(self, pattern_type: str, n_samples: int = 5) -> List[str]:
        """Get sample text descriptions for samples containing a specific pattern type."""
        matching_descriptions = []
        
        for i, sample_patterns in enumerate(self.pattern_types):
            if pattern_type in sample_patterns:
                matching_descriptions.append(self.text_descriptions[i])
        
        # Randomly sample
        if len(matching_descriptions) > n_samples:
            indices = np.random.choice(len(matching_descriptions), n_samples, replace=False)
            matching_descriptions = [matching_descriptions[i] for i in indices]
        
        return matching_descriptions
    
    def get_statistics(self) -> Dict:
        """Get multi-agent dataset statistics."""
        # Count active agents per sample
        agents_per_sample = self.agent_masks.sum(axis=1)
        
        # Count pattern occurrences
        pattern_counts = {}
        for sample_patterns in self.pattern_types:
            for pattern in sample_patterns:
                pattern_counts[pattern] = pattern_counts.get(pattern, 0) + 1
        
        # Count region occurrences
        region_counts = {}
        for sample_regions in self.region_names:
            for region in sample_regions:
                region_counts[region] = region_counts.get(region, 0) + 1
        
        # Agent distribution
        agent_distribution = {}
        for n in range(1, self.max_agents + 1):
            agent_distribution[f'{n}_agents'] = int(np.sum(agents_per_sample == n))
        
        stats = {
            'total_samples': len(self.trajectories),
            'max_agents': self.max_agents,
            'sequence_length': self.seq_len,
            'coordinate_dim': self.coord_dim,
            'text_embedding_dim': self.text_embeddings.shape[1],
            'is_text_conditioned': self.is_text_conditioned,
            'multi_agent': True,
            'avg_agents_per_sample': float(agents_per_sample.mean()),
            'agent_distribution': agent_distribution,
            'pattern_counts': pattern_counts,
            'region_counts': region_counts,
            'unique_patterns': self.unique_patterns,
            'unique_regions': self.unique_regions,
            'data_range': {
                'min': float(self.trajectories.min()),
                'max': float(self.trajectories.max()),
                'mean': float(self.trajectories.mean()),
                'std': float(self.trajectories.std())
            }
        }
        
        # Sample text descriptions for each pattern
        stats['sample_text_descriptions'] = {}
        for pattern in self.unique_patterns[:5]:  # First 5 pattern types
            stats['sample_text_descriptions'][pattern] = self.get_text_samples(pattern, 3)
        
        return stats


# Backward compatibility alias
TrajectoryDataset = MultiAgentTrajectoryDataset


class MultiAgentTrajectoryDataModule:
    """Data module for handling multi-agent train/val/test splits and data loaders."""
    
    def __init__(
        self,
        data_dir: str = "data",
        sequence_length: int = 100,
        batch_size: int = 32,
        train_split: float = 0.8,
        val_split: float = 0.1,
        num_workers: int = 4,
        pin_memory: bool = True,
        text_encoder: Optional[TextEncoder] = None
    ):
        """
        Initialize the multi-agent data module.
        
        Args:
            data_dir: Directory containing dataset
            sequence_length: Length of trajectory sequences
            batch_size: Batch size for data loaders
            train_split: Fraction of data for training
            val_split: Fraction of data for validation
            num_workers: Number of workers for data loading
            pin_memory: Whether to pin memory for faster GPU transfer
            text_encoder: Shared text encoder for all datasets
        """
        self.data_dir = data_dir
        self.sequence_length = sequence_length
        self.batch_size = batch_size
        self.train_split = train_split
        self.val_split = val_split
        self.num_workers = num_workers
        self.pin_memory = pin_memory
        
        # Create shared text encoder
        if text_encoder is None:
            self.text_encoder = TextEncoder()
        else:
            self.text_encoder = text_encoder
        
        # Load full dataset
        self.full_dataset = MultiAgentTrajectoryDataset(
            data_dir=data_dir,
            sequence_length=sequence_length,
            text_encoder=self.text_encoder
        )
        
        # Create splits
        self._create_splits()
    
    def _create_splits(self):
        """Create train/validation/test splits."""
        total_size = len(self.full_dataset)
        train_size = int(self.train_split * total_size)
        val_size = int(self.val_split * total_size)
        test_size = total_size - train_size - val_size
        
        # Create random split
        indices = torch.randperm(total_size)
        train_indices = indices[:train_size]
        val_indices = indices[train_size:train_size + val_size]
        test_indices = indices[train_size + val_size:]
        
        # Create subset datasets
        self.train_dataset = torch.utils.data.Subset(self.full_dataset, train_indices)
        self.val_dataset = torch.utils.data.Subset(self.full_dataset, val_indices)
        self.test_dataset = torch.utils.data.Subset(self.full_dataset, test_indices)
        
        print(f"Multi-agent dataset splits created:")
        print(f"  Train: {len(self.train_dataset)} samples")
        print(f"  Validation: {len(self.val_dataset)} samples")
        print(f"  Test: {len(self.test_dataset)} samples")
    
    def train_dataloader(self) -> DataLoader:
        """Get training data loader."""
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            collate_fn=multi_agent_collate_fn
        )
    
    def val_dataloader(self) -> DataLoader:
        """Get validation data loader."""
        return DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            collate_fn=multi_agent_collate_fn
        )
    
    def test_dataloader(self) -> DataLoader:
        """Get test data loader."""
        return DataLoader(
            self.test_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            collate_fn=multi_agent_collate_fn
        )
    
    def get_sample_batch(self, split: str = 'train') -> Dict[str, torch.Tensor]:
        """Get a sample batch for inspection."""
        if split == 'train':
            loader = self.train_dataloader()
        elif split == 'val':
            loader = self.val_dataloader()
        else:
            loader = self.test_dataloader()
        
        return next(iter(loader))
    
    def encode_text(self, text_descriptions: List[str]) -> torch.Tensor:
        """Encode new text descriptions using the shared encoder."""
        return self.text_encoder.encode(text_descriptions)


# Backward compatibility alias
TrajectoryDataModule = MultiAgentTrajectoryDataModule


def create_data_loaders(
    data_dir: str = "data",
    batch_size: int = 32,
    sequence_length: int = 100,
    train_split: float = 0.8,
    val_split: float = 0.1,
    num_workers: int = 4
) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """
    Convenience function to create multi-agent train, validation, and test data loaders.
    
    Returns:
        Tuple of (train_loader, val_loader, test_loader)
    """
    data_module = MultiAgentTrajectoryDataModule(
        data_dir=data_dir,
        sequence_length=sequence_length,
        batch_size=batch_size,
        train_split=train_split,
        val_split=val_split,
        num_workers=num_workers
    )
    
    return (
        data_module.train_dataloader(),
        data_module.val_dataloader(),
        data_module.test_dataloader()
    )


if __name__ == "__main__":
    # Test the multi-agent text-conditioned dataset
    print("Testing Multi-Agent Text-Conditioned TrajectoryDataset...")
    
    # Create data module
    data_module = MultiAgentTrajectoryDataModule(
        data_dir="data",
        batch_size=4,
        sequence_length=100,
        num_workers=0  # Disable multiprocessing to avoid CUDA issues
    )
    
    # Get sample batch
    sample_batch = data_module.get_sample_batch('train')
    
    print(f"\nSample batch shapes:")
    print(f"  Trajectories: {sample_batch['trajectories'].shape}")  # [batch, max_agents, seq_len, 2]
    print(f"  Agent masks: {sample_batch['agent_mask'].shape}")      # [batch, max_agents]
    print(f"  Pattern indices: {sample_batch['pattern_indices'].shape}")  # [batch, max_agents]
    print(f"  Region indices: {sample_batch['region_indices'].shape}")    # [batch, max_agents]
    print(f"  Text embeddings: {sample_batch['text_embedding'].shape}")   # [batch, embedding_dim]
    print(f"  N agents: {sample_batch['n_agents'].shape}")                # [batch]
    
    print(f"\nSample data:")
    for i in range(2):  # Show first 2 samples
        n_agents = sample_batch['n_agents'][i].item()
        patterns = sample_batch['pattern_names'][i]
        regions = sample_batch['region_names'][i]
        description = sample_batch['text_description'][i]
        
        print(f"  Sample {i+1}: {n_agents} agents")
        print(f"    Patterns: {patterns}")
        print(f"    Regions: {regions}")
        print(f"    Description: \"{description}\"")
        print(f"    Active agents: {sample_batch['agent_mask'][i].sum().item()}")
    
    # Print dataset statistics
    stats = data_module.full_dataset.get_statistics()
    print(f"\nMulti-agent dataset statistics:")
    for key, value in stats.items():
        if key not in ['sample_text_descriptions', 'pattern_counts', 'region_counts']:
            print(f"  {key}: {value}")
    
    print(f"\nPattern distribution:")
    for pattern, count in stats['pattern_counts'].items():
        print(f"  {pattern}: {count}")
    
    print(f"\nAgent distribution:")
    for agents, count in stats['agent_distribution'].items():
        print(f"  {agents}: {count}") 