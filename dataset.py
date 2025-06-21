"""
PyTorch Dataset for 2D Trajectory Diffusion Model
"""

import torch
import numpy as np
import json
import os
from torch.utils.data import Dataset, DataLoader
from typing import Dict, List, Optional, Tuple


class TrajectoryDataset(Dataset):
    """Dataset for 2D trajectory data."""
    
    def __init__(
        self,
        data_dir: str = "data",
        sequence_length: int = 100,
        transform: Optional[callable] = None
    ):
        """
        Initialize the trajectory dataset.
        
        Args:
            data_dir: Directory containing the dataset files
            sequence_length: Length of trajectory sequences
            transform: Optional transform to apply to trajectories
        """
        self.data_dir = data_dir
        self.sequence_length = sequence_length
        self.transform = transform
        
        # Load trajectories and labels
        self.trajectories = np.load(os.path.join(data_dir, 'trajectories.npy'))
        
        with open(os.path.join(data_dir, 'labels.json'), 'r') as f:
            self.labels = json.load(f)
        
        with open(os.path.join(data_dir, 'metadata.json'), 'r') as f:
            self.metadata = json.load(f)
        
        # Create label to index mapping
        self.unique_labels = list(set(self.labels))
        self.label_to_idx = {label: idx for idx, label in enumerate(self.unique_labels)}
        self.label_indices = [self.label_to_idx[label] for label in self.labels]
        
        print(f"Loaded dataset with {len(self.trajectories)} trajectories")
        print(f"Trajectory shape: {self.trajectories.shape}")
        print(f"Pattern types: {self.unique_labels}")
    
    def __len__(self) -> int:
        return len(self.trajectories)
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """
        Get a single trajectory sample.
        
        Returns:
            Dictionary containing:
                - 'trajectory': Trajectory data as tensor [seq_len, 2]
                - 'label': Pattern type as integer
                - 'label_name': Pattern type as string
        """
        trajectory = self.trajectories[idx]  # Shape: [seq_len, 2]
        label = self.label_indices[idx]
        label_name = self.labels[idx]
        
        # Ensure correct sequence length
        if len(trajectory) != self.sequence_length:
            # Interpolate to desired length if needed
            trajectory = self._interpolate_trajectory(trajectory, self.sequence_length)
        
        # Apply transform if provided
        if self.transform:
            trajectory = self.transform(trajectory)
        
        # Convert to torch tensors
        trajectory = torch.FloatTensor(trajectory)  # [seq_len, 2]
        label = torch.LongTensor([label])[0]  # Scalar tensor
        
        return {
            'trajectory': trajectory,
            'label': label,
            'label_name': label_name
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
    
    def get_pattern_samples(self, pattern_type: str, n_samples: int = 5) -> List[np.ndarray]:
        """Get sample trajectories for a specific pattern type."""
        indices = [i for i, label in enumerate(self.labels) if label == pattern_type]
        selected_indices = np.random.choice(indices, min(n_samples, len(indices)), replace=False)
        return [self.trajectories[i] for i in selected_indices]
    
    def get_statistics(self) -> Dict:
        """Get dataset statistics."""
        stats = {
            'total_samples': len(self.trajectories),
            'sequence_length': self.trajectories.shape[1],
            'feature_dim': self.trajectories.shape[2],
            'pattern_counts': {},
            'data_range': {
                'min': float(self.trajectories.min()),
                'max': float(self.trajectories.max()),
                'mean': float(self.trajectories.mean()),
                'std': float(self.trajectories.std())
            }
        }
        
        # Count samples per pattern
        for label in self.unique_labels:
            stats['pattern_counts'][label] = self.labels.count(label)
        
        return stats


class TrajectoryDataModule:
    """Data module for handling train/val/test splits and data loaders."""
    
    def __init__(
        self,
        data_dir: str = "data",
        sequence_length: int = 100,
        batch_size: int = 32,
        train_split: float = 0.8,
        val_split: float = 0.1,
        num_workers: int = 4,
        pin_memory: bool = True
    ):
        """
        Initialize the data module.
        
        Args:
            data_dir: Directory containing dataset
            sequence_length: Length of trajectory sequences
            batch_size: Batch size for data loaders
            train_split: Fraction of data for training
            val_split: Fraction of data for validation
            num_workers: Number of workers for data loading
            pin_memory: Whether to pin memory for faster GPU transfer
        """
        self.data_dir = data_dir
        self.sequence_length = sequence_length
        self.batch_size = batch_size
        self.train_split = train_split
        self.val_split = val_split
        self.num_workers = num_workers
        self.pin_memory = pin_memory
        
        # Load full dataset
        self.full_dataset = TrajectoryDataset(
            data_dir=data_dir,
            sequence_length=sequence_length
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
        
        print(f"Dataset splits created:")
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
            pin_memory=self.pin_memory
        )
    
    def val_dataloader(self) -> DataLoader:
        """Get validation data loader."""
        return DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory
        )
    
    def test_dataloader(self) -> DataLoader:
        """Get test data loader."""
        return DataLoader(
            self.test_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory
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


def create_data_loaders(
    data_dir: str = "data",
    batch_size: int = 32,
    sequence_length: int = 100,
    train_split: float = 0.8,
    val_split: float = 0.1,
    num_workers: int = 4
) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """
    Convenience function to create train, validation, and test data loaders.
    
    Returns:
        Tuple of (train_loader, val_loader, test_loader)
    """
    data_module = TrajectoryDataModule(
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
    # Test the dataset
    print("Testing TrajectoryDataset...")
    
    # Create data module
    data_module = TrajectoryDataModule(
        data_dir="data",
        batch_size=8,
        sequence_length=100
    )
    
    # Get sample batch
    sample_batch = data_module.get_sample_batch('train')
    
    print(f"Sample batch shapes:")
    print(f"  Trajectories: {sample_batch['trajectory'].shape}")
    print(f"  Labels: {sample_batch['label'].shape}")
    print(f"  Label names: {sample_batch['label_name']}")
    
    # Print dataset statistics
    stats = data_module.full_dataset.get_statistics()
    print(f"\nDataset statistics:")
    for key, value in stats.items():
        print(f"  {key}: {value}") 