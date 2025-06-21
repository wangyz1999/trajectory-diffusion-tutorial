"""
PyTorch Dataset for Text-Conditioned 2D Trajectory Diffusion Model
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
    

class TrajectoryDataset(Dataset):
    """Dataset for text-conditioned 2D trajectory data."""
    
    def __init__(
        self,
        data_dir: str = "data",
        sequence_length: int = 100,
        transform: Optional[callable] = None,
        text_encoder: Optional[TextEncoder] = None
    ):
        """
        Initialize the trajectory dataset.
        
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
        
        # Load trajectories and labels
        print("📂 Loading trajectories and labels...")
        self.trajectories = np.load(os.path.join(data_dir, 'trajectories.npy'))
        
        with open(os.path.join(data_dir, 'labels.json'), 'r') as f:
            self.labels = json.load(f)
        
        # Load text descriptions
        text_file = os.path.join(data_dir, 'text_descriptions.json')
        if os.path.exists(text_file):
            print("📝 Loading text descriptions...")
            with open(text_file, 'r') as f:
                self.text_descriptions = json.load(f)
            self.is_text_conditioned = True
            print(f"✅ Loaded text-conditioned dataset with {len(self.text_descriptions):,} descriptions")
        else:
            # Fallback for old datasets without text
            print("⚠️  No text descriptions found, creating simple descriptions...")
            self.text_descriptions = [f"A {label}" for label in self.labels]
            self.is_text_conditioned = False
            print("✅ Created simple descriptions")
        
        # Load metadata
        metadata_file = os.path.join(data_dir, 'metadata.json')
        if os.path.exists(metadata_file):
            with open(metadata_file, 'r') as f:
                self.metadata = json.load(f)
        else:
            self.metadata = {}
        
        # Create label to index mapping
        self.unique_labels = list(set(self.labels))
        self.label_to_idx = {label: idx for idx, label in enumerate(self.unique_labels)}
        self.label_indices = [self.label_to_idx[label] for label in self.labels]
        
        # Pre-encode all text descriptions for efficiency with progress bar
        print(f"🔤 Encoding {len(self.text_descriptions):,} text descriptions...")
        self.text_embeddings = self.text_encoder.encode_for_dataset(
            self.text_descriptions, 
            batch_size=1024,  # Reasonable batch size for encoding
            show_progress=True
        )
        print("📱 Text embeddings stored on CPU for DataLoader compatibility")
        
        print(f"✅ Dataset loaded successfully!")
        print(f"📊 Dataset statistics:")
        print(f"  • Trajectories: {len(self.trajectories):,} samples")
        print(f"  • Trajectory shape: {self.trajectories.shape}")
        print(f"  • Text embedding shape: {self.text_embeddings.shape}")
        print(f"  • Pattern types: {len(self.unique_labels)} ({', '.join(self.unique_labels)})")
    
    def __len__(self) -> int:
        return len(self.trajectories)
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """
        Get a single trajectory sample with text conditioning.
        
        Returns:
            Dictionary containing:
                - 'trajectory': Trajectory data as tensor [seq_len, 2]
                - 'text_embedding': Text embedding [embedding_dim]
                - 'text_description': Original text description
                - 'label': Pattern type as integer
                - 'label_name': Pattern type as string
        """
        trajectory = self.trajectories[idx]  # Shape: [seq_len, 2]
        label = self.label_indices[idx]
        label_name = self.labels[idx]
        text_description = self.text_descriptions[idx]
        text_embedding = self.text_embeddings[idx]  # Pre-computed embedding
        
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
            'text_embedding': text_embedding.clone(),  # Clone to avoid reference issues
            'text_description': text_description,
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
    
    def get_text_samples(self, pattern_type: str, n_samples: int = 5) -> List[str]:
        """Get sample text descriptions for a specific pattern type."""
        indices = [i for i, label in enumerate(self.labels) if label == pattern_type]
        selected_indices = np.random.choice(indices, min(n_samples, len(indices)), replace=False)
        return [self.text_descriptions[i] for i in selected_indices]
    
    def get_statistics(self) -> Dict:
        """Get dataset statistics."""
        stats = {
            'total_samples': len(self.trajectories),
            'sequence_length': self.trajectories.shape[1],
            'feature_dim': self.trajectories.shape[2],
            'text_embedding_dim': self.text_embeddings.shape[1],
            'is_text_conditioned': self.is_text_conditioned,
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
        
        # Sample text descriptions
        stats['sample_text_descriptions'] = {}
        for label in self.unique_labels[:5]:  # First 5 pattern types
            stats['sample_text_descriptions'][label] = self.get_text_samples(label, 3)
        
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
        pin_memory: bool = True,
        text_encoder: Optional[TextEncoder] = None
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
        self.full_dataset = TrajectoryDataset(
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
    
    def encode_text(self, text_descriptions: List[str]) -> torch.Tensor:
        """Encode new text descriptions using the shared encoder."""
        return self.text_encoder.encode(text_descriptions)


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
    # Test the text-conditioned dataset
    print("Testing Text-Conditioned TrajectoryDataset...")
    
    # Create data module
    data_module = TrajectoryDataModule(
        data_dir="data",
        batch_size=8,
        sequence_length=100,
        num_workers=0  # Disable multiprocessing to avoid CUDA issues
    )
    
    # Get sample batch
    sample_batch = data_module.get_sample_batch('train')
    
    print(f"Sample batch shapes:")
    print(f"  Trajectories: {sample_batch['trajectory'].shape}")
    print(f"  Text embeddings: {sample_batch['text_embedding'].shape}")
    print(f"  Labels: {sample_batch['label'].shape}")
    print(f"  Label names: {sample_batch['label_name']}")
    print(f"  Text descriptions: {sample_batch['text_description']}")
    
    # Print dataset statistics
    stats = data_module.full_dataset.get_statistics()
    print(f"\nDataset statistics:")
    for key, value in stats.items():
        if key != 'sample_text_descriptions':
            print(f"  {key}: {value}")
    
    # Show sample text descriptions
    print(f"\nSample text descriptions:")
    for pattern, texts in stats['sample_text_descriptions'].items():
        print(f"  {pattern}: {texts}") 