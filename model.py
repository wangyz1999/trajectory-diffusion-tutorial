"""
Neural Network Architecture for 2D Trajectory Diffusion Model
Uses UNet1D architecture for sequence-to-sequence diffusion.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Optional, Tuple


class SinusoidalPositionEmbeddings(nn.Module):
    """Sinusoidal position embeddings for time steps."""
    
    def __init__(self, dim: int):
        super().__init__()
        self.dim = dim
    
    def forward(self, time: torch.Tensor) -> torch.Tensor:
        device = time.device
        half_dim = self.dim // 2
        embeddings = math.log(10000) / (half_dim - 1)
        embeddings = torch.exp(torch.arange(half_dim, device=device) * -embeddings)
        embeddings = time[:, None] * embeddings[None, :]
        embeddings = torch.cat((embeddings.sin(), embeddings.cos()), dim=-1)
        return embeddings


class ResidualBlock1D(nn.Module):
    """1D Residual block with time embedding."""
    
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        time_emb_dim: int,
        kernel_size: int = 3,
        dropout: float = 0.1
    ):
        super().__init__()
        self.time_mlp = nn.Linear(time_emb_dim, out_channels)
        
        self.conv1 = nn.Conv1d(in_channels, out_channels, kernel_size, padding=kernel_size//2)
        self.conv2 = nn.Conv1d(out_channels, out_channels, kernel_size, padding=kernel_size//2)
        
        self.norm1 = nn.GroupNorm(8, out_channels)
        self.norm2 = nn.GroupNorm(8, out_channels)
        
        self.dropout = nn.Dropout(dropout)
        
        # Residual connection
        if in_channels != out_channels:
            self.residual_conv = nn.Conv1d(in_channels, out_channels, 1)
        else:
            self.residual_conv = nn.Identity()
    
    def forward(self, x: torch.Tensor, time_emb: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Input tensor [batch, channels, sequence_length]
            time_emb: Time embedding [batch, time_emb_dim]
        """
        residual = self.residual_conv(x)
        
        # First conv block
        h = self.conv1(x)
        h = self.norm1(h)
        
        # Add time embedding
        time_emb = self.time_mlp(time_emb)
        h = h + time_emb[:, :, None]  # Broadcast time embedding
        
        h = F.silu(h)
        h = self.dropout(h)
        
        # Second conv block
        h = self.conv2(h)
        h = self.norm2(h)
        
        return F.silu(h + residual)


class AttentionBlock1D(nn.Module):
    """1D Self-attention block."""
    
    def __init__(self, channels: int, num_heads: int = 8):
        super().__init__()
        self.channels = channels
        self.num_heads = num_heads
        self.head_dim = channels // num_heads
        
        assert channels % num_heads == 0, "channels must be divisible by num_heads"
        
        self.norm = nn.GroupNorm(8, channels)
        self.to_qkv = nn.Conv1d(channels, channels * 3, 1, bias=False)
        self.to_out = nn.Conv1d(channels, channels, 1)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Input tensor [batch, channels, sequence_length]
        """
        batch, channels, seq_len = x.shape
        residual = x
        
        x = self.norm(x)
        qkv = self.to_qkv(x).chunk(3, dim=1)
        q, k, v = map(
            lambda t: t.view(batch, self.num_heads, self.head_dim, seq_len).transpose(-1, -2),
            qkv
        )
        
        # Attention
        attention = torch.softmax(q @ k.transpose(-1, -2) / math.sqrt(self.head_dim), dim=-1)
        out = attention @ v
        
        # Reshape and project
        out = out.transpose(-1, -2).contiguous().view(batch, channels, seq_len)
        out = self.to_out(out)
        
        return out + residual


class UNet1D(nn.Module):
    """1D U-Net for trajectory diffusion."""
    
    def __init__(
        self,
        in_channels: int = 2,  # x, y coordinates
        out_channels: int = 2,
        time_emb_dim: int = 128,
        base_channels: int = 64,
        channel_multipliers: Tuple[int, ...] = (1, 2, 4, 8),
        num_res_blocks: int = 2,
        attention_levels: Tuple[int, ...] = (2, 3),
        dropout: float = 0.1
    ):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.time_emb_dim = time_emb_dim
        
        # Time embedding
        self.time_embedding = nn.Sequential(
            SinusoidalPositionEmbeddings(time_emb_dim // 4),
            nn.Linear(time_emb_dim // 4, time_emb_dim),
            nn.SiLU(),
            nn.Linear(time_emb_dim, time_emb_dim)
        )
        
        # Initial projection
        self.init_conv = nn.Conv1d(in_channels, base_channels, 7, padding=3)
        
        # Encoder
        self.encoder_blocks = nn.ModuleList()
        self.downsample_blocks = nn.ModuleList()
        
        channels = [base_channels] + [base_channels * m for m in channel_multipliers]
        
        for i, (in_ch, out_ch) in enumerate(zip(channels[:-1], channels[1:])):
            # Residual blocks
            blocks = nn.ModuleList()
            for _ in range(num_res_blocks):
                blocks.append(ResidualBlock1D(in_ch, out_ch, time_emb_dim, dropout=dropout))
                in_ch = out_ch
            
            # Attention block if specified
            if i in attention_levels:
                blocks.append(AttentionBlock1D(out_ch))
            
            self.encoder_blocks.append(blocks)
            
            # Downsampling (except for last level)
            if i < len(channel_multipliers) - 1:
                self.downsample_blocks.append(nn.Conv1d(out_ch, out_ch, 4, stride=2, padding=1))
            else:
                self.downsample_blocks.append(nn.Identity())
        
        # Middle block
        mid_channels = channels[-1]
        self.middle_block = nn.ModuleList([
            ResidualBlock1D(mid_channels, mid_channels, time_emb_dim, dropout=dropout),
            AttentionBlock1D(mid_channels),
            ResidualBlock1D(mid_channels, mid_channels, time_emb_dim, dropout=dropout)
        ])
        
        # Decoder
        self.decoder_blocks = nn.ModuleList()
        self.upsample_blocks = nn.ModuleList()
        
        reversed_channels = list(reversed(channels))
        
        for i, (in_ch, out_ch) in enumerate(zip(reversed_channels[:-1], reversed_channels[1:])):
            # Upsampling (except for first level)
            if i > 0:
                self.upsample_blocks.append(nn.ConvTranspose1d(in_ch, in_ch, 4, stride=2, padding=1))
            else:
                self.upsample_blocks.append(nn.Identity())
            
            # Determine if this level gets a skip connection
            # Skip connections come from encoder levels in reverse order
            # But we skip the deepest level (which is used as middle block)
            encoder_level = len(channels) - 2 - i  # Which encoder level this corresponds to
            has_skip = encoder_level >= 0
            
            # Residual blocks 
            blocks = nn.ModuleList()
            for j in range(num_res_blocks):
                if j == 0 and has_skip:
                    # First block receives concatenated input (upsampled + skip connection)
                    # Skip connection has the output channels from the corresponding encoder level
                    skip_channels = channels[encoder_level + 1]  # Encoder level produces channels[level+1] output
                    concat_channels = in_ch + skip_channels
                    blocks.append(ResidualBlock1D(concat_channels, out_ch, time_emb_dim, dropout=dropout))
                elif j == 0:
                    # First block, no skip connection
                    blocks.append(ResidualBlock1D(in_ch, out_ch, time_emb_dim, dropout=dropout))
                else:
                    blocks.append(ResidualBlock1D(out_ch, out_ch, time_emb_dim, dropout=dropout))
            
            # Attention block if specified
            level_idx = len(channel_multipliers) - 1 - i
            if level_idx in attention_levels:
                blocks.append(AttentionBlock1D(out_ch))
            
            self.decoder_blocks.append(blocks)
        
        # Final output
        self.final_conv = nn.Sequential(
            nn.GroupNorm(8, base_channels),
            nn.SiLU(),
            nn.Conv1d(base_channels, out_channels, 7, padding=3)
        )
    
    def forward(self, x: torch.Tensor, timestep: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Input trajectory [batch, 2, sequence_length]
            timestep: Diffusion timestep [batch]
        
        Returns:
            Predicted noise [batch, 2, sequence_length]
        """
        # Time embedding
        time_emb = self.time_embedding(timestep)
        
        # Initial convolution
        x = self.init_conv(x)
        
        # Store skip connections
        skip_connections = []
        
        # Encoder
        for encoder_block, downsample in zip(self.encoder_blocks, self.downsample_blocks):
            for block in encoder_block:
                if isinstance(block, ResidualBlock1D):
                    x = block(x, time_emb)
                else:  # Attention block
                    x = block(x)
            
            skip_connections.append(x)
            x = downsample(x)
        
        # Middle
        for block in self.middle_block:
            if isinstance(block, ResidualBlock1D):
                x = block(x, time_emb)
            else:  # Attention block
                x = block(x)
        
        # Decoder
        skip_connections = skip_connections[::-1]  # Reverse order
        
        for i, (decoder_block, upsample) in enumerate(zip(self.decoder_blocks, self.upsample_blocks)):
            x = upsample(x)
            
            # Check if this decoder level should use a skip connection
            skip_level = len(skip_connections) - 1 - i  # Which skip connection to use
            if skip_level >= 0 and i < len(skip_connections):
                skip = skip_connections[i]
                # Handle potential size mismatch due to downsampling/upsampling
                if x.shape[-1] != skip.shape[-1]:
                    min_len = min(x.shape[-1], skip.shape[-1])
                    x = x[..., :min_len]
                    skip = skip[..., :min_len]
                x = torch.cat([x, skip], dim=1)
            
            # Apply decoder blocks
            for block in decoder_block:
                if isinstance(block, ResidualBlock1D):
                    x = block(x, time_emb)
                else:  # Attention block
                    x = block(x)
        
        # Final output
        return self.final_conv(x)


def create_model(
    sequence_length: int = 100,
    in_channels: int = 2,
    base_channels: int = 64,
    time_emb_dim: int = 128
) -> UNet1D:
    """Create a UNet1D model with appropriate configuration for trajectory diffusion."""
    
    return UNet1D(
        in_channels=in_channels,
        out_channels=in_channels,  # Predict noise of same dimension
        time_emb_dim=time_emb_dim,
        base_channels=base_channels,
        channel_multipliers=(1, 2, 4, 8),
        num_res_blocks=2,
        attention_levels=(2, 3),
        dropout=0.1
    )


if __name__ == "__main__":
    # Test the model
    print("Testing UNet1D model...")
    
    # Create model
    model = create_model(sequence_length=100)
    
    # Create dummy input
    batch_size = 4
    sequence_length = 100
    x = torch.randn(batch_size, 2, sequence_length)  # [batch, channels, seq_len]
    timestep = torch.randint(0, 1000, (batch_size,))
    
    print(f"Input shape: {x.shape}")
    print(f"Timestep shape: {timestep.shape}")
    
    # Forward pass
    with torch.no_grad():
        output = model(x, timestep)
    
    print(f"Output shape: {output.shape}")
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    # Test that output has same shape as input
    assert output.shape == x.shape, f"Output shape {output.shape} doesn't match input shape {x.shape}"
    
    print("Model test passed!") 