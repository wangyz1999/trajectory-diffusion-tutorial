"""
Neural Network Architecture for Multi-Agent 2D Trajectory Diffusion Model
Uses UNet1D architecture adapted for multi-agent sequence-to-sequence diffusion.
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
    """1D Residual block with time and text embedding."""
    
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        time_emb_dim: int,
        text_emb_dim: int,
        kernel_size: int = 3,
        dropout: float = 0.1
    ):
        super().__init__()
        self.time_mlp = nn.Linear(time_emb_dim, out_channels)
        self.text_mlp = nn.Linear(text_emb_dim, out_channels)
        
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
    
    def forward(self, x: torch.Tensor, time_emb: torch.Tensor, text_emb: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Input tensor [batch, channels, sequence_length]
            time_emb: Time embedding [batch, time_emb_dim]
            text_emb: Text embedding [batch, text_emb_dim]
        """
        residual = self.residual_conv(x)
        
        # First conv block
        h = self.conv1(x)
        h = self.norm1(h)
        
        # Add time and text embeddings
        time_emb_proj = self.time_mlp(time_emb)
        text_emb_proj = self.text_mlp(text_emb)
        combined_emb = time_emb_proj + text_emb_proj
        h = h + combined_emb[:, :, None]  # Broadcast combined embedding
        
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


class CrossAttentionBlock1D(nn.Module):
    """1D Cross-attention block for text conditioning."""
    
    def __init__(self, channels: int, text_emb_dim: int, num_heads: int = 8):
        super().__init__()
        self.channels = channels
        self.text_emb_dim = text_emb_dim
        self.num_heads = num_heads
        self.head_dim = channels // num_heads
        
        assert channels % num_heads == 0, "channels must be divisible by num_heads"
        
        self.norm = nn.GroupNorm(8, channels)
        self.text_norm = nn.LayerNorm(text_emb_dim)
        
        # Query from trajectory features, Key/Value from text
        self.to_q = nn.Conv1d(channels, channels, 1, bias=False)
        self.to_k = nn.Linear(text_emb_dim, channels, bias=False)
        self.to_v = nn.Linear(text_emb_dim, channels, bias=False)
        self.to_out = nn.Conv1d(channels, channels, 1)
        
        self.dropout = nn.Dropout(0.1)
    
    def forward(self, x: torch.Tensor, text_emb: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Input tensor [batch, channels, sequence_length]
            text_emb: Text embedding [batch, text_emb_dim]
        """
        batch, channels, seq_len = x.shape
        residual = x
        
        # Normalize inputs
        x_norm = self.norm(x)
        text_norm = self.text_norm(text_emb)
        
        # Get query from trajectory features
        q = self.to_q(x_norm)  # [batch, channels, seq_len]
        q = q.view(batch, self.num_heads, self.head_dim, seq_len).transpose(-1, -2)  # [batch, heads, seq_len, head_dim]
        
        # Get key and value from text embedding
        k = self.to_k(text_norm)  # [batch, channels]
        v = self.to_v(text_norm)  # [batch, channels]
        
        # Reshape key and value for cross-attention
        k = k.view(batch, self.num_heads, self.head_dim, 1).transpose(-1, -2)  # [batch, heads, 1, head_dim]
        v = v.view(batch, self.num_heads, self.head_dim, 1).transpose(-1, -2)  # [batch, heads, 1, head_dim]
        
        # Cross-attention: trajectory queries attend to text keys/values
        attention = torch.softmax(q @ k.transpose(-1, -2) / math.sqrt(self.head_dim), dim=-1)  # [batch, heads, seq_len, 1]
        out = attention @ v  # [batch, heads, seq_len, head_dim]
        
        # Reshape and project
        out = out.transpose(-1, -2).contiguous().view(batch, channels, seq_len)
        out = self.to_out(out)
        out = self.dropout(out)
        
        return out + residual


class MultiAgentUNet1D(nn.Module):
    """1D U-Net adapted for multi-agent text-conditioned trajectory diffusion."""
    
    def __init__(
        self,
        max_agents: int = 5,
        in_channels_per_agent: int = 2,  # x, y coordinates per agent
        time_emb_dim: int = 128,
        text_emb_dim: int = 384,  # Text embedding dimension from sentence transformer
        base_channels: int = 64,
        channel_multipliers: Tuple[int, ...] = (1, 2, 4, 8),
        num_res_blocks: int = 2,
        attention_levels: Tuple[int, ...] = (2, 3),
        dropout: float = 0.1
    ):
        super().__init__()
        self.max_agents = max_agents
        self.in_channels_per_agent = in_channels_per_agent
        self.in_channels = max_agents * in_channels_per_agent  # Total input channels
        self.out_channels = self.in_channels  # Same as input
        self.time_emb_dim = time_emb_dim
        self.text_emb_dim = text_emb_dim
        
        # Time embedding
        self.time_embedding = nn.Sequential(
            SinusoidalPositionEmbeddings(time_emb_dim // 4),
            nn.Linear(time_emb_dim // 4, time_emb_dim),
            nn.SiLU(),
            nn.Linear(time_emb_dim, time_emb_dim)
        )
        
        # Initial projection
        self.init_conv = nn.Conv1d(self.in_channels, base_channels, 7, padding=3)
        
        # Encoder
        self.encoder_blocks = nn.ModuleList()
        self.downsample_blocks = nn.ModuleList()
        
        channels = [base_channels] + [base_channels * m for m in channel_multipliers]
        
        for i, (in_ch, out_ch) in enumerate(zip(channels[:-1], channels[1:])):
            # Residual blocks
            blocks = nn.ModuleList()
            for _ in range(num_res_blocks):
                blocks.append(ResidualBlock1D(in_ch, out_ch, time_emb_dim, text_emb_dim, dropout=dropout))
                in_ch = out_ch
            
            # Attention blocks if specified
            if i in attention_levels:
                blocks.append(AttentionBlock1D(out_ch))
                blocks.append(CrossAttentionBlock1D(out_ch, text_emb_dim))
            
            self.encoder_blocks.append(blocks)
            
            # Downsampling (except for last level)
            if i < len(channel_multipliers) - 1:
                self.downsample_blocks.append(nn.Conv1d(out_ch, out_ch, 3, stride=2, padding=1))
            else:
                self.downsample_blocks.append(nn.Identity())
        
        # Middle block
        mid_channels = channels[-1]
        self.middle_block = nn.ModuleList([
            ResidualBlock1D(mid_channels, mid_channels, time_emb_dim, text_emb_dim, dropout=dropout),
            AttentionBlock1D(mid_channels),
            CrossAttentionBlock1D(mid_channels, text_emb_dim),
            ResidualBlock1D(mid_channels, mid_channels, time_emb_dim, text_emb_dim, dropout=dropout)
        ])
        
        # Decoder
        self.decoder_blocks = nn.ModuleList()
        self.upsample_blocks = nn.ModuleList()
        
        reversed_channels = list(reversed(channels))
        
        for i, (in_ch, out_ch) in enumerate(zip(reversed_channels[:-1], reversed_channels[1:])):
            # Upsampling (except for first level)
            if i > 0:
                self.upsample_blocks.append(nn.ConvTranspose1d(in_ch, in_ch, 3, stride=2, padding=1, output_padding=1))
            else:
                self.upsample_blocks.append(nn.Identity())
            
            # Determine if this level gets a skip connection
            encoder_level = len(channels) - 2 - i
            has_skip = encoder_level >= 0
            
            # Residual blocks 
            blocks = nn.ModuleList()
            for j in range(num_res_blocks):
                if j == 0 and has_skip:
                    # First block receives concatenated input (upsampled + skip connection)
                    skip_channels = channels[encoder_level + 1]
                    concat_channels = in_ch + skip_channels
                    blocks.append(ResidualBlock1D(concat_channels, out_ch, time_emb_dim, text_emb_dim, dropout=dropout))
                elif j == 0:
                    # First block, no skip connection
                    blocks.append(ResidualBlock1D(in_ch, out_ch, time_emb_dim, text_emb_dim, dropout=dropout))
                else:
                    blocks.append(ResidualBlock1D(out_ch, out_ch, time_emb_dim, text_emb_dim, dropout=dropout))
            
            # Attention blocks if specified
            level_idx = len(channel_multipliers) - 1 - i
            if level_idx in attention_levels:
                blocks.append(AttentionBlock1D(out_ch))
                blocks.append(CrossAttentionBlock1D(out_ch, text_emb_dim))
            
            self.decoder_blocks.append(blocks)
        
        # Final output
        self.final_conv = nn.Sequential(
            nn.GroupNorm(8, base_channels),
            nn.SiLU(),
            nn.Conv1d(base_channels, self.out_channels, 7, padding=3)
        )
    
    def forward(self, x: torch.Tensor, timestep: torch.Tensor, text_emb: torch.Tensor, agent_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Args:
            x: Input multi-agent trajectories [batch, max_agents, sequence_length, 2]
            timestep: Diffusion timestep [batch]
            text_emb: Text embedding [batch, text_emb_dim]
            agent_mask: Boolean mask for active agents [batch, max_agents]
        
        Returns:
            Predicted noise [batch, max_agents, sequence_length, 2]
        """
        batch_size, max_agents, seq_len, coord_dim = x.shape
        
        # Reshape to [batch, max_agents * coord_dim, seq_len] for 1D conv processing
        x_reshaped = x.permute(0, 1, 3, 2).contiguous()  # [batch, max_agents, 2, seq_len]
        x_reshaped = x_reshaped.view(batch_size, max_agents * coord_dim, seq_len)  # [batch, max_agents*2, seq_len]
        
        # Store original sequence length for final interpolation
        self._input_seq_len = seq_len
        
        # Time embedding
        time_emb = self.time_embedding(timestep)
        
        # Initial convolution
        x_conv = self.init_conv(x_reshaped)
        
        # Store skip connections
        skip_connections = []
        
        # Encoder
        for encoder_block, downsample in zip(self.encoder_blocks, self.downsample_blocks):
            for block in encoder_block:
                if isinstance(block, ResidualBlock1D):
                    x_conv = block(x_conv, time_emb, text_emb)
                elif isinstance(block, CrossAttentionBlock1D):
                    x_conv = block(x_conv, text_emb)
                else:  # Self-attention block
                    x_conv = block(x_conv)
            
            skip_connections.append(x_conv)
            x_conv = downsample(x_conv)
        
        # Middle
        for block in self.middle_block:
            if isinstance(block, ResidualBlock1D):
                x_conv = block(x_conv, time_emb, text_emb)
            elif isinstance(block, CrossAttentionBlock1D):
                x_conv = block(x_conv, text_emb)
            else:  # Self-attention block
                x_conv = block(x_conv)
        
        # Decoder
        skip_connections = skip_connections[::-1]  # Reverse order
        
        for i, (decoder_block, upsample) in enumerate(zip(self.decoder_blocks, self.upsample_blocks)):
            x_conv = upsample(x_conv)
            
            # Check if this decoder level should use a skip connection
            skip_level = len(skip_connections) - 1 - i
            if skip_level >= 0 and i < len(skip_connections):
                skip = skip_connections[i]
                # Handle potential size mismatch due to downsampling/upsampling
                if x_conv.shape[-1] != skip.shape[-1]:
                    target_len = skip.shape[-1]
                    x_conv = F.interpolate(x_conv, size=target_len, mode='linear', align_corners=False)
                x_conv = torch.cat([x_conv, skip], dim=1)
            
            # Apply decoder blocks
            for block in decoder_block:
                if isinstance(block, ResidualBlock1D):
                    x_conv = block(x_conv, time_emb, text_emb)
                elif isinstance(block, CrossAttentionBlock1D):
                    x_conv = block(x_conv, text_emb)
                else:  # Self-attention block
                    x_conv = block(x_conv)
        
        # Final output
        x_conv = self.final_conv(x_conv)
        
        # Ensure output matches input sequence length exactly
        if hasattr(self, '_input_seq_len') and x_conv.shape[-1] != self._input_seq_len:
            x_conv = F.interpolate(x_conv, size=self._input_seq_len, mode='linear', align_corners=False)
        
        # Reshape back to multi-agent format: [batch, max_agents*2, seq_len] -> [batch, max_agents, seq_len, 2]
        output = x_conv.view(batch_size, max_agents, coord_dim, seq_len)  # [batch, max_agents, 2, seq_len]
        output = output.permute(0, 1, 3, 2).contiguous()  # [batch, max_agents, seq_len, 2]
        
        # Apply agent mask to zero out inactive agents
        if agent_mask is not None:
            # Expand mask to match output shape
            mask_expanded = agent_mask.unsqueeze(-1).unsqueeze(-1)  # [batch, max_agents, 1, 1]
            output = output * mask_expanded.float()
        
        return output


def create_multi_agent_model(
    max_agents: int = 5,
    sequence_length: int = 100,
    in_channels_per_agent: int = 2,
    base_channels: int = 64,
    time_emb_dim: int = 128,
    text_emb_dim: int = 384
) -> MultiAgentUNet1D:
    """Create a MultiAgentUNet1D model for text-conditioned multi-agent trajectory diffusion."""
    
    return MultiAgentUNet1D(
        max_agents=max_agents,
        in_channels_per_agent=in_channels_per_agent,
        time_emb_dim=time_emb_dim,
        text_emb_dim=text_emb_dim,
        base_channels=base_channels,
        channel_multipliers=(1, 2, 4, 8),
        num_res_blocks=2,
        attention_levels=(2, 3),
        dropout=0.1
    )


# Keep the original single-agent model for backward compatibility
UNet1D = MultiAgentUNet1D
create_model = create_multi_agent_model


if __name__ == "__main__":
    # Test the multi-agent model
    print("Testing Multi-Agent UNet1D model...")
    
    # Create model
    model = create_multi_agent_model(max_agents=5, sequence_length=100, text_emb_dim=384)
    
    # Create dummy input
    batch_size = 4
    max_agents = 5
    sequence_length = 100
    
    # Multi-agent trajectories: [batch, max_agents, seq_len, 2]
    x = torch.randn(batch_size, max_agents, sequence_length, 2)
    timestep = torch.randint(0, 1000, (batch_size,))
    text_emb = torch.randn(batch_size, 384)  # [batch, text_emb_dim]
    
    # Agent mask: some agents might be inactive
    agent_mask = torch.rand(batch_size, max_agents) > 0.3  # Random mask
    
    print(f"Input multi-agent trajectory shape: {x.shape}")
    print(f"Timestep shape: {timestep.shape}")
    print(f"Text embedding shape: {text_emb.shape}")
    print(f"Agent mask shape: {agent_mask.shape}")
    print(f"Active agents per sample: {agent_mask.sum(dim=1).tolist()}")
    
    # Forward pass
    with torch.no_grad():
        output = model(x, timestep, text_emb, agent_mask)
    
    print(f"Output shape: {output.shape}")
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    # Test that output has same shape as input
    assert output.shape == x.shape, f"Output shape {output.shape} doesn't match input shape {x.shape}"
    
    # Test that inactive agents have zero output
    inactive_agents = ~agent_mask
    if inactive_agents.any():
        inactive_output = output[inactive_agents]
        assert torch.allclose(inactive_output, torch.zeros_like(inactive_output)), "Inactive agents should have zero output"
    
    print("✅ Multi-agent model test passed!")
    print("🤖 Model now supports multi-agent trajectory diffusion with agent masking") 