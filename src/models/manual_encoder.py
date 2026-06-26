"""ASL Translation model components."""
from __future__ import annotations
from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class FeedForwardModule(nn.Module):
    """
    Feed Forward Module (FFN) with half-step residual connection (Macaron style).
    """
    def __init__(self, d_model: int, expansion_factor: int = 4, dropout: float = 0.1):
        super().__init__()
        self.ln = nn.LayerNorm(d_model)
        self.w_1 = nn.Linear(d_model, d_model * expansion_factor)
        self.act = nn.SiLU()
        self.w_2 = nn.Linear(d_model * expansion_factor, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Input tensor of shape (batch, seq_len, d_model).
        Returns:
            Output tensor of shape (batch, seq_len, d_model).
        """
        residual = x
        x = self.ln(x)
        x = self.w_1(x)
        x = self.act(x)
        x = self.dropout(x)
        x = self.w_2(x)
        x = self.dropout(x)
        # Scaled by 0.5 for Macaron sandwich style
        return residual + 0.5 * x

class ConformerConvModule(nn.Module):
    """
    Convolution Module inside the Conformer block.
    """
    def __init__(self, d_model: int, kernel_size: int = 31, dropout: float = 0.1):
        super().__init__()
        # Depthwise separable convolution setup
        # Input shape expected: (batch, channels, seq_len)
        self.ln = nn.LayerNorm(d_model)
        
        # 1x1 Pointwise Conv to project to 2 * d_model (for GLU)
        self.pointwise_conv1 = nn.Conv1d(d_model, 2 * d_model, kernel_size=1)
        self.glu = nn.GLU(dim=1)
        
        # Depthwise Conv1d (groups = d_model)
        padding = (kernel_size - 1) // 2
        self.depthwise_conv = nn.Conv1d(
            d_model, d_model, kernel_size=kernel_size, 
            padding=padding, groups=d_model
        )
        
        self.conv_norm = nn.LayerNorm(d_model)
        self.act = nn.SiLU()
        
        # 1x1 Pointwise Conv to project back to d_model
        self.pointwise_conv2 = nn.Conv1d(d_model, d_model, kernel_size=1)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, key_padding_mask: Optional[torch.BoolTensor] = None) -> torch.Tensor:
        """
        Args:
            x: Input tensor of shape (batch, seq_len, d_model).
            key_padding_mask: Boolean mask of shape (batch, seq_len), True for padded positions.
        Returns:
            Output tensor of shape (batch, seq_len, d_model).
        """
        # x shape: (batch, seq_len, d_model)
        residual = x
        x = self.ln(x)
        
        if key_padding_mask is not None:
            x = x.masked_fill(key_padding_mask.unsqueeze(-1), 0.0)
        
        # Transpose for Conv1d: (batch, d_model, seq_len)
        x = x.transpose(1, 2)
        
        x = self.pointwise_conv1(x)
        x = self.glu(x)
        x = self.depthwise_conv(x)
        
        # Transpose to (batch, seq_len, d_model) for LayerNorm to avoid padding bias
        x = x.transpose(1, 2)
        x = self.conv_norm(x)
        
        x = x.transpose(1, 2)
        
        x = self.act(x)
        x = self.pointwise_conv2(x)
        x = self.dropout(x)
        
        # Transpose back: (batch, seq_len, d_model)
        x = x.transpose(1, 2)
        
        if key_padding_mask is not None:
            x = x.masked_fill(key_padding_mask.unsqueeze(-1), 0.0)
            
        return residual + x

class ConformerAttentionModule(nn.Module):
    """
    Multi-Head Self-Attention Module.

    Note: Uses absolute sinusoidal positional encoding added before the Conformer blocks,
    rather than the relative sinusoidal positional encoding described in Gulati et al. (2020).
    This is a deliberate simplification for implementation tractability.
    """
    def __init__(self, d_model: int, num_heads: int = 4, dropout: float = 0.1):
        super().__init__()
        self.ln = nn.LayerNorm(d_model)
        self.attn = nn.MultiheadAttention(embed_dim=d_model, num_heads=num_heads, dropout=dropout, batch_first=True)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, key_padding_mask: Optional[torch.BoolTensor] = None) -> torch.Tensor:
        """
        Args:
            x: Input tensor of shape (batch, seq_len, d_model).
            key_padding_mask: Boolean mask of shape (batch, seq_len), True for padded positions.
        Returns:
            Output tensor of shape (batch, seq_len, d_model).
        """
        # x shape: (batch, seq_len, d_model)
        residual = x
        x = self.ln(x)
        
        # Prevent NaN when a sequence is completely padded.
        # If all elements in a row are True, set the first element to False.
        if key_padding_mask is not None:
            all_padded = key_padding_mask.all(dim=-1, keepdim=True)
            first_element_mask = torch.zeros_like(key_padding_mask)
            first_element_mask[:, 0] = True
            key_padding_mask = key_padding_mask & ~(all_padded & first_element_mask)
        
        # PyTorch MultiheadAttention expects batch_first=True
        # key_padding_mask should be shape (batch, seq_len) containing True for padded values
        attn_out, _ = self.attn(x, x, x, key_padding_mask=key_padding_mask)
        
        return residual + self.dropout(attn_out)

class ConformerBlock(nn.Module):
    """
    A single Gulati-style Conformer Block.
    """
    def __init__(self, d_model: int, num_heads: int = 4, kernel_size: int = 31, expansion_factor: int = 4, dropout: float = 0.1):
        super().__init__()
        self.ffn1 = FeedForwardModule(d_model, expansion_factor, dropout)
        self.attn = ConformerAttentionModule(d_model, num_heads, dropout)
        self.conv = ConformerConvModule(d_model, kernel_size, dropout)
        self.ffn2 = FeedForwardModule(d_model, expansion_factor, dropout)
        self.final_ln = nn.LayerNorm(d_model)

    def forward(self, x: torch.Tensor, key_padding_mask: Optional[torch.BoolTensor] = None) -> torch.Tensor:
        """
        Args:
            x: Input tensor of shape (batch, seq_len, d_model).
            key_padding_mask: Boolean mask of shape (batch, seq_len), True for padded positions.
        Returns:
            Output tensor of shape (batch, seq_len, d_model).
        """
        # Macaron-style sandwich structure
        x = self.ffn1(x)
        x = self.attn(x, key_padding_mask=key_padding_mask)
        x = self.conv(x, key_padding_mask=key_padding_mask)
        x = self.ffn2(x)
        x = self.final_ln(x)
        
        if key_padding_mask is not None:
            x = x.masked_fill(key_padding_mask.unsqueeze(-1), 0.0)
            
        return x

class PositionalEncoding(nn.Module):
    """
    Sinusoidal Positional Encoding for sequence temporal awareness.
    """
    def __init__(self, d_model: int, max_len: int = 10000):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        
        # Ensure correct shapes in case d_model is odd
        sin_term = torch.sin(position * div_term)
        cos_term = torch.cos(position * div_term)
        
        pe[:, 0::2] = sin_term[:, :pe[:, 0::2].size(1)]
        pe[:, 1::2] = cos_term[:, :pe[:, 1::2].size(1)]
        
        # Register as buffer so it moves with the model device
        self.register_buffer('pe', pe.unsqueeze(0))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Input tensor of shape (batch, seq_len, d_model).
        Returns:
            Output tensor of shape (batch, seq_len, d_model).
        """
        # x shape: (batch, seq_len, d_model)
        return x + self.pe[:, :x.size(1)]

class ConformerEncoder(nn.Module):
    """
    Complete Conformer Encoder with spatial projection, positional encoding,
    conformer blocks, and temporal downsampling.
    """
    def __init__(self, input_dim: int = 534, d_model: int = 512, num_layers: int = 4, num_heads: int = 4, kernel_size: int = 31, dropout: float = 0.1):
        super().__init__()
        # 1. Feature Projection Layer
        self.projection = nn.Sequential(
            nn.Linear(input_dim, d_model),
            nn.LayerNorm(d_model)
        )
        
        # 2. Temporal Pyramidal Downsampling (1D learnable Conv1d replacing MaxPool1d)
        # Downsamples the sequence length by 2 at the input stage
        self.downsample = nn.Conv1d(
            in_channels=d_model, 
            out_channels=d_model, 
            kernel_size=3, 
            stride=2, 
            padding=1
        )
        
        self.pos_encoding = PositionalEncoding(d_model)
        self.dropout = nn.Dropout(dropout)
        
        # 3. Conformer Blocks
        self.num_layers = num_layers
        self.blocks = nn.ModuleList([
            ConformerBlock(d_model, num_heads, kernel_size, dropout=dropout)
            for _ in range(num_layers)
        ])
        
    def forward(self, x: torch.Tensor, attention_mask: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, Optional[torch.BoolTensor]]:
        """
        Args:
            x (Tensor): Landmark features of shape (batch, seq_len, input_dim).
            attention_mask (Tensor): Mask of shape (batch, seq_len) with 1 for real frames, 0 for pad.
        Returns:
            Tuple of:
                - Output tensor of shape (batch, seq_len // 2, d_model)
                - Downsampled boolean padding mask of shape (batch, seq_len // 2)
        """
        # Guard against completely empty sequence length input
        if x.size(1) == 0:
            raise ValueError(f"Input feature sequence has length 0. Batch size: {x.size(0)}")
            
        # Feature projection
        x = self.projection(x)
        
        # Pyramidal Temporal Downsampling
        # Conv1d expects shape: (batch, channels, seq_len)
        x = x.transpose(1, 2)
        x = self.downsample(x)
        x = x.transpose(1, 2)
        
        # Positional Encoding and Dropout
        x = self.pos_encoding(x)
        x = self.dropout(x)
        
        # Convert attention_mask (1 for valid, 0 for pad) to PyTorch key_padding_mask (True for pad, False for valid)
        # Using < 0.5 is more robust across FP16/BF16 than == 0
        key_padding_mask = None
        if attention_mask is not None:
            valid_mask = attention_mask.float().unsqueeze(1)  # (B, 1, L)
            downsampled_valid = F.max_pool1d(valid_mask, kernel_size=3, stride=2, padding=1)
            key_padding_mask = (downsampled_valid.squeeze(1) < 0.5)
            # Ensure sequence length alignment (handles stride boundary)
            key_padding_mask = key_padding_mask[:, :x.size(1)]
                
        # Conformer Blocks forward pass
        for block in self.blocks:
            x = block(x, key_padding_mask=key_padding_mask)
            
        return x, key_padding_mask
