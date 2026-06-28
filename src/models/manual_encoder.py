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
            col_indices = torch.arange(key_padding_mask.size(1), device=key_padding_mask.device)
            first_element_mask = (col_indices == 0).unsqueeze(0)
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
    """Sinusoidal Positional Encoding with dynamic expansion."""
    def __init__(self, d_model: int, max_len: int = 10000) -> None:
        super().__init__()
        self.d_model = d_model
        # Initialize default buffer on CPU
        self.register_buffer('pe', self._get_pe(max_len, torch.device("cpu"), torch.float32))

    def _get_pe(self, length: int, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
        position = torch.arange(0, length, dtype=dtype, device=device).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, self.d_model, 2, dtype=dtype, device=device) * (-math.log(10000.0) / self.d_model))
        
        pe = torch.zeros(length, self.d_model, device=device, dtype=dtype)
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        return pe.unsqueeze(0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        seq_len = x.size(1)
        if self.pe.size(1) < seq_len:
            # Generate on-the-fly without registering/mutating buffers
            pe = self._get_pe(seq_len, x.device, x.dtype)
            return x + pe
        return x + self.pe[:, :seq_len].to(device=x.device, dtype=x.dtype)

class ConformerEncoder(nn.Module):
    """
    Complete Conformer Encoder with spatial projection, positional encoding,
    conformer blocks, and temporal downsampling.
    """
    def __init__(self, input_dim: int = 534, d_model: int = 512, num_layers: int = 4, num_heads: int = 4, kernel_size: int = 31, dropout: float = 0.1):
        super().__init__()
        self.projection = nn.Sequential(
            nn.Linear(input_dim, d_model),
            nn.LayerNorm(d_model)
        )
        
        self.downsample = nn.Conv1d(
            in_channels=d_model, 
            out_channels=d_model, 
            kernel_size=3, 
            stride=2, 
            padding=1
        )
        
        self.pos_encoding = PositionalEncoding(d_model)
        self.dropout = nn.Dropout(dropout)
        
        self.num_layers = num_layers
        self.blocks = nn.ModuleList([
            ConformerBlock(d_model, num_heads, kernel_size, dropout=dropout)
            for _ in range(num_layers)
        ])
        
    def forward(self, x: torch.Tensor, attention_mask: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, Optional[torch.BoolTensor]]:
        if x.size(1) == 0:
            raise ValueError(f"Input feature sequence has length 0. Batch size: {x.size(0)}")
            
        x = self.projection(x)
        
        # Downsample temporally
        x = x.transpose(1, 2)
        x = self.downsample(x)
        x = x.transpose(1, 2)
        
        x = self.pos_encoding(x)
        x = self.dropout(x)
        
        key_padding_mask = None
        if attention_mask is not None:
            # Downsample attention mask to match the feature downsampling Conv1d parameters (1 for valid, 0 for pad)
            # Use max_pool1d with same parameters (kernel_size=3, stride=2, padding=1) to ensure perfect alignment
            downsampled_mask = F.max_pool1d(
                attention_mask.unsqueeze(1).float(),
                kernel_size=3,
                stride=2,
                padding=1
            ).squeeze(1)
            key_padding_mask = (downsampled_mask < 0.5)
            
            # Align mask length with features in a trace-friendly manner
            key_padding_mask = key_padding_mask[:, :x.size(1)]
                
        for block in self.blocks:
            x = block(x, key_padding_mask=key_padding_mask)
            
        return x, key_padding_mask

