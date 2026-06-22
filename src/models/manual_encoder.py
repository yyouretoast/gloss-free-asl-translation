import torch
import torch.nn as nn
import math

class FeedForwardModule(nn.Module):
    """
    Feed Forward Module (FFN) with half-step residual connection (Macaron style).
    """
    def __init__(self, d_model, expansion_factor=4, dropout=0.1):
        super().__init__()
        self.ln = nn.LayerNorm(d_model)
        self.w_1 = nn.Linear(d_model, d_model * expansion_factor)
        self.act = nn.SiLU()
        self.w_2 = nn.Linear(d_model * expansion_factor, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
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
    def __init__(self, d_model, kernel_size=31, dropout=0.1):
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

    def forward(self, x, key_padding_mask=None):
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
        
        if key_padding_mask is not None:
            x = x.masked_fill(key_padding_mask.unsqueeze(-1), 0.0)
            
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
    Multi-Head Self-Attention Module with positional bias.
    """
    def __init__(self, d_model, num_heads=4, dropout=0.1):
        super().__init__()
        self.ln = nn.LayerNorm(d_model)
        self.attn = nn.MultiheadAttention(embed_dim=d_model, num_heads=num_heads, dropout=dropout, batch_first=True)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, key_padding_mask=None):
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
    def __init__(self, d_model, num_heads=4, kernel_size=31, expansion_factor=4, dropout=0.1):
        super().__init__()
        self.ffn1 = FeedForwardModule(d_model, expansion_factor, dropout)
        self.attn = ConformerAttentionModule(d_model, num_heads, dropout)
        self.conv = ConformerConvModule(d_model, kernel_size, dropout)
        self.ffn2 = FeedForwardModule(d_model, expansion_factor, dropout)
        self.final_ln = nn.LayerNorm(d_model)

    def forward(self, x, key_padding_mask=None):
        # Macaron-style sandwich structure
        x = self.ffn1(x)
        if key_padding_mask is not None:
            x = x.masked_fill(key_padding_mask.unsqueeze(-1), 0.0)
            
        x = self.attn(x, key_padding_mask=key_padding_mask)
        if key_padding_mask is not None:
            x = x.masked_fill(key_padding_mask.unsqueeze(-1), 0.0)
            
        x = self.conv(x, key_padding_mask=key_padding_mask)
        if key_padding_mask is not None:
            x = x.masked_fill(key_padding_mask.unsqueeze(-1), 0.0)
            
        x = self.ffn2(x)
        x = self.final_ln(x)
        if key_padding_mask is not None:
            x = x.masked_fill(key_padding_mask.unsqueeze(-1), 0.0)
            
        return x

class PositionalEncoding(nn.Module):
    """
    Sinusoidal Positional Encoding for sequence temporal awareness.
    """
    def __init__(self, d_model, max_len=10000):
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

    def forward(self, x):
        # x shape: (batch, seq_len, d_model)
        return x + self.pe[:, :x.size(1)]

class ConformerEncoder(nn.Module):
    """
    Complete Conformer Encoder with spatial projection, positional encoding,
    conformer blocks, and temporal downsampling.
    """
    def __init__(self, input_dim=534, d_model=512, num_layers=4, num_heads=4, kernel_size=31, dropout=0.1):
        super().__init__()
        # 1. Feature Projection Layer
        self.projection = nn.Sequential(
            nn.Linear(input_dim, d_model),
            nn.LayerNorm(d_model)
        )
        
        self.pos_encoding = PositionalEncoding(d_model)
        self.dropout = nn.Dropout(dropout)
        
        # 2. Conformer Blocks (before downsampling)
        # We will split layers to add a temporal downsampling pooling layer in the middle
        self.num_layers = num_layers
        self.first_half = nn.ModuleList([
            ConformerBlock(d_model, num_heads, kernel_size, dropout=dropout)
            for _ in range(num_layers // 2)
        ])
        
        # 3. Temporal Pyramidal Downsampling (1D learnable Conv1d replacing MaxPool1d)
        # Downsamples the sequence length by 2
        self.downsample = nn.Conv1d(
            in_channels=d_model, 
            out_channels=d_model, 
            kernel_size=3, 
            stride=2, 
            padding=1
        )
        
        # 4. Conformer Blocks (after downsampling)
        self.second_half = nn.ModuleList([
            ConformerBlock(d_model, num_heads, kernel_size, dropout=dropout)
            for _ in range(num_layers - (num_layers // 2))
        ])
        
    def forward(self, x, attention_mask=None):
        """
        Args:
            x (Tensor): Landmark features of shape (batch, seq_len, input_dim).
            attention_mask (Tensor): Mask of shape (batch, seq_len) with 1 for real frames, 0 for pad.
        """
        # Guard against completely empty sequence length input
        if x.size(1) == 0:
            raise ValueError(f"Input feature sequence has length 0. Batch size: {x.size(0)}")
            
        # Feature projection
        x = self.projection(x)
        x = self.pos_encoding(x)
        x = self.dropout(x)
        
        # Convert attention_mask (1 for valid, 0 for pad) to PyTorch key_padding_mask (True for pad, False for valid)
        # Using < 0.5 is more robust across FP16/BF16 than == 0
        key_padding_mask = None
        if attention_mask is not None:
            key_padding_mask = (attention_mask < 0.5)
            
        # First half of blocks
        for block in self.first_half:
            x = block(x, key_padding_mask=key_padding_mask)
            
        # Pyramidal Temporal Downsampling
        # Conv1d expects shape: (batch, channels, seq_len)
        x = x.transpose(1, 2)
        x = self.downsample(x)
        x = x.transpose(1, 2)
        
        # Downsample the attention mask/padding mask as well
        downsampled_mask = None
        if key_padding_mask is not None:
            # Slices every 2nd index along seq_len to match Conv1d stride=2
            downsampled_mask = key_padding_mask[:, ::2]
            # Ensure sequence length alignment (handles stride boundary)
            downsampled_mask = downsampled_mask[:, :x.size(1)]
                
        # Second half of blocks
        for block in self.second_half:
            x = block(x, key_padding_mask=downsampled_mask)
            
        return x, downsampled_mask
