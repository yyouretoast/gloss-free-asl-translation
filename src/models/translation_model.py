"""
ASL Translation Model Module.

This module contains the sequence-to-sequence model connecting a custom
Conformer Encoder to a pretrained T5 Decoder for sign language translation.
"""
from __future__ import annotations

import torch
import torch.nn as nn
from typing import Optional, Any
from collections import OrderedDict
from transformers import T5ForConditionalGeneration
from transformers.modeling_outputs import BaseModelOutput
from src.models.manual_encoder import ConformerEncoder

class ASLTranslationModel(nn.Module):
    """
    Sequence-to-sequence model connecting the custom Conformer Encoder
    directly to a pretrained T5 Decoder.
    """
    def __init__(
        self, 
        input_dim: int = 534, 
        input_i3d_dim: Optional[int] = None,
        d_model: int = 512, 
        t5_model_name: str = 't5-small', 
        num_layers: int = 4, 
        num_heads: int = 4, 
        kernel_size: int = 31
    ) -> None:
        super().__init__()
        # 1. Custom Conformer Encoder
        self.encoder = ConformerEncoder(
            input_dim=input_dim,
            d_model=d_model,
            num_layers=num_layers,
            num_heads=num_heads,
            kernel_size=kernel_size
        )
        
        # 2. Pretrained T5 Decoder
        self.t5 = T5ForConditionalGeneration.from_pretrained(t5_model_name)
        
        # Delete unused T5 encoder to save VRAM
        del self.t5.encoder
        
        # Define I3D projection and temporal downsampling (if configured)
        self.input_i3d_dim = input_i3d_dim
        if input_i3d_dim:
            self.i3d_projection = nn.Conv1d(
                in_channels=input_i3d_dim,
                out_channels=d_model,
                kernel_size=3,
                stride=2,
                padding=1
            )
            # Gated fusion layers using a 1D temporal convolution (3-frame window)
            self.gate_conv = nn.Conv1d(
                in_channels=2 * d_model,
                out_channels=d_model,
                kernel_size=3,
                padding=1
            )
            
        # T5-Small d_model is 512, T5-Base is 768, T5-Large is 1024
        t5_d_model = self.t5.config.d_model
        
        # Non-Linear MLP Modality Bridge to project Conformer features to text embedding space
        self.modality_bridge = nn.Sequential(
            nn.Linear(d_model, t5_d_model),
            nn.GELU(),
            nn.LayerNorm(t5_d_model)
        )

        # Expose generation config for Seq2SeqTrainer
        self.generation_config = self.t5.generation_config

    def forward(
        self, 
        input_features: torch.Tensor, 
        attention_mask: Optional[torch.Tensor] = None, 
        labels: Optional[torch.Tensor] = None, 
        decoder_attention_mask: Optional[torch.Tensor] = None, 
        input_i3d_features: Optional[torch.Tensor] = None,
        **kwargs: Any
    ) -> Any:
        # Extract visual sequence representation
        # outputs shape: (batch, seq_len // 2, d_model)
        outputs, downsampled_mask = self.encoder(input_features, attention_mask=attention_mask)
        
        # Gated Multimodal Fusion (if I3D features are present)
        if input_i3d_features is not None and self.input_i3d_dim is not None:
            import torch.nn.functional as F
            # Shape: (batch, seq_len_i3d // 2, d_model)
            i3d_proj = self.i3d_projection(input_i3d_features.transpose(1, 2)).transpose(1, 2)
            
            # Align downsampled sequence lengths exactly without python branching (ONNX compatible)
            i3d_proj = F.pad(i3d_proj, (0, 0, 0, outputs.size(1)))[:, :outputs.size(1), :]
                
            # Compute temporal gate over moving window
            combined = torch.cat([outputs, i3d_proj], dim=-1) # (B, seq_len // 2, 2 * d_model)
            gate = torch.sigmoid(self.gate_conv(combined.transpose(1, 2)).transpose(1, 2)) # (B, seq_len // 2, d_model)
            
            # Apply gated fusion blending
            outputs = gate * outputs + (1.0 - gate) * i3d_proj
        
        # Project dimensions using the MLP bridge
        outputs = self.modality_bridge(outputs)
        
        # Pass visual embeddings directly as T5 encoder output hidden states
        encoder_outputs = BaseModelOutput(last_hidden_state=outputs)
        
        t5_attention_mask = None
        if downsampled_mask is not None:
            valid_mask = (~downsampled_mask).long()
            if valid_mask.size(1) > 0:
                t5_attention_mask = torch.cat([
                    torch.ones((valid_mask.size(0), 1), dtype=valid_mask.dtype, device=valid_mask.device),
                    valid_mask[:, 1:]
                ], dim=1)
            else:
                t5_attention_mask = valid_mask
        
        # Run T5 Decoder forward pass
        return self.t5(
            encoder_outputs=encoder_outputs,
            attention_mask=t5_attention_mask,
            labels=labels,
            decoder_attention_mask=decoder_attention_mask
        )

    def generate(
        self, 
        input_features: torch.Tensor, 
        attention_mask: Optional[torch.Tensor] = None, 
        input_i3d_features: Optional[torch.Tensor] = None,
        **kwargs: Any
    ) -> Any:
        """
        Overrides generate to enable visual-to-text sequence generation.
        Used during evaluation/inference.
        """
        with torch.no_grad():
            outputs, downsampled_mask = self.encoder(input_features, attention_mask=attention_mask)
            
            # Gated Multimodal Fusion (if I3D features are present)
            if input_i3d_features is not None and self.input_i3d_dim is not None:
                import torch.nn.functional as F
                i3d_proj = self.i3d_projection(input_i3d_features.transpose(1, 2)).transpose(1, 2)
                
                # Align downsampled sequence lengths exactly without python branching (ONNX compatible)
                i3d_proj = F.pad(i3d_proj, (0, 0, 0, outputs.size(1)))[:, :outputs.size(1), :]
                    
                combined = torch.cat([outputs, i3d_proj], dim=-1)
                gate = torch.sigmoid(self.gate_conv(combined.transpose(1, 2)).transpose(1, 2))
                outputs = gate * outputs + (1.0 - gate) * i3d_proj

            outputs = self.modality_bridge(outputs)
            encoder_outputs = BaseModelOutput(last_hidden_state=outputs)
            
            t5_attention_mask = None
            if downsampled_mask is not None:
                valid_mask = (~downsampled_mask).long()
                if valid_mask.size(1) > 0:
                    t5_attention_mask = torch.cat([
                        torch.ones((valid_mask.size(0), 1), dtype=valid_mask.dtype, device=valid_mask.device),
                        valid_mask[:, 1:]
                    ], dim=1)
                else:
                    t5_attention_mask = valid_mask
            
            # Clean kwargs that are not accepted by T5's generate method
            kwargs.pop('file_ids', None)
            kwargs.pop('labels', None)
            kwargs.pop('decoder_attention_mask', None)
            
            # Use T5's auto-regressive generation
            return self.t5.generate(
                encoder_outputs=encoder_outputs,
                attention_mask=t5_attention_mask,
                bos_token_id=self.t5.config.decoder_start_token_id,
                **kwargs
            )

    def state_dict(self, *args: Any, destination: Optional[OrderedDict] = None, prefix: str = '', keep_vars: bool = False) -> OrderedDict:
        """
        Overrides state_dict to clone all tensors, preventing safetensors from raising
        a RuntimeError about shared memory tensors (like T5 embed_tokens/shared weights).
        """
        sd = super().state_dict(*args, destination=destination, prefix=prefix, keep_vars=keep_vars)
        if keep_vars:
            return sd
        # Clone to prevent shared parameter issues in safetensors
        cloned_sd = OrderedDict((k, v.clone()) for k, v in sd.items())
        if destination is not None:
            destination.clear()
            destination.update(cloned_sd)
            return destination
        return cloned_sd

    def gradient_checkpointing_enable(self, **kwargs: Any) -> None:
        """
        Delegates gradient checkpointing to the T5 module.
        Required by Seq2SeqTrainer when gradient_checkpointing=True is set.
        """
        self.t5.gradient_checkpointing_enable(**kwargs)

    def gradient_checkpointing_disable(self) -> None:
        """
        Delegates disabling of gradient checkpointing to the T5 module.
        """
        self.t5.gradient_checkpointing_disable()
