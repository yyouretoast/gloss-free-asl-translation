"""Exports the Conformer Encoder component to ONNX for fast inference."""
from __future__ import annotations
import os
import argparse
import torch
import torch.nn as nn
import onnx
from typing import Optional

from src.models.translation_model import ASLTranslationModel

class ONNXEncoderWrapper(nn.Module):
    """
    Wrapper that chains the ConformerEncoder and the Modality Bridge together.
    This ensures that the exported ONNX model outputs projected features that
    directly match the T5 decoder's input embedding dimension.
    """
    def __init__(self, encoder: nn.Module, modality_bridge: nn.Module):
        super().__init__()
        self.encoder = encoder
        self.modality_bridge = modality_bridge
        
    def forward(self, input_features: torch.Tensor, attention_mask: Optional[torch.Tensor] = None):
        outputs, downsampled_mask = self.encoder(input_features, attention_mask=attention_mask)
        outputs = self.modality_bridge(outputs)
        return outputs, downsampled_mask

def export_encoder_to_onnx(
    model_path: Optional[str] = None, 
    output_onnx_path: str = "results/conformer_encoder.onnx", 
    input_dim: int = 534,
    d_model: int = 512,
    t5_model_name: str = "t5-small",
    num_layers: int = 4,
    num_heads: int = 4,
    kernel_size: int = 31
) -> None:
    """
    Exports only the Conformer Encoder part of our ASL translation model to ONNX.
    This bypasses the T5 Decoder's auto-regressive generation loop, which cannot
    be easily traced or exported to ONNX due to conditional control flows.
    """
    print(f"Initializing model for ONNX export (input_dim={input_dim}, t5_model_name={t5_model_name})...")
    model = ASLTranslationModel(
        input_dim=input_dim,
        d_model=d_model,
        t5_model_name=t5_model_name,
        num_layers=num_layers,
        num_heads=num_heads,
        kernel_size=kernel_size
    )
    
    # Load weights if path is provided
    if model_path:
        if os.path.isdir(model_path):
            safetensors_path = os.path.join(model_path, "model.safetensors")
            bin_path = os.path.join(model_path, "pytorch_model.bin")
            if os.path.exists(safetensors_path):
                model_path = safetensors_path
            elif os.path.exists(bin_path):
                model_path = bin_path
            else:
                raise FileNotFoundError(f"No checkpoint weights (model.safetensors or pytorch_model.bin) found in directory {model_path}")
                
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model path {model_path} does not exist.")
            
        print(f"Loading weights from {model_path}...")
        if model_path.endswith(".safetensors"):
            try:
                from safetensors.torch import load_file
                state_dict = load_file(model_path, device="cpu")
            except ImportError:
                raise ImportError("Found model.safetensors, but safetensors package is not installed. Please run `pip install safetensors`.")
        else:
            state_dict = torch.load(model_path, map_location='cpu', weights_only=True)
        model.load_state_dict(state_dict)
    
    # Extract encoder and modality bridge modules, wrap them for export
    wrapper = ONNXEncoderWrapper(model.encoder, model.modality_bridge)
    wrapper.eval()
    
    # Create dummy inputs: (batch_size, seq_len, input_dim)
    dummy_input_features = torch.randn(1, 150, input_dim)
    dummy_attention_mask = torch.ones(1, 150, dtype=torch.float32)
    
    os.makedirs(os.path.dirname(output_onnx_path), exist_ok=True)
    
    print(f"Exporting ONNXEncoderWrapper to {output_onnx_path}...")
    torch.onnx.export(
        wrapper,
        (dummy_input_features, dummy_attention_mask),
        output_onnx_path,
        export_params=True,
        opset_version=18,  # Opsets 14+ support LayerNorm and Swish (SiLU) operations efficiently
        do_constant_folding=True,
        input_names=['input_features', 'attention_mask'],
        output_names=['encoder_outputs', 'downsampled_mask'],
        dynamic_axes={
            'input_features': {0: 'batch_size', 1: 'sequence_length'},
            'attention_mask': {0: 'batch_size', 1: 'sequence_length'},
            'encoder_outputs': {0: 'batch_size', 1: 'downsampled_sequence_length'},
            'downsampled_mask': {0: 'batch_size', 1: 'downsampled_sequence_length'}
        },
        dynamo=False
    )
    
    # Validate the generated ONNX model
    print("Validating the exported ONNX model...")
    onnx.checker.check_model(output_onnx_path)
    print("ONNX export and validation completed successfully!")

def main() -> None:
    parser = argparse.ArgumentParser(description="Export ASL Conformer Encoder to ONNX.")
    parser.add_argument("--model-path", "--model_path", dest="model_path", type=str, default=None, help="Path to trained PyTorch model weights.")
    parser.add_argument("--output", type=str, default="results/conformer_encoder.onnx", help="Output path for ONNX model.")
    parser.add_argument("--input-dim", "--input_dim", dest="input_dim", type=int, default=534, help="Input feature dimension.")
    parser.add_argument("--d-model", "--d_model", dest="d_model", type=int, default=512, help="Conformer embedding dimension.")
    parser.add_argument("--num-layers", "--num_layers", dest="num_layers", type=int, default=4, help="Number of Conformer blocks.")
    parser.add_argument("--num-heads", "--num_heads", dest="num_heads", type=int, default=4, help="Number of attention heads.")
    parser.add_argument("--kernel-size", "--kernel_size", dest="kernel_size", type=int, default=31, help="Conformer conv kernel size.")
    parser.add_argument("--t5-model", "--t5_model", dest="t5_model", type=str, default="t5-small", help="T5 model name used during training.")
    
    args = parser.parse_args()
    
    export_encoder_to_onnx(
        model_path=args.model_path,
        output_onnx_path=args.output,
        input_dim=args.input_dim,
        d_model=args.d_model,
        t5_model_name=args.t5_model,
        num_layers=args.num_layers,
        num_heads=args.num_heads,
        kernel_size=args.kernel_size
    )

if __name__ == "__main__":
    main()
