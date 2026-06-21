import os
import sys
import torch

# Add the project root to path so we can import src modules
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from src.train import ASLTranslationModel

def export_encoder_to_onnx(model_path=None, output_onnx_path="results/conformer_encoder.onnx", input_dim=534):
    """
    Exports only the Conformer Encoder part of our ASL translation model to ONNX.
    This bypasses the T5 Decoder's auto-regressive generation loop, which cannot
    be easily traced or exported to ONNX due to conditional control flows.
    """
    print(f"Initializing model for ONNX export (input_dim={input_dim})...")
    model = ASLTranslationModel(
        input_dim=input_dim,
        d_model=512,
        num_layers=4,
        num_heads=4,
        kernel_size=31
    )
    
    # Load weights if path is provided
    if model_path and os.path.exists(model_path):
        print(f"Loading weights from {model_path}...")
        state_dict = torch.load(model_path, map_location='cpu')
        model.load_state_dict(state_dict)
    
    # Extract only the encoder module
    encoder = model.encoder
    encoder.eval()
    
    # Create dummy inputs: (batch_size, seq_len, input_dim)
    dummy_input_features = torch.randn(1, 150, input_dim)
    dummy_attention_mask = torch.ones(1, 150, dtype=torch.float32)
    
    os.makedirs(os.path.dirname(output_onnx_path), exist_ok=True)
    
    print(f"Exporting ConformerEncoder to {output_onnx_path}...")
    torch.onnx.export(
        encoder,
        (dummy_input_features, dummy_attention_mask),
        output_onnx_path,
        export_params=True,
        opset_version=14,  # Opsets 14+ support LayerNorm and Swish (SiLU) operations efficiently
        do_constant_folding=True,
        input_names=['input_features', 'attention_mask'],
        output_names=['encoder_outputs', 'downsampled_mask'],
        dynamic_axes={
            'input_features': {0: 'batch_size', 1: 'sequence_length'},
            'attention_mask': {0: 'batch_size', 1: 'sequence_length'},
            'encoder_outputs': {0: 'batch_size', 1: 'downsampled_sequence_length'},
            'downsampled_mask': {0: 'batch_size', 1: 'downsampled_sequence_length'}
        }
    )
    print("ONNX export completed successfully!")

if __name__ == "__main__":
    # Check if there is an argument passed to override input dimension (e.g. 411 for How2Sign)
    input_dim = 534
    if len(sys.argv) > 1:
        try:
            input_dim = int(sys.argv[1])
        except ValueError:
            pass
            
    export_encoder_to_onnx(input_dim=input_dim)
