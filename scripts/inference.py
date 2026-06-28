"""Standalone inference script for ASL landmark-to-English translation.

Usage:
    python -m scripts.inference --checkpoint results/checkpoints/best --input data/landmarks/sample.npz
    python -m scripts.inference --checkpoint results/checkpoints/best --input-dir data/landmarks/ --limit 10
"""
from __future__ import annotations
import argparse
import os
import torch
from typing import List
from transformers import T5TokenizerFast

from src.models.translation_model import ASLTranslationModel
from src.dataset import ASLLandmarkDataset

def run_inference(
    checkpoint: str,
    input_paths: List[str],
    t5_model_name: str = "t5-small",
    no_face: bool = False,
    max_len: int = 150,
    device: str = "auto",
    i3d_dir: str | None = None,
    num_beams: int = 5
) -> None:
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    
    print(f"Loading tokenizer: {t5_model_name}")
    tokenizer = T5TokenizerFast.from_pretrained(t5_model_name)
    
    print(f"Loading dataset (i3d_dir={i3d_dir})...")
    # Create dataset just to load features properly using the existing logic
    dataset = ASLLandmarkDataset(
        data_dir="", # Unused when file_list is provided
        file_list=input_paths,
        max_len=max_len,
        include_face=not no_face,
        normalize=True,
        skip_empty_labels=False,
        i3d_dir=i3d_dir
    )
    
    if len(dataset) == 0:
        print("No valid landmark files found.")
        return
        
    input_dim = dataset[0]['features'].shape[1]
    
    # Load state dict first to dynamically auto-detect if checkpoint was trained with Multi-Stream Gated Fusion
    print(f"Loading checkpoint from: {checkpoint}")
    checkpoint_bin = os.path.join(checkpoint, "pytorch_model.bin")
    checkpoint_safe = os.path.join(checkpoint, "model.safetensors")
    
    if os.path.exists(checkpoint_safe):
        try:
            from safetensors.torch import load_file
            print(f"Loading weights from {checkpoint_safe}")
            state_dict = load_file(checkpoint_safe, device="cpu")
        except ImportError:
            raise ImportError("Found model.safetensors, but safetensors package is not installed. Please run `pip install safetensors`.")
    elif os.path.exists(checkpoint_bin):
        print(f"Loading weights from {checkpoint_bin}")
        state_dict = torch.load(checkpoint_bin, map_location="cpu", weights_only=True)
    else:
        raise FileNotFoundError(f"Neither model.safetensors nor pytorch_model.bin found in {checkpoint}")

    # Strip 'module.' prefix from state dict keys if present
    state_dict = {k[7:] if k.startswith("module.") else k: v for k, v in state_dict.items()}

    # Inspect state dict keys for multimodal I3D parameters
    has_i3d_weights = any('i3d_projection' in k or 'gate_conv' in k for k in state_dict.keys())
    input_i3d_dim = 1024 if has_i3d_weights else None
    
    if has_i3d_weights and not i3d_dir:
        print("\nWARNING: Loaded checkpoint was trained with Gated Fusion (I3D), but no --i3d-dir was provided. The model will run in landmark-only mode, which may result in poor translations.\n")
        
    print(f"Initializing Model (Conformer -> T5) with input_dim={input_dim}, input_i3d_dim={input_i3d_dim}...")
    model = ASLTranslationModel(
        input_dim=input_dim,
        input_i3d_dim=input_i3d_dim,
        d_model=512,
        t5_model_name=t5_model_name,
        num_layers=4,
        num_heads=4,
        kernel_size=31
    )
    
    model.load_state_dict(state_dict)
    
    model = model.to(device)
    model.eval()
    
    print("\n--- Inference Results ---")
    with torch.no_grad():
        for i in range(len(dataset)):
            sample = dataset[i]
            features = sample['features'].unsqueeze(0).to(device)
            attention_mask = torch.ones((1, features.shape[1]), dtype=torch.float32, device=device)
            
            i3d_feats = None
            if 'i3d_features' in sample:
                i3d_feats = sample['i3d_features'].unsqueeze(0).to(device)
                
            output_ids = model.generate(
                input_features=features,
                attention_mask=attention_mask,
                input_i3d_features=i3d_feats,
                max_new_tokens=30,
                num_beams=num_beams
            )
            
            prediction = tokenizer.decode(output_ids[0], skip_special_tokens=True)
            print(f"[{sample['file_id']}] → \"{prediction}\"")

def main() -> None:
    parser = argparse.ArgumentParser(description="Run inference on ASL landmarks.")
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to saved model checkpoint.")
    parser.add_argument("--input", type=str, help="Path to single .npz or .npy file.")
    parser.add_argument("--input-dir", "--input_dir", dest="input_dir", type=str, help="Directory of landmarks.")
    parser.add_argument("--i3d-dir", "--i3d_dir", dest="i3d_dir", type=str, default=None, help="Directory of precomputed I3D features.")
    parser.add_argument("--limit", type=int, default=10, help="Max samples to process from directory.")
    parser.add_argument("--t5-model", "--t5_model", dest="t5_model", type=str, default="t5-small", help="T5 model name.")
    parser.add_argument("--no-face", "--no_face", dest="no_face", action="store_true", help="Disable facial expression landmarks.")
    parser.add_argument("--max-len", "--max_len", dest="max_len", type=int, default=150, help="Maximum frame sequence length.")
    parser.add_argument("--num-beams", "--num_beams", dest="num_beams", type=int, default=5, help="Number of beams for sequence generation.")
    parser.add_argument("--device", type=str, default="auto", choices=["auto", "cpu", "cuda"])
    
    args = parser.parse_args()
    
    if not args.input and not args.input_dir:
        parser.error("Must provide either --input or --input-dir")
        
    input_paths = []
    if args.input:
        input_paths.append(args.input)
    if args.input_dir:
        from src.utils.io_utils import discover_landmark_paths
        discovered = discover_landmark_paths(args.input_dir)
        input_paths.extend(discovered[:args.limit])
        
    run_inference(
        checkpoint=args.checkpoint,
        input_paths=input_paths,
        t5_model_name=args.t5_model,
        no_face=args.no_face,
        max_len=args.max_len,
        device=args.device,
        i3d_dir=args.i3d_dir,
        num_beams=args.num_beams
    )

if __name__ == "__main__":
    main()
