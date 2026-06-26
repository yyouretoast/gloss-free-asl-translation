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

# TODO: Import from src.models.translation_model in Wave 2. Using src.train for now.
from src.train import ASLTranslationModel
from src.dataset import ASLLandmarkDataset

def run_inference(
    checkpoint: str,
    input_paths: List[str],
    t5_model_name: str = "t5-small",
    no_face: bool = False,
    max_len: int = 150,
    device: str = "auto"
) -> None:
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    
    print(f"Loading tokenizer: {t5_model_name}")
    tokenizer = T5TokenizerFast.from_pretrained(t5_model_name)
    
    print("Loading dataset...")
    # Create dataset just to load features properly using the existing logic
    dataset = ASLLandmarkDataset(
        data_dir="", # Unused when file_list is provided
        file_list=input_paths,
        max_len=max_len,
        include_face=not no_face,
        normalize=True,
        skip_empty_labels=False
    )
    
    if len(dataset) == 0:
        print("No valid landmark files found.")
        return
        
    input_dim = dataset[0]['features'].shape[1]
    
    print(f"Initializing Model (Conformer -> T5) with input_dim={input_dim}...")
    model = ASLTranslationModel(
        input_dim=input_dim,
        d_model=512,
        t5_model_name=t5_model_name,
        num_layers=4,
        num_heads=4,
        kernel_size=31
    )
    
    print(f"Loading checkpoint from: {checkpoint}")
    state_dict = torch.load(os.path.join(checkpoint, "pytorch_model.bin"), map_location="cpu", weights_only=True)
    model.load_state_dict(state_dict)
    
    model = model.to(device)
    model.eval()
    
    print("\n--- Inference Results ---")
    with torch.no_grad():
        for i in range(len(dataset)):
            sample = dataset[i]
            features = sample['features'].unsqueeze(0).to(device)
            attention_mask = torch.ones((1, features.shape[1]), dtype=torch.float32, device=device)
            
            output_ids = model.generate(
                input_features=features,
                attention_mask=attention_mask,
                max_new_tokens=30
            )
            
            prediction = tokenizer.decode(output_ids[0], skip_special_tokens=True)
            print(f"[{sample['file_id']}] → \"{prediction}\"")

def main() -> None:
    parser = argparse.ArgumentParser(description="Run inference on ASL landmarks.")
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to saved model checkpoint.")
    parser.add_argument("--input", type=str, help="Path to single .npz or OpenPose dir.")
    parser.add_argument("--input-dir", type=str, help="Directory of landmarks.")
    parser.add_argument("--limit", type=int, default=10, help="Max samples to process from directory.")
    parser.add_argument("--t5-model", type=str, default="t5-small", help="T5 model name.")
    parser.add_argument("--no-face", action="store_true", help="Disable facial expression landmarks.")
    parser.add_argument("--max-len", type=int, default=150, help="Maximum frame sequence length.")
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
        device=args.device
    )

if __name__ == "__main__":
    main()
