"""Standalone evaluation script computing BLEU-4 and WER on a checkpoint.

Usage:
    python -m scripts.evaluate --checkpoint results/checkpoints/best --data-dir data/landmarks --metadata metadata.csv
"""
from __future__ import annotations
import argparse
import os
import torch
import numpy as np
from transformers import T5TokenizerFast

from src.models.translation_model import ASLTranslationModel
from src.dataset import ASLLandmarkDataset, CollateLandmarks
from torch.utils.data import DataLoader

try:
    import jiwer
    import sacrebleu
except ImportError:
    jiwer = None
    sacrebleu = None

def evaluate(
    checkpoint: str,
    data_dir: str,
    metadata_file: str,
    t5_model_name: str = "t5-small",
    no_face: bool = False,
    max_len: int = 150,
    batch_size: int = 8,
    device: str = "auto"
) -> None:
    if not jiwer or not sacrebleu:
        print("Error: jiwer and sacrebleu packages are required for evaluation. Run `pip install jiwer sacrebleu`.")
        return
        
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
        
    print(f"Loading tokenizer: {t5_model_name}")
    tokenizer = T5TokenizerFast.from_pretrained(t5_model_name)
    
    # Simple metadata loading (can be updated to use src.utils.metadata in Wave 2)
    metadata = {}
    if metadata_file and os.path.exists(metadata_file):
        import pandas as pd
        df = pd.read_csv(metadata_file)
        if 'id' in df.columns and 'text' in df.columns:
            metadata = dict(zip(df['id'].astype(str), df['text'].astype(str)))
    else:
        print("Warning: No metadata provided. Evaluation requires target labels.")
        return
        
    print(f"Loading dataset from: {data_dir}")
    dataset = ASLLandmarkDataset(
        data_dir=data_dir,
        metadata_dict=metadata,
        max_len=max_len,
        include_face=not no_face,
        normalize=True,
        skip_empty_labels=True
    )
    
    if len(dataset) == 0:
        print("No valid landmark files found.")
        return
        
    collate_fn = CollateLandmarks(tokenizer=tokenizer)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False, collate_fn=collate_fn)
    
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
    
    all_preds = []
    all_labels = []
    
    print("Running evaluation...")
    with torch.no_grad():
        for batch in dataloader:
            features = batch['input_features'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].numpy()
            
            output_ids = model.generate(
                input_features=features,
                attention_mask=attention_mask,
                max_new_tokens=30
            )
            
            preds = tokenizer.batch_decode(output_ids, skip_special_tokens=True)
            
            # Replace -100 in labels so they can be decoded
            labels = np.where(labels != -100, labels, tokenizer.pad_token_id)
            decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)
            
            all_preds.extend([p.strip() for p in preds])
            all_labels.extend([lbl.strip() for lbl in decoded_labels])
            
    # Compute metrics
    valid_preds = [p if p else " " for p in all_preds]
    valid_labels = [lbl if lbl else " " for lbl in all_labels]
    
    wer = jiwer.wer(valid_labels, valid_preds)
    bleu = sacrebleu.corpus_bleu(valid_preds, [valid_labels]).score
    
    print("\n--- Evaluation Results ---")
    print(f"Total Samples: {len(valid_labels)}")
    print(f"BLEU-4: {bleu:.2f}")
    print(f"WER:    {wer:.4f}")
    
    print("\n--- Sample Predictions ---")
    for i in range(min(5, len(valid_labels))):
        print(f"Label: {valid_labels[i]}")
        print(f"Pred:  {valid_preds[i]}")
        print()

def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate ASL translation model.")
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to saved model checkpoint.")
    parser.add_argument("--data-dir", type=str, required=True, help="Directory of landmarks.")
    parser.add_argument("--metadata", type=str, required=True, help="Path to metadata CSV.")
    parser.add_argument("--t5-model", type=str, default="t5-small", help="T5 model name.")
    parser.add_argument("--no-face", action="store_true", help="Disable facial expression landmarks.")
    parser.add_argument("--max-len", type=int, default=150, help="Maximum frame sequence length.")
    parser.add_argument("--batch-size", type=int, default=8, help="Batch size.")
    parser.add_argument("--device", type=str, default="auto", choices=["auto", "cpu", "cuda"])
    
    args = parser.parse_args()
    
    evaluate(
        checkpoint=args.checkpoint,
        data_dir=args.data_dir,
        metadata_file=args.metadata,
        t5_model_name=args.t5_model,
        no_face=args.no_face,
        max_len=args.max_len,
        batch_size=args.batch_size,
        device=args.device
    )

if __name__ == "__main__":
    main()
