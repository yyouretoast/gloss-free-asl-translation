"""Profiles memory usage for the ASL Translation model on CPU/GPU."""
from __future__ import annotations
import os
import sys
import torch
from transformers import T5Config, T5ForConditionalGeneration

def profile_model_memory(model_name: str = 't5-small', batch_size: int = 8, seq_len: int = 150, target_len: int = 30, d_model: int = 512) -> None:
    """
    Profiles CPU/GPU memory usage for the T5 decoder forward and backward pass.
    """
    print(f"\n--- Profiling {model_name} (Batch: {batch_size}, Input Frames: {seq_len}, Target Text Tokens: {target_len}) ---")
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Track initial memory if CUDA
    if device.type == 'cuda':
        torch.cuda.empty_cache()
        mem_start = torch.cuda.memory_allocated() / (1024 ** 2)
        print(f"Base CUDA Memory: {mem_start:.2f} MB")
    
    # 1. Load Pretrained T5 model (with decoder and custom encoder input support)
    print(f"Loading {model_name}...")
    try:
        model = T5ForConditionalGeneration.from_pretrained(model_name).to(device)
    except Exception as e:
        print(f"Error loading {model_name}: {e}. Initializing a mock configuration instead.")
        # Fallback to random weights if offline or Hugging Face fails
        config = T5Config(d_model=d_model, num_layers=6 if 'small' in model_name else 12)
        model = T5ForConditionalGeneration(config).to(device)

    # 2. Mock visual features from Conformer Encoder (batch, seq_len, d_model)
    # T5-Small d_model is 512. T5-Base d_model is 768.
    visual_embeddings = torch.randn(batch_size, seq_len, d_model, requires_grad=True, device=device)
    
    # Mock target translation token IDs (batch, target_len)
    labels = torch.randint(100, 3000, (batch_size, target_len), dtype=torch.long, device=device)
    
    # 3. Simulate Forward Pass
    # In Hugging Face, we wrap the custom visual embeddings in a tuple and pass it as `encoder_outputs`
    # this bypasses the T5 text encoder entirely and feeds visual states directly into the decoder.
    encoder_outputs = (visual_embeddings,)
    
    if device.type == 'cuda':
        torch.cuda.reset_peak_memory_stats()
        
    print("Running forward pass...")
    outputs = model(
        encoder_outputs=encoder_outputs,
        labels=labels
    )
    loss = outputs.loss
    
    # 4. Simulate Backward Pass (gradient calculation)
    print("Running backward pass...")
    loss.backward()
    
    # 5. Measure VRAM/RAM allocation
    if device.type == 'cuda':
        mem_allocated = torch.cuda.memory_allocated() / (1024 ** 2)
        mem_peak = torch.cuda.max_memory_allocated() / (1024 ** 2)
        print(f"CUDA Memory Allocated: {mem_allocated:.2f} MB")
        print(f"CUDA Peak Memory:      {mem_peak:.2f} MB")
        print(f"Total VRAM Delta:      {mem_peak - mem_start:.2f} MB")
    else:
        print("Running locally on CPU. No CUDA VRAM stats available.")
        print(f"Backward pass succeeded. Loss: {loss.item():.4f}")
        print("Note: To get accurate VRAM profiles, run this script inside a Kaggle GPU notebook.")

    # Cleanup memory
    del model, visual_embeddings, labels, outputs, loss
    if device.type == 'cuda':
        torch.cuda.empty_cache()

def main() -> None:
    # Profile t5-small (d_model=512)
    profile_model_memory(model_name='t5-small', batch_size=8, seq_len=150, target_len=30, d_model=512)
    
    # Profile t5-base (d_model=768)
    profile_model_memory(model_name='t5-base', batch_size=4, seq_len=150, target_len=30, d_model=768)

if __name__ == "__main__":
    main()
