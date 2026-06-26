"""Checks the local environment for GPU availability and MediaPipe setup."""
from __future__ import annotations
import sys
import torch

def check_gpu() -> None:
    print("--- PyTorch GPU Check ---")
    print(f"PyTorch Version: {torch.__version__}")
    if torch.cuda.is_available():
        print("CUDA is AVAILABLE.")
        print(f"Device Name: {torch.cuda.get_device_name(0)}")
        print(f"CUDA Version: {torch.version.cuda}")
        print(f"cuDNN Version: {torch.backends.cudnn.version()}")
    else:
        print("CUDA is NOT available. PyTorch will use CPU.")
    print()

def check_mediapipe() -> None:
    print("--- MediaPipe Check ---")
    try:
        import mediapipe as mp
        print(f"MediaPipe Version: {mp.__version__}")
        print("MediaPipe imported successfully.")
    except ImportError as e:
        print(f"Failed to import MediaPipe: {e}")
        print("Install with: pip install mediapipe")
    print()

def main() -> None:
    check_gpu()
    check_mediapipe()
    print("Environment check complete.")

if __name__ == "__main__":
    main()
