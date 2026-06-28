"""Checks the local environment for GPU availability and MediaPipe setup."""

from __future__ import annotations
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
        import importlib.metadata

        try:
            version = importlib.metadata.version("mediapipe")
        except importlib.metadata.PackageNotFoundError:
            version = getattr(mp, "__version__", "unknown")
        print(f"MediaPipe Version: {version}")
        print("MediaPipe imported successfully.")
    except ImportError as e:
        print(f"Failed to import MediaPipe: {e}")
        print("Install with: pip install mediapipe")
    print()


def check_package(package_name: str) -> None:
    print(f"--- {package_name} Check ---")
    try:
        import importlib

        pkg = importlib.import_module(package_name)
        import importlib.metadata

        try:
            version = importlib.metadata.version(package_name)
        except importlib.metadata.PackageNotFoundError:
            version = getattr(pkg, "__version__", "unknown")
        print(f"{package_name} Version: {version}")
        print(f"{package_name} imported successfully.")
    except ImportError as e:
        print(f"Failed to import {package_name}: {e}")
        print(f"Install with: pip install {package_name}")
    print()


def main() -> None:
    check_gpu()
    check_mediapipe()
    for pkg in ["onnx", "onnxruntime", "safetensors", "jiwer", "sacrebleu"]:
        check_package(pkg)
    print("Environment check complete.")


if __name__ == "__main__":
    main()
