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
    device: str = "auto",
    i3d_dir: str | None = None,
    num_beams: int = 5,
) -> None:
    if not jiwer or not sacrebleu:
        print(
            "Error: jiwer and sacrebleu packages are required for evaluation. Run `pip install jiwer sacrebleu`."
        )
        return

    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"

    print(f"Loading tokenizer: {t5_model_name}")
    tokenizer = T5TokenizerFast.from_pretrained(t5_model_name)

    # Simple metadata loading (can be updated to use src.utils.metadata in Wave 2)
    metadata = {}
    if metadata_file and os.path.exists(metadata_file):
        from src.utils.metadata import load_metadata

        metadata, _ = load_metadata(metadata_file)
    else:
        print("Warning: No metadata provided. Evaluation requires target labels.")
        return

    include_face = not no_face
    print(
        f"Loading evaluation dataset from: {data_dir} (include_face={include_face}, i3d_dir={i3d_dir})"
    )
    dataset = ASLLandmarkDataset(
        data_dir=data_dir,
        metadata_dict=metadata,
        max_len=max_len,
        include_face=include_face,
        skip_empty_labels=True,
        i3d_dir=i3d_dir,
    )

    if len(dataset) == 0:
        print("No valid landmark files found.")
        return

    collate_fn = CollateLandmarks(tokenizer=tokenizer)
    dataloader = DataLoader(
        dataset, batch_size=batch_size, shuffle=False, collate_fn=collate_fn
    )

    input_dim = dataset[0]["features"].shape[1]

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
            raise ImportError(
                "Found model.safetensors, but safetensors package is not installed. Please run `pip install safetensors`."
            )
    elif os.path.exists(checkpoint_bin):
        print(f"Loading weights from {checkpoint_bin}")
        state_dict = torch.load(checkpoint_bin, map_location="cpu", weights_only=True)
    else:
        raise FileNotFoundError(
            f"Neither model.safetensors nor pytorch_model.bin found in {checkpoint}"
        )

    # Strip 'module.' prefix from state dict keys if present
    state_dict = {
        k[7:] if k.startswith("module.") else k: v for k, v in state_dict.items()
    }

    # Inspect state dict keys for multimodal I3D parameters
    has_i3d_weights = any(
        "i3d_projection" in k or "gate_conv" in k for k in state_dict.keys()
    )
    input_i3d_dim = 1024 if has_i3d_weights else None

    if has_i3d_weights and not i3d_dir:
        print(
            "\nWARNING: Loaded checkpoint was trained with Gated Fusion (I3D), but no --i3d-dir was provided. The model will run in landmark-only mode, which may result in poor translations.\n"
        )

    print(
        f"Initializing Model (Conformer -> T5) with input_dim={input_dim}, input_i3d_dim={input_i3d_dim}..."
    )
    model = ASLTranslationModel(
        input_dim=input_dim,
        input_i3d_dim=input_i3d_dim,
        d_model=512,
        t5_model_name=t5_model_name,
        num_layers=4,
        num_heads=4,
        kernel_size=31,
    )

    model.load_state_dict(state_dict)

    model = model.to(device)
    model.eval()

    all_preds = []
    all_labels = []

    print("Running evaluation...")
    try:
        with torch.no_grad():
            for batch in dataloader:
                features = batch["input_features"].to(device)
                attention_mask = batch["attention_mask"].to(device)
                labels = batch["labels"].numpy()

                i3d_feats = None
                if "input_i3d_features" in batch:
                    i3d_feats = batch["input_i3d_features"].to(device)

                output_ids = model.generate(
                    input_features=features,
                    attention_mask=attention_mask,
                    input_i3d_features=i3d_feats,
                    max_new_tokens=30,
                    num_beams=num_beams,
                )

                preds = tokenizer.batch_decode(output_ids, skip_special_tokens=True)

                # Replace -100 in labels so they can be decoded
                labels = np.where(labels != -100, labels, tokenizer.pad_token_id)
                decoded_labels = tokenizer.batch_decode(
                    labels, skip_special_tokens=True
                )

                all_preds.extend([p.strip() for p in preds])
                all_labels.extend([lbl.strip() for lbl in decoded_labels])
    except torch.cuda.OutOfMemoryError:
        print(
            "Error: GPU Out of Memory occurred during evaluation generation. Try decreasing --batch-size."
        )
        return

    # Compute metrics
    valid_preds = [p if p else " " for p in all_preds]
    valid_labels = [lbl if lbl else " " for lbl in all_labels]

    try:
        wer = jiwer.wer(valid_labels, valid_preds)
    except Exception:
        wer = 1.0
    try:
        bleu = sacrebleu.corpus_bleu(valid_preds, [valid_labels]).score
    except Exception:
        bleu = 0.0

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
    parser.add_argument(
        "--checkpoint", type=str, required=True, help="Path to saved model checkpoint."
    )
    parser.add_argument(
        "--data-dir",
        "--data_dir",
        dest="data_dir",
        type=str,
        required=True,
        help="Directory of landmarks.",
    )
    parser.add_argument(
        "--i3d-dir",
        "--i3d_dir",
        dest="i3d_dir",
        type=str,
        default=None,
        help="Directory of precomputed I3D features.",
    )
    parser.add_argument(
        "--metadata",
        "--metadata-file",
        "--metadata_file",
        dest="metadata",
        type=str,
        required=True,
        help="Path to metadata CSV.",
    )
    parser.add_argument(
        "--t5-model",
        "--t5_model",
        dest="t5_model",
        type=str,
        default="t5-small",
        help="T5 model name (e.g., t5-small, t5-base, or t5-large).",
    )
    parser.add_argument(
        "--no-face",
        "--no_face",
        dest="no_face",
        action="store_true",
        help="Disable facial expression landmarks.",
    )
    parser.add_argument(
        "--max-len",
        "--max_len",
        dest="max_len",
        type=int,
        default=150,
        help="Maximum frame sequence length.",
    )
    parser.add_argument(
        "--batch-size",
        "--batch_size",
        dest="batch_size",
        type=int,
        default=8,
        help="Batch size.",
    )
    parser.add_argument(
        "--num-beams",
        "--num_beams",
        dest="num_beams",
        type=int,
        default=5,
        help="Number of beams for sequence generation.",
    )
    parser.add_argument(
        "--device", type=str, default="auto", choices=["auto", "cpu", "cuda"]
    )

    args = parser.parse_args()

    evaluate(
        checkpoint=args.checkpoint,
        data_dir=args.data_dir,
        metadata_file=args.metadata,
        t5_model_name=args.t5_model,
        no_face=args.no_face,
        max_len=args.max_len,
        batch_size=args.batch_size,
        device=args.device,
        i3d_dir=args.i3d_dir,
        num_beams=args.num_beams,
    )


if __name__ == "__main__":
    main()
