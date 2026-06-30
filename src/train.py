"""
ASL Translation Model Training Orchestrator.
"""

from __future__ import annotations


import os
import glob
import torch
import argparse
import numpy as np
import jiwer
import sacrebleu

from transformers import (
    T5TokenizerFast,
    Seq2SeqTrainer,
    Seq2SeqTrainingArguments,
    EarlyStoppingCallback,
)

from src.dataset import ASLLandmarkDataset, CollateLandmarks
from src.models.translation_model import ASLTranslationModel
from src.utils.metadata import load_metadata
from src.utils.splits import split_by_signer


def main():
    parser = argparse.ArgumentParser(
        description="Train ASL Landmark to English translation model."
    )
    parser.add_argument(
        "--data_dir",
        "--data-dir",
        dest="data_dir",
        type=str,
        default="data/landmarks",
        help="Path to landmark directory",
    )
    parser.add_argument(
        "--i3d_dir",
        "--i3d-dir",
        dest="i3d_dir",
        type=str,
        default=None,
        help="Path to precomputed training I3D feature directory (enables Multi-Stream Gated Fusion)",
    )
    parser.add_argument(
        "--i3d_val_dir",
        "--i3d-val-dir",
        dest="i3d_val_dir",
        type=str,
        default=None,
        help="Path to precomputed validation I3D feature directory (defaults to --i3d_dir)",
    )
    parser.add_argument(
        "--epochs", type=int, default=10, help="Number of training epochs"
    )
    parser.add_argument(
        "--batch_size",
        "--batch-size",
        dest="batch_size",
        type=int,
        default=8,
        help="Batch size per device",
    )
    parser.add_argument("--lr", type=float, default=1e-4, help="Learning rate")
    parser.add_argument(
        "--output_dir",
        "--output-dir",
        dest="output_dir",
        type=str,
        default="results/checkpoints",
        help="Output checkpoints folder",
    )
    parser.add_argument(
        "--metadata_file",
        "--metadata-file",
        dest="metadata_file",
        type=str,
        default=None,
        help="Path to metadata CSV/TSV file mapping video IDs to translations",
    )
    parser.add_argument(
        "--no_face",
        "--no-face",
        dest="no_face",
        action="store_true",
        help="Disable facial expression landmarks (for ablation study)",
    )
    parser.add_argument(
        "--max_len",
        "--max-len",
        dest="max_len",
        type=int,
        default=150,
        help="Maximum frame sequence length (caps longer sequences to prevent OOM)",
    )
    parser.add_argument(
        "--max_target_len",
        "--max-target-len",
        dest="max_target_len",
        type=int,
        default=30,
        help="Maximum target text sequence token length",
    )
    parser.add_argument(
        "--resume_from_checkpoint",
        "--resume-from-checkpoint",
        dest="resume_from_checkpoint",
        type=str,
        default=None,
        help="Path to checkpoint folder to resume training from, or 'latest' to auto-detect",
    )
    parser.add_argument(
        "--t5_model",
        "--t5-model",
        dest="t5_model",
        type=str,
        default="t5-small",
        help="Hugging Face T5 decoder checkpoint (t5-small, t5-base, or t5-large)",
    )
    args = parser.parse_args()

    # Suppress redundant missing I3D warnings to prevent console output flood
    import warnings

    warnings.filterwarnings("ignore", message=".*Failed to load I3D.*")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Training on device: {device}")

    # 1. Initialize tokenizer
    t5_model_name = args.t5_model
    print(f"Loading tokenizer: {t5_model_name}")
    tokenizer = T5TokenizerFast.from_pretrained(t5_model_name)

    # 2. Setup metadata mapping and signer info
    metadata, video_to_signer = load_metadata(args.metadata_file)

    # 3. Create complete Dataset to extract file list
    include_face = not args.no_face
    print(
        f"Loading dataset from: {args.data_dir} (include_face={include_face}, i3d_dir={args.i3d_dir})"
    )
    full_dataset = ASLLandmarkDataset(
        data_dir=args.data_dir,
        metadata_dict=metadata,
        max_len=args.max_len,
        include_face=include_face,
        i3d_dir=args.i3d_dir,
    )

    if len(full_dataset) == 0:
        raise ValueError(f"No landmark files found in {args.data_dir}. Verify path.")

    # 4. Partition files strictly by Signer ID (Signer-Independent splits)
    train_files, val_files = split_by_signer(full_dataset.filepaths, video_to_signer)

    train_dataset = ASLLandmarkDataset(
        data_dir=args.data_dir,
        metadata_dict=metadata,
        file_list=train_files,
        max_len=args.max_len,
        include_face=include_face,
        i3d_dir=args.i3d_dir,
        training=True,
    )
    val_dataset = ASLLandmarkDataset(
        data_dir=args.data_dir,
        metadata_dict=metadata,
        file_list=val_files,
        max_len=args.max_len,
        include_face=include_face,
        i3d_dir=args.i3d_val_dir if args.i3d_val_dir is not None else args.i3d_dir,
        training=False,
    )

    # 5. Collator
    # Pass tokenizer to collator to convert English targets to token IDs
    collate_fn = CollateLandmarks(
        tokenizer=tokenizer, max_target_len=args.max_target_len
    )

    # 6. Initialize Model dynamically using input shape
    sample_batch = train_dataset[0]
    input_dim = sample_batch["features"].shape[1]
    input_i3d_dim = 1024 if args.i3d_dir is not None else None
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
    # Set model's autoregressive generation config maximum length to avoid T5 default 20-token truncation
    model.generation_config.max_length = args.max_target_len

    # Calculate warmup steps dynamically based on dataset size, epochs, and gradient accumulation
    # 7. Define Seq2Seq Training Arguments
    report_to = "none"
    try:
        import tensorboard  # noqa: F401

        report_to = "tensorboard"
    except ImportError:
        pass

    training_args = Seq2SeqTrainingArguments(
        output_dir=args.output_dir,
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        gradient_accumulation_steps=8,  # Simulates effective batch size of 64
        learning_rate=args.lr,
        weight_decay=0.01,
        logging_dir=os.path.join(args.output_dir, "logs"),
        logging_steps=10,  # Updated to 10 per requirements
        eval_strategy="epoch",
        save_strategy="epoch",
        save_total_limit=3,  # Limits checkpoints to keep only the 3 most recent, preventing disk overflow
        predict_with_generate=True,  # Enables generating actual text during eval
        generation_max_length=30,  # Added per requirements
        generation_num_beams=1,  # Enable greedy search (beam width 1) during training validation to speed up evaluation epochs
        fp16=torch.cuda.is_available(),  # Use mixed precision if GPU available
        report_to=report_to,  # Enables TensorBoard logging if installed
        remove_unused_columns=False,
        warmup_ratio=0.1,  # Dynamic warmup ratio to stabilize training early
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",  # Use loss instead of WER — WER stays flat at ~1.0 for many early epochs and triggers premature early stopping
        greater_is_better=False,
        # --- Performance & Memory Enhancements ---
        optim="adafactor",  # Native T5 optimizer, saves massive VRAM
        gradient_checkpointing=True,  # Reduces activation memory footprint
        dataloader_num_workers=2,  # Parallelizes CPU data fetching
        dataloader_pin_memory=True,  # Accelerates CPU-to-GPU data transfers
    )

    # 8. Define metrics computation
    def compute_metrics(eval_preds):
        preds, labels = eval_preds
        if isinstance(preds, tuple):
            preds = preds[0]

        # Filter -100 from predictions before decoding
        preds = np.where(preds != -100, preds, tokenizer.pad_token_id)

        # Replace -100 in labels so they can be decoded
        labels = np.where(labels != -100, labels, tokenizer.pad_token_id)

        decoded_preds = tokenizer.batch_decode(preds, skip_special_tokens=True)
        decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)

        # Log metric evaluation count to verify DDP/predict_with_generate gathering
        print(
            f"\n--- Evaluated {len(decoded_preds)} predictions inside compute_metrics ---"
        )

        # Clean whitespaces
        decoded_preds = [p.strip() for p in decoded_preds]
        decoded_labels = [lbl.strip() for lbl in decoded_labels]

        # Avoid empty strings causing jiwer errors
        decoded_preds = [p if p else " " for p in decoded_preds]
        decoded_labels = [lbl if lbl else " " for lbl in decoded_labels]

        try:
            wer = jiwer.wer(decoded_labels, decoded_preds)
        except Exception:
            wer = 1.0
        try:
            bleu = sacrebleu.corpus_bleu(decoded_preds, [decoded_labels]).score
        except Exception:
            bleu = 0.0

        return {"bleu": float(bleu), "wer": float(wer)}

    # 8. Define subclassed Trainer to create optimizer dynamically (preserving checkpoint resumability)
    class CustomSeq2SeqTrainer(Seq2SeqTrainer):
        def create_optimizer(self):
            if self.optimizer is None:
                decay_parameters = self.get_decay_parameter_names(self.model)
                optimizer_grouped_parameters = [
                    {
                        "params": [
                            p
                            for n, p in self.model.named_parameters()
                            if p.requires_grad
                            and n in decay_parameters
                            and "t5" not in n
                        ],
                        "weight_decay": self.args.weight_decay,
                        "lr": self.args.learning_rate,
                    },
                    {
                        "params": [
                            p
                            for n, p in self.model.named_parameters()
                            if p.requires_grad
                            and n not in decay_parameters
                            and "t5" not in n
                        ],
                        "weight_decay": 0.0,
                        "lr": self.args.learning_rate,
                    },
                    {
                        "params": [
                            p
                            for n, p in self.model.named_parameters()
                            if p.requires_grad and n in decay_parameters and "t5" in n
                        ],
                        "weight_decay": self.args.weight_decay,
                        "lr": self.args.learning_rate * 0.1,
                    },
                    {
                        "params": [
                            p
                            for n, p in self.model.named_parameters()
                            if p.requires_grad
                            and n not in decay_parameters
                            and "t5" in n
                        ],
                        "weight_decay": 0.0,
                        "lr": self.args.learning_rate * 0.1,
                    },
                ]
                from transformers.optimization import Adafactor

                self.optimizer = Adafactor(
                    optimizer_grouped_parameters,
                    scale_parameter=False,
                    relative_step=False,
                    warmup_init=False,
                    lr=self.args.learning_rate,
                )
            return self.optimizer

    # 9. Initialize Trainer
    trainer = CustomSeq2SeqTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        data_collator=collate_fn,
        processing_class=tokenizer,
        compute_metrics=compute_metrics,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=10)],
    )

    # 10. Start Training Loop
    print("\nStarting training loop...")
    resume_path = args.resume_from_checkpoint
    if resume_path == "latest":
        # Auto-detect latest checkpoint under output_dir
        checkpoint_dirs = sorted(
            glob.glob(os.path.join(args.output_dir, "checkpoint-*")),
            key=lambda x: int(x.split("-")[-1]),
        )
        resume_path = checkpoint_dirs[-1] if checkpoint_dirs else None
        if resume_path:
            print(f"Auto-detected latest checkpoint to resume from: {resume_path}")
        else:
            print("No checkpoints found to resume from. Starting from scratch.")

    trainer.train(resume_from_checkpoint=resume_path)
    print("Training completed successfully!")


if __name__ == "__main__":
    main()
