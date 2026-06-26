"""
ASL Translation Model Training Orchestrator.
"""

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
    EarlyStoppingCallback
)

from src.dataset import ASLLandmarkDataset, CollateLandmarks
from src.models.translation_model import ASLTranslationModel
from src.utils.metadata import load_metadata
from src.utils.splits import split_by_signer

def main():
    parser = argparse.ArgumentParser(description="Train ASL Landmark to English translation model.")
    parser.add_argument("--data_dir", type=str, default="data/landmarks", help="Path to landmark .npz directory")
    parser.add_argument("--epochs", type=int, default=10, help="Number of training epochs")
    parser.add_argument("--batch_size", type=int, default=4, help="Batch size per device")
    parser.add_argument("--lr", type=float, default=1e-4, help="Learning rate")
    parser.add_argument("--output_dir", type=str, default="results/checkpoints", help="Output checkpoints folder")
    parser.add_argument("--metadata_file", type=str, default=None, help="Path to metadata CSV/TSV file mapping video IDs to translations")
    parser.add_argument("--no_face", action="store_true", help="Disable facial expression landmarks (for ablation study)")
    parser.add_argument("--max_len", type=int, default=150, help="Maximum frame sequence length (caps longer sequences to prevent OOM)")
    parser.add_argument("--max_target_len", type=int, default=30, help="Maximum target text sequence token length")
    parser.add_argument("--resume_from_checkpoint", type=str, default=None, help="Path to checkpoint folder to resume training from, or 'latest' to auto-detect")
    args = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Training on device: {device}")

    # 1. Initialize tokenizer
    t5_model_name = 't5-small'
    print(f"Loading tokenizer: {t5_model_name}")
    tokenizer = T5TokenizerFast.from_pretrained(t5_model_name)

    # 2. Setup metadata mapping and signer info
    metadata, video_to_signer = load_metadata(args.metadata_file)

    # 3. Create complete Dataset to extract file list
    include_face = not args.no_face
    print(f"Loading dataset from: {args.data_dir} (include_face={include_face})")
    full_dataset = ASLLandmarkDataset(
        data_dir=args.data_dir,
        metadata_dict=metadata,
        max_len=args.max_len,
        include_face=include_face
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
        include_face=include_face
    )
    val_dataset = ASLLandmarkDataset(
        data_dir=args.data_dir,
        metadata_dict=metadata,
        file_list=val_files,
        max_len=args.max_len,
        include_face=include_face
    )

    # 5. Collator
    # Pass tokenizer to collator to convert English targets to token IDs
    collate_fn = CollateLandmarks(tokenizer=tokenizer, max_target_len=args.max_target_len)

    # 6. Initialize Model dynamically using input shape
    sample_batch = train_dataset[0]
    input_dim = sample_batch['features'].shape[1]
    print(f"Initializing Model (Conformer -> T5-Small) with input_dim={input_dim}...")
    model = ASLTranslationModel(
        input_dim=input_dim,
        d_model=512,
        t5_model_name=t5_model_name,
        num_layers=4,
        num_heads=4,
        kernel_size=31
    )
    # Set model's autoregressive generation config maximum length to avoid T5 default 20-token truncation
    model.generation_config.max_length = args.max_target_len

    # Calculate warmup steps dynamically based on dataset size and epochs
    steps_per_epoch = len(train_dataset) // args.batch_size
    if steps_per_epoch == 0:
        steps_per_epoch = 1
    total_training_steps = steps_per_epoch * args.epochs
    warmup_steps = int(0.1 * total_training_steps)

    # 7. Define Seq2Seq Training Arguments
    training_args = Seq2SeqTrainingArguments(
        output_dir=args.output_dir,
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        learning_rate=args.lr,
        weight_decay=0.01,
        logging_dir=os.path.join(args.output_dir, "logs"),
        logging_steps=10,             # Updated to 10 per requirements
        eval_strategy="epoch",
        save_strategy="epoch",
        save_total_limit=3,          # Limits checkpoints to keep only the 3 most recent, preventing disk overflow
        predict_with_generate=True,  # Enables generating actual text during eval
        generation_max_length=30,    # Added per requirements
        fp16=torch.cuda.is_available(), # Use mixed precision if GPU available
        report_to="none",  # Prevents wandb prompts on Kaggle
        remove_unused_columns=False,
        warmup_steps=warmup_steps,    # Dynamic warmup steps to stabilize training early
        load_best_model_at_end=True,  # Added per requirements
        metric_for_best_model="wer",  # Added per requirements
        
        # --- Performance & Memory Enhancements ---
        optim="adafactor",                  # Native T5 optimizer, saves massive VRAM
        gradient_checkpointing=True,        # Reduces activation memory footprint
        dataloader_num_workers=2,           # Parallelizes CPU data fetching
        dataloader_pin_memory=True          # Accelerates CPU-to-GPU data transfers
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
        
        # Clean whitespaces
        decoded_preds = [p.strip() for p in decoded_preds]
        decoded_labels = [l.strip() for l in decoded_labels]
        
        # Avoid empty strings causing jiwer errors
        decoded_preds = [p if p else " " for p in decoded_preds]
        decoded_labels = [l if l else " " for l in decoded_labels]
        
        wer = jiwer.wer(decoded_labels, decoded_preds)
        bleu = sacrebleu.corpus_bleu(decoded_preds, [decoded_labels]).score
        
        return {
            "bleu": float(bleu),
            "wer": float(wer)
        }

    # 9. Initialize Trainer
    trainer = Seq2SeqTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        data_collator=collate_fn,
        processing_class=tokenizer,
        compute_metrics=compute_metrics,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=3)] # Added per requirements
    )

    # 10. Start Training Loop
    print("\nStarting training loop...")
    resume_path = args.resume_from_checkpoint
    if resume_path == "latest":
        # Auto-detect latest checkpoint under output_dir
        checkpoint_dirs = sorted(glob.glob(os.path.join(args.output_dir, "checkpoint-*")), key=lambda x: int(x.split("-")[-1]))
        resume_path = checkpoint_dirs[-1] if checkpoint_dirs else None
        if resume_path:
            print(f"Auto-detected latest checkpoint to resume from: {resume_path}")
        else:
            print("No checkpoints found to resume from. Starting from scratch.")
            
    trainer.train(resume_from_checkpoint=resume_path)
    print("Training completed successfully!")

if __name__ == "__main__":
    main()
