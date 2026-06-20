import os
import torch
import torch.nn as nn
from transformers import (
    T5TokenizerFast, 
    T5ForConditionalGeneration, 
    Seq2SeqTrainer, 
    Seq2SeqTrainingArguments
)
from src.dataset import ASLLandmarkDataset, CollateLandmarks
from src.models.manual_encoder import ConformerEncoder

class ASLTranslationModel(nn.Module):
    """
    Sequence-to-sequence model connecting the custom Conformer Encoder
    directly to a pretrained T5 Decoder.
    """
    def __init__(self, input_dim=534, d_model=512, t5_model_name='t5-small', num_layers=4, num_heads=4, kernel_size=31):
        super().__init__()
        # 1. Custom Conformer Encoder
        self.encoder = ConformerEncoder(
            input_dim=input_dim,
            d_model=d_model,
            num_layers=num_layers,
            num_heads=num_heads,
            kernel_size=kernel_size
        )
        
        # 2. Pretrained T5 Decoder
        self.t5 = T5ForConditionalGeneration.from_pretrained(t5_model_name)
        
        # Ensure T5 projection dims match our d_model
        # t5-small has d_model=512, t5-base has d_model=768
        t5_d_model = self.t5.config.d_model
        if d_model != t5_d_model:
            self.dimension_projection = nn.Linear(d_model, t5_d_model)
        else:
            self.dimension_projection = nn.Identity()

        # Expose generation config for Seq2SeqTrainer
        self.generation_config = self.t5.generation_config

    def forward(self, input_features, attention_mask=None, labels=None, decoder_attention_mask=None, **kwargs):
        # Extract visual sequence representation
        # outputs shape: (batch, seq_len // 2, d_model)
        outputs, downsampled_mask = self.encoder(input_features, attention_mask=attention_mask)
        
        # Project dimensions if needed (Identity if d_model == t5_d_model)
        outputs = self.dimension_projection(outputs)
        
        # Pass visual embeddings directly as T5 encoder output hidden states
        encoder_outputs = (outputs,)
        
        # T5 expects 1/True for valid, 0/False for padding frames
        t5_attention_mask = (~downsampled_mask).long() if downsampled_mask is not None else None
        
        # Run T5 Decoder forward pass
        return self.t5(
            encoder_outputs=encoder_outputs,
            attention_mask=t5_attention_mask,
            labels=labels,
            decoder_attention_mask=decoder_attention_mask
        )

    def generate(self, input_features, attention_mask=None, **kwargs):
        """
        Overrides generate to enable visual-to-text sequence generation.
        Used during evaluation/inference.
        """
        with torch.no_grad():
            outputs, downsampled_mask = self.encoder(input_features, attention_mask=attention_mask)
            outputs = self.dimension_projection(outputs)
            encoder_outputs = (outputs,)
            
            # T5 expects 1/True for valid, 0/False for padding frames
            t5_attention_mask = (~downsampled_mask).long() if downsampled_mask is not None else None
            
            # Clean kwargs that are not accepted by T5's generate method
            kwargs.pop('file_ids', None)
            kwargs.pop('labels', None)
            kwargs.pop('decoder_attention_mask', None)
            
            # Use T5's auto-regressive generation
            return self.t5.generate(
                encoder_outputs=encoder_outputs,
                attention_mask=t5_attention_mask,
                bos_token_id=self.t5.config.decoder_start_token_id,
                **kwargs
            )

    def state_dict(self, *args, destination=None, prefix='', keep_vars=False):
        """
        Overrides state_dict to clone all tensors, preventing safetensors from raising
        a RuntimeError about shared memory tensors (like T5 embed_tokens/shared weights).
        """
        sd = super().state_dict(*args, destination=destination, prefix=prefix, keep_vars=keep_vars)
        return {k: v.clone() for k, v in sd.items()}

def main():
    import argparse
    parser = argparse.ArgumentParser(description="Train ASL Landmark to English translation model.")
    parser.add_argument("--data_dir", type=str, default="data/landmarks", help="Path to landmark .npz directory")
    parser.add_argument("--epochs", type=int, default=10, help="Number of training epochs")
    parser.add_argument("--batch_size", type=int, default=4, help="Batch size per device")
    parser.add_argument("--lr", type=float, default=1e-4, help="Learning rate")
    parser.add_argument("--output_dir", type=str, default="results/checkpoints", help="Output checkpoints folder")
    args = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Training on device: {device}")

    # 1. Initialize tokenizer
    t5_model_name = 't5-small'
    print(f"Loading tokenizer: {t5_model_name}")
    tokenizer = T5TokenizerFast.from_pretrained(t5_model_name)

    # 2. Setup mock metadata dict (For actual run on Kaggle, parse the real CSV file)
    # Mapping coordinates files to English sentences
    mock_metadata = {
        'signer01_video_0000': "hello",
        'signer01_video_0001': "please thank you",
        'signer01_video_0002': "good morning",
        'signer03_video_0003': "how are you",
        'signer01_video_0004': "sign language"
    }

    # 3. Create Dataset
    print(f"Loading dataset from: {args.data_dir}")
    dataset = ASLLandmarkDataset(
        data_dir=args.data_dir,
        metadata_dict=mock_metadata,
        max_len=150,
        include_face=True  # 534 input dimensions
    )

    # Split dataset into train and validation
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = torch.utils.data.random_split(
        dataset, [train_size, val_size], generator=torch.Generator().manual_seed(42)
    )
    print(f"Splits: {len(train_dataset)} train, {len(val_dataset)} validation samples.")

    # 4. Collator
    # Pass tokenizer to collator to convert English targets to token IDs
    collate_fn = CollateLandmarks(tokenizer=tokenizer, max_target_len=30)

    # 5. Initialize Model
    # input_dim=534, d_model=512 matches t5-small config d_model
    print("Initializing Model (Conformer -> T5-Small)...")
    model = ASLTranslationModel(
        input_dim=534,
        d_model=512,
        t5_model_name=t5_model_name,
        num_layers=4,
        num_heads=4,
        kernel_size=31
    )

    # 6. Define Seq2Seq Training Arguments
    training_args = Seq2SeqTrainingArguments(
        output_dir=args.output_dir,
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        learning_rate=args.lr,
        weight_decay=0.01,
        logging_dir=os.path.join(args.output_dir, "logs"),
        logging_steps=1,
        eval_strategy="epoch",
        save_strategy="epoch",
        predict_with_generate=True,  # Enables generating actual text during eval
        fp16=torch.cuda.is_available(), # Use mixed precision if GPU available
        report_to="none",  # Prevents wandb prompts on Kaggle
        remove_unused_columns=False
    )

    # 7. Define metrics computation
    import numpy as np
    import jiwer
    import sacrebleu

    def compute_metrics(eval_preds):
        preds, labels = eval_preds
        if isinstance(preds, tuple):
            preds = preds[0]
        
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

    # 8. Initialize Trainer
    trainer = Seq2SeqTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        data_collator=collate_fn,
        processing_class=tokenizer,
        compute_metrics=compute_metrics
    )

    # 8. Start Training Loop
    print("\nStarting training loop...")
    trainer.train()
    print("Training completed successfully!")

if __name__ == "__main__":
    main()
