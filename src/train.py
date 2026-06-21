import os
import torch
import torch.nn as nn
from collections import OrderedDict
from transformers import (
    T5TokenizerFast, 
    T5ForConditionalGeneration, 
    Seq2SeqTrainer, 
    Seq2SeqTrainingArguments
)
from transformers.modeling_outputs import BaseModelOutput
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
        encoder_outputs = BaseModelOutput(last_hidden_state=outputs)
        
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
            encoder_outputs = BaseModelOutput(last_hidden_state=outputs)
            
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
        return OrderedDict((k, v.clone()) for k, v in sd.items())

def main():
    import argparse
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
    args = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Training on device: {device}")

    # 1. Initialize tokenizer
    t5_model_name = 't5-small'
    print(f"Loading tokenizer: {t5_model_name}")
    tokenizer = T5TokenizerFast.from_pretrained(t5_model_name)

    # 2. Setup metadata mapping and signer info
    metadata = {}
    video_to_signer = {}
    
    if args.metadata_file:
        import pandas as pd
        import glob
        print(f"Loading metadata from {args.metadata_file}")
        
        # If metadata_file matches a split (e.g. "_train.csv"), merge all splits dynamically
        if '_train.' in args.metadata_file or '_val.' in args.metadata_file or '_test.' in args.metadata_file:
            dir_name = os.path.dirname(args.metadata_file)
            base_name = os.path.basename(args.metadata_file)
            
            # Identify wildcard pattern
            wildcard = base_name
            for term in ['_train', '_val', '_test']:
                if term in base_name:
                    wildcard = base_name.replace(term, '_*')
                    break
            pattern = os.path.join(dir_name, wildcard)
            csv_files = sorted(glob.glob(pattern))
            print(f"Detected split manifests. Merging: {csv_files}")
            
            dfs = []
            for f_path in csv_files:
                sep = '\t' if 'realigned' in f_path else None
                dfs.append(pd.read_csv(f_path, sep=sep, engine='python'))
            df = pd.concat(dfs, ignore_index=True)
        else:
            if args.metadata_file.endswith('.tsv') or args.metadata_file.endswith('.txt'):
                df = pd.read_csv(args.metadata_file, sep=None, engine='python')
            else:
                sep = '\t' if 'realigned' in args.metadata_file else ','
                df = pd.read_csv(args.metadata_file, sep=sep)
            
        # Detect appropriate identifier and label columns dynamically
        file_candidates = [c for c in df.columns if any(x in c.lower() for x in ['id', 'file', 'video', 'key', 'name'])]
        
        # Sort candidates to prefer segment/sentence/file specific IDs over video ID
        def file_col_priority(col):
            c_low = col.lower()
            if 'sentence' in c_low and 'id' in c_low:
                return 0
            if 'segment' in c_low and 'id' in c_low:
                return 1
            if 'file' in c_low and 'id' in c_low:
                return 2
            if 'id' in c_low and 'video' not in c_low:
                return 3
            if 'video' in c_low:
                return 4
            return 5
            
        file_candidates.sort(key=file_col_priority)
        file_col = file_candidates
        
        # Target/Text column: matches text, trans, gloss, sentence, caption, but NOT key/id/file/video/name words
        text_col = [c for c in df.columns if any(x in c.lower() for x in ['text', 'trans', 'gloss', 'sentence', 'caption'])
                    and not any(x in c.lower() for x in ['id', 'key', 'file', 'video', 'name'])]
        signer_col = [c for c in df.columns if any(x in c.lower() for x in ['signer', 'channel', 'uploader', 'author', 'subject'])]
        
        if file_col and text_col:
            f_col = file_col[0]
            t_col = text_col[0]
            print(f"Mapping columns: File ID '{f_col}' -> Text '{t_col}'")
            metadata = dict(zip(df[f_col].astype(str), df[t_col].astype(str)))
            if signer_col:
                s_col = signer_col[0]
                print(f"Mapping signer ID column: '{s_col}'")
                video_to_signer = dict(zip(df[f_col].astype(str), df[s_col].astype(str)))
        else:
            raise ValueError(f"Could not find matching columns. Columns: {df.columns.tolist()}")
    else:
        print("Warning: No metadata_file provided. Falling back to mock metadata.")
        metadata = {
            'signer01_video_0000': "hello",
            'signer01_video_0001': "please thank you",
            'signer01_video_0002': "good morning",
            'signer03_video_0003': "how are you",
            'signer01_video_0004': "sign language"
        }

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
    from collections import defaultdict
    signer_groups = defaultdict(list)
    unknown_files = []
    
    for filepath in full_dataset.filepaths:
        basename = os.path.splitext(os.path.basename(filepath))[0]
        signer_id = None
        
        # Try to retrieve signer from metadata mapping
        if basename in video_to_signer:
            signer_id = str(video_to_signer[basename]).strip()
            
        # Try to infer from filename prefix (e.g. "signer01_video_0000" -> "signer01")
        if not signer_id:
            parts = basename.split('_')
            if len(parts) > 1 and (parts[0].isalnum() or 'signer' in parts[0].lower() or 'channel' in parts[0].lower()):
                signer_id = parts[0]
                
        if signer_id and signer_id.lower() != 'unknown':
            signer_groups[signer_id].append(filepath)
        else:
            unknown_files.append(filepath)
            
    sorted_signers = sorted(list(signer_groups.keys()))
    train_files = []
    val_files = []
    
    if len(sorted_signers) > 0:
        total_known_count = sum(len(signer_groups[s]) for s in sorted_signers)
        target_train_count = 0.8 * total_known_count
        current_train_count = 0
        
        for signer in sorted_signers:
            files = signer_groups[signer]
            if current_train_count < target_train_count:
                train_files.extend(files)
                current_train_count += len(files)
            else:
                val_files.extend(files)
                
        # Drop unknown-signer files from validation/evaluation entirely to prevent leakage
        train_files.extend(unknown_files)
        
        # Safeguard: if there is only 1 signer or val_files is empty, split train_files to populate it
        if len(val_files) == 0 and len(train_files) > 1:
            print("\nWARNING: Signer-based split left validation set empty. Splitting train files 80/20 to populate validation.")
            import random
            # Use deterministic seed for reproducibility
            rng = random.Random(42)
            shuffled_train = list(train_files)
            rng.shuffle(shuffled_train)
            split_idx = int(0.8 * len(shuffled_train))
            train_files = shuffled_train[:split_idx]
            val_files = shuffled_train[split_idx:]
            
        print(f"Signer splits: {len(train_files)} train (includes {len(unknown_files)} unknown-signer clips), {len(val_files)} validation files.")
    else:
        # Fallback to standard random split if no signer proxy info is available
        print("\n" + "="*80)
        print("WARNING: No signer, channel, or uploader metadata found in filenames or CSV columns.")
        print("Falling back to standard random split. Note: validation metrics may suffer from signer data leakage.")
        print("="*80 + "\n")
        
        all_files = full_dataset.filepaths
        import random
        random.seed(42)
        shuffled = list(all_files)
        random.shuffle(shuffled)
        split_idx = int(0.8 * len(shuffled))
        train_files = shuffled[:split_idx]
        val_files = shuffled[split_idx:]
        print(f"Fallback splits: {len(train_files)} train, {len(val_files)} validation files.")
        
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
        remove_unused_columns=False,
        warmup_steps=warmup_steps     # Dynamic warmup steps to stabilize training early
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
