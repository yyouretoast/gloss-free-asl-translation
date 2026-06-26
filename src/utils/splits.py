"""
Dataset splitting utilities.
"""

import os
import random
from collections import defaultdict
from typing import List, Dict, Tuple

def split_by_signer(filepaths: List[str], video_to_signer: Dict[str, str]) -> Tuple[List[str], List[str]]:
    """
    Partitions files strictly by Signer ID (Signer-Independent splits).
    """
    signer_groups = defaultdict(list)
    unknown_files = []
    
    for filepath in filepaths:
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
        
        # Sort signers by dataset size descending to make the split more balanced
        sorted_signers_by_size = sorted(sorted_signers, key=lambda s: len(signer_groups[s]), reverse=True)
        
        for signer in sorted_signers_by_size:
            files = signer_groups[signer]
            # Greedily allocate signers to train/val to keep ratios close to 80/20,
            # ensuring validation doesn't end up empty or heavily starved.
            if len(train_files) == 0 or (len(train_files) + len(files)) / total_known_count <= 0.85:
                train_files.extend(files)
            else:
                val_files.extend(files)
                
        # Drop unknown-signer files from validation/evaluation entirely to prevent leakage
        train_files.extend(unknown_files)
        
        # Safeguard: if there is only 1 signer or val_files is empty, split train_files to populate it
        if len(val_files) == 0 and len(train_files) > 1:
            print("\nWARNING: Signer-based split left validation set empty. Splitting train files 80/20 to populate validation.")
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
        
        all_files = filepaths
        random.seed(42)
        shuffled = list(all_files)
        random.shuffle(shuffled)
        split_idx = int(0.8 * len(shuffled))
        train_files = shuffled[:split_idx]
        val_files = shuffled[split_idx:]
        print(f"Fallback splits: {len(train_files)} train, {len(val_files)} validation files.")
        
    return train_files, val_files
