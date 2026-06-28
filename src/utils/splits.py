"""
Dataset splitting utilities.
"""
from __future__ import annotations


import os
import random
from collections import defaultdict
from typing import List, Dict, Tuple

def split_by_signer(filepaths: List[str], video_to_signer: Dict[str, str]) -> Tuple[List[str], List[str]]:
    """Partition files strictly by Signer ID (Signer-Independent splits)."""
    signer_groups = defaultdict(list)
    unknown_files = []
    
    for filepath in filepaths:
        basename = os.path.splitext(os.path.basename(filepath))[0]
        clean_basename = basename.replace('_holistic', '').replace('_landmarks', '')
        signer_id = None
        
        # Try metadata mapping.
        if clean_basename in video_to_signer:
            signer_id = str(video_to_signer[clean_basename]).strip()
            
        # Infer from filename prefix.
        if not signer_id:
            parts = basename.split('_')
            if len(parts) > 1 and parts[0].isalnum() and parts[0].lower() not in ['video', 'sentence', 'segment', 'clip']:
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
        
        # Sort signers by size descending for a more balanced split.
        sorted_signers_by_size = sorted(sorted_signers, key=lambda s: len(signer_groups[s]), reverse=True)
        
        for signer in sorted_signers_by_size:
            files = signer_groups[signer]
            # Greedily allocate signers to maintain ~80/20 train/val ratio.
            if len(train_files) == 0 or (len(train_files) + len(files)) / total_known_count <= 0.85:
                train_files.extend(files)
            else:
                val_files.extend(files)
                
        # Place unknown-signer files in training to avoid validation leakage.
        train_files.extend(unknown_files)
        
        # Split training if validation set is empty.
        if len(val_files) == 0 and len(train_files) > 1:
            print("\nWARNING: Signer-based split left validation set empty. Splitting train files 80/20 to populate validation.")
            rng = random.Random(42)
            shuffled_train = list(train_files)
            rng.shuffle(shuffled_train)
            split_idx = int(0.8 * len(shuffled_train))
            train_files = shuffled_train[:split_idx]
            val_files = shuffled_train[split_idx:]
            
        print(f"Signer splits: {len(train_files)} train (includes {len(unknown_files)} unknown-signer clips), {len(val_files)} validation files.")
    else:
        # Fallback to random split if signer info is unavailable.
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
