"""
Metadata loading utilities.
"""
from __future__ import annotations


import os
import glob
import pandas as pd
from typing import Tuple, Dict

def load_metadata(metadata_file: str | None) -> Tuple[Dict[str, str], Dict[str, str]]:
    """Load metadata mapping video IDs to translation text and signer IDs."""
    metadata: Dict[str, str] = {}
    video_to_signer: Dict[str, str] = {}
    
    if metadata_file:
        print(f"Loading metadata from {metadata_file}")
        
        # Merge all split CSVs dynamically if one split is passed.
        if any(term in metadata_file for term in ['_train.', '_val.', '_test.']):
            dir_name = os.path.dirname(metadata_file)
            base_name = os.path.basename(metadata_file)
            
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
            if metadata_file.endswith('.tsv') or metadata_file.endswith('.txt'):
                df = pd.read_csv(metadata_file, sep=None, engine='python')
            else:
                sep = '\t' if 'realigned' in metadata_file else ','
                df = pd.read_csv(metadata_file, sep=sep)
            
        file_candidates = [c for c in df.columns if any(x in c.lower() for x in ['id', 'file', 'video', 'key', 'name'])]
        
        def file_col_priority(col: str) -> int:
            c_low = col.lower()
            checks = [
                'sentence' in c_low and 'name' in c_low,
                'segment' in c_low and 'name' in c_low,
                'file' in c_low and 'name' in c_low,
                'sentence' in c_low and 'id' in c_low,
                'segment' in c_low and 'id' in c_low,
                'file' in c_low and 'id' in c_low,
                'name' in c_low and 'video' not in c_low,
                'id' in c_low and 'video' not in c_low,
                'video' in c_low,
            ]
            return checks.index(True) if True in checks else len(checks)
            
        file_candidates.sort(key=file_col_priority)
        file_col = file_candidates
        
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
    return metadata, video_to_signer
