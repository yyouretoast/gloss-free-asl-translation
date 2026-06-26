"""Preprocesses How2Sign OpenPose JSON files into compressed .npz landmarks."""
from __future__ import annotations

import os
import argparse
import sys
import tempfile
from pathlib import Path
from multiprocessing import Pool, cpu_count
from typing import Tuple

import numpy as np
from tqdm import tqdm

try:
    import orjson
except ImportError:
    orjson = None
import json

def process_single_folder(args_tuple: Tuple[str, str]) -> bool:
    input_folder, output_dir = args_tuple
    try:
        basename = os.path.basename(input_folder)
        output_filepath = os.path.join(output_dir, f"{basename}.npz")
        
        json_files = sorted([os.path.join(input_folder, f) for f in os.listdir(input_folder) if f.endswith('.json')])
        num_frames = len(json_files)
        if num_frames == 0:
            return False
            
        pose_list, left_hand_list, right_hand_list, face_list = [], [], [], []
        
        for jf in json_files:
            with open(jf, 'rb') as f:
                content = f.read()
                if orjson is not None:
                    data = orjson.loads(content)
                else:
                    data = json.loads(content)
            
            people = data.get('people', [])
            if people:
                p = people[0]
                pose_raw = p.get('pose_keypoints_2d', [])
                pose_arr = np.array(pose_raw, dtype=np.float32).reshape(25, 3) if len(pose_raw) == 75 else np.zeros((25, 3), dtype=np.float32)
                
                face_raw = p.get('face_keypoints_2d', [])
                face_arr = np.array(face_raw, dtype=np.float32).reshape(70, 3) if len(face_raw) == 210 else np.zeros((70, 3), dtype=np.float32)
                
                lh_raw = p.get('hand_left_keypoints_2d', [])
                lh_arr = np.array(lh_raw, dtype=np.float32).reshape(21, 3) if len(lh_raw) == 63 else np.zeros((21, 3), dtype=np.float32)
                
                rh_raw = p.get('hand_right_keypoints_2d', [])
                rh_arr = np.array(rh_raw, dtype=np.float32).reshape(21, 3) if len(rh_raw) == 63 else np.zeros((21, 3), dtype=np.float32)
            else:
                pose_arr = np.zeros((25, 3), dtype=np.float32)
                face_arr = np.zeros((70, 3), dtype=np.float32)
                lh_arr = np.zeros((21, 3), dtype=np.float32)
                rh_arr = np.zeros((21, 3), dtype=np.float32)
                
            pose_list.append(pose_arr)
            face_list.append(face_arr)
            left_hand_list.append(lh_arr)
            right_hand_list.append(rh_arr)
            
        pose = np.stack(pose_list, axis=0)
        left_hand = np.stack(left_hand_list, axis=0)
        right_hand = np.stack(right_hand_list, axis=0)
        face = np.stack(face_list, axis=0)
        
        # Save as compressed npz to fit in RAM disk limits (/dev/shm)
        np.savez_compressed(output_filepath, pose=pose, left_hand=left_hand, right_hand=right_hand, face=face)
        return True
    except (IOError, json.JSONDecodeError, KeyError, ValueError) as e:
        print(f"Error processing {input_folder}: {e}", file=sys.stderr)
        return False

def main() -> None:
    parser = argparse.ArgumentParser(description="Preprocess How2Sign dataset from OpenPose JSON to .npz format.")
    parser.add_argument("--input-dir", type=str, default=None, help="Input directory containing OpenPose JSONs.")
    parser.add_argument("--output-dir", type=str, default=None, help="Output directory for .npz files.")
    parser.add_argument("--workers", type=int, default=None, help="Number of CPU workers for multiprocessing.")
    
    args = parser.parse_args()

    # 1. Resolve input directory
    input_dir = args.input_dir
    if not input_dir:
        for path in [
            '/kaggle/input/datasets/nazarboholii/how2sign',
            '/kaggle/input/how2sign',
            '/kaggle/input/how2sign-keypoints',
            '/kaggle/input/datasets/nazarboholii/how2sign-keypoints'
        ]:
            if os.path.exists(path):
                input_dir = path
                break
        else:
            input_dir = '/kaggle/input/datasets/nazarboholii/how2sign'
            
    json_cand = os.path.join(input_dir, "train_2D_keypoints/openpose_output/json")
    if not os.path.exists(json_cand):
        print(f"Error: Candidate OpenPose json folder {json_cand} does not exist!", file=sys.stderr)
        sys.exit(1)
        
    # 2. Resolve output directory 
    if args.output_dir:
        output_dir = args.output_dir
    else:
        output_dir = str(Path(tempfile.gettempdir()) / 'how2sign_npz')
        
    os.makedirs(output_dir, exist_ok=True)
    
    # 3. List folders to process
    print("Scanning OpenPose directories...")
    all_folders = [os.path.join(json_cand, d) for d in os.listdir(json_cand) if os.path.isdir(os.path.join(json_cand, d))]
    
    # Get set of already processed files
    existing_files = {os.path.splitext(f)[0] for f in os.listdir(output_dir) if f.endswith('.npz')}
    
    # Filter out already processed folders
    folders = [f for f in all_folders if os.path.basename(f) not in existing_files]
    print(f"Found {len(all_folders)} folders. {len(folders)} folders remaining to process.")
    
    # Prepare arguments for pool
    tasks = [(f, output_dir) for f in folders]
    
    # 4. Process in parallel using a CPU pool
    num_workers = args.workers if args.workers else min(cpu_count(), 4)
    print(f"Processing in parallel using {num_workers} CPU workers...")
    
    success_count = 0
    with Pool(num_workers) as pool:
        for result in tqdm(pool.imap_unordered(process_single_folder, tasks), total=len(tasks)):
            if result:
                success_count += 1
                
    print(f"\nPreprocessing completed! Successfully converted {success_count}/{len(folders)} videos to .npz.")
    print(f"Preprocessed dataset saved at: {output_dir}")

if __name__ == "__main__":
    main()
