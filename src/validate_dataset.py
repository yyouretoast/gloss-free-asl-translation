from __future__ import annotations
import os
import json
import numpy as np
from typing import Optional, Dict, Any, Tuple

from src.utils.io_utils import discover_landmark_paths

def _compute_missing_rates(left_hand: np.ndarray, right_hand: np.ndarray, face: np.ndarray, num_frames: int) -> Tuple[float, float, float, np.ndarray, np.ndarray, np.ndarray]:
    """Computes tracking failure rates based on all-zero frames."""
    left_hand_missing = np.all(left_hand == 0, axis=(1, 2))
    right_hand_missing = np.all(right_hand == 0, axis=(1, 2))
    face_missing = np.all(face == 0, axis=(1, 2))
    
    left_hand_missing_pct = (np.sum(left_hand_missing) / num_frames) * 100.0
    right_hand_missing_pct = (np.sum(right_hand_missing) / num_frames) * 100.0
    face_missing_pct = (np.sum(face_missing) / num_frames) * 100.0
    
    return left_hand_missing_pct, right_hand_missing_pct, face_missing_pct, left_hand_missing, right_hand_missing, face_missing

def _compute_wrist_noise(pose: np.ndarray, left_hand_missing: np.ndarray, right_hand_missing: np.ndarray, left_wrist_idx: int, right_wrist_idx: int) -> float:
    """Estimates scale-normalized coordinate noise based on frame-to-frame wrist jitter."""
    num_frames = pose.shape[0]
    if num_frames <= 1:
        return 0.0
        
    # Calculate shoulder width to normalize coordinate delta
    shoulder_idx_1 = 11
    shoulder_idx_2 = 12
        
    shoulder_diff = pose[:, shoulder_idx_1, :2] - pose[:, shoulder_idx_2, :2]
    shoulder_widths = np.linalg.norm(shoulder_diff, axis=-1)
    mean_shoulder_width = float(np.mean(shoulder_widths))
    # Use 0.05 threshold to match dataset.py
    mean_shoulder_width = max(0.05, mean_shoulder_width)

    # Extract left and right wrist (x, y, z)
    left_wrists = pose[:, left_wrist_idx, :3]
    right_wrists = pose[:, right_wrist_idx, :3]
    
    # Calculate frame-to-frame Euclidean distance
    left_deltas = np.linalg.norm(np.diff(left_wrists, axis=0), axis=1)
    right_deltas = np.linalg.norm(np.diff(right_wrists, axis=0), axis=1)
    
    # Average delta across frames that have valid movement.
    valid_left_mask = (~left_hand_missing[:-1]) & (~left_hand_missing[1:])
    valid_right_mask = (~right_hand_missing[:-1]) & (~right_hand_missing[1:])
    
    valid_left = left_deltas[valid_left_mask]
    valid_right = right_deltas[valid_right_mask]
    
    all_deltas = []
    if len(valid_left) > 0:
        all_deltas.extend(valid_left)
    if len(valid_right) > 0:
        all_deltas.extend(valid_right)
        
    if all_deltas:
        return float(np.mean(all_deltas)) / mean_shoulder_width
    return 0.0

def _extract_signer_id(file_path: str) -> str:
    """Extracts Signer ID from filename (assuming format: signerID_videoID.npz or similar)."""
    filename = os.path.basename(file_path)
    parts = filename.split('_')
    return parts[0] if len(parts) > 1 else 'unknown'

def analyze_single_file(file_path: str) -> Optional[Dict[str, Any]]:
    """
    Analyzes a single landmark .npz file or .npy file to compute statistics.
    """
    try:
        if file_path.endswith('.npy'):
            from src.utils.io_utils import load_holistic_npy
            landmarks = load_holistic_npy(file_path)
            pose = landmarks['pose']
            left_hand = landmarks['left_hand']
            right_hand = landmarks['right_hand']
            face = landmarks['face']
        else:
            with np.load(file_path) as data:
                # Check keys
                required_keys = {'pose', 'left_hand', 'right_hand', 'face'}
                if not required_keys.issubset(data.files):
                    print(f"Warning: {file_path} is missing keys. Found: {list(data.files)}")
                    return None

                pose = data['pose']
                left_hand = data['left_hand']
                right_hand = data['right_hand']
                face = data['face']
    except (IOError, OSError, ValueError, KeyError) as e:
        import warnings
        warnings.warn(f"Skipping {file_path}: {e}")
        return None
    
    left_wrist_pose_idx = 15
    right_wrist_pose_idx = 16

    num_frames = pose.shape[0]
    signer_id = _extract_signer_id(file_path)

    if num_frames == 0:
        return {
            'num_frames': 0,
            'left_hand_missing_pct': 100.0,
            'right_hand_missing_pct': 100.0,
            'face_missing_pct': 100.0,
            'avg_wrist_noise': 0.0,
            'signer_id': signer_id
        }

    lh_pct, rh_pct, face_pct, lh_miss, rh_miss, face_miss = _compute_missing_rates(
        left_hand, right_hand, face, num_frames
    )
    
    avg_wrist_noise = _compute_wrist_noise(
        pose, lh_miss, rh_miss, left_wrist_pose_idx, right_wrist_pose_idx
    )

    return {
        'num_frames': num_frames,
        'left_hand_missing_pct': lh_pct,
        'right_hand_missing_pct': rh_pct,
        'face_missing_pct': face_pct,
        'avg_wrist_noise': avg_wrist_noise,
        'signer_id': signer_id
    }

def validate_dataset(directory_path: str, limit: int = 50) -> Optional[Dict[str, Any]]:
    """
    Scans a directory of landmark .npz or .npy files and computes aggregate statistics.
    """
    files = discover_landmark_paths(directory_path)
            
    if not files:
        print(f"No coordinate files (.npz or .npy) found in {directory_path}")
        return None
        
    if limit and len(files) > limit:
        print(f"Dataset contains {len(files)} files/folders. Limiting analysis to first {limit} items for speed.")
        files = files[:limit]

    print(f"Scanning {len(files)} files in {directory_path}...\n")
    
    lengths = []
    lh_missing_pcts = []
    rh_missing_pcts = []
    face_missing_pcts = []
    noises = []
    signer_ids = set()
    
    for f in files:
        stats = analyze_single_file(f)
        if stats is not None:
            lengths.append(stats['num_frames'])
            lh_missing_pcts.append(stats['left_hand_missing_pct'])
            rh_missing_pcts.append(stats['right_hand_missing_pct'])
            face_missing_pcts.append(stats['face_missing_pct'])
            noises.append(stats['avg_wrist_noise'])
            signer_ids.add(stats['signer_id'])

    if not lengths:
        print("No valid files processed.")
        return None

    # Aggregate stats
    lengths_arr = np.array(lengths)
    lh_missing_arr = np.array(lh_missing_pcts)
    rh_missing_arr = np.array(rh_missing_pcts)
    face_missing_arr = np.array(face_missing_pcts)
    noises_arr = np.array(noises)

    print("=" * 60)
    print("           LANDMARK DATASET VALIDATION REPORT           ")
    print("=" * 60)
    print(f"Total Videos Profiled:     {len(files)}")
    print(f"Total Signers Identified:  {len(signer_ids)}")
    print(f"Signer IDs:                {sorted(list(signer_ids))}")
    print("-" * 60)
    print("Sequence Lengths (Frames):")
    print(f"  Min:                     {lengths_arr.min()}")
    print(f"  Max:                     {lengths_arr.max()}")
    print(f"  Mean:                    {lengths_arr.mean():.1f}")
    print(f"  Median:                  {np.median(lengths_arr):.1f}")
    print(f"  Std Dev:                 {lengths_arr.std():.1f}")
    print("-" * 60)
    print("Tracking Failure Rates (All-Zero Frames):")
    print(f"  Left Hand Missing:       {lh_missing_arr.mean():.1f}% avg (max: {lh_missing_arr.max():.1f}%)")
    print(f"  Right Hand Missing:      {rh_missing_arr.mean():.1f}% avg (max: {rh_missing_arr.max():.1f}%)")
    print(f"  Face Mesh Missing:       {face_missing_arr.mean():.1f}% avg (max: {face_missing_arr.max():.1f}%)")
    print("-" * 60)
    print("Landmark Coordinate Noise (Wrist frame-to-frame delta):")
    print(f"  Mean Wrist Jitter:       {noises_arr.mean():.5f}")
    print("=" * 60)
    
    return {
        'total_files': len(files),
        'signers': sorted(list(signer_ids)),
        'lengths_mean': lengths_arr.mean(),
        'lengths_std': lengths_arr.std(),
        'lh_missing_mean': lh_missing_arr.mean(),
        'rh_missing_mean': rh_missing_arr.mean(),
        'face_missing_mean': face_missing_arr.mean(),
        'wrist_noise_mean': noises_arr.mean(),
    }

def generate_mock_dataset(directory_path: str, num_files: int = 5) -> None:
    """
    Generates dummy/mock landmark .npz files for local testing.
    """
    os.makedirs(directory_path, exist_ok=True)
    print(f"Generating {num_files} mock landmark files for local validation test...")
    
    rng = np.random.default_rng(42)
    signers = ['signer01', 'signer02', 'signer03']
    
    for i in range(num_files):
        num_frames = rng.integers(60, 180)
        signer = rng.choice(signers)
        filename = f"{signer}_video_{i:04d}.npz"
        filepath = os.path.join(directory_path, filename)
        
        # 1. Pose landmarks (33, 4) - simulate slight motion with noise
        pose = rng.random((num_frames, 33, 4), dtype=np.float32) * 0.1 + 0.5
        
        # 2. Hands (21, 3) - simulate some dropouts
        left_hand = rng.random((num_frames, 21, 3), dtype=np.float32) * 0.05 + 0.3
        right_hand = rng.random((num_frames, 21, 3), dtype=np.float32) * 0.05 + 0.7
        
        # Inject random tracking dropouts (set frames to zero)
        # Left hand missing 30% of the time, Right hand missing 10%
        lh_dropout_mask = rng.random(num_frames) < 0.3
        rh_dropout_mask = rng.random(num_frames) < 0.1
        
        left_hand[lh_dropout_mask] = 0
        right_hand[rh_dropout_mask] = 0
        
        # 3. Face (92, 3) - standard face expression coordinates
        face = rng.random((num_frames, 92, 3), dtype=np.float32) * 0.1 + 0.5
        # Inject occasional face dropout (5% of frames)
        face_dropout_mask = rng.random(num_frames) < 0.05
        face[face_dropout_mask] = 0
        
        np.savez_compressed(filepath, pose=pose, left_hand=left_hand, right_hand=right_hand, face=face)
    print("Mock generation complete.")

def main() -> None:
    import argparse
    parser = argparse.ArgumentParser(description="Validate ASL landmark dataset stats.")
    parser.add_argument("--data_dir", type=str, default="data/landmarks", help="Path to landmark files.")
    parser.add_argument("--limit", type=int, default=50, help="Maximum number of files/folders to profile.")
    args = parser.parse_args()
    
    # Resolve absolute path
    dir_path = os.path.abspath(args.data_dir)
    
    # Check if there are any npz files or subdirectories
    files = discover_landmark_paths(dir_path)
        
    # Generate mock files if path is empty/doesn't exist for quick local verification
    if not os.path.exists(dir_path) or not files:
        # Gracefully handle read-only partitions (like Kaggle input datasets) to prevent write PermissionErrors
        if dir_path.startswith('/kaggle/input') or not os.access(os.path.dirname(dir_path) or '.', os.W_OK):
            import sys
            print(f"Error: Dataset directory {dir_path} does not exist and is in a read-only partition.")
            sys.exit(1)
        generate_mock_dataset(dir_path, num_files=5)
        
    validate_dataset(dir_path, limit=args.limit)

if __name__ == "__main__":
    main()
