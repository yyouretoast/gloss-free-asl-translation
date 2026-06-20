import os
import glob
import numpy as np

def analyze_single_file(file_path):
    """
    Analyzes a single landmark .npz file or OpenPose directory to compute statistics.
    """
    if os.path.isdir(file_path):
        import json
        json_files = sorted(glob.glob(os.path.join(file_path, "*.json")))
        num_frames = len(json_files)
        if num_frames == 0:
            return None
        
        pose_list, left_hand_list, right_hand_list, face_list = [], [], [], []
        for jf in json_files:
            try:
                with open(jf, 'r') as f:
                    data = json.load(f)
                people = data.get('people', [])
                if people:
                    p = people[0]
                    pose_arr = np.array(p.get('pose_keypoints_2d', [])).reshape(25, 3)
                    face_arr = np.array(p.get('face_keypoints_2d', [])).reshape(70, 3)
                    lh_arr = np.array(p.get('hand_left_keypoints_2d', [])).reshape(21, 3)
                    rh_arr = np.array(p.get('hand_right_keypoints_2d', [])).reshape(21, 3)
                else:
                    pose_arr = np.zeros((25, 3))
                    face_arr = np.zeros((70, 3))
                    lh_arr = np.zeros((21, 3))
                    rh_arr = np.zeros((21, 3))
                pose_list.append(pose_arr)
                face_list.append(face_arr)
                left_hand_list.append(lh_arr)
                right_hand_list.append(rh_arr)
            except Exception:
                return None
        
        pose = np.stack(pose_list, axis=0)
        left_hand = np.stack(left_hand_list, axis=0)
        right_hand = np.stack(right_hand_list, axis=0)
        face = np.stack(face_list, axis=0)
        
        left_wrist_pose_idx = 7
        right_wrist_pose_idx = 4
    else:
        try:
            data = np.load(file_path)
        except Exception as e:
            print(f"Error loading {file_path}: {e}")
            return None

        # Check keys
        required_keys = {'pose', 'left_hand', 'right_hand', 'face'}
        if not required_keys.issubset(data.files):
            print(f"Warning: {file_path} is missing keys. Found: {list(data.files)}")
            return None

        pose = data['pose']
        left_hand = data['left_hand']
        right_hand = data['right_hand']
        face = data['face']
        
        left_wrist_pose_idx = 15
        right_wrist_pose_idx = 16

    num_frames = pose.shape[0]
    if num_frames == 0:
        return {
            'num_frames': 0,
            'left_hand_missing_pct': 100.0,
            'right_hand_missing_pct': 100.0,
            'face_missing_pct': 100.0,
            'avg_wrist_noise': 0.0,
            'signer_id': 'unknown'
        }

    # 1. Check hand missing tracking (all zeros in a frame)
    left_hand_missing = np.all(left_hand == 0, axis=(1, 2))
    right_hand_missing = np.all(right_hand == 0, axis=(1, 2))
    
    left_hand_missing_pct = (np.sum(left_hand_missing) / num_frames) * 100.0
    right_hand_missing_pct = (np.sum(right_hand_missing) / num_frames) * 100.0

    # 2. Check face missing tracking
    face_missing = np.all(face == 0, axis=(1, 2))
    face_missing_pct = (np.sum(face_missing) / num_frames) * 100.0

    # 3. Estimate coordinate noise (average delta of wrist joints frame-to-frame)
    avg_wrist_noise = 0.0
    if num_frames > 1:
        # Extract left and right wrist (x, y, z)
        left_wrists = pose[:, left_wrist_pose_idx, :3]
        right_wrists = pose[:, right_wrist_pose_idx, :3]
        
        # Calculate frame-to-frame Euclidean distance
        left_deltas = np.linalg.norm(np.diff(left_wrists, axis=0), axis=1)
        right_deltas = np.linalg.norm(np.diff(right_wrists, axis=0), axis=1)
        
        # Average delta across frames that have valid movement (avoiding zero transitions if tracking dropped)
        valid_left = left_deltas[~left_hand_missing[:-1]]
        valid_right = right_deltas[~right_hand_missing[:-1]]
        
        all_deltas = []
        if len(valid_left) > 0:
            all_deltas.extend(valid_left)
        if len(valid_right) > 0:
            all_deltas.extend(valid_right)
            
        if all_deltas:
            avg_wrist_noise = float(np.mean(all_deltas))

    # 4. Extract Signer ID from filename (assuming format: signerID_videoID.npz or similar)
    filename = os.path.basename(file_path)
    parts = filename.split('_')
    signer_id = parts[0] if len(parts) > 1 else 'unknown'

    return {
        'num_frames': num_frames,
        'left_hand_missing_pct': left_hand_missing_pct,
        'right_hand_missing_pct': right_hand_missing_pct,
        'face_missing_pct': face_missing_pct,
        'avg_wrist_noise': avg_wrist_noise,
        'signer_id': signer_id
    }

def validate_dataset(directory_path):
    """
    Scans a directory of landmark .npz files or OpenPose JSON directories and prints aggregate statistics.
    """
    search_path = os.path.join(directory_path, "*.npz")
    files = glob.glob(search_path)
    
    # If no .npz files, try finding directories of JSONs (OpenPose structure)
    if not files:
        json_dirs = glob.glob(os.path.join(directory_path, "**/openpose_output/json/*"), recursive=True)
        files = [d for d in json_dirs if os.path.isdir(d)]
        
        # Fallback to directories directly inside directory_path
        if not files and os.path.exists(directory_path):
            files = [os.path.join(directory_path, d) for d in os.listdir(directory_path) if os.path.isdir(os.path.join(directory_path, d))]
            
    if not files:
        print(f"No coordinate files (.npz) or OpenPose directories found in {directory_path}")
        return False

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
        return False

    # Aggregate stats
    lengths = np.array(lengths)
    lh_missing_pcts = np.array(lh_missing_pcts)
    rh_missing_pcts = np.array(rh_missing_pcts)
    face_missing_pcts = np.array(face_missing_pcts)
    noises = np.array(noises)

    print("=" * 60)
    print("           LANDMARK DATASET VALIDATION REPORT           ")
    print("=" * 60)
    print(f"Total Videos Profiled:     {len(files)}")
    print(f"Total Signers Identified:  {len(signer_ids)}")
    print(f"Signer IDs:                {sorted(list(signer_ids))}")
    print("-" * 60)
    print("Sequence Lengths (Frames):")
    print(f"  Min:                     {lengths.min()}")
    print(f"  Max:                     {lengths.max()}")
    print(f"  Mean:                    {lengths.mean():.1f}")
    print(f"  Median:                  {np.median(lengths):.1f}")
    print(f"  Std Dev:                 {lengths.std():.1f}")
    print("-" * 60)
    print("Tracking Failure Rates (All-Zero Frames):")
    print(f"  Left Hand Missing:       {lh_missing_pcts.mean():.1f}% avg (max: {lh_missing_pcts.max():.1f}%)")
    print(f"  Right Hand Missing:      {rh_missing_pcts.mean():.1f}% avg (max: {rh_missing_pcts.max():.1f}%)")
    print(f"  Face Mesh Missing:       {face_missing_pcts.mean():.1f}% avg (max: {face_missing_pcts.max():.1f}%)")
    print("-" * 60)
    print("Landmark Coordinate Noise (Wrist frame-to-frame delta):")
    print(f"  Mean Wrist Jitter:       {noises.mean():.5f}")
    print("=" * 60)
    
    return True

def generate_mock_dataset(directory_path, num_files=5):
    """
    Generates dummy/mock landmark .npz files for local testing.
    """
    os.makedirs(directory_path, exist_ok=True)
    print(f"Generating {num_files} mock landmark files for local validation test...")
    
    np.random.seed(42)
    signers = ['signer01', 'signer02', 'signer03']
    
    for i in range(num_files):
        num_frames = np.random.randint(60, 180)
        signer = np.random.choice(signers)
        filename = f"{signer}_video_{i:04d}.npz"
        filepath = os.path.join(directory_path, filename)
        
        # 1. Pose landmarks (33, 4) - simulate slight motion with noise
        pose = np.random.rand(num_frames, 33, 4) * 0.1 + 0.5
        
        # 2. Hands (21, 3) - simulate some dropouts
        left_hand = np.random.rand(num_frames, 21, 3) * 0.05 + 0.3
        right_hand = np.random.rand(num_frames, 21, 3) * 0.05 + 0.7
        
        # Inject random tracking dropouts (set frames to zero)
        # Left hand missing 30% of the time, Right hand missing 10%
        lh_dropout_mask = np.random.rand(num_frames) < 0.3
        rh_dropout_mask = np.random.rand(num_frames) < 0.1
        
        left_hand[lh_dropout_mask] = 0
        right_hand[rh_dropout_mask] = 0
        
        # 3. Face (92, 3) - standard face expression coordinates
        face = np.random.rand(num_frames, 92, 3) * 0.1 + 0.5
        # Inject occasional face dropout (5% of frames)
        face_dropout_mask = np.random.rand(num_frames) < 0.05
        face[face_dropout_mask] = 0
        
        np.savez_compressed(filepath, pose=pose, left_hand=left_hand, right_hand=right_hand, face=face)
    print("Mock generation complete.")

def main():
    import argparse
    parser = argparse.ArgumentParser(description="Validate ASL landmark dataset stats.")
    parser.add_argument("--data_dir", type=str, default="data/landmarks", help="Path to landmark files.")
    args = parser.parse_args()
    
    # Resolve absolute path
    dir_path = os.path.abspath(args.data_dir)
    
    # Check if there are any npz files or subdirectories
    has_npz = len(glob.glob(os.path.join(dir_path, "*.npz"))) > 0
    has_subdirs = False
    if os.path.exists(dir_path):
        has_subdirs = any(os.path.isdir(os.path.join(dir_path, d)) for d in os.listdir(dir_path))
        
    # Generate mock files if path is empty/doesn't exist for quick local verification
    if not os.path.exists(dir_path) or (not has_npz and not has_subdirs):
        generate_mock_dataset(dir_path, num_files=5)
        
    validate_dataset(dir_path)

if __name__ == "__main__":
    main()
