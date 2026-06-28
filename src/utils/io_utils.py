"""Shared I/O utilities for landmark loading and directory discovery."""
from __future__ import annotations

import glob
import json
import os
from pathlib import Path

import numpy as np

# Optional fast JSON parser
try:
    import orjson
except ImportError:
    orjson = None


# OpenPose BODY_25 format constants
OPENPOSE_POSE_KEYPOINTS = 25
OPENPOSE_POSE_FEATURES = 75   # 25 * 3
OPENPOSE_FACE_KEYPOINTS = 70
OPENPOSE_FACE_FEATURES = 210  # 70 * 3
OPENPOSE_HAND_KEYPOINTS = 21
OPENPOSE_HAND_FEATURES = 63   # 21 * 3

# MediaPipe format constants
MEDIAPIPE_POSE_KEYPOINTS = 33
MEDIAPIPE_FACE_KEYPOINTS = 92
MEDIAPIPE_HAND_KEYPOINTS = 21

# 92 key facial landmark indices from the MediaPipe 468-point face mesh.
# These target expressive regions critical for ASL grammar (lips, eyes, eyebrows).
# Slices the 468 mesh landmarks down to a subset of 92 points.
MEDIAPIPE_FACE_SUBSET_INDICES = sorted(list(set(
    [61, 146, 91, 181, 84, 17, 314, 405, 321, 375, 291, 308, 324, 318, 402, 317, 14, 87, 178, 88, 95, 185, 40, 39, 37, 0, 267, 269, 270, 409, 415, 310, 311, 312, 13, 82, 81, 42, 183, 78] + # lips
    [362, 382, 381, 380, 374, 373, 390, 249, 263, 466, 388, 387, 386, 385, 384, 398] + # left eye
    [33, 7, 163, 144, 145, 153, 154, 155, 133, 173, 157, 158, 159, 160, 161, 246] + # right eye
    [336, 296, 334, 293, 300, 276, 283, 282, 295, 285] + # left eyebrow
    [70, 63, 105, 66, 107, 55, 65, 52, 53, 46] # right eyebrow
)))


def load_json(filepath: str | Path) -> dict:
    """Load JSON file."""
    with open(filepath, 'rb') as f:
        content = f.read()
    if orjson is not None:
        return orjson.loads(content)
    return json.loads(content)


def parse_openpose_frame(person: dict) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Parse single frame keypoints from OpenPose person dictionary."""
    pose_raw = person.get('pose_keypoints_2d', [])
    pose = (np.array(pose_raw, dtype=np.float32).reshape(OPENPOSE_POSE_KEYPOINTS, 3)
            if len(pose_raw) == OPENPOSE_POSE_FEATURES
            else np.zeros((OPENPOSE_POSE_KEYPOINTS, 3), dtype=np.float32))
    
    face_raw = person.get('face_keypoints_2d', [])
    face = (np.array(face_raw, dtype=np.float32).reshape(OPENPOSE_FACE_KEYPOINTS, 3)
            if len(face_raw) == OPENPOSE_FACE_FEATURES
            else np.zeros((OPENPOSE_FACE_KEYPOINTS, 3), dtype=np.float32))
    
    lh_raw = person.get('hand_left_keypoints_2d', [])
    left_hand = (np.array(lh_raw, dtype=np.float32).reshape(OPENPOSE_HAND_KEYPOINTS, 3)
                 if len(lh_raw) == OPENPOSE_HAND_FEATURES
                 else np.zeros((OPENPOSE_HAND_KEYPOINTS, 3), dtype=np.float32))
    
    rh_raw = person.get('hand_right_keypoints_2d', [])
    right_hand = (np.array(rh_raw, dtype=np.float32).reshape(OPENPOSE_HAND_KEYPOINTS, 3)
                  if len(rh_raw) == OPENPOSE_HAND_FEATURES
                  else np.zeros((OPENPOSE_HAND_KEYPOINTS, 3), dtype=np.float32))
    
    return pose, face, left_hand, right_hand


def load_openpose_directory(dir_path: str | Path) -> dict[str, np.ndarray]:
    """Load OpenPose JSON frames from a directory."""
    json_files = sorted(glob.glob(os.path.join(str(dir_path), '*.json')))
    if not json_files:
        raise ValueError(f'OpenPose directory {dir_path} contains no JSON files.')
    
    pose_list, face_list, lh_list, rh_list = [], [], [], []
    
    for jf in json_files:
        try:
            data = load_json(jf)
            people = data.get('people', [])
            if people:
                pose, face, lh, rh = parse_openpose_frame(people[0])
            else:
                pose = np.zeros((OPENPOSE_POSE_KEYPOINTS, 3), dtype=np.float32)
                face = np.zeros((OPENPOSE_FACE_KEYPOINTS, 3), dtype=np.float32)
                lh = np.zeros((OPENPOSE_HAND_KEYPOINTS, 3), dtype=np.float32)
                rh = np.zeros((OPENPOSE_HAND_KEYPOINTS, 3), dtype=np.float32)
            
            pose_list.append(pose)
            face_list.append(face)
            lh_list.append(lh)
            rh_list.append(rh)
        except Exception as e:
            raise IOError(f'Failed to read OpenPose file {jf}: {e}') from e
    
    return {
        'pose': np.stack(pose_list, axis=0),
        'face': np.stack(face_list, axis=0),
        'left_hand': np.stack(lh_list, axis=0),
        'right_hand': np.stack(rh_list, axis=0),
    }


def load_holistic_npy(filepath: str | Path) -> dict[str, np.ndarray]:
    """Load combined MediaPipe Holistic landmarks .npy file and slice to components."""
    try:
        data = np.load(filepath)
    except Exception as e:
        raise IOError(f"Failed to read Holistic npy file {filepath}: {e}") from e
        
    if data.ndim != 3 or data.shape[1] != 543 or data.shape[2] != 3:
        raise ValueError(f"Invalid Holistic npy array shape {data.shape}, expected (num_frames, 543, 3)")
        
    pose = data[:, 0:33, :]
    face_all = data[:, 33:501, :]
    left_hand = data[:, 501:522, :]
    right_hand = data[:, 522:543, :]
    
    face = face_all[:, MEDIAPIPE_FACE_SUBSET_INDICES, :]
    
    return {
        'pose': pose,
        'face': face,
        'left_hand': left_hand,
        'right_hand': right_hand
    }


def load_i3d_npy(filepath: str | Path) -> np.ndarray:
    """Load precomputed I3D features from a .npy file."""
    try:
        data = np.load(filepath)
    except Exception as e:
        raise IOError(f"Failed to read I3D features npy file {filepath}: {e}") from e
        
    if data.ndim > 2:
        data = np.squeeze(data)
    if data.ndim == 1:
        data = np.expand_dims(data, axis=0)
        
    if data.ndim != 2 or data.shape[1] != 1024:
        raise ValueError(f"Invalid I3D features shape {data.shape}, expected (num_frames, 1024)")
        
    return data


def discover_landmark_paths(data_dir: str | Path) -> list[str]:
    """Discover landmark files (.npz, .npy) or OpenPose directories."""
    data_dir_str = str(data_dir)
    
    if os.path.isfile(data_dir_str):
        return [data_dir_str]
    
    npz_files = glob.glob(os.path.join(data_dir_str, '*.npz'))
    if npz_files:
        return sorted(npz_files)
        
    npy_files = glob.glob(os.path.join(data_dir_str, '*.npy'))
    if npy_files:
        return sorted(npy_files)
    
    candidates = [
        data_dir_str,
        os.path.join(data_dir_str, 'train_2D_keypoints/openpose_output/json'),
        os.path.join(data_dir_str, 'openpose_output/json'),
    ]
    
    for cand in candidates:
        if not os.path.isdir(cand):
            continue
        subdirs = [
            os.path.join(cand, d)
            for d in os.listdir(cand)
            if os.path.isdir(os.path.join(cand, d))
        ]
        if subdirs and glob.glob(os.path.join(subdirs[0], '*.json')):
            return sorted(subdirs)
    
    return []
