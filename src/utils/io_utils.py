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


def load_json(filepath: str | Path) -> dict:
    """Load a JSON file using orjson (if available) or stdlib json."""
    with open(filepath, 'rb') as f:
        content = f.read()
    if orjson is not None:
        return orjson.loads(content)
    return json.loads(content)


def parse_openpose_frame(person: dict) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Parse a single person's keypoints from an OpenPose JSON frame.
    
    Returns:
        Tuple of (pose, face, left_hand, right_hand) as numpy arrays.
        pose: shape (25, 3), face: shape (70, 3), hands: shape (21, 3) each.
        Falls back to zeros if keypoints are missing or malformed.
    """
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
    """Load all OpenPose JSON frames from a directory.
    
    Returns:
        Dictionary with 'pose', 'face', 'left_hand', 'right_hand' arrays,
        each of shape (num_frames, keypoints, dims).
    
    Raises:
        ValueError: If the directory contains no JSON files.
        IOError: If any JSON file cannot be read.
    """
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


def discover_landmark_paths(data_dir: str | Path) -> list[str]:
    """Discover landmark files (.npz) or OpenPose JSON directories in a data directory.
    
    Searches for .npz files first. If none found, looks for OpenPose-style
    subdirectories containing JSON files.
    
    Returns:
        List of file/directory paths.
    """
    data_dir = str(data_dir)
    
    # Try .npz files first
    npz_files = glob.glob(os.path.join(data_dir, '*.npz'))
    if npz_files:
        return npz_files
    
    # Look for OpenPose JSON directory structure
    candidates = [
        data_dir,
        os.path.join(data_dir, 'train_2D_keypoints/openpose_output/json'),
        os.path.join(data_dir, 'openpose_output/json'),
    ]
    
    for cand in candidates:
        if not os.path.exists(cand):
            continue
        subdirs = [
            os.path.join(cand, d)
            for d in os.listdir(cand)
            if os.path.isdir(os.path.join(cand, d))
        ]
        if subdirs and glob.glob(os.path.join(subdirs[0], '*.json')):
            return subdirs
    
    return []
