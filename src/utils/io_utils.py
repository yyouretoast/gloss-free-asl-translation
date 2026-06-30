"""Shared I/O utilities for landmark loading and directory discovery."""

from __future__ import annotations

import glob
import os
from pathlib import Path

import numpy as np

# Optional fast JSON parser removed (unused)


# MediaPipe format constants
MEDIAPIPE_POSE_KEYPOINTS = 33
MEDIAPIPE_FACE_KEYPOINTS = 92
MEDIAPIPE_HAND_KEYPOINTS = 21

# 92 key facial landmark indices from the MediaPipe 468-point face mesh.
# These target expressive regions critical for ASL grammar (lips, eyes, eyebrows).
# Slices the 468 mesh landmarks down to a subset of 92 points.
MEDIAPIPE_FACE_SUBSET_INDICES = sorted(
    list(
        set(
            [
                61,
                146,
                91,
                181,
                84,
                17,
                314,
                405,
                321,
                375,
                291,
                308,
                324,
                318,
                402,
                317,
                14,
                87,
                178,
                88,
                95,
                185,
                40,
                39,
                37,
                0,
                267,
                269,
                270,
                409,
                415,
                310,
                311,
                312,
                13,
                82,
                81,
                42,
                183,
                78,
            ]  # lips
            + [
                362,
                382,
                381,
                380,
                374,
                373,
                390,
                249,
                263,
                466,
                388,
                387,
                386,
                385,
                384,
                398,
            ]  # left eye
            + [
                33,
                7,
                163,
                144,
                145,
                153,
                154,
                155,
                133,
                173,
                157,
                158,
                159,
                160,
                161,
                246,
            ]  # right eye
            + [336, 296, 334, 293, 300, 276, 283, 282, 295, 285]  # left eyebrow
            + [70, 63, 105, 66, 107, 55, 65, 52, 53, 46]  # right eyebrow
        )
    )
)


# load_json helper removed (unused)


def load_holistic_npy(filepath: str | Path) -> dict[str, np.ndarray]:
    """Load combined MediaPipe Holistic landmarks .npy file and slice to components."""
    try:
        data = np.load(filepath)
    except Exception as e:
        raise IOError(f"Failed to read Holistic npy file {filepath}: {e}") from e

    if data.ndim != 3 or data.shape[1] != 543 or data.shape[2] != 3:
        raise ValueError(
            f"Invalid Holistic npy array shape {data.shape}, expected (num_frames, 543, 3)"
        )

    pose = data[:, 0:33, :]
    face_all = data[:, 33:501, :]
    left_hand = data[:, 501:522, :]
    right_hand = data[:, 522:543, :]

    face = face_all[:, MEDIAPIPE_FACE_SUBSET_INDICES, :]

    return {
        "pose": pose,
        "face": face,
        "left_hand": left_hand,
        "right_hand": right_hand,
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
        raise ValueError(
            f"Invalid I3D features shape {data.shape}, expected (num_frames, 1024)"
        )

    return data


def discover_landmark_paths(data_dir: str | Path) -> list[str]:
    """Discover landmark files (.npz, .npy) recursively if none found at top level."""
    data_dir_str = str(data_dir)

    if os.path.isfile(data_dir_str):
        return [data_dir_str]

    # First try top-level discovery
    npz_files = glob.glob(os.path.join(data_dir_str, "*.npz"))
    if npz_files:
        return sorted(npz_files)

    npy_files = glob.glob(os.path.join(data_dir_str, "*.npy"))
    if npy_files:
        return sorted(npy_files)

    # If no files found at top-level, search recursively using os.walk
    discovered = []
    for root, _, files in os.walk(data_dir_str):
        for f in files:
            if f.endswith(".npz") or f.endswith(".npy"):
                discovered.append(os.path.join(root, f))
    return sorted(discovered)
