from __future__ import annotations
import os
import numpy as np
import torch
from torch.utils.data import Dataset
from typing import Dict, List, Optional, Any

from src.utils.io_utils import discover_landmark_paths


class ASLLandmarkDataset(Dataset):
    """Dataset for loading pre-extracted sign language landmarks."""

    def __init__(
        self,
        data_dir: str,
        metadata_dict: Optional[Dict[str, str]] = None,
        file_list: Optional[List[str]] = None,
        max_len: int = 150,
        include_face: bool = True,
        normalize: bool = True,
        skip_empty_labels: bool = True,
        i3d_dir: Optional[str] = None,
    ):
        self.data_dir = data_dir
        self.metadata_dict = metadata_dict if metadata_dict is not None else {}
        self.max_len = max_len
        self.include_face = include_face
        self.normalize = normalize
        self.i3d_dir = i3d_dir

        self.i3d_file_map = {}
        if i3d_dir:
            # Build clean_basename -> absolute_path mapping recursively (fast scan)
            for root, _, files in os.walk(i3d_dir):
                for f in files:
                    if f.endswith(".npy"):
                        bname = (
                            os.path.splitext(f)[0]
                            .replace("_holistic", "")
                            .replace("_landmarks", "")
                        )
                        self.i3d_file_map[bname] = os.path.join(root, f)

        # Identify if dataset is for training to enable augmentations.
        self.is_training = False
        if data_dir and "train" in data_dir.lower():
            self.is_training = True
        elif file_list and any("train" in fp.lower() for fp in file_list):
            self.is_training = True

        if file_list is not None:
            self.filepaths = file_list
        else:
            self.filepaths = discover_landmark_paths(data_dir)

        if len(self.filepaths) == 0:
            import warnings

            warnings.warn(f"No landmark files (.npz or .npy) found in '{data_dir}'.")

        # Filter out files with missing/empty labels.
        if self.metadata_dict and skip_empty_labels:
            missing_count = 0
            valid_filepaths = []
            for fp in self.filepaths:
                basename = os.path.splitext(os.path.basename(fp))[0]
                clean_basename = basename.replace("_holistic", "").replace(
                    "_landmarks", ""
                )
                if not self.metadata_dict.get(clean_basename, "").strip():
                    missing_count += 1
                else:
                    valid_filepaths.append(fp)

            if missing_count > 0:
                import warnings

                warnings.warn(
                    f"{missing_count}/{len(self.filepaths)} samples have empty labels "
                    f"and will be skipped to avoid contributing zero loss during training."
                )
            self.filepaths = valid_filepaths

    def __len__(self) -> int:
        return len(self.filepaths)

    def _normalize_landmarks(
        self,
        pose: np.ndarray,
        left_hand: np.ndarray,
        right_hand: np.ndarray,
        face: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Normalize coordinates relative to skeleton reference points."""
        spatial_dim = 3
        shoulder_idx_1 = 11
        shoulder_idx_2 = 12

        # 1. Pose Normalization
        pose_spatial = pose[..., :spatial_dim]
        pose_extra = pose[..., spatial_dim:]

        mid_shoulder = (
            pose_spatial[:, shoulder_idx_1, :] + pose_spatial[:, shoulder_idx_2, :]
        ) / 2.0
        mid_shoulder = np.expand_dims(mid_shoulder, axis=1)

        shoulder_width = np.linalg.norm(
            pose_spatial[:, shoulder_idx_1, :] - pose_spatial[:, shoulder_idx_2, :],
            axis=-1,
            keepdims=True,
        )
        shoulder_width = np.expand_dims(shoulder_width, axis=1)

        # Avoid division by zero/near-zero by using a fallback scale.
        shoulder_width = np.where(shoulder_width < 0.05, 0.25, shoulder_width)

        norm_pose_spatial = (pose_spatial - mid_shoulder) / shoulder_width
        norm_pose = np.concatenate([norm_pose_spatial, pose_extra], axis=-1)

        # 2. Hand Normalization
        lh_spatial = left_hand[..., :spatial_dim]
        lh_extra = left_hand[..., spatial_dim:]
        left_wrist = np.expand_dims(lh_spatial[:, 0, :], axis=1)
        norm_lh_spatial = (lh_spatial - left_wrist) / shoulder_width
        norm_left_hand = np.concatenate([norm_lh_spatial, lh_extra], axis=-1)

        rh_spatial = right_hand[..., :spatial_dim]
        rh_extra = right_hand[..., spatial_dim:]
        right_wrist = np.expand_dims(rh_spatial[:, 0, :], axis=1)
        norm_rh_spatial = (rh_spatial - right_wrist) / shoulder_width
        norm_right_hand = np.concatenate([norm_rh_spatial, rh_extra], axis=-1)

        # 3. Face Normalization
        face_spatial = face[..., :spatial_dim]
        face_extra = face[..., spatial_dim:]
        face_centroid = np.mean(face_spatial, axis=1, keepdims=True)
        norm_face_spatial = (face_spatial - face_centroid) / shoulder_width
        norm_face = np.concatenate([norm_face_spatial, face_extra], axis=-1)

        return norm_pose, norm_left_hand, norm_right_hand, norm_face

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        file_path = self.filepaths[idx]
        basename = os.path.splitext(os.path.basename(file_path))[0]

        load_failed = False
        try:
            if file_path.endswith(".npy"):
                from src.utils.io_utils import load_holistic_npy

                landmarks = load_holistic_npy(file_path)
                pose = landmarks["pose"]
                left_hand = landmarks["left_hand"]
                right_hand = landmarks["right_hand"]
                face = landmarks["face"]
            else:
                with np.load(file_path) as data:
                    pose = data["pose"][..., :3]
                    left_hand = data["left_hand"]
                    right_hand = data["right_hand"]
                    face = data["face"]

            num_frames = pose.shape[0]
            if num_frames == 0:
                load_failed = True
        except Exception as e:
            import warnings

            warnings.warn(
                f"Failed to load landmark file {file_path}: {e}. Returning dummy zeros."
            )
            load_failed = True

        if load_failed:
            # Fallback to dummy zero frame to prevent pipeline crashes.
            pose = np.zeros((1, 33, 3), dtype=np.float32)
            left_hand = np.zeros((1, 21, 3), dtype=np.float32)
            right_hand = np.zeros((1, 21, 3), dtype=np.float32)
            face = np.zeros((1, 92, 3), dtype=np.float32)
            num_frames = 1

        # Load and align I3D features if present.
        i3d_features = None
        if self.i3d_dir:
            from src.utils.io_utils import load_i3d_npy

            i3d_basename = basename.replace("_holistic", "").replace("_landmarks", "")
            i3d_path = self.i3d_file_map.get(i3d_basename)
            if i3d_path is None:
                i3d_path = os.path.join(self.i3d_dir, f"{i3d_basename}.npy")
            try:
                i3d_features = load_i3d_npy(i3d_path)
            except Exception as e:
                import warnings

                warnings.warn(
                    f"Failed to load I3D file {i3d_path}: {e}. Returning dummy zeros."
                )
                i3d_features = np.zeros((num_frames, 1024), dtype=np.float32)

            # Align sequence length via interpolation.
            if i3d_features.shape[0] == 0:
                i3d_features = np.zeros((num_frames, 1024), dtype=np.float32)
            elif num_frames <= 1 or i3d_features.shape[0] <= 1:
                i3d_features = np.tile(i3d_features[0], (num_frames, 1))
            elif i3d_features.shape[0] != num_frames:
                old_x = np.linspace(0, 1, i3d_features.shape[0])
                new_x = np.linspace(0, 1, num_frames)
                interpolated = np.zeros(
                    (num_frames, i3d_features.shape[1]), dtype=np.float32
                )
                for c in range(i3d_features.shape[1]):
                    interpolated[:, c] = np.interp(new_x, old_x, i3d_features[:, c])
                i3d_features = interpolated

        # Apply training-only data augmentations.
        if self.is_training and not load_failed:
            # Horizontal mirroring.
            if np.random.rand() < 0.5:
                pose[..., 0] = -pose[..., 0]
                left_hand[..., 0] = -left_hand[..., 0]
                right_hand[..., 0] = -right_hand[..., 0]
                face[..., 0] = -face[..., 0]
                left_hand, right_hand = right_hand.copy(), left_hand.copy()

                # Swap symmetric joints.
                if pose.shape[1] == 33:
                    sym_pairs = [
                        (1, 4),
                        (2, 5),
                        (3, 6),
                        (7, 8),
                        (9, 10),
                        (11, 12),
                        (13, 14),
                        (15, 16),
                        (17, 18),
                        (19, 20),
                        (21, 22),
                        (23, 24),
                        (25, 26),
                        (27, 28),
                        (29, 30),
                        (31, 32),
                    ]
                    for i, j in sym_pairs:
                        pose[:, [i, j]] = pose[:, [j, i]]

                # Swap symmetric face landmarks (eyes, eyebrows, lips)
                if face.shape[1] == 92:
                    face_sym_pairs = [
                        (48, 1),
                        (49, 5),
                        (50, 6),
                        (51, 7),
                        (52, 8),
                        (53, 10),
                        (54, 11),
                        (55, 12),
                        (56, 13),
                        (57, 14),
                        (58, 15),
                        (59, 16),
                        (60, 17),
                        (61, 18),
                        (62, 19),
                        (64, 20),
                        (65, 21),
                        (66, 22),
                        (67, 23),
                        (68, 24),
                        (69, 25),
                        (70, 26),
                        (71, 27),
                        (72, 28),
                        (73, 29),
                        (74, 30),
                        (75, 31),
                        (76, 32),
                        (77, 33),
                        (78, 34),
                        (79, 35),
                        (80, 36),
                        (81, 37),
                        (82, 38),
                        (83, 39),
                        (84, 40),
                        (85, 41),
                        (86, 42),
                        (87, 43),
                        (88, 44),
                        (89, 46),
                        (91, 47),
                    ]
                    for i, j in face_sym_pairs:
                        face[:, [i, j]] = face[:, [j, i]]

            # Temporal jittering.
            if np.random.rand() < 0.5 and num_frames > 5:
                num_to_change = max(1, int(num_frames * 0.1))
                indices = list(range(num_frames))
                if np.random.rand() < 0.5:
                    drop_indices = set(
                        np.random.choice(num_frames, num_to_change, replace=False)
                    )
                    indices = [i for i in indices if i not in drop_indices]
                else:
                    dup_indices = np.random.choice(
                        num_frames, num_to_change, replace=False
                    )
                    for d_idx in dup_indices:
                        indices.append(d_idx)
                    indices = sorted(indices)

                pose = pose[indices]
                left_hand = left_hand[indices]
                right_hand = right_hand[indices]
                face = face[indices]
                if i3d_features is not None:
                    i3d_features = i3d_features[indices]
                num_frames = len(indices)

            # Hand joint dropout.
            if np.random.rand() < 0.05:
                if np.random.rand() < 0.5:
                    left_hand = np.zeros_like(left_hand)
                else:
                    right_hand = np.zeros_like(right_hand)

        if self.normalize:
            pose, left_hand, right_hand, face = self._normalize_landmarks(
                pose, left_hand, right_hand, face
            )

        # Flatten and combine landmarks.
        pose_flat = pose.reshape(num_frames, -1)
        left_hand_flat = left_hand.reshape(num_frames, -1)
        right_hand_flat = right_hand.reshape(num_frames, -1)

        manual_feats = np.concatenate(
            [pose_flat, left_hand_flat, right_hand_flat], axis=1
        )

        if self.include_face:
            face_flat = face.reshape(num_frames, -1)
            features = np.concatenate([manual_feats, face_flat], axis=1)
        else:
            features = manual_feats

        # Stride-based downsampling.
        seq_len = len(features)
        if seq_len > self.max_len:
            stride = int(np.ceil(seq_len / self.max_len))
            features = features[::stride]
            if i3d_features is not None:
                i3d_features = i3d_features[::stride]

            if len(features) > self.max_len:
                features = features[: self.max_len]
                if i3d_features is not None:
                    i3d_features = i3d_features[: self.max_len]

        clean_basename = basename.replace("_holistic", "").replace("_landmarks", "")
        target_text = self.metadata_dict.get(clean_basename, "")

        sample = {
            "features": torch.tensor(features, dtype=torch.float32),
            "text": target_text,
            "file_id": basename,
        }
        if i3d_features is not None:
            sample["i3d_features"] = torch.as_tensor(i3d_features, dtype=torch.float32)

        return sample


class CollateLandmarks:
    """Collate and pad variable-length sequences and texts into batches."""

    def __init__(self, tokenizer: Any = None, max_target_len: int = 30):
        self.tokenizer = tokenizer
        self.max_target_len = max_target_len

    def __call__(self, batch: list[dict]) -> dict:
        if not batch:
            return {}
        features = [
            torch.as_tensor(item["features"], dtype=torch.float32) for item in batch
        ]
        texts = [item["text"] for item in batch]
        file_ids = [item["file_id"] for item in batch]

        # Pad features.
        feature_dim = features[0].shape[1]
        lengths = [len(f) for f in features]
        max_seq_len = max(lengths)

        padded_features = torch.zeros(len(batch), max_seq_len, feature_dim)
        attention_mask = torch.zeros(len(batch), max_seq_len, dtype=torch.float32)

        for i, f in enumerate(features):
            if f.shape[1] != feature_dim:
                raise ValueError(
                    f"Feature dimension mismatch in batch! First item has dim {feature_dim}, "
                    f"but item at index {i} has dim {f.shape[1]}."
                )
            seq_len = len(f)
            padded_features[i, :seq_len] = f
            attention_mask[i, :seq_len] = 1.0

        batch_dict = {
            "input_features": padded_features,
            "attention_mask": attention_mask,
            "file_ids": file_ids,
        }

        # Pad and align I3D features if present in batch.
        has_any_i3d = any("i3d_features" in item for item in batch)
        if has_any_i3d:
            padded_i3d = torch.zeros(len(batch), max_seq_len, 1024)
            for i, item in enumerate(batch):
                if "i3d_features" in item:
                    i3d_f = item["i3d_features"]
                    seq_len = len(i3d_f)
                    limit_len = min(seq_len, max_seq_len)
                    padded_i3d[i, :limit_len] = i3d_f[:limit_len]
            batch_dict["input_i3d_features"] = padded_i3d

        # Tokenize target text.
        if self.tokenizer is not None:
            tokenized = self.tokenizer(
                texts,
                padding="max_length",
                truncation=True,
                max_length=self.max_target_len,
                return_tensors="pt",
            )
            # Ignore pad token index in loss.
            labels = tokenized.input_ids
            if self.tokenizer.pad_token_id is not None:
                labels[labels == self.tokenizer.pad_token_id] = -100
            batch_dict["labels"] = labels
            batch_dict["decoder_attention_mask"] = tokenized.attention_mask
        else:
            # Return raw text when no tokenizer is used.
            batch_dict["labels"] = texts

        return batch_dict
