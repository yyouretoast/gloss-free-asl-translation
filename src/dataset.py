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
        training: bool = False,
    ):
        self.data_dir = data_dir
        self.metadata_dict = metadata_dict if metadata_dict is not None else {}
        self.max_len = max_len
        self.include_face = include_face
        self.normalize = normalize
        self.i3d_dir = i3d_dir
        self.is_training = training

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

    def _load_landmarks(
        self, file_path: str
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, int, bool]:
        """Loads pose, left hand, right hand, and face landmarks from file, returning fallbacks on failure."""
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
            pose = np.zeros((1, 33, 3), dtype=np.float32)
            left_hand = np.zeros((1, 21, 3), dtype=np.float32)
            right_hand = np.zeros((1, 21, 3), dtype=np.float32)
            face = np.zeros((1, 92, 3), dtype=np.float32)
            num_frames = 1

        return pose, left_hand, right_hand, face, num_frames, load_failed

    def _load_and_align_i3d(self, basename: str, num_frames: int) -> np.ndarray | None:
        """Loads and aligns precomputed I3D features matching sequence length."""
        if not self.i3d_dir:
            return None

        from src.utils.io_utils import load_i3d_npy

        # Convert landmark basename to I3D naming convention (e.g. video_id_front_holistic -> video_id-rgb_front)
        clean = basename.replace("_holistic", "").replace("_landmarks", "")
        view = "front" if "front" in clean.lower() else "side"
        sentence_name = clean.replace("_front", "").replace("_side", "")
        i3d_basename = f"{sentence_name}-rgb_{view}"

        i3d_path = self.i3d_file_map.get(i3d_basename)
        if i3d_path is None:
            i3d_path = self.i3d_file_map.get(clean)
        if i3d_path is None:
            i3d_path = self.i3d_file_map.get(basename)
        if i3d_path is None:
            i3d_path = os.path.join(self.i3d_dir, f"{clean}.npy")

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

        return i3d_features

    def _apply_augmentations(
        self,
        pose: np.ndarray,
        left_hand: np.ndarray,
        right_hand: np.ndarray,
        face: np.ndarray,
        i3d_features: np.ndarray | None,
        num_frames: int,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray | None, int]:
        """Applies random horizontal flip, temporal dropping/duplication, and hand dropout using PyTorch RNG."""
        # Horizontal mirroring.
        if torch.rand(1).item() < 0.5:
            pose[..., 0] = -pose[..., 0]
            left_hand[..., 0] = -left_hand[..., 0]
            right_hand[..., 0] = -right_hand[..., 0]
            face[..., 0] = -face[..., 0]
            left_hand, right_hand = right_hand.copy(), left_hand.copy()

            # Swap symmetric joints.
            if pose.shape[1] == 33:
                sym_pairs = [
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
                    (1, 4),
                    (2, 5),
                    (3, 6),
                ]
                for i, j in sym_pairs:
                    pose[:, [i, j]] = pose[:, [j, i]]

            if face.shape[1] == 92:
                # 42 symmetric index pairs (0-indexed within 92-subset)
                sym_pairs_face = [
                    (0, 2),
                    (3, 11),
                    (4, 10),
                    (5, 9),
                    (6, 8),
                    (12, 22),
                    (13, 21),
                    (14, 20),
                    (15, 19),
                    (16, 18),
                    (23, 31),
                    (24, 30),
                    (25, 29),
                    (26, 28),
                    (32, 42),
                    (33, 41),
                    (34, 40),
                    (35, 39),
                    (36, 38),
                    (43, 49),
                    (44, 48),
                    (45, 47),
                    (50, 56),
                    (51, 55),
                    (52, 54),
                    (58, 68),
                    (59, 67),
                    (60, 66),
                    (61, 65),
                    (62, 64),
                    (69, 79),
                    (70, 78),
                    (71, 77),
                    (72, 76),
                    (73, 75),
                    (80, 86),
                    (81, 85),
                    (82, 84),
                    (88, 91),
                    (89, 90),
                    (17, 17),
                    (27, 27),
                ]
                for i, j in sym_pairs_face:
                    face[:, [i, j]] = face[:, [j, i]]

        # Temporal jittering.
        if torch.rand(1).item() < 0.5 and num_frames > 5:
            num_to_change = max(1, int(num_frames * 0.1))
            indices = list(range(num_frames))
            if torch.rand(1).item() < 0.5:
                # Drop indices
                drop_indices = set(torch.randperm(num_frames)[:num_to_change].tolist())
                indices = [i for i in indices if i not in drop_indices]
            else:
                # Duplicate indices
                dup_indices = torch.randperm(num_frames)[:num_to_change].tolist()
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
        if torch.rand(1).item() < 0.05:
            if torch.rand(1).item() < 0.5:
                left_hand = np.zeros_like(left_hand)
            else:
                right_hand = np.zeros_like(right_hand)

        return pose, left_hand, right_hand, face, i3d_features, num_frames

    def _apply_spatial_augmentations(
        self,
        pose: np.ndarray,
        left_hand: np.ndarray,
        right_hand: np.ndarray,
        face: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Applies random 3D rotation, scaling, and translation jitter to normalized coordinates."""
        # 1. Random Scaling: scale between 0.95 and 1.05
        scale = 0.95 + torch.rand(1).item() * 0.1

        # 2. Random 3D Rotation: yaw, pitch, roll angles within [-5, 5] degrees (approx -0.087 to +0.087 radians)
        angles = (torch.rand(3) * 2.0 - 1.0) * 0.087  # -5 to +5 degrees in radians
        alpha, beta, gamma = angles[0].item(), angles[1].item(), angles[2].item()

        # Compute rotation matrix R
        cos_a, sin_a = np.cos(alpha), np.sin(alpha)
        cos_b, sin_b = np.cos(beta), np.sin(beta)
        cos_g, sin_g = np.cos(gamma), np.sin(gamma)

        R_x = np.array(
            [[1, 0, 0], [0, cos_a, -sin_a], [0, sin_a, cos_a]], dtype=np.float32
        )

        R_y = np.array(
            [[cos_b, 0, sin_b], [0, 1, 0], [-sin_b, 0, cos_b]], dtype=np.float32
        )

        R_z = np.array(
            [[cos_g, -sin_g, 0], [sin_g, cos_g, 0], [0, 0, 1]], dtype=np.float32
        )

        R = R_x @ R_y @ R_z

        # 3. Random Translation Jitter: translation offset within [-0.02, 0.02]
        translation = (torch.rand(3).numpy() * 2.0 - 1.0) * 0.02

        # Apply to spatial (x, y, z) coordinates (first 3 channels)
        pose[..., :3] = (pose[..., :3] @ R) * scale + translation
        left_hand[..., :3] = (left_hand[..., :3] @ R) * scale + translation
        right_hand[..., :3] = (right_hand[..., :3] @ R) * scale + translation
        face[..., :3] = (face[..., :3] @ R) * scale + translation

        return pose, left_hand, right_hand, face

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        file_path = self.filepaths[idx]
        basename = os.path.splitext(os.path.basename(file_path))[0]

        # 1. Load landmarks with graceful fallback.
        pose, left_hand, right_hand, face, num_frames, load_failed = (
            self._load_landmarks(file_path)
        )

        # 2. Load and align I3D features.
        i3d_features = self._load_and_align_i3d(basename, num_frames)

        # 3. Apply training-only data augmentations.
        if self.is_training and not load_failed:
            pose, left_hand, right_hand, face, i3d_features, num_frames = (
                self._apply_augmentations(
                    pose, left_hand, right_hand, face, i3d_features, num_frames
                )
            )

        # 4. Normalize spatial coordinates.
        if self.normalize:
            pose, left_hand, right_hand, face = self._normalize_landmarks(
                pose, left_hand, right_hand, face
            )

        # Apply random spatial (3D rotation, scaling, translation) augmentations during training
        if self.is_training and not load_failed:
            pose, left_hand, right_hand, face = self._apply_spatial_augmentations(
                pose, left_hand, right_hand, face
            )

        # 5. Flatten and combine landmarks.
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

        # 6. Stride-based downsampling.
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
