from __future__ import annotations
import os
import numpy as np
import torch
from torch.utils.data import Dataset
from typing import Dict, List, Optional, Any, Union

from src.utils.io_utils import discover_landmark_paths, load_openpose_directory

class ASLLandmarkDataset(Dataset):
    """
    A PyTorch Dataset that loads pre-extracted landmark files (.npz)
    and maps them to target English text labels.
    """
    def __init__(
        self, 
        data_dir: str, 
        metadata_dict: Optional[Dict[str, str]] = None, 
        file_list: Optional[List[str]] = None, 
        max_len: int = 150, 
        include_face: bool = True, 
        normalize: bool = True,
        skip_empty_labels: bool = True
    ):
        """
        Args:
            data_dir (str): Directory containing .npz files.
            metadata_dict (dict): Dictionary mapping file basenames (without extension)
                                  to their English translation text.
            file_list (list): Optional list of specific file paths. If None, scans data_dir.
            max_len (int): Maximum frame sequence length. Longer sequences are truncated.
            include_face (bool): If True, concatenates face expression landmarks (276 dims for MediaPipe, 
                                 210 dims for OpenPose) with manual landmarks (258 dims for MediaPipe, 
                                 201 dims for OpenPose) for a total of 534 or 411 dims respectively.
            normalize (bool): If True, applies frame-level geometric normalization.
            skip_empty_labels (bool): If True, filters out files that have empty labels.
        """
        self.data_dir = data_dir
        self.metadata_dict = metadata_dict if metadata_dict is not None else {}
        self.max_len = max_len
        self.include_face = include_face
        self.normalize = normalize
        
        if file_list is not None:
            self.filepaths = file_list
        else:
            self.filepaths = discover_landmark_paths(data_dir)
        
        if len(self.filepaths) == 0:
            import warnings
            warnings.warn(f"No landmark files (.npz) or OpenPose folders found in '{data_dir}'.")
            
        # Filter out samples with missing labels to avoid wasting training compute
        if self.metadata_dict and skip_empty_labels:
            missing_count = 0
            valid_filepaths = []
            for fp in self.filepaths:
                basename = os.path.splitext(os.path.basename(fp))[0]
                if not self.metadata_dict.get(basename, "").strip():
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

    def _normalize_landmarks(self, pose: np.ndarray, left_hand: np.ndarray, right_hand: np.ndarray, face: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        Normalizes landmark coordinates:
        - Pose: centered relative to mid-shoulder, scaled by shoulder width.
        - Hands: centered relative to their respective wrist joints and scaled by shoulder width.
        - Face: centered relative to face centroid and scaled by shoulder width.
        Normalizes only spatial coordinates, leaving visibility/confidence untouched.
        """
        # OpenPose BODY_25 has 25 keypoints (shoulders are index 2 and 5)
        # MediaPipe has 33 keypoints (shoulders are index 11 and 12)
        if pose.shape[1] == 25:
            spatial_dim = 2
            shoulder_idx_1 = 5
            shoulder_idx_2 = 2
        else:
            spatial_dim = 3
            shoulder_idx_1 = 11
            shoulder_idx_2 = 12
            
        # 1. Pose Normalization
        pose_spatial = pose[..., :spatial_dim]
        pose_extra = pose[..., spatial_dim:]
        
        mid_shoulder = (pose_spatial[:, shoulder_idx_1, :] + pose_spatial[:, shoulder_idx_2, :]) / 2.0  # (num_frames, spatial_dim)
        mid_shoulder = np.expand_dims(mid_shoulder, axis=1)  # (num_frames, 1, spatial_dim)
        
        shoulder_width = np.linalg.norm(pose_spatial[:, shoulder_idx_1, :] - pose_spatial[:, shoulder_idx_2, :], axis=-1, keepdims=True)
        shoulder_width = np.expand_dims(shoulder_width, axis=1)  # (num_frames, 1, 1)
        
        # Avoid coordinate explosion when shoulder width is too small (e.g. tracking failure)
        # Use 0.05 threshold (not 0.01) to catch near-failure tracking, and fall back
        # to a fixed 0.25 scale (approximate normalized shoulder width) rather than
        # the near-zero value, to prevent coordinate explosion.
        shoulder_width = np.where(shoulder_width < 0.05, 0.25, shoulder_width)
        
        norm_pose_spatial = (pose_spatial - mid_shoulder) / shoulder_width
        norm_pose = np.concatenate([norm_pose_spatial, pose_extra], axis=-1)
        
        # 2. Hand Normalization (Wrist-relative & body-scale normalized)
        lh_spatial = left_hand[..., :spatial_dim]
        lh_extra = left_hand[..., spatial_dim:]
        left_wrist = np.expand_dims(lh_spatial[:, 0, :], axis=1)  # (num_frames, 1, spatial_dim)
        norm_lh_spatial = (lh_spatial - left_wrist) / shoulder_width
        norm_left_hand = np.concatenate([norm_lh_spatial, lh_extra], axis=-1)
        
        rh_spatial = right_hand[..., :spatial_dim]
        rh_extra = right_hand[..., spatial_dim:]
        right_wrist = np.expand_dims(rh_spatial[:, 0, :], axis=1)  # (num_frames, 1, spatial_dim)
        norm_rh_spatial = (rh_spatial - right_wrist) / shoulder_width
        norm_right_hand = np.concatenate([norm_rh_spatial, rh_extra], axis=-1)
        
        # 3. Face Normalization (Centroid-relative & body-scale normalized)
        face_spatial = face[..., :spatial_dim]
        face_extra = face[..., spatial_dim:]
        face_centroid = np.mean(face_spatial, axis=1, keepdims=True)  # (num_frames, 1, spatial_dim)
        norm_face_spatial = (face_spatial - face_centroid) / shoulder_width
        norm_face = np.concatenate([norm_face_spatial, face_extra], axis=-1)
        
        return norm_pose, norm_left_hand, norm_right_hand, norm_face

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        file_path = self.filepaths[idx]
        basename = os.path.splitext(os.path.basename(file_path))[0]
        
        # Load landmarks (handling both .npz files and directories of OpenPose JSONs)
        if os.path.isdir(file_path):
            landmarks = load_openpose_directory(file_path)
            pose = landmarks['pose']
            left_hand = landmarks['left_hand']
            right_hand = landmarks['right_hand']
            face = landmarks['face']
            num_frames = pose.shape[0]
        else:
            try:
                with np.load(file_path) as data:
                    pose = data['pose']          # shape (num_frames, 33, 4)
                    left_hand = data['left_hand']  # shape (num_frames, 21, 3)
                    right_hand = data['right_hand'] # shape (num_frames, 21, 3)
                    face = data['face']          # shape (num_frames, 92, 3)
            except Exception as e:
                raise IOError(f"Failed to load landmark file {file_path}: {e}")
                
            num_frames = pose.shape[0]
            
        if num_frames == 0:
            raise ValueError(f"Landmark file {file_path} contains 0 frames. Corrupt sequence.")

        if self.normalize:
            pose, left_hand, right_hand, face = self._normalize_landmarks(pose, left_hand, right_hand, face)
        
        # Flatten landmark dimensions per frame
        # Pose: (N, 132), Hands: (N, 63) each
        pose_flat = pose.reshape(num_frames, -1)
        left_hand_flat = left_hand.reshape(num_frames, -1)
        right_hand_flat = right_hand.reshape(num_frames, -1)
        
        # Combine manual features: shape (num_frames, 258) for MediaPipe or (num_frames, 201) for OpenPose
        manual_feats = np.concatenate([pose_flat, left_hand_flat, right_hand_flat], axis=1)
        
        if self.include_face:
            # Face: (N, 276)
            face_flat = face.reshape(num_frames, -1)
            # Combine manual + face: shape (num_frames, 534) for MediaPipe or (num_frames, 411) for OpenPose
            features = np.concatenate([manual_feats, face_flat], axis=1)
        else:
            features = manual_feats

        # Truncate if sequence exceeds max_len
        if len(features) > self.max_len:
            features = features[:self.max_len]
            
        # Get target text label (default to empty string if not in metadata)
        target_text = self.metadata_dict.get(basename, "")

        return {
            'features': torch.tensor(features, dtype=torch.float32),
            'text': target_text,
            'file_id': basename
        }

class CollateLandmarks:
    """
    Collate function to pad variable-length landmark sequences and target texts into batches.
    """
    def __init__(self, tokenizer: Any = None, max_target_len: int = 30):
        """
        Args:
            tokenizer: Pretrained Hugging Face tokenizer (e.g. T5Tokenizer) to convert text to IDs.
            max_target_len (int): Maximum token length for target text.
        """
        self.tokenizer = tokenizer
        self.max_target_len = max_target_len

    def __call__(self, batch: List[Dict[str, Any]]) -> Dict[str, Any]:
        # Extract features and texts
        features = [item['features'] for item in batch]
        texts = [item['text'] for item in batch]
        file_ids = [item['file_id'] for item in batch]
        
        # Pad features sequence lengths
        # Each feature tensor has shape (seq_len, feature_dim)
        feature_dim = features[0].shape[1]
        lengths = [len(f) for f in features]
        max_seq_len = max(lengths)
        
        padded_features = torch.zeros(len(batch), max_seq_len, feature_dim)
        attention_mask = torch.zeros(len(batch), max_seq_len, dtype=torch.float32)
        
        for i, f in enumerate(features):
            # Check for feature dimension mismatch in batch
            if f.shape[1] != feature_dim:
                raise ValueError(
                    f"Feature dimension mismatch in batch! First item has dim {feature_dim}, "
                    f"but item at index {i} has dim {f.shape[1]}."
                )
            seq_len = len(f)
            padded_features[i, :seq_len] = f
            attention_mask[i, :seq_len] = 1.0  # 1 for valid frames, 0 for pad

        batch_dict = {
            'input_features': padded_features,    # shape (batch, max_seq_len, feature_dim)
            'attention_mask': attention_mask,    # shape (batch, max_seq_len)
            'file_ids': file_ids
        }
        
        # Tokenize target text if tokenizer is provided
        if self.tokenizer is not None:
            tokenized = self.tokenizer(
                texts,
                padding=True,
                truncation=True,
                max_length=self.max_target_len,
                return_tensors="pt"
            )
            # Replace padding token ID with -100 so PyTorch CrossEntropyLoss ignores it
            labels = tokenized.input_ids
            labels[labels == self.tokenizer.pad_token_id] = -100
            batch_dict['labels'] = labels        # shape (batch, max_target_len)
            batch_dict['decoder_attention_mask'] = tokenized.attention_mask
        else:
            # Return raw texts if tokenizer is absent (e.g. during testing)
            batch_dict['labels'] = texts

        return batch_dict
