import os
import glob
import numpy as np
import torch
from torch.utils.data import Dataset

class ASLLandmarkDataset(Dataset):
    """
    A PyTorch Dataset that loads pre-extracted landmark files (.npz)
    and maps them to target English text labels.
    """
    def __init__(self, data_dir, metadata_dict=None, file_list=None, max_len=150, include_face=True, normalize=True):
        """
        Args:
            data_dir (str): Directory containing .npz files.
            metadata_dict (dict): Dictionary mapping file basenames (without extension)
                                  to their English translation text.
            file_list (list): Optional list of specific file paths. If None, scans data_dir.
            max_len (int): Maximum frame sequence length. Longer sequences are truncated.
            include_face (bool): If True, concatenates face expression landmarks (276 dims)
                                 with manual landmarks (258 dims) for 534 dims total.
            normalize (bool): If True, applies frame-level geometric normalization.
        """
        self.data_dir = data_dir
        self.metadata_dict = metadata_dict if metadata_dict is not None else {}
        self.max_len = max_len
        self.include_face = include_face
        self.normalize = normalize
        
        if file_list is not None:
            self.filepaths = file_list
        else:
            self.filepaths = glob.glob(os.path.join(data_dir, "*.npz"))

    def __len__(self):
        return len(self.filepaths)

    def _normalize_landmarks(self, pose, left_hand, right_hand, face):
        """
        Normalizes landmark coordinates:
        - Pose: centered relative to mid-shoulder, scaled by shoulder width.
        - Hands: centered relative to their respective wrist joints and scaled by shoulder width.
        - Face: centered relative to face centroid and scaled by shoulder width.
        """
        # 1. Pose Normalization
        pose_coords = pose[..., :3]
        visibility = pose[..., 3:4]
        
        mid_shoulder = (pose_coords[:, 11, :] + pose_coords[:, 12, :]) / 2.0  # (num_frames, 3)
        mid_shoulder = np.expand_dims(mid_shoulder, axis=1)  # (num_frames, 1, 3)
        
        shoulder_width = np.linalg.norm(pose_coords[:, 11, :] - pose_coords[:, 12, :], axis=-1, keepdims=True)
        shoulder_width = np.expand_dims(shoulder_width, axis=1)  # (num_frames, 1, 1)
        
        # Clip shoulder width to a safe minimum of 0.01 to prevent NaNs or division by zero
        shoulder_width = np.clip(shoulder_width, a_min=0.01, a_max=None)
        
        norm_pose_coords = (pose_coords - mid_shoulder) / shoulder_width
        norm_pose = np.concatenate([norm_pose_coords, visibility], axis=-1)
        
        # 2. Hand Normalization (Wrist-relative & body-scale normalized)
        left_wrist = np.expand_dims(left_hand[:, 0, :], axis=1)  # (num_frames, 1, 3)
        norm_left_hand = (left_hand - left_wrist) / shoulder_width
        
        right_wrist = np.expand_dims(right_hand[:, 0, :], axis=1)  # (num_frames, 1, 3)
        norm_right_hand = (right_hand - right_wrist) / shoulder_width
        
        # 3. Face Normalization (Centroid-relative & body-scale normalized)
        face_centroid = np.mean(face, axis=1, keepdims=True)  # (num_frames, 1, 3)
        norm_face = (face - face_centroid) / shoulder_width
        
        return norm_pose, norm_left_hand, norm_right_hand, norm_face

    def __getitem__(self, idx):
        file_path = self.filepaths[idx]
        basename = os.path.splitext(os.path.basename(file_path))[0]
        
        # Load landmarks
        try:
            data = np.load(file_path)
            pose = data['pose']          # shape (num_frames, 33, 4)
            left_hand = data['left_hand']  # shape (num_frames, 21, 3)
            right_hand = data['right_hand'] # shape (num_frames, 21, 3)
            face = data['face']          # shape (num_frames, 92, 3)
        except Exception as e:
            # Return dummy zero tensors if load fails
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
        
        # Combine manual features: shape (num_frames, 258)
        manual_feats = np.concatenate([pose_flat, left_hand_flat, right_hand_flat], axis=1)
        
        if self.include_face:
            # Face: (N, 276)
            face_flat = face.reshape(num_frames, -1)
            # Combine manual + face: shape (num_frames, 534)
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
    def __init__(self, tokenizer=None, max_target_len=30):
        """
        Args:
            tokenizer: Pretrained Hugging Face tokenizer (e.g. T5Tokenizer) to convert text to IDs.
            max_target_len (int): Maximum token length for target text.
        """
        self.tokenizer = tokenizer
        self.max_target_len = max_target_len

    def __call__(self, batch):
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
