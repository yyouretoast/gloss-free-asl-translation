import cv2
import mediapipe as mp
import numpy as np
import os
from tqdm import tqdm

class ASLLandmarkExtractor:
    """
    A class to extract landmarks from ASL videos using MediaPipe Holistic.
    It separates the manual features (hands + pose) and non-manual features (face mesh)
    and handles missing detections by zero-padding.
    """
    def __init__(self, model_complexity=1, min_detection_confidence=0.5, min_tracking_confidence=0.5):
        self.mp_holistic = mp.solutions.holistic
        self.model_complexity = model_complexity
        self.min_detection_confidence = min_detection_confidence
        self.min_tracking_confidence = min_tracking_confidence
        
        # 92 key indices for facial expression (eyes, eyebrows, mouth/lips)
        lips = [61, 146, 91, 181, 84, 17, 314, 405, 321, 375, 291, 308, 324, 318, 402, 317, 14, 87, 178, 88, 95, 185, 40, 39, 37, 0, 267, 269, 270, 409, 415, 310, 311, 312, 13, 82, 81, 42, 183, 78]
        left_eye = [362, 382, 381, 380, 374, 373, 390, 249, 263, 466, 388, 387, 386, 385, 384, 398]
        right_eye = [33, 7, 163, 144, 145, 153, 154, 155, 133, 173, 157, 158, 159, 160, 161, 246]
        left_eyebrow = [336, 296, 334, 293, 300, 276, 283, 282, 295, 285]
        right_eyebrow = [70, 63, 105, 66, 107, 55, 65, 52, 53, 46]
        self.face_indices = sorted(list(set(lips + left_eye + right_eye + left_eyebrow + right_eyebrow)))

    def extract_landmarks(self, video_path):
        """
        Processes a video file frame-by-frame and extracts Pose, Left Hand, Right Hand,
        and Facial Expression landmarks.

        Args:
            video_path (str): Path to the input video file.

        Returns:
            dict: A dictionary containing NumPy arrays for each stream:
                  - 'pose': shape (num_frames, 33, 4) -> x, y, z, visibility
                  - 'left_hand': shape (num_frames, 21, 3) -> x, y, z
                  - 'right_hand': shape (num_frames, 21, 3) -> x, y, z
                  - 'face': shape (num_frames, 92, 3) -> x, y, z
        """
        if not os.path.exists(video_path):
            raise FileNotFoundError(f"Video file not found: {video_path}")

        cap = cv2.VideoCapture(video_path)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        pose_list = []
        left_hand_list = []
        right_hand_list = []
        face_list = []

        # Initialize Holistic tracker in video stream mode
        with self.mp_holistic.Holistic(
            static_image_mode=False,
            model_complexity=self.model_complexity,
            refine_face_landmarks=False,  # Enforce standard 468 face mesh landmarks
            min_detection_confidence=self.min_detection_confidence,
            min_tracking_confidence=self.min_tracking_confidence
        ) as holistic:
            
            pbar = tqdm(total=total_frames, desc=f"Processing {os.path.basename(video_path)}")
            while cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                    break

                # Convert BGR (OpenCV format) to RGB (MediaPipe format)
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                results = holistic.process(frame_rgb)

                # 1. Pose landmarks (33 landmarks, 4D: x, y, z, visibility)
                if results.pose_landmarks:
                    pose = np.array([[lm.x, lm.y, lm.z, lm.visibility] for lm in results.pose_landmarks.landmark])
                else:
                    pose = np.zeros((33, 4))
                pose_list.append(pose)

                # 2. Left Hand landmarks (21 landmarks, 3D: x, y, z)
                if results.left_hand_landmarks:
                    left_hand = np.array([[lm.x, lm.y, lm.z] for lm in results.left_hand_landmarks.landmark])
                else:
                    left_hand = np.zeros((21, 3))
                left_hand_list.append(left_hand)

                # 3. Right Hand landmarks (21 landmarks, 3D: x, y, z)
                if results.right_hand_landmarks:
                    right_hand = np.array([[lm.x, lm.y, lm.z] for lm in results.right_hand_landmarks.landmark])
                else:
                    right_hand = np.zeros((21, 3))
                right_hand_list.append(right_hand)

                # 4. Face landmarks (92 expression landmarks, 3D: x, y, z)
                if results.face_landmarks:
                    all_face = np.array([[lm.x, lm.y, lm.z] for lm in results.face_landmarks.landmark])
                    face = all_face[self.face_indices]
                else:
                    face = np.zeros((len(self.face_indices), 3))
                face_list.append(face)

                pbar.update(1)
            pbar.close()

        cap.release()

        # Concatenate lists into arrays
        return {
            'pose': np.stack(pose_list, axis=0) if pose_list else np.empty((0, 33, 4)),
            'left_hand': np.stack(left_hand_list, axis=0) if left_hand_list else np.empty((0, 21, 3)),
            'right_hand': np.stack(right_hand_list, axis=0) if right_hand_list else np.empty((0, 21, 3)),
            'face': np.stack(face_list, axis=0) if face_list else np.empty((0, 468, 3))
        }

    @staticmethod
    def save_landmarks(landmarks_dict, output_path):
        """
        Saves the extracted landmark dictionary to a compressed .npz file.
        """
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        np.savez_compressed(output_path, **landmarks_dict)
        print(f"Saved landmarks to {output_path}")

    @staticmethod
    def load_landmarks(file_path):
        """
        Loads landmarks from a compressed .npz file.
        """
        with np.load(file_path) as data:
            return {key: data[key] for key in data.files}
