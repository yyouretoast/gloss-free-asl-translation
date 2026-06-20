import os
import sys
import numpy as np
import cv2

# Add the project root to path so we can import src modules
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from src.data_pipeline import ASLLandmarkExtractor

def create_dummy_video(output_path, duration_seconds=3, fps=30, width=640, height=480):
    """
    Creates a simple dummy video with a moving white circle on a black background
    to simulate a frame stream for the landmark extractor.
    """
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
    
    total_frames = duration_seconds * fps
    for i in range(total_frames):
        # Create a black frame
        frame = np.zeros((height, width, 3), dtype=np.uint8)
        
        # Draw a moving circle (to simulate some activity in frame)
        cx = int(width / 2 + 100 * np.sin(2 * np.pi * i / fps))
        cy = int(height / 2 + 100 * np.cos(2 * np.pi * i / fps))
        cv2.circle(frame, (cx, cy), 50, (255, 255, 255), -1)
        
        out.write(frame)
        
    out.release()
    print(f"Created dummy video at {output_path} ({total_frames} frames)")

def main():
    dummy_video_path = os.path.abspath("data/dummy_test.mp4")
    output_npz_path = os.path.abspath("data/dummy_landmarks.npz")
    
    # 1. Create a dummy video
    create_dummy_video(dummy_video_path, duration_seconds=3, fps=30)
    
    # 2. Extract landmarks
    print("\nInitializing ASLLandmarkExtractor...")
    extractor = ASLLandmarkExtractor(model_complexity=1)
    
    print("Running extraction...")
    landmarks = extractor.extract_landmarks(dummy_video_path)
    
    # 3. Verify structures
    num_frames = 3 * 30  # 3 seconds * 30 fps = 90 frames
    print("\nVerifying extracted landmark shapes:")
    for key, arr in landmarks.items():
        print(f" - Key: '{key}', Shape: {arr.shape}")
        
    # Check shape constraints
    assert landmarks['pose'].shape == (num_frames, 33, 4), f"Pose shape mismatch: {landmarks['pose'].shape}"
    assert landmarks['left_hand'].shape == (num_frames, 21, 3), f"Left hand shape mismatch: {landmarks['left_hand'].shape}"
    assert landmarks['right_hand'].shape == (num_frames, 21, 3), f"Right hand shape mismatch: {landmarks['right_hand'].shape}"
    assert landmarks['face'].shape == (num_frames, 92, 3), f"Face shape mismatch: {landmarks['face'].shape}"
    print("Shape validations passed successfully!")

    # 4. Test Serialization
    print("\nTesting serialization...")
    extractor.save_landmarks(landmarks, output_npz_path)
    assert os.path.exists(output_npz_path), "Failed to save NPZ file!"
    
    loaded_landmarks = extractor.load_landmarks(output_npz_path)
    for key in landmarks.keys():
        assert np.array_equal(landmarks[key], loaded_landmarks[key]), f"Data mismatch after loading key '{key}'!"
    
    print("Serialization and verification tests PASSED!")
    
    # Cleanup temporary test files
    try:
        os.remove(dummy_video_path)
        os.remove(output_npz_path)
        print("Cleaned up temporary test files.")
    except Exception as e:
        print(f"Warning during cleanup: {e}")

if __name__ == "__main__":
    main()
