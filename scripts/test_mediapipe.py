import cv2
import mediapipe as mp
import numpy as np

def main():
    print("Testing MediaPipe and OpenCV installation...")
    print(f"OpenCV version: {cv2.__version__}")
    print(f"MediaPipe version: {mp.__version__}")
    
    # Initialize Holistic
    mp_holistic = mp.solutions.holistic
    
    # Create a blank black image
    dummy_image = np.zeros((480, 640, 3), dtype=np.uint8)
    
    print("Initializing MediaPipe Holistic model...")
    with mp_holistic.Holistic(
        static_image_mode=True,
        model_complexity=1,
        refine_face_landmarks=False
    ) as holistic:
        # Convert BGR (OpenCV) to RGB (MediaPipe)
        rgb_image = cv2.cvtColor(dummy_image, cv2.COLOR_BGR2RGB)
        
        # Process the image
        results = holistic.process(rgb_image)
        
        print("Model initialized and dummy frame processed successfully!")
        
        # Verify landmarks are None (since it's a blank black image)
        # but the code shouldn't crash.
        if results.pose_landmarks is None:
            print("No landmarks detected (expected on blank black frame).")
        else:
            print("Detected landmarks (unexpected but processed).")

if __name__ == "__main__":
    main()
