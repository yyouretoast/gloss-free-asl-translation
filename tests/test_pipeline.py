import pytest
import os
import cv2
import tempfile
import numpy as np


@pytest.mark.slow
def test_pipeline_extraction():
    pytest.importorskip("mediapipe")
    try:
        from src.data_pipeline import ASLLandmarkExtractor

        extractor = ASLLandmarkExtractor()
    except AttributeError as e:
        pytest.skip(
            f"MediaPipe solutions module is not available in this environment: {e}"
        )

    with tempfile.NamedTemporaryFile(suffix=".mp4", delete=False) as tmp:
        video_path = tmp.name

    try:
        # Create a dummy video
        out = cv2.VideoWriter(
            video_path, cv2.VideoWriter_fourcc(*"mp4v"), 30, (640, 480)
        )
        for _ in range(10):
            frame = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
            out.write(frame)
        out.release()

        results = extractor.extract_landmarks(video_path)

        assert "pose" in results
        assert "left_hand" in results
        assert "right_hand" in results
        assert "face" in results

        assert results["pose"].shape == (10, 33, 4)
        assert results["left_hand"].shape == (10, 21, 3)
        assert results["right_hand"].shape == (10, 21, 3)
        assert results["face"].shape == (10, 92, 3)

    finally:
        if os.path.exists(video_path):
            os.remove(video_path)
