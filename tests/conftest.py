import pytest
import numpy as np
import os
import tempfile

@pytest.fixture
def mock_landmark_data():
    """Generates mock landmark data for testing."""
    num_frames = 100
    return {
        'pose': np.random.rand(num_frames, 33, 4).astype(np.float32),
        'left_hand': np.random.rand(num_frames, 21, 3).astype(np.float32),
        'right_hand': np.random.rand(num_frames, 21, 3).astype(np.float32),
        'face': np.random.rand(num_frames, 92, 3).astype(np.float32)
    }

@pytest.fixture
def mock_dataset_dir(mock_landmark_data):
    """Creates a temporary directory with mock .npz files."""
    with tempfile.TemporaryDirectory() as tmpdir:
        for i in range(3):
            filepath = os.path.join(tmpdir, f"video_{i:03d}.npz")
            np.savez_compressed(filepath, **mock_landmark_data)
        yield tmpdir
