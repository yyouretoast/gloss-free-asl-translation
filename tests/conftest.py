import pytest
import numpy as np
import os
import tempfile


@pytest.fixture
def mock_landmark_data():
    """Generates mock landmark data for testing."""
    num_frames = 100
    return {
        "pose": np.random.rand(num_frames, 33, 4).astype(np.float32),
        "left_hand": np.random.rand(num_frames, 21, 3).astype(np.float32),
        "right_hand": np.random.rand(num_frames, 21, 3).astype(np.float32),
        "face": np.random.rand(num_frames, 92, 3).astype(np.float32),
    }


@pytest.fixture
def mock_dataset_dir(mock_landmark_data):
    """Creates a temporary directory with mock .npz files."""
    with tempfile.TemporaryDirectory() as tmpdir:
        for i in range(3):
            filepath = os.path.join(tmpdir, f"video_{i:03d}.npz")
            np.savez_compressed(filepath, **mock_landmark_data)
        yield tmpdir


@pytest.fixture
def mock_holistic_data():
    """Generates mock MediaPipe 3D Holistic combined landmark data."""
    num_frames = 100
    # 543 landmarks, 3 coordinates each (x, y, z)
    return np.random.rand(num_frames, 543, 3).astype(np.float32)


@pytest.fixture
def mock_holistic_dir(mock_holistic_data):
    """Creates a temporary directory with mock MediaPipe 3D Holistic .npy files."""
    with tempfile.TemporaryDirectory() as tmpdir:
        for i in range(3):
            filepath = os.path.join(tmpdir, f"video_{i:03d}.npy")
            np.save(filepath, mock_holistic_data)
        yield tmpdir


@pytest.fixture
def mock_i3d_data():
    """Generates mock I3D spatiotemporal feature data."""
    num_frames = 100
    # 1024-dimensional feature vectors
    return np.random.rand(num_frames, 1024).astype(np.float32)


@pytest.fixture
def mock_i3d_dir(mock_i3d_data):
    """Creates a temporary directory with mock precomputed I3D features."""
    with tempfile.TemporaryDirectory() as tmpdir:
        for i in range(3):
            # Filenames map directly to sentence clips
            filepath = os.path.join(tmpdir, f"video_{i:03d}.npy")
            np.save(filepath, mock_i3d_data)
        yield tmpdir
