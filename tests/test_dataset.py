import pytest
from src.dataset import ASLLandmarkDataset, CollateLandmarks

def test_dataset_loading_and_normalization(mock_dataset_dir):
    metadata = {
        "video_000": "hello world",
        "video_001": "how are you",
        "video_002": "good morning"
    }
    
    dataset = ASLLandmarkDataset(
        data_dir=mock_dataset_dir,
        metadata_dict=metadata,
        max_len=150,
        include_face=True,
        normalize=True
    )
    
    assert len(dataset) == 3
    
    sample = dataset[0]
    assert 'features' in sample
    assert 'text' in sample
    assert 'file_id' in sample
    
    # 258 for mediapipe manual + 276 face = 534
    assert sample['features'].shape[1] == 534

def test_dataset_missing_labels(mock_dataset_dir):
    metadata = {
        "video_000": "hello world",
        # video_001 missing
        "video_002": "" # empty label
    }
    
    dataset = ASLLandmarkDataset(
        data_dir=mock_dataset_dir,
        metadata_dict=metadata,
        skip_empty_labels=True
    )
    
    # Should only load video_000
    assert len(dataset) == 1

@pytest.mark.slow
def test_collate_landmarks():
    from transformers import T5TokenizerFast
    tokenizer = T5TokenizerFast.from_pretrained("t5-small")
    
    collate = CollateLandmarks(tokenizer=tokenizer, max_target_len=10)
    
    import torch
    batch = [
        {'features': torch.randn(50, 534), 'text': "hello", 'file_id': "id1"},
        {'features': torch.randn(30, 534), 'text': "world", 'file_id': "id2"}
    ]
    
    collated = collate(batch)
    
    assert collated['input_features'].shape == (2, 50, 534)
    assert collated['attention_mask'].shape == (2, 50)
    assert collated['labels'].shape == (2, 10)
    
    # Check padding mask
    assert collated['attention_mask'][0, -1] == 1.0
    assert collated['attention_mask'][1, -1] == 0.0 # Padded

def test_dataset_holistic_loading(mock_holistic_dir):
    metadata = {
        "video_000": "hello world",
        "video_001": "how are you",
        "video_002": "good morning"
    }
    
    # 1. Test with face included (501 dimensions)
    dataset_with_face = ASLLandmarkDataset(
        data_dir=mock_holistic_dir,
        metadata_dict=metadata,
        max_len=150,
        include_face=True,
        normalize=True
    )
    assert len(dataset_with_face) == 3
    sample_with_face = dataset_with_face[0]
    assert sample_with_face['features'].shape[1] == 501
    
    # 2. Test with face excluded (225 dimensions)
    dataset_no_face = ASLLandmarkDataset(
        data_dir=mock_holistic_dir,
        metadata_dict=metadata,
        max_len=150,
        include_face=False,
        normalize=True
    )
    assert len(dataset_no_face) == 3
    sample_no_face = dataset_no_face[0]
    assert sample_no_face['features'].shape[1] == 225

def test_dataset_multi_stream_loading_and_collation(mock_holistic_dir, mock_i3d_dir):
    metadata = {
        "video_000": "hello world",
        "video_001": "how are you",
        "video_002": "good morning"
    }
    
    # 1. Test dataset __getitem__ returning both streams
    dataset = ASLLandmarkDataset(
        data_dir=mock_holistic_dir,
        metadata_dict=metadata,
        max_len=150,
        include_face=True,
        normalize=True,
        i3d_dir=mock_i3d_dir
    )
    
    assert len(dataset) == 3
    sample = dataset[0]
    assert 'features' in sample
    assert 'i3d_features' in sample
    
    # Verify time alignment (same frame length)
    landmark_len = sample['features'].shape[0]
    i3d_len = sample['i3d_features'].shape[0]
    assert landmark_len == i3d_len
    assert sample['i3d_features'].shape[1] == 1024
    
    # 2. Test batch collator with I3D features
    collate = CollateLandmarks(tokenizer=None, max_target_len=10)
    batch = [dataset[0], dataset[1]]
    collated = collate(batch)
    
    assert 'input_features' in collated
    assert 'input_i3d_features' in collated
    
    max_seq_len = max(sample['features'].shape[0], dataset[1]['features'].shape[0])
    assert collated['input_features'].shape == (2, max_seq_len, 501)
    assert collated['input_i3d_features'].shape == (2, max_seq_len, 1024)


