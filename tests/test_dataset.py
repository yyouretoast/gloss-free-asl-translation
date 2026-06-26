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
