import os
import sys
import torch
from torch.utils.data import DataLoader

# Add project root to path to import modules
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from src.dataset import ASLLandmarkDataset, CollateLandmarks
from src.models.manual_encoder import ConformerEncoder

def main():
    print("Testing Dataset and ConformerEncoder integration...\n")
    
    # 1. Setup mock metadata dictionary (mapping our previously generated mock files)
    mock_metadata = {
        'signer01_video_0000': "hello",
        'signer02_video_0001': "please thank you",
        'signer03_video_0002': "good morning",
        'signer01_video_0003': "how are you",
        'signer03_video_0004': "sign language"
    }
    
    # Locate mock landmarks directory
    data_dir = os.path.abspath("data/landmarks")
    if not os.path.exists(data_dir):
        print(f"Warning: Mock data dir {data_dir} not found. Please run src/validate_dataset.py first to generate it.")
        return

    # 2. Instantiate Dataset
    print("Initializing ASLLandmarkDataset (including 92 face expression landmarks)...")
    dataset = ASLLandmarkDataset(
        data_dir=data_dir,
        metadata_dict=mock_metadata,
        max_len=150,
        include_face=True  # Manual (258) + Face (276) = 534 dimensions
    )
    print(f"Dataset initialized with {len(dataset)} samples.")
    
    # 3. Instantiate Collate Wrapper & DataLoader
    collate_fn = CollateLandmarks(tokenizer=None)  # Return raw strings for labels during testing
    dataloader = DataLoader(dataset, batch_size=2, shuffle=True, collate_fn=collate_fn)
    
    # 4. Fetch a batch and verify shape
    print("\nFetching first batch from DataLoader...")
    batch = next(iter(dataloader))
    
    input_features = batch['input_features']
    attention_mask = batch['attention_mask']
    labels = batch['labels']
    
    print(f"Batch loaded:")
    print(f" - input_features shape: {input_features.shape}")
    print(f" - attention_mask shape: {attention_mask.shape}")
    print(f" - Labels text batch:   {labels}")
    
    # Assert input feature dimensions: (batch, seq_len, 534)
    assert len(input_features.shape) == 3, "Input features should be 3D tensor: (batch, seq_len, dim)"
    assert input_features.shape[2] == 534, f"Input dimensions should be 534. Found: {input_features.shape[2]}"
    assert input_features.shape[0] == 2, "Batch size should be 2"
    
    print("Dataset loading and padding checks PASSED!")
    
    # 5. Initialize Conformer Encoder
    print("\nInitializing ConformerEncoder (d_model=512, 4 layers, 4 attention heads)...")
    encoder = ConformerEncoder(
        input_dim=534,
        d_model=512,
        num_layers=4,
        num_heads=4,
        kernel_size=31,
        dropout=0.1
    )
    
    # 6. Run Forward Pass
    print("Running forward pass through ConformerEncoder...")
    outputs, downsampled_mask = encoder(input_features, attention_mask=attention_mask)
    
    print(f"Encoder outputs shape: {outputs.shape}")
    if downsampled_mask is not None:
        print(f"Downsampled mask shape: {downsampled_mask.shape}")
        
    # Sequence length is downsampled by 2 due to Conv1d temporal downsampling
    expected_seq_len = (input_features.shape[1] + 1) // 2
    assert outputs.shape == (2, expected_seq_len, 512), f"Output shape mismatch: {outputs.shape}"
    assert not torch.isnan(outputs).any(), "Found NaN values in encoder outputs!"
    assert not torch.isinf(outputs).any(), "Found Inf values in encoder outputs!"
    
    print("\nConformerEncoder Forward Pass check PASSED!")
    print("=" * 60)
    print("All Model and Dataset tests completed successfully!")
    print("=" * 60)

if __name__ == "__main__":
    main()
