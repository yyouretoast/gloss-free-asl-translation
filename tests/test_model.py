import pytest
import torch
from src.models.translation_model import ASLTranslationModel
from src.models.manual_encoder import ConformerEncoder

@pytest.fixture
def model():
    return ASLTranslationModel(
        input_dim=534,
        d_model=128,
        num_layers=2,
        num_heads=4,
        kernel_size=15,
        t5_model_name="t5-small"
    )

def test_encoder_forward():
    encoder = ConformerEncoder(input_dim=534, d_model=128, num_layers=2)
    x = torch.randn(2, 50, 534)
    mask = torch.ones(2, 50)
    out, out_mask = encoder(x, mask)
    
    assert out.shape == (2, 25, 128)
    assert out_mask.shape == (2, 25)

@pytest.mark.slow
def test_model_forward(model):
    input_features = torch.randn(2, 50, 534)
    attention_mask = torch.ones(2, 50)
    labels = torch.randint(0, 1000, (2, 10))
    
    outputs = model(input_features, attention_mask, labels)
    
    assert outputs.loss is not None
    assert outputs.logits.shape[:2] == (2, 10)

@pytest.mark.slow
def test_model_generate(model):
    input_features = torch.randn(1, 50, 534)
    attention_mask = torch.ones(1, 50)
    
    output_ids = model.generate(input_features, attention_mask, max_new_tokens=5)
    assert output_ids.shape[1] <= 6  # max_new_tokens + 1 (BOS or similar depending on HF config)

@pytest.mark.slow
def test_model_multi_stream_fusion():
    # Initialize model with I3D features enabled
    model_fused = ASLTranslationModel(
        input_dim=501,
        input_i3d_dim=1024,
        d_model=128,
        num_layers=2,
        num_heads=4,
        kernel_size=15,
        t5_model_name="t5-small"
    )
    
    input_features = torch.randn(2, 50, 501)
    input_i3d_features = torch.randn(2, 50, 1024)
    attention_mask = torch.ones(2, 50)
    labels = torch.randint(0, 1000, (2, 10))
    
    # 1. Test forward pass
    outputs = model_fused(
        input_features=input_features,
        attention_mask=attention_mask,
        labels=labels,
        input_i3d_features=input_i3d_features
    )
    assert outputs.loss is not None
    assert outputs.logits.shape[:2] == (2, 10)
    
    # 2. Test generate method with gated fusion enabled
    output_ids = model_fused.generate(
        input_features=input_features,
        attention_mask=attention_mask,
        input_i3d_features=input_i3d_features,
        max_new_tokens=5
    )
    assert output_ids.shape[0] == 2
    assert output_ids.shape[1] <= 6

def test_long_sequence_positional_encoding():
    encoder = ConformerEncoder(input_dim=534, d_model=128, num_layers=1)
    # Sequence length 20100 downsamples to 10050, exceeding 10000
    x = torch.randn(1, 20100, 534)
    mask = torch.ones(1, 20100)
    out, out_mask = encoder(x, mask)
    assert out.shape == (1, 10050, 128)
    assert out_mask.shape == (1, 10050)


