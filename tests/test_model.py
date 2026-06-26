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

def test_model_forward(model):
    input_features = torch.randn(2, 50, 534)
    attention_mask = torch.ones(2, 50)
    labels = torch.randint(0, 1000, (2, 10))
    
    outputs = model(input_features, attention_mask, labels)
    
    assert outputs.loss is not None
    assert outputs.logits.shape[:2] == (2, 10)

def test_model_generate(model):
    input_features = torch.randn(1, 50, 534)
    attention_mask = torch.ones(1, 50)
    
    output_ids = model.generate(input_features, attention_mask, max_new_tokens=5)
    assert output_ids.shape[1] <= 6  # max_new_tokens + 1 (BOS or similar depending on HF config)
