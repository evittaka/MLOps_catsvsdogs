import pytest
import torch
from catsvsdogs.model import MobileNetV3
from omegaconf import OmegaConf
from torch import nn


@pytest.fixture
def mock_cfg():
    """Mock configuration for testing."""
    return OmegaConf.create(
        {
            "model": {
                "pretrained": False  # Set pretrained to False to avoid external dependencies
            }
        }
    )


def test_model_setup(mock_cfg):
    """Test the MobileNetV3 model class."""
    model = MobileNetV3(mock_cfg)
    # Check if model is an instance of nn.Module
    assert isinstance(model, nn.Module), "Model is not an instance of nn.Module"


@pytest.mark.parametrize("batch_size", [1, 4, 8, 16, 32, 64, 128])
def test_model_behaviour(mock_cfg, batch_size: int):
    """Test the MobileNetV3 model with different batch sizes."""
    model = MobileNetV3(mock_cfg)
    # Check data pipeline
    dummy_input = torch.randn(batch_size, 3, 128, 128)  # Input tensor with batch size, 3 channels, 128x128 resolution
    output = model(dummy_input)
    assert output.shape == torch.Size([batch_size, 2]), (
        f"Expected output to have shape [batch_size, 2], got {output.shape}"
    )
