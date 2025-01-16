import pytest
import torch
from catsvsdogs.model import MobileNetV3
from torch import nn


def test_model_setup():
    """Test the SimpleCNN model class."""
    model = MobileNetV3()
    # Check if model is an instance of nn.Module
    assert isinstance(model, nn.Module), "Model is not an instance of nn.Module"


# Test output for different batch sizes
@pytest.mark.parametrize("batch_size", [1, 4, 8, 16, 32, 64, 128])
def test_model_behaviour(batch_size: int) -> None:
    model = MobileNetV3()
    # Check data pipeline
    dummy_input = torch.randn(batch_size, 3, 128, 128)
    output = model(dummy_input)
    assert output.shape == torch.Size([batch_size, 2]), "Expected output to have shape [batch_size, 2]"
