import pytest
import torch
from catsvsdogs.train import loss_function

# Test output for different batch sizes
@pytest.mark.parametrize("batch_size", [1, 4, 8, 16, 32, 64, 128])
def test_loss_function_behaviour(batch_size: int) -> None:
    loss_fn = loss_function()

    assert callable(loss_fn), "Loss function needs to be a callable function"

    # Create dummy data to pass to the model.
    # Use random values with correct shape for training images
    # and random 1D torch with values 0 or 1 for training targets
    output_dummy = torch.randn(batch_size, 1)
    target_dummy = torch.randn(batch_size, 1)

    loss = loss_fn(output_dummy, target_dummy)
    assert torch.is_tensor(loss), "Loss function should output a tensor"
    assert loss.size() == torch.Size([]), "Loss function output should be scalar"
