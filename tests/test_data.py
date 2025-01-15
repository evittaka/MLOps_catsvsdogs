from catsvsdogs.data import catsvsdogs
from torch.utils.data import Dataset


def test_catsvsdogs():
    """Test the catsvsdogs class."""
    train_set, test_set = catsvsdogs()
    assert isinstance(train_set, Dataset), "Train set is not a Torch.Dataset type"
    assert isinstance(test_set, Dataset), "Test set is not a Torch.Dataset type"
