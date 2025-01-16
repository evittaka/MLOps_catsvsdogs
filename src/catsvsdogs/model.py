import timm
import typer
from hydra import compose, initialize
from hydra.core.global_hydra import GlobalHydra
from omegaconf import DictConfig
from torch import nn

app = typer.Typer()

class MobileNetV3(nn.Module):
    def __init__(self, cfg: DictConfig):
        super(MobileNetV3, self).__init__()
        self.model = timm.create_model("mobilenetv3_large_100", pretrained=cfg.model.pretrained)
        self.model.classifier = nn.Linear(self.model.classifier.in_features, 2)

    def forward(self, x):
        return self.model(x)


@app.command()
def print_model():
    """Print the model architecture based on the configuration."""
    if not GlobalHydra().is_initialized():
        initialize(config_path="../../configs", job_name="model", version_base=None)
    hydra_cfg = compose(config_name="config")

    model = MobileNetV3(hydra_cfg)
    print(model)


if __name__ == "__main__":
    app()
