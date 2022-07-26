import itertools
import pathlib
import types
import warnings

import jsonargparse
from jsonargparse.typing import PositiveInt
import matplotlib.pyplot as plt
import pytorch_lightning as pl
import torch

from torchnf.abc import Transformer
from torchnf.conditioners import MaskedConditioner
from torchnf.models import BijectiveAutoencoder
from torchnf.networks import DenseNet
from torchnf.layers import FlowLayer, Composition
from torchnf.utils.datasets import Moons
from torchnf.utils.distribution import diagonal_gaussian
from torchnf.utils.optim import OptimizerConfig
from torchnf.utils.tensor import tuple_concat

_DEFAULT_CONFIG = (
    pathlib.Path(__file__)
    .parent.joinpath("default_config")
    .joinpath("moons.yaml")
)

# I don't care about how many workers the dataloader has
warnings.filterwarnings(
    action="ignore",
    category=pl.utilities.warnings.PossibleUserWarning,
)


def make_flow(
    transformer: Transformer,
    net: DenseNet,
    flow_depth: PositiveInt,
) -> Composition:
    mask = torch.tensor([True, False], dtype=bool)

    conditioner = lambda mask_: MaskedConditioner(  # noqa: E731
        net(1, transformer.n_params), mask_
    )

    layers = [
        FlowLayer(transformer, conditioner(m))
        for _, m in zip(range(flow_depth), itertools.cycle([mask, ~mask]))
    ]
    return Composition(*layers)


Flow = jsonargparse.class_from_function(make_flow)

parser = jsonargparse.ArgumentParser(
    prog="Moons", default_config_files=[str(_DEFAULT_CONFIG)]
)

parser.add_class_arguments(Flow, "flow")
parser.add_class_arguments(OptimizerConfig, "optimizer", skip={"submodule"})
parser.add_class_arguments(Moons, "moons")
parser.add_argument("--epochs", type=PositiveInt)

parser.add_argument("-c", "--config", action=jsonargparse.ActionConfigFile)


# Add an extra method which collects all of the data generated
# during validation and plots a scatter
def validation_epoch_end(self, data: list[torch.Tensor]):
    z, _ = tuple_concat(*data)
    fig, ax = plt.subplots()
    ax.scatter(*z.T)
    self.logger.experiment.add_figure("Encoded_data", fig, self.global_step)


# Add a hook which plots decoded Gaussian
def on_validation_epoch_end(self):
    sample_size = len(self.trainer.datamodule._val_dataset)
    x, _ = self.sample(sample_size)
    fig, ax = plt.subplots()
    ax.scatter(*x.T)
    self.logger.experiment.add_figure("Decoded_noise", fig, self.global_step)


def main(config: dict = {}):

    # Parse args and instantiate classes
    config = parser.parse_object(config) if config else parser.parse_args()
    flattened_config_dict = vars(config.as_flat())
    config = parser.instantiate_classes(config)

    flow, optimizer, moons, epochs = (
        config.flow,
        config.optimizer,
        config.moons,
        config.epochs,
    )

    # Build model - a bijective auto-encoder
    prior = diagonal_gaussian([2])
    model = BijectiveAutoencoder(flow, prior)

    # Add hparams for easier comparison
    for param in [
        "__default_config__",
        "config",
        "flow.net.in_features",
        "flow.net.out_features",
    ]:
        flattened_config_dict.pop(param)
    model.save_hyperparameters(flattened_config_dict)

    # Add extra methods for visualising in tensorboard
    model.validation_epoch_end = types.MethodType(validation_epoch_end, model)
    model.on_validation_epoch_end = types.MethodType(
        on_validation_epoch_end, model
    )

    # Add the optimizer and lr scheduler
    optimizer.add_to(model)

    # Train
    trainer = pl.Trainer(
        max_epochs=epochs,
        gpus=torch.cuda.device_count(),
        enable_checkpointing=False,
    )
    trainer.fit(model, moons)

    (metrics,) = trainer.test(model, moons, verbose=True)

    return metrics


if __name__ == "__main__":
    main()
