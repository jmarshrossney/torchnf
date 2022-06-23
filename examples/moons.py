import itertools
import jsonargparse
from jsonargparse.typing import PositiveInt
import pathlib
import torch
import pytorch_lightning as pl
import matplotlib.pyplot as plt
import types

import torchnf.lit_models
import torchnf.distributions
import torchnf.flow

from torchnf.lit_models import OptimizerConfig
from torchnf.transformers import Transformer
from torchnf.data.toy_datasets import Moons
from torchnf.recipes.layers import CouplingLayerDenseNet
from torchnf.recipes.networks import DenseNet

default_config = (
    pathlib.Path(__file__)
    .parent.joinpath("default_config")
    .joinpath("moons.yaml")
)


def make_flow(
    transformer: Transformer,
    net: DenseNet,
    flow_depth: PositiveInt,
) -> torchnf.flow.Flow:
    mask = torch.tensor([True, False], dtype=bool)
    layer_x = CouplingLayerDenseNet(transformer, net, mask)
    layer_y = CouplingLayerDenseNet(transformer, net, ~mask)
    layers = [
        layer()
        for _, layer in zip(
            range(flow_depth), itertools.cycle([layer_x, layer_y])
        )
    ]
    return torchnf.flow.Flow(*layers)


Flow = jsonargparse.class_from_function(make_flow)

parser = jsonargparse.ArgumentParser(
    prog="Moons", default_config_files=[str(default_config)]
)

parser.add_class_arguments(Flow, "flow")
parser.add_argument("--optimizer", type=OptimizerConfig)
parser.add_argument("--epochs", type=PositiveInt)
parser.add_class_arguments(Moons, "moons")
parser.add_class_arguments(pl.Trainer, "trainer")

parser.add_argument("-c", "--config", action=jsonargparse.ActionConfigFile)


def main(config: dict = {}):

    # Parse args and instantiate classes
    config = parser.parse_object(config) if config else parser.parse_args()
    config = parser.instantiate_classes(config)
    flow, optimizer, trainer, moons = (
        config.flow,
        config.optimizer,
        config.trainer,
        config.moons,
    )

    # Build model - a bijective auto-encoder
    prior = torchnf.distributions.expand_dist(
        torch.distributions.Normal(0, 1), [2]
    )
    model = torchnf.lit_models.BijectiveAutoEncoder(flow, prior)

    # Add an extra method which collects all of the data generated
    # during validation and plots a scatter
    def validation_epoch_end(self, data: list[torch.Tensor]):
        x, y = torch.cat(data).T
        fig, ax = plt.subplots()
        ax.scatter(x, y)
        self.logger.experiment.add_figure(
            "Validation/encoded_data", fig, self.global_step
        )

    model.validation_epoch_end = types.MethodType(validation_epoch_end, model)

    # Add the optimizer and lr scheduler
    optimizer.add_to(model)

    # Train
    trainer.fit(model, datamodule=moons)

    (metrics,) = trainer.test(model, datamodule=moons)

    print(metrics)

    return metrics


if __name__ == "__main__":
    main()
