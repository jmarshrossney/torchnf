import itertools
import jsonargparse
from jsonargparse.typing import PositiveInt
import pathlib
import torch
import pytorch_lightning as pl
import matplotlib.pyplot as plt

from torchnf.lit_models import BijectiveAutoEncoder, OptimizerConfig
from torchnf.distributions import expand_dist
from torchnf.flow import Flow

from torchnf.data.toy_datasets import Moons

from torchnf.recipes.layers import AffineCouplingLayer
from torchnf.recipes.networks import DenseNet

_default_config = str(pathlib.Path(__file__).with_name("moons_config.yaml"))

parser = jsonargparse.ArgumentParser(default_config_files=[_default_config])
parser.add_argument("--net_hidden_shape", type=PositiveInt, nargs="*")
parser.add_argument("--net_activation", type=str)
parser.add_argument("--flow_depth", type=PositiveInt)
parser.add_argument("--epochs", type=PositiveInt)
parser.add_argument("--optimizer", type=OptimizerConfig)
parser.add_class_arguments(Moons, "moons")
parser.add_argument("-c", "--config", action=jsonargparse.ActionConfigFile)


class Model(BijectiveAutoEncoder):
    def validation_epoch_end(self, z) -> None:
        z = torch.cat(z)
        fig, ax = plt.subplots()
        ax.scatter(*z.T)

        self.logger.experiment.add_figure(
            "Validation/encoded_data", fig, self.global_step
        )


def main():

    config = parser.parse_args()
    # config = parser.instantiate_classes(config)

    net = DenseNet(
        in_features=1,
        out_features=2,
        hidden_shape=config.net_hidden_shape,
        activation=config.net_activation,
    )
    layer_x = AffineCouplingLayer(net, mask=torch.Tensor([1, 0]).bool())
    layer_y = AffineCouplingLayer(net, mask=torch.Tensor([0, 1]).bool())

    layers = [
        layer()
        for i, layer in zip(
            range(config.flow_depth), itertools.cycle([layer_x, layer_y])
        )
    ]
    flow = Flow(*layers)

    latent = expand_dist(torch.distributions.Normal(0, 1), [2])

    model = Model(flow, latent)

    optimizer = OptimizerConfig(**config.optimizer)
    optimizer.add_to(model)

    moons = Moons(**config.moons)

    trainer = pl.Trainer(
        enable_checkpointing=False,
        max_epochs=config.epochs,
        check_val_every_n_epoch=1,
    )
    trainer.fit(model, datamodule=moons)

    # z = torch.cat(trainer.predict(model, datamodule=moons))
    # plt.scatter(z[:, 0], z[:, 1], alpha=0.5)
    # plt.show()


if __name__ == "__main__":
    main()
