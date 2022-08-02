import itertools
import pathlib
import warnings

from jsonargparse import ArgumentParser, ActionConfigFile, class_from_function
from jsonargparse.typing import PositiveInt
import matplotlib.pyplot as plt
import pytorch_lightning as pl
import torch

from torchnf.abc import Transformer
from torchnf.conditioners import MaskedConditioner, SimpleConditioner
from torchnf.model import FlowBasedModel
from torchnf.networks import DenseNet
from torchnf.layers import FlowLayer, Flow
from torchnf.transformers import Rescaling
from torchnf.utils.datasets import Moons
from torchnf.utils.distribution import diagonal_gaussian
from torchnf.utils.decorators import skip_if_logging_disabled

DEFAULT_CONFIG = pathlib.Path(__file__).with_name("default_config.yaml")

DEFAULT_TRAINER_CONFIG = dict(
    gpus=torch.cuda.device_count(),
    enable_checkpointing=False,
    logger=pl.loggers.TensorBoardLogger(save_dir=".", default_hp_metric=False),
)

# I don't care about how many workers the dataloader has
warnings.filterwarnings(
    action="ignore",
    category=pl.utilities.warnings.PossibleUserWarning,
)


class Model(FlowBasedModel, pl.LightningModule):
    def __init__(self, flow: Flow) -> None:
        super().__init__()
        self.flow = flow
        self.prior = diagonal_gaussian([2])

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.flow.parameters(), lr=0.001)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=self.trainer.max_epochs
        )
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "interval": "epoch",
                "frequency": 1,
            },
        }

    def training_step(
        self, batch: list[torch.Tensor], *_, **__
    ) -> torch.Tensor:
        (x,) = batch
        outputs = self.inference_step(x)
        loss = outputs.pop("loss")
        self.log("train_loss", loss)
        return loss

    def validation_step(
        self, batch: list[torch.Tensor], *_, **__
    ) -> torch.Tensor:
        (x,) = batch
        outputs = self.inference_step(x)
        loss = outputs.pop("loss")
        self.log("val_loss", loss, on_step=False, on_epoch=True)
        return outputs["sample"]

    @skip_if_logging_disabled
    def validation_epoch_end(self, outputs: list[torch.Tensor]):
        """Logs a scatterplot of validation outputs."""
        z = torch.cat(outputs)

        fig, ax = plt.subplots()
        ax.scatter(*z.T)
        self.logger.experiment.add_figure(
            "Encoded_data", fig, self.global_step
        )

    @skip_if_logging_disabled
    def on_validation_epoch_end(self):
        sample_size = len(self.trainer.datamodule.val_dataset)
        x = self.sample(sample_size)
        fig, ax = plt.subplots()
        ax.scatter(*x.T)
        self.logger.experiment.add_figure(
            "Decoded_noise", fig, self.global_step
        )

    def test_step(self, batch: list[torch.Tensor], *_, **__) -> torch.Tensor:
        (x,) = batch
        outputs = self.inference_step(x)
        loss = outputs.pop("loss")
        self.log("test_loss", loss, on_step=False, on_epoch=True)

    @skip_if_logging_disabled
    def test_epoch_end(self, _) -> None:
        # Log test loss along with hparams for tabular comparison
        self.logger.log_hyperparams(
            self.hparams, metrics=self.trainer.logged_metrics
        )


def make_flow(
    transformer: Transformer,
    net: DenseNet,
    depth: PositiveInt,
) -> Flow:
    mask = torch.tensor([True, False], dtype=bool)
    conditioner = lambda mask_: MaskedConditioner(  # noqa: E731
        net(1, transformer.n_params), mask_
    )
    layers = [
        FlowLayer(transformer, conditioner(m))
        for _, m in zip(range(depth), itertools.cycle([mask, ~mask]))
    ]
    layers.append(FlowLayer(Rescaling(), SimpleConditioner([0])))
    return Flow(*layers)


parser = ArgumentParser(default_config_files=[str(DEFAULT_CONFIG)])

parser.add_argument(
    "--epochs", type=PositiveInt, help="Number of epochs to train for"
)
parser.add_class_arguments(class_from_function(make_flow), "flow")
parser.add_class_arguments(Moons, "moons")
parser.add_argument("-c", "--config", action=ActionConfigFile)


def main(config: dict = {}, trainer_config: dict = {}):

    config = parser.parse_object(config) if config else parser.parse_args()

    config_as_flat_dict = {
        k: v for k, v in vars(config.as_flat()).items() if "config" not in k
    }

    config = parser.instantiate_classes(config)

    flow, moons, epochs = config.flow, config.moons, config.epochs

    model = Model(flow)
    model.save_hyperparameters(config_as_flat_dict)

    trainer_config = (
        DEFAULT_TRAINER_CONFIG | {"max_epochs": epochs} | trainer_config
    )
    trainer = pl.Trainer(**trainer_config)
    trainer.fit(model, moons)

    (metrics,) = trainer.test(model, moons)

    return metrics


if __name__ == "__main__":
    main()
