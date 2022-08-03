from itertools import cycle
import pathlib
import warnings

from jsonargparse import ArgumentParser, ActionConfigFile, class_from_function
from jsonargparse.typing import PositiveInt
import matplotlib.pyplot as plt
import pytorch_lightning as pl
import torch

from torchnf.abc import Transformer
from torchnf.conditioners import SimpleConditioner
from torchnf.model import FlowBasedModel
from torchnf.flow import FlowLayer, Flow
from torchnf.transformers import Rescaling
from torchnf.utils.datasets import Moons
from torchnf.utils.distribution import diagonal_gaussian
from torchnf.utils.decorators import skip_if_logging_disabled
from torchnf.utils.nn import Activation, make_fnn

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
    net_hidden_shape: list[PositiveInt],
    net_activation: Activation,
    net_final_activation: Activation,
    depth: PositiveInt,
) -> Flow:

    select_x = (
        lambda mod, input: input[0].select(1, 0).unsqueeze(-1)
    )  # noqa: E731
    select_y = (
        lambda mod, input: input[0].select(1, 1).unsqueeze(-1)
    )  # noqa: E731
    join_x = lambda mod, input, output: torch.dstack(  # noqa: E731
        (torch.full_like(output, float("nan")), output)
    )
    join_y = lambda mod, input, output: torch.dstack(  # noqa: E731
        (output, torch.full_like(output, float("nan")))
    )

    flow = []
    for _, pre_hook, post_hook in zip(
        range(depth), cycle((select_x, select_y)), cycle((join_x, join_y))
    ):
        net = make_fnn(
            in_features=1,
            out_features=transformer.n_params,
            hidden_shape=net_hidden_shape,
            activation=net_activation,
            final_activation=net_final_activation,
        )
        net.register_forward_pre_hook(pre_hook)
        net.register_forward_hook(post_hook)

        flow.append(FlowLayer(transformer, net))

    flow.append(FlowLayer(Rescaling(), SimpleConditioner([0])))
    return Flow(*flow)


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
