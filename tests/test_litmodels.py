import pytest
import torch
import pytorch_lightning as pl

from torchnf.lit_models import BoltzmannGenerator, OptimizerConfig
from torchnf.distributions import PriorDataModule, expand_dist
from torchnf.transformers import Translation
from torchnf.conditioners import SimpleConditioner
from torchnf.flow import Flow, FlowLayer

TRAIN_STEPS = 1000


@pytest.fixture
def trainer_args():
    # Like fast_dev_run=TRAIN_STEPS, but don't auto-run validation
    # since we can't then recover the metrics!
    return dict(
        max_epochs=1,
        max_steps=TRAIN_STEPS,
        num_sanity_val_steps=0,
        val_check_interval=10,
        check_val_every_n_epoch=10,
        limit_val_batches=10,
        enable_checkpointing=False,
        logger=False,
        enable_progress_bar=False,
        enable_model_summary=False,
    )


def test_shifted_gauss(trainer_args):
    prior = expand_dist(torch.distributions.Normal(0, 1), [36])
    target = expand_dist(torch.distributions.Normal(1, 1), [36])
    transformer = Translation()
    conditioner = SimpleConditioner(transformer.identity_params)
    flow = Flow(FlowLayer(transformer, conditioner))
    model = BoltzmannGenerator(flow, prior, target)
    model.configure_training(batch_size=1000, epoch_length=1000)

    optimizer_config = OptimizerConfig(
        "Adam",
        {"lr": 0.1},
        "CosineAnnealingLR",
        {"T_max": TRAIN_STEPS},
        submodule="flow",
    )
    optimizer_config.add_to(model)

    trainer = pl.Trainer(**trainer_args)
    trainer.fit(model)
    (metrics,) = trainer.validate(model, verbose=False)

    # NOTE: why does this seem to do slightly worse than pure PyTorch??
    assert metrics["Validation"]["AcceptanceRate"] > 0.96

    (metrics,) = trainer.test(model, verbose=False)

    assert metrics["Test"]["AcceptanceRate"] > 0.96


def test_override_dataloader(trainer_args):
    prior = expand_dist(torch.distributions.Normal(0, 1), [36])
    target = expand_dist(torch.distributions.Normal(1, 1), [36])
    transformer = Translation()
    conditioner = SimpleConditioner(transformer.identity_params)
    flow = Flow(FlowLayer(transformer, conditioner))
    model = BoltzmannGenerator(flow, prior, target)

    # Instead of configure_training, we pass a datamodule to trainer
    datamodule = PriorDataModule(prior, batch_size=1000, epoch_length=1000)

    optimizer_config = OptimizerConfig(
        "Adam",
        {"lr": 0.1},
        "CosineAnnealingLR",
        {"T_max": TRAIN_STEPS},
        submodule="flow",
    )
    optimizer_config.add_to(model)

    trainer = pl.Trainer(**trainer_args)
    trainer.fit(model, datamodule=datamodule)
    (metrics,) = trainer.validate(model, datamodule=datamodule, verbose=False)

    # NOTE: why does this seem to do slightly worse than pure PyTorch??
    assert metrics["Validation"]["AcceptanceRate"] > 0.96
