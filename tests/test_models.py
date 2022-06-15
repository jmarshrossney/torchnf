import pytest
import torch
import torchmetrics

from torchnf.models import BoltzmannGenerator
from torchnf.distributions import SimplePrior
from torchnf.flow import Flow, FlowLayer
from torchnf.transformers import AffineTransform, Translation
from torchnf.conditioners import (
    SimpleConditioner,
    MaskedConditioner,
    MaskedConditionerStructurePreserving,
)
from torchnf.recipes.networks import DenseNet, ConvNetCircular
import torchnf.utils

from torchnf.recipes.models import MultivariateGaussianSampler
from torchnf.recipes.layers import AffineCouplingLayer


# torch.use_deterministic_algorithms(True)


@pytest.fixture
def training_spec():
    return dict(
        steps=1000,
        batch_size=100,
        optimizer="Adam",
        optimizer_kwargs={"lr": 0.1},
        scheduler="CosineAnnealingLR",
        scheduler_kwargs={"T_max": 1000},
        ckpt_interval=None,
        pbar_interval=None,
    )


@pytest.fixture
def val_metrics():
    return torchnf.metrics.LogStatWeightMetrics(mcmc=True)


@pytest.fixture
def validation_spec(val_metrics):
    return dict(
        batch_size=1000,
        batches=1,
        metrics=val_metrics,
        interval=None,
    )


def test_sampling(training_spec):
    model = MultivariateGaussianSampler(
        flow=Flow(FlowLayer(Translation(), SimpleConditioner([0]))),
        loc=torch.full([4], 1),
        covariance_matrix=torch.eye(4),
    )
    model.no_logging()
    model.configure_training(**training_spec)
    model.fit()

    sample, log_weights = model.weighted_sample(1000)
    markov_chain = model.mcmc_sample(1000)
    assert len(sample) == 1000
    assert len(log_weights) == 1000
    assert len(markov_chain) == 1000

    sample, log_weights = model.weighted_sample(1000, 2)
    markov_chain = model.mcmc_sample(1000, 2)
    assert len(sample) == 2000
    assert len(log_weights) == 2000
    assert len(markov_chain) == 2000


def test_shifted_gaussian_target(training_spec, validation_spec):
    torch.manual_seed(123456789)
    model = MultivariateGaussianSampler(
        flow=Flow(FlowLayer(Translation(), SimpleConditioner([0]))),
        loc=torch.full([36], 1),
        covariance_matrix=torch.eye(36),
    )
    model.no_logging()
    model.configure_training(**training_spec)
    model.configure_validation(**validation_spec)
    model.fit()
    final_metrics = model.validate()
    assert final_metrics["AcceptanceRate"] > 0.98


def test_shifted_rescaled_gaussian_target(training_spec, validation_spec):
    torch.manual_seed(123456789)
    model = MultivariateGaussianSampler(
        flow=Flow(FlowLayer(AffineTransform(), SimpleConditioner([0, 0]))),
        loc=torch.full([36], 1),
        covariance_matrix=torch.eye(36).mul(0.5),
    )
    model.no_logging()
    model.configure_training(**training_spec)
    model.configure_validation(**validation_spec)
    model.fit()
    final_metrics = model.validate()
    assert final_metrics["AcceptanceRate"] > 0.97


def test_coupling_densenet(training_spec, validation_spec):
    mask = torch.zeros(36).bool()
    mask[::2] = True
    net = DenseNet(
        in_features=18,
        out_features=36,  # s, t params
        hidden_shape=[18],
        activation="Tanh",
    )
    layer1 = AffineCouplingLayer(net, mask)
    layer2 = AffineCouplingLayer(net, ~mask)
    flow = Flow(layer1(), layer2())

    model = MultivariateGaussianSampler(
        flow=flow,
        loc=torch.ones(36),
        covariance_matrix=torch.eye(36),
    )
    model.no_logging()
    model.configure_training(**training_spec)
    model.configure_validation(**validation_spec)
    model.fit()
    final_metrics = model.validate()
    assert final_metrics["AcceptanceRate"] > 0.9


def test_coupling_convnet(training_spec, validation_spec):
    mask = torch.zeros(36).bool()
    mask[::2] = True
    net = ConvNetCircular(
        dim=1,
        in_channels=1,
        out_channels=2,
        hidden_shape=[1],
        activation="Tanh",
        kernel_size=3,
    )
    flow = Flow(
        FlowLayer(
            AffineTransform(),
            MaskedConditionerStructurePreserving(
                net(), mask, create_channel_dim=True
            ),
        ),
        FlowLayer(
            AffineTransform(),
            MaskedConditionerStructurePreserving(
                net(), ~mask, create_channel_dim=True
            ),
        ),
    )

    model = MultivariateGaussianSampler(
        flow=flow,
        loc=torch.ones(36),
        covariance_matrix=torch.eye(36),
    )
    model.no_logging()
    model.configure_training(**training_spec)
    model.configure_validation(**validation_spec)
    model.fit()
    final_metrics = model.validate()
    assert final_metrics["AcceptanceRate"] > 0.9
