import pytest
import torch

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
        train_steps=1000,
        train_batch_size=100,
        val_steps=1,
        val_batch_size=1000,
        optimizer="Adam",
        optimizer_kwargs={"lr": 0.1},
        scheduler="CosineAnnealingLR",
        scheduler_kwargs={"T_max": 1000},
        val_interval=None,
        ckpt_interval=None,
        pbar_interval=None,
        logging_interval=None,
    )


def test_sampling(training_spec):
    model = MultivariateGaussianSampler(
        flow=Flow(FlowLayer(Translation(), SimpleConditioner([0]))),
        loc=torch.full([4], 1),
        covariance_matrix=torch.eye(4),
    )
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


def test_combine_metrics(training_spec):
    # TODO: this test belongs in test_metrics
    model = MultivariateGaussianSampler(
        flow=Flow(FlowLayer(Translation(), SimpleConditioner([0]))),
        loc=torch.full([4], 1),
        covariance_matrix=torch.eye(4),
    )
    training_spec.update(val_steps=10)
    model.configure_training(**training_spec)
    model.fit()
    final_metrics = model.validate()
    assert all(
        [isinstance(metric, torch.Tensor) for metric in final_metrics.values()]
    )
    assert all(
        [metric.shape == torch.Size([10]) for metric in final_metrics.values()]
    )


def test_shifted_gaussian_target(training_spec):
    torch.manual_seed(123456789)
    model = MultivariateGaussianSampler(
        flow=Flow(FlowLayer(Translation(), SimpleConditioner([0]))),
        loc=torch.full([36], 1),
        covariance_matrix=torch.eye(36),
    )
    model.configure_training(**training_spec)
    model.fit()
    final_metrics = model.validate()
    assert final_metrics["acceptance"] > 0.98


def test_shifted_rescaled_gaussian_target(training_spec):
    torch.manual_seed(123456789)
    model = MultivariateGaussianSampler(
        flow=Flow(FlowLayer(AffineTransform(), SimpleConditioner([0, 0]))),
        loc=torch.full([36], 1),
        covariance_matrix=torch.eye(36).mul(0.5),
    )
    model.configure_training(**training_spec)
    model.fit()
    final_metrics = model.validate()
    assert final_metrics["acceptance"] > 0.97


def _test_correlated_gaussian_target(training_spec):
    # NOTE: simple conditioner + affine transform not enough to model
    # correlated target
    # It's vector-vector product when we need a matrix-vector product
    torch.manual_seed(123456789)
    scale_tril = (
        torch.empty([36, 36]).uniform_(0, 0.1).tril(-1).add(torch.eye(36))
    )
    model = MultivariateGaussianSampler(
        flow=Flow(
            FlowLayer(AffineTransform(), SimpleConditioner(torch.zeros(2, 36)))
        ),
        loc=torch.empty([36]).uniform_().sub(0.5),
        covariance_matrix=torch.mm(scale_tril, scale_tril.T),
    )
    model.configure_training(**training_spec)
    model.fit()
    final_metrics = model.validate()
    assert final_metrics["acceptance"] > 0.98


def test_coupling_densenet(training_spec):
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
    model.configure_training(**training_spec)
    model.fit()
    final_metrics = model.validate()
    assert final_metrics["acceptance"] > 0.9


def test_coupling_convnet(training_spec):
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
    model.configure_training(**training_spec)
    model.fit()
    final_metrics = model.validate()
    assert final_metrics["acceptance"] > 0.9
