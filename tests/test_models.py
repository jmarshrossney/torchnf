import torch

from torchnf.models import BoltzmannGenerator
from torchnf.prior import SimplePrior
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


def test_shifted_gaussian_target():
    torch.manual_seed(123456789)
    model = MultivariateGaussianSampler(
        flow=Flow(FlowLayer(Translation(), SimpleConditioner([0]))),
        loc=torch.full([36], 1),
        covariance_matrix=torch.eye(36),
        batch_size=100,
    )
    model.fit(1000)
    final_metrics = model.validate()
    assert final_metrics["acceptance"] > 0.98


def test_shifted_rescaled_gaussian_target():
    torch.manual_seed(123456789)
    model = MultivariateGaussianSampler(
        flow=Flow(FlowLayer(AffineTransform(), SimpleConditioner([0, 0]))),
        loc=torch.full([36], 1),
        covariance_matrix=torch.eye(36).mul(0.5),
        batch_size=500,
    )
    model.fit(1000)
    final_metrics = model.validate()
    assert final_metrics["acceptance"] > 0.98


def _test_correlated_gaussian_target():
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
        batch_size=500,
    )
    model.fit(1000)
    final_metrics = model.validate()
    assert final_metrics["acceptance"] > 0.98


def test_coupling_densenet():
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
        batch_size=500,
    )
    model.fit(1000)
    final_metrics = model.validate()
    assert final_metrics["acceptance"] > 0.92


def test_coupling_convnet():
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
        batch_size=500,
    )
    model.fit(1000)
    final_metrics = model.validate()
    assert final_metrics["acceptance"] > 0.92
