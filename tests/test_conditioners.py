import pytest
import torch
from torch.nn.parameter import UninitializedParameter

from torchnf.conditioners import *
from torchnf.transformers import Translation, Rescaling


def test_trainable_parameters():
    conditioner = TrainableParameters([0])
    data = torch.empty(100, 4, 4)
    params = conditioner(data)

    assert params.shape == torch.Size([100, 1, 4, 4])
    assert torch.allclose(params, torch.zeros_like(params))


@pytest.mark.filterwarnings("ignore:Lazy modules")
def test_lazy_trainable_parameters():
    conditioner = TrainableParameters()
    transformer = Rescaling()
    conditioner.transformer = transformer

    assert isinstance(conditioner.params, UninitializedParameter)

    data = torch.rand(10, 4, 4)
    params = conditioner(data)

    assert torch.allclose(
        conditioner.params, transformer.identity_params.float()
    )

    outputs, _ = transformer(data, params)
    assert torch.allclose(outputs, data)

    # With context
    params = conditioner(data, {"_": None})
    assert torch.allclose(
        conditioner.params, transformer.identity_params.float()
    )

    params = conditioner(data, context={"_": None})
    assert torch.allclose(
        conditioner.params, transformer.identity_params.float()
    )


def test_simple_fnn_conditioner():
    N = 4
    mask = torch.tensor([True, False])
    net = simple_fnn_conditioner(
        transformer=Translation(),
        mask=mask,
        hidden_shape=[100],
        activation=torch.nn.Identity(),
        final_activation=torch.nn.Identity(),
        bias=True,
    )
    input = torch.rand(N, 2)

    # Check correct shape
    assert net(input).shape == torch.Size([N, 1, 2])

    # Check outputs are NaNs where inputs were not masked
    assert torch.equal(net(input).isnan(), mask.expand(N, 1, 2))

    input_2 = torch.stack([input[:, 0], torch.rand(N)], dim=1)
    assert torch.allclose(net(input), net(input_2), equal_nan=True)
