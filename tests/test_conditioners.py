import pytest
import torch
from torch.nn.parameter import UninitializedParameter

from torchnf.conditioners import *
from torchnf.transformers import Rescaling


def test_simple_conditioner():
    conditioner = SimpleConditioner([0])
    data = torch.empty(100, 4, 4)
    params = conditioner(data)

    assert params.shape == torch.Size([100, 1, 4, 4])
    assert torch.allclose(params, torch.zeros_like(params))


@pytest.mark.filterwarnings("ignore:Lazy modules")
def test_lazy_simple_conditioner():
    conditioner = LazySimpleConditioner()
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
