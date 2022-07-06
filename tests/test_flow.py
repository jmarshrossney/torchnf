import pytest
import torch

from torchnf.conditioners import *
from torchnf.flow import *
from torchnf.networks import *
from torchnf.transformers import *


@pytest.mark.parametrize(
    "transformer", [Translation(), Rescaling(), AffineTransform()]
)
def test_identity_simple(transformer):
    conditioner = SimpleConditioner(transformer.identity_params)
    layer = FlowLayer(transformer, conditioner)

    x = torch.empty(100, 4, 4).normal_()
    y, ldj = layer(x)

    assert torch.allclose(x, y)
    assert torch.allclose(ldj, torch.zeros_like(ldj))
