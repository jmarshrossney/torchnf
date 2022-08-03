import pytest
import torch

from torchnf.conditioners import *
from torchnf.flow import *
from torchnf.transformers import *


@pytest.mark.parametrize(
    "transformer", [Translation(), Rescaling(), AffineTransform()]
)
def test_identity_simple(transformer):
    conditioner = SimpleConditioner(transformer.identity_params)
    layer = FlowLayer(transformer, conditioner)

    x = torch.empty(100, 4, 4).normal_()
    ldj = torch.zeros(100)
    y, ldj = layer(x, ldj)

    assert torch.allclose(x, y)
    assert torch.allclose(ldj, torch.zeros_like(ldj))
