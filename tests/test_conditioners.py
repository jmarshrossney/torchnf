import torch

from torchnf.conditioners import *


def test_simple_conditioner():
    conditioner = SimpleConditioner([0])
    data = torch.empty(100, 4, 4)
    params = conditioner(data)

    assert params.shape == torch.Size([100, 1, 4, 4])
    assert torch.allclose(params, torch.zeros_like(params))
