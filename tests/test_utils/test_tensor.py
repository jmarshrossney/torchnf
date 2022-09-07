import torch

from torchnf.utils.tensor import *


def test_expand_elements():
    x = torch.rand(10)
    shape = torch.Size([3, 5, 4, 2])
    stack_dim = 1
    out = expand_elements(x, shape, stack_dim)
    assert out.shape == torch.Size([3, 10, 5, 4, 2])


def _test_stacked_nan_to_num():
    # TODO check differen tensor shapes
    pass


def test_scatter_into_nantensor():
    x = torch.rand(10, 2)
    mask = torch.tensor([True, False, True, False])

    y = scatter_into_nantensor(x, mask)

    assert y.shape == torch.Size([10, 1, 4])

    assert torch.equal(y.isnan(), mask.expand(10, 1, 4))

    assert torch.allclose(y[:, 0, ~mask], x)
