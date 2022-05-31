import torch

from torchnf.utils import sum_except_batch, expand_elements


def test_expand_elements():
    x = torch.rand(10)
    shape = torch.Size([3, 5, 4, 2])
    stack_dim = 1
    out = expand_elements(x, shape, stack_dim)
    assert out.shape == torch.Size([3, 10, 5, 4, 2])
