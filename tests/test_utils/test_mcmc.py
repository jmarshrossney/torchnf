import pytest
import torch

from torchnf.utils.mcmc import *


@pytest.fixture
def trivial_generator():
    def generator():
        torch.manual_seed(123456789)
        while True:
            yield torch.rand([1]), torch.tensor([1])

    return generator


def test_metropolis_hastings(trivial_generator):

    reference = torch.Tensor(
        [k[0] for _, k in zip(range(11), trivial_generator())]
    )[1:]

    generator = trivial_generator()
    init_state = next(generator)
    chain = metropolis_hastings(generator, steps=10, init_state=init_state)

    assert torch.allclose(reference, chain)
