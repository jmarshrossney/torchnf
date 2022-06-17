import math
import pytest
import random
from hypothesis import given, strategies as st

import torch

from torchnf.distributions import Prior, Target, IterablePrior, expand_dist

_distributions = [
    torch.distributions.Normal(0, 1),
    torch.distributions.Uniform(0, 2 * math.pi),
    torch.distributions.VonMises(0, 1),
    torch.distributions.MultivariateNormal(torch.zeros(2), torch.eye(2)),
]

def test_types():
    dist = torch.distributions.Normal(0, 1)
    assert isinstance(dist, Prior)
    assert isinstance(dist, Target)
    assert not isinstance(dist, IterablePrior)

    idist = IterablePrior(dist)
    assert isinstance(idist, Prior)
    assert isinstance(idist, Target)
    assert isinstance(idist, IterablePrior)


@pytest.mark.parametrize("dist", _distributions)
@given(
    data_shape=st.lists(st.integers(1, 10), min_size=0, max_size=2),
    batch_shape=st.lists(st.integers(1, 10), min_size=0, max_size=2),
)
def test_expand_dist(dist, data_shape, batch_shape):
    edist = expand_dist(dist, data_shape, batch_shape)
    assert list(edist.event_shape) == (
        data_shape + list(dist.batch_shape) + list(dist.event_shape)
    )
    assert list(edist.batch_shape) == batch_shape


@pytest.mark.parametrize("prior", _distributions)
def test_iterable_prior(prior):
    iprior = IterablePrior(prior)
    iprior_batch = IterablePrior(prior, [10])

    seed = torch.random.seed()
    sample_1 = iprior.sample()
    torch.random.manual_seed(seed)
    sample_2 = next(iter(iprior))
    assert torch.allclose(sample_1, sample_2)

    assert iprior.sample().shape == prior.event_shape
    assert iprior_batch.sample().shape == torch.Size([10, *prior.event_shape])

    # test inherits attrs from inner dist
    assert hasattr(iprior, "mean")
