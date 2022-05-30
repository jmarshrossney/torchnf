import math
import pytest
import random

import torch.distributions

from torchnf.prior import Prior

_event_shape = [6, 6]
_builtin_distributions = [
    torch.distributions.Normal(
        loc=torch.zeros(_event_shape), scale=torch.ones(_event_shape)
    ),
    torch.distributions.Uniform(
        low=torch.zeros(_event_shape),
        high=torch.full(_event_shape, 2 * math.pi),
    ),
    torch.distributions.VonMises(
        loc=torch.zeros(_event_shape),
        concentration=torch.ones(_event_shape),
    ),
]
_batch_size = 10


@pytest.mark.parametrize("dist", _builtin_distributions)
def test_sample_shape(dist):
    assert dist.sample().shape == torch.Size(_event_shape)
    assert dist.sample([1]).shape == torch.Size([1, *_event_shape])
    assert dist.sample([2, 1]).shape == torch.Size([2, 1, *_event_shape])


@pytest.mark.parametrize("dist", _builtin_distributions)
def test_prior_construction(dist):
    """Test that prior object can be constructed."""
    _ = Prior(dist, _batch_size)


@pytest.mark.parametrize("dist", _builtin_distributions)
def test_prior_shape(dist):
    """Test that sample and log prob are correct shape."""
    prior = Prior(dist, _batch_size)

    sample, log_prob = next(prior)

    assert sample.shape == torch.Size([_batch_size, *_event_shape])
    assert log_prob.shape == torch.Size([_batch_size])


@pytest.mark.parametrize("dist", _builtin_distributions)
def test_sample_values(dist):
    """Test that sample and log prob match those of given distribution."""
    prior = Prior(dist, _batch_size)

    seed = random.randint(int(1e9), int(1e10))

    torch.manual_seed(seed)
    sample_from_dist = dist.sample([_batch_size])
    log_prob_from_dist = (
        dist.log_prob(sample_from_dist).flatten(start_dim=1).sum(dim=1)
    )

    torch.manual_seed(seed)
    sample_from_prior, log_prob_from_prior = next(prior)

    assert torch.allclose(sample_from_dist, sample_from_prior)
    assert torch.allclose(log_prob_from_dist, log_prob_from_prior)
