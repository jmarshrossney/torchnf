import math
import pytest
import random

import torch.distributions

from torchnf.distributions import SimplePrior

_builtin_distributions = [
    torch.distributions.Normal(0, 1),
    torch.distributions.Uniform(0, 2 * math.pi),
    torch.distributions.VonMises(0, 1),
    torch.distributions.MultivariateNormal(torch.zeros(2), torch.eye(2)),
]
_data_shape = [6, 6]
_batch_size = 10


@pytest.mark.parametrize("dist", _builtin_distributions)
def test_prior_construction(dist):
    """Test that prior object can be constructed."""
    _ = SimplePrior(dist, _batch_size, _data_shape)


@pytest.mark.parametrize("dist", _builtin_distributions)
def test_prior_shape(dist):
    """Test that sample and log prob are correct shape."""
    prior = SimplePrior(dist, _batch_size, _data_shape)

    sample, log_prob = next(prior)

    assert sample.shape == torch.Size(
        [_batch_size, *_data_shape, *dist.event_shape]
    )
    assert log_prob.shape == torch.Size([_batch_size])


@pytest.mark.parametrize("dist", _builtin_distributions)
def test_sample_values(dist):
    """Test that sample and log prob match those of given distribution."""
    prior = SimplePrior(dist, _batch_size, _data_shape)

    seed = random.randint(int(1e9), int(1e10))

    torch.manual_seed(seed)
    sample_from_dist = dist.sample([_batch_size, *_data_shape])
    log_prob_from_dist = (
        dist.log_prob(sample_from_dist).flatten(start_dim=1).sum(dim=1)
    )

    torch.manual_seed(seed)
    sample_from_prior, log_prob_from_prior = next(prior)

    assert torch.allclose(sample_from_dist, sample_from_prior)
    assert torch.allclose(log_prob_from_dist, log_prob_from_prior)
