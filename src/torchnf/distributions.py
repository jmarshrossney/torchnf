"""
"""
import abc
from collections.abc import Iterable

import torch


class Prior(abc.ABC):
    """
    Abstract base class for prior distributions.

    All prior distributions must implement ``sample``:

    .. code:: python

        def sample(self, sample_shape: Iterable) -> torch.Tensor:
            ...

    and ``log_prob``:

    .. code:: python

        def log_prob(self, sample: torch.Tensor) -> torch.Tensor:
            ...
    """

    @classmethod
    def __subclasshook__(cls, C):
        if hasattr(C, "sample") and hasattr(C, "log_prob"):
            return True
        return False

    @abc.abstractmethod
    def sample(self, sample_shape: Iterable[int]) -> torch.Tensor:
        ...

    @abc.abstractmethod
    def log_prob(self, sample: torch.Tensor) -> torch.Tensor:
        ...


class Target(abc.ABC):
    """
    Abstract base class for target distributions.

    All target distributions must implement ``log_prob``.

    .. code:: python

        def log_prob(self, sample: torch.Tensor) -> torch.Tensor:
            ...

    """

    @classmethod
    def __subclasshook__(cls, C):
        if hasattr(C, "log_prob"):
            return True
        return False

    @abc.abstractmethod
    def log_prob(self, sample: torch.Tensor) -> torch.Tensor:
        ...


class IterablePrior(torch.utils.data.IterableDataset):
    """
    Class which wraps around a distribution allow sampling to be iterated.

    Args:
        distribution:
            The distribution, whose ``sample`` method will be called at
            each iteration step
        batch_shape:
            Batch shape returned by each iteration step

    Example:

        >>> # Create a prior distribution using expand_independent
        >>> uv_gauss = torch.distributions.Normal(0, 1)
        >>> prior = expand_independent(uv_gauss, [6, 6])
        >>> prior.sample().shape
        torch.Size([6, 6])
        >>>
        >>> # Make an iterable prior with batch size 100
        >>> iprior = IterablePrior(prior, 100)
        >>> next(iter(iprior)).shape
        torch.Size([100, 6, 6])
        >>>
        >>> # iprior has the same attributes as prior
        >>> iprior.mean()
    """

    def __init__(
        self,
        distribution: torch.distributions.Distribution,
        batch_shape: Iterable[int] = [],
    ) -> None:
        assert isinstance(
            distribution, Prior
        ), "Distribution must implement 'sample' and 'log_prob'"
        self.distribution = distribution.expand(batch_shape)

    def __getattr__(self, attr):
        return getattr(self.distribution, attr)

    def __iter__(self):
        return self

    def __next__(self):
        return self.sample()

    def sample(
        self, sample_shape: Iterable[int] = torch.Size([])
    ) -> torch.Tensor:
        # NOTE: define these explicitly rather than relying on getattr, since
        # otherwise does not register as instance of Prior
        return self.distribution.sample(sample_shape)

    def log_prob(self, sample: torch.Tensor) -> torch.Tensor:
        return self.distribution.log_prob(sample)


def expand_dist(
    distribution: torch.distributions.Distribution,
    event_shape: Iterable[int],
    batch_shape: Iterable[int] = torch.Size([]),
) -> torch.distributions.Independent:
    """
    Constructs a multivariate distribution with iid components.

    The components of the resulting distribution are independent and
    identically distributed according to the input distribution. The
    resulting distribution has an event shape that is the concatenation
    of ``event_shape`` with the shape(s) of the original distribution,
    and a batch shape given by ``batch_shape``.

    Args:
        distribution:
            The distribution of the iid components
        event_shape:
            The event shape of the resulting multivariate distribution,
            not including the shape(s) of the original distribution
        batch_shape:
            The batch shape of the resulting distribution

    Returns:
        A multivariate distibution with independent components

    Example:

        This will create a multivariate Gaussian with diagonal covariance:

        >>> # Create a univariate Gaussian distribution
        >>> uv_gauss = torch.distributions.Normal(0, 1)
        >>> uv_gauss.batch_shape, uv_gauss.event_shape
        (torch.Size([]), torch.Size([]))
        >>> uv_gauss.sample().shape
        torch.Size([])
        >>>
        >>> # Create a multivariate Gaussian with diagonal covariance
        >>> mv_gauss = expand_independent(uv_gauss, [6, 6])
        >>> mv_gauss.batch_shape, mv_gauss.event_shape
        (torch.Size([]), torch.Size([6, 6]))
        >>> mv_gauss.sample().shape
        torch.Size([6, 6])
        >>>
        >>> # What happens when we compute the log-prob?
        >>> uv_gauss.log_prob(mv_gauss.sample()).shape
        torch.Size([6, 6])
        >>> mv_gauss.log_prob(mv_gauss.sample()).shape
        torch.Size([])
    """
    # Expand original distribution by 'event_shape'
    distribution = distribution.expand(event_shape)
    # Register the components as being part of one distribution
    distribution = torch.distributions.Independent(
        distribution, len(distribution.batch_shape)
    )
    # Expand to the batch shape
    distribution = distribution.expand(batch_shape)
    return distribution
