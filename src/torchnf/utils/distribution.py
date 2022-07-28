"""
A collection of utils for constructing objects based on distributions.
"""
from collections.abc import Iterable
from typing import Optional, Union

from jsonargparse.typing import PositiveInt
import torch
from torch.distributions import Distribution
import pytorch_lightning as pl

from torchnf.utils.tensor import sum_except_batch

__all__ = [
    "expand_dist",
    "diagonal_gaussian",
    "DistributionLazyShape",
    "IterableDistribution",
    "DistributionDataModule",
]


def expand_dist(
    distribution: Distribution,
    event_shape: Iterable[PositiveInt],
    batch_shape: Iterable[PositiveInt] = torch.Size([]),
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


def diagonal_gaussian(
    event_shape: Iterable[PositiveInt],
    batch_shape: Iterable[PositiveInt] = torch.Size([]),
) -> torch.distributions.Normal:
    """
    Creates a Gaussian with null mean and unit diagonal covariance.

    This is equivalent to calling :func:`expand_dist` with
    ``distribution=torch.distributions.Normal(0, 1)``.
    """
    return expand_dist(
        torch.distributions.Normal(0, 1), event_shape, batch_shape
    )


class DistributionLazyShape:
    """
    Wraps a distribution to allow sampling various shapes.
    """

    def __init__(self, distribution: Distribution) -> None:
        self.distribution = distribution

    def __getattr__(self, attr):
        return getattr(self.distribution, attr)

    def log_prob(self, sample: torch.Tensor) -> torch.Tensor:
        logp = self.distribution.log_prob(sample)
        return sum_except_batch(logp) if logp.dim() > 1 else logp


class IterableDistribution(torch.utils.data.IterableDataset):
    r"""
    Wraps a distribution to allow sampling to be iterated over.

    The motivation for this is that an instance of IterableDistribution
    may be used as a ``DataLoader`` in PyTorch Lightning.

    Args:
        distribution:
            The distribution, whose ``sample`` method will be called at
            each iteration step
        batch_size:
            Batch size or shape returned by each iteration step
        length:
            Optionally specify a length for the generator, which is
            interpreted by PyTorch Lightning as the number of steps in
            a single 'epoch'

    Example:

        >>> # Create a prior distribution using expand_independent
        >>> uv_gauss = torch.distributions.Normal(0, 1)
        >>> prior = expand_independent(uv_gauss, [6, 6])
        >>> prior.sample().shape
        torch.Size([6, 6])
        >>>
        >>> # Make an iterable prior with batch size 100
        >>> iprior = IterableDistribution(prior, 100)
        >>> next(iter(iprior)).shape
        torch.Size([100, 6, 6])
        >>>
        >>> # iprior has the same attributes as prior
        >>> iprior.mean()

    """

    def __init__(
        self,
        distribution: torch.distributions.Distribution,
        batch_size: Union[PositiveInt, Iterable[PositiveInt]] = [],
        length: Optional[PositiveInt] = None,
    ) -> None:
        assert isinstance(distribution, Distribution)
        batch_shape = (
            batch_size if isinstance(batch_size, Iterable) else [batch_size]
        )
        self.distribution = distribution.expand(batch_shape)
        self.length = length

    def __getattr__(self, attr):
        return getattr(self.distribution, attr)

    def __iter__(self):
        return self

    def __next__(self) -> torch.Tensor:
        return self.sample()

    def __len__(self) -> PositiveInt:
        return self.length

    def sample(
        self, sample_shape: Iterable[PositiveInt] = torch.Size([])
    ) -> torch.Tensor:
        # NOTE: define these explicitly rather than relying on getattr, since
        # otherwise does not register as instance of Prior
        return self.distribution.sample(sample_shape)

    def log_prob(self, sample: torch.Tensor) -> torch.Tensor:
        return self.distribution.log_prob(sample)


class DistributionDataModule(pl.LightningDataModule):
    """
    Wraps a distribution in a DataModule.

    Args:
        distribution:
            The distribution serving as the prior for a Normalizing Flow
        batch_size:
            Size of each batch drawn from the distribution
        epoch_length:
            Number of batches constituting an 'epoch'
        val/test/pred_batch_size:
            Batch sizes for the validation, test, predict steps, if they
            should be different than the training ``batch_size``
        val/test/pred_epoch_length:
            Epoch lengths for validation, test, predict steps, if they
            should be different than the training ``epoch_length``

    .. note::
        The epoch length has no significance whatsoever, since batches
        are generated on demand and never recycled. However, since Pytorch
        Lightning has lots of hooks which execute at the start and end of
        an epoch, it can be convenient to define an epoch in terms of a
        fixed number of batches.

    """

    def __init__(
        self,
        distribution: Distribution,
        batch_size: PositiveInt,
        *,
        train_epoch: Optional[PositiveInt] = None,
        val_batch_size: Optional[PositiveInt] = None,
        val_epoch: PositiveInt = 1,
        test_batch_size: Optional[PositiveInt] = None,
        test_epoch: PositiveInt = 1,
        predict_batch_size: Optional[PositiveInt] = None,
    ) -> None:
        super().__init__()
        self.distribution = distribution
        self.batch_size = batch_size
        self.train_epoch = train_epoch
        self.val_batch_size = val_batch_size or batch_size
        self.val_epoch = val_epoch
        self.test_batch_size = test_batch_size or batch_size
        self.test_epoch = test_epoch
        self.predict_batch_size = predict_batch_size or batch_size

    def train_dataloader(self) -> IterableDistribution:
        """
        Returns an iterable version of the prior distribution.
        """
        return IterableDistribution(
            self.distribution, self.batch_size, self.train_epoch
        )

    def val_dataloader(self) -> IterableDistribution:
        """
        Returns an iterable version of the prior distribution.
        """
        return IterableDistribution(
            self.distribution, self.val_batch_size, self.val_epoch
        )

    def test_dataloader(self) -> IterableDistribution:
        """
        Returns an iterable version of the prior distribution.
        """
        return IterableDistribution(
            self.distribution, self.test_batch_size, self.test_epoch
        )

    def predict_dataloader(self) -> IterableDistribution:
        """
        Returns an iterable version of the prior distribution.
        """
        return IterableDistribution(
            self.distribution, self.predict_batch_size, None
        )
