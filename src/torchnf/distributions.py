"""
"""
import abc
from collections.abc import Iterable
from typing import Optional, Union

import torch
import pytorch_lightning as pl


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


class IterablePrior(torch.utils.data.IterableDataset):
    """
    Wraps a distribution to allow sampling to be iterated over.

    The motivation for this is that instances of IterablePrior may be used
    as ``DataLoader``s in PyTorch Lightning.

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
        batch_size: Union[int, Iterable[int]] = [],
        length: Optional[int] = None,
    ) -> None:
        assert isinstance(
            distribution, Prior
        ), "Distribution must implement 'sample' and 'log_prob'"
        batch_shape = (
            batch_size if isinstance(batch_size, Iterable) else [batch_size]
        )
        self.distribution = distribution.expand(batch_shape)
        self.length = length

    def __getattr__(self, attr):
        return getattr(self.distribution, attr)

    def __iter__(self):
        return self

    def __next__(self) -> tuple[torch.Tensor, torch.Tensor]:
        sample = self.sample()
        return sample, self.log_prob(sample)

    def __len__(self) -> int:
        return self.length

    def sample(
        self, sample_shape: Iterable[int] = torch.Size([])
    ) -> torch.Tensor:
        # NOTE: define these explicitly rather than relying on getattr, since
        # otherwise does not register as instance of Prior
        return self.distribution.sample(sample_shape)

    def log_prob(self, sample: torch.Tensor) -> torch.Tensor:
        return self.distribution.log_prob(sample)


class PriorDataModule(pl.LightningDataModule):
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
        distribution: torch.distributions.Distribution,
        batch_size: int,
        epoch_length: Optional[int] = None,
        *,
        val_batch_size: Optional[int] = None,
        test_batch_size: Optional[int] = None,
        pred_batch_size: Optional[int] = None,
        val_epoch_length: Optional[int] = None,
        test_epoch_length: Optional[int] = None,
        pred_epoch_length: Optional[int] = None,
    ) -> None:
        super().__init__()
        self.distribution = distribution
        self.batch_size = batch_size
        self.epoch_length = epoch_length

        self.val_batch_size = val_batch_size or batch_size
        self.test_batch_size = test_batch_size or batch_size
        self.pred_batch_Size = pred_batch_size or batch_size
        self.val_epoch_length = val_epoch_length or epoch_length
        self.test_epoch_length = test_epoch_length or epoch_length
        self.pred_epoch_length = pred_epoch_length or epoch_length

    def train_dataloader(self) -> IterablePrior:
        return IterablePrior(
            self.distribution, self.batch_size, self.epoch_length
        )

    def val_dataloader(self) -> IterablePrior:
        return IterablePrior(
            self.distribution, self.val_batch_size, self.val_epoch_length
        )

    def test_dataloader(self) -> IterablePrior:
        return IterablePrior(
            self.distribution, self.test_batch_size, self.test_epoch_length
        )

    def predict_dataloader(self) -> IterablePrior:
        return IterablePrior(
            self.distribution, self.pred_batch_size, self.pred_epoch_length
        )
