"""
"""
import abc
import torch


class Prior:
    r"""
    Base class for prior distributions.

    The two essential requirements for a prior distribution are (a)
    that it can be (efficiently) sampled from, and (b) that the log-
    probability of a sample can be computed up to an unimportant
    normalisation. Hence, derived classes should override :meth:`sample`
    and :meth:`log_prob`.

    Objects derived from this class can also be iterated over:

    .. code-block:: python

        prior = Prior(...)
        prior = iter(prior)  # constructs a generator
        sample, log_prob = next(prior)

    This behaviour is similar to that of an
    :py:class:`torch.utils.data.IterableDataset`, which means they can
    be used in place of a ``DataLoader`` in PyTorch Lightning.

    Args:
        batch_size
            The default batch size, i.e. number of independent and
            identically distributed data points to draw when sampling.
            Used where prior is called as in ``next(iter(self))``.
    """

    def __init__(self, batch_size: int = 1) -> None:
        self._batch_size = batch_size

    def __iter__(self):
        return self

    def __next__(self) -> tuple[torch.Tensor]:
        """
        Alias for ``self.forward(self.batch_size)``.
        """
        return self.forward(self._batch_size)

    def __call__(self, sample: torch.Tensor) -> torch.Tensor:
        """
        Alias for :meth:`log_prob`.
        """
        return self.log_prob(sample)

    @property
    def batch_size(self) -> int:
        """
        The default batch size used when calling ``next(self)``.
        """
        return self._batch_size

    @batch_size.setter
    def batch_size(self, new: int) -> None:
        assert isinstance(new, int) and new > 0
        self._batch_size = new

    def forward(self, batch_size: int) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Returns a new sample and its log probability density.

        This simply does:

        .. code-block:: python

            sample = self.sample(batch_size)
            return sample, self.log_prob(sample)
        """
        sample = self.sample(batch_size)
        return sample, self.log_prob(sample)

    def sample(self, batch_size: int) -> torch.Tensor:
        """
        Returns a new sample.
        """
        raise NotImplementedError

    def log_prob(self, sample: torch.Tensor) -> torch.Tensor:
        """
        Computes the log probability density for a given sample.
        """
        raise NotImplementedError


class SimplePrior(Prior):
    """Wraps around torch.distributions.Distribution to make it iterable."""

    def __init__(
        self,
        distribution: torch.distributions.Distribution,
        batch_size: int = 1,
        expand_shape: list[int] = [],
    ):
        super().__init__(batch_size)
        distribution = distribution.expand(expand_shape)
        distribution = torch.distributions.Independent(
            distribution, len(distribution.batch_shape)
        )
        self.distribution = distribution

    def sample(self, batch_size: int) -> torch.Tensor:
        return self.distribution.sample([batch_size])

    def log_prob(self, sample: torch.Tensor) -> torch.Tensor:
        return self.distribution.log_prob(sample)


class Target(abc.ABC):
    """
    Abstract base class for target distributions.

    All target distributions must implement ``log_prob``

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
