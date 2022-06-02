"""
"""
import torch


class Prior:
    """
    Base class for prior distributions.

    Args:
        batch_size
            Number of independent and identically distributed data
            points to draw when sampling. This can be modified after
            instantiation.

    Derived classes should override
        - :meth:`sample`
        - :meth:`log_prob`

    TODO: discuss requirements for sample
    TODO: discuss requirements for log_prob
    """

    def __init__(self, batch_size: int = 1) -> None:
        self._batch_size = batch_size

    def __iter__(self):
        return self

    def __next__(self) -> tuple[torch.Tensor]:
        """
        Alias for :meth:`forward`.
        """
        return self.forward()

    def __call__(self, sample: torch.Tensor) -> torch.Tensor:
        """
        Alias for :meth:`log_prob`.
        """
        return self.log_prob(sample)

    @property
    def batch_size(self) -> int:
        """
        The batch size returned by :meth:`sample`.
        """
        return self._batch_size

    @batch_size.setter
    def batch_size(self, new: int) -> None:
        assert type(new) is int and new > 0
        self._batch_size = new

    def forward(self) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Returns a new sample and its log probability density.

        This simply does:

        .. code-block:: python

            sample = self.sample()
            return sample, self.log_prob(sample)
        """
        sample = self.sample()
        return sample, self.log_prob(sample)

    def sample(self) -> torch.Tensor:
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

    def sample(self) -> torch.Tensor:
        return self.distribution.sample([self.batch_size])

    def log_prob(self, sample: torch.Tensor) -> torch.Tensor:
        return self.distribution.log_prob(sample)
