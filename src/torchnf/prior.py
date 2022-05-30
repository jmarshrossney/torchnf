from typing import Union

import torch


class Prior(torch.utils.data.IterableDataset):
    """Wraps around torch.distributions.Distribution to make it iterable."""

    def __init__(
        self,
        distribution: torch.distributions.Distribution,
        batch_size: int = 1,
    ):
        super().__init__()
        assert isinstance(distribution, torch.distributions.Distribution)
        self._distribution = distribution
        self._batch_size = batch_size

    def __call__(self) -> tuple[torch.Tensor]:
        sample = self.sample(self.batch_size)
        log_prob = self.log_prob(sample)
        return sample, log_prob

    def __iter__(self):
        return self

    def __next__(self) -> tuple[torch.Tensor]:
        return self.__call__()

    @property
    def batch_size(self) -> int:
        return self._batch_size

    @batch_size.setter
    def batch_size(self, new: int):
        if type(new) is not int:
            raise TypeError("batch_size should be an integer")
        elif new < 1:
            raise ValueError("batch_size should be positive")
        self._batch_size = new

    @property
    def distribution(self) -> torch.distributions.Distribution:
        return self._distribution

    def sample(self, batch_size: int = 1) -> torch.Tensor:
        return self._distribution.sample([batch_size])

    def log_prob(self, sample: torch.Tensor) -> torch.Tensor:
        return (
            self._distribution.log_prob(sample).flatten(start_dim=1).sum(dim=1)
        )
