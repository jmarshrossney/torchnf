import math
from typing import Optional


from jsonargparse.typing import (
    PositiveInt,
    PositiveFloat,
    ClosedUnitInterval,
    OpenUnitInterval,
)
import torch
from torch.utils.data import TensorDataset, DataLoader, random_split
import pytorch_lightning as pl

__all__ = [
    "Moons",
]

PI = math.pi


class _ToyDataModule(pl.LightningDataModule):
    def __init__(
        self,
        batch_size: PositiveInt,
        train_frac: ClosedUnitInterval = 0.7,
    ) -> None:
        self.batch_size = batch_size
        self.train_frac = train_frac

        self.prepare_data_per_node = False

    def setup(self, stage: Optional[str] = None) -> None:
        """
        Setup stage.

        Calls :meth:`generate_data`, wrapping the resulting tensor
        in a :py:class:`torch.utils.data.TensorDataset`. Then, performs
        the train/validation/test split and sets state.
        """
        dataset = TensorDataset(self.generate_data())

        n_total = len(dataset)
        n_train = int(n_total * self.train_frac)
        n_val = (n_total - n_train) // 2
        n_test = n_total - n_val - n_train

        (
            self.train_dataset,
            self.val_dataset,
            self.test_dataset,
        ) = random_split(dataset, [n_train, n_val, n_test])

    def train_dataloader(self) -> DataLoader:
        """
        Returns a dataloader with the training set.
        """
        return DataLoader(self.train_dataset, self.batch_size, num_workers=1)

    def val_dataloader(self) -> DataLoader:
        """
        Returns a dataloader with the validation set.
        """
        return DataLoader(self.val_dataset, self.batch_size, num_workers=1)

    def test_dataloader(self) -> DataLoader:
        """
        Returns a dataloader with the test set.
        """
        return DataLoader(self.test_dataset, self.batch_size, num_workers=1)

    def predict_dataloader(self) -> DataLoader:
        """
        Returns a dataloader with the test set.
        """
        return DataLoader(self.test_dataset, self.batch_size, num_workers=1)

    def generate_data(self) -> torch.Tensor:
        """
        Generates the dataset. Called in :meth:`setup`.
        """
        raise NotImplementedError


def _generate_moons(
    sample_size: PositiveInt,
    *,
    noise: Optional[PositiveFloat] = None,
    seed: PositiveInt = None,
) -> torch.Tensor:
    if seed is not None:
        torch.random.manual_seed(seed)

    u = torch.empty(sample_size).uniform_(0, PI)
    x = torch.stack([u.cos(), u.sin()]).T

    # transform ~ 1/2 by a y-axis reflection and shift
    k = torch.randint(0, 2, [sample_size]).bool()
    x[k, 0] = 1 - x[k, 0]
    x[k, 1] = 1 - x[k, 1] - 0.5

    if noise is not None:
        x.add_(torch.empty_like(x).normal_(mean=0, std=noise))

    return x


class Moons(_ToyDataModule):
    """
    A 2-dimensional toy dataset which resembles moons in the (x, y) plane.
    """

    def __init__(
        self,
        total_size: PositiveInt,
        batch_size: PositiveInt,
        noise: OpenUnitInterval = 0,
        train_frac: ClosedUnitInterval = 0.7,
    ) -> None:
        super().__init__(batch_size, train_frac)
        self.save_hyperparameters()

    def generate_data(self) -> torch.Tensor:
        return _generate_moons(
            self.hparams.total_size, noise=self.hparams.noise
        )
