from collections.abc import Iterable
import math
from typing import Optional, Union

from jsonargparse.typing import PositiveInt, PositiveFloat
import torch
import pytorch_lightning as pl

import torchnf.distributions

PI = math.pi


class DiagonalGaussian(torchnf.distributions.PriorDataModule):
    def __init__(
        data_shape: Iterable[PositiveInt],
        batch_size: PositiveInt,
        epoch_length: Optional[PositiveInt] = None,
        **kwargs: dict[str, Optional[PositiveInt]],
    ) -> None:
        distribution = torchnf.distributions.expand_dist(
            torch.distributions.Normal(0, 1),
            data_shape,
        )
        super().__init__(distribution, batch_size, epoch_length, **kwargs)
