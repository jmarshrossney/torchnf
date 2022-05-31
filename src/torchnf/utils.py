import datetime
import torch


def timestamp(fmt: str = "%y%m%dT%H%M%S") -> str:
    return datetime.datetime.now().strftime(fmt)


def sum_except_batch(x: torch.Tensor) -> torch.Tensor:
    return x.flatten(start_dim=1).sum(dim=1)


def expand_elements(
    x: torch.Tensor, data_shape: torch.Size, stack_dim: int
) -> torch.Tensor:
    return torch.stack(
        [el.expand(data_shape) for el in x.split(1)],
        dim=stack_dim,
    )
