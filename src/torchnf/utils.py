"""
"""
import datetime
import torch


def prod(elements: list[float]) -> float:
    """
    Returns the product of list elements.
    """
    res = 1
    for el in elements:
        res *= el
    return res


def timestamp(fmt: str = "%y%m%dT%H%M%S") -> str:
    """
    Returns a string representation of the current datetime.
    """
    return datetime.datetime.now().strftime(fmt)


def sum_except_batch(x: torch.Tensor) -> torch.Tensor:
    """
    Sum over all but the first dimension of the input tensor.
    """
    return x.flatten(start_dim=1).sum(dim=1)


def expand_elements(
    x: torch.Tensor, data_shape: torch.Size, stack_dim: int = 0
) -> torch.Tensor:
    """
    Expands and stacks each element of a one-dimensional tensor.

    The input tensor is split into chunks of size 1 along the zeroth
    dimension, each element expanded, and then the result stacked.

    Args:
        x
            One-dimensional input tensor
        data_shape
            Shape that each element will be expanded to
        stack_dim
            Dimension that the resulting expanded elements will be
            stacked on

    Effectively this does the following:

    .. code-block:: python

        elements = x.split(1)
        elements = [el.expand(data_shape) for el in elements]
        out = torch.stack(elements, dim=stack_dim)
        return out
    """
    return torch.stack(
        [el.expand(data_shape) for el in x.split(1)],
        dim=stack_dim,
    )
