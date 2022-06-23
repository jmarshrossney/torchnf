"""
"""
import contextlib
import datetime
import math
import random

from jsonargparse.typing import NonNegativeInt
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
    x: torch.Tensor, data_shape: torch.Size, stack_dim: NonNegativeInt = 0
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


def tuple_concat(*tuples: tuple[torch.Tensor]) -> tuple[torch.Tensor]:
    """
    Dim 0 concatenation of tuples of torch.Tensors.

    Example:

        >>> a, b, c = (
                (torch.rand(1), torch.rand(10), torch.rand(1, 10))
                for _ in range(3)
        )
        >>> x, y, z = tuple_concat(a, b, c)
        >>> x.shape, y.shape, z.shape
        (torch.Size([3]), torch.Size([30]), torch.Size([3, 10]))

    """
    return (torch.cat(tensors) for tensors in map(list, zip(*tuples)))


def metropolis_test(log_weights: torch.Tensor) -> list:
    r"""
    Subjects a set of log-weights to the Metropolis test.

    The Metropolis-Hastings algorithm generates an asymptotically unbiased
    sample from some 'target' distribution :math:`p(y)` given a proposal
    distribution :math:`q(y)` which can be sampled from exactly.

    For each member of the sample, :math:`y`, the log-weight is defined as

    .. math::

        \log w(y) = \log p(y) - \log q(y)

    A Markov chain is constructed by running through the sample sequentially
    and transitioning to a new state with probability

    .. math::

        A(y \to y^\prime) = \min \left( 1,
        \frac{q(y)}{p(y)} \frac{p(y^\prime)}{q(y^\prime)} \right) \, .

    Args:
        log_weights
            One-dimensional tensor containing `N` log weights

    Returns:
        A set of `N-1` indices corresponding to the state of the Markov chain
        at each step.

    .. note::

        Note that the Markov chain is initialised using the first element.
    """
    log_weights = log_weights.tolist()
    current = log_weights.pop(0)

    idx = 0
    indices = []

    for proposal in log_weights:
        # Deal with this case separately to avoid overflow
        if proposal > current:
            current = proposal
            idx += 1
        elif random.random() < min(1, math.exp(proposal - current)):
            current = proposal
            idx += 1

        indices.append(idx)

    return indices


@contextlib.contextmanager
def eval_mode(model):
    """
    Temporarily switch to evaluation mode.

    Snippet adapted from user Christoph_Heindl, posted on
    _`discuss.pytorch.org/t/opinion-eval-should-be-a-context-manager/18998`
    (MIT license)
    """
    was_training = model.training
    try:
        model.eval()
        yield model
    finally:
        if was_training:
            model.train()


def raise_(exc: Exception):
    raise Exception
