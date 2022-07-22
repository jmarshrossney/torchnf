from collections.abc import Iterator, Sized
import logging
import math
import random
from typing import Optional

from jsonargparse.typing import PositiveInt
import torch

log = logging.getLogger(__name__)

__all__ = [
    "metropolis_test",
    "metropolis_hastings",
]


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


def metropolis_hastings(
    generator: Iterator[tuple[torch.Tensor, torch.Tensor]],
    steps: Optional[PositiveInt] = None,
    init_state: Optional[tuple[torch.Tensor, torch.Tensor]] = None,
) -> None:
    """
    Builds a Markov chain using the Metropolis Hastings algorithm.
    """
    current = init_state or next(generator)

    if isinstance(generator, Sized):  # i.e. hasattr __len__
        generator_length = len(generator)
        if steps is not None:
            if generator_length < steps:
                log.warning("`steps` is larger than the generator size")
        else:
            steps = generator_length
    else:
        if steps is None:
            raise ValueError("Require `steps` for infinite iterators")

    chain = []
    for step in range(steps):
        proposal = next(generator)

        log_delta_weight = float(proposal[1] - current[1])
        if log_delta_weight > 0:
            current = proposal
        elif random.random() < min(1, math.exp(log_delta_weight)):
            current = proposal

        chain.append(current[0])

    return torch.cat(chain)
