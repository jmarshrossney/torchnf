import math
import random

import torch


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
