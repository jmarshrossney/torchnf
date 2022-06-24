"""
Abstract base classes for the core components of Normalizing Flow models.
"""
import abc
from inspect import signature

import torch

__all__ = [
    "Conditioner",
    "DensityTransform",
    "TargetDistribution",
    "Transformer",
]


def _check_methods_match(cls: type[abc.ABC], C: type, *methods: str) -> bool:
    """
    Check if meth is equivalent in cls and C.
    """
    for m in methods:
        reference = getattr(cls, m)

        # Check that C has the method
        try:
            method = getattr(C, m)
        except AttributeError:
            return False

        reference = signature(reference)
        method = signature(method)

        # Check that the signature is the correct length
        if len(method.parameters) != len(reference.parameters):
            return False

        # NOTE: seems to strict to demand matching type annotations...

    return True


class Conditioner(abc.ABC):
    """
    Abstract base class for conditioners.
    """

    @abc.abstractmethod
    def forward(
        self, inputs: torch.Tensor, context: dict = {}
    ) -> torch.Tensor:
        """
        Computes and returns the forward transformation and log det Jacobian.
        """
        ...

    @classmethod
    def __subclasshook__(cls, C):
        return _check_methods_match(cls, C, "forward")


class DensityTransform(torch.nn.Module, metaclass=abc.ABCMeta):
    """
    Abstract base class for density transformations.
    """

    @abc.abstractmethod
    def forward(
        self, x: torch.Tensor, context: dict = {}
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Computes and returns the forward transformation and log det Jacobian.
        """
        ...

    @abc.abstractmethod
    def inverse(
        self, y: torch.Tensor, context: dict = {}
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Computes and returns the inverse transformation and log det Jacobian.
        """
        ...

    @classmethod
    def __subclasshook__(cls, C):
        return issubclass(C, torch.nn.Module) and _check_methods_match(
            cls, C, "forward", "inverse"
        )


class NetBuilder:
    """
    Abstract base class for neural network builder.

    Network builders implement a ``__call__`` method which returns an
    instance of :py:class:`torch.nn.Sequential`.
    """

    @abc.abstractmethod
    def __call__(self, *args, **kwargs) -> torch.nn.Sequential:
        ...

    @classmethod
    def __subclasshook__(cls, C):
        # this is quite useless
        # Could be reasonable to inspect signature for Sequential
        return hasattr(C, "__call__")


class TargetDistribution(abc.ABC):
    """
    Abstract base class for target distributions.

    All target distributions must implement ``log_prob``.

    .. code:: python

        def log_prob(self, sample: torch.Tensor) -> torch.Tensor:
            ...

    """

    @abc.abstractmethod
    def log_prob(self, sample: torch.Tensor) -> torch.Tensor:
        ...

    @classmethod
    def __subclasshook__(cls, C):
        return _check_methods_match(cls, C, "log_prob")


class Transformer(abc.ABC):
    """
    Abstract base class for transformers.
    """

    @abc.abstractmethod
    def forward(
        self, x: torch.Tensor, params: torch.Tensor, context: dict = {}
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Computes and returns the forward transformation and log det Jacobian.
        """
        ...

    @abc.abstractmethod
    def inverse(
        self, y: torch.Tensor, params: torch.Tensor, context: dict = {}
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Computes and returns the inverse transformation and log det Jacobian.
        """
        ...

    @classmethod
    def __subclasshook__(cls, C):
        return _check_methods_match(cls, C, "forward", "inverse")
