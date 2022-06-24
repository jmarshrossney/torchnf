"""
Module containing convenience classes which generate neural networks for
use in conditioners.

Specifically, classes defined herein implement a :code:`__call__()` method
which returns an instance of :py:class:`torch.nn.Sequential` instance
representing the neural network. This allows one to easily generate multiple
independent network (e.g. one for each flow layer) using the same spec.

TODO: document a convenient way of creating new class with identical
parameters except for some user-specified changes E.g. dataclasses.replace
"""
import dataclasses
from typing import Optional

from jsonargparse.typing import PositiveInt, restricted_number_type
import torch

__all__ = [
    "DenseNet",
    "ConvNet",
    "ConvNetCircular",
    "GenericNetBuilder",
]

ConvDim = restricted_number_type(
    name="ConvDim",
    base_type=int,
    restrictions=[(">", 0), ("<", 1)],
    join="and",
    docstring="Convolution dimension: {1, 2, 3} are supported.",
)


def raise_(exc: Exception):
    raise Exception


@dataclasses.dataclass
class DenseNet:
    """
    Fully-connected feed-forward neural network.

    See :py:class:`torch.nn.Linear`.

    .. note:: ``in_features`` and ``out_features`` may alternatively
       be passed to ``__call__`` rather than ``__init__``
    """

    hidden_shape: list[PositiveInt]
    activation: str
    activation_kwargs: dict = dataclasses.field(default_factory=dict)
    skip_final_activation: bool = False
    linear_kwargs: dict = dataclasses.field(default_factory=dict)
    in_features: Optional[PositiveInt] = None
    out_features: Optional[PositiveInt] = None

    def _activation(self) -> torch.nn.Module:
        return getattr(torch.nn, self.activation)(**self.activation_kwargs)

    def __call__(
        self,
        in_features: Optional[PositiveInt] = None,
        out_features: Optional[PositiveInt] = None,
    ) -> torch.nn.Sequential:
        # Maybe: (allows other recipes to set values, e.g. out_features)
        # def __call__(self, replacements: dict):
        #     self.__dict__.update(replacements)
        # -- or, for frozen instance --
        #     config = dataclasses.asdict(self).update(replacements)
        in_features = in_features or self.in_features
        out_features = (
            out_features
            or self.out_features
            or raise_(ValueError("'out_features' must be provided"))
        )

        net_shape = [in_features, *self.hidden_shape, out_features]
        activations = [self._activation() for _ in self.hidden_shape] + [
            torch.nn.Identity()
            if self.skip_final_activation
            else self._activation()
        ]
        layers = []
        for f_in, f_out, activation in zip(
            net_shape[:-1], net_shape[1:], activations
        ):
            linear = (
                torch.nn.LazyLinear(f_out, **self.linear_kwargs)
                if f_in is None
                else torch.nn.Linear(f_in, f_out, **self.linear_kwargs)
            )
            layers.append(linear)
            layers.append(activation)

        return torch.nn.Sequential(*layers)


@dataclasses.dataclass
class ConvNet:
    """
    Convolutional neural network.

    See :py:class:`torch.nn.Conv2d`, and 1d/3d versions.

    .. note::

        ``in_channels`` and ``out_channels`` may alternatively
        be passed to ``__call__`` rather than ``__init__``
    """

    dim: ConvDim
    hidden_shape: list[PositiveInt]
    activation: str
    kernel_size: PositiveInt
    activation_kwargs: dict = dataclasses.field(default_factory=dict)
    skip_final_activation: bool = False
    conv_kwargs: dict = dataclasses.field(default_factory=dict)
    in_channels: Optional[PositiveInt] = None
    out_channels: Optional[PositiveInt] = None

    def _activation(self) -> torch.nn.Module:
        return getattr(torch.nn, self.activation)(**self.activation_kwargs)

    def __call__(
        self,
        in_channels: Optional[PositiveInt] = None,
        out_channels: Optional[PositiveInt] = None,
    ) -> torch.nn.Sequential:

        in_channels = in_channels or self.in_channels
        out_channels = (
            out_channels
            or self.out_channels
            or raise_(ValueError("'out_channels' must be provided"))
        )

        net_shape = [in_channels, *self.hidden_shape, out_channels]
        activations = [self._activation() for _ in self.hidden_shape] + [
            torch.nn.Identity()
            if self.skip_final_activation
            else self._activation()
        ]
        Conv = {
            1: torch.nn.Conv1d,
            2: torch.nn.Conv2d,
            3: torch.nn.Conv3d,
        }.__getitem__(self.dim)
        LazyConv = {
            1: torch.nn.LazyConv1d,
            2: torch.nn.LazyConv2d,
            3: torch.nn.LazyConv3d,
        }.__getitem__(self.dim)

        layers = []
        for c_in, c_out, activation in zip(
            net_shape[:-1], net_shape[1:], activations
        ):
            conv = (
                LazyConv(c_out, self.kernel_size, **self.conv_kwargs)
                if c_in is None
                else Conv(c_in, c_out, self.kernel_size, **self.conv_kwargs)
            )
            layers.append(conv)
            layers.append(activation)

        return torch.nn.Sequential(*layers)


@dataclasses.dataclass
class ConvNetCircular(ConvNet):
    """
    Convolutional neural network with circular padding.
    """

    def __post_init__(self) -> None:
        self.conv_kwargs.update(
            padding=self.kernel_size // 2,
            padding_mode="circular",
        )


class GenericNetBuilder:
    """
    General-purpose network builder.

    .. warning:: Untested! Work in progress.
    """

    def __init__(self, spec: list[dict]) -> None:
        self.spec = spec

    def __call__(self) -> torch.nn.Sequential:
        # TODO: augment namespace with custom modules to allow user
        # to request custom layers?

        layers = []
        for layer_spec in self.spec:

            cls = getattr(torch.nn, layer_spec["class"])
            args = layer_spec["args"]
            layer = cls(**args)
            layers.append(layer)

        return torch.nn.Sequential(*layers)
