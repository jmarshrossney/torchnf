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
from typing import Optional, Union

from jsonargparse.typing import PositiveInt, restricted_number_type
import torch

# TODO: 'block builder' where user defines a custom block, e.g.
# Linear -> ReLU -> BatchNorm, that has a __call__(size_in, size_out)

__all__ = [
    "DenseNetBuilder",
    "ConvNetBuilder",
    "CircularConvNetBuilder",
    "GenericNetBuilder",
]

ConvDim = restricted_number_type(
    name="ConvDim",
    base_type=int,
    restrictions=[(">", 0), ("<", 1)],
    join="and",
    docstring="Convolution dimension: {1, 2, 3} are supported.",
)

# So the help isn't flooded with all available nn.Modules
# NOTE: torch master branch has __all__ attribute. When that is merged replace
# explicit list with torch.nn.modules.activations.__all__
_ACTIVATIONS = tuple(
    getattr(torch.nn, a)
    for a in [
        "ELU",
        "Hardshrink",
        "Hardsigmoid",
        "Hardtanh",
        "Hardswish",
        "LeakyReLU",
        "LogSigmoid",
        "PReLU",
        "ReLU",
        "ReLU6",
        "RReLU",
        "SELU",
        "CELU",
        "GELU",
        "Sigmoid",
        "SiLU",
        "Mish",
        "Softplus",
        "Softshrink",
        "Softsign",
        "Tanh",
        "Tanhshrink",
        "Threshold",
        "GLU",
    ]
)
ACTIVATIONS = Union[_ACTIVATIONS]


def raise_(exc: Exception):
    raise Exception


@dataclasses.dataclass
class DenseNetBuilder:
    """
    Fully-connected feed-forward neural network.

    See :py:class:`torch.nn.Linear`.

    .. note:: ``in_features`` and ``out_features`` are specified in
    ``__call__`` rather than ``__init__``.
    """

    hidden_shape: list[PositiveInt]
    activation: Union[str, ACTIVATIONS]
    skip_final_activation: bool = False
    bias: bool = True

    def __post_init__(self) -> None:
        if isinstance(self.activation, str):
            # Instantiate with no arguments
            self.activation = getattr(torch.nn, self.activation)()

    def __call__(
        self,
        in_features: Optional[PositiveInt],  # can pass None, but explicit
        out_features: PositiveInt,
    ) -> torch.nn.Sequential:
        in_features = in_features or self.in_features
        out_features = (
            out_features
            or self.out_features
            or raise_(ValueError("'out_features' must be provided"))
        )

        net_shape = [in_features, *self.hidden_shape, out_features]
        activations = [self.activation for _ in self.hidden_shape] + [
            torch.nn.Identity()
            if self.skip_final_activation
            else self.activation
        ]
        layers = []
        for f_in, f_out, activation in zip(
            net_shape[:-1], net_shape[1:], activations
        ):
            linear = (
                torch.nn.LazyLinear(f_out, bias=self.bias)
                if f_in is None
                else torch.nn.Linear(f_in, f_out, bias=self.bias)
            )
            layers.append(linear)
            layers.append(activation)

        return torch.nn.Sequential(*layers)


@dataclasses.dataclass
class ConvNetBuilder:
    """
    Convolutional neural network.

    See :py:class:`torch.nn.Conv2d`, and 1d/3d versions.

    .. note::

        ``in_channels`` and ``out_channels`` are specified in
        ``__call__`` rather than ``__init__``
    """

    dim: ConvDim
    hidden_shape: list[PositiveInt]
    activation: Union[str, ACTIVATIONS]
    kernel_size: PositiveInt
    skip_final_activation: bool = False
    conv_kwargs: dict = dataclasses.field(default_factory=dict)

    def __post_init__(self) -> None:
        if isinstance(self.activation, str):
            # Instantiate with no arguments
            self.activation = getattr(torch.nn, self.activation)()

    def __call__(
        self,
        in_channels: Optional[PositiveInt],
        out_channels: PositiveInt,
    ) -> torch.nn.Sequential:

        in_channels = in_channels or self.in_channels
        out_channels = (
            out_channels
            or self.out_channels
            or raise_(ValueError("'out_channels' must be provided"))
        )

        net_shape = [in_channels, *self.hidden_shape, out_channels]
        activations = [self.activation for _ in self.hidden_shape] + [
            torch.nn.Identity()
            if self.skip_final_activation
            else self.activation
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
class CircularConvNetBuilder(ConvNetBuilder):
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
