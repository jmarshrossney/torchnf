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

from jsonargparse.typing import PositiveInt
import torch


class NetBuilder:
    """
    Base class for neural network builders. Simply defines a __call__ method.
    """

    # NOTE: Should i have a @property for network_preserves_structure for
    # input/output? Would help decide which conditioner to use automatically

    def __call__(self) -> torch.nn.Sequential:
        raise NotImplementedError


@dataclasses.dataclass
class DenseNet(NetBuilder):
    """
    Fully-connected feed-forward neural network.

    See :py:class:`torch.nn.Linear`.
    """

    in_features: PositiveInt
    out_features: PositiveInt
    hidden_shape: list[PositiveInt]
    activation: str
    activation_kwargs: dict = dataclasses.field(default_factory=dict)
    skip_final_activation: bool = False
    linear_kwargs: dict = dataclasses.field(default_factory=dict)

    def _activation(self) -> torch.nn.Module:
        return getattr(torch.nn, self.activation)(**self.activation_kwargs)

    def __call__(self) -> torch.nn.Sequential:
        # Maybe: (allows other recipes to set values, e.g. out_features)
        # def __call__(self, replacements: dict):
        #     self.__dict__.update(replacements)
        # -- or, for frozen instance --
        #     config = dataclasses.asdict(self).update(replacements)
        net_shape = [self.in_features, *self.hidden_shape, self.out_features]
        activations = [self._activation() for _ in self.hidden_shape] + [
            torch.nn.Identity()
            if self.skip_final_activation
            else self._activation()
        ]
        layers = []
        for f_in, f_out, activation in zip(
            net_shape[:-1], net_shape[1:], activations
        ):
            layers.append(torch.nn.Linear(f_in, f_out, **self.linear_kwargs))
            layers.append(activation)

        return torch.nn.Sequential(*layers)


@dataclasses.dataclass
class ConvNet(NetBuilder):
    """
    Convolutional neural network.

    See :py:class:`torch.nn.Conv2d`, and 1d/3d versions.
    """

    dim: PositiveInt
    in_channels: PositiveInt
    out_channels: PositiveInt
    hidden_shape: list[PositiveInt]
    activation: str
    kernel_size: PositiveInt
    activation_kwargs: dict = dataclasses.field(default_factory=dict)
    skip_final_activation: bool = False
    conv_kwargs: dict = dataclasses.field(default_factory=dict)

    def _activation(self) -> torch.nn.Module:
        return getattr(torch.nn, self.activation)(**self.activation_kwargs)

    def __call__(self) -> torch.nn.Sequential:
        net_shape = [self.in_channels, *self.hidden_shape, self.out_channels]
        activations = [self._activation() for _ in self.hidden_shape] + [
            torch.nn.Identity()
            if self.skip_final_activation
            else self._activation()
        ]
        conv = {
            1: torch.nn.Conv1d,
            2: torch.nn.Conv2d,
            3: torch.nn.Conv3d,
        }.__getitem__(self.dim)
        layers = []
        for c_in, c_out, activation in zip(
            net_shape[:-1], net_shape[1:], activations
        ):
            layers.append(
                conv(c_in, c_out, self.kernel_size, **self.conv_kwargs)
            )
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


class GenericNetBuilder(NetBuilder):
    """
    General-purpose network builder.

    Work in progress.
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
