import dataclasses
import torch


def simple_net_builder(net_spec: list[dict]) -> torch.nn.Sequential:

    # TODO: augment namespace with custom modules to allow user
    # to request custom layers?

    layers = []
    for layer_spec in net_spec:

        cls = getattr(torch.nn, layer_spec["class"])
        args = layer_spec["args"]
        layer = cls(**args)
        layers.append(layer)

    return torch.nn.Sequential(*layers)


@dataclasses.dataclass
class DenseNet:
    in_features: int
    out_features: int
    hidden_shape: list[int]
    activation: str
    activation_kwargs: dict = dataclasses.field(default_factory=dict)
    skip_final_activation: bool = False
    linear_kwargs: dict = dataclasses.field(default_factory=dict)

    def _activation(self) -> torch.nn.Module:
        return getattr(torch.nn, self.activation)(**self.activation_kwargs)

    def __call__(self) -> torch.nn.Sequential:
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
class ConvNet:
    dim: int
    in_channels: int
    out_channels: int
    hidden_shape: list[int]
    activation: str
    kernel_size: int
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
    def __post_init__(self) -> None:
        self.conv_kwargs.update(
            padding=self.kernel_size // 2,
            padding_mode="circular",
        )
