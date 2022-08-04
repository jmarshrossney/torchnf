"""
The :code:`forward` method should return a set of parameters upon which the
transformer should be conditioned.
"""
from collections.abc import Iterable
from typing import Optional, Union

from jsonargparse.typing import PositiveInt
import torch
from torch.nn.modules.lazy import LazyModuleMixin
from torch.nn.parameter import UninitializedParameter

from torchnf.abc import Transformer
from torchnf.utils.tensor import (
    expand_elements,
    set_to_nan_where_mask,
    scatter_into_nantensor,
)
from torchnf.utils.nn import Activation, make_fnn, make_cnn

__all__ = [
    "TrainableParameters",
    "MaskedConditioner",
    "simple_fnn_conditioner",
    "simple_cnn_conditioner",
]


class _TrainableParameters(torch.nn.Module):
    """
    Conditioner in which the parameters are independent of the input.

    The parameters are registered as a :py:class:`torch.nn.Parameter`,
    which makes them learnable parameters of the model. The
    :meth:`forward` method simply returns these parameters.

    Args:
        init_params
            Initial values for the parameters
    """

    transformer: Transformer

    def __init__(
        self,
        init_params: Union[torch.Tensor, Iterable[float]],
    ) -> None:
        super().__init__()

        if not isinstance(init_params, torch.Tensor):
            init_params = torch.tensor([*init_params])
        self.params = torch.nn.Parameter(init_params.float())

    def forward(
        self, inputs: torch.Tensor, context: dict = {}
    ) -> torch.Tensor:
        """
        Returns the parameters, expanded to the correct shape.
        """
        batch_size, *data_shape = inputs.shape
        if self.params.dim() == 1:
            params = expand_elements(
                self.params,
                data_shape,
                stack_dim=0,
            )
        else:
            params = self.params
        return params.expand([batch_size, -1, *data_shape])


class _LazyTrainableParameters(LazyModuleMixin, _TrainableParameters):
    """
    A lazily initialised version of TrainableParameters.

    The ``identity_params`` attribute of the transformer are used to
    initialise the parameters of this conditioner.
    """

    cls_to_become = _TrainableParameters

    def __init__(self) -> None:
        super().__init__([])
        self.params = UninitializedParameter()

    def initialize_parameters(
        self, inputs: torch.Tensor, context: dict = {}
    ) -> None:
        try:
            init_params = self.transformer.identity_params
        except AttributeError as exc:
            raise AttributeError(
                "Unable to initialize LazySimpleConditioner"
            ) from exc

        assert isinstance(init_params, torch.Tensor)

        self.params = torch.nn.Parameter(init_params.float())


def TrainableParameters(
    init_params: Optional[torch.Tensor] = None,
) -> Union[_TrainableParameters, _LazyTrainableParameters]:
    if init_params is None:
        return _LazyTrainableParameters()
    else:
        return _TrainableParameters(init_params)


class MaskedConditioner(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.register_forward_pre_hook(self._forward_pre_hook)
        self.register_forward_hook(self._forward_post_hook)

    def get_mask(
        self, x: torch.Tensor, context: dict = {}
    ) -> torch.BoolTensor:
        """
        Returns the mask that delineates the partitions of the coupling layer.
        """
        raise NotImplementedError

    @staticmethod
    def _forward_pre_hook(
        self, inputs: tuple[torch.Tensor, dict]
    ) -> torch.Tensor:
        x, _ = inputs
        mask = self.get_mask(*inputs)
        return self.apply_mask_to_input(x, mask)

    @staticmethod
    def _forward_post_hook(
        self, inputs: tuple[torch.Tensor, dict], output: torch.Tensor
    ) -> torch.Tensor:
        mask = self.get_mask(*inputs)
        return self.apply_mask_to_output(output, mask)

    @staticmethod
    def apply_mask_to_input(
        x: torch.Tensor, mask: torch.BoolTensor
    ) -> torch.Tensor:
        raise NotImplementedError

    @staticmethod
    def apply_mask_to_output(
        params: torch.Tensor, mask: torch.BoolTensor
    ) -> torch.Tensor:
        raise NotImplementedError

    def forward(self, x: torch.Tensor, context: dict = {}) -> torch.Tensor:
        raise NotImplementedError


def simple_fnn_conditioner(
    transformer: Transformer,
    mask: torch.BoolTensor,
    hidden_shape: list[PositiveInt],
    activation: Activation,
    final_activation: Activation = torch.nn.Identity(),
    bias: bool = True,
) -> torch.nn.Sequential:

    net = make_fnn(
        in_features=int(mask.sum()),
        out_features=int(mask.logical_not().sum()) * transformer.n_params,
        hidden_shape=hidden_shape,
        activation=activation,
        final_activation=final_activation,
        bias=bias,
    )

    def apply_mask_to_input(self, inputs):
        return inputs[0][:, self.mask]

    def apply_mask_to_output(self, inputs, output):
        return scatter_into_nantensor(output, self.mask)

    net.register_buffer("mask", mask)
    net.register_forward_pre_hook(apply_mask_to_input)
    net.register_forward_hook(apply_mask_to_output)

    return net


def simple_cnn_conditioner(
    transformer: Transformer,
    mask: torch.BoolTensor,
    hidden_shape: list[PositiveInt],
    kernel_size: PositiveInt,
    activation: Activation,
    final_activation: Activation = torch.nn.Identity(),
    circular: bool = False,
    conv_kwargs: dict = {},
) -> torch.nn.Sequential:

    net = make_cnn(
        dim=mask.dim(),
        in_channels=1,
        out_channels=transformer.n_params,
        hidden_shape=hidden_shape,
        kernel_size=kernel_size,
        activation=activation,
        final_activation=final_activation,
        circular=circular,
        conv_kwargs=conv_kwargs,
    )

    def apply_mask_to_input(self, inputs):
        return inputs[0].mul(self.mask).unsqueeze(dim=1)

    def apply_mask_to_output(self, inputs, output):
        return set_to_nan_where_mask(output, self.mask)

    net.register_buffer("mask", mask)
    net.register_forward_pre_hook(apply_mask_to_input)
    net.register_forward_hook(apply_mask_to_output)
