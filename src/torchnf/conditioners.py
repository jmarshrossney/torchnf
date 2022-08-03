"""
The :code:`forward` method should return a set of parameters upon which the
transformer should be conditioned.
"""
from collections.abc import Iterable
from functools import partial
from typing import Callable, Union

from jsonargparse.typing import PositiveInt
import torch
from torch.nn.modules.lazy import LazyModuleMixin
from torch.nn.parameter import UninitializedParameter, UninitializedBuffer

from torchnf.abc import Transformer
import torchnf.utils.tensor
from torchnf.utils.nn import Activation, make_fnn

__all__ = [
    "SimpleConditioner",
    "LazySimpleConditioner",
    "MaskedConditioner",
    "LazyMaskedConditioner",
    # "AutoregressiveConditioner",
]


class SimpleConditioner(torch.nn.Module):
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
            params = torchnf.utils.tensor.expand_elements(
                self.params,
                data_shape,
                stack_dim=0,
            )
        else:
            params = self.params
        return params.expand([batch_size, -1, *data_shape])


class LazySimpleConditioner(LazyModuleMixin, SimpleConditioner):
    """
    A lazily initialised version of SimpleConditioner.

    The ``identity_params`` attribute of the transformer are used to
    initialise the parameters of this conditioner.
    """

    cls_to_become = SimpleConditioner

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


class MaskedConditioner(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()

    def get_mask(
        self, x: torch.Tensor, context: dict = {}
    ) -> torch.BoolTensor:
        """
        Returns the mask that delineates the partitions of the coupling layer.
        """
        raise NotImplementedError

    @staticmethod
    def index_input_with_mask(self, input):
        """
        Use the mask to index the input tensor.
        """
        x, *ctx = input
        mask = self.get_mask(*input)
        return (x[:, mask], *ctx)

    @staticmethod
    def mul_input_by_mask(self, input):
        """
        Multiply the input tensor by the binary mask.
        """
        x, *ctx = input
        mask = self.get_mask(*input)
        return (x.mul(mask), *ctx)

    @staticmethod
    def set_output_to_nan_where_mask(self, input, output):
        mask = self.get_mask(*input)
        params = output
        # TODO: check if params[mask] = float("nan") is faster
        return params.masked_fill(mask, float("nan"))

    @staticmethod
    def scatter_output_into_nantensor(self, input, output):
        mask = self.get_mask(*input)
        params = output
        assert params.dim() == 2
        params_shape = torch.Size(
            [
                params.shape[0],
                params[0].numel() // int(mask.logical_not().sum()),
                *mask.shape,
            ]
        )
        # TODO: check if tensor indexing is faster
        return torch.full(params_shape, float("nan")).masked_scatter(
            mask.logical_not(), params
        )

    def forward(self, x: torch.Tensor, context: dict = {}) -> torch.Tensor:
        raise NotImplementedError


class LazyMaskedConditioner(LazyModuleMixin, MaskedConditioner):
    cls_to_become = MaskedConditioner

    def __init__(self) -> None:
        super().__init__()
        self.mask = UninitializedBuffer()

    def get_mask(
        self, x: torch.Tensor, context: dict = {}
    ) -> torch.BoolTensor:
        return self.mask

    def initialize_parameters(
        self, inputs: torch.Tensor, context: dict = {}
    ) -> None:
        raise NotImplementedError


class MaskedConditionerFNN(MaskedConditioner):
    def __init__(
        self,
        mask: torch.BoolTensor,
        hidden_shape: list[PositiveInt],
        activation: Activation,
        final_activation: Activation = torch.nn.Identity(),
        bias: bool = True,
    ) -> None:
        super().__init__()

        self.register_buffer("mask", mask)
        self.register_forward_pre_hook(self.index_input_with_mask)
        self.register_forward_hook(self.scatter_output_into_nantensor)

        # make net
        net = make_fnn(
            in_features=int(mask.sum()),
            out_features=int(mask.logical_not().sum()),
            hidden_shape=hidden_shape,
            activation=activation,
            final_activation=final_activation,
            bias=bias,
        )
        self.register_module("net", net)

    def get_mask(
        self, x: torch.Tensor, context: dict = {}
    ) -> torch.BoolTensor:
        return self.mask

    def forward(self, x: torch.Tensor, context: dict = {}) -> torch.Tensor:
        return self.net(x)


class LazyMaskedConditionerFNN(LazyModuleMixin, MaskedConditioner):
    cls_to_become = MaskedConditioner

    def __init__(
        self,
        mask_fn: Callable[Iterable[PositiveInt], torch.BoolTensor],
        hidden_shape: list[PositiveInt],
        activation: Activation,
        final_activation: Activation = torch.nn.Identity(),
        bias: bool = True,
    ) -> None:
        super().__init__()
        self.mask = UninitializedBuffer()
        self.net = UninitializedParameter()

        self._mask_fn = mask_fn
        self._net_fn = partial(
            make_fnn(
                hidden_shape=hidden_shape,
                activation=activation,
                final_activation=final_activation,
                bias=bias,
            )
        )

    def initialize_parameters(
        self, inputs: torch.Tensor, context: dict = {}
    ) -> None:
        data_shape = inputs.shape[1:]
        mask = self._mask_fn(data_shape)
        net = self._net_fn(
            in_features=int(mask.sum()),
            out_features=int(mask.logical_not().sum()),
        )
        self.mask = mask
        self.net = net

        del self._mask_fn
        del self._net_fn

    def forward(self, x: torch.Tensor, context: dict = {}) -> torch.Tensor:
        return self.net(x)


"""
class MaskedConditionerCNN(MaskedConditioner):
    def __init__(
        self,
        mask: torch.BoolTensor,
        dim: ConvDim,
        hidden_shape: list[PositiveInt],
        activation: Union[ACTIVATIONS],
        kernel_size: PositiveInt,
        skip_final_activation: bool = False,
        circular: bool = False,
        conv_kwargs: dict = {},
    ) -> None:
        super().__init__(mask)

        self.register_forward_pre_hook(self.mul_input_by_mask)
        self.register_forward_hook(self.set_output_to_nan_where_mask)

        net = make_cnn(...)
"""


class AutoregressiveConditioner(torch.nn.Module):
    """
    TODO

    Results in a triangular determinant.
    """

    pass
