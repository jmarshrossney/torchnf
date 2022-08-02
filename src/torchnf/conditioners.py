"""
The :code:`forward` method should return a set of parameters upon which the
transformer should be conditioned.
"""
from collections.abc import Iterable
from typing import Callable, Optional, Union
import torch

import torchnf.utils.tensor

__all__ = [
    "SimpleConditioner",
    "MaskedConditioner",
    "AutoregressiveConditioner",
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


class MaskedConditioner(torch.nn.Module):
    r"""
    Masked conditioner.

    In this conditioner, the inputs :math:`x` are first masked so that
    only a subset can influence the resulting parameters. Furthermore,
    the resulting parameters :math:`\{\lambda\}` are masked using the
    logical negation of the original mask.

    When a point-wise transformation of :math:`x` is conditioned on
    :math:`\{\lambda\}`, the Jacobian is triangular.

    A Normalizing Flow layer where the transformer is conditioned on
    these parameters is called a Coupling Layer. The archetypal example
    is Real NVP (:arxiv:`1605.08803`).

    Derived classes may override :meth:`forward_` and :attr:`mask`.

    Args:
        net:
            Neural network that takes the masked input tensor and returns
            a set of parameters
        mask:
            Boolean mask where the 'False' elements are the masked ones,
            i.e. they will not be passed to ``forward_``
        create_channel_dim:
            If True, an extra dimension of size 1 will be added to the
            input tensor before passing it to ``forward_``
        mask_mode:
            TODO: document
    """

    def __init__(
        self,
        net: Optional[torch.nn.Sequential] = None,
        mask: Optional[torch.BoolTensor] = None,
        mask_mode: Union[
            str, Callable[[torch.Tensor, torch.BoolTensor, dict], torch.Tensor]
        ] = "auto",
        create_channel_dim: bool = False,
    ) -> None:
        super().__init__()
        if net is not None:
            self.net = net
        if mask is not None:
            self.register_buffer("mask", mask)

        if mask_mode == "index":
            self._apply_mask_to_input = self.index_with_mask
        elif mask_mode == "mul":
            self._apply_mask_to_input = self.mul_by_mask
        elif mask_mode == "auto":
            self._apply_mask_to_input = self._choose_mask_fn()
        elif callable(mask_mode):
            self._apply_mask_to_input = mask_mode
        else:
            raise ValueError(
                f"Expected 'mask_mode' to be one of ('auto', 'index', 'mul', Callable) but got '{mask_mode}'"  # noqa: E501
            )

        self.create_channel_dim = create_channel_dim

    def _choose_mask_fn(self) -> None:
        """
        Auto-decide how to apply mask based on type of layers in network.
        """
        assert hasattr(
            self, "net"
        ), "Cannot auto-choose mask mode: require `net` attribute"
        Conv = torch.nn.modules.conv._ConvNd
        Linear = torch.nn.modules.linear.Linear
        # Look for first layer that is either a Conv or Linear
        for layer in self.net:
            if isinstance(layer, Conv):
                return self.mul_by_mask
            elif isinstance(layer, Linear):
                return self.index_with_mask

        raise Exception(
            "Net has no Linear or Convolutional layers. Unable to auto-decide mask func"  # noqa: E501
        )

    def get_mask(self) -> torch.BoolTensor:
        """
        Returns the mask that delineates the partitions of the coupling layer.
        """
        # NOTE: for variable shaped inputs, use self.context to get shape
        return self.mask

    @staticmethod
    def index_with_mask(x: torch.Tensor, mask: torch.Tensor, context: dict):
        """
        Use the mask to index the input tensor.
        """
        return x[:, mask]

    @staticmethod
    def mul_by_mask(x: torch.Tensor, mask: torch.Tensor, context: dict):
        """
        Multiply the input tensor by the binary mask
        """
        return x.mul(mask)

    def apply_mask_to_input(self, x: torch.Tensor) -> torch.Tensor:
        """
        Applies mask to the input tensor.
        """
        return self._apply_mask_to_input(x, self.get_mask(), self.context)

    def apply_mask_to_output(self, params: torch.Tensor) -> torch.Tensor:
        r"""
        Applies mask to the output parameters.

        The output parameters must be either (a) a tensor of dimension >2
        where the dimensions are ``(batch_size, n_params, *data_shape)``,
        or (b) a tensor of dimension 2, where the dimensions are
        ``(batch_size, n_params * n_masked_elements)``.

        Returns:
            A tensor with dimensions ``(batch_size, n_params, *data_shape)``,
            where the elements corresponding to input data that was *not*
            masked are ``NaN``
        """
        mask = self.get_mask()

        # If output has dims (n_batch, n_params, ...)
        if params.dim() > 2:
            # TODO: check if params[mask] = float("nan") is faster
            return params.masked_fill(mask, float("nan"))

        # Otherwise, assume flattened data dimension containing the
        # correct number of parameters (corresponding to masked elements)
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

    def forward_(self, x_masked: torch.Tensor) -> torch.Tensor:
        """
        Computes and returns the parameters.

        Unless overridden, this simply called the :code:`forward`
        method of ``self.net``.
        """
        return self.net(x_masked)

    def forward(self, x: torch.Tensor, context: dict = {}) -> torch.Tensor:
        """
        Returns the parameters, after masking.

        Masked elements are NaN's.

        This does the following:

        .. code-block:: python
          :linenos:

            self.context = context
            x_masked = self.apply_mask_to_input(x)
            if self.create_channel_dim:
                x_masked.unsqueeze_(1)
            params = self._forward(x_masked)
            params = self.apply_mask_to_output(params)
            return params
        """
        self.context = context
        x_masked = self.apply_mask_to_input(x)
        if self.create_channel_dim:
            x_masked.unsqueeze_(1)
        params = self.forward_(x_masked)
        params = self.apply_mask_to_output(params)
        return params


class AutoregressiveConditioner(torch.nn.Module):
    """
    TODO

    Results in a triangular determinant.
    """

    pass
