"""
The :code:`forward` method should return a set of parameters upon which the
transformer should be conditioned.
"""
from typing import Optional, Union
import torch

import torchnf.utils


class Conditioner(torch.nn.Module):
    """
    Alias of :py:class:`torch.nn.Module`, to be inherited by all conditioners.
    """

    def forward(
        self, inputs: torch.Tensor, context: dict = {}
    ) -> torch.Tensor:
        """
        Returns a set of parameters, optionally conditioned on the inputs.
        """
        raise NotImplementedError


class SimpleConditioner(Conditioner):
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
        init_params: Union[torch.Tensor, list[float]],
    ) -> None:
        super().__init__()
        self.params = torch.nn.Parameter(torch.Tensor(init_params))

    def forward(
        self, inputs: torch.Tensor, context: dict = {}
    ) -> torch.Tensor:
        """
        Returns the parameters, expanded to the correct shape.
        """
        batch_size, *data_shape = inputs.shape
        if self.params.dim() == 1:
            params = torchnf.utils.expand_elements(
                self.params,
                data_shape,
                stack_dim=0,
            )
        else:
            params = self.params
        return params.expand([batch_size, -1, *data_shape])


class MaskedConditioner(Conditioner):
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

    Derived classes may override :meth:`_forward` and :meth:`get_mask`.

    TODO
        Explain how to get a mask which depends on the data shape.
        Consider allowing functionality to mask x with values that aren't
        zero, e.g. noise drawn from a known distribution...

    """

    def __init__(
        self,
        net: Optional[torch.nn.Sequential] = None,
        mask: Optional[torch.BoolTensor] = None,
        create_channel_dim: bool = False,
    ) -> None:
        super().__init__()
        if mask is not None:
            self.register_buffer("_mask", mask)
        if net is not None:
            self._net = net
        self.create_channel_dim = create_channel_dim

    @property
    def mask(self) -> torch.BoolTensor:
        """
        Returns the mask that delineates the partitions of the coupling layer.
        """
        # NOTE: for variable shaped inputs, use self.context to get shape
        return self._mask

    def apply_mask_to_input(self, x: torch.Tensor) -> torch.Tensor:
        """
        Applies mask to the input tensor.
        """
        return x[:, self.mask]

    def apply_mask_to_output(self, params: torch.Tensor) -> torch.Tensor:
        """
        Applies mask to the output parameters.
        """
        params_shape = torch.Size(
            [
                params.shape[0],
                params[0].numel() // int(self.mask.logical_not().sum()),
                *self.mask.shape,
            ]
        )
        return torch.full(params_shape, float("nan")).masked_scatter(
            ~self.mask, params
        )

    def _forward(self, x_masked: torch.Tensor) -> torch.Tensor:
        """
        Computes and returns the parameters.

        Unless overridden, this simply called the :code:`forward`
        method of the :code:`net` argument to the constructor.
        """
        if self.create_channel_dim:
            x_masked.unsqueeze_(1)
        return self._net(x_masked)

    def forward(self, x: torch.Tensor, context: dict = {}) -> torch.Tensor:
        """
        Returns the parameters, after masking.

        Masked elements are NaN's.

        This does the following:

        .. code-block:: python
          :linenos:

            self.context = context
            x_masked = self.apply_mask_to_inputs(x)
            params = self._forward(x_masked)
            params = self.apply_mask_to_outputs(params)
            return params
        """
        self.context = context
        return self.apply_mask_to_output(
            self._forward(self.apply_mask_to_input(x))
        )


class MaskedConditionerStructurePreserving(MaskedConditioner):
    def apply_mask_to_input(self, x: torch.Tensor) -> torch.Tensor:
        """
        Applies mask to the input tensor.
        """
        return x.mul(self.mask)

    def apply_mask_to_output(self, params: torch.Tensor) -> torch.Tensor:
        """
        Applies mask to the output parameters.
        """
        return params.masked_fill(self.mask, float("nan"))


class AutoregressiveConditioner(Conditioner):
    """
    TODO

    Results in a triangular determinant.
    """

    pass
