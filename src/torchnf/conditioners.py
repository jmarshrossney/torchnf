"""
The :code:`forward` method should return a set of parameters upon which the
transformer should be conditioned.
"""
from typing import Callable, Optional, Union
import torch

import torchnf.utils


class Conditioner(torch.nn.Module):
    """
    Alias of :py:class:`torch.nn.Module`, to be inherited by all conditioners.
    """

    pass


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
        params = torchnf.utils.expand_elements(
            self.params,
            data_shape,
            stack_dim=0,
        )
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
        mask: Optional[torch.BoolTensor] = None,
        net: Optional[torch.nn.Sequential] = None,
    ) -> None:
        super().__init__()
        if mask:
            self.register_buffer("_mask", mask)
        if net:
            self._net = net

    def get_mask(self, x: torch.Tensor) -> torch.BoolTensor:
        """
        Returns the mask that multiplies the input tensor.
        """
        return self._mask

    def _forward(self, x_masked: torch.Tensor) -> torch.Tensor:
        """
        Computes and returns the parameters.

        Unless overridden, this simply called the :code:`forward`
        method of the :code:`net` argument to the constructor.
        """
        return self._net(x_masked)

    def forward(self, x: torch.Tensor, context: dict = {}) -> torch.Tensor:
        """
        Returns the parameters, after masking.

        Masked elements are NaN's.

        Effectively this does the following:

        .. code-block:: python
          :linenos:

            self.context = context
            mask = self.get_mask(x)
            x = x.mul(mask)
            params = self._forward(x)
            params = params.mul(~mask)  # ~ denotes logical negation
            params = params.add(
                torch.full_like(params, float('nan')).mul(mask)
            )
            return params
        """
        self.context = context
        mask = self.get_mask(x)
        params = self._forward(x.mul(mask))
        return params.mul(~mask).add(
            torch.full_like(params, fill_value=float("nan")).mul(mask)
        )


class AutoregressiveConditioner(Conditioner):
    """
    TODO

    Results in a triangular determinant.
    """

    pass
