from collections import OrderedDict
from typing import Union

import torch

from torchnf.abc import Conditioner, Transformer, DensityTransform
import torchnf.utils.flow

__all__ = [
    "FlowLayer",
    "Composition",
    "Flow",
]


class FlowLayer(DensityTransform):
    """
    Class representing a Normalizing Flow layer.

    This is a subclass of :py:class:`torch.nn.Module` with a specific
    modular structure that provides the flexibility to construct a variety
    of different types of conditional density transformations. Such
    transformations can be chained to make a Normalizing Flow.

    Args:
        transformer
            A class which implements the forward and inverse transformations,
            given an input tensor and a tensor of parameters which the
            transformation is conditioned on
        conditioner
            A class whose :code:`forward` method takes the inputs and returns
            a tensor of parameters for the transformer
    """

    def __init__(
        self,
        transformer: Transformer,
        conditioner: Conditioner,
    ) -> None:
        super().__init__()
        self.transformer = transformer
        self.conditioner = conditioner

    @classmethod
    def inverted(
        cls,
        transformer: Transformer,
        conditioner: Conditioner,
    ) -> "FlowLayer":
        return torchnf.utils.flow.inverted(cls)(transformer, conditioner)

    def conditioner_forward(self, xy: torch.Tensor) -> torch.Tensor:
        """Forward pass of the conditioner.

        Unless overriden, this method simply returns
        :code:`self.conditioner(xy, self.context)`.
        """
        return self.conditioner(xy, self.context)

    def transformer_forward(
        self,
        x: torch.Tensor,
        params: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass of the transformer.

        Unless overriden, this method simply returns
        :code:`self.transfomer(x, params, self.context)`.
        """
        return self.transformer(x, params, self.context)

    def transformer_inverse(
        self,
        y: torch.Tensor,
        params: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Inverse pass of the transformer.

        Unless overriden, this method simply returns
        :code:`self.transfomer.inverse(y, params, self.context)`.
        """
        return self.transformer.inverse(y, params, self.context)

    def forward(
        self, x: torch.Tensor, ldj: torch.Tensor, context: dict = {}
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass of the density transformation.

        Unless overridden, this method does the following:

        .. code-block:: python

            self.context = context
            params = self.conditioner_forward(x)
            y, ldj_this = self.transformer_forward(x, params)
            ldj += ldj_this
            return y, ldj
        """
        self.context = context
        params = self.conditioner_forward(x)
        y, ldj_this = self.transformer_forward(x, params)
        ldj.add_(ldj_this)
        return y, ldj

    def inverse(
        self, y: torch.Tensor, ldj: torch.Tensor, context: dict = {}
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        The inverse of :meth:`forward`.

        Unless overridden, this method does the following:

        .. code-block:: python

            self.context = context
            params = self.conditioner_forward(y)
            x, ldj_this = self.transformer_inverse(y, params)
            ldj += ldj_this
            return x, ldj
        """
        self.context = context
        params = self.conditioner_forward(y)
        x, ldj_this = self.transformer_inverse(y, params)
        ldj.add_(ldj_this)
        return x, ldj

    def freeze(self) -> None:
        """
        Detaches all parameters from the autograd graph.
        """
        for param in self.parameters():
            param.requires_grad = False

    def unfreeze(self) -> None:
        """
        Reattaches all parameters from the autograd graph.
        """
        for param in self.parameters():
            param.requires_grad = True


class Composition(torch.nn.Sequential):
    """
    Composes density transformations.
    """

    def __init__(
        self,
        *transforms: Union[
            DensityTransform, OrderedDict[str, DensityTransform]
        ],
    ) -> None:
        super().__init__(*transforms)

    @classmethod
    def inverted(
        cls,
        *transforms: Union[
            DensityTransform, OrderedDict[str, DensityTransform]
        ],
    ) -> "Composition":
        return torchnf.utils.flow.inverted(cls)(*transforms)

    def forward(
        self,
        x: torch.Tensor,
        ldj: torch.Tensor,
        context: dict = {},
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass of the density transformation.
        """
        y = x
        for transform in self:
            y, ldj = transform(y, ldj, context)
        return y, ldj

    def inverse(
        self, y: torch.Tensor, ldj: torch.Tensor, context: dict = {}
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Inverse of :meth:`forward`.
        """
        x = y
        for transform in reversed(self):
            x, ldj = transform.inverse(x, ldj, context)
        return x, ldj


class Flow(Composition):
    """
    Composes density transformations, initalising the ldj with zeros.
    """

    def forward(
        self, x: torch.Tensor, context: dict = {}
    ) -> tuple[torch.Tensor, torch.Tensor]:
        ldj = torch.zeros(x.shape[0], device=x.device)
        return super().forward(x, ldj, context)

    def inverse(
        self, y: torch.Tensor, context: dict = {}
    ) -> tuple[torch.Tensor, torch.Tensor]:
        ldj = torch.zeros(y.shape[0], device=y.device)
        return super().inverse(y, ldj, context)
