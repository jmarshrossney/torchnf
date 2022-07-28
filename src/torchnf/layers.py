import torch

from torchnf.abc import Conditioner, Transformer, DensityTransform
import torchnf.utils.flow

__all__ = [
    "FlowLayer",
    "Composition",
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
        self, x: torch.Tensor, context: dict = {}
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass of the density transformation.

        Unless overridden, this method does the following:

        .. code-block:: python

            self.context = context
            params = self.conditioner_forward(x)
            y, ldj = self.transformer_forward(x, params)
            return y, ldj
        """
        self.context = context
        params = self.conditioner_forward(x)
        y, ldj = self.transformer_forward(x, params)
        return y, ldj

    def inverse(
        self, y: torch.Tensor, context: dict = {}
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        The inverse of :meth:`forward`.

        Unless overridden, this method does the following:

        .. code-block:: python

            self.context = context
            params = self.conditioner_forward(y)
            x, ldj = self.transformer_inverse(y, params)
            return x, ldj
        """
        self.context = context
        params = self.conditioner_forward(y)
        x, ldj = self.transformer_inverse(y, params)
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
    Composes density transformations and aggregates log det Jacobians.
    """

    def __init__(self, *transforms: DensityTransform) -> None:
        super().__init__(*transforms)

    @classmethod
    def inverted(
        cls,
        *transforms: DensityTransform,
    ) -> "Composition":
        return torchnf.utils.flow.inverted(cls)(*transforms)

    def forward(
        self, x: torch.Tensor, context: dict = {}
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass of the density transformation.
        """
        log_det_jacob = torch.zeros(x.shape[0], device=x.device)
        y = x
        for transform in self:
            y, ldj = transform(y, context)
            log_det_jacob = log_det_jacob.add(ldj)
        return y, log_det_jacob

    def inverse(
        self, y: torch.Tensor, context: dict = {}
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Inverse of :meth:`forward`.
        """
        log_det_jacob = torch.zeros(y.shape[0], device=y.device)
        x = y
        for transform in reversed(self):
            x, ldj = transform.inverse(x, context)
            log_det_jacob = log_det_jacob.add(ldj)
        return x, log_det_jacob

    def step_forward(self, x: torch.Tensor, context: dict = {}):
        """
        WIP
        """
        # TODO maybe replace this
        log_det_jacob = torch.zeros(x.shape[0]).type_as(x)
        y = x
        for transform in self:
            y, ldj = transform(y)
            log_det_jacob.add_(ldj)
            yield y, log_det_jacob
