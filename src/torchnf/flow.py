import torch

import torchnf.conditioners
import torchnf.transformers


class FlowLayer(torch.nn.Module):
    """
    Class representing a layer of a Normalizing Flow.

    This is a subclass of :py:class:`torch.nn.Module` with a specific
    modular structure that provides the flexibility to construct a variety
    of different flows.

    Args:
        transformer
            An instance of :class:`torchnf.transformers.Transformer` which
            implements the forward and inverse transformations, given an
            input tensor and a tensor of parameters which the transformation
            is conditioned on
        conditioner
            An instance of :class:`torchnf.conditioners.Conditioner` whose
            :code:`forward` method takes the flow layer inputs and returns
            a tensor of parameters for the transformer
    """

    def __init__(
        self,
        transformer: torchnf.transformers.Transformer,
        conditioner: torchnf.conditioners.Conditioner,
    ) -> None:
        super().__init__()
        self.transformer = transformer
        self.conditioner = conditioner

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
        Forward pass of the flow layer.

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
        x, ldj = self.transformer_forward(y, params)
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


class Flow(torch.nn.Sequential):
    """
    Class representing a Normalizing Flow.

    This is a subclass of :py:class:`torch.nn.Sequential` which composes
    several :class:`torchnf.flow.FlowLayer`'s and aggregates their
    log Jacobian determinants.
    """

    def __init__(self, *layers: FlowLayer) -> None:
        super().__init__(*layers)

    def forward(
        self, x: torch.Tensor, context: dict = {}
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass of the Normalizing Flow.
        """
        log_det_jacob = torch.zeros(x.shape[0], device=x.device)
        y = x
        for layer in self:
            y, ldj = layer(y, context)
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
        for layer in reversed(self):
            x, ldj = layer.inverse(x, context)
            log_det_jacob = log_det_jacob.add(ldj)
        return x, log_det_jacob

    def step_forward(self, x: torch.Tensor, context: dict = {}):
        """
        WIP
        """
        # TODO maybe replace this
        log_det_jacob = torch.zeros(x.shape[0]).type_as(x)
        y = x
        for layer in self:
            y, ldj = layer(y)
            log_det_jacob.add_(ldj)
            yield y, log_det_jacob
