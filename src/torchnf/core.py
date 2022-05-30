from typing import Callable, NamedTuple, Optional, Union
import torch

import torchnf.conditioners
import torchnf.transformers


class Flow(torch.nn.Sequential):
    def forward(
        self, x: torch.Tensor, context: dict = {}
    ) -> tuple[torch.Tensor, torch.Tensor]:
        log_det_jacob = torch.zeros(x.shape[0]).type_as(x)
        y = x
        for layer in self:
            y, ldj = layer(y, context)
            log_det_jacob.add_(ldj)
        return y, log_det_jacob

    def inverse(
        self, y: torch.Tensor, context: dict = {}
    ) -> tuple[torch.Tensor, torch.Tensor]:
        log_det_jacob = torch.zeros(y.shape[0]).type_as(y)
        x = y
        for layer in reversed(self):
            x, ldj = layer.inverse(x, context)
            log_det_jacob.add_(ldj)
        return x, log_det_jacob

    def step_forward(self, x: torch.Tensor, context: dict = {}):
        # TODO maybe replace this
        log_det_jacob = torch.zeros(x.shape[0]).type_as(x)
        y = x
        for layer in self:
            y, ldj = layer(y)
            log_det_jacob.add_(ldj)
            yield y, log_det_jacob


class _FlowLayer(torch.nn.Module):
    def forward(
        self, x: torch.Tensor, context: dict = {}
    ) -> tuple[torch.Tensor, torch.Tensor]:
        raise NotImplementedError

    def inverse(
        self, x: torch.Tensor, context: dict = {}
    ) -> tuple[torch.Tensor, torch.Tensor]:
        raise NotImplementedError

    def freeze(self) -> None:
        for param in self.parameters():
            param.requires_grad = False

    def unfreeze(self) -> None:
        for param in self.parameters():
            param.requires_grad = True


class FlowLayer(_FlowLayer):
    def __init__(
        self,
        transformer: torchnf.transformers.Transformer,
        conditioner: torchnf.conditioners.Conditioner,
    ) -> None:
        super().__init__()
        self.transformer = transformer
        self.conditioner = conditioner

    def conditioner_forward(self, xy: torch.Tensor) -> torch.Tensor:
        return self.conditioner(xy)

    def transformer_forward(
        self,
        x: torch.Tensor,
        params: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        return self.transformer(x, params)

    def transformer_inverse(
        self,
        y: torch.Tensor,
        params: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        return self.transformer.inv(y, params)

    def forward(
        self, x: torch.Tensor, context: dict = {}
    ) -> tuple[torch.Tensor, torch.Tensor]:
        self.context = context
        params = self.conditioner_forward(x, context)
        y, ldj = self.transformer_forward(x, params, context)
        return y, ldj

    def inverse(
        self, y: torch.Tensor, context: dict = {}
    ) -> tuple[torch.Tensor, torch.Tensor]:
        self.context = context
        params = self.conditioner_forward(y, context)
        x, ldj = self.transformer_forward(y, params, context)
        return x, ldj
