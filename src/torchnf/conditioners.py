from typing import Callable, Optional, Union
import torch

import torchnf.utils


class Conditioner(torch.nn.Module):
    pass


class SimpleConditioner(Conditioner):
    """Diagonal Jacobian"""

    def __init__(
        self,
        init_params: Union[torch.Tensor, list[float]],
        params_dim: int = 1,
    ) -> None:
        super().__init__()
        self.params = torch.nn.Parameter(torch.Tensor(init_params))
        self.params_dim = params_dim

    def forward(
        self, inputs: torch.Tensor, context: dict = {}
    ) -> torch.Tensor:
        return torchnf.utils.expand_elements(
            self.params,
            shape=inputs.shape[1:],
            stack_dim=self.params_dim,
        )


class MaskedConditioner(Conditioner):
    """Coupling-layer type jacobian"""

    def __init__(
        self,
        mask: Optional[torch.BoolTensor] = None,
        net_spec: Optional[list[dict]] = None,
        net_builder: Callable[
            list[dict], torch.nn.Sequential
        ] = torchnf.utils.simple_net_builder,
    ) -> None:
        super().__init__()
        if mask:
            self.register_buffer("_mask", mask)
        if net_spec:
            self._net = net_builder(net_spec)

    def get_mask(self, x: torch.Tensor) -> torch.BoolTensor:
        return self._mask

    def _forward(self, x: torch.Tensor) -> torch.Tensor:
        return self._net(x)

    def forward(self, x: torch.Tensor, context: dict = {}) -> torch.Tensor:
        self.context = context
        mask = self.get_mask(x)
        params = self._forward(x.mul(mask))
        return params.mul(~mask).add(
            torch.full_like(params, fill_value=float("nan")).mul(mask)
        )


class AutoregressiveConditioner(Conditioner):
    """Triangular jacobian"""

    pass
