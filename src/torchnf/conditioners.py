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
    ) -> None:
        super().__init__()
        self.params = torch.nn.Parameter(torch.Tensor(init_params))

    def forward(
        self, inputs: torch.Tensor, context: dict = {}
    ) -> torch.Tensor:
        batch_size, *data_shape = inputs.shape
        params = torchnf.utils.expand_elements(
            self.params,
            data_shape,
            stack_dim=0,
        )
        return params.expand([batch_size, -1, *data_shape])


class MaskedConditioner(Conditioner):
    """Coupling-layer type jacobian"""

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
