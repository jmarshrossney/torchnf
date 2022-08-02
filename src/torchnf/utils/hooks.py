from typing import Callable

import torch


def capture_data_shape(
    hook_handle: str, attr_name: str = "data_shape"
) -> Callable:
    """
    Dynamically extract the data shape on first call to ``forward``.
    """

    def hook(model: torch.nn.Module, input: torch.Tensor) -> None:
        data_shape = input.shape[1:]
        setattr(model, attr_name, data_shape)
        handle = getattr(model, hook_handle)
        handle.remove()
