import dataclasses

import torch

import torchnf.flow
import torchnf.recipes.layers
import torchnf.recipes.networks


@dataclasses.dataclass
class NICE:
    net: torchnf.recipes.networks.NetBuilder
    mask: torch.BoolTensor
    n_blocks: int

    def __post_init__(self) -> None:
        pass

    def __call__(self) -> torchnf.flow.Flow:
        raise NotImplementedError


@dataclasses.dataclass
class RealNVP:
    n_layers: int

    def __post_init__(self) -> None:
        pass

    def __call__(self) -> torchnf.flow.Flow:
        raise NotImplementedError
