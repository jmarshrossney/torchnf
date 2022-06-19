import dataclasses

import torch
from jsonargparse.typing import PositiveInt

import torchnf.conditioners
import torchnf.flow
import torchnf.transformers
import torchnf.recipes.networks


@dataclasses.dataclass
class CouplingLayer:

    net: torchnf.recipes.networks.NetBuilder
    mask: torch.BoolTensor

    @property
    def transformer(self) -> torchnf.transformers.Transformer:
        raise NotImplementedError

    @property
    def conditioner(self) -> torchnf.conditioners.MaskedConditioner:
        # raise NotImplementedError
        return torchnf.conditioners.MaskedConditioner(self.net(), self.mask)

    def __call__(self) -> torchnf.flow.FlowLayer:
        return torchnf.flow.FlowLayer(self.transformer, self.conditioner)


class AdditiveCouplingLayer(CouplingLayer):
    @property
    def transformer(self) -> torchnf.transformers.Translation:
        return torchnf.transformers.Translation()


class MultiplicativeCouplingLayer(CouplingLayer):
    @property
    def transformer(self) -> torchnf.transformers.Rescaling:
        return torchnf.transformers.Rescaling()


class AffineCouplingLayer(CouplingLayer):
    @property
    def transformer(self) -> torchnf.transformers.AffineTransform:
        return torchnf.transformers.AffineTransform()


@dataclasses.dataclass
class RQSplineCouplingLayer(CouplingLayer):
    n_segments: PositiveInt
    interval: tuple[float]
    domain: str = "reals"

    def __post_init__(self) -> None:
        assert self.domain in ("reals", "circle", "interval"), "invalid domain"
        self._transformer_cls = {
            "reals": torchnf.transformers.RQSplineTransform,
            "interval": torchnf.transformers.RQSplineTransformIntervalDomain,
            "circle": torchnf.transformers.RQSplineTransformCircularDomain,
        }.__getitem__(self.domain)

    @property
    def transformer(self) -> torchnf.transformers.RQSplineTransform:
        return self._transformer_cls(
            n_segments=self.n_segments, interval=self.interval
        )
