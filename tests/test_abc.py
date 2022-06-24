import torch

from torchnf.abc import *
from torchnf.flow import *


def test_isinstance_custom_densitytransform():
    class MyTransform:
        def forward(
            self, x: torch.Tensor, context: dict = {}
        ) -> tuple[torch.Tensor, torch.Tensor]:
            ...

        def inverse(
            self, x: torch.Tensor, context: dict = {}
        ) -> tuple[torch.Tensor, torch.Tensor]:
            ...

    class MyTransformModule(MyTransform, torch.nn.Module):
        def forward(
            self, x: torch.Tensor, context: dict = {}
        ) -> tuple[torch.Tensor, torch.Tensor]:
            ...

        def inverse(
            self, x: torch.Tensor, context: dict = {}
        ) -> tuple[torch.Tensor, torch.Tensor]:
            ...

    assert not isinstance(MyTransform(), DensityTransform)
    assert not isinstance(torch.nn.Module(), DensityTransform)
    assert isinstance(MyTransformModule(), DensityTransform)


def test_isinstance_flow_densitytransform():
    layer = FlowLayer(Transformer, Conditioner)
    assert isinstance(layer, DensityTransform)
    composed = Composition(layer)
    assert isinstance(composed, DensityTransform)


def test_isinstance_torchdist_target():
    gauss = torch.distributions.Normal(0, 1)
    assert isinstance(gauss, TargetDistribution)


def test_isinstance_custom_target():
    class MyTarget:
        def log_prob(self, sample: torch.Tensor) -> torch.Tensor:
            ...

    assert isinstance(MyTarget(), TargetDistribution)
