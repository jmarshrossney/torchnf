"""
"""
from typing import Optional

import torch

import torchnf.models
import torchnf.prior


class MultivariateGaussianSampler(torchnf.models.BoltzmannGenerator):
    def __init__(
        self,
        flow: torchnf.flow.Flow,
        *,
        loc: torch.Tensor,
        covariance_matrix: Optional[torch.Tensor] = None,
        precision_matrix: Optional[torch.Tensor] = None,
        batch_size: int = 100,
    ):
        target = torch.distributions.MultivariateNormal(
            loc=loc,
            covariance_matrix=covariance_matrix,
            precision_matrix=precision_matrix,
        )
        prior = torchnf.prior.SimplePrior(
            torch.distributions.Normal(0, 1),
            batch_size=batch_size,
            expand_shape=target.event_shape,
        )
        super().__init__(prior=prior, target=target, flow=flow)

    def configure_optimizers(self) -> None:
        self.optimizer = torch.optim.Adam(self.flow.parameters(), lr=0.1)
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer,
            T_max=1000,
            eta_min=1e-6,
        )


class VonMisesSampler(torchnf.models.BoltzmannGenerator):
    def __init__(self, flow: torchnf.flow.Flow, batch_size: int) -> None:
        pass
