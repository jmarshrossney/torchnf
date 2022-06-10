"""
"""
from typing import Optional, Union
import os

import torch

import torchnf.models
import torchnf.distributions


class MultivariateGaussianSampler(torchnf.models.BoltzmannGenerator):
    def __init__(
        self,
        flow: torchnf.flow.Flow,
        *,
        loc: torch.Tensor,
        covariance_matrix: Optional[torch.Tensor] = None,
        precision_matrix: Optional[torch.Tensor] = None,
        optimizer_spec: Optional[torchnf.models.OptimizerSpec] = None,
        output_dir: Optional[Union[str, os.PathLike]] = None,
    ):
        target = torch.distributions.MultivariateNormal(
            loc=loc,
            covariance_matrix=covariance_matrix,
            precision_matrix=precision_matrix,
        )
        prior = torchnf.distributions.SimplePrior(
            torch.distributions.Normal(0, 1),
            expand_shape=target.event_shape,
        )
        super().__init__(
            prior=prior,
            target=target,
            flow=flow,
            optimizer_spec=optimizer_spec,
            output_dir=output_dir,
        )


class VonMisesSampler(torchnf.models.BoltzmannGenerator):
    """
    .. attention:: Not yet implemented
    """

    def __init__(self, flow: torchnf.flow.Flow, batch_size: int) -> None:
        pass
