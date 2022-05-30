from typing import Optional, Union
import torch

import torchnf


class BoltzmannGenerator(torch.nn.Module):
    def __init__(
        self,
        prior: torchnf.Prior,
        target: torchnf.Target,
        flow: torchnf.Flow,
    ) -> None:
        super().__init__()
        self.prior = prior
        self.target = target
        self.flow = flow

    def forward(self) -> tuple[torch.Tensor, torch.Tensor]:
        x, log_prob_prior = self.prior()
        y, log_det_jacob = self.flow(x)
        log_prob_target = self.target(y)
        log_weights = log_prob_target - log_prob_prior + log_det_jacob
        return y, log_weights

    def inverse(self, y: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Inverse pass of the Boltzmann Generator.

        Takes a sample drawn from the target distribution, passes
        it through the inverse-flow, and evaluates the density
        of the outputs under the prior distribution.
        """
        raise NotImplementedError

    @torch.no_grad
    def sample(self) -> tuple[torch.Tensor, torch.Tensor]:
        return self.forward()
