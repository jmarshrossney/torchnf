"""
Alternative implementations of the models from :mod:`torchnf.models`, based
on :py:class:`pytorch_lightning.LightningModule` rather than the standard
:py:class:`torch.nn.Module`.

.. attention:: This is a work in progress. Do not use.
"""
import torch
import pytorch_lightning as pl

from torchnf.distributions import Target
import torchnf.flow
import torchnf.metrics


class LitBoltzmannGenerator(pl.LightningModule):
    """
    Model representing Boltzmann Generator based on Normalizing Flow.

    .. seealso:: :py:class:`torchnf.models.BoltzmannGenerator`
    """

    def __init__(
        self,
        flow: torchnf.flow.Flow,
        target: Target,
    ) -> None:
        super().__init__()
        self.flow = flow
        self.target = target

        self.metrics = torchnf.metrics.LogStatWeightMetricCollection()

    def forward(
        self, batch: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Passes inputs through the flow and evaluates the statistical weights.

        What this does is

        .. code::

            x, log_prob_prior = batch
            y, log_det_jacob = self.flow.forward(x)
            log_prob_target = self.target.log_prob(y)
            log_weights = log_prob_target - log_prob_prior + log_det_jacob
            return y, log_weights
        """
        x, log_prob_prior = batch
        y, log_det_jacob = self.flow(x)
        log_prob_target = self.target.log_prob(y)
        log_weights = log_prob_target - log_prob_prior + log_det_jacob
        return y, log_weights

    def inverse(self, y: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Inverse pass of the Boltzmann Generator.

        Takes a sample drawn from the target distribution, passes
        it through the inverse-flow, and evaluates the density
        of the outputs under the prior distribution.

        Raises:
            NotImplementedError
        """
        raise NotImplementedError

    def training_step_rev_kl(self, batch: torch.Tensor) -> torch.Tensor:
        """
        Performs a single 'reverse KL' training step.

        This simply does the following:

        .. code-block:: python

            _, log_weights = self.forward(batch)
            loss = log_weights.mean().neg()
            return loss
        """
        _, log_weights = self(batch)
        loss = log_weights.mean().neg()
        return loss

    def training_step_fwd_kl(self, batch: torch.Tensor, _):
        raise NotImplementedError
        # x, something = self.inverse(batch)
        # loss =

    def training_step(self, batch: torch.Tensor, _):
        """
        Performs a single training step, returning the loss.
        """
        loss = self.training_step_rev_kl(batch)
        self.log("train/loss", loss)
        return loss

    def validation_step(self, batch: torch.Tensor, _):
        """
        Performs a single validation step.
        """
        y, log_weights = self(batch)
        self.metrics.update(log_weights)

    def validation_epoch_end(self, val_outputs):
        """
        Compute and log metrics at the end of an epoch.
        """
        metrics = self.metrics.compute()
        self.log("validation", metrics)
        self.metrics.reset()

    def test_step(self, batch: torch.Tensor, _):
        """
        Performs a single test step.
        """
        y, log_weights = self(batch)
        self.metrics.update(log_weights)
