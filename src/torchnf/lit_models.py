"""
Alternative implementations of the models from :mod:`torchnf.models`, based
on :py:class:`pytorch_lightning.LightningModule` rather than the standard
:py:class:`torch.nn.Module`.

.. attention:: This is a work in progress. Do not use.
"""
import dataclasses
import functools
import types
from typing import Optional, Union

import torch
import pytorch_lightning as pl

from torchnf.distributions import Target
import torchnf.flow
import torchnf.metrics


@dataclasses.dataclass
class OptimizerConfig:
    optimizer: Union[str, type[torch.optim.Optimizer]]
    optimizer_init: dict = dataclasses.field(default_factory=dict)
    scheduler: Optional[
        Union[str, type[torch.optim.lr_scheduler._LRScheduler]]
    ] = None
    scheduler_init: dict = dataclasses.field(default_factory=dict)
    submodule: Optional[str] = None

    def __post_init__(self) -> None:
        if isinstance(self.optimizer, str):
            self.optimizer = getattr(torch.optim, self.optimizer)
        if isinstance(self.scheduler, str):
            self.scheduler = getattr(torch.optim.lr_scheduler, self.scheduler)

    @staticmethod
    def configure_optimizers(
        model: pl.LightningModule,
        optimizer: torch.optim.Optimizer,
        scheduler: Optional[torch.optim.lr_scheduler._LRScheduler] = None,
    ):
        if scheduler is None:
            return optimizer
        return [optimizer], [scheduler]

    def add_to(self, model: pl.LightningModule) -> None:
        module = getattr(model, self.submodule) if self.submodule else model
        optimizer = self.optimizer(module.parameters(), **self.optimizer_init)
        scheduler = (
            self.scheduler(optimizer, **self.scheduler_init)
            if self.scheduler is not None
            else self.scheduler
        )

        configure_optimizers = functools.partial(
            self.configure_optimizers,
            optimizer=optimizer,
            scheduler=scheduler,
        )

        # Adds __wrapped__ attribute to partial fn, required for
        # PyTorch Lightning to regard configure_optimizers as overridden
        # (see pytorch_lightning.utilities.model_helpers.is_overridden)
        functools.update_wrapper(
            configure_optimizers, self.configure_optimizers
        )

        model.configure_optimizers = types.MethodType(
            configure_optimizers, model
        )


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
