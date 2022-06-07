"""
Module containing classes which implement some standard models based on
Normalizing Flows.

See :mod:`torchnf.lit_models` for equivalent versions based on
:py:class:`pytorch_lightning.LightningModule`.
"""
from functools import cached_property
import pathlib
from typing import Callable, Optional, Union
import torch
import logging
import os
from tqdm.auto import trange

import torchnf.flow
import torchnf.prior
import torchnf.utils
import torchnf.metrics

log = logging.getLogger(__name__)


class Model(torch.nn.Module):
    """
    Base class for models.

    Essentially this class acts as a container for a Normalizing Flow,
    in which it can be trained, saved, loaded etc. It assigns a specific
    directory to the model, where checkpoints, metrics, state dicts etc
    can be saved and loaded.

    Args:
        output_dir:
            A dedicated directory for the model, where checkpoints, metrics,
            logs, outputs etc. will be saved to and loaded from. If not
            provided, resorts to the default of `<class_name>_<timestamp>`
            in the current working directory.
    """

    def __init__(self, output_dir: Optional[Union[str, os.PathLike]] = None):
        super().__init__()
        if output_dir is not None:
            self._output_dir = pathlib.Path(str(output_dir)).resolve()

        self._global_step = 0
        self._most_recent_checkpoint = 0

    def _default_output_dir(self) -> pathlib.Path:
        ts = torchnf.utils.timestamp()
        name = self.__class__.__name__
        output_dir = pathlib.Path(f"{name}_{ts}").resolve()
        assert (
            not output_dir.exists()
        ), "Default output dir '{output_dir}' already exists!"

    @property
    def output_dir(self) -> pathlib.Path:
        """
        Path to dedicated directory for this model.
        """
        try:
            return self._output_dir
        except AttributeError:
            output_dir = self._default_output_dir()
            log.info("Using output directory: %s" % output_dir)
            self._output_dir = output_dir
            return output_dir

    @property
    def global_step(self) -> int:
        """
        Total number of training steps since initialisation.
        """
        return self._global_step

    def configure_optimizers(self) -> None:
        """
        Assigns an optimizer and learning rate scheduler.

        .. attention:: This method should be overidden by the user!

        Example:
            .. code-block:: python

                self.optimizer = torch.optim.Adam(
                    self.flow.parameters(),
                    lr=0.001,
                )
                self.scheduler = torch.optim.lr_schedulers.CosineAnnealingLR(
                    self.optimizer,
                    T_max=10000,
                )
        """
        raise NotImplementedError

    def save_checkpoint(self) -> None:
        """
        Saves a checkpoint containing the current model state.

        The checkpoint gets saved to `self.output_dir/checkpoints/`
        under the name `ckpt_<self.global_step>.pt`. It contains
        a snapshot of the state dict of the model, optimizer, and
        scheduler, as well as the global step.

        .. seealso:: :meth:`load_checkpoint`
        """
        ckpt = {
            "global_step": self._global_step,
            "model_state_dict": self.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "scheduler_state_dict": self.scheduler.state_dict(),
        }
        ckpt_path = self._output_dir / "checkpoints"
        ckpt_path.mkdirs(exists_ok=True, parents=True)
        torch.save(
            ckpt,
            ckpt_path / "ckpt_{self._global_step}.pt",
        )
        log.info("Checkpoint saved at step: {self._global_step}")
        self._most_recent_checkpoint = self._global_step

    def load_checkpoint(self, step: Optional[int] = None) -> None:
        """
        Loads a checkpoint from a `.pt` file.

        See Also:
            :meth:`save_checkpoint`
        """
        if self._global_step == 0:
            self.configure_optimizers()

        ckpt_path = self._output_dir / "checkpoints"
        step = step or self._most_recent_checkpoint
        ckpt = torch.load(ckpt_path / "ckpt_{step}.pt")

        assert ckpt["global_step"] == step

        self._global_step = ckpt["global_step"]
        self.load_state_dict(ckpt["model_state_dict"])
        self.optimizer.load_state_dict(ckpt["optimizer_state_dict"])
        self.scheduler.load_state_dict(ckpt["scheduler_state_dict"])

        log.info("Loaded checkpoint from step: {step}")

    def fit(
        self,
        n_steps: int,
        val_interval: Optional[int] = None,
        ckpt_interval: Optional[int] = None,
        pbar_interval: int = 25.0,
    ) -> None:
        """
        Runs the training loop.

        Essentially what this does is the following:

        .. code-block::

            for step in range(n_steps):
                self.global_step += 1

                loss = self.training_step()

                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
                self.scheduler.step()

        Args:
            n_steps
                Number of training steps to run
            val_interval
                Number of steps between validation runs
            ckpt_interval
                Number of steps between saving checkpoints
            pbar_interval
                Number of steps between updates of the progress bar

        Notes:
            The conditions for running validation and saving a checkpoint are
            based on the global step, not the step number within the current
            call to :code:`fit`.
        """
        if self._global_step == 0:
            self.configure_optimizers()

        self.train()
        torch.set_grad_enabled(True)

        pbar = trange(n_steps, desc="Training")
        for step in pbar:
            self._global_step += 1

            loss = self.training_step()

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            self.scheduler.step()

            if self._global_step % pbar_interval == 0:
                pbar.set_postfix({"loss": f"{loss:.3e}"})

            if val_interval:
                if self._global_step % val_interval == 0:
                    _ = self.validation_step()
            if ckpt_interval:
                if self._global_step % ckpt_interval == 0:
                    self.save_checkpoint()

    def validate(self) -> dict:
        """
        Runs the validation loop.

        Unless overidden, this will just call :meth:`validation_step`
        once and return the result.
        """
        return self.validation_step()

    def training_step(self) -> torch.Tensor:
        """
        Performs a single training step, returning the loss.

        .. attention:: This must be overridden by the user!

        """
        raise NotImplementedError

    def validation_step(self) -> dict:
        """
        Performs a single validation step, returning any metrics.

        .. attention:: This must be overridden by the user!

        """
        raise NotImplementedError


class BoltzmannGenerator(Model):
    """
    Model representing a Boltzmann Generator based on a Normalizing Flow.

    References:
        The term 'Boltzmann Generator' was introduced in
        :arxiv:`1812.01729`.
    """

    def __init__(
        self,
        *,
        prior: torchnf.prior.Prior,
        target: Callable[torch.Tensor, torch.Tensor],
        flow: torchnf.flow.Flow,
    ) -> None:
        super().__init__()
        self.prior = prior
        self.target = target
        self.flow = flow

    def forward(self) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Samples from the model and evaluates the statistical weights.

        What this does is

        .. code::

            x, log_prob_prior = self.prior.forward()
            y, log_det_jacob = self.flow.forward(x)
            log_prob_target = self.target.log_prob(y)
            log_weights = log_prob_target - log_prob_prior + log_det_jacob
            return y, log_weights
        """
        x, log_prob_prior = self.prior.forward()
        y, log_det_jacob = self.flow.forward(x)
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

    def training_step_rev_kl(self) -> torch.Tensor:
        """
        Performs a single 'reverse KL' training step.

        This simply does the following:

        .. code-block:: python

            _, log_weights = self.forward()
            loss = log_weights.mean().neg()
            return loss
        """
        _, log_weights = self.forward()
        loss = log_weights.mean().neg()
        return loss

    def training_step_fwd_kl(self) -> torch.Tensor:
        """
        Raises:
            NotImplementedError
        """
        raise NotImplementedError

    def training_step(self) -> torch.Tensor:
        """
        Performs a single training step, returning the loss.
        """
        return self.training_step_rev_kl()

    def validation_step(self) -> dict:
        """
        Performs a single validation step, returning some metrics.
        """
        _, log_weights = self.forward()
        metrics = torchnf.metrics.LogWeightMetrics(log_weights)
        return metrics.asdict()

    @torch.no_grad()
    def sample(self) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Sample from the model.

        This simply called the :meth:`forward` method with gradients
        disabled.
        """
        return self.forward()
