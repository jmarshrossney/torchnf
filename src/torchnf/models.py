"""
Module containing classes which implement some standard models based on
Normalizing Flows.

See :mod:`torchnf.lit_models` for equivalent versions based on
:py:class:`pytorch_lightning.LightningModule`.
"""
import pathlib
from typing import Any, Callable, Optional, Union
import torch
import logging
import os
import tqdm.auto
import torch.utils.tensorboard as tensorboard

import torchnf.flow
import torchnf.distributions
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
        output_dir
            A dedicated directory for the model, where checkpoints, metrics,
            logs, outputs etc. will be saved to and loaded from. If not
            provided, the fallback is to call :meth:`generate_output_dir`.

    .. attention:: Currently multiple optimizers are not supported.
    """

    def __init__(self, output_dir: Optional[Union[str, os.PathLike]] = None):
        super().__init__()
        output_dir = output_dir or self.generate_output_dir()
        self._output_dir = pathlib.Path(str(output_dir)).resolve()

        self._global_step = 0
        self._most_recent_checkpoint = 0

    def generate_output_dir(self) -> Union[str, os.PathLike]:
        """
        Generates a unique path to a dedicated output directory for the model.
        """
        ts = torchnf.utils.timestamp()
        name = self.__class__.__name__
        output_dir = pathlib.Path(f"{name}_{ts}").resolve()
        return output_dir

    @property
    def output_dir(self) -> pathlib.Path:
        """
        Path to dedicated directory for this model.
        """
        return self._output_dir

    @property
    def logger(self) -> tensorboard.SummaryWriter:
        """
        Handles logging to Tensorboard.
        """
        if not hasattr(self, "_logger"):
            self._logger = tensorboard.SummaryWriter(
                self._output_dir.joinpath("logs")
            )
        return self._logger

    @property
    def global_step(self) -> int:
        """
        Total number of training steps since initialisation.
        """
        return self._global_step

    def configure_training(
        self,
        train_steps: int,
        train_batch_size: int,
        val_steps: int,
        val_batch_size: int,
        optimizer: str,
        optimizer_kwargs: dict,
        scheduler: str,
        scheduler_kwargs: dict,
        val_interval: Union[int, None] = -1,
        ckpt_interval: Union[int, None] = -1,
        pbar_interval: Union[int, None] = 25,
        logging_interval: Union[int, None] = 25,
    ) -> None:
        """
        Set up the model for training.

        This method must be executed before running :meth:`fit`.

        Args:
            train_steps
                Number of training steps to run
            train_batch_size
                Size of each training batch
            val_steps
                Number of validation steps to run, each time validation
                occurs
            val_batch_size
                Size of each validation batch
            optimizer
                String denoting an optimizer defined in ``torch.optim``
            optimizer_kwargs
                Keyword arguments for the optimizer
            scheduler
                String denoting a learning rate scheduler defined in
                ``torch.optim.lr_schedulers``
            scheduler_kwargs
                Keyword arguments for the scheduler
            val_interval
                Number of steps between validation runs. Set to `-1` to
                run validation on final step. Set to falsey to never
                run validation.
            ckpt_interval
                Number of steps between saving checkpoints. Set to `-1`
                to save checkpoint after final step. Set to `falsey` to
                never save checkpoints,
            pbar_interval
                Number of steps between updates of the progress bar.
                Set to `None` to disable progress bar.
            logging_interval
                Number of steps between calls to :meth:`self.log_training`.
        """
        self.train_steps = train_steps
        self.train_batch_size = train_batch_size
        self.val_steps = val_steps
        self.val_batch_size = val_batch_size
        self.optimizer = getattr(torch.optim, optimizer)(
            self.parameters(), **optimizer_kwargs
        )
        self.scheduler = getattr(torch.optim.lr_scheduler, scheduler)(
            self.optimizer, **scheduler_kwargs
        )
        self.val_interval = val_interval
        self.ckpt_interval = ckpt_interval
        self.pbar_interval = pbar_interval
        self.logging_interval = logging_interval

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
        ckpt_path = self.output_dir / "checkpoints"
        ckpt_path.mkdir(exists_ok=True, parents=True)
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
        # Have to instantiate the optimizers before we can load state dict
        if not hasattr(self, "optimizer") or not hasattr(self, "scheduler"):
            raise Exception("Optimizers have not yet been configured.")

        ckpt_path = self.output_dir / "checkpoints"
        step = step or self._most_recent_checkpoint
        ckpt = torch.load(ckpt_path / "ckpt_{step}.pt")

        assert ckpt["global_step"] == step

        self._global_step = ckpt["global_step"]
        self.load_state_dict(ckpt["model_state_dict"])
        self.optimizer.load_state_dict(ckpt["optimizer_state_dict"])
        self.scheduler.load_state_dict(ckpt["scheduler_state_dict"])

        log.info("Loaded checkpoint from step: {step}")

    def optimization_step(self, loss: torch.Tensor) -> None:
        """
        Performs a single optimization step.

        Unless overridden, this simply does the following:

        .. code-block:: python

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            self.scheduler.step()
        """
        # TODO multiple optimizers, ReduceLROnPlateau
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        self.scheduler.step()

    def fit(self) -> None:
        """
        Runs the training loop.

        Essentially what this does is the following:

        .. code-block::

            for _ in range(steps):
                loss = self.training_step(batch_size)
                self.optimizer_step(loss)

        along with incrementing :attr:`global_step` and, optionally,
        validating and saving checkpoints.

        Notes:
            The conditions for running validation and saving a checkpoint are
            based on the global step, not the step number within the current
            call to ``fit``.
        """
        # unchanged if +ve, final_step if -ve, final_step + 1 if falsey
        val_interval = (
            (self.val_interval if self.val_interval > 0 else self.train_steps)
            if self.val_interval
            else self.train_steps + 1
        )
        ckpt_interval = (
            (
                self.ckpt_interval
                if self.ckpt_interval > 0
                else self.train_steps
            )
            if self.ckpt_interval
            else self.train_steps + 1
        )

        pbar = (
            tqdm.auto.trange(
                self._global_step, self.train_steps, desc="Training"
            )
            if self.pbar_interval
            else range(self._global_step, self.train_steps)
        )
        pbar_interval = self.pbar_interval or self.train_steps + 1

        logging_interval = self.logging_interval or self.train_steps + 1

        self.train()
        torch.set_grad_enabled(True)

        for step in pbar:
            self._global_step += 1

            loss = self.training_step()
            self.optimization_step(loss)

            if self._global_step % pbar_interval == 0:
                pbar.set_postfix({"loss": f"{loss:.3e}"})

            if self._global_step % logging_interval == 0:
                self.log_training(loss)

            if self._global_step % val_interval == 0:
                pbar.set_description("Validating")
                with torch.no_grad(), torchnf.utils.eval_mode(self):
                    self.log_validation(self.validate())
                pbar.set_description("Training")

            if self._global_step % ckpt_interval == 0:
                self.save_checkpoint()

        # NOTE: close logger here?

        self.eval()

    def validate(self) -> list[Any]:
        """
        Runs the validation loop.
        Unless overidden, this will just call :meth:`validation_step`
        :code:`self.val_steps` times and return a list containing the
        returned values.
        """
        out = []
        for _ in range(self.val_steps):
            out.append(self.validation_step())
        return out

    def log_training(self, loss: torch.Tensor) -> None:
        """
        Log information during training.
        """
        self.logger.add_scalar("Training/loss", loss, self.global_step)
        self.logger.add_scalar(
            "Training/lr",
            torch.Tensor([self.scheduler.get_last_lr()]),
            self.global_step,
        )

    def log_validation(self, val_outputs: dict[str, torch.Tensor]) -> None:
        """
        Logs the outputs of :meth:`self.validate`.

        By default, this logs each element of `val_outputs` as a scalar.
        """
        for metric, tensor in val_outputs.items():
            self.logger.add_scalar(
                f"Validation/{metric}", tensor, self.global_step
            )

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
        prior: torchnf.distributions.Prior,
        target: torchnf.distributions.Target,
        flow: torchnf.flow.Flow,
        output_dir: Optional[Union[str, os.PathLike]] = None,
    ) -> None:
        super().__init__(output_dir)
        self.prior = prior
        self.target = target
        self.flow = flow

    def forward(self, batch_size: int) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Samples from the model and evaluates the statistical weights.

        What this does is

        .. code::

            x, log_prob_prior = self.prior.forward(batch_size)
            y, log_det_jacob = self.flow.forward(x)
            log_prob_target = self.target.log_prob(y)
            log_weights = log_prob_target - log_prob_prior + log_det_jacob
            return y, log_weights
        """
        x, log_prob_prior = self.prior.forward(batch_size)
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

            _, log_weights = self.forward(self.train_batch_size)
            loss = log_weights.mean().neg()
            return loss
        """
        _, log_weights = self.forward(self.train_batch_size)
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

    def validation_step(self) -> torchnf.metrics.LogWeightMetrics:
        """
        Performs a single validation step, returning some metrics.
        """
        _, log_weights = self.forward(self.val_batch_size)
        metrics = torchnf.metrics.LogWeightMetrics(log_weights)
        return metrics

    def validate(self) -> dict[str, torch.Tensor]:
        """
        Returns a dict of tensors containing validation metrics.
        """
        return torchnf.metrics.LogWeightMetrics.combine(super().validate())

    @staticmethod
    def _concat_samples(
        *samples: tuple[torch.Tensor, torch.Tensor],
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Helper function to concatenate samples and their log weights.
        """
        data, weights = list(map(list, zip(*samples)))
        return torch.cat(data, dim=0), torch.cat(weights, dim=0)

    @torch.no_grad()
    def weighted_sample(
        self, batch_size: int, batches: int = 1
    ) -> torch.Tensor:
        """
        Generate a weighted sample from the model.

        This simply called the :meth:`forward` method multiple times
        (with gradients disabled), and concatenates the result.

        Sampling from the model results in a biased sample with respect
        to the target distribution. However, the statistical weights
        allow unbiased expectation values to be calculated.

        Args:
            batch_size:
                Size of each batch
            batches
                Number of batches in the sample

        Returns:
            Tuple containing (1) the sample and (2) the logarithm of
            statistical weights.
        """
        out = []
        for _ in range(batches):
            out.append(self.forward(batch_size))
        return self._concat_samples(*out)

    @property
    def mcmc_current_state(
        self,
    ) -> Union[tuple[torch.Tensor, torch.Tensor], None]:
        """
        Current state of the Markov chain, and it's log statistical weight.

        This is used by :meth:`mcmc_sample` to initialise the Markov chain.

        Usually, there is no need for the user to set this explicitly;
        it will be updated automatically at the end of an MCMC sampling
        phase.
        """
        try:
            return self._mcmc_current_state
        except AttributeError:
            return None

    @mcmc_current_state.setter
    def mcmc_current_state(
        self, state: tuple[torch.Tensor, torch.Tensor]
    ) -> None:
        self._mcmc_current_state = state

    @mcmc_current_state.deleter
    def mcmc_current_state(self) -> None:
        del self._mcmc_current_state

    @torch.no_grad()
    def mcmc_sample(
        self,
        batch_size: int,
        batches: int = 1,
    ) -> torch.Tensor:
        r"""
        Generate an unbiased sample from the target distribution.

        The Boltzmann Generator is used as the proposal distribution in
        the Metropolis-Hastings algorithm. An asymptotically unbiased
        sample is generated by accepting or rejecting data drawn from
        the model using the Metropolis test:

        .. math::

            A(y \to y^\prime) = \min \left( 1,
           \frac{q(y)}{p(y)} \frac{p(y^\prime)}{q(y^\prime)} \right) \, .

        Args:
            batch_size:
                Size of each batch
            batches
                Number of batches in the sample

        Notes:
            If :attr:`mcmc_current_state` is not :code:`None`, the
            Markov chain will be initialised using this state. Oherwise,
            an initial state will be drawn from the **prior** distribution.

        .. seealso:: :py:func:`torchnf.utils.metropolis_test`
        """
        # Initialise with a random state drawn from the prior
        if self.mcmc_current_state is None:
            self.mcmc_current_state = self.prior.forward(1)

        out = []

        for _ in range(batches):
            y, logw = self._concat_samples(
                self.mcmc_current_state, self.forward(batch_size)
            )
            indices = torchnf.utils.metropolis_test(logw)

            out.append(y[indices])

            curr_idx = indices[-1]
            self.mcmc_current_state = (
                y[curr_idx].unsqueeze(0),
                logw[curr_idx].unsqueeze(0),
            )

        return torch.cat(out, dim=0)
