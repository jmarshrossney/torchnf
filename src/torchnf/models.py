from functools import cached_property
import pathlib
from typing import Callable, Optional, Union
import torch
import logging
import os

import torchnf
import torchnf.utils
import torchnf.metrics

log = logging.getLogger(__name__)


class Model(torch.nn.Module):
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
        try:
            return self._output_dir
        except AttributeError:
            output_dir = self._default_output_dir()
            log.info("Using output directory: %s" % output_dir)
            self._output_dir = output_dir
            return output_dir

    @property
    def global_step(self) -> int:
        return self._global_step

    def configure_optimizers(self) -> None:
        raise NotImplementedError

    def save_checkpoint(self) -> None:
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
    ) -> None:

        if self._global_step == 0:
            self.configure_optimizers()

        self.train()
        torch.set_grad_enabled(True)

        for step in range(n_steps):
            self._global_step += 1

            loss = self.training_step()

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            self.scheduler.step()

            if val_interval:
                if self._global_step % val_interval == 0:
                    _ = self.validation_step()
            if ckpt_interval:
                if self._global_step % ckpt_interval == 0:
                    self.save_checkpoint()

    def validate(self) -> dict:
        return self.validation_step()

    def training_step(self) -> torch.Tensor:
        raise NotImplementedError

    def validation_step(self):
        raise NotImplementedError


class BoltzmannGenerator(Model):
    def __init__(
        self,
        prior: torchnf.Prior,
        target: Callable[torch.Tensor, torch.Tensor],
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

    def _training_step_rev_kl(self) -> float:
        _, log_weights = self.forward()
        loss = log_weights.mean().neg()
        return loss

    def _training_step_fwd_kl(self) -> float:
        raise NotImplementedError

    def training_step(self) -> torch.Tensor:
        return self._training_step_rev_kl()

    def validation_step(self) -> float:
        _, log_weights = self.forward()
        metrics = torchnf.metrics.LogWeightMetrics(log_weights)
        return metrics.asdict()

    @torch.no_grad()
    def sample(self) -> tuple[torch.Tensor, torch.Tensor]:
        return self.forward()
