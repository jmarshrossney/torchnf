from typing import Callable
import torch
import pytorch_lightning as pl

import torchnf


class LitBoltzmannGenerator(pl.LightningModule):
    """Test"""
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

    def forward(self, batch):
        """."""
        x, log_prob_prior = batch
        y, log_det_jacob = self.flow(x)
        log_prob_target = self.target(y)
        log_weights = log_prob_target - log_prob_prior + log_det_jacob
        return y, log_weights

    def train_dataloader(self):
        """."""
        return self.prior

    def training_step(self, batch, batch_idx):
        """."""
        y, log_weights = self.forward(batch)
        loss = log_weights.mean().neg()
        return loss

    def training_step_end(self, outputs):
        """."""
        print(outputs)

    def val_dataloader(self):
        """."""
        return self.prior

    def validation_step(self, batch, batch_idx):
        """."""
        y, log_weights = self.forward(batch)

    def validation_epoch_end(self, metrics):
        """."""
        # combine metrics into mean and std. dev.
        # log metrics
        pass

    def configure_optimizers(self):
        """."""
        # raise NotImplementedError
        optimizer = torch.optim.Adam(self.flow.parameters())
        return optimizer

    @torch.no_grad()
    def sample(self) -> tuple[torch.Tensor]:
        return self.forward(self.prior())
