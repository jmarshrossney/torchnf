"""
"""
import functools

from jsonargparse.typing import PositiveInt
import torch
from torch.distributions import Distribution
import pytorch_lightning as pl

from torchnf.abc import DensityTransform
from torchnf.utils.tensor import tuple_concat


def eval_mode(meth):
    """
    Decorator which sets a model to eval mode for the duration of the method.
    """

    @functools.wraps(meth)
    def wrapper(model: torch.nn.Module, *args, **kwargs):
        original_state = model.training
        model.eval()
        out = meth(model, *args, **kwargs)
        model.train(original_state)
        return out

    return wrapper


class BijectiveAutoencoder(pl.LightningModule):
    def __init__(
        self,
        flow: DensityTransform,
        latent_dist: Distribution,
    ) -> None:
        super().__init__()
        self.flow = flow
        self.latent_dist = latent_dist

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        r"""
        Encodes the input data and computes the log likelihood.

        The log likelihood of the point :math:`x` under the model is

        .. math::

            \log \ell(x) = \log q(z)
            + \log \left\lvert \frac{\partial z}{\partial x} \right\rvert

        where :math:`z` is the corresponding point in latent space,
        :math:`q` is the latent distribution, and the Jacobian is that
        of the encoding transformation.

        Args:
            x:
                A batch of data drawn from the target distribution

        Returns:
            Tuple containing the encoded data and the log likelihood
            under the model
        """
        z, log_det_jacob = self.flow.forward(x)  # TODO: self.flow_forward?
        log_prob_z = self.latent_dist.log_prob(z)
        log_prob_x = log_prob_z + log_det_jacob
        return z, log_prob_x

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        r"""
        Decodes the latent data and computes the log statistical weights.

        The log likelihood of the point :math:`x` under the model is

        .. math::

            \log \ell(x) = \log q(z)
            - \log \left\lvert \frac{\partial x}{\partial z} \right\rvert

        where :math:`z` is the corresponding point in latent space,
        :math:`q` is the latent distribution, and the Jacobian is that
        of the decoding transformation.

        Args:
            z:
                A batch of latent variables drawn from the latent
                distribution

        Returns:
            Tuple containing the decoded data and the log likelihood
            under the model
        """
        log_prob_z = self.latent_dist.log_prob(z)
        x, log_det_jacob = self.flow.inverse(z)
        log_prob_x = log_prob_z - log_det_jacob
        return x, log_prob_x

    def forward(
        self, batch: list[torch.Tensor], *_, **__
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Calls :meth:`encode` on the input batch.
        """
        (x,) = batch
        return self.encode(x)

    def training_step(
        self, batch: list[torch.Tensor], *_, **__
    ) -> torch.Tensor:
        r"""
        Performs a single training step, returning the loss.

        The loss returned is the mean of the negative log-likelihood
        of the inputs, under the model, i.e.

        .. math::

            L(\{x\}) = \frac{1}{N} \sum_{\{x\}} -\log q(x)
        """
        z, log_prob_x = self(batch)
        loss = log_prob_x.mean().neg()  # forward KL
        self.log("loss/train", loss, on_step=False, on_epoch=True)
        return loss

    def validation_step(
        self, batch: list[torch.Tensor], *_, **__
    ) -> torch.Tensor:
        """
        Performs a single validation step, returning the encoded data.
        """
        z, log_prob_x = self(batch)
        loss = log_prob_x.mean().neg()
        self.log("loss/validation", loss, on_step=False, on_epoch=True)
        return z, log_prob_x

    def test_step(self, batch: list[torch.Tensor], *_, **__) -> torch.Tensor:
        """
        Returns encoded data and log prob.
        """
        return self(batch)

    def test_epoch_end(
        self, test_outputs: list[tuple[torch.Tensor, torch.Tensor]]
    ) -> None:
        """
        Computes test metrics and logs result.
        """
        z, log_prob_x = tuple_concat(*test_outputs)
        loss = log_prob_x.mean().neg()
        self.logger.log_hyperparams(
            self.hparams or {"_": -1}, {"loss/test": loss}
        )

    @torch.no_grad()
    @eval_mode
    def sample(
        self, batch_size: PositiveInt, batches: PositiveInt = 1
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Generate synthetic data by sampling from the model.

        Returns:
            Tuple containing generated sample and un-normalised log
            probability density under the model
        """
        out = []
        for _ in range(batches):
            z = self.latent_dist.sample([batch_size])
            x, log_prob_x = self.decode(z)
            out.append((x, log_prob_x))
        return tuple_concat(*out)
