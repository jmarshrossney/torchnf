"""
"""
from collections.abc import Iterator
import functools
from typing import Optional, Union

from jsonargparse.typing import PositiveInt
import torch
from torch.distributions import Distribution
import pytorch_lightning as pl

from torchnf.abc import DensityTransform, TargetDistribution
import torchnf.metrics
from torchnf.utils.distribution import IterableDistribution
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


class BoltzmannGenerator(pl.LightningModule):
    r"""
    Latent Variable Model whose target distribution is a known functional.

    If the target distribution has a known functional form,

    .. math::

        \int \mathrm{d} x p(x) = \frac{1}{Z} \int \mathrm{d} x e^{-E(x)}

    then we can estimate the 'reverse' Kullbach-Leibler divergence
    between the model :math:`q(x)` and the target,

    .. math::

        D_{KL}(q \Vert p) = \int \mathrm{d} x q(x)
        \log \frac{q(x)}{p(x)}

    up to an unimportant normalisation due to :math:`\log Z`, using

    .. math::

        \hat{D}_{KL} = \mathrm{E}_{x \sim q} \left[
        -E(x) - \log q(x) \right]

    This serves as a loss function for 'reverse-KL training'.

    Furthermore, data generated from the model can be assigned an
    un-normalised statistical weight

    .. math::

        \log w(x) = -E(x) - \log q(x)

    which allows for (asymptotically) unbiased inference.
    """

    def __init__(self, flow: DensityTransform) -> None:
        super().__init__()
        self.flow = flow

    @property
    def prior(self) -> Distribution:
        try:
            return self._prior
        except AttributeError:
            raise AttributeError("prior has not been defined")

    @prior.setter
    def prior(self, new: Distribution) -> None:
        self._prior = new

    @property
    def target(self) -> TargetDistribution:
        try:
            return self._target
        except AttributeError:
            raise AttributeError("target has not been defined")

    @target.setter
    def target(self, new: TargetDistribution) -> None:
        self._target = new

    def setup(self, stage) -> None:
        """
        Sets up the model for training, validation, testing or prediction.

        Crucially, if a datamodule has been passed to the trainer, the model's
        `prior` and `target` attributes are set to point to those of the
        datamodule.

        If no datamodule is provided, the `prior` and `target` attributes
        must be set manually.
        """
        try:
            prior = self.trainer.datamodule.prior
        except AttributeError:
            pass
        else:
            self.prior = prior
        try:
            target = self.trainer.datamodule.target
        except AttributeError:
            pass
        else:
            self.target = target

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
        z, log_det_jacob = self.flow.inverse(x)
        log_prob_z = self.prior.log_prob(z)
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
        log_prob_z = self.prior.log_prob(z)
        x, log_det_jacob = self.flow.forward(z)
        log_prob_x = log_prob_z - log_det_jacob
        return x, log_prob_x

    def forward(self, z: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        r"""
        Decodes a batch of latent variables and computes the log weights.

        This decodes the latent variables, computes the log probability
        of the decoded data under the model and the target, and returns
        the decoded batch along with the logarithm of un-normalised
        statistical weights.

        The statistical weights are defined

        .. math::

            \log w(x) = \log p(x) - \log q(x)
        """
        x, log_prob_x = self.decode(z)
        log_prob_target = self.target.log_prob(x)
        log_stat_weight = log_prob_target - log_prob_x
        return x, log_stat_weight

    def reverse_kl_step(self, z: torch.Tensor) -> torch.Tensor:
        x, log_stat_weight = self(z)
        loss = log_stat_weight.mean().neg()
        return loss

    def training_step(self, batch: torch.Tensor, *_, **__) -> torch.Tensor:
        r"""
        Performs a single 'reverse' KL step.

        This decodes the latent variables, computes the log probability
        of the decoded data under the model and the target, and returns
        an estimate of the 'reverse' Kullbach-Leibler divergence, up to
        the unknown shift due to normalisations.

        The loss returned is defined by

        .. math::

            L(\{x\}) = \frac{1}{N} \sum_{\{x\}} \left[
            \log p(x) - \log q(x) \right]

        Args:
            z:
                A batch of latent variables drawn from the prior
                distribution

        Returns:
            The mean of the negative log-likelihood of the inputs,
            under the model

        .. note:: This relies on :meth:`forward`.
        """
        x, log_stat_weight = self(batch)
        loss = log_stat_weight.mean().neg()
        self.log("loss/train", loss, on_step=True)
        return loss

    def validation_step(
        self, batch: torch.Tensor, *_, **__
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Single validation step.
        """
        x, log_stat_weight = self(batch)
        loss = log_stat_weight.mean().neg()
        self.log("loss/validation", loss, on_step=False, on_epoch=True)
        return x, log_stat_weight

    def test_step(
        self, batch: torch.Tensor, *_, **__
    ) -> tuple[torch.Tensor, torch.Tensor]:
        return self(batch)

    def test_epoch_end(
        self, test_outputs: list[tuple[torch.Tensor, torch.Tensor]]
    ) -> None:
        x, log_stat_weight = tuple_concat(**test_outputs)
        loss = log_stat_weight.mean().neg()
        self.logger.log_hyperparams(
            dict(self.hparams) or {"_": -1}, {"loss/test": loss}
        )

    @torch.no_grad()
    @eval_mode
    def sample(
        self, batch_size: PositiveInt, batches: PositiveInt = 1
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Generate a weighted sample by sampling from the model.

        Essentially, this does

        .. code:: python

            for _ in range(batches):
                z = self.prior.sample([batch_size])
                x, log_prob_x = self.decode(z)
                log_prob_target = self.target.log_prob(x)
                log_stat_weight = log_prob_target - log_prob_x
                ...

        The returned tuple ``(x, log_stat_weight)`` contains the
        concatenation of all of the batches.

        .. note:: This calls :meth:`forward`.
        """
        out = []
        for _ in range(batches):
            z = self.prior.sample([batch_size])
            out.append(self(z))
        return tuple_concat(*out)

    def __iter__(self) -> Iterator:
        return self.generator()

    @torch.no_grad()
    @eval_mode
    def generator(self) -> Iterator[tuple[torch.Tensor, torch.Tensor]]:
        """
        Returns an infinite iterator over states drawn from the model.
        """
        batch = zip(*self([self.batch_size]))
        while True:
            try:
                yield next(batch)
            except StopIteration:
                batch = zip(*self([self.batch_size]))
