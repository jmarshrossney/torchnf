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

    def __init__(
        self,
        flow: DensityTransform,
        prior: Distribution,
        target: TargetDistribution,
        *,
        batch_size: PositiveInt,
        val_batch_size: Optional[PositiveInt] = None,
        val_batches: PositiveInt = 1,
        test_batch_size: Optional[PositiveInt] = None,
        test_batches: PositiveInt = 1,
        epoch_length: Optional[PositiveInt] = None,
    ) -> None:
        super().__init__(flow, prior, forward_is_encode=False)
        self.target = target

        self.batch_size = batch_size
        self.val_batch_size = val_batch_size or batch_size
        self.val_batches = val_batches
        self.test_batch_size = test_batch_size or batch_size
        self.test_batches = test_batches
        self.epoch_length = epoch_length

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

    def training_step_rev_kl(self, z: torch.Tensor) -> torch.Tensor:
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
        x, log_stat_weight = self(z)
        loss = log_stat_weight.mean().neg()
        return loss

    def _prior_as_dataloader(
        self, batch_size: PositiveInt, epoch_length: Union[PositiveInt, None]
    ) -> IterableDistribution:
        """
        Returns an iterable version of the prior distribution.
        """
        if not hasattr(self, "batch_size"):
            raise Exception("First, run 'configure_training'")
        return IterableDistribution(
            self.prior,
            batch_size,
            epoch_length,
        )

    def train_dataloader(self) -> IterableDistribution:
        """
        An iterable version of the prior distribution.
        """
        return self._prior_as_dataloader(self.batch_size, self.epoch_length)

    def val_dataloader(self) -> IterableDistribution:
        """
        An iterable version of the prior distribution.
        """
        return self._prior_as_dataloader(self.val_batch_size, self.val_batches)

    def test_dataloader(self) -> IterableDistribution:
        """
        An iterable version of the prior distribution.
        """
        return self._prior_as_dataloader(
            self.test_batch_size, self.test_batches
        )

    def training_step(
        self, batch: torch.Tensor, batch_idx: int
    ) -> torch.Tensor:
        """
        Single training step.

        Unless overridden, this just calls :meth:`training_step_rev_kl`.
        """
        # TODO: flag to switch to forward KL training?
        loss = self.training_step_rev_kl(batch)
        return loss

    def validation_step(self, batch: torch.Tensor, batch_idx: int) -> None:
        """
        Single validation step.

        Unless overridden, this just calls :meth:`training_step_rev_kl`.
        """
        loss = self.training_step_rev_kl(batch)
        self.log("Validation/loss", loss, on_step=False, on_epoch=True)
        return loss

    @torch.no_grad()
    @eval_mode
    def weighted_sample(
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
        return torchnf.utils.tuple_concat(*out)

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

    def setup(self, stage) -> None:
        """
        Sets up the model for training, validation, testing or prediction.

        Crucially, if a datamodule has been passed to the trainer, the model's
        `prior` attribute is set to point to that of the datamodule.

        If no datamodule is provided, the `prior` attribute must be set
        manually.
        """
        # stage == "fit", "validate", "test", "predict"
        prior = None
        try:
            prior = self.prior
        except AttributeError:
            pass

        if self.trainer.datamodule is not None:
            try:
                prior = self.trainer.datamodule.prior
            except AttributeError:
                pass

        if prior is None:
            raise Exception("no prior defined")  # todo

        self.prior = prior
