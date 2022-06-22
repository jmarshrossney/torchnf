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

from jsonargparse.typing import PositiveInt
import torch
import pytorch_lightning as pl

from torchnf.distributions import Target, Prior
import torchnf.flow
import torchnf.metrics


@dataclasses.dataclass
class OptimizerConfig:
    """
    Dataclass representing a single optimizer with optional lr scheduler.

    This class provides, via the :meth:`add_to`` method, an alternative
    way to configure an optimizer and lr scheduler, as opposed to
    defining ``configure_optimizers`` in the ``LightningModule`` itself.

    Args:
        optimizer:
            The optimizer class
        optimizer_init:
            Keyword args to instantiate optimizer
        scheduler:
            The lr scheduler class
        scheduler_init:
            Keyword args to instantiate scheduelr
        submodule:
            Optionally specify a submodule whose ``parameters()``
            will be passed to the optimizer.

    Example:

        >>> optimizer_config = OptimizerConfig(
                "Adam",
                {"lr": 0.001},
                "CosineAnnealingLR",
                {"T_max": 1000},
            )
        >>> # MyModel does not override configure_optimizers
        >>> model = MyModel(...)
        >>> optimizer_config.add_to(model)
    """

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
        """
        Simple function used to override ``configure_optimizers``.
        """
        if scheduler is None:
            return optimizer
        return [optimizer], [scheduler]

    def add_to(self, model: pl.LightningModule) -> None:
        """
        Add the optimizer and scheduler to an existing ``LightningModule``.
        """
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


class FlowBasedModel(pl.LightningModule):
    """
    Base LightningModule for Normalizing Flows.

    Args:
        flow:
            A Normalizing Flow. If this is not provided then
            :meth:`flow_forward` and :meth:`flow_inverse` should be
            overridden to implement the flow.
    """

    def __init__(self, flow: torchnf.flow.Flow) -> None:
        super().__init__()
        self.flow = flow
        self.configure_metrics()

    def flow_forward(
        self, x: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass of the Normalizing Flow.

        Unless overridden, this simply returns ``self.flow(x)``.
        """
        return self.flow(x)

    def flow_inverse(
        self, y: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Inverse pass of the Normalizing Flow.

        Unless overridden, this simply returns ``self.flow.inverse(y)``.
        """
        return self.flow.inverse(y)

    def configure_metrics(self) -> None:
        """
        Instantiate metrics (called in constructor).
        """
        ...


class BijectiveAutoEncoder(FlowBasedModel):
    """
    Latent variables model based on a Normalizing Flow.

    Args:
        flow:
            A Normalizing Flow
        prior:
            The distribution from which latent variables are drawn.
        forward_is_encode:
            If True, the ``forward`` method of the Normalizing Flow
            performs the encoding step, and the ``inverse`` method
            performs the decoding; if False, the converse
    """

    def __init__(
        self,
        flow: torchnf.flow.Flow,
        prior: Prior,
        *,
        forward_is_encode: bool = True,
    ) -> None:
        super().__init__(flow)
        self.prior = prior

        self._encode, self._decode = (
            (self.flow_forward, self.flow_inverse)
            if forward_is_encode
            else (self.flow_inverse, self.flow_forward)
        )

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        r"""
        Encodes the input data and computes the log likelihood.

        The log likelihood of the point :math:`x` under the model is

        .. math::

            \log \ell(x) = \log q(z)
            + \log \left\lvert \frac{\partial z}{\partial x} \right\rvert

        where :math:`z` is the corresponding point in latent space,
        :math:`q` is the prior distribution, and the Jacobian is that
        of the encoding transformation.

        Args:
            x:
                A batch of data drawn from the target distribution

        Returns:
            Tuple containing the encoded data and the log likelihood
            under the model
        """
        z, log_det_jacob = self._encode(x)
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
        :math:`q` is the prior distribution, and the Jacobian is that
        of the decoding transformation.

        Args:
            z:
                A batch of latent variables drawn from the prior
                distribution

        Returns:
            Tuple containing the decoded data and the log likelihood
            under the model
        """
        log_prob_z = self.prior.log_prob(z)
        x, log_det_jacob = self._decode(z)
        log_prob_x = log_prob_z - log_det_jacob
        return x, log_prob_x

    def training_step(
        self, batch: list[torch.Tensor], batch_idx: int
    ) -> torch.Tensor:
        r"""
        Performs a single training step, returning the loss.

        The loss returned is the mean of the negative log-likelihood
        of the inputs, under the model, i.e.

        .. math::

            L(\{x\}) = \frac{1}{N} \sum_{\{x\}} -\log q(x)
        """
        (x,) = batch
        z, log_prob_x = self.encode(x)
        loss = log_prob_x.mean().neg()  # forward KL
        self.logger.experiment.add_scalars(
            "loss", {"train": loss}, self.global_step
        )
        return loss

    def validation_step(
        self, batch: list[torch.Tensor], batch_idx: int
    ) -> torch.Tensor:
        """
        Performs a single validation step, returning the encoded data.
        """
        (x,) = batch
        z, log_prob_x = self.encode(x)
        loss = log_prob_x.mean().neg()  # forward KL
        self.logger.experiment.add_scalars(
            "loss", {"validation": loss}, self.global_step
        )
        return z

    @torch.no_grad()
    def sample(
        self, batch_size: PositiveInt, batches: PositiveInt = 1
    ) -> torch.Tensor:
        """
        Generate synthetic data by sampling from the model.
        """
        z = self.prior.sample([batch_size])
        x, _ = self._decode(z)
        return x


class BoltzmannGenerator(BijectiveAutoEncoder):
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
        flow: torchnf.flow.Flow,
        prior: Prior,
        target: Target,
    ) -> None:
        super().__init__(flow, prior, forward_is_encode=False)
        self.target = target

    def configure_metrics(self) -> None:
        self.metrics = torchnf.metrics.LogStatWeightMetricCollection()

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

    def configure_training(
        self,
        batch_size: PositiveInt,
        epoch_length: PositiveInt,
        val_batch_size: Optional[PositiveInt] = None,
        val_epoch_length: Optional[PositiveInt] = None,
        test_batch_size: Optional[PositiveInt] = None,
        test_epoch_length: Optional[PositiveInt] = None,
    ) -> None:
        """
        Sets the batch sizes and epoch lengths for reverse KL training.

        Before training in reverse KL mode (i.e. decoding latent
        variables and evaluating log p - log q), this method must be
        executed in order to set the batch sizes and epoch lengths.
        """
        self.batch_size = batch_size
        self.epoch_length = epoch_length
        self.val_batch_size = val_batch_size or batch_size
        self.val_epoch_length = val_epoch_length or epoch_length
        self.test_batch_size = test_batch_size or batch_size
        self.test_epoch_length = test_epoch_length or epoch_length

    def _prior_as_dataloader(
        self, batch_size: PositiveInt, epoch_length: PositiveInt
    ) -> torchnf.distributions.IterablePrior:
        """
        Returns an iterable version of the prior distribution.
        """
        if not hasattr(self, "batch_size"):
            raise Exception("First, run 'configure_training'")
        return torchnf.distributions.IterablePrior(
            self.prior,
            batch_size,
            epoch_length,
        )

    def train_dataloader(self) -> torchnf.distributions.IterablePrior:
        return self._prior_as_dataloader(self.batch_size, self.epoch_length)

    def val_dataloader(self) -> torchnf.distributions.IterablePrior:
        return self._prior_as_dataloader(
            self.val_batch_size, self.val_epoch_length
        )

    def test_dataloader(self) -> torchnf.distributions.IterablePrior:
        return self._prior_as_dataloader(
            self.test_batch_size, self.test_epoch_length
        )

    def predict_dataloader(self) -> torchnf.distributions.IterablePrior:
        return self._prior_as_dataloader(
            self.pred_batch_size, self.pred_epoch_length
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

        Unless overridden, this simply does the following:

        .. code-block:: python

            y, log_stat_weights = self(batch)
            self.metrics.update(log_stat_weights)

        """
        y, log_stat_weights = self(batch)
        self.metrics.update(log_stat_weights)

    def validation_epoch_end(self, val_outputs):
        """
        Compute and log metrics at the end of an epoch.
        """
        metrics = self.metrics.compute()
        self.log("Validation", metrics)
        self.metrics.reset()

    def test_step(self, batch: torch.Tensor, batch_idx: int) -> None:
        """
        Single test step.

        Unless overridden, this simply calls :meth:`validation_step`.
        """
        return self.validation_step(batch, batch_idx)

    def test_epoch_end(self, val_outputs):
        """
        Compute and log metrics at the end of an epoch.
        """
        metrics = self.metrics.compute()
        self.log("Test", metrics)
        self.metrics.reset()

    @torch.no_grad()
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
