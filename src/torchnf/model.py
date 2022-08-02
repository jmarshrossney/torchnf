from jsonargparse.typing import PositiveInt
import torch

from torchnf.utils.flow import eval_mode


class FlowBasedModel(torch.nn.Module):
    def flow_forward(
        self, inputs: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Forward transformation of the Normalizing Flow.

        By default, this returns ``self.flow(inputs)``.

        .. admonition:: Requires Attribute

            ``flow`` - A :class:`torchnf.layers.Flow` or another ``Module``
            that implements a ``forward`` method.
        """
        return self.flow(inputs)

    def flow_inverse(
        self, inputs: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Inverse transformation of the Normalizing Flow.

        By default, this returns ``self.flow.inverse(inputs)``.

        .. admonition:: Requires Attribute

            ``flow`` - A :class:`torchnf.layers.Flow` or another ``Module``
            that implements an ``inverse`` method.
        """
        return self.flow.inverse(inputs)

    def encode(self, data: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Encoding transformation.

        By default, this just passes the inputs to :meth:`flow_forward`.
        """
        return self.flow_forward(data)

    def decode(self, noise: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Decoding transformation.

        By default, this just passes the inputs to :meth:`flow_inverse`.
        """
        return self.flow_inverse(noise)

    def prior_sample(self, sample_size: PositiveInt) -> torch.Tensor:
        """
        Samples from the prior (latent) distribution.

        By default, this calls ``self.prior.sample([sample_size])``.

        .. admonition:: Requires Attribute

            ``prior`` - A :py:class:`torch.distributions.Distribution`,
            or another object that implements a ``log_prob`` method.
        """
        return self.prior.sample([sample_size])

    def prior_log_prob(self, sample: torch.Tensor) -> torch.Tensor:
        """
        Returns the log-probability density of the prior distribution.

        By default, this returns ``self.prior.log_prob(sample)``.

        .. admonition:: Requires Attribute

            ``prior`` - A :py:class:`torch.distributions.Distribution`,
            or another object that implements a ``log_prob`` method.
        """
        return self.prior.log_prob(sample)

    def target_log_prob(self, sample: torch.Tensor) -> torch.Tensor:
        """
        Returns the log-probability density of the target distribution.

        By default, this returns ``self.target.log_prob(sample)``.

        .. admonition:: Requires Attribute

            ``target`` - A :py:class`torch.distributions.Distribution`,
            or another object that implements a ``log_prob`` method.
        """
        return self.target.log_prob(sample)

    def inference_step(
        self, data: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        r"""
        Encodes the data and computes the density under the model.

        *Encoding step:*

        First, :meth:`encode` is called to transform the data :math:`x`
        to :math:`z = \mathrm{encode}(x)`, and compute the log Jacobian
        determinant :math:`\log \lvert \partial z / \partial x \rvert`
        of the transformation.

        *Density estimation:*

        Computes an estimate of the log probability density of the input data.

        This estimate is the log probability density of the data under the
        model, :math:`q(x)`. That is, the push-forward of :math:`q(z)` by the
        decoding transformation. Since the encoder/decoder is invertible, this
        can be calculated as follows:

        .. math::

            \log q(x) = \log q(z) + \log
            \left\lvert \frac{\partial z}{\partial x} \right\rvert

        *Log likelihood:*

        The log-likelihood of the input data with respect to the model
        is given by

        .. math::

            \log \ell(\{x\}) = \frac{1}{n} \sum_{\{x\}}
            - \log q(x) \, .

        Args:
            data: the input data, :math:`x`

        Returns:
            dict[str, torch.Tensor]:

            * **sample** -- the encoded sample, :math:`z`
            * **log_prob** -- the log probability density, :math:`\log q(x)`
            * **loss** -- the negative log-likelihood, :math:`\log \ell(\{x\})`
        """
        x = data
        z, log_dzdx = self.encode(x)
        log_q_z = self.prior_log_prob(z)
        log_q_x = log_q_z + log_dzdx
        log_likelihood = log_q_x.negative().mean(dim=0, keepdim=True)
        outputs = dict(sample=z, log_prob=log_q_x, loss=log_likelihood)
        return outputs

    def sampling_step(self, noise: torch.Tensor) -> dict[str, torch.Tensor]:
        r"""
        Decodes the latents and computes the density under the model.

        *Decoding step:*

        First, :meth:`decode` is called to transform the latent variables
        :math:`z` to :math:`x = \mathrm{decode}(z)`, and compute the log
        Jacobian determinant of the transformation,
        :math:`\log\lvert\partial x/\partial z\rvert`

        *Push-forward:*

        The probability density at point :math:`x = \mathrm{decode}(z)`
        is the push-forward of :math:`q(z)` by the decoding transform:

        .. math::

            \log q(x) = \log q(z)
            - \log \left\lvert \frac{\partial x}{\partial z} \right\rvert


        Args:
            noise: A batch of variables :math:`z` drawn from the latent
            (prior) distribution

        Returns:
            dict[str, torch.Tensor]:

            * **sample** -- the decoded sample, :math:`x`
            * **log_prob** -- the log probability density, :math:`\log q(x)`
        """
        z = noise
        log_q_z = self.prior_log_prob(z)
        x, log_dxdz = self.decode(z)
        log_q_x = log_q_z - log_dxdz
        return dict(sample=x, log_prob=log_q_x)

    def importance_sampling_step(
        self, noise: torch.Tensor
    ) -> dict[str, torch.Tensor]:
        r"""
        Decodes the latents and computes importance weights.

        *Sampling:*

        See documentation for :meth:`sampling_step`.

        *Importance weights:*

        The importance weights allow us to compute expectation values
        for the target distribution by sampling from the model.

        .. math::

            \log w(x) := \log \frac{p(x)}{q(x)}
            = \log p(x) - \log q(x)

        Args:
            noise: A batch of variables :math:`z` drawn from the latent
            (prior) distribution

        Returns:
            dict[str, torch.Tensor]:

            * **sample** -- the encoded sample, :math:`z`
            * **log_prob** -- the log probability density, :math:`\log q(x)`
            * **log_weight** -- the importance weights, :math:`\log w(x)`
            * **loss** -- the KL divergence estimate
        """
        z = noise
        outputs = self.sampling_step(z)
        log_p = self.log_p(outputs["data"])
        log_w = log_p - outputs["log_q"]
        kl_div = log_w.negative().mean(dim=0, keepdim=True)
        outputs |= dict(log_weight=log_w, loss=kl_div)
        return outputs

    @torch.no_grad()
    @eval_mode
    def sample(
        self,
        sample_size: PositiveInt,
    ) -> torch.Tensor:
        """
        Generates synthetic data by sampling from the model.

        The sample is generated by first sampling a set of latent variables
        using :meth:`prior_sample`, and then calling :meth:`sampling_step`.

        Parameters:
            sample_size: Size of the sample.

        Returns:
            torch.Tensor: A sample drawn from the model.
        """
        z = self.prior_sample(sample_size)
        sampling_outputs = self.sampling_step(z)
        return sampling_outputs["sample"]

    @torch.no_grad()
    @eval_mode
    def weighted_sample(
        self,
        sample_size: PositiveInt,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Generates synthetic data along with importance weights.

        The sample is generated by first sampling a set of latent variables
        using :meth:`prior_sample`, and then calling
        :meth:`importance_sampling_step`.

        Parameters:
            sample_size: Size of the sample.

        Returns:
            tuple[torch.Tensor, torch.Tensor]:

            A sample drawn from the model, along with its importance
            weight with respect to the target distribution defined by
            :meth:`target_log_prob`.
        """
        z = self.prior_sample(sample_size)
        sampling_outputs = self.importance_sampling_step(z)
        return sampling_outputs["sample"], sampling_outputs["log_weight"]
