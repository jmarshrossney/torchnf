r"""
Module containing parametrised bijective transformations and their inverses.

Transformation functions take an input tensor and one or more tensors that
parametrise the transformation, and return the transformed inputs along
with the logarithm of the Jacobian determinant of the transformation. They
should be called as

.. code-block:: python

    output, log_det_jacob = transform(input, params, context)

In maths, the transformations do

.. math::

    x, \{\lambda\} \longrightarrow y = f(x ; \{\lambda\}),
    \log \left\lvert\frac{\partial y}{\partial x} \right\rvert

The inverse of these transformations are

.. math::

    y, \{\lambda\} \longrightarrow x = f^{-1}(y ; \{\lambda\}),
    \log \left\lvert\frac{\partial x}{\partial y} \right\rvert

Thanks to the inverse function theorem, the log-Jacobian-determinants
for the forward and inverse transformations are related by

.. math::

    \log \left\lvert \frac{\partial x}{\partial y} \right\rvert
    = -\log \left\lvert \frac{\partial y}{\partial x} \right\rvert

"""

import logging
import math
from typing import ClassVar

from jsonargparse.typing import PositiveInt
import torch
import torch.nn.functional as F

from torchnf.abc import Transformer
from torchnf.utils.tensor import stacked_nan_to_num, sum_except_batch

log = logging.getLogger(__name__)

INF = float("inf")
REALS = (-INF, INF)
PI = math.pi

__all__ = [
    "Translation",
    "Rescaling",
    "AffineTransform",
    "RQSplineTransform",
    "RQSplineTransformIntervalDomain",
    "RQSplineTransformCircularDomain",
]


class _Transformer(Transformer):
    """
    Base class that should be inherited by all transformers.

    Derived classes should override the following methods
    and properties:
    - :code:`n_params()`
    - :code:`identity_params()`
    - :code:`_forward()`
    - :code:`_inverse()`

    .. attention:: Do **not** override :meth:`forward()` or :meth:`inverse()`!

    :meta public:
    """

    domain: ClassVar[tuple[float, float]]
    codomain: ClassVar[tuple[float, float]]

    @property
    def n_params(self) -> PositiveInt:
        """
        Number of parameters specifying how to transform a single element.
        """
        return NotImplemented

    @property
    def identity_params(self) -> torch.Tensor:
        """
        The parameters which result in an identity transformation.
        """
        return NotImplemented

    def _forward(
        self, x: torch.Tensor, params: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass of the transformer.

        Args:
            x
                The input tensor, shape `(batch_size, *data_shape)`
            params
                Parameters upon which the transformation is conditioned,
                shape `(batch_size, n_params, *data_shape)`

        Returns:
            Tuple of tensors containing (1) the transformed inputs, and
            (2) the logarithm of the Jacobian determinant

        :meta public:
        """
        raise NotImplementedError

    def _inverse(
        self, y: torch.Tensor, params: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Inverse of :meth:`_forward`.

        :meta public:
        """
        raise NotImplementedError

    def forward(
        self, x: torch.Tensor, params: torch.Tensor, context: dict = {}
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Wrapper around the forward pass of the transformer.

        This does the following:

        .. code-block::

            self.context = context
            params = stacked_nan_to_num(
                params, self.identity_params, dim=1
            )
            return self._forward(x, params)

        """
        self.context = context
        params = stacked_nan_to_num(
            params,
            self.identity_params.to(params.device),
            dim=1,
        )
        return self._forward(x, params)

    def inverse(
        self, y: torch.Tensor, params: torch.Tensor, context: dict = {}
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Inverse of :meth:`forward`.
        """
        self.context = context
        params = stacked_nan_to_num(
            params,
            self.identity_params.to(params.device),
            dim=1,
        )
        return self._inverse(y, params)

    def __call__(
        self, x: torch.Tensor, params: torch.Tensor, context: dict = {}
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Alias for :meth:`forward`.
        """
        return self.forward(x, params, context)


class Translation(_Transformer):
    r"""Performs a pointwise translation of the input tensor.

    The forward and inverse transformations are, respectively

    .. math::

        x \mapsto y = x + t \, ,

        y \mapsto x = y - t \, .

    There is no change in the volume element, i.e.

    .. math::

        \log \left\lvert \frac{\partial y}{\partial x} \right\rvert
        \log \left\lvert \frac{\partial x}{\partial y} \right\rvert
        = 0
    """
    domain = REALS
    codomain = REALS

    @property
    def n_params(self) -> PositiveInt:
        return 1

    @property
    def identity_params(self) -> torch.Tensor:
        return torch.tensor([0])

    def _forward(
        self, x: torch.Tensor, params: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        shift = params.view_as(x)
        y = x.add(shift)
        ldj = torch.zeros(x.shape[0], device=y.device)
        return y, ldj

    def _inverse(
        self, y: torch.Tensor, params: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Performs a pointwise translation of the input tensor."""
        shift = params.view_as(y)
        x = y.sub(shift)
        ldj = torch.zeros(y.shape[0], device=x.device)
        return x, ldj


class Rescaling(_Transformer):
    r"""Performs a pointwise rescaling of the input tensor.

    The forward and inverse transformations are, respectively,

    .. math::

        x \mapsto y = x \odot e^{-s} \, ,

        y \mapsto x = y \odot e^{s} \, .

    The logarithm of the Jacobian determinant is

    .. math::

        \log \left\lvert \frac{\partial y}{\partial x} \right\rvert
        = \sum_i -s_i

    where :math:`i` runs over the degrees of freedom being transformed.
    """
    domain = REALS
    codomain = REALS

    @property
    def n_params(self) -> PositiveInt:
        return 1

    @property
    def identity_params(self) -> torch.Tensor:
        return torch.tensor([0])

    def _forward(
        self, x: torch.Tensor, params: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        log_scale = params.view_as(x)
        y = x.mul(log_scale.neg().exp())
        ldj = sum_except_batch(log_scale.neg())
        return y, ldj

    def _inverse(
        self, y: torch.Tensor, params: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        log_scale = params.view_as(y)
        x = y.mul(log_scale.exp())
        ldj = sum_except_batch(log_scale)
        return x, ldj


class AffineTransform(_Transformer):
    r"""Performs a pointwise affine transformation of the input tensor.

    The forward and inverse transformations are, respectively,

    .. math::

        x \mapsto y = x \odot e^{-s} + t

        y \mapsto x = (y - t) \odot e^{s}

    .. math::

        \log \left\lvert \frac{\partial y}{\partial x} \right\rvert
        = \sum_i -s_i

    where :math:`i` runs over the degrees of freedom being transformed.

    The :code:`params` argument to :meth:`forward` and :meth:`inverse`
    should be a tensor with :math:`s` and :math:`t` stacked, in that
    order, on the `1` dimension. The parameters will be split as

    .. code-block:: python

        s = params[:, 0]
        t = parmas[:, 1]
    """
    domain = REALS
    codomain = REALS

    @property
    def n_params(self) -> PositiveInt:
        return 2

    @property
    def identity_params(self) -> torch.Tensor:
        return torch.tensor([0, 0])

    def _forward(
        self, x: torch.Tensor, params: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        log_scale, shift = [p.view_as(x) for p in params.split(1, dim=1)]
        y = x.mul(log_scale.neg().exp()).add(shift)
        ldj = sum_except_batch(log_scale.neg())
        return y, ldj

    def _inverse(
        self, y: torch.Tensor, params: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        log_scale, shift = [p.view_as(y) for p in params.split(1, dim=1)]
        x = y.sub(shift).mul(log_scale.exp())
        ldj = sum_except_batch(log_scale)
        return x, ldj


class RQSplineTransform(_Transformer):
    r"""
    Pointwise rational quadratic spline transformation.
    """
    domain = REALS
    codomain = REALS

    def __init__(
        self,
        n_segments: int,
        interval: tuple[float],
    ) -> None:
        self._n_segments = n_segments
        self._interval = interval

    @property
    def _n_knots(self) -> PositiveInt:
        return self._n_segments - 1

    @property
    def n_params(self) -> PositiveInt:
        return 2 * self._n_segments + self._n_knots

    @property
    def identity_params(self) -> torch.Tensor:
        return torch.cat(
            (
                torch.full(
                    size=(2 * self._n_segments,),
                    fill_value=1 / self._n_segments,
                ),
                (torch.ones(self._n_knots).exp() - 1).log(),
            ),
            dim=0,
        )

    def handle_inputs_outside_interval(
        self, outside_interval_mask: torch.BoolTensor
    ) -> None:
        """
        Handle inputs falling outside the spline interval.

        Unless overridden, this method submits a :code:`log.debug` logging
        event if more than 1/1000 inputs fall outside the spline interval.

        Args:
            outside_interval_mask
                BoolTensor of the same shape as the layer input where the
                :code:`True` elements correspond to inputs which fell outside
                the spline interval.

        """
        if outside_interval_mask.sum() > outside_interval_mask.numel() / 1000:
            log.debug(
                "More than 1/1000 inputs fell outside the spline interval"
            )

    @staticmethod
    def _pad_derivs(derivs: torch.Tensor) -> torch.Tensor:
        return F.pad(derivs, (1, 1), "constant", 1)

    def build_spline(
        self,
        widths: torch.Tensor,
        heights: torch.Tensor,
        derivs: torch.Tensor,
    ) -> tuple[torch.Tensor]:
        """Builds a rational quadratic spline function.

        This uses the parametrisation introduced by Gregory and Delbourgo
        (1983)

        Args:
            widths
                Un-normalised segment sizes in the domain
            heights
                Un-normalised segment sizes in the codomain
            derivs
                Unconstrained derivatives at the knots

        Returns:
            Tuple of tensors containing (1) Normalised segment sizes in
            the domain, (2) Normalised segment sizes in the codomain, (3)
            Constrained derivatives at the knots, (4) Coordinates of the
            knots in the domain, (5) Coordinates of the knots in the codomain


        References:
            Gregory, J. A. & Delbourgo, R. C2 Rational \
Quadratic Spline Interpolation to Monotonic Data, IMA Journal of \
Numerical Analysis, 1983, 3, 141-152

        """
        # Normalise the widths and heights to the interval
        interval_size = self._interval[1] - self._interval[0]
        widths = F.softmax(widths, dim=-1).mul(interval_size)
        heights = F.softmax(heights, dim=-1).mul(interval_size)

        # Let the derivatives be positive definite
        derivs = F.softplus(derivs)
        derivs = self.pad_derivs(derivs)

        # Just a convenient way to ensure it's on the correct device
        zeros = torch.zeros_like(widths).sum(dim=-1, keepdim=True)

        knots_xcoords = torch.cat(
            (
                zeros,
                torch.cumsum(widths, dim=-1),
            ),
            dim=-1,
        ).add(self._interval[0])
        knots_ycoords = torch.cat(
            (
                zeros,
                torch.cumsum(heights, dim=-1),
            ),
            dim=-1,
        ).add(self._interval[0])

        return widths, heights, derivs, knots_xcoords, knots_ycoords

    def _forward(
        self, x: torch.Tensor, params: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        # TODO: transpose 1 -> -1 using movedim or tranpose
        # Why: torch.nn.functional.pad, dims start from -1
        # Also, torch.searchsorted requires bins dim = -q
        # One possibility: flatten dim 2, and restore structure at end
        widths, heights, derivs = params.split(
            (self._n_segments, self._n_segments, self._n_knots),
            dim=1,
        )
        (
            widths,
            heights,
            derivs,
            knots_xcoords,
            knots_ycoords,
        ) = self.build_spline(widths, heights, derivs)

        outside_interval_mask = torch.logical_or(
            x < knots_xcoords[..., 0],
            x > knots_xcoords[..., -1],
        )
        self.handle_inputs_outside_interval(outside_interval_mask)

        segment_idx = torch.searchsorted(knots_xcoords, x) - 1
        segment_idx.clamp_(0, widths.shape[-1])

        # Get parameters of the segments that x falls in
        w = torch.gather(widths, -1, segment_idx)
        h = torch.gather(heights, -1, segment_idx)
        d0 = torch.gather(derivs, -1, segment_idx)
        d1 = torch.gather(derivs, -1, segment_idx + 1)
        x0 = torch.gather(knots_xcoords, -1, segment_idx)
        y0 = torch.gather(knots_ycoords, -1, segment_idx)

        # NOTE: these will fail because some x are outside interval
        # Hence, alpha.clamp_(0, 1) will silently hide bugs
        # TODO: thnk of a smart and cheap way to check
        # eps = 1e-5
        # assert torch.all(x > x0 - eps)
        # assert torch.all(x < x0 + w + eps)

        s = h / w
        alpha = (x - x0) / w
        alpha.clamp_(0, 1)

        denominator_recip = torch.reciprocal(
            s + (d1 + d0 - 2 * s) * alpha * (1 - alpha)
        )
        beta = (
            s * alpha.pow(2) + d0 * alpha * (1 - alpha)
        ) * denominator_recip
        y = y0 + h * beta

        gradient = (
            s.pow(2)
            * (
                d1 * alpha.pow(2)
                + 2 * s * alpha * (1 - alpha)
                + d0 * (1 - alpha).pow(2)
            )
            * denominator_recip.pow(2)
        )
        assert torch.all(gradient > 0)

        y[outside_interval_mask] = x[outside_interval_mask]

        ldj = sum_except_batch(gradient.log())

        return y, ldj

    def _inverse(
        self, y: torch.Tensor, params: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        widths, heights, derivs = params.split(
            (self._n_segments, self._n_segments, self._n_knots),
            dim=self._params_dim,
        )
        (
            widths,
            heights,
            derivs,
            knots_xcoords,
            knots_ycoords,
        ) = self.build_spline(widths, heights, derivs)

        outside_interval_mask = torch.logical_or(
            y < knots_ycoords[..., 0],
            y > knots_ycoords[..., -1],
        )
        if outside_interval_mask.sum() > 0.001 * y.numel():
            log.debug(
                "More than 1/1000 inputs fell outside the spline interval"
            )

        segment_idx = torch.searchsorted(knots_ycoords, y) - 1
        segment_idx.clamp_(0, widths.shape[-1])

        # Get parameters of the segments that x falls in
        w = torch.gather(widths, -1, segment_idx)
        h = torch.gather(heights, -1, segment_idx)
        d0 = torch.gather(derivs, -1, segment_idx)
        d1 = torch.gather(derivs, -1, segment_idx + 1)
        x0 = torch.gather(knots_xcoords, -1, segment_idx)
        y0 = torch.gather(knots_ycoords, -1, segment_idx)

        # eps = 1e-5
        # assert torch.all(y > y0 - eps)
        # assert torch.all(y < y0 + h + eps)

        s = h / w
        beta = (y - y0) / w
        beta.clamp_(0, 1)

        b = d0 - (d1 + d0 - 2 * s) * beta
        a = s - b
        c = -s * beta
        alpha = -2 * c * torch.reciprocal(b + (b.pow(2) - 4 * a * c).sqrt())
        x = x0 + w * alpha

        denominator_recip = torch.reciprocal(
            s + (d1 + d0 - 2 * s) * alpha * (1 - alpha)
        )
        gradient_fwd = (
            s.pow(2)
            * (
                d1 * alpha.pow(2)
                + 2 * s * alpha * (1 - alpha)
                + d0 * (1 - alpha).pow(2)
            )
            * denominator_recip.pow(2)
        )

        ldj = sum_except_batch(gradient_fwd.log().neg())

        return x, ldj


class RQSplineTransformIntervalDomain(RQSplineTransform):
    def __init__(
        self,
        n_segments: int,
        interval: tuple[float],
    ):
        super().__init__(n_segments, interval)
        self.domain = interval
        self.codomain = interval

    @property
    def _n_knots(self) -> PositiveInt:
        return self._n_segments + 1

    @staticmethod
    def _pad_derivs(derivs: torch.Tensor) -> torch.Tensor:
        return derivs


class RQSplineTransformCircularDomain(RQSplineTransform):
    def __init__(
        self,
        n_segments: PositiveInt,
        interval: tuple[float],
    ):
        super().__init__(n_segments, interval)
        assert math.isclose(interval[1] - interval[0], 2 * PI)
        self.domain = interval
        self.codomain = interval

    @property
    def _n_knots(self) -> PositiveInt:
        return self._n_segments

    @staticmethod
    def _pad_derivs(derivs: torch.Tensor) -> torch.Tensor:
        return F.pad(derivs, (0, 1), "circular")
