r"""
Module containing parametrised bijective transformations and their inverses.

Transformation functions take an input tensor and one or more tensors that
parametrise the transformation, and return the transformed inputs along
with the logarithm of the Jacobian determinant of the transformation. They
should be called as

    output, log_det_jacob = transform(input, param1, param2, ...)

In maths, the transformations do

.. math::

    x, \{\lambda\} \longrightarrow f(x ; \{\lambda\}),
    \log \left\lvert\frac{\partial f(x ; \{\lambda\})}{\partial x} \right\rvert

Note that the log-det-Jacobian is that of the *forward* transformation.
"""

import logging
import math
from typing import ClassVar, NamedTuple

import torch
import torch.nn.functional as F

import torchnf.utils

log = logging.getLogger(__name__)

INF = float("inf")
REALS = (-INF, INF)
PI = math.pi


class Transformer:
    domain: ClassVar[tuple[float, float]]
    codomain: ClassVar[tuple[float, float]]

    @property
    def n_params(self) -> int:
        return NotImplemented

    @property
    def identity_params(self) -> list[float]:
        return NotImplemented

    def _nan_to_identity(
        self, params: torch.Tensor, data_shape: torch.Size
    ) -> torch.Tensor:
        return params.nan_to_num(0).add(
            torchnf.utils.expand_elements(
                torch.Tensor(self.identity_params, device=params.device),
                shape=data_shape,
                stack_dim=0,
            ).mul(params.isnan())
        )

    def _forward(
        self, x: torch.Tensor, params: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        raise NotImplementedError

    def _inverse(
        self, y: torch.Tensor, params: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        raise NotImplementedError

    def forward(
        self, x: torch.Tensor, params: torch.Tensor, context: dict = {}
    ) -> tuple[torch.Tensor, torch.Tensor]:
        self.context = context
        return self._forward(x, self._nan_to_identity(params, x.shape[1:]))

    def inverse(
        self, y: torch.Tensor, params: torch.Tensor, context: dict = {}
    ) -> tuple[torch.Tensor, torch.Tensor]:
        self.context = context
        return self._inverse(y, self._nan_to_identity(params, y.shape[1:]))

    def __call__(
        self, x: torch.Tensor, params: torch.Tensor, context: dict = {}
    ) -> tuple[torch.Tensor, torch.Tensor]:
        return self.forward(x, params, context)


class Translation(Transformer):
    r"""Performs a pointwise translation of the input tensor.

    .. math::

        x \mapsto y = x + t

        \log \left\lvert \frac{\partial y}{\partial x} \right\rvert = 0

    .. math::

        y \mapsto x = y - t

        \log \left\lvert \frac{\partial x}{\partial y} \right\rvert = 0


    Parameters
    ----------
    x
        Tensor to be transformed
    shift
        The translation, :math:`t`

    """
    domain = REALS
    codomain = REALS

    @property
    def n_params(self) -> int:
        return 1

    @property
    def params_dim(self) -> int:
        return 1  # it doesn't matter

    @property
    def identity_params(self) -> list[float]:
        return [0]

    def _forward(
        self, x: torch.Tensor, params: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Performs a pointwise translation of the input tensor."""
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


class Rescaling(Transformer):
    r"""Performs a pointwise rescaling of the input tensor.

    .. math::

        x \mapsto y = x \odot e^{-s}

    Parameters
    ----------
    x
        Tensor to be transformed
    log_scale
        The scaling factor, :math:`s`

    See Also
    --------
    :py:func:`torchlft.functional.inv_rescaling`

    Performs a pointwise rescaling of the input tensor.

    .. math::

        y \mapsto x = y \odot e^{s}

    """
    domain = REALS
    codomain = REALS

    @property
    def n_params(self) -> int:
        return 1

    @property
    def identity_params(self) -> list[float]:
        return [0]

    def _forward(
        self, x: torch.Tensor, params: torch.Tensor
    ) -> tuple[torch.Tensor]:
        log_scale = params.view_as(x)
        y = x.mul(log_scale.neg().exp())
        ldj = torchnf.utils.sum_except_batch(log_scale.neg())
        return y, ldj

    def _inverse(
        self, y: torch.Tensor, params: torch.Tensor
    ) -> tuple[torch.Tensor]:
        log_scale = params.view_as(y)
        x = y.mul(log_scale.exp())
        ldj = torchnf.utils.sum_except_batch(log_scale)
        return x, ldj


class AffineTransform(Transformer):
    r"""Performs a pointwise affine transformation of the input tensor.

    .. math::

        x \mapsto y = x \odot e^{-s} + t

        \log \left\lvert \frac{\partial y}{\partial x} \right\rvert = -s

    .. math::

        y \mapsto x = (y - t) \odot e^{s}

        \log \left\lvert \frac{\partial x}{\partial y} \right\rvert = s

    Parameters
    ----------
    x
        Tensor to be transformed
    log_scale
        The scaling factor, :math:`s`
    shift
        The translation, :math:`t`

    """
    domain = REALS
    codomain = REALS

    @property
    def n_params(self) -> int:
        return 2

    @property
    def identity_params(self) -> list[float]:
        return [0, 0]

    def _forward(
        self, x: torch.Tensor, params: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        log_scale, shift = [p.view_as(x) for p in params.split(1, dim=1)]
        y = x.mul(log_scale.neg().exp()).add(shift)
        ldj = torchnf.utils.sum_except_batch(log_scale.neg())
        return y, ldj

    def _inverse(
        self, y: torch.Tensor, params: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        log_scale, shift = [p.view_as(y) for p in params.split(1, dim=1)]
        x = y.sub(shift).mul(log_scale.exp())
        ldj = torchnf.utils.sum_except_batch(log_scale)
        return x, ldj


class RQSplineTransform(Transformer):
    domain = REALS
    codomain = REALS

    def __init__(
        self,
        n_segments: int,
        interval: tuple[float],
    ) -> None:
        self._n_segments = n_segments
        self._interval = interval

        self._identity_params = torch.cat(
            (
                torch.full(
                    size=(2 * self._n_segments,),
                    fill_value=1 / self._n_segments,
                ),
                (torch.ones(self._n_knots).exp() - 1).log(),
            ),
            dim=0,
        ).tolist()

    @property
    def _n_knots(self) -> int:
        return self._n_segments - 1

    @property
    def n_params(self) -> int:
        return 2 * self._n_segments + self._n_knots

    @property
    def identity_params(self) -> list[float]:
        return self._identity_params

    def handle_inputs_outside_interval(
        self, outside_interval_mask: torch.BoolTensor
    ) -> None:
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
        r"""Builds a rational quadratic spline function.

        This uses the parametrisation introduced by [Gregory and Delbourgo]_

        Parameters
        ----------
        widths
            Un-normalised segment sizes in the domain
        heights
            Un-normalised segment sizes in the codomain
        derivs
            Unconstrained derivatives at the knots

        Returns
        -------
        widths
            Normalised segment sizes in the domain
        heights
            Normalised segment sizes in the codomain
        derivs
            Constrained derivatives at the knots
        knots_xcoords
            Coordinates of the knots in the domain
        knots_ycoords
            Coordinates of the knots in the codomain


        References
        ----------
        .. [Gregory and Delbourgo]
        Gregory, J. A. & Delbourgo, R. C2 Rational Quadratic Spline
        Interpolation to Monotonic Data, IMA Journal of Numerical Analysis,
        1983, 3, 141-152
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

        ldj = torchnf.utils.sum_except_batch(gradient.log())

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

        ldj = torchnf.utils.sum_except_batch(gradient_fwd.log().neg())

        return x, ldj


class RQSplineTransformClosedInterval(RQSplineTransform):
    def __init__(
        self,
        n_segments: int,
        interval: tuple[float],
        params_dim: int = 1,
    ):
        super().__init__(n_segments, interval, params_dim)
        self.domain = interval
        self.codomain = interval

    @property
    def _n_knots(self) -> int:
        return self._n_segments + 1

    @staticmethod
    def _pad_derivs(derivs: torch.Tensor) -> torch.Tensor:
        return derivs


class RQSplineTransformCircular(RQSplineTransform):
    def __init__(
        self,
        n_segments: int,
        interval: tuple[float],
        params_dim: int = 1,
    ):
        super().__init__(n_segments, interval, params_dim)
        assert math.isclose(interval[1] - interval[0], 2 * PI)
        self.domain = interval
        self.codomain = interval

    @property
    def _n_knots(self) -> int:
        return self._n_segments

    @staticmethod
    def _pad_derivs(derivs: torch.Tensor) -> torch.Tensor:
        return F.pad(derivs, (0, 1), "circular")
