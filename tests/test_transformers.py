import math

from hypothesis import given, strategies as st
import pytest
import torch

from torchnf.transformers import *
from torchnf.utils.tensor import expand_elements

PI = math.pi

_shapes = st.lists(st.integers(1, 10), min_size=2, max_size=5)


def _test_call(transformer, x, params):
    _, _ = transformer(x, params)
    _, _ = transformer.forward(x, params)
    _, _ = transformer.inverse(x, params)


def _test_identity(transformer, x):
    params = expand_elements(
        transformer.identity_params,
        x.shape,
        stack_dim=1,
    )
    y, ldj = transformer(x, params)
    assert torch.allclose(x, y)
    assert torch.allclose(ldj, torch.zeros_like(ldj))
    z, ldj = transformer.inverse(y, params)
    assert torch.allclose(y, z)
    assert torch.allclose(ldj, torch.zeros_like(ldj))


def _test_nan(transformer, x):
    params = expand_elements(
        torch.full([transformer.n_params], float("nan")),
        x.shape,
        stack_dim=1,
    )
    y, ldj = transformer(x, params)
    assert torch.allclose(x, y)
    assert torch.allclose(ldj, torch.zeros_like(ldj))
    z, ldj = transformer.inverse(y, params)
    assert torch.allclose(y, z)
    assert torch.allclose(ldj, torch.zeros_like(ldj))


def _test_roundtrip(transformer, x, params):
    y, ldj_fwd = transformer(x, params)
    z, ldj_inv = transformer.inverse(y, params)
    assert torch.allclose(x, z, atol=1e-4)
    assert torch.allclose(ldj_fwd, ldj_inv.neg(), atol=1e-4)


@given(x_shape=_shapes)
@pytest.mark.parametrize(
    "transformer", [Translation(), Rescaling(), AffineTransform()]
)
def test_call(transformer, x_shape):
    x = torch.empty(x_shape).normal_()
    params = torch.stack(
        [torch.empty(x_shape).normal_() for _ in range(transformer.n_params)],
        dim=1,
    )
    _test_call(transformer, x, params)


@given(x_shape=_shapes)
@pytest.mark.parametrize(
    "transformer", [Translation(), Rescaling(), AffineTransform()]
)
def test_identity(transformer, x_shape):
    x = torch.empty(x_shape).normal_()
    _test_identity(transformer, x)


@given(x_shape=_shapes)
@pytest.mark.parametrize(
    "transformer", [Translation(), Rescaling(), AffineTransform()]
)
def test_nan(transformer, x_shape):
    x = torch.empty(x_shape).normal_()
    _test_nan(transformer, x)


@given(x_shape=_shapes)
@pytest.mark.parametrize(
    "transformer", [Translation(), Rescaling(), AffineTransform()]
)
def test_roundtrip(transformer, x_shape):
    x = torch.empty(x_shape).normal_()
    params = torch.stack(
        [torch.empty(x_shape).normal_() for _ in range(transformer.n_params)],
        dim=1,
    )
    _test_roundtrip(transformer, x, params)


@given(x_shape=_shapes)
@pytest.mark.parametrize(
    "transformer_cls",
    [
        RQSplineTransform,
        RQSplineTransformIntervalDomain,
        RQSplineTransformCircularDomain,
    ],
)
def test_spline(transformer_cls, x_shape):
    """
    Separate test for the spline; allclose requires a smaller atol.
    """
    transformer = transformer_cls(n_segments=8, interval=[-PI, PI])

    x = torch.empty(x_shape).uniform_(-PI + 1e-3, PI - 1e-3)
    params = torch.stack(
        [torch.empty(x_shape).normal_() for _ in range(transformer.n_params)],
        dim=1,
    )

    # Test build spline
    w, h, d = params.movedim(1, -1).split((8, 8, transformer._n_knots), dim=-1)
    w, h, d, kx, ky = transformer.build_spline(w, h, d)
    assert torch.all(w > 0)
    assert torch.all(h > 0)
    assert torch.all(d > 0)
    assert torch.all(kx[..., 1:] > kx[..., :-1])
    assert torch.all(ky[..., 1:] > ky[..., :-1])

    # Test that the forward transformation works
    y, ldj = transformer(x, params)

    # Reverse transformation
    z, ldj_inv = transformer.inverse(y, params)

    def reveal():
        k = (x - z).flatten().abs().argmax()
        print("w:", w.flatten(0, -2)[k])
        print("h:", h.flatten(0, -2)[k])
        print("d:", d.flatten(0, -2)[k])
        print("kx:", kx.flatten(0, -2)[k])
        print("ky:", ky.flatten(0, -2)[k])


    # Test round trip
    # NOTE: These tolerances are terrible. Some edge cases are not working well..
    assert torch.allclose(x, z, atol=1e-1), reveal()
    assert torch.allclose(ldj, ldj_inv.neg(), atol=1e-1)
