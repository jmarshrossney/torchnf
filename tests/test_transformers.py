from hypothesis import given, strategies as st
import pytest
import random
import torch

from torchnf.transformers import (
    Translation,
    Rescaling,
    AffineTransform,
    RQSplineTransform,
    RQSplineTransformClosedInterval,
    RQSplineTransformCircular,
)
from torchnf.utils import expand_elements

_shapes = st.lists(st.integers(1, 10), min_size=2, max_size=5)


def _test_call(transformer, x, params):
    _, _ = transformer(x, params)
    _, _ = transformer.forward(x, params)
    _, _ = transformer.inverse(x, params)


def _test_identity(transformer, x):
    params = expand_elements(
        torch.Tensor(transformer.identity_params),
        shape=x.shape,
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
        shape=x.shape,
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
    assert torch.allclose(x, z, atol=1e-5)
    assert torch.allclose(ldj_fwd, ldj_inv.neg(), atol=1e-5)


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
