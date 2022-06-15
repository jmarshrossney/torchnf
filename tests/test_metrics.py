from __future__ import annotations

import math
import random

import pytest
import torch

from torchnf.metrics import (
    ShiftedKLDivergence,
    EffectiveSampleSize,
    AcceptanceRate,
    LongestRejectionRun,
    IntegratedAutocorrelation,
    LogStatWeightMetrics,
)


logweight_metrics = [
    ShiftedKLDivergence,
    EffectiveSampleSize,
    AcceptanceRate,
    LongestRejectionRun,
    IntegratedAutocorrelation,
]
metric_collections = [
    LogStatWeightMetrics(True),
    LogStatWeightMetrics(False),
]


def _gen_log_weights(seed=None):
    if seed is not None:
        random.seed(seed)
        _ = torch.random.manual_seed(seed)
    q = torch.distributions.Normal(0, 1)
    x = q.sample([10000])
    p = torch.distributions.Normal(0, 1.2)
    log_weights = p.log_prob(x) - q.log_prob(x)
    return log_weights


@pytest.fixture
def log_weights():
    seed = 8967452301
    return _gen_log_weights(seed)


@pytest.mark.parametrize("metric", logweight_metrics)
def test_metrics_update_and_compute(metric, log_weights):
    metric = metric()
    metric.update(log_weights)
    result = metric.compute()
    assert isinstance(result, torch.Tensor)
    assert result.shape == torch.Size([])


@pytest.mark.parametrize("metric", metric_collections)
def test_collections_update_and_compute(metric, log_weights):
    metric.update(log_weights)
    result = metric.compute()
    assert all([isinstance(m, torch.Tensor) for m in result.values()])
    assert all([m.shape == torch.Size([]) for m in result.values()])


@pytest.mark.parametrize("metric", logweight_metrics)
def test_metrics_forward(metric, log_weights):
    metric = metric()
    result = metric(log_weights)
    assert isinstance(result, torch.Tensor)
    assert result.shape == torch.Size([])


@pytest.mark.parametrize("metric", metric_collections)
def test_collections_forward(metric, log_weights):
    result = metric(log_weights)
    assert all([isinstance(m, torch.Tensor) for m in result.values()])
    assert all([m.shape == torch.Size([]) for m in result.values()])


def test_compute_groups(log_weights):
    assert len(LogStatWeightMetrics(mcmc=False).compute_groups) == 1
    assert len(LogStatWeightMetrics(mcmc=True).compute_groups) == 2


@pytest.mark.parametrize("metric", logweight_metrics)
def test_multiple_batches(metric):
    metric = metric()
    metric.update(_gen_log_weights())
    metric.update(_gen_log_weights())
    result = metric.compute()
    assert isinstance(result, torch.Tensor)
    assert result.shape == torch.Size([2])


"""
def test_metrics(metrics):
    assert math.isclose(metrics.kl_divergence, 0.0285, abs_tol=1e-4)
    assert math.isclose(metrics.acceptance, 0.8841, abs_tol=1e-4)
    assert math.isclose(
        metrics.integrated_autocorrelation, 0.7164, abs_tol=1e-4
    )
    assert math.isclose(metrics.effective_sample_size, 0.8913, abs_tol=1e-4)
    assert metrics.longest_rejection_run == 24
"""
