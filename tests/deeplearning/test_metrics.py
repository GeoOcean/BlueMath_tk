"""Tests for public reconstruction metrics and losses."""

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from bluemath_tk.deeplearning.metrics import (  # noqa: E402
    ReconstructionLoss,
    evaluate_reconstruction,
    reconstruction_error,
)


@pytest.fixture(autouse=True)
def _preserve_global_state():
    previous_threads = torch.get_num_threads()
    numpy_state = np.random.get_state()
    torch_state = torch.random.get_rng_state()
    cuda_states = torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None
    try:
        torch.set_num_threads(1)
        yield
    finally:
        torch.set_num_threads(previous_threads)
        np.random.set_state(numpy_state)
        torch.random.set_rng_state(torch_state)
        if cuda_states is not None:
            torch.cuda.set_rng_state_all(cuda_states)


def test_reconstruction_error_numpy_sample_mse_matches_manual():
    """Per-sample MSE should match the manual NumPy calculation."""
    y_true = np.array(
        [
            [[0.0, 1.0], [2.0, 3.0]],
            [[1.0, 1.0], [1.0, 1.0]],
        ],
        dtype=np.float32,
    )
    y_pred = np.array(
        [
            [[1.0, 1.0], [4.0, 3.0]],
            [[2.0, 1.0], [1.0, 5.0]],
        ],
        dtype=np.float32,
    )

    errors = reconstruction_error(
        y_true,
        y_pred,
        metric="mse",
        reduction="sample",
    )
    manual_errors = np.mean((y_pred - y_true) ** 2, axis=(1, 2))

    assert isinstance(errors, np.ndarray)
    assert errors.shape == (2,)
    assert np.allclose(errors, manual_errors)


def test_reconstruction_error_numpy_reduction_none_and_sum():
    """Elementwise and summed MAE reductions should match manual results."""
    y_true = np.array([[0.0, 1.0, 2.0], [1.0, 1.0, 1.0]], dtype=np.float32)
    y_pred = np.array([[1.0, 1.0, 4.0], [2.0, 3.0, 1.0]], dtype=np.float32)

    elementwise = reconstruction_error(
        y_true,
        y_pred,
        metric="mae",
        reduction="none",
    )
    summed = reconstruction_error(
        y_true,
        y_pred,
        metric="mae",
        reduction="sum",
    )

    manual_elementwise = np.abs(y_pred - y_true)
    manual_summed = np.sum(np.mean(manual_elementwise, axis=1))

    assert np.allclose(elementwise, manual_elementwise)
    assert np.isclose(summed, manual_summed)


def test_reconstruction_error_numpy_rmse_sample_matches_manual():
    """Per-sample RMSE should be sqrt of per-sample MSE."""
    y_true = np.array([[0.0, 0.0], [0.0, 0.0]], dtype=np.float32)
    y_pred = np.array([[3.0, 4.0], [0.0, 12.0]], dtype=np.float32)

    errors = reconstruction_error(
        y_true,
        y_pred,
        metric="rmse",
        reduction="sample",
    )
    manual_errors = np.sqrt(np.mean((y_pred - y_true) ** 2, axis=1))

    assert np.allclose(errors, manual_errors)


def test_evaluate_reconstruction_returns_summary_statistics():
    """evaluate_reconstruction should return useful scalar summary statistics."""
    y_true = np.array([[0.0, 0.0], [1.0, 1.0]], dtype=np.float32)
    y_pred = np.array([[1.0, 1.0], [1.0, 3.0]], dtype=np.float32)

    summary = evaluate_reconstruction(y_true, y_pred, metric="mse")

    assert summary["metric"] == "mse"
    assert summary["n_samples"] == 2
    for key in ["mean", "std", "median", "min", "max"]:
        assert key in summary
        assert np.isfinite(summary[key])
    assert summary["min"] <= summary["mean"] <= summary["max"]


def test_evaluate_reconstruction_uses_stable_float64_summary_mean(monkeypatch):
    y_true = np.zeros((3, 1), dtype=np.float64)
    y_pred = np.full((3, 1), 1e308, dtype=np.float64)

    # Isolate mean accumulation from NumPy's independent extreme-value std overflow.
    monkeypatch.setattr(np, "std", lambda _values: np.float64(0.0))
    summary = evaluate_reconstruction(y_true, y_pred, metric="mae")

    assert summary["n_samples"] == 3
    assert summary["mean"] == pytest.approx(1e308)


def test_reconstruction_error_torch_mean_is_differentiable():
    """Tensor inputs should keep gradients so the metric can be used as a loss."""
    y_true = torch.zeros((4, 3), dtype=torch.float32)
    y_pred = torch.randn((4, 3), dtype=torch.float32, requires_grad=True)

    loss = reconstruction_error(
        y_true,
        y_pred,
        metric="mse",
        reduction="mean",
    )
    loss.backward()

    assert torch.is_tensor(loss)
    assert loss.ndim == 0
    assert y_pred.grad is not None
    assert torch.isfinite(y_pred.grad).all()


def test_reconstruction_error_supports_mixed_numpy_target_and_tensor_prediction():
    """NumPy targets should be converted to the prediction tensor device/dtype."""
    y_true = np.zeros((4, 3), dtype=np.float32)
    y_pred = torch.ones((4, 3), dtype=torch.float32, requires_grad=True)

    loss = reconstruction_error(
        y_true,
        y_pred,
        metric="mse",
        reduction="mean",
    )
    loss.backward()

    assert torch.is_tensor(loss)
    assert y_pred.grad is not None
    assert torch.isfinite(y_pred.grad).all()


def test_reconstruction_loss_module_can_be_used_in_training():
    """ReconstructionLoss should behave like a PyTorch criterion."""
    y_true = torch.zeros((4, 3), dtype=torch.float32)
    y_pred = torch.randn((4, 3), dtype=torch.float32, requires_grad=True)
    criterion = ReconstructionLoss(metric="mae", reduction="mean")

    loss = criterion(y_pred, y_true)
    loss.backward()

    assert torch.is_tensor(loss)
    assert loss.ndim == 0
    assert y_pred.grad is not None
    assert torch.isfinite(y_pred.grad).all()


def test_reconstruction_metrics_validate_inputs():
    """Invalid options and shape mismatches should raise clear errors."""
    y_true = np.zeros((4, 3), dtype=np.float32)
    y_pred = np.zeros((4, 2), dtype=np.float32)

    with pytest.raises(ValueError, match="same shape"):
        reconstruction_error(y_true, y_pred)

    with pytest.raises(ValueError, match="metric"):
        reconstruction_error(y_true, y_true, metric="invalid")

    with pytest.raises(ValueError, match="reduction"):
        reconstruction_error(y_true, y_true, reduction="invalid")


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
@pytest.mark.parametrize("reduction", ["none", "sample", "mean", "sum"])
def test_exact_zero_rmse_has_finite_zero_gradient(reduction, dtype):
    y_true = torch.zeros((3, 2, 4), dtype=dtype)
    y_pred = torch.zeros((3, 2, 4), dtype=dtype, requires_grad=True)

    loss = reconstruction_error(
        y_true,
        y_pred,
        metric="rmse",
        reduction=reduction,
        eps=0.0,
    )
    loss.sum().backward()

    assert torch.count_nonzero(loss) == 0
    assert torch.isfinite(y_pred.grad).all()
    assert torch.count_nonzero(y_pred.grad) == 0


def test_reconstruction_loss_exact_zero_rmse_has_finite_zero_gradient():
    y_true = torch.zeros((3, 5), dtype=torch.float32)
    y_pred = torch.zeros((3, 5), dtype=torch.float32, requires_grad=True)
    criterion = ReconstructionLoss(metric="rmse", reduction="mean", eps=0.0)

    loss = criterion(y_pred, y_true)
    loss.backward()

    assert loss.item() == 0.0
    assert torch.isfinite(y_pred.grad).all()
    assert torch.count_nonzero(y_pred.grad) == 0


def test_rmse_values_and_gradients_match_analytic_result_away_from_zero():
    y_true = torch.zeros((2, 2, 2), dtype=torch.float64)
    values = torch.tensor(
        [[[1.0, 2.0], [3.0, 4.0]], [[2.0, 3.0], [4.0, 5.0]]],
        dtype=torch.float64,
    )
    y_pred = values.clone().requires_grad_(True)
    reference_pred = values.clone().requires_grad_(True)

    actual = reconstruction_error(
        y_true,
        y_pred,
        metric="rmse",
        reduction="mean",
        eps=0.0,
    )
    expected = torch.sqrt(reference_pred.pow(2).mean(dim=(1, 2))).mean()
    actual.backward()
    expected.backward()

    assert torch.allclose(actual, expected)
    assert torch.allclose(y_pred.grad, reference_pred.grad)
    assert torch.isfinite(y_pred.grad).all()


def _assert_constant_float32_rmse(residual, reduction):
    y_true = torch.zeros((2, 2), dtype=torch.float32)
    y_pred = torch.full(
        (2, 2),
        residual,
        dtype=torch.float32,
        requires_grad=True,
    )

    actual = reconstruction_error(
        y_true,
        y_pred,
        metric="rmse",
        reduction=reduction,
        eps=0.0,
    )
    actual.sum().backward()

    stored_residual = float(y_pred.detach()[0, 0])
    expected_value = stored_residual * 2 if reduction == "sum" else stored_residual
    expected = torch.full_like(actual, expected_value)
    gradient_scale = {
        "none": 1.0,
        "sample": 0.5,
        "mean": 0.25,
        "sum": 0.5,
    }[reduction]
    if stored_residual == 0:
        gradient_scale = 0.0
    expected_gradient = torch.full_like(y_pred, gradient_scale)
    numpy_result = reconstruction_error(
        np.zeros((2, 2), dtype=np.float32),
        np.full((2, 2), residual, dtype=np.float32),
        metric="rmse",
        reduction=reduction,
        eps=0.0,
    )

    assert actual.dtype == torch.float32
    assert actual.device == y_pred.device
    assert torch.isfinite(actual).all()
    torch.testing.assert_close(actual, expected, rtol=5e-6, atol=0.0)
    np.testing.assert_allclose(
        actual.detach().cpu().numpy(),
        np.asarray(numpy_result),
        rtol=5e-6,
        atol=0.0,
    )
    assert y_pred.grad is not None
    assert torch.isfinite(y_pred.grad).all()
    torch.testing.assert_close(
        y_pred.grad,
        expected_gradient,
        rtol=5e-6,
        atol=0.0,
    )


@pytest.mark.parametrize(
    "residual",
    [
        pytest.param(0.0, id="zero"),
        pytest.param(1e-25, id="small"),
        pytest.param(3.0, id="moderate"),
    ],
)
@pytest.mark.parametrize("reduction", ["none", "sample", "mean", "sum"])
def test_float32_rmse_multisample_reductions_are_stable(residual, reduction):
    _assert_constant_float32_rmse(residual, reduction)


@pytest.mark.parametrize("reduction", ["none", "sample", "mean"])
def test_float32_extreme_rmse_multisample_mean_is_stable(reduction):
    _assert_constant_float32_rmse(3e38, reduction)


@pytest.mark.parametrize(
    ("torch_dtype", "numpy_dtype", "metric", "residual"),
    [
        pytest.param(
            torch.float32,
            np.float32,
            "mae",
            3e38,
            id="float32-mae",
        ),
        pytest.param(
            torch.float32,
            np.float32,
            "rmse",
            3e38,
            id="float32-rmse",
        ),
        pytest.param(
            torch.float32,
            np.float32,
            "mse",
            1.8e19,
            id="float32-mse",
        ),
        pytest.param(
            torch.float64,
            np.float64,
            "mae",
            1e308,
            id="float64-mae",
        ),
        pytest.param(
            torch.float64,
            np.float64,
            "rmse",
            1e308,
            id="float64-rmse",
        ),
        pytest.param(
            torch.float64,
            np.float64,
            "mse",
            1.3e154,
            id="float64-mse",
        ),
    ],
)
def test_multisample_extreme_metric_means_match_numpy_and_gradients(
    torch_dtype,
    numpy_dtype,
    metric,
    residual,
):
    y_true = torch.zeros((2, 2), dtype=torch_dtype)
    y_pred = torch.full(
        (2, 2),
        residual,
        dtype=torch_dtype,
        requires_grad=True,
    )
    numpy_true = np.zeros((2, 2), dtype=numpy_dtype)
    numpy_pred = np.full((2, 2), residual, dtype=numpy_dtype)

    sample_result = reconstruction_error(
        y_true,
        y_pred,
        metric=metric,
        reduction="sample",
    )
    mean_result = reconstruction_error(
        y_true,
        y_pred,
        metric=metric,
        reduction="mean",
    )
    numpy_sample = reconstruction_error(
        numpy_true,
        numpy_pred,
        metric=metric,
        reduction="sample",
    )
    numpy_mean = reconstruction_error(
        numpy_true,
        numpy_pred,
        metric=metric,
        reduction="mean",
    )
    mean_result.backward()

    stored_residual = float(y_pred.detach()[0, 0])
    if metric == "mse":
        expected_value = stored_residual * stored_residual
        expected_gradient = stored_residual / 2
    else:
        expected_value = stored_residual
        expected_gradient = 0.25
    rtol = 5e-6 if torch_dtype == torch.float32 else 5e-15

    assert sample_result.shape == (2,)
    assert sample_result.dtype == torch_dtype
    assert mean_result.shape == ()
    assert mean_result.dtype == torch_dtype
    assert mean_result.device == y_pred.device
    assert numpy_sample.shape == (2,)
    assert numpy_sample.dtype == numpy_dtype
    assert torch.isfinite(sample_result).all()
    assert torch.isfinite(mean_result)
    torch.testing.assert_close(
        sample_result,
        torch.full_like(sample_result, expected_value),
        rtol=rtol,
        atol=0.0,
    )
    torch.testing.assert_close(
        mean_result,
        torch.tensor(expected_value, dtype=torch_dtype),
        rtol=rtol,
        atol=0.0,
    )
    np.testing.assert_allclose(
        numpy_sample,
        np.full((2,), expected_value, dtype=numpy_dtype),
        rtol=rtol,
        atol=0.0,
    )
    np.testing.assert_allclose(
        numpy_mean,
        expected_value,
        rtol=rtol,
        atol=0.0,
    )
    assert y_pred.grad is not None
    assert torch.isfinite(y_pred.grad).all()
    torch.testing.assert_close(
        y_pred.grad,
        torch.full_like(y_pred, expected_gradient),
        rtol=rtol,
        atol=0.0,
    )


@pytest.mark.parametrize(
    ("torch_dtype", "numpy_dtype", "residual"),
    [
        pytest.param(torch.float32, np.float32, 2e19, id="float32"),
        pytest.param(torch.float64, np.float64, 1.5e154, id="float64"),
    ],
)
def test_mse_grouping_avoids_nonrepresentable_elementwise_square(
    torch_dtype,
    numpy_dtype,
    residual,
):
    y_true = torch.zeros((2, 2), dtype=torch_dtype)
    y_pred = torch.tensor(
        [[residual, 0.0], [residual, 0.0]],
        dtype=torch_dtype,
        requires_grad=True,
    )
    numpy_true = np.zeros((2, 2), dtype=numpy_dtype)
    numpy_pred = np.array(
        [[residual, 0.0], [residual, 0.0]],
        dtype=numpy_dtype,
    )

    assert torch.isinf(y_pred.detach().square()).any()
    with np.errstate(over="ignore"):
        assert np.isinf(numpy_pred**2).any()

    sample_result = reconstruction_error(
        y_true,
        y_pred,
        metric="mse",
        reduction="sample",
    )
    mean_result = reconstruction_error(
        y_true,
        y_pred,
        metric="mse",
        reduction="mean",
    )
    numpy_sample = reconstruction_error(
        numpy_true,
        numpy_pred,
        metric="mse",
        reduction="sample",
    )
    numpy_mean = reconstruction_error(
        numpy_true,
        numpy_pred,
        metric="mse",
        reduction="mean",
    )
    mean_result.backward()

    stored_residual = float(y_pred.detach()[0, 0])
    expected_value = stored_residual * (stored_residual / 2)
    expected_gradient = y_pred.detach() / 2
    rtol = 5e-6 if torch_dtype == torch.float32 else 5e-15

    assert sample_result.dtype == torch_dtype
    assert mean_result.dtype == torch_dtype
    assert numpy_sample.dtype == numpy_dtype
    assert torch.isfinite(sample_result).all()
    assert torch.isfinite(mean_result)
    torch.testing.assert_close(
        sample_result,
        torch.full_like(sample_result, expected_value),
        rtol=rtol,
        atol=0.0,
    )
    torch.testing.assert_close(
        mean_result,
        torch.tensor(expected_value, dtype=torch_dtype),
        rtol=rtol,
        atol=0.0,
    )
    np.testing.assert_allclose(
        numpy_sample,
        np.full((2,), expected_value, dtype=numpy_dtype),
        rtol=rtol,
        atol=0.0,
    )
    np.testing.assert_allclose(
        numpy_mean,
        expected_value,
        rtol=rtol,
        atol=0.0,
    )
    assert y_pred.grad is not None
    assert torch.isfinite(y_pred.grad).all()
    torch.testing.assert_close(
        y_pred.grad,
        expected_gradient,
        rtol=rtol,
        atol=0.0,
    )


@pytest.mark.parametrize("metric", ["mse", "mae"])
@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_exact_zero_scaled_mean_has_zero_gradient(metric, dtype):
    y_true = torch.zeros((2, 2), dtype=dtype)
    y_pred = torch.zeros((2, 2), dtype=dtype, requires_grad=True)

    result = reconstruction_error(
        y_true,
        y_pred,
        metric=metric,
        reduction="mean",
    )
    result.backward()

    assert result.dtype == dtype
    assert result.item() == 0.0
    assert torch.isfinite(y_pred.grad).all()
    assert torch.count_nonzero(y_pred.grad) == 0


@pytest.mark.parametrize(
    "residual",
    [
        pytest.param(1e-25, id="small"),
        pytest.param(3e38, id="large"),
    ],
)
def test_reconstruction_loss_rmse_preserves_extreme_float32_gradients(residual):
    y_true = torch.zeros((2, 2), dtype=torch.float32)
    y_pred = torch.full(
        (2, 2),
        residual,
        dtype=torch.float32,
        requires_grad=True,
    )
    criterion = ReconstructionLoss(metric="rmse", reduction="mean", eps=0.0)

    loss = criterion(y_pred, y_true)
    loss.backward()

    torch.testing.assert_close(
        loss,
        torch.tensor(residual, dtype=torch.float32),
        rtol=5e-6,
        atol=0.0,
    )
    assert torch.isfinite(y_pred.grad).all()
    torch.testing.assert_close(
        y_pred.grad,
        torch.full_like(y_pred, 0.25),
        rtol=5e-6,
        atol=0.0,
    )


def test_scaled_rmse_preserves_eps_inside_the_square_root():
    y_true = torch.zeros((1, 2), dtype=torch.float32)
    y_pred = torch.tensor([[3.0, 4.0]], requires_grad=True)
    eps = 2.25

    loss = reconstruction_error(
        y_true,
        y_pred,
        metric="rmse",
        reduction="sample",
        eps=eps,
    )
    loss.sum().backward()

    expected_value = np.sqrt((3.0**2 + 4.0**2) / 2 + eps)
    expected_gradient = torch.tensor(
        [[3.0 / (2 * expected_value), 4.0 / (2 * expected_value)]],
        dtype=torch.float32,
    )
    torch.testing.assert_close(
        loss,
        torch.tensor([expected_value], dtype=torch.float32),
    )
    torch.testing.assert_close(y_pred.grad, expected_gradient)


@pytest.mark.parametrize("kind", ["numpy", "torch"])
def test_rmse_sum_rejects_a_nonrepresentable_float32_final_result(kind):
    if kind == "torch":
        y_true = torch.zeros((2, 1), dtype=torch.float32)
        y_pred = torch.full((2, 1), 3e38, dtype=torch.float32)
    else:
        y_true = np.zeros((2, 1), dtype=np.float32)
        y_pred = np.full((2, 1), 3e38, dtype=np.float32)

    with pytest.raises(FloatingPointError, match="Metric reduction"):
        reconstruction_error(
            y_true,
            y_pred,
            metric="rmse",
            reduction="sum",
            eps=0.0,
        )


@pytest.mark.parametrize("metric", ["mse", "mae", "rmse"])
@pytest.mark.parametrize(
    "kind",
    ["numpy", "torch", "numpy-target", "numpy-prediction"],
)
def test_metrics_support_valid_numpy_torch_and_mixed_inputs(metric, kind):
    numpy_true = np.zeros((3, 2), dtype="float32")
    numpy_pred = np.ones((3, 2), dtype="float32")
    if kind == "numpy":
        y_true, y_pred = numpy_true, numpy_pred
    elif kind == "torch":
        y_true = torch.from_numpy(numpy_true)
        y_pred = torch.from_numpy(numpy_pred)
    elif kind == "numpy-target":
        y_true, y_pred = numpy_true, torch.from_numpy(numpy_pred)
    else:
        y_true, y_pred = torch.from_numpy(numpy_true), numpy_pred

    result = reconstruction_error(y_true, y_pred, metric=metric, reduction="mean")
    summary = evaluate_reconstruction(y_true, y_pred, metric=metric)

    if torch.is_tensor(result):
        assert result.item() == pytest.approx(1.0)
    else:
        assert result == pytest.approx(1.0)
    assert summary["mean"] == pytest.approx(1.0)


@pytest.mark.parametrize("invalid", [float("nan"), float("inf"), -float("inf")])
@pytest.mark.parametrize("kind", ["numpy", "torch", "mixed"])
def test_metrics_reject_nonfinite_inputs(invalid, kind):
    y_true = np.zeros((2, 3), dtype="float32")
    y_pred = np.zeros((2, 3), dtype="float32")
    y_pred[0, 0] = invalid
    if kind == "torch":
        y_true = torch.from_numpy(y_true)
        y_pred = torch.from_numpy(y_pred)
    elif kind == "mixed":
        y_pred = torch.from_numpy(y_pred)

    with pytest.raises(ValueError, match="finite"):
        reconstruction_error(y_true, y_pred)
    with pytest.raises(ValueError, match="finite"):
        evaluate_reconstruction(y_true, y_pred)


@pytest.mark.parametrize(
    "values",
    [
        np.zeros((2, 3), dtype="complex64"),
        torch.zeros((2, 3), dtype=torch.complex64),
    ],
)
def test_metrics_reject_complex_inputs(values):
    zeros = np.zeros((2, 3), dtype="float32")

    with pytest.raises(TypeError, match="real-valued"):
        reconstruction_error(zeros, values)


@pytest.mark.parametrize(
    "values",
    [
        np.full((2, 3), "bad"),
        np.ones((2, 3), dtype=bool),
        torch.ones((2, 3), dtype=torch.bool),
        [[0.0, 0.0], [0.0, 0.0]],
    ],
)
def test_metrics_reject_unsupported_or_non_numeric_inputs(values):
    zeros = np.zeros((2, 3), dtype="float32")

    with pytest.raises(TypeError, match="numeric|NumPy array"):
        reconstruction_error(zeros, values)


@pytest.mark.parametrize(
    "shape",
    [(), (0,), (0, 3), (2, 0)],
)
def test_metrics_reject_empty_or_scalar_inputs(shape):
    values = np.empty(shape, dtype="float32")

    with pytest.raises(ValueError, match="sample dimension|non-empty"):
        reconstruction_error(values, values)


@pytest.mark.parametrize(
    "eps",
    [
        -1.0,
        float("nan"),
        float("inf"),
        -float("inf"),
        True,
        False,
        "0.0",
        1 + 0j,
        [0.0],
        np.array([0.0]),
    ],
)
def test_all_public_metrics_reject_invalid_eps(eps):
    values = np.zeros((2, 3), dtype="float32")

    with pytest.raises(ValueError, match="eps"):
        reconstruction_error(values, values, metric="rmse", eps=eps)
    with pytest.raises(ValueError, match="eps"):
        evaluate_reconstruction(values, values, metric="rmse", eps=eps)
    with pytest.raises(ValueError, match="eps"):
        ReconstructionLoss(metric="rmse", eps=eps)


@pytest.mark.parametrize("kind", ["numpy", "torch", "mixed-cast"])
def test_metrics_reject_nonfinite_arithmetic_overflow(kind):
    if kind == "numpy":
        limit = np.finfo(np.float64).max
        y_true = np.full((2, 2), limit, dtype="float64")
        y_pred = np.full((2, 2), -limit, dtype="float64")
    elif kind == "torch":
        limit = torch.finfo(torch.float32).max
        y_true = torch.full((2, 2), limit, dtype=torch.float32)
        y_pred = torch.full((2, 2), -limit, dtype=torch.float32)
    else:
        y_true = np.full((2, 2), 1e300, dtype="float64")
        y_pred = torch.zeros((2, 2), dtype=torch.float32)

    with pytest.raises(FloatingPointError, match="finite"):
        reconstruction_error(y_true, y_pred, metric="mse")


def test_evaluate_reconstruction_rejects_summary_overflow():
    limit = np.finfo(np.float64).max
    y_true = np.zeros((2, 1), dtype="float64")
    y_pred = np.array([[limit], [0.0]], dtype="float64")

    with pytest.raises(FloatingPointError, match="summary"):
        evaluate_reconstruction(y_true, y_pred, metric="mae")


def test_reconstruction_loss_validates_forward_inputs():
    criterion = ReconstructionLoss(metric="mse")
    y_true = torch.zeros((2, 3), dtype=torch.float32)
    y_pred = torch.zeros((2, 3), dtype=torch.float32)
    y_pred[0, 0] = float("nan")

    with pytest.raises(ValueError, match="finite"):
        criterion(y_pred, y_true)
