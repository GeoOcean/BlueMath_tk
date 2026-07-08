import numpy as np
import pytest

torch = pytest.importorskip("torch")

from bluemath_tk.deeplearning.metrics import (
    ReconstructionLoss,
    evaluate_reconstruction,
    reconstruction_error,
)


torch.set_num_threads(1)


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
