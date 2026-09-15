"""Unit tests for PCA-like latent-structure utilities."""

import numpy as np
import pytest
import torch

from bluemath_tk.deeplearning.latent_structure import (
    LatentStructureRegularizer,
    compute_latent_diagnostics,
    prefix_latent,
    projection_orthogonality_error,
    validate_latent_structure_mode,
)


def test_validate_latent_structure_mode():
    assert validate_latent_structure_mode("NONE") == "none"
    assert validate_latent_structure_mode("orthogonal") == "orthogonal"
    assert validate_latent_structure_mode("pca_like") == "pca_like"
    with pytest.raises(ValueError):
        validate_latent_structure_mode("pca")


def test_none_mode_is_exact_passthrough_and_has_no_state():
    z = torch.randn(5, 3, requires_grad=True)
    regularizer = LatentStructureRegularizer(k=3, mode="none")
    output = regularizer(z)
    assert output is z
    assert regularizer.regularization_losses() == {}
    assert regularizer.state_dict() == {}


def test_identity_projection_has_zero_orthogonality_penalty():
    z = torch.randn(8, 3)
    weight = torch.eye(3, requires_grad=True)
    regularizer = LatentStructureRegularizer(
        k=3,
        mode="orthogonal",
        orthogonality_weight=1.0,
        decorrelation_weight=0.0,
    )
    regularizer.train()
    regularizer(z, projection_weight=weight)
    losses = regularizer.regularization_losses()
    assert torch.allclose(
        losses["latent_orthogonality"],
        torch.zeros((), dtype=losses["latent_orthogonality"].dtype),
        atol=1e-7,
    )


def test_projection_requires_enough_input_features_for_row_orthogonality():
    z = torch.randn(5, 4)
    weight = torch.randn(4, 3)
    regularizer = LatentStructureRegularizer(
        k=4,
        mode="orthogonal",
        orthogonality_weight=1.0,
        decorrelation_weight=0.0,
    )
    with pytest.raises(ValueError, match="in_features=3"):
        regularizer(z, projection_weight=weight)


def test_decorrelation_penalty_is_differentiable():
    torch.manual_seed(7)
    z = torch.randn(16, 4, requires_grad=True)
    weight = torch.eye(4, requires_grad=True)
    regularizer = LatentStructureRegularizer(
        k=4,
        mode="orthogonal",
        orthogonality_weight=0.5,
        decorrelation_weight=0.5,
    )
    regularizer(z, projection_weight=weight)
    loss = sum(regularizer.regularization_losses().values())
    loss.backward()
    assert z.grad is not None
    assert torch.isfinite(z.grad).all()
    assert weight.grad is not None
    assert torch.isfinite(weight.grad).all()


def test_pca_like_ordering_only_masks_during_training():
    z = torch.ones(128, 6)
    weight = torch.randn(6, 8)
    regularizer = LatentStructureRegularizer(
        k=6,
        mode="pca_like",
        orthogonality_weight=0.0,
        decorrelation_weight=0.0,
        ordering_probability=1.0,
    )

    torch.manual_seed(11)
    regularizer.train()
    masked = regularizer(z, projection_weight=weight)
    assert torch.all(masked[:, 0] == 1)
    assert torch.any(masked[:, 1:] == 0)

    regularizer.eval()
    unmasked = regularizer(z, projection_weight=weight)
    assert torch.equal(unmasked, z)


def test_compute_latent_diagnostics_detects_uncorrelated_ordered_scores():
    # Orthogonal columns with decreasing variance.
    z = np.array(
        [
            [-3.0, -1.0],
            [-1.0,  3.0],
            [ 1.0, -3.0],
            [ 3.0,  1.0],
        ],
        dtype=float,
    )
    diagnostics = compute_latent_diagnostics(z)
    assert diagnostics["k"] == 2
    assert diagnostics["mean_abs_offdiag_correlation"] < 1e-12


def test_prefix_latent():
    z = np.arange(12, dtype=float).reshape(3, 4)
    prefixed = prefix_latent(z, 2)
    np.testing.assert_array_equal(prefixed[:, :2], z[:, :2])
    np.testing.assert_array_equal(prefixed[:, 2:], 0.0)


def test_projection_orthogonality_error_identity():
    assert projection_orthogonality_error(np.eye(4)) == pytest.approx(0.0)

def test_decorrelation_penalty_is_scale_invariant():
    """Per-coordinate rescaling must not change the correlation penalty."""
    z = torch.tensor(
        [
            [-2.0, -1.0, 0.5],
            [-1.0, 0.2, 1.0],
            [0.5, 0.7, 1.8],
            [1.5, 1.2, 2.4],
            [2.0, 1.8, 3.2],
        ],
        dtype=torch.float64,
    )
    scales = torch.tensor([0.1, 7.0, 2.5], dtype=torch.float64)

    regularizer = LatentStructureRegularizer(
        k=3,
        mode="orthogonal",
        orthogonality_weight=0.0,
        decorrelation_weight=1.0,
    )

    regularizer(z, projection_weight=None)
    base = regularizer.regularization_losses()["latent_decorrelation"]

    regularizer(z * scales, projection_weight=None)
    scaled = regularizer.regularization_losses()["latent_decorrelation"]

    assert torch.allclose(base, scaled, atol=1e-10, rtol=1e-10)


def test_decorrelation_penalty_detects_redundant_latent_coordinates():
    """Highly correlated latent coordinates should receive a clear penalty."""
    x = torch.linspace(-2.0, 2.0, 32)
    z = torch.stack([x, 2.0 * x, -0.5 * x], dim=1)

    regularizer = LatentStructureRegularizer(
        k=3,
        mode="orthogonal",
        orthogonality_weight=0.0,
        decorrelation_weight=1.0,
    )
    regularizer(z, projection_weight=None)
    loss = regularizer.regularization_losses()["latent_decorrelation"]

    assert loss.item() > 0.95


def test_decorrelation_penalty_near_zero_for_orthogonal_scores():
    """Pairwise uncorrelated score columns should have negligible penalty."""
    z = torch.tensor(
        [
            [-1.0, -1.0],
            [-1.0, 1.0],
            [1.0, -1.0],
            [1.0, 1.0],
        ],
        dtype=torch.float64,
    )

    regularizer = LatentStructureRegularizer(
        k=2,
        mode="orthogonal",
        orthogonality_weight=0.0,
        decorrelation_weight=1.0,
    )
    regularizer(z, projection_weight=None)
    loss = regularizer.regularization_losses()["latent_decorrelation"]

    assert loss.item() < 1e-12

def test_decorrelation_penalty_remains_scale_invariant_across_old_eps_floor():
    """Extreme nonzero rescaling must not weaken the correlation objective."""
    z = torch.tensor(
        [
            [-2.0, -1.0, 0.5],
            [-1.0, 0.2, 1.0],
            [0.5, 0.7, 1.8],
            [1.5, 1.2, 2.4],
            [2.0, 1.8, 3.2],
        ],
        dtype=torch.float32,
    )
    scales = torch.tensor([1e-20, 1e20, 1e-10], dtype=torch.float32)

    regularizer = LatentStructureRegularizer(
        k=3,
        mode="orthogonal",
        orthogonality_weight=0.0,
        decorrelation_weight=1.0,
    )

    regularizer(z, projection_weight=None)
    base = regularizer.regularization_losses()["latent_decorrelation"]

    regularizer(z * scales, projection_weight=None)
    scaled = regularizer.regularization_losses()["latent_decorrelation"]

    assert torch.allclose(base, scaled, atol=1e-5, rtol=1e-5)


def test_decorrelation_penalty_does_not_reward_collapsed_coordinate():
    """Exact collapse must not reduce a correlated representation's loss."""
    x = torch.linspace(-2.0, 2.0, 32)
    correlated = torch.stack([x, 2.0 * x, -0.5 * x], dim=1)
    collapsed = correlated.clone()
    collapsed[:, 1] = 0.0

    regularizer = LatentStructureRegularizer(
        k=3,
        mode="orthogonal",
        orthogonality_weight=0.0,
        decorrelation_weight=1.0,
    )

    regularizer(correlated, projection_weight=None)
    correlated_loss = regularizer.regularization_losses()[
        "latent_decorrelation"
    ]

    regularizer(collapsed, projection_weight=None)
    collapsed_loss = regularizer.regularization_losses()[
        "latent_decorrelation"
    ]

    assert correlated_loss.item() == pytest.approx(1.0, abs=1e-6)
    assert collapsed_loss.item() >= correlated_loss.item() - 1e-6


def test_structured_linear_preserves_linear_state_keys_and_parameter_count():
    """Structured projection must remain checkpoint-compatible with nn.Linear."""
    from bluemath_tk.deeplearning.latent_structure import StructuredLatentLinear

    plain = torch.nn.Linear(7, 3)
    structured = StructuredLatentLinear(7, 3, mode="none")

    assert list(structured.state_dict()) == list(plain.state_dict())
    assert sum(p.numel() for p in structured.parameters()) == sum(
        p.numel() for p in plain.parameters()
    )


def test_ordering_boundary_probabilities_are_well_defined():
    """p=0 and k=1 are exact pass-through cases in training mode."""
    z = torch.randn(16, 3)
    zero_probability = LatentStructureRegularizer(
        k=3,
        mode="pca_like",
        orthogonality_weight=0.0,
        decorrelation_weight=0.0,
        ordering_probability=0.0,
    )
    zero_probability.train()
    assert torch.equal(zero_probability(z), z)

    z_single = torch.randn(16, 1)
    one_dimension = LatentStructureRegularizer(
        k=1,
        mode="pca_like",
        orthogonality_weight=0.0,
        decorrelation_weight=0.0,
        ordering_probability=1.0,
    )
    one_dimension.train()
    assert torch.equal(one_dimension(z_single), z_single)

def test_near_collapsed_coordinate_has_finite_nonzero_gradient():
    """A tiny but nonzero active coordinate must still receive a gradient."""
    z = torch.tensor(
        [
            [-2.0, 0.3e-8, 1.0],
            [-1.0, -0.7e-8, 0.2],
            [0.0, 1.2e-8, -0.5],
            [1.0, 0.1e-8, 1.1],
            [2.0, 1.8e-8, -1.2],
            [3.0, -0.4e-8, 0.7],
        ],
        dtype=torch.float64,
        requires_grad=True,
    )
    regularizer = LatentStructureRegularizer(
        k=3,
        mode="orthogonal",
        orthogonality_weight=0.0,
        decorrelation_weight=1.0,
    )

    regularizer(z, projection_weight=None)
    loss = regularizer.regularization_losses()["latent_decorrelation"]
    loss.backward()

    assert z.grad is not None
    assert torch.isfinite(z.grad).all()
    assert torch.sum(torch.abs(z.grad[:, 1])).item() > 0.0


def test_latent_diagnostics_correlation_is_scale_invariant():
    """Public Pearson diagnostics must survive extreme coordinate rescaling."""
    z = np.array(
        [
            [-2.0, -1.0, 0.5],
            [-1.0, 0.2, 1.0],
            [0.5, 0.7, 1.8],
            [1.5, 1.2, 2.4],
            [2.0, 1.8, 3.2],
        ],
        dtype=np.float64,
    )
    scales = np.array([1e-100, 1e100, 1e-50], dtype=np.float64)

    base = compute_latent_diagnostics(z)
    scaled = compute_latent_diagnostics(z * scales)

    assert scaled["mean_abs_offdiag_correlation"] == pytest.approx(
        base["mean_abs_offdiag_correlation"],
        rel=1e-12,
        abs=1e-12,
    )
    assert scaled["max_abs_offdiag_correlation"] == pytest.approx(
        base["max_abs_offdiag_correlation"],
        rel=1e-12,
        abs=1e-12,
    )
    assert scaled["n_collapsed_coordinates"] == 0
    assert scaled["collapsed_coordinates"] == []


def test_latent_diagnostics_reports_exactly_collapsed_coordinates():
    """Undefined Pearson coordinates must be explicit in public diagnostics."""
    z = np.array(
        [
            [-2.0, 5.0, 1.0],
            [-1.0, 5.0, 0.5],
            [0.0, 5.0, -0.5],
            [1.0, 5.0, -1.0],
            [2.0, 5.0, 0.2],
        ],
        dtype=np.float64,
    )

    diagnostics = compute_latent_diagnostics(z)

    assert diagnostics["n_collapsed_coordinates"] == 1
    assert diagnostics["collapsed_coordinates"] == [1]
    assert np.isfinite(diagnostics["mean_abs_offdiag_correlation"])
    assert np.isfinite(diagnostics["max_abs_offdiag_correlation"])
