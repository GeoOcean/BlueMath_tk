"""Regression coverage for branch-test global-state isolation."""

from pathlib import Path

import numpy as np
import pytest

torch = pytest.importorskip("torch")


class _FailSelectedTestCalls:
    def pytest_runtest_call(self, item):
        pytest.fail(
            f"deliberate fixture-cleanup probe for {item.nodeid}",
            pytrace=False,
        )


def _capture_state():
    return {
        "threads": torch.get_num_threads(),
        "numpy": np.random.get_state(),
        "torch": torch.random.get_rng_state().clone(),
        "cuda": (
            [state.clone() for state in torch.cuda.get_rng_state_all()]
            if torch.cuda.is_available()
            else None
        ),
    }


def _restore_state(state):
    torch.set_num_threads(state["threads"])
    np.random.set_state(state["numpy"])
    torch.random.set_rng_state(state["torch"])
    if state["cuda"] is not None:
        torch.cuda.set_rng_state_all(state["cuda"])


def _assert_state_equal(actual, expected):
    assert actual["threads"] == expected["threads"]
    assert actual["numpy"][0] == expected["numpy"][0]
    assert np.array_equal(actual["numpy"][1], expected["numpy"][1])
    assert actual["numpy"][2:] == expected["numpy"][2:]
    assert torch.equal(actual["torch"], expected["torch"])
    if expected["cuda"] is None:
        assert actual["cuda"] is None
    else:
        assert len(actual["cuda"]) == len(expected["cuda"])
        for actual_state, expected_state in zip(
            actual["cuda"],
            expected["cuda"],
        ):
            assert torch.equal(actual_state, expected_state)


def test_branch_test_fixtures_restore_threads_and_rng_state(tmp_path):
    original = _capture_state()
    tests_dir = Path(__file__).resolve().parent
    try:
        torch.set_num_threads(2)
        np.random.seed(991)
        torch.manual_seed(991)
        baseline = _capture_state()
        selected_nodes = [
            (
                "test_advanced_autoencoder_integration.py",
                "test_advanced_autoencoders_are_publicly_exported",
            ),
            (
                "test_autoencoder_hardening.py",
                "test_nonfinite_custom_targets_are_rejected",
            ),
            (
                "test_metrics.py",
                "test_reconstruction_error_numpy_sample_mse_matches_manual",
            ),
            (
                "test_spatial_token_autoencoder.py",
                "test_spatial_token_rejects_non_sequence_input",
            ),
            (
                "test_variational_autoencoder.py",
                "test_vae_kl_has_known_analytic_values",
            ),
        ]
        selected_tests = [
            f"{tests_dir / module_name}::{test_name}"
            for module_name, test_name in selected_nodes
        ]

        passing_result = pytest.main(
            [
                "-q",
                "-p",
                "no:cacheprovider",
                "--basetemp",
                str(tmp_path / "passing-inner-pytest"),
                *selected_tests,
            ]
        )

        assert passing_result == pytest.ExitCode.OK
        _assert_state_equal(_capture_state(), baseline)

        failing_result = pytest.main(
            [
                "-q",
                "-p",
                "no:cacheprovider",
                "--basetemp",
                str(tmp_path / "failing-inner-pytest"),
                *selected_tests,
            ],
            plugins=[_FailSelectedTestCalls()],
        )

        assert failing_result == pytest.ExitCode.TESTS_FAILED
        _assert_state_equal(_capture_state(), baseline)
    finally:
        _restore_state(original)
