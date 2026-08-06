"""Tests for reproducible chronological validation splits."""

from __future__ import annotations

from datetime import UTC, datetime
from fractions import Fraction

import numpy as np
import pandas as pd
import pytest

from bluemath_tk.validation import (
    ChronologicalSplit,
    ValidationSplitManifest,
    apply_split_manifest,
    split_chronologically,
)


def test_default_fraction_split_is_deterministic_and_ordered():
    times = np.arange(20)
    first = split_chronologically(sample_times=times)
    second = split_chronologically(sample_times=times)

    assert first.train_indices.tolist() == list(range(14))
    assert first.validation_indices.tolist() == [14, 15, 16]
    assert first.test_indices.tolist() == [17, 18, 19]
    assert first.excluded_indices.size == 0
    assert first.manifest.to_json() == second.manifest.to_json()
    assert first.counts == {"train": 14, "validation": 3, "test": 3, "excluded": 0}


def test_fraction_split_uses_cumulative_floor_and_test_remainder():
    split = split_chronologically(
        sample_times=np.arange(11),
        fractions=(0.6, 0.2, 0.2),
    )
    assert split.train_indices.tolist() == list(range(6))
    assert split.validation_indices.tolist() == [6, 7]
    assert split.test_indices.tolist() == [8, 9, 10]


def test_explicit_index_boundaries():
    split = split_chronologically(
        sample_times=np.arange(10),
        boundary_indices=(5, 8),
    )
    assert split.train_indices.tolist() == [0, 1, 2, 3, 4]
    assert split.validation_indices.tolist() == [5, 6, 7]
    assert split.test_indices.tolist() == [8, 9]
    assert split.manifest.method == "boundary_indices"


def test_explicit_numeric_time_boundaries():
    split = split_chronologically(
        sample_times=np.arange(0.0, 10.0),
        boundary_times=(5.0, 8.0),
    )
    assert split.train_indices.tolist() == [0, 1, 2, 3, 4]
    assert split.validation_indices.tolist() == [5, 6, 7]
    assert split.test_indices.tolist() == [8, 9]
    assert split.manifest.method == "boundary_times"


def test_datetime64_boundaries_and_manifest_round_trip(tmp_path):
    times = np.arange("2020-01-01", "2020-01-11", dtype="datetime64[D]")
    split = split_chronologically(
        sample_times=times,
        boundary_times=(np.datetime64("2020-01-06"), np.datetime64("2020-01-09")),
    )
    path = tmp_path / "split.json"
    split.manifest.save(path)
    loaded = ValidationSplitManifest.load(path)
    replay = apply_split_manifest(loaded, sample_times=times)

    assert loaded.to_json() == split.manifest.to_json()
    assert replay.train_indices.tolist() == [0, 1, 2, 3, 4]
    assert replay.validation_indices.tolist() == [5, 6, 7]
    assert replay.test_indices.tolist() == [8, 9]


def test_timezone_aware_times_are_normalized_to_utc():
    times = pd.date_range("2020-01-01", periods=10, freq="h", tz="Europe/London")
    split = split_chronologically(
        sample_times=times,
        boundary_times=(times[5], times[8]),
    )
    equivalent = times.tz_convert("UTC")
    replay = apply_split_manifest(split.manifest, sample_times=equivalent)
    assert replay.train_indices.tolist() == list(range(5))


def test_mixed_timezone_awareness_is_rejected():
    values = [
        datetime(2020, 1, 1),
        datetime(2020, 1, 2, tzinfo=UTC),
        datetime(2020, 1, 3, tzinfo=UTC),
    ]
    with pytest.raises(ValueError, match="mixes timezone-aware"):
        split_chronologically(sample_times=values)


def test_complete_interval_policy_excludes_boundary_crossing_windows():
    starts = np.arange(10)
    ends = starts + 2
    split = split_chronologically(
        sample_start_times=starts,
        sample_end_times=ends,
        boundary_indices=(5, 8),
    )
    assert split.train_indices.tolist() == [0, 1, 2]
    assert split.validation_indices.tolist() == [5]
    assert split.test_indices.tolist() == [8, 9]
    assert split.excluded_indices.tolist() == [3, 4, 6, 7]


def test_interval_touching_boundary_is_excluded_from_previous_partition():
    starts = np.array([0, 1, 2, 3, 4, 5])
    ends = np.array([0, 1, 3, 3, 4, 5])
    split = split_chronologically(
        sample_start_times=starts,
        sample_end_times=ends,
        boundary_indices=(3, 5),
    )
    assert 2 in split.excluded_indices
    assert 2 not in split.train_indices


def test_gap_removes_samples_immediately_before_later_partitions():
    split = split_chronologically(
        sample_times=np.arange(12),
        boundary_indices=(6, 9),
        gap=1,
    )
    assert split.train_indices.tolist() == [0, 1, 2, 3, 4]
    assert split.validation_indices.tolist() == [6, 7]
    assert split.test_indices.tolist() == [9, 10, 11]
    assert split.excluded_indices.tolist() == [5, 8]


def test_indices_are_read_only_and_directly_index_multidimensional_data():
    times = np.arange(10)
    data = np.arange(10 * 2 * 3).reshape(10, 2, 3)
    split = split_chronologically(sample_times=times, boundary_indices=(5, 8))
    assert np.array_equal(data[split.train_indices], data[:5])
    assert not split.train_indices.flags.writeable
    with pytest.raises(ValueError):
        split.train_indices[0] = 99


@pytest.mark.parametrize(
    "times,match",
    [
        (np.array([0, 2, 1, 3]), "strictly increasing"),
        (np.array([0, 1, 1, 2]), "strictly increasing"),
        (np.array([0.0, np.nan, 2.0]), "NaN or infinite"),
        (np.array([0.0, np.inf, 2.0]), "NaN or infinite"),
        (np.array([True, False, True]), "Boolean"),
        (np.array(["a", "b", "c"]), "real numeric values or datetime-like"),
    ],
)
def test_invalid_time_coordinates_are_rejected(times, match):
    with pytest.raises((TypeError, ValueError), match=match):
        split_chronologically(sample_times=times)


def test_nat_is_rejected():
    times = np.array(["2020-01-01", "NaT", "2020-01-03"], dtype="datetime64[D]")
    with pytest.raises(ValueError, match="NaT"):
        split_chronologically(sample_times=times)


def test_interval_validation_rejects_missing_mismatched_or_reverse_inputs():
    with pytest.raises(ValueError, match="provided together"):
        split_chronologically(sample_start_times=np.arange(4))
    with pytest.raises(ValueError, match="same length"):
        split_chronologically(
            sample_start_times=np.arange(4),
            sample_end_times=np.arange(3),
        )
    with pytest.raises(ValueError, match="end_time >= start_time"):
        split_chronologically(
            sample_start_times=np.arange(4),
            sample_end_times=np.array([0, 0, 2, 3]),
        )


def test_point_and_interval_forms_are_mutually_exclusive():
    with pytest.raises(ValueError, match="not both forms"):
        split_chronologically(
            sample_times=np.arange(5),
            sample_start_times=np.arange(5),
            sample_end_times=np.arange(5),
        )


@pytest.mark.parametrize(
    "fractions,error_type,match",
    [
        ((0.7, 0.3), ValueError, "exactly"),
        ((0.7, 0.2, 0.2), ValueError, "sum"),
        ((0.0, 0.5, 0.5), ValueError, "strictly"),
        ((True, 0.4, 0.6), TypeError, "Boolean"),
        ((np.nan, 0.5, 0.5), ValueError, "finite"),
    ],
)
def test_invalid_fractions_are_rejected(fractions, error_type, match):
    with pytest.raises(error_type, match=match):
        split_chronologically(sample_times=np.arange(10), fractions=fractions)


def test_multiple_split_specifications_are_rejected():
    with pytest.raises(ValueError, match="only one"):
        split_chronologically(
            sample_times=np.arange(10),
            fractions=(0.6, 0.2, 0.2),
            boundary_indices=(6, 8),
        )


@pytest.mark.parametrize("gap", [-1, True, 1.5, "1"])
def test_invalid_gap_is_rejected(gap):
    with pytest.raises((TypeError, ValueError), match="gap"):
        split_chronologically(sample_times=np.arange(10), gap=gap)


def test_gap_cannot_empty_a_partition():
    with pytest.raises(ValueError, match="removes every validation"):
        split_chronologically(
            sample_times=np.arange(8),
            boundary_indices=(4, 6),
            gap=2,
        )


def test_manifest_rejects_reordered_or_changed_data():
    times = np.arange(10)
    split = split_chronologically(sample_times=times, boundary_indices=(5, 8))
    reordered = times.copy()
    reordered[[0, 1]] = reordered[[1, 0]]
    with pytest.raises(ValueError, match="changed or been reordered"):
        apply_split_manifest(split.manifest, sample_times=reordered)
    changed = times.copy()
    changed[-1] = 99
    with pytest.raises(ValueError, match="changed or been reordered"):
        apply_split_manifest(split.manifest, sample_times=changed)


def test_manifest_rejects_different_time_kind_and_sample_count():
    split = split_chronologically(sample_times=np.arange(10), boundary_indices=(5, 8))
    with pytest.raises(ValueError, match="samples"):
        apply_split_manifest(split.manifest, sample_times=np.arange(11))
    with pytest.raises(ValueError, match="time kind"):
        apply_split_manifest(split.manifest, sample_times=np.arange(10, dtype=float))


def test_manifest_json_is_stable_and_has_newline():
    split = split_chronologically(sample_times=np.arange(10), boundary_indices=(5, 8))
    first = split.manifest.to_json()
    second = ValidationSplitManifest.from_dict(split.manifest.to_dict()).to_json()
    assert first == second
    assert first.endswith("\n")
    assert '"schema_version": 1' in first
    assert '"axis_mode": "point"' in first


def test_manifest_rejects_unknown_or_missing_fields():
    split = split_chronologically(sample_times=np.arange(10), boundary_indices=(5, 8))
    payload = split.manifest.to_dict()
    payload["unknown"] = 1
    with pytest.raises(ValueError, match="unsupported fields"):
        ValidationSplitManifest.from_dict(payload)
    payload = split.manifest.to_dict()
    del payload["method"]
    with pytest.raises(ValueError, match="missing required fields"):
        ValidationSplitManifest.from_dict(payload)


def test_split_does_not_mutate_inputs_or_global_numpy_rng():
    times = np.arange(10)
    original = times.copy()
    np.random.seed(1234)
    before = np.random.get_state()
    split_chronologically(sample_times=times, boundary_indices=(5, 8))
    after = np.random.get_state()
    assert np.array_equal(times, original)
    assert before[0] == after[0]
    assert np.array_equal(before[1], after[1])
    assert before[2:] == after[2:]


def test_boundary_time_kind_must_match_axis_kind():
    with pytest.raises(TypeError, match="integer value"):
        split_chronologically(
            sample_times=np.arange(10),
            boundary_times=(5.0, 8.0),
        )


def test_boundary_order_and_bounds_must_leave_nonempty_partitions():
    with pytest.raises(ValueError, match="must precede"):
        split_chronologically(
            sample_times=np.arange(10),
            boundary_times=(8, 5),
        )
    with pytest.raises(ValueError, match="non-empty"):
        split_chronologically(
            sample_times=np.arange(10),
            boundary_times=(-1, 8),
        )


def test_explicit_boundaries_are_strictly_validated():
    with pytest.raises(TypeError, match="integer index"):
        split_chronologically(sample_times=np.arange(10), boundary_indices=(True, 8))
    with pytest.raises(ValueError, match="less than"):
        split_chronologically(sample_times=np.arange(10), boundary_indices=(8, 5))
    with pytest.raises(ValueError, match="strictly inside"):
        split_chronologically(sample_times=np.arange(10), boundary_indices=(0, 8))


def test_manifest_parameters_are_immutable():
    split = split_chronologically(sample_times=np.arange(10), boundary_indices=(5, 8))
    with pytest.raises(TypeError):
        split.manifest.parameters["gap_samples_before_later_partition"] = 2


def test_timezone_resolution_is_canonical_and_awareness_is_bound():
    aware_us = pd.date_range(
        "2024-01-01",
        periods=10,
        freq="h",
        tz="UTC",
    ).as_unit("us")
    aware_ns = aware_us.as_unit("ns")
    split = split_chronologically(
        sample_times=aware_us,
        boundary_times=(aware_us[5], aware_us[8]),
    )
    replay = apply_split_manifest(split.manifest, sample_times=aware_ns)
    assert replay.train_indices.tolist() == list(range(5))
    assert split.manifest.time_kind == "datetime64[ns]-aware-utc"

    naive = aware_ns.tz_localize(None)
    with pytest.raises(ValueError, match="time kind"):
        apply_split_manifest(split.manifest, sample_times=naive)


def test_datetime_fingerprint_rejects_1970_collision_candidate():
    aware = pd.date_range("2024-01-01", periods=10, freq="h", tz="UTC").as_unit("us")
    split = split_chronologically(sample_times=aware, boundary_indices=(5, 8))
    collision_candidate = pd.to_datetime(aware.asi8, unit="ns", utc=True)
    assert collision_candidate[0].year == 1970
    with pytest.raises(ValueError, match="fingerprint"):
        apply_split_manifest(split.manifest, sample_times=collision_candidate)


def test_datetime_outside_nanosecond_range_is_rejected_without_wraparound():
    times = np.arange(
        np.datetime64("2263-01-01", "us"),
        np.datetime64("2263-01-05", "us"),
        np.timedelta64(1, "D"),
    )
    with pytest.raises(ValueError, match="nanosecond"):
        split_chronologically(sample_times=times)


def test_mixed_aware_interval_endpoints_and_boundaries_are_rejected():
    starts = pd.date_range("2024-01-01", periods=10, freq="h", tz="UTC")
    naive_ends = starts.tz_localize(None) + pd.Timedelta(minutes=30)
    with pytest.raises(TypeError, match="timezone-awareness"):
        split_chronologically(
            sample_start_times=starts,
            sample_end_times=naive_ends,
            boundary_indices=(5, 8),
        )
    with pytest.raises(TypeError, match="expected.*aware"):
        split_chronologically(
            sample_times=starts,
            boundary_times=(
                starts[5].tz_localize(None),
                starts[8].tz_localize(None),
            ),
        )


def test_point_and_interval_modes_are_bound_to_manifest():
    times = np.arange(10)
    split = split_chronologically(sample_times=times, boundary_indices=(5, 8))
    with pytest.raises(ValueError, match="axis mode"):
        apply_split_manifest(
            split.manifest,
            sample_start_times=times,
            sample_end_times=times,
        )


def test_unsigned_values_above_int64_range_are_rejected_without_wraparound():
    values = np.array([2**63, 2**63 + 1, 2**63 + 2, 2**63 + 3], dtype=np.uint64)
    with pytest.raises(ValueError, match="unsigned integers"):
        split_chronologically(sample_times=values)


def test_unsigned_and_signed_coordinate_kinds_do_not_collide():
    unsigned = np.arange(10, dtype=np.uint64)
    split = split_chronologically(sample_times=unsigned, boundary_indices=(5, 8))
    with pytest.raises(ValueError, match="time kind"):
        apply_split_manifest(split.manifest, sample_times=unsigned.astype(np.int64))


@pytest.mark.parametrize(
    "field,value,error",
    [
        ("schema_version", True, TypeError),
        ("schema_version", 1.0, TypeError),
        ("n_samples", True, TypeError),
        ("n_samples", 10.0, TypeError),
    ],
)
def test_manifest_rejects_coercible_scalar_types(field, value, error):
    split = split_chronologically(sample_times=np.arange(10), boundary_indices=(5, 8))
    payload = split.manifest.to_dict()
    payload[field] = value
    with pytest.raises(error):
        ValidationSplitManifest.from_dict(payload)


@pytest.mark.parametrize("value", [True, 0.9, "0", np.float64(0.0)])
def test_manifest_rejects_coercible_index_values(value):
    split = split_chronologically(sample_times=np.arange(10), boundary_indices=(5, 8))
    payload = split.manifest.to_dict()
    payload["train_indices"][0] = value
    with pytest.raises(TypeError, match="exact non-Boolean integer"):
        ValidationSplitManifest.from_dict(payload)


@pytest.mark.parametrize(
    "replacement",
    [
        {},
        {"boundary_indices": [5, 8]},
        {
            "boundary_policy": "complete_interval_half_open",
            "partition_closure": (
                "train:end<b1;validation:start>=b1,end<b2;test:start>=b2"
            ),
            "gap_samples_before_later_partition": 0,
            "boundary_indices": [5, 8],
            "resolved_boundary_values": ["5", "8"],
            "extra": 1,
        },
    ],
)
def test_manifest_rejects_missing_or_extra_parameter_schema(replacement):
    split = split_chronologically(sample_times=np.arange(10), boundary_indices=(5, 8))
    payload = split.manifest.to_dict()
    payload["parameters"] = replacement
    with pytest.raises(ValueError, match="parameters"):
        ValidationSplitManifest.from_dict(payload)


@pytest.mark.parametrize("bad_value", [{1, 2}, np.int64(1), np.nan, np.inf])
def test_manifest_parameters_reject_non_json_or_nonfinite_values(bad_value):
    split = split_chronologically(sample_times=np.arange(10), boundary_indices=(5, 8))
    payload = split.manifest.to_dict()
    payload["parameters"]["gap_samples_before_later_partition"] = bad_value
    with pytest.raises((TypeError, ValueError)):
        ValidationSplitManifest.from_dict(payload)


def test_manifest_fraction_parameters_are_cross_validated():
    split = split_chronologically(
        sample_times=np.arange(20),
        fractions=(0.7, 0.15, 0.15),
    )
    for field, value in (
        ("fractions", [0.6, 0.2, 0.2]),
        ("resolved_boundary_indices", [13, 17]),
        ("gap_samples_before_later_partition", 9),
    ):
        payload = split.manifest.to_dict()
        payload["parameters"][field] = value
        if field == "gap_samples_before_later_partition":
            altered = ValidationSplitManifest.from_dict(payload)
            with pytest.raises(ValueError, match="contradicts"):
                apply_split_manifest(altered, sample_times=np.arange(20))
        else:
            with pytest.raises(ValueError, match="contradict"):
                ValidationSplitManifest.from_dict(payload)


def test_manifest_replay_rejects_shifted_but_structurally_valid_partitions():
    times = np.arange(20)
    split = split_chronologically(sample_times=times)
    payload = split.manifest.to_dict()
    payload["train_indices"] = list(range(13))
    payload["validation_indices"] = list(range(13, 17))
    payload["excluded_indices"] = [13]
    payload["validation_indices"] = list(range(14, 17))
    altered = ValidationSplitManifest.from_dict(payload)
    with pytest.raises(ValueError, match="contradicts"):
        apply_split_manifest(altered, sample_times=times)


def test_strict_json_loader_rejects_nan_and_infinity(tmp_path):
    path = tmp_path / "invalid.json"
    path.write_text('{"schema_version": NaN}', encoding="utf-8")
    with pytest.raises(ValueError, match="Non-standard JSON constant"):
        ValidationSplitManifest.load(path)


def test_direct_chronological_split_requires_a_validated_manifest():
    with pytest.raises(ValueError, match="validated manifest"):
        ChronologicalSplit([0], [1], [2])


@pytest.mark.parametrize(
    "train,validation,test",
    [
        ([1, 0], [2], [3]),
        ([-2, -1], [0], [1]),
        ([0.9], [1.9], [2.9]),
        (["0"], [1], [2]),
    ],
)
def test_direct_split_invalid_indices_cannot_bypass_manifest(train, validation, test):
    with pytest.raises(ValueError, match="validated manifest"):
        ChronologicalSplit(train, validation, test)


@pytest.mark.parametrize(
    "fractions",
    [
        (Fraction(7, 10), Fraction(1, 10), Fraction(1, 5)),
        (Fraction(7, 10), np.float32(0.1), Fraction(1, 5)),
    ],
)
def test_fraction_accepts_fraction_and_numpy_real_scalars(fractions):
    split = split_chronologically(
        sample_times=np.arange(10),
        fractions=fractions,
    )
    assert split.train_indices.tolist() == list(range(7))
    assert split.validation_indices.tolist() == [7]
    assert split.test_indices.tolist() == [8, 9]


@pytest.mark.parametrize(
    "keyword,value",
    [
        ("fractions", 0.5),
        ("fractions", np.array(0.5)),
        ("boundary_indices", np.array(5)),
        ("boundary_times", np.datetime64("2020-01-01")),
    ],
)
def test_scalar_split_specifications_raise_argument_specific_errors(keyword, value):
    kwargs = {keyword: value}
    with pytest.raises(TypeError, match=keyword):
        split_chronologically(sample_times=np.arange(10), **kwargs)


def test_fraction_rounding_and_half_open_touching_rules_are_explicit():
    split = split_chronologically(
        sample_times=np.arange(11),
        fractions=(0.6, 0.2, 0.2),
    )
    assert split.manifest.parameters["rounding_policy"] == "cumulative_floor"
    assert split.manifest.parameters["resolved_boundary_indices"] == (6, 8)
    assert split.validation_indices[0] == 6
    assert split.test_indices[0] == 8

    starts = np.array([0, 1, 2, 3, 4, 5, 6])
    ends = np.array([0, 1, 3, 3, 4, 5, 6])
    interval = split_chronologically(
        sample_start_times=starts,
        sample_end_times=ends,
        boundary_times=(3, 5),
    )
    assert 2 in interval.excluded_indices
    assert 3 in interval.validation_indices
    assert 5 in interval.test_indices


def test_manifest_axis_mode_and_timezone_kind_are_serialized():
    times = pd.date_range("2024-01-01", periods=10, freq="h", tz="UTC")
    split = split_chronologically(sample_times=times, boundary_indices=(5, 8))
    payload = split.manifest.to_dict()
    assert payload["axis_mode"] == "point"
    assert payload["time_kind"] == "datetime64[ns]-aware-utc"


def test_manifest_replay_recomputes_gap_partitions():
    times = np.arange(12)
    split = split_chronologically(
        sample_times=times,
        boundary_indices=(6, 9),
        gap=1,
    )
    payload = split.manifest.to_dict()
    payload["parameters"]["gap_samples_before_later_partition"] = 0
    altered = ValidationSplitManifest.from_dict(payload)
    with pytest.raises(ValueError, match="contradicts"):
        apply_split_manifest(altered, sample_times=times)
