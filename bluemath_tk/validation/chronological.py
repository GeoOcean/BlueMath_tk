"""Reproducible chronological train, validation, and test splitting."""

from __future__ import annotations

import hashlib
import json
import math
import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from datetime import date, datetime
from fractions import Fraction
from numbers import Integral, Real
from pathlib import Path
from types import MappingProxyType
from typing import Any, TypeAlias

import numpy as np
import pandas as pd

_SCHEMA_VERSION = 2
_FINGERPRINT_SCHEMA = "bluemath-time-axis-v3"
_DEFAULT_FRACTIONS = (0.7, 0.15, 0.15)
_BOUNDARY_POLICY = "complete_interval_half_open"
_PARTITION_CLOSURE = "train:end<b1;validation:start>=b1,end<b2;test:start>=b2"
_ROUNDING_POLICY = "cumulative_floor"
_SUPPORTED_TIME_KINDS = {
    "datetime64[ns]-naive",
    "datetime64[ns]-aware-utc",
    "float64",
    "integer-signed",
    "integer-unsigned",
}

JsonScalar = str | int | float | bool | None
JsonValue = JsonScalar | list["JsonValue"] | dict[str, "JsonValue"]
RealScalar: TypeAlias = Real | np.floating[Any] | np.integer[Any]
CanonicalFraction = tuple[Fraction, str]

_FRACTION_SUM_TOLERANCE = Fraction(1, 10_000_000)
_FIXED_DATETIME_UNIT_TO_NS: dict[str, Fraction] = {
    "W": Fraction(604_800_000_000_000, 1),
    "D": Fraction(86_400_000_000_000, 1),
    "h": Fraction(3_600_000_000_000, 1),
    "m": Fraction(60_000_000_000, 1),
    "s": Fraction(1_000_000_000, 1),
    "ms": Fraction(1_000_000, 1),
    "us": Fraction(1_000, 1),
    "ns": Fraction(1, 1),
    "ps": Fraction(1, 1_000),
    "fs": Fraction(1, 1_000_000),
    "as": Fraction(1, 1_000_000_000),
}


def _is_exact_integer(value: Any) -> bool:
    return isinstance(value, Integral) and not isinstance(value, (bool, np.bool_))


def _validate_index_values(
    values: Sequence[int] | np.ndarray,
    *,
    name: str,
    n_samples: int,
) -> tuple[int, ...]:
    if isinstance(values, (str, bytes)):
        raise TypeError(f"{name} must be a one-dimensional sequence of integers.")
    array = np.asarray(values, dtype=object)
    if array.ndim != 1:
        raise ValueError(f"{name} must be one-dimensional.")
    normalized: list[int] = []
    for value in array.tolist():
        if not _is_exact_integer(value):
            raise TypeError(f"{name} must contain exact non-Boolean integer values.")
        normalized.append(int(value))
    result = tuple(normalized)
    if result != tuple(sorted(result)):
        raise ValueError(f"{name} must be sorted in ascending order.")
    if len(set(result)) != len(result):
        raise ValueError(f"{name} must not contain repeated indices.")
    if any(value < 0 or value >= n_samples for value in result):
        raise ValueError(f"{name} contains an index outside [0, n_samples).")
    return result


def _validate_manifest_index_values(
    values: Any,
    *,
    name: str,
    n_samples: int,
) -> tuple[int, ...]:
    if type(values) not in {list, tuple}:
        raise TypeError(f"{name} must be a JSON-style list or tuple of integers.")
    if any(type(value) is not int for value in values):
        raise TypeError(f"{name} must contain exact built-in integer values.")
    result = tuple(values)
    if result != tuple(sorted(result)):
        raise ValueError(f"{name} must be sorted in ascending order.")
    if len(set(result)) != len(result):
        raise ValueError(f"{name} must not contain repeated indices.")
    if any(value < 0 or value >= n_samples for value in result):
        raise ValueError(f"{name} contains an index outside [0, n_samples).")
    return result


def _readonly_int_array(
    values: Sequence[int] | np.ndarray,
    *,
    name: str,
    n_samples: int,
) -> np.ndarray:
    normalized = _validate_index_values(values, name=name, n_samples=n_samples)
    array = np.array(normalized, dtype=np.int64, copy=True)
    array.setflags(write=False)
    return array


def _validate_json_value(value: Any, *, path: str) -> JsonValue:
    if value is None or type(value) in {str, bool, int}:
        return value
    if type(value) is float:
        if not math.isfinite(value):
            raise ValueError(f"{path} must not contain NaN or infinity.")
        return value
    if type(value) is list:
        return [
            _validate_json_value(item, path=f"{path}[{index}]")
            for index, item in enumerate(value)
        ]
    if type(value) is dict:
        validated: dict[str, JsonValue] = {}
        for key, item in value.items():
            if type(key) is not str:
                raise TypeError(f"{path} must use string object keys.")
            validated[key] = _validate_json_value(item, path=f"{path}.{key}")
        return validated
    raise TypeError(f"{path} contains a non-JSON value of type {type(value).__name__}.")


def _freeze_json(value: JsonValue) -> Any:
    if isinstance(value, dict):
        return MappingProxyType(
            {key: _freeze_json(item) for key, item in value.items()}
        )
    if isinstance(value, list):
        return tuple(_freeze_json(item) for item in value)
    return value


def _thaw_json(value: Any) -> JsonValue:
    if isinstance(value, Mapping):
        return {str(key): _thaw_json(item) for key, item in value.items()}
    if isinstance(value, tuple):
        return [_thaw_json(item) for item in value]
    return value


def _canonical_json(payload: Mapping[str, Any], *, indent: int | None = None) -> str:
    return json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":") if indent is None else None,
        indent=indent,
        ensure_ascii=True,
        allow_nan=False,
    )


def _json_loads_strict(text: str) -> Any:
    def reject_constant(value: str) -> None:
        raise ValueError(f"Non-standard JSON constant {value!r} is not supported.")

    return json.loads(text, parse_constant=reject_constant)


def _require_sequence(value: Any, *, name: str, length: int) -> list[Any]:
    if isinstance(value, (str, bytes)):
        raise TypeError(
            f"{name} must be a one-dimensional sequence of length {length}."
        )
    if isinstance(value, np.ndarray):
        array = np.asarray(value)
    else:
        array = np.asarray(value, dtype=object)
    if array.ndim == 0:
        raise TypeError(
            f"{name} must be a one-dimensional sequence of length {length}."
        )
    if array.ndim != 1:
        raise ValueError(f"{name} must be one-dimensional.")
    if array.size != length:
        raise ValueError(f"{name} must contain exactly {length} values.")
    return [array[index] for index in range(array.size)]


def _validate_gap(gap: Any) -> int:
    if not _is_exact_integer(gap):
        raise TypeError("gap must be a non-negative integer sample count.")
    gap_int = int(gap)
    if gap_int < 0:
        raise ValueError("gap must be non-negative.")
    return gap_int


def _canonicalize_fraction_scalar(value: Any, *, name: str) -> CanonicalFraction:
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, Real):
        raise TypeError(f"{name} must be a finite real number, not Boolean.")
    if isinstance(value, Fraction):
        fraction = value
    elif isinstance(value, np.floating):
        if not bool(np.isfinite(value)):
            raise ValueError(f"{name} must be finite.")
        fraction = Fraction(str(value))
    elif isinstance(value, float):
        if not math.isfinite(value):
            raise ValueError(f"{name} must be finite.")
        fraction = Fraction(str(value))
    elif isinstance(value, Integral):
        fraction = Fraction(int(value), 1)
    else:
        try:
            number = float(value)
        except (TypeError, ValueError, OverflowError) as exc:
            raise TypeError(f"{name} must be a finite real number.") from exc
        if not math.isfinite(number):
            raise ValueError(f"{name} must be finite.")
        fraction = Fraction(str(value))
    if fraction <= 0 or fraction >= 1:
        raise ValueError(f"{name} must be strictly between zero and one.")
    canonical = f"{fraction.numerator}/{fraction.denominator}"
    return fraction, canonical


def _fraction_from_canonical(value: Any, *, name: str) -> Fraction:
    if (
        type(value) is not str
        or re.fullmatch(
            r"[1-9][0-9]*/[1-9][0-9]*",
            value,
        )
        is None
    ):
        raise TypeError(
            f"{name} must be a canonical positive numerator/denominator string."
        )
    numerator_text, denominator_text = value.split("/", maxsplit=1)
    fraction = Fraction(int(numerator_text), int(denominator_text))
    if fraction <= 0 or fraction >= 1:
        raise ValueError(
            f"{name} must represent a value strictly between zero and one."
        )
    if value != f"{fraction.numerator}/{fraction.denominator}":
        raise ValueError(f"{name} must use reduced canonical fraction form.")
    return fraction


def _validate_runtime_fractions(
    fractions: Any,
) -> tuple[tuple[Fraction, Fraction, Fraction], tuple[str, str, str]]:
    values = _require_sequence(fractions, name="fractions", length=3)
    canonicalized = tuple(
        _canonicalize_fraction_scalar(value, name=f"fractions[{index}]")
        for index, value in enumerate(values)
    )
    exact = tuple(item[0] for item in canonicalized)
    canonical = tuple(item[1] for item in canonicalized)
    if abs(sum(exact, start=Fraction(0, 1)) - 1) > _FRACTION_SUM_TOLERANCE:
        raise ValueError("Train, validation, and test fractions must sum to 1.0.")
    return exact, canonical


def _validate_manifest_fractions(
    fractions: Any,
) -> tuple[tuple[Fraction, Fraction, Fraction], tuple[str, str, str]]:
    values = _require_sequence(
        fractions,
        name="parameters.fractions",
        length=3,
    )
    exact = tuple(
        _fraction_from_canonical(
            value,
            name=f"parameters.fractions[{index}]",
        )
        for index, value in enumerate(values)
    )
    if abs(sum(exact, start=Fraction(0, 1)) - 1) > _FRACTION_SUM_TOLERANCE:
        raise ValueError("Manifest fractions must sum to 1.0.")
    return exact, tuple(values)


def _fraction_boundary_indices(
    n_samples: int,
    fractions: tuple[Fraction, Fraction, Fraction],
) -> tuple[int, int]:
    """Resolve cumulative-floor boundaries using exact rational arithmetic."""
    train, validation, _ = fractions
    validation_index = (n_samples * train.numerator) // train.denominator
    cumulative = train + validation
    test_index = (n_samples * cumulative.numerator) // cumulative.denominator
    return validation_index, test_index


def _validate_index_boundary(value: Any, *, name: str, n_samples: int) -> int:
    if not _is_exact_integer(value):
        raise TypeError(f"{name} must be an integer index.")
    index = int(value)
    if index <= 0 or index >= n_samples:
        raise ValueError(f"{name} must lie strictly inside [0, n_samples].")
    return index


def _is_datetime_object(value: Any) -> bool:
    return isinstance(value, (datetime, date, np.datetime64, pd.Timestamp))


def _datetime_is_aware(value: Any) -> bool:
    if isinstance(value, np.datetime64) or (
        isinstance(value, date) and not isinstance(value, datetime)
    ):
        return False
    timestamp = pd.Timestamp(value)
    return timestamp.tzinfo is not None and timestamp.utcoffset() is not None


def _datetime_to_ns(value: Any, *, name: str, aware: bool) -> int:
    try:
        timestamp = pd.Timestamp(value)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError(
            f"{name} contains an invalid datetime value: {value!r}."
        ) from exc
    if pd.isna(timestamp):
        raise ValueError(f"{name} contains NaT values.")
    actual_aware = timestamp.tzinfo is not None and timestamp.utcoffset() is not None
    if actual_aware != aware:
        state = "timezone-aware" if aware else "timezone-naive"
        raise ValueError(f"{name} must contain only {state} datetime values.")
    if aware:
        timestamp = timestamp.tz_convert("UTC")
    try:
        timestamp_ns = timestamp.as_unit("ns", round_ok=False)
        value_ns = int(timestamp_ns.value)
    except (ValueError, OverflowError) as exc:
        raise ValueError(
            f"{name} contains a datetime that cannot be represented exactly at "
            "nanosecond resolution."
        ) from exc
    try:
        round_trip = pd.Timestamp(
            value_ns,
            unit="ns",
            tz="UTC" if aware else None,
        )
    except (ValueError, OverflowError) as exc:
        raise ValueError(
            f"{name} contains a datetime outside the supported nanosecond range."
        ) from exc
    if round_trip != timestamp_ns:
        raise ValueError(
            f"{name} contains a datetime that failed the nanosecond round-trip check."
        )
    return value_ns


def _numpy_datetime_scalar_to_ns(value: np.datetime64, *, name: str) -> int:
    if np.isnat(value):
        raise ValueError(f"{name} contains NaT values.")
    unit, step = np.datetime_data(value.dtype)
    if unit == "generic":
        raise ValueError(f"{name} uses an unsupported generic datetime64 unit.")
    raw = int(value.astype(np.int64))
    if unit in {"Y", "M"}:
        calendar_offset = raw * step
        if unit == "Y":
            year = 1970 + calendar_offset
            month = None
        else:
            year_offset, month_index = divmod(calendar_offset, 12)
            year = 1970 + year_offset
            month = month_index + 1
        if year < 1677 or year > 2262:
            raise ValueError(
                f"{name} contains a datetime outside the supported nanosecond range."
            )
        text = f"{year:04d}" if month is None else f"{year:04d}-{month:02d}"
        return _datetime_to_ns(text, name=name, aware=False)
    scale = _FIXED_DATETIME_UNIT_TO_NS.get(unit)
    if scale is None:
        raise ValueError(f"{name} uses unsupported datetime64 unit {unit!r}.")
    exact_ns = Fraction(raw * step, 1) * scale
    if exact_ns.denominator != 1:
        raise ValueError(
            f"{name} contains a datetime64[{unit}] value that is not an exact "
            "nanosecond multiple."
        )
    value_ns = exact_ns.numerator
    if value_ns < np.iinfo(np.int64).min + 1 or value_ns > np.iinfo(np.int64).max:
        raise ValueError(
            f"{name} contains a datetime outside the supported nanosecond range."
        )
    return int(value_ns)


def _normalize_numpy_datetime_array(
    array: np.ndarray,
    *,
    name: str,
) -> tuple[np.ndarray, str, tuple[str, ...]]:
    integers = np.array(
        [_numpy_datetime_scalar_to_ns(value, name=name) for value in array],
        dtype=np.int64,
    )
    return (
        integers,
        "datetime64[ns]-naive",
        tuple(str(int(value)) for value in integers),
    )


def _normalize_datetime_values(
    items: list[Any],
    *,
    name: str,
) -> tuple[np.ndarray, str, tuple[str, ...]]:
    awareness = [_datetime_is_aware(item) for item in items]
    if any(awareness) and not all(awareness):
        raise ValueError(f"{name} mixes timezone-aware and timezone-naive values.")
    aware = all(awareness)
    normalized: list[int] = []
    for item in items:
        if isinstance(item, np.datetime64):
            if aware:
                raise ValueError(
                    f"{name} mixes timezone-aware values with NumPy datetime64 values."
                )
            normalized.append(_numpy_datetime_scalar_to_ns(item, name=name))
        else:
            normalized.append(_datetime_to_ns(item, name=name, aware=aware))
    integers = np.array(normalized, dtype=np.int64)
    kind = "datetime64[ns]-aware-utc" if aware else "datetime64[ns]-naive"
    return integers, kind, tuple(str(int(value)) for value in integers)


def _normalize_integer_values(
    array: np.ndarray,
    *,
    name: str,
) -> tuple[np.ndarray, str, tuple[str, ...]]:
    unsigned = np.issubdtype(array.dtype, np.unsignedinteger)
    if unsigned:
        maximum = int(np.max(array))
        if maximum > np.iinfo(np.int64).max:
            raise ValueError(
                f"{name} contains unsigned integers outside the supported int64 range."
            )
        kind = "integer-unsigned"
    else:
        kind = "integer-signed"
    integers = np.asarray(array, dtype=np.int64)
    return integers, kind, tuple(str(int(value)) for value in integers)


def _normalize_time_values(
    values: Any,
    *,
    name: str,
) -> tuple[np.ndarray, str, tuple[str, ...]]:
    if isinstance(values, (str, bytes)):
        raise TypeError(f"{name} must be a one-dimensional sequence, not a string.")
    array = np.asarray(values)
    if array.ndim == 0:
        raise TypeError(f"{name} must be a one-dimensional sequence.")
    if array.ndim != 1:
        raise ValueError(f"{name} must be one-dimensional.")
    if array.size == 0:
        raise ValueError(f"{name} must not be empty.")

    if np.issubdtype(array.dtype, np.datetime64):
        return _normalize_numpy_datetime_array(array, name=name)
    if np.issubdtype(array.dtype, np.bool_):
        raise TypeError(f"{name} must not contain Boolean values.")
    if np.issubdtype(array.dtype, np.integer):
        return _normalize_integer_values(array, name=name)
    if np.issubdtype(array.dtype, np.floating):
        floats = np.asarray(array, dtype=np.float64)
        if not np.all(np.isfinite(floats)):
            raise ValueError(f"{name} contains NaN or infinite values.")
        return floats, "float64", tuple(float(value).hex() for value in floats)

    if array.dtype == object:
        items = array.tolist()
        if all(_is_datetime_object(item) for item in items):
            return _normalize_datetime_values(items, name=name)
        if all(_is_exact_integer(item) for item in items):
            if any(int(item) < np.iinfo(np.int64).min for item in items) or any(
                int(item) > np.iinfo(np.int64).max for item in items
            ):
                raise ValueError(
                    f"{name} contains integers outside the supported int64 range."
                )
            unsigned = all(isinstance(item, np.unsignedinteger) for item in items)
            integers = np.array([int(item) for item in items], dtype=np.int64)
            kind = "integer-unsigned" if unsigned else "integer-signed"
            return integers, kind, tuple(str(int(value)) for value in integers)
        if all(
            isinstance(item, Real) and not isinstance(item, (bool, np.bool_))
            for item in items
        ):
            floats = np.asarray([float(item) for item in items], dtype=np.float64)
            if not np.all(np.isfinite(floats)):
                raise ValueError(f"{name} contains NaN or infinite values.")
            return floats, "float64", tuple(float(value).hex() for value in floats)

    raise TypeError(
        f"{name} must contain real numeric values or datetime-like values; "
        f"received dtype {array.dtype}."
    )


@dataclass(frozen=True)
class _TimeAxis:
    starts: np.ndarray
    ends: np.ndarray
    time_kind: str
    axis_mode: str
    canonical_starts: tuple[str, ...]
    canonical_ends: tuple[str, ...]

    @property
    def n_samples(self) -> int:
        return int(self.starts.shape[0])


def _prepare_time_axis(
    *,
    sample_times: Sequence[Any] | np.ndarray | pd.Index | pd.Series | None,
    sample_start_times: Sequence[Any] | np.ndarray | pd.Index | pd.Series | None,
    sample_end_times: Sequence[Any] | np.ndarray | pd.Index | pd.Series | None,
) -> _TimeAxis:
    uses_points = sample_times is not None
    uses_intervals = sample_start_times is not None or sample_end_times is not None
    if uses_points and uses_intervals:
        raise ValueError(
            "Provide sample_times for point samples or both sample_start_times and "
            "sample_end_times for interval samples, not both forms."
        )
    if not uses_points and not uses_intervals:
        raise ValueError(
            "Provide sample_times or both sample_start_times and sample_end_times."
        )
    if uses_intervals and (sample_start_times is None or sample_end_times is None):
        raise ValueError(
            "sample_start_times and sample_end_times must be provided together."
        )

    if uses_points:
        starts, time_kind, canonical_starts = _normalize_time_values(
            sample_times,
            name="sample_times",
        )
        ends = np.array(starts, copy=True)
        canonical_ends = canonical_starts
        axis_mode = "point"
    else:
        starts, start_kind, canonical_starts = _normalize_time_values(
            sample_start_times,
            name="sample_start_times",
        )
        ends, end_kind, canonical_ends = _normalize_time_values(
            sample_end_times,
            name="sample_end_times",
        )
        if start_kind != end_kind:
            raise TypeError(
                "sample_start_times and sample_end_times must use the same time kind "
                "and timezone-awareness state."
            )
        time_kind = start_kind
        axis_mode = "interval"
        if starts.shape != ends.shape:
            raise ValueError(
                "sample_start_times and sample_end_times must have the same length."
            )
        if np.any(ends < starts):
            raise ValueError(
                "Every sample interval must satisfy end_time >= start_time."
            )

    if starts.size < 3:
        raise ValueError("At least three chronological samples are required.")
    if np.any(starts[1:] <= starts[:-1]):
        raise ValueError(
            "Sample start times must be strictly increasing with no duplicates."
        )

    starts_copy = np.array(starts, copy=True)
    ends_copy = np.array(ends, copy=True)
    starts_copy.setflags(write=False)
    ends_copy.setflags(write=False)
    return _TimeAxis(
        starts=starts_copy,
        ends=ends_copy,
        time_kind=time_kind,
        axis_mode=axis_mode,
        canonical_starts=canonical_starts,
        canonical_ends=canonical_ends,
    )


def _fingerprint_time_axis(axis: _TimeAxis) -> str:
    payload = {
        "schema": _FINGERPRINT_SCHEMA,
        "axis_mode": axis.axis_mode,
        "time_kind": axis.time_kind,
        "starts": list(axis.canonical_starts),
        "ends": list(axis.canonical_ends),
    }
    return hashlib.sha256(_canonical_json(payload).encode("utf-8")).hexdigest()


def _normalize_boundary_scalar(
    value: Any,
    *,
    time_kind: str,
    name: str,
) -> tuple[Any, str]:
    if time_kind.startswith("datetime64[ns]"):
        if not _is_datetime_object(value):
            raise TypeError(f"{name} must be a datetime-like value.")
        normalized, scalar_kind, canonical = _normalize_datetime_values(
            [value],
            name=name,
        )
        if scalar_kind != time_kind:
            raise TypeError(
                f"{name} uses time kind {scalar_kind!r}; expected {time_kind!r}."
            )
        return normalized[0], canonical[0]
    if time_kind == "float64":
        if isinstance(value, (bool, np.bool_)) or not isinstance(value, Real):
            raise TypeError(f"{name} must be a finite real number.")
        number = float(value)
        if not math.isfinite(number):
            raise ValueError(f"{name} must be finite.")
        return number, number.hex()
    if time_kind in {"integer-signed", "integer-unsigned"}:
        if not _is_exact_integer(value):
            raise TypeError(f"{name} must be an integer value.")
        number = int(value)
        if number < np.iinfo(np.int64).min or number > np.iinfo(np.int64).max:
            raise ValueError(f"{name} lies outside the supported int64 range.")
        if time_kind == "integer-unsigned" and number < 0:
            raise ValueError(f"{name} must be non-negative for unsigned coordinates.")
        return number, str(number)
    raise ValueError(f"Unsupported time kind {time_kind!r}.")


def _canonical_to_scalar(value: str, *, time_kind: str, name: str) -> Any:
    if type(value) is not str:
        raise TypeError(f"{name} must be a canonical string value.")
    try:
        if time_kind == "float64":
            result = float.fromhex(value)
            if not math.isfinite(result):
                raise ValueError
            return result
        if time_kind in {
            "integer-signed",
            "integer-unsigned",
            "datetime64[ns]-naive",
            "datetime64[ns]-aware-utc",
        }:
            result = int(value)
            if result < np.iinfo(np.int64).min or result > np.iinfo(np.int64).max:
                raise ValueError
            if time_kind == "integer-unsigned" and result < 0:
                raise ValueError
            return result
    except (TypeError, ValueError) as exc:
        raise ValueError(
            f"{name} is not a valid canonical value for {time_kind!r}."
        ) from exc
    raise ValueError(f"Unsupported time kind {time_kind!r}.")


def _base_parameters(gap: int) -> dict[str, JsonValue]:
    return {
        "boundary_policy": _BOUNDARY_POLICY,
        "partition_closure": _PARTITION_CLOSURE,
        "gap_samples_before_later_partition": gap,
    }


def _validate_parameter_keys(
    parameters: dict[str, JsonValue],
    *,
    required: set[str],
) -> None:
    missing = sorted(required.difference(parameters))
    extra = sorted(set(parameters).difference(required))
    if missing:
        raise ValueError(f"Manifest parameters are missing required fields: {missing}.")
    if extra:
        raise ValueError(f"Manifest parameters contain unsupported fields: {extra}.")


def _validate_string_list(value: Any, *, name: str) -> list[str]:
    values = _require_sequence(value, name=name, length=2)
    if any(type(item) is not str for item in values):
        raise TypeError(f"{name} must contain canonical string values.")
    return [str(item) for item in values]


def _validate_manifest_parameters(
    method: str,
    parameters: Mapping[str, Any],
    *,
    n_samples: int,
    time_kind: str,
) -> dict[str, JsonValue]:
    if not isinstance(parameters, Mapping):
        raise TypeError("parameters must be a mapping.")
    raw = dict(parameters)
    validated_json = _validate_json_value(raw, path="parameters")
    if not isinstance(validated_json, dict):
        raise TypeError("parameters must be a JSON object.")
    common = {
        "boundary_policy",
        "partition_closure",
        "gap_samples_before_later_partition",
    }
    if method == "fractions":
        required = common | {
            "fractions",
            "rounding_policy",
            "resolved_boundary_indices",
            "resolved_boundary_values",
        }
    elif method == "boundary_indices":
        required = common | {"boundary_indices", "resolved_boundary_values"}
    elif method == "boundary_times":
        required = common | {"boundary_times"}
    else:
        raise ValueError(f"Unsupported split method: {method!r}.")
    _validate_parameter_keys(validated_json, required=required)

    if validated_json["boundary_policy"] != _BOUNDARY_POLICY:
        raise ValueError("Manifest boundary_policy is unsupported or inconsistent.")
    if validated_json["partition_closure"] != _PARTITION_CLOSURE:
        raise ValueError("Manifest partition_closure is unsupported or inconsistent.")
    gap = _validate_gap(validated_json["gap_samples_before_later_partition"])
    validated_json["gap_samples_before_later_partition"] = gap

    if method == "fractions":
        if validated_json["rounding_policy"] != _ROUNDING_POLICY:
            raise ValueError("Manifest rounding_policy is unsupported.")
        fractions, canonical_fractions = _validate_manifest_fractions(
            validated_json["fractions"]
        )
        validation_index, test_index = _fraction_boundary_indices(
            n_samples,
            fractions,
        )
        resolved = _require_sequence(
            validated_json["resolved_boundary_indices"],
            name="parameters.resolved_boundary_indices",
            length=2,
        )
        if not all(_is_exact_integer(value) for value in resolved):
            raise TypeError(
                "parameters.resolved_boundary_indices must contain exact integers."
            )
        resolved_indices = [int(value) for value in resolved]
        if resolved_indices != [validation_index, test_index]:
            raise ValueError(
                "Manifest resolved_boundary_indices contradict the stored fractions."
            )
        if (
            validation_index <= 0
            or test_index <= validation_index
            or test_index >= n_samples
        ):
            raise ValueError(
                "Manifest fractions do not produce non-empty candidate partitions."
            )
        resolved_values = _validate_string_list(
            validated_json["resolved_boundary_values"],
            name="parameters.resolved_boundary_values",
        )
        for index, value in enumerate(resolved_values):
            _canonical_to_scalar(
                value,
                time_kind=time_kind,
                name=f"parameters.resolved_boundary_values[{index}]",
            )
        validated_json["fractions"] = list(canonical_fractions)
        validated_json["resolved_boundary_indices"] = resolved_indices
        validated_json["resolved_boundary_values"] = resolved_values
    elif method == "boundary_indices":
        values = _require_sequence(
            validated_json["boundary_indices"],
            name="parameters.boundary_indices",
            length=2,
        )
        validation_index = _validate_index_boundary(
            values[0],
            name="parameters.validation_start_index",
            n_samples=n_samples,
        )
        test_index = _validate_index_boundary(
            values[1],
            name="parameters.test_start_index",
            n_samples=n_samples,
        )
        if validation_index >= test_index:
            raise ValueError(
                "Manifest validation_start_index must be less than test_start_index."
            )
        resolved_values = _validate_string_list(
            validated_json["resolved_boundary_values"],
            name="parameters.resolved_boundary_values",
        )
        for index, value in enumerate(resolved_values):
            _canonical_to_scalar(
                value,
                time_kind=time_kind,
                name=f"parameters.resolved_boundary_values[{index}]",
            )
        validated_json["boundary_indices"] = [validation_index, test_index]
        validated_json["resolved_boundary_values"] = resolved_values
    else:
        boundary_values = _validate_string_list(
            validated_json["boundary_times"],
            name="parameters.boundary_times",
        )
        normalized = [
            _canonical_to_scalar(
                value,
                time_kind=time_kind,
                name=f"parameters.boundary_times[{index}]",
            )
            for index, value in enumerate(boundary_values)
        ]
        if normalized[0] >= normalized[1]:
            raise ValueError(
                "Manifest validation_start_time must precede test_start_time."
            )
        validated_json["boundary_times"] = boundary_values
    return validated_json


def _classify_complete_intervals(
    axis: _TimeAxis,
    *,
    validation_boundary: Any,
    test_boundary: Any,
    gap: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    indices = np.arange(axis.n_samples, dtype=np.int64)
    train = indices[axis.ends < validation_boundary]
    validation = indices[
        (axis.starts >= validation_boundary) & (axis.ends < test_boundary)
    ]
    test = indices[axis.starts >= test_boundary]

    if gap:
        if train.size <= gap:
            raise ValueError("gap removes every training sample.")
        if validation.size <= gap:
            raise ValueError("gap removes every validation sample.")
        train = train[:-gap]
        validation = validation[:-gap]

    included = np.concatenate([train, validation, test])
    excluded = np.setdiff1d(indices, included, assume_unique=False)
    if train.size == 0 or validation.size == 0 or test.size == 0:
        raise ValueError(
            "The requested boundaries, interval policy, and gap must leave non-empty "
            "train, validation, and test partitions."
        )
    return train, validation, test, excluded


def _derive_split_from_parameters(
    axis: _TimeAxis,
    *,
    method: str,
    parameters: Mapping[str, Any],
) -> tuple[
    tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray],
    dict[str, JsonValue],
]:
    validated = _validate_manifest_parameters(
        method,
        parameters,
        n_samples=axis.n_samples,
        time_kind=axis.time_kind,
    )
    gap = int(validated["gap_samples_before_later_partition"])
    if method == "fractions":
        validation_index, test_index = validated["resolved_boundary_indices"]
        expected_values = [
            axis.canonical_starts[validation_index],
            axis.canonical_starts[test_index],
        ]
        if validated["resolved_boundary_values"] != expected_values:
            raise ValueError(
                "Manifest resolved boundary values contradict the supplied time axis."
            )
        validation_boundary = axis.starts[validation_index]
        test_boundary = axis.starts[test_index]
    elif method == "boundary_indices":
        validation_index, test_index = validated["boundary_indices"]
        expected_values = [
            axis.canonical_starts[validation_index],
            axis.canonical_starts[test_index],
        ]
        if validated["resolved_boundary_values"] != expected_values:
            raise ValueError(
                "Manifest resolved boundary values contradict the supplied time axis."
            )
        validation_boundary = axis.starts[validation_index]
        test_boundary = axis.starts[test_index]
    else:
        validation_boundary, test_boundary = [
            _canonical_to_scalar(
                value,
                time_kind=axis.time_kind,
                name=f"parameters.boundary_times[{index}]",
            )
            for index, value in enumerate(validated["boundary_times"])
        ]
    partitions = _classify_complete_intervals(
        axis,
        validation_boundary=validation_boundary,
        test_boundary=test_boundary,
        gap=gap,
    )
    return partitions, validated


@dataclass(frozen=True)
class ValidationSplitManifest:
    """Serializable description of a reproducible chronological split."""

    method: str
    n_samples: int
    dataset_fingerprint: str
    time_kind: str
    axis_mode: str
    parameters: Mapping[str, Any]
    train_indices: tuple[int, ...]
    validation_indices: tuple[int, ...]
    test_indices: tuple[int, ...]
    excluded_indices: tuple[int, ...] = ()
    schema_version: int = _SCHEMA_VERSION

    def __post_init__(self) -> None:
        """Validate and freeze manifest fields after dataclass construction."""
        if type(self.schema_version) is not int:
            raise TypeError("schema_version must be an exact non-Boolean integer.")
        if self.schema_version != _SCHEMA_VERSION:
            raise ValueError(
                f"Unsupported split-manifest schema version {self.schema_version}; "
                f"expected {_SCHEMA_VERSION}."
            )
        if type(self.n_samples) is not int:
            raise TypeError("n_samples must be an exact non-Boolean integer.")
        if self.n_samples < 3:
            raise ValueError("n_samples must be at least 3.")
        if type(self.method) is not str:
            raise TypeError("method must be an exact built-in string.")
        if self.method not in {
            "fractions",
            "boundary_indices",
            "boundary_times",
        }:
            raise ValueError(f"Unsupported split method: {self.method!r}.")
        if type(self.axis_mode) is not str:
            raise TypeError("axis_mode must be an exact built-in string.")
        if self.axis_mode not in {"point", "interval"}:
            raise ValueError("axis_mode must be 'point' or 'interval'.")
        if type(self.time_kind) is not str:
            raise TypeError("time_kind must be an exact built-in string.")
        if self.time_kind not in _SUPPORTED_TIME_KINDS:
            raise ValueError(f"Unsupported time_kind: {self.time_kind!r}.")
        if type(self.dataset_fingerprint) is not str:
            raise TypeError("dataset_fingerprint must be an exact built-in string.")
        if re.fullmatch(r"[0-9a-f]{64}", self.dataset_fingerprint) is None:
            raise ValueError(
                "dataset_fingerprint must be a lowercase 64-character SHA-256 "
                "hex digest."
            )

        validated_parameters = _validate_manifest_parameters(
            self.method,
            self.parameters,
            n_samples=self.n_samples,
            time_kind=self.time_kind,
        )
        object.__setattr__(
            self,
            "parameters",
            _freeze_json(validated_parameters),
        )

        partitions = {
            "train_indices": self.train_indices,
            "validation_indices": self.validation_indices,
            "test_indices": self.test_indices,
            "excluded_indices": self.excluded_indices,
        }
        normalized: dict[str, tuple[int, ...]] = {}
        for name, values in partitions.items():
            normalized[name] = _validate_manifest_index_values(
                values,
                name=name,
                n_samples=self.n_samples,
            )
            object.__setattr__(self, name, normalized[name])

        for name in ("train_indices", "validation_indices", "test_indices"):
            if not normalized[name]:
                raise ValueError(f"{name} must not be empty.")
        seen: set[int] = set()
        for name, values in normalized.items():
            overlap = seen.intersection(values)
            if overlap:
                raise ValueError(
                    f"{name} overlaps another partition at indices {sorted(overlap)}."
                )
            seen.update(values)
        if seen != set(range(self.n_samples)):
            missing = sorted(set(range(self.n_samples)).difference(seen))
            raise ValueError(
                "Split manifest must classify every sample as train, validation, test, "
                f"or excluded; missing indices: {missing}."
            )
        if max(self.train_indices) >= min(self.validation_indices):
            raise ValueError("Training indices must precede validation indices.")
        if max(self.validation_indices) >= min(self.test_indices):
            raise ValueError("Validation indices must precede test indices.")

    def to_dict(self) -> dict[str, JsonValue]:
        """Return a JSON-compatible dictionary with deterministic field content."""
        return {
            "schema_version": self.schema_version,
            "method": self.method,
            "n_samples": self.n_samples,
            "dataset_fingerprint": self.dataset_fingerprint,
            "time_kind": self.time_kind,
            "axis_mode": self.axis_mode,
            "parameters": _thaw_json(self.parameters),
            "train_indices": list(self.train_indices),
            "validation_indices": list(self.validation_indices),
            "test_indices": list(self.test_indices),
            "excluded_indices": list(self.excluded_indices),
        }

    def to_json(self, *, indent: int | None = 2) -> str:
        """Serialize the manifest deterministically as strict JSON."""
        return _canonical_json(self.to_dict(), indent=indent) + "\n"

    def save(self, path: str | Path) -> None:
        """Write the manifest to a UTF-8 JSON file."""
        destination = Path(path)
        destination.write_text(self.to_json(indent=2), encoding="utf-8", newline="\n")

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> ValidationSplitManifest:
        """Construct a validated manifest from a dictionary."""
        if not isinstance(payload, Mapping):
            raise TypeError("Manifest payload must be a mapping.")
        if any(type(key) is not str for key in payload):
            raise TypeError("Manifest field names must be exact built-in strings.")
        required = {
            "schema_version",
            "method",
            "n_samples",
            "dataset_fingerprint",
            "time_kind",
            "axis_mode",
            "parameters",
            "train_indices",
            "validation_indices",
            "test_indices",
            "excluded_indices",
        }
        missing = sorted(required.difference(payload))
        extra = sorted(set(payload).difference(required))
        if missing:
            raise ValueError(f"Manifest is missing required fields: {missing}.")
        if extra:
            raise ValueError(f"Manifest contains unsupported fields: {extra}.")
        if type(payload["parameters"]) is not dict:
            raise TypeError("Manifest parameters must be an exact JSON object.")
        for field_name in (
            "train_indices",
            "validation_indices",
            "test_indices",
            "excluded_indices",
        ):
            if type(payload[field_name]) is not list:
                raise TypeError(f"Manifest {field_name} must be an exact JSON list.")
        _validate_json_value(dict(payload), path="manifest")
        return cls(**{key: payload[key] for key in required})

    @classmethod
    def load(cls, path: str | Path) -> ValidationSplitManifest:
        """Load and validate a manifest from a UTF-8 JSON file."""
        source = Path(path)
        try:
            payload = _json_loads_strict(source.read_text(encoding="utf-8"))
        except json.JSONDecodeError as exc:
            raise ValueError(f"Invalid split-manifest JSON: {exc}.") from exc
        if not isinstance(payload, dict):
            raise ValueError("Split-manifest JSON must contain one object.")
        return cls.from_dict(payload)

    def to_split(self) -> ChronologicalSplit:
        """Return immutable NumPy index arrays for this manifest."""
        return ChronologicalSplit(
            train_indices=self.train_indices,
            validation_indices=self.validation_indices,
            test_indices=self.test_indices,
            excluded_indices=self.excluded_indices,
            manifest=self,
        )

    def validate_against(
        self,
        *,
        sample_times: Sequence[Any] | np.ndarray | pd.Index | pd.Series | None = None,
        sample_start_times: (
            Sequence[Any] | np.ndarray | pd.Index | pd.Series | None
        ) = None,
        sample_end_times: (
            Sequence[Any] | np.ndarray | pd.Index | pd.Series | None
        ) = None,
    ) -> None:
        """Reject changed data and manifests inconsistent with their parameters."""
        try:
            axis = _prepare_time_axis(
                sample_times=sample_times,
                sample_start_times=sample_start_times,
                sample_end_times=sample_end_times,
            )
        except (TypeError, ValueError) as exc:
            raise ValueError(
                "The supplied time coordinates are incompatible with the manifest. "
                "The dataset may have changed or been reordered: "
                f"{exc}"
            ) from exc
        if axis.n_samples != self.n_samples:
            raise ValueError(
                f"Manifest expects {self.n_samples} samples, received {axis.n_samples}."
            )
        if axis.time_kind != self.time_kind:
            raise ValueError(
                f"Manifest expects time kind {self.time_kind!r}, "
                f"received {axis.time_kind!r}."
            )
        if axis.axis_mode != self.axis_mode:
            raise ValueError(
                f"Manifest expects axis mode {self.axis_mode!r}, "
                f"received {axis.axis_mode!r}."
            )
        if _fingerprint_time_axis(axis) != self.dataset_fingerprint:
            raise ValueError(
                "The supplied time coordinates do not match the manifest fingerprint. "
                "The dataset may have changed or been reordered."
            )
        try:
            derived, _ = _derive_split_from_parameters(
                axis,
                method=self.method,
                parameters=_thaw_json(self.parameters),
            )
        except (TypeError, ValueError) as exc:
            raise ValueError(
                "Manifest state contradicts the supplied time axis or do not "
                f"produce a valid split: {exc}"
            ) from exc
        names = (
            "train_indices",
            "validation_indices",
            "test_indices",
            "excluded_indices",
        )
        for name, values in zip(names, derived):
            stored = tuple(getattr(self, name))
            recomputed = tuple(int(value) for value in values)
            if stored != recomputed:
                raise ValueError(
                    f"Manifest {name} contradicts its parameters and time axis."
                )


@dataclass(frozen=True)
class ChronologicalSplit:
    """Immutable index partitions backed by a validated manifest."""

    train_indices: np.ndarray
    validation_indices: np.ndarray
    test_indices: np.ndarray
    excluded_indices: np.ndarray = field(
        default_factory=lambda: np.empty(0, dtype=np.int64)
    )
    manifest: ValidationSplitManifest | None = field(
        default=None,
        repr=False,
        compare=False,
    )

    def __post_init__(self) -> None:
        """Normalize split indices and verify partition consistency."""
        if self.manifest is None:
            raise ValueError(
                "ChronologicalSplit must be constructed from a validated manifest."
            )
        names = (
            "train_indices",
            "validation_indices",
            "test_indices",
            "excluded_indices",
        )
        for name in names:
            object.__setattr__(
                self,
                name,
                _readonly_int_array(
                    getattr(self, name),
                    name=name,
                    n_samples=self.manifest.n_samples,
                ),
            )
        for name in ("train_indices", "validation_indices", "test_indices"):
            if getattr(self, name).size == 0:
                raise ValueError(f"{name} must not be empty.")
        combined = np.concatenate(
            [
                self.train_indices,
                self.validation_indices,
                self.test_indices,
                self.excluded_indices,
            ]
        )
        if np.unique(combined).size != combined.size:
            raise ValueError(
                "Split partitions must not overlap or contain repeated indices."
            )
        if set(combined.tolist()) != set(range(self.manifest.n_samples)):
            raise ValueError("Split partitions must classify every manifest sample.")
        if self.train_indices[-1] >= self.validation_indices[0]:
            raise ValueError("Training indices must precede validation indices.")
        if self.validation_indices[-1] >= self.test_indices[0]:
            raise ValueError("Validation indices must precede test indices.")
        expected = self.manifest.to_dict()
        for name in names:
            if getattr(self, name).tolist() != expected[name]:
                raise ValueError(f"{name} does not match the attached manifest.")

    @property
    def counts(self) -> Mapping[str, int]:
        """Return immutable partition counts."""
        return MappingProxyType(
            {
                "train": int(self.train_indices.size),
                "validation": int(self.validation_indices.size),
                "test": int(self.test_indices.size),
                "excluded": int(self.excluded_indices.size),
            }
        )


def split_chronologically(
    *,
    sample_times: Sequence[Any] | np.ndarray | pd.Index | pd.Series | None = None,
    sample_start_times: Sequence[Any] | np.ndarray | pd.Index | pd.Series | None = None,
    sample_end_times: Sequence[Any] | np.ndarray | pd.Index | pd.Series | None = None,
    fractions: Sequence[RealScalar] | None = None,
    boundary_indices: Sequence[int] | None = None,
    boundary_times: Sequence[Any] | None = None,
    gap: int = 0,
) -> ChronologicalSplit:
    """Create a deterministic chronological split and reproducibility manifest.

    Exactly one split specification may be supplied. When none is supplied, the
    fractions are ``(0.7, 0.15, 0.15)``. Fraction boundaries are resolved with
    ``floor(n * train_fraction)`` and
    ``floor(n * (train_fraction + validation_fraction))``; the test partition gets
    the remaining samples.

    Point partitions are half-open: train uses ``time < b1``, validation uses
    ``b1 <= time < b2``, and test uses ``time >= b2``. For interval samples, train
    requires ``end < b1``, validation requires ``start >= b1`` and ``end < b2``,
    and test requires ``start >= b2``. Therefore an interval ending exactly on a
    boundary is excluded. ``gap`` removes the final ``gap`` samples from train and
    validation after interval classification.
    """
    axis = _prepare_time_axis(
        sample_times=sample_times,
        sample_start_times=sample_start_times,
        sample_end_times=sample_end_times,
    )
    gap_int = _validate_gap(gap)
    specifications = sum(
        specification is not None
        for specification in (fractions, boundary_indices, boundary_times)
    )
    if specifications > 1:
        raise ValueError(
            "Provide only one of fractions, boundary_indices, or boundary_times."
        )
    if specifications == 0:
        fractions = _DEFAULT_FRACTIONS

    parameters = _base_parameters(gap_int)
    if fractions is not None:
        validated_fractions, canonical_fractions = _validate_runtime_fractions(
            fractions
        )
        validation_index, test_index = _fraction_boundary_indices(
            axis.n_samples,
            validated_fractions,
        )
        if (
            validation_index <= 0
            or test_index <= validation_index
            or test_index >= axis.n_samples
        ):
            raise ValueError(
                "Fractions do not produce non-empty candidate partitions for this "
                "dataset."
            )
        method = "fractions"
        parameters.update(
            {
                "fractions": list(canonical_fractions),
                "rounding_policy": _ROUNDING_POLICY,
                "resolved_boundary_indices": [validation_index, test_index],
                "resolved_boundary_values": [
                    axis.canonical_starts[validation_index],
                    axis.canonical_starts[test_index],
                ],
            }
        )
    elif boundary_indices is not None:
        values = _require_sequence(
            boundary_indices,
            name="boundary_indices",
            length=2,
        )
        validation_index = _validate_index_boundary(
            values[0],
            name="validation_start_index",
            n_samples=axis.n_samples,
        )
        test_index = _validate_index_boundary(
            values[1],
            name="test_start_index",
            n_samples=axis.n_samples,
        )
        if validation_index >= test_index:
            raise ValueError(
                "validation_start_index must be less than test_start_index."
            )
        method = "boundary_indices"
        parameters.update(
            {
                "boundary_indices": [validation_index, test_index],
                "resolved_boundary_values": [
                    axis.canonical_starts[validation_index],
                    axis.canonical_starts[test_index],
                ],
            }
        )
    else:
        values = _require_sequence(
            boundary_times,
            name="boundary_times",
            length=2,
        )
        validation_boundary, validation_canonical = _normalize_boundary_scalar(
            values[0],
            time_kind=axis.time_kind,
            name="validation_start_time",
        )
        test_boundary, test_canonical = _normalize_boundary_scalar(
            values[1],
            time_kind=axis.time_kind,
            name="test_start_time",
        )
        if validation_boundary >= test_boundary:
            raise ValueError("validation_start_time must precede test_start_time.")
        method = "boundary_times"
        parameters["boundary_times"] = [
            validation_canonical,
            test_canonical,
        ]

    partitions, validated_parameters = _derive_split_from_parameters(
        axis,
        method=method,
        parameters=parameters,
    )
    train, validation, test, excluded = partitions
    manifest = ValidationSplitManifest(
        method=method,
        n_samples=axis.n_samples,
        dataset_fingerprint=_fingerprint_time_axis(axis),
        time_kind=axis.time_kind,
        axis_mode=axis.axis_mode,
        parameters=validated_parameters,
        train_indices=tuple(int(value) for value in train),
        validation_indices=tuple(int(value) for value in validation),
        test_indices=tuple(int(value) for value in test),
        excluded_indices=tuple(int(value) for value in excluded),
    )
    return manifest.to_split()


def apply_split_manifest(
    manifest: ValidationSplitManifest,
    *,
    sample_times: Sequence[Any] | np.ndarray | pd.Index | pd.Series | None = None,
    sample_start_times: Sequence[Any] | np.ndarray | pd.Index | pd.Series | None = None,
    sample_end_times: Sequence[Any] | np.ndarray | pd.Index | pd.Series | None = None,
) -> ChronologicalSplit:
    """Validate current time coordinates and reproduce a saved split exactly."""
    if not isinstance(manifest, ValidationSplitManifest):
        raise TypeError("manifest must be a ValidationSplitManifest instance.")
    manifest.validate_against(
        sample_times=sample_times,
        sample_start_times=sample_start_times,
        sample_end_times=sample_end_times,
    )
    return manifest.to_split()
