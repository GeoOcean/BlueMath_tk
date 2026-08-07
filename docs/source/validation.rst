Reproducible chronological validation
=====================================

Random train/validation/test splits can leak future information into climate and
other time-series experiments. BlueMath_tk therefore provides a chronological
splitter that returns reusable indices rather than copying large data arrays.

Basic use
---------

.. code-block:: python

   import numpy as np
   from bluemath_tk.validation import split_chronologically

   times = np.arange("2000-01", "2020-01", dtype="datetime64[M]")
   split = split_chronologically(
       sample_times=times,
       fractions=(0.7, 0.15, 0.15),
   )

   X_train = X[split.train_indices]
   X_validation = X[split.validation_indices]
   X_test = X[split.test_indices]

The returned index arrays are read-only and can index NumPy-compatible arrays
with any trailing dimensions, including ``(n_samples, T, C, H, W)`` data.

Fraction rounding
-----------------

Fraction splits use cumulative flooring. For ``n`` samples and fractions
``(f_train, f_validation, f_test)``, the validation and test start indices are:

.. code-block:: text

   validation_start = floor(n * f_train)
   test_start = floor(n * (f_train + f_validation))

The test partition receives the remainder. Fractions must be positive finite
real scalars, must sum to one within numerical tolerance, and must leave all
three candidate partitions non-empty. Boundary resolution preserves an exact
canonical rational for every scalar. ``fractions.Fraction`` values retain their
exact numerator and denominator. Python and NumPy floating scalars use their
round-trip decimal spelling before conversion to a rational. Homogeneous NumPy
arrays retain their original scalar dtype during sequence validation, so common
values such as ``np.array([0.7, 0.1, 0.2], dtype=np.float32)`` resolve to
``(7/10, 1/10, 1/5)`` without an intermediate Python ``float`` conversion. The
manifest stores reduced ``numerator/denominator`` strings and therefore replays
with the same arithmetic after JSON round trips.

Explicit boundaries
-------------------

Use either two start indices or two time values. The first boundary starts the
validation partition and the second starts the test partition.

.. code-block:: python

   split = split_chronologically(
       sample_times=times,
       boundary_times=(np.datetime64("2014-01"), np.datetime64("2017-01")),
   )

Only one split specification may be supplied: fractions, boundary indices, or
boundary times.

Half-open boundary rules
------------------------

For point samples with boundaries ``b1`` and ``b2``:

* train: ``time < b1``;
* validation: ``b1 <= time < b2``;
* test: ``time >= b2``.

A point exactly on a boundary therefore joins the later partition.

For interval samples:

* train: ``end < b1``;
* validation: ``start >= b1`` and ``end < b2``;
* test: ``start >= b2``.

An interval ending exactly on a boundary is not assigned to the preceding
partition. If it started before that boundary, it crosses the boundary and is
recorded in ``excluded_indices``.

Windowed samples and leakage prevention
---------------------------------------

For samples representing complete source intervals, supply both start and end
coordinates:

.. code-block:: python

   split = split_chronologically(
       sample_start_times=window_starts,
       sample_end_times=window_ends,
       boundary_times=(validation_start, test_start),
       gap=2,
   )

A window is assigned only when its complete interval lies within one partition.
A window crossing a boundary is recorded in ``excluded_indices`` and is never
silently assigned by its target or final time.

``gap`` is a non-negative sample count. After interval classification, it
removes the final ``gap`` samples from the training partition and the final
``gap`` samples from the validation partition. Those samples are added to
``excluded_indices`` immediately before the later partitions.

Datetime and numeric coordinates
--------------------------------

Timezone-aware datetime values are converted explicitly to UTC and represented
at checked nanosecond resolution. Timezone awareness remains part of the
manifest and fingerprint, so an aware axis cannot be replayed as a naive axis.
Values outside the supported nanosecond range, or values that cannot round-trip
exactly at that resolution, are rejected rather than wrapped or truncated.
NumPy ``datetime64`` arrays retain their dtype unit during normalization.
Sub-nanosecond units such as picoseconds and femtoseconds are accepted only
when every value is an exact nanosecond multiple; otherwise the split is
rejected. Calendar units, including stepped year and month dtypes, are checked
with arbitrary-precision offsets before NumPy renders a date, so extreme counts
cannot wrap into ordinary in-range dates. Equivalent exact
picosecond/nanosecond axes therefore share a fingerprint, while scaled,
truncated, or wrapped axes do not.

Timezone-naive datetime values remain distinct from timezone-aware values.
Interval starts, ends, and explicit time boundaries must use the same awareness
state. Unsigned integer coordinates are supported only within the signed
64-bit range and remain distinguishable from signed integer coordinates in the
manifest fingerprint.

Reproducibility manifests
-------------------------

Every split includes a deterministic manifest:

.. code-block:: python

   split.manifest.save("split_manifest.json")

   from bluemath_tk.validation import (
       ValidationSplitManifest,
       apply_split_manifest,
   )

   manifest = ValidationSplitManifest.load("split_manifest.json")
   reproduced = apply_split_manifest(manifest, sample_times=times)

The manifest stores the method, strict method-specific parameters, resolved
boundaries, gap, partition indices, excluded indices, point-or-interval mode,
time-coordinate kind, sample count, schema version, and a SHA-256 fingerprint
of the ordered sample coordinates.

Manifest JSON is strict and deterministic. NaN, infinity, non-JSON values,
unknown parameters, missing parameters, coercible non-integer indices, and
unsupported schema versions are rejected. During replay, BlueMath_tk recomputes
the split from the stored parameters and supplied coordinates and compares all
partitions. Changed, reordered, or contradictory data therefore fail clearly.
SHA-256 digests are stored as exactly 64 lowercase hexadecimal characters.

Preprocessing
-------------

The splitter returns indices only. Fit scalers, PCA, or other preprocessing
objects using ``train_indices`` and apply those fitted objects to validation and
test data. This module does not automatically fit preprocessing and therefore
does not introduce validation or test information into training.

Current limitations
-------------------

This first API implements one deterministic chronological split. Rolling-origin,
expanding-window, event-based, and grouped station/site validation are not yet
included. The gap is measured in samples rather than elapsed time. Time inputs
must be strictly increasing, and duplicate sample start times are rejected.
