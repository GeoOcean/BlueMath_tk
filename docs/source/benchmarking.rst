PCA and autoencoder reconstruction benchmarking
===============================================

Comparing a PCA baseline against an autoencoder is only meaningful when both
methods are fitted on the same training samples and scored on the same held-out
samples. ``bluemath_tk.benchmarking`` provides that shared infrastructure so
that reconstruction results are comparable rather than accidental.

What the framework measures
---------------------------

The framework measures **reconstruction error on a held-out test partition**,
in the original sample space, for any number of dimensionality-reduction
methods that share one latent budget.

It deliberately does not measure downstream scientific skill. A method with the
lowest reconstruction error is not automatically the best choice for a physical
diagnostic, an extreme-value analysis, or a forecast. The report therefore
contains no ranking and no "best method" field.

Why chronological partitions matter
-----------------------------------

Random splits of a time series place neighbouring, highly correlated samples on
both sides of the split, so a model can reach a low error by memorising almost
identical neighbours. Partition membership therefore always comes from
:func:`~bluemath_tk.validation.chronological.split_chronologically` or from a
replayed :class:`~bluemath_tk.validation.chronological.ValidationSplitManifest`.

The benchmark never creates a split of its own, and it refuses to run without a
manifest-backed :class:`~bluemath_tk.validation.chronological.ChronologicalSplit`.

Basic use
---------

.. code-block:: python

   import numpy as np
   from bluemath_tk.benchmarking import (
       autoencoder_benchmark_method,
       pca_benchmark_method,
       run_reconstruction_benchmark,
   )
   from bluemath_tk.deeplearning.autoencoders import StandardAutoencoder
   from bluemath_tk.validation import split_chronologically

   rng = np.random.default_rng(0)
   times = np.arange("2000-01", "2005-01", dtype="datetime64[M]")
   latent = rng.normal(size=(times.size, 3))
   mixing = rng.normal(size=(3, 12))
   X = (latent @ mixing).reshape(times.size, 3, 4)

   split = split_chronologically(sample_times=times, fractions=(0.6, 0.2, 0.2))

   report = run_reconstruction_benchmark(
       X,
       split=split,
       methods=[
           pca_benchmark_method("pca-k3", n_components=3),
           autoencoder_benchmark_method(
               "standard-ae-k3",
               model_factory=lambda: StandardAutoencoder(k=3, hidden_dims=[32]),
               latent_dimension=3,
               configuration={
                   "architecture": "StandardAutoencoder",
                   "hidden_dims": [32],
               },
               fit_kwargs={"epochs": 50, "batch_size": 16},
           ),
       ],
       metrics=("mse", "mae", "rmse"),
       seed=0,
       sample_times=times,
   )

   for result in report.results:
       print(result.name, result.test_metrics)

Passing ``sample_times`` is optional but recommended. When supplied, the split
manifest is validated against those coordinates, which proves the manifest is
being replayed against the dataset it was created from rather than against a
reordered or different dataset.

The recorded configuration of the autoencoder above describes the **effective**
training run, not only what the caller happened to spell out. Defaults that were
never passed are filled in, so the serialized identity is complete:

.. code-block:: python

   >>> report.results[1].configuration["fit_kwargs"]
   {'batch_size': 16, 'epochs': 50, 'learning_rate': 0.001, 'patience': 20}
   >>> report.results[1].configuration["predict_kwargs"]
   {'batch_size': 64}

Changing ``epochs``, ``learning_rate``, ``batch_size``, or ``patience``
therefore changes ``identity_digest()``. Nothing has to be duplicated by hand
inside ``configuration``.

How PCA and autoencoders are compared
-------------------------------------

Every method is presented to the runner through the same small interface: it is
fitted, then asked to reconstruct samples. Adding a new reconstruction model
later therefore does not change the scientific core of the benchmark.

``PCAReconstruction`` wraps the existing
:class:`bluemath_tk.datamining.pca.PCA` implementation. Samples of shape
``(n_samples, d1, ..., dm)`` are presented as one stacked variable. Stacking and
the inverse reshape both use C order, so sample order and the per-sample shape
survive the round trip unchanged. Impossible component counts are rejected with
an explicit message before scikit-learn is reached.

``AutoencoderReconstruction`` wraps a BlueMath autoencoder and uses only its
accepted public workflow: ``fit`` with explicit chronological validation data
and ``predict`` for reconstruction. Model architectures are never modified.

The two families are inherently different procedures, so a few asymmetries are
unavoidable. Each one is recorded rather than hidden:

* PCA has no early stopping and no validation-driven model selection, so it is
  fitted on the training partition alone and never sees the validation
  partition. Autoencoders receive the validation partition for early stopping
  only. This is recorded per method as ``uses_validation_partition``.
* PCA has a closed-form solution, while autoencoders are fitted by mini-batch
  gradient descent. Training samples are therefore permuted once before an
  autoencoder is fitted, so that mini-batches are not contiguous blocks of
  adjacent timestamps, which would otherwise make batch statistics and gradient
  estimates reflect temporally correlated neighbours. The permutation is drawn
  inside the benchmark's isolated random state, so it is controlled by the run
  ``seed`` and never touches the caller's random state. It reorders training
  rows only: partition membership, and in particular validation membership, is
  unchanged. This is recorded per method as ``shuffle_training_data`` and can be
  disabled with ``shuffle_training_data=False``.

No other asymmetry is introduced. Both families see the same training samples,
neither sees the test partition, and both are scored by the same metric code on
the same test samples.

Supplying a model
-----------------

Methods are supplied as specifications, not as instances. Each specification
carries a **factory** that returns a fresh, unfitted model. A new instance is
built for every run, so fitted state cannot leak between runs, and the runner
rejects a factory that returns an already fitted or already used instance.

Factories are never introspected. Because an arbitrary Python callable cannot be
serialized reproducibly, the reproducible description of a method comes from the
explicit, user-supplied ``method_type`` and ``configuration`` fields, which must
be JSON-compatible and are recorded verbatim.

A method may also be written from scratch. Anything exposing
``latent_dimension``, ``is_fitted``, ``fit(X_train, X_validation)``, and
``reconstruct(X)`` satisfies the ``ReconstructionMethod`` protocol and can be
wrapped in a :class:`~bluemath_tk.benchmarking.BenchmarkMethod`.

Two constructor arguments are refused for autoencoders. ``validation_data`` and
``validation_split`` are refused because the benchmark controls partition
membership. ``optimizer`` and ``criterion`` are refused because a stateful
object built for one model instance would be silently reused by the fresh
instance of the next run; a PyTorch optimizer bound to another model's
parameters skips them without raising, which would report an untrained network
as a legitimate result.

A wrapped autoencoder must also declare ``validation_data`` explicitly in its
``fit`` signature. A ``fit(self, X, **kwargs)`` signature would absorb the
argument silently and remain free to build its own random validation split,
which is exactly the leakage this framework exists to prevent, so it is
rejected.

Everything else in ``fit_kwargs`` and ``predict_kwargs`` must be
JSON-compatible, and is rejected with a clear error otherwise. This is a
deliberate constraint rather than an implementation limit: a training setting
that cannot be recorded cannot form part of a reproducible identity, and
guessing at a ``repr`` of an arbitrary object would produce an identity that
looks precise while meaning nothing.

The recorded values are deep copies. Mutating a nested mapping you passed as
``configuration``, ``fit_kwargs``, or ``predict_kwargs`` after building the
specification changes neither what later runs execute nor what the report
records, and each run receives its own containers.

This automatic recording belongs to ``autoencoder_benchmark_method``. If you
assemble a :class:`~bluemath_tk.benchmarking.BenchmarkMethod` by hand around a
custom method, the ``configuration`` you supply is the whole of what gets
recorded, so it must describe everything that defines the experiment.

Common metrics
--------------

Metrics are computed with the accepted implementation in
:mod:`bluemath_tk.deeplearning.metrics`. The available metrics are ``mse``,
``mae``, and ``rmse``, all reported with ``reduction="mean"``.

Benchmark numbers are therefore bit-identical to a direct call to
``reconstruction_error(y_true, y_pred, metric=..., reduction="mean")``. They
agree with the per-model ``evaluate_reconstruction`` summaries to floating-point
rounding rather than bit for bit, because those summarise per-sample errors
through a different reduction path.

Note that ``rmse`` follows the BlueMath convention: it is the mean over samples
of the per-sample root-mean-square error, which is not the same quantity as the
square root of the reported ``mse``.

Latent dimensionality is not storage compression
------------------------------------------------

Each result reports:

.. code-block:: text

   original_scalars_per_sample
   latent_scalars_per_sample
   latent_dimensionality_ratio

``latent_dimensionality_ratio`` is ``latent_scalars_per_sample`` divided by
``original_scalars_per_sample``. It is a **dimensionality** ratio only.

It is explicitly **not** a bitrate, a storage compression ratio, an entropy
coding result, or a compressed file size, because it ignores latent numeric
precision, quantisation, entropy coding, and the storage cost of the model
parameters themselves. Quantisation and real storage accounting belong to a
later contribution.

How the test partition is kept isolated
---------------------------------------

The test partition never reaches any fitting step:

* PCA is fitted on ``X[split.train_indices]`` only, including the optional
  ``StandardScaler`` statistics when ``scale_data=True``.
* Autoencoder optimisation receives ``X[split.train_indices]`` only.
* Autoencoder validation loss and early stopping receive
  ``X[split.validation_indices]`` only.
* Metrics are computed on ``X[split.test_indices]`` only.

Explicit validation data
~~~~~~~~~~~~~~~~~~~~~~~~

BlueMath autoencoders historically derived their validation set by shuffling
the samples and cutting at ``validation_split``, which cannot express a
chronological validation partition. ``fit`` therefore accepts an explicit
``validation_data`` pair:

.. code-block:: python

   model.fit(X_train, validation_data=(X_validation, None))

When ``validation_data`` is supplied, ``validation_split`` is ignored, all of
``X`` is used for optimisation in the order given, exactly the supplied samples
drive the validation loss and early stopping, and the global NumPy random state
is left untouched. Passing ``y_validation=None`` reconstructs ``X_validation``
itself. The historical ``validation_split`` behaviour is unchanged when
``validation_data`` is omitted.

Preprocessing
~~~~~~~~~~~~~

Domain-specific normalisation is the caller's responsibility in this first
release. The framework applies no shared preprocessing of its own, which
guarantees that no method receives a transformation the others do not. If you
normalise, fit the transformation on the training partition alone and apply the
identical transformation to every compared method.

The one exception is intrinsic to PCA itself: scikit-learn's PCA always centers
the data internally. The additional ``StandardScaler`` step of the BlueMath PCA
is exposed as ``scale_data`` and defaults to ``False``, so it is never applied
silently.

Variational autoencoders
~~~~~~~~~~~~~~~~~~~~~~~~

For a fair comparison, reconstruction must be deterministic. ``predict`` on
:class:`~bluemath_tk.deeplearning.variational_autoencoders.VariationalAutoencoder`
uses the posterior mean by default, and the benchmark adapter rejects
``stochastic=True`` rather than silently comparing one stochastic draw against
deterministic PCA and autoencoder reconstructions.

Reproducibility
---------------

Results are returned as a
:class:`~bluemath_tk.benchmarking.ReconstructionBenchmarkReport`:

.. code-block:: python

   payload = report.to_dict()
   text = report.to_json()
   digest = report.identity_digest()

``to_json`` is strict and deterministic: keys are sorted, NaN and infinity are
rejected, and no timestamps are written.

``identity()`` and ``identity_digest()`` describe *what was compared*: dataset
shape, a digest of the data values, partition sizes, split identity, requested
metrics, seed, and every method specification. They deliberately exclude
measured outcomes. Metric values and wall-clock timings are observational and
are not reproducible bit for bit across machines, library versions, or devices,
so including them in a reproducibility identity would make that identity
meaningless.

Two fingerprints are recorded, and they cover different things:

* ``split_identity["time_axis_fingerprint"]`` comes from the split manifest and
  covers the **time coordinates** only.
* ``data_digest`` covers the **benchmarked values**, their dtype, and their
  shape. It is normalised to C order, so C-ordered and Fortran-ordered copies of
  the same data produce the same digest.

Both are needed: identical time coordinates do not imply identical data.

A report rebuilt with ``from_dict`` is re-validated across fields, not only
field by field, so internally inconsistent scientific metadata is rejected
rather than silently trusted. The partition sizes must sum to ``n_samples``;
train, validation, and test must each be non-empty; ``split_identity`` must
record the same ``n_samples`` as the report and well-formed digests; each
result's ``original_scalars_per_sample`` must equal the product of
``sample_shape``; ``latent_dimension`` must equal ``latent_scalars_per_sample``;
``latent_dimensionality_ratio`` must equal the ratio it claims to be; and no
reconstruction metric may be negative, since MSE, MAE, and RMSE are
non-negative by construction.

Timing and random state
-----------------------

``fit_seconds`` and ``reconstruction_seconds`` come from
:func:`time.perf_counter`. They are observational measurements of one run on one
machine and should not be treated as deterministic benchmark outputs.

Each method is built and fitted inside an isolated random state. The caller's
global NumPy state and PyTorch generator states are restored afterwards. When
``seed`` is supplied, every method starts from the same seeded state. Seeding
makes a run repeatable on the same machine, device, and library versions; it
does not guarantee bitwise-identical PyTorch results across devices, because
algorithm selection and reduction order may differ.

``torch.manual_seed`` reseeds every visible CUDA device, not only the current
one, so every visible CUDA device is forked and restored. When CUDA is
unavailable no device is queried at all, which keeps CPU-only environments free
of any CUDA initialization.

One limit of that isolation is worth stating precisely: Python's standard
library ``random`` module is not isolated, so a model that draws from it is
neither seeded nor restored.

Current limitations
-------------------

* Reconstruction error only. Downstream scientific evaluation is not included.
* One fixed chronological split per run. Rolling-origin, expanding-window, and
  walk-forward validation are not included.
* No hyperparameter optimisation. Architectures and hyperparameters are supplied
  by the caller.
* PCA is benchmarked with an explicit integer component count. The
  explained-variance-ratio mode of :class:`bluemath_tk.datamining.pca.PCA` is not
  exposed here, because a fixed latent budget is what makes the comparison
  against an autoencoder latent dimension meaningful.
* No latent quantisation, entropy coding, bitrate, or storage-size accounting.
* No shared preprocessing, and no dataset-specific loaders or downloads.
* Metrics require the ``deeplearning`` extra, because they are reused from
  :mod:`bluemath_tk.deeplearning.metrics`.
* Partitions are materialised as copies, and each method receives its own copies
  so that a method writing to its inputs cannot affect later methods. Peak
  memory therefore includes two copies of the train, validation, and test
  partitions.
* :class:`bluemath_tk.datamining.pca.PCA` logs at INFO and WARNING level on every
  fit and every transform, and creates a ``logs/`` directory in the working
  directory. A benchmark run surfaces that existing behaviour and offers no way
  to quiet it; ``verbose=0`` applies to autoencoders only.
* ``PCAReconstruction`` does not pass ``random_state`` to scikit-learn. For large
  problems ``svd_solver="auto"`` may select randomized SVD, which draws from the
  global NumPy random state; supply a benchmark ``seed`` if you need that to be
  reproducible.
* ``uses_validation_partition`` is a declaration by the specification, not a
  measurement. It states whether a method is handed the validation partition; it
  cannot verify what the method then does with it.
* The runner cannot verify that ``X`` is ordered by the manifest's time axis
  unless the time coordinates are passed through ``sample_times``,
  ``sample_start_times``, or ``sample_end_times``.
