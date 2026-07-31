"""Regression tests for shared autoencoder hardening."""

import copy

import numpy as np
import pytest
import torch
import torch.nn as nn

from bluemath_tk.deeplearning._base_model import BaseDeepLearningModel
from bluemath_tk.deeplearning.autoencoders import (
    CNNAutoencoder,
    ConvLSTMAutoencoder,
    HybridConvLSTMTransformerAutoencoder,
    LSTMAutoencoder,
    OrthogonalAutoencoder,
    SpatialTokenConvLSTMTransformerAutoencoder,
    StandardAutoencoder,
    VariationalAutoencoder,
    VisionTransformerAutoencoder,
)
from bluemath_tk.deeplearning.layers import LinearSelfAttention


@pytest.fixture(autouse=True)
def _set_seed():
    previous_threads = torch.get_num_threads()
    numpy_state = np.random.get_state()
    torch_state = torch.random.get_rng_state()
    cuda_states = torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None
    try:
        np.random.seed(607)
        torch.manual_seed(607)
        torch.set_num_threads(1)
        yield
    finally:
        torch.set_num_threads(previous_threads)
        np.random.set_state(numpy_state)
        torch.random.set_rng_state(torch_state)
        if cuda_states is not None:
            torch.cuda.set_rng_state_all(cuda_states)


class _TinyAutoencoder(BaseDeepLearningModel):
    def __init__(self, device="cpu"):
        self.mutable_config = {
            "encoder": {"widths": [2, 1]},
            "decoder": {"widths": [2, 1]},
            "flags": {"transactional", "tiny"},
        }
        super().__init__(device=device)

    def _build_model(self, input_shape, **kwargs):
        sample_shape = tuple(input_shape[1:])

        class _Model(nn.Module):
            def __init__(self):
                super().__init__()
                self.anchor = nn.Parameter(torch.zeros(()))
                self.tail = nn.Parameter(torch.zeros(2))
                self.register_buffer("running_value", torch.zeros(()))

            def forward(self, x):
                return torch.zeros_like(x) + self.anchor * 0

            def encode_forward(self, x):
                return x.reshape(len(x), -1)

            def decode_forward(self, z):
                return z.reshape(len(z), *sample_shape)

        return _Model()


def _find_mutable_objects(value, path=(), active_ids=frozenset()):
    mutable_objects = {}
    if isinstance(value, (dict, list, set)):
        mutable_objects[path] = value
        value_id = id(value)
        if value_id in active_ids:
            return mutable_objects
        active_ids = active_ids | {value_id}

    if isinstance(value, dict):
        for key, nested_value in value.items():
            mutable_objects.update(
                _find_mutable_objects(
                    nested_value,
                    path + (("key", key),),
                    active_ids,
                )
            )
    elif isinstance(value, (list, tuple)):
        for index, nested_value in enumerate(value):
            mutable_objects.update(
                _find_mutable_objects(
                    nested_value,
                    path + (("index", index),),
                    active_ids,
                )
            )

    return mutable_objects


def _snapshot_model(model):
    metadata = {
        name: value
        for name, value in model.__dict__.items()
        if name not in {"_logger", "model"}
    }
    mutable_metadata = {}
    for name, value in metadata.items():
        mutable_metadata.update(
            _find_mutable_objects(value, path=(("attribute", name),))
        )

    return {
        "model": model.model,
        "logger": model.__dict__["_logger"],
        "training": model.model.training if model.model is not None else None,
        "state": {
            name: value.detach().clone()
            for name, value in (
                model.model.state_dict().items() if model.model is not None else []
            )
        },
        "metadata_keys": set(model.__dict__),
        "metadata": copy.deepcopy(metadata),
        "mutable_metadata": mutable_metadata,
    }


def _assert_model_unchanged(model, snapshot):
    assert set(model.__dict__) == snapshot["metadata_keys"]
    assert model.model is snapshot["model"]
    assert model.__dict__["_logger"] is snapshot["logger"]
    for name, value in snapshot["metadata"].items():
        assert model.__dict__[name] == value

    mutable_metadata = {}
    for name in snapshot["metadata"]:
        mutable_metadata.update(
            _find_mutable_objects(
                model.__dict__[name],
                path=(("attribute", name),),
            )
        )
    assert mutable_metadata.keys() == snapshot["mutable_metadata"].keys()
    for path, value in snapshot["mutable_metadata"].items():
        assert mutable_metadata[path] is value

    if model.model is None:
        assert snapshot["training"] is None
        assert not snapshot["state"]
        return
    assert model.model.training == snapshot["training"]
    current_state = model.model.state_dict()
    assert current_state.keys() == snapshot["state"].keys()
    for name, value in snapshot["state"].items():
        assert torch.equal(current_state[name], value)


def test_unchanged_state_helper_rejects_new_metadata_and_nested_aliases():
    model = _TinyAutoencoder()
    snapshot = _snapshot_model(model)

    try:
        model.checkpoint_version = "unexpected-derived-metadata"
        with pytest.raises(AssertionError):
            _assert_model_unchanged(model, snapshot)
    finally:
        del model.checkpoint_version

    _assert_model_unchanged(model, snapshot)
    original_decoder_widths = model.mutable_config["decoder"]["widths"]
    try:
        model.mutable_config["decoder"]["widths"] = model.mutable_config["encoder"][
            "widths"
        ]
        assert model.mutable_config == snapshot["metadata"]["mutable_config"]
        with pytest.raises(AssertionError):
            _assert_model_unchanged(model, snapshot)
    finally:
        model.mutable_config["decoder"]["widths"] = original_decoder_widths

    _assert_model_unchanged(model, snapshot)


def _legacy_factories_and_inputs():
    return [
        (
            lambda: StandardAutoencoder(
                k=2,
                hidden_dims=[6, 4],
                device="cpu",
            ),
            np.zeros((8, 3), dtype="float32"),
        ),
        (
            lambda: OrthogonalAutoencoder(
                k=2,
                hidden_dims=[6, 4],
                device="cpu",
            ),
            np.zeros((8, 3), dtype="float32"),
        ),
        (
            lambda: LSTMAutoencoder(
                k=2,
                hidden=(5, 4),
                device="cpu",
            ),
            np.zeros((8, 3, 2), dtype="float32"),
        ),
        (
            lambda: CNNAutoencoder(k=2, device="cpu"),
            np.zeros((8, 1, 4, 4), dtype="float32"),
        ),
        (
            lambda: VisionTransformerAutoencoder(
                k=2,
                patch_size=2,
                d_model=4,
                depth_enc=1,
                depth_dec=1,
                heads=1,
                device="cpu",
            ),
            np.zeros((8, 1, 4, 4), dtype="float32"),
        ),
        (
            lambda: ConvLSTMAutoencoder(k=2, device="cpu"),
            np.zeros((8, 2, 1, 4, 4), dtype="float32"),
        ),
        (
            lambda: HybridConvLSTMTransformerAutoencoder(
                k=2,
                d_model=4,
                n_heads=1,
                n_layers=1,
                efficient_attention=None,
                device="cpu",
            ),
            np.zeros((8, 2, 1, 4, 4), dtype="float32"),
        ),
        (
            lambda: VariationalAutoencoder(
                k=2,
                hidden_dims=[6, 4],
                device="cpu",
            ),
            np.zeros((8, 3), dtype="float32"),
        ),
        (
            lambda: SpatialTokenConvLSTMTransformerAutoencoder(
                k=2,
                spatial_pool_size=(1, 1),
                d_model=4,
                n_heads=1,
                n_layers=1,
                device="cpu",
            ),
            np.zeros((8, 2, 1, 4, 4), dtype="float32"),
        ),
    ]


def _learning_rate_factories_and_inputs():
    return [
        (
            lambda: StandardAutoencoder(
                k=2,
                hidden_dims=[4],
                device="cpu",
            ),
            np.zeros((8, 3), dtype="float32"),
        ),
        (
            lambda: OrthogonalAutoencoder(
                k=2,
                hidden_dims=[4],
                lambda_W=0.0,
                lambda_Z=0.0,
                device="cpu",
            ),
            np.zeros((8, 3), dtype="float32"),
        ),
        (
            lambda: VariationalAutoencoder(
                k=2,
                hidden_dims=[4],
                beta=0.1,
                device="cpu",
            ),
            np.zeros((8, 3), dtype="float32"),
        ),
    ]


def _unbuilt_learning_rate_factories_and_inputs():
    return [
        (
            lambda: _TinyAutoencoder(device="cpu"),
            np.zeros((8, 3), dtype="float32"),
        ),
        (
            lambda: OrthogonalAutoencoder(
                k=2,
                hidden_dims=[4],
                lambda_W=0.0,
                lambda_Z=0.0,
                device="cpu",
            ),
            np.zeros((8, 3), dtype="float32"),
        ),
        (
            lambda: VariationalAutoencoder(
                k=2,
                hidden_dims=[4],
                beta=0.1,
                device="cpu",
            ),
            np.zeros((8, 3), dtype="float32"),
        ),
    ]


def _assert_unbuilt_training_state(model):
    assert model.model is None
    assert model._build_input_shape is None
    assert model.is_fitted is False
    for name in (
        "optimizer",
        "_optimizer",
        "history",
        "_history",
        "training_history",
        "best_model_state",
        "_best_model_state",
    ):
        assert name not in model.__dict__


@pytest.mark.parametrize(
    "learning_rate",
    [
        True,
        False,
        -1.0,
        float("nan"),
        float("inf"),
        -float("inf"),
        "1e-3",
        1e-3 + 0j,
        [1e-3],
        np.array([1e-3]),
        None,
    ],
)
@pytest.mark.parametrize(
    ("factory", "X"),
    _learning_rate_factories_and_inputs(),
)
def test_invalid_learning_rates_do_not_mutate_fit_state(
    factory,
    X,
    learning_rate,
):
    model = factory()
    model._build_input_shape = tuple(X.shape)
    model.model = model._build_model(X.shape).to(model.device)
    model.is_fitted = True
    snapshot = _snapshot_model(model)

    with pytest.raises(ValueError, match="learning_rate"):
        model.fit(
            X,
            validation_split=0.25,
            epochs=1,
            batch_size=4,
            learning_rate=learning_rate,
            patience=2,
            verbose=0,
        )

    _assert_model_unchanged(model, snapshot)


@pytest.mark.parametrize(
    "learning_rate",
    [
        True,
        False,
        -1.0,
        float("nan"),
        float("inf"),
        -float("inf"),
        "1e-3",
        1e-3 + 0j,
        np.array([1e-3]),
    ],
)
@pytest.mark.parametrize(
    ("factory", "X"),
    _unbuilt_learning_rate_factories_and_inputs(),
)
def test_invalid_learning_rates_leave_unbuilt_models_pristine(
    factory,
    X,
    learning_rate,
):
    model = factory()
    _assert_unbuilt_training_state(model)
    snapshot = _snapshot_model(model)

    with pytest.raises(ValueError, match="learning_rate"):
        model.fit(
            X,
            validation_split=0.25,
            epochs=1,
            batch_size=4,
            learning_rate=learning_rate,
            patience=2,
            verbose=0,
        )

    _assert_model_unchanged(model, snapshot)
    _assert_unbuilt_training_state(model)


@pytest.mark.parametrize("learning_rate", [0.0, 1e-3])
@pytest.mark.parametrize(
    ("factory", "X"),
    _learning_rate_factories_and_inputs(),
)
def test_valid_learning_rates_fit_all_training_paths(
    factory,
    X,
    learning_rate,
):
    model = factory()

    history = model.fit(
        X,
        validation_split=0.25,
        epochs=1,
        batch_size=4,
        learning_rate=learning_rate,
        patience=2,
        verbose=0,
    )

    assert model.is_fitted
    assert len(history["train_loss"]) == 1
    assert len(history["val_loss"]) == 1


@pytest.mark.parametrize(
    ("factory", "X"),
    _legacy_factories_and_inputs(),
)
def test_all_autoencoders_share_latent_input_validation(factory, X):
    model = factory()
    model._build_input_shape = tuple(X.shape)
    model.model = model._build_model(X.shape).to(model.device)
    model.is_fitted = True

    single = model.decode(np.zeros(model.k, dtype="float32"), verbose=0)
    batch = model.decode(np.zeros((3, model.k), dtype="float32"), verbose=0)
    assert single.shape == (1, *X.shape[1:])
    assert batch.shape == (3, *X.shape[1:])

    invalid_shapes = [
        np.zeros((2, 1, model.k), dtype="float32"),
        np.zeros((2, model.k + 1), dtype="float32"),
        np.zeros((2, 0), dtype="float32"),
        np.zeros((0, model.k), dtype="float32"),
    ]
    for invalid in invalid_shapes:
        with pytest.raises(ValueError, match="Z|latent"):
            model.decode(invalid, verbose=0)

    for invalid in [float("nan"), float("inf"), -float("inf")]:
        latent = np.zeros((2, model.k), dtype="float32")
        latent[0, 0] = invalid
        with pytest.raises(ValueError, match="finite"):
            model.decode(latent, verbose=0)

    with pytest.raises(TypeError, match="real-valued"):
        model.decode(np.zeros((2, model.k), dtype="complex64"), verbose=0)
    with pytest.raises(TypeError, match="numeric"):
        model.decode(np.full((2, model.k), "bad"), verbose=0)


@pytest.mark.parametrize(
    ("factory", "X"),
    _legacy_factories_and_inputs(),
)
def test_all_autoencoders_reject_broadcastable_targets(factory, X):
    model = factory()
    target = X[(slice(None),) + (slice(0, 1),) * (X.ndim - 1)]

    with pytest.raises(ValueError, match="Target shape"):
        model.fit(
            X,
            y=target,
            validation_split=0.25,
            epochs=1,
            batch_size=4,
            patience=2,
            verbose=0,
        )


@pytest.mark.parametrize(
    ("factory", "X", "bad_X"),
    [
        (
            lambda: StandardAutoencoder(k=2, hidden_dims=[4], device="cpu"),
            np.zeros((4, 2, 5), dtype="float32"),
            np.zeros((2, 10), dtype="float32"),
        ),
        (
            lambda: OrthogonalAutoencoder(k=2, hidden_dims=[4], device="cpu"),
            np.zeros((4, 2, 5), dtype="float32"),
            np.zeros((2, 10), dtype="float32"),
        ),
        (
            lambda: LSTMAutoencoder(k=2, hidden=(4, 3), device="cpu"),
            np.zeros((4, 3, 2), dtype="float32"),
            np.zeros((2, 4, 2), dtype="float32"),
        ),
        (
            lambda: CNNAutoencoder(k=2, device="cpu"),
            np.zeros((4, 1, 7, 9), dtype="float32"),
            np.zeros((2, 1, 6, 10), dtype="float32"),
        ),
        (
            lambda: VisionTransformerAutoencoder(
                k=2,
                patch_size=2,
                d_model=4,
                depth_enc=1,
                depth_dec=1,
                heads=1,
                device="cpu",
            ),
            np.zeros((4, 1, 7, 9), dtype="float32"),
            np.zeros((2, 1, 6, 10), dtype="float32"),
        ),
    ],
)
def test_predict_and_encode_reject_changed_sample_shapes(factory, X, bad_X):
    model = factory()
    model._build_input_shape = tuple(X.shape)
    model.model = model._build_model(X.shape).to(model.device)
    model.is_fitted = True

    with pytest.raises(ValueError, match="Expected per-sample shape"):
        model.predict(bad_X, verbose=0)
    with pytest.raises(ValueError, match="Expected per-sample shape"):
        model.encode(bad_X, verbose=0)


@pytest.mark.parametrize("invalid", [np.nan, np.inf, -np.inf])
def test_nonfinite_training_data_is_rejected(invalid):
    X = np.zeros((8, 3), dtype="float32")
    X[0, 0] = invalid
    model = StandardAutoencoder(k=2, hidden_dims=[4], device="cpu")

    with pytest.raises(ValueError, match="finite"):
        model.fit(
            X,
            validation_split=0.25,
            epochs=1,
            batch_size=4,
            patience=2,
            verbose=0,
        )
    assert not model.is_fitted


def test_nonfinite_custom_targets_are_rejected():
    X = np.zeros((8, 3), dtype="float32")
    y = X.copy()
    y[0, 0] = np.nan
    model = StandardAutoencoder(k=2, hidden_dims=[4], device="cpu")

    with pytest.raises(ValueError, match="finite"):
        model.fit(
            X,
            y=y,
            validation_split=0.25,
            epochs=1,
            batch_size=4,
            patience=2,
            verbose=0,
        )


@pytest.mark.parametrize("operation", ["fit", "predict", "decode"])
def test_float64_values_must_remain_finite_after_float32_cast(operation):
    X = np.zeros((8, 3), dtype="float32")
    huge = np.full((8, 3), 1e100, dtype="float64")
    model = _TinyAutoencoder()

    if operation != "fit":
        model.fit(
            X,
            validation_split=0.25,
            epochs=1,
            batch_size=4,
            learning_rate=0.0,
            patience=2,
            verbose=0,
        )

    with pytest.raises(ValueError, match="converted to float32"):
        if operation == "fit":
            model.fit(
                huge,
                validation_split=0.25,
                epochs=1,
                batch_size=4,
                patience=2,
                verbose=0,
            )
        elif operation == "predict":
            model.predict(huge, verbose=0)
        else:
            model.decode(huge, verbose=0)


def test_reconstruction_metrics_reject_invalid_custom_targets():
    X = np.zeros((8, 3), dtype="float32")
    model = _TinyAutoencoder()
    model.fit(
        X,
        validation_split=0.25,
        epochs=1,
        batch_size=4,
        learning_rate=0.0,
        patience=2,
        verbose=0,
    )

    with pytest.raises(ValueError, match="Target shape"):
        model.reconstruction_error(X, y=X[:, :1])

    nonfinite_target = X.copy()
    nonfinite_target[0, 0] = np.nan
    with pytest.raises(ValueError, match="finite"):
        model.evaluate_reconstruction(X, y=nonfinite_target)


@pytest.mark.parametrize(
    "eps",
    [-1.0, float("nan"), float("inf"), -float("inf"), True, "0.0", [0.0]],
)
@pytest.mark.parametrize(
    "operation",
    ["reconstruction_error", "evaluate_reconstruction", "evaluate"],
)
def test_model_metric_wrappers_validate_eps_before_prediction(operation, eps):
    X = np.zeros((4, 3), dtype="float32")
    model = _TinyAutoencoder()
    model._build_input_shape = tuple(X.shape)
    model.model = model._build_model(X.shape)
    model.model.train()
    model.is_fitted = True

    with pytest.raises(ValueError, match="eps"):
        getattr(model, operation)(X, eps=eps)

    assert model.model.training


class _NegatingAutoencoder(BaseDeepLearningModel):
    def __init__(self):
        super().__init__(device="cpu")

    def _build_model(self, input_shape, **kwargs):
        class _Model(nn.Module):
            def __init__(self):
                super().__init__()
                self.anchor = nn.Parameter(torch.zeros(()))

            def forward(self, x):
                return -x + self.anchor * 0

        return _Model()


@pytest.mark.parametrize(
    "operation",
    ["reconstruction_error", "evaluate_reconstruction", "evaluate"],
)
def test_model_metric_wrappers_reject_arithmetic_overflow(operation):
    X = np.full((4, 3), np.finfo(np.float32).max, dtype="float32")
    model = _NegatingAutoencoder()
    model._build_input_shape = tuple(X.shape)
    model.model = model._build_model(X.shape)
    model.is_fitted = True

    with pytest.raises(FloatingPointError, match="Metric subtraction"):
        getattr(model, operation)(X)


class _NonfiniteInferenceAutoencoder(BaseDeepLearningModel):
    def __init__(self):
        super().__init__(device="cpu")

    def _build_model(self, input_shape, **kwargs):
        class _Model(nn.Module):
            def __init__(self):
                super().__init__()
                self.anchor = nn.Parameter(torch.zeros(()))

            def forward(self, x):
                return torch.full_like(x, float("inf")) + self.anchor * 0

            def encode_forward(self, x):
                shape = (len(x), int(np.prod(x.shape[1:])))
                return (
                    torch.full(
                        shape,
                        float("inf"),
                        dtype=x.dtype,
                        device=x.device,
                    )
                    + self.anchor * 0
                )

            def decode_forward(self, z):
                return torch.full_like(z, float("inf")) + self.anchor * 0

        return _Model()


@pytest.mark.parametrize("operation", ["predict", "encode", "decode"])
def test_nonfinite_inference_results_are_rejected(operation):
    X = np.zeros((4, 3), dtype="float32")
    model = _NonfiniteInferenceAutoencoder()
    model._build_input_shape = tuple(X.shape)
    model.model = model._build_model(X.shape)
    model.is_fitted = True

    with pytest.raises(FloatingPointError, match="not finite"):
        getattr(model, operation)(X, verbose=0)


class _InfiniteBufferAutoencoder(BaseDeepLearningModel):
    def __init__(self):
        super().__init__(device="cpu")

    def _build_model(self, input_shape, **kwargs):
        class _Model(nn.Module):
            def __init__(self):
                super().__init__()
                self.anchor = nn.Parameter(torch.zeros(()))
                self.register_buffer("running_value", torch.zeros(()))

            def forward(self, x):
                self.running_value.fill_(float("inf"))
                return torch.zeros_like(x) + self.anchor * 0

        return _Model()


def test_nonfinite_model_buffers_abort_training():
    X = np.ones((8, 2), dtype="float32")
    model = _InfiniteBufferAutoencoder()

    with pytest.raises(FloatingPointError, match="Buffer"):
        model.fit(
            X,
            validation_split=0.25,
            epochs=1,
            batch_size=4,
            patience=2,
            verbose=0,
        )
    assert not model.is_fitted


class _ValidationOnlyInfiniteBufferAutoencoder(BaseDeepLearningModel):
    def __init__(self):
        super().__init__(device="cpu")

    def _build_model(self, input_shape, **kwargs):
        class _Model(nn.Module):
            def __init__(self):
                super().__init__()
                self.anchor = nn.Parameter(torch.zeros(()))
                self.register_buffer("running_value", torch.zeros(()))

            def forward(self, x):
                if not self.training:
                    self.running_value.fill_(float("inf"))
                return torch.zeros_like(x) + self.anchor * 0

        return _Model()


def test_validation_only_nonfinite_buffers_abort_training():
    X = np.ones((8, 2), dtype="float32")
    model = _ValidationOnlyInfiniteBufferAutoencoder()

    with pytest.raises(FloatingPointError, match="Buffer"):
        model.fit(
            X,
            validation_split=0.25,
            epochs=1,
            batch_size=4,
            patience=2,
            verbose=0,
        )
    assert not model.is_fitted


def test_inference_only_nonfinite_buffers_are_rejected():
    X = np.ones((4, 2), dtype="float32")
    model = _ValidationOnlyInfiniteBufferAutoencoder()
    model._build_input_shape = tuple(X.shape)
    model.model = model._build_model(X.shape)
    model.is_fitted = True

    with pytest.raises(FloatingPointError, match="Buffer"):
        model.predict(X, verbose=0)


class _ValidationOnlyInfiniteParameterAutoencoder(BaseDeepLearningModel):
    def __init__(self):
        super().__init__(device="cpu")

    def _build_model(self, input_shape, **kwargs):
        class _Model(nn.Module):
            def __init__(self):
                super().__init__()
                self.anchor = nn.Parameter(torch.zeros(()))

            def forward(self, x):
                if not self.training:
                    self.anchor.fill_(float("inf"))
                    return torch.zeros_like(x)
                return torch.zeros_like(x) + self.anchor * 0

        return _Model()


def test_validation_only_nonfinite_parameters_abort_training():
    X = np.ones((8, 2), dtype="float32")
    model = _ValidationOnlyInfiniteParameterAutoencoder()

    with pytest.raises(FloatingPointError, match="Parameter"):
        model.fit(
            X,
            validation_split=0.25,
            epochs=1,
            batch_size=4,
            patience=2,
            verbose=0,
        )
    assert not model.is_fitted


def test_inference_only_nonfinite_parameters_are_rejected():
    X = np.ones((4, 2), dtype="float32")
    model = _ValidationOnlyInfiniteParameterAutoencoder()
    model._build_input_shape = tuple(X.shape)
    model.model = model._build_model(X.shape)
    model.is_fitted = True

    with pytest.raises(FloatingPointError, match="Parameter"):
        model.predict(X, verbose=0)


class _NonTensorOutputAutoencoder(BaseDeepLearningModel):
    def __init__(self):
        super().__init__(device="cpu")

    def _build_model(self, input_shape, **kwargs):
        class _Model(nn.Module):
            def __init__(self):
                super().__init__()
                self.anchor = nn.Parameter(torch.zeros(()))

            def forward(self, x):
                return (x + self.anchor * 0,)

        return _Model()


def test_non_tensor_model_outputs_are_rejected_clearly():
    X = np.zeros((8, 3), dtype="float32")
    model = _NonTensorOutputAutoencoder()

    with pytest.raises(TypeError, match="Model output"):
        model.fit(
            X,
            validation_split=0.25,
            epochs=1,
            batch_size=4,
            patience=2,
            verbose=0,
        )
    assert not model.is_fitted


class _WrongShapeAutoencoder(BaseDeepLearningModel):
    def __init__(self):
        super().__init__(device="cpu")

    def _build_model(self, input_shape, **kwargs):
        class _Model(nn.Module):
            def __init__(self):
                super().__init__()
                self.anchor = nn.Parameter(torch.zeros(()))

            def forward(self, x):
                return x[:, :1] + self.anchor * 0

        return _Model()


def test_model_outputs_cannot_broadcast_against_targets():
    X = np.zeros((8, 3), dtype="float32")
    model = _WrongShapeAutoencoder()

    with pytest.raises(ValueError, match="output shape"):
        model.fit(
            X,
            validation_split=0.25,
            epochs=1,
            batch_size=4,
            patience=2,
            verbose=0,
        )
    assert not model.is_fitted


def test_nonfinite_loss_aborts_without_marking_model_fitted():
    class _NaNLoss(nn.Module):
        reduction = "mean"

        def forward(self, output, target):
            return output.sum() * torch.tensor(float("nan"))

    X = np.ones((8, 2), dtype="float32")
    model = _TinyAutoencoder()

    with pytest.raises(FloatingPointError, match="not finite"):
        model.fit(
            X,
            validation_split=0.25,
            epochs=1,
            batch_size=3,
            criterion=_NaNLoss(),
            patience=2,
            verbose=0,
        )
    assert not model.is_fitted


class _InfiniteGradientLoss(nn.Module):
    reduction = "mean"

    class _Operation(torch.autograd.Function):
        @staticmethod
        def forward(ctx, value):
            return value.sum() * 0

        @staticmethod
        def backward(ctx, gradient):
            return torch.full_like(gradient, float("inf"))

    def forward(self, output, target):
        return self._Operation.apply(output.sum())


def test_nonfinite_gradients_abort_without_marking_model_fitted():
    X = np.ones((8, 2), dtype="float32")
    model = _TinyAutoencoder()

    with pytest.raises(FloatingPointError, match="Gradient"):
        model.fit(
            X,
            validation_split=0.25,
            epochs=1,
            batch_size=4,
            criterion=_InfiniteGradientLoss(),
            patience=2,
            verbose=0,
        )
    assert not model.is_fitted


class _InfiniteParameterOptimizer(torch.optim.Optimizer):
    def __init__(self, parameters):
        super().__init__(parameters, {})

    @torch.no_grad()
    def step(self, closure=None):
        for group in self.param_groups:
            for parameter in group["params"]:
                parameter.fill_(float("inf"))


def test_nonfinite_parameters_abort_without_marking_model_fitted():
    X = np.ones((8, 2), dtype="float32")
    model = _TinyAutoencoder()
    model.model = model._build_model(X.shape)
    optimizer = _InfiniteParameterOptimizer(model.model.parameters())

    with pytest.raises(FloatingPointError, match="Parameter"):
        model.fit(
            X,
            validation_split=0.25,
            epochs=1,
            batch_size=4,
            optimizer=optimizer,
            patience=2,
            verbose=0,
        )
    assert not model.is_fitted


def test_epoch_histories_weight_short_batches_by_sample_count():
    X = np.array([[0.0], [1.0], [2.0], [3.0], [4.0], [5.0], [20.0]])
    X = X.astype("float32")
    validation_split = 0.4

    np.random.seed(31)
    indices = np.arange(len(X))
    np.random.shuffle(indices)
    split = int((1 - validation_split) * len(X))
    train_indices = indices[:split]
    validation_indices = indices[split:]
    expected_train = float(np.mean(X[train_indices] ** 2))
    expected_validation = float(np.mean(X[validation_indices] ** 2))

    np.random.seed(31)
    model = _TinyAutoencoder()
    history = model.fit(
        X,
        validation_split=validation_split,
        epochs=1,
        batch_size=3,
        learning_rate=0.0,
        patience=2,
        verbose=0,
    )

    assert history["train_loss"][0] == pytest.approx(expected_train)
    assert history["val_loss"][0] == pytest.approx(expected_validation)


def test_sum_reduced_histories_weight_short_batches_by_sample_count():
    X = np.array([[0.0], [1.0], [2.0], [3.0], [4.0], [5.0], [20.0]])
    X = X.astype("float32")
    validation_split = 0.4

    np.random.seed(43)
    indices = np.arange(len(X))
    np.random.shuffle(indices)
    split = int((1 - validation_split) * len(X))
    expected_train = float(np.mean(X[indices[:split]] ** 2))
    expected_validation = float(np.mean(X[indices[split:]] ** 2))

    np.random.seed(43)
    model = _TinyAutoencoder()
    history = model.fit(
        X,
        validation_split=validation_split,
        epochs=1,
        batch_size=3,
        learning_rate=0.0,
        criterion=nn.MSELoss(reduction="sum"),
        patience=2,
        verbose=0,
    )

    assert history["train_loss"][0] == pytest.approx(expected_train)
    assert history["val_loss"][0] == pytest.approx(expected_validation)


def test_orthogonal_histories_weight_short_batches_by_sample_count():
    class _ZeroRegularizedModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.anchor = nn.Parameter(torch.zeros(()))

        def forward(self, x):
            return torch.zeros_like(x) + self.anchor * 0

        def get_regularization_losses(self):
            zero = self.anchor * 0
            return zero, zero

    X = np.array([[0.0], [1.0], [2.0], [3.0], [4.0], [5.0], [20.0]])
    X = X.astype("float32")
    validation_split = 0.4

    np.random.seed(37)
    indices = np.arange(len(X))
    np.random.shuffle(indices)
    split = int((1 - validation_split) * len(X))
    expected_train = float(np.mean(X[indices[:split]] ** 2))
    expected_validation = float(np.mean(X[indices[split:]] ** 2))

    np.random.seed(37)
    model = OrthogonalAutoencoder(
        k=1,
        hidden_dims=[2],
        lambda_W=0.0,
        lambda_Z=0.0,
        device="cpu",
    )
    model.model = _ZeroRegularizedModel()
    history = model.fit(
        X,
        validation_split=validation_split,
        epochs=1,
        batch_size=3,
        learning_rate=0.0,
        patience=2,
        verbose=0,
    )

    assert history["train_loss"][0] == pytest.approx(expected_train)
    assert history["val_loss"][0] == pytest.approx(expected_validation)


def test_nonfinite_checkpoint_state_is_rejected_before_loading(tmp_path):
    X = np.zeros((8, 3), dtype="float32")
    model = _TinyAutoencoder()
    model.fit(
        X,
        validation_split=0.25,
        epochs=1,
        batch_size=4,
        learning_rate=0.0,
        patience=2,
        verbose=0,
    )
    original_anchor = model.model.anchor.detach().clone()

    checkpoint_path = tmp_path / "tiny.pt"
    model.save_pytorch_model(checkpoint_path)
    checkpoint = torch.load(checkpoint_path, weights_only=False)
    checkpoint["model_state_dict"]["anchor"] = torch.tensor(float("nan"))
    torch.save(checkpoint, checkpoint_path)

    with pytest.raises(FloatingPointError, match="Checkpoint state entry"):
        model.load_pytorch_model(checkpoint_path, weights_only=False)
    assert model.is_fitted
    assert torch.equal(model.model.anchor.detach(), original_anchor)

    with pytest.raises(FloatingPointError, match="Checkpoint state entry"):
        _TinyAutoencoder.from_pytorch_model(
            checkpoint_path,
            weights_only=False,
        )


def test_late_checkpoint_shape_mismatch_is_transactional(tmp_path):
    X = np.zeros((8, 3), dtype="float32")
    model = _TinyAutoencoder()
    model.fit(
        X,
        validation_split=0.25,
        epochs=1,
        batch_size=4,
        learning_rate=0.0,
        patience=2,
        verbose=0,
    )
    model.model.eval()
    snapshot = _snapshot_model(model)
    checkpoint_path = tmp_path / "late-shape-mismatch.pt"
    model.save_pytorch_model(checkpoint_path)
    checkpoint = torch.load(checkpoint_path, weights_only=False)
    checkpoint["model_state_dict"]["anchor"] = torch.ones(())
    checkpoint["model_state_dict"]["tail"] = torch.ones(3)
    torch.save(checkpoint, checkpoint_path)

    with pytest.raises(RuntimeError, match="tail.*shape"):
        model.load_pytorch_model(checkpoint_path, weights_only=False)

    _assert_model_unchanged(model, snapshot)


def test_checkpoint_cast_overflow_is_transactional(tmp_path):
    X = np.zeros((8, 3), dtype="float32")
    model = _TinyAutoencoder()
    model.fit(
        X,
        validation_split=0.25,
        epochs=1,
        batch_size=4,
        learning_rate=0.0,
        patience=2,
        verbose=0,
    )
    snapshot = _snapshot_model(model)
    checkpoint_path = tmp_path / "cast-overflow.pt"
    model.save_pytorch_model(checkpoint_path)
    checkpoint = torch.load(checkpoint_path, weights_only=False)
    checkpoint["model_state_dict"]["anchor"] = torch.tensor(
        1e300,
        dtype=torch.float64,
    )
    torch.save(checkpoint, checkpoint_path)

    with pytest.raises(FloatingPointError, match="after conversion"):
        model.load_pytorch_model(checkpoint_path, weights_only=False)

    _assert_model_unchanged(model, snapshot)
    with pytest.raises(FloatingPointError, match="after conversion"):
        _TinyAutoencoder.from_pytorch_model(
            checkpoint_path,
            weights_only=False,
        )


def test_from_pytorch_model_rejects_late_shape_mismatch(tmp_path):
    X = np.zeros((8, 3), dtype="float32")
    model = _TinyAutoencoder()
    model.fit(
        X,
        validation_split=0.25,
        epochs=1,
        batch_size=4,
        learning_rate=0.0,
        patience=2,
        verbose=0,
    )
    checkpoint_path = tmp_path / "from-late-shape-mismatch.pt"
    model.save_pytorch_model(checkpoint_path)
    checkpoint = torch.load(checkpoint_path, weights_only=False)
    checkpoint["model_state_dict"]["anchor"] = torch.ones(())
    checkpoint["model_state_dict"]["tail"] = torch.ones(3)
    torch.save(checkpoint, checkpoint_path)

    with pytest.raises(RuntimeError, match="tail.*shape"):
        _TinyAutoencoder.from_pytorch_model(
            checkpoint_path,
            weights_only=False,
        )


def test_failed_unbuilt_load_preserves_constructor_configuration(tmp_path):
    X = np.zeros((8, 3), dtype="float32")
    source = StandardAutoencoder(k=2, hidden_dims=[4], device="cpu")
    source.fit(
        X,
        validation_split=0.25,
        epochs=1,
        batch_size=4,
        learning_rate=0.0,
        patience=2,
        verbose=0,
    )
    checkpoint_path = tmp_path / "unbuilt-config-mismatch.pt"
    source.save_pytorch_model(checkpoint_path)
    checkpoint = torch.load(checkpoint_path, weights_only=False)
    state_keys = list(checkpoint["model_state_dict"])
    checkpoint["model_state_dict"][state_keys[0]] = (
        checkpoint["model_state_dict"][state_keys[0]] + 1
    )
    later_value = checkpoint["model_state_dict"][state_keys[-1]]
    checkpoint["model_state_dict"][state_keys[-1]] = torch.zeros(
        later_value.numel() + 1,
        dtype=later_value.dtype,
    )
    torch.save(checkpoint, checkpoint_path)

    target = StandardAutoencoder(k=5, hidden_dims=[7], device="cpu")
    snapshot = _snapshot_model(target)

    with pytest.raises(RuntimeError, match="shape"):
        target.load_pytorch_model(checkpoint_path, weights_only=False)

    _assert_model_unchanged(target, snapshot)
    assert target.k == 5
    assert target.hidden_dims == [7]


@pytest.mark.parametrize(
    "factory",
    [
        lambda: StandardAutoencoder(k=0),
        lambda: StandardAutoencoder(k=2, hidden_dims=[4, 0]),
        lambda: OrthogonalAutoencoder(lambda_W=-1.0),
        lambda: OrthogonalAutoencoder(lambda_Z=float("nan")),
        lambda: LSTMAutoencoder(hidden=(4,)),
        lambda: CNNAutoencoder(k=0),
        lambda: VisionTransformerAutoencoder(patch_size=0),
        lambda: VisionTransformerAutoencoder(d_model=1, heads=1),
        lambda: VisionTransformerAutoencoder(d_model=2, heads=1),
        lambda: VisionTransformerAutoencoder(
            d_model=10,
            heads=3,
        ),
        lambda: ConvLSTMAutoencoder(k=0),
        lambda: HybridConvLSTMTransformerAutoencoder(
            d_model=1,
            n_heads=1,
        ),
        lambda: HybridConvLSTMTransformerAutoencoder(
            d_model=2,
            n_heads=1,
        ),
        lambda: HybridConvLSTMTransformerAutoencoder(efficient_attention="linera"),
    ],
)
def test_existing_autoencoders_reject_invalid_constructor_values(factory):
    with pytest.raises(ValueError):
        factory()


def test_linear_attention_accepts_long_sequences_without_quadratic_scores():
    layer = LinearSelfAttention(d_model=16, num_heads=4)
    values = torch.randn(2, 512, 16)

    output = layer(values)

    assert output.shape == values.shape
    assert torch.isfinite(output).all()


@pytest.mark.parametrize(
    "kwargs",
    [
        {"d_model": 0, "num_heads": 1},
        {"d_model": 4, "num_heads": 0},
        {"d_model": 5, "num_heads": 2},
        {"d_model": True, "num_heads": 1},
    ],
)
def test_linear_attention_rejects_invalid_constructor_values(kwargs):
    with pytest.raises(ValueError):
        LinearSelfAttention(**kwargs)


def test_linear_attention_rejects_incompatible_input_shape():
    layer = LinearSelfAttention(d_model=4, num_heads=1)

    with pytest.raises(ValueError, match="expects input shape"):
        layer(torch.randn(2, 8, 3))
