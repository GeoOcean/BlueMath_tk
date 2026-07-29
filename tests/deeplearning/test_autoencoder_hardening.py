"""Regression tests for shared autoencoder hardening."""

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
    np.random.seed(607)
    torch.manual_seed(607)
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(previous_threads)


class _TinyAutoencoder(BaseDeepLearningModel):
    def __init__(self, device="cpu"):
        super().__init__(device=device)

    def _build_model(self, input_shape, **kwargs):
        sample_shape = tuple(input_shape[1:])

        class _Model(nn.Module):
            def __init__(self):
                super().__init__()
                self.anchor = nn.Parameter(torch.zeros(()))

            def forward(self, x):
                return torch.zeros_like(x) + self.anchor * 0

            def encode_forward(self, x):
                return x.reshape(len(x), -1)

            def decode_forward(self, z):
                return z.reshape(len(z), *sample_shape)

        return _Model()


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
