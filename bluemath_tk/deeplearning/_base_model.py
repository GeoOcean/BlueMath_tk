import copy
import inspect
from abc import abstractmethod
from numbers import Real

import numpy as np
import torch
import torch.nn as nn
from tqdm import tqdm

from ..core.models import BlueMathModel
from .metrics import _validate_eps
from .metrics import evaluate_reconstruction as evaluate_reconstruction_metric
from .metrics import reconstruction_error as reconstruction_error_metric


class BaseDeepLearningModel(BlueMathModel):
    """
    Base class for all PyTorch deep learning BlueMath models.

    This class provides the basic structure for all deep learning models,
    including common functionality for training, evaluation, and prediction.

    Attributes
    ----------
    model : torch.nn.Module
        The PyTorch model.
    device : torch.device
        The device (CPU/GPU) the model is on.
    is_fitted : bool
        Whether the model has been fitted.
    """

    @abstractmethod
    def __init__(self, device: str | torch.device | None = None, **kwargs):
        """
        Initialize the base deep learning model.

        Parameters
        ----------
        device : str or torch.device, optional
            Device to run the model on ('cpu', 'cuda', etc.).
            If None, uses 'cuda' if available, else 'cpu'.
            Default is None.
        **kwargs
            Additional keyword arguments passed to BlueMathModel.
        """

        super().__init__(**kwargs)

        if device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        elif isinstance(device, str):
            self.device = torch.device(device)
        else:
            self.device = device

        self.model: nn.Module | None = None
        self.is_fitted = False
        self._build_input_shape: tuple | None = None

        self._exclude_attributes = [
            "model",
        ]

    @abstractmethod
    def _build_model(self, *args, **kwargs) -> nn.Module:
        """
        Build the PyTorch model.

        Returns
        -------
        torch.nn.Module
            The PyTorch model.
        """

        pass

    def _get_reconstruction_target(self, X: np.ndarray) -> np.ndarray:
        """Return the default reconstruction target for ``X``."""
        return X

    @staticmethod
    def _batch_slices(
        n_samples: int,
        batch_size: int,
        avoid_singleton: bool = False,
    ) -> list[tuple[int, int]]:
        """Return batch slices, optionally avoiding a final singleton batch."""
        if avoid_singleton and batch_size == 1:
            raise ValueError(
                "batch_size=1 is incompatible with this model's BatchNorm1d "
                "layers. Use batch_size>=2."
            )

        slices = [
            (start, min(start + batch_size, n_samples))
            for start in range(0, n_samples, batch_size)
        ]

        if avoid_singleton and len(slices) > 1 and slices[-1][1] - slices[-1][0] == 1:
            previous_start, previous_stop = slices[-2]
            final_stop = slices[-1][1]
            previous_size = previous_stop - previous_start

            if previous_size > 2:
                slices[-2] = (previous_start, previous_stop - 1)
                slices[-1] = (previous_stop - 1, final_stop)
            else:
                slices[-2] = (previous_start, final_stop)
                slices.pop()

        return slices

    def _requires_non_singleton_training_batches(self) -> bool:
        """Return whether BatchNorm1d requires at least two training samples."""
        if self.model is None:
            return False
        return any(
            isinstance(module, nn.BatchNorm1d) for module in self.model.modules()
        )

    def _validate_or_set_build_input_shape(self, input_shape: tuple) -> None:
        """Store the build shape or reject incompatible repeated fitting."""
        input_shape = tuple(input_shape)
        if self._build_input_shape is None:
            self._build_input_shape = input_shape
            return

        if tuple(self._build_input_shape[1:]) != tuple(input_shape[1:]):
            raise ValueError(
                "The fitted model input shape is incompatible with the new data. "
                f"Expected per-sample shape {self._build_input_shape[1:]}, "
                f"got {input_shape[1:]}."
            )

    @staticmethod
    def _validate_learning_rate(learning_rate: float) -> float:
        """Return a finite, non-negative real scalar learning rate."""
        if (
            not isinstance(learning_rate, Real)
            or isinstance(learning_rate, (bool, np.bool_))
            or not np.isfinite(float(learning_rate))
            or learning_rate < 0
        ):
            raise ValueError(
                "learning_rate must be a finite, non-negative real scalar."
            )
        return float(learning_rate)

    @staticmethod
    def _validate_finite_array(array: np.ndarray, name: str) -> None:
        """Require a finite, real-valued NumPy array."""
        if not isinstance(array, np.ndarray):
            raise TypeError(f"{name} must be a NumPy array.")
        if not np.issubdtype(array.dtype, np.number):
            raise TypeError(f"{name} must contain numeric values.")
        if np.issubdtype(array.dtype, np.complexfloating):
            raise TypeError(f"{name} must contain real-valued data.")
        if not np.isfinite(array).all():
            raise ValueError(f"{name} must contain only finite values.")
        float32_limit = np.finfo(np.float32).max
        if np.any(array > float32_limit) or np.any(array < -float32_limit):
            raise ValueError(f"{name} must remain finite when converted to float32.")

    def _validate_target_shape(
        self,
        X: np.ndarray,
        target: np.ndarray,
    ) -> None:
        """Require the target to match the reconstruction shape exactly."""
        if not isinstance(target, np.ndarray):
            raise TypeError("y must be a NumPy array.")
        if tuple(target.shape) != tuple(X.shape):
            raise ValueError(
                f"Target shape {target.shape} is incompatible with "
                f"reconstruction shape {X.shape}."
            )

    def _validate_inference_inputs(
        self,
        X: np.ndarray,
        batch_size: int,
        name: str = "X",
        check_expected_shape: bool = False,
    ) -> None:
        """Validate prediction, encoding, decoding, and metric inputs."""
        if not isinstance(X, np.ndarray):
            raise TypeError(f"{name} must be a NumPy array.")
        if X.ndim < 1:
            raise ValueError(f"{name} must have at least one dimension.")
        if len(X) == 0:
            raise ValueError(f"{name} must contain at least one sample.")
        if (
            not isinstance(batch_size, int)
            or isinstance(batch_size, bool)
            or batch_size < 1
        ):
            raise ValueError("batch_size must be a positive integer.")
        self._validate_finite_array(X, name)

        if check_expected_shape and self._build_input_shape is not None:
            expected = tuple(self._build_input_shape[1:])
            actual = tuple(X.shape[1:])
            if actual != expected:
                raise ValueError(f"Expected per-sample shape {expected}, got {actual}.")

    def _validate_latent_inputs(
        self,
        Z: np.ndarray,
        batch_size: int,
    ) -> None:
        """Require latent data with the public ``(batch, k)`` shape."""
        expected_width = getattr(self, "k", None)
        if Z.ndim != 2:
            expected = (
                f"(batch, {expected_width})"
                if expected_width is not None
                else "(batch, latent_width)"
            )
            raise ValueError(f"Z must have shape {expected}; got shape {Z.shape}.")
        if Z.shape[0] < 1:
            raise ValueError("Z must contain at least one latent vector.")
        if expected_width is not None and Z.shape[1] != expected_width:
            raise ValueError(
                f"Z must have latent width {expected_width}; got {Z.shape[1]}."
            )
        self._validate_inference_inputs(Z, batch_size, name="Z")

    def _validate_fit_inputs(
        self,
        X: np.ndarray,
        y: np.ndarray,
        validation_split: float,
        batch_size: int,
        epochs: int,
        patience: int,
        validation_data: tuple[np.ndarray, np.ndarray | None] | None = None,
    ) -> None:
        """Validate common training inputs before building the model."""
        if not isinstance(X, np.ndarray):
            raise TypeError("X must be a NumPy array.")
        if not isinstance(y, np.ndarray):
            raise TypeError("y must be a NumPy array.")
        if X.ndim < 2:
            raise ValueError(
                "X must include a leading sample dimension. "
                "For tabular data use shape (n_samples, n_features)."
            )
        if any(dimension < 1 for dimension in X.shape[1:]):
            raise ValueError("Every per-sample dimension must be positive.")
        if len(X) != len(y):
            raise ValueError(
                "X and y must contain the same number of samples; "
                f"got {len(X)} and {len(y)}."
            )
        self._validate_target_shape(X, y)
        self._validate_finite_array(X, "X")
        self._validate_finite_array(y, "y")

        if validation_data is None and (
            not isinstance(validation_split, (int, float))
            or isinstance(validation_split, bool)
            or not np.isfinite(float(validation_split))
            or not 0.0 < validation_split < 1.0
        ):
            raise ValueError(
                "validation_split must be finite and strictly between 0 and 1."
            )
        for name, value in (
            ("batch_size", batch_size),
            ("epochs", epochs),
            ("patience", patience),
        ):
            if not isinstance(value, int) or isinstance(value, bool) or value < 1:
                raise ValueError(f"{name} must be a positive integer.")

        if validation_data is not None:
            # The pair itself is validated once, in _resolve_fit_partitions,
            # which runs immediately after this method and before the model is
            # built.
            if len(X) < 2:
                raise ValueError(
                    "Explicit validation_data requires at least two training "
                    "samples in X."
                )
            return

        split = int((1 - validation_split) * len(X))
        if split < 2:
            raise ValueError(
                "The training split must contain at least two samples. "
                "Increase the dataset size or reduce validation_split."
            )
        if len(X) - split < 1:
            raise ValueError("The validation split must contain at least one sample.")

    def _validate_validation_data(
        self,
        X: np.ndarray,
        validation_data: tuple[np.ndarray, np.ndarray | None],
    ) -> tuple[np.ndarray, np.ndarray]:
        """Validate explicit validation data and return the resolved arrays.

        Parameters
        ----------
        X : np.ndarray
            The training inputs, used to check the per-sample contract.
        validation_data : tuple
            An ``(X_validation, y_validation)`` pair. ``y_validation`` may be
            ``None``, in which case the model's default reconstruction target
            is derived from ``X_validation``.

        Returns
        -------
        tuple of np.ndarray
            The validated ``(X_validation, y_validation)`` arrays.
        """
        if not isinstance(validation_data, tuple) or len(validation_data) != 2:
            raise TypeError(
                "validation_data must be an (X_validation, y_validation) tuple; "
                "pass y_validation=None to reconstruct X_validation itself."
            )
        X_validation, y_validation = validation_data
        if not isinstance(X_validation, np.ndarray):
            raise TypeError("validation_data[0] must be a NumPy array.")
        if X_validation.ndim != X.ndim:
            raise ValueError(
                f"validation_data[0] must have {X.ndim} dimensions to match X; "
                f"got {X_validation.ndim}."
            )
        if len(X_validation) < 1:
            raise ValueError("validation_data[0] must contain at least one sample.")
        if tuple(X_validation.shape[1:]) != tuple(X.shape[1:]):
            raise ValueError(
                "validation_data[0] per-sample shape "
                f"{tuple(X_validation.shape[1:])} does not match the training "
                f"per-sample shape {tuple(X.shape[1:])}."
            )
        self._validate_finite_array(X_validation, "validation_data[0]")

        if y_validation is None:
            y_validation = self._get_reconstruction_target(X_validation)
        if not isinstance(y_validation, np.ndarray):
            raise TypeError("validation_data[1] must be a NumPy array or None.")
        if len(y_validation) != len(X_validation):
            raise ValueError(
                "validation_data arrays must contain the same number of samples; "
                f"got {len(X_validation)} and {len(y_validation)}."
            )
        self._validate_target_shape(X_validation, y_validation)
        self._validate_finite_array(y_validation, "validation_data[1]")
        return X_validation, y_validation

    def _resolve_fit_partitions(
        self,
        X: np.ndarray,
        y: np.ndarray,
        validation_split: float,
        validation_data: tuple[np.ndarray, np.ndarray | None] | None,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Return the training and validation arrays used by one fit call.

        With ``validation_data=None`` the historical behaviour is preserved
        exactly: one random permutation of ``X`` is cut at ``validation_split``
        using the global NumPy random state.

        When explicit ``validation_data`` is supplied, ``validation_split`` is
        ignored, every sample of ``X`` is used for optimisation in the given
        order, and the global NumPy random state is left untouched. This makes
        the validation membership exactly reproducible, which is required for
        chronological validation.
        """
        if validation_data is None:
            indices = np.arange(len(X))
            np.random.shuffle(indices)
            split = int((1 - validation_split) * len(X))
            train_indices, validation_indices = indices[:split], indices[split:]
            return (
                X[train_indices],
                y[train_indices],
                X[validation_indices],
                y[validation_indices],
            )

        X_validation, y_validation = self._validate_validation_data(X, validation_data)
        return X, y, X_validation, y_validation

    def _get_init_config(self) -> dict:
        """Collect constructor parameters needed to recreate this model."""
        config = {}
        missing = []
        signature = inspect.signature(self.__class__.__init__)
        for name, parameter in signature.parameters.items():
            if name in {"self", "device"}:
                continue
            if parameter.kind is inspect.Parameter.VAR_KEYWORD:
                continue
            if hasattr(self, name):
                config[name] = getattr(self, name)
            else:
                missing.append(name)

        if missing:
            raise ValueError(
                f"{self.__class__.__name__} cannot create a self-describing "
                f"checkpoint because these constructor parameters are not "
                f"stored as attributes: {missing}. Override _get_init_config()."
            )
        return config

    @staticmethod
    def _require_scalar_loss(loss: torch.Tensor) -> None:
        """Raise when a criterion does not return one scalar tensor."""
        if not isinstance(loss, torch.Tensor):
            raise TypeError("The training criterion must return a PyTorch tensor.")
        if loss.ndim != 0:
            raise ValueError(
                "The training criterion must return a scalar loss. "
                "Use reduction='mean' or reduction='sum'."
            )

    @staticmethod
    def _require_finite_tensor(value: torch.Tensor, phase: str) -> None:
        """Reject non-tensor or non-finite model results."""
        if not isinstance(value, torch.Tensor):
            raise TypeError(f"{phase} must be a PyTorch tensor.")
        if not torch.isfinite(value).all():
            raise FloatingPointError(f"{phase} is not finite.")

    @classmethod
    def _require_finite_loss(cls, loss: torch.Tensor, phase: str) -> None:
        """Abort immediately when a training objective is not finite."""
        cls._require_finite_tensor(loss, f"{phase} loss")

    @staticmethod
    def _require_matching_output_shape(
        output: torch.Tensor,
        target: torch.Tensor,
        phase: str,
    ) -> None:
        """Reject non-tensor or broadcastable model outputs."""
        if not isinstance(output, torch.Tensor):
            raise TypeError("Model output must be a PyTorch tensor.")
        if not isinstance(target, torch.Tensor):
            raise TypeError("Training target must be a PyTorch tensor.")
        if tuple(output.shape) != tuple(target.shape):
            raise ValueError(
                f"{phase} output shape {tuple(output.shape)} does not match "
                f"target shape {tuple(target.shape)}."
            )

    def _require_finite_gradients(self) -> None:
        """Abort when any model gradient contains NaN or infinity."""
        if self.model is None:
            raise ValueError("Model must be built before checking gradients.")
        for name, parameter in self.model.named_parameters():
            gradient = parameter.grad
            if gradient is not None and not torch.isfinite(gradient).all():
                raise FloatingPointError(
                    f"Gradient for parameter {name!r} is not finite."
                )

    def _require_finite_buffers(self) -> None:
        """Abort when a registered model buffer is not finite."""
        if self.model is None:
            raise ValueError("Model must be built before checking buffers.")
        for name, buffer in self.model.named_buffers():
            if not torch.isfinite(buffer).all():
                raise FloatingPointError(f"Buffer {name!r} is not finite.")

    def _require_finite_parameters(self) -> None:
        """Abort when model parameters or registered buffers are not finite."""
        if self.model is None:
            raise ValueError("Model must be built before checking parameters.")
        for name, parameter in self.model.named_parameters():
            if not torch.isfinite(parameter).all():
                raise FloatingPointError(f"Parameter {name!r} is not finite.")
        self._require_finite_buffers()

    @staticmethod
    def _require_finite_state_dict(
        state_dict: dict,
        phase: str = "Checkpoint",
    ) -> None:
        """Reject non-finite floating or complex checkpoint tensors."""
        if not isinstance(state_dict, dict):
            raise TypeError("model_state_dict must be a dictionary.")
        for name, value in state_dict.items():
            if not isinstance(value, torch.Tensor):
                raise TypeError(f"{phase} state entry {name!r} must be a tensor.")
            if (value.is_floating_point() or value.is_complex()) and not (
                torch.isfinite(value).all()
            ):
                raise FloatingPointError(f"{phase} state entry {name!r} is not finite.")

    @classmethod
    def _validate_checkpoint_state_compatibility(
        cls,
        model: nn.Module,
        state_dict: dict,
    ) -> None:
        """Validate checkpoint keys, shapes, and destination conversions."""
        cls._require_finite_state_dict(state_dict)
        destination_state = model.state_dict()
        missing = [name for name in destination_state if name not in state_dict]
        unexpected = [name for name in state_dict if name not in destination_state]
        if missing or unexpected:
            details = []
            if missing:
                details.append(f"missing keys: {missing}")
            if unexpected:
                details.append(f"unexpected keys: {unexpected}")
            raise RuntimeError(
                "Checkpoint state_dict is incompatible with the model ("
                + "; ".join(details)
                + ")."
            )

        for name, destination in destination_state.items():
            stored = state_dict[name]
            if tuple(stored.shape) != tuple(destination.shape):
                raise RuntimeError(
                    f"Checkpoint state entry {name!r} has shape "
                    f"{tuple(stored.shape)}, but the model expects "
                    f"{tuple(destination.shape)}."
                )
            try:
                converted = stored.to(
                    device=destination.device,
                    dtype=destination.dtype,
                )
            except (RuntimeError, TypeError) as error:
                raise RuntimeError(
                    f"Checkpoint state entry {name!r} cannot be converted "
                    f"to {destination.dtype} on {destination.device}."
                ) from error
            if (
                converted.is_floating_point() or converted.is_complex()
            ) and not torch.isfinite(converted).all():
                raise FloatingPointError(
                    f"Checkpoint state entry {name!r} is not finite after "
                    f"conversion to {destination.dtype}."
                )

    @staticmethod
    def _validate_checkpoint_structure(checkpoint: dict) -> None:
        """Require the mapping structure used by PyTorch checkpoints."""
        if not isinstance(checkpoint, dict):
            raise TypeError("The PyTorch checkpoint must be a dictionary.")

    def _stage_checkpoint(self, checkpoint: dict):
        """Load checkpoint data into an isolated copy of this instance."""
        self._validate_checkpoint_structure(checkpoint)
        staged = object.__new__(self.__class__)
        staged.__dict__ = copy.deepcopy(self.__dict__)
        checkpoint_config_names = []

        if staged.model is None:
            build_input_shape = checkpoint.get("build_input_shape")
            if build_input_shape is None:
                raise ValueError(
                    "This legacy checkpoint does not include build_input_shape. "
                    "Build the model manually before loading it."
                )

            init_config = checkpoint.get("init_config", {})
            if not isinstance(init_config, dict):
                raise TypeError("Checkpoint init_config must be a dictionary.")
            for name, value in init_config.items():
                if hasattr(staged, name):
                    setattr(staged, name, copy.deepcopy(value))
                    checkpoint_config_names.append(name)

            try:
                staged._build_input_shape = tuple(build_input_shape)
            except TypeError as error:
                raise TypeError(
                    "Checkpoint build_input_shape must be an iterable shape."
                ) from error
            staged.model = staged._build_model(staged._build_input_shape)
            staged.model = staged.model.to(staged.device)

        checkpoint_state = checkpoint.get("model_state_dict")
        staged._validate_checkpoint_state_compatibility(
            staged.model,
            checkpoint_state,
        )
        staged.model.load_state_dict(checkpoint_state)
        staged._require_finite_parameters()
        staged.is_fitted = checkpoint.get("is_fitted", False)
        if staged._build_input_shape is None:
            shape = checkpoint.get("build_input_shape")
            if shape is not None:
                staged._build_input_shape = tuple(shape)

        return staged, checkpoint_config_names

    def _commit_staged_checkpoint(
        self,
        staged,
        checkpoint_config_names: list[str],
    ) -> None:
        """Commit an already validated staged checkpoint atomically."""
        for name in checkpoint_config_names:
            setattr(self, name, getattr(staged, name))
        self.model = staged.model
        self._build_input_shape = staged._build_input_shape
        self.is_fitted = staged.is_fitted

    @staticmethod
    def _loss_to_sample_total(
        loss: torch.Tensor,
        batch_sample_count: int,
        criterion: nn.Module,
    ) -> float:
        """Convert a scalar batch loss to a sample-total contribution.

        Mean-reduced and custom scalar criteria are treated as batch means.
        Sum-reduced criteria are already totals. Epoch histories are then
        divided by the number of samples, so short final batches are weighted
        correctly.
        """
        reduction = getattr(criterion, "reduction", None)
        value = float(loss.item())
        if reduction == "sum":
            return value
        return value * batch_sample_count

    def fit(
        self,
        X: np.ndarray,
        y: np.ndarray | None = None,
        validation_split: float = 0.2,
        epochs: int = 500,
        batch_size: int = 64,
        learning_rate: float = 1e-3,
        optimizer: torch.optim.Optimizer | None = None,
        criterion: nn.Module | None = None,
        patience: int = 20,
        verbose: int = 1,
        validation_data: tuple[np.ndarray, np.ndarray | None] | None = None,
        **kwargs,
    ) -> dict[str, list]:
        """Fit a reconstruction model with finite, sample-weighted losses.

        Parameters
        ----------
        validation_data : tuple, optional
            An explicit ``(X_validation, y_validation)`` pair. When supplied,
            ``validation_split`` is ignored, all of ``X`` is used for
            optimisation, and exactly these samples drive the validation loss
            and early stopping. ``y_validation`` may be ``None`` to reconstruct
            ``X_validation`` itself. Default is None, which keeps the historical
            random ``validation_split`` behaviour.
        """
        learning_rate = self._validate_learning_rate(learning_rate)
        if not isinstance(X, np.ndarray):
            raise TypeError("X must be a NumPy array.")
        if y is None:
            y = self._get_reconstruction_target(X)

        self._validate_fit_inputs(
            X,
            y,
            validation_split,
            batch_size,
            epochs,
            patience,
            validation_data=validation_data,
        )
        (
            X_train_array,
            y_train_array,
            X_validation_array,
            y_validation_array,
        ) = self._resolve_fit_partitions(X, y, validation_split, validation_data)
        self._validate_or_set_build_input_shape(tuple(X.shape))
        self.is_fitted = False

        if self.model is None:
            self.model = self._build_model(X.shape, **kwargs)
            self.model = self.model.to(self.device)

        avoid_singleton = self._requires_non_singleton_training_batches()
        if optimizer is None:
            optimizer = torch.optim.Adam(self.model.parameters(), lr=learning_rate)
        if criterion is None:
            criterion = nn.MSELoss()

        X_train = torch.as_tensor(
            X_train_array, dtype=torch.float32, device=self.device
        )
        y_train = torch.as_tensor(
            y_train_array, dtype=torch.float32, device=self.device
        )
        X_validation = torch.as_tensor(
            X_validation_array, dtype=torch.float32, device=self.device
        )
        y_validation = torch.as_tensor(
            y_validation_array, dtype=torch.float32, device=self.device
        )

        history = {"train_loss": [], "val_loss": []}
        best_validation_loss = float("inf")
        patience_counter = 0
        best_model_state = None

        epoch_range = range(epochs)
        progress_bar = None
        if verbose > 0:
            progress_bar = tqdm(epoch_range, desc="Training", unit="epoch")
            epoch_range = progress_bar

        for epoch in epoch_range:
            self.model.train()
            train_total = 0.0
            train_sample_count = 0
            for start, stop in self._batch_slices(
                len(X_train),
                batch_size,
                avoid_singleton=avoid_singleton,
            ):
                batch_X = X_train[start:stop]
                batch_y = y_train[start:stop]
                current_batch_size = stop - start

                optimizer.zero_grad()
                output = self.model(batch_X)
                self._require_matching_output_shape(output, batch_y, "Training")
                self._require_finite_tensor(output, "Training output")
                self._require_finite_buffers()
                loss = criterion(output, batch_y)
                self._require_scalar_loss(loss)
                self._require_finite_loss(loss, "Training")
                loss.backward()
                self._require_finite_gradients()
                optimizer.step()
                self._require_finite_parameters()

                train_total += self._loss_to_sample_total(
                    loss,
                    current_batch_size,
                    criterion,
                )
                train_sample_count += current_batch_size

            train_loss = train_total / train_sample_count
            history["train_loss"].append(train_loss)

            self.model.eval()
            validation_total = 0.0
            validation_sample_count = 0
            with torch.no_grad():
                for start, stop in self._batch_slices(
                    len(X_validation),
                    batch_size,
                ):
                    batch_X = X_validation[start:stop]
                    batch_y = y_validation[start:stop]
                    current_batch_size = stop - start
                    output = self.model(batch_X)
                    self._require_matching_output_shape(output, batch_y, "Validation")
                    self._require_finite_tensor(output, "Validation output")
                    self._require_finite_parameters()
                    loss = criterion(output, batch_y)
                    self._require_scalar_loss(loss)
                    self._require_finite_loss(loss, "Validation")
                    validation_total += self._loss_to_sample_total(
                        loss,
                        current_batch_size,
                        criterion,
                    )
                    validation_sample_count += current_batch_size

            validation_loss = validation_total / validation_sample_count
            history["val_loss"].append(validation_loss)

            if validation_loss < best_validation_loss:
                best_validation_loss = validation_loss
                patience_counter = 0
                best_model_state = copy.deepcopy(self.model.state_dict())
            else:
                patience_counter += 1
                if patience_counter >= patience:
                    if progress_bar is not None:
                        progress_bar.set_postfix_str(
                            f"Early stopping at epoch {epoch + 1}"
                        )
                    break

            if progress_bar is not None:
                progress_bar.set_postfix_str(
                    f"Train Loss: {train_loss:.6f}, "
                    f"Val Loss: {validation_loss:.6f}, "
                    f"Patience: {patience_counter}/{patience}"
                )

        if best_model_state is None:
            raise FloatingPointError(
                "Training completed without a finite validation loss."
            )
        self.model.load_state_dict(best_model_state)
        self.is_fitted = True
        return history

    def predict(
        self, X: np.ndarray, batch_size: int = 64, verbose: int = 1
    ) -> np.ndarray:
        """
        Make predictions.

        Parameters
        ----------
        X : np.ndarray
            Input data.
        batch_size : int, optional
            Batch size for prediction. Default is 64.
        verbose : int, optional
            Verbosity level. If > 0, shows progress bar. Default is 1.

        Returns
        -------
        np.ndarray
            Predictions.

        Raises
        ------
        ValueError
            If model is not fitted.
        """

        if not self.is_fitted or self.model is None:
            raise ValueError("Model must be fitted before prediction.")
        self._validate_inference_inputs(X, batch_size, check_expected_shape=True)

        self.model.eval()
        X_tensor = torch.FloatTensor(X).to(self.device)
        predictions = []

        n_batches = (len(X) + batch_size - 1) // batch_size
        batch_range = range(0, len(X), batch_size)

        if verbose > 0 and n_batches > 1:
            batch_range = tqdm(
                batch_range, desc="Predicting", unit="batch", total=n_batches
            )

        with torch.no_grad():
            for i in batch_range:
                batch_X = X_tensor[i : i + batch_size]
                output = self.model(batch_X)
                self._require_finite_tensor(output, "Prediction output")
                self._require_finite_parameters()
                predictions.append(output.cpu().numpy())

        return np.concatenate(predictions, axis=0)

    def encode(
        self, X: np.ndarray, batch_size: int = 64, verbose: int = 1
    ) -> np.ndarray:
        """
        Encode input data to latent space.

        Parameters
        ----------
        X : np.ndarray
            Input data.
        batch_size : int, optional
            Batch size for encoding. Default is 64.
        verbose : int, optional
            Verbosity level. If > 0, shows progress bar. Default is 1.

        Returns
        -------
        np.ndarray
            Latent representations.

        Raises
        ------
        ValueError
            If model is not fitted or doesn't support encoding.
        """

        if not self.is_fitted or self.model is None:
            raise ValueError("Model must be fitted before encoding.")
        self._validate_inference_inputs(X, batch_size, check_expected_shape=True)

        # Check if model has encode_forward method
        if not hasattr(self.model, "encode_forward"):
            raise ValueError(
                f"Model {self.__class__.__name__} does not support encoding. "
                "The model must have an 'encode_forward' method."
            )

        self.model.eval()
        X_tensor = torch.FloatTensor(X).to(self.device)
        encodings = []

        n_batches = (len(X) + batch_size - 1) // batch_size
        batch_range = range(0, len(X), batch_size)

        if verbose > 0 and n_batches > 1:
            batch_range = tqdm(
                batch_range, desc="Encoding", unit="batch", total=n_batches
            )

        with torch.no_grad():
            for i in batch_range:
                batch_X = X_tensor[i : i + batch_size]
                encoding = self.model.encode_forward(batch_X)
                self._require_finite_tensor(encoding, "Encoding output")
                self._require_finite_parameters()
                encodings.append(encoding.cpu().numpy())

        return np.concatenate(encodings, axis=0)

    def decode(
        self,
        Z: np.ndarray,
        batch_size: int = 64,
        verbose: int = 1,
    ) -> np.ndarray:
        """Decode latent vectors into the reconstruction space."""
        if not self.is_fitted or self.model is None:
            raise ValueError("Model must be fitted before decoding.")
        if not hasattr(self.model, "decode_forward"):
            raise ValueError(
                f"Model {self.__class__.__name__} does not support decoding."
            )

        Z = np.asarray(Z)
        if Z.ndim == 1:
            Z = Z[None, :]
        self._validate_latent_inputs(Z, batch_size)

        self.model.eval()
        Z_tensor = torch.as_tensor(Z, dtype=torch.float32, device=self.device)
        outputs = []
        batch_range = range(0, len(Z), batch_size)
        n_batches = (len(Z) + batch_size - 1) // batch_size

        if verbose > 0 and n_batches > 1:
            batch_range = tqdm(
                batch_range,
                desc="Decoding",
                unit="batch",
                total=n_batches,
            )

        with torch.no_grad():
            for start in batch_range:
                output = self.model.decode_forward(Z_tensor[start : start + batch_size])
                self._require_finite_tensor(output, "Decoding output")
                self._require_finite_parameters()
                outputs.append(output.cpu().numpy())

        return np.concatenate(outputs, axis=0)

    def reconstruction_error(
        self,
        X: np.ndarray,
        y: np.ndarray | None = None,
        metric: str = "mse",
        reduction: str = "sample",
        batch_size: int = 64,
        verbose: int = 0,
        eps: float = 0.0,
    ):
        """Compute reconstruction error with the shared metrics module."""
        eps = _validate_eps(eps)
        self._validate_inference_inputs(X, batch_size, check_expected_shape=True)
        target = self._get_reconstruction_target(X) if y is None else y
        self._validate_target_shape(X, target)
        self._validate_finite_array(target, "y")
        prediction = self.predict(X, batch_size=batch_size, verbose=verbose)
        return reconstruction_error_metric(
            target,
            prediction,
            metric=metric,
            reduction=reduction,
            eps=eps,
        )

    def evaluate_reconstruction(
        self,
        X: np.ndarray,
        y: np.ndarray | None = None,
        metric: str = "mse",
        batch_size: int = 64,
        verbose: int = 0,
        eps: float = 0.0,
    ) -> dict[str, float]:
        """Return summary statistics for reconstruction error."""
        eps = _validate_eps(eps)
        self._validate_inference_inputs(X, batch_size, check_expected_shape=True)
        target = self._get_reconstruction_target(X) if y is None else y
        self._validate_target_shape(X, target)
        self._validate_finite_array(target, "y")
        prediction = self.predict(X, batch_size=batch_size, verbose=verbose)
        return evaluate_reconstruction_metric(
            target,
            prediction,
            metric=metric,
            eps=eps,
        )

    def evaluate(self, X: np.ndarray, **kwargs) -> dict[str, float]:
        """Alias for :meth:`evaluate_reconstruction`."""
        return self.evaluate_reconstruction(X, **kwargs)

    def save_pytorch_model(self, model_path: str, **kwargs):
        """Save weights and metadata required to rebuild the model."""
        if self.model is None:
            raise ValueError("PyTorch model must be built before saving.")
        if self._build_input_shape is None:
            raise ValueError(
                "The model input shape is unknown. Fit or load the model before saving."
            )

        torch.save(
            {
                "checkpoint_version": 2,
                "model_state_dict": self.model.state_dict(),
                "is_fitted": self.is_fitted,
                "model_class": self.__class__.__name__,
                "init_config": self._get_init_config(),
                "build_input_shape": self._build_input_shape,
            },
            model_path,
            **kwargs,
        )
        self.logger.info(f"PyTorch model saved to {model_path}")

    def load_pytorch_model(
        self,
        model_path: str,
        map_location=None,
        **kwargs,
    ):
        """Load a checkpoint into this instance."""
        if map_location is None:
            map_location = self.device

        checkpoint = torch.load(
            model_path,
            map_location=map_location,
            **kwargs,
        )
        self._validate_checkpoint_structure(checkpoint)
        checkpoint_class = checkpoint.get("model_class")
        if checkpoint_class and checkpoint_class != self.__class__.__name__:
            raise ValueError(
                f"Checkpoint contains {checkpoint_class}, "
                f"not {self.__class__.__name__}."
            )

        staged, checkpoint_config_names = self._stage_checkpoint(checkpoint)
        self._commit_staged_checkpoint(staged, checkpoint_config_names)
        self.logger.info(f"PyTorch model loaded from {model_path}")
        return self

    @classmethod
    def from_pytorch_model(
        cls,
        model_path: str,
        device: str | torch.device | None = "cpu",
        map_location="cpu",
        **kwargs,
    ):
        """Create a fitted model from a self-describing checkpoint."""
        checkpoint = torch.load(
            model_path,
            map_location=map_location,
            **kwargs,
        )
        cls._validate_checkpoint_structure(checkpoint)
        checkpoint_class = checkpoint.get("model_class")
        if checkpoint_class and checkpoint_class != cls.__name__:
            raise ValueError(
                f"Checkpoint contains {checkpoint_class}, not {cls.__name__}."
            )

        init_config = checkpoint.get("init_config")
        build_input_shape = checkpoint.get("build_input_shape")
        if init_config is None or build_input_shape is None:
            raise ValueError(
                "Automatic loading requires a version-2 checkpoint containing "
                "init_config and build_input_shape."
            )

        if not isinstance(init_config, dict):
            raise TypeError("Checkpoint init_config must be a dictionary.")
        instance = cls(device=device, **copy.deepcopy(init_config))
        staged, _ = instance._stage_checkpoint(checkpoint)
        return staged
