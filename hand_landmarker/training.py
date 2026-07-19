"""TensorFlow 2.9 training orchestration for the fixed Hand Landmarker API.

The data module owns canonical JSONL parsing and sampling.  Its public
``create_sequences(config)`` function returns ``(train_seq, val_seq,
data_report)``; each sequence batch is ``(x, targets, sample_weights)``.  The
extra ``structure`` weight is non-zero only for valid human-Gold positives.

TensorFlow is imported only inside :func:`train_from_config` so repository
inspection remains possible before the documented server environment exists.
"""

from __future__ import annotations

import copy
import datetime as _datetime
import json
import math
import os
import platform
import random
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Mapping, MutableMapping, Optional, Sequence, Tuple

from .config import resolve_path
from .contracts import (
    STRUCTURAL_BONES,
    normalize_supervision_tier_loss_weights,
    validate_model_checkpoint_stage,
)
from .io_utils import sha256_file, write_json


_SEMANTIC_OUTPUTS = ("landmarks", "hand_flag", "handedness")
_KERAS_OUTPUT_NAMES = ("convld_21_2d", "activation_handflag", "activation_handedness")
_EXPECTED_INPUT_SHAPE = (None, 1, 256, 256)
_EXPECTED_OUTPUT_SHAPES = (
    (None, 1, 1, 42),
    (None, 1, 1, 1),
    (None, 1, 1, 1),
)
def _jsonable(value: Any) -> Any:
    """Convert common runtime values to stable JSON-compatible structures."""

    if isinstance(value, Path):
        return str(value)
    if isinstance(value, Mapping):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    if hasattr(value, "item"):
        try:
            return value.item()
        except (TypeError, ValueError):
            pass
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    return str(value)


def _utc_now() -> str:
    return _datetime.datetime.now(_datetime.timezone.utc).isoformat()


def _require_mapping(config: Mapping[str, Any], key: str) -> Mapping[str, Any]:
    value = config.get(key)
    if not isinstance(value, Mapping):
        raise KeyError("Configuration key {!r} must be a mapping".format(key))
    return value


def _configured_paths(value: Any, config: Mapping[str, Any]) -> List[Path]:
    if value is None:
        return []
    values = value if isinstance(value, (list, tuple)) else [value]
    return [resolve_path(str(item), config) for item in values]


def _hash_records(value: Any, config: Mapping[str, Any]) -> List[Dict[str, Any]]:
    records: List[Dict[str, Any]] = []
    for path in _configured_paths(value, config):
        record: Dict[str, Any] = {"path": str(path), "exists": path.is_file()}
        if path.is_file():
            record["size_bytes"] = int(path.stat().st_size)
            record["sha256"] = sha256_file(path)
        records.append(record)
    return records


def _git_metadata(repo_root: Path) -> Dict[str, Any]:
    def run(*args: str) -> Optional[str]:
        try:
            completed = subprocess.run(
                ["git", *args],
                cwd=str(repo_root),
                check=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                universal_newlines=True,
            )
        except (OSError, subprocess.CalledProcessError):
            return None
        return completed.stdout.strip()

    commit = run("rev-parse", "HEAD")
    status = run("status", "--short")
    return {
        "commit": commit,
        "dirty": bool(status) if status is not None else None,
        "status_short": status.splitlines() if status else [],
    }


def _prepare_environment(config: Mapping[str, Any]) -> None:
    runtime = config.get("runtime", {})
    if bool(runtime.get("deterministic", False)):
        os.environ.setdefault("TF_DETERMINISTIC_OPS", "1")
        os.environ.setdefault("PYTHONHASHSEED", str(config.get("experiment", {}).get("seed", 0)))


def _configure_tensorflow(tf: Any, config: Mapping[str, Any]) -> Dict[str, Any]:
    runtime = config.get("runtime", {})
    environment = config.get("environment", {})
    physical_gpus = list(tf.config.list_physical_devices("GPU"))
    if bool(environment.get("require_gpu", False)) and not physical_gpus:
        raise RuntimeError("environment.require_gpu=true, but TensorFlow found no GPU")

    if bool(runtime.get("gpu_memory_growth", False)):
        for device in physical_gpus:
            try:
                tf.config.experimental.set_memory_growth(device, True)
            except (RuntimeError, ValueError) as exc:
                raise RuntimeError("Could not enable GPU memory growth for {}".format(device)) from exc

    deterministic_enabled = False
    if bool(runtime.get("deterministic", False)):
        enable = getattr(tf.config.experimental, "enable_op_determinism", None)
        if enable is not None:
            try:
                enable()
                deterministic_enabled = True
            except (RuntimeError, TypeError):
                # TF_DETERMINISTIC_OPS was set before importing TensorFlow and
                # remains the compatibility path on older TF 2.9 builds.
                deterministic_enabled = False

    return {
        "tensorflow_version": str(tf.__version__),
        "physical_gpus": [str(device) for device in physical_gpus],
        "deterministic_api_enabled": deterministic_enabled,
        "tf_deterministic_ops": os.environ.get("TF_DETERMINISTIC_OPS"),
    }


def _set_random_seeds(tf: Any, np: Any, seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)


def assert_model_interface(model: Any) -> Dict[str, Any]:
    """Fail fast if a model revision changes any board-visible Keras shape/order."""

    input_shape = tuple(model.input_shape)
    output_shapes = tuple(tuple(shape) for shape in model.output_shape)
    output_names = tuple(str(name) for name in model.output_names)
    if input_shape != _EXPECTED_INPUT_SHAPE:
        raise ValueError(
            "Hand Landmarker input shape changed: got {}, expected {}".format(
                input_shape, _EXPECTED_INPUT_SHAPE
            )
        )
    if output_shapes != _EXPECTED_OUTPUT_SHAPES:
        raise ValueError(
            "Hand Landmarker output shapes changed: got {}, expected {}".format(
                output_shapes, _EXPECTED_OUTPUT_SHAPES
            )
        )
    if output_names != _KERAS_OUTPUT_NAMES:
        raise ValueError(
            "Hand Landmarker output order/names changed: got {}, expected {}".format(
                output_names, _KERAS_OUTPUT_NAMES
            )
        )
    return {
        "input_shape": list(input_shape),
        "output_shapes": [list(shape) for shape in output_shapes],
        "output_names": list(output_names),
        "output_semantics": list(_SEMANTIC_OUTPUTS),
        "parameter_count": int(model.count_params()),
    }


def _validate_loss_config(config: Mapping[str, Any]) -> Dict[str, Dict[str, Any]]:
    losses = _require_mapping(config, "losses")
    expected = {
        "landmarks": "huber",
        "hand_flag": "binary_crossentropy",
        "handedness": "binary_crossentropy",
        "bone_vector": "huber",
        "spread_ratio": "huber",
    }
    normalized: Dict[str, Dict[str, Any]] = {}
    for semantic, expected_name in expected.items():
        item = losses.get(semantic)
        if not isinstance(item, Mapping):
            raise KeyError("losses.{} must be configured".format(semantic))
        name = str(item.get("name", "")).strip().lower()
        if name != expected_name:
            raise ValueError(
                "losses.{}.name must be {!r}; got {!r}".format(semantic, expected_name, name)
            )
        coefficient = float(item.get("coefficient", 1.0))
        if not math.isfinite(coefficient) or coefficient < 0.0:
            raise ValueError("losses.{}.coefficient must be finite and non-negative".format(semantic))
        normalized[semantic] = dict(item)
        normalized[semantic]["name"] = name
        normalized[semantic]["coefficient"] = coefficient

    for semantic in ("landmarks", "bone_vector", "spread_ratio"):
        delta = float(normalized[semantic].get("delta", 0.01))
        if not math.isfinite(delta) or delta <= 0.0:
            raise ValueError("losses.{}.delta must be finite and positive".format(semantic))
        normalized[semantic]["delta"] = delta
    for semantic in ("hand_flag", "handedness"):
        if bool(normalized[semantic].get("from_logits", False)):
            raise ValueError("{} uses a sigmoid model output; from_logits must be false".format(semantic))
    if not bool(losses.get("honor_record_loss_weights", True)):
        raise ValueError("losses.honor_record_loss_weights must remain true for canonical labels")
    normalize_supervision_tier_loss_weights(losses.get("supervision_tier_weights"))
    return normalized


def _build_optimizer(tf: Any, config: Mapping[str, Any]):
    training = _require_mapping(config, "training")
    optimizer_config = training.get("optimizer")
    if not isinstance(optimizer_config, Mapping):
        raise KeyError("training.optimizer must be configured")
    name = str(optimizer_config.get("name", "adam")).strip().lower()
    if name != "adam":
        raise ValueError("Only the required Adam optimizer is supported; got {!r}".format(name))

    clipnorm_value = training.get("gradient_clip_norm")
    kwargs: Dict[str, Any] = {
        "learning_rate": float(optimizer_config.get("learning_rate", 1e-3)),
        "beta_1": float(optimizer_config.get("beta_1", 0.9)),
        "beta_2": float(optimizer_config.get("beta_2", 0.999)),
        "epsilon": float(optimizer_config.get("epsilon", 1e-7)),
    }
    if clipnorm_value is not None:
        clipnorm = float(clipnorm_value)
        if not math.isfinite(clipnorm) or clipnorm <= 0.0:
            raise ValueError("training.gradient_clip_norm must be positive when configured")
        kwargs["clipnorm"] = clipnorm
    optimizer = tf.keras.optimizers.Adam(**kwargs)
    if bool(training.get("mixed_precision", False)):
        optimizer = tf.keras.mixed_precision.LossScaleOptimizer(optimizer)
    return optimizer


def _build_weighted_trainer(tf: Any, backbone: Any, loss_config: Mapping[str, Mapping[str, Any]]):
    coefficients = tuple(float(loss_config[name]["coefficient"]) for name in _SEMANTIC_OUTPUTS)
    huber_delta = float(loss_config["landmarks"]["delta"])
    bone_coefficient = float(loss_config["bone_vector"]["coefficient"])
    bone_delta = float(loss_config["bone_vector"]["delta"])
    spread_coefficient = float(loss_config["spread_ratio"]["coefficient"])
    spread_delta = float(loss_config["spread_ratio"]["delta"])
    structure_enabled = bone_coefficient > 0.0 or spread_coefficient > 0.0
    metric_coefficients = (*coefficients, bone_coefficient, spread_coefficient)

    class WeightedPerSampleMean(tf.keras.metrics.Metric):
        def __init__(self, name: str):
            super().__init__(name=name, dtype=tf.float32)
            self.weighted_total = self.add_weight(name="weighted_total", initializer="zeros")
            self.weight_total = self.add_weight(name="weight_total", initializer="zeros")

        def update_state(self, values: Any, sample_weight: Any) -> None:
            values = tf.reshape(tf.cast(values, tf.float32), [-1])
            sample_weight = tf.reshape(tf.cast(sample_weight, tf.float32), [-1])
            self.weighted_total.assign_add(tf.reduce_sum(values * sample_weight))
            self.weight_total.assign_add(tf.reduce_sum(sample_weight))

        def result(self):
            epsilon = tf.cast(tf.keras.backend.epsilon(), tf.float32)
            return self.weighted_total / tf.maximum(self.weight_total, epsilon)

        def reset_state(self) -> None:
            self.weighted_total.assign(0.0)
            self.weight_total.assign(0.0)

        def reset_states(self) -> None:  # TensorFlow 2.9 compatibility alias.
            self.reset_state()

    class CompositeLossMetric(tf.keras.metrics.Metric):
        """Expose the coefficient-weighted epoch head metrics as total_loss."""

        def __init__(self, components: Sequence[Any], regularization: Any):
            super().__init__(name="total_loss", dtype=tf.float32)
            self.components = tuple(components)
            self.regularization = regularization

        def update_state(self, *args: Any, **kwargs: Any) -> None:
            del args, kwargs

        def result(self):
            total = tf.cast(self.regularization.result(), tf.float32)
            for coefficient, component in zip(metric_coefficients, self.components):
                total = total + tf.cast(coefficient, tf.float32) * tf.cast(
                    component.result(), tf.float32
                )
            return total

        def reset_state(self) -> None:
            # The component metrics own and reset all state.
            return None

        def reset_states(self) -> None:  # TensorFlow 2.9 compatibility alias.
            return None

    class WeightedMultiHeadModel(tf.keras.Model):
        def __init__(self, model: Any):
            super().__init__(name="weighted_{}".format(model.name))
            self.backbone = model
            # Gradients use per-batch normalization below. Epoch metrics instead
            # aggregate each head's weighted numerator and denominator globally,
            # so batches with little effective supervision are not over-counted.
            self.landmarks_loss_metric = WeightedPerSampleMean("landmarks_loss")
            self.hand_flag_loss_metric = WeightedPerSampleMean("hand_flag_loss")
            self.handedness_loss_metric = WeightedPerSampleMean("handedness_loss")
            self.bone_vector_loss_metric = WeightedPerSampleMean("bone_vector_loss")
            self.spread_ratio_loss_metric = WeightedPerSampleMean("spread_ratio_loss")
            self.regularization_loss_metric = tf.keras.metrics.Mean(
                name="regularization_loss", dtype=tf.float32
            )
            self.total_loss_metric = CompositeLossMetric(
                (
                    self.landmarks_loss_metric,
                    self.hand_flag_loss_metric,
                    self.handedness_loss_metric,
                    self.bone_vector_loss_metric,
                    self.spread_ratio_loss_metric,
                ),
                self.regularization_loss_metric,
            )
            self.landmark_mae_metric = WeightedPerSampleMean("landmark_mae")
            self.hand_flag_accuracy_metric = WeightedPerSampleMean("hand_flag_accuracy")
            self.handedness_accuracy_metric = WeightedPerSampleMean("handedness_accuracy")

        @property
        def metrics(self) -> List[Any]:
            return [
                self.total_loss_metric,
                self.landmarks_loss_metric,
                self.hand_flag_loss_metric,
                self.handedness_loss_metric,
                self.bone_vector_loss_metric,
                self.spread_ratio_loss_metric,
                self.regularization_loss_metric,
                self.landmark_mae_metric,
                self.hand_flag_accuracy_metric,
                self.handedness_accuracy_metric,
            ]

        def call(self, inputs: Any, training: bool = False):
            return self.backbone(inputs, training=training)

        @staticmethod
        def _ordered(container: Any, label: str) -> Tuple[Any, Any, Any]:
            if isinstance(container, Mapping):
                values = []
                for semantic, keras_name in zip(_SEMANTIC_OUTPUTS, _KERAS_OUTPUT_NAMES):
                    if semantic in container:
                        values.append(container[semantic])
                    elif keras_name in container:
                        values.append(container[keras_name])
                    else:
                        raise KeyError("{} is missing key {!r}".format(label, semantic))
                return tuple(values)  # type: ignore[return-value]
            if isinstance(container, (list, tuple)) and len(container) == 3:
                return container[0], container[1], container[2]
            raise ValueError("{} must be a three-item list/tuple or semantic mapping".format(label))

        @staticmethod
        def _weights(weight: Any, batch_size: Any):
            if weight is None:
                return tf.ones([batch_size], dtype=tf.float32)
            value = tf.cast(tf.convert_to_tensor(weight), tf.float32)
            if value.shape.rank == 0:
                value = tf.fill([batch_size], value)
            else:
                value = tf.reshape(value, [-1])
                tf.debugging.assert_equal(
                    tf.shape(value)[0],
                    batch_size,
                    "each head must provide exactly one sample weight per record",
                )
            tf.debugging.assert_all_finite(value, "sample weights must be finite")
            tf.debugging.assert_greater_equal(value, tf.zeros_like(value), "sample weights must be non-negative")
            return value

        @staticmethod
        def _weighted_mean(values: Any, weights: Any):
            values = tf.cast(values, tf.float32)
            weights = tf.cast(weights, tf.float32)
            numerator = tf.reduce_sum(values * weights)
            epsilon = tf.cast(tf.keras.backend.epsilon(), tf.float32)
            denominator = tf.maximum(tf.reduce_sum(weights), epsilon)
            return numerator / denominator

        @staticmethod
        def _binary_crossentropy_per_sample(target: Any, prediction: Any):
            target = tf.cast(target, tf.float32)
            prediction = tf.cast(prediction, tf.float32)
            epsilon = tf.cast(tf.keras.backend.epsilon(), tf.float32)
            prediction = tf.clip_by_value(prediction, epsilon, 1.0 - epsilon)
            losses = -target * tf.math.log(prediction) - (1.0 - target) * tf.math.log(1.0 - prediction)
            return tf.reduce_mean(losses, axis=1)

        @staticmethod
        def _huber_per_sample(target: Any, prediction: Any):
            error = tf.abs(tf.cast(target, tf.float32) - tf.cast(prediction, tf.float32))
            quadratic = tf.minimum(error, tf.cast(huber_delta, tf.float32))
            linear = error - quadratic
            losses = 0.5 * tf.square(quadratic) + tf.cast(huber_delta, tf.float32) * linear
            return tf.reduce_mean(losses, axis=1)

        @staticmethod
        def _huber_values(error: Any, delta: float):
            absolute = tf.abs(tf.cast(error, tf.float32))
            quadratic = tf.minimum(absolute, tf.cast(delta, tf.float32))
            return 0.5 * tf.square(quadratic) + tf.cast(delta, tf.float32) * (absolute - quadratic)

        def _structure_terms(self, target: Any, prediction: Any):
            target_points = tf.reshape(target, [-1, 21, 2])
            prediction_points = tf.reshape(prediction, [-1, 21, 2])
            target_bones = tf.stack(
                [target_points[:, end] - target_points[:, start] for start, end in STRUCTURAL_BONES],
                axis=1,
            )
            prediction_bones = tf.stack(
                [prediction_points[:, end] - prediction_points[:, start] for start, end in STRUCTURAL_BONES],
                axis=1,
            )
            bone = tf.reduce_mean(
                self._huber_values(prediction_bones - target_bones, bone_delta), axis=[1, 2]
            )
            target_centered = target_points - target_points[:, 0:1, :]
            prediction_centered = prediction_points - prediction_points[:, 0:1, :]
            target_spread = tf.sqrt(tf.reduce_mean(tf.square(target_centered), axis=[1, 2]))
            prediction_spread = tf.sqrt(tf.reduce_mean(tf.square(prediction_centered), axis=[1, 2]))
            epsilon = tf.cast(1.0e-6, tf.float32)
            log_ratio = tf.math.log((prediction_spread + epsilon) / (target_spread + epsilon))
            spread = self._huber_values(log_ratio, spread_delta)
            return bone, spread

        def _loss_terms(self, x: Any, y: Any, sample_weight: Any, training: bool):
            targets = self._ordered(y, "targets")
            predictions = self._ordered(self.backbone(x, training=training), "model outputs")
            if sample_weight is None:
                raw_weights = (None, None, None)
                raw_structure_weight = None
            else:
                raw_weights = self._ordered(sample_weight, "sample weights")
                raw_structure_weight = (
                    sample_weight.get("structure") if isinstance(sample_weight, Mapping) else None
                )
            if structure_enabled and raw_structure_weight is None:
                raise ValueError(
                    "Enabled structure losses require an explicit per-record structure mask"
                )

            batch_size = tf.shape(x)[0]
            y_landmarks = tf.reshape(tf.cast(targets[0], tf.float32), [batch_size, 42])
            p_landmarks = tf.reshape(tf.cast(predictions[0], tf.float32), [batch_size, 42])
            y_flag = tf.reshape(tf.cast(targets[1], tf.float32), [batch_size, 1])
            p_flag = tf.reshape(tf.cast(predictions[1], tf.float32), [batch_size, 1])
            y_handed = tf.reshape(tf.cast(targets[2], tf.float32), [batch_size, 1])
            p_handed = tf.reshape(tf.cast(predictions[2], tf.float32), [batch_size, 1])

            weights = tuple(self._weights(value, batch_size) for value in raw_weights)
            structure_weight = (
                self._weights(raw_structure_weight, batch_size)
                if raw_structure_weight is not None
                else tf.zeros([batch_size], dtype=tf.float32)
            )
            per_sample = (
                self._huber_per_sample(y_landmarks, p_landmarks),
                self._binary_crossentropy_per_sample(y_flag, p_flag),
                self._binary_crossentropy_per_sample(y_handed, p_handed),
            )
            head_losses = tuple(
                self._weighted_mean(loss, weight) for loss, weight in zip(per_sample, weights)
            )
            bone_per_sample, spread_per_sample = self._structure_terms(y_landmarks, p_landmarks)
            bone_loss = self._weighted_mean(bone_per_sample, structure_weight)
            spread_loss = self._weighted_mean(spread_per_sample, structure_weight)
            if self.backbone.losses:
                regularization_loss = tf.add_n(
                    [tf.cast(value, tf.float32) for value in self.backbone.losses]
                )
            else:
                regularization_loss = tf.constant(0.0, dtype=tf.float32)
            total_loss = regularization_loss
            for coefficient, value in zip(coefficients, head_losses):
                total_loss = total_loss + tf.cast(coefficient, tf.float32) * value
            total_loss = total_loss + tf.cast(bone_coefficient, tf.float32) * bone_loss
            total_loss = total_loss + tf.cast(spread_coefficient, tf.float32) * spread_loss

            auxiliary = {
                "landmark_mae": tf.reduce_mean(tf.abs(y_landmarks - p_landmarks), axis=1),
                "hand_flag_accuracy": tf.reduce_mean(
                    tf.cast(tf.equal(y_flag >= 0.5, p_flag >= 0.5), tf.float32), axis=1
                ),
                "handedness_accuracy": tf.reduce_mean(
                    tf.cast(tf.equal(y_handed >= 0.5, p_handed >= 0.5), tf.float32), axis=1
                ),
            }
            return (
                total_loss,
                regularization_loss,
                (*per_sample, bone_per_sample, spread_per_sample),
                (*weights, structure_weight, structure_weight),
                auxiliary,
            )

        def _update_metrics(
            self,
            regularization_loss: Any,
            per_sample: Sequence[Any],
            weights: Sequence[Any],
            auxiliary: Mapping[str, Any],
        ) -> None:
            self.landmarks_loss_metric.update_state(per_sample[0], weights[0])
            self.hand_flag_loss_metric.update_state(per_sample[1], weights[1])
            self.handedness_loss_metric.update_state(per_sample[2], weights[2])
            self.bone_vector_loss_metric.update_state(per_sample[3], weights[3])
            self.spread_ratio_loss_metric.update_state(per_sample[4], weights[4])
            self.regularization_loss_metric.update_state(regularization_loss)
            self.landmark_mae_metric.update_state(auxiliary["landmark_mae"], weights[0])
            self.hand_flag_accuracy_metric.update_state(auxiliary["hand_flag_accuracy"], weights[1])
            self.handedness_accuracy_metric.update_state(auxiliary["handedness_accuracy"], weights[2])

        def train_step(self, data: Any):
            x, y, sample_weight = tf.keras.utils.unpack_x_y_sample_weight(data)
            with tf.GradientTape() as tape:
                (
                    total_loss,
                    regularization_loss,
                    per_sample,
                    weights,
                    auxiliary,
                ) = self._loss_terms(x, y, sample_weight, training=True)
                gradient_loss = total_loss
                get_scaled_loss = getattr(self.optimizer, "get_scaled_loss", None)
                if get_scaled_loss is not None:
                    gradient_loss = get_scaled_loss(gradient_loss)

            gradients = tape.gradient(gradient_loss, self.backbone.trainable_variables)
            get_unscaled_gradients = getattr(self.optimizer, "get_unscaled_gradients", None)
            if get_unscaled_gradients is not None:
                gradients = get_unscaled_gradients(gradients)
            pairs = [
                (gradient, variable)
                for gradient, variable in zip(gradients, self.backbone.trainable_variables)
                if gradient is not None
            ]
            if not pairs:
                raise RuntimeError("No gradients were produced for the Hand Landmarker")
            self.optimizer.apply_gradients(pairs)
            self._update_metrics(regularization_loss, per_sample, weights, auxiliary)
            return {metric.name: metric.result() for metric in self.metrics}

        def test_step(self, data: Any):
            x, y, sample_weight = tf.keras.utils.unpack_x_y_sample_weight(data)
            (
                total_loss,
                regularization_loss,
                per_sample,
                weights,
                auxiliary,
            ) = self._loss_terms(x, y, sample_weight, training=False)
            self._update_metrics(regularization_loss, per_sample, weights, auxiliary)
            return {metric.name: metric.result() for metric in self.metrics}

    return WeightedMultiHeadModel(backbone)


def _atomic_save_weights(model: Any, path: Path) -> None:
    """Write a single HDF5 weights file even when config uses a .keras suffix."""

    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists() and path.is_dir():
        raise IsADirectoryError("Checkpoint path must be a file, not a SavedModel directory: {}".format(path))
    temporary = path.with_name(path.name + ".tmp.h5")
    if temporary.exists():
        temporary.unlink()
    try:
        model.save_weights(str(temporary), save_format="h5")
        os.replace(str(temporary), str(path))
    finally:
        if temporary.exists():
            temporary.unlink()


def _state_directory(weights_path: Path) -> Path:
    return Path(str(weights_path) + ".state")


def _state_json_path(weights_path: Path) -> Path:
    return Path(str(weights_path) + ".state.json")


def _read_state_json(weights_path: Path) -> Dict[str, Any]:
    path = _state_json_path(weights_path)
    if not path.is_file():
        return {}
    try:
        with path.open("r", encoding="utf-8") as handle:
            value = json.load(handle)
    except (OSError, ValueError):
        return {}
    return value if isinstance(value, dict) else {}


def _monitor_mode(monitor: str, configured: Optional[str] = None) -> str:
    if configured is not None:
        mode = str(configured).lower()
        if mode not in {"min", "max"}:
            raise ValueError("monitor mode must be min or max; got {!r}".format(configured))
        return mode

    name = str(monitor).lower()
    minimize_tokens = ("loss", "mae", "mse", "rmse", "error", "nme", "distance")
    maximize_tokens = ("accuracy", "acc", "auc", "precision", "recall", "f1", "pck", "iou")
    minimize = any(token in name for token in minimize_tokens)
    maximize = any(token in name for token in maximize_tokens)
    if minimize == maximize:
        raise ValueError(
            "Could not infer whether monitor {!r} should be minimized or maximized; "
            "set mode: min or mode: max explicitly".format(monitor)
        )
    return "min" if minimize else "max"


def _multitask_monitor_value(
    logs: Mapping[str, Any], monitor_config: Mapping[str, Any]
) -> Optional[float]:
    """Return a geometry-first validation score, or None without validation logs."""

    required = (
        "val_landmark_mae",
        "val_hand_flag_accuracy",
        "val_handedness_accuracy",
    )
    if any(logs.get(name) is None for name in required):
        return None
    weights = {
        "landmark_mae": float(monitor_config.get("landmark_mae_weight", 1.0)),
        "hand_flag_error": float(monitor_config.get("hand_flag_error_weight", 0.02)),
        "handedness_error": float(monitor_config.get("handedness_error_weight", 0.005)),
    }
    if any(not math.isfinite(value) or value < 0.0 for value in weights.values()):
        raise ValueError("training.multitask_monitor weights must be finite and non-negative")
    values = {name: float(logs[name]) for name in required}
    if any(not math.isfinite(value) for value in values.values()):
        raise FloatingPointError("Cannot compute multitask monitor from non-finite validation metrics")
    return (
        weights["landmark_mae"] * values["val_landmark_mae"]
        + weights["hand_flag_error"] * (1.0 - values["val_hand_flag_accuracy"])
        + weights["handedness_error"] * (1.0 - values["val_handedness_accuracy"])
    )


def _build_callbacks(
    tf: Any,
    trainer: Any,
    config: Mapping[str, Any],
    best_path: Path,
    last_path: Path,
    logs_dir: Path,
    training_checkpoint: Any,
    epoch_variable: Any,
    resume_enabled: bool,
) -> List[Any]:
    training = _require_mapping(config, "training")
    outputs = _require_mapping(config, "outputs")
    checkpoint = training.get("checkpoint", {})
    early = training.get("early_stopping", {})
    lr_schedule = training.get("learning_rate_schedule", {})
    default_monitor = "val_total_loss"
    monitor = str(
        checkpoint.get("monitor")
        or early.get("monitor")
        or lr_schedule.get("monitor")
        or default_monitor
    )
    mode = _monitor_mode(monitor, checkpoint.get("mode"))

    lr_monitor = str(lr_schedule.get("monitor") or monitor)
    lr_mode = _monitor_mode(lr_monitor, lr_schedule.get("mode"))
    early_monitor = str(early.get("monitor") or monitor)
    early_mode = _monitor_mode(early_monitor, early.get("mode"))
    if bool(early.get("enabled", True)) and lr_monitor == early_monitor and lr_mode == early_mode:
        lr_patience = int(lr_schedule.get("patience", 4))
        early_patience = int(early.get("patience", 10))
        if lr_patience >= early_patience:
            raise ValueError(
                "ReduceLROnPlateau patience must be smaller than EarlyStopping patience "
                "when both monitor the same metric; got {} >= {}".format(
                    lr_patience, early_patience
                )
            )

    multitask_monitor = dict(training.get("multitask_monitor") or {})
    periodic = dict(training.get("periodic_checkpoint") or {})
    wall_time_value = training.get("max_wall_time_hours")
    wall_time_hours = None if wall_time_value in (None, "") else float(wall_time_value)
    if wall_time_hours is not None and (not math.isfinite(wall_time_hours) or wall_time_hours <= 0.0):
        raise ValueError("training.max_wall_time_hours must be a finite positive number or null")

    class DerivedMultitaskMonitor(tf.keras.callbacks.Callback):
        def on_epoch_end(self, epoch: int, logs: Optional[MutableMapping[str, Any]] = None) -> None:
            del epoch
            if logs is None:
                return
            value = _multitask_monitor_value(logs, multitask_monitor)
            if value is not None:
                logs[str(multitask_monitor.get("name", "val_multitask_score"))] = value

    class BackboneCheckpoint(tf.keras.callbacks.Callback):
        def __init__(self, path: Path, save_best_only: bool):
            super().__init__()
            self.path = path
            self.save_best_only = save_best_only
            state = _read_state_json(path) if resume_enabled else {}
            default_best = math.inf if mode == "min" else -math.inf
            try:
                self.best = float(state.get("best", default_best))
            except (TypeError, ValueError):
                self.best = default_best
            self.manager = tf.train.CheckpointManager(
                training_checkpoint,
                directory=str(_state_directory(path)),
                max_to_keep=1,
                checkpoint_name="ckpt",
            )

        def _improved(self, current: float) -> bool:
            return current < self.best if mode == "min" else current > self.best

        def on_epoch_end(self, epoch: int, logs: Optional[Mapping[str, Any]] = None) -> None:
            logs = logs or {}
            current_raw = logs.get(monitor)
            if current_raw is None:
                if self.save_best_only:
                    validation_frequency = int(config.get("validation", {}).get("every_epochs", 1))
                    if monitor.startswith("val_") and (int(epoch) + 1) % validation_frequency != 0:
                        return
                    raise KeyError(
                        "Checkpoint monitor {!r} was not produced; available logs: {}".format(
                            monitor, sorted(logs)
                        )
                    )
                current = float("nan")
            else:
                current = float(current_raw)

            if self.save_best_only and (not math.isfinite(current) or not self._improved(current)):
                return
            if self.save_best_only:
                self.best = current
            epoch_variable.assign(int(epoch) + 1)
            _atomic_save_weights(trainer.backbone, self.path)
            state_checkpoint = self.manager.save(checkpoint_number=int(epoch) + 1)
            write_json(
                _state_json_path(self.path),
                {
                    "completed_epoch": int(epoch) + 1,
                    "monitor": monitor,
                    "mode": mode,
                    "value": current if math.isfinite(current) else None,
                    "best": self.best if math.isfinite(self.best) else None,
                    "weights_path": str(self.path),
                    "training_state_checkpoint": state_checkpoint,
                    "saved_at_utc": _utc_now(),
                },
            )

    class PeriodicCheckpoint(tf.keras.callbacks.Callback):
        def __init__(self):
            super().__init__()
            self.every_epochs = int(periodic.get("every_epochs", 5))
            self.max_to_keep = int(periodic.get("max_to_keep", 16))
            if self.every_epochs <= 0 or self.max_to_keep <= 0:
                raise ValueError(
                    "training.periodic_checkpoint every_epochs and max_to_keep must be positive"
                )
            configured = outputs.get("periodic_checkpoints_dir")
            self.directory = (
                resolve_path(str(configured), config)
                if configured
                else best_path.parent / "periodic"
            )
            self.directory.mkdir(parents=True, exist_ok=True)
            self.manager = tf.train.CheckpointManager(
                training_checkpoint,
                directory=str(self.directory / "training_state"),
                max_to_keep=self.max_to_keep,
                checkpoint_name="ckpt",
            )

        def on_epoch_end(self, epoch: int, logs: Optional[Mapping[str, Any]] = None) -> None:
            completed_epoch = int(epoch) + 1
            if completed_epoch % self.every_epochs:
                return
            epoch_variable.assign(completed_epoch)
            weights_path = self.directory / "epoch_{:04d}.weights.h5".format(
                completed_epoch
            )
            _atomic_save_weights(trainer.backbone, weights_path)
            state_checkpoint = self.manager.save(checkpoint_number=completed_epoch)
            metrics = {}
            for key, value in sorted((logs or {}).items()):
                try:
                    number = float(value)
                except (TypeError, ValueError):
                    continue
                metrics[str(key)] = number if math.isfinite(number) else None
            write_json(
                weights_path.with_suffix(".json"),
                {
                    "completed_epoch": completed_epoch,
                    "weights_path": str(weights_path),
                    "training_state_checkpoint": state_checkpoint,
                    "metrics": metrics,
                    "saved_at_utc": _utc_now(),
                },
            )

    class MaxWallTime(tf.keras.callbacks.Callback):
        def __init__(self, hours: float):
            super().__init__()
            self.hours = float(hours)
            self.started = None

        def on_train_begin(self, logs=None) -> None:
            del logs
            self.started = time.monotonic()

        def on_epoch_end(self, epoch: int, logs: Optional[Mapping[str, Any]] = None) -> None:
            del logs
            if self.started is None:
                return
            elapsed_seconds = time.monotonic() - self.started
            if elapsed_seconds < self.hours * 3600.0:
                return
            self.model.stop_training = True
            write_json(
                logs_dir / "wall_time_stop.json",
                {
                    "completed_epoch": int(epoch) + 1,
                    "configured_hours": self.hours,
                    "elapsed_seconds": elapsed_seconds,
                    "stopped_at_utc": _utc_now(),
                },
            )

    class FailOnNonFinite(tf.keras.callbacks.Callback):
        @staticmethod
        def _check(where: str, logs: Optional[Mapping[str, Any]]) -> None:
            for key, value in (logs or {}).items():
                try:
                    number = float(value)
                except (TypeError, ValueError):
                    continue
                if not math.isfinite(number):
                    raise FloatingPointError("Non-finite metric at {}: {}={}".format(where, key, value))

        def on_train_batch_end(self, batch: int, logs: Optional[Mapping[str, Any]] = None) -> None:
            self._check("train batch {}".format(batch), logs)

        def on_test_batch_end(self, batch: int, logs: Optional[Mapping[str, Any]] = None) -> None:
            self._check("validation batch {}".format(batch), logs)

        def on_epoch_end(self, epoch: int, logs: Optional[Mapping[str, Any]] = None) -> None:
            self._check("epoch {}".format(epoch + 1), logs)

    logs_dir.mkdir(parents=True, exist_ok=True)
    callbacks: List[Any] = []
    if bool(multitask_monitor.get("enabled", False)):
        derived_name = str(multitask_monitor.get("name", "val_multitask_score"))
        if not derived_name.startswith("val_"):
            raise ValueError("training.multitask_monitor.name must start with val_")
        used_monitors = {monitor, lr_monitor, early_monitor}
        if derived_name not in used_monitors:
            raise ValueError(
                "Enabled multitask monitor {!r} must be used by checkpoint, LR, or early stopping".format(
                    derived_name
                )
            )
        callbacks.append(DerivedMultitaskMonitor())
    callbacks.extend([
        tf.keras.callbacks.CSVLogger(str(logs_dir / "history.csv"), append=resume_enabled),
        tf.keras.callbacks.TensorBoard(log_dir=str(logs_dir / "tensorboard"), update_freq="epoch"),
    ])
    if bool(config.get("runtime", {}).get("fail_on_nan", True)):
        callbacks.append(FailOnNonFinite())
    callbacks.extend(
        [
            BackboneCheckpoint(best_path, save_best_only=True),
            BackboneCheckpoint(last_path, save_best_only=False),
        ]
    )
    if bool(periodic.get("enabled", False)):
        callbacks.append(PeriodicCheckpoint())
    if wall_time_hours is not None:
        callbacks.append(MaxWallTime(wall_time_hours))
    if str(lr_schedule.get("name", "reduce_on_plateau")).lower() != "reduce_on_plateau":
        raise ValueError("Only training.learning_rate_schedule.name=reduce_on_plateau is supported")
    callbacks.append(
        tf.keras.callbacks.ReduceLROnPlateau(
            monitor=lr_monitor,
            factor=float(lr_schedule.get("factor", 0.5)),
            patience=int(lr_schedule.get("patience", 4)),
            min_delta=float(lr_schedule.get("min_delta", 1.0e-4)),
            min_lr=float(lr_schedule.get("min_learning_rate", 0.0)),
            cooldown=int(lr_schedule.get("cooldown", 0)),
            mode=lr_mode,
            verbose=1,
        )
    )
    if bool(early.get("enabled", True)):
        callbacks.append(
            tf.keras.callbacks.EarlyStopping(
                monitor=early_monitor,
                min_delta=float(early.get("min_delta", 0.0)),
                patience=int(early.get("patience", 10)),
                mode=early_mode,
                restore_best_weights=bool(early.get("restore_best_weights", True)),
                verbose=1,
            )
        )
    return callbacks


def _verify_best_checkpoint_selection(
    best_path: Path,
    history_payload: Mapping[str, Any],
    training: Mapping[str, Any],
    resumed: bool,
    validation: Optional[Mapping[str, Any]] = None,
) -> Dict[str, Any]:
    """Prove that the persisted best state uses the configured direction."""

    state = _read_state_json(best_path)
    if not state:
        raise FileNotFoundError("Best checkpoint state was not created for {}".format(best_path))
    checkpoint = training.get("checkpoint", {})
    early = training.get("early_stopping", {})
    lr_schedule = training.get("learning_rate_schedule", {})
    monitor = str(
        checkpoint.get("monitor")
        or early.get("monitor")
        or lr_schedule.get("monitor")
        or "val_total_loss"
    )
    mode = _monitor_mode(monitor, checkpoint.get("mode"))
    if state.get("monitor") != monitor or state.get("mode") != mode:
        raise RuntimeError(
            "Best checkpoint selection state conflicts with config: state={}/{} config={}/{}".format(
                state.get("monitor"), state.get("mode"), monitor, mode
            )
        )
    result = {
        "monitor": monitor,
        "mode": mode,
        "completed_epoch": state.get("completed_epoch"),
        "value": state.get("value"),
        "verified_against_current_history": not resumed,
    }
    if resumed:
        return result
    values = list((history_payload.get("history") or {}).get(monitor) or [])
    epochs = list(history_payload.get("epochs") or [])
    monitor_epochs = epochs
    if monitor.startswith("val_"):
        validation_frequency = int((validation or {}).get("every_epochs", 1))
        if validation_frequency <= 0:
            raise ValueError("validation.every_epochs must be positive")
        monitor_epochs = [
            epoch for epoch in epochs if int(epoch) % validation_frequency == 0
        ]
    if not values or len(values) != len(monitor_epochs):
        raise RuntimeError(
            "Checkpoint monitor {!r} is missing or misaligned in training history".format(monitor)
        )
    numeric = [float(value) for value in values]
    expected_value = min(numeric) if mode == "min" else max(numeric)
    expected_index = numeric.index(expected_value)
    expected_epoch = int(monitor_epochs[expected_index])
    if not math.isclose(float(state.get("value")), expected_value, rel_tol=1.0e-6, abs_tol=1.0e-8):
        raise RuntimeError(
            "Best checkpoint value {} does not equal history {} {}".format(
                state.get("value"), mode, expected_value
            )
        )
    if int(state.get("completed_epoch")) != expected_epoch:
        raise RuntimeError(
            "Best checkpoint epoch {} does not equal history best epoch {}".format(
                state.get("completed_epoch"), expected_epoch
            )
        )
    result["history_best_epoch"] = expected_epoch
    result["history_best_value"] = expected_value
    return result


def _load_starting_state(
    tf: Any,
    trainer: Any,
    config: Mapping[str, Any],
    training_checkpoint: Any,
    epoch_variable: Any,
) -> Dict[str, Any]:
    training = _require_mapping(config, "training")
    initial_value = training.get("initial_checkpoint")
    resume_value = training.get("resume_checkpoint")
    if initial_value and resume_value:
        raise ValueError("training.initial_checkpoint and training.resume_checkpoint are mutually exclusive")

    result: Dict[str, Any] = {
        "mode": "scratch",
        "path": None,
        "sha256": None,
        "initial_epoch": 0,
        "optimizer_state_restored": False,
    }
    selected = resume_value or initial_value
    if not selected:
        if str(config.get("stage", "")).lower() == "finetune":
            raise ValueError("Finetune stage requires training.initial_checkpoint or resume_checkpoint")
        return result

    path = resolve_path(str(selected), config)
    if not path.is_file():
        raise FileNotFoundError("Starting checkpoint not found: {}".format(path))
    trainer.backbone.load_weights(str(path))
    result.update(
        {
            "mode": "resume" if resume_value else "initial_weights",
            "path": str(path),
            "sha256": sha256_file(path),
        }
    )

    if resume_value:
        latest = tf.train.latest_checkpoint(str(_state_directory(path)))
        if latest:
            status = training_checkpoint.restore(latest)
            status.expect_partial()
            result["initial_epoch"] = int(epoch_variable.numpy())
            result["optimizer_state_restored"] = True
            result["training_state_checkpoint"] = str(latest)
        else:
            state = _read_state_json(path)
            result["initial_epoch"] = int(state.get("completed_epoch", 0) or 0)
            result["resume_warning"] = (
                "No TensorFlow optimizer state sidecar was found; model weights and recorded epoch were resumed"
            )
    return result


def _write_model_summary(model: Any, path: Path) -> None:
    lines: List[str] = []
    model.summary(print_fn=lines.append)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _guard_training_outputs(
    run_dir: Path,
    artifact_paths: Sequence[Path],
    resume_requested: bool,
    overwrite: bool,
) -> None:
    """Prevent an accidental fresh run from replacing an existing experiment."""

    if resume_requested or overwrite:
        return
    existing: List[str] = []
    if run_dir.exists():
        if not run_dir.is_dir():
            existing.append(str(run_dir))
        else:
            entries = sorted(run_dir.iterdir(), key=lambda path: str(path))
            if entries:
                existing.extend(str(path) for path in entries[:20])
    for path in artifact_paths:
        if path.exists() and str(path) not in existing:
            existing.append(str(path))
    if existing:
        raise FileExistsError(
            "Training output already exists; use resume_checkpoint, set outputs.overwrite=true, "
            "or choose a new outputs.run_dir: {}".format(existing)
        )


def train_from_config(config: Mapping[str, Any]) -> Dict[str, Any]:
    """Run pretraining or finetuning from one resolved YAML configuration."""

    config = copy.deepcopy(dict(config))
    if config.get("schema_version", 1) != 1:
        raise ValueError("Unsupported configuration schema_version")
    if str(config.get("task", "train")).lower() != "train":
        raise ValueError("Training entry point requires task: train")
    stage = str(config.get("stage", "")).strip().lower()
    if stage not in {"pretrain", "finetune"}:
        raise ValueError("Training stage must be pretrain or finetune; got {!r}".format(stage))
    model_checkpoint_stage = validate_model_checkpoint_stage(config)
    if model_checkpoint_stage != stage:
        raise ValueError(
            "Training stage and model.checkpoint_stage conflict: {} != {}".format(
                stage, model_checkpoint_stage
            )
        )
    if "dataset" in config:
        raise ValueError("config.dataset is obsolete; use config.data")
    selected_data = config.get("data", {})
    configured_stage = selected_data.get("require_training_stage") if isinstance(selected_data, Mapping) else None
    if configured_stage not in (None, "") and str(configured_stage).lower() != stage:
        raise ValueError(
            "stage and data.require_training_stage conflict: {} != {}".format(
                stage, configured_stage
            )
        )

    _prepare_environment(config)
    try:
        import numpy as np
        import tensorflow as tf
    except ImportError as exc:
        raise RuntimeError(
            "TensorFlow and NumPy are required; create the documented TensorFlow 2.9 conda environment first"
        ) from exc

    from models.hand_landmarker.registry import build_model

    try:
        from .data import create_sequences
    except ImportError as exc:
        raise RuntimeError("hand_landmarker.data.create_sequences is required for training") from exc

    runtime_report = _configure_tensorflow(tf, config)
    experiment = _require_mapping(config, "experiment")
    seed = int(experiment.get("seed", 20260713))
    _set_random_seeds(tf, np, seed)

    training = _require_mapping(config, "training")
    model_config = _require_mapping(config, "model")
    outputs = _require_mapping(config, "outputs")
    configured_input_shape = tuple(int(value) for value in model_config.get("input_shape", []))
    if configured_input_shape != (1, 256, 256):
        raise ValueError("model.input_shape must be [1, 256, 256]")
    if str(model_config.get("input_layout", "NCHW")) != "NCHW":
        raise ValueError("model.input_layout must be NCHW")
    configured_order = tuple(model_config.get("output_order", []))
    if configured_order != _SEMANTIC_OUTPUTS:
        raise ValueError("model.output_order must be {}".format(list(_SEMANTIC_OUTPUTS)))
    configured_sizes = dict(model_config.get("output_sizes", {}))
    if configured_sizes != {"landmarks": 42, "hand_flag": 1, "handedness": 1}:
        raise ValueError(
            "model.output_sizes must be landmarks=42, hand_flag=1, handedness=1"
        )

    if bool(training.get("mixed_precision", False)):
        tf.keras.mixed_precision.set_global_policy("mixed_float16")
    else:
        tf.keras.mixed_precision.set_global_policy("float32")

    backbone = build_model(
        str(model_config.get("version", "v2")),
        num_iterations=model_config.get("num_iterations", 8),
    )
    interface_report = assert_model_interface(backbone)
    loss_config = _validate_loss_config(config)
    trainer = _build_weighted_trainer(tf, backbone, loss_config)
    optimizer = _build_optimizer(tf, config)
    trainer.compile(
        optimizer=optimizer,
        run_eagerly=bool(training.get("run_eagerly", False)),
    )

    train_seq, val_seq, data_report = create_sequences(config)
    if train_seq is None:
        raise ValueError("create_sequences returned no training sequence")
    validation_enabled = bool(config.get("validation", {}).get("enabled", True))
    if validation_enabled and val_seq is None:
        raise ValueError("validation.enabled=true, but create_sequences returned no validation sequence")
    if not validation_enabled:
        val_seq = None

    run_dir = resolve_path(str(outputs.get("run_dir")), config)
    checkpoints_dir = resolve_path(str(outputs.get("checkpoints_dir", run_dir / "checkpoints")), config)
    logs_dir = resolve_path(str(outputs.get("logs_dir", run_dir / "logs")), config)
    best_path = resolve_path(
        str(outputs.get("best_checkpoint", checkpoints_dir / "best.weights.h5")), config
    )
    last_path = resolve_path(
        str(outputs.get("last_checkpoint", checkpoints_dir / "last.weights.h5")), config
    )
    final_path = resolve_path(
        str(outputs.get("final_weights", checkpoints_dir / "final.weights.h5")), config
    )
    if best_path == last_path:
        raise ValueError("outputs.best_checkpoint and outputs.last_checkpoint must differ")
    _guard_training_outputs(
        run_dir,
        (best_path, last_path, final_path),
        resume_requested=bool(training.get("resume_checkpoint")),
        overwrite=bool(outputs.get("overwrite", False)),
    )
    run_dir.mkdir(parents=True, exist_ok=True)
    checkpoints_dir.mkdir(parents=True, exist_ok=True)
    logs_dir.mkdir(parents=True, exist_ok=True)

    epoch_variable = tf.Variable(0, trainable=False, dtype=tf.int64, name="completed_epoch")
    training_checkpoint = tf.train.Checkpoint(
        model=trainer.backbone,
        optimizer=trainer.optimizer,
        epoch=epoch_variable,
    )
    starting_state = _load_starting_state(
        tf, trainer, config, training_checkpoint, epoch_variable
    )
    initial_epoch = int(starting_state["initial_epoch"])
    epochs = int(training.get("epochs", 1))
    if epochs <= 0:
        raise ValueError("training.epochs must be positive")
    if initial_epoch >= epochs:
        raise ValueError(
            "Resume epoch {} is not below configured training.epochs {}".format(initial_epoch, epochs)
        )

    sequence_set_epoch = getattr(train_seq, "set_epoch", None)
    sequence_epoch_synchronized = callable(sequence_set_epoch)
    if sequence_epoch_synchronized:
        sequence_set_epoch(initial_epoch)
    elif initial_epoch > 0:
        raise RuntimeError(
            "Resuming at epoch {} requires the training Sequence to implement set_epoch(epoch)".format(
                initial_epoch
            )
        )

    callbacks = _build_callbacks(
        tf,
        trainer,
        config,
        best_path,
        last_path,
        logs_dir,
        training_checkpoint,
        epoch_variable,
        resume_enabled=starting_state["mode"] == "resume",
    )

    config_path_value = config.get("_meta", {}).get("config_path")
    config_path = Path(str(config_path_value)).resolve() if config_path_value else None
    repo_root = Path(str(config.get("_meta", {}).get("repo_root", Path.cwd()))).resolve()
    data = config.get("data", {})
    validation = config.get("validation", {})
    train_labels = data.get("labels") if isinstance(data, Mapping) else None
    label_hashes = {
        "train": _hash_records(train_labels, config),
        "validation": _hash_records(validation.get("labels"), config),
    }
    metadata_path = run_dir / "experiment_metadata.json"
    metadata: Dict[str, Any] = {
        "status": "running",
        "started_at_utc": _utc_now(),
        "experiment": _jsonable(experiment),
        "stage": stage,
        "seed": seed,
        "model_version": str(model_config.get("version", "v2")),
        "model_interface": interface_report,
        "losses": _jsonable(loss_config),
        "starting_state": _jsonable(starting_state),
        "data_sequence_epoch": {
            "epoch": initial_epoch,
            "synchronized": sequence_epoch_synchronized,
            "keras_batch_shuffle": False,
        },
        "data_report": _jsonable(data_report),
        "label_hashes": label_hashes,
        "config_path": str(config_path) if config_path else None,
        "config_sha256": sha256_file(config_path) if config_path and config_path.is_file() else None,
        "runtime": runtime_report,
        "outputs_overwrite_authorized": bool(outputs.get("overwrite", False)),
        "python": {
            "version": sys.version,
            "executable": sys.executable,
            "platform": platform.platform(),
            "numpy_version": str(np.__version__),
        },
        "git": _git_metadata(repo_root),
        "artifacts": {
            "run_dir": str(run_dir),
            "best_checkpoint": str(best_path),
            "last_checkpoint": str(last_path),
            "final_weights": str(final_path),
            "logs_dir": str(logs_dir),
        },
        "resolved_config": _jsonable(config),
    }
    write_json(metadata_path, metadata)
    _write_model_summary(backbone, run_dir / "model_summary.txt")

    fit_kwargs: MutableMapping[str, Any] = {
        "x": train_seq,
        "validation_data": val_seq,
        "epochs": epochs,
        "initial_epoch": initial_epoch,
        "callbacks": callbacks,
        "verbose": int(training.get("verbose", 1)),
        # CanonicalSequence already performs deterministic, epoch-indexed
        # stratified sampling.  Keras' independent Sequence batch shuffle
        # would make an epoch-boundary resume depend on process RNG history.
        "shuffle": False,
    }
    steps_per_epoch = training.get("steps_per_epoch")
    if steps_per_epoch is not None:
        fit_kwargs["steps_per_epoch"] = int(steps_per_epoch)
    if val_seq is not None:
        fit_kwargs["validation_freq"] = int(validation.get("every_epochs", 1))
        validation_steps = validation.get("steps")
        if validation_steps is not None:
            fit_kwargs["validation_steps"] = int(validation_steps)

    try:
        history = trainer.fit(**fit_kwargs)
    except BaseException as exc:
        metadata["status"] = "failed"
        metadata["finished_at_utc"] = _utc_now()
        metadata["error"] = {"type": type(exc).__name__, "message": str(exc)}
        write_json(metadata_path, metadata)
        raise

    history_payload = {
        "epochs": [int(epoch) + 1 for epoch in history.epoch],
        "initial_epoch": initial_epoch,
        "history": _jsonable(history.history),
    }
    history_path = resolve_path(
        str(outputs.get("history_path", run_dir / "history.json")), config
    )
    write_json(history_path, history_payload)
    checkpoint_selection = _verify_best_checkpoint_selection(
        best_path,
        history_payload,
        training,
        resumed=starting_state["mode"] == "resume",
        validation=validation,
    )
    # ``final`` is an explicit, global-best convenience artifact.  This also
    # prevents a resumed fit from publishing only the best weights from its
    # post-resume callback window.
    backbone.load_weights(str(best_path))
    _atomic_save_weights(backbone, final_path)
    checkpoint_selection["final_weights_source"] = "best_checkpoint"

    artifacts: Dict[str, Any] = {}
    for name, path in (
        ("best_checkpoint", best_path),
        ("last_checkpoint", last_path),
        ("final_weights", final_path),
        ("history", history_path),
    ):
        if not path.is_file():
            raise FileNotFoundError("Expected training artifact was not created: {}".format(path))
        artifacts[name] = {
            "path": str(path),
            "size_bytes": int(path.stat().st_size),
            "sha256": sha256_file(path),
        }

    metadata["status"] = "complete"
    metadata["finished_at_utc"] = _utc_now()
    metadata["completed_epochs"] = history_payload["epochs"]
    metadata["checkpoint_selection"] = checkpoint_selection
    metadata["artifacts"] = artifacts
    write_json(metadata_path, metadata)

    report = {
        "status": "complete",
        "stage": stage,
        "experiment": str(experiment.get("name", "hand_landmarker")),
        "model_version": str(model_config.get("version", "v2")),
        "initial_epoch": initial_epoch,
        "completed_epochs": history_payload["epochs"],
        "artifacts": artifacts,
        "checkpoint_selection": checkpoint_selection,
        "metadata_path": str(metadata_path),
        "data_report": _jsonable(data_report),
        "label_hashes": label_hashes,
    }
    report_path = resolve_path(
        str(outputs.get("report_path", run_dir / "training_report.json")), config
    )
    write_json(report_path, report)
    report["report_path"] = str(report_path)
    return report


__all__ = ["assert_model_interface", "train_from_config"]
