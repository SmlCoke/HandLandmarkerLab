"""Export the fixed Hand Landmarker interface to ONNX and verify parity."""

from __future__ import annotations

import math
import os
from pathlib import Path
from typing import Any, Dict, List, Mapping, Sequence

from .config import load_config, resolve_path
from .contracts import (
    BOARD_CONTRACT,
    MODEL_IO,
    validate_checkpoint_path_stage,
    validate_model_checkpoint_stage,
)
from .conversion_datasets import (
    generate_conversion_datasets,
    guard_conversion_dataset_output,
)
from .io_utils import sha256_file, write_json


def _shape_from_onnx(value_info) -> List[Any]:
    shape = []
    for dimension in value_info.type.tensor_type.shape.dim:
        if dimension.HasField("dim_value"):
            shape.append(int(dimension.dim_value))
        elif dimension.HasField("dim_param"):
            shape.append(str(dimension.dim_param))
        else:
            shape.append(None)
    return shape


def _assert_model_interface(model) -> None:
    input_shape = tuple(model.input_shape)
    if input_shape[1:] != (1, 256, 256):
        raise ValueError("Model input interface changed: {}".format(input_shape))
    output_shapes = [tuple(shape) for shape in model.output_shape]
    expected = [(None, 1, 1, 42), (None, 1, 1, 1), (None, 1, 1, 1)]
    if output_shapes != expected:
        raise ValueError("Model output interface changed: got {}, expected {}".format(output_shapes, expected))
    expected_names = ["convld_21_2d", "activation_handflag", "activation_handedness"]
    if list(model.output_names) != expected_names:
        raise ValueError(
            "Model output order/names changed: got {}, expected {}".format(
                list(model.output_names), expected_names
            )
        )


def _numeric_parity(
    model,
    session,
    input_name: str,
    absolute_tolerance: float,
    relative_tolerance: float,
    random_samples: int,
) -> Dict[str, Any]:
    import numpy as np

    random = np.random.RandomState(20260713)
    cases = {
        "zeros": np.zeros((1, 1, 256, 256), dtype=np.float32),
        "ones": np.ones((1, 1, 256, 256), dtype=np.float32),
    }
    for index in range(max(0, int(random_samples))):
        cases["random_{}".format(index)] = random.uniform(
            0.0, 1.0, size=(1, 1, 256, 256)
        ).astype(np.float32)
    report: Dict[str, Any] = {
        "absolute_tolerance": float(absolute_tolerance),
        "relative_tolerance": float(relative_tolerance),
        "random_samples": int(random_samples),
        "cases": {},
    }
    collected_onnx_outputs: List[List[Any]] = []
    for name, tensor in cases.items():
        keras_values = model(tensor, training=False)
        if not isinstance(keras_values, (list, tuple)):
            keras_values = [keras_values]
        keras_arrays = [np.asarray(value) for value in keras_values]
        onnx_arrays = [np.asarray(value) for value in session.run(None, {input_name: tensor})]
        if not collected_onnx_outputs:
            collected_onnx_outputs = [[] for _value in onnx_arrays]
        for output_index, value in enumerate(onnx_arrays):
            collected_onnx_outputs[output_index].append(value.reshape(-1))
        if len(keras_arrays) != len(onnx_arrays):
            raise ValueError("ONNX output count differs from Keras")
        maximums = []
        relative_maximums = []
        keras_ranges = []
        onnx_ranges = []
        for output_index, (keras_array, onnx_array) in enumerate(zip(keras_arrays, onnx_arrays)):
            if keras_array.size != onnx_array.size:
                raise ValueError(
                    "ONNX output element count differs from Keras: {} != {}".format(
                        onnx_array.size, keras_array.size
                    )
                )
            expected = keras_array.astype(np.float64).reshape(-1)
            actual = onnx_array.astype(np.float64).reshape(-1)
            if not np.all(np.isfinite(expected)):
                raise ValueError(
                    "Keras export reference produced NaN/Inf for {} output {}".format(
                        name, output_index
                    )
                )
            if not np.all(np.isfinite(actual)):
                raise ValueError(
                    "ONNX export produced NaN/Inf for {} output {}".format(
                        name, output_index
                    )
                )
            keras_ranges.append(
                {
                    "minimum": float(np.min(expected)),
                    "maximum": float(np.max(expected)),
                    "max_abs": float(np.max(np.abs(expected))),
                }
            )
            onnx_ranges.append(
                {
                    "minimum": float(np.min(actual)),
                    "maximum": float(np.max(actual)),
                    "max_abs": float(np.max(np.abs(actual))),
                }
            )
            absolute = np.abs(expected - actual)
            relative = absolute / np.maximum(np.abs(expected), np.finfo(np.float64).eps)
            maximums.append(float(np.max(absolute)))
            relative_maximums.append(float(np.max(relative)))
            if not np.allclose(
                expected,
                actual,
                atol=float(absolute_tolerance),
                rtol=float(relative_tolerance),
            ):
                raise ValueError(
                    "ONNX parity failed for {} output {}: max_abs_error={}, max_rel_error={}".format(
                        name, len(maximums) - 1, maximums[-1], relative_maximums[-1]
                    )
                )
        report["cases"][name] = {
            "max_abs_error_by_output": maximums,
            "max_rel_error_by_output": relative_maximums,
            "max_abs_error": max(maximums),
            "max_rel_error": max(relative_maximums),
            "keras_output_ranges": keras_ranges,
            "onnx_output_ranges": onnx_ranges,
        }
    report["aggregate_onnx_output_ranges"] = []
    for values in collected_onnx_outputs:
        combined = np.concatenate(values).astype(np.float64)
        minimum = float(np.min(combined))
        maximum = float(np.max(combined))
        report["aggregate_onnx_output_ranges"].append(
            {
                "minimum": minimum,
                "maximum": maximum,
                "dynamic_range": maximum - minimum,
                "standard_deviation": float(np.std(combined)),
            }
        )
    return report


def _reparameterization_parity(
    training_model,
    deploy_model,
    absolute_tolerance: float,
    relative_tolerance: float,
) -> Dict[str, Any]:
    """Verify that branch fusion preserves all three Keras outputs."""

    import numpy as np

    random = np.random.RandomState(20260714)
    cases = {
        "zeros": np.zeros((1, 1, 256, 256), dtype=np.float32),
        "random": random.uniform(0.0, 1.0, (1, 1, 256, 256)).astype(np.float32),
    }
    report: Dict[str, Any] = {
        "absolute_tolerance": float(absolute_tolerance),
        "relative_tolerance": float(relative_tolerance),
        "training_parameter_count": int(training_model.count_params()),
        "deploy_parameter_count": int(deploy_model.count_params()),
        "cases": {},
    }
    for name, tensor in cases.items():
        reference = training_model(tensor, training=False)
        actual = deploy_model(tensor, training=False)
        reference = reference if isinstance(reference, (list, tuple)) else [reference]
        actual = actual if isinstance(actual, (list, tuple)) else [actual]
        maximums = []
        for output_index, (expected_value, actual_value) in enumerate(zip(reference, actual)):
            expected = np.asarray(expected_value, dtype=np.float64)
            observed = np.asarray(actual_value, dtype=np.float64)
            maximums.append(float(np.max(np.abs(expected - observed))))
            if not np.allclose(
                expected,
                observed,
                atol=float(absolute_tolerance),
                rtol=float(relative_tolerance),
            ):
                raise ValueError(
                    "v2 branch fusion parity failed for {} output {}: max_abs_error={}".format(
                        name, output_index, maximums[-1]
                    )
                )
        report["cases"][name] = {"max_abs_error_by_output": maximums}
    return report


def _a1_attribute_audit(graph, onnx_module) -> Dict[str, Any]:
    """Validate the project-9 A1 operator attributes, not just op names."""

    initializers = {item.name: item for item in graph.graph.initializer}
    violations: List[Dict[str, Any]] = []
    checked = {"Conv": 0, "MaxPool": 0}

    def add(node, rule: str, value: Any) -> None:
        violations.append(
            {
                "node": str(node.name or "<unnamed>"),
                "op_type": str(node.op_type),
                "rule": rule,
                "value": value,
            }
        )

    for node in graph.graph.node:
        if node.op_type not in checked:
            continue
        checked[node.op_type] += 1
        attributes = {
            item.name: onnx_module.helper.get_attribute_value(item) for item in node.attribute
        }
        if node.op_type == "Conv":
            weights = initializers.get(node.input[1]) if len(node.input) > 1 else None
            if weights is None or len(weights.dims) < 3:
                add(node, "Conv weights must be a static initializer", None)
                continue
            kernel = list(attributes.get("kernel_shape") or list(weights.dims[-2:]))
            strides = list(attributes.get("strides") or [1] * len(kernel))
            pads = list(attributes.get("pads") or [0] * (2 * len(kernel)))
            for label, values in (("kernel", kernel), ("stride", strides), ("pad", pads)):
                if any(int(value) < 0 or int(value) > 16 for value in values):
                    add(node, "Conv {} values must not exceed 16".format(label), values)
            kernel_volume = int(math.prod(int(value) for value in list(weights.dims)[1:]))
            if kernel_volume > 2048:
                add(node, "Conv Kw*Kh*Cin per output kernel must not exceed 2048", kernel_volume)
        elif node.op_type == "MaxPool":
            kernel = list(attributes.get("kernel_shape") or [])
            if not kernel or any(int(value) <= 0 or int(value) > 8 for value in kernel):
                add(node, "MaxPool kernel dimensions must be in [1,8]", kernel)
    return {"checked_nodes": checked, "violations": violations}


def _quantization_readiness_audit(
    graph, onnx_module, maximum_depthwise_group: int = 128
) -> Dict[str, Any]:
    """Reject graph degeneracies known to break the A1 INT8 quantizer."""

    import numpy as np

    maximum_group = int(maximum_depthwise_group)
    if maximum_group < 1:
        raise ValueError("export.maximum_depthwise_group must be positive")
    initializers = {
        item.name: onnx_module.numpy_helper.to_array(item)
        for item in graph.graph.initializer
    }
    violations: List[Dict[str, Any]] = []
    checked_conv_count = 0
    grouped_conv_count = 0
    observed_maximum_group = 1
    for node in graph.graph.node:
        if node.op_type != "Conv":
            continue
        checked_conv_count += 1
        weights = initializers.get(node.input[1]) if len(node.input) > 1 else None
        if weights is None:
            violations.append(
                {
                    "node": str(node.name or "<unnamed>"),
                    "rule": "Conv weights must be a static initializer for quantization",
                    "value": None,
                }
            )
            continue
        if not np.all(np.isfinite(weights)):
            violations.append(
                {
                    "node": str(node.name or "<unnamed>"),
                    "rule": "Conv weights must be finite",
                    "value": None,
                }
            )
        elif not np.any(weights != 0.0):
            violations.append(
                {
                    "node": str(node.name or "<unnamed>"),
                    "rule": "Conv weight tensor must not be entirely zero",
                    "value": list(weights.shape),
                }
            )
        attributes = {
            item.name: onnx_module.helper.get_attribute_value(item)
            for item in node.attribute
        }
        group = int(attributes.get("group", 1))
        observed_maximum_group = max(observed_maximum_group, group)
        if group > 1:
            grouped_conv_count += 1
        if group > maximum_group:
            violations.append(
                {
                    "node": str(node.name or "<unnamed>"),
                    "rule": "Depthwise/grouped Conv group exceeds verified A1 envelope",
                    "value": group,
                }
            )
    return {
        "checked_conv_count": checked_conv_count,
        "grouped_conv_count": grouped_conv_count,
        "maximum_depthwise_group": maximum_group,
        "observed_maximum_group": observed_maximum_group,
        "violations": violations,
    }


def _guard_export_outputs(output_path: Path, contract_path: Path, overwrite: bool) -> None:
    if output_path == contract_path:
        raise ValueError("export.contract_path must differ from export.model_path")
    if overwrite:
        return
    existing = [str(path) for path in (output_path, contract_path) if path.exists()]
    if existing:
        raise FileExistsError(
            "Export output already exists; set export.overwrite=true or choose new paths: {}".format(
                existing
            )
        )


def _finetune_training_provenance(config: Mapping[str, Any]) -> Dict[str, Any]:
    """Authenticate the finetune inputs that produced an exported checkpoint."""

    conversion = ((config.get("export") or {}).get("conversion_datasets") or {})
    sets = conversion.get("sets") or {}
    calibration = sets.get("calibrate_datasets") or {}
    train_source = (calibration.get("sources") or {}).get("train") or {}
    config_value = train_source.get("config_path")
    if not config_value:
        raise ValueError(
            "Finetune export requires the train calibration source config_path"
        )
    train_config_path = resolve_path(str(config_value), config)
    if not train_config_path.is_file() or train_config_path.is_symlink():
        raise FileNotFoundError(
            "Finetune training config is missing or a symlink: {}".format(
                train_config_path
            )
        )
    train_config = load_config(train_config_path)
    if (
        str(train_config.get("task")) != "train"
        or str(train_config.get("stage")) != "finetune"
        or validate_model_checkpoint_stage(train_config) != "finetune"
    ):
        raise ValueError(
            "Finetune export calibration must use a finetune training config"
        )

    def authenticated_file(value: Any, label: str) -> Dict[str, Any]:
        if value in (None, ""):
            raise ValueError("Finetune training config requires {}".format(label))
        path = resolve_path(str(value), train_config)
        if not path.is_file() or path.is_symlink():
            raise FileNotFoundError(
                "Finetune {} is missing or a symlink: {}".format(label, path)
            )
        return {"path": str(path.resolve(strict=True)), "sha256": sha256_file(path)}

    return {
        "schema_version": "finetune_export_provenance_v1",
        "training_config": {
            "path": str(train_config_path.resolve(strict=True)),
            "sha256": sha256_file(train_config_path),
        },
        "curation_manifest": authenticated_file(
            (train_config.get("data") or {}).get("curation_manifest"),
            "curation manifest",
        ),
        "initial_multitask_checkpoint": authenticated_file(
            (train_config.get("training") or {}).get("initial_checkpoint"),
            "initial multitask checkpoint",
        ),
    }


def _validated_model_size(path: Path, maximum_model_size_mb: float):
    maximum = float(maximum_model_size_mb)
    if not math.isfinite(maximum) or maximum <= 0.0:
        raise ValueError("export.maximum_model_size_mb must be a positive finite value")
    size_bytes = int(path.stat().st_size)
    size_mb = size_bytes / float(1024 * 1024)
    if size_mb > maximum:
        raise ValueError(
            "Exported ONNX is {:.3f} MiB, exceeding the configured {:.3f} MiB limit".format(
                size_mb, maximum
            )
        )
    return size_bytes, size_mb


def export_from_config(config: Mapping[str, Any]) -> Dict[str, Any]:
    from .runtime import normalize_runtime_config

    config = normalize_runtime_config(config)
    if config.get("schema_version", 1) != 1:
        raise ValueError("Unsupported configuration schema_version")
    if str(config.get("task", "export_onnx")).lower() != "export_onnx":
        raise ValueError("ONNX export entry point requires task: export_onnx")
    runtime_device = str(config.get("runtime", {}).get("device", "cpu")).strip().lower()
    if runtime_device != "cpu":
        raise ValueError("ONNX export runtime.device must remain cpu")
    model_checkpoint_stage = validate_model_checkpoint_stage(config)
    preflight_export = config.get("export", {})
    preflight_weights = config.get("hand", {}).get("model_path")
    if preflight_weights:
        validate_checkpoint_path_stage(
            config,
            resolve_path(str(preflight_weights), config),
        )
    guard_conversion_dataset_output(
        config,
        overwrite=bool(preflight_export.get("overwrite", False)),
    )
    # Must be set before importing TensorFlow; this script is a CPU-side
    # serialization/parity job and must not reserve the training GPU.
    os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
    conversion_datasets = None
    try:
        import onnx
        import onnxruntime as ort
        import tensorflow as tf
        import tf2onnx
    except ImportError as exc:
        raise RuntimeError("TensorFlow, tf2onnx, onnx, and onnxruntime are required for export") from exc

    from models.hand_landmarker.registry import build_model, reparameterize_for_deploy

    export_config = config.get("export", {})
    validate_config = export_config.get("validate", {})
    if not isinstance(validate_config, Mapping):
        raise ValueError("export.validate must be a mapping")
    if not bool(validate_config.get("enabled", True)):
        raise ValueError("export.validate.enabled must remain true for a deployment export")
    if str(validate_config.get("backend", "onnxruntime_cpu")) != "onnxruntime_cpu":
        raise ValueError("export.validate.backend must be onnxruntime_cpu")
    if int(validate_config.get("random_samples", 4)) < 1:
        raise ValueError("export.validate.random_samples must be at least 1")
    if int(export_config.get("opset", 11)) != 11:
        raise ValueError("The verified A1 conversion contract requires ONNX opset 11")
    if bool(export_config.get("dynamic_batch", False)):
        raise ValueError("A1 Hand Landmarker export requires static batch 1")
    maximum_model_size_mb = float(export_config.get("maximum_model_size_mb", 15.0))
    if not math.isfinite(maximum_model_size_mb) or maximum_model_size_mb <= 0.0:
        raise ValueError("export.maximum_model_size_mb must be a positive finite value")
    input_name = str(export_config.get("input_name", MODEL_IO["input_name"]))
    if input_name != str(MODEL_IO["input_name"]):
        raise ValueError("export.input_name must remain {!r}".format(MODEL_IO["input_name"]))
    output_semantics = list(
        export_config.get("output_names", ["landmarks", "hand_flag", "handedness"])
    )
    if output_semantics != ["landmarks", "hand_flag", "handedness"]:
        raise ValueError("export.output_names must preserve landmarks, hand_flag, handedness order")
    if not bool(validate_config.get("require_output_order_match", True)):
        raise ValueError("export.validate.require_output_order_match must remain true")
    hand_config = config.get("hand", {})
    model_config = config.get("model", {})
    weights_value = hand_config.get("model_path")
    output_value = export_config.get("model_path")
    if not weights_value or not output_value:
        raise KeyError("Export config requires hand.model_path (weights) and export.model_path")
    weights_path = resolve_path(str(weights_value), config)
    output_path = resolve_path(str(output_value), config)
    validate_checkpoint_path_stage(config, weights_path)
    if not weights_path.is_file():
        raise FileNotFoundError("Weights not found: {}".format(weights_path))
    training_provenance = (
        _finetune_training_provenance(config)
        if model_checkpoint_stage == "finetune"
        else None
    )
    if output_path.suffix.lower() != ".onnx":
        raise ValueError("export.model_path must end in .onnx")
    contract_value = export_config.get("contract_path") or str(
        output_path.with_suffix(".contract.json")
    )
    contract_path = resolve_path(str(contract_value), config)
    _guard_export_outputs(
        output_path,
        contract_path,
        overwrite=bool(export_config.get("overwrite", False)),
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)

    model_version = str(model_config.get("version", "v2"))
    training_model = build_model(
        model_version,
        num_iterations=model_config.get("num_iterations", 8),
    )
    training_model.load_weights(str(weights_path))
    _assert_model_interface(training_model)
    model = reparameterize_for_deploy(
        training_model,
        version=model_version,
        num_iterations=model_config.get("num_iterations", 8),
    )
    _assert_model_interface(model)
    fusion_parity = _reparameterization_parity(
        training_model,
        model,
        absolute_tolerance=float(validate_config.get("fusion_absolute_tolerance", 5e-5)),
        relative_tolerance=float(validate_config.get("fusion_relative_tolerance", 5e-4)),
    )
    signature = (tf.TensorSpec([1, 1, 256, 256], tf.float32, name=input_name),)
    temporary = output_path.with_name(output_path.name + ".tmp")
    if temporary.exists():
        temporary.unlink()
    try:
        tf2onnx.convert.from_keras(
            model,
            input_signature=signature,
            opset=int(export_config.get("opset", 11)),
            output_path=str(temporary),
        )
        graph = onnx.load(str(temporary))
        onnx.checker.check_model(graph)
        model_size_bytes, model_size_mb = _validated_model_size(
            temporary, maximum_model_size_mb
        )
        session = ort.InferenceSession(str(temporary), providers=["CPUExecutionProvider"])
        inputs = [
            {"name": item.name, "shape": _shape_from_onnx(item), "type": item.type.tensor_type.elem_type}
            for item in graph.graph.input
        ]
        outputs = [
            {"name": item.name, "shape": _shape_from_onnx(item), "type": item.type.tensor_type.elem_type}
            for item in graph.graph.output
        ]
        if (
            len(inputs) != 1
            or inputs[0]["name"] != input_name
            or inputs[0]["shape"] != [1, 1, 256, 256]
        ):
            raise ValueError("Exported ONNX input interface is invalid: {}".format(inputs))
        float_type = int(onnx.TensorProto.FLOAT)
        if inputs[0]["type"] != float_type:
            raise ValueError("Exported ONNX input must be FLOAT32: {}".format(inputs))
        output_shapes = [item["shape"] for item in outputs]
        valid_landmark_shapes = ([1, 42, 1, 1], [1, 1, 1, 42])
        if (
            len(output_shapes) != 3
            or output_shapes[0] not in valid_landmark_shapes
            or output_shapes[1:] != [[1, 1, 1, 1], [1, 1, 1, 1]]
        ):
            raise ValueError("Exported ONNX output interface is invalid: {}".format(outputs))
        if any(item["type"] != float_type for item in outputs):
            raise ValueError("All three A1 Hand outputs must remain FLOAT32: {}".format(outputs))
        parity = _numeric_parity(
            model,
            session,
            session.get_inputs()[0].name,
            absolute_tolerance=float(validate_config.get("absolute_tolerance", 1e-5)),
            relative_tolerance=float(validate_config.get("relative_tolerance", 1e-4)),
            random_samples=int(validate_config.get("random_samples", 4)),
        )
        operators = sorted({node.op_type for node in graph.graph.node})
        allowed_operators = set(
            export_config.get("a1_allowed_operators")
            or ["Conv", "Add", "Relu", "MaxPool", "Sigmoid", "Identity", "Reshape"]
        )
        unsupported = sorted(set(operators) - allowed_operators)
        strict_a1 = bool(export_config.get("strict_a1_operators", True))
        force_a1_operator_export = bool(
            export_config.get("force_a1_operator_export", False)
        )
        enforce_a1_operator_contract = strict_a1 and not force_a1_operator_export
        if unsupported and enforce_a1_operator_contract:
            raise ValueError("ONNX contains operators outside the A1 deployment contract: {}".format(unsupported))
        attribute_audit = _a1_attribute_audit(graph, onnx)
        if attribute_audit["violations"] and enforce_a1_operator_contract:
            raise ValueError(
                "ONNX violates A1 operator attribute constraints: {}".format(
                    attribute_audit["violations"]
                )
            )
        quantization_audit = _quantization_readiness_audit(
            graph,
            onnx,
            maximum_depthwise_group=int(
                export_config.get("maximum_depthwise_group", 128)
            ),
        )
        if quantization_audit["violations"]:
            raise ValueError(
                "ONNX is not ready for A1 INT8 quantization: {}".format(
                    quantization_audit["violations"]
                )
            )
        minimum_output_range = float(
            export_config.get("minimum_quantization_output_range", 1.0e-6)
        )
        if not math.isfinite(minimum_output_range) or minimum_output_range <= 0.0:
            raise ValueError(
                "export.minimum_quantization_output_range must be positive and finite"
            )
        inactive_outputs = [
            index
            for index, value in enumerate(
                parity.get("aggregate_onnx_output_ranges", [])
            )
            if float(value.get("dynamic_range", 0.0)) < minimum_output_range
        ]
        if inactive_outputs:
            raise ValueError(
                "ONNX output head(s) {} are constant across quantization probes; "
                "refusing a degenerate INT8 preflight".format(inactive_outputs)
            )
        # Build conversion inputs only after the temporary ONNX has passed all
        # interface, operator, and parity checks.  A dataset failure therefore
        # cannot replace the final ONNX artifact.
        conversion_datasets = generate_conversion_datasets(config)
        os.replace(str(temporary), str(output_path))
    finally:
        if temporary.exists():
            temporary.unlink()

    report = {
        "model_checkpoint_stage": model_checkpoint_stage,
        "model_path": str(output_path),
        "model_sha256": sha256_file(output_path),
        "model_size_bytes": model_size_bytes,
        "model_size_mb": model_size_mb,
        "maximum_model_size_mb": maximum_model_size_mb,
        "weights_path": str(weights_path),
        "weights_sha256": sha256_file(weights_path),
        "training_provenance": training_provenance,
        "model_version": model_version,
        "opset": int(export_config.get("opset", 11)),
        "inputs": inputs,
        "outputs": outputs,
        "output_semantics_in_order": output_semantics,
        "operators": operators,
        "a1_operator_audit": {
            "allowed": sorted(allowed_operators),
            "unsupported": unsupported,
            "strict": strict_a1,
            "forced": force_a1_operator_export,
            "enforced": enforce_a1_operator_contract,
            "attributes": attribute_audit,
        },
        "quantization_readiness": {
            "graph": quantization_audit,
            "minimum_output_range": minimum_output_range,
            "aggregate_output_ranges": parity.get(
                "aggregate_onnx_output_ranges", []
            ),
        },
        "numeric_parity": parity,
        "reparameterization_parity": fusion_parity,
        "deployment_metadata": dict(export_config.get("metadata", {})),
        "normalization": {"dtype": "float32", "range": [0.0, 1.0], "board_source": "uint8 gray / 255"},
        "contract": MODEL_IO,
        "board_runtime_contract": BOARD_CONTRACT,
        "conversion_datasets": conversion_datasets,
    }
    write_json(contract_path, report)
    report["contract_path"] = str(contract_path)
    return report
