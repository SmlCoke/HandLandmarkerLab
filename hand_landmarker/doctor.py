"""Read-only environment and configuration diagnostics."""

from __future__ import annotations

import importlib
import platform
import re
import sys
from importlib import metadata
from pathlib import Path
from typing import Any, Dict, Mapping, Optional, Sequence, Tuple

from .config import resolve_path


PACKAGES = {
    "numpy": "numpy",
    "keras": "keras",
    "h5py": "h5py",
    "protobuf": "google.protobuf",
    "flatbuffers": "flatbuffers",
    "Pillow": "PIL",
    "opencv-python-headless": "cv2",
    "PyYAML": "yaml",
    "tqdm": "tqdm",
    "tensorflow": "tensorflow",
    "tf2onnx": "tf2onnx",
    "onnx": "onnx",
    "onnxruntime": "onnxruntime",
}

UNKNOWN = "unknown"


def _json_safe(value: Any) -> Any:
    if value is None or isinstance(value, (bool, int, float, str)):
        return value
    if isinstance(value, Mapping):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    return str(value)


def _version_parts(value: Any) -> Tuple[int, ...]:
    match = re.search(r"\d+(?:\.\d+)*", str(value))
    if not match:
        return ()
    return tuple(int(part) for part in match.group(0).split("."))


def _matches_tensorflow_version(actual: str, expected: str) -> bool:
    specification = str(expected).strip()
    if specification.startswith("=="):
        specification = specification[2:].strip()
    if specification.endswith((".x", ".*")):
        prefix = specification[:-2]
        return actual == prefix or actual.startswith(prefix + ".")
    return actual.split("+", 1)[0] == specification


def _matches_version_prefix(actual: Any, expected: Any) -> bool:
    actual_parts = _version_parts(actual)
    expected_parts = _version_parts(expected)
    return bool(expected_parts) and actual_parts[: len(expected_parts)] == expected_parts


def _build_version(raw: Any) -> Optional[str]:
    if raw is None:
        return None
    if isinstance(raw, (list, tuple)):
        value = ".".join(str(part) for part in raw)
    else:
        value = str(raw).strip()
    return value or None


def _backend_version_check(
    name: str,
    expected: Any,
    build_version: Optional[str],
    failures: list,
    warnings: list,
) -> Dict[str, Any]:
    """Compare build metadata without presenting it as a runtime version check."""

    result: Dict[str, Any] = {
        "expected": str(expected) if expected is not None else None,
        "build_version": build_version or UNKNOWN,
        "build_status": "not_requested" if expected is None else UNKNOWN,
        # TensorFlow does not expose a dependable cross-version query for the
        # exact CUDA/cuDNN shared-library version loaded by this process.
        "runtime_version": UNKNOWN,
        "runtime_status": "not_requested" if expected is None else UNKNOWN,
    }
    if expected is None:
        return result

    if build_version is None:
        warnings.append(
            "{} build version is unknown; TensorFlow build info did not expose it".format(name)
        )
    elif _matches_version_prefix(build_version, expected):
        result["build_status"] = "match"
    else:
        result["build_status"] = "mismatch"
        failures.append(
            "TensorFlow {} build version mismatch: expected {}, found {}".format(
                name, expected, build_version
            )
        )

    warnings.append(
        "{} runtime version cannot be determined reliably by this diagnostic; "
        "runtime_version is unknown (build metadata is not runtime proof)".format(name)
    )
    return result


def environment_report(config: Optional[Mapping[str, Any]] = None) -> Dict[str, Any]:
    packages: Dict[str, Any] = {}
    loaded_modules: Dict[str, Any] = {}
    failures = []
    warnings = []
    for distribution, module_name in PACKAGES.items():
        try:
            module = importlib.import_module(module_name)
            loaded_modules[distribution] = module
            try:
                version = metadata.version(distribution)
            except metadata.PackageNotFoundError:
                version = getattr(module, "__version__", UNKNOWN)
            packages[distribution] = {"installed": True, "version": str(version)}
        except Exception as exc:
            packages[distribution] = {"installed": False, "error": str(exc)}
            failures.append("missing/import-failed package: {}".format(distribution))

    configured_environment = config.get("environment", {}) if config is not None else {}
    if not isinstance(configured_environment, Mapping):
        failures.append("config.environment must be a mapping")
        configured_environment = {}

    expected_tensorflow = configured_environment.get("tensorflow", "2.9.x")
    expected_cuda = configured_environment.get("cuda")
    expected_cudnn = configured_environment.get(
        "cudnn", configured_environment.get("cudnn_major")
    )
    require_gpu = bool(configured_environment.get("require_gpu", False))
    expectations = {
        "tensorflow": str(expected_tensorflow),
        "cuda": str(expected_cuda) if expected_cuda is not None else None,
        "cudnn": str(expected_cudnn) if expected_cudnn is not None else None,
        "require_gpu": require_gpu,
    }

    tensorflow_info: Dict[str, Any] = {
        "version": UNKNOWN,
        "built_with_cuda": UNKNOWN,
        "physical_gpus": [],
        "build_info": {},
        "compatibility": {},
    }
    tensorflow_module = loaded_modules.get("tensorflow")
    if tensorflow_module is not None:
        tf = tensorflow_module
        tensorflow_version = str(getattr(tf, "__version__", UNKNOWN))

        try:
            built_with_cuda: Any = bool(tf.test.is_built_with_cuda())
        except Exception as exc:
            built_with_cuda = UNKNOWN
            warnings.append("could not query whether TensorFlow was built with CUDA: {}".format(exc))

        try:
            gpu_devices: Sequence[Any] = tf.config.list_physical_devices("GPU")
            physical_gpus = [getattr(device, "name", str(device)) for device in gpu_devices]
            gpu_query_failed = False
        except Exception as exc:
            physical_gpus = []
            gpu_query_failed = True
            warnings.append("could not query TensorFlow physical GPUs: {}".format(exc))

        try:
            raw_build_info = tf.sysconfig.get_build_info()
            if isinstance(raw_build_info, Mapping):
                build_info = dict(raw_build_info)
            else:
                build_info = {}
                warnings.append("TensorFlow build info was not a mapping")
        except Exception as exc:
            build_info = {}
            warnings.append("could not query TensorFlow build info: {}".format(exc))

        cuda_build_version = _build_version(build_info.get("cuda_version"))
        cudnn_build_version = _build_version(build_info.get("cudnn_version"))
        tensorflow_status = (
            "match"
            if _matches_tensorflow_version(tensorflow_version, str(expected_tensorflow))
            else "mismatch"
        )
        if tensorflow_status == "mismatch":
            failures.append(
                "TensorFlow version mismatch: expected {}, found {}".format(
                    expected_tensorflow, tensorflow_version
                )
            )

        tensorflow_info = {
            "version": tensorflow_version,
            "built_with_cuda": built_with_cuda,
            "physical_gpus": physical_gpus,
            "build_info": _json_safe(build_info),
            "compatibility": {
                "tensorflow": {
                    "expected": str(expected_tensorflow),
                    "actual": tensorflow_version,
                    "status": tensorflow_status,
                },
                "cuda": _backend_version_check(
                    "CUDA", expected_cuda, cuda_build_version, failures, warnings
                ),
                "cudnn": _backend_version_check(
                    "cuDNN", expected_cudnn, cudnn_build_version, failures, warnings
                ),
            },
        }

        if require_gpu:
            if built_with_cuda is not True:
                if built_with_cuda == UNKNOWN:
                    failures.append(
                        "environment.require_gpu=true, but TensorFlow CUDA build support is unknown"
                    )
                else:
                    failures.append(
                        "environment.require_gpu=true, but TensorFlow is not built with CUDA support"
                    )
            if gpu_query_failed:
                failures.append(
                    "environment.require_gpu=true, but TensorFlow physical GPU discovery failed"
                )
            elif not physical_gpus:
                failures.append(
                    "environment.require_gpu=true, but TensorFlow found no physical GPU"
                )
    else:
        tensorflow_info["compatibility"] = {
            "tensorflow": {
                "expected": str(expected_tensorflow),
                "actual": UNKNOWN,
                "status": UNKNOWN,
            },
            "cuda": _backend_version_check(
                "CUDA", expected_cuda, None, failures, warnings
            ),
            "cudnn": _backend_version_check(
                "cuDNN", expected_cudnn, None, failures, warnings
            ),
        }
        if require_gpu:
            failures.append(
                "environment.require_gpu=true, but TensorFlow could not be imported"
            )

    path_checks = []
    if config is not None:
        path_specs = [
            ("hand.model_path", config.get("hand", {}).get("model_path")),
            ("palm.model_path", config.get("palm", {}).get("model_path")),
        ]
        for label, value in path_specs:
            if value:
                path = resolve_path(str(value), config)
                exists = path.is_file()
                path_checks.append({"key": label, "path": str(path), "exists": exists})
                if not exists:
                    failures.append("configured file not found: {}".format(path))
        data_config = config.get("data", {})
        for key in ("labels",):
            if data_config.get(key):
                path = resolve_path(str(data_config[key]), config)
                exists = path.is_file()
                path_checks.append({"key": "data.{}".format(key), "path": str(path), "exists": exists})
                if not exists:
                    failures.append("configured label file not found: {}".format(path))
        validation_labels = config.get("validation", {}).get("labels")
        if validation_labels:
            path = resolve_path(str(validation_labels), config)
            exists = path.is_file()
            path_checks.append(
                {"key": "validation.labels", "path": str(path), "exists": exists}
            )
            if not exists:
                failures.append("configured label file not found: {}".format(path))

    if sys.version_info[:2] != (3, 8):
        failures.append("Production environment requires Python 3.8; found {}".format(platform.python_version()))
    return {
        "ok": not failures,
        "python": {
            "version": platform.python_version(),
            "executable": sys.executable,
            "platform": platform.platform(),
        },
        "expectations": expectations,
        "packages": packages,
        "tensorflow": tensorflow_info,
        "path_checks": path_checks,
        "warnings": warnings,
        "failures": failures,
    }
