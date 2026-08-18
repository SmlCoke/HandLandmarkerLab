#!/usr/bin/env python3
"""Export an untrained selected Iris ONNX and real A1 conversion inputs."""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from hand_landmarker.config import load_config, resolve_path
from hand_landmarker.io_utils import write_json


def prepare_quantization_probe_weights(model, seed):
    """Replace training-stability zeros with deterministic non-degenerate probes.

    The trainable model intentionally starts residual tails and heads at zero.
    Exporting that exact initialization creates all-zero Conv tensors and a
    constant-output graph, which cannot exercise an INT8 quantizer.  These
    disposable weights are used only by the converter preflight.
    """

    import numpy as np
    from tensorflow.keras.layers import BatchNormalization

    random = np.random.RandomState(int(seed) + 101)
    activated_batch_norms = []
    seen = set()
    for batch_norm in model.submodules:
        if not isinstance(batch_norm, BatchNormalization) or id(batch_norm) in seen:
            continue
        seen.add(id(batch_norm))
        gamma, beta, moving_mean, moving_variance = batch_norm.get_weights()
        if np.all(gamma == 0.0):
            gamma = random.uniform(0.04, 0.06, gamma.shape).astype("float32")
            batch_norm.set_weights([gamma, beta, moving_mean, moving_variance])
            activated_batch_norms.append(batch_norm.name)

    activated_heads = []
    for name in ("convld_21_2d", "conv_handflag", "conv_handedness"):
        layer = model.get_layer(name)
        kernel, bias = layer.get_weights()
        kernel = random.normal(0.0, 0.002, kernel.shape).astype("float32")
        layer.set_weights([kernel, bias])
        activated_heads.append(name)

    if not activated_batch_norms or len(activated_heads) != 3:
        raise ValueError(
            "Preflight expected zero-initialized residual tails and all three heads"
        )
    return {
        "method": "deterministic_nonzero_quantization_probe_weights",
        "activated_batch_norm_count": len(activated_batch_norms),
        "activated_heads": activated_heads,
        "residual_gamma_range": [0.04, 0.06],
        "head_kernel_stddev": 0.002,
    }


def build_preflight_bundle(config):
    export_config = config.get("export", {})
    if export_config.get("preflight_untrained") is not True:
        raise ValueError("Preflight config must set export.preflight_untrained=true")
    if export_config.get("overwrite") is not True:
        raise ValueError("Disposable preflight artifacts require export.overwrite=true")

    weights_value = (config.get("hand") or {}).get("model_path")
    if not weights_value:
        raise KeyError("Preflight config requires hand.model_path")
    weights_path = resolve_path(str(weights_value), config)
    normalized_parts = [part.lower() for part in weights_path.parts]
    if "preflight" not in normalized_parts or "untrained" not in weights_path.name.lower():
        raise ValueError(
            "Refusing to write random weights outside a clearly named preflight/untrained path"
        )

    # This is a CPU serialization check and must not reserve a training GPU.
    os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
    try:
        import tensorflow as tf
    except ImportError as exc:
        raise RuntimeError("TensorFlow is required for the export preflight") from exc

    from hand_landmarker.export import _assert_model_interface, export_from_config
    from models.hand_landmarker.registry import build_model

    seed = int((config.get("experiment") or {}).get("seed", 20260714))
    tf.keras.backend.clear_session()
    tf.keras.utils.set_random_seed(seed)
    model_config = config.get("model", {})
    training_model = build_model(
        str(model_config.get("version", "v3-pro")),
        num_iterations=model_config.get("num_iterations"),
    )
    _assert_model_interface(training_model)
    quantization_probe = prepare_quantization_probe_weights(training_model, seed)
    weights_path.parent.mkdir(parents=True, exist_ok=True)
    training_model.save_weights(str(weights_path), overwrite=True)

    report = export_from_config(config)
    report["preflight"] = {
        "untrained": True,
        "accuracy_model": False,
        "purpose": "A1 converter graph/operator compatibility before training",
        "seed": seed,
        "quantization_probe": quantization_probe,
    }
    write_json(Path(str(report["contract_path"])), report)
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, help="Preflight export YAML file")
    args = parser.parse_args()
    print(
        json.dumps(
            build_preflight_bundle(load_config(args.config)),
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
