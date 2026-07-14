"""Lazy registry for versioned Hand Landmarker architectures.

TensorFlow is intentionally imported only after :func:`build_model` selects a
known version.  This keeps configuration inspection and command ``--help``
usable on machines where the documented training environment is not installed.
"""

from __future__ import annotations

from importlib import import_module
from typing import Any, Tuple


DEFAULT_VERSION = "v2"
_MODEL_MODULES = {"v2": "models.hand_landmarker.v2"}


def available_versions() -> Tuple[str, ...]:
    """Return model versions in deterministic order without importing TensorFlow."""

    return tuple(sorted(_MODEL_MODULES))


def build_model(version: str = DEFAULT_VERSION, **kwargs: Any):
    """Build a fixed-interface Hand Landmarker, rejecting unknown versions."""

    normalized = str(version).strip().lower()
    module_name = _MODEL_MODULES.get(normalized)
    if module_name is None:
        raise ValueError(
            "Unknown Hand Landmarker model version {!r}; available versions: {}".format(
                version, ", ".join(available_versions())
            )
        )

    requested_shape = tuple(kwargs.pop("input_size", (1, 256, 256)))
    if requested_shape != (1, 256, 256):
        raise ValueError(
            "Hand Landmarker input interface is fixed at (1, 256, 256); got {}".format(
                requested_shape
            )
        )

    module = import_module(module_name)
    return module.hand_landmark_2d_model(input_size=requested_shape, **kwargs)


def reparameterize_for_deploy(model, version: str = DEFAULT_VERSION, **kwargs: Any):
    """Fold training-only branches for versions that implement deployment fusion."""

    normalized = str(version).strip().lower()
    module_name = _MODEL_MODULES.get(normalized)
    if module_name is None:
        raise ValueError(
            "Unknown Hand Landmarker model version {!r}; available versions: {}".format(
                version, ", ".join(available_versions())
            )
        )
    module = import_module(module_name)
    converter = getattr(module, "reparameterize_for_deploy", None)
    if converter is None:
        return model
    return converter(model, **kwargs)
