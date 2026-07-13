"""YAML configuration loading with inheritance and environment expansion."""

from __future__ import annotations

import copy
import os
import re
from pathlib import Path
from typing import Any, Dict, Iterable, Mapping, MutableMapping, Optional, Union


PathLike = Union[str, Path]
_ENV_PATTERN = re.compile(r"\$\{([A-Za-z_][A-Za-z0-9_]*)(?::-([^}]*))?\}")
REPO_ROOT = Path(__file__).resolve().parents[1]


def _deep_merge(base: MutableMapping[str, Any], overlay: Mapping[str, Any]) -> MutableMapping[str, Any]:
    for key, value in overlay.items():
        if isinstance(value, Mapping) and isinstance(base.get(key), MutableMapping):
            _deep_merge(base[key], value)
        else:
            base[key] = copy.deepcopy(value)
    return base


def _expand_string(value: str) -> str:
    def replace(match: re.Match) -> str:
        name = match.group(1)
        default = match.group(2)
        if name in os.environ:
            return os.environ[name]
        if default is not None:
            return default
        raise ValueError("Environment variable {} is required by configuration".format(name))

    return os.path.expanduser(_ENV_PATTERN.sub(replace, value))


def _expand_tree(value: Any) -> Any:
    if isinstance(value, str):
        return _expand_string(value)
    if isinstance(value, list):
        return [_expand_tree(item) for item in value]
    if isinstance(value, tuple):
        return tuple(_expand_tree(item) for item in value)
    if isinstance(value, Mapping):
        return {key: _expand_tree(item) for key, item in value.items()}
    return value


def _read_yaml(path: Path) -> Dict[str, Any]:
    try:
        import yaml
    except ImportError as exc:
        raise RuntimeError("PyYAML is required; create the documented conda environment first") from exc
    with path.open("r", encoding="utf-8") as handle:
        value = yaml.safe_load(handle) or {}
    if not isinstance(value, dict):
        raise ValueError("Configuration root must be a mapping: {}".format(path))
    return value


def _load_recursive(path: Path, stack: Iterable[Path]) -> Dict[str, Any]:
    path = path.resolve()
    active = list(stack)
    if path in active:
        chain = " -> ".join(str(item) for item in active + [path])
        raise ValueError("Cyclic configuration inheritance: {}".format(chain))
    if not path.is_file():
        raise FileNotFoundError("Configuration file not found: {}".format(path))
    raw = _read_yaml(path)
    parent_ref = raw.pop("extends", None)
    merged: Dict[str, Any] = {}
    if parent_ref:
        parents = parent_ref if isinstance(parent_ref, list) else [parent_ref]
        for parent in parents:
            parent_path = Path(str(parent))
            if not parent_path.is_absolute():
                parent_path = path.parent / parent_path
            _deep_merge(merged, _load_recursive(parent_path, active + [path]))
    _deep_merge(merged, raw)
    return merged


def load_config(path: PathLike) -> Dict[str, Any]:
    """Load a YAML file, recursively merge ``extends``, and expand env vars."""

    config_path = Path(path).resolve()
    config = _expand_tree(_load_recursive(config_path, []))
    config["_meta"] = {
        "config_path": str(config_path),
        "config_dir": str(config_path.parent),
        "repo_root": str(REPO_ROOT),
    }
    return config


def resolve_path(value: PathLike, config: Optional[Mapping[str, Any]] = None, base: str = "repo") -> Path:
    """Resolve a configured path relative to the repository or config file."""

    path = Path(str(value))
    if path.is_absolute():
        return path
    if base == "config" and config is not None:
        root = Path(str(config.get("_meta", {}).get("config_dir", REPO_ROOT)))
    else:
        root = Path(str((config or {}).get("_meta", {}).get("repo_root", REPO_ROOT)))
    return (root / path).resolve()


def require(config: Mapping[str, Any], dotted_key: str) -> Any:
    """Fetch a required dotted key with a useful error message."""

    current: Any = config
    for part in dotted_key.split("."):
        if not isinstance(current, Mapping) or part not in current:
            raise KeyError("Missing required configuration key: {}".format(dotted_key))
        current = current[part]
    return current

