"""Small filesystem, JSONL, hashing, and image helpers."""

from __future__ import annotations

import hashlib
import json
import os
import tempfile
from pathlib import Path
from typing import Any, Dict, Iterable, Iterator, List, Mapping, Optional, Sequence, Union


PathLike = Union[str, Path]
VALID_IMAGE_EXTENSIONS = {".tif", ".tiff", ".png", ".jpg", ".jpeg", ".bmp", ".webp"}


def iter_jsonl(path: PathLike) -> Iterator[Dict[str, Any]]:
    path = Path(path)
    with path.open("r", encoding="utf-8") as handle:
        for line_number, raw in enumerate(handle, start=1):
            text = raw.strip()
            if not text:
                continue
            try:
                row = json.loads(text)
            except json.JSONDecodeError as exc:
                raise ValueError("Invalid JSON at {}:{}: {}".format(path, line_number, exc)) from exc
            if not isinstance(row, dict):
                raise ValueError("JSONL row must be an object at {}:{}".format(path, line_number))
            # Internal provenance always reflects the physical file position;
            # a serialized row may not override it.
            row["_jsonl_line"] = line_number
            yield row


def read_jsonl(path: PathLike) -> List[Dict[str, Any]]:
    return list(iter_jsonl(path))


def _atomic_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    handle = tempfile.NamedTemporaryFile(
        mode="w", encoding="utf-8", newline="\n", delete=False, dir=str(path.parent), prefix=path.name + ".tmp."
    )
    temp_path = Path(handle.name)
    try:
        with handle:
            handle.write(text)
        os.replace(str(temp_path), str(path))
    finally:
        if temp_path.exists():
            temp_path.unlink()


def write_json(path: PathLike, value: Any) -> None:
    text = json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n"
    _atomic_text(Path(path), text)


def write_jsonl(path: PathLike, rows: Iterable[Mapping[str, Any]]) -> None:
    text = "".join(json.dumps(dict(row), ensure_ascii=False, separators=(",", ":")) + "\n" for row in rows)
    _atomic_text(Path(path), text)


def sha256_file(path: PathLike, chunk_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        while True:
            chunk = handle.read(chunk_size)
            if not chunk:
                break
            digest.update(chunk)
    return digest.hexdigest()


def image_files(root: PathLike, recursive: bool = True) -> List[Path]:
    root = Path(root)
    iterator = root.rglob("*") if recursive else root.iterdir()
    paths = [path for path in iterator if path.is_file() and path.suffix.lower() in VALID_IMAGE_EXTENSIONS]
    return sorted(paths, key=lambda path: str(path).lower())


def build_basename_index(roots: Sequence[PathLike]) -> Dict[str, List[Path]]:
    index: Dict[str, List[Path]] = {}
    for root_value in roots:
        root = Path(root_value)
        if not root.is_dir():
            continue
        for path in image_files(root, recursive=True):
            index.setdefault(path.name, []).append(path.resolve())
    return index


def resolve_record_image(
    row: Mapping[str, Any],
    roots: Sequence[PathLike],
    index: Optional[Mapping[str, Sequence[Path]]] = None,
    path_key: str = "crop_path",
) -> Path:
    """Resolve a canonical image without ever globbing samples into the dataset."""

    recorded = row.get(path_key)
    if recorded:
        candidate = Path(str(recorded))
        if candidate.is_file():
            return candidate.resolve()

    basename = Path(str(recorded or "")).name
    if not basename:
        raise FileNotFoundError("Record has no resolvable {}: {}".format(path_key, row.get("crop_id")))
    if index is not None:
        matches = list(index.get(basename, []))
    else:
        matches = []
        for root_value in roots:
            candidate = Path(root_value) / basename
            if candidate.is_file():
                matches.append(candidate.resolve())
    unique = sorted({str(path): Path(path) for path in matches}.values(), key=str)
    if not unique:
        raise FileNotFoundError("Image {} was not found under configured roots".format(basename))
    if len(unique) > 1:
        raise ValueError("Ambiguous image basename {}: {}".format(basename, [str(path) for path in unique]))
    return unique[0]


def read_image(path: PathLike, flags: Optional[int] = None):
    import cv2
    import numpy as np

    mode = cv2.IMREAD_UNCHANGED if flags is None else int(flags)
    data = np.fromfile(str(path), dtype=np.uint8)
    if data.size == 0:
        return None
    return cv2.imdecode(data, mode)


def write_image(path: PathLike, image) -> None:
    import cv2

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    extension = path.suffix or ".png"
    ok, encoded = cv2.imencode(extension, image)
    if not ok:
        raise RuntimeError("Could not encode image: {}".format(path))
    encoded.tofile(str(path))


def to_uint8_gray(image):
    import cv2
    import numpy as np

    if image.ndim == 2:
        gray = image
    elif image.ndim == 3 and image.shape[2] == 1:
        gray = image[:, :, 0]
    elif image.ndim == 3 and image.shape[2] == 4:
        gray = cv2.cvtColor(image, cv2.COLOR_BGRA2GRAY)
    elif image.ndim == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        raise ValueError("Unsupported image shape: {}".format(image.shape))
    if gray.dtype == np.uint8:
        return gray
    if np.issubdtype(gray.dtype, np.integer):
        scale = 255.0 / float(np.iinfo(gray.dtype).max)
        return np.clip(np.rint(gray.astype(np.float32) * scale), 0, 255).astype(np.uint8)
    values = gray.astype(np.float32)
    finite = values[np.isfinite(values)]
    if finite.size == 0:
        return np.zeros(gray.shape, dtype=np.uint8)
    scale = 255.0 if float(np.max(finite)) <= 1.0 else 255.0 / max(float(np.max(finite)), 1e-12)
    return np.clip(np.rint(values * scale), 0, 255).astype(np.uint8)


def ensure_bgr(image):
    import cv2

    if image.ndim == 2:
        return cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    if image.ndim == 3 and image.shape[2] == 1:
        return cv2.cvtColor(image[:, :, 0], cv2.COLOR_GRAY2BGR)
    if image.ndim == 3 and image.shape[2] == 4:
        return cv2.cvtColor(image, cv2.COLOR_BGRA2BGR)
    return image.copy()
