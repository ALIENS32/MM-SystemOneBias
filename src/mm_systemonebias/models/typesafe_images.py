"""Prepare and validate images for the TypeSafe Jev image interface."""

from __future__ import annotations

import json
import threading
from pathlib import Path

from PIL import Image, ImageOps


DEFAULT_VERSION = "typesafe-jev-image-v1"
DEFAULT_MIN_SIDE = 280
DEFAULT_MAX_SIDE = 1568
DEFAULT_MAX_IMAGE_BYTES = 16 * 1024 * 1024
DEFAULT_MAX_REQUEST_BYTES = 32 * 1024 * 1024


def _save_image(image: Image.Image, destination: Path, max_bytes: int) -> tuple[str, int, Path]:
    destination.parent.mkdir(parents=True, exist_ok=True)
    image.save(destination, format="PNG", optimize=True)
    size = destination.stat().st_size
    if size <= max_bytes:
        return "png", size, destination
    jpeg_destination = destination.with_suffix(".jpg")
    image.save(jpeg_destination, format="JPEG", quality=95, optimize=True)
    size = jpeg_destination.stat().st_size
    if size > max_bytes:
        raise ValueError(
            f"Jev image exceeds {max_bytes} bytes after PNG and JPEG encoding: {destination}"
        )
    destination.unlink()
    return "jpeg", size, jpeg_destination


def _prepare_image(
    source: Path, destination: Path, min_side: int, max_side: int, max_bytes: int
) -> dict:
    with Image.open(source) as opened:
        image = ImageOps.exif_transpose(opened).convert("RGB")
    original_width, original_height = image.size
    scale = min(1.0, max_side / max(original_width, original_height))
    width = max(1, round(original_width * scale))
    height = max(1, round(original_height * scale))
    image = (
        image.resize((width, height), Image.Resampling.LANCZOS)
        if (width, height) != image.size
        else image
    )
    if min(width, height) < min_side and max(width, height) * min_side <= max_side * min(
        width, height
    ):
        scale = min_side / min(width, height)
        width = round(width * scale)
        height = round(height * scale)
        image = image.resize((width, height), Image.Resampling.LANCZOS)
    content_width, content_height = width, height
    canvas_width = max(width, min_side)
    canvas_height = max(height, min_side)
    if (canvas_width, canvas_height) != image.size:
        canvas = Image.new("RGB", (canvas_width, canvas_height), "white")
        canvas.paste(image, ((canvas_width - width) // 2, (canvas_height - height) // 2))
        image = canvas
    if max(image.size) > max_side:
        raise ValueError(f"Jev image normalization exceeded {max_side}px: {source} -> {image.size}")
    encoding, byte_size, saved_path = _save_image(image, destination, max_bytes)
    if min(image.size) < min_side:
        raise ValueError(
            f"Jev image normalization produced an invalid minimum side: {source} -> {image.size}"
        )
    return {
        "source_path": str(source),
        "original_width": original_width,
        "original_height": original_height,
        "derived_path": str(saved_path),
        "derived_width": image.width,
        "derived_height": image.height,
        "format": encoding,
        "bytes": byte_size,
        "transform": {
            "resize": [original_width, original_height] != [image.width, image.height],
            "padding": image.width != content_width or image.height != content_height,
            "content_width": content_width,
            "content_height": content_height,
        },
    }


class TypeSafeImageStore:
    """Versioned source-path to derived-image mapping shared by all variants."""

    def __init__(self, output_dir: str | Path, config: dict):
        self.output_dir = Path(output_dir).expanduser().resolve()
        self.manifest_path = self.output_dir / "manifest.json"
        self.version = config.get("image_version", DEFAULT_VERSION)
        self.min_side = int(config.get("image_min_side", DEFAULT_MIN_SIDE))
        self.max_side = int(config.get("image_max_side", DEFAULT_MAX_SIDE))
        self.max_image_bytes = int(config.get("image_max_bytes", DEFAULT_MAX_IMAGE_BYTES))
        self.max_request_bytes = int(
            config.get("image_max_request_bytes", DEFAULT_MAX_REQUEST_BYTES)
        )
        self._lock = threading.Lock()
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self._entries = self._load()
        self._by_source = {entry["source_path"]: entry for entry in self._entries}

    def _load(self) -> list[dict]:
        if not self.manifest_path.exists():
            return []
        manifest = json.loads(self.manifest_path.read_text(encoding="utf-8"))
        if manifest.get("version") != self.version:
            raise ValueError(f"Jev image manifest version differs: {self.manifest_path}")
        expected = {
            "min_side": self.min_side,
            "max_side": self.max_side,
            "max_image_bytes": self.max_image_bytes,
        }
        if manifest.get("limits") != expected:
            raise ValueError(f"Jev image manifest limits differ: {self.manifest_path}")
        return list(manifest.get("images", []))

    def _save_manifest(self) -> None:
        payload = {
            "version": self.version,
            "limits": {
                "min_side": self.min_side,
                "max_side": self.max_side,
                "max_image_bytes": self.max_image_bytes,
            },
            "images": self._entries,
        }
        self.manifest_path.write_text(
            json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
        )

    def prepare(self, source_path: str | Path) -> str:
        source = Path(source_path).expanduser().resolve()
        if not source.is_file():
            raise FileNotFoundError(f"Jev source image not found: {source}")
        source_key = str(source)
        with self._lock:
            existing = self._by_source.get(source_key)
            if existing and Path(existing["derived_path"]).is_file():
                return existing["derived_path"]
            index = len(self._entries)
            destination = self.output_dir / "images" / f"image_{index:06d}.png"
            entry = _prepare_image(
                source, destination, self.min_side, self.max_side, self.max_image_bytes
            )
            entry["source_path"] = source_key
            entry["image_version"] = self.version
            self._entries.append(entry)
            self._by_source[source_key] = entry
            self._save_manifest()
            return entry["derived_path"]

    def prepare_record(self, record: dict) -> list[str]:
        return [self.prepare(path) for path in record.get("images", [])]

    def validate_request(self, record: dict, body: dict) -> None:
        request_bytes = len(
            json.dumps(body, ensure_ascii=False, separators=(",", ":")).encode("utf-8")
        )
        if request_bytes > self.max_request_bytes:
            raise ValueError(
                f"Jev request exceeds {self.max_request_bytes} bytes for sample {record.get('sample_id')}: "
                f"{request_bytes} bytes"
            )
