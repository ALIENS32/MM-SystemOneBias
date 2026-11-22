"""Load and download the packed test dataset."""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any


DEFAULT_REPO_ID = "ALIENS232/MM-SystemOneBias"
DEFAULT_CACHE = Path.home() / ".cache" / "mm-systemonebias"


def _load_packed(dataset_path: str | Path | None, repo_id: str, revision: str | None):
    from datasets import load_dataset

    if dataset_path is not None:
        root = Path(dataset_path).expanduser()
        files = (
            sorted(root.glob("*.parquet"))
            if root.is_dir()
            else ([root] if root.suffix == ".parquet" else [])
        )
        if root.is_dir() and not files and (root / "test").is_dir():
            files = sorted((root / "test").glob("*.parquet"))
        if not files:
            raise FileNotFoundError(f"No Parquet shards found at {root}")
        return load_dataset("parquet", data_files={"test": [str(p) for p in files]}, split="test")
    kwargs: dict[str, Any] = {"split": "test"}
    if revision is not None:
        kwargs["revision"] = revision
    return load_dataset(repo_id, **kwargs)


def _materialize_images(
    images: list[dict[str, Any]], names: list[str], namespace: str, row_index: int, sample_id: str
) -> list[str]:
    sample_slug = re.sub(r"[^A-Za-z0-9._-]+", "_", sample_id)
    cache = DEFAULT_CACHE / "images" / namespace / f"{row_index:05d}-{sample_slug}"
    cache.mkdir(parents=True, exist_ok=True)
    paths = []
    for index, item in enumerate(images):
        name = Path(names[index]).name if names else f"image-{index}.bin"
        slug = re.sub(r"[^A-Za-z0-9._-]+", "_", name) or f"image-{index}.bin"
        suffix = Path(name).suffix or ".bin"
        target = cache / f"{index:04d}-{slug}{'' if slug.endswith(suffix) else suffix}"
        if not target.exists():
            payload = item.get("bytes")
            if payload is None and item.get("path"):
                payload = Path(item["path"]).read_bytes()
            if payload is None:
                raise ValueError(f"Packed image {index} has no bytes")
            target.write_bytes(payload)
        paths.append(str(target))
    return paths


def _expand(record: dict[str, Any], image_indices: list[int], paths: list[str]) -> dict[str, Any]:
    result = dict(record)
    result["images"] = [paths[index] for index in image_indices]
    options = []
    for option in result.get("options", []):
        item = dict(option)
        index = item.pop("image_index", None)
        if index is not None:
            item["image"] = paths[index]
        item = {key: value for key, value in item.items() if value is not None}
        options.append(item)
    result["options"] = options
    return result


def load_records(
    dataset_path: str | Path | None = None,
    repo_id: str = DEFAULT_REPO_ID,
    revision: str | None = None,
) -> list[dict[str, Any]]:
    """Load canonical expanded records and materialize embedded images under the user cache."""
    packed = _load_packed(dataset_path, repo_id, revision)
    from datasets import Image as HFImage

    packed = packed.cast_column("images", [HFImage(decode=False)])
    namespace = getattr(packed, "_fingerprint", None) or re.sub(r"\W+", "_", revision or repo_id)
    result = []
    for row_index, row in enumerate(packed):
        paths = _materialize_images(
            row["images"], row["image_names"], namespace, row_index, row["sample_id"]
        )
        base = {
            key: row[key]
            for key in (
                "sample_id",
                "root_id",
                "source_group",
                "question_group",
                "scenario",
                "task_type",
                "question",
                "options",
                "gold",
                "bias",
                "version",
                "intervention",
                "template",
                "dataset",
                "evaluation_metadata",
            )
        }
        base["metadata"] = json.loads(row["metadata"])
        result.append(_expand(base, row["input_image_indices"], paths))
        for variant in row["variants"]:
            item = {
                key: variant[key]
                for key in (
                    "sample_id",
                    "root_id",
                    "source_group",
                    "question_group",
                    "scenario",
                    "task_type",
                    "question",
                    "options",
                    "gold",
                    "bias",
                    "version",
                    "intervention",
                    "template",
                    "dataset",
                    "evaluation_metadata",
                )
            }
            item["metadata"] = json.loads(variant["metadata"])
            result.append(_expand(item, variant["input_image_indices"], paths))
    return result


def dataset_identity(
    dataset_path: str | Path | None = None,
    repo_id: str = DEFAULT_REPO_ID,
    revision: str | None = None,
) -> str:
    """Return a stable dataset source and resolved revision identity for run resumption."""
    if dataset_path is not None:
        packed = _load_packed(dataset_path, repo_id, revision)
        fingerprint = getattr(packed, "_fingerprint", "unknown")
        source = str(Path(dataset_path).expanduser().resolve())
        return f"local:{source}:{fingerprint}"
    from huggingface_hub import HfApi

    resolved = HfApi().dataset_info(repo_id, revision=revision).sha
    packed = _load_packed(None, repo_id, resolved)
    fingerprint = getattr(packed, "_fingerprint", "unknown")
    return f"hf:{repo_id}@{resolved}:{fingerprint}"


def download_dataset(
    repo_id: str = DEFAULT_REPO_ID,
    revision: str | None = None,
    output_dir: str | Path | None = None,
    allow_patterns: list[str] | None = None,
) -> Path:
    """Download a dataset snapshot into the user cache or requested directory."""
    from huggingface_hub import snapshot_download

    destination = Path(output_dir).expanduser() if output_dir else DEFAULT_CACHE / "dataset"
    snapshot_download(
        repo_id=repo_id,
        repo_type="dataset",
        revision=revision,
        local_dir=destination,
        allow_patterns=allow_patterns,
    )
    print(f"Dataset downloaded to {destination}")
    return destination
