"""Mappings between canonical task records and native decision interfaces."""

from __future__ import annotations

import base64
import importlib.util
import math
import sys
import types
from pathlib import Path


def option_text(option: dict) -> str:
    separator = ": " if "value" in option else ". "
    return f"{option['label']}{separator}{option['text']}"


def native_question(record: dict) -> dict:
    kind = record["task_type"]
    question = {"type": "noul" if kind == "predicate" else kind, "instructions": record["question"]}
    if kind == "choice":
        question["criteria"] = {str(o["label"]): option_text(o) for o in record["options"]}
    elif kind == "score":
        question["criteria"] = [option_text(o) for o in record["options"]]
    else:
        question["criteria"] = {
            ("true" if o["label"].lower() == "yes" else "false"): o["text"]
            for o in record["options"]
        }
    return question


def native_probabilities(record: dict, answer: dict) -> dict[str, float]:
    if record["task_type"] == "predicate":
        yes = float(answer["noul"])
        return {o["id"]: yes if o["label"].lower() == "yes" else 1 - yes for o in record["options"]}
    keys = (
        [str(i) for i in range(len(record["options"]))]
        if record["task_type"] == "score"
        else [str(o["label"]) for o in record["options"]]
    )
    return {o["id"]: float(answer["probabilities"][key]) for o, key in zip(record["options"], keys)}


def validate_probabilities(values: dict[str, float], record: dict) -> dict[str, float]:
    expected = {o["id"] for o in record["options"]}
    if set(values) != expected or any(not math.isfinite(v) or v < 0 for v in values.values()):
        raise ValueError(f"Invalid candidate probabilities for {record['sample_id']}: {values}")
    total = sum(values.values())
    if abs(total - 1) > 1e-4:
        raise ValueError(f"Candidate probabilities sum to {total}, sample={record['sample_id']}")
    return {key: value / total for key, value in values.items()}


def load_release_module(root: Path, filename: str):
    name = (
        "_mm_bias_release_" + root.parent.name.replace("-", "_") + "_" + root.name.replace("-", "_")
    )
    if name not in sys.modules:
        package = types.ModuleType(name)
        package.__path__ = [str(root)]
        sys.modules[name] = package
    module_name = name + "." + Path(filename).stem
    if module_name not in sys.modules:
        spec = importlib.util.spec_from_file_location(module_name, root / filename)
        if spec is None or spec.loader is None:
            raise ImportError(str(root / filename))
        module = importlib.util.module_from_spec(spec)
        sys.modules[module_name] = module
        spec.loader.exec_module(module)
    return sys.modules[module_name]


def image_data_url(path: str) -> str:
    mime = {
        ".png": "image/png",
        ".jpg": "image/jpeg",
        ".jpeg": "image/jpeg",
        ".webp": "image/webp",
    }.get(Path(path).suffix.lower())
    if mime is None:
        raise ValueError(f"Unsupported image format: {path}")
    return f"data:{mime};base64," + base64.b64encode(Path(path).read_bytes()).decode("ascii")


def question_state(record: dict) -> str:
    return record.get("state", record.get("metadata", {}).get("state", ""))


def native_result(record: dict, answer: dict, raw: dict, elapsed: float, native_type: str) -> dict:
    return {
        "probabilities": validate_probabilities(native_probabilities(record, answer), record),
        "raw_response": raw,
        "usage": raw.get("usage", {}),
        "elapsed_seconds": elapsed,
        "native_task_type": native_type,
    }
