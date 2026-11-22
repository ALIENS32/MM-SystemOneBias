"""Official System One image interface client for TypeSafe Jev."""

from __future__ import annotations

import base64
import io
import time
import uuid
from pathlib import Path

from .common import native_probabilities, native_question, question_state, validate_probabilities
from .http_client import api_client, post_json
from .typesafe_images import TypeSafeImageStore


def response_answers(raw: dict) -> tuple[dict, str | None, dict]:
    """Extract answers from either top-level `answers` or Community `data.answers` response wrappers."""
    payload = raw.get("data", raw)
    if not isinstance(payload, dict) or not isinstance(payload.get("answers"), dict):
        raise ValueError("Jev response does not contain an answers object")
    return payload["answers"], raw.get("model") or payload.get("model"), payload


def build_request(
    record: dict, config: dict, image_store: TypeSafeImageStore | None = None
) -> dict:
    assets, references = {}, []
    image_paths = image_store.prepare_record(record) if image_store else record["images"]
    for path in image_paths:
        key = str(uuid.uuid4())
        payload = io.BytesIO(Path(path).read_bytes())
        assets[key] = {
            "type": "image",
            "format": Path(path).suffix.removeprefix(".").lower() or "png",
            "data_base64": base64.b64encode(payload.getvalue()).decode("ascii"),
        }
        references.append(key)
    return {
        "model": config["model"],
        "state": {"context": question_state(record), "images": references},
        "assets": assets,
        "questions": {"answer": native_question(record)},
    }


class TypeSafePredictor:
    def __init__(self, config: dict):
        self.config = config
        self.client = api_client(config)
        self.image_store = (
            TypeSafeImageStore(config["image_output_dir"], config)
            if config.get("image_output_dir")
            else None
        )

    def close(self):
        self.client.close()

    @staticmethod
    def _endpoint(config: dict) -> str:
        endpoint = config.get("endpoint", "/typesafe/v1/systemone")
        if config["base_url"].rstrip("/").endswith("/v1") and endpoint.startswith("/v1/"):
            return endpoint[3:]
        return endpoint

    def predict(self, record: dict) -> dict:
        start = time.perf_counter()
        body = build_request(record, self.config, self.image_store)
        if self.image_store:
            self.image_store.validate_request(record, body)
        raw = post_json(self.client, self._endpoint(self.config), body)
        answers, resolved_model, payload = response_answers(raw)
        values = native_probabilities(record, answers["answer"])
        total = sum(values.values())
        if total <= 0 or any(not 0 <= value <= 1 for value in values.values()):
            raise ValueError(f"Invalid Jev probabilities for {record['sample_id']}: {values}")
        result = {
            "probabilities": validate_probabilities(
                {key: value / total for key, value in values.items()}, record
            ),
            "native_probability_mass": total,
            "raw_response": raw,
            "elapsed_seconds": time.perf_counter() - start,
            "native_task_type": body["questions"]["answer"]["type"],
        }
        result["usage"] = raw.get("usage", payload.get("usage", {}))
        if resolved_model:
            result["resolved_model"] = resolved_model
        if self.image_store:
            result["image_version"] = self.image_store.version
            result["image_paths"] = self.image_store.prepare_record(record)
        return result
