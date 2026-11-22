"""System One readout client for the official AutoTrust service."""

from __future__ import annotations

import time

from .common import image_data_url, option_text, question_state, validate_probabilities
from .http_client import post_json


def context_counter(config):
    from transformers import AutoProcessor
    from .context_tokens import qwen_length
    from .resources import require_model

    processor = AutoProcessor.from_pretrained(require_model(config), local_files_only=True)

    def count(record):
        predicate = record["task_type"] == "predicate"
        kind = "noul" if predicate else "choice"
        options = (
            ["false", "true"]
            if predicate
            else [f"{chr(65 + i)}) {option_text(o)}" for i, o in enumerate(record["options"])]
        )
        images = "<|vision_start|><|image_pad|><|vision_end|>" * len(record["images"])
        text = (
            f"[kind] {kind}\n[state] {question_state(record)}{images}"
            f"\n[question] {record['question']}\n[options]\n" + "\n".join(options) + "\n[decision]:"
        )
        return qwen_length(processor, text, record["images"])

    return count


class AutoTrustPredictor:
    def __init__(self, config: dict):
        import httpx

        self.config = config
        self.client = httpx.Client(
            base_url=config["base_url"].rstrip("/") + "/", timeout=config["timeout_seconds"]
        )

    def close(self):
        self.client.close()

    def predict(self, record: dict) -> dict:
        start = time.perf_counter()
        predicate = record["task_type"] == "predicate"
        body = {
            "kind": "noul" if predicate else "choice",
            "state": [question_state(record)]
            + [{"image": image_data_url(path)} for path in record["images"]],
            "question": record["question"],
            "thinking": "off",
            "strategy": "single",
        }
        if not predicate:
            body["options"] = [option_text(o) for o in record["options"]]
        raw = post_json(self.client, "v1/decide", body)
        if raw.get("thinking", {}).get("used"):
            raise RuntimeError("AutoTrust returned a System 2 mixture for a System 1 request")
        values = dict(zip(raw["options"], raw["probabilities"]))
        if predicate:
            probabilities = {
                o["id"]: values["true" if o["label"].lower() == "yes" else "false"]
                for o in record["options"]
            }
        else:
            probabilities = {o["id"]: values[option_text(o)] for o in record["options"]}
        return {
            "probabilities": validate_probabilities(probabilities, record),
            "raw_response": raw,
            "usage": raw.get("usage", {}),
            "elapsed_seconds": time.perf_counter() - start,
            "native_task_type": body["kind"],
            "resolved_model": raw.get("model"),
        }
