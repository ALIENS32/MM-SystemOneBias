"""Run vision models via an OpenAI-compatible Chat Completions endpoint."""

from __future__ import annotations

import math
import re
import time

from .common import image_data_url, option_text, question_state
from .http_client import api_client, post_json


def answer_prompt(record: dict, instruction: str) -> str:
    rows = []
    context = question_state(record)
    if context:
        rows.extend(["Context:", context, ""])
    rows.extend([record["question"], "", "Options:"])
    rows.extend(option_text(option) for option in record["options"])
    rows.extend(["", instruction])
    return "\n".join(rows)


def build_request(record: dict, config: dict) -> dict:
    content = [{"type": "text", "text": answer_prompt(record, config["answer_instruction"])}]
    content.extend(
        {
            "type": "image_url",
            "image_url": {"url": image_data_url(path), "detail": config["image_detail"]},
        }
        for path in record["images"]
    )
    body = {
        "model": config["model"],
        "messages": [{"role": "user", "content": content}],
        "temperature": config["temperature"],
        "max_tokens": config["max_tokens"],
    }
    if config["logprobs"]:
        body["logprobs"] = True
        body["top_logprobs"] = config["top_logprobs"]
    if config.get("reasoning_effort"):
        body["reasoning_effort"] = config["reasoning_effort"]
    return body


def _message_text(raw: dict, sample_id: str) -> tuple[dict, str]:
    choices = raw.get("choices", [])
    if len(choices) != 1:
        raise RuntimeError(f"Expected one OpenAI-compatible choice for {sample_id}: {raw}")
    choice = choices[0]
    if choice.get("finish_reason") not in {None, "stop"}:
        raise RuntimeError(f"OpenAI-compatible response did not finish for {sample_id}: {raw}")
    message = choice.get("message", {})
    if message.get("refusal"):
        raise RuntimeError(
            f"OpenAI-compatible model refused sample {sample_id}: {message['refusal']}"
        )
    content = message.get("content")
    if isinstance(content, str):
        text = content
    elif isinstance(content, list):
        text = "".join(item.get("text", "") for item in content if isinstance(item, dict))
    else:
        text = ""
    if not text.strip():
        raise ValueError(f"OpenAI-compatible response has no text for {sample_id}: {raw}")
    return choice, text.strip()


def _answer_id(record: dict, text: str) -> str:
    lines = [line.strip() for line in text.splitlines() if line.strip()]
    candidate = lines[0] if lines else ""
    candidate = re.sub(
        r"^(?:(?:answer|label)\s*[:：]\s*|option\s+)", "", candidate, flags=re.IGNORECASE
    )
    candidate = candidate.strip("`*_ ").strip()
    for option in record["options"]:
        label = str(option["label"])
        if candidate == label or candidate in {
            f"{label}.",
            f"{label})",
            f"{label}:",
            option_text(option),
        }:
            return option["id"]
    raise ValueError(
        f"OpenAI-compatible model returned an invalid label for {record['sample_id']}: {text!r}"
    )


def _logsumexp(values: list[float]) -> float:
    maximum = max(values)
    return maximum + math.log(math.fsum(math.exp(value - maximum) for value in values))


def _first_token_probabilities(record: dict, choice: dict) -> tuple[dict | None, str]:
    labels = [str(option["label"]) for option in record["options"]]
    if any(
        left != right and (left.startswith(right) or right.startswith(left))
        for left in labels
        for right in labels
    ):
        return None, "overlapping_answer_labels"

    content = (choice.get("logprobs") or {}).get("content", [])
    if not content:
        return None, "first_token_logprobs_unavailable"
    first = content[0]
    entries = list(first.get("top_logprobs") or [])
    if first.get("token") is not None and first.get("logprob") is not None:
        entries.append({"token": first["token"], "logprob": first["logprob"]})

    token_scores = {}
    for entry in entries:
        token = entry.get("token")
        score = entry.get("logprob")
        if token is not None and score is not None:
            token_scores[token] = float(score)

    by_label = {label: [] for label in labels}
    for token, score in token_scores.items():
        label = token.strip()
        if label in by_label:
            by_label[label].append(score)
    if any(not scores for scores in by_label.values()):
        return None, "incomplete_legal_label_logprobs"

    label_scores = [_logsumexp(by_label[label]) for label in labels]
    maximum = max(label_scores)
    weights = [math.exp(score - maximum) for score in label_scores]
    total = math.fsum(weights)
    return (
        {option["id"]: weight / total for option, weight in zip(record["options"], weights)},
        "complete_first_token_distribution",
    )


class OpenAIChatPredictor:
    def __init__(self, config: dict):
        self.config = config
        self.client = api_client(config)

    def predict(self, record: dict) -> dict:
        start = time.perf_counter()
        raw = post_json(self.client, "chat/completions", build_request(record, self.config))
        try:
            choice, text = _message_text(raw, record["sample_id"])
            prediction = _answer_id(record, text)
            probabilities, status = (
                _first_token_probabilities(record, choice)
                if self.config["logprobs"]
                else (None, "logprobs_not_requested")
            )
        except (ValueError, RuntimeError) as error:
            error.raw_response = raw
            raise
        result = {
            "prediction": prediction,
            "probabilities": None,
            "probability_status": "full_answer_distribution_unavailable",
            "probability_method": "none",
            "first_token_probabilities": probabilities,
            "first_token_probability_status": status,
            "raw_response": raw,
            "usage": raw.get("usage", {}),
            "elapsed_seconds": time.perf_counter() - start,
            "native_task_type": "openai_chat_completion",
            "resolved_model": raw.get("model", self.config["model"]),
        }
        return result

    def close(self):
        self.client.close()
