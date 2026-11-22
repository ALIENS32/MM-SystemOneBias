"""Full-answer sequence conditional probabilities and first-token probabilities for VLMs."""

from __future__ import annotations

import inspect
import time

from .common import option_text, validate_probabilities
from .hf_loading import input_device, load_kwargs, move_inputs
from .resources import require_model


def answer_prompt(record: dict, instruction: str) -> str:
    rows = [record["question"], "", "Options:"]
    rows.extend(option_text(o) for o in record["options"])
    rows.extend(["", instruction])
    return "\n".join(rows)


def context_counter(config):
    from functools import lru_cache
    from transformers import AutoProcessor
    from .context_tokens import image_size

    processor = AutoProcessor.from_pretrained(require_model(config), local_files_only=True)

    @lru_cache(maxsize=None)
    def image_tokens(path):
        height, width = image_size(path)
        if config["family"] == "lfm_vlm":
            ip = processor.image_processor
            new_width, new_height = ip.smart_resize(
                height,
                width,
                ip.downsample_factor,
                ip.min_image_tokens,
                ip.max_image_tokens,
                ip.encoder_patch_size,
            )
            large = ip._is_image_too_large(
                height,
                width,
                ip.max_image_tokens,
                ip.encoder_patch_size,
                ip.downsample_factor,
                ip.max_pixels_tolerance,
            )
            cols, rows = (
                ip._get_grid_layout(height, width, ip.min_tiles, ip.max_tiles, ip.tile_size)[:2]
                if large and ip.do_image_splitting
                else (1, 1)
            )
            return processor.replace_image_token(
                {
                    "image_rows": [rows],
                    "image_cols": [cols],
                    "image_sizes": [[new_height, new_width]],
                },
                0,
                use_image_special_tokens=True,
            )
        count = processor._get_num_multimodal_tokens(
            image_sizes=[(height, width)]
        ).num_image_tokens[0]
        if config["family"] == "gemma_vlm":
            return processor.replace_image_token({"num_soft_tokens_per_image": [count]}, 0)
        return processor.image_token * count

    def count(record):
        content = [{"type": "image"} for _ in record["images"]]
        content.append(
            {"type": "text", "text": answer_prompt(record, config["answer_instruction"])}
        )
        text = processor.apply_chat_template(
            [{"role": "user", "content": content}],
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=False,
        )
        parts = text.split(processor.image_token)
        replacements = [image_tokens(path) for path in record["images"]]
        if len(parts) != len(replacements) + 1:
            raise ValueError(f"Image placeholder count differs: {record['sample_id']}")
        expanded = (
            "".join(part + replacement for part, replacement in zip(parts, replacements))
            + parts[-1]
        )
        # Legal answer strings are scored alongside the prompt; include the longest candidate length.
        return len(processor.tokenizer.encode(expanded, add_special_tokens=False)) + max(
            len(processor.tokenizer.encode(str(option["label"]), add_special_tokens=False))
            for option in record["options"]
        )

    return count


def normalize_log_scores(scores: list[float]) -> list[float]:
    import math

    maximum = max(scores)
    weights = [math.exp(value - maximum) for value in scores]
    total = sum(weights)
    return [value / total for value in weights]


class VLMPredictor:
    def __init__(self, config: dict):
        from transformers import AutoModelForImageTextToText, AutoProcessor

        self.config = config
        path = require_model(config)
        self.processor = AutoProcessor.from_pretrained(path, local_files_only=True)
        self.model = AutoModelForImageTextToText.from_pretrained(path, **load_kwargs(config)).eval()
        self.device = input_device(self.model)
        self.parameters = inspect.signature(self.model.forward).parameters
        self.processor.tokenizer.padding_side = "right"

    def predict_batch(self, records: list[dict]) -> list[dict]:
        import torch
        from PIL import Image

        tokenizer = self.processor.tokenizer
        tokens = [
            [tokenizer.encode(str(o["label"]), add_special_tokens=False) for o in row["options"]]
            for row in records
        ]
        if self.config["thinking_budget"] or any(len(t) != 1 for row in tokens for t in row):
            return [self.predict(row) for row in records]
        start = time.perf_counter()
        texts, pictures = [], []
        for record in records:
            images = [Image.open(path).convert("RGB") for path in record["images"]]
            content = [{"type": "image"} for _ in images]
            content.append(
                {"type": "text", "text": answer_prompt(record, self.config["answer_instruction"])}
            )
            texts.append(
                self.processor.apply_chat_template(
                    [{"role": "user", "content": content}],
                    tokenize=False,
                    add_generation_prompt=True,
                    enable_thinking=False,
                )
            )
            pictures.append(images)
        inputs = self.processor(
            text=texts, images=pictures, padding=True, return_tensors="pt", add_special_tokens=False
        )
        if inputs["input_ids"].shape[-1] > self.config["max_input_tokens"]:
            raise ValueError("VLM batch input exceeds configured token limit")
        lengths = inputs["attention_mask"].sum(-1).tolist()
        inputs = move_inputs(inputs, self.device)
        positions = torch.tensor(lengths, device=self.device) - 1
        extra = (
            {"logits_to_keep": positions.cpu()}
            if "logits_to_keep" in self.parameters
            else {"num_logits_to_keep": positions.cpu()}
            if "num_logits_to_keep" in self.parameters
            else {}
        )
        with torch.inference_mode():
            if self.config.get("family") == "qwen_vlm":
                hidden = self.model.model(**inputs, use_cache=False)[0]
                row_ids = torch.arange(len(records), device=hidden.device)
                selected = hidden[row_ids, positions.to(hidden.device)]
                rows = self.model.get_output_embeddings()(selected).float().log_softmax(-1)
            else:
                logits = self.model(**inputs, use_cache=False, **extra).logits
                row_ids = torch.arange(len(records), device=logits.device)
                rows = (
                    logits[row_ids, row_ids if extra else positions.to(logits.device)]
                    .float()
                    .log_softmax(-1)
                )
            scores = [
                [float(row[item[0]]) for item in answer]
                for row, answer in zip(rows, tokens, strict=True)
            ]
        elapsed = (time.perf_counter() - start) / len(records)
        results = []
        for record, answer, log_scores, length in zip(
            records, tokens, scores, lengths, strict=True
        ):
            values = normalize_log_scores(log_scores)
            probabilities = validate_probabilities(
                {o["id"]: p for o, p in zip(record["options"], values)}, record
            )
            results.append(
                {
                    "probabilities": probabilities,
                    "first_token_probabilities": probabilities,
                    "raw_response": {
                        "answer_strings": [str(o["label"]) for o in record["options"]],
                        "answer_token_ids": answer,
                        "sequence_log_probabilities": log_scores,
                        "first_token_log_probabilities": log_scores,
                        "thinking_tokens": 0,
                        "forced_thinking_end": False,
                    },
                    "usage": {
                        "input_tokens": length,
                        "processed_prompt_tokens": length,
                        "output_tokens": 0,
                        "scored_answer_tokens": len(answer),
                    },
                    "elapsed_seconds": elapsed,
                    "inference_batch_size": len(records),
                    "native_task_type": "sequence_scoring",
                }
            )
        return results

    def encode(self, text: str, images: list):
        inputs = self.processor(
            text=[text], images=images or None, return_tensors="pt", add_special_tokens=False
        )
        if inputs["input_ids"].shape[-1] > self.config["max_input_tokens"]:
            raise ValueError("VLM input exceeds configured token limit")
        return move_inputs(inputs, self.device)

    def reasoning_prefix(self, prefix: str, images: list) -> tuple[str, dict]:
        import torch
        from transformers import StoppingCriteria, StoppingCriteriaList
        from tqdm import tqdm

        budget = self.config["thinking_budget"]
        if not budget:
            return prefix, {"thinking_tokens": 0, "forced_thinking_end": False}
        tokenizer = self.processor.tokenizer
        end_ids = tokenizer.encode(self.config["thinking_end"], add_special_tokens=False)
        inputs = self.encode(prefix, images)
        prompt_length = inputs["input_ids"].shape[-1]
        progress = tqdm(total=budget, desc="thinking", unit="token", leave=False)

        class EndThinking(StoppingCriteria):
            def __call__(self, input_ids, scores, **kwargs):
                progress.update(input_ids.shape[-1] - prompt_length - progress.n)
                suffix = input_ids[0, -len(end_ids) :].tolist()
                return suffix == end_ids

        with progress, torch.inference_mode():
            generated = self.model.generate(
                **inputs,
                do_sample=False,
                max_new_tokens=budget,
                stopping_criteria=StoppingCriteriaList([EndThinking()]),
                use_cache=True,
            )
        tokens = generated[0, inputs["input_ids"].shape[-1] :].tolist()
        ended = len(tokens) >= len(end_ids) and tokens[-len(end_ids) :] == end_ids
        reasoning = tokenizer.decode(tokens, skip_special_tokens=False)
        tail = (
            self.config["after_thinking"]
            if ended
            else self.config["thinking_end"] + self.config["after_thinking"]
        )
        return prefix + reasoning + tail, {
            "thinking_tokens": len(tokens),
            "forced_thinking_end": not ended,
            "reasoning": reasoning,
        }

    def predict(self, record: dict) -> dict:
        import torch
        from PIL import Image

        start = time.perf_counter()
        images = [Image.open(path).convert("RGB") for path in record["images"]]
        content = [{"type": "image"} for _ in images]
        content.append(
            {"type": "text", "text": answer_prompt(record, self.config["answer_instruction"])}
        )
        kwargs = {"enable_thinking": bool(self.config["thinking_budget"])}
        prefix = self.processor.apply_chat_template(
            [{"role": "user", "content": content}],
            tokenize=False,
            add_generation_prompt=True,
            **kwargs,
        )
        original_input_tokens = (
            self.encode(prefix, images)["input_ids"].shape[-1]
            if self.config["thinking_budget"]
            else None
        )
        prefix, reasoning = self.reasoning_prefix(prefix, images)
        prefix_inputs = self.encode(prefix, images)
        prefix_ids = prefix_inputs["input_ids"][0].tolist()
        tokenizer = self.processor.tokenizer
        labels = [str(o["label"]) for o in record["options"]]
        log_scores, token_counts, first_logits = [], [], None
        token_sequences = [tokenizer.encode(label, add_special_tokens=False) for label in labels]
        single_token_answers = all(len(tokens) == 1 for tokens in token_sequences)
        with torch.inference_mode():
            for label, answer_ids in zip(labels, token_sequences, strict=True):
                inputs = dict(prefix_inputs)
                appended = torch.tensor(
                    [answer_ids], device=self.device, dtype=inputs["input_ids"].dtype
                )
                inputs["input_ids"] = torch.cat([inputs["input_ids"], appended], dim=-1)
                for key in ("attention_mask", "token_type_ids", "mm_token_type_ids"):
                    if key in inputs:
                        fill = 1 if key == "attention_mask" else 0
                        extension = torch.full_like(appended, fill, dtype=inputs[key].dtype)
                        inputs[key] = torch.cat([inputs[key], extension], dim=-1)
                if inputs["input_ids"].shape[-1] > self.config["max_input_tokens"]:
                    raise ValueError("VLM prompt and answer exceed configured token limit")
                if not answer_ids:
                    raise ValueError(f"Answer has no tokens: {label!r}")
                extra = {}
                if "logits_to_keep" in self.parameters:
                    extra["logits_to_keep"] = len(answer_ids) + 1
                elif "num_logits_to_keep" in self.parameters:
                    extra["num_logits_to_keep"] = len(answer_ids) + 1
                logits = self.model(**inputs, use_cache=False, **extra).logits[0]
                logits = logits[-len(answer_ids) - 1 : -1].float()
                if first_logits is None:
                    first_logits = logits[0].log_softmax(-1)
                targets = torch.tensor(answer_ids, device=logits.device)
                score = logits.log_softmax(-1).gather(-1, targets[:, None]).sum().item()
                log_scores.append(score)
                token_counts.append(len(answer_ids))
                if single_token_answers:
                    log_scores = [float(first_logits[tokens[0]]) for tokens in token_sequences]
                    token_counts = [1] * len(labels)
                    break
        values = normalize_log_scores(log_scores)
        first_scores = [float(first_logits[tokens[0]]) for tokens in token_sequences]
        first_values = normalize_log_scores(first_scores)
        probabilities = {o["id"]: p for o, p in zip(record["options"], values)}
        first = {o["id"]: p for o, p in zip(record["options"], first_values)}
        return {
            "probabilities": validate_probabilities(probabilities, record),
            "first_token_probabilities": validate_probabilities(first, record),
            "raw_response": {
                "answer_strings": labels,
                "answer_token_ids": token_sequences,
                "sequence_log_probabilities": log_scores,
                "first_token_log_probabilities": first_scores,
                **reasoning,
            },
            "usage": {
                "input_tokens": original_input_tokens
                if original_input_tokens is not None
                else len(prefix_ids),
                "processed_prompt_tokens": len(prefix_ids),
                "output_tokens": reasoning["thinking_tokens"],
                "scored_answer_tokens": sum(token_counts),
            },
            "elapsed_seconds": time.perf_counter() - start,
            "native_task_type": "sequence_scoring",
        }
