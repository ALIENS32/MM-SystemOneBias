"""Native candidate readout for Liquid d1."""

from __future__ import annotations

import importlib
import time

from .common import native_question, native_result, question_state
from .hf_loading import load_kwargs
from .resources import require_model


def context_counter(config):
    import json
    import math
    from types import SimpleNamespace
    from transformers import AutoProcessor
    from .common import load_release_module
    from .context_tokens import image_size

    path = require_model(config)
    processor = AutoProcessor.from_pretrained(path, trust_remote_code=True, local_files_only=True)
    runner = load_release_module(path, "runner.py")
    prompt = load_release_module(path, "prompt.py")
    model_type = json.loads((path / "config.json").read_text())["model_type"]
    markup = {}

    def count(record):
        size = len(record["images"])
        if size not in markup:
            markup[size] = runner.SystemOne._image_markup(
                SimpleNamespace(processor=processor), size
            )
        prefix = prompt.prefix_text(
            processor.tokenizer,
            question_state(record),
            processor.tokenizer.bos_token or "",
            prompt.DEFAULT_STATE_STYLE,
            prompt.DEFAULT_SYSTEM,
            markup[size],
        )
        text = prefix + prompt.suffix_text(
            processor.tokenizer,
            prompt.as_question(native_question(record)),
            prompt.default_lead(model_type),
            "desc",
        )
        replacements = []
        ip = processor.image_processor
        for path in record["images"]:
            height, width = image_size(path)
            if height * width > runner.VISION_MAX_PIXELS:
                scale = math.sqrt(runner.VISION_MAX_PIXELS / (height * width))
                height, width = max(1, int(height * scale)), max(1, int(width * scale))
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
            replacements.append(
                processor.replace_image_token(
                    {
                        "image_rows": [rows],
                        "image_cols": [cols],
                        "image_sizes": [[new_height, new_width]],
                    },
                    0,
                    use_image_special_tokens=True,
                )
            )
        parts = text.split(processor.image_token)
        if len(parts) != len(replacements) + 1:
            raise ValueError(f"Image placeholder count differs: {record['sample_id']}")
        expanded = (
            "".join(part + replacement for part, replacement in zip(parts, replacements))
            + parts[-1]
        )
        return len(processor.tokenizer.encode(expanded, add_special_tokens=False))

    return count


class D1Predictor:
    def __init__(self, config: dict):
        from transformers import AutoModel

        self.config = config
        path = require_model(config)
        self.model = AutoModel.from_pretrained(
            path, trust_remote_code=True, **load_kwargs(config)
        ).eval()
        self.engine = self.model.engine
        package = self.engine.__class__.__module__.rsplit(".", 1)[0]
        self.prompt = importlib.import_module(package + ".prompt")
        self.api = importlib.import_module(package + ".api")
        self.runner = importlib.import_module(package + ".runner")
        self.engine.tokenizer.padding_side = "right"
        self.engine._image_markup(1)
        self.engine.processor.tokenizer.padding_side = "right"

    def predict(self, record: dict) -> dict:
        return self.predict_batch([record])[0]

    def predict_batch(self, records: list[dict]) -> list[dict]:
        from PIL import Image
        import torch

        start = time.perf_counter()
        texts, pictures, questions = [], [], []
        for record in records:
            images = [
                self.runner.cap_pixels(Image.open(path).convert("RGB")) for path in record["images"]
            ]
            question = self.prompt.as_question(native_question(record))
            prefix = self.prompt.prefix_text(
                self.engine.tokenizer,
                question_state(record),
                self.engine.bos,
                self.engine.state_style,
                self.engine.system,
                self.engine._image_markup(len(images)) if images else "",
            )
            texts.append(
                prefix
                + self.prompt.suffix_text(
                    self.engine.tokenizer, question, self.engine.lead, self.engine.option_style
                )
            )
            pictures.append(images)
            questions.append(question)
        inputs = self.engine.processor(
            text=texts, images=pictures, padding=True, return_tensors="pt", add_special_tokens=False
        )
        if "pixel_attention_mask" in inputs:
            patches = int(inputs["pixel_attention_mask"].sum(1).max())
            inputs["pixel_values"] = inputs["pixel_values"][:, :patches]
            inputs["pixel_attention_mask"] = inputs["pixel_attention_mask"][:, :patches]
        lengths = inputs["attention_mask"].sum(-1).tolist()
        inputs = {
            key: value.to(self.engine.device) if hasattr(value, "to") else value
            for key, value in inputs.items()
        }
        length = inputs["input_ids"].shape[-1]
        if length > self.config["max_input_tokens"]:
            raise ValueError(
                f"d1 batch input exceeds token limit: {[r['sample_id'] for r in records]}"
            )
        with torch.inference_mode():
            positions = torch.tensor(lengths, device=self.engine.device) - 1
            outputs = self.engine._one_pass(**inputs, logits_to_keep=positions).logits
            rows = torch.arange(len(records), device=outputs.device)
            logits = outputs[rows, rows].float()
        results = []
        elapsed = (time.perf_counter() - start) / len(records)
        for record, q, row, length in zip(records, questions, logits, lengths, strict=True):
            probabilities = self.engine._readout(q, row - torch.logsumexp(row, dim=-1))
            raw = {
                "answers": {"answer": self.api.answer(q, probabilities)},
                "usage": {"input_tokens": length, "output_tokens": 0},
            }
            result = native_result(
                record, raw["answers"]["answer"], raw, elapsed, native_question(record)["type"]
            )
            result["inference_batch_size"] = len(records)
            results.append(result)
        return results
