"""Single-image classification-head inference for Jev-Omni."""

from __future__ import annotations

import json
import time

from .common import load_release_module, option_text, question_state, validate_probabilities
from .hf_loading import input_device, load_kwargs
from .resources import require_model


def context_counter(config):
    from transformers import AutoProcessor
    from .context_tokens import image_size

    path = require_model(config)
    reference = load_release_module(path, "jev_omni.py")
    processor = AutoProcessor.from_pretrained(path, local_files_only=True)

    def count(record):
        content = [{"type": "image", "image": path} for path in record["images"]]
        content.append(
            {
                "type": "text",
                "text": reference._prompt(
                    question_state(record),
                    record["question"],
                    [option_text(o) for o in record["options"]],
                ),
            }
        )
        text = processor.apply_chat_template(
            [{"role": "user", "content": content}],
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=False,
        )
        visual = sum(
            processor._get_num_multimodal_tokens(image_sizes=[image_size(path)]).num_image_tokens[0]
            + 1
            for path in record["images"]
        )
        return len(processor.tokenizer.encode(text, add_special_tokens=False)) + visual

    return count


class JevOmniPredictor:
    def __init__(self, config: dict):
        import torch
        import transformers
        from transformers import AutoConfig, AutoProcessor

        path = require_model(config)
        reference = load_release_module(path, "jev_omni.py")
        model_config = AutoConfig.from_pretrained(path, local_files_only=True)
        model = (
            getattr(transformers, model_config.architectures[0])
            .from_pretrained(path, **load_kwargs(config))
            .eval()
        )
        device = str(input_device(model))
        decision = json.loads((path / "decision_config.json").read_text())
        head = reference._Head256(decision["hidden_size"]).to(device).eval()
        head.load_state_dict(torch.load(path / "head.pt", map_location=device, weights_only=True))
        head.register_forward_pre_hook(
            lambda module, args: tuple(value.to(device) for value in args)
        )
        _, decoder = reference._find_backbone(model)
        self.classifier = reference.JevOmni(
            model,
            head,
            AutoProcessor.from_pretrained(path, local_files_only=True),
            decoder,
            device=device,
        )
        self.classifier.processor.tokenizer.padding_side = "left"
        self.reference = reference
        self.config = config

    def predict_batch(self, records: list[dict]) -> list[dict]:
        import torch
        from PIL import Image

        start = time.perf_counter()
        conversations, option_lists = [], []
        for record in records:
            if len(record["images"]) != 1:
                raise ValueError(f"Jev-Omni requires one image: {record['sample_id']}")
            options = [option_text(o) for o in record["options"]]
            content = [
                {"type": "image", "image": Image.open(record["images"][0]).convert("RGB")},
                {
                    "type": "text",
                    "text": self.reference._prompt(
                        question_state(record), record["question"], options
                    ),
                },
            ]
            conversations.append([{"role": "user", "content": content}])
            option_lists.append(options)
        engine = self.classifier
        inputs = engine.processor.apply_chat_template(
            conversations,
            add_generation_prompt=True,
            tokenize=True,
            return_dict=True,
            return_tensors="pt",
            enable_thinking=False,
            padding=True,
        )
        if inputs["input_ids"].shape[-1] > self.config["max_input_tokens"]:
            raise ValueError("Jev-Omni batch input exceeds token limit")
        lengths = inputs["attention_mask"].sum(-1).tolist()
        inputs = {
            k: v.to(engine.device, dtype=torch.bfloat16)
            if torch.is_floating_point(v)
            else v.to(engine.device)
            for k, v in inputs.items()
        }
        engine._capture.clear()
        with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
            engine.model(**inputs, use_cache=False, **engine._extra)
            logits = engine.head(
                engine._capture["hidden"],
                torch.tensor([len(o) for o in option_lists], device=engine.device),
            )
            values = [
                row[: len(options)].softmax(-1).float().tolist()
                for row, options in zip(logits, option_lists, strict=True)
            ]
        elapsed = (time.perf_counter() - start) / len(records)
        results = []
        for record, options, scores, length in zip(
            records, option_lists, values, lengths, strict=True
        ):
            best = max(range(len(scores)), key=scores.__getitem__)
            raw = {
                "prediction": options[best],
                "prediction_index": best,
                "confidence": scores[best],
                "probabilities": dict(zip(options, scores)),
            }
            results.append(
                {
                    "probabilities": validate_probabilities(
                        {o["id"]: s for o, s in zip(record["options"], scores)}, record
                    ),
                    "raw_response": raw,
                    "usage": {"input_tokens": length, "output_tokens": 0},
                    "elapsed_seconds": elapsed,
                    "inference_batch_size": len(records),
                    "native_task_type": "choice",
                }
            )
        return results

    def predict(self, record: dict) -> dict:
        if len(record["images"]) != 1:
            raise ValueError(f"Jev-Omni requires one image: {record['sample_id']}")
        start = time.perf_counter()
        options = [option_text(o) for o in record["options"]]
        raw = self.classifier.predict(
            state=question_state(record),
            question=record["question"],
            options=options,
            media=record["images"][0],
            modality="image",
        )
        probabilities = {
            o["id"]: raw["probabilities"][text] for o, text in zip(record["options"], options)
        }
        return {
            "probabilities": validate_probabilities(probabilities, record),
            "raw_response": raw,
            "usage": {},
            "elapsed_seconds": time.perf_counter() - start,
            "native_task_type": "choice",
        }
