"""Joint-schema head inference for Clef."""

from __future__ import annotations

import json
import time

from .common import (
    load_release_module,
    native_probabilities,
    native_question,
    question_state,
    validate_probabilities,
)
from .hf_loading import input_device, linearize_patch_embed, load_kwargs
from .resources import require_model


def context_counter(config):
    from transformers import AutoProcessor
    from .context_tokens import image_size

    path = require_model(config)
    processor = AutoProcessor.from_pretrained(path, local_files_only=True)
    reference = load_release_module(path, "joint_schema_model.py")

    def count(record):
        request = {
            "state": question_state(record),
            "questions": {"answer": native_question(record)},
        }
        text_length = len(
            reference.encode_record(processor.tokenizer, request, max_length=2**31 - 1).input_ids
        )
        visual = sum(
            processor.image_processor.get_number_of_image_patches(*image_size(path), {})
            // processor.image_processor.merge_size**2
            + 2
            for path in record["images"]
        )
        return text_length + visual + int(bool(record["images"]))

    return count


class ClefPredictor:
    def __init__(self, config: dict):
        import torch
        from safetensors.torch import load_file
        from transformers import AutoProcessor, Qwen3_5ForConditionalGeneration

        self.config = config
        path = require_model(config)
        self.reference = load_release_module(path, "joint_schema_model.py")
        self.model = Qwen3_5ForConditionalGeneration.from_pretrained(
            path, **load_kwargs(config)
        ).eval()
        if config["vision_patch_projection"] == "linear":
            if not linearize_patch_embed(self.model):
                raise ValueError("Clef checkpoint has no supported vision patch projection")
        elif config["vision_patch_projection"] != "conv3d":
            raise ValueError(
                f"Unknown vision patch projection: {config['vision_patch_projection']}"
            )
        self.processor = AutoProcessor.from_pretrained(path, local_files_only=True)
        self.head = self.reference.JointSchemaHead(
            **json.loads((path / "joint_head_config.json").read_text())
        )
        self.head.load_state_dict(load_file(path / "joint_head.safetensors"))
        self.head.to(
            device=self.model.get_output_embeddings().weight.device, dtype=torch.bfloat16
        ).eval()

    def predict(self, record: dict) -> dict:
        return self.predict_batch([record])[0]

    def predict_batch(self, records: list[dict]) -> list[dict]:
        import torch
        from PIL import Image

        start = time.perf_counter()
        requests, encodings = [], []
        for record in records:
            request = {
                "state": question_state(record),
                "questions": {"answer": native_question(record)},
                "images": [Image.open(path).convert("RGB") for path in record["images"]],
            }
            if self.config.get("max_image_pixels"):
                request["media_kwargs"] = {"max_pixels": self.config["max_image_pixels"]}
            encoded = self.reference.encode_record(
                self.processor.tokenizer, request, max_length=2**31 - 1, processor=self.processor
            )
            if len(encoded.input_ids) > self.config["max_input_tokens"]:
                raise ValueError(f"Clef input exceeds token limit: {record['sample_id']}")
            requests.append(request)
            encodings.append(encoded)
        batch = self.reference.collate_records(
            encodings, self.processor.tokenizer.pad_token_id, input_device(self.model)
        )
        with torch.inference_mode():
            outputs = self.model.model(
                input_ids=batch["input_ids"],
                attention_mask=batch["attention_mask"],
                use_cache=False,
                return_dict=True,
                **(batch.get("media") or {}),
            )
            device = next(self.head.parameters()).device
            logits = self.head(
                outputs.last_hidden_state.to(device),
                batch["input_ids"].to(device),
                batch["attention_mask"].to(device),
                batch["records"],
                self.model.get_output_embeddings().weight,
            )
            values = [item[0].float().softmax(-1).tolist() for item in logits]
        results = []
        elapsed = (time.perf_counter() - start) / len(records)
        for record, request, encoded, scores in zip(
            records, requests, encodings, values, strict=True
        ):
            probabilities = dict(zip(encoded.questions[0].option_ids, scores))
            answer = (
                {"noul": probabilities["true"]}
                if record["task_type"] == "predicate"
                else {"probabilities": probabilities}
            )
            results.append(
                {
                    "vision_patch_projection": self.config["vision_patch_projection"],
                    "probabilities": validate_probabilities(
                        native_probabilities(record, answer), record
                    ),
                    "raw_response": {
                        "probabilities": probabilities,
                        "native_option_order": list(encoded.questions[0].option_ids),
                    },
                    "usage": {"input_tokens": len(encoded.input_ids), "output_tokens": 0},
                    "elapsed_seconds": elapsed,
                    "inference_batch_size": len(records),
                    "native_task_type": request["questions"]["answer"]["type"],
                }
            )
        return results
