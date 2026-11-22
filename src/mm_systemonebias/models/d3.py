"""Official d3 prompting, non-causal encoder, and candidate-code readout."""

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
from .hf_loading import input_device, load_kwargs, move_inputs
from .resources import require_model


def context_counter(config):
    from .context_tokens import qwen_length

    path = require_model(config)
    reference = load_release_module(path, "d3_runtime.py")
    saved = json.loads((path / "decision_config.json").read_text())
    processor = reference.load_processor(path)

    def count(record):
        question = reference.normalize_question(native_question(record))
        codes = saved["codes"][: len(record["options"])]
        mapping = config.get("code_mapping", "identity")
        if mapping == "reverse":
            codes = codes[::-1]
        elif mapping == "rotate":
            codes = codes[1:] + codes[:1]
        messages = reference.image_messages(
            saved["prompt"], question_state(record), question.rendered, codes, len(record["images"])
        )
        text = processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True, enable_thinking=False
        )
        return qwen_length(processor, text, record["images"])

    return count


class D3Predictor:
    def __init__(self, config: dict):
        import torch
        from safetensors.torch import load_file

        self.config = config
        path = require_model(config)
        self.reference = load_release_module(path, "d3_runtime.py")
        self.saved = json.loads((path / "decision_config.json").read_text())
        cls = self.reference.backbone_class(self.reference.backbone_type(path))
        self.backbone = cls.from_pretrained(path, **load_kwargs(config)).eval()
        self.torch = torch
        self.processor = self.reference.load_processor(path)
        self.tokenizer = self.processor.tokenizer
        self.tokenizer.padding_side = "right"
        if not self.reference.linearize_patch_embed(self.backbone):
            raise RuntimeError("d3 vision patch embedding could not be configured")
        self.encoder = (
            self.reference.PromptEncoder(self.backbone.language_model)
            if self.saved["attention_mode"] == "noncausal_full_attention"
            else None
        )
        dtype = getattr(torch, self.saved.get("readout_dtype", "float32"))
        self.readout = load_file(path / "readout.safetensors")["weight"].to(
            input_device(self.backbone), dtype
        )
        self.codes = self.saved["codes"]
        expected_codes, token_ids = self.reference.answer_codes(self.tokenizer)
        if expected_codes != self.codes or token_ids != self.saved["token_ids"]:
            raise ValueError("d3 code vocabulary differs from the fixed checkpoint")

    def predict(self, record: dict) -> dict:
        return self.predict_batch([record])[0]

    def predict_batch(self, records: list[dict]) -> list[dict]:
        import torch

        start = time.perf_counter()
        questions, texts, images, mappings, code_lists = [], [], [], [], []
        mapping = self.config.get("code_mapping", "identity")
        for record in records:
            question = self.reference.normalize_question(native_question(record))
            indices = list(range(len(record["options"])))
            if mapping == "reverse":
                indices.reverse()
            elif mapping == "rotate":
                indices = indices[1:] + indices[:1]
            elif mapping != "identity":
                raise ValueError(f"Unknown d3 code mapping: {mapping}")
            codes = [self.codes[i] for i in indices]
            images.extend(self.reference.load_image(path) for path in record["images"])
            messages = self.reference.image_messages(
                self.saved["prompt"],
                question_state(record),
                question.rendered,
                codes,
                len(record["images"]),
            )
            texts.append(
                self.processor.apply_chat_template(
                    messages, tokenize=False, add_generation_prompt=True, enable_thinking=False
                )
            )
            questions.append(question)
            mappings.append(indices)
            code_lists.append(codes)
        inputs = self.processor(text=texts, images=images, padding=True, return_tensors="pt")
        length = inputs["input_ids"].shape[-1]
        if length > self.config["max_input_tokens"]:
            raise ValueError(
                f"d3 batch input exceeds token limit: {[r['sample_id'] for r in records]}"
            )
        lengths = inputs["attention_mask"].sum(-1).tolist()
        inputs = move_inputs(inputs, input_device(self.backbone))
        with torch.inference_mode():
            if self.encoder is not None:
                embeds, mrope = self.reference.D3.image_embeddings(self, inputs)
                hidden = self.encoder(embeds, inputs["attention_mask"], mrope)
            else:
                hidden = self.backbone(**inputs, use_cache=False).last_hidden_state
            positions = torch.tensor(lengths, device=hidden.device) - 1
            hidden = hidden[torch.arange(len(records), device=hidden.device), positions]
            hidden = hidden.to(self.readout.device)
            values = []
            for row, indices in zip(hidden, mappings, strict=True):
                weights = self.readout[indices]
                logits = (
                    row.float() @ weights.T
                    if weights.dtype == torch.float32
                    else torch.nn.functional.linear(row, weights).float()
                )
                values.append((logits / self.saved["temperature"]).softmax(-1).tolist())
        elapsed = (time.perf_counter() - start) / len(records)
        results = []
        for record, question, codes, scores, length in zip(
            records, questions, code_lists, values, lengths, strict=True
        ):
            answer = self.reference.product_answer(question, scores)
            results.append(
                {
                    "probabilities": validate_probabilities(
                        native_probabilities(record, answer), record
                    ),
                    "raw_response": {
                        "answer": answer,
                        "code_mapping": mapping,
                        "codes": codes,
                        "candidate_codes": {
                            option["id"]: code for option, code in zip(record["options"], codes)
                        },
                        "temperature": self.saved["temperature"],
                    },
                    "usage": {"input_tokens": length, "output_tokens": 0},
                    "elapsed_seconds": elapsed,
                    "inference_batch_size": len(records),
                    "native_task_type": question.kind,
                }
            )
        return results
