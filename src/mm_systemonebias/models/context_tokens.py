"""Compute multimodal input token lengths using each processor's image geometry rules."""

from functools import lru_cache


@lru_cache(maxsize=None)
def image_size(path):
    from PIL import Image

    with Image.open(path) as image:
        width, height = image.size
        if image.format != "PNG" or "exif" in image.info:
            if image.getexif().get(274) in {5, 6, 7, 8}:
                width, height = height, width
    return height, width


def qwen_length(processor, text, images):
    text_length = len(processor.tokenizer.encode(text, add_special_tokens=False))
    visual = sum(
        processor.image_processor.get_number_of_image_patches(*image_size(path), {})
        // processor.image_processor.merge_size**2
        for path in images
    )
    return text_length + visual - len(images)


def build_context_counter(config):
    from importlib import import_module

    family = config["family"]
    if family in {"qwen_vlm", "qwen_vllm", "gemma_vlm", "lfm_vlm"}:
        return import_module("mm_systemonebias.models.vlm").context_counter(config)
    if family not in {"clef", "d1", "d3", "pplx", "intern_decision", "jev_omni", "autotrust"}:
        raise ValueError(f"No local native context encoder for {family}")
    return import_module(f"mm_systemonebias.models.{family}").context_counter(config)
