"""Fixed-revision multimodal model adapters."""


def build_predictor(config: dict):
    family = config["family"]
    if config.get("cpu_threads") and family not in {"typesafe", "openai_compatible", "autotrust"}:
        import torch

        torch.set_num_threads(int(config["cpu_threads"]))
    if family == "autotrust":
        from .autotrust import AutoTrustPredictor

        return AutoTrustPredictor(config)
    if family == "clef":
        from .clef import ClefPredictor

        return ClefPredictor(config)
    if family == "d3":
        from .d3 import D3Predictor

        return D3Predictor(config)
    if family == "d1":
        from .d1 import D1Predictor

        return D1Predictor(config)
    if family == "jev_omni":
        from .jev_omni import JevOmniPredictor

        return JevOmniPredictor(config)
    if family == "typesafe":
        from .typesafe import TypeSafePredictor

        return TypeSafePredictor(config)
    if family == "openai_compatible":
        from .openai_chat import OpenAIChatPredictor

        return OpenAIChatPredictor(config)
    if family in {"qwen_vlm", "lfm_vlm", "gemma_vlm"}:
        from .vlm import VLMPredictor

        return VLMPredictor(config)
    raise ValueError(f"Unknown model family: {family}")
