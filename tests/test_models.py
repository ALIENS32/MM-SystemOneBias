import sys

from mm_systemonebias.models import build_predictor
from mm_systemonebias.models.resources import model_directory


def test_core_model_imports_do_not_load_torch():
    sys.modules.pop("torch", None)
    assert callable(build_predictor)
    assert "torch" not in sys.modules


def test_model_path_defaults_outside_repository():
    path = model_directory(
        {"id": "qwen_4b", "revision": "abc", "resource_root": "~/.cache/mm-systemonebias/models"}
    )
    assert str(path).endswith("qwen_4b/abc")


def test_generic_openai_config_is_credential_indirected(monkeypatch):
    from mm_systemonebias.inference import load_config

    monkeypatch.setenv("OPENAI_BASE_URL", "https://api.example")
    monkeypatch.setenv("OPENAI_API_KEY", "test-key")
    monkeypatch.setenv("OPENAI_MODEL", "vision-model")
    config = load_config("configs/models/openai_compatible.yaml")
    assert config["api_key"] == "test-key"
    assert config["base_url"] == "https://api.example"
