import json
import pytest

from mm_systemonebias.inference import run_inference, select_records
from mm_systemonebias.models.openai_chat import _first_token_probabilities, build_request


def record(sample, version="orig", group="q"):
    return {
        "sample_id": sample,
        "root_id": sample,
        "source_group": "instance-" + sample,
        "question_group": group,
        "task_type": "choice",
        "question": "Pick",
        "images": [],
        "options": [
            {"id": "x", "label": "A", "text": "Alpha"},
            {"id": "y", "label": "B", "text": "Beta"},
        ],
        "gold": "x",
        "metadata": {},
        "bias": None,
        "version": version,
        "template": 0,
        "record_key": (group, sample, version, 0),
    }


def test_selection_keeps_all_variants_of_limited_originals():
    rows = [record("one"), record("one", "changed"), record("two"), record("two", "changed")]
    selected = select_records(rows, 1)
    assert [r["sample_id"] for r in selected] == ["one", "one"]


def test_empty_selection_fails_instead_of_creating_complete_run(tmp_path, monkeypatch):
    monkeypatch.setattr(
        "mm_systemonebias.inference.build_predictor",
        lambda _: (_ for _ in ()).throw(AssertionError("empty runs must not load a model")),
    )
    with pytest.raises(ValueError, match="No inference records selected"):
        run_inference({"id": "fake", "family": "d3"}, [], tmp_path, dataset_revision="rev")


def test_multitoken_label_probability_requires_all_valid_labels():
    row = record("one")
    probs, status = _first_token_probabilities(
        row,
        {
            "logprobs": {
                "content": [
                    {
                        "token": "A",
                        "logprob": -0.1,
                        "top_logprobs": [
                            {"token": "A", "logprob": -0.1},
                            {"token": "B", "logprob": -1.0},
                        ],
                    }
                ]
            }
        },
    )
    assert probs == {"x": pytest.approx(0.7109495), "y": pytest.approx(0.2890505)}
    assert status == "complete_first_token_distribution"


def test_prediction_record_does_not_require_gold(tmp_path, monkeypatch):
    class Predictor:
        def predict(self, row):
            assert "gold" not in row
            return {"prediction": "x", "probabilities": {"x": 0.7, "y": 0.3}, "raw_response": {}}

    monkeypatch.setattr("mm_systemonebias.inference.build_predictor", lambda _: Predictor())
    row = record("one")
    row.pop("gold")
    run_inference(
        {"id": "fake", "family": "openai_compatible", "retries": 1},
        [row],
        tmp_path,
        dataset_revision="test",
    )
    result = json.loads((tmp_path / "predictions.jsonl").read_text())
    assert result["prediction"] == "x"
    assert result["probabilities"] == {"x": 0.7, "y": 0.3}
    assert "record" not in result


def test_resume_manifest_mismatch_fails(tmp_path, monkeypatch):
    class Predictor:
        def predict(self, row):
            return {"prediction": "x", "probabilities": {"x": 0.6, "y": 0.4}}

    monkeypatch.setattr("mm_systemonebias.inference.build_predictor", lambda _: Predictor())
    cfg = {"id": "fake", "family": "openai_compatible", "retries": 1}
    run_inference(cfg, [record("one")], tmp_path, dataset_revision="rev-a")
    with pytest.raises(ValueError, match="dataset_revision"):
        run_inference(cfg, [record("one")], tmp_path, resume=True, dataset_revision="rev-b")


def test_retry_replaces_only_incomplete_record(tmp_path, monkeypatch):
    calls = []

    class Predictor:
        def predict(self, row):
            calls.append(row["sample_id"])
            if row["sample_id"] == "one" and calls.count("one") == 1:
                raise RuntimeError("temporary")
            return {"probabilities": {"x": 0.6, "y": 0.4}}

    monkeypatch.setattr("mm_systemonebias.inference.build_predictor", lambda _: Predictor())
    cfg = {"id": "fake", "family": "openai_compatible", "retries": 1}
    run_inference(cfg, [record("one")], tmp_path, dataset_revision="rev")
    run_inference(cfg, [record("one")], tmp_path, resume=True, retry=True, dataset_revision="rev")
    rows = [json.loads(line) for line in (tmp_path / "predictions.jsonl").read_text().splitlines()]
    assert len(rows) == 1 and rows[0]["status"] == "complete" and rows[0]["prediction"] == "x"
    assert json.loads((tmp_path / "manifest.json").read_text())["attempts"] == 2


def test_prompt_never_serializes_gold():
    row = record("one")
    row["gold"] = "SECRET-GOLD"
    request = build_request(
        row,
        {
            "answer_instruction": "Return label",
            "image_detail": "high",
            "model": "test",
            "temperature": 0,
            "max_tokens": 10,
            "logprobs": False,
        },
    )
    assert "SECRET-GOLD" not in request["messages"][0]["content"][0]["text"]


def test_local_batch_predictor_and_complete_resume(tmp_path, monkeypatch):
    calls = []

    class Predictor:
        def predict_batch(self, rows):
            calls.append(len(rows))
            return [{"probabilities": {"x": 0.6, "y": 0.4}} for _ in rows]

    monkeypatch.setattr("mm_systemonebias.inference.build_predictor", lambda _: Predictor())
    config = {"id": "batch", "family": "d3", "batch_size": 2}
    rows = [record("one"), record("two"), record("three")]
    run_inference(config, rows, tmp_path, dataset_revision="rev")
    assert calls == [2, 1]
    monkeypatch.setattr(
        "mm_systemonebias.inference.build_predictor",
        lambda _: (_ for _ in ()).throw(AssertionError("complete resume should not load a model")),
    )
    result = run_inference(config, rows, tmp_path, resume=True, dataset_revision="rev")
    assert result["status"] == "complete"
    assert calls == [2, 1]


def test_api_budget_rejects_before_predictor_construction(tmp_path, monkeypatch):
    called = []
    monkeypatch.setattr("mm_systemonebias.inference.build_predictor", lambda _: called.append(True))
    with pytest.raises(ValueError, match="attempt budget"):
        run_inference(
            {"id": "api", "family": "typesafe", "retries": 3},
            [record("one")],
            tmp_path,
            dataset_revision="rev",
            max_attempts=2,
        )
    assert called == []


def test_manifest_redacts_api_credentials():
    from mm_systemonebias.inference import _manifest_config

    cleaned = _manifest_config(
        {
            "api_key": "secret",
            "access_token": "secret",
            "base_url": "https://user:pass@api.example/v1?api_key=secret",
            "headers": {"Authorization": "secret", "TypeSafe-Danger-Zone-Images": "true"},
            "max_tokens": 20,
        }
    )
    assert "api_key" not in cleaned and "access_token" not in cleaned
    assert cleaned["base_url"] == "https://api.example/v1?api_key=%3Credacted%3E"
    assert cleaned["headers"] == {"TypeSafe-Danger-Zone-Images": "true"}
    assert cleaned["max_tokens"] == 20


def test_transient_api_errors_retry_and_count_attempts(tmp_path, monkeypatch):
    import httpx

    calls = 0

    class Predictor:
        def predict(self, row):
            nonlocal calls
            calls += 1
            if calls == 1:
                raise httpx.TimeoutException("temporary timeout")
            return {"prediction": "x", "probabilities": None}

    monkeypatch.setattr("mm_systemonebias.inference.build_predictor", lambda _: Predictor())
    result = run_inference(
        {"id": "api", "family": "openai_compatible", "retries": 2},
        [record("one")],
        tmp_path,
        dataset_revision="rev",
    )
    assert result["status"] == "complete"
    assert result["attempts"] == 2
    assert calls == 2


def test_rejected_resumes_preserve_attempt_budget_history(tmp_path, monkeypatch):
    import httpx

    calls = 0

    class Predictor:
        def predict(self, row):
            nonlocal calls
            calls += 1
            if calls == 1:
                raise httpx.TimeoutException("temporary")
            return {"prediction": "x", "probabilities": None}

    monkeypatch.setattr("mm_systemonebias.inference.build_predictor", lambda _: Predictor())
    config = {"id": "api", "family": "typesafe", "retries": 1}
    rows = [record("one")]
    run_inference(config, rows, tmp_path, dataset_revision="rev", max_attempts=1)
    for _ in range(2):
        with pytest.raises(ValueError, match="attempt budget"):
            run_inference(
                config,
                rows,
                tmp_path,
                resume=True,
                retry=True,
                dataset_revision="rev",
                max_attempts=1,
            )
        manifest = json.loads((tmp_path / "manifest.json").read_text())
        assert manifest["attempts"] == 1
    result = run_inference(
        config,
        rows,
        tmp_path,
        resume=True,
        retry=True,
        dataset_revision="rev",
        max_attempts=2,
    )
    assert result["status"] == "complete"
    assert result["attempts"] == 2
    assert calls == 2
