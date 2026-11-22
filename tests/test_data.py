import json
from pathlib import Path
from types import SimpleNamespace

from PIL import Image

from mm_systemonebias.data import dataset_identity, download_dataset, load_records
from mm_systemonebias.schema import record_id, record_key


def _packed_row(image_bytes: bytes, sample_id: str = "C1:sample") -> dict:
    base = {
        "sample_id": sample_id,
        "root_id": f"root-{sample_id}",
        "source_group": "group:one",
        "question_group": "C1",
        "scenario": "Image safety moderation",
        "task_type": "choice",
        "question": "Original?",
        "options": [
            {"id": "yes", "label": "A", "text": "Yes", "value": None, "image_index": 0},
            {"id": "no", "label": "B", "text": "No", "value": None, "image_index": None},
        ],
        "gold": "yes",
        "bias": None,
        "version": "orig",
        "intervention": "Original",
        "template": 1,
        "dataset": "Example",
        "evaluation_metadata": {},
        "metadata": json.dumps({"source_id": sample_id}),
        "input_image_indices": [0],
    }
    variant = {
        **base,
        "question": "Variant?",
        "bias": "B3",
        "version": "user_gold",
        "intervention": "User opinion supporting gold",
        "input_image_indices": [0],
    }
    return {
        **base,
        "variants": [variant],
        "images": [{"bytes": image_bytes, "path": None}],
        "image_names": ["source.png"],
    }


def test_loader_expands_a_mocked_packed_record_and_restores_image_order(tmp_path, monkeypatch):
    from io import BytesIO

    buffer = BytesIO()
    Image.new("RGB", (3, 2), (20, 30, 40)).save(buffer, format="PNG")
    image_bytes = buffer.getvalue()
    second_buffer = BytesIO()
    Image.new("RGB", (3, 2), (80, 90, 100)).save(second_buffer, format="PNG")
    second_bytes = second_buffer.getvalue()

    class FakeDataset(list):
        _fingerprint = "fixture-fingerprint"

        def cast_column(self, *_args):
            return self

    monkeypatch.setattr(
        "mm_systemonebias.data._load_packed",
        lambda *_args: FakeDataset(
            [_packed_row(image_bytes), _packed_row(second_bytes, "C1:second")]
        ),
    )
    monkeypatch.setattr("mm_systemonebias.data.DEFAULT_CACHE", tmp_path / "cache")

    records = load_records(dataset_path=tmp_path / "fixture")

    assert len(records) == 4
    assert [row["question"] for row in records] == [
        "Original?",
        "Variant?",
        "Original?",
        "Variant?",
    ]
    assert [option["id"] for option in records[0]["options"]] == ["yes", "no"]
    image_path = Path(records[0]["images"][0])
    assert image_path.read_bytes() == image_bytes
    second_image_path = Path(records[2]["images"][0])
    assert second_image_path.name.endswith("source.png")
    assert second_image_path != image_path
    assert second_image_path.read_bytes() == second_bytes
    assert records[0]["options"][0]["image"] == str(image_path)
    assert "image" not in records[0]["options"][1]
    with Image.open(image_path) as image:
        image.verify()
    assert record_key(records[1]) == ("C1", "C1:sample", "user_gold", 1)
    assert record_id(records[1]) == "C1/C1:sample/user_gold/template-1"


def test_remote_dataset_identity_loads_the_resolved_revision(monkeypatch):
    resolved_sha = "a1b2c3d4"
    calls = []

    class FakeApi:
        def dataset_info(self, repo_id, revision=None):
            assert repo_id == "owner/dataset"
            assert revision == "main"
            return SimpleNamespace(sha=resolved_sha)

    def fake_load(dataset_path, repo_id, revision):
        calls.append((dataset_path, repo_id, revision))
        return SimpleNamespace(_fingerprint="resolved-fingerprint")

    monkeypatch.setattr("huggingface_hub.HfApi", FakeApi)
    monkeypatch.setattr("mm_systemonebias.data._load_packed", fake_load)

    identity = dataset_identity(repo_id="owner/dataset", revision="main")

    assert calls == [(None, "owner/dataset", resolved_sha)]
    assert identity == f"hf:owner/dataset@{resolved_sha}:resolved-fingerprint"


def test_download_dataset_passes_allow_patterns_and_output_dir(tmp_path, monkeypatch):
    calls = []

    def fake_snapshot_download(**kwargs):
        calls.append(kwargs)

    monkeypatch.setattr("huggingface_hub.snapshot_download", fake_snapshot_download)
    destination = tmp_path / "download"
    patterns = ["test-00000.parquet", "README.md", "dataset_info.json"]

    result = download_dataset("owner/dataset", "revision-sha", destination, patterns)

    assert result == destination
    assert calls == [
        {
            "repo_id": "owner/dataset",
            "repo_type": "dataset",
            "revision": "revision-sha",
            "local_dir": destination,
            "allow_patterns": patterns,
        }
    ]
