"""Resolve and download fixed Hugging Face model revisions."""

from __future__ import annotations
import json
import os
from pathlib import Path


def model_directory(config: dict) -> Path:
    explicit = config.get("model_path") or os.environ.get(config.get("model_path_env", ""))
    if explicit:
        return Path(explicit).expanduser().resolve()
    return (
        Path(config.get("resource_root", "~/.cache/mm-systemonebias/models")).expanduser()
        / config["id"]
        / config["revision"]
    )


def require_model(config: dict) -> Path:
    path = model_directory(config)
    if not (path / "config.json").is_file():
        raise FileNotFoundError(
            f"Model resources missing at {path}; run scripts/download_model.sh --config configs/models/{config['id']}.yaml"
        )
    return path


def download(config: dict) -> Path:
    from huggingface_hub import snapshot_download

    if not config.get("source") or not config.get("revision") or len(config["revision"]) != 40:
        raise ValueError(f"A fixed model source and commit are required: {config.get('id')}")
    path = model_directory(config)
    snapshot_download(
        repo_id=config["source"],
        revision=config["revision"],
        local_dir=path,
        max_workers=config.get("download_workers", 4),
    )
    (path / "resource.json").write_text(
        json.dumps(
            {"model": config["id"], "source": config["source"], "revision": config["revision"]},
            indent=2,
        )
        + "\n"
    )
    print(f"artifacts: {path}")
    return path


def main():
    import argparse
    import yaml

    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    args = parser.parse_args()
    config = yaml.safe_load(Path(args.config).read_text(encoding="utf-8"))
    download(config)


if __name__ == "__main__":
    main()
