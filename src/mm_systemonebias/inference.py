"""Run fixed model adapters and maintain resumable JSONL prediction runs."""

from __future__ import annotations
import argparse
import json
import os
import re
import time
from pathlib import Path
from urllib.parse import parse_qsl, urlencode, urlsplit, urlunsplit
import yaml
from tqdm import tqdm
from .models import build_predictor
from .data import dataset_identity, load_records
from .schema import record_key
from .models.common import validate_probabilities

_SECRET_KEYS = {"api_key", "token", "authorization", "api_key_env", "cookie"}


def _resolve(value):
    if isinstance(value, dict):
        return {k: _resolve(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_resolve(v) for v in value]
    if isinstance(value, str):

        def replace(match):
            return os.environ.get(match.group(1), "")

        return re.sub(r"\$\{([A-Za-z_][A-Za-z0-9_]*)\}", replace, value)
    return value


def load_config(path: str, overrides=()) -> dict:
    config = _resolve(yaml.safe_load(Path(path).read_text()))
    for entry in overrides:
        key, raw = entry.split("=", 1)
        cursor = config
        parts = key.split(".")
        for part in parts[:-1]:
            cursor = cursor.setdefault(part, {})
        cursor[parts[-1]] = _resolve(yaml.safe_load(raw))
    if not config.get("id") or not config.get("family"):
        raise ValueError("Model YAML requires id and family")
    family = config["family"]
    if family in {"typesafe", "openai_compatible"}:
        missing = [name for name in ("base_url", "api_key", "model") if not config.get(name)]
        if missing:
            raise ValueError(f"Missing API model settings: {', '.join(missing)}")
    if family == "autotrust" and not config.get("base_url"):
        raise ValueError("AutoTrust requires a local service base_url")
    if family not in {"typesafe", "openai_compatible", "autotrust"}:
        if not config.get("source") or len(config.get("revision", "")) != 40:
            raise ValueError(f"Model YAML requires a fixed Hugging Face revision: {config['id']}")
    if int(config.get("retries", 3)) < 1 or int(config.get("batch_size", 1)) < 1:
        raise ValueError("retries and batch_size must be positive")
    return config


def _manifest_config(config):
    sanitized = {}
    for key, value in config.items():
        name = key.lower()
        if name in _SECRET_KEYS or name.endswith(("_api_key", "_token", "_secret", "_password")):
            continue
        if name == "headers" and isinstance(value, dict):
            value = {
                header: item
                for header, item in value.items()
                if not any(
                    secret in header.lower()
                    for secret in ("authorization", "api-key", "token", "secret", "cookie")
                )
            }
        if name == "base_url" and isinstance(value, str):
            parsed = urlsplit(value)
            host = parsed.hostname or ""
            if parsed.port:
                host += f":{parsed.port}"
            query = [
                (
                    key,
                    "<redacted>"
                    if any(
                        secret in key.lower()
                        for secret in ("key", "token", "secret", "password", "auth")
                    )
                    else item,
                )
                for key, item in parse_qsl(parsed.query, keep_blank_values=True)
            ]
            value = urlunsplit((parsed.scheme, host, parsed.path, urlencode(query), ""))
        sanitized[key] = value
    return sanitized


def _root(record):
    return record.get("root_id") or record.get("metadata", {}).get("root_id") or record["sample_id"]


def select_records(records, cap):
    chosen = set()
    roots = {}
    for row in records:
        group = row.get("question_group") or "default"
        roots.setdefault(group, [])
        root = _root(row)
        if root not in roots[group]:
            roots[group].append(root)
    for group, values in roots.items():
        chosen.update((group, root) for root in values[:cap] if cap is not None)
    if cap is None:
        return records
    return [
        row for row in records if ((row.get("question_group") or "default"), _root(row)) in chosen
    ]


def _key(row):
    return json.dumps(
        list(
            row.get("record_key")
            or (
                row.get("question_group"),
                row["sample_id"],
                row.get("version"),
                row.get("template"),
            )
        ),
        ensure_ascii=False,
    )


def run_inference(
    config,
    records,
    output_dir,
    *,
    resume=False,
    retry=False,
    cap=None,
    dataset_revision=None,
    dataset_identity=None,
    question_groups=None,
    max_attempts=None,
):
    if cap is not None and cap < 1:
        raise ValueError("limit-originals-per-group must be at least 1")
    if max_attempts is not None and max_attempts < 1:
        raise ValueError("max_attempts must be at least 1")
    output = Path(output_dir).expanduser().resolve()
    output.mkdir(parents=True, exist_ok=True)
    excluded_roots = []
    roots_scanned = 0
    if question_groups:
        allowed = set(question_groups)
        records = [row for row in records if row.get("question_group") in allowed]
    if config["family"] == "jev_omni":
        supported = {}
        for row in records:
            key = (row.get("question_group"), _root(row))
            supported[key] = supported.get(key, True) and len(row.get("images", [])) == 1
        roots_scanned = len(supported)
        excluded_roots = [key for key, valid in supported.items() if not valid]
        records = [row for row in records if supported[(row.get("question_group"), _root(row))]]
    unsupported_by_group = {}
    for group, _root_id in excluded_roots:
        unsupported_by_group[group] = unsupported_by_group.get(group, 0) + 1
    records = select_records(records, cap)
    if not records:
        raise ValueError("No inference records selected; check the dataset and selection filters")
    selected_roots = {(row.get("question_group"), _root(row)) for row in records}
    model_path = output / "predictions.jsonl"
    manifest_path = output / "manifest.json"
    manifest = {
        "model_id": config["id"],
        "model_revision": config.get("revision"),
        "dataset_revision": dataset_revision,
        "dataset_identity": dataset_identity,
        "max_attempts": max_attempts,
        "selection": {
            "limit_originals_per_group": cap,
            "question_groups": sorted(question_groups) if question_groups else None,
            "original_roots_scanned": roots_scanned,
            "unsupported_roots_scanned": len(excluded_roots),
            "unsupported_roots_by_group": unsupported_by_group,
            "selected_original_roots": len(selected_roots),
            "record_keys": [list(record_key(row)) for row in records],
        },
        "inference_config": _manifest_config(config),
        "status": "running",
    }
    previous_attempts = 0
    if resume and not manifest_path.exists():
        raise FileNotFoundError(f"Cannot resume without a manifest: {manifest_path}")
    if resume and manifest_path.exists():
        previous = json.loads(manifest_path.read_text())
        previous_attempts = previous.get("attempts", 0)
        for name in (
            "model_id",
            "model_revision",
            "dataset_revision",
            "dataset_identity",
            "selection",
            "inference_config",
        ):
            if previous.get(name) != manifest.get(name):
                raise ValueError(f"Cannot resume: {name} differs from existing manifest")
    else:
        if model_path.exists() and not resume:
            raise FileExistsError(f"Prediction output already exists: {model_path}; use --resume")
        manifest_path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + "\n")
    done = {}
    if resume and model_path.exists():
        for line in model_path.read_text().splitlines():
            if line.strip():
                row = json.loads(line)
                done[_key(row)] = row
    pending = [
        r
        for r in records
        if _key(r) not in done or (retry and done[_key(r)].get("status") != "complete")
    ]
    if not pending and done:
        previous = json.loads(manifest_path.read_text())
        failures = sum(item.get("status") != "complete" for item in done.values())
        previous.update(
            status="partial" if failures else "complete",
            records=len(records),
            failures=failures,
            attempts=previous.get(
                "attempts", sum(item.get("attempts", 1) for item in done.values())
            ),
        )
        manifest_path.write_text(json.dumps(previous, indent=2, ensure_ascii=False) + "\n")
        print(f"status: {previous['status']}\nartifacts: {output}")
        return previous
    api_family = config["family"] in {"typesafe", "openai_compatible", "autotrust"}
    retry_limit = int(config.get("retries", 3)) if api_family else 1
    if retry_limit < 1:
        raise ValueError("retries must be at least 1")
    possible_attempts = previous_attempts + len(pending) * retry_limit
    if max_attempts is not None and possible_attempts > max_attempts:
        manifest["status"] = "failed"
        manifest["error"] = "AttemptBudgetExceeded"
        manifest["attempts"] = previous_attempts
        if resume:
            manifest["records"] = previous.get("records", len(records))
            manifest["failures"] = previous.get("failures")
        manifest_path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + "\n")
        raise ValueError(f"Run may exceed the attempt budget: {possible_attempts} > {max_attempts}")
    mode = "a" if resume else "w"
    try:
        predictor = build_predictor(config)
    except Exception as error:
        manifest["status"] = "failed"
        manifest["error"] = type(error).__name__
        manifest["attempts"] = previous_attempts
        if resume:
            manifest["records"] = previous.get("records", len(records))
            manifest["failures"] = previous.get("failures")
        manifest_path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + "\n")
        raise
    failures = 0
    actual_attempts = 0
    try:
        with model_path.open(mode, encoding="utf-8") as stream:
            batch_size = int(config.get("batch_size", 1))
            if batch_size < 1:
                raise ValueError("batch_size must be at least 1")
            if api_family:
                batch_size = 1
            batch_method = getattr(predictor, "predict_batch", None)
            if batch_size > 1 and callable(batch_method):
                offsets = range(0, len(pending), batch_size)
                for offset in tqdm(
                    offsets,
                    total=(len(pending) + batch_size - 1) // batch_size,
                    desc=config["id"],
                    unit="batch",
                ):
                    batch = pending[offset : offset + batch_size]
                    actual_attempts += len(batch)
                    results = batch_method(batch)
                    if len(results) != len(batch):
                        raise ValueError(
                            f"predict_batch returned {len(results)} results for {len(batch)} records"
                        )
                    rows = [
                        _prediction_row(config, record, result, 1)
                        for record, result in zip(batch, results, strict=True)
                    ]
                    for row in rows:
                        stream.write(json.dumps(row, ensure_ascii=False, allow_nan=False) + "\n")
                    stream.flush()
            else:
                for record in tqdm(pending, desc=config["id"], unit="record"):
                    attempts = int(config.get("retries", 3)) if api_family else 1
                    last_error = None
                    row = None
                    for attempt in range(attempts):
                        try:
                            actual_attempts += 1
                            result = predictor.predict(record)
                            row = _prediction_row(config, record, result, attempt + 1)
                            break
                        except Exception as error:
                            if not api_family or getattr(error, "fatal", False):
                                raise
                            last_error = error
                            if not _retryable(error) or attempt + 1 >= attempts:
                                break
                            time.sleep(min(2**attempt, 8))
                    if row is None:
                        failures += 1
                        row = {
                            **record,
                            "model_id": config["id"],
                            "prediction": None,
                            "probabilities": None,
                            "probability_status": "unavailable",
                            "probability_method": "none",
                            "raw_response": getattr(last_error, "raw_response", None),
                            "error": f"{type(last_error).__name__}: {last_error}",
                            "attempts": attempt + 1,
                            "status": "failed",
                        }
                    stream.write(json.dumps(row, ensure_ascii=False, allow_nan=False) + "\n")
                    stream.flush()
    except Exception as error:
        manifest["status"] = "failed"
        manifest["error"] = type(error).__name__
        manifest["attempts"] = previous_attempts + actual_attempts
        manifest_path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + "\n")
        raise
    finally:
        close = getattr(predictor, "close", None)
        if close:
            close()
    if resume:
        latest = dict(done)
        if model_path.exists():
            for line in model_path.read_text().splitlines():
                if line.strip():
                    item = json.loads(line)
                    latest[_key(item)] = item
        model_path.write_text(
            "".join(
                json.dumps(item, ensure_ascii=False, allow_nan=False) + "\n"
                for item in latest.values()
            )
        )
    failures = sum(
        1
        for item in (
            json.loads(line) for line in model_path.read_text().splitlines() if line.strip()
        )
        if item.get("status") != "complete"
    )
    manifest["status"] = "partial" if failures else "complete"
    manifest["records"] = len(records)
    manifest["failures"] = failures
    manifest["attempts"] = previous_attempts + actual_attempts
    manifest_path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + "\n")
    print(
        f"status: {manifest['status']}\noriginal roots: {len(selected_roots)}\ninference records: {len(records)}"
    )
    if roots_scanned:
        print(
            f"single-image eligibility: {roots_scanned} original roots screened; "
            f"{len(excluded_roots)} unsupported roots across {len(unsupported_by_group)} groups"
        )
    print(f"artifacts: {output}")
    return manifest


def _prediction_row(config, record, result, attempts):
    probabilities = result.get("probabilities")
    prediction = result.get("prediction")
    if probabilities is not None:
        probabilities = validate_probabilities(probabilities, record)
        prediction = max(probabilities, key=probabilities.__getitem__)
    legal_ids = {option["id"] for option in record["options"]}
    if prediction not in legal_ids:
        raise ValueError(f"Predictor returned an invalid option id for {record['sample_id']}")
    if probabilities is not None:
        probability_method = (
            "full_answer_sequence_log_probability"
            if config["family"] in {"qwen_vlm", "gemma_vlm", "lfm_vlm"}
            else "native_schema_distribution"
            if config["family"] == "typesafe"
            else "native_model_distribution"
        )
    else:
        probability_method = result.get("probability_method", "none")
    return {
        **record,
        "model_id": config["id"],
        "prediction": prediction,
        "probabilities": probabilities,
        "probability_status": result.get(
            "probability_status",
            "complete_distribution" if probabilities is not None else "unavailable",
        ),
        "probability_method": probability_method,
        "raw_response": result.get("raw_response"),
        "usage": result.get("usage", {}),
        "elapsed_seconds": result.get("elapsed_seconds"),
        "resolved_model": result.get("resolved_model"),
        "native_task_type": result.get("native_task_type"),
        "status": "complete",
        "attempts": attempts,
    }


def _retryable(error):
    import httpx

    if isinstance(error, (httpx.TimeoutException, httpx.NetworkError)):
        return True
    return bool(getattr(error, "retryable", False))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--dataset-path")
    parser.add_argument("--repo-id", default="ALIENS232/MM-SystemOneBias")
    parser.add_argument("--revision")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--limit-originals-per-group", type=int)
    parser.add_argument("--question-groups", help="Comma-separated groups to include")
    parser.add_argument("--max-attempts", type=int, help="Maximum API attempts including retries")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--retry", action="store_true")
    parser.add_argument("--set", dest="overrides", action="append", default=[])
    args = parser.parse_args()
    args.resume = args.resume or args.retry
    config = load_config(args.config, args.overrides)
    effective_revision = args.revision
    if args.dataset_path is None:
        from huggingface_hub import HfApi

        effective_revision = HfApi().dataset_info(args.repo_id, revision=args.revision).sha
    records = load_records(
        dataset_path=args.dataset_path, repo_id=args.repo_id, revision=effective_revision
    )
    identity = dataset_identity(args.dataset_path, args.repo_id, effective_revision)
    dataset_revision = identity
    manifest = run_inference(
        config,
        records,
        args.output_dir,
        resume=args.resume,
        retry=args.retry,
        cap=args.limit_originals_per_group,
        dataset_revision=dataset_revision,
        dataset_identity=identity,
        question_groups=[
            group.strip() for group in args.question_groups.split(",") if group.strip()
        ]
        if args.question_groups
        else None,
        max_attempts=args.max_attempts,
    )
    if manifest["status"] != "complete":
        raise SystemExit(2)


if __name__ == "__main__":
    main()
