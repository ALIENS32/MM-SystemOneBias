"""Command line entry point for portable paper metric summaries."""

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path
from .core import aggregate_predictions


def _complete_probability(row):
    probs = row.get("probabilities")
    if not isinstance(probs, dict) or not probs:
        return False
    try:
        values = [float(value) for value in probs.values()]
        if any(value < 0 for value in values) or abs(sum(values) - 1) > 1e-5:
            return False
        options = row.get("options") or []
        legal = {
            str(option["id"]) for option in options if isinstance(option, dict) and "id" in option
        }
        return not legal or {str(key) for key in probs} == legal
    except (TypeError, ValueError, KeyError):
        return False


def _coverage(rows):
    by_model = defaultdict(list)
    for row in rows:
        by_model[str(row.get("model_id", "unknown"))].append(row)
    coverage = []
    for model_id, items in sorted(by_model.items()):
        originals = {
            (row.get("question_group"), row.get("sample_id"), row.get("template")): row
            for row in items
            if row.get("version") in ("orig", "original")
        }
        variants = [
            row
            for row in items
            if row.get("version") not in ("orig", "original") and row.get("bias")
        ]
        unpaired = sum(
            (row.get("question_group"), row.get("sample_id"), row.get("template")) not in originals
            for row in variants
        )
        valid_pairs = sum(
            (key := (row.get("question_group"), row.get("sample_id"), row.get("template")))
            in originals
            and row.get("prediction") is not None
            and originals[key].get("prediction") is not None
            for row in variants
        )
        failed = sum(
            row.get("prediction") is None
            and str(row.get("status", "")).lower() not in ("unsupported", "not_supported")
            for row in items
        )
        coverage.append(
            {
                "model_id": model_id,
                "observed_records": len(items),
                "valid_decisions": sum(
                    row.get("prediction") is not None and row.get("gold") is not None
                    for row in items
                ),
                "failed_or_missing_prediction": failed,
                "complete_probability_distributions": sum(
                    _complete_probability(row) for row in items
                ),
                "unpaired_variants": unpaired,
                "valid_pairs": valid_pairs,
            }
        )
    return coverage


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--predictions", nargs="+", required=True, help="Prediction JSONL file(s)")
    p.add_argument(
        "--output-dir", required=True, help="Directory for portable CSV and JSON summaries"
    )
    a = p.parse_args(argv)
    rows = []
    for filename in a.predictions:
        with open(filename, encoding="utf-8") as f:
            for line_no, line in enumerate(f, 1):
                if line.strip():
                    obj = json.loads(line)
                    if not isinstance(obj, dict):
                        raise ValueError(f"{filename}:{line_no}: expected JSON object")
                    if isinstance(obj.get("record"), dict):
                        pred = {k: v for k, v in obj.items() if k != "record"}
                        obj = {**obj["record"], **pred}
                    rows.append(obj)
    if not rows:
        raise ValueError("Prediction input is empty; refusing to create a complete summary")
    summary, matched = aggregate_predictions(rows)
    out = Path(a.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    coverage = _coverage(rows)
    status = (
        "partial"
        if any(r["failed_or_missing_prediction"] or r["unpaired_variants"] for r in coverage)
        else "complete"
    )
    payload = {
        "schema_version": 1,
        "status": status,
        "input_files": [str(Path(x).resolve()) for x in a.predictions],
        "coverage": coverage,
        "conditions": summary,
        "matched_backbones": matched,
    }
    (out / "summary.json").write_text(
        json.dumps(payload, indent=2, allow_nan=False) + "\n", encoding="utf-8"
    )

    def write_csv(name, items):
        if not items:
            (out / name).write_text("", encoding="utf-8")
            return
        fields = list(dict.fromkeys(k for x in items for k in x))
        with (out / name).open("w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=fields)
            w.writeheader()
            w.writerows(items)

    write_csv("conditions.csv", summary)
    write_csv("matched_backbones.csv", matched)
    pair_count = sum(row["valid_pairs"] for row in coverage)
    print(f"status: {status}; pairs: {pair_count}; artifacts: {out}")
    return 2 if status == "partial" else 0


if __name__ == "__main__":
    raise SystemExit(main())
