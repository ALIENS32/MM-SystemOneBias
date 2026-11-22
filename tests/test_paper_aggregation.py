"""Independent acceptance cases for paper-level prediction aggregation."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from mm_systemonebias.metrics import aggregate_predictions, compare_pair
from mm_systemonebias.metrics.cli import main as metrics_main

MODEL = "qwen_27b"


def _record(
    model: str,
    sample: str,
    *,
    qgroup: str = "C1",
    scenario: str = "Safety moderation",
    version: str = "orig",
    bias: str | None = None,
    prediction: str = "yes",
    gold: str = "yes",
    task: str = "choice",
    template: int = 1,
    options: list[dict] | None = None,
    metadata: dict | None = None,
    probabilities: dict | None = None,
) -> dict:
    return {
        "model_id": model,
        "question_group": qgroup,
        "scenario": scenario,
        "sample_id": sample,
        "template": template,
        "task_type": task,
        "options": options
        or [
            {"id": "yes", "label": "Yes", "text": "Yes"},
            {"id": "no", "label": "No", "text": "No"},
        ],
        "gold": gold,
        "prediction": prediction,
        "probabilities": probabilities,
        "bias": bias,
        "version": version,
        "metadata": metadata or {},
    }


def _macro(rows, *, model=MODEL, bias, condition):
    return next(
        row
        for row in rows
        if row.get("kind") == "condition_macro"
        and row.get("model_id") == model
        and row.get("bias") == bias
        and row.get("condition") == condition
        and not row.get("exploratory", False)
    )


def test_b3_wrong_sources_macro_by_scenario_not_question_group_or_sample_count():
    records = []
    wrong_sources = ("user_wrong", "hist_wrong", "tool_wrong", "userstrong_wrong")
    interfaces = (
        ("C1", "Safety moderation", 3, False),
        ("C1-C", "Safety moderation", 1, True),
        ("C2", "Industrial anomaly detection", 1, True),
    )
    for version in wrong_sources:
        for qgroup, scenario, count, remains_correct in interfaces:
            for i in range(count):
                sample = f"{qgroup}-{i}"
                records.append(_record(MODEL, sample, qgroup=qgroup, scenario=scenario))
                records.append(
                    _record(
                        MODEL,
                        sample,
                        qgroup=qgroup,
                        scenario=scenario,
                        version=version,
                        bias="B3",
                        prediction="yes" if remains_correct else "no",
                    )
                )

    summaries, _ = aggregate_predictions(records)
    wrong = _macro(summaries, bias="B3", condition="wrong")
    # C1 and C1-C are one scenario mean; that scenario and C2 then receive equal weight.
    assert wrong["delta_acc"] == pytest.approx(-0.25)
    assert wrong["question_group_count"] == 3
    source_versions = {"user_wrong", "hist_wrong", "tool_wrong", "userstrong_wrong"}
    raw_rows = [
        row
        for row in summaries
        if row.get("model_id") == MODEL
        and row.get("bias") == "B3"
        and row.get("raw_version") in source_versions
    ]
    assert {row["raw_version"] for row in raw_rows} == source_versions


def test_b1_unequal_raw_version_counts_use_version_macro_and_joint_complete_coverage():
    records = []
    four_options = [
        {"id": key, "label": label, "text": label}
        for key, label in (("yes", "A"), ("no", "B"), ("other1", "C"), ("other2", "D"))
    ]
    orderings = {
        "perm1": ("yes", "other1", "no", "other2"),
        "perm2": ("other1", "no", "other2", "yes"),
        "perm3": ("no", "other2", "yes", "other1"),
    }
    versions_by_sample = {
        "complete": {"perm1": "yes", "perm2": "no", "perm3": "no"},
        "perm1-only-a": {"perm1": "yes"},
        "perm1-only-b": {"perm1": "yes"},
    }
    for sample, versions in versions_by_sample.items():
        records.append(_record(MODEL, sample, options=four_options))
        for version, prediction in versions.items():
            permuted_options = [
                next(option for option in four_options if option["id"] == key)
                for key in orderings[version]
            ]
            records.append(
                _record(
                    MODEL,
                    sample,
                    version=version,
                    bias="B1",
                    prediction=prediction,
                    options=permuted_options,
                )
            )

    summaries, _ = aggregate_predictions(records)
    order = _macro(summaries, bias="B1", condition="all_orders")
    assert order["flip_rate"] == pytest.approx(2 / 3)
    assert order["n"] == 5
    assert order["version_count"] == 3
    joint = next(
        row
        for row in summaries
        if row.get("kind") == "joint_consistency"
        and row.get("model_id") == MODEL
        and row.get("bias") == "B1"
    )
    assert joint["n_complete"] == 1
    assert joint["joint_consistency"] == pytest.approx(0.0)


def test_matched_backbones_intersect_raw_versions_and_weight_decision_models_then_backbones():
    records = []

    def add_matched(
        model: str,
        backbone: str,
        sample: str,
        qgroup: str,
        model_pred: str,
        backbone_pred: str,
        version: str = "perm1",
    ):
        records.extend(
            [
                _record(model, sample, qgroup=qgroup),
                _record(
                    model, sample, qgroup=qgroup, version=version, bias="B1", prediction=model_pred
                ),
                _record(backbone, sample, qgroup=qgroup),
                _record(
                    backbone,
                    sample,
                    qgroup=qgroup,
                    version=version,
                    bias="B1",
                    prediction=backbone_pred,
                ),
            ]
        )

    # Two decision models share qwen_27b. One has two scenario cells, the other one.
    add_matched("d3_27b", "qwen_27b", "a", "C1", "yes", "no")
    add_matched("d3_27b", "qwen_27b", "b", "C2", "yes", "no")
    add_matched("clef_27b", "qwen_27b", "c", "C1", "no", "no")
    # A second backbone contributes equally to the across-backbone mean.
    add_matched("d3_mini", "qwen_4b", "d", "C2", "yes", "no")
    # Same sample and models, different raw versions: this pair must not be matched.
    records.extend(
        [
            _record("d3_mini", "mismatch", qgroup="C1"),
            _record("d3_mini", "mismatch", qgroup="C1", version="perm1", bias="B1"),
            _record("qwen_4b", "mismatch", qgroup="C1"),
            _record("qwen_4b", "mismatch", qgroup="C1", version="perm2", bias="B1"),
            _record("d3_mini", "template-mismatch", qgroup="C1"),
            _record(
                "d3_mini", "template-mismatch", qgroup="C1", template=2, version="perm1", bias="B1"
            ),
            _record("qwen_4b", "template-mismatch", qgroup="C1"),
            _record("qwen_4b", "template-mismatch", qgroup="C1", version="perm1", bias="B1"),
        ]
    )

    _, matched = aggregate_predictions(records)
    details = [row for row in matched if row.get("kind") == "matched_detail"]
    assert not any(
        row.get("backbone_id") == "qwen_4b"
        and row.get("question_group") == "C1"
        and row.get("raw_version") == "perm1"
        for row in details
    )
    macro = next(
        row
        for row in matched
        if row.get("kind") == "matched_macro"
        and row.get("bias") == "option_order"
        and row.get("condition") == "all_orders"
    )
    assert macro["backbone_count"] == 2
    # qwen_27b: equal model mean=(1+0)/2; qwen_4b: 1; equal backbone mean=.75.
    assert macro["delta_acc_difference"] == pytest.approx(0.75)


def test_matched_order_macro_averages_versions_within_task_before_averaging_tasks():
    records = []
    model = "d3_27b"
    backbone = "qwen_27b"
    four_options = [
        {"id": key, "label": label, "text": label}
        for key, label in (("yes", "A"), ("no", "B"), ("other1", "C"), ("other2", "D"))
    ]
    permutations = {
        "perm1": ("no", "yes", "other1", "other2"),
        "perm2": ("other1", "no", "yes", "other2"),
        "perm3": ("other2", "other1", "no", "yes"),
    }
    for model_id in (model, backbone):
        records.append(
            _record(model_id, "four-choice", qgroup="C1", scenario="Safety", options=four_options)
        )
        for version, order in permutations.items():
            reordered = [
                next(option for option in four_options if option["id"] == key) for key in order
            ]
            records.append(
                _record(
                    model_id,
                    "four-choice",
                    qgroup="C1",
                    scenario="Safety",
                    version=version,
                    bias="B1",
                    prediction="no" if model_id == model else "yes",
                    options=reordered,
                )
            )

    two_options = [four_options[0], four_options[1]]
    swapped = list(reversed(two_options))
    for model_id in (model, backbone):
        records.extend(
            [
                _record(
                    model_id,
                    "two-choice",
                    qgroup="C2",
                    scenario="Anomaly detection",
                    options=two_options,
                ),
                _record(
                    model_id,
                    "two-choice",
                    qgroup="C2",
                    scenario="Anomaly detection",
                    version="swap",
                    bias="B1",
                    options=swapped,
                ),
            ]
        )

    _, matched = aggregate_predictions(records)
    macro = next(
        row
        for row in matched
        if row.get("kind") == "matched_macro"
        and row.get("bias") == "option_order"
        and row.get("condition") == "all_orders"
    )
    # Three four-option permutations collapse to the C1 mean first; C2 contributes equally.
    assert macro["delta_acc_difference"] == pytest.approx(-0.5)
    assert macro["flip_rate_difference"] == pytest.approx(0.5)


def test_matched_b6_uses_absolute_canonical_grade_shift_not_signed_cancellation():
    decision_model = "d3_mini"
    backbone = "qwen_4b"
    options = [
        {"id": f"level_{grade}", "label": str(grade), "text": f"quality {grade}", "value": grade}
        for grade in range(1, 6)
    ]
    versions = ("from0", "from6", "x10", "center")
    backbone_predictions = ("level_4", "level_2", "level_4", "level_2")
    records = []
    for model_id in (decision_model, backbone):
        records.append(
            _record(
                model_id,
                "absolute-scale",
                qgroup="J3",
                scenario="Image quality",
                task="score",
                prediction="level_3",
                gold="level_3",
                options=options,
            )
        )
        predictions = ("level_3",) * 4 if model_id == decision_model else backbone_predictions
        for version, prediction in zip(versions, predictions):
            records.append(
                _record(
                    model_id,
                    "absolute-scale",
                    qgroup="J3",
                    scenario="Image quality",
                    task="score",
                    version=version,
                    bias="B6",
                    prediction=prediction,
                    gold="level_3",
                    options=options,
                )
            )

    _, matched = aggregate_predictions(records)
    score_macro = next(
        row
        for row in matched
        if row.get("kind") == "matched_macro"
        and row.get("bias") == "rating_scale"
        and row.get("condition") == "all_scales"
    )
    # The backbone shifts +1,-1,+1,-1: signed mean is zero, absolute normalized shift is 1/4.
    assert score_macro["normalized_absolute_grade_shift_difference"] == pytest.approx(-0.25)
    assert score_macro["delta_acc_difference"] is None


def test_cli_reports_rating_correlations_and_complete_five_scale_consistency(tmp_path: Path):
    records = []
    grades = (1, 2, 4, 5)
    mos_values = (0.1, 0.4, 0.7, 0.9)
    scale_labels = {
        "orig": ("0", "1", "2", "3", "4"),
        "from0": ("0", "1", "2", "3", "4"),
        "from6": ("6", "7", "8", "9", "10"),
        "x10": ("10", "20", "30", "40", "50"),
        "center": ("-2", "-1", "0", "1", "2"),
    }
    versions = ("from0", "from6", "x10", "center")
    for i, (grade, mos) in enumerate(zip(grades, mos_values)):
        options = [
            {
                "id": f"level_{value}",
                "label": labels[value - 1],
                "text": f"quality {value}",
                "value": value,
            }
            for value in range(1, 6)
            for labels in [scale_labels["orig"]]
        ]
        original = _record(
            MODEL,
            f"score-{i}",
            qgroup="J3",
            scenario="Image quality",
            prediction=f"level_{grade}",
            gold=f"level_{grade}",
            task="score",
            options=options,
            metadata={"mos_quality": mos},
        )
        records.append(original)
        for version in versions:
            variant_options = [
                {**option, "label": scale_labels[version][j]} for j, option in enumerate(options)
            ]
            records.append(
                _record(
                    MODEL,
                    f"score-{i}",
                    qgroup="J3",
                    scenario="Image quality",
                    version=version,
                    bias="B6",
                    prediction=f"level_{grade}",
                    gold=f"level_{grade}",
                    task="score",
                    options=variant_options,
                    metadata={"mos_quality": mos},
                )
            )

    source = tmp_path / "predictions.jsonl"
    source.write_text("".join(json.dumps(row) + "\n" for row in records), encoding="utf-8")
    out = tmp_path / "metrics"
    metrics_main(["--predictions", str(source), "--output-dir", str(out)])
    payload = json.loads((out / "summary.json").read_text(encoding="utf-8"))
    rating = [row for row in payload["conditions"] if row.get("kind") == "rating_metric"]
    original_rho = next(row for row in rating if row.get("condition") == "original_spearman_mos")
    assert original_rho["n"] == 4
    assert original_rho["spearman_mos"] == pytest.approx(1.0)
    per_scale = {
        row.get("raw_version"): row for row in rating if row.get("raw_version") in versions
    }
    assert set(per_scale) == set(versions)
    assert all(row["spearman_mos"] == pytest.approx(1.0) for row in per_scale.values())
    complete = next(
        row for row in rating if row.get("condition") == "complete_five_scale_agreement"
    )
    assert complete["n"] == 4
    assert complete["all_scale_consistency"] == pytest.approx(1.0)
    assert complete["icc_a1"] == pytest.approx(1.0)


def test_custom_model_without_probabilities_keeps_decision_metrics():
    records = [
        _record("custom_closed_api", "one", prediction="yes", probabilities=None),
        _record(
            "custom_closed_api",
            "one",
            version="user_wrong",
            bias="B3",
            prediction="no",
            probabilities=None,
        ),
    ]
    summaries, _ = aggregate_predictions(records)
    row = next(
        row
        for row in summaries
        if row.get("kind") == "condition_macro"
        and row.get("model_id") == "custom_closed_api"
        and row.get("bias") == "B3"
    )
    assert row["n"] == 1
    assert row["n_probability"] == 0
    assert row["flip_rate"] == pytest.approx(1.0)
    assert row["delta_acc"] == pytest.approx(-1.0)


def test_exploratory_b4_and_multitarget_context_are_not_misreported_as_single_target_following():
    records = [
        _record(MODEL, "explore", qgroup="E3", scenario="Grasping stage"),
        _record(
            MODEL,
            "explore",
            qgroup="E3",
            scenario="Grasping stage",
            version="conflict",
            bias="B4",
            prediction="no",
        ),
    ]
    summaries, _ = aggregate_predictions(records)
    exploratory = next(
        row
        for row in summaries
        if row.get("kind") == "condition_macro"
        and row.get("bias") == "B4"
        and row.get("condition") == "conflict"
        and row.get("exploratory") is True
    )
    assert exploratory["exploratory"] is True
    main = next(
        row for row in summaries if row.get("kind") == "main_macro" and row.get("model_id") == MODEL
    )
    assert main["component_count"] == 0

    options = [{"id": key, "label": key, "text": key} for key in ("a", "b", "c")]
    original = _record(
        MODEL,
        "multi-target",
        qgroup="G1",
        scenario="GUI next action",
        prediction="a",
        gold="a",
        options=options,
        probabilities={"a": 0.8, "b": 0.1, "c": 0.1},
    )
    intervened = _record(
        MODEL,
        "multi-target",
        qgroup="G1",
        scenario="GUI next action",
        version="user_wrong",
        bias="B3",
        prediction="a",
        gold="a",
        options=options,
        metadata={"target_ids": ["b", "c"]},
        probabilities={"a": 0.5, "b": 0.3, "c": 0.2},
    )
    pair = compare_pair(original, intervened)
    assert pair["n_probability"] == 1
    assert pair["target_prob_shift"] == pytest.approx(0.3)
    assert pair["target_follow_den"] == 0
