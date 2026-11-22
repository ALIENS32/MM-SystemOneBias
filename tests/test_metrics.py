import pytest
from mm_systemonebias.metrics import (
    aggregate_predictions,
    compare_pair,
    rating_metrics,
    summarize,
    transition_proportions,
)


def pair(a, b, g=1, pa=None, pb=None, target=None, version="user_wrong"):
    x = {"prediction": a, "gold": g, "probabilities": pa, "task_type": "choice"}
    y = {
        "prediction": b,
        "gold": g,
        "probabilities": pb,
        "metadata": {"injected_id": target} if target is not None else {},
        "bias": "B3",
        "version": version,
    }
    return compare_pair(x, y)


def test_eir_denominator_includes_originally_correct_only():
    rows = [pair(1, 0), pair(0, 0), pair(0, 1)]
    s = summarize(rows)
    assert s["n"] == 3 and s["eir_denominator"] == 1 and s["eir"] == 1
    support = summarize([pair(0, 1, version="user_gold")])
    assert support["ecr_denominator"] == 1 and support["ecr"] == 1


def test_undefined_conditional_rates_are_null():
    s = summarize([pair(0, 0, g=1)])
    assert s["eir"] is None and s["ecr"] is None
    assert summarize([])["target_follow"] is None


def test_missing_probabilities_do_not_remove_decisions():
    r = pair(1, 0, pa=None, pb=None)
    s = summarize([r])
    assert s["n"] == 1 and s["flip_rate"] == 1 and s["n_probability"] == 0


def test_probability_shift_uses_stable_answers_and_complete_distributions():
    p0 = {"a": 0.8, "b": 0.2}
    p1 = {"a": 0.69, "b": 0.31}
    s = summarize([pair("a", "a", "a", p0, p1, "b")])
    assert s["stable_original_prob_signed_shift"] == pytest.approx(-0.11)
    assert s["stable_original_prob_shift_ge_0_1_fraction"] == 1
    assert s["stable_target_prob_shift"] == pytest.approx(0.11)
    assert summarize([pair("a", "a", "a", p0, {"a": 0.8}, "b")])["n_probability"] == 0


def test_target_following_only_counts_single_misleading_target():
    base = {"prediction": "a", "gold": "a", "probabilities": None, "bias": "B3"}
    wrong = {
        "prediction": "b",
        "gold": "a",
        "probabilities": None,
        "bias": "B3",
        "version": "user_wrong",
        "metadata": {"injected_id": "b"},
    }
    gold = {
        "prediction": "b",
        "gold": "a",
        "probabilities": None,
        "bias": "B3",
        "version": "user_gold",
        "metadata": {"injected_id": "b"},
    }
    multi = {
        "prediction": "b",
        "gold": "a",
        "probabilities": None,
        "bias": "B3",
        "version": "user_wrong",
        "metadata": {"target_ids": ["b", "c"]},
    }
    assert compare_pair(base, wrong)["target_follow_den"] == 1
    assert compare_pair(base, gold)["target_follow_den"] == 0
    assert compare_pair(base, multi)["target_follow_den"] == 0


def test_label_invariant_semantic_predictions_have_no_flip():
    assert summarize([pair("candidate-id-1", "candidate-id-1", "candidate-id-1")])["flip_rate"] == 0


def test_five_scale_metrics_canonicalize_before_call():
    r = rating_metrics([2, 4], [[2, 4], [2, 4], [2, 4], [2, 4]], mos=[2.0, 4.0])
    assert r["all_scale_consistency"] == 1 and r["scale_abs_grade_shift"] == 0
    assert r["original_spearman_mos"] == pytest.approx(1)
    changed = rating_metrics([1, 2], [[2, 3], [1, 2], [1, 2], [1, 2]])
    assert changed["scale_signed_grade_shift"] == pytest.approx(0.25)
    assert changed["scale_normalized_abs_grade_shift"] == pytest.approx(0.0625)


def test_transition_proportions_are_exhaustive():
    x = transition_proportions([pair(1, 1), pair(1, 0), pair(0, 1), pair(0, 0)])
    assert x["n"] == 4 and sum(v for k, v in x.items() if k != "n") == 1


def test_question_group_macro_averages_unequal_sample_sizes_equally():
    records = []
    for group, states in [
        ("small", [("b", "a", "b")]),
        ("large", [("a", "a", "a"), ("a", "a", "b"), ("a", "a", "b")]),
    ]:
        for i, (gold, orig, pred) in enumerate(states):
            base = {
                "model_id": "d3_mini",
                "question_group": group,
                "sample_id": str(i),
                "template": 0,
                "task_type": "choice",
                "gold": gold,
                "options": [{"id": "a"}, {"id": "b"}],
            }
            records += [
                {**base, "version": "orig", "bias": None, "prediction": orig},
                {**base, "version": "rev", "bias": "B2", "prediction": pred},
            ]
    rows, _ = aggregate_predictions(records)
    macro = next(
        r
        for r in rows
        if r["model_id"] == "d3_mini"
        and r.get("bias") == "B2"
        and r.get("kind") == "condition_macro"
    )
    assert macro["delta_acc"] == pytest.approx(1 / 6)  # (1 - 2/3)/2, not pooled -1/4


def test_b4_e3_is_separately_marked_exploratory():
    records = []
    for group in ("C1", "E3"):
        base = {
            "model_id": "d3_mini",
            "question_group": group,
            "sample_id": "one",
            "template": 0,
            "task_type": "choice",
            "gold": "yes",
        }
        records += [
            {**base, "version": "orig", "prediction": "yes"},
            {**base, "version": "conflict", "bias": "B4", "prediction": "no"},
        ]
    rows, _ = aggregate_predictions(records)
    assert any(r.get("question_group") == "E3" and r.get("exploratory") for r in rows)
    assert any(r.get("question_group") == "C1" and not r.get("exploratory") for r in rows)


def test_cli_writes_json_and_csv(tmp_path):
    import json
    from mm_systemonebias.metrics.cli import main

    source = tmp_path / "pred.jsonl"
    output = tmp_path / "out"
    base = {
        "model_id": "d3_mini",
        "question_group": "C1",
        "sample_id": "one",
        "template": 0,
        "task_type": "choice",
        "gold": "yes",
    }
    source.write_text(
        "\n".join(
            json.dumps(x)
            for x in [
                {**base, "version": "orig", "prediction": "yes"},
                {**base, "version": "num", "bias": "B2", "prediction": "no"},
            ]
        )
        + "\n"
    )
    main(["--predictions", str(source), "--output-dir", str(output)])
    payload = json.loads((output / "summary.json").read_text())
    assert (output / "conditions.csv").exists() and (output / "matched_backbones.csv").exists()
    assert payload["status"] == "complete"
    assert any(x.get("flip_rate") == 1 for x in payload["conditions"])


def test_cli_rejects_empty_prediction_input(tmp_path):
    from mm_systemonebias.metrics.cli import main

    source = tmp_path / "empty.jsonl"
    source.write_text("")
    with pytest.raises(ValueError, match="Prediction input is empty"):
        main(["--predictions", str(source), "--output-dir", str(tmp_path / "out")])


def test_cli_marks_missing_predictions_partial_but_not_missing_probabilities(tmp_path):
    import json

    from mm_systemonebias.metrics.cli import main

    source = tmp_path / "incomplete.jsonl"
    rows = [
        {
            "model_id": "custom_api",
            "question_group": "C1",
            "sample_id": "one",
            "template": 0,
            "version": "orig",
            "task_type": "choice",
            "gold": "yes",
            "prediction": "yes",
            "probabilities": None,
        },
        {
            "model_id": "custom_api",
            "question_group": "C1",
            "sample_id": "orphan",
            "template": 0,
            "version": "perm1",
            "bias": "B1",
            "task_type": "choice",
            "gold": "yes",
            "prediction": None,
            "status": "error",
        },
    ]
    source.write_text("".join(json.dumps(row) + "\n" for row in rows))
    output = tmp_path / "partial"
    assert main(["--predictions", str(source), "--output-dir", str(output)]) == 2
    payload = json.loads((output / "summary.json").read_text())
    assert payload["status"] == "partial"
    assert payload["coverage"][0]["complete_probability_distributions"] == 0
    assert payload["coverage"][0]["failed_or_missing_prediction"] == 1
    assert payload["coverage"][0]["unpaired_variants"] == 1
