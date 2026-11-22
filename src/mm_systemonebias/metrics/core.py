"""Pure metric and aggregation functions (no plotting or model dependencies)."""

from __future__ import annotations

import math
from collections import defaultdict
from typing import Any, Iterable

import numpy as np

MODEL_NAMES = {
    "autotrust_jev_27b": "JEV-27B-VL",
    "clef_27b": "Clef",
    "d3_27b": "d3",
    "clef_flash_9b": "Clef-Flash",
    "d3_mini": "d3-mini",
    "d1_3b": "d1-3B",
    "jev_omni": "Jev-Omni",
    "typesafe_jev": "TypeSafe Jev",
    "qwen_27b": "Qwen3.8-27B",
    "qwen_9b": "Qwen3.5-9B",
    "qwen_4b": "Qwen3.5-4B",
    "lfm_3b": "LFM2.5-VL-3B",
    "gemma_12b": "Gemma 4 12B IT",
}
BIAS_NAMES = {
    "B1": "option_order",
    "B2": "option_label",
    "B3": "external_judgment",
    "B4": "image_text",
    "B5": "irrelevant_information",
    "B6": "rating_scale",
}
BACKBONES = {
    "autotrust_jev_27b": "qwen_27b",
    "clef_27b": "qwen_27b",
    "d3_27b": "qwen_27b",
    "clef_flash_9b": "qwen_9b",
    "d3_mini": "qwen_4b",
    "d1_3b": "lfm_3b",
    "jev_omni": "gemma_12b",
}


def _finite(x: Any) -> float | None:
    try:
        v = float(x)
        return v if math.isfinite(v) else None
    except (TypeError, ValueError):
        return None


def _rho(xs: list[float], ys: list[float]) -> float | None:
    if len(xs) < 2 or len(set(xs)) < 2 or len(set(ys)) < 2:
        return None
    from scipy.stats import spearmanr

    v = float(spearmanr(xs, ys).statistic)
    return v if math.isfinite(v) else None


def compare_pair(original: dict, intervened: dict) -> dict:
    """Metrics for one valid original/intervened decision pair; absent subsets are null."""
    y0, y1, gold = original.get("prediction"), intervened.get("prediction"), original.get("gold")
    bias = str(intervened.get("bias", "")).upper()
    task = str(original.get("task_type", "")).lower()
    scoring = task == "score"
    valid = y0 is not None and y1 is not None and gold is not None
    if not valid:
        return {"n": 0, "n_probability": 0}
    correct0, correct1 = y0 == gold, y1 == gold
    score0 = _option_value(original, y0) if scoring else None
    score1 = _option_value(intervened, y1) if scoring else None
    version = str(intervened.get("version", "")).removesuffix("_t2")
    targets = _targets(intervened)
    direction = version.rsplit("_", 1)[-1]
    single_wrong_target = (
        not scoring
        and bias in ("B3", "B4")
        and direction in ("wrong", "conflict")
        and len(targets) == 1
        and gold not in targets
    )
    row = {
        "n": 1,
        "acc_n": 0 if scoring else 1,
        "flip": int(y0 != y1),
        "acc_original": None if scoring else int(correct0),
        "acc_intervened": None if scoring else int(correct1),
        "delta_acc": None if scoring else int(correct1) - int(correct0),
        "eir_num": int(correct0 and not correct1),
        "eir_den": int(correct0),
        "ecr_num": int(not correct0 and correct1),
        "ecr_den": int(not correct0),
        "target_follow_num": int(y1 in targets) if correct0 and single_wrong_target else 0,
        "target_follow_den": int(correct0 and single_wrong_target),
        "scoring": scoring,
        "grade_shift": (score1 - score0)
        if scoring and score1 is not None and score0 is not None
        else None,
        "normalized_grade_shift": ((score1 - score0) / 4)
        if scoring and score1 is not None and score0 is not None
        else None,
        "normalized_absolute_grade_shift": (abs(score1 - score0) / 4)
        if scoring and score1 is not None and score0 is not None
        else None,
        "score_prediction": score1,
        "score_original": score0,
    }
    # EIR/ECR only apply to misleading/supportive text judgments, never scale scores or neutral families.
    cond = str(intervened.get("version", "")).lower()
    meta = intervened.get("metadata") or {}
    direction = cond.removesuffix("_t2").rsplit("_", 1)[-1]
    supportive = direction in ("gold", "consistent") or meta.get("judgment_correct") is True
    misleading = (
        direction in ("wrong", "conflict", "conflict_strict")
        or meta.get("judgment_correct") is False
    )
    if bias in ("B3", "B4") and not scoring:
        row["eir_num"] = int(correct0 and not correct1) if misleading else 0
        row["eir_den"] = int(correct0) if misleading else 0
        row["ecr_num"] = int(not correct0 and correct1) if supportive else 0
        row["ecr_den"] = int(not correct0) if supportive else 0
    else:
        row["eir_num"] = row["eir_den"] = row["ecr_num"] = row["ecr_den"] = 0
    p0, p1 = original.get("probabilities"), intervened.get("probabilities")
    if (
        isinstance(p0, dict)
        and isinstance(p1, dict)
        and _valid_aligned_pair(p0, p1, original, intervened)
        and set(p0)
    ):
        q0, q1 = {k: float(p0[k]) for k in p0}, {k: float(p1[k]) for k in p1}
        # decision coverage is independent of probability coverage.
        row["n_probability"] = 1
        if y0 == y1:
            d = q1.get(y0, 0.0) - q0.get(y0, 0.0)
            targets = _targets(intervened)
            row.update(
                stable_n=1,
                original_prob_shift=d,
                target_prob_shift=(
                    sum(q1.get(t, 0) - q0.get(t, 0) for t in targets) if targets else None
                ),
                nll_delta=(-math.log(q1[gold]) + math.log(q0[gold]))
                if q0.get(gold, 0) > 0 and q1.get(gold, 0) > 0
                else None,
                brier_delta=sum(
                    (q1.get(k, 0) - (k == gold)) ** 2 - (q0.get(k, 0) - (k == gold)) ** 2
                    for k in set(q0) | set(q1)
                ),
            )
        else:
            row["stable_n"] = 0
    else:
        row["n_probability"] = 0
    return row


def _targets(row):
    meta = row.get("metadata") or {}
    vals = meta.get("target_ids", meta.get("injected_id"))
    if vals is None:
        return set()
    return set(vals) if isinstance(vals, (list, tuple, set)) else {vals}


def _option_value(record, answer):
    for option in record.get("options") or []:
        if option.get("id") == answer:
            return _finite(option.get("value"))
    return None


def _valid_distribution(p):
    vals = [_finite(v) for v in p.values()]
    return bool(vals) and all(v is not None and v >= 0 for v in vals) and abs(sum(vals) - 1) < 1e-5


def _valid_aligned_pair(p0, p1, original, intervened):
    if not (_valid_distribution(p0) and _valid_distribution(p1) and set(p0) == set(p1)):
        return False
    for row in (original, intervened):
        options = row.get("options")
        if isinstance(options, list) and options:
            legal = {
                str(x.get("id")) for x in options if isinstance(x, dict) and x.get("id") is not None
            }
            if legal and {str(k) for k in p0} != legal:
                return False
    return True


def _mean(vals):
    vals = [v for v in vals if v is not None and math.isfinite(float(v))]
    return sum(vals) / len(vals) if vals else None


def rating_metrics(original_grades: list, condition_grades: list, mos: list | None = None) -> dict:
    """Five-scale grade summaries; grades are canonical 1..5 before this call."""
    matrices = np.asarray([original_grades, *condition_grades], dtype=float).T
    if matrices.ndim != 2 or matrices.shape[1] != 5:
        raise ValueError("Expected original plus four alternative scales")
    shifts = matrices[:, 1:] - matrices[:, [0]]
    # ICC(A,1), two-way random effects, absolute agreement.
    n, k = matrices.shape
    if n < 2:
        icc = None
    else:
        grand = matrices.mean()
        rowmeans = matrices.mean(1)
        colmeans = matrices.mean(0)
        msr = k * ((rowmeans - grand) ** 2).sum() / (n - 1)
        msc = n * ((colmeans - grand) ** 2).sum() / (k - 1)
        mse = ((matrices - rowmeans[:, None] - colmeans[None, :] + grand) ** 2).sum() / (
            (n - 1) * (k - 1)
        )
        denom = msr + (k - 1) * mse + k * (msc - mse) / n
        icc = (msr - mse) / denom if denom else None
    result = {
        "n": n,
        "all_scale_consistency": float(np.all(matrices == matrices[:, [0]], axis=1).mean())
        if n
        else None,
        "icc_a1": float(icc) if icc is not None and math.isfinite(icc) else None,
        "scale_signed_grade_shift": float(shifts.mean()) if shifts.size else None,
        "scale_abs_grade_shift": float(np.abs(shifts).mean()) if shifts.size else None,
        "scale_normalized_abs_grade_shift": float(np.abs(shifts).mean() / 4)
        if shifts.size
        else None,
    }
    if mos is not None and len(mos) == n:
        result["original_spearman_mos"] = _rho(matrices[:, 0].tolist(), list(map(float, mos)))
    return result


def transition_proportions(rows: Iterable[dict]) -> dict:
    """Correctness transition proportions, each normalized over all valid pairs."""
    counts = {
        "correct_to_correct": 0,
        "correct_to_incorrect": 0,
        "incorrect_to_correct": 0,
        "incorrect_to_incorrect": 0,
    }
    for r in rows:
        if not r.get("n"):
            continue
        a, b = bool(r["acc_original"]), bool(r["acc_intervened"])
        counts[("correct" if a else "incorrect") + "_to_" + ("correct" if b else "incorrect")] += 1
    n = sum(counts.values())
    return {"n": n, **{k: (v / n if n else None) for k, v in counts.items()}}


def summarize(rows: Iterable[dict]) -> dict:
    """Aggregate pair rows with explicit denominators and null undefined rates."""
    rows = list(rows)
    n = sum(r.get("n", 0) for r in rows)

    def sums(key):
        return sum(r.get(key, 0) or 0 for r in rows)

    def rate(a, b):
        return sums(a) / sums(b) if sums(b) else None

    stable = [r for r in rows if r.get("stable_n", 0)]
    classified = [r for r in rows if r.get("acc_n", 0)]
    transitions = transition_proportions(
        [
            {
                "n": r["n"],
                "acc_original": bool(r["acc_original"]),
                "acc_intervened": bool(r["acc_intervened"]),
            }
            for r in classified
        ]
    )
    return {
        "n": n,
        "n_accuracy": sums("acc_n"),
        "n_probability": sums("n_probability"),
        "n_stable_probability": sums("stable_n"),
        "n_stable_original_answer_shift": sum(
            r.get("original_prob_shift") is not None for r in stable
        ),
        "n_stable_target_probability_shift": sum(
            r.get("target_prob_shift") is not None for r in stable
        ),
        "n_stable_delta_nll": sum(r.get("nll_delta") is not None for r in stable),
        "n_stable_delta_brier": sum(r.get("brier_delta") is not None for r in stable),
        "acc_original": rate("acc_original", "acc_n"),
        "acc_intervened": rate("acc_intervened", "acc_n"),
        "delta_acc": rate("delta_acc", "acc_n"),
        "flip_rate": rate("flip", "n"),
        "eir": rate("eir_num", "eir_den"),
        "eir_denominator": sums("eir_den"),
        "ecr": rate("ecr_num", "ecr_den"),
        "ecr_denominator": sums("ecr_den"),
        "target_follow": rate("target_follow_num", "target_follow_den"),
        "target_follow_denominator": sums("target_follow_den"),
        "signed_grade_shift": _mean([r.get("grade_shift") for r in rows]),
        "absolute_grade_shift": _mean(
            [abs(r["grade_shift"]) for r in rows if r.get("grade_shift") is not None]
        ),
        "normalized_absolute_grade_shift": (
            _mean([abs(r["grade_shift"]) for r in rows if r.get("grade_shift") is not None]) / 4
            if any(r.get("grade_shift") is not None for r in rows)
            else None
        ),
        "stable_original_prob_signed_shift": _mean([r.get("original_prob_shift") for r in stable]),
        "stable_original_prob_abs_shift": _mean(
            [
                abs(r["original_prob_shift"])
                for r in stable
                if r.get("original_prob_shift") is not None
            ]
        ),
        "stable_original_prob_shift_ge_0_1_fraction": _mean(
            [
                abs(r["original_prob_shift"]) >= 0.1
                for r in stable
                if r.get("original_prob_shift") is not None
            ]
        ),
        "stable_target_prob_shift": _mean([r.get("target_prob_shift") for r in stable]),
        "stable_delta_nll": _mean([r.get("nll_delta") for r in stable]),
        "stable_delta_brier": _mean([r.get("brier_delta") for r in stable]),
        "transition_n": transitions["n"],
        **{f"transition_{k}": v for k, v in transitions.items() if k != "n"},
    }


def _is_primary(bias, version):
    base = version
    expected = {
        "B1": {"swap", "perm1", "perm2", "perm3"},
        "B2": {"num", "rev"},
        "B3": {
            f"{source}_{direction}"
            for source in ("user", "hist", "tool", "userstrong")
            for direction in ("gold", "wrong")
        },
        "B4": {"conflict", "consistent"},
        "B5": {"n1", "n2", "n4"},
        "B6": {"from0", "from6", "x10", "center"},
    }
    return base in expected.get(bias, set())


def _family(bias, version):
    base = version.removesuffix("_t2")
    if bias == "B1":
        return "all_orders"
    if bias == "B2":
        return "all_labels"
    if bias == "B3":
        return base.rsplit("_", 1)[-1]
    if bias == "B4":
        return base
    if bias == "B5":
        return "all_conditions"
    if bias == "B6":
        return "all_scales"
    return base


def _macro(records, key, metric):
    """Equal mean across strata, retaining the underlying sample count separately."""
    bins = defaultdict(list)
    for row in records:
        val = row.get(metric)
        if val is not None:
            bins[row.get(key)].append(float(val))
    per = {k: _mean(v) for k, v in bins.items()}
    return _mean(list(per.values())), len(per)


def aggregate_predictions(records: Iterable[dict]) -> tuple[list[dict], list[dict]]:
    """Paper metrics: raw-version summaries, macro condition families, and matched backbones."""
    raw = list(records)
    clean = []
    for row in raw:
        if isinstance(row.get("record"), dict):
            row = {**row["record"], **{k: v for k, v in row.items() if k != "record"}}
        if row.get("model_id"):
            clean.append(row)
    original = {}
    variants = []
    for row in clean:
        key = (row["model_id"], row["question_group"], row["sample_id"], row.get("template"))
        if row.get("version") == "orig":
            original[key] = row
        elif row.get("bias") and _is_primary(str(row["bias"]), str(row.get("version", ""))):
            variants.append(row)
    pairs = []
    for changed in variants:
        bias = str(changed["bias"])
        key = (
            changed["model_id"],
            changed["question_group"],
            changed["sample_id"],
            changed.get("template"),
        )
        orig = original.get(key)
        if orig is None:
            continue
        m = compare_pair(orig, changed)
        if not m.get("n"):
            continue
        m.update(
            model_id=changed["model_id"],
            model_name=MODEL_NAMES.get(changed["model_id"], changed["model_id"]),
            question_group=changed["question_group"],
            scenario=changed.get("scenario", changed["question_group"]),
            task_type=changed["task_type"],
            bias=bias,
            raw_version=changed["version"],
            condition=_family(bias, changed["version"]),
            template=changed.get("template"),
            sample_id=changed["sample_id"],
            flip_rate=m.get("flip"),
            exploratory=(bias == "B4" and changed["question_group"] == "E3"),
        )
        m["original_record"] = orig
        m["changed_record"] = changed
        pairs.append(m)
    output = []
    # First summarize by question group and raw intervention version. This preserves version counts.
    rawcells = defaultdict(list)
    for p in pairs:
        rawcells[
            (p["model_id"], p["question_group"], p["bias"], p["raw_version"], p["template"])
        ].append(p)
    fine = []
    for (model, qg, bias, version, template), rows in sorted(rawcells.items()):
        s = summarize(rows)
        sample = rows[0]
        s.update(
            model_id=model,
            model_name=MODEL_NAMES.get(model, model),
            question_group=qg,
            scenario=sample["scenario"],
            task_type=sample["task_type"],
            bias=bias,
            raw_version=version,
            condition=_family(bias, version),
            template=template,
            exploratory=sample["exploratory"],
        )
        fine.append(s)
        output.append(s)
    # Equal weight raw versions within each interface/task group for the intervention family.
    groupcells = defaultdict(list)
    for p in pairs:
        groupcells[
            (
                p["model_id"],
                p["question_group"],
                p["scenario"],
                p["task_type"],
                p["bias"],
                p["condition"],
                p["exploratory"],
            )
        ].append(p)
    group_rows = []
    for key, rows in groupcells.items():
        model, qg, scenario, task, bias, condition, explore = key
        version_rows = [
            r
            for r in fine
            if r["model_id"] == model
            and r["question_group"] == qg
            and r["bias"] == bias
            and r["condition"] == condition
            and r["template"] in {p["template"] for p in rows}
        ]
        metrics = (
            "acc_original",
            "acc_intervened",
            "delta_acc",
            "flip_rate",
            "eir",
            "ecr",
            "target_follow",
            "signed_grade_shift",
            "absolute_grade_shift",
            "normalized_absolute_grade_shift",
            "stable_original_prob_signed_shift",
            "stable_original_prob_abs_shift",
            "stable_original_prob_shift_ge_0_1_fraction",
            "stable_target_prob_shift",
            "stable_delta_nll",
            "stable_delta_brier",
            "transition_correct_to_correct",
            "transition_correct_to_incorrect",
            "transition_incorrect_to_correct",
            "transition_incorrect_to_incorrect",
        )
        vals = {m: _mean([v.get(m) for v in version_rows]) for m in metrics}
        # Preserve distinct template/task evidence; task/template strata contribute equal weight.
        vals.update(
            model_id=model,
            model_name=MODEL_NAMES.get(model, model),
            question_group=qg,
            scenario=scenario,
            task_type=task,
            bias=bias,
            condition=condition,
            exploratory=explore,
            raw_version="macro_versions",
            n=len({p["sample_id"] for p in rows}),
            n_comparisons=sum(r["n"] for r in version_rows),
            n_probability=sum(r["n_probability"] for r in version_rows),
            n_stable_probability=sum(r["n_stable_probability"] for r in version_rows),
            eir_denominator=sum(r["eir_denominator"] for r in version_rows),
            ecr_denominator=sum(r["ecr_denominator"] for r in version_rows),
            version_count=len({r["raw_version"] for r in version_rows}),
            template_count=len({r["template"] for r in version_rows}),
            interface_key=f"{scenario}::{task}",
            _sample_ids=sorted({p["sample_id"] for p in rows}),
        )
        group_rows.append(vals)
    output.extend(group_rows)
    # Main macro: FR families average task-interface groups (14 combinations);
    # textual conditions average interfaces within a scenario then scenarios equally.
    families = defaultdict(list)
    for r in group_rows:
        families[(r["model_id"], r["bias"], r["condition"], r["exploratory"])].append(r)
    macros = []
    for (model, bias, condition, explore), rows in families.items():
        metrics = (
            "acc_original",
            "acc_intervened",
            "delta_acc",
            "flip_rate",
            "eir",
            "ecr",
            "target_follow",
            "signed_grade_shift",
            "absolute_grade_shift",
            "normalized_absolute_grade_shift",
            "stable_original_prob_signed_shift",
            "stable_original_prob_abs_shift",
            "stable_original_prob_shift_ge_0_1_fraction",
            "stable_target_prob_shift",
            "stable_delta_nll",
            "stable_delta_brier",
        )
        summary = {}
        for metric in metrics:
            use_scenario = bias in ("B3", "B4") and metric in (
                "acc_original",
                "acc_intervened",
                "delta_acc",
                "eir",
                "ecr",
                "target_follow",
            )
            probability_metric = metric.startswith("stable_")
            key = (
                "question_group"
                if probability_metric
                else "scenario"
                if use_scenario
                else "interface_key"
            )
            value, nstrata = _macro(rows, key, metric)
            summary[metric] = value
            summary[metric + "_strata"] = nstrata
        summary.update(
            model_id=model,
            model_name=MODEL_NAMES.get(model, model),
            bias=bias,
            condition=condition,
            exploratory=explore,
            n=sum(r.get("n_comparisons", r["n"]) for r in rows),
            n_probability=sum(r["n_probability"] for r in rows),
            n_stable_probability=sum(r["n_stable_probability"] for r in rows),
            eir_denominator=sum(r["eir_denominator"] for r in rows),
            ecr_denominator=sum(r["ecr_denominator"] for r in rows),
            question_group_count=len({r["question_group"] for r in rows}),
            version_count=max((r.get("version_count", 0) for r in rows), default=0),
            kind="condition_macro",
        )
        macros.append(summary)
    for row in group_rows:
        row.pop("_sample_ids", None)
    output.extend(macros)
    all_models = sorted({r["model_id"] for r in clean})
    models = all_models
    output.extend(_joint_rows(original, pairs, all_models))
    output.extend(_paper_performance_rows(original, clean, macros))
    output.extend(_rating_rows(original, variants, models))
    output.extend(_contrast_rows(pairs, models))
    output.extend(_model_group_rows(output))
    # Matched models: intersect exact model/backbone (group,sample,bias,raw-version,template) before aggregation.
    matched = _matched_backbone_summaries(pairs)
    return output, matched


def _matched_backbone_summaries(pairs):
    """Aggregate exact common raw-version pairs through the paper weighting hierarchy."""
    matched = []
    for model, backbone in BACKBONES.items():
        idx = {}
        bidx = {}
        for p in pairs:
            key = (p["question_group"], p["sample_id"], p["bias"], p["raw_version"], p["template"])
            if p["model_id"] == model:
                idx[key] = p
            if p["model_id"] == backbone:
                bidx[key] = p
        shared = sorted(idx.keys() & bidx.keys())
        drows = []
        for key in shared:
            left, right = idx[key], bidx[key]
            drows.append(
                {
                    "key": key,
                    "model_delta": left["delta_acc"],
                    "backbone_delta": right["delta_acc"],
                    "delta_difference": (left["delta_acc"] - right["delta_acc"])
                    if left.get("delta_acc") is not None and right.get("delta_acc") is not None
                    else None,
                    "flip_difference": left["flip_rate"] - right["flip_rate"],
                    "score_difference": (
                        left.get("normalized_absolute_grade_shift")
                        - right.get("normalized_absolute_grade_shift")
                    )
                    if left.get("normalized_absolute_grade_shift") is not None
                    and right.get("normalized_absolute_grade_shift") is not None
                    else None,
                    "model_flip": left["flip_rate"],
                    "backbone_flip": right["flip_rate"],
                    "bias": left["bias"],
                    "version": left["raw_version"],
                    "group": left["question_group"],
                    "scenario": left["scenario"],
                    "task_type": left["task_type"],
                    "exploratory": left["exploratory"],
                }
            )
        for bias in BIAS_NAMES:
            members = [x for x in drows if x["bias"] == bias]
            if not members:
                continue
            # raw version then task group equal means; output data rows and higher-level equal-backbone aggregate below.
            by_version_group = defaultdict(list)
            for x in members:
                by_version_group[(x["version"], x["group"])].append(x)
            for (version, qg), items in by_version_group.items():
                matched.append(
                    {
                        "model_id": model,
                        "backbone_id": backbone,
                        "bias": BIAS_NAMES[bias],
                        "raw_version": version,
                        "question_group": qg,
                        "scenario": items[0]["scenario"],
                        "task_type": items[0]["task_type"],
                        "exploratory": items[0]["exploratory"],
                        "n_common": len(items),
                        "model_delta_acc": _mean([x["model_delta"] for x in items]),
                        "backbone_delta_acc": _mean([x["backbone_delta"] for x in items]),
                        "delta_acc_difference": _mean([x["delta_difference"] for x in items]),
                        "flip_rate_difference": _mean([x["flip_difference"] for x in items]),
                        "normalized_absolute_grade_shift_difference": _mean(
                            [x["score_difference"] for x in items]
                        ),
                        "model_flip_rate": _mean([x["model_flip"] for x in items]),
                        "backbone_flip_rate": _mean([x["backbone_flip"] for x in items]),
                        "kind": "matched_detail",
                    }
                )
    # Explicit coverage-aware equal-backbone estimates, with only actually observed backbone count.
    ma = defaultdict(list)
    for r in matched:
        if r["kind"] == "matched_detail":
            ma[
                (
                    r["bias"],
                    _family(
                        next(k for k, v in BIAS_NAMES.items() if v == r["bias"]), r["raw_version"]
                    ),
                    r["exploratory"],
                )
            ].append(r)
    for (bias, condition, exploratory), rows in ma.items():
        bybb = defaultdict(lambda: defaultdict(list))
        for r in rows:
            bybb[r["backbone_id"]][r["model_id"]].append(r)
        bbmeans = []
        bbfr = []
        bbscore = []
        for decision_models in bybb.values():
            modelmeans = []
            modelfr = []
            modelscore = []
            for model_rows in decision_models.values():
                # Version cells are equal within question group; groups equal within model.
                if bias in ("image_text", "external_judgment"):
                    scenarios = defaultdict(lambda: defaultdict(list))
                    for x in model_rows:
                        scenarios[x["scenario"]][x["question_group"]].append(x)

                    def group_values(metric):
                        return _mean(
                            [
                                _mean(
                                    [
                                        _mean([z.get(metric) for z in version_rows])
                                        for version_rows in qgroups.values()
                                    ]
                                )
                                for qgroups in scenarios.values()
                            ]
                        )

                    modelmeans.append(group_values("delta_acc_difference"))
                    modelfr.append(group_values("flip_rate_difference"))
                    modelscore.append(group_values("normalized_absolute_grade_shift_difference"))
                else:
                    qgroups = defaultdict(list)
                    for x in model_rows:
                        qgroups[x["question_group"]].append(x)
                    modelmeans.append(
                        _mean(
                            [
                                _mean([x["delta_acc_difference"] for x in xs])
                                for xs in qgroups.values()
                            ]
                        )
                    )
                    modelfr.append(
                        _mean(
                            [
                                _mean([x["flip_rate_difference"] for x in xs])
                                for xs in qgroups.values()
                            ]
                        )
                    )
                    modelscore.append(
                        _mean(
                            [
                                _mean([x["normalized_absolute_grade_shift_difference"] for x in xs])
                                for xs in qgroups.values()
                            ]
                        )
                    )
            bbmeans.append(_mean(modelmeans))
            bbfr.append(_mean(modelfr))
            bbscore.append(_mean(modelscore))
        matched.append(
            {
                "aggregate": "equal_backbone_macro",
                "bias": bias,
                "condition": condition,
                "exploratory": exploratory,
                "backbone_count": len(bbmeans),
                "expected_backbone_count": 5,
                "delta_acc_difference": _mean(bbmeans),
                "flip_rate_difference": _mean(bbfr),
                "normalized_absolute_grade_shift_difference": _mean(bbscore),
                "kind": "matched_macro",
            }
        )
    return matched


def _paper_performance_rows(original, clean, macros):
    results = []
    by_model_interface = defaultdict(list)
    for r in original.values():
        if (
            r.get("task_type") != "score"
            and r.get("prediction") is not None
            and r.get("gold") is not None
        ):
            by_model_interface[
                (r["model_id"], r.get("scenario", r["question_group"]), r.get("task_type"))
            ].append(float(r["prediction"] == r["gold"]))
    models = sorted({r["model_id"] for r in clean})
    for model in models:
        scenarios = defaultdict(list)
        for (m, scenario, _task), v in by_model_interface.items():
            if m == model:
                scenarios[scenario].append(_mean(v))
        means = [_mean(v) for v in scenarios.values()]
        results.append(
            {
                "model_id": model,
                "model_name": MODEL_NAMES.get(model, model),
                "bias": "original",
                "condition": "scenario_macro_accuracy",
                "acc_original": _mean(means),
                "scenario_count": len(means),
                "n": sum(len(v) for (m, _, _), v in by_model_interface.items() if m == model),
                "kind": "original_macro",
            }
        )
    # Required five-component mean accuracy drop; missing components remain visible.
    for model in models:
        selections = [
            r
            for r in macros
            if r["model_id"] == model
            and not r["exploratory"]
            and (
                (r["bias"] == "B1" and r["condition"] == "all_orders")
                or (r["bias"] == "B2" and r["condition"] == "all_labels")
                or (r["bias"] == "B3" and r["condition"] == "wrong")
                or (r["bias"] == "B4" and r["condition"] == "conflict")
                or (r["bias"] == "B5" and r["condition"] == "all_conditions")
            )
        ]
        components = [-r["delta_acc"] for r in selections if r.get("delta_acc") is not None]
        results.append(
            {
                "model_id": model,
                "model_name": MODEL_NAMES.get(model, model),
                "bias": "main",
                "condition": "mean_accuracy_drop",
                "mean_accuracy_drop": _mean(components),
                "component_count": len(components),
                "expected_component_count": 5,
                "missing_component_count": 5 - len(components),
                "kind": "main_macro",
            }
        )
    return results


def _rating_rows(original, variants, models):
    results = []
    for model in models:
        score_orig = [
            r for r in original.values() if r["model_id"] == model and r.get("task_type") == "score"
        ]
        vals = []
        for r in score_orig:
            score = _option_value(r, r.get("prediction"))
            mos = _finite((r.get("metadata") or {}).get("mos_quality"))
            if score is not None and mos is not None:
                vals.append((r.get("scenario", r["question_group"]), mos, score))
        for scenario in {v[0] for v in vals}:
            part = [v for v in vals if v[0] == scenario]
            results.append(
                {
                    "model_id": model,
                    "model_name": MODEL_NAMES.get(model, model),
                    "bias": "B6",
                    "condition": "original_spearman_mos",
                    "scenario": scenario,
                    "spearman_mos": _rho([v[1] for v in part], [v[2] for v in part]),
                    "n": len(part),
                    "kind": "rating_metric",
                }
            )
        for version in ("from0", "from6", "x10", "center"):
            per = []
            for changed in variants:
                if (
                    changed["model_id"] != model
                    or changed["bias"] != "B6"
                    or changed["version"].removesuffix("_t2") != version
                ):
                    continue
                key = (
                    model,
                    changed["question_group"],
                    changed["sample_id"],
                    changed.get("template"),
                )
                orig = original.get(key)
                score = _option_value(changed, changed.get("prediction"))
                mos = _finite((changed.get("metadata") or {}).get("mos_quality"))
                if orig and score is not None and mos is not None:
                    per.append((changed.get("scenario", changed["question_group"]), mos, score))
            for scenario in {x[0] for x in per}:
                part = [x for x in per if x[0] == scenario]
                results.append(
                    {
                        "model_id": model,
                        "model_name": MODEL_NAMES.get(model, model),
                        "bias": "B6",
                        "condition": "scale_spearman_mos",
                        "raw_version": version,
                        "scenario": scenario,
                        "spearman_mos": _rho([x[1] for x in part], [x[2] for x in part]),
                        "n": len(part),
                        "kind": "rating_metric",
                    }
                )
        matrix = []
        mosvals = []
        for r in score_orig:
            key = (model, r["question_group"], r["sample_id"], r.get("template"))
            versions = [
                v
                for v in variants
                if v["model_id"] == model
                and v["bias"] == "B6"
                and v["question_group"] == key[1]
                and v["sample_id"] == key[2]
                and v.get("template") == key[3]
            ]
            found = {v["version"].removesuffix("_t2"): v for v in versions}
            if set(found) != {"from0", "from6", "x10", "center"}:
                continue
            allrows = [r] + [found[x] for x in ("from0", "from6", "x10", "center")]
            grades = [_option_value(x, x.get("prediction")) for x in allrows]
            if all(x is not None for x in grades):
                matrix.append(grades)
                mosvals.append(_finite((r.get("metadata") or {}).get("mos_quality")))
        if matrix:
            a = np.asarray(matrix)
            icc = rating_metrics(a[:, 0].tolist(), [a[:, i].tolist() for i in range(1, 5)])
            results.append(
                {
                    "model_id": model,
                    "model_name": MODEL_NAMES.get(model, model),
                    "bias": "B6",
                    "condition": "complete_five_scale_agreement",
                    "n": len(matrix),
                    "all_scale_consistency": float(np.all(a == a[:, [0]], axis=1).mean()),
                    "icc_a1": icc["icc_a1"],
                    "signed_grade_shift": float((a[:, 1:] - a[:, [0]]).mean()),
                    "absolute_grade_shift": float(np.abs(a[:, 1:] - a[:, [0]]).mean()),
                    "normalized_absolute_grade_shift": float(
                        np.abs(a[:, 1:] - a[:, [0]]).mean() / 4
                    ),
                    "kind": "rating_metric",
                }
            )
    return results


def _contrast_rows(pairs, models):
    results = []
    bysample = defaultdict(dict)
    for p in pairs:
        if p["bias"] in ("B4", "B5"):
            bysample[
                (p["model_id"], p["question_group"], p["sample_id"], p["template"], p["bias"])
            ][p["raw_version"]] = p
    for model in models:
        description = defaultdict(list)
        doses = defaultdict(list)
        for (m, qg, sample, template, bias), versions in bysample.items():
            if m != model:
                continue
            if bias == "B4" and "consistent" in versions and "conflict" in versions:
                yes, no = versions["consistent"], versions["conflict"]
                description[yes["scenario"]].append(
                    (
                        float(yes["changed_record"]["prediction"] == yes["changed_record"]["gold"])
                        - float(no["changed_record"]["prediction"] == no["changed_record"]["gold"]),
                        yes,
                        no,
                    )
                )
            if bias == "B5" and {"n1", "n2", "n4"} <= set(versions):
                chosen = [versions[v] for v in ("n1", "n2", "n4")]
                doses[qg].append(chosen)
        if description:
            scenario_gaps = [_mean([x[0] for x in items]) for items in description.values()]
            results.append(
                {
                    "kind": "condition_macro",
                    "model_id": model,
                    "model_name": MODEL_NAMES.get(model, model),
                    "bias": "B4",
                    "condition": "consistent_minus_conflicting_accuracy_gap",
                    "gap": _mean(scenario_gaps),
                    "scenario_count": len(scenario_gaps),
                    "n_common": sum(len(x) for x in description.values()),
                    "exploratory": False,
                }
            )
        for qg, groups in doses.items():
            for i, j in ((0, 1), (1, 2), (0, 2)):
                diffs = []
                for versions in groups:
                    left, right = versions[i], versions[j]
                    lc = float(
                        left["changed_record"]["prediction"] == left["changed_record"]["gold"]
                    )
                    rc = float(
                        right["changed_record"]["prediction"] == right["changed_record"]["gold"]
                    )
                    diffs.append(
                        {
                            "accuracy_difference": rc - lc,
                            "flip": int(
                                left["changed_record"]["prediction"]
                                != right["changed_record"]["prediction"]
                            ),
                        }
                    )
                results.append(
                    {
                        "kind": "dose_comparison",
                        "model_id": model,
                        "model_name": MODEL_NAMES.get(model, model),
                        "bias": "B5",
                        "question_group": qg,
                        "condition": f"{('n1', 'n2', 'n4')[i]}_to_{('n1', 'n2', 'n4')[j]}",
                        "n_common": len(diffs),
                        "delta_acc": _mean([x["accuracy_difference"] for x in diffs]),
                        "flip_rate": _mean([x["flip"] for x in diffs]),
                    }
                )
        complete_n = []
        for (m, qg, sample, template, bias), versions in bysample.items():
            if m == model and bias == "B5" and {"n1", "n2", "n4"} <= set(versions):
                original_prediction = versions["n1"]["original_record"]["prediction"]
                predictions = [original_prediction] + [
                    versions[x]["changed_record"]["prediction"] for x in ("n1", "n2", "n4")
                ]
                complete_n.append(len(set(predictions)))
        if complete_n:
            results.append(
                {
                    "kind": "joint_consistency",
                    "model_id": model,
                    "model_name": MODEL_NAMES.get(model, model),
                    "bias": "B5",
                    "condition": "all_irrelevant_doses",
                    "n_complete": len(complete_n),
                    "all_condition_consistency": _mean([x == 1 for x in complete_n]),
                }
            )
    return results


def _model_group_rows(output):
    results = []
    decision_ids = set(MODEL_NAMES) - set(BACKBONES.values())
    backbone_ids = set(BACKBONES.values())
    group_specs = (("decision_models", decision_ids, 8), ("backbones", backbone_ids, 5))
    source_rows = [
        r
        for r in output
        if r.get("kind") in ("condition_macro", "original_macro", "main_macro", "rating_metric")
    ]
    group_keys = defaultdict(list)
    for row in source_rows:
        for group, ids, expected in group_specs:
            if row.get("model_id") in ids and not row.get("exploratory", False):
                group_keys[
                    (
                        group,
                        expected,
                        row.get("kind"),
                        row.get("bias"),
                        row.get("condition"),
                        row.get("scenario"),
                        row.get("raw_version"),
                    )
                ].append(row)
    aggregate_metrics = (
        "acc_original",
        "acc_intervened",
        "delta_acc",
        "flip_rate",
        "eir",
        "ecr",
        "target_follow",
        "mean_accuracy_drop",
        "stable_original_prob_signed_shift",
        "stable_original_prob_abs_shift",
        "stable_original_prob_shift_ge_0_1_fraction",
        "stable_target_prob_shift",
        "stable_delta_nll",
        "stable_delta_brier",
        "spearman_mos",
        "all_scale_consistency",
        "icc_a1",
        "absolute_grade_shift",
        "normalized_absolute_grade_shift",
        "gap",
    )
    for (group, expected, kind, bias, condition, scenario, version), rows in group_keys.items():
        item = {
            "kind": "model_group_macro",
            "model_group": group,
            "source_kind": kind,
            "bias": bias,
            "condition": condition,
            "scenario": scenario,
            "raw_version": version,
            "model_count": len({r["model_id"] for r in rows}),
            "expected_model_count": expected,
        }
        for metric in aggregate_metrics:
            present = [r[metric] for r in rows if r.get(metric) is not None]
            if present:
                item[metric] = _mean(present)
        results.append(item)
    return results


def _joint_rows(original, pairs, all_models):
    results = []
    for model in all_models:
        for bias in ("B1", "B6"):
            complete = []
            for key, orig in original.items():
                if key[0] != model:
                    continue
                rows = {
                    p["raw_version"].removesuffix("_t2"): p
                    for p in pairs
                    if p["model_id"] == model
                    and p["question_group"] == key[1]
                    and p["sample_id"] == key[2]
                    and p["template"] == key[3]
                    and p["bias"] == bias
                }
                available = set(rows)
                expected = (
                    ({"swap"} if "swap" in available else {"perm1", "perm2", "perm3"})
                    if bias == "B1"
                    else {"from0", "from6", "x10", "center"}
                )
                if set(rows) == expected:
                    predictions = [orig["prediction"]] + [
                        rows[v]["changed_record"]["prediction"] for v in sorted(expected)
                    ]
                    values = (
                        [_option_value(orig, predictions[0])]
                        + [
                            _option_value(
                                rows[v]["changed_record"], rows[v]["changed_record"]["prediction"]
                            )
                            for v in sorted(expected)
                        ]
                        if bias == "B6"
                        else []
                    )
                    complete.append((predictions, values))
            if complete:
                results.append(
                    {
                        "kind": "joint_consistency",
                        "model_id": model,
                        "model_name": MODEL_NAMES.get(model, model),
                        "bias": bias,
                        "n_complete": len(complete),
                        "joint_consistency": _mean([len(set(x[0])) == 1 for x in complete]),
                    }
                )
    return results
