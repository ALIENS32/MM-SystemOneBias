# Metrics and Aggregation

The evaluator computes decision stability, task performance, probability shifts on unchanged decisions, and matched comparisons between decision models and their general-purpose backbones.

---

## 1. Running the Summarizer

Install the package (or activate a configured profile) and pass one or more prediction JSONL files to `scripts/summarize.sh` or `mm_systemonebias.metrics.cli`:

```bash
bash scripts/summarize.sh "$HOME/.cache/mm-systemonebias/runs/summary" \
  predictions/model-a.jsonl predictions/model-b.jsonl
```

Or invoke the Python module directly:

```bash
python -m mm_systemonebias.metrics.cli \
  --predictions run.jsonl \
  --output-dir "$HOME/.cache/mm-systemonebias/runs/summary"
```

### Output Files and Units

The output directory contains three artifacts:
- `summary.json`: run status (`complete` or `partial`), per-model `coverage`, condition-level metrics (`conditions`), and matched backbone comparisons (`matched_backbones`).
- `conditions.csv`: tabular per-model, per-intervention, and group-macro summary.
- `matched_backbones.csv`: comparisons between each open-weight decision model and its matched backbone over shared task coverage.

All rates and probability shifts in `summary.json` and the CSVs are stored as **fractions in $[0, 1]$** (multiply by `100` to obtain paper percentages). Grade shifts are in grade units (`0` to `4`), and NLL / Brier differences use standard score units. Numerator/denominator counts accompany every rate.

---

## 2. Input Format and Original–Intervened Alignment

Each line in a prediction JSONL file represents one model input, accepted either as a flat JSON object or nested under `{"record": ..., ...prediction fields...}`:

- **Identity & task fields**: `model_id`, `sample_id`, `root_id`, `question_group`, `scenario`, `task_type`, `options`, `gold`, `version`, `template`, `bias`, and `metadata` / `evaluation_metadata`.
- **Prediction fields**: `prediction` (semantically aligned answer ID), `probabilities` (dictionary keyed by legal semantic answer IDs), and `probability_status`.

To form **original–intervened comparisons**, the evaluator indexes each original instance by `(model_id, question_group, sample_id, template)` (`version` in `{"orig", "original"}`) and matches each intervened instance using the same key plus `bias` and `version`.

> **Convention:** Predictions must already be mapped to canonical semantic answer IDs across all option-order, option-label, and rating-scale variants. Use the standard version IDs (`swap` / `perm1`–`perm3`, `num` / `rev`, external-judgment source + `_gold` / `_wrong`, `consistent` / `conflict`, `n1` / `n2` / `n4`, and `from0` / `from6` / `x10` / `center`).

---

## 3. Metric Definitions

### Decision and Task-Outcome Metrics

- **Accuracy (`Acc`) & Accuracy Change ($\Delta\mathrm{Acc}$)**: Computed on *choice* and *yes/no judgment* tasks; $\Delta\mathrm{Acc}$ equals intervened accuracy minus original accuracy.
- **Semantic Flip Rate (`FR`)**: Fraction of original–intervened comparisons in which the aligned semantic decision changes ($\hat y^{(b,v)} \neq \hat y^{(0)}$).
- **Error Introduction Rate (`EIR`)**: On *choice* and *yes/no judgment* tasks under misleading **External Judgment** or conflicting **Image–Text Description** conditions, the fraction of originally correct decisions that turn incorrect. Reported as `null` if the denominator is zero.
- **Error Correction Rate (`ECR`)**: On the same tasks under correct **External Judgment** or consistent **Image–Text Description** conditions, the fraction of originally incorrect decisions that become correct. Reported as `null` if the denominator is zero.
- **Target-Following Rate**: When a misleading intervention specifies a single injected wrong target (`injected_id` or single-element `target_ids`), the fraction of originally correct decisions that move to that specific target. Contexts with multiple target IDs contribute to target-set probability mass shifts without a single-target rate denominator.
- **Correctness Transition Proportions**: Four-way breakdown (`correct -> correct`, `correct -> incorrect`, `incorrect -> correct`, `incorrect -> incorrect`).
- **Scoring Metrics**: Semantic option IDs map to canonical numeric grades (`1` to `5`) via `options[].value`. The evaluator reports original and per-scale Spearman's $\rho$ against `metadata.mos_quality`, signed and mean absolute grade shifts, **normalized absolute grade shift** (absolute shift divided by the 4-grade range), exact grade consistency across all five scales (`orig`, `from0`, `from6`, `x10`, `center`), and absolute-agreement `ICC(A,1)`.
- **All-Condition Consistency**: For **Option Order**, consistency is computed across `orig` + `swap` (2-way choice) or `orig` + `perm1`–`perm3` (4-way choice); for **Irrelevant Information**, comparisons cover pairwise dose changes and complete three-dose (`n1`, `n2`, `n4`) consistency.

### Answer-Probability Metrics (Unchanged Decisions)

Probability metrics require a complete, non-negative distribution summing to `1.0` over all legal semantic answer IDs in both the original and intervened records; incomplete distributions are omitted from probability metrics without discarding valid decision comparisons. On instances where the semantic decision stays unchanged ($\hat y^{(b,v)} = \hat y^{(0)}$), the evaluator reports:

- Mean signed and absolute shift of the original answer's probability ($|\Delta p_o|$) and the fraction of comparisons shifting by $\ge 0.10$ ($10\%$).
- Target-set probability shift ($\Delta p_t$) summed over `target_ids` / `injected_id`.
- Intervened-minus-original negative log-likelihood ($\Delta\text{NLL}$) and multiclass Brier score ($\Delta\text{Brier}$). Zero gold-answer probabilities yield undefined NLL differences and are excluded from the NLL mean.

---

## 4. Aggregation and Matched Backbone Protocol

### Bias Code Mapping

| Bias Code | Paper Intervention Name | Aggregation Scope |
| --- | --- | --- |
| `B1` | Option Order | Equal weighting across conditions within task–interface groups, then equal task–interface group weights |
| `B2` | Option Label | Equal weighting across conditions within task–interface groups, then equal task–interface group weights |
| `B3` | External Judgment | Average interfaces within each task scenario, then weight scenarios equally |
| `B4` | Image–Text Description | Average interfaces within each task scenario, then weight confirmatory scenarios equally (`B4/E3` robotic grasping stage is flagged `exploratory` and excluded from confirmatory macro averages) |
| `B5` | Irrelevant Information | Equal weighting across conditions within task–interface groups, then equal task–interface group weights |
| `B6` | Rating Scale | Computed on the scoring task (`AGIQA-3K`) |

- **Mean accuracy drop**: Averages original-minus-intervened accuracy equally across the five non-scale intervention families (`B1`, `B2`, incorrect `B3`, conflicting `B4`, and `B5`) alongside the number of observed components.
- **Stable-probability aggregation**: Averages conditions within question groups, then weights question groups equally.
- **Group macro rows (`model_group_macro`)**: Equal-model averages across the 8 decision models and 5 general-purpose backbones (custom external `model_id` values remain in per-model summaries without entering paper group averages).

### Matched Comparisons Between Decision Models and Backbones

To compare each of the 7 open-weight decision models against its corresponding backbone fairly:

1. Comparisons intersect exact `(question_group, sample_id, bias, version, template)` keys across shared task coverage (for **Jev-Omni** vs. **Gemma 4 12B IT**, this restricts to single-image tasks).
2. Metrics average conditions within each matched question group.
3. Differences (`decision model - backbone`) are averaged equally across decision models sharing a backbone (`JEV-27B-VL`, `Clef`, and `d3` on `Qwen3.8-27B`), and then weighted equally across the 5 independent backbones (`backbone_count` out of `5`).
4. Reported differences cover `FR` for `B1` / `B2` / `B5`, accuracy change for `B3` / `B4`, and normalized absolute grade shift for `B6`.

### Run Coverage Status

`summary.json` records per-model `coverage` (`observed_records`, `valid_decisions`, `failed_or_missing_prediction`, `complete_probability_distributions`, `unpaired_variants`, and `valid_pairs`). Top-level `status` is `partial` (exit code `2`) if any prediction failed/is missing or any intervened record lacks its matching original record; unsupported multi-image exclusions (for **Jev-Omni**) and missing probability distributions do not mark a run `partial`.
