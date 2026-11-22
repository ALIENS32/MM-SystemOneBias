# Dataset and Model Inference Guide

## 1. Environment Setup

Python **3.12+** is required. You can install the core package for dataset loading, API inference, and metric aggregation with [uv](https://docs.astral.sh/uv/):

```bash
uv sync --extra dev
source .venv/bin/activate
```

For local model inference, create an isolated profile environment (`core`, `vlm`, `gemma`, `clef`, `d1`, `d3`, or `autotrust`) under `~/.cache/mm-systemonebias/envs/<profile>/`:

```bash
bash scripts/setup.sh core
source "$HOME/.cache/mm-systemonebias/envs/core/bin/activate"
```

See [Model Profiles](models.md) for the full mapping from paper model names and versions to config IDs and runtime profiles.

---

## 2. Dataset Preparation

Download the pinned public test split from Hugging Face Hub:

```bash
bash scripts/download_dataset.sh \
  --repo-id ALIENS232/MM-SystemOneBias \
  --revision 8c83db38ab824854534996cfd0a46ca9d57f4aee \
  --output-dir "$HOME/.cache/mm-systemonebias/dataset"
```

For quick local verification, download a single Parquet shard into a separate directory:

```bash
bash scripts/download_dataset.sh \
  --repo-id ALIENS232/MM-SystemOneBias \
  --revision 8c83db38ab824854534996cfd0a46ca9d57f4aee \
  --output-dir "$HOME/.cache/mm-systemonebias/dataset-smoke" \
  --allow-patterns test-00000.parquet README.md dataset_info.json
```

In Python, `datasets.load_dataset("ALIENS232/MM-SystemOneBias", split="test")` loads the packed Parquet split, and `mm_systemonebias.data.load_records()` expands each packed row into canonical inference records with locally materialized images.

---

## 3. Running Local Inference

Activate the runtime profile corresponding to your target model, download its pinned checkpoint, and run either a smoke check or the full benchmark (illustrated below for **d3-mini**, config ID `d3_mini`):

```bash
bash scripts/setup.sh d3
source "$HOME/.cache/mm-systemonebias/envs/d3/bin/activate"

bash scripts/download_model.sh --config configs/models/d3_mini.yaml

# Smoke run: selects 1 original instance per question group along with its intervened instances
bash scripts/smoke.sh d3_mini \
  --dataset-path "$HOME/.cache/mm-systemonebias/dataset-smoke"

# Full test-split run
bash scripts/infer.sh --config configs/models/d3_mini.yaml \
  --dataset-path "$HOME/.cache/mm-systemonebias/dataset" \
  --output-dir "$HOME/.cache/mm-systemonebias/runs/d3_mini"
```

### Selection, Resumption, and Retries

- `--question-groups C1,C2,G2,J2`: restricts evaluation to the listed question groups.
- `--limit-originals-per-group N`: selects the first `N` original roots per question group along with all of their intervened instances.
- `--resume`: continues an interrupted run into the same `--output-dir` after verifying that the model, dataset revision, and selection match `manifest.json`.
- `--retry`: implies `--resume` and re-evaluates records whose previous status was not `complete`.
- `--max-attempts N`: sets a hard cap on total API attempts across retries.
- `--set key=value`: overrides YAML fields at runtime (for example, `--set base_url=http://127.0.0.1:8000`).

---

## 4. Service and API Inference

### AutoTrust Local Service (`JEV-27B-VL`)

**JEV-27B-VL** (`configs/models/autotrust_jev_27b.yaml`) runs via a local vLLM server (`tensor_parallel_size: 2`, `max_model_len: 32768`, `max_num_seqs: 16`, `gpu_memory_utilization: 0.9`). Adjust `CUDA_VISIBLE_DEVICES` and `--tensor-parallel-size` to match available GPUs:

```bash
# Terminal 1: launch local vLLM service
bash scripts/setup.sh autotrust
source "$HOME/.cache/mm-systemonebias/envs/autotrust/bin/activate"
bash scripts/download_model.sh --config configs/models/autotrust_jev_27b.yaml
CUDA_VISIBLE_DEVICES=0,1 bash scripts/serve_autotrust.sh \
  --config configs/models/autotrust_jev_27b.yaml --port 8000
```

```bash
# Terminal 2: run inference client
bash scripts/setup.sh core
source "$HOME/.cache/mm-systemonebias/envs/core/bin/activate"
bash scripts/infer.sh --config configs/models/autotrust_jev_27b.yaml \
  --set base_url=http://127.0.0.1:8000 \
  --dataset-path "$HOME/.cache/mm-systemonebias/dataset" \
  --output-dir "$HOME/.cache/mm-systemonebias/runs/autotrust_jev_27b"
```

### TypeSafe Jev (`jev-1.14.0`) and OpenAI-Compatible Endpoints

Use the `core` profile and export credentials before invoking `scripts/infer.sh`:

```bash
# TypeSafe Jev (jev-1.14.0)
export TYPESAFE_BASE_URL="https://service.example"
export TYPESAFE_API_KEY="..."
bash scripts/infer.sh --config configs/models/typesafe_jev.yaml \
  --dataset-path "$HOME/.cache/mm-systemonebias/dataset" \
  --output-dir "$HOME/.cache/mm-systemonebias/runs/typesafe_jev"

# Generic OpenAI-compatible vision endpoint
export OPENAI_BASE_URL="https://api.example/v1"
export OPENAI_API_KEY="..."
export OPENAI_MODEL="vision-model"
bash scripts/infer.sh --config configs/models/openai_compatible.yaml \
  --dataset-path "$HOME/.cache/mm-systemonebias/dataset" \
  --output-dir "$HOME/.cache/mm-systemonebias/runs/openai_compatible"
```

---

## 5. Output Artifacts and Probability Extraction

Each run writes two files to `--output-dir`:
- `manifest.json`: model ID and pinned revision, resolved dataset identity, selection filters, sanitized inference configuration, and run status (`complete`, `partial`, or `failed`; exits with code `2` if any record failed).
- `predictions.jsonl`: one JSON line per model input containing the canonical record fields, `model_id`, semantic `prediction`, `probabilities`, `probability_status`, `probability_method`, `raw_response`, `usage`, `elapsed_seconds`, and `status`.

Probability extraction follows the model interface:
- **Decision models** (**JEV-27B-VL**, **Clef**, **d3**, **Clef-Flash**, **d3-mini**, **d1-3B**, **Jev-Omni**, **TypeSafe Jev**): return native candidate distributions over the legal semantic answer set (with thinking disabled for **JEV-27B-VL**). **Jev-Omni** accepts single-image inputs only (`max_images: 1`); multi-image task scenarios are automatically excluded by its adapter.
- **General-purpose VLMs** (**Qwen3.8-27B**, **Qwen3.5-9B**, **Qwen3.5-4B**, **LFM2.5-VL-3B**, **Gemma 4 12B IT**): run with a 0-token thinking budget (`thinking_budget: 0`) and compute normalized conditional token log-probabilities summed across each complete legal answer label.
- **OpenAI-compatible APIs**: provide semantic `prediction` labels (and first-token log-probabilities when supported by the endpoint), contributing to all decision-level metrics even when full answer distributions are unavailable.

Once inference finishes, compute summary metrics with:

```bash
bash scripts/summarize.sh "$HOME/.cache/mm-systemonebias/runs/summary" \
  "$HOME/.cache/mm-systemonebias/runs/d3_mini/predictions.jsonl"
```

See [Metrics and Aggregation](metrics.md) for output schemas and aggregation rules.
