# Models, Versions, and Runtime Profiles

The benchmark evaluates **8 multimodal System One decision models** (7 open-weight models and 1 hosted API service) alongside **5 general-purpose vision-language models (VLMs)** serving as matched backbones.

---

## Model Registry and Code-Alias Mapping

In the codebase, YAML configuration filenames and `model_id` values use short identifiers (for example, `qwen_27b`, `lfm_3b`, `gemma_12b`, `autotrust_jev_27b`, and `typesafe_jev`). The table below records the exact correspondence between each paper model name, its repository config ID, runtime profile, matched backbone, and pinned source revision:

| Paper Name | Model Group | Config ID | Runtime Profile | Matched Backbone | Source / Exact Version | Pinned Revision |
| --- | --- | --- | --- | --- | --- | --- |
| **JEV-27B-VL** | Decision Model | `autotrust_jev_27b` | `autotrust` | Qwen3.8-27B (`qwen_27b`) | [`autotrust/JEV-27B-VL`](https://huggingface.co/autotrust/JEV-27B-VL) | `9e53b8a809d2ab98e3e6a55b2b2dbf94b4c1e84a` |
| **Clef** | Decision Model | `clef_27b` | `clef` | Qwen3.8-27B (`qwen_27b`) | [`Cloudflare/clef`](https://huggingface.co/Cloudflare/clef) | `0b331204bb13fbd2ca93a64df1956f5b55478ce5` |
| **d3** | Decision Model | `d3_27b` | `d3` | Qwen3.8-27B (`qwen_27b`) | [`vllm-sr/d3`](https://huggingface.co/vllm-sr/d3) | `225be8b1e2270becab69c5fe05b26f99cbbdd66b` |
| **Clef-Flash** | Decision Model | `clef_flash_9b` | `clef` | Qwen3.5-9B (`qwen_9b`) | [`Cloudflare/clef-flash`](https://huggingface.co/Cloudflare/clef-flash) | `8b2e5fd17c09fd49bc2880b805d5323ec7ff4ff3` |
| **d3-mini** | Decision Model | `d3_mini` | `d3` | Qwen3.5-4B (`qwen_4b`) | [`vllm-sr/d3-mini`](https://huggingface.co/vllm-sr/d3-mini) | `97dd054df4b3e0b3beb6af23ab8c82b7c0052e0e` |
| **d1-3B** | Decision Model | `d1_3b` | `d1` | LFM2.5-VL-3B (`lfm_3b`) | [`LiquidAI/d1-3B`](https://huggingface.co/LiquidAI/d1-3B) | `051bcc464b01b9f92942b364d9586b0ef5912432` |
| **Jev-Omni** | Decision Model | `jev_omni` | `gemma` | Gemma 4 12B IT (`gemma_12b`) | [`akhilaaa3/Jev-Omni`](https://huggingface.co/akhilaaa3/Jev-Omni) | `fee89b076e2263bfa8bce8d34142caa4ca733a48` |
| **TypeSafe Jev** | Decision Model (API) | `typesafe_jev` | `core` | — | [TypeSafe Jev](https://typesafe.ai/blog/introducing-system-one-models-and-jev) (`jev-1.14.0`) | Hosted API |
| **Qwen3.8-27B** | General-Purpose VLM | `qwen_27b` | `vlm` | — | [`Qwen/Qwen3.8-27B`](https://huggingface.co/Qwen/Qwen3.8-27B) | `1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0` |
| **Qwen3.5-9B** | General-Purpose VLM | `qwen_9b` | `vlm` | — | [`Qwen/Qwen3.5-9B`](https://huggingface.co/Qwen/Qwen3.5-9B) | `c202236235762e1c871ad0ccb60c8ee5ba337b9a` |
| **Qwen3.5-4B** | General-Purpose VLM | `qwen_4b` | `vlm` | — | [`Qwen/Qwen3.5-4B`](https://huggingface.co/Qwen/Qwen3.5-4B) | `851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a` |
| **LFM2.5-VL-3B** | General-Purpose VLM | `lfm_3b` | `vlm` | — | [`LiquidAI/LFM2.5-VL-3B`](https://huggingface.co/LiquidAI/LFM2.5-VL-3B) | `35a118d938ce6d123ac2d371649f24a8efb69058` |
| **Gemma 4 12B IT** | General-Purpose VLM | `gemma_12b` | `gemma` | — | [`google/gemma-4-12B-it`](https://huggingface.co/google/gemma-4-12B-it) | `707f0a3b8a3c7ad586ed01e27eafbad8a27dd0f7` |

> **Note:** In code, `d3_27b` refers to `d3`, `autotrust_jev_27b` refers to `JEV-27B-VL`, `qwen_27b` / `qwen_9b` / `qwen_4b` refer to `Qwen3.8-27B` / `Qwen3.5-9B` / `Qwen3.5-4B`, `lfm_3b` refers to `LFM2.5-VL-3B`, `gemma_12b` refers to `Gemma 4 12B IT`, and `typesafe_jev` targets model version `jev-1.14.0`. In addition, [configs/models/openai_compatible.yaml](../configs/models/openai_compatible.yaml) provides a generic adapter for external OpenAI-compatible vision endpoints.

---

## Runtime Profiles

Each profile in `configs/environments/` installs pinned dependencies into `~/.cache/mm-systemonebias/envs/<profile>/`:

| Profile | Target Models |
| --- | --- |
| `core` | Dataset download, metric aggregation, API inference (`TypeSafe Jev`, OpenAI-compatible endpoints, and AutoTrust client) |
| `vlm` | `Qwen3.8-27B` (`qwen_27b`), `Qwen3.5-9B` (`qwen_9b`), `Qwen3.5-4B` (`qwen_4b`), `LFM2.5-VL-3B` (`lfm_3b`) |
| `gemma` | `Gemma 4 12B IT` (`gemma_12b`), `Jev-Omni` (`jev_omni`) |
| `clef` | `Clef` (`clef_27b`), `Clef-Flash` (`clef_flash_9b`) |
| `d1` | `d1-3B` (`d1_3b`) |
| `d3` | `d3` (`d3_27b`), `d3-mini` (`d3_mini`) |
| `autotrust` | `JEV-27B-VL` (`autotrust_jev_27b`) local vLLM service |

Example setup and inference for **d3-mini** (`d3_mini`):

```bash
bash scripts/setup.sh d3
source "$HOME/.cache/mm-systemonebias/envs/d3/bin/activate"
bash scripts/download_model.sh --config configs/models/d3_mini.yaml
bash scripts/infer.sh --config configs/models/d3_mini.yaml \
  --dataset-path "$HOME/.cache/mm-systemonebias/dataset" \
  --output-dir "$HOME/.cache/mm-systemonebias/runs/d3_mini"
```

Model checkpoints are downloaded at their pinned revisions and cached under `~/.cache/mm-systemonebias/models/`.

---

## AutoTrust Local Service (`JEV-27B-VL`)

**JEV-27B-VL** (`autotrust_jev_27b`) runs through a local vLLM service configured in BF16 with `tensor_parallel_size: 2`, `max_model_len: 32768`, `max_num_seqs: 16`, `gpu_memory_utilization: 0.9`, and thinking disabled (`thinking: "off"`).

Start the server in one terminal:

```bash
bash scripts/setup.sh autotrust
source "$HOME/.cache/mm-systemonebias/envs/autotrust/bin/activate"
bash scripts/download_model.sh --config configs/models/autotrust_jev_27b.yaml
CUDA_VISIBLE_DEVICES=0,1 bash scripts/serve_autotrust.sh \
  --config configs/models/autotrust_jev_27b.yaml --port 8000
```

Run inference against the local service from a second terminal using the `core` profile:

```bash
bash scripts/setup.sh core
source "$HOME/.cache/mm-systemonebias/envs/core/bin/activate"
bash scripts/infer.sh --config configs/models/autotrust_jev_27b.yaml \
  --set base_url=http://127.0.0.1:8000 \
  --dataset-path "$HOME/.cache/mm-systemonebias/dataset" \
  --output-dir "$HOME/.cache/mm-systemonebias/runs/autotrust_jev_27b"
```

---

## Hosted APIs (`TypeSafe Jev` & OpenAI-Compatible)

**TypeSafe Jev** (`typesafe_jev`, version `jev-1.14.0`) uses the `core` profile and reads `TYPESAFE_BASE_URL` and `TYPESAFE_API_KEY`:

```bash
bash scripts/setup.sh core
source "$HOME/.cache/mm-systemonebias/envs/core/bin/activate"
export TYPESAFE_BASE_URL="https://service.example"
export TYPESAFE_API_KEY="..."
bash scripts/infer.sh --config configs/models/typesafe_jev.yaml \
  --dataset-path "$HOME/.cache/mm-systemonebias/dataset" \
  --output-dir "$HOME/.cache/mm-systemonebias/runs/typesafe_jev"
```

For any external OpenAI-compatible vision endpoint, configure `configs/models/openai_compatible.yaml` via environment variables:

```bash
export OPENAI_BASE_URL="https://api.example/v1"
export OPENAI_API_KEY="..."
export OPENAI_MODEL="vision-model"
bash scripts/infer.sh --config configs/models/openai_compatible.yaml \
  --dataset-path "$HOME/.cache/mm-systemonebias/dataset" \
  --output-dir "$HOME/.cache/mm-systemonebias/runs/openai_compatible"
```
