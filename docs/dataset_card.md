---
configs:
  - config_name: default
    data_files:
      - split: test
        path: test-*.parquet
---

# MM-SystemOneBias Dataset Card

**MM-SystemOneBias** is a controlled evaluation benchmark for measuring decision and probability stability in multimodal System One decision models. It aggregates **4,228 source instances** from **9 public datasets** into **5,323 original instances** across **11 task scenarios** and **3 decision interfaces** (*choice*, *yes/no judgment*, and *scoring*). Applying **6 single-factor interventions** yields **76,439 intervened instances**, for **81,762 total model inputs** in the public test split.

---

## Task Scenarios and Decision Interfaces

Two source datasets support multiple decision interfaces (*image safety moderation* via LlavaGuard and *generated-image quality* via AGIQA-3K), resulting in 14 task–interface combinations across the 11 task scenarios:

| Task Family | Task Scenario | Source Dataset | Decision Interface(s) | Original Instances |
| --- | --- | --- | --- | ---: |
| GUI & robotics | GUI next-action selection | Multimodal-Mind2Web | Choice | 400 |
| GUI & robotics | GUI task completion judgment | AgentRewardBench | Yes/no judgment | 368 |
| GUI & robotics | Robot next-action selection | Robo2VLM-1 | Choice | 400 |
| GUI & robotics | Robot task success judgment | Robo2VLM-1 | Yes/no judgment | 347 |
| GUI & robotics | Robotic grasping stage | Robo2VLM-1 | Choice | 482 |
| Documents & safety | Document type classification | RVL-CDIP | Choice | 342 |
| Documents & safety | Image safety moderation | LlavaGuard | Yes/no judgment, Choice | 590 (295 each) |
| Industrial inspection | Industrial anomaly detection | MVTec AD | Yes/no judgment | 394 |
| Preference & quality | Image-grounded response preference | VL-RewardBench | Choice | 400 |
| Preference & quality | Generated-image preference | HPDv2 | Choice | 400 |
| Preference & quality | Generated-image quality | AGIQA-3K | Scoring, Yes/no judgment, Choice | 1,200 (400 each) |
| **Total** | **11 scenarios** | **9 datasets** | **3 interfaces (14 combinations)** | **5,323** |

---

## Dataset Structure

Each packed row in the Parquet test split corresponds to one original instance and bundles all of its intervened instances along with a deduplicated `List(Image)` image pool:

- **Task & alignment fields**: `sample_id`, `root_id`, `source_group`, `question_group`, `scenario`, `task_type` (`choice`, `predicate`, or `score`), `question`, `options`, `gold` (reference semantic answer ID), and `template`.
- **Intervention fields**: `bias` (`B1`–`B6` or `null`), `version`, `intervention` (readable condition name), and `variants` (list of intervened records inheriting the same structure).
- **Image references**: `images` (deduplicated `List(Image)` pool), `image_names`, `input_image_indices`, and per-option `image_index` when options contain images.
- **Metadata**: `metadata` (JSON-serialized source and task metadata) and `evaluation_metadata` (including `mos_quality`, quality grade levels, `injected_id`, and `target_ids` when applicable).

All intervened instances pass a two-stage verification pipeline consisting of deterministic programmatic checks (candidate identity, grade correspondence, and target mapping) and model-based screening for visual factuality and background irrelevance.

---

## Usage

Load the expanded canonical inference records directly with the repository loader (which unpacks variants and materializes deduplicated images under `~/.cache/mm-systemonebias/images`):

```python
from mm_systemonebias.data import load_records

# Loads from Hugging Face Hub, or pass dataset_path="/path/to/local/parquet"
records = load_records()
```

Or inspect the raw packed Parquet split via Hugging Face `datasets`:

```python
from datasets import load_dataset

dataset = load_dataset("ALIENS232/MM-SystemOneBias", split="test")
```

---

## Sources and Licensing

The benchmark draws on nine public multimodal datasets whose upstream licenses and attribution requirements differ:

| Source Dataset | Upstream Link | Retrieved License Status |
| --- | --- | --- |
| **Multimodal-Mind2Web** | [Hugging Face](https://huggingface.co/datasets/osunlp/Multimodal-Mind2Web) | OpenRAIL |
| **AgentRewardBench** | [Hugging Face](https://huggingface.co/datasets/McGill-NLP/agent-reward-bench) | No license stated |
| **Robo2VLM-1** | [Hugging Face](https://huggingface.co/datasets/keplerccc/Robo2VLM-1) | Apache-2.0 |
| **RVL-CDIP** | [Hugging Face](https://huggingface.co/datasets/aharley/rvl_cdip) | Other terms |
| **LlavaGuard** | [Hugging Face](https://huggingface.co/datasets/AIML-TUDA/LlavaGuard); [Image repository](https://huggingface.co/datasets/AIML-TUDA/smid) | Apache-2.0 for dataset metadata; images include a separate source and external URLs |
| **MVTec AD** | [Official page](https://www.mvtec.com/research-teaching/datasets/mvtec-ad) | Mirrored material is marked CC BY-NC-SA 4.0; official rights basis remains to be confirmed |
| **VL-RewardBench** | [Hugging Face](https://huggingface.co/datasets/MMInstruction/VL-RewardBench) | MIT |
| **HPDv2** | [Hugging Face](https://huggingface.co/datasets/ymhao/HPDv2) | Apache-2.0 |
| **AGIQA-3K** | [GitHub repository](https://github.com/lcysyzxdxc/AGIQA-3k-Database) | License status to be confirmed |

No unified license is assigned to the assembled dataset. Please review and comply with each upstream source dataset's terms and attribution requirements before redistribution.
