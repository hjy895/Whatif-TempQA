<img width="2487" height="1129" alt="Main figure 1-1" src="https://github.com/user-attachments/assets/46715cd5-9f07-4e4c-89e4-0e93a8ae0635" />

### *What-ifTempQA: A Half-Million Benchmark for Counterfactual Temporal QA and LLM Timeline Hallucinations*

Official repository and dataset release for our EMNLP 2026 paper.  
**Code will be released here soon.**

<p align="center">
  <img src="[https://huggingface.co/datasets/YOUR_USERNAME/What-ifTempQA/resolve/main/assets/whatiftempqa_overview.png](https://huggingface.co/datasets/huggingface/badges/resolve/main/dataset-on-hf-md.svg)]([https://huggingface.co/datasets/YOUR_USERNAME/What-ifTempQA](https://huggingface.co/datasets/hjav/What-ifTempQA))" alt="What-ifTempQA: counterfactual temporal coverage across historical timelines (1900–2025)" width="900"/>
</p>

<p align="center"><i>Counterfactual temporal coverage across historical timelines.</i></p>

*Upload your overview figure to `assets/` on Hugging Face and replace `YOUR_USERNAME` and the filename in the image URL above.*

[![Dataset on Hugging Face](https://huggingface.co/datasets/huggingface/badges/resolve/main/dataset-on-hf-md.svg)]([https://huggingface.co/datasets/YOUR_USERNAME/What-ifTempQA](https://huggingface.co/datasets/hjav/What-ifTempQA))

---

## About

**What-ifTempQA** is a large-scale benchmark for **counterfactual temporal question answering**. Every item embeds an explicit hypothetical edit in the question—*If X were delayed until Y…*, *Suppose X did not occur in Z…*—and provides a **verified** gold answer on the **revised** timeline.

The dataset targets a failure mode we call **timeline hallucination**: models answer as if the real world still holds, even when the question states a clear alternative calendar. What-ifTempQA complements factual temporal QA corpora by measuring whether LLMs can **reason under stated hypotheticals**, not only recall when events occurred.

| | |
|---|---|
| **Paper** | *What-ifTempQA: A Half-Million Benchmark for Counterfactual Temporal QA and LLM Timeline Hallucinations* |
| **Size** | **500,231** question–answer pairs |
| **Batches** | 21 CSV files (`whatif_batch_001.csv` … `whatif_batch_021.csv`) |
| **Question types** | **16** counterfactual families in **4** taxonomy pillars |
| **Knowledge base** | 82 curated historical events (~1879–2024) |
| **Verification** | Deterministic answer engine + duplicate filtering (`curated_verified`) |
| **Code** | **Coming soon** in this repository |

---

## Main contributions

1. **Half-million scale** counterfactual temporal QA with template-verified gold labels.  
2. **16-type taxonomy** covering timeline shifts, ordering/state, quantification, and cross-event dependencies.  
3. **Hallucination-oriented metadata** (`hallucination_target`, `perturbation_type`) for fine-grained error analysis.  
4. **Baseline evaluation** of LLaMA-2, LLaMA-3, Mistral-Instruct, and Gemma models showing large gaps on quantification and aggregate-answer items.

---

## Taxonomy (Figure 2)

| Pillar | Question types |
|--------|----------------|
| **Timeline shift** | Attribute, Duration, Delay, Cancellation |
| **Ordering & state** | Comparison, Time Reordering, Entity State, Role Shift |
| **Quantification** | Counting, Causal Reasoning, Recursive Time, Multihop |
| **Dependency & bounds** | Science Dependency, Technology Dependency, Historical Alternative, Impossible Time |

---

## Example

**Question:** If Olympic Games Munich were delayed until 1976, how many events among Olympic Games Munich and Berlin Wall Construction occur on or before 1975?

**Answer:** `1`

Berlin Wall Construction (1961) counts on the revised timeline; shifted Munich (1976) does not.

| Question | Answer | Type |
|----------|--------|------|
| If Fall of Berlin Wall moved from 1989 to 2015, what is the shift size in years? | 26 | `counterfactual_attribute` |
| If Moon Landing occurred in 1977 rather than 1969, which is earlier: Sputnik Launch (1957) or Moon Landing? | Sputnik Launch | `counterfactual_comparison` |
| If Olympic Games Munich were delayed from 1972 to 1976, what is the revised year of Olympic Games Munich? | 1976 | `counterfactual_delay` |

---

## Data format

Each row is a self-contained CSV record. Core fields:

| Field | Description |
|-------|-------------|
| `question` | Natural-language item with counterfactual premise |
| `answer` | Verified gold label on the revised timeline |
| `question_type` | One of 16 `counterfactual_*` labels |
| `answer_form` | `year`, `count`, `entity_name`, `yes_no`, `before_after`, `duration`, … |
| `difficulty` | 1–5 (shift magnitude) |
| `hop_count` | Reasoning depth (typically 2–4) |
| `domain` | e.g., `science`, `military`, `culture` |
| `perturbation_type` | `delay`, `cancellation`, `failure`, `compound` |
| `hallucination_target` | Expected failure mode if premise is ignored |
| `entities_question` | JSON list of referenced event names |
| `time_span_start` / `time_span_end` | Bounding window for referenced years |
| `source_type` | `curated_verified` for the v5 release |

Full schema: [`data/schema.json`](data/schema.json).

---

## Download

### Hugging Face (recommended)

```python
from datasets import load_dataset

ds = load_dataset("YOUR_USERNAME/What-ifTempQA")
print(ds["train"][0])
```

Replace `YOUR_USERNAME` after the dataset is published.

### Local release

Verified v5 files: **`data/generated_v5/`**

> Do not use `data/generated/` or other older folders—those runs contain heavy duplication.

---

## Repository layout

```
What-iFTempQA/
├── README.md                 # this file
├── data/
│   ├── generated_v5/         # release CSVs (500,231 rows)
│   ├── schema.json
│   └── events_curated.json
├── figures/                  # paper figures
├── data_generation/          # generation & verification (code coming soon)
├── core/
├── src/
└── scripts/
```

---

## Evaluation metrics

We report **Precision**, **Recall**, **F1**, **Answer Containment (C)**, and **Exact Match (EM)**. EM is especially important for counterfactual QA: it penalizes correct-sounding answers that use the **factual** instead of the **hypothetical** timeline.

Evaluation scripts will be added when code is released.

---

## Citation

If you use What-ifTempQA, please cite:

```bibtex
@inproceedings{whatiftempqa2026,
  title     = {What-ifTempQA: A Half-Million Benchmark for Counterfactual Temporal QA and LLM Timeline Hallucinations},
  author    = {TODO},
  booktitle = {Proceedings of the 2026 Conference on Empirical Methods in Natural Language Processing (EMNLP)},
  year      = {2026}
}
```

---

## License

TODO: Add license (e.g., CC BY 4.0).

---

## Contact

For questions or issues, open a GitHub issue or contact the authors listed in the paper.

**Paper:** *What-ifTempQA: A Half-Million Benchmark for Counterfactual Temporal QA and LLM Timeline Hallucinations* (EMNLP 2026).
