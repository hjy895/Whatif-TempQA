# What-ifTempQA

[![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)](https://python.org)

**What-ifTempQA** is a large-scale counterfactual temporal question answering benchmark containing **500K+** verified question-answer pairs designed to evaluate temporal reasoning capabilities in large language models.

## Key Features

- **Scale**: 500,231 high-quality counterfactual temporal QA pairs
- **Coverage**: Spans events from approximately 1900 to 2025
- **Diversity**: Covers events, entities, and timelines across multiple domains
- **Complexity**: 16 distinct counterfactual question types organised into 4 families
- **Multi-hop**: Supports complex multi-hop inferential reasoning tasks
- **Verified Labels**: Deterministic answer engine with independent verification

## Question Types

| **Timeline Shifts** | **Ordering & State** | **Quantification** | **Dependencies & Bounds** |
|---------------------|----------------------|--------------------|---------------------------|
| Attribute           | Comparison           | Counting           | Science Dependency        |
| Duration            | Time Reordering      | Causal Reasoning   | Technology Dependency     |
| Delay               | Entity State         | Recursive Time     | Impossible Time           |
| Cancellation        | Role Shift           | Multihop           | Historical Alternative    |

## Repository Structure

```
What-ifTempQA/
├── README.md
├── requirements.txt
├── LICENSE
├── setup.py
├── .gitignore
├── src/
│   ├── main_generator.py          # Dataset generation entry point
│   ├── main_evaluator.py          # Model evaluation entry point
│   ├── data_generation/           # Dataset generation modules
│   │   ├── __init__.py
│   │   ├── core_generator.py      # Core generator
│   │   ├── knowledge_base.py      # Historical knowledge base
│   │   ├── question_types.py      # Question type definitions
│   │   ├── templates.py           # Question templates
│   │   └── validators.py          # Quality validators
│   ├── evaluation/                # Evaluation modules
│   │   ├── evaluator.py           # Main evaluator
│   │   ├── model_manager.py       # Model loading/inference
│   │   ├── metrics.py             # Evaluation metrics
│   │   ├── prompt_builder.py      # Few-shot prompt construction
│   │   └── result_analyzer.py     # Result analysis and reporting
│   └── utils/                     # Utility modules
│       ├── __init__.py
│       ├── config.py              # Configuration classes
│       ├── file_utils.py          # File I/O helpers
│       └── logging_utils.py       # Logging setup
├── data/
│   ├── sample_data.csv            # Sample dataset for testing
│   └── schema.json                # Dataset schema
├── samples/                       # Sample batch files
│   ├── temporal_qa_batch_1.csv
│   ├── temporal_qa_batch_2.csv
│   └── temporal_qa_batch_3.csv
└── tests/
    ├── __init__.py
    └── test_generation.py         # Unit tests
```

## Installation

```bash
git clone https://github.com/hjy895/whatif-Tempqa.git
cd whatif-Tempqa
pip install -r requirements.txt
```

## Usage

### Generate Dataset

```bash
python src/main_generator.py --output_dir data/generated --num_batches 5 --verbose
```

### Evaluate Models

```bash
python src/main_evaluator.py --dataset data/sample_data.csv --output_dir results --sample_size 50 --max_shots 3
```

### Run Tests

```bash
pytest tests/ -v
```

## Dataset Access

The full dataset is available on Hugging Face: [hjav/What-ifTempQA](https://huggingface.co/datasets/hjav/What-ifTempQA)

Sample batch files are included in the `/samples` directory for quick experimentation.

## Dataset Statistics

| Metric | Value |
|--------|-------|
| **Total Questions** | 500,231 |
| **Question Types** | 16 Categories |
| **Time Range** | 1900–2025 |
| **Domains** | History, Science, Technology, Politics, Culture |
| **Language** | English |
| **Format** | CSV |

## Evaluation Metrics

- **Precision (P)**: Token-level precision
- **Recall (R)**: Token-level recall
- **F1 Score**: Harmonic mean of precision and recall
- **Containment (C)**: Whether the key answer token appears in the output
- **Exact Match (EM)**: Full normalised match to the gold answer

## License

This project is licensed under the MIT License. See [LICENSE](LICENSE) for details.

## Citation

If you use this dataset in your research, please cite:

```bibtex
@inproceedings{whatiftempqa2026,
  title={What-ifTempQA: A Half-Million Benchmark for Counterfactual Temporal QA and LLM Timeline Hallucinations},
  year={2026}
}
```

## Contributing

Contributions are welcome. Please open an issue or submit a pull request.
