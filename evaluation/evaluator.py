"""
Main evaluation orchestrator: loads dataset, runs models, collects results.
"""

import json
import numpy as np
import pandas as pd
from datetime import datetime
from pathlib import Path

from .model_manager import ModelManager
from .metrics import TemporalQAMetrics
from .prompt_builder import PromptBuilder
from .result_analyzer import ResultAnalyzer
from ..utils.config import EvaluationConfig


class ModelEvaluator:
    """Evaluates language models on a temporal QA dataset."""

    def __init__(self, config: EvaluationConfig):
        self.config = config
        self.output_dir = Path(config.output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        self.model_manager = ModelManager()
        self.metrics = TemporalQAMetrics()
        self.prompt_builder = PromptBuilder()
        self.analyzer = ResultAnalyzer()

        print(f"Evaluator initialized")
        print(f"Dataset: {config.dataset_path}")
        print(f"Models: {len(config.models)}")
        print(f"Sample size: {config.sample_size}")

    def evaluate(self) -> pd.DataFrame:
        df = self._load_dataset()
        eval_df = self._create_sample(df)
        examples = self._create_examples(df)

        print(f"Evaluation sample: {len(eval_df)} questions")
        print(f"Few-shot pool: {len(examples)} examples")

        all_results = []
        for model_name in self.config.models:
            print(f"\nEvaluating: {model_name}")
            if self.model_manager.load_model(model_name):
                results = self._evaluate_model(eval_df, examples, model_name)
                all_results.extend(results)
                self.model_manager.unload_model()
            else:
                print(f"  Skipping {model_name} (failed to load)")

        results_df = pd.DataFrame(all_results)
        if not results_df.empty:
            self._save_results(results_df)
            self.analyzer.generate_report(results_df, self.output_dir)

        return results_df

    def _load_dataset(self) -> pd.DataFrame:
        print(f"Loading dataset: {self.config.dataset_path}")
        df = pd.read_csv(self.config.dataset_path)
        df = df.dropna(subset=['question', 'answer'])
        df = df[df['question'].str.len() > 10]
        df = df[df['answer'].str.len() > 0]
        print(f"Loaded {len(df):,} questions")
        return df

    def _create_sample(self, df: pd.DataFrame) -> pd.DataFrame:
        """Stratified sample across question types for diversity."""
        question_types = df['question_type'].unique()[:5]
        samples_per_type = max(1, self.config.sample_size // len(question_types))
        samples = []
        for qtype in question_types:
            type_df = df[df['question_type'] == qtype]
            if len(type_df) > 0:
                samples.append(type_df.sample(n=min(samples_per_type, len(type_df)), random_state=42))
        return pd.concat(samples).reset_index(drop=True)

    def _create_examples(self, df: pd.DataFrame) -> list:
        """Select high-quality, low-difficulty examples for few-shot prompts."""
        pool = df.copy()
        if 'confidence_score' in df.columns:
            pool = pool[pool['confidence_score'] >= 0.9]
        if 'difficulty' in df.columns:
            pool = pool[pool['difficulty'] <= 2]

        examples = []
        for qtype in df['question_type'].unique()[:5]:
            type_pool = pool[pool['question_type'] == qtype]
            if len(type_pool) > 0:
                for _, row in type_pool.sample(n=min(10, len(type_pool)), random_state=42).iterrows():
                    examples.append({
                        'question': row['question'],
                        'answer': row['answer'],
                        'type': row['question_type'],
                    })
        return examples[:50]

    def _evaluate_model(self, eval_df: pd.DataFrame, examples: list, model_name: str) -> list:
        results = []
        for shots in range(self.config.max_shots + 1):
            print(f"  Testing {shots}-shot...")
            shot_examples = examples[:shots] if shots > 0 else []

            for _, row in eval_df.iterrows():
                try:
                    prompt = self.prompt_builder.create_prompt(row['question'], shot_examples)
                    prediction = self.model_manager.generate(prompt)
                    metrics = self.metrics.calculate_all_metrics(prediction, row['answer'])
                    results.append({
                        'model': model_name,
                        'shots': shots,
                        'question_type': row.get('question_type', 'unknown'),
                        'domain': row.get('domain', 'general'),
                        'question': row['question'],
                        'true_answer': row['answer'],
                        'predicted_answer': prediction,
                        **metrics,
                    })
                except Exception as e:
                    print(f"    Error: {e}")
                    continue

            shot_results = [r for r in results if r['shots'] == shots]
            if shot_results:
                avg_f1 = np.mean([r['f1'] for r in shot_results])
                print(f"    {shots}-shot F1: {avg_f1:.3f}")

        return results

    def _save_results(self, results_df: pd.DataFrame):
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

        results_file = self.output_dir / f"evaluation_results_{timestamp}.csv"
        results_df.to_csv(results_file, index=False)

        summary = {
            'evaluation_date': timestamp,
            'dataset': str(self.config.dataset_path),
            'models_evaluated': list(results_df['model'].unique()),
            'total_predictions': len(results_df),
            'sample_size': self.config.sample_size,
            'config': self.config.__dict__,
        }
        summary_file = self.output_dir / f"evaluation_summary_{timestamp}.json"
        with open(summary_file, 'w') as f:
            json.dump(summary, f, indent=2)

        print(f"Results saved to {results_file}")
        print(f"Summary saved to {summary_file}")
