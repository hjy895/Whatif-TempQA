"""
Aggregation and reporting of evaluation results.
"""

import pandas as pd
import numpy as np
from pathlib import Path


class ResultAnalyzer:
    """Analyzes evaluation results and produces summary reports."""

    def generate_report(self, results_df: pd.DataFrame, output_dir: Path):
        print("\nGenerating evaluation report...")
        self._print_results_table(results_df)
        analysis = self._analyze_results(results_df)
        self._print_analysis(analysis)
        self._save_report(results_df, analysis, output_dir)

    def _print_results_table(self, results_df: pd.DataFrame):
        print("\n" + "=" * 90)
        print("Evaluation Results")
        print("=" * 90)

        table = results_df.groupby(['model', 'shots']).agg({
            'precision': 'mean',
            'recall': 'mean',
            'f1': 'mean',
            'containment': 'mean',
            'exact_match': 'mean',
        }).round(3)

        print(f"{'Model':<30} {'Shots':<6} {'Precision':<10} {'Recall':<10} {'F1':<10} {'Containment':<12} {'EM':<10}")
        print("-" * 88)

        for (model, shots) in table.index:
            model_short = model.split('/')[-1][:25]
            row = table.loc[(model, shots)]
            print(f"{model_short:<30} {shots:<6} {row['precision']:<10.3f} {row['recall']:<10.3f} "
                  f"{row['f1']:<10.3f} {row['containment']:<12.3f} {row['exact_match']:<10.3f}")

    def _analyze_results(self, results_df: pd.DataFrame) -> dict:
        analysis = {}

        model_stats = results_df.groupby(['model', 'shots'])['f1'].mean().reset_index()
        analysis['best_configs'] = model_stats.sort_values('f1', ascending=False).head(10).to_dict('records')

        improvements = []
        for model in results_df['model'].unique():
            shots_data = results_df[results_df['model'] == model].groupby('shots')['f1'].mean()
            if 0 in shots_data.index and len(shots_data) > 1:
                improvements.append({
                    'model': model,
                    'zero_shot_f1': shots_data[0],
                    'best_f1': shots_data.max(),
                    'improvement': shots_data.max() - shots_data[0],
                })
        analysis['improvements'] = sorted(improvements, key=lambda x: x['improvement'], reverse=True)[:5]

        analysis['overall_stats'] = results_df.groupby('shots').agg({
            'f1': ['mean', 'std'],
            'exact_match': ['mean', 'std'],
            'precision': ['mean', 'std'],
            'recall': ['mean', 'std'],
        }).round(3)

        analysis['question_type_performance'] = results_df.groupby('question_type').agg({
            'f1': 'mean',
            'exact_match': 'mean',
        }).round(3).sort_values('f1', ascending=False)

        return analysis

    def _print_analysis(self, analysis: dict):
        print("\n## Analysis")
        print("-" * 50)

        print("Top 5 configurations:")
        for i, cfg in enumerate(analysis['best_configs'][:5]):
            print(f"  {i+1}. {cfg['model'].split('/')[-1][:20]} ({cfg['shots']}-shot): F1={cfg['f1']:.3f}")

        if analysis['improvements']:
            print("\nFew-shot learning improvements (0-shot → best):")
            for imp in analysis['improvements']:
                print(f"  {imp['model'].split('/')[-1][:20]}: {imp['zero_shot_f1']:.3f} → "
                      f"{imp['best_f1']:.3f} (+{imp['improvement']:.3f})")

        print("\nPerformance by question type:")
        qtype_perf = analysis['question_type_performance']
        for qtype in qtype_perf.head(5).index:
            print(f"  {qtype}: F1={qtype_perf.loc[qtype, 'f1']:.3f}, "
                  f"EM={qtype_perf.loc[qtype, 'exact_match']:.3f}")

        print("\nPerformance by shot count:")
        overall = analysis['overall_stats']
        for shots in sorted(overall.index):
            f1_mean = overall.loc[shots, ('f1', 'mean')]
            f1_std = overall.loc[shots, ('f1', 'std')]
            print(f"  {shots}-shot: F1={f1_mean:.3f} ± {f1_std:.3f}")

    def _save_report(self, results_df: pd.DataFrame, analysis: dict, output_dir: Path):
        report_file = output_dir / "evaluation_report.txt"
        with open(report_file, 'w') as f:
            f.write("Temporal QA Evaluation Report\n")
            f.write("=" * 40 + "\n\n")
            f.write(f"Total predictions: {len(results_df)}\n")
            f.write(f"Models evaluated: {len(results_df['model'].unique())}\n")
            f.write(f"Question types: {len(results_df['question_type'].unique())}\n\n")

            f.write("Top Configurations:\n")
            for i, cfg in enumerate(analysis['best_configs'][:10]):
                f.write(f"  {i+1}. {cfg['model']} ({cfg['shots']}-shot): F1={cfg['f1']:.3f}\n")

            f.write("\nFew-shot Improvements:\n")
            for imp in analysis['improvements']:
                f.write(f"  {imp['model']}: {imp['zero_shot_f1']:.3f} → "
                        f"{imp['best_f1']:.3f} (+{imp['improvement']:.3f})\n")

        print(f"Report saved to {report_file}")
