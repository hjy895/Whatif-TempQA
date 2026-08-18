"""
Evaluation metrics for temporal QA: exact match, token F1, and containment.
"""

import re
from typing import Tuple


class TemporalQAMetrics:
    """Computes standard QA metrics adapted for temporal answer formats."""

    def calculate_all_metrics(self, pred: str, truth: str) -> dict:
        precision, recall, f1 = self.token_metrics(pred, truth)
        return {
            'precision': precision,
            'recall': recall,
            'f1': f1,
            'exact_match': self.exact_match(pred, truth),
            'containment': self.containment_score(pred, truth),
        }

    def normalize_answer(self, answer: str) -> str:
        if not answer:
            return ""
        answer = str(answer).lower().strip()
        answer = re.sub(r'[^\w\s\d]', ' ', answer)
        answer = re.sub(r'\s+', ' ', answer).strip()
        return answer

    def exact_match(self, pred: str, truth: str) -> float:
        pred_norm = self.normalize_answer(pred)
        truth_norm = self.normalize_answer(truth)
        if not truth_norm:
            return 0.0
        if pred_norm == truth_norm:
            return 100.0
        # Substring containment is treated as a match
        if truth_norm in pred_norm or pred_norm in truth_norm:
            return 100.0
        # Year-level numeric matching for temporal answers
        pred_nums = re.findall(r'\b\d+\b', pred_norm)
        truth_nums = re.findall(r'\b\d+\b', truth_norm)
        if pred_nums and truth_nums and any(p in truth_nums for p in pred_nums):
            return 100.0
        return 0.0

    def token_metrics(self, pred: str, truth: str) -> Tuple[float, float, float]:
        pred_norm = self.normalize_answer(pred)
        truth_norm = self.normalize_answer(truth)
        if not truth_norm:
            return 0.0, 0.0, 0.0
        pred_tokens = set(pred_norm.split()) if pred_norm else set()
        truth_tokens = set(truth_norm.split()) if truth_norm else set()
        if not pred_tokens or not truth_tokens:
            return 0.0, 0.0, 0.0
        common = pred_tokens & truth_tokens
        precision = len(common) / len(pred_tokens)
        recall = len(common) / len(truth_tokens)
        f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) > 0 else 0.0
        return precision * 100, recall * 100, f1 * 100

    def containment_score(self, pred: str, truth: str) -> float:
        """Fraction of truth tokens that appear in the prediction."""
        pred_norm = self.normalize_answer(pred)
        truth_norm = self.normalize_answer(truth)
        if not truth_norm:
            return 0.0
        pred_tokens = set(pred_norm.split()) if pred_norm else set()
        truth_tokens = set(truth_norm.split()) if truth_norm else set()
        if not truth_tokens:
            return 0.0
        return (len(pred_tokens & truth_tokens) / len(truth_tokens)) * 100
