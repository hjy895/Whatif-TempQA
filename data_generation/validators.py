"""
Quality validation for generated temporal questions.
"""

from .question_types import TemporalQuestion


class QuestionValidator:
    """Validates question quality against configurable thresholds."""

    def __init__(self, min_confidence: float = 0.7):
        self.min_confidence = min_confidence

    def validate(self, question: TemporalQuestion) -> bool:
        if not question:
            return False
        return (
            self._validate_basic_fields(question)
            and self._validate_content(question)
            and self._validate_quality(question)
        )

    def _validate_basic_fields(self, question: TemporalQuestion) -> bool:
        if not question.question or not question.answer:
            return False
        if not question.id or not question.question_type:
            return False
        if not (10 <= len(question.question) <= 300):
            return False
        if not (1 <= len(question.answer) <= 100):
            return False
        return True

    def _validate_content(self, question: TemporalQuestion) -> bool:
        # Reject unfilled template placeholders and degenerate answers
        bad_tokens = ['{', '}', 'None', 'N/A', 'null']
        if any(t in question.question for t in bad_tokens):
            return False
        if any(t in question.answer for t in bad_tokens):
            return False
        if question.answer.lower().strip() in {'unknown', 'none', '', '0'}:
            return False
        if len(question.question.split()) < 5:
            return False
        return True

    def _validate_quality(self, question: TemporalQuestion) -> bool:
        if question.confidence_score < self.min_confidence:
            return False
        if not (1 <= question.difficulty <= 5):
            return False
        if not (1 <= question.hop_count <= 10):
            return False
        return True

    def _validate_temporal_consistency(self, question: TemporalQuestion) -> bool:
        """Check that time_span_start does not exceed time_span_end."""
        if question.time_span_start and question.time_span_end:
            try:
                start = int(question.time_span_start.split('-')[0])
                end = int(question.time_span_end.split('-')[0])
                return start <= end
            except Exception:
                return False
        return True

    def get_validation_errors(self, question: TemporalQuestion) -> list:
        errors = []
        if not question:
            return ["Question is None"]
        if not question.question:
            errors.append("Missing question text")
        if not question.answer:
            errors.append("Missing answer")
        if question.confidence_score < self.min_confidence:
            errors.append(f"Low confidence: {question.confidence_score}")
        if len(question.question) < 10:
            errors.append("Question too short")
        if len(question.question) > 300:
            errors.append("Question too long")
        return errors
