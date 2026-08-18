"""
Core dataset generator: orchestrates batched question generation.
"""

import json
import random
import gc
from pathlib import Path
from datetime import datetime

from tqdm import tqdm

from .knowledge_base import KnowledgeBase
from .question_types import QuestionType, TemporalQuestion
from .templates import QuestionTemplates
from .validators import QuestionValidator
from ..utils.config import Config
from ..utils.file_utils import CSVWriter


class DatasetGenerator:
    """Generates temporal QA datasets in configurable batches."""

    def __init__(self, config: Config):
        self.config = config
        self.output_dir = Path(config.output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        self.knowledge_base = KnowledgeBase()
        self.templates = QuestionTemplates()
        self.validator = QuestionValidator()

        print(f"Dataset Generator initialized")
        print(f"Target: {config.total_questions:,} questions in {config.num_batches} batches")

    def generate(self):
        """Run the full dataset generation pipeline."""
        start_time = datetime.now()
        self.knowledge_base.load()

        for batch_num in range(1, self.config.num_batches + 1):
            print(f"\nGenerating batch {batch_num}/{self.config.num_batches}")
            self._generate_batch(batch_num)
            gc.collect()

        self._create_summary()
        duration = datetime.now() - start_time
        print(f"\nGeneration complete. Duration: {duration}")

    def _generate_batch(self, batch_num: int):
        batch_file = self.output_dir / f"batch_{batch_num:03d}.csv"
        writer = CSVWriter(batch_file)
        writer.write_header(TemporalQuestion.__annotations__.keys())

        questions_generated = 0
        buffer = []

        question_types = list(QuestionType)
        questions_per_type = self.config.questions_per_batch // len(question_types)

        with tqdm(total=self.config.questions_per_batch, desc=f"Batch {batch_num}") as pbar:
            for qtype in question_types:
                for _ in range(questions_per_type):
                    question = self._generate_question(qtype, batch_num)

                    if question and self.validator.validate(question):
                        buffer.append(question.to_dict())
                        questions_generated += 1
                        pbar.update(1)

                        if len(buffer) >= self.config.batch_write_size:
                            writer.write_batch(buffer)
                            buffer = []

                    if questions_generated >= self.config.questions_per_batch:
                        break
                if questions_generated >= self.config.questions_per_batch:
                    break

        if buffer:
            writer.write_batch(buffer)
        writer.close()
        print(f"Saved: {batch_file} ({questions_generated:,} questions)")

    def _generate_question(self, qtype: QuestionType, batch_id: int) -> TemporalQuestion:
        try:
            if qtype == QuestionType.ATTRIBUTE_EVENT:
                return self._generate_event_attribute(batch_id)
            elif qtype == QuestionType.COMPARISON_EVENT:
                return self._generate_event_comparison(batch_id)
            elif qtype == QuestionType.COUNTING_EVENT:
                return self._generate_event_counting(batch_id)
            else:
                return self._generate_generic_question(qtype, batch_id)
        except Exception:
            return None

    def _generate_event_attribute(self, batch_id: int) -> TemporalQuestion:
        event = random.choice(self.knowledge_base.events)
        template = self.templates.get_event_attribute_template()
        question = template['question'].format(event=event['name'])
        answer = template['answer_func'](event)
        return TemporalQuestion(
            id=f"evt_attr_{batch_id}_{random.randint(100000, 999999)}",
            question=question,
            answer=answer,
            question_type=QuestionType.ATTRIBUTE_EVENT.value,
            difficulty=random.randint(1, 3),
            temporal_granularity="year",
            time_span_start=f"{self.config.start_year}-01-01",
            time_span_end=f"{self.config.end_year}-12-31",
            entities_question=json.dumps([event['name']]),
            countries_question=json.dumps([event.get('location', 'Unknown')]),
            hop_count=1,
            confidence_score=0.95,
            domain=event.get('domain', 'general'),
            requires_calculation=False,
            complexity_score=0.3,
            source_type='curated',
            batch_id=batch_id,
        )

    def _generate_event_comparison(self, batch_id: int) -> TemporalQuestion:
        events = random.sample(self.knowledge_base.events, 2)
        template = self.templates.get_event_comparison_template()
        question = template['question'].format(
            event1=events[0]['name'],
            event2=events[1]['name'],
        )
        answer = template['answer_func'](events[0], events[1])
        return TemporalQuestion(
            id=f"evt_comp_{batch_id}_{random.randint(100000, 999999)}",
            question=question,
            answer=answer,
            question_type=QuestionType.COMPARISON_EVENT.value,
            difficulty=random.randint(2, 4),
            temporal_granularity="year",
            time_span_start=f"{self.config.start_year}-01-01",
            time_span_end=f"{self.config.end_year}-12-31",
            entities_question=json.dumps([e['name'] for e in events]),
            countries_question=json.dumps([e.get('location', 'Unknown') for e in events]),
            hop_count=2,
            confidence_score=0.90,
            domain='comparison',
            requires_calculation=False,
            complexity_score=0.6,
            source_type='curated',
            batch_id=batch_id,
        )

    def _generate_event_counting(self, batch_id: int) -> TemporalQuestion:
        domain = random.choice(['military', 'science', 'politics'])
        start_year = random.randint(1900, 2000)
        end_year = start_year + random.randint(10, 50)
        count = len([
            e for e in self.knowledge_base.events
            if e.get('domain') == domain and start_year <= e.get('year', 0) <= end_year
        ])
        question = f"How many {domain} events occurred between {start_year} and {end_year}?"
        return TemporalQuestion(
            id=f"evt_count_{batch_id}_{random.randint(100000, 999999)}",
            question=question,
            answer=str(count),
            question_type=QuestionType.COUNTING_EVENT.value,
            difficulty=random.randint(3, 4),
            temporal_granularity="decade",
            time_span_start=f"{start_year}-01-01",
            time_span_end=f"{end_year}-12-31",
            entities_question=json.dumps([]),
            countries_question=json.dumps([]),
            hop_count=2,
            confidence_score=0.98,
            domain=domain,
            requires_calculation=True,
            complexity_score=0.7,
            source_type='curated',
            batch_id=batch_id,
        )

    def _generate_generic_question(self, qtype: QuestionType, batch_id: int) -> TemporalQuestion:
        """Placeholder generator for question types not yet fully implemented."""
        return TemporalQuestion(
            id=f"generic_{batch_id}_{random.randint(100000, 999999)}",
            question=f"Sample {qtype.value} question about a historical event",
            answer="Sample answer",
            question_type=qtype.value,
            difficulty=3,
            temporal_granularity="year",
            time_span_start=f"{self.config.start_year}-01-01",
            time_span_end=f"{self.config.end_year}-12-31",
            entities_question=json.dumps([]),
            countries_question=json.dumps([]),
            hop_count=2,
            confidence_score=0.75,
            domain='general',
            requires_calculation=False,
            complexity_score=0.5,
            source_type='generated',
            batch_id=batch_id,
        )

    def _create_summary(self):
        """Write a JSON summary of the generated dataset."""
        import json
        summary = {
            'total_batches': self.config.num_batches,
            'questions_per_batch': self.config.questions_per_batch,
            'total_questions': self.config.total_questions,
            'start_year': self.config.start_year,
            'end_year': self.config.end_year,
            'output_dir': str(self.output_dir),
        }
        summary_path = self.output_dir / "dataset_summary.json"
        with open(summary_path, 'w') as f:
            json.dump(summary, f, indent=2)
        print(f"Summary saved to {summary_path}")
