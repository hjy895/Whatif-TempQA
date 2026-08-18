"""
Prompt construction for zero-shot and few-shot temporal QA evaluation.
"""


class PromptBuilder:
    """Builds evaluation prompts in multiple formats."""

    def create_prompt(self, question: str, examples: list = None) -> str:
        if not examples:
            return self._zero_shot(question)
        return self._few_shot(question, examples)

    def _zero_shot(self, question: str) -> str:
        return (
            "Answer this question with a short, precise answer (1-3 words maximum).\n\n"
            f"Question: {question}\n"
            "Answer:"
        )

    def _few_shot(self, question: str, examples: list) -> str:
        prompt = "Answer questions with short, precise answers (1-3 words maximum). Examples:\n\n"
        for ex in examples:
            prompt += f"Question: {ex['question']}\nAnswer: {ex['answer']}\n\n"
        prompt += f"Question: {question}\nAnswer:"
        return prompt

    def create_instruction_prompt(self, question: str, examples: list = None) -> str:
        instruction = (
            "You are a helpful assistant that answers temporal questions accurately. "
            "Provide short, factual answers."
        )
        if examples:
            prompt = f"{instruction}\n\nExamples:\n"
            for ex in examples:
                prompt += f"Q: {ex['question']}\nA: {ex['answer']}\n\n"
            prompt += f"Q: {question}\nA:"
        else:
            prompt = f"{instruction}\n\nQ: {question}\nA:"
        return prompt

    def create_chat_prompt(self, question: str, examples: list = None) -> str:
        if examples:
            prompt = "Here are some example questions and answers:\n\n"
            for ex in examples:
                prompt += f"Human: {ex['question']}\nAssistant: {ex['answer']}\n\n"
            prompt += f"Human: {question}\nAssistant:"
        else:
            prompt = f"Human: {question}\nAssistant:"
        return prompt
