import json
import logging
import os
from concurrent.futures import ThreadPoolExecutor

from openai import OpenAI

logger = logging.getLogger(__name__)


SYSTEM_PROMPT = """You are an expert answerability evaluator.
Your single task is to judge how answerable a question is based only on one provided sentence.

Output Rules:
- Output exactly one JSON object.
- JSON must strictly match this schema:
  {"score": <integer in [1,5]>}
- Valid scores and meaning:
  1 → Definitely answerable: sentence directly and fully answers the question.
  2 → Very likely answerable: clear, minor inference acceptable, no ambiguity.
  3 → Unclear/partial: incomplete or ambiguous support.
  4 → Probably not answerable: little/no support from the sentence.
  5 → Definitely not answerable: no relevant information at all.

Constraints:
- Use only the provided sentence. Ignore outside knowledge, assumptions, or context.
- Do not include explanations, reasoning, comments, or extra fields.
- Do not output natural language or markdown.
- Output must be valid JSON parsable by a strict JSON parser.
- If uncertain, always choose: {"score": 3}

Breaking any rule = invalid output.
"""


class AnswerabilityEvaluator:
    def __init__(self, model_name: str = "gpt-4o-mini", api_key: str | None = None) -> None:
        """Initialize the AnswerabilityEvaluator.

        Args:
            model_name: Name of the model to use with the API.
            api_key: Optional API key. Falls back to env var `OPENAI_API_KEY`.

        Returns:
            None
        """
        self.model = model_name
        self.client = OpenAI(api_key=api_key or os.getenv("OPENAI_API_KEY"))

    def get_rewards(self, questions: list[str], new_sentences: list[str]) -> list[float]:
        """Compute rewards via batched inference using a structured JSON score.

        Each (question, new_sentence) pair is rated on how answerable the question becomes
        using ONLY information from the new_sentence, on a 1–5 integer scale where 1 means
        definitely answerable and 5 means definitely not answerable. The score is then mapped
        linearly to a reward in [0.0, 1.0] via: reward = (score - 1) / 4.

        Args:
            questions: List of questions, length N.
            new_sentences: List of candidate new sentences (same length as questions).

        Returns:
            A list of float rewards of length N in [0.0, 1.0].
        """
        if len(questions) != len(new_sentences):
            raise ValueError("questions and new_sentences must have the same length")

        if not questions:
            return []

        def infer_one(question: str, sentence: str) -> float:
            """
            Infer the reward for a single (question, sentence) pair.

            Args:
                question: The question to evaluate.
                sentence: The sentence to rate for answerability.

            Returns:
                Reward in [0.0, 1.0], mapped from the integer score in [1, 5]
                returned by the model using reward = (score - 1) / 4.
            """
            user_prompt = f"""You are an expert answerability evaluator.
Your task is to rate how answerable the given question is using ONLY the provided sentence.

Output Format:
- Always output exactly one JSON object.
- JSON must strictly match this schema: {"score": <integer in [1,5]>}

Scale:
1 → Definitely answerable (sentence directly and completely answers).
2 → Likely answerable (minor, unambiguous inference is acceptable).
3 → Unclear or partial (ambiguous or insufficient support).
4 → Likely not answerable (sentence provides little or no support).
5 → Definitely not answerable (sentence provides no relevant information).

Guidelines:
- Use only the given sentence; ignore outside knowledge or context.
- If uncertain, default to {"score": 3}.
- Do not include explanations, comments, or extra fields.
- Output must be valid JSON parsable by a strict JSON parser.

Examples:
Q: What is the largest planet in our solar system?
S: Jupiter is the largest planet in our solar system.
→ {"score": 1}

Q: Who painted the Mona Lisa?
S: The Mona Lisa is displayed in the Louvre in Paris.
→ {"score": 5}

Q: In which year did World War II end?
S: World War II ended in 1945.
→ {"score": 1}

Q: What is the boiling point of water in Celsius?
S: Ice melts at 0 degrees Celsius.
→ {"score": 4}

Now evaluate this pair:
Q: {question}
S: {sentence}"""

            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": user_prompt},
                ],
                temperature=1.0,
                response_format={"type": "json_object"},
            )
            content = response.choices[0].message.content or "{}"
            parsed = json.loads(content)
            raw_score = parsed.get("score", 4)
            try:
                score = int(raw_score)
            except (TypeError, ValueError):
                score = 4
            # Clamp to [1, 5]
            if score < 1:
                score = 1
            if score > 5:
                score = 5
            # Map 1..5 -> [0.0, 1.0]
            return (score - 1) / 4.0

        max_workers = min(8, len(questions))
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            rewards: list[float] = list(executor.map(infer_one, questions, new_sentences))

        return rewards
