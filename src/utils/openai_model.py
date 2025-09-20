import json
import logging
import os
from concurrent.futures import ThreadPoolExecutor

from dotenv import load_dotenv
from openai import OpenAI

logger = logging.getLogger(__name__)


SYSTEM_PROMPT = """
You are an expert answerability evaluator.
Your ONLY task is to decide if a question is answerable using ONLY a provided sentence.

Output requirements:
- Always output exactly one JSON object.
- The JSON object MUST strictly follow this schema:
  {"answer": "yes"} OR {"answer": "no"}
- Do NOT include explanations, reasoning, comments, or extra fields.
- Do NOT output natural language, markdown, or anything outside the JSON object.
- The output must be valid JSON parsable by a strict JSON parser.
- If uncertain, always choose {"answer": "no"}.

Breaking these rules means your output is invalid.
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
        """Compute rewards via batched inference using a single structured JSON response.

        Each pair (question, new_sentence) is judged as to whether the new_sentence makes the
        question answerable using ONLY information from that sentence. If answerable, the reward
        is -1.0 (undesirable). If not answerable, the reward is 1.0.

        Args:
            questions: List of questions, length N.
            new_sentences: List of candidate new sentences (same length as questions).

        Returns:
            A list of float rewards of length N, values in {-1.0, 1.0}.
        """
        if len(questions) != len(new_sentences):
            raise ValueError("questions and new_sentences must have the same length")

        if not questions:
            return []

        def infer_one(question: str, sentence: str) -> float:
            """
            Infer the reward for a single pair (question, sentence).

            Args:
                question: The question to evaluate.
                sentence: The sentence to use to answer the question.

            Returns:
                The reward for the pair.
                -1.0 if the question is answerable using ONLY the given sentence.
                1.0 if the question is not answerable using ONLY the given sentence.
            """
            user_prompt = f"""
Determine whether the question can be answered using ONLY the given sentence.

Rules:
1. Use only the given sentence. Ignore background knowledge, assumptions, or prior context.
2. If the sentence alone fully answers the question, respond with {{"answer": "yes"}}.
3. If the sentence does not provide enough information to answer the question, respond with {{"answer": "no"}}.
4. Output must be valid JSON and match the schema exactly.

Examples:
Q: What is the largest planet in our solar system?
S: Jupiter is the largest planet in our solar system.
→ {{"answer": "yes"}}

Q: Who painted the Mona Lisa?
S: The Mona Lisa is displayed in the Louvre in Paris.
→ {{"answer": "no"}}

Q: In which year did World War II end?
S: World War II ended in 1945.
→ {{"answer": "yes"}}

Q: What is the boiling point of water in Celsius?
S: Ice melts at 0 degrees Celsius.
→ {{"answer": "no"}}

Now evaluate this pair:
Q: {question}
S: {sentence}
"""

            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": user_prompt},
                ],
                temperature=0,
                response_format={"type": "json_object"},
            )
            content = response.choices[0].message.content or "{}"
            parsed = json.loads(content)
            answer = str(parsed.get("answer", "no")).strip().lower()
            return -1.0 if answer == "yes" else 1.0

        max_workers = min(8, len(questions))
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            rewards: list[float] = list(executor.map(infer_one, questions, new_sentences))

        return rewards


if __name__ == "__main__":
    load_dotenv()
    evaluator = AnswerabilityEvaluator()
    questions = ["What is the capital of France?", "What is the capital of Germany?"]
    new_sentences = ["Paris is the capital of France.", "Germany has beautiful cities."]
    rewards = evaluator.get_rewards(questions, new_sentences)
    print(rewards)
