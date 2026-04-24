import os
from typing import Optional, Type, TypeVar

import yaml
from dotenv import load_dotenv
from google import genai
from google.genai import types
from pydantic import BaseModel, Field

T = TypeVar("T", bound=BaseModel)
MODEL = "gemini-3.1-flash-lite-preview"

_client: Optional[genai.Client] = None
_CATEGORIES: Optional[list[dict]] = None


def init_client() -> genai.Client:
    global _client

    if _client is None:
        load_dotenv()
        api_key = os.getenv("GOOGLE_API_KEY")
        if not api_key:
            raise RuntimeError("GOOGLE_API_KEY not found in environment")

        _client = genai.Client()

    return _client


def load_categories(path: str = "categories.yaml") -> list[dict]:
    global _CATEGORIES

    if _CATEGORIES is None:
        with open(path) as f:
            data = yaml.safe_load(f)
        _CATEGORIES = data["categories"]

    return _CATEGORIES


class CategoryScore(BaseModel):
    reasoning: str
    score: int = Field(
        ...,
        ge=1,
        le=5,
        description="Score from 1 to 5, where 1 means 'not present at all' and 5 means 'very strongly present'",
    )


class AllCategoryResults(BaseModel):
    results: dict[str, CategoryScore]


class Verdict(BaseModel):
    reasoning: str
    ai_probability: int = Field(
        ...,
        ge=0,
        le=100,
        description="Estimated probability that the text was written by AI, from 0 to 100",
    )


def call_llm(prompt: str, response_model: Type[T], retries: int = 1) -> T:
    """Make a structured LLM call, with retries if the output is not valid JSON conforming to the provided Pydantic model"""

    client = init_client()

    config = types.GenerateContentConfig(
        temperature=0.0,
        thinking_config=types.ThinkingConfig(thinking_level="minimal"),
        response_mime_type="application/json",
        response_json_schema=response_model.model_json_schema(),
    )

    for attempt in range(retries + 1):
        response = client.models.generate_content(
            model=MODEL,
            contents=prompt,
            config=config,
        )

        text = response.text

        try:
            return response_model.model_validate_json(text)
        except Exception as e:
            if attempt == retries:
                raise ValueError(f"Invalid structured output:\n{text}") from e


def score_category(category: dict, text: str) -> CategoryScore:
    """Run the rubric for a single category and return the score and reasoning"""

    examples_str = "\n".join(category["examples"])
    prompt = f"""
You are an expert linguistic evaluator. Your task is to assess whether the given text exhibits a specific writing pattern.
You must be precise, consistent, and conservative in your judgments. Do not overgeneralize or infer intent beyond the text.

## EVALUATION CATEGORY:
Name: {category["name"]}

Definition: 
{category["description"]}

Examples (for calibration only, do NOT copy wording or assume exact matches are required):
{examples_str}

## SCORING GUIDELINES
Score from 1 to 5 based on how strongly the pattern is present:

1 = Not present at all
2 = Possibly present (weak or ambiguous signal)  
3 = Moderately present (clear instance, but limited or isolated)  
4 = Strongly present (multiple clear instances or a dominant pattern)  
5 = Very strongly present (pervasive, defining characteristic of the text)

Important:
- Base your score ONLY on observable text features.
- Do NOT rely on assumptions about the author or intent.
- Do NOT penalize or reward general writing quality.
- The pattern must match the definition, not just vaguely resemble it.

## OUTPUT FORMAT:
Return JSON with:
- reasoning: brief explanation of your decision
- score: 1-5

## INPUT TEXT:
{text}
""".strip()

    return call_llm(prompt, CategoryScore)


def final_verdict(categories: AllCategoryResults, word_count: int) -> Verdict:
    prompt = f"""
You are an impartial judge. Your task is to estimate the probability that a text was written by an AI system.
You MUST base your decision strictly on the provided category scores and reasoning.

## INPUT CATEGORY SCORES:
Each category includes:
- a score from 1 (not present at all) to 5 (very strongly present)
- reasoning describing the evidence

{categories.model_dump_json(indent=2)}

## OUTPUT PROBABILITY GUIDELINES
Use these rough anchors for calibration:

- 0-20: little to no evidence of AI patterns
- 21-40: weak or sparse signals
- 41-60: mixed or moderate evidence
- 61-80: strong evidence across multiple categories
- 81-100: overwhelming and consistent AI-like patterns

When in doubt, favor the human-written hypothesis: unless the evidence is both strong and consistent, keep the probability close to 50 rather than assigning a high AI likelihood.

## LENGTH ADJUSTMENT RULES
Text length: {word_count} words
- <50 words: low reliability -> reduce confidence
- >=200 words: higher reliability -> increase confidence

## OUTPUT FORMAT:
Return JSON with:
- reasoning: brief explanation of your decision
- ai_probability: estimated probability from 0 to 100
""".strip()

    return call_llm(prompt, Verdict)


def analyze_text(text: str, categories_path: str = "categories.yaml") -> dict:
    """Run each category rubric, then feed results into final verdict prompt"""

    categories = load_categories(categories_path)

    results = {}
    for c in categories:
        results[c["name"]] = score_category(c, text)

    category_results = AllCategoryResults(results=results)

    verdict = final_verdict(category_results, len(text.split()))

    return {
        "rubrics": category_results.model_dump(),
        "verdict": verdict.model_dump(),
    }


if __name__ == "__main__":
    result = analyze_text("Test")
    print(result)
