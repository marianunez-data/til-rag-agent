"""
evaluation.py
-------------
LLM-as-judge evaluation pipeline for the TIL agent.
Loads interaction logs, generates structured quality assessments,
and aggregates results into a Pandas DataFrame for analysis.
"""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

import pandas as pd
from openai import OpenAI

OLLAMA_BASE_URL = "http://localhost:11434/v1"
OLLAMA_MODEL = "llama3.2"

EVALUATION_CRITERIA = [
    "instructions_follow",
    "instructions_avoid",
    "answer_relevant",
    "answer_clear",
    "answer_citations",
    "completeness",
    "tool_call_search",
]

JUDGE_SYSTEM_PROMPT = """
You are an objective evaluator of AI assistant responses.

Given the agent's instructions, a user question, and the agent's answer,
evaluate the response against the following criteria. Respond ONLY with
a valid JSON object — no prose, no markdown fences.

Criteria:
- instructions_follow: the agent followed its system instructions
- instructions_avoid: the agent avoided prohibited behaviours
- answer_relevant: the response directly addresses the question
- answer_clear: the answer is accurate and clearly written
- answer_citations: the response includes source citations where required
- completeness: the response covers all key aspects of the question
- tool_call_search: the search tool was invoked before answering

Response format:
{
  "instructions_follow": true,
  "instructions_avoid": true,
  "answer_relevant": true,
  "answer_clear": true,
  "answer_citations": false,
  "completeness": true,
  "tool_call_search": true,
  "summary": "one-sentence overall assessment"
}
""".strip()


# ---------------------------------------------------------------------------
# Judge
# ---------------------------------------------------------------------------

def evaluate_log(log_record: dict) -> dict[str, Any]:
    """
    Run the LLM judge on a single interaction log.

    Args:
        log_record: A log dict as returned by logs.load_log().

    Returns:
        Dict with boolean values for each criterion and a summary string.
    """
    client = OpenAI(base_url=OLLAMA_BASE_URL, api_key="ollama")

    messages = log_record["messages"]
    instructions = log_record.get("system_prompt", "")
    question = messages[0]["parts"][0]["content"]
    answer = messages[-1]["parts"][0]["content"]

    user_content = (
        f"<INSTRUCTIONS>{instructions}</INSTRUCTIONS>\n"
        f"<QUESTION>{question}</QUESTION>\n"
        f"<ANSWER>{answer}</ANSWER>"
    )

    response = client.chat.completions.create(
        model=OLLAMA_MODEL,
        messages=[
            {"role": "system", "content": JUDGE_SYSTEM_PROMPT},
            {"role": "user", "content": user_content},
        ],
    )
    raw = response.choices[0].message.content

    try:
        match = re.search(r"\{.*\}", raw, re.DOTALL)
        if match:
            return json.loads(match.group())
    except Exception:
        pass

    # Fallback defaults if parsing fails
    return {c: False for c in EVALUATION_CRITERIA} | {"summary": "parse error"}


# ---------------------------------------------------------------------------
# Batch evaluation
# ---------------------------------------------------------------------------

def evaluate_batch(log_records: list[dict]) -> pd.DataFrame:
    """
    Evaluate a list of interaction logs and return a metrics DataFrame.

    Args:
        log_records: List of log dicts.

    Returns:
        DataFrame with one row per interaction and one column per criterion.
    """
    rows = []
    for log_record in log_records:
        try:
            result = evaluate_log(log_record)
        except Exception as exc:
            result = {c: False for c in EVALUATION_CRITERIA}
            result["summary"] = str(exc)

        messages = log_record["messages"]
        row = {
            "log_file": log_record["log_file"].name,
            "question": messages[0]["parts"][0]["content"][:100],
            "answer_preview": messages[-1]["parts"][0]["content"][:150],
        }
        for criterion in EVALUATION_CRITERIA:
            row[criterion] = result.get(criterion, False)
        row["summary"] = result.get("summary", "")
        rows.append(row)

    return pd.DataFrame(rows)


def print_metrics(df: pd.DataFrame) -> None:
    """Print a formatted summary of evaluation pass rates."""
    print("Evaluation Results")
    print("=" * 40)
    for col in EVALUATION_CRITERIA:
        if col in df.columns:
            rate = df[col].mean()
            bar = "#" * int(rate * 20)
            print(f"  {col:<25} {rate:.0%}  {bar}")
    print()
    print(f"  Total interactions evaluated: {len(df)}")
