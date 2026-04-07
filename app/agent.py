"""
agent.py
--------
Pydantic AI agent backed by a local Ollama instance.
Exposes the hybrid search pipeline as a tool so the agent can ground
its answers in content from the jbranchaud/til repository.
"""

from __future__ import annotations

from typing import Any

from openai import OpenAI
from pydantic_ai import Agent
from pydantic_ai.models.openai import OpenAIModel
from pydantic_ai.providers.openai import OpenAIProvider
from search import SearchPipeline

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

OLLAMA_BASE_URL = "http://localhost:11434/v1"
OLLAMA_MODEL = "llama3.2"

SYSTEM_PROMPT = """
You are a technical assistant. When a user asks a programming question,
you MUST call the search_til function with a relevant search query.
Never show the function schema to the user.
Always execute the search and use the results to write your answer.

Example: if the user asks "How do I squash commits?",
call search_til with query="squash commits git" and answer based on results.
""".strip()


# ---------------------------------------------------------------------------
# Agent factory
# ---------------------------------------------------------------------------


def build_agent(pipeline: SearchPipeline) -> Agent:
    """
    Instantiate a Pydantic AI agent with the hybrid search tool attached.

    Args:
        pipeline: A fully initialised SearchPipeline instance.

    Returns:
        Configured pydantic_ai.Agent ready to run queries.
    """
    model = OpenAIModel(
        model_name=OLLAMA_MODEL,
        provider=OpenAIProvider(
            base_url=OLLAMA_BASE_URL,
            api_key="ollama",
        ),
    )

    def search_til(query: str) -> list[Any]:
        """
        Search the jbranchaud/til repository for programming tips.

        Uses hybrid retrieval (lexical + semantic) to surface the most
        relevant TIL entries for the given query.

        Args:
            query: Natural-language search string.

        Returns:
            List of matching document chunks with content and filename.
        """
        return pipeline.search(query, n=5)

    agent = Agent(
        name="til_agent",
        instructions=SYSTEM_PROMPT,
        tools=[search_til],
        model=model,
    )
    return agent


# ---------------------------------------------------------------------------
# Ollama connectivity check
# ---------------------------------------------------------------------------


def check_ollama_connection() -> bool:
    """
    Verify that the local Ollama service is reachable.

    Returns:
        True if a test completion succeeds, False otherwise.
    """
    try:
        client = OpenAI(base_url=OLLAMA_BASE_URL, api_key="ollama")
        client.chat.completions.create(
            model=OLLAMA_MODEL,
            messages=[{"role": "user", "content": "ping"}],
            max_tokens=5,
        )
        return True
    except Exception:
        return False
