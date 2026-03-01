"""LLM calling with retry logic, OpenAI-compatible endpoints."""

from __future__ import annotations

import logging
import time
from typing import List, Optional

from .config import LinmemConfig

logger = logging.getLogger(__name__)


def call_llm(
    prompt: str,
    context_passages: List[str],
    config: LinmemConfig,
) -> str:
    """Send a question + context to an LLM and return the answer."""
    try:
        from openai import OpenAI
    except ImportError:
        raise RuntimeError(
            "openai package required for 'ask' command. "
            "Install with: uv pip install linmem[llm]"
        )

    client = OpenAI(
        base_url=config.llm_base_url,
        api_key=config.llm_api_key or "not-set",
    )

    context = "\n\n---\n\n".join(context_passages)
    messages = [
        {
            "role": "system",
            "content": (
                "You are a helpful assistant. Answer the user's question "
                "based on the provided context passages. If the context "
                "doesn't contain enough information, say so."
            ),
        },
        {
            "role": "user",
            "content": f"Context:\n{context}\n\nQuestion: {prompt}",
        },
    ]

    last_err = None
    for attempt in range(config.llm_max_retries):
        try:
            resp = client.chat.completions.create(
                model=config.llm_model,
                messages=messages,
            )
            return resp.choices[0].message.content or ""
        except Exception as e:
            last_err = e
            wait = 2 ** attempt
            logger.warning("LLM call failed (attempt %d): %s, retrying in %ds", attempt + 1, e, wait)
            time.sleep(wait)

    raise RuntimeError(f"LLM call failed after {config.llm_max_retries} retries: {last_err}")
