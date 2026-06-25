"""
services/llm_client.py
───────────────────────
Async HTTP client for OpenAI-compatible LLM APIs.

Default target: Ollama running locally (http://localhost:11434/v1).
Compatible with any OpenAI-compatible endpoint (Together AI, Groq, OpenAI, etc.)
by changing LLM_API_URL and LLM_API_KEY in .env.

Uses httpx (async) — no SDK dependency, works with Ollama without an API key.
"""

from __future__ import annotations

import asyncio
import logging
import time
from typing import Any

import httpx

from config import get_settings

logger = logging.getLogger(__name__)


async def call_llm(
    prompt: str,
    system: str = "",
    *,
    model: str | None = None,
    temperature: float = 0.3,
    max_tokens: int = 4096,
) -> str:
    """
    Send a chat completion request to an OpenAI-compatible LLM endpoint.

    Parameters
    ----------
    prompt      : user-turn content (the text to clean/rewrite)
    system      : system prompt (role / instruction)
    model       : override model name (defaults to settings.llm_model)
    temperature : sampling temperature — 0.3 gives consistent edits
    max_tokens  : maximum response tokens

    Returns
    -------
    The assistant's response text, stripped of leading/trailing whitespace.

    Raises
    ------
    RuntimeError
        If all retries are exhausted or the server returns a fatal error.
    """
    cfg = get_settings()
    _model   = model or cfg.llm_model
    base_url = cfg.llm_api_url.rstrip("/")
    api_key  = cfg.llm_api_key or "ollama"   # Ollama ignores the key

    messages: list[dict[str, str]] = []
    if system:
        messages.append({"role": "system", "content": system})
    messages.append({"role": "user", "content": prompt})

    payload: dict[str, Any] = {
        "model":       _model,
        "messages":    messages,
        "temperature": temperature,
        "max_tokens":  max_tokens,
        "stream":      False,
    }

    headers = {
        "Content-Type":  "application/json",
        "Authorization": f"Bearer {api_key}",
    }

    last_error: Exception | None = None

    for attempt in range(1, cfg.llm_max_retries + 1):
        try:
            t0 = time.monotonic()
            async with httpx.AsyncClient(timeout=cfg.llm_request_timeout) as client:
                response = await client.post(
                    f"{base_url}/chat/completions",
                    json=payload,
                    headers=headers,
                )

            elapsed = time.monotonic() - t0

            if response.status_code == 200:
                data = response.json()
                content: str = (
                    data.get("choices", [{}])[0]
                    .get("message", {})
                    .get("content", "")
                    .strip()
                )
                if not content:
                    raise ValueError("LLM returned an empty response")
                logger.debug(
                    "LLM call OK (attempt %d/%d, %.1f s, %d chars output)",
                    attempt, cfg.llm_max_retries, elapsed, len(content),
                )
                return content

            # 4xx errors are not retryable
            if 400 <= response.status_code < 500:
                body = _safe_body(response)
                raise RuntimeError(
                    f"LLM API client error {response.status_code}: {body}"
                )

            # 5xx — retry
            body = _safe_body(response)
            last_error = RuntimeError(
                f"LLM API server error {response.status_code}: {body}"
            )
            logger.warning(
                "LLM server error (attempt %d/%d): %s",
                attempt, cfg.llm_max_retries, last_error,
            )

        except (httpx.ConnectError, httpx.TimeoutException) as exc:
            last_error = exc
            logger.warning(
                "LLM request failed (attempt %d/%d): %s: %s",
                attempt, cfg.llm_max_retries, type(exc).__name__, exc,
            )

        if attempt < cfg.llm_max_retries:
            backoff = 2 ** (attempt - 1)   # 1s, 2s, 4s…
            logger.info("Retrying in %d s…", backoff)
            await asyncio.sleep(backoff)

    raise RuntimeError(
        f"LLM call failed after {cfg.llm_max_retries} attempts. "
        f"Last error: {last_error}"
    )


def _safe_body(response: httpx.Response) -> str:
    try:
        return response.json().get("error", {}).get("message", response.text[:300])
    except Exception:
        return response.text[:300]
