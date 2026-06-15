"""
backend/llm/groq_provider.py
Groq LLM provider with retry, timeout, streaming, and error handling.
"""

import asyncio
import time
from typing import AsyncGenerator, Optional

from groq import Groq, AsyncGroq, APIError, RateLimitError

from backend.config import get_settings
from backend.logging_config import get_logger

logger = get_logger(__name__)
settings = get_settings()

SUPPORTED_MODELS = {
    "llama-3.3-70b-versatile",
    "llama-3.1-8b-instant",
    "mixtral-8x7b-32768",
}

MAX_RETRIES = 3
RETRY_DELAY = 2.0  # seconds


class GroqProvider:
    """Wraps the Groq API with retry logic, timeout handling, and model switching."""

    def __init__(self, model: Optional[str] = None):
        self.model = model or settings.default_llm_model
        if self.model not in SUPPORTED_MODELS:
            raise ValueError(
                f"Model '{self.model}' not supported. Choose from: {sorted(SUPPORTED_MODELS)}"
            )
        self._sync_client = Groq(api_key=settings.groq_api_key)
        self._async_client = AsyncGroq(api_key=settings.groq_api_key)

    def generate(
        self,
        system_message: str,
        user_message: str,
        max_tokens: int = 1024,
        temperature: float = 0.1,
    ) -> str:
        """Synchronous generation with retry logic."""
        for attempt in range(1, MAX_RETRIES + 1):
            try:
                response = self._sync_client.chat.completions.create(
                    model=self.model,
                    messages=[
                        {"role": "system", "content": system_message},
                        {"role": "user", "content": user_message},
                    ],
                    max_tokens=max_tokens,
                    temperature=temperature,
                    timeout=settings.llm_timeout,
                )
                return response.choices[0].message.content or ""
            except RateLimitError:
                logger.warning(f"Rate limit hit, attempt {attempt}/{MAX_RETRIES}")
                if attempt < MAX_RETRIES:
                    time.sleep(RETRY_DELAY * attempt)
                else:
                    raise
            except APIError as e:
                logger.error("Groq API error", extra={"error": str(e), "attempt": attempt})
                if attempt < MAX_RETRIES:
                    time.sleep(RETRY_DELAY)
                else:
                    raise

        raise RuntimeError("Groq generation failed after max retries.")

    async def async_generate(
        self,
        system_message: str,
        user_message: str,
        max_tokens: int = 1024,
        temperature: float = 0.1,
    ) -> str:
        """Async generation with retry logic."""
        for attempt in range(1, MAX_RETRIES + 1):
            try:
                response = await self._async_client.chat.completions.create(
                    model=self.model,
                    messages=[
                        {"role": "system", "content": system_message},
                        {"role": "user", "content": user_message},
                    ],
                    max_tokens=max_tokens,
                    temperature=temperature,
                    timeout=settings.llm_timeout,
                )
                return response.choices[0].message.content or ""
            except RateLimitError:
                logger.warning(f"Rate limit, attempt {attempt}/{MAX_RETRIES}")
                if attempt < MAX_RETRIES:
                    await asyncio.sleep(RETRY_DELAY * attempt)
                else:
                    raise
            except APIError as e:
                logger.error("Groq API error", extra={"error": str(e), "attempt": attempt})
                if attempt < MAX_RETRIES:
                    await asyncio.sleep(RETRY_DELAY)
                else:
                    raise

        raise RuntimeError("Groq async generation failed after max retries.")

    async def stream_generate(
        self,
        system_message: str,
        user_message: str,
        max_tokens: int = 1024,
        temperature: float = 0.1,
    ) -> AsyncGenerator[str, None]:
        """
        Streaming generation – yields text chunks as they arrive.
        Usage:
            async for token in provider.stream_generate(sys, user):
                print(token, end="", flush=True)
        """
        stream = await self._async_client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": system_message},
                {"role": "user", "content": user_message},
            ],
            max_tokens=max_tokens,
            temperature=temperature,
            stream=True,
        )
        async for chunk in stream:
            delta = chunk.choices[0].delta.content
            if delta:
                yield delta
