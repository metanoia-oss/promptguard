"""Google Gemini provider implementation."""

from __future__ import annotations

import os
from collections.abc import AsyncIterator, Iterator
from typing import Any

from promptguard.core.exceptions import (
    AuthenticationError,
    ContentFilteredError,
    ContextLengthExceededError,
    ModelNotFoundError,
    ProviderError,
    RateLimitError,
    TimeoutError,
)
from promptguard.core.logging import get_logger
from promptguard.providers.base import (
    CompletionResponse,
    LLMProvider,
    Message,
    MessageRole,
    UsageStats,
)
from promptguard.providers.registry import ProviderRegistry

logger = get_logger(__name__)


@ProviderRegistry.register("gemini")
class GeminiProvider(LLMProvider):
    """Google Gemini API provider.

    Supports Gemini 1.5 Pro, Gemini 1.5 Flash, and other Gemini models.

    Features:
        - System instruction support
        - Streaming support

    Environment Variables:
        GOOGLE_API_KEY: API key for authentication

    Note:
        Async support is limited in the current google-generativeai package.
        For async operations, consider using the REST API directly.
    """

    provider_name = "gemini"

    def __init__(
        self,
        api_key: str | None = None,
    ) -> None:
        """Initialize Gemini provider.

        Args:
            api_key: Google API key. Falls back to GOOGLE_API_KEY env var.

        Raises:
            ProviderError: If google-generativeai package is not installed.
        """
        try:
            import google.generativeai as genai
        except ImportError:
            raise ProviderError(
                "Google Generative AI package not installed. Run: pip install promptguard[google]",
                provider="gemini",
            )

        self._api_key = api_key or os.getenv("GOOGLE_API_KEY")
        genai.configure(api_key=self._api_key)
        self._genai = genai

    def _convert_messages(self, messages: list[Message]) -> tuple[str | None, list[dict[str, Any]]]:
        """Convert messages to Gemini format.

        Returns:
            Tuple of (system_instruction, chat_history)
        """
        system_instruction: str | None = None
        chat_history: list[dict[str, Any]] = []

        for msg in messages:
            if msg.role == MessageRole.SYSTEM:
                system_instruction = msg.content
            elif msg.role == MessageRole.USER:
                chat_history.append(
                    {
                        "role": "user",
                        "parts": [msg.content],
                    }
                )
            elif msg.role == MessageRole.ASSISTANT:
                chat_history.append(
                    {
                        "role": "model",
                        "parts": [msg.content],
                    }
                )

        return system_instruction, chat_history

    def _handle_api_error(self, e: Exception, model: str) -> None:
        """Convert API exceptions to specific PromptGuard exceptions.

        Args:
            e: The caught exception.
            model: The model name for error context.

        Raises:
            TimeoutError: If the request timed out.
            AuthenticationError: If authentication failed (401).
            RateLimitError: If rate limited (429).
            ModelNotFoundError: If model not found (404).
            ContextLengthExceededError: If prompt too long.
            ContentFilteredError: If content was blocked.
            ProviderError: For other API errors.
        """
        error_str = str(e).lower()

        if "timeout" in error_str or "timed out" in error_str:
            logger.error(
                "Request timed out", extra={"provider": self.provider_name, "model": model}
            )
            raise TimeoutError(self.provider_name, 60.0) from e
        elif (
            "401" in error_str or "invalid api key" in error_str or "api key not valid" in error_str
        ):
            logger.error("Authentication failed", extra={"provider": self.provider_name})
            raise AuthenticationError(self.provider_name) from e
        elif "429" in error_str or "rate limit" in error_str or "quota" in error_str:
            logger.warning("Rate limit exceeded", extra={"provider": self.provider_name})
            raise RateLimitError(self.provider_name) from e
        elif "404" in error_str or "model not found" in error_str or "not found" in error_str:
            logger.error("Model not found", extra={"provider": self.provider_name, "model": model})
            raise ModelNotFoundError(model, self.provider_name) from e
        elif "context" in error_str or "token" in error_str or "too long" in error_str:
            logger.error(
                "Context length exceeded", extra={"provider": self.provider_name, "model": model}
            )
            raise ContextLengthExceededError(self.provider_name, model) from e
        elif "safety" in error_str or "blocked" in error_str:
            logger.warning("Content filtered", extra={"provider": self.provider_name})
            raise ContentFilteredError(self.provider_name) from e
        else:
            logger.error("API error: %s", e, extra={"provider": self.provider_name})
            raise ProviderError(f"Gemini API error: {e}", provider=self.provider_name) from e

    def complete(
        self,
        messages: list[Message],
        model: str,
        temperature: float = 0.7,
        max_tokens: int | None = None,
        json_schema: dict[str, Any] | None = None,
        **kwargs: Any,
    ) -> CompletionResponse:
        """Execute a synchronous completion request."""
        system_instruction, chat_history = self._convert_messages(messages)

        generation_config = self._genai.GenerationConfig(
            temperature=temperature,
            max_output_tokens=max_tokens,
        )

        model_kwargs: dict[str, Any] = {
            "model_name": model,
            "generation_config": generation_config,
        }

        if system_instruction:
            model_kwargs["system_instruction"] = system_instruction

        logger.debug(
            "Starting completion request", extra={"provider": self.provider_name, "model": model}
        )

        try:
            gemini_model = self._genai.GenerativeModel(**model_kwargs)

            # If only one user message, use generate_content
            if len(chat_history) == 1:
                response = gemini_model.generate_content(chat_history[0]["parts"][0])
            else:
                # Use chat for multi-turn
                chat = gemini_model.start_chat(history=chat_history[:-1])
                last_message = chat_history[-1]["parts"][0]
                response = chat.send_message(last_message)

        except Exception as e:
            self._handle_api_error(e, model)

        # Check for content filtering
        if hasattr(response, "prompt_feedback") and response.prompt_feedback:
            feedback = response.prompt_feedback
            if hasattr(feedback, "block_reason") and feedback.block_reason:
                logger.warning("Content filtered", extra={"provider": self.provider_name})
                raise ContentFilteredError(self.provider_name, str(feedback.block_reason))

        content = response.text if hasattr(response, "text") else ""

        # Extract usage if available
        usage = None
        if hasattr(response, "usage_metadata"):
            metadata = response.usage_metadata
            usage = UsageStats(
                prompt_tokens=getattr(metadata, "prompt_token_count", 0),
                completion_tokens=getattr(metadata, "candidates_token_count", 0),
                total_tokens=getattr(metadata, "total_token_count", 0),
            )
            logger.debug(
                "Completion finished",
                extra={
                    "provider": self.provider_name,
                    "model": model,
                    "prompt_tokens": getattr(metadata, "prompt_token_count", 0),
                    "completion_tokens": getattr(metadata, "candidates_token_count", 0),
                },
            )

        return CompletionResponse(
            content=content,
            model=model,
            provider=self.provider_name,
            usage=usage,
            raw_response=response,
        )

    async def acomplete(
        self,
        messages: list[Message],
        model: str,
        temperature: float = 0.7,
        max_tokens: int | None = None,
        json_schema: dict[str, Any] | None = None,
        **kwargs: Any,
    ) -> CompletionResponse:
        """Execute an asynchronous completion request.

        Note: The google-generativeai package has limited async support.
        This method wraps the sync call for now.
        """
        import asyncio

        return await asyncio.get_event_loop().run_in_executor(
            None,
            lambda: self.complete(messages, model, temperature, max_tokens, json_schema, **kwargs),
        )

    def stream(
        self,
        messages: list[Message],
        model: str,
        temperature: float = 0.7,
        max_tokens: int | None = None,
        **kwargs: Any,
    ) -> Iterator[str]:
        """Execute a synchronous streaming completion request."""
        system_instruction, chat_history = self._convert_messages(messages)

        generation_config = self._genai.GenerationConfig(
            temperature=temperature,
            max_output_tokens=max_tokens,
        )

        model_kwargs: dict[str, Any] = {
            "model_name": model,
            "generation_config": generation_config,
        }

        if system_instruction:
            model_kwargs["system_instruction"] = system_instruction

        logger.debug(
            "Starting streaming request", extra={"provider": self.provider_name, "model": model}
        )

        try:
            gemini_model = self._genai.GenerativeModel(**model_kwargs)

            if len(chat_history) == 1:
                response = gemini_model.generate_content(
                    chat_history[0]["parts"][0],
                    stream=True,
                )
            else:
                chat = gemini_model.start_chat(history=chat_history[:-1])
                last_message = chat_history[-1]["parts"][0]
                response = chat.send_message(last_message, stream=True)

            for chunk in response:
                if hasattr(chunk, "text") and chunk.text:
                    yield chunk.text

        except Exception as e:
            self._handle_api_error(e, model)

    async def astream(
        self,
        messages: list[Message],
        model: str,
        temperature: float = 0.7,
        max_tokens: int | None = None,
        **kwargs: Any,
    ) -> AsyncIterator[str]:
        """Execute an asynchronous streaming completion request.

        Note: Wraps sync stream for compatibility.
        """
        import asyncio

        def _stream():
            return list(self.stream(messages, model, temperature, max_tokens, **kwargs))

        chunks = await asyncio.get_event_loop().run_in_executor(None, _stream)
        for chunk in chunks:
            yield chunk

    def get_model_info(self, model: str) -> dict[str, Any]:
        """Get information about a model."""
        info: dict[str, Any] = {
            "model": model,
            "provider": self.provider_name,
            "supports_json_mode": False,
            "supports_structured_output": False,
        }

        # Add known context lengths
        if "1.5-pro" in model:
            info["context_length"] = 2097152  # 2M tokens
        elif "1.5-flash" in model:
            info["context_length"] = 1048576  # 1M tokens
        elif "1.0-pro" in model:
            info["context_length"] = 32768

        return info
