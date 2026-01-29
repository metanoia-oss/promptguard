"""Anthropic Claude provider implementation."""

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


@ProviderRegistry.register("anthropic")
class AnthropicProvider(LLMProvider):
    """Anthropic Claude API provider.

    Supports Claude 3 (Opus, Sonnet, Haiku) and Claude 3.5 models.

    Features:
        - System prompt handling (separate from messages)
        - Streaming support
        - Async support

    Environment Variables:
        ANTHROPIC_API_KEY: API key for authentication
    """

    provider_name = "anthropic"

    # Default max tokens if not specified
    DEFAULT_MAX_TOKENS = 4096

    def __init__(
        self,
        api_key: str | None = None,
        timeout: float = 60.0,
    ) -> None:
        """Initialize Anthropic provider.

        Args:
            api_key: Anthropic API key. Falls back to ANTHROPIC_API_KEY env var.
            timeout: Request timeout in seconds.

        Raises:
            ProviderError: If anthropic package is not installed.
        """
        try:
            from anthropic import Anthropic, AsyncAnthropic
        except ImportError:
            raise ProviderError(
                "Anthropic package not installed. Run: pip install promptguard[anthropic]",
                provider="anthropic",
            )

        self._api_key = api_key or os.getenv("ANTHROPIC_API_KEY")
        self._timeout = timeout

        self._client = Anthropic(
            api_key=self._api_key,
            timeout=self._timeout,
        )
        self._async_client = AsyncAnthropic(
            api_key=self._api_key,
            timeout=self._timeout,
        )

    def _convert_messages(self, messages: list[Message]) -> tuple[str | None, list[dict[str, str]]]:
        """Convert messages, extracting system message separately.

        Anthropic's API handles system messages differently from user/assistant
        messages, so we need to extract it separately.

        Returns:
            Tuple of (system_message, api_messages)
        """
        system_message: str | None = None
        api_messages: list[dict[str, str]] = []

        for msg in messages:
            if msg.role == MessageRole.SYSTEM:
                # Anthropic only supports one system message
                system_message = msg.content
            else:
                api_messages.append(
                    {
                        "role": msg.role.value,
                        "content": msg.content,
                    }
                )

        return system_message, api_messages

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
            ProviderError: For other API errors.
        """
        error_str = str(e).lower()

        if "timeout" in error_str or "timed out" in error_str:
            logger.error(
                "Request timed out", extra={"provider": self.provider_name, "model": model}
            )
            raise TimeoutError(self.provider_name, self._timeout) from e
        elif "401" in error_str or "unauthorized" in error_str or "invalid api key" in error_str:
            logger.error("Authentication failed", extra={"provider": self.provider_name})
            raise AuthenticationError(self.provider_name) from e
        elif "429" in error_str or "rate limit" in error_str:
            logger.warning("Rate limit exceeded", extra={"provider": self.provider_name})
            raise RateLimitError(self.provider_name) from e
        elif "404" in error_str or "model not found" in error_str or "does not exist" in error_str:
            logger.error("Model not found", extra={"provider": self.provider_name, "model": model})
            raise ModelNotFoundError(model, self.provider_name) from e
        elif (
            "context" in error_str
            or "too many tokens" in error_str
            or "prompt is too long" in error_str
        ):
            logger.error(
                "Context length exceeded", extra={"provider": self.provider_name, "model": model}
            )
            raise ContextLengthExceededError(self.provider_name, model) from e
        else:
            logger.error("API error: %s", e, extra={"provider": self.provider_name})
            raise ProviderError(f"Anthropic API error: {e}", provider=self.provider_name) from e

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
        system_message, api_messages = self._convert_messages(messages)

        api_kwargs: dict[str, Any] = {
            "model": model,
            "messages": api_messages,
            "max_tokens": max_tokens or self.DEFAULT_MAX_TOKENS,
            "temperature": temperature,
        }

        if system_message:
            api_kwargs["system"] = system_message

        # Pass through additional kwargs
        for key in ["stop_sequences", "top_p", "top_k"]:
            if key in kwargs:
                api_kwargs[key] = kwargs[key]

        logger.debug(
            "Starting completion request", extra={"provider": self.provider_name, "model": model}
        )

        try:
            response = self._client.messages.create(**api_kwargs)
        except Exception as e:
            self._handle_api_error(e, model)

        # Check for content filtering
        if response.stop_reason == "content_filter":
            logger.warning("Content filtered", extra={"provider": self.provider_name})
            raise ContentFilteredError(self.provider_name)

        # Extract text content
        content = ""
        for block in response.content:
            if hasattr(block, "text"):
                content += block.text

        usage = UsageStats(
            prompt_tokens=response.usage.input_tokens,
            completion_tokens=response.usage.output_tokens,
            total_tokens=response.usage.input_tokens + response.usage.output_tokens,
        )

        logger.debug(
            "Completion finished",
            extra={
                "provider": self.provider_name,
                "model": model,
                "prompt_tokens": response.usage.input_tokens,
                "completion_tokens": response.usage.output_tokens,
            },
        )

        return CompletionResponse(
            content=content,
            model=response.model,
            provider=self.provider_name,
            usage=usage,
            raw_response=response,
            finish_reason=response.stop_reason,
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
        """Execute an asynchronous completion request."""
        system_message, api_messages = self._convert_messages(messages)

        api_kwargs: dict[str, Any] = {
            "model": model,
            "messages": api_messages,
            "max_tokens": max_tokens or self.DEFAULT_MAX_TOKENS,
            "temperature": temperature,
        }

        if system_message:
            api_kwargs["system"] = system_message

        for key in ["stop_sequences", "top_p", "top_k"]:
            if key in kwargs:
                api_kwargs[key] = kwargs[key]

        logger.debug(
            "Starting async completion request",
            extra={"provider": self.provider_name, "model": model},
        )

        try:
            response = await self._async_client.messages.create(**api_kwargs)
        except Exception as e:
            self._handle_api_error(e, model)

        # Check for content filtering
        if response.stop_reason == "content_filter":
            logger.warning("Content filtered", extra={"provider": self.provider_name})
            raise ContentFilteredError(self.provider_name)

        content = ""
        for block in response.content:
            if hasattr(block, "text"):
                content += block.text

        usage = UsageStats(
            prompt_tokens=response.usage.input_tokens,
            completion_tokens=response.usage.output_tokens,
            total_tokens=response.usage.input_tokens + response.usage.output_tokens,
        )

        logger.debug(
            "Async completion finished",
            extra={
                "provider": self.provider_name,
                "model": model,
                "prompt_tokens": response.usage.input_tokens,
                "completion_tokens": response.usage.output_tokens,
            },
        )

        return CompletionResponse(
            content=content,
            model=response.model,
            provider=self.provider_name,
            usage=usage,
            raw_response=response,
            finish_reason=response.stop_reason,
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
        system_message, api_messages = self._convert_messages(messages)

        api_kwargs: dict[str, Any] = {
            "model": model,
            "messages": api_messages,
            "max_tokens": max_tokens or self.DEFAULT_MAX_TOKENS,
            "temperature": temperature,
        }

        if system_message:
            api_kwargs["system"] = system_message

        logger.debug(
            "Starting streaming request", extra={"provider": self.provider_name, "model": model}
        )

        try:
            with self._client.messages.stream(**api_kwargs) as stream:
                yield from stream.text_stream
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
        """Execute an asynchronous streaming completion request."""
        system_message, api_messages = self._convert_messages(messages)

        api_kwargs: dict[str, Any] = {
            "model": model,
            "messages": api_messages,
            "max_tokens": max_tokens or self.DEFAULT_MAX_TOKENS,
            "temperature": temperature,
        }

        if system_message:
            api_kwargs["system"] = system_message

        logger.debug(
            "Starting async streaming request",
            extra={"provider": self.provider_name, "model": model},
        )

        try:
            async with self._async_client.messages.stream(**api_kwargs) as stream:
                async for text in stream.text_stream:
                    yield text
        except Exception as e:
            self._handle_api_error(e, model)

    def get_model_info(self, model: str) -> dict[str, Any]:
        """Get information about a model."""
        info: dict[str, Any] = {
            "model": model,
            "provider": self.provider_name,
            "supports_json_mode": False,
            "supports_structured_output": False,
        }

        # Add known context lengths
        context_lengths = {
            "claude-3-opus": 200000,
            "claude-3-sonnet": 200000,
            "claude-3-haiku": 200000,
            "claude-3-5-sonnet": 200000,
            "claude-3-5-haiku": 200000,
        }
        for prefix, length in context_lengths.items():
            if prefix in model:
                info["context_length"] = length
                break

        return info
