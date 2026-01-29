"""OpenAI provider implementation."""

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
    UsageStats,
)
from promptguard.providers.registry import ProviderRegistry

logger = get_logger(__name__)


@ProviderRegistry.register("openai")
class OpenAIProvider(LLMProvider):
    """OpenAI API provider.

    Supports GPT-4, GPT-4 Turbo, GPT-4o, GPT-3.5 Turbo, and other
    OpenAI chat completion models.

    Features:
        - JSON mode for compatible models
        - Structured output (JSON schema) for GPT-4o models
        - Streaming support
        - Async support

    Environment Variables:
        OPENAI_API_KEY: API key for authentication
        OPENAI_BASE_URL: Optional custom base URL
        OPENAI_ORGANIZATION: Optional organization ID
    """

    provider_name = "openai"

    # Models supporting native structured output with JSON schema
    STRUCTURED_OUTPUT_MODELS = frozenset(
        {
            "gpt-4o",
            "gpt-4o-2024-08-06",
            "gpt-4o-2024-11-20",
            "gpt-4o-mini",
            "gpt-4o-mini-2024-07-18",
        }
    )

    # Models supporting JSON mode
    JSON_MODE_MODELS = frozenset(
        {
            "gpt-4o",
            "gpt-4o-mini",
            "gpt-4-turbo",
            "gpt-4-turbo-preview",
            "gpt-4-turbo-2024-04-09",
            "gpt-3.5-turbo",
            "gpt-3.5-turbo-0125",
            "gpt-3.5-turbo-1106",
        }
    )

    def __init__(
        self,
        api_key: str | None = None,
        base_url: str | None = None,
        organization: str | None = None,
        timeout: float = 60.0,
    ) -> None:
        """Initialize OpenAI provider.

        Args:
            api_key: OpenAI API key. Falls back to OPENAI_API_KEY env var.
            base_url: Custom API base URL for proxies or Azure.
            organization: OpenAI organization ID.
            timeout: Request timeout in seconds.

        Raises:
            ProviderError: If openai package is not installed.
        """
        try:
            from openai import AsyncOpenAI, OpenAI
        except ImportError:
            raise ProviderError(
                "OpenAI package not installed. Run: pip install promptguard[openai]",
                provider="openai",
            )

        self._api_key = api_key or os.getenv("OPENAI_API_KEY")
        self._base_url = base_url or os.getenv("OPENAI_BASE_URL")
        self._organization = organization or os.getenv("OPENAI_ORGANIZATION")
        self._timeout = timeout

        self._client = OpenAI(
            api_key=self._api_key,
            base_url=self._base_url,
            organization=self._organization,
            timeout=self._timeout,
        )
        self._async_client = AsyncOpenAI(
            api_key=self._api_key,
            base_url=self._base_url,
            organization=self._organization,
            timeout=self._timeout,
        )

    def _prepare_schema_for_structured_output(self, schema: dict[str, Any]) -> dict[str, Any]:
        """Prepare a JSON schema for OpenAI structured output.

        OpenAI requires additionalProperties: false on all object types
        when using strict mode.
        """
        schema = dict(schema)  # shallow copy
        if schema.get("type") == "object":
            schema["additionalProperties"] = False
            if "properties" in schema:
                schema["properties"] = {
                    k: self._prepare_schema_for_structured_output(v) if isinstance(v, dict) else v
                    for k, v in schema["properties"].items()
                }
        # Handle $defs / definitions for Pydantic models
        for defs_key in ("$defs", "definitions"):
            if defs_key in schema:
                schema[defs_key] = {
                    k: self._prepare_schema_for_structured_output(v) if isinstance(v, dict) else v
                    for k, v in schema[defs_key].items()
                }
        return schema

    def _convert_messages(self, messages: list[Message]) -> list[dict[str, str]]:
        """Convert internal Message format to OpenAI API format."""
        return [{"role": m.role.value, "content": m.content} for m in messages]

    def _build_api_kwargs(
        self,
        messages: list[Message],
        model: str,
        temperature: float,
        max_tokens: int | None,
        json_schema: dict[str, Any] | None,
        **kwargs: Any,
    ) -> dict[str, Any]:
        """Build kwargs for OpenAI API call."""
        api_kwargs: dict[str, Any] = {
            "model": model,
            "messages": self._convert_messages(messages),
            "temperature": temperature,
        }

        if max_tokens:
            api_kwargs["max_tokens"] = max_tokens

        # Handle structured output (JSON schema)
        if json_schema:
            if self.supports_structured_output(model):
                # Use native structured output
                schema_name = json_schema.get("title", "response")
                # OpenAI requires additionalProperties: false for strict mode
                prepared_schema = self._prepare_schema_for_structured_output(json_schema)
                api_kwargs["response_format"] = {
                    "type": "json_schema",
                    "json_schema": {
                        "name": schema_name,
                        "schema": prepared_schema,
                        "strict": True,
                    },
                }
            elif self.supports_json_mode(model):
                # Fall back to JSON mode
                api_kwargs["response_format"] = {"type": "json_object"}

        # Pass through any additional kwargs
        for key in ["stop", "presence_penalty", "frequency_penalty", "logit_bias", "user", "seed"]:
            if key in kwargs:
                api_kwargs[key] = kwargs[key]

        return api_kwargs

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
            "context" in error_str or "maximum context length" in error_str or "token" in error_str
        ):
            logger.error(
                "Context length exceeded", extra={"provider": self.provider_name, "model": model}
            )
            raise ContextLengthExceededError(self.provider_name, model) from e
        else:
            logger.error("API error: %s", e, extra={"provider": self.provider_name})
            raise ProviderError(f"OpenAI API error: {e}", provider=self.provider_name) from e

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
        api_kwargs = self._build_api_kwargs(
            messages, model, temperature, max_tokens, json_schema, **kwargs
        )

        logger.debug(
            "Starting completion request", extra={"provider": self.provider_name, "model": model}
        )

        try:
            response = self._client.chat.completions.create(**api_kwargs)
        except Exception as e:
            self._handle_api_error(e, model)

        # Validate response has choices
        if not response.choices:
            logger.error(
                "Empty response from API", extra={"provider": self.provider_name, "model": model}
            )
            raise ProviderError("Empty response from API", provider=self.provider_name)

        # Check for content filtering
        if response.choices[0].finish_reason == "content_filter":
            logger.warning("Content filtered", extra={"provider": self.provider_name})
            raise ContentFilteredError(self.provider_name)

        content = response.choices[0].message.content or ""
        usage = None
        if response.usage:
            usage = UsageStats(
                prompt_tokens=response.usage.prompt_tokens,
                completion_tokens=response.usage.completion_tokens,
                total_tokens=response.usage.total_tokens,
            )
            logger.debug(
                "Completion finished",
                extra={
                    "provider": self.provider_name,
                    "model": model,
                    "prompt_tokens": response.usage.prompt_tokens,
                    "completion_tokens": response.usage.completion_tokens,
                },
            )

        return CompletionResponse(
            content=content,
            model=response.model,
            provider=self.provider_name,
            usage=usage,
            raw_response=response,
            finish_reason=response.choices[0].finish_reason,
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
        api_kwargs = self._build_api_kwargs(
            messages, model, temperature, max_tokens, json_schema, **kwargs
        )

        logger.debug(
            "Starting async completion request",
            extra={"provider": self.provider_name, "model": model},
        )

        try:
            response = await self._async_client.chat.completions.create(**api_kwargs)
        except Exception as e:
            self._handle_api_error(e, model)

        # Validate response has choices
        if not response.choices:
            logger.error(
                "Empty response from API", extra={"provider": self.provider_name, "model": model}
            )
            raise ProviderError("Empty response from API", provider=self.provider_name)

        # Check for content filtering
        if response.choices[0].finish_reason == "content_filter":
            logger.warning("Content filtered", extra={"provider": self.provider_name})
            raise ContentFilteredError(self.provider_name)

        content = response.choices[0].message.content or ""
        usage = None
        if response.usage:
            usage = UsageStats(
                prompt_tokens=response.usage.prompt_tokens,
                completion_tokens=response.usage.completion_tokens,
                total_tokens=response.usage.total_tokens,
            )
            logger.debug(
                "Async completion finished",
                extra={
                    "provider": self.provider_name,
                    "model": model,
                    "prompt_tokens": response.usage.prompt_tokens,
                    "completion_tokens": response.usage.completion_tokens,
                },
            )

        return CompletionResponse(
            content=content,
            model=response.model,
            provider=self.provider_name,
            usage=usage,
            raw_response=response,
            finish_reason=response.choices[0].finish_reason,
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
        api_kwargs: dict[str, Any] = {
            "model": model,
            "messages": self._convert_messages(messages),
            "temperature": temperature,
            "stream": True,
        }

        if max_tokens:
            api_kwargs["max_tokens"] = max_tokens

        logger.debug(
            "Starting streaming request", extra={"provider": self.provider_name, "model": model}
        )

        try:
            response = self._client.chat.completions.create(**api_kwargs)
            for chunk in response:
                if chunk.choices and chunk.choices[0].delta.content:
                    yield chunk.choices[0].delta.content
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
        api_kwargs: dict[str, Any] = {
            "model": model,
            "messages": self._convert_messages(messages),
            "temperature": temperature,
            "stream": True,
        }

        if max_tokens:
            api_kwargs["max_tokens"] = max_tokens

        logger.debug(
            "Starting async streaming request",
            extra={"provider": self.provider_name, "model": model},
        )

        try:
            response = await self._async_client.chat.completions.create(**api_kwargs)
            async for chunk in response:
                if chunk.choices and chunk.choices[0].delta.content:
                    yield chunk.choices[0].delta.content
        except Exception as e:
            self._handle_api_error(e, model)

    def supports_json_mode(self, model: str) -> bool:
        """Check if the model supports JSON mode."""
        return any(supported in model for supported in self.JSON_MODE_MODELS)

    def supports_structured_output(self, model: str) -> bool:
        """Check if the model supports structured output with JSON schema."""
        return any(supported in model for supported in self.STRUCTURED_OUTPUT_MODELS)

    def get_model_info(self, model: str) -> dict[str, Any]:
        """Get information about a model."""
        info: dict[str, Any] = {
            "model": model,
            "provider": self.provider_name,
            "supports_json_mode": self.supports_json_mode(model),
            "supports_structured_output": self.supports_structured_output(model),
        }

        # Add known context lengths
        context_lengths = {
            "gpt-4o": 128000,
            "gpt-4o-mini": 128000,
            "gpt-4-turbo": 128000,
            "gpt-4": 8192,
            "gpt-3.5-turbo": 16385,
        }
        for prefix, length in context_lengths.items():
            if model.startswith(prefix):
                info["context_length"] = length
                break

        return info
