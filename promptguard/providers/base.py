"""Base classes for LLM provider abstraction."""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import AsyncIterator, Iterator
from dataclasses import dataclass
from enum import Enum
from typing import Any


class MessageRole(str, Enum):
    """Role of a message in a conversation."""

    SYSTEM = "system"
    USER = "user"
    ASSISTANT = "assistant"


@dataclass
class Message:
    """A message in a conversation.

    Attributes:
        role: The role of the message sender.
        content: The text content of the message.
    """

    role: MessageRole
    content: str

    @classmethod
    def system(cls, content: str) -> Message:
        """Create a system message."""
        return cls(role=MessageRole.SYSTEM, content=content)

    @classmethod
    def user(cls, content: str) -> Message:
        """Create a user message."""
        return cls(role=MessageRole.USER, content=content)

    @classmethod
    def assistant(cls, content: str) -> Message:
        """Create an assistant message."""
        return cls(role=MessageRole.ASSISTANT, content=content)


@dataclass
class UsageStats:
    """Token usage statistics from an LLM call.

    Attributes:
        prompt_tokens: Number of tokens in the prompt.
        completion_tokens: Number of tokens in the completion.
        total_tokens: Total tokens used.
    """

    prompt_tokens: int
    completion_tokens: int
    total_tokens: int

    @property
    def cost_estimate(self) -> float | None:
        """Estimate cost based on token usage (not implemented for all providers)."""
        return None


@dataclass
class CompletionResponse:
    """Response from an LLM completion request.

    Attributes:
        content: The generated text content.
        model: The model that generated the response.
        provider: The provider that handled the request.
        usage: Token usage statistics if available.
        raw_response: The raw response object from the provider.
        finish_reason: Why the model stopped generating (e.g., "stop", "length").
    """

    content: str
    model: str
    provider: str
    usage: UsageStats | None = None
    raw_response: Any | None = None
    finish_reason: str | None = None


class LLMProvider(ABC):
    """Abstract base class for LLM providers.

    All LLM providers must implement this interface to be usable
    with PromptGuard. The interface provides both synchronous and
    asynchronous methods for completion and streaming.

    Subclasses should set the `provider_name` class attribute.

    Example:
        class MyProvider(LLMProvider):
            provider_name = "my_provider"

            def complete(self, messages, model, **kwargs):
                # Implementation
                ...
    """

    provider_name: str = "base"

    @abstractmethod
    def complete(
        self,
        messages: list[Message],
        model: str,
        temperature: float = 0.7,
        max_tokens: int | None = None,
        json_schema: dict[str, Any] | None = None,
        **kwargs: Any,
    ) -> CompletionResponse:
        """Execute a synchronous completion request.

        Args:
            messages: List of messages in the conversation.
            model: Model identifier to use.
            temperature: Sampling temperature (0.0 to 2.0).
            max_tokens: Maximum tokens to generate.
            json_schema: Optional JSON schema for structured output.
            **kwargs: Additional provider-specific parameters.

        Returns:
            CompletionResponse with the generated content.

        Raises:
            ProviderError: If the API call fails.
        """
        ...

    @abstractmethod
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

        Args:
            messages: List of messages in the conversation.
            model: Model identifier to use.
            temperature: Sampling temperature (0.0 to 2.0).
            max_tokens: Maximum tokens to generate.
            json_schema: Optional JSON schema for structured output.
            **kwargs: Additional provider-specific parameters.

        Returns:
            CompletionResponse with the generated content.

        Raises:
            ProviderError: If the API call fails.
        """
        ...

    @abstractmethod
    def stream(
        self,
        messages: list[Message],
        model: str,
        temperature: float = 0.7,
        max_tokens: int | None = None,
        **kwargs: Any,
    ) -> Iterator[str]:
        """Execute a synchronous streaming completion request.

        Args:
            messages: List of messages in the conversation.
            model: Model identifier to use.
            temperature: Sampling temperature.
            max_tokens: Maximum tokens to generate.
            **kwargs: Additional provider-specific parameters.

        Yields:
            String chunks of the generated content.

        Raises:
            ProviderError: If the API call fails.
        """
        ...

    @abstractmethod
    async def astream(
        self,
        messages: list[Message],
        model: str,
        temperature: float = 0.7,
        max_tokens: int | None = None,
        **kwargs: Any,
    ) -> AsyncIterator[str]:
        """Execute an asynchronous streaming completion request.

        Args:
            messages: List of messages in the conversation.
            model: Model identifier to use.
            temperature: Sampling temperature.
            max_tokens: Maximum tokens to generate.
            **kwargs: Additional provider-specific parameters.

        Yields:
            String chunks of the generated content.

        Raises:
            ProviderError: If the API call fails.
        """
        ...

    def supports_json_mode(self, model: str) -> bool:
        """Check if the model supports JSON mode.

        JSON mode ensures the model outputs valid JSON, but doesn't
        guarantee conformance to a specific schema.

        Args:
            model: Model identifier to check.

        Returns:
            True if JSON mode is supported.
        """
        return False

    def supports_structured_output(self, model: str) -> bool:
        """Check if the model supports structured output with JSON schema.

        Structured output ensures the model output conforms to a
        specific JSON schema.

        Args:
            model: Model identifier to check.

        Returns:
            True if structured output is supported.
        """
        return False

    def get_model_info(self, model: str) -> dict[str, Any]:
        """Get information about a model.

        Args:
            model: Model identifier.

        Returns:
            Dictionary with model information (context length, etc.).
        """
        return {"model": model, "provider": self.provider_name}
