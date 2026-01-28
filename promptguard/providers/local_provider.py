"""Local LLM provider implementation for OpenAI-compatible APIs."""

from __future__ import annotations

import os
from typing import Any, AsyncIterator, Iterator, Optional

from promptguard.core.exceptions import ProviderError
from promptguard.providers.base import (
    CompletionResponse,
    LLMProvider,
    Message,
    MessageRole,
    UsageStats,
)
from promptguard.providers.registry import ProviderRegistry


@ProviderRegistry.register("local")
class LocalProvider(LLMProvider):
    """Local LLM provider for OpenAI-compatible APIs.

    Supports local inference servers that implement the OpenAI API format:
        - Ollama (http://localhost:11434/v1)
        - LM Studio (http://localhost:1234/v1)
        - vLLM
        - LocalAI
        - llama.cpp server

    Features:
        - No API key required by default
        - Configurable base URL
        - Streaming support
        - Async support

    Environment Variables:
        LOCAL_LLM_BASE_URL: Base URL for the local server
        LOCAL_LLM_API_KEY: Optional API key if required
    """

    provider_name = "local"

    # Default URLs for common local servers
    DEFAULT_URLS = {
        "ollama": "http://localhost:11434/v1",
        "lmstudio": "http://localhost:1234/v1",
        "vllm": "http://localhost:8000/v1",
        "localai": "http://localhost:8080/v1",
    }

    def __init__(
        self,
        base_url: Optional[str] = None,
        api_key: Optional[str] = None,
        timeout: float = 120.0,
        server_type: Optional[str] = None,
    ) -> None:
        """Initialize local provider.

        Args:
            base_url: Base URL for the local server.
                Falls back to LOCAL_LLM_BASE_URL env var or Ollama default.
            api_key: Optional API key. Falls back to LOCAL_LLM_API_KEY.
                Most local servers don't require this.
            timeout: Request timeout in seconds (default higher for local).
            server_type: Type of server for default URL lookup
                ("ollama", "lmstudio", "vllm", "localai").

        Raises:
            ProviderError: If httpx package is not installed.
        """
        try:
            import httpx
        except ImportError:
            raise ProviderError(
                "httpx package required for local provider. "
                "Run: pip install httpx",
                provider="local",
            )

        # Determine base URL
        if base_url:
            self._base_url = base_url
        elif os.getenv("LOCAL_LLM_BASE_URL"):
            self._base_url = os.getenv("LOCAL_LLM_BASE_URL")
        elif server_type and server_type in self.DEFAULT_URLS:
            self._base_url = self.DEFAULT_URLS[server_type]
        else:
            # Default to Ollama
            self._base_url = self.DEFAULT_URLS["ollama"]

        self._api_key = api_key or os.getenv("LOCAL_LLM_API_KEY", "not-needed")
        self._timeout = timeout

        # Use httpx for HTTP requests
        self._client = httpx.Client(
            base_url=self._base_url,
            timeout=self._timeout,
            headers={"Authorization": f"Bearer {self._api_key}"},
        )
        self._async_client = httpx.AsyncClient(
            base_url=self._base_url,
            timeout=self._timeout,
            headers={"Authorization": f"Bearer {self._api_key}"},
        )

    def _convert_messages(self, messages: list[Message]) -> list[dict[str, str]]:
        """Convert internal Message format to OpenAI API format."""
        return [{"role": m.role.value, "content": m.content} for m in messages]

    def complete(
        self,
        messages: list[Message],
        model: str,
        temperature: float = 0.7,
        max_tokens: Optional[int] = None,
        json_schema: Optional[dict[str, Any]] = None,
        **kwargs: Any,
    ) -> CompletionResponse:
        """Execute a synchronous completion request."""
        payload: dict[str, Any] = {
            "model": model,
            "messages": self._convert_messages(messages),
            "temperature": temperature,
        }

        if max_tokens:
            payload["max_tokens"] = max_tokens

        # Some local servers support JSON mode
        if json_schema:
            payload["response_format"] = {"type": "json_object"}

        try:
            response = self._client.post("/chat/completions", json=payload)
            response.raise_for_status()
            data = response.json()
        except Exception as e:
            raise ProviderError(
                f"Local LLM API error: {e}",
                provider=self.provider_name,
            ) from e

        content = data["choices"][0]["message"]["content"]

        usage = None
        if "usage" in data:
            usage = UsageStats(
                prompt_tokens=data["usage"].get("prompt_tokens", 0),
                completion_tokens=data["usage"].get("completion_tokens", 0),
                total_tokens=data["usage"].get("total_tokens", 0),
            )

        return CompletionResponse(
            content=content,
            model=data.get("model", model),
            provider=self.provider_name,
            usage=usage,
            raw_response=data,
            finish_reason=data["choices"][0].get("finish_reason"),
        )

    async def acomplete(
        self,
        messages: list[Message],
        model: str,
        temperature: float = 0.7,
        max_tokens: Optional[int] = None,
        json_schema: Optional[dict[str, Any]] = None,
        **kwargs: Any,
    ) -> CompletionResponse:
        """Execute an asynchronous completion request."""
        payload: dict[str, Any] = {
            "model": model,
            "messages": self._convert_messages(messages),
            "temperature": temperature,
        }

        if max_tokens:
            payload["max_tokens"] = max_tokens

        if json_schema:
            payload["response_format"] = {"type": "json_object"}

        try:
            response = await self._async_client.post("/chat/completions", json=payload)
            response.raise_for_status()
            data = response.json()
        except Exception as e:
            raise ProviderError(
                f"Local LLM API error: {e}",
                provider=self.provider_name,
            ) from e

        content = data["choices"][0]["message"]["content"]

        usage = None
        if "usage" in data:
            usage = UsageStats(
                prompt_tokens=data["usage"].get("prompt_tokens", 0),
                completion_tokens=data["usage"].get("completion_tokens", 0),
                total_tokens=data["usage"].get("total_tokens", 0),
            )

        return CompletionResponse(
            content=content,
            model=data.get("model", model),
            provider=self.provider_name,
            usage=usage,
            raw_response=data,
            finish_reason=data["choices"][0].get("finish_reason"),
        )

    def stream(
        self,
        messages: list[Message],
        model: str,
        temperature: float = 0.7,
        max_tokens: Optional[int] = None,
        **kwargs: Any,
    ) -> Iterator[str]:
        """Execute a synchronous streaming completion request."""
        payload: dict[str, Any] = {
            "model": model,
            "messages": self._convert_messages(messages),
            "temperature": temperature,
            "stream": True,
        }

        if max_tokens:
            payload["max_tokens"] = max_tokens

        try:
            with self._client.stream(
                "POST", "/chat/completions", json=payload
            ) as response:
                response.raise_for_status()
                for line in response.iter_lines():
                    if line.startswith("data: "):
                        data_str = line[6:]
                        if data_str == "[DONE]":
                            break
                        try:
                            import json
                            data = json.loads(data_str)
                            content = data["choices"][0]["delta"].get("content", "")
                            if content:
                                yield content
                        except Exception:
                            continue
        except Exception as e:
            raise ProviderError(
                f"Local LLM API streaming error: {e}",
                provider=self.provider_name,
            ) from e

    async def astream(
        self,
        messages: list[Message],
        model: str,
        temperature: float = 0.7,
        max_tokens: Optional[int] = None,
        **kwargs: Any,
    ) -> AsyncIterator[str]:
        """Execute an asynchronous streaming completion request."""
        payload: dict[str, Any] = {
            "model": model,
            "messages": self._convert_messages(messages),
            "temperature": temperature,
            "stream": True,
        }

        if max_tokens:
            payload["max_tokens"] = max_tokens

        try:
            async with self._async_client.stream(
                "POST", "/chat/completions", json=payload
            ) as response:
                response.raise_for_status()
                async for line in response.aiter_lines():
                    if line.startswith("data: "):
                        data_str = line[6:]
                        if data_str == "[DONE]":
                            break
                        try:
                            import json
                            data = json.loads(data_str)
                            content = data["choices"][0]["delta"].get("content", "")
                            if content:
                                yield content
                        except Exception:
                            continue
        except Exception as e:
            raise ProviderError(
                f"Local LLM API streaming error: {e}",
                provider=self.provider_name,
            ) from e

    def list_models(self) -> list[str]:
        """List available models from the local server.

        Returns:
            List of model names.
        """
        try:
            response = self._client.get("/models")
            response.raise_for_status()
            data = response.json()
            return [m["id"] for m in data.get("data", [])]
        except Exception:
            return []

    def get_model_info(self, model: str) -> dict[str, Any]:
        """Get information about a model."""
        return {
            "model": model,
            "provider": self.provider_name,
            "base_url": self._base_url,
            "supports_json_mode": True,  # Most local servers support this
            "supports_structured_output": False,
        }
