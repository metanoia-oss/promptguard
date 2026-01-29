"""Main execution engine for PromptGuard."""

from __future__ import annotations

import json
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any, Generic, TypeVar

from promptguard.core.config import PromptGuardConfig, RetryConfig
from promptguard.core.hashing import PromptHasher, PromptVersion, VersionStore
from promptguard.core.logging import get_logger
from promptguard.core.repair import AsyncRepairLoop, RepairLoop
from promptguard.core.validator import OutputValidator
from promptguard.providers.base import LLMProvider, Message, MessageRole
from promptguard.providers.registry import ProviderRegistry
from promptguard.schemas.adapters import create_adapter

T = TypeVar("T")

logger = get_logger(__name__)


@dataclass
class LLMCallResult(Generic[T]):
    """Result of a llm_call execution.

    Attributes:
        data: The validated, typed output data.
        raw_response: The raw string response from the LLM.
        model: The model that generated the response.
        provider: The provider used.
        version_hash: Hash of this prompt version (if saved).
        repair_attempts: Number of repair attempts made.
        usage: Token usage statistics if available.
        duration_ms: Total execution time in milliseconds.
    """

    data: T
    raw_response: str
    model: str
    provider: str
    version_hash: str | None
    repair_attempts: int
    usage: dict[str, int] | None
    duration_ms: float


class PromptGuardEngine:
    """Main execution orchestrator for PromptGuard.

    The engine coordinates:
    - Provider selection and management
    - Schema validation
    - Automatic repair loops
    - Prompt versioning

    Example:
        engine = PromptGuardEngine()
        result = engine.call(
            prompt="Extract data from...",
            model="gpt-4o",
            schema=MyPydanticModel,
        )
    """

    def __init__(self, config: PromptGuardConfig | None = None) -> None:
        """Initialize the engine.

        Args:
            config: Configuration instance. Falls back to environment config.
        """
        self.config = config or PromptGuardConfig.from_env()
        self.hasher = PromptHasher(
            algorithm=self.config.versioning.hash_algorithm,
            include_model=self.config.versioning.include_model_in_hash,
            include_temperature=self.config.versioning.include_temperature_in_hash,
        )
        self.version_store = VersionStore(self.config.versioning.storage_path)

    def _get_provider(self, model: str) -> tuple[LLMProvider, str]:
        """Determine provider from model string.

        Supports formats:
        - "provider:model" (e.g., "anthropic:claude-3-opus")
        - "model" (auto-detected, e.g., "gpt-4o" → OpenAI)

        Args:
            model: Model identifier string.

        Returns:
            Tuple of (provider instance, model name).
        """
        # Check for explicit provider prefix
        if ":" in model:
            provider_name, model_name = model.split(":", 1)
        else:
            # Auto-detect provider from model name
            model_lower = model.lower()

            if any(x in model_lower for x in ["gpt-", "o1-", "o3-", "davinci", "curie"]):
                provider_name = "openai"
            elif any(x in model_lower for x in ["claude", "opus", "sonnet", "haiku"]):
                provider_name = "anthropic"
            elif any(x in model_lower for x in ["gemini", "palm"]):
                provider_name = "gemini"
            elif any(x in model_lower for x in ["llama", "mistral", "mixtral", "phi"]):
                provider_name = "local"
            else:
                provider_name = self.config.default_provider

            model_name = model

        provider = ProviderRegistry.get(provider_name)
        return provider, model_name

    def _build_system_prompt(
        self,
        schema: dict[str, Any],
        user_system_prompt: str | None,
    ) -> str:
        """Build system prompt with schema instructions.

        Args:
            schema: JSON schema for the expected output.
            user_system_prompt: Optional user-provided system prompt.

        Returns:
            Complete system prompt string.
        """
        schema_str = json.dumps(schema, indent=2)

        schema_instruction = f"""You must respond with valid JSON that matches this schema:
```json
{schema_str}
```
Do not include any text before or after the JSON. Do not wrap the JSON in markdown code blocks.
Respond with ONLY the JSON object."""

        if user_system_prompt:
            return f"{user_system_prompt}\n\n{schema_instruction}"
        return schema_instruction

    def call(
        self,
        prompt: str,
        model: str,
        schema: Any,
        system_prompt: str | None = None,
        temperature: float = 0.7,
        max_tokens: int | None = None,
        retry_config: RetryConfig | None = None,
        save_version: bool = True,
        metadata: dict[str, Any] | None = None,
    ) -> LLMCallResult[T]:
        """Execute an LLM call with schema validation and auto-repair.

        Args:
            prompt: The user prompt to send to the LLM.
            model: Model identifier (e.g., "gpt-4o", "anthropic:claude-3-opus").
            schema: Pydantic model, TypedDict, dataclass, or JSON Schema dict.
            system_prompt: Optional system prompt to prepend.
            temperature: Sampling temperature (0.0 to 2.0).
            max_tokens: Maximum tokens to generate.
            retry_config: Override default retry configuration.
            save_version: Whether to save this execution as a version.
            metadata: Additional metadata to store with version.

        Returns:
            LLMCallResult with validated data and metadata.

        Raises:
            RepairExhaustedError: If validation fails after all repair attempts.
            ProviderError: If there's an error with the LLM provider.

        Example:
            from pydantic import BaseModel

            class Person(BaseModel):
                name: str
                age: int

            result = engine.call(
                prompt="Extract: John is 30 years old",
                model="gpt-4o",
                schema=Person,
            )
            print(result.data.name)  # "John"
        """
        start_time = time.time()

        # Setup components
        provider, model_name = self._get_provider(model)
        logger.debug(
            "Starting LLM call",
            extra={"provider": provider.provider_name, "model": model_name},
        )
        adapter = create_adapter(schema)
        validator = OutputValidator(adapter)
        json_schema = adapter.to_json_schema()

        # Build messages
        full_system_prompt = self._build_system_prompt(json_schema, system_prompt)
        messages = [
            Message(role=MessageRole.SYSTEM, content=full_system_prompt),
            Message(role=MessageRole.USER, content=prompt),
        ]

        # Make initial LLM call
        use_structured = provider.supports_structured_output(model_name)
        response = provider.complete(
            messages=messages,
            model=model_name,
            temperature=temperature,
            max_tokens=max_tokens,
            json_schema=json_schema if use_structured else None,
        )

        # Setup repair loop
        retry_cfg = retry_config or self.config.retry
        repair_loop = RepairLoop(retry_cfg, validator)

        def repair_complete_fn(repair_prompt: str) -> str:
            """Function called for each repair attempt."""
            repair_messages = [
                Message(role=MessageRole.SYSTEM, content=repair_loop.REPAIR_SYSTEM_PROMPT),
                Message(role=MessageRole.USER, content=repair_prompt),
            ]
            repair_response = provider.complete(
                messages=repair_messages,
                model=model_name,
                # Lower temperature for repair attempts
                temperature=max(0.3, temperature - 0.2),
                max_tokens=max_tokens,
                json_schema=json_schema if use_structured else None,
            )
            return repair_response.content

        # Run validation/repair
        repair_result = repair_loop.run(
            complete_fn=repair_complete_fn,
            initial_output=response.content,
            original_prompt=prompt,
        )

        # Version management
        version_hash = None
        if save_version:
            version = PromptVersion(
                hash=self.hasher.hash_prompt(prompt, model, temperature, json_schema),
                prompt=prompt,
                model=model,
                schema_hash=self.hasher.hash_schema(json_schema),
                temperature=temperature,
                created_at=datetime.now(timezone.utc),
                metadata=metadata or {},
            )
            version_hash = self.version_store.save(version)

        # Build result
        usage_dict = None
        if response.usage:
            usage_dict = {
                "prompt_tokens": response.usage.prompt_tokens,
                "completion_tokens": response.usage.completion_tokens,
                "total_tokens": response.usage.total_tokens,
            }

        duration_ms = (time.time() - start_time) * 1000
        logger.info(
            "LLM call completed",
            extra={
                "provider": response.provider,
                "model": response.model,
                "duration_ms": duration_ms,
                "repair_attempts": len(repair_result.attempts),
            },
        )

        return LLMCallResult(
            data=repair_result.result,
            raw_response=response.content,
            model=response.model,
            provider=response.provider,
            version_hash=version_hash,
            repair_attempts=len(repair_result.attempts),
            usage=usage_dict,
            duration_ms=duration_ms,
        )

    async def acall(
        self,
        prompt: str,
        model: str,
        schema: Any,
        system_prompt: str | None = None,
        temperature: float = 0.7,
        max_tokens: int | None = None,
        retry_config: RetryConfig | None = None,
        save_version: bool = True,
        metadata: dict[str, Any] | None = None,
    ) -> LLMCallResult[T]:
        """Async version of call().

        See call() for full documentation.
        """
        start_time = time.time()

        # Setup components
        provider, model_name = self._get_provider(model)
        logger.debug(
            "Starting async LLM call",
            extra={"provider": provider.provider_name, "model": model_name},
        )
        adapter = create_adapter(schema)
        validator = OutputValidator(adapter)
        json_schema = adapter.to_json_schema()

        # Build messages
        full_system_prompt = self._build_system_prompt(json_schema, system_prompt)
        messages = [
            Message(role=MessageRole.SYSTEM, content=full_system_prompt),
            Message(role=MessageRole.USER, content=prompt),
        ]

        # Make initial LLM call
        use_structured = provider.supports_structured_output(model_name)
        response = await provider.acomplete(
            messages=messages,
            model=model_name,
            temperature=temperature,
            max_tokens=max_tokens,
            json_schema=json_schema if use_structured else None,
        )

        # Setup async repair loop
        retry_cfg = retry_config or self.config.retry
        repair_loop = AsyncRepairLoop(retry_cfg, validator)

        async def async_repair_fn(repair_prompt: str) -> str:
            """Async function called for each repair attempt."""
            repair_messages = [
                Message(role=MessageRole.SYSTEM, content=repair_loop.REPAIR_SYSTEM_PROMPT),
                Message(role=MessageRole.USER, content=repair_prompt),
            ]
            repair_response = await provider.acomplete(
                messages=repair_messages,
                model=model_name,
                temperature=max(0.3, temperature - 0.2),
                max_tokens=max_tokens,
                json_schema=json_schema if use_structured else None,
            )
            return repair_response.content

        # Run validation/repair
        repair_result = await repair_loop.run(
            complete_fn=async_repair_fn,
            initial_output=response.content,
            original_prompt=prompt,
        )

        # Version management
        version_hash = None
        if save_version:
            version = PromptVersion(
                hash=self.hasher.hash_prompt(prompt, model, temperature, json_schema),
                prompt=prompt,
                model=model,
                schema_hash=self.hasher.hash_schema(json_schema),
                temperature=temperature,
                created_at=datetime.now(timezone.utc),
                metadata=metadata or {},
            )
            version_hash = self.version_store.save(version)

        # Build result
        usage_dict = None
        if response.usage:
            usage_dict = {
                "prompt_tokens": response.usage.prompt_tokens,
                "completion_tokens": response.usage.completion_tokens,
                "total_tokens": response.usage.total_tokens,
            }

        duration_ms = (time.time() - start_time) * 1000
        logger.info(
            "Async LLM call completed",
            extra={
                "provider": response.provider,
                "model": response.model,
                "duration_ms": duration_ms,
                "repair_attempts": len(repair_result.attempts),
            },
        )

        return LLMCallResult(
            data=repair_result.result,
            raw_response=response.content,
            model=response.model,
            provider=response.provider,
            version_hash=version_hash,
            repair_attempts=len(repair_result.attempts),
            usage=usage_dict,
            duration_ms=duration_ms,
        )


# Global engine instance
_engine: PromptGuardEngine | None = None


def get_engine() -> PromptGuardEngine:
    """Get the global engine instance.

    Creates the engine on first call with default configuration.

    Returns:
        PromptGuardEngine instance.
    """
    global _engine
    if _engine is None:
        _engine = PromptGuardEngine()
    return _engine


def configure(config: PromptGuardConfig) -> None:
    """Configure the global engine.

    Args:
        config: Configuration to use.

    Example:
        from promptguard import configure, PromptGuardConfig, RetryConfig

        configure(PromptGuardConfig(
            retry=RetryConfig(max_retries=5),
            default_provider="anthropic",
        ))
    """
    global _engine
    _engine = PromptGuardEngine(config)


def llm_call(
    prompt: str,
    model: str,
    schema: Any,
    **kwargs: Any,
) -> LLMCallResult:
    """Execute an LLM call with schema validation and auto-repair.

    This is the main entry point for PromptGuard. It guarantees that
    the returned data conforms to the provided schema.

    Args:
        prompt: The user prompt to send to the LLM.
        model: Model identifier (e.g., "gpt-4o", "anthropic:claude-3-opus").
        schema: Pydantic model, TypedDict, dataclass, or JSON Schema dict.
        **kwargs: Additional options passed to engine.call().

    Returns:
        LLMCallResult with validated data and metadata.

    Raises:
        RepairExhaustedError: If validation fails after all repair attempts.
        ProviderError: If there's an error with the LLM provider.

    Example:
        from promptguard import llm_call
        from pydantic import BaseModel

        class Person(BaseModel):
            name: str
            age: int

        result = llm_call(
            prompt="Extract: John is 30 years old",
            model="gpt-4o",
            schema=Person,
        )
        print(result.data.name)  # "John"
        print(result.data.age)   # 30
    """
    return get_engine().call(prompt=prompt, model=model, schema=schema, **kwargs)


async def allm_call(
    prompt: str,
    model: str,
    schema: Any,
    **kwargs: Any,
) -> LLMCallResult:
    """Async version of llm_call.

    See llm_call() for full documentation.

    Example:
        import asyncio
        from promptguard import allm_call
        from pydantic import BaseModel

        class Person(BaseModel):
            name: str
            age: int

        async def main():
            result = await allm_call(
                prompt="Extract: John is 30 years old",
                model="gpt-4o",
                schema=Person,
            )
            print(result.data)

        asyncio.run(main())
    """
    return await get_engine().acall(prompt=prompt, model=model, schema=schema, **kwargs)
