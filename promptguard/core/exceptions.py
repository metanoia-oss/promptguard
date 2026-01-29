"""Exception hierarchy for PromptGuard."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    pass


class PromptGuardError(Exception):
    """Base exception for all PromptGuard errors.

    All exceptions raised by PromptGuard inherit from this class,
    allowing users to catch all PromptGuard-related errors with
    a single except clause.
    """

    pass


class ValidationError(PromptGuardError):
    """Raised when LLM output fails schema validation.

    Attributes:
        raw_output: The raw output string from the LLM.
        errors: List of validation error details.
    """

    def __init__(
        self,
        message: str,
        raw_output: str,
        errors: list[dict[str, Any]],
    ) -> None:
        super().__init__(message)
        self.raw_output = raw_output
        self.errors = errors

    def __str__(self) -> str:
        error_details = "\n".join(f"  - {err.get('msg', str(err))}" for err in self.errors[:5])
        if len(self.errors) > 5:
            error_details += f"\n  ... and {len(self.errors) - 5} more errors"
        return f"{self.args[0]}\nErrors:\n{error_details}"


class RepairExhaustedError(PromptGuardError):
    """Raised when maximum repair attempts are exhausted.

    Attributes:
        attempts: Number of repair attempts made.
        last_error: The last validation error encountered.
    """

    def __init__(
        self,
        message: str,
        attempts: int,
        last_error: ValidationError,
    ) -> None:
        super().__init__(message)
        self.attempts = attempts
        self.last_error = last_error

    def __str__(self) -> str:
        return (
            f"{self.args[0]}\n"
            f"Total attempts: {self.attempts}\n"
            f"Last error: {self.last_error.args[0]}"
        )


class ProviderError(PromptGuardError):
    """Raised when an LLM provider returns an error.

    Attributes:
        provider: Name of the provider that raised the error.
        status_code: HTTP status code if applicable.
        response_body: Response body from the provider if available.
    """

    def __init__(
        self,
        message: str,
        provider: str,
        status_code: int | None = None,
        response_body: str | None = None,
    ) -> None:
        super().__init__(message)
        self.provider = provider
        self.status_code = status_code
        self.response_body = response_body

    def __str__(self) -> str:
        parts = [f"[{self.provider}] {self.args[0]}"]
        if self.status_code:
            parts.append(f"Status code: {self.status_code}")
        return "\n".join(parts)


class ProviderNotFoundError(PromptGuardError):
    """Raised when a requested provider is not installed or available.

    Attributes:
        provider: Name of the provider that was requested.
        install_hint: Suggested installation command.
    """

    def __init__(
        self,
        provider: str,
        install_hint: str | None = None,
    ) -> None:
        self.provider = provider
        self.install_hint = install_hint or f"pip install promptguard[{provider}]"
        super().__init__(f"Provider '{provider}' not found. Install with: {self.install_hint}")


class SchemaError(PromptGuardError):
    """Raised when a schema is invalid or unsupported.

    Attributes:
        schema_type: The type of the invalid schema.
        reason: Explanation of why the schema is invalid.
    """

    def __init__(
        self,
        message: str,
        schema_type: str | None = None,
        reason: str | None = None,
    ) -> None:
        super().__init__(message)
        self.schema_type = schema_type
        self.reason = reason


class ConfigurationError(PromptGuardError):
    """Raised when there's an error in configuration.

    Attributes:
        config_key: The configuration key that caused the error.
    """

    def __init__(
        self,
        message: str,
        config_key: str | None = None,
    ) -> None:
        super().__init__(message)
        self.config_key = config_key


class TimeoutError(PromptGuardError):
    """Request timed out.

    Attributes:
        provider: Name of the provider that timed out.
        timeout_seconds: The timeout duration in seconds.
    """

    def __init__(self, provider: str, timeout_seconds: float) -> None:
        self.provider = provider
        self.timeout_seconds = timeout_seconds
        super().__init__(
            f"[{provider}] Request timeout after {timeout_seconds}s. "
            "Check network connectivity or increase timeout."
        )


class RateLimitError(ProviderError):
    """API rate limit (429) exceeded.

    Attributes:
        retry_after: Suggested retry delay in seconds, if provided.
    """

    def __init__(self, provider: str, retry_after: int | None = None) -> None:
        self.retry_after = retry_after
        msg = "Rate limit exceeded"
        if retry_after:
            msg += f". Retry after {retry_after}s"
        super().__init__(msg, provider=provider, status_code=429)


class AuthenticationError(ProviderError):
    """API authentication failed (401)."""

    def __init__(self, provider: str) -> None:
        super().__init__(
            f"Invalid API credentials. Check your {provider.upper()}_API_KEY.",
            provider=provider,
            status_code=401,
        )


class ModelNotFoundError(ProviderError):
    """Requested model doesn't exist (404).

    Attributes:
        model: The model name that was not found.
    """

    def __init__(self, model: str, provider: str) -> None:
        self.model = model
        super().__init__(
            f"Model '{model}' not found. Verify model name and API access.",
            provider=provider,
            status_code=404,
        )


class ContextLengthExceededError(ProviderError):
    """Prompt exceeds model context length.

    Attributes:
        model: The model that rejected the prompt.
    """

    def __init__(self, provider: str, model: str, details: str = "") -> None:
        self.model = model
        msg = f"Prompt too long for {model}"
        if details:
            msg += f". {details}"
        super().__init__(msg, provider=provider, status_code=400)


class ContentFilteredError(ProviderError):
    """Content blocked by safety filters.

    Attributes:
        reason: The reason for content filtering, if provided.
    """

    def __init__(self, provider: str, reason: str | None = None) -> None:
        self.reason = reason
        msg = f"Content filtered by {provider} safety policy"
        if reason:
            msg += f": {reason}"
        super().__init__(msg, provider=provider, status_code=400)
