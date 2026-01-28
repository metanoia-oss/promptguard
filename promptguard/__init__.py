"""PromptGuard - Reliable, structured, production-safe LLM outputs."""

from promptguard.core.config import (
    LoggingConfig,
    PromptGuardConfig,
    RetryConfig,
    VersioningConfig,
)
from promptguard.core.exceptions import (
    PromptGuardError,
    ProviderError,
    ProviderNotFoundError,
    RepairExhaustedError,
    SchemaError,
    ValidationError,
)

__version__ = "0.1.0"

__all__ = [
    # Main API
    "llm_call",
    "allm_call",
    # Engine
    "PromptGuardEngine",
    "LLMCallResult",
    "get_engine",
    "configure",
    # Config
    "PromptGuardConfig",
    "RetryConfig",
    "VersioningConfig",
    "LoggingConfig",
    # Exceptions
    "PromptGuardError",
    "ValidationError",
    "RepairExhaustedError",
    "ProviderError",
    "ProviderNotFoundError",
    "SchemaError",
    # Versioning
    "PromptVersion",
    "PromptHasher",
]


def __getattr__(name: str):
    """Lazy import for heavy modules."""
    if name in (
        "llm_call",
        "allm_call",
        "PromptGuardEngine",
        "LLMCallResult",
        "get_engine",
        "configure",
    ):
        from promptguard.core.engine import (
            LLMCallResult,
            PromptGuardEngine,
            allm_call,
            configure,
            get_engine,
            llm_call,
        )

        return {
            "llm_call": llm_call,
            "allm_call": allm_call,
            "PromptGuardEngine": PromptGuardEngine,
            "LLMCallResult": LLMCallResult,
            "get_engine": get_engine,
            "configure": configure,
        }[name]

    if name in ("PromptVersion", "PromptHasher"):
        from promptguard.core.hashing import PromptHasher, PromptVersion

        return {"PromptVersion": PromptVersion, "PromptHasher": PromptHasher}[name]

    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
