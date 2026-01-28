"""Configuration classes for PromptGuard."""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal

import yaml


@dataclass
class RetryConfig:
    """Configuration for automatic repair loop.

    Attributes:
        max_retries: Maximum number of repair attempts before giving up.
        backoff_strategy: Strategy for delays between retries.
        base_delay: Base delay in seconds before first retry.
        max_delay: Maximum delay in seconds between retries.
        jitter: Whether to add random jitter to delays.
    """

    max_retries: int = 3
    backoff_strategy: Literal["exponential", "linear", "fixed"] = "exponential"
    base_delay: float = 1.0
    max_delay: float = 60.0
    jitter: bool = True


@dataclass
class VersioningConfig:
    """Configuration for prompt versioning.

    Attributes:
        storage_path: Directory path for storing version files.
        hash_algorithm: Hash algorithm to use for versioning.
        include_model_in_hash: Whether to include model name in hash.
        include_temperature_in_hash: Whether to include temperature in hash.
    """

    storage_path: Path = field(default_factory=lambda: Path(".promptguard/versions"))
    hash_algorithm: Literal["sha256", "sha1", "md5"] = "sha256"
    include_model_in_hash: bool = True
    include_temperature_in_hash: bool = False


@dataclass
class LoggingConfig:
    """Observability and logging configuration.

    Attributes:
        level: Logging level.
        format: Output format for logs.
        log_prompts: Whether to log prompt text (privacy consideration).
        log_responses: Whether to log response text (privacy consideration).
    """

    level: Literal["DEBUG", "INFO", "WARNING", "ERROR"] = "INFO"
    format: Literal["json", "text"] = "text"
    log_prompts: bool = False
    log_responses: bool = False


@dataclass
class PromptGuardConfig:
    """Main configuration container for PromptGuard.

    Attributes:
        retry: Configuration for retry/repair behavior.
        versioning: Configuration for prompt versioning.
        logging: Configuration for logging and observability.
        default_provider: Default LLM provider to use.
    """

    retry: RetryConfig = field(default_factory=RetryConfig)
    versioning: VersioningConfig = field(default_factory=VersioningConfig)
    logging: LoggingConfig = field(default_factory=LoggingConfig)
    default_provider: str = "openai"

    @classmethod
    def from_env(cls) -> PromptGuardConfig:
        """Load configuration from environment variables.

        Environment variables:
            PROMPTGUARD_MAX_RETRIES: Maximum retry attempts (default: 3)
            PROMPTGUARD_DEFAULT_PROVIDER: Default provider (default: "openai")
            PROMPTGUARD_LOG_LEVEL: Logging level (default: "INFO")
            PROMPTGUARD_STORAGE_PATH: Version storage path

        Returns:
            PromptGuardConfig instance with values from environment.
        """
        storage_path = os.getenv("PROMPTGUARD_STORAGE_PATH")

        return cls(
            retry=RetryConfig(
                max_retries=int(os.getenv("PROMPTGUARD_MAX_RETRIES", "3")),
            ),
            versioning=VersioningConfig(
                storage_path=Path(storage_path) if storage_path else Path(".promptguard/versions"),
            ),
            logging=LoggingConfig(
                level=os.getenv("PROMPTGUARD_LOG_LEVEL", "INFO"),  # type: ignore
            ),
            default_provider=os.getenv("PROMPTGUARD_DEFAULT_PROVIDER", "openai"),
        )

    @classmethod
    def from_yaml(cls, path: Path | str) -> PromptGuardConfig:
        """Load configuration from a YAML file.

        Args:
            path: Path to YAML configuration file.

        Returns:
            PromptGuardConfig instance with values from file.

        Example YAML structure:
            default_provider: openai
            retry:
              max_retries: 3
              backoff_strategy: exponential
            versioning:
              storage_path: .promptguard/versions
              hash_algorithm: sha256
            logging:
              level: INFO
              format: text
        """
        path = Path(path)
        with open(path) as f:
            data = yaml.safe_load(f) or {}

        retry_data = data.get("retry", {})
        versioning_data = data.get("versioning", {})
        logging_data = data.get("logging", {})

        # Handle storage_path conversion
        if "storage_path" in versioning_data:
            versioning_data["storage_path"] = Path(versioning_data["storage_path"])

        return cls(
            retry=RetryConfig(**retry_data) if retry_data else RetryConfig(),
            versioning=VersioningConfig(**versioning_data)
            if versioning_data
            else VersioningConfig(),
            logging=LoggingConfig(**logging_data) if logging_data else LoggingConfig(),
            default_provider=data.get("default_provider", "openai"),
        )

    def to_yaml(self, path: Path | str) -> None:
        """Save configuration to a YAML file.

        Args:
            path: Path to save YAML configuration file.
        """
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)

        data = {
            "default_provider": self.default_provider,
            "retry": {
                "max_retries": self.retry.max_retries,
                "backoff_strategy": self.retry.backoff_strategy,
                "base_delay": self.retry.base_delay,
                "max_delay": self.retry.max_delay,
                "jitter": self.retry.jitter,
            },
            "versioning": {
                "storage_path": str(self.versioning.storage_path),
                "hash_algorithm": self.versioning.hash_algorithm,
                "include_model_in_hash": self.versioning.include_model_in_hash,
                "include_temperature_in_hash": self.versioning.include_temperature_in_hash,
            },
            "logging": {
                "level": self.logging.level,
                "format": self.logging.format,
                "log_prompts": self.logging.log_prompts,
                "log_responses": self.logging.log_responses,
            },
        }

        with open(path, "w") as f:
            yaml.dump(data, f, default_flow_style=False, sort_keys=False)
