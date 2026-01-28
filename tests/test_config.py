"""Tests for configuration module."""

import tempfile
from pathlib import Path

import pytest

from promptguard.core.config import (
    LoggingConfig,
    PromptGuardConfig,
    RetryConfig,
    VersioningConfig,
)


class TestRetryConfig:
    """Tests for RetryConfig."""

    def test_default_values(self):
        config = RetryConfig()
        assert config.max_retries == 3
        assert config.backoff_strategy == "exponential"
        assert config.base_delay == 1.0
        assert config.max_delay == 60.0
        assert config.jitter is True

    def test_custom_values(self):
        config = RetryConfig(
            max_retries=5,
            backoff_strategy="linear",
            base_delay=0.5,
        )
        assert config.max_retries == 5
        assert config.backoff_strategy == "linear"
        assert config.base_delay == 0.5


class TestVersioningConfig:
    """Tests for VersioningConfig."""

    def test_default_values(self):
        config = VersioningConfig()
        assert config.storage_path == Path(".promptguard/versions")
        assert config.hash_algorithm == "sha256"
        assert config.include_model_in_hash is True
        assert config.include_temperature_in_hash is False


class TestPromptGuardConfig:
    """Tests for PromptGuardConfig."""

    def test_default_values(self):
        config = PromptGuardConfig()
        assert config.default_provider == "openai"
        assert isinstance(config.retry, RetryConfig)
        assert isinstance(config.versioning, VersioningConfig)
        assert isinstance(config.logging, LoggingConfig)

    def test_from_env(self, monkeypatch):
        monkeypatch.setenv("PROMPTGUARD_MAX_RETRIES", "5")
        monkeypatch.setenv("PROMPTGUARD_DEFAULT_PROVIDER", "anthropic")

        config = PromptGuardConfig.from_env()
        assert config.retry.max_retries == 5
        assert config.default_provider == "anthropic"

    def test_to_yaml_and_from_yaml(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            config_path = Path(tmpdir) / "config.yaml"

            # Create and save config
            original = PromptGuardConfig(
                default_provider="anthropic",
                retry=RetryConfig(max_retries=7),
            )
            original.to_yaml(config_path)

            # Load and compare
            loaded = PromptGuardConfig.from_yaml(config_path)
            assert loaded.default_provider == "anthropic"
            assert loaded.retry.max_retries == 7
