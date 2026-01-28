"""Custom configuration examples for PromptGuard.

This example demonstrates:
- Custom retry configuration
- Configuration from files
- Provider configuration

Prerequisites:
    pip install promptguard[openai]
    export OPENAI_API_KEY=your-key
"""

from pathlib import Path

from pydantic import BaseModel, Field

from promptguard import (
    PromptGuardConfig,
    RetryConfig,
    VersioningConfig,
    configure,
    llm_call,
)


class DataPoint(BaseModel):
    """Schema for extracted data points."""

    value: float = Field(description="Numeric value")
    unit: str = Field(description="Unit of measurement")
    context: str = Field(description="Context for the value")


def with_custom_config():
    """Example with custom configuration."""
    # Create custom configuration
    config = PromptGuardConfig(
        retry=RetryConfig(
            max_retries=5,  # More retries
            backoff_strategy="linear",
            base_delay=0.5,
        ),
        versioning=VersioningConfig(
            storage_path=Path("./my_versions"),
            include_temperature_in_hash=True,
        ),
        default_provider="openai",
    )

    # Apply configuration globally
    configure(config)

    # Now all llm_call() will use this config
    result = llm_call(
        prompt="Extract: The temperature is 72.5 degrees Fahrenheit",
        model="gpt-4o-mini",
        schema=DataPoint,
    )

    print(f"Value: {result.data.value} {result.data.unit}")
    print(f"Context: {result.data.context}")


def with_per_call_config():
    """Example with per-call configuration override."""
    # Override retry config for a specific call
    custom_retry = RetryConfig(
        max_retries=10,
        backoff_strategy="exponential",
    )

    result = llm_call(
        prompt="Extract: The distance is 42.195 kilometers",
        model="gpt-4o-mini",
        schema=DataPoint,
        retry_config=custom_retry,  # Override just for this call
    )

    print(f"Value: {result.data.value} {result.data.unit}")


def from_yaml_config():
    """Example loading configuration from YAML file."""
    config_path = Path(".promptguard/config.yaml")

    if config_path.exists():
        config = PromptGuardConfig.from_yaml(config_path)
        configure(config)
        print(f"Loaded config from {config_path}")
    else:
        print(f"No config file at {config_path}, using defaults")


def main():
    print("Custom Configuration Examples\n")

    print("1. With Custom Config:")
    with_custom_config()
    print()

    print("2. Per-Call Override:")
    with_per_call_config()
    print()

    print("3. From YAML:")
    from_yaml_config()


if __name__ == "__main__":
    main()
