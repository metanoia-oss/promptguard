#!/usr/bin/env python3
"""Smoke test for PromptGuard.

Run this after installation to verify everything works:
    python scripts/smoke_test.py

Requires OPENAI_API_KEY environment variable.
"""

import os
import sys
import tempfile
from pathlib import Path


def print_status(msg: str, success: bool = True):
    """Print status with color."""
    symbol = "[PASS]" if success else "[FAIL]"
    color = "\033[92m" if success else "\033[91m"
    reset = "\033[0m"
    print(f"{color}{symbol}{reset} {msg}")


def test_imports():
    """Test that all imports work."""
    print("\n--- Testing Imports ---")

    try:
        from promptguard import allm_call, llm_call

        print_status("Core imports (llm_call, allm_call)")
    except ImportError as e:
        print_status(f"Core imports: {e}", False)
        return False

    try:
        from promptguard import PromptGuardConfig, RetryConfig, ValidationError

        print_status("Config imports")
    except ImportError as e:
        print_status(f"Config imports: {e}", False)
        return False

    try:
        from promptguard.providers import LLMProvider, ProviderRegistry

        print_status("Provider imports")
    except ImportError as e:
        print_status(f"Provider imports: {e}", False)
        return False

    try:
        from promptguard.schemas import PydanticAdapter, create_adapter

        print_status("Schema imports")
    except ImportError as e:
        print_status(f"Schema imports: {e}", False)
        return False

    try:
        from promptguard.testing import SnapshotMatcher, SnapshotStore

        print_status("Testing imports")
    except ImportError as e:
        print_status(f"Testing imports: {e}", False)
        return False

    return True


def test_schema_validation():
    """Test schema validation without API calls."""
    print("\n--- Testing Schema Validation ---")

    from pydantic import BaseModel

    from promptguard.core.validator import OutputValidator

    class Person(BaseModel):
        name: str
        age: int

    validator = OutputValidator.for_schema(Person)

    # Test valid JSON
    try:
        result = validator.validate('{"name": "John", "age": 30}')
        assert result.name == "John"
        assert result.age == 30
        print_status("Valid JSON parsing")
    except Exception as e:
        print_status(f"Valid JSON parsing: {e}", False)
        return False

    # Test JSON in code block
    try:
        result = validator.validate('```json\n{"name": "Jane", "age": 25}\n```')
        assert result.name == "Jane"
        print_status("Code block extraction")
    except Exception as e:
        print_status(f"Code block extraction: {e}", False)
        return False

    # Test invalid JSON
    from promptguard.core.exceptions import ValidationError

    try:
        validator.validate('{"name": "Test"}')  # Missing age
        print_status("Invalid JSON detection", False)
        return False
    except ValidationError:
        print_status("Invalid JSON detection")

    return True


def test_config():
    """Test configuration."""
    print("\n--- Testing Configuration ---")

    from promptguard.core.config import PromptGuardConfig, RetryConfig

    try:
        config = PromptGuardConfig()
        assert config.default_provider == "openai"
        assert config.retry.max_retries == 3
        print_status("Default config creation")
    except Exception as e:
        print_status(f"Default config: {e}", False)
        return False

    try:
        config = PromptGuardConfig(retry=RetryConfig(max_retries=5), default_provider="anthropic")
        assert config.retry.max_retries == 5
        print_status("Custom config creation")
    except Exception as e:
        print_status(f"Custom config: {e}", False)
        return False

    return True


def test_version_store():
    """Test version storage."""
    print("\n--- Testing Version Store ---")

    from datetime import datetime

    from promptguard.core.hashing import PromptHasher, PromptVersion, VersionStore

    with tempfile.TemporaryDirectory() as tmpdir:
        store = VersionStore(Path(tmpdir))
        hasher = PromptHasher()

        try:
            version = PromptVersion(
                hash=hasher.hash_prompt("test prompt"),
                prompt="test prompt",
                model="gpt-4o",
                schema_hash="abc123",
                temperature=0.7,
                created_at=datetime.utcnow(),
            )
            store.save(version)
            print_status("Version save")
        except Exception as e:
            print_status(f"Version save: {e}", False)
            return False

        try:
            loaded = store.load(version.hash)
            assert loaded is not None
            assert loaded.prompt == "test prompt"
            print_status("Version load")
        except Exception as e:
            print_status(f"Version load: {e}", False)
            return False

        try:
            versions = store.list_versions()
            assert len(versions) == 1
            print_status("Version listing")
        except Exception as e:
            print_status(f"Version listing: {e}", False)
            return False

    return True


def test_cli():
    """Test CLI commands."""
    print("\n--- Testing CLI ---")

    import subprocess

    try:
        result = subprocess.run(
            ["promptguard", "--help"],
            capture_output=True,
            text=True,
        )
        assert result.returncode == 0
        assert "PromptGuard" in result.stdout
        print_status("CLI help command")
    except Exception as e:
        print_status(f"CLI help: {e}", False)
        return False

    try:
        result = subprocess.run(
            ["promptguard", "--version"],
            capture_output=True,
            text=True,
        )
        assert result.returncode == 0
        print_status("CLI version command")
    except Exception as e:
        print_status(f"CLI version: {e}", False)
        return False

    with tempfile.TemporaryDirectory() as tmpdir:
        try:
            result = subprocess.run(
                ["promptguard", "init"],
                cwd=tmpdir,
                capture_output=True,
                text=True,
            )
            assert result.returncode == 0
            assert (Path(tmpdir) / ".promptguard").is_dir()
            print_status("CLI init command")
        except Exception as e:
            print_status(f"CLI init: {e}", False)
            return False

    return True


def test_llm_call():
    """Test actual LLM API call."""
    print("\n--- Testing LLM API Call ---")

    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        print_status("Skipped (no OPENAI_API_KEY)")
        return True

    from pydantic import BaseModel

    from promptguard import llm_call

    class Person(BaseModel):
        name: str
        age: int

    try:
        result = llm_call(
            prompt="Extract: John Smith is 30 years old",
            model="gpt-4o-mini",
            schema=Person,
            save_version=False,
        )
        assert result.data.name == "John Smith"
        assert result.data.age == 30
        print_status(f"LLM call successful (name={result.data.name}, age={result.data.age})")
    except Exception as e:
        print_status(f"LLM call: {e}", False)
        return False

    return True


def test_async_call():
    """Test async LLM call."""
    print("\n--- Testing Async LLM Call ---")

    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        print_status("Skipped (no OPENAI_API_KEY)")
        return True

    import asyncio

    from pydantic import BaseModel

    from promptguard import allm_call

    class Person(BaseModel):
        name: str
        age: int

    async def run_async():
        result = await allm_call(
            prompt="Extract: Jane Doe is 25 years old",
            model="gpt-4o-mini",
            schema=Person,
            save_version=False,
        )
        return result

    try:
        result = asyncio.run(run_async())
        assert result.data.name == "Jane Doe"
        assert result.data.age == 25
        print_status(f"Async call successful (name={result.data.name}, age={result.data.age})")
    except Exception as e:
        print_status(f"Async call: {e}", False)
        return False

    return True


def main():
    """Run all smoke tests."""
    print("=" * 60)
    print("PromptGuard Smoke Test")
    print("=" * 60)

    all_passed = True
    tests = [
        test_imports,
        test_config,
        test_schema_validation,
        test_version_store,
        test_cli,
        test_llm_call,
        test_async_call,
    ]

    for test in tests:
        try:
            if not test():
                all_passed = False
        except Exception as e:
            print_status(f"Test {test.__name__} crashed: {e}", False)
            all_passed = False

    print("\n" + "=" * 60)
    if all_passed:
        print("\033[92mAll smoke tests passed!\033[0m")
        print("=" * 60)
        return 0
    else:
        print("\033[91mSome smoke tests failed!\033[0m")
        print("=" * 60)
        return 1


if __name__ == "__main__":
    sys.exit(main())
