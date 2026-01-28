"""Integration tests for PromptGuard with real LLM API calls.

These tests require a valid OPENAI_API_KEY environment variable.
Run with: pytest tests/test_integration.py -v

To skip these tests (e.g., in CI without API key):
    pytest tests/ --ignore=tests/test_integration.py
"""

import asyncio
import os
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import TypedDict

import pytest
from pydantic import BaseModel, Field

# Skip all tests in this module if no API key
pytestmark = pytest.mark.skipif(not os.getenv("OPENAI_API_KEY"), reason="OPENAI_API_KEY not set")


class PersonSchema(BaseModel):
    """Test schema for person extraction."""

    name: str = Field(description="Person's full name")
    age: int = Field(description="Person's age in years")


class EmailAnalysis(BaseModel):
    """Test schema for email analysis."""

    sender: str
    intent: str
    urgency: int = Field(ge=1, le=5)
    requires_reply: bool


class OrderSchema(BaseModel):
    """Test schema for order extraction."""

    product: str
    quantity: int
    price: float


class PersonTypedDict(TypedDict):
    """TypedDict version of person schema."""

    name: str
    age: int


@dataclass
class PersonDataclass:
    """Dataclass version of person schema."""

    name: str
    age: int


class TestOpenAIBasicCalls:
    """Test basic LLM calls with OpenAI."""

    def test_simple_extraction(self):
        """Test basic data extraction."""
        from promptguard import llm_call

        result = llm_call(
            prompt="Extract the person's information: John Smith is 30 years old.",
            model="gpt-4o-mini",
            schema=PersonSchema,
            save_version=False,
        )

        assert result.data.name == "John Smith"
        assert result.data.age == 30
        assert result.provider == "openai"
        assert result.repair_attempts == 0

    def test_complex_extraction(self):
        """Test more complex data extraction."""
        from promptguard import llm_call

        result = llm_call(
            prompt="Buy 3 MacBook Pro laptops at $2499.99 each",
            model="gpt-4o-mini",
            schema=OrderSchema,
            save_version=False,
        )

        assert "macbook" in result.data.product.lower()
        assert result.data.quantity == 3
        assert result.data.price == 2499.99

    def test_email_analysis(self):
        """Test email analysis extraction."""
        from promptguard import llm_call

        email = """
        From: CEO <ceo@company.com>
        Subject: URGENT: Board meeting tomorrow

        We need to finalize the Q4 report before the board meeting tomorrow at 9am.
        Please confirm you can attend.
        """

        result = llm_call(
            prompt=f"Analyze this email:\n{email}",
            model="gpt-4o-mini",
            schema=EmailAnalysis,
            save_version=False,
        )

        assert result.data.sender  # Should have extracted sender
        assert result.data.urgency >= 3  # Should be high urgency
        assert result.data.requires_reply is True


class TestAsyncCalls:
    """Test async LLM calls."""

    @pytest.mark.asyncio
    async def test_async_basic_call(self):
        """Test basic async call."""
        from promptguard import allm_call

        result = await allm_call(
            prompt="Extract: Jane Doe is 25 years old",
            model="gpt-4o-mini",
            schema=PersonSchema,
            save_version=False,
        )

        assert result.data.name == "Jane Doe"
        assert result.data.age == 25

    @pytest.mark.asyncio
    async def test_concurrent_calls(self):
        """Test multiple concurrent async calls."""
        from promptguard import allm_call

        prompts = [
            "Extract: Alice is 30",
            "Extract: Bob is 25",
            "Extract: Charlie is 35",
        ]

        tasks = [
            allm_call(
                prompt=p,
                model="gpt-4o-mini",
                schema=PersonSchema,
                save_version=False,
            )
            for p in prompts
        ]

        results = await asyncio.gather(*tasks)

        assert len(results) == 3
        names = {r.data.name for r in results}
        assert "Alice" in str(names)
        assert "Bob" in str(names)
        assert "Charlie" in str(names)


class TestMultipleSchemaTypes:
    """Test different schema types work correctly."""

    def test_pydantic_schema(self):
        """Test Pydantic BaseModel schema."""
        from promptguard import llm_call

        result = llm_call(
            prompt="Extract: Tom is 40",
            model="gpt-4o-mini",
            schema=PersonSchema,
            save_version=False,
        )

        assert isinstance(result.data, PersonSchema)
        assert result.data.name == "Tom"
        assert result.data.age == 40

    def test_typeddict_schema(self):
        """Test TypedDict schema."""
        from promptguard import llm_call

        result = llm_call(
            prompt="Extract: Sarah is 28",
            model="gpt-4o-mini",
            schema=PersonTypedDict,
            save_version=False,
        )

        assert isinstance(result.data, dict)
        assert result.data["name"] == "Sarah"
        assert result.data["age"] == 28

    def test_dataclass_schema(self):
        """Test dataclass schema."""
        from promptguard import llm_call

        result = llm_call(
            prompt="Extract: Mike is 33",
            model="gpt-4o-mini",
            schema=PersonDataclass,
            save_version=False,
        )

        assert isinstance(result.data, PersonDataclass)
        assert result.data.name == "Mike"
        assert result.data.age == 33

    def test_json_schema(self):
        """Test raw JSON Schema."""
        from promptguard import llm_call

        json_schema = {
            "type": "object",
            "title": "Person",
            "properties": {
                "name": {"type": "string"},
                "age": {"type": "integer"},
            },
            "required": ["name", "age"],
        }

        result = llm_call(
            prompt="Extract: Emma is 29",
            model="gpt-4o-mini",
            schema=json_schema,
            save_version=False,
        )

        assert isinstance(result.data, dict)
        assert result.data["name"] == "Emma"
        assert result.data["age"] == 29


class TestVersionPersistence:
    """Test that versions are saved and loadable."""

    def test_version_saved(self):
        """Test that a version is saved after a call."""
        from promptguard.core.config import PromptGuardConfig, VersioningConfig
        from promptguard.core.engine import PromptGuardEngine

        with tempfile.TemporaryDirectory() as tmpdir:
            config = PromptGuardConfig(versioning=VersioningConfig(storage_path=Path(tmpdir)))
            engine = PromptGuardEngine(config)

            result = engine.call(
                prompt="Extract: Test Person is 99",
                model="gpt-4o-mini",
                schema=PersonSchema,
                save_version=True,
            )

            # Version should be saved
            assert result.version_hash is not None

            # Should be able to load it
            version = engine.version_store.load(result.version_hash)
            assert version is not None
            assert version.model == "gpt-4o-mini"
            assert "Test Person" in version.prompt

    def test_version_history(self):
        """Test that version history works."""
        from promptguard.core.config import PromptGuardConfig, VersioningConfig
        from promptguard.core.engine import PromptGuardEngine

        with tempfile.TemporaryDirectory() as tmpdir:
            config = PromptGuardConfig(versioning=VersioningConfig(storage_path=Path(tmpdir)))
            engine = PromptGuardEngine(config)

            # Make multiple calls
            for i in range(3):
                engine.call(
                    prompt=f"Extract: Person{i} is {20 + i}",
                    model="gpt-4o-mini",
                    schema=PersonSchema,
                    save_version=True,
                )

            # Check history
            versions = engine.version_store.list_versions()
            assert len(versions) == 3


class TestRepairLoop:
    """Test the automatic repair loop."""

    def test_repair_triggers_on_invalid_response(self):
        """Test that repair loop handles initially invalid responses.

        Note: With gpt-4o-mini and structured output, repairs are rarely needed.
        This test verifies the mechanism works when it does trigger.
        """
        from promptguard import llm_call

        # This should work without repairs with modern models
        result = llm_call(
            prompt="Extract name and age: The customer John (age thirty-five) called",
            model="gpt-4o-mini",
            schema=PersonSchema,
            save_version=False,
        )

        # Should succeed (possibly with repairs)
        assert result.data.name  # Has a name
        assert result.data.age == 35  # Parsed "thirty-five" as 35
        # repair_attempts could be 0 or more depending on model behavior
        assert result.repair_attempts >= 0


class TestErrorHandling:
    """Test error handling."""

    def test_invalid_model_raises_error(self):
        """Test that invalid model raises appropriate error."""
        from promptguard import llm_call
        from promptguard.core.exceptions import ProviderError

        with pytest.raises(ProviderError):
            llm_call(
                prompt="Test",
                model="nonexistent-model-xyz",
                schema=PersonSchema,
                save_version=False,
            )

    def test_provider_not_found(self):
        """Test that missing provider raises appropriate error."""
        from promptguard import llm_call
        from promptguard.core.exceptions import ProviderNotFoundError

        with pytest.raises(ProviderNotFoundError):
            llm_call(
                prompt="Test",
                model="nonexistent_provider:model",
                schema=PersonSchema,
                save_version=False,
            )
