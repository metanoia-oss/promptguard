"""Tests for hashing and versioning module."""

import tempfile
from datetime import datetime
from pathlib import Path

import pytest

from promptguard.core.hashing import PromptHasher, PromptVersion, VersionStore


class TestPromptHasher:
    """Tests for PromptHasher."""

    def test_hash_prompt_deterministic(self):
        hasher = PromptHasher()
        hash1 = hasher.hash_prompt("test prompt")
        hash2 = hasher.hash_prompt("test prompt")
        assert hash1 == hash2

    def test_hash_prompt_different_for_different_prompts(self):
        hasher = PromptHasher()
        hash1 = hasher.hash_prompt("prompt one")
        hash2 = hasher.hash_prompt("prompt two")
        assert hash1 != hash2

    def test_hash_includes_model_when_configured(self):
        hasher = PromptHasher(include_model=True)
        hash1 = hasher.hash_prompt("test", model="gpt-4o")
        hash2 = hasher.hash_prompt("test", model="gpt-3.5")
        assert hash1 != hash2

    def test_hash_excludes_model_when_configured(self):
        hasher = PromptHasher(include_model=False)
        hash1 = hasher.hash_prompt("test", model="gpt-4o")
        hash2 = hasher.hash_prompt("test", model="gpt-3.5")
        assert hash1 == hash2

    def test_hash_includes_temperature_when_configured(self):
        hasher = PromptHasher(include_temperature=True)
        hash1 = hasher.hash_prompt("test", temperature=0.5)
        hash2 = hasher.hash_prompt("test", temperature=0.9)
        assert hash1 != hash2

    def test_hash_schema(self):
        hasher = PromptHasher()
        schema = {"type": "object", "properties": {"x": {"type": "integer"}}}
        hash1 = hasher.hash_schema(schema)
        hash2 = hasher.hash_schema(schema)
        assert hash1 == hash2
        assert len(hash1) == 12  # Default hash length

    def test_custom_hash_length(self):
        hasher = PromptHasher(hash_length=8)
        hash1 = hasher.hash_prompt("test")
        assert len(hash1) == 8


class TestPromptVersion:
    """Tests for PromptVersion."""

    def test_to_dict_and_from_dict(self):
        version = PromptVersion(
            hash="abc123",
            prompt="test prompt",
            model="gpt-4o",
            schema_hash="def456",
            temperature=0.7,
            created_at=datetime(2024, 1, 1, 12, 0, 0),
            metadata={"key": "value"},
        )

        data = version.to_dict()
        restored = PromptVersion.from_dict(data)

        assert restored.hash == version.hash
        assert restored.prompt == version.prompt
        assert restored.model == version.model
        assert restored.temperature == version.temperature
        assert restored.metadata == version.metadata


class TestVersionStore:
    """Tests for VersionStore."""

    def test_save_and_load(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            store = VersionStore(Path(tmpdir))

            version = PromptVersion(
                hash="test123",
                prompt="test prompt",
                model="gpt-4o",
                schema_hash="schema123",
                temperature=0.7,
                created_at=datetime.utcnow(),
            )

            store.save(version)
            loaded = store.load("test123")

            assert loaded is not None
            assert loaded.hash == version.hash
            assert loaded.prompt == version.prompt

    def test_load_nonexistent(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            store = VersionStore(Path(tmpdir))
            assert store.load("nonexistent") is None

    def test_exists(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            store = VersionStore(Path(tmpdir))

            version = PromptVersion(
                hash="test123",
                prompt="test",
                model="gpt-4o",
                schema_hash=None,
                temperature=0.7,
                created_at=datetime.utcnow(),
            )

            assert not store.exists("test123")
            store.save(version)
            assert store.exists("test123")

    def test_delete(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            store = VersionStore(Path(tmpdir))

            version = PromptVersion(
                hash="test123",
                prompt="test",
                model="gpt-4o",
                schema_hash=None,
                temperature=0.7,
                created_at=datetime.utcnow(),
            )

            store.save(version)
            assert store.exists("test123")

            store.delete("test123")
            assert not store.exists("test123")

    def test_list_versions(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            store = VersionStore(Path(tmpdir))

            for i in range(5):
                version = PromptVersion(
                    hash=f"hash{i}",
                    prompt=f"prompt {i}",
                    model="gpt-4o",
                    schema_hash=None,
                    temperature=0.7,
                    created_at=datetime.utcnow(),
                )
                store.save(version)

            versions = store.list_versions(limit=3)
            assert len(versions) == 3

    def test_get_stats(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            store = VersionStore(Path(tmpdir))

            models = ["gpt-4o", "gpt-4o", "claude-3"]
            for i, model in enumerate(models):
                version = PromptVersion(
                    hash=f"hash{i}",  # Use unique hashes
                    prompt="test",
                    model=model,
                    schema_hash=None,
                    temperature=0.7,
                    created_at=datetime.utcnow(),
                )
                store.save(version)

            stats = store.get_stats()
            assert stats["total_versions"] == 3
            assert stats["models"]["gpt-4o"] == 2
            assert stats["models"]["claude-3"] == 1
