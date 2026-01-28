"""Prompt versioning and hashing utilities."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Literal


@dataclass
class PromptVersion:
    """Represents a versioned prompt execution.

    Stores all relevant information about a prompt execution for
    reproducibility and debugging.

    Attributes:
        hash: Unique hash identifying this version.
        prompt: The prompt text.
        model: Model used for execution.
        schema_hash: Hash of the output schema.
        temperature: Temperature setting used.
        created_at: When this version was created.
        metadata: Additional metadata.
    """

    hash: str
    prompt: str
    model: str
    schema_hash: str | None
    temperature: float
    created_at: datetime
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "hash": self.hash,
            "prompt": self.prompt,
            "model": self.model,
            "schema_hash": self.schema_hash,
            "temperature": self.temperature,
            "created_at": self.created_at.isoformat(),
            "metadata": self.metadata,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> PromptVersion:
        """Create from dictionary."""
        return cls(
            hash=data["hash"],
            prompt=data["prompt"],
            model=data["model"],
            schema_hash=data.get("schema_hash"),
            temperature=data.get("temperature", 0.7),
            created_at=datetime.fromisoformat(data["created_at"]),
            metadata=data.get("metadata", {}),
        )


class PromptHasher:
    """Generates stable hashes for prompts and configurations.

    The hasher creates deterministic hashes that can be used to
    identify unique prompt configurations.

    Example:
        hasher = PromptHasher()
        version_hash = hasher.hash_prompt(
            prompt="Extract data from...",
            model="gpt-4o",
            schema={"type": "object", ...}
        )
    """

    def __init__(
        self,
        algorithm: Literal["sha256", "sha1", "md5"] = "sha256",
        include_model: bool = True,
        include_temperature: bool = False,
        hash_length: int = 12,
    ) -> None:
        """Initialize the hasher.

        Args:
            algorithm: Hash algorithm to use.
            include_model: Whether to include model in hash calculation.
            include_temperature: Whether to include temperature in hash.
            hash_length: Length of the returned hash string.
        """
        self.algorithm = algorithm
        self.include_model = include_model
        self.include_temperature = include_temperature
        self.hash_length = hash_length

    def _get_hasher(self):
        """Get a new hasher instance."""
        return hashlib.new(self.algorithm)

    def hash_prompt(
        self,
        prompt: str,
        model: str | None = None,
        temperature: float | None = None,
        schema: dict[str, Any] | None = None,
    ) -> str:
        """Generate a stable hash for a prompt configuration.

        Args:
            prompt: The prompt text.
            model: Model identifier (included if include_model is True).
            temperature: Temperature value (included if include_temperature is True).
            schema: JSON schema dict (always included if provided).

        Returns:
            Truncated hash string.

        Example:
            hash = hasher.hash_prompt("Hello", model="gpt-4o")
            # Returns something like "a1b2c3d4e5f6"
        """
        hasher = self._get_hasher()

        # Always hash the prompt text
        hasher.update(prompt.encode("utf-8"))

        # Optionally include model
        if self.include_model and model:
            hasher.update(f"|model:{model}".encode())

        # Optionally include temperature
        if self.include_temperature and temperature is not None:
            hasher.update(f"|temp:{temperature}".encode())

        # Include schema if provided
        if schema:
            # Sort keys for deterministic hashing
            schema_str = json.dumps(schema, sort_keys=True)
            hasher.update(f"|schema:{schema_str}".encode())

        return hasher.hexdigest()[: self.hash_length]

    def hash_schema(self, schema: dict[str, Any]) -> str:
        """Generate a hash for a schema.

        Args:
            schema: JSON schema dictionary.

        Returns:
            Truncated hash string.
        """
        hasher = self._get_hasher()
        schema_str = json.dumps(schema, sort_keys=True)
        hasher.update(schema_str.encode("utf-8"))
        return hasher.hexdigest()[: self.hash_length]

    def hash_content(self, content: str) -> str:
        """Generate a hash for arbitrary content.

        Args:
            content: String content to hash.

        Returns:
            Truncated hash string.
        """
        hasher = self._get_hasher()
        hasher.update(content.encode("utf-8"))
        return hasher.hexdigest()[: self.hash_length]


class VersionStore:
    """Stores and retrieves prompt versions.

    Uses file-based JSON storage for simplicity and version control
    compatibility.

    Example:
        store = VersionStore(Path(".promptguard/versions"))
        store.save(version)
        loaded = store.load("a1b2c3d4e5f6")
    """

    def __init__(self, storage_path: Path) -> None:
        """Initialize the version store.

        Args:
            storage_path: Directory path for storing version files.
        """
        self.storage_path = Path(storage_path)
        self.storage_path.mkdir(parents=True, exist_ok=True)
        self._index_path = self.storage_path / "index.json"
        self._index: dict[str, dict[str, Any]] = self._load_index()

    def _load_index(self) -> dict[str, dict[str, Any]]:
        """Load the version index."""
        if self._index_path.exists():
            try:
                with open(self._index_path) as f:
                    return json.load(f)
            except (OSError, json.JSONDecodeError):
                return {}
        return {}

    def _save_index(self) -> None:
        """Save the version index."""
        with open(self._index_path, "w") as f:
            json.dump(self._index, f, indent=2)

    def save(self, version: PromptVersion) -> str:
        """Save a prompt version.

        Args:
            version: PromptVersion to save.

        Returns:
            The version hash.
        """
        version_file = self.storage_path / f"{version.hash}.json"

        with open(version_file, "w") as f:
            json.dump(version.to_dict(), f, indent=2)

        # Update index with summary info
        self._index[version.hash] = {
            "model": version.model,
            "created_at": version.created_at.isoformat(),
            "prompt_preview": version.prompt[:100],
        }
        self._save_index()

        return version.hash

    def load(self, hash: str) -> PromptVersion | None:
        """Load a prompt version by hash.

        Args:
            hash: Version hash to load.

        Returns:
            PromptVersion if found, None otherwise.
        """
        version_file = self.storage_path / f"{hash}.json"

        if not version_file.exists():
            return None

        try:
            with open(version_file) as f:
                data = json.load(f)
            return PromptVersion.from_dict(data)
        except (OSError, json.JSONDecodeError, KeyError):
            return None

    def list_versions(self, limit: int = 50) -> list[dict[str, Any]]:
        """List recent versions.

        Args:
            limit: Maximum number of versions to return.

        Returns:
            List of version summaries sorted by creation date (newest first).
        """
        versions = sorted(
            self._index.items(),
            key=lambda x: x[1].get("created_at", ""),
            reverse=True,
        )
        return [{"hash": h, **v} for h, v in versions[:limit]]

    def exists(self, hash: str) -> bool:
        """Check if a version exists.

        Args:
            hash: Version hash to check.

        Returns:
            True if the version exists.
        """
        return hash in self._index

    def delete(self, hash: str) -> bool:
        """Delete a version.

        Args:
            hash: Version hash to delete.

        Returns:
            True if the version was deleted.
        """
        version_file = self.storage_path / f"{hash}.json"

        if version_file.exists():
            version_file.unlink()

        if hash in self._index:
            del self._index[hash]
            self._save_index()
            return True

        return False

    def clear(self) -> int:
        """Clear all versions.

        Returns:
            Number of versions deleted.
        """
        count = 0
        for hash in list(self._index.keys()):
            if self.delete(hash):
                count += 1
        return count

    def get_stats(self) -> dict[str, Any]:
        """Get statistics about stored versions.

        Returns:
            Dictionary with version statistics.
        """
        if not self._index:
            return {
                "total_versions": 0,
                "models": {},
                "oldest": None,
                "newest": None,
            }

        models: dict[str, int] = {}
        dates: list[str] = []

        for info in self._index.values():
            model = info.get("model", "unknown")
            models[model] = models.get(model, 0) + 1
            if "created_at" in info:
                dates.append(info["created_at"])

        sorted_dates = sorted(dates) if dates else []

        return {
            "total_versions": len(self._index),
            "models": models,
            "oldest": sorted_dates[0] if sorted_dates else None,
            "newest": sorted_dates[-1] if sorted_dates else None,
        }
