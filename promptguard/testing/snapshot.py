"""Snapshot testing for prompt regression detection."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, Optional


@dataclass
class Snapshot:
    """A captured snapshot of LLM output.

    Snapshots store the expected output for a specific prompt configuration,
    allowing detection of behavioral changes over time.

    Attributes:
        prompt_hash: Hash identifying the prompt configuration.
        prompt: The original prompt text.
        model: Model used for generation.
        schema_hash: Hash of the output schema.
        output: The validated output data.
        raw_output: The raw LLM response.
        captured_at: When this snapshot was captured.
        metadata: Additional metadata.
    """
    prompt_hash: str
    prompt: str
    model: str
    schema_hash: str
    output: Any
    raw_output: str
    captured_at: datetime
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for serialization."""
        output_data = self.output
        if hasattr(self.output, "model_dump"):
            output_data = self.output.model_dump()
        elif hasattr(self.output, "__dict__"):
            output_data = self.output.__dict__

        return {
            "prompt_hash": self.prompt_hash,
            "prompt": self.prompt,
            "model": self.model,
            "schema_hash": self.schema_hash,
            "output": output_data,
            "raw_output": self.raw_output,
            "captured_at": self.captured_at.isoformat(),
            "metadata": self.metadata,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> Snapshot:
        """Create from dictionary."""
        return cls(
            prompt_hash=data["prompt_hash"],
            prompt=data["prompt"],
            model=data["model"],
            schema_hash=data["schema_hash"],
            output=data["output"],
            raw_output=data["raw_output"],
            captured_at=datetime.fromisoformat(data["captured_at"]),
            metadata=data.get("metadata", {}),
        )


@dataclass
class SnapshotDiff:
    """Difference between expected and actual output.

    Attributes:
        field_path: Path to the differing field (e.g., "user.name").
        expected: Expected value.
        actual: Actual value.
        diff_type: Type of difference.
    """
    field_path: str
    expected: Any
    actual: Any
    diff_type: str  # "value_changed", "field_added", "field_removed", "type_changed"


class SnapshotStore:
    """Manages snapshot storage and retrieval.

    Uses file-based JSON storage for easy version control integration.

    Example:
        store = SnapshotStore(Path(".promptguard/snapshots"))
        store.save(snapshot)
        loaded = store.load("a1b2c3d4e5f6")
    """

    def __init__(self, base_path: Path) -> None:
        """Initialize the snapshot store.

        Args:
            base_path: Directory for storing snapshots.
        """
        self.base_path = Path(base_path)
        self.base_path.mkdir(parents=True, exist_ok=True)

    def _snapshot_path(self, prompt_hash: str) -> Path:
        """Get the file path for a snapshot."""
        return self.base_path / f"{prompt_hash}.json"

    def save(self, snapshot: Snapshot) -> None:
        """Save a snapshot.

        Args:
            snapshot: Snapshot to save.
        """
        path = self._snapshot_path(snapshot.prompt_hash)
        with open(path, "w") as f:
            json.dump(snapshot.to_dict(), f, indent=2, default=str)

    def load(self, prompt_hash: str) -> Optional[Snapshot]:
        """Load a snapshot by hash.

        Args:
            prompt_hash: Hash of the snapshot to load.

        Returns:
            Snapshot if found, None otherwise.
        """
        path = self._snapshot_path(prompt_hash)
        if not path.exists():
            return None

        try:
            with open(path) as f:
                data = json.load(f)
            return Snapshot.from_dict(data)
        except (json.JSONDecodeError, IOError, KeyError):
            return None

    def exists(self, prompt_hash: str) -> bool:
        """Check if a snapshot exists.

        Args:
            prompt_hash: Hash to check.

        Returns:
            True if snapshot exists.
        """
        return self._snapshot_path(prompt_hash).exists()

    def list_snapshots(self) -> list[str]:
        """List all snapshot hashes.

        Returns:
            List of snapshot hashes.
        """
        return [p.stem for p in self.base_path.glob("*.json")]

    def delete(self, prompt_hash: str) -> bool:
        """Delete a snapshot.

        Args:
            prompt_hash: Hash of snapshot to delete.

        Returns:
            True if deleted.
        """
        path = self._snapshot_path(prompt_hash)
        if path.exists():
            path.unlink()
            return True
        return False

    def clear(self) -> int:
        """Clear all snapshots.

        Returns:
            Number of snapshots deleted.
        """
        count = 0
        for hash in self.list_snapshots():
            if self.delete(hash):
                count += 1
        return count


class SnapshotMatcher:
    """Compares snapshots and generates diffs.

    Can optionally use semantic comparison for text fields.

    Example:
        matcher = SnapshotMatcher()
        diffs = matcher.compare(expected_snapshot, actual_output)
        if diffs:
            print("Output changed!")
    """

    def __init__(
        self,
        semantic_comparator: Optional[Callable[[Any, Any], float]] = None,
        semantic_threshold: float = 0.85,
        ignore_fields: Optional[list[str]] = None,
    ) -> None:
        """Initialize the matcher.

        Args:
            semantic_comparator: Optional function for semantic similarity (0-1).
            semantic_threshold: Minimum similarity score to consider equivalent.
            ignore_fields: List of field paths to ignore in comparison.
        """
        self.semantic_comparator = semantic_comparator
        self.semantic_threshold = semantic_threshold
        self.ignore_fields = set(ignore_fields or [])

    def compare(self, expected: Snapshot, actual: Any) -> list[SnapshotDiff]:
        """Compare expected snapshot with actual output.

        Args:
            expected: Expected snapshot.
            actual: Actual output to compare.

        Returns:
            List of differences found.
        """
        diffs: list[SnapshotDiff] = []
        expected_data = expected.output

        # Convert actual to dict if needed
        if hasattr(actual, "model_dump"):
            actual_data = actual.model_dump()
        elif hasattr(actual, "__dict__"):
            actual_data = actual.__dict__
        else:
            actual_data = actual

        self._compare_recursive(expected_data, actual_data, "", diffs)
        return diffs

    def _compare_recursive(
        self,
        expected: Any,
        actual: Any,
        path: str,
        diffs: list[SnapshotDiff],
    ) -> None:
        """Recursively compare two values."""
        # Check if this path should be ignored
        if path in self.ignore_fields:
            return

        # Handle dict comparison
        if isinstance(expected, dict) and isinstance(actual, dict):
            all_keys = set(expected.keys()) | set(actual.keys())
            for key in all_keys:
                new_path = f"{path}.{key}" if path else key

                if new_path in self.ignore_fields:
                    continue

                if key not in actual:
                    diffs.append(SnapshotDiff(
                        field_path=new_path,
                        expected=expected[key],
                        actual=None,
                        diff_type="field_removed",
                    ))
                elif key not in expected:
                    diffs.append(SnapshotDiff(
                        field_path=new_path,
                        expected=None,
                        actual=actual[key],
                        diff_type="field_added",
                    ))
                else:
                    self._compare_recursive(
                        expected[key], actual[key], new_path, diffs
                    )
            return

        # Handle list comparison
        if isinstance(expected, list) and isinstance(actual, list):
            max_len = max(len(expected), len(actual))
            for i in range(max_len):
                new_path = f"{path}[{i}]"
                if i >= len(actual):
                    diffs.append(SnapshotDiff(
                        field_path=new_path,
                        expected=expected[i],
                        actual=None,
                        diff_type="field_removed",
                    ))
                elif i >= len(expected):
                    diffs.append(SnapshotDiff(
                        field_path=new_path,
                        expected=None,
                        actual=actual[i],
                        diff_type="field_added",
                    ))
                else:
                    self._compare_recursive(
                        expected[i], actual[i], new_path, diffs
                    )
            return

        # Handle type mismatch
        if type(expected) != type(actual):
            diffs.append(SnapshotDiff(
                field_path=path,
                expected=expected,
                actual=actual,
                diff_type="type_changed",
            ))
            return

        # Handle value comparison
        if expected != actual:
            # Try semantic comparison for strings
            if (
                isinstance(expected, str) and
                isinstance(actual, str) and
                self.semantic_comparator
            ):
                similarity = self.semantic_comparator(expected, actual)
                if similarity >= self.semantic_threshold:
                    return  # Semantically equivalent

            diffs.append(SnapshotDiff(
                field_path=path or "root",
                expected=expected,
                actual=actual,
                diff_type="value_changed",
            ))

    def is_match(self, expected: Snapshot, actual: Any) -> bool:
        """Check if actual output matches expected snapshot.

        Args:
            expected: Expected snapshot.
            actual: Actual output.

        Returns:
            True if they match (no differences).
        """
        return len(self.compare(expected, actual)) == 0

    def format_diff_report(self, diffs: list[SnapshotDiff]) -> str:
        """Format differences as a human-readable report.

        Args:
            diffs: List of differences.

        Returns:
            Formatted report string.
        """
        if not diffs:
            return "No differences found."

        lines = [f"Found {len(diffs)} difference(s):"]

        for diff in diffs:
            if diff.diff_type == "value_changed":
                lines.append(
                    f"  {diff.field_path}: "
                    f"expected {diff.expected!r}, got {diff.actual!r}"
                )
            elif diff.diff_type == "field_added":
                lines.append(f"  {diff.field_path}: unexpected field with value {diff.actual!r}")
            elif diff.diff_type == "field_removed":
                lines.append(f"  {diff.field_path}: missing field (expected {diff.expected!r})")
            elif diff.diff_type == "type_changed":
                lines.append(
                    f"  {diff.field_path}: type changed from "
                    f"{type(diff.expected).__name__} to {type(diff.actual).__name__}"
                )

        return "\n".join(lines)
