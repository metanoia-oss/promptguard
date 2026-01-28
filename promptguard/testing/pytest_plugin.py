"""Pytest integration for PromptGuard testing."""

from __future__ import annotations

from datetime import datetime
from pathlib import Path
from typing import Any, Optional

from promptguard.core.hashing import PromptHasher
from promptguard.testing.semantic_diff import SemanticComparator
from promptguard.testing.snapshot import Snapshot, SnapshotMatcher, SnapshotStore


class PromptGuardFixture:
    """Pytest fixture for PromptGuard snapshot testing.

    Provides utilities for capturing and comparing LLM output snapshots
    in pytest tests.

    Example:
        def test_extraction(promptguard):
            result = llm_call(...)
            promptguard.assert_output_matches(
                prompt="Hi how are you",
                model="gpt-4o",
                actual_output=result.data,
                schema=MySchema
            )
    """

    def __init__(
        self,
        snapshot_dir: Path,
        update_snapshots: bool = False,
        semantic_comparison: bool = True,
        semantic_threshold: float = 0.85,
    ) -> None:
        """Initialize the fixture.

        Args:
            snapshot_dir: Directory for storing snapshots.
            update_snapshots: If True, update existing snapshots instead of comparing.
            semantic_comparison: Enable semantic comparison for text fields.
            semantic_threshold: Minimum similarity for semantic equivalence.
        """
        self.store = SnapshotStore(snapshot_dir)
        self.hasher = PromptHasher()
        self.update_snapshots = update_snapshots
        self.semantic_comparison = semantic_comparison
        self.semantic_threshold = semantic_threshold
        self._comparator: Optional[SemanticComparator] = None

    @property
    def comparator(self) -> Optional[SemanticComparator]:
        """Get the semantic comparator (lazy loaded)."""
        if self.semantic_comparison and self._comparator is None:
            try:
                self._comparator = SemanticComparator()
            except ImportError:
                # sentence-transformers not installed
                pass
        return self._comparator

    def assert_output_matches(
        self,
        prompt: str,
        model: str,
        actual_output: Any,
        schema: Any,
        raw_output: str = "",
        metadata: Optional[dict[str, Any]] = None,
    ) -> None:
        """Assert that output matches stored snapshot.

        If no snapshot exists, creates one. If update_snapshots is True,
        always updates the snapshot.

        Args:
            prompt: The prompt used.
            model: The model used.
            actual_output: The actual output to compare.
            schema: The schema used for validation.
            raw_output: Optional raw LLM response.
            metadata: Optional metadata.

        Raises:
            AssertionError: If output doesn't match snapshot.
            pytest.skip: If creating a new snapshot.
        """
        import pytest

        # Generate hash
        prompt_hash = self.hasher.hash_prompt(prompt, model)

        # Get schema hash
        if hasattr(schema, "model_json_schema"):
            schema_dict = schema.model_json_schema()
        elif isinstance(schema, dict):
            schema_dict = schema
        else:
            schema_dict = {}
        schema_hash = self.hasher.hash_schema(schema_dict)

        # Load existing snapshot
        stored = self.store.load(prompt_hash)

        if stored is None or self.update_snapshots:
            # Create new snapshot
            snapshot = Snapshot(
                prompt_hash=prompt_hash,
                prompt=prompt,
                model=model,
                schema_hash=schema_hash,
                output=actual_output,
                raw_output=raw_output,
                captured_at=datetime.utcnow(),
                metadata=metadata or {},
            )
            self.store.save(snapshot)

            if stored is None:
                pytest.skip(f"Snapshot created for {prompt_hash}")
            return

        # Compare with stored snapshot
        comparator_fn = None
        if self.comparator:
            comparator_fn = self.comparator.similarity

        matcher = SnapshotMatcher(
            semantic_comparator=comparator_fn,
            semantic_threshold=self.semantic_threshold,
        )

        diffs = matcher.compare(stored, actual_output)

        if diffs:
            diff_report = matcher.format_diff_report(diffs)
            pytest.fail(
                f"Output mismatch for prompt '{prompt[:50]}...':\n{diff_report}\n\n"
                f"Run with --update-snapshots to update the baseline."
            )

    def get_snapshot(self, prompt: str, model: str) -> Optional[Snapshot]:
        """Get a stored snapshot.

        Args:
            prompt: The prompt.
            model: The model.

        Returns:
            Snapshot if found.
        """
        prompt_hash = self.hasher.hash_prompt(prompt, model)
        return self.store.load(prompt_hash)

    def delete_snapshot(self, prompt: str, model: str) -> bool:
        """Delete a stored snapshot.

        Args:
            prompt: The prompt.
            model: The model.

        Returns:
            True if deleted.
        """
        prompt_hash = self.hasher.hash_prompt(prompt, model)
        return self.store.delete(prompt_hash)


def pytest_addoption(parser) -> None:
    """Add pytest command line options."""
    group = parser.getgroup("promptguard")
    group.addoption(
        "--update-snapshots",
        action="store_true",
        default=False,
        help="Update stored snapshots with new values",
    )
    group.addoption(
        "--no-semantic",
        action="store_true",
        default=False,
        help="Disable semantic comparison",
    )
    group.addoption(
        "--semantic-threshold",
        type=float,
        default=0.85,
        help="Minimum similarity for semantic equivalence (0-1)",
    )
    group.addoption(
        "--snapshot-dir",
        type=str,
        default=None,
        help="Directory for storing snapshots",
    )


def pytest_configure(config) -> None:
    """Register the promptguard marker."""
    config.addinivalue_line(
        "markers",
        "promptguard: mark test as a PromptGuard snapshot test",
    )


# Fixture factory - to be used with pytest
def _create_fixture(request):
    """Create the promptguard fixture."""
    config = request.config

    # Determine snapshot directory
    snapshot_dir_opt = config.getoption("--snapshot-dir")
    if snapshot_dir_opt:
        snapshot_dir = Path(snapshot_dir_opt)
    else:
        snapshot_dir = Path(config.rootdir) / ".promptguard" / "snapshots"

    return PromptGuardFixture(
        snapshot_dir=snapshot_dir,
        update_snapshots=config.getoption("--update-snapshots", default=False),
        semantic_comparison=not config.getoption("--no-semantic", default=False),
        semantic_threshold=config.getoption("--semantic-threshold", default=0.85),
    )


# Export the fixture for pytest to discover
try:
    import pytest

    @pytest.fixture
    def promptguard(request):
        """Pytest fixture providing PromptGuard testing utilities.

        Example:
            def test_extraction(promptguard):
                result = llm_call(
                    prompt="Extract: John is 30",
                    model="gpt-4o",
                    schema=Person,
                )
                promptguard.assert_output_matches(
                    prompt="Extract: John is 30",
                    model="gpt-4o",
                    actual_output=result.data,
                    schema=Person
                )
        """
        return _create_fixture(request)

except ImportError:
    # pytest not installed
    pass
