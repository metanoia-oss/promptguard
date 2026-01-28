"""Tests for snapshot testing module."""

import tempfile
from datetime import datetime
from pathlib import Path

from promptguard.testing.snapshot import (
    Snapshot,
    SnapshotDiff,
    SnapshotMatcher,
    SnapshotStore,
)


class TestSnapshot:
    """Tests for Snapshot."""

    def test_to_dict_and_from_dict(self):
        snapshot = Snapshot(
            prompt_hash="abc123",
            prompt="test prompt",
            model="gpt-4o",
            schema_hash="schema123",
            output={"name": "John", "age": 30},
            raw_output='{"name": "John", "age": 30}',
            captured_at=datetime(2024, 1, 1, 12, 0, 0),
            metadata={"key": "value"},
        )

        data = snapshot.to_dict()
        restored = Snapshot.from_dict(data)

        assert restored.prompt_hash == snapshot.prompt_hash
        assert restored.output == snapshot.output
        assert restored.metadata == snapshot.metadata


class TestSnapshotStore:
    """Tests for SnapshotStore."""

    def test_save_and_load(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            store = SnapshotStore(Path(tmpdir))

            snapshot = Snapshot(
                prompt_hash="test123",
                prompt="test prompt",
                model="gpt-4o",
                schema_hash="schema123",
                output={"result": "value"},
                raw_output="raw",
                captured_at=datetime.utcnow(),
            )

            store.save(snapshot)
            loaded = store.load("test123")

            assert loaded is not None
            assert loaded.prompt_hash == "test123"
            assert loaded.output == {"result": "value"}

    def test_exists(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            store = SnapshotStore(Path(tmpdir))

            assert not store.exists("test123")

            snapshot = Snapshot(
                prompt_hash="test123",
                prompt="test",
                model="gpt-4o",
                schema_hash="s",
                output={},
                raw_output="",
                captured_at=datetime.utcnow(),
            )
            store.save(snapshot)

            assert store.exists("test123")

    def test_list_snapshots(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            store = SnapshotStore(Path(tmpdir))

            for i in range(3):
                snapshot = Snapshot(
                    prompt_hash=f"hash{i}",
                    prompt="test",
                    model="gpt-4o",
                    schema_hash="s",
                    output={},
                    raw_output="",
                    captured_at=datetime.utcnow(),
                )
                store.save(snapshot)

            snapshots = store.list_snapshots()
            assert len(snapshots) == 3


class TestSnapshotMatcher:
    """Tests for SnapshotMatcher."""

    def test_compare_identical(self):
        matcher = SnapshotMatcher()

        snapshot = Snapshot(
            prompt_hash="test",
            prompt="test",
            model="gpt-4o",
            schema_hash="s",
            output={"name": "John", "age": 30},
            raw_output="",
            captured_at=datetime.utcnow(),
        )

        diffs = matcher.compare(snapshot, {"name": "John", "age": 30})
        assert len(diffs) == 0

    def test_compare_value_changed(self):
        matcher = SnapshotMatcher()

        snapshot = Snapshot(
            prompt_hash="test",
            prompt="test",
            model="gpt-4o",
            schema_hash="s",
            output={"name": "John", "age": 30},
            raw_output="",
            captured_at=datetime.utcnow(),
        )

        diffs = matcher.compare(snapshot, {"name": "John", "age": 31})
        assert len(diffs) == 1
        assert diffs[0].diff_type == "value_changed"
        assert diffs[0].field_path == "age"
        assert diffs[0].expected == 30
        assert diffs[0].actual == 31

    def test_compare_field_added(self):
        matcher = SnapshotMatcher()

        snapshot = Snapshot(
            prompt_hash="test",
            prompt="test",
            model="gpt-4o",
            schema_hash="s",
            output={"name": "John"},
            raw_output="",
            captured_at=datetime.utcnow(),
        )

        diffs = matcher.compare(snapshot, {"name": "John", "extra": "field"})
        assert len(diffs) == 1
        assert diffs[0].diff_type == "field_added"

    def test_compare_field_removed(self):
        matcher = SnapshotMatcher()

        snapshot = Snapshot(
            prompt_hash="test",
            prompt="test",
            model="gpt-4o",
            schema_hash="s",
            output={"name": "John", "age": 30},
            raw_output="",
            captured_at=datetime.utcnow(),
        )

        diffs = matcher.compare(snapshot, {"name": "John"})
        assert len(diffs) == 1
        assert diffs[0].diff_type == "field_removed"

    def test_compare_nested(self):
        matcher = SnapshotMatcher()

        snapshot = Snapshot(
            prompt_hash="test",
            prompt="test",
            model="gpt-4o",
            schema_hash="s",
            output={"user": {"name": "John", "settings": {"theme": "dark"}}},
            raw_output="",
            captured_at=datetime.utcnow(),
        )

        diffs = matcher.compare(
            snapshot,
            {"user": {"name": "John", "settings": {"theme": "light"}}},
        )
        assert len(diffs) == 1
        assert "theme" in diffs[0].field_path

    def test_is_match(self):
        matcher = SnapshotMatcher()

        snapshot = Snapshot(
            prompt_hash="test",
            prompt="test",
            model="gpt-4o",
            schema_hash="s",
            output={"x": 1},
            raw_output="",
            captured_at=datetime.utcnow(),
        )

        assert matcher.is_match(snapshot, {"x": 1})
        assert not matcher.is_match(snapshot, {"x": 2})

    def test_ignore_fields(self):
        matcher = SnapshotMatcher(ignore_fields=["timestamp"])

        snapshot = Snapshot(
            prompt_hash="test",
            prompt="test",
            model="gpt-4o",
            schema_hash="s",
            output={"value": 1, "timestamp": "2024-01-01"},
            raw_output="",
            captured_at=datetime.utcnow(),
        )

        diffs = matcher.compare(
            snapshot,
            {"value": 1, "timestamp": "2024-12-31"},
        )
        assert len(diffs) == 0

    def test_format_diff_report(self):
        matcher = SnapshotMatcher()
        diffs = [
            SnapshotDiff("name", "John", "Jane", "value_changed"),
            SnapshotDiff("extra", None, "value", "field_added"),
        ]

        report = matcher.format_diff_report(diffs)
        assert "2 difference(s)" in report
        assert "name" in report
        assert "extra" in report
