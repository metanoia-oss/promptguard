"""End-to-end tests for PromptGuard.

These tests verify complete workflows work correctly.
"""

import json
import os
import subprocess
import tempfile
from pathlib import Path

import pytest
import yaml


class TestPackageImport:
    """Test that the package imports correctly."""

    def test_main_imports(self):
        """Test main module imports."""
        from promptguard import (
            allm_call,
            llm_call,
        )

        assert callable(llm_call)
        assert callable(allm_call)

    def test_provider_imports(self):
        """Test provider imports."""
        from promptguard.providers import (
            ProviderRegistry,
        )

        assert ProviderRegistry is not None

    def test_schema_imports(self):
        """Test schema adapter imports."""
        from promptguard.schemas import (
            create_adapter,
        )

        assert callable(create_adapter)

    def test_testing_imports(self):
        """Test testing module imports."""
        from promptguard.testing import (
            SnapshotStore,
        )

        assert SnapshotStore is not None


class TestCLIInit:
    """Test CLI init command."""

    def test_init_creates_structure(self):
        """Test that init creates the correct directory structure."""
        with tempfile.TemporaryDirectory() as tmpdir:
            result = subprocess.run(
                ["promptguard", "init"],
                cwd=tmpdir,
                capture_output=True,
                text=True,
            )

            assert result.returncode == 0
            assert "successfully" in result.stdout.lower()

            # Check directories created
            assert (Path(tmpdir) / ".promptguard").is_dir()
            assert (Path(tmpdir) / ".promptguard" / "versions").is_dir()
            assert (Path(tmpdir) / ".promptguard" / "snapshots").is_dir()
            assert (Path(tmpdir) / "prompts").is_dir()

            # Check files created
            assert (Path(tmpdir) / ".promptguard" / "config.yaml").is_file()
            assert (Path(tmpdir) / "prompts" / "example.yaml").is_file()

    def test_init_config_valid(self):
        """Test that generated config is valid YAML."""
        with tempfile.TemporaryDirectory() as tmpdir:
            subprocess.run(
                ["promptguard", "init"],
                cwd=tmpdir,
                capture_output=True,
            )

            config_path = Path(tmpdir) / ".promptguard" / "config.yaml"
            with open(config_path) as f:
                config = yaml.safe_load(f)

            assert "default_provider" in config
            assert "retry" in config
            assert "versioning" in config


class TestCLIHelp:
    """Test CLI help commands."""

    def test_main_help(self):
        """Test main help command."""
        result = subprocess.run(
            ["promptguard", "--help"],
            capture_output=True,
            text=True,
        )

        assert result.returncode == 0
        assert "PromptGuard" in result.stdout
        assert "init" in result.stdout
        assert "run" in result.stdout
        assert "test" in result.stdout

    def test_init_help(self):
        """Test init help command."""
        result = subprocess.run(
            ["promptguard", "init", "--help"],
            capture_output=True,
            text=True,
        )

        assert result.returncode == 0
        assert "Initialize" in result.stdout

    def test_run_help(self):
        """Test run help command."""
        result = subprocess.run(
            ["promptguard", "run", "--help"],
            capture_output=True,
            text=True,
        )

        assert result.returncode == 0
        assert "YAML" in result.stdout or "prompt" in result.stdout.lower()


class TestCLIHistory:
    """Test CLI history command."""

    def test_history_empty(self):
        """Test history with no versions."""
        with tempfile.TemporaryDirectory() as tmpdir:
            # Init first
            subprocess.run(["promptguard", "init"], cwd=tmpdir, capture_output=True)

            result = subprocess.run(
                ["promptguard", "history"],
                cwd=tmpdir,
                capture_output=True,
                text=True,
            )

            assert result.returncode == 0
            assert "No versions" in result.stdout or "History" in result.stdout


class TestCLIStats:
    """Test CLI stats command."""

    def test_stats_empty(self):
        """Test stats with no data."""
        with tempfile.TemporaryDirectory() as tmpdir:
            # Init first
            subprocess.run(["promptguard", "init"], cwd=tmpdir, capture_output=True)

            result = subprocess.run(
                ["promptguard", "stats"],
                cwd=tmpdir,
                capture_output=True,
                text=True,
            )

            assert result.returncode == 0
            assert "Statistics" in result.stdout or "0" in result.stdout


@pytest.mark.skipif(not os.getenv("OPENAI_API_KEY"), reason="OPENAI_API_KEY not set")
class TestCLIRunWithAPI:
    """Test CLI run command with real API calls."""

    def test_run_yaml_prompt(self):
        """Test running a YAML prompt file."""
        with tempfile.TemporaryDirectory() as tmpdir:
            # Create a prompt file
            prompt_file = Path(tmpdir) / "test_prompt.yaml"
            prompt_data = {
                "name": "test_extraction",
                "model": "gpt-4o-mini",
                "schema": {
                    "type": "object",
                    "properties": {
                        "name": {"type": "string"},
                        "age": {"type": "integer"},
                    },
                    "required": ["name", "age"],
                },
                "prompt_template": "Extract person info: {text}",
            }

            with open(prompt_file, "w") as f:
                yaml.dump(prompt_data, f)

            # Run the prompt
            result = subprocess.run(
                [
                    "promptguard",
                    "run",
                    str(prompt_file),
                    "-i",
                    '{"text": "Alice is 30 years old"}',
                ],
                cwd=tmpdir,
                capture_output=True,
                text=True,
            )

            assert result.returncode == 0
            # Output should be valid JSON
            output = result.stdout.strip()
            # Find the JSON in the output (might have Rich formatting)
            if "{" in output:
                json_start = output.index("{")
                json_end = output.rindex("}") + 1
                json_str = output[json_start:json_end]
                data = json.loads(json_str)
                assert "name" in data
                assert "age" in data


class TestFullWorkflow:
    """Test complete end-to-end workflows."""

    @pytest.mark.skipif(not os.getenv("OPENAI_API_KEY"), reason="OPENAI_API_KEY not set")
    def test_python_api_workflow(self):
        """Test complete Python API workflow."""
        from pydantic import BaseModel

        from promptguard.core.config import PromptGuardConfig, VersioningConfig
        from promptguard.core.engine import PromptGuardEngine

        class Task(BaseModel):
            title: str
            priority: int
            completed: bool

        with tempfile.TemporaryDirectory() as tmpdir:
            # Setup engine with temp storage
            config = PromptGuardConfig(versioning=VersioningConfig(storage_path=Path(tmpdir)))
            engine = PromptGuardEngine(config)

            # Make a call
            result = engine.call(
                prompt="Create a task: Review PR #123, high priority, not done yet",
                model="gpt-4o-mini",
                schema=Task,
                save_version=True,
            )

            # Verify result
            assert result.data.title
            assert result.data.priority > 0
            assert result.data.completed is False

            # Verify version saved
            assert result.version_hash is not None
            version = engine.version_store.load(result.version_hash)
            assert version is not None

            # Verify stats
            stats = engine.version_store.get_stats()
            assert stats["total_versions"] == 1
