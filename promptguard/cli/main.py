"""CLI commands for PromptGuard."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import click
import yaml
from rich.console import Console
from rich.syntax import Syntax
from rich.table import Table

console = Console()


@click.group()
@click.version_option(package_name="promptguard")
def cli() -> None:
    """PromptGuard - Reliable LLM outputs with schema validation.

    A CLI tool for working with schema-guaranteed LLM outputs,
    prompt versioning, and regression testing.
    """
    pass


@cli.command()
@click.option("--path", "-p", default=".", help="Project path")
def init(path: str) -> None:
    """Initialize PromptGuard in a project.

    Creates the necessary directory structure and configuration files.
    """
    project_path = Path(path)

    # Create directory structure
    dirs = [
        project_path / ".promptguard",
        project_path / ".promptguard" / "versions",
        project_path / ".promptguard" / "snapshots",
        project_path / "prompts",
    ]

    for dir_path in dirs:
        dir_path.mkdir(parents=True, exist_ok=True)
        console.print(f"  Created: {dir_path}")

    # Create default config
    config = {
        "default_provider": "openai",
        "retry": {
            "max_retries": 3,
            "backoff_strategy": "exponential",
            "base_delay": 1.0,
        },
        "versioning": {
            "storage_path": ".promptguard/versions",
            "hash_algorithm": "sha256",
        },
        "logging": {
            "level": "INFO",
            "format": "text",
        },
    }

    config_path = project_path / ".promptguard" / "config.yaml"
    if not config_path.exists():
        with open(config_path, "w") as f:
            yaml.dump(config, f, default_flow_style=False, sort_keys=False)
        console.print(f"  Created: {config_path}")

    # Create example prompt
    example_prompt = {
        "name": "example_extraction",
        "description": "Extract structured data from text",
        "model": "gpt-4o",
        "schema": {
            "type": "object",
            "title": "Person",
            "properties": {
                "name": {"type": "string", "description": "Person's name"},
                "age": {"type": "integer", "description": "Person's age"},
            },
            "required": ["name", "age"],
        },
        "prompt_template": "Extract the person's name and age from: {text}",
    }

    example_path = project_path / "prompts" / "example.yaml"
    if not example_path.exists():
        with open(example_path, "w") as f:
            yaml.dump(example_prompt, f, default_flow_style=False, sort_keys=False)
        console.print(f"  Created: {example_path}")

    console.print("\n[green]PromptGuard initialized successfully![/green]")
    console.print("\nNext steps:")
    console.print("  1. Set your API key: export OPENAI_API_KEY=sk-...")
    console.print(
        '  2. Run a prompt: promptguard run prompts/example.yaml -i \'{"text": "John is 30"}\''
    )


@cli.command()
@click.argument("prompt_file", type=click.Path(exists=True))
@click.option("--input", "-i", "input_data", help="Input data (JSON string or file path)")
@click.option("--output", "-o", help="Output file path")
@click.option("--verbose", "-v", is_flag=True, help="Verbose output")
@click.option("--model", "-m", help="Override model")
def run(
    prompt_file: str,
    input_data: str | None,
    output: str | None,
    verbose: bool,
    model: str | None,
) -> None:
    """Run a prompt from a YAML file.

    The prompt file should contain:
    - model: LLM model to use
    - schema: JSON Schema for output
    - prompt_template: Template with {variables}
    """
    from promptguard import llm_call

    # Load prompt definition
    with open(prompt_file) as f:
        prompt_def = yaml.safe_load(f)

    # Parse input
    variables: dict = {}
    if input_data:
        if Path(input_data).exists():
            with open(input_data) as f:
                variables = json.load(f)
        else:
            try:
                variables = json.loads(input_data)
            except json.JSONDecodeError:
                console.print(f"[red]Error: Invalid JSON input: {input_data}[/red]")
                sys.exit(1)

    # Format prompt
    prompt_template = prompt_def.get("prompt_template", prompt_def.get("prompt", ""))
    try:
        prompt = prompt_template.format(**variables)
    except KeyError as e:
        console.print(f"[red]Error: Missing variable in input: {e}[/red]")
        sys.exit(1)

    if verbose:
        console.print("[dim]Prompt:[/dim]")
        console.print(Syntax(prompt, "text"))
        console.print()

    # Get model
    use_model = model or prompt_def.get("model", "gpt-4o")

    # Execute
    with console.status(f"Calling {use_model}..."):
        try:
            result = llm_call(
                prompt=prompt,
                model=use_model,
                schema=prompt_def["schema"],
            )
        except Exception as e:
            console.print(f"[red]Error: {e}[/red]")
            sys.exit(1)

    # Format output
    if hasattr(result.data, "model_dump"):
        output_data = result.data.model_dump()
    elif hasattr(result.data, "__dict__"):
        output_data = result.data.__dict__
    else:
        output_data = result.data

    output_json = json.dumps(output_data, indent=2, default=str)

    if output:
        with open(output, "w") as f:
            f.write(output_json)
        console.print(f"[green]Output saved to {output}[/green]")
    else:
        console.print(Syntax(output_json, "json"))

    if verbose:
        console.print()
        console.print(f"[dim]Version: {result.version_hash}[/dim]")
        console.print(f"[dim]Repair attempts: {result.repair_attempts}[/dim]")
        console.print(f"[dim]Duration: {result.duration_ms:.0f}ms[/dim]")
        if result.usage:
            console.print(f"[dim]Tokens: {result.usage.get('total_tokens', 'N/A')}[/dim]")


@cli.command()
@click.option("--path", "-p", default=".", help="Project path")
@click.option("--update", "-u", is_flag=True, help="Update snapshots")
@click.option("--filter", "-f", "filter_pattern", help="Filter tests by pattern")
@click.option("--verbose", "-v", is_flag=True, help="Verbose output")
def test(
    path: str,
    update: bool,
    filter_pattern: str | None,
    verbose: bool,
) -> None:
    """Run prompt regression tests.

    Runs pytest with PromptGuard snapshot testing enabled.
    """
    import subprocess

    args = ["pytest", path]

    if verbose:
        args.append("-v")

    if update:
        args.append("--update-snapshots")

    if filter_pattern:
        args.extend(["-k", filter_pattern])

    # Add our plugin
    args.extend(["-p", "promptguard.testing.pytest_plugin"])

    console.print(f"[dim]Running: {' '.join(args)}[/dim]\n")

    result = subprocess.run(args)
    sys.exit(result.returncode)


@cli.command()
@click.argument("hash1")
@click.argument("hash2", required=False)
@click.option("--semantic", "-s", is_flag=True, help="Use semantic comparison")
def diff(hash1: str, hash2: str | None, semantic: bool) -> None:
    """Compare two prompt versions or snapshots.

    If only one hash is provided, shows the version details.
    If two hashes are provided, shows the differences.
    """
    from promptguard.core.hashing import VersionStore

    version_store = VersionStore(Path(".promptguard/versions"))

    # Load version(s)
    v1 = version_store.load(hash1)

    if v1 is None:
        console.print(f"[red]Version {hash1} not found[/red]")
        sys.exit(1)

    if hash2:
        v2 = version_store.load(hash2)
        if v2 is None:
            console.print(f"[red]Version {hash2} not found[/red]")
            sys.exit(1)

        # Compare two versions
        table = Table(title="Version Comparison")
        table.add_column("Field", style="cyan")
        table.add_column(f"Version {hash1[:8]}", style="green")
        table.add_column(f"Version {hash2[:8]}", style="yellow")

        table.add_row("Model", v1.model, v2.model)
        table.add_row("Temperature", str(v1.temperature), str(v2.temperature))
        table.add_row("Created", v1.created_at.isoformat()[:19], v2.created_at.isoformat()[:19])
        table.add_row("Schema Hash", v1.schema_hash or "N/A", v2.schema_hash or "N/A")

        console.print(table)

        if v1.prompt != v2.prompt:
            console.print("\n[bold]Prompt changed[/bold]")
            console.print("[dim]Version 1:[/dim]")
            console.print(Syntax(v1.prompt[:500], "text"))
            console.print("[dim]Version 2:[/dim]")
            console.print(Syntax(v2.prompt[:500], "text"))

    else:
        # Show single version
        console.print(f"[bold]Version {hash1}[/bold]\n")

        table = Table(show_header=False)
        table.add_column("Field", style="cyan")
        table.add_column("Value")

        table.add_row("Hash", v1.hash)
        table.add_row("Model", v1.model)
        table.add_row("Temperature", str(v1.temperature))
        table.add_row("Schema Hash", v1.schema_hash or "N/A")
        table.add_row("Created", v1.created_at.isoformat())

        if v1.metadata:
            table.add_row("Metadata", json.dumps(v1.metadata, indent=2))

        console.print(table)

        console.print("\n[bold]Prompt:[/bold]")
        console.print(Syntax(v1.prompt, "text"))


@cli.command()
@click.option("--limit", "-n", default=20, help="Number of versions to show")
@click.option("--model", "-m", help="Filter by model")
def history(limit: int, model: str | None) -> None:
    """Show prompt version history.

    Lists recent prompt versions with their details.
    """
    from promptguard.core.hashing import VersionStore

    version_store = VersionStore(Path(".promptguard/versions"))
    versions = version_store.list_versions(limit=limit * 2)  # Get extra for filtering

    # Filter by model if specified
    if model:
        versions = [v for v in versions if model.lower() in v.get("model", "").lower()]

    versions = versions[:limit]

    if not versions:
        console.print("[dim]No versions found[/dim]")
        return

    table = Table(title="Prompt Version History")
    table.add_column("Hash", style="cyan")
    table.add_column("Model", style="green")
    table.add_column("Created", style="yellow")
    table.add_column("Preview", style="dim")

    for v in versions:
        preview = v.get("prompt_preview", "")
        if len(preview) > 40:
            preview = preview[:40] + "..."

        table.add_row(
            v["hash"],
            v.get("model", "unknown"),
            v.get("created_at", "")[:19],
            preview,
        )

    console.print(table)


@cli.command()
def stats() -> None:
    """Show statistics about stored versions and snapshots."""
    from promptguard.core.hashing import VersionStore
    from promptguard.testing.snapshot import SnapshotStore

    version_store = VersionStore(Path(".promptguard/versions"))
    snapshot_store = SnapshotStore(Path(".promptguard/snapshots"))

    version_stats = version_store.get_stats()
    snapshot_count = len(snapshot_store.list_snapshots())

    console.print("[bold]PromptGuard Statistics[/bold]\n")

    table = Table(show_header=False)
    table.add_column("Metric", style="cyan")
    table.add_column("Value")

    table.add_row("Total Versions", str(version_stats["total_versions"]))
    table.add_row("Total Snapshots", str(snapshot_count))

    if version_stats["oldest"]:
        table.add_row("Oldest Version", version_stats["oldest"][:19])
    if version_stats["newest"]:
        table.add_row("Newest Version", version_stats["newest"][:19])

    console.print(table)

    if version_stats["models"]:
        console.print("\n[bold]Models Used:[/bold]")
        model_table = Table()
        model_table.add_column("Model", style="green")
        model_table.add_column("Count")

        for model_name, count in sorted(
            version_stats["models"].items(),
            key=lambda x: x[1],
            reverse=True,
        ):
            model_table.add_row(model_name, str(count))

        console.print(model_table)


@cli.command()
@click.confirmation_option(prompt="Are you sure you want to clear all versions?")
def clear() -> None:
    """Clear all stored versions and snapshots."""
    from promptguard.core.hashing import VersionStore
    from promptguard.testing.snapshot import SnapshotStore

    version_store = VersionStore(Path(".promptguard/versions"))
    snapshot_store = SnapshotStore(Path(".promptguard/snapshots"))

    versions_deleted = version_store.clear()
    snapshots_deleted = snapshot_store.clear()

    console.print(
        f"[green]Cleared {versions_deleted} versions and {snapshots_deleted} snapshots[/green]"
    )


if __name__ == "__main__":
    cli()
