# Contributing to PromptGuard

Thank you for your interest in contributing to PromptGuard. Every contribution matters, whether it's a bug report, a documentation fix, or a new feature.

## Mission

PromptGuard exists to make LLMs behave like dependable software components. Every decision we make serves that goal: reliability over cleverness, correctness over speed, simplicity over flexibility.

## Governance

PromptGuard is maintained by a small group of core contributors. Decisions are made through discussion on GitHub issues and pull requests. There is no formal committee. If your argument is sound and your code is solid, it gets merged.

Major architectural changes go through an RFC process: open an issue with the `[RFC]` prefix, describe the problem, propose a solution, and wait for feedback before writing code.

## Maintainer Philosophy

- **Contributors are more important than code.** A rough PR from a first-time contributor is worth more than a perfect PR that discourages participation.
- **Ship small, ship often.** We prefer small, focused PRs over large rewrites.
- **Tests are not optional.** If it changes behavior, it needs a test.
- **Documentation is not an afterthought.** If a user can't figure it out, it doesn't work.
- **Be direct, not harsh.** Code review should be honest and constructive. No ego.

## How to Contribute

### Reporting Bugs

Open an issue with:
- What you expected to happen
- What actually happened
- Minimal reproduction steps
- Python version, OS, and PromptGuard version

### Suggesting Features

Open an issue with the `[Feature]` prefix. Describe the problem you're solving, not just the solution you want. Context helps us design the right thing.

### Submitting Code

1. **Fork the repository** and create a branch from `main`.

2. **Set up your environment:**
   ```bash
   python -m venv .venv
   source .venv/bin/activate
   pip install -e ".[dev]"  # PyPI package: llm-promptguard
   ```

3. **Make your changes.** Keep them focused. One PR, one concern.

4. **Write tests.** Run the existing suite to make sure nothing breaks:
   ```bash
   pytest tests/ -v
   ```

5. **Follow the existing code style.** We use ruff for linting:
   ```bash
   ruff check promptguard/
   ruff format promptguard/
   ```

6. **Open a pull request.** Describe what changed and why. Link to the relevant issue if one exists.

### What Makes a Good PR

- Solves one problem
- Includes tests for new behavior
- Doesn't break existing tests
- Has a clear description
- Follows existing patterns in the codebase

### What We Will Not Merge

- PRs without tests for behavior changes
- Large refactors without prior discussion
- Changes that break backward compatibility without an RFC
- Code that introduces security vulnerabilities

## Development Guide

### Project Structure

```
promptguard/
  core/         # Engine, config, validation, repair, hashing
  providers/    # LLM provider implementations (OpenAI, Anthropic, etc.)
  schemas/      # Schema adapters (Pydantic, TypedDict, dataclass, JSON Schema)
  testing/      # Snapshot testing, semantic diff, pytest plugin
  cli/          # CLI commands
tests/          # Test suite
scripts/        # Development and testing scripts
examples/       # Usage examples
```

### Running Tests

```bash
# Unit tests (no API key needed)
pytest tests/test_config.py tests/test_adapters.py tests/test_validator.py tests/test_hashing.py tests/test_snapshot.py -v

# E2E tests
pytest tests/test_e2e.py -v

# Integration tests (requires OPENAI_API_KEY)
pytest tests/test_integration.py -v

# Everything
./scripts/test_all.sh
```

### Adding a New Provider

1. Create `promptguard/providers/your_provider.py`
2. Subclass `LLMProvider` from `promptguard/providers/base.py`
3. Register with `@ProviderRegistry.register("your_provider")`
4. Add optional dependency in `pyproject.toml`
5. Add tests

### Adding a New Schema Adapter

1. Subclass `SchemaAdapter` from `promptguard/schemas/adapters.py`
2. Implement `to_json_schema()`, `validate()`, `get_validation_errors()`
3. Register in `create_adapter()` factory
4. Add tests

## First-Time Contributors

Look for issues labeled `good first issue`. These are scoped to be approachable for someone new to the codebase.

If you're stuck, open a draft PR and ask for help. We'd rather guide you through it than have you give up.

## License

By contributing, you agree that your contributions will be licensed under the Apache License 2.0.
