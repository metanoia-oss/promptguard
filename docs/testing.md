# Testing & Regression

PromptGuard includes built-in tools for snapshot testing and regression detection of LLM prompts.

## Why test prompts?

LLM outputs change when:

- You update the prompt text
- The provider updates the model
- You switch models
- Temperature or parameters change

Without testing, these changes break silently. PromptGuard makes prompt behavior testable.

## Setup

```bash
promptguard init
```

This creates a `.promptguard/` directory in your project to store snapshots and history.

## Running tests

```bash
promptguard test
```

This replays your registered prompts against their schemas, compares outputs to stored snapshots, and reports any drift.

## Snapshot workflow

1. Define prompts with schemas
2. Run `promptguard test` to generate baseline snapshots
3. On subsequent runs, PromptGuard compares new outputs to baselines
4. Review and accept changes with `promptguard history` and `promptguard diff`

## Versioning

Every prompt execution is versioned. View history:

```bash
promptguard history
promptguard diff <hash>
```

This lets you track how outputs change over time and catch regressions before they reach production.

## Statistics

```bash
promptguard stats
```

View success rates, repair frequency, and drift metrics across your prompts.
