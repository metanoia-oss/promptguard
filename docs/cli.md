# CLI Reference

PromptGuard provides a command-line interface for managing prompts, running tests, and inspecting history.

## Commands

### `promptguard init`

Initialize PromptGuard in the current project. Creates the `.promptguard/` directory for snapshots and configuration.

```bash
promptguard init
```

### `promptguard run`

Execute a prompt defined in a YAML file.

```bash
promptguard run prompt.yaml
```

### `promptguard test`

Run regression tests against stored snapshots. Detects prompt drift and schema validation failures.

```bash
promptguard test
```

### `promptguard history`

Show version history for tracked prompts.

```bash
promptguard history
```

### `promptguard diff`

Compare two versions of a prompt's output.

```bash
promptguard diff <hash>
```

### `promptguard stats`

Display statistics: success rates, repair frequency, and drift metrics.

```bash
promptguard stats
```
