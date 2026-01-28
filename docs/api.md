# API Reference

## Core Functions

### `llm_call`

Synchronous structured LLM call with schema validation and auto-repair.

```python
from promptguard import llm_call

result = llm_call(
    model="gpt-4o",
    prompt="Extract the person's info",
    schema=Person,
    max_retries=3,       # repair attempts on validation failure
    temperature=0.0,
)
```

**Parameters:**

| Parameter | Type | Description |
|-----------|------|-------------|
| `model` | `str` | Model identifier (e.g. `"gpt-4o"`, `"claude-3-sonnet"`) |
| `prompt` | `str` | The prompt to send |
| `schema` | `type` | Pydantic model, TypedDict, dataclass, or JSON schema dict |
| `max_retries` | `int` | Number of repair attempts (default: 3) |
| `temperature` | `float` | Sampling temperature |

**Returns:** `LLMResult` with `.data` (validated instance) and `.metadata`.

---

### `allm_call`

Async version of `llm_call`. Same parameters and behavior.

```python
from promptguard import allm_call

result = await allm_call(
    model="gpt-4o",
    prompt="Extract info",
    schema=Person,
)
```

---

## Schema Types

PromptGuard accepts four schema types:

### Pydantic BaseModel

```python
from pydantic import BaseModel

class Person(BaseModel):
    name: str
    age: int
```

### TypedDict

```python
from typing import TypedDict

class Person(TypedDict):
    name: str
    age: int
```

### dataclass

```python
from dataclasses import dataclass

@dataclass
class Person:
    name: str
    age: int
```

### JSON Schema (dict)

```python
schema = {
    "type": "object",
    "properties": {
        "name": {"type": "string"},
        "age": {"type": "integer"}
    },
    "required": ["name", "age"]
}
```

---

## Configuration

PromptGuard auto-detects providers from model names. Provider-specific API keys are read from environment variables:

| Provider | Environment Variable |
|----------|---------------------|
| OpenAI | `OPENAI_API_KEY` |
| Anthropic | `ANTHROPIC_API_KEY` |
| Google | `GOOGLE_API_KEY` |
| Local | `OPENAI_API_BASE` (for compatible servers) |
