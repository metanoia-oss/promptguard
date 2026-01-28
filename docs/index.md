# PromptGuard

**Never parse LLM output again.**

PromptGuard is a production-grade reliability layer that turns Large Language Models into safe, structured, testable software components.

## What it does

- **Schema-valid outputs** — every time
- **Automatic repair** when models return bad data
- **Prompt regression testing** to catch drift
- **Multi-provider** — OpenAI, Anthropic, Google, local models
- **CLI tooling** for versioning, testing, and debugging

## Quick example

```python
from promptguard import llm_call
from pydantic import BaseModel

class Person(BaseModel):
    name: str
    age: int

result = llm_call(
    model="gpt-4o",
    prompt="John is 30 years old",
    schema=Person
)

print(result.data)
# Person(name='John', age=30)
```

If the model returns invalid output, PromptGuard automatically detects the violation, re-prompts, repairs, and returns guaranteed valid data.

## Install

```bash
pip install llm-promptguard

# With provider extras
pip install llm-promptguard[openai]
pip install llm-promptguard[anthropic]
pip install llm-promptguard[google]
pip install llm-promptguard[all]
```

## Next steps

- [Quick Start](quickstart.md) — get running in 5 minutes
- [API Reference](api.md) — full function and schema docs
- [Providers](providers.md) — setup guides for each provider
- [Testing](testing.md) — snapshot and regression testing
- [CLI Reference](cli.md) — command-line tools
