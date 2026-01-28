# We built an open-source reliability layer for LLMs

## The problem

Every team building with LLMs hits the same wall: the model returns garbage.

Not always. Not even often. But often enough to break production. Invalid JSON, missing fields, hallucinated keys, changed formats after a model update. The failure modes are endless and they all look the same — a crash at 2am, a corrupted pipeline, a broken agent.

The standard fix is defensive parsing: `try/except`, regex extraction, retry loops, manual validation. It works until it doesn't. And it never composes.

## Why existing tools aren't enough

**Instructor** is a good library. It patches the OpenAI client to return Pydantic models. But it's OpenAI-centric, gives you one retry, and has no concept of prompt versioning or regression testing. When your model updates and outputs drift, you find out from your users.

**Outlines** takes a different approach — constrained generation at the token level. Powerful for local models, but not applicable to API-based providers and not focused on the production reliability story.

Neither gives you the full picture: validation + repair + versioning + testing + multi-provider support.

## What PromptGuard does

PromptGuard is a reliability layer. You give it a prompt, a model, and a schema. It returns guaranteed valid structured data.

```python
from promptguard import llm_call
from pydantic import BaseModel

class Order(BaseModel):
    product: str
    quantity: int
    price: float

order = llm_call(
    model="gpt-4o",
    prompt="Buy two iPhones for $999 each",
    schema=Order
)
# Order(product='iPhone', quantity=2, price=999.0)
```

When validation fails, PromptGuard automatically re-prompts with the error context and repairs the output. Configurable retries, not just one shot.

It works with OpenAI, Anthropic, Google, and any OpenAI-compatible local server (Ollama, vLLM, LM Studio).

It accepts Pydantic models, TypedDicts, dataclasses, and raw JSON schemas.

It versions every prompt execution and provides CLI tools for regression testing:

```bash
promptguard test     # catch drift before production
promptguard history  # track output changes over time
promptguard diff     # compare versions
```

## Who it's for

If you're running LLMs in production — agents, workflows, document parsing, voice systems, background jobs — and you need structured output that doesn't break, PromptGuard is the missing layer.

## Get started

```bash
pip install llm-promptguard[openai]
```

- [GitHub](https://github.com/metanoia-oss/promptguard)
- [Documentation](https://metanoia-oss.github.io/promptguard/)
- [Quick Start](https://metanoia-oss.github.io/promptguard/quickstart/)

Apache 2.0 licensed. Contributions welcome.
