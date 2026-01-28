# Quick Start

## Installation

```bash
pip install llm-promptguard[openai]
```

## Set your API key

```bash
export OPENAI_API_KEY=sk-...
```

## Your first structured call

```python
from promptguard import llm_call
from pydantic import BaseModel

class Sentiment(BaseModel):
    label: str
    confidence: float

result = llm_call(
    model="gpt-4o-mini",
    prompt="The product is fantastic and well-made",
    schema=Sentiment
)

print(result.data)
# Sentiment(label='positive', confidence=0.95)
```

## Async usage

```python
from promptguard import allm_call

result = await allm_call(
    model="gpt-4o",
    prompt="Summarize this document...",
    schema=Summary
)
```

## Using different schema types

PromptGuard supports Pydantic models, TypedDicts, dataclasses, and raw JSON schemas:

```python
from typing import TypedDict

class Person(TypedDict):
    name: str
    age: int

result = llm_call(model="gpt-4o", prompt="John is 30", schema=Person)
```

```python
from dataclasses import dataclass

@dataclass
class Person:
    name: str
    age: int

result = llm_call(model="gpt-4o", prompt="John is 30", schema=Person)
```

## Running regression tests

```bash
promptguard init    # set up in your project
promptguard test    # run snapshot tests
```

See the [Testing guide](testing.md) for details.
