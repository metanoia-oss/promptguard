# PromptGuard

[![CI](https://github.com/metanoia-oss/promptguard/actions/workflows/ci.yml/badge.svg)](https://github.com/metanoia-oss/promptguard/actions/workflows/ci.yml)
[![PyPI version](https://img.shields.io/pypi/v/llm-promptguard.svg)](https://pypi.org/project/llm-promptguard/)
[![Python versions](https://img.shields.io/pypi/pyversions/llm-promptguard.svg)](https://pypi.org/project/llm-promptguard/)
[![License](https://img.shields.io/badge/license-Apache%202.0-blue.svg)](LICENSE)

**Never parse LLM output again.**

LLMs are probabilistic.
Software is deterministic.

**PromptGuard bridges the gap.**

PromptGuard is a production-grade reliability layer that turns Large Language Models into safe, structured, testable software components.

If your application depends on LLM outputs — agents, workflows, background jobs, voice systems, document parsing — then PromptGuard prevents the failures that eventually break every LLM app in production.

---

## Why PromptGuard Exists

Every LLM developer eventually runs into this:

```python
json.loads(llm_output)  # crashes in prod
```

Because LLMs:
- return invalid JSON
- hallucinate fields
- change output formats
- break silently after model updates
- fail one out of every N requests

These failures cause:
- background job crashes
- broken agents
- corrupted pipelines
- silent data loss
- 2am production incidents

**PromptGuard eliminates this entire class of problems.**

---

## What PromptGuard Guarantees

- **Schema-valid outputs** — always
- **Automatic repair** when models misbehave
- **Deterministic structured data**
- **Prompt regression testing**
- **Provider-agnostic execution**

No regex.
No fragile parsing.
No silent failures.

---

## Why PromptGuard Over Alternatives?

| Feature | PromptGuard | Instructor | Outlines |
|---------|------------|-----------|----------|
| Auto repair loop | Yes (N retries) | 1 retry | No |
| Multi-provider | 4 built-in | OpenAI-centric | Multiple |
| Prompt versioning | Built-in | No | No |
| Regression testing | Built-in | No | No |
| Schema types | 4 (Pydantic, TypedDict, dataclass, JSON) | Pydantic | Limited |
| CLI tooling | Yes | No | No |

PromptGuard is a **reliability layer**, not just a parser. Versioning + testing + repair in one package.

---

## Installation

```bash
pip install llm-promptguard

# With provider-specific dependencies
pip install llm-promptguard[openai]
pip install llm-promptguard[anthropic]
pip install llm-promptguard[google]
pip install llm-promptguard[all]
```

---

## Quick Start

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

If the model returns invalid output, PromptGuard automatically:
1. detects the schema violation
2. explains the error
3. re-prompts the model
4. repairs the output
5. returns guaranteed valid data

---

## Real-World Examples

### Resume & Document Parsing

```python
class Resume(BaseModel):
    name: str
    skills: list[str]
    years_experience: int

resume = llm_call(
    prompt=resume_text,
    model="gpt-4o",
    schema=Resume
)
```

- No missing fields
- No malformed JSON
- No broken pipelines

---

### Email Triage Automation

```python
class EmailIntent(BaseModel):
    intent: str
    urgency: int
    requires_reply: bool

intent = llm_call(
    prompt=email_body,
    model="gpt-4o",
    schema=EmailIntent
)
```

Safe to run in background workers.
Safe to store in databases.
Safe to trigger workflows.

---

### AI Agents (Tool Calling)

```python
class ToolArgs(BaseModel):
    action: str
    resource_id: str

args = llm_call(
    prompt=agent_prompt,
    model="gpt-4o",
    schema=ToolArgs
)

run_tool(**args.data.model_dump())
```

PromptGuard prevents agents from:
- hallucinating arguments
- calling tools incorrectly
- freezing execution chains

---

### Voice Agents & Call Automation

```
Speech → LLM → Action
```

If structure breaks, the call fails.

PromptGuard ensures voice systems always receive valid commands.

---

### Background Jobs & Queues

```python
@worker.task
async def process_document(text):
    result = await allm_call(
        prompt=text,
        model="gpt-4o",
        schema=Extraction
    )
    save(result.data)
```

No retries.
No poison messages.
No corrupted jobs.

---

## LangChain Integration

PromptGuard works seamlessly with LangChain.

```python
from langchain.tools import Tool
from promptguard import llm_call
from pydantic import BaseModel

class SearchArgs(BaseModel):
    query: str

search_tool = Tool(
    name="search",
    func=lambda q: search_api(q),
    description="Search the web"
)

args = llm_call(
    prompt="Search for Tesla earnings",
    model="gpt-4o",
    schema=SearchArgs
)

search_tool.run(args.data.query)
```

PromptGuard becomes the type-safe boundary between agents and tools.

---

## Prompt Regression Testing

```bash
promptguard test
```

- detects prompt drift
- catches model behavior changes
- prevents silent regressions

LLM prompts finally become testable.

---

## The Demo

```python
class Order(BaseModel):
    product: str
    quantity: int
    price: float

order = llm_call(
    prompt="Buy two iPhones for $999 each",
    model="gpt-4o",
    schema=Order
)

print(order.data)
# Order(product='iPhone', quantity=2, price=999.0)
```

No parsing.
No retries.
No crashes.

**Just software-safe AI.**

---

## Mental Model

**Without PromptGuard:**
```
LLM → text → hope → bugs
```

**With PromptGuard:**
```
LLM → contract → software
```

---

## When You Should Use PromptGuard

If your application:
- runs LLMs in production
- executes workflows or agents
- parses model output
- stores results in databases
- depends on structure

Then **PromptGuard is not optional.**

---

## CLI Commands

```bash
promptguard init           # Initialize in a project
promptguard run prompt.yaml  # Run a prompt from YAML
promptguard test           # Run regression tests
promptguard history        # Show version history
promptguard diff <hash>    # Compare versions
promptguard stats          # Show statistics
```

---

## Supported Providers

| Provider | Model Examples |
|----------|---------------|
| OpenAI | gpt-4o, gpt-4o-mini, gpt-4-turbo |
| Anthropic | claude-3-opus, claude-3-sonnet, claude-3-haiku |
| Google | gemini-1.5-pro, gemini-1.5-flash |
| Local | Ollama, LM Studio, vLLM (OpenAI-compatible) |

---

## Testing with real LLM Calls
- export OPENAI_API_KEY=sk-...                                                                                                                                                                                                                                                                                            
- ./scripts/test_all.sh

---
## Documentation

Full docs at [metanoia-oss.github.io/promptguard](https://metanoia-oss.github.io/promptguard/).

---

## License

Apache 2.0

---

**PromptGuard** — because production AI must be dependable.
