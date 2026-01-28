# Migration Guide

## From raw OpenAI SDK

**Before:**

```python
import json
from openai import OpenAI

client = OpenAI()

response = client.chat.completions.create(
    model="gpt-4o",
    messages=[{"role": "user", "content": "Extract: John is 30"}],
)

# Hope for valid JSON
try:
    data = json.loads(response.choices[0].message.content)
    name = data["name"]
    age = data["age"]
except (json.JSONDecodeError, KeyError) as e:
    # Now what?
    raise
```

**After:**

```python
from promptguard import llm_call
from pydantic import BaseModel

class Person(BaseModel):
    name: str
    age: int

result = llm_call(model="gpt-4o", prompt="Extract: John is 30", schema=Person)
print(result.data.name, result.data.age)
# Guaranteed valid. Auto-repaired if needed.
```

---

## From Instructor

**Before:**

```python
import instructor
from openai import OpenAI
from pydantic import BaseModel

client = instructor.from_openai(OpenAI())

class Person(BaseModel):
    name: str
    age: int

person = client.chat.completions.create(
    model="gpt-4o",
    response_model=Person,
    messages=[{"role": "user", "content": "Extract: John is 30"}],
)
```

**After:**

```python
from promptguard import llm_call
from pydantic import BaseModel

class Person(BaseModel):
    name: str
    age: int

result = llm_call(model="gpt-4o", prompt="Extract: John is 30", schema=Person)
person = result.data
```

### What you gain by switching

| Capability | Instructor | PromptGuard |
|-----------|-----------|-------------|
| Schema validation | Yes | Yes |
| Auto repair retries | 1 | Configurable (N) |
| Multi-provider | OpenAI-centric | OpenAI, Anthropic, Google, Local |
| Prompt versioning | No | Built-in |
| Regression testing | No | Built-in |
| CLI tooling | No | Yes |
| TypedDict / dataclass support | Limited | Full |
