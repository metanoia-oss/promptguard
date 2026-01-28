# Provider Setup

PromptGuard supports multiple LLM providers out of the box. Install the relevant extra and set your API key.

## OpenAI

```bash
pip install llm-promptguard[openai]
export OPENAI_API_KEY=sk-...
```

```python
result = llm_call(model="gpt-4o", prompt="...", schema=MySchema)
```

Supported models: `gpt-4o`, `gpt-4o-mini`, `gpt-4-turbo`, and other OpenAI chat models.

## Anthropic

```bash
pip install llm-promptguard[anthropic]
export ANTHROPIC_API_KEY=sk-ant-...
```

```python
result = llm_call(model="claude-3-sonnet-20240229", prompt="...", schema=MySchema)
```

Supported models: `claude-3-opus`, `claude-3-sonnet`, `claude-3-haiku`, and other Anthropic models.

## Google Gemini

```bash
pip install llm-promptguard[google]
export GOOGLE_API_KEY=...
```

```python
result = llm_call(model="gemini-1.5-pro", prompt="...", schema=MySchema)
```

Supported models: `gemini-1.5-pro`, `gemini-1.5-flash`, and other Gemini models.

## Local Models (Ollama, LM Studio, vLLM)

Any OpenAI-compatible API server works with PromptGuard:

```bash
export OPENAI_API_BASE=http://localhost:11434/v1
```

```python
result = llm_call(model="llama3", prompt="...", schema=MySchema)
```

This works with:

- **Ollama** — `ollama serve` exposes an OpenAI-compatible endpoint
- **LM Studio** — enable the local server in settings
- **vLLM** — start with `--api-key` and `--served-model-name`
