"""Multi-provider PromptGuard usage examples.

This example demonstrates using different LLM providers:
- OpenAI (GPT-4o)
- Anthropic (Claude)
- Google (Gemini)
- Local models (Ollama)

Prerequisites:
    pip install promptguard[all]
    export OPENAI_API_KEY=your-key
    export ANTHROPIC_API_KEY=your-key
    export GOOGLE_API_KEY=your-key
"""

from pydantic import BaseModel, Field

from promptguard import llm_call
from promptguard.core.exceptions import ProviderNotFoundError


class Translation(BaseModel):
    """Schema for translation output."""

    original: str = Field(description="Original text")
    translated: str = Field(description="Translated text")
    source_language: str = Field(description="Detected source language")
    confidence: float = Field(ge=0, le=1, description="Translation confidence")


def translate_with_provider(text: str, model: str) -> dict:
    """Attempt translation with a specific provider."""
    try:
        result = llm_call(
            prompt=f"Translate this text to English:\n\n{text}",
            model=model,
            schema=Translation,
        )
        return {
            "provider": result.provider,
            "model": result.model,
            "translation": result.data.translated,
            "source_language": result.data.source_language,
            "duration_ms": result.duration_ms,
        }
    except ProviderNotFoundError as e:
        return {"provider": model.split(":")[0] if ":" in model else model, "error": str(e)}
    except Exception as e:
        return {"provider": model, "error": str(e)}


def main():
    text = "Bonjour, comment allez-vous aujourd'hui?"

    print(f"Text to translate: {text}\n")

    # Different providers and their model identifiers
    providers = [
        ("OpenAI", "gpt-4o-mini"),
        ("Anthropic", "anthropic:claude-3-haiku-20240307"),
        ("Google", "gemini:gemini-1.5-flash"),
        ("Local", "local:llama2"),  # Requires Ollama running
    ]

    print("Testing providers:\n")

    for name, model in providers:
        print(f"{name} ({model}):")
        result = translate_with_provider(text, model)

        if "error" in result:
            print(f"  Error: {result['error']}")
        else:
            print(f"  Translation: {result['translation']}")
            print(f"  Source: {result['source_language']}")
            print(f"  Duration: {result['duration_ms']:.0f}ms")
        print()


if __name__ == "__main__":
    main()
