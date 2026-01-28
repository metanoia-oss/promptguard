"""Basic PromptGuard usage examples.

This example demonstrates the core functionality of PromptGuard:
- Schema-guaranteed LLM outputs
- Automatic validation
- Type-safe results

Prerequisites:
    pip install promptguard[openai]
    export OPENAI_API_KEY=your-key
"""

from pydantic import BaseModel, Field

from promptguard import llm_call


# Define output schema using Pydantic
class EmailSummary(BaseModel):
    """Schema for email analysis output."""

    sender: str = Field(description="Email sender name")
    intent: str = Field(description="Primary intent of the email")
    urgency: int = Field(ge=1, le=5, description="Urgency level 1-5")
    summary: str = Field(max_length=200, description="Brief summary")
    action_items: list[str] = Field(default_factory=list, description="Required actions")


def main():
    # Sample email text
    email_text = """
    From: John Smith <john@example.com>
    Subject: Urgent: Project deadline moved up

    Hi team,

    I wanted to let you know that the client has requested we move up
    the project deadline by two weeks. This means we need to deliver
    by March 15th instead of March 29th.

    Please review your tasks and let me know by EOD if you foresee
    any blockers.

    Thanks,
    John
    """

    # Make the LLM call with guaranteed schema
    result = llm_call(
        prompt=f"Analyze this email:\n\n{email_text}",
        model="gpt-4o",
        schema=EmailSummary,
    )

    # Access typed data
    print(f"Sender: {result.data.sender}")
    print(f"Intent: {result.data.intent}")
    print(f"Urgency: {result.data.urgency}/5")
    print(f"Summary: {result.data.summary}")
    print(f"Action Items: {result.data.action_items}")

    # Metadata about the call
    print(f"\nVersion Hash: {result.version_hash}")
    print(f"Repair Attempts: {result.repair_attempts}")
    print(f"Duration: {result.duration_ms:.0f}ms")


if __name__ == "__main__":
    main()
