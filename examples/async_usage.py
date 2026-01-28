"""Async PromptGuard usage examples.

This example demonstrates async operations with PromptGuard:
- Concurrent LLM calls
- Async/await syntax
- Batch processing

Prerequisites:
    pip install promptguard[openai]
    export OPENAI_API_KEY=your-key
"""

import asyncio

from pydantic import BaseModel, Field

from promptguard import allm_call


class SentimentResult(BaseModel):
    """Schema for sentiment analysis output."""
    sentiment: str = Field(description="positive, negative, or neutral")
    confidence: float = Field(ge=0, le=1, description="Confidence score")
    keywords: list[str] = Field(description="Key words that indicate sentiment")


async def analyze_single(review: str) -> SentimentResult:
    """Analyze a single review asynchronously."""
    result = await allm_call(
        prompt=f"Analyze the sentiment of this review:\n\n{review}",
        model="gpt-4o-mini",
        schema=SentimentResult,
    )
    return result.data


async def analyze_reviews(reviews: list[str]) -> list[SentimentResult]:
    """Analyze multiple reviews concurrently."""
    tasks = [analyze_single(review) for review in reviews]
    return await asyncio.gather(*tasks)


async def main():
    reviews = [
        "This product exceeded my expectations! Absolutely love it.",
        "Terrible experience. The item broke after one day.",
        "It's okay, nothing special but does the job.",
        "Fast shipping and great customer service!",
        "Would not recommend. Poor quality for the price.",
    ]

    print("Analyzing reviews concurrently...\n")

    results = await analyze_reviews(reviews)

    for review, result in zip(reviews, results):
        print(f"Review: {review[:50]}...")
        print(f"  Sentiment: {result.sentiment}")
        print(f"  Confidence: {result.confidence:.0%}")
        print(f"  Keywords: {', '.join(result.keywords)}")
        print()


if __name__ == "__main__":
    asyncio.run(main())
