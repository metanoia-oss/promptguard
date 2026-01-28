"""Semantic comparison utilities for testing."""

from __future__ import annotations

from typing import Any, Optional


class SemanticComparator:
    """Compares text using semantic similarity.

    Uses sentence-transformers to generate embeddings and compute
    cosine similarity between texts.

    Example:
        comparator = SemanticComparator()
        similarity = comparator.similarity(
            "The cat sat on the mat",
            "A feline rested on the rug"
        )
        print(f"Similarity: {similarity:.2f}")  # ~0.8
    """

    def __init__(self, model_name: str = "all-MiniLM-L6-v2") -> None:
        """Initialize the comparator.

        Args:
            model_name: Name of the sentence-transformers model to use.
        """
        self._model = None
        self._model_name = model_name

    @property
    def model(self):
        """Lazy load the sentence transformer model."""
        if self._model is None:
            try:
                from sentence_transformers import SentenceTransformer
                self._model = SentenceTransformer(self._model_name)
            except ImportError:
                raise ImportError(
                    "sentence-transformers required for semantic comparison. "
                    "Install with: pip install promptguard[testing]"
                )
        return self._model

    def similarity(self, text1: str, text2: str) -> float:
        """Calculate cosine similarity between two texts.

        Args:
            text1: First text.
            text2: Second text.

        Returns:
            Similarity score between 0 and 1.
        """
        import numpy as np

        embeddings = self.model.encode([text1, text2])
        similarity = np.dot(embeddings[0], embeddings[1]) / (
            np.linalg.norm(embeddings[0]) * np.linalg.norm(embeddings[1])
        )
        return float(similarity)

    def batch_similarity(
        self,
        texts1: list[str],
        texts2: list[str],
    ) -> list[float]:
        """Calculate similarities for multiple pairs efficiently.

        Args:
            texts1: List of first texts.
            texts2: List of second texts.

        Returns:
            List of similarity scores.

        Raises:
            ValueError: If lists have different lengths.
        """
        if len(texts1) != len(texts2):
            raise ValueError("Lists must have the same length")

        import numpy as np

        # Encode all texts at once for efficiency
        all_texts = texts1 + texts2
        embeddings = self.model.encode(all_texts)

        n = len(texts1)
        similarities = []
        for i in range(n):
            sim = np.dot(embeddings[i], embeddings[n + i]) / (
                np.linalg.norm(embeddings[i]) * np.linalg.norm(embeddings[n + i])
            )
            similarities.append(float(sim))
        return similarities

    def most_similar(
        self,
        query: str,
        candidates: list[str],
        top_k: int = 5,
    ) -> list[tuple[str, float]]:
        """Find most similar texts to a query.

        Args:
            query: Query text.
            candidates: List of candidate texts.
            top_k: Number of results to return.

        Returns:
            List of (text, similarity) tuples, sorted by similarity.
        """
        import numpy as np

        all_texts = [query] + candidates
        embeddings = self.model.encode(all_texts)

        query_embedding = embeddings[0]
        candidate_embeddings = embeddings[1:]

        similarities = []
        for i, emb in enumerate(candidate_embeddings):
            sim = np.dot(query_embedding, emb) / (
                np.linalg.norm(query_embedding) * np.linalg.norm(emb)
            )
            similarities.append((candidates[i], float(sim)))

        similarities.sort(key=lambda x: x[1], reverse=True)
        return similarities[:top_k]


class LLMJudge:
    """Uses an LLM to evaluate semantic equivalence.

    For complex comparisons where embedding similarity isn't sufficient,
    an LLM can be used to judge if two outputs are semantically equivalent.

    Example:
        judge = LLMJudge(model="gpt-4o-mini")
        result = judge.judge(output1, output2)
        if result["equivalent"]:
            print("Outputs are equivalent!")
    """

    JUDGE_PROMPT = """Compare these two outputs and determine if they are semantically equivalent.
Consider: Do they convey the same meaning? Do they contain the same key information?

Output 1:
{output1}

Output 2:
{output2}

Respond with a JSON object:
{{"equivalent": true/false, "confidence": 0.0-1.0, "explanation": "brief explanation"}}"""

    def __init__(self, model: str = "gpt-4o-mini") -> None:
        """Initialize the judge.

        Args:
            model: Model to use for judging.
        """
        self.model = model

    def judge(self, output1: Any, output2: Any) -> dict[str, Any]:
        """Judge if two outputs are semantically equivalent.

        Args:
            output1: First output.
            output2: Second output.

        Returns:
            Dictionary with:
                - equivalent: bool
                - confidence: float (0-1)
                - explanation: str
        """
        from pydantic import BaseModel

        from promptguard import llm_call

        class JudgeResult(BaseModel):
            equivalent: bool
            confidence: float
            explanation: str

        prompt = self.JUDGE_PROMPT.format(
            output1=str(output1),
            output2=str(output2),
        )

        result = llm_call(
            prompt=prompt,
            model=self.model,
            schema=JudgeResult,
            save_version=False,  # Don't save judge calls
        )
        return result.data.model_dump()

    def batch_judge(
        self,
        pairs: list[tuple[Any, Any]],
    ) -> list[dict[str, Any]]:
        """Judge multiple pairs.

        Args:
            pairs: List of (output1, output2) tuples.

        Returns:
            List of judge results.
        """
        return [self.judge(a, b) for a, b in pairs]


def create_semantic_comparator(
    use_llm: bool = False,
    model_name: Optional[str] = None,
) -> callable:
    """Factory function to create a semantic comparator function.

    Args:
        use_llm: Whether to use LLM-based comparison.
        model_name: Model name for the comparator.

    Returns:
        Comparator function that takes two strings and returns similarity (0-1).
    """
    if use_llm:
        judge = LLMJudge(model=model_name or "gpt-4o-mini")

        def llm_compare(text1: str, text2: str) -> float:
            result = judge.judge(text1, text2)
            return result["confidence"] if result["equivalent"] else 0.0

        return llm_compare
    else:
        comparator = SemanticComparator(model_name=model_name or "all-MiniLM-L6-v2")
        return comparator.similarity
