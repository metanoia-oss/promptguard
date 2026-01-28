"""Automatic repair loop for invalid LLM outputs."""

from __future__ import annotations

import json
import random
import time
from collections.abc import Callable, Coroutine
from dataclasses import dataclass
from typing import Any, TypeVar

from promptguard.core.config import RetryConfig
from promptguard.core.exceptions import RepairExhaustedError, ValidationError
from promptguard.core.validator import OutputValidator

T = TypeVar("T")


@dataclass
class RepairAttempt:
    """Record of a single repair attempt.

    Attributes:
        attempt_number: The attempt number (1-indexed).
        raw_output: The raw LLM output for this attempt.
        error: ValidationError if the attempt failed, None if successful.
        duration_ms: Time taken for this attempt in milliseconds.
    """

    attempt_number: int
    raw_output: str
    error: ValidationError | None
    duration_ms: float


@dataclass
class RepairResult:
    """Result of the repair loop.

    Attributes:
        success: Whether validation eventually succeeded.
        result: The validated result if successful.
        attempts: List of all repair attempts made.
        total_duration_ms: Total time for all attempts in milliseconds.
    """

    success: bool
    result: Any | None
    attempts: list[RepairAttempt]
    total_duration_ms: float


class RepairLoop:
    """Automatic repair loop for invalid LLM outputs.

    When an LLM returns invalid output, the repair loop:
    1. Analyzes the validation errors
    2. Constructs a repair prompt explaining what went wrong
    3. Re-prompts the LLM with the correction instructions
    4. Repeats until valid output or max retries reached

    Example:
        validator = OutputValidator.for_schema(MySchema)
        loop = RepairLoop(config, validator)

        def complete_fn(prompt: str) -> str:
            return llm.complete(prompt)

        result = loop.run(
            complete_fn=complete_fn,
            initial_output=first_response,
            original_prompt="Extract data from...",
        )
    """

    REPAIR_SYSTEM_PROMPT = """You are a helpful assistant that produces valid JSON output.
Your previous response did not meet the required format. Please correct it.
You must respond with ONLY valid JSON that matches the schema - no explanation or markdown.
Do not include any text before or after the JSON object."""

    def __init__(
        self,
        config: RetryConfig,
        validator: OutputValidator,
        on_repair_attempt: Callable[[RepairAttempt], None] | None = None,
    ) -> None:
        """Initialize the repair loop.

        Args:
            config: Retry configuration.
            validator: Output validator instance.
            on_repair_attempt: Optional callback called after each attempt.
        """
        self.config = config
        self.validator = validator
        self.on_repair_attempt = on_repair_attempt

    def calculate_delay(self, attempt: int) -> float:
        """Calculate delay before the next retry attempt.

        Args:
            attempt: Current attempt number (0-indexed).

        Returns:
            Delay in seconds.
        """
        if self.config.backoff_strategy == "exponential":
            delay = self.config.base_delay * (2**attempt)
        elif self.config.backoff_strategy == "linear":
            delay = self.config.base_delay * (attempt + 1)
        else:  # fixed
            delay = self.config.base_delay

        # Apply max delay cap
        delay = min(delay, self.config.max_delay)

        # Apply jitter if enabled
        if self.config.jitter:
            delay = delay * (0.5 + random.random())

        return delay

    def build_repair_prompt(
        self,
        original_prompt: str,
        error: ValidationError,
        schema: dict[str, Any],
    ) -> str:
        """Build a repair prompt for the LLM.

        Args:
            original_prompt: The original user prompt.
            error: The validation error from the previous attempt.
            schema: The JSON schema for the expected output.

        Returns:
            Repair prompt string.
        """
        error_description = self.validator.format_errors_for_repair(error)
        schema_str = json.dumps(schema, indent=2)

        # Truncate the invalid output if too long
        invalid_output = error.raw_output
        if len(invalid_output) > 1000:
            invalid_output = invalid_output[:1000] + "... (truncated)"

        return f"""{error_description}

Original request: {original_prompt}

Required JSON Schema:
```json
{schema_str}
```

Previous invalid output:
```
{invalid_output}
```

Please provide a corrected response that matches the schema exactly.
Respond with ONLY the JSON object, no other text."""

    def run(
        self,
        complete_fn: Callable[[str], str],
        initial_output: str,
        original_prompt: str,
    ) -> RepairResult:
        """Run the repair loop.

        Args:
            complete_fn: Function that takes a prompt and returns LLM output.
            initial_output: The initial (potentially invalid) LLM output.
            original_prompt: The original user prompt.

        Returns:
            RepairResult with success status and validated result.

        Raises:
            RepairExhaustedError: If max retries reached without valid output.
        """
        attempts: list[RepairAttempt] = []
        start_time = time.time()
        current_output = initial_output
        last_error: ValidationError | None = None

        # First, try validating the initial output
        try:
            result = self.validator.validate(current_output)
            return RepairResult(
                success=True,
                result=result,
                attempts=[],
                total_duration_ms=(time.time() - start_time) * 1000,
            )
        except ValidationError as e:
            last_error = e

        schema = self.validator.get_json_schema()

        # Repair loop
        for attempt in range(self.config.max_retries):
            attempt_start = time.time()

            # Apply delay before retry (except first attempt)
            if attempt > 0:
                delay = self.calculate_delay(attempt - 1)
                time.sleep(delay)

            # Build repair prompt
            repair_prompt = self.build_repair_prompt(original_prompt, last_error, schema)

            # Get new output from LLM
            try:
                current_output = complete_fn(repair_prompt)
                result = self.validator.validate(current_output)

                # Success!
                attempt_record = RepairAttempt(
                    attempt_number=attempt + 1,
                    raw_output=current_output,
                    error=None,
                    duration_ms=(time.time() - attempt_start) * 1000,
                )
                attempts.append(attempt_record)

                if self.on_repair_attempt:
                    self.on_repair_attempt(attempt_record)

                return RepairResult(
                    success=True,
                    result=result,
                    attempts=attempts,
                    total_duration_ms=(time.time() - start_time) * 1000,
                )

            except ValidationError as e:
                last_error = e
                attempt_record = RepairAttempt(
                    attempt_number=attempt + 1,
                    raw_output=current_output,
                    error=e,
                    duration_ms=(time.time() - attempt_start) * 1000,
                )
                attempts.append(attempt_record)

                if self.on_repair_attempt:
                    self.on_repair_attempt(attempt_record)

        # All retries exhausted
        raise RepairExhaustedError(
            f"Failed to get valid output after {self.config.max_retries} repair attempts",
            attempts=len(attempts),
            last_error=last_error,  # type: ignore
        )


class AsyncRepairLoop(RepairLoop):
    """Async version of the repair loop.

    Identical to RepairLoop but uses async/await for the completion function.
    """

    async def run(
        self,
        complete_fn: Callable[[str], Coroutine[Any, Any, str]],
        initial_output: str,
        original_prompt: str,
    ) -> RepairResult:
        """Run the async repair loop.

        Args:
            complete_fn: Async function that takes a prompt and returns LLM output.
            initial_output: The initial (potentially invalid) LLM output.
            original_prompt: The original user prompt.

        Returns:
            RepairResult with success status and validated result.

        Raises:
            RepairExhaustedError: If max retries reached without valid output.
        """
        import asyncio

        attempts: list[RepairAttempt] = []
        start_time = time.time()
        current_output = initial_output
        last_error: ValidationError | None = None

        # First, try validating the initial output
        try:
            result = self.validator.validate(current_output)
            return RepairResult(
                success=True,
                result=result,
                attempts=[],
                total_duration_ms=(time.time() - start_time) * 1000,
            )
        except ValidationError as e:
            last_error = e

        schema = self.validator.get_json_schema()

        # Repair loop
        for attempt in range(self.config.max_retries):
            attempt_start = time.time()

            # Apply delay before retry (except first attempt)
            if attempt > 0:
                delay = self.calculate_delay(attempt - 1)
                await asyncio.sleep(delay)

            # Build repair prompt
            repair_prompt = self.build_repair_prompt(original_prompt, last_error, schema)

            # Get new output from LLM
            try:
                current_output = await complete_fn(repair_prompt)
                result = self.validator.validate(current_output)

                # Success!
                attempt_record = RepairAttempt(
                    attempt_number=attempt + 1,
                    raw_output=current_output,
                    error=None,
                    duration_ms=(time.time() - attempt_start) * 1000,
                )
                attempts.append(attempt_record)

                if self.on_repair_attempt:
                    self.on_repair_attempt(attempt_record)

                return RepairResult(
                    success=True,
                    result=result,
                    attempts=attempts,
                    total_duration_ms=(time.time() - start_time) * 1000,
                )

            except ValidationError as e:
                last_error = e
                attempt_record = RepairAttempt(
                    attempt_number=attempt + 1,
                    raw_output=current_output,
                    error=e,
                    duration_ms=(time.time() - attempt_start) * 1000,
                )
                attempts.append(attempt_record)

                if self.on_repair_attempt:
                    self.on_repair_attempt(attempt_record)

        # All retries exhausted
        raise RepairExhaustedError(
            f"Failed to get valid output after {self.config.max_retries} repair attempts",
            attempts=len(attempts),
            last_error=last_error,  # type: ignore
        )
