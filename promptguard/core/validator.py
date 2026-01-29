"""Output validation for LLM responses."""

from __future__ import annotations

import json
import re
from typing import Any, TypeVar

from promptguard.core.exceptions import ValidationError
from promptguard.core.logging import get_logger
from promptguard.schemas.adapters import SchemaAdapter, create_adapter

logger = get_logger(__name__)

T = TypeVar("T")


class OutputValidator:
    """Validates LLM output against schemas.

    The validator handles:
    - JSON extraction from markdown code blocks
    - Schema validation using the appropriate adapter
    - Error formatting for repair prompts

    Example:
        from pydantic import BaseModel

        class Person(BaseModel):
            name: str
            age: int

        validator = OutputValidator.for_schema(Person)
        result = validator.validate('{"name": "John", "age": 30}')
    """

    def __init__(self, adapter: SchemaAdapter) -> None:
        """Initialize with a schema adapter.

        Args:
            adapter: SchemaAdapter instance for validation.
        """
        self.adapter = adapter

    @classmethod
    def for_schema(cls, schema: Any) -> OutputValidator:
        """Create a validator for any supported schema type.

        Args:
            schema: Pydantic model, TypedDict, dataclass, or JSON Schema dict.

        Returns:
            OutputValidator instance.

        Example:
            validator = OutputValidator.for_schema(MyPydanticModel)
        """
        return cls(create_adapter(schema))

    def extract_json(self, text: str) -> str:
        """Extract JSON from LLM response text.

        Handles common LLM output patterns:
        - Raw JSON
        - JSON in markdown code blocks (```json ... ```)
        - JSON embedded in explanatory text

        Args:
            text: Raw text from LLM response.

        Returns:
            Extracted JSON string.

        Example:
            >>> validator.extract_json('```json\\n{"name": "John"}\\n```')
            '{"name": "John"}'
        """
        text = text.strip()

        # Try to find JSON in markdown code blocks
        code_block_patterns = [
            r"```json\s*([\s\S]*?)```",  # ```json ... ```
            r"```\s*([\s\S]*?)```",  # ``` ... ```
        ]

        for pattern in code_block_patterns:
            matches = re.findall(pattern, text, re.IGNORECASE)
            if matches:
                # Return the first match that looks like JSON
                for match in matches:
                    match = match.strip()
                    if match.startswith(("{", "[")):
                        return match

        # Try to find a JSON object or array directly
        # Look for balanced braces/brackets
        json_pattern = r"(\{[\s\S]*\}|\[[\s\S]*\])"
        match = re.search(json_pattern, text)
        if match:
            potential_json = match.group(1)
            # Verify it's valid JSON by attempting to parse
            try:
                json.loads(potential_json)
                return potential_json
            except json.JSONDecodeError:
                pass

        # If nothing worked, return the original text stripped
        return text

    def validate(self, raw_output: str) -> Any:
        """Validate LLM output and return parsed result.

        Args:
            raw_output: Raw output string from LLM.

        Returns:
            Validated and parsed data (type depends on schema).

        Raises:
            ValidationError: If output fails validation.

        Example:
            try:
                result = validator.validate(llm_response)
            except ValidationError as e:
                print(f"Validation failed: {e.errors}")
        """
        # Extract JSON from the output
        json_str = self.extract_json(raw_output)

        # Try to parse JSON
        try:
            data = json.loads(json_str)
        except json.JSONDecodeError as e:
            logger.warning("JSON decode error: %s", e)
            raise ValidationError(
                f"Invalid JSON in LLM output: {e}",
                raw_output=raw_output,
                errors=[
                    {
                        "type": "json_decode_error",
                        "msg": str(e),
                        "loc": [],
                        "pos": e.pos,
                    }
                ],
            )

        # Validate against schema
        errors = self.adapter.get_validation_errors(data)
        if errors:
            logger.warning(
                "Schema validation failed with %d error(s)",
                len(errors),
            )
            raise ValidationError(
                f"Schema validation failed with {len(errors)} error(s)",
                raw_output=raw_output,
                errors=errors,
            )

        logger.debug("Validation successful")
        # Return validated/parsed data
        return self.adapter.validate(data)

    def get_json_schema(self) -> dict[str, Any]:
        """Get the JSON Schema for the output format.

        Returns:
            JSON Schema dictionary.
        """
        return self.adapter.to_json_schema()

    def format_errors_for_repair(self, error: ValidationError) -> str:
        """Format validation errors for use in a repair prompt.

        Creates a human-readable error description that can be included
        in a prompt to help the LLM correct its output.

        Args:
            error: ValidationError with error details.

        Returns:
            Formatted error string for repair prompt.

        Example:
            >>> print(validator.format_errors_for_repair(error))
            The previous output had the following validation errors:
            1. Field 'age': Input should be a valid integer
        """
        lines = ["The previous output had the following validation errors:"]

        for i, err in enumerate(error.errors, 1):
            loc = err.get("loc", [])
            msg = err.get("msg", str(err))

            if loc:
                # Format the path to the field
                path = ".".join(str(p) for p in loc)
                lines.append(f"{i}. Field '{path}': {msg}")
            else:
                lines.append(f"{i}. {msg}")

        return "\n".join(lines)

    def format_schema_for_prompt(self) -> str:
        """Format the JSON schema for inclusion in a prompt.

        Returns:
            JSON schema as a formatted string.
        """
        schema = self.get_json_schema()
        return json.dumps(schema, indent=2)


def validate_output(
    raw_output: str,
    schema: Any,
) -> Any:
    """Convenience function to validate output against a schema.

    Args:
        raw_output: Raw output string from LLM.
        schema: Pydantic model, TypedDict, dataclass, or JSON Schema dict.

    Returns:
        Validated and parsed data.

    Raises:
        ValidationError: If validation fails.

    Example:
        from pydantic import BaseModel

        class Person(BaseModel):
            name: str
            age: int

        result = validate_output('{"name": "John", "age": 30}', Person)
    """
    validator = OutputValidator.for_schema(schema)
    return validator.validate(raw_output)
