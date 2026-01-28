"""Tests for validation module."""

import json

import pytest
from pydantic import BaseModel

from promptguard.core.exceptions import ValidationError
from promptguard.core.validator import OutputValidator


class Person(BaseModel):
    """Test schema."""
    name: str
    age: int


class TestOutputValidator:
    """Tests for OutputValidator."""

    def test_validate_valid_json(self):
        validator = OutputValidator.for_schema(Person)
        result = validator.validate('{"name": "John", "age": 30}')
        assert result.name == "John"
        assert result.age == 30

    def test_validate_json_in_code_block(self):
        validator = OutputValidator.for_schema(Person)
        result = validator.validate('```json\n{"name": "Jane", "age": 25}\n```')
        assert result.name == "Jane"
        assert result.age == 25

    def test_validate_json_with_surrounding_text(self):
        validator = OutputValidator.for_schema(Person)
        result = validator.validate('Here is the data: {"name": "Bob", "age": 40}')
        assert result.name == "Bob"
        assert result.age == 40

    def test_validate_invalid_json(self):
        validator = OutputValidator.for_schema(Person)
        with pytest.raises(ValidationError) as exc_info:
            validator.validate('{"name": "John", "age":}')
        assert "Invalid JSON" in str(exc_info.value)

    def test_validate_missing_field(self):
        validator = OutputValidator.for_schema(Person)
        with pytest.raises(ValidationError) as exc_info:
            validator.validate('{"name": "John"}')
        assert len(exc_info.value.errors) > 0

    def test_validate_wrong_type(self):
        validator = OutputValidator.for_schema(Person)
        with pytest.raises(ValidationError) as exc_info:
            validator.validate('{"name": "John", "age": "thirty"}')
        assert len(exc_info.value.errors) > 0

    def test_extract_json(self):
        validator = OutputValidator.for_schema(Person)

        # Plain JSON
        assert validator.extract_json('{"a": 1}') == '{"a": 1}'

        # In code block
        assert validator.extract_json('```json\n{"a": 1}\n```') == '{"a": 1}'

        # In code block without json tag
        assert validator.extract_json('```\n{"a": 1}\n```') == '{"a": 1}'

        # With surrounding text
        extracted = validator.extract_json('Result: {"a": 1} end')
        assert json.loads(extracted) == {"a": 1}

    def test_format_errors_for_repair(self):
        validator = OutputValidator.for_schema(Person)
        try:
            validator.validate('{"name": "John"}')
        except ValidationError as e:
            formatted = validator.format_errors_for_repair(e)
            assert "validation errors" in formatted
            assert "age" in formatted.lower() or "required" in formatted.lower()

    def test_get_json_schema(self):
        validator = OutputValidator.for_schema(Person)
        schema = validator.get_json_schema()
        assert schema["type"] == "object"
        assert "properties" in schema
        assert "name" in schema["properties"]
        assert "age" in schema["properties"]
