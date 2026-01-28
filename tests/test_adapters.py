"""Tests for schema adapters."""

from dataclasses import dataclass
from typing import TypedDict

import pytest
from pydantic import BaseModel

from promptguard.core.exceptions import SchemaError
from promptguard.schemas.adapters import (
    DataclassAdapter,
    JSONSchemaAdapter,
    PydanticAdapter,
    TypedDictAdapter,
    create_adapter,
)


class PydanticPerson(BaseModel):
    name: str
    age: int


class TypedDictPerson(TypedDict):
    name: str
    age: int


@dataclass
class DataclassPerson:
    name: str
    age: int


class TestPydanticAdapter:
    """Tests for PydanticAdapter."""

    def test_to_json_schema(self):
        adapter = PydanticAdapter(PydanticPerson)
        schema = adapter.to_json_schema()
        assert schema["type"] == "object"
        assert "name" in schema["properties"]
        assert "age" in schema["properties"]

    def test_validate_dict(self):
        adapter = PydanticAdapter(PydanticPerson)
        result = adapter.validate({"name": "John", "age": 30})
        assert isinstance(result, PydanticPerson)
        assert result.name == "John"
        assert result.age == 30

    def test_validate_json_string(self):
        adapter = PydanticAdapter(PydanticPerson)
        result = adapter.validate('{"name": "Jane", "age": 25}')
        assert result.name == "Jane"

    def test_get_validation_errors(self):
        adapter = PydanticAdapter(PydanticPerson)
        errors = adapter.get_validation_errors({"name": "John"})
        assert len(errors) > 0  # Missing age

    def test_get_schema_name(self):
        adapter = PydanticAdapter(PydanticPerson)
        assert adapter.get_schema_name() == "PydanticPerson"

    def test_invalid_type_raises_error(self):
        with pytest.raises(SchemaError):
            PydanticAdapter(dict)


class TestTypedDictAdapter:
    """Tests for TypedDictAdapter."""

    def test_to_json_schema(self):
        adapter = TypedDictAdapter(TypedDictPerson)
        schema = adapter.to_json_schema()
        assert schema["type"] == "object"
        assert "name" in schema["properties"]

    def test_validate(self):
        adapter = TypedDictAdapter(TypedDictPerson)
        result = adapter.validate({"name": "John", "age": 30})
        assert result["name"] == "John"

    def test_get_validation_errors_missing_field(self):
        adapter = TypedDictAdapter(TypedDictPerson)
        errors = adapter.get_validation_errors({"name": "John"})
        assert len(errors) > 0


class TestDataclassAdapter:
    """Tests for DataclassAdapter."""

    def test_to_json_schema(self):
        adapter = DataclassAdapter(DataclassPerson)
        schema = adapter.to_json_schema()
        assert schema["type"] == "object"
        assert "name" in schema["properties"]

    def test_validate(self):
        adapter = DataclassAdapter(DataclassPerson)
        result = adapter.validate({"name": "John", "age": 30})
        assert isinstance(result, DataclassPerson)
        assert result.name == "John"

    def test_invalid_type_raises_error(self):
        with pytest.raises(SchemaError):
            DataclassAdapter(dict)


class TestJSONSchemaAdapter:
    """Tests for JSONSchemaAdapter."""

    def test_to_json_schema(self):
        schema = {"type": "object", "properties": {"x": {"type": "integer"}}}
        adapter = JSONSchemaAdapter(schema)
        assert adapter.to_json_schema() == schema

    def test_validate(self):
        schema = {"type": "object", "properties": {"x": {"type": "integer"}}}
        adapter = JSONSchemaAdapter(schema)
        result = adapter.validate({"x": 42})
        assert result == {"x": 42}


class TestCreateAdapter:
    """Tests for create_adapter factory function."""

    def test_creates_pydantic_adapter(self):
        adapter = create_adapter(PydanticPerson)
        assert isinstance(adapter, PydanticAdapter)

    def test_creates_dataclass_adapter(self):
        adapter = create_adapter(DataclassPerson)
        assert isinstance(adapter, DataclassAdapter)

    def test_creates_json_schema_adapter(self):
        schema = {"type": "object"}
        adapter = create_adapter(schema)
        assert isinstance(adapter, JSONSchemaAdapter)

    def test_unsupported_type_raises_error(self):
        with pytest.raises(SchemaError):
            create_adapter("invalid")
