"""Schema adapters for different schema types."""

from __future__ import annotations

import json
from abc import ABC, abstractmethod
from dataclasses import fields as dataclass_fields
from dataclasses import is_dataclass
from typing import Any, TypeVar, Union, get_args, get_origin, get_type_hints

from promptguard.core.exceptions import SchemaError

T = TypeVar("T")


class SchemaAdapter(ABC):
    """Abstract base class for schema adapters.

    Schema adapters provide a unified interface for working with different
    schema types (Pydantic, TypedDict, dataclass, JSON Schema).

    Each adapter can:
    - Convert the schema to JSON Schema format
    - Validate data against the schema
    - Return detailed validation errors
    """

    @abstractmethod
    def to_json_schema(self) -> dict[str, Any]:
        """Convert the schema to JSON Schema format.

        Returns:
            JSON Schema dictionary.
        """
        ...

    @abstractmethod
    def validate(self, data: Any) -> Any:
        """Validate and parse data against the schema.

        Args:
            data: Data to validate (dict or JSON string).

        Returns:
            Validated and parsed data.

        Raises:
            ValidationError: If validation fails.
        """
        ...

    @abstractmethod
    def get_validation_errors(self, data: Any) -> list[dict[str, Any]]:
        """Get detailed validation errors for data.

        Args:
            data: Data to validate.

        Returns:
            List of error dictionaries with details.
        """
        ...

    def get_schema_name(self) -> str:
        """Get a name for the schema."""
        return "schema"


class PydanticAdapter(SchemaAdapter):
    """Adapter for Pydantic BaseModel schemas.

    This is the primary and most feature-complete adapter.

    Example:
        from pydantic import BaseModel

        class Person(BaseModel):
            name: str
            age: int

        adapter = PydanticAdapter(Person)
        result = adapter.validate({"name": "John", "age": 30})
    """

    def __init__(self, model: type[Any]) -> None:
        """Initialize with a Pydantic model.

        Args:
            model: Pydantic BaseModel class.

        Raises:
            SchemaError: If not a valid Pydantic model.
        """
        try:
            from pydantic import BaseModel
        except ImportError:
            raise SchemaError(
                "Pydantic is required for PydanticAdapter",
                schema_type="pydantic",
            )

        if not isinstance(model, type) or not issubclass(model, BaseModel):
            raise SchemaError(
                f"Expected Pydantic BaseModel, got {type(model)}",
                schema_type=str(type(model)),
            )

        self.model = model

    def to_json_schema(self) -> dict[str, Any]:
        """Convert Pydantic model to JSON Schema."""
        return self.model.model_json_schema()

    def validate(self, data: Any) -> Any:
        """Validate data and return Pydantic model instance."""
        if isinstance(data, str):
            data = json.loads(data)
        return self.model.model_validate(data)

    def get_validation_errors(self, data: Any) -> list[dict[str, Any]]:
        """Get Pydantic validation errors."""
        from pydantic import ValidationError

        try:
            if isinstance(data, str):
                data = json.loads(data)
            self.model.model_validate(data)
            return []
        except json.JSONDecodeError as e:
            return [{"type": "json_decode_error", "msg": str(e), "loc": []}]
        except ValidationError as e:
            return [
                {
                    "type": err["type"],
                    "loc": list(err["loc"]),
                    "msg": err["msg"],
                    "input": err.get("input"),
                }
                for err in e.errors()
            ]

    def get_schema_name(self) -> str:
        """Get the model name."""
        return self.model.__name__


class TypedDictAdapter(SchemaAdapter):
    """Adapter for TypedDict schemas.

    Example:
        from typing import TypedDict

        class Person(TypedDict):
            name: str
            age: int

        adapter = TypedDictAdapter(Person)
    """

    def __init__(self, typed_dict: type[Any]) -> None:
        """Initialize with a TypedDict class.

        Args:
            typed_dict: TypedDict class.

        Raises:
            SchemaError: If not a valid TypedDict.
        """
        # Check if it's a TypedDict
        if not (hasattr(typed_dict, "__annotations__") and hasattr(typed_dict, "__total__")):
            raise SchemaError(
                f"Expected TypedDict, got {type(typed_dict)}",
                schema_type=str(type(typed_dict)),
            )

        self.typed_dict = typed_dict
        self._hints = get_type_hints(typed_dict)
        self._required = getattr(typed_dict, "__required_keys__", set(self._hints.keys()))

    def to_json_schema(self) -> dict[str, Any]:
        """Convert TypedDict to JSON Schema."""
        properties: dict[str, Any] = {}

        for name, type_hint in self._hints.items():
            properties[name] = self._type_to_schema(type_hint)

        return {
            "type": "object",
            "title": self.typed_dict.__name__,
            "properties": properties,
            "required": list(self._required),
            "additionalProperties": False,
        }

    def _type_to_schema(self, type_hint: type[Any]) -> dict[str, Any]:
        """Convert a Python type hint to JSON Schema."""
        origin = get_origin(type_hint)
        args = get_args(type_hint)

        if type_hint is str:
            return {"type": "string"}
        elif type_hint is int:
            return {"type": "integer"}
        elif type_hint is float:
            return {"type": "number"}
        elif type_hint is bool:
            return {"type": "boolean"}
        elif type_hint is type(None):
            return {"type": "null"}
        elif origin is list:
            items_schema = self._type_to_schema(args[0]) if args else {}
            return {"type": "array", "items": items_schema}
        elif origin is dict:
            return {"type": "object"}
        elif origin is Union:
            # Handle Optional (Union with None)
            non_none_args = [a for a in args if a is not type(None)]
            if len(non_none_args) == 1 and len(args) == 2:
                # This is Optional[T]
                base_schema = self._type_to_schema(non_none_args[0])
                return {"anyOf": [base_schema, {"type": "null"}]}
            else:
                return {"anyOf": [self._type_to_schema(a) for a in args]}
        else:
            return {}

    def validate(self, data: Any) -> dict[str, Any]:
        """Validate data against TypedDict schema."""
        if isinstance(data, str):
            data = json.loads(data)

        if not isinstance(data, dict):
            raise ValueError(f"Expected dict, got {type(data)}")

        # Check required fields
        for field in self._required:
            if field not in data:
                raise ValueError(f"Missing required field: {field}")

        return data

    def get_validation_errors(self, data: Any) -> list[dict[str, Any]]:
        """Get validation errors for TypedDict."""
        errors: list[dict[str, Any]] = []

        try:
            if isinstance(data, str):
                data = json.loads(data)
        except json.JSONDecodeError as e:
            return [{"type": "json_decode_error", "msg": str(e), "loc": []}]

        if not isinstance(data, dict):
            return [{"type": "type_error", "msg": f"Expected dict, got {type(data)}", "loc": []}]

        # Check required fields
        for field in self._required:
            if field not in data:
                errors.append(
                    {
                        "type": "missing",
                        "loc": [field],
                        "msg": "Field required",
                    }
                )

        # Check for extra fields
        allowed = set(self._hints.keys())
        for key in data.keys():
            if key not in allowed:
                errors.append(
                    {
                        "type": "extra_forbidden",
                        "loc": [key],
                        "msg": "Extra inputs are not permitted",
                    }
                )

        return errors

    def get_schema_name(self) -> str:
        """Get the TypedDict name."""
        return self.typed_dict.__name__


class DataclassAdapter(SchemaAdapter):
    """Adapter for dataclass schemas.

    Example:
        from dataclasses import dataclass

        @dataclass
        class Person:
            name: str
            age: int

        adapter = DataclassAdapter(Person)
    """

    def __init__(self, dataclass_type: type[Any]) -> None:
        """Initialize with a dataclass.

        Args:
            dataclass_type: Dataclass class.

        Raises:
            SchemaError: If not a valid dataclass.
        """
        if not is_dataclass(dataclass_type):
            raise SchemaError(
                f"Expected dataclass, got {type(dataclass_type)}",
                schema_type=str(type(dataclass_type)),
            )

        self.dataclass_type = dataclass_type
        self._fields = dataclass_fields(dataclass_type)

    def to_json_schema(self) -> dict[str, Any]:
        """Convert dataclass to JSON Schema."""
        properties: dict[str, Any] = {}
        required: list[str] = []

        for field in self._fields:
            properties[field.name] = self._type_to_schema(field.type)

            # Check if field has default or default_factory
            has_default = (
                field.default is not field.default_factory  # type: ignore
                or field.default_factory is not field.default_factory  # type: ignore
            )
            if not has_default:
                required.append(field.name)

        return {
            "type": "object",
            "title": self.dataclass_type.__name__,
            "properties": properties,
            "required": required,
        }

    def _type_to_schema(self, type_hint: Any) -> dict[str, Any]:
        """Convert a Python type hint to JSON Schema."""
        # Handle string annotations
        if isinstance(type_hint, str):
            type_hint = eval(type_hint)  # noqa: S307

        origin = get_origin(type_hint)
        args = get_args(type_hint)

        if type_hint is str:
            return {"type": "string"}
        elif type_hint is int:
            return {"type": "integer"}
        elif type_hint is float:
            return {"type": "number"}
        elif type_hint is bool:
            return {"type": "boolean"}
        elif origin is list:
            items_schema = self._type_to_schema(args[0]) if args else {}
            return {"type": "array", "items": items_schema}
        elif origin is dict:
            return {"type": "object"}
        elif origin is Union:
            non_none_args = [a for a in args if a is not type(None)]
            if len(non_none_args) == 1 and len(args) == 2:
                base_schema = self._type_to_schema(non_none_args[0])
                return {"anyOf": [base_schema, {"type": "null"}]}
            else:
                return {"anyOf": [self._type_to_schema(a) for a in args]}
        else:
            return {}

    def validate(self, data: Any) -> Any:
        """Validate data and return dataclass instance."""
        if isinstance(data, str):
            data = json.loads(data)

        return self.dataclass_type(**data)

    def get_validation_errors(self, data: Any) -> list[dict[str, Any]]:
        """Get validation errors for dataclass."""
        errors: list[dict[str, Any]] = []

        try:
            if isinstance(data, str):
                data = json.loads(data)
        except json.JSONDecodeError as e:
            return [{"type": "json_decode_error", "msg": str(e), "loc": []}]

        if not isinstance(data, dict):
            return [{"type": "type_error", "msg": f"Expected dict, got {type(data)}", "loc": []}]

        try:
            self.dataclass_type(**data)
        except TypeError as e:
            errors.append(
                {
                    "type": "type_error",
                    "msg": str(e),
                    "loc": [],
                }
            )

        return errors

    def get_schema_name(self) -> str:
        """Get the dataclass name."""
        return self.dataclass_type.__name__


class JSONSchemaAdapter(SchemaAdapter):
    """Adapter for raw JSON Schema.

    Example:
        schema = {
            "type": "object",
            "properties": {
                "name": {"type": "string"},
                "age": {"type": "integer"}
            },
            "required": ["name", "age"]
        }
        adapter = JSONSchemaAdapter(schema)
    """

    def __init__(self, schema: dict[str, Any]) -> None:
        """Initialize with a JSON Schema.

        Args:
            schema: JSON Schema dictionary.
        """
        if not isinstance(schema, dict):
            raise SchemaError(
                f"Expected dict for JSON Schema, got {type(schema)}",
                schema_type="dict",
            )

        self.schema = schema

    def to_json_schema(self) -> dict[str, Any]:
        """Return the JSON Schema as-is."""
        return self.schema

    def validate(self, data: Any) -> Any:
        """Validate data against JSON Schema."""
        if isinstance(data, str):
            data = json.loads(data)

        try:
            import jsonschema

            jsonschema.validate(data, self.schema)
        except ImportError:
            # Fall back to basic validation if jsonschema not installed
            pass

        return data

    def get_validation_errors(self, data: Any) -> list[dict[str, Any]]:
        """Get validation errors using jsonschema library."""
        try:
            if isinstance(data, str):
                data = json.loads(data)
        except json.JSONDecodeError as e:
            return [{"type": "json_decode_error", "msg": str(e), "loc": []}]

        try:
            import jsonschema

            validator = jsonschema.Draft7Validator(self.schema)
            errors = list(validator.iter_errors(data))
            return [
                {
                    "type": "schema_error",
                    "loc": list(err.absolute_path),
                    "msg": err.message,
                }
                for err in errors
            ]
        except ImportError:
            # Can't validate without jsonschema
            return []

    def get_schema_name(self) -> str:
        """Get the schema title if available."""
        return self.schema.get("title", "schema")


def create_adapter(schema: Any) -> SchemaAdapter:
    """Factory function to create the appropriate adapter for a schema.

    Automatically detects the schema type and returns the appropriate adapter.

    Args:
        schema: Schema to create adapter for. Can be:
            - Pydantic BaseModel class
            - TypedDict class
            - Dataclass class
            - JSON Schema dict

    Returns:
        Appropriate SchemaAdapter instance.

    Raises:
        SchemaError: If schema type is not supported.

    Example:
        from pydantic import BaseModel

        class Person(BaseModel):
            name: str

        adapter = create_adapter(Person)
    """
    # Check for dict (JSON Schema)
    if isinstance(schema, dict):
        return JSONSchemaAdapter(schema)

    # Check for type
    if not isinstance(schema, type):
        raise SchemaError(
            f"Unsupported schema type: {type(schema)}. "
            "Expected Pydantic model, TypedDict, dataclass, or JSON Schema dict.",
            schema_type=str(type(schema)),
        )

    # Check for Pydantic BaseModel
    try:
        from pydantic import BaseModel

        if issubclass(schema, BaseModel):
            return PydanticAdapter(schema)
    except ImportError:
        pass

    # Check for dataclass
    if is_dataclass(schema):
        return DataclassAdapter(schema)

    # Check for TypedDict
    if hasattr(schema, "__annotations__") and hasattr(schema, "__total__"):
        return TypedDictAdapter(schema)

    raise SchemaError(
        f"Unsupported schema type: {schema}. "
        "Expected Pydantic model, TypedDict, dataclass, or JSON Schema dict.",
        schema_type=str(schema),
    )
