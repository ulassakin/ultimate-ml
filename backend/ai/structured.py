"""Convert Pydantic schemas to the strict JSON-Schema subset used by Responses."""
from copy import deepcopy


def strict_response_schema(model) -> dict:
    schema = deepcopy(model.model_json_schema())
    _make_strict(schema)
    return schema


def _make_strict(node):
    if isinstance(node, list):
        for item in node:
            _make_strict(item)
        return
    if not isinstance(node, dict):
        return
    properties = node.get("properties")
    if isinstance(properties, dict):
        # Strict Structured Outputs requires each object to reject unknown keys
        # and name every property as required. Nullable fields represent optional
        # application values while keeping the provider schema deterministic.
        node["additionalProperties"] = False
        node["required"] = list(properties)
        for value in properties.values():
            value.pop("default", None)
            _make_strict(value)
    for key, value in node.items():
        if key != "properties":
            _make_strict(value)
