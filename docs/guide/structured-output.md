# 4. Structured output

Make the model produce JSON that matches a JSON Schema. The backend is **xgrammar 0.2.3**, pinned for the whole engine; there is no fallback to another backend.

## Generate

```python
schema = {
    "type": "object",
    "required": ["city", "population"],
    "additionalProperties": False,
    "properties": {
        "city": {"type": "string", "maxLength": 40},
        "population": {"type": "integer", "minimum": 0, "maximum": 100000000},
    },
}

with LLM("Qwen/Qwen2.5-0.5B-Instruct", config=StokastiskConfig(max_model_len=2048)) as llm:
    results = llm.generate_structured(["Describe Paris."], schema, max_tokens=96)
    print(results)       # [{'city': 'Paris', 'population': 20000000}]
```

`generate_structured` returns one parsed object per prompt, **already validated** against the schema.

| Raises | When |
|---|---|
| `StructuredOutputError` | an output is not valid JSON, violates the schema, or was cut off at `max_tokens` |
| `EngineError` | the backend rejected the schema or request |
| `TypeError` / `ValueError` | prompts that are not non-empty strings |

## Bound open-ended fields

A grammar keeps the model inside the schema but cannot make it stop. An unbounded integer or string can run to `max_tokens`, and the output is then reported as truncated. Always set `maxLength`, `maximum`, `maxItems`.

## Validation helpers (no GPU needed)

```python
from stokastisk import validate_json_output
from stokastisk.structured import check_schema, json_schema_params

check_schema(schema)                                     # returns the schema or raises StructuredOutputError
obj = validate_json_output('{"city": "Rome", "population": 2800000}', schema)
params = json_schema_params(schema, max_tokens=96)      # vllm SamplingParams for llm.generate(...)
```

`validate_json_output` never returns invalid data: it raises `StructuredOutputError`.

Next: [OpenAI-compatible server](server.md).
