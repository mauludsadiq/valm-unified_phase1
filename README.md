# valm-unified (Phase 1)

Phase 1 delivers a working, testable core integration surface:

- `VALMConfig` (Pydantic v2) with strict validation
- `VALMInterface` orchestrating KV memory, parameter memory, verification, registry
- `StateRegistry` (Phase 1 simplified Merkle-like root over ordered leaves)
- Stubs for memory + verification subsystems

## Run tests

```bash
pytest -q
```

## Minimal usage (with mocks)

See `examples/basic_usage.py` for a reference integration sketch.
