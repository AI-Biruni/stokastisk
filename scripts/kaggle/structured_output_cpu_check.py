#!/usr/bin/env python3.13
"""CPU check of vLLM's structured-output stack with the INSTALLED xgrammar (no GPU, no engine).

It drives vLLM's own `XgrammarBackend` / `XgrammarGrammar` (vllm/v1/structured_output/backend_xgrammar.py) through a simulated constrained decode:
  random-but-seeded logits -> vLLM's bitmask allocation/fill -> xgrammar `apply_token_bitmask_inplace` -> sample -> `accept_tokens`
and validates that the produced text is JSON that satisfies the schema, that tokens outside the grammar are rejected, and that rollback / reset / termination work.
It ALSO verifies the import-time contract that broke Kaggle build 2: `SamplingParams` structured-output validation imports `backend_lm_format_enforcer`, whose
dataclass annotation evaluates `lmformatenforcer.TokenEnforcer` at import time, so `lmformatenforcer` must be importable for ANY structured-output request.

This is NOT a GPU test: it proves xgrammar <-> vLLM API compatibility and grammar correctness only. Real structured generation is verified by Kaggle notebook 15.
usage: structured_output_cpu_check.py [MODEL_ID] [--json OUT]       (needs the tokenizer files of MODEL_ID: network or HF cache)"""
from __future__ import annotations

import importlib.metadata as md
import json
import random
import sys
import types

SCHEMA = {"type": "object", "properties": {"city": {"type": "string"}, "population": {"type": "integer"}, "tags": {"type": "array", "items": {"type": "string"}, "maxItems": 3}},
          "required": ["city", "population"], "additionalProperties": False}


def conforms(obj) -> bool:
    return (isinstance(obj, dict) and isinstance(obj.get("city"), str) and isinstance(obj.get("population"), int) and not isinstance(obj.get("population"), bool)
            and set(obj) <= {"city", "population", "tags"} and (("tags" not in obj) or (isinstance(obj["tags"], list) and len(obj["tags"]) <= 3 and all(isinstance(t, str) for t in obj["tags"]))))


def main(argv: list[str]) -> int:
    out = None
    if "--json" in argv:
        i = argv.index("--json"); out = argv[i + 1]; del argv[i:i + 2]
    model = argv[0] if argv else "Qwen/Qwen2.5-0.5B-Instruct"
    results: list[dict] = []

    def rec(name, ok, detail=""):
        results.append({"check": name, "status": "PASS" if ok else "FAIL", "detail": str(detail)})
        print(("PASS " if ok else "FAIL ") + name + (f" -- {detail}" if detail else ""))
        return ok

    import torch
    import xgrammar as xgr
    from transformers import AutoTokenizer
    versions = {"xgrammar": md.version("xgrammar"), "apache-tvm-ffi": md.version("apache-tvm-ffi"), "transformers": md.version("transformers"), "torch": torch.__version__}
    print("versions:", versions)

    # 1. the import-time contract that failed on Kaggle
    try:
        import importlib
        importlib.import_module("vllm.v1.structured_output.backend_lm_format_enforcer")
        rec("vllm.v1.structured_output.backend_lm_format_enforcer imports (lmformatenforcer is installed)", True, md.version("lm-format-enforcer"))
    except ModuleNotFoundError as e:
        rec("vllm.v1.structured_output.backend_lm_format_enforcer imports (lmformatenforcer is installed)", False, e)

    from vllm.v1.structured_output.backend_types import StructuredOutputOptions
    from vllm.v1.structured_output.backend_xgrammar import XgrammarBackend, has_xgrammar_unsupported_json_features
    tok = AutoTokenizer.from_pretrained(model)
    vocab = 151936 if "Qwen2.5" in model else len(tok)
    cfg = types.SimpleNamespace(structured_outputs_config=types.SimpleNamespace(disable_any_whitespace=False), speculative_config=None)
    backend = XgrammarBackend(vllm_config=cfg, tokenizer=tok, vocab_size=vocab)
    rec("XgrammarBackend constructs", True, f"vocab_size={vocab}")
    rec("schema has no xgrammar-unsupported feature", not has_xgrammar_unsupported_json_features(SCHEMA))
    grammar = backend.compile_grammar(StructuredOutputOptions.JSON, json.dumps(SCHEMA))
    bitmask = backend.allocate_token_bitmask(1)
    rec("allocate_token_bitmask shape", tuple(bitmask.shape)[0] == 1, tuple(bitmask.shape))

    # 2. simulated constrained decode (the sampler path: bitmask applied to logits). The mock "model" WANTS to emit `want`; the grammar must let a schema-valid
    #    answer through unchanged, and must mask every token that would violate it.
    rng = random.Random(1234)
    want = tok.encode('{"city": "Paris", "population": 2102650, "tags": ["capital", "france"]}', add_special_tokens=False)
    ids: list[int] = []
    for step in range(len(want) + 8):
        grammar.fill_bitmask(bitmask, 0)
        logits = torch.tensor([rng.random() for _ in range(vocab)]).unsqueeze(0)
        if step < len(want):
            logits[0, want[step]] += 10.0                   # the model's preference
        xgr.apply_token_bitmask_inplace(logits, bitmask)
        t = int(torch.argmax(logits, dim=-1))
        if not grammar.accept_tokens("cpu-check", [t]):
            rec("sampled token accepted by the grammar", False, t); break
        ids.append(t)
        if grammar.is_terminated():
            break
    text = tok.decode(ids, skip_special_tokens=True)
    try:
        obj = json.loads(text)
        rec("constrained decode produced valid JSON", True, text[:100])
        rec("decoded JSON satisfies the schema", conforms(obj), obj)
    except json.JSONDecodeError as e:
        rec("constrained decode produced valid JSON", False, f"{e}: {text[:100]!r}")
    rec("grammar terminated after the closing brace", grammar.is_terminated())
    # a model that prefers prose must be masked out: the first allowed tokens may not include 'Paris'
    g_mask = backend.compile_grammar(StructuredOutputOptions.JSON, json.dumps(SCHEMA)); bm = backend.allocate_token_bitmask(1); g_mask.fill_bitmask(bm, 0)
    prose = tok.encode("Paris", add_special_tokens=False)[0]
    brace = tok.encode("{", add_special_tokens=False)[0]
    allowed = lambda t: bool((bm[0, t // 32] >> (t % 32)) & 1)  # noqa: E731
    rec("bitmask excludes a prose token at the start of the object", not allowed(prose), prose)
    rec("bitmask allows '{' at the start of the object", allowed(brace), brace)

    # 3. rejection of out-of-grammar output (invalid-output detection)
    g2 = backend.compile_grammar(StructuredOutputOptions.JSON, json.dumps(SCHEMA))
    bad = tok.encode("Paris is nice", add_special_tokens=False)
    rec("validate_tokens rejects non-JSON text", g2.validate_tokens(bad) == [], g2.validate_tokens(bad))
    good = tok.encode('{"city": "Paris", "population": 2}', add_special_tokens=False)
    rec("validate_tokens accepts schema-valid JSON in full", g2.validate_tokens(good) == good)
    wrong_type = tok.encode('{"city": 5', add_special_tokens=False)
    acc = g2.validate_tokens(wrong_type)
    rec("validate_tokens stops at the first schema violation (city must be a string)", len(acc) < len(wrong_type), len(acc))
    rec("accept_tokens refuses a token that violates the grammar", g2.accept_tokens("cpu-check", bad[:1]) is False)

    # 4. rollback / reset
    g3 = backend.compile_grammar(StructuredOutputOptions.JSON, json.dumps(SCHEMA))
    g3.accept_tokens("cpu-check", good[:3]); n = g3.num_processed_tokens
    g3.rollback(1)
    rec("rollback reduces processed token count", g3.num_processed_tokens == n - 1, (n, g3.num_processed_tokens))
    g3.reset()
    rec("reset clears processed token count", g3.num_processed_tokens == 0)

    status = "PASS" if all(r["status"] == "PASS" for r in results) else "FAIL"
    print(f"\nOVERALL: {status}  versions={versions}")
    if out:
        with open(out, "w") as f:
            json.dump({"status": status, "versions": versions, "model": model, "checks": results}, f, indent=1)
    return 0 if status == "PASS" else 1


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
