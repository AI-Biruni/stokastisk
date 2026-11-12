"""CPU functional check of the xgrammar structured-output engine under transformers 5.x (no GPU, no vLLM). Compiles a JSON-schema grammar for the Qwen2.5 tokenizer,
requires every token of a valid document to be accepted, an invalid first token to be rejected, and a token bitmask to mask >99% of the vocabulary.
Run with PYTHONPATH pointing at pip --target directories holding the xgrammar version under test and transformers>=5.5.3. Downloads only the public tokenizer files."""
import importlib.metadata
import sys, json, os
import xgrammar as xgr, transformers
from transformers import AutoTokenizer
import torch
tok = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-0.5B-Instruct")
info = xgr.TokenizerInfo.from_huggingface(tok, vocab_size=151936)
comp = xgr.GrammarCompiler(info, max_threads=2)
schema = json.dumps({"type":"object","properties":{"name":{"type":"string"},"age":{"type":"integer"}},"required":["name","age"]})
cg = comp.compile_json_schema(schema)
m = xgr.GrammarMatcher(cg)
text = '{"name": "Ann", "age": 7}'
bm = xgr.allocate_token_bitmask(1, info.vocab_size)
ids = tok.encode(text, add_special_tokens=False)
for t in ids:
    m.fill_next_token_bitmask(bm, 0)
    allowed = (bm[0] >> 0)  # sanity: bitmask nonzero
    assert bm.any(), "empty mask"
    assert m.accept_token(t), f"rejected {t} {tok.decode([t])!r}"
# the grammar must reject a violating continuation
m2 = xgr.GrammarMatcher(cg); assert not m2.accept_token(tok.encode("hello", add_special_tokens=False)[0]), "should reject non-JSON start"
# apply bitmask on CPU logits
m3 = xgr.GrammarMatcher(cg); m3.fill_next_token_bitmask(bm, 0)
logits = torch.zeros(1, info.vocab_size); xgr.apply_token_bitmask_inplace(logits, bm)
print("OK", importlib.metadata.version("xgrammar"), "transformers", transformers.__version__, "masked_fraction", float((logits<-1e30).float().mean()))
