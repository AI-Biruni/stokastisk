"""Stage 13 (0.4.0), run by v040_12_acceptance.py."""
import sys
from _common import check, finish, run_guarded
def main():
    import stokastisk
    R = []
    cfg = stokastisk.StokastiskConfig(tensor_parallel_size=1)
    check(R, "0.3.0 constructor call valid; defaults unchanged", (cfg.dtype, cfg.attention_backend, cfg.strategy, cfg.pipeline_parallel_size) == ("half", "TRITON_ATTN", "single_gpu", 1))
    check(R, "engine_kwargs identical to 0.3.0 (no pipeline kwarg at PP=1)", "pipeline_parallel_size" not in cfg.engine_kwargs())
    llm = stokastisk.LLM("Qwen/Qwen2.5-0.5B-Instruct", config=cfg)
    try:
        t = llm.complete("The capital of France is", max_tokens=16)
        check(R, "TP=1 generation works through the unchanged 0.3.0 API", bool(t.strip()), t)
        check(R, "TRITON_ATTN", llm.verify_attention_backend() == "TRITON_ATTN")
    finally:
        llm.close()
    finish("13_api_compat", R); return 0
if __name__ == "__main__":
    import multiprocessing; multiprocessing.freeze_support(); run_guarded("13_api_compat", main)
