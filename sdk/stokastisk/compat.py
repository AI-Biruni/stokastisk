"""Static Tesla T4 / SM75 / TRITON_ATTN compatibility inspection of the bundled vLLM 0.25.1.

Everything here is derived from SOURCE INSPECTION of the shipped vLLM package and from the owner's executed Kaggle notebooks. Nothing imports vllm, touches a GPU or
needs the network, so it can run (and be unit-tested) anywhere. It never claims that an unexercised feature works:

  status    SUPPORTED | SUPPORTED_WITH_CONDITIONS | NEEDS_ADAPTATION | NOT_SUPPORTED_SM75 | EXCLUDED_FLASH_DEPENDENCY | NOT_IMPLEMENTED | UNSUPPORTED_ENTRY_POINT | UNKNOWN_REQUIRES_EVIDENCE
  evidence  kaggle (executed by the owner on 2 x Tesla T4: the notebook 15 acceptance suite on integrated build 3 [installer 32, static 79, engine 44, parallel 23, server 21, summary 15 checks],
            the end-user notebooks 16, 18 and 19, or the earlier notebooks 01-14 - the ``notebooks`` field of each record says which) |
            kaggle-kernel (Triton kernel compiled and launched, output not asserted) | cpu-tests | static (source inspection only) | implemented-not-executed
  The Kaggle runs used build 3 (acceptance wheel b31588c9...; the published build-3 wheel differs from it only in METADATA/RECORD). Build 4 changes only this module's evidence
  records and the ``serving`` docstring: its inference, server and native code are byte-identical to build 3 and it was NOT itself executed on Kaggle.

Model-specific evidence: ``MODEL_EVIDENCE`` lists the exact Hugging Face model ids that were run, with dtype, quantization, context length and parallelism modes.
An architecture being present in vLLM (or having been run for some other checkpoint) is never evidence for a different checkpoint.

``check_*`` functions raise :class:`StokastiskCompatibilityError` with the reason when something cannot work on SM75, so the user never meets an obscure ImportError.
"""
from __future__ import annotations

import ast
import importlib.util
import functools
import json
import pathlib
from typing import Any, Iterable

from .errors import StokastiskCompatibilityError

SM75 = (7, 5)
BLOCKING = ("NOT_SUPPORTED_SM75", "EXCLUDED_FLASH_DEPENDENCY", "NOT_IMPLEMENTED", "NEEDS_ADAPTATION", "UNSUPPORTED_ENTRY_POINT")


def _f(status: str, evidence: str, note: str, notebooks: str = "") -> dict[str, str]:
    return {"status": status, "evidence": evidence, "note": note, "notebooks": notebooks}


FEATURES: dict[str, dict[str, str]] = {
    # --- executed on Kaggle 2 x T4 -------------------------------------------------------------------------------------------------------------
    "offline_generation": _f("SUPPORTED", "kaggle", "vllm.LLM generate() in FP16, enforce_eager, TRITON_ATTN. Exact models run: Qwen/Qwen2.5-0.5B-Instruct (TP=1; TP=2, PP=2 and 2 replicas in the notebook 15 parallel stage), "
                             "Qwen/Qwen2.5-3B-Instruct (TP=2, PP=2) and Qwen/Qwen2.5-Coder-7B-Instruct (TP=2, PP=2); max_model_len 2048. See MODEL_EVIDENCE; other checkpoints are not covered", "15,16,18,19"),
    "batch_generation": _f("SUPPORTED", "kaggle", "batches of 1..64 requests, 64 output tokens", "09,11,14"),
    "triton_attention": _f("SUPPORTED", "kaggle", "TRITON_ATTN selected and confirmed in the live VllmConfig; kernel_unified_attention compiled; no Flash library mapped in any process", "05,13"),
    "tensor_parallel_2": _f("SUPPORTED", "kaggle", "both GPUs hold a shard; slower than TP=1 for a model that fits one T4 (about 0.55x decode, earlier notebooks); generated with Qwen2.5-0.5B (nb 15), Qwen2.5-3B (nb 18) and Qwen2.5-Coder-7B (nb 19)", "05,06,13,15,18,19"),
    "pipeline_parallel_2": _f("SUPPORTED_WITH_CONDITIONS", "kaggle", "experimental in this SDK; both GPUs hold stages; about 0.6x of TP=1 decode for a model that fits one T4 (earlier notebooks); generated with Qwen2.5-0.5B (nb 15), Qwen2.5-3B (nb 18) and Qwen2.5-Coder-7B (nb 19)", "08,13,15,18,19"),
    "replica_parallel": _f("SUPPORTED_WITH_CONDITIONS", "kaggle", "SDK feature (ReplicaPool): one TP=1 engine per GPU, both receive work; no throughput gain over one engine up to batch 64; also the notebook 15 parallel stage (two workers on two GPUs, both received work, clean close)", "09,14,15"),
    "gpu_telemetry": _f("SUPPORTED", "kaggle", "nvidia-smi sampling during benchmarks", "09,11,14"),
    "topology_discovery": _f("SUPPORTED", "kaggle", "PHB; CUDA peer access measured per session (true in one Kaggle VM, false in another)", "01,07"),
    "nccl_collectives": _f("SUPPORTED", "kaggle", "all_reduce/all_gather/reduce_scatter/broadcast on both ranks", "04"),
    # --- Triton kernels compiled/launched on Kaggle, behaviour not asserted --------------------------------------------------------------------
    "sampling_temperature_top_k_top_p": _f("SUPPORTED_WITH_CONDITIONS", "kaggle-kernel", "kernels compiled during runs; only greedy (temperature 0) output was asserted"),
    "logprobs": _f("SUPPORTED_WITH_CONDITIONS", "kaggle-kernel", "logprob / prompt-logprob kernels compiled; output not asserted"),
    "penalties_min_p_bad_words_logit_bias": _f("SUPPORTED_WITH_CONDITIONS", "kaggle-kernel", "kernels compiled; output not asserted"),
    # --- exist in the source, never exercised ---------------------------------------------------------------------------------------------------
    "chat_generation": _f("SUPPORTED", "kaggle", "LLM.chat and LLM.complete returned text on 2 x T4: notebook 15 engine stage (build 3), the notebook 16 quickstart, notebook 18 (0.5B TP=1; 3B TP=2 and PP=2) and notebook 19 (Coder-7B TP=2 and PP=2)", "15,16,18,19"),
    "openai_server": _f("SUPPORTED_WITH_CONDITIONS", "kaggle", "the real `vllm serve` launched by stokastisk.serving.ServerProcess was started and exercised on 2 x T4 (notebook 15, server stage, 21/21: /health, /v1/models, chat completions, SSE streaming, "
                        "HTTP error mapping, structured output over HTTP, /metrics, restart on the same port, graceful SIGINT shutdown, GPU memory released). Conditions: exercised with Qwen/Qwen2.5-0.5B-Instruct, TP=1 only; Kaggle's fastapi 0.141.1 is newer "
                        "than upstream vLLM's <0.137 cap; the SageMaker /ping and /invocations routes were NOT exercised and are not claimed. Calling upstream `vllm serve` directly is unsupported (see direct_upstream_vllm)", "15"),
    "streaming": _f("SUPPORTED", "kaggle", "stream_chat: SSE through the real server yielded several non-empty deltas and ended with [DONE] (notebook 15, server stage, Qwen2.5-0.5B, TP=1); the truncated-stream error path is CPU-tested against a fake server only", "15"),
    "prometheus_metrics": _f("SUPPORTED_WITH_CONDITIONS", "kaggle", "GET /metrics on the running server exposed vllm: samples (notebook 15, server stage). Only the presence of those samples was asserted, not their values; the fetch_metrics text parser is CPU-tested", "15"),
    "structured_outputs": _f("SUPPORTED_WITH_CONDITIONS", "kaggle", "JSON-schema output with the pinned xgrammar 0.2.3 backend (never 'auto'): notebook 15 engine stage (every generate_structured output satisfied the bounded schema; not-JSON, wrong type, missing key, extra key "
                             "and truncated outputs were each detected; a schema xgrammar does not support was rejected with no fallback; xgrammar's native library was mapped and outlines_core's was not), the server stage (structured output over HTTP) "
                             "and notebooks 16 and 18 (generate_structured). Conditions: bound open-ended schema fields (a grammar cannot make a model stop); acceptance model Qwen2.5-0.5B", "15,16,18"),
    "direct_upstream_vllm": _f("UNSUPPORTED_ENTRY_POINT", "static", "calling vllm.LLM or `vllm serve` directly bypasses the SDK's dtype, NCCL, structured-output and backend checks and is NOT a supported Stokastisk entry point "
                               "(use stokastisk.LLM / ServerProcess / ReplicaPool). The Flash-free tree makes upstream auto-selection resolve to TRITON_ATTN on SM75 (CPU-evaluated), "
                               "and FlashAttention/FlashInfer/FlashMLA are refused with ProhibitedBackendError, but only the SDK paths are verified"),
    "prefix_caching": _f("UNKNOWN_REQUIRES_EVIDENCE", "static", "no run asserts it"),
    "chunked_prefill": _f("UNKNOWN_REQUIRES_EVIDENCE", "static", "no run asserts it"),
    "cuda_graphs": _f("UNKNOWN_REQUIRES_EVIDENCE", "static", "every Kaggle run used enforce_eager=True (the SDK default); graph capture on T4 is unverified"),
    "custom_all_reduce": _f("UNKNOWN_REQUIRES_EVIDENCE", "static", "disabled by the SDK default; needs CUDA peer access, which differs between Kaggle VMs"),
    "lora": _f("UNKNOWN_REQUIRES_EVIDENCE", "static", "present in the source; no run"),
    "multimodal": _f("UNKNOWN_REQUIRES_EVIDENCE", "static", "present in the source; no run"),
    "embedding_pooling_reranking": _f("UNKNOWN_REQUIRES_EVIDENCE", "static", "present in the source; no run"),
    "speculative_decoding": _f("UNKNOWN_REQUIRES_EVIDENCE", "static", "present in the source; no run"),
    "mla_models_deepseek": _f("UNKNOWN_REQUIRES_EVIDENCE", "static", "FlashMLA/FlashInfer-MLA backends are excluded; other MLA backends were not run"),
    "worker_failure_recovery": _f("SUPPORTED_WITH_CONDITIONS", "cpu-tests", "ReplicaPool restarts/retries covered by CPU tests with fake workers only"),
    # --- cannot work on this build -------------------------------------------------------------------------------------------------------------
    "bfloat16": _f("NOT_SUPPORTED_SM75", "static", "vllm/platforms/cuda.py rejects bfloat16 below compute capability 8.0; use dtype='half'"),
    "flash_attention": _f("EXCLUDED_FLASH_DEPENDENCY", "static", "FlashAttention is not shipped and not allowed"),
    "flashinfer": _f("EXCLUDED_FLASH_DEPENDENCY", "static", "FlashInfer attention, sampler and all-reduce fusion are not shipped"),
    "flashmla": _f("EXCLUDED_FLASH_DEPENDENCY", "static", "FlashMLA is not shipped"),
    "allreduce_rms_fusion": _f("EXCLUDED_FLASH_DEPENDENCY", "static", "needs FlashInfer comm; pass_manager raises a clear error if requested"),
    "fp8_mxfp4_nvfp4_native": _f("NOT_SUPPORTED_SM75", "static", "native FP8/FP4 kernels need SM89/SM90/SM100 (see check_quantization for the per-method minimum declared in the source)"),
    "expert_parallel_all2all": _f("NOT_SUPPORTED_SM75", "static", "DeepEP / NIXL / FlashInfer all2all backends are not shipped or need newer GPUs"),
    "qwen_omni_models": _f("NEEDS_ADAPTATION", "static", "qwen2_5_omni_thinker was removed from the T4 wheel by the module-level flash-import rule (its flash_attn import is guarded); restoring it needs a new vLLM wheel"),
    "l4_sm89": _f("NOT_IMPLEMENTED", "static", "planned only; no SM89 binaries, no support claim"),
}

# Per-method minimum compute capability exactly as declared by get_min_capability() in the shipped vllm/model_executor/layers/quantization sources (x10: 75 = SM7.5).
QUANT_MIN_CAPABILITY: dict[str, tuple[int, str]] = {
    "awq": (75, "auto_awq.py"), "auto_awq": (75, "auto_awq.py"), "gptq": (60, "auto_gptq.py"), "auto_gptq": (60, "auto_gptq.py"),
    "bitsandbytes": (70, "bitsandbytes.py"), "compressed-tensors": (70, "compressed_tensors/compressed_tensors.py"), "moe_wna16": (70, "moe_wna16.py"),
    "torchao": (75, "torchao.py"), "humming": (75, "humming.py"), "fp8": (75, "fp8.py"),
    "experts_int8": (80, "experts_int8.py"), "fbgemm_fp8": (80, "fbgemm_fp8.py"), "mxfp4": (80, "mxfp4.py"), "fp_quant": (100, "fp_quant.py"),
}
# Mistral-family checkpoints are NOT supported on this build: they use the mistral_common tokenizer/loader (removed from every dependency profile), are published in bfloat16, and the
# Pixtral/Voxtral/Mistral3 variants add vision/audio towers; none of them was ever run on the T4s. Refused up front instead of failing late with an ImportError.
UNSUPPORTED_ARCHITECTURE_MARKERS = ("Mistral", "Ministral", "Pixtral", "Voxtral", "Mixtral")
# Exact checkpoints that were actually run on 2 x Tesla T4 (build 3). This is the ONLY source of model-level evidence; ``check_model`` never infers it from an architecture.
# Qwen2.5-1.5B-Instruct is not listed: its only recorded runs were on earlier builds whose raw evidence is no longer part of the project.
# Free-memory figures are torch.cuda.mem_get_info() readings taken inside the engine context (notebooks 18/19); every GPU returned to 14.46 of 14.56 GiB free after close().
MODEL_EVIDENCE: dict[str, dict[str, Any]] = {
    "Qwen/Qwen2.5-0.5B-Instruct": {
        "architecture": "Qwen2ForCausalLM", "dtype": "half", "quantization": None, "max_tested_model_len": 2048,
        "tested_modes": ("TP=1", "TP=2", "PP=2", "replicas=2"),
        "evidence": ("notebook 15 (build 3 acceptance): engine 44/44 and server 21/21 at TP=1; parallel 23/23 with TP=2, PP=2 and two replicas", "notebook 16 (quickstart script, TP=1)", "notebook 18 (plain cells, TP=1)"),
        "limitations": ("FP16, eager execution, max_model_len 2048 only", "TP=2, PP=2 and replicas are slower than TP=1 for a model this small"),
    },
    "Qwen/Qwen2.5-3B-Instruct": {
        "architecture": "Qwen2ForCausalLM", "dtype": "half", "quantization": None, "max_tested_model_len": 2048,
        "tested_modes": ("TP=2", "PP=2"),
        "evidence": ("notebook 18 (build 3): completion and chat with TP=2, and with PP=2",),
        "limitations": ("TP=1 was not tested for this model", "free memory during inference: about 1.09 GiB per GPU at TP=2, 1.96 and 1.48 GiB at PP=2 (gpu_memory_utilization 0.88)", "PP=2 is experimental"),
    },
    "Qwen/Qwen2.5-Coder-7B-Instruct": {
        "architecture": "Qwen2ForCausalLM", "dtype": "half", "quantization": None, "max_tested_model_len": 2048,
        "tested_modes": ("TP=2", "PP=2"),
        "evidence": ("notebook 19 (build 3): original unquantized checkpoint (BF16 weights cast to FP16); one Python-coding chat answer with TP=2 and one with PP=2",),
        "limitations": ("does not fit one T4 in FP16, so TP=1 is not an option", "memory is tight: about 1.34 GiB free per GPU during inference at TP=2, 2.42 and 1.93 GiB at PP=2 (max_model_len 2048, gpu_memory_utilization 0.88)",
                        "a single prompt per mode; this is not a throughput measurement and does not show that other 7B models fit", "PP=2 is experimental"),
    },
}
# Architectures that at least one verified checkpoint above has exercised. Evidence about the ARCHITECTURE only: a checkpoint of the same architecture that is not in MODEL_EVIDENCE stays UNKNOWN_REQUIRES_EVIDENCE.
KAGGLE_VERIFIED_ARCHITECTURES = frozenset(r["architecture"] for r in MODEL_EVIDENCE.values())
_VERIFIED_DTYPES = {"half": "half", "float16": "half", "fp16": "half"}


def feature_matrix() -> dict[str, dict[str, str]]:
    """A copy of the feature table (safe to mutate)."""
    return {k: dict(v) for k, v in FEATURES.items()}


def get_feature(name: str) -> dict[str, str]:
    try:
        return dict(FEATURES[name])
    except KeyError:
        raise StokastiskCompatibilityError(f"unknown feature {name!r}; known: {', '.join(sorted(FEATURES))}") from None


def check_feature(name: str) -> dict[str, str]:
    """Return the feature record, or raise if the feature cannot work on this build. UNKNOWN_REQUIRES_EVIDENCE is returned, not raised: it is untested, not forbidden."""
    rec = get_feature(name)
    if rec["status"] in BLOCKING:
        raise StokastiskCompatibilityError(f"{name} is {rec['status']} on the Tesla T4 / SM75 / TRITON_ATTN build: {rec['note']}")
    return rec


def check_dtype(dtype: str | None, capability: tuple[int, int] = SM75) -> str:
    """Reject an explicit bfloat16 below SM80 with the same reasoning as vllm/platforms/cuda.py, but before an engine is started."""
    d = (dtype or "half").lower().replace("torch.", "")
    if d in ("bfloat16", "bf16") and tuple(capability) < (8, 0):
        raise StokastiskCompatibilityError(f"bfloat16 needs compute capability >= 8.0 (this GPU is {capability[0]}.{capability[1]}); use dtype='half'")
    return d      # 'auto' is allowed: it works when the checkpoint is fp16, and vLLM itself raises if the model config resolves to bfloat16


def check_quantization(method: str | None, capability: tuple[int, int] = SM75) -> dict[str, Any]:
    """Compare a vLLM quantization method with the minimum capability declared in the source. Unknown methods are reported UNKNOWN_REQUIRES_EVIDENCE, never assumed to work."""
    if not method:
        return {"method": None, "status": "SUPPORTED", "evidence": "kaggle", "note": "no quantization (all Kaggle runs)"}
    m = method.lower()
    if m not in QUANT_MIN_CAPABILITY:
        return {"method": m, "status": "UNKNOWN_REQUIRES_EVIDENCE", "evidence": "static", "note": "no minimum capability declared in the inspected source (or not inspected); never run on Kaggle"}
    need, src = QUANT_MIN_CAPABILITY[m]
    have = capability[0] * 10 + capability[1]
    if need > have:
        raise StokastiskCompatibilityError(f"quantization {m!r} declares a minimum compute capability of {need / 10:.1f} ({src}); the T4 is {capability[0]}.{capability[1]}")
    return {"method": m, "status": "UNKNOWN_REQUIRES_EVIDENCE", "evidence": "static",
            "note": f"source declares minimum capability {need / 10:.1f} ({src}) which SM7.5 meets; the kernels were never run on Kaggle"}


# ---------------------------------------------------------------- model architecture check (static; parses the installed vllm package without importing it)
def _vllm_root(package_root: str | pathlib.Path | None) -> pathlib.Path | None:
    if package_root is not None:
        return pathlib.Path(package_root)
    spec = importlib.util.find_spec("vllm")
    if spec is None or not spec.submodule_search_locations:
        return None
    return pathlib.Path(next(iter(spec.submodule_search_locations)))


def read_registry(package_root: str | pathlib.Path | None = None) -> dict[str, tuple[str, str]]:
    """architecture name -> (module, class) from vllm/model_executor/models/registry.py, read with ast (no import)."""
    root = _vllm_root(package_root)
    if root is None:
        raise StokastiskCompatibilityError("vllm is not installed; cannot inspect its model registry")
    path = root / "model_executor" / "models" / "registry.py"
    if not path.is_file():
        raise StokastiskCompatibilityError(f"model registry not found at {path}")
    out: dict[str, tuple[str, str]] = {}
    for node in ast.parse(path.read_text()).body:
        tgt = node.target if isinstance(node, ast.AnnAssign) else (node.targets[0] if isinstance(node, ast.Assign) and len(node.targets) == 1 else None)
        val = getattr(node, "value", None)
        if isinstance(tgt, ast.Name) and tgt.id.startswith("_") and tgt.id.endswith("_MODELS") and isinstance(val, ast.Dict):
            for k, v in zip(val.keys, val.values):
                if isinstance(k, ast.Constant) and isinstance(v, ast.Tuple) and len(v.elts) == 2 and all(isinstance(e, ast.Constant) for e in v.elts):
                    out.setdefault(str(k.value), (str(v.elts[0].value), str(v.elts[1].value)))
    return out


def _module_file(root: pathlib.Path, dotted: str) -> pathlib.Path | None:
    parts = dotted.split(".")[1:]
    base = root.joinpath(*parts)
    for cand in (base.with_suffix(".py"), base / "__init__.py"):
        if cand.is_file():
            return cand
    return None


@functools.lru_cache(maxsize=None)      # an import closure re-reads the same few hundred files for every architecture; per-process cache keyed by (path, root)
def _unconditional_vllm_imports(path: pathlib.Path, root: pathlib.Path) -> tuple[str, ...]:
    """vllm.* modules imported by plain module-level statements (not inside try/if/def), i.e. imports that fail at import time if the module is absent."""
    try:
        tree = ast.parse(path.read_text())
    except (OSError, SyntaxError):
        return ()
    rel = path.relative_to(root).with_suffix("")
    pkg = ["vllm", *rel.parts[:-1]]      # containing package; identical for module.py and package/__init__.py
    out: list[str] = []
    for n in tree.body:
        if isinstance(n, ast.Import):
            out += [a.name for a in n.names if a.name.split(".")[0] == "vllm"]
        elif isinstance(n, ast.ImportFrom):
            base = n.module or ""
            if n.level:
                base = ".".join(pkg[: len(pkg) - (n.level - 1)] + ([base] if base else []))
            if base.split(".")[0] == "vllm":
                out.append(base)
    return tuple(out)


STUB_MODULE = "vllm.utils.t4_prohibited_stubs"


def uses_removed_backend(module_file: pathlib.Path, root: pathlib.Path) -> bool:
    """True if the architecture's OWN module imports names from the Flash-family placeholder module at import time, i.e. the model is built on a removed backend.
    (Core modules such as vllm._custom_ops import placeholder predicates too; those are deliberately not inspected.)"""
    return STUB_MODULE in _unconditional_vllm_imports(module_file, root)


def find_missing_imports(module_file: pathlib.Path, root: pathlib.Path, *, max_depth: int = 6) -> list[list[str]]:
    """Chains of unconditional vllm imports from ``module_file`` that end in a module absent from the shipped package."""
    missing: list[list[str]] = []
    seen: set[pathlib.Path] = set()

    def walk(f: pathlib.Path, chain: list[str], depth: int) -> None:
        if f in seen or depth > max_depth:
            return
        seen.add(f)
        for dotted in _unconditional_vllm_imports(f, root):
            target = _module_file(root, dotted)
            if target is None:
                missing.append(chain + [dotted])
            else:
                walk(target, chain + [dotted], depth + 1)
    walk(module_file, [], 0)
    return missing


def _model_record(model: str | None) -> tuple[str, dict[str, Any]] | None:
    """The MODEL_EVIDENCE record for an exact Hugging Face id (case-insensitive, as the Hub treats ids), else None."""
    if not model:
        return None
    key = str(model).strip().casefold()
    for mid, rec in MODEL_EVIDENCE.items():
        if mid.casefold() == key:
            return mid, rec
    return None


def _mode_name(mode: str) -> str:
    return str(mode).replace(" ", "").upper().replace("REPLICAS=", "replicas=")


def check_model(model: str | None = None, *, architectures: Iterable[str] | None = None, package_root: str | pathlib.Path | None = None,
                capability: tuple[int, int] = SM75, dtype: str | None = "half", quantization: str | None = None, mode: str | None = None) -> dict[str, Any]:
    """Static, OFFLINE pre-flight for a model on the T4 build (nothing is downloaded and no GPU is touched).

    ``model`` is a Hugging Face id or a local directory with config.json; ``architectures=[...]`` may be given instead. ``mode`` (e.g. ``"TP=2"``) optionally names the
    parallelism you intend to use. Raises StokastiskCompatibilityError when the architecture's module (or an unconditional import below it) is absent from the shipped wheel,
    for Mistral-family architectures, and when the dtype or the quantization cannot work on SM75.

    Two separate questions are answered:

    * ``architecture_status`` - is the architecture present in the shipped vLLM (statically checked) and has any verified checkpoint exercised it on Kaggle;
    * ``model_status`` - was THIS checkpoint (exact model id, FP16, no quantization) run on Kaggle (``MODEL_EVIDENCE``). ``status`` is ``SUPPORTED`` only when ``model_status`` is
      ``SUPPORTED``; any other checkpoint, even of a Kaggle-exercised architecture, is ``UNKNOWN_REQUIRES_EVIDENCE``.

    A local directory is matched to a verified model through the ``_name_or_path`` entry of its config.json; this identifies the configuration, not the weights.
    Result fields: model, dtype, quantization, architectures, status, model_status, architecture_status, tested_modes, evidence, limitations, max_tested_model_len, mode_status
    (``TESTED`` / ``UNTESTED``, only when ``mode`` is given), note."""
    archs = list(architectures) if architectures is not None else None
    ident = model
    if model and pathlib.Path(model).is_dir() and (pathlib.Path(model) / "config.json").is_file():
        cfg = json.loads((pathlib.Path(model) / "config.json").read_text())
        if archs is None:
            archs = list(cfg.get("architectures") or [])
        if quantization is None:
            quantization = (cfg.get("quantization_config") or {}).get("quant_method")
        ident = cfg.get("_name_or_path") or None
    found = _model_record(ident)
    if archs is None and found is not None:
        archs = [found[1]["architecture"]]
    report: dict[str, Any] = {"model": model, "dtype": check_dtype(dtype, capability), "quantization": check_quantization(quantization, capability), "architectures": [],
                              "model_status": "UNKNOWN_REQUIRES_EVIDENCE", "architecture_status": "UNKNOWN_REQUIRES_EVIDENCE", "tested_modes": [], "evidence": [], "limitations": []}
    if not archs:
        report["note"] = "architecture unknown: pass architectures=[...] or a local model directory (no network access is used)"
        report["status"] = "UNKNOWN_REQUIRES_EVIDENCE"
        return report
    root = _vllm_root(package_root)
    registry = read_registry(package_root)
    arch_statuses: list[str] = []
    for a in archs:
        if any(m in a for m in UNSUPPORTED_ARCHITECTURE_MARKERS):
            raise StokastiskCompatibilityError(f"{a}: Mistral-family models are not supported on the Tesla T4 / SM75 / TRITON_ATTN build (mistral_common is not part of the supported dependency set; "
                                               "no Mistral-family model was ever run on the T4s). Use a dense Qwen2-family model.")
        if a not in registry:
            report["architectures"].append({"architecture": a, "status": "UNKNOWN_REQUIRES_EVIDENCE", "evidence": "static",
                                            "note": "not in the vLLM 0.25.1 registry; vLLM may fall back to its Transformers backend, which was not run on Kaggle"})
            arch_statuses.append("UNKNOWN_REQUIRES_EVIDENCE")
            continue
        module, cls = registry[a]
        dotted = module if module.startswith("vllm.") else f"vllm.model_executor.models.{module}"     # the registry mixes bare names, package directories and dotted paths
        f = _module_file(root, dotted)
        if f is None:
            raise StokastiskCompatibilityError(
                f"{a} is registered by vLLM 0.25.1 but its implementation module {dotted} is not part of this T4/Triton-only build "
                "(modules with a module-level FlashAttention/FlashInfer import were removed). It cannot be loaded here.")
        if uses_removed_backend(f, root):
            raise StokastiskCompatibilityError(f"{a}: {dotted} is built on a FlashAttention/FlashInfer/FlashMLA implementation that this T4/Triton-only build does not ship "
                                               "(it imports the placeholder module vllm.utils.t4_prohibited_stubs); it cannot run here.")
        gaps = find_missing_imports(f, root)
        if gaps:
            raise StokastiskCompatibilityError(f"{a}: {dotted} imports {' -> '.join(gaps[0])}, which is absent from this T4/Triton-only build; the model cannot be imported here.")
        if a in KAGGLE_VERIFIED_ARCHITECTURES:
            verified = sorted(m for m, r in MODEL_EVIDENCE.items() if r["architecture"] == a)
            rec = {"architecture": a, "module": module, "status": "SUPPORTED_WITH_CONDITIONS", "evidence": "kaggle", "verified_models": verified,
                   "note": "present in the shipped vLLM and exercised on Kaggle for the verified_models only; any other checkpoint of this architecture is UNKNOWN_REQUIRES_EVIDENCE until it is run"}
            arch_statuses.append("SUPPORTED_WITH_CONDITIONS")
        else:
            rec = {"architecture": a, "module": module, "status": "UNKNOWN_REQUIRES_EVIDENCE", "evidence": "static",
                   "note": "module and its unconditional vllm imports are present in the wheel; no checkpoint of this architecture was ever run on Kaggle"}
            arch_statuses.append("UNKNOWN_REQUIRES_EVIDENCE")
        report["architectures"].append(rec)
    report["architecture_status"] = "UNKNOWN_REQUIRES_EVIDENCE" if "UNKNOWN_REQUIRES_EVIDENCE" in arch_statuses else "SUPPORTED_WITH_CONDITIONS"
    notes: list[str] = []
    if found is not None:
        mid, rec = found
        problems = []
        if set(archs) != {rec["architecture"]}:
            problems.append(f"architectures {archs} differ from the verified {rec['architecture']}")
        if _VERIFIED_DTYPES.get(report["dtype"]) != rec["dtype"]:
            problems.append(f"dtype {report['dtype']!r} was not run (verified: {rec['dtype']!r})")
        if report["quantization"]["method"] is not None:
            problems.append(f"quantization {report['quantization']['method']!r} was not run (verified: unquantized)")
        if problems:
            notes.append(f"{mid} is a verified checkpoint, but this configuration is not what was run: " + "; ".join(problems))
        else:
            report["model_status"] = "SUPPORTED"
            report.update({"verified_model_id": mid, "tested_modes": list(rec["tested_modes"]), "evidence": list(rec["evidence"]), "limitations": list(rec["limitations"]),
                           "max_tested_model_len": rec["max_tested_model_len"]})
            notes.append(f"{mid}: run on Kaggle 2 x T4 in FP16 (see evidence); only the tested_modes and max_tested_model_len are covered")
    else:
        notes.append("this exact checkpoint was never run on Kaggle" + (f" (verified checkpoints: {', '.join(sorted(MODEL_EVIDENCE))})" if report["architecture_status"] != "UNKNOWN_REQUIRES_EVIDENCE" else ""))
    if mode is not None:
        want = _mode_name(mode)
        report["mode_status"] = "TESTED" if report["model_status"] == "SUPPORTED" and want in {_mode_name(m) for m in report["tested_modes"]} else "UNTESTED"
    report["status"] = report["model_status"]
    report["note"] = "; ".join(notes)
    return report


__all__ = ["FEATURES", "BLOCKING", "UNSUPPORTED_ARCHITECTURE_MARKERS", "STUB_MODULE", "uses_removed_backend", "QUANT_MIN_CAPABILITY", "KAGGLE_VERIFIED_ARCHITECTURES", "MODEL_EVIDENCE", "feature_matrix", "get_feature", "check_feature", "check_dtype", "check_quantization",
           "read_registry", "find_missing_imports", "check_model"]
