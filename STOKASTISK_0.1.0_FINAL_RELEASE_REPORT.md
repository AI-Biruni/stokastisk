# Stokastisk 0.1.0 — final release report (2026-11-12)

**Verdict: NOT fully verified. The integrated wheel is built, statically inspected and CPU-tested; it is awaiting Kaggle notebook 15.** Legacy two-wheel bundle: `KAGGLE_T4_VERIFIED`. Project version unchanged: **0.1.0**.

## Integrated wheel (canonical)
* File: `stokastisk-0.1.0-2-cp313-cp313-linux_x86_64.whl` — size **38,151,399** bytes — SHA256 `7e4b700929ec7f3489570e93252ef5a49da2d1682c111c5feefb55efead84369`.
* Built by `scripts/release/assemble_integrated_release.py --build-tag 2` from source commit `a69ba96021a6c0ab1a8917209dd002142ab05047` (clean tracked tree) with Python 3.13.15; build manifest `manifests/BUILD-MANIFEST.json`.
* Release ZIP `stokastisk-0.1.0-integrated-kaggle-t4-sm75-kaggle-input.zip` — 37,052,663 B — SHA256 `f28ccdf230fb80f374e6c44b1549a7d36538917aa8d99f16c07b952b9c7d3378`.
* Why a new build tag (2) instead of reusing the candidate (build tag 1, `815f9783…`): the SDK changed, so the hash changed; the candidate stays published and unchanged, marked superseded.
* Contents: 52 SDK modules (`stokastisk/**` equal to `sdk/` byte for byte) + 2543 `vllm/**` files (each byte-identical to the Kaggle-verified vLLM wheel `cc3bcafb…`) + `vllm-0.25.1.dist-info` via `*.data/platlib/` + console script `vllm`. RECORD: 2605 entries, 0 mismatches, 0 unlisted members.
* Native objects (re-audited with `cuobjdump` 13.0): `_C_stable_libtorch` and `_moe_C_stable_libtorch` — `sm_75` cubin only, no PTX, `libcudart.so.13`; `cumem_allocator`, `fs_io_C`, `spinloop`, `_rust_tool_parser` — no GPU code; none needs a CUDA 12 soname; max GLIBC 2.38; no Flash markers.

## What was changed
**SDK (hardening; `docs/SOURCE_DIFF_VS_KAGGLE_EXECUTED.md`):** `LLM` — closed-state, idempotent `close()` with logged (not swallowed) failures, context manager, input validation (`TypeError`/`ValueError`), model-id validation, engine closed if the backend check fails; `StokastiskConfig` — rejects bool/str sizes, `max_model_len<16`, non-numeric memory fraction; `ReplicaPool` — input validation, serialised `generate()` (concurrent callers previously shared one result queue), queue cleanup on close, logged send errors; worker reports a failing `llm.close()`.
**Scripts:** new `02_install_integrated.py` (one wheel; Python/torch/triton/CUDA/GPU checks; wheel SHA256 vs manifest; removes a legacy two-wheel install — `stokastisk` and `vllm` only; `--no-deps --no-index`; pins downloaded with `--require-hashes`; before/after package inventory; protected-package diff; honest `pip check` before/after; imports; `TRITON_ATTN`; installed files byte-equal to the wheel; PASS/FAIL/WARNING/SKIPPED; non-zero exit on required failure), `candidate_15_integration.py`, hash-pinned `requirements/kaggle-t4-core.hashes.txt`; `_common.finish` now fails a stage with no PASS check. The legacy `02_install.py` is untouched and refuses the one-wheel bundle (it requires exactly two wheels).
**Notebooks:** 01–14 first markdown cell states the executed Kaggle run (code cells unchanged); notebook 13 and 14 no longer say "UNEXECUTED — verification pending"; new `notebooks/kaggle-t4-sm75/integrated/15_stokastisk_candidate_sdk_integration.ipynb` (UNEXECUTED; 4 stages: static, engine, server, summary; produces `15-stokastisk-candidate-acceptance.zip`).
**Docs:** README, INSTALL, RELEASE, CHANGELOG, Hub card (primary landing page), `docs/INTEGRATED_RELEASE.md`, `docs/DEPENDENCY_COMPATIBILITY.md`, `docs/KAGGLE_T4_VERIFICATION.md` (+json), `docs/SOURCE_DIFF_VS_KAGGLE_EXECUTED.md`, pre-change engineering report, preserved previous GitHub release notes.

## Files: retained / integrated / replaced
* **Retained unchanged:** legacy wheels (`29edaa55…`, `cc3bcafb…`), legacy ZIP (`46439612…`), legacy bundle, build-tag-1 candidate wheel and manifests (`815f9783…`), tag `v0.1.0` (`1acc49c`), GitHub release `v0.1.0` original assets, Hub `latest.json`, all executed notebooks and acceptance ZIPs.
* **Integrated:** the candidate's modules were already in `sdk/` (the candidate was built from it) — integration = hardening + packaging, not copying files over a tree.
* **Replaced (edited in place):** `engine.py`, `config.py`, `parallel/replica.py`, `parallel/worker.py`, `_common.py`, notebook first cells, README/INSTALL/RELEASE/Hub card/verification docs, GitHub release `v0.1.0` *notes* (previous text preserved in `docs/release-notes-v0.1.0-previous.md`).

## Tests actually executed (local, CPU only; the GeForce 940M was never used for GPU work)
* `python3.13 -m pytest tests` → **253 passed**, 37 subtests, 0 failures (new: `test_hardening.py` 18, `test_installer_integrated.py` 6, `test_integrated_assets.py` 6).
* `scripts/validation/validate_notebooks.py` → STATIC_PASS for 15 notebooks; `scripts/release/verify_project_version.py` → 6/6 checks.
* `scripts/audit/verify_sdk_candidate.py` on the built wheel and on the copy downloaded from the Hub → all static checks pass.
* `pip install --no-deps --no-index --target` of the final wheel: `vllm 0.25.1` and `stokastisk 0.1.0` distributions resolve, `import stokastisk` works, default backend `TRITON_ATTN`, `LLM.chat` present. (Local layout test; not Kaggle.)
* Installer `--dry-run` on this machine (no Kaggle packages) aborts with FAIL and installs nothing — the intended abort path; `pip download --require-hashes` of the 8 pins succeeded.
* xgrammar CPU functional check under transformers 5.22.2 (0.2.5.post1 and 0.2.6): pass.
Not run: any GPU test, any Kaggle notebook, any local vLLM import/inference.

## Kaggle evidence used
Notebooks 13 and 14 (executed 2026-11-12) — provenance cell printed SDK wheel `29edaa55…`, vLLM wheel `cc3bcafb…`, bundle `SHA256SUMS` sha `97ad6660…`: **notebook 13** 7/7 stages PASS, TP=1/TP=2/PP=2 12/12 checks each (peak GPU MiB 13249/3, 13751/13751, 12895/13353; memory freed to 3/3), informational WARNING that TP=2 text is not byte-identical to TP=1 (preserved); **notebook 14** 50/50 checks, 15 rounds each split evenly, both workers `TRITON_ATTN` on distinct UUIDs, memory released. Decode, batch 8 (Qwen2.5-1.5B): TP=1 238.4, TP=2 133.1, PP=2 148.1 tok/s; two replicas 0.91–0.95× of TP=1 — **no speed-up from TP, PP or replicas is claimed**. These runs used the legacy two-wheel bundle; they are **not** evidence that the integrated wheel installs or that `compat`, `serving` or `LLM.chat` work. Originals: `evidence/kaggle-legacy-two-wheel/` (GitHub) and `…/acceptance-zips/` (Hub).

## Dependency conflict and resolution
`xgrammar 0.2.5.post1` declares `transformers<5` but Kaggle has 5.20.2; the legacy installer's `pip check` filter hid it behind a PASS. No xgrammar release removes that bound while accepting vLLM's `apache-tvm-ffi==0.1.9` (0.2.6+ need tvm-ffi ≥0.1.10). xgrammar's only transformers imports are tokenizer base classes; JSON-schema compile/accept/reject/bitmask pass on CPU under transformers 5.22.2. **Decision:** keep the Kaggle-executed pins, do not edit metadata, report the conflict as a WARNING everywhere; notebook 15 tests GPU structured decoding and the decision must be reopened if it fails. `pip check` is therefore **not clean** (documented). Protected packages are never resolved or replaced.

## Publication
* GitHub: https://github.com/AI-Biruni/stokastisk — source commit `a69ba96`, pushed head `c20e0f6cacba30802f24b4609ffaa86f7640a953` (fast-forward, no force); release `v0.1.0` notes updated and assets added (wheel, ZIP, `.sha256`, manifests, `SHA256SUMS-integrated-build2`) with no asset overwritten; tag `v0.1.0` still `1acc49c`; pre-release `sdk-candidate-0.1.0-build1` retitled SUPERSEDED. The first ZIP upload hit HTTP 408 timeouts through `gh release upload`; it was uploaded via `gh api` and its hash re-verified after download.
* Hugging Face: https://huggingface.co/waqasm86/stokastisk — commit `256d7ec0cde653c9a21fe69129ae3fc48495853b` (additive; 175 files). Layout: `releases/stokastisk-0.1.0-integrated-kaggle-t4-sm75/` (canonical), `releases/stokastisk-0.1.0-kaggle-t4-sm75/` (legacy, unchanged), `sdk-candidate/` (superseded candidate), `evidence/kaggle-legacy-two-wheel/` (executed notebooks, reports, 14 original acceptance ZIPs), `docs/`, `source/`, `latest.json` (legacy, untouched), `latest-integrated.json`.
* Direct wheel URL: https://huggingface.co/waqasm86/stokastisk/resolve/main/releases/stokastisk-0.1.0-integrated-kaggle-t4-sm75/wheels/stokastisk-0.1.0-2-cp313-cp313-linux_x86_64.whl
* Independent post-publish verification: both remotes' wheel and ZIP re-downloaded and hashed (equal to the local build); legacy wheels, legacy ZIP and the candidate wheel re-hashed on the Hub (unchanged); `verify_sdk_candidate.py` re-run on the Hub copy (pass).

## Remaining blockers / what is not verified
1. **Notebook 15 has not been run.** Until it passes on Kaggle 2 × T4 the integrated wheel is labelled *verification pending* everywhere. Unverified: installation on Kaggle, `import vllm` through the integrated wheel, `importlib.metadata` in the Kaggle image, the `_C_stable_libtorch` ABI via this install path, `LLM.chat`, `compat`, `serving` (HTTP, SSE, metrics), GPU xgrammar decoding, the hardened `LLM`/`ReplicaPool` lifecycle on GPUs, CUDA graphs, fp8/bfloat16 (rejected by design), speculative decoding, LoRA, multimodal, MLA.
2. `pip check` unclean (documented xgrammar WARNING).
3. Upstream behaviour in the frozen vLLM bytes: calling `vllm.LLM` directly (not through the SDK) uses upstream backend auto-selection (probes `FLASH_ATTN`, invalid on SM75) and ViT attention falls back to `TORCH_SDPA`.
4. Hub `latest.json` still points at the legacy release (intentionally; `latest-integrated.json` points at the canonical one).
5. Worker-failure recovery and native transport remain CPU-tested / experimental as before.
