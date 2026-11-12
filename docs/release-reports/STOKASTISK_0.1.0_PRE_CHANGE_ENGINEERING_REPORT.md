# Stokastisk 0.1.0 — pre-change engineering report (2026-11-12)

Written from the audit performed before any file of this release task was edited (the audit commands ran first; this report was committed with the change set, not before it).

| Item | Finding |
|---|---|
| Source revision at start | `e416fb08055790197c82fbdd03aa45c605e56178` on `main`, clean tree, equal to `origin/main` after `git fetch` |
| Project version | `0.1.0` (`VERSION`, `sdk/stokastisk/_version.py`); tag `v0.1.0` → `1acc49c…` (annotated tag object `b31f1995…`), not touched |
| Candidate wheel | `dist-rc/sdk-candidate-0.1.0-build1/stokastisk-0.1.0-1-cp313-cp313-linux_x86_64.whl`, 38,148,975 bytes, sha256 `815f978346cfc6d85b7b47f9b96ce6805b5630ea2e0d3c0070c45f538c11cf6b` — verified locally, unchanged |
| Existing release assets (GitHub `v0.1.0`) | `SHA256SUMS`, release ZIP (+ `.sha256`), `stokastisk-0.1.0-py3-none-any.whl` (`29edaa55…`), `vllm-0.25.1-cp38-abi3-linux_x86_64.whl` (`cc3bcafb…`); pre-release `sdk-candidate-0.1.0-build1` holds the candidate |
| Hub at start | commit `00e6e2dd…`; `releases/stokastisk-0.1.0-kaggle-t4-sm75/…`, `sdk-candidate/stokastisk-0.1.0-build1/…`, `docs/`, `source/`, `latest.json`, `scripts/fetch_release.py` |
| Kaggle evidence supplied | `notebooks-executed-with-output/13-stokastisk-tp-pp-fulfilment.ipynb`, `13-stokastisk-acceptance.zip`, `14-stokastisk-replica-balance.ipynb`, `14-stokastisk-acceptance.zip` (and 01–12). Local names differ from the upload names only by the dash/underscore form used in the task text. |
| Provenance of notebooks 13/14 | The executed notebooks' provenance cell printed SDK wheel `29edaa55…` and vLLM wheel `cc3bcafb…`; bundle `SHA256SUMS` sha256 `97ad6660…`. They ran the **legacy two-wheel bundle**, released SDK wheel itself — not the integrated wheel. |
| Notebook 13 evidence | 7/7 required stages PASS; TP=1/TP=2/PP=2 each 12/12 checks; informational WARNING: TP=2 text not byte-identical to TP=1 |
| Notebook 14 evidence | 50/50 checks PASS; 15 rounds, all split evenly; replicas 0.91–0.95× of TP=1 |

## Files to merge / modify
* `sdk/stokastisk/` already contained the candidate's modules (`compat.py`, `serving.py`, `LLM.chat`); the candidate wheel was built from it. Integration therefore means hardening and verifying that tree, not copying files over it. Edited: `engine.py`, `config.py`, `parallel/replica.py`, `parallel/worker.py`.
* New: `scripts/kaggle/02_install_integrated.py`, `candidate_15_integration.py`, `scripts/release/{build_sdk_candidate_wheel,assemble_integrated_release,gen_pin_hashes}.py`, `scripts/audit/{verify_sdk_candidate,xgrammar_cpu_check}.py`, notebook 15, tests, `requirements/kaggle-t4-core.hashes.txt`, docs.
* Modified: `scripts/kaggle/_common.py` (`finish`), `scripts/make_notebooks.py`, `scripts/validation/validate_notebooks.py`, notebooks 01–14 (first markdown cell only), README/INSTALL/RELEASE/Hub card/verification docs.

## Retained as historical artifacts (never overwritten)
Legacy two-wheel bundle and its ZIP, both historical wheels, the build-tag-1 candidate wheel and its manifests, all executed notebooks and acceptance ZIPs, tag `v0.1.0`, GitHub release `v0.1.0` assets, `latest.json`.

## Blockers / risks found by the audit
1. **xgrammar 0.2.5.post1 declares `transformers<5`** while Kaggle has transformers 5.20.2; the legacy installer's `pip check` filter hid this behind a PASS. See `docs/DEPENDENCY_COMPATIBILITY.md` (functionally verified on CPU; GPU structured decoding unverified).
2. **Integrated wheel never executed**: not installed on Kaggle, `import vllm` through it unverified, `LLM.chat`, `compat`, `serving` unexecuted. Gate: notebook 15.
3. **Legacy installer incompatible with one-wheel bundles** (requires exactly two wheels) — a dedicated integrated installer is needed and protects against the double-install.
4. **`vllm` distribution metadata**: vLLM calls `importlib.metadata.version("vllm")`; the integrated wheel bundles a `vllm-0.25.1.dist-info` through the wheel `.data/platlib` path (verified by a local `pip install --target`, not on Kaggle).
5. Production-readiness gaps found in the SDK (fixed in this change set): `LLM` methods failed with `AttributeError` after `close()`; no input validation; `ReplicaPool.generate` was not safe for concurrent callers (shared result queue); queues were never closed; worker `close()` failures were swallowed; `StokastiskConfig` accepted `True`/strings as sizes; `finish()` could pass a stage made only of WARNING/SKIPPED checks.
6. Upstream behaviour not changed (vLLM bytes are frozen): auto backend selection probes FLASH_ATTN before TRITON_ATTN when a user calls `vllm.LLM` directly; ViT attention on SM75 uses `TORCH_SDPA`.
