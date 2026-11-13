# Changelog

Project version is permanently **0.1.0** (VERSION_POLICY.md). Entries are distinguished by date and build tag, never by version.

## 2026-11-13 — integrated wheel, build tag 3 (Kaggle verification pending)
* **Why:** build 2 failed Kaggle notebook 15 — structured output (`No module named 'lmformatenforcer'`), HTTP serving (`No module named 'model_hosting_container_standards'`), a package-inventory comparison that used two different name normalisers, Flash `.pyc`/source files in the vLLM tree, and an input-discovery defect that silently downloaded build 2 instead of the uploaded wheel. Evidence preserved: `evidence/kaggle/build-2-failed-notebook-15/`.
* **Wheel:** `stokastisk-0.1.0-3-cp313-cp313-linux_x86_64.whl`. vLLM tree = the byte-pinned vLLM wheel through `scripts/release/t4_flash_free.py`: 28 Flash-family modules removed, imports redirected to one explicit placeholder, backend registries/`AttentionConfig` refuse Flash backends, import guard for `flashinfer`/`flash_attn`/`flash_mla`; native objects byte-identical. Reproducible build (two builds → identical bytes).
* **Dependencies:** complete hash-pinned locks (`core`, `structured-output`, `serving`, optional `models`) resolved against the real Kaggle inventory; xgrammar **0.2.3** (metadata-compatible with transformers 5.x and apache-tvm-ffi 0.1.9 — no patched distribution); `pip check` must show no new line; observability extras unsupported (protobuf conflict).
* **SDK:** structured-output backend pinned to xgrammar (`StokastiskConfig.structured_output_backend`, no `auto` fallback) + `LLM.generate_structured` + `stokastisk.structured.validate_json_output`; `LLM` lifecycle (`EngineClosedError`, `EngineTimeoutError`, `EngineCloseError`, serialised calls, `timeout=`); `serving` (persistent log, `ServerStartupError` with exit code/command/log/excerpt, validation, port check, process-group cleanup, restart, typed HTTP errors, truncated-stream detection); `ReplicaPool` (unique batch ids, close problems reported, kill escalation); strict config types (bool is not an int); no silent `except: pass`.
* **Kaggle framework:** single inventory implementation; status vocabulary `PASS FAIL ERROR WARNING SKIPPED NOT_RUN` (only PASS passes); bundle discovery/verification embedded in the notebook; no HF fallback unless `STOKASTISK_ALLOW_HF_FALLBACK=1`; acceptance ZIP always written; notebook fails unless every stage is PASS.
* Builds 1 and 2 and the legacy release are untouched.

## 2026-11-12 — integrated wheel, build tag 2 (Kaggle verification pending)
* New canonical distribution: `stokastisk-0.1.0-2-cp313-cp313-linux_x86_64.whl` = SDK + bundled vLLM 0.25.1 (byte-identical to the verified vLLM wheel) + CUDA 13.0 `sm_75` objects; installer `02_install_integrated.py`; release `stokastisk-0.1.0-integrated-kaggle-t4-sm75`.
* SDK: `stokastisk.compat`, `stokastisk.serving`, `LLM.chat`, explicit bfloat16 pre-check packaged; `LLM` closed-state/idempotent close/context manager/input validation; `ReplicaPool` input validation, serialised `generate`, queue cleanup; stricter `StokastiskConfig` validation; worker close errors reported.
* Dependency finding documented: xgrammar 0.2.5.post1 metadata (`transformers<5`) vs transformers 5.x — `docs/DEPENDENCY_COMPATIBILITY.md`; core pins now hash-verified.
* Evidence: notebooks 13 and 14 re-run on the released legacy bundle (SDK wheel 29edaa55…) — PASS; preserved under `evidence/kaggle-legacy-two-wheel/`. Notebook 15 (integrated wheel) added, unexecuted.
* Kaggle stage framework: a stage with no PASS check now fails.
* Build-tag-1 candidate (`sdk-candidate-0.1.0-build1`) superseded, kept.

## 2026-11-12 — build-tag-1 candidate
* First integrated wheel (unexecuted), GitHub pre-release `sdk-candidate-0.1.0-build1`, Hub `sdk-candidate/stokastisk-0.1.0-build1/`.

## 2026-11-08 … 2026-11-12 — legacy two-wheel release `stokastisk-0.1.0-kaggle-t4-sm75`
* Kaggle-verified on 2 × Tesla T4 (notebooks 01–14). See RELEASE.md.
