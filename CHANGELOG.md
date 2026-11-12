# Changelog

Project version is permanently **0.1.0** (VERSION_POLICY.md). Entries are distinguished by date and build tag, never by version.

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
