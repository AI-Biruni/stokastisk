# Stokastisk 0.1.0 — Implementation report (Codex deviation)

**Deviation from section 5 of the brief:** Codex CLI is not installed, and on 2026-11-11 the owner instructed "Do not use codex cli. Use claude cli only." All implementation below was done directly by Claude Code; no Codex run took place and no Codex output exists. This is recorded as a deviation, not as a Codex result.

## Changes (all under the locked version 0.1.0; vLLM stays 0.25.1)
| Area | Change |
|---|---|
| `sdk/stokastisk/parallel/routing.py` | `effective_speed` / `projected_load` with a `reference_speed`: an unobserved worker is assumed to run at the mean observed speed instead of the default 1.0, which had made it look ~100× slower than a worker with an observed tokens/s figure |
| `tests/test_parallel.py` | `test_unobserved_worker_still_receives_work_after_peer_learns_speed` (failed before the fix: plan `[0]`), `test_batches_stay_balanced_over_repeated_rounds` |
| `scripts/kaggle/dual_15_tp_pp_fulfilment.py`, `dual_16_replica_balance.py` | strict TP=1/TP=2/PP=2 and replica-balance stages |
| `scripts/make_notebooks.py`, `notebooks/kaggle-t4-sm75/notebooks/13_*`, `14_*` | two new unexecuted notebooks |
| `scripts/validation/validate_notebooks.py`, tests | fourteen expected notebooks |
| `scripts/release/assemble_release.py`, `prepare_hf_release.py` | writable base copy; `PRIOR-KAGGLE-EVIDENCE.json`; Hub `latest.json` reads `docs/kaggle-t4-verification.json` and asserts the executed manifest hash equals the released bundle |
| tests | rc4/rc2 baseline wheels restored under `work/baselines/` (from the verified backup) so no test is skipped |

Not changed on purpose: anything else in `sdk/` (the SDK wheel was executed on Kaggle in notebooks 13/14; a change would invalidate that evidence).

---
# Update 2026-11-12 — vLLM T4 coverage round
**Codex was not used.** The brief asked for Codex CLI to implement the changes, but the CLI is still not installed on this machine (`codex: command not found`), and the owner's earlier instruction in this project was "Do not use codex cli. Use claude cli only." Installing and authenticating a new third-party CLI was not requested, so Claude Code implemented the changes directly. This is a recorded deviation, not a Codex result; no "Codex diff" exists.
The brief's safeguards were applied to Claude's own work: inspect before editing (archive audit, wheel-vs-stage diff, source reads of `platforms/cuda.py`, `triton_attn.py`, `arg_utils.py`, `registry.py` before writing code); project version 0.1.0 and vLLM 0.25.1 untouched; CUDA binaries and vLLM wheel untouched; no existing API removed; no assertion weakened or test deleted; no Flash dependency introduced.
Independent self-review of the full diff found and fixed: (1) `check_dtype("auto")` would have rejected a working fp16 `auto` setup — now allowed; (2) the model resolver only handled `<name>.py`, producing 20 false "module removed" blocks for package-directory and dotted registry entries — fixed (356-architecture sweep now 346 pass / 10 genuine blocks); (3) per-architecture re-parsing made a sweep take minutes — added a per-file cache; (4) the test fixture lacked `__init__.py` files and would have failed — fixed; (5) clumsy environment code in `ServerProcess.start` — rewritten. Tests were written but not run, as instructed.
