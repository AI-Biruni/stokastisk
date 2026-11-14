# Version policy — Stokastisk project version is locked at 0.1.0

## Canonical version

The Stokastisk **project version** is exactly:

```
0.1.0
```

The single authoritative declaration is the file [`VERSION`](VERSION) (content `0.1.0` plus a newline). Everything else is derived from it or verified against it:

| Declaration | Mechanism |
|---|---|
| Python package metadata / wheel / sdist | `pyproject.toml` uses `dynamic = ["version"]` read from `VERSION` |
| `stokastisk.__version__` | `sdk/stokastisk/_version.py` (verified equal to `VERSION`) |
| Release name, directory, ZIP | `stokastisk-0.1.0-kaggle-t4-sm75` |
| Release manifest, model card, README, notebooks, installer | checked by `scripts/release/verify_project_version.py` |
| Git tag / GitHub release | `v0.1.0` only |

## Permanent lock

The version is **locked**. It must not change because of any ordinary development activity, including: adding a feature, fixing a bug, improving performance, adding a GPU architecture, updating documentation, adding a notebook, changing CUDA or vLLM integration, adding a scheduler, refactoring, publishing files, committing, uploading to Hugging Face, building a release archive, completing a benchmark, or passing more acceptance tests. Semantic-versioning conventions do **not** override this lock.

## Owner authorization

Only a direct, explicit instruction from the project owner to change the project version authorizes a change. No person, tool, bot or workflow may infer that authorization.

## No automatic versioning

* No release script, CI job or packaging tool may increment, derive or guess a project version.
* Version derivation from Git tags (setuptools-scm, versioneer, `git describe`, bump-version tools, semantic-release) is forbidden.
* `scripts/release/assemble_build3_release.py` asserts that `VERSION` says `0.1.0` before building, names every artifact `0.1.0`, and verifies the built wheel (`scripts/audit/verify_integrated_wheel.py`).
* `scripts/release/verify_project_version.py` carries its own `LOCKED_VERSION` constant, so editing only `VERSION` makes it fail. CI (`.github/workflows/version-lock.yml`) and `tests/test_version_policy.py` run it on every push and pull request.
* The Kaggle installer (`scripts/kaggle/02_install_integrated.py`) installs the Stokastisk wheel with `--force-reinstall --no-deps --no-index`, so an already-installed distribution with a different version is replaced, and it fails unless the installed version is exactly `0.1.0`.

## Project version vs. dependency versions

The project version identifies Stokastisk itself. These are **platform and dependency versions** and are unrelated to it; they change independently and must never be edited to satisfy this policy:

* vLLM 0.25.1 · Python 3.13 · CUDA Toolkit 13.0 · PyTorch 2.11.0+cu130 · Triton 3.6.0 · NCCL, model revisions, git commit identifiers and third-party package pins.

## Procedure for a future, explicitly authorized change

Only after the owner explicitly instructs a project-version change, in one reviewed commit:

1. Update `VERSION`, `sdk/stokastisk/_version.py`, and `LOCKED_VERSION` / `LOCKED` in `scripts/release/verify_project_version.py`, `scripts/release/assemble_build3_release.py`, `scripts/kaggle/02_install_integrated.py`, and `tests/test_version_policy.py`.
2. Update the release name and directory, the user-guide notebooks in `notebooks/`, README, RELEASE, INSTALL, `docs/` (including `docs/HUB_README.md`, the Hugging Face model card) and the package index (`scripts/release/make_pip_index.py`).
3. Run the verifier and the full test suite, rebuild the release, and record the owner's instruction in the commit message.
