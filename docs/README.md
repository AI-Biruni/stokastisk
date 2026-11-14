# Documentation

| For | Where |
|---|---|
| **Kaggle users** — install, use, every module | [User guide](guide/index.md) · web version: <https://stokastisk-ai.github.io/stokastisk/docs/> · [API reference](guide/api-reference.md) |
| What is verified and how | [KAGGLE_ACCEPTANCE.md](KAGGLE_ACCEPTANCE.md) · [PRODUCTION_READINESS.md](PRODUCTION_READINESS.md) · generated [feature matrix](guide/feature-matrix.md) · raw artifacts `../evidence/kaggle/` |
| How the wheel is built and why dependencies are pinned | [ARCHITECTURE.md](ARCHITECTURE.md) · [DEPENDENCY_COMPATIBILITY.md](DEPENDENCY_COMPATIBILITY.md) · [../BUILD.md](../BUILD.md) |
| Release facts (hashes, install command) | [../RELEASE.md](../RELEASE.md) · [../RELEASE.json](../RELEASE.json) · [../INSTALL.md](../INSTALL.md) · [../CHANGELOG.md](../CHANGELOG.md) · [../VERSION_POLICY.md](../VERSION_POLICY.md) |
| Hugging Face model card (uploaded as `README.md` there) | [HUB_README.md](HUB_README.md) |

## Keeping the documentation current

* The user guide is hand-written Markdown in `guide/`; `guide/api-reference.md` and `guide/feature-matrix.md` are **generated** (the second one from `stokastisk.compat`) from the SDK source: run `python3.13 scripts/release/build_docs_site.py <out-dir>` after changing `sdk/`, and commit the regenerated file (a test fails if it is stale).
* The HTML site is rebuilt and pushed to the `gh-pages` branch (`/docs/`) by `.github/workflows/docs.yml` whenever `docs/guide/`, `sdk/` or the builder changes on `main`.
* `HUB_README.md` is uploaded to Hugging Face by hand: `hf upload waqasm86/stokastisk docs/HUB_README.md README.md`.
* `tests/test_readme_examples.py` checks links, that referenced repository files exist, and the API reference.
