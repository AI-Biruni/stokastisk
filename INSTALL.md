# Install on Kaggle (2 × T4, Python 3.13)
1. Notebook settings: *Accelerator: GPU T4 × 2*, *Internet: ON*.
2. One cell:
```bash
pip install stokastisk --extra-index-url https://stokastisk-ai.github.io/stokastisk/simple/
```
(`!pip install …` in a notebook cell.) The index page on GitHub Pages points pip at the wheel on the Hugging Face CDN (`https://huggingface.co/waqasm86/stokastisk/resolve/main/wheels/`) and carries its SHA256; pip downloads 38 MB, verifies it and installs 19 pinned dependencies. Kaggle's PyTorch, Triton, transformers and every other pre‑installed package stay untouched.
3. Check: `python -c "import stokastisk; print(stokastisk.__version__, stokastisk.prohibited_report()['clean'])"` → `0.1.0 True`.

Uninstall: `pip uninstall stokastisk`. Acceptance testing (maintainers): [docs/KAGGLE_ACCEPTANCE.md](docs/KAGGLE_ACCEPTANCE.md).
