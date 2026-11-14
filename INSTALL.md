# Install on Kaggle (2 × T4, Python 3.13)

1. Notebook settings: *Accelerator: GPU T4 × 2*, *Internet: ON*.
2. One cell — this pins the wheel by URL **and SHA256** and uses PyPI as the only package index:
```bash
!python -m pip install --index-url https://pypi.org/simple "stokastisk @ https://huggingface.co/waqasm86/stokastisk/resolve/main/wheels/stokastisk-0.1.0-4-cp313-cp313-linux_x86_64.whl#sha256=052156d71adc6ad9966901565f86dbded9ff41c620c782db4ff377a1a2b4f62e"
```
pip downloads the 38 MB wheel from the Hugging Face CDN, **refuses it if its SHA256 differs**, and installs 19 pinned dependencies from PyPI. Kaggle's PyTorch, Triton, transformers and every other pre‑installed package stay untouched. The command is the same for everyone who wants build 4; the current file name, size and hash are also recorded in [RELEASE.json](RELEASE.json).
3. Check: `python -c "import stokastisk; print(stokastisk.__version__, stokastisk.prohibited_report()['clean'])"` → `0.1.0 True`.

## Why not `--extra-index-url`?
`pip install stokastisk --extra-index-url https://stokastisk-ai.github.io/stokastisk/simple/` also works (the index lists the same build‑4 file with the same SHA256), but `--extra-index-url` makes pip consider **PyPI as well as the extra index** and pick the best match. If someone later published a different project named `stokastisk` on PyPI, pip could install that instead. The name is unregistered on PyPI today; the pinned command above removes the question. The index is kept as a convenience only.

## Roll back to build 3
Build 3 (the Kaggle‑acceptance build) stays on Hugging Face. Same command with the build‑3 file and hash:
```bash
!python -m pip install --force-reinstall --no-deps --index-url https://pypi.org/simple "stokastisk @ https://huggingface.co/waqasm86/stokastisk/resolve/main/wheels/stokastisk-0.1.0-3-cp313-cp313-linux_x86_64.whl#sha256=72039ca4044e48fdf5edca8051dcb48b42fe61a69e07715fa8803e029581fe12"
```
Build 3 and build 4 differ only in the compatibility reporting (`stokastisk.compat`, which models/features are reported as Kaggle-tested); inference, server and binaries are identical ([RELEASE.md](RELEASE.md)).

Uninstall: `pip uninstall stokastisk`. Acceptance testing (maintainers): [docs/KAGGLE_ACCEPTANCE.md](docs/KAGGLE_ACCEPTANCE.md).
