# Install on Kaggle (2 × T4, Python 3.13)

Release `stokastisk-0.1.0-kaggle-t4-sm75` (project version 0.1.0). Status: `KAGGLE_GPU_VERIFICATION_PENDING` — see [RELEASE.md](RELEASE.md).

**Option A – Hub CDN, no dataset (recommended):** needs *Internet ON*. `scripts/fetch_release.py` (stdlib only, a single HTTP stream (parallel ranges are opt-in via `workers`; 4 parallel ranges stalled on the Hub CDN in a workstation test), SHA256-verified) downloads the ~37 MB release:
```python
import urllib.request, runpy
open('/tmp/fetch_release.py','wb').write(urllib.request.urlopen('https://huggingface.co/waqasm86/stokastisk/resolve/main/scripts/fetch_release.py').read())
bundle = runpy.run_path('/tmp/fetch_release.py')['fetch_release']('stokastisk-0.1.0-kaggle-t4-sm75')
```

**Option B – Kaggle dataset:** upload `stokastisk-0.1.0-kaggle-t4-sm75-kaggle-input.zip` (from the Hub `releases/stokastisk-0.1.0-kaggle-t4-sm75/`) as a dataset and attach it.
Its root must contain the directory `stokastisk-0.1.0-kaggle-t4-sm75/`. The tooling accepts only that exact release name.

**Manual:**
```bash
python3.13 scripts/fetch_release.py stokastisk-0.1.0-kaggle-t4-sm75 /tmp/stokastisk-release
python3.13 /tmp/stokastisk-release/stokastisk-0.1.0-kaggle-t4-sm75/scripts/kaggle/02_install.py /tmp/stokastisk-release/stokastisk-0.1.0-kaggle-t4-sm75 --dry-run   # plan only
python3.13 .../02_install.py <bundle>                                                                                                                        # install
```
The installer changes nothing if the plan has a conflict (`02_install_plan.json`), fails if any protected package (torch, triton, CUDA, NCCL, transformers 5.x…) would change,
and never installs FlashAttention/FlashInfer. The two wheels are installed with `--force-reinstall --no-deps --no-index`, so a pre-existing Stokastisk distribution of any other version is replaced,
and the run fails unless the installed version is exactly `0.1.0`. Kaggle starts every notebook in a fresh session, so each notebook that imports `stokastisk` must run the installer first.
