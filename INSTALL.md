# Install on Kaggle (2 × T4, Python 3.13)

Release `stokastisk-0.1.0-kaggle-t4-sm75` (project version 0.1.0). Status: `KAGGLE_T4_VERIFIED` (all 14 notebooks run on Kaggle 2 × T4) — see [RELEASE.md](RELEASE.md) and [docs/KAGGLE_T4_VERIFICATION.md](docs/KAGGLE_T4_VERIFICATION.md).

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

## Canonical: integrated wheel (Kaggle verification pending — notebook 15)
Release `stokastisk-0.1.0-integrated-kaggle-t4-sm75`: one wheel `stokastisk-0.1.0-2-cp313-cp313-linux_x86_64.whl` (SDK + bundled vLLM 0.25.1 + CUDA 13.0 `sm_75` objects), project version 0.1.0.
```python
import urllib.request, runpy, subprocess, sys
open('/tmp/fetch_release.py','wb').write(urllib.request.urlopen('https://huggingface.co/waqasm86/stokastisk/resolve/main/scripts/fetch_release.py').read())
bundle = runpy.run_path('/tmp/fetch_release.py')['fetch_release']('stokastisk-0.1.0-integrated-kaggle-t4-sm75')
subprocess.run([sys.executable, f'{bundle}/scripts/kaggle/02_install_integrated.py', str(bundle)], check=True)    # --dry-run plans only
```
Manual: `python3.13 scripts/kaggle/02_install_integrated.py <bundle> [--dry-run]`. Requirements checked: Python 3.13, `torch==2.11.0+cu130`, `triton==3.6.0`, CUDA 13.0, transformers `>=5.5.3,<6`, two Tesla T4 (SM75), wheel SHA256 equals the manifest. The installer never runs a dependency resolver: wheel `--no-deps --no-index`, eight pins with `--require-hashes`. A legacy two-wheel install is removed first (`stokastisk` and `vllm` only). Afterwards it proves that no preinstalled package (torch, triton, nvidia-*, NCCL, transformers …) changed and reports `pip check` honestly (documented xgrammar WARNING, anything else FAILS). Exit code is non-zero if any required check fails. Do not use `02_install.py` for this bundle (it requires exactly two wheels and refuses), and never install the integrated wheel next to the standalone `vllm` wheel.
Verify in a fresh notebook session with notebook 15 (`notebooks/kaggle-t4-sm75/integrated/`).
