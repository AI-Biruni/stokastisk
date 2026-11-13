#!/usr/bin/env python3.13
"""Generate Kaggle notebook 15 for the Stokastisk 0.1.0 BUILD 3 integrated wheel.
usage: make_notebook15_build3.py OUT.ipynb WHEEL_SHA256 [BUNDLE_ZIP_SHA256]

Cell 1 embeds scripts/kaggle/bundle.py VERBATIM (the bundle's own scripts cannot be used before the bundle is found) and then locates, verifies and extracts the ATTACHED bundle:
no silent Hugging Face fallback, no substitution of a different release. The remaining cells install (02_install_integrated.py), run the four stages as separate processes with
their output captured to 15c_<stage>.log, ALWAYS build the acceptance ZIP, and finally fail the notebook unless every stage is PASS."""
import json, pathlib, sys

import nbformat

ROOT = pathlib.Path(__file__).resolve().parents[2]
WHEEL_NAME = "stokastisk-0.1.0-3-cp313-cp313-linux_x86_64.whl"
BUNDLE_PY = (ROOT / "scripts/kaggle/bundle.py").read_text()


def build(wheel_sha256: str, bundle_zip_sha256: str | None) -> nbformat.NotebookNode:
    nb = nbformat.v4.new_notebook()
    nb.metadata = {"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"}, "language_info": {"name": "python", "version": "3.13"},
                   "stokastisk": {"notebook": "15_stokastisk_candidate_sdk_integration", "release": "0.1.0", "build": "3", "wheel": WHEEL_NAME, "wheel_sha256": wheel_sha256}}
    md = lambda s: nbformat.v4.new_markdown_cell(s)          # noqa: E731
    code = lambda s: nbformat.v4.new_code_cell(s)            # noqa: E731
    cells = [
        md("# Stokastisk 0.1.0 - build 3 - integrated wheel acceptance (Kaggle 2 x Tesla T4)\n\n"
           "**Settings:** Accelerator *GPU T4 x2*, Internet *ON*, Python 3.13. **Input:** attach the dataset that contains `stokastisk-0.1.0-build3-kaggle-t4-sm75-kaggle-input.zip` "
           "(the complete release bundle - a wheel-only archive is rejected).\n\n"
           f"Verifies exactly one wheel, `{WHEEL_NAME}` (SHA256 `{wheel_sha256}`), against the attached bundle. There is **no** download fallback unless you set `STOKASTISK_ALLOW_HF_FALLBACK=1`.\n\n"
           "Stages: **installer** (protected runtime, complete hash-pinned dependency locks) -> **static** (package/Flash audit, inventory, configuration, backend registry, CPU structured-output stack) -> "
           "**engine** (FP16 inference, chat, complete, structured output with the pinned xgrammar backend, lifecycle, TRITON_ATTN) -> **server** (`vllm serve`, OpenAI API, streaming, restart, cleanup) -> **summary**.\n\n"
           "The acceptance ZIP is written even when a stage fails; the last cell then fails the notebook unless every stage is `PASS`. Only `PASS` passes: `WARNING`, `SKIPPED`, `NOT_RUN`, `ERROR`, `FAIL` do not."),
        code("# ---- cell 1: locate, verify and extract the ATTACHED release bundle (bundle.py is embedded verbatim below) ----\n" + BUNDLE_PY +
             "\n\n# ---- notebook configuration (compiled in when the notebook is generated) ----\n"
             f"EXPECTED_WHEEL_NAME = {WHEEL_NAME!r}\nEXPECTED_WHEEL_SHA256 = {wheel_sha256!r}\nEXPECTED_BUNDLE_ZIP_SHA256 = {bundle_zip_sha256!r}\n"
             "OUT = pathlib.Path('/kaggle/working/stokastisk-acceptance'); OUT.mkdir(parents=True, exist_ok=True)\nos.environ['STOKASTISK_OUT'] = str(OUT)\n"
             "PROV = locate_and_verify(input_root='/kaggle/input', work_dir='/kaggle/working/stokastisk-release', expected_wheel_name=EXPECTED_WHEEL_NAME,\n"
             "                         expected_wheel_sha256=EXPECTED_WHEEL_SHA256, expected_bundle_zip_sha256=EXPECTED_BUNDLE_ZIP_SHA256, fallback=None)\n"
             "ROOT = pathlib.Path(PROV['release_root']); SCRIPTS = ROOT / 'scripts' / 'kaggle'\n(OUT / 'provenance.json').write_text(json.dumps(PROV, indent=2))\nprint('release root:', ROOT)\n"),
        code("# ---- cell 2: runtime snapshot ----\nimport subprocess, sys\nprint(sys.version)\nprint(subprocess.run('nvidia-smi', capture_output=True, text=True).stdout)\n"
             "print(subprocess.run([sys.executable, '-m', 'pip', '--version'], capture_output=True, text=True).stdout)\n"),
        code("# ---- cell 3: helper that runs one stage as its own process and keeps its complete output ----\nimport subprocess, sys, os, pathlib\n"
             "def run_stage(name, *cmd, log=None):\n"
             "    env = dict(os.environ, STOKASTISK_OUT=str(OUT), PYTHONUNBUFFERED='1', PYTHONPATH=str(SCRIPTS))\n"
             "    logp = OUT / (log or f'15c_{name}.log')\n"
             "    with open(logp, 'w') as lf:\n"
             "        p = subprocess.Popen([sys.executable, *cmd], cwd=str(SCRIPTS), env=env, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1)\n"
             "        for line in p.stdout:\n            sys.stdout.write(line); lf.write(line)\n        p.wait()\n"
             "    print(f'--- stage {name}: exit code {p.returncode}  (log {logp})'); return p.returncode\n"),
        code("# ---- cell 4: install (never replaces torch/triton/CUDA/NCCL; exits non-zero on any failed check) ----\n"
             "RC = {}\nRC['installer'] = run_stage('installer', '02_install_integrated.py', str(ROOT), log='02_install_integrated.log')\n"),
        code("# ---- cell 5: static audit ----\nRC['static'] = run_stage('static', 'candidate_15_integration.py', 'static') if RC['installer'] == 0 else print('installer failed: static NOT_RUN') or 'NOT_RUN'\n"),
        code("# ---- cell 6: engine (GPU inference) ----\nRC['engine'] = run_stage('engine', 'candidate_15_integration.py', 'engine') if RC['installer'] == 0 else print('installer failed: engine NOT_RUN') or 'NOT_RUN'\n"),
        code("# ---- cell 7: HTTP server ----\nRC['server'] = run_stage('server', 'candidate_15_integration.py', 'server') if RC['installer'] == 0 else print('installer failed: server NOT_RUN') or 'NOT_RUN'\n"),
        code("# ---- cell 8: summary (a missing stage report is NOT_RUN, never a pass) ----\nif RC['installer'] == 0:\n    RC['summary'] = run_stage('summary', 'candidate_15_integration.py', 'summary')\n"
             "else:\n    (OUT / '15c_summary.json').write_text(json.dumps({'stage': '15c_summary', 'status': 'NOT_RUN', 'detail': 'installer failed', 'rc': RC}, indent=2)); RC['summary'] = 'NOT_RUN'\n"
             "print(RC)\n"),
        code("# ---- cell 9: ALWAYS build the acceptance archive, then fail the notebook unless every stage is PASS ----\nimport zipfile, hashlib, shutil\n"
             "ZIP = pathlib.Path('/kaggle/working/15-stokastisk-build3-acceptance.zip'); ZIP.unlink(missing_ok=True)\n"
             "(OUT / 'RUN-RC.json').write_text(json.dumps({k: str(v) for k, v in RC.items()}, indent=1))\n"
             "sums = []\n"
             "with zipfile.ZipFile(ZIP, 'w', zipfile.ZIP_DEFLATED) as z:\n"
             "    for p in sorted(OUT.rglob('*')):\n        if p.is_file():\n"
             "            z.write(p, 'stokastisk-acceptance/' + p.relative_to(OUT).as_posix()); sums.append(f'{sha256_file(p)}  stokastisk-acceptance/{p.relative_to(OUT).as_posix()}')\n"
             "    z.writestr('stokastisk-acceptance/ACCEPTANCE-SHA256SUMS', '\\n'.join(sums) + '\\n')\n"
             "print('acceptance archive:', ZIP, ZIP.stat().st_size, 'bytes, sha256', sha256_file(ZIP))\n"
             "summ = json.loads((OUT / '15c_summary.json').read_text()) if (OUT / '15c_summary.json').is_file() else {'status': 'NOT_RUN'}\n"
             "modes = summ.get('modes', {}); print('stage results:', modes); print('OVERALL:', summ.get('status'))\n"
             "assert summ.get('status') == 'PASS' and all(v == 'PASS' for v in modes.values()) and set(modes) >= {'static', 'engine', 'server', 'installer'}, f'build 3 is NOT verified: {summ.get(\"status\")} {modes}'\n"
             "print('ALL STAGES PASS for', EXPECTED_WHEEL_NAME)\n"),
    ]
    nb.cells = cells
    return nb


def main(argv):
    out, wsha = pathlib.Path(argv[0]), argv[1]
    bsha = argv[2] if len(argv) > 2 else None
    nb = build(wsha, bsha)
    nbformat.validate(nb)
    out.parent.mkdir(parents=True, exist_ok=True)
    nbformat.write(nb, out)
    print(out)


if __name__ == "__main__":
    main(sys.argv[1:])
