#!/usr/bin/env python3.13
"""Install the Stokastisk 0.1.0 integrated wheel (build 3) on Kaggle 2 x Tesla T4 WITHOUT touching the protected runtime.

usage: 02_install_integrated.py [BUNDLE_DIR] [--dry-run] [--with-models]        exit code 0 only if every check PASSED

What it does, in this order (nothing is installed until every pre-flight check has passed):
 1. verifies the bundle (SHA256SUMS over every file, RELEASE-MANIFEST, exactly one wheel with the manifest's name/size/SHA256, build tag 3);
 2. records the BEFORE inventory with the single inventory implementation (`_inventory.py`: pip list --format=json, PEP 503 names) and the Kaggle runtime
    (Python, torch, Triton, CUDA, NCCL, GPUs);
 3. audits the wheel itself: tags, exactly the packages `stokastisk` + `vllm`, bundled vllm dist-info, and the strict Flash policy (`prohibited.py`: path hits in .py/.pyc/.so/...,
    native markers, bindings) - a wheel containing a Flash-family file is refused;
 4. plans each dependency lock (`requirements/{core,structured-output,serving}.lock`, plus `models.lock` with --with-models): a package absent from Kaggle is installed at its
    pinned version; one already present at the pinned version is kept; one present at ANOTHER version is a CONFLICT (the baseline is protected: nothing is installed);
 5. downloads the pins with `pip download --no-deps --require-hashes --only-binary=:all:` (no resolver), then installs the wheel and the pins with `--no-deps --no-index`;
 6. records the AFTER inventory and proves: no pre-installed package changed or disappeared, only the planned packages were added, torch/triton/CUDA/NCCL are identical;
 7. runs `pip check` before and after: ANY new conflict is a FAIL (there is no tolerated conflict in build 3 - the xgrammar/transformers one is resolved by the lock);
 8. imports every module the advertised features need (vLLM LLM, serve, structured output, SDK), checks the TRITON_ATTN backend resolves, that the Flash import guard is active,
    that the installed files are byte-identical to the wheel and that the installed vllm tree contains no Flash-family file (source, bytecode or native).
Evidence: `02_install_integrated.json`, `02_packages_before.json`, `02_packages_after.json`, `02_pip_list_before/after.json`, `02_install_plan.json` in $STOKASTISK_OUT/prereq-install/."""
import argparse, hashlib, importlib, importlib.metadata as md, json, os, pathlib, platform, re, subprocess, sys, tempfile, time, traceback, zipfile

HERE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import _inventory as inv                                   # noqa: E402
import prohibited                                          # noqa: E402  (same bytes as sdk/stokastisk/prohibited.py; checked by tests)
from bundle import BundleError, verify_release             # noqa: E402

VERSION, BUILD = "0.1.0", "3"
WHEEL_NAME = f"stokastisk-{VERSION}-{BUILD}-cp313-cp313-linux_x86_64.whl"
LOCK = {"stokastisk": VERSION, "vllm": "0.25.1", "torch": "2.11.0+cu130", "triton": "3.6.0", "cuda": "13.0"}
PROFILES = ("core", "structured-output", "serving")
OWN = ("stokastisk", "vllm")
# distributions that must keep exactly the version they have: torch/triton exactly, CUDA/NCCL runtimes by prefix (checked on the whole inventory by diff())
PROTECTED_PREFIXES = ("torch", "triton", "nvidia-", "cuda-", "numpy", "transformers", "tokenizers", "pydantic", "fastapi", "starlette", "protobuf")
# NOTE: lmformatenforcer.integrations.vllm is deliberately NOT imported: lm-format-enforcer 0.11.3 imports vllm.transformers_utils.tokenizer.MistralTokenizer, which does not exist in
# vLLM 0.25.1. Only the lm-format-enforcer *backend* uses that adapter and Stokastisk pins xgrammar; vLLM itself needs just the top-level package (see dependencies.py).
IMPORTS = ["transformers", "tokenizers", "vllm", "stokastisk", "stokastisk.compat", "stokastisk.serving", "stokastisk.structured", "stokastisk.dependencies", "xgrammar", "tvm_ffi", "llguidance",
           "lmformatenforcer", "outlines_core", "diskcache", "interegular", "fastapi_cli", "sse_starlette", "pydantic_settings",
           "model_hosting_container_standards.sagemaker", "prometheus_fastapi_instrumentator", "vllm._C_stable_libtorch", "vllm._moe_C_stable_libtorch", "vllm.cumem_allocator",
           "vllm.spinloop", "vllm.fs_io_C", "vllm.engine.arg_utils", "vllm.v1.attention.backends.registry", "vllm.v1.attention.selector", "vllm.v1.attention.backends.triton_attn",
           "vllm.v1.engine.llm_engine", "vllm.v1.executor.multiproc_executor", "vllm.distributed.device_communicators.cuda_communicator", "vllm.model_executor.models.qwen2",
           "vllm.entrypoints.llm", "vllm.entrypoints.cli.main", "vllm.entrypoints.openai.api_server", "vllm.v1.structured_output.backend_xgrammar",
           "vllm.v1.structured_output.backend_lm_format_enforcer", "vllm.t4_backend_policy", "vllm.utils.t4_prohibited_stubs"]


# ---------------------------------------------------------------- pure functions (unit-tested in tests/test_installer_integrated.py)
def parse_lock(text):
    """pip --require-hashes lock -> {canonical name: {'version', 'hashes', 'block'}}."""
    out, cur = {}, None
    for raw in text.splitlines():
        if not raw.strip() or raw.lstrip().startswith("#"):
            continue
        if not raw.startswith(" "):
            m = re.match(r"^([A-Za-z0-9_.\-]+)==([^\s\\]+)", raw)
            if not m:
                raise ValueError(f"malformed lock line: {raw!r}")
            cur = inv.canon(m.group(1)); out[cur] = {"version": m.group(2), "hashes": [], "block": [raw.rstrip()]}
        else:
            h = re.search(r"--hash=sha256:([0-9a-f]{64})", raw)
            if not h or cur is None:
                raise ValueError(f"malformed hash line: {raw!r}")
            out[cur]["hashes"].append(h.group(1)); out[cur]["block"].append(raw.rstrip())
    for n, v in out.items():
        if not v["hashes"]:
            raise ValueError(f"lock entry {n} has no hashes")
        v["block"] = "\n".join(v["block"])
    return out


def make_plan(installed, locks):
    """installed: canonical pip-list map. locks: {profile: parse_lock()}. -> install/keep/conflicts (the baseline is protected: a different version is a conflict)."""
    plan = {"install": {}, "keep": {}, "conflicts": [], "by_profile": {}}
    for prof, lock in locks.items():
        plan["by_profile"][prof] = sorted(lock)
        for n, e in lock.items():
            if n in plan["install"] and plan["install"][n] != e["version"]:
                plan["conflicts"].append(f"{n}: pinned {plan['install'][n]} and {e['version']} by different locks")
            have = installed.get(n)
            if have is None:
                plan["install"][n] = e["version"]
            elif have == e["version"]:
                plan["keep"][n] = have
            else:
                plan["conflicts"].append(f"{n}: Kaggle has {have}, the lock pins {e['version']} - the baseline is protected, nothing is changed")
    return plan


def check_baseline(installed):
    errs = []
    if installed.get("torch") != LOCK["torch"]:
        errs.append(f"torch {installed.get('torch')} != {LOCK['torch']}")
    if installed.get("triton") != LOCK["triton"]:
        errs.append(f"triton {installed.get('triton')} != {LOCK['triton']}")
    tv = installed.get("transformers")
    if not tv or not re.match(r"^5\.(\d+)\.", tv) or tuple(int(x) for x in tv.split(".")[:3] if x.isdigit()) < (5, 5, 3):
        errs.append(f"transformers {tv!r}: vLLM 0.25.1 needs >=5.5.3,<6")
    bad = sorted(set(map(inv.canon, prohibited.PROHIBITED_DISTS)) & set(map(inv.canon, installed)))
    if bad:
        errs.append(f"prohibited distributions present: {bad}")
    return errs


def classify_pip_check(before_lines, after_lines):
    """New `pip check` lines after the install. Build 3 tolerates NONE: pre-existing Kaggle lines (identical before and after) are the only ones ignored."""
    b = set(before_lines)
    new = [l for l in after_lines if l not in b]
    own = [l for l in new if re.match(r"(vllm|stokastisk) ", l)]
    return {"new": new, "own": own, "other": [l for l in new if l not in own], "preexisting_unchanged": sorted(b & set(after_lines))}


def inspect_wheel(path):
    z = zipfile.ZipFile(path)
    names = z.namelist()
    scan = prohibited.scan_members(names, z.read)
    di = next(n.split("/")[0] for n in names if n.endswith(".dist-info/METADATA") and ".data/" not in n)
    meta, wheel = z.read(di + "/METADATA").decode(), z.read(di + "/WHEEL").decode()
    tops = sorted({n.split("/")[0] for n in names if ".dist-info/" not in n and ".data/" not in n})
    return {"scan": {k: v for k, v in scan.items() if k not in ("references", "allowed_flash_named")}, "scan_reference_files": len(scan["references"]), "metadata": meta, "top_level": tops,
            "vllm_dist_info_bundled": any(n.endswith("vllm-0.25.1.dist-info/METADATA") for n in names), "files": len(names),
            "tag": re.search(r"^Tag: (.*)$", wheel, re.M).group(1), "build": (re.search(r"^Build: (.*)$", wheel, re.M) or [None, None])[1]}


def pip(*a, **kw):
    return subprocess.run([sys.executable, "-m", "pip", *a], capture_output=True, text=True, **kw)


def sha(path):
    return hashlib.sha256(pathlib.Path(path).read_bytes()).hexdigest()


# ---------------------------------------------------------------- driver
def main(argv=None):
    ap = argparse.ArgumentParser(); ap.add_argument("bundle", nargs="?"); ap.add_argument("--dry-run", action="store_true"); ap.add_argument("--with-models", action="store_true")
    a = ap.parse_args(argv)
    from _common import OUT, check, finish, skipped, warn
    out = OUT / "prereq-install"; out.mkdir(parents=True, exist_ok=True)
    R = []; rep = {"stage": "02_install_integrated", "started": time.strftime("%FT%T%z"), "dry_run": a.dry_run, "installer": "02_install_integrated.py", "installer_sha256": sha(__file__), "lock": LOCK,
                   "version": VERSION, "build": BUILD, "wheel": WHEEL_NAME}
    t0 = time.time()
    root = pathlib.Path(a.bundle or os.environ.get("STOKASTISK_BUNDLE") or HERE.parents[1])
    rep["bundle"] = str(root)

    def done():
        """Write the installer report (prereq-install/02_install_integrated.json) and exit: 0 only for PASS (or a dry run without failures)."""
        import _common
        rep["seconds"] = round(time.time() - t0, 1)
        _common.OUT = out
        try:
            finish("02_install_integrated", R, rep, script_file=__file__)
        except SystemExit as e:
            if a.dry_run and not any(r["status"] in ("FAIL", "ERROR") for r in R):
                sys.exit(0)
            raise e

    try:
        prov = verify_release(root, expected_wheel_name=WHEEL_NAME, expected_wheel_sha256=os.environ.get("STOKASTISK_EXPECTED_WHEEL_SHA256") or None, expected_version=VERSION, expected_build=BUILD)
        check(R, "bundle verified: every file against SHA256SUMS, manifest, exactly one wheel with the manifest's name/size/SHA256", True, f"{prov['files_verified']} files; wheel sha256 {prov['wheel_sha256']}")
        rep.update(wheel_sha256=prov["wheel_sha256"], wheel_size=prov["wheel_size"], manifest_sha256=prov["manifest_sha256"], release_root=prov["release_root"])
    except BundleError as e:
        check(R, "bundle verified", False, e); done()
    wheel = root / "wheels" / WHEEL_NAME
    check(R, "python 3.13", sys.version_info[:2] == (3, 13), platform.python_version())
    before = inv.snapshot(); rep["before_count"] = before["count"]
    (out / "02_packages_before.json").write_text(json.dumps(before["packages"], indent=0, sort_keys=True))
    (out / "02_pip_list_before.json").write_text(json.dumps(before, indent=1, sort_keys=True))
    errs = check_baseline(before["packages"])
    check(R, "baseline protected (torch/triton exact, transformers>=5.5.3,<6, no prohibited distributions)", not errs,
          errs or {k: before["packages"].get(k) for k in ("torch", "triton", "transformers", "numpy", "fastapi", "starlette", "protobuf", "nvidia-cuda-nvcc", "nvidia-nvvm")})
    try:
        import torch
        check(R, "PyTorch CUDA build is 13.0", torch.version.cuda == "13.0", torch.version.cuda)
        n = torch.cuda.device_count(); caps = [torch.cuda.get_device_capability(i) for i in range(n)]; names = [torch.cuda.get_device_name(i) for i in range(n)]
        ok = n == 2 and all(c == (7, 5) for c in caps) and all("T4" in x for x in names)
        if ok:
            check(R, "two Tesla T4 GPUs (SM75)", True, f"{names} {caps}")
        elif a.dry_run:
            skipped(R, "two Tesla T4 GPUs (SM75)", f"dry run on a machine with {n} GPU(s) {caps}: not enforced")
        else:
            check(R, "two Tesla T4 GPUs (SM75)", False, f"{n} gpu(s) {names} {caps}")
    except Exception as e:  # noqa: BLE001
        (skipped if a.dry_run else lambda r, n_, d: check(r, n_, False, d))(R, "torch/CUDA probe", repr(e))
    wi = inspect_wheel(wheel); rep["wheel_inspection"] = {k: v for k, v in wi.items() if k != "metadata"}
    check(R, f"wheel tags: cp313-cp313-linux_x86_64, build tag {BUILD}", wi["tag"] == "cp313-cp313-linux_x86_64" and wi["build"] == BUILD, (wi["tag"], wi["build"]))
    check(R, "wheel has exactly the packages stokastisk + vllm (no duplicate vLLM tree)", wi["top_level"] == ["stokastisk", "vllm"], wi["top_level"])
    check(R, "wheel registers the vllm 0.25.1 distribution (vLLM calls importlib.metadata.version('vllm'))", wi["vllm_dist_info_bundled"])
    sc = wi["scan"]
    check(R, "wheel contains no FlashAttention/FlashInfer/FlashMLA file (source, bytecode, native, binding) and no __pycache__", sc["clean"] and not sc["pycache_dirs"],
          {"path_hits": sc["path_hits"][:5], "native_hits": list(sc["native_hits"])[:5], "binding_hits": sc["binding_hits"][:5], "pycache": sc["pycache_dirs"][:3]} if not (sc["clean"] and not sc["pycache_dirs"]) else
          f"{sc['files']} files, {sc['py']} .py, {sc['native']} native, 0 prohibited ({wi['scan_reference_files']} informational references to Flash package names)")
    profiles = list(PROFILES) + (["models"] if a.with_models else [])
    try:
        locks = {p: parse_lock((root / "requirements" / f"{p}.lock").read_text()) for p in profiles}
        check(R, "every lock entry has sha256 hashes", all(e["hashes"] for l in locks.values() for e in l.values()), {p: len(l) for p, l in locks.items()})
    except (OSError, ValueError) as e:
        check(R, "dependency locks readable", False, e); done()
    plan = make_plan(before["packages"], locks); rep["plan"] = plan
    for c in plan["conflicts"]:
        check(R, "plan: " + c, False)
    check(R, "dependency plan has no conflicts (nothing is installed if this fails)", not plan["conflicts"], {"install": plan["install"], "keep": plan["keep"]})
    (out / "02_install_plan.json").write_text(json.dumps(plan, indent=1))
    if any(r["status"] in ("FAIL", "ERROR") for r in R):
        done()
    tmp = pathlib.Path(tempfile.mkdtemp(prefix="stk-pins-")); dl = 0
    if plan["install"]:
        req = tmp / "pins.txt"
        req.write_text("\n".join(l for p in profiles for n, e in locks[p].items() if n in plan["install"] for l in [e["block"]]) + "\n")
        r = pip("download", "--no-deps", "--require-hashes", "--only-binary=:all:", "-r", str(req), "-d", str(tmp)); rep["pip_download_rc"] = r.returncode
        check(R, "download the pinned wheels with --require-hashes --no-deps --only-binary (no resolver)", r.returncode == 0,
              r.stderr[-400:] if r.returncode else f"{len(plan['install'])} pins hash-verified")
        dl = sum(p.stat().st_size for p in tmp.glob("*.whl")); rep["downloaded_bytes"] = dl
    pc_before = pip("check").stdout.strip().splitlines(); rep["pip_check_before"] = pc_before
    if a.dry_run:
        rep["would_install"] = {"wheel": WHEEL_NAME, "pins": plan["install"]}
        skipped(R, "installation", "dry run"); done()
    if any(r["status"] in ("FAIL", "ERROR") for r in R):
        done()
    r = pip("install", "--force-reinstall", "--no-deps", "--no-index", str(wheel)); check(R, "install the integrated wheel --force-reinstall --no-deps --no-index", r.returncode == 0, r.stderr[-500:])
    if plan["install"]:
        r = pip("install", "--no-deps", "--no-index", "--find-links", str(tmp), *[f"{n}=={v}" for n, v in plan["install"].items()])
        check(R, "install the locked dependency pins --no-deps --no-index", r.returncode == 0, r.stderr[-500:])
    importlib.invalidate_caches()
    after = inv.snapshot(); (out / "02_packages_after.json").write_text(json.dumps(after["packages"], indent=0, sort_keys=True)); (out / "02_pip_list_after.json").write_text(json.dumps(after, indent=1, sort_keys=True))
    d = inv.diff(before["packages"], after["packages"], expected_new=list(plan["install"]), own=OWN); rep["diff"] = d
    check(R, "no pre-installed package (torch, triton, CUDA, NCCL, ...) changed or removed", not d["changed"] and not d["removed"], {"changed": d["changed"], "removed": d["removed"]} if (d["changed"] or d["removed"]) else "identical")
    check(R, "only the planned packages were added (and all of them)", not d["unexpected_added"] and not d["expected_but_absent"], {"added": d["added"], "unexpected": d["unexpected_added"], "absent": d["expected_but_absent"]})
    errs = check_baseline(after["packages"]); check(R, "protected baseline still intact", not errs, errs or "ok")
    pc = pip("check").stdout.strip().splitlines(); rep["pip_check_after"] = pc; cls = classify_pip_check(pc_before, pc); rep["pip_check_new"] = cls
    check(R, "pip check: no new conflict involving stokastisk/vllm", not cls["own"], cls["own"] or "none")
    check(R, "pip check: no new conflict at all (build 3 tolerates none)", not cls["other"], cls["other"] or f"none new; {len(cls['preexisting_unchanged'])} pre-existing Kaggle lines unchanged")
    tb = {}
    for m in IMPORTS:
        try:
            importlib.import_module(m)
        except BaseException:  # noqa: BLE001
            tb[m] = traceback.format_exc()[-1200:]
    check(R, "imports (full traceback captured on failure)", not tb, tb or f"{len(IMPORTS)} modules")
    import stokastisk, vllm
    from stokastisk import dependencies
    miss = dependencies.missing_for(*PROFILES); rep["dependency_import_names_missing"] = miss
    check(R, "every import name of the core + structured-output + serving profiles is importable", not miss, miss or "complete")
    try:
        from vllm.v1.attention.backends.registry import AttentionBackendEnum
        c = AttentionBackendEnum.TRITON_ATTN.get_class(); check(R, "TRITON_ATTN backend class resolves", c.__name__ == "TritonAttentionBackend", c.__name__)
        blocked = []
        for nme in ("FLASH_ATTN", "FLASHINFER", "FLASHMLA"):
            try:
                getattr(AttentionBackendEnum, nme).get_class(); blocked.append(nme + ":ALLOWED")
            except ImportError:
                pass
        check(R, "FLASH_ATTN / FLASHINFER / FLASHMLA backends are refused with an explicit error", not blocked, blocked or "refused")
    except BaseException:  # noqa: BLE001
        check(R, "backend registry checks", False, traceback.format_exc()[-800:])
    guard = [m for m in prohibited.BLOCKED_IMPORT_NAMES if sys.modules.get(m, "x") is not None]
    check(R, "Flash import guard is active (import flashinfer / flash_attn ... fails explicitly)", not guard, guard or "all blocked")
    try:
        sv, vv = md.version("stokastisk"), md.version("vllm")
    except md.PackageNotFoundError as e:
        sv = vv = repr(e)
    check(R, "versions: stokastisk 0.1.0 (module+distribution), vllm 0.25.1 (module+distribution)", stokastisk.__version__ == sv == LOCK["stokastisk"] and vllm.__version__ == vv == LOCK["vllm"],
          {"stokastisk": [stokastisk.__version__, sv], "vllm": [vllm.__version__, vv]})
    check(R, "single vLLM package tree on sys.path", len(list(vllm.__path__)) == 1, list(vllm.__path__))
    z = zipfile.ZipFile(wheel); site = pathlib.Path(vllm.__file__).resolve().parent.parent; badf = []
    for nme in z.namelist():
        if nme.startswith(("stokastisk/", "vllm/")) and nme.endswith((".py", ".so")):
            f = site / nme
            if not f.is_file() or hashlib.sha256(f.read_bytes()).hexdigest() != hashlib.sha256(z.read(nme)).hexdigest():
                badf.append(nme)
    check(R, "installed .py/.so files are byte-identical to the wheel", not badf, badf[:5] or "identical")
    ins = prohibited.scan_tree(site / "vllm"); rep["installed_tree_scan"] = {k: v for k, v in ins.items() if k not in ("references", "allowed_flash_named")}
    check(R, "installed vllm tree: no Flash-family source, bytecode or native file", ins["clean"], ins["path_hits"][:6] or ins["native_hits"] or ins["binding_hits"] or f"{ins['files']} files, 0 prohibited")
    pr = stokastisk.prohibited_report(); rep["prohibited"] = pr
    check(R, "stokastisk.prohibited_report() is clean (distributions, importable modules, files)", pr["clean"], pr)
    try:
        cfg = stokastisk.StokastiskConfig(); check(R, "SDK defaults: TRITON_ATTN + pinned xgrammar", cfg.attention_backend == "TRITON_ATTN" and cfg.structured_output_backend == "xgrammar", (cfg.attention_backend, cfg.structured_output_backend))
        for bad in ("FLASH_ATTN", "FLASHINFER"):
            try:
                stokastisk.StokastiskConfig(attention_backend=bad); check(R, f"SDK rejects {bad}", False, "accepted")
            except stokastisk.ConfigurationError:
                check(R, f"SDK rejects {bad}", True)
    except BaseException:  # noqa: BLE001
        check(R, "SDK configuration checks", False, traceback.format_exc()[-600:])
    rep["downloaded_bytes"] = dl
    done()


if __name__ == "__main__":
    main()
