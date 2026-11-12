#!/usr/bin/env python3
"""Stage 02 (integrated wheel): deterministic installer for the single Stokastisk 0.1.0 integrated wheel (SDK + bundled vLLM 0.25.1, Triton attention, SM75).

Not the legacy installer: that one requires exactly TWO wheels (`stokastisk` + `vllm`) and refuses this bundle; this one requires exactly ONE.
 * No dependency resolver ever runs. The wheel is installed `--no-deps --no-index`; the few packages absent from the Kaggle image are the exact pins in
   requirements/kaggle-t4-core.hashes.txt, downloaded with `--require-hashes --no-deps` and installed `--no-deps --no-index`.
 * Every decision is taken before anything is installed; any unsatisfied requirement, protected-package mismatch or hash mismatch aborts untouched.
 * A legacy two-wheel install (`vllm` with a RECORD file) is removed first -- only the two distributions `stokastisk` and `vllm`, never anything else.
 * torch / triton / CUDA / NCCL and every other preinstalled distribution must be byte-for-byte the same version afterwards, otherwise FAIL.
 * `pip check` is compared before/after. The ONE documented known conflict (xgrammar 0.2.5.post1 metadata says transformers<5; see
   docs/DEPENDENCY_COMPATIBILITY.md) is reported as WARNING, never hidden and never counted as clean; any other new conflict FAILS.
Every check is PASS / FAIL / WARNING / SKIPPED; the exit code is non-zero if any required check failed.
usage: 02_install_integrated.py [BUNDLE_DIR] [--dry-run]"""
import argparse, glob, hashlib, importlib.metadata as md, json, os, pathlib, platform, re, subprocess, sys, tempfile, time, traceback, zipfile

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
LOCK = {"stokastisk": "0.1.0", "vllm": "0.25.1", "torch": "2.11.0+cu130", "triton": "3.6.0", "cuda": "13.0"}
WHEEL_NAME = "stokastisk-0.1.0-2-cp313-cp313-linux_x86_64.whl"
PROTECTED_EXACT = {"torch": LOCK["torch"], "triton": LOCK["triton"]}
OWN = ("stokastisk", "vllm")
PROHIBITED = {"flash-attn", "flashinfer", "flashinfer-python", "flashinfer-cubin", "flashinfer-jit-cache", "vllm-flash-attn", "cuda-tile", "nvidia-cuda-tileiras",
              "humming-kernels", "nvidia-cutlass-dsl", "nvidia-cutlass-dsl-libs-base", "nvidia-cutlass-dsl-libs-cu13", "quack-kernels", "tokenspeed-mla", "tokenspeed-triton"}
PROHIBITED_WHEEL_PATHS = ("vllm_flash_attn", "flashinfer", "_vllm_fa", "flashmla")
# the single documented, functionally verified metadata conflict (docs/DEPENDENCY_COMPATIBILITY.md)
KNOWN_PIP_CHECK = re.compile(r"xgrammar 0\.2\.5\.post1 has requirement transformers<5,>=4\.38\.0, but you have transformers \S+\.")
norm = lambda s: re.sub(r"[-_.]+", "-", s).lower()  # noqa: E731


# ---------------- pure functions (unit-tested in tests/test_installer_integrated.py) ----------------
def snapshot():
    return {norm(d.metadata["Name"]): d.version for d in md.distributions() if d.metadata["Name"]}


def parse_pins(text):
    pins = {}
    for line in text.splitlines():
        line = line.split("#")[0].strip()
        if line:
            n, _, v = line.partition("==")
            pins[norm(n)] = v.strip()
    return pins


def parse_hash_blocks(text):
    """requirements file with `--hash` continuation lines -> {normalised name: full requirement block}."""
    blocks, cur, name = {}, [], None
    for raw in text.splitlines():
        if raw.startswith("#") or not raw.strip():
            continue
        if not raw.startswith(" "):
            if name: blocks[name] = "\n".join(cur)
            name = norm(raw.split("==")[0]); cur = [raw]
        else:
            cur.append(raw)
    if name: blocks[name] = "\n".join(cur)
    return blocks


def core_requirements(metadata_text):
    """Non-extra Requires-Dist lines of a wheel's METADATA -> packaging Requirement list."""
    from packaging.requirements import Requirement
    out = []
    for l in metadata_text.splitlines():
        if l.startswith("Requires-Dist: "):
            r = Requirement(l[15:])
            if r.marker is None or (r.marker.evaluate({"extra": ""}) and "extra" not in str(r.marker)):
                out.append(r)
    return out


def make_plan(installed, pins, requirements):
    from packaging.version import Version
    plan = {"install": {}, "satisfied": [], "compatible_kept": [], "conflicts": []}
    for n, v in pins.items():
        if n in PROHIBITED:
            plan["conflicts"].append(f"core pin {n} is prohibited")
        elif installed.get(n) == v:
            plan["satisfied"].append(f"{n}=={v}")
        elif n not in installed:
            plan["install"][n] = v
        else:
            plan["compatible_kept"].append(f"{n} {installed[n]} (pin {v}; kept, checked against the requirement)")
    for r in requirements:
        n = norm(r.name)
        if n in PROHIBITED:
            plan["conflicts"].append(f"wheel requires prohibited package {n}")
        have = installed.get(n) or (pins.get(n) if n in plan["install"] else None)
        if have is None:
            plan["conflicts"].append(f"requirement {r} is not installed and not in the core pin list")
        elif r.specifier and not r.specifier.contains(Version(have), prereleases=True):
            plan["conflicts"].append(f"requirement {r} not satisfied by {have}")
    return plan


def check_baseline(installed):
    from packaging.version import Version
    errs = [f"{n} must be {v}, found {installed.get(n)}" for n, v in PROTECTED_EXACT.items() if installed.get(n) != v]
    tv = installed.get("transformers")
    if tv is None:
        errs.append("transformers not installed")
    elif Version(tv) < Version("5.5.3") or Version(tv).major >= 6:
        errs.append(f"transformers {tv} rejected: vLLM 0.25.1 needs >=5.5.3,<6")
    bad = sorted(PROHIBITED & set(installed))
    if bad:
        errs.append(f"prohibited packages present: {bad}")
    return errs


def diff_snapshots(before, after, expected_new, allowed_change=OWN):
    ok_names = {norm(x) for x in expected_new}
    changed = {n: (before[n], after.get(n)) for n in before if after.get(n) != before[n] and n not in allowed_change}
    added = sorted(set(after) - set(before))
    return {"preinstalled_changed_or_removed": changed, "added": added, "unexpected_added": sorted(set(added) - ok_names - set(allowed_change)),
            "ok": not changed and not (set(added) - ok_names - set(allowed_change))}


def classify_pip_check(before_lines, after_lines):
    """Split the NEW `pip check` lines into: known documented warning / conflicts involving our own distributions / any other new conflict."""
    new = [l for l in after_lines if l not in set(before_lines)]
    known = [l for l in new if KNOWN_PIP_CHECK.fullmatch(l)]
    own = [l for l in new if re.match(r"(vllm|stokastisk) ", l)]
    other = [l for l in new if l not in known and l not in own]
    return {"new": new, "known": known, "own": own, "other": other}


def inspect_wheel(path):
    z = zipfile.ZipFile(path); names = z.namelist()
    bad = [n for n in names if any(k in n.lower() for k in PROHIBITED_WHEEL_PATHS) and not n.endswith(".py")] + [n for n in names if "vllm_flash_attn" in n]
    di = next(n.split("/")[0] for n in names if n.endswith(".dist-info/METADATA") and "/" in n and ".data/" not in n)
    meta = z.read(di + "/METADATA").decode(); wheel = z.read(di + "/WHEEL").decode()
    tops = sorted({n.split("/")[0] for n in names if not n.endswith(".dist-info/") and ".dist-info/" not in n and ".data/" not in n})
    return {"prohibited": bad, "metadata": meta, "wheel": wheel, "top_level": tops,
            "vllm_dist_info_bundled": any(n.endswith("vllm-0.25.1.dist-info/METADATA") for n in names),
            "tag": re.search(r"^Tag: (.*)$", wheel, re.M).group(1), "build": (re.search(r"^Build: (.*)$", wheel, re.M) or [None, None])[1]}


def legacy_two_wheel_installed():
    """True if `vllm` is installed from the legacy standalone wheel (it has a RECORD; the integrated wheel's bundled vllm dist-info has none)."""
    try:
        return md.distribution("vllm").read_text("RECORD") is not None
    except md.PackageNotFoundError:
        return False


# ---------------- driver ----------------
def find_bundle(arg):
    cands = [arg] if arg else []
    cands += [os.environ.get("STOKASTISK_BUNDLE", "")]
    cands += [str(pathlib.Path(p).parent.parent) for p in glob.glob("/kaggle/input/**/manifests/SHA256SUMS", recursive=True)]
    cands += [str(pathlib.Path(__file__).resolve().parents[2])]
    for c in cands:
        if c and (pathlib.Path(c) / "manifests" / "SHA256SUMS").is_file():
            return pathlib.Path(c)
    sys.exit("bundle not found: pass its directory (containing manifests/SHA256SUMS)")


def pip(*a, **kw):
    return subprocess.run([sys.executable, "-m", "pip", *a], capture_output=True, text=True, **kw)


def main(argv=None):
    ap = argparse.ArgumentParser(); ap.add_argument("bundle", nargs="?"); ap.add_argument("--dry-run", action="store_true"); a = ap.parse_args(argv)
    from _common import OUT, check, finish, skipped, warn
    R = []; rep = {"started": time.strftime("%FT%T%z"), "dry_run": a.dry_run, "installer": "02_install_integrated.py", "lock": LOCK}; t0 = time.time()
    b = find_bundle(a.bundle); rep["bundle"] = str(b)
    sums = [l.split(None, 1) for l in (b / "manifests" / "SHA256SUMS").read_text().splitlines() if l.strip()]
    bad = [n for h, n in sums if not (b / n.strip().lstrip("*")).is_file() or hashlib.sha256((b / n.strip().lstrip("*")).read_bytes()).hexdigest() != h]
    check(R, "SHA256SUMS verified before any install", not bad, bad or f"{len(sums)} files match")
    if bad: finish("02_install_integrated", R, rep)
    manifest = json.loads((b / "manifests" / "RELEASE-MANIFEST.json").read_text()); rep["release"] = manifest.get("name")
    wheels = sorted(glob.glob(str(b / "wheels" / "*.whl")))
    check(R, "exactly one wheel, the integrated wheel", [os.path.basename(w) for w in wheels] == [WHEEL_NAME], [os.path.basename(w) for w in wheels])
    if [os.path.basename(w) for w in wheels] != [WHEEL_NAME]: finish("02_install_integrated", R, rep)
    w = wheels[0]; wsha = hashlib.sha256(open(w, "rb").read()).hexdigest(); rep["wheel_sha256"] = wsha
    check(R, "wheel SHA256 equals RELEASE-MANIFEST.json", manifest.get("integrated_wheel", {}).get("sha256") == wsha, wsha)
    check(R, "python 3.13", sys.version_info[:2] == (3, 13), platform.python_version())
    before = snapshot(); rep["before_count"] = len(before); (OUT / "02_packages_before.json").write_text(json.dumps(before, indent=0, sort_keys=True))
    errs = check_baseline(before)
    check(R, "baseline protected (torch/triton exact, transformers>=5.5.3,<6, no prohibited packages)", not errs,
          errs or {k: before.get(k) for k in ("torch", "triton", "transformers", "numpy", "nvidia-cuda-nvcc", "nvidia-nvvm")})
    try:
        import torch
        cv = torch.version.cuda; check(R, "PyTorch CUDA build is 13.0", cv == "13.0", cv)
        n = torch.cuda.device_count(); caps = [torch.cuda.get_device_capability(i) for i in range(n)]; names = [torch.cuda.get_device_name(i) for i in range(n)]
        ok = n == 2 and all(c == (7, 5) for c in caps) and all("T4" in x for x in names)
        if ok: check(R, "two Tesla T4 GPUs (SM75)", True, f"{names} {caps}")
        elif a.dry_run: skipped(R, "two Tesla T4 GPUs (SM75)", f"dry run on a machine with {n} GPU(s) {caps}: not enforced")
        else: check(R, "two Tesla T4 GPUs (SM75)", False, f"{n} gpu(s) {names} {caps}")
    except Exception as e:  # noqa: BLE001
        (skipped if a.dry_run else lambda r, n_, d: check(r, n_, False, d))(R, "torch/CUDA probe", repr(e))
    wi = inspect_wheel(w); rep["wheel_inspection"] = {k: v for k, v in wi.items() if k != "metadata"}
    check(R, "wheel tags: cp313-cp313-linux_x86_64, build tag 2", wi["tag"] == "cp313-cp313-linux_x86_64" and wi["build"] == "2", (wi["tag"], wi["build"]))
    check(R, "wheel has exactly the packages stokastisk + vllm (no duplicate vLLM tree)", wi["top_level"] == ["stokastisk", "vllm"], wi["top_level"])
    check(R, "wheel registers the vllm 0.25.1 distribution (vLLM calls importlib.metadata.version('vllm'))", wi["vllm_dist_info_bundled"])
    check(R, "wheel contains no FlashAttention/FlashInfer/FlashMLA files", not wi["prohibited"], wi["prohibited"] or "none")
    pins = parse_pins((b / "requirements" / "kaggle-t4-core.txt").read_text()); blocks = parse_hash_blocks((b / "requirements" / "kaggle-t4-core.hashes.txt").read_text())
    check(R, "every core pin has sha256 hashes", set(pins) == set(blocks), sorted(set(pins) ^ set(blocks)) or f"{len(pins)} pins")
    plan = make_plan(before, pins, core_requirements(wi["metadata"])); rep["plan"] = plan
    for c in plan["conflicts"]: check(R, "plan: " + c, False)
    check(R, "dependency plan has no conflicts (nothing is installed if this fails)", not plan["conflicts"], {"install": plan["install"]})
    (OUT / "02_install_plan.json").write_text(json.dumps(plan, indent=1))
    if any(r["status"] == "FAIL" for r in R):
        rep["seconds"] = time.time() - t0; finish("02_install_integrated", R, rep)
    legacy = legacy_two_wheel_installed(); rep["legacy_two_wheel_detected"] = legacy
    try: already = md.distribution("stokastisk").read_text("METADATA") == wi["metadata"] and not legacy
    except md.PackageNotFoundError: already = False
    rep["already_installed"] = already
    tmp = pathlib.Path(tempfile.mkdtemp(prefix="stk-pins-")); dl = 0
    if plan["install"]:
        req = tmp / "pins.txt"; req.write_text("\n".join(blocks[n] for n in plan["install"]) + "\n")
        r = pip("download", "--no-deps", "--require-hashes", "-r", str(req), "-d", str(tmp)); rep["pip_download_rc"] = r.returncode
        check(R, "download exact pins with --require-hashes --no-deps (no resolver)", r.returncode == 0, r.stderr[-300:] if r.returncode else f"{len(plan['install'])} pins hash-verified")
        dl = sum(p.stat().st_size for p in tmp.glob("*.whl")); rep["downloaded_bytes"] = dl
    pc_before = pip("check").stdout.strip().splitlines(); rep["pip_check_before"] = pc_before
    if a.dry_run:
        rep["would_install"] = {"wheel": WHEEL_NAME, "uninstall_legacy": legacy, "pins": plan["install"]}; rep["seconds"] = time.time() - t0
        skipped(R, "installation", "dry run"); finish("02_install_integrated_dryrun", R, rep); return
    if any(r["status"] == "FAIL" for r in R):
        rep["seconds"] = time.time() - t0; finish("02_install_integrated", R, rep)
    if legacy:
        r = pip("uninstall", "-y", "stokastisk", "vllm"); check(R, "legacy two-wheel install removed (stokastisk + vllm only)", r.returncode == 0, r.stdout[-300:] + r.stderr[-300:])
    if not already:
        r = pip("install", "--force-reinstall", "--no-deps", "--no-index", w); check(R, "install integrated wheel --force-reinstall --no-deps --no-index", r.returncode == 0, r.stderr[-400:])
    else:
        skipped(R, "install integrated wheel", "identical wheel metadata already installed")
    if plan["install"]:
        r = pip("install", "--no-deps", "--no-index", "--find-links", str(tmp), *[f"{n}=={v}" for n, v in plan["install"].items()])
        check(R, "install core pins --no-deps --no-index", r.returncode == 0, r.stderr[-400:])
    after = snapshot(); (OUT / "02_packages_after.json").write_text(json.dumps(after, indent=0, sort_keys=True))
    d = diff_snapshots(before, after, plan["install"]); rep["diff"] = d
    check(R, "no preinstalled package (torch, triton, CUDA, NCCL, ...) changed or removed", not d["preinstalled_changed_or_removed"], d["preinstalled_changed_or_removed"] or "identical")
    check(R, "only expected packages were added", not d["unexpected_added"], d["added"])
    errs = check_baseline(after); check(R, "protected baseline still intact", not errs, errs or "ok")
    pc = pip("check").stdout.strip().splitlines(); rep["pip_check_after"] = pc; cls = classify_pip_check(pc_before, pc); rep["pip_check_new"] = cls
    check(R, "pip check: no new conflict involving stokastisk/vllm", not cls["own"], cls["own"] or "none")
    check(R, "pip check: no new conflict beyond the documented xgrammar one", not cls["other"], cls["other"] or "none")
    if cls["known"]: warn(R, "pip check is NOT clean: documented known conflict xgrammar 0.2.5.post1 -> transformers<5", cls["known"][0] + "  [functionally verified; docs/DEPENDENCY_COMPATIBILITY.md]")
    else: check(R, "pip check: documented xgrammar conflict absent", True, "absent")
    imports = ["transformers", "tokenizers", "vllm", "stokastisk", "stokastisk.compat", "stokastisk.serving", "xgrammar", "vllm._C_stable_libtorch", "vllm._moe_C_stable_libtorch",
               "vllm.cumem_allocator", "vllm.spinloop", "vllm.fs_io_C", "vllm.engine.arg_utils", "vllm.v1.attention.backends.registry", "vllm.v1.attention.selector",
               "vllm.v1.attention.backends.triton_attn", "vllm.v1.engine.llm_engine", "vllm.v1.executor.multiproc_executor",
               "vllm.distributed.device_communicators.cuda_communicator", "vllm.model_executor.models.qwen2", "vllm.entrypoints.llm"]
    tb = {}
    for m in imports:
        try: __import__(m)
        except BaseException:  # noqa: BLE001
            tb[m] = traceback.format_exc()[-1500:]
    check(R, "imports (full traceback captured on failure)", not tb, tb or f"{len(imports)} modules")
    try:
        from vllm.v1.attention.backends.registry import AttentionBackendEnum
        cls_ = AttentionBackendEnum.TRITON_ATTN.get_class(); check(R, "TRITON_ATTN backend class resolves", cls_.__name__ == "TritonAttentionBackend", cls_.__name__)
    except BaseException:  # noqa: BLE001
        check(R, "TRITON_ATTN backend class resolves", False, traceback.format_exc()[-800:])
    import stokastisk, vllm
    try: vv = md.version("vllm"); sv = md.version("stokastisk")
    except md.PackageNotFoundError as e: vv = sv = repr(e)
    check(R, "versions: stokastisk 0.1.0 (module+distribution), vllm 0.25.1 (module+distribution)", stokastisk.__version__ == sv == LOCK["stokastisk"] and vllm.__version__ == vv == LOCK["vllm"],
          {"stokastisk": [stokastisk.__version__, sv], "vllm": [vllm.__version__, vv]})
    check(R, "single vLLM package tree on sys.path", len(list(vllm.__path__)) == 1 and not legacy_two_wheel_installed(), list(vllm.__path__))
    z = zipfile.ZipFile(w); root = pathlib.Path(vllm.__file__).resolve().parent.parent; badf = []
    for n in z.namelist():
        if n.startswith(("stokastisk/", "vllm/")) and n.endswith((".py", ".so")):
            f = root / n
            if not f.is_file() or hashlib.sha256(f.read_bytes()).hexdigest() != hashlib.sha256(z.read(n)).hexdigest(): badf.append(n)
    check(R, "installed .py/.so files are byte-identical to the wheel", not badf, badf[:5] or "identical")
    pr = stokastisk.prohibited_report(); rep["prohibited"] = pr
    check(R, "no FlashAttention/FlashInfer distributions, modules or binaries installed", pr["clean"], pr)
    try:
        a_cfg = stokastisk.StokastiskConfig(); check(R, "SDK default attention backend is TRITON_ATTN", a_cfg.attention_backend == "TRITON_ATTN", a_cfg.attention_backend)
        for bad_backend in ("FLASH_ATTN", "FLASHINFER"):
            try: stokastisk.StokastiskConfig(attention_backend=bad_backend); check(R, f"SDK rejects {bad_backend}", False, "accepted")
            except stokastisk.ConfigurationError: check(R, f"SDK rejects {bad_backend}", True)
    except BaseException:  # noqa: BLE001
        check(R, "SDK configuration checks", False, traceback.format_exc()[-600:])
    rep["seconds"] = time.time() - t0; rep["downloaded_bytes"] = dl
    finish("02_install_integrated", R, rep)


if __name__ == "__main__":
    main()
