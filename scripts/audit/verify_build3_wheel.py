#!/usr/bin/env python3.13
"""Independent verification of the build-3 integrated wheel (does not trust the builder).
usage: verify_build3_wheel.py WHEEL VLLM_PINNED_WHEEL OUT.json
Checks: tags/version/build, top-level packages, RECORD hashes of every member, the strict Flash policy (path, native, binding, bytecode), every native object byte-identical to the
byte-pinned vLLM wheel, every vllm/** file equal to the pinned original EXCEPT exactly the files the transform removes/patches/adds (the transform is re-run here and its output
compared byte for byte), METADATA requirements equal to the lock roots, and no __pycache__. Exit 1 on any failure."""
import base64, csv, hashlib, io, json, pathlib, re, sys, zipfile

ROOT = pathlib.Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts/release")); sys.path.insert(0, str(ROOT / "sdk"))
import t4_flash_free  # noqa: E402
from stokastisk import prohibited  # noqa: E402

sha = lambda b: hashlib.sha256(b).hexdigest()  # noqa: E731
canon = lambda s: re.sub(r"[-_.]+", "-", s).lower()  # noqa: E731


def main(wheel, vllm_wheel, out):
    checks = []

    def chk(name, ok, detail=""):
        checks.append({"check": name, "status": "PASS" if ok else "FAIL", "detail": str(detail)[:600]}); print(("PASS " if ok else "FAIL ") + name + (f" -- {str(detail)[:200]}" if not ok else ""))

    z = zipfile.ZipFile(wheel); names = z.namelist(); vz = zipfile.ZipFile(vllm_wheel)
    wname = pathlib.Path(wheel).name
    chk("filename is stokastisk-0.1.0-<build>-cp313-cp313-linux_x86_64.whl", re.fullmatch(r"stokastisk-0\.1\.0-\d+-cp313-cp313-linux_x86_64\.whl", wname) is not None, wname)
    di = "stokastisk-0.1.0.dist-info"
    wh = z.read(f"{di}/WHEEL").decode()
    chk("WHEEL tag cp313-cp313-linux_x86_64 and a build tag >= 3", "Tag: cp313-cp313-linux_x86_64" in wh and int(re.search(r"Build: (\d+)", wh).group(1)) >= 3, wh.replace("\n", " | "))
    meta = z.read(f"{di}/METADATA").decode()
    chk("METADATA Version is 0.1.0", re.search(r"^Version: 0\.1\.0$", meta, re.M) is not None)
    tops = sorted({n.split("/")[0] for n in names if ".dist-info/" not in n and ".data/" not in n})
    chk("top-level packages are exactly stokastisk + vllm (one vLLM tree)", tops == ["stokastisk", "vllm"], tops)
    rec = {r[0]: r for r in csv.reader(io.StringIO(z.read(f"{di}/RECORD").decode()))}
    bad = [n for n in names if n != f"{di}/RECORD" and (n not in rec or rec[n][1] != "sha256=" + base64.urlsafe_b64encode(hashlib.sha256(z.read(n)).digest()).rstrip(b"=").decode())]
    chk("RECORD lists every member with the right sha256", not bad and set(rec) == set(names), bad[:5])
    sc = prohibited.scan_members(names, z.read)
    chk("strict Flash policy: 0 path hits (.py/.pyc/.so/...), 0 native markers, 0 bindings, 0 __pycache__", sc["clean"] and not sc["pycache_dirs"], {k: sc[k] for k in ("path_hits", "native_hits", "binding_hits", "pycache_dirs")})
    chk("the pinned vLLM wheel DID contain the prohibited files (the scanner is not vacuous)", len(prohibited.scan_members(vz.namelist(), vz.read)["path_hits"]) >= 28, len(prohibited.scan_members(vz.namelist(), vz.read)["path_hits"]))
    # natives byte-identical
    nat = [n for n in vz.namelist() if n.startswith("vllm/") and prohibited.is_binary(n)]
    chk("every native object equals the byte-pinned vLLM wheel's (nothing recompiled, nothing Flash added)", all(n in names and z.read(n) == vz.read(n) for n in nat) and {n for n in names if n.startswith("vllm/") and prohibited.is_binary(n)} == set(nat), nat)
    # transform reproduced
    vt = {n: vz.read(n) for n in vz.namelist() if n.startswith("vllm/") and not n.endswith("/") and "__pycache__" not in n}
    exp, rep = t4_flash_free.transform(dict(vt))
    got = {n: z.read(n) for n in names if n.startswith("vllm/") and not n.endswith("/")}
    diff = sorted(n for n in set(exp) | set(got) if exp.get(n) != got.get(n))
    chk("vllm/** equals (pinned wheel + the documented transform) byte for byte", not diff, diff[:5])
    unchanged = sum(1 for n in vt if n in got and got[n] == vt[n])
    chk("everything the transform did not touch is byte-identical to the pinned wheel", unchanged == len(vt) - len(rep["removed"]) - len({x["file"] for x in rep["rewritten"]}) - len(rep["patched"]) + len({x["file"] for x in rep["rewritten"]} & {x["path"] for x in rep["patched"]}), f"{unchanged} identical of {len(vt)}")
    # dependency metadata
    reqs = [l.split(":", 1)[1].strip() for l in meta.splitlines() if l.startswith("Requires-Dist:") and "extra ==" not in l]
    roots = []
    for prof in ("core", "structured-output", "serving"):
        roots += [l.split("#")[0].strip() for l in (ROOT / "requirements" / f"{prof}.roots").read_text().splitlines() if l.split("#")[0].strip()]
    chk("METADATA requires every lock root exactly (core + structured-output + serving)", all(r in reqs for r in roots), [r for r in roots if r not in reqs])
    chk("METADATA pins torch==2.11.0+cu130 and triton==3.6.0, transformers>=5.5.3,<6", {"torch==2.11.0+cu130", "triton==3.6.0", "transformers>=5.5.3,<6"} <= set(reqs))
    chk("METADATA does not require xgrammar<…transformers caps: xgrammar==0.2.3", "xgrammar==0.2.3" in reqs and not any(canon(r.split("=")[0]) == "xgrammar" and "0.2.5" in r for r in reqs))
    chk("models extra present, not in the default requirements", "Provides-Extra: models" in meta and "compressed-tensors" not in " ".join(reqs))
    chk("vllm 0.25.1 dist-info bundled (vLLM resolves its own version through it)", f"stokastisk-0.1.0.data/platlib/vllm-0.25.1.dist-info/METADATA" in names)
    for must in ("vllm/t4_backend_policy.py", "vllm/utils/t4_prohibited_stubs.py", "stokastisk/prohibited.py", "stokastisk/dependencies.py", "stokastisk/structured.py"):
        chk(f"{must} is present", must in names)
    status = "PASS" if all(c["status"] == "PASS" for c in checks) else "FAIL"
    pathlib.Path(out).write_text(json.dumps({"status": status, "wheel": wname, "wheel_sha256": sha(pathlib.Path(wheel).read_bytes()), "files": len(names), "scan": {k: v for k, v in sc.items() if k != "references"},
                                              "transform_summary": rep["summary"], "checks": checks}, indent=1))
    print("OVERALL:", status)
    return 0 if status == "PASS" else 1


if __name__ == "__main__":
    sys.exit(main(*sys.argv[1:4]))
