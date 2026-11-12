#!/usr/bin/env python3.13
"""Static (no GPU, no Kaggle, no vLLM import) verification of the integrated SDK candidate wheel.
usage: verify_sdk_candidate.py CANDIDATE.whl HISTORICAL_VLLM.whl HISTORICAL_SDK.whl OUT.json"""
import base64, csv, hashlib, io, json, re, sys, zipfile
cand, vl, hs, out = sys.argv[1:5]
z, v, h = zipfile.ZipFile(cand), zipfile.ZipFile(vl), zipfile.ZipFile(hs)
sha = lambda b: hashlib.sha256(b).hexdigest()  # noqa: E731
names = z.namelist(); R = {}
R["wheel"] = cand.split("/")[-1]; R["sha256"] = sha(open(cand, "rb").read())
R["filename_ok"] = bool(re.fullmatch(r"stokastisk-0\.1\.0-1-cp313-cp313-linux_x86_64\.whl", R["wheel"]))
R["testzip"] = z.testzip()
md = z.read("stokastisk-0.1.0.dist-info/METADATA").decode(); wh = z.read("stokastisk-0.1.0.dist-info/WHEEL").decode()
R["metadata"] = {"Name": re.search(r"^Name: (.*)$", md, re.M).group(1), "Version": re.search(r"^Version: (.*)$", md, re.M).group(1),
                 "Requires-Python": re.search(r"^Requires-Python: (.*)$", md, re.M).group(1),
                 "requires_vllm_dist": bool(re.search(r"^Requires-Dist: vllm\b", md, re.M)),
                 "pins": [l for l in md.splitlines() if re.match(r"Requires-Dist: (torch|triton|transformers)", l)]}
R["wheel_file"] = dict(l.split(": ", 1) for l in wh.strip().splitlines())
R["forbidden_dependencies_in_metadata"] = [l for l in md.splitlines() if l.startswith("Requires-Dist:") and re.search(r"flash|flashinfer", l, re.I)]
# RECORD integrity
rec = list(csv.reader(io.StringIO(z.read("stokastisk-0.1.0.dist-info/RECORD").decode())))
bad = []; listed = set()
for n, hsh, size in rec:
    listed.add(n)
    if n.endswith("RECORD"): continue
    d = "sha256=" + base64.urlsafe_b64encode(hashlib.sha256(z.read(n)).digest()).rstrip(b"=").decode()
    if d != hsh or int(size) != z.getinfo(n).file_size: bad.append(n)
R["record"] = {"entries": len(rec), "mismatches": bad, "unlisted_members": sorted(set(names) - listed)}
# SDK modules
sdk = {n for n in names if n.startswith("stokastisk/") and n.endswith(".py")}
R["sdk_modules"] = {"count": len(sdk), "compat": "stokastisk/compat.py" in sdk, "serving": "stokastisk/serving.py" in sdk, "engine": "stokastisk/engine.py" in sdk}
eng = z.read("stokastisk/engine.py").decode()
R["sdk_features"] = {"LLM.chat": "def chat(" in eng, "bfloat16_precheck": "check_dtype(" in eng and "def check_dtype" in z.read("stokastisk/compat.py").decode(),
                     "serving_ServerProcess": "class ServerProcess" in z.read("stokastisk/serving.py").decode(),
                     "serving_forces_TRITON_ATTN": "--attention-backend" in z.read("stokastisk/serving.py").decode() and "TRITON_ATTN" in z.read("stokastisk/serving.py").decode(),
                     "config_only_TRITON_ATTN": 'ALLOWED_ATTENTION_BACKENDS = ("TRITON_ATTN",)' in z.read("stokastisk/config.py").decode()}
# historical SDK wheel = subset of candidate stokastisk/ (old files may differ only where sdk/ changed)
hsdk = {n for n in h.namelist() if n.startswith("stokastisk/") and n.endswith(".py")}
R["historical_sdk_files_all_present"] = sorted(hsdk - sdk) == []
R["historical_sdk_files_changed_in_candidate"] = sorted(n for n in hsdk if z.read(n) != h.read(n))
R["new_sdk_files_vs_historical"] = sorted(sdk - hsdk)
# vLLM tree identical
vf = [n for n in v.namelist() if n.startswith("vllm/") and not n.endswith("/")]
diff = [n for n in vf if n not in z.namelist() or z.read(n) != v.read(n)]
extra = sorted(n for n in names if n.startswith("vllm/") and n not in set(vf))
R["vllm_tree"] = {"files": len(vf), "differences_vs_verified_vllm_wheel": diff, "extra_files": extra}
R["vllm_dist_info_bundled"] = "stokastisk-0.1.0.data/platlib/vllm-0.25.1.dist-info/METADATA" in names
R["vllm_version_file"] = re.search(r'__version__ = "([^"]+)"', z.read("vllm/_version.py").decode()).group(1)
# no Flash/FlashInfer binaries or packages
R["flash_markers_nonpy"] = [n for n in names if re.search(r"flashinfer|vllm_flash_attn/_vllm_fa|flashmla|flash_attn_2_cuda|libflash", n, re.I) and not n.endswith(".py")]
R["flash_named_py_modules"] = sorted(n for n in names if n.startswith("vllm/") and n.endswith(".py") and re.search(r"flash", n, re.I))
R["native_objects"] = sorted(n for n in names if n.endswith(".so"))
R["python_executed"] = False; R["gpu_executed"] = False; R["kaggle_executed"] = False
R["status"] = "BUILT AND PACKAGED — STATICALLY INSPECTED — NOT EXECUTED ON KAGGLE DUAL T4 GPUs"
ok = (R["filename_ok"] and R["testzip"] is None and R["metadata"]["Version"] == "0.1.0" and not R["metadata"]["requires_vllm_dist"] and not bad and not R["record"]["unlisted_members"]
      and all(R["sdk_modules"].values()) and all(R["sdk_features"].values()) and not diff and not extra and R["vllm_dist_info_bundled"] and R["vllm_version_file"] == "0.25.1"
      and not R["flash_markers_nonpy"] and not R["forbidden_dependencies_in_metadata"] and R["wheel_file"]["Tag"] == "cp313-cp313-linux_x86_64")
R["all_static_checks_pass"] = bool(ok)
json.dump(R, open(out, "w"), indent=1); print(json.dumps({k: R[k] for k in ("all_static_checks_pass", "sha256", "wheel")}, indent=1)); sys.exit(0 if ok else 1)
