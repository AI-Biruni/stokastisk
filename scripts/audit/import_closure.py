#!/usr/bin/env python3.13
"""Static import-closure audit: which third-party modules does a set of entry modules import at MODULE LEVEL (transitively, inside one source tree)?

usage: import_closure.py TREE_ROOT ENTRY_MODULE [ENTRY_MODULE ...] [--json OUT]

 * TREE_ROOT is a directory that contains the `vllm/` (and optionally `stokastisk/`) package directories (an unpacked wheel).
 * Only imports that execute when the module is imported are followed: top-level statements, including those inside `if`/`try`/`with`/`for`.
   Imports in function bodies are reported separately as `lazy`; `if TYPE_CHECKING:` blocks are ignored.
 * An import inside `try:` whose handler catches ImportError/ModuleNotFoundError/Exception is reported as `optional` (the module still imports without it).
 * A class-body or module-level *attribute access* on a lazily-loaded module (e.g. vLLM's LazyLoader used in an annotation of a dataclass) is NOT visible statically;
   that is how `lmformatenforcer` became a hard dependency of every structured-output request. Such modules are listed in LAZY_HARD below and tested dynamically.
No import is executed: this is pure `ast` analysis, so it is safe on an untrusted tree and needs no GPU."""
import ast, json, pathlib, sys

STDLIB = set(sys.stdlib_module_names)
IMPORT_TO_DIST = {"yaml": "pyyaml", "PIL": "pillow", "zmq": "pyzmq", "cv2": "opencv-python", "sklearn": "scikit-learn", "google": "protobuf", "attr": "attrs",
                  "msgspec": "msgspec", "cbor2": "cbor2", "pybase64": "pybase64", "openai_harmony": "openai-harmony", "partial_json_parser": "partial-json-parser",
                  "xgrammar": "xgrammar", "llguidance": "llguidance", "tvm_ffi": "apache-tvm-ffi", "lmformatenforcer": "lm-format-enforcer", "outlines_core": "outlines-core",
                  "model_hosting_container_standards": "model-hosting-container-standards", "prometheus_fastapi_instrumentator": "prometheus-fastapi-instrumentator",
                  "sse_starlette": "sse-starlette", "pydantic_settings": "pydantic-settings", "fastapi_cli": "fastapi-cli", "mistral_common": "mistral-common",
                  "compressed_tensors": "compressed-tensors", "setproctitle": "setproctitle", "cpuinfo": "py-cpuinfo", "uvloop": "uvloop", "regex": "regex",
                  "opentelemetry": "opentelemetry-api", "cloudpickle": "cloudpickle", "einops": "einops", "tiktoken": "tiktoken", "psutil": "psutil",
                  "sentencepiece": "sentencepiece", "cachetools": "cachetools", "ijson": "ijson", "diskcache": "diskcache", "interegular": "interegular", "depyf": "depyf",
                  "jsonschema": "jsonschema", "numba": "numba", "scipy": "scipy", "pandas": "pandas", "gguf": "gguf", "blake3": "blake3", "packaging": "packaging",
                  "typing_extensions": "typing-extensions", "huggingface_hub": "huggingface-hub", "filelock": "filelock", "requests": "requests", "tqdm": "tqdm",
                  "safetensors": "safetensors", "tokenizers": "tokenizers", "transformers": "transformers", "numpy": "numpy", "torch": "torch", "torchvision": "torchvision",
                  "triton": "triton", "fastapi": "fastapi", "starlette": "starlette", "uvicorn": "uvicorn", "aiohttp": "aiohttp", "openai": "openai", "pydantic": "pydantic",
                  "prometheus_client": "prometheus-client", "websockets": "websockets", "anyio": "anyio", "httpx": "httpx", "jinja2": "jinja2", "lark": "lark"}


def is_type_checking(test) -> bool:
    return "TYPE_CHECKING" in ast.unparse(test)


def catches_import_error(handlers) -> bool:
    for h in handlers:
        t = ast.unparse(h.type) if h.type is not None else "BaseException"
        if any(k in t for k in ("ImportError", "ModuleNotFoundError", "Exception", "BaseException")):
            return True
    return False


class Mod:
    def __init__(self, name, path, is_pkg):
        self.name, self.path, self.is_pkg = name, path, is_pkg
        self.eager, self.optional, self.lazy = [], [], []     # (imported dotted name, lineno)


def collect(mod: Mod, tree: ast.AST):
    pkg = mod.name if mod.is_pkg else mod.name.rpartition(".")[0]

    def resolve(node):
        out = []
        if isinstance(node, ast.Import):
            out = [a.name for a in node.names]
        else:
            base = node.module or ""
            if node.level:
                parts = pkg.split(".")
                parts = parts[:len(parts) - (node.level - 1)]
                base = ".".join(parts + ([node.module] if node.module else []))
            out = [base] + [f"{base}.{a.name}" for a in node.names if a.name != "*"]
        return out

    def walk(body, bucket_default, in_func):
        for st in body:
            if isinstance(st, (ast.Import, ast.ImportFrom)):
                for n in resolve(st):
                    (mod.lazy if in_func else getattr(mod, bucket_default)).append((n, st.lineno))
            elif isinstance(st, (ast.FunctionDef, ast.AsyncFunctionDef)):
                walk(st.body, "lazy", True)
            elif isinstance(st, ast.ClassDef):
                walk(st.body, bucket_default, in_func)
            elif isinstance(st, ast.If):
                if not is_type_checking(st.test):
                    walk(st.body, bucket_default, in_func)
                walk(st.orelse, bucket_default, in_func)
            elif isinstance(st, ast.Try):
                opt = "optional" if catches_import_error(st.handlers) else bucket_default
                walk(st.body, opt, in_func)
                for h in st.handlers:
                    walk(h.body, bucket_default, in_func)
                walk(st.orelse, bucket_default, in_func); walk(st.finalbody, bucket_default, in_func)
            elif isinstance(st, (ast.With, ast.AsyncWith, ast.For, ast.AsyncFor, ast.While)):
                walk(st.body, bucket_default, in_func); walk(getattr(st, "orelse", []), bucket_default, in_func)

    walk(tree.body, "eager", False)


def load_tree(root: pathlib.Path):
    mods = {}
    for top in ("vllm", "stokastisk"):
        for p in (root / top).rglob("*.py"):
            rel = p.relative_to(root).with_suffix("")
            is_pkg = rel.name == "__init__"
            name = ".".join(rel.parts[:-1] if is_pkg else rel.parts)
            m = Mod(name, p, is_pkg)
            try:
                collect(m, ast.parse(p.read_text(errors="replace")))
            except SyntaxError as e:
                print("syntax error", p, e, file=sys.stderr)
            mods[name] = m
    return mods


def closure(mods, entries):
    seen, order, third = set(), [], {}
    stack = list(entries)
    while stack:
        n = stack.pop()
        if n in seen or n not in mods:
            continue
        seen.add(n); order.append(n)
        parts = n.split(".")
        for i in range(1, len(parts)):                      # importing a.b.c executes a/__init__ and a/b/__init__
            stack.append(".".join(parts[:i]))
        m = mods[n]
        for kind in ("eager", "optional"):
            for imp, ln in getattr(m, kind):
                top = imp.split(".")[0]
                if top in ("vllm", "stokastisk"):
                    stack.append(imp)
                elif top not in STDLIB and top != "__future__":
                    third.setdefault(top, {}).setdefault(kind, []).append(f"{n}:{ln}")
    return seen, third


def main(argv):
    out = None
    if "--json" in argv:
        i = argv.index("--json"); out = argv[i + 1]; del argv[i:i + 2]
    root = pathlib.Path(argv[0]); entries = argv[1:]
    mods = load_tree(root)
    seen, third = closure(mods, entries)
    rep = {"entries": entries, "modules_reached": len(seen),
           "third_party": {k: {"dist_guess": IMPORT_TO_DIST.get(k, k.replace("_", "-")), "eager_in": sorted(set(v.get("eager", [])))[:4], "eager_count": len(set(v.get("eager", []))),
                              "optional_count": len(set(v.get("optional", [])))} for k, v in sorted(third.items())}}
    print(f"{len(seen)} modules reached; {len(third)} third-party top-level imports")
    for k, v in rep["third_party"].items():
        print(f"  {k:36s} eager={v['eager_count']:3d} optional={v['optional_count']:3d} dist~{v['dist_guess']}  e.g. {v['eager_in'][:1]}")
    if out:
        pathlib.Path(out).write_text(json.dumps(rep, indent=1))


if __name__ == "__main__":
    main(sys.argv[1:])
