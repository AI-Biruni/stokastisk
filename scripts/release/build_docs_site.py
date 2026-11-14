#!/usr/bin/env python3.13
"""Build the user documentation.

usage: build_docs_site.py OUT_DIR     regenerate docs/guide/api-reference.md from the SDK source, then render docs/guide/*.md to a static HTML site in OUT_DIR
The site is published on the `gh-pages` branch under /docs/ (GitHub Pages). Needs `mistune` and `pygments`; reads only files in this repository."""
import ast, html, pathlib, re, sys

ROOT = pathlib.Path(__file__).resolve().parents[2]
GUIDE = ROOT / "docs/guide"
SDK = ROOT / "sdk/stokastisk"
BASE = "https://github.com/stokastisk-ai/stokastisk"
ORDER = ["index", "getting-started", "configuration", "engine", "structured-output", "server", "multi-gpu", "diagnostics", "feature-matrix", "errors", "advanced", "troubleshooting", "api-reference"]
TITLES = {"index": "Overview", "getting-started": "1. Getting started", "configuration": "2. Configuration", "engine": "3. The LLM engine", "structured-output": "4. Structured output",
          "server": "5. OpenAI-compatible server", "multi-gpu": "6. Using both GPUs", "diagnostics": "7. Checks and diagnostics", "feature-matrix": "   Feature matrix", "errors": "8. Errors", "advanced": "9. Advanced modules",
          "troubleshooting": "10. Troubleshooting", "api-reference": "API reference"}


def first_line(doc):
    return (doc or "").strip().splitlines()[0].strip() if doc and doc.strip() else ""


def api_reference() -> str:
    out = ["# API reference", "", "Every Python module of the `stokastisk` package and its public names. Generated from the source by `scripts/release/build_docs_site.py`.", "",
           "Names exported at the top level (`import stokastisk`) are in `stokastisk.__all__`; submodule names are imported as shown, e.g. `from stokastisk.serving import ServerProcess`.", ""]
    for p in sorted(SDK.rglob("*.py")):
        rel = p.relative_to(SDK).with_suffix("")
        parts = list(rel.parts)
        if parts[-1] == "__init__":
            parts = parts[:-1]
        mod = ".".join(["stokastisk", *parts])
        if any(x.startswith("_") for x in parts):
            continue
        tree = ast.parse(p.read_text())
        rows = []
        for n in tree.body:
            if isinstance(n, ast.ClassDef) and not n.name.startswith("_"):
                rows.append((f"class {n.name}", first_line(ast.get_docstring(n))))
                for m in n.body:
                    if isinstance(m, (ast.FunctionDef, ast.AsyncFunctionDef)) and (not m.name.startswith("_") or m.name == "__init__"):
                        rows.append((f"&nbsp;&nbsp;{n.name}.{m.name}({ast.unparse(m.args)})", first_line(ast.get_docstring(m))))
            elif isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef)) and not n.name.startswith("_"):
                rows.append((f"def {n.name}({ast.unparse(n.args)})", first_line(ast.get_docstring(n))))
        out += [f"## `{mod}`", "", first_line(ast.get_docstring(tree)) or "", ""]
        if rows:
            out += ["| Name | Summary |", "|---|---|"] + [f"| <code>{r[0].replace('|', '&#124;')}</code> | {r[1].replace('|', '&#124;')} |" for r in rows] + [""]
    return "\n".join(out) + "\n"


def feature_matrix_page() -> str:
    """docs/guide/feature-matrix.md, generated from stokastisk.compat so the documentation cannot contradict the SDK."""
    sys.path.insert(0, str(ROOT / "sdk"))
    from stokastisk import compat
    esc = lambda s: str(s).replace("|", "&#124;")  # noqa: E731
    out = ["# Feature matrix and model evidence", "",
           "Generated from `stokastisk.compat` (`FEATURES`, `MODEL_EVIDENCE`) by `scripts/release/build_docs_site.py`; do not edit by hand. These are the records the installed SDK reports through `stokastisk.feature_matrix()` and `stokastisk.check_model()`.", "",
           "**Evidence classes.** `kaggle`: executed on Kaggle 2 × T4 (the *notebooks* column says which: 15 = acceptance suite on build 3, 16/18/19 = end-user notebooks, 01–14 = earlier notebooks). `kaggle-kernel`: a Triton kernel compiled and launched, output not asserted. `cpu-tests`: CPU unit tests only. `static`: source inspection only. Builds 4 and 5 were not executed on Kaggle; their inference code is byte-identical to build 3.", "",
           "## Features", "", "| Feature | Status | Evidence | Notebooks | Note |", "|---|---|---|---|---|"]
    for n, r in compat.feature_matrix().items():
        out.append(f"| `{n}` | {r['status']} | {r['evidence']} | {esc(r['notebooks']) or '—'} | {esc(r['note'])} |")
    out += ["", "## Models that were run", "", "| Model id | Architecture | Dtype | Max tested `max_model_len` | Modes | Evidence | Limitations |", "|---|---|---|---|---|---|---|"]
    for mid, r in compat.MODEL_EVIDENCE.items():
        out.append(f"| `{mid}` | {r['architecture']} | {r['dtype']} | {r['max_tested_model_len']} | {', '.join(r['tested_modes'])} | {esc('; '.join(r['evidence']))} | {esc('; '.join(r['limitations']))} |")
    out += ["", "Any other checkpoint — including other `Qwen2ForCausalLM` models — is `UNKNOWN_REQUIRES_EVIDENCE` in `check_model()`; see [Checks and diagnostics](diagnostics.md).", ""]
    return "\n".join(out)


CSS = """:root{--bg:#fff;--fg:#1f2328;--mut:#59636e;--bd:#d1d9e0;--ac:#0969da;--code:#f6f8fa;--side:#f6f8fa;--call:#ddf4ff}
@media(prefers-color-scheme:dark){:root{--bg:#0d1117;--fg:#e6edf3;--mut:#9198a1;--bd:#3d444d;--ac:#4493f8;--code:#161b22;--side:#010409;--call:#0c2d6b}}
*{box-sizing:border-box}body{margin:0;background:var(--bg);color:var(--fg);font:16px/1.6 -apple-system,Segoe UI,Helvetica,Arial,sans-serif}
.wrap{display:flex;min-height:100vh}nav{width:270px;flex:none;background:var(--side);border-right:1px solid var(--bd);padding:20px 16px;position:sticky;top:0;height:100vh;overflow:auto}
nav h1{font-size:18px;margin:0 0 4px}nav .v{color:var(--mut);font-size:13px;margin-bottom:14px}nav a{display:block;padding:5px 10px;border-radius:6px;color:var(--fg);text-decoration:none;font-size:14px}
nav a:hover{background:var(--bd)}nav a.cur{background:var(--ac);color:#fff}nav .ext{margin-top:14px;border-top:1px solid var(--bd);padding-top:10px}
main{flex:1;min-width:0;max-width:960px;padding:32px 40px 80px}h1,h2,h3{line-height:1.25}h1{border-bottom:1px solid var(--bd);padding-bottom:.3em}h2{border-bottom:1px solid var(--bd);padding-bottom:.25em;margin-top:2em}
a{color:var(--ac)}code{background:var(--code);padding:.15em .35em;border-radius:5px;font:85% ui-monospace,SFMono-Regular,Menlo,monospace}
pre{background:var(--code);padding:14px 16px;border-radius:8px;overflow:auto;border:1px solid var(--bd)}pre code{background:none;padding:0;font-size:13.5px}
table{border-collapse:collapse;display:block;overflow:auto;max-width:100%}th,td{border:1px solid var(--bd);padding:6px 12px;text-align:left;vertical-align:top}th{background:var(--code)}
.callout{background:var(--call);border-left:4px solid var(--ac);padding:8px 16px;border-radius:6px;margin:1em 0}.callout p{margin:.4em 0}
footer{color:var(--mut);font-size:13px;margin-top:3em;border-top:1px solid var(--bd);padding-top:12px}
@media(max-width:800px){.wrap{display:block}nav{position:static;width:auto;height:auto}main{padding:20px 16px}}
"""


def render(out: pathlib.Path):
    import mistune
    from pygments import highlight
    from pygments.formatters import HtmlFormatter
    from pygments.lexers import get_lexer_by_name, TextLexer

    class R(mistune.HTMLRenderer):
        def block_code(self, code, info=None):
            lang = (info or "").split()[0] if info else ""
            if lang in ("bash", "python"):
                try:
                    lex = get_lexer_by_name("bash" if lang == "bash" else "python")
                except Exception:
                    lex = TextLexer()
                return highlight(code, lex, HtmlFormatter(nowrap=False, cssclass="hl"))
            return f"<pre><code>{html.escape(code)}</code></pre>"

    md = mistune.create_markdown(renderer=R(escape=False), plugins=["table", "strikethrough"])
    out.mkdir(parents=True, exist_ok=True)
    pyg = HtmlFormatter(style="default").get_style_defs(".hl") + "\n@media(prefers-color-scheme:dark){" + HtmlFormatter(style="github-dark").get_style_defs(".hl") + "}\n.hl{background:var(--code)!important}"
    (out / "style.css").write_text(CSS + pyg)
    for name in ORDER:
        src = (GUIDE / f"{name}.md").read_text()
        src = re.sub(r"^!!! (.*)$", r'<div class="callout">\n\n\1\n\n</div>', src, flags=re.M)
        src = re.sub(r"\]\(\.\./\.\./([^)]+)\)", rf"]({BASE}/blob/main/\1)", src)
        src = re.sub(r"\]\(\.\./([^)]+\.md)\)", rf"]({BASE}/blob/main/docs/\1)", src)
        body = md(src).replace('href="index.md"', 'href="index.html"')
        body = re.sub(r'href="([a-z-]+)\.md(#[^"]*)?"', r'href="\1.html\2"', body)
        nav = "".join(f'<a class="{"cur" if n == name else ""}" href="{n}.html">{html.escape(TITLES[n])}</a>' for n in ORDER)
        page = (f'<!doctype html><html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">'
                f'<title>{html.escape(TITLES[name])} · Stokastisk 0.1.0</title><link rel="stylesheet" href="style.css"></head><body><div class="wrap">'
                f'<nav><h1>Stokastisk</h1><div class="v">0.1.0 · Kaggle 2 × T4 · Python 3.13</div>{nav}'
                f'<div class="ext"><a href="{BASE}">GitHub</a><a href="https://huggingface.co/waqasm86/stokastisk">Hugging Face</a><a href="../">Package index</a></div></nav>'
                f'<main>{body}<footer>Stokastisk 0.1.0 · Apache-2.0 · <a href="{BASE}/blob/main/docs/guide/{name}.md">edit this page on GitHub</a></footer></main></div></body></html>')
        (out / f"{name}.html").write_text(page)


if __name__ == "__main__":
    (GUIDE / "api-reference.md").write_text(api_reference())
    (GUIDE / "feature-matrix.md").write_text(feature_matrix_page())
    render(pathlib.Path(sys.argv[1]))
    print("built", sys.argv[1])
