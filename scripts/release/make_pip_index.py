#!/usr/bin/env python3.13
"""Generate the static PEP 503 package index that makes `pip install stokastisk --extra-index-url <INDEX>` work on Kaggle.
usage: make_pip_index.py WHEEL OUT_DIR [--wheel-url URL]

PyPI rejects `linux_x86_64` wheels and Hugging Face serves .html files as text/plain (pip refuses those), so the index is a tiny static site (GitHub Pages) whose project page links to the
wheel on the Hugging Face CDN. The link carries `#sha256=<hash>`, which pip verifies after the download, and `data-requires-python`, which makes pip skip it on other Pythons.
Output: OUT_DIR/.nojekyll, OUT_DIR/index.html, OUT_DIR/simple/index.html, OUT_DIR/simple/stokastisk/index.html."""
import hashlib, html, pathlib, re, sys

HF = "https://huggingface.co/waqasm86/stokastisk/resolve/main/wheels/"
INDEX = "https://ai-biruni.github.io/stokastisk/simple/"
PAGE = '<!DOCTYPE html>\n<html><head><meta charset="utf-8"><meta name="pypi:repository-version" content="1.0"><title>{title}</title></head>\n<body>\n{body}\n</body></html>\n'


def build(wheel: pathlib.Path, out: pathlib.Path, wheel_url: str | None = None) -> dict:
    name = wheel.name
    m = re.fullmatch(r"stokastisk-(0\.1\.0)-(\d+)-cp313-cp313-linux_x86_64\.whl", name)
    if not m:
        raise SystemExit(f"unexpected wheel name {name!r}")
    sha = hashlib.sha256(wheel.read_bytes()).hexdigest()
    url = wheel_url or HF + name
    (out / "simple" / "stokastisk").mkdir(parents=True, exist_ok=True)
    (out / ".nojekyll").write_text("")
    (out / "simple" / "stokastisk" / "index.html").write_text(PAGE.format(title="Links for stokastisk", body=f'<h1>Links for stokastisk</h1>\n<a href="{html.escape(url)}#sha256={sha}" data-requires-python="&gt;=3.13,&lt;3.14">{name}</a><br/>'))
    (out / "simple" / "index.html").write_text(PAGE.format(title="Simple index", body='<a href="stokastisk/">stokastisk</a>'))
    (out / "index.html").write_text(PAGE.format(title="Stokastisk package index", body=f'<h1>Stokastisk 0.1.0 package index</h1>\n<p>Kaggle (2 x Tesla T4, Python 3.13): <code>pip install stokastisk --extra-index-url {INDEX}</code></p>\n<p>Project: <a href="https://github.com/AI-Biruni/stokastisk">github.com/AI-Biruni/stokastisk</a> · binaries: <a href="https://huggingface.co/waqasm86/stokastisk">huggingface.co/waqasm86/stokastisk</a></p>'))
    return {"wheel": name, "sha256": sha, "url": url, "index": INDEX}


if __name__ == "__main__":
    a = sys.argv[1:]
    url = a[a.index("--wheel-url") + 1] if "--wheel-url" in a else None
    print(build(pathlib.Path(a[0]), pathlib.Path(a[1]), url))
