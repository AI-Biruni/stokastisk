"""Combine benchmark records into one JSON + a markdown table. Missing strategies are listed as NOT_RUN, never omitted silently."""
from __future__ import annotations

from typing import Any


def comparison_table(records: dict[str, dict[str, Any]], workload: str) -> str:
    rows = [f"workload `{workload}`", "", "| strategy | status | decode tok/s (median) | prefill tok/s (median) | TTFT batch s (median) | ITL s (median) |", "|---|---|---|---|---|---|"]
    f = lambda v: "-" if v is None else f"{v:,.4g}"  # noqa: E731
    for name, rec in records.items():
        r = (rec.get("results") or {}).get(workload)
        if rec.get("status") != "MEASURED" or not r:
            rows.append(f"| {name} | {rec.get('status', 'NOT_RUN')} | - | - | - | - |"); continue
        g = lambda k: (r.get(k) or {}).get("median")  # noqa: E731
        rows.append(f"| {name} | MEASURED | {f(g('decode_tok_per_s'))} | {f(g('prefill_tok_per_s'))} | {f(g('ttft_batch_seconds'))} | {f(g('inter_token_latency_s'))} |")
    return "\n".join(rows)


def relative_to(records: dict[str, dict[str, Any]], workload: str, base: str = "tp1") -> dict[str, float | None]:
    get = lambda k: ((records.get(k) or {}).get("results") or {}).get(workload, {}).get("decode_tok_per_s", {}).get("median")  # noqa: E731
    b = get(base)
    return {k: (None if not (b and get(k)) else get(k) / b) for k in records}
