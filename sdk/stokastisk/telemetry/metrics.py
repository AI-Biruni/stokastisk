"""Pure statistics + latency helpers (no GPU needed)."""
from __future__ import annotations

import math
from typing import Any, Sequence


def percentile(values: Sequence[float], q: float) -> float:
    """Linear-interpolated percentile (q in [0,100]); raises on empty input."""
    if not values:
        raise ValueError("percentile of empty sequence")
    v = sorted(values); k = (len(v) - 1) * q / 100.0
    lo, hi = math.floor(k), math.ceil(k)
    return float(v[lo] if lo == hi else v[lo] + (v[hi] - v[lo]) * (k - lo))


def summarize(values: Sequence[float]) -> dict[str, Any]:
    """min / median / p95 / max / mean over ALL runs (never only the best one)."""
    if not values:
        return {"n": 0}
    return {"n": len(values), "min": min(values), "median": percentile(values, 50), "p95": percentile(values, 95), "max": max(values),
            "mean": sum(values) / len(values), "runs": list(values)}


def imbalance_percent(t0: float, t1: float) -> float:
    """abs(T0 - T1) / max(T0, T1) * 100. 0 = perfectly balanced; 0.0 when both are 0."""
    m = max(t0, t1)
    return 0.0 if m <= 0 else abs(t0 - t1) / m * 100.0


def throughput(tokens: int, seconds: float) -> float | None:
    return tokens / seconds if seconds > 0 else None


def token_latencies(first_token_s: float | None, total_s: float, n_tokens: int) -> dict[str, float | None]:
    """TTFT and mean inter-token latency from a single request's timing."""
    itl = (total_s - first_token_s) / (n_tokens - 1) if first_token_s is not None and n_tokens > 1 else None
    return {"time_to_first_token_s": first_token_s, "inter_token_latency_s": itl}
