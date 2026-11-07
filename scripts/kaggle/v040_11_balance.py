"""Stage 11 (0.4.0): load-balance + strategy comparison from bench_*.json written earlier in the SAME session by v040_bench.py."""
import json
from _common import OUT, check, finish, run_guarded, save, skipped, versions, warn


def main() -> int:
    from stokastisk.benchmarks.gpu_balance import from_replica_record
    from stokastisk.benchmarks.report import comparison_table, relative_to
    R = []; recs = {}
    for mode in ("tp1", "tp2", "pp2", "replicas2"):
        p = OUT / f"bench_{mode}_standard.json"
        recs[mode] = json.loads(p.read_text()) if p.exists() else {"status": "NOT_RUN"}
        measured = recs[mode].get("status") == "MEASURED"
        detail = recs[mode].get("error") or recs[mode].get("status")
        if mode in ("tp1", "replicas2"):              # mandatory comparison points
            check(R, f"{mode}: benchmark MEASURED", measured, detail)
        elif measured:
            check(R, f"{mode}: benchmark MEASURED", True)
        else:                                          # tp2 / pp2: reported, never silently dropped
            skipped(R, f"{mode}: benchmark", detail)
    tables, rel, bal = {}, {}, {}
    for w in ("short", "long"):
        tables[w] = comparison_table(recs, w); rel[w] = relative_to(recs, w)
    if recs["replicas2"].get("status") == "MEASURED":
        for w in ("short", "long"):
            bal[w] = from_replica_record(recs["replicas2"], w)
            check(R, f"{w}: balance computed for 2 replicas", bal[w].get("status") == "MEASURED", bal[w].get("compute_imbalance_percent"))
            if bal[w].get("status") == "MEASURED" and bal[w]["compute_imbalance_percent"] > 25:
                warn(R, f"{w}: replica compute imbalance above 25%", f'{bal[w]["compute_imbalance_percent"]:.1f}%')
    if recs["pp2"].get("status") == "MEASURED":
        bal["pp2_proxy"] = {w: r.get("balance_proxy") for w, r in recs["pp2"]["results"].items()}
    save("11_tables.md.json", tables)
    for t in tables.values(): print(t, "\n")
    finish("11_balance", R, {"versions": versions(), "relative_to_tp1": rel, "balance": bal, "tables_markdown": tables,
                             "caveat": "GPU utilisation percentages alone do not prove balanced computation; replica balance uses worker-reported compute seconds, PP balance is a labelled proxy"})
    return 0


if __name__ == "__main__":
    run_guarded("11_balance", main)
