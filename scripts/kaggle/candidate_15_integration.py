"""Stage dispatcher: candidate_15_integration.py MODE   MODE: static | engine | server | summary. Each mode is a separate process (fresh CUDA context, clean accounting)."""
import sys
from _common import run_guarded

if __name__ == "__main__":
    import multiprocessing; multiprocessing.freeze_support()
    if len(sys.argv) < 2 or sys.argv[1] not in ("static", "engine", "parallel", "server", "summary"):
        raise SystemExit(__doc__)
    mode = sys.argv[1]
    mod = __import__(f"candidate_15_{mode}")
    run_guarded(f"15c_{mode}", getattr(mod, mode))
