"""Reproducible dual-GPU benchmarks (import lazily: they need vLLM / CUDA only when run)."""
from .baseline import LONG, SCALING, SHORT, Workload
from .report import comparison_table, relative_to

__all__ = ["Workload", "SHORT", "LONG", "SCALING", "comparison_table", "relative_to"]
