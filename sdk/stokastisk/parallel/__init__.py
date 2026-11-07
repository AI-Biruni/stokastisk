"""Application-level parallelism: independent GPU replicas, token-aware scheduling, rule-based recommendations."""
from .memory import ModelSpec, load_model_spec
from .queue import Request, RequestQueue, estimate_tokens
from .recommender import Recommendation, recommend_parallelism
from .routing import WorkerLoad, choose_worker
from .scheduler import Scheduler


def __getattr__(name):          # ReplicaPool pulls in multiprocessing machinery; import it lazily
    if name == "ReplicaPool":
        from .replica import ReplicaPool
        return ReplicaPool
    raise AttributeError(name)


__all__ = ["ModelSpec", "load_model_spec", "Request", "RequestQueue", "estimate_tokens", "Recommendation", "recommend_parallelism",
           "WorkerLoad", "choose_worker", "Scheduler", "ReplicaPool"]
