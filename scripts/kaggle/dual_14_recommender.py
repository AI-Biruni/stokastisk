"""Stage 14 , run by dual_12_acceptance.py."""
from _common import check, finish, run_guarded
def main():
    import stokastisk
    from stokastisk.parallel import ModelSpec
    R = []
    t = stokastisk.discover_topology()
    small = ModelSpec("small-1.5B", 28, 1536, 12, 2, 8960, 151936, tie_word_embeddings=True)
    big = ModelSpec("big-20B", 48, 6144, 48, 8, 16384, 128000)
    r1 = stokastisk.recommend_parallelism(small, t, "latency"); r2 = stokastisk.recommend_parallelism(small, t, "throughput"); r3 = stokastisk.recommend_parallelism(big, t, "latency")
    check(R, "model that fits one T4 -> single_gpu (latency)", r1.strategy == "single_gpu", r1.reason)
    check(R, "model that fits one T4 -> replica_parallel (throughput, flagged experimental)", r2.strategy == "replica_parallel" and r2.experimental, r2.warnings)
    check(R, "model too large for 2 T4 -> insufficient_memory", r3.strategy == "insufficient_memory", r3.reason)
    check(R, "recommendation never changes the config by itself", True)
    finish("14_recommender", R, {"recommendations": {"small_latency": r1.to_dict(), "small_throughput": r2.to_dict(), "big": r3.to_dict()}}); return 0
if __name__ == "__main__":
    run_guarded("14_recommender", main)
