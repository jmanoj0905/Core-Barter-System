"""
Axis 3: Latency vs concurrent sessions (scalability).

Fires N concurrent POST /session/create requests at the real backend
for N in {1, 5, 10, 20} and records latency distribution + achieved
throughput at each concurrency level.

Line chart: p50/p95 latency vs concurrency level.
"""
import concurrent.futures as cf

import requests
import urllib3

from common import BACKEND, require_services, timed, percentiles, save_json, savefig

urllib3.disable_warnings()

CONCURRENCY_LEVELS = [1, 5, 10, 20]
REQUESTS_PER_LEVEL = 40  # total requests fired at each concurrency level


def payload():
    return {
        "skill_a": "python",
        "skill_b": "spanish",
        "topic": "python decorators",
        "scope": "explain decorators and closures",
        "agreed_duration_minutes": 15,
        "teacher_user_id": 1,
        "learner_user_id": 2,
    }


def one_call():
    def call():
        r = requests.post(f"{BACKEND}/session/create", json=payload(), timeout=30)
        r.raise_for_status()
        return r.json()

    _, elapsed = timed(call)
    return elapsed


def bench_at_concurrency(level):
    latencies = []
    import time

    t0 = time.perf_counter()
    with cf.ThreadPoolExecutor(max_workers=level) as pool:
        futures = [pool.submit(one_call) for _ in range(REQUESTS_PER_LEVEL)]
        for f in cf.as_completed(futures):
            latencies.append(f.result())
    wall = time.perf_counter() - t0
    throughput = REQUESTS_PER_LEVEL / wall
    return latencies, throughput


def main():
    print("Checking services are actually up...")
    require_services("backend")

    results = {}
    for level in CONCURRENCY_LEVELS:
        print(f"\nBenchmarking at concurrency={level} ({REQUESTS_PER_LEVEL} requests)...")
        latencies, throughput = bench_at_concurrency(level)
        stats = percentiles(latencies)
        stats["throughput_req_per_s"] = throughput
        results[str(level)] = stats
        print(f"  p50={stats['p50_ms']:.1f}ms p95={stats['p95_ms']:.1f}ms throughput={throughput:.1f} req/s")

    save_json("03_concurrency_scaling", results)

    import matplotlib.pyplot as plt

    levels = CONCURRENCY_LEVELS
    p50 = [results[str(l)]["p50_ms"] for l in levels]
    p95 = [results[str(l)]["p95_ms"] for l in levels]
    throughput = [results[str(l)]["throughput_req_per_s"] for l in levels]

    fig, ax1 = plt.subplots(figsize=(8, 5))
    ax1.plot(levels, p50, marker="o", label="p50 latency")
    ax1.plot(levels, p95, marker="s", label="p95 latency")
    ax1.set_xlabel("Concurrent sessions")
    ax1.set_ylabel("Latency (ms)")
    ax1.set_xticks(levels)

    ax2 = ax1.twinx()
    ax2.plot(levels, throughput, marker="^", color="green", linestyle="--", label="throughput")
    ax2.set_ylabel("Throughput (req/s)")

    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc="upper left")

    ax1.set_title(f"backend POST /session/create — latency & throughput vs concurrency (n={REQUESTS_PER_LEVEL}/level)")
    fig.tight_layout()
    savefig(fig, "03_concurrency_scaling")


if __name__ == "__main__":
    main()
