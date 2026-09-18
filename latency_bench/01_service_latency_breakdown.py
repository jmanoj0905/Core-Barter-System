"""
Axis 1: Per-service latency breakdown.

Measures real request latency against each running service's actual
scoring/decision endpoint:
  - semantic_analysis  POST /ingest/segment   (SBERT embed + cosine-sim window scoring)
  - warning_engine     POST /window/result    (escalation decision)
  - backend             POST /session/create   (session lifecycle write)

Bar chart: p50 / p95 / p99 latency per service, grouped.
"""
import requests
import urllib3

from common import BACKEND, SEMANTIC, WARNING_ENGINE, require_services, timed, percentiles, save_json, savefig

urllib3.disable_warnings()

N_REQUESTS = 30
BARTER_ID_BASE = 900000


def bench_semantic_analysis():
    latencies = []
    for i in range(N_REQUESTS):
        barter_id = BARTER_ID_BASE + i
        requests.post(
            f"{SEMANTIC}/session/{barter_id}/contract",
            json={
                "barter_id": barter_id,
                "topic": "python decorators",
                "scope": "explain decorators, closures, and functools.wraps",
                "teacher_user_id": 1,
                "learner_user_id": 2,
            },
            timeout=10,
        ).raise_for_status()

        def call():
            r = requests.post(
                f"{SEMANTIC}/ingest/segment",
                json={
                    "barter_id": barter_id,
                    "user_id": 1,
                    "text": (
                        "A decorator in Python is a function that wraps another "
                        "function to extend its behavior without modifying its "
                        "source code directly, commonly used with functools.wraps "
                        "to preserve metadata."
                    ),
                    "duration_seconds": 26.0,
                    "timestamp_start": 0.0,
                    "timestamp_end": 26.0,
                },
                timeout=15,
            )
            r.raise_for_status()
            return r.json()

        _, elapsed = timed(call)
        latencies.append(elapsed)
    return latencies


def bench_warning_engine():
    latencies = []
    for i in range(N_REQUESTS):
        barter_id = BARTER_ID_BASE + 100000 + i
        requests.post(f"{WARNING_ENGINE}/session/{barter_id}/init", json={}, timeout=10)

        def call():
            r = requests.post(
                f"{WARNING_ENGINE}/window/result",
                json={
                    "barter_id": barter_id,
                    "window_id": 1,
                    "classification": "incorrect",
                    "similarity_score": 0.12,
                    "timestamp_start": 0.0,
                    "timestamp_end": 25.0,
                    "text_preview": "talking about lunch plans",
                },
                timeout=10,
            )
            r.raise_for_status()
            return r.json()

        _, elapsed = timed(call)
        latencies.append(elapsed)
    return latencies


def bench_backend():
    latencies = []
    for i in range(N_REQUESTS):

        def call():
            r = requests.post(
                f"{BACKEND}/session/create",
                json={
                    "skill_a": "python",
                    "skill_b": "spanish",
                    "topic": "python decorators",
                    "scope": "explain decorators and closures",
                    "agreed_duration_minutes": 15,
                    "teacher_user_id": 1,
                    "learner_user_id": 2,
                },
                timeout=15,
            )
            r.raise_for_status()
            return r.json()

        _, elapsed = timed(call)
        latencies.append(elapsed)
    return latencies


def main():
    print("Checking services are actually up...")
    require_services("semantic_analysis", "warning_engine", "backend")

    print(f"\nBenchmarking semantic_analysis /ingest/segment ({N_REQUESTS} requests)...")
    sem_lat = bench_semantic_analysis()

    print(f"Benchmarking warning_engine /window/result ({N_REQUESTS} requests)...")
    warn_lat = bench_warning_engine()

    print(f"Benchmarking backend /session/create ({N_REQUESTS} requests)...")
    backend_lat = bench_backend()

    results = {
        "semantic_analysis": percentiles(sem_lat),
        "warning_engine": percentiles(warn_lat),
        "backend": percentiles(backend_lat),
    }
    for name, stats in results.items():
        print(f"  {name}: p50={stats['p50_ms']:.1f}ms p95={stats['p95_ms']:.1f}ms p99={stats['p99_ms']:.1f}ms")
    save_json("01_service_latency_breakdown", results)

    import matplotlib.pyplot as plt
    import numpy as np

    services = list(results.keys())
    p50 = [results[s]["p50_ms"] for s in services]
    p95 = [results[s]["p95_ms"] for s in services]
    p99 = [results[s]["p99_ms"] for s in services]

    x = np.arange(len(services))
    width = 0.25
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.bar(x - width, p50, width, label="p50")
    ax.bar(x, p95, width, label="p95")
    ax.bar(x + width, p99, width, label="p99")
    ax.set_xticks(x)
    ax.set_xticklabels(services)
    ax.set_ylabel("Latency (ms)")
    ax.set_title(f"Per-service latency (real endpoints, n={N_REQUESTS} each)")
    ax.legend()
    fig.tight_layout()
    savefig(fig, "01_service_latency_breakdown")


if __name__ == "__main__":
    main()
