"""
Axis 2: Network hop overhead — direct-to-service vs through nginx proxy.

Same backend endpoint (POST /session/create), hit two ways:
  - direct:  http://localhost:8000/session/create
  - via nginx: https://localhost/session/create  (TLS termination + proxy_pass)

Bar chart: p50/p95/p99 latency, direct vs nginx.
"""
import requests
import urllib3

from common import BACKEND, NGINX, require_services, timed, percentiles, save_json, savefig

urllib3.disable_warnings()

N_REQUESTS = 30


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


def bench(url):
    latencies = []
    for _ in range(N_REQUESTS):
        def call():
            r = requests.post(url, json=payload(), timeout=15, verify=False)
            r.raise_for_status()
            return r.json()

        _, elapsed = timed(call)
        latencies.append(elapsed)
    return latencies


def main():
    print("Checking services are actually up...")
    require_services("backend", "nginx/frontend")

    print(f"\nBenchmarking direct backend ({N_REQUESTS} requests)...")
    direct_lat = bench(f"{BACKEND}/session/create")

    print(f"Benchmarking via nginx proxy ({N_REQUESTS} requests)...")
    nginx_lat = bench(f"{NGINX}/session/create")

    results = {
        "direct_backend": percentiles(direct_lat),
        "via_nginx": percentiles(nginx_lat),
    }
    for name, stats in results.items():
        print(f"  {name}: p50={stats['p50_ms']:.1f}ms p95={stats['p95_ms']:.1f}ms p99={stats['p99_ms']:.1f}ms")
    overhead_p50 = results["via_nginx"]["p50_ms"] - results["direct_backend"]["p50_ms"]
    print(f"  nginx hop overhead (p50): {overhead_p50:+.1f}ms")
    results["nginx_overhead_p50_ms"] = overhead_p50
    save_json("02_nginx_hop_overhead", results)

    import matplotlib.pyplot as plt
    import numpy as np

    labels = ["direct_backend", "via_nginx"]
    p50 = [results[l]["p50_ms"] for l in labels]
    p95 = [results[l]["p95_ms"] for l in labels]
    p99 = [results[l]["p99_ms"] for l in labels]
    x = np.arange(len(labels))
    width = 0.25
    fig, ax = plt.subplots(figsize=(7, 5))
    ax.bar(x - width, p50, width, label="p50")
    ax.bar(x, p95, width, label="p95")
    ax.bar(x + width, p99, width, label="p99")
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylabel("Latency (ms)")
    ax.set_title(f"nginx proxy overhead on POST /session/create (n={N_REQUESTS} each)")
    ax.legend()
    fig.tight_layout()
    savefig(fig, "02_nginx_hop_overhead")


if __name__ == "__main__":
    main()
