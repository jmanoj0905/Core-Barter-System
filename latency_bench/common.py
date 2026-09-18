"""Shared helpers for latency benchmark scripts.

All scripts in this dir hit the REAL running docker compose services
(exposed on host via docker-compose.override.yml) and refuse to fabricate
numbers if a service is unreachable.
"""
import json
import sys
import time
from pathlib import Path

import numpy as np
import requests

RESULTS_DIR = Path(__file__).parent / "results"
RESULTS_DIR.mkdir(exist_ok=True)

BACKEND = "http://localhost:8000"
AUDIO_PIPELINE = "http://localhost:8001"
SEMANTIC = "http://localhost:8002"
WARNING_ENGINE = "http://localhost:8003"
NGINX = "https://localhost"

SERVICES = {
    "backend": f"{BACKEND}/docs",
    "audio_pipeline": f"{AUDIO_PIPELINE}/docs",
    "semantic_analysis": f"{SEMANTIC}/docs",
    "warning_engine": f"{WARNING_ENGINE}/docs",
    "nginx/frontend": f"{NGINX}/",
}


def require_services(*names):
    """Abort loudly if any required service isn't actually responding."""
    down = []
    for name in names:
        url = SERVICES[name]
        try:
            r = requests.get(url, timeout=5, verify=False)
            if r.status_code >= 500:
                down.append(f"{name} ({url}) -> HTTP {r.status_code}")
        except requests.exceptions.RequestException as e:
            down.append(f"{name} ({url}) -> {e.__class__.__name__}: {e}")
    if down:
        print("\n!!! SERVICE(S) DOWN — cannot produce a rooted-in-reality result !!!")
        for d in down:
            print(f"  - {d}")
        print("\nFix the service(s) above and re-run. Refusing to fabricate data.\n")
        sys.exit(1)


def timed(fn, *args, **kwargs):
    """Run fn, return (result, elapsed_seconds). Raises through on error."""
    t0 = time.perf_counter()
    result = fn(*args, **kwargs)
    return result, time.perf_counter() - t0


def percentiles(latencies_s):
    arr = np.array(latencies_s) * 1000.0  # -> ms
    return {
        "n": len(arr),
        "mean_ms": float(np.mean(arr)),
        "p50_ms": float(np.percentile(arr, 50)),
        "p95_ms": float(np.percentile(arr, 95)),
        "p99_ms": float(np.percentile(arr, 99)),
        "min_ms": float(np.min(arr)),
        "max_ms": float(np.max(arr)),
    }


def save_json(name, data):
    path = RESULTS_DIR / f"{name}.json"
    path.write_text(json.dumps(data, indent=2))
    print(f"  raw data -> {path}")


def savefig(fig, name):
    png_path = RESULTS_DIR / f"{name}.png"
    fig.savefig(png_path, dpi=150, bbox_inches="tight")
    jpg_path = RESULTS_DIR / f"{name}.jpg"
    fig.savefig(jpg_path, dpi=150, bbox_inches="tight", facecolor="white")
    print(f"  graph -> {png_path}, {jpg_path}")
