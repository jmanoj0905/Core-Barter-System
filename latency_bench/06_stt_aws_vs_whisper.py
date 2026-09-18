"""
Axis 6: STT backend comparison — AWS Transcribe vs local Whisper (faster-whisper large-v3).

Same real speech clip, same real WebSocket pipeline, only the backend
flips via POST /stt/config. Requires the audio_pipeline container to
have been started with STT_BACKEND=whisper (Whisper model is only
loaded once, at container startup lifespan) so the whisper path
actually has a loaded model to call.

Bar chart: p50/p95 end-to-end latency, aws vs whisper.
"""
import asyncio
import sys
import time
from pathlib import Path

import requests
import websockets

from common import AUDIO_PIPELINE, BACKEND, require_services, percentiles, save_json, savefig

N_RUNS = 5
BARTER_ID_BASE = {"aws": 980000, "whisper": 981000}
SAMPLE = Path(__file__).parent / "sample.webm"
POLL_TIMEOUT_S = 90


async def one_run(barter_id):
    audio_bytes = SAMPLE.read_bytes()
    t0 = time.perf_counter()

    ws_url = f"ws://localhost:8001/audio/{barter_id}/1"
    async with websockets.connect(ws_url) as ws:
        chunk_size = max(len(audio_bytes) // 4, 1024)
        for i in range(0, len(audio_bytes), chunk_size):
            await ws.send(audio_bytes[i : i + chunk_size])
            await asyncio.sleep(0.05)

    deadline = time.time() + POLL_TIMEOUT_S
    while time.time() < deadline:
        r = requests.get(f"{BACKEND}/session/{barter_id}/transcript", timeout=10)
        r.raise_for_status()
        segments = r.json()
        if segments:
            elapsed = time.perf_counter() - t0
            return elapsed, segments[0]["text"]
        await asyncio.sleep(0.5)
    raise TimeoutError(f"no transcript appeared within {POLL_TIMEOUT_S}s for barter_id={barter_id}")


async def bench_backend(name):
    r = requests.post(f"{AUDIO_PIPELINE}/stt/config", json={"backend": name}, timeout=10)
    r.raise_for_status()
    print(f"STT backend set to: {r.json()}")

    latencies, transcripts = [], []
    for i in range(N_RUNS):
        barter_id = BARTER_ID_BASE[name] + i
        print(f"  [{name}] run {i+1}/{N_RUNS} (barter_id={barter_id})...")
        try:
            elapsed, text = await one_run(barter_id)
        except Exception as e:
            print(f"    FAILED: {e}")
            continue
        print(f"    {elapsed*1000:.0f}ms  transcript: {text!r}")
        latencies.append(elapsed)
        transcripts.append(text)
    return latencies, transcripts


async def main():
    print("Checking services are actually up...")
    require_services("audio_pipeline", "backend")

    if not SAMPLE.exists():
        print(f"\n!!! {SAMPLE} not found. Generate it first (see 05_stt_backend_latency.py docstring).")
        sys.exit(1)

    results = {}
    for name in ["aws", "whisper"]:
        print(f"\n=== Benchmarking {name} ===")
        latencies, transcripts = await bench_backend(name)
        if not latencies:
            print(f"\n!!! ALL {name.upper()} RUNS FAILED — not a real comparison point. !!!")
            print(f"Check `docker compose logs audio_pipeline` before trusting any {name} number.")
            sys.exit(1)
        stats = percentiles(latencies)
        stats["sample_transcripts"] = transcripts
        results[name] = stats
        print(f"{name}: p50={stats['p50_ms']:.0f}ms p95={stats['p95_ms']:.0f}ms n={stats['n']}")

    save_json("06_stt_aws_vs_whisper", results)

    import matplotlib.pyplot as plt
    import numpy as np

    backends = ["aws", "whisper"]
    p50 = [results[b]["p50_ms"] for b in backends]
    p95 = [results[b]["p95_ms"] for b in backends]
    x = np.arange(len(backends))
    width = 0.35
    fig, ax = plt.subplots(figsize=(7, 5))
    ax.bar(x - width / 2, p50, width, label="p50")
    ax.bar(x + width / 2, p95, width, label="p95")
    ax.set_xticks(x)
    ax.set_xticklabels(["AWS Transcribe", "Whisper large-v3 (local, int8)"])
    ax.set_ylabel("End-to-end latency (ms)")
    ax.set_title(f"STT backend comparison — real speech, real WS pipeline (n={N_RUNS}/backend)")
    ax.legend()
    fig.tight_layout()
    savefig(fig, "06_stt_aws_vs_whisper")


if __name__ == "__main__":
    asyncio.run(main())
