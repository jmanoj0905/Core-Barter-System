"""
Axis 5: STT backend latency (real audio through the real websocket pipeline).

Sends a real webm/opus-encoded speech clip (generate with sample_audio.sh)
over audio_pipeline's WS endpoint, closes the connection to force a buffer
flush, then polls the backend transcript endpoint until the transcript
lands. Measures true end-to-end: WS submit -> ffmpeg transcode -> STT
backend call -> POST to backend -> visible in GET /session/{id}/transcript.

NOTE ON SCOPE: STT_BACKEND is runtime-switchable via
POST /stt/config {"backend": ...}. This repo currently only has a working
AWS Transcribe path — the "whisper" backend's cached faster-whisper model
is broken (expired HF token, see audio_pipeline logs), so it cannot serve
as a real comparison point without separately fixing that model cache.
This script measures AWS only and says so explicitly rather than
fabricating a whisper number.
"""
import asyncio
import json
import sys
import time
from pathlib import Path

import requests
import websockets

from common import AUDIO_PIPELINE, BACKEND, require_services, percentiles, save_json, savefig

N_RUNS = 5
BARTER_ID_BASE = 970000
SAMPLE = Path(__file__).parent / "sample.webm"
POLL_TIMEOUT_S = 90


async def one_run(barter_id):
    audio_bytes = SAMPLE.read_bytes()
    t0 = time.perf_counter()

    ws_url = f"ws://localhost:8001/audio/{barter_id}/1"
    async with websockets.connect(ws_url) as ws:
        # split into a few chunks like a real MediaRecorder would emit
        chunk_size = max(len(audio_bytes) // 4, 1024)
        for i in range(0, len(audio_bytes), chunk_size):
            await ws.send(audio_bytes[i : i + chunk_size])
            await asyncio.sleep(0.05)
    # closing the ws triggers process_buffer() flush on the server

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


async def main():
    print("Checking services are actually up...")
    require_services("audio_pipeline", "backend")

    if not SAMPLE.exists():
        print(f"\n!!! {SAMPLE} not found. Generate it first:")
        print('  say -o sample.aiff "your sentence here"')
        print("  ffmpeg -y -i sample.aiff -ar 16000 -ac 1 -c:a libopus -f webm sample.webm")
        sys.exit(1)

    r = requests.post(f"{AUDIO_PIPELINE}/stt/config", json={"backend": "aws"}, timeout=10)
    r.raise_for_status()
    print(f"STT backend set to: {r.json()}")

    latencies = []
    transcripts = []
    for i in range(N_RUNS):
        barter_id = BARTER_ID_BASE + i
        print(f"\nRun {i+1}/{N_RUNS} (barter_id={barter_id}) — submitting real speech over WS...")
        try:
            elapsed, text = await one_run(barter_id)
        except Exception as e:
            print(f"  FAILED: {e}")
            continue
        print(f"  end-to-end latency: {elapsed*1000:.0f}ms  transcript: {text!r}")
        latencies.append(elapsed)
        transcripts.append(text)

    if not latencies:
        print("\n!!! ALL RUNS FAILED — AWS Transcribe path is not actually working. !!!")
        print("Check AWS credentials, S3 bucket permissions, and audio_pipeline logs")
        print("(`docker compose logs audio_pipeline`) before trusting any STT latency number.")
        sys.exit(1)

    stats = percentiles(latencies)
    stats["backend"] = "aws"
    stats["sample_transcripts"] = transcripts
    stats["note"] = (
        "whisper backend unavailable in this environment (cached model corrupted, "
        "HF token expired) — only AWS Transcribe measured. Not a fabricated comparison."
    )
    print(f"\nAWS Transcribe: p50={stats['p50_ms']:.0f}ms p95={stats['p95_ms']:.0f}ms n={stats['n']}")
    save_json("05_stt_backend_latency", stats)

    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(7, 5))
    ax.bar(["aws (measured)"], [stats["p50_ms"]], color="#4C72B0", label="p50")
    ax.bar(["aws (measured)"], [stats["p95_ms"] - stats["p50_ms"]], bottom=[stats["p50_ms"]], color="#DD8452", label="p95-p50")
    ax.bar(["whisper (unavailable)"], [0], color="lightgray")
    ax.text(1, max(stats["p95_ms"], 1) * 0.05, "model cache broken —\nnot measured", ha="center", color="gray")
    ax.set_ylabel("End-to-end latency (ms)")
    ax.set_title(f"STT end-to-end latency: WS submit -> transcript visible (n={stats['n']})")
    ax.legend()
    fig.tight_layout()
    savefig(fig, "05_stt_backend_latency")


if __name__ == "__main__":
    asyncio.run(main())
