"""
Axis 4: Semantic scoring latency vs transcript length.

audio_pipeline's window threshold (BUFFER_THRESHOLD_SECONDS=5s wall-clock)
and semantic_analysis's window threshold (WINDOW_DURATION_THRESHOLD=25s) are
hardcoded, not runtime-configurable, so a true "5s vs 10s window" A/B isn't
possible without code changes + container restarts per config.

What IS measurable live: how SBERT embed+cosine-sim latency in
semantic_analysis /ingest/segment scales with transcript length (word count) —
a direct proxy for "longer window = more text per scoring call".

Line chart: latency vs transcript word count.
"""
import requests

from common import SEMANTIC, require_services, timed, percentiles, save_json, savefig

N_PER_LENGTH = 15
BARTER_ID_BASE = 950000

BASE_SENTENCE = (
    "A decorator in Python is a function that wraps another function to "
    "extend its behavior without modifying its source code directly. "
)

WORD_COUNTS = [10, 25, 50, 100, 200]


def transcript_of_length(target_words):
    words = (BASE_SENTENCE * ((target_words // len(BASE_SENTENCE.split())) + 2)).split()
    return " ".join(words[:target_words])


def bench_length(word_count, offset):
    latencies = []
    for i in range(N_PER_LENGTH):
        barter_id = BARTER_ID_BASE + offset * 1000 + i
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

        text = transcript_of_length(word_count)

        def call():
            r = requests.post(
                f"{SEMANTIC}/ingest/segment",
                json={
                    "barter_id": barter_id,
                    "user_id": 1,
                    "text": text,
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


def main():
    print("Checking services are actually up...")
    require_services("semantic_analysis")

    results = {}
    for idx, wc in enumerate(WORD_COUNTS):
        print(f"\nBenchmarking transcript length={wc} words ({N_PER_LENGTH} requests)...")
        latencies = bench_length(wc, idx)
        stats = percentiles(latencies)
        results[str(wc)] = stats
        print(f"  p50={stats['p50_ms']:.1f}ms p95={stats['p95_ms']:.1f}ms")

    save_json("04_transcript_length_latency", results)

    import matplotlib.pyplot as plt

    p50 = [results[str(wc)]["p50_ms"] for wc in WORD_COUNTS]
    p95 = [results[str(wc)]["p95_ms"] for wc in WORD_COUNTS]

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(WORD_COUNTS, p50, marker="o", label="p50")
    ax.plot(WORD_COUNTS, p95, marker="s", label="p95")
    ax.set_xlabel("Transcript length (words)")
    ax.set_ylabel("Latency (ms)")
    ax.set_title(f"semantic_analysis /ingest/segment — latency vs transcript length (n={N_PER_LENGTH}/point)")
    ax.legend()
    fig.tight_layout()
    savefig(fig, "04_transcript_length_latency")


if __name__ == "__main__":
    main()
