import argparse
import csv
import random

LEVELS = {"high": 0.85, "medium": 0.5, "low": 0.15}
NOISE_SIGMA = 0.1
DISAGREEMENT_FRACTION = 0.2


def _clamp01(x: float) -> float:
    return max(0.0, min(1.0, x))


def generate_corpus(n_scenarios: int, seed: int) -> list[dict]:
    rng = random.Random(seed)
    level_names = list(LEVELS)
    rows = []
    for scenario_id in range(n_scenarios):
        speech_level_name = rng.choice(level_names)
        if rng.random() < DISAGREEMENT_FRACTION:
            video_level_name = rng.choice([n for n in level_names if n != speech_level_name])
        else:
            video_level_name = speech_level_name
        speech_level = LEVELS[speech_level_name]
        video_level = LEVELS[video_level_name]
        speech_score = _clamp01(rng.gauss(speech_level, NOISE_SIGMA))
        video_score = _clamp01(rng.gauss(video_level, NOISE_SIGMA))
        true_level = (speech_level + video_level) / 2.0
        rows.append({
            "scenario_id": scenario_id,
            "speech_score": round(speech_score, 4),
            "video_score": round(video_score, 4),
            "engaged_label": 1 if true_level >= 0.5 else 0,
        })
    return rows


def main(argv=None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--n-scenarios", type=int, default=200)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output", default="fusion_corpus.csv")
    args = parser.parse_args(argv)

    rows = generate_corpus(n_scenarios=args.n_scenarios, seed=args.seed)
    with open(args.output, "w", newline="", encoding="utf-8") as output:
        writer = csv.DictWriter(
            output,
            fieldnames=["scenario_id", "speech_score", "video_score", "engaged_label"],
        )
        writer.writeheader()
        writer.writerows(rows)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
