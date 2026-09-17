# Video Engagement — Weight Design Choices

Status: **placeholder — not yet run against real data.**

`apps/video_engagement/scoring.py`'s `DEFAULT_WEIGHTS = (0.4, 0.4, 0.2)` and
`warning_engine`'s `ENGAGEMENT_FUSION_W_SPEECH=0.7` / `ENGAGEMENT_FUSION_W_VIDEO=0.3`
are engineering guesses, not experimentally chosen values.

Run `python apps/video_engagement/weight_search.py --barter-id <id>` against a
pilot session's real data to replace this file with actual grid-search
results and a chosen weight set, per the method in
`docs/superpowers/specs/2026-09-17-video-engagement-design.md`.

## Known bias: local vs aws sample differently

`local` scores are the mean of sub-signals across every frame captured in
the ~5s window (`apps/video_engagement/main.py`, `process_buffer`'s
`local`/`both` branch). `aws` scores come from a single mid-window frame
only (`frames[len(frames) // 2]`), to limit Rekognition API call volume —
this is a deliberate cost tradeoff, not a bug.

Consequence: in `both` mode, `local` and `aws` are not directly comparable
per-window — a blink or head-turn that happens to land on the sampled
middle frame can swing the `aws` score for a window where the `local`
average stays stable, and vice versa. Any local-vs-cloud accuracy
comparison drawn from pilot data should account for this before treating a
per-window discrepancy as a backend-quality signal rather than a sampling
artifact.
