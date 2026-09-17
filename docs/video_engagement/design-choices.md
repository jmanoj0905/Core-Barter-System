# Video Engagement — Weight Design Choices

Status: **placeholder — not yet run against real data.**

`apps/video_engagement/scoring.py`'s `DEFAULT_WEIGHTS = (0.4, 0.4, 0.2)` and
`warning_engine`'s `ENGAGEMENT_FUSION_W_SPEECH=0.7` / `ENGAGEMENT_FUSION_W_VIDEO=0.3`
are engineering guesses, not experimentally chosen values.

Run `python apps/video_engagement/weight_search.py --barter-id <id>` against a
pilot session's real data to replace this file with actual grid-search
results and a chosen weight set, per the method in
`docs/superpowers/specs/2026-09-17-video-engagement-design.md`.
