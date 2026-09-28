# Documentation index

Start with [architecture.md](architecture.md) for the shape of the system, or
[development.md](development.md) if you just want it running.

## Current

| Document | Covers |
|---|---|
| [architecture.md](architecture.md) | Services, ports, data flow, session lifecycle, escrow flow, known limitations |
| [development.md](development.md) | Building, running, configuring and testing locally |
| [issues.md](issues.md) | Issue register (ISSUE-001…027) with evidence class, priority and repair order |
| [design-system.md](design-system.md) | Frontend visual language — colors, typography, components, motion |
| [threshold-calibration.md](threshold-calibration.md) | Every tuned threshold and weight in one table, each pointing at the experiment that produced it |

## Subsystems

| Document | Covers |
|---|---|
| [video_engagement/design-choices.md](video_engagement/design-choices.md) | Sub-signal formula, weight fitting on labeled public frames, dataset licences, limitations |
| [eval_dataset/aws-setup.md](eval_dataset/aws-setup.md) | AWS account, IAM policy and end-to-end run for the eval-dataset tooling |
| [../eval_dataset/README.md](../eval_dataset/README.md) | The evaluation corpus itself — layout, sources, tools |

Semantic-analysis research notes live next to the scripts that produced them, in
[`apps/semantic_analysis/ground_truth/`](../apps/semantic_analysis/ground_truth/):

- `design_decisions.md` — threshold calibration decisions D1–D5
- `labeling_rubric.md` — how windows were labeled
- `threshold_experiment_findings.md` — frozen-SBERT threshold sweep
- `rho_calibration_findings.md` — the relative threshold `thr = RHO · R`
- `analogy_handling_findings.md` — how the classifier behaves on sustained analogies
- `finetune_spike_findings.md`, `finetune_alternatives_research.md` — why SBERT was left frozen

## Specs and plans

[superpowers/specs/](superpowers/specs/) holds design specs and [superpowers/plans/](superpowers/plans/)
the implementation plans written against them, named by date and feature. These are the authority
for work that is in progress or recently landed.

## Archive

[archive/](archive/) holds superseded documents — the original decision log, implementation plan,
build roadmap, and the long-form technical reference. They are historical context, not a
description of the current system; see [archive/README.md](archive/README.md) for how each one
disagrees with the code.

[archive/technical-details-2026-08.md](archive/technical-details-2026-08.md) is worth knowing
about despite being archived: it is still the deepest writeup of per-service internals, the key
data flows and the trust/QA formulas. Nothing here has replaced it, and rewriting it against the
current code is outstanding work.
