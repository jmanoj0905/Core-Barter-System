# Eval Dataset Design — Scripted & Synthesized Barter Sessions

> Date: 2026-08-11
> Status: Approved
> Feeds: Research Gap & Novelty Action Plan (`00_Research Gap.md`), Phase 2 (build eval dataset) and Phase 3 (run baselines).

## Goal

Build a 40-session labeled evaluation dataset to support head-to-head baseline
comparisons (SBERT bi-encoder vs cross-encoder vs LLM-judge vs NeMo TopicControl
vs BERTopic) and later FP/FN escalation-ladder analysis. Sessions are
scripted and synthesized (not live-recorded), produced in parallel by a
4-person team with zero-collision task partitioning.

This directly unblocks Research Gap Section 4, Phase 2 ("build the evaluation
dataset — non-negotiable, no paper survives without it") and sets up Phase 3
(baseline runs).

## Team split

- 4 people, each independently scripts + synthesizes **10 sessions** = 40 total.
- Each person's 10 sessions cover all 5 scenario categories, 2 sessions per
  category — self-contained mini-dataset per person, resilient to any one
  person falling behind, and reviewable independently.
- Categories: `clean` (on-topic), `gradual_drift`, `adversarial` (off-topic),
  `code_switch` (topic-adjacent/pedagogical drift), `silence` (low engagement).
- Each person is pre-assigned 10 topics from a shared 40-topic pool (below) —
  no overlap, no coordination needed mid-work.
- Annotation is cross-person: each session gets 2-3 annotators drawn from the
  *other* three team members, never the session's own author — avoids
  self-bias in ground truth.

## Repo layout

```
eval_dataset/
├── topics/topic_pool.md              # 40 skill-pair topics, pre-split 4x10
├── scripts/
│   ├── person_A/sess_A01.txt … sess_A10.txt
│   ├── person_B/sess_B01.txt … sess_B10.txt
│   ├── person_C/...
│   └── person_D/...
├── audio/                            # Polly-synthesized .wav, Git LFS tracked
│   └── sess_A01.wav ...
├── transcripts/
│   ├── raw/sess_A01_wer0.json        # real AWS Transcribe output
│   └── synthetic/sess_A01_wer{10,20,30}.json
├── annotations/
│   └── sess_A01_annotator{1,2,3}.json
└── tools/
    ├── synthesize.py                 # script -> Polly audio
    ├── transcribe.py                 # audio -> AWS Transcribe -> raw transcript
    ├── inject_wer.py                 # raw transcript -> WER variants
    ├── compute_kappa.py              # annotations -> inter-annotator agreement
    └── annotation_app/               # small React+FastAPI labeling tool
```

Audio (`.wav`) is tracked via Git LFS. Everything else (scripts, transcripts,
annotations) is plain text/JSON in normal git — the whole dataset lives in
this repo, no external storage system.

## Topic pool (40 skill pairs, 10 per person)

Each pair is `Teaches ↔ Wants to learn`. No topic repeats across the pool.

**Person A**
1. Python fundamentals ↔ Guitar basics
2. Excel/spreadsheets ↔ Watercolor painting
3. Spanish conversation ↔ Chess openings
4. Public speaking ↔ Web design basics
5. Statistics ↔ Photography composition
6. JavaScript basics ↔ Creative writing
7. Yoga fundamentals ↔ Personal finance/budgeting
8. Cooking (Italian) ↔ Music theory
9. Resume/interview coaching ↔ Video editing basics
10. SQL basics ↔ Sketching/drawing

**Person B**
1. French conversation ↔ Data visualization
2. Guitar (intermediate) ↔ Excel macros
3. Digital marketing basics ↔ Pottery
4. Calculus tutoring ↔ Salsa dancing
5. Git/version control ↔ Piano basics
6. Negotiation skills ↔ Mandarin basics
7. UX design basics ↔ Home baking
8. Machine learning intro ↔ Watercolor painting (advanced)
9. Photography (portrait) ↔ Excel pivot tables
10. Meditation/mindfulness ↔ React/frontend basics

**Person C**
1. German conversation ↔ Drumming basics
2. Investing basics ↔ Illustration/digital art
3. Public speaking (advanced) ↔ Docker/containers
4. Chess strategy (advanced) ↔ Copywriting
5. Node.js basics ↔ Ceramics
6. Wine tasting fundamentals ↔ Excel dashboards
7. Japanese conversation ↔ Statistics for beginners
8. Interior design basics ↔ Python data analysis
9. Voice/singing basics ↔ Time management
10. Woodworking basics ↔ Social media strategy

**Person D**
1. Italian conversation ↔ Personal branding
2. Poker strategy ↔ Figma/UI design
3. Gardening basics ↔ Podcasting/audio editing
4. Algebra tutoring ↔ Improv/acting basics
5. Cloud computing basics (AWS) ↔ Watercolor sketching
6. Korean conversation ↔ Excel VBA
7. Running/fitness coaching ↔ Music production basics
8. Negotiation (sales) ↔ Origami
9. Blockchain basics ↔ Calligraphy
10. Debate/argumentation ↔ Basic accounting

Each person assigns categories freely across their 10 topics (2 per category),
noted in each script's `CATEGORY` field.

## Script format

Plain screenplay-style `.txt`, one file per session:

```
TOPIC: Python fundamentals (variables, loops, functions)
TEACHER: A
LEARNER: B
CATEGORY: gradual_drift

A: Alright, let's start with variables. In Python you just write x = 5, no type declaration needed.
B: Oh nice, so it's dynamically typed?
A: Exactly. Now let's look at loops — for and while.
...
```

`CATEGORY` sets author intent (what kind of drift the script is meant to
exercise) but is not the ground truth label — ground truth comes from
independent annotation (see below). Target length: 5-8 minutes, ~15-20
dialogue turns per session — enough turns for real drift/return dynamics
(~60-100 five-second windows) without making scripting/annotation slow.

## TTS synthesis — `synthesize.py`

- Input: one script `.txt` file.
- Assigns each speaker (teacher/learner) a distinct AWS Polly neural voice,
  consistent for that session.
- Synthesizes each turn via AWS Polly, concatenates with ~0.5-1s natural
  pauses into one `.wav` per session.
- Emits `turns_timing.json`: per-line start/end timestamps, used later to
  align 5-second windows back to script-known intent for spot-checks.

## Cloud services required

| Service | Purpose | Status |
|---|---|---|
| AWS Polly | TTS synthesis of scripted dialogue | New — needs Polly IAM permission added to existing AWS credentials |
| AWS Transcribe | Real ASR pass per session → WER-0 ground-truth transcript | Existing (already used by `audio_pipeline`); needs S3 as transient job input (uploaded, then can be deleted — not permanent storage, dataset itself lives in git) |
| Anthropic / OpenAI API | LLM-as-judge baseline (Phase 3, not dataset-building) | New — provision keys early so Phase 3 isn't blocked later |

## WER injection — `inject_wer.py`

- Input: real WER-0 transcript from AWS Transcribe.
- Applies controlled synthetic corruption (word substitution from a
  near-homophone confusion list, deletion, insertion) at rates calibrated to
  hit target aggregate WER ±2%.
- Outputs `sess_X_wer{10,20,30}.json` per session.
- Result: 4 transcript variants (0/10/20/30% WER) × 40 sessions = 160
  transcript files, feeding Phase 3's ASR-robustness comparison table.

## Annotation tool — `tools/annotation_app/`

Small FastAPI + React page:
- Loads a session's audio + WER-0 transcript, chunked into 5-second windows
  (matches the production `Window Result` unit).
- Annotator plays each chunk, labels `correct / weakly_correct / incorrect`
  against the session's declared `TOPIC`.
- Exports `sess_X_annotatorN.json`: list of `{window_index, start, end, label}`.
- Each session gets 2-3 annotators, drawn only from the other three team
  members (never the session's own scripter).

`compute_kappa.py` aggregates all annotator files, computes Cohen's/Fleiss'
kappa per session and overall, flags any session below 0.7 kappa for
re-annotation or discussion.

## Deliverables & success criteria

- 40 scripted sessions (10 per person × 5 categories × 2 each), each with
  synthesized audio, one real transcript (WER-0) and 3 synthetic WER variants
  (10/20/30%).
- Each session annotated by 2-3 people; overall inter-annotator kappa ≥ 0.7
  (matches Research Gap doc's stated target).
- Dataset feeds directly into Research Gap Phase 3 (baseline head-to-head:
  SBERT / cross-encoder / NeMo TopicControl / LLM-judge / BERTopic) with no
  further data-collection work required.
