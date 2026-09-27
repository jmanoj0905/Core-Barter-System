# AnnoMI source vendoring notes

## LICENCE WARNING — READ BEFORE USING THIS DATA

**The upstream repository (`uccollab/AnnoMI`, revision
`42936645ec3857a9c84ab296a36a3c34b779ef49`) ships no LICENSE file and
conveys no grant.** This was verified directly against the GitHub API and
the repository tree, not inferred:

- The GitHub API reports `license: null` for this repository.
- The repository tree contains exactly three files —
  `AnnoMI-full.csv`, `AnnoMI-simple.csv`, `README.md` — and none of them is
  a licence file, and none of them states licence terms.
- This is **not** "the licence is unclear." It is the absence of any
  licence grant whatsoever. Do not soften this fact in downstream
  documentation.

A **CC0** claim circulates for AnnoMI in some secondary sources (blog
posts, dataset aggregators, etc.). That claim is **third-party hearsay**,
was **not** found anywhere in the upstream repository itself, and must
never be asserted as fact. If it is mentioned at all in any downstream
material, it must be attributed explicitly as an unverified third-party
claim, not stated as this project's understanding of the licence.

Additionally, the transcripts in this dataset are themselves derived from
**third-party YouTube video** — every utterance row carries a `video_url`
column pointing at the source video. Even if the AnnoMI authors had shipped
a clear licence grant for their annotations and CSV structure, that would
not by itself settle rights to the underlying video/transcript content,
which remains third-party material.

**Open action item:** contact the AnnoMI authors (contact details are in
the upstream `README.md`, fetched to `raw/README.md` by
`build_dataset.py` — gitignored, not committed, since it is third-party
text) to ask directly about licensing terms for this corpus. This has not
yet been done as of the writing of this file. Until it is resolved, treat
this corpus as unlicensed third-party data.

**Mitigation used in this repository:** raw AnnoMI bytes (the CSVs and the
upstream README) are never committed. `build_dataset.py` fetches them
on demand into `raw/`, which is gitignored (see `.gitignore` in this
directory). Only this authored `SOURCE_README.md`, the pinned
`source_manifest.json` (URLs, revision, and sha256 digests — no content),
and the loader code are tracked in git.

---

## What this is

`eval_dataset/curated/annomi/` vendors the AnnoMI motivational-interviewing
corpus (real human therapist/client dialogue, transcribed from YouTube
videos of motivational-interviewing sessions) as the source of real human
dialogue for two of the ten "real-data" categories in the 25-session
evaluation corpus: `abstract_on_topic` and `depressed_baseline`. See
`.superpowers/sdd/2026-09-27-analogy-corpus/` for the plan this belongs to.

Task 11 will excerpt ten scripts verbatim from conversations loaded here.
Task 10 enforces verbatim provenance for those excerpts. This task (Task 9)
only vendors the source and provides a loader — it does not select or
excerpt anything.

## Pinned revision

- Repository: `https://github.com/uccollab/AnnoMI`
- Revision: `42936645ec3857a9c84ab296a36a3c34b779ef49` (branch `main`,
  committed 2023-03-14)
- Retrieved on: 2026-09-27

## Files vendored (fetched on demand, never committed)

| local path            | upstream path        | bytes     |
|------------------------|-----------------------|-----------|
| `raw/AnnoMI-full.csv`   | `AnnoMI-full.csv`     | 3,795,371 |
| `raw/AnnoMI-simple.csv` | `AnnoMI-simple.csv`   | 2,386,609 |
| `raw/README.md`         | `README.md`           | 6,637     |

Exact sha256 digests are recorded in `source_manifest.json` and were
computed from the actual downloaded bytes, not assumed.

## How to fetch

```bash
python3 eval_dataset/curated/annomi/build_dataset.py
```

This downloads any missing manifest file into `raw/` and verifies every
file's sha256 and byte count against `source_manifest.json`, raising
`ManifestVerificationError` on any mismatch rather than silently accepting
a corrupted or altered download.

## Loader

```python
from eval_dataset.curated.annomi.build_dataset import load_conversations

conversations = load_conversations()
```

Returns `list[Conversation]`, where `Conversation` has `id`, `topic`, and
`utterances`, and each `Utterance` has `speaker_role` (exactly
`"therapist"` or `"client"`) and `text`.

`AnnoMI-simple.csv` is utterance-level: one row per utterance, with a
`transcript_id` grouping utterances into one conversation, an
`utterance_id` giving order within that conversation, and a per-row
`topic` column. Utterances are grouped by `transcript_id`, ordered by
`utterance_id`, and each conversation's topic is taken verbatim from the
data (`topic` is constant within a transcript in this corpus) — never
invented, since these topic labels are third-party.

## Why real data at all, given the licence gap

The project's headline measurement (false-accusation rate under drift
detection) needs some `abstract_on_topic` / `depressed_baseline` sessions
that are not LLM-authored, to avoid the corpus being entirely synthetic
drift scripts scored against a synthetic-drift detector. Spec §8 accepts
the licence gap knowingly: fetch-don't-redistribute is treated as a
mitigation, not a resolution, and contacting the authors remains a
standing action item (see above). This is a decision made with the facts
in front of it, not an oversight.
