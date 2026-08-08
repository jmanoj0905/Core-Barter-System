# Design Decisions — Core Barter System (Session Quality & Safety Monitor)

> This document records every major design choice, the alternatives considered, and the rationale for each decision. Written for capstone review preparation and codebase context.
>
> Last updated: 2026-03-18

---

## 1. System Identity & Scope

### Decision: This repo is a proof-of-concept for the Session Quality Monitor + Session Safety Monitor

**What this repo builds:**
- Real-time audio pipeline (STT, semantic analysis, warning engine)
- Session Safety Monitor (toxicity on transcript, NSFW on video)
- Admin monitoring dashboard
- Minimal credit/escrow demonstration
- Backend core that stores verdicts, trust scores, and session data

**What this repo does NOT build:**
- User authentication / registration
- Skill matching algorithm
- Chat/negotiation system
- Full escrow locking flow
- Feedback collection (subjective ratings)

**Rationale:** The Session Quality Monitor is the core technical innovation of the Skill Barter Marketplace capstone. It's the non-negotiable feature that differentiates this platform from existing peer-exchange systems. Building it as a standalone proof-of-concept allows focused development and clear demonstration of the novel contribution.

---

## 2. Architecture: 4 Microservices

### Decision: Keep the 4-service split with Docker Compose

| Service | Port | Responsibility |
|---|---|---|
| Backend Core | 8000 | PostgreSQL, session lifecycle, WebSocket warnings, verdicts, trust, credits |
| Audio Pipeline | 8001 | WebSocket audio ingestion, STT (faster-whisper), transcript segments |
| Semantic Analysis | 8002 | Sentence-BERT embeddings, sliding window, topic classification, engagement scoring |
| Warning Engine | 8003 | Warning escalation logic, drift tracking, safety monitor integration |

**Alternatives considered:**
- Monolith (simpler deployment) — rejected because the microservice split demonstrates the system design from the capstone proposal and maps to the multi-agent architecture described in the research
- 5th service for Safety Monitor — rejected; Safety Monitor runs as a module within the existing pipeline (toxicity checks added to semantic analysis and warning engine)

**Rationale:** The split demonstrates event-driven multi-agent orchestration (a key thesis point). Docker Compose eliminates the operational burden while preserving the architectural story.

---

## 3. Speech-to-Text: `faster-whisper` base + OpenAI API fallback

### Decision: Hybrid STT pipeline

**Primary:** `faster-whisper` with CTranslate2 quantization, `base` model
**Fallback:** OpenAI Whisper API (cloud)

**Trade-off analysis (all options evaluated):**

| Option | Latency (25s chunk) | Accuracy (WER) | Cost | Internet | Demo Risk |
|---|---|---|---|---|---|
| Whisper `tiny` (local) | 1-4s | ~8% | Free | No | Misranscribes technical terms |
| Whisper `base` (local, current) | 5-15s | ~5% | Free | No | **Warnings 10-20s late** |
| **faster-whisper `base` (local)** | **2-5s** | **~5%** | **Free** | **No** | **Best local option** |
| OpenAI Whisper API | 1-3s | ~3% | $0.006/min | Yes | WiFi dependency |
| Deepgram (streaming) | Sub-second | ~4% | Free 200 min/mo | Yes | WiFi dependency |

**Why `faster-whisper`:**
1. **3x faster than stock Whisper** via CTranslate2 int8 quantization — same model weights, optimized inference
2. **Same accuracy** as current implementation (no quality regression)
3. **No internet dependency** — demo works on campus WiFi or offline
4. **Buffer optimization:** With 2-5s transcription time, we buffer 15s of audio instead of 25s. Warnings arrive within **5-8 seconds** of someone going off-topic.
5. **Capstone framing:** "We optimized the inference pipeline using CTranslate2 quantization, reducing STT latency by 3x while maintaining word error rate. Cloud API fallback provides resilience."

**Why OpenAI API as fallback:**
- For the live demo, if we want sub-3s latency, we switch to cloud
- Demonstrates "resilient hybrid pipeline with graceful degradation"
- Cost is negligible (~$0.06 for a 10-minute demo)

**Code impact:** `faster-whisper` has a nearly identical API to `openai-whisper`. The `WhisperModel` class replaces `whisper.load_model()` with minimal changes.

---

## 4. Embeddings: Sentence-BERT (all-MiniLM-L6-v2)

### Decision: Keep Sentence-BERT, do NOT switch to OpenAI embeddings

**Alternatives considered:**
- OpenAI `text-embedding-3-small` — higher quality but adds API latency (200-400ms per call) and cost
- OpenAI `text-embedding-3-large` — best quality but expensive at scale
- Fine-tuned Sentence-BERT — better domain accuracy but requires training data we don't have

**Rationale:**
1. **Zero latency** — model runs locally, embeddings computed in <50ms
2. **No API costs** — budget saved for STT (faster-whisper is free) and scope generation (LLM calls)
3. **Proven sufficient** — the current classification (correct/weakly_correct/incorrect/out_of_scope) at thresholds UPPER=0.55, LOWER=0.35 works for the demo
4. **Capstone framing:** "We use a lightweight transformer model (22M parameters) for real-time semantic similarity, enabling sub-100ms window classification without cloud dependencies"

**Future consideration:** If thresholds prove too coarse during calibration, we can switch to OpenAI embeddings for the topic embedding only (cached once at session start) while keeping Sentence-BERT for per-window analysis.

---

## 5. Sliding Window Configuration

### Decision: ~100-second windows, 15-second audio buffer chunks

| Parameter | Value | Rationale |
|---|---|---|
| Audio buffer threshold | **15 seconds** (changed from 25s) | Faster chunks → faster-whisper processes in 2-5s → warnings in 5-8s total |
| Semantic window threshold | **100 seconds** of accumulated transcript | Enough context for meaningful similarity. Too short = noisy. Too long = slow to detect drift. |
| Filler word removal | Yes (uh, um, like, you know, etc.) | Cleaner embeddings improve similarity accuracy |

**Capstone framing:** "We use a hierarchical buffering strategy: short audio chunks (15s) for low-latency transcription, aggregated into semantic windows (~100s) for robust topic classification. This balances responsiveness with classification accuracy."

---

## 6. Window Classification Scheme

### Decision: 4-class soft classification

| Class | Condition | Meaning |
|---|---|---|
| `correct` | similarity >= 0.55 | Clearly on-topic |
| `weakly_correct` | 0.35 <= similarity < 0.55 | Related but drifting |
| `incorrect` | similarity < 0.35 | Off-topic |
| `out_of_scope` | Disallowed concept detected | Scope violation (embedding similarity + keyword check) |

**Future extension:** Add `tangent` category for brief acceptable diversions (e.g., explaining RAM while teaching Python lists). Not in current scope.

**Scope violation detection:** Currently keyword matching. Will be upgraded to embedding similarity against disallowed concept vectors (cosine distance < threshold = violation). This catches semantic violations that keywords miss (e.g., topic is "basic math" and someone discusses "stock derivatives" — keyword "derivatives" might not be in the list, but embedding similarity to "calculus, advanced math" would catch it).

---

## 7. Warning Escalation Policy

### Decision: No auto-termination. Consecutive count resets on-topic, but total drift tracked.

| Consecutive off-topic windows | Action |
|---|---|
| 1 | Silent (logged, no user notification) |
| 2 | Mild warning (yellow banner, both users see it) |
| 3-4 | Strong warning (orange banner) |
| 5+ | Severe warning (red banner) — **no auto-terminate** |

**Key behaviors:**
- Going back on-topic **resets consecutive count to 0**
- A `total_drift_incidents` counter is maintained for the verdict (never resets)
- `out_of_scope` (scope violation) triggers **immediate strong warning** regardless of count
- Both users see all warnings in real-time via WebSocket

**Alternatives considered:**
- Auto-terminate at 5+ (original code) — rejected because users should have agency to decide
- No reset on going back on-topic — rejected as too punitive; people naturally have brief tangents

**Capstone framing:** "Our escalation policy uses a forgiving-but-recording approach: consecutive off-topic behavior triggers escalating warnings, but returning to topic resets the escalation. However, total drift incidents are permanently tracked for post-session verdict calculation, ensuring accountability without premature session termination."

---

## 8. Teacher vs. Learner Monitoring

### Decision: Asymmetric monitoring based on session contract role

**Teacher** (identified from session contract metadata):
- Full topic relevance monitoring (semantic similarity per window)
- Drift detection and warning escalation
- Speaking quality signals

**Learner:**
- Engagement score only (NOT topic relevance)
- Engagement = combined signal from:
  - Speaking ratio (learner talks 20-40% = healthy, <10% = passive)
  - Question frequency (detected from transcript: "?", "how", "what", "why" patterns)
  - Audio activity (presence of acknowledgment sounds: "uh-huh", "okay", "right")
- Low engagement triggers a softer "engagement alert" (not a topic warning)

**Demo format:** One direction only — A teaches B. This proves the monitoring pipeline works without doubling the complexity of turn-taking.

**Rationale:** Monitoring the learner for topic relevance doesn't make sense — a learner asking "what's a variable?" during a Python session isn't off-topic, even though the sentence alone has low similarity to "Python programming." Engagement tracking is the right signal for the receiving side.

---

## 9. Session Safety Monitor

### Decision: Separate conceptual module from Quality Monitor, both feed into warning engine

**Quality Monitor** (existing pipeline):
- Topic relevance (semantic similarity)
- Engagement balance
- Delivery quality scoring

**Safety Monitor** (new module):
- Transcript toxicity → OpenAI Moderation API
- Video NSFW → nudenet (local)

### 9a. Transcript Toxicity: OpenAI Moderation API

**Why:**
- **Free** — no cost, ever (it's a free endpoint from OpenAI)
- **Fast** — sub-200ms per request
- **Comprehensive** — covers hate, harassment, self-harm, sexual content, violence with sub-categories
- **Returns scores** — threshold at 0.7+ for flagging, 0.9+ for hard-block
- **Zero training needed** — production-grade, works out of the box

**Fallback:** Detoxify (open-source PyTorch model, ~50ms local inference) if API is unavailable.

**Integration point:** After STT produces a transcript segment, it's checked against the Moderation API before/during semantic analysis. Toxic content triggers an immediate safety warning regardless of topic relevance.

**Capstone framing:** "We integrate a production-grade content safety system as our toxicity detection layer within the Session Safety Monitor, achieving comprehensive coverage of hate speech, harassment, and harmful content categories with sub-200ms classification latency."

### 9b. Video NSFW: nudenet (local)

**Why:**
- **Local inference** — no video data sent to external services (privacy-preserving)
- **Lightweight** — ~50ms per frame on CPU
- **Sampling strategy:** 1 frame every 5-10 seconds (0.1-0.2 FPS) — sufficient for NSFW detection, minimal compute
- **Detection categories:** Nudity, exposed body parts, explicit content

**Capstone framing:** "We deploy a local NSFW classifier operating at 0.1 FPS for session safety, processing video frames entirely on-device without transmitting visual data to external services. This privacy-preserving design ensures content safety while respecting user data sovereignty."

---

## 10. Scope Generation via LLM

### Decision: LLM generates allowed/disallowed concepts at session creation, user reviews and locks

**Flow:**
1. User provides topic (e.g., "Python basics for beginners")
2. LLM generates structured output:
   ```json
   {
     "allowed_concepts": ["variables", "data types", "loops", "functions", "strings", "lists", "conditionals"],
     "disallowed_concepts": ["machine learning", "web frameworks", "databases", "DevOps", "stocks", "politics"],
     "scope_description": "Beginner Python programming covering fundamental syntax and basic data structures"
   }
   ```
3. User reviews, can add/remove items
4. Locked for the session — semantic analysis uses these as ground truth

**LLM choice:** Cheapest and fastest option for structured output. Candidates:
- GPT-4o-mini (~$0.15/1M input tokens) — fast, cheap, excellent at structured output
- Any model with JSON mode / function calling support

**Rationale:** Manually defining allowed/disallowed concepts is tedious and error-prone. LLM generation provides a strong default that users can refine. This is also a demo-worthy feature — "the system understands the topic and automatically defines monitoring boundaries."

---

## 11. Admin Monitoring Dashboard

### Decision: Real-time admin view showing both users' data, with full controls

**Display:**
- Live transcripts from both users, logically separated (left/right or tabbed)
- Per-window similarity score overlaid on teacher's transcript
- Learner engagement score (combined metric)
- Warning log with timestamps and severity
- Visual timeline: green (on-topic) / yellow (weak) / red (off-topic) windows
- Current escalation level

**Admin controls:**
- Manual warning: send a custom warning message to both users
- Pause session: temporarily halt monitoring and notify users
- Terminate session: end session early (triggers verdict generation)
- Adjust thresholds: modify UPPER/LOWER similarity thresholds mid-session

**Capstone framing:** "The admin dashboard provides a panopticon view of the barter session, enabling both passive observation and active intervention. This demonstrates the platform's capability for human-in-the-loop oversight alongside automated monitoring."

---

## 12. Credits & Escrow (Minimal)

### Decision: Show credit balance going up/down based on verdict — no full escrow locking flow

**What's implemented:**
- Users have a `credit_balance` field
- After verdict: balance increases (successful) or decreases (failed/dispute)
- Simple display: "Before: 50 credits → After: 65 credits (+15 for successful session)"

**What's NOT implemented:**
- Asymmetric escrow calculation
- Credit locking before session
- Wallet management with locked/available split
- Transaction ledger
- Settlement formulas with bonuses

**Rationale:** The economic loop is important for the thesis but the full escrow system is in the main platform repo. This repo demonstrates the principle: "good session = rewards, bad session = penalties."

---

## 13. Database: PostgreSQL Only

### Decision: PostgreSQL with async SQLAlchemy, no Redis or S3

**Rationale:**
- In-memory state (session buffers, warning counters) is acceptable for a demo with <10 concurrent sessions
- Audio files are temporary (deleted after transcription) — no persistent storage needed
- Redis caching adds operational complexity with no demo benefit
- S3 for recordings is a production concern, not a POC concern

**Known limitation:** If a service crashes, in-memory session state is lost. Acceptable for demo. Production would use Redis for state persistence.

---

## 14. Frontend: Minimal

### Decision: Minimal React app or curl/Postman for API testing

**For the demo:**
- Simple React app with session setup screen and live session screen (showing warnings)
- Admin dashboard is a separate page/window
- No authentication UI — hardcoded demo users

**Rationale:** The backend/ML pipeline is the technical contribution. Frontend polish is not graded. A working demo that shows the pipeline in action is sufficient.

---

## 15. Demo Scenario

### Setup:
- 2 browser tabs (User A: teacher, User B: learner)
- 1 browser tab (Admin dashboard)
- Topic: "Python basics for beginners"
- LLM-generated scope locked before session starts

### Flow:
1. Session created with topic + scope
2. Both users join video/audio session
3. Teacher speaks on-topic → admin sees green windows, high similarity
4. Teacher deliberately goes off-topic → admin sees red windows, warnings appear on both users' screens
5. Teacher returns to topic → warnings de-escalate, admin sees green again
6. Session ends → verdict generated → credit balance updated → trust score adjusted
7. Admin sees full session timeline: green-green-red-red-yellow-green-green

### What impresses the panel:
- **3rd laptop showing admin view** with live transcripts, sliding windows working in real-time
- **Visible cause-and-effect**: speak off-topic → warning appears within 5-8 seconds
- **The pipeline**: mic → STT → embeddings → classification → warning → WebSocket → UI (all visible)

---

## Summary of Technology Choices

| Component | Choice | Alternative Considered | Why This One |
|---|---|---|---|
| STT (primary) | faster-whisper base | openai-whisper, Deepgram | 3x faster than stock, free, offline |
| STT (fallback) | OpenAI Whisper API | Deepgram streaming | Best accuracy, simple integration |
| Embeddings | Sentence-BERT (all-MiniLM-L6-v2) | OpenAI embeddings | Zero latency, free, sufficient quality |
| Toxicity (text) | OpenAI Moderation API | Detoxify | Free, fast, comprehensive, production-grade |
| NSFW (video) | nudenet | OpenAI Vision | Local, privacy-preserving, lightweight |
| Scope generation | GPT-4o-mini | Local LLM, manual input | Cheap, fast, structured output |
| Database | PostgreSQL | +Redis, +S3 | Simplicity for POC |
| Framework | FastAPI (Python) | — | Already implemented |
| Frontend | Minimal React | — | Backend/ML is the contribution |
