# Core Barter System — Issue Register

Reviewed: 2026-09-17. Status: open unless explicitly marked otherwise.

This records the eight findings from the project review, additional issues observed during that review, and concerns that still need targeted testing. The reviewed snapshot includes the existing uncommitted changes. No application fixes were made as part of the review.

The project is a capstone core prototype. Missing production features are distinguished from bugs that affect the current two-person demo. This is not an exhaustive security audit or an evaluation of model accuracy.

## Evidence and priorities

- **Reproduced:** observed using the application code against an isolated in-memory SQLite database, with outbound service calls mocked, or by running the stated tooling command.
- **Code-confirmed:** directly visible in the implementation; not exercised through a complete deployed session.
- **Needs validation:** a plausible failure mechanism found in the code that requires a targeted integration or concurrency test.
- **High:** incorrect verdicts, trust or credit changes, invalid completion, or broken termination.
- **Medium:** data loss, unreliable reporting, configuration failures, or missing safeguards.
- **Low:** documentation or diagnostic shortcomings.

## Original review findings

### ISSUE-001 — Escrow is settled before the final verdict exists

**Priority:** High. **Evidence:** Reproduced.

**Locations:** [confirmation and settlement](apps/backend/app/routes.py), `confirm_session`, around lines 291–315; [results screen](apps/frontend/src/screens/PostSession.jsx), `load`.

When the second participant confirms, `confirm_session` reads the current verdict and immediately calls `apply_settlement`. A missing verdict or a `PENDING` verdict maps to `qa_score = 0`. The frontend generates the final verdict afterward, when the results screen loads. The summary pipeline creates a pending verdict, not a final successful/partial verdict.

**Reproduction:** Create and start a one-minute session, set its start time two minutes in the past, and confirm both users before calling verdict generation. Settlement penalizes the teacher and refunds the learner. Generating the verdict afterward returns `SUCCESSFUL`.

**Impact:** A legitimate successful session can permanently forfeit the teacher's deposit and miss the teaching bonus. A later correct verdict does not repair settlement.

**Fix direction:** Finish analysis and generate the authoritative verdict before settling. Persist finalization progress and make settlement run once.

**Acceptance:** A successful session releases both deposits and awards the intended bonus regardless of which browser opens results first. Repeated finalization cannot change balances again.

### ISSUE-002 — Results viewing repeatedly modifies trust

**Priority:** High. **Evidence:** Reproduced.

**Locations:** [PostSession.jsx](apps/frontend/src/screens/PostSession.jsx), `load`, around lines 43–44; [routes.py](apps/backend/app/routes.py), `update_trust`, around lines 808–841.

Every results load calls `/trust/{barter_id}/update`. The endpoint applies the trust formula again without checking whether that session has already affected trust. Both participants opening their own results naturally repeats the operation; the Retry button can also repeat it.

**Reproduction:** Apply trust updates twice for one successful session. Trust changes from `1.0` to `0.93`, then to `0.909`. The second call also replaces the recorded delta with the latest delta rather than the original session effect.

**Fix direction:** Update trust once inside server-side finalization. Results viewing should read persisted results. Enforce idempotency in the database, including concurrent requests.

**Acceptance:** Repeated or concurrent update requests return the original result without further mutation.

### ISSUE-003 — Topic monitoring does not influence the verdict

**Priority:** High. **Evidence:** Reproduced.

**Location:** [routes.py](apps/backend/app/routes.py), `generate_verdict`, around lines 719–746.

The verdict depends on elapsed duration, confirmation count, and the session's terminated status. It does not use topic correctness, warning history, or engagement results.

**Reproduction:** Submit a summary with ten incorrect windows out of ten, nine warnings, and 100% incorrect content. With sufficient duration and both confirmations, verdict generation returns `SUCCESSFUL`.

**Impact:** The core monitoring system can identify complete topic failure while the verdict declares success. Those monitoring results also do not affect the verdict-derived trust score.

**Fix direction:** Define an explicit, documented quality policy and apply it to finalized evidence. Distinguish insufficient evidence from good evidence; do not invent thresholds without evaluation.

**Acceptance:** Tests cover successful, partly off-topic, wholly off-topic, terminated, and no-evidence sessions under the agreed policy.

### ISSUE-004 — Severe warnings only stop the browser

**Priority:** High. **Evidence:** Code-confirmed.

**Locations:** [warning engine](apps/warning_engine/main.py), `run_warning_decision` and `receive_safety_alert`; [LiveSession.jsx](apps/frontend/src/screens/LiveSession.jsx), warning WebSocket handler around line 98.

Three or more consecutive incorrect windows produce a severe warning, but the warning engine does not set `state["terminated"]` or call backend termination. A safety alert with `hard_block` likewise becomes a severe warning without server-side termination. The browser sets local termination state and calls `halt()`.

**Impact:** The UI says the session is terminated while the backend remains active. The warning engine's terminated guard never takes effect through these paths. The UI hides completion controls, leaving no normal route to final results.

**Fix direction:** Make the server own termination and broadcast the committed state. Treat a hard block as an actual state transition if that remains the intended policy.

**Acceptance:** Severe termination is reflected consistently in backend status, warning state, both clients, pipeline shutdown, verdict, and settlement.

### ISSUE-005 — Manual termination leaves escrow locked

**Priority:** High. **Evidence:** Reproduced for locked escrow; code-confirmed for missing cleanup.

**Locations:** [routes.py](apps/backend/app/routes.py), `terminate_session`; [LiveSession.jsx](apps/frontend/src/screens/LiveSession.jsx), `handleTerminate`.

The termination endpoint sets status and end time only. It does not settle/refund deposits, end downstream analysis, or broadcast termination to the peer. The frontend stops its own streams but does not navigate to results.

**Reproduction:** Create and start a session, terminate it, then fetch its escrow records. Both records remain `locked`.

**Fix direction:** Route manual and automatic termination through a common finalization operation with an explicit termination settlement policy.

**Acceptance:** Both clients learn about termination, credits reach a terminal state, and a final report becomes available without manual API calls.

### ISSUE-006 — Arbitrary users can satisfy participant confirmation

**Priority:** High. **Evidence:** Reproduced.

**Location:** [routes.py](apps/backend/app/routes.py), `confirm_session`, around lines 253–273.

The endpoint accepts a supplied user ID, checks only for an existing confirmation from that ID, and treats any two confirmation rows as both participants confirming. It does not check membership in the contract.

**Reproduction:** Confirm Alice and Bob's session using IDs `99` and `100`. The reviewed SQLite configuration accepts them and the second response reports `both_confirmed: true`.

**Fix direction:** Require confirmation from each distinct contract participant. Enforce membership even before full authentication is added. Ultimately derive the actor from authenticated identity rather than the request body.

**Acceptance:** Unknown users, nonparticipants, duplicate users, and invalid session states cannot complete the session.

### ISSUE-007 — Repeated escrow locking strands deposits

**Priority:** Medium. **Evidence:** Reproduced.

**Locations:** [routes.py](apps/backend/app/routes.py), `lock_escrow_endpoint`; [escrow.py](apps/backend/app/escrow.py), `lock_escrow` and `apply_settlement`; [models.py](apps/backend/app/models.py), `Escrow`.

Each `/escrow/lock` call creates another escrow for each user. There is no unique constraint for session/user escrow ownership. Settlement builds a dictionary keyed by user ID, retaining only one of that user's locked records for the current call.

**Reproduction:** Start a session, call `/escrow/lock` twice, then settle once. Six escrow records exist; two are released and four remain locked.

**Impact:** Credits remain stranded after apparent settlement. Further settlement calls can process remaining deposits and award further bonuses/trust changes.

**Fix direction:** Enforce the intended uniqueness invariant in the database and make locking and settlement idempotent. Plan explicit repair for existing duplicate records.

**Acceptance:** Repeated/concurrent lock calls create only the intended deposits; one settlement resolves all legitimate obligations exactly once.

### ISSUE-008 — End-to-end test collection uses obsolete paths

**Priority:** Medium. **Evidence:** Reproduced.

**Location:** [tests/conftest.py](tests/conftest.py), around lines 30–38 and warning-engine fixture paths.

Fixtures reference `ROOT/backend` and `ROOT/warning_engine`, but the services are under `ROOT/apps/`. Running `pytest tests --collect-only -q` fails with `ModuleNotFoundError: No module named 'app'` in the reviewed environment.

**Fix direction:** Correct all service paths, document/install test dependencies, and add a CI collection check. Review the fixture's outdated PostgreSQL assumptions and datetime monkeypatch after restoring collection.

**Acceptance:** Collection and execution succeed from the documented project-root command without manually adding import paths.

## Additional observed issues

### ISSUE-009 — Completion notification precedes durable finalization

**Priority:** High. **Evidence:** Code-confirmed ordering; race effects need integration validation.

**Locations:** [routes.py](apps/backend/app/routes.py), `confirm_session`, around lines 276–315; [LiveSession.jsx](apps/frontend/src/screens/LiveSession.jsx), `both_confirmed` handler.

The backend broadcasts `both_confirmed` before committing the confirmation transaction, finishing audio analysis, or settling escrow. Both clients immediately leave the live view and start generating verdicts/updating trust.

**Impact:** Clients can read or act on stale/incomplete database state. This is a separate ordering problem from ISSUE-001: a valid finalization calculation still needs a committed completion notification.

**Fix direction:** Use a finalizing state, commit final results, then announce completion. Do not let results-page rendering drive finalization.

### ISSUE-010 — Session state transitions are not consistently validated

**Priority:** High. **Evidence:** Code-confirmed.

**Location:** [routes.py](apps/backend/app/routes.py), `confirm_session`, `terminate_session`, and `generate_verdict`.

Confirmation does not require an active session. It can complete a proposed session or overwrite a terminated session once two confirmations exist. Termination can overwrite an already completed session. Verdict generation can run while a session is active and uses the current time when no end time exists.

**Impact:** Previously terminal outcomes can change, or financial/trust calculations can use an unfinished session.

**Fix direction:** Define allowed transitions and enforce them centrally. Make terminal states immutable except through an explicit correction process.

### ISSUE-011 — Public API clients can supply settlement decisions

**Priority:** High for access outside a controlled demo. **Evidence:** Code-confirmed.

**Locations:** [routes.py](apps/backend/app/routes.py), `/settlement/{barter_id}`, `/escrow/release`, `/warnings/log`, `/window/result`, and summary routes; [main.py](apps/backend/app/main.py); [nginx.conf](apps/frontend/nginx.conf).

The reviewed routes have no authentication or authorization dependencies. Publicly proxied paths include internal evidence-writing and financial endpoints. A caller can supply a QA score directly to settlement or choose an escrow release operation. WebSockets also accept supplied session/user IDs without verifying membership.

**Impact:** A reachable deployment cannot trust its identities, monitoring evidence, or financial actions. This may be an intentional prototype omission, but it is a clear integration boundary for the complete capstone.

**Fix direction:** Separate participant-facing routes from service-only operations, authenticate both, authorize membership, and compute settlement inputs from persisted server evidence.

### ISSUE-012 — Escrow release lacks amount and operation validation

**Priority:** Medium. **Evidence:** Code-confirmed.

**Locations:** [schemas.py](apps/backend/app/schemas.py), `EscrowReleaseRequest`; [escrow.py](apps/backend/app/escrow.py), `release_escrow`, around lines 108–114.

`release_type` is an unrestricted string and `penalty_amount` has no range constraint. A penalty greater than the deposit produces a negative released amount and deducts additional available credits. An unknown release type performs no transition but can still return a successful endpoint response.

**Fix direction:** Use a release-operation enum and enforce `0 <= penalty_amount <= escrow.amount`, plus operation-specific rules, before any mutation.

### ISSUE-013 — Session contract roles and session participants can disagree

**Priority:** Medium. **Evidence:** Code-confirmed.

**Locations:** [routes.py](apps/backend/app/routes.py), `create_session`, `start_session`, `lock_escrow_endpoint`, and `update_trust`; [main.py](apps/backend/app/main.py), `signal_ws`; [PostSession.jsx](apps/frontend/src/screens/PostSession.jsx).

Session creation hardcodes session participants as users 1 and 2 while accepting arbitrary teacher and learner IDs in the contract. Starting locks deposits for contract users, but manual escrow locking and trust updates use session users. Signaling routes to user 1 or 2, and the results screen hardcodes Alice as teacher and Bob as learner.

**Impact:** Expanding beyond the fixed demo pair can charge one pair and update another pair's trust. Even role reversal can produce misleading teacher/learner labels.

**Fix direction:** Either explicitly restrict the prototype contract to the supported roles or consistently derive identities and labels from the contract.

### ISSUE-014 — Basic session inputs allow invalid contracts

**Priority:** Medium. **Evidence:** Code-confirmed.

**Location:** [schemas.py](apps/backend/app/schemas.py), `SessionCreateRequest`.

Duration is any integer; topic and scope have no nonempty constraint; teacher and learner can be equal. No creation-time check ensures those users exist. Zero/negative durations trivially satisfy the duration check once a session starts.

**Fix direction:** Validate positive bounded duration, meaningful topic input, distinct valid participants, and supported role assignments before creating records.

### ISSUE-015 — SQLite foreign keys are declared but not enabled

**Priority:** Medium. **Evidence:** Code-confirmed configuration; orphan confirmations reproduced in ISSUE-006.

**Location:** [database.py](apps/backend/app/database.py).

The engine does not configure SQLite's `PRAGMA foreign_keys=ON` for connections. Declaring SQLAlchemy foreign keys alone does not enable SQLite enforcement. The isolated run accepted confirmations for nonexistent users.

**Fix direction:** Enable enforcement per connection, add explicit API validation, and test nonexistent-user/session writes. Audit existing orphan records before changing enforcement.

### ISSUE-016 — Backend settings fail when loaded from the shared root environment file

**Priority:** Medium for local development. **Evidence:** Reproduced.

**Location:** [config.py](apps/backend/app/config.py), line 11.

Settings load `.env` while accepting only the backend's declared fields. Importing backend routes from the project root failed with `extra_forbidden` validation errors for audio/cloud configuration fields in the shared environment file.

This was observed in local execution, not the Docker environment-injection path. The validation exception also printed rejected configuration values, including credential-valued fields; those values are intentionally not reproduced here.

**Fix direction:** Use service-specific configuration files or deliberately ignore unrelated dotenv fields. Prevent validation diagnostics from exposing secret values. Treat credential values captured in the review's tool output as exposed and rotate the affected keys.

### ISSUE-017 — uv project metadata is in the wrong section

**Priority:** Low. **Evidence:** Reproduced tooling warning.

**Location:** [pyproject.toml](pyproject.toml).

`name`, `version`, and `description` are under `[tool.uv]`, where the installed uv rejects them as unknown configuration fields. Commands continued with warnings in the reviewed environment; this was not a total execution blocker. Python compatibility is also unspecified.

**Fix direction:** Put project metadata under `[project]`, specify supported Python versions, and either configure a valid uv workspace or document that dependencies are managed per service.

### ISSUE-018 — Trust policy differs between completion paths

**Priority:** Medium. **Evidence:** Code-confirmed.

**Locations:** [escrow.py](apps/backend/app/escrow.py), `apply_settlement`; [routes.py](apps/backend/app/routes.py), `confirm_session`, `settle_session`, and `update_trust`.

Settlement calculates role-specific trust deltas. The explicit settlement endpoint applies them, while confirmation calls the helper without applying them. The results screen instead calls a different formula that updates both users equally and assumes a fixed satisfaction rating of 4/5.

**Impact:** The same outcome can affect trust differently depending on which endpoint is used. A learner may receive a negative trust change even where the settlement policy specifies no learner penalty.

**Fix direction:** Choose one documented policy and apply it through a single finalization path, recording the applied deltas once.

### ISSUE-019 — No analyzed windows can be reported as 100% on-topic

**Priority:** Medium. **Evidence:** Code-confirmed.

**Locations:** [warning engine](apps/warning_engine/main.py), `end_session`; [routes.py](apps/backend/app/routes.py), `receive_drift_summary`.

With zero windows, the warning engine reports `percent_incorrect = 0`. The backend converts this into `on_topic_percentage = 100 - percent_incorrect`, yielding 100% despite no analyzed evidence.

**Fix direction:** Represent unavailable evidence explicitly and report coverage alongside quality. Do not interpret absence of incorrect windows as demonstrated accuracy.

### ISSUE-020 — Report duration is never supplied by the verdict endpoint

**Priority:** Low. **Evidence:** Code-confirmed.

**Locations:** [PostSession.jsx](apps/frontend/src/screens/PostSession.jsx), Session Summary duration; [routes.py](apps/backend/app/routes.py), `get_verdict`.

The UI reads `verdict.actual_duration_seconds`, but the endpoint never returns that field, so the duration displays a dash.

**Fix direction:** Include the finalized duration in the response and test the response/display contract.

### ISSUE-021 — Architecture documentation disagrees with the implementation

**Priority:** Low. **Evidence:** Code-confirmed.

**Locations:** [ARCHITECTURE.md](ARCHITECTURE.md), [CLAUDE.md](CLAUDE.md), [docker-compose.yml](docker-compose.yml).

Architecture documentation describes PostgreSQL and a Whisper-centered pipeline, while Compose configures SQLite and defaults STT to AWS. Window-duration descriptions disagree, and service inventories omit the video-engagement service in places.

**Impact:** Setup guidance and capstone explanations may describe a different system from the one being demonstrated. Outdated database assumptions also appear in test fixtures.

**Fix direction:** Document the current default deployment, distinguish optional STT backends, and keep service diagrams and timing descriptions aligned with configuration.

## Concerns requiring targeted validation

### ISSUE-022 — Finalization holds a SQLite write transaction across service callbacks

**Priority:** High if confirmed. **Evidence:** Needs integration validation.

**Locations:** [routes.py](apps/backend/app/routes.py), `confirm_session`; [audio pipeline](apps/audio_pipeline/main.py), `end_session`; [semantic analysis](apps/semantic_analysis/main.py), `end_session`.

Confirmation flushes a database write, then awaits audio shutdown before committing. Audio/semantic/warning shutdown calls back into backend routes that also write to SQLite. Those callbacks can wait for the original transaction's write lock while the original request waits for downstream completion. The backend gives the audio call five seconds and suppresses exceptions.

**Test needed:** Run the full shutdown chain against file-backed SQLite with pending audio. Record transaction timing, callback failures, lock errors, and missing summaries. The in-memory mocked reproductions did not exercise this chain.

**Fix direction:** Commit a finalizing state before network calls; avoid holding database transactions across outbound requests; make callbacks and retries idempotent.

### ISSUE-023 — Multiple shutdown paths can process the same audio buffer

**Priority:** Medium. **Evidence:** Needs concurrency validation.

**Location:** [audio pipeline](apps/audio_pipeline/main.py), `audio_ws` and `end_session`.

Normal threshold processing, WebSocket disconnect, and the end endpoint all access/process the same mutable buffer without an ownership lock. Processing awaits external calls. The end endpoint deletes dictionary entries while the WebSocket retains a local buffer reference; it does not itself close that socket.

**Possible impact:** Duplicate segments, late segments after summaries, or missing-key errors during concurrent cleanup.

**Test needed:** End/disconnect while transcription is paused at an await point and assert exactly-once segment delivery and orderly cleanup.

### ISSUE-024 — Audio segment boundaries may duplicate sound and distort duration

**Priority:** Medium. **Evidence:** Needs media-level validation.

**Location:** [audio pipeline](apps/audio_pipeline/main.py), `audio_ws`, `reset_buffer`, and `process_buffer`.

The first entire MediaRecorder blob is retained as `header_chunk` and prepended to every later segment. That blob can contain audio as well as a WebM header. Disconnect also processes a buffer containing only this retained blob. Duration is measured using wall time and reset after transcription, even though incoming audio can queue during transcription.

**Possible impact:** Repeated opening words, extra final transcription, or inaccurate segment duration and engagement timing.

**Test needed:** Feed a recording with uniquely identifiable phrases through multiple windows under artificial STT latency; compare decoded audio, transcript coverage, and timestamps.

### ISSUE-025 — Service delivery failures can silently discard evidence

**Priority:** Medium. **Evidence:** Code-confirmed failure handling; recovery impact needs integration validation.

**Locations:** [audio pipeline](apps/audio_pipeline/main.py), `post_segment` and `process_buffer`; [warning engine](apps/warning_engine/main.py), `post_to_backend` and `end_session`; [routes.py](apps/backend/app/routes.py), service notifications.

Several HTTP failures are logged/suppressed without retry. Some writes do not check HTTP status. Audio resets its buffer after processing returns even when downstream delivery failed. Warning shutdown deletes local state after its summary-post helper returns, including after that helper catches a failure.

**Test needed:** Inject transient backend failures and verify whether transcripts, windows, and summaries can be recovered. Currently there is no demonstrated durable retry path.

**Fix direction:** Use bounded retry with idempotent event IDs or durable pending-delivery records. Report incomplete analysis explicitly.

### ISSUE-026 — Check-then-write operations lack concurrency guarantees

**Priority:** Medium. **Evidence:** Needs concurrency validation.

**Locations:** [routes.py](apps/backend/app/routes.py), start/confirmation/trust routes; [escrow.py](apps/backend/app/escrow.py); [models.py](apps/backend/app/models.py).

Several operations first read state and then mutate it. Confirmation lacks a session/user uniqueness constraint, escrow lacks one too, and balance updates use read-modify-write objects. Sequential duplicate protection does not establish correctness for concurrent requests.

**Test needed:** Concurrent starts, same-user confirmations, releases, and trust updates against the deployment database. Assert one finalization, consistent balances, and no duplicate rewards. SQLite may manifest contention as request failures rather than every hypothesized duplicate outcome.

**Fix direction:** Combine database constraints, atomic conditional transitions, and transaction handling appropriate to the selected database.

### ISSUE-027 — In-memory monitoring state is not restart-safe

**Priority:** Medium beyond a single uninterrupted demo. **Evidence:** Code-confirmed architecture; recovery behavior untested.

**Locations:** Module-level dictionaries in [audio pipeline](apps/audio_pipeline/main.py), [semantic analysis](apps/semantic_analysis/main.py), and [warning engine](apps/warning_engine/main.py).

Audio buffers, warning counters, semantic windows, and engagement accumulators live in process memory. Restarting a service loses that state. Multiple workers would maintain separate copies. Semantic contract fetching can restore a contract but does not restore the full prior monitoring history.

**Fix direction:** Document the single-process constraint. Before requiring restart recovery or multiple workers, persist recoverable session state or replay durable events.

## Validation performed and limits

- Six isolated backend reproductions confirmed ISSUE-001, ISSUE-002, ISSUE-003, ISSUE-005, ISSUE-006, and ISSUE-007. Outbound service calls were mocked; no cloud transcription was invoked.
- Root end-to-end collection failed as described in ISSUE-008.
- Six warning-engine engagement-fusion tests passed. Five runtime warnings reported an unawaited coroutine from the HTTP mock; inspect mock response construction so passing tests do not hide response-handling problems.
- uv emitted the configuration warning in ISSUE-017.
- A local backend import encountered the dotenv validation failure in ISSUE-016. The isolated reproductions were then run from a temporary directory to avoid reading the shared dotenv file.
- Live microphone capture, real WebRTC peers, cloud STT, model accuracy, full Docker shutdown, and concurrency behavior were not validated.
- Existing uncommitted source changes were preserved. This register does not claim that every issue has been dynamically reproduced or that the project has no other issues.

## Suggested repair order

1. Restore end-to-end test collection and add regression coverage for the reproduced failures.
2. Implement one server-controlled finalization flow: stop/flush analysis, freeze evidence, generate verdict, settle escrow, apply trust once, commit results, then notify clients.
3. Enforce participant membership, valid state transitions, input bounds, database constraints, and repeat-safe financial operations.
4. Define and test the quality/trust policy, including missing evidence and role-specific responsibility.
5. Validate shutdown concurrency, SQLite callbacks, audio boundaries, and delivery recovery.
6. Align configuration, documentation, and report fields; establish access controls before exposing the demo to untrusted clients.
