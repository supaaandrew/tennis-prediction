---
name: adversarial-review
description: Generate a tailored Codex adversarial review prompt for the current working tree diff. Trigger when asked for adversarial review, Codex review, pre-commit review, or audit.
---

Read these before generating the prompt:
- git diff HEAD (uncommitted changes)
- DECISIONS.md in full
- spec.md (what this session was asked to build)
- Every file touched in the diff

Generate a tailored Codex adversarial review prompt and write it to
.codex_prompt.txt in the project root (do NOT print it for copy-paste).
Do NOT invoke /codex:adversarial-review yourself.

Fill in SESSION-SPECIFIC CHECKS based on what the diff
actually contains. Build this full prompt:

---
Codex adversarial review. Target: working tree diff.
Read DECISIONS.md in full before reviewing.

Focus on what will break in production under real data,
concurrent writes, and failure conditions.
No style suggestions. Only functional violations.

PROJECT CONTEXT (always include in session-specific checks):

This is a production ATP tennis prediction pipeline.
Key patterns to check in every diff:

- PIT safety: no feature uses data with terminal timestamp
  >= as_of_ts. The fm_no_lookahead trigger is defense-in-depth
  only — primary enforcement is in Python.
- Sackmann adapter: match_date_source='sackmann' on every
  MatchRow, venue_id=None acceptable, qualifying rounds
  silently skipped not dead-lettered.
- Player resolution: player_aliases table only, never
  players.aliases JSONB (H6). Shadow players keep stable
  hashed ID forever (A9).
- Elo snapshots: append-only, PK is (player_id, surface,
  match_id) not as_of_ts. Lookup uses as_of_ts <= target
  ORDER BY as_of_ts DESC LIMIT 1.
- Watermarks: failure-aware completion — incomplete status
  on any repo failure, last successful item preserved.
- Dead letter: raw payload before parsing, never the
  parsed intermediate. Never raises including logger failure.
- Kelly sizing: per-match cap AND total daily cap both apply.
- Noise injection: Modeling Agent training loop only, never
  at feature storage time (H1).
  
SESSION-SPECIFIC CHECKS:
[fill in based on diff — for each new module ask: what are
the failure modes? what assumptions could be wrong? what
happens under load, partial failure, or bad input?]

STANDARD CHECKS:

1. LOCKED DECISIONS
   - Every decision in DECISIONS.md — does the diff violate any?
   - Any hardcoded value that should come from config?
   - Any deviation from patterns in existing code?

2. CONTRACTS AND INTERFACES
   - Method signatures match how callers will use them?
   - Return types correct — Optional vs non-Optional?
   - Protocol implementations satisfy the Protocol?
   - Silent type mismatches?

3. FAILURE HANDLING
   - Every external call (DB, HTTP, filesystem) has error handling?
   - Single item failure can abort entire batch or loop?
   - Errors routed correctly (dead letter, retry, raise)?
   - Never-raise contract holds on append-only methods?

4. STATE CORRECTNESS
   - Any operation leave partial state hard to detect/recover?
   - Watermarks/cursors only advanced after confirmed success?
   - Idempotency guaranteed — same operation twice is safe?

5. SECURITY AND SECRETS
   - Secrets in log output, error messages, or stored payloads?
   - Full URLs with credentials logged?
   - Error messages include request parameters?

6. CONCURRENCY AND RACE CONDITIONS
   - Read-then-write patterns that could race?
   - Audit trail writes atomic with the operation they audit?
   - Two concurrent writers can corrupt shared state?

7. TIMESTAMP AND TIMEZONE CONTRACT
   - All datetimes timezone-aware?
   - UTC enforced at every entry point?
   - Naive datetimes rejected explicitly?

8. DATA INTEGRITY
   - Upserts target correct unique constraint?
   - Upsert silently overwrites fields it should not?
   - Numeric inputs validated before storage?

9. TEST COVERAGE
   - Every failure path has a test?
   - Locked decisions covered by regression tests?
   - Happy-path tests actually testing the right thing?
   - Tests for concurrent/partial failure scenarios?

10. ANYTHING ELSE inconsistent with DECISIONS.md or existing
    codebase patterns. Focus on failure modes under real
    data volume or edge case inputs.

Report every finding:
  SEVERITY | file:line | what is wrong | exact fix required

Summary table:
  CRITICAL: N
  HIGH: N
  MEDIUM: N
  LOW: N
  TOTAL: N
---

After building the prompt, write it to .codex_prompt.txt:

    import pathlib
    pathlib.Path('.codex_prompt.txt').write_text(prompt)

Then output exactly:
"Prompt written to .codex_prompt.txt

Run Codex with:
/codex:adversarial-review \"$(cat .codex_prompt.txt)\""
