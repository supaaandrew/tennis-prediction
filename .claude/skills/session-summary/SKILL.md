---
name: session-summary
description: Write session summary to prompts.md and generate spec.md for the next session. Trigger when asked to wrap up, end the session, write summary, or prepare for next session.
---

IMPORTANT: This skill should only be invoked AFTER:
1. All tests are green
2. RUN REVIEW hook has fired and CRITICAL issues fixed
3. Adversarial review (/codex:adversarial-review) is complete
4. All Codex findings are fixed
5. Tests are green again

If any of these steps are incomplete, stop and tell the user
to complete them before running this skill.

Do two things:

PART 1 — Append to prompts.md:
  ## Session [date] — [what was built]
  **Prompt:** [one-line summary of what was asked]
  **Shipped:** [files created, test count before/after]
  **Codex findings:** [HIGH/CRITICAL found and fixed]
  **New locked decisions:** [new DECISIONS.md entries]
  **Next:** [what P-step comes next]

PART 2 — Write spec.md for next session:
Based on CLAUDE.md build status and what was just completed,
write spec.md covering:
  - Full context of what was just built (patterns used,
    naming conventions, edge cases handled, repo interfaces
    confirmed working)
  - What the next session needs to build
  - Relevant DECISIONS.md sections to read
  - Any gotchas or constraints from this session that
    the next session must know

The generated spec.md's wrap-up section MUST include this
strict ordering verbatim:

  Wrap-up order (strict):
  1. pytest green
  2. End message with RUN REVIEW
  3. Fix CRITICAL from review.md
  4. /adversarial-review → Codex → fix findings
  5. pytest green again
  6. Run: decisions-update skill
  7. Run: session-summary skill
  8. git commit
  9. /clear

Write spec.md to project root overwriting existing content.
Show the full spec.md after writing.
Also update CURRENT STATUS in CLAUDE.md to mark this
session complete with correct test count.
