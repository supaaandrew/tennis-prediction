---
name: decisions-update
description: Update DECISIONS.md after a session completes. Trigger when asked to update decisions, log what shipped, or record new locked decisions.
---

Update DECISIONS.md in place. Make exactly these changes:

1. Section 5 (File topology): add every new file created
   this session to the correct location in the tree.
   Match existing indentation and comment style exactly.

2. Section 14 (Day-by-day commit summary): add one row:
   | Day X | [commit hash or TBD] | [what shipped, test
   count, Codex findings fixed] |

3. New locked decisions: if any new architectural decisions
   were made this session, add them as new rows in the
   appropriate section continuing the current letter sequence.
   Format: | X# | Decision | Rationale |

4. Update "Next session resumes at..." line at bottom of
   section 14 to reflect what comes next.

Do not change any existing content. Only add new content.
Show a diff of every change made.
