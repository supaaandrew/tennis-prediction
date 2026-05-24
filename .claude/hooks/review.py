"""
Stop hook — fires automatically at end of every Claude Code session.
Reads spec.md + modified files from transcript + DECISIONS.md,
calls Anthropic API, writes review to review.md and review_history.md.
Exits 1 (blocks session) if CRITICAL findings. Exits 0 if clean.
"""

import json
import os
import re
import sys
from datetime import datetime, timezone


# Patterns that ALWAYS mean zero criticals. Checked before any positive
# signal so phrasing like "### CRITICAL\n\nNone found." or a summary table
# row "| CRITICAL | 0 |" doesn't trip a false-positive on the bare word.
_NO_CRITICAL_PATTERNS: tuple[re.Pattern[str], ...] = (
    re.compile(r"\bno critical\b"),
    re.compile(r"\b0 critical\b"),
    re.compile(r"\bzero critical\b"),
    # ## CRITICAL\n\nNone | N/A | — | --- | nothing
    re.compile(r"#{1,6}\s+critical\b[^\n]*\n+\s*(?:none|n/a|—|---|nothing)\b"),
    # | CRITICAL | 0 |  — summary-table row with zero count
    re.compile(r"\|\s*critical\s*\|\s*0\s*\|"),
)

# Patterns that mean a real CRITICAL finding was raised. Only consulted
# when no _NO_CRITICAL_PATTERNS matched.
_HAS_CRITICAL_PATTERNS: tuple[re.Pattern[str], ...] = (
    # | CRITICAL | N |  with N >= 1
    re.compile(r"\|\s*critical\s*\|\s*[1-9]\d*\s*\|"),
    # "N CRITICAL" prose with N >= 1
    re.compile(r"\b[1-9]\d*\s+critical\b"),
    # **CRITICAL** / [CRITICAL] severity tag
    re.compile(r"\*\*critical\b"),
    re.compile(r"\[critical\b"),
)


def _detect_critical(review: str) -> bool:
    """Return True iff the review actually raised a CRITICAL finding.

    Negative signals (explicit zero / "None found" under a heading / table
    row with 0) win first. If none fire, structural positive signals are
    checked. As a conservative fallback when neither side matches but the
    word ``critical`` is present, treat it as a finding — better to block
    a session unnecessarily than to merge with an unrecognised CRITICAL.
    """
    lowered = review.lower()
    if any(pat.search(lowered) for pat in _NO_CRITICAL_PATTERNS):
        return False
    if any(pat.search(lowered) for pat in _HAS_CRITICAL_PATTERNS):
        return True
    return "critical" in lowered


def load_file_safe(path: str, default: str = "") -> str:
    try:
        with open(path, encoding="utf-8") as f:
            return f.read()
    except Exception:
        return default


def _iter_transcript_entries(transcript_path: str):
    """Yield message-like dicts from a transcript.

    Claude Code writes transcripts as JSONL (one JSON object per line);
    older/test fixtures may be a single JSON document with a top-level
    "messages" array. Both are supported, and unparseable lines are skipped.
    """
    # Single-document JSON first (legacy / test fixtures).
    try:
        with open(transcript_path, encoding="utf-8") as f:
            data = json.load(f)
    except Exception:
        data = None
    if isinstance(data, dict) and isinstance(data.get("messages"), list):
        yield from data["messages"]
        return
    if isinstance(data, list):
        yield from data
        return
    if isinstance(data, dict):
        yield data
        return

    # Fall back to JSONL: one JSON object per line.
    try:
        with open(transcript_path, encoding="utf-8") as f:
            for line in f:
                stripped = line.strip()
                if not stripped:
                    continue
                try:
                    yield json.loads(stripped)
                except Exception:
                    continue
    except Exception:
        return


def _content_blocks(entry: object) -> list:
    """Return the content-block list from a transcript entry, handling both
    the nested {"message": {"content": [...]}} shape (real Claude Code
    format) and the flat {"content": [...]} shape (legacy)."""
    if not isinstance(entry, dict):
        return []
    inner = entry.get("message")
    if isinstance(inner, dict) and isinstance(inner.get("content"), list):
        return inner["content"]
    content = entry.get("content")
    return content if isinstance(content, list) else []


def _last_user_text(transcript_path: str) -> str:
    """Return the text of the most recent user-typed message, or "".

    A `role="user"` transcript entry can be either a real user message
    (text content blocks) or a `tool_result` the harness forwarded back to
    the model — only the former counts as "the last thing the user said".
    We scan the whole transcript and remember the last qualifying entry.
    Missing/unreadable transcript → empty string (gate will skip).
    """
    if not transcript_path or not os.path.exists(transcript_path):
        return ""
    last_text = ""
    for entry in _iter_transcript_entries(transcript_path):
        if not isinstance(entry, dict):
            continue
        inner = entry.get("message")
        msg = inner if isinstance(inner, dict) else entry
        if not isinstance(msg, dict) or msg.get("role") != "user":
            continue
        content = msg.get("content")
        # Legacy/string form: the whole content is the user's text.
        if isinstance(content, str):
            last_text = content
            continue
        if not isinstance(content, list):
            continue
        # Block-list form: keep only text blocks. `tool_result` blocks
        # share role=user but are harness-injected, not user-typed.
        texts: list[str] = []
        for block in content:
            if isinstance(block, dict) and block.get("type") == "text":
                value = block.get("text")
                if isinstance(value, str):
                    texts.append(value)
        if texts:
            last_text = "\n".join(texts)
    return last_text


def extract_modified_files(transcript_path: str, cwd: str) -> dict[str, str]:
    """Return ``{path: current_disk_content}`` for files written this session.

    The transcript is parsed only for the *list of paths* touched by
    Write/Edit/MultiEdit tool calls; the actual content is then read from
    disk. This is deliberate — transcript entries can be chunked, redacted,
    or truncated, so the content embedded there is not a reliable mirror of
    what the file ended up as. Reading from disk guarantees the reviewer
    sees the final, complete file content (including any later edits to the
    same path within the session).

    Paths that no longer exist on disk (e.g. the session deleted or moved
    them) are silently skipped — the reviewer still learns the path was
    touched via the absence of stale content from the transcript.
    """
    if not transcript_path or not os.path.exists(transcript_path):
        return {}

    paths: list[str] = []
    seen: set[str] = set()
    write_tools = {"Write", "Edit", "MultiEdit", "write", "edit"}

    for entry in _iter_transcript_entries(transcript_path):
        for block in _content_blocks(entry):
            if not isinstance(block, dict):
                continue
            if block.get("type") != "tool_use":
                continue
            if block.get("name") not in write_tools:
                continue
            inp = block.get("input", {})
            if not isinstance(inp, dict):
                continue
            path = inp.get("file_path") or inp.get("path") or ""
            if not path or path in seen:
                continue
            seen.add(path)
            paths.append(path)

    modified: dict[str, str] = {}
    for path in paths:
        # os.path.join keeps `path` as-is when it is already absolute,
        # so this correctly handles both absolute Write paths and any
        # relative paths that older transcript shapes might carry.
        full_path = os.path.join(cwd, path)
        if not os.path.exists(full_path):
            continue
        try:
            with open(full_path, encoding="utf-8") as fh:
                modified[path] = fh.read()
        except Exception:
            # Binary file, permission error, or transient IO — skip
            # rather than poison the review with a partial read.
            continue

    return modified


def main() -> None:
    # The review text and status markers contain non-ASCII (emoji, em-dashes,
    # accented player names). Force UTF-8 on stdout/stderr so a console with a
    # legacy codepage (e.g. Windows cp1252) can't turn a print() into an
    # unhandled UnicodeEncodeError that crashes the Stop hook.
    for _stream in (sys.stdout, sys.stderr):
        try:
            _stream.reconfigure(encoding="utf-8", errors="replace")  # type: ignore[union-attr]
        except Exception:
            pass

    # Read hook context from stdin
    try:
        hook_data = json.load(sys.stdin)
    except Exception:
        hook_data = {}

    transcript_path = hook_data.get("transcript_path", "")
    cwd = hook_data.get("cwd", os.getcwd())

    # Load context files
    decisions = load_file_safe(
        os.path.join(cwd, "DECISIONS.md"),
        default="DECISIONS.md not found."
    )
    spec = load_file_safe(
        os.path.join(cwd, "spec.md"),
        default="No spec.md found for this session."
    )

    # Opt-in gate, per-turn. The Stop hook fires after every user turn under
    # the same `spec.md`, so gating on spec content would run the API on
    # every exchange. Gate instead on the LAST user-typed message AND require
    # the marker at the very END (after .strip()) so that skill templates,
    # documentation, or audit prompts that merely *mention* "RUN REVIEW" in
    # passing don't accidentally fire a paid API call. The human ends the
    # turn they want reviewed with the literal sentinel "RUN REVIEW"; every
    # other turn exits 0 immediately and leaves review.md untouched (so the
    # prior verdict stays visible, which is the safest default).
    # `spec.md` is still loaded above and fed to the API as ground-truth
    # context when the gate passes — only the *trigger* moved.
    trigger_text = _last_user_text(transcript_path)
    if not trigger_text.strip().endswith("RUN REVIEW"):
        print(
            "AUTO REVIEW — skipped (last user message does not end with "
            "'RUN REVIEW'). End your next turn with the marker on its own "
            "trailing line to enable the API-backed review."
        )
        sys.exit(0)

    claude_md = load_file_safe(
        os.path.join(cwd, "CLAUDE.md"),
        default="No CLAUDE.md found."
    )

    # Extract modified files from transcript. Paths come from the transcript;
    # content is re-read from disk so the reviewer sees the final, complete
    # state of each file — not a potentially truncated transcript chunk.
    modified = extract_modified_files(transcript_path, cwd)
    if modified:
        modified_files_content = "\n\n".join(
            f"### {path}\n```\n{content}\n```"
            for path, content in modified.items()
        )
    else:
        modified_files_content = "No modified files detected in transcript."

    timestamp = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")

    # Load ANTHROPIC_API_KEY from a local .claude/hooks/.env if present (the
    # file is gitignored), so the hook works without the key being exported
    # into the shell. A value in the .env file takes precedence here.
    env_file = os.path.join(os.path.dirname(__file__), ".env")
    if os.path.exists(env_file):
        with open(env_file, encoding="utf-8") as f:
            for line in f:
                if line.startswith("ANTHROPIC_API_KEY="):
                    os.environ["ANTHROPIC_API_KEY"] = line.strip().split("=", 1)[1]
                    break

    # Call Anthropic API
    api_key = os.environ.get("ANTHROPIC_API_KEY", "")
    if not api_key:
        output = (
            f"# Auto Review — {timestamp}\n"
            "Status: ⚠️ SKIPPED — ANTHROPIC_API_KEY not set\n"
            f"Transcript: {transcript_path}\n\n"
            "ERROR: ANTHROPIC_API_KEY not set. Review skipped.\n"
        )
        _write_review(cwd, output, transcript_path)
        print(output)
        sys.exit(0)

    # Import lazily so the no-key path above never requires the SDK, and an
    # absent/broken anthropic install degrades to an ERROR review (exit 0)
    # rather than an unhandled ImportError that crashes the Stop hook.
    review_failed = False
    try:
        import anthropic

        client = anthropic.Anthropic(api_key=api_key)
        response = client.messages.create(
            model="claude-sonnet-4-6",
            max_tokens=8000,
            system=f"""You are an adversarial code reviewer for a
production ATP tennis prediction system.

Ground truth — locked decisions (never violate these):
{decisions}

Project conventions (from CLAUDE.md):
{claude_md}

Your job:
- Check whether the implementation matches the session spec
- Check whether any locked decision in DECISIONS.md is violated
- Flag issues as CRITICAL / HIGH / MEDIUM / LOW only
- Be specific: file name and line number where possible
- No style suggestions — only functional violations
- If everything looks correct, say PASS clearly""",
            messages=[{
                "role": "user",
                "content": f"""Session spec (what was asked this session):
<spec>
{spec}
</spec>

Files modified this session:
<files>
{modified_files_content}
</files>

Does the implementation match the spec?
Does it violate any locked decision?
Report findings by severity."""
            }]
        )
        review = response.content[0].text
    except Exception as exc:
        review = f"ERROR: API call failed — {exc}"
        review_failed = True

    # Determine status AFTER the API call settles. Three outcomes:
    #   - review_failed: the SDK/network/auth blew up — never claim ✅
    #   - CRITICAL    : a genuine finding
    #   - clean       : review returned with no critical flagged
    # Exit code blocks only on a real CRITICAL — infrastructure failures
    # surface in the banner but don't gate the session.
    if review_failed:
        status = "❗ ERROR — review did not complete"
        is_critical = False
    else:
        is_critical = _detect_critical(review)
        status = (
            "❌ CRITICAL FOUND — session blocked"
            if is_critical
            else "✅ No critical issues"
        )

    review_content = f"""# Auto Review — {timestamp}
Status: {status}
Transcript: {transcript_path}

{review}
"""

    _write_review(cwd, review_content, transcript_path)

    # Print so Claude Code sees it inline
    print(f"\n{'='*60}")
    print(f"AUTO REVIEW — {timestamp}")
    print(f"Status: {status}")
    print(f"{'='*60}")
    print(review)
    print(f"{'='*60}\n")

    sys.exit(1 if is_critical else 0)


def _write_review(cwd: str, content: str, transcript_path: str) -> None:
    try:
        with open(os.path.join(cwd, "review.md"), "w", encoding="utf-8") as f:
            f.write(content)
    except Exception as exc:
        print(f"WARNING: could not write review.md — {exc}")

    try:
        with open(os.path.join(cwd, "review_history.md"), "a", encoding="utf-8") as f:
            f.write(content)
            f.write("\n---\n")
    except Exception as exc:
        print(f"WARNING: could not write review_history.md — {exc}")


if __name__ == "__main__":
    main()