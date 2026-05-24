"""
Stop hook — fires automatically at end of every Claude Code session.
Reads spec.md + modified files from transcript + DECISIONS.md,
calls Anthropic API, writes review to review.md and review_history.md.
Exits 1 (blocks session) if CRITICAL findings. Exits 0 if clean.
"""

import json
import sys
import os
from datetime import datetime, timezone


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


def extract_modified_files(transcript_path: str) -> dict[str, str]:
    """Parse transcript for Write/Edit/MultiEdit tool calls."""
    if not transcript_path or not os.path.exists(transcript_path):
        return {}

    modified: dict[str, str] = {}
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
            content_written = (
                inp.get("content")
                or inp.get("new_string")
                or inp.get("new_content")
                or ""
            )
            if path:
                modified[path] = content_written

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
    claude_md = load_file_safe(
        os.path.join(cwd, "CLAUDE.md"),
        default="No CLAUDE.md found."
    )

    # Extract modified files from transcript
    modified = extract_modified_files(transcript_path)
    if modified:
        modified_files_content = "\n\n".join(
            f"### {path}\n```\n{content[:3000]}\n```"
            for path, content in modified.items()
        )
    else:
        modified_files_content = "No modified files detected in transcript."

    timestamp = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")

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
    try:
        import anthropic

        client = anthropic.Anthropic(api_key=api_key)
        response = client.messages.create(
            model="claude-sonnet-4-6",
            max_tokens=2000,
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

    # Determine exit code. Block only on a genuine CRITICAL finding — guard
    # against false positives like "No critical issues" / "0 critical".
    lowered = review.lower()
    is_critical = "critical" in lowered and not (
        "no critical" in lowered or "0 critical" in lowered
    )

    # Write review output
    status = "❌ CRITICAL FOUND — session blocked" if is_critical else "✅ No critical issues"

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