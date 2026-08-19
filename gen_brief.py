#!/usr/bin/env python3
"""Generate brief.md with MCP-style list brief from guides in the guides/ directory."""

from pathlib import Path

GUIDES_DIR = Path(__file__).parent / "guides"
BRIEF_OUTPUT = Path(__file__).parent / "brief.md"
BRIEF_MAX_LENGTH = 250


def extract_brief(content: str, max_length: int = BRIEF_MAX_LENGTH) -> str:
    """Extract a brief description from guide content (same logic as MCP server)."""
    lines = content.split("\n")
    brief_parts = []

    # Skip the title (first line if it starts with #)
    start_idx = 1 if lines and lines[0].strip().startswith("#") else 0

    for line in lines[start_idx:]:
        line = line.strip()
        if not line:
            continue
        if line == "---":
            break
        if line.startswith(("#", "```")):
            continue
        brief_parts.append(line)
        if len(" ".join(brief_parts)) >= max_length:
            break

    brief = " ".join(brief_parts)
    if len(brief) > max_length:
        brief = brief[:max_length].rsplit(" ", 1)[0] + "..."

    return brief if brief else "No description available."


def main() -> None:
    if not GUIDES_DIR.exists():
        raise SystemExit(f"Guides directory not found: {GUIDES_DIR}")

    entries = []
    for guide_file in sorted(GUIDES_DIR.glob("*.md")):
        try:
            content = guide_file.read_text(encoding="utf-8")
            brief = extract_brief(content)
            entries.append(f"guides://{guide_file.name} - {brief}")
        except Exception as e:
            entries.append(
                f"guides://{guide_file.name} - Error reading description ({e})"
            )

    BRIEF_OUTPUT.write_text("\n".join(entries) + "\n", encoding="utf-8")
    print(f"Wrote {len(entries)} entries to {BRIEF_OUTPUT}")


if __name__ == "__main__":
    main()
