"""Audit Markdown for renderer-compatible single-line display math.

The script is read-only. It recursively scans Markdown files, ignores fenced
code blocks, prints each issue with its path and line number, and exits with a
nonzero status when findings are present.
"""

from __future__ import annotations

import re
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]

UNSUPPORTED_PATTERNS = {
    "unsupported operatorname macro": re.compile(r"\\operatorname(?![A-Za-z])"),
    "malformed arg max notation": re.compile(r"\\arg\s*\\max(?![A-Za-z])"),
    "malformed arg min notation": re.compile(r"\\arg\s*\\min(?![A-Za-z])"),
}


def audit_markdown(root: Path) -> list[tuple[Path, int, str, str]]:
    """Return findings as path, line number, reason, and offending line."""
    findings: list[tuple[Path, int, str, str]] = []

    for path in sorted(root.rglob("*.md")):
        lines = path.read_text(encoding="utf-8-sig").splitlines()
        in_fence = False

        for line_number, line in enumerate(lines, start=1):
            stripped = line.strip()
            if stripped.startswith("```"):
                in_fence = not in_fence
                continue
            if in_fence:
                continue

            rendered_line = re.sub(r"`[^`]*`", "", line)
            rendered_stripped = rendered_line.strip()

            if r"\[" in rendered_line or r"\]" in rendered_line:
                findings.append(
                    (
                        path,
                        line_number,
                        "bracket-style display delimiter is forbidden",
                        line.strip(),
                    )
                )

            for reason, pattern in UNSUPPORTED_PATTERNS.items():
                if pattern.search(rendered_line):
                    findings.append((path, line_number, reason, line.strip()))

            delimiter_count = rendered_line.count("$$")
            if delimiter_count == 0:
                continue

            if rendered_stripped == "$$":
                findings.append(
                    (
                        path,
                        line_number,
                        "standalone double-dollar delimiter line",
                        line.strip(),
                    )
                )

            if delimiter_count % 2 != 0:
                findings.append(
                    (
                        path,
                        line_number,
                        "display formula must open and close on the same line",
                        line.strip(),
                    )
                )
                continue

            if delimiter_count != 2:
                findings.append(
                    (
                        path,
                        line_number,
                        "expected exactly one display formula on the line",
                        line.strip(),
                    )
                )
                continue

            if not re.fullmatch(r"\s*\$\$.+\$\$\s*", rendered_line):
                findings.append(
                    (
                        path,
                        line_number,
                        "display formula must occupy one complete line",
                        line.strip(),
                    )
                )

    return findings


def main() -> int:
    """Print audit results and return a shell-friendly status code."""
    findings = audit_markdown(ROOT)
    if not findings:
        print("Markdown math audit passed.")
        return 0

    print("Markdown math audit found rendering issues:")
    for path, line_number, reason, line in findings:
        relative_path = path.relative_to(ROOT)
        print(f"- {relative_path}:{line_number}: {reason}: {line}")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())