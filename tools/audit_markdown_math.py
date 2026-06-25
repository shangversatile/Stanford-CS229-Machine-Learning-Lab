"""Audit Markdown for renderer-compatible math.

The script is read-only. It recursively scans Markdown files, ignores fenced
code blocks, prints each issue with its path and line number, and exits with a
nonzero status when findings are present.
"""

from __future__ import annotations

import re
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]

FORBIDDEN_MACROS = {
    "unsupported operatorname macro": re.compile(r"\\operatorname"),
    "malformed arg max notation": re.compile(r"\\arg\s*\\max"),
    "malformed arg min notation": re.compile(r"\\arg\s*\\min"),
    "DeclareMathOperator is forbidden": re.compile(r"\\DeclareMathOperator"),
}

RAW_COMMAND_LINE = re.compile(
    r"^\s*\\(?:(?:frac|nabla|Delta|underset)(?![A-Za-z])|"
    r"(?:begin|end)\s*\{)"
)
FENCE_START = re.compile(r"^\s*(`{3,}|~{3,})")


def _closing_fence(line: str, marker: str, length: int) -> bool:
    """Return whether line closes the active Markdown fence."""
    return re.fullmatch(
        rf"\s*{re.escape(marker)}{{{length},}}\s*", line
    ) is not None


def _has_double_backslash_outside_math(line: str) -> bool:
    """Return whether a LaTeX row break appears outside math delimiters."""
    state: str | None = None
    index = 0

    while index < len(line):
        if state is None:
            if line.startswith("$$", index):
                state = "display"
                index += 2
                continue
            if line.startswith(r"\(", index):
                state = "inline"
                index += 2
                continue
            if line.startswith("\\\\", index):
                return True
        elif state == "display" and line.startswith("$$", index):
            state = None
            index += 2
            continue
        elif state == "inline" and line.startswith(r"\)", index):
            state = None
            index += 2
            continue

        index += 1

    return False


def _display_math_issue(line: str) -> str | None:
    """Return a display-math issue for line, if one exists."""
    delimiter_count = line.count("$$")
    if delimiter_count == 0:
        return None

    stripped = line.strip()
    if stripped == "$$":
        return "standalone double-dollar delimiter line"
    if delimiter_count != 2:
        if delimiter_count == 1 and stripped.endswith(".$$"):
            return "display formula appears to close after starting on an earlier line"
        return "display formula must contain exactly two double-dollar delimiters"
    if not stripped.startswith("$$") or not stripped.endswith("$$"):
        return "display formula must occupy one complete line without prose"
    if len(stripped) <= 4 or not stripped[2:-2].strip():
        return "display formula must contain content between its delimiters"
    return None


def _has_single_dollar_delimiter(line: str) -> bool:
    """Return whether line contains a legacy single-dollar delimiter."""
    index = 0
    while index < len(line):
        if line.startswith("$$", index):
            index += 2
            continue
        if line[index] == "$" and (index == 0 or line[index - 1] != "\\"):
            return True
        index += 1
    return False


def audit_markdown(root: Path) -> list[tuple[Path, int, str, str]]:
    """Return findings as path, line number, reason, and offending line."""
    findings: list[tuple[Path, int, str, str]] = []

    for path in sorted(root.rglob("*.md")):
        lines = path.read_text(encoding="utf-8-sig").splitlines()
        fence_marker: str | None = None
        fence_length = 0

        for line_number, line in enumerate(lines, start=1):
            if fence_marker is not None:
                if _closing_fence(line, fence_marker, fence_length):
                    fence_marker = None
                    fence_length = 0
                continue

            fence_match = FENCE_START.match(line)
            if fence_match:
                fence = fence_match.group(1)
                fence_marker = fence[0]
                fence_length = len(fence)
                continue

            if r"\[" in line or r"\]" in line:
                findings.append(
                    (
                        path,
                        line_number,
                        "bracket-style display delimiter is forbidden",
                        line,
                    )
                )

            for reason, pattern in FORBIDDEN_MACROS.items():
                if pattern.search(line):
                    findings.append((path, line_number, reason, line))

            display_issue = _display_math_issue(line)
            if display_issue is not None:
                findings.append(
                    (
                        path,
                        line_number,
                        display_issue,
                        line,
                    )
                )

            if _has_single_dollar_delimiter(line):
                findings.append(
                    (
                        path,
                        line_number,
                        "inline math must use backslash-parenthesis delimiters",
                        line,
                    )
                )

            if RAW_COMMAND_LINE.match(line):
                findings.append(
                    (
                        path,
                        line_number,
                        "line starts with a raw LaTeX command outside math delimiters",
                        line,
                    )
                )

            if _has_double_backslash_outside_math(line):
                findings.append(
                    (
                        path,
                        line_number,
                        "double backslash appears outside math delimiters",
                        line,
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
