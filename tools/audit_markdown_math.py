"""Audit Markdown for GitHub-compatible math rendering.

The script is read-only. It recursively scans Markdown files, distinguishes
ordinary code fences from GitHub ```math fences, prints each issue with its
path and line number, and exits with a nonzero status when findings exist.
"""

from __future__ import annotations

import re
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]

FORBIDDEN_PATTERNS = (
    ("backslash-parenthesis inline delimiter is forbidden", re.compile(r"\\[()]")),
    ("bracket-style display delimiter is forbidden", re.compile(r"\\[\[\]]")),
    ("unsupported operatorname macro", re.compile(r"\\operatorname")),
    ("unsupported DeclareMathOperator macro", re.compile(r"\\DeclareMathOperator")),
    ("malformed arg max notation", re.compile(r"\\arg\s*\\max")),
    ("malformed arg min notation", re.compile(r"\\arg\s*\\min")),
)

MATH_FENCE_FORBIDDEN_PATTERNS = (
    ("double-dollar delimiter is forbidden inside math fence", re.compile(r"\$\$")),
    ("backslash-parenthesis delimiter is forbidden inside math fence", re.compile(r"\\[()]")),
    ("bracket-style delimiter is forbidden inside math fence", re.compile(r"\\[\[\]]")),
    ("unsupported operatorname macro inside math fence", re.compile(r"\\operatorname")),
    ("unsupported DeclareMathOperator macro inside math fence", re.compile(r"\\DeclareMathOperator")),
    ("malformed arg max notation inside math fence", re.compile(r"\\arg\s*\\max")),
    ("malformed arg min notation inside math fence", re.compile(r"\\arg\s*\\min")),
)

RAW_COMMAND_LINE = re.compile(
    r"^\s*\\(?:frac|nabla|Delta|underset|begin|end|hat|left|right|lVert|rVert|"
    r"mathbb|mathrm|sum|prod|log|exp)(?![A-Za-z])"
)
FENCE_START = re.compile(r"^\s*(`{3,}|~{3,})")
INLINE_CODE_SPAN = re.compile(r"`[^`]*`")
DOLLAR_BACKTICK_INLINE_MATH = re.compile(r"\$`([^`]*)`\$")


def _closing_fence(line: str, marker: str, length: int) -> bool:
    """Return whether line closes the active Markdown fence."""
    return re.fullmatch(rf"\s*{re.escape(marker)}{{{length},}}\s*", line) is not None


def _normalize_dollar_backtick_math(line: str) -> str:
    """Convert GitHub $`...`$ inline math to ordinary $...$ for checks."""
    return DOLLAR_BACKTICK_INLINE_MATH.sub(lambda match: f"${match.group(1)}$", line)


def _line_without_inline_code(line: str) -> str:
    """Remove normal inline code spans before checking prose-level math rules."""
    normalized = _normalize_dollar_backtick_math(line)
    return INLINE_CODE_SPAN.sub("", normalized)


def _is_valid_one_line_display(line: str) -> bool:
    """Return whether line is a complete one-line $$...$$ display formula."""
    stripped = line.strip()
    return (
        stripped.startswith("$$")
        and stripped.endswith("$$")
        and stripped.count("$$") == 2
        and bool(stripped[2:-2].strip())
    )


def _display_math_issue(line: str) -> str | None:
    """Return a display-math issue for line, if one exists."""
    checked_line = _line_without_inline_code(line)
    delimiter_count = checked_line.count("$$")
    if delimiter_count == 0:
        return None

    stripped = checked_line.strip()
    if stripped == "$$":
        return "standalone double-dollar delimiter line"
    if delimiter_count == 1:
        return "one-sided double-dollar delimiter"
    if delimiter_count != 2:
        return "display formula must contain exactly two double-dollar delimiters"
    if not stripped.startswith("$$") or not stripped.endswith("$$"):
        return "display formula must occupy one complete line without prose"
    if not stripped[2:-2].strip():
        return "display formula must contain content between its delimiters"
    return None


def _single_dollar_count(line: str) -> int:
    """Count dollar characters that are not part of $$ delimiters."""
    count = 0
    index = 0
    while index < len(line):
        if line[index] != "$":
            index += 1
            continue
        if index + 1 < len(line) and line[index + 1] == "$":
            index += 2
            continue
        count += 1
        index += 1
    return count


def _inline_dollar_issue(line: str) -> str | None:
    """Return an inline-dollar issue for line, if one exists."""
    checked_line = _line_without_inline_code(line)
    if _is_valid_one_line_display(checked_line):
        return None
    if "$ $" in checked_line:
        return "empty inline math span"
    if _single_dollar_count(checked_line) % 2 == 1:
        return "unbalanced single-dollar inline math delimiters"
    return None


def _opening_fence(line: str) -> tuple[str, int] | None:
    """Return opening fence marker and length, if line starts a Markdown fence."""
    fence_match = FENCE_START.match(line)
    if not fence_match:
        return None
    fence = fence_match.group(1)
    return fence[0], len(fence)


def audit_markdown(root: Path) -> list[tuple[Path, int, str, str]]:
    """Return findings as path, line number, reason, and offending line."""
    findings: list[tuple[Path, int, str, str]] = []

    for path in sorted(root.rglob("*.md")):
        lines = path.read_text(encoding="utf-8-sig").splitlines()
        fence_kind: str | None = None
        fence_marker: str | None = None
        fence_length = 0
        fence_open_line = 0
        fence_open_text = ""

        for line_number, line in enumerate(lines, start=1):
            if fence_kind is not None:
                assert fence_marker is not None
                if _closing_fence(line, fence_marker, fence_length):
                    fence_kind = None
                    fence_marker = None
                    fence_length = 0
                    fence_open_line = 0
                    fence_open_text = ""
                    continue

                if fence_kind == "math":
                    for reason, pattern in MATH_FENCE_FORBIDDEN_PATTERNS:
                        if pattern.search(line):
                            findings.append((path, line_number, reason, line))
                continue

            if line.strip() == "```math":
                fence_kind = "math"
                fence_marker = "`"
                fence_length = 3
                fence_open_line = line_number
                fence_open_text = line
                continue

            opening = _opening_fence(line)
            if opening is not None:
                fence_kind = "code"
                fence_marker, fence_length = opening
                fence_open_line = line_number
                fence_open_text = line
                continue

            checked_line = _line_without_inline_code(line)

            for reason, pattern in FORBIDDEN_PATTERNS:
                if pattern.search(checked_line):
                    findings.append((path, line_number, reason, line))

            display_issue = _display_math_issue(line)
            if display_issue is not None:
                findings.append((path, line_number, display_issue, line))

            inline_issue = _inline_dollar_issue(line)
            if inline_issue is not None:
                findings.append((path, line_number, inline_issue, line))

            if RAW_COMMAND_LINE.match(checked_line):
                findings.append(
                    (
                        path,
                        line_number,
                        "line starts with a raw LaTeX command outside math delimiters",
                        line,
                    )
                )

        if fence_kind == "math":
            findings.append((path, fence_open_line, "math fence is not closed", fence_open_text))

    return findings


def main() -> int:
    """Print audit results and return a shell-friendly status code."""
    findings = audit_markdown(ROOT)
    if not findings:
        print("Markdown math audit passed.")
        return 0

    for path, line_number, reason, line in findings:
        relative_path = path.relative_to(ROOT)
        print(f"{relative_path}:{line_number}: {reason}: {line}")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
