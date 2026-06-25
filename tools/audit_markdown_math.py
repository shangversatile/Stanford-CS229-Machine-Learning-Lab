"""Report Markdown math patterns that may be unsupported by the renderer.

This tool is intentionally read-only. It recursively scans Markdown files,
prints each suspicious match with its path and line number, and exits with a
nonzero status when findings are present.
"""

from __future__ import annotations

import re
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]

SUSPICIOUS_PATTERNS = {
    "operatorname": re.compile(r"\\operatorname(?![A-Za-z])"),
    "DeclareMathOperator": re.compile(r"\\DeclareMathOperator(?![A-Za-z])"),
    "arg max": re.compile(r"\\arg\s*\\max(?![A-Za-z])"),
    "arg min": re.compile(r"\\arg\s*\\min(?![A-Za-z])"),
    "col": re.compile(r"\\col(?![A-Za-z])"),
    "rank": re.compile(r"\\rank(?![A-Za-z])"),
    "Span": re.compile(r"\\Span(?![A-Za-z])"),
    "Tr": re.compile(r"\\Tr(?![A-Za-z])"),
    "trace": re.compile(r"\\trace(?![A-Za-z])"),
    "diag": re.compile(r"\\diag(?![A-Za-z])"),
    "proj": re.compile(r"\\proj(?![A-Za-z])"),
    "argmax": re.compile(r"\\argmax(?![A-Za-z])"),
    "argmin": re.compile(r"\\argmin(?![A-Za-z])"),
    "single-line multi-row matrix": re.compile(
        r"\\begin\{bmatrix\}.*\\end\{bmatrix\}"
    ),
}


def audit_markdown(root: Path) -> list[tuple[Path, int, str, str]]:
    """Return suspicious matches as path, line number, reason, and line."""
    findings: list[tuple[Path, int, str, str]] = []

    for path in sorted(root.rglob("*.md")):
        lines = path.read_text(encoding="utf-8-sig").splitlines()
        in_fence = False
        in_display = False
        display_open_line = 0

        for line_number, line in enumerate(lines, start=1):
            if "$$" in line:
                findings.append(
                    (
                        path,
                        line_number,
                        "double-dollar display delimiter is forbidden",
                        line.strip(),
                    )
                )

            for label, pattern in SUSPICIOUS_PATTERNS.items():
                if pattern.search(line):
                    findings.append((path, line_number, label, line.strip()))

            stripped = line.strip()
            if stripped.startswith("```"):
                in_fence = not in_fence
                continue
            if in_fence:
                continue

            rendered_line = re.sub(r"`[^`]*`", "", line)
            rendered_stripped = rendered_line.strip()

            if r"\[" in rendered_line and rendered_stripped != r"\[":
                findings.append(
                    (
                        path,
                        line_number,
                        r"display opener \[ must be alone on its line",
                        line.strip(),
                    )
                )
            if r"\]" in rendered_line and rendered_stripped != r"\]":
                findings.append(
                    (
                        path,
                        line_number,
                        r"display closer \] must be alone on its line",
                        line.strip(),
                    )
                )

            if rendered_stripped == r"\[":
                if in_display:
                    findings.append(
                        (
                            path,
                            line_number,
                            "nested display opener",
                            line.strip(),
                        )
                    )
                else:
                    in_display = True
                    display_open_line = line_number
            elif rendered_stripped == r"\]":
                if not in_display:
                    findings.append(
                        (
                            path,
                            line_number,
                            "display closer has no matching opener",
                            line.strip(),
                        )
                    )
                else:
                    in_display = False
                    display_open_line = 0

        if in_display:
            findings.append(
                (
                    path,
                    display_open_line,
                    "display opener has no matching closer",
                    r"\[",
                )
            )

    return findings


def main() -> int:
    """Print the audit result and return a shell-friendly status code."""
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