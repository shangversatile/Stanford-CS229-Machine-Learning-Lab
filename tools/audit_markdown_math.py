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
}


def audit_markdown(root: Path) -> list[tuple[Path, int, str, str]]:
    """Return suspicious matches as path, line number, pattern, and line."""
    findings: list[tuple[Path, int, str, str]] = []
    for path in sorted(root.rglob("*.md")):
        text = path.read_text(encoding="utf-8-sig")
        for line_number, line in enumerate(text.splitlines(), start=1):
            for label, pattern in SUSPICIOUS_PATTERNS.items():
                if pattern.search(line):
                    findings.append((path, line_number, label, line.strip()))
    return findings


def main() -> int:
    """Print the audit result and return a shell-friendly status code."""
    findings = audit_markdown(ROOT)
    if not findings:
        print("Markdown math audit passed: no suspicious patterns found.")
        return 0

    print("Markdown math audit found suspicious patterns:")
    for path, line_number, label, line in findings:
        relative_path = path.relative_to(ROOT)
        print(f"- {relative_path}:{line_number}: {label}: {line}")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
