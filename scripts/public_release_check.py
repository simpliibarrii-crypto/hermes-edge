from __future__ import annotations

import re
import subprocess
from pathlib import Path

PATTERNS = {
    "private key": re.compile(r"-----BEGIN (?:RSA |EC |OPENSSH )?PRIVATE KEY-----"),
    "OpenAI-style token": re.compile(r"\bsk-(?:proj-|live-|test-)?[A-Za-z0-9_-]{20,}\b"),
    "GitHub token": re.compile(r"\b(?:gh[pousr]_[A-Za-z0-9]{20,}|github_pat_[A-Za-z0-9_]{20,})\b"),
    "Hugging Face token": re.compile(r"\bhf_[A-Za-z0-9]{20,}\b"),
    "AWS access key": re.compile(r"\bAKIA[0-9A-Z]{16}\b"),
    "Slack token": re.compile(r"\bxox[baprs]-[A-Za-z0-9-]{20,}\b"),
    "Bearer token": re.compile(r"(?i)\bbearer\s+[A-Za-z0-9._~-]{24,}\b"),
    "embedded password": re.compile(
        r"(?i)\b(?:password|passwd|pwd)\s*[:=]\s*['\"](?!change-me|example|placeholder|test)[^'\"]{8,}['\"]"
    ),
    "phone-number-like personal data": re.compile(
        r"(?<!\d)(?:\+?1[ .-]?)?\(?[2-9]\d{2}\)?[ .-]?[2-9]\d{2}[ .-]?\d{4}(?!\d)"
    ),
}

TEXT_SUFFIXES = {
    ".c", ".cfg", ".conf", ".css", ".env", ".example", ".go", ".h", ".html",
    ".ini", ".java", ".js", ".json", ".jsx", ".md", ".mjs", ".py", ".rs",
    ".sh", ".toml", ".ts", ".tsx", ".txt", ".yaml", ".yml",
}


def tracked_files() -> list[Path]:
    output = subprocess.check_output(["git", "ls-files", "-z"])
    return [Path(raw.decode()) for raw in output.split(b"\0") if raw]


def is_text_candidate(path: Path) -> bool:
    return path.name in {"Dockerfile", "Makefile", "LICENSE"} or path.suffix.lower() in TEXT_SUFFIXES


def main() -> int:
    findings: list[str] = []
    for path in tracked_files():
        if not path.is_file() or not is_text_candidate(path):
            continue
        try:
            text = path.read_text(encoding="utf-8")
        except UnicodeDecodeError:
            continue
        for line_number, line in enumerate(text.splitlines(), 1):
            for label, pattern in PATTERNS.items():
                if pattern.search(line):
                    findings.append(f"{path}:{line_number}: possible {label}")

    if findings:
        print("Public release check failed:")
        print("\n".join(findings))
        return 1

    print("Public release check passed: no configured secret or personal-data patterns found.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
