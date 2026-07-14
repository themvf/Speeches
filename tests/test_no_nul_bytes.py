"""Guard against stray NUL (0x00) bytes in source files.

Two real incidents motivated this (see CLAUDE.md): a NUL embedded in a
template literal in neon.ts's prepareMentionBatch, and a pair of NULs in
intraday-attention.ts (SEC-23) where the ticker-author dedup key's separator
and its matching split() argument were both written as 0x00 instead of a
space. tsc/eslint/next build all tolerate NULs silently, the code can even
work by accident when the corruption is self-consistent, and the byte makes
files look binary to other tooling - so CI is the right place to catch it.
"""

from __future__ import annotations

from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent

SOURCE_EXTENSIONS = {".py", ".ts", ".tsx", ".js", ".jsx", ".json", ".yml", ".yaml", ".md", ".css"}
SKIP_DIRS = {"node_modules", ".next", ".git", "__pycache__", ".venv", "venv", ".claude"}


def _source_files():
    for path in REPO_ROOT.rglob("*"):
        if not path.is_file() or path.suffix not in SOURCE_EXTENSIONS:
            continue
        if any(part in SKIP_DIRS for part in path.parts):
            continue
        yield path


def test_no_nul_bytes_in_source_files():
    offenders = []
    for path in _source_files():
        data = path.read_bytes()
        if b"\x00" in data:
            line_numbers = [
                i for i, line in enumerate(data.split(b"\n"), 1) if b"\x00" in line
            ]
            offenders.append(f"{path.relative_to(REPO_ROOT)} (lines {line_numbers[:5]})")
    assert not offenders, (
        "NUL (0x00) bytes found in source files - almost certainly write-path "
        "corruption where a space was intended:\n" + "\n".join(offenders)
    )
