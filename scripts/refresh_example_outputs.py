#!/usr/bin/env python
"""Re-run the commands pasted into a Markdown file and update their output.

``examples/README.md`` claims, in so many words, that every console block in it
is what the command really prints. That claim goes stale silently: a line number
moves, a fact count changes, a tolerance is retuned, and the document keeps
asserting the old numbers with no test to contradict it. This script is the
test. It finds every fenced ``console`` block whose first line is a single
shell prompt, runs that command, and replaces the rest of the block with what
the command actually wrote.

Two conventions make that safe to automate.

*One command per block.* A block whose first line is not ``$ <command>``, or
which has more than one prompt line, is left exactly as it is: those are the
summary blocks that list several commands without pasting any output. The
intent is visible in the document rather than in a marker.

*Standard output only.* loopy writes warnings to standard error (a
``ParameterFinderWarning`` about finding ``n`` from a flat array, for one), and
those are noise that would change with the toolchain rather than with loopty.
They are shown to whoever runs this script and kept out of the document.

Usage, from anywhere::

    uv run python scripts/refresh_example_outputs.py            # rewrite
    uv run python scripts/refresh_example_outputs.py --check    # exit 1 if stale

``--check`` is what a CI job or a pre-release hook would run: it rewrites
nothing and fails if any block is out of date, naming the block.
"""

from __future__ import annotations

import argparse
import shlex
import subprocess
from dataclasses import dataclass
from pathlib import Path

#: The documents whose console blocks are regenerated, relative to the root.
DOCUMENTS = ("examples/README.md",)

#: How long any one demo is allowed to take. Generous: a cold run compiles C.
TIMEOUT = 300

PROMPT = "$ "
FENCE = "```"


@dataclass
class Block:
    """One fenced block: the command it claims to show, and where it sits."""

    command: str
    start: int  # index of the line after the prompt
    stop: int  # index of the closing fence


def repository_root() -> Path:
    """The checkout this script lives in, found without asking git."""
    return Path(__file__).resolve().parent.parent


def blocks_of(lines: list[str]) -> list[Block]:
    """Every refreshable console block in a document, in order.

    A block qualifies when it is fenced with ``console``, its first line is a
    prompt, and it has exactly one. Anything else is a block that was written to
    be read rather than to be regenerated, and is left alone.
    """
    out: list[Block] = []
    index = 0
    while index < len(lines):
        if lines[index].strip() != FENCE + "console":
            index += 1
            continue
        opening = index
        closing = opening + 1
        while closing < len(lines) and lines[closing].strip() != FENCE:
            closing += 1
        if closing >= len(lines):  # an unterminated fence; leave the file alone
            break
        body = lines[opening + 1 : closing]
        prompts = [line for line in body if line.startswith(PROMPT)]
        if len(prompts) == 1 and body and body[0].startswith(PROMPT):
            out.append(
                Block(
                    command=body[0][len(PROMPT) :].strip(),
                    start=opening + 2,
                    stop=closing,
                )
            )
        index = closing + 1
    return out


def run(command: str, root: Path) -> list[str]:
    """Run one command in the checkout and return its standard output lines."""
    text = command.split("#", 1)[0].strip()
    result = subprocess.run(  # noqa: S603 - the commands come from the document
        shlex.split(text),
        cwd=root,
        capture_output=True,
        text=True,
        timeout=TIMEOUT,
        check=False,
    )
    if result.stderr.strip():
        print(f"  (stderr, not pasted) {result.stderr.strip().splitlines()[0]}")
    return [line.rstrip() for line in result.stdout.rstrip("\n").split("\n")]


def refresh(path: Path, root: Path, check: bool) -> bool:
    """Refresh one document. Returns whether it was (or would be) changed."""
    lines = path.read_text(encoding="utf-8").split("\n")
    changed = False
    # Backwards, so that rewriting one block does not move the next one's bounds.
    for block in reversed(blocks_of(lines)):
        print(f"{path.name}: {block.command}")
        captured = run(block.command, root)
        if lines[block.start : block.stop] == captured:
            continue
        changed = True
        if check:
            print(f"  STALE: {block.command}")
            continue
        lines[block.start : block.stop] = captured
    if changed and not check:
        path.write_text("\n".join(lines), encoding="utf-8")
    return changed


def main(argv: list[str] | None = None) -> int:
    """Refresh every document, or report which ones are stale."""
    parser = argparse.ArgumentParser(description=__doc__.split("\n", 1)[0])
    parser.add_argument(
        "--check",
        action="store_true",
        help="rewrite nothing; exit 1 if any block is out of date",
    )
    parser.add_argument(
        "documents",
        nargs="*",
        default=None,
        help=f"documents to refresh (default: {', '.join(DOCUMENTS)})",
    )
    args = parser.parse_args(argv)

    root = repository_root()
    names = args.documents or list(DOCUMENTS)
    stale = [name for name in names if refresh(root / name, root, args.check)]
    if not stale:
        print("every console block is current")
        return 0
    verb = "are out of date" if args.check else "were rewritten"
    print(f"{', '.join(stale)} {verb}")
    return 1 if args.check else 0


if __name__ == "__main__":
    raise SystemExit(main())
