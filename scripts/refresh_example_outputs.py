#!/usr/bin/env python
"""Re-run the commands pasted into a Markdown file and update their output.

``examples/README.md`` claims, in so many words, that every console block in it
is what the command really prints, and ``README.md`` and ``docs/quickstart.md``
show transcripts too. That claim goes stale silently: a line number moves, a
fact count changes, a tolerance is retuned, and the document keeps asserting the
old numbers with no test to contradict it. This script is the test. It finds
every fenced ``console`` block whose first line is a single shell prompt, runs
that command, and replaces the rest of the block with what the command actually
wrote.

Three conventions make that safe to automate.

*One command per block.* A block whose first line is not ``$ <command>``, or
which has more than one prompt line, is left exactly as it is: those are the
summary blocks that list several commands without pasting any output. The
intent is visible in the document rather than in a marker.

*Standard output only.* loopy writes warnings to standard error (a
``ParameterFinderWarning`` about finding ``n`` from a flat array, for one), and
those are noise that would change with the toolchain rather than with loopty.
They are shown to whoever runs this script and kept out of the document.

*An elided block keeps a checked excerpt.* A block with a line that is just
``...`` shows part of what its command prints, such as a few rows of a ledger.
Every other line of it has to be a line of the output, in order, and the block
is current when each one is there verbatim. A refresh finds each line again,
verbatim or else as the first later line of the same shape (the same text once
numbers, column padding and rules of dashes are blurred, which is how a moved
line number or a wider column shows up), and puts the new text in its place. A
line that matches nothing fails the block, which is left as it was: which rows
an excerpt keeps is a decision for whoever edits the document.

A command that exits non-zero is a failure, in both modes. Its output is not
pasted, because the partial output of a broken demo is not what the document
claims the demo prints, and ``--check`` does not report its block current just
because the text still matches: a document whose command no longer runs is out
of date whatever its blocks say. The block is left as it was and the script
exits 1, naming the command.

Usage, from anywhere::

    uv run python scripts/refresh_example_outputs.py            # rewrite
    uv run python scripts/refresh_example_outputs.py --check    # exit 1 if stale

``--check`` is what CI runs: it rewrites nothing and fails if any block is out
of date or any command fails, naming the block.
"""

from __future__ import annotations

import argparse
import re
import shlex
import subprocess
from dataclasses import dataclass
from pathlib import Path

#: The documents whose console blocks are regenerated, relative to the root.
DOCUMENTS = ("examples/README.md", "README.md", "docs/quickstart.md")

#: How long any one demo is allowed to take. Generous: a cold run compiles C.
TIMEOUT = 300

PROMPT = "$ "
FENCE = "```"

#: A line of a console block that stands for output left out.
ELISION = "..."


@dataclass
class Block:
    """One fenced block: the command it claims to show, and where it sits."""

    command: str
    start: int  # index of the line after the prompt
    stop: int  # index of the closing fence
    elided: bool = False  # it shows an excerpt; see excerpt()


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
                    elided=any(line.strip() == ELISION for line in body[1:]),
                )
            )
        index = closing + 1
    return out


def shape(line: str) -> str:
    """A line with what moves between runs blurred: numbers, padding and rules.

    Two lines of the same shape are the same line of a transcript at another
    location, count or column width: ``spmv.py:79`` and ``spmv.py:80``, or a
    ledger's rule of dashes under a wider column.
    """
    text = " ".join(line.split())
    text = re.sub(r"\d+", "#", text)
    return re.sub(r"-{2,}", "-", text)


def excerpt(kept: list[str], output: list[str]) -> tuple[list[str], list[str]]:
    """An elided block's lines brought up to date against its command's output.

    Every line but an elision is looked for after the line before it matched:
    verbatim, or else as the first line of the same :func:`shape`, whose text
    takes its place. Returns the updated lines and the lines that matched
    nothing, which are kept as they were.
    """
    updated: list[str] = []
    missing: list[str] = []
    cursor = 0
    for line in kept:
        if line.strip() == ELISION:
            updated.append(line)
            continue
        found = next(
            (k for k in range(cursor, len(output)) if output[k] == line), None
        )
        if found is None:
            found = next(
                (
                    k
                    for k in range(cursor, len(output))
                    if shape(output[k]) == shape(line)
                ),
                None,
            )
        if found is None:
            missing.append(line)
            updated.append(line)
            continue
        updated.append(output[found])
        cursor = found + 1
    return updated, missing


class CommandFailed(RuntimeError):
    """A documented command exited non-zero, so its output is not a transcript."""

    def __init__(self, command: str, returncode: int, stderr: str) -> None:
        self.command = command
        self.returncode = returncode
        self.stderr = stderr
        super().__init__(f"{command!r} exited with status {returncode}")


def run(command: str, root: Path) -> list[str]:
    """Run one command in the checkout and return its standard output lines.

    Raises :class:`CommandFailed` when the command exits non-zero. The exit
    status is checked here rather than by ``subprocess`` so that the standard
    error of a failure can still be shown to whoever ran the script.
    """
    text = command.split("#", 1)[0].strip()
    result = subprocess.run(  # noqa: S603 - the commands come from the document
        shlex.split(text),
        cwd=root,
        capture_output=True,
        text=True,
        timeout=TIMEOUT,
        check=False,
    )
    if result.returncode != 0:
        raise CommandFailed(command, result.returncode, result.stderr)
    if result.stderr.strip():
        print(f"  (stderr, not pasted) {result.stderr.strip().splitlines()[0]}")
    return [line.rstrip() for line in result.stdout.rstrip("\n").split("\n")]


def refresh(path: Path, root: Path, check: bool) -> tuple[bool, list[str]]:
    """Refresh one document.

    Returns whether it was (or would be) changed, and the commands that failed.
    A failed command's block is left exactly as it was, in both modes.
    """
    lines = path.read_text(encoding="utf-8").split("\n")
    changed = False
    failed: list[str] = []
    label = path.relative_to(root) if path.is_relative_to(root) else path.name
    # Backwards, so that rewriting one block does not move the next one's bounds.
    for block in reversed(blocks_of(lines)):
        print(f"{label}: {block.command}")
        try:
            captured = run(block.command, root)
        except CommandFailed as exc:
            print(f"  FAILED (exit status {exc.returncode}): {block.command}")
            for line in exc.stderr.strip().splitlines()[-5:]:
                print(f"    {line}")
            failed.append(block.command)
            continue
        if block.elided:
            captured, missing = excerpt(lines[block.start : block.stop], captured)
            if missing:
                print(f"  FAILED (lines not in the output): {block.command}")
                for line in missing[:5]:
                    print(f"    {line}")
                failed.append(block.command)
                continue
        if lines[block.start : block.stop] == captured:
            continue
        changed = True
        if check:
            print(f"  STALE: {block.command}")
            continue
        lines[block.start : block.stop] = captured
    if changed and not check:
        path.write_text("\n".join(lines), encoding="utf-8")
    return changed, failed


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
    stale: list[str] = []
    failed: list[str] = []
    for name in names:
        changed, failures = refresh(root / name, root, args.check)
        if changed:
            stale.append(name)
        failed.extend(failures)
    if stale:
        verb = "are out of date" if args.check else "were rewritten"
        print(f"{', '.join(stale)} {verb}")
    if failed:
        # A failure is never current and never a rewrite: the document still
        # shows a command that does not run, in either mode.
        count = f"{len(failed)} command" + ("s" if len(failed) > 1 else "")
        print(f"{count} failed; the blocks were left as they were")
        return 1
    if not stale:
        print("every console block is current")
        return 0
    return 1 if args.check else 0


if __name__ == "__main__":
    raise SystemExit(main())
