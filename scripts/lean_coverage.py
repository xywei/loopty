#!/usr/bin/env python
"""How much of what isl decides Lean proves: a coverage report over loopty's kernels.

Every example (``examples/*.py``) and every test kernel (``tests/kernels/*.py``)
is imported, and every fact isl decides about it is offered to lanky's Lean
oracle as loopty states it (:mod:`loopty.obligations`): the facts of each
kernel and program, the casts of each schedule the file defines, and the casts
it refuses, which the file's ``rejected*`` functions ask for (and, in a file
with none, its ``main``). The report counts, file by file and kind by kind,

- the facts Lean proved, from loopty's own description of the kernel;
- the facts isl decided and Lean did not prove, each with why: a fact loopty
  states for isl only, a part ``omega`` did not close, or one that ran out of
  time;
- the refuted facts whose negation Lean proved at isl's counterexample, and
  any it could not, which would be a disagreement between the two (a refuted
  fact loopty states for isl only is counted apart);

and the time Lean took per part (a dependence class of a cast, a place of an
access), from an empty proof cache, so that the times are Lean's and not the
cache's. Facts that are not isl's to decide (by type, tested, assumed) are
counted apart.

Run it where Lean is installed, from the repository root::

    uv run python scripts/lean_coverage.py                  # Markdown on stdout
    uv run python scripts/lean_coverage.py --json OUT.json  # and every fact

The Markdown is what ``docs/lean-checks.md`` shows under "Coverage".
"""

from __future__ import annotations

import argparse
import contextlib
import importlib.util
import io
import json
import statistics
import sys
import tempfile
import time
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parent.parent


def _load(path: Path) -> Any:
    name = f"coverage_{path.parent.name}_{path.stem}"
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    sys.path.insert(0, str(path.parent))
    try:
        spec.loader.exec_module(module)
    finally:
        sys.path.remove(str(path.parent))
    return module


def _refusals(module: Any) -> list[Any]:
    """The facts of the casts a file refuses, caught as they are raised."""
    from loopty import schedule as schedule_module

    caught: list[Any] = []
    original = schedule_module.IllegalCast.__init__

    def init(self: Any, message: str, witness: Any = None, fact: Any = None) -> None:
        original(self, message, witness, fact)
        if fact is not None:
            caught.append(fact)

    asking = [
        getattr(module, name)
        for name in sorted(vars(module))
        if name.startswith("rejected") and callable(getattr(module, name))
    ]
    if not asking and callable(getattr(module, "main", None)):
        asking = [module.main]
    schedule_module.IllegalCast.__init__ = init
    try:
        for ask in asking:
            with contextlib.redirect_stdout(io.StringIO()):
                try:
                    ask()
                except BaseException:  # noqa: BLE001 - a demo's own ending is not ours
                    pass
    finally:
        schedule_module.IllegalCast.__init__ = original
    return caught


def _facts(path: Path) -> list[Any]:
    """Every fact of a file: its kernels', programs', schedules', and refused casts."""
    from loopty.cli import collect
    from loopty.kernel import Kernel, Program

    module = _load(path)
    facts: list[Any] = []
    for value in list(vars(module).values()):
        if isinstance(value, Kernel | Program):
            facts.extend(value.facts())
    schedules, _ = collect(module, [])
    for schedule in schedules:
        facts.extend(schedule.facts())
    facts.extend(_refusals(module))
    seen: dict[str, Any] = {}
    for fact in facts:
        seen.setdefault(fact.id, fact)
    return list(seen.values())


def _isls(fact: Any) -> bool:
    """Whether isl decides the fact: its term is a question, or a cast isl answered."""
    from loopty.oracle import IslQuestion

    return isinstance(fact.term, IslQuestion) or (
        callable(getattr(fact.term, "lean_obligation", None))
        and fact.decided_by in ("isl", None)
    )


def check(paths: list[Path]) -> list[dict[str, Any]]:
    """Every isl fact of ``paths``, offered to Lean; one record each."""
    from lanky.check import establish
    from lanky.ledger import Status
    from lanky.oracles.lean import LeanOracle, ProofCache
    from lanky.plugins import registry

    from loopty.oracle import IslOracle

    lean = LeanOracle()
    available, reason = lean.availability()
    if not available:
        raise SystemExit(f"no Lean here: {reason}")
    scratch = tempfile.mkdtemp(prefix="lean-coverage-")
    lean.proofs = ProofCache(scratch)
    registry.oracles = [lean, IslOracle()]
    records: list[dict[str, Any]] = []
    for path in paths:
        where = str(path.relative_to(ROOT))
        for fact in _facts(path):
            if not _isls(fact):
                records.append(
                    {"file": where, "kind": fact.kind, "statement": fact.statement,
                     "outcome": "not isl's", "by": fact.decided_by or fact.status.value}
                )
                continue
            start = time.monotonic()
            if fact.status is Status.ASSUMED:
                result = establish(fact)
            else:
                result = lean.establish(fact)
            seconds = time.monotonic() - start
            provenance = result.provenance
            parts = provenance.get("lean_parts") or []
            refuted = result.status is Status.REFUTED or fact.status is Status.REFUTED
            if result.status is Status.PROVED:
                outcome, why = "proved", ""
            elif refuted and provenance.get("lean_declined") and not parts:
                # isl's alone, its counterexample too: nothing Lean disagrees with
                outcome = "refuted, not stated"
                why = provenance["lean_declined"]
            elif refuted:
                said = provenance.get("lean_counterexample", "")
                outcome = (
                    "refuted, negation proved"
                    if said.startswith("Lean proved the claim false")
                    else "refuted, negation not proved"
                )
                why = said
            else:
                outcome = "decided, not proved"
                why = (
                    provenance.get("lean_reason")
                    or provenance.get("lean_declined")
                    or ""
                )
            records.append(
                {
                    "file": where,
                    "owner": fact.owner,
                    "kind": fact.kind,
                    "statement": fact.statement,
                    "outcome": outcome,
                    "why": why,
                    "seconds": round(seconds, 3),
                    "parts": [
                        {
                            "name": part.get("name"),
                            "proved": part.get("proved"),
                            "seconds": part.get("seconds"),
                            "how": part.get("how", ""),
                        }
                        for part in parts
                    ],
                }
            )
    return records


def report(records: list[dict[str, Any]]) -> str:
    """The records as the Markdown of ``docs/lean-checks.md``."""
    lines: list[str] = []
    isl = [r for r in records if r["outcome"] != "not isl's"]
    counts = Counter(r["outcome"] for r in isl)
    proved = counts["proved"]
    lines.append(
        f"Of {len(isl)} facts isl decides in {len({r['file'] for r in records})} "
        f"files, Lean proved {proved}; {counts['decided, not proved']} stay "
        f"decided; {counts['refuted, negation proved']} refuted facts have "
        "their negation proved at isl's counterexample"
        + (
            f", and {counts['refuted, negation not proved']} do not"
            if counts["refuted, negation not proved"]
            else ""
        )
        + (
            f"; {counts['refuted, not stated']} refuted facts are stated for isl "
            "only"
            if counts["refuted, not stated"]
            else ""
        )
        + f". {len(records) - len(isl)} more facts are not isl's to decide "
        "(by type, tested, assumed)."
    )
    lines.append("")
    lines.append(
        "| file | isl facts | proved | decided, not proved "
        "| refuted, negation proved | Lean time |"
    )
    lines.append("|---|---:|---:|---:|---:|---:|")
    by_file: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for record in isl:
        by_file[record["file"]].append(record)
    for file in sorted(by_file):
        rows = by_file[file]
        c = Counter(r["outcome"] for r in rows)
        total = sum(r["seconds"] for r in rows)
        lines.append(
            f"| `{file}` | {len(rows)} | {c['proved']} | {c['decided, not proved']} | "
            f"{c['refuted, negation proved']} | {total:.1f} s |"
        )
    lines.append("")
    lines.append("By kind:")
    lines.append("")
    lines.append(
        "| kind | isl facts | proved | decided, not proved "
        "| refuted, negation proved | parts | median s per part | slowest part |"
    )
    lines.append("|---|---:|---:|---:|---:|---:|---:|---:|")
    by_kind: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for record in isl:
        by_kind[record["kind"]].append(record)
    for kind in sorted(by_kind):
        rows = by_kind[kind]
        c = Counter(r["outcome"] for r in rows)
        times = [
            part["seconds"]
            for r in rows
            for part in r.get("parts", ())
            if part.get("seconds") is not None
        ]
        median = f"{statistics.median(times):.3f}" if times else "-"
        slowest = f"{max(times):.2f} s" if times else "-"
        lines.append(
            f"| {kind} | {len(rows)} | {c['proved']} | {c['decided, not proved']} | "
            f"{c['refuted, negation proved']} | {len(times)} | {median} | {slowest} |"
        )
    settled = ("proved", "refuted, negation proved")
    missed = [r for r in isl if r["outcome"] not in settled]
    lines.append("")
    if missed:
        lines.append("Every fact Lean did not prove, and why:")
        lines.append("")
        for record in missed:
            why = " ".join(str(record["why"]).split())
            lines.append(
                f"- `{record['file']}`, {record['owner']}: "
                f"{record['statement']}. {why}"
            )
    else:
        lines.append("Lean proved every fact isl decides.")
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--json", metavar="OUT", help="also write every record as JSON")
    args = parser.parse_args(argv)
    paths = [
        *sorted((ROOT / "examples").glob("*.py")),
        *sorted((ROOT / "tests" / "kernels").glob("*.py")),
    ]
    records = check(paths)
    print(report(records))
    if args.json:
        Path(args.json).write_text(json.dumps(records, indent=1), encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
