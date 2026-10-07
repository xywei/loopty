"""The ``loopty`` command.

Two verbs. ``loopty run FILE`` imports the file, lowers its kernels through
loopy, runs them on the requested target, and compares against the native numpy
run. ``loopty check FILE`` is an alias of ``lanky check FILE``: the checking verb
belongs to the host, and loopty reaches it through the plugin registry rather
than reimplementing it, so that one command checks theorems and kernels together.

``RunVerb`` is the same verb registered under ``lanky.verbs``, which is how
``lanky run`` and ``loopty run`` stay one implementation.

Where the inputs come from
--------------------------

A kernel takes its outputs as parameters and its sizes from its arrays, so there
is nothing for ``run`` to invent. The file says what to run on, in one of two
ways, tried in this order:

1. ``Schedule.example(x=..., y=...)`` records the arrays with the schedule::

       sched = Schedule(spmv).tag(r="g.0").example(off=off, col=col, val=val,
                                                   x=x, y=y)

2. a module-level ``example_inputs()`` returning a dictionary. Either a flat
   ``{"x": ..., "y": ...}``, used for every object in the file, or one keyed by
   kernel name, ``{"spmv": {"x": ...}}``, when the file has several.

Every schedule in the file is run, and every kernel that no schedule mentions is
run through the identity schedule, so that a file of plain kernels is still
something ``loopty run`` can do something with. Each run is compared against the
kernel's own Python body on the same inputs, and the comparison lands in the
ledger as a fact with the tolerance its types state.

A program is run the same way. Its term is its kernels' statements in call
order (:mod:`loopty.compose`), which lowers into one kernel, and that kernel is
compared with the program's own body run natively, on the inputs the file
offers under the program's name. A program whose term cannot be built is named
with the reason, as a kernel that cannot be traced is, and counts as failed.

What ``--target`` means
-----------------------

It retargets, rather than filters. A schedule is written against a target, and
running one built for ``"c"`` on the C target while the user asked for
``opencl`` would report a device run that never touched a device. So each
schedule is rebuilt for the requested target through
:meth:`~loopty.schedule.Schedule.retarget`, which replays every step and checks
every cast again, and any schedule that cannot be rebuilt is named, with the
reason, and makes the command exit non-zero. Without the flag, each schedule
keeps the target it was written for and an unscheduled kernel gets ``"c"``.

A schedule the checker accepts but the target cannot generate code for is
reported before anything is compiled, from the ``buildable`` fact the schedule
carries; see :mod:`loopty.schedule`.

What a refutation prints
------------------------

The ledger is printed after the runs, and every ``REFUTED`` fact in it is then
repeated under the table the way ``lanky check`` repeats it, with lanky's own
:func:`lanky.cli.refutation_lines` underneath: the counterexample when there is
one, the fact's ``reason`` (the limit a ``buildable`` fact hits, or the outputs
a compiled run disagreed on), or a line saying nothing was recorded. The
command then exits 1, as it does when a kernel cannot be scheduled, a schedule
cannot be retargeted, a run raises, or two kernels claim one fact id (below).

A run that raises, whatever it raises, is reported by the exception's type and
message, the kernel counts as failed, and the file's other kernels are still
run; so is a file's ``example_inputs()`` that raises, and code generation that
fails under ``--emit-code``. A body that tracing refuses natively (a ``when``
guard whose native value is an integer) is a refuted agreement fact instead,
with the refusal as its reason; see
:meth:`loopty.executor.LoopyExecutor.differential`.

Every schedule keeps its own facts in the ledger, however many schedules of one
kernel the file has, and however many kernels of one name it schedules (one
defined there and one imported, say), because a fact's id names the kernel's
definition and the schedule it is about (see
:meth:`loopty.schedule.Schedule.fact_id`). A cast that rests on the ``layout``
fact of a kernel that rewrites its ragged layout
(:func:`loopty.typing.layout_facts`) has that fact beside it in the ledger, as
``lanky check`` lists it among the kernel's, so that the assumption it is
decided under is a row of the table and not only an id.

Two kernels one definition makes
--------------------------------

A function that decorates a nested definition each time it is called makes
kernels that share that definition, and so every fact id: two schedules of
them with the same steps make two claims of each id, and a ledger holds one
fact per id. ``lanky check`` refuses such claims (lanky's #52), and so does
this command: the first kernel's fact is kept, the other's claim is recorded
on it as ``duplicate_claims`` in its provenance, as ``lanky check`` records
it, and a ``DUPLICATE`` block under the table names the kernel, each id and
the claims of it, in the table and not, and the command exits 1. The command
has decided and run every claim by then, so a claim not in the table that was
refuted is said to be, with what explains it, as the ``REFUTED`` block would
explain it; the ``--json`` ledger keeps the same under ``refused_claims``,
each claim with its status and the counterexample, witness or reason it has
(#99). Each kernel needs an id of its own: a definition of its own, or a
``__qualname__`` of its own given to the function before it is decorated; a
term scheduled with no kernel behind it, or any object with no definition to
name, is named by its name, and needs a name of its own. One kernel scheduled
several times is not refused: two of its schedules that share their first
steps share the facts about them, and a schedule run twice keeps both
agreement facts, the second under its id with ``#2`` after it.
"""

from __future__ import annotations

import argparse
import dataclasses
from pathlib import Path
from typing import Any

from loopty import __version__

__all__ = [
    "RunVerb",
    "build_parser",
    "check",
    "collect",
    "example_inputs",
    "main",
    "select_inputs",
]

REPO_URL = "https://github.com/xywei/loopty"

#: The name of the module-level function a file may define to supply inputs.
EXAMPLE_FUNCTION = "example_inputs"


def collect(module: Any, new_objects: list) -> tuple[list, list]:
    """The schedules and kernels of a checked module, in a stable order.

    Both places are searched: the registry, which the decorators filled in
    import order, and the module namespace, where a schedule that was never
    decorated still lives. Duplicates are removed by identity, not equality,
    because a term's ``==`` builds a proposition rather than answering.

    Whether an object has a ``term`` is asked without reading it:
    ``Kernel.term`` traces the body on first use, and a body that cannot be
    traced has to be reported by name where it is scheduled, not raise out of
    the search for it.
    """
    import inspect
    import types

    from loopty.schedule import Schedule
    from loopty.term import Term

    schedules: list = []
    kernels: list = []
    seen: set[int] = set()
    missing = object()

    def consider(obj: Any) -> None:
        if id(obj) in seen or isinstance(obj, type | types.ModuleType):
            # A class that defines ``trace`` is not a kernel; its instances are.
            # A module that does is numpy, which also has one.
            return
        if isinstance(obj, Schedule):
            seen.add(id(obj))
            schedules.append(obj)
        elif isinstance(obj, Term) or (
            hasattr(obj, "trace")
            and inspect.getattr_static(obj, "term", missing) is not missing
        ):
            seen.add(id(obj))
            kernels.append(obj)

    for obj in new_objects:
        consider(obj)
    for name in getattr(module, "__dict__", {}):
        if name.startswith("_"):
            continue
        consider(getattr(module, name))
    return schedules, kernels


def example_inputs(module: Any, name: str) -> dict[str, Any] | None:
    """The inputs a module offers for one object, or ``None``."""
    factory = getattr(module, EXAMPLE_FUNCTION, None)
    if factory is None:
        return None
    inputs = factory()
    if not isinstance(inputs, dict):
        raise TypeError(f"{EXAMPLE_FUNCTION}() must return a dictionary")
    return select_inputs(inputs, name)


def select_inputs(inputs: dict[str, Any], name: str) -> dict[str, Any] | None:
    """The part of what ``example_inputs()`` returned that is for ``name``.

    A dictionary keyed by kernel name gives each kernel its own entry, and a
    kernel it does not name gets nothing; a flat one is for every kernel. The
    faithfulness fact (:mod:`loopty.faithful`) reads a module's inputs the same
    way.
    """
    chosen = inputs.get(name)
    if isinstance(chosen, dict):
        return dict(chosen)
    if any(isinstance(value, dict) for value in inputs.values()):
        return None
    return dict(inputs)


def _name_of(obj: Any) -> str:
    """A readable name for a kernel, a schedule, or a term.

    A kernel's own ``__name__`` comes first, which is also the name its term
    gets, because reading ``term`` traces the body: the name of a kernel that
    cannot be traced is exactly what the message about it needs.
    """
    name = getattr(obj, "__name__", None)
    if isinstance(name, str):
        return name
    term = getattr(obj, "term", obj)
    return getattr(term, "name", repr(obj))


def _retargeted(schedules: list, target: str) -> tuple[list, int]:
    """Rebuild every schedule for ``target``, keeping the ones that survive.

    Retargeting re-checks each cast and re-asks whether the new target can
    generate the code, so a schedule that only works on one target is refused
    here, by name, rather than quietly run on the target it was written for.
    """
    from loopty.schedule import IllegalCast, UnbuildableSchedule

    out: list = []
    failures = 0
    for schedule in schedules:
        if schedule.target == target:
            out.append(schedule)
            continue
        try:
            out.append(schedule.retarget(target))
        except (
            UnbuildableSchedule,
            IllegalCast,
            TypeError,
            ValueError,
            # `--target opencl` on a machine with no pyopencl: loopy imports it
            # when the target is constructed. Saying so is the whole point of
            # this function, so it is reported like any other refusal.
            ImportError,
        ) as exc:
            print(
                f"cannot retarget {_name_of(schedule)} {schedule!r} to "
                f"{target}: {type(exc).__name__}: {exc}"
            )
            failures += 1
    return out, failures


def _native(schedule: Any) -> Any:
    """The Python body of a scheduled kernel, if there is one to compare with.

    Through any schedule of a schedule (:func:`_kernel_of`): the body of
    ``Schedule(Schedule(k)).split("i", 2)`` is ``k``'s, and a run of it is
    compared with ``k`` as a run of ``Schedule(k).split("i", 2)`` is (#98).
    A term scheduled with no kernel behind it has no body, and its run is
    only executed.
    """
    source = _kernel_of(schedule)
    return source if callable(source) else None


#: The provenance key a fact records the other claims of its id under, as
#: :func:`lanky.check.check_path` records them. Read here rather than through
#: ``Ledger.duplicated``, which lanky added after the version loopty requires.
DUPLICATE_CLAIMS = "duplicate_claims"

#: loopty's own provenance key beside :data:`DUPLICATE_CLAIMS`: each claim
#: recorded there, as ``loopty run`` decided or ran it, with its status and
#: what explains it (see :func:`_claim_record`). ``lanky check`` leaves a
#: later claim unchecked and records its statement alone; ``loopty run`` has
#: decided every claim by the time it records one, and a claim not in the
#: table that was refuted is said to be in the JSON as on the console (#99).
REFUSED_CLAIMS = "refused_claims"

#: The provenance keys that explain a refutation, as
#: :func:`lanky.cli.refutation_lines` reads them.
_EXPLAINING = ("counterexample", "witness", "reason")


def _claim_record(fact: Any) -> dict[str, Any]:
    """One claim not in the table, as :data:`REFUSED_CLAIMS` keeps it.

    Its statement, status and deciding oracle, and each of the keys that
    explain a refutation that it has a value under: what the ``DUPLICATE``
    block prints under a refuted claim, and the reason of an ``assumed`` one.
    """
    record: dict[str, Any] = {
        "statement": fact.statement,
        "status": fact.status.value,
        "decided_by": fact.decided_by,
    }
    for key in _EXPLAINING:
        value = fact.provenance.get(key)
        if value is not None and value != {} and value != "":
            record[key] = value
    return record


def _kernel_of(schedule: Any) -> Any:
    """The kernel, program or term a schedule is of, through any schedule of it.

    ``Schedule(Schedule(k))`` is a schedule of ``k``, as its fact ids say
    (see :func:`loopty.schedule.definition_of`), so it and ``Schedule(k)``
    are claims about one kernel.
    """
    from loopty.schedule import Schedule

    source = schedule.source
    while isinstance(source, Schedule):
        source = source.source
    return source


class _Claims:
    """The ledger of one ``loopty run``, and the kernel each of its ids is about.

    A fact's id names the kernel's definition and the schedule it is about
    (see :meth:`loopty.schedule.Schedule.fact_id`), and a ledger holds one
    fact per id. Two schedules of one kernel that share their first steps
    share the facts about those steps, which are the same claims, and the
    ledger holds them once. Two kernels that one definition makes, as a
    factory does each time it is called, share every id and are two
    kernels: the second one's facts replaced the first's, a refuted one by
    a decided one as readily, and the run exited 0 (#95). So the first
    kernel to claim an id keeps it, and the claim of any other kernel is
    recorded on that fact, by its statement, under ``duplicate_claims`` in
    its provenance, as :func:`lanky.check.check_path` records the claims
    ``lanky check`` refuses (lanky's #52), and with its status and what
    explains it under ``refused_claims`` (:func:`_claim_record`); the run
    then fails (see :func:`_report_duplicates`). The first claim stays in
    the ledger whatever claims the id after it:
    :meth:`lanky.ledger.Ledger.add` would replace it, and the record of the
    other kernel's claim with it.

    A kernel is what a schedule was built from, through any schedule of it
    (:func:`_kernel_of`), told apart from another by identity, as
    :func:`collect` tells them apart. Two kernels with one id are refused
    even when their claims read alike, as ``lanky check`` refuses them: one
    of them is not in the table, and nothing says it is the same claim.
    """

    def __init__(self, ledger: Any) -> None:
        self.ledger = ledger
        #: Each id, with the kernels that claimed it, by identity, the one
        #: whose fact the ledger holds first.
        self._kernels: dict[str, list[int]] = {}
        #: How many runs recorded an agreement fact under each id.
        self._runs: dict[str, int] = {}
        #: Each id other kernels claimed too, with their facts, in the order
        #: ``duplicate_claims`` names them: ``loopty run`` decided them all,
        #: and one that was refuted is said to be (see :func:`_report_duplicates`).
        self.refused: dict[str, list[Any]] = {}
        #: The ids of kernels with no definition to name, which are named by
        #: their name (see :func:`loopty.schedule.definition_of`).
        self.by_name: set[str] = set()

    def add(self, fact: Any, schedule: Any) -> None:
        """Add a fact a schedule makes, unless its id is claimed already.

        By another schedule of the kernel, whose fact is the same claim, or by
        another kernel, whose fact is kept with this claim recorded on it.
        """
        if not self._another(fact, schedule) and fact.id not in self.ledger:
            self.ledger.add(fact)

    def add_run(self, fact: Any, schedule: Any) -> Any:
        """Add the agreement fact of one run of a schedule, and return it.

        Two schedules of one kernel with one id are one schedule, but a file
        may run it twice, on two sets of inputs: each run keeps its fact, the
        second under the id with ``#2`` after it, so that a refuted one is not
        replaced by a later tested one. A run of another kernel whose
        agreement has the id is that kernel's claim, recorded and not kept.
        """
        if self._another(fact, schedule):
            return fact
        count = self._runs[fact.id] = self._runs.get(fact.id, 0) + 1
        if count > 1:
            fact = dataclasses.replace(fact, id=f"{fact.id}#{count}")
        self.ledger.add(fact)
        return fact

    def _another(self, fact: Any, schedule: Any) -> bool:
        """Whether another kernel claimed ``fact.id`` first; records the claim if so.

        A kernel's claim is recorded once per id, however many of its
        schedules make it.
        """
        from loopty.schedule import definition_of

        kernel = _kernel_of(schedule)
        kernels = self._kernels.get(fact.id)
        if kernels is None:
            self._kernels[fact.id] = [id(kernel)]
            if definition_of(schedule, schedule.term)["line"] is None:
                self.by_name.add(fact.id)
            return False
        if kernels[0] == id(kernel):
            return False
        if id(kernel) not in kernels:
            kernels.append(id(kernel))
            kept = self.ledger[fact.id]
            claims = [*kept.provenance.get(DUPLICATE_CLAIMS, ()), fact.statement]
            records = [*kept.provenance.get(REFUSED_CLAIMS, ()), _claim_record(fact)]
            self.ledger.add(
                kept.with_status(
                    kept.status,
                    **{DUPLICATE_CLAIMS: claims, REFUSED_CLAIMS: records},
                )
            )
            self.refused.setdefault(fact.id, []).append(fact)
        return True


def _report_duplicates(claims: _Claims) -> bool:
    """Name each kernel whose id another kernel's claims had; whether there was one.

    The ``DUPLICATE`` block ``lanky check`` prints, one per owner, with each
    id and the claims of it by their statements, and what to do about it.
    The words differ where the two commands do: ``lanky check`` leaves a
    later claim of an id unchecked, and ``loopty run`` has decided every
    schedule's casts by the time the file is imported, and runs every
    schedule, so a claim is said to be in the table or not in it, and one
    not in it that was refuted is said to be, with lanky's own lines of what
    explains it (:func:`lanky.cli.refutation_lines`), since no row and no
    ``REFUTED`` block shows it. The fix named is the one that gives the
    kernel an id of its own: its definition names it, or, when it has none
    (see :func:`loopty.schedule.definition_of`), its name does.
    """
    from lanky.cli import refutation_lines
    from lanky.ledger import Status

    duplicated = [
        fact for fact in claims.ledger if fact.provenance.get(DUPLICATE_CLAIMS)
    ]
    by_owner: dict[str, list[Any]] = {}
    for fact in duplicated:
        by_owner.setdefault(fact.owner, []).append(fact)
    for owner, facts in by_owner.items():
        first = facts[0]
        print()
        if len(facts) == 1:
            count = len(first.provenance[DUPLICATE_CLAIMS]) + 1
            print(
                f"DUPLICATE {owner} at {first.where}: {count} claims have the id "
                f"{first.id}"
            )
            indent = "  "
        else:
            print(
                f"DUPLICATE {owner} at {first.where}: several claims have each of "
                f"the {len(facts)} ids below"
            )
            indent = "    "
        for fact in facts:
            if len(facts) > 1:
                print(f"  {fact.id}")
            print(f"{indent}in the table: {fact.statement}")
            for claim in claims.refused[fact.id]:
                if claim.status is not Status.REFUTED:
                    print(f"{indent}not in the table: {claim.statement}")
                    continue
                print(f"{indent}not in the table, refuted: {claim.statement}")
                for line in refutation_lines(claim):
                    print(f"{indent}  {line}")
        if any(fact.id not in claims.by_name for fact in facts):
            print(
                "  each kernel needs an id of its own: a definition of its own, or "
                "a __qualname__ of its own before it is decorated"
            )
        if any(fact.id in claims.by_name for fact in facts):
            print(
                "  each kernel needs an id of its own: one with no definition to "
                "name, such as a term, is named by its name, and needs a name of "
                "its own"
            )
    return bool(duplicated)


class RunVerb:
    """The ``run`` subcommand, registered with lanky under ``lanky.verbs``."""

    name = "run"
    help = "lower a file's kernels through loopy and run them"

    def add_arguments(self, parser: argparse.ArgumentParser, /) -> None:
        """Declare the verb's options on ``parser``."""
        parser.add_argument("file", help="the Python file to run")
        parser.add_argument(
            "--target",
            default=None,
            choices=("c", "opencl"),
            help=(
                "the loopy target to compile for; retargets every schedule in "
                "the file, re-checking its casts (default: each schedule's own "
                "target, and 'c' for an unscheduled kernel)"
            ),
        )
        parser.add_argument(
            "--emit-code",
            action="store_true",
            help="print the generated code before running",
        )
        parser.add_argument(
            "--json", dest="json_out", metavar="OUT", help="write the ledger as JSON"
        )

    def run(self, args: Any, /) -> int:
        """Run the verb; returns the process exit code."""
        from lanky.check import import_path
        from lanky.cli import refutation_lines
        from lanky.ledger import Ledger, Status
        from lanky.plugins import registry

        from loopty.executor import LoopyExecutor, emit_code
        from loopty.oracle import IslOracle
        from loopty.schedule import Schedule, definition_of
        from loopty.trace import TraceError
        from loopty.typing import layout_facts

        target = getattr(args, "target", None)
        before = len(registry.objects)
        module = import_path(args.file)
        schedules, kernels = collect(module, registry.objects[before:])

        # A kernel some schedule is of, through any schedule of a schedule,
        # is scheduled, and is not run again through the identity (#98).
        scheduled = {id(_kernel_of(schedule)) for schedule in schedules}
        unscheduled = 0
        for kernel in kernels:
            if id(kernel) in scheduled:
                continue
            try:
                schedules.append(Schedule(kernel, target=target or "c"))
            except (TypeError, ValueError, ImportError, TraceError) as exc:
                # ImportError is `--target opencl` with no pyopencl installed,
                # which is a thing to say plainly rather than a traceback. So is
                # a TraceError: a body that cannot be traced has no term to
                # schedule, and its message names the fix.
                print(
                    f"cannot schedule {_name_of(kernel)} for "
                    f"{target or 'c'}: {type(exc).__name__}: {exc}"
                )
                unscheduled += 1

        # ``--target`` means what it says: a schedule the file pinned to another
        # target is rebuilt for this one, which re-checks every cast. Silently
        # running it on the target it was written for would report a device run
        # that never touched a device.
        if target is not None:
            schedules, refused = _retargeted(schedules, target)
        else:
            refused = 0
        failures = unscheduled + refused

        # The target is the executor's option, not a keyword of the run: every
        # keyword of ``run`` is a kernel argument, and an input named
        # ``target`` has to reach the kernel. Each schedule already carries the
        # target it runs on, so this only makes the executor insist on it.
        executor = LoopyExecutor(target=target)
        oracle = IslOracle()
        ledger = Ledger()
        claims = _Claims(ledger)
        for schedule in schedules:
            name = _name_of(schedule)
            print(f"{name}: {schedule!r}")
            facts = schedule.facts()
            for fact in facts:
                claims.add(fact, schedule)
            resting = {identifier for fact in facts for identifier in fact.rests_on}
            for fact in layout_facts(
                schedule.term, **definition_of(schedule, schedule.term)
            ):
                if fact.id in resting:
                    # A layout fact that can be decided is an isl question,
                    # answered here as ``lanky check`` answers it.
                    if oracle.can_establish(fact):
                        fact = oracle.establish(fact) or fact
                    claims.add(fact, schedule)
            ok, reason = schedule.buildable
            if not ok:
                print(f"  not buildable for the {schedule.target} target: {reason}")
                continue
            try:
                if args.emit_code:
                    print(emit_code(schedule))
                inputs = schedule.examples or example_inputs(module, name)
                if inputs is None:
                    print(f"  no example inputs for {name}; add {EXAMPLE_FUNCTION}()")
                    continue
                native = _native(schedule)
                if native is None:
                    executor.run(schedule, **inputs)
                    print(f"  ran {name} on the {schedule.target} target")
                    continue
                fact = executor.differential(native, schedule, inputs)
            except Exception as exc:  # noqa: BLE001 - reported, as every failure is
                # Whatever a run raises is the file's to fix: a cell that is not
                # there, a division by zero, a KeyError of the body's own or of
                # its example_inputs(), or a refusal of the compiled half. It is
                # said by type and message, the kernel counts as failed, and
                # the file's other kernels still run. A native TraceError is
                # not among them: it comes back from ``differential`` as a
                # refuted agreement fact.
                print(f"  {type(exc).__name__}: {exc}")
                failures += 1
                continue
            fact = claims.add_run(fact, schedule)
            outputs = fact.provenance.get("outputs", {})
            for output, detail in outputs.items():
                print(
                    f"  {output}: difference {detail['difference']:.3g} "
                    f"within {detail['tolerance']:.3g} "
                    f"({detail['exactness']}) -> {fact.status.value}"
                )

        if len(ledger):
            print()
            print(ledger.render())
        if args.json_out:
            Path(args.json_out).write_text(ledger.to_json(), encoding="utf-8")
        duplicated = _report_duplicates(claims)
        # The same block ``lanky check`` prints, through lanky's own printer:
        # what explains each refutation belongs under its line, and not only
        # in the JSON ledger.
        refuted = ledger.by_status(Status.REFUTED)
        if refuted:
            print()
        for fact in refuted:
            print(f"REFUTED {fact.owner} at {fact.where}: {fact.statement}")
            for line in refutation_lines(fact):
                print(f"  {line}")
        return 1 if refuted or failures or duplicated else 0

    # lanky's Verb protocol calls ``run``; ``loopty run`` used to call the verb
    # itself, and both spellings are kept so that neither caller has to know.
    __call__ = run


def build_parser() -> argparse.ArgumentParser:
    """The argument parser for the ``loopty`` command."""
    parser = argparse.ArgumentParser(
        prog="loopty",
        description="loopy, with types: a typed polyhedral layer over loopy.",
        epilog=REPO_URL,
    )
    parser.add_argument(
        "--version", action="store_true", help="print the version and exit"
    )
    verbs = parser.add_subparsers(dest="verb")

    run_verb = RunVerb()
    run_parser = verbs.add_parser(run_verb.name, help=run_verb.help)
    run_verb.add_arguments(run_parser)

    check_parser = verbs.add_parser(
        "check", help="check a file's obligations (an alias of 'lanky check')"
    )
    check_parser.add_argument("file", help="the Python file to check")
    check_parser.add_argument(
        "--json", dest="json_out", help="write the ledger as JSON"
    )
    check_parser.add_argument(
        "--verbose", action="store_true", help="say why an oracle was skipped"
    )
    return parser


def check(args: Any) -> int:
    """Run ``lanky check`` on the file, so that one ledger covers both halves."""
    from lanky.cli import CheckVerb

    verb = CheckVerb()
    namespace = argparse.Namespace(
        file=args.file,
        json=getattr(args, "json_out", None),
        verbose=getattr(args, "verbose", False),
    )
    return verb.run(namespace)


def main(argv: list[str] | None = None) -> int:
    """Entry point of the ``loopty`` console script."""
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.version:
        print(f"loopty {__version__}")
        return 0
    if args.verb is None:
        parser.print_help()
        return 0
    if args.verb == "run":
        return RunVerb().run(args)
    return check(args)


if __name__ == "__main__":
    raise SystemExit(main())
