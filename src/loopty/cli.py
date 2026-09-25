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
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

from loopty import __version__

__all__ = ["RunVerb", "build_parser", "check", "collect", "example_inputs", "main"]

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


def _native(obj: Any) -> Any:
    """The Python body of a scheduled kernel, if there is one to compare with."""
    source = getattr(obj, "source", obj)
    return source if callable(source) else None


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
        from lanky.ledger import Ledger, Status
        from lanky.plugins import registry

        from loopty.executor import LoopyExecutor, emit_code
        from loopty.schedule import IllegalCast, Schedule
        from loopty.trace import TraceError

        target = getattr(args, "target", None)
        before = len(registry.objects)
        module = import_path(args.file)
        schedules, kernels = collect(module, registry.objects[before:])

        scheduled = {id(schedule.source) for schedule in schedules}
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
        ledger = Ledger()
        for schedule in schedules:
            name = _name_of(schedule)
            print(f"{name}: {schedule!r}")
            for fact in schedule.facts():
                ledger.add(fact)
            ok, reason = schedule.buildable
            if not ok:
                print(f"  not buildable for the {schedule.target} target: {reason}")
                continue
            if args.emit_code:
                print(emit_code(schedule))
            inputs = schedule.examples or example_inputs(module, name)
            if inputs is None:
                print(f"  no example inputs for {name}; add {EXAMPLE_FUNCTION}()")
                continue
            try:
                native = _native(schedule)
                if native is None:
                    executor.run(schedule, **inputs)
                    print(f"  ran {name} on the {schedule.target} target")
                    continue
                fact = executor.differential(native, schedule, inputs)
            except (IllegalCast, RuntimeError, ValueError, TypeError) as exc:
                print(f"  {type(exc).__name__}: {exc}")
                failures += 1
                continue
            ledger.add(fact)
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
        refuted = ledger.by_status(Status.REFUTED)
        for fact in refuted:
            print(f"REFUTED {fact.owner}: {fact.statement}")
        return 1 if refuted or failures else 0

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
