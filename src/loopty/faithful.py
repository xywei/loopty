"""The faithfulness fact: the traced term computes what the body computes.

A kernel has two meanings. The native run is the body on real arrays, the
reference implementation. The traced term is what the body did once, at a
generic point, and it is what every other fact is about and what lowering
compiles. Tracing refuses the ways a body can hide state from it that it knows
how to see (loop-carried names, containers, globals, whole-array operations,
side effects; see :mod:`loopty.trace`), and each of those checks has an edge.
The fact here is the check that has none, because it looks at results rather
than at spellings: whatever the body keeps where the tracer does not look shows
up as the two meanings disagreeing.

For each kernel, :func:`faithfulness_fact` runs the native body and the
interpreted term (:func:`loopty.interpret.interpret`) on the same inputs, each
on copies of its own, and compares every array argument afterwards. The inputs
are the module's example inputs, when it has any (the ``example_inputs()`` that
``loopty run`` reads, see :mod:`loopty.cli`), and :data:`SAMPLES` inputs drawn
from the declared types with every size at least 2, from a fixed seed. A
program gets the same fact (:meth:`loopty.kernel.Program.facts`): its term is
composed from what its body does with placeholders (:mod:`loopty.compose`),
which a body can tell from its arguments, and its inputs are drawn from the
types the term gives its parameters, which are its callees'. An
output is compared at its exactness class (:func:`loopty.tolerance.output_class`):
bit for bit when it is ``exact``, within the class's tolerance otherwise.

The fact, of kind ``trace-faithful``, is

* ``tested`` when every input either agreed or could not be run natively and
  at least one agreed,
* ``refuted`` at the first input that disagrees, with the input and the first
  differing cell, or with the error the term raised where the body ran, or
  with the :class:`~loopty.trace.TraceError` a native run of the body raised
  (a ``when`` guard whose native value is an integer, say), which refuses the
  body's spelling on every input, whether or not the term ran,
* ``assumed``, with the reason, when no input ran natively or when the term
  holds something the interpreter has no meaning for.

An input the body itself cannot run (a sampled input can break an assumption
the types do not state, such as offsets consistent with counts) says nothing
either way, and is listed in the provenance as skipped.

A kernel's postcondition is tested on the same runs (:func:`postcondition_fact`):
its return annotation is a claim about its parameters once the body has run,
so it is evaluated at what every native run left in them, and the fact of kind
``postcondition`` is ``tested`` by ``native`` when it held after every run that
ran and after one at least, ``refuted`` at the first run after which it is
false, and ``assumed``, with the reason, when nothing ran or it could not be
evaluated after some run. Deciding it from the term instead, by which
statement writes each cell last, is a question for isl that is not asked yet.

A ``layout`` fact the typing rules leave ``assumed`` is tried on the same runs
(:func:`layout_fact`): the claim that the rows of a ragged family stay inside
their buffer and apart while the kernel writes its counts or its offsets is
``refuted`` by ``native`` at the first run that reads a row the layout puts off
the buffer (a :class:`~loopty.arr.LayoutError`), or after which two rows share
a cell or one lies off the buffer. A run that keeps them so is no test of it,
since the claim is about every state of the run and not only its last.
"""

from __future__ import annotations

import dataclasses
import warnings
from collections.abc import Iterator, Mapping, Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np
from lanky.ledger import Fact, Status, fact_id
from lanky.prelude import FinType, Refined
from lanky.terms import evaluate, free_variables

from loopty.arr import Arr, LayoutError
from loopty.interpret import CheckFailed, InterpretError, TooLarge, Touch, interpret
from loopty.term import ArrType, Term, declared_layout
from loopty.tolerance import disagreement, output_class
from loopty.trace import TraceError

__all__ = [
    "KIND",
    "MAX_INSTANCES",
    "SAMPLES",
    "SEED",
    "SIZES",
    "Stopped",
    "Raised",
    "Collision",
    "Logged",
    "collision_fact",
    "collisions",
    "faithfulness_fact",
    "layout_fact",
    "no_term_fact",
    "postcondition_fact",
    "sample_arguments",
]

#: The kind of the fact, as the ledger and its JSON name it.
KIND = "trace-faithful"

#: How many inputs are drawn from the declared types, besides the examples.
SAMPLES = 3

#: The seed of the draws. Sample ``k`` is drawn from ``default_rng((SEED, k))``,
#: so each one can be drawn again on its own from what the fact records.
SEED = 0

#: The half-open range every size is drawn from. At least 2, so that a loop
#: runs more than one iteration, which is where state carried from one
#: iteration to the next shows.
SIZES = (2, 5)

#: How many statement instances and reduction terms, together, one interpreted
#: run may have. The interpreter runs in Python, one instance at a time, and an
#: example input written for a benchmark is skipped rather than interpreted, or
#: run natively, for minutes.
MAX_INSTANCES = 200_000

#: What the statement of the fact says, for every kernel.
STATEMENT = "the traced term computes what the body computes"


class _NoSample(Exception):
    """A declared type this module cannot draw a value of."""


def faithfulness_fact(
    kernel: Any,
    term: Term,
    owner: str,
    where: str,
    *,
    module: str | None = None,
    line: int | None = None,
    observed: list[tuple[str, Any, dict[str, Any]]] | None = None,
    touched: list[Logged] | None = None,
) -> Fact:
    """The ``trace-faithful`` fact of one kernel, established by running it.

    ``kernel`` is the decorated kernel, or program, which is called as the
    native run, and ``term`` its traced term. The id is keyed by the kernel's
    definition, ``owner``, ``module`` and ``line``, as the typing rules key
    theirs (see :func:`loopty.typing.facts_for`). Nothing here raises: an
    input that cannot be run is skipped, and anything unexpected leaves the
    fact ``assumed`` with the error as its reason.

    ``observed``, when given, collects every native run that ran, as
    ``(label, record, arguments)``, the arguments as the run left them, for
    :func:`postcondition_fact` to evaluate the postcondition at: on every
    input, those after a disagreement too, which settles this fact and not
    the postcondition. A run the body raised in is collected too, with a
    :class:`Raised` in place of the arguments, for :func:`layout_fact`.

    ``touched``, when given, collects the cells the interpreted term touched
    on every input on which it agreed with the body, as :class:`Logged`
    entries, for :func:`collision_fact`.
    """
    identifier = fact_id(KIND, owner, module=module, line=line)
    inputs: list[dict[str, Any]] = []

    def fact(status: Status, **provenance: Any) -> Fact:
        return Fact(
            id=identifier,
            kind=KIND,
            statement=STATEMENT,
            term=None,
            status=status,
            decided_by=None if status is Status.ASSUMED else "interpreter",
            provenance={
                "inputs": inputs,
                "seed": SEED,
                "samples": SAMPLES,
                **provenance,
            },
            where=where,
            owner=owner,
        )

    # The comparison is settled at the first input the term cannot be judged
    # on or differs at; the inputs after it still run natively when
    # ``observed`` is asked for, since the postcondition is tested on every
    # input, and a run past a disagreement can refute it.
    verdict: Fact | None = None
    try:
        for label, arguments, recorded in _inputs(kernel, term):
            if verdict is not None and observed is None:
                break
            if isinstance(arguments, str):
                if verdict is None:
                    inputs.append({"input": label, "outcome": f"skipped: {arguments}"})
                continue
            after = failed = None
            if observed is not None:

                def after(
                    native: dict[str, Any], label: str = label, recorded: Any = recorded
                ) -> None:
                    observed.append((label, recorded, native))

                def failed(
                    error: Exception, label: str = label, recorded: Any = recorded
                ) -> None:
                    observed.append((label, recorded, Raised(error)))

            log: list[Touch] | None = [] if touched is not None else None
            outcome = _compare(kernel, term, label, arguments, after, failed, log)
            if outcome is None and touched is not None and log is not None:
                touched.append(Logged(label, _sizes_of(term, arguments), log))
            if verdict is not None:
                continue
            if outcome is None:
                inputs.append({"input": label, "outcome": "agreed"})
                continue
            kind, detail = outcome
            if kind == "skipped":
                inputs.append({"input": label, "outcome": f"skipped: {detail}"})
                continue
            if kind == "unknown":
                verdict = fact(Status.ASSUMED, reason=detail)
                continue
            counterexample, reason = detail
            inputs.append({"input": label, "outcome": "differed"})
            extra = {"arguments": recorded} if recorded is not None else {}
            verdict = fact(
                Status.REFUTED, counterexample=counterexample, reason=reason, **extra
            )
    except Exception as exc:  # noqa: BLE001 - a fact, never a crash of the check
        if observed is not None:
            # The inputs after this one were never run: the postcondition is
            # not tested on what was collected so far alone.
            observed.append(
                ("the inputs left", None, Stopped(f"{type(exc).__name__}: {exc}"))
            )
        if verdict is None:
            return fact(
                Status.ASSUMED,
                reason=(
                    f"the comparison could not be made: {type(exc).__name__}: {exc}"
                ),
            )
    if verdict is not None:
        return verdict
    agreed = sum(entry["outcome"] == "agreed" for entry in inputs)
    if not agreed:
        skipped = "; ".join(f"{e['input']}: {e['outcome']}" for e in inputs)
        return fact(
            Status.ASSUMED,
            reason=f"no input ran natively, so nothing was compared ({skipped})",
        )
    return fact(Status.TESTED, compared=agreed)


def no_term_fact(
    owner: str,
    where: str,
    reason: str,
    *,
    module: str | None = None,
    line: int | None = None,
) -> Fact:
    """The ``trace-faithful`` fact of something with no term to compare.

    ``assumed``, with ``reason``, under the id :func:`faithfulness_fact` would
    give it: a program whose term cannot be built (:mod:`loopty.compose`),
    which still runs natively, has nothing for the interpreter to run.
    """
    return Fact(
        id=fact_id(KIND, owner, module=module, line=line),
        kind=KIND,
        statement=STATEMENT,
        term=None,
        status=Status.ASSUMED,
        provenance={"reason": reason},
        where=where,
        owner=owner,
    )


@dataclass(frozen=True)
class Raised:
    """In place of a run's arguments: the body raised ``error`` on the input.

    :func:`faithfulness_fact` collects one for an input the body cannot run,
    which says nothing of the term or of the postcondition, and
    :func:`layout_fact` reads, since a body that reads a row off its buffer
    is one whose layout the run broke.
    """

    error: Exception


@dataclass(frozen=True)
class Stopped:
    """In place of a run's arguments: the runs stopped here, and why.

    :func:`faithfulness_fact` ends what it collects with one when an input
    raised past what it catches, so that :func:`postcondition_fact` counts
    the inputs never run as runs it could not evaluate the postcondition
    after, and leaves it ``assumed``.
    """

    reason: str


def postcondition_fact(
    kernel: Any,
    term: Term,
    observed: Sequence[tuple[str, Any, Mapping[str, Any]]],
    owner: str,
    where: str,
    *,
    module: str | None = None,
    line: int | None = None,
) -> Fact:
    """A kernel's postcondition, tested against what its native runs left.

    ``observed`` is what :func:`faithfulness_fact` collected: for every input
    the body ran, its label, what the provenance keeps of it, and the
    arguments as the run left them. The postcondition is evaluated there,
    each array by its cells and each size as the arguments determine it
    (:func:`loopty.contract.resolve_sizes`). The fact keeps the id, the
    statement and the kind of the one :func:`loopty.typing.postcondition_facts`
    states, and its term is an :class:`~loopty.typing.AfterCall`, which no
    oracle mistakes for a closed proposition.

    A call of a function of :mod:`loopty.math` in it is numpy's, as in the
    body: ``-> all(y[i] == sqrt(x[i]) for i in Fin[n])``.

    It is ``tested`` only when it held after every run: one after which it
    cannot be evaluated (it reads a cell the run's arrays do not have, or a
    name nothing gives a value) leaves it ``assumed``, however many others
    it held after, since a program that calls the kernel skips a check on the
    strength of a ``tested`` postcondition (see :mod:`loopty.compose`).
    """
    from lanky.terms import render, truth_value

    from loopty.contract import resolve_sizes
    from loopty.math import FUNCTIONS as MATH
    from loopty.typing import AfterCall, postcondition_id

    post = term.post
    inputs: list[dict[str, Any]] = []

    def fact(status: Status, **provenance: Any) -> Fact:
        return Fact(
            id=postcondition_id(owner, module=module, line=line),
            kind="postcondition",
            statement=render(post),
            term=AfterCall(term.name, post),
            status=status,
            decided_by="native" if status is not Status.ASSUMED else None,
            provenance={
                "kernel": term.name,
                "inputs": inputs,
                "seed": SEED,
                "samples": SAMPLES,
                **provenance,
            },
            where=where,
            owner=owner,
        )

    held = 0
    for label, recorded, arguments in observed:
        if isinstance(arguments, Raised):
            # The body refused the input: no run the contract let in.
            continue
        if isinstance(arguments, Stopped):
            inputs.append({"input": label, "outcome": f"not run: {arguments.reason}"})
            continue
        context: dict[str, Any] = dict(arguments)
        try:
            sizes = resolve_sizes(getattr(kernel, "arg_types", {}), arguments)
        except Exception:  # noqa: BLE001 - a size the claim names is missed below
            sizes = {}
        for name, value in sizes.items():
            context.setdefault(name, value)
        # A claim may call loopty.math's functions, as the body does.
        for name, function in MATH.items():
            context.setdefault(name, function)
        try:
            with np.errstate(all="ignore"), warnings.catch_warnings():
                warnings.simplefilter("ignore")
                holds = truth_value(evaluate(post, context), post)
        except Exception as exc:  # noqa: BLE001 - said in the provenance
            inputs.append(
                {
                    "input": label,
                    "outcome": f"not evaluated: {type(exc).__name__}: {exc}",
                }
            )
            continue
        if not holds:
            inputs.append({"input": label, "outcome": "false"})
            extra = {"arguments": recorded} if recorded is not None else {}
            return fact(
                Status.REFUTED,
                counterexample={"input": label},
                reason=(
                    f"on {label}, the body run natively leaves {render(post)} "
                    f"false, so {term.name} does not establish its postcondition"
                ),
                **extra,
            )
        inputs.append({"input": label, "outcome": "held"})
        held += 1
    if not held:
        tried = "; ".join(f"{e['input']}: {e['outcome']}" for e in inputs)
        return fact(
            Status.ASSUMED,
            reason=(
                "no native run left anything to evaluate it at"
                + (f" ({tried})" if tried else "")
            ),
        )
    unevaluated = [e for e in inputs if e["outcome"] != "held"]
    if unevaluated:
        # A run after which the claim cannot be evaluated is one it says
        # nothing of: it reads a cell the run's arrays do not have, say,
        # which an empty input never reaches. Held after the others, it is
        # not borne out by every run, and a program would skip a check on it.
        tried = "; ".join(f"{e['input']}: {e['outcome']}" for e in unevaluated)
        return fact(
            Status.ASSUMED,
            compared=held,
            reason=(
                f"it held after {held} native run{'s' if held != 1 else ''}, and "
                f"could not be evaluated after the others ({tried})"
            ),
        )
    return fact(Status.TESTED, compared=held)


def layout_fact(
    term: Term,
    fact: Fact,
    observed: Sequence[tuple[str, Any, Any]],
) -> Fact:
    """A ``layout`` fact the typing rules left ``assumed``, tried on native runs.

    ``fact`` is one :func:`loopty.typing.layout_facts` states, and
    ``observed`` what :func:`faithfulness_fact` collected. A fact with an isl
    question is the isl oracle's, and is returned as it is, as is one with
    another status. Otherwise the runs are read in order: one in which the
    body read a row of the family's arrays off its buffer, through the
    counts and the offsets the kernel declares as the run had left them,
    refutes the fact with that input, as does one that leaves two rows of
    the family on one cell, or one off the buffer. The rows are where the
    native run reads them: row ``q`` starts at ``off[q]``, or at the array's
    own offset when the kernel declares none, and is ``cnt[q]`` long, or
    ``off[q + 1] - off[q]`` without counts, and an empty row is nowhere.
    A run that leaves them apart and inside is no test of the fact, which is
    about every state the run passes through; the fact stays ``assumed``.
    """
    if fact.status is not Status.ASSUMED or fact.term is not None:
        return fact
    provenance = fact.provenance
    arrays = list(provenance.get("arrays", ()))
    counts = provenance.get("counts")
    if not arrays or counts is None:
        return fact
    for label, recorded, arguments in observed:
        if isinstance(arguments, Stopped):
            continue
        if isinstance(arguments, Raised):
            error = arguments.error
            if not (isinstance(error, LayoutError) and error.array in arrays):
                continue
            why = (
                f"on {label}, the body run natively reads {error.array} through "
                f"the layout it has written, and {error}"
            )
        else:
            broken = _broken_layout(term, counts, arrays, arguments)
            if broken is None:
                continue
            why = f"on {label}, the body run natively leaves {broken}"
        extra = {"arguments": recorded} if recorded is not None else {}
        return fact.with_status(
            Status.REFUTED,
            decided_by="native",
            counterexample={"input": label},
            reason=f"{why}, so the rows do not stay inside the buffer and apart",
            **extra,
        )
    return fact


@dataclass(frozen=True)
class Logged:
    """The cells an interpreted run touched on one input (:func:`faithfulness_fact`).

    ``label`` names the input, ``sizes`` are its sizes, and ``log`` the
    cells, in the order the run touched them.
    """

    label: str
    sizes: dict[str, int]
    log: list[Touch]


@dataclass(frozen=True)
class Collision:
    """Two instances that touched one cell of an array on one input.

    ``first`` and ``second`` are ``(statement id, coordinates)``, the
    coordinates by loop name, ``first`` the one the run reached first;
    ``reads`` are the cells the subscript of each read on the way to the
    cell, with what they held, ``cell[0, 1] = 1``. ``relation`` is for a
    caller that asked about several relations, which of them the pair is in
    (:meth:`loopty.schedule.Schedule._real_pair`).
    """

    label: str
    sizes: dict[str, int]
    array: str
    cell: tuple[Any, ...]
    first: tuple[str, dict[str, int]]
    second: tuple[str, dict[str, int]]
    reads: tuple[str, ...] = ()
    relation: int = 0

    @property
    def instances(self) -> str:
        """``S1[c=0, j=1] and S1[c=1, j=2]``."""
        return f"{_instance(*self.first)} and {_instance(*self.second)}"

    @property
    def at(self) -> str:
        """The input and its sizes, ``example_inputs() (nc=2, nv=4)``."""
        if self.label.startswith("sample ") and self.label.endswith(")"):
            return self.label  # a drawn input's label has its sizes
        sizes = ", ".join(
            f"{name}={value}" for name, value in sorted(self.sizes.items())
        )
        return f"{self.label}" + (f" ({sizes})" if sizes else "")

    @property
    def cell_text(self) -> str:
        """``y[1]``."""
        return f"{self.array}[{', '.join(str(index) for index in self.cell)}]"

    @property
    def where(self) -> str:
        """``: cell[0, 1] = 1 and cell[1, 2] = 1``, or nothing."""
        return f", where {' and '.join(self.reads)}" if self.reads else ""


def _instance(stmt_id: str, coords: Mapping[str, int]) -> str:
    return f"{stmt_id}[{', '.join(f'{k}={v}' for k, v in coords.items())}]"


def _sizes_of(term: Term, arguments: Mapping[str, Any]) -> dict[str, int]:
    """The sizes one input gives the term, or none where they cannot be read."""
    from loopty.contract import resolve_sizes

    try:
        sizes = resolve_sizes(dict(term.params), arguments)
    except Exception:  # noqa: BLE001 - a witness is printed without them
        return {}
    return {
        name: int(value)
        for name, value in sizes.items()
        if isinstance(value, int | np.integer)
    }


def collisions(
    term: Term,
    logged: Logged,
    array: str,
    pairs: Any,
    limit: int | None = 10_000,
    only: Any = None,
) -> Iterator[Collision]:
    """Pairs of instances that touched one cell of ``array`` on one input.

    ``pairs(first, second)`` says which pairs are wanted, each a
    :class:`loopty.interpret.Touch`, ``first`` the earlier: the writes of
    one statement for a disjoint-writes fact, the two ends of a dependence
    a schedule reverses for a cast. The cells are those the run computed,
    which isl cannot when a subscript is read from an array, ``y[cell[c,
    j]]``. The pairs come in the order the run reached the second of the
    two, and at most ``limit`` touches of the array are looked at, every
    one with ``None``. ``only(touch)``, when given, keeps the touches the
    pairs are among, and the others are not counted: the writes of one
    statement, of which a run has as many as the statement has instances.
    """
    by_stmt = {stmt.id: stmt for stmt in term.stmts}
    seen: dict[tuple[Any, ...], list[Touch]] = {}
    reads: dict[tuple[str, tuple[int, ...]], dict[tuple[str, tuple], Any]] | None
    reads = None
    count = 0
    for touch in logged.log:
        if touch.array != array or (only is not None and not only(touch)):
            continue
        count += 1
        if limit is not None and count > limit:
            return
        earlier = seen.setdefault(touch.cell, [])
        for first in earlier:
            if (first.stmt, first.coords) == (touch.stmt, touch.coords):
                continue
            if not pairs(first, touch):
                continue
            if reads is None:
                reads = _reads_by_instance(logged)
            yield Collision(
                label=logged.label,
                sizes=dict(logged.sizes),
                array=array,
                cell=touch.cell,
                first=(first.stmt, _coords(by_stmt, first)),
                second=(touch.stmt, _coords(by_stmt, touch)),
                reads=_subscript_reads(by_stmt, logged, (first, touch), reads),
            )
        earlier.append(touch)


def _coords(by_stmt: Mapping[str, Any], touch: Touch) -> dict[str, int]:
    stmt = by_stmt.get(touch.stmt)
    names = stmt.inames if stmt is not None else ()
    return dict(zip(names, (int(value) for value in touch.coords), strict=False))


def _reads_by_instance(
    logged: Logged,
) -> dict[tuple[str, tuple[int, ...]], dict[tuple[str, tuple], Any]]:
    """Every cell each instance read on one input, with what it held there."""
    out: dict[tuple[str, tuple[int, ...]], dict[tuple[str, tuple], Any]] = {}
    for entry in logged.log:
        if entry.kind == "read":
            cells = out.setdefault((entry.stmt, entry.coords), {})
            cells.setdefault((entry.array, entry.cell), entry.value)
    return out


def _subscript_reads(
    by_stmt: Mapping[str, Any],
    logged: Logged,
    touches: Sequence[Touch],
    reads: Mapping[tuple[str, tuple[int, ...]], Mapping[tuple[str, tuple], Any]],
) -> tuple[str, ...]:
    """What each instance's cell was reached through, ``cell[0, 1] = 1``.

    The reads inside the subscript of the access that touched the cell, at
    the instance's loop variables, with the value the run read there: the
    cell the statement writes, for a write, and the reads of the array in
    its right-hand side and its guard, for a read. ``reads`` is
    :func:`_reads_by_instance` of ``logged``.
    """
    from lanky.terms import evaluate as value_of

    from loopty.trace import accesses_in

    out: list[str] = []
    for touch in touches:
        stmt = by_stmt.get(touch.stmt)
        if stmt is None:
            continue
        if touch.kind == "write":
            outer = [stmt.assignee]
        else:
            outer = [
                access
                for part in (stmt.expr, stmt.guard)
                if part is not None
                for access in accesses_in(part)
                if access.array == touch.array
            ]
        env = {**logged.sizes, **_coords(by_stmt, touch)}
        held = reads.get((touch.stmt, touch.coords), {})
        for access in outer:
            for inner in accesses_in(tuple(access.indices)):
                try:
                    cell = tuple(int(value_of(index, env)) for index in inner.indices)
                except Exception:  # noqa: BLE001 - a subscript not read back is left out
                    continue
                if (inner.array, cell) not in held:
                    continue
                text = (
                    f"{inner.array}[{', '.join(str(i) for i in cell)}] = "
                    f"{_python(held[inner.array, cell])}"
                )
                if text not in out:
                    out.append(text)
    return tuple(out)


def collision_fact(term: Term, fact: Fact, touched: Sequence[Logged]) -> Fact:
    """A disjoint-writes fact isl cannot decide, decided on the kernel's runs.

    ``fact`` is one :func:`loopty.typing.write_disjointness_facts` leaves to
    the native runs: a write through a subscript read from an array,
    ``y[perm[i]] = x[i]``, which isl widens to every cell. ``touched`` is what
    :func:`faithfulness_fact` logged of the term's runs, the example inputs
    first. Two instances of the statement that wrote one cell on an input
    refute it, with that input, and the cells their subscripts read: the
    collision is real, not one of the widened relation. With none on every
    input that ran it is ``tested``, and ``assumed`` when no input ran.
    """
    if fact.status is not Status.ASSUMED or not fact.provenance.get("native"):
        return fact
    stmt_id = fact.provenance["statement"]
    array = fact.provenance["array"]

    def written(touch: Touch) -> bool:
        return touch.kind == "write" and touch.stmt == stmt_id

    def writes_of_one(first: Touch, second: Touch) -> bool:
        return written(first) and written(second)

    tried: list[str] = []
    for logged in touched:
        # Every write of the statement is looked at: a scan cut short could
        # miss a later collision and read the input as tried. Each cell holds
        # one write until the first collision, so the scan is one pass.
        found = next(
            collisions(term, logged, array, writes_of_one, limit=None, only=written),
            None,
        )
        if found is None:
            tried.append(logged.label)
            continue
        return fact.with_status(
            Status.REFUTED,
            decided_by="native",
            witness=found.instances,
            counterexample={"input": found.label, **found.sizes},
            reason=(
                f"{found.instances} both write {found.cell_text} on {found.at}"
                f"{found.where}, so the later write overwrites the earlier one"
            ),
        )
    if not tried:
        return fact.with_status(
            Status.ASSUMED,
            reason=(
                f"{fact.provenance.get('reason', '')}, and no input ran natively"
            ),
        )
    # No ``reason``: on a tested fact lanky reads one as a pass on thin
    # evidence, and this one is a pass on every input that ran.
    provenance = {key: v for key, v in fact.provenance.items() if key != "reason"}
    return dataclasses.replace(
        fact,
        status=Status.TESTED,
        decided_by="native",
        provenance={
            **provenance,
            "inputs": tried,
            "detail": (
                f"no two instances of {stmt_id} wrote one cell of {array} on "
                + ", ".join(tried)
            ),
        },
    )


def _broken_layout(
    term: Term, counts: str, arrays: Sequence[str], arguments: Mapping[str, Any]
) -> str | None:
    """How the rows of a family lie off their buffer or on one another, or ``None``.

    Read off what a native run left in ``arguments``, as :func:`layout_fact`
    says.
    """
    first = arguments.get(arrays[0])
    if not isinstance(first, Arr) or not first.is_ragged:
        return None
    size = int(np.asarray(first.numpy()).size)
    offsets_name = term.offsets_of(counts)
    declared = arguments.get(offsets_name) if offsets_name is not None else None
    starts = (
        np.asarray(declared.numpy() if isinstance(declared, Arr) else declared)
        if declared is not None
        else np.asarray(first.offsets)
    ).reshape(-1)
    held = arguments.get(counts)
    rows = int(np.asarray(first.offsets).size) - 1
    if held is not None:
        cells = np.asarray(held.numpy() if isinstance(held, Arr) else held)
        lengths = [max(int(cells.reshape(-1)[q]), 0) for q in range(rows)]
    else:
        lengths = [max(int(starts[q + 1]) - int(starts[q]), 0) for q in range(rows)]
    laid = [
        (int(starts[q]), int(starts[q]) + lengths[q], q)
        for q in range(rows)
        if lengths[q] > 0
    ]
    for start, stop, q in laid:
        if start < 0 or stop > size:
            return (
                f"row {q} of {arrays[0]} at cells {start} to {stop} of a buffer of "
                f"{size}"
            )
    laid.sort()
    for (start, stop, q), (other, _end, p) in zip(laid, laid[1:], strict=False):
        if other < stop:
            return (
                f"rows {q} and {p} of {arrays[0]} on cell {other}, row {q} at cells "
                f"{start} to {stop} and row {p} from {other} on"
            )
    return None


# {{{ inputs


def _inputs(
    kernel: Any, term: Term
) -> Iterator[tuple[str, dict[str, Any] | str, dict[str, Any] | None]]:
    """``(label, arguments, record)`` for every input, examples first.

    ``arguments`` is a string instead when the input could not be made, saying
    why; ``record`` is what the provenance keeps of a sampled input, which is
    small enough to write down, and ``None`` for an example.
    """
    from loopty.cli import EXAMPLE_FUNCTION, select_inputs

    factory = getattr(kernel.fn, "__globals__", {}).get(EXAMPLE_FUNCTION)
    label = f"{EXAMPLE_FUNCTION}()"
    if callable(factory):
        try:
            everything = factory()
        except Exception as exc:  # noqa: BLE001 - the module's code, reported
            yield label, f"{label} raised {type(exc).__name__}: {exc}", None
        else:
            if not isinstance(everything, dict):
                yield label, f"{label} did not return a dictionary", None
            else:
                chosen = select_inputs(everything, kernel.__name__)
                if chosen is not None:
                    yield label, chosen, None
    for index in range(SAMPLES):
        try:
            arguments, sizes = sample_arguments(term, (SEED, index))
        except _NoSample as exc:
            yield f"sample {index + 1}", str(exc), None
            continue
        shown = ", ".join(f"{name}={value}" for name, value in sorted(sizes.items()))
        label = f"sample {index + 1}" + (f" ({shown})" if shown else "")
        yield label, arguments, _record(arguments)


def _record(arguments: Mapping[str, Any]) -> dict[str, Any]:
    """A sampled input, written down for the provenance."""
    out: dict[str, Any] = {}
    for name, value in arguments.items():
        if isinstance(value, Arr) and value.domain is not None:
            out[name] = value.cells().tolist()
            out[f"{name} domain"] = repr(value.domain)
        elif isinstance(value, Arr):
            out[name] = value.numpy().tolist()
            if value.is_ragged:
                out[f"{name} offsets"] = value.offsets.tolist()
        else:
            out[name] = value
    return out


def sample_arguments(
    term: Term, seed: Any = SEED
) -> tuple[dict[str, Any], dict[str, int]]:
    """Arguments for ``term`` drawn from its declared types, with their sizes.

    Every size a shape or a sort names is drawn from :data:`SIZES`, and every
    array has the extents its axes then evaluate to, with a ragged axis taking
    its row lengths from the counts array it names, and an array over a
    polyhedral domain a draw at each of the domain's points, stored in its box.
    Elements are drawn by sort:
    a point of ``Fin[m]`` below ``m``, a ``Nat`` below 4 (which keeps ragged
    rows short, and allows empty ones), an ``Int`` between -3 and 3, a ``Real``
    from a standard normal. Arrays the kernel writes are drawn too, because a
    kernel may read what it writes before writing it, and both runs have to
    start from the same values. An integral scalar named like a size is that
    size. Offsets the kernel declares beside a ragged array
    (:func:`loopty.term.declared_layout`) are that array's offsets rather than
    a draw, since a call has to pass them so (:mod:`loopty.contract`).
    """
    rng = np.random.default_rng(seed)
    types = dict(term.params)
    names = sorted((_size_names(term) - set(types)) | _integral_scalars(term))
    sizes = {name: int(rng.integers(*SIZES)) for name in names}
    arguments: dict[str, Any] = {}
    ragged: list[tuple[str, ArrType]] = []
    for name, typ in term.params:
        if isinstance(typ, ArrType):
            if typ.domain is not None:
                needed = typ.domain.size_names()
                fixed = typ.domain.fixed({size: sizes[size] for size in needed})
                values = _draw(typ.dtype, (fixed.count,), sizes, rng)
                arguments[name] = Arr.from_cells(typ.domain, values, **fixed.sizes)
                continue
            if any(typ.ragged):
                ragged.append((name, typ))
                continue
            shape = tuple(_extent(axis, sizes) for axis in typ.axes)
            arguments[name] = Arr.from_numpy(_draw(typ.dtype, shape, sizes, rng))
        elif name in sizes:
            arguments[name] = sizes[name]
        else:
            arguments[name] = _draw(typ, (), sizes, rng).item()
    for name, typ in ragged:
        if typ.ragged != (False, True):
            raise _NoSample(f"{name} is ragged below its second axis")
        counts_name = getattr(typ.axes[1], "name", None)
        counts = arguments.get(counts_name)
        if not isinstance(counts, Arr) or counts.numpy().ndim != 1:
            raise _NoSample(f"the row lengths of {name} are not a drawn array")
        lengths = counts.numpy()
        if lengths.dtype.kind not in "iu" or np.any(lengths < 0):
            raise _NoSample(f"the row lengths {counts_name} of {name} are negative")
        values = _draw(typ.dtype, (int(lengths.sum()),), sizes, rng)
        arguments[name] = Arr.ragged(lengths, values=values)
    for name, (_counts, offsets) in declared_layout(
        term.params, term.offsets
    ).items():
        drawn = arguments.get(offsets) if offsets else None
        layout = arguments.get(name)
        if not isinstance(drawn, Arr) or not isinstance(layout, Arr):
            continue
        if drawn.numpy().shape == layout.offsets.shape:
            arguments[offsets] = Arr.from_numpy(
                layout.offsets.astype(drawn.dtype, copy=True)
            )
    ordered = {name: arguments[name] for name, _typ in term.params}
    return ordered, sizes


def _size_names(term: Term) -> set[str]:
    """Every name a shape or a sort of ``term`` mentions."""
    out = set(term.sizes)
    for _name, typ in term.params:
        sort = typ.dtype if isinstance(typ, ArrType) else typ
        base = sort.base if isinstance(sort, Refined) else sort
        if isinstance(base, FinType):
            out |= set(free_variables(base.bound))
        if isinstance(typ, ArrType):
            for axis in typ.shape_terms:
                out |= set(free_variables(axis))
    return out


def _integral_scalars(term: Term) -> set[str]:
    """The scalar parameters of an integral sort that a shape or a sort names."""
    from loopty.contract import integral_sort

    named = _size_names(term)
    return {
        name
        for name, typ in term.params
        if not isinstance(typ, ArrType) and name in named and integral_sort(typ)
    }


def _extent(axis: Any, sizes: Mapping[str, int]) -> int:
    """The number of cells of one dense axis at the drawn sizes."""
    try:
        value = evaluate(axis, dict(sizes))
    except Exception as exc:  # noqa: BLE001 - reported as a sample not drawn
        raise _NoSample(f"the axis {axis} has no value at {dict(sizes)}") from exc
    if isinstance(value, bool) or not isinstance(value, int | np.integer):
        raise _NoSample(f"the axis {axis} is {value!r} at {dict(sizes)}")
    if value < 0:
        raise _NoSample(f"the axis {axis} is negative at {dict(sizes)}")
    return int(value)


def _draw(
    sort: Any, shape: tuple[int, ...], sizes: Mapping[str, int], rng: Any
) -> np.ndarray:
    """Values of one sort, of one shape; see :func:`sample_arguments`."""
    base = sort.base if isinstance(sort, Refined) else sort
    if isinstance(base, FinType):
        bound = _extent(base.bound, sizes)
        if bound == 0:
            if int(np.prod(shape)) == 0:
                return np.zeros(shape, dtype=np.int64)
            raise _NoSample(f"{base} has no points at {dict(sizes)}")
        return rng.integers(0, bound, size=shape, dtype=np.int64)
    name = getattr(base, "name", None)
    if name == "Nat":
        return rng.integers(0, 4, size=shape, dtype=np.int64)
    if name == "Int":
        return rng.integers(-3, 4, size=shape, dtype=np.int64)
    if name == "Bool":
        return rng.integers(0, 2, size=shape).astype(bool)
    if name == "Real":
        return rng.standard_normal(shape)
    try:
        dtype = np.dtype(base)
    except TypeError as exc:
        raise _NoSample(f"no way to draw a value of the sort {base}") from exc
    if dtype.kind == "f":
        return rng.standard_normal(shape).astype(dtype)
    if dtype.kind == "c":
        pair = rng.standard_normal((*shape, 2))
        return (pair[..., 0] + 1j * pair[..., 1]).astype(dtype)
    if dtype.kind == "b":
        return rng.integers(0, 2, size=shape).astype(bool)
    if dtype.kind == "u":
        return rng.integers(0, 4, size=shape).astype(dtype)
    if dtype.kind == "i":
        return rng.integers(-3, 4, size=shape).astype(dtype)
    raise _NoSample(f"no way to draw a value of the sort {base}")


# }}}


# {{{ one comparison


def _copy(value: Any) -> Any:
    """A private copy of one argument, so that the two runs cannot see each other."""
    if isinstance(value, Arr):
        return value.copy()
    if isinstance(value, np.ndarray):
        return value.copy()
    return value


def _buffer(value: Any) -> np.ndarray | None:
    """The numpy buffer of an array argument, or ``None`` for a scalar.

    An array over a domain gives its cells in their order, which is what the
    positions a message names count (:func:`loopty.contract._cell_label`).
    """
    if isinstance(value, Arr) and value.domain is not None:
        return value.cells()
    if isinstance(value, Arr):
        return value.numpy()
    if isinstance(value, np.ndarray):
        return value
    return None


def _python(value: Any) -> Any:
    """A numpy scalar as the Python number it is, for a message and for JSON."""
    return value.item() if isinstance(value, np.generic) else value


def _cell(name: str, value: Any, position: int) -> str:
    """``y[1, 0]``: the cell at a flat position of an argument."""
    from loopty.contract import _cell_label

    return _cell_label(name, value, position)


def _compare(
    kernel: Any,
    term: Term,
    label: str,
    arguments: Mapping[str, Any],
    after: Any = None,
    failed: Any = None,
    log: list[Touch] | None = None,
) -> tuple[str, Any] | None:
    """Run both meanings on one input; ``None`` when they agree.

    Otherwise ``("skipped", why)`` when the body cannot run the input or the
    input is too large to interpret, ``("unknown", why)`` when the term cannot
    be interpreted at all, and ``("differ", (counterexample, reason))`` when
    the two disagree. A :class:`~loopty.trace.TraceError` from the native run
    is a disagreement and not a skipped input: it refuses the body's spelling,
    not the values it was given. So is a checked point of a program's term
    that stops the interpreted run where the body ran
    (:class:`~loopty.interpret.CheckFailed`), since the compiled program
    would refuse an input the native one takes.

    The interpreter runs first, because it is the one with a bound on its work
    (:data:`MAX_INSTANCES`): an example input written for a benchmark is
    skipped without running the body on it either. What it raised is judged
    only once the body has run, since an input the body refuses says nothing
    about the term.

    ``after``, when given, is called with the native run's arguments once the
    body has run them without an error, as the run left them, and
    ``failed`` with what the body raised on an input it cannot run. ``log``,
    when given, collects the cells the interpreted term touches
    (:func:`loopty.interpret.interpret`).
    """
    native = {name: _copy(value) for name, value in arguments.items()}
    interpreted = {name: _copy(value) for name, value in arguments.items()}
    with np.errstate(all="ignore"), warnings.catch_warnings():
        warnings.simplefilter("ignore")
        failure: Exception | None = None
        try:
            interpret(term, interpreted, limit=MAX_INSTANCES, log=log)
        except TooLarge as exc:
            return "skipped", f"too large to interpret: {exc}"
        except Exception as exc:  # noqa: BLE001 - judged below, after the body
            failure = exc
        try:
            kernel(**native)
        except TraceError as exc:
            # Not an input the body refuses but a spelling it is refused for,
            # whatever the input: a guard such as ``~(i > 0)``, whose native
            # value is a bitwise integer where the trace recorded ``not``. So
            # it refutes whether or not the term ran, and says which.
            error = f"{type(exc).__name__}: {exc}"
            counterexample = {"input": label, "body raised": error}
            if failure is None:
                ran = "the traced term, interpreted, runs and "
            else:
                counterexample["term raised"] = (
                    f"{type(failure).__name__}: {failure}"
                )
                ran = ""
            return "differ", (
                counterexample,
                f"on {label} {ran}the body, run natively, is refused, so the "
                f"term is not what the body computes. {error}",
            )
        except Exception as exc:  # noqa: BLE001 - an input the body refuses
            if failed is not None:
                failed(exc)
            return "skipped", f"the body raised {type(exc).__name__}: {exc}"
    if after is not None:
        after(native)
    if isinstance(failure, CheckFailed):
        error = f"{type(failure).__name__}: {failure}"
        return "differ", (
            {"input": label, "term raised": error},
            f"on {label} the body runs, and the traced term, interpreted, "
            f"stops at a checked point the native run passes: {failure}",
        )
    if isinstance(failure, InterpretError):
        return "unknown", f"the term cannot be interpreted: {failure}"
    if isinstance(failure, IndexError | ArithmeticError):
        # What a program raises: a cell that is not there, a division by zero.
        # The body ran the same input without either.
        error = f"{type(failure).__name__}: {failure}"
        return "differ", (
            {"input": label, "term raised": error},
            f"on {label} the body runs and the traced term, interpreted, "
            f"raises {error}, so the term is not what the body computes",
        )
    if failure is not None:
        # The interpreter's own failure, not the term's.
        return "unknown", (
            f"the interpreter failed on {label}: {type(failure).__name__}: {failure}"
        )
    for name, typ in term.params:
        if not isinstance(typ, ArrType):
            continue
        want, got = _buffer(native[name]), _buffer(interpreted[name])
        if want is None or got is None:
            continue
        exactness = output_class(term, name)
        mask = disagreement(got, want, exactness).reshape(-1)
        if not mask.any():
            continue
        position = int(np.flatnonzero(mask)[0])
        cell = _cell(name, native[name], position)
        body = _python(want.reshape(-1)[position])
        traced = _python(got.reshape(-1)[position])
        how = (
            "bit for bit"
            if exactness == "exact"
            else f"at the tolerance of the {exactness} class"
        )
        return "differ", (
            {"input": label, "cell": cell, "body": body, "term": traced},
            f"on {label}, {cell} is {body!r} after the native run and "
            f"{traced!r} after the traced term, interpreted, and the two are "
            f"compared {how}. The term does not compute what the body computes: "
            "tracing runs the body once, at one generic point, so state the "
            "body keeps outside the arrays, or reaches in a way the trace does "
            "not follow, is seen at one iteration only.",
        )
    return None


# }}}
