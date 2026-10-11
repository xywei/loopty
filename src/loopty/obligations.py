"""loopty's facts stated for Lean: isl decides, Lean checks.

The design idea. isl decides every fact of :mod:`loopty.typing` and every cast
of :mod:`loopty.schedule` in milliseconds, at the trust of a decision
procedure. isl produces no proofs, only verdicts, so a fact it decided reads
``decided``. Lean can re-decide the same fact on its own, and does: each fact
is stated as a lanky term, a guarded universal, and lanky's Lean oracle proves
it with ``omega`` (:mod:`lanky.obligations`). A fact Lean proves reads
``proved``.

That is worth something only if Lean's statement is independent of isl's
work. So every statement here is built from loopty's own description of the
kernel (:mod:`loopty.description`): the loops' names, bounds and guards as
loopty wrote them for isl, the subscripts as the tracer recorded them, the
source order as the 2d+1 vectors :mod:`loopty.flow` defines, and a
schedule's steps as the maps each step wrote (:class:`loopty.schedule.Step`).
Never from a set isl computed: no dependence relation, no image, no
coalesced or simplified set. A fact over such a set is not stated, and says
why, and stays ``decided``.

isl's work is used, but only as hints, which Lean re-checks and which carry
no trust (see :mod:`lanky.obligations`): the functional form of a step's map
as the witness that the step is total, the pieces of ``lexmax`` as the
witness of an existential, and isl's counterexample to a refuted fact, at
which Lean proves the fact false. A wrong hint leaves a goal ``omega``
cannot close, and a wrong counterexample one Lean cannot prove false, which
the ledger names.

A fact is stated as parts, each a closed term proved on its own: one per
place an access is made over, one per dependence class of a cast. A part
reads like the claim on a whiteboard::

    ∀ (n : Int) (cnt : Int → Int) (r j : Int),
      n ≥ 0 → (0 ≤ r ∧ r < n ∧ 0 ≤ j ∧ j < cnt r ∧ cnt r ≥ 0) →
      0 ≤ r ∧ r < n ∧ 0 ≤ j ∧ j < cnt r

A reflected array cell, ``cnt[r]`` in a ragged bound, is the application
``cnt r``: an atom for ``omega`` as its parameter is one for isl.

Nothing here runs unless Lean is asked: a fact's question carries a function
that states it (:attr:`loopty.oracle.Subset.lean`), which lanky's Lean oracle
calls, and a check without Lean costs nothing more than before.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from functools import partial
from typing import Any

import islpy as isl
import pymbolic.primitives as prim
from lanky.prelude import FnType, Int
from lanky.terms import (
    Comparison,
    Forall,
    LogicalAnd,
    LogicalNot,
    LogicalOr,
    Var,
    conjuncts,
    render,
)

from loopty import flow
from loopty.description import (
    Description,
    Unstatable,
    conjunction,
    description_of,
    disjunction,
)
from loopty.hypotheses import substitute
from loopty.term import Stmt, Term

__all__ = [
    "ISL_ONLY",
    "Claim",
    "IslOnly",
    "Stated",
    "defined_obligation",
    "discharged_obligation",
    "disjoint_writes_obligation",
    "element_sort_obligation",
    "in_bounds_obligation",
    "mixed_in_bounds_obligation",
    "cast_obligation",
    "lexicographically_before",
    "ordering_obligation",
    "prove",
    "shadow_hints",
    "stated_writers",
    "written_before",
]


def _lanky_obligations() -> Any:
    """lanky's obligation types, imported when Lean asks, never with loopty.

    A lanky without them has a Lean oracle that never asks, so loopty runs
    against it as before.
    """
    import lanky.obligations

    return lanky.obligations


# {{{ a closed claim: its variables, hypotheses and goal


def _names(expr: Any, out: dict[str, int], bound: frozenset[str] = frozenset()) -> None:
    """Collect the free names of ``expr``: a variable's arity is 0, an array's its rank.

    Raises:
        Unstatable: If a name is read both as a number and as an array, or as
            arrays of two ranks.
    """

    def note(name: str, arity: int) -> None:
        if name in bound:
            return
        if out.get(name, arity) != arity:
            raise Unstatable(f"{name} is read both as a number and as an array")
        out[name] = arity

    if isinstance(expr, tuple | list):
        for item in expr:
            _names(item, out, bound)
        return
    if isinstance(expr, Forall) or type(expr).__name__ == "Exists":
        inner = bound | {var.name for var, _ in expr.binders}
        _names(expr.body, out, inner)
        if expr.guard is not None:
            _names(expr.guard, out, inner)
        return
    if isinstance(expr, prim.Subscript):
        index = expr.index if isinstance(expr.index, tuple) else (expr.index,)
        if isinstance(expr.aggregate, prim.Variable):
            note(expr.aggregate.name, len(index))
        else:
            _names(expr.aggregate, out, bound)
        _names(index, out, bound)
        return
    if isinstance(expr, prim.Call):
        if isinstance(expr.function, prim.Variable):
            note(expr.function.name, len(expr.parameters))
        else:
            _names(expr.function, out, bound)
        _names(tuple(expr.parameters), out, bound)
        return
    if isinstance(expr, prim.Variable):
        note(expr.name, 0)
        return
    if isinstance(expr, prim.ExpressionNode):
        from lanky.terms import init_args

        for child in init_args(expr):
            _names(child, out, bound)


def _array_sort(rank: int) -> Any:
    """``Int → Int → Int`` for a rank-two array: a cell is an integer atom."""
    sort: Any = Int
    for _ in range(rank):
        sort = FnType(Int, sort)
    return sort


class Claim:
    """One part's term, built up: its coordinates, hypotheses, and goal.

    The variables of the closed term are the coordinates the builder made
    (:meth:`coordinates`), and every other name the hypotheses and the goal
    mention: a size or a scalar of the kernel, an ``Int``, and an array whose
    cells a bound or a subscript reads, a function from its indices. A size
    of the kernel (:func:`loopty.flow.size_names`) is not negative, as isl is
    told of every size, and the claim says so.
    """

    def __init__(
        self,
        term: Term | None,
        sizes: frozenset[str] | None = None,
        taken: frozenset[str] = frozenset(),
    ) -> None:
        self.term = term
        self.taken: set[str] = set(taken)
        if term is not None:
            self.sizes = flow.size_names(term) if sizes is None else sizes
            self.taken |= {name for name, _ in (*term.params, *term.temporaries)}
            self.taken |= set(term.sizes) | {name for name, _ in term.reflected}
            for stmt in term.stmts:
                self.taken |= set(stmt.inames)
        else:
            self.sizes = frozenset() if sizes is None else sizes
        self.coords: list[Var] = []
        self.hypotheses: list[Any] = []

    def fresh(self, stem: str) -> Var:
        """A variable named like ``stem`` and like nothing else the claim names."""
        name = stem
        while name in self.taken:
            name += "_"
        self.taken.add(name)
        return Var(name)

    def coordinates(self, inames: Sequence[str], tag: str = "") -> list[Var]:
        """Variables for one instance's loops: the loops' own names, or tagged."""
        out = []
        for iname in inames:
            if tag:
                var = self.fresh(f"{iname}_{tag}")
            else:
                var = Var(iname)
                self.taken.add(iname)
            out.append(var)
        self.coords.extend(out)
        return out

    def fresh_all(self, names: Sequence[str], tag: str) -> dict[str, Var]:
        """Fresh variables for the loops ``names``, which the goal binds."""
        return {name: self.fresh(f"{name}_{tag}") for name in names}

    def close(self, goal: Any) -> Forall:
        """The closed term: every variable, the hypotheses, the goal."""
        names: dict[str, int] = {}
        _names(self.hypotheses, names)
        _names(goal, names)
        own = [var.name for var in self.coords]
        others = sorted(name for name in names if name not in own)
        scalars = [name for name in others if names[name] == 0]
        arrays = [name for name in others if names[name] > 0]
        binders = [(Var(name), Int) for name in scalars]
        binders += [(Var(name), _array_sort(names[name])) for name in arrays]
        binders += [(var, Int) for var in self.coords]
        sized = [
            Comparison(Var(name), ">=", 0) for name in scalars if name in self.sizes
        ]
        # each hypothesis once, conjunct by conjunct, and a size's
        # non-negativity left out of the goal, where a cell set repeats it
        seen: dict[str, Any] = {}
        for hypothesis in (*sized, *self.hypotheses):
            for part in conjuncts(hypothesis):
                if part is not True:
                    seen.setdefault(render(part), part)
        if isinstance(goal, LogicalAnd):
            said = {render(part) for part in sized}
            kept = [part for part in conjuncts(goal) if render(part) not in said]
            goal = conjunction(kept)
        guard = conjunction(list(seen.values()))
        return Forall(tuple(binders), goal, None if guard is True else guard)


def _equal(left: Sequence[Any], right: Sequence[Any]) -> Any:
    """``l0 = r0 ∧ l1 = r1 ∧ ...``."""
    pairs = zip(left, right, strict=True)
    return conjunction([Comparison(a, "==", b) for a, b in pairs])


def _described(domain: Any, what: str) -> Description:
    """The description loopty kept on ``domain``, or why there is none.

    Raises:
        Unstatable: For a set isl computed.
    """
    found = description_of(domain)
    if found is None:
        raise Unstatable(
            f"{what} is a set isl computed, which has no description of loopty's to "
            "state it from"
        )
    return found


def _renamed(expr: Any, inames: Sequence[str], coords: Sequence[Any]) -> Any:
    """``expr`` over the loops ``inames``, at the coordinates ``coords``."""
    mapping = {name: coord for name, coord in zip(inames, coords, strict=True)}
    return substitute(expr, mapping) if mapping else expr


def _affine(expr: Any) -> bool:
    """Whether isl reads ``expr`` exactly: no array read, no product of unknowns."""
    try:
        flow.expr_text(expr, None, None)
    except (flow.NonAffine, ValueError):
        return False
    return True


# }}}


# {{{ what a fact's question carries


@dataclass(frozen=True)
class Stated:
    """How a fact is stated for Lean: a function of isl's verdict, kept on its question.

    :attr:`loopty.oracle.Subset.lean` and the other questions hold one, and
    their ``lean_obligation()`` calls it with what :func:`loopty.oracle.decide`
    answers, so that a refuted fact is stated with its counterexample.
    """

    state: Callable[[Any], Any]
    rests_on: tuple[str, ...] = field(default=())

    def __call__(self, verdict: Any) -> Any:
        return self.state(verdict)


def _declined(reason: str) -> Any:
    """An obligation that says why there is nothing for Lean to check."""
    return _lanky_obligations().Obligation(declined=reason)


#: Why a fact loopty states for isl only is not stated for Lean.
ISL_ONLY = (
    "loopty states this kind of fact for isl only, from sets isl computed, and "
    "not yet from its own description of the kernel"
)


@dataclass(frozen=True)
class IslOnly:
    """The claim of a fact isl decided from sets it computed, which Lean is not asked.

    A fact with no term reads as one isl did not decide, to the coverage
    report and to ``--verbose``; a fact with this one says it is isl's, and
    why Lean is not asked about it. The ``definedness`` fact of
    :meth:`loopty.schedule.Schedule.substitute` has it: it is decided over the
    dependences of what the substitution leaves, which isl computes.
    """

    reason: str = ISL_ONLY

    def lean_obligation(self) -> Any:
        """Nothing for Lean to check, and why (see :mod:`lanky.obligations`)."""
        return _declined(self.reason)

    def __str__(self) -> str:
        return f"stated for isl only: {self.reason}"


#: Why a fact isl refuted, with no counterexample Lean can check, is not
#: offered to Lean as a claim.
REFUTED_UNSTATED = (
    "isl refuted it, and no counterexample to a part of it could be stated for "
    "Lean, so Lean is not asked to prove it"
)


def _obligation(
    parts: Sequence[Any],
    counterexample: Any = None,
    rests_on: Sequence[str] = (),
    refuted: bool = False,
) -> Any:
    """The obligation of ``parts``; a refuted one is checked at its counterexample.

    A fact isl refuted is offered to Lean only as its counterexample, which
    Lean proves the claim false at. With none that can be stated, Lean is not
    asked at all: were the parts offered as a claim, a proof of them would
    read ``proved`` where isl refuted, and the refutation would be lost.
    """
    if refuted and counterexample is None:
        return _declined(REFUTED_UNSTATED)
    lanky = _lanky_obligations()
    if "fallback" in getattr(lanky.Part, "__dataclass_fields__", {}):
        # what isl finds only when Lean asks: the case splits of shadow_hints
        parts = [
            part
            if part.fallback is not None
            else lanky.Part(
                part.name,
                part.label,
                part.term,
                part.hints,
                fallback=partial(shadow_hints, part.term),
            )
            for part in parts
        ]
    extra = {}
    if "stated_by" in getattr(lanky.Obligation, "__dataclass_fields__", {}):
        # who stated the parts, for the ledger's LEAN line
        extra["stated_by"] = "loopty"
    return lanky.Obligation(
        parts=tuple(parts),
        counterexample=counterexample,
        rests_on=tuple(rests_on),
        decider="isl",
        **extra,
    )


def _point(names: Sequence[str], values: Sequence[int]) -> dict[str, int]:
    return {name: int(value) for name, value in zip(names, values, strict=True)}


def _counterexample(part: Any, point: Mapping[str, int], text: str) -> Any:
    """A counterexample to ``part``, with a value for each of its variables.

    Variables the point does not reach (a size that is not one of isl's
    parameters, since nothing in the question mentions it) take ``0``,
    which a size's non-negativity allows; an array has no value, and a
    counterexample to a part that reads one is not stated
    (:func:`lanky.obligations.negation_source` says so).
    """
    values: dict[str, int] = {}
    for var, sort in part.term.binders:
        if sort is Int:
            values[var.name] = int(point.get(var.name, 0))
    return _lanky_obligations().Counterexample(part.name, values, text)


# }}}


# {{{ in bounds


def in_bounds_obligation(
    term: Term,
    places: Sequence[tuple[tuple[Any, ...], tuple[str, ...], Any]],
    cells: Any,
    array: str,
    text: str,
    verdict: Any,
) -> Any:
    """The in-bounds fact of one access, one part per place it is made over.

    ``places`` are what :func:`loopty.typing.in_bounds_facts` gathers for the
    access: ``(indices, inames, domain)``, the access made over the loops
    ``inames`` at the points of ``domain``. ``cells`` is the set of cells the
    array has, which :func:`loopty.flow.cell_set` built and described. Each
    part says: every point of the domain reaches a cell of the array.

    A refuted fact is stated with its counterexample: isl names a cell
    outside the array, and the instance that reaches it is read off the
    access, as a hint.
    """
    try:
        cell_set = _described(cells, f"the cells of {array}")
        parts = []
        for k, (indices, inames, domain) in enumerate(places):
            claim = Claim(term)
            coords = claim.coordinates(inames)
            described = _described(domain, f"the loops {', '.join(inames)} of {text}")
            claim.hypotheses.append(described.predicate(coords))
            at = [_renamed(index, inames, coords) for index in indices]
            goal = cell_set.predicate(at, others=dict(zip(inames, coords, strict=True)))
            name = "in_bounds" if len(places) == 1 else f"in_bounds_{k}"
            label = f"{text} over the loops {', '.join(inames)}"
            parts.append(_lanky_obligations().Part(name, label, claim.close(goal)))
    except Unstatable as exc:
        return _declined(str(exc))
    counterexample = None
    if not verdict.ok:
        counterexample = _reaching(places, verdict, parts, text)
    return _obligation(parts, counterexample, refuted=verdict.ok is False)


def _reaching(
    places: Sequence[tuple[tuple[Any, ...], tuple[str, ...], Any]],
    verdict: Any,
    parts: Sequence[Any],
    text: str,
) -> Any:
    """The instance that reaches isl's cell outside the array, as a counterexample."""
    cell = verdict.witness
    parameters = dict(verdict.parameters or {})
    if cell is None:
        return None
    for (indices, inames, domain), part in zip(places, parts, strict=True):
        try:
            relation = flow.access_relation(inames, domain, indices)
            names = relation.get_var_names(isl.dim_type.param)
            for name, value in parameters.items():
                if name in names:
                    relation = relation.fix_val(
                        isl.dim_type.param,
                        names.index(name),
                        isl.Val.int_from_si(relation.get_ctx(), int(value)),
                    )
            for position, value in enumerate(cell):
                relation = relation.fix_val(
                    isl.dim_type.out,
                    position,
                    isl.Val.int_from_si(relation.get_ctx(), int(value)),
                )
            reached = relation.domain()
            if reached.is_empty():
                continue
            point = reached.sample_point()
            coords = [
                point.get_coordinate_val(isl.dim_type.set, k).to_python()
                for k in range(len(inames))
            ]
        except Exception:  # noqa: BLE001 - a hint that cannot be had is no hint
            continue
        values = {**parameters, **_point(inames, coords)}
        shown = ", ".join(f"{name}={value}" for name, value in values.items())
        return _counterexample(part, values, shown)
    return None


# }}}


def mixed_in_bounds_obligation(
    term: Term,
    place: tuple[tuple[Any, ...], tuple[str, ...], Any],
    typed: Sequence[tuple[int, Any]],
    cells: Any,
    text: str,
    verdict: Any,
) -> Any:
    """An access with mixed subscripts in bounds, every axis of it.

    One part: at every point of the domain, if each index read from an array
    of a ``Fin`` sort is a point of that sort, which is what the type says
    and the contract and the element-sort facts keep, the cell is a cell of
    the array, on every axis. isl decided the affine axes alone; Lean checks
    the whole access, the typed axes from their sort.
    """
    indices, inames, domain = place
    try:
        cell_set = _described(cells, f"the cells of {text}")
        claim = Claim(term)
        coords = claim.coordinates(inames)
        described = _described(domain, f"the loops {', '.join(inames)} of {text}")
        claim.hypotheses.append(described.predicate(coords))
        at = [_renamed(index, inames, coords) for index in indices]
        for axis, bound in typed:
            claim.hypotheses.append(Comparison(0, "<=", at[axis]))
            claim.hypotheses.append(
                Comparison(at[axis], "<", _renamed(bound, inames, coords))
            )
        goal = cell_set.predicate(at, others=dict(zip(inames, coords, strict=True)))
        loops = ", ".join(inames)
        label = f"{text} over the loops {loops}, its typed indices in their sorts"
        part = _lanky_obligations().Part("in_bounds", label, claim.close(goal))
    except Unstatable as exc:
        return _declined(str(exc))
    return _obligation([part], refuted=verdict.ok is False)


# {{{ distinct instances write distinct cells


def disjoint_writes_obligation(
    term: Term, stmt: Stmt, array: str, indices: Sequence[Any], verdict: Any
) -> Any:
    """Distinct instances of ``stmt`` write distinct cells of ``array``.

    One part: two points of the statement's domain that differ write cells
    that differ. A write isl widened, through an index that is no affine
    expression, is about every cell and not the one written, and is not
    stated. A refuted fact is stated with isl's pair.
    """
    if not all(_affine(index) for index in indices):
        return _declined(
            f"the write {array}[...] has an index isl does not read exactly, so isl "
            "decided the question for every cell of the array, which is not this claim"
        )
    try:
        domain = _described(stmt.domain, f"the loops of {stmt.id}")
        claim = Claim(term)
        first = claim.coordinates(stmt.inames, "a")
        second = claim.coordinates(stmt.inames, "b")
        claim.hypotheses += [domain.predicate(first), domain.predicate(second)]
        if stmt.inames:
            claim.hypotheses.append(LogicalNot(_equal(first, second)))
            goal = LogicalNot(
                _equal(
                    [_renamed(i, stmt.inames, first) for i in indices],
                    [_renamed(i, stmt.inames, second) for i in indices],
                )
            )
        else:
            goal = True
        label = f"two instances of {stmt.id} writing {array}"
        part = _lanky_obligations().Part("disjoint", label, claim.close(goal))
    except Unstatable as exc:
        return _declined(str(exc))
    counterexample = None
    if not verdict.ok and verdict.witness:
        (source, sink) = verdict.witness
        values = dict(verdict.parameters or {})
        values |= _point([v.name for v in first], source[1 : 1 + len(first)])
        values |= _point([v.name for v in second], sink[1 : 1 + len(second)])
        counterexample = _counterexample(
            part, values, ", ".join(f"{k}={v}" for k, v in values.items())
        )
    return _obligation([part], counterexample, refuted=verdict.ok is False)


# }}}


# {{{ the value written is a point of the element sort


def element_sort_obligation(
    term: Term, stmt: Stmt, cells: Any, sort: Any, verdict: Any
) -> Any:
    """The value ``stmt`` writes is a point of ``sort``, at every instance."""
    try:
        domain = _described(stmt.domain, f"the loops of {stmt.id}")
        points = _described(cells, f"the points of {sort}")
        claim = Claim(term)
        coords = claim.coordinates(stmt.inames)
        claim.hypotheses.append(domain.predicate(coords))
        value = _renamed(stmt.expr, stmt.inames, coords)
        others = dict(zip(stmt.inames, coords, strict=True))
        goal = points.predicate([value], others=others)
        label = f"the value {stmt.id} writes"
        part = _lanky_obligations().Part("element_sort", label, claim.close(goal))
    except Unstatable as exc:
        return _declined(str(exc))
    counterexample = None
    if not verdict.ok and verdict.witness is not None:
        values = dict(verdict.parameters or {})
        values |= _point(stmt.inames, verdict.witness)
        counterexample = _counterexample(
            part, values, ", ".join(f"{k}={v}" for k, v in values.items())
        )
    return _obligation([part], counterexample, refuted=verdict.ok is False)


# }}}


# {{{ the order of instances


def lexicographically_before(first: Sequence[Any], second: Sequence[Any]) -> Any:
    """``first`` before ``second`` in lexicographic order, as a lanky proposition.

    ``a0 < b0 ∨ (a0 = b0 ∧ (a1 < b1 ∨ ...))``, with the positions both sides
    hold as numbers settled here: a time vector's constants, which say where
    a statement sits among its siblings, mostly are.
    """
    out: Any = False
    for a, b in reversed(list(zip(first, second, strict=True))):
        if isinstance(a, int) and isinstance(b, int):
            if a < b:
                out = True
            elif a > b:
                out = False
            continue
        earlier = Comparison(a, "<", b)
        if out is False:
            out = earlier
        elif out is True:
            out = LogicalOr((earlier, Comparison(a, "==", b)))
        else:
            out = LogicalOr((earlier, LogicalAnd((Comparison(a, "==", b), out))))
    return out


# }}}



# {{{ the source order runs every dependence forward


def _source_time(stmt: Stmt, depth: int, coords: Sequence[Any]) -> list[Any]:
    """A statement's 2d+1 time vector, as :func:`loopty.flow.schedule_of` writes it."""
    order = stmt.order or (0,) * (len(stmt.inames) + 1)
    out: list[Any] = []
    for k in range(depth):
        out.append(order[k] if k < len(order) else 0)
        out.append(coords[k] if k < len(stmt.inames) else 0)
    out.append(order[depth] if depth < len(order) else 0)
    return out


def _footprint_place(
    claim: Claim,
    stmt: Stmt,
    coords: Sequence[Any],
    inames: Sequence[str],
    domain: Any,
    tag: str,
) -> list[Any]:
    """The loops one access of ``stmt`` is made in, at an instance ``coords``.

    An access inside a sum is made at the sum's binders too, which are
    variables of their own here, in the sum's domain; one made once per row,
    the length of a ragged row, at the outer loops of the instance. Either way
    the access's domain is a hypothesis, as the footprint intersects with it
    (:func:`loopty.flow.footprints`).
    """
    own = len(stmt.inames)
    if len(inames) > own:
        extra = claim.coordinates(inames[own:], tag)
        place = [*coords, *extra]
    else:
        place = list(coords[: len(inames)])
    described = _described(domain, f"the loops {', '.join(inames)} of {stmt.id}")
    claim.hypotheses.append(described.predicate(place))
    return place


def _access_label(a: Stmt, first: tuple, b: Stmt, second: tuple) -> str:
    """``S0 writes u[t + 1, i] before S0 reads u[t, i - 1]``: one pair of accesses."""
    verbs = {"write": "writes", "read": "reads", "acc": "updates"}

    def said(stmt: Stmt, access: tuple) -> str:
        cell = f"{access[0]}[{', '.join(render(i) for i in access[1])}]"
        return f"{stmt.id} {verbs.get(access[2], access[2])} {cell}"

    return f"{said(a, first)} before {said(b, second)}"


def ordering_obligation(term: Term, verdict: Any) -> Any:
    """The source order runs every dependence forward, one part per pair of accesses.

    A dependence is *defined* as two instances that touch one cell, one of
    them writing it, which the source runs in that order
    (:func:`loopty.flow.dependences`), so each part is that definition and
    its conclusion: the claim holds by it, and Lean checks it holds of the
    footprints and the time vectors loopty wrote, whatever isl made of them.
    An access whose index isl does not read exactly touches every cell of
    its array, as its footprint does.
    """
    lanky = _lanky_obligations()
    depth = flow.instance_space_depth(term)
    try:
        accesses = [
            (stmt, access)
            for stmt in term.stmts
            for access in flow.statement_accesses(stmt, term)
        ]
        parts = []
        for n, ((a, first), (b, second)) in enumerate(
            (one, other) for one in accesses for other in accesses
        ):
            if first[0] != second[0]:
                continue
            if first[2] not in ("write", "acc") and second[2] not in ("write", "acc"):
                continue
            if len(first[1]) != len(second[1]):
                continue
            claim = Claim(term)
            x = claim.coordinates(a.inames, "a")
            y = claim.coordinates(b.inames, "b")
            for stmt, coords in ((a, x), (b, y)):
                domain = _described(stmt.domain, f"the loops of {stmt.id}")
                claim.hypotheses.append(domain.predicate(coords))
            at_a = _footprint_place(claim, a, x, first[3], first[4], "c")
            at_b = _footprint_place(claim, b, y, second[3], second[4], "d")
            if all(_affine(index) for index in (*first[1], *second[1])):
                claim.hypotheses.append(
                    _equal(
                        [_renamed(i, first[3], at_a) for i in first[1]],
                        [_renamed(i, second[3], at_b) for i in second[1]],
                    )
                )
            before = lexicographically_before(
                _source_time(a, depth, x), _source_time(b, depth, y)
            )
            claim.hypotheses.append(before)
            parts.append(
                lanky.Part(
                    f"dependence_{n}",
                    _access_label(a, first, b, second),
                    claim.close(before),
                )
            )
    except Unstatable as exc:
        return _declined(str(exc))
    if not parts:
        return _declined("the term has no two accesses a dependence could join")
    return _obligation(parts, refuted=verdict.ok is False)


# }}}


# {{{ inside one kernel, every read of an array sees an earlier write


def _both(first: Any, second: Any) -> tuple[Any, Any]:
    """Two isl objects with one list of parameters, so that isl combines them."""
    second = second.align_params(first.get_space())
    first = first.align_params(second.get_space())
    return first, second


def _time_map(stmt: Stmt, inames: Sequence[str], depth: int) -> Any:
    """``{ [loops] -> [time] }``: the 2d+1 vector of :func:`loopty.flow.schedule_of`.

    Over ``inames``, which may hold a sum's binders after the statement's
    loops; they do not move the time, since a sum runs inside one instance.
    """
    order = stmt.order or (0,) * (len(stmt.inames) + 1)
    time: list[str] = []
    for k in range(depth):
        time.append(str(order[k]) if k < len(order) else "0")
        time.append(stmt.inames[k] if k < len(stmt.inames) else "0")
    time.append(str(order[depth]) if depth < len(order) else "0")
    return isl.Map(f"{{ [{', '.join(inames)}] -> [{', '.join(time)}] }}")


def stated_writers(term: Term, writers: Sequence[Stmt]) -> list[Stmt]:
    """Writers whose domain is where they store: none under a guard isl cannot state.

    A guard conjunct isl cannot state (:attr:`loopty.term.Stmt.unnarrowed`,
    ``with when(x[i] > 0)``) leaves a statement's domain wider than the
    instances that store, so an instance of it in its domain is no witness
    that a cell was stored. Such a writer counts for nothing towards a cell
    being written, as :func:`loopty.flow.definedness` counts it. The flag of
    a checked point a program puts before a call is no such guard: no
    statement runs where one is set.
    """
    from lanky.terms import Subscript

    flags = {
        render(Comparison(Subscript(Var(flag), 0), "==", 0)) for flag, _ in term.checks
    }
    return [
        writer
        for writer in writers
        if all(conjunct in flags for conjunct, _why in writer.unnarrowed)
    ]


def written_before(
    term: Term, array: str, writers: Sequence[Stmt], places: Sequence[tuple]
) -> tuple[Any, list[list[Any]]] | None:
    """Whether every read of ``array`` sees a cell an earlier instance wrote.

    ``places`` are ``(statement, indices, inames, domain)`` of each read, and
    ``writers`` the statements that store into the array. For each read,
    each writer gives the pairs of a reading point and a writing instance of
    one cell, the write before the read in the source order; the read is
    defined where some writer gives one. ``None`` when isl shows a read that
    is not, which a kernel may well mean (it reads what the array held when
    the call started), or when a read or a write is one isl cannot list,
    which a write under a guard isl cannot state is (:func:`stated_writers`
    says which writers to ask about).
    Otherwise the question, the union of the reads no write precedes, an
    empty set, and the pairs, read by read and writer by writer, for the
    witness (:func:`defined_obligation`).
    """
    depth = flow.instance_space_depth(term)
    sizes = flow.size_names(term)
    if not writers or len(stated_writers(term, writers)) != len(writers):
        # an instance of a writer whose domain is wider than where it
        # stores witnesses nothing; the callers leave such writers out
        return None
    if not all(_affine(index) for w in writers for index in w.assignee.indices):
        return None
    question = None
    relations: list[list[Any]] = []
    for k, (stmt, indices, inames, domain) in enumerate(places):
        if len(inames) < len(stmt.inames) or not all(_affine(i) for i in indices):
            return None
        reads = flow.access_relation(inames, domain, indices)
        reading = _time_map(stmt, inames, depth)
        per_writer = []
        covered = None
        for writer in writers:
            writes = flow.access_relation(
                writer.inames, writer.domain, writer.assignee.indices
            )
            first, second = _both(reads, writes.reverse())
            same = first.apply_range(second)
            later = reading.apply_range(isl.Map.lex_gt(reading.get_space().range()))
            later = later.apply_range(_time_map(writer, writer.inames, depth).reverse())
            same, later = _both(same, later)
            pairs = same.intersect(later)
            per_writer.append(pairs)
            found = pairs.domain()
            if covered is None:
                covered = found
            else:
                covered, found = _both(covered, found)
                covered = covered.union(found)
        read_points = reads.domain()
        if covered is not None:
            read_points, covered = _both(read_points, covered)
            missing = read_points.subtract(covered)
        else:
            missing = read_points
        missing = flow.assume_sizes(missing, sizes)
        if not missing.is_empty():
            return None
        relations.append(per_writer)
        piece = isl.UnionSet.from_set(missing.set_tuple_name(f"read{k}"))
        question = piece if question is None else question.union(piece)
    if question is None:
        return None
    return question, relations


def defined_obligation(
    term: Term,
    array: str,
    writers: Sequence[Stmt],
    places: Sequence[tuple],
    relations: Sequence[Sequence[Any]],
    verdict: Any,
) -> Any:
    """Every read of ``array`` sees a cell an earlier instance wrote: a ∀∃, witnessed.

    One part per read: at every point of the read's domain, some writer has
    an instance in its domain that writes the cell read, earlier in the
    source order. The goal is a disjunction, one existential per writer, and
    its witness is isl's ``lexmax`` of the writing instances, in pieces: the
    region each writer is the witness in, a writer after the ones before it.
    """
    lanky = _lanky_obligations()
    depth = flow.instance_space_depth(term)
    reflected = dict(term.reflected)
    parts = []
    try:
        for k, ((stmt, indices, inames, domain), per_writer) in enumerate(
            zip(places, relations, strict=True)
        ):
            claim = Claim(term)
            coords = claim.coordinates(inames)
            at = dict(zip(inames, coords, strict=True))
            loops = ", ".join(inames)
            described = _described(domain, f"the loops {loops} of {stmt.id}")
            claim.hypotheses.append(described.predicate(coords))
            read = [_renamed(index, inames, coords) for index in indices]
            now = _source_time(stmt, depth, coords[: len(stmt.inames)])
            options = []
            for writer in writers:
                ys = claim.fresh_all(writer.inames, "w")
                values = [ys[name] for name in writer.inames]
                written = _described(writer.domain, f"the loops of {writer.id}")
                body = conjunction(
                    [
                        written.predicate(values),
                        _equal(
                            [
                                _renamed(i, writer.inames, values)
                                for i in writer.assignee.indices
                            ],
                            read,
                        ),
                        lexicographically_before(
                            _source_time(writer, depth, values), now
                        ),
                    ]
                )
                if values:
                    from lanky.terms import Exists

                    binders = tuple((ys[name], Int) for name in writer.inames)
                    body = Exists(binders, body)
                options.append(body)
            goal = disjunction(options) if len(options) > 1 else options[0]
            pieces = _writer_pieces(
                per_writer, coords, at, reflected, multiple=len(options) > 1
            )
            hints = lanky.Hints(witness=tuple(pieces)) if pieces else lanky.Hints()
            label = f"{stmt.id} reads {array}[{', '.join(render(i) for i in indices)}]"
            parts.append(lanky.Part(f"read_{k}", label, claim.close(goal), hints))
    except Unstatable as exc:
        return _declined(str(exc))
    return _obligation(parts, refuted=verdict.ok is False)


def _writer_pieces(
    per_writer: Sequence[Any],
    coords: Sequence[Any],
    at: Mapping[str, Any],
    reflected: Mapping[str, Any],
    multiple: bool,
) -> list[Any]:
    """isl's witness of who wrote each cell read: ``lexmax``, writer after writer.

    A hint: a region isl gives that this cannot write as a proposition, or a
    function it cannot write as terms, leaves the part without one.
    """
    lanky = _lanky_obligations()
    pieces = []
    taken = None
    try:
        for index, pairs in enumerate(per_writer):
            region = pairs.domain()
            if taken is not None:
                region, earlier = _both(region, taken)
                region = region.subtract(earlier)
            if region.is_empty():
                continue
            params = {
                name: (
                    substitute(reflected[name], dict(at))
                    if name in reflected
                    else Var(name)
                )
                for name in pairs.get_var_names(isl.dim_type.param)
            }
            best = isl.PwMultiAff.from_map(pairs.intersect_domain(region).lexmax())
            found: list[tuple[Any, Any]] = []
            best.foreach_piece(lambda where, value: found.append((where, value)))
            for where, value in found:
                condition = _set_term(where, list(coords), params)
                values = tuple(
                    _aff_term(value.get_at(k), list(coords), params)
                    for k in range(value.dim(isl.dim_type.out))
                )
                pieces.append(
                    lanky.Piece(values, condition, index if multiple else None)
                )
            if taken is None:
                taken = pairs.domain()
            else:
                taken, more = _both(taken, pairs.domain())
                taken = taken.union(more)
    except (Unstatable, isl.Error, KeyError, ValueError):
        return []
    if pieces:
        last = pieces[-1]
        pieces[-1] = lanky.Piece(last.values, None, last.disjunct)
    return pieces


# }}}


# {{{ a program's facts, under what held where a call is made


#: The term classes core Lean has no form for: a sum, an absolute value, an
#: elementary function and true division are Mathlib's.
_UNPRINTABLE = ("Sum", "Abs", "Elementary", "Quotient")


def _statable(expr: Any, arrays: frozenset[str], names: frozenset[str]) -> bool:
    """Whether ``expr`` is integer arithmetic over ``names`` and cells of ``arrays``.

    A hypothesis about a real cell, read over the integers, would be another
    hypothesis, one that may give Lean what the reals do not; it is left
    out, which can only make the claim harder.
    """
    found: dict[str, int] = {}
    try:
        _names(expr, found)
    except Unstatable:
        return False
    for name, arity in found.items():
        if arity and name not in arrays:
            return False
        if not arity and name not in names:
            return False

    def plain(node: Any) -> bool:
        if isinstance(node, float | complex) or type(node).__name__ in _UNPRINTABLE:
            return False
        if isinstance(node, tuple | list):
            return all(plain(item) for item in node)
        if isinstance(node, Forall) or type(node).__name__ == "Exists":
            return plain(node.body) and (node.guard is None or plain(node.guard))
        if isinstance(node, prim.ExpressionNode):
            from lanky.terms import init_args

            return all(plain(child) for child in init_args(node))
        return True

    return plain(expr)


def discharged_obligation(
    domain: Any,
    goal: Any,
    record: Sequence[tuple],
    integral: frozenset[str],
    known: frozenset[str],
    nonneg: frozenset[str],
    verdict: Any,
) -> Any:
    """A claim isl decided under hypotheses (:func:`loopty.hypotheses.discharge`).

    One part: at every point of the claim's domain, the hypotheses isl used
    imply the claim. A hypothesis is a claim about the program's arrays (a
    callee's postcondition, a theorem it cites, a type its contract checks),
    and an array is a function of its indices here, so the part holds of
    every array the hypotheses hold of. A universal hypothesis is what
    ``omega`` does not instantiate, and the instances isl took of it, at the
    cells the claim reads, are the part's hints. A hypothesis or a part of
    one that is not integer arithmetic over integer arrays is left out, as
    isl leaves it out (:func:`_statable`).
    """
    lanky = _lanky_obligations()
    try:
        described = _described(domain, "the claim's domain")
        dims = tuple(domain.get_var_names(isl.dim_type.set))
        claims = [entry[2] for entry in record if entry[0] == "claim"]
        names = frozenset(known) | frozenset(dims)
        if not _statable(goal, integral, names):
            return _declined(
                "the claim reads what is not integer arithmetic over integer "
                "arrays, which is stated for isl only"
            )
        claim = Claim(None, sizes=frozenset(nonneg), taken=names)
        coords = claim.coordinates(dims)
        claim.hypotheses.append(described.predicate(coords))
        for hypothesis in claims:
            for part in conjuncts(hypothesis):
                if _statable(part, integral, names):
                    claim.hypotheses.append(part)
        term = claim.close(goal)
    except Unstatable as exc:
        return _declined(str(exc))
    guard = (
        [render(part) for part in conjuncts(term.guard)]
        if term.guard is not None
        else []
    )
    instances = []
    seen: set[tuple[int, tuple[str, ...]]] = set()
    for entry in record:
        if entry[0] != "instance" or len(instances) >= 16:
            continue
        _kind, conjunct, values = entry
        try:
            position = guard.index(render(conjunct))
        except ValueError:
            continue
        # by their text: a term's == builds a proposition
        key = (position, tuple(render(value) for value in values))
        if key not in seen:
            seen.add(key)
            instances.append(lanky.Instance(position, tuple(values)))
    hints = lanky.Hints(instances=tuple(instances))
    part = lanky.Part("claim", "the claim where the hypotheses hold", term, hints)
    return _obligation([part], refuted=verdict.ok is False)


# }}}


# {{{ isl's case splits, for a part omega did not close


#: How fine the rational relaxation is read: a rational point with this
#: denominator is an integer point of the set scaled by it.
_SCALE = 64

#: The most values a variable may take for a split on it to be offered.
_SPLIT_VALUES = 4


def shadow_hints(term: Any) -> Any:
    """isl's case splits for a part ``omega`` did not close: a fallback hint.

    ``omega`` has no dark or grey shadows: it eliminates over the reals and
    can miss that the integers in between are none (Pugh's example, ``27 <=
    11x + 13y <= 45`` and ``-10 <= 7x - 9y <= 4``). Two splits give it the
    cases a person would:

    - *by rational bounds*: the variable whose values the hypotheses confine,
      over the rationals, to the fewest integers, at most four, split on each
      of them (``x = 1 ∨ x = 2``). The bounds are isl's, of the set scaled by
      :data:`_SCALE`, rounded outward, so the split holds of every point of
      the hypotheses;
    - *by the carrying level*: a hypothesis that is a disjunction, the source
      order of a dependence (``t_a < t_b ∨ t_a = t_b ∧ i_a < i_b``), split on
      the disjuncts isl finds possible beside the other hypotheses, the
      levels that can carry it.

    Only the linear hypotheses are read, an array cell an atom, and a
    hypothesis this cannot read is left out, which only widens the set:
    the splits hold all the same, and Lean proves each before it uses it.
    """
    lanky = _lanky_obligations()
    if not isinstance(term, Forall) or term.guard is None:
        return lanky.Hints()
    try:
        atoms: dict[str, Any] = {}
        names = [var.name for var, sort in term.binders if sort is Int]
        linear: list[str] = []
        disjunctive: list[Any] = []
        for hypothesis in conjuncts(term.guard):
            if isinstance(hypothesis, LogicalOr):
                disjunctive.append(hypothesis)
                continue
            text = _linear_text(hypothesis, names, atoms)
            if text is not None:
                linear.append(text)
    except (Unstatable, isl.Error, ValueError):
        return lanky.Hints()
    variables = [*names, *atoms]
    splits: list[Any] = []
    try:
        bound = _bounded_variable(linear, variables, names)
    except (Unstatable, isl.Error, ValueError):
        bound = None
    if bound is not None:
        name, values = bound
        splits.append(
            disjunction([Comparison(Var(name), "==", value) for value in values])
        )
    for hypothesis in disjunctive[:1]:
        possible = []
        try:
            for option in hypothesis.children:
                text = _linear_text(option, names, atoms)
                if text is None or _satisfiable([*linear, text], [*names, *atoms]):
                    possible.append(option)
        except (Unstatable, isl.Error, ValueError):
            continue
        if possible and len(possible) < len(hypothesis.children):
            splits.append(disjunction(possible))
    return lanky.Hints(splits=tuple(splits))


def _linear_text(prop: Any, names: Sequence[str], atoms: dict[str, Any]) -> str | None:
    """Linear comparisons, conjoined, as isl text over ``v0, v1, ...``; else ``None``.

    An array cell becomes a variable of its own, named in ``atoms`` by its
    text, the same cell the same variable.
    """
    from loopty.hypotheses import linear

    if isinstance(prop, LogicalAnd):
        parts = [_linear_text(child, names, atoms) for child in prop.children]
        return None if None in parts else " and ".join(f"({p})" for p in parts)
    if not isinstance(prop, prim.Comparison) or prop.operator == "!=":
        return None
    difference = prim.Sum((prop.left, prim.Product((-1, prop.right))))
    difference = _atomized(difference, atoms)
    form = linear(difference)
    if form is None:
        return None
    coefficients, constant = form
    order = [*names, *atoms]
    terms = []
    for name, value in coefficients.items():
        if name not in order:
            return None
        terms.append(f"{value} * v{order.index(name)}")
    expression = " + ".join([*terms, str(constant)]) or "0"
    relation = {"==": "=", "<": "<", "<=": "<=", ">": ">", ">=": ">="}[prop.operator]
    return f"{expression} {relation} 0"


def _atomized(expr: Any, atoms: dict[str, Any]) -> Any:
    """``expr`` with each array cell a variable named after its text."""
    if isinstance(expr, prim.Subscript | prim.Call):
        key = f"__cell_{len(atoms)}"
        for name, cell in atoms.items():
            if render(cell) == render(expr):
                return Var(name)
        atoms[key] = expr
        return Var(key)
    if isinstance(expr, prim.ExpressionNode):
        from lanky.terms import init_args

        return type(expr)(*(_atomized(arg, atoms) for arg in init_args(expr)))
    if isinstance(expr, tuple):
        return tuple(_atomized(item, atoms) for item in expr)
    return expr


def _satisfiable(constraints: Sequence[str], variables: Sequence[str]) -> bool:
    """Whether the constraints, over ``v0, v1, ...``, have an integer point."""
    dims = ", ".join(f"v{k}" for k in range(len(variables)))
    body = " and ".join(f"({c})" for c in constraints) or "1 = 1"
    return not isl.Set(f"{{ [{dims}] : {body} }}").is_empty()


def _bounded_variable(
    constraints: Sequence[str], variables: Sequence[str], names: Sequence[str]
) -> tuple[str, list[int]] | None:
    """The variable the rational relaxation confines to the fewest integers.

    The relaxation is read off the set scaled by :data:`_SCALE`: ``x = w / D``
    turns a rational point with denominator ``D`` into an integer point ``w``,
    so the bounds of ``w`` over it, divided by ``D`` and rounded outward,
    bound ``x`` over the rationals to within ``1 / D``, which can only add
    values. ``None`` when no variable is confined to :data:`_SPLIT_VALUES`
    integers or fewer, or the relaxation is empty, which ``omega``'s own
    elimination shows.
    """
    if not constraints:
        return None
    dims = ", ".join(f"v{k}" for k in range(len(variables)))
    scaled = [_scaled(text) for text in constraints]
    points = isl.Set(f"{{ [{dims}] : {' and '.join(scaled)} }}")
    if points.is_empty():
        return None
    best: tuple[str, list[int]] | None = None
    for position, name in enumerate(variables):
        if name not in names:
            continue
        try:
            low = _extreme(points.dim_min(position))
            high = _extreme(points.dim_max(position))
        except (isl.Error, ValueError):
            continue
        if low is None or high is None:
            continue
        values = list(range(low // _SCALE, -(-high // _SCALE) + 1))
        fewer = best is None or len(values) < len(best[1])
        if len(values) <= _SPLIT_VALUES and fewer:
            best = (name, values)
    return best


def _scaled(text: str) -> str:
    """``a * v + c rel 0`` as ``a * v + c * D rel 0``, the constraint at ``v / D``."""
    expression, relation = text.rsplit(" ", 2)[0], text.rsplit(" ", 2)[1]
    *terms, constant = expression.split(" + ")
    return " + ".join([*terms, str(int(constant) * _SCALE)]) + f" {relation} 0"


def _extreme(value: Any) -> int | None:
    """The constant a bound isl gives is, or ``None`` for one that is not."""
    pieces = value.get_pieces()
    if len(pieces) != 1:
        return None
    aff = pieces[0][1]
    if any(
        not aff.get_coefficient_val(isl.dim_type.in_, k).is_zero()
        for k in range(aff.dim(isl.dim_type.in_))
    ):
        return None
    found = aff.get_constant_val()
    if not found.is_int():
        return None
    return int(found.to_python())


# }}}


# {{{ isl's functions, as terms: the hints


def _rational(value: Any) -> Any:
    """An isl value as a Python ``Fraction``."""
    from fractions import Fraction

    return Fraction(str(value))


def _aff_term(aff: Any, inputs: Sequence[Any], params: Mapping[str, Any]) -> Any:
    """An isl quasi-affine function as a lanky term over ``inputs``.

    Raises:
        Unstatable: For a function whose value is not an integer combination
            of its inputs, its parameters and the floors of such.
    """
    parts: list[tuple[Any, Any]] = []
    for k, value in enumerate(inputs):
        parts.append((_rational(aff.get_coefficient_val(isl.dim_type.in_, k)), value))
    names = aff.get_var_names(isl.dim_type.param)
    for k, name in enumerate(names):
        parts.append(
            (_rational(aff.get_coefficient_val(isl.dim_type.param, k)), params[name])
        )
    for k in range(aff.dim(isl.dim_type.div)):
        floor = _floor_term(aff.get_div(k), inputs, params)
        parts.append((_rational(aff.get_coefficient_val(isl.dim_type.div, k)), floor))
    return _linear(parts, _rational(aff.get_constant_val()))


def _floor_term(argument: Any, inputs: Sequence[Any], params: Mapping[str, Any]) -> Any:
    """``floor(argument)`` for an isl aff with rational coefficients, as ``e // m``.

    Raises:
        Unstatable: For an argument that is itself a floor of something.
    """
    from math import lcm

    names = argument.get_var_names(isl.dim_type.param)
    coefficients = [
        _rational(argument.get_coefficient_val(isl.dim_type.in_, j))
        for j in range(len(inputs))
    ]
    coefficients += [
        _rational(argument.get_coefficient_val(isl.dim_type.param, j))
        for j in range(len(names))
    ]
    constant = _rational(argument.get_constant_val())
    if any(
        not argument.get_coefficient_val(isl.dim_type.div, j).is_zero()
        for j in range(argument.dim(isl.dim_type.div))
    ):
        raise Unstatable("a floor of a floor in isl's function")
    scale = lcm(*(c.denominator for c in (*coefficients, constant)))
    values = (*inputs, *(params[n] for n in names))
    numerator = _linear(
        [(c * scale, v) for c, v in zip(coefficients, values, strict=True)],
        constant * scale,
    )
    return prim.FloorDiv(numerator, scale) if scale != 1 else numerator


def _set_term(obj: Any, inputs: Sequence[Any], params: Mapping[str, Any]) -> Any:
    """An isl set over ``inputs`` as a lanky proposition: a piece's condition.

    Each basic set a conjunction of its constraints, the existential ones
    through the floors isl defines them by. A hint, so a set this cannot
    read raises and the hint is not given.

    Raises:
        Unstatable: For a constraint on a variable isl gives no floor for.
    """
    out = []
    for basic in obj.get_basic_sets():
        local = basic.get_local_space()
        floors = [
            _floor_term(local.get_div(k), inputs, params)
            for k in range(basic.dim(isl.dim_type.div))
        ]
        names = basic.get_var_names(isl.dim_type.param)
        parts = []
        for constraint in basic.get_constraints():
            terms = [
                (_rational(constraint.get_coefficient_val(isl.dim_type.set, k)), value)
                for k, value in enumerate(inputs)
            ]
            terms += [
                (
                    _rational(constraint.get_coefficient_val(isl.dim_type.param, k)),
                    params[n],
                )
                for k, n in enumerate(names)
            ]
            terms += [
                (_rational(constraint.get_coefficient_val(isl.dim_type.div, k)), floor)
                for k, floor in enumerate(floors)
            ]
            value = _linear(terms, _rational(constraint.get_constant_val()))
            relation = "==" if constraint.is_equality() else ">="
            parts.append(Comparison(value, relation, 0))
        out.append(conjunction(parts))
    return disjunction(out)


def _linear(parts: Sequence[tuple[Any, Any]], constant: Any) -> Any:
    """``c0 * x0 + ... + constant`` from integer coefficients, with zeros left out."""
    terms: list[Any] = []
    for coefficient, value in parts:
        if coefficient.denominator != 1:
            raise Unstatable(f"a rational coefficient {coefficient} in isl's function")
        coefficient = int(coefficient)
        if coefficient == 0:
            continue
        terms.append(value if coefficient == 1 else prim.Product((coefficient, value)))
    if constant.denominator != 1:
        raise Unstatable(f"a rational constant {constant} in isl's function")
    if int(constant) or not terms:
        terms.append(int(constant))
    return terms[0] if len(terms) == 1 else prim.Sum(tuple(terms))


def _functional_form(
    mapping: Any, inputs: Sequence[Any], params: Mapping[str, Any]
) -> list[tuple[Any, list[Any]]] | None:
    """isl's functional form of a step's map, piece by piece: the witness.

    Each piece is ``(condition, values)``: where it applies, as a proposition
    over ``inputs`` (``None`` for the last, which applies wherever the ones
    before do not), and one value per output. ``None`` when isl gives none,
    or one this cannot write as terms, which leaves the part to ``omega``
    alone (and to fail, since its goal is an existential): the witness is a
    hint and nothing rests on it.
    """
    try:
        function = isl.PwMultiAff.from_map(mapping)
        pieces: list[tuple[Any, Any]] = []
        function.foreach_piece(lambda where, value: pieces.append((where, value)))
        if not pieces:
            return None
        out = []
        for k, (where, value) in enumerate(pieces):
            condition = (
                None if k == len(pieces) - 1 else _set_term(where, inputs, params)
            )
            values = [
                _aff_term(value.get_at(j), inputs, params)
                for j in range(value.dim(isl.dim_type.out))
            ]
            out.append((condition, values))
        return out
    except (Unstatable, isl.Error, KeyError, ValueError):
        return None


# }}}


# {{{ a schedule's casts


def _schedule_module() -> Any:
    """:mod:`loopty.schedule`, imported when a cast is stated (it imports this one)."""
    from loopty import schedule

    return schedule


def _time(
    coords: Mapping[str, tuple[str, ...]],
    order: Sequence[str],
    tags: Mapping[str, str],
    stmt_id: str,
    values: Mapping[str, Any],
) -> list[Any]:
    """A statement's time vector at ``values``, as :func:`loopty.schedule._time_map`.

    The loops interleaved with the constants that place the statement among
    its siblings, its parallel loops left out, padded with zeros to the
    length the deepest statement needs.
    """
    schedule = _schedule_module()
    layout = schedule._Layout(stmt_ids=tuple(coords), coords=dict(coords))
    nests = schedule._nests(layout, order, dict(tags))
    coefficients = schedule._coefficients(layout.stmt_ids, nests)
    depth = max((len(nest) for nest in nests.values()), default=0)
    nest = nests[stmt_id]
    out: list[Any] = []
    for level, iname in enumerate(nest):
        out.append(coefficients[stmt_id][level])
        out.append(values[iname])
    out.append(coefficients[stmt_id][len(nest)])
    return out + [0] * (2 * depth + 1 - len(out))


def _relation(
    step: Any, stmt_id: str, before: Mapping[str, Any], after: Mapping[str, Any]
) -> Any:
    """One step's map of one statement, as a proposition between its coordinates.

    The statement's map as the step wrote it, and every coordinate the map
    does not take kept as it was, as :func:`loopty.schedule._step_map` keeps it.

    Raises:
        Unstatable: For a map isl computed (the part of a tile over one of its
            loops), which has no text of loopty's.
    """
    parts: list[Any] = []
    inputs: tuple[str, ...] = ()
    if stmt_id in step.isl_maps:
        written = step.maps.get(stmt_id)
        if written is None:
            raise Unstatable(
                f"{step.text} moves {stmt_id} by the part of its map over the loops "
                f"{stmt_id} runs in, which isl computed"
            )
        inputs = written.inputs
        parts.append(written.relation(before, after))
    for name in step.old[stmt_id]:
        if name not in inputs:
            parts.append(Comparison(after[name], "==", before[name]))
    return conjunction(parts)


def _chain(
    claim: Claim,
    origin: Any,
    stmt: Stmt,
    steps: Sequence[Any],
    tag: str,
) -> list[dict[str, Any]]:
    """One instance of ``stmt`` through ``steps``: its coordinates at every stage.

    Stage 0 is the instance in the term's loops, in the statement's domain;
    each step's map relates a stage to the next (:func:`_relation`). The
    hypotheses go to ``claim``, and the coordinates of each stage come back,
    by loop name.
    """
    names = origin._origin_layout.coords[stmt.id]
    coords = claim.coordinates(names, tag)
    domain = _described(stmt.domain, f"the loops of {stmt.id}")
    claim.hypotheses.append(domain.predicate(coords))
    stages = [dict(zip(names, coords, strict=True))]
    for k, step in enumerate(steps, start=1):
        after = dict(
            zip(
                step.new[stmt.id],
                claim.coordinates(step.new[stmt.id], f"{tag}{k}"),
                strict=True,
            )
        )
        claim.hypotheses.append(_relation(step, stmt.id, stages[-1], after))
        stages.append(after)
    return stages


def _classes(origin: Any) -> list[tuple[Stmt, Stmt, tuple, tuple]]:
    """Every pair of accesses a dependence can join, as the checker lists them.

    ``(a, b, (kind, array, indices, part), (kind, array, indices, part))``: an
    access of ``a`` and one of ``b`` to the same array, of the same rank, at
    least one a write (:func:`loopty.schedule._accesses`). A class isl finds
    empty is listed too: that it is empty is isl's verdict, and Lean proves
    it again.
    """
    schedule = _schedule_module()
    term = origin._term
    out = []
    for a in term.stmts:
        for b in term.stmts:
            for first in schedule._accesses(a, term):
                for second in schedule._accesses(b, term):
                    if first[1] != second[1] or len(first[2]) != len(second[2]):
                        continue
                    if first[0] == "read" and second[0] == "read":
                        continue
                    out.append((a, b, first, second))
    return out


def _same_cell(
    origin: Any,
    a: Stmt,
    b: Stmt,
    first: Sequence[Any],
    second: Sequence[Any],
    at_a: Mapping[str, Any],
    at_b: Mapping[str, Any],
) -> Any:
    """The two accesses touch one cell, axis by axis, as the checker reads it.

    An axis either side cannot state in isl is left out, as
    :func:`loopty.schedule._same_cell` leaves it out, and an access with an
    index that is not affine at all is about every cell, as
    :func:`loopty.flow.access_relation` widens it; the checker asks about
    the union of the two, and leaving out is the wider.
    """
    schedule = _schedule_module()
    plain = schedule._plain
    if not all(_affine(index) for index in (*first, *second)):
        return True
    params = set(origin._params) | set(origin._term.sizes)
    names_a = origin._origin_layout.coords[a.id]
    names_b = origin._origin_layout.coords[b.id]
    renaming_a = {name: prim.Variable(f"x{k}'") for k, name in enumerate(names_a)}
    renaming_b = {name: prim.Variable(f"y{k}'") for k, name in enumerate(names_b)}
    allowed_a = {f"x{k}'" for k in range(len(names_a))} | params
    allowed_b = {f"y{k}'" for k in range(len(names_b))} | params
    parts = []
    for left, right in zip(first, second, strict=True):
        if (
            schedule._index_text(left, renaming_a, allowed_a) is None
            or schedule._index_text(right, renaming_b, allowed_b) is None
        ):
            continue
        parts.append(
            Comparison(
                substitute(plain(left), dict(at_a)),
                "==",
                substitute(plain(right), dict(at_b)),
            )
        )
    return conjunction(parts)


_VERBS = {
    ("write", "read"): "read after",
    ("read", "write"): "overwritten after",
    ("write", "write"): "written again after",
}


def _class_label(a: Stmt, b: Stmt, first: tuple, second: tuple) -> str:
    """``u[t, i - 1] read by S0 after u[t + 1, i] is written by S0``."""
    from lanky.terms import render

    def cell(access: tuple) -> str:
        return f"{access[1]}[{', '.join(render(i) for i in access[2])}]"

    verb = {
        ("write", "read"): ("read", "written"),
        ("read", "write"): ("overwritten", "read"),
        ("write", "write"): ("written again", "written"),
    }[first[0], second[0]]
    return (
        f"{cell(second)} {verb[0]} by {b.id} after {cell(first)} is {verb[1]} "
        f"by {a.id}"
    )


def cast_obligation(cast: Any) -> Any:
    """A cast fact of a schedule, stated for Lean (:class:`loopty.schedule.Cast`).

    ``monotone``: one part per dependence class. Two instances, one of each
    statement, in their domains, that touch one cell and that the source runs
    in that order, keep that order after every step: the time vectors of the
    last step's order (its parallel loops left out) compare as the source's
    do. A cast whose tags put a loop on a hardware axis claims besides that no
    dependence joins two work items, which is not stated here yet, and the
    cast is left to isl with that reason.

    ``bijective``: three parts per statement the last step moves, over the
    instances as the steps before it left them: every instance has an image
    (an existential, with isl's functional form of the map as the witness),
    one only, and no two instances share one.

    A cast after a substitution is not stated: it changed what the instances
    read, and the dependences its later steps are checked against are isl's.
    """
    origin = cast.schedule
    steps = tuple(cast.steps)
    if any(step.substitution for step in steps):
        return _declined(
            "a substitution before this step changed what the instances read, and "
            "the dependences the step is checked against are isl's, not loopty's "
            "description"
        )
    try:
        if cast.kind == "bijective":
            parts = _bijective_parts(origin, steps)
        else:
            hardware = sorted(
                name
                for name, tag in steps[-1].tags.items()
                if tag.startswith(("g.", "l."))
            )
            if hardware:
                return _declined(
                    f"the loops {', '.join(hardware)} are on hardware axes, and the "
                    "claim that no dependence joins two work items is stated for "
                    "isl only"
                )
            parts = _monotone_parts(origin, steps)
    except Unstatable as exc:
        return _declined(str(exc))
    counterexample = None
    if cast.refusal is not None:
        counterexample = _refusal(origin, steps, cast.refusal, parts)
    return _obligation(parts, counterexample, refuted=cast.refuted)


def _monotone_parts(origin: Any, steps: Sequence[Any]) -> list[Any]:
    lanky = _lanky_obligations()
    term = origin._term
    stages_order = steps[-1].order if steps else origin._origin_order
    tags = steps[-1].tags if steps else {}
    last = steps[-1].new if steps else origin._origin_layout.coords
    parts = []
    assumed = steps[-1].assumptions if steps else ()
    for n, (a, b, first, second) in enumerate(_classes(origin)):
        claim = Claim(term)
        claim.hypotheses.extend(assumed)
        chain_a = _chain(claim, origin, a, steps, "a")
        chain_b = _chain(claim, origin, b, steps, "b")
        claim.hypotheses.append(
            _same_cell(origin, a, b, first[2], second[2], chain_a[0], chain_b[0])
        )
        source = origin._origin_layout.coords
        claim.hypotheses.append(
            lexicographically_before(
                _time(source, origin._origin_order, {}, a.id, chain_a[0]),
                _time(source, origin._origin_order, {}, b.id, chain_b[0]),
            )
        )
        goal = lexicographically_before(
            _time(last, stages_order, tags, a.id, chain_a[-1]),
            _time(last, stages_order, tags, b.id, chain_b[-1]),
        )
        label = _class_label(a, b, first, second)
        parts.append(lanky.Part(f"dependence_{n}", label, claim.close(goal)))
    return parts


def _bijective_parts(origin: Any, steps: Sequence[Any]) -> list[Any]:
    lanky = _lanky_obligations()
    term = origin._term
    step = steps[-1]
    earlier = steps[:-1]
    assumed = step.assumptions
    parts = []
    for stmt in term.stmts:
        if stmt.id not in step.isl_maps:
            continue
        old = step.old[stmt.id]
        new = step.new[stmt.id]
        mapping = step.isl_maps[stmt.id]
        inputs = tuple(mapping.get_var_names(isl.dim_type.in_))

        # every instance has an image, isl's functional form the witness
        claim = Claim(term)
        claim.hypotheses.extend(assumed)
        before = _chain(claim, origin, stmt, earlier, "a")[-1]
        images = claim.fresh_all(new, "image")
        relation = _relation(step, stmt.id, before, images)
        params = {name: Var(name) for name in mapping.get_var_names(isl.dim_type.param)}
        form = _functional_form(mapping, [before[name] for name in inputs], params)
        witness = []
        if form is not None:
            names = mapping.get_var_names(isl.dim_type.out)
            for condition, found in form:
                outputs = dict(zip(names, found, strict=True))
                values = tuple(outputs.get(name, before.get(name)) for name in new)
                witness.append(lanky.Piece(values, condition))
        from lanky.terms import Exists

        goal = Exists(tuple((images[name], Int) for name in new), relation)
        hints = lanky.Hints(witness=tuple(witness)) if witness else lanky.Hints()
        parts.append(
            lanky.Part(
                f"{stmt.id}_total",
                f"every instance of {stmt.id} has an image",
                claim.close(goal),
                hints,
            )
        )

        # one image only
        claim = Claim(term)
        claim.hypotheses.extend(assumed)
        before = _chain(claim, origin, stmt, earlier, "a")[-1]
        first = dict(zip(new, claim.coordinates(new, "c"), strict=True))
        second = dict(zip(new, claim.coordinates(new, "d"), strict=True))
        claim.hypotheses.append(_relation(step, stmt.id, before, first))
        claim.hypotheses.append(_relation(step, stmt.id, before, second))
        goal = _equal([first[n] for n in new], [second[n] for n in new])
        parts.append(
            lanky.Part(
                f"{stmt.id}_single_valued",
                f"no instance of {stmt.id} has two images",
                claim.close(goal),
            )
        )

        # no two instances share an image
        claim = Claim(term)
        claim.hypotheses.extend(assumed)
        one = _chain(claim, origin, stmt, earlier, "a")[-1]
        other = _chain(claim, origin, stmt, earlier, "b")[-1]
        image = dict(zip(new, claim.coordinates(new, "c"), strict=True))
        claim.hypotheses.append(_relation(step, stmt.id, one, image))
        claim.hypotheses.append(_relation(step, stmt.id, other, image))
        goal = _equal([one[n] for n in old], [other[n] for n in old])
        parts.append(
            lanky.Part(
                f"{stmt.id}_injective",
                f"no two instances of {stmt.id} share an image",
                claim.close(goal),
            )
        )
    if not parts:
        # a step that moves no statement renames nothing, which is one for one
        claim = Claim(term)
        parts.append(
            lanky.Part(
                "identity",
                f"{step.text} moves no instance",
                claim.close(True),
            )
        )
    return parts


def _refusal(
    origin: Any, steps: Sequence[Any], refusal: Any, parts: Sequence[Any]
) -> Any:
    """The refused class and isl's pair, as a counterexample to its part.

    isl's pair is two instances in the term's loops and the sizes; the
    coordinates after each step are read off the steps' maps by isl, a hint
    the counterexample's check re-reads.
    """
    from lanky.terms import structurally_equal

    dep, witness = refusal
    if not witness:
        return None
    (source_id, source), (sink_id, sink), params = witness
    kinds = {
        "raw": ("write", "read"),
        "war": ("read", "write"),
        "waw": ("write", "write"),
    }
    wanted = kinds.get(dep.kind)
    for n, (a, b, first, second) in enumerate(_classes(origin)):
        if (a.id, b.id) != (dep.source, dep.sink) or (first[0], second[0]) != wanted:
            continue
        if first[1] != dep.array or not structurally_equal(
            tuple(first[2]), tuple(dep.source_indices)
        ) or not structurally_equal(tuple(second[2]), tuple(dep.sink_indices)):
            continue
        part = next((p for p in parts if p.name == f"dependence_{n}"), None)
        if part is None:
            return None
        try:
            values = dict(params)
            for stmt, coords, tag in ((a, source, "a"), (b, sink, "b")):
                values |= _stages(origin, stmt, steps, coords, params, tag, part)
        except (Unstatable, isl.Error, KeyError, ValueError):
            return None
        text = (
            f"{source_id}{_shown(source)} and {sink_id}{_shown(sink)} at "
            + ", ".join(f"{k}={v}" for k, v in params.items())
        )
        return _counterexample(part, values, text)
    return None


def _shown(coords: Mapping[str, int]) -> str:
    return "[" + ", ".join(f"{k}={v}" for k, v in coords.items()) + "]"


def _stages(
    origin: Any,
    stmt: Stmt,
    steps: Sequence[Any],
    coords: Mapping[str, int],
    params: Mapping[str, int],
    tag: str,
    part: Any,
) -> dict[str, int]:
    """Every coordinate of one instance at every stage, by its name in ``part``.

    The names are the part's variables, which :func:`_chain` made as
    ``loop_tag`` and ``loop_tagk``; each stage's values are the image of the
    one before by the step's map, which isl computes at the point.
    """
    binders = {var.name for var, _ in part.term.binders}

    def named(loop: str, suffix: str) -> str:
        name = f"{loop}_{suffix}"
        while name not in binders and name + "_" in binders:
            name += "_"
        return name

    out: dict[str, int] = {}
    current = dict(coords)
    for loop, value in current.items():
        out[named(loop, tag)] = int(value)
    for k, step in enumerate(steps, start=1):
        after: dict[str, int] = {}
        mapping = step.isl_maps.get(stmt.id)
        if mapping is not None:
            inputs = mapping.get_var_names(isl.dim_type.in_)
            outputs = mapping.get_var_names(isl.dim_type.out)
            fixed = mapping
            names = fixed.get_var_names(isl.dim_type.param)
            for name, value in params.items():
                if name in names:
                    fixed = fixed.fix_val(
                        isl.dim_type.param,
                        names.index(name),
                        isl.Val.int_from_si(fixed.get_ctx(), int(value)),
                    )
            for position, name in enumerate(inputs):
                fixed = fixed.fix_val(
                    isl.dim_type.in_,
                    position,
                    isl.Val.int_from_si(fixed.get_ctx(), int(current[name])),
                )
            point = fixed.range().sample_point()
            for position, name in enumerate(outputs):
                value = point.get_coordinate_val(isl.dim_type.set, position)
                after[name] = value.to_python()
        for name in step.old[stmt.id]:
            if name not in after and name in current:
                after[name] = current[name]
        current = {name: after[name] for name in step.new[stmt.id]}
        for loop, value in current.items():
            out[named(loop, f"{tag}{k}")] = int(value)
    return out


# }}}


def prove(fact: Any) -> Any:
    """``fact`` as Lean leaves it, where a Lean oracle is registered and available.

    For a fact loopty's checker settled when it was made, a schedule's casts
    decided or refuted as each step is taken, which no check offers the
    oracles, so that a ledger of them (``loopty run``'s) shows what Lean
    made of them: ``proved`` by ``lean``, or the status it had with Lean's
    reason (a refuted one with whether Lean proved it false at isl's pair).
    Without Lean, the fact comes back as it is.
    """
    from lanky.ledger import Status
    from lanky.plugins import oracle_availability, registry

    for oracle in registry.sorted_oracles():
        if oracle.trust_class() != "kernel":
            continue
        available, _ = oracle_availability(oracle)
        if not available:
            continue
        try:
            if not oracle.can_establish(fact):
                continue
            result = oracle.establish(fact)
        except Exception:  # noqa: BLE001 - a failing oracle leaves the fact as it was
            continue
        if result is None:
            continue
        if result.status is Status.PROVED:
            return result.with_status(Status.PROVED, trust_class=oracle.trust_class())
        return result
    return fact
