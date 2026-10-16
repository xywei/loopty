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
from lanky.prelude import FinType, FnType, Int
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
from lanky.terms import Sum as LankySum

from loopty import flow
from loopty.description import (
    Description,
    Unstatable,
    conjunction,
    description_of,
    disjunction,
)
from loopty.hypotheses import _sort_terms, substitute
from loopty.term import Stmt, Term

__all__ = [
    "ISL_ONLY",
    "Claim",
    "IslOnly",
    "Stated",
    "argument_obligation",
    "defined_obligation",
    "conjunct_obligations",
    "discharged_obligation",
    "disjoint_writes_obligation",
    "distinct_arguments_obligation",
    "element_sort_obligation",
    "in_bounds_obligation",
    "mixed_in_bounds_obligation",
    "cast_obligation",
    "layout_bounds_obligation",
    "layout_injective_obligation",
    "layout_onto_obligation",
    "layout_same_obligation",
    "lexicographically_before",
    "merged_obligation",
    "row_starts_obligation",
    "window_obligation",
    "ordering_obligation",
    "prove",
    "shadow_hints",
    "single_assignment_obligation",
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
    if isinstance(expr, Forall | LankySum) or type(expr).__name__ == "Exists":
        # a binder's sort may name what is free, a size in Fin[n + 1] or in
        # Fn[Fin[n], Real], read with the binders before it bound; a sum
        # binds as a quantifier does
        inner = set(bound)
        for var, sort in expr.binders:
            _names(_sort_terms(sort), out, frozenset(inner))
            inner.add(var.name)
        _names(expr.body, out, frozenset(inner))
        if expr.guard is not None:
            _names(expr.guard, out, frozenset(inner))
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


def _array_sort(rank: int, element: Any = Int) -> Any:
    """``Int → Int → Int`` for a rank-two array: a cell is an integer atom.

    ``element`` is what a cell is, ``Int`` but for an array of reals in a
    claim about them, whose cells are ``Real`` (:func:`discharged_obligation`).
    """
    sort: Any = element
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
        sorts: Mapping[str, Any] | None = None,
    ) -> None:
        self.term = term
        #: The sort of a name that is not an integer: a scalar's, or the
        #: cells' of an array, ``Real``; every other name is an ``Int``.
        self.sorts = dict(sorts or {})
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
        binders = [(Var(name), self.sorts.get(name, Int)) for name in scalars]
        binders += [
            (Var(name), _array_sort(names[name], self.sorts.get(name, Int)))
            for name in arrays
        ]
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
    decider: str = "isl",
) -> Any:
    """The obligation of ``parts``; a refuted one is checked at its counterexample.

    A fact isl refuted is offered to Lean only as its counterexample, which
    Lean proves the claim false at. With none that can be stated, Lean is not
    asked at all: were the parts offered as a claim, a proof of them would
    read ``proved`` where isl refuted, and the refutation would be lost.

    ``decider`` names who decided the claim and found its hints, which the
    ledger's LEAN line counts the fact under; ``""`` for a claim nobody but
    Lean decides, a program's postcondition unfolded from what its calls
    store (:mod:`loopty.unfolding`).
    """
    if refuted and counterexample is None:
        return _declined(REFUTED_UNSTATED)
    lanky = _lanky_obligations()
    if "fallback" in getattr(lanky.Part, "__dataclass_fields__", {}):
        # what isl finds only when Lean asks: the case splits of shadow_hints
        from dataclasses import replace

        parts = [
            part
            if part.fallback is not None
            else replace(part, fallback=partial(shadow_hints, part.term))
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
        decider=decider,
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


# {{{ what a call of another kernel owes


def argument_obligation(
    term: Term, call: Any, value: Any, sort: Any, verdict: Any
) -> Any:
    """A scalar a call passes is a point of its sort, at every point of the call.

    One part: at every point of the call's domain (the caller's loops and the
    binders of the sums around the call, as loopty described them), ``0 <=
    v`` and, for ``Fin[m]``, ``v < m``, the value and the bound in the
    caller's names (:func:`loopty.typing.call_facts`).
    """
    try:
        domain = _described(call.domain, f"the points of the call {call.label}")
        claim = Claim(term, taken=frozenset(call.inames))
        coords = claim.coordinates(call.inames)
        claim.hypotheses.append(domain.predicate(coords))
        at = _renamed(value, call.inames, coords)
        parts = [Comparison(0, "<=", at)]
        if isinstance(sort, FinType):
            parts.append(Comparison(at, "<", _renamed(sort.bound, call.inames, coords)))
        label = f"the value {call.label} is given"
        part = _lanky_obligations().Part(
            "argument", label, claim.close(conjunction(parts))
        )
    except Unstatable as exc:
        return _declined(str(exc))
    counterexample = None
    if not verdict.ok and verdict.witness is not None:
        values = dict(verdict.parameters or {})
        values |= _point(call.inames, verdict.witness)
        counterexample = _counterexample(
            part, values, ", ".join(f"{k}={v}" for k, v in values.items())
        )
    return _obligation([part], counterexample, refuted=verdict.ok is False)


def distinct_arguments_obligation(
    term: Term, call: Any, pairs: Sequence[tuple[Any, Any]], verdict: Any
) -> Any:
    """Two rows of one array a call passes are two rows, at every point of it.

    One part: at every point of the call's domain, the indices the two rows
    fix differ on one of the axes both fix.
    """
    try:
        domain = _described(call.domain, f"the points of the call {call.label}")
        claim = Claim(term, taken=frozenset(call.inames))
        coords = claim.coordinates(call.inames)
        claim.hypotheses.append(domain.predicate(coords))
        first = [_renamed(p, call.inames, coords) for p, _ in pairs]
        second = [_renamed(q, call.inames, coords) for _, q in pairs]
        goal = LogicalNot(_equal(first, second))
        label = f"the rows {call.label} is given"
        part = _lanky_obligations().Part("distinct", label, claim.close(goal))
    except Unstatable as exc:
        return _declined(str(exc))
    counterexample = None
    if not verdict.ok and verdict.witness is not None:
        values = dict(verdict.parameters or {})
        values |= _point(call.inames, verdict.witness)
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


# {{{ inside one kernel, a cell stored once, and read after its store


def single_assignment_obligation(
    term: Term,
    array: str,
    writers: Sequence[Stmt],
    places: Sequence[tuple],
    verdict: Any,
) -> Any:
    """No cell of ``array`` is stored twice, or read before its store.

    The fact :mod:`loopty.dataflow` decides a postcondition under.

    One part per pair of statements that store into the array, a statement
    with itself among them: two instances, distinct where they are of one
    statement, store into cells that differ. And one part per read of the
    array and statement that stores into it: an instance that stores the
    cell an instance reads runs before it, in the source order, and so is
    not the instance that reads, which reads before it stores. A
    read whose index isl does not read exactly is about every cell, as its
    footprint is, so no instance may store into the array after it. A read
    made at a statement's outer loops alone, the length of a ragged row, is
    taken at every instance of the row, its inner loops free, as isl takes
    it.
    """
    lanky = _lanky_obligations()
    depth = flow.instance_space_depth(term)
    parts = []
    try:
        for k, first in enumerate(writers):
            for second in writers[k:]:
                claim = Claim(term)
                x = claim.coordinates(first.inames, "a")
                y = claim.coordinates(second.inames, "b")
                for stmt, coords in ((first, x), (second, y)):
                    domain = _described(stmt.domain, f"the loops of {stmt.id}")
                    claim.hypotheses.append(domain.predicate(coords))
                if second is first:
                    # a statement without loops runs once: no two of its
                    # instances differ, a hypothesis False as written
                    claim.hypotheses.append(LogicalNot(_equal(x, y)) if x else False)
                goal = LogicalNot(
                    _equal(
                        [_renamed(i, first.inames, x) for i in first.assignee.indices],
                        [
                            _renamed(i, second.inames, y)
                            for i in second.assignee.indices
                        ],
                    )
                )
                label = (
                    f"two instances of {first.id} storing into {array}"
                    if second is first
                    else f"{first.id} and {second.id} storing into {array}"
                )
                parts.append(
                    lanky.Part(f"stores_{len(parts)}", label, claim.close(goal))
                )
        for stmt, indices, inames, domain in places:
            for writer in writers:
                claim = Claim(term)
                x = claim.coordinates(inames, "r")
                inner = claim.coordinates(stmt.inames[len(inames) :], "i")
                y = claim.coordinates(writer.inames, "w")
                loops = ", ".join(inames)
                described = _described(domain, f"the loops {loops} of {stmt.id}")
                claim.hypotheses.append(described.predicate(x))
                written = _described(writer.domain, f"the loops of {writer.id}")
                claim.hypotheses.append(written.predicate(y))
                if all(_affine(index) for index in indices):
                    claim.hypotheses.append(
                        _equal(
                            [_renamed(i, inames, x) for i in indices],
                            [
                                _renamed(i, writer.inames, y)
                                for i in writer.assignee.indices
                            ],
                        )
                    )
                at = [*x, *inner][: len(stmt.inames)]
                # strictly before: the instance that reads a cell and stores
                # it reads it first, as isl's lex_le counts it
                # (loopty.dataflow); the positions both times hold as numbers
                # may settle it
                goal = lexicographically_before(
                    _source_time(writer, depth, y), _source_time(stmt, depth, at)
                )
                cell = f"{array}[{', '.join(render(i) for i in indices)}]"
                parts.append(
                    lanky.Part(
                        f"read_{len(parts)}",
                        f"{writer.id} storing the {cell} {stmt.id} reads, before it",
                        claim.close(goal),
                    )
                )
    except Unstatable as exc:
        return _declined(str(exc))
    if not parts:
        return _declined(f"no statement stores into {array}")
    return _obligation(parts, refuted=verdict.ok is False)


def conjunct_obligations(
    questions: Sequence[Any], labels: Sequence[str], verdict: Any
) -> Any:
    """A conjunction, each conjunct decided on its own: their parts, together.

    A kernel's postcondition decided under what its statements store
    (:mod:`loopty.dataflow`) is one question per conjunct, each stated as
    :func:`discharged_obligation` states it; a conjunct not stated for Lean
    leaves the claim unstated, with its reason.
    """
    from dataclasses import replace

    parts = []
    rests: list[str] = []
    for k, (question, label) in enumerate(zip(questions, labels, strict=True)):
        obligation = question.lean_obligation()
        if obligation is None or getattr(obligation, "declined", ""):
            reason = getattr(obligation, "declined", "") or "it is not stated for Lean"
            return _declined(f"the conjunct {label}: {reason}")
        for part in obligation.parts:
            name = f"conjunct_{k}"
            if len(obligation.parts) > 1:
                name = f"{name}_{part.name}"
            parts.append(replace(part, name=name, label=label))
        rests.extend(obligation.rests_on)
    return _obligation(
        parts, rests_on=tuple(dict.fromkeys(rests)), refuted=verdict.ok is False
    )


# }}}


# {{{ a program's facts, under what held where a call is made


#: The term classes core Lean has no form for: a sum, an absolute value, an
#: elementary function and true division are Mathlib's.
_UNPRINTABLE = ("Sum", "Abs", "Elementary", "Quotient")


def _statable(
    expr: Any, arrays: frozenset[str], names: frozenset[str], reals: bool = False
) -> bool:
    """Whether ``expr`` is integer arithmetic over ``names`` and cells of ``arrays``.

    A hypothesis about a real cell, read over the integers, would be another
    hypothesis, one that may give Lean what the reals do not; it is left
    out, which can only make the claim harder. With ``reals``, the cells of
    an array of reals, a real scalar, a float and true division are
    arithmetic too, which Mathlib's dialect states (:func:`discharged_obligation`);
    a sum of loopty's is not, whatever the dialect.
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
        if reals:
            if isinstance(node, complex) or type(node).__name__ in ("Sum", "Reduction"):
                return False
        elif isinstance(node, float | complex) or type(node).__name__ in _UNPRINTABLE:
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


#: The most instances of the hypotheses isl took that a part is given: each
#: is a ``have`` line, and the order of the rows a write keeps, by induction
#: over the run, takes four universals at the cells around the one written.
_MOST_INSTANCES = 32


def discharged_obligation(
    domain: Any,
    goal: Any,
    record: Sequence[tuple],
    integral: frozenset[str],
    known: frozenset[str],
    nonneg: frozenset[str],
    verdict: Any,
    sorts: tuple[tuple[str, Any], ...] = (),
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

    ``sorts`` names what is real (``Real``): arrays of reals and real
    scalars, which a claim about what a kernel stores may read
    (:mod:`loopty.dataflow`). A claim that reads one is a part about reals,
    printed in Mathlib's dialect and closed by ``linarith``
    (:attr:`lanky.obligations.Part.mathlib`), its arrays of reals functions
    into ``ℝ``; isl decided it with each equation of reals an atom, which
    is what ``linarith`` does with them too. A goal ``¬g ∨ b``, the claim
    of a universal with a guard, is stated with ``g`` among the hypotheses
    and ``b`` as the goal, which says the same and ``linarith`` can read.
    """
    lanky = _lanky_obligations()
    real = dict(sorts)
    reals = bool(real) and "mathlib" in getattr(lanky.Part, "__dataclass_fields__", {})
    try:
        described = _described(domain, "the claim's domain")
        dims = tuple(domain.get_var_names(isl.dim_type.set))
        claims = [entry[2] for entry in record if entry[0] == "claim"]
        names = frozenset(known) | frozenset(dims)
        arrays = frozenset(integral)
        if reals:
            names |= frozenset(name for name in real if name not in integral)
            arrays |= frozenset(real)
        if not _statable(goal, arrays, names, reals=reals):
            return _declined(
                "the claim reads what is not integer arithmetic over integer "
                "arrays, which is stated for isl only"
                if not real
                else "the claim reads what neither core Lean nor Mathlib's dialect "
                "states as loopty reads it, which is stated for isl only"
            )
        claim = Claim(
            None,
            sizes=frozenset(nonneg),
            taken=names,
            sorts=dict(real) if reals else None,
        )
        coords = claim.coordinates(dims)
        claim.hypotheses.append(described.predicate(coords))
        under, goal = _split_guard(goal)
        claim.hypotheses.extend(under)
        for hypothesis in claims:
            for part in conjuncts(hypothesis):
                if _statable(part, arrays, names, reals=reals):
                    claim.hypotheses.append(part)
        term = claim.close(goal)
    except Unstatable as exc:
        return _declined(str(exc))
    uses_reals = reals and _reads_any(term, frozenset(real))
    guard = (
        [render(part) for part in conjuncts(term.guard)]
        if term.guard is not None
        else []
    )
    instances = []
    seen: set[tuple[int, tuple[str, ...]]] = set()
    for entry in record:
        if entry[0] != "instance" or len(instances) >= _MOST_INSTANCES:
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
    extra = {"mathlib": True} if uses_reals else {}
    label = "the claim where the hypotheses hold"
    part = lanky.Part("claim", label, term, hints, **extra)
    return _obligation([part], refuted=verdict.ok is False)


def _split_guard(goal: Any) -> tuple[list[Any], Any]:
    """``¬g ∨ b`` as ``([g], b)``: a guarded claim's guard among the hypotheses.

    Any other goal is ``([], goal)``. The two say the same at every point.
    """
    if (
        isinstance(goal, LogicalOr)
        and len(goal.children) == 2
        and isinstance(goal.children[0], LogicalNot)
    ):
        return list(conjuncts(goal.children[0].child)), goal.children[1]
    return [], goal


def _reads_any(term: Any, names: frozenset[str]) -> bool:
    """Whether ``term`` mentions one of ``names``, bound or free."""
    found: dict[str, int] = {}
    try:
        _names(term, found)
    except Unstatable:
        return True
    binders = {var.name for var, _ in getattr(term, "binders", ())}
    return bool((set(found) | binders) & names)


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
    it again. Two updates of one array that commute, where they can update
    one cell, are no class: they need no order
    (:func:`loopty.schedule._updates`), and what they ask of a schedule
    instead is stated for isl only.
    """
    schedule = _schedule_module()
    term = origin._term
    updating = {(dep.source, dep.sink) for dep in origin._updates}
    out = []
    for a in term.stmts:
        for b in term.stmts:
            for first in schedule._accesses(a, term):
                for second in schedule._accesses(b, term):
                    if first[1] != second[1] or len(first[2]) != len(second[2]):
                        continue
                    if first[0] == "read" and second[0] == "read":
                        continue
                    if (a.id, b.id) in updating and schedule._updating(
                        a, b, first[1]
                    ):
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
    dependence joins two work items: one more part per dependence class, that
    the two instances run on one work item of every axis (see
    :func:`_work_item_goal`), and one per read a statement's sum makes of the
    array its own instruction writes, on an axis where the two run apart.

    ``bijective``: three parts per statement the last step moves, over the
    instances as the steps before it left them: every instance has an image
    (an existential, with isl's functional form of the map as the witness),
    one only, and no two instances share one.

    The classes are those of the term as the steps leave it
    (:func:`_cast_classes`): a statement a substitution or a window took out
    is in none; the reads a substitution computes again read what its
    producer read, at the instance that stored the cell; and a window's
    accesses, two that touch one slot, keep the order they ran in when it
    was taken.
    """
    origin = cast.schedule
    steps = tuple(cast.steps)
    try:
        if cast.kind == "bijective":
            parts = _bijective_parts(origin, steps)
        else:
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
    parts = []
    assumed = steps[-1].assumptions if steps else ()
    work = steps[-1].work if steps else None
    for found in _cast_classes(origin, steps):
        claim = Claim(term)
        claim.hypotheses.extend(assumed)
        a, b = found.a, found.b
        chain_a = _chain(claim, origin, a, steps, "a")
        chain_b = _chain(claim, origin, b, steps, "b")
        if found.kind == "window":
            first, second = found.accesses
            at_a = _footprint_place(
                claim, a, list(chain_a[0].values()), first.inames, first.domain, "c"
            )
            at_b = _footprint_place(
                claim, b, list(chain_b[0].values()), second.inames, second.domain, "d"
            )
            placement = found.window.placement
            claim.hypotheses.append(
                _equal(
                    placement.at(
                        [_renamed(i, first.inames, at_a) for i in first.indices]
                    ),
                    placement.at(
                        [_renamed(i, second.inames, at_b) for i in second.indices]
                    ),
                )
            )
            stage = len(found.window.steps)
            claim.hypotheses.append(
                lexicographically_before(
                    _stage_time(origin, steps, stage, a.id, chain_a[stage]),
                    _stage_time(origin, steps, stage, b.id, chain_b[stage]),
                )
            )
        elif found.kind == "carried":
            producer, at, stored_first = found.through
            p = claim.coordinates(producer.inames, "p")
            domain = _described(producer.domain, f"the loops of {producer.id}")
            claim.hypotheses.append(domain.predicate(p))
            at_p = dict(zip(producer.inames, p, strict=True))
            reader, chain_r = (a, chain_a) if stored_first else (b, chain_b)
            # the producer's instance stored the cell the reader reads
            claim.hypotheses.append(
                _equal(
                    [
                        _renamed(i, producer.inames, p)
                        for i in producer.assignee.indices
                    ],
                    [substitute(plain(i), dict(chain_r[0])) for i in at],
                )
            )
            claim.hypotheses.append(
                _same_cell(
                    origin,
                    a,
                    b,
                    found.first[2],
                    found.second[2],
                    chain_a[0],
                    chain_b[0],
                )
            )
            source = origin._origin_layout.coords
            writer, chain_x = (b, chain_b) if stored_first else (a, chain_a)
            then = _time(source, origin._origin_order, {}, producer.id, at_p)
            other = _time(source, origin._origin_order, {}, writer.id, chain_x[0])
            claim.hypotheses.append(
                lexicographically_before(then, other)
                if stored_first
                else lexicographically_before(other, then)
            )
            if a.id == b.id:
                names = list(chain_a[0])
                claim.hypotheses.append(
                    LogicalNot(
                        _equal(
                            [chain_a[0][name] for name in names],
                            [chain_b[0][name] for name in names],
                        )
                    )
                )
        else:
            claim.hypotheses.append(
                _same_cell(
                    origin,
                    a,
                    b,
                    found.first[2],
                    found.second[2],
                    chain_a[0],
                    chain_b[0],
                )
            )
            source = origin._origin_layout.coords
            claim.hypotheses.append(
                lexicographically_before(
                    _time(source, origin._origin_order, {}, a.id, chain_a[0]),
                    _time(source, origin._origin_order, {}, b.id, chain_b[0]),
                )
            )
        goal = lexicographically_before(
            _stage_time(origin, steps, len(steps), a.id, chain_a[-1]),
            _stage_time(origin, steps, len(steps), b.id, chain_b[-1]),
        )
        parts.append(lanky.Part(found.name, found.label, claim.close(goal)))
        if work is not None:
            together = _work_item_goal(
                claim,
                work,
                (a.id, found.first[3], chain_a[-1]),
                (b.id, found.second[3], chain_b[-1]),
            )
            parts.append(
                lanky.Part(
                    f"{found.name}_work_item",
                    f"{found.label}, on one work item",
                    claim.close(together),
                )
            )
    if work is not None:
        parts += _within_parts(origin, steps, work)
    if not parts:
        # no two accesses a dependence can join: there is nothing to reverse
        parts.append(
            lanky.Part(
                "no_dependence",
                f"{term.name} has no two accesses a dependence can join",
                Claim(term).close(True),
            )
        )
    return parts


def _work_item_goal(
    claim: Claim,
    work: Any,
    one: tuple[str, str, Mapping[str, Any]],
    other: tuple[str, str, Mapping[str, Any]],
) -> Any:
    """Two instances run on one work item of every hardware axis, as a proposition.

    ``one`` and ``other`` are each a statement, where in it the access is
    made (:func:`loopty.schedule._accesses`'s part) and the instance's loops
    after the last step. An instance's work item along an axis is its loop
    on the axis less where loopy starts that loop, as
    :func:`loopty.schedule._work_items` reads it: the start loopy's code
    generation gives (``0`` mostly), or a variable of its own where loopy
    gives none. A statement's instruction whose sum is on the axis, and no
    loop of it, runs on one work item of it, the same for every such
    statement, which is a variable of the claim's; and an access that runs
    on every work item of the axis, a statement with no loop on it, or the
    body of a sum on it, shares a work item with nothing, so the claim is
    that no two such instances touch one cell: ``False``.
    """
    starts: dict[str, Any] = {}
    results: dict[str, Any] = {}

    def start(loop: str) -> Any:
        if loop not in starts:
            known = work.starts.get(loop)
            starts[loop] = (
                claim.fresh(f"start_{loop}") if known is None else plain(known)
            )
        return starts[loop]

    def at(end: tuple[str, str, Mapping[str, Any]], axis: str) -> Any:
        stmt_id, part, values = end
        loop = work.loops[stmt_id].get(axis)
        if loop is not None:
            begins = start(loop)
            if isinstance(begins, int) and not isinstance(begins, bool) and not begins:
                # loopy counts the work items from 0, as it mostly does
                return values[loop]
            return values[loop] - begins
        if part == "" and axis in work.summed[stmt_id]:
            if axis not in results:
                name = "".join(c if c.isalnum() else "_" for c in axis)
                results[axis] = claim.fresh(f"result_{name}")
            return results[axis]
        return None

    equal = []
    for axis in work.axes:
        first, second = at(one, axis), at(other, axis)
        if first is None or second is None:
            return False
        equal.append(Comparison(first, "==", second))
    return conjunction(equal)


def _within_parts(origin: Any, steps: Sequence[Any], work: Any) -> list[Any]:
    """A statement's sum reading a cell its own instruction writes, apart.

    One instance makes both accesses, so the pair is no dependence class, and
    needs no order; on a device the two can run on two work items, the body
    of a sum on a local axis on every one of them and the instruction on one
    (:func:`loopty.schedule._within_instances`). Where the statement has no
    loop on an axis, the two are apart there, and the claim is that they
    never touch one cell. After a substitution a reader's sum reads what the
    producer read, at the cell the producer stored, as
    :meth:`loopty.schedule.Schedule._carried_over` reads it, and those reads
    are pairs too. A read's subscripts can name the sum's own loops, which
    are not stated, so the cell is compared on the other axes alone, the
    wider claim.
    """
    lanky = _lanky_obligations()
    schedule = _schedule_module()
    term = origin._term
    gone: set[str] = set()
    for step in steps:
        gone |= set(step.gone)
    # (statement, the cell its sum reads, its instruction's write)
    pairs: list[tuple[Stmt, tuple, tuple]] = []
    for stmt in term.stmts:
        if stmt.id in gone:
            continue
        accesses = schedule._accesses(stmt, term)
        for read in accesses:
            if read[0] != "read" or read[3] != "sum":
                continue
            for write in accesses:
                if write[0] == "write" and write[3] == "" and write[1] == read[1]:
                    pairs.append((stmt, tuple(read[2]), write))
    for step in steps:
        if step.substituted is None:
            continue
        array, producer, readers = step.substituted
        stored = producer.assignee.indices
        reads = [
            (name, indices)
            for kind, name, indices, _part in schedule._accesses(producer, term)
            if kind == "read"
        ]
        for reader in readers:
            if reader.id in gone:
                continue
            accesses = schedule._accesses(reader, term)
            for kind, name, at, part in accesses:
                if kind != "read" or name != array or part != "sum":
                    continue
                renaming = {
                    index.name: plain(value)
                    for index, value in zip(stored, at, strict=True)
                }
                for read, indices in reads:
                    moved = tuple(substitute(plain(i), renaming) for i in indices)
                    for write in accesses:
                        if write[0] == "write" and write[3] == "" and write[1] == read:
                            pairs.append((reader, moved, write))
    assumed = steps[-1].assumptions
    parts = []
    for count, (stmt, cell, write) in enumerate(
        pair for pair in pairs if len(pair[1]) == len(pair[2][2])
    ):
        loops = work.loops[stmt.id]
        apart = [axis for axis in work.axes if loops.get(axis) is None]
        if not apart:
            continue
        claim = Claim(term)
        claim.hypotheses.extend(assumed)
        chain = _chain(claim, origin, stmt, steps, "a")
        same = _same_cell(origin, stmt, stmt, cell, write[2], chain[0], chain[0])
        claim.hypotheses.append(same)
        written = f"{write[1]}[{', '.join(render(i) for i in write[2])}]"
        parts.append(
            lanky.Part(
                f"within_{count}",
                f"the sum of {stmt.id} reads {written}, which its instruction "
                f"writes, on another work item of {', '.join(apart)}",
                claim.close(False),
            )
        )
    return parts


def plain(expr: Any) -> Any:
    """``expr`` with the casts of the trace taken off, as the checker reads it."""
    return _schedule_module()._plain(expr)


def _stage_time(
    origin: Any, steps: Sequence[Any], stage: int, stmt_id: str, values: Any
) -> list[Any]:
    """A statement's time vector after the first ``stage`` steps."""
    if stage == 0:
        return _time(
            origin._origin_layout.coords, origin._origin_order, {}, stmt_id, values
        )
    step = steps[stage - 1]
    return _time(step.new, step.order, step.tags, stmt_id, values)


@dataclass(frozen=True)
class _Class:
    """One dependence class of a cast, as the steps before it leave the term.

    ``kind`` is ``"source"`` for two accesses of the term, in the source
    order; ``"window"`` for two accesses of one slot of ``window``, in the
    order the window was taken in (``accesses`` are the two, as
    :func:`loopty.window.accesses` lists them); and ``"carried"`` for a read
    a substitution computes again and a write of what it reads, in the
    order of the write and the producer's instance that stored the cell
    (``through``: the producer, the reader's index into the array, and
    whether the reader is first, the class of a read before the write).
    """

    name: str
    label: str
    a: Stmt
    b: Stmt
    first: tuple
    second: tuple
    kind: str = "source"
    window: Any = None
    accesses: tuple = ()
    through: Any = None


def _cast_classes(origin: Any, steps: Sequence[Any]) -> list[_Class]:
    """Every dependence class a cast after ``steps`` is checked on.

    As :func:`_classes` lists them for the term, without a statement a step
    took out (:attr:`loopty.schedule.Step.gone`); with the reads a
    substitution computes again, two classes per write of what its producer
    read (the read before it, and after it); and with the pairs of accesses
    of a window, at least one a write, that touch one slot. isl checks a
    cast against the same dependences (:meth:`loopty.schedule.Schedule.
    substitute`, :meth:`loopty.schedule.Schedule.contract`).
    """
    from loopty import window as win

    schedule = _schedule_module()
    term = origin._term
    gone: set[str] = set()
    for step in steps:
        gone |= set(step.gone)
    out: list[_Class] = []
    for n, (a, b, first, second) in enumerate(_classes(origin)):
        if a.id in gone or b.id in gone:
            continue
        out.append(
            _Class(
                f"dependence_{n}",
                _class_label(a, b, first, second),
                a,
                b,
                first,
                second,
            )
        )
    count = 0
    for step in steps:
        if step.substituted is None:
            continue
        array, producer, readers = step.substituted
        stored = producer.assignee.indices
        # each read once: u[j] * u[j] reads one cell
        reads = list(
            {
                (access[1], tuple(render(i) for i in access[2])): access
                for access in schedule._accesses(producer, term)
                if access[0] == "read"
            }.values()
        )
        writes = [
            (stmt, access)
            for stmt in term.stmts
            if stmt.id not in gone
            for access in schedule._accesses(stmt, term)
            if access[0] == "write"
        ]
        for reader in readers:
            if reader.id in gone:
                continue
            for kind, name, at, part in schedule._accesses(reader, term):
                if kind != "read" or name != array:
                    continue
                renaming = {
                    index.name: plain(value)
                    for index, value in zip(stored, at, strict=True)
                }
                for _kind, read, indices, _part in reads:
                    moved = tuple(substitute(plain(i), renaming) for i in indices)
                    mine = ("read", read, moved, part)
                    for writer, access in writes:
                        if access[1] != read or len(access[2]) != len(moved):
                            continue
                        for stored_first in (True, False):
                            pair = (
                                (reader, writer, mine, access)
                                if stored_first
                                else (writer, reader, access, mine)
                            )
                            a, b, first, second = pair
                            out.append(
                                _Class(
                                    f"carried_{count}",
                                    _class_label(a, b, first, second)
                                    + f", through {producer.id}",
                                    a,
                                    b,
                                    first,
                                    second,
                                    kind="carried",
                                    through=(producer, tuple(at), stored_first),
                                )
                            )
                            count += 1
    count = 0
    for step in steps:
        window = step.window
        if window is None:
            continue
        found = win.window_accesses(term, window)
        for one in found:
            for other in found:
                if one.kind == "read" and other.kind == "read":
                    continue
                first = (one.kind, window.array, one.indices, one.part)
                second = (other.kind, window.array, other.indices, other.part)
                out.append(
                    _Class(
                        f"window_{count}",
                        _class_label(one.stmt, other.stmt, first, second)
                        + f", one slot of {window.placement}",
                        one.stmt,
                        other.stmt,
                        first,
                        second,
                        kind="window",
                        window=window,
                        accesses=(one, other),
                    )
                )
                count += 1
    return out


def _bijective_parts(origin: Any, steps: Sequence[Any]) -> list[Any]:
    lanky = _lanky_obligations()
    term = origin._term
    step = steps[-1]
    earlier = steps[:-1]
    assumed = step.assumptions
    gone: set[str] = set()
    for earlier_step in steps:
        gone |= set(earlier_step.gone)
    parts = []
    for stmt in term.stmts:
        if stmt.id not in step.isl_maps or stmt.id in gone:
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
    for found in _cast_classes(origin, steps):
        a, b, first, second = found.a, found.b, found.first, found.second
        if (found.kind == "window") != (getattr(dep, "window", None) is not None):
            continue
        if (a.id, b.id) != (dep.source, dep.sink) or (first[0], second[0]) != wanted:
            continue
        if first[1] != dep.array or not structurally_equal(
            tuple(first[2]), tuple(dep.source_indices)
        ) or not structurally_equal(tuple(second[2]), tuple(dep.sink_indices)):
            continue
        if found.kind == "carried" or (
            found.kind == "window" and "sum" in (first[3], second[3])
        ):
            # the producer's instance, a sum's binders: no point isl gave
            return None
        part = next((p for p in parts if p.name == found.name), None)
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


# {{{ a layout a kernel rewrites, kept by every write


def _lean_name(text: str) -> str:
    """``scan_flat_2_S0``: an id as a part's name, which Lean reads."""
    import re

    return re.sub(r"\W", "_", text)


def row_starts_obligation(
    term: Term,
    offsets: str,
    counts: str,
    rows: Any,
    stmts: Sequence[Stmt],
    verdict: Any,
) -> Any:
    """A layout fact decided by the row starts, stated for Lean: the induction step.

    The rows of a family read through ``offsets`` stay where the contract
    found them while every write leaves the offsets as they were (see
    :func:`loopty.typing.layout_facts`). One part per statement that
    writes them: where the offsets start at 0 and step by the counts, which
    the contract checks when the call starts (``off[0] = 0`` and ``off[q] =
    off[q - 1] + cnt[q - 1]`` for ``1 <= q <= n``), the value each instance
    stores is the value its cell already holds. That is the induction step
    over the run; its base case is the contract's check, which every fact on
    the layout rests on as the run does. The universal hypothesis is taken
    at the cell written, a hint, with the equations between the cells it
    reads and the claim's that lanky adds.
    """
    from lanky.terms import Subscript

    lanky = _lanky_obligations()
    parts = []
    try:
        for stmt in stmts:
            claim = Claim(term)
            coords = claim.coordinates(stmt.inames)
            domain = _described(stmt.domain, f"the loops of {stmt.id}")
            claim.hypotheses.append(domain.predicate(coords))
            q = claim.fresh("q")
            off, cnt = Var(offsets), Var(counts)
            start = Comparison(Subscript(off, 0), "==", 0)
            step = Forall(
                ((q, Int),),
                Comparison(
                    Subscript(off, q),
                    "==",
                    Subscript(off, q - 1) + Subscript(cnt, q - 1),
                ),
                LogicalAnd((Comparison(1, "<=", q), Comparison(q, "<=", rows))),
            )
            claim.hypotheses += [start, step]
            (cell,) = stmt.assignee.indices
            at = _renamed(plain(cell), stmt.inames, coords)
            value = _renamed(plain(stmt.expr), stmt.inames, coords)
            goal = Comparison(value, "==", Subscript(off, at))
            closed = claim.close(goal)
            guard = [render(part) for part in conjuncts(closed.guard)]
            hints = lanky.Hints(
                instances=(lanky.Instance(guard.index(render(step)), (at,)),)
            )
            label = (
                f"{stmt.id} stores into {offsets}[{render(cell)}] the value it "
                f"holds, where {offsets} starts at 0 and steps by {counts}"
            )
            parts.append(
                lanky.Part(f"{_lean_name(stmt.id)}_keeps", label, closed, hints)
            )
    except (Unstatable, ValueError) as exc:
        return _declined(str(exc))
    return _obligation(parts, refuted=verdict.ok is False)


def merged_obligation(questions: Sequence[tuple[str, Any]], verdict: Any) -> Any:
    """One obligation of several questions' parts, each named by its statement.

    A layout fact decided by induction over the run is a question per
    statement that writes the layout (:func:`loopty.typing._layout_by_induction`),
    each stated as the claim isl decided under the order before the write
    (:func:`discharged_obligation`); the fact is all of them.
    """
    lanky = _lanky_obligations()
    parts = []
    for name, question in questions:
        obligation = question.lean_obligation()
        if obligation.declined:
            return _declined(f"{name}: {obligation.declined}")
        for part in obligation.parts:
            parts.append(
                lanky.Part(
                    f"{_lean_name(name)}_{part.name}",
                    f"{name}: {part.label}",
                    part.term,
                    part.hints,
                )
            )
    return _obligation(parts, refuted=verdict.ok is False)


# }}}


# {{{ a declared layout: in bounds, one for one, onto, the same as another


def _layout_claim(layout: Any, *others: Any) -> Claim:
    """A claim over a layout's cells: its sizes are not negative."""
    sizes: set[str] = set()
    for one in (layout, *others):
        sizes |= set(one.size_names())
    return Claim(None, sizes=frozenset(sizes))


def _piece_place(claim: Claim, piece: Any, tag: str = "") -> list[Any]:
    """Coordinates for a cell of ``piece``, in its cells: a hypothesis."""
    coords = claim.coordinates(piece.names, tag)
    cells = _described(piece.cells, f"the cells of {piece.label}")
    claim.hypotheses.append(cells.predicate(coords))
    return coords


def _address(piece: Any, coords: Sequence[Any], written: bool = False) -> Any:
    """The address ``piece`` stores the cell ``coords`` at.

    ``written``: as the layout's function writes it, where Python folded the
    module constants it reads, ``3**3 * e`` for ``27*e``, and the strides of
    loopy's dim_tags as the products of the extents nested inside, ``3 * 3 *
    3`` (:attr:`loopty.layout.Piece.written`), so that Lean does the
    arithmetic (#275).
    """
    address = piece.address
    if written and getattr(piece, "written", None) is not None:
        address = piece.written
    return _renamed(plain(address), piece.names, coords)


def _in_buffer(layout: Any, address: Any) -> Any:
    """``0 <= address < k``: an address of the layout's buffer, ``Fin[k]``.

    As isl was given it for a layout isl states, and as the annotation writes
    it for one at every size, ``Fin[ne * n**3]``, whose bound isl cannot read.
    """
    if not layout.polynomial:
        buffer = _described(layout.addresses(), "the addresses of the buffer")
        return buffer.predicate([address])
    return conjunction(
        [Comparison(0, "<=", address), Comparison(address, "<", plain(layout.buffer))]
    )


# {{{ a layout at every size: the steps omega lacks

# A layout whose address multiplies an index by a size, ``n**3 * e + (i + n * j
# + n**2 * k)``, is one omega reads with each product an atom it knows nothing
# of. Where it is a mixed radix (:meth:`loopty.layout.Layout.radix`), each digit
# times the product of the extents nested inside it, the facts follow by a few
# steps on the digits, which Lean is given (:mod:`lanky.obligations`):
#
# - every stride and every digit's term is not negative, and a digit's term is
#   at most the next stride less its own, ``n * (n - 1 - j)``: in bounds;
# - two cells at one address agree on the first digit, from the outermost, on
#   which they would differ, since a larger digit there outweighs every digit
#   inside it: one for one, by a case split from the outermost digit;
# - an address's digits are its remainders by the extents, from the innermost,
#   ``c % n``, ``c // n % n``: onto, with each quotient's equation multiplied
#   by its stride, and the case of an extent that is zero, where the buffer is
#   empty.
#
# The arithmetic is multiplied out before ``omega`` reads it, so that each
# product of unknowns is one atom however it is written.


def _is_one(term: Any) -> bool:
    return isinstance(term, int) and not isinstance(term, bool) and term == 1


def _multiplies() -> bool:
    """Whether lanky takes the hints of arithmetic where unknowns multiply.

    A lanky before them gets none, and Lean, given no steps, leaves a layout
    at every size where isl did, ``tested``.
    """
    lanky = _lanky_obligations()
    fields = getattr(lanky.Hints, "__dataclass_fields__", {})
    return hasattr(lanky, "Product") and "ring" in fields


def _ring() -> Any:
    """The hint that multiplies the arithmetic out, where lanky takes it."""
    lanky = _lanky_obligations()
    return lanky.Hints(ring=True) if _multiplies() else lanky.Hints()


def _product_of(factors: Sequence[Any]) -> Any:
    """``a * b * c``, nested from the left, as a stride is."""
    out = factors[0]
    for factor in factors[1:]:
        out = prim.Product((out, factor))
    return out


def _stride_products(levels: Sequence[Any]) -> list[Any]:
    """Every stride that is a product of extents is not negative: ``0 ≤ n * n``."""
    lanky = _lanky_obligations()
    out = []
    for before, level in zip(levels, levels[1:]):
        if not _is_one(before.stride):
            out.append(lanky.Product(before.stride, before.extent))
    return out


def _digit_products(levels: Sequence[Any], cell: Mapping[str, Any]) -> list[Any]:
    """Each digit's term between ``0`` and the next stride less its own."""
    lanky = _lanky_obligations()
    out = []
    for level in levels:
        if _is_one(level.stride):
            continue
        digit = cell[level.name]
        out.append(lanky.Product(level.stride, digit))
        out.append(lanky.Product(level.stride, level.extent - 1 - digit))
    return out


def _radix_bounds_hints(layout: Any, coords: Sequence[Any]) -> Any:
    """The steps of a mixed radix's in-bounds part (see above)."""
    lanky = _lanky_obligations()
    levels = layout.radix()
    if levels is None or not _multiplies():
        return _ring()
    cell = dict(zip(layout.pieces[0].names, coords, strict=True))
    return lanky.Hints(
        products=(*_stride_products(levels), *_digit_products(levels, cell)),
        ring=True,
    )


def _radix_injective_hints(
    layout: Any, one: Sequence[Any], other: Sequence[Any]
) -> Any:
    """The steps of a mixed radix's one-for-one part, split from the outermost digit."""
    lanky = _lanky_obligations()
    levels = layout.radix()
    if levels is None or not _multiplies():
        return _ring()
    names = layout.pieces[0].names
    c = dict(zip(names, one, strict=True))
    d = dict(zip(names, other, strict=True))
    products = [
        *_stride_products(levels),
        *_digit_products(levels, c),
        *_digit_products(levels, d),
    ]
    cases: list[Any] = []
    equal: list[Any] = []
    for level in reversed(levels[1:]):
        x, y = c[level.name], d[level.name]
        cases.append(conjunction([*equal, Comparison(x, "<", y)]))
        cases.append(conjunction([*equal, Comparison(y, "<", x)]))
        equal.append(Comparison(x, "==", y))
        stride = level.stride
        products += [
            lanky.Product(stride, y - x - 1, in_cases=True),
            lanky.Product(stride, x - y - 1, in_cases=True),
            lanky.Product(stride, x - y, in_cases=True),
            lanky.Product(stride, y - x, in_cases=True),
        ]
    splits = ()
    if cases:
        cases.append(conjunction(equal))
        splits = (disjunction(cases),)
    return lanky.Hints(products=tuple(products), splits=splits, ring=True)


def _radix_onto_hints(layout: Any, address: Any) -> Any:
    """The steps of a mixed radix's onto part: the digits by division, the witness.

    ``q0 = c``, ``q[d + 1] = q[d] // X[d]``; the digit of level ``d`` is
    ``q[d] % X[d]``, and the outermost's is its quotient. Lean is told what it
    knows of each quotient by an extent that is no number, each equation
    ``q[d] = q[d] % X[d] + q[d] // X[d] * X[d]`` multiplied by the stride of
    level ``d``, each digit's term not negative, and, in the case of an extent
    that is zero, the buffer not positive: there is no address to find a cell
    for. A case split on the outermost digit leaves the address below the
    buffer only where it is inside its extent.
    """
    lanky = _lanky_obligations()
    levels = layout.radix()
    if levels is None or not _multiplies():
        return _ring()
    quotients: list[Any] = [address]
    for level in levels[:-1]:
        quotients.append(prim.FloorDiv(quotients[-1], level.extent))
    digits = {
        level.name: prim.Remainder(quotients[d], level.extent)
        for d, level in enumerate(levels[:-1])
    }
    top = levels[-1]
    digits[top.name] = quotients[-1]
    witness = tuple(digits[name] for name in layout.pieces[0].names)

    def literal(term: Any) -> bool:
        return isinstance(term, int) and not isinstance(term, bool) and term > 0

    divisors: list[Any] = []
    for level in levels[:-1]:
        if not literal(level.extent) and all(
            render(level.extent) != render(other) for other in divisors
        ):
            divisors.append(level.extent)
    extents = [level.extent for level in levels]
    everywhere = list(_stride_products(levels))
    in_cases = []
    splits = [
        LogicalOr((Comparison(divisor, "<=", 0), Comparison(divisor, ">", 0)))
        for divisor in divisors
    ]
    for divisor in divisors:
        # an extent of zero leaves the buffer empty: its product is not positive
        rest = list(extents)
        rest.pop(next(k for k, e in enumerate(rest) if render(e) == render(divisor)))
        if rest:
            for k in range(1, len(rest)):
                everywhere.append(lanky.Product(_product_of(rest[:k]), rest[k]))
            in_cases.append(
                lanky.Product(0 - divisor, _product_of(rest), in_cases=True)
            )
    splits.append(
        LogicalOr(
            (
                Comparison(quotients[-1], "<", top.extent),
                Comparison(top.extent, "<=", quotients[-1]),
            )
        )
    )
    for d, level in enumerate(levels[:-1]):
        if d == 0:
            continue
        stride = level.stride
        q, nxt = quotients[d], quotients[d + 1]
        written = prim.Remainder(q, level.extent) + nxt * level.extent
        in_cases += [
            lanky.Product(stride, q - written, in_cases=True),
            lanky.Product(stride, written - q, in_cases=True),
            lanky.Product(stride, digits[level.name], in_cases=True),
        ]
    if not _is_one(top.stride):
        in_cases += [
            lanky.Product(top.stride, quotients[-1], in_cases=True),
            lanky.Product(top.stride, quotients[-1] - top.extent, in_cases=True),
        ]
    return lanky.Hints(
        witness=(lanky.Piece(witness),),
        splits=tuple(splits),
        products=(*everywhere, *in_cases),
        quotients=tuple(
            lanky.Quotient(quotients[d], level.extent)
            for d, level in enumerate(levels[:-1])
        ),
        ring=True,
    )


# }}}


def layout_bounds_obligation(layout: Any, verdict: Any) -> Any:
    """Every cell of a layout is at an address of its buffer: a part per piece."""
    lanky = _lanky_obligations()
    parts = []
    try:
        for k, piece in enumerate(layout.pieces):
            claim = _layout_claim(layout)
            coords = _piece_place(claim, piece)
            goal = _in_buffer(layout, _address(piece, coords))
            hints = (
                _radix_bounds_hints(layout, coords)
                if layout.polynomial
                else lanky.Hints()
            )
            parts.append(
                lanky.Part(
                    f"in_bounds_{k}",
                    f"the address of a cell of {piece.label}",
                    claim.close(goal),
                    hints,
                )
            )
    except Unstatable as exc:
        return _declined(str(exc))
    return _obligation(parts, refuted=verdict.ok is False)


def layout_injective_obligation(layout: Any, verdict: Any) -> Any:
    """No two cells of a layout at one address: a part per pair of pieces.

    Two cells, one of each piece (a piece with itself too), stored at one
    address, are one cell. A refuted one is stated with isl's pair, when
    the layout has one piece.
    """
    lanky = _lanky_obligations()
    parts = []
    try:
        pieces = layout.pieces
        for i, first in enumerate(pieces):
            for j in range(i, len(pieces)):
                second = pieces[j]
                claim = _layout_claim(layout)
                one = _piece_place(claim, first, "c")
                other = _piece_place(claim, second, "d")
                claim.hypotheses.append(
                    Comparison(_address(first, one), "==", _address(second, other))
                )
                hints = (
                    _radix_injective_hints(layout, one, other)
                    if layout.polynomial
                    else lanky.Hints()
                )
                parts.append(
                    lanky.Part(
                        f"one_for_one_{i}_{j}",
                        f"a cell of {first.label} and one of {second.label} at one "
                        "address",
                        claim.close(_equal(one, other)),
                        hints,
                    )
                )
    except Unstatable as exc:
        return _declined(str(exc))
    counterexample = None
    if not verdict.ok and verdict.witness and len(layout.pieces) == 1:
        rank = layout.rank
        part = parts[0]
        names = [var.name for var, _ in part.term.binders]
        coords = [name for name in names if name.endswith(("_c", "_d"))]
        values = dict(verdict.parameters or {})
        values |= _point(coords, list(verdict.witness[: 2 * rank]))
        counterexample = _counterexample(
            part, values, ", ".join(f"{k}={v}" for k, v in values.items())
        )
    return _obligation(parts, counterexample, refuted=verdict.ok is False)


def layout_onto_obligation(layout: Any, verdict: Any) -> Any:
    """Every address of a layout's buffer stores a cell: a ∀∃, witnessed.

    One part: at every address, some piece has a cell stored there, a
    disjunction of existentials, one per piece. The witness is isl's
    inverse of the layout, piece by piece, as ``lexmin`` gives it: the
    region of addresses each piece stores at, after the ones before it, and
    the cell there, with ``//`` and ``%`` by numbers. A layout at every size
    has no inverse isl can compute; where it is a mixed radix, the witness
    is its digits, ``c % n`` and ``c // n``, the inverse isl gives at a
    number (see above).
    """
    from lanky.terms import Exists

    lanky = _lanky_obligations()
    try:
        claim = _layout_claim(layout)
        (x,) = claim.coordinates(("address",))
        claim.hypotheses.append(_in_buffer(layout, x))
        options = []
        for piece in layout.pieces:
            ys = claim.fresh_all(piece.names, "cell")
            values = [ys[name] for name in piece.names]
            cells = _described(piece.cells, f"the cells of {piece.label}")
            body = conjunction(
                [
                    cells.predicate(values),
                    Comparison(_address(piece, values), "==", x),
                ]
            )
            options.append(
                Exists(tuple((ys[name], Int) for name in piece.names), body)
            )
        goal = disjunction(options) if len(options) > 1 else options[0]
        if layout.polynomial:
            hints = _radix_onto_hints(layout, x)
        else:
            hints = _isl_inverse(layout, x)
        part = lanky.Part(
            "onto",
            f"a cell at every address of Fin({layout.buffer})",
            claim.close(goal),
            hints,
        )
    except (Unstatable, isl.Error) as exc:
        return _declined(str(exc))
    return _obligation([part], refuted=verdict.ok is False)


def _isl_inverse(layout: Any, x: Any) -> Any:
    """isl's inverse of a layout, piece by piece, as the witness of its onto part."""
    lanky = _lanky_obligations()
    buffer = layout.addresses()
    pieces: list[Any] = []
    taken = None
    multiple = len(layout.pieces) > 1
    for index, piece in enumerate(layout.pieces):
        inverse = piece.relation().reverse()
        inverse, space = inverse.align_params(buffer.get_space()), buffer
        space = space.align_params(inverse.get_space())
        inverse = inverse.intersect_domain(space)
        region = inverse.domain()
        if taken is not None:
            region, earlier = _both(region, taken)
            region = region.subtract(earlier)
        if region.is_empty():
            continue
        params = {
            name: Var(name) for name in inverse.get_var_names(isl.dim_type.param)
        }
        best = isl.PwMultiAff.from_map(inverse.intersect_domain(region).lexmin())
        found: list[tuple[Any, Any]] = []
        best.foreach_piece(lambda where, value: found.append((where, value)))
        for where, value in found:
            pieces.append(
                lanky.Piece(
                    tuple(
                        _aff_term(value.get_at(k), [x], params)
                        for k in range(value.dim(isl.dim_type.out))
                    ),
                    _set_term(where, [x], params),
                    index if multiple else None,
                )
            )
        taken = region if taken is None else _both(taken, region)[0].union(
            _both(region, taken)[0]
        )
    if pieces:
        last = pieces[-1]
        pieces[-1] = lanky.Piece(last.values, None, last.disjunct)
    return lanky.Hints(witness=tuple(pieces)) if pieces else lanky.Hints()


def layout_same_obligation(first: Any, second: Any, verdict: Any) -> Any:
    """Two layouts store every cell at one address: a part per pair of pieces.

    A cell of a piece of each is stored at one address by both; and every
    cell of each is a cell of the other, a part per piece, when their pieces
    are not the same cells. A refuted one is stated with isl's cell when both
    have one piece. Where a size multiplies an index in either, the first
    claim is a polynomial identity, which Lean multiplies out. Each address is
    as the layout writes it, its module constants and dim_tags' strides not
    folded, so that ``27 * e + a + 3 * b + 9 * c`` on both sides, which
    Python computed, is ``3 ^ 3 * e + a + 3 * b + 3 ^ 2 * c = 3 * 3 * 3 * e
    + a + 3 * b + 3 * 3 * c``, which Lean does (#275).
    """
    lanky = _lanky_obligations()
    parts = []
    ring = _ring() if first.polynomial or second.polynomial else None
    try:
        for i, one in enumerate(first.pieces):
            for j, other in enumerate(second.pieces):
                claim = _layout_claim(first, second)
                coords = _piece_place(claim, one)
                cells = _described(other.cells, f"the cells of {other.label}")
                claim.hypotheses.append(cells.predicate(coords))
                goal = Comparison(
                    _address(one, coords, written=True),
                    "==",
                    _address(other, coords, written=True),
                )
                parts.append(
                    lanky.Part(
                        f"same_{i}_{j}",
                        f"a cell of {one.label} and of {other.label}",
                        claim.close(goal),
                        ring or lanky.Hints(),
                    )
                )
        for mine, theirs, which in ((first, second, "a"), (second, first, "b")):
            for k, piece in enumerate(mine.pieces):
                if all(p.cells is piece.cells for p in theirs.pieces):
                    continue
                claim = _layout_claim(first, second)
                coords = _piece_place(claim, piece)
                goal = disjunction(
                    [
                        _described(p.cells, f"the cells of {p.label}").predicate(coords)
                        for p in theirs.pieces
                    ]
                )
                parts.append(
                    lanky.Part(
                        f"cells_{which}_{k}",
                        f"a cell of {piece.label} is a cell of {theirs.name}",
                        claim.close(goal),
                    )
                )
    except Unstatable as exc:
        return _declined(str(exc))
    counterexample = None
    if (
        not verdict.ok
        and verdict.witness
        and len(first.pieces) == 1
        and len(second.pieces) == 1
    ):
        part = parts[0]
        values = dict(verdict.parameters or {})
        values |= _point(first.pieces[0].names, list(verdict.witness[: first.rank]))
        counterexample = _counterexample(
            part, values, ", ".join(f"{k}={v}" for k, v in values.items())
        )
    return _obligation(parts, counterexample, refuted=verdict.ok is False)


# }}}


# {{{ a window: no read sees its slot overwritten


def window_obligation(window: Any) -> Any:
    """The ``no-clobber`` fact of a :class:`loopty.schedule.Window`, for Lean.

    One part per read of the array, and per two of its producers where
    several store it. Three instances, each in its domain and through every
    step taken before the window: a read ``r``, a producer's instance ``w``
    that stored the cell it reads, and an instance ``v`` of a producer that
    stored a different cell into the same slot, after ``w``. Then ``v`` does
    not run before ``r``, in the order the steps leave: their time vectors
    do not compare that way. The slot is the placement, a term with ``%``
    and ``//`` by numbers, which ``omega`` reads; the time vectors are those
    the checker's order has (:func:`_time`). A refused window is stated with
    isl's triple, at which Lean proves the part false.

    ``w`` is any instance that stored the cell, not only the last before
    the read, which claims more than the window needs where a cell is stored
    twice, and is what isl decided (:func:`loopty.window.clobbers`).
    """
    from loopty import window as win

    lanky = _lanky_obligations()
    origin = window.schedule
    steps = tuple(window.steps)
    term = origin._term
    producers = tuple(window.producers)
    placement = window.placement
    assumed = steps[-1].assumptions if steps else ()
    found = [
        access
        for access in win.accesses(term, window.array, window.readers)
        if access.kind == "read"
    ]
    parts = []
    try:
        for k, access in enumerate(found):
            for p, first in enumerate(producers):
                for q, second in enumerate(producers):
                    claim = Claim(term)
                    claim.hypotheses.extend(assumed)
                    chain_r = _chain(claim, origin, access.stmt, steps, "r")
                    chain_w = _chain(claim, origin, first, steps, "w")
                    chain_v = _chain(claim, origin, second, steps, "v")
                    at_r = _footprint_place(
                        claim,
                        access.stmt,
                        list(chain_r[0].values()),
                        access.inames,
                        access.domain,
                        "s",
                    )
                    read = [_renamed(i, access.inames, at_r) for i in access.indices]
                    stored = [
                        _renamed(i, first.inames, list(chain_w[0].values()))
                        for i in first.assignee.indices
                    ]
                    other = [
                        _renamed(i, second.inames, list(chain_v[0].values()))
                        for i in second.assignee.indices
                    ]
                    last = len(steps)
                    time_r = _stage_time(
                        origin, steps, last, access.stmt.id, chain_r[-1]
                    )
                    time_w = _stage_time(origin, steps, last, first.id, chain_w[-1])
                    time_v = _stage_time(origin, steps, last, second.id, chain_v[-1])
                    claim.hypotheses += [
                        _equal(read, stored),
                        _equal(placement.at(other), placement.at(stored)),
                        LogicalNot(_equal(other, stored)),
                        lexicographically_before(time_w, time_v),
                    ]
                    overwritten = lexicographically_before(time_v, time_r)
                    goal = (
                        not overwritten
                        if isinstance(overwritten, bool)
                        else LogicalNot(overwritten)
                    )
                    cell = (
                        f"{window.array}"
                        f"[{', '.join(render(i) for i in access.indices)}]"
                    )
                    label = (
                        f"{access.stmt.id} reads {cell} before another cell is "
                        f"stored in its slot, {placement}"
                    )
                    name = f"read_{k}"
                    if len(producers) > 1:
                        name = f"read_{k}_{p}_{q}"
                        label += f", stored by {first.id}, then by {second.id}"
                    parts.append(lanky.Part(name, label, claim.close(goal)))
    except Unstatable as exc:
        return _declined(str(exc))
    counterexample = None
    if window.refusal is not None:
        counterexample = _window_refusal(origin, steps, window, found, parts)
    return _obligation(parts, counterexample, refuted=window.refuted)


def _window_refusal(
    origin: Any,
    steps: Sequence[Any],
    window: Any,
    found: Sequence[Any],
    parts: Sequence[Any],
) -> Any:
    """isl's triple of a refused window, as a counterexample to its read's part.

    A read in a sum reads at its binders too, which the triple does not
    give, and is not stated.
    """
    k, triple = window.refusal
    if not triple or not 0 <= k < len(found) or found[k].part == "sum":
        return None
    access = found[k]
    (reader, at), (stored_by, stored), (written_by, other), params = triple
    producers = {producer.id: producer for producer in window.producers}
    if stored_by not in producers or written_by not in producers:
        return None
    name = f"read_{k}"
    if len(producers) > 1:
        ids = list(producers)
        name = f"read_{k}_{ids.index(stored_by)}_{ids.index(written_by)}"
    part = next((p for p in parts if p.name == name), None)
    if part is None:
        return None
    try:
        values = dict(params)
        for stmt, coords, tag in (
            (access.stmt, at, "r"),
            (producers[stored_by], stored, "w"),
            (producers[written_by], other, "v"),
        ):
            values |= _stages(origin, stmt, steps, coords, params, tag, part)
    except (Unstatable, isl.Error, KeyError, ValueError):
        return None
    text = (
        f"{reader}{_shown(at)}, {stored_by}{_shown(stored)} and "
        f"{written_by}{_shown(other)} at "
        + ", ".join(f"{k}={v}" for k, v in params.items())
    )
    return _counterexample(part, values, text)


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
