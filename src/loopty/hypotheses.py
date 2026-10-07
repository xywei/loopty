"""Facts that travel: a claim decided by isl under the facts it may assume.

Every other isl question loopty asks is about one term under its own
contract. A program is several terms in a row, and what one call leaves in an
array is what the next one reads, so a fact about the second call may assume
what the first one established: its postcondition, a theorem about it, the
zeros an ``Arr.zeros_like`` starts an array with, the types the program's
contract checks of the arrays nothing has written yet. :func:`discharge`
asks isl whether a claim holds under such hypotheses, and
:func:`theorem_instances` turns a theorem a program cites into hypotheses
about its arrays.

The question is the one the rest of loopty asks, the emptiness of a set of
bad instances: the points of the claim's domain at which the hypotheses hold
and the claim does not. Four decisions make it sound.

*A cell is a parameter.* isl cannot read an array, so every cell a claim or a
hypothesis reads, ``off[q]`` or ``cnt[q - 1]``, becomes an isl parameter, one
per cell as written. Its index is put in a canonical affine form first, so
that ``off[(q - 1) + 1]``, which instantiating a hypothesis produces, is the
parameter of ``off[q]``. isl answers for every value of a parameter, so a
verdict is a proof schema over every value the cells can hold, which is sound
for the reason a reflected ragged bound is (:mod:`loopty.flow`): it can only
widen. Two cells written differently are two parameters even where they are
one cell, which loses a verdict and never makes one.

*A universal hypothesis is used at the cells the claim reads.* ``all(off[r +
1] == off[r] + cnt[r] for r in Fin[n])`` says something about every row, and
isl takes no quantifier over an array. So its binder is solved from a cell
the hypothesis reads and a cell the claim reads (``r = q - 1`` from
``off[r + 1]`` and ``off[q]``), and the instance holds where the binder's
domain and guard do: ``0 <= q - 1 < n`` implies ``off[q] == off[q - 1] +
cnt[q - 1]``. An instance reads cells of its own, which are instantiated at
in turn, for :data:`ROUNDS` rounds. Any instance of a true universal is true,
so whichever instances are chosen, the hypotheses stay true.

*What cannot be stated is dropped toward the safe side.* The hypotheses and
the negated claim are put in negation normal form, and a part isl cannot
state (a comparison of reals, a product of two unknowns, a quantifier left
over) becomes ``true``. In negation normal form every part occurs
positively, so the set only grows: a hypothesis assumes less, and the negated
claim looks for more bad instances. Neither can empty a set that is not
empty.

*A fact rests on what it used.* When the set is empty, every hypothesis is
left out in turn, the last first, and kept only if the set is no longer
empty without it. A
fact decided this way rests on the facts the remaining ones rest on, and
nothing else, so a cited theorem the claim does not need does not count in
what the fact is worth.

*Hypotheses that contradict each other decide nothing.* If the ones a claim
used leave no point of its domain, whatever the cells hold, they are never
all true where the claim has a point (an instance of a true universal is
true at the cells a run holds), and every claim would follow from them. Such
a claim is not decided, and the hypotheses are named as contradicting each
other, which says that one of them is false.

A set that is not empty is no refutation: the hypotheses are what a program
knows, not all that is true of it. Its sample is reported as the room the
hypotheses leave, with the values of the cells the claim reads, and a
requirement left so is checked when the program runs instead
(:mod:`loopty.compose`). What isl could not state is reported beside it.
"""

from __future__ import annotations

import dataclasses
import re
from collections.abc import Callable, Collection, Iterator, Mapping, Sequence
from dataclasses import dataclass
from itertools import product
from typing import Any

import islpy as isl
import numpy as np
import pymbolic.primitives as prim
from lanky.prelude import FinType, FnType, Refined
from lanky.terms import (
    Add,
    Comparison,
    Exists,
    Forall,
    LogicalAnd,
    LogicalNot,
    LogicalOr,
    Product,
    Subscript,
    Var,
    conjuncts,
    init_args,
    render,
    structurally_equal,
)

from loopty.flow import assume_sizes, free_names
from loopty.oracle import Empty, sample_parameters, sample_point
from loopty.term import Hypothesis

__all__ = [
    "MAX_INSTANCES",
    "ROUNDS",
    "Discharge",
    "cells_in",
    "discharge",
    "linear",
    "linear_expr",
    "structural_key",
    "substitute",
    "theorem_instances",
]

#: How many rounds of instantiation a hypothesis gets: the cells the claim
#: reads, then the cells the first instances read, and so on.
ROUNDS = 3

#: How many instances one universal hypothesis may have in one round, a bound
#: on the work rather than on what is true.
MAX_INSTANCES = 64

#: The comparisons a negation flips into.
_FLIPPED = {"==": "!=", "!=": "==", "<": ">=", "<=": ">", ">": "<=", ">=": "<"}

#: The comparisons that say the same with their sides swapped.
_MIRRORED = {"==": "==", "!=": "!=", "<": ">", ">": "<", "<=": ">=", ">=": "<="}


# {{{ linear forms and substitution


def linear(expr: Any) -> tuple[dict[str, int], int] | None:
    """``expr`` as an integer combination of names plus a constant, or ``None``.

    ``q - 1`` is ``({"q": 1}, -1)``. A product with more than one factor that
    is not an integer, a quotient, a subscript, anything but names, integers,
    sums and products, is not linear.
    """
    if isinstance(expr, bool | np.bool_):
        return None
    if isinstance(expr, int | np.integer):
        return {}, int(expr)
    if isinstance(expr, prim.Variable):
        return {expr.name: 1}, 0
    if isinstance(expr, prim.Sum):
        names: dict[str, int] = {}
        constant = 0
        for child in expr.children:
            part = linear(child)
            if part is None:
                return None
            for name, coefficient in part[0].items():
                names[name] = names.get(name, 0) + coefficient
            constant += part[1]
        return {name: c for name, c in names.items() if c}, constant
    if isinstance(expr, prim.Product):
        factor = 1
        rest: tuple[dict[str, int], int] | None = None
        for child in expr.children:
            if isinstance(child, int | np.integer) and not isinstance(
                child, bool | np.bool_
            ):
                factor *= int(child)
                continue
            if rest is not None:
                return None
            rest = linear(child)
            if rest is None:
                return None
        if rest is None:
            return {}, factor
        return (
            {name: c * factor for name, c in rest[0].items() if c * factor},
            rest[1] * factor,
        )
    return None


def linear_expr(names: Mapping[str, int], constant: int) -> Any:
    """The expression of a linear form, its names in order and its constant last.

    One spelling for one form, which is what makes two cells written
    differently, ``off[(q - 1) + 1]`` and ``off[q]``, one key.
    """
    terms: list[Any] = []
    for name in sorted(names):
        coefficient = names[name]
        if not coefficient:
            continue
        terms.append(
            Var(name) if coefficient == 1 else Product((coefficient, Var(name)))
        )
    if constant or not terms:
        terms.append(constant)
    return terms[0] if len(terms) == 1 else Add(tuple(terms))


def structural_key(expr: Any) -> Any:
    """A hashable key that two terms share exactly when they are one term.

    Not ``repr``, which pymbolic abbreviates past a depth (``Add((..., 1))``),
    so that ``off[r + 1]`` and ``off[r + 2]`` would share a key, and with it
    an isl parameter, which asserts the two cells equal. A node is keyed by
    its mapper method, so lanky's ``Add`` and pymbolic's ``Sum`` are one
    kind, and by the keys of its constructor arguments.
    """
    if isinstance(expr, prim.ExpressionNode):
        kind = getattr(expr, "mapper_method", None) or type(expr).__name__
        return (kind, tuple(structural_key(arg) for arg in init_args(expr)))
    if isinstance(expr, tuple | list):
        return ("tuple", tuple(structural_key(item) for item in expr))
    if isinstance(expr, FinType):
        return ("Fin", structural_key(expr.bound))
    if isinstance(expr, Refined):
        return ("Refined", structural_key(expr.base), structural_key(tuple(expr.props)))
    if isinstance(expr, bool | np.bool_):
        return ("bool", bool(expr))
    if isinstance(expr, int | np.integer):
        return ("int", int(expr))
    if isinstance(expr, float | np.floating):
        return ("float", float(expr))
    if isinstance(expr, str):
        return ("str", expr)
    return ("other", type(expr).__name__, str(expr))


def _canonical(expr: Any) -> Any:
    """``expr`` in its canonical linear spelling when it has one, as it is else."""
    form = linear(expr)
    return expr if form is None else linear_expr(*form)


def _names(expr: Any) -> set[str]:
    """Every variable name in ``expr``, bound or free."""
    out: set[str] = set()

    def visit(node: Any) -> None:
        if isinstance(node, prim.Variable):
            out.add(node.name)
        elif isinstance(node, Forall | Exists):
            for var, domain in node.binders:
                out.add(var.name)
                visit(_sort_terms(domain))
            visit(node.body)
            visit(node.guard)
        elif isinstance(node, prim.ExpressionNode):
            for arg in init_args(node):
                visit(arg)
        elif isinstance(node, tuple | list):
            for item in node:
                visit(item)

    visit(expr)
    return out


def _sort_terms(sort: Any) -> tuple[Any, ...]:
    """The terms a binder's sort mentions: a ``Fin``'s bound, a refinement's props."""
    if isinstance(sort, FinType):
        return (sort.bound,)
    if isinstance(sort, Refined):
        return (*_sort_terms(sort.base), *sort.props)
    return ()


def _substitute_sort(sort: Any, mapping: Mapping[str, Any]) -> Any:
    """A binder's sort with ``mapping`` applied to the terms it mentions."""
    if isinstance(sort, FinType):
        return dataclasses.replace(sort, bound=substitute(sort.bound, mapping))
    if isinstance(sort, Refined) and dataclasses.is_dataclass(sort):
        return dataclasses.replace(
            sort,
            base=_substitute_sort(sort.base, mapping),
            props=tuple(substitute(prop, mapping) for prop in sort.props),
        )
    return sort


def substitute(expr: Any, mapping: Mapping[str, Any]) -> Any:
    """``expr`` with every free name in ``mapping`` replaced, capture-avoiding.

    A quantifier's binders shadow the names they bind, and a binder whose
    name occurs in a replacement is renamed first, so that ``r`` replaced by
    ``q - 1`` under ``forall q`` does not become the ``q`` the quantifier
    binds. Nodes are rebuilt from their constructor arguments, so lanky's
    classes stay lanky's.
    """
    if not mapping:
        return expr
    if isinstance(expr, prim.Variable):
        return mapping.get(expr.name, expr)
    if isinstance(expr, Forall | Exists):
        inner = dict(mapping)
        captured = set()
        for value in mapping.values():
            captured |= _names(value)
        taken = captured | _names(expr) | set(mapping)
        binders = []
        renames: dict[str, Any] = {}
        for var, domain in expr.binders:
            domain = _substitute_sort(_substitute_sort(domain, renames), inner)
            inner.pop(var.name, None)
            if var.name in captured:
                fresh = _fresh(var.name, taken)
                taken.add(fresh)
                renames[var.name] = Var(fresh)
                binders.append((Var(fresh), domain))
            else:
                binders.append((var, domain))
        body = substitute(substitute(expr.body, renames), inner)
        guard = (
            None
            if expr.guard is None
            else substitute(substitute(expr.guard, renames), inner)
        )
        return type(expr)(tuple(binders), body, guard)
    if isinstance(expr, prim.ExpressionNode):
        args = init_args(expr)
        new = tuple(substitute(arg, mapping) for arg in args)
        if all(old is changed for old, changed in zip(args, new, strict=True)):
            return expr
        return type(expr)(*new)
    if isinstance(expr, tuple):
        return tuple(substitute(item, mapping) for item in expr)
    return expr


def _fresh(stem: str, taken: Collection[str]) -> str:
    """``stem`` with a numeric suffix nothing in ``taken`` has."""
    suffix = 0
    name = f"{stem}_{suffix}"
    while name in taken:
        suffix += 1
        name = f"{stem}_{suffix}"
    return name


def cells_in(expr: Any) -> list[prim.Subscript]:
    """Every cell ``expr`` reads, ``a[i]`` with ``a`` a name, outermost first.

    A cell under a quantifier that names one of its binders is not a cell of
    the expression but a pattern of the quantifier, and is left out.
    """
    out: list[prim.Subscript] = []

    def visit(node: Any, bound: frozenset[str]) -> None:
        if isinstance(node, prim.Subscript) and isinstance(
            node.aggregate, prim.Variable
        ):
            if not (_names(node.index) & bound):
                out.append(node)
            visit(node.index, bound)
        elif isinstance(node, Forall | Exists):
            inner = bound | {var.name for var, _ in node.binders}
            for _var, domain in node.binders:
                for term in _sort_terms(domain):
                    visit(term, inner)
            visit(node.body, inner)
            visit(node.guard, inner)
        elif isinstance(node, prim.ExpressionNode):
            for arg in init_args(node):
                visit(arg, bound)
        elif isinstance(node, tuple | list):
            for item in node:
                visit(item, bound)

    visit(expr, frozenset())
    return out


def _indices(cell: prim.Subscript) -> tuple[Any, ...]:
    index = cell.index
    return index if isinstance(index, tuple) else (index,)


def _cell(array: str, indices: Sequence[Any]) -> Subscript:
    """``array[indices]``, each index in its canonical spelling."""
    canonical = tuple(_canonical(index) for index in indices)
    return Subscript(Var(array), canonical[0] if len(canonical) == 1 else canonical)


# }}}


# {{{ cells as parameters


class _Cells:
    """The isl parameter each cell stands for, keyed by the cell as written."""

    def __init__(self, reserved: Collection[str], integral: Collection[str]) -> None:
        self.taken = set(reserved)
        self.integral = set(integral)
        self.by_key: dict[Any, str] = {}
        #: Each parameter's cell, in its canonical spelling.
        self.cells: dict[str, Subscript] = {}

    def adopt(self, name: str, cell: prim.Subscript) -> None:
        canonical = _cell(cell.aggregate.name, _indices(cell))
        self.by_key[structural_key(canonical)] = name
        self.cells[name] = canonical
        self.taken.add(name)

    def param(self, cell: prim.Subscript) -> str | None:
        """The parameter of ``cell``, or ``None`` for an array of no integers."""
        array = cell.aggregate.name
        if array not in self.integral:
            return None
        canonical = _cell(array, _indices(cell))
        key = structural_key(canonical)
        found = self.by_key.get(key)
        if found is not None:
            return found
        stem = re.sub(r"\W+", "_", render(canonical)).strip("_") or array
        name = stem
        suffix = 2
        while name in self.taken:
            name = f"{stem}_{suffix}"
            suffix += 1
        self.adopt(name, canonical)
        return name

    def known(self) -> list[Subscript]:
        return list(self.cells.values())


# }}}


# {{{ propositions as isl text


def _text(expr: Any, cells: _Cells, known: Collection[str]) -> str | None:
    """``expr`` as isl's affine syntax, or ``None`` where isl cannot state it."""
    if isinstance(expr, bool | np.bool_):
        return None
    if isinstance(expr, int | np.integer):
        return str(int(expr))
    if isinstance(expr, prim.Variable):
        return expr.name if expr.name in known else None
    if isinstance(expr, prim.Subscript):
        if not isinstance(expr.aggregate, prim.Variable):
            return None
        return cells.param(expr)
    if isinstance(expr, prim.Sum):
        parts = [_text(child, cells, known) for child in expr.children]
        if any(part is None for part in parts):
            return None
        return "(" + " + ".join(parts) + ")"  # type: ignore[arg-type]
    if isinstance(expr, prim.Product):
        constants = [
            int(c)
            for c in expr.children
            if isinstance(c, int | np.integer) and not isinstance(c, bool | np.bool_)
        ]
        others = [
            c
            for c in expr.children
            if not (
                isinstance(c, int | np.integer) and not isinstance(c, bool | np.bool_)
            )
        ]
        if len(others) > 1:
            return None
        factor = 1
        for c in constants:
            factor *= c
        if not others:
            return str(factor)
        inner = _text(others[0], cells, known)
        return None if inner is None else f"({factor} * {inner})"
    if isinstance(expr, prim.FloorDiv | prim.Remainder):
        denominator = expr.denominator
        if not (
            isinstance(denominator, int | np.integer)
            and not isinstance(denominator, bool | np.bool_)
            and int(denominator) > 0
        ):
            return None
        numerator = _text(expr.numerator, cells, known)
        if numerator is None:
            return None
        if isinstance(expr, prim.FloorDiv):
            return f"floord({numerator}, {int(denominator)})"
        return f"(({numerator}) mod {int(denominator)})"
    return None


def _encode(
    prop: Any,
    cells: _Cells,
    known: Collection[str],
    negate: bool = False,
    dropped: list[Any] | None = None,
) -> str:
    """``prop``, or its negation, in negation normal form as isl text.

    A part isl cannot state is ``true`` (see the module docstring): every
    part of a formula in negation normal form occurs positively, so what is
    stated is implied by what was meant. Each such part is appended to
    ``dropped``, when given, so that a reason can say what was not used.
    """
    if isinstance(prop, bool | np.bool_):
        return "1 = 1" if bool(prop) != negate else "1 = 0"
    if isinstance(prop, prim.LogicalNot):
        return _encode(prop.child, cells, known, not negate, dropped)
    if isinstance(prop, prim.LogicalAnd | prim.LogicalOr):
        conjunction = isinstance(prop, prim.LogicalAnd) != negate
        parts = [
            _encode(child, cells, known, negate, dropped) for child in prop.children
        ]
        if not parts:
            return "1 = 1" if conjunction else "1 = 0"
        joiner = " and " if conjunction else " or "
        return "(" + joiner.join(parts) + ")"
    if isinstance(prop, prim.Comparison):
        operator = _FLIPPED[prop.operator] if negate else prop.operator
        left = _text(prop.left, cells, known)
        right = _text(prop.right, cells, known)
        if left is None or right is None or operator not in _FLIPPED:
            if dropped is not None:
                dropped.append(prop)
            return "1 = 1"
        if operator == "==":
            return f"({left} = {right})"
        if operator == "!=":
            return f"({left} < {right} or {left} > {right})"
        return f"({left} {operator} {right})"
    if dropped is not None:
        dropped.append(prop)
    return "1 = 1"


# }}}


# {{{ instantiating a universal hypothesis


def _domain_condition(var: Var, sort: Any, value: Any) -> list[Any] | None:
    """What ``value`` has to satisfy to be a point of ``sort``, or ``None``.

    ``None`` is a sort no instance can be taken in, such as ``Real``.
    """
    if isinstance(sort, FinType):
        return [Comparison(0, "<=", value), Comparison(value, "<", sort.bound)]
    if isinstance(sort, Refined):
        base = _domain_condition(var, sort.base, value)
        if base is None:
            return None
        return [*base, *(substitute(prop, {var.name: value}) for prop in sort.props)]
    name = getattr(sort, "name", None)
    if name == "Nat":
        return [Comparison(value, ">=", 0)]
    if name == "Int":
        return []
    return None


def _candidates(
    universal: Forall, cells: Sequence[Subscript]
) -> dict[str, list[Any]] | None:
    """The values each binder of ``universal`` is instantiated at.

    For a binder ``r`` and a cell ``a[..., r + c, ...]`` the hypothesis reads,
    every known cell of ``a`` gives ``r`` its index there less ``c``. ``None``
    when a binder gets no value, since then no instance is about a cell the
    claim reads.
    """
    binders = [var.name for var, _ in universal.binders]
    patterns = cells_in_pattern(universal, set(binders))
    by_array: dict[str, list[Subscript]] = {}
    for cell in cells:
        by_array.setdefault(cell.aggregate.name, []).append(cell)
    out: dict[str, list[Any]] = {name: [] for name in binders}
    seen: dict[str, set[Any]] = {name: set() for name in binders}
    for pattern in patterns:
        indices = _indices(pattern)
        for position, index in enumerate(indices):
            form = linear(index)
            if form is None:
                continue
            names, constant = form
            mine = [name for name in names if name in binders]
            if len(mine) != 1 or names[mine[0]] != 1:
                continue
            binder = mine[0]
            offset = {name: c for name, c in names.items() if name != binder}
            for cell in by_array.get(pattern.aggregate.name, ()):
                target = _indices(cell)
                if len(target) != len(indices):
                    continue
                found = linear(target[position])
                if found is None:
                    if offset or constant:
                        continue
                    value = target[position]
                else:
                    values = dict(found[0])
                    for name, c in offset.items():
                        values[name] = values.get(name, 0) - c
                    value = linear_expr(values, found[1] - constant)
                if _names(value) & set(binders):
                    continue
                key = structural_key(value)
                if key not in seen[binder]:
                    seen[binder].add(key)
                    out[binder].append(value)
    if any(not values for values in out.values()):
        return None
    return out


def cells_in_pattern(universal: Forall, binders: set[str]) -> list[prim.Subscript]:
    """The cells a quantifier's body and guard read that name its binders."""
    out: list[prim.Subscript] = []

    def visit(node: Any) -> None:
        if isinstance(node, prim.Subscript) and isinstance(
            node.aggregate, prim.Variable
        ):
            if _names(node.index) & binders:
                out.append(node)
            visit(node.index)
        elif isinstance(node, prim.ExpressionNode):
            for arg in init_args(node):
                visit(arg)
        elif isinstance(node, tuple | list):
            for item in node:
                visit(item)

    visit(universal.body)
    visit(universal.guard)
    for _var, domain in universal.binders:
        for term in _sort_terms(domain):
            visit(term)
    return out


def _instances(claim: Any, cells: Sequence[Subscript]) -> Iterator[Any]:
    """Ground instances of ``claim`` at ``cells``: its conjuncts, instantiated.

    A conjunct with no quantifier is its own instance. A universal one is
    instantiated at the values :func:`_candidates` gives its binders, each
    instance holding where the binders' domains and the guard do. Any other
    conjunct is used as it is, which :func:`_encode` reads as far as it can.
    """
    for conjunct in conjuncts(claim):
        if not isinstance(conjunct, Forall) or not conjunct.binders:
            if isinstance(conjunct, Forall):
                # No binders: a sequent, its guard the antecedent.
                if conjunct.guard is None:
                    yield conjunct.body
                else:
                    yield LogicalOr((LogicalNot(conjunct.guard), conjunct.body))
                continue
            yield conjunct
            continue
        values = _candidates(conjunct, cells)
        if values is None:
            continue
        names = [var.name for var, _ in conjunct.binders]
        for count, choice in enumerate(
            product(*(values[name] for name in names))
        ):
            if count >= MAX_INSTANCES:
                break
            bound = _bind(conjunct.binders, choice)
            if bound is None:
                # A binder of a sort no instance can be taken in, at any value.
                break
            mapping, conditions = bound
            if conjunct.guard is not None:
                conditions.append(substitute(conjunct.guard, mapping))
            body = substitute(conjunct.body, mapping)
            if not conditions:
                yield body
                continue
            antecedent = (
                conditions[0] if len(conditions) == 1 else LogicalAnd(tuple(conditions))
            )
            yield LogicalOr((LogicalNot(antecedent), body))


def _bind(
    binders: Sequence[tuple[Var, Any]], choice: Sequence[Any]
) -> tuple[dict[str, Any], list[Any]] | None:
    """The values of ``binders`` at ``choice``, with what makes them points.

    ``None`` when a binder's sort is one no instance can be taken in.
    """
    mapping: dict[str, Any] = {}
    conditions: list[Any] = []
    for (var, sort), value in zip(binders, choice, strict=True):
        condition = _domain_condition(var, _substitute_sort(sort, mapping), value)
        if condition is None:
            return None
        conditions.extend(condition)
        mapping[var.name] = value
    return mapping, conditions


# }}}


@dataclass(frozen=True)
class Discharge:
    """What :func:`discharge` found.

    ``decided`` when no point of the claim's domain breaks it where the
    hypotheses hold; ``question`` is then that set, empty, built from the
    hypotheses ``used`` alone, for the isl oracle to answer again in the
    ledger. Otherwise ``question`` is ``None`` and ``reason`` says why: the
    point the hypotheses leave room for, with the values of the cells the
    claim reads there, or, where the hypotheses contradict each other at
    every point of the domain, which ones do (``contradicting``), since a
    claim decided under them would be decided vacuously. ``unstated`` lists
    the parts of the hypotheses and of the claim that isl cannot state,
    each with where it comes from, which were read as saying nothing.
    """

    decided: bool
    question: Empty | None
    used: tuple[Hypothesis, ...]
    reason: str
    contradicting: tuple[Hypothesis, ...] = ()
    unstated: tuple[str, ...] = ()


def _align(first: Any, second: Any) -> tuple[Any, Any]:
    second = second.align_params(first.get_space())
    first = first.align_params(second.get_space())
    return first, second


def _formula_set(text: str, dims: Sequence[str]) -> isl.Set:
    """``{ [dims] : text }`` with every other name in ``text`` a parameter."""
    params = sorted(free_names(text) - set(dims))
    head = f"[{', '.join(params)}] -> " if params else ""
    return isl.Set(f"{head}{{ [{', '.join(dims)}] : {text} }}")


def _intersect(found: isl.Set, text: str, dims: Sequence[str]) -> isl.Set:
    if text == "1 = 1":
        return found
    piece = _formula_set(text, dims)
    found, piece = _align(found, piece)
    return found.intersect(piece).coalesce()


def _assume(
    found: isl.Set,
    prop: Any,
    cells: _Cells,
    known: Collection[str],
    dims: Sequence[str],
    dropped: list[Any] | None = None,
) -> isl.Set:
    """``found`` where ``prop`` holds, or a set that contains it.

    ``not A or B`` is the shape of every instance of a universal, and it is
    taken apart rather than handed to isl whole, which would double the
    pieces of the set with every instance. Its two halves are encoded each
    toward the safe side (:func:`_encode`), so the result, the points of
    ``found`` where ``not A`` may hold and those where ``B`` may, contains
    every point of ``found`` where the instance holds. Where ``found`` lies
    inside ``A`` that is ``found`` and ``B``, and where it lies outside the
    instance says nothing about it.
    """
    if (
        isinstance(prop, prim.LogicalOr)
        and len(prop.children) == 2
        and isinstance(prop.children[0], prim.LogicalNot)
    ):
        antecedent = prop.children[0].child
        consequent = prop.children[1]
        outside = _encode(antecedent, cells, known, negate=True, dropped=dropped)
        if outside == "1 = 1":
            return found
        escaped = _formula_set(outside, dims)
        found, escaped = _align(found, escaped)
        escaped = found.intersect(escaped)
        if escaped.is_equal(found):
            return found
        held = _intersect(
            found, _encode(consequent, cells, known, dropped=dropped), dims
        )
        if escaped.is_empty():
            return held
        escaped, held = _align(escaped, held)
        return escaped.union(held).coalesce()
    return _intersect(found, _encode(prop, cells, known, dropped=dropped), dims)


def _run(
    domain: isl.Set,
    goal: Any,
    hypotheses: Sequence[Hypothesis],
    integral: Collection[str],
    known: Collection[str],
    nonneg: Collection[str],
    reflected: Mapping[str, Any],
    *,
    negated: bool = True,
    unstated: list[str] | None = None,
) -> tuple[isl.Set, _Cells]:
    """The bad set of ``goal`` over ``domain`` under ``hypotheses``.

    With ``negated=False``, the points of ``domain`` where the hypotheses
    hold, instantiated at the cells ``goal`` reads as they would be for the
    bad set, whether ``goal`` holds there or not: that set is empty when the
    hypotheses contradict each other at every point. ``unstated``, when
    given, collects what isl cannot state, each part with where it comes
    from.
    """
    dims = list(domain.get_var_names(isl.dim_type.set))
    params = list(domain.get_var_names(isl.dim_type.param))
    reserved = set(dims) | set(params) | set(known) | _names(goal)
    for hypothesis in hypotheses:
        reserved |= _names(hypothesis.claim)
    cells = _Cells(reserved, integral)
    for name, expr in reflected.items():
        if isinstance(expr, prim.Subscript) and isinstance(
            expr.aggregate, prim.Variable
        ):
            cells.adopt(name, expr)
    names = set(dims) | set(known) | set(params)
    for cell in cells_in(goal):
        cells.param(cell)

    def note(dropped: list[Any], where: str) -> None:
        if unstated is None:
            return
        for part in dropped:
            said = f"{render(part)} ({where})"
            if said not in unstated:
                unstated.append(said)

    found = domain
    if negated:
        dropped: list[Any] = []
        found = _intersect(
            found, _encode(goal, cells, names, negate=True, dropped=dropped), dims
        )
        note(dropped, "of the claim")
    found = assume_sizes(found, set(nonneg) | set(reflected))
    seen: set[Any] = set()
    pending = list(hypotheses)
    for _round in range(ROUNDS):
        known_cells = cells.known()
        grew = False
        for hypothesis in pending:
            for instance in _instances(hypothesis.claim, known_cells):
                key = structural_key(instance)
                if key in seen:
                    continue
                seen.add(key)
                grew = True
                for cell in cells_in(instance):
                    cells.param(cell)
                dropped = []
                found = _assume(found, instance, cells, names, dims, dropped)
                note(dropped, f"of {hypothesis.source}")
                if found.is_empty():
                    return found, cells
        if not grew:
            break
    return found, cells


def discharge(
    domain: isl.Set,
    goal: Any,
    hypotheses: Sequence[Hypothesis],
    *,
    integral: Collection[str],
    known: Collection[str] = (),
    nonneg: Collection[str] = (),
    reflected: Mapping[str, Any] | None = None,
    description: str = "",
) -> Discharge:
    """Ask isl whether ``goal`` holds at every point of ``domain`` under ``hypotheses``.

    ``domain`` is an isl set over the claim's binders, which ``goal``
    mentions by name. ``integral`` names the arrays whose cells are
    integers, and only their cells become parameters; ``known`` the other
    names the propositions may mention (sizes, scalars of an integral sort);
    ``nonneg`` the parameters that are sizes and so not negative; and
    ``reflected`` the parameters of ``domain`` that already stand for a
    cell, ``nl_cnt_r`` for ``cnt[r]``, so that a hypothesis about ``cnt[r]``
    is about the bound of the domain too.
    """
    reflected = dict(reflected or {})
    unstated: list[str] = []
    found, cells = _run(
        domain,
        goal,
        hypotheses,
        integral,
        known,
        nonneg,
        reflected,
        unstated=unstated,
    )
    dims = tuple(domain.get_var_names(isl.dim_type.set))
    if not found.is_empty():
        return Discharge(
            False,
            None,
            (),
            _room(found, cells, dims, goal, domain),
            unstated=tuple(unstated),
        )
    used = list(hypotheses)
    # Last first: what a call derives from others (a theorem's instance, the
    # requirement a call is checked for) goes before what it derives from,
    # so that a fact rests on the postcondition rather than on a restatement
    # of it whenever either would do.
    for hypothesis in reversed(list(hypotheses)):
        trial = [kept for kept in used if kept is not hypothesis]
        found_without, _ = _run(
            domain, goal, trial, integral, known, nonneg, reflected
        )
        if found_without.is_empty():
            used = trial
    if used:
        # Hypotheses that leave no point of the domain at all contradict each
        # other there, and decide every claim about it: vacuously, which is
        # no decision. A domain empty without them is a claim about nothing,
        # and is decided as such.
        held, _ = _run(
            domain, goal, used, integral, known, nonneg, reflected, negated=False
        )
        if held.is_empty():
            alone, _ = _run(
                domain, goal, (), integral, known, nonneg, reflected, negated=False
            )
            if not alone.is_empty():
                sources = "; ".join(hypothesis.source for hypothesis in used)
                said = (
                    f"no cells satisfy {sources} at any point of the claim's "
                    "domain, so it is false wherever the claim has a point"
                    if len(used) == 1
                    else f"no cells satisfy all of {sources} at any point of the "
                    "claim's domain, so they contradict each other there and one "
                    "of them is false"
                )
                return Discharge(
                    False,
                    None,
                    (),
                    f"{said}; a claim decided under that would be decided vacuously",
                    contradicting=tuple(used),
                    unstated=tuple(unstated),
                )
    final, _ = _run(domain, goal, used, integral, known, nonneg, reflected)
    question = Empty(final, description=description, labels=dims)
    return Discharge(True, question, tuple(used), "")


def _room(
    found: isl.Set, cells: _Cells, dims: Sequence[str], goal: Any, domain: isl.Set
) -> str:
    """The point the hypotheses leave room for, with its cells, in words.

    Only the cells the claim reads or its domain is bounded by are named,
    with the sizes: the cells an instance of a hypothesis reads beyond those
    are what the instantiation reached, and say little about why the claim
    fails.
    """
    point = sample_point(found)
    parameters = sample_parameters(found)
    parts = []
    if point is not None and dims:
        parts.append(
            "[" + ", ".join(f"{d}={v}" for d, v in zip(dims, point, strict=True)) + "]"
        )
    read = {cells.param(cell) for cell in cells_in(goal)}
    read |= set(domain.get_var_names(isl.dim_type.param))
    values = []
    for name, value in parameters.items():
        cell = cells.cells.get(name)
        if cell is not None and name not in read:
            continue
        values.append(f"{render(cell) if cell is not None else name} = {value}")
    where = " ".join(parts)
    if values:
        where = f"{where} with {', '.join(values)}".strip()
    return where or "a point"


# {{{ theorems as hypotheses


class _Matcher:
    """Matches a theorem's proposition against a program's, binding its variables.

    The theorem's families (``off: Fn[...]``, applied as ``off(r)``) match
    the program's arrays (read as ``off[r]``), its scalar variables match
    expressions of the program's sizes, and its binders match the program's
    binders one for one.
    """

    def __init__(self, functions: Collection[str], scalars: Collection[str]) -> None:
        self.functions = set(functions)
        self.scalars = set(scalars)

    def match(
        self, pattern: Any, target: Any, sigma: dict[str, Any], bound: dict[str, str]
    ) -> bool:
        if isinstance(pattern, bool | np.bool_) or isinstance(target, bool | np.bool_):
            return isinstance(target, bool | np.bool_) and isinstance(
                pattern, bool | np.bool_
            ) and bool(pattern) == bool(target)
        if isinstance(pattern, int | np.integer):
            return (
                isinstance(target, int | np.integer)
                and not isinstance(target, bool | np.bool_)
                and int(target) == int(pattern)
            )
        if isinstance(pattern, prim.Variable):
            name = pattern.name
            if name in bound:
                return isinstance(target, prim.Variable) and target.name == bound[name]
            if name in self.scalars:
                if name in sigma:
                    return _same(sigma[name], target)
                if _names(target) & set(bound.values()):
                    return False
                sigma[name] = target
                return True
            if name in self.functions:
                if not isinstance(target, prim.Variable):
                    return False
                if name in sigma:
                    return sigma[name] == target.name
                sigma[name] = target.name
                return True
            return isinstance(target, prim.Variable) and target.name == name
        if isinstance(pattern, prim.Call):
            function = pattern.function
            if not (
                isinstance(function, prim.Variable) and function.name in self.functions
            ):
                return structurally_equal(pattern, target)
            if not (
                isinstance(target, prim.Subscript)
                and isinstance(target.aggregate, prim.Variable)
            ):
                return False
            array = target.aggregate.name
            if sigma.get(function.name, array) != array:
                return False
            sigma[function.name] = array
            indices = _indices(target)
            if len(indices) != len(pattern.parameters):
                return False
            return all(
                self.match(p, t, sigma, bound)
                for p, t in zip(pattern.parameters, indices, strict=True)
            )
        if isinstance(pattern, prim.Comparison):
            if not isinstance(target, prim.Comparison):
                return False
            if pattern.operator == target.operator:
                trial = dict(sigma)
                if self.match(pattern.left, target.left, trial, bound) and self.match(
                    pattern.right, target.right, trial, bound
                ):
                    sigma.update(trial)
                    return True
            if _MIRRORED.get(pattern.operator) == target.operator:
                trial = dict(sigma)
                if self.match(pattern.left, target.right, trial, bound) and self.match(
                    pattern.right, target.left, trial, bound
                ):
                    sigma.update(trial)
                    return True
            return False
        if isinstance(pattern, Forall | Exists):
            if type(pattern) is not type(target) or len(pattern.binders) != len(
                target.binders
            ):
                return False
            inner = dict(bound)
            for (pvar, psort), (tvar, tsort) in zip(
                pattern.binders, target.binders, strict=True
            ):
                if not self._match_sort(psort, tsort, sigma, inner):
                    return False
                inner[pvar.name] = tvar.name
            if (pattern.guard is None) != (target.guard is None):
                return False
            if pattern.guard is not None and not self.match(
                pattern.guard, target.guard, sigma, inner
            ):
                return False
            return self.match(pattern.body, target.body, sigma, inner)
        if isinstance(
            pattern, prim.Sum | prim.Product | prim.LogicalAnd | prim.LogicalOr
        ):
            if not isinstance(target, _base_of(pattern)):
                return False
            children = pattern.children
            targets = target.children
            if len(children) != len(targets):
                return False
            orders = [targets]
            if len(targets) == 2 and isinstance(pattern, prim.Sum | prim.Product):
                orders.append((targets[1], targets[0]))
            for order in orders:
                trial = dict(sigma)
                if all(
                    self.match(p, t, trial, bound)
                    for p, t in zip(children, order, strict=True)
                ):
                    sigma.update(trial)
                    return True
            return False
        if isinstance(pattern, prim.LogicalNot):
            return isinstance(target, prim.LogicalNot) and self.match(
                pattern.child, target.child, sigma, bound
            )
        return structurally_equal(pattern, target)

    def _match_sort(
        self, pattern: Any, target: Any, sigma: dict[str, Any], bound: dict[str, str]
    ) -> bool:
        if isinstance(pattern, FinType):
            return isinstance(target, FinType) and self.match(
                pattern.bound, target.bound, sigma, bound
            )
        return str(pattern) == str(target)


def _base_of(node: Any) -> type:
    """The pymbolic class a lanky node subclasses, to compare kinds of node."""
    for kind in (prim.Sum, prim.Product, prim.LogicalAnd, prim.LogicalOr):
        if isinstance(node, kind):
            return kind
    return type(node)


def _same(left: Any, right: Any) -> bool:
    """Two expressions that are equal as terms, or as linear forms."""
    if structurally_equal(left, right):
        return True
    first, second = linear(left), linear(right)
    return first is not None and second is not None and first == second


def _apply(expr: Any, sigma: Mapping[str, Any], functions: Collection[str]) -> Any:
    """A theorem's proposition with its variables replaced by what they matched.

    A family applied, ``off(r)``, becomes the cell ``off[r]`` of the array it
    matched; a scalar variable becomes the expression it matched.
    """
    if isinstance(expr, prim.Call) and isinstance(expr.function, prim.Variable):
        if expr.function.name in functions:
            args = tuple(_apply(arg, sigma, functions) for arg in expr.parameters)
            return Subscript(
                Var(sigma[expr.function.name]), args[0] if len(args) == 1 else args
            )
    if isinstance(expr, prim.Variable):
        if expr.name in sigma and expr.name not in functions:
            return sigma[expr.name]
        return expr
    if isinstance(expr, Forall | Exists):
        binders = tuple(
            (var, _apply_sort(sort, sigma, functions)) for var, sort in expr.binders
        )
        inner = {
            name: value
            for name, value in sigma.items()
            if name not in {var.name for var, _ in expr.binders}
        }
        return type(expr)(
            binders,
            _apply(expr.body, inner, functions),
            None if expr.guard is None else _apply(expr.guard, inner, functions),
        )
    if isinstance(expr, prim.ExpressionNode):
        args = init_args(expr)
        new = tuple(_apply(arg, sigma, functions) for arg in args)
        if all(old is changed for old, changed in zip(args, new, strict=True)):
            return expr
        return type(expr)(*new)
    if isinstance(expr, tuple):
        return tuple(_apply(item, sigma, functions) for item in expr)
    return expr


def _apply_sort(sort: Any, sigma: Mapping[str, Any], functions: Collection[str]) -> Any:
    if isinstance(sort, FinType):
        return dataclasses.replace(sort, bound=_apply(sort.bound, sigma, functions))
    return sort


def theorem_instances(
    theorem: Any,
    available: Sequence[Hypothesis],
    applicable: Callable[[str, Any, Any], str | None],
    nonnegative: Callable[[Any], bool],
) -> tuple[list[Hypothesis], list[str]]:
    """The instances of ``theorem`` that ``available`` hypotheses establish.

    ``theorem`` is a lanky :class:`~lanky.theory.Theorem`: its variables,
    its hypotheses and its goal. Every hypothesis of the theorem has to
    match a conjunct of one of ``available`` (up to the names of binders,
    with a family ``off(r)`` matching a cell ``off[r]``), which binds the
    theorem's families to arrays and its scalars to expressions of the
    program's sizes. The goal, with those put in, is a hypothesis about the
    program, resting on the theorem's fact and on the facts the matched
    hypotheses rest on.

    A theorem is about families of its sorts, so a binding is checked
    against them before it is used: ``applicable(array, domain, codomain)``
    says why an array cannot stand for a family ``Fn[domain, codomain]``
    there, or ``None`` when it can, and ``nonnegative(expr)`` whether an
    expression a ``Nat`` variable matched is never negative. A binding that
    fails is not used, and the reasons come back beside the instances.
    """
    variables = dict(getattr(theorem, "variables", ()))
    functions = {name for name, sort in variables.items() if isinstance(sort, FnType)}
    scalars = set(variables) - functions
    matcher = _Matcher(functions, scalars)
    wanted = [prop for _name, prop in getattr(theorem, "hypotheses", ())]
    goal = getattr(theorem, "goal", None)
    name = getattr(theorem, "__name__", None) or getattr(theorem, "qualname", "theorem")
    if goal is None or not wanted:
        return [], [f"{name} has no hypotheses for a program's facts to establish"]
    pool = [
        (index, conjunct)
        for index, hypothesis in enumerate(available)
        for conjunct in conjuncts(hypothesis.claim)
    ]
    matches: list[tuple[dict[str, Any], tuple[int, ...]]] = []

    def search(k: int, sigma: dict[str, Any], used: tuple[int, ...]) -> None:
        if len(matches) >= MAX_INSTANCES:
            return
        if k == len(wanted):
            matches.append((dict(sigma), used))
            return
        for index, conjunct in pool:
            trial = dict(sigma)
            if matcher.match(wanted[k], conjunct, trial, {}):
                search(k + 1, trial, (*used, index))

    search(0, {}, ())
    out: list[Hypothesis] = []
    reasons: list[str] = []
    seen: set[Any] = set()
    for sigma, used in matches:
        missing = [var for var in variables if var not in sigma]
        if missing:
            reasons.append(
                f"{name} leaves {', '.join(missing)} unbound by the hypotheses it "
                "matched"
            )
            continue
        why = None
        for var, sort in variables.items():
            if var in functions:
                domain = _apply_sort(sort.domain, sigma, functions)
                why = applicable(sigma[var], domain, sort.codomain)
            elif getattr(sort, "name", None) == "Nat":
                if not nonnegative(sigma[var]):
                    why = (
                        f"{var} of {name} is a Nat, and {render(sigma[var])}, "
                        "which it matched, may be negative"
                    )
            elif getattr(sort, "name", None) != "Int":
                why = (
                    f"{var} of {name} is of the sort {sort}, which no binding "
                    "is checked against"
                )
            if why is not None:
                break
        if why is not None:
            reasons.append(why)
            continue
        claim = _apply(goal, sigma, functions)
        key = structural_key(claim)
        if key in seen:
            continue
        seen.add(key)
        rests_on: list[str] = []
        mentions: set[str] = set()
        for index in used:
            for fact in available[index].rests_on:
                if fact not in rests_on:
                    rests_on.append(fact)
            mentions |= available[index].mentions
        theorem_id = getattr(theorem, "fact_id", None)
        bindings = ", ".join(
            f"{var} = {sigma[var] if var in functions else render(sigma[var])}"
            for var in variables
        )
        mentions |= {sigma[var] for var in functions}
        out.append(
            Hypothesis(
                claim=claim,
                source=f"{name} at {bindings}",
                rests_on=tuple(
                    [*([theorem_id] if isinstance(theorem_id, str) else []), *rests_on]
                ),
                mentions=frozenset(mentions),
            )
        )
    if not matches:
        reasons.append(
            f"no hypothesis that held matches every hypothesis of {name}"
        )
    return out, reasons


# }}}
