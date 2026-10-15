"""A postcondition's equations of reals, at the class of what they compare.

The design idea. A kernel's postcondition is evaluated at what its native
runs leave (:func:`loopty.faithful.postcondition_fact`), and an equation of
reals there was compared bit for bit. A closed form the kernel computes
another way is then false in the last bits, and could not be a
postcondition: ``mass_ffc``'s matrix is ``|J| (1 + delta_jk) / 24`` only to
the 15 digits of FFC's tables. The compiled run is compared with the native
one at the class of each output (:mod:`loopty.tolerance`), ``approx`` for
``Real``, and an equation of a postcondition is now read the same way::

    a == b    read as    |a - b| <= eps_class * (|b| + 1)

with the class the weakest of the arrays and scalars it reads
(:func:`loopty.tolerance.output_class`). An equation that says the same of
floats as of reals is compared exactly, as before: one of integers, and one
whose sides are cells, scalars and literals, perhaps negated, which nothing
in it rounds (:func:`loopty.dataflow.exact`); so is one whose class is
``exact``. The fact's statement says which: ``A[0, 0] == abs(...) / 12 and
..., to the approx tolerance``.

*Proved over the reals.* Where every cell of an array the kernel writes is
stored once and read after (:func:`loopty.dataflow.stores`), each cell holds
what its statement computes there. A claim at concrete cells, ``A[0, 0]``,
whose statements' sums run over concrete bounds and read tables at concrete
points, is then a claim about the inputs alone, which Lean proves with
Mathlib (:class:`AtTheClass`): ``A[0, 0]`` is ``2/3 * 2/3 * 1/6 * |J| + ...``
with each table at the value it holds, ``|J|`` is an atom to ``linarith``,
and the allowance is what it shows. That is a proof about the term read over
the reals: the floats the kernel produces round, which the native runs test.
So the fact rests on the arrays' ``single-assignment`` facts and on the
``trace-faithful`` fact, which says the term is what the body computes, and
it is worth ``tested``, as a theorem proved under a tested reading is.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

import islpy as isl
import numpy as np
import pymbolic.primitives as prim
from lanky.ledger import Fact, Status
from lanky.terms import Abs, Comparison, Subscript, conjuncts, init_args, render

from loopty.term import ArrType, Reduction, Term
from loopty.typing import AfterCall

__all__ = [
    "AtTheClass",
    "at_the_class",
    "comparison_class",
    "proved_over_the_reals",
    "statement",
    "tolerance_words",
]


# {{{ the class of an equation


def _numbers(term: Term) -> set[str]:
    """The kernel's arrays and scalars of reals or complex numbers."""
    from loopty.dataflow import _real

    out = set()
    for name, typ in (*term.params, *term.temporaries):
        sort = typ.dtype if isinstance(typ, ArrType) else typ
        base = sort.base if hasattr(sort, "base") else sort
        if _real(sort) or getattr(base, "name", None) == "Complex":
            out.add(name)
            continue
        try:
            if np.dtype(base).kind == "c":
                out.add(name)
        except TypeError:
            pass
    return out


def comparison_class(term: Term, comparison: Any) -> str:
    """The class an equation of ``term``'s numbers is compared at, ``exact`` if none.

    The weakest of the classes of the arrays it reads, an output's
    accumulations counted (:func:`loopty.tolerance.output_class`), and of the
    scalars it reads. ``exact`` for one that says the same of floats as of
    reals (:func:`loopty.dataflow.exact`), whatever it reads.
    """
    from loopty.dataflow import exact
    from loopty.hypotheses import _names
    from loopty.tolerance import element_class, output_class
    from loopty.trace import EXACTNESS_ORDER

    numbers = _numbers(term)
    names = _names(comparison) & numbers
    if not names or exact(comparison, numbers):
        return "exact"
    types = dict((*term.params, *term.temporaries))
    classes = ["exact"]
    for name in names:
        typ = types.get(name)
        if isinstance(typ, ArrType):
            classes.append(output_class(term, name))
        elif typ is not None:
            classes.append(element_class(typ))
    known = [c for c in classes if c in EXACTNESS_ORDER]
    return max(known, key=EXACTNESS_ORDER.index)


def _within(left: Any, right: Any, exactness: str) -> Any:
    """``|left - right| <= eps * (|right| + 1)``: the comparison of a class."""
    from loopty.tolerance import TOLERANCE, TOLERANCE_FLOOR

    return Comparison(
        Abs(left - right),
        "<=",
        TOLERANCE[exactness] * (Abs(right) + TOLERANCE_FLOOR),
    )


def _classes(term: Term, prop: Any) -> list[str]:
    """The classes of the equations of ``prop`` that are compared within one."""
    out = []
    for node in _walk(prop):
        if isinstance(node, prim.Comparison) and node.operator == "==":
            exactness = comparison_class(term, node)
            if exactness != "exact":
                out.append(exactness)
    return out


def _walk(expr: Any) -> list[Any]:
    out: list[Any] = []
    stack = [expr]
    while stack:
        node = stack.pop()
        if isinstance(node, tuple | list):
            stack.extend(node)
        elif isinstance(node, prim.ExpressionNode):
            out.append(node)
            stack.extend(init_args(node))
    return out


def at_the_class(term: Term, prop: Any) -> Any:
    """``prop`` with each equation of numbers that rounds read at its class.

    See the module. A proposition with none is returned as it is, the same
    object.
    """
    from loopty.math import _mapped

    def leaf(node: Any) -> Any:
        if isinstance(node, prim.Comparison) and node.operator == "==":
            exactness = comparison_class(term, node)
            if exactness != "exact":
                return _within(node.left, node.right, exactness)
            return node
        return None

    return _mapped(prop, leaf)


def tolerance_words(term: Term, prop: Any) -> str:
    """``, to the approx tolerance`` where an equation of ``prop`` is at its class."""
    classes = list(dict.fromkeys(_classes(term, prop)))
    return f", to the {' and '.join(classes)} tolerance" if classes else ""


def statement(term: Term, prop: Any) -> str:
    """The claim as the ledger says it: its text and the tolerance it is compared to."""
    return render(prop) + tolerance_words(term, prop)


# }}}


# {{{ proved over the reals


class _Unstated(Exception):
    """A claim at the class that is not stated for Lean, and why."""


def _concrete(value: Any) -> int | None:
    if isinstance(value, bool | np.bool_):
        return None
    if isinstance(value, int | np.integer):
        return int(value)
    return None


def _points(
    domain: Any, fixed: dict[str, int], inames: Sequence[str]
) -> list[dict[str, int]]:
    """The points of ``domain`` with the ``fixed`` dimensions at their values."""
    names = domain.get_var_names(isl.dim_type.set)
    found = domain
    for position, name in enumerate(domain.get_var_names(isl.dim_type.param)):
        if name not in fixed:
            raise _Unstated(f"a sum's bound is {name}, which has no value here")
        found = found.fix_val(isl.dim_type.param, position, fixed[name])
    for position, name in enumerate(names):
        if name in fixed:
            found = found.fix_val(isl.dim_type.set, position, fixed[name])
    if not found.is_bounded():
        raise _Unstated("a sum has no end")
    points: list[dict[str, int]] = []

    def take(point: Any) -> None:
        values = {
            name: int(point.get_coordinate_val(isl.dim_type.set, k).to_python())
            for k, name in enumerate(names)
        }
        points.append({name: values[name] for name in inames})

    found.foreach_point(take)
    points.sort(key=lambda point: [point[name] for name in inames])
    return points


def _expanded(expr: Any, values: dict[str, int], term: Term) -> Any:
    """``expr`` at ``values``: each sum written out and each table read at its point."""
    from loopty.hypotheses import substitute

    tables = dict(term.tables)

    def walk(node: Any) -> Any:
        if isinstance(node, Reduction):
            if node.op != "sum":
                raise _Unstated(f"a {node.op} is no sum of terms")
            # the sum is no pymbolic node, and its body is put at the point
            # with the loops around it
            pieces = [
                walk(substitute(node.body, {**values, **point}))
                for point in _points(node.domain, values, node.inames)
            ]
            if not pieces:
                return 0
            total = pieces[0]
            for piece in pieces[1:]:
                total = total + piece
            return total
        if isinstance(node, prim.Subscript) and isinstance(
            node.aggregate, prim.Variable
        ):
            index = node.index if isinstance(node.index, tuple) else (node.index,)
            index = tuple(walk(item) for item in index)
            if node.aggregate.name in tables:
                at = tuple(_concrete(item) for item in index)
                if any(item is None for item in at):
                    raise _Unstated(
                        f"{render(node)} reads a table at a point with no value"
                    )
                return float(np.asarray(tables[node.aggregate.name])[at])
            return Subscript(node.aggregate, index if len(index) > 1 else index[0])
        if isinstance(node, tuple):
            return tuple(walk(item) for item in node)
        if isinstance(node, prim.ExpressionNode):
            args = init_args(node)
            mapped = tuple(walk(arg) for arg in args)
            if all(a is b for a, b in zip(mapped, args, strict=True)):
                return node
            return type(node)(*mapped)
        return node

    return walk(substitute(expr, dict(values)))


def _cell_value(term: Term, cell: Any, written: set[str]) -> tuple[str, Any]:
    """The array of ``cell`` and what its statement leaves there, over the inputs.

    Raises:
        _Unstated: If no one statement stores the cell at a concrete instance
            from inputs alone, through its loops, its sums and tables.
    """
    array = cell.aggregate.name
    index = cell.index if isinstance(cell.index, tuple) else (cell.index,)
    at = tuple(_concrete(item) for item in index)
    if any(item is None for item in at):
        raise _Unstated(f"{render(cell)} is no cell at a point with a value")
    found = []
    for stmt in term.stmts:
        if (
            stmt.assignee.array != array
            or stmt.kind != "assign"
            or stmt.guard is not None
        ):
            continue
        indices = tuple(stmt.assignee.indices)
        if len(indices) != len(at):
            continue
        values: dict[str, int] = {}
        stored_here = True
        for item, value in zip(indices, at, strict=True):
            if isinstance(item, prim.Variable) and item.name in stmt.inames:
                if values.setdefault(item.name, value) != value:
                    stored_here = False
            elif _concrete(item) is not None:
                # a cell the statement names, y[0]
                stored_here = stored_here and _concrete(item) == value
            else:
                raise _Unstated(
                    f"{stmt.id} stores {array} at cells that are not its loops"
                )
        if not stored_here:
            continue
        if set(values) != set(stmt.inames):
            raise _Unstated(f"{stmt.id} stores {array} at cells that are not its loops")
        domain = stmt.domain
        if domain.dim(isl.dim_type.param):
            raise _Unstated(f"{stmt.id}'s loops run to a size, which has no value here")
        point = domain
        for position, name in enumerate(domain.get_var_names(isl.dim_type.set)):
            point = point.fix_val(isl.dim_type.set, position, values[name])
        if point.is_empty():
            continue
        from loopty.hypotheses import _names

        if _names(stmt.expr) & written:
            raise _Unstated(f"{stmt.id} reads what {term.name} writes")
        found.append(_expanded(stmt.expr, values, term))
    if len(found) != 1:
        raise _Unstated(f"no one statement of {term.name} stores {render(cell)}")
    return array, found[0]


def _cells(prop: Any, written: set[str]) -> list[Any]:
    """The cells of written arrays ``prop`` reads, each once, in order."""
    out: dict[str, Any] = {}
    for node in _walk(prop):
        if (
            isinstance(node, prim.Subscript)
            and isinstance(node.aggregate, prim.Variable)
            and node.aggregate.name in written
        ):
            out.setdefault(render(node), node)
    return list(out.values())


def _sorts(term: Term) -> dict[str, Any]:
    from lanky.prelude import Real

    from loopty.dataflow import _real

    return {
        name: Real
        for name, typ in term.params
        if _real(typ.dtype if isinstance(typ, ArrType) else typ)
    }


@dataclass(frozen=True)
class AtTheClass(AfterCall):
    """A postcondition of reals as its fact's term: run natively, proved over the reals.

    ``claim`` is the postcondition as written, and the fact's status is what
    the native runs found, at the class (:func:`at_the_class`), as an
    :class:`~loopty.typing.AfterCall` says. ``term`` is the kernel's, from
    which Lean is shown each conjunct at the class, the cells it reads
    replaced by what the kernel stores there (see the module).
    """

    term: Term | None = None

    def lean_obligation(self) -> Any:
        """One part per conjunct, about reals: Mathlib's (:mod:`lanky.obligations`)."""
        from loopty.obligations import Claim, _declined, _lanky_obligations

        lanky = _lanky_obligations()
        if "mathlib" not in getattr(lanky.Part, "__dataclass_fields__", {}):
            return _declined(
                "the claim is about real numbers, and this lanky states no part of "
                "Mathlib's"
            )
        term = self.term
        assert term is not None
        written = {stmt.assignee.array for stmt in term.stmts}
        sorts = _sorts(term)
        parts = []
        try:
            for k, conjunct in enumerate(conjuncts(self.claim)):
                if not isinstance(conjunct, prim.Comparison):
                    raise _Unstated(f"{render(conjunct)} is not a comparison")
                claim = Claim(term, sorts=sorts)
                for cell in _cells(conjunct, written):
                    _array, value = _cell_value(term, cell, written)
                    claim.hypotheses.append(Comparison(cell, "==", value))
                goal = at_the_class(term, conjunct)
                closed = claim.close(goal)
                parts.append(
                    lanky.Part(
                        f"conjunct_{k}",
                        render(conjunct),
                        closed,
                        lanky.Hints(),
                        mathlib=True,
                    )
                )
        except _Unstated as exc:
            return _declined(f"the claim is not stated for Lean: {exc}")
        extra = {}
        if "stated_by" in getattr(lanky.Obligation, "__dataclass_fields__", {}):
            extra["stated_by"] = "loopty"
        # no solver decided it, or found hints: the native runs tested it
        return lanky.Obligation(parts=tuple(parts), **extra)


def proved_over_the_reals(
    term: Term, post: Fact, stored: Any, faithful: str
) -> tuple[Fact, list[Fact]] | None:
    """``post`` with a term Lean can prove it from, and what it rests on, or ``None``.

    For a postcondition with an equation read at its class (:func:`at_the_class`)
    that the native runs did not refute: its cells are of arrays with a
    ``single-assignment`` fact (``stored``, :func:`loopty.dataflow.stores`),
    which it rests on, with the ``trace-faithful`` fact named ``faithful``.
    Whether Lean can state it is asked when Lean is (:class:`AtTheClass`).
    """
    claim = term.post
    if claim is None or post.status is Status.REFUTED or not _classes(term, claim):
        return None
    written = {stmt.assignee.array for stmt in term.stmts}
    arrays = list(
        dict.fromkeys(cell.aggregate.name for cell in _cells(claim, written))
    )
    if not arrays or not all(array in stored.facts for array in arrays):
        return None
    needed = [stored.facts[array] for array in arrays]
    fact = Fact(
        id=post.id,
        kind=post.kind,
        statement=post.statement,
        term=AtTheClass(term.name, claim, term),
        status=post.status,
        decided_by=post.decided_by,
        provenance={
            **post.provenance,
            "rule": (
                "each equation of reals is compared within its class's tolerance, "
                "as the compiled run is with the native one; Lean proves it of "
                "the term read over the reals, from what the kernel stores, no "
                "cell of " + ", ".join(arrays) + " stored twice or read before "
                "its store; the floats the kernel produces are what the native "
                "runs test"
            ),
        },
        where=post.where,
        owner=post.owner,
        rests_on=(*post.rests_on, *(fact.id for fact in needed), faithful),
    )
    return fact, needed


# }}}
