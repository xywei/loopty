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
from lanky.terms import (
    Abs,
    Comparison,
    Subscript,
    Var,
    conjuncts,
    init_args,
    render,
)

from loopty.term import ArrType, Reduction, Term
from loopty.typing import AfterCall

__all__ = [
    "AtTheClass",
    "ByTheorem",
    "at_the_class",
    "cited_over_the_reals",
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


def _expanded(
    expr: Any, values: dict[str, int], term: Term, read_tables: bool = True
) -> Any:
    """``expr`` at ``values``: each sum written out and each table read at its point.

    With ``read_tables`` false a table's cell stays a cell, ``FE0[0, 1]``,
    which a claim about any tables of the kernel's shape reads.
    """
    from loopty.hypotheses import substitute

    tables = dict(term.tables) if read_tables else {}

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


def _cell_value(
    term: Term, cell: Any, written: set[str], read_tables: bool = True
) -> tuple[str, Any]:
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
    from loopty.handles import statement_names

    named = statement_names(term)
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
                    f"{named(stmt.id)} stores {array} at cells that are not its loops"
                )
        if not stored_here:
            continue
        if set(values) != set(stmt.inames):
            raise _Unstated(
                f"{named(stmt.id)} stores {array} at cells that are not its loops"
            )
        domain = stmt.domain
        if domain.dim(isl.dim_type.param):
            raise _Unstated(
                f"the loops of {named(stmt.id)} run to a size, which has no value here"
            )
        point = domain
        for position, name in enumerate(domain.get_var_names(isl.dim_type.set)):
            point = point.fix_val(isl.dim_type.set, position, values[name])
        if point.is_empty():
            continue
        from loopty.hypotheses import _names

        if _names(stmt.expr) & written:
            raise _Unstated(f"{named(stmt.id)} reads what {term.name} writes")
        found.append(_expanded(stmt.expr, values, term, read_tables))
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


# {{{ by a cited theorem, under conditions on the kernel's tables


def _written_out(expr: Any) -> Any:
    """``expr`` with each sum over a literal range written out, term by term."""
    from lanky.terms import Sum as LankySum

    from loopty.hypotheses import substitute

    if isinstance(expr, LankySum):
        bounds = []
        for _var, sort in expr.binders:
            bound = getattr(sort, "bound", None)
            if expr.guard is not None or _concrete(bound) is None:
                body = _written_out(expr.body)
                return LankySum(expr.binders, body, expr.guard)
            bounds.append(_concrete(bound))
        pieces: list[Any] = []

        def walk(k: int, env: dict[str, int]) -> None:
            if k == len(bounds):
                pieces.append(_written_out(substitute(expr.body, dict(env))))
                return
            for value in range(bounds[k]):
                walk(k + 1, {**env, expr.binders[k][0].name: value})

        walk(0, {})
        if not pieces:
            return 0
        total = pieces[0]
        for piece in pieces[1:]:
            total = total + piece
        return total
    if isinstance(expr, prim.ExpressionNode):
        args = init_args(expr)
        mapped = tuple(_written_out(arg) for arg in args)
        if all(a is b for a, b in zip(mapped, args, strict=True)):
            return expr
        return type(expr)(*mapped)
    if isinstance(expr, tuple):
        return tuple(_written_out(item) for item in expr)
    return expr


def _flat(expr: Any) -> Any:
    """``expr`` with nested sums and products flat, and a sum's zero terms left out.

    lanky's ``sum`` of a literal range reads ``0 + a + b + c``, and what a
    kernel stores reads ``a + b + c``; flat, both are the sum of ``a``,
    ``b`` and ``c``, which match term by term.
    """
    if isinstance(expr, prim.Sum | prim.Product):
        kind = prim.Sum if isinstance(expr, prim.Sum) else prim.Product
        children: list[Any] = []
        for child in expr.children:
            child = _flat(child)
            if isinstance(child, kind):
                children.extend(child.children)
            else:
                children.append(child)
        if kind is prim.Sum:
            children = [
                child
                for child in children
                if not (isinstance(child, int | float) and child == 0)
            ]
        if not children:
            return 0 if kind is prim.Sum else 1
        if len(children) == 1:
            return children[0]
        return type(expr)(tuple(children))
    if isinstance(expr, prim.ExpressionNode):
        args = init_args(expr)
        mapped = tuple(_flat(arg) for arg in args)
        if all(a is b for a, b in zip(mapped, args, strict=True)):
            return expr
        return type(expr)(*mapped)
    if isinstance(expr, tuple):
        return tuple(_flat(item) for item in expr)
    return expr


@dataclass(frozen=True)
class ByTheorem(AfterCall):
    """A postcondition of reals that a cited theorem concludes, as its fact's term.

    One part per conjunct, about reals: the cells it reads are what the
    kernel stores there, its tables left as cells; the theorem is restated
    as it was proved; and its conditions at the kernel's tables are
    hypotheses, which rows of their own test at the tables' digits
    (:func:`cited_over_the_reals`). The claim is exact: the tables are what
    the theorem takes them to be, and the floats the kernel produces are
    what the native runs test, at the class.
    """

    term: Term | None = None
    theorem: Any = None
    conditions: tuple[Any, ...] = ()

    def lean_obligation(self) -> Any:
        from lanky.prelude import Real

        from loopty.obligations import Claim, _declined, _lanky_obligations, cited
        from loopty.rewriting import cited as statement_of

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
        sorts.update({name: Real for name, _ in term.tables})
        statement = statement_of(self.theorem)
        if statement is None:
            return _declined(f"{self.theorem!r} has no statement to restate")
        parts = []
        try:
            for k, conjunct in enumerate(conjuncts(self.claim)):
                if not isinstance(conjunct, prim.Comparison):
                    raise _Unstated(f"{render(conjunct)} is not a comparison")
                claim = Claim(term, sorts=sorts)
                for cell in _cells(conjunct, written):
                    _array, value = _cell_value(term, cell, written, read_tables=False)
                    claim.hypotheses.append(Comparison(cell, "==", value))
                claim.hypotheses.append(statement)
                claim.hypotheses.extend(self.conditions)
                closed = claim.close(conjunct)
                guards = [render(part) for part in conjuncts(closed.guard)]
                parts.append(
                    lanky.Part(
                        f"conjunct_{k}",
                        render(conjunct),
                        closed,
                        _unfolding(lanky, closed, render(statement)),
                        mathlib=True,
                        **cited(guards, (self.theorem,)),
                    )
                )
        except _Unstated as exc:
            return _declined(f"the claim is not stated for Lean: {exc}")
        extra = {}
        if "stated_by" in getattr(lanky.Obligation, "__dataclass_fields__", {}):
            extra["stated_by"] = "loopty"
        return lanky.Obligation(parts=tuple(parts), **extra)


def _unfolding(lanky: Any, closed: Any, statement: str) -> Any:
    """A :class:`ByTheorem` part's script: its equations rewritten with, then ``ring``.

    What the kernel stores in each cell the conjunct reads, and the
    theorem's conditions at the kernel's tables, ``FE0[0, 0] == 2/3``, are
    equations among the part's hypotheses; rewritten with in turn, they
    leave arithmetic of the tables' values and the kernel's inputs, which
    ``ring`` compares with the claim (``mass_ffc``'s ``A[0, 0]`` is
    ``|J| / 12``). ``linarith`` alone, which reads a product of a table and
    an input as an atom of its own, did not. The theorem's statement is
    left as it is. A lanky without rewrite steps gets no hints.
    """
    rewrite = getattr(lanky, "Rewrite", None)
    fields = getattr(lanky.Hints, "__dataclass_fields__", {})
    if rewrite is None or "rewrites" not in fields:
        return lanky.Hints()
    steps = tuple(
        rewrite(position)
        for position, part in enumerate(conjuncts(closed.guard))
        if isinstance(part, prim.Comparison)
        and part.operator == "=="
        and render(part) != statement
    )
    return lanky.Hints(rewrites=steps) if steps else lanky.Hints()


def _concluded(
    theorem: Any, term: Term, claim: Any, written: set[str]
) -> dict[str, Any] | None:
    """Where ``theorem`` concludes ``claim`` of what ``term`` stores: its bindings.

    Each conjunct of the claim, its cells read as what the kernel stores there
    with its tables left as cells, has to have the left side of an equation
    the theorem concludes, written out, with one binding of the theorem's
    variables for all of them: ``element_mass``'s ``FE`` is ``FE0``, its
    ``W`` is ``W3`` and its ``J`` the cell's Jacobian. The equation's right
    side at that binding has to be the conjunct's right side, as a term or
    at drawn values of what both read (:func:`_one_value`):
    ``abs(J) * 2 / 24`` is ``abs(jacobian(x)) / 12``, while a theorem that
    concludes something else of what the kernel stores concludes nothing of
    this claim. ``None`` where it does not.
    """
    from lanky.prelude import FnType

    from loopty.hypotheses import _Matcher
    from loopty.rewriting import _base

    goal = getattr(theorem, "goal", None)
    variables = dict(getattr(theorem, "variables", ()) or ())
    if goal is None or not variables:
        return None
    functions = {
        name for name, sort in variables.items() if isinstance(_base(sort), FnType)
    }
    scalars = set(variables) - functions
    equations = [
        part
        for part in conjuncts(_written_out(goal))
        if isinstance(part, prim.Comparison) and part.operator == "=="
    ]
    sigma: dict[str, Any] = {}
    for conjunct in conjuncts(claim):
        if not (isinstance(conjunct, prim.Comparison) and conjunct.operator == "=="):
            return None
        cells = _cells(conjunct.left, written)
        if len(cells) != 1 or cells[0] is not conjunct.left:
            return None
        try:
            _array, value = _cell_value(term, cells[0], written, read_tables=False)
        except _Unstated:
            return None
        value = _flat(value)
        if _cells(conjunct.right, written):
            # a right side that reads what the kernel writes is not about its inputs
            return None
        for equation in equations:
            trial = dict(sigma)
            matcher = _Matcher(functions, scalars)
            if not matcher.match(_flat(equation.left), value, trial, {}):
                continue
            concluded = _concluded_side(equation.right, trial, variables, functions)
            if _one_value(concluded, conjunct.right, _complex_names(term)):
                sigma = trial
                break
        else:
            return None
    return sigma


def _concluded_side(
    side: Any, sigma: dict[str, Any], variables: dict[str, Any], functions: set[str]
) -> Any:
    """A side of a theorem's equation at ``sigma``; ``None`` if a variable is free."""
    from lanky.terms import free_variables

    from loopty.unfolding import _instantiate

    if (set(variables) - set(sigma)) & set(free_variables(side)):
        return None
    bound = {
        name: (Var(value) if isinstance(value, str) else value)
        for name, value in sigma.items()
    }
    return _instantiate(side, bound, functions)


#: How many draws tell two values of the inputs apart.
_DRAWS = 8


def _complex_names(term: Term) -> frozenset[str]:
    """The kernel's arrays and scalars of complex numbers, by name."""
    from loopty.rewriting import _inexact

    out = set()
    for name, typ in (*term.params, *term.temporaries, *term.tables):
        dtype = _inexact(getattr(typ, "dtype", typ))
        if dtype is not None and dtype.kind == "c":
            out.add(name)
    return frozenset(out)


def _one_value(
    concluded: Any, claimed: Any, complex_names: frozenset[str] = frozenset()
) -> bool:
    """Whether a theorem's right side at its binding is the claim's: terms, or draws.

    Each cell at a point with a value, ``x[2, 1]``, and each free name is
    drawn at random, a rational number, the same in both, and both sides are
    evaluated in lanky's exact reading (:func:`lanky.terms.exact_reading`),
    where a float literal is the rational it holds; they have to be the same
    number at every draw, exactly, so ``x * (1 + 1e-13)`` is not taken for
    ``x``. The draws take both signs, all negative once, all positive once
    and mixed after, so ``abs(x)`` is not taken for ``x``; a cell of an array
    named in ``complex_names``, or a scalar so named, is drawn with an
    imaginary part as well, its parts a scaled Pythagorean pair so that its
    modulus is rational too, so ``abs(z)**2`` is not taken for ``z*z``. A
    side with a cell at a point with no value, one that cannot be evaluated
    at a draw, or one whose value there is no exact number (a square root
    that is not rational), is not shown to be the other, and ``False`` is
    returned.
    """
    from fractions import Fraction

    from lanky.intervals import ComplexValue
    from lanky.terms import evaluate, exact_reading, free_variables

    from loopty.hypotheses import _same

    if concluded is None:
        return False
    concluded, claimed = _flat(concluded), _flat(claimed)
    if _same(concluded, claimed):
        return True
    names: dict[str, Var] = {}
    complex_atoms = set(complex_names)

    def atoms(node: Any) -> Any:
        if isinstance(node, prim.Subscript) and isinstance(
            node.aggregate, prim.Variable
        ):
            index = node.index if isinstance(node.index, tuple) else (node.index,)
            if any(_concrete(item) is None for item in index):
                raise _Unstated(f"{render(node)} is no cell at a point with a value")
            atom = names.setdefault(render(node), Var(f"_cell_{len(names)}"))
            if node.aggregate.name in complex_names:
                complex_atoms.add(atom.name)
            return atom
        if isinstance(node, prim.ExpressionNode):
            return type(node)(*(atoms(arg) for arg in init_args(node)))
        if isinstance(node, tuple):
            return tuple(atoms(item) for item in node)
        return node

    rng = np.random.default_rng(20261018)

    def drawn(sign: int) -> Fraction:
        return Fraction(sign * int(rng.integers(1, 64)), int(rng.integers(1, 64)))

    try:
        left, right = atoms(concluded), atoms(claimed)
        free = sorted(set(free_variables(left)) | set(free_variables(right)))
        for draw in range(_DRAWS):
            signs = (
                [-1] * len(free)
                if draw == 0
                else [1] * len(free)
                if draw == 1
                else [int(sign) for sign in rng.choice([-1, 1], size=len(free))]
            )
            context: dict[str, Any] = {}
            for name, sign in zip(free, signs, strict=True):
                context[name] = drawn(sign)
                if name in complex_atoms:
                    # parts of a Pythagorean triple, scaled: off the real
                    # axis, with a modulus that is rational, so abs is exact
                    m = int(rng.integers(2, 9))
                    n = int(rng.integers(1, m))
                    scale = context[name]
                    one, other = scale * (m * m - n * n), scale * 2 * m * n
                    if int(rng.integers(2)):
                        one, other = other, one
                    context[name] = ComplexValue(one, int(rng.choice([-1, 1])) * other)
            with exact_reading():
                a, b = evaluate(left, context), evaluate(right, context)
            if not _exactly_equal(a, b):
                return False
    except Exception:  # noqa: BLE001 - what cannot be evaluated is not shown equal
        return False
    return True


def _exactly_equal(a: Any, b: Any) -> bool:
    """Whether two values of the exact reading are one number, both exact."""
    import numbers
    from fractions import Fraction

    from lanky.intervals import ComplexValue

    def parts(value: Any) -> tuple[Fraction, Fraction] | None:
        if isinstance(value, bool | np.bool_):
            return None
        if isinstance(value, numbers.Rational):
            return Fraction(value), Fraction(0)
        if isinstance(value, ComplexValue) and all(
            isinstance(part, numbers.Rational) and not isinstance(part, bool)
            for part in (value.real, value.imag)
        ):
            return Fraction(value.real), Fraction(value.imag)
        return None

    first, second = parts(a), parts(b)
    return first is not None and first == second


def cited_over_the_reals(
    term: Term,
    post: Fact,
    stored: Any,
    faithful: str,
    theorems: Sequence[Any],
    owner: str,
    identifier: Any,
) -> tuple[Fact, list[Fact], list[Fact]] | None:
    """``post`` concluded by a theorem the kernel cites, and the rows it rests on.

    ``@kernel(uses=[element_mass])`` on ``mass_ffc``: the theorem concludes
    the element's mass in closed form of any tables that are the P1 basis at
    the rule's points and its weights. Where its conclusion has the left side
    of each conjunct of the postcondition, the cells read as what the kernel
    stores (:func:`_concluded`), its hypotheses at those bindings are
    conditions on the kernel's tables, each a row tested at the tables'
    digits (:func:`loopty.rewriting.condition_fact`), and Lean proves the
    claim over the reals from them, the theorem and what the kernel stores
    (:class:`ByTheorem`). Returns the fact, the ``single-assignment`` facts
    it rests on, and the condition rows; ``None`` where no cited theorem
    concludes it.
    """
    from loopty.rewriting import (
        Condition,
        _condition_kind,
        condition_fact,
    )
    from loopty.unfolding import _instantiate

    claim = term.post
    if claim is None or post.status is Status.REFUTED or not _classes(term, claim):
        return None
    written = {stmt.assignee.array for stmt in term.stmts}
    arrays = list(
        dict.fromkeys(cell.aggregate.name for cell in _cells(claim, written))
    )
    if not arrays or not all(array in stored.facts for array in arrays):
        return None
    for theorem in theorems:
        try:
            sigma = _concluded(theorem, term, claim, written)
        except Exception:  # noqa: BLE001 - a theorem that does not conclude it
            sigma = None
        if sigma is None:
            continue
        from lanky.prelude import FnType

        from loopty.rewriting import _base

        variables = dict(theorem.variables)
        functions = {
            name for name, sort in variables.items() if isinstance(_base(sort), FnType)
        }
        bound = {
            name: (Var(value) if isinstance(value, str) else value)
            for name, value in sigma.items()
        }
        cell = _cells(claim, written)[0]
        stmt = next(
            other for other in term.stmts if other.assignee.array == cell.aggregate.name
        )
        conditions = []
        for name, prop in getattr(theorem, "hypotheses", ()):
            instance = _instantiate(prop, dict(bound), functions)
            conditions.append(
                Condition(
                    theorem=theorem.__name__,
                    hypothesis=name,
                    claim=instance,
                    kind=_condition_kind(instance, term, ()),
                    over=(),
                    stmt=stmt.id,
                    spelled=tuple(
                        zip(stmt.inames, stmt.spelled or stmt.inames, strict=False)
                    ),
                )
            )
        rows = [
            condition_fact(
                condition,
                term,
                None,
                identifier(f"cited-condition:{position}"),
                owner,
                post.where,
                theorem.__name__,
            )
            for position, condition in enumerate(conditions)
        ]
        needed = [stored.facts[array] for array in arrays]
        tables = _listed([name for name, _ in term.tables])
        fact = Fact(
            id=post.id,
            kind=post.kind,
            statement=f"{render(claim)}, by {theorem.__name__}",
            term=ByTheorem(
                term.name,
                claim,
                term,
                theorem,
                tuple(condition.claim for condition in conditions),
            ),
            status=post.status,
            decided_by=post.decided_by,
            provenance={
                **post.provenance,
                "rule": (
                    f"{theorem.__name__} concludes it of what the kernel stores, "
                    f"where its tables {tables} are as {theorem.__name__} takes "
                    "them, which a row tests at the tables' digits; Lean proves it "
                    "over the reals from them, no cell of "
                    + ", ".join(arrays)
                    + " stored twice or read before its store; the floats the "
                    "kernel produces are what the native runs test, at the class"
                ),
            },
            where=post.where,
            owner=post.owner,
            rests_on=(
                *post.rests_on,
                *(fact.id for fact in needed),
                faithful,
                getattr(theorem, "fact_id", theorem.__name__),
                *(row.id for row in rows),
            ),
        )
        return fact, needed, rows
    return None


def _listed(names: Sequence[str]) -> str:
    if len(names) <= 1:
        return "".join(names)
    return f"{', '.join(names[:-1])} and {names[-1]}"


# }}}
