"""A cited theorem applied to the terms a kernel adds: one row per series.

The design idea. A kernel cites theorems with ``@kernel(uses=[...])``, and a
theorem whose hypotheses are conjuncts of the kernel's postcondition gives
its conclusion at the kernel's arrays (:func:`loopty.dataflow.application_facts`).
A theorem about a series is about what a kernel adds rather than what it
leaves. ``line_taylor`` concludes ``all(c(n) == coefficient(-beta / alpha, n)
for n in Nat if n >= 1)``: a table that satisfies the line-Taylor recurrence
is the closed form. ``single_layer`` adds ``coefficient(ratio(alpha, beta),
k)`` for ``k`` from 1 to its order, at ``alpha = c[t] - y[s]`` and ``beta =
z[t] - c[t]``, with the ratio computed by the conjugate, one real division
where ``-beta / alpha`` takes three. An identity the kernel cites ties the
two, ``by_the_conjugate``: ``ratio(alpha, beta) == -beta / alpha``. So the
kernel's ledger has the row::

    single_layer adds line_taylor's c(n) for n in 1..5, for every c with
    away, first and ode, read over the complex numbers

*The theorems.* A theorem whose conclusion is ``all(c(n) == E for n in ...)``,
with ``c`` a family of its own that ``E`` does not name, is about a series,
whose ``n``-th term is ``E``. One whose conclusion is an equation ``L == R``
of scalar variables is an identity. ``E`` is rewritten by each identity whose
right side it holds, right to left: ``-beta / alpha`` in it becomes
``ratio(alpha, beta)``, the kernel's way.

*The terms.* The terms of each sum in the kernel's statements, put in the
theorems' spelling (:func:`loopty.math.as_stated`), are matched with the
rewritten ``E`` (:class:`loopty.hypotheses._Matcher`): a term of an
addition anywhere in the statement, its sign and factors included, or the
body of a kernel's sum. A node that is no term of a sum, ``(x * 0.5) ** 2``
in ``1 / (x * 0.5) ** 2``, is not one the kernel adds. A match binds the
theorem's scalar variables to expressions of the kernel and ``n`` to a whole
number the term writes: the trace writes a sum over ``Fin[ORDER + 1]`` out,
term by term. The number has to be a point of ``n``'s sort that the
conclusion's guard admits, and each binding of the sort the theorem, or the
identity, takes it at: a theorem about reals says nothing of complex
numbers. The matches in one statement at the same bindings are one row.

*The claim.* For every value of the family and of each variable nothing
bound, under the theorem's hypotheses and the identities' at the bindings,
each term the kernel adds is the family at its order: ``c(3) == -(ratio(c[t]
- y[s], z[t] - c[t]))**3 / 3``, for every ``t`` and ``s``. It is about the
term the kernel evaluates, read over the reals or the complex numbers, as a
theorem is; the floats the kernel produces round, and the row says nothing
of them, which it says in words. It is ``decided`` by ``theorem`` and rests
on the theorems' facts; with Mathlib, Lean checks it one part per term: the
identities rewritten in the goal, then the theorem applied, with the
bindings given by name, and its premises the claim's hypotheses or
arithmetic (:class:`lanky.obligations.Hints`).
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np
import pymbolic.primitives as prim
from lanky.ledger import Fact, Status, fact_id
from lanky.prelude import Complex, FnType, Real, Refined, Sort
from lanky.terms import Call, Comparison, Forall, Var, conjuncts, init_args, render

from loopty.term import ArrType, Stmt, Term

__all__ = ["Series", "SeriesTerms", "series_applications", "series_facts"]

#: The kind of the rows, a cited theorem's application
#: (:data:`loopty.dataflow.APPLIED`).
KIND = "by-theorem"

#: The most nodes of a statement's term a theorem's term is matched against.
_MOST_NODES = 20_000


# {{{ the theorems


@dataclass(frozen=True)
class _SeriesTheorem:
    """A theorem that concludes a series' terms: ``all(c(n) == E for n ...)``."""

    theorem: Any
    family: str
    binder: Var
    sort: Any
    term: Any
    guard: Any


@dataclass(frozen=True)
class _Identity:
    """A theorem whose conclusion is an equation of scalar variables: ``L == R``."""

    theorem: Any
    left: Any
    right: Any


def _families(theorem: Any) -> set[str]:
    return {
        name
        for name, sort in getattr(theorem, "variables", ())
        if isinstance(sort, FnType)
    }


def _series_theorem(theorem: Any) -> _SeriesTheorem | None:
    """``theorem`` read as one about a series' terms, or ``None``."""
    from loopty.hypotheses import _call_spine, _names

    goal = getattr(theorem, "goal", None)
    families = _families(theorem)
    if not isinstance(goal, Forall) or len(goal.binders) != 1:
        return None
    ((var, sort),) = goal.binders
    body = goal.body
    if not (isinstance(body, prim.Comparison) and body.operator == "=="):
        return None
    for left, right in ((body.left, body.right), (body.right, body.left)):
        spine = _call_spine(left)
        if spine is None or spine[0] not in families or len(spine[1]) != 1:
            continue
        (argument,) = spine[1]
        if not (isinstance(argument, prim.Variable) and argument.name == var.name):
            continue
        if _names(right) & families:
            # a term the family itself gives is no closed form
            continue
        return _SeriesTheorem(theorem, spine[0], var, sort, right, goal.guard)
    return None


def _identity(theorem: Any) -> _Identity | None:
    """``theorem`` read as an identity of its scalar variables, or ``None``."""
    goal = getattr(theorem, "goal", None)
    if _families(theorem) or not (
        isinstance(goal, prim.Comparison) and goal.operator == "=="
    ):
        return None
    return _Identity(theorem, goal.left, goal.right)


def _scalars(theorem: Any) -> set[str]:
    return {
        name
        for name, sort in getattr(theorem, "variables", ())
        if not isinstance(sort, FnType)
    }


# }}}


# {{{ the terms a kernel adds


@dataclass(frozen=True)
class Series:
    """A series theorem's terms, found in one statement of a kernel.

    ``bindings`` give the theorem's scalar variables the kernel's
    expressions, in the theorems' spelling; ``rewrites`` the identities
    used, each with the values of its variables; ``terms`` the order of
    each term the statement adds and the term, in order.
    """

    theorem: Any
    family: str
    binder: str
    stmt: Stmt
    bindings: tuple[tuple[str, Any], ...]
    rewrites: tuple[tuple[Any, tuple[tuple[str, Any], ...]], ...]
    terms: tuple[tuple[int, Any], ...]

    @property
    def orders(self) -> tuple[int, ...]:
        return tuple(order for order, _ in self.terms)


def _nodes(expr: Any) -> list[Any]:
    """Every node of ``expr``, outermost first, at most :data:`_MOST_NODES`.

    The body of a kernel's sum (:class:`loopty.term.Reduction`) is searched
    too, and the sum itself is no node: its terms are what it adds.
    """
    from loopty.term import Reduction

    out: list[Any] = []
    stack = [expr]
    while stack and len(out) < _MOST_NODES:
        node = stack.pop()
        if isinstance(node, tuple | list):
            stack.extend(reversed(node))
            continue
        if isinstance(node, Reduction):
            # a kernel's sum, whose body holds what it adds
            stack.append(node.body)
            continue
        if not isinstance(node, prim.ExpressionNode):
            continue
        out.append(node)
        stack.extend(reversed(init_args(node)))
    return out


def _addends(expr: Any) -> list[Any]:
    """What ``expr`` adds: each term of a sum in it, outermost first.

    A term of an addition, ``t`` in ``a + t``, its sign and its factors
    included (``a - t`` adds ``-t``), and the body of a kernel's sum, at any
    depth: ``single_layer``'s series is a sum inside ``real(...)``. A node
    that is no term of a sum, ``(x * 0.5) ** 2`` in ``1 / (x * 0.5) ** 2``,
    is no term the kernel adds, and no row says it is.
    """
    from loopty.term import Reduction

    out: list[Any] = []
    stack = [expr]
    seen = 0
    while stack and seen < _MOST_NODES:
        node = stack.pop()
        if isinstance(node, tuple | list):
            stack.extend(reversed(node))
            continue
        if isinstance(node, Reduction):
            if node.op == "sum":
                out.append(node.body)
            stack.append(node.body)
            continue
        if not isinstance(node, prim.ExpressionNode):
            continue
        seen += 1
        if isinstance(node, prim.Sum):
            out.extend(node.children)
        stack.extend(reversed(init_args(node)))
    return out


def _kernel_sorts(term: Term) -> dict[str, Any]:
    """Each of the kernel's arrays and scalars as Lean reads it, of numbers."""
    from lanky.prelude import Int

    from loopty.obligations import _array_sort

    out: dict[str, Any] = {}
    for name, typ in term.params:
        sort = _number_sort(typ) or Int
        out[name] = _array_sort(typ.ndim, sort) if isinstance(typ, ArrType) else sort
    return out


def _kind(expr: Any, types: dict[str, Any]) -> str | None:
    """``ℂ``, ``ℝ`` or ``Int``: what ``expr`` is a number of, in ``types``."""
    from lanky.lean import number_type

    try:
        return number_type(expr, types, mathlib=True)
    except Exception:  # noqa: BLE001 - what cannot be read is of no kind
        return None


def _of_its_sorts(
    series: _SeriesTheorem,
    sigma: dict[str, Any],
    used: Sequence[tuple[_Identity, dict[str, Any]]],
    added: Any,
    types: dict[str, Any],
) -> bool:
    """Whether each binding is of the sort the theorem takes it at.

    A theorem about reals says nothing of complex numbers, and one about
    complex numbers is applied by Lean only to complex ones: each of the
    theorem's scalars and each identity's is bound to an expression of its
    own sort, and the term the kernel adds is of the family's values.
    """
    from loopty.hypotheses import substitute

    def same(ours: Any, name: Any, scope: dict[str, Any]) -> bool:
        # a side whose kind cannot be read is not known to be the same
        kind = _kind(ours, types)
        return kind is not None and kind == _kind(name, scope)

    theirs = dict(series.theorem.variables)
    for name, value in sigma.items():
        if not same(value, Var(name), theirs):
            return False
    for identity, values in used:
        own = dict(identity.theorem.variables)
        for name, value in values.items():
            if not same(substitute(value, sigma), Var(name), own):
                return False
    return same(added, Call(Var(series.family), (0,)), theirs)


def _replaced(expr: Any, found: Any, by: Any) -> Any:
    """``expr`` with the node ``found``, that one object, replaced by ``by``."""
    if expr is found:
        return by
    if isinstance(expr, tuple):
        items = tuple(_replaced(item, found, by) for item in expr)
        return expr if all(a is b for a, b in zip(items, expr, strict=True)) else items
    if isinstance(expr, prim.ExpressionNode):
        args = init_args(expr)
        items = tuple(_replaced(arg, found, by) for arg in args)
        if all(a is b for a, b in zip(items, args, strict=True)):
            return expr
        return type(expr)(*items)
    return expr


#: The most ways of rewriting a series' term by the identities that are tried.
_MOST_PATTERNS = 64


def _patterns(
    series: _SeriesTheorem, identities: Sequence[_Identity]
) -> list[tuple[Any, tuple[tuple[_Identity, dict[str, Any]], ...]]]:
    """The series' term, and each way of rewriting it by the identities, right to left.

    An identity's right side may match more than one node of the term, and
    not every match is the one the kernel computes: ``-beta / alpha`` matches
    ``coefficient(-beta / alpha, n)`` itself, ``-(...)**n / n``, with
    ``alpha`` the order. So each match is a pattern of its own, with the
    values of the identity's variables there, and the kernel's term decides
    which say what it adds. The term unrewritten is the first.
    """
    from loopty.hypotheses import _Matcher, substitute

    patterns: list[tuple[Any, tuple[tuple[_Identity, dict[str, Any]], ...]]] = [
        (series.term, ())
    ]
    for identity in identities:
        matcher = _Matcher(set(), _scalars(identity.theorem))
        more = []
        for pattern, used in patterns:
            for node in _nodes(pattern):
                sigma: dict[str, Any] = {}
                if not matcher.match(identity.right, node, sigma, {}):
                    continue
                rewritten = _replaced(pattern, node, substitute(identity.left, sigma))
                more.append((rewritten, (*used, (identity, sigma))))
        patterns = (patterns + more)[:_MOST_PATTERNS]
    return patterns


def _order(value: Any, series: _SeriesTheorem) -> int | None:
    """``value`` as a point of the series' index that its guard admits, or ``None``."""
    from lanky.terms import evaluate

    if isinstance(value, bool | np.bool_) or not isinstance(value, int | np.integer):
        return None
    order = int(value)
    base = series.sort
    while isinstance(base, Refined):
        base = base.base
    if isinstance(base, Sort) and base.name == "Nat" and order < 0:
        return None
    if not (isinstance(base, Sort) and base.name in ("Nat", "Int")):
        return None
    if series.guard is not None:
        try:
            admitted = evaluate(series.guard, {series.binder.name: order})
        except Exception:  # noqa: BLE001 - a guard it cannot evaluate admits nothing
            return None
        if admitted is not True and not (
            isinstance(admitted, bool | np.bool_) and bool(admitted)
        ):
            return None
    return order


def series_applications(term: Term, theorems: Sequence[Any]) -> list[Series]:
    """The terms each cited series theorem gives, statement by statement.

    See the module. A theorem whose term, rewritten no way or any way the
    identities allow (:func:`_patterns`), matches nothing, or matches only
    at a number its index does not take, gives nothing.
    """
    from loopty.hypotheses import _Matcher, structural_key, substitute
    from loopty.math import as_stated

    series_theorems = [found for t in theorems if (found := _series_theorem(t))]
    identities = [found for t in theorems if (found := _identity(t))]
    types = _kernel_sorts(term)
    out: list[Series] = []
    for series in series_theorems:
        scalars = _scalars(series.theorem) | {series.binder.name}
        matcher = _Matcher(set(), scalars)
        patterns = _patterns(series, identities)
        for stmt in term.stmts:
            nodes = _addends(as_stated(stmt.expr))
            groups: dict[Any, dict[str, Any]] = {}
            for position, (pattern, used) in enumerate(patterns):
                for node in nodes:
                    sigma: dict[str, Any] = {}
                    if not matcher.match(pattern, node, sigma, {}):
                        continue
                    order = _order(sigma.pop(series.binder.name, None), series)
                    if order is None:
                        continue
                    if not _of_its_sorts(series, sigma, used, node, types):
                        continue
                    key = (
                        position,
                        tuple(
                            (name, structural_key(sigma[name]))
                            for name in sorted(sigma)
                        ),
                    )
                    group = groups.setdefault(
                        key, {"sigma": sigma, "used": used, "terms": {}}
                    )
                    group["terms"].setdefault(order, node)
            for group in groups.values():
                sigma = group["sigma"]
                rewrites = tuple(
                    (
                        identity,
                        tuple(
                            (name, substitute(value, sigma))
                            for name, value in values.items()
                        ),
                    )
                    for identity, values in group["used"]
                )
                out.append(
                    Series(
                        theorem=series.theorem,
                        family=series.family,
                        binder=series.binder.name,
                        stmt=stmt,
                        bindings=tuple(
                            (name, sigma[name])
                            for name, _sort in series.theorem.variables
                            if name in sigma
                        ),
                        rewrites=rewrites,
                        terms=tuple(sorted(group["terms"].items())),
                    )
                )
    return out


# }}}


# {{{ the claim, for Lean


def _number_sort(typ: Any) -> Any:
    """``Real`` or ``Complex`` for an array or a scalar of numbers, else ``None``."""
    dtype = typ.dtype if isinstance(typ, ArrType) else typ
    base = dtype
    while isinstance(base, Refined):
        base = base.base
    if isinstance(base, Sort):
        return base if base.name in ("Real", "Complex") else None
    try:
        kind = np.dtype(base).kind
    except TypeError:
        return None
    return Complex if kind == "c" else Real if kind == "f" else None


def _codomain(sort: Any) -> Any:
    while isinstance(sort, FnType):
        sort = sort.codomain
    return sort


def _over(term: Term, series: Series) -> str:
    """What the row reads the kernel's term over: the complex numbers or the reals."""
    sorts = [
        _number_sort(typ) for _name, typ in term.params
    ] + [
        _codomain(sort) for _name, sort in series.theorem.variables
    ]
    complex_ = any(getattr(sort, "name", None) == "Complex" for sort in sorts)
    return "the complex numbers" if complex_ else "the reals"


def _listed(names: Sequence[str]) -> str:
    if len(names) <= 1:
        return "".join(names)
    return f"{', '.join(names[:-1])} and {names[-1]}"


def _orders_text(orders: Sequence[int]) -> str:
    """``1..5`` for a run of whole numbers, else ``1, 3``."""
    if len(orders) > 2 and list(orders) == list(range(orders[0], orders[-1] + 1)):
        return f"{orders[0]}..{orders[-1]}"
    return ", ".join(str(order) for order in orders)


def _premises(series: Series) -> list[str]:
    """What the row assumes, by name: the theorem's hypotheses, then the identities'.

    An identity's hypothesis is a premise of each part Lean is handed, as the
    theorem's are, so the row names it too, as ``by_the_conjugate's away``,
    unless at its values it is one of the theorem's at the bindings, as
    ``by_the_conjugate``'s ``alpha != 0`` is ``line_taylor``'s ``away``.
    """
    from loopty.hypotheses import structural_key, substitute

    bound = dict(series.bindings)
    names: list[str] = []
    seen: set[Any] = set()
    for name, prop in getattr(series.theorem, "hypotheses", ()):
        names.append(name)
        seen.add(structural_key(substitute(prop, bound)))
    for identity, values in series.rewrites:
        own = getattr(identity.theorem, "__name__", "an identity")
        for name, prop in getattr(identity.theorem, "hypotheses", ()):
            key = structural_key(substitute(prop, dict(values)))
            if key not in seen:
                seen.add(key)
                names.append(f"{own}'s {name}")
    return names


def series_statement(owner: str, term: Term, series: Series) -> str:
    """The ledger's words for a row: the kernel, the theorem's terms, for what."""
    from loopty.hypotheses import held_under

    theorem = getattr(series.theorem, "__name__", "claim")
    bound = {name for name, _ in series.bindings}
    free = [name for name, _ in series.theorem.variables if name not in bound]
    under, given = held_under(series.theorem)
    hypotheses = [name for name in _premises(series) if name not in given]
    words = (
        f"{owner} adds {theorem}'s {series.family}({series.binder}) for "
        f"{series.binder} in {_orders_text(series.orders)}"
    )
    if under:
        words += f", {under}"
    if free:
        words += f", for every {_listed(free)}"
        if hypotheses:
            words += f" with {_listed(hypotheses)}"
    elif hypotheses:
        words += f", given {_listed(hypotheses)}"
    return f"{words}, read over {_over(term, series)}"


@dataclass(frozen=True)
class SeriesTerms:
    """The claim of a series row, as the term of its fact, for Lean.

    One part per term the kernel adds: the theorem and each identity as they
    were proved, and the theorem's hypotheses and the identities' at the
    bindings, imply that the family at the order is the kernel's term. Lean
    rewrites by the identities, applies the theorem with its variables given
    by name, and closes what is left with the hypotheses or arithmetic. A
    claim about reals or complex numbers is Mathlib's.

    The theorem's and the identities' hypotheses are citations
    (:func:`loopty.obligations.cited`): lanky restates each from the text
    its own command proved, not by printing it again.
    """

    term: Term
    series: Series
    owner: str

    def lean_obligation(self) -> Any:
        """The parts, one per order (see :mod:`lanky.obligations`)."""
        from loopty.hypotheses import _fresh, substitute
        from loopty.obligations import (
            Claim,
            _declined,
            _lanky_obligations,
            _obligation,
            cited,
        )

        lanky = _lanky_obligations()
        if "rewrites" not in getattr(lanky.Hints, "__dataclass_fields__", {}):
            return _declined(
                "this lanky's Lean oracle rewrites by no identity a hypothesis of a "
                "part is; lanky 0.3 does"
            )
        if "mathlib" not in getattr(lanky.Part, "__dataclass_fields__", {}):
            return _declined(
                "the claim is about numbers that are not integers, and this lanky "
                "states no part of Mathlib's"
            )
        series = self.series
        theorem = series.theorem
        theorem_term = getattr(theorem, "term", None)
        if theorem_term is None:
            return _declined("the claim has no statement to apply")
        sorts: dict[str, Any] = {}
        for name, typ in self.term.params:
            sort = _number_sort(typ)
            if sort is not None:
                sorts[name] = sort
        taken = {name for name, _ in self.term.params} | set(self.term.sizes)
        for stmt in self.term.stmts:
            taken |= set(stmt.inames)
        bound = dict(series.bindings)
        mapping: dict[str, Any] = dict(bound)
        for name, sort in theorem.variables:
            if name in bound:
                continue
            fresh = name if name not in taken else _fresh(name, taken)
            taken.add(fresh)
            mapping[name] = Var(fresh)
            sorts[fresh] = _codomain(sort)
        hypotheses = [
            substitute(prop, mapping)
            for _name, prop in getattr(theorem, "hypotheses", ())
        ]
        rewritten: list[tuple[Any, tuple[tuple[str, Any], ...]]] = []
        for identity, values in series.rewrites:
            identity_term = getattr(identity.theorem, "term", None)
            if identity_term is None:
                return _declined("an identity it rewrites by has no statement")
            given = dict(values)
            hypotheses += [
                substitute(prop, given)
                for _name, prop in getattr(identity.theorem, "hypotheses", ())
            ]
            rewritten.append((identity_term, values))
        family = mapping[series.family]
        theorems = (theorem, *(identity.theorem for identity, _ in series.rewrites))
        parts = []
        for order, added in series.terms:
            claim = Claim(self.term, sorts=sorts)
            claim.hypotheses.append(theorem_term)
            claim.hypotheses.extend(term for term, _ in rewritten)
            claim.hypotheses.extend(hypotheses)
            closed = claim.close(Comparison(Call(family, (order,)), "==", added))
            guards = [render(part) for part in conjuncts(closed.guard)]
            try:
                index = guards.index(render(theorem_term))
                rewrites = tuple(
                    lanky.Application(guards.index(render(term)), values)
                    for term, values in rewritten
                )
            except ValueError:
                return _declined("a claim is no hypothesis of its own part")
            hints = lanky.Hints(
                applications=(lanky.Application(index, tuple(mapping.items())),),
                rewrites=rewrites,
            )
            parts.append(
                lanky.Part(
                    f"term_{order}",
                    f"{getattr(theorem, '__name__', 'theorem')}'s "
                    f"{series.family}({order}) is what {self.owner} adds",
                    closed,
                    hints,
                    mathlib=True,
                    **cited(guards, theorems),
                )
            )
        return _obligation(
            parts,
            rests_on=tuple(
                getattr(t, "fact_id", "")
                for t in (
                    theorem,
                    *(identity.theorem for identity, _ in series.rewrites),
                )
            ),
        )

    def __str__(self) -> str:
        return series_statement(self.owner, self.term, self.series)


def series_facts(
    term: Term,
    theorems: Sequence[Any],
    owner: str,
    *,
    module: str | None = None,
    line: int | None = None,
) -> list[Fact]:
    """One row per series a cited theorem gives (:func:`series_applications`)."""
    from loopty.handles import statement_names

    named = statement_names(term)
    out = []
    seen: dict[str, int] = {}
    for series in series_applications(term, theorems):
        theorem = series.theorem
        name = getattr(theorem, "__name__", "theorem")
        identifier = getattr(theorem, "fact_id", name)
        identities = [identity.theorem for identity, _ in series.rewrites]
        shown = ", ".join(
            f"{var} = {render(value)}" for var, value in series.bindings
        )
        rule = (
            f"{named(series.stmt.id)} adds {render(series.terms[0][1])} and the "
            "terms like "
            f"it, which are {name}'s {series.family}({series.binder}) at {shown}"
        )
        if identities:
            rule += ", once " + _listed(
                [
                    getattr(identity, "__name__", "an identity")
                    for identity in identities
                ]
            ) + " rewrites the claim's term into the kernel's"
        detail = f"{identifier}:{series.stmt.id}"
        count = seen.get(detail, 0)
        seen[detail] = count + 1
        if count:
            # a second series of the theorem's in one statement, at other bindings
            detail = f"{detail}:{count}"
        out.append(
            Fact(
                id=fact_id(KIND, owner, module=module, line=line, detail=detail),
                kind=KIND,
                statement=series_statement(owner, term, series),
                term=SeriesTerms(term, series, owner),
                status=Status.DECIDED,
                decided_by="claim",
                provenance={
                    "theorem": identifier,
                    "identities": [
                        getattr(identity, "fact_id", "") for identity in identities
                    ],
                    "bindings": shown,
                    "orders": list(series.orders),
                    "terms": [render(added) for _order, added in series.terms],
                    "rule": rule,
                },
                where=series.stmt.where,
                owner=owner,
                rests_on=(
                    identifier,
                    *(getattr(identity, "fact_id", "") for identity in identities),
                ),
            )
        )
    return out


# }}}
