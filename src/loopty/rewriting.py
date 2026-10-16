"""Math steps of a schedule: what a statement computes, rewritten by a theorem.

The design idea. A kernel's body says what it computes, and a schedule says
how. Most steps of a schedule change only the order of the instances (a
cast) or where values live (a storage step), and the kernel computes the
same numbers in another order. A *math step* changes the expression: sumpy
divides through the conjugate, ``x * conj(alpha) * (1 / real(alpha *
conj(alpha)))``, where the math says ``x / alpha``, and grudge applies the
Kronecker product of three matrices one axis at a time where the math says
the dense sum. A math step is a "how" that needs a "why", so it cites a
theorem::

    as_sumpy_computes = Schedule(single_layer).rewrite(
        single_layer.out, by=(conjugate_form, log_of_the_modulus)
    )

reads "as sumpy computes it: the single layer, with ``out`` rewritten by the
conjugate form and the log of the modulus". Each theorem reads left to
right, from what the kernel says to what it should compute: wherever what
``out`` stores contains the theorem's left side, the schedule computes its
right side instead (:func:`rewrite_term`).

*Matching, by type.* The theorem's left side is matched against what the
statement stores, outermost first, in the theorem's spelling
(:func:`loopty.math.as_stated`): a theorem's variable binds only a term of
its sort, so ``conjugate_form``'s ``alpha``, a ``Complex``, binds ``c[t] -
y[s]`` and never the order ``k`` the trace wrote out, which is an integer,
in ``-(ratio**k) / k`` of the same shape. A family binds an array, or a
section of one (``Fr[d]`` of ``Fr[d, i, a]``), as a program's theorems are
matched (:class:`loopty.unfolding._Sections`). The theorems of ``by=(f,
g)`` apply in order, each once, to the statement as the one before left it,
and a theorem never rewrites its own output.

*Two terms.* The schedule keeps both: ``T``, the term the body traces, which
postconditions, cited theorems and ``trace-faithful`` stay about, and ``T'``,
the rewritten term it compiles. The loopy instruction is generated from
``T'`` by today's lowering, never edited beside it, so the Lean row and the
compiled code come from one term, and the compared run covers the lowering.

*What is checked, and by whom.* Each theorem's step is one row
(:class:`MathRewrite`): over the reals or the complex numbers, the statement
as the step found it equals the statement as it leaves it, wherever the
theorem's conditions hold. Lean proves it, one part per statement, with the
theorem restated as it was proved (:func:`cited`) and rewritten into the
goal under the sums too, each condition stated over the domain of the sum it
is under; without Lean the row reads ``tested``, from draws of ``T == T'``
on the term interpreter. The theorem's conditions are rows of their own,
split by kind (:class:`Condition`): a condition on indices is decided by
isl; one on data, ``c[t] - y[s] != 0``, is tested on the native runs and
never proved; one on a constant table is tested at the table's digits. The
class of the output becomes ``approx`` (an ``exact`` output is refused, as
``realize`` refuses it), and the compared row says how far the floats moved,
measured: on the example inputs, where a failure fails the row, and at
drawn magnitude extremes, where a failure is a note with the input, since a
rewrite like sumpy's changes the float range on purpose
(:func:`compared_fact`).

*What is never claimed.* No floating-point result is called proved: the
rows say "over the reals" or "over the complex numbers".
"""

from __future__ import annotations

import dataclasses
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

import islpy as isl
import numpy as np
import pymbolic.primitives as prim
from lanky.ledger import Fact, Status
from lanky.prelude import FinType, FnType, Refined
from lanky.rewrites import RewriteTerm
from lanky.terms import (
    Comparison,
    Forall,
    Var,
    conjuncts,
    init_args,
    render,
)
from lanky.terms import Sum as LankySum

from loopty.term import ArrType, Reduction, Stmt, Term

__all__ = [
    "KIND",
    "Condition",
    "IllegalRewrite",
    "Identity",
    "Match",
    "MathRewrite",
    "Rewritten",
    "cited",
    "identity_of",
    "rewrite_term",
]

#: The kind of a math step's equality row: lanky's rewrite shape
#: (:func:`lanky.rewrites.rewrite_fact`).
KIND = "rewrite"

#: The kind of a row that tests or decides a theorem's condition at the kernel.
CONDITION = "rewrite-condition"

#: The kind of the row that compares the compiled rewritten run with the native one.
COMPARED = "rewrite-compared"

#: The most nodes of a statement a theorem's side is matched against.
_MOST_NODES = 50_000


class IllegalRewrite(TypeError):
    """A math step loopty refuses, with the reason and the fix in the user's terms.

    ``fact`` is the ``refuted`` row the step would have carried, where there
    is one: a match that draws tell apart from what the kernel computes.
    """

    def __init__(self, message: str, fact: Any = None) -> None:
        super().__init__(message)
        self.fact = fact


# {{{ the theorems


def _base(sort: Any) -> Any:
    while isinstance(sort, Refined):
        sort = sort.base
    return sort


@dataclass(frozen=True, eq=False)
class Identity:
    """A theorem read as a rewrite rule: its left side, its right side, its conditions.

    ``variables`` are the theorem's, by name with their sorts; ``functions``
    the families among them (``Fn[...]``), which bind arrays or sections of
    arrays, and ``scalars`` the rest. ``hypotheses`` are its named
    conditions, which each match states at the kernel (:class:`Condition`).
    """

    theorem: Any
    name: str
    variables: tuple[tuple[str, Any], ...]
    functions: frozenset[str]
    scalars: frozenset[str]
    left: Any
    right: Any
    hypotheses: tuple[tuple[str, Any], ...]

    @property
    def fact_id(self) -> str:
        return getattr(self.theorem, "fact_id", self.name)


def identity_of(theorem: Any, text: str) -> Identity:
    """``theorem`` as a rule that turns its left side into its right side.

    Raises:
        IllegalRewrite: If it is no theorem whose conclusion is an equation,
            or its left side is a bare variable, which would match every term.
    """
    name = getattr(theorem, "__name__", None) or getattr(theorem, "qualname", None)
    goal = getattr(theorem, "goal", None)
    variables = getattr(theorem, "variables", None)
    if name is None or variables is None or goal is None:
        raise IllegalRewrite(
            f"{text}: by= takes theorems, each stated as 'what the kernel says == "
            f"what it should compute', and {theorem!r} is not one; decorate it "
            "with @theorem"
        )
    if not (isinstance(goal, prim.Comparison) and goal.operator == "=="):
        raise IllegalRewrite(
            f"{text}: {name} concludes {render(goal)}, which is no equation; a "
            "rewrite turns the left side of an equation into its right side"
        )
    variables = tuple(variables)
    functions = frozenset(
        var for var, sort in variables if isinstance(_base(sort), FnType)
    )
    scalars = frozenset(var for var, _ in variables) - functions
    left, right = goal.left, goal.right
    if isinstance(left, prim.Variable) and left.name in functions | scalars:
        raise IllegalRewrite(
            f"{text}: the left side of {name} is the variable {left.name}, which "
            "matches every term; state the theorem so that its left side says "
            "what the kernel computes"
        )
    unbound = _names(right) & (functions | scalars) - _names(left)
    if unbound:
        listed = ", ".join(sorted(unbound))
        raise IllegalRewrite(
            f"{text}: the right side of {name} names {listed}, which its left "
            "side does not bind, so a match would not say what to compute; "
            f"state {name} with {listed} on its left side too"
        )
    return Identity(
        theorem=theorem,
        name=name,
        variables=variables,
        functions=functions,
        scalars=scalars,
        left=left,
        right=right,
        hypotheses=tuple(getattr(theorem, "hypotheses", ())),
    )


def cited(theorem: Any) -> Any:
    """The statement a rewrite's Lean part restates ``theorem`` by, as a hypothesis.

    The seam for lanky's ``as_cited`` text (the proof-interface design's
    citation rule): where lanky offers the text its own command proved, a
    citing part restates that, byte for byte; until then, the theorem's term,
    as loopty's other restatements (``Applied``, ``SeriesTerms``,
    ``Unfolded``) restate it today, trusting the printer.
    """
    lanky_cited = getattr(theorem, "as_cited", None)
    if lanky_cited is not None and not callable(lanky_cited):
        return lanky_cited
    return getattr(theorem, "term", None)


# }}}


# {{{ names and kinds


def _names(expr: Any) -> set[str]:
    from loopty.hypotheses import _names as names

    return names(expr)


def _kernel_types(term: Term) -> dict[str, Any]:
    """Each array and scalar of ``term`` as Lean reads it, of numbers."""
    from loopty.summands import _kernel_sorts

    types = _kernel_sorts(term)
    for name, typ in term.temporaries:
        from loopty.obligations import _array_sort
        from loopty.summands import _number_sort

        sort = _number_sort(typ)
        if sort is not None:
            types[name] = _array_sort(typ.ndim, sort)
    for stmt in term.stmts:
        for iname in stmt.inames:
            types.setdefault(iname, _int())
    return types


def _int() -> Any:
    from lanky.prelude import Int

    return Int


def _kind(expr: Any, types: Mapping[str, Any]) -> str | None:
    """``ℂ``, ``ℝ`` or ``Int``: what ``expr`` is a number of, or ``None``."""
    from loopty.summands import _kind as kind

    return kind(expr, dict(types))


def _sort_kind(sort: Any) -> str | None:
    """What a theorem's scalar variable is a number of: ``ℂ``, ``ℝ`` or ``Int``."""
    base = _base(sort)
    if isinstance(base, FinType):
        return "Int"
    name = getattr(base, "name", None)
    return {"Complex": "ℂ", "Real": "ℝ", "Nat": "Int", "Int": "Int"}.get(name)


def _codomain(sort: Any) -> Any:
    sort = _base(sort)
    while isinstance(sort, FnType):
        sort = _base(sort.codomain)
    return sort


def _arity(sort: Any) -> int:
    sort = _base(sort)
    count = 0
    while isinstance(sort, FnType):
        count += 1
        sort = _base(sort.codomain)
    return count


def _number_name(sort: Any) -> str | None:
    """``Real`` or ``Complex`` for a sort or an element type of numbers."""
    from loopty.summands import _number_sort

    found = _number_sort(sort)
    return getattr(found, "name", None)


# }}}


# {{{ matching


@dataclass(frozen=True, eq=False)
class Condition:
    """A condition a theorem needs, at the kernel: tested, decided or proved by kind.

    ``claim`` is the theorem's hypothesis at a match, in the theorem's
    spelling, over the loops of the statement and the sums it is under,
    ``over``, each with its exclusive bound. ``kind`` is ``"index"`` for a
    claim of integers alone, which isl decides; ``"data"`` for one that
    reads an argument of the kernel, which is tested on the native runs and
    never proved; ``"table"`` for one that reads only constant tables,
    tested at the table's digits.
    """

    theorem: str
    hypothesis: str
    claim: Any
    kind: str
    over: tuple[tuple[str, Any], ...]
    stmt: str
    #: How the source names the statement's loops, ``(("x_0", "x"), ...)``.
    spelled: tuple[tuple[str, str], ...] = ()

    @property
    def shown(self) -> str:
        """The claim with the statement's loops as the source names them."""
        return render(_respelled(self.claim, self.spelled))

    @property
    def text(self) -> str:
        """``c[t] - y[s] != 0 for every t and every s``."""
        if not self.over:
            return self.shown
        names = dict(self.spelled)
        return f"{self.shown} for " + " and ".join(
            f"every {names.get(name, name)}" for name, _ in self.over
        )


def _respelled(expr: Any, spelled: Sequence[tuple[str, str]]) -> Any:
    """``expr`` with loops renamed as the source spells them, where none collide."""
    from loopty.hypotheses import _names as names_of
    from loopty.hypotheses import substitute

    names = {old: new for old, new in spelled if old != new}
    used = names_of(expr)
    names = {old: new for old, new in names.items() if old in used}
    if not names:
        return expr
    kept = used - set(names)
    if kept & set(names.values()) or len(set(names.values())) != len(names):
        return expr
    return substitute(expr, {old: Var(new) for old, new in names.items()})


@dataclass(frozen=True, eq=False)
class Match:
    """One place a theorem's left side was found, and what replaced it.

    ``old`` and ``new`` are in the kernel's spelling, the node of the
    statement and the term the schedule compiles in its place; ``stated``
    and ``replaced`` the same in the theorem's spelling, its sums lanky's.
    ``sigma`` binds the theorem's variables, in the theorem's spelling, and
    ``within`` names the sums of the statement the node is under, outermost
    first, each with its bound.
    """

    identity: Identity
    stmt: str
    old: Any
    new: Any
    stated: Any
    replaced: Any
    sigma: tuple[tuple[str, Any], ...]
    within: tuple[tuple[str, Any], ...]
    conditions: tuple[Condition, ...]


def _statement_outer(stmt: Stmt, term: Term | None = None) -> frozenset[str]:
    """Where the statement runs, conjunct by conjunct, and every size not negative.

    What a sum's domain repeats of where the statement runs, and the
    non-negativity of a size, which every claim about the term assumes
    (:class:`loopty.obligations.Claim`): :func:`loopty.unfolding.as_sums`
    leaves these out of the sum's guard.
    """
    from loopty.description import description_of

    names = set(stmt.domain.get_var_names(isl.dim_type.param))
    if term is not None:
        names |= set(term.sizes)
    sizes = frozenset(render(Comparison(Var(name), ">=", 0)) for name in names)
    described = description_of(stmt.domain)
    if described is None:
        return sizes
    dims = stmt.domain.get_var_names(isl.dim_type.set)
    predicate = described.predicate([Var(name) for name in dims])
    return sizes | frozenset(render(part) for part in conjuncts(predicate))


def _reduction_parts(reduction: Reduction) -> frozenset[str]:
    from loopty.description import description_of

    described = description_of(reduction.domain)
    if described is None:
        return frozenset()
    dims = reduction.domain.get_var_names(isl.dim_type.set)
    predicate = described.predicate([Var(name) for name in dims])
    return frozenset(render(part) for part in conjuncts(predicate))


def _bound_of(reduction: Reduction, iname: str) -> Any:
    """The exclusive upper bound of one of a sum's binders, as loopty wrote it."""
    from loopty.unfolding import Unstatable, _as_sum

    try:
        found = _as_sum(reduction, frozenset())
    except Unstatable:
        return None
    for var, sort in found.binders:
        if var.name == iname and isinstance(sort, FinType):
            return sort.bound
    return None


class _Walk:
    """One theorem matched against one statement, outermost first, and replaced."""

    def __init__(
        self,
        identity: Identity,
        stmt: Stmt,
        term: Term,
        types: Mapping[str, Any],
        replace: bool = True,
        pattern: Any = None,
    ) -> None:
        self.identity = identity
        self.stmt = stmt
        self.term = term
        self.types = types
        self.replace = replace
        self.pattern = identity.left if pattern is None else pattern
        self.other = identity.right if pattern is None else identity.left
        self.matches: list[Match] = []
        self.refused: list[str] = []
        self.seen = 0

    def run(self, expr: Any) -> Any:
        outer = _statement_outer(self.stmt, self.term)
        return self.visit(expr, (), outer)

    def visit(self, node: Any, within: tuple, outer: frozenset[str]) -> Any:
        from loopty.math import as_stated
        from loopty.unfolding import Unstatable, _as_sum

        self.seen += 1
        if self.seen > _MOST_NODES:
            return node
        if isinstance(node, tuple):
            items = tuple(self.visit(item, within, outer) for item in node)
            same = all(a is b for a, b in zip(items, node, strict=True))
            return node if same else items
        if isinstance(node, Reduction):
            if isinstance(self.pattern, LankySum) and node.op == "sum":
                try:
                    stated = _as_sum(as_stated(node), outer)
                except Unstatable:
                    stated = None
                if stated is not None:
                    found = self.attempt(node, stated, within, exactness=node.exactness)
                    if found is not None:
                        return found
            bounds = tuple((iname, _bound_of(node, iname)) for iname in node.inames)
            body = self.visit(
                node.body, (*within, *bounds), outer | _reduction_parts(node)
            )
            return node if body is node.body else dataclasses.replace(node, body=body)
        if isinstance(node, prim.ExpressionNode):
            if not isinstance(self.pattern, LankySum):
                found = self.attempt(node, as_stated(node), within)
                if found is not None:
                    return found
            args = init_args(node)
            new = tuple(self.visit(arg, within, outer) for arg in args)
            if all(a is b for a, b in zip(new, args, strict=True)):
                return node
            return type(node)(*new)
        return node

    def attempt(
        self, node: Any, stated: Any, within: tuple, exactness: str | None = None
    ) -> Any:
        """The replacement of ``node`` where the pattern matches it, or ``None``."""
        from loopty.unfolding import _instantiate, _Sections

        identity = self.identity
        outside = frozenset(name for name, _ in within)
        matcher = _Sections(set(identity.functions), set(identity.scalars), outside)
        sigma: dict[str, Any] = {}
        if not matcher.match(self.pattern, stated, sigma, {}):
            return None
        why = _of_its_sorts(identity, sigma, self.types, self.term)
        if why is not None:
            self.refused.append(why)
            return None
        if not self.replace:
            self.matches.append(
                Match(
                    identity,
                    self.stmt.id,
                    node,
                    node,
                    stated,
                    stated,
                    tuple(sigma.items()),
                    within,
                    (),
                )
            )
            return node
        replaced = _instantiate(self.other, sigma, set(identity.functions))
        exactness = exactness or _stored_class(self.term, self.stmt)
        new = _as_kernel(replaced, self.stmt, within, exactness, self.term)
        conditions = _conditions(identity, sigma, self.stmt, within, self.term)
        self.matches.append(
            Match(
                identity=identity,
                stmt=self.stmt.id,
                old=node,
                new=new,
                stated=stated,
                replaced=replaced,
                sigma=tuple(sigma.items()),
                within=within,
                conditions=conditions,
            )
        )
        return new


def _of_its_sorts(
    identity: Identity, sigma: Mapping[str, Any], types: Mapping[str, Any], term: Term
) -> str | None:
    """Why a binding is not of the sort the theorem takes it at, or ``None``.

    A theorem's scalar binds only a term of its kind (``ℂ``, ``ℝ`` or an
    integer), as Lean reads the kernel's term; a family binds an array, or a
    section of one, whose cells are numbers of its values' sort and which
    has as many axes left as the family takes arguments.
    """
    variables = dict(identity.variables)
    for name, value in sigma.items():
        sort = variables.get(name)
        if sort is None:
            continue
        if name in identity.functions:
            array = (
                value.aggregate.name
                if isinstance(value, prim.Subscript)
                else (value.name if isinstance(value, prim.Variable) else None)
            )
            typ = term.array_types.get(array) if array else None
            if not isinstance(typ, ArrType):
                return (
                    f"{name} of {identity.name} is a family, and matched "
                    f"{render(value)}"
                )
            fixed = (
                len(value.index)
                if isinstance(value, prim.Subscript) and isinstance(value.index, tuple)
                else (1 if isinstance(value, prim.Subscript) else 0)
            )
            if typ.ndim - fixed != _arity(sort):
                return (
                    f"{name} of {identity.name} takes {_arity(sort)} indices, and "
                    f"{render(value)} has {typ.ndim - fixed} left"
                )
            want = getattr(_codomain(sort), "name", None)
            got = _number_name(typ.dtype)
            if want in ("Real", "Complex") and want != got:
                return (
                    f"{name} of {identity.name} holds {want} values, and the cells of "
                    f"{array} are not"
                )
            continue
        want = _sort_kind(sort)
        if want is None:
            return (
                f"{name} of {identity.name} is of the sort {sort}, which nothing binds"
            )
        got = _kind(value, types)
        if got != want:
            return (
                f"{name} of {identity.name} is {_sort_word(sort)}, and "
                f"{render(value)}, which it matched, is "
                f"{_kind_word(got)}"
            )
    return None


def _sort_word(sort: Any) -> str:
    base = _base(sort)
    if isinstance(base, FinType):
        return f"a point of Fin[{render(base.bound)}]"
    name = getattr(base, "name", str(base))
    return f"{'an' if name[:1] in 'AEIOU' else 'a'} {name}"


def _kind_word(kind: str | None) -> str:
    return {
        "ℂ": "a complex number",
        "ℝ": "a real number",
        "Int": "an integer",
    }.get(kind, "of no sort loopty reads")


def _stored_class(term: Term, stmt: Stmt) -> str:
    """The class of the array ``stmt`` stores: the class a new sum in it adds at."""
    from loopty.tolerance import element_class

    typ = term.array_types.get(stmt.assignee.array)
    return element_class(typ.dtype) if isinstance(typ, ArrType) else "approx"


# }}}


# {{{ the right side, in the kernel's spelling


def _as_kernel(expr: Any, stmt: Stmt, within: tuple, exactness: str, term: Term) -> Any:
    """A theorem's side, instantiated, as the kernel spells it: calls, loopty's sums.

    lanky's functions become the calls the lowering knows
    (:func:`loopty.math.as_traced`), and each of lanky's sums a
    :class:`~loopty.term.Reduction` over the statement's loops and its own
    binders, under the sums the match is under (:func:`_sum_domain`).
    """
    from loopty.math import as_traced

    def visit(node: Any, enclosing: tuple) -> Any:
        if isinstance(node, LankySum):
            binders = []
            for var, sort in node.binders:
                base = _base(sort)
                if not isinstance(base, FinType):
                    raise IllegalRewrite(
                        f"the right side sums over {var.name} in {sort}, and a "
                        "kernel sums over Fin[n]"
                    )
                binders.append((var.name, base.bound))
            if node.guard is not None:
                raise IllegalRewrite(
                    "the right side sums under a condition, which a kernel's sum "
                    "states over its domain; state the theorem's sums over Fin[n]"
                )
            inner = (*enclosing, *binders)
            body = visit(node.body, inner)
            domain = _sum_domain(stmt, enclosing, tuple(binders), term)
            return Reduction(
                "sum",
                tuple(name for name, _ in binders),
                domain,
                body,
                exactness,
            )
        if isinstance(node, prim.ExpressionNode):
            args = init_args(node)
            new = tuple(visit(arg, enclosing) for arg in args)
            if all(a is b for a, b in zip(new, args, strict=True)):
                return node
            return type(node)(*new)
        if isinstance(node, tuple):
            return tuple(visit(item, enclosing) for item in node)
        return node

    return visit(as_traced(expr), within)


def _sum_domain(stmt: Stmt, enclosing: tuple, binders: tuple, term: Term) -> isl.Set:
    """The domain of a new sum: the statement's loops, then the sum's own binders.

    The statement's own constraints, each enclosing sum's binder bounded as a
    parameter, as the trace states a nested sum's domain, and ``0 <= b <
    bound`` for each of its binders, with the description loopty keeps on
    the sets it writes, which a Lean statement reads the sum back from.
    """
    from loopty.description import Description, Disjunct, describe, description_of
    from loopty.flow import _assemble, expr_text, free_names

    table = term.reflections
    reflected: dict[str, Any] = {}
    described = description_of(stmt.domain)
    dims = tuple(stmt.domain.get_var_names(isl.dim_type.set))
    if (
        described is not None
        and len(described.disjuncts) == 1
        and not described.renamings
    ):
        pieces = list(described.disjuncts[0].constraints)
        nonneg = list(described.disjuncts[0].nonneg)
        names = tuple(described.dims)
    else:
        raise IllegalRewrite(
            f"the statement {stmt.text or stmt.id} runs over a set isl computed, and "
            "a sum the rewrite adds is stated over the statement's own domain"
        )
    if names != dims:
        raise IllegalRewrite(
            f"the statement {stmt.text or stmt.id} names its loops apart from its "
            "domain, and a sum the rewrite adds is stated over that domain"
        )
    for name, bound in (*enclosing, *binders):
        if bound is None:
            raise IllegalRewrite(
                f"the sum over {name} has no bound loopty wrote, so a sum inside it "
                "has no domain to be stated over"
            )
        text = expr_text(bound, None, reflected, table)
        pieces.append(f"0 <= {name} < {text}")
        nonneg.extend(
            sorted(
                free_names(text) - {n for n, _ in (*enclosing, *binders)} - set(dims)
            )
        )
    all_dims = (*dims, *(name for name, _ in binders))
    free: set[str] = set()
    for piece in pieces:
        free |= free_names(piece)
    parameters = sorted(free - set(all_dims))
    nonneg = [name for name in dict.fromkeys(nonneg) if name in parameters]
    out = isl.Set(_assemble(list(all_dims), pieces, parameters, set(nonneg)))
    return describe(
        out,
        Description(
            dims=all_dims,
            names=all_dims,
            disjuncts=(Disjunct(tuple(pieces), tuple(nonneg)),),
            reflected=tuple(reflected.items()),
        ),
    )


# }}}


# {{{ the conditions


def _conditions(
    identity: Identity,
    sigma: Mapping[str, Any],
    stmt: Stmt,
    within: tuple,
    term: Term,
) -> tuple[Condition, ...]:
    """The theorem's conditions at a match, each with its kind and what it is over.

    Its hypotheses at the bindings, and for a scalar of a ``Fin[n]`` or a
    ``Nat`` sort, that the term it matched is a point of the sort.
    """
    from loopty.unfolding import _instantiate

    variables = dict(identity.variables)
    claims: list[tuple[str, Any]] = []
    for name, prop in identity.hypotheses:
        claims.append((name, _instantiate(prop, dict(sigma), set(identity.functions))))
    for name, value in sigma.items():
        sort = _base(variables.get(name))
        if isinstance(sort, FinType):
            bound = _instantiate(sort.bound, dict(sigma), set(identity.functions))
            claims.append(
                (
                    f"{name} in Fin",
                    _conjunction(
                        [Comparison(value, ">=", 0), Comparison(value, "<", bound)]
                    ),
                )
            )
        elif getattr(sort, "name", None) == "Nat":
            claims.append((f"{name} in Nat", Comparison(value, ">=", 0)))
    loops = [(iname, None) for iname in stmt.inames]
    out = []
    for name, claim in claims:
        if claim is True:
            continue
        mentioned = _names(claim)
        over = tuple(
            (iname, bound) for iname, bound in (*loops, *within) if iname in mentioned
        )
        out.append(
            Condition(
                theorem=identity.name,
                hypothesis=name,
                claim=claim,
                kind=_condition_kind(claim, term),
                over=over,
                stmt=stmt.id,
                spelled=tuple(
                    zip(stmt.inames, stmt.spelled or stmt.inames, strict=False)
                ),
            )
        )
    return tuple(out)


def _conjunction(parts: Sequence[Any]) -> Any:
    from loopty.description import conjunction

    return conjunction(list(parts))


def _condition_kind(claim: Any, term: Term) -> str:
    """``index``, ``table`` or ``data``: who settles a condition."""
    from loopty.unfolding import _read

    arrays = _read(claim)
    if not arrays:
        return "index"
    tables = {name for name, _ in term.tables}
    if arrays <= tables:
        return "table"
    return "data"


# }}}


# {{{ the rewrite of a term


@dataclass(frozen=True, eq=False)
class Kept:
    """An array ``keep=`` stores a nested sum in, and the statement that stores it.

    ``indices`` are the loops the statement runs over, as the term names
    them, and ``spelled`` as the source names them; ``stmt`` is the
    statement's id, and ``reads`` the arrays its sum reads.
    """

    name: str
    stmt: str
    indices: tuple[str, ...]
    spelled: tuple[str, ...]
    shape: tuple[Any, ...]


@dataclass(frozen=True, eq=False)
class Step:
    """One theorem's step: the statements as it found them and as it left them."""

    identity: Identity
    before: Term
    after: Term
    matches: tuple[Match, ...]


@dataclass(frozen=True, eq=False)
class Rewritten:
    """A rewrite of one array's statements by theorems, in order.

    ``before`` is the term the step was taken on, ``after`` the one it
    compiles, and ``steps`` each theorem's part of it. ``text`` is the step
    as written, ``rewrite(single_layer.out, by=conjugate_form)``, and
    ``array`` the array whose statements were rewritten.
    """

    array: str
    shown: str
    text: str
    before: Term
    after: Term
    steps: tuple[Step, ...]
    #: The arrays ``keep=`` stores the nested sums in, innermost first.
    kept: tuple[Kept, ...] = ()
    #: The program ``into=`` names as the target, or ``None``.
    into: Any = None
    #: The claim of a structural rewrite, for Lean: the kernel's statement as
    #: the postcondition of what the schedule compiles, proved by unfolding
    #: what its statements store (:class:`loopty.unfolding.Unfolded`), and the
    #: facts that claim rests on; ``None`` for a rewrite inside a statement.
    proof: Any = None

    @property
    def matches(self) -> tuple[Match, ...]:
        return tuple(match for step in self.steps for match in step.matches)

    @property
    def conditions(self) -> tuple[Condition, ...]:
        """Every condition of every match, once, with every theorem that needs it."""
        from loopty.hypotheses import structural_key

        found: dict[Any, Condition] = {}
        needs: dict[Any, list[str]] = {}
        index: dict[str, list[Condition]] = {}
        for match in self.matches:
            for condition in match.conditions:
                key = (condition.kind, structural_key(condition.claim))
                if key not in found and condition.kind == "index":
                    # a statement's conditions on indices are one row, as isl
                    # decides them in one question
                    index.setdefault(condition.stmt, []).append(condition)
                found.setdefault(key, condition)
                names = needs.setdefault(key, [])
                if condition.theorem not in names:
                    names.append(condition.theorem)
        out = []
        merged: set[str] = set()
        for key, condition in found.items():
            if condition.kind != "index":
                out.append(dataclasses.replace(condition, theorem=_listed(needs[key])))
                continue
            if condition.stmt in merged:
                continue
            merged.add(condition.stmt)
            parts = index[condition.stmt]
            theorems = list(
                dict.fromkeys(
                    name
                    for part in parts
                    for name in needs[("index", structural_key(part.claim))]
                )
            )
            over = tuple(
                dict.fromkeys(
                    (name, bound) for part in parts for name, bound in part.over
                )
            )
            out.append(
                dataclasses.replace(
                    condition,
                    claim=_conjunction([part.claim for part in parts]),
                    over=over,
                    theorem=_listed(theorems),
                )
            )
        return tuple(out)


def _listed(names: Sequence[str]) -> str:
    """``a``, ``a and b``, ``a, b and c``."""
    if len(names) <= 1:
        return "".join(names)
    return f"{', '.join(names[:-1])} and {names[-1]}"


def _occurs(
    identity: Identity, stmt: Stmt, term: Term, types: Mapping[str, Any]
) -> bool:
    """Whether the theorem's right side is in what ``stmt`` stores."""
    walk = _Walk(identity, stmt, term, types, replace=False, pattern=identity.right)
    walk.run(stmt.expr)
    return bool(walk.matches)


def rewrite_term(
    term: Term,
    array: str,
    identities: Sequence[Identity],
    text: str,
    shown: str,
) -> Rewritten:
    """``term`` with what ``array`` stores rewritten by each theorem in turn.

    Raises:
        IllegalRewrite: If ``term`` stores nothing in ``array``, or a theorem's
            left side is nowhere in what it stores: "state it the other way
            round" where its right side is, and the statement otherwise.
    """
    from loopty.handles import statement_names

    stmts = [stmt for stmt in term.stmts if stmt.assignee.array == array]
    if not stmts:
        raise IllegalRewrite(f"{text}: {term.name} stores nothing in {shown}")
    named = statement_names(term)
    types = _kernel_types(term)
    current = term
    steps: list[Step] = []
    for identity in identities:
        by_id = {stmt.id: stmt for stmt in current.stmts}
        matches: list[Match] = []
        refused: list[str] = []
        new_stmts = []
        for stmt in current.stmts:
            if stmt.assignee.array != array:
                new_stmts.append(stmt)
                continue
            walk = _Walk(identity, stmt, current, types)
            expr = walk.run(stmt.expr)
            matches.extend(walk.matches)
            refused.extend(walk.refused)
            new_stmts.append(
                stmt if expr is stmt.expr else dataclasses.replace(stmt, expr=expr)
            )
        if not matches:
            where = " and ".join(named(stmt.id) for stmt in stmts)
            if any(_occurs(identity, by_id[stmt.id], current, types) for stmt in stmts):
                raise IllegalRewrite(
                    f"{text}: {render(identity.right)}, the right side of "
                    f"{identity.name}, is in what {shown} stores, and its left side "
                    f"is not: state it the other way round, as 'what the kernel "
                    f"says == what it should compute' ({where})"
                )
            reason = ""
            if refused:
                reason = f"; it matched only where {refused[0]}"
            raise IllegalRewrite(
                f"{text}: {render(identity.left)}, the left side of {identity.name}, "
                f"is nowhere in what {shown} stores: {where}{reason}"
            )
        after = dataclasses.replace(current, stmts=tuple(new_stmts))
        steps.append(Step(identity, current, after, tuple(matches)))
        current = after
    return Rewritten(
        array=array,
        shown=shown,
        text=text,
        before=term,
        after=current,
        steps=tuple(steps),
    )


# }}}


# {{{ the equality row


def _over_words(term: Term, identity: Identity) -> str:
    """``the complex numbers`` or ``the reals``: what the row reads the terms over."""
    from loopty.summands import _number_sort

    sorts = [_number_sort(typ) for _name, typ in term.params]
    sorts += [_codomain(sort) for _name, sort in identity.variables]
    complex_ = any(getattr(sort, "name", None) == "Complex" for sort in sorts)
    return "the complex numbers" if complex_ else "the reals"


def _stated_statement(stmt: Stmt, term: Term) -> Any:
    """What ``stmt`` stores, in the theorems' spelling, its sums lanky's."""
    from loopty.math import as_stated
    from loopty.unfolding import as_sums

    return as_sums(as_stated(stmt.expr), _statement_outer(stmt, term))


@dataclass(frozen=True, eq=False)
class MathRewrite(RewriteTerm):
    """One theorem's step as the claim of its row, for Lean: ``T == T'`` where it holds.

    One part per statement the step rewrote: for every instance of the
    statement, wherever the theorem's conditions hold over the sums they are
    under, what it stored before the step equals what it stores after,
    over the reals or the complex numbers. The theorem is a hypothesis of the
    part, restated as it was proved (:func:`cited`), and Lean rewrites the
    goal with it, under the sums as well, each condition shown from the
    part's hypotheses and the range of the sum it is under.
    """

    step: Any = None
    owner: str = ""
    label: str = ""

    def lean_obligation(self) -> Any:
        from loopty.description import Unstatable
        from loopty.obligations import Claim, _declined, _lanky_obligations, _obligation
        from loopty.summands import _number_sort

        lanky = _lanky_obligations()
        if not hasattr(lanky, "Rewrite") or "mathlib" not in getattr(
            lanky.Part, "__dataclass_fields__", {}
        ):
            return _declined(
                "this lanky's Lean oracle rewrites no part's goal with its "
                "hypotheses; lanky 0.3 does"
            )
        step: Step = self.step
        theorem_term = cited(step.identity.theorem)
        if theorem_term is None:
            return _declined(f"{step.identity.name} has no statement to restate")
        before = {stmt.id: stmt for stmt in step.before.stmts}
        sorts: dict[str, Any] = {}
        for name, typ in (*step.after.params, *step.after.temporaries):
            sort = _number_sort(typ)
            if sort is not None:
                sorts[name] = sort
        parts = []
        rewritten = sorted({match.stmt for match in step.matches})
        for stmt_id in rewritten:
            old = before[stmt_id]
            new = step.after.stmt(stmt_id)
            claim = Claim(step.after, sorts=sorts)
            claim.coordinates(old.inames)
            claim.hypotheses.append(theorem_term)
            try:
                claim.hypotheses.extend(_domain_conditions(old))
                for condition in _stmt_conditions(step, stmt_id):
                    claim.hypotheses.append(_quantified(condition, old))
                goal = Comparison(
                    _stated_statement(old, step.before),
                    "==",
                    _stated_statement(new, step.after),
                )
                closed = claim.close(goal)
            except (Unstatable, IllegalRewrite) as exc:
                return _declined(str(exc))
            guards = [render(part) for part in conjuncts(closed.guard)]
            try:
                position = guards.index(render(theorem_term))
            except ValueError:
                return _declined("the theorem is no hypothesis of its own part")
            parts.append(
                lanky.Part(
                    f"rewrite_{stmt_id}",
                    f"{step.identity.name} rewrites what {self.owner}'s "
                    f"{old.assignee.array} stores, {old.text or stmt_id}",
                    closed,
                    lanky.Hints(rewrites=(lanky.Rewrite(position),)),
                    mathlib=True,
                )
            )
        return _obligation(parts, rests_on=(step.identity.fact_id,), decider="")

    def __str__(self) -> str:
        return self.label


def _domain_conditions(stmt: Stmt) -> list[Any]:
    """Where ``stmt`` runs, as hypotheses over its loops."""
    from loopty.description import Unstatable, description_of

    described = description_of(stmt.domain)
    if described is None:
        raise Unstatable(f"statement {stmt.id} runs over a set isl computed")
    dims = stmt.domain.get_var_names(isl.dim_type.set)
    return list(conjuncts(described.predicate([Var(name) for name in dims])))


def _stmt_conditions(step: Step, stmt_id: str) -> list[Condition]:
    from loopty.hypotheses import structural_key

    out: list[Condition] = []
    seen: set[Any] = set()
    for match in step.matches:
        if match.stmt != stmt_id:
            continue
        for condition in match.conditions:
            key = structural_key(condition.claim)
            if key not in seen:
                seen.add(key)
                out.append(condition)
    return out


def _quantified(condition: Condition, stmt: Stmt) -> Any:
    """A condition over the sums it is under, as a hypothesis of the statement's part.

    ``c[t] - y[s] != 0`` under the sum over ``s`` is ``all(c[t] - y[s] != 0
    for s in Fin[ns])``: the statement's loops are the part's own
    variables, and a sum's binder is quantified over the sum's range.
    """
    loops = set(stmt.inames)
    binders = [
        (Var(name), FinType(bound))
        for name, bound in condition.over
        if name not in loops
    ]
    if not binders:
        return condition.claim
    if any(sort.bound is None for _var, sort in binders):
        raise IllegalRewrite(
            f"{condition.text} is under a sum with no bound loopty wrote"
        )
    return Forall(tuple(binders), condition.claim, None)


# }}}


# {{{ the rows


def _drawn_equal(
    rewritten: Rewritten, step: Step, seed: int = 0
) -> tuple[bool | None, str]:
    """``T == T'`` at drawn inputs, on the interpreter: agreed, differed or unknown."""
    return _drawn_terms(
        step.before, step.after, rewritten.array, f"{step.identity.name}", seed
    )


def _drawn_terms(
    before: Term, after: Term, array: str, what: str, seed: int = 0
) -> tuple[bool | None, str]:
    """``before`` and ``after`` at drawn inputs agree in ``array``, on the interpreter.

    ``True`` when every draw agrees at the array's class, ``False`` with the
    first cell that differs, and ``None`` with why when nothing could be
    compared. ``what`` names the step in the words.
    """
    from loopty.faithful import _NoSample, sample_arguments
    from loopty.interpret import InterpretError, interpret
    from loopty.tolerance import disagreement, output_class

    exactness = output_class(before, array)
    if exactness == "exact":
        exactness = "approx"
    compared = 0
    for draw in range(3):
        try:
            arguments, _sizes = sample_arguments(before, seed=seed + draw)
        except _NoSample as exc:
            return None, f"no input could be drawn: {exc}"
        try:
            first = interpret(before, _copied(arguments), limit=2_000_000)
            second = interpret(after, _copied(arguments), limit=2_000_000)
        except (InterpretError, ArithmeticError, ValueError, KeyError) as exc:
            return None, f"the terms could not be interpreted: {exc}"
        want, got = first.get(array), second.get(array)
        if want is None or got is None:
            continue
        want, got = np.asarray(want), np.asarray(got)
        with np.errstate(all="ignore"):
            finite = np.isfinite(want) & np.isfinite(got)
            differs = np.asarray(disagreement(got, want, exactness)) & finite
        if differs.any():
            cell = tuple(int(k) for k in np.argwhere(differs)[0])
            return False, (
                f"at the inputs drawn with seed {seed + draw}, {array}{list(cell)} is "
                f"{want[cell]} before {what} and {got[cell]} after"
            )
        compared += 1
    if not compared:
        return None, "no drawn input was compared"
    return True, f"{compared} drawn inputs agree at {exactness}"


def _copied(arguments: Mapping[str, Any]) -> dict[str, Any]:
    from loopty.faithful import _copy

    return {name: _copy(value) for name, value in arguments.items()}


def _shortened(text: str, most: int = 72) -> str:
    return text if len(text) <= most else text[: most - 3] + "..."


def equality_facts(
    rewritten: Rewritten,
    owner: str,
    identifier: Any,
    where: str,
    condition_ids: Sequence[str],
) -> list[Fact]:
    """One row per theorem: over ℝ or ℂ, what the array stores is unchanged.

    ``identifier(detail)`` names a row; ``condition_ids`` are the rows of the
    data conditions the claim rests on, as Lean proves it under them.
    """
    from loopty.hypotheses import structural_key

    out = []
    for position, step in enumerate(rewritten.steps):
        identity = step.identity
        words = _equality_words(rewritten, step)
        term = MathRewrite(
            source=tuple(match.old for match in step.matches),
            target=tuple(match.new for match in step.matches),
            obligation="equal",
            step=step,
            owner=owner,
            label=words,
        )
        agreed, detail = _drawn_equal(rewritten, step)
        status = (
            Status.TESTED
            if agreed
            else Status.REFUTED
            if agreed is False
            else Status.ASSUMED
        )
        provenance: dict[str, Any] = {
            "theorem": identity.fact_id,
            "matches": len(step.matches),
            "detail": detail,
            "bindings": [
                ", ".join(f"{name} = {render(value)}" for name, value in match.sigma)
                for match in step.matches
            ],
        }
        if agreed is False:
            provenance["reason"] = (
                f"{rewritten.text}: the match of {identity.name} does not compute "
                f"what the kernel computes; {detail}"
            )
        if agreed is None:
            provenance["reason"] = detail
        keys = {
            structural_key(condition.claim)
            for match in step.matches
            for condition in match.conditions
        }
        rests = [
            identifier_
            for identifier_, condition in condition_ids
            if structural_key(condition.claim) in keys
        ]
        out.append(
            Fact(
                id=identifier(f"{KIND}:{position}:{identity.name}"),
                kind=KIND,
                statement=words,
                term=term,
                status=status,
                decided_by=None if status is Status.ASSUMED else "interpreter",
                provenance=provenance,
                where=where,
                owner=owner,
                rests_on=(identity.fact_id, *rests),
            )
        )
    return out


def _equality_words(rewritten: Rewritten, step: Step) -> str:
    """The row's statement, in the ledger's words.

    ``rewrite(out, by=conjugate_form): over the complex numbers, what out
    stores is unchanged wherever c[t] - y[s] != 0: -(z[t] - c[t]) / (c[t] -
    y[s]) becomes its conjugate form, in 5 places``.
    """
    from loopty.hypotheses import structural_key

    identity = step.identity
    first = step.matches[0]
    kinds: dict[Any, Any] = {}
    for match in step.matches:
        kinds.setdefault(structural_key(match.stated), match)
    conditions = []
    seen: set[Any] = set()
    for match in step.matches:
        for condition in match.conditions:
            key = structural_key(condition.claim)
            if key not in seen and condition.kind != "index":
                seen.add(key)
                conditions.append(condition.shown)
    count = len(step.matches)
    places = "in 1 place" if count == 1 else f"in {count} places"
    phrase = identity.name.replace("_", " ")
    olds = [_shortened(render(match.stated), 48) for match in kinds.values()]
    if len(olds) == 1:
        new = render(first.replaced)
        becomes = f"becomes {new if len(new) <= 60 else f'its {phrase}'}"
        said = olds[0]
    else:
        becomes = (
            f"each become its {phrase}"
            if phrase.endswith("s")
            else f"become their {phrase}s"
        )
        said = _listed(olds[:2] + ([f"{len(olds) - 2} more"] if len(olds) > 2 else []))
    wherever = f" wherever {' and '.join(conditions)}" if conditions else ""
    return (
        f"rewrite({rewritten.array}, by={identity.name}): over "
        f"{_over_words(step.before, identity)}, what {rewritten.array} stores is "
        f"unchanged{wherever}: {said} {becomes}, {places}"
    )


# }}}


# {{{ the conditions' rows


def _sizes_of(term: Term, arguments: Mapping[str, Any]) -> dict[str, int]:
    from loopty.faithful import _sizes_of as sizes_of

    return sizes_of(term, arguments)


def _values(arguments: Mapping[str, Any]) -> dict[str, Any]:
    """The arguments as numpy arrays and numbers, as lanky evaluates a claim at them."""
    out: dict[str, Any] = {}
    for name, value in arguments.items():
        numpy = getattr(value, "numpy", None)
        out[name] = numpy() if callable(numpy) else value
    return out


def _statement_points(stmt: Stmt, sizes: Mapping[str, int]) -> list[dict[str, int]]:
    """Every point of ``stmt``'s domain at ``sizes``, by its loops' names."""
    domain = stmt.domain
    for position in range(domain.dim(isl.dim_type.param)):
        name = domain.get_dim_name(isl.dim_type.param, position)
        if name in sizes:
            domain = domain.fix_val(isl.dim_type.param, position, int(sizes[name]))
    domain = (
        domain.project_out(isl.dim_type.param, 0, domain.dim(isl.dim_type.param))
        if all(
            domain.get_dim_name(isl.dim_type.param, k) in sizes
            for k in range(domain.dim(isl.dim_type.param))
        )
        else domain
    )
    if domain.dim(isl.dim_type.param):
        raise ValueError(
            "the statement's domain names a parameter the inputs do not fix: "
            + ", ".join(
                domain.get_dim_name(isl.dim_type.param, k)
                for k in range(domain.dim(isl.dim_type.param))
            )
        )
    names = stmt.domain.get_var_names(isl.dim_type.set)
    points: list[dict[str, int]] = []

    def take(point: Any) -> None:
        points.append(
            {
                name: int(point.get_coordinate_val(isl.dim_type.set, k).to_python())
                for k, name in enumerate(names)
            }
        )

    domain.foreach_point(take)
    return points


#: The most points a condition is evaluated at, on one input.
_MOST_POINTS = 2_000_000


def failing_point(
    condition: Condition,
    term: Term,
    arguments: Mapping[str, Any],
) -> tuple[bool, str]:
    """Whether ``condition`` holds at every point it is over, on ``arguments``.

    ``(True, "4752 points")`` when it does, and ``(False, words)`` naming the
    first point where it does not, as the body writes the claim.
    """
    from lanky.terms import evaluate

    stmt = term.stmt(condition.stmt)
    sizes = _sizes_of(term, arguments)
    values = _values(arguments)
    values.update(sizes)
    count = 0
    loops = set(stmt.inames)
    sums = [(name, bound) for name, bound in condition.over if name not in loops]

    def walk(env: dict[str, Any], rest: list) -> tuple[bool, str] | None:
        nonlocal count
        if not rest:
            count += 1
            if count > _MOST_POINTS:
                raise ValueError(f"more than {_MOST_POINTS} points to test")
            held = evaluate(condition.claim, env)
            if not (held is True or (isinstance(held, np.bool_) and bool(held))):
                names = dict(condition.spelled)
                at = ", ".join(
                    f"{names.get(name, name)}={env[name]}" for name, _ in condition.over
                )
                return False, f"{condition.shown} fails at {at}"
            return None
        (name, bound), tail = rest[0], rest[1:]
        extent = int(evaluate(bound, env))
        for value in range(extent):
            found = walk({**env, name: value}, tail)
            if found is not None:
                return found
        return None

    for point in _statement_points(stmt, sizes):
        found = walk({**values, **point}, sums)
        if found is not None:
            return found
    return True, f"{count} points"


def condition_fact(
    condition: Condition,
    term: Term,
    arguments: Mapping[str, Any] | None,
    identifier: str,
    owner: str,
    where: str,
    text: str,
) -> Fact:
    """The row of one condition: tested on the inputs, decided by isl, or at a table.

    ``arguments`` are the example inputs, which a data condition is tested
    on; without them the row is ``assumed`` and says so.
    """
    verb = "need" if " and " in condition.theorem else "needs"
    statement = f"{condition.text}, which {condition.theorem} {verb}"
    provenance: dict[str, Any] = {
        "theorem": condition.theorem,
        "hypothesis": condition.hypothesis,
        "condition": render(condition.claim),
        "kind": condition.kind,
    }

    def fact(status: Status, decided_by: str | None, words: str, **extra: Any) -> Fact:
        return Fact(
            id=identifier,
            kind=CONDITION,
            statement=words,
            term=None,
            status=status,
            decided_by=decided_by,
            provenance={**provenance, **extra},
            where=where,
            owner=owner,
        )

    if condition.kind == "index":
        decided, why = _decided_by_isl(condition, term)
        if decided is True:
            return fact(Status.DECIDED, "isl", statement, detail=why)
        if decided is False:
            return fact(Status.REFUTED, "isl", statement, reason=why, detail=why)
        if arguments is None:
            return fact(Status.ASSUMED, None, statement, reason=why)
    if condition.kind == "table":
        arguments = {**(arguments or {}), **dict(term.tables)}
        statement = f"{statement}, at the table's digits"
    elif arguments is None:
        return fact(
            Status.ASSUMED,
            None,
            f"{statement}, on the example inputs",
            reason=(
                "no example inputs to test it on: give the schedule some with "
                ".example(...), or the file an example_inputs()"
            ),
        )
    else:
        statement = f"{statement}, on the example inputs"
    try:
        held, detail = failing_point(condition, term, arguments)
    except Exception as exc:  # noqa: BLE001 - a row, never a crash of the check
        return fact(
            Status.ASSUMED,
            None,
            statement,
            reason=f"the condition could not be evaluated: {type(exc).__name__}: {exc}",
        )
    if held:
        return fact(Status.TESTED, "run", statement, detail=detail)
    return fact(Status.REFUTED, "run", statement, reason=detail, detail=detail)


def _decided_by_isl(condition: Condition, term: Term) -> tuple[bool | None, str]:
    """Whether isl shows a condition of indices on every point it is over."""
    from loopty.description import description_of
    from loopty.flow import NonAffine, _assemble, expr_text, free_names

    stmt = term.stmt(condition.stmt)
    described = description_of(stmt.domain)
    if described is None or len(described.disjuncts) != 1:
        return None, "the statement runs over a set isl computed"
    pieces = list(described.disjuncts[0].constraints)
    nonneg = set(described.disjuncts[0].nonneg)
    dims = list(described.dims)
    loops = set(stmt.inames)
    try:
        for name, bound in condition.over:
            if name in loops:
                continue
            dims.append(name)
            pieces.append(f"0 <= {name} < {expr_text(bound)}")
        negated = []
        for part in conjuncts(condition.claim):
            if not isinstance(part, prim.Comparison):
                return None, f"{render(part)} is no comparison isl reads"
            operator = {
                "==": "!=",
                "!=": "=",
                "<": ">=",
                "<=": ">",
                ">": "<=",
                ">=": "<",
            }[part.operator]
            if operator == "!=":
                left, right = expr_text(part.left), expr_text(part.right)
                negated.append(f"({left} < {right} or {left} > {right})")
            else:
                negated.append(
                    f"{expr_text(part.left)} {operator} {expr_text(part.right)}"
                )
    except (NonAffine, KeyError) as exc:
        return None, f"isl does not read {render(condition.claim)}: {exc}"
    free: set[str] = set()
    for piece in pieces:
        free |= free_names(piece)
    for piece in negated:
        free |= free_names(piece)
    params = sorted(free - set(dims))
    for piece in negated:
        text = _assemble(dims, [*pieces, piece], params, nonneg)
        failing = isl.Set(text)
        if not failing.is_empty():
            point = failing.sample_point()
            return False, (f"{render(condition.claim)} fails where isl finds {point}")
    return True, f"{render(condition.claim)} holds on every point, isl decides"


# }}}


# {{{ the compared row


#: The magnitudes the compared row draws the inputs at, beside the example's.
EXTREMES = (1e-170, 1e170)

#: How many inputs the compared row draws from the kernel's types, beside the example's.
DRAWS = 3


def _scaled(
    arguments: Mapping[str, Any], factor: float, written: set[str]
) -> dict[str, Any]:
    """The arguments with every floating array the kernel reads scaled by ``factor``."""
    from loopty.arr import Arr

    out: dict[str, Any] = {}
    for name, value in arguments.items():
        numpy = getattr(value, "numpy", None)
        array = numpy() if callable(numpy) else None
        if (
            name not in written
            and isinstance(array, np.ndarray)
            and array.dtype.kind in "fc"
            and getattr(value, "domain", None) is None
            and not getattr(value, "is_ragged", False)
        ):
            out[name] = Arr.from_numpy(array * factor)
        else:
            from loopty.faithful import _copy

            out[name] = _copy(value)
    return out


def _measured(got: np.ndarray, want: np.ndarray) -> float:
    """The largest difference of finite cells, over the largest native value."""
    with np.errstate(all="ignore"):
        finite = np.isfinite(got) & np.isfinite(want)
        if not finite.any():
            return 0.0
        largest = float(np.max(np.abs(want[finite])))
        difference = float(np.max(np.abs(got[finite] - want[finite])))
    return difference / largest if largest else difference


def _runs(
    kernel: Any, schedule: Any, arguments: Mapping[str, Any]
) -> tuple[dict, dict]:
    """The native run and the compiled one, on copies of ``arguments``."""
    from loopty.executor import LoopyExecutor
    from loopty.faithful import _copy

    native = {name: _copy(value) for name, value in arguments.items()}
    kernel(**native)
    compiled_args = {name: _copy(value) for name, value in arguments.items()}
    compiled = LoopyExecutor().run(schedule, **compiled_args)
    want = {}
    for name in compiled:
        value = native.get(name)
        numpy = getattr(value, "numpy", None)
        want[name] = np.asarray(numpy() if callable(numpy) else value)
    return {name: np.asarray(value) for name, value in compiled.items()}, want


def _first_disagreement(
    got: np.ndarray, want: np.ndarray, exactness: str
) -> tuple[int, ...] | None:
    from loopty.tolerance import disagreement

    differs = np.asarray(disagreement(got, want, exactness))
    if not differs.any():
        return None
    return tuple(int(k) for k in np.argwhere(differs)[0])


def compared_fact(
    rows: Any,
    schedule: Any,
    arguments: Mapping[str, Any] | None,
    identifier: str,
    owner: str,
    where: str,
    name: str,
) -> Fact:
    """The compiled rewritten schedule against the native kernel, measured.

    On the example inputs, where a disagreement fails the row, and with the
    inputs scaled to the magnitude extremes of :data:`EXTREMES`, where a
    cell the native run computes finite and the compiled one does not, or
    one beyond the class's tolerance, is a note with the input: the rewrite
    changes the float range, as sumpy's arithmetic does, and the row says so.
    """
    from loopty.executor import exactness_of_output

    kernel = rows.kernel
    after = rows.schedule
    array = rows.rewritten.array
    base = f"the compiled {name} agrees with the native {owner} in {array}"

    def fact(status: Status, words: str, **provenance: Any) -> Fact:
        return Fact(
            id=identifier,
            kind=COMPARED,
            statement=words,
            term=None,
            status=status,
            decided_by=None if status is Status.ASSUMED else "run",
            provenance=provenance,
            where=where,
            owner=owner,
        )

    if arguments is None:
        return fact(
            Status.ASSUMED,
            f"{base}, on its example inputs",
            reason="no example inputs to run it on",
        )
    if not callable(kernel):
        return fact(Status.ASSUMED, base, reason="no native body to compare with")
    try:
        got, want = _runs(kernel, after, arguments)
    except Exception as exc:  # noqa: BLE001 - a row, never a crash of the check
        return fact(
            Status.ASSUMED,
            f"{base}, on its example inputs",
            reason=f"the runs could not be made: {type(exc).__name__}: {exc}",
        )
    from loopty.tolerance import TOLERANCE

    outputs: dict[str, Any] = {}
    failures: list[str] = []
    for output in got:
        exactness = exactness_of_output(after.term, after, output)
        cell = _first_disagreement(got[output], want[output], exactness)
        measured = _measured(got[output], want[output])
        outputs[output] = {
            "exactness": exactness,
            "tolerance": TOLERANCE[exactness],
            "measured": measured,
            "agree": cell is None,
        }
        if cell is not None:
            failures.append(
                f"{output}{list(cell)} is {got[output][cell]} compiled and "
                f"{want[output][cell]} natively, beyond {exactness}"
            )
    exactness = outputs.get(array, {}).get("exactness", "approx")
    tolerance = TOLERANCE.get(exactness, 0.0)
    measured = outputs.get(array, {}).get("measured", 0.0)
    words = (
        f"{base} at {exactness}, within {tolerance:g} of each value (measured "
        f"{measured:.0e} of the largest), on its example inputs"
    )
    if failures:
        return fact(
            Status.REFUTED,
            words,
            outputs=outputs,
            reason="on the example inputs, " + "; ".join(failures),
        )
    notes: list[str] = []
    written = set(after.lowering.results)
    drawn = 0
    for seed in range(DRAWS):
        try:
            from loopty.faithful import _NoSample, sample_arguments

            arguments_d, _sizes = sample_arguments(rows.rewritten.before, seed=seed)
        except _NoSample:
            break
        try:
            got_d, want_d = _runs(kernel, after, arguments_d)
        except Exception:  # noqa: BLE001 - a draw the native body refuses is skipped
            continue
        drawn += 1
        for output in got_d:
            exactness = exactness_of_output(after.term, after, output)
            cell = _first_disagreement(got_d[output], want_d[output], exactness)
            if cell is not None:
                notes.append(
                    f"at the inputs drawn with seed {seed}, {output}{list(cell)} is "
                    f"{got_d[output][cell]} compiled and {want_d[output][cell]} "
                    f"natively, beyond {exactness}"
                )
    if drawn:
        words = words.replace(
            "on its example inputs", f"on its example inputs and {drawn} drawn ones"
        )
    for factor in EXTREMES:
        scaled = _scaled(arguments, factor, written)
        try:
            got_x, want_x = _runs(kernel, after, scaled)
        except Exception as exc:  # noqa: BLE001 - a note, never a failure
            notes.append(
                f"with the inputs scaled by {factor:g} the runs stop: "
                f"{type(exc).__name__}: {exc}"
            )
            continue
        for output in got_x:
            g, w = got_x[output], want_x[output]
            with np.errstate(all="ignore"):
                lost = np.isfinite(w) & ~np.isfinite(g)
            if lost.any():
                cell = tuple(int(k) for k in np.argwhere(lost)[0])
                notes.append(
                    f"with the inputs scaled by {factor:g}"
                    f"{rows.magnitude(scaled, cell)}, "
                    f"{output}{list(cell)} is {g[cell]} where the native run gives "
                    f"{w[cell]:.3g}"
                )
                continue
            exactness = exactness_of_output(after.term, after, output)
            cell = _first_disagreement(g, w, exactness)
            if cell is not None:
                notes.append(
                    f"with the inputs scaled by {factor:g}, {output}{list(cell)} is "
                    f"{g[cell]} compiled and {w[cell]} natively, beyond {exactness}"
                )
    if notes:
        words = f"{words}; note: " + "; ".join(notes)
    return fact(Status.TESTED, words, outputs=outputs, notes=notes, drawn=drawn)


# }}}


# {{{ the rows of one math step


class Rows:
    """The rows of one math step, made when they are first asked for.

    ``schedule`` is the schedule the step left, which the compared row
    compiles, ``kernel`` the kernel whose body is the reference, and
    ``where`` the line the step was taken at. The rows of a schedule built
    on from this one are the same rows: each is keyed by the steps up to the
    math step (:meth:`facts`).
    """

    def __init__(
        self,
        rewritten: Rewritten,
        schedule: Any,
        kernel: Any,
        where: str,
        identifier: Any,
        compared_id: str,
    ) -> None:
        self.rewritten = rewritten
        self.schedule = schedule
        self.kernel = kernel
        self.where = where
        self.identifier = identifier
        self.compared_id = compared_id
        self._made: dict[int, tuple[Fact, ...]] = {}

    def magnitude(self, arguments: Mapping[str, Any], cell: tuple[int, ...]) -> str:
        """`` (|c[t] - y[s]| about 1e-170)``: a data condition's operand there."""
        from lanky.terms import evaluate

        for condition in self.rewritten.conditions:
            if condition.kind != "data":
                continue
            claim = condition.claim
            if not (isinstance(claim, prim.Comparison) and claim.operator == "!="):
                continue
            subject = claim.left
            stmt = self.rewritten.before.stmt(condition.stmt)
            env = _values(arguments)
            env.update(_sizes_of(self.rewritten.before, arguments))
            for name, value in zip(stmt.inames, cell, strict=False):
                env[name] = value
            for name, _bound in condition.over:
                env.setdefault(name, 0)
            try:
                value = abs(complex(evaluate(subject, env)))
            except Exception:  # noqa: BLE001 - the words are a courtesy
                return ""
            return f" (|{render(subject)}| about {value:.0e})"
        return ""

    def facts(self, schedule: Any) -> tuple[Fact, ...]:
        arguments = _example_arguments(schedule, self.kernel)
        key = id(arguments) if arguments is not None else 0
        made = self._made.get(key)
        if made is None:
            made = self._make(schedule, arguments)
            self._made[key] = made
        return made

    def _make(
        self, schedule: Any, arguments: Mapping[str, Any] | None
    ) -> tuple[Fact, ...]:
        rewritten = self.rewritten
        owner = rewritten.before.name
        text = f"rewrite({rewritten.array}, ...)"
        conditions = []
        for position, condition in enumerate(rewritten.conditions):
            conditions.append(
                (
                    condition_fact(
                        condition,
                        rewritten.before,
                        arguments,
                        self.identifier(f"{CONDITION}:{position}"),
                        owner,
                        self.where,
                        text,
                    ),
                    condition,
                )
            )
        data = [
            (fact.id, condition)
            for fact, condition in conditions
            if condition.kind != "index"
        ]
        if rewritten.kept or rewritten.into is not None:
            equal = structural_facts(
                rewritten,
                owner,
                self.identifier,
                self.where,
                [identifier_ for identifier_, _condition in data],
            )
        else:
            equal = equality_facts(rewritten, owner, self.identifier, self.where, data)
        exactness = Fact(
            id=self.identifier(f"exactness:{rewritten.array}"),
            kind="exactness",
            statement=(
                f"after {text}, {rewritten.array} is compared at approx (it was "
                f"{self.was})"
            ),
            term=None,
            status=Status.DECIDED,
            decided_by="type",
            provenance={"detail": f"exactness of {rewritten.array} after a math step"},
            where=self.where,
            owner=owner,
        )
        name = _schedule_name(self, self.kernel) or (
            f"{rewritten.before.name} after {text}"
        )
        compared = compared_fact(
            self,
            schedule,
            arguments,
            self.compared_id,
            owner,
            self.where,
            name,
        )
        notes = _cost_notes(rewritten)
        if notes:
            compared = compared.with_status(compared.status, cost=notes)
        return (
            *equal,
            *(fact for fact, _condition in conditions),
            exactness,
            compared,
        )

    @property
    def was(self) -> str:
        from loopty.tolerance import output_class

        return output_class(self.rewritten.before, self.rewritten.array)


def _example_arguments(schedule: Any, kernel: Any) -> Mapping[str, Any] | None:
    """The example inputs of ``schedule``, or the file's ``example_inputs()``'s."""
    examples = getattr(schedule, "_examples", None)
    if examples is not None:
        return examples
    fn = getattr(kernel, "fn", None)
    globals_ = getattr(fn, "__globals__", None) or {}
    maker = globals_.get("example_inputs")
    if not callable(maker):
        return None
    cache = globals_.setdefault("__loopty_rewrite_examples__", {})
    if "inputs" not in cache:
        try:
            cache["inputs"] = maker()
        except Exception:  # noqa: BLE001 - no inputs, then
            cache["inputs"] = None
    inputs = cache["inputs"]
    if not isinstance(inputs, Mapping):
        return None
    name = getattr(kernel, "__name__", None)
    found = inputs.get(name)
    return found if isinstance(found, Mapping) else None


def _schedule_name(rows: Rows, kernel: Any) -> str | None:
    """The name the file binds the schedule the math step left to, if any."""
    fn = getattr(kernel, "fn", None)
    globals_ = getattr(fn, "__globals__", None) or {}
    for name, value in list(globals_.items()):
        steps = getattr(value, "_rewrites", None)
        if (
            not name.startswith("_")
            and steps
            and steps[-1] is rows
            and getattr(value, "_steps", ((None,),))[-1][0] == "rewrite"
        ):
            return name
    return None


def _cost_notes(rewritten: Rewritten) -> list[str]:
    """What a rewrite that nests sums and stores none costs each cell, and the fix.

    Without ``keep=``, a cell of the array still adds every term of every
    nested sum: the nesting saves nothing until the sums it nests are
    stored.
    """
    from loopty.term import Reduction as Kernels

    notes = []
    for match in rewritten.matches:
        depth = _nesting(match.new, Kernels)
        if depth > 1:
            notes.append(
                f"{match.identity.name} nests {depth} sums in what "
                f"{rewritten.array} stores and stores none, so each cell still "
                f"adds every term of them; keep=(...) stores the inner sums"
            )
            break
    return notes


def _nesting(expr: Any, kind: type) -> int:
    if isinstance(expr, kind):
        return 1 + _nesting(expr.body, kind)
    if isinstance(expr, prim.ExpressionNode):
        return max((_nesting(arg, kind) for arg in init_args(expr)), default=0)
    if isinstance(expr, tuple):
        return max((_nesting(item, kind) for item in expr), default=0)
    return 0


# }}}


# {{{ structural rewrites: keep= and into=


def _fold(exprs: Sequence[Any], binder: str) -> Any:
    """The one term ``exprs`` are at ``binder = 0, 1, ...``, or ``None``.

    The matches a sum the trace wrote out gives, one per term, differ only
    where the sum's binder was: ``Fr[0, i, a]``, ``Fr[1, i, a]``,
    ``Fr[2, i, a]`` are ``Fr[d, i, a]``. Every other place is the same in
    all of them.
    """
    from loopty.hypotheses import structural_key

    count = len(exprs)
    variable = Var(binder)

    def visit(items: list[Any]) -> Any:
        first = items[0]
        if all(isinstance(item, bool) for item in items):
            return first if len(set(items)) == 1 else _NO_FOLD
        if all(isinstance(item, int) and not isinstance(item, bool) for item in items):
            if len(set(items)) == 1:
                return first
            if items == list(range(count)):
                return variable
            return _NO_FOLD
        if all(isinstance(item, tuple) for item in items):
            if len({len(item) for item in items}) != 1:
                return _NO_FOLD
            out = tuple(visit([item[k] for item in items]) for k in range(len(first)))
            return _NO_FOLD if any(part is _NO_FOLD for part in out) else out
        if all(isinstance(item, prim.ExpressionNode) for item in items):
            if len({type(item) for item in items}) != 1:
                return _NO_FOLD
            args = [init_args(item) for item in items]
            out = visit([tuple(arg) for arg in args])
            if out is _NO_FOLD:
                return _NO_FOLD
            return type(first)(*out)
        keys = {structural_key(item) for item in items}
        return first if len(keys) == 1 else _NO_FOLD

    found = visit(list(exprs))
    return None if found is _NO_FOLD else found


#: What :func:`_fold` returns inside where the terms do not fold.
_NO_FOLD = object()


def _chain(expr: Any, depth: int) -> list[Any] | None:
    """The sums nested in ``expr``, outermost first, each a factor of the one outside.

    ``sum(Ft[k, c] * sum(Fs[j, b] * sum(Fr[i, a] * u[c, b, a] ...) ...) ...)``
    is three: the sum over ``c``, the one over ``b`` in its body, and the one
    over ``a`` in that. ``None`` when ``expr`` nests fewer than ``depth``
    such sums.
    """
    out = []
    node = expr
    while isinstance(node, LankySum) and len(out) < depth:
        out.append(node)
        inner = [
            factor for factor in _factors(node.body) if isinstance(factor, LankySum)
        ]
        if len(inner) != 1:
            break
        node = inner[0]
    return out if len(out) >= depth else None


def _factors(expr: Any) -> list[Any]:
    """The factors of a product, nested products flattened; ``[expr]`` otherwise."""
    if isinstance(expr, prim.Product):
        return [f for child in expr.children for f in _factors(child)]
    return [expr]


def _replaced_node(expr: Any, old: Any, new: Any) -> Any:
    """``expr`` with the node ``old``, that object, replaced by ``new``."""
    if expr is old:
        return new
    if isinstance(expr, LankySum):
        body = _replaced_node(expr.body, old, new)
        return expr if body is expr.body else LankySum(expr.binders, body, expr.guard)
    if isinstance(expr, prim.ExpressionNode):
        args = init_args(expr)
        items = tuple(_replaced_node(arg, old, new) for arg in args)
        if all(a is b for a, b in zip(items, args, strict=True)):
            return expr
        return type(expr)(*items)
    if isinstance(expr, tuple):
        items = tuple(_replaced_node(item, old, new) for item in expr)
        return expr if all(a is b for a, b in zip(items, expr, strict=True)) else items
    return expr


def _kept_indices(
    inner: LankySum, loops: Sequence[str], summed: Sequence[str] = ()
) -> list[str] | None:
    """The loops a kept sum is stored over, in the order the passes run.

    The cell of the array the sum reads along its binder, ``w[x, d, e, c, b,
    a]`` in ``sum(Fr[d, i, a] * w[x, d, e, c, b, a] for a ...)``, with the
    binder's place taken by the loop of the factor that runs along it,
    ``i``: the pass along ``r`` stores ``w`` with ``a`` replaced by ``i``.
    Any other loop the sum names is put after, in the statement's order.
    ``None`` when the sum's free names are not loops.
    """
    from loopty.hypotheses import _names as names_of

    ((binder, _sort),) = inner.binders
    free = [name for name in loops if name in names_of(inner.body)]
    cells = [
        factor
        for factor in _factors(inner.body)
        if isinstance(factor, prim.Subscript)
        and isinstance(factor.aggregate, prim.Variable)
        and binder.name in names_of(factor.index)
    ]
    if not cells:
        return free or None
    # the data the pass runs along: the cell that reads the most of the sums'
    # binders, w[x, d, e, c, b, a] beside Fr[d, i, a]
    binders = {*summed, binder.name}
    cell = max(
        cells,
        key=lambda found: (
            len(names_of(found.index) & binders),
            len(_index_tuple(found)),
        ),
    )
    indices = _index_tuple(cell)
    placed: list[str] = []
    for index in indices:
        if isinstance(index, prim.Variable):
            placed.append(index.name)
        elif not (isinstance(index, int) and not isinstance(index, bool)):
            return None
    others = [name for name in free if name not in placed]
    if binder.name in placed and len(others) == 1:
        placed[placed.index(binder.name)] = others[0]
        others = []
    placed = [name for name in placed if name != binder.name]
    out = [*placed, *others]
    return out if all(name in loops for name in out) else None


def _index_tuple(cell: prim.Subscript) -> tuple[Any, ...]:
    index = cell.index
    return index if isinstance(index, tuple) else (index,)


def _extent(name: str, stmt: Stmt, term: Term, bounds: Mapping[str, Any]) -> Any:
    """The extent of a loop of a kept array: a sum's bound, or an axis the loop indexes.

    A loop of the statement is as long as the axis it indexes on its own in
    the cell the statement stores, ``x`` of ``grad[x, e, k, j, i]`` the
    first axis of ``grad``.
    """
    if name in bounds:
        return bounds[name]
    types = term.array_types
    accesses = [stmt.assignee]
    for access in accesses:
        typ = types.get(access.array)
        if not isinstance(typ, ArrType) or typ.domain is not None:
            continue
        for position, index in enumerate(access.indices):
            if isinstance(index, prim.Variable) and index.name == name:
                return typ.axes[position]
    return None


def _fresh_name(stem: str, taken: set[str]) -> str:
    from loopty.hypotheses import _fresh

    name = stem if stem not in taken else _fresh(stem, taken)
    taken.add(name)
    return name


def keep_term(rewritten: Rewritten, keep: Any, text: str, at: str) -> Rewritten:
    """The rewrite with the sums its right side nests stored, ``keep=("r_pass", ...)``.

    Each name stores one nested sum, innermost first, over the loops it reads
    (:func:`_kept_indices`), in a statement of its own that runs before the
    rewritten one; the next sum out reads the kept array where it summed.
    The matches one written-out sum gave, one per term, are folded back into
    one nest over the sum's binder (:func:`_fold`). The kept arrays are the
    term's own, as a program's are, and each nest is a scope its handle
    names: ``factored.r_pass`` and ``factored.r_pass.c``.

    Raises:
        IllegalRewrite: If the names are not new names, the rewrite changed
            more than one statement, the theorem nests fewer sums than are
            kept, or the matches do not fold.
    """
    from loopty.flow import domain_set
    from loopty.hypotheses import substitute
    from loopty.term import Access, Binder, Callsite

    names = (keep,) if isinstance(keep, str) else tuple(keep)
    term = rewritten.after
    taken = _taken(term)
    for name in names:
        if not isinstance(name, str) or not name.isidentifier():
            raise IllegalRewrite(
                f"{text}: keep= names the arrays the nested sums are stored in, "
                f"and {name!r} is no name"
            )
        if name in taken or names.count(name) > 1:
            raise IllegalRewrite(
                f"{text}: keep= names a new array, and {term.name} has a {name} already"
            )
    taken |= set(names)
    by_stmt: dict[str, list[Match]] = {}
    for match in rewritten.matches:
        by_stmt.setdefault(match.stmt, []).append(match)
    if len(by_stmt) != 1:
        raise IllegalRewrite(
            f"{text}: keep= stores the sums of one statement's rewrite, and the "
            f"rewrite changed {len(by_stmt)}"
        )
    ((stmt_id, matches),) = by_stmt.items()
    stmt = term.stmt(stmt_id)
    original = rewritten.before.stmt(stmt_id)
    theorem = matches[0].identity.name
    folded, generic = None, matches[0].replaced
    if len(matches) > 1:
        generic = None
        for binder, count in stmt.unrolled:
            if count != len(matches):
                continue
            folded = _fresh_name(binder, taken)
            generic = _fold([match.replaced for match in matches], folded)
            if generic is not None:
                break
        if generic is None:
            raise IllegalRewrite(
                f"{text}: the {len(matches)} matches of {theorem} do not come "
                "from one written-out sum, so keep= has no one nest to store "
                "each sum in; rewrite without keep="
            )
    if _chain(generic, len(names) + 1) is None:
        raise IllegalRewrite(
            f"{text}: {theorem} nests fewer sums than keep= stores plus one; "
            f"keep= names the inner ones, innermost first, and names "
            f"{len(names)}"
        )
    loops = [*stmt.inames, *([folded] if folded else [])]
    spelled = dict(zip(stmt.inames, stmt.spelled or stmt.inames, strict=False))
    bounds: dict[str, Any] = {}
    if folded:
        bounds[folded] = len(matches)
        spelled[folded] = next(
            binder for binder, count in stmt.unrolled if count == len(matches)
        )
    for found in _chain(generic, len(names) + 1) or ():
        for var, sort in found.binders:
            bounds[var.name] = _base(sort).bound
            spelled.setdefault(var.name, var.name)
    exactness = _stored_class(rewritten.before, original)
    element = term.array_types[stmt.assignee.array].dtype
    where = at or stmt.where
    new_stmts: list[Stmt] = []
    kept: list[Kept] = []
    binders: list[Any] = []
    callsites: list[Any] = []
    temporaries: list[tuple[str, ArrType]] = []
    outer = generic
    for depth, name in enumerate(names):
        chain = _chain(outer, len(names) + 1 - depth)
        if chain is None:
            raise IllegalRewrite(f"{text}: the sums {theorem} nests do not nest here")
        inner = chain[-1]
        open_loops = [
            *loops,
            *(var.name for found in chain[:-1] for var, _ in found.binders),
        ]
        indices = _kept_indices(
            inner,
            open_loops,
            [var.name for found in chain for var, _ in found.binders],
        )
        if indices is None:
            raise IllegalRewrite(
                f"{text}: the sum keep= stores in {name} reads names that are no "
                "loops of the statement, so it has no cells to be stored in"
            )
        shape = []
        for index in indices:
            extent = _extent(index, original, rewritten.before, bounds)
            if extent is None:
                raise IllegalRewrite(
                    f"{text}: the loop {spelled.get(index, index)} of {name} has "
                    "no extent loopty can read off the statement"
                )
            shape.append(extent)
        fresh = {
            index: _fresh_name(spelled.get(index, index), taken) for index in indices
        }
        ((binder_var, binder_sort),) = inner.binders
        summed = _fresh_name(binder_var.name, taken)
        body = substitute(
            inner.body,
            {
                **{old: Var(new) for old, new in fresh.items()},
                binder_var.name: Var(summed),
            },
        )
        inames = tuple(fresh[index] for index in indices)
        stmt_new = Stmt(
            id=_fresh_name(f"{stmt.id}_{name}", taken),
            inames=inames,
            domain=domain_set(
                inames,
                tuple(shape),
                reflections=term.reflections,
            ),
            assignee=Access(name, tuple(Var(iname) for iname in inames)),
            expr=0,
            kind="assign",
            guard=None,
            where=where,
            # a block of its own, before the rewritten statement's
            order=(-1 - depth, *([0] * len(inames))),
            text=(
                f"{name}[{', '.join(spelled.get(i, i) for i in indices)}] = "
                "reduce_sum(...)"
            ),
            spelled=tuple(spelled.get(index, index) for index in indices),
        )
        expr = _as_kernel(
            LankySum(((Var(summed), binder_sort),), body, None),
            stmt_new,
            (),
            exactness,
            term,
        )
        new_stmts.append(dataclasses.replace(stmt_new, expr=expr))
        temporaries.append(
            (
                name,
                ArrType(axes=tuple(shape), dtype=element, ragged=(False,) * len(shape)),
            )
        )
        kept.append(Kept(name, stmt_new.id, inames, stmt_new.spelled, tuple(shape)))
        for index, iname in zip(indices, inames, strict=True):
            binders.append(
                Binder(iname, spelled.get(index, index), where, scope=(name,))
            )
        binders.append(Binder(summed, binder_var.name, where, sum=True, scope=(name,)))
        callsites.append(Callsite((name,), name, where))
        cell = prim.Subscript(Var(name), tuple(Var(index) for index in indices))
        outer = _replaced_node(outer, inner, cell)
    # the rewritten statement reads the outermost kept array where it summed
    expr = stmt.expr
    for position, match in enumerate(matches):
        here = outer if folded is None else substitute(outer, {folded: position})
        new = _as_kernel(here, stmt, match.within, exactness, term)
        expr = _replaced_identity(expr, match.new, new)
    ordered: list[Stmt] = []
    for other in term.stmts:
        if other.id == stmt_id:
            ordered.extend(new_stmts)
            ordered.append(dataclasses.replace(stmt, expr=expr))
        else:
            ordered.append(other)
    after = dataclasses.replace(
        term,
        stmts=_renumbered(ordered),
        temporaries=(*term.temporaries, *temporaries),
        binders=(*_binders(term), *binders),
        callsites=(*term.callsites, *callsites),
    )
    return dataclasses.replace(rewritten, after=after, kept=tuple(kept))


def _taken(term: Term) -> set[str]:
    """Every name ``term`` uses: arrays, sizes, loops and the binders of its sums."""
    from loopty.trace import reductions_in

    out = {name for name, _ in (*term.params, *term.temporaries, *term.tables)}
    out |= set(term.sizes) | {name for name, _ in term.reflected}
    for stmt in term.stmts:
        out |= set(stmt.inames)
        out.add(stmt.id)
        for reduction in reductions_in(stmt.expr):
            out |= set(reduction.inames)
    for binder in term.binders:
        out.add(binder.name)
    return out


def _binders(term: Term) -> tuple[Any, ...]:
    from loopty.handles import binders_of

    return binders_of(term)


def _replaced_identity(expr: Any, old: Any, new: Any) -> Any:
    """``expr``, a statement's term, with the node ``old`` replaced by ``new``."""
    if expr is old:
        return new
    if isinstance(expr, Reduction):
        body = _replaced_identity(expr.body, old, new)
        return expr if body is expr.body else dataclasses.replace(expr, body=body)
    if isinstance(expr, prim.ExpressionNode):
        args = init_args(expr)
        items = tuple(_replaced_identity(arg, old, new) for arg in args)
        if all(a is b for a, b in zip(items, args, strict=True)):
            return expr
        return type(expr)(*items)
    if isinstance(expr, tuple):
        items = tuple(_replaced_identity(item, old, new) for item in expr)
        return expr if all(a is b for a, b in zip(items, expr, strict=True)) else items
    return expr


def _renumbered(stmts: Sequence[Stmt]) -> tuple[Stmt, ...]:
    """The statements in sequence at the top of the loop tree, in this order.

    Each statement's position among the outermost blocks is the first entry
    of its :attr:`~loopty.term.Stmt.order`; a statement that shared a block
    with the one before keeps sharing it.
    """
    out: list[Stmt] = []
    block = -1
    previous = None
    for stmt in stmts:
        first = stmt.order[0] if stmt.order else None
        if previous is None or first != previous or first is None:
            block += 1
        previous = first
        order = (block, *stmt.order[1:]) if stmt.order else (block,)
        out.append(dataclasses.replace(stmt, order=order))
    return tuple(out)


def into_term(rewritten: Rewritten, target: Any, source: Any, text: str) -> Rewritten:
    """The rewrite with a program the file has as what the schedule compiles.

    ``rewrite(weak_grad.grad, by=sum_factorization, into=passes)``: the
    program takes the kernel's arguments, by name and in order, and its
    term is what the schedule compiles. The claim is that it leaves what the
    kernel's statement says, which today's unfolding of what its calls
    store proves (:func:`structural_proof`).

    Raises:
        IllegalRewrite: If ``target`` is no program, or does not take the
            kernel's arguments.
    """
    from loopty.kernel import Program

    if not isinstance(target, Program):
        raise IllegalRewrite(
            f"{text}: into= names a program the file has, which computes what "
            f"the kernel computes, and {target!r} is not one"
        )
    term = target.term
    kernel = rewritten.before
    mine = [(name, _arity_of(typ)) for name, typ in kernel.params]
    theirs = [(name, _arity_of(typ)) for name, typ in term.params]
    if mine != theirs:
        want = ", ".join(name for name, _ in kernel.params)
        got = ", ".join(name for name, _ in term.params)
        raise IllegalRewrite(
            f"{text}: {target.__name__} takes ({got}), and the schedule's program "
            f"takes what {kernel.name} takes, ({want}), each array of as many "
            "axes"
        )
    return dataclasses.replace(rewritten, after=term, into=target)


def _arity_of(typ: Any) -> int | None:
    return typ.ndim if isinstance(typ, ArrType) else None


def statement_claims(term: Term, arrays: Sequence[str]) -> list[Any]:
    """What ``term``'s statements store in ``arrays``, as claims about what it leaves.

    One per statement: at every instance, the cell it stores is what it
    computes, read over the reals or the complex numbers, the sums lanky's.
    The postcondition a structural rewrite's passes are proved to leave.
    """
    out = []
    for stmt in term.stmts:
        if stmt.assignee.array not in arrays:
            continue
        binders = tuple((Var(iname), _int()) for iname in stmt.inames)
        cell = prim.Subscript(Var(stmt.assignee.array), stmt.assignee.indices)
        body = Comparison(cell, "==", _stated_statement(stmt, term))
        guard = _conjunction(_domain_conditions(stmt))
        out.append(Forall(binders, body, None if guard is True else guard))
    return out


def structural_proof(
    rewritten: Rewritten,
    owner: str,
    identifier: Any,
) -> tuple[Any, tuple[Fact, ...], str]:
    """The claim of a structural rewrite for Lean, and the facts it rests on.

    The kernel's statement of the rewritten array is the postcondition of
    what the schedule compiles, and :func:`loopty.unfolding.unfolded`
    proves it from what that term's statements store and the theorems, as a
    program's postcondition is proved (R4): unfold the kept arrays (or the
    program's calls), rewrite each instance of a theorem's side, write the
    literal sums out, ``ring``. Returns the term of the row (``None`` with
    the reason when nothing unfolds), the ``single-assignment`` facts the
    equations rest on, and words for the provenance.
    """
    from loopty.dataflow import _stored, program_stores
    from loopty.unfolding import unfolded

    after = rewritten.after
    claims = statement_claims(rewritten.before, (rewritten.array,))
    post = _conjunction(claims)
    theorems = [step.identity.theorem for step in rewritten.steps]
    if rewritten.into is not None:
        theorems += [
            theorem
            for theorem in getattr(rewritten.into, "uses", ())
            if all(theorem is not seen for seen in theorems)
        ]
        stores = program_stores(after, owner)
    else:
        stores = _stored(after, owner)
    found = unfolded(dataclasses.replace(after, post=post), stores.equations, theorems)
    if isinstance(found, str):
        return None, (), found
    proof, used, words = found
    needed = {identifier_ for hypothesis in used for identifier_ in hypothesis.rests_on}
    stored = sorted(
        (fact for fact in stores.facts.values() if fact.id in needed),
        key=lambda fact: fact.id,
    )
    renamed = {
        fact.id: identifier(f"single-assignment:{fact.provenance.get('array', k)}")
        for k, fact in enumerate(stored)
    }
    facts = tuple(
        dataclasses.replace(fact, id=renamed[fact.id], owner=owner) for fact in stored
    )
    rests = tuple(renamed.get(fact, fact) for fact in proof.rests_on)
    return dataclasses.replace(proof, rests_on=rests), facts, words


def structural(
    rewritten: Rewritten,
    source: Any,
    *,
    keep: Any,
    into: Any,
    text: str,
    shown: str,
    at: str = "",
) -> Rewritten:
    """The rewrite with its nested sums stored (``keep=``), or into a program."""
    if keep is not None and into is not None:
        raise IllegalRewrite(
            f"{text}: keep= derives the passes and into= names them; give one"
        )
    if into is not None:
        return into_term(rewritten, into, source, text)
    return keep_term(rewritten, keep, text, at)


def _spelled_in(expr: Any, stmt: Stmt) -> Any:
    """``expr`` with the statement's loops as the source names them."""
    from loopty.hypotheses import substitute

    names = dict(zip(stmt.inames, stmt.spelled or stmt.inames, strict=False))
    return substitute(expr, {old: Var(new) for old, new in names.items() if old != new})


def _structural_words(rewritten: Rewritten) -> str:
    """The structural row's statement: what the passes leave, and how it is shown."""
    first = rewritten.steps[0]
    stmt = rewritten.before.stmt(first.matches[0].stmt)
    owner = rewritten.before.name
    head = _short_text(rewritten)
    over = _over_words(rewritten.before, first.identity)
    if rewritten.into is not None:
        what = f"{rewritten.into.__name__} leaves"
        through = "its calls"
    else:
        what = "the passes leave"
        through = _listed([kept.name for kept in rewritten.kept])
    matches = first.matches
    sigma = [dict(match.sigma) for match in matches]
    shown: list[str] = []
    binder = None
    if len(matches) > 1:
        binder = next(
            (name for name, count in stmt.unrolled if count == len(matches)), None
        )
    for name in first.identity.functions:
        values = [found.get(name) for found in sigma]
        if any(value is None for value in values):
            continue
        value = values[0] if binder is None else _fold(values, binder)
        if value is None:
            continue
        shown.append(f"{name} := {render(_spelled_in(value, stmt))}")
    at = f"{first.identity.name} at {', '.join(sorted(shown))}" if shown else ""
    places = ""
    if binder is not None:
        places = f", for {binder} = " + _listed([str(k) for k in range(len(matches))])
    return (
        f"{head}: over {over}, {what} in {rewritten.array} what {owner}'s statement "
        f"says, unfolded through {through}" + (f"; {at}{places}" if at else "")
    )


def _short_text(rewritten: Rewritten) -> str:
    """``rewrite(grad, by=sum_factorization, keep=(r_pass, s_pass))``."""
    theorems = [step.identity.name for step in rewritten.steps]
    by = theorems[0] if len(theorems) == 1 else f"({', '.join(theorems)})"
    extra = ""
    if rewritten.kept:
        extra = f", keep=({', '.join(kept.name for kept in rewritten.kept)})"
    elif rewritten.into is not None:
        extra = f", into={rewritten.into.__name__}"
    return f"rewrite({rewritten.array}, by={by}{extra})"


def structural_facts(
    rewritten: Rewritten,
    owner: str,
    identifier: Any,
    where: str,
    condition_ids: Sequence[str],
) -> list[Fact]:
    """The rows of a structural rewrite: what the passes leave, and their own facts.

    The claim that what the schedule compiles leaves the kernel's statement
    in the array, tested by draws of both terms on the interpreter and
    proved by Lean through :func:`structural_proof`; the
    ``single-assignment`` facts its equations rest on; and the kept arrays'
    facts, in bounds, distinct writes and stored before read, decided by
    isl (:func:`kept_facts`).
    """
    proof, assignment, detail_words = structural_proof(rewritten, owner, identifier)
    agreed, detail = _drawn_terms(
        rewritten.before, rewritten.after, rewritten.array, _short_text(rewritten)
    )
    status = (
        Status.TESTED
        if agreed
        else Status.REFUTED
        if agreed is False
        else Status.ASSUMED
    )
    provenance: dict[str, Any] = {
        "theorems": [step.identity.fact_id for step in rewritten.steps],
        "detail": detail,
        "proof": detail_words,
    }
    if agreed is False:
        provenance["reason"] = (
            f"{rewritten.text}: what the schedule compiles does not leave what the "
            f"kernel's statement says; {detail}"
        )
    elif agreed is None:
        provenance["reason"] = detail
    if proof is None:
        provenance["lean"] = f"not stated for Lean: {detail_words}"
    rests = tuple(
        dict.fromkeys(
            [
                *(step.identity.fact_id for step in rewritten.steps),
                *(getattr(proof, "rests_on", ()) or ()),
                *condition_ids,
            ]
        )
    )
    row = Fact(
        id=identifier(f"{KIND}:passes"),
        kind=KIND,
        statement=_structural_words(rewritten),
        term=proof,
        status=status,
        decided_by=None if status is Status.ASSUMED else "interpreter",
        provenance=provenance,
        where=where,
        owner=owner,
        rests_on=rests,
    )
    return [row, *assignment, *kept_facts(rewritten, owner, identifier)]


def kept_facts(rewritten: Rewritten, owner: str, identifier: Any) -> list[Fact]:
    """The kept arrays' facts: in bounds, distinct writes, and stored before read.

    Stated as for a kernel whose arrays the kept ones are, decided by isl
    where they are made, so that ``loopty run`` shows them settled.
    """
    if not rewritten.kept:
        return []
    from loopty.oracle import IslOracle
    from loopty.typing import (
        in_bounds_facts,
        kernel_definedness_facts,
        write_disjointness_facts,
    )

    after = rewritten.after
    view = dataclasses.replace(
        after, params=(*after.params, *after.temporaries), temporaries=()
    )
    stmts = {kept.stmt for kept in rewritten.kept}
    arrays = {kept.name for kept in rewritten.kept}
    rewritten_stmts = {match.stmt for match in rewritten.matches}
    found: list[Fact] = []
    for fact in in_bounds_facts(view, owner):
        statement = fact.provenance.get("statement")
        access = str(fact.provenance.get("access", ""))
        if statement in stmts or (
            statement in rewritten_stmts and access.split("[", 1)[0] in arrays
        ):
            found.append(fact)
    for fact in write_disjointness_facts(view, owner):
        if fact.provenance.get("statement") in stmts:
            found.append(fact)
    for fact in kernel_definedness_facts(view, owner):
        if fact.provenance.get("array") in arrays:
            found.append(fact)
    oracle = IslOracle()
    out = []
    for k, fact in enumerate(found):
        fact = dataclasses.replace(fact, id=identifier(f"kept:{fact.kind}:{k}"))
        try:
            if oracle.can_establish(fact):
                fact = oracle.establish(fact) or fact
        except Exception:  # noqa: BLE001 - a fact isl cannot answer stays assumed
            pass
        out.append(fact)
    return out


# }}}
