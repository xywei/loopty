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

    @property
    def text(self) -> str:
        """``c[t] - y[s] != 0 for every t and every s``."""
        shown = render(self.claim)
        if not self.over:
            return shown
        return f"{shown} for " + " and ".join(f"every {name}" for name, _ in self.over)


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

    @property
    def matches(self) -> tuple[Match, ...]:
        return tuple(match for step in self.steps for match in step.matches)

    @property
    def conditions(self) -> tuple[Condition, ...]:
        """Every condition of every match, once, with every theorem that needs it."""
        from loopty.hypotheses import structural_key

        found: dict[Any, Condition] = {}
        needs: dict[Any, list[str]] = {}
        for match in self.matches:
            for condition in match.conditions:
                key = (condition.kind, structural_key(condition.claim))
                found.setdefault(key, condition)
                names = needs.setdefault(key, [])
                if condition.theorem not in names:
                    names.append(condition.theorem)
        return tuple(
            dataclasses.replace(condition, theorem=_listed(needs[key]))
            for key, condition in found.items()
        )


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
    from loopty.faithful import _NoSample, sample_arguments
    from loopty.interpret import InterpretError, interpret
    from loopty.tolerance import disagreement, output_class

    array = rewritten.array
    exactness = output_class(step.before, array)
    if exactness == "exact":
        exactness = "approx"
    compared = 0
    for draw in range(3):
        try:
            arguments, _sizes = sample_arguments(step.before, seed=seed + draw)
        except _NoSample as exc:
            return None, f"no input could be drawn: {exc}"
        try:
            first = interpret(step.before, _copied(arguments), limit=200_000)
            second = interpret(step.after, _copied(arguments), limit=200_000)
        except (InterpretError, ArithmeticError, ValueError) as exc:
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
                f"{want[cell]} before {step.identity.name} and {got[cell]} after"
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
                conditions.append(render(condition.claim))
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
                at = ", ".join(f"{name}={env[name]}" for name, _ in condition.over)
                return False, f"{render(condition.claim)} fails at {at}"
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
            stmt = self.rewritten.after.stmt(condition.stmt)
            env = _values(arguments)
            env.update(_sizes_of(self.rewritten.after, arguments))
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
                        rewritten.after,
                        arguments,
                        self.identifier(f"{CONDITION}:{position}"),
                        owner,
                        self.where,
                        text,
                    ),
                    condition,
                )
            )
        equal = equality_facts(
            rewritten,
            owner,
            self.identifier,
            self.where,
            [
                (fact.id, condition)
                for fact, condition in conditions
                if condition.kind != "index"
            ],
        )
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


# {{{ structural rewrites: into= and keep=


def structural(
    rewritten: Rewritten,
    source: Any,
    *,
    keep: Any,
    into: Any,
    text: str,
    shown: str,
) -> Rewritten:
    """The rewrite with its nested sums stored (``keep=``) or a program as the target.

    Raises:
        IllegalRewrite: Until the structural rewrites land.
    """
    raise IllegalRewrite(f"{text}: keep= and into= are not supported yet")


# }}}
