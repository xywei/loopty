"""What a kernel leaves in its arrays, read off its term: exact dataflow.

The design idea. A kernel's postcondition is a claim about its arrays once the
body has run: ``scan`` claims ``off[0] == 0`` and ``off[r + 1] == off[r] +
cnt[r]``. Its native runs test the claim at the inputs they try
(:func:`loopty.faithful.postcondition_fact`). The term can decide it, for
every input, once one thing is known of an array the kernel writes: that no
cell of it is stored twice, or read before its store. Then the value a
statement stores is the value the cell holds when the kernel returns, and a
cell the statement reads holds then what it held when the statement read it,
since nothing stores it after the read. So each statement that stores into
such an array is an equation about the arrays as the kernel leaves them, its
*value equation*, at every instance of its domain::

    off[0] == 0                                                       (S0)
    all(off[r + 1] == off[r] + cnt[r] for r with 0 <= r < n)          (S1)

and the postcondition is decided under them, as a program's requirement is
decided under what its calls left (:func:`loopty.hypotheses.discharge`): the
points at which the claim fails where the equations hold are an isl set, and
it is empty. Each equation is used at the cells the claim reads, and the cells
that equation reads in turn, so the claim ``off[r + 1] == off[r] + cnt[r]``
takes ``S1``'s equation at ``r``. A claim about reals, ``R[a + 2, b] ==
-R[a, b + 2]``, is decided the same way, with each equation of values isl
cannot read an atom of its own (``atoms=True``): the claim is the equation
``S2`` stores, at ``(a + 2, b)``, and needs no arithmetic of reals.

That a cell is stored at most once and never read before its store is a fact
of the kernel's own, of kind ``single-assignment``, one per array, decided by
isl and stated for Lean (:func:`loopty.obligations.single_assignment_obligation`):

- two instances that store into the array, of one statement or of two, store
  into cells that differ; and
- an instance that stores a cell runs before every instance that reads it,
  in the source order (:func:`loopty.flow.schedule_of`), for every read of
  the array the kernel makes: in a statement's value, in a sum, in a
  subscript, in a guard, and the reads a ragged layout makes. The instance
  that stores a cell is not before itself: ``x[i] = 2 * x[i + 1 - 1] - 5``
  reads ``x[i]`` as it was before its store, so it says nothing of what the
  kernel leaves there, and ``x`` has no such fact.

An array that is updated in place (``y[r] += ...``, or a scatter-add through
a map, ``y[cell[c, k]] += ...``), whose cells hold what they held on entry
plus every update and so no one statement's value, that is stored under a
guard isl cannot state (whose instances store nowhere it can name), or that
is stored or read through an index isl does not read exactly, has no such
fact, and its statements no equation: the postcondition is then left to the
native runs, as before, with the reason. So is a statement that reads an
array without the fact: what it read may have been stored over since.

The postcondition fact rests on the ``single-assignment`` facts of the
arrays whose equations decided it, and on any theorem the kernel cites that
did (``@kernel(uses=[...])``, see :func:`theorem_instances`). It keeps what
the native runs found, in its provenance, and a native run that refutes it
still refutes it: the term and the body would then disagree, which the
``trace-faithful`` fact is about, and a counterexample is the stronger word.

What is not done: a cell stored twice, as a loop that overwrites a running
value does, needs the last store before each read, which is isl's ``lexmax``
of the stores before it; an array with such a cell has no equations here,
and its postcondition stays tested.
"""

from __future__ import annotations

from collections.abc import Collection, Mapping, Sequence
from dataclasses import dataclass, field
from functools import partial
from typing import Any

import islpy as isl
import pymbolic.primitives as prim
from lanky.ledger import Fact, Status, fact_id
from lanky.prelude import FinType, Int, Real, Refined, Sort
from lanky.terms import (
    Comparison,
    Forall,
    LogicalNot,
    LogicalOr,
    Subscript,
    Var,
    conjuncts,
    render,
)

from loopty import flow
from loopty.description import (
    Description,
    Disjunct,
    Unstatable,
    describe,
    description_of,
)
from loopty.handles import statement_names
from loopty.oracle import Empty
from loopty.source import as_spelled
from loopty.term import ArrType, Hypothesis, Stmt, Term

__all__ = [
    "KIND",
    "Decision",
    "Stores",
    "decide_postcondition",
    "program_stores",
    "single_assignment_id",
    "stores",
]

#: The kind of the fact that an array is stored at most once, and read after.
KIND = "single-assignment"


def single_assignment_id(
    owner: str, array: str, *, module: str | None = None, line: int | None = None
) -> str:
    """``single-assignment:spmv.scan@69:off``: the kernel's definition and the array."""
    return fact_id(KIND, owner, module=module, line=line, detail=array)


# {{{ what the statements store


@dataclass(frozen=True)
class Stores:
    """What a kernel's statements store, as equations about what it leaves.

    ``facts`` are the ``single-assignment`` facts of the arrays whose
    statements store final values, one per array, by array (a program's,
    one per array a call stores, by the fact's id); ``equations``
    the value equations of those statements (:class:`~loopty.term.Hypothesis`,
    each resting on the facts it needs); ``why`` says, for each array the
    kernel writes that has none, why.
    """

    facts: Mapping[str, Fact] = field(default_factory=dict)
    equations: tuple[Hypothesis, ...] = ()
    why: Mapping[str, str] = field(default_factory=dict)


def stores(
    term: Term, owner: str, *, module: str | None = None, line: int | None = None
) -> Stores:
    """The ``single-assignment`` facts of a kernel's arrays and its value equations.

    A program's term is not a kernel's (its calls are one after another, and
    its arrays pass between them), and gets none here; see
    :func:`program_stores`.
    """
    if term.scopes or term.temporaries:
        return Stores()
    return _stored(term, owner, module=module, line=line)


def program_stores(
    term: Term, owner: str, *, module: str | None = None, line: int | None = None
) -> Stores:
    """What each call of a program stores, as equations about what the program leaves.

    A program's calls run one after another, so each call's statements are
    read as a kernel's are, on their own: where the call stores no cell of
    an array twice, and reads none before its store, a fact of kind
    ``single-assignment`` of the program's says so for that call, and each
    of its statements that stores into the array is an equation about what
    the call leaves. The zeros an ``Arr.zeros_like`` starts an array with
    are no call's, and a call that stores every cell of the array leaves
    them nowhere.

    An equation of a call holds after it until a later statement of the
    program writes an array it names, the stored one or one it reads, as a
    callee's postcondition does (:meth:`loopty.compose._Composer.travel`);
    those that hold when the program returns are the ones kept, each resting
    on the facts of the call's arrays it needs. ``why`` says, for each array
    a call writes that has no fact, why, keyed ``array in call``.
    """
    from dataclasses import replace

    position = {stmt.id: k for k, stmt in enumerate(term.stmts)}
    facts: dict[str, Fact] = {}
    equations: list[Hypothesis] = []
    why: dict[str, str] = {}
    for scope in term.scopes:
        statements = set(scope.statements)
        mine = [stmt for stmt in term.stmts if stmt.id in statements]
        if not mine:
            continue
        last = max(position[stmt.id] for stmt in mine)
        later = {stmt.assignee.array for stmt in term.stmts[last + 1 :]}
        found = _stored(
            replace(term, stmts=tuple(mine)),
            owner,
            module=module,
            line=line,
            call=scope.call,
        )
        for array, reason in found.why.items():
            why[f"{array} in {scope.call}"] = reason
        by_id = {fact.id: fact for fact in found.facts.values()}
        for equation in found.equations:
            if equation.mentions & later:
                continue
            equations.append(equation)
            for identifier in equation.rests_on:
                facts.setdefault(identifier, by_id[identifier])
    return Stores(facts, tuple(equations), why)


def _stored(
    term: Term,
    owner: str,
    *,
    module: str | None = None,
    line: int | None = None,
    call: str | None = None,
) -> Stores:
    """:func:`stores` of ``term``'s statements, or of one call's, with ``call``.

    For a call, the facts are the program's, named after the call as well
    as the array, and an equation mentions every array it names, since a
    later call that writes one it only reads retires it as surely.
    """
    from loopty.obligations import Stated, single_assignment_obligation

    written = list(dict.fromkeys(stmt.assignee.array for stmt in term.stmts))
    facts: dict[str, Fact] = {}
    why: dict[str, str] = {}
    for array in written:
        found = _single_assignment(term, array)
        if isinstance(found, str):
            why[array] = found
            continue
        question, writers, places = found
        readers = list(dict.fromkeys(stmt.id for stmt, *_ in places))
        if call is None:
            identifier = single_assignment_id(owner, array, module=module, line=line)
            statement = f"no cell of {array} is stored twice, or read before its store"
        else:
            identifier = single_assignment_id(
                owner, f"{call}:{array}", module=module, line=line
            )
            statement = (
                f"{call} stores no cell of {array} twice, and reads none before "
                "its store"
            )
        facts[array] = Fact(
            id=identifier,
            kind=KIND,
            statement=statement,
            term=Empty(
                question,
                description=(
                    f"instances that store a cell of {array} another instance "
                    "stores, or that one reads before"
                ),
                lean=Stated(
                    partial(
                        single_assignment_obligation,
                        term,
                        array,
                        tuple(writers),
                        tuple(places),
                    )
                ),
            ),
            # decided where it is made, since what the postcondition is
            # decided under rests on it; the check asks isl, and Lean, again
            status=Status.DECIDED,
            decided_by="isl",
            provenance={
                "array": array,
                "writers": [writer.id for writer in writers],
                "readers": readers,
                **({"call": call} if call is not None else {}),
            },
            where=writers[0].where,
            owner=owner,
        )
    equations: list[Hypothesis] = []
    named = statement_names(term)
    for stmt in term.stmts:
        array = stmt.assignee.array
        if array not in facts:
            continue
        accessed = {
            name: kind
            for name, _indices, kind, _inames, _domain in flow.statement_accesses(
                stmt, term
            )
        }
        reads = {
            name
            for name, kind in accessed.items()
            if kind == "read" and name in written
        }
        if not reads <= set(facts):
            # what it read may have been stored over since
            continue
        equation = _equation(stmt)
        if equation is None:
            continue
        needs = [array, *sorted(reads - {array})]
        mentions = {array, *reads}
        if call is not None:
            mentions |= set(accessed)
        equations.append(
            Hypothesis(
                claim=equation,
                source=f"what {named(stmt.id)} stores in {array}",
                rests_on=tuple(facts[name].id for name in needs),
                mentions=frozenset(mentions),
            )
        )
    return Stores(facts, tuple(equations), why)


def _single_assignment(term: Term, array: str) -> tuple[Any, list[Stmt], list] | str:
    """isl's empty set of conflicts, the writers and the reads; or why there is none."""
    from loopty.obligations import _affine, _both, _time_map, stated_writers

    writers = [stmt for stmt in term.stmts if stmt.assignee.array == array]
    named = statement_names(term)
    for stmt in writers:
        if stmt.kind == "assign":
            continue
        if stmt.exactness is None:
            # y[i] = 2 * y[i]: the store reads the cell it stores, first
            return f"{named(stmt.id)} reads a cell of {array} before it stores it"
        # an update, y[i] += v or a scatter-add y[cell[c, k]] += v: the cell
        # holds what it held on entry and every update, which is no one
        # statement's value
        shown = ", ".join(
            str(index) for index in as_spelled(stmt, *stmt.assignee.indices)
        )
        return (
            f"{named(stmt.id)} updates {array}[{shown}] in place, so a cell of {array} "
            "holds what it held on entry plus every update, not what one "
            "statement stores"
        )
    if len(stated_writers(term, writers)) != len(writers):
        return (
            f"a statement stores into {array} under a guard isl cannot state, so "
            "its domain is wider than where it stores"
        )
    if not all(_affine(index) for stmt in writers for index in stmt.assignee.indices):
        return f"a statement stores into {array} at an index isl does not read exactly"
    places = [
        (stmt, tuple(indices), tuple(inames), domain)
        for stmt in term.stmts
        for name, indices, kind, inames, domain in flow.statement_accesses(stmt, term)
        if name == array and kind == "read"
    ]
    sizes = flow.size_names(term)
    depth = flow.instance_space_depth(term)
    pieces: list[Any] = []

    def keep(pairs: Any, tag: str) -> bool:
        found = flow.assume_sizes(pairs.domain(), sizes)
        pieces.append(found.set_tuple_name(tag))
        return found.is_empty()

    for k, first in enumerate(writers):
        stores_first = flow.access_relation(
            first.inames, first.domain, first.assignee.indices
        )
        for second in writers[k:]:
            stores_second = flow.access_relation(
                second.inames, second.domain, second.assignee.indices
            )
            mine, theirs = _both(stores_first, stores_second.reverse())
            pairs = mine.apply_range(theirs)
            if second is first:
                identity = isl.Map.identity(pairs.get_space())
                pairs, identity = _both(pairs, identity)
                pairs = pairs.subtract(identity)
            if not keep(pairs, f"store{len(pieces)}"):
                if second is first:
                    return (
                        f"two instances of {named(first.id)} store into one cell "
                        f"of {array}"
                    )
                return (
                    f"{named(first.id)} and {named(second.id)} store into one cell "
                    f"of {array}"
                )
    for stmt, indices, inames, domain in places:
        reads = flow.access_relation(inames, domain, indices)
        reading = _reading_time(stmt, inames, depth)
        for writer in writers:
            stores_writer = flow.access_relation(
                writer.inames, writer.domain, writer.assignee.indices
            )
            mine, theirs = _both(reads, stores_writer.reverse())
            same = mine.apply_range(theirs)
            # a store at the read's own time is the store of the instance
            # that reads, which comes after the read: lex_le, not lex_lt
            after = reading.apply_range(isl.Map.lex_le(reading.get_space().range()))
            after = after.apply_range(_time_map(writer, writer.inames, depth).reverse())
            same, after = _both(same, after)
            if not keep(same.intersect(after), f"read{len(pieces)}"):
                if writer is stmt:
                    return (
                        f"{named(stmt.id)} reads a cell of {array} before it "
                        "stores it"
                    )
                return (
                    f"{named(stmt.id)} reads a cell of {array} before "
                    f"{named(writer.id)} stores it"
                )
    question = None
    for piece in pieces:
        found = isl.UnionSet.from_set(piece)
        question = found if question is None else question.union(found)
    if question is None:
        question = isl.UnionSet("{ }")
    return question, writers, places


def _reading_time(stmt: Stmt, inames: Sequence[str], depth: int) -> Any:
    """``{ [loops] -> [time] }``: when a read over ``inames`` of ``stmt`` is made.

    A read in a sum is made inside the instance, at its time
    (:func:`loopty.obligations._time_map`). A read made at the outer loops
    alone, the length of a ragged row, is made once per row, before the
    row's instances (:func:`loopty.flow.layout_reads`); it is taken to be
    made at the time of every instance of the row, the inner loops left
    free, which finds a store after it wherever one comes after any of
    them: more than the read itself, never less.
    """
    from loopty.obligations import _time_map

    if len(inames) >= len(stmt.inames):
        return _time_map(stmt, inames, depth)
    full = _time_map(stmt, stmt.inames, depth)
    inner = len(stmt.inames) - len(inames)
    return full.project_out(isl.dim_type.in_, len(inames), inner)


def _equation(stmt: Stmt) -> Any:
    """The value equation of one statement: the cell it stores is what it stores there.

    ``all(off[r + 1] == off[r] + cnt[r] for r with <its domain>)``, a
    universal over the statement's loops, of sort ``Int``, whose guard is the
    statement's domain as loopty wrote it (:mod:`loopty.description`), so
    that an instance is taken only where the statement runs. ``None`` for a
    statement whose domain isl computed.
    """
    described = description_of(stmt.domain)
    if described is None:
        return None
    coords = [Var(name) for name in stmt.inames]
    try:
        guard = described.predicate(coords)
    except Unstatable:
        return None
    indices = tuple(stmt.assignee.indices)
    cell = Subscript(
        Var(stmt.assignee.array), indices[0] if len(indices) == 1 else indices
    )
    body = Comparison(cell, "==", stmt.expr)
    if not coords:
        return body if guard is True else Forall((), body, guard)
    binders = tuple((coord, Int) for coord in coords)
    return Forall(binders, body, None if guard is True else guard)


# }}}


# {{{ the postcondition, decided under them


@dataclass(frozen=True)
class Decision:
    """A postcondition isl decided under what the statements store.

    ``question`` is the isl question the fact keeps, every conjunct's empty
    set of points at which the claim fails where the hypotheses it used hold,
    and ``used`` the hypotheses, each once: value equations, and a cited
    theorem's instances.
    """

    question: Empty
    used: tuple[Hypothesis, ...]

    @property
    def rests_on(self) -> tuple[str, ...]:
        """The facts the hypotheses it used rest on, each once, in order."""
        return tuple(
            dict.fromkeys(
                fact for hypothesis in self.used for fact in hypothesis.rests_on
            )
        )


def decide_postcondition(
    term: Term,
    hypotheses: Sequence[Hypothesis],
    *,
    theorems: Sequence[Any] = (),
    entry: Sequence[Hypothesis] = (),
) -> Decision | str:
    """The postcondition decided conjunct by conjunct, or why it is not.

    Each conjunct is asked of isl under ``hypotheses``, the value equations
    (:func:`stores`), with every equation isl cannot state an atom of its own
    (:func:`loopty.hypotheses.discharge`). A conjunct they do not decide is
    asked again under the instances of the ``theorems`` the kernel cites,
    whose hypotheses the conjuncts decided before it, and the types of the
    arrays nothing writes (``entry``), match (:func:`theorem_instances`).
    """
    from loopty.hypotheses import discharge
    from loopty.obligations import Stated

    if term.post is None:
        return "the kernel claims nothing"
    parts = list(conjuncts(term.post))
    sizes = flow.size_names(term)
    types = dict(term.params)
    types.update(term.temporaries)
    integral = {
        name
        for name, typ in types.items()
        if isinstance(typ, ArrType) and _integral(typ.dtype)
    }
    known = set(term.sizes) | {
        name
        for name, typ in term.params
        if not isinstance(typ, ArrType) and _integral(typ)
    }
    # what is real, which Lean states in Mathlib's dialect
    sorts = {
        name: Real
        for name, typ in types.items()
        if _real(typ.dtype if isinstance(typ, ArrType) else typ)
    }
    decided: dict[int, Any] = {}
    reasons: dict[int, str] = {}
    pending = list(range(len(parts)))
    offered = list(hypotheses)
    for _round in range(2):
        for index in list(pending):
            stated = _claim(parts[index], term)
            if isinstance(stated, str):
                reasons[index] = stated
                pending.remove(index)
                continue
            domain, goal, reflected = stated
            outcome = discharge(
                domain,
                goal,
                offered,
                integral=integral,
                known=known,
                nonneg=set(sizes),
                reflected=reflected,
                description=(
                    f"points at which {term.name} leaves {render(parts[index])} false, "
                    "where what its statements store holds"
                ),
                atoms=True,
                sorts=sorts,
            )
            if outcome.decided:
                decided[index] = outcome
                pending.remove(index)
            else:
                reasons[index] = outcome.reason
        if not pending or not theorems:
            break
        # the conjuncts decided so far are hypotheses of a cited theorem,
        # those about reals where they read the same as reals (exact)
        reals = real_names(term)
        available = [
            Hypothesis(
                parts[index],
                f"the postcondition of {term.name}, its conjunct "
                f"{render(parts[index])}",
                rests_on=tuple(
                    dict.fromkeys(f for h in decided[index].used for f in h.rests_on)
                ),
                mentions=frozenset(),
            )
            for index in sorted(decided)
            if exact(parts[index], reals)
        ]
        found = theorem_instances(term, theorems, [*available, *entry], hypotheses)
        # a theorem's conclusion is about the reals, and decides a conjunct
        # the floats are claimed of only where the two readings agree
        found = [instance for instance in found if exact(instance.claim, reals)]
        if not found:
            break
        offered = [*hypotheses, *found]
    if pending or len(decided) != len(parts):
        missing = [index for index in range(len(parts)) if index not in decided]
        shown = "; ".join(
            f"{render(parts[index])}: {reasons.get(index, 'not decided')}"
            for index in missing[:2]
        )
        return f"what the statements store does not decide {shown}"
    used: list[Hypothesis] = []
    pieces = None
    questions = []
    for index in range(len(parts)):
        outcome = decided[index]
        for hypothesis in outcome.used:
            if not any(hypothesis is seen for seen in used):
                used.append(hypothesis)
        questions.append(outcome.question)
        piece = isl.UnionSet.from_set(
            outcome.question.obj.set_tuple_name(f"post{index}")
        )
        pieces = piece if pieces is None else pieces.union(piece)
    assert pieces is not None
    from loopty.obligations import conjunct_obligations

    question = Empty(
        pieces,
        description=f"points at which {term.name} leaves its postcondition false",
        lean=Stated(
            partial(
                conjunct_obligations,
                tuple(questions),
                tuple(render(part) for part in parts),
            )
        ),
    )
    return Decision(question, tuple(used))


def _claim(conjunct: Any, term: Term) -> tuple[Any, Any, dict[str, Any]] | str:
    """A conjunct as isl asks it: its binders' domain, the claim, the cells it reflects.

    A universal's binders are the dimensions, each bounded by its sort: a
    point of ``Fin[e]`` is ``0 <= x < e``, a natural ``x >= 0``, an integer
    anything; the guard goes into the claim, as what it is claimed under. A
    bound that reads an array, ``Fin[cnt[r]]``,
    is a parameter of the domain that stands for the cell. A conjunct with no
    binder is claimed at the one point of a set of no dimensions.
    """
    from loopty.idx import Reflections

    if isinstance(conjunct, Forall):
        binders = list(conjunct.binders)
        claims = [conjunct.body]
        under = list(conjuncts(conjunct.guard))
    else:
        binders = []
        claims = [conjunct]
        under = []
    dims = [var.name for var, _ in binders]
    table = term.reflections
    if isinstance(table, Reflections):
        table.reserve(dims)
    reflected: dict[str, Any] = {}
    constraints: list[str] = []
    bounded: set[str] = set()
    for var, sort in binders:
        # a refinement's propositions are left out: the claim is then asked
        # over more points, which is harder, never easier
        base = sort
        while isinstance(base, Refined):
            base = base.base
        if isinstance(base, FinType):
            try:
                text = flow.expr_text(base.bound, None, reflected, table)
            except Exception:  # noqa: BLE001 - a bound isl cannot read
                return f"isl cannot read the bound {render(base.bound)} of {var.name}"
            constraints.append(f"0 <= {var.name} < {text}")
            bounded.add(var.name)
        elif isinstance(base, Sort) and base.name == "Nat":
            constraints.append(f"{var.name} >= 0")
        elif isinstance(base, Sort) and base.name == "Int":
            continue
        else:
            return (
                f"the postcondition quantifies {var.name} over {sort}, which isl cannot"
            )
    sizes = flow.size_names(term)
    free: set[str] = set()
    for text in constraints:
        free |= flow.free_names(text)
    params = sorted(free - set(dims))
    nonneg = [name for name in params if name in sizes or name in reflected]
    domain = isl.Set(flow._assemble(dims, constraints, params, set(nonneg)))
    domain = describe(
        domain,
        Description(
            dims=tuple(dims),
            disjuncts=(Disjunct(tuple(constraints), tuple(nonneg)),),
            reflected=tuple(reflected.items()),
        ),
    )
    goal = claims[0]
    if under:
        guard = under[0] if len(under) == 1 else prim.LogicalAnd(tuple(under))
        goal = LogicalOr((LogicalNot(guard), goal))
    return domain, goal, reflected


def _real(sort: Any) -> bool:
    """Whether the points of ``sort`` are reals: ``Real``, or a numpy float type."""
    import numpy as np

    base = sort.base if isinstance(sort, Refined) else sort
    if isinstance(base, Sort):
        return base.name == "Real"
    try:
        return bool(np.issubdtype(np.dtype(base), np.floating))
    except TypeError:
        return False


def _integral(sort: Any) -> bool:
    """Whether the points of ``sort`` are integers, which isl can compare."""
    from loopty.contract import integral_sort

    return integral_sort(sort)


def _codomain(sort: Any) -> Any:
    """The values of a family of families, ``Real`` for ``Fn[Nat, Fn[Nat, Real]]``."""
    from lanky.prelude import FnType

    while isinstance(sort, FnType):
        sort = sort.codomain
    return sort


def real_names(term: Term) -> set[str]:
    """The kernel's arrays of reals and its scalars that are reals."""
    return {
        name
        for name, typ in term.params
        if _real(typ.dtype if isinstance(typ, ArrType) else typ)
    }


def exact(prop: Any, reals: set[str], families: Collection[str] = ()) -> bool:
    """Whether ``prop`` says the same of floats read as reals as it says of floats.

    A conjunct of a postcondition is decided as the kernel computes, in
    floating point, and a theorem the kernel cites is about the reals. The
    two readings of a comparison of reals agree where nothing in it rounds:
    cells, scalars and literals (a float is the rational it holds, and Lean
    reads it so), each perhaps negated. ``R[a + 2, b] == -R[a, b + 2]`` is
    one, and a theorem may take it as a hypothesis. ``s[i] == a[i] + b[i]``
    is not: the stored sum is rounded, so ``s[i]`` is not ``a[i] + b[i]`` as
    reals, and a theorem that concluded ``s[i] - a[i] == b[i]`` from it
    would say of the arrays what they do not hold. Comparisons of integers
    are exact. A value of one of ``families``, a theorem's table that
    nothing bound (``D(a)(b)``), is a real as it stands, and exact too.
    """
    from loopty.hypotheses import _call_spine, _names

    def value(expr: Any) -> bool:
        if isinstance(expr, bool):
            return False
        if isinstance(expr, int | float):
            return True
        if isinstance(expr, prim.Variable):
            return True
        if isinstance(expr, prim.Subscript):
            return isinstance(expr.aggregate, prim.Variable)
        if isinstance(expr, prim.Call):
            spine = _call_spine(expr)
            return spine is not None and spine[0] in families
        if isinstance(expr, prim.Product):
            # -x, which pymbolic writes as a product with -1
            rest = [child for child in expr.children if not isinstance(child, int)]
            signs = [child for child in expr.children if isinstance(child, int)]
            return (
                len(rest) == 1 and all(s in (1, -1) for s in signs) and value(rest[0])
            )
        return False

    def visit(node: Any) -> bool:
        if isinstance(node, Forall):
            return visit(node.body) and (node.guard is None or visit(node.guard))
        if isinstance(node, prim.LogicalAnd | prim.LogicalOr):
            return all(visit(child) for child in node.children)
        if isinstance(node, prim.LogicalNot):
            return visit(node.child)
        if isinstance(node, prim.Comparison):
            if not (_names(node.left) | _names(node.right)) & reals:
                return True
            return value(node.left) and value(node.right)
        return not (_names(node) & reals)

    return visit(prop)


# }}}


# {{{ a cited theorem


def theorem_instances(
    term: Term,
    theorems: Sequence[Any],
    available: Sequence[Hypothesis],
    equations: Sequence[Hypothesis],
) -> list[Hypothesis]:
    """The instances of the cited theorems whose hypotheses ``available`` give.

    Every hypothesis of a theorem has to match a conjunct of one of
    ``available``, as a program's cited theorem does
    (:func:`loopty.hypotheses.theorem_instances`), and its families stand for
    the kernel's arrays where :func:`applicable` says they may.
    """
    from loopty.hypotheses import linear
    from loopty.hypotheses import theorem_instances as instances_of

    sizes = set(flow.size_names(term))
    written = {stmt.assignee.array for stmt in term.stmts}

    def nonnegative(expr: Any) -> bool:
        form = linear(expr)
        if form is None:
            return False
        names, constant = form
        return constant >= 0 and all(
            coefficient >= 0 and name in sizes for name, coefficient in names.items()
        )

    integral = sizes | {
        name
        for name, typ in term.params
        if not isinstance(typ, ArrType) and _integral(typ)
    }

    def integer(expr: Any) -> bool:
        form = linear(expr)
        return form is not None and all(name in integral for name in form[0])

    def fits(array: str, domain: Any, codomain: Any) -> str | None:
        return applicable(term, written, array, codomain)

    scalars = real_scalars(term)

    def real(expr: Any) -> bool:
        return _real_value(expr, scalars)

    out: list[Hypothesis] = []
    for theorem in theorems:
        found, _reasons = instances_of(
            theorem, available, fits, nonnegative, integer, real
        )
        out.extend(found)
    return out


def real_scalars(term: Term) -> set[str]:
    """The kernel's scalars that are reals: ``s`` of ``s: Real``."""
    return {
        name
        for name, typ in term.params
        if not isinstance(typ, ArrType) and _real(typ)
    }


def _real_value(expr: Any, scalars: Collection[str]) -> bool:
    """Whether a theorem's ``Real`` may stand for ``expr``: a real scalar, or a number.

    A real scalar of the kernel is the real its float holds, and a number
    the rational it is, as Lean reads it. Anything computed from them may
    round, and is refused, as a conjunct that rounds is not matched
    (:func:`exact`).
    """
    if isinstance(expr, bool):
        return False
    if isinstance(expr, int | float):
        return True
    return isinstance(expr, prim.Variable) and expr.name in scalars


def applicable(term: Term, written: set[str], array: str, codomain: Any) -> str | None:
    """Why the kernel's ``array`` cannot stand for a family into ``codomain``.

    ``None`` when it can.

    A family's values are points of its codomain, and an array's cells are
    points of its element sort only where something says so: a real array
    stands for a family of reals, and an integral one for a family of
    integers (``Int``); a family of naturals, or of a ``Fin`` sort, takes an
    array the kernel does not write, whose cells the contract checks are
    points of that sort when the kernel is called. The cells a kernel writes
    are what it stores, which nothing says are naturals: a theorem about
    them is over ``Int``.
    """
    types = dict(term.params)
    typ = types.get(array)
    if not isinstance(typ, ArrType):
        return f"{array} is no array of {term.name}'s"
    element = typ.dtype.base if isinstance(typ.dtype, Refined) else typ.dtype
    wanted = codomain.base if isinstance(codomain, Refined) else codomain
    name = getattr(wanted, "name", None)
    if name == "Real":
        if _real(element):
            return None
        return f"the elements of {array} are not reals"
    if not _integral(element):
        return f"the elements of {array} are not integers"
    if name == "Int":
        return None
    if array in written:
        return (
            f"{term.name} writes {array}, and nothing says its cells are points of "
            f"{codomain}: a claim about them is over Int"
        )
    if name == "Nat" and (
        isinstance(element, FinType) or getattr(element, "name", None) == "Nat"
    ):
        return None
    if isinstance(wanted, FinType) and isinstance(element, FinType):
        from loopty.hypotheses import structural_key

        if structural_key(wanted.bound) == structural_key(element.bound):
            return None
    return f"the elements of {array} are points of {typ.dtype}, not of {codomain}"


# }}}


# {{{ a cited theorem, applied to the postcondition

#: The kind of a fact a cited theorem gives a kernel, applied to its postcondition.
APPLIED = "by-theorem"

#: How many ways of matching a theorem's hypotheses are looked at, at most.
_SEARCH = 4096


@dataclass(frozen=True)
class Application:
    """A cited theorem applied to a kernel's postcondition.

    ``bindings`` give each theorem variable the kernel's postcondition
    binds: a family, the name of the kernel's array it stands for; a scalar,
    the expression of the kernel's sizes. ``matched`` pairs each hypothesis
    of the theorem a conjunct of the postcondition gives with that
    conjunct's position. ``claim`` is what follows: the theorem's conclusion
    at the kernel's arrays, for every value of the variables nothing binds,
    under the hypotheses nothing gives. ``conclusion`` is that conclusion
    alone, ``free`` the names of the variables nothing binds and ``unmatched``
    the names of the hypotheses nothing gives, which the ledger's row names.
    """

    theorem: Any
    bindings: tuple[tuple[str, Any], ...]
    matched: tuple[tuple[int, int], ...]
    claim: Any
    conclusion: Any = None
    free: tuple[str, ...] = ()
    unmatched: tuple[str, ...] = ()


def theorem_applications(
    term: Term, theorems: Sequence[Any], given: Sequence[Any]
) -> list[Application]:
    """What each cited theorem gives the kernel, from its postcondition's conjuncts.

    As many of a theorem's hypotheses as can be are matched with conjuncts
    of the postcondition, ``given``, up to the names of their binders, a
    family ``R(a)(b)`` matching a cell ``R[a, b]`` (:mod:`loopty.hypotheses`),
    which binds the families the hypotheses apply to the kernel's arrays and
    their scalars to expressions of its sizes. The theorem's conclusion with
    those put in is the kernel's, for every value of the variables nothing
    bound, under the hypotheses nothing matched: ``on_the_triangle``, whose
    copies and recurrence are the conjuncts of ``reconstruct``'s
    postcondition, gives that ``R`` is ``D`` on the triangle for every table
    ``D`` of the PDE's whose stored entries ``c`` holds.

    A match is taken only where it says something of what the kernel
    computes, from what the kernel was shown to leave:

    - its conclusion is about an array the kernel writes, and every
      hypothesis about such an array is a conjunct of the postcondition. A
      hypothesis about what the kernel stores that nothing matched would be
      a guard of the claim the kernel does not establish. A kernel whose
      recurrence is off by one matches only the theorem's copies, and so
      gives nothing, rather than a claim under a recurrence it does not
      compute;
    - a conjunct about reals is matched only where it reads the same as
      reals as it does in floating point (:func:`exact`), since the theorem
      is about the reals;
    - two families of the theorem stand for two arrays of the kernel, never
      for one: the copies ``R(a)(b) == c(a)(b)`` read backwards would
      otherwise match the stored entries ``c(a)(b) == D(a)(b)``, with ``D``
      and ``R`` both ``c``, and give a claim about ``c`` and itself;
    - each binding is one an array or a size can stand for (:func:`_misfit`);
    - the claim names nothing but the kernel's names and its own binders.
      The theorem's sizes are put in wherever they are, the sort of a family
      nothing bound included (``D: Fn[Fin[n], Real]`` with ``n`` bound to
      ``p`` ranges over ``Fin[p]``), so a name of the theorem's left in the
      claim would make it about a size the kernel does not have, or about
      one of its own that is not the size the theorem was taken at.

    Of the matches that pass, the one that matches the most hypotheses is
    taken. A theorem with none, or whose conclusion is a conjunct of the
    postcondition already, gives nothing.
    """
    return [
        found
        for _theorem, found in theorem_outcomes(term, theorems, given)
        if isinstance(found, Application)
    ]


#: How far a match got before it was refused, for the nearest one's reason:
#: two families one array, a conclusion about no array the kernel writes, a
#: hypothesis about one that nothing matched, a binding nothing stands for, a
#: conclusion that rounds, one that cannot be stated, one that names more.
_STAGES = ("shared", "unwritten", "unmatched", "misfit", "rounds", "unstated", "names")


def theorem_outcomes(
    term: Term, theorems: Sequence[Any], given: Sequence[Any]
) -> list[tuple[Any, Application | str]]:
    """Each cited theorem with what it gives the kernel, or why it gives nothing.

    The applications are :func:`theorem_applications`'. A theorem that gives
    nothing says why, in the kernel's terms (#234), from the match that got
    furthest before it was refused (:data:`_STAGES`), the most hypotheses
    matched among those: its hypothesis about an array the kernel writes
    that no conjunct of the postcondition is, put in the kernel's names, with
    the nearest conjunct (``the nearest is ...``); a binding no array or size
    of the kernel's can stand for; its conclusion, where it rounds, or is a
    conjunct already. Where nothing matched, a conjunct about reals that a
    hypothesis would match and that rounds, or that no conjunct is any of its
    hypotheses.
    """
    from lanky.prelude import FnType

    from loopty.hypotheses import (
        _apply,
        _Matcher,
        _names,
        free_in,
        linear,
        structural_key,
    )

    sizes = set(flow.size_names(term))
    written = {stmt.assignee.array for stmt in term.stmts}
    given = list(given)
    reals = real_names(term)
    usable = [exact(part, reals) for part in given]
    have = {structural_key(part) for part in given}
    known = (
        {name for name, _ in term.params}
        | sizes
        | {name for name, _ in term.reflected}
        | (free_in(term.post) if term.post is not None else set())
    )
    out: list[tuple[Any, Application | str]] = []
    for theorem in theorems:
        variables = dict(getattr(theorem, "variables", ()))
        functions = {
            name for name, sort in variables.items() if isinstance(sort, FnType)
        }
        matcher = _Matcher(functions, set(variables) - functions)
        named = list(getattr(theorem, "hypotheses", ()))
        hypotheses = [prop for _name, prop in named]
        goal = getattr(theorem, "goal", None)
        if goal is None or not hypotheses:
            out.append(
                (
                    theorem,
                    "it has no hypothesis for a conjunct of the postcondition to be",
                )
            )
            continue
        families = [_names(prop) & functions for prop in hypotheses]
        concluded = _names(goal) & functions
        reals_too = reals | {
            name for name, sort in variables.items() if _real(_codomain(sort))
        }
        best: list[Any] = [None]
        nearest: list[Any] = [None]
        tried = [0]

        def refused(sigma: dict[str, Any], matched: tuple) -> Any:
            """``None`` for a match that applies, else its stage and its words."""
            arrays = [sigma[name] for name in functions if name in sigma]
            # two families of a theorem are two arrays: one array standing
            # for both makes a claim about the array and itself
            if len(set(arrays)) != len(arrays):
                shared = next(a for a in arrays if arrays.count(a) > 1)
                both = [f for f in sorted(functions) if sigma.get(f) == shared]
                return 0, lambda: (
                    f"its {' and '.join(both)} would both be {shared}, and two of "
                    "its families are two arrays"
                )
            outputs = {name for name in functions if sigma.get(name) in written}
            if not outputs or not (concluded & outputs):
                return 1, lambda: (
                    f"its conclusion would be about no array {term.name} writes"
                )
            used = {k for k, _ in matched}
            missing = [
                k
                for k in range(len(hypotheses))
                if k not in used and families[k] & outputs
            ]
            if missing:
                return 2, partial(
                    _unmatched_words,
                    term,
                    named,
                    missing,
                    sigma,
                    functions,
                    given,
                    {position for _k, position in matched},
                )
            misfit = _misfit(term, written, sizes, variables, sigma, linear)
            if misfit is not None:
                return 3, lambda: misfit
            # the conclusion is about the reals, and says what the floats
            # hold only where the two readings agree
            free = functions - set(sigma)
            bound = {sigma[name] for name in functions if name in sigma}
            conclusion = _apply(goal, sigma, set(sigma) & functions)
            if not exact(conclusion, reals_too | bound, free):
                return 4, lambda: (
                    f"its conclusion there, {render(conclusion)}, holds of the "
                    f"reals and not of the floats {term.name} computes, which round"
                )
            made = _applied_claim(
                term, theorem, variables, sigma, matched, set(sigma) & functions
            )
            if made is None:
                return 5, lambda: (
                    "its conclusion cannot be stated there: the sorts of the "
                    "variables nothing binds name one another"
                )
            claim, _conclusion, names = made
            extra = free_in(claim) - set(names) - known
            if extra:
                return 6, lambda: (
                    f"its conclusion there would name {', '.join(sorted(extra))}, "
                    f"which {term.name} does not"
                )
            return None

        def search(
            k: int, sigma: dict[str, Any], matched: tuple[tuple[int, int], ...]
        ) -> None:
            tried[0] += 1
            if tried[0] > _SEARCH:
                return
            if k == len(hypotheses):
                if not matched or (
                    best[0] is not None and len(matched) <= len(best[0][1])
                ):
                    return
                why = refused(sigma, matched)
                if why is None:
                    best[0] = (dict(sigma), matched)
                    return
                rank = (why[0], len(matched))
                if nearest[0] is None or rank > nearest[0][0]:
                    nearest[0] = (rank, why[1])
                return
            for position, conjunct in enumerate(given):
                if not usable[position]:
                    continue
                trial = dict(sigma)
                if matcher.match(hypotheses[k], conjunct, trial, {}):
                    search(k + 1, trial, (*matched, (k, position)))
            search(k + 1, sigma, matched)

        search(0, {}, ())
        if best[0] is None:
            if nearest[0] is not None:
                out.append((theorem, nearest[0][1]()))
                continue
            out.append(
                (
                    theorem,
                    _nothing_matched(
                        named, given, usable, matcher, tried[0] > _SEARCH
                    ),
                )
            )
            continue
        sigma, matched = best[0]
        bound = {name for name in functions if name in sigma}
        made = _applied_claim(term, theorem, variables, sigma, matched, bound)
        assert made is not None
        claim, conclusion, free = made
        if structural_key(claim) in have:
            out.append(
                (
                    theorem,
                    f"its conclusion, {render(claim)}, is a conjunct of the "
                    "postcondition already",
                )
            )
            continue
        used = {k for k, _ in matched}
        out.append(
            (
                theorem,
                Application(
                    theorem,
                    tuple((name, sigma[name]) for name in variables if name in sigma),
                    matched,
                    claim,
                    conclusion,
                    free,
                    tuple(name for k, (name, _) in enumerate(named) if k not in used),
                ),
            )
        )
    return out


def _unmatched_words(
    term: Term,
    named: Sequence[tuple[str, Any]],
    missing: Sequence[int],
    sigma: Mapping[str, Any],
    functions: Collection[str],
    given: Sequence[Any],
    taken: Collection[int],
) -> str:
    """``no conjunct of the postcondition is its hypothesis recurrence, ...``.

    The hypothesis in the kernel's names, as the match put them, and the
    conjunct nearest it: one about the same arrays that no hypothesis
    matched, else any about them.
    """
    from loopty.hypotheses import _apply, _names

    k = missing[0]
    name, prop = named[k]
    wanted = _apply(prop, sigma, set(sigma) & set(functions))
    arrays = {sigma[f] for f in _names(prop) & set(functions) if f in sigma}
    about = [
        position
        for position, conjunct in enumerate(given)
        if arrays & _names(conjunct)
    ]
    others = [position for position in about if position not in taken]
    words = (
        f"no conjunct of the postcondition of {term.name} is its hypothesis "
        f"{name}, {render(wanted)}"
    )
    if others or about:
        words += f"; the nearest is {render(given[(others or about)[0]])}"
    rest = [named[j][0] for j in missing[1:]]
    if rest:
        words += f"; nor is any its {' or '.join(rest)}"
    return words


def _nothing_matched(
    named: Sequence[tuple[str, Any]],
    given: Sequence[Any],
    usable: Sequence[bool],
    matcher: Any,
    stopped: bool,
) -> str:
    """Why no hypothesis of a theorem matched a conjunct of the postcondition.

    A conjunct about reals that rounds, which a hypothesis would match: a
    theorem about the reals is not given it. Else none is any hypothesis.
    """
    for name, prop in named:
        for position, conjunct in enumerate(given):
            if usable[position]:
                continue
            if matcher.match(prop, conjunct, {}, {}):
                return (
                    f"the conjunct {render(conjunct)} would be its hypothesis "
                    f"{name}, and it is about reals and rounds in floating point, "
                    "so a claim about the reals is not given it"
                )
    if stopped:
        return (
            f"no match of its hypotheses was found in the {_SEARCH} ways of "
            "matching them that were tried"
        )
    if len(named) == 1:
        ((name, prop),) = named
        return (
            f"no conjunct of the postcondition is its hypothesis {name}, "
            f"{render(prop)}"
        )
    names = ", ".join(name for name, _ in named)
    return f"no conjunct of the postcondition is any of its hypotheses, {names}"


def _misfit(
    term: Term,
    written: set[str],
    sizes: set[str],
    variables: Mapping[str, Any],
    sigma: Mapping[str, Any],
    linear: Any,
) -> str | None:
    """Why a binding is not one the theorem may be taken at, or ``None``.

    A family stands for an array of as many axes as the family takes
    arguments, whose cells are points of the family's values
    (:func:`applicable`); a natural scalar for an expression of the sizes
    that is never negative; an integer scalar for an affine expression of
    the sizes and the kernel's integer scalars, never of a real one: a
    theorem about an integer ``k`` taken at a real ``s`` says what may be
    false of it; and a real scalar for one of the kernel's real scalars, or
    a number, which read as reals as their floats do (:func:`_real_value`).
    """
    from lanky.prelude import FnType

    reals = real_scalars(term)
    types = dict(term.params)
    integral = set(sizes) | {
        name
        for name, typ in term.params
        if not isinstance(typ, ArrType) and _integral(typ)
    }
    for name, sort in variables.items():
        if name not in sigma:
            continue
        value = sigma[name]
        shown = value if isinstance(value, str) else render(value)
        if isinstance(sort, FnType):
            rank, codomain = 0, sort
            while isinstance(codomain, FnType):
                rank += 1
                codomain = codomain.codomain
            typ = types.get(value)
            if not isinstance(typ, ArrType):
                return (
                    f"its {name} would be {shown}, which is no array of {term.name}'s"
                )
            if typ.ndim != rank:
                return (
                    f"its {name} takes {rank} arguments, and {shown}, which it "
                    f"would be, has {typ.ndim} axes"
                )
            why = applicable(term, written, value, codomain)
            if why is not None:
                return f"its {name} would be {shown}, but {why}"
        elif getattr(sort, "name", None) == "Nat":
            form = linear(value)
            if form is None or form[1] < 0 or any(
                c < 0 or n not in sizes for n, c in form[0].items()
            ):
                return (
                    f"its natural {name} would be {shown}, which is no sum of "
                    f"{term.name}'s sizes that is never negative"
                )
        elif getattr(sort, "name", None) == "Int":
            form = linear(value)
            if form is None or any(n not in integral for n in form[0]):
                return (
                    f"its integer {name} would be {shown}, which is no integer "
                    f"of {term.name}'s"
                )
        elif getattr(sort, "name", None) == "Real":
            if not _real_value(value, reals):
                return (
                    f"its real {name} would be {shown}, which is neither a real "
                    f"of {term.name}'s nor a number that reads as a real as its "
                    "float does"
                )
        else:
            return f"its {name} is of {sort}, which nothing of {term.name}'s stands for"
    return None


def _applied_claim(
    term: Term,
    theorem: Any,
    variables: Mapping[str, Any],
    sigma: Mapping[str, Any],
    matched: Sequence[tuple[int, int]],
    bound: set[str],
) -> tuple[Any, Any, tuple[str, ...]] | None:
    """The theorem's conclusion at the kernel's arrays, under what nothing matched.

    A variable nothing bound is a binder of the claim, named apart from the
    kernel's names, and a hypothesis nothing matched its guard. Every sort is
    put in the kernel's names with the rest (:func:`loopty.hypotheses._apply_sort`),
    and a binder whose sort names another variable nothing bound
    (``D: Fn[Fin[m], Real]``) comes after it. Returns the claim, the
    conclusion alone, and the names of the claim's binders; ``None`` when no
    order of the binders has each sort name only those before it.
    """
    from loopty.hypotheses import _apply, _apply_sort, _fresh, _names, _sort_terms

    taken = {name for name, _ in term.params} | set(term.sizes)
    for stmt in term.stmts:
        taken |= set(stmt.inames)
    left = [name for name in variables if name not in sigma]
    order: list[str] = []
    while left:
        ready = next(
            (
                name
                for name in left
                if not (_names(_sort_terms(variables[name])) - {name}) & set(left)
            ),
            None,
        )
        if ready is None:
            return None
        order.append(ready)
        left.remove(ready)
    renames: dict[str, Any] = {}
    binders = []
    for name in order:
        sort = variables[name]
        fresh = name if name not in taken else _fresh(name, taken)
        taken.add(fresh)
        if fresh != name:
            renames[name] = Var(fresh)
        binders.append((Var(fresh), sort))
    # a variable nothing bound is renamed, and a family bound is the array
    full = {**sigma, **renames}
    used = {k for k, _ in matched}
    hypotheses = [prop for _name, prop in getattr(theorem, "hypotheses", ())]
    rest = [
        _apply(prop, full, bound) for k, prop in enumerate(hypotheses) if k not in used
    ]
    goal = _apply(theorem.goal, full, bound)
    binders = [(var, _apply_sort(sort, full, bound)) for var, sort in binders]
    names = tuple(var.name for var, _ in binders)
    if not binders and not rest:
        return goal, goal, names
    guard = None
    if rest:
        guard = rest[0] if len(rest) == 1 else prim.LogicalAnd(tuple(rest))
    return Forall(tuple(binders), goal, guard), goal, names


@dataclass(frozen=True)
class Applied:
    """The claim a cited theorem gives a kernel, as the term of its fact, for Lean.

    One part: the theorem as it was proved and the conjuncts of the
    postcondition it was matched with, as hypotheses, imply the claim. Lean
    applies the theorem, with the variables loopty bound given by name, and
    finds the others where its conclusion is the goal; each premise left is
    a conjunct of the postcondition, a hypothesis of the claim's own, or
    arithmetic. So Lean checks that the theorem applies, and not only what
    follows from it. A claim about reals is a part of Mathlib's
    (:attr:`lanky.obligations.Part.mathlib`).

    The theorem's hypothesis is a citation (:func:`loopty.obligations.cited`):
    lanky restates it from the text the theorem's own command proved, not by
    printing the theorem again.
    """

    term: Term
    application: Application
    given: tuple[Any, ...]
    owner: str

    def lean_obligation(self) -> Any:
        """The part, stated from the theorem's term and the kernel's postcondition."""
        from loopty.obligations import (
            Claim,
            _declined,
            _lanky_obligations,
            _obligation,
            cited,
        )

        lanky = _lanky_obligations()
        if not hasattr(lanky, "Application"):
            return _declined(
                "this lanky's Lean oracle applies no claim a hypothesis of a part "
                "is; lanky 0.3 does"
            )
        from lanky.prelude import FnType

        application = self.application
        sorts = {
            name: Real
            for name, typ in self.term.params
            if _real(typ.dtype if isinstance(typ, ArrType) else typ)
        }
        reals = bool(sorts) and "mathlib" in getattr(
            lanky.Part, "__dataclass_fields__", {}
        )
        if sorts and not reals:
            return _declined(
                "the claim is about real numbers, and this lanky states no part "
                "of Mathlib's"
            )
        # an array a family stands for has the family's values, which
        # applicable() checked its cells are: cnt of Nat is Int -> Nat, as the
        # theorem's family prints, so that Lean can apply the theorem to it
        variables = dict(getattr(application.theorem, "variables", ()))
        for name, value in application.bindings:
            sort = variables.get(name)
            if isinstance(sort, FnType) and isinstance(value, str):
                while isinstance(sort, FnType):
                    sort = sort.codomain
                sorts[value] = sort
        theorem_term = getattr(application.theorem, "term", None)
        if theorem_term is None:
            return _declined("the claim has no statement to apply")
        claim = Claim(self.term, sorts=sorts)
        claim.hypotheses.append(theorem_term)
        for _k, position in application.matched:
            claim.hypotheses.append(self.given[position])
        closed = claim.close(application.claim)
        guards = [render(part) for part in conjuncts(closed.guard)]
        try:
            index = guards.index(render(theorem_term))
        except ValueError:
            return _declined("the claim is no hypothesis of its own part")
        values = tuple(
            (name, Var(value) if isinstance(value, str) else value)
            for name, value in application.bindings
        )
        hints = lanky.Hints(applications=(lanky.Application(index, values),))
        name = getattr(application.theorem, "__name__", "theorem")
        extra = {"mathlib": True} if reals else {}
        part = lanky.Part(
            "applied",
            f"{name} applied to the postcondition of {self.term.name}",
            closed,
            hints,
            **extra,
            **cited(guards, (application.theorem,)),
        )
        return _obligation(
            [part],
            rests_on=(getattr(application.theorem, "fact_id", ""),),
        )

    def __str__(self) -> str:
        return render(self.application.claim)


def applied_statement(name: str, application: Application) -> str:
    """The ledger's words for an application: the theorem, for what, its conclusion.

    ``by on_the_triangle, for every D with pde and stored: forall a in
    Fin(p + 1), b in Fin(p + 1) where a + b <= p. R[a, b] == D(a)(b)``: the
    variables nothing bound, and the hypotheses nothing matched, by the names
    the theorem gives them, which the full claim (:attr:`Application.claim`)
    spells out.
    """
    from loopty.hypotheses import held_under

    conclusion = application.conclusion
    if conclusion is None:
        conclusion = application.claim
    under, given = held_under(application.theorem)
    unmatched = [name for name in application.unmatched if name not in given]
    words = f"by {name} {under}" if under else f"by {name}"
    if application.free:
        words += f", for every {', '.join(application.free)}"
        if unmatched:
            words += f" with {' and '.join(unmatched)}"
    elif unmatched:
        words += f", given {' and '.join(unmatched)}"
    return f"{words}: {render(conclusion)}"


def application_facts(
    term: Term,
    theorems: Sequence[Any],
    post: Fact,
    owner: str,
    *,
    module: str | None = None,
    line: int | None = None,
) -> list[Fact]:
    """One fact per claim a cited theorem gives (:func:`theorem_applications`).

    ``decided`` by ``theorem``: the theorem's hypotheses it matched are
    conjuncts of the postcondition, so its conclusion holds of the kernel's
    arrays, resting on the theorem's fact and the postcondition's. Lean
    checks the application (:class:`Applied`).
    """
    if term.post is None:
        return []
    given = list(conjuncts(term.post))
    out = []
    for application in theorem_applications(term, theorems, given):
        theorem = application.theorem
        name = getattr(theorem, "__name__", "theorem")
        identifier = getattr(theorem, "fact_id", name)
        bindings = ", ".join(
            f"{var} = {value if isinstance(value, str) else render(value)}"
            for var, value in application.bindings
        )
        out.append(
            Fact(
                id=fact_id(APPLIED, owner, module=module, line=line, detail=identifier),
                kind=APPLIED,
                statement=applied_statement(name, application),
                term=Applied(term, application, tuple(given), owner),
                status=Status.DECIDED,
                decided_by="claim",
                provenance={
                    "theorem": identifier,
                    "bindings": bindings,
                    "matched": [
                        render(given[position]) for _k, position in application.matched
                    ],
                    "rule": (
                        f"the hypotheses of {name} it matched are conjuncts of the "
                        f"postcondition of {term.name}, so its conclusion holds at "
                        f"{bindings}"
                    ),
                },
                where=post.where,
                owner=owner,
                rests_on=(identifier, post.id),
            )
        )
    return out


def unapplied_facts(
    term: Term,
    theorems: Sequence[Any],
    post: Fact | None,
    owner: str,
    *,
    raised: Mapping[int, str] | None = None,
    module: str | None = None,
    line: int | None = None,
    where: str = "",
) -> list[Fact]:
    """A row for each cited theorem that gives the kernel nothing, with why (#234).

    ``theorems`` are the ones no fact of the kernel rests on: no row of
    theirs (:func:`application_facts`, :func:`loopty.summands.series_facts`),
    and no conjunct of the postcondition decided under them. The citation
    claims the theorem applies, and nothing established that, so the row is
    ``assumed``, kind ``by-theorem``, and says why under ``declined``, which
    ``lanky check`` prints under the table: a hypothesis about an array the
    kernel writes that no conjunct of the postcondition is, with the nearest
    conjunct; a conjunct about reals that rounds; a binding no array or size
    can stand for; a conclusion that rounds; the postcondition refuted, or
    none; and an exception, as the exception (``raised``, by the theorem's
    ``id``). Before, such a theorem was not in the ledger at all.
    """
    from loopty.summands import _identity, _series_theorem

    raised = dict(raised or {})
    outcomes: dict[int, Any] = {}
    if post is not None and post.status is not Status.REFUTED and term.post is not None:
        pending = [theorem for theorem in theorems if id(theorem) not in raised]
        try:
            given = list(conjuncts(term.post))
            for theorem, found in theorem_outcomes(term, pending, given):
                outcomes[id(theorem)] = found
        except Exception as exc:  # noqa: BLE001 - said as the exception
            for theorem in pending:
                raised[id(theorem)] = (
                    f"matching its hypotheses raised {type(exc).__name__}: {exc}"
                )
    out = []
    for theorem in theorems:
        name = getattr(theorem, "__name__", "theorem")
        identifier = getattr(theorem, "fact_id", name)
        series = _series_theorem(theorem)
        reasons = []
        if id(theorem) in raised:
            reasons.append(raised[id(theorem)])
        elif _identity(theorem) is not None:
            reasons.append(
                "it is an identity, which puts a series claim's terms in the "
                f"kernel's spelling, and no series {term.name} adds is read with it"
            )
        else:
            if series is not None:
                reasons.append(
                    f"no sum of {term.name} adds the terms of its series "
                    f"{series.family}({series.binder.name})"
                )
            if post is None or term.post is None:
                if series is None:
                    reasons.append(
                        f"{term.name} has no postcondition whose conjuncts would be "
                        "its hypotheses"
                    )
            elif post.status is Status.REFUTED:
                reasons.append(
                    f"the postcondition of {term.name} is refuted, and a claim is "
                    "applied to one that holds"
                )
            else:
                found = outcomes.get(id(theorem))
                if isinstance(found, str):
                    reasons.append(found)
        if not reasons:
            continue
        why = "; and ".join(reasons)
        out.append(
            Fact(
                id=fact_id(APPLIED, owner, module=module, line=line, detail=identifier),
                kind=APPLIED,
                statement=f"{name}, which {term.name} cites, applies to it",
                term=None,
                status=Status.ASSUMED,
                provenance={
                    "theorem": identifier,
                    "unapplied": why,
                    "declined": f"{name} gives {term.name} nothing: {why}",
                },
                where=post.where if post is not None else where,
                owner=owner,
            )
        )
    return out


# }}}
