"""The Term IR: what tracing produces and what lowering consumes.

A term is a kernel in one form: its parameters with their array types, its free
size parameters, and a tuple of statements. Every statement carries its iteration
domain as an isl set, so a statement's type is an isl object and nothing has to
be re-derived from source. Index expressions are pymbolic, in the array's own
index-type axes rather than flattened, so a layout change is a change of map and
not a rewrite of the indices. The ``where`` string is ``file:line`` taken from
the frame that executed the assignment, which is how diagnostics point back at
Python source without an AST pass.

The dataclasses are frozen: a term is a value, and a transformation produces a
new one rather than mutating the old, which is what lets a cast be checked by
comparing the two.
"""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass
from typing import Any

import islpy as isl
import pymbolic.primitives as prim

__all__ = [
    "COUNT_PARAM",
    "COUNT_PARAM_REFLECTED",
    "OFFSETS_CANDIDATES",
    "Access",
    "ArrType",
    "Expression",
    "Hypothesis",
    "Reduction",
    "Requirement",
    "Scope",
    "Stmt",
    "Term",
    "count_param_names",
    "declared_layout",
    "declared_offsets",
    "free_name_sorts",
    "free_name_sorts_message",
]

#: A pymbolic expression. Kept loose on purpose: lanky's term classes (``Sum``,
#: ``Forall``, ``Abs``) and pymbolic's primitives both appear here.
Expression = Any

#: Candidate names for the offsets array of a ragged axis, most specific first.
#: The first one that is a parameter of the term is the array a ragged access is
#: flattened through; if none is, lowering adds an argument named
#: ``off_<counts>``. The rule lives here rather than in :mod:`loopty.lower`
#: because several modules read it: lowering, which indexes through the
#: offsets, :func:`loopty.flow.statement_accesses`, which lists that index as a
#: read, and the native run and the interpreter, which index through the same
#: array (:func:`declared_layout`).
OFFSETS_CANDIDATES = ("off_{counts}", "{counts}_off", "off")

#: How a ragged loop bound appears as a parameter of a statement domain: the
#: counts name, an underscore, and the enclosing iname. For a loop over
#: ``val.dom[r]`` of ``val: Arr[Fin[n], Fin[cnt], Real]`` that is ``cnt_r``, and
#: lowering assigns it ``off[r+1] - off[r]`` (or ``cnt[r]`` when the counts array
#: itself is a parameter) in a scalar temporary inside the ``r`` loop. It lives
#: here, with :data:`COUNT_PARAM_REFLECTED`, because two modules recognize it:
#: lowering, which assigns the parameter, and :mod:`loopty.flow`, which lists
#: the read that assignment makes.
COUNT_PARAM = "{counts}_{iname}"

#: The same bound as the tracer spells it when it reflects the non-affine term
#: ``cnt[r]`` into a fresh isl parameter (see ``loopty.idx``). Both spellings are
#: recognized, so that a hand-written term and a traced one lower the same way.
#:
#: It is a spelling and not the definition. A traced term records what it
#: actually allocated on :attr:`Term.reflected`, which is where a parameter that
#: had to be suffixed to dodge a collision is found; this pattern is the
#: fallback for a term written by hand, which records nothing.
COUNT_PARAM_REFLECTED = "nl_{counts}_{iname}"


def count_param_names(counts: str, iname: str) -> tuple[str, ...]:
    """Every spelling of one ragged bound parameter, most direct first."""
    return (
        COUNT_PARAM.format(counts=counts, iname=iname),
        COUNT_PARAM_REFLECTED.format(counts=counts, iname=iname),
    )


def declared_offsets(params: Iterable[tuple[str, Any]], counts: str) -> str | None:
    """The parameter holding the offsets of a ragged axis over ``counts``.

    The first of :data:`OFFSETS_CANDIDATES` that names a parameter, or ``None``
    when the kernel declares none of them, in which case the offsets are an
    argument lowering adds and nothing in the body can name, let alone write.
    """
    names = {name for name, _ in params}
    for pattern in OFFSETS_CANDIDATES:
        candidate = pattern.format(counts=counts)
        if candidate in names:
            return candidate
    return None


def declared_layout(
    params: Iterable[tuple[str, Any]],
    stated: Iterable[tuple[str, str | None]] = (),
) -> dict[str, tuple[str | None, str | None]]:
    """The arguments each ragged parameter is indexed through, as lowering does.

    ``{"val": ("cnt", "off")}`` for ``val: Arr[Fin[n], Fin[cnt], Real]`` beside
    ``cnt`` and ``off`` parameters: the first is the counts parameter, which
    bounds a row (``val.dom[r]`` is ``cnt[r]`` long), and the second the offsets
    parameter (:func:`declared_offsets`), where a row starts (``val[r, j]`` is
    ``val[off[r] + j]``). Either is ``None`` when the kernel does not declare
    it, and a ragged parameter that declares neither is left out: it has only
    its own layout.

    ``stated`` is a term's :attr:`Term.offsets`: the offsets of a counts family
    the term states rather than leaves to the parameter names, which it wins
    over. A program's term states them, see :mod:`loopty.compose`.

    This is the layout the lowered kernel reads, because those are the
    arguments it is handed, and the one the native run and the interpreter read
    too, so that a kernel that writes its counts or its offsets means one thing
    however it is run. See :meth:`loopty.arr.Arr.through`.
    """
    listed = tuple(params)
    types = dict(listed)
    overrides = dict(stated)
    out: dict[str, tuple[str | None, str | None]] = {}
    for name, typ in listed:
        if not isinstance(typ, ArrType) or not any(typ.ragged):
            continue
        axis = typ.ragged.index(True)
        size = typ.axes[axis]
        if axis != 1 or len(typ.axes) != 2 or not isinstance(size, prim.Variable):
            continue
        counts = size.name if isinstance(types.get(size.name), ArrType) else None
        if size.name in overrides:
            offsets = overrides[size.name]
        else:
            offsets = declared_offsets(listed, size.name)
        if counts is None and offsets is None:
            continue
        out[name] = (counts, offsets)
    return out


@dataclass(frozen=True)
class Access:
    """One array reference.

    ``indices`` are in the array's index-type axes, not in flat storage: a
    ragged array is indexed ``(r, j)``, and the offsets enter only at lowering.
    """

    array: str
    indices: tuple[Expression, ...]


@dataclass(frozen=True)
class Reduction:
    """A reduction over ``inames``, the term ``lanky.sum`` and friends lower to.

    ``domain`` is the reduced iteration space, with the enclosing inames present
    as parameters or dimensions so that a ragged reduction bound can depend on
    the row. ``exactness`` is the floating-point contract of the accumulation:
    ``exact`` forbids reassociation (no trees, no atomics), ``reassoc`` permits
    it and marks the result, ``approx`` carries a tolerance.

    ``loop_domain`` is ``domain`` before the statement's guard and the
    generator's own ``if`` clause narrowed it, as :attr:`Stmt.loop_domain` is
    a statement's: the points at which the sum starts and reads its bounds,
    since a ``when`` masks the write and does not skip the right-hand side,
    and ``for k in val.dom[r] if k == j`` reads the row's length before it
    tests ``k``. isl simplifies the narrowed set, and ``k == j`` leaves no
    constraint of the sum's own bound in it, so what reads a bound reads it
    here (#111). ``None`` means "the same as ``domain``", which is right for
    a sum with no guard around it and no clause, and for a term written by
    hand.
    """

    op: str
    inames: tuple[str, ...]
    domain: isl.Set
    body: Expression
    exactness: str
    loop_domain: isl.Set | None = None


@dataclass(frozen=True)
class Stmt:
    """One statement instance family: a domain plus what it computes.

    ``inames`` are the enclosing loop variables, outer to inner, and ``domain``
    is the isl set of their values. ``guard`` is the condition of an enclosing
    ``when`` block, which masks the write rather than skipping it.

    ``kind`` is ``"assign"`` or ``"accumulate"``, and the difference is a claim
    about ``expr``, not a licence to guess:

    * ``"assign"`` means ``expr`` does not read the cell ``assignee`` names.
    * ``"accumulate"`` means it does. ``expr`` is always the **complete**
      right-hand side, so ``y[r] += t`` is recorded as
      ``assignee=y[r], expr=y[r] + t``, never as the increment ``t`` alone.

    That convention is fixed here because it cannot be recovered afterwards:
    ``y[r] = y[r] + t`` and ``y[r] += t`` are the same Python, an increment and
    a full right-hand side are both well-formed expressions, and a lowering that
    guessed wrong would silently double the term. :mod:`loopty.lower` therefore
    checks the invariant and refuses a term that breaks it, rather than
    repairing it; :mod:`loopty.flow` reads the same convention when it decides
    that the accumulated cell's read is already covered by the ``acc``
    footprint.

    ``order`` is the statement's position in the loop tree, one integer per
    level from the outermost block down to the statement itself, so that
    ``order`` interleaved with ``inames`` is the 2d+1 time vector of the source
    schedule. It is what tells two statements in the same loop nest apart from
    two statements in sequence; see :mod:`loopty.flow`. Defaulted, because a
    term written by hand in a test does not have to care.

    ``loop_domain`` is the enclosing loop nest *before* any guard narrowed it:
    the set over which the guard's own reads happen, because a ``when``
    evaluates its whole condition at every point of the nest (Python's ``&`` is
    eager) and masks the write rather than skipping the block. ``None`` means
    "the same as ``domain``", which is right for a statement without a guard
    and for a term written by hand.

    ``unnarrowed`` lists the conjuncts of ``guard`` that ``domain`` does not
    state, each as ``(conjunct, why)`` with the conjunct as the body spells it.
    isl states an affine comparison of integers (loop variables, sizes, scalars
    of an integral sort) and nothing else, so a guard that reads an array,
    compares with ``!=``, or compares with a ``Real`` scalar is evaluated at
    run time only, and ``domain`` is wider than the instances that write. That
    is sound, since an obligation over a wider set is harder, but a fact stated
    over the domain is then about instances that write nothing, and says so
    (see :func:`loopty.typing.in_bounds_facts`). Empty for a statement whose
    guard is stated whole, and for a term written by hand.
    """

    id: str
    inames: tuple[str, ...]
    domain: isl.Set
    assignee: Access
    expr: Expression
    kind: str
    guard: Expression | None
    where: str
    order: tuple[int, ...] = ()
    loop_domain: isl.Set | None = None
    unnarrowed: tuple[tuple[str, str], ...] = ()


@dataclass(frozen=True)
class ArrType:
    """The type of an array parameter: axis sizes, element sort, raggedness.

    An axis whose size names another array parameter is ragged: for
    ``val: Arr[Fin[n], Fin[cnt], Real]`` the axes are ``(n, cnt)`` and
    ``ragged == (False, True)``, meaning the bound of the second axis at row
    ``r`` is ``cnt[r]``.

    An array over a polyhedral domain (``Arr[Where[...], Real]``, a
    ``Sigma[...]`` or a union of pieces; see :mod:`loopty.domain`) has its
    index set in ``domain`` instead, and no axes of its own: its shape is the
    domain, and a rule that reads ``axes`` as a box finds none to read.
    """

    axes: tuple[Expression, ...]
    dtype: object
    ragged: tuple[bool, ...]
    domain: Any = None

    def __post_init__(self) -> None:
        if len(self.axes) != len(self.ragged):
            raise ValueError(
                f"{len(self.axes)} axes but {len(self.ragged)} raggedness flags"
            )
        if self.domain is not None and self.axes:
            raise ValueError("an array over a domain has no axes of its own")

    @property
    def ndim(self) -> int:
        """Number of index axes."""
        if self.domain is not None:
            return int(self.domain.ndim)
        return len(self.axes)

    @property
    def shape_terms(self) -> tuple[Expression, ...]:
        """The terms the array's shape mentions sizes in.

        Its axes, or for an array over a domain one variable for each size the
        domain names, so that a rule collecting the sizes of a signature finds
        the domain's too.
        """
        if self.domain is None:
            return self.axes
        from lanky.terms import Var

        return tuple(Var(name) for name in sorted(self.domain.size_names()))


@dataclass(frozen=True, eq=False)
class Hypothesis:
    """A proposition a fact about a program may assume, and what it rests on.

    ``claim`` is a proposition about the program's arrays and sizes, in the
    program's names: a callee's postcondition after the call, a theorem the
    program cites instantiated at its arrays, the type the program's
    contract checks of an argument nothing has written yet, or the zeros an
    ``Arr.zeros_like`` starts an array with. ``source`` says which, in words,
    for a fact's provenance; ``rests_on`` names the facts the claim is
    established by, which a fact that uses it rests on in turn (lanky's
    ``rests_on``), and is empty for what the program's contract or its own
    statements establish. ``mentions`` names the arrays the claim is about,
    so that a call writing one of them can retire it.

    Compared by identity, as :class:`Requirement` and :class:`Scope` are: a
    claim is a lanky term, whose ``==`` builds a proposition.
    """

    claim: Any
    source: str
    rests_on: tuple[str, ...] = ()
    mentions: frozenset[str] = frozenset()


@dataclass(frozen=True, eq=False)
class Requirement:
    """What a call's contract checks of an array an earlier call wrote.

    A kernel's requirements on its inputs are its argument types, and three
    of them are about what an array's cells hold: an element of a ``Fin[m]``
    sort is a point of it (``kind="element"``), one of the ``Nat`` sort is
    not negative (``kind="nat"``), and the offsets a ragged family is read
    through are the offsets its counts give (``kind="layout"``), which makes
    them naturals too. Natively the call's contract checks them; a program
    is one call, so where an earlier call wrote the array, the requirement
    is an obligation of the program (see :mod:`loopty.compose`).

    ``call`` is the label of the call (``gather``, ``scan@2``) and
    ``definition`` the callee's (``spmv.scan@69``); ``param`` is the
    callee's parameter and ``array`` the program's array passed for it, and
    ``writer`` says what wrote the array first. ``statement`` is the
    requirement in words, and ``claim`` as a proposition in the program's
    names.

    ``offered`` lists the hypotheses that held at the call. When isl
    decides the requirement under them, ``question`` is the set of cells
    that break it where the hypotheses ``used`` hold, an empty one, and the
    fact rests on what those rest on. Otherwise ``question`` is ``None``,
    ``reason`` says why, and the requirement is a checked point: the
    lowered program checks the cells between the calls and sets the cell
    of the one-cell array ``flag`` when one fails, every later statement is
    guarded by that cell, and the compiled run raises ``message``.

    A decided requirement is a checked point too when a fact it rests on is
    not at least ``tested``: a postcondition its kernel's native runs
    refuted, or a theorem the property tester does not pass. It keeps its
    ``question``, and ``reason`` says why it is checked all the same; the
    compiled program trusts no more than a run has borne out (#115).
    """

    call: str
    kernel: str
    definition: str
    where: str
    param: str
    array: str
    kind: str
    writer: str
    statement: str
    claim: Any
    offered: tuple[Hypothesis, ...] = ()
    used: tuple[Hypothesis, ...] = ()
    question: Any = None
    reason: str = ""
    flag: str | None = None
    message: str = ""

    @property
    def decided(self) -> bool:
        """Whether isl decided the requirement; ``flag`` says whether it is checked."""
        return self.question is not None


@dataclass(frozen=True, eq=False)
class Scope:
    """What holds where one call of a program runs.

    ``statements`` are the ids of the call's statements in the program's
    term, and ``hypotheses`` what may be assumed there: the postconditions
    of earlier calls that nothing has written over, the theorems the
    program cites, instantiated at them, and the callee's own argument
    types, which the program's contract checks when it starts, or a
    requirement decides or checks at the call. A callee's fact that its own
    term leaves ``assumed`` can be decided here; see
    :func:`loopty.typing.scoped_in_bounds_facts`.
    """

    call: str
    kernel: str
    where: str
    statements: tuple[str, ...]
    hypotheses: tuple[Hypothesis, ...] = ()


@dataclass(frozen=True)
class Term:
    """A traced kernel.

    ``params`` are in signature order, each with its :class:`ArrType` (arrays) or
    a scalar sort. ``sizes`` are the free size parameters, which are the isl
    parameters of every statement domain. ``post`` is the return annotation, a
    proposition about the parameters, which becomes a fact to establish rather
    than an assertion to trust.

    ``reflected`` names the isl parameters that stand for the term's non-affine
    subexpressions, each with the subexpression it stands for: ``nl_cnt_r`` and
    ``cnt[r]``. It travels with the term because the allocation is a table
    rather than a rule (see :class:`loopty.idx.Reflections`), so anything that
    builds another isl set about this term, or that has to read ``cnt[r]`` back
    out of a domain parameter, has to be told what was allocated instead of
    guessing from the spelling. A term written by hand leaves it empty and is
    read by spelling, which is what :data:`COUNT_PARAM_REFLECTED` is for.

    Four fields are empty for a kernel's term and filled in for a program's
    (:mod:`loopty.compose`), which is one term made of several kernels':

    * ``temporaries`` are the arrays the term writes and reads that are not
      parameters, each with its type: an array a program makes for itself,
      which the lowering declares as a loopy temporary rather than an
      argument, so nobody passes it.
    * ``offsets`` states the offsets array each counts family is indexed
      through, as ``(counts, offsets)``, with ``None`` for the array's own
      offsets, which lowering adds as an argument. A kernel leaves this to the
      names of its parameters (:func:`declared_offsets`); a program cannot,
      because its parameters are named by the program and not by the kernels
      whose layout they carry, and a program parameter called ``off`` is not
      the offsets of a kernel that declares none. See :meth:`offsets_of`.
    * ``where`` is ``file:line`` of the definition when the term is not one
      kernel's statements, for the facts about the whole term; a kernel's term
      leaves it empty, and those facts point at its first statement.
    * ``temporaries_like`` pairs a temporary with the parameter whose dtype it
      has natively, ``(f, u)`` for ``f = Arr.zeros_like(u)`` given no dtype.
      Natively that dtype is whatever ``u`` is called with, so it is checked
      when the compiled term is run
      (:func:`loopty.contract.inherited_storage`).

    Two more are a program's alone, and are how its callees' facts travel
    from one call to the next:

    * ``requirements`` lists what a call's contract checks of an array an
      earlier call wrote (:class:`Requirement`), each decided under the
      hypotheses that held at the call or checked by the lowered program
      between the calls, through a one-cell temporary each
      (:attr:`checks`).
    * ``scopes`` gives, call by call, the hypotheses that held where it ran
      (:class:`Scope`).
    * ``deferred_offsets`` names the offsets arrays an earlier call writes
      before any call reads rows through them. The contract of the compiled
      program does not compare them with the rows' own offsets when it
      starts, as a native call that does not read through them does not: a
      layout requirement does, where they are first read.
    """

    name: str
    params: tuple[tuple[str, ArrType | object], ...]
    sizes: tuple[str, ...]
    stmts: tuple[Stmt, ...]
    post: Expression | None
    reflected: tuple[tuple[str, Expression], ...] = ()
    temporaries: tuple[tuple[str, ArrType], ...] = ()
    offsets: tuple[tuple[str, str | None], ...] = ()
    where: str = ""
    temporaries_like: tuple[tuple[str, str], ...] = ()
    requirements: tuple[Requirement, ...] = ()
    scopes: tuple[Scope, ...] = ()
    deferred_offsets: tuple[str, ...] = ()

    @property
    def param_names(self) -> tuple[str, ...]:
        """Parameter names, in signature order."""
        return tuple(name for name, _ in self.params)

    @property
    def checks(self) -> tuple[tuple[str, str], ...]:
        """The checked points: each flag temporary, with what a failure raises.

        In the order the program reaches them. A flag is a one-cell
        temporary of the term that a check statement sets when a cell fails
        its requirement; the lowering passes it as an argument, so that the
        executor can read it after the run and raise the message, and the
        interpreter raises it too (:class:`Requirement`).
        """
        return tuple(
            (requirement.flag, requirement.message)
            for requirement in self.requirements
            if requirement.flag is not None
        )

    @property
    def checked_arrays(self) -> frozenset[str]:
        """The arrays a checked point reads, each stored in 64 bits compiled.

        A checked point reads, between two calls, the cells an earlier call
        wrote, so that the compiled program stops where the later call's
        native contract refuses them. The native run holds an integer in 64
        bits as a rule, and the compiled one stores an index in fewer:
        ``perm[i] = 2**32`` was narrowed to ``0`` by the store, and the
        checked point read a point of ``Fin[n]`` where the native contract
        refuses ``2**32`` (#128). So the lowering stores an integral array a
        checked point reads in 64 bits (:func:`loopty.contract.array_storage`),
        and the compiled run refuses one passed in fewer, which the native run
        would narrow (:func:`loopty.contract.checked_storage`): the checked
        point reads what the native contract reads.
        """
        return frozenset(
            requirement.array
            for requirement in self.requirements
            if requirement.flag is not None
        )

    @property
    def array_types(self) -> dict[str, ArrType]:
        """The type of every array the term touches: parameters and temporaries."""
        out = {name: typ for name, typ in self.params if isinstance(typ, ArrType)}
        out.update(self.temporaries)
        return out

    def offsets_of(self, counts: str) -> str | None:
        """The offsets array the rows over ``counts`` are indexed through.

        What :attr:`offsets` states for the family, when it states anything,
        and otherwise the parameter :func:`declared_offsets` finds by name.
        ``None`` means the array's own offsets, an argument the lowering adds
        and nothing in the term can name.
        """
        stated = dict(self.offsets)
        if counts in stated:
            return stated[counts]
        return declared_offsets(self.params, counts)

    @property
    def reflections(self) -> Any:
        """This term's :class:`loopty.idx.Reflections`, rebuilt from the record.

        Every parameter it already allocated is adopted, and every name the term
        uses is reserved, so a set built later (the cell set of an in-bounds
        obligation, say) reuses the parameter for a term it has already seen and
        cannot collide with one it has not.
        """
        from loopty.idx import Reflections

        table = Reflections()
        table.reserve(self.param_names)
        table.reserve(name for name, _ in self.temporaries)
        table.reserve(self.sizes)
        for stmt in self.stmts:
            table.reserve(stmt.inames)
        for name, expr in self.reflected:
            table.adopt(name, expr)
        return table

    def stmt(self, stmt_id: str) -> Stmt:
        """The statement with the given id."""
        for stmt in self.stmts:
            if stmt.id == stmt_id:
                return stmt
        raise KeyError(stmt_id)


# {{{ sorts that are free names


#: What to write instead of a builtin type name, by that name.
_SORT_HINTS = {
    "float": "Real (from lanky.prelude) or a numpy type such as np.float64",
    "complex": "a numpy type such as np.complex128",
    "int": (
        "Nat or Int (from lanky.prelude), Fin[n] for an index, or a numpy "
        "type such as np.int64"
    ),
}


def free_name_sorts(params: Iterable[tuple[str, Any]]) -> tuple[tuple[str, str], ...]:
    """Every parameter whose sort is a bare free name, with that name.

    A kernel's annotations are evaluated by lanky in a scope that invents the
    names it does not define, which is how a size such as ``n`` in
    ``Arr[Fin[n], Real]`` comes to exist. Under ``from __future__ import
    annotations`` the builtins are among those names, so ``a: float`` and
    ``Arr[Fin[n], float]`` give the sort ``Var("float")``: a free variable,
    not a type. It has no numpy dtype, it is not an integral sort, and lanky
    reads anything that is not one of its sorts as an exact index type, so an
    accumulation of it would be called ``exact``. A misspelled or unimported
    sort (``Reel``) arrives the same way. The sort of a scalar parameter and the
    element sort of an array are the places a sort is written; a free name in
    an axis is a size, and is not asked about here.
    """
    out: list[tuple[str, str]] = []
    for name, typ in params:
        sort = typ.dtype if isinstance(typ, ArrType) else typ
        if isinstance(sort, prim.Variable):
            out.append((name, sort.name))
    return tuple(out)


def free_name_sorts_message(
    owner: str, params: Iterable[tuple[str, Any]], found: Iterable[tuple[str, str]]
) -> str:
    """The refusal of a signature whose sorts include free names."""
    arrays = {name for name, typ in params if isinstance(typ, ArrType)}
    found = tuple(found)
    items = [
        f"the elements of {name} as {sort}" if name in arrays else f"{name}: {sort}"
        for name, sort in found
    ]
    listing = items[-1]
    if len(items) > 1:
        listing = f"{', '.join(items[:-1])} and {listing}"
    hints = []
    for sort in dict.fromkeys(sort for _name, sort in found):
        hint = _SORT_HINTS.get(sort)
        if hint is None:
            hint = (
                "Real or a numpy type such as np.float64 for a floating-point "
                "value, and Nat, Int or Fin[n] for a whole number, or import "
                "the sort you meant"
            )
        hints.append(f"for {sort} write {hint}")
    advice = "; ".join(hints)
    return (
        f"{owner} declares {listing}, and a sort written that way is a free "
        "name, not a type. An annotation is evaluated in a scope that invents "
        "every name it does not define, and under 'from __future__ import "
        "annotations' that includes builtins such as float and int, so loopy "
        "would get no dtype from it and the ledger would call it an exact index "
        f"type. {advice[0].upper()}{advice[1:]}."
    )


# }}}
