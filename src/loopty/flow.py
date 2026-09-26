"""Footprints and the dependence relation.

The type of a statement is its domain (an isl set) plus three isl maps from
statement instances to array cells: ``reads``, ``writes`` and ``accs``
(accumulations). Dependences are not declared, they are *defined* from the
footprints: two instances depend on each other when they touch the same cell,
at least one of them writes it, and the source order runs one before the other.
isl's own flow analysis is a faster way to get the same relation, but it is a
hint; keeping the definition here in one place is what lets a certifier recheck
a schedule against the definition rather than against a cached answer.

Three modelling decisions live in this module.

*The instance space is padded and statement-tagged.* Every statement instance is
a point ``[s, d0, ..., d_{D-1}]`` where ``s`` is the statement's index in source
order, ``d_k`` is its ``k``-th enclosing iname, and the dimensions past its own
depth are pinned to zero. Statements therefore all live in one isl space, so a
dependence relation and a schedule are plain :class:`islpy.Map` objects rather
than union maps, and the oracle's ``is_monotone`` can take them unchanged. The
price is that a witness is a padded tuple, which
:func:`loopty.typing.render_instance` turns back into ``S1[r=2, j=0]``.

*Time is the 2d+1 vector.* The source schedule maps an instance to
``[p0, d0, p1, d1, ..., pD]``, alternating the position of the enclosing block
in its parent with the iname at that level. Two statements in the same loop nest
are then correctly interleaved, and statements in sequence are correctly
ordered, which a schedule of the form "statement index first" would get wrong.

*Ragged bounds become reflected parameters.* The domain of a loop over
``val.dom[r]`` is ``{[r, j] : 0 <= r < n and 0 <= j < cnt[r]}``, and ``cnt[r]``
is an array read, which Presburger arithmetic cannot express. It is reflected as
a fresh isl parameter named after the term (``nl_cnt_r``, the convention
:func:`loopty.idx.to_set` already uses), so the domain becomes
``[n, nl_cnt_r] -> {[r, j] : 0 <= r < n and 0 <= j < nl_cnt_r}``. isl answers
its questions for *every* value of a parameter, so a subset or emptiness verdict
proved this way is a proof schema over all counts, which is exactly the
per-generic-row statement wanted, and it is sound because it can only widen.
The parameter does not hide the array it comes from: the read of ``cnt[r]`` that
computes it is an access of the statement like any other
(:func:`layout_reads`), so a statement that writes the counts is ordered
against the rows they bound.

The allocation of those names is a table and not a function of the term, because
the readable spelling is not injective: ``cnt[r]`` and ``cnt*r`` both read
``nl_cnt_r``, and a kernel is free to declare a size of that name. One
:class:`loopty.idx.Reflections` per term keys the parameter on the *term* and
suffixes the name when it is taken, so two different bounds are never silently
asserted equal, and it is shared by every set built about that term, so the
statement domain, the reduction domain and the cell set of an in-bounds
obligation all call ``cnt[r]`` by the same parameter. The table travels on
:attr:`loopty.term.Term.reflected`.

The limit of that choice is worth stating, and is stated again in the README's
status list, because it is the one place where loopty knows less than a reader
might assume. Because the parameter is named after the *term*, ``cnt[r]`` and
``cnt[r + 1]`` are unrelated parameters: nothing here knows that counts are
non-negative, that they sum to the offsets, or that ``off`` is monotone.

The visible consequence is the flat CSR layout. An access written against flat
storage, ``val[off[r] + j]``, is not quasi-affine (``off[r]`` is an array read)
and its element type does not bound it either, so :mod:`loopty.typing` states
the obligation and reports it **ASSUMED**, with the reason in its provenance.
It is not ``DECIDED``, and it must not be: deciding it needs the scan kernel's
postcondition (``off`` is monotone and ends at ``nnz``) as a hypothesis, and no
oracle here can take one. Widening the reflected parameter until isl could
answer would turn "unknown" into "proved", which is the one failure mode a
ledger exists to prevent. The ragged form, ``val[r, j]`` over ``0 <= j <
cnt[r]``, *is* decided, because raggedness is in the type there rather than in
the arithmetic; that is the form the demos and the tracer use.

The alternative formulation, ``off[r] <= a < off[r + 1]`` with the offsets
constrained by the scan recurrence, keeps those relations and is the natural
next step. It is not implemented, and it is not needed for in-bounds or
disjointness on the ragged form, which is all the MVP's typing rules ask for.

*Distinct parameters are distinct storage.* :func:`dependences` compares
footprints array by array and reports nothing between two differently named
arrays, so the whole dependence relation, and every legality verdict derived
from it, rests on the assumption that two parameters never name overlapping
memory. That is an assumption about the *call*, not about the term: a kernel
reading ``x[i - 1]`` and writing ``y[i]`` carries no dependence and may tag
``i`` parallel, and the same kernel called with ``x is y`` is a race. The
assumption is therefore enforced where calls happen rather than left implicit:
:mod:`loopty.contract` detects overlapping storage among distinct array
arguments with ``numpy.shares_memory`` and raises ``ValueError`` naming both,
and :class:`loopty.executor.LoopyExecutor` and
:meth:`loopty.kernel.Kernel.__call__` both ask it before running anything.
Without that check the condition is invisible: the differential test copies each
argument on its own, which destroys the alias and makes the two runs agree.
"""

from __future__ import annotations

import re
from collections.abc import Collection, Mapping, Sequence
from dataclasses import dataclass
from typing import Any

import islpy as isl
import pymbolic.primitives as prim
from lanky.terms import init_args

from loopty.idx import Reflections
from loopty.term import ArrType, Stmt, Term, count_param_names, declared_offsets

__all__ = [
    "Footprint",
    "LayoutRead",
    "assume_sizes",
    "bounds_dimension",
    "NonAffine",
    "cell_set",
    "counts_families",
    "dependences",
    "free_names",
    "domain_set",
    "expr_text",
    "footprints",
    "instance_domain",
    "instance_space_depth",
    "layout_reads",
    "loops_outside",
    "pad_map",
    "ragged_bound_params",
    "schedule_of",
    "size_names",
]


class NonAffine(ValueError):
    """Raised when a term has to be reflected but the caller forbade it."""


def expr_text(
    expr: Any,
    rename: Mapping[str, str] | None = None,
    reflected: dict[str, Any] | None = None,
    table: Reflections | None = None,
) -> str:
    """Render ``expr`` in isl's input syntax, reflecting what is not affine.

    ``rename`` maps variable names to isl dimension names, which is how an
    iname ``r`` becomes the dimension ``d0`` of the padded instance space.
    ``reflected`` collects the parameters this rendering used for non-affine
    terms (an array read such as ``cnt[r]``, a product of two unknowns); the
    parameter is looked up by the term *before* renaming, so the same term
    always reflects to the same parameter. Passing ``reflected=None`` refuses to
    widen and raises :class:`NonAffine` instead.

    ``table`` is the :class:`~loopty.idx.Reflections` that allocates those
    parameters. It is what makes the allocation collision-free and shared across
    every set built for one term; callers that hold a term pass its table.
    Without one, each rendering allocates in a table of its own, which is right
    for a single set standing alone and wrong for a set that has to be compared
    with another.
    """
    rename = rename or {}
    if isinstance(expr, bool):  # pragma: no cover - a bool is not an index
        return "1" if expr else "0"
    if isinstance(expr, int):
        return str(expr)
    if isinstance(expr, prim.Variable):
        return rename.get(expr.name, expr.name)
    if isinstance(expr, prim.Sum):
        parts = [expr_text(c, rename, reflected, table) for c in expr.children]
        return "(" + " + ".join(parts) + ")"
    if isinstance(expr, prim.Product):
        constants = [c for c in expr.children if isinstance(c, int)]
        others = [c for c in expr.children if not isinstance(c, int)]
        if len(others) <= 1:
            parts = [str(c) for c in constants] + [
                expr_text(c, rename, reflected, table) for c in others
            ]
            return "(" + " * ".join(parts) + ")"
    elif isinstance(expr, prim.FloorDiv | prim.Remainder):
        denominator = expr.denominator
        if isinstance(denominator, int) and denominator > 0:
            numerator = expr_text(expr.numerator, rename, reflected, table)
            if isinstance(expr, prim.FloorDiv):
                return f"floord({numerator}, {denominator})"
            return f"(({numerator}) % {denominator})"
    if reflected is None:
        raise NonAffine(f"{expr!r} is not quasi-affine and may not be reflected here")
    name = (table if table is not None else Reflections()).symbol(expr)
    reflected[name] = expr
    return name


#: Words that may appear in isl text without being a parameter.
_ISL_WORDS = frozenset(
    {
        "floord", "exists", "and", "or", "not",
        "min", "max", "mod", "implies", "false", "true",
    }
)


def free_names(text: str) -> set[str]:
    """The identifiers an isl constraint mentions.

    Parameters are read back off the rendered text rather than off the term,
    because a term that was reflected contributes its fresh parameter and *not*
    the names inside it: once ``cnt[r]`` has become ``nl_cnt_r``, ``cnt`` is no
    longer part of the set.
    """
    return set(re.findall(r"[A-Za-z_][A-Za-z_0-9]*", text)) - _ISL_WORDS


def _assemble(
    dims: Sequence[str],
    constraints: Sequence[str],
    params: Sequence[str],
    nonneg: Collection[str],
) -> str:
    """The isl text of a set, given its dimensions, constraints and parameters.

    A parameter that stands for an extent or a count is non-negative, and isl
    has to be told: a parameter otherwise ranges over all the integers, and an
    obligation as ordinary as ``off[0]`` being in bounds would be refuted at a
    negative size. ``nonneg`` names exactly those parameters. Nothing else about
    them is assumed, in particular no relation between two of them.

    Not every parameter qualifies. A guard brings its own names into a domain,
    and a signed scalar is a perfectly ordinary thing to guard on:
    ``with when(a < 0)`` with ``a : Int`` would make the domain empty if ``a``
    were assumed non-negative here, and an empty domain discharges every
    obligation over it vacuously. Which is why the caller says which names are
    sizes rather than this function assuming all of them are.
    """
    unique = list(dict.fromkeys(params))
    head = f"[{', '.join(unique)}] -> " if unique else ""
    body = ", ".join(dims)
    pieces = [*constraints, *[f"{name} >= 0" for name in unique if name in nonneg]]
    if not pieces:
        return f"{head}{{ [{body}] }}"
    return f"{head}{{ [{body}] : {' and '.join(pieces)} }}"


def size_names(term: Term) -> frozenset[str]:
    """The parameter names of ``term`` that stand for an extent or a count.

    An array axis extent (``n`` in ``Arr[Fin[n], Real]``), the free sizes the
    term records, and the parameter a ragged bound reflects to (``nl_cnt_r``,
    which stands for ``cnt[r]`` and is therefore a number of cells) are all
    non-negative, and :func:`assume_sizes` may say so.

    A scalar the kernel takes as a parameter is not one of those. ``a : Int`` is
    signed, and a guard ``with when(a < 0)`` puts it among the parameters of a
    statement's domain; assuming it non-negative would make that domain empty
    and turn every obligation over it into a vacuous truth. Scalar parameters
    are therefore excluded by name, whatever a derived object's parameter list
    happens to contain.
    """
    scalars = {name for name, typ in term.params if not isinstance(typ, ArrType)}
    out: set[str] = set(term.sizes)
    for _, typ in term.params:
        if isinstance(typ, ArrType):
            out |= _names_in(typ.axes)
    for stmt in term.stmts:
        out |= set(stmt.domain.get_var_names(isl.dim_type.param))
        for reduction in _reductions_in(stmt.expr):
            out |= set(reduction.domain.get_var_names(isl.dim_type.param))
    return frozenset(out - scalars)


def _names_in(expr: Any) -> set[str]:
    """Every variable name occurring in a size term, or a tuple of them."""
    if isinstance(expr, prim.Variable):
        return {expr.name}
    if isinstance(expr, prim.ExpressionNode):
        out: set[str] = set()
        for arg in init_args(expr):
            out |= _names_in(arg)
        return out
    if isinstance(expr, tuple | list):
        out = set()
        for item in expr:
            out |= _names_in(item)
        return out
    return set()


def _reductions_in(expr: Any) -> tuple[Any, ...]:
    """The reductions of an expression. Imported late: trace imports this module."""
    from loopty.trace import reductions_in  # noqa: PLC0415

    return reductions_in(expr)


def assume_sizes(obj: Any, names: Collection[str] | None = None) -> Any:
    """Constrain the size parameters of ``obj`` to be non-negative.

    The same assumption :func:`_assemble` builds into a set it writes, applied
    to a set or map that isl derived (the range of an access map, say), whose
    parameters arrived by alignment and carry no constraints of their own.
    Without it a question is answered for negative array sizes too, and
    ``off[0]`` is "refuted" at ``n = -1``.

    ``names`` says which parameters are sizes; :func:`size_names` computes it
    from a term. With ``names`` omitted every parameter is taken to be one,
    which is right for an object whose parameters all came from array shapes and
    wrong for one that a guard contributed to, so a caller holding a term should
    pass it.
    """
    present = list(obj.get_var_names(isl.dim_type.param))
    if names is not None:
        present = [name for name in present if name in names]
    if not present:
        return obj
    head = "[" + ", ".join(present) + "] -> "
    body = " and ".join(f"{name} >= 0" for name in present)
    return obj.intersect_params(isl.Set(f"{head}{{ : {body} }}"))


def domain_set(
    inames: Sequence[str],
    bounds: Sequence[Any],
    params: Sequence[str] = (),
    constraints: Sequence[str] = (),
    names: Sequence[str] | None = None,
    reflections: Reflections | None = None,
) -> isl.Set:
    """The isl set ``{ [i0, ...] : 0 <= i0 < b0 and ... }`` of a loop nest.

    ``bounds`` are the exclusive upper bounds as terms, outermost first, and may
    mention the enclosing inames (a triangular loop) or reflect a ragged count.
    ``names`` overrides the isl dimension names, which the padded instance space
    needs; the terms are renamed to match.

    Only the names a *bound* contributes are assumed non-negative, along with
    the parameters a ragged bound reflects to and any ``params`` the caller
    names. A name that reaches the set through ``constraints`` alone comes from
    a guard, and a guard may perfectly well be ``a < 0`` on a signed scalar;
    assuming that name non-negative would empty the domain. See
    :func:`_assemble`.

    ``reflections`` is the term's parameter table for non-affine bounds; see
    :func:`expr_text`. A caller that builds more than one set about the same
    term has to pass one, or two sets will call two different things by the same
    name, or one thing by two names.
    """
    if names is None:
        names = tuple(inames)
    rename = dict(zip(inames, names, strict=True))
    table = reflections
    if table is None:
        reserved = {*inames, *names, *params, *_names_in(list(bounds))}
        for piece in constraints:
            reserved |= free_names(piece)
        table = Reflections(reserved)
    reflected: dict[str, Any] = {}
    bound_pieces: list[str] = []
    for name, bound in zip(names, bounds, strict=True):
        bound_pieces.append(
            f"0 <= {name} < {expr_text(bound, rename, reflected, table)}"
        )
    pieces = [*constraints, *bound_pieces]
    free: set[str] = set()
    for piece in pieces:
        free |= free_names(piece)
    parameters = [*sorted(free - set(names)), *params]
    sized: set[str] = set(params) | set(reflected)
    for piece in bound_pieces:
        sized |= free_names(piece)
    return isl.Set(
        _assemble(list(names), pieces, parameters, sized - set(names))
    )


def cell_set(
    arrtype: ArrType,
    indices: Sequence[Any] | None = None,
    names: Sequence[str] | None = None,
    reflections: Reflections | None = None,
) -> isl.Set:
    """The isl set of cells an array has.

    A dense axis contributes ``0 <= a_k < size``. A ragged axis contributes
    ``0 <= a_k < cnt[e]`` where ``cnt`` is the counts array named by the axis
    and ``e`` is the index expression the access uses for the axis before it, so
    that the bound of the fiber is the bound of *that row*. With ``indices``
    omitted the row index is the cell coordinate itself, which is not affine and
    reflects to one parameter for the whole array: a coarser set, used where a
    footprint only has to be sound.

    ``reflections`` has to be the term's table whenever the result is going to
    be compared with something else isl built (an in-bounds obligation compares
    it with the range of an access map). Both sides must call ``cnt[r]`` by the
    same parameter, or the comparison is between two unrelated unknowns.
    """
    if names is None:
        names = tuple(f"a{k}" for k in range(len(arrtype.axes)))
    bounds: list[Any] = []
    for axis, (size, ragged) in enumerate(
        zip(arrtype.axes, arrtype.ragged, strict=True)
    ):
        if not ragged:
            bounds.append(size)
            continue
        counts = size.name if isinstance(size, prim.Variable) else size
        row: Any
        if indices is not None and axis >= 1:
            row = indices[axis - 1]
        else:
            row = prim.Variable(names[axis - 1]) if axis >= 1 else 0
        bounds.append(prim.Subscript(prim.Variable(str(counts)), row))
    return domain_set(
        tuple(names), bounds, names=tuple(names), reflections=reflections
    )


# {{{ the padded instance space


def instance_space_depth(term: Term) -> int:
    """How many loop dimensions the padded instance space needs."""
    return max((len(stmt.inames) for stmt in term.stmts), default=0)


def _instance_dims(depth: int) -> list[str]:
    """The dimension names of the padded instance space."""
    return ["s", *[f"d{k}" for k in range(depth)]]


def pad_map(n_inames: int, index: int, depth: int) -> isl.Map:
    """The map from a statement's own iteration space to the instance space.

    ``{ [r, j] -> [s, d0, d1, d2] : s = 1 and d0 = r and d1 = j and d2 = 0 }``:
    the statement index goes in front, the inames keep their order, and the
    unused dimensions are pinned so that the image is a set of points and not a
    slab.
    """
    source = ", ".join(f"i{k}" for k in range(n_inames))
    dims = _instance_dims(depth)
    constraints = [f"s = {index}"]
    for k in range(depth):
        constraints.append(f"d{k} = i{k}" if k < n_inames else f"d{k} = 0")
    target = ", ".join(dims)
    where = " and ".join(constraints)
    return isl.Map(f"{{ [{source}] -> [{target}] : {where} }}")


def _align(obj: Any, space: isl.Space) -> Any:
    """Give ``obj`` the parameters of ``space``, so that isl will combine them."""
    return obj.align_params(space.params())


def instance_domain(stmt: Stmt, index: int, depth: int) -> isl.Set:
    """A statement's instances as points of the padded instance space."""
    pad = _align(pad_map(len(stmt.inames), index, depth), stmt.domain.get_space())
    return stmt.domain.apply(pad)


# }}}


# {{{ footprints


@dataclass(frozen=True)
class Footprint:
    """What one statement does to one array, as a map instance -> cell.

    ``kind`` is ``"read"``, ``"write"`` or ``"acc"``. An accumulation both reads
    and writes its cell, and is recorded once as ``acc`` rather than twice, so
    that a rule about reassociation can find it.
    """

    stmt: str
    index: int
    array: str
    kind: str
    relation: isl.Map

    @property
    def touches_memory(self) -> bool:
        """Whether this footprint writes, hence can carry a dependence."""
        return self.kind in ("write", "acc")


def access_relation(
    inames: Sequence[str],
    domain: isl.Set,
    indices: Sequence[Any],
) -> isl.Map:
    """The map from the points of ``domain`` to the cells one access reaches.

    ``inames`` names the dimensions of ``domain``, which are the enclosing loop
    variables and, for an access inside a reduction, the reduction's own. When
    every index is quasi-affine the map is exact. When one is not (the archetype
    is the indirection ``x[col[r, j]]``) the access is widened to every cell,
    which is the sound direction for a dependence: more pairs are reported,
    never fewer, and an in-bounds obligation over such an access is left to a
    rule that can read the index's *type* instead.
    """
    names = tuple(inames)
    source = ", ".join(names)
    try:
        texts = [expr_text(index, None, None) for index in indices]
    except NonAffine:
        cells = isl.Set(f"{{ [{', '.join(f'a{k}' for k in range(len(indices)))}] }}")
        return isl.Map.from_domain_and_range(domain, _align(cells, domain.get_space()))
    free: set[str] = set()
    for text in texts:
        free |= free_names(text)
    params = sorted(free - set(names))
    head = f"[{', '.join(params)}] -> " if params else ""
    relation = isl.Map(f"{head}{{ [{source}] -> [{', '.join(texts)}] }}")
    return relation.intersect_domain(_align(domain, relation.get_space()))


#: One entry of :func:`statement_accesses`: array, indices, kind, inames, domain.
_Access = tuple[str, tuple[Any, ...], str, tuple[str, ...], isl.Set]


def statement_accesses(stmt: Stmt, term: Term) -> tuple[_Access, ...]:
    """Every cell family a statement touches, with the domain it touches it over.

    Each entry is ``(array, indices, kind, inames, domain)``, where ``inames``
    name the dimensions of ``domain``. An access written directly in the
    statement ranges over the statement's own domain; an access inside a
    reduction ranges over the reduction's domain, which carries the enclosing
    inames as its outer dimensions and the reduction's binders as its inner
    ones. Keeping the two apart is what makes ``val[r, j]`` an obligation about
    ``j`` in the row's count rather than about an unconstrained ``j``. The read
    of a ragged loop's bound ranges over the loop nest up to its row, the
    statement's first few inames, because it happens once per row (see
    :func:`layout_reads`).

    A statement evaluates three expressions, and all three are read: the
    right-hand side, the *subscripts of the assignee*, and the guard.

    * ``y[col[i + 1]] = v`` reads ``col[i + 1]`` to find the cell of ``y`` it
      writes. Recording only the write leaves that read with no in-bounds
      obligation, while the write itself is discharged *by type* from ``col``'s
      element sort, so a kernel reading past the end of ``col`` would be
      reported clean.
    * ``with when(flag[i] != 0)`` reads ``flag[i]`` to decide whether the write
      happens. Recording only the write loses the RAW dependence on an earlier
      statement that writes ``flag``, so a reordering could let the predicate
      observe the old value, and leaves the guard's own access unbounded.

    The guard's reads are stated over the statement's ``loop_domain``, the loop
    nest before the guard narrowed it, and not over ``domain``. A ``when``
    evaluates its whole condition at every point and only masks the write, so
    ``when((i + 1 < n) & (flag[i + 1] != 0))`` reads ``flag[n]`` at
    ``i = n - 1`` natively even though no write happens there; stating that
    read over the narrowed domain would prove it in bounds by the very
    condition that does not protect it.

    Two more reads belong to none of the three expressions: the layout's, which
    is why ``term`` is needed at all. A ragged ``val: Arr[Fin[n], Fin[cnt],
    Real]`` is stored flat, so ``val[r, j]`` is ``val[off[r] + j]`` once
    lowered, and a loop over ``val.dom[r]`` runs to ``cnt[r]``, which the
    lowered kernel reads once per row. See :func:`layout_reads` for when those
    reads are listed and why.

    This is the one collector: :mod:`loopty.typing` states its in-bounds
    obligations from it, :func:`footprints` builds the dependence relation from
    it, :func:`loopty.schedule._accesses` checks casts against it and
    :mod:`loopty.lower` derives an instruction's dependencies from it. An
    omission here is an omission everywhere, which is the point: it used to be
    possible for four collectors to disagree about what a statement reads, and
    the offsets read and the bound's read were, for a while, known to the
    lowering alone.
    """
    return tuple(_with_layout_reads(stmt, term))


def source_accesses(stmt: Stmt, term: Term) -> tuple[_Access, ...]:
    """The accesses of :func:`statement_accesses` that the source spells.

    Everything but the reads the layout of a ragged array adds (see
    :func:`layout_reads`). Rules ask :func:`statement_accesses`; this is for
    saying where an access came from.
    """
    from lanky.terms import structurally_equal

    from loopty.trace import accesses_in, reductions_in

    out: list[_Access] = []
    written = stmt.assignee
    kind = "acc" if stmt.kind == "accumulate" else "write"
    out.append((written.array, written.indices, kind, stmt.inames, stmt.domain))
    # The right-hand side first, so that the order the rules see is the order
    # the source reads in; then the assignee's own subscripts and the guard.
    guard_domain = stmt.loop_domain if stmt.loop_domain is not None else stmt.domain
    for source, domain in (
        (stmt.expr, stmt.domain),
        (tuple(written.indices), stmt.domain),
        (stmt.guard, guard_domain),
    ):
        if source is None:
            continue
        for access in accesses_in(source, into_reductions=False):
            # The right-hand side's read of the accumulated cell is already
            # covered by the "acc" footprint; a read of another cell of the same
            # array is not, and dropping it would lose a dependence. The guard's
            # read of that same cell is kept even so: it happens over the
            # un-narrowed loop domain, which the "acc" footprint does not cover.
            if (
                source is stmt.expr
                and stmt.kind == "accumulate"
                and access.array == written.array
                and structurally_equal(access.indices, written.indices)
            ):
                continue
            out.append((access.array, access.indices, "read", stmt.inames, domain))
        for reduction in reductions_in(source):
            inames = (*stmt.inames, *reduction.inames)
            for access in accesses_in(reduction.body, into_reductions=False):
                out.append(
                    (access.array, access.indices, "read", inames, reduction.domain)
                )
    return tuple(out)


# {{{ the reads a ragged layout makes


@dataclass(frozen=True)
class LayoutRead:
    """One read the layout of a ragged array makes, which the source never spells.

    ``read`` is the entry :func:`statement_accesses` lists for it. ``part`` says
    what the cell it reads is to row ``row``: ``"start"``, where the row starts
    in flat storage (``off[r]``), ``"end"``, where it ends (``off[r + 1]``),
    ``"length"``, how many entries it has (``cnt[r]``), or ``"row"``, a read
    the row expression itself makes (``p[i]`` in ``cnt[p[i]]``). A read a
    ragged access is flattened through names that access, as
    :func:`source_accesses` lists it, in ``access``; a read a loop's bound is
    computed from names the loops it bounds in ``loops``, by the names the term
    gives them.
    """

    read: _Access
    part: str
    row: Any
    access: _Access | None = None
    loops: tuple[str, ...] = ()


def layout_reads(stmt: Stmt, term: Term) -> tuple[LayoutRead, ...]:
    """Every read the layout of a ragged array makes in ``stmt``, and why.

    A ragged ``val: Arr[Fin[n], Fin[cnt], Real]`` is a flat buffer, and the
    lowered kernel reads two things to use it that the body never names:

    * **The start of a row**, for every access to ``val[r, j]``, read or
      written: the flat index is ``off[r] + j``, so the access reads
      ``off[r]`` (:func:`_index_reads`).
    * **The length of a row**, for every loop over ``val.dom[r]``, a ``for`` or
      a reduction's generator: the loop runs to a scalar lowering assigns once
      per row, ``cnt[r]`` when the counts are a parameter, which they always
      are in a traced kernel, and ``off[r + 1] - off[r]`` in a term built by
      hand whose counts are not. A loop over the fiber of any other row,
      ``val.dom[r - 1]`` or ``val.dom[p[i]]``, reads that row's length where
      it starts, and whatever the row expression reads (:func:`_bound_reads`).

    ``off[r + 1]`` is therefore listed only where a row's length is computed
    from the offsets. The flat index does not read it, and neither does
    anything else in a kernel whose counts are a parameter, so listing it there
    would order a write of the offsets against a read the code never makes.

    Listing these reads is what makes the layout's uses visible to every rule
    and not only to the lowering. A statement that writes ``off`` or ``cnt`` is
    ordered against one that indexes through them or loops over their rows, in
    the dependence relation and so in every cast's legality check; and each
    read is an in-bounds obligation of its own, which is where counts or
    offsets declared a cell short are caught. The native run and the term
    interpreter read the same arrays, the ones the kernel declares
    (:func:`loopty.term.declared_layout`), so a kernel that writes them means
    one thing however it runs.

    Nothing is listed for offsets the kernel does not declare
    (:func:`loopty.term.declared_offsets`). Lowering then adds them as an
    argument of ``n + 1`` cells that nothing in the body can name, so there is
    no writer to order against, and the row index being in bounds of ``val``'s
    first axis, which ``val[r, j]``'s own obligation states, keeps the reads in
    bounds. Nor is anything listed for a ragged axis that is not bounded by a
    counts name, which lowering refuses on its own.
    """
    out: list[LayoutRead] = []
    for access in source_accesses(stmt, term):
        for read in _index_reads(term, access):
            out.append(LayoutRead(read, "start", read[1][0], access=access))
    out.extend(_bound_reads(stmt, term))
    return tuple(out)


def _index_reads(term: Term, access: _Access) -> tuple[_Access, ...]:
    """The read of the offsets that one access makes through its flat index.

    For ``val[r, j]`` of a ragged ``val`` flattened through ``off``, that is
    ``off[r]``, a read over the domain of the access, whatever its kind: a
    write to ``val[r, j]`` goes through ``off[r]`` just as a read does. This is
    the read :func:`loopty.lower.lower_generic` indexes through.
    """
    array, indices, _kind, inames, domain = access
    typ = dict(term.params).get(array)
    if not isinstance(typ, ArrType):
        return ()
    axis = next((k for k, flag in enumerate(typ.ragged) if flag), None)
    if axis is None or axis == 0 or axis > len(indices):
        return ()
    counts = typ.axes[axis]
    if not isinstance(counts, prim.Variable):
        return ()
    offsets = declared_offsets(term.params, counts.name)
    if offsets is None:
        return ()
    return ((offsets, (indices[axis - 1],), "read", inames, domain),)


def counts_families(term: Term) -> tuple[str, ...]:
    """The counts name of every ragged parameter, in signature order.

    ``cnt`` for ``val: Arr[Fin[n], Fin[cnt], Real]``. A ragged axis whose bound
    is not a bare name has no counts family, and is skipped here; lowering
    refuses it, with the reason.
    """
    names: list[str] = []
    for _name, typ in term.params:
        if not isinstance(typ, ArrType):
            continue
        for size, ragged in zip(typ.axes, typ.ragged, strict=True):
            if ragged and isinstance(size, prim.Variable) and size.name not in names:
                names.append(size.name)
    return tuple(names)


def _counts_subscript(
    expr: Any, families: Collection[str], inames: Collection[str]
) -> tuple[str, str] | None:
    """``(counts, iname)`` if ``expr`` is ``cnt[r]`` for a counts array and iname."""
    if not isinstance(expr, prim.Subscript):
        return None
    if not isinstance(expr.aggregate, prim.Variable):
        return None
    if expr.aggregate.name not in families:
        return None
    index = expr.index
    if isinstance(index, tuple):
        if len(index) != 1:
            return None
        index = index[0]
    if not isinstance(index, prim.Variable) or index.name not in inames:
        return None
    return (expr.aggregate.name, index.name)


def ragged_bound_params(term: Term) -> dict[str, tuple[str, str]]:
    """Domain parameters standing for a ragged bound: name -> ``(counts, iname)``.

    ``nl_cnt_r`` stands for ``cnt[r]``, the length of row ``r`` of the arrays
    laid out over ``cnt``, with ``r`` a loop variable of a statement. Those are
    the bounds lowering can assign in a scalar temporary inside the loop over
    the row, which is why :func:`layout_reads` states their read once per row.

    Two sources, because a term reaches here two ways. A term written by hand
    spells the parameter, ``cnt_r`` or ``nl_cnt_r``, and is recognized by
    :func:`loopty.term.count_param_names`. A traced term records what it
    allocated on :attr:`loopty.term.Term.reflected`, and a parameter there is a
    ragged bound when the term it stands for is a counts array subscripted by
    a loop variable. The second source is what keeps a parameter that had to be
    suffixed (because the readable spelling was taken) recognizable as the row
    length it is, instead of being declared as a size argument nobody passes.
    """
    families = counts_families(term)
    inames = {iname for stmt in term.stmts for iname in stmt.inames}
    out: dict[str, tuple[str, str]] = {}
    for counts in families:
        for stmt in term.stmts:
            for iname in stmt.inames:
                for param in count_param_names(counts, iname):
                    out.setdefault(param, (counts, iname))
    for symbol, expr in term.reflected:
        pair = _counts_subscript(expr, families, inames)
        if pair is not None:
            out.setdefault(symbol, pair)
    return out


def bounds_dimension(domain: isl.Set, position: int, param: int) -> bool:
    """Does parameter ``param`` bound set dimension ``position`` of ``domain``?

    Read off the constraints rather than inferred by projecting: a constraint
    that mentions both the dimension and the parameter is one in which the
    parameter helps bound it, and nothing else counts. Projecting the other
    dimensions out instead would answer yes far too often, because eliminating
    a dimension leaves behind what its own existence implied. For the spmv
    reduction over ``[r, j]`` with ``0 <= j < nl_cnt_r``, projecting ``j`` out
    leaves ``nl_cnt_r >= 1``, which would make the *row* loop look as if its
    bound came from data.
    """
    found = False

    def visit_basic_set(basic_set: isl.BasicSet) -> None:
        nonlocal found
        for constraint in basic_set.get_constraints():
            if constraint.get_coefficient_val(isl.dim_type.set, position).is_zero():
                continue
            if not constraint.get_coefficient_val(
                isl.dim_type.param, param
            ).is_zero():
                found = True
                return

    domain.foreach_basic_set(visit_basic_set)
    return found


def loops_outside(domain: isl.Set, keep: int) -> isl.Set:
    """``domain`` over its first ``keep`` dimensions, whatever the loops inside do.

    Where something runs once per iteration of the outer loops, as the read of
    a ragged loop's bound does, it runs whether or not the inner loops have an
    iteration. So the constraints that mention an inner dimension are dropped
    before the inner dimensions are projected out: projecting ``j`` out of
    ``0 <= j < cnt_r`` alone would leave ``cnt_r >= 1`` behind, and a row with
    no entries reads its length all the same. The constraints on the outer
    dimensions and on the parameters alone are kept. When dropping leaves a
    dimension unbounded (a bound written through an inner loop, ``0 <= r <= j <
    n``, which no loop a person writes has) the plain projection is used, which
    bounds the same loops. :func:`loopty.lower._outer_part` is the lowering's
    variant, which also drops the constraints on the parameters alone.
    """
    total = domain.dim(isl.dim_type.set)
    if keep >= total:
        return domain
    inner = total - keep
    dropped = domain.drop_constraints_involving_dims(
        isl.dim_type.set, keep, inner
    ).project_out(isl.dim_type.set, keep, inner)
    if dropped.is_bounded():
        return dropped
    return domain.project_out(isl.dim_type.set, keep, inner)


def _bound_rows(term: Term) -> dict[str, tuple[str, Any]]:
    """Domain parameters that stand for the length of a row: name -> ``(counts, row)``.

    Every parameter :func:`ragged_bound_params` recognizes, with its loop
    variable as the row, and every parameter the term reflected for a counts
    array subscripted by any one index, ``cnt[r - 1]`` or ``cnt[p[i]]``: a
    bound lowering cannot assign, since no loop has that row as its variable,
    and one the native run reads all the same.
    """
    families = counts_families(term)
    out: dict[str, tuple[str, Any]] = {
        param: (counts, prim.Variable(iname))
        for param, (counts, iname) in ragged_bound_params(term).items()
    }
    for symbol, expr in term.reflected:
        if symbol in out or not isinstance(expr, prim.Subscript):
            continue
        if not isinstance(expr.aggregate, prim.Variable):
            continue
        if expr.aggregate.name not in families:
            continue
        index = expr.index
        if isinstance(index, tuple):
            if len(index) != 1:
                continue
            index = index[0]
        out[symbol] = (expr.aggregate.name, index)
    return out


def _bounded_domains(stmt: Stmt) -> list[tuple[isl.Set, tuple[str | None, ...], int]]:
    """The domains a ragged bound can occur in, with names and a first loop.

    Each is ``(domain, names, first)``, and the bounds of a domain are looked
    for from dimension ``first`` on. The statement's loop nest comes first,
    over its loop variables and before a guard narrowed it (``loop_domain``,
    since a guard masks writes and the loops run whatever it says), from ``0``.
    Then the domain of every reduction it evaluates, in the right-hand side,
    the assignee's subscripts or the guard, over the loop variables and then
    the reduction's binders, from the first binder, since a bound of a loop
    variable there is the statement's own again. A dimension whose name cannot
    be told is ``None``.
    """
    from loopty.trace import reductions_in

    loops = stmt.loop_domain if stmt.loop_domain is not None else stmt.domain
    out: list[tuple[isl.Set, tuple[str | None, ...], int]] = [
        (loops, tuple(stmt.inames), 0)
    ]
    for source in (stmt.expr, tuple(stmt.assignee.indices), stmt.guard):
        if source is None:
            continue
        for reduction in reductions_in(source):
            names: tuple[str | None, ...] = (*stmt.inames, *reduction.inames)
            n_dim = reduction.domain.dim(isl.dim_type.set)
            if len(names) != n_dim:
                lead = n_dim - len(reduction.inames)
                names = (*(None,) * max(lead, 0), *reduction.inames)[-n_dim:]
            out.append((reduction.domain, names, n_dim - len(reduction.inames)))
    return out


def _bound_reads(stmt: Stmt, term: Term) -> list[LayoutRead]:
    """The reads the bounds of the ragged loops of ``stmt`` make.

    For each parameter of the statement's domain, or of a reduction's, that
    stands for the length of a row (:func:`_bound_rows`): ``cnt[e]`` when the
    counts are a parameter, which they always are in a traced kernel, together
    with whatever ``e`` reads itself (``p[i]`` in ``cnt[p[i]]``), or ``off[e]``
    and ``off[e + 1]`` when only the offsets are.

    Where the read happens depends on who computes the bound. The length of
    row ``r`` of a loop variable ``r`` (:func:`ragged_bound_params`) is
    assigned by lowering once per row, in the loop over ``r`` and outside any
    loop inside it or any guard, whether or not the row has entries, so the
    read is stated over the loop nest up to ``r``: :func:`loops_outside` of
    the statement's ``loop_domain``. Any other bound is read where the loop it
    bounds starts, once per iteration of the loops around that one, which are
    the dimensions of the domain before the first one it bounds.
    """
    rows = _bound_rows(term)
    lowered = ragged_bound_params(term)
    params = dict(term.params)
    base = stmt.loop_domain if stmt.loop_domain is not None else stmt.domain
    per_row: dict[tuple[str, str], tuple[Any, list[str]]] = {}
    elsewhere: list[tuple[str, Any, tuple[str, ...], isl.Set, list[str]]] = []
    for domain, names, first in _bounded_domains(stmt):
        for param_position, param in enumerate(
            domain.get_var_names(isl.dim_type.param)
        ):
            row_of = rows.get(param)
            if row_of is None:
                continue
            bounded = [
                position
                for position in range(first, len(names))
                if bounds_dimension(domain, position, param_position)
            ]
            if not bounded:
                continue
            loops = [name for name in (names[k] for k in bounded) if name is not None]
            counts, row = row_of
            pair = lowered.get(param)
            if pair is not None and pair[1] in stmt.inames:
                _, seen = per_row.setdefault(pair, (row, []))
                seen.extend(name for name in loops if name not in seen)
                continue
            start = bounded[0]
            inames = tuple(name or f"i{k}" for k, name in enumerate(names[:start]))
            elsewhere.append(
                (counts, row, inames, loops_outside(domain, start), loops)
            )
    out: list[LayoutRead] = []
    for (counts, iname), (row, loops) in per_row.items():
        depth = stmt.inames.index(iname) + 1
        read_over = (tuple(stmt.inames[:depth]), loops_outside(base, depth))
        out.extend(_row_length_reads(term, params, counts, row, read_over, loops))
    for counts, row, inames, domain, loops in elsewhere:
        out.extend(
            _row_length_reads(term, params, counts, row, (inames, domain), loops)
        )
    return out


def _row_length_reads(
    term: Term,
    params: Mapping[str, Any],
    counts: str,
    row: Any,
    read_over: tuple[tuple[str, ...], isl.Set],
    loops: Sequence[str],
) -> list[LayoutRead]:
    """The reads computing the length of row ``row``, over ``read_over``."""
    from loopty.trace import accesses_in

    inames, domain = read_over
    bounded = tuple(loops)
    if isinstance(params.get(counts), ArrType):
        out = [
            LayoutRead(
                (counts, (row,), "read", inames, domain), "length", row, loops=bounded
            )
        ]
        for access in accesses_in(row):
            if isinstance(params.get(access.array), ArrType):
                read = (access.array, access.indices, "read", inames, domain)
                out.append(LayoutRead(read, "row", row, loops=bounded))
        return out
    offsets = declared_offsets(term.params, counts)
    if offsets is None:
        return []
    return [
        LayoutRead(
            (offsets, (index,), "read", inames, domain), part, row, loops=bounded
        )
        for part, index in (("start", row), ("end", row + 1))
    ]


def _with_layout_reads(stmt: Stmt, term: Term) -> list[_Access]:
    """The source's accesses, each ragged one followed by its offsets read,
    and then the reads the ragged loops' bounds make.

    A layout read already listed over the same domain is not listed twice:
    ``val[r, j]`` and ``col[r, j]`` in one reduction share their offsets, a
    statement may read ``off[r]`` or ``cnt[r]`` itself, and two reductions over
    ``val.dom[r]`` share their bound. Over a different domain it is kept,
    because that is a different set of instances reading it.
    """
    from lanky.terms import structurally_equal

    out: list[_Access] = []

    def add(read: _Access) -> None:
        if not any(
            seen[0] == read[0]
            and seen[2] == read[2]
            and seen[3] == read[3]
            and seen[4] is read[4]
            and structurally_equal(seen[1], read[1])
            for seen in out
        ):
            out.append(read)

    for access in source_accesses(stmt, term):
        out.append(access)
        for read in _index_reads(term, access):
            add(read)
    for layout in _bound_reads(stmt, term):
        add(layout.read)
    return out


# }}}


def _lift(n_inames: int, n_prefix: int) -> isl.Map:
    """``{ [i0, i1, i2] -> [i0, i1] }``: a statement's instance to a loop prefix."""
    source = ", ".join(f"i{k}" for k in range(n_inames))
    target = ", ".join(f"i{k}" for k in range(n_prefix))
    return isl.Map(f"{{ [{source}] -> [{target}] }}")


def footprints(term: Term) -> tuple[Footprint, ...]:
    """Every read, write and accumulation of every statement in ``term``.

    Reads are read off the right-hand side, which is where the tracer left them:
    an array reference in an expression is a pymbolic subscript, and the
    assignee is the statement's :class:`~loopty.term.Access`. Relations are in
    the padded instance space, so footprints of different statements compose;
    the dimensions a reduction adds are projected out, because a reduction is
    part of one statement instance and not a set of instances of its own.

    A read stated over fewer loops than the statement has, the bound of a
    ragged loop read once per row, is a read of every instance in that row:
    each of them runs after the bound is read, so a writer of the counts has
    to be ordered against all of them, which can only add pairs. The instances
    are the statement's, since the order the pairs are intersected with is
    defined on those alone.
    """
    depth = instance_space_depth(term)
    out: list[Footprint] = []
    for index, stmt in enumerate(term.stmts):
        pad = _align(pad_map(len(stmt.inames), index, depth), stmt.domain.get_space())
        into_instances = pad.reverse()
        for array, indices, kind, inames, domain in statement_accesses(stmt, term):
            relation = access_relation(inames, domain, indices)
            extra = len(inames) - len(stmt.inames)
            if extra > 0:
                relation = relation.project_out(
                    isl.dim_type.in_, len(stmt.inames), extra
                )
            elif extra < 0:
                lift = _lift(len(stmt.inames), len(inames))
                lift = _align(lift, relation.get_space())
                relation = lift.apply_range(_align(relation, lift.get_space()))
            padded = into_instances.apply_range(
                _align(relation, into_instances.get_space())
            )
            out.append(Footprint(stmt.id, index, array, kind, padded))
    return tuple(out)


# }}}


# {{{ order and dependences


def schedule_of(term: Term) -> isl.Map:
    """The source schedule: instance -> logical time, as a 2d+1 vector.

    ``[p0, d0, p1, d1, ..., pD]`` where ``p_k`` is the position of the block the
    statement sits in at depth ``k`` (its ``order`` path, recorded by the
    tracer) and ``d_k`` is the iname at that depth. Lexicographic order on these
    vectors is the order in which plain Python ran the body, which is the order
    every legality question is asked against.
    """
    depth = instance_space_depth(term)
    dims = _instance_dims(depth)
    union: isl.Map | None = None
    for index, stmt in enumerate(term.stmts):
        order = stmt.order or (0,) * (len(stmt.inames) + 1)
        time: list[str] = []
        for k in range(depth):
            time.append(str(order[k]) if k < len(order) else "0")
            time.append(f"d{k}" if k < len(stmt.inames) else "0")
        time.append(str(order[depth]) if depth < len(order) else "0")
        piece = isl.Map(
            f"{{ [{', '.join(dims)}] -> [{', '.join(time)}] }}"
        )
        instances = instance_domain(stmt, index, depth)
        piece = piece.intersect_domain(_align(instances, piece.get_space()))
        if union is None:
            union = piece
        else:
            union = union.union(_align(piece, union.get_space()))
    if union is None:
        dims = ", ".join(_instance_dims(0))
        nothing = isl.Set(f"{{ [{dims}] : 1 = 0 }}")
        return isl.Map(f"{{ [{dims}] -> [0] }}").intersect_domain(nothing)
    return union


def program_order(term: Term) -> isl.Map:
    """Pairs of instances that the source order runs one strictly before the other."""
    schedule = schedule_of(term)
    lex = isl.Map.lex_lt(schedule.get_space().range())
    return schedule.apply_range(_align(lex, schedule.get_space())).apply_range(
        schedule.reverse()
    )


def dependences(term: Term) -> isl.Map:
    """The dependence relation of ``term``, defined from its footprints.

    A pair of instances is a dependence when they touch the same cell of the
    same array, at least one of them writes it (RAW, WAR, WAW, and accumulation
    against anything), and the source order runs the first before the second.
    The relation is a plain map in the padded instance space, so the oracle can
    ask whether a proposed schedule is monotone on it without further surgery.

    "The same array" is by name. Two differently named parameters are taken to
    be disjoint storage, which is an assumption about how the kernel is called;
    :mod:`loopty.contract` enforces it at every executor and native entry point,
    and the module docstring says why it cannot be left implicit.
    """
    prints = footprints(term)
    before = program_order(term)
    deps: isl.Map | None = None
    for first in prints:
        for second in prints:
            if first.array != second.array:
                continue
            if not (first.touches_memory or second.touches_memory):
                continue
            shared = first.relation.apply_range(second.relation.reverse())
            pairs = shared.intersect(_align(before, shared.get_space()))
            if pairs.is_empty():
                continue
            if deps is None:
                deps = pairs
            else:
                deps = deps.union(_align(pairs, deps.get_space()))
    if deps is None:
        empty = before.copy()
        return empty.subtract(empty)
    return deps


# }}}
