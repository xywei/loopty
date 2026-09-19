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
from loopty.term import ArrType, Stmt, Term

__all__ = [
    "Footprint",
    "assume_sizes",
    "NonAffine",
    "cell_set",
    "dependences",
    "free_names",
    "domain_set",
    "expr_text",
    "footprints",
    "instance_domain",
    "instance_space_depth",
    "pad_map",
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


def statement_accesses(
    stmt: Stmt,
) -> tuple[tuple[str, tuple[Any, ...], str, tuple[str, ...], isl.Set], ...]:
    """Every cell family a statement touches, with the domain it touches it over.

    Each entry is ``(array, indices, kind, inames, domain)``. An access written
    directly in the statement ranges over the statement's own domain; an access
    inside a reduction ranges over the reduction's domain, which carries the
    enclosing inames as its outer dimensions and the reduction's binders as its
    inner ones. Keeping the two apart is what makes ``val[r, j]`` an obligation
    about ``j`` in the row's count rather than about an unconstrained ``j``.
    """
    from lanky.terms import structurally_equal

    from loopty.trace import accesses_in, reductions_in

    out: list[tuple[str, tuple[Any, ...], str, tuple[str, ...], isl.Set]] = []
    written = stmt.assignee
    kind = "acc" if stmt.kind == "accumulate" else "write"
    out.append((written.array, written.indices, kind, stmt.inames, stmt.domain))
    for access in accesses_in(stmt.expr, into_reductions=False):
        # The read of the accumulated cell is already covered by the "acc"
        # footprint; a read of another cell of the same array is not, and
        # dropping it would lose a dependence.
        if (
            stmt.kind == "accumulate"
            and access.array == written.array
            and structurally_equal(access.indices, written.indices)
        ):
            continue
        out.append((access.array, access.indices, "read", stmt.inames, stmt.domain))
    for reduction in reductions_in(stmt.expr):
        inames = (*stmt.inames, *reduction.inames)
        for access in accesses_in(reduction.body, into_reductions=False):
            out.append((access.array, access.indices, "read", inames, reduction.domain))
    return tuple(out)


def footprints(term: Term) -> tuple[Footprint, ...]:
    """Every read, write and accumulation of every statement in ``term``.

    Reads are read off the right-hand side, which is where the tracer left them:
    an array reference in an expression is a pymbolic subscript, and the
    assignee is the statement's :class:`~loopty.term.Access`. Relations are in
    the padded instance space, so footprints of different statements compose;
    the dimensions a reduction adds are projected out, because a reduction is
    part of one statement instance and not a set of instances of its own.
    """
    depth = instance_space_depth(term)
    out: list[Footprint] = []
    for index, stmt in enumerate(term.stmts):
        pad = _align(pad_map(len(stmt.inames), index, depth), stmt.domain.get_space())
        into_instances = pad.reverse()
        for array, indices, kind, inames, domain in statement_accesses(stmt):
            relation = access_relation(inames, domain, indices)
            extra = len(inames) - len(stmt.inames)
            if extra:
                relation = relation.project_out(
                    isl.dim_type.in_, len(stmt.inames), extra
                )
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
