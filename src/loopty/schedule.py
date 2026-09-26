"""Schedules: loop transformations as casts, checked one at a time.

A transformation is untrusted. ``.split``, ``.tile``, ``.interchange``,
``.skew``, ``.affine``, ``.tag`` and ``.realize`` apply the corresponding loopy
transform and then hand the result to a small checker, which asks isl two
questions: is the reindexing a bijection on statement instances, and is the new
execution order monotone on the dependence relation? Parallel inames (``g.*``,
``l.*``) carry no order, so they are dropped from the order before the second
question is asked. This is the de Bruijn criterion applied to scheduling: any
Python transformation is admissible because its output is checked, not its code.

Three things have to be written down for those two questions to be askable.

*One instance space.* A statement instance is encoded as ``[s, x0, ..., x_{W-1}]``
with ``s`` the statement's position in the term and ``x_j`` the value of its
``j``-th coordinate, padded with zeros. All statements then live in one isl space,
so the dependence relation is a single map and not a union over pairs of spaces,
and ``loopty.oracle``'s primitives apply directly.

*A reindexing map per transformation.* Splitting replaces a coordinate by two
related ones, skewing shifts one by another, and interchanging and tagging change
no coordinate at all: they change only the order. Each transformation states its
map as an isl map from the loops it replaces to the loops that replace them, and
isl decides whether it is a bijection on the instances that exist. ``.affine``
takes that map from the caller, so split, tile and skew are three ways of
writing particular affine maps, and the diamond ``(t, i) -> (t + i, t - i)`` is
a fourth that none of them can write.

*An order as a map into logical time.* The time of an instance is
``[c_0, i_0, c_1, i_1, ..., c_k]``: the loop values interleaved with constants
that place the statement among its siblings at each nesting level, which is the
standard way of making "the order the source is written in" a lexicographic
comparison. Dropping the parallel inames from that vector is what makes tagging
checkable: two instances that differ only in a parallel iname get the same time,
so a dependence between them is no longer ordered forward, and the cast is
rejected.

The order checked is also the order imposed: every accepted step sets loopy's
loop priority to the nest it just checked, so the generated code runs the nest
the checker approved rather than one loopy chose for itself. loopy takes a
priority as a preference, though, and drops one it cannot keep; a nest it
cannot keep is therefore refused as unbuildable (below), rather than left for
loopy to replace with a nest nobody checked.

A rejected cast raises :class:`IllegalCast` carrying ``witness``, the pair of
statement instances the transformation would reorder. That is the difference
between "tiling is illegal here" and "instance (0, 8) writes what instance
(1, 7) reads, and your tiling runs them the other way round". The refuted fact
the exception carries says the same thing: its ``reason`` is the exception's
message, which is what lanky prints under a ``REFUTED`` line.

Legal is not the same as buildable
----------------------------------

The two isl questions are about meaning, and meaning is all isl can see. A
transformation can preserve it and still be one the backend cannot generate
code for. loopy 2025.2 has such limits, two of which the design's own spmv
device schedule walks into: it will not put a hardware axis (``g.*``, ``l.*``)
inside a loop whose bound comes from an array, which is exactly what a CSR
inner loop is, and it will not generate a reduction whose inames are partly
parallel and partly sequential. The others known are about reductions too (a
hardware axis on one nested in another, a reduction on a group axis or across
two local axes, a local axis whose extent has no numeric maximum) and about
order (a loop put outside a loop loopy nests it inside, which loopy cannot
run in that order). None is a wrong verdict about the cast, and none used to
be reported: the casts were all ``DECIDED`` and loopy then threw during code
generation, several steps away from the line that caused it, or ran a nest of
its own choosing.

So every accepted step is asked a third question, this one about the target
rather than about meaning, and its answer is a fact of kind ``buildable``
decided by ``loopy-target``. A schedule that fails it still exists, still
carries its ``DECIDED`` cast facts, and still reports what it is: the refusal
happens when something asks for code (see :class:`UnbuildableSchedule`), which
is the moment the claim actually matters. The question is about the schedule
as it stands, so it is asked again after every step, and a later step can put
right what an earlier one broke: an interchange that puts a row loop back
outside its fiber makes a tiled ragged loop buildable again, and the schedule
then carries no ``buildable`` fact, as any buildable schedule does. A kernel
the rewrite of :meth:`Schedule.affine` could not write stays unwritten.

Fact ids
--------

A fact's id names the schedule it is about, precisely enough to tell it from
every other schedule of the kernel: the kernel, the target, and every step up
to the one the fact is about, with every argument it was given
(:attr:`Schedule.key`). Two schedules of one kernel in one file therefore keep
their facts apart in a ledger, which keeps one fact per id, while two that
share their first steps share the facts about those steps, which are the same
claims.

Maps whose image has holes
--------------------------

An affine map need not be unimodular. The diamond ``(t, i) -> (t + i, t - i)``
has determinant ``-2``: its image is only the points whose two coordinates have
the same parity, and ``t`` is ``(a + b) / 2`` there, not an integer affine
expression of the new loops. The checker does not care, because isl reasons
about that image exactly. loopy does: ``lp.map_domain`` and
``lp.affine_map_inames`` both solve for each old iname with a unit coefficient
and refuse this map. So the kernel is rewritten here instead, from the same isl
map (see :func:`_affine_kernel`): the new domain is the image isl computes, with
the parity as an existentially quantified constraint, and each old iname becomes
the quasi-affine expression isl gives for the inverse, ``floor((a + b)/2)``.
loopy 2025.2 generates correct code for that kernel, and for one tiled after
it. It enumerates the image's bounding loops and tests the parity with an
``if`` inside the innermost one rather than stepping by two, so half the
iterations of that loop do nothing. The answer is recorded in
``docs/loopy-notes.md``.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any

import islpy as isl
import loopy as lp
import pymbolic.primitives as prim
from pymbolic.mapper.substitutor import substitute

from loopty import idx
from loopty import oracle as isl_oracle
from loopty.flow import bounds_dimension
from loopty.lower import (
    Lowering,
    _plain,
    _reduction_nesting,
    is_reserved,
    lower_generic,
    reductions_of,
)
from loopty.term import Stmt, Term

__all__ = [
    "IllegalCast",
    "Schedule",
    "UnbuildableSchedule",
    "parallel_tag",
]

#: Iname tags that impose no order: two instances differing only in such an
#: iname may run in either order, or at the same time.
PARALLEL_TAG_PREFIXES = ("g.", "l.", "ilp")


def parallel_tag(tag: str) -> bool:
    """Does ``tag`` mark an iname as carrying no order?"""
    return any(tag.startswith(prefix) for prefix in PARALLEL_TAG_PREFIXES)


class IllegalCast(TypeError):
    """A transformation that would change the meaning of the program.

    ``witness`` is a ``(source_instance, sink_instance)`` pair from the isl
    oracle, and ``str()`` renders the explanation: which dependence, which two
    instances, and which way round the new order would run them. ``fact`` is the
    ``REFUTED`` ledger entry, carried on the exception because the schedule that
    would have held it was never built; its ``reason`` is this message, and its
    ``witness`` this witness when isl gave one.
    """

    def __init__(self, message: str, witness: Any = None, fact: Any = None) -> None:
        super().__init__(message)
        self.witness = witness
        self.fact = fact


class UnbuildableSchedule(TypeError):
    """A schedule the checker accepts and the backend cannot generate code for.

    This is not an :class:`IllegalCast`: nothing about the meaning of the
    program is wrong, and the cast facts stay ``DECIDED``. What is wrong is the
    combination of the schedule and the target, so the exception carries
    ``reason`` in the words of the limit it hits, and ``fact``, the ``REFUTED``
    ``buildable`` entry the schedule has been carrying since the step that
    caused it, whose ``reason`` is the same words.
    """

    def __init__(self, message: str, reason: str = "", fact: Any = None) -> None:
        super().__init__(message)
        self.reason = reason
        self.fact = fact


# {{{ the uniform instance space


@dataclass(frozen=True)
class _Layout:
    """Which coordinate each statement's instances are indexed by.

    ``coords[stmt_id]`` is the tuple of iname names that are the coordinates of
    that statement's instances, in a canonical order that only splitting changes.
    The loop *order* is kept separately, because interchanging loops renames no
    instance.
    """

    stmt_ids: tuple[str, ...]
    coords: dict[str, tuple[str, ...]]

    @property
    def width(self) -> int:
        """Number of coordinate dimensions, the deepest statement's nesting."""
        return max((len(c) for c in self.coords.values()), default=0)

    def dims(self, letter: str, suffix: str = "") -> str:
        """isl dimension names for this space, such as ``s, x0, x1``."""
        names = [f"s{suffix}"] + [f"{letter}{k}" for k in range(self.width)]
        return ", ".join(names)

    def index(self, stmt_id: str) -> int:
        """The position of a statement in the term."""
        return self.stmt_ids.index(stmt_id)


def _embed(domain: isl.Set, index: int, layout: _Layout) -> isl.Set:
    """Lift one statement's domain into the uniform instance space."""
    k = domain.dim(isl.dim_type.set)
    source = ", ".join(f"i{j}" for j in range(k))
    target = layout.dims("x")
    constraints = [f"s = {index}"]
    constraints += [f"x{j} = i{j}" for j in range(k)]
    constraints += [f"x{j} = 0" for j in range(k, layout.width)]
    lift = isl.Map(f"{{ [{source}] -> [{target}] : {' and '.join(constraints)} }}")
    return domain.apply(lift)


def _instances(term: Term, layout: _Layout, domains: dict[str, isl.Set]) -> isl.Set:
    """The set of all statement instances of a term, in one space."""
    out: isl.Set | None = None
    for stmt_id, domain in domains.items():
        lifted = _embed(domain, layout.index(stmt_id), layout)
        out = lifted if out is None else out.union(lifted)
    if out is None:  # pragma: no cover - a term always has a statement
        raise ValueError("a term with no statements has no instances")
    return out.coalesce()


def _coefficients(
    stmt_ids: Sequence[str], nests: dict[str, tuple[str, ...]]
) -> dict[str, tuple[int, ...]]:
    """Per-level sibling positions, the constants of the time vector.

    Two statements in the same loop interleave; two statements in different
    loops do not. The difference is recorded by walking the statements in
    program order and numbering, at each level, the distinct loop (or statement)
    that follows the prefix seen so far. That is the classical scattering
    construction, and it is what makes lexicographic comparison of time vectors
    reproduce the order of the source.
    """
    counters: dict[tuple, dict[tuple, int]] = {}
    out: dict[str, tuple[int, ...]] = {}
    for stmt_id in stmt_ids:
        prefix: tuple = ()
        coefficients: list[int] = []
        for iname in nests[stmt_id]:
            key = ("loop", iname)
            table = counters.setdefault(prefix, {})
            coefficients.append(table.setdefault(key, len(table)))
            prefix = (*prefix, key)
        table = counters.setdefault(prefix, {})
        coefficients.append(table.setdefault(("stmt", stmt_id), len(table)))
        out[stmt_id] = tuple(coefficients)
    return out


def _nests(
    layout: _Layout, order: Sequence[str], tags: dict[str, str]
) -> dict[str, tuple[str, ...]]:
    """Each statement's ordered loops, with the unordered ones left out."""
    return {
        stmt_id: tuple(
            iname
            for iname in order
            if iname in coords and not parallel_tag(tags.get(iname, ""))
        )
        for stmt_id, coords in layout.coords.items()
    }


def _time_map(
    layout: _Layout, order: Sequence[str], tags: dict[str, str]
) -> isl.Map:
    """The map from statement instances to logical time."""
    nests = _nests(layout, order, tags)
    coefficients = _coefficients(layout.stmt_ids, nests)
    depth = max((len(nest) for nest in nests.values()), default=0)
    length = 2 * depth + 1
    source = layout.dims("x")
    target = ", ".join(f"t{k}" for k in range(length))
    out: isl.Map | None = None
    for stmt_id in layout.stmt_ids:
        nest = nests[stmt_id]
        coords = layout.coords[stmt_id]
        constraints = [f"s = {layout.index(stmt_id)}"]
        used = 0
        for level, iname in enumerate(nest):
            constraints.append(f"t{2 * level} = {coefficients[stmt_id][level]}")
            constraints.append(f"t{2 * level + 1} = x{coords.index(iname)}")
            used = 2 * level + 2
        constraints.append(f"t{used} = {coefficients[stmt_id][len(nest)]}")
        constraints += [f"t{k} = 0" for k in range(used + 1, length)]
        piece = isl.Map(
            f"{{ [{source}] -> [{target}] : {' and '.join(constraints)} }}"
        )
        out = piece if out is None else out.union(piece)
    assert out is not None
    return out.coalesce()


def _step_map(
    layout_old: _Layout,
    layout_new: _Layout,
    mappings: Mapping[str, isl.Map] | None = None,
) -> isl.Map:
    """The reindexing map of one transformation.

    ``mappings`` gives each statement the transformation renames its map,
    from the loops it replaces to the loops that replace them (see
    :func:`_reindexing` and :meth:`Schedule._reindex_into`); a transformation
    that renames nothing gives none. Coordinates a statement's map does not
    touch are equated by name, which makes an interchange or a tag the
    identity and leaves every statement outside the mapped loops as it was;
    the mapped ones are lifted into the uniform space by :func:`_lift`.
    """
    mappings = mappings or {}
    source = layout_old.dims("x")
    target = layout_new.dims("y", suffix="_")
    out: isl.Map | None = None
    for stmt_id in layout_old.stmt_ids:
        old = layout_old.coords[stmt_id]
        new = layout_new.coords[stmt_id]
        index = layout_old.index(stmt_id)
        mapping = mappings.get(stmt_id)
        inputs = () if mapping is None else _dim_names(mapping, isl.dim_type.in_)
        pieces = [f"s = {index}", f"s_ = {index}"]
        for name in old:
            if name in inputs:
                continue
            pieces.append(f"y{new.index(name)} = x{old.index(name)}")
        pieces += [f"y{k} = 0" for k in range(len(new), layout_new.width)]
        pieces += [f"x{k} = 0" for k in range(len(old), layout_old.width)]
        piece = isl.Map(
            f"{{ [{source}] -> [{target}] : {' and '.join(pieces)} }}"
        )
        if mapping is not None:
            piece = piece.intersect(
                _lift(mapping, old, new, layout_old, layout_new)
            )
        out = piece if out is None else out.union(piece)
    assert out is not None
    return out.coalesce()


def _lift(
    mapping: isl.Map,
    old: Sequence[str],
    new: Sequence[str],
    layout_old: _Layout,
    layout_new: _Layout,
) -> isl.Map:
    """``mapping`` between one statement's coordinates, in the uniform space.

    The coordinates it reads are picked out of the old instance, it is applied
    to them, and what it produces is placed at the new instance's coordinates
    of those names; every other coordinate is left free, for
    :func:`_step_map` to equate. isl matches spaces by the number of
    dimensions and not by their names, so the caller's map is used as it
    stands, parameters included.
    """
    inputs = _dim_names(mapping, isl.dim_type.in_)
    outputs = _dim_names(mapping, isl.dim_type.out)
    source = layout_old.dims("x")
    target = layout_new.dims("y", suffix="_")
    picked = ", ".join(f"x{old.index(name)}" for name in inputs)
    placed = ", ".join(f"y{new.index(name)}" for name in outputs)
    # A name on both sides of an isl map is an equality, so these two select
    # and place coordinates without a constraint being written.
    select = isl.Map(f"{{ [{source}] -> [{picked}] }}")
    place = isl.Map(f"{{ [{placed}] -> [{target}] }}")
    return select.apply_range(mapping).apply_range(place)


def _dim_names(mapping: isl.Map, kind: Any) -> tuple[str, ...]:
    """The names of one tuple of ``mapping``; every dimension has to have one."""
    names = mapping.get_var_names(kind)
    if any(not name for name in names):
        side = "input" if kind == isl.dim_type.in_ else "output"
        raise ValueError(
            f"every {side} dimension of {mapping} has to be named, because the "
            "names are the loops it maps"
        )
    return tuple(names)


def _inherited(mapping: isl.Map) -> dict[str, set[str]]:
    """For each new loop, the old loops its value is a function of.

    Read off the map as isl writes it as a function: a tile's outer and inner
    halves of ``t`` involve ``t`` alone, a skewed ``i`` involves ``i`` and
    ``t``, and a diamond coordinate involves both. A map that is a function
    only on the instances, and not everywhere, is taken to make every new loop
    depend on every old one, which only over-approximates.
    """
    inputs = _dim_names(mapping, isl.dim_type.in_)
    outputs = _dim_names(mapping, isl.dim_type.out)
    try:
        function = isl.PwMultiAff.from_map(mapping)
    except isl.Error:
        return {name: set(inputs) for name in outputs}
    out: dict[str, set[str]] = {}
    for k, name in enumerate(outputs):
        value = function.get_pw_aff(k)
        out[name] = {
            old
            for position, old in enumerate(inputs)
            if value.involves_dims(isl.dim_type.in_, position, 1)
        }
    return out


def _part(mapping: isl.Map, inside: Sequence[str]) -> isl.Map | None:
    """The part of ``mapping`` over the loops ``inside``, or ``None``.

    A statement that runs in some of a map's loops and not the others can take
    the map only when the map is that part side by side with the rest: the
    new loops whose values depend on ``inside`` alone, as a map of ``inside``,
    next to every other new loop as a map of the other old ones. A tile is two
    splits side by side, so a statement in only one of its loops is split, as
    ``split_iname`` splits it in the kernel. A skew or a diamond mixes its
    loops and has no such part, and neither has a map isl cannot write as a
    function (see :func:`_inherited`).
    """
    inputs = _dim_names(mapping, isl.dim_type.in_)
    outputs = _dim_names(mapping, isl.dim_type.out)
    inherited = _inherited(mapping)
    mine = [name for name in outputs if inherited[name] <= set(inside)]
    if not mine:
        return None
    others = [name for name in inputs if name not in inside]
    theirs = [name for name in outputs if name not in mine]

    def restricted(ins: Sequence[str], outs: Sequence[str]) -> isl.Map:
        # The map with the other inputs existentially quantified and the other
        # outputs projected out, in the order the names are given.
        pick_in = _picking(len(inputs), [inputs.index(name) for name in ins])
        pick_out = _picking(len(outputs), [outputs.index(name) for name in outs])
        return pick_in.reverse().apply_range(mapping).apply_range(pick_out)

    part = restricted(inside, mine)
    whole = restricted((*inside, *others), (*mine, *theirs))
    if not whole.is_equal(part.flat_product(restricted(others, theirs))):
        return None
    for k, name in enumerate(inside):
        part = part.set_dim_name(isl.dim_type.in_, k, name)
    for k, name in enumerate(mine):
        part = part.set_dim_name(isl.dim_type.out, k, name)
    return part


def _reindexing(
    inputs: Sequence[str], outputs: Sequence[str], constraints: Sequence[str]
) -> isl.Map:
    """The map from the loops ``inputs`` to the loops ``outputs``.

    ``constraints`` name the inputs ``a0, a1, ...`` and the outputs ``b0, b1,
    ...``, and the real names are put on afterwards. That is what lets an
    output keep the name of the input it replaces (a skewed ``i`` is still
    ``i``), which the isl syntax would read as an equality, and it keeps a
    loop variable that happens to be spelled like an isl keyword out of the
    text.
    """
    source = ", ".join(f"a{k}" for k in range(len(inputs)))
    target = ", ".join(f"b{k}" for k in range(len(outputs)))
    out = isl.Map(f"{{ [{source}] -> [{target}] : {' and '.join(constraints)} }}")
    for k, name in enumerate(inputs):
        out = out.set_dim_name(isl.dim_type.in_, k, name)
    for k, name in enumerate(outputs):
        out = out.set_dim_name(isl.dim_type.out, k, name)
    return out


def _as_map(mapping: Any) -> isl.Map:
    """What ``affine`` was given, as an isl map: a map, a basic map, or text."""
    if isinstance(mapping, str):
        try:
            return isl.Map(mapping)
        except isl.Error as exc:
            raise ValueError(f"{mapping!r} is not an isl map: {exc}") from exc
    if isinstance(mapping, isl.BasicMap):
        return isl.Map.from_basic_map(mapping)
    if isinstance(mapping, isl.Map):
        return mapping
    if isinstance(mapping, isl.UnionMap):
        raise TypeError(
            "affine() takes one map for every statement in the loops it names, "
            "not a union of maps per statement: loopy gives the statements of a "
            "loop one domain, so they cannot be moved separately"
        )
    raise TypeError(
        f"affine() takes an isl map or its text, not {type(mapping).__name__}"
    )


# }}}


# {{{ dependences


@dataclass(frozen=True)
class _Dep:
    """One dependence: which instances, through which cell of which array."""

    kind: str  # "raw", "war", "waw"
    array: str
    source: str  # statement id
    sink: str
    source_indices: tuple[Any, ...]
    sink_indices: tuple[Any, ...]
    relation: isl.Map

    def verbs(self) -> tuple[str, str]:
        """How to say, in the rejection message, what each end did."""
        return {
            "raw": ("writes", "read"),
            "war": ("reads", "overwritten"),
            "waw": ("writes", "also written"),
        }[self.kind]


def _accesses(stmt: Stmt, term: Term) -> list[tuple[str, str, tuple[Any, ...]]]:
    """Every array reference of a statement, as ``(kind, array, indices)``.

    The list comes from :func:`loopty.flow.statement_accesses`, which is the one
    place the question "what does this statement touch?" is answered: the
    assignee, everything in the right-hand side including a reduction body, the
    reads inside the assignee's own subscripts, the reads inside the guard, and
    the reads of a ragged array's offsets that its flat index makes. A legality
    verdict is only as good as that list, and it used to be written out a
    second time here, which is how the guard came to be missing from it.

    The shared collector records an accumulation once, as ``acc``; the
    dependence computation below wants the write and the read separately, so
    that is the one thing unpacked here. The domains it reports are dropped: a
    schedule's coordinates are the layout's, not the term's, and an index that
    does not fit them is widened by :func:`_index_text`.
    """
    from loopty.flow import statement_accesses

    out: list[tuple[str, str, tuple[Any, ...]]] = []
    for array, indices, kind, _inames, _domain in statement_accesses(stmt, term):
        if kind == "acc":
            out.append(("write", array, tuple(indices)))
            out.append(("read", array, tuple(indices)))
        else:
            out.append((kind, array, tuple(indices)))
    return out


def _index_text(expr: Any, renaming: dict[str, Any], allowed: set[str]) -> str | None:
    """An index expression as isl text, or ``None`` when it has to be widened.

    A subscript inside a subscript (``x[col[r, j]]``), a reduction iname that is
    not a coordinate of the statement, or any other non-affine term has no isl
    form. Returning ``None`` means "this access reaches an unknown cell of that
    axis", and the caller then leaves the axis unconstrained, which widens the
    footprint and so can only add dependences, never drop one.
    """
    try:
        plain = substitute(_plain(expr), renaming)
    except Exception:  # pragma: no cover - defensive against exotic terms
        return None
    if not idx.is_affine(plain):
        return None
    free = set(idx.size_params([plain]))
    if not free <= allowed:
        return None
    try:
        return idx.isl_expr(plain)
    except Exception:  # pragma: no cover - is_affine already ruled this out
        return None


def _dependences(
    term: Term,
    layout: _Layout,
    instances: isl.Set,
    before: isl.Map,
    params: set[str],
) -> tuple[_Dep, ...]:
    """The dependence relation, defined from the term's footprints.

    Two instances depend on each other when they touch the same cell of the same
    array, at least one of them writes it, and the original order runs one before
    the other. Nothing is declared: this is the definition, evaluated by isl.
    """
    source_dims = layout.dims("x")
    target_dims = layout.dims("y", suffix="_")
    out: list[_Dep] = []
    for a in term.stmts:
        renaming_a = {
            iname: prim.Variable(f"x{k}")
            for k, iname in enumerate(layout.coords[a.id])
        }
        allowed_a = {f"x{k}" for k in range(len(layout.coords[a.id]))} | params
        for b in term.stmts:
            renaming_b = {
                iname: prim.Variable(f"y{k}")
                for k, iname in enumerate(layout.coords[b.id])
            }
            allowed_b = {f"y{k}" for k in range(len(layout.coords[b.id]))} | params
            for a_kind, a_array, a_indices in _accesses(a, term):
                for b_kind, b_array, b_indices in _accesses(b, term):
                    if a_array != b_array:
                        continue
                    if a_kind == "read" and b_kind == "read":
                        continue
                    if len(a_indices) != len(b_indices):
                        continue
                    constraints = [
                        f"s = {layout.index(a.id)}",
                        f"s_ = {layout.index(b.id)}",
                    ]
                    for a_index, b_index in zip(a_indices, b_indices, strict=True):
                        left = _index_text(a_index, renaming_a, allowed_a)
                        right = _index_text(b_index, renaming_b, allowed_b)
                        if left is None or right is None:
                            continue
                        constraints.append(f"{left} = {right}")
                    relation = isl.Map(
                        f"{{ [{source_dims}] -> [{target_dims}] : "
                        f"{' and '.join(constraints)} }}"
                    )
                    relation = (
                        relation.intersect_domain(instances)
                        .intersect_range(instances)
                        .intersect(before)
                    )
                    if relation.is_empty():
                        continue
                    kind = {
                        ("write", "read"): "raw",
                        ("read", "write"): "war",
                        ("write", "write"): "waw",
                    }[a_kind, b_kind]
                    out.append(
                        _Dep(
                            kind=kind,
                            array=a_array,
                            source=a.id,
                            sink=b.id,
                            source_indices=a_indices,
                            sink_indices=b_indices,
                            relation=relation.coalesce(),
                        )
                    )
    return tuple(out)


def _union(relations: Iterable[isl.Map]) -> isl.Map | None:
    """Union of dependence relations, or ``None`` when there are none."""
    out: isl.Map | None = None
    for relation in relations:
        out = relation if out is None else out.union(relation)
    return None if out is None else out.coalesce()


def _cross_check(term: Term, mine: isl.Map | None) -> tuple[isl.Map | None, str]:
    """Reconcile the dependences computed here with ``loopty.flow``'s.

    The definition of the dependence relation belongs in ``flow.py``, and this
    module computes its own only because it needs each dependence separately, to
    be able to say *which* array cell a rejected schedule would reorder; the
    union is what the verdict is actually about. The two are compared, and the
    schedule is checked against the union of both, so that a dependence either of
    them finds is one the schedule has to respect. A disagreement is recorded in
    the provenance of every cast fact rather than left for a reader to notice.
    """
    try:
        from loopty.flow import dependences
    except ImportError:  # pragma: no cover - flow.py is always present
        return mine, "loopty.flow is not importable"
    try:
        theirs = dependences(term)
    except NotImplementedError:
        return mine, "loopty.flow.dependences is not implemented yet"
    except Exception as exc:  # pragma: no cover - depends on the other wave
        return mine, f"loopty.flow.dependences raised {type(exc).__name__}: {exc}"
    if theirs is None:  # pragma: no cover - defensive
        return mine, "loopty.flow.dependences returned nothing"
    if mine is None:
        return (
            None if theirs.is_empty() else theirs,
            "no dependences here; loopty.flow found "
            + ("none either" if theirs.is_empty() else "some"),
        )
    if theirs.dim(isl.dim_type.in_) != mine.dim(isl.dim_type.in_):
        return mine, (
            "loopty.flow uses an instance space of "
            f"{theirs.dim(isl.dim_type.in_)} dimensions, this one uses "
            f"{mine.dim(isl.dim_type.in_)}; not compared"
        )
    if mine.is_equal(theirs):
        return mine, "agree with loopty.flow"
    return mine.union(theirs), "differ from loopty.flow; checked against the union"


# }}}


# {{{ what the target can build


def _data_dependent_in(
    domain: isl.Set, inames: Sequence[str], known: set[str]
) -> set[str]:
    """The inames of one domain whose bound comes from data.

    ``known`` names the parameters that do not: the sizes, and the loops and
    binders a domain may name as parameters because it is nested in them.
    """
    params = list(domain.get_var_names(isl.dim_type.param))
    positions = [k for k, name in enumerate(params) if name not in known]
    if not positions:
        return set()
    out: set[str] = set()
    for position in range(min(domain.dim(isl.dim_type.set), len(inames))):
        if any(bounds_dimension(domain, position, param) for param in positions):
            out.add(inames[position])
    return out


def data_dependent_inames(
    term: Term, reduction_inames: Mapping[str, Sequence[str]] | None = None
) -> frozenset[str]:
    """Loop variables whose extent is read out of an array.

    A ragged fiber is the case that matters: the bound of ``j`` in
    ``val.dom[r]`` is ``cnt[r]``, which the tracer reflects into an isl
    parameter that is not one of the term's sizes. Such a loop is a perfectly
    ordinary loop to isl and to C, and an impossible one to put on a hardware
    axis, because the number of work items is not known when the kernel is
    launched.

    ``reduction_inames`` is :attr:`loopty.lower.Lowering.reduction_inames`, so
    that a reduction the lowering gave fresh inames is reported under them.

    A nested reduction's domain names the binders of the reductions around it
    as parameters, and they are not data: ``j`` in
    ``reduce_sum(reduce_sum(a[i, j] for j in Fin[i + 1]) for i in a.dom)`` is
    bounded by the outer binder ``i``, a triangle and not a ragged fiber. Nor
    is a loop variable of the statement. Only a parameter that is none of
    these, and not a size, is read out of an array.
    """
    sizes = set(term.sizes)
    renamed = reduction_inames or {}
    out: set[str] = set()
    for stmt in term.stmts:
        out |= _data_dependent_in(stmt.domain, stmt.inames, sizes)
        nesting = _reduction_nesting(stmt.expr)
        for position, (reduction, enclosing) in enumerate(nesting):
            names = renamed.get(f"{stmt.id}:{position}", reduction.inames)
            loops = {
                binder for k in enclosing for binder in nesting[k][0].inames
            }
            out |= _data_dependent_in(
                reduction.domain,
                (*stmt.inames, *names),
                sizes | set(stmt.inames) | loops,
            )
    return frozenset(out)


def _unbuildable_reason(draft: _Draft) -> str | None:
    """Why loopy could not generate code for this draft, or ``None``.

    Limits of loopy 2025.2, all measured rather than guessed, in the order
    they are asked:

    * a parallel tag inside a ragged fiber: a device run of the design's spmv
      schedule fails on it;
    * a hardware axis on a reduction nested in another: code generation for a
      double sum with its inner reduction on a local axis fails ("instruction
      ... does not use all local hw axes");
    * the three ways loopy refuses to realize a reduction, asked as it asks
      them (see :func:`_reduction_reason`): partly on a local axis and partly
      in sequence (applying loopy's own ``split_reduction_outward`` remedy to
      the spmv schedule fails on this one), across two local axes, and on any
      other concurrent axis, a group axis above all;
    * a local axis whose extent has no numeric maximum, where a reduction on
      a local axis needs one (see :func:`_extent_reason`);
    * a loop ordered outside a loop loopy nests it inside, which loopy cannot
      run in that order (see :func:`_nest_reason`).

    Note 11 of ``docs/loopy-notes.md`` has the table of what loopy says to
    each, and note 6 the loop orders.
    """
    parallel = {name for name, tag in draft.tags.items() if parallel_tag(tag)}
    inside = sorted(parallel & draft.data_dependent)
    if inside:
        names = ", ".join(inside)
        return (
            f"the parallel tag on {names} sits inside a loop whose bound comes "
            "from an array (a ragged fiber), and loopy will not put a hardware "
            "axis in a domain with a data-dependent parameter. Parallelize an "
            "enclosing loop with a size known at launch instead, such as the "
            "rows of a CSR product"
        )
    nested = sorted(
        name
        for name in parallel
        if draft.reductions.get(name) in draft.nested_in
    )
    if nested:
        names = ", ".join(nested)
        outer = draft.nested_in[draft.reductions[nested[0]]]
        around = ", ".join(
            sorted(name for name, key in draft.reductions.items() if key == outer)
        )
        return (
            f"the parallel tag on {names} puts a hardware axis on a reduction "
            f"nested in the reduction over {around}, and loopy cannot generate "
            "code for it: the enclosing reduction's accumulator is set and "
            f"updated outside the loop over {names}, by instructions that do "
            "not run on its axis. Put the axis on the enclosing reduction, or "
            "on a loop of the statement, instead"
        )
    by_reduction: dict[str, list[str]] = {}
    for iname, key in draft.reductions.items():
        by_reduction.setdefault(key, []).append(iname)
    for key, inames in sorted(by_reduction.items()):
        reason = _reduction_reason(draft, key, inames)
        if reason is not None:
            return reason
    for key, inames in sorted(by_reduction.items()):
        reason = _extent_reason(draft, key, inames)
        if reason is not None:
            return reason
    return _nest_reason(draft)


def _reduction_role(tag: str | None) -> str:
    """How loopy realizes a reduction over a loop with this tag.

    ``"local"``, ``"sequential"``, ``"unrolled"``, ``"ilp"`` or ``"refused"``,
    as ``loopy.transform.realize_reduction`` classifies a reduction's inames,
    and read off loopy's own tag classes so that the two cannot drift apart:
    an untagged loop is summed in sequence, and so is one loopy unrolls
    (``unr``, and ``ilp``, whose accumulator loopy also privatizes, which is
    why it is told apart); a local axis (``l.*``) is summed in a tree across
    the work items of a group; and a reduction over any other concurrent axis
    (a group axis ``g.*``, ``ilp.seq``, ``vec``) is not generated at all. The
    checker reads ``ilp`` differently, as an order-free loop
    (:data:`PARALLEL_TAG_PREFIXES`), which is what makes it ask an
    accumulation's permission to be reassociated before a reduction loop is
    tagged so.
    """
    from loopy.kernel.data import (
        ConcurrentTag,
        LocalInameTagBase,
        UnrolledIlpTag,
        UnrollTag,
        parse_tag,
    )

    parsed = parse_tag(tag)
    if isinstance(parsed, UnrolledIlpTag):
        return "ilp"
    if isinstance(parsed, UnrollTag):
        return "unrolled"
    if isinstance(parsed, LocalInameTagBase):
        return "local"
    if isinstance(parsed, ConcurrentTag):
        return "refused"
    return "sequential"


def _reduction_reason(draft: _Draft, key: str, inames: Sequence[str]) -> str | None:
    """Why loopy would refuse to realize one reduction, or ``None``.

    Asked in the order loopy asks (``map_reduction`` in
    ``loopy.transform.realize_reduction``), so that the reason is the error
    loopy would raise: part of it on a local axis and part in sequence, then
    two local axes, then a concurrent axis that is not a local one. A
    reduction is generated only when every loop of it is sequential, or when
    it is one loop, on a local axis.

    One more is asked after those, because loopy meets it later, when it
    privatizes the temporaries of an ``ilp`` loop: a reduction over an
    ``ilp`` loop has its accumulator privatized along that loop, and whether
    loopy then accepts the instruction that initializes it outside the loop
    changes from one run to the next with Python's string hash seed (the same
    kernel was refused under some seeds and built under others, in 2025.2). A
    fact that holds only on some runs is not a fact, so it is refused; ``unr``
    unrolls the sum in order and builds.
    """
    accumulated = draft.reduction_info.get(key, (key, ""))[0]
    roles = {name: _reduction_role(draft.tags.get(name)) for name in inames}
    local = sorted(name for name, role in roles.items() if role == "local")
    sequential = sorted(
        name
        for name, role in roles.items()
        if role in ("sequential", "unrolled", "ilp")
    )
    refused = sorted(name for name, role in roles.items() if role == "refused")
    privatized = sorted(name for name, role in roles.items() if role == "ilp")
    if local and sequential:
        unrolled = [n for n in sequential if roles[n] in ("unrolled", "ilp")]
        how = (
            f" (loopy unrolls {', '.join(unrolled)}, which is a sequence)"
            if unrolled
            else ""
        )
        return (
            f"the reduction into {accumulated} runs over {', '.join(local)} "
            f"in parallel and {', '.join(sequential)} in sequence{how}, and "
            "loopy generates code only for a reduction whose inames are all one "
            "or all the other. loopty has no split_reduction transform to offer "
            "as the remedy"
        )
    if len(local) > 1:
        return (
            f"the reduction into {accumulated} runs over {', '.join(local)} on "
            f"{len(local)} local axes, and loopy sums a reduction across one "
            "local axis at most, and only when that axis is the whole of it. "
            "loopty has no split_reduction transform to offer as the remedy"
        )
    if refused:
        tags = ", ".join(f"{name}={draft.tags[name]!r}" for name in refused)
        return (
            f"the reduction into {accumulated} runs over {tags}, and loopy "
            "runs a reduction in parallel only across the work items of a "
            "group, on a local axis (l.*): a group axis, ilp.seq or vec on the "
            "loop of a reduction is refused. Put it on a local axis instead, or "
            "put the axis on a loop of the statement"
        )
    if privatized:
        return (
            f"the reduction into {accumulated} runs over {', '.join(privatized)} "
            "on an ilp axis, and loopy privatizes the accumulator along it and "
            "then accepts or refuses the instruction that initializes it outside "
            "the loop depending on Python's string hash seed, so the code cannot "
            "be counted on. Tag it unr instead, which unrolls the sum in order"
        )
    return None


def _extent_reason(draft: _Draft, key: str, inames: Sequence[str]) -> str | None:
    """A reduction on a local axis whose partial sums loopy cannot size.

    loopy sums a reduction on a local axis as a tree, over an array in local
    memory with one cell per work item of every local axis the statement runs
    on: the reduction's own, and those of the statement's loops around it
    (``map_reduction_local`` in ``loopy.transform.realize_reduction``). The
    shape of that array has to be a number when the code is generated, so
    each of those extents needs a numeric maximum over every value of the
    sizes. loopy looks for it with ``static_max_of_pw_aff(...,
    constants_only=True)`` on the loop's bounds, and this asks it the same way:
    ``y[i] = reduce_sum(a[i, j] for j in Fin[i + 1])`` with ``j`` on ``l.0``
    has none while ``n`` is free, and has 8 in a kernel over ``Fin[8]``.
    """
    if draft.kernel is None:
        return None
    local = [n for n in inames if _reduction_role(draft.tags.get(n)) == "local"]
    if not local:
        return None
    statement = key.rsplit(":", 1)[0]
    around = [
        name
        for name in draft.coords.get(statement, ())
        if _reduction_role(draft.tags.get(name)) == "local"
    ]
    accumulated = draft.reduction_info.get(key, (key, ""))[0]
    for name in (*local, *around):
        extent = _unbounded_extent(draft.kernel, name)
        if extent is None:
            continue
        whose = (
            f"the reduction into {accumulated} runs over {name} on a local axis"
            if name in local
            else f"the reduction into {accumulated} runs on a local axis inside "
            f"the loop {name}, which is on a local axis too"
        )
        return (
            f"{whose}, and loopy sums such a reduction in local memory, in an "
            "array with a cell per work item whose shape has to be a number "
            f"when the code is generated; the extent of {name} is {extent}, "
            "which no number bounds while the sizes it names are free. Declare "
            "that extent as a number (Fin[8] rather than Fin[n]), or run the "
            "reduction in sequence"
        )
    return None


def _unbounded_extent(kernel: Any, iname: str) -> str | None:
    """The extent of ``iname`` when it has no numeric maximum, or ``None``.

    Asked as ``loopy.transform.realize_reduction`` asks it, of the size in the
    loop's bounds: the extent over every value of the sizes and of the loops
    around it.
    """
    from loopy.diagnostic import StaticValueFindingError
    from loopy.isl_helpers import static_max_of_pw_aff
    from loopy.symbolic import pw_aff_to_expr

    size = kernel.default_entrypoint.get_iname_bounds(iname).size
    try:
        static_max_of_pw_aff(size, constants_only=True)
    except StaticValueFindingError:
        try:
            return f"at most {pw_aff_to_expr(size)}"
        except Exception:  # noqa: BLE001 - a piecewise extent is shown as isl has it
            return f"at most {size}"
    return None


def _enclosing_loops(kernel: Any) -> dict[str, dict[str, str]]:
    """For each loop of ``kernel``, the loops loopy nests it inside, and why.

    Read off loopy's own ``find_loop_nest_around_map``, the nesting both of
    its schedulers keep to, so that the two cannot drift apart. loopy nests a
    loop inside another for one of two reasons. A domain that names a loop as
    a parameter is defined inside that loop, with every loop it defines: a
    ragged fiber's domain names its row, and so does the inner domain of a
    nest another statement leaves (note 10 of ``docs/loopy-notes.md``). And a
    loop whose instructions are some of another loop's, and not all of them,
    runs inside it, whatever the domains say: ``k`` in ``w[r + 1, k] = w[r, k]
    + reduce_sum(val[r, j] for j in val.dom[r])`` shares one domain with ``r``
    and is still nested in it, because the length of row ``r`` is assigned in
    ``r`` and outside ``k``. The loops a loop is nested in are these, and the
    ones those are nested in, in turn, each with the reason in words.
    """
    from loopy.schedule import find_loop_nest_around_map

    entry = kernel.default_entrypoint
    direct = find_loop_nest_around_map(entry)
    named: dict[str, set[str]] = {}
    for domain in entry.domains:
        params = set(domain.get_var_names(isl.dim_type.param))
        for name in domain.get_var_names(isl.dim_type.set):
            named.setdefault(name, set()).update(params)
    insns = entry.iname_to_insns()

    def why(inner: str, outer: str) -> str:
        if outer in named.get(inner, ()):
            return (
                f"the domain of {inner} names {outer}, as a ragged fiber's "
                "names its row"
            )
        others = ", ".join(sorted(insns[outer] - insns[inner]))
        return f"{outer} runs every instruction {inner} runs, and {others} as well"

    out: dict[str, dict[str, str]] = {}
    for name in sorted(direct):
        reached: dict[str, str] = {}
        pending = [(outer, why(name, outer)) for outer in sorted(direct[name])]
        while pending:
            current, reason = pending.pop(0)
            if current in reached:
                continue
            reached[current] = reason
            pending.extend(
                (outer, f"{reason}, and {current} is nested in {outer} in turn")
                for outer in sorted(direct.get(current, ()))
            )
        out[name] = reached
    return out


def _nest_reason(draft: _Draft) -> str | None:
    """A loop ordered outside a loop loopy nests it inside, or ``None``.

    loopy keeps a nesting of its own (see :func:`_enclosing_loops`): a ragged
    fiber inside its row, because the row's length is read there, the loops
    of a statement that another statement leaves inside the loops around it
    (note 10 of ``docs/loopy-notes.md``), and a statement loop inside the row
    of a ragged reduction in it, because the row's length is assigned outside
    the statement loop. The loop priority :func:`_with_priority` sets is only
    a preference, and loopy drops it when the two disagree ("Cannot satisfy
    constraint that iname ... must be nested within ...") and runs a nest of
    its own choosing, which the cast facts did not check and which can run a
    dependence backwards. A tile of a ragged fiber with its row orders
    ``j_outer`` before ``r_inner``, and an interchange can put the fiber
    before the row outright.

    Read off the kernel after the step and asked of each statement's ordered
    loops, which is the nest the priority states; a parallel loop has no
    place in it. Checking loopy's own linearized nest against the order would
    be the complete answer, and would cost a scheduling pass per step.
    """
    if draft.kernel is None:
        return None
    around = _enclosing_loops(draft.kernel)
    layout = _Layout(stmt_ids=tuple(draft.coords), coords=dict(draft.coords))
    for nest in _nests(layout, draft.order, draft.tags).values():
        for position, loop in enumerate(nest):
            for inner in nest[position + 1 :]:
                why = around.get(loop, {}).get(inner)
                if why is None:
                    continue
                return (
                    f"the loop {loop} is ordered outside {inner}, but loopy "
                    f"nests it inside {inner}: {why}. So loopy cannot run the "
                    "order that was checked: it would drop the loop priority "
                    "and run a nest of its own choosing, which the cast facts "
                    f"say nothing about. Order {inner} outside {loop}, as "
                    f"interchange({inner!r}, {loop!r}) does"
                )
    return None


# }}}


def _term_of(obj: Any) -> Term:
    """The term of a kernel, a schedule, or a term."""
    if isinstance(obj, Term):
        return obj
    term = getattr(obj, "term", None)
    if term is None and hasattr(obj, "trace"):
        term = obj.trace()
    if isinstance(term, Term):
        return term
    raise TypeError(f"{obj!r} is not a kernel and has no term to schedule")


def _with_priority(kernel: Any, orders: Iterable[Sequence[str]]) -> Any:
    """Set loopy's loop priority to exactly these nests.

    Replacing rather than adding is the point: ``lp.prioritize_loops``
    accumulates, and after an interchange the old priority contradicts the new
    one, which loopy can only report as an unschedulable kernel.
    """
    entry = kernel.default_entrypoint
    priorities = frozenset(tuple(order) for order in orders if len(order) > 1)
    return kernel.with_kernel(entry.copy(loop_priority=priorities))


@dataclass
class _Draft:
    """The state one transformation builds, before the checker sees it."""

    coords: dict[str, tuple[str, ...]]
    order: list[str]
    tags: dict[str, str]
    #: The loopy kernel, or ``None`` once a step could not be written as one;
    #: see :attr:`Schedule.kernel`.
    kernel: Any
    #: Reduction iname -> the key of the reduction it belongs to. A term may
    #: have two reductions writing the same array with different exactness, so
    #: an iname has to name *which* one, not merely what it accumulates into.
    reductions: dict[str, str] = field(default_factory=dict)
    #: Reduction key -> ``(accumulated array, exactness class)``.
    reduction_info: dict[str, tuple[str, str]] = field(default_factory=dict)
    #: The step's reindexing of each statement it renames, from the loops it
    #: replaces to the loops that replace them: the step's own map, or the part
    #: of it over a statement's loops; see :meth:`Schedule._reindex_into`.
    mappings: dict[str, isl.Map] = field(default_factory=dict)
    reassoc: set[str] = field(default_factory=set)
    #: Loop variables whose extent comes from an array; see
    #: :func:`data_dependent_inames`. A split passes the property to both halves.
    data_dependent: set[str] = field(default_factory=set)
    #: Why the step could not be written as a loopy kernel, when it could not.
    unbuildable: str | None = None
    #: Reduction key -> the key of the reduction it is nested in, for every
    #: reduction that is nested in another; see :func:`_unbuildable_reason`.
    nested_in: dict[str, str] = field(default_factory=dict)


def _transformed(kernel: Any, transform: Any, *args: Any, **kwargs: Any) -> Any:
    """``transform(kernel, ...)``, or ``None`` when there is no kernel left.

    A schedule whose kernel could not be rewritten (see :func:`_affine_kernel`)
    keeps being checked, so that its casts and its ``buildable`` fact say what
    it is, but there is nothing for a later loopy transform to act on.
    """
    return None if kernel is None else transform(kernel, *args, **kwargs)


class Schedule:
    """A kernel plus the transformations applied to it, each one checked.

    ``Schedule(kernel, target="c")`` starts from the identity schedule. Every
    method returns a new schedule, so a rejected step leaves the previous one
    intact. ``.facts()`` yields the cast facts, ``DECIDED`` for the steps isl
    accepted and ``REFUTED`` with a witness for one it did not.

    ``sizes`` is a hint, not a constraint: the checks are made with the size
    parameters free, and the hint is used only to print a witness with concrete
    numbers in it, and as the default example inputs for ``loopty run``.
    """

    def __init__(
        self,
        kernel: Any,
        target: str = "c",
        sizes: dict[str, int] | None = None,
    ) -> None:
        self._source = kernel
        self._term = _term_of(kernel)
        self._target = target
        self._sizes = dict(sizes or {})
        self._lowering: Lowering = lower_generic(self._term, target)
        self._kernel = self._lowering.kernel

        stmt_ids = tuple(stmt.id for stmt in self._term.stmts)
        self._layout = _Layout(
            stmt_ids=stmt_ids,
            coords={stmt.id: tuple(stmt.inames) for stmt in self._term.stmts},
        )
        self._domains = {stmt.id: _set_over(stmt) for stmt in self._term.stmts}
        self._instances = _instances(self._term, self._layout, self._domains)
        self._origin = self._instances
        self._origin_layout = self._layout

        order: list[str] = []
        for stmt in self._term.stmts:
            for iname in stmt.inames:
                if iname not in order:
                    order.append(iname)
        self._order = order
        self._tags: dict[str, str] = {}
        # A reduction iname is not a coordinate of the instance space: a whole
        # reduction happens inside one statement instance. Splitting or tagging
        # one therefore renames no instance and reorders no dependence; what it
        # changes is the order of the accumulation, which is a question about
        # exactness rather than about dependences. A reduction is named by the
        # inames it has in the kernel, which are its binders unless the
        # lowering had to give it fresh ones (see Lowering.reduction_inames),
        # so that a step names the loop it acts on.
        self._reductions: dict[str, str] = {}
        self._reduction_info: dict[str, tuple[str, str]] = {}
        #: Which reduction each nested one sits in, by key; nesting is a fact
        #: about the term, so no step changes it.
        self._nested_in: dict[str, str] = {}
        reduction_inames = self._lowering.reduction_inames
        for stmt in self._term.stmts:
            nesting = _reduction_nesting(stmt.expr)
            for position, (reduction, enclosing) in enumerate(nesting):
                key = f"{stmt.id}:{position}"
                self._reduction_info[key] = (
                    stmt.assignee.array,
                    reduction.exactness,
                )
                if enclosing:
                    self._nested_in[key] = f"{stmt.id}:{enclosing[-1]}"
                for iname in reduction_inames.get(key, reduction.inames):
                    self._reductions[iname] = key
        self._reassoc: frozenset[str] = frozenset()
        self._data_dependent = data_dependent_inames(self._term, reduction_inames)
        self._history: tuple[str, ...] = ()
        #: Each step as ``(method, args, kwargs)``, so that :meth:`retarget` can
        #: replay it against another target and have every cast checked again.
        self._steps: tuple[tuple[str, tuple, dict], ...] = ()
        self._facts: tuple[Any, ...] = ()
        self._unbuildable: str | None = None
        self._examples: dict[str, Any] | None = None

        params = set()
        for domain in self._domains.values():
            params |= set(domain.get_var_names(isl.dim_type.param))
        self._params = params

        time = _time_map(self._layout, self._order, {})
        lex = isl.Map.lex_lt(time.get_space().range())
        before = (
            time.apply_range(lex)
            .apply_range(time.reverse())
            .intersect_domain(self._instances)
            .intersect_range(self._instances)
        )
        self._deps = _dependences(
            self._term, self._layout, self._instances, before, params
        )
        self._deps_total, self._flow_note = _cross_check(
            self._term, _union(dep.relation for dep in self._deps)
        )
        self._reindex = isl.Map.identity(
            self._instances.get_space().map_from_set()
        ).intersect_domain(self._instances)

    # {{{ plumbing

    def _clone(self) -> Schedule:
        """A shallow copy; every public method builds one rather than mutating."""
        other = object.__new__(Schedule)
        other.__dict__.update(self.__dict__)
        return other

    def _draft(self) -> _Draft:
        return _Draft(
            coords=dict(self._layout.coords),
            order=list(self._order),
            tags=dict(self._tags),
            kernel=self._kernel,
            reductions=dict(self._reductions),
            reduction_info=dict(self._reduction_info),
            reassoc=set(self._reassoc),
            data_dependent=set(self._data_dependent),
            nested_in=dict(self._nested_in),
        )

    @property
    def term(self) -> Term:
        """The term being scheduled."""
        return self._term

    @property
    def source(self) -> Any:
        """The object the schedule was built from (a kernel, or a term)."""
        return self._source

    @property
    def target(self) -> str:
        """The name of the loopy target, ``"c"`` or ``"opencl"``."""
        return self._target

    @property
    def sizes(self) -> dict[str, int]:
        """The size hint given at construction."""
        return dict(self._sizes)

    @property
    def kernel(self) -> Any:
        """The loopy kernel as transformed so far.

        ``None`` once a step could not be written as a loopy kernel at all,
        which only :meth:`affine` can cause; the schedule's ``buildable`` fact
        says why, and :meth:`require_buildable` refuses before anything reads
        this.
        """
        return self._kernel

    @property
    def lowering(self) -> Lowering:
        """The lowering the schedule started from."""
        return self._lowering

    @property
    def history(self) -> tuple[str, ...]:
        """The transformations applied, as they would be written in Python."""
        return self._history

    @property
    def order(self) -> tuple[str, ...]:
        """The loop nest, outermost first."""
        return tuple(self._order)

    @property
    def tags(self) -> dict[str, str]:
        """Iname tags applied so far."""
        return dict(self._tags)

    @property
    def reassociated(self) -> frozenset[str]:
        """Arrays whose accumulation has been marked reassociated."""
        return self._reassoc

    @property
    def buildable(self) -> tuple[bool, str]:
        """Can this target generate code for this schedule, and if not, why not?

        A pair, ``(ok, reason)``, with ``reason`` empty when it is buildable.
        The question is asked of every accepted step; see
        :func:`_unbuildable_reason` for the limits it knows about.
        """
        return (self._unbuildable is None, self._unbuildable or "")

    def require_buildable(self) -> None:
        """Raise :class:`UnbuildableSchedule` unless code can be generated.

        Called by everything that is about to ask loopy for code, so that the
        refusal names the schedule and the limit rather than arriving as a
        ``LoopyError`` from inside code generation.
        """
        if self._unbuildable is None:
            return
        raise UnbuildableSchedule(
            f"{self!r} cannot be built for the {self._target} target: "
            f"{self._unbuildable}",
            reason=self._unbuildable,
            fact=next(
                (fact for fact in self._facts if fact.kind == "buildable"), None
            ),
        )

    def retarget(self, target: str) -> Schedule:
        """The same transformations, checked again against another target.

        A schedule is written against a target: the lowering it starts from has
        that target's dtypes and its code generator, and the question of what
        can be built is a question about that target. So retargeting is not a
        relabelling; it lowers the term again and replays every step, which
        re-checks every cast and re-asks the buildability question. The casts
        will answer the same way, because they are about meaning; the third
        question need not.

        This is what ``loopty run --target opencl`` uses on a file whose
        schedules are written for ``"c"``, rather than running them on C and
        reporting a device run that never happened.
        """
        if target == self._target:
            return self
        out = Schedule(self._source, target=target, sizes=dict(self._sizes))
        for method, args, kwargs in self._steps:
            out = getattr(out, method)(*args, **kwargs)
        if self._examples is not None:
            out = out.example(**self._examples)
        return out

    def example(self, **arrays: Any) -> Schedule:
        """Record example inputs for ``loopty run``; returns a new schedule."""
        other = self._clone()
        other._examples = dict(arrays)
        return other

    @property
    def examples(self) -> dict[str, Any] | None:
        """The example inputs recorded with :meth:`example`, if any."""
        return None if self._examples is None else dict(self._examples)

    @property
    def key(self) -> str:
        """The schedule, as the ids of its facts name it.

        The kernel's name, the target in brackets, and every step as it was
        called, with every argument it was given::

            spmv[c].split('j', 2, inner='j_in', outer='j_out').realize('y', tree=True)

        The steps are the recipes :meth:`retarget` replays, so two schedules
        with one key are one schedule. :attr:`history` and the repr are for
        reading and leave out what does not change the text (``split(j, 2)``
        does not name its halves), which two different schedules can differ
        in. The id of a cast fact is ``cast:`` and this key up to the step it
        is about, then its kind; that of the agreement a run of the schedule
        records (:func:`loopty.executor.agreement`) is ``agreement:`` and the
        whole key.
        """
        return _key(self._term.name, self._target, self._steps)

    def __repr__(self) -> str:
        steps = "".join(f".{step}" for step in self._history)
        return f"Schedule({self._term.name}, target={self._target!r}){steps}"

    # }}}

    # {{{ transformations

    def tag(self, **inames: str) -> Schedule:
        """Tag inames with target coordinates, such as ``r="g.0"``.

        A tag is a coordinate in the target's execution type, and the parallel
        ones are the interesting case: they remove the iname from the order, so
        tagging a loop that carries a dependence is rejected here rather than
        producing a race at run time.

        A name that is neither a loop nor the loop of a reduction, and a tag
        loopy cannot read, are refused with a ``ValueError`` before anything
        else, as :meth:`split` refuses an unknown loop. loopy's own
        ``tag_inames`` refuses both too, but it is not asked once a step has
        left the schedule with no kernel (see :attr:`kernel`), and the steps
        after that one are still checked, so that their facts say what the
        schedule is.
        """
        from loopy.kernel.data import parse_tag

        draft = self._draft()
        unknown = [
            name
            for name in inames
            if name not in draft.order and name not in draft.reductions
        ]
        if unknown:
            listed = ", ".join(repr(name) for name in unknown)
            verb = "is not an iname" if len(unknown) == 1 else "are not inames"
            raise ValueError(f"{listed} {verb} of {self._term.name}")
        for name, tag in inames.items():
            try:
                parse_tag(tag)
            except ValueError as exc:
                raise ValueError(
                    f"tag({name}={tag!r}): loopy cannot read the tag: {exc}"
                ) from exc
        for name, tag in inames.items():
            if name not in draft.reductions or not parallel_tag(tag):
                continue
            # Running the pieces of a reduction at the same time sums them in an
            # order the source did not write, which is a reassociation and needs
            # the accumulation's permission. The permission belongs to the
            # reduction this iname is an iname *of*: two reductions can write
            # one array with different exactness, and consulting the first one
            # found would read the wrong contract.
            accumulated, exactness = draft.reduction_info[draft.reductions[name]]
            if exactness == "exact":
                message = (
                    f"tag({name}={tag!r}) illegal: it would run the pieces of "
                    f"the accumulation into {accumulated} at the same time, "
                    "which reassociates an exact reduction"
                )
                raise IllegalCast(
                    message,
                    witness=None,
                    fact=self._fact(
                        "exactness",
                        f"the accumulation into {accumulated} may be reassociated",
                        status="refuted",
                        witness=None,
                        detail=f"the accumulation into {accumulated} is exact",
                        step=("tag", (), dict(inames)),
                        reason=message,
                        about=accumulated,
                    ),
                )
            draft.reassoc.add(accumulated)
        draft.tags.update(inames)
        draft.kernel = _transformed(draft.kernel, lp.tag_inames, dict(inames))
        text = "tag(" + ", ".join(f"{k}={v!r}" for k, v in inames.items()) + ")"
        return self._commit(draft, text, ("tag", (), dict(inames)))

    def split(
        self,
        iname: str,
        factor: int,
        inner: str | None = None,
        outer: str | None = None,
    ) -> Schedule:
        """Split ``iname`` by ``factor`` into an outer and an inner iname."""
        inner = inner or f"{iname}_inner"
        outer = outer or f"{iname}_outer"
        if iname in self._reductions:
            return self._split_reduction(iname, factor, inner, outer)
        if iname not in self._order:
            raise ValueError(f"{iname!r} is not an iname of {self._term.name}")
        _check_factor(factor)
        draft = self._draft()
        text = f"split({iname}, {factor})"
        self._reindex_into(
            draft,
            _reindexing(
                (iname,),
                (outer, inner),
                [f"a0 = {factor} * b0 + b1", f"0 <= b1 < {factor}"],
            ),
            text,
        )
        position = draft.order.index(iname)
        draft.order[position : position + 1] = [outer, inner]
        draft.kernel = _transformed(
            draft.kernel,
            lp.split_iname,
            iname,
            factor,
            inner_iname=inner,
            outer_iname=outer,
        )
        return self._commit(
            draft, text, ("split", (iname, factor), {"inner": inner, "outer": outer})
        )

    def _reindex_into(self, draft: _Draft, mapping: isl.Map, text: str) -> None:
        """Record ``mapping`` as the draft's reindexing, and rename its loops.

        ``mapping`` goes from the loops it replaces to the loops that replace
        them, by name. Every statement that runs in those loops gets the
        outputs in place of the inputs, as one block where the first input
        was: the coordinates of an instance are a canonical order, not the
        loop order, which each transformation sets for itself. A statement in
        some of the loops and not the others gets the part of the map over its
        own loops when the map has one (see :func:`_part`), as a statement in
        one of two tiled loops is split and not tiled, and is refused when the
        map mixes its loops, because the map has no meaning for it then. An
        output spelled like a loop, a size, an array or another name of the
        kernel that the map does not replace is refused too, since it would
        make two things one.

        A loop whose extent comes from an array passes that property to each
        new loop whose value depends on it (see :func:`_inherited`): both
        halves of a split ragged loop, and neither half of the dense loop it
        is tiled with.
        """
        if mapping.has_tuple_name(isl.dim_type.in_) or mapping.has_tuple_name(
            isl.dim_type.out
        ):
            raise ValueError(
                f"{text}: a map with a named tuple, such as S0[t, i], would "
                "move one statement and not the others, and loopy gives the "
                "statements of a loop one domain; name the loops only"
            )
        inputs = _dim_names(mapping, isl.dim_type.in_)
        outputs = _dim_names(mapping, isl.dim_type.out)
        if not inputs:
            raise ValueError(f"{text}: the map names no loop to replace")
        for side, names in (("input", inputs), ("output", outputs)):
            doubled = sorted({name for name in names if names.count(name) > 1})
            if doubled:
                raise ValueError(
                    f"{text}: {', '.join(doubled)} is named twice among the "
                    f"map's {side}s"
                )
        folded = [name for name in inputs if name in draft.reductions]
        if folded:
            raise ValueError(
                f"{text}: {', '.join(folded)} is the loop of a reduction, which "
                "runs inside one statement instance and has no instances to "
                "rename; split it instead"
            )
        unknown = [name for name in inputs if name not in draft.order]
        if unknown:
            raise ValueError(
                f"{text}: not loops of {self._term.name}: {', '.join(unknown)}"
            )
        self._check_new_names(draft, outputs, inputs, text)
        foreign = sorted(
            set(mapping.get_var_names(isl.dim_type.param))
            - set(self._params)
            - set(self._term.sizes)
        )
        if foreign:
            raise ValueError(
                f"{text}: the map's parameters {', '.join(foreign)} are not "
                f"sizes of {self._term.name}"
            )
        mappings: dict[str, isl.Map] = {}
        for stmt_id, coords in list(draft.coords.items()):
            inside = [name for name in inputs if name in coords]
            if not inside:
                continue
            own = mapping
            if len(inside) != len(inputs):
                part = _part(mapping, inside)
                if part is None:
                    outside = [name for name in inputs if name not in coords]
                    raise ValueError(
                        f"{text}: {stmt_id} runs in {', '.join(inside)} but not "
                        f"in {', '.join(outside)}, and the map mixes them, so it "
                        "has no part that moves the loops of that statement alone"
                    )
                own = part
            first = min(coords.index(name) for name in inside)
            kept = [name for name in coords if name not in inside]
            new = _dim_names(own, isl.dim_type.out)
            draft.coords[stmt_id] = (*kept[:first], *new, *kept[first:])
            mappings[stmt_id] = own
        dependent = draft.data_dependent & set(inputs)
        if dependent:
            inherited = _inherited(mapping)
            draft.data_dependent -= set(inputs)
            draft.data_dependent |= {
                name for name in outputs if inherited[name] & dependent
            }
        draft.mappings = mappings

    def _check_new_names(
        self,
        draft: _Draft,
        names: Sequence[str],
        replaced: Sequence[str],
        text: str,
    ) -> None:
        """Refuse a new loop name that generated code cannot use as one.

        It has to be an identifier C does not reserve, and it may not be the
        name of anything else in the kernel: a loop or a reduction loop that
        stays, a size, an array, a domain parameter, or an argument or
        temporary the lowering added (the offsets of a ragged array, say).
        Taking one of those would make two things one, which loopy and isl
        otherwise report from inside. The loops being replaced may be reused.
        """
        entry = self._lowering.kernel.default_entrypoint
        taken = (
            set(draft.order)
            | set(draft.reductions)
            | set(self._params)
            | set(self._term.sizes)
            | {name for name, _ in self._term.params}
            | (set(entry.all_variable_names()) - set(entry.all_inames()))
        ) - set(replaced)
        for name in names:
            if not name.isidentifier() or is_reserved(name):
                raise ValueError(
                    f"{text}: {name!r} cannot name a loop in generated code"
                )
            if name in taken:
                raise ValueError(
                    f"{text}: the new loop {name!r} would share its name with "
                    f"a loop, a size, an array or another name of "
                    f"{self._term.name}"
                )

    def _split_reduction(
        self, iname: str, factor: int, inner: str, outer: str
    ) -> Schedule:
        """Split a reduction iname: the instances are untouched, the sum is not.

        Splitting the reduced domain is the first half of realizing a reduction
        as a tree; on its own it changes nothing about exactness, because the
        pieces still run in order. Tagging the inner piece parallel is what
        reassociates, and that is checked in :meth:`tag`.
        """
        _check_factor(factor)
        draft = self._draft()
        text = f"split({iname}, {factor})"
        if inner == outer:
            raise ValueError(f"{text}: {inner} is named twice among the new loops")
        self._check_new_names(draft, (outer, inner), (iname,), text)
        key = draft.reductions.pop(iname)
        draft.reductions[inner] = key
        draft.reductions[outer] = key
        if iname in draft.data_dependent:
            draft.data_dependent.discard(iname)
            draft.data_dependent.update((inner, outer))
        draft.kernel = _transformed(
            draft.kernel,
            lp.split_iname,
            iname,
            factor,
            inner_iname=inner,
            outer_iname=outer,
        )
        return self._commit(
            draft,
            text,
            ("split", (iname, factor), {"inner": inner, "outer": outer}),
        )

    def interchange(self, *inames: str) -> Schedule:
        """Reorder the named loops into the order given.

        With two names this is the familiar interchange; with more it is a
        permutation. The instances are untouched, so the bijection is trivial and
        the whole question is whether the dependences still run forward. As an
        affine map it is the identity, with a new order; :meth:`affine` given
        the permutation of the loops that keeps their names makes the same
        cast, and this is the spelling that leaves the kernel alone.
        """
        return self._reorder(
            inames, f"interchange({', '.join(inames)})", "interchange"
        )

    def prioritize(self, *inames: str) -> Schedule:
        """Alias of :meth:`interchange`, in loopy's vocabulary."""
        return self._reorder(
            inames, f"prioritize({', '.join(inames)})", "prioritize"
        )

    def _reorder(self, inames: Sequence[str], text: str, method: str) -> Schedule:
        unknown = [iname for iname in inames if iname not in self._order]
        if unknown:
            raise ValueError(f"not inames of {self._term.name}: {unknown}")
        draft = self._draft()
        positions = sorted(draft.order.index(iname) for iname in inames)
        for position, iname in zip(positions, inames, strict=True):
            draft.order[position] = iname
        return self._commit(draft, text, (method, tuple(inames), {}))

    def tile(
        self,
        first: str,
        second: str,
        first_factor: int,
        second_factor: int,
    ) -> Schedule:
        """Split two loops and interchange the pieces, tiling the nest.

        One cast, not three: the tiles are what the user asked for, so the
        witness of a rejection names the tiling and not the interchange inside
        it. The reindexing is one affine map, the two splits side by side,
        and the kernel is split with loopy's own ``split_iname``, whose loop
        bounds loopy knows how to simplify.
        """
        for iname in (first, second):
            if iname not in self._order:
                raise ValueError(f"{iname!r} is not an iname of {self._term.name}")
        if first == second:
            raise ValueError(f"tile() needs two different loops, not {first!r} twice")
        for factor in (first_factor, second_factor):
            _check_factor(factor)
        draft = self._draft()
        text = f"tile({first},{second},{first_factor},{second_factor})"
        outer_first, inner_first = f"{first}_outer", f"{first}_inner"
        outer_second, inner_second = f"{second}_outer", f"{second}_inner"
        self._reindex_into(
            draft,
            _reindexing(
                (first, second),
                (outer_first, inner_first, outer_second, inner_second),
                [
                    f"a0 = {first_factor} * b0 + b1",
                    f"0 <= b1 < {first_factor}",
                    f"a1 = {second_factor} * b2 + b3",
                    f"0 <= b3 < {second_factor}",
                ],
            ),
            text,
        )
        for iname, factor, outer, inner in (
            (first, first_factor, outer_first, inner_first),
            (second, second_factor, outer_second, inner_second),
        ):
            position = draft.order.index(iname)
            draft.order[position : position + 1] = [outer, inner]
            draft.kernel = _transformed(
                draft.kernel,
                lp.split_iname,
                iname,
                factor,
                inner_iname=inner,
                outer_iname=outer,
            )
        wanted = [outer_first, outer_second, inner_first, inner_second]
        positions = sorted(draft.order.index(iname) for iname in wanted)
        for position, iname in zip(positions, wanted, strict=True):
            draft.order[position] = iname
        return self._commit(
            draft,
            text,
            ("tile", (first, second, first_factor, second_factor), {}),
        )

    def skew(self, iname: str, by: str, factor: int = 1) -> Schedule:
        """Skew ``iname`` by ``factor`` times ``by``, making tiling legal.

        The instances are renamed, so this is the one transformation whose
        bijectivity is a real question, and the reason a skew makes a rectangular
        tiling legal is visible in the map: the dependence vectors it adds to
        every instance are exactly what stops a tile boundary from running a sink
        before its source.

        A skew is the affine map ``(by, iname) -> (by, iname + factor * by)``
        with both loops keeping their names, and it is checked and applied to
        the kernel exactly as :meth:`affine` would apply that map. The loop
        order is unchanged.
        """
        if iname not in self._order or by not in self._order:
            raise ValueError(f"not inames of {self._term.name}: {iname}, {by}")
        if iname == by:
            raise ValueError(f"skew() needs two different loops, not {iname!r} twice")
        draft = self._draft()
        text = f"skew({iname}, by={by!r}" + (
            f", factor={factor})" if factor != 1 else ")"
        )
        mapping = _reindexing(
            (by, iname), (by, iname), ["b0 = a0", f"b1 = a1 + {factor} * a0"]
        )
        self._reindex_into(draft, mapping, text)
        self._affine_into(draft, mapping)
        return self._commit(
            draft, text, ("skew", (iname,), {"by": by, "factor": factor})
        )

    def affine(self, mapping: Any) -> Schedule:
        """Reindex loops along an injective affine map, checked like every cast.

        ``mapping`` is an isl map, or its text, from loops of the kernel to the
        loops that replace them, by name::

            schedule.affine("{ [t, i] -> [a, b] : a = t + i and b = t - i }")

        Every statement that runs in the loops named on the left gets the loops
        named on the right instead, which take the places of the old ones in
        the loop order (one for one when there are as many; as one block where
        the first was otherwise). A statement in only some of those loops gets
        the part of the map over them, when the map is that part side by side
        with the rest, and is refused otherwise. A new loop may keep the name
        of one it replaces, which is how :meth:`skew` is this method with a
        particular map, and its parameters, if any, are sizes of the kernel.

        The map is untrusted like any other transformation. It has to be
        defined on every instance and send no two of them to one point (the
        ``bijective`` fact, refuted with the instance it misses or the pair it
        merges), and the new order has to run every dependence forward (the
        ``monotone`` fact, refuted with the pair of instances and the array
        cell between them). It need not be unimodular: the image of the diamond
        above is only the points of equal parity, and the kernel is rewritten
        over that image, as the module docstring describes.

        What the kernel rewrite cannot express, such as loops that more than
        one loopy domain defines, is a ``refuted`` ``buildable`` fact with the
        reason, and the schedule then has no kernel.
        """
        mapping = _as_map(mapping)
        text = f"affine({mapping})"
        draft = self._draft()
        self._reindex_into(draft, mapping, text)
        inputs = _dim_names(mapping, isl.dim_type.in_)
        outputs = _dim_names(mapping, isl.dim_type.out)
        tagged = sorted(name for name in inputs if name in draft.tags)
        if tagged:
            raise ValueError(
                f"{text}: {', '.join(tagged)} carries a tag; apply the map "
                "before tagging the loops it makes"
            )
        positions = sorted(draft.order.index(name) for name in inputs)
        if len(outputs) == len(inputs):
            for position, name in zip(positions, outputs, strict=True):
                draft.order[position] = name
        else:
            kept = [name for name in draft.order if name not in inputs]
            first = positions[0]
            draft.order = [*kept[:first], *outputs, *kept[first:]]
        self._affine_into(draft, mapping)
        return self._commit(draft, text, ("affine", (mapping,), {}))

    def _affine_into(self, draft: _Draft, mapping: isl.Map) -> None:
        """Rewrite the draft's kernel along ``mapping``, or record why not."""
        if draft.kernel is None:
            return
        kernel, reason = _affine_kernel(draft.kernel, mapping)
        if reason is None:
            draft.kernel = kernel
        else:
            draft.kernel = None
            draft.unbuildable = reason

    def realize(self, var: str, tree: bool = True) -> Schedule:
        """Realize an accumulation, optionally as a reduction tree.

        A tree reassociates, so the result's exactness class drops to
        ``reassoc`` and the fact records it; over an ``exact`` accumulation the
        cast is rejected instead, because ``exact`` is a request for the bits and
        a tree does not give them.
        """
        exactness = self._exactness_of(var)
        if exactness is None:
            raise ValueError(f"{var!r} is not accumulated by {self._term.name}")
        text = f"realize({var!r}, tree={tree})"
        if tree and exactness == "exact":
            message = (
                f"{text} illegal: the accumulation into {var} is exact, and a "
                "reduction tree reassociates it; ask for the accumulation at "
                "'reassoc' if the bits may change"
            )
            fact = self._fact(
                "exactness",
                f"the accumulation into {var} may be reassociated",
                status="refuted",
                witness=None,
                detail=f"the accumulation into {var} is exact",
                step=("realize", (var,), {"tree": tree}),
                reason=message,
                about=var,
            )
            raise IllegalCast(message, witness=None, fact=fact)
        draft = self._draft()
        if tree:
            draft.reassoc.add(var)
        return self._commit(draft, text, ("realize", (var,), {"tree": tree}))

    def _exactness_of(self, var: str) -> str | None:
        """The strictest exactness class of any accumulation into ``var``.

        ``realize`` is a statement about the whole accumulation into an array,
        so when several reductions write one array it has to answer for all of
        them. Taking the strictest is the conservative join: one ``exact``
        reduction among a dozen ``approx`` ones still forbids a reduction tree,
        which is the direction a refusal has to err in. :meth:`tag` asks a
        narrower question and gets a narrower answer, through
        ``_Draft.reduction_info``.
        """
        classes: list[str] = []
        for stmt in self._term.stmts:
            if stmt.assignee.array != var:
                continue
            reductions = reductions_of(stmt.expr)
            classes.extend(reduction.exactness for reduction in reductions)
            if not reductions and stmt.kind == "accumulate":
                classes.append(_element_exactness(self._term, var))
        known = [name for name in classes if name in _EXACTNESS_ORDER]
        if not known:
            return classes[0] if classes else None
        return min(known, key=_EXACTNESS_ORDER.index)

    # }}}

    # {{{ the checker

    def _commit(
        self,
        draft: _Draft,
        text: str,
        recipe: tuple[str, tuple, dict],
    ) -> Schedule:
        """Check one transformation and return the schedule it produces.

        ``recipe`` is how the transformation would be written in Python, as
        ``(method, args, kwargs)``, kept so that :meth:`retarget` can replay it,
        and so that the facts about this step name it (see :attr:`key`).
        """
        layout = _Layout(
            stmt_ids=self._layout.stmt_ids, coords=dict(draft.coords)
        )
        step = _step_map(self._layout, layout, draft.mappings)

        facts: list[Any] = []

        # Defined on every instance, and one for one there: a map that misses
        # an instance drops it from the program as surely as one that merges
        # two, and only a caller's map (``affine``) can do either.
        verdict = isl_oracle.is_bijection_on(step, self._instances)
        step = step.intersect_domain(self._instances)
        message = (
            ""
            if verdict.ok
            else f"{text} illegal: the reindexing is not a bijection on the "
            f"instances of {self._term.name}; {verdict.detail}"
        )
        facts.append(
            self._fact(
                "bijective",
                f"{text} renames the instances of {self._term.name} one for one",
                status="decided" if verdict.ok else "refuted",
                witness=verdict.witness,
                detail=verdict.detail,
                step=recipe,
                reason=message,
            )
        )
        if not verdict.ok:
            raise IllegalCast(message, witness=verdict.witness, fact=facts[-1])

        reindex = self._reindex.apply_range(step)
        instances = step.range()
        time = _time_map(layout, draft.order, draft.tags)
        schedule = reindex.apply_range(time).intersect_domain(self._origin)

        total = self._deps_total
        overall = (
            isl_oracle.is_monotone(schedule, total)
            if total is not None
            else isl_oracle.Verdict(True, None, "no dependences to violate")
        )
        # Attribution costs one isl question per dependence, and is only needed
        # to explain a refusal, so it is asked only when there is one to explain.
        bad = None if overall.ok else self._first_violation(schedule)
        refused = bad is not None or not overall.ok
        witness = overall.witness if bad is None else bad[1]
        if bad is not None:
            message = self._render_violation(text, *bad)
        elif refused:
            message = f"{text} illegal: {overall.detail}"
        else:
            message = ""
        facts.append(
            self._fact(
                "monotone",
                f"the order after {text} runs every dependence of "
                f"{self._term.name} forward",
                status="refuted" if refused else "decided",
                witness=witness,
                detail=overall.detail if bad is None else bad[2],
                step=recipe,
                reason=message,
            )
        )
        if refused:
            raise IllegalCast(message, witness=witness, fact=facts[-1])

        other = self._clone()
        other._layout = layout
        other._instances = instances
        other._reindex = reindex
        other._order = list(draft.order)
        other._tags = dict(draft.tags)
        other._kernel = _transformed(
            draft.kernel,
            _with_priority,
            _nests(layout, draft.order, draft.tags).values(),
        )
        other._reassoc = frozenset(draft.reassoc)
        other._reductions = dict(draft.reductions)
        other._reduction_info = dict(draft.reduction_info)
        other._data_dependent = frozenset(draft.data_dependent)
        # The cast is legal; whether the target can build it is a separate
        # question, about the schedule as it now stands, so it is asked again
        # at every step. A kernel the rewrite could not write stays unwritten;
        # anything else is read off the draft, and a later step can put right
        # what an earlier one broke, such as an order loopy cannot keep.
        if draft.kernel is None:
            reason = draft.unbuildable or self._unbuildable
        else:
            reason = _unbuildable_reason(draft)
        other._unbuildable = reason
        earlier = self._facts
        if reason != self._unbuildable:
            # The one ``buildable`` fact a schedule carries is about its
            # current reason, and is the fact ``require_buildable`` raises
            # with, so it goes when the reason goes or changes.
            earlier = tuple(fact for fact in earlier if fact.kind != "buildable")
            if reason is not None:
                facts.append(
                    self._fact(
                        "buildable",
                        f"{self._target} code can be generated for "
                        f"{self._term.name} after {text}",
                        status="refuted",
                        witness=None,
                        detail=reason,
                        step=recipe,
                        oracle="loopy-target",
                        reason=reason,
                    )
                )
        for accumulated in sorted(set(draft.reassoc) - set(self._reassoc)):
            facts.append(
                self._fact(
                    "exactness",
                    f"the accumulation into {accumulated} is reassociated by "
                    f"{text}, so its result is compared at 'reassoc'",
                    status="decided",
                    witness=None,
                    detail=f"exactness of {accumulated} lowered to reassoc",
                    step=recipe,
                    about=accumulated,
                )
            )
        other._history = (*self._history, text)
        other._steps = (*self._steps, recipe)
        other._facts = (*earlier, *facts)
        return other

    def _first_violation(
        self, schedule: isl.Map
    ) -> tuple[_Dep, tuple, str] | None:
        """The first dependence the new order runs backwards, with its witness.

        Checking dependence by dependence rather than on the union is what lets
        the message name the array cell: the pair of instances alone does not say
        which access made them dependent.
        """
        for dep in self._deps:
            verdict = isl_oracle.is_monotone(schedule, dep.relation)
            if verdict.ok:
                continue
            witness = self._witness(schedule, dep)
            return dep, witness, verdict.detail
        return None

    def _witness(self, schedule: isl.Map, dep: _Dep) -> tuple:
        """A concrete violating pair, with the size parameters instantiated.

        The verdict is decided with the sizes free; the witness is printed with
        them fixed, because "instance ``S[t=0, i=8]``" is a sentence and
        "instance ``S[t=0, i=n-8]``" is a puzzle. The hint given to the schedule
        is preferred, and isl chooses when there is none.
        """
        relation = dep.relation.subtract(
            isl.Map.identity(dep.relation.get_space().domain().map_from_set())
        )
        timed = relation.apply_domain(schedule).apply_range(schedule)
        lex = isl.Map.lex_lt(timed.get_space().domain())
        violating_times = timed.subtract(lex)
        inverse = schedule.reverse()
        bad = (
            violating_times.apply_domain(inverse)
            .apply_range(inverse)
            .intersect(relation)
        )
        if self._sizes:
            names = bad.get_var_names(isl.dim_type.param)
            fixed = bad
            for name, value in self._sizes.items():
                if name in names:
                    fixed = fixed.fix_val(
                        isl.dim_type.param,
                        names.index(name),
                        isl.Val.int_from_si(bad.get_ctx(), value),
                    )
            if not fixed.is_empty():
                bad = fixed
        n_params = bad.dim(isl.dim_type.param)
        n_in = bad.dim(isl.dim_type.in_)
        wrapped = bad.wrap()
        if n_params:
            wrapped = wrapped.move_dims(
                isl.dim_type.set, 0, isl.dim_type.param, 0, n_params
            )
        point = isl_oracle.sample_point(wrapped)
        if point is None:  # pragma: no cover - the map is known to be non-empty
            return ()
        params = dict(
            zip(
                bad.get_var_names(isl.dim_type.param),
                point[:n_params],
                strict=False,
            )
        )
        source = point[n_params : n_params + n_in]
        sink = point[n_params + n_in :]
        return (self._name_instance(source), self._name_instance(sink), params)

    def _name_instance(self, point: Sequence[int]) -> tuple[str, dict[str, int]]:
        """A uniform-space point read back as ``(statement id, coordinates)``."""
        stmt_id = self._origin_layout.stmt_ids[int(point[0])]
        coords = self._origin_layout.coords[stmt_id]
        return stmt_id, {
            iname: int(value)
            for iname, value in zip(coords, point[1:], strict=False)
        }

    def _render_violation(self, text: str, dep: _Dep, witness: tuple, _: str) -> str:
        """The rejection message: which instance, which cell, which way round.

        The sizes are part of the message, not decoration. The verdict is
        decided with them free, so the witness is one violating pair out of
        many, and which one isl picks depends on the ``sizes`` hint (and, when
        the hint makes the violating set empty, is isl's own choice instead).
        A reader comparing two runs of the same demo needs to see at which sizes
        the pair was read off, or the numbers look unstable.
        """
        if not witness:  # pragma: no cover - a violation always has a witness
            return f"{text} illegal: {dep.kind} on {dep.array} runs backwards"
        (source_id, source_coords), (sink_id, sink_coords), params = witness
        source_verb, sink_verb = dep.verbs()
        cell = _cell_text(dep.source_indices, source_coords, params)
        at = _sizes_text(params, self._sizes)
        return (
            f"{text} illegal: instance {_instance_text(source_id, source_coords)} "
            f"{source_verb} {dep.array}[{cell}] {sink_verb} by "
            f"{_instance_text(sink_id, sink_coords)} scheduled earlier{at}"
        )

    # }}}

    def _fact(
        self,
        kind: str,
        statement: str,
        status: str,
        witness: Any,
        detail: str,
        step: tuple[str, tuple, dict],
        oracle: str = "isl",
        reason: str = "",
        about: str = "",
    ) -> Any:
        """One ledger entry for one question about one step.

        ``step`` is the step's recipe, ``(method, args, kwargs)``, and the
        fact's id is :attr:`key` with it as the last step: the steps up to
        this one, and not a count of them, because two schedules of one kernel
        both have a first step and keep facts in one ledger. ``about`` tells
        apart two facts of one kind about one step, such as the exactness of
        two accumulations one ``tag`` reassociates, and is the array's name.

        ``oracle`` is who answered: ``isl`` for the two questions about meaning,
        ``loopy-target`` for the one about what the backend can generate.

        ``detail`` is the answer in the oracle's own words, and every fact
        keeps it. A refuted fact also carries ``reason``, the explanation a
        reader is owed: for a refused cast, the message of the
        :class:`IllegalCast` it raises, and for a schedule the target cannot
        build, the limit in words. ``reason`` is what lanky prints under a
        ``REFUTED`` line (``lanky.cli.refutation_lines``), and ``detail``
        reaches only the JSON ledger, so a refutation explained by ``detail``
        alone reads as unexplained on the screen. A refuted fact given no
        ``reason`` falls back on its ``detail``, so none is left without one.

        ``witness`` is recorded when isl gave one, which is for the two
        questions about meaning: an instance a reindexing misses, or a pair of
        instances that shows it is not one for one, or the dependence a new
        order runs backwards.
        Exactness and buildability are not questions for isl, and their facts
        have none.
        """
        from lanky.ledger import Fact, Status

        provenance: dict[str, Any] = {
            "oracle": oracle,
            "detail": detail,
            "dependences": self._flow_note,
            "target": self._target,
        }
        if witness:
            provenance["witness"] = witness
        if status == "refuted":
            provenance["reason"] = reason or detail
        suffix = f":{about}" if about else ""
        return Fact(
            id=f"cast:{_key(self._term.name, self._target, (*self._steps, step))}"
            f":{kind}{suffix}",
            kind=kind,
            statement=statement,
            term=None,
            status=Status.REFUTED if status == "refuted" else Status.DECIDED,
            decided_by=oracle,
            provenance=provenance,
            where=self._term.stmts[0].where if self._term.stmts else "",
            owner=self._term.name,
        )

    def facts(self) -> tuple:
        """The cast facts accumulated by the transformations applied so far."""
        return self._facts


def _key(name: str, target: str, steps: Sequence[tuple[str, tuple, dict]]) -> str:
    """``spmv[c].split('j', 2, inner='j_in', outer='j_out')``: :attr:`Schedule.key`."""
    return f"{name}[{target}]" + "".join(f".{_call_text(step)}" for step in steps)


def _call_text(step: tuple[str, tuple, dict]) -> str:
    """One recipe as the call it is, every argument written out.

    An isl map, which is what :meth:`Schedule.affine` records, is written as
    its text, which is what ``affine`` also accepts.
    """
    method, args, kwargs = step

    def shown(value: Any) -> str:
        if isinstance(value, isl.Map | isl.BasicMap):
            return repr(str(value))
        return repr(value)

    parts = [shown(arg) for arg in args]
    parts += [f"{name}={shown(value)}" for name, value in kwargs.items()]
    return f"{method}({', '.join(parts)})"


def _sizes_text(params: dict[str, int], hint: dict[str, int]) -> str:
    """`` at n=16, nx=16`` (`` hinted``), or nothing when there are no sizes."""
    if not params:
        return ""
    inner = ", ".join(f"{name}={value}" for name, value in sorted(params.items()))
    honoured = all(hint.get(name, value) == value for name, value in params.items())
    how = "as hinted" if hint and honoured else "isl's choice"
    return f" (at {inner}, {how})"


def _instance_text(stmt_id: str, coords: dict[str, int]) -> str:
    """``S[t=0, i=8]``."""
    inner = ", ".join(f"{name}={value}" for name, value in coords.items())
    return f"{stmt_id}[{inner}]"


def _cell_text(
    indices: Sequence[Any], coords: dict[str, int], params: dict[str, int]
) -> str:
    """The array cell an instance touches, evaluated where it can be."""
    from pymbolic import evaluate

    context = {**params, **coords}
    out = []
    for index in indices:
        try:
            out.append(str(evaluate(_plain(index), context)))
        except Exception:
            out.append(str(index))
    return ", ".join(out)


def _set_over(stmt: Stmt) -> isl.Set:
    """A statement's domain with its set dimensions named after its inames."""
    domain = stmt.domain
    for k, iname in enumerate(stmt.inames):
        domain = domain.set_dim_name(isl.dim_type.set, k, iname)
    return domain


#: Exactness classes, strictest first. ``exact`` is a request for the bits;
#: ``approx`` promises only a few digits.
_EXACTNESS_ORDER = ("exact", "reassoc", "approx")


def _element_exactness(term: Term, name: str) -> str:
    """The exactness class of an array's element type.

    A program's temporaries are arrays with element sorts too
    (:attr:`loopty.term.Term.temporaries`), and are asked like parameters.
    """
    for param, typ in (*term.params, *term.temporaries):
        if param != name:
            continue
        dtype = getattr(typ, "dtype", typ)
        try:
            from lanky.prelude import exactness_of

            return exactness_of(dtype)
        except Exception:
            break
    return "approx"


def _check_factor(factor: int) -> None:
    """A split or tile factor has to be a positive integer."""
    if factor < 1:
        raise ValueError(f"a split factor must be positive, not {factor}")


class _Inexpressible(ValueError):
    """A reindexing the kernel rewrite cannot write for loopy, and why."""


def _affine_kernel(kernel: Any, mapping: isl.Map) -> tuple[Any, str | None]:
    """The loopy kernel reindexed along ``mapping``, or why it cannot be.

    ``lp.map_domain`` would be the obvious call, and it is what the skew used
    to make. It solves the map for each old iname and accepts only an equation
    with a unit coefficient, so it refuses every map that is not unimodular,
    the diamond among them ("No suitable equation for 't' found"), and some
    that are, depending on the order in which isl eliminates. The same rewrite
    is done here from what isl says about the map directly:

    * the domain that defines the mapped loops becomes its image under the
      map (the identity on its other loops), which isl states exactly, with an
      existentially quantified constraint where the image has holes;
    * a domain nested in it, which names mapped loops as parameters (the fiber
      of a ragged loop, the domain of a reduction), becomes its image too, with
      the new loops as its parameters;
    * each old loop variable is replaced, in every instruction, by the
      quasi-affine expression of the new ones that isl gives for the inverse on
      that domain, such as ``floor((a + b)/2)``, which is exact on the image.

    The map is the same object the checker reasons about, so the two cannot
    drift apart. What cannot be written this way comes back as the reason, and
    the kernel unchanged: loops that no one domain defines, an image that is
    not one basic set, an inverse that is piecewise, or an instruction in some
    of the mapped loops and not the others.
    """
    from loopy.match import parse_stack_match
    from loopy.symbolic import (
        RuleAwareSubstitutionMapper,
        SubstitutionRuleMappingContext,
        pw_aff_to_expr,
    )
    from pymbolic.mapper.substitutor import make_subst_func

    inputs = _dim_names(mapping, isl.dim_type.in_)
    outputs = _dim_names(mapping, isl.dim_type.out)
    mapped = set(inputs)
    loops = ", ".join(inputs)
    entry = kernel.default_entrypoint

    homes = [
        k
        for k, domain in enumerate(entry.domains)
        if mapped & set(domain.get_var_names(isl.dim_type.set))
    ]
    if len(homes) != 1 or not mapped <= set(
        entry.domains[homes[0]].get_var_names(isl.dim_type.set)
    ):
        return kernel, (
            f"the loops {loops} are not all defined by one loopy domain, and "
            "the kernel is rewritten along a map only in the domain that "
            "defines every loop the map replaces"
        )
    home = homes[0]
    try:
        domains = list(entry.domains)
        domains[home] = _image(entry.domains[home], mapping)
        for k, domain in enumerate(entry.domains):
            if k != home and mapped & set(domain.get_var_names(isl.dim_type.param)):
                domains[k] = _image_of_params(domain, mapping)
        inverse = _inverse(mapping, entry.domains[home])
        substitution = {}
        for k, name in enumerate(inputs):
            piece = inverse.get_pw_aff(k).coalesce()
            if piece.n_piece() != 1:
                raise _Inexpressible(
                    f"the inverse of the map is piecewise in {name} ({piece}), "
                    "and a loop variable is replaced by one expression"
                )
            substitution[name] = pw_aff_to_expr(piece)
    except _Inexpressible as exc:
        return kernel, str(exc)
    except isl.Error as exc:
        return kernel, f"isl could not rewrite the kernel along the map: {exc}"

    insns = []
    for insn in entry.instructions:
        inside = mapped & insn.within_inames
        if inside and inside != mapped:
            return kernel, (
                f"instruction {insn.id} runs in {', '.join(sorted(inside))} "
                f"and not in all of {loops}"
            )
        if inside:
            insn = insn.copy(
                within_inames=(insn.within_inames - mapped) | set(outputs)
            )
        insns.append(insn)
    # loopy's own transforms refuse to remap an iname a loop priority
    # mentions, and the old priority names loops that no longer exist. The
    # caller sets the priority to the nest it has just checked.
    entry = entry.copy(
        domains=domains, instructions=insns, loop_priority=frozenset()
    )
    context = SubstitutionRuleMappingContext(
        entry.substitutions, entry.get_var_name_generator()
    )
    mapper = RuleAwareSubstitutionMapper(
        context, make_subst_func(substitution), within=parse_stack_match(None)
    )
    entry = context.finish_kernel(
        mapper.map_kernel(entry, map_args=False, map_tvs=False)
    )
    return kernel.with_kernel(entry), None


def _picking(count: int, positions: Sequence[int]) -> isl.Map:
    """The map that keeps the dimensions at ``positions`` of a ``count``-tuple.

    Placeholder names, so that a loop variable spelled like an isl keyword
    never reaches the parser; isl matches the tuple by its length.
    """
    source = ", ".join(f"d{k}" for k in range(count))
    target = ", ".join(f"d{k}" for k in positions)
    return isl.Map(f"{{ [{source}] -> [{target}] }}")


def _alongside(mapping: isl.Map, count: int) -> isl.Map:
    """``mapping`` on the first dimensions, the identity on ``count`` more."""
    if not count:
        return mapping
    rest = ", ".join(f"r{k}" for k in range(count))
    return mapping.flat_product(isl.Map(f"{{ [{rest}] -> [{rest}] }}"))


def _one_basic_set(image: isl.Set, before: Any) -> isl.BasicSet:
    """``image`` as the basic set loopy wants a domain to be."""
    pieces = image.coalesce().get_basic_sets()
    if len(pieces) != 1:
        raise _Inexpressible(
            f"the image of the domain {before} is {image}, which is not one "
            "basic set, and loopy wants each domain to be one"
        )
    return pieces[0]


def _image(domain: Any, mapping: isl.Map) -> isl.BasicSet:
    """The domain that defines the mapped loops, moved along ``mapping``."""
    inputs = _dim_names(mapping, isl.dim_type.in_)
    outputs = _dim_names(mapping, isl.dim_type.out)
    names = list(domain.get_var_names(isl.dim_type.set))
    rest = [name for name in names if name not in inputs]
    positions = [names.index(name) for name in (*inputs, *rest)]
    image = (
        _as_set(domain)
        .apply(_picking(len(names), positions))
        .apply(_alongside(mapping, len(rest)))
    )
    for k, name in enumerate((*outputs, *rest)):
        image = image.set_dim_name(isl.dim_type.set, k, name)
    return _one_basic_set(image, domain)


def _image_of_params(domain: Any, mapping: isl.Map) -> isl.BasicSet:
    """A domain nested in the mapped loops, with the new loops as parameters.

    The mapped loops are moved from its parameters to the front of its own
    dimensions (a mapped loop it does not name is added, unconstrained), the
    map is applied to them, and the loops that replace them are moved back to
    the parameters, which is where loopy reads the nesting from.
    """
    inputs = _dim_names(mapping, isl.dim_type.in_)
    outputs = _dim_names(mapping, isl.dim_type.out)
    own = list(domain.get_var_names(isl.dim_type.set))
    moved = _as_set(domain)
    for name in inputs:
        if name not in moved.get_var_names(isl.dim_type.param):
            moved = moved.add_dims(isl.dim_type.param, 1)
            moved = moved.set_dim_name(
                isl.dim_type.param, moved.dim(isl.dim_type.param) - 1, name
            )
    for k, name in enumerate(inputs):
        position = moved.get_var_names(isl.dim_type.param).index(name)
        moved = moved.move_dims(isl.dim_type.set, k, isl.dim_type.param, position, 1)
    image = moved.apply(_alongside(mapping, len(own)))
    for name in outputs:
        last = image.dim(isl.dim_type.param)
        image = image.move_dims(isl.dim_type.param, last, isl.dim_type.set, 0, 1)
        image = image.set_dim_name(isl.dim_type.param, last, name)
    for k, name in enumerate(own):
        image = image.set_dim_name(isl.dim_type.set, k, name)
    return _one_basic_set(image, domain)


def _inverse(mapping: isl.Map, domain: Any) -> isl.PwMultiAff:
    """The old loops as functions of the new ones, on the loops that exist.

    Restricted to the domain first, because a map need only be one for one
    on the instances: ``(i, j) -> 10 i + j`` merges points, and is invertible
    wherever ``j`` stays below 10.
    """
    inputs = _dim_names(mapping, isl.dim_type.in_)
    names = list(domain.get_var_names(isl.dim_type.set))
    reached = _as_set(domain).apply(
        _picking(len(names), [names.index(name) for name in inputs])
    )
    return isl.PwMultiAff.from_map(mapping.intersect_domain(reached).reverse())


def _as_set(domain: Any) -> isl.Set:
    """A loopy domain, which is a basic set, as a set."""
    if isinstance(domain, isl.BasicSet):
        return isl.Set.from_basic_set(domain)
    return domain
