"""Schedules: loop transformations as casts, checked one at a time.

A transformation is untrusted. ``.split``, ``.tile``, ``.interchange``,
``.skew``, ``.affine``, ``.tag`` and ``.realize`` apply the corresponding loopy
transform and then hand the result to a small checker, which asks isl two
questions: is the reindexing a bijection on statement instances, and is the new
execution order monotone on the dependence relation? Parallel inames (``g.*``,
``l.*``, ``ilp``, ``vec``) carry no order, so they are dropped from the order
before the second question is asked, and a hardware axis (``g.*``, ``l.*``)
also may not join two work items by a dependence (see "Work items" below).
This is the de Bruijn criterion applied to scheduling: any Python
transformation is admissible because its output is checked, not its code.

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
a fourth that none of them can write. A tag belongs to a loop, so a step that
replaces loops (``.split``, ``.tile``, ``.affine``) refuses a loop that carries
one, and the loops it makes are tagged after it.

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

Work items
----------

A loop on a hardware axis is more than unordered. loopy runs it as the launch
grid, outside every loop of the kernel: each work item runs the whole kernel at
its own value of the loop, and nothing in a kernel orders two work items
through global memory. There is no barrier across groups, loopy puts one
between the work items of a group only for a local temporary, and its own
check of the barriers a kernel needs asks about pairs of instructions, so a
statement that depends on itself across a sequential loop is never asked
about. Leaving the hardware loop out of the time vector reads as "the loops
around it order everything else", and on a device they order nothing between
two work items: ``jacobi`` with ``i`` on ``g.0`` reads at step ``t + 1`` what
the neighbouring group wrote at step ``t``, with no barrier between them.

So the second question has a second half, asked when the order passes: no
dependence may join instances on two work items. An instance's work item
along an axis is the value of its loop on that axis, counted from where loopy
starts that loop (see :func:`_work_items`). A statement with no loop on an
axis that a loop of another statement is on runs on every work item of it, as
loopy would run it, so every dependence to or from it joins two work items.
The loop of a sum is not a loop of its statement here: the whole sum happens
inside one instance. loopy runs a sum on a local axis as partial sums on every
work item of the axis, combined through local memory with barriers between
them, and then stores the result from one work item. So what the sum's body
reads is read on every work item of the axis, and what the statement's own
instruction reads and writes, on one, the same one for every statement whose
sum is on the axis. Local sums stay allowed, and so does a sum's result read
by another statement's instruction; read in the body of another sum, or of
the same sum at a later step, it is refused, as loopy refuses it for want of
a global barrier, and so is a sum that reads the cell its own statement
writes, which is one instance (see :func:`_within_instances`). ``ilp`` and
``vec`` loops run inside one work item, and only lose their order. A refusal
names the dependence and the two work items: ``S0[t=0, i=1] writes u[1, 1]
read by S0[t=1, i=2] on another work item``.

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
two local axes, a local axis whose extent has no numeric maximum), about
hardware axes (numbered from 0 with none left out, one loop of an instruction
per axis, every instruction on every axis the kernel uses), about loops loopy
writes out (``unr``, ``ilp`` and ``vec`` need a length that is a number when
the code is generated), about order (a loop put outside a loop loopy nests it
inside, which loopy cannot run in that order), and about the target itself
(the C target has no hardware axes). None is a wrong verdict about the cast,
and none used to be reported: the casts were all ``DECIDED`` and loopy then
threw during code generation, several steps away from the line that caused
it, or ran a nest of its own choosing.

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
every other schedule of every kernel: the kernel's definition, then the
target and every step up to the one the fact is about, with every argument it
was given (:meth:`Schedule.fact_id`). The definition is the module the
kernel's file's path gives it, its qualified name and its line, through
:func:`lanky.ledger.fact_id`, as every fact of the kernel itself is keyed, so
two kernels of one name, one defined in a file and one imported into it, keep
their schedules' facts apart. Two schedules of one kernel in one file keep
theirs apart too, since a ledger keeps one fact per id, while two that share
their first steps share the facts about those steps, which are the same
claims. Two kernels one definition makes, as a factory does each time it is
called, share every id, and their schedules' facts are two claims of each,
which ``loopty run`` refuses as ``lanky check`` does (see :mod:`loopty.cli`).
:attr:`Schedule.key` is the readable call text, which names the kernel by its
name.

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
it, but it enumerates the image's bounding loops and tests the parity with an
``if`` inside the innermost one, so half the iterations of that loop do
nothing. The kernel code is generated from therefore counts the steps of such
a loop instead, ``b = 2*b_step - a`` (see :func:`_stepped` and
:attr:`Schedule.strides`), and meets only the points of the image. The answer
is recorded in ``docs/loopy-notes.md``, note 13.

A map per statement
-------------------

Two statements that feed each other in one loop nest often need to move
differently: the coupled acoustic pair of ``examples/wavefront_acoustic.py``
tiles in diamonds only once its second statement sits half a step after the
first. The checker needs nothing new for that, because an instance already
carries its statement: :func:`_step_map` takes a map per statement, and the
two questions are asked of the maps together, over every dependence between
the statements and within each. The kernel does: loopy gives a loop one
domain, and the statements of a loop have to keep sharing it to keep
interleaving. So their loops run over the union of the images, and each
statement is predicated on its own image and uses its own inverse (see
:func:`_affine_kernel`); for the acoustic pair the images are the points
where ``a + b`` is even and those where it is odd, and every point of the
loops is one statement's.

Fusion
------

The maps per statement need not take the same loops. Two statements in two
loops one after the other, as a program's two calls are (:mod:`loopty.compose`),
take each its own loop to one new loop, ``{ flux_S0[j] -> [j];
divergence_S0[i] -> [j] : j = i + 1 }``, and that is a fusion: the checker
asks its two questions as for any map per statement, the time map puts the
statements in the one loop in the order the term has them, and a dependence
from the producer to a consumer that would now run first is refused with the
pair. :meth:`Schedule.fuse` builds that map from the two statements' loops and
a shift. In the kernel each statement's loops were a domain of their own;
they go, and the new loops get one domain, the union of the images, each
statement predicated on its own (:func:`_fused_plan`). The instructions'
dependencies, which the lowering drew by array in the order of the term, are
drawn again from the dependences in the new order
(:meth:`Schedule._ordered_by_dependences`).

Storing less
------------

An array a program makes is a temporary of its kernel, stored in full from
the call that writes it to the calls that read it. :meth:`Schedule.substitute`
stores none of it: a pointwise producer becomes a substitution rule (loopy's
``assignment_to_subst``), computed again at every read. That reorders no
instance and is not a cast; it changes what the instances read, and is legal
when each read is of a cell the producer stored before it (the
``definedness`` fact) and the producer's own reads would see the same cells
where the value is now computed. The second is a question about dependences:
those of the producer's reads are carried over to the reads that replace them,
those of the statements that no longer run are dropped, and the schedule's
order has to run the result forward. Every later step is checked against
that, the dependences of the program as it then runs, so a loop that carried
the array from one step to the next may go on a hardware axis once the array
is computed where it is read. Contraction, keeping only the cells live at
once, is not done.
"""

from __future__ import annotations

import dataclasses
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any

import islpy as isl
import loopy as lp
import pymbolic.primitives as prim
from lanky.ledger import fact_id
from pymbolic.mapper.substitutor import substitute

from loopty import idx
from loopty import oracle as isl_oracle
from loopty.flow import bounds_dimension
from loopty.lower import (
    Lowering,
    _plain,
    _reduction_nesting,
    _sanitize,
    is_reserved,
    lower_generic,
    reductions_of,
)
from loopty.term import Stmt, Term
from loopty.typing import layout_fact_ids

__all__ = [
    "IllegalCast",
    "Schedule",
    "UnbuildableSchedule",
    "definition_of",
    "parallel_tag",
]

#: Iname tags that impose no order: two instances differing only in such an
#: iname may run in either order, or at the same time. ``ilp`` and ``vec`` are
#: among them although neither is launched in parallel: loopy runs such a loop
#: around each instruction of its body separately (unrolled, or in vectors),
#: so two statements of the loop no longer interleave as the source wrote.
#: ``g.*`` and ``l.*`` are launched in parallel, and the other loops do not
#: order their instances either; :func:`_work_items` is the rest of that.
PARALLEL_TAG_PREFIXES = ("g.", "l.", "ilp", "vec")


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
    ``witness`` this witness when isl gave one. A substitution refused on a
    question isl could not answer carries it ``ASSUMED`` instead (see
    :meth:`Schedule.substitute`).
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
    """Each statement's ordered loops, with the unordered ones left out.

    A loop on a hardware axis is left out as ``ilp`` and ``vec`` are, which
    is right only for two instances on one work item; that a dependence joins
    no two work items is asked separately (see :func:`_work_items`).
    """
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


def _taken_loops(pieces: Iterable[isl.Map], order: Sequence[str]) -> tuple[str, ...]:
    """Every loop some map of ``pieces`` takes, each once, in loop order.

    Maps per statement that fuse loops take different loops (see
    :meth:`Schedule.fuse`), and the step replaces all of them; a loop that is
    not in ``order`` comes after those that are, as the maps name it, to be
    refused as no loop of the kernel.
    """
    names: list[str] = []
    for piece in pieces:
        for name in _dim_names(piece, isl.dim_type.in_):
            if name not in names:
                names.append(name)
    rank = {name: k for k, name in enumerate(order)}
    return tuple(
        sorted(names, key=lambda name: (rank.get(name, len(rank)), names.index(name)))
    )


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


def _as_maps(mapping: Any) -> tuple[Any, dict[str, isl.Map] | None]:
    """What ``affine`` was given: one map for every statement, or one each.

    A map, a basic map, a union map, or the text of any of them. A map whose
    input tuple is unnamed moves every statement in its loops; maps whose
    input tuples name statements, ``S0[t, i] -> [a, b]``, move each statement
    by its own (see :meth:`Schedule.affine`). Returns what the step records,
    an isl map or union map, and for maps per statement each statement's map
    with its tuple name taken off, since the checker and the kernel rewrite
    match loops by name and not by tuple; ``None`` for one map.
    """
    if isinstance(mapping, str):
        try:
            mapping = isl.UnionMap(mapping)
        except isl.Error as exc:
            raise ValueError(f"{mapping!r} is not an isl map: {exc}") from exc
    if isinstance(mapping, isl.BasicMap):
        mapping = isl.Map.from_basic_map(mapping)
    if isinstance(mapping, isl.Map):
        mapping = isl.UnionMap.from_map(mapping)
    if not isinstance(mapping, isl.UnionMap):
        raise TypeError(
            "affine() takes an isl map, a union of maps per statement, or the "
            f"text of either, not {type(mapping).__name__}"
        )
    maps: list[isl.Map] = []
    mapping.foreach_map(maps.append)
    if not maps:
        raise ValueError(f"{mapping} names no loop to replace")
    for piece in maps:
        if piece.has_tuple_name(isl.dim_type.out):
            raise ValueError(
                f"{mapping}: the output tuple {piece.get_tuple_name(isl.dim_type.out)}"
                " is named, and a map makes loops, which its output dimensions "
                "name; leave the tuple unnamed"
            )
    named = [piece.has_tuple_name(isl.dim_type.in_) for piece in maps]
    if not any(named):
        if len(maps) != 1:
            raise ValueError(
                f"{mapping}: these maps take or make different numbers of "
                "loops, and one map moves every statement in its loops; give "
                "one map, or name the statement of each"
            )
        return maps[0], None
    if not all(named):
        raise ValueError(
            f"{mapping}: some maps name a statement and some do not; name the "
            "statement of every map, or give one map for every statement"
        )
    pieces: dict[str, isl.Map] = {}
    for piece in maps:
        stmt_id = piece.get_tuple_name(isl.dim_type.in_)
        if stmt_id in pieces:
            # isl keeps maps between different spaces apart, so two maps of
            # one statement take or make different numbers of loops.
            raise ValueError(
                f"{mapping}: {stmt_id} is given two maps, and a statement "
                "moves by one"
            )
        pieces[stmt_id] = piece.reset_tuple_id(isl.dim_type.in_)
    return mapping, pieces


# }}}


# {{{ dependences


@dataclass(frozen=True)
class _Dep:
    """One dependence: which instances, through which cell of which array.

    ``source_part`` and ``sink_part`` say where in its statement each end's
    access is made (see :func:`_accesses`): ``""`` in the statement's own
    instruction, ``"sum"`` in the body of a sum, ``"row"`` where the length of
    a ragged row is read. Only the work items an access runs on depend on it
    (see :func:`_work_items`).
    """

    kind: str  # "raw", "war", "waw"
    array: str
    source: str  # statement id
    sink: str
    source_indices: tuple[Any, ...]
    sink_indices: tuple[Any, ...]
    relation: isl.Map
    source_part: str = ""
    sink_part: str = ""

    def verbs(self) -> tuple[str, str]:
        """How to say, in the rejection message, what each end did."""
        return {
            "raw": ("writes", "read"),
            "war": ("reads", "overwritten"),
            "waw": ("writes", "also written"),
        }[self.kind]


def _accesses(
    stmt: Stmt, term: Term
) -> list[tuple[str, str, tuple[Any, ...], str]]:
    """Every array reference of a statement, as ``(kind, array, indices, part)``.

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

    ``part`` is read off the loops the collector says an access is made in:
    ``""`` for the statement's own loops, which is its own instruction,
    ``"sum"`` for more, the body of a sum, and ``"row"`` for fewer, the length
    of a ragged row read where its loop starts. On a device these are
    different instructions, which can run on different work items (see
    :func:`_work_items`). The length of a row a sum runs over is read in the
    statement's loops, and is counted with its own instruction; that is
    wrong only where loopy generates no code: with that sum on a local axis
    (a concurrent loop in a ragged fiber), or another of the statement's sums
    there (the instruction that reads the length runs in no loop on the axis).
    """
    from loopty.flow import statement_accesses

    own = len(stmt.inames)
    out: list[tuple[str, str, tuple[Any, ...], str]] = []
    for array, indices, kind, inames, _domain in statement_accesses(stmt, term):
        part = "" if len(inames) == own else "sum" if len(inames) > own else "row"
        if kind == "acc":
            out.append(("write", array, tuple(indices), part))
            out.append(("read", array, tuple(indices), part))
        else:
            out.append((kind, array, tuple(indices), part))
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


def _same_cell(
    a: Stmt,
    b: Stmt,
    a_indices: Sequence[Any],
    b_indices: Sequence[Any],
    layout: _Layout,
    params: set[str],
) -> isl.Map:
    """The pairs of an instance of ``a`` and one of ``b`` that touch one cell.

    ``a_indices`` and ``b_indices`` are the two accesses' subscripts, compared
    axis by axis; an axis either one cannot state in isl is left
    unconstrained (see :func:`_index_text`), which can only add pairs.
    """
    source_dims = layout.dims("x")
    target_dims = layout.dims("y", suffix="_")
    renaming_a = {
        iname: prim.Variable(f"x{k}") for k, iname in enumerate(layout.coords[a.id])
    }
    allowed_a = {f"x{k}" for k in range(len(layout.coords[a.id]))} | params
    renaming_b = {
        iname: prim.Variable(f"y{k}") for k, iname in enumerate(layout.coords[b.id])
    }
    allowed_b = {f"y{k}" for k in range(len(layout.coords[b.id]))} | params
    constraints = [f"s = {layout.index(a.id)}", f"s_ = {layout.index(b.id)}"]
    for a_index, b_index in zip(a_indices, b_indices, strict=True):
        left = _index_text(a_index, renaming_a, allowed_a)
        right = _index_text(b_index, renaming_b, allowed_b)
        if left is None or right is None:
            continue
        constraints.append(f"{left} = {right}")
    return isl.Map(
        f"{{ [{source_dims}] -> [{target_dims}] : {' and '.join(constraints)} }}"
    )


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
    out: list[_Dep] = []
    for a in term.stmts:
        for b in term.stmts:
            for a_kind, a_array, a_indices, a_part in _accesses(a, term):
                for b_kind, b_array, b_indices, b_part in _accesses(b, term):
                    if a_array != b_array:
                        continue
                    if a_kind == "read" and b_kind == "read":
                        continue
                    if len(a_indices) != len(b_indices):
                        continue
                    relation = (
                        _same_cell(a, b, a_indices, b_indices, layout, params)
                        .intersect_domain(instances)
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
                            source_part=a_part,
                            sink_part=b_part,
                        )
                    )
    return tuple(out)


def _within_instances(
    term: Term, layout: _Layout, instances: isl.Set, params: set[str]
) -> tuple[_Dep, ...]:
    """The cells a statement's sums read that its own instruction then writes.

    One instance reads and writes such a cell, so the pair is no dependence
    of :func:`_dependences`, which are between two instances, and needs no
    order. On a device it can be two work items: a sum on a local axis reads
    on every work item of it, and the statement's instruction writes on one
    (see :func:`_work_items`), which loopy asks a global barrier between.
    Each is a ``war`` from an instance to itself, and names the written cell
    at both ends, since the read's subscripts can name the sum's own loops.
    """
    identity = isl.Map.identity(instances.get_space().map_from_set())
    identity = identity.intersect_domain(instances)
    out: list[_Dep] = []
    for stmt in term.stmts:
        accesses = _accesses(stmt, term)
        for r_kind, r_array, r_indices, r_part in accesses:
            if r_kind != "read" or r_part != "sum":
                continue
            for w_kind, w_array, w_indices, w_part in accesses:
                if w_kind != "write" or w_part != "" or w_array != r_array:
                    continue
                if len(r_indices) != len(w_indices):
                    continue
                relation = _same_cell(
                    stmt, stmt, r_indices, w_indices, layout, params
                ).intersect(identity)
                if relation.is_empty():
                    continue
                out.append(
                    _Dep(
                        kind="war",
                        array=r_array,
                        source=stmt.id,
                        sink=stmt.id,
                        source_indices=w_indices,
                        sink_indices=w_indices,
                        relation=relation.coalesce(),
                        source_part="sum",
                        sink_part="",
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


# {{{ work items


def _grid_axis(tag: Any, name: str) -> str | None:
    """The group or local axis a tag puts the loop ``name`` on, or ``None``.

    Read as loopy reads the tag, and named as loopy prints it: ``g.0``,
    ``l.1``. ``l.auto``, whose axis loopy has not chosen, is taken to be an
    axis of the loop's own, which can only refuse more; loopy refuses the
    tag when it generates code in any case. ``vec`` is not one: a vectorized
    loop runs in one work item.
    """
    from loopy.kernel.data import AutoLocalInameTagBase, GroupInameTag, LocalInameTag

    parsed = _loopy_tag(tag)
    if isinstance(parsed, GroupInameTag | LocalInameTag):
        return str(parsed)
    if isinstance(parsed, AutoLocalInameTagBase):
        return f"{parsed} ({name})"
    return None


def _grid_base(kernel: Any, name: str) -> tuple[str, tuple[str, ...]] | None:
    """Where loopy starts counting the work items of the loop ``name``.

    loopy runs a loop on a hardware axis as ``name = base + index``, where
    ``index`` is the work item's along the axis and ``base`` the static
    minimum of the loop's lower bound over the sizes
    (``loopy.kernel.tools.get_hw_axis_base_for_codegen``, which loopy's code
    generation and its own race check both use). Returns ``base`` as isl
    text with the parameters it names, or ``None`` when there is no kernel to
    ask or loopy finds no such minimum.
    """
    if kernel is None:
        return None
    from loopy.kernel.tools import get_hw_axis_base_for_codegen
    from loopy.symbolic import aff_to_expr

    try:
        base = get_hw_axis_base_for_codegen(kernel.default_entrypoint, name)
        text = idx.isl_expr(aff_to_expr(base))
    except Exception:  # noqa: BLE001 - a start loopy cannot say is not known
        return None
    space = base.get_space()
    params = tuple(
        space.get_dim_name(isl.dim_type.param, k)
        for k in range(space.dim(isl.dim_type.param))
    )
    return text, params


@dataclass(frozen=True)
class _WorkItems:
    """Where on the launch grid each statement instance runs.

    ``axes`` are the group and local axes a loop of some statement or of
    some sum is on, by name, and ``loops[stmt_id][axis]`` is that
    statement's loop on the axis, or ``None`` when it has no loop there.
    ``summed[stmt_id]`` are the axes a sum of the statement is on and no
    loop of it is.

    ``coordinates[axis]`` maps an instance of the schedule's layout to the
    work item along the axis its own instruction runs on, and
    ``spread[axis]`` to the work items every part of it runs on: the
    instruction, its sums and the reads of its rows' lengths. The two differ
    only for a statement a sum of which is on the axis and no loop: its own
    instruction runs on one work item of the axis once the sum is done, and
    its sum on all of them. ``known`` names the loops whose start loopy says,
    and ``hidden`` the parameters that stand for the start of each of the
    others, and for the work item of each axis where a sum's statement runs.
    """

    axes: tuple[str, ...]
    loops: dict[str, dict[str, str | None]]
    summed: dict[str, frozenset[str]]
    coordinates: dict[str, isl.Map]
    spread: dict[str, isl.Map]
    known: frozenset[str]
    hidden: frozenset[str]

    def placed(self, part: str, axis: str) -> isl.Map:
        """The work items along ``axis`` of an access made in ``part``.

        ``part`` as :func:`_accesses` says it: ``""`` for the statement's own
        instruction, which runs where :attr:`coordinates` puts it, and
        anything else for a sum's body or a row's length, which run where
        :attr:`spread` does.
        """
        return (self.coordinates if part == "" else self.spread)[axis]


def _work_items(
    layout: _Layout,
    tags: Mapping[str, str],
    kernel: Any,
    taken: set[str],
    sums: Mapping[str, Sequence[str]] | None = None,
) -> _WorkItems | None:
    """Each instance's work item along every hardware axis, or ``None``.

    ``sums[stmt_id]`` are the loops of the statement's sums, by the names
    they have now. ``None`` when no loop of a statement or of a sum is on a
    group or a local axis, which leaves nothing to ask.

    An instance's work item along an axis is ``x - base``: ``x`` the value of
    its statement's loop on the axis and ``base`` where loopy starts counting
    that loop's work items (see :func:`_grid_base`). Two loops on one axis
    can start at different values, and then ``i = 3`` in ``1 <= i < n`` and
    ``k = 3`` in ``0 <= k < n`` are two work items while ``i = 4`` and ``k =
    3`` are one. Without a kernel to read the start off (see
    :attr:`Schedule.kernel`), or when loopy finds none, the start is a
    parameter of its own, not one of ``taken``, so that two instances are
    found to share a work item only when they are of one loop and agree on
    it.

    A statement with neither a loop nor a sum on the axis runs on every work
    item of it, and so is taken to be at any work item at all; so is one with
    two loops on the axis, which loopy refuses to generate code for. Every
    dependence to or from either joins two work items.

    A sum's loop is not one of its statement's: the sum happens inside one
    instance, and loopy runs a sum on a local axis as partial sums on every
    work item of the axis, combined through local memory with barriers
    between them. Then, on one work item, the statement's own instruction
    stores the result (``if (lid(0) == 0)`` in the code loopy generates). So
    what the sum's body reads it reads on every work item, and what the
    instruction reads and writes it reads and writes on one, the same one
    for every statement whose sum is on the axis, which is a parameter of its
    own here; the two sums of a row that the row's statements pass a value
    between therefore stay allowed, and a sum that reads what another
    statement's instruction wrote is refused.
    """
    sums = sums or {}
    on: dict[str, dict[str, list[str]]] = {}
    summed: dict[str, frozenset[str]] = {}
    for stmt_id in layout.stmt_ids:
        mine: dict[str, list[str]] = {}
        for name in layout.coords[stmt_id]:
            axis = _grid_axis(tags[name], name) if name in tags else None
            if axis is not None:
                mine.setdefault(axis, []).append(name)
        on[stmt_id] = mine
        summed_on: set[str] = set()
        for name in sums.get(stmt_id, ()):
            axis = _grid_axis(tags[name], name) if name in tags else None
            if axis is not None and axis not in mine:
                summed_on.add(axis)
        summed[stmt_id] = frozenset(summed_on)
    axes = tuple(
        sorted(
            {axis for mine in on.values() for axis in mine}
            | {axis for theirs in summed.values() for axis in theirs}
        )
    )
    if not axes:
        return None

    hidden: set[str] = set()

    def fresh(name: str) -> str:
        while name in taken or name in hidden:
            name += "_"
        hidden.add(name)
        return name

    starts: dict[str, tuple[str, tuple[str, ...]]] = {}
    known: set[str] = set()
    for name in sorted({n for mine in on.values() for ns in mine.values() for n in ns}):
        start = _grid_base(kernel, name)
        if start is None:
            param = fresh(f"start_{name}")
            start = (param, (param,))
        else:
            known.add(name)
        starts[name] = start
    results = {
        axis: fresh("result_" + "".join(c if c.isalnum() else "_" for c in axis))
        for axis in axes
        if any(axis in mine for mine in summed.values())
    }

    # Primed names, which no size or loop of a kernel can have, since a
    # Python identifier cannot hold a prime and isl's syntax can.
    source = ", ".join(["s'", *(f"x{k}'" for k in range(layout.width))])

    def coordinate(axis: str, after_sums: bool) -> isl.Map:
        out: isl.Map | None = None
        for stmt_id in layout.stmt_ids:
            loops = on[stmt_id].get(axis, [])
            constraints = [f"s' = {layout.index(stmt_id)}"]
            params: tuple[str, ...] = ()
            if len(loops) == 1:
                base, params = starts[loops[0]]
                position = layout.coords[stmt_id].index(loops[0])
                constraints.append(f"h' = x{position}' - ({base})")
            elif after_sums and not loops and axis in summed[stmt_id]:
                params = (results[axis],)
                constraints.append(f"h' = {results[axis]}")
            prefix = f"[{', '.join(params)}] -> " if params else ""
            piece = isl.Map(
                f"{prefix}{{ [{source}] -> [h'] : {' and '.join(constraints)} }}"
            )
            out = piece if out is None else out.union(piece)
        assert out is not None
        return out.coalesce()

    loops_on: dict[str, dict[str, str | None]] = {}
    for stmt_id in layout.stmt_ids:
        loops_on[stmt_id] = {}
        for axis in axes:
            loops = on[stmt_id].get(axis, [])
            loops_on[stmt_id][axis] = loops[0] if len(loops) == 1 else None
    return _WorkItems(
        axes=axes,
        loops=loops_on,
        summed=summed,
        coordinates={axis: coordinate(axis, after_sums=True) for axis in axes},
        spread={axis: coordinate(axis, after_sums=False) for axis in axes},
        known=frozenset(known),
        hidden=frozenset(hidden),
    )


#: Two work items of one axis: two values of its coordinate.
_OTHER_WORK_ITEM = "{ [h] -> [g] : g < h or g > h }"


def _apart(
    reindex: isl.Map, source: isl.Map, sink: isl.Map | None = None
) -> isl.Map:
    """The pairs of the term's instances a schedule puts on two work items.

    ``reindex`` takes the term's instances to the schedule's, and ``source``
    and ``sink`` take those to the work items along one axis that the two
    ends of a pair run on (see :func:`_work_items`), ``sink`` the same as
    ``source`` when it is not given; a pair is apart when some work item of
    the first differs from some work item of the second, which for an end
    that runs on every work item is always.
    """
    first = reindex.apply_range(source)
    second = first if sink is None else reindex.apply_range(sink)
    return first.apply_range(isl.Map(_OTHER_WORK_ITEM)).apply_range(
        second.reverse()
    )


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


def _loopy_tag(tag: Any) -> Any:
    """loopy's own reading of a tag, whether it is given as text or not."""
    from loopy.kernel.data import parse_tag

    return parse_tag(tag)


def _hardware_axis(tag: Any) -> bool:
    """Is ``tag`` a group or a local axis (``g.*``, ``l.*``), as loopy reads it?

    Not ``vec``, which loopy counts among its hardware tags too: a vectorized
    loop runs in one work item, and neither the C target's lack of axes nor
    the numbering of a grid is about it.
    """
    from loopy.kernel.data import GroupInameTag, LocalInameTagBase

    return isinstance(_loopy_tag(tag), GroupInameTag | LocalInameTagBase)


def _concurrent(tag: Any) -> bool:
    """Is ``tag`` one of loopy's concurrent tags: a hardware axis, ilp or vec?

    This is loopy's own class (``ConcurrentTag``), which its check for a
    loop whose extent is read out of an array asks. It names the tags
    :func:`parallel_tag` names by their text, read as loopy reads them.
    """
    from loopy.kernel.data import ConcurrentTag

    return isinstance(_loopy_tag(tag), ConcurrentTag)


def _shown_tag(draft: _Draft, name: str, tag: Any) -> str:
    """The tag on a loop as the schedule was given it (``ilp``, not ``ilp.unr``)."""
    return draft.tags.get(name, str(tag))


def _kernel_tags(draft: _Draft) -> dict[str, tuple[Any, ...]]:
    """The tags of every loop of the draft's kernel, as loopy holds them.

    Read off the kernel rather than off :attr:`_Draft.tags`, so that a loop is
    asked about under the name it has now, after whatever renamed it.
    """
    entry = draft.kernel.default_entrypoint
    return {
        name: tuple(iname.tags)
        for name, iname in sorted(entry.inames.items())
        if iname.tags
    }


def _statement_of(draft: _Draft, insn_id: str) -> str | None:
    """The term statement an instruction of the kernel is, if it is one.

    The others assign a ragged row's length (:func:`loopty.lower._count_inits`),
    which the term has no statement for.
    """
    statements = {_sanitize(stmt_id): stmt_id for stmt_id in draft.coords}
    return statements.get(insn_id)


def _instruction_text(draft: _Draft, insn_id: str) -> str:
    """An instruction of the kernel, in the words of the term."""
    statement = _statement_of(draft, insn_id)
    if statement is not None:
        return f"statement {statement}"
    return f"the instruction {insn_id}, which reads the length of a ragged row,"


def _unbuildable_reason(draft: _Draft) -> str | None:
    """Why loopy could not generate code for this draft, or ``None``.

    Limits of loopy 2025.2, all measured rather than guessed, in the order
    they are asked:

    * a concurrent tag (a hardware axis, ``ilp`` or ``vec``) inside a ragged
      fiber, or on a loop loopy defines in one domain with a fiber's length:
      a device run of the design's spmv schedule fails on it (see
      :func:`_ragged_reason`);
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
    * loopy's rules for sharing and numbering hardware axes: an axis loopy is
      asked to choose (``l.auto``), two loops of one instruction on one axis,
      an instruction that runs on fewer axes than the kernel uses, and an axis
      numbered past an unused one (see :func:`_axis_reason`);
    * an unrolled or vectorized loop whose length is not a number when the
      code is generated (see :func:`_unroll_reason`), and a temporary loopy
      misreads once it has given an ``ilp`` or ``vec`` loop a copy of it per
      iteration (see :func:`_privatized_reason`);
    * a loop ordered outside a loop loopy nests it inside, which loopy cannot
      run in that order (see :func:`_nest_reason`);
    * what the target itself cannot do: the C target has no hardware axes,
      and no vector types for a temporary (see :func:`_target_reason`).

    The target's own limit comes last, because it is the one limit that
    :meth:`Schedule.retarget` removes; every other is loopy's on every target,
    and is said first so that retargeting does not merely trade one refusal
    for the next.

    Notes 11 and 14 of ``docs/loopy-notes.md`` have the tables of what loopy
    says to each, and note 6 the loop orders.
    """
    reason = _ragged_reason(draft)
    if reason is not None:
        return reason
    hardware = {name for name, tag in draft.tags.items() if _hardware_axis(tag)}
    nested = sorted(
        name
        for name in hardware
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
    for check in (
        _axis_reason,
        _unroll_reason,
        _privatized_reason,
        _nest_reason,
        _target_reason,
    ):
        reason = check(draft)
        if reason is not None:
            return reason
    return None


def _ragged_reason(draft: _Draft) -> str | None:
    """A concurrent loop in a domain whose extent is read out of an array.

    loopy refuses any concurrent loop (``ConcurrentTag``: a hardware axis,
    ``ilp`` or ``vec``) in a domain that names a temporary as a parameter
    (``check_for_data_dependent_parallel_bounds``), and a ragged row's length
    is such a temporary: it is assigned inside the row loop and bounds the
    fiber. Two loops are caught by that, and asked in turn. A ragged fiber
    itself, whose extent is the row's length: a hardware axis there cannot be
    launched, since the number of work items is not known when the kernel
    starts, and ``ilp`` and ``vec`` are refused alike. And a loop between the
    row and its fiber (``i`` in ``for r: for i in x.dom: for j in
    val.dom[r]``), which is not ragged at all and which the lowering defines in
    one domain with the fiber, ``[r, nl_cnt_r] -> { [i, j] }``: loopy reads
    its domain, not its extent, and refuses it the same way. The second is
    read off the kernel's domains, as loopy reads them.
    """
    concurrent = {name for name, tag in draft.tags.items() if _concurrent(tag)}
    inside = sorted(concurrent & draft.data_dependent)
    if inside:
        names = ", ".join(inside)
        if all(_hardware_axis(draft.tags[name]) for name in inside):
            return (
                f"the parallel tag on {names} sits inside a loop whose bound "
                "comes from an array (a ragged fiber), and loopy will not put "
                "a hardware axis in a domain with a data-dependent parameter. "
                "Parallelize an enclosing loop with a size known at launch "
                "instead, such as the rows of a CSR product"
            )
        tags = ", ".join(f"{name}={draft.tags[name]!r}" for name in inside)
        return (
            f"the tag {tags} makes a loop inside a ragged fiber concurrent, "
            "and loopy will not run a concurrent loop (a hardware axis, ilp or "
            "vec) in a domain with a data-dependent parameter, which the "
            "fiber's is: its bound comes from an array. Leave the fiber's loop "
            "sequential, and put an enclosing loop with a size known at launch "
            "on a hardware axis instead, such as the rows of a CSR product"
        )
    if draft.kernel is None:
        return None
    entry = draft.kernel.default_entrypoint
    temporaries = set(entry.temporary_variables)
    tags = _kernel_tags(draft)
    for domain in entry.domains:
        if not set(domain.get_var_names(isl.dim_type.param)) & temporaries:
            continue
        loops = [
            name
            for name in domain.get_var_names(isl.dim_type.set)
            if any(_concurrent(tag) for tag in tags.get(name, ()))
        ]
        if not loops:
            continue
        name = loops[0]
        fibers = [
            other
            for other in domain.get_var_names(isl.dim_type.set)
            if other != name and other in draft.data_dependent
        ]
        beside = f", beside the ragged fiber {fibers[0]}" if fibers else ""
        shown = _shown_tag(
            draft, name, next(tag for tag in tags[name] if _concurrent(tag))
        )
        return (
            f"the tag {name}={shown!r} makes the loop {name} concurrent, and "
            f"loopy defines {name} in one domain with the length of a ragged "
            f"row{beside}, which is read out of an array inside the row's "
            "loop; loopy will not run a concurrent loop (a hardware axis, ilp "
            "or vec) in a domain with a data-dependent parameter. Put the axis "
            f"on the row's loop or one outside it instead, or leave {name} "
            "sequential"
        )
    return None


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
    tagged so, and ``vec`` too.
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


def _axis_reason(draft: _Draft) -> str | None:
    """A hardware axis loopy will not assign as the schedule asks, or ``None``.

    loopy has four rules about the axes of a kernel that the casts cannot
    see, asked here in the order loopy meets them, each off the kernel's own
    instructions and tags:

    * ``l.auto`` asks loopy to choose a local axis, which it does only inside
      its own transforms (``precompute``, ``buffer_array``); a kernel that
      still has one is refused when loopy prepares it for code generation;
    * one loop of an instruction per axis (``check_for_double_use_of_hw_axes``),
      and ``vec`` counts as an axis there: two loops of one statement on
      ``l.0`` are refused, and the loop of a reduction is one of its
      statement's, because the sum runs inside the statement's loops (the
      instruction loopy names is then the accumulator's initialization);
    * every instruction runs on every group and local axis the kernel uses
      (``check_for_unused_hw_axes``): a statement beside a loop on ``g.0``
      that runs in no loop on ``g.0`` is refused, and so is a sum whose
      accumulator is set up and updated outside the loop on that axis, as a
      sum beside another on a local axis is. loopy names a remedy,
      ``add_inames_for_unused_hw_axes``, which runs the instruction once per
      work item, all of them writing one cell; loopty does not apply it;
    * the axes of each kind are numbered from 0 up with none left out, over
      the whole kernel (``get_grid_sizes_for_insn_ids``): ``l.1`` needs a
      loop on ``l.0``.

    Which loops an instruction of a reduction runs in follows
    ``loopy.transform.realize_reduction``: a sum's own instructions run in the
    statement's loops, its own, and those of the sums around it; the
    statement's instruction runs in its loops and on the axis of every sum in
    it that is summed on a local axis, since that sum's result is read by
    every work item of the group.
    """
    if draft.kernel is None:
        return None
    from loopy.kernel.data import (
        AutoLocalInameTagBase,
        GroupInameTag,
        LocalInameTag,
        UniqueInameTag,
        VectorizeTag,
    )

    entry = draft.kernel.default_entrypoint
    tags = _kernel_tags(draft)

    auto = sorted(
        name
        for name, found in tags.items()
        if any(isinstance(tag, AutoLocalInameTagBase) for tag in found)
    )
    if auto:
        return (
            f"the tag l.auto on {', '.join(auto)} asks loopy to choose a local "
            "axis, and loopy chooses one only inside its own transforms "
            "(precompute, buffer_array); a kernel that still has one is "
            "refused when loopy prepares it for code generation. Name the "
            "axis, as l.0 does"
        )

    order = {name: k for k, name in enumerate(draft.order)}

    def nest_order(name: str) -> tuple[int, str]:
        return (order.get(name, len(order)), name)

    for insn in entry.instructions:
        summed = set(insn.reduction_inames())
        loops = sorted(set(insn.within_inames) | summed, key=nest_order)
        seen: dict[Any, str] = {}
        for name in loops:
            for tag in tags.get(name, ()):
                if not isinstance(tag, UniqueInameTag):
                    continue
                first = seen.setdefault(tag.key, name)
                if first == name:
                    continue
                shown = _shown_tag(draft, name, tag)
                what = _instruction_text(draft, insn.id)
                rule = (
                    "loopy vectorizes one loop of an instruction at most"
                    if isinstance(tag, VectorizeTag)
                    else "loopy runs one loop of an instruction on each hardware "
                    "axis"
                )
                why = (
                    "; the loop of a sum is one of its statement's, because "
                    "the sum runs inside the statement's loops"
                    if {first, name} & summed
                    else ""
                )
                return (
                    f"{what} runs in two loops tagged {shown}, {first} and "
                    f"{name}, and {rule}{why}. Put one of the two on another "
                    "axis, or leave it sequential"
                )

    grid_tags = (GroupInameTag, LocalInameTag)

    def axes(names: Iterable[str]) -> set[Any]:
        return {
            tag.key
            for name in names
            for tag in tags.get(name, ())
            if isinstance(tag, grid_tags)
        }

    grid: dict[Any, tuple[Any, str]] = {}
    for name in sorted(tags, key=nest_order):
        for tag in tags[name]:
            if isinstance(tag, grid_tags):
                grid.setdefault(tag.key, (tag, name))
    for insn in entry.instructions:
        base = set(insn.within_inames)
        sums: dict[str, list[str]] = {}
        for name in sorted(insn.reduction_inames(), key=nest_order):
            sums.setdefault(draft.reductions.get(name, name), []).append(name)
        what = _instruction_text(draft, insn.id)
        local = {
            name
            for names in sums.values()
            for name in names
            if any(isinstance(tag, LocalInameTag) for tag in tags.get(name, ()))
        }
        runs = [(what, None, base | local)]
        for key, names in sums.items():
            around: list[str] = []
            outer = draft.nested_in.get(key)
            while outer is not None:
                around.extend(sums.get(outer, ()))
                outer = draft.nested_in.get(outer)
            runs.append(
                (
                    f"the sum over {', '.join(names)} in {what}",
                    names,
                    base | set(names) | set(around),
                )
            )
        for subject, over, loops in runs:
            missing = [key for key in grid if key not in axes(loops)]
            if not missing:
                continue
            tag, loop = grid[missing[0]]
            kind = "group" if isinstance(tag, GroupInameTag) else "local"
            shown = _shown_tag(draft, loop, tag)
            why = (
                "loopy sets up and updates a sum's accumulator in instructions "
                "of their own, which run in the loops of the statement and of "
                "the sums around it, and it generates code only when every "
                "instruction of a kernel runs on every hardware axis the kernel "
                "uses"
                if over
                else "loopy generates code only when every instruction of a "
                "kernel runs on every hardware axis the kernel uses"
            )
            statement = _statement_of(draft, insn.id)
            remedy = (
                f"put a loop of statement {statement} on {shown} as well"
                if statement is not None
                else "put the axis on the row's loop or one outside it"
            )
            return (
                f"{subject} runs in no loop on {shown}, the {kind} axis {loop} "
                f"is on: {why}. Leave {loop} sequential, or {remedy}"
            )

    for kind, cls, letter in (
        ("group", GroupInameTag, "g"),
        ("local", LocalInameTag, "l"),
    ):
        numbers: dict[int, str] = {}
        for name in sorted(tags, key=nest_order):
            for tag in tags[name]:
                if isinstance(tag, cls):
                    numbers.setdefault(tag.axis, name)
        for axis in range(max(numbers, default=-1) + 1):
            if axis in numbers:
                continue
            above = min(number for number in numbers if number > axis)
            return (
                f"the loop {numbers[above]} is on the {kind} axis "
                f"{letter}.{above}, and no loop of the kernel is on "
                f"{letter}.{axis}: loopy numbers the {kind} axes of a kernel "
                "from 0 up, with none left out. Use "
                f"{letter}.{axis} first"
            )
    return None


def _has_numeric_length(kernel: Any, iname: str) -> bool:
    """Does ``iname`` run a number of times known when the code is generated?

    Asked as loopy asks it before it unrolls or vectorizes a loop
    (``generate_unroll_loop`` and ``generate_vectorize_loop`` in
    ``loopy.codegen.loop``): of the loop's bounds with every size and every
    other loop projected out, which isl cannot bound when a size is free
    (``unbounded optimum``, raised from inside code generation, naming no
    loop). A triangle ``j < i + 1`` inside ``i < 8`` has one: at most 8.
    """
    from loopy.diagnostic import StaticValueFindingError
    from loopy.isl_helpers import static_max_of_pw_aff

    entry = kernel.default_entrypoint
    try:
        size = entry.get_iname_bounds(iname, constants_only=True).size
        static_max_of_pw_aff(size, constants_only=True)
    except (isl.Error, StaticValueFindingError):
        return False
    return True


def _extent_text(kernel: Any, iname: str) -> str:
    """The extent of ``iname`` in words, as its bounds give it."""
    from loopy.symbolic import pw_aff_to_expr

    size = kernel.default_entrypoint.get_iname_bounds(iname).size
    try:
        return f"at most {pw_aff_to_expr(size)}"
    except Exception:  # noqa: BLE001 - a piecewise extent is shown as isl has it
        return f"at most {size}"


def _unroll_reason(draft: _Draft) -> str | None:
    """An unrolled or vectorized loop without a numeric length, or ``None``.

    loopy writes out the body of a loop tagged ``unr`` or ``ilp`` once per
    iteration, and vectorizes a loop tagged ``vec`` into vectors of a fixed
    length, or unrolls it where it cannot; each needs the number of
    iterations as a number when the code is generated (see
    :func:`_has_numeric_length`). A loop over ``Fin[n]`` with ``n`` free has
    none, on any target and whether it is a statement's loop or a reduction's.
    Split by a fixed factor, the inner half has one, and builds.
    """
    if draft.kernel is None:
        return None
    from loopy.kernel.data import UnrolledIlpTag, UnrollTag, VectorizeTag

    for name, found in _kernel_tags(draft).items():
        unrolled = [
            tag
            for tag in found
            if isinstance(tag, UnrollTag | UnrolledIlpTag | VectorizeTag)
        ]
        if not unrolled or _has_numeric_length(draft.kernel, name):
            continue
        shown = _shown_tag(draft, name, unrolled[0])
        how = (
            "vectorizes a loop tagged vec into vectors of a length fixed when "
            "the code is generated, or unrolls it where it cannot"
            if isinstance(unrolled[0], VectorizeTag)
            else f"unrolls a loop tagged {shown}, writing its body out once per "
            "iteration"
        )
        extent = (
            "read out of an array (a ragged fiber)"
            if name in draft.data_dependent
            else _extent_text(draft.kernel, name)
        )
        return (
            f"the loop {name} is tagged {shown}, and loopy {how}, so the number "
            f"of its iterations has to be a number when the code is generated; "
            f"the extent of {name} is {extent}, which no number bounds while "
            "the sizes it names are free. Split it by a fixed factor and tag "
            f"the inner loop instead, as split({name!r}, 4) makes one of "
            "length 4, or declare the extent as a number (Fin[8] rather than "
            "Fin[n])"
        )
    return None


def _privatized_reason(draft: _Draft) -> str | None:
    """A temporary loopy gives an ``ilp`` or ``vec`` loop and then misreads.

    loopy gives a temporary written inside a loop tagged ``ilp`` (or
    ``ilp.seq``) or ``vec`` a copy per iteration of the loop: an array along
    it, or for ``vec`` a vector (``privatize_temporaries_with_inames``, from
    ``realize_ilp``). A sum's accumulator survives that, and two temporaries
    do not, on any target:

    * the length of a ragged row read inside the loop: it bounds the loop
      over the row's fiber, and loopy's bound still reads the temporary by its
      name, as one number, once it has become an array or a vector. Under
      ``ilp`` loopy generates the code and gets it wrong: the fiber's loop
      compares its variable with the whole array (``j <= -1 + nl_cnt_r``
      after ``int32_t nl_cnt_r[8]``), and the C run reads past the row, or
      crashes. Under ``vec`` code generation fails from inside loopy (a
      ``TypeError`` or an ``AssertionError``, in 2025.2), or writes OpenCL
      that reads a vector as the bound and does not compile;
    * the partial sums of a reduction on a local axis inside a ``vec`` loop,
      which loopy keeps in an array in local memory (a ``TypeError``). An
      ``ilp`` loop around one builds.

    Both were measured on loops over ``Fin[8]``; over ``Fin[n]`` the loop has
    no length to unroll or vectorize by, which :func:`_unroll_reason` says
    first. A sequential sum in a ``vec`` loop builds on OpenCL; on C it is
    :func:`_target_reason`'s.
    """
    if draft.kernel is None:
        return None
    from loopy.kernel.data import IlpBaseTag, LocalInameTagBase, VectorizeTag

    entry = draft.kernel.default_entrypoint
    tags = _kernel_tags(draft)
    privatizing = {
        name: tag
        for name, found in tags.items()
        for tag in found
        if isinstance(tag, IlpBaseTag | VectorizeTag)
    }
    bounds = {
        name
        for domain in entry.domains
        for name in domain.get_var_names(isl.dim_type.param)
    } & set(entry.temporary_variables)
    for insn in entry.instructions:
        loops = sorted(set(privatizing) & set(insn.within_inames))
        if not loops:
            continue
        if set(insn.assignee_var_names()) & bounds:
            name = loops[0]
            shown = _shown_tag(draft, name, privatizing[name])
            if isinstance(privatizing[name], VectorizeTag):
                what = (
                    "loopy cannot generate working code for it: it keeps a "
                    "temporary written inside a vec loop as a vector along it, "
                    "and the row's length bounds the loop over its fiber, which "
                    "needs one number"
                )
            else:
                what = (
                    "the code loopy generates for it is wrong: it gives a "
                    f"temporary written inside a loop tagged {shown} an array "
                    "along the loop, one cell per iteration, and the loop over "
                    "the row's fiber still reads the row's length as one "
                    "number, so it compares its variable with the whole array"
                )
            return (
                f"the length of a ragged row is read inside the loop {name}, "
                f"which is tagged {shown}, and {what}. Leave {name} "
                "sequential, or unroll it (unr) instead"
            )
        vectorized = [
            name for name in loops if isinstance(privatizing[name], VectorizeTag)
        ]
        local = sorted(
            name
            for name in insn.reduction_inames()
            if any(isinstance(tag, LocalInameTagBase) for tag in tags.get(name, ()))
        )
        if vectorized and local:
            what = _instruction_text(draft, insn.id)
            return (
                f"the sum over {', '.join(local)} in {what} runs on a local "
                f"axis inside the loop {vectorized[0]}, which is tagged vec, "
                "and loopy cannot generate code for it: it keeps the partial "
                "sums of a reduction on a local axis in an array in local "
                "memory, and a temporary written inside a vec loop as a vector "
                f"along it. Leave {vectorized[0]} sequential, or unroll it "
                "(unr) instead"
            )
    return None


#: The targets whose code runs in one thread, with no hardware axes.
_C_TARGETS = ("c", "c-source")


def _target_reason(draft: _Draft) -> str | None:
    """What the target cannot do at all, or ``None``.

    The C targets have no hardware axes: loopy's C code runs in one thread,
    and code generation for a loop on ``g.*`` or ``l.*`` stops with "plain C
    does not have group hw axes" (or local). Nor do they have vector types:
    loopy keeps a temporary written inside a ``vec`` loop, the accumulator of
    a sum in the loop, as a vector along it, and its C code generator does not
    know how to declare one. A ``vec`` loop that writes no temporary is only
    unrolled, and builds. (The one other temporary, the length of a ragged
    row, cannot be a vector on any target; :func:`_privatized_reason` says
    so.)

    Asked last (see :func:`_unbuildable_reason`): it is the one limit that
    retargeting to OpenCL removes.
    """
    if draft.target not in _C_TARGETS or draft.kernel is None:
        return None
    from loopy.kernel.data import VectorizeTag

    tags = _kernel_tags(draft)
    hardware = [
        (name, tag)
        for name, found in tags.items()
        for tag in found
        if _hardware_axis(tag)
    ]
    if hardware:
        order = {name: k for k, name in enumerate(draft.order)}
        hardware.sort(key=lambda pair: (order.get(pair[0], len(order)), pair[0]))
        shown = ", ".join(
            f"{name}={_shown_tag(draft, name, tag)!r}" for name, tag in hardware
        )
        return (
            f"the tag {shown} puts a loop on a hardware axis, and the C target "
            "has none: loopy's C code runs in one thread, with no groups or "
            "work items to spread a loop over. Retarget to opencl, or leave the "
            "loop sequential"
        )
    vectorized = {
        name
        for name, found in tags.items()
        if any(isinstance(tag, VectorizeTag) for tag in found)
    }
    for insn in draft.kernel.default_entrypoint.instructions:
        loops = sorted(vectorized & set(insn.within_inames))
        if not loops or not insn.reduction_inames():
            continue
        what = _instruction_text(draft, insn.id)
        return (
            f"the loop {loops[0]} is tagged vec, and loopy keeps a temporary "
            f"written inside it, the accumulator of the sum in {what}, as a "
            "vector along it; the C target has no vector types to declare it "
            f"with. Leave {loops[0]} sequential, unroll it (unr), or retarget "
            "to opencl"
        )
    return None


# }}}


def definition_of(obj: Any, term: Term) -> dict[str, Any]:
    """What the facts about ``obj`` name it by: its owner, module and line.

    The keywords :func:`lanky.ledger.fact_id` takes. A kernel or a program
    gives its definition, the qualified name, the module its file's path
    gives it and its line, which is what its own facts are keyed by
    (:mod:`loopty.kernel`); a schedule gives the definition of what it
    schedules; a term gives its name alone, since nothing says where it was
    defined.
    """
    if isinstance(obj, Schedule):
        return dict(obj._definition)
    qualname = getattr(obj, "qualname", None)
    if isinstance(qualname, str) and qualname and not isinstance(obj, Term):
        line = getattr(obj, "line", None)
        return {
            "owner": qualname,
            "module": getattr(obj, "module", None) or None,
            "line": line if isinstance(line, int) else None,
        }
    return {"owner": term.name, "module": None, "line": None}


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
    #: The target the schedule is written for; what it cannot do at all is
    #: asked last (see :func:`_target_reason`).
    target: str = "c"


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
        *,
        _layouts: dict[str, str] | None = None,
    ) -> None:
        self._source = kernel
        self._term = _term_of(kernel)
        #: What the ids of the schedule's facts name the kernel by; see
        #: :meth:`fact_id`.
        self._definition = definition_of(kernel, self._term)
        #: The layout facts a ``monotone`` cast rests on: the dependences of a
        #: ragged array are computed over ``[r, j]``, which are its cells only
        #: while its layout keeps the rows apart, and a kernel that rewrites
        #: its counts or offsets is taken to (see :func:`loopty.typing.layout_facts`).
        self._layout_ids = tuple(
            dict.fromkeys(
                identifier
                for ids in layout_fact_ids(self._term, **self._definition).values()
                for identifier in ids
            )
        )
        self._target = target
        self._sizes = dict(sizes or {})
        #: The layout of each array over a domain that is not boxed; only
        #: :meth:`pack` sets it, so that the steps say how every array is kept.
        self._layouts = dict(_layouts or {})
        self._lowering: Lowering = lower_generic(self._term, target, self._layouts)
        self._kernel = self._lowering.kernel
        #: The kernel code is generated from: :attr:`_kernel`, which the steps
        #: transform, with its loops over a lattice counted (see
        #: :func:`_stepped`), and those loops' expressions.
        self._code = self._kernel
        self._strides: dict[str, str] = {}

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
        #: The source order on the term's instances, which a substitution
        #: reads the direction of each dependence it carries over from
        #: (see :meth:`substitute`).
        self._before = before
        #: The arrays :meth:`substitute` computed where they are read, and
        #: the statements it took out of the kernel, which have no instances
        #: and no loops in the schedule any more.
        self._substituted: tuple[str, ...] = ()
        self._gone: frozenset[str] = frozenset()
        self._deps = _dependences(
            self._term, self._layout, self._instances, before, params
        )
        #: Pairs that only work items can separate (see :func:`_work_items`).
        self._within = _within_instances(
            self._term, self._layout, self._instances, params
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
            target=self._target,
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
        """The loopy kernel as transformed so far, the one code is made from.

        A loop whose domain has holes, such as the image of the diamond,
        runs over a counter of its steps rather than over the loop the checker
        knows, once a step has set the nest; :attr:`strides` names each such
        loop, and every other loop of :attr:`order` is the kernel's loop of
        that name. Statements that :meth:`affine` moved by maps of their own
        share their loops, and each runs at the points of its own image only,
        which its instruction's predicates state.

        ``None`` once a step could not be written as a loopy kernel at all,
        which only :meth:`affine` can cause; the schedule's ``buildable`` fact
        says why, and :meth:`require_buildable` refuses before anything reads
        this.
        """
        return self._code

    @property
    def strides(self) -> dict[str, str]:
        """The loops :attr:`kernel` steps through, and how.

        Each loop of :attr:`order` whose values are a lattice given the loops
        outside it, with the expression of the counter it is written as:
        ``{"b": "2*b_step - a"}`` after the diamond ``(t, i) -> (t + i, t -
        i)``, whose image is the points where ``a + b`` is even. The kernel
        loops over ``b_step`` and meets only those points, where loopy alone
        would loop over ``b`` and test the parity at each. Empty when there is
        no such loop; see :func:`_stepped`.
        """
        return dict(self._strides)

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
        in. The facts are not named by this key, which names the kernel by
        its name, and two kernels can share a name; see :meth:`fact_id`.
        """
        return _key(self._term.name, self._target, self._steps)

    def fact_id(self, kind: str, detail: str = "") -> str:
        """The id of a fact of ``kind`` about this schedule.

        :func:`lanky.ledger.fact_id` over the kernel's definition, the module
        its file's path gives it, its qualified name and its line, with the
        target and the steps as :attr:`key` writes them, and ``detail``
        after them::

            agreement:spmv.spmv@102:[c].split('j', 2, inner='j_in', outer='j_out')

        That is the id of the agreement a run of the schedule records
        (:func:`loopty.executor.agreement`). A cast fact's is ``cast:`` and
        the steps up to the one it is about, then its kind. Keyed by the
        definition, as every fact of the kernel itself is, so that two
        kernels of one name scheduled in one file, one defined there and one
        imported, keep their facts apart (#75); a term scheduled with no
        kernel behind it is named by its name.
        """
        return self._fact_id(kind, self._steps, detail)

    def _fact_id(
        self, kind: str, steps: Sequence[tuple[str, tuple, dict]], detail: str = ""
    ) -> str:
        """:meth:`fact_id`, over the steps given rather than the schedule's."""
        text = _steps_text(self._target, steps)
        return fact_id(
            kind, **self._definition, detail=f"{text}:{detail}" if detail else text
        )

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
        producing a race at run time. A hardware axis (``g.*``, ``l.*``) is
        the launch grid besides, and nothing orders two of its work items, so
        a dependence between instances on two work items is rejected too,
        whichever loop carries it: ``i`` on ``g.0`` in a stencil whose step
        ``t + 1`` reads the neighbours of step ``t``, and every dependence to
        or from a statement with no loop on the axis, which runs on every
        work item of it (see "Work items" in the module docstring).

        A name that is neither a loop nor the loop of a reduction, and a tag
        loopy cannot read, are refused with a ``ValueError`` before anything
        else, as :meth:`split` refuses an unknown loop. loopy's own
        ``tag_inames`` refuses both too, but it is not asked once a step has
        left the schedule with no kernel (see :attr:`kernel`), and the steps
        after that one are still checked, so that their facts say what the
        schedule is.

        A tag goes on after the steps that replace its loop: :meth:`split`,
        :meth:`tile` and :meth:`affine` refuse a loop that carries one, and
        the loops they make are tagged for themselves.
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
        """Split ``iname`` by ``factor`` into an outer and an inner iname.

        A loop that carries a tag is refused with a ``ValueError``, a sum's
        loop as well, whether or not the schedule still has a kernel: split
        it before tagging the loops it makes (see :meth:`_refuse_tagged`).
        """
        inner = inner or f"{iname}_inner"
        outer = outer or f"{iname}_outer"
        self._refuse_tagged(
            (iname,),
            f"split({iname}, {factor})",
            "split it before tagging the loops it makes",
        )
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

    def _reindex_into(
        self,
        draft: _Draft,
        mapping: isl.Map,
        text: str,
        pieces: Mapping[str, isl.Map] | None = None,
    ) -> None:
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

        ``pieces``, when given, is a map per statement, by statement id, and
        ``mapping`` is one of them. The statements of a loop share its loops
        in the kernel before the step and after it, so maps that take a loop
        in common take the same loops to the same new ones, and every
        statement that runs in those loops has to run in all of them and be
        given a map; anything else is refused, naming the statement. Maps of
        statements in different loops may take those loops to the same new
        ones, which fuses them (see :meth:`fuse`); each statement then gets
        the new loops in place of its own.

        A loop whose extent comes from an array passes that property to each
        new loop whose value depends on it (see :func:`_inherited`): both
        halves of a split ragged loop, and neither half of the dense loop it
        is tiled with.
        """
        inputs = _dim_names(mapping, isl.dim_type.in_)
        outputs = _dim_names(mapping, isl.dim_type.out)
        if pieces is not None:
            inputs = _taken_loops(pieces.values(), draft.order)
            self._check_pieces(draft, pieces, inputs, outputs, text)
        if not inputs:
            raise ValueError(f"{text}: the map names no loop to replace")
        every = [mapping] if pieces is None else list(pieces.values())
        for piece in every:
            for side, kind in (
                ("input", isl.dim_type.in_),
                ("output", isl.dim_type.out),
            ):
                names = _dim_names(piece, kind)
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
            {
                name
                for piece in every
                for name in piece.get_var_names(isl.dim_type.param)
            }
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
            if not inside or stmt_id in self._gone:
                # A statement a substitution took out has no instances, and
                # keeps the loops it had, which are no loops of the kernel.
                continue
            own = mapping if pieces is None else pieces[stmt_id]
            if pieces is None and len(inside) != len(inputs):
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
            inherited = {name: set() for name in outputs}
            for piece in every:
                for name, olds in _inherited(piece).items():
                    inherited[name] |= olds
            draft.data_dependent -= set(inputs)
            draft.data_dependent |= {
                name for name in outputs if inherited[name] & dependent
            }
        draft.mappings = mappings

    def _check_pieces(
        self,
        draft: _Draft,
        pieces: Mapping[str, isl.Map],
        inputs: Sequence[str],
        outputs: Sequence[str],
        text: str,
    ) -> None:
        """Refuse maps per statement that the statements' loops cannot take.

        See :meth:`_reindex_into`: the maps name statements of the kernel and
        make the same new loops; two maps that take a loop in common take the
        same loops; and every statement that runs in a loop some map takes is
        given a map, which takes every such loop it runs in and only those.
        Two maps that take different loops to the same new ones fuse those
        loops (see :meth:`fuse`). ``inputs`` are every loop the maps take, and
        ``outputs`` the loops they make.
        """
        unknown = sorted(set(pieces) - set(draft.coords))
        if unknown:
            verb = "is not a statement" if len(unknown) == 1 else "are not statements"
            raise ValueError(
                f"{text}: {', '.join(unknown)} {verb} of {self._term.name}"
            )
        gone = sorted(set(pieces) & self._gone)
        if gone:
            raise ValueError(
                f"{text}: {', '.join(gone)} runs no more: a substitution took it "
                "out of the kernel"
            )
        first = next(iter(pieces))
        first_ins = _dim_names(pieces[first], isl.dim_type.in_)
        taken: dict[str, tuple[str, tuple[str, ...]]] = {}
        for stmt_id, piece in pieces.items():
            ins = _dim_names(piece, isl.dim_type.in_)
            outs = _dim_names(piece, isl.dim_type.out)
            shared = next((taken[name] for name in ins if name in taken), None)
            if outs != tuple(outputs) or (shared is not None and shared[1] != ins):
                other, other_ins = shared if shared is not None else (first, first_ins)
                other_outs = _dim_names(pieces[other], isl.dim_type.out)
                why = (
                    "the statements of one loop share their loops in the kernel, "
                    "so their maps have to take the same loops to the same new "
                    "ones"
                    if set(ins) & set(other_ins)
                    else "the statements the maps fuse share the loops the maps "
                    "make, so their maps have to make the same new ones"
                )
                raise ValueError(
                    f"{text}: the map of {stmt_id} takes [{', '.join(ins)}] to "
                    f"[{', '.join(outs)}] and the map of {other} takes "
                    f"[{', '.join(other_ins)}] to [{', '.join(other_outs)}]; {why}"
                )
            for name in ins:
                taken.setdefault(name, (stmt_id, ins))
        for stmt_id, coords in draft.coords.items():
            if stmt_id in self._gone:
                continue
            inside = [name for name in inputs if name in coords]
            piece = pieces.get(stmt_id)
            ins = () if piece is None else _dim_names(piece, isl.dim_type.in_)
            if not inside:
                if piece is not None:
                    raise ValueError(
                        f"{text}: {stmt_id} does not run in {', '.join(ins)}, so "
                        "its map has no loop of it to move"
                    )
                continue
            if piece is None:
                raise ValueError(
                    f"{text}: {stmt_id} runs in {', '.join(inside)} and is given "
                    "no map; with a map per statement, every statement in the "
                    "loops the maps name needs one"
                )
            outside = [name for name in ins if name not in coords]
            if outside:
                mine = [name for name in ins if name in coords]
                raise ValueError(
                    f"{text}: {stmt_id} runs in {', '.join(mine)} but not in "
                    f"{', '.join(outside)}, and a map per statement moves "
                    "statements that run in every loop it names"
                )
            extra = [name for name in inside if name not in ins]
            if extra:
                owner = taken[extra[0]][0]
                raise ValueError(
                    f"{text}: {stmt_id} runs in {', '.join(extra)} as well, which "
                    f"the map of {owner} moves and its own does not; a "
                    "statement's map moves every loop of the step it runs in"
                )

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

    def _refuse_tagged(self, inames: Sequence[str], text: str, fix: str) -> None:
        """Refuse a step that replaces a loop carrying a tag, naming ``fix``.

        A tag belongs to a loop, and a split or a tiling replaces the loop by
        new ones, which no tag names. loopy refuses to split a loop with any
        tag but ``for``, with a ``LoopyError`` that named no fix, and splits a
        loop tagged ``for`` into two with no tag. Once a step had left the
        schedule with no kernel (see :attr:`kernel`) nothing refused, and
        either way the tag stayed on a loop the schedule no longer has, so
        the checker read the loops that replaced it as untagged (#93): the
        order made them sequential, and a loop on a hardware axis was no
        longer one. So the step is refused here, before loopy is asked, with
        a kernel or without one, as :meth:`affine` refuses a map over a
        tagged loop. A sum's loop is refused the same way, and any tag,
        ``for`` included, since the loops the step makes are to be tagged for
        themselves.
        """
        tagged = [name for name in inames if name in self._tags]
        if not tagged:
            return
        carried = " and ".join(
            f"{name} carries the tag {self._tags[name]!r}" for name in tagged
        )
        raise ValueError(f"{text}: {carried}; {fix}")

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
        bounds loopy knows how to simplify. So a loop that carries a tag is
        refused, as :meth:`split` refuses one: tile the loops before tagging
        the loops the tiling makes.
        """
        for iname in (first, second):
            if iname not in self._order:
                raise ValueError(f"{iname!r} is not an iname of {self._term.name}")
        if first == second:
            raise ValueError(f"tile() needs two different loops, not {first!r} twice")
        for factor in (first_factor, second_factor):
            _check_factor(factor)
        text = f"tile({first},{second},{first_factor},{second_factor})"
        self._refuse_tagged(
            (first, second), text, "tile before tagging the loops the tiling makes"
        )
        draft = self._draft()
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

        The statements of a loop can also move by maps of their own. Name each
        statement's input tuple, in one union map or its text::

            schedule.affine(
                "{ S0[t, i] -> [a, b] : a = t + i and b = t - i; "
                "S1[t, i] -> [a, b] : a = t + i and b = t - i + 1 }"
            )

        puts ``S1`` half a step after ``S0`` along the diamond, which is what a
        diamond tiling of a pair of statements that feed each other needs. A
        program's statements are named after their calls, ``flux.S0`` and
        ``step@2.S0`` (:mod:`loopty.compose`), which isl cannot read as tuple
        names, so a tuple names a statement by its id with every character
        other than a letter, a digit or an underscore spelled ``_`` as well,
        ``flux_S0`` and ``step_2_S0``, which is the id loopy gives its
        instruction, and so names one statement. The
        statements still share their loops, so every map has to take the same
        loops to the same new ones, and every statement in those loops has to
        run in all of them and be given one map, and only one; each of those
        is a ``ValueError`` naming the statement. The two questions below are
        asked of the maps together, over the dependences between the
        statements as well as within each, and the kernel runs the shared
        loops over the union of the images, each statement at its own points
        only (see :func:`_affine_kernel`).

        The map is untrusted like any other transformation. It has to be
        defined on every instance and send no two of them to one point (the
        ``bijective`` fact, refuted with the instance it misses or the pair it
        merges), and the new order has to run every dependence forward (the
        ``monotone`` fact, refuted with the pair of instances and the array
        cell between them). It need not be unimodular: the image of the diamond
        above is only the points of equal parity, and the kernel is rewritten
        over that image, and loops over it without visiting the holes, as the
        module docstring describes.

        What the kernel rewrite cannot express, such as loops that more than
        one loopy domain defines, is a ``refuted`` ``buildable`` fact with the
        reason, and the schedule then has no kernel.
        """
        try:
            recorded, pieces = _as_maps(mapping)
        except ValueError as exc:
            hint = self._spelling_hint(mapping)
            if hint is None:
                raise
            raise ValueError(f"{exc}\n{hint}") from exc
        text = f"affine({recorded})"
        if pieces is not None:
            pieces = self._statements_named(pieces)
        return self._moved(recorded, pieces, text, ("affine", (recorded,), {}))

    def _moved(
        self,
        recorded: Any,
        pieces: Mapping[str, isl.Map] | None,
        text: str,
        recipe: tuple[str, tuple, dict],
    ) -> Schedule:
        """The step that moves loops along one map, or a map per statement.

        :meth:`affine` and :meth:`fuse` are this step, the first with the
        map it is given and the second with the maps it builds; ``recorded``
        is the map as the step is written, and ``recipe`` how it is replayed.
        """
        mapping = recorded if pieces is None else next(iter(pieces.values()))
        draft = self._draft()
        self._reindex_into(draft, mapping, text, pieces)
        inputs = (
            _dim_names(mapping, isl.dim_type.in_)
            if pieces is None
            else _taken_loops(pieces.values(), draft.order)
        )
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
        self._affine_into(draft, mapping, pieces)
        if pieces is not None and len(
            {_dim_names(piece, isl.dim_type.in_) for piece in pieces.values()}
        ) > 1:
            draft.kernel = self._ordered_by_dependences(draft)
        return self._commit(draft, text, recipe)

    def _ordered_by_dependences(self, draft: _Draft) -> Any:
        """The draft's kernel, its statements ordered as the dependences order them.

        The lowering draws an instruction's dependencies by array, in the
        order of the term: a statement depends on every earlier one that
        writes what it reads, whatever the cells. While the loops run as the
        term runs them that is enough, and two steps change that. A fusion
        puts two loops in one, and a statement between them in the term can
        then be ordered after the loop by the cells and before it by an
        array the second reads at cells it never writes: loopy, given both,
        finds no order (a ``CycleError`` from code generation). A
        substitution takes out the producer, and a later write of what it
        read depended on the producer and on nothing that now reads it: loopy
        may then run the write first.

        So every dependency between two statements is drawn again from the
        dependences the casts are checked against, the ones of the program as
        it runs now (see :meth:`substitute`): one statement's instruction
        depends on another's where a dependence joins an instance of one to
        an instance of the other, in either direction, and in the order the
        schedule puts the two in a step of the loops they share, which is
        the order of their constants in the time map (see
        :func:`_coefficients`). Two that touch one variable and that no
        dependence joins are said to need no order (loopy's ``no_sync_with``,
        which loopy asks for), and no barrier either. The instructions that
        compute a ragged row's length keep their dependencies, and so do the
        statements' on them.
        """
        kernel = draft.kernel
        if kernel is None:
            return None
        total = self._deps_total
        statement_of = {insn: stmt for stmt, insn in self._lowering.insn_ids.items()}
        layout = self._origin_layout
        instances = {
            stmt_id: _embed(self._domains[stmt_id], layout.index(stmt_id), layout)
            for stmt_id in layout.stmt_ids
        }
        joined: set[frozenset[str]] = set()
        if total is not None:
            for first in layout.stmt_ids:
                for second in layout.stmt_ids:
                    if first == second:
                        continue
                    pairs = total.intersect_domain(
                        instances[first].align_params(total.get_space().domain())
                    ).intersect_range(
                        instances[second].align_params(total.get_space().range())
                    )
                    if not pairs.is_empty():
                        joined.add(frozenset((first, second)))
        current = _Layout(stmt_ids=self._layout.stmt_ids, coords=dict(draft.coords))
        position = _coefficients(current.stmt_ids, _nests(current, draft.order, {}))
        entry = kernel.default_entrypoint
        present = {
            statement_of[insn.id]: insn
            for insn in entry.instructions
            if insn.id in statement_of
        }
        touched = {
            stmt_id: (
                set(insn.assignee_var_names()),
                set(insn.assignee_var_names()) | set(insn.read_dependency_names()),
            )
            for stmt_id, insn in present.items()
        }
        insns = []
        for insn in entry.instructions:
            mine = statement_of.get(insn.id)
            if mine is None:
                insns.append(insn)
                continue
            kept = {other for other in insn.depends_on if other not in statement_of}
            free: set[str] = set()
            for theirs, other in present.items():
                if theirs == mine:
                    continue
                if frozenset((mine, theirs)) in joined:
                    if position[theirs] < position[mine]:
                        kept.add(other.id)
                    continue
                writes, reads = touched[mine]
                their_writes, their_reads = touched[theirs]
                if writes & their_reads or their_writes & reads:
                    free.add(other.id)
            insns.append(
                insn.copy(
                    depends_on=frozenset(kept),
                    no_sync_with=frozenset(
                        item for item in insn.no_sync_with if item[0] not in free
                    )
                    | frozenset((other, "any") for other in free),
                )
            )
        return kernel.with_kernel(entry.copy(instructions=insns))

    def fuse(
        self, producer: str, consumer: str, shift: int | Sequence[int] = 0
    ) -> Schedule:
        """Run the consumer's loops inside the producer's, ``shift`` steps behind.

        ``producer`` and ``consumer`` name statements: a statement by its id
        (``S0``, ``flux.S0``), or a call of a program by its label (``flux``,
        ``step@2``), which names every statement of the call and not the
        checked points before it (see :meth:`_fused_side`). The producer's
        come first in the term. The step is :meth:`affine` with a map per
        statement that this method builds, and it is checked as that is. The
        producer's loops keep their names and their values; the consumer's
        loops, as many as both have, outermost first, become the producer's,
        ``shift`` steps behind:

            { flux_S0[j] -> [j]; divergence_S0[i] -> [j] : j = i + 1 }

        is ``fuse("flux", "divergence", shift=1)``. Every other statement in
        those loops moves with the side whose loops it runs in. Inside the
        fused loops the producer's statements come first, as they did in the
        term, so a value the consumer reads in the step it is written is
        already there. One shift per fused loop, or one number for a single
        loop.

        The two sides' loops have to be in sequence, not one inside the
        other. The fused loops are the outer loops of each side's nest, and a
        nest the lowering wrote as one domain is cut after them (see
        :func:`_cut_for`); a fusion the kernel rewrite cannot write otherwise
        leaves the casts decided and the kernel refused as unbuildable, as
        :meth:`affine` leaves a map it cannot write. A fusion
        that runs a dependence backwards is refused with the pair of
        instances and the cell between them, as every cast is, and the
        message names the least shift at which the fusion is accepted, when
        there is one and a number per loop gives it.
        """
        return self._fuse(producer, consumer, shift, hint=True)

    def _fuse(
        self,
        producer: str,
        consumer: str,
        shift: int | Sequence[int],
        hint: bool,
    ) -> Schedule:
        """:meth:`fuse`, with ``hint`` saying whether a refusal names a shift."""
        given = shift if isinstance(shift, int) else tuple(shift)
        text = f"fuse({producer}, {consumer}" + (
            f", shift={given!r})" if given else ")"
        )
        recipe = ("fuse", (producer, consumer), {"shift": given} if given else {})
        first = self._fused_side(producer, text)
        second = self._fused_side(consumer, text)
        both = sorted(set(first) & set(second))
        if both:
            raise ValueError(
                f"{text}: {', '.join(both)} is named on both sides of the fusion"
            )
        position = self._layout.index
        if max(position(s) for s in first) > min(position(s) for s in second):
            raise ValueError(
                f"{text}: {consumer} comes before {producer} in "
                f"{self._term.name}; the producer is the one that comes first"
            )
        mine = self._fused_loops(first, producer, text)
        theirs = self._fused_loops(second, consumer, text)
        depth = min(len(mine), len(theirs))
        mine, theirs = mine[:depth], theirs[:depth]
        shared = [name for name in mine if name in theirs]
        if shared:
            raise ValueError(
                f"{text}: {producer} and {consumer} already run in "
                f"{', '.join(shared)}"
            )
        shifts = _shifts(given, depth, text)
        identity = _reindexing(mine, mine, [f"b{k} = a{k}" for k in range(depth)])
        behind = _reindexing(
            theirs,
            mine,
            [f"b{k} = a{k} + ({shifts[k]})" for k in range(depth)],
        )
        pieces: dict[str, isl.Map] = {}
        for stmt_id, coords in self._layout.coords.items():
            if stmt_id in self._gone:
                continue
            if set(coords) & set(mine):
                pieces[stmt_id] = identity
            elif set(coords) & set(theirs):
                pieces[stmt_id] = behind
        try:
            return self._moved(identity, pieces, text, recipe)
        except IllegalCast as exc:
            if not hint or exc.fact is None or exc.fact.kind != "monotone":
                raise
            least = self._least_shift(first, second, mine, theirs)
            if least is None or least == shifts:
                raise
            suggested: int | tuple[int, ...] = least[0] if depth == 1 else least
            try:
                self._fuse(producer, consumer, suggested, hint=False)
            except (IllegalCast, ValueError):
                raise exc from None
            message = (
                f"{exc}; fuse({producer!r}, {consumer!r}, shift={suggested!r}) "
                "runs every dependence between them forward"
            )
            provenance = {**exc.fact.provenance, "reason": message}
            fact = dataclasses.replace(exc.fact, provenance=provenance)
            raise IllegalCast(message, witness=exc.witness, fact=fact) from None

    def _fused_side(self, name: str, text: str) -> list[str]:
        """The statements one side of :meth:`fuse` names, in term order.

        A call's label names the call's own statements, its scope's (see
        :class:`loopty.term.Scope`), and not the checked points the program
        put before it (``gather.check.perm``): those check what an earlier
        call left for this one, and a fusion that would run the call before
        them is refused for the flag they set, which the call reads. Those a
        substitution took out run no more, and are left out; a side that
        names only such statements is refused.
        """
        ids = self._layout.stmt_ids
        calls = {scope.call: scope.statements for scope in self._term.scopes}
        named = (
            [name]
            if name in ids
            else [stmt_id for stmt_id in calls[name] if stmt_id in ids]
            if name in calls
            else [stmt_id for stmt_id in ids if stmt_id.startswith(f"{name}.")]
            or [stmt_id for stmt_id in ids if _sanitize(stmt_id) == name]
        )
        if not named:
            raise ValueError(
                f"{text}: {name!r} names no statement of {self._term.name} and "
                f"no call of it; its statements are {', '.join(ids)}"
            )
        live = [stmt_id for stmt_id in named if stmt_id not in self._gone]
        if not live:
            raise ValueError(
                f"{text}: {', '.join(named)} runs no more: a substitution took "
                "it out of the kernel"
            )
        return live

    def _fused_loops(
        self, stmts: Sequence[str], name: str, text: str
    ) -> tuple[str, ...]:
        """The loops every statement of one side runs in, outermost first."""
        nests = [
            [loop for loop in self._order if loop in self._layout.coords[stmt_id]]
            for stmt_id in stmts
        ]
        common: list[str] = []
        for loops in zip(*nests, strict=False):
            if any(loop != loops[0] for loop in loops):
                break
            common.append(loops[0])
        if not common:
            raise ValueError(
                f"{text}: the statements {name} names, {', '.join(stmts)}, run in "
                "no loop all of them share, and a fusion moves loops"
            )
        return tuple(common)

    def _least_shift(
        self,
        first: Sequence[str],
        second: Sequence[str],
        mine: Sequence[str],
        theirs: Sequence[str],
    ) -> tuple[int, ...] | None:
        """The least shift per loop that runs the dependences of two sides forward.

        For each fused loop, the largest distance, the producer's value less
        the consumer's, over the dependences from a statement of ``first`` to
        one of ``second``, at the loops as they are now; ``None`` when one of
        them is not a number for every value of the sizes. Each loop is asked
        on its own, which is enough for a shift to be legal for these
        dependences, and :meth:`fuse` asks the checker before it names one.
        """
        layout = self._layout
        width = layout.width + 1
        source = ", ".join(f"a{k}" for k in range(width))
        target = ", ".join(f"b{k}" for k in range(width))
        least: list[int] = []
        for mine_loop, their_loop in zip(mine, theirs, strict=True):
            best: int | None = None
            for dep in self._deps:
                if dep.source not in first or dep.sink not in second:
                    continue
                relation = dep.relation.apply_domain(self._reindex).apply_range(
                    self._reindex
                )
                a = layout.coords[dep.source].index(mine_loop) + 1
                b = layout.coords[dep.sink].index(their_loop) + 1
                distance = isl.Map(
                    f"{{ [{source}, {target}] -> [d] : d = a{a} - b{b} }}"
                ).align_params(relation.get_space())
                values = relation.wrap().flatten().apply(distance)
                if values.is_empty():
                    continue
                for _piece, value in values.dim_max(0).get_pieces():
                    if not value.is_cst():
                        return None
                    found = value.get_constant_val().to_python()
                    best = found if best is None else max(best, found)
            least.append(0 if best is None else int(best))
        return tuple(least)

    def _statements_named(self, pieces: Mapping[str, isl.Map]) -> dict[str, isl.Map]:
        """The maps per statement of :meth:`affine`, by statement id.

        A tuple name that is a statement's id names it. One that is not is
        read as a statement's id spelled as an isl name, every character other
        than a letter, a digit or an underscore written ``_`` (``flux_S0`` for
        the program statement ``flux.S0``). That is the id of the statement's
        instruction in the lowered kernel, and loopy refuses two instructions
        of one id, so in a term that lowers the spelling names one statement.
        A name that is neither is passed on as it is, and refused as not a
        statement of the kernel (:meth:`_check_pieces`).
        """
        spelled = {insn: stmt_id for stmt_id, insn in self._lowering.insn_ids.items()}
        ids = {stmt.id for stmt in self._term.stmts}
        # isl reads no id that differs from its spelling, so no two names
        # given here are one statement's.
        return {
            name if name in ids else spelled.get(name, name): piece
            for name, piece in pieces.items()
        }

    def _spelling_hint(self, mapping: Any) -> str | None:
        """What to write for a statement whose id isl cannot read, if one is.

        For the text of a map that isl refused, when the text names a
        statement of the term by an id that is not an isl name, as a
        program's statement ids are (``flux.S0``).
        """
        if not isinstance(mapping, str):
            return None
        named = [
            stmt.id
            for stmt in self._term.stmts
            if _sanitize(stmt.id) != stmt.id and f"{stmt.id}[" in mapping
        ]
        if not named:
            return None
        spellings = ", ".join(f"{stmt_id} as {_sanitize(stmt_id)}" for stmt_id in named)
        return (
            "A statement whose id isl cannot read as a tuple name is named with "
            "every character other than a letter, a digit or an underscore "
            f"spelled _: {spellings}"
        )

    def _affine_into(
        self,
        draft: _Draft,
        mapping: isl.Map,
        pieces: Mapping[str, isl.Map] | None = None,
    ) -> None:
        """Rewrite the draft's kernel along ``mapping``, or record why not.

        ``pieces`` is a map per statement, by statement id, as
        :meth:`_reindex_into` takes it; the rewrite is told them by the
        instruction each statement lowered to.
        """
        if draft.kernel is None:
            return
        if pieces is not None:
            ids = self._lowering.insn_ids
            pieces = {ids[stmt_id]: piece for stmt_id, piece in pieces.items()}
        kernel, reason = _affine_kernel(
            draft.kernel, mapping, pieces, self._term.sizes
        )
        if reason is None:
            draft.kernel = kernel
        else:
            draft.kernel = None
            draft.unbuildable = reason

    def pack(self, *arrays: str) -> Schedule:
        """Store arrays over a polyhedral domain packed, a row at a time.

        An array over ``Where[...]``, ``Sigma[...]`` or a union is boxed unless
        a schedule says otherwise: the box of its binders, with the cells
        outside the domain wasted. Packed, its cells are kept in lexicographic
        order and ``L[i, j]`` is read through a table of row starts,
        ``L[off_L[i] + j]`` (see :mod:`loopty.domain`), which the executor
        computes from the domain and passes in.

        Not a cast: no instance moves, the facts are about cells and not
        about where they are kept, and the step emits none. The schedule is
        lowered again from its kernel with the arrays packed, and every step
        so far is replayed, so each one is checked again against the kernel
        that stores them so. A domain with a row that is not an interval
        cannot be packed and is refused when it is lowered.
        """
        if not arrays:
            raise ValueError("pack() names the arrays to store packed")
        for name in arrays:
            if name not in self._lowering.storage:
                raise ValueError(
                    f"pack({name!r}): {name} is not an array over a Where, Sigma "
                    f"or union domain of {self._term.name}, and a dense or "
                    "ragged array is stored one way"
                )
        layouts = {**self._layouts, **dict.fromkeys(arrays, "packed")}
        out = Schedule(
            self._source, target=self._target, sizes=self._sizes, _layouts=layouts
        )
        for method, args, kwargs in self._steps:
            out = getattr(out, method)(*args, **kwargs)
        out = out._clone()
        out._steps = (*out._steps, ("pack", tuple(arrays), {}))
        out._history = (*out._history, f"pack({', '.join(arrays)})")
        out._examples = None if self._examples is None else dict(self._examples)
        return out

    def substitute(self, array: str) -> Schedule:
        """Compute an array the program makes where it is read, and store none of it.

        ``f = Arr.zeros_like(u)`` in a program is a temporary of its kernel
        (:mod:`loopty.compose`), stored in full between the call that writes
        it and the calls that read it. When one statement writes it, one
        value per cell at the cell its loop variables name (``f[j] = 0.5 *
        u[j] * u[j]``), every read ``f[i + 1]`` can be that value instead,
        computed where it is read: ``0.5 * u[i + 1] * u[i + 1]``. The kernel
        is rewritten with loopy's own ``assignment_to_subst``, which turns the
        statement into a substitution rule and drops it and the temporary,
        once the statement that zeroes the array where the program made it
        is dropped too. Each read gets the value converted to the array's
        element type, as storing it converted it (a ``float32`` array rounds
        what it stores; see :func:`_substituted_kernel`).

        That is a storage decision and not a reordering, and it is legal when
        three things hold, each asked before anything is rewritten:

        * the array is the program's own, which no caller sees, and one
          statement writes it besides the zeros: a ``ValueError`` names what
          stands in the way otherwise, as it does a statement that is not
          pointwise (a cell other than its loop variables, each once, a sum,
          a guard isl cannot state, a read of the array itself);
        * every cell of the array any statement reads, that statement wrote
          before the read, so no read sees the zeros: the ``definedness``
          fact, decided by isl, or refuted with the cell, and the read, that
          shows otherwise, or assumed, and refused all the same, where a read
          cannot be listed;
        * nothing writes what the statement read between its run and a
          read of what it stored, in the order the schedule has now: each
          dependence of the statement's reads is carried over to the reads
          that compute it again, and the order has to run every one of them
          forward, which is the ``monotone`` fact, refuted with the pair of
          instances and the cell between them, as every cast is.

        The statements it no longer runs have no instances left, and the
        loops only they ran in are no loops of the schedule, so no later step
        can name them, and every later step is checked against the
        dependences of the program as it now runs: the reads of the array are
        gone, and the reads that replace them are there. The kernel's
        instructions are ordered by those too
        (:meth:`_ordered_by_dependences`). A substitution through two arrays,
        one computed from the other, is refused. Contracting an array to the
        window of cells that are live at once is the other way of storing it
        less, and is not done.
        """
        text = f"substitute({array!r})"
        recipe = ("substitute", (array,), {})
        term = self._term
        if array not in dict(term.temporaries) or array in dict(term.checks):
            what = (
                f"a parameter of {term.name}, which its caller passes and sees"
                if array in term.param_names
                else f"the flag of a checked point of {term.name}"
                if array in dict(term.checks)
                else f"no array of {term.name}"
            )
            raise ValueError(
                f"{text}: {array} is {what}, and only an array the program "
                "makes with Arr.zeros_like is its own to store or not"
            )
        if array in self._substituted:
            raise ValueError(f"{text}: {array} is substituted already")
        zeros = f"{array}.zeros"
        writers = [stmt for stmt in term.stmts if stmt.assignee.array == array]
        producers = [stmt for stmt in writers if stmt.id != zeros]
        if len(producers) != 1:
            which = ", ".join(stmt.id for stmt in producers) or "no statement"
            raise ValueError(
                f"{text}: {array} is written by {which}, and a substitution "
                "computes the value of the one statement that stores it"
            )
        (producer,) = producers
        why = _not_pointwise(producer, array, term)
        if why is not None:
            raise ValueError(f"{text}: {why}")
        readers = [
            stmt
            for stmt in term.stmts
            if stmt not in writers
            and any(
                kind == "read" and name == array
                for kind, name, _indices, _part in _accesses(stmt, term)
            )
        ]
        if not readers:
            raise ValueError(f"{text}: nothing in {term.name} reads {array}")
        # The dependences carried over are those of what the producer reads,
        # at the instances that read the array; a statement an earlier
        # substitution took out reads nothing, and an array it computes is
        # read through what its own producer reads. Both are refused.
        taken = [stmt.id for stmt in readers if stmt.id in self._gone]
        through = sorted(
            {
                name
                for kind, name, _indices, _part in _accesses(producer, term)
                if kind == "read" and name in self._substituted
            }
        )
        if taken or through:
            how = (
                f"{taken[0]} reads {array} and runs no more"
                if taken
                else f"{producer.id} reads {through[0]}, which is computed where "
                "it is read"
            )
            raise ValueError(
                f"{text}: {how}, after an earlier substitution; a substitution "
                "through two arrays is not done, so substitute only one of them"
            )

        fact = self._definedness_fact(producer, readers, array, text, recipe)
        if fact.status.value != "decided":
            raise IllegalCast(
                fact.provenance["reason"],
                witness=fact.provenance.get("witness"),
                fact=fact,
            )
        gone = {stmt.id for stmt in writers}
        staged = self._clone()
        staged._substituted = (*self._substituted, array)
        staged._gone = self._gone | gone
        # The statements taken out have no instances left, and the loops
        # only they ran in are no loops of the kernel any more, so no later
        # step can name them.
        layout = self._origin_layout
        removed = [
            _embed(self._domains[stmt_id], layout.index(stmt_id), layout)
            for stmt_id in sorted(gone)
        ]
        origin = self._origin
        for instances in removed:
            instances = instances.align_params(origin.get_space())
            origin = origin.align_params(instances.get_space()).subtract(instances)
        staged._origin = origin.coalesce()
        staged._deps, staged._within = self._carried_over(
            producer, gone, readers, array
        )
        # The dependences the schedule was checked against join the
        # statements that still run as they did, since only the reads of the
        # array change, and they include what loopy.flow found besides the
        # ones listed here (see _cross_check); the carried ones go with them.
        kept = (
            None
            if self._deps_total is None
            else self._deps_total.intersect_domain(staged._origin).intersect_range(
                staged._origin
            )
        )
        staged._deps_total = _union(
            relation
            for relation in (
                *(dep.relation for dep in staged._deps),
                *(() if kept is None or kept.is_empty() else (kept,)),
            )
        )
        staged._flow_note = (
            f"those of {term.name} with {array} computed where it is read: the "
            f"dependences of {', '.join(sorted(gone))} dropped, and those of "
            f"{producer.id}'s reads carried over to the reads of {array}"
        )
        staged._reindex = self._reindex.intersect_domain(staged._origin)
        staged._instances = staged._reindex.range().coalesce()
        live = {
            loop
            for stmt_id, coords in self._layout.coords.items()
            if stmt_id not in staged._gone
            for loop in coords
        }
        staged._order = [loop for loop in self._order if loop in live]
        draft = staged._draft()
        if draft.kernel is not None:
            ids = self._lowering.insn_ids
            removed = [ids[stmt.id] for stmt in writers if stmt.id == zeros]
            kernel, reason = _substituted_kernel(
                draft.kernel,
                array,
                removed,
                ids[producer.id],
                _computed_in(term, producer, array, readers),
            )
            draft.kernel = kernel
            if reason is not None:
                draft.unbuildable = reason
            draft.kernel = staged._ordered_by_dependences(draft)
        return staged._commit(draft, text, recipe, leading=(fact,))

    @property
    def substituted(self) -> tuple[str, ...]:
        """The arrays :meth:`substitute` computes where they are read, in order."""
        return self._substituted

    def _definedness_fact(
        self,
        producer: Stmt,
        readers: Sequence[Stmt],
        array: str,
        text: str,
        recipe: tuple[str, tuple, dict],
    ) -> Any:
        """The ``definedness`` fact of :meth:`substitute`, decided or refuted.

        Two questions about the reads of ``array``: are their cells cells
        ``producer`` writes (:func:`loopty.flow.definedness`), and does any
        read come before the write of its cell, a dependence from the read
        to the producer, which would read the zeros. The first refutation is
        the fact's. A read isl cannot list (an index that is not affine, a
        guard it cannot state) leaves the first question open: the step is
        refused all the same, and the fact is ``assumed``, not refuted.
        """
        from loopty.flow import definedness

        statement = (
            f"every cell of {array} that {self._term.name} reads, {producer.id} "
            "has stored by the time it is read"
        )
        verdict = definedness(self._term, array, [producer], readers)
        message = ""
        witness: Any = None
        detail = "every read of the array is of a cell the statement stores, after it"
        if verdict.ok is not True:
            witness = verdict.witness
            detail = verdict.detail
            message = (
                f"{text} illegal: {verdict.detail}, so the read would see the "
                "zeros the array was made with, and computing it again would not"
                if verdict.ok is False
                else f"{text} illegal: {verdict.detail}"
            )
        else:
            early = next(
                (
                    dep
                    for dep in self._deps
                    if dep.kind == "war"
                    and dep.array == array
                    and dep.sink == producer.id
                ),
                None,
            )
            if early is not None:
                witness = self._pair_in(early.relation)
                (reader, coords), (_, stored), params = witness
                cell = _cell_text(early.source_indices, coords, params)
                detail = f"{reader} reads {array} before {producer.id} stores it"
                message = (
                    f"{text} illegal: instance {_instance_text(reader, coords)} "
                    f"reads {array}[{cell}] before "
                    f"{_instance_text(producer.id, stored)} stores it"
                    f"{_sizes_text(params, self._sizes)}, so the read sees the "
                    "zeros the array was made with, and computing it again would not"
                )
        return self._fact(
            "definedness",
            statement,
            status=(
                "decided"
                if not message
                else "assumed"
                if verdict.ok is None
                else "refuted"
            ),
            witness=witness,
            detail=detail,
            step=recipe,
            reason=message,
            about=array,
        )

    def _carried_over(
        self, producer: Stmt, gone: set[str], readers: Sequence[Stmt], array: str
    ) -> tuple[tuple[_Dep, ...], tuple[_Dep, ...]]:
        """The dependences of the program once ``array`` is computed where it is read.

        Those of the statements that no longer run (``gone``: the producer and
        the zeros) are dropped, and the reads of ``array`` are reads of what
        the producer read, at the cells the producer read at the instance
        that stored the cell: ``f[i + 1]`` read for ``f[j] = 0.5 * u[j] *
        u[j]`` reads ``u[i + 1]``. Each of those reads has a dependence with
        every write of its cell, in the direction the producer's read had:
        after a write that came before the producer's instance (``raw``), and
        before one that came after it (``war``), since the value read has to
        be the one the producer read. A statement whose sum reads such a cell
        that its own instruction writes has the pair :func:`_within_instances`
        lists, as it would for a read written out.
        """
        term = self._term
        layout = self._origin_layout
        instances = self._origin
        params = set(self._params)
        identity = isl.Map.identity(instances.get_space().map_from_set())
        kept = [
            dep for dep in self._deps if dep.source not in gone and dep.sink not in gone
        ]
        within = [dep for dep in self._within if dep.source not in gone]
        stored = producer.assignee.indices
        reads = [
            (name, indices, part)
            for kind, name, indices, part in _accesses(producer, term)
            if kind == "read"
        ]
        writes = [
            (stmt, name, indices, part)
            for stmt in term.stmts
            if stmt.id not in gone
            for kind, name, indices, part in _accesses(stmt, term)
            if kind == "write"
        ]
        carried: list[_Dep] = []
        for reader in readers:
            for kind, name, at, part in _accesses(reader, term):
                if kind != "read" or name != array:
                    continue
                renaming = {
                    index.name: _plain(value)
                    for index, value in zip(stored, at, strict=True)
                }
                to_producer = (
                    _same_cell(reader, producer, at, stored, layout, params)
                    .intersect_domain(instances)
                    .intersect_range(instances)
                )
                later = to_producer.apply_range(self._before)
                earlier = to_producer.apply_range(self._before.reverse())
                for read, indices, _producer_part in reads:
                    moved = tuple(
                        substitute(_plain(index), renaming) for index in indices
                    )
                    for writer, written, cells, writer_part in writes:
                        if written != read or len(cells) != len(moved):
                            continue
                        same = (
                            _same_cell(reader, writer, moved, cells, layout, params)
                            .intersect_domain(instances)
                            .intersect_range(instances)
                        )
                        after = same.intersect(later).subtract(identity).coalesce()
                        if not after.is_empty():
                            carried.append(
                                _Dep(
                                    kind="war",
                                    array=read,
                                    source=reader.id,
                                    sink=writer.id,
                                    source_indices=moved,
                                    sink_indices=cells,
                                    relation=after,
                                    source_part=part,
                                    sink_part=writer_part,
                                )
                            )
                        before = same.intersect(earlier).subtract(identity).coalesce()
                        if not before.is_empty():
                            carried.append(
                                _Dep(
                                    kind="raw",
                                    array=read,
                                    source=writer.id,
                                    sink=reader.id,
                                    source_indices=cells,
                                    sink_indices=moved,
                                    relation=before.reverse(),
                                    source_part=writer_part,
                                    sink_part=part,
                                )
                            )
                        if writer is reader and part == "sum" and writer_part == "":
                            mine = same.intersect(identity)
                            if not mine.is_empty():
                                within.append(
                                    _Dep(
                                        kind="war",
                                        array=read,
                                        source=reader.id,
                                        sink=reader.id,
                                        source_indices=cells,
                                        sink_indices=cells,
                                        relation=mine.coalesce(),
                                        source_part="sum",
                                        sink_part="",
                                    )
                                )
        return (*kept, *carried), tuple(within)

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
        leading: Sequence[Any] = (),
    ) -> Schedule:
        """Check one transformation and return the schedule it produces.

        ``recipe`` is how the transformation would be written in Python, as
        ``(method, args, kwargs)``, kept so that :meth:`retarget` can replay it,
        and so that the facts about this step name it (see :attr:`key`).
        ``leading`` are facts the step decided before it came here, which go
        before the ones decided here (see :meth:`substitute`).
        """
        layout = _Layout(
            stmt_ids=self._layout.stmt_ids, coords=dict(draft.coords)
        )
        step = _step_map(self._layout, layout, draft.mappings)

        facts: list[Any] = list(leading)

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
        detail = overall.detail if bad is None else bad[2]
        if bad is not None:
            message = self._render_violation(text, *bad)
        elif refused:
            message = f"{text} illegal: {overall.detail}"
        else:
            message = ""
        # The order leaves the loops on hardware axes out, which is right for
        # two instances on one work item; a dependence between two work items
        # is refused whatever the order (see "Work items" in the module
        # docstring).
        sums: dict[str, list[str]] = {}
        for name, key in draft.reductions.items():
            sums.setdefault(key.rsplit(":", 1)[0], []).append(name)
        work = _work_items(
            layout,
            draft.tags,
            draft.kernel,
            set(self._params) | set(self._term.sizes),
            sums,
        )
        if not refused and work is not None:
            crossing = self._first_crossing(text, reindex, work)
            if crossing is not None:
                refused = True
                message, witness, detail = crossing
        within = "" if work is None else ", within one work item"
        facts.append(
            self._fact(
                "monotone",
                f"the order after {text} runs every dependence of "
                f"{self._term.name} forward{within}",
                status="refuted" if refused else "decided",
                witness=witness,
                detail=detail,
                step=recipe,
                reason=message,
                rests_on=self._layout_ids,
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
        other._code, other._strides = _stepped(
            other._kernel, draft.order, draft.tags
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
        return self._pair_in(
            violating_times.apply_domain(inverse)
            .apply_range(inverse)
            .intersect(relation)
        )

    def _pair_in(self, relation: isl.Map) -> tuple:
        """One pair of instances ``relation`` holds, named, and its sizes.

        Read off at the size hint where the relation has a pair there, as a
        witness is (see :meth:`_witness`), and at isl's choice otherwise.
        """
        bad = self._at_hint(relation)
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

    def _at_hint(self, obj: Any) -> Any:
        """``obj`` with the sizes fixed at the hint, if it has a point there.

        A witness is read off at the sizes the schedule was given when it
        can be (see :meth:`_witness`), and at isl's choice when the hint
        leaves nothing to read.
        """
        if not self._sizes:
            return obj
        names = obj.get_var_names(isl.dim_type.param)
        fixed = obj
        for name, value in self._sizes.items():
            if name in names:
                fixed = fixed.fix_val(
                    isl.dim_type.param,
                    names.index(name),
                    isl.Val.int_from_si(obj.get_ctx(), value),
                )
        return obj if fixed.is_empty() else fixed

    def _first_crossing(
        self, text: str, reindex: isl.Map, work: _WorkItems
    ) -> tuple[str, tuple, str] | None:
        """A dependence the schedule puts on two work items, or ``None``.

        Asked of the union of the dependences first, with every part of a
        statement where any part of it runs (:attr:`_WorkItems.spread`),
        which is what the verdict is about when nothing crosses. When
        something might, each dependence and each axis is asked with the
        work items its two accesses run on (:meth:`_WorkItems.placed`), and
        what ``loopy.flow`` finds besides with the union's. The pairs within
        one instance that only work items separate (see
        :func:`_within_instances`) are asked as dependences are. Returns the
        message, the witness and the detail of the refusal. ``reindex`` takes
        the term's instances to the schedule's.
        """
        total = self._deps_total
        crossed = False
        spread: dict[str, isl.Map] = {}
        if total is not None:
            total = total.subtract(
                isl.Map.identity(total.get_space().domain().map_from_set())
            )
            spread = {
                axis: _apart(reindex, work.spread[axis]) for axis in work.axes
            }
            crossed = any(
                not total.intersect(spread[axis]).is_empty() for axis in work.axes
            )
        for dep in (*(self._deps if crossed else ()), *self._within):
            for axis in work.axes:
                apart = _apart(
                    reindex,
                    work.placed(dep.source_part, axis),
                    work.placed(dep.sink_part, axis),
                )
                if dep.relation.intersect(apart).is_empty():
                    continue
                return self._render_crossing(text, dep, axis, reindex, work)
        if not crossed or total is None:
            return None
        mine = _union(dep.relation for dep in self._deps)
        others = total if mine is None else total.subtract(mine)
        for axis in work.axes:
            crossing = others.intersect(spread[axis])
            if crossing.is_empty():
                continue
            return self._unattributed_crossing(  # pragma: no cover - see below
                text, crossing, axis, reindex, work
            )
        return None

    def _unattributed_crossing(
        self,
        text: str,
        relation: isl.Map,
        axis: str,
        reindex: isl.Map,
        work: _WorkItems,
    ) -> tuple[str, tuple, str]:  # pragma: no cover - loopy.flow agrees here
        """A crossing only ``loopy.flow``'s dependences have, which name no cell.

        The schedule is checked against the union of both (see
        :func:`_cross_check`), so a dependence only the other finds is refused
        too, with the pair of instances and no array cell.
        """
        spread = work.spread[axis]
        witness, _ = self._crossing_witness(
            relation, axis, reindex, work, spread, spread
        )
        message = (
            f"{text} illegal: a dependence loopty.flow finds joins two work "
            f"items of {axis}, and nothing in a kernel orders two work items "
            "through global memory"
        )
        return message, witness, f"a dependence joins two work items of {axis}"

    def _crossing_witness(
        self,
        relation: isl.Map,
        axis: str,
        reindex: isl.Map,
        work: _WorkItems,
        source: isl.Map,
        sink: isl.Map,
    ) -> tuple[tuple, tuple[int, int]]:
        """A pair of ``relation`` on two work items of ``axis``, and the two.

        ``source`` and ``sink`` give the work items each end runs on (see
        :meth:`_WorkItems.placed`). The pair and its work items are sampled
        together, at the size hint where there is a point there, and the
        parameters that stand for a start loopy could not say, or for the
        work item a sum's statement runs on, are dropped first, since a
        witness is read at sizes and not at those.
        """
        pairs = (
            reindex.apply_range(source)
            .product(reindex.apply_range(sink))
            .intersect_domain(relation.wrap())
            .intersect_range(isl.Map(_OTHER_WORK_ITEM).wrap())
        )
        points = pairs.wrap().flatten()
        for name in work.hidden:
            names = points.get_var_names(isl.dim_type.param)
            if name in names:
                points = points.project_out(isl.dim_type.param, names.index(name), 1)
        points = self._at_hint(points)
        n_params = points.dim(isl.dim_type.param)
        names = points.get_var_names(isl.dim_type.param)
        if n_params:
            points = points.move_dims(
                isl.dim_type.set, 0, isl.dim_type.param, 0, n_params
            )
        point = isl_oracle.sample_point(points)
        if point is None:  # pragma: no cover - the pairs are known to be there
            return (), (0, 0)
        params = dict(zip(names, point[:n_params], strict=True))
        width = self._origin_layout.width + 1
        values = point[n_params:]
        source_instance = self._name_instance(values[:width])
        sink_instance = self._name_instance(values[width : 2 * width])
        return (source_instance, sink_instance, params), (
            values[2 * width],
            values[2 * width + 1],
        )

    def _render_crossing(
        self, text: str, dep: _Dep, axis: str, reindex: isl.Map, work: _WorkItems
    ) -> tuple[str, tuple, str]:
        """The refusal of a dependence between two work items, and its witness.

        Which instance, which cell, and why the two run on two work items:
        the loop on the axis puts them there, by its values, or one of the
        statements has no loop on the axis and runs on every work item of it,
        or a sum on the axis runs the access on every work item of it and the
        statement's own instruction on one.
        """
        witness, (first, second) = self._crossing_witness(
            dep.relation,
            axis,
            reindex,
            work,
            work.placed(dep.source_part, axis),
            work.placed(dep.sink_part, axis),
        )
        (source_id, source_coords), (sink_id, sink_coords), params = witness
        source_loop = work.loops[source_id][axis]
        sink_loop = work.loops[sink_id][axis]
        if source_loop is None or sink_loop is None:
            active = {
                "raw": ("writes", "reads"),
                "war": ("reads", "writes"),
                "waw": ("writes", "writes"),
            }[dep.kind]
            everywhere: list[str] = []
            parts: list[str] = []
            for stmt_id, loop, part, verb in (
                (source_id, source_loop, dep.source_part, active[0]),
                (sink_id, sink_loop, dep.sink_part, active[1]),
            ):
                if loop is not None:
                    continue
                if axis not in work.summed[stmt_id]:
                    if not everywhere:
                        parts.append("")
                    if stmt_id not in everywhere:
                        everywhere.append(stmt_id)
                elif part == "":
                    parts.append(
                        f"{stmt_id} {verb} it on one work item of {axis}, once "
                        f"its sum on {axis} is done"
                    )
                elif part == "sum":
                    parts.append(
                        f"{stmt_id} {verb} it in a sum, on every work item of {axis}"
                    )
                else:
                    parts.append(
                        f"{stmt_id} {verb} it for the length of a row, on every "
                        f"work item of {axis}"
                    )
            if everywhere:
                verb = "runs" if len(everywhere) == 1 else "run"
                parts[parts.index("")] = (
                    f"{' and '.join(everywhere)} {verb} in no loop on {axis}, so "
                    "on every work item of it"
                )
            how = "; ".join(parts)
        else:
            one = source_loop == sink_loop
            loops = source_loop if one else f"{source_loop} and {sink_loop}"
            which = (
                f"work items {first} and {second} of it"
                if {source_loop, sink_loop} <= work.known
                else "two work items of it"
            )
            how = (
                f"the {'loop' if one else 'loops'} {loops} on {axis} "
                f"{'runs' if one else 'run'} them on {which}"
            )
        source_verb, sink_verb = dep.verbs()
        cell = _cell_text(dep.source_indices, source_coords, params)
        at = _sizes_text(params, self._sizes)
        message = (
            f"{text} illegal: instance {_instance_text(source_id, source_coords)} "
            f"{source_verb} {dep.array}[{cell}] {sink_verb} by "
            f"{_instance_text(sink_id, sink_coords)} on another work item{at}: "
            f"{how}, and nothing in a kernel orders two work items through "
            "global memory"
        )
        detail = (
            f"a {dep.kind} dependence on {dep.array} joins two work items of {axis}"
        )
        return message, witness, detail

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
        rests_on: tuple[str, ...] = (),
    ) -> Any:
        """One ledger entry for one question about one step.

        ``step`` is the step's recipe, ``(method, args, kwargs)``, and the
        fact's id is :meth:`fact_id` with it as the last step: the steps up to
        this one, and not a count of them, because two schedules of one kernel
        both have a first step and keep facts in one ledger. ``about`` tells
        apart two facts of one kind about one step, such as the exactness of
        two accumulations one ``tag`` reassociates, and is the array's name.

        ``oracle`` is who answered: ``isl`` for the two questions about meaning,
        ``loopy-target`` for the one about what the backend can generate.

        ``status`` is ``"decided"``, ``"refuted"``, or ``"assumed"`` for a
        question the oracle could not answer, which a step that refuses all
        the same records (see :meth:`_definedness_fact`), with no oracle.

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

        ``rests_on`` names the facts the answer takes for granted, which for a
        ``monotone`` cast of a kernel that rewrites its layout are the layout's
        facts (see :attr:`_layout_ids`).
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
        if status in ("refuted", "assumed"):
            provenance["reason"] = reason or detail
        suffix = f":{about}" if about else ""
        return Fact(
            id=self._fact_id("cast", (*self._steps, step), f"{kind}{suffix}"),
            kind=kind,
            statement=statement,
            term=None,
            status=(
                Status.REFUTED
                if status == "refuted"
                else Status.ASSUMED
                if status == "assumed"
                else Status.DECIDED
            ),
            decided_by=None if status == "assumed" else oracle,
            provenance=provenance,
            where=self._term.stmts[0].where if self._term.stmts else "",
            owner=self._term.name,
            rests_on=rests_on,
        )

    def facts(self) -> tuple:
        """The cast facts accumulated by the transformations applied so far."""
        return self._facts


def _key(name: str, target: str, steps: Sequence[tuple[str, tuple, dict]]) -> str:
    """``spmv[c].split('j', 2, inner='j_in', outer='j_out')``: :attr:`Schedule.key`."""
    return name + _steps_text(target, steps)


def _steps_text(target: str, steps: Sequence[tuple[str, tuple, dict]]) -> str:
    """``[c].split('j', 2, inner='j_in', outer='j_out')``: a key after the name."""
    return f"[{target}]" + "".join(f".{_call_text(step)}" for step in steps)


def _call_text(step: tuple[str, tuple, dict]) -> str:
    """One recipe as the call it is, every argument written out.

    An isl map or union map, which is what :meth:`Schedule.affine` records,
    is written as its text, which is what ``affine`` also accepts.
    """
    method, args, kwargs = step

    def shown(value: Any) -> str:
        if isinstance(value, isl.Map | isl.BasicMap | isl.UnionMap):
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


def _not_pointwise(stmt: Stmt, array: str, term: Term) -> str | None:
    """Why :meth:`Schedule.substitute` cannot compute ``stmt`` where it is read.

    ``None`` when it can: an assignment of one cell per instance, at its own
    loop variables, each once (``f[j]``, or ``f[k, j]`` in loops ``j`` and
    ``k``), of an expression with no sum in it, under no guard isl cannot
    state, and reading nothing of ``array``. Then each read ``f[e]`` is the
    expression with the loop variables replaced by ``e``, which is what
    loopy's ``assignment_to_subst`` writes.
    """
    indices = stmt.assignee.indices
    names = [
        index.name if isinstance(index, prim.Variable) else None for index in indices
    ]
    if stmt.kind != "assign":
        return (
            f"{stmt.id} accumulates into {array}, and a substitution computes a "
            "value stored once"
        )
    if None in names or len(set(names)) != len(names) or set(names) != set(stmt.inames):
        shown = ", ".join(str(index) for index in indices)
        return (
            f"{stmt.id} stores {array}[{shown}], and a substitution computes a "
            "statement that stores one cell per instance, at its own loop "
            f"variables ({', '.join(stmt.inames)}), each once"
        )
    if reductions_of(stmt.expr):
        return (
            f"{stmt.id} stores a sum, which a substitution would compute again "
            "at every read of it; only a pointwise statement is substituted"
        )
    if stmt.unnarrowed:
        conjuncts = ", ".join(conjunct for conjunct, _why in stmt.unnarrowed)
        return (
            f"{stmt.id} runs under a guard isl cannot state ({conjuncts}), so "
            "which cells it stores is not known"
        )
    if any(
        kind == "read" and name == array
        for kind, name, _indices, _part in _accesses(stmt, term)
    ):
        return f"{stmt.id} reads {array} as well as storing it"
    return None


def _computed_in(
    term: Term, producer: Stmt, array: str, readers: Sequence[Stmt]
) -> tuple[Any, str | None]:
    """The dtype the lowered code computes ``producer``'s value in, and a reader.

    The dtype is :class:`loopty.promotion.Promotion`'s, the one the lowering
    plans every operation of the value by, or ``None`` where it is not known
    there; :func:`_substituted_kernel` compares it with the dtype ``array``
    is stored in. The reader is the first of ``readers`` that reads
    ``array`` inside a subscript (``y[p[i]]``, or ``x[p[i]]``), or ``None``:
    loopy simplifies a subscript as an affine expression, which a cast is
    not.
    """
    from loopty.promotion import Promotion
    from loopty.trace import accesses_in

    _native, computed = Promotion(term).types(producer.expr)
    for stmt in readers:
        subscripts = [
            stmt.assignee.indices,
            *(
                access.indices
                for source in (stmt.expr, stmt.guard)
                if source is not None
                for access in accesses_in(source)
            ),
        ]
        if any(
            access.array == array
            for indices in subscripts
            for access in accesses_in(tuple(indices))
        ):
            return computed, stmt.id
    return computed, None


def _loopy_dtype(kernel: Any, insn_id: str) -> Any:
    """The dtype loopy reads the value of instruction ``insn_id`` as, or ``None``."""
    from loopy.type_inference import TypeReader

    try:
        typed = lp.infer_unknown_types(kernel, expect_completion=True)
        entry = typed.default_entrypoint
        (found,) = TypeReader(entry, typed.callables_table)(
            entry.id_to_insn[insn_id].expression
        )
        return found.numpy_dtype
    except Exception:  # noqa: BLE001 - any doubt is read as a conversion
        return None


def _substituted_kernel(
    kernel: Any,
    array: str,
    removed: Sequence[str],
    producer: str,
    computed: tuple[Any, str | None] = (None, None),
) -> tuple[Any, str | None]:
    """``kernel`` with ``array`` computed where it is read, or why not.

    The instructions ``removed`` go first: they zero the array where the
    program made it, and loopy's ``assignment_to_subst`` takes an array
    whose every read has one writer before it. That writer, the instruction
    ``producer``, becomes a substitution rule, and loopy drops it, the
    temporary and the loops it leaves empty once no read is left; the loops
    the zeros leave empty go too, which loopy would otherwise warn of.

    A store converts the value to the array's element type: ``u[j] * 0.1``
    of a ``Real`` ``u`` is a double, which a ``float32`` cell rounds, and a
    ``Real`` stored in a ``Nat`` cell is truncated. The rule would hand
    every read the value unconverted. So where the dtype the value is
    computed in (``computed``, from :func:`_computed_in`, or loopy's own
    reading of the instruction where that is not known) is not the array's,
    the producer's value is cast to the array's dtype first (loopy's
    ``TypeCast``), which C converts exactly as it converts a store; a read
    of the array inside a subscript, which loopy cannot simplify through a
    cast, leaves the kernel unwritten, with the reason. Returns the kernel,
    or ``None`` and the reason in words.
    """
    from loopy.symbolic import TypeCast

    value, indexed = computed
    try:
        entry = kernel.default_entrypoint
        dtype = entry.temporary_variables[array].dtype
        stored = None if dtype is None or dtype is lp.auto else dtype.numpy_dtype
        if value is None:
            value = _loopy_dtype(kernel, producer)
        if stored is not None and value != stored:
            if indexed is not None:
                return None, (
                    f"the value {array} is computed from is "
                    f"{value if value is not None else 'of a type not known'}, "
                    f"which storing it as {stored} converts, and {indexed} reads "
                    f"{array} in a subscript, which loopy cannot simplify "
                    "through that conversion; keep it stored"
                )
            entry = entry.copy(
                instructions=[
                    insn.copy(expression=TypeCast(dtype, insn.expression))
                    if insn.id == producer
                    else insn
                    for insn in entry.instructions
                ]
            )
            kernel = kernel.with_kernel(entry)
        if removed:
            loops = {
                name
                for insn in entry.instructions
                if insn.id in removed
                for name in insn.within_inames
            }
            kernel = lp.remove_instructions(kernel, set(removed))
            kernel = lp.remove_unused_inames(kernel, loops)
        kernel = lp.assignment_to_subst(kernel, array)
    except Exception as exc:  # noqa: BLE001 - loopy's refusals are of many kinds
        return None, (
            f"loopy could not compute {array} where it is read: "
            f"{type(exc).__name__}: {exc}"
        )
    return kernel, None


def _shifts(shift: int | tuple[int, ...], depth: int, text: str) -> tuple[int, ...]:
    """The shift of each loop :meth:`Schedule.fuse` fuses, as it was given.

    A number shifts the one loop of a fusion of one loop, and ``0`` every
    loop of any fusion; otherwise there is one number per loop, outermost
    first.
    """
    if isinstance(shift, bool):
        raise TypeError(f"{text}: a shift is a whole number, not {shift!r}")
    if isinstance(shift, int):
        if depth == 1 or shift == 0:
            return (shift,) * depth
        raise ValueError(
            f"{text}: the fusion fuses {depth} loops, so it takes one shift per "
            f"loop, outermost first, such as shift=({shift}, 0)"
        )
    values = tuple(shift)
    if len(values) != depth or not all(
        isinstance(value, int) and not isinstance(value, bool) for value in values
    ):
        raise ValueError(
            f"{text}: the fusion fuses {depth} loop{'s' if depth > 1 else ''}, "
            f"so it takes {depth} whole number{'s' if depth > 1 else ''} as its "
            f"shift, outermost first, not {shift!r}"
        )
    return values


def _check_factor(factor: int) -> None:
    """A split or tile factor has to be a positive integer."""
    if factor < 1:
        raise ValueError(f"a split factor must be positive, not {factor}")


class _Inexpressible(ValueError):
    """A reindexing the kernel rewrite cannot write for loopy, and why."""


def _affine_kernel(
    kernel: Any,
    mapping: isl.Map,
    pieces: Mapping[str, isl.Map] | None = None,
    sizes: Sequence[str] = (),
) -> tuple[Any, str | None]:
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

    ``pieces``, when given, maps each instruction in the mapped loops to a
    map of its own, all of them over the loops ``mapping`` names on both
    sides. Maps that are all one map are that map. Otherwise loopy still
    gives the new loops one domain, since the statements share them, and the
    rewrite gives each instruction back its own part of it (see
    :func:`_shared_image`): the domain is the union of the images, or the
    polyhedral hull of the union when that is not one basic set, and in each
    instruction the old loops are replaced by its own inverse, and it is
    predicated on its own image, such as ``(1 + a + b) mod 2 = 0`` for a
    statement moved half a step along the diamond. A domain nested in the
    loops moves along the map of the instructions that run in it, and has to
    have one, and so does the instruction that computes a ragged row's length
    for the fiber of those instructions (see :func:`_with_bound_maps`).

    Maps per statement that take different loops to the same new ones fuse
    those loops (see :meth:`Schedule.fuse`), and each statement's loops are
    then a domain of their own: two loops in sequence. The new loops get one
    domain all the same, the union of every statement's image as above, in
    place of the domains the maps take, and a domain nested in one of those
    moves along the map of the statements in it (see :func:`_fused_plan`).

    The map is the same object the checker reasons about, so the two cannot
    drift apart. What cannot be written this way comes back as the reason, and
    the kernel unchanged: loops that no one domain defines, an image that is
    not one basic set, an inverse that is piecewise, an instruction in some of
    the mapped loops and not the others, and, with maps per statement, a
    nested domain whose instructions move by different maps, a row's length
    that bounds fibers whose statements do, or an instruction in the loops
    that is no statement's and bounds none of their loops; and, for maps
    that fuse, loops that share a domain with a loop no map takes, and that
    are not its outer ones (see :func:`_cut_for`).
    """
    from loopy.match import Id, parse_stack_match
    from loopy.symbolic import (
        RuleAwareSubstitutionMapper,
        SubstitutionRuleMappingContext,
    )
    from pymbolic.mapper.substitutor import make_subst_func

    entry = kernel.default_entrypoint
    taken = (
        set()
        if pieces is None
        else {_dim_names(piece, isl.dim_type.in_) for piece in pieces.values()}
    )
    try:
        if len(taken) > 1:
            assert pieces is not None
            domains, insns, substitutions, predicates = _fused_plan(
                entry, pieces, sizes
            )
        else:
            domains, insns, substitutions, predicates = _shared_plan(
                entry, mapping, pieces, sizes
            )
    except _Inexpressible as exc:
        return kernel, str(exc)
    except isl.Error as exc:
        return kernel, f"isl could not rewrite the kernel along the map: {exc}"

    # loopy's own transforms refuse to remap an iname a loop priority
    # mentions, and the old priority names loops that no longer exist. The
    # caller sets the priority to the nest it has just checked.
    entry = entry.copy(
        domains=domains, instructions=insns, loop_priority=frozenset()
    )
    for key, substitution in substitutions.items():
        context = SubstitutionRuleMappingContext(
            entry.substitutions, entry.get_var_name_generator()
        )
        mapper = RuleAwareSubstitutionMapper(
            context,
            make_subst_func(substitution),
            within=parse_stack_match(None if key is None else Id(key)),
        )
        entry = context.finish_kernel(
            mapper.map_kernel(entry, map_args=False, map_tvs=False)
        )
    if predicates:
        # Added after the substitution, because they are written in the new
        # loops already, and a new loop may keep the name of an old one.
        entry = entry.copy(
            instructions=[
                insn.copy(predicates=_first(predicates[insn.id], insn.predicates))
                if insn.id in predicates
                else insn
                for insn in entry.instructions
            ]
        )
    return kernel.with_kernel(entry), None


#: What a plan of :func:`_affine_kernel` gives: the new domains, the
#: instructions in their new loops, each instruction's substitution of its old
#: loops (``None`` for every instruction), and the predicate of each that runs
#: at fewer points than the loops it shares.
_Plan = tuple[list[Any], list[Any], dict[Any, dict[str, Any]], dict[str, Any]]


def _shared_plan(
    entry: Any,
    mapping: isl.Map,
    pieces: Mapping[str, isl.Map] | None,
    sizes: Sequence[str] = (),
) -> _Plan:
    """The rewrite along one map, or maps per statement over the same loops.

    One domain defines every loop the map takes, and the new loops replace
    them there; see :func:`_affine_kernel`. Raises :class:`_Inexpressible`
    with what cannot be written.
    """
    inputs = _dim_names(mapping, isl.dim_type.in_)
    outputs = _dim_names(mapping, isl.dim_type.out)
    mapped = set(inputs)
    loops = ", ".join(inputs)

    homes = [
        k
        for k, domain in enumerate(entry.domains)
        if mapped & set(domain.get_var_names(isl.dim_type.set))
    ]
    if len(homes) != 1 or not mapped <= set(
        entry.domains[homes[0]].get_var_names(isl.dim_type.set)
    ):
        raise _Inexpressible(
            f"the loops {loops} are not all defined by one loopy domain, and "
            "the kernel is rewritten along a map only in the domain that "
            "defines every loop the map replaces"
        )
    home = homes[0]
    before = entry.domains[home]
    if pieces is not None:
        first = next(iter(pieces.values()))
        if all(piece.is_equal(first) for piece in pieces.values()):
            mapping, pieces = first, None

    insns = []
    for insn in entry.instructions:
        inside = mapped & insn.within_inames
        if inside and inside != mapped:
            raise _Inexpressible(
                f"instruction {insn.id} runs in {', '.join(sorted(inside))} "
                f"and not in all of {loops}"
            )
        if inside:
            insn = insn.copy(
                within_inames=(insn.within_inames - mapped) | set(outputs)
            )
        insns.append(insn)

    if pieces is not None:
        pieces = _with_bound_maps(entry, mapped, pieces)
    domains = list(entry.domains)
    predicates: dict[str, Any] = {}
    substitutions: dict[Any, dict[str, Any]]
    if pieces is None:
        domains[home] = _image(before, mapping)
        substitutions = {None: _substitution(mapping, before)}
    else:
        images = {key: _image_set(before, piece) for key, piece in pieces.items()}
        domains[home], predicates = _shared_image(images, before, sizes)
        substitutions = {
            key: _substitution(piece, before) for key, piece in pieces.items()
        }
    for k, domain in enumerate(entry.domains):
        if k != home and mapped & set(domain.get_var_names(isl.dim_type.param)):
            moved = mapping if pieces is None else _governing(entry, domain, pieces)
            domains[k] = _image_of_params(domain, moved)
    return domains, insns, substitutions, predicates


def _fused_plan(
    entry: Any, pieces: Mapping[str, isl.Map], sizes: Sequence[str] = ()
) -> _Plan:
    """The rewrite along maps per statement that fuse loops of two domains.

    The maps take different loops to the same new ones, and the statements
    of each take loops that one domain defines and no other loop: two loops
    the lowering wrote one after the other, ``{ [j] }`` and ``{ [i] }``, or
    the outer loops of a domain, which is cut after them first
    (:func:`_cut_for`).
    Each such domain goes, and the new loops get one domain in place of the
    first of them, the union of every statement's image under its own map,
    or its polyhedral hull, with each statement predicated on its own image
    and given its own inverse, as for maps per statement over one domain
    (see :func:`_shared_image`). A domain nested in the loops a map takes
    moves along that map; the list is then nested again, since a domain
    nested in the second loop now hangs from the first
    (:func:`loopty.lower._nest_domains`). Raises :class:`_Inexpressible`
    with what cannot be written.
    """
    from loopty.lower import _nest_domains

    outputs = _dim_names(next(iter(pieces.values())), isl.dim_type.out)
    groups: dict[tuple[str, ...], list[str]] = {}
    for key, piece in pieces.items():
        groups.setdefault(_dim_names(piece, isl.dim_type.in_), []).append(key)
    mapped = {name for inputs in groups for name in inputs}
    entry = entry.copy(domains=_cut_for(entry.domains, groups))
    homes: dict[tuple[str, ...], int] = {}
    for inputs in groups:
        defining = [
            k
            for k, domain in enumerate(entry.domains)
            if set(inputs) & set(domain.get_var_names(isl.dim_type.set))
        ]
        names = (
            set(entry.domains[defining[0]].get_var_names(isl.dim_type.set))
            if len(defining) == 1
            else set()
        )
        if names != set(inputs):
            raise _Inexpressible(
                f"the loops {', '.join(inputs)} are not a loopy domain of their "
                "own, and maps that fuse loops rewrite the kernel only where "
                "each statement's loops are one domain with no other loop in it"
            )
        homes[inputs] = defining[0]

    pieces = _with_bound_maps(entry, mapped, pieces)
    home_of = {
        key: homes[_dim_names(piece, isl.dim_type.in_)] for key, piece in pieces.items()
    }
    insns = []
    for insn in entry.instructions:
        inside = mapped & insn.within_inames
        if not inside:
            insns.append(insn)
            continue
        piece = pieces.get(insn.id)
        if piece is None or inside != set(_dim_names(piece, isl.dim_type.in_)):
            raise _Inexpressible(
                f"instruction {insn.id} runs in {', '.join(sorted(inside))}, "
                "which no one map of the step takes"
            )
        insns.append(
            insn.copy(within_inames=(insn.within_inames - inside) | set(outputs))
        )

    images = {
        key: _image_set(entry.domains[home_of[key]], piece)
        for key, piece in pieces.items()
    }
    common = next(iter(images.values()))
    for image in images.values():
        common = common.align_params(image.get_space())
    images = {
        key: image.align_params(common.get_space()) for key, image in images.items()
    }
    first = min(homes.values())
    shared, predicates = _shared_image(images, entry.domains[first], sizes)
    substitutions: dict[Any, dict[str, Any]] = {
        key: _substitution(piece, entry.domains[home_of[key]])
        for key, piece in pieces.items()
    }
    domains: list[Any] = []
    for k, domain in enumerate(entry.domains):
        if k in homes.values():
            if k == first:
                domains.append(shared)
            continue
        if mapped & set(domain.get_var_names(isl.dim_type.param)):
            domain = _image_of_params(domain, _governing(entry, domain, pieces))
        domains.append(domain)
    return _nest_domains(domains), insns, substitutions, predicates


def _cut_for(domains: Sequence[Any], groups: Iterable[Sequence[str]]) -> list[Any]:
    """``domains``, with each one whose outer loops a map takes cut after them.

    A fusion of the outer loop of a nest, ``{ [i, j] }``, with a loop of one
    level, ``{ [w] }``, takes ``i`` and not ``j``, and the two share one
    domain because no statement of the kernel left the nest between them.
    The domain is cut as the lowering cuts it when one does
    (:func:`loopty.lower._statement_domains`): ``{ [i] }`` over the loops the
    map takes, which are the first of the domain's, and ``[i] -> { [j] }``,
    nested in it, with every constraint of the domain. A domain whose loops
    a map takes are not its first ones is left as it is, and refused by
    :func:`_fused_plan`.
    """
    from loopty.lower import _outer_part

    out = list(domains)
    for inputs in groups:
        defining = [
            k
            for k, domain in enumerate(out)
            if set(inputs) & set(domain.get_var_names(isl.dim_type.set))
        ]
        if len(defining) != 1:
            continue
        (k,) = defining
        names = list(out[k].get_var_names(isl.dim_type.set))
        keep = len(inputs)
        if len(names) <= keep or set(names[:keep]) != set(inputs):
            continue
        outer = _outer_part(out[k], keep)
        inner = out[k].move_dims(
            isl.dim_type.param,
            out[k].dim(isl.dim_type.param),
            isl.dim_type.set,
            0,
            keep,
        )
        out[k : k + 1] = [outer, inner]
    return out


def _first(own: Any, others: frozenset[Any]) -> frozenset[Any]:
    """The predicates ``own`` and ``others``, with ``own`` evaluated first.

    loopy joins an instruction's predicates with ``&&`` in no particular
    order, and a ``when`` guard among them may read an array. At a point of
    the shared loops that is another statement's, this instruction's inverse
    names a loop value it does not have, so each other predicate that could
    fault there, by reading an array, calling a function or dividing by
    something that is not a number, becomes ``own and it``, which C
    evaluates left to right; one that only compares loop variables and sizes
    is safe anywhere and stays as it is. ``own`` stays a predicate of its
    own: loopy's bounds check reads the instruction's domain off the
    predicates it can turn into sets, and skips one it cannot, such as a
    conjunction with a guard that reads an array.
    """
    from loopty.lower import walk

    def could_fault(node: Any) -> bool:
        if isinstance(node, prim.Subscript | prim.Call):
            return True
        return isinstance(
            node, prim.FloorDiv | prim.Quotient | prim.Remainder
        ) and not isinstance(node.denominator, int)

    return frozenset(
        [
            own,
            *(
                prim.LogicalAnd((own, other))
                if any(could_fault(node) for node in walk(other))
                else other
                for other in others
            ),
        ]
    )


def _substitution(mapping: isl.Map, domain: Any) -> dict[str, Any]:
    """Each old loop as the expression of the new ones that ``mapping`` gives."""
    from loopy.symbolic import pw_aff_to_expr

    inverse = _inverse(mapping, domain)
    out = {}
    for k, name in enumerate(_dim_names(mapping, isl.dim_type.in_)):
        piece = inverse.get_pw_aff(k).coalesce()
        if piece.n_piece() != 1:
            raise _Inexpressible(
                f"the inverse of the map is piecewise in {name} ({piece}), "
                "and a loop variable is replaced by one expression"
            )
        out[name] = pw_aff_to_expr(piece)
    return out


def _shared_image(
    images: Mapping[str, isl.Set], before: Any, sizes: Sequence[str] = ()
) -> tuple[isl.BasicSet, dict[str, Any]]:
    """One domain for loops whose statements move by maps of their own.

    loopy gives a loop one domain, and a statement one set of loops, so two
    statements that interleave in the new loops have to share them, and the
    loops have to run over every point either statement has. The domain is
    the union of the images when isl coalesces it into one basic set, which
    keeps a lattice both images lie on (the diamond's parity, say) for
    :func:`_stepped` to step over, and the polyhedral hull of the union
    otherwise, which only over-approximates. Each instruction whose image is
    less than that domain gets it back as a predicate, the gist of its image
    in the domain as loopy writes a condition, as the lowering cuts back a
    statement a ``when`` narrows; a gist loopy cannot write is refused rather
    than dropped, since the instruction would then run at points it does not
    have.

    The hull of two loops that run to two sizes, ``{ [j] : 0 <= j < n }``
    and ``{ [j] : 0 <= j < m }`` fused, has no upper bound where a size may
    be negative, and loopy can write no loop over it. ``sizes``, the
    program's sizes, are never negative (the contract refuses an argument
    that would make one so, :func:`loopty.contract.sizes_not_negative`), and
    the hull is taken where they are not: ``j < n + m``, the statements
    predicated on their own images within it. A hull that is unbounded all
    the same is refused, with the union.

    Returns the domain and the predicate of each instruction that needs one.
    """
    from loopy.symbolic import set_to_cond_expr

    union: isl.Set | None = None
    for image in images.values():
        union = image if union is None else union.union(image)
    assert union is not None
    union = union.coalesce()
    if union.n_basic_set() == 1:
        (shared,) = union.get_basic_sets()
    else:
        shared = union.polyhedral_hull()
        known = [
            name
            for name in sizes
            if name in union.get_var_names(isl.dim_type.param)
        ]
        if not shared.is_bounded() and known:
            context = isl.Set(
                f"[{', '.join(known)}] -> "
                f"{{ : {' and '.join(f'{name} >= 0' for name in known)} }}"
            )
            shared = union.intersect_params(
                context.align_params(union.get_space())
            ).polyhedral_hull()
        if not shared.is_bounded():
            raise _Inexpressible(
                f"the new loops run over every point of the images, {union}, "
                "and no one loopy domain bounds them: their hull is unbounded"
            )
    wide = _as_set(shared)
    predicates: dict[str, Any] = {}
    for key, image in images.items():
        narrow = image.align_params(wide.get_space())
        if narrow.is_equal(wide.align_params(narrow.get_space())):
            continue
        extra = narrow.gist(wide.align_params(narrow.get_space()))
        try:
            predicates[key] = set_to_cond_expr(extra)
        except Exception as exc:  # noqa: BLE001 - loopy's words for it vary
            raise _Inexpressible(
                f"the image of the domain {before} for {key} is {image}, "
                f"narrower than the domain its loops share, {shared}, and the "
                f"difference cannot be written as a condition ({exc})"
            ) from exc
    return shared, predicates


def _with_bound_maps(
    entry: Any, mapped: set[str], pieces: Mapping[str, isl.Map]
) -> dict[str, isl.Map]:
    """``pieces``, with a map for every other instruction in the mapped loops.

    Such an instruction is no statement's: it computes the length of a
    ragged row, which bounds the fiber of the statements inside it, in the
    row's loop, once per row. It moves along the map of the statements whose
    loops it bounds, which is where the checker counts its read, as part of
    each statement over the loops up to the row, and it has to have one. An
    instruction that bounds nothing is refused, and so is one whose
    statements move by different maps.
    """
    out = dict(pieces)
    for insn in entry.instructions:
        if insn.id in out or not mapped & insn.within_inames:
            continue
        written = set(insn.assignee_var_names())
        bounded = [
            domain
            for domain in entry.domains
            if written & set(domain.get_var_names(isl.dim_type.param))
        ]
        if not bounded:
            raise _Inexpressible(
                f"instruction {insn.id} runs in {', '.join(sorted(mapped))} "
                "and belongs to no statement, so it has no map of its own, "
                "and the statements in those loops move by different maps"
            )
        maps = [_governing(entry, domain, pieces) for domain in bounded]
        if not all(m.is_equal(maps[0]) for m in maps):
            # Two fibers of one row, whose statements move apart.
            raise _Inexpressible(
                f"instruction {insn.id} bounds loops whose statements move by "
                "different maps, and it runs once per row"
            )
        out[insn.id] = maps[0]
    return out


def _governing(entry: Any, domain: Any, pieces: Mapping[str, isl.Map]) -> isl.Map:
    """The map a domain nested in the mapped loops moves along.

    The one map of the instructions that run in it, as loops or as the loops
    of a reduction; a domain that holds instructions moving by different maps
    would have to be two domains, one per map, which is refused.
    """
    names = set(domain.get_var_names(isl.dim_type.set))
    users = [
        insn.id
        for insn in entry.instructions
        if names & (set(insn.within_inames) | set(insn.reduction_inames()))
    ]
    if not users:  # pragma: no cover - a domain holds a loop some instruction runs in
        return next(iter(pieces.values()))
    maps = [pieces[user] for user in users if user in pieces]
    if len(maps) != len(users) or not all(m.is_equal(maps[0]) for m in maps):
        raise _Inexpressible(
            f"the domain {domain} is nested in the mapped loops and holds "
            f"{', '.join(users)}, which move by different maps, and loopy "
            "gives the loops it defines one domain"
        )
    return maps[0]


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
    return _one_basic_set(_image_set(domain, mapping), domain)


def _image_set(domain: Any, mapping: isl.Map) -> isl.Set:
    """:func:`_image`, as the set isl computes, one basic set or not."""
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
    return image


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


# {{{ loops over a lattice


def _ranked(names: Sequence[str], order: Sequence[str]) -> list[str]:
    """``names`` outermost first: in ``order``, then the rest as they come."""
    rank = {name: k for k, name in enumerate(order)}
    return sorted(
        names, key=lambda name: (rank.get(name, len(order)), names.index(name))
    )


def _stride_of(domain: Any, name: str, inner: Sequence[str]) -> tuple[int, Any]:
    """The stride of ``name`` in ``domain`` given its outer loops, and its offset.

    The loops inside it are projected out first, so that the offset is an
    affine function of the parameters and the loops outside it only; isl's
    ``get_stride_info`` would otherwise express it in any of the others.
    """
    projected = _as_set(domain)
    for other in inner:
        dims = projected.get_var_names(isl.dim_type.set)
        if other in dims:
            projected = projected.project_out(
                isl.dim_type.set, dims.index(other), 1
            )
    info = projected.get_stride_info(
        projected.get_var_names(isl.dim_type.set).index(name)
    )
    return info.get_stride().to_python(), info.get_offset()


def _stepped(
    kernel: Any, order: Sequence[str], tags: Mapping[str, str]
) -> tuple[Any, dict[str, str]]:
    """``kernel`` with each loop over a lattice written as a count of steps.

    The image of a map that is not unimodular has holes: the diamond's is the
    points where ``a + b`` is even, which isl states as an existentially
    quantified constraint. loopy loops over such a domain's bounding box and
    tests the constraint with an ``if`` in the innermost loop, so that half
    the iterations of that loop do nothing (note 13 of
    ``docs/loopy-notes.md``). Here every loop is asked, outermost first,
    whether isl finds a stride for it given the loops outside it: ``b`` steps
    by 2 from ``-a``. Such a loop is replaced by a counter, ``b = 2*b_step -
    a``, in the domain (whose preimage has no holes left) and in every
    instruction, and the new loop runs over the counter. For fixed values of
    the loops outside it, the counter and the loop it replaces increase
    together and meet the same points, so the kernel runs the same instances
    in the same order: the change of variables is the one the order the
    checker approved already fixes, and isl confirms, for each loop, that the
    new domain maps back onto the old one exactly.

    Done last, on the kernel code is generated from, and not in the kernel
    later steps transform: a tile of ``b`` splits the loop the checker knows,
    not the counter. A loop is left as loopy has it when it carries a tag
    (a hardware axis is not a loop loopy steps through), is the loop of a
    reduction, is named by another domain as a parameter (a fiber nested in
    it), or has a stride whose offset involves a loop that does not run
    around every instruction in it.

    Returns the kernel, and for each loop replaced the expression it became,
    as text: ``{"b": "2*b_step - a"}``.
    """
    if kernel is None:
        return None, {}
    from loopy.match import parse_stack_match
    from loopy.symbolic import (
        RuleAwareSubstitutionMapper,
        SubstitutionRuleMappingContext,
        aff_to_expr,
        get_dependencies,
    )
    from pymbolic.mapper.substitutor import make_subst_func

    entry = kernel.default_entrypoint
    within: dict[str, set[str]] = {}
    folded: set[str] = set()
    for insn in entry.instructions:
        for name in insn.within_inames:
            within.setdefault(name, set()).add(insn.id)
        folded |= set(insn.reduction_inames())
    nested: set[str] = set()
    for domain in entry.domains:
        nested |= set(domain.get_var_names(isl.dim_type.param))
    fresh = entry.get_var_name_generator()
    domains = list(entry.domains)
    substitution: dict[str, Any] = {}
    renamed: dict[str, str] = {}
    shown: dict[str, str] = {}
    for position, domain in enumerate(domains):
        params = set(domain.get_var_names(isl.dim_type.param))
        ranked = _ranked(list(domain.get_var_names(isl.dim_type.set)), order)
        for k, name in enumerate(ranked):
            if name in tags or name not in within or name in folded | nested:
                continue
            try:
                stride, offset = _stride_of(domain, name, ranked[k + 1 :])
            except isl.Error:  # pragma: no cover - isl finds no stride
                continue
            if stride <= 1:
                continue
            around = {
                renamed.get(outer, outer)
                for outer in ranked[:k]
                if within[name] <= within.get(outer, set())
            }
            value = aff_to_expr(offset)
            if not set(get_dependencies(value)) - params <= around:
                continue
            counter = fresh(f"{name}_step")
            value = stride * prim.Variable(counter) + value
            counted = _counted(domain, name, counter, value)
            if counted is None:
                continue
            domain = counted
            substitution[name] = value
            renamed[name] = counter
            shown[name] = _stepped_text(stride, counter, offset)
        domains[position] = domain
    if not renamed:
        return kernel, {}
    insns = [
        insn.copy(
            within_inames=frozenset(renamed.get(n, n) for n in insn.within_inames)
        )
        if insn.within_inames & set(renamed)
        else insn
        for insn in entry.instructions
    ]
    priority = frozenset(
        tuple(renamed.get(name, name) for name in nest)
        for nest in entry.loop_priority
    )
    entry = entry.copy(domains=domains, instructions=insns, loop_priority=priority)
    context = SubstitutionRuleMappingContext(
        entry.substitutions, entry.get_var_name_generator()
    )
    mapper = RuleAwareSubstitutionMapper(
        context, make_subst_func(substitution), within=parse_stack_match(None)
    )
    entry = context.finish_kernel(
        mapper.map_kernel(entry, map_args=False, map_tvs=False)
    )
    return kernel.with_kernel(entry), shown


def _counted(domain: Any, name: str, counter: str, value: Any) -> Any:
    """``domain`` over ``counter`` in place of ``name``, where ``name = value``.

    The preimage of the domain under the change of variables, which isl
    confirms maps back onto the domain exactly. ``None`` when it does not, or
    is not one basic set.
    """
    from loopy.symbolic import aff_from_expr

    before = _as_set(domain)
    names = list(before.get_var_names(isl.dim_type.set))
    space = before.get_space().set_dim_name(
        isl.dim_type.set, names.index(name), counter
    )
    try:
        listed = isl.AffList.alloc(space.get_ctx(), len(names))
        for other in names:
            listed = listed.add(
                aff_from_expr(space, value if other == name else prim.Variable(other))
            )
        function = isl.MultiAff.from_aff_list(
            space.map_from_domain_and_range(before.get_space()), listed
        )
        after = before.preimage_multi_aff(function).coalesce()
        if not after.apply(isl.Map.from_multi_aff(function)).is_equal(before):
            return None  # pragma: no cover - isl's stride holds by construction
    except isl.Error:  # pragma: no cover - defensive against isl declining
        return None
    if after.n_basic_set() != 1:  # pragma: no cover - a preimage of a basic set
        return None
    (out,) = after.get_basic_sets()
    return out


def _stepped_text(stride: int, counter: str, offset: Any) -> str:
    """``2*b_step - a``: a loop as its counter, the way a reader writes it."""
    from loopy.symbolic import aff_to_expr

    if offset.dim(isl.dim_type.div) or offset.get_denominator_val().to_python() != 1:
        return f"{stride}*{counter} + {aff_to_expr(offset)}"
    terms = [(stride, counter)]
    for kind in (isl.dim_type.in_, isl.dim_type.param):
        for k in range(offset.dim(kind)):
            coefficient = offset.get_coefficient_val(kind, k).to_python()
            if coefficient:
                terms.append((coefficient, offset.get_dim_name(kind, k)))
    out = ""
    for coefficient, name in terms:
        magnitude = "" if abs(coefficient) == 1 else f"{abs(coefficient)}*"
        if not out:
            out = f"{'-' if coefficient < 0 else ''}{magnitude}{name}"
        else:
            out += f" {'-' if coefficient < 0 else '+'} {magnitude}{name}"
    constant = offset.get_constant_val().to_python()
    if constant:
        out += f" {'-' if constant < 0 else '+'} {abs(constant)}"
    return out


# }}}
