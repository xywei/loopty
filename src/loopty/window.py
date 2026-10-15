"""Window contraction: an array kept in the cells that are live at once.

An array a program makes (``f = Arr.zeros_like(u)``) is stored in full, from
the call that writes it to the calls that read it. Once a fusion runs the
reader a step behind the writer, only a few cells of it are live at any step:
the divergence of the Burgers flux, fused one step behind the flux, reads
``f[j]`` and ``f[j - 2]`` at step ``j``, so the cells ``f[j - 2]``, ``f[j - 1]``
and ``f[j]`` are all that has to be kept, and ``f`` can live in three slots,
``f[j]`` in slot ``j % 3``. That is window contraction, and
:meth:`loopty.schedule.Schedule.contract` is the step.

A *placement* says where each cell is kept: a function from the indices of a
cell to the indices of its slot, written in Python, ``lambda j: j % 3``. It
is quasi-affine, a sum of the indices times numbers with ``//`` and ``%`` by
a number, so isl can state it, and so can Lean. The cells of an array over a
domain are the domain's points, and its placement takes the domain's
binders: sumpy's L2P keeps one box's triangle of ``R`` over ``Where[i:
Fin(nb), a: Fin(11), b: Fin(11), a + b <= 10]`` as ``lambda i, a, b: (a,
b)``, in the 11 x 11 box the triangle spans.

The array may be stored by several statements (a copy where ``a < 2`` and a
recurrence elsewhere), and a statement that stores it may read it, as the
recurrence ``R[i, a, b] = -R[i, a - 2, b + 2]`` does. Whether a placement is
safe is a question about the schedule as it stands, and isl decides it: *no
read sees its slot overwritten*. A read of a cell ``c``, stored by an
instance ``w``, is safe unless an instance ``w'`` stores a cell ``c' != c``
in the same slot after ``w`` and before the read. The question is the
emptiness of

    { (r, w') : exists w: r reads the cell w stores,
                slot(cell(w')) = slot(cell(w)), cell(w') != cell(w),
                w before w' before r }

with "before" the order the schedule runs and ``w`` and ``w'`` instances of
any statement that stores the array, and Lean states it from loopty's own
description as a universal over three instances: their domains, the cells
their accesses name, the placement, and the time vectors of the steps taken.
Together with the strict definedness of the array (every cell read was
stored before the read) it is what makes the window compute what the array
did.

The window is kept safe by the steps that follow: the pairs of accesses that
touch one slot, at least one of them a write, are dependences of the schedule
from then on, in the order it ran them when the window was taken (see
:func:`slot_dependences`). A later step that runs one of them backwards is
refused with its pair, as every cast is. A window of a number of cells is
each work item's own, as loopy keeps a private temporary, so two accesses
on two work items touch two windows and need no order: the boxes of the L2P
on a group axis each keep a triangle. A read is refused there where the
store of its cell is on another work item. A window of a size, ``f_window[n]``, is
one in global memory on OpenCL, for every work item, and a step that puts
two accesses of one slot of it on two work items is refused with the pair.

A window keeps a value stored once per cell. An update, ``f[j] += v`` or a
scatter-add ``f[cell[c, k]] += v``, adds to the zeros the array was made
with, which the window drops, and a slot holds another cell when the next
cell's first update comes, so a window of an array that is updated is
refused, naming the fix: store each cell once (a ``reduce_sum`` where the
updates run over a loop), or keep the array whole. An update of another
array may read a window, ``y[node[i]] += f[i + 1] - f[i - 1]``: its updates
commute with each other as any update's do, and the window's slots keep
their order all the same, which :meth:`~loopty.schedule.Schedule.atomic`
does not lift.

Two lowerings. Modulo addressing, on every target: the array becomes a
temporary of the window's shape, ``f_window[j % 3]``, and every access goes
through the placement. A shift register, for registers on a device: the
window along one loop, every access at a fixed distance from it
(``f[j]``, ``f[j - 2]``), is a row of registers shifted by one at each step,
``f_window[0] = f_window[1]; f_window[1] = f_window[2]``, so that every access
has a constant index, which a GPU keeps in registers where an index computed
at run time puts the array in memory.

This is #970's ``compute`` reuse, the relation between the slots one step
reads and the ones the next reuses, with the check ``compute`` leaves out.
"""

from __future__ import annotations

import inspect
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

import islpy as isl
import pymbolic.primitives as prim
from pymbolic.mapper.substitutor import substitute

from loopty import flow, idx
from loopty.term import Stmt, Term

__all__ = [
    "Access",
    "Placement",
    "WindowError",
    "accesses",
    "clobbers",
    "earlier",
    "infer",
    "live_pairs",
    "placement_of",
    "window_accesses",
]


class WindowError(ValueError):
    """A contraction that cannot be asked, with what stands in the way."""


# {{{ the placement


def _render(expr: Any) -> str:
    """A slot's expression as Python writes it: ``j % 3``, ``2*b + 1``."""
    return str(expr)


@dataclass(frozen=True, eq=False)
class Placement:
    """Where each cell of a contracted array is kept: a slot of its window.

    ``names`` name the array's indices, one per axis, and ``slots`` are the
    slot's indices as terms over them. Its repr is the Python it reads as,
    ``lambda j: j % 3``, which is how a schedule's steps and the ids of its
    facts name it.
    """

    names: tuple[str, ...]
    slots: tuple[Any, ...]

    @property
    def text(self) -> str:
        """``lambda j: j % 3``: the placement as the Python it is."""
        if len(self.slots) == 1:
            body = _render(self.slots[0])
        else:
            body = "(" + ", ".join(_render(slot) for slot in self.slots) + ")"
        return f"lambda {', '.join(self.names)}: {body}"

    def __repr__(self) -> str:
        return self.text

    def __str__(self) -> str:
        return self.text

    def __eq__(self, other: object) -> bool:
        return isinstance(other, Placement) and other.text == self.text

    def __hash__(self) -> int:
        return hash(self.text)

    def at(self, indices: Sequence[Any]) -> tuple[Any, ...]:
        """The slot of the cell ``indices``, as terms in what they are in."""
        mapping = {name: index for name, index in zip(self.names, indices, strict=True)}
        return tuple(substitute(slot, mapping) for slot in self.slots)

    def slot_text(self, indices: Sequence[Any] | None = None) -> str:
        """``j % 3``, or ``(a, b)``: the slot of a cell, in words for a message."""
        slots = self.slots if indices is None else self.at(indices)
        if len(slots) == 1:
            return _render(slots[0])
        return "(" + ", ".join(_render(slot) for slot in slots) + ")"

    def isl_map(self) -> isl.Map:
        """``{ [c0, c1] -> [slot0] }``: the placement as an isl map."""
        cells = [f"c{k}'" for k in range(len(self.names))]
        renaming = {
            name: prim.Variable(cell)
            for name, cell in zip(self.names, cells, strict=True)
        }
        texts = [idx.isl_expr(substitute(slot, renaming)) for slot in self.slots]
        return isl.Map(f"{{ [{', '.join(cells)}] -> [{', '.join(texts)}] }}")


def placement_of(given: Any, rank: int, array: str, text: str) -> Placement:
    """The placement an author gave, read off the Python function it is.

    ``given`` takes one index per axis of ``array`` and returns the slot's
    index, or a tuple of them: ``lambda j: j % 3``, ``lambda e, a, b: b``. It
    is called once with symbolic indices, so what it returns has to be
    quasi-affine in them, a sum of the indices times numbers with ``//`` and
    ``%`` by a number, which is what isl and Lean can state; anything else is
    a :class:`WindowError` naming the slot.
    """
    if isinstance(given, Placement):
        if len(given.names) != rank:
            raise WindowError(
                f"{text}: the placement {given} takes {len(given.names)} indices, "
                f"and {array} has {rank} axes"
            )
        return given
    example = "lambda j: j % 3" if rank == 1 else "lambda e, a, b: b"
    if not callable(given):
        raise TypeError(
            f"{text}: a placement is a function from the indices of a cell of "
            f"{array} to the indices of its slot, as {example} is, not {given!r}"
        )
    try:
        parameters = list(inspect.signature(given).parameters.values())
    except (TypeError, ValueError) as exc:
        raise TypeError(
            f"{text}: the placement has no signature Python can read ({exc}); "
            f"write it as a lambda, {example}"
        ) from None
    positional = (
        inspect.Parameter.POSITIONAL_ONLY,
        inspect.Parameter.POSITIONAL_OR_KEYWORD,
    )
    if len(parameters) != rank or any(p.kind not in positional for p in parameters):
        raise TypeError(
            f"{text}: the placement takes one index per axis of {array}, "
            f"{rank} of them, as {example} does"
        )
    names = tuple(parameter.name for parameter in parameters)
    try:
        result = given(*(prim.Variable(name) for name in names))
    except Exception as exc:  # noqa: BLE001 - said in the author's words
        raise WindowError(
            f"{text}: the placement could not be computed on symbolic indices: "
            f"{type(exc).__name__}: {exc}; write it with +, -, * by a number, "
            "// and % by a number"
        ) from None
    slots = tuple(result) if isinstance(result, tuple | list) else (result,)
    if not slots:
        raise WindowError(f"{text}: the placement gives a slot no index")
    for slot in slots:
        if isinstance(slot, bool) or not (isinstance(slot, int) or idx.is_affine(slot)):
            raise WindowError(
                f"{text}: the slot {slot} is not a sum of the indices times "
                "numbers with // and % by a number, which isl and Lean can state"
            )
        free = {node.name for node in _walk(slot) if isinstance(node, prim.Variable)}
        if not free <= set(names):
            stray = ", ".join(sorted(free - set(names)))
            raise WindowError(
                f"{text}: the slot {slot} names {stray}, which is not an index of "
                f"{array}; a window's slots are numbers, so write the sizes as "
                "numbers"
            )
    return Placement(names, slots)


def _walk(expr: Any) -> list[Any]:
    """Every node of an expression, the expression first, none of them hashed.

    A guard is a lanky proposition, whose ``==`` is symbolic, so the nodes
    are listed and never put in a set.
    """
    from lanky.terms import init_args

    out: list[Any] = []
    stack = [expr]
    while stack:
        node = stack.pop()
        if isinstance(node, tuple | list):
            stack.extend(node)
            continue
        if not isinstance(node, prim.ExpressionNode):
            continue
        out.append(node)
        if isinstance(node, prim.Subscript):
            stack.extend((node.aggregate, node.index))
        elif isinstance(node, prim.Variable):
            continue
        else:
            stack.extend(init_args(node))
    return out


# }}}


# {{{ isl relations over the schedule's instance space


def _aligned(*objs: Any) -> list[Any]:
    """isl objects with one list of parameters, so that isl combines them."""
    out = list(objs)
    for _ in range(2):
        for k, obj in enumerate(out):
            for other in out:
                obj = obj.align_params(other.get_space().params())
            out[k] = obj
    return out


def lift(stmt: Stmt, index: int, width: int) -> isl.Map:
    """``{ [s, x0, x1] -> [i0] : s = index and x0 = i0 and x1 = 0 }``.

    From the schedule's uniform instance space to one statement's own loops,
    the inverse of the padding :func:`loopty.schedule._embed` lifts a domain
    by. Primed names, which no size can have.
    """
    own = len(stmt.inames)
    source = ", ".join(["s'", *(f"x{k}'" for k in range(width))])
    target = ", ".join(f"i{k}'" for k in range(own))
    constraints = [f"s' = {index}"]
    constraints += [f"x{k}' = i{k}'" for k in range(own)]
    constraints += [f"x{k}' = 0" for k in range(own, width)]
    return isl.Map(f"{{ [{source}] -> [{target}] : {' and '.join(constraints)} }}")


def uniform_access(
    stmt: Stmt,
    index: int,
    width: int,
    inames: Sequence[str],
    domain: isl.Set,
    indices: Sequence[Any],
) -> isl.Map:
    """The cells one access reaches, from the instances of its statement.

    ``{ [s, x0, ...] -> [cell] }`` over the uniform instance space, at the
    points of ``domain`` the access is made over (a sum's binders after the
    statement's loops, which an instance does not hold, so they are
    projected out: the access reaches every cell its sum reads). Exact, for
    an access whose indices are quasi-affine, which the caller asks first.
    """
    relation = flow.access_relation(inames, domain, indices)
    extra = len(inames) - len(stmt.inames)
    if extra > 0:
        relation = relation.project_out(isl.dim_type.in_, len(stmt.inames), extra)
    pad = lift(stmt, index, width)
    pad, relation = _aligned(pad, relation)
    return pad.apply_range(relation)


def affine_indices(indices: Sequence[Any]) -> bool:
    """Whether isl reads every index of an access exactly."""
    try:
        for index in indices:
            flow.expr_text(index, None, None)
    except (flow.NonAffine, ValueError):
        return False
    return True


def _in(expr: Any, array: str) -> bool:
    """Whether ``expr`` reads a cell of ``array``."""
    if expr is None:
        return False
    return any(
        isinstance(node, prim.Subscript)
        and isinstance(node.aggregate, prim.Variable)
        and node.aggregate.name == array
        for node in _walk(expr)
    )


@dataclass(frozen=True)
class Access:
    """One access of the contracted array: who makes it, where, at which cell.

    ``kind`` is ``"write"`` or ``"read"``, and ``part`` what
    :func:`loopty.schedule._accesses` says of where in its statement it is
    made: ``""`` in its own instruction, ``"sum"`` in the body of a sum.
    """

    stmt: Stmt
    kind: str
    indices: tuple[Any, ...]
    inames: tuple[str, ...]
    domain: Any
    part: str


def accesses(term: Term, array: str, stmts: Sequence[Stmt]) -> list[Access]:
    """The accesses of ``array`` the statements ``stmts`` make, writes first."""
    out: list[Access] = []
    for stmt in stmts:
        own = len(stmt.inames)
        for name, indices, kind, inames, domain in flow.statement_accesses(stmt, term):
            if name != array:
                continue
            part = "" if len(inames) == own else "sum" if len(inames) > own else "row"
            kinds = ("write", "read") if kind == "acc" else (kind,)
            for one in kinds:
                out.append(
                    Access(stmt, one, tuple(indices), tuple(inames), domain, part)
                )
    return sorted(out, key=lambda access: access.kind != "write")


def window_accesses(term: Term, window: Any) -> list[Access]:
    """Every access of a window's array: its producers' stores, then the reads.

    The reads are its readers', a producer that reads the array among them
    (a recurrence, ``R[i, a, b] = -R[i, a - 2, b + 2]``), each read once.
    """
    return [
        *(
            access
            for access in accesses(term, window.array, window.producers)
            if access.kind == "write"
        ),
        *(
            access
            for access in accesses(term, window.array, window.readers)
            if access.kind == "read"
        ),
    ]


def preconditions(
    term: Term,
    array: str,
    gone: frozenset[str],
    text: str,
) -> tuple[list[Stmt], list[Stmt]]:
    """The producers and the readers of ``array``, or why it cannot be contracted.

    The array is one the program made (``Arr.zeros_like``, ``Arr.zeros``),
    which no caller sees. The statements that store it besides its zeros
    are its producers, one or several (``R[i, a, b] = C[i, a, b]`` where
    ``a < 2`` and the recurrence ``R[i, a, b] = -R[i, a - 2, b + 2]``
    elsewhere), and each stores a value (``assign``, at indices isl reads
    exactly, under no guard isl cannot state): not an update, ``f[j] +=
    v``, which adds to the zeros a window drops. Every statement that reads
    it, a producer of it included, reads it in a value, at indices isl reads
    exactly, not in a guard or in the subscript of what it stores, since a
    window is safe only where the order of the reads is the instances'.
    Each is a :class:`WindowError` naming what stands in the way. A
    producer that reads the array, a recurrence, is among the readers.
    """
    if array not in dict(term.temporaries) or array in dict(term.checks):
        what = (
            f"a parameter of {term.name}, which its caller passes and sees"
            if array in term.param_names
            else f"the flag of a checked point of {term.name}"
            if array in dict(term.checks)
            else f"no array of {term.name}"
        )
        raise WindowError(
            f"{text}: {array} is {what}, and only an array the program makes "
            "with Arr.zeros_like or Arr.zeros is its own to keep in a window"
        )
    zeros = f"{array}.zeros"
    writers = [
        stmt
        for stmt in term.stmts
        if stmt.assignee.array == array and stmt.id not in gone
    ]
    producers = [stmt for stmt in writers if stmt.id != zeros]
    if not producers:
        raise WindowError(
            f"{text}: nothing in {term.name} stores {array} but its zeros, which a "
            "window does not keep"
        )
    for producer in producers:
        if producer.kind != "assign":
            # An update, f[j] += v, starts from the zeros, which a window drops,
            # and its slot holds another cell when the next cell's first update
            # comes: the window would add to that cell's value (#213).
            shown = ", ".join(str(index) for index in producer.assignee.indices)
            cell = f"{array}[{shown}]"
            fix = (
                f"store each cell once, {cell} = ... (a reduce_sum where the "
                f"updates run over a loop), or keep {array} whole"
                if affine_indices(producer.assignee.indices)
                else f"keep {array} whole"
            )
            raise WindowError(
                f"{text}: {producer.id} updates {cell} in place, starting from the "
                f"zeros {array} was made with, and a window keeps a value stored "
                f"once, without them; {fix}"
            )
        if not affine_indices(producer.assignee.indices):
            shown = ", ".join(str(index) for index in producer.assignee.indices)
            raise WindowError(
                f"{text}: {producer.id} stores {array}[{shown}], an index isl "
                "cannot read, so which slot it stores into is not known"
            )
        if producer.unnarrowed:
            conjuncts = ", ".join(conjunct for conjunct, _why in producer.unnarrowed)
            raise WindowError(
                f"{text}: {producer.id} stores {array} under a guard isl cannot "
                f"state ({conjuncts}), so which cells it stores is not known"
            )
    readers = [
        stmt
        for stmt in term.stmts
        if stmt.id not in gone
        and stmt.id != zeros
        and any(access.kind == "read" for access in accesses(term, array, [stmt]))
    ]
    if all(any(reader is producer for producer in producers) for reader in readers):
        raise WindowError(f"{text}: nothing in {term.name} reads {array}")
    for reader in readers:
        if _in(reader.guard, array) or any(
            _in(index, array) for index in reader.assignee.indices
        ):
            raise WindowError(
                f"{text}: {reader.id} reads {array} in a guard or in the subscript "
                "of what it stores, which is evaluated where it stores nothing; a "
                "window is taken of an array read in values"
            )
        for access in accesses(term, array, [reader]):
            if access.part == "row" or not affine_indices(access.indices):
                shown = ", ".join(str(index) for index in access.indices)
                raise WindowError(
                    f"{text}: {reader.id} reads {array}[{shown}], an index isl "
                    "cannot read, so which slot it reads is not known"
                )
    return producers, readers


# }}}


# {{{ the window's questions


def earlier(schedule: isl.Map, instances: isl.Set) -> isl.Map:
    """``{ a -> b : a runs before b }`` under ``schedule``, over ``instances``.

    ``schedule`` takes the term's instances to their time, as the checker's
    order does (:meth:`loopty.schedule.Schedule._commit`): the steps' maps,
    then the time map of the order as it stands. Two instances with one
    time, which only a loop with no order gives, are neither before the
    other.
    """
    lex = isl.Map.lex_lt(schedule.get_space().range())
    out = schedule.apply_range(lex).apply_range(schedule.reverse())
    out, instances = _aligned(out, instances)
    return out.intersect_domain(instances).intersect_range(instances)


def same_slot(first: isl.Map, second: isl.Map, placement: isl.Map) -> isl.Map:
    """``{ a -> b }``: the cells ``a`` and ``b`` touch share a slot."""
    first, second, placement = _aligned(first, second, placement)
    return first.apply_range(placement).apply_range(
        second.apply_range(placement).reverse()
    )


def clobbers(
    reads: isl.Map, writes: isl.Map, placement: isl.Map, before: isl.Map
) -> isl.Map:
    """``{ r -> [w -> w'] }``: a read whose slot is overwritten before it.

    ``r`` reads the cell ``w`` stored, ``w'`` stores another cell into the
    same slot, after ``w`` and before ``r``. Empty when the window is safe
    for these reads.
    """
    reads, writes, placement, before = _aligned(reads, writes, placement, before)
    stored = reads.apply_range(writes.reverse())
    shared = same_slot(writes, writes, placement)
    one_cell = writes.apply_range(writes.reverse())
    shared, one_cell, stored = _aligned(shared, one_cell, stored)
    other = shared.subtract(one_cell).intersect(before)
    late = before.reverse()
    stored, late, other = _aligned(stored, late, other)
    return stored.range_product(late).intersect_range(other.wrap())


def live_pairs(reads: isl.Map, writes: isl.Map, before: isl.Map) -> isl.Map:
    """``{ c -> c' }``: a cell read, and a cell stored after it and before the read.

    The pairs of cells that are live at once: ``c`` was stored by ``w``, is
    read by ``r``, and ``c'`` is stored by ``w'`` between the two. A
    placement is safe for these reads exactly when it puts no two different
    cells of a pair in one slot, which :func:`infer` reads a window off.
    """
    reads, writes, before = _aligned(reads, writes, before)
    stored = reads.apply_range(writes.reverse())
    late = before.reverse()
    writers = writes.domain()
    between = before.intersect_domain(writers).intersect_range(writers)
    stored, late, between = _aligned(stored, late, between)
    # w -> [r -> w'] with r reading what w stored and w' between them
    triples = stored.reverse().range_product(between).intersect_range(late.wrap())
    return triples.range_factor_range().apply_domain(writes).apply_range(writes)


def _without_params(obj: isl.Set) -> isl.Set:
    """``obj`` with its parameters existentially quantified: every size at once."""
    count = obj.dim(isl.dim_type.param)
    if not count:
        return obj
    return obj.move_dims(isl.dim_type.set, 0, isl.dim_type.param, 0, count).project_out(
        isl.dim_type.set, 0, count
    )


def _collide(rank: int, moduli: Mapping[int, int | None], skip: int) -> isl.Map:
    """``{ c -> c' }`` colliding on every axis but ``skip`` under ``moduli``.

    An axis with a modulus ``m`` collides where the two indices differ by a
    multiple of ``m`` (``1`` is an axis dropped from the slot, which always
    collides); one with ``None`` is kept, and collides where they are equal.
    """
    source = ", ".join(f"c{k}'" for k in range(rank))
    target = ", ".join(f"d{k}'" for k in range(rank))
    constraints = ["1 = 1"]
    for k in range(rank):
        if k == skip:
            continue
        modulus = moduli.get(k)
        if modulus is None:
            constraints.append(f"d{k}' = c{k}'")
        elif modulus > 1:
            constraints.append(f"(d{k}' - c{k}') mod {modulus} = 0")
    return isl.Map(f"{{ [{source}] -> [{target}] : {' and '.join(constraints)} }}")


def infer(
    pairs: isl.Map,
    rank: int,
    extents: Sequence[Any],
    names: Sequence[str],
) -> Placement:
    """The smallest window per axis, outermost first, for the live ``pairs``.

    For each axis, with the axes before it as decided and those after it
    kept: the pairs of different cells that collide on every other axis
    are those this axis alone has to tell apart, and the largest distance
    between them along it, plus one, is the modulus it needs. No such pair
    drops the axis from the slot, a distance that grows with the sizes keeps
    it whole, and so does a modulus no smaller than its extent, where that is
    a number. The result is the placement ``lambda j: j % 3`` that
    :func:`clobbers` is then asked about.
    """
    moduli: dict[int, int | None] = {}
    identity = isl.Map.identity(
        isl.Space.create_from_names(
            pairs.get_ctx(), set=[f"c{k}'" for k in range(rank)]
        ).map_from_set()
    )
    for axis in range(rank):
        collide = _collide(rank, moduli, axis)
        candidates, collide, same = _aligned(pairs, collide, identity)
        apart = candidates.intersect(collide).subtract(same)
        if apart.is_empty():
            moduli[axis] = 1
            continue
        distances = _without_params(
            apart.deltas()
            .project_out(isl.dim_type.set, 0, axis)
            .project_out(isl.dim_type.set, 1, rank - axis - 1)
        )
        if not distances.is_bounded():
            moduli[axis] = None
            continue
        high = distances.dim_max_val(0).to_python()
        low = distances.dim_min_val(0).to_python()
        modulus = int(max(abs(high), abs(low))) + 1
        extent = extents[axis]
        if isinstance(extent, int) and modulus >= extent:
            moduli[axis] = None
            continue
        moduli[axis] = modulus
    slots: list[Any] = []
    for axis, name in enumerate(names):
        modulus = moduli[axis]
        variable = prim.Variable(name)
        if modulus is None:
            slots.append(variable)
        elif modulus > 1:
            slots.append(prim.Remainder(variable, modulus))
    return Placement(tuple(names), tuple(slots) if slots else (0,))


def extents_of(
    placement: Placement,
    cells: isl.Set,
    sizes: Sequence[Any],
    text: str,
    array: str,
) -> tuple[Any, ...]:
    """The window's shape: one extent per slot index, a number or a size.

    A slot that is an index of the array keeps that axis's size; any other
    is bounded by a number whatever the sizes, which isl finds over the
    cells of the array. A slot below zero, or one that grows with the sizes
    and is not an axis, is a :class:`WindowError`.
    """
    mapping = placement.isl_map()
    mapping, cells = _aligned(mapping, cells)
    image = cells.apply(mapping)
    out: list[Any] = []
    for k, slot in enumerate(placement.slots):
        values = image.project_out(
            isl.dim_type.set, k + 1, len(placement.slots) - k - 1
        ).project_out(isl.dim_type.set, 0, k)
        if values.is_empty():
            out.append(1)
            continue
        below = values.intersect(
            isl.Set("{ [v] : v < 0 }").align_params(values.get_space())
        )
        if not below.is_empty():
            raise WindowError(
                f"{text}: the slot {_render(slot)} of {array} is below zero for some "
                "cells; a slot is counted from zero"
            )
        if isinstance(slot, prim.Variable) and slot.name in placement.names:
            out.append(sizes[placement.names.index(slot.name)])
            continue
        bounded = _without_params(values)
        if not bounded.is_bounded():
            raise WindowError(
                f"{text}: the slot {_render(slot)} of {array} grows with the sizes, "
                "so the window would not be smaller than the array; keep an axis "
                "whole as its index, or take a remainder by a number"
            )
        out.append(int(bounded.dim_max_val(0).to_python()) + 1)
    return tuple(out)


# }}}


# {{{ the kernel, rewritten


def _fresh(name: str, taken: set[str]) -> str:
    """``name``, or ``name_`` and so on, apart from every name in ``taken``."""
    while name in taken:
        name += "_"
    taken.add(name)
    return name


def window_name(array: str, taken: set[str]) -> str:
    """``f_window``: the temporary a window of ``f`` is kept in."""
    return _fresh(f"{array}_window", set(taken))


def _offset(index: Any, loop: str) -> int | None:
    """``d`` where ``index`` is ``loop + d``, and ``None`` where it is not."""
    from pymbolic import evaluate

    free = {node.name for node in _walk(index) if isinstance(node, prim.Variable)}
    if not free <= {loop}:
        return None
    try:
        at_zero = evaluate(index, {loop: 0})
        at_one = evaluate(index, {loop: 1})
    except Exception:  # noqa: BLE001 - not an expression of the loop alone
        return None
    if not isinstance(at_zero, int) or at_one - at_zero != 1:
        return None
    return int(at_zero)


def register_plan(
    kernel: Any,
    array: str,
    producers: Sequence[str],
    placement: Placement,
    skip: Sequence[str] = (),
    names: Mapping[str, str] | None = None,
) -> tuple[str, int, int] | str:
    """The loop, the first offset and the count of a shift register, or why not.

    A shift register keeps a window along one loop: the placement takes one
    axis modulo a number and drops the others, and every access of the array
    in the kernel, at that axis, is the loop plus a number. The producers
    (their instructions) store the newest cell, the largest offset; the
    reads are at offsets no larger, a producer's own reads among them (a
    recurrence, ``f[j] = f[j - 1] + u[j]``, stores at 0 and reads at -1),
    and the register count is the span of the offsets. The instructions
    ``skip`` (the zeros the window drops) are not asked about. ``names``
    gives the statement each instruction is, for the reason. Returns
    ``(loop, lowest offset, count)``, or the reason in words.
    """
    shown = dict(names or {})

    def said(insn_id: str) -> str:
        return shown.get(insn_id, insn_id)

    windowed = [
        (k, slot)
        for k, slot in enumerate(placement.slots)
        if isinstance(slot, prim.Remainder)
        and isinstance(slot.numerator, prim.Variable)
        and slot.numerator.name in placement.names
        and isinstance(slot.denominator, int)
    ]
    if len(placement.slots) != 1 or len(windowed) != 1:
        return (
            f"a shift register keeps a window along one loop, as lambda j: j % 3 "
            f"does, and {placement} is not one"
        )
    (_, slot) = windowed[0]
    axis = placement.names.index(slot.numerator.name)
    modulus = slot.denominator
    entry = kernel.default_entrypoint
    # (instruction, index, whether the access is the instruction's store)
    found: list[tuple[Any, Any, bool]] = []

    from loopy.symbolic import WalkMapper

    def indices_of(expr: Any) -> tuple[Any, ...] | None:
        """The index of ``expr`` if it is a subscript of the array."""
        if (
            isinstance(expr, prim.Subscript)
            and isinstance(expr.aggregate, prim.Variable)
            and expr.aggregate.name == array
        ):
            return expr.index if isinstance(expr.index, tuple) else (expr.index,)
        return None

    class _Reads(WalkMapper):
        def __init__(self, insn: Any) -> None:
            super().__init__()
            self.insn = insn

        def map_subscript(self, expr: Any, *args: Any, **kwargs: Any) -> None:
            index = indices_of(expr)
            if index is not None:
                found.append((self.insn, index, False))
            super().map_subscript(expr, *args, **kwargs)

    for insn in entry.instructions:
        if insn.id in skip:
            continue
        # an instruction stores its assignees and reads everything else in
        # it, the subscripts of what it stores among them
        reads = _Reads(insn)
        for assignee in getattr(insn, "assignees", ()):
            index = indices_of(assignee)
            if index is not None:
                found.append((insn, index, True))
            if isinstance(assignee, prim.Subscript):
                inner = assignee.index
                for part in inner if isinstance(inner, tuple) else (inner,):
                    reads(part)
        for expr in (getattr(insn, "expression", None), *insn.predicates):
            if expr is not None:
                reads(expr)
    stored_by = set(producers)
    loop: str | None = None
    for producer in producers:
        writes = [
            index for insn, index, stores in found if insn.id == producer and stores
        ]
        if not writes:
            return f"{said(producer)} stores no cell of {array} the kernel can find"
        insn = next(insn for insn, _, _ in found if insn.id == producer)
        loops = sorted(
            name
            for name in insn.within_inames
            if _offset(writes[0][axis], name) is not None
        )
        if len(loops) != 1 or loop not in (None, loops[0]):
            return (
                f"{said(producer)} stores {array} at no index that is a loop plus a "
                "number, which is what a shift register shifts along"
            )
        loop = loops[0]
    assert loop is not None  # there is a producer
    offsets = []
    for insn, index, stores in found:
        offset = _offset(index[axis], loop) if loop in insn.within_inames else None
        if offset is None:
            return (
                f"{said(insn.id)} {'stores' if stores else 'reads'} "
                f"{array}[{', '.join(map(str, index))}], which is not at {loop} "
                "plus a number, so its register would change from step to step"
            )
        offsets.append((stores and insn.id in stored_by, offset))
    newest = max(offset for _, offset in offsets)
    stored = {offset for written, offset in offsets if written}
    if stored != {newest}:
        which = (
            f"{said(producers[0])} stores a cell older than one it reads"
            if len(producers) == 1
            else f"{' and '.join(map(said, producers))} store a cell older than "
            "one they read"
        )
        return f"{which}, and a shift register's newest register is the one stored"
    lowest = min(offset for _, offset in offsets)
    count = newest - lowest + 1
    if count != modulus:
        return (
            f"the accesses span {count} cells of {array} along {loop}, and the "
            f"placement {placement} keeps {modulus}; contract with lambda "
            f"{placement.names[axis]}: {placement.names[axis]} % {count}"
        )
    return loop, lowest, count


def windowed_kernel(
    kernel: Any,
    array: str,
    removed: Sequence[str],
    placement: Placement,
    extents: Sequence[Any],
    name: str,
    space: Any,
    register: tuple[str, int, int] | None = None,
    outer: Sequence[str] = (),
) -> tuple[Any, str | None]:
    """``kernel`` with ``array`` kept in a window, or why loopy could not.

    The instructions ``removed`` zero the array where the program made it,
    and go first, as :meth:`loopty.schedule.Schedule.substitute` drops them:
    no read sees them (the strict ``definedness`` fact). The array becomes
    the temporary ``name`` of shape ``extents`` in ``space``, and each
    access ``f[e]`` reads or stores the slot of ``e``.

    With ``register`` (see :func:`register_plan`), the slot is the access's
    offset from the loop, a number, and the window is shifted down by one at
    the start of every step of the loop, before anything else in the step
    touches it: ``f_window[0] = f_window[1]``, ``f_window[1] = f_window[2]``.
    The shifts run in the loops ``outer`` the window's loop is nested in, as
    the accesses do, so that each step of it inside each of theirs shifts
    once.
    """
    import loopy as lp
    from loopy.symbolic import IdentityMapper

    try:
        entry = kernel.default_entrypoint
        old = entry.temporary_variables[array]
        if removed:
            loops = {
                loop
                for insn in entry.instructions
                if insn.id in removed
                for loop in insn.within_inames
            }
            kernel = lp.remove_instructions(kernel, set(removed))
            kernel = lp.remove_unused_inames(kernel, loops)
            entry = kernel.default_entrypoint

        if register is None:

            def slot(index: tuple[Any, ...]) -> tuple[Any, ...]:
                return placement.at(index)

            shape = tuple(extents)
        else:
            loop, lowest, count = register
            axis = placement.names.index(placement.slots[0].numerator.name)

            def slot(index: tuple[Any, ...]) -> tuple[Any, ...]:
                offset = _offset(index[axis], loop)
                assert offset is not None  # register_plan found every one
                return (offset - lowest,)

            shape = (count,)

        class _Window(IdentityMapper):
            def map_subscript(self, expr: Any, *args: Any, **kwargs: Any) -> Any:
                if (
                    isinstance(expr.aggregate, prim.Variable)
                    and expr.aggregate.name == array
                ):
                    index = (
                        expr.index if isinstance(expr.index, tuple) else (expr.index,)
                    )
                    index = tuple(self.rec(i, *args, **kwargs) for i in index)
                    slots = slot(index)
                    return prim.Subscript(
                        prim.Variable(name),
                        slots if len(slots) > 1 else slots[0],
                    )
                return super().map_subscript(expr, *args, **kwargs)

        mapper = _Window()
        instructions = [
            insn.with_transformed_expressions(mapper) for insn in entry.instructions
        ]
        temporaries = {
            key: value
            for key, value in entry.temporary_variables.items()
            if key != array
        }
        temporaries[name] = lp.TemporaryVariable(
            name, old.dtype, shape=shape, address_space=space
        )
        if register is not None:
            loop, _lowest, count = register
            taken = {insn.id for insn in instructions}
            shifts: list[Any] = []
            # zeros before the loop, so that no shift copies a register
            # nothing has set; no read sees them (the definedness fact)
            cleared = []
            for k in range(count):
                clear_id = _fresh(f"{name}_clear_{k}", taken)
                shifts.append(
                    lp.Assignment(
                        prim.Subscript(prim.Variable(name), k),
                        0,
                        id=clear_id,
                        within_inames=frozenset(),
                    )
                )
                cleared.append(clear_id)
            previous: str | None = None
            for k in range(count - 1):
                shift_id = _fresh(f"{name}_shift_{k}", taken)
                shifts.append(
                    lp.Assignment(
                        prim.Subscript(prim.Variable(name), k),
                        prim.Subscript(prim.Variable(name), k + 1),
                        id=shift_id,
                        within_inames=frozenset({*outer, loop}),
                        depends_on=frozenset(cleared)
                        if previous is None
                        else frozenset({previous}),
                    )
                )
                previous = shift_id
            if previous is None:
                # one register: nothing to shift, and the clearing comes first
                previous = cleared[-1]
            if previous is not None:
                touching = []
                for insn in instructions:
                    names = insn.dependency_names() | set(insn.assignee_var_names())
                    if name in names and loop in insn.within_inames:
                        insn = insn.copy(depends_on=insn.depends_on | {previous})
                    touching.append(insn)
                instructions = [*shifts, *touching]
        entry = entry.copy(instructions=instructions, temporary_variables=temporaries)
        kernel = kernel.with_kernel(entry)
    except Exception as exc:  # noqa: BLE001 - loopy's refusals are of many kinds
        return None, (
            f"loopy could not keep {array} in a window: {type(exc).__name__}: {exc}"
        )
    return kernel, None


# }}}


# {{{ in words


def describe_cells(cells: int | Any) -> str:
    """``3 cells``, ``a row of n cells``: how many cells a window has, in words."""
    return f"{cells} cell{'s' if cells != 1 else ''}"


def window_cells(extents: Sequence[Any]) -> Any:
    """The number of cells of a window of ``extents``, a number or a term."""
    total: Any = 1
    for extent in extents:
        total = (
            total * extent if not (isinstance(total, int) and total == 1) else extent
        )
    return total


def statement_text(array: str, placement: Placement, extents: Sequence[Any]) -> str:
    """``f kept in 3 cells, f[j] in slot j % 3``: the window, in plain words."""
    cells = window_cells(extents)
    cell = f"{array}[{', '.join(placement.names)}]"
    return (
        f"{array} kept in {describe_cells(cells)}, {cell} in slot "
        f"{placement.slot_text()}"
    )


# }}}
