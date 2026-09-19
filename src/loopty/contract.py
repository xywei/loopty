"""What an argument list owes a term, checked before anything runs.

Three of loopty's claims are claims about the *call* and not about the term, so
nothing inside the type system can establish them and nothing downstream can
notice when they fail. They are checked here, in one place, by every entry point
that turns Python values into a run: :class:`loopty.executor.LoopyExecutor` for a
compiled run and a differential comparison, and
:meth:`loopty.kernel.Kernel.__call__` for the native one.

*Distinct parameters are distinct storage.* The dependence relation
(:mod:`loopty.flow`) compares footprints array by array and reports nothing
between two differently named parameters, so a kernel reading ``x[i - 1]`` and
writing ``y[i]`` carries no dependence and may legally tag ``i`` parallel. Called
with ``x is y`` it is a race, and the differential test cannot see it: it copies
each argument on its own, which destroys the alias and makes the two runs agree.
:func:`disjoint_arguments` asks numpy whether two arguments share storage and
refuses the call if they do.

*A ragged argument agrees with its counts family.* ``val: Arr[Fin[n], Fin[cnt],
Real]`` says row ``r`` of ``val`` has ``cnt[r]`` entries. The generated loop is
bounded by ``cnt[r]`` while the flattened access is ``off[r] + j``, so a ``val``
whose own offsets disagree with ``cnt`` makes compiled C read past a row while
the native run, which follows the ``Arr``'s own counts, stays inside it. Two
ragged arguments over one counts family are flattened through *one* offsets
argument, so they have to agree with each other as well, and with an offsets
array the caller supplies by hand. :func:`ragged_arguments` checks all three.

*An element of a refined sort really is one.* ``col: Arr[..., Fin[m]]`` is what
discharges ``x[col[r, j]]`` in bounds **by type**, with no isl call and no
runtime test in the generated code (see :mod:`loopty.typing`). That fact is
sound exactly to the extent that the entries of ``col`` are points of ``Fin[m]``,
which is a property of the data and of nothing else. :func:`element_types`
enforces it on the way in, so the ledger's ``decided by type`` row is backed by
a check rather than by a hope.

Nothing here checks array *shapes*; see ``docs/loopy-notes.md`` for why lowering
has to declare some arrays without one, and the README's status list for the
consequence.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import numpy as np
import pymbolic.primitives as prim

from loopty.arr import Arr
from loopty.term import ArrType

__all__ = [
    "check_arguments",
    "counts_family",
    "disjoint_arguments",
    "element_bound",
    "element_types",
    "ragged_arguments",
    "resolve_sizes",
]


def _buffer(value: Any) -> np.ndarray | None:
    """The numpy storage an argument owns, or ``None`` if it owns none.

    A ragged :class:`~loopty.arr.Arr` owns its flat values buffer; its offsets
    are layout, are not a parameter of the term, and are compared separately.
    """
    if isinstance(value, Arr):
        return value.numpy()
    if isinstance(value, np.ndarray):
        return value
    return None


def _values(value: Any) -> np.ndarray | None:
    """An argument's elements as one flat array, or ``None``."""
    buffer = _buffer(value)
    return None if buffer is None else np.asarray(buffer).reshape(-1)


def _integers(value: Any) -> np.ndarray | None:
    """An argument read as a flat array of numbers, or ``None``."""
    buffer = _buffer(value)
    if buffer is None:
        return None
    flat = np.asarray(buffer).reshape(-1)
    return flat if flat.dtype.kind in "biufc" else None


def counts_family(typ: ArrType) -> str | None:
    """The counts array a ragged type names, or ``None`` if the type is dense."""
    for axis, ragged in enumerate(typ.ragged):
        if not ragged:
            continue
        size = typ.axes[axis]
        return size.name if isinstance(size, prim.Variable) else None
    return None


# {{{ aliasing


def disjoint_arguments(supplied: Mapping[str, Any]) -> None:
    """Refuse a call in which two distinct array parameters share storage.

    ``numpy.shares_memory`` is asked for the exact answer rather than
    ``may_share_memory``'s conservative one, because a false positive here
    refuses a legal call. The arrays a kernel is run on are the arrays of a
    differential test and of a demo, which are small; a size at which the exact
    decision is expensive is a size at which the run dwarfs it.
    """
    buffers = [
        (name, buffer)
        for name, buffer in ((n, _buffer(v)) for n, v in supplied.items())
        if buffer is not None and buffer.size
    ]
    for position, (first, left) in enumerate(buffers):
        for second, right in buffers[position + 1 :]:
            if not np.shares_memory(left, right):
                continue
            raise ValueError(
                f"the arguments {first} and {second} share storage. loopty "
                "assumes two distinct array parameters are two distinct "
                "buffers: dependences are computed per array name, so nothing "
                "between them is ever reported and a schedule may run them in "
                "parallel. Pass separate arrays, or write the kernel with one "
                "parameter for the buffer they share"
            )


# }}}


# {{{ ragged arguments and their counts


def ragged_arguments(
    types: Mapping[str, Any],
    supplied: Mapping[str, Any],
    offsets_args: Mapping[str, str] | None = None,
) -> None:
    """Check every ragged argument against the counts family its type names.

    Three ways the same fact can be contradicted, all three refused:

    * a ragged argument whose per-row counts differ from the concrete counts
      array its type names;
    * two ragged arguments over one counts family whose offsets differ, since
      lowering flattens them through a single offsets argument;
    * an offsets array the caller passes explicitly that differs from the one
      the ragged arguments carry.

    ``offsets_args`` maps an array to the offsets argument its flat storage is
    indexed through (:attr:`loopty.lower.Lowering.ragged`). The native run has
    no such argument, so it is omitted there and the third check does not apply.
    """
    offsets_args = offsets_args or {}
    families: dict[str, list[tuple[str, np.ndarray]]] = {}
    for name, typ in types.items():
        if not isinstance(typ, ArrType):
            continue
        value = supplied.get(name)
        if not (isinstance(value, Arr) and value.is_ragged):
            continue
        family = counts_family(typ)
        if family is None:
            continue
        families.setdefault(family, []).append((name, np.asarray(value.offsets)))
        declared = _integers(supplied.get(family))
        if declared is None:
            continue
        counts = np.asarray(value.counts)
        if counts.shape != declared.shape or not np.array_equal(counts, declared):
            raise ValueError(
                f"the ragged argument {name} has row counts "
                f"{counts.tolist()}, but the counts array its type names, "
                f"{family}, holds {declared.tolist()}. The type "
                f"{name}: Arr[..., Fin[{family}], ...] says those are the same "
                f"numbers: the generated loop over a row is bounded by "
                f"{family}[r] while the access into {name} is flattened through "
                "its offsets, so a disagreement reads past the end of a row"
            )

    for family, members in families.items():
        first_name, first = members[0]
        for name, offsets in members[1:]:
            if first.shape == offsets.shape and np.array_equal(first, offsets):
                continue
            raise ValueError(
                f"the ragged arguments {first_name} and {name} are both laid "
                f"out over the counts family {family} but have different "
                f"offsets, {first.tolist()} and {offsets.tolist()}. They are "
                "flattened through one offsets argument, so one of the two "
                "would be indexed by the other's row starts"
            )
        for name, offsets in members:
            explicit = offsets_args.get(name)
            if explicit is None or explicit not in supplied:
                continue
            given = _integers(supplied[explicit])
            if given is None:
                continue
            if given.shape == offsets.shape and np.array_equal(given, offsets):
                continue
            raise ValueError(
                f"the offsets argument {explicit} holds {given.tolist()}, but "
                f"the ragged argument {name} it flattens carries "
                f"{offsets.tolist()}. The two are the same layout written twice "
                "and have to agree; pass one of them, or make them equal"
            )


# }}}


# {{{ refined element types


def resolve_sizes(
    types: Mapping[str, Any], supplied: Mapping[str, Any]
) -> dict[str, int]:
    """The concrete value of every size name the arguments determine.

    Sizes are not parameters: they come from the data, so ``m`` in
    ``x: Arr[Fin[m], Real]`` is ``len(x)``. A ragged axis determines nothing (its
    extent is per row) and a size written as an expression, ``Fin[n + 1]``, is
    left alone rather than solved for.
    """
    sizes: dict[str, int] = {}
    for name, typ in types.items():
        value = supplied.get(name)
        if value is None:
            continue
        if not isinstance(typ, ArrType):
            if isinstance(value, int | np.integer) and not isinstance(value, bool):
                sizes.setdefault(name, int(value))
            continue
        shape = _index_shape(value, typ)
        if shape is None:
            continue
        for axis, (size, ragged) in enumerate(
            zip(typ.axes, typ.ragged, strict=True)
        ):
            if ragged or axis >= len(shape):
                continue
            if isinstance(size, prim.Variable):
                sizes.setdefault(size.name, int(shape[axis]))
    return sizes


def _index_shape(value: Any, typ: ArrType) -> tuple[int, ...] | None:
    """The extents of an argument in its *index* axes, not in flat storage."""
    if isinstance(value, Arr):
        if value.is_ragged:
            return (int(np.asarray(value.offsets).size - 1),)
        return tuple(int(s) for s in value.numpy().shape)
    if isinstance(value, np.ndarray):
        if any(typ.ragged):
            return None
        return tuple(int(s) for s in value.shape)
    return None


def element_bound(
    typ: ArrType, sizes: Mapping[str, int]
) -> tuple[int, int | None] | None:
    """The half-open range an element of ``typ`` has to lie in, if its sort says.

    ``Fin[m]`` says ``0 <= v < m``, with ``m`` resolved from the sizes the other
    arguments determine; ``Fin[m] & p`` contributes its ``Fin`` and the
    propositions are left to an oracle. ``Nat`` says ``0 <= v`` and nothing
    above, which is ``None`` for the upper end. Every other sort says nothing
    about the value at all, and the whole pair is ``None``.
    """
    from lanky.prelude import FinType, Refined

    sort = typ.dtype
    if isinstance(sort, Refined):
        sort = sort.base
    if isinstance(sort, FinType):
        bound = sort.bound
        if isinstance(bound, int | np.integer):
            return (0, int(bound))
        if isinstance(bound, prim.Variable) and bound.name in sizes:
            return (0, int(sizes[bound.name]))
        return None
    if getattr(sort, "name", None) == "Nat":
        return (0, None)
    return None


def element_types(
    types: Mapping[str, Any],
    supplied: Mapping[str, Any],
    sizes: Mapping[str, int] | None = None,
) -> None:
    """Refuse an argument holding a value its declared element type excludes.

    This is the check the ``in bounds by type`` rule stands on. ``col`` declared
    ``Arr[..., Fin[m]]`` makes ``x[col[r, j]]`` in bounds with no proof and no
    generated test, so a ``col`` entry of ``-1`` or of ``m`` reaches the
    compiled code as an address outside ``x``. The first offending cell is named
    with its index and its value, because "some entry is out of range" is not
    something a caller can act on.
    """
    sizes = resolve_sizes(types, supplied) if sizes is None else sizes
    for name, typ in types.items():
        if not isinstance(typ, ArrType):
            continue
        value = supplied.get(name)
        flat = _values(value)
        if flat is None or not flat.size or flat.dtype.kind not in "biuf":
            continue
        limits = element_bound(typ, sizes)
        if limits is None:
            continue
        low, high = limits
        outside = flat < low if high is None else (flat < low) | (flat >= high)
        offenders = np.flatnonzero(outside)
        if not offenders.size:
            continue
        position = int(offenders[0])
        where = _cell_label(name, value, position)
        allowed = f"{low} <= v" if high is None else f"{low} <= v < {high}"
        raise ValueError(
            f"{where} is {flat[position]}, which is not a value of "
            f"{typ.dtype}: an element of {name} has to satisfy {allowed}. "
            "loopty discharges an indirection through this array as in bounds "
            "*by type*, with no check in the generated code, so the element "
            "type has to hold of the data that is passed in"
        )


def _cell_label(name: str, value: Any, position: int) -> str:
    """``col[1, 0]``: where in an argument the flat offset ``position`` is."""
    if isinstance(value, Arr) and value.is_ragged:
        offsets = np.asarray(value.offsets)
        row = int(np.searchsorted(offsets, position, side="right") - 1)
        return f"{name}[{row}, {position - int(offsets[row])}]"
    buffer = _buffer(value)
    if buffer is not None and buffer.ndim > 1:
        index = ", ".join(str(int(k)) for k in np.unravel_index(position, buffer.shape))
        return f"{name}[{index}]"
    return f"{name}[{position}]"


# }}}


def check_arguments(
    types: Mapping[str, Any],
    supplied: Mapping[str, Any],
    offsets_args: Mapping[str, str] | None = None,
) -> None:
    """Everything an argument list owes a term, in one call.

    ``types`` maps a parameter to its :class:`~loopty.term.ArrType` or scalar
    sort, ``supplied`` maps a parameter to the value the caller passed, and
    ``offsets_args`` is the lowering's flattening map when there is a lowering.
    Raises :class:`ValueError` naming the argument at fault.
    """
    disjoint_arguments(supplied)
    ragged_arguments(types, supplied, offsets_args)
    element_types(types, supplied)
