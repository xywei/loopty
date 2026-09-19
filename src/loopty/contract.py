"""What an argument list owes a term, checked before anything runs.

Four of loopty's claims are claims about the *call* and not about the term, so
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
a check rather than by a hope. Being a point is two questions and not one: a
range test is a pair of comparisons, and ``nan`` fails both of them, so
integrality (:func:`_not_an_integer`) is asked first and separately.

*A scalar parameter of a refined sort really is one.* The same rule reaches the
other half of the signature. ``i: Fin[n]`` in a kernel writing ``x[i]`` makes
that access in bounds by type just as an indirection is, so ``run(..., i=-1)``
used to reach generated C as an address in front of ``x``.
:func:`scalar_parameters` asks the declared sort of every non-array argument
the same two questions, against the sizes the arrays of the call determine.

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
    "integral_sort",
    "ragged_arguments",
    "resolve_sizes",
    "scalar_parameters",
    "sort_bound",
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
    extent is per row). An axis written as an affine expression in one name,
    ``Fin[n + 1]``, is solved for that name when no bare axis determines it: a
    kernel whose only arrays have extents ``n + 1`` still has an ``n``, and a
    scalar ``i: Fin[n + 1]`` has to be measured against it. An expression in
    several names is left alone.
    """
    from lanky.terms import evaluate, free_variables

    sizes: dict[str, int] = {}
    deferred: list[tuple[Any, int]] = []
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
            elif not isinstance(size, int | np.integer):
                deferred.append((size, int(shape[axis])))
    # Second pass, so that a bare axis always wins over a solved one.
    for size, extent in deferred:
        try:
            free = free_variables(size)
        except Exception:
            continue
        if len(free) != 1:
            continue
        (name,) = free
        if name in sizes:
            continue
        try:
            offset = evaluate(size, {name: 0})
            slope = evaluate(size, {name: 1}) - offset
        except Exception:
            continue
        if not isinstance(slope, int | np.integer) or slope == 0:
            continue
        if (extent - offset) % slope:
            continue
        solved = (extent - offset) // slope
        if solved >= 0:
            sizes[name] = int(solved)
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


def _base_sort(sort: Any) -> Any:
    """A refinement's base sort; anything else unchanged.

    ``Fin[m] & p`` contributes its ``Fin`` here and leaves ``p`` to an oracle:
    a refinement can only narrow the sort, so checking the base is sound and
    checking the proposition is somebody else's job.
    """
    from lanky.prelude import Refined

    return sort.base if isinstance(sort, Refined) else sort


def integral_sort(sort: Any) -> bool:
    """Whether the points of ``sort`` are integers, so a value of it has to be one.

    ``Fin[m]``, ``Nat`` and ``Int`` are the refined integer sorts loopty knows.
    A value of one of them is used as an index or as a count, and both the
    compiled run and the native run read it as a whole number; a fractional or
    non-finite entry is not a point of the sort, whatever numpy stored it in.
    Every other sort (``Real``, a bare numpy dtype in a hand-written term) says
    nothing of the kind and is left alone.
    """
    from lanky.prelude import FinType

    sort = _base_sort(sort)
    if isinstance(sort, FinType):
        return True
    return getattr(sort, "name", None) in ("Nat", "Int")


def sort_bound(
    sort: Any, sizes: Mapping[str, int]
) -> tuple[int, int | None] | None:
    """The half-open range a value of ``sort`` has to lie in, if the sort says.

    ``Fin[m]`` says ``0 <= v < m``, with ``m`` resolved from the sizes the other
    arguments determine; ``Fin[m] & p`` contributes its ``Fin`` and the
    propositions are left to an oracle. ``Nat`` says ``0 <= v`` and nothing
    above, which is ``None`` for the upper end. Every other sort says nothing
    about the value at all, and the whole pair is ``None``.
    """
    from lanky.prelude import FinType

    sort = _base_sort(sort)
    if isinstance(sort, FinType):
        bound = sort.bound
        if isinstance(bound, int | np.integer):
            return (0, int(bound))
        if isinstance(bound, prim.Variable) and bound.name in sizes:
            return (0, int(sizes[bound.name]))
        # ``Fin[n + 1]``: an affine bound is evaluated under the resolved sizes.
        # When a size is missing the upper end is unknown, not absent: a point
        # of *some* ``Fin`` is still never negative, so the floor stays.
        return (0, _evaluate_bound(bound, sizes))
    if getattr(sort, "name", None) == "Nat":
        return (0, None)
    return None


def _evaluate_bound(bound: Any, sizes: Mapping[str, int]) -> int | None:
    """``bound`` as an integer under ``sizes``, or ``None`` when a name is missing."""
    from lanky.terms import evaluate, free_variables

    try:
        free = free_variables(bound)
    except Exception:
        return None
    if not free or not free <= set(sizes):
        return None
    try:
        value = evaluate(bound, {name: int(sizes[name]) for name in free})
    except Exception:
        return None
    if isinstance(value, bool) or not isinstance(value, int | np.integer):
        return None
    return int(value)


def element_bound(
    typ: ArrType, sizes: Mapping[str, int]
) -> tuple[int, int | None] | None:
    """The range an element of ``typ`` has to lie in: :func:`sort_bound` of its sort."""
    return sort_bound(typ.dtype, sizes)


def _not_an_integer(flat: np.ndarray) -> int | None:
    """The flat position of the first entry that is not a finite integer.

    An integer dtype passes without a test. A float array is checked value by
    value and *accepted* when every entry is finite and equal to its own
    rounding: ``0.0`` is the point ``0`` of ``Fin[m]`` however it is stored, and
    the cast :func:`loopty.executor._as_numpy` makes on the way into compiled
    code is exact on such a value, so both runs see the same index. ``1.5`` and
    ``inf`` are refused because that cast would silently truncate them, and
    ``nan`` because it compares false against every bound, which is how it used
    to pass a range test that both of its comparisons failed.
    """
    if flat.dtype.kind in "biu":
        return None
    whole = np.isfinite(flat) & (flat == np.rint(flat))
    offenders = np.flatnonzero(~whole)
    return int(offenders[0]) if offenders.size else None


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

    Being an integer comes before being in range, and is a separate question. A
    range test compares, and a comparison is no test at all for ``nan``, which
    is neither ``< 0`` nor ``>= m`` and used to pass; ``1.5`` passes it honestly
    and is still not a point of ``Fin[m]``, and the cast into the compiled
    kernel's integer dtype would make it the point ``1`` while the native run
    kept the float. So a refined integer sort (``Fin``, ``Nat``, ``Int``) asks
    :func:`_not_an_integer` first. See its docstring for why an integer-valued
    float array is accepted.
    """
    sizes = resolve_sizes(types, supplied) if sizes is None else sizes
    for name, typ in types.items():
        if not isinstance(typ, ArrType):
            continue
        value = supplied.get(name)
        flat = _values(value)
        if flat is None or not flat.size or flat.dtype.kind not in "biufc":
            continue
        if flat.dtype.kind == "c":
            # A complex entry is a whole number only when its imaginary part is
            # zero; anything else would be truncated by the cast into the
            # compiled kernel's integer dtype while the native run refused it.
            if not integral_sort(typ.dtype):
                continue
            whole = (
                np.isfinite(flat) & (flat.imag == 0) & (flat.real == np.rint(flat.real))
            )
            offenders = np.flatnonzero(~whole)
            if offenders.size:
                position = int(offenders[0])
                raise ValueError(
                    f"{_cell_label(name, value, position)} is {flat[position]}, "
                    f"which is not a value of {typ.dtype}: an element of {name} "
                    "has to be a finite whole number, and a complex entry is one "
                    "only when its imaginary part is zero"
                )
            flat = flat.real
        if integral_sort(typ.dtype):
            position = _not_an_integer(flat)
            if position is not None:
                raise ValueError(
                    f"{_cell_label(name, value, position)} is {flat[position]}, "
                    f"which is not a value of {typ.dtype}: an element of {name} "
                    "has to be a finite whole number. loopty discharges an "
                    "indirection through this array as in bounds *by type*, and "
                    "the compiled run reads the element as an integer, so a "
                    "fractional or non-finite entry is an index nobody declared"
                )
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


def _scalar(value: Any) -> float | None:
    """An argument read as one number, or ``None`` when it is not one.

    A ``bool`` is not read as a number: ``True`` is not what anybody means by
    an index, and the rest of loopty already refuses to treat it as one (see
    :func:`resolve_sizes`).
    """
    if isinstance(value, bool | np.bool_):
        return None
    if isinstance(value, int | float | np.integer | np.floating):
        return float(value)
    if isinstance(value, np.ndarray) and value.ndim == 0 and value.dtype.kind in "iuf":
        return float(value)
    return None


def scalar_parameters(
    types: Mapping[str, Any],
    supplied: Mapping[str, Any],
    sizes: Mapping[str, int] | None = None,
) -> None:
    """Refuse a scalar argument that its declared sort excludes.

    The same claim as :func:`element_types`, about the other half of the
    signature. A kernel taking ``i: Fin[n]`` and writing ``x[i]`` has that
    access discharged *by type*, exactly as an indirection through a column
    array is: the declaration says ``i`` is a point of ``Fin[n]``, so the
    generated code contains no test, and ``run(..., i=-1)`` used to reach C as
    an address in front of ``x``. Nothing inside the type system can establish
    it, because it is a property of the call.

    ``Nat`` says non-negative and ``Int`` says nothing but whole, and both are
    asked the same way. The sizes come from the arrays the call supplies, so
    ``Fin[n]`` is only range-checked when some argument determines ``n``;
    without that the value is still required to be an integer.
    """
    sizes = resolve_sizes(types, supplied) if sizes is None else sizes
    for name, sort in types.items():
        if isinstance(sort, ArrType) or not integral_sort(sort):
            continue
        if name not in supplied:
            continue
        value = supplied[name]
        if isinstance(value, bool | np.bool_):
            raise ValueError(
                f"the argument {name} is {value!r}, a boolean, which is not a "
                f"value of {sort}: a point of {sort} is a whole number, and a "
                "boolean would reach the compiled code as 0 or 1 without anybody "
                "having declared that index"
            )
        number = _scalar(value)
        if number is None:
            raise ValueError(
                f"the argument {name} is {value!r} of type "
                f"{type(value).__name__}, which is not a number and so not a "
                f"value of {sort}"
            )
        if not np.isfinite(number) or number != np.rint(number):
            raise ValueError(
                f"the argument {name} is {supplied[name]}, which is not a value "
                f"of {sort}: {name} has to be a finite whole number. loopty "
                f"discharges an access indexed by {name} as in bounds *by "
                "type*, with no check in the generated code"
            )
        limits = sort_bound(sort, sizes)
        if limits is None:
            continue
        low, high = limits
        if low <= number and (high is None or number < high):
            continue
        allowed = f"{low} <= {name}" if high is None else f"{low} <= {name} < {high}"
        raise ValueError(
            f"the argument {name} is {supplied[name]}, which is not a value of "
            f"{sort}: {name} has to satisfy {allowed}. loopty discharges an "
            f"access indexed by {name} as in bounds *by type*, with no check in "
            "the generated code, so the parameter's type has to hold of the "
            "value that is passed in"
        )


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

    The sizes are resolved once and handed to both element checks, so that an
    array and a scalar declared over the same ``Fin[n]`` are measured against
    the same ``n``.
    """
    disjoint_arguments(supplied)
    ragged_arguments(types, supplied, offsets_args)
    sizes = resolve_sizes(types, supplied)
    element_types(types, supplied, sizes)
    scalar_parameters(types, supplied, sizes)
