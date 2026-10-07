"""What an argument list owes a term, checked before anything runs.

Several of loopty's claims are claims about the *call* and not about the term, so
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
whose own offsets disagree with ``cnt`` makes compiled C read past a row. Two
ragged arguments over one counts family are flattened through *one* offsets
argument, so they have to agree with each other as well, and with an offsets
array the caller supplies by hand. :func:`ragged_arguments` checks all three.
Both runs then index through the counts and offsets the kernel declares, as the
kernel leaves them (:meth:`loopty.arr.Arr.through`), and this is the check
that the declared layout is the argument's own when the run starts.

*An element of a refined sort really is one.* ``col: Arr[..., Fin[m]]`` is what
discharges ``x[col[r, j]]`` in bounds **by type**, with no isl call and no
runtime test in the generated code (see :mod:`loopty.typing`). That fact is
sound exactly to the extent that the entries of ``col`` are points of ``Fin[m]``,
which is a property of the data and of nothing else. :func:`element_types`
enforces it on the way in, so the ledger's ``decided by type`` row is backed by
a check rather than by a hope. Being a point is two questions and not one: a
range test is a pair of comparisons, and ``nan`` fails both of them, so
integrality (:func:`_not_an_integer`) is asked first and separately. Being a
point of an integral sort is also being inside :data:`INTEGRAL_RANGE`, the
32-bit integers the compiled run stores one in, since the native run holds it
in 64 bits and would compute with a value the compiled run narrows.

*An array over a domain is over that domain.* ``L: Arr[Where[i: Fin[n], j:
Fin[n], j < i], Real]`` has its in-bounds obligations decided over the exact
triangle, and both runs index ``L`` at its points through a layout computed
from the domain (:mod:`loopty.domain`). An argument over another set of points
would make the compiled run read and write through a table or a box that is
not its own, so :func:`domain_arguments` asks that the argument's points be
the declared domain's at the sizes the call determines.

*A scalar parameter of a refined sort really is one.* The same rule reaches the
other half of the signature. ``i: Fin[n]`` in a kernel writing ``x[i]`` makes
that access in bounds by type just as an indirection is, so ``run(..., i=-1)``
used to reach generated C as an address in front of ``x``.
:func:`scalar_parameters` asks the declared sort of every non-array argument
the same two questions, against the sizes the arrays of the call determine,
and asks one more: an integral scalar has to be stored as an integer, because
neither run can use ``1.0`` as an index.

*An array is stored as its sort is, or it is only read.* The compiled run
converts every array argument into the dtype the lowering stores its element
sort in, and the native run computes in the dtype it is given, so an integer
array passed for ``x: Arr[Fin[n], Real]`` is two arrays: the native run
truncates every real written into it, and the compiled run keeps it. An array
the term writes has to be stored as :func:`native_storage` says
(:func:`written_storage`). One it only reads is read natively through a copy
in that dtype (:func:`read_storage`), which is what the compiled run reads
too. :func:`element_types` asks the values the questions the conversion would
otherwise answer without a word: a complex entry of a sort that is not complex
has no imaginary part, and an entry of ``Bool`` stored as a number is ``0`` or
``1``. A scalar is passed by value, so both runs convert it into that dtype
(:func:`native_scalar`), and :func:`scalar_parameters` asks it the same two
questions.

Nothing here checks array *shapes* against each other; see
``docs/loopy-notes.md`` for why lowering has to declare some arrays without
one, and the README's status list for the consequence. One thing about a shape
is checked, because the facts rest on it: an axis written ``Fin[n + 1]`` is
never shorter than a non-negative ``n`` allows (:func:`sizes_not_negative`).
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from typing import Any

import numpy as np
import pymbolic.primitives as prim

from loopty.arr import Arr
from loopty.idx import is_affine
from loopty.term import ArrType

__all__ = [
    "INT64_RANGE",
    "INTEGRAL_RANGE",
    "INTEGRAL_STORAGE",
    "axis_extents",
    "check_arguments",
    "compiled_storage",
    "counts_family",
    "disjoint_arguments",
    "domain_arguments",
    "element_bound",
    "element_types",
    "holds_natively",
    "inherited_storage",
    "integral_sort",
    "native_copy",
    "native_scalar",
    "native_storage",
    "ragged_arguments",
    "read_storage",
    "resolve_sizes",
    "scalar_parameters",
    "sizes_not_negative",
    "sort_bound",
    "storage_wanted",
    "truth_sort",
    "written_storage",
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
    """An argument's elements as one flat array, or ``None``.

    An array over a domain has its elements at the domain's points, in their
    order (:meth:`loopty.arr.Arr.cells`); a cell its box keeps outside the
    domain is not one of them.
    """
    if isinstance(value, Arr) and value.domain is not None:
        return value.cells()
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
    indexed through: :attr:`loopty.lower.Lowering.ragged` for a compiled run,
    and the offsets the kernel declares (:func:`loopty.term.declared_layout`)
    for a native one, which indexes through them too. Offsets the caller does
    not pass are the array's own, and the third check has nothing to compare.
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

    So is an expression that is not *linear* in its name (:func:`_linear`).
    The solution is read off two evaluations, at ``0`` and at ``1``, which is
    the slope and the offset of a line and nothing else: ``n * n`` evaluates to
    ``0`` and ``1`` there, so an extent of 9 used to give ``n = 9`` rather than
    3, and ``(n + 1) // 2`` an ``n`` of 3 whose axis has 2 cells. A size left
    unresolved is not a size assumed: every check that needs it keeps what it
    can without it.
    """
    from lanky.terms import evaluate, free_variables

    sizes: dict[str, int] = {}
    for name, typ in types.items():
        value = supplied.get(name)
        if value is None:
            continue
        if not isinstance(typ, ArrType):
            if isinstance(value, int | np.integer) and not isinstance(value, bool):
                sizes.setdefault(name, int(value))
            continue
        if typ.domain is not None:
            # An array over a domain was built at sizes of its own, by name; a
            # name the declared domain shares is that size of the call.
            if isinstance(value, Arr) and value.domain is not None:
                declared = typ.domain.size_names()
                for size_name, extent in value.sizes.items():
                    if size_name in declared:
                        sizes.setdefault(size_name, extent)
            continue
        for size, extent in _axis_extents_of(value, typ):
            if isinstance(size, prim.Variable):
                sizes.setdefault(size.name, extent)
    # Second pass, so that a bare axis always wins over a solved one.
    for size, extent in axis_extents(types, supplied):
        try:
            free = free_variables(size)
        except Exception:
            continue
        if len(free) != 1:
            continue
        (name,) = free
        if name in sizes or not _linear(size):
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


def _axis_extents_of(value: Any, typ: ArrType) -> list[tuple[Any, int]]:
    """Every dense axis of one array argument, with the extent it has."""
    shape = _index_shape(value, typ)
    if shape is None:
        return []
    return [
        (size, int(shape[axis]))
        for axis, (size, ragged) in enumerate(zip(typ.axes, typ.ragged, strict=True))
        if not ragged and axis < len(shape)
    ]


def axis_extents(
    types: Mapping[str, Any], supplied: Mapping[str, Any]
) -> tuple[tuple[Any, int], ...]:
    """Every axis written as an expression, with the extent the call gives it.

    ``y: Arr[Fin[n * n], Real]`` of nine cells says that ``n * n`` is 9 in this
    call, whether or not ``n`` itself can be recovered from it (it cannot be,
    by :func:`resolve_sizes`, because the expression is not linear). A value of
    ``Fin[n * n]`` is then measured against 9 directly; see :func:`sort_bound`.
    The terms are kept as terms, in a tuple rather than as dictionary keys,
    because a lanky term answers ``==`` with a proposition.
    """
    return tuple(
        (size, extent)
        for name, typ in types.items()
        if isinstance(typ, ArrType) and supplied.get(name) is not None
        for size, extent in _axis_extents_of(supplied[name], typ)
        if not isinstance(size, prim.Variable | int | np.integer)
    )


def sizes_not_negative(
    types: Mapping[str, Any], supplied: Mapping[str, Any]
) -> None:
    """Refuse an array shorter than its type allows for any value of its sizes.

    A size counts cells, so it is never negative, and every fact about a
    kernel is decided with its sizes non-negative; the lowering tells loopy so
    too (:func:`loopty.lower._scalar_assumptions`). An axis written as an
    expression in one size, ``Fin[n + 1]``, then has a least extent, and an
    argument with fewer cells there stands for a negative ``n``:
    :func:`resolve_sizes` leaves such an ``n`` unresolved, and loopy reads
    ``n = -1`` off an empty ``off``, where ``off[n]`` is the cell in front of
    it. So that argument is refused here, naming the size it would make
    negative.
    """
    from lanky.terms import evaluate, free_variables, render

    for name, typ in types.items():
        value = supplied.get(name)
        if not isinstance(typ, ArrType) or value is None:
            continue
        for size, extent in _axis_extents_of(value, typ):
            if isinstance(size, prim.Variable | int | np.integer):
                continue
            try:
                free = free_variables(size)
            except Exception:
                continue
            if len(free) != 1 or not _linear(size):
                continue
            (size_name,) = free
            try:
                offset = evaluate(size, {size_name: 0})
                slope = evaluate(size, {size_name: 1}) - offset
            except Exception:
                continue
            if not isinstance(slope, int | np.integer) or slope == 0:
                continue
            if (extent - offset) % slope or (extent - offset) // slope >= 0:
                continue
            raise ValueError(
                f"{name} has {extent} cells along an axis its type says is "
                f"{render(size)} long, which would make the size {size_name} "
                f"{(extent - offset) // slope}. A size counts cells and is never "
                "negative, and every fact about the kernel is decided with its "
                "sizes non-negative, so the argument is too short for its type"
            )


#: The dtype the lowering stores an integral sort in, ``Fin[m]``, ``Nat`` and
#: ``Int`` alike: an index into an array is 32 bits wide on every target loopy
#: generates for. See :func:`compiled_storage`.
INTEGRAL_STORAGE = np.dtype(np.int32)

#: The half-open range of the values an integral sort holds compiled, those of
#: :data:`INTEGRAL_STORAGE`. The native run holds such a value in 64 bits
#: (:func:`native_storage`), so one outside this range would run natively as
#: it is and be narrowed by the compiled run's conversion: ``2**32 + 5`` is
#: ``5`` compiled. The contract refuses it (:func:`element_types`,
#: :func:`scalar_parameters`).
INTEGRAL_RANGE = (
    int(np.iinfo(INTEGRAL_STORAGE).min),
    int(np.iinfo(INTEGRAL_STORAGE).max) + 1,
)


def compiled_storage(sort: Any) -> np.dtype | None:
    """The dtype the compiled run stores ``sort`` in, or ``None`` for no sort.

    ``Real`` is double precision, ``Nat``, ``Int`` and an index type such as
    ``Fin[m]`` are :data:`INTEGRAL_STORAGE`, ``Bool`` is a byte (OpenCL takes
    no ``bool`` argument), and a numpy dtype or scalar type is itself.
    Python's ``float`` and ``complex`` are double precision and its ``int``
    is integral. :func:`loopty.lower.numpy_dtype` is this, refusing a sort
    with none; it lives here because the contract and
    :mod:`loopty.promotion` ask it without importing loopy.
    """
    if isinstance(sort, np.dtype):
        return sort
    if isinstance(sort, type) and issubclass(sort, np.generic):
        return np.dtype(sort)
    if sort is float:
        return np.dtype(np.float64)
    if sort is complex:
        return np.dtype(np.complex128)
    if sort is int:
        return INTEGRAL_STORAGE
    if sort is bool:
        return np.dtype(np.int8)
    base = getattr(sort, "base", None)  # a lanky refinement T & prop
    if base is not None and base is not sort:
        return compiled_storage(base)
    name = getattr(sort, "name", None)
    if name == "Real":
        return np.dtype(np.float64)
    if name in ("Nat", "Int"):
        return INTEGRAL_STORAGE
    if name == "Bool":
        return np.dtype(np.int8)
    if hasattr(sort, "bound") or hasattr(sort, "size"):  # an index type Fin[m]
        return INTEGRAL_STORAGE
    return None


def native_storage(sort: Any) -> np.dtype | None:
    """The dtype a native array of ``sort`` holds values in as the compiled one does.

    A program's temporary is stored compiled as the lowering stores its
    element sort (:func:`compiled_storage`), and natively in whatever
    dtype ``Arr.zeros_like`` gave it. The two runs compute one thing only when
    the native array holds every value written into it as the compiled one
    does, which is this dtype:

    * a numpy dtype or scalar type names a storage and is itself:
      ``np.float32`` rounds, and ``np.complex128`` keeps an imaginary part;
    * ``Real``, exact or not, and ``float`` are ``float64``, and ``complex``
      is ``complex128``;
    * ``Bool`` and ``bool`` are ``bool``. The compiled temporary is a byte,
      which holds a truth value as a bool does; natively ``~``, ``&`` and
      ``|`` are logical only on a bool (bitwise on an integer, refused on a
      float), and ``when`` refuses an integer that is not one;
    * an integral sort, ``Fin[m]``, ``Nat``, ``Int`` or ``int``, is ``int64``,
      and any signed integer of 32 bits or more holds it
      (:func:`holds_natively`): the compiled one is 32 bits wide, and a
      native argument of such a sort is 64 bits wide as a rule. Its values
      are inside the narrower range (:data:`INTEGRAL_RANGE`).

    Anything else is ``None``, and is not asked.
    """
    base = _base_sort(sort)
    if isinstance(base, np.dtype):
        return base
    if isinstance(base, type) and issubclass(base, np.generic):
        return np.dtype(base)
    if base is bool or getattr(base, "name", None) == "Bool":
        return np.dtype(np.bool_)
    if base is int or integral_sort(base):
        return np.dtype(np.int64)
    if base is float or getattr(base, "name", None) == "Real":
        return np.dtype(np.float64)
    if base is complex:
        return np.dtype(np.complex128)
    return None


def holds_natively(sort: Any, dtype: Any) -> bool:
    """Whether a native array of ``dtype`` holds ``sort`` as the compiled one does.

    The dtype :func:`native_storage` says, or for an integral sort any signed
    integer of 32 bits or more. A narrower one wraps round where the compiled
    32-bit one does not. A sort with no storage is not asked.
    """
    want = native_storage(sort)
    if want is None:
        return True
    try:
        got = np.dtype(dtype)
    except TypeError:
        return False
    base = _base_sort(sort)
    if base is int or integral_sort(base):
        return got.kind == "i" and got.itemsize >= 4
    return got == want


def storage_wanted(sort: Any) -> str:
    """How :func:`holds_natively` says the native storage of ``sort``, in words."""
    base = _base_sort(sort)
    if base is int or integral_sort(base):
        return "a signed integer of 32 bits or more"
    return str(native_storage(sort))


def inherited_storage(
    types: Mapping[str, Any],
    like: Iterable[tuple[str, str]],
    supplied: Mapping[str, Any],
) -> None:
    """Refuse a parameter whose dtype a program's temporary inherits, if it is wrong.

    ``Arr.zeros_like(u)`` in a program's body is natively an array of ``u``'s
    dtype, whatever ``u`` is called with, and in the compiled program it is a
    temporary of the element sort its kernels declare, stored as that sort is
    (:attr:`loopty.term.Term.temporaries_like` lists them). An integer ``u``
    makes the native one truncate every real written into it, a ``float32``
    one rounds it, and a real one drops the imaginary part of a complex
    value, where the compiled one does none of these, so the two runs would
    compute two things; the call is refused, naming the dtype to give that
    ``Arr.zeros_like``. What each sort has to be stored as natively is
    :func:`native_storage`.

    ``types`` is the type of every array of the term, parameters and
    temporaries alike (:attr:`loopty.term.Term.array_types`). Passing ``u``
    in that dtype is named as a fix too, but only when the dtype also holds
    every value of ``u``'s own element sort: a real ``u`` passed as a bool to
    make a ``Bool`` temporary of it would be a different ``u``.
    """
    for temporary, name in like:
        sort = getattr(types.get(temporary), "dtype", None)
        want = native_storage(sort)
        value = supplied.get(name)
        if want is None or value is None:
            continue
        got = np.asarray(value.numpy() if isinstance(value, Arr) else value).dtype
        if holds_natively(sort, got):
            continue
        fix = f"Give that Arr.zeros_like dtype={want}"
        own = native_storage(getattr(types.get(name), "dtype", None))
        if own is not None and np.can_cast(own, want, casting="safe"):
            fix += f", or pass {name} as {want}"
        raise ValueError(
            f"the argument {name} is stored as {got}, and the program makes "
            f"{temporary} with Arr.zeros_like from it, which natively is an "
            f"array of {got} too; the kernels {temporary} is passed to declare "
            f"its elements {_shown_sort(sort)}, which the native run has to "
            f"store as {storage_wanted(sort)} to hold them as the compiled "
            f"program does, so the two runs would compute {temporary} "
            f"differently. {fix}"
        )


def written_storage(
    types: Mapping[str, Any],
    supplied: Mapping[str, Any],
    written: Iterable[str],
) -> None:
    """Refuse an array argument that is written and not stored as its sort is.

    The compiled run converts every array argument into the dtype the
    lowering stores its element sort in (:func:`compiled_storage`),
    computes in that, and writes the results back; the native run computes in
    the array it was given. An array that is written holds what each run
    writes into it, and the two hold one thing only when the native array is
    stored as :func:`native_storage` says. An integer ``x`` for ``x:
    Arr[Fin[n], Real]`` truncates every write the compiled one keeps, so
    halving ``[3, 5]`` and doubling it again leaves ``[2, 4]`` natively and
    ``[3, 5]`` compiled; a ``float32`` one rounds what the compiled one keeps;
    a complex one keeps an imaginary part the compiled one never had; a real
    one for ``Bool`` refuses ``~``. A copy in the right dtype would make the
    runs agree and break the native run's promise that its writes land in the
    caller's array, so the call is refused, naming the dtype to pass. An array
    that is only read is read through such a copy instead (:func:`read_storage`).

    ``written`` names the arrays the term writes: the lowering's outputs for a
    compiled run, and the term's assignees for a native one.
    """
    written = set(written)
    for name, typ in types.items():
        if name not in written or not isinstance(typ, ArrType):
            continue
        buffer = _buffer(supplied.get(name))
        if buffer is None:
            continue
        sort = typ.dtype
        want = native_storage(sort)
        if want is None or holds_natively(sort, buffer.dtype):
            continue
        raise ValueError(
            f"the argument {name} is stored as {buffer.dtype}, and {name} is "
            f"written: its elements are {_shown_sort(sort)}, which the native "
            f"run has to store as {storage_wanted(sort)} to hold what is written "
            "into them as the compiled run does, so the two runs would compute "
            f"{name} differently. Pass {name} as {want}"
        )


def read_storage(sort: Any, dtype: Any) -> np.dtype | None:
    """The dtype the native run reads an array of ``sort`` stored as ``dtype`` in.

    ``None`` when the array holds ``sort`` natively as it is
    (:func:`holds_natively`), and :func:`native_storage` otherwise: an integer
    array of ``Real`` elements is read as ``float64``, a float one of
    ``Fin[m]`` as ``int64``, one of ``0`` and ``1`` for ``Bool`` as ``bool``.
    That is what the compiled run reads as well, since it converts the array
    into its own dtype on the way in, but for an integral sort, which is 32
    bits wide there (see :func:`holds_natively`). The native run used to
    compute in the dtype it was given, so an integer ``x`` of a ``Real``
    parameter overflowed at ``x[i] * x[i]`` where the compiled run squares a
    double, and a ``bool`` one added ``True + True`` to ``True``. A dtype that
    is not a number's is ``None`` too: nothing converts it.

    Only an array the term does not write is read through a copy; one it
    writes is refused instead (:func:`written_storage`).
    """
    want = native_storage(sort)
    if want is None:
        return None
    try:
        got = np.dtype(dtype)
    except TypeError:
        return None
    if got.kind not in "biufc" or holds_natively(sort, got):
        return None
    return want


def native_copy(value: Any, dtype: np.dtype) -> Any:
    """A copy of an array argument in ``dtype``: what :func:`read_storage` asks for.

    An :class:`~loopty.arr.Arr` stays one, ragged or over a domain, with its
    own layout. A complex array loses its imaginary part only for a dtype that
    is not complex, for whose sort :func:`element_types` has required it to be
    zero; it is dropped here and not by the cast, which would warn.
    """
    buffer = value.numpy() if isinstance(value, Arr) else np.asarray(value)
    if buffer.dtype.kind == "c" and dtype.kind != "c":
        buffer = buffer.real
    converted = buffer.astype(dtype)
    return value._replaced(converted) if isinstance(value, Arr) else converted


def native_scalar(sort: Any, value: Any) -> Any:
    """A scalar argument as the native run computes with it, for ``sort``.

    The compiled run passes a scalar as the C type the lowering declares for
    its sort (:func:`compiled_storage`), and the native run used to
    compute with whatever it was given: ``np.int64(2**32)`` for ``a: Real``
    overflowed at ``a * a`` where the compiled run squares a double,
    ``np.int8(100)`` for ``a: Nat`` wrapped at ``a + a``, and a ``np.float32``
    added in single precision. A scalar is passed by value, so converting it
    breaks no promise about where writes land, and the native run and the
    interpreter compute with it in :func:`native_storage`'s dtype, as the
    compiled run does. A truth value is a numpy bool, Python's ``True``
    included: ``~`` on a Python bool is the integer ``-2``, which ``when``
    refuses and a bool array stores as ``True``, where the compiled run
    computes ``!flag``.

    A value that holds its sort already (:func:`holds_natively`) is returned
    as it is, and so is a Python ``int`` of an integral sort, which no
    conversion makes more exact, and anything that is not a number.
    :func:`scalar_parameters` has required the value to be one of the sort: a
    whole number for an integral one, ``0`` or ``1`` for ``Bool``, and no
    imaginary part for a sort that is not complex, which is dropped here.
    """
    want = native_storage(sort)
    if want is None:
        return value
    try:
        got = np.asarray(value).dtype
    except (TypeError, ValueError, OverflowError):
        return value
    if np.ndim(value) or got.kind not in "biufc":
        return value
    base = _base_sort(sort)
    if (base is int or integral_sort(base)) and isinstance(value, int):
        return value
    if want.kind == "b":
        return value if isinstance(value, np.bool_) else np.bool_(np.real(value))
    if holds_natively(sort, got):
        return value
    if got.kind == "c" and want.kind != "c":
        value = np.real(value)
    return want.type(value)


def _shown_sort(sort: Any) -> str:
    """A sort as a message says it: a numpy scalar type by its dtype's name."""
    if isinstance(sort, type) and issubclass(sort, np.generic):
        return np.dtype(sort).name
    return str(sort)


def _linear(expr: Any) -> bool:
    """Whether ``expr`` is an integer combination of names plus a constant.

    :func:`loopty.idx.is_affine` without the quasi: floor division and
    remainder are affine to isl, but they are not invertible, and a size is
    solved for by inverting its axis. A product with more than one factor that
    is not an integer literal is not linear either, which is what rules out
    ``n * n``.
    """
    if isinstance(expr, prim.FloorDiv | prim.Remainder):
        return False
    if isinstance(expr, prim.Sum | prim.Product):
        return is_affine(expr) and all(_linear(child) for child in expr.children)
    return is_affine(expr)


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


def truth_sort(sort: Any) -> bool:
    """Whether the points of ``sort`` are truth values compiled code keeps in a byte.

    ``Bool``, and Python's ``bool``, which the lowering stores as a byte
    (:func:`compiled_storage`) because OpenCL takes no ``bool``
    argument, and the native run as a numpy bool (:func:`native_storage`).
    The two hold one value only while the byte is ``0`` or ``1``: C converts
    ``0.5`` into a byte as ``0`` and keeps ``2`` as ``2``, where a bool holds
    ``True`` for both. A numpy bool given as the sort is C's ``bool`` compiled,
    whose conversion is numpy's, and is not one of these.
    """
    base = _base_sort(sort)
    return base is bool or getattr(base, "name", None) == "Bool"


def sort_bound(
    sort: Any,
    sizes: Mapping[str, int],
    extents: tuple[tuple[Any, int], ...] = (),
) -> tuple[int, int | None] | None:
    """The half-open range a value of ``sort`` has to lie in, if the sort says.

    ``Fin[m]`` says ``0 <= v < m``, with ``m`` resolved from the sizes the other
    arguments determine; ``Fin[m] & p`` contributes its ``Fin`` and the
    propositions are left to an oracle. ``Nat`` says ``0 <= v`` and nothing
    above, which is ``None`` for the upper end. Every other sort says nothing
    about the value at all, and the whole pair is ``None``.

    A bound written as an expression is evaluated under the sizes and, when a
    name in it is not resolved, looked up among ``extents``
    (:func:`axis_extents`): ``Fin[n * n]`` is bounded by the extent of an axis
    written ``Fin[n * n]`` even though ``n`` is not known.
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
        high = _evaluate_bound(bound, sizes)
        if high is None:
            high = _matching_extent(bound, extents)
        return (0, high)
    if getattr(sort, "name", None) == "Nat":
        return (0, None)
    return None


def _matching_extent(bound: Any, extents: tuple[tuple[Any, int], ...]) -> int | None:
    """The extent of an axis written exactly as ``bound``, or ``None``.

    Two axes written the same way but given different extents by a call are a
    shape mismatch, which is not this module's to diagnose; the smaller one is
    the one every value of the sort has to fit, so it is the one returned.
    """
    from lanky.terms import structurally_equal

    matches = [
        extent for expr, extent in extents if structurally_equal(expr, bound)
    ]
    return min(matches) if matches else None


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
    typ: ArrType,
    sizes: Mapping[str, int],
    extents: tuple[tuple[Any, int], ...] = (),
) -> tuple[int, int | None] | None:
    """The range an element of ``typ`` has to lie in: :func:`sort_bound` of its sort."""
    return sort_bound(typ.dtype, sizes, extents)


#: The half-open range of the integers a float-stored element of an integral
#: sort is converted to: the native run reads such an array as ``int64`` (see
#: :meth:`loopty.kernel.Kernel._storage_copies`). Both ends are floats that
#: ``float64`` holds exactly, and every whole float inside them converts
#: exactly.
INT64_RANGE = (-(2.0**63), 2.0**63)


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

    A whole float is also refused outside :data:`INT64_RANGE`. The native run
    reads a float-stored index array as ``int64``, and ``1e20`` is a whole
    float that no ``int64`` holds: the conversion used to give an unrelated
    integer, and the reference run computed with it. For ``Fin[m]`` the range
    test would refuse such a value anyway; ``Nat`` and ``Int`` have no upper
    end, so this is the check that does.
    """
    if flat.dtype.kind in "biu":
        return None
    low, high = INT64_RANGE
    whole = np.isfinite(flat) & (flat == np.rint(flat)) & (flat >= low) & (flat < high)
    offenders = np.flatnonzero(~whole)
    return int(offenders[0]) if offenders.size else None


def element_types(
    types: Mapping[str, Any],
    supplied: Mapping[str, Any],
    sizes: Mapping[str, int] | None = None,
    extents: tuple[tuple[Any, int], ...] | None = None,
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

    Two more sorts say something about a value that the compiled run's
    conversion would otherwise change. A complex entry is a value of a sort
    that is not complex, ``Real`` say, only when its imaginary part is zero:
    the cast into ``float64`` drops it, and the native run computes with it.
    And a ``Bool`` entry stored as a number has to be ``0`` or ``1``
    (:func:`truth_sort`), because the compiled run converts it into a byte as
    it is, and the native run reads it as a truth value.

    An entry of an integral sort is also inside :data:`INTEGRAL_RANGE`, the
    32-bit integers the compiled run stores it in. The native run holds it in
    64 bits, so ``2**32 + 5`` of a ``Nat`` used to run natively as it is and be
    ``5`` compiled, and a ``uint64`` entry from ``2**63`` on was read natively
    through an ``int64`` copy as a negative number.
    """
    sizes = resolve_sizes(types, supplied) if sizes is None else sizes
    extents = axis_extents(types, supplied) if extents is None else extents
    for name, typ in types.items():
        if not isinstance(typ, ArrType):
            continue
        value = supplied.get(name)
        flat = _values(value)
        if flat is None or not flat.size or flat.dtype.kind not in "biufc":
            continue
        if flat.dtype.kind == "c":
            storage = native_storage(typ.dtype)
            if storage is None or storage.kind == "c":
                continue
            if integral_sort(typ.dtype):
                # A complex entry is a whole number only when its imaginary
                # part is zero; anything else would be truncated by the cast
                # into the compiled kernel's integer dtype while the native run
                # refused it.
                whole = (
                    np.isfinite(flat)
                    & (flat.imag == 0)
                    & (flat.real == np.rint(flat.real))
                )
                offenders = np.flatnonzero(~whole)
                if offenders.size:
                    position = int(offenders[0])
                    raise ValueError(
                        f"{_cell_label(name, value, position)} is "
                        f"{flat[position]}, which is not a value of {typ.dtype}: "
                        f"an element of {name} has to be a finite whole number, "
                        "and a complex entry is one only when its imaginary part "
                        "is zero"
                    )
            # Any other sort that is not complex: the cast into the compiled
            # kernel's dtype drops the imaginary part, which the native run
            # computes with.
            offenders = np.flatnonzero(flat.imag != 0)
            if offenders.size:
                position = int(offenders[0])
                raise ValueError(
                    f"{_cell_label(name, value, position)} is {flat[position]}, "
                    f"which is not a value of {_shown_sort(typ.dtype)}: a complex "
                    "entry is one only when its imaginary part is zero. The "
                    f"compiled run converts {name} into the dtype it stores "
                    f"{_shown_sort(typ.dtype)} in, which drops the imaginary "
                    "part, while the native run computes with it"
                )
            flat = flat.real
        if truth_sort(typ.dtype) and flat.dtype.kind != "b":
            offenders = np.flatnonzero((flat != 0) & (flat != 1))
            if offenders.size:
                position = int(offenders[0])
                raise ValueError(
                    f"{_cell_label(name, value, position)} is {flat[position]}, "
                    f"which is not a value of {typ.dtype}: an element of {name} "
                    "has to be a truth value, stored as a bool or as 0 or 1. The "
                    f"compiled run stores {name} as bytes, and a byte holds a "
                    "truth value as a bool does only when it is 0 or 1: C "
                    "converts 0.5 into a byte as 0 and keeps 2 as 2, where a "
                    "bool holds True for both"
                )
        if integral_sort(typ.dtype):
            position = _not_an_integer(flat)
            if position is not None:
                raise ValueError(
                    f"{_cell_label(name, value, position)} is {flat[position]}, "
                    f"which is not a value of {typ.dtype}: an element of {name} "
                    "has to be a finite whole number, and one stored as a float "
                    "has to fit in a 64-bit integer. loopty discharges an "
                    "indirection through this array as in bounds *by type*, and "
                    "both runs read the element as an integer, so a fractional, "
                    "non-finite or unrepresentable entry is an index nobody "
                    "declared"
                )
        limits = element_bound(typ, sizes, extents)
        if limits is not None:
            low, high = limits
            outside = flat < low if high is None else (flat < low) | (flat >= high)
            offenders = np.flatnonzero(outside)
            if offenders.size:
                position = int(offenders[0])
                where = _cell_label(name, value, position)
                allowed = f"{low} <= v" if high is None else f"{low} <= v < {high}"
                raise ValueError(
                    f"{where} is {flat[position]}, which is not a value of "
                    f"{typ.dtype}: an element of {name} has to satisfy {allowed}. "
                    "loopty discharges an indirection through this array as in "
                    "bounds *by type*, with no check in the generated code, so the "
                    "element type has to hold of the data that is passed in"
                )
        if integral_sort(typ.dtype):
            low, high = INTEGRAL_RANGE
            offenders = np.flatnonzero((flat < low) | (flat >= high))
            if offenders.size:
                position = int(offenders[0])
                raise ValueError(
                    _integral_range_message(
                        _cell_label(name, value, position),
                        flat[position],
                        typ.dtype,
                        f"the elements of {name}",
                    )
                )


def _integral_range_message(where: str, value: Any, sort: Any, held: str) -> str:
    """The refusal of a value of an integral sort outside :data:`INTEGRAL_RANGE`.

    ``held`` names what is declared of the sort: an argument, or the elements
    of an array.
    """
    low, high = INTEGRAL_RANGE
    return (
        f"{where} is {value}, which is outside {low} <= v < {high}, the range of "
        f"the 32-bit integers the compiled run stores a value of {sort} in. The "
        "native run holds it in 64 bits, so it would run natively as it is and "
        "be narrowed by the compiled run's conversion, and the two runs would "
        "compute different things. Pass a value inside that range, or declare "
        f"{held} as a numpy integer such as np.int64, which both runs store as "
        "it is"
    )


def _cell_label(name: str, value: Any, position: int) -> str:
    """``col[1, 0]``: where in an argument the flat offset ``position`` is.

    For an array over a domain the position is among its cells, in the order
    of :meth:`loopty.arr.Arr.cells`.
    """
    if isinstance(value, Arr) and value.domain is not None:
        point = value.domain.points()[position]
        return f"{name}[{', '.join(str(k) for k in point)}]"
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
    extents: tuple[tuple[Any, int], ...] | None = None,
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

    An integral scalar has to be *stored* as an integer: a Python ``int``, a
    numpy integer or a zero-dimensional integer array. ``1.0`` is refused
    although it is a whole number, unlike the float-stored arrays
    :func:`element_types` accepts, because neither run can use it: the
    compiled run passes it to a C integer argument, which a float cannot be
    converted to, and the native run indexes with it, which numpy refuses.
    Converting it would be the caller's choice to make, so the message says
    ``int(...)``. It is inside :data:`INTEGRAL_RANGE` too, which is what the
    compiled run's C integer argument holds; the native run computes with it
    as it is, so a value outside would be narrowed by one run only.

    A scalar of any other sort is asked what :func:`element_types` asks an
    entry, because both runs convert it into its sort's dtype
    (:func:`native_scalar`): a complex one of a sort that is not complex has
    no imaginary part, and one of ``Bool`` given as a number is ``0`` or ``1``
    (see :func:`_scalar_value`).
    """
    sizes = resolve_sizes(types, supplied) if sizes is None else sizes
    extents = axis_extents(types, supplied) if extents is None else extents
    for name, sort in types.items():
        if isinstance(sort, ArrType) or name not in supplied:
            continue
        if not integral_sort(sort):
            _scalar_value(name, sort, supplied[name])
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
        if np.asarray(value).dtype.kind == "f":
            whole = bool(np.isfinite(number) and number == np.rint(number))
            advice = (
                f"Pass int({name}) if the whole number is what you meant"
                if whole
                else "It is not a finite whole number either"
            )
            raise ValueError(
                f"the argument {name} is {value!r}, a float, which is not a "
                f"value of {sort}: an integral parameter has to be passed as an "
                "integer (a Python int, a numpy integer or a zero-dimensional "
                "integer array). The compiled run cannot pass a float to a C "
                f"integer argument, and the native run cannot index with one. {advice}"
            )
        limits = sort_bound(sort, sizes, extents)
        if limits is not None:
            low, high = limits
            if not (low <= number and (high is None or number < high)):
                allowed = (
                    f"{low} <= {name}" if high is None else f"{low} <= {name} < {high}"
                )
                raise ValueError(
                    f"the argument {name} is {supplied[name]}, which is not a value "
                    f"of {sort}: {name} has to satisfy {allowed}. loopty discharges "
                    f"an access indexed by {name} as in bounds *by type*, with no "
                    "check in the generated code, so the parameter's type has to "
                    "hold of the value that is passed in"
                )
        low, high = INTEGRAL_RANGE
        if not low <= int(value) < high:
            raise ValueError(
                _integral_range_message(f"the argument {name}", value, sort, name)
            )


def _scalar_value(name: str, sort: Any, value: Any) -> None:
    """Refuse a scalar of a sort that is not integral, if its conversion changes it.

    Both runs convert the scalar into the dtype of its sort: the compiled one
    into the C type the lowering declares, the native one and the interpreter
    into :func:`native_storage`'s dtype (:func:`native_scalar`). A complex
    value is converted into a sort that is not complex by dropping its
    imaginary part, so it is a value of the sort only when that part is zero.
    A number given for ``Bool`` reaches compiled code as a byte, and the native
    run as a bool, and the two hold one value only for ``0`` and ``1``: ``2``
    is stored as ``2`` compiled and as ``True`` natively. Anything that is not
    a number is left to the runs.
    """
    want = native_storage(sort)
    if want is None or isinstance(value, bool | np.bool_):
        return
    try:
        number = np.asarray(value)
    except (TypeError, ValueError, OverflowError):
        return
    if number.ndim or number.dtype.kind not in "iufc":
        return
    if number.dtype.kind == "c" and want.kind != "c" and number.imag != 0:
        raise ValueError(
            f"the argument {name} is {value!r}, which is not a value of "
            f"{_shown_sort(sort)}: a complex number is one only when its "
            f"imaginary part is zero. Both runs convert {name} into the dtype "
            f"{_shown_sort(sort)} is stored in, which drops the imaginary part"
        )
    if truth_sort(sort) and np.real(number) not in (0, 1):
        raise ValueError(
            f"the argument {name} is {value!r}, which is not a value of {sort}: "
            f"{name} has to be a truth value, given as a bool or as 0 or 1. The "
            f"compiled run passes {name} as a byte, and a byte holds a truth "
            "value as a bool does only when it is 0 or 1: C converts 0.5 into a "
            "byte as 0 and keeps 2 as 2, where a bool holds True for both"
        )


# }}}


def domain_arguments(
    types: Mapping[str, Any],
    supplied: Mapping[str, Any],
    sizes: Mapping[str, int] | None = None,
) -> None:
    """Refuse an argument that is not an array over its declared domain.

    ``L: Arr[Where[i: Fin[n], j: Fin[n], j < i], Real]`` needs an array whose
    points are the triangle's at the ``n`` of the call: built with
    :meth:`loopty.arr.Arr.zeros` or :meth:`~loopty.arr.Arr.from_cells` from the
    kernel's own domain (``kernel.arg_types["L"].domain``) or from any domain
    with the same points. The points are compared, not the spelling. A plain
    ``ndarray`` is refused, because nothing in it says which of its cells are
    the domain's, and so is an array over other points: the compiled run would
    read it through a box or a table of rows that is not its own, and the
    in-bounds facts are about the declared domain. The sizes the domain names
    have to be determined by the call, by its arrays or its scalars, and a size
    a binder's bound runs up to has to be non-negative, as the facts and the
    layouts assume it is (a scalar can be negative where an array's extent
    cannot). The other way round is refused too: an array over a domain passed
    for a parameter
    whose type is a box, or ragged rows, has cells that type does not have.
    """
    from loopty.domain import fixed_set, same_points

    sizes = resolve_sizes(types, supplied) if sizes is None else sizes
    for name, typ in types.items():
        if not isinstance(typ, ArrType) or name not in supplied:
            continue
        value = supplied[name]
        if typ.domain is None:
            if isinstance(value, Arr) and value.domain is not None:
                raise ValueError(
                    f"the argument {name} is an array over {value.domain!r}, and "
                    f"{name}'s type has no domain: its cells are a box, or ragged "
                    "rows, and an array over a domain has only the domain's"
                )
            continue
        if not (isinstance(value, Arr) and value.domain is not None):
            what = (
                "an array with no domain"
                if isinstance(value, Arr | np.ndarray)
                else f"{value!r}"
            )
            raise ValueError(
                f"the argument {name} is {what}, and {name}: Arr[{typ.domain}, "
                "...] is an array over a domain, which says which cells are the "
                "array's. Build it with Arr.zeros(domain, ...) or "
                "Arr.from_cells(domain, values, ...), passing the sizes by name; "
                "the kernel's own domain is kernel.arg_types"
                f"[{name!r}].domain"
            )
        needed = typ.domain.size_names()
        missing = sorted(needed - set(sizes))
        if missing:
            raise ValueError(
                f"the call does not determine {', '.join(missing)}, which the "
                f"domain {typ.domain} of {name} names, so the points {name} has "
                "to have are not known. Pass an array whose axis is that size, "
                f"or build {name} over the kernel's own domain, whose sizes "
                "have those names"
            )
        fixed = {size: sizes[size] for size in needed}
        negative = sorted(
            size for size in typ.domain.extent_names() if fixed[size] < 0
        )
        if negative:
            raise ValueError(
                f"the call gives {negative[0]} = {fixed[negative[0]]}, a size a "
                f"bound of the domain {typ.domain} of {name} runs up to. Such a "
                "size is an extent and is never negative: the in-bounds facts "
                "and the layouts are stated with it non-negative, as a loop "
                "nest's sizes are"
            )
        same, witness = same_points(
            fixed_set(typ.domain, fixed), value.domain.isl_points()
        )
        if same:
            continue
        if witness is None:
            differ = (
                f"the one has {value.domain.ndim} axes and the other "
                f"{typ.domain.ndim}"
            )
        elif value.domain.contains(witness):
            differ = (
                f"{list(witness)} is a point of the argument's and not of the "
                "declared one"
            )
        else:
            differ = (
                f"{list(witness)} is a point of the declared domain and not of "
                "the argument's"
            )
        declared = ", ".join(f"{size}={fixed[size]}" for size in sorted(fixed))
        raise ValueError(
            f"the argument {name} is over {value.domain!r}, and its type says "
            f"{typ.domain} at {declared or 'no sizes'}: {differ}. Both runs "
            f"index {name} at the declared "
            "domain's points through a layout computed from it, and the "
            "in-bounds facts are about that domain, so an array over other "
            "points would be read and written through a layout that is not "
            "its own"
        )


def check_arguments(
    types: Mapping[str, Any],
    supplied: Mapping[str, Any],
    offsets_args: Mapping[str, str] | None = None,
    written: Iterable[str] | None = None,
) -> None:
    """Everything an argument list owes a term, in one call.

    ``types`` maps a parameter to its :class:`~loopty.term.ArrType` or scalar
    sort, ``supplied`` maps a parameter to the value the caller passed, and
    ``offsets_args`` is the lowering's flattening map when there is a lowering.
    ``written`` names the arrays the term writes, when the caller knows them,
    and their storage is checked too (:func:`written_storage`); a native call
    knows them only once the body is traced, which it does only when some
    array is not stored as its sort is (see
    :meth:`loopty.kernel.Kernel._storage_copies`). Raises :class:`ValueError`
    naming the argument at fault.

    The sizes are resolved once and handed to both element checks, so that an
    array and a scalar declared over the same ``Fin[n]`` are measured against
    the same ``n``.
    """
    disjoint_arguments(supplied)
    ragged_arguments(types, supplied, offsets_args)
    sizes_not_negative(types, supplied)
    sizes = resolve_sizes(types, supplied)
    domain_arguments(types, supplied, sizes)
    extents = axis_extents(types, supplied)
    element_types(types, supplied, sizes, extents)
    scalar_parameters(types, supplied, sizes, extents)
    if written is not None:
        written_storage(types, supplied, written)
