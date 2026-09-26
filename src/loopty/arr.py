"""``Arr``: the runtime array, dense or ragged, backed by numpy.

The same object serves two masters. Under plain ``python`` a kernel body runs
against real data and ``Arr`` is an ordinary array with bounds checking. Under
tracing (``loopty.trace``, wave 2) the body runs once against a symbolic proxy
with the same surface, so the body needs no variant: it iterates ``arr.dom``
rather than a ``range`` of a size name, and a ragged inner loop iterates
``arr.dom[r]``. That is the point of ``.dom``: the sizes come from the data, so
the body is closed and can be executed, while the type says where those sizes
come from.

Ragged storage is a dependent sum stored flat: row ``r`` occupies
``offsets[r]:offsets[r+1]`` of a one-dimensional buffer, and the per-row count
is the difference of consecutive offsets. Indexing is ``arr[r, j]``, in the
index-type axes, never in flat addresses; the layout (``loopty.idx``) owns the
translation.

Inside a kernel a ragged array is read through the layout the kernel declares
rather than its own (:meth:`Arr.through`): the counts array its type names
bounds a row, and a declared offsets array says where a row starts, as they do
in the lowered kernel, which is handed those arrays and reads them as the
kernel has left them.

An array over a polyhedral domain (``Arr[Where[i: Fin[n], j: Fin[n], j < i],
Real]``, a ``Sigma[...]`` or a union of pieces; see :mod:`loopty.domain`) holds
the domain at its sizes, and is indexed at the domain's points and nowhere
else: a cell of the bounding box outside the triangle is not a cell of the
array, whatever the storage keeps there. It is stored in one of two layouts,
the box or the packed rows, and :meth:`Arr.cells` reads it in one order that
does not depend on which.
"""

from __future__ import annotations

from collections.abc import Iterator, Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np

from loopty.domain import STORAGES, Fixed, index_domain
from loopty.idx import axis_size
from loopty.term import ArrType

__all__ = ["Arr", "ArrSpec", "Dom"]


@dataclass(frozen=True)
class ArrSpec:
    """The written form ``Arr[Fin[n], Fin[cnt], Real]`` of an array type.

    Annotations are evaluated, not parsed, so ``Arr[...]`` has to produce an
    object. This is that object: the axes as written (index types or terms) and
    the element sort. Turning it into an :class:`~loopty.term.ArrType`, which
    needs to know which axes are ragged, is the tracer's job, because
    raggedness is visible only once the axis names are matched against the other
    parameters.
    """

    axes: tuple[Any, ...]
    dtype: Any

    def __repr__(self) -> str:
        inner = ", ".join(repr(a) for a in (*self.axes, self.dtype))
        return f"Arr[{inner}]"


class Dom:
    """The iteration domain of one axis of an array.

    ``arr.dom`` is the outer axis; ``arr.dom[i]`` is the fiber over ``i``, which
    for a ragged array has a different extent for each ``i``, and
    ``arr.dom[i, j]`` is ``arr.dom[i][j]``. Iterating yields
    the concrete indices; under tracing the symbolic counterpart yields a single
    generic point and pushes the bound onto the enclosing isl domain.
    """

    __slots__ = ("array", "prefix")

    def __init__(self, array: Arr, prefix: tuple[int, ...] = ()) -> None:
        self.array = array
        self.prefix = prefix

    @property
    def axis(self) -> int:
        """Which axis of the array this domain runs over."""
        return len(self.prefix)

    @property
    def size(self) -> int:
        """The extent of this axis, given the fixed prefix.

        Over a domain, the bound of the axis's binder at the prefix, which is
        what the traced ``.size`` is too; the points it runs over may be
        fewer, and ``len`` counts those.
        """
        return self.array._extent(self.prefix)

    def __len__(self) -> int:
        return len(self.array._fiber(self.prefix))

    def __iter__(self) -> Iterator[int]:
        return iter(self.array._fiber(self.prefix))

    def __getitem__(self, index: int | tuple[int, ...]) -> Dom:
        """The fiber over ``index``: the domain of the next axis.

        A tuple fixes one axis per entry, the way ``arr[r, j]`` indexes a cell,
        so ``a.dom[r, i]`` is ``a.dom[r][i]``.
        """
        if isinstance(index, tuple):
            if not index:
                raise TypeError("a domain index needs at least one entry")
            fiber = self
            for part in index:
                fiber = fiber[part]
            return fiber
        if not isinstance(index, int | np.integer):
            raise TypeError(f"domain index must be an integer, got {index!r}")
        domain = self.array.domain
        if domain is not None:
            # Over a domain a fiber at a point outside it is empty, as it is in
            # a trace (see loopty.domain); a piece of a union has to exist.
            if self.axis + 1 >= self.array.ndim:
                raise IndexError(
                    f"{self.array!r} has {self.array.ndim} axes; "
                    f"there is no axis {self.axis + 1} to take a fiber of"
                )
            if domain.union and self.axis == 0:
                domain.split((int(index),))
            return Dom(self.array, (*self.prefix, int(index)))
        size = self.size
        if not 0 <= int(index) < size:
            raise IndexError(f"index {index} out of range for axis of size {size}")
        if self.axis + 1 >= self.array.ndim:
            raise IndexError(
                f"{self.array!r} has {self.array.ndim} axes; "
                f"there is no axis {self.axis + 1} to take a fiber of"
            )
        return Dom(self.array, (*self.prefix, int(index)))

    def __repr__(self) -> str:
        return f"Dom(axis={self.axis}, size={self.size})"


class Arr:
    """A dense or ragged array over numpy.

    Build one with :meth:`zeros`, :meth:`from_numpy`, or :meth:`ragged`. A dense
    array wraps an ``ndarray`` of the same shape; a ragged array wraps a flat
    values buffer plus an offsets array of length ``nrows + 1``, so that row
    ``r`` is ``values[offsets[r]:offsets[r+1]]``. An array over a polyhedral
    domain is built with :meth:`zeros` or :meth:`from_cells` from the domain
    and its sizes, and wraps the buffer of its layout (see
    :mod:`loopty.domain`).
    """

    __slots__ = ("_domain", "_layout", "_offsets", "_storage", "_values")

    def __init__(
        self, values: np.ndarray, offsets: np.ndarray | None = None
    ) -> None:
        values = np.asarray(values)
        if offsets is None:
            self._values = values
            self._offsets: np.ndarray | None = None
        else:
            offsets = np.asarray(offsets, dtype=np.int64)
            if values.ndim != 1:
                raise ValueError("a ragged array stores its values flat (1-D)")
            if offsets.ndim != 1 or offsets.size < 1:
                raise ValueError("offsets must be a 1-D array with at least one entry")
            if np.any(np.diff(offsets) < 0):
                raise ValueError("offsets must be non-decreasing")
            if int(offsets[0]) != 0:
                # Only differences and the last entry were checked once, so a
                # monotone but shifted family such as [-1, 2] was accepted. It
                # cannot be: native indexing computes offsets[r] + column, and a
                # negative flat index wraps round to the end of the buffer under
                # numpy while the generated C reads in front of it. The CSR
                # convention is the fix and the contract, so it is enforced.
                raise ValueError(
                    f"offsets must start at 0, not {int(offsets[0])}: row r "
                    "occupies values[offsets[r]:offsets[r + 1]] of the flat "
                    "buffer, so the first row starts at its beginning"
                )
            if np.any(offsets < 0):  # pragma: no cover - implied by the two above
                raise ValueError("offsets must be non-negative")
            if int(offsets[-1]) != values.size:
                raise ValueError(
                    f"offsets end at {int(offsets[-1])} but there are "
                    f"{values.size} values"
                )
            self._values = values
            self._offsets = offsets
        self._layout: tuple[Any, Any] | None = None
        self._domain: Fixed | None = None
        self._storage: str | None = None

    # -- constructors ----------------------------------------------------

    @classmethod
    def zeros(
        cls, shape: Any, dtype: Any = np.float64, *, storage: str = "box", **sizes: int
    ) -> Arr:
        """An array of zeros over ``shape``.

        ``shape`` is an index type or a tuple of them: ``Arr.zeros(Fin[4])`` and
        ``Arr.zeros((Fin[3], 4))`` both work, as does a plain int. It may also
        be a polyhedral domain, ``Where[...]``, ``Sigma[...]`` or a union such
        as ``Fin[2] + Fin[3]``, with a value for every size it names and the
        layout to store it in: ``Arr.zeros(triangle, n=4, storage="packed")``.
        A kernel's declared domain is ``kernel.arg_types[name].domain``.
        """
        domain = index_domain(shape)
        if domain is not None:
            fixed = domain.fixed(sizes)
            values = np.zeros(fixed.storage_shape(_storage(storage)), dtype=dtype)
            return cls._over(fixed, storage, values)
        if sizes:
            raise TypeError(
                f"sizes {', '.join(sorted(sizes))} were given for {shape!r}, which "
                "is not a domain; a shape of index types has its sizes in it"
            )
        return cls(np.zeros(cls._concrete_shape(shape), dtype=dtype))

    @classmethod
    def from_cells(
        cls,
        domain: Any,
        values: Any,
        dtype: Any = None,
        *,
        storage: str = "box",
        **sizes: int,
    ) -> Arr:
        """An array over a domain holding ``values`` at its points.

        ``values`` are in the order of :meth:`cells`, the domain's points in
        lexicographic order (a union's pieces one after another), whatever the
        layout: the same values give the same array boxed or packed.
        """
        found = index_domain(domain)
        if found is None:
            raise TypeError(
                f"{domain!r} is not a domain: from_cells takes Where[...], "
                "Sigma[...] or a sum of pieces"
            )
        fixed = found.fixed(sizes)
        flat = np.asarray(values) if dtype is None else np.asarray(values, dtype=dtype)
        flat = flat.reshape(-1)
        if flat.size != fixed.count:
            raise ValueError(
                f"{flat.size} values for the {fixed.count} points of {fixed!r}"
            )
        storage = _storage(storage)
        return cls._over(fixed, storage, fixed.scatter(flat, storage))

    @classmethod
    def _over(cls, fixed: Fixed, storage: str, values: np.ndarray) -> Arr:
        """An array over ``fixed`` whose ``storage`` buffer is ``values``."""
        storage = _storage(storage)
        if storage == "packed":
            fixed.table()  # refuses a domain whose rows are not intervals
        array = object.__new__(cls)
        array._values = values
        array._offsets = None
        array._layout = None
        array._domain = fixed
        array._storage = storage
        return array

    @classmethod
    def from_numpy(cls, values: Any, dtype: Any = None) -> Arr:
        """A dense array sharing (or converting) an existing numpy array."""
        array = np.asarray(values) if dtype is None else np.asarray(values, dtype=dtype)
        return cls(array)

    @classmethod
    def ragged(
        cls, cnt: Sequence[int] | np.ndarray, values: Any = None, dtype: Any = None
    ) -> Arr:
        """A ragged array with ``cnt[r]`` entries in row ``r``.

        The offsets are the exclusive prefix sums of ``cnt``, which is the CSR
        convention: ``offsets[0] == 0`` and ``offsets[r+1] == offsets[r] +
        cnt[r]``. ``values`` is the flat buffer; omit it for zeros.
        """
        counts = np.asarray(cnt, dtype=np.int64)
        if counts.ndim != 1:
            raise ValueError("cnt must be a 1-D sequence of counts")
        if np.any(counts < 0):
            raise ValueError("counts must be non-negative")
        offsets = np.zeros(counts.size + 1, dtype=np.int64)
        np.cumsum(counts, out=offsets[1:])
        total = int(offsets[-1])
        if values is None:
            flat = np.zeros(total, dtype=np.float64 if dtype is None else dtype)
        else:
            flat = np.asarray(values) if dtype is None else np.asarray(
                values, dtype=dtype
            )
            flat = flat.reshape(-1)
            if flat.size != total:
                raise ValueError(f"{flat.size} values for {total} ragged slots")
        return cls(flat, offsets)

    @staticmethod
    def _concrete_shape(shape: Any) -> tuple[int, ...]:
        axes = shape if isinstance(shape, tuple | list) else (shape,)
        sizes = []
        for axis in axes:
            size = axis_size(axis)
            if not isinstance(size, int | np.integer):
                raise TypeError(
                    f"cannot allocate an array with symbolic axis {axis!r}; "
                    "sizes come from data at run time"
                )
            sizes.append(int(size))
        return tuple(sizes)

    # -- the written type ------------------------------------------------

    def __class_getitem__(cls, item: Any) -> ArrSpec:
        """``Arr[Fin[n], ..., Real]``: axes first, element sort last."""
        parts = item if isinstance(item, tuple) else (item,)
        if len(parts) < 1:
            raise TypeError("Arr[...] needs at least an element type")
        return ArrSpec(axes=tuple(parts[:-1]), dtype=parts[-1])

    # -- the declared layout ------------------------------------------------

    def through(self, counts: Any = None, offsets: Any = None) -> Arr:
        """This ragged array, read through the layout a kernel declares.

        ``counts`` and ``offsets`` are the arguments a kernel declares beside
        it, ``cnt`` and ``off`` for ``val: Arr[Fin[n], Fin[cnt], Real]`` (see
        :func:`loopty.term.declared_layout`), either one ``None`` when the
        kernel declares no such parameter. The result shares this array's
        buffers and reads those arrays at every use: row ``r`` is
        ``counts[r]`` long, or ``offsets[r + 1] - offsets[r]`` without counts,
        and ``[r, j]`` is the cell ``offsets[r] + j`` of the flat buffer. Where
        one is ``None`` this array's own offsets stand in.

        That is what the lowered kernel does: it is handed those arrays as
        arguments, runs a loop over ``val.dom[r]`` to the count it reads from
        them once per row, and indexes ``val[off[r] + j]`` with ``off`` as the
        kernel has left it. So a native run that reads the same arrays gives a
        kernel that writes its counts or its offsets the meaning the compiled
        run gives it, where the array's own layout gave it another. The
        contract checks that the two layouts agree on entry
        (:func:`loopty.contract.ragged_arguments`); after that the declared one
        is the one both runs follow.

        The number of rows stays this array's own, and every read is still
        checked: a column against the length of its row, and the cell it lands
        on against the flat buffer, since offsets a kernel has rewritten can
        point anywhere.
        """
        if self._offsets is None:
            raise TypeError("a dense array has no layout to read through")
        view = self._shared(type(self))
        view._layout = (counts, offsets)
        return view

    def _shared(self, kind: type[Arr]) -> Arr:
        """An array of class ``kind`` sharing every buffer and field of this one."""
        view = object.__new__(kind)
        for slot in Arr.__slots__:
            setattr(view, slot, getattr(self, slot))
        return view

    def _replaced(self, values: np.ndarray) -> Arr:
        """This array with another buffer of the same shape, as its own layout.

        A ragged view keeps its offsets and drops a declared layout it was
        read through; an array over a domain keeps its domain and its storage.
        """
        if self._domain is not None:
            return Arr._over(self._domain, self._storage or "box", values)
        if self._offsets is not None:
            return Arr(values, self._offsets)
        return Arr(values)

    def copy(self) -> Arr:
        """A copy with buffers of its own, following its own layout."""
        if self._offsets is not None:
            return Arr(self._values.copy(), self._offsets.copy())
        return self._replaced(self._values.copy())

    @property
    def layout(self) -> tuple[Any, Any] | None:
        """The ``(counts, offsets)`` this array is read through, if any.

        ``None`` for an array that follows its own offsets; see
        :meth:`through`.
        """
        return self._layout

    def _row_start(self, row: int) -> int:
        """Where row ``row`` starts in the flat buffer, by the layout followed."""
        declared = None if self._layout is None else self._layout[1]
        if declared is not None:
            return int(_flat(declared)[row])
        assert self._offsets is not None
        return int(self._offsets[row])

    def _row_length(self, row: int) -> int:
        """How many entries row ``row`` has, by the layout followed.

        Never negative: a count a kernel has made negative, or offsets it has
        made decrease, give a loop over the row no iterations, as they give the
        compiled ``for (j = 0; j < len; ++j)``.
        """
        if self._layout is not None:
            counts, offsets = self._layout
            if counts is not None:
                return max(int(_flat(counts)[row]), 0)
            if offsets is not None:
                flat = _flat(offsets)
                return max(int(flat[row + 1]) - int(flat[row]), 0)
        assert self._offsets is not None
        return int(self._offsets[row + 1] - self._offsets[row])

    def _row_slice(self, row: int) -> slice:
        """The flat range of row ``row``, checked against the buffer."""
        start = self._row_start(row)
        stop = start + self._row_length(row)
        if start < 0 or stop > self._values.size:
            raise IndexError(
                f"row {row} occupies cells {start} to {stop} of the flat buffer "
                f"by {self._described_layout()}, and the buffer has "
                f"{self._values.size}"
            )
        return slice(start, stop)

    def _described_layout(self) -> str:
        """How the layout followed reads in a message."""
        if self._layout is None:
            return "its own offsets"
        return "the counts and offsets the kernel declares"

    # -- shape ------------------------------------------------------------

    @property
    def is_ragged(self) -> bool:
        """Is this a dependent-sum (ragged) array?"""
        return self._offsets is not None

    @property
    def domain(self) -> Fixed | None:
        """The polyhedral domain this array is over, at its sizes, or ``None``."""
        return self._domain

    @property
    def storage(self) -> str | None:
        """``"box"`` or ``"packed"`` for an array over a domain, else ``None``."""
        return self._storage

    @property
    def sizes(self) -> dict[str, int]:
        """The sizes of an array's domain, by name: ``{"n": 4}``."""
        return {} if self._domain is None else dict(self._domain.sizes)

    @property
    def ndim(self) -> int:
        """Number of index axes (2 for a ragged array)."""
        if self._domain is not None:
            return self._domain.ndim
        return 2 if self.is_ragged else self._values.ndim

    def cells(self) -> np.ndarray:
        """The values at the domain's points, in lexicographic order.

        The order of :meth:`loopty.domain.Fixed.points`, the same whichever
        layout the array is stored in, so two arrays over one domain are
        compared by comparing these. A dense or ragged array has no domain.
        """
        if self._domain is None:
            raise TypeError("only an array over a domain has cells in its order")
        return self._domain.gather(self._values, self._storage or "box")

    def stored(self, storage: str) -> np.ndarray:
        """This array's values in the buffer of ``storage``: its own, or a copy."""
        if self._domain is None:
            raise TypeError("only an array over a domain has a layout to choose")
        if storage == self._storage:
            return self._values
        return self._domain.scatter(self.cells(), _storage(storage))

    def load(self, storage: str, buffer: Any) -> None:
        """Write into this array what a buffer of ``storage`` holds at the points."""
        if self._domain is None:
            raise TypeError("only an array over a domain has a layout to load")
        buffer = np.asarray(buffer)
        if storage == self._storage and buffer.size == self._values.size:
            self._values[...] = buffer.reshape(self._values.shape)
            return
        shape = self._domain.storage_shape(storage)
        values = self._domain.gather(buffer.reshape(shape), storage)
        self._domain.scatter(values, self._storage or "box", into=self._values)

    def table(self) -> np.ndarray:
        """The table of row starts of the packed layout of this array's domain."""
        if self._domain is None:
            raise TypeError("only an array over a domain has a table of rows")
        return self._domain.table()

    @property
    def offsets(self) -> np.ndarray:
        """Row start offsets of a ragged array."""
        if self._offsets is None:
            raise AttributeError("a dense array has no offsets")
        return self._offsets

    @property
    def counts(self) -> np.ndarray:
        """Per-row counts of a ragged array."""
        return np.diff(self.offsets)

    @property
    def shape(self) -> tuple[int, ...]:
        """Dense shape; for a ragged array, ``(nrows, max_count)``.

        The second entry of a ragged shape is only an envelope. The exact bound
        of row ``r`` is ``counts[r]``, which is what ``dom[r]`` iterates and what
        the type records. An array over a domain has the shape of the buffer
        its layout keeps: the box, or the number of its points.
        """
        if self._domain is not None:
            return tuple(int(s) for s in self._values.shape)
        if self._offsets is None:
            return tuple(int(s) for s in self._values.shape)
        counts = self.counts
        return (int(counts.size), int(counts.max()) if counts.size else 0)

    @property
    def dtype(self) -> np.dtype:
        """Element dtype."""
        return self._values.dtype

    @property
    def type(self) -> ArrType:
        """The array's type, with concrete sizes.

        A runtime array knows its sizes, so the axes are ints and a ragged
        second axis carries the tuple of counts. A traced array instead names
        the counts array as the axis, which is the form ``lower`` needs. An
        array over a domain has the domain as it was written.
        """
        if self._domain is not None:
            return ArrType(
                axes=(), dtype=self.dtype, ragged=(), domain=self._domain.domain
            )
        if self._offsets is None:
            return ArrType(
                axes=self.shape,
                dtype=self.dtype,
                ragged=(False,) * self._values.ndim,
            )
        counts = tuple(int(c) for c in self.counts)
        return ArrType(
            axes=(len(counts), counts), dtype=self.dtype, ragged=(False, True)
        )

    def _extent(self, prefix: tuple[int, ...]) -> int:
        """Extent of the axis after fixing ``prefix``."""
        if self._domain is not None:
            return self._domain.bound(prefix)
        if self._offsets is not None:
            if len(prefix) == 0:
                return int(self._offsets.size - 1)
            if len(prefix) == 1:
                return self._row_length(prefix[0])
            raise IndexError(f"a ragged array has 2 axes, not {len(prefix) + 1}")
        if len(prefix) >= self._values.ndim:
            raise IndexError(
                f"this array has {self._values.ndim} axes, not {len(prefix) + 1}"
            )
        return int(self._values.shape[len(prefix)])

    def _fiber(self, prefix: tuple[int, ...]) -> Any:
        """The indices the axis after ``prefix`` runs over."""
        if self._domain is not None:
            return self._domain.fiber(prefix)
        return range(self._extent(prefix))

    @property
    def dom(self) -> Dom:
        """The iteration domain of the outer axis."""
        return Dom(self, ())

    # -- element access ----------------------------------------------------

    def _flat_index(self, key: tuple[Any, ...]) -> Any:
        row, column = key
        row = int(row)
        offsets = self._offsets
        assert offsets is not None
        if not 0 <= row < offsets.size - 1:
            raise IndexError(f"row {row} out of range for {offsets.size - 1} rows")
        count = self._row_length(row)
        column = int(column)
        if not 0 <= column < count:
            raise IndexError(
                f"column {column} out of range for row {row} of length {count}"
            )
        flat = self._row_start(row) + column
        if not 0 <= flat < self._values.size:
            raise IndexError(
                f"[{row}, {column}] is cell {flat} of the flat buffer by "
                f"{self._described_layout()}, and the buffer has "
                f"{self._values.size}"
            )
        return flat

    def _dense_key(self, key: Any) -> Any:
        """``key``, once it is known to hold no negative integer index.

        numpy reads ``x[-1]`` as the last cell, and a dense array would inherit
        that for free if it handed the key straight on. It must not: ``Fin[n]``
        has no negative points, the typing rules state every access over the
        index type and not over numpy's wraparound, and the generated C reads
        ``x[-1]`` as the cell in front of the buffer. A native run that wrapped
        would therefore agree with nothing the ledger says, so the index is
        refused with an ``IndexError``, the same exception numpy raises at the
        other end, which is also what lets a masked read under a false ``when``
        answer zero here as it does there.
        """
        parts = key if isinstance(key, tuple) else (key,)
        for part in parts:
            if isinstance(part, bool | np.bool_):
                continue
            if isinstance(part, int | np.integer):
                negative = part < 0
            elif isinstance(part, np.ndarray) and part.dtype.kind in "iu":
                negative = bool(np.any(part < 0))
            else:
                continue
            if negative:
                where = f" in {key!r}" if isinstance(key, tuple) else ""
                raise IndexError(
                    f"index {part}{where} is negative: an index of a loopty "
                    "array is a point of its index type, which starts at 0, and "
                    "is never counted from the end"
                )
        return key

    def _domain_address(self, key: Any) -> Any:
        """Where the point ``key`` is stored, once it is known to be a point.

        An index tuple of every axis, each an integer, and a point of the
        domain: a cell of the box outside the domain is refused with an
        ``IndexError``, the exception an index out of range raises, since the
        type says the array has no such cell.
        """
        domain = self._domain
        assert domain is not None
        parts = key if isinstance(key, tuple) else (key,)
        if len(parts) != domain.ndim or not all(
            isinstance(part, int | np.integer) and not isinstance(part, bool | np.bool_)
            for part in parts
        ):
            raise IndexError(
                f"an array over {domain!r} is indexed at a point, one integer for "
                f"each of its {domain.ndim} axes, and not at {key!r}"
            )
        point = tuple(int(part) for part in parts)
        if not domain.contains(point):
            raise IndexError(
                f"{list(point)} is not a point of {domain!r}, the domain of this "
                "array; a cell outside the domain is not one of its cells, "
                "whatever its storage keeps there"
            )
        return domain.address(point, self._storage or "box")

    def __getitem__(self, key: Any) -> Any:
        if self._domain is not None:
            return self._values[self._domain_address(key)]
        if self._offsets is None:
            return self._values[self._dense_key(key)]
        if isinstance(key, tuple):
            if len(key) != 2:
                raise IndexError("a ragged array is indexed [row, column]")
            return self._values[self._flat_index(key)]
        row = int(key)
        offsets = self._offsets
        if not 0 <= row < offsets.size - 1:
            raise IndexError(f"row {row} out of range for {offsets.size - 1} rows")
        return self._values[self._row_slice(row)]

    def __setitem__(self, key: Any, value: Any) -> None:
        if self._domain is not None:
            self._values[self._domain_address(key)] = value
            return
        if self._offsets is None:
            self._values[self._dense_key(key)] = value
            return
        if isinstance(key, tuple):
            if len(key) != 2:
                raise IndexError("a ragged array is indexed [row, column]")
            self._values[self._flat_index(key)] = value
            return
        row = int(key)
        offsets = self._offsets
        if not 0 <= row < offsets.size - 1:
            raise IndexError(f"row {row} out of range for {offsets.size - 1} rows")
        self._values[self._row_slice(row)] = value

    def numpy(self) -> np.ndarray:
        """The backing numpy array: the dense array, or the flat ragged values."""
        return self._values

    def __array__(self, dtype: Any = None, copy: bool | None = None) -> np.ndarray:
        array = self._values
        if dtype is not None:
            array = array.astype(dtype, copy=False)
        return np.array(array, copy=True) if copy else array

    def __len__(self) -> int:
        return self._extent(())

    def __repr__(self) -> str:
        if self._domain is not None:
            return (
                f"Arr({self._domain!r}, storage={self._storage!r}, dtype={self.dtype})"
            )
        if self._offsets is None:
            return f"Arr(shape={self.shape}, dtype={self.dtype})"
        counts = tuple(int(c) for c in self.counts)
        return f"Arr.ragged(counts={counts}, dtype={self.dtype})"


def _storage(storage: str) -> str:
    """``storage``, once it is known to be a layout; see :mod:`loopty.domain`."""
    if storage not in STORAGES:
        raise ValueError(
            f"an array over a domain is stored {' or '.join(map(repr, STORAGES))}, "
            f"not {storage!r}"
        )
    return storage


def _flat(value: Any) -> np.ndarray:
    """The elements of a counts or offsets argument, as one flat array.

    The buffer itself, not a copy and not through the argument's own
    ``__getitem__``: the layout is read as the kernel has left it, and a
    masking view's answer of zero under a false guard is for the body's reads,
    not for the row a read lands in.
    """
    if isinstance(value, Arr):
        value = value.numpy()
    return np.asarray(value).reshape(-1)
