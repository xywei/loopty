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
"""

from __future__ import annotations

from collections.abc import Iterator, Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np

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
    for a ragged array has a different extent for each ``i``. Iterating yields
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
        """The extent of this axis, given the fixed prefix."""
        return self.array._extent(self.prefix)

    def __len__(self) -> int:
        return self.size

    def __iter__(self) -> Iterator[int]:
        return iter(range(self.size))

    def __getitem__(self, index: int) -> Dom:
        """The fiber over ``index``: the domain of the next axis."""
        size = self.size
        if not isinstance(index, int | np.integer):
            raise TypeError(f"domain index must be an integer, got {index!r}")
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
    ``r`` is ``values[offsets[r]:offsets[r+1]]``.
    """

    __slots__ = ("_offsets", "_values")

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

    # -- constructors ----------------------------------------------------

    @classmethod
    def zeros(cls, shape: Any, dtype: Any = np.float64) -> Arr:
        """A dense array of zeros over ``shape``.

        ``shape`` is an index type or a tuple of them: ``Arr.zeros(Fin[4])`` and
        ``Arr.zeros((Fin[3], 4))`` both work, as does a plain int.
        """
        sizes = cls._concrete_shape(shape)
        return cls(np.zeros(sizes, dtype=dtype))

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

    # -- shape ------------------------------------------------------------

    @property
    def is_ragged(self) -> bool:
        """Is this a dependent-sum (ragged) array?"""
        return self._offsets is not None

    @property
    def ndim(self) -> int:
        """Number of index axes (2 for a ragged array)."""
        return 2 if self.is_ragged else self._values.ndim

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
        the type records.
        """
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
        the counts array as the axis, which is the form ``lower`` needs.
        """
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
        if self._offsets is not None:
            if len(prefix) == 0:
                return int(self._offsets.size - 1)
            if len(prefix) == 1:
                row = prefix[0]
                return int(self._offsets[row + 1] - self._offsets[row])
            raise IndexError(f"a ragged array has 2 axes, not {len(prefix) + 1}")
        if len(prefix) >= self._values.ndim:
            raise IndexError(
                f"this array has {self._values.ndim} axes, not {len(prefix) + 1}"
            )
        return int(self._values.shape[len(prefix)])

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
        count = int(offsets[row + 1] - offsets[row])
        column = int(column)
        if not 0 <= column < count:
            raise IndexError(
                f"column {column} out of range for row {row} of length {count}"
            )
        return int(offsets[row]) + column

    def __getitem__(self, key: Any) -> Any:
        if self._offsets is None:
            return self._values[key]
        if isinstance(key, tuple):
            if len(key) != 2:
                raise IndexError("a ragged array is indexed [row, column]")
            return self._values[self._flat_index(key)]
        row = int(key)
        offsets = self._offsets
        if not 0 <= row < offsets.size - 1:
            raise IndexError(f"row {row} out of range for {offsets.size - 1} rows")
        return self._values[offsets[row] : offsets[row + 1]]

    def __setitem__(self, key: Any, value: Any) -> None:
        if self._offsets is None:
            self._values[key] = value
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
        self._values[offsets[row] : offsets[row + 1]] = value

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
        if self._offsets is None:
            return f"Arr(shape={self.shape}, dtype={self.dtype})"
        counts = tuple(int(c) for c in self.counts)
        return f"Arr.ragged(counts={counts}, dtype={self.dtype})"
