"""Constant tables: values a kernel reads that are known when it is compiled.

Quadrature weights, a basis tabulated at the quadrature points, a
differentiation matrix: the numbers are fixed before the kernel runs, and a
code generator writes them into the kernel rather than passing them in.
TSFC's element kernels are written so, with each table a read-only loopy
temporary with an initializer. In loopty a table is declared once, next to
the kernel, as the numpy array it is::

    #: The 3-point rule on the reference triangle, and the P1 basis at it.
    W = table([1 / 6, 1 / 6, 1 / 6])
    PHI = table([[2 / 3, 1 / 6, 1 / 6], [1 / 6, 2 / 3, 1 / 6], [1 / 6, 1 / 6, 2 / 3]])

    @kernel
    def mass(u: Arr[Fin[nc], Fin[3], Real], y: Arr[Fin[nc], Fin[3], Real]):
        for c in y.dom:
            for j in y.dom[c]:
                y[c, j] = reduce_sum(
                    W[q] * PHI[q, j] * PHI[q, k] * u[c, k]
                    for q in W.dom
                    for k in PHI.dom[q]
                )

and the body reads it as it reads an array. Natively a :class:`Table` is a
read-only numpy array, ``W[q]`` a numpy scalar of its dtype. When the body is
traced, the table is recorded in the term under the name the body reads it
by (``W``), each read is an access whose in-bounds fact isl decides over the
table's shape, and the lowering declares it as a read-only loopy temporary
with the values as its initializer: a ``static`` array on the C target, a
``__constant`` one on OpenCL, written with every digit, so that both runs
read the same numbers. The interpreter reads the same values.

A table is copied when it is declared and cannot be written afterwards, by
the kernel or by anyone: its values are part of the kernel's code, and a
change after the kernel was compiled would make the native run read numbers
the compiled one does not. For the same reason it is not an argument: a
kernel reads it by name, and an array of its values is what is passed. Its
values are finite, since the compiled code writes each one as a literal.

Natively a table is read as an array is, one value at a time: an index is a
point of its axis, never counted from the end, and a read out of range inside
a ``when`` block whose guard is false answers zero, as for an array.
"""

from __future__ import annotations

from typing import Any

import numpy as np

__all__ = ["Table", "table", "table_type"]


class Table:
    """A constant table a kernel reads; see :func:`table` and the module."""

    __slots__ = ("_array", "_values")

    def __init__(self, values: Any, dtype: Any = None) -> None:
        from loopty.arr import Arr

        array = np.array(values, dtype=dtype, copy=True)
        if array.ndim == 0:
            raise ValueError(
                f"a table has at least one axis, and {values!r} has none; a "
                "single number is written in the kernel as it is"
            )
        if array.size == 0:
            raise ValueError(f"a table has at least one value, and {values!r} has none")
        if array.dtype.kind not in "biufc":
            raise TypeError(
                f"a table holds numbers, and {values!r} holds {array.dtype}"
            )
        if array.dtype.kind in "fc" and array.dtype not in _COMPILED_FLOATS:
            raise TypeError(
                f"a table is held in a type the compiled kernel has, and "
                f"{array.dtype} is none: C and OpenCL C have float32 and "
                "float64, and complex64 and complex128. Pass dtype=np.float32 "
                "or dtype=np.float64"
            )
        _refuse_non_finite(array)
        self._values = _frozen(array)
        self._array = Arr(self._values)

    # {{{ what it holds

    @property
    def values(self) -> np.ndarray:
        """The values, a numpy array that cannot be made writeable."""
        return self._values

    def numpy(self) -> np.ndarray:
        """The values, read-only, as :meth:`loopty.Arr.numpy` gives an array's."""
        return self._values

    @property
    def dtype(self) -> np.dtype:
        """The dtype of the values, which the kernel reads them in."""
        return self._values.dtype

    @property
    def shape(self) -> tuple[int, ...]:
        """The extent of each axis."""
        return self._values.shape

    @property
    def ndim(self) -> int:
        """The number of axes."""
        return self._values.ndim

    def __len__(self) -> int:
        return len(self._values)

    def __repr__(self) -> str:
        return f"table({self._values.tolist()!r}, dtype=np.{self.dtype.name})"

    # }}}

    # {{{ read by a kernel

    @property
    def dom(self) -> Any:
        """The points of the table's first axis, as an array's ``.dom`` is.

        ``W.dom`` runs over ``Fin[3]`` for three weights, and ``PHI.dom[q]``
        over the second axis at ``q``. When the body is traced it is the
        recorded table's, so that a loop or a sum over it binds a variable.
        """
        proxy = _traced(self)
        return self._array.dom if proxy is None else proxy.dom

    def __getitem__(self, key: Any) -> Any:
        """The value at ``key``, natively; an access of the table, when traced.

        Natively a table is read as an array is: an index is a point of the
        axis, so a negative one is refused with an ``IndexError`` rather than
        counted from the end, as the compiled kernel would read in front of
        the table; and inside a ``when`` block whose guard is false, a read
        out of range answers zero, since nothing the block computes is
        stored (:func:`loopty.trace.mask_writes`).
        """
        proxy = _traced(self)
        if proxy is not None:
            return proxy[key]
        parts = key if isinstance(key, tuple) else (key,)
        if 0 < len(parts) < self.ndim and all(
            isinstance(part, int | np.integer) and not isinstance(part, bool | np.bool_)
            for part in parts
        ):
            # A row, PHI[q], which a kernel the body calls takes as an array
            # (loopty.calls): a view of the values, which nothing can write.
            # Out of range under a false guard it is a row of zeros, as a
            # cell is zero there; the call it is given to is not made.
            from loopty.arr import Arr
            from loopty.trace import _writes_are_masked

            try:
                if any(part < 0 for part in parts):
                    raise IndexError(
                        f"index {key!r} of a table is negative: an index is a "
                        "point of the table's axis, and is never counted from "
                        "the end"
                    )
                return Arr(self._values[key])
            except IndexError:
                if not _writes_are_masked():
                    raise
                zeros = np.zeros(self._values.shape[len(parts) :], self._values.dtype)
                zeros.flags.writeable = False
                return Arr(zeros)
        if not (
            isinstance(key, tuple)
            and len(key) == self.ndim
            or (not isinstance(key, tuple) and self.ndim == 1)
        ):
            raise IndexError(
                f"a table of {self.ndim} axes is read one value at a time, with "
                f"{self.ndim} indices, such as W[q] or PHI[q, j], or a row at a "
                "time, PHI[q], which a kernel it calls takes as an array: a "
                "kernel computes cell by cell"
            )
        from loopty.trace import _writes_are_masked

        try:
            return self._array[key]
        except IndexError:
            if _writes_are_masked():
                return 0
            raise

    def __setitem__(self, key: Any, value: Any) -> None:
        raise TypeError(
            f"{self!r} is a constant table: its values are part of the "
            "kernel's code, and nothing writes them. Pass an array the kernel "
            "may write instead"
        )

    # }}}


def table(values: Any, dtype: Any = None) -> Table:
    """A constant table of ``values``, a kernel reads; see :mod:`loopty.table`.

    ``values`` is anything ``numpy.array`` takes, of finite numbers, with at
    least one axis; ``dtype`` is the dtype to hold them in, ``float64`` for
    floats by default, as numpy chooses. The values are copied, and cannot be
    changed afterwards.
    """
    return Table(values, dtype)


#: The floating and complex dtypes a table may be held in: those C and OpenCL
#: C have. ``float16`` and ``longdouble`` failed when the kernel was lowered.
_COMPILED_FLOATS = frozenset(
    np.dtype(t) for t in (np.float32, np.float64, np.complex64, np.complex128)
)


def _frozen(array: np.ndarray) -> np.ndarray:
    """``array``'s values in an array no one can make writeable again.

    An array that owns its buffer can be made writeable by anyone who holds
    it (``W.values.flags.writeable = True``), and so can the base of a view.
    This one's buffer is a ``bytes`` object, which is immutable, so numpy
    refuses the flag ("cannot set WRITEABLE flag to True of this array") on
    it and on every view of it: a change after a kernel was compiled would
    make the native run read numbers the compiled one does not.
    """
    contiguous = np.ascontiguousarray(array)
    return np.frombuffer(contiguous.tobytes(), dtype=contiguous.dtype).reshape(
        contiguous.shape
    )


def _refuse_non_finite(array: np.ndarray) -> None:
    """Refuse a table holding a ``nan`` or an infinity, naming the first.

    The lowering writes each value into the kernel's code, and C has no
    literal for either: loopy writes the macros ``NAN`` and ``INFINITY``,
    which ``math.h`` defines and loopy does not include, and a complex
    number as ``re + im * I``, which is ``nan`` in the real part when ``im``
    is infinite. A ``nan`` also defeats loopy's caches, which compare
    kernels by their values. Quadrature weights, basis tables and
    differentiation matrices are finite; a value that is not is passed in an
    array.
    """
    if array.dtype.kind not in "fc":
        return
    finite = np.isfinite(array)
    if finite.all():
        return
    index = tuple(int(i) for i in np.argwhere(~finite)[0])
    at = index[0] if len(index) == 1 else list(index)
    raise ValueError(
        f"a table holds finite numbers, and the value at {at} is "
        f"{array[index].item()!r}: the compiled kernel holds each value of a "
        "table in its code, where a nan or an infinity is not written as numpy "
        "holds it. Pass an array holding such values to the kernel instead"
    )


def passed_message(param: str, value: Table) -> str:
    """Why a table is refused as an argument, with the fix: pass its values."""
    return (
        f"the argument {param} is a constant table of shape {value.shape}: a "
        "table is part of the kernel's code, read in the body by the name it "
        "is declared by (W[q]), and is not passed to a kernel. To pass the same "
        "values, pass an array of them, Arr.from_numpy(W.values.copy()) for a "
        "table W"
    )


def table_type(values: Table) -> Any:
    """The :class:`loopty.term.ArrType` a table is read by: its shape, its dtype."""
    from loopty.term import ArrType

    return ArrType(
        axes=tuple(int(extent) for extent in values.shape),
        dtype=values.dtype.type,
        ragged=(False,) * values.ndim,
    )


def _traced(table: Table) -> Any:
    """The tracer's proxy for ``table`` while a body is traced, else ``None``.

    The table is recorded under the name the frame that reads it gives it,
    a global, a closure variable or a local of the body or of a helper it
    calls (:meth:`loopty.trace.Tracer.table`).
    """
    import sys

    from loopty.trace import current_tracer

    tracer = current_tracer()
    if tracer is None:
        return None
    return tracer.table(table, sys._getframe(2))
