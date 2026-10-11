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
the compiled one does not.
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
        array.flags.writeable = False
        self._values = array
        self._array = Arr(array)

    # {{{ what it holds

    @property
    def values(self) -> np.ndarray:
        """The values, a read-only numpy array."""
        return self._values

    def numpy(self) -> np.ndarray:
        """The values, a read-only numpy array, as :meth:`loopty.Arr.numpy`."""
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
        """The value at ``key``, natively; an access of the table, when traced."""
        proxy = _traced(self)
        if proxy is not None:
            return proxy[key]
        if isinstance(key, tuple) and len(key) == self.ndim or (
            not isinstance(key, tuple) and self.ndim == 1
        ):
            return self._values[key]
        raise IndexError(
            f"a table of {self.ndim} axes is read one value at a time, with "
            f"{self.ndim} indices, such as W[q] or PHI[q, j]: a kernel computes "
            "cell by cell"
        )

    def __setitem__(self, key: Any, value: Any) -> None:
        raise TypeError(
            f"{self!r} is a constant table: its values are part of the "
            "kernel's code, and nothing writes them. Pass an array the kernel "
            "may write instead"
        )

    # }}}


def table(values: Any, dtype: Any = None) -> Table:
    """A constant table of ``values``, a kernel reads; see :mod:`loopty.table`.

    ``values`` is anything ``numpy.array`` takes, of numbers, with at least one
    axis; ``dtype`` is the dtype to hold them in, ``float64`` for floats by
    default, as numpy chooses. The values are copied, and cannot be changed
    afterwards.
    """
    return Table(values, dtype)


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
