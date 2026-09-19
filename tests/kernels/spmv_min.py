"""A ragged kernel: sparse matrix-vector product over a dependent sum.

This is the design's smallest interesting file. Two things in it are the
reason loopty exists.

*The ragged shape is a type.* ``col: Arr[Fin[n], Fin[cnt], Fin[m]]`` says: for
each of the ``n`` rows, ``cnt[r]`` entries, each of them a point of ``Fin[m]``.
The second axis names the counts array, which is what makes the axis dependent
rather than rectangular; the inner loop iterates ``val.dom[r]``, the fiber over
the row, and its bound is the count of *that* row.

*An indirection is in bounds by type.* ``x[col[r, j]]`` needs no proof at all:
the entries of ``col`` are points of ``Fin[m]`` and ``x`` has ``m`` cells, so
the typing rule discharges it with ``decided_by="type"`` and never calls the
oracle. The remaining obligations are affine and isl decides them, including
the ragged ones, whose bound it sees as a parameter standing for the row's
count.

``lanky check`` on this file prints one line per obligation and who settled it.
"""

from __future__ import annotations

import numpy as np
from lanky.prelude import Nat, Real

from loopty import Arr, Fin, kernel, program
from loopty import sum as reduce_sum


@kernel
def scan(
    cnt: Arr[Fin[n], Nat], off: Arr[Fin[n + 1], Nat]
) -> (off[0] == 0) & all(off[r + 1] == off[r] + cnt[r] for r in Fin[n]):
    """Exclusive prefix sum of the counts: the offsets of the flat storage.

    The return annotation is not a result type: outputs are parameters, so it is
    the postcondition, a claim about ``off`` that the ledger records and that a
    later oracle (or a Lean proof of the recurrence) has to establish.
    """
    off[0] = 0
    for r in cnt.dom:
        off[r + 1] = off[r] + cnt[r]


@kernel
def spmv(
    cnt: Arr[Fin[n], Nat],
    col: Arr[Fin[n], Fin[cnt], Fin[m]],
    val: Arr[Fin[n], Fin[cnt], Real],
    x: Arr[Fin[m], Real],
    y: Arr[Fin[n], Real],
):
    """One row of ``y`` per row of the matrix, summed over that row's entries."""
    for r in y.dom:
        y[r] = reduce_sum(val[r, j] * x[col[r, j]] for j in val.dom[r])


@program
def solve(cnt, col, val, x, y, off):
    """Lay out the rows, then multiply: the composition the ledger records."""
    scan(cnt, off)
    spmv(cnt, col, val, x, y)


def example() -> tuple:
    """A 3 by 4 matrix with two, zero and three entries, and a vector."""
    counts = [2, 0, 3]
    cnt = Arr.from_numpy(np.array(counts, dtype=np.int64))
    col = Arr.ragged(counts, values=[0, 2, 1, 2, 3], dtype=np.int64)
    val = Arr.ragged(counts, values=[1.0, 2.0, 3.0, 4.0, 5.0])
    x = Arr.from_numpy(np.array([1.0, 10.0, 100.0, 1000.0]))
    y = Arr.zeros(3)
    off = Arr.zeros(4, dtype=np.int64)
    return cnt, col, val, x, y, off


def main() -> None:
    """Run the program on the example and print what it computed."""
    cnt, col, val, x, y, off = example()
    solve(cnt, col, val, x, y, off)
    print("off =", off.numpy())
    print("y   =", y.numpy())


if __name__ == "__main__":
    main()
