"""Programs whose calls assume what earlier calls established (#13, #65).

Each program here passes an array an earlier call wrote to a later call
whose argument types require something of its cells: a ``Fin`` element
sort, or the offsets a ragged family is read through. Natively the later
call's contract checks it; in the compiled program the requirement is
decided under the earlier call's postcondition and the theorems the program
cites, or checked between the calls.
"""

from __future__ import annotations

import numpy as np
from lanky import theorem
from lanky.prelude import Fn, Int, Nat, Real

from loopty import Arr, Fin, kernel, program, reduce_sum

# {{{ a permutation, computed and then used


@kernel
def number(perm: Arr[Fin[n], Fin[n]]) -> all(perm[i] == n - 1 - i for i in Fin[n]):
    """Reverse the cells: ``perm[i] = n - 1 - i``, and say so."""
    for i in perm.dom:
        perm[i] = perm.dom.size - 1 - i


@kernel
def number_up(perm: Arr[Fin[n], Fin[n]]):
    """Off by one at the end, ``perm[n - 1]`` is ``n``, and nothing said."""
    for i in perm.dom:
        perm[i] = i + 1


@kernel
def gather(
    perm: Arr[Fin[n], Fin[n]], x: Arr[Fin[n], Real], y: Arr[Fin[n], Real]
):
    """``y[i] = x[perm[i]]``, in bounds by the element type of ``perm``."""
    for i in y.dom:
        y[i] = x[perm[i]]


@program
def permuted(perm, x, y):
    """#65's program: gather's requirement is number's postcondition."""
    number(perm)
    gather(perm, x, y)


@program
def permuted_up(perm, x, y):
    """The same with a permutation off by one: gather's requirement is checked."""
    number_up(perm)
    gather(perm, x, y)


# }}}


# {{{ offsets computed by a scan, and rows read through them


@kernel
def scan(
    cnt: Arr[Fin[n], Nat], off: Arr[Fin[n + 1], Nat]
) -> (off[0] == 0) & all(off[r + 1] == off[r] + cnt[r] for r in Fin[n]):
    """Exclusive prefix sum of the counts."""
    off[0] = 0
    for r in cnt.dom:
        off[r + 1] = off[r] + cnt[r]


@theorem
def scan_monotone(
    n: Nat,
    cnt: Fn[Fin[n], Nat],
    off: Fn[Fin[n + 1], Nat],
    h0: off(0) == 0,
    hs: all(off(r + 1) == off(r) + cnt(r) for r in Fin[n]),
) -> all(off(a) <= off(b) for a in Fin[n + 1] for b in Fin[n + 1] if a <= b):
    """The offsets a scan produces are monotone."""


@kernel
def rowsums(
    cnt: Arr[Fin[n], Nat],
    off: Arr[Fin[n + 1], Nat],
    val: Arr[Fin[n], Fin[cnt], Real],
    y: Arr[Fin[n], Real],
):
    """Sum each row of ``val``, read through the offsets it declares."""
    for r in y.dom:
        y[r] = reduce_sum(val[r, j] for j in val.dom[r])


@kernel
def bump(off: Arr[Fin[n], Nat]):
    """Add one to every offset, and say nothing about it."""
    for r in off.dom:
        off[r] = off[r] + 1


@program(uses=[scan_monotone])
def through(cnt, off, val, y):
    """Scan the counts into the offsets, then read the rows through them."""
    scan(cnt, off)
    rowsums(cnt, off, val, y)


@program
def bumped(cnt, off, val, y):
    """A write after the scan retires its postcondition: the layout is checked."""
    scan(cnt, off)
    bump(off)
    rowsums(cnt, off, val, y)


# }}}


# {{{ a requirement only a theorem decides


@kernel
def scan_unit(
    cnt: Arr[Fin[n], Fin[2]], off: Arr[Fin[n + 1], Fin[n + 1]]
) -> (off[0] == 0) & all(off[r + 1] == off[r] + cnt[r] for r in Fin[n]) & (
    off[n] <= n
):
    """Offsets of rows of at most one entry, which end at most at ``n``."""
    off[0] = 0
    for r in cnt.dom:
        off[r + 1] = off[r] + cnt[r]


@theorem
def scan_monotone_int(
    n: Nat,
    cnt: Fn[Fin[n], Nat],
    off: Fn[Fin[n + 1], Int],
    h0: off(0) == 0,
    hs: all(off(r + 1) == off(r) + cnt(r) for r in Fin[n]),
) -> all(off(a) <= off(b) for a in Fin[n + 1] for b in Fin[n + 1] if a <= b):
    """The same theorem over integer offsets, which a written array may hold."""


@kernel
def pick(
    off: Arr[Fin[n + 1], Fin[n + 1]],
    x: Arr[Fin[n + 1], Real],
    y: Arr[Fin[n + 1], Real],
):
    """``y[i] = x[off[i]]``, in bounds by the element type of ``off``."""
    for i in y.dom:
        y[i] = x[off[i]]


@program(uses=[scan_monotone_int])
def picked(cnt, off, x, y):
    """``0 = off[0] <= off[i] <= off[n] <= n`` needs the offsets monotone."""
    scan_unit(cnt, off)
    pick(off, x, y)


@program
def picked_alone(cnt, off, x, y):
    """Without the theorem the requirement is checked between the calls."""
    scan_unit(cnt, off)
    pick(off, x, y)


# }}}


# {{{ flat CSR after the scan


@kernel
def scan_csr(
    cnt: Arr[Fin[n], Nat], off: Arr[Fin[n + 1], Fin[nnz + 1]]
) -> (off[0] == 0) & all(off[r + 1] == off[r] + cnt[r] for r in Fin[n]):
    """The scan into offsets of a buffer of ``nnz`` cells."""
    off[0] = 0
    for r in cnt.dom:
        off[r + 1] = off[r] + cnt[r]


@kernel
def weigh_flat(
    cnt: Arr[Fin[n], Nat],
    off: Arr[Fin[n + 1], Fin[nnz + 1]],
    wt: Arr[Fin[n], Fin[cnt], Real],
    val: Arr[Fin[nnz], Real],
    y: Arr[Fin[n], Real],
):
    """Weigh the entries of each row, read from a flat buffer through ``off``.

    ``val[off[r] + j]`` is in bounds only for offsets that lay the rows out
    inside ``val``, which nothing in this kernel says: alone, the fact is
    assumed.
    """
    for r in y.dom:
        y[r] = reduce_sum(wt[r, j] * val[off[r] + j] for j in wt.dom[r])


@program
def flat(cnt, off, wt, val, y):
    """The flat access is in bounds under the scan's postcondition."""
    scan_csr(cnt, off)
    weigh_flat(cnt, off, wt, val, y)


# }}}


def csr(counts: list[int], seed: int = 0) -> dict:
    """A matrix with these row counts, as the arrays the programs take."""
    rng = np.random.default_rng(seed)
    total = int(sum(counts))
    return {
        "cnt": Arr.from_numpy(np.array(counts, dtype=np.int64)),
        "off": Arr.zeros(len(counts) + 1, dtype=np.int64),
        "val": Arr.ragged(counts, values=rng.normal(size=total)),
        "y": Arr.zeros(len(counts)),
    }


def example_inputs() -> dict:
    """What a check of this file runs the programs on, besides drawn inputs."""
    counts = [2, 0, 3, 1]
    flat_inputs = csr(counts)
    flat_inputs["wt"] = flat_inputs.pop("val")
    flat_inputs["val"] = Arr.from_numpy(np.arange(6.0))
    flat_inputs["off"] = Arr.zeros(5, dtype=np.int64)
    return {
        "permuted": {
            "perm": Arr.zeros(4, dtype=np.int64),
            "x": Arr.from_numpy(np.arange(4.0)),
            "y": Arr.zeros(4),
        },
        "through": csr(counts),
        "picked": {
            "cnt": Arr.from_numpy(np.array([1, 0, 1], dtype=np.int64)),
            "off": Arr.zeros(4, dtype=np.int64),
            "x": Arr.from_numpy(np.arange(4.0)),
            "y": Arr.zeros(4),
        },
        "flat": flat_inputs,
    }
