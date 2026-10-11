"""Kernels whose postconditions are decided from what their statements store, or not.

``tests/test_dataflow.py`` checks each: what is decided, under what, and why
the rest is not.
"""

from __future__ import annotations

import numpy as np
from lanky import theorem
from lanky.prelude import Fn, Int, Nat, Real

from loopty import Arr, Fin, Schedule, Where, kernel, when

# {{{ decided


@kernel
def scan(cnt: Arr[Fin[n], Nat], off: Arr[Fin[n + 1], Int]) -> (off[0] == 0) & all(
    off[r + 1] == off[r] + cnt[r] for r in Fin[n]
):
    """The exclusive scan into offsets of ``Int``: each cell stored once, read after."""
    off[0] = 0
    for r in cnt.dom:
        off[r + 1] = off[r] + cnt[r]


@kernel
def triangle(
    c: Arr[Where[a : Fin[2], b : Fin[p + 1], a + b <= p], Real],
    R: Arr[Where[a : Fin[p + 1], b : Fin[p + 1], a + b <= p], Real],
) -> all(
    R[a, b] == c[a, b] for a in Fin[p + 1] for b in Fin[p + 1] if a < 2 if a + b <= p
) & all(
    R[a + 2, b] == -R[a, b + 2]
    for a in Fin[p + 1]
    for b in Fin[p + 1]
    if a + b + 2 <= p
):
    """The compressed Taylor reconstruction: reals, each equation an atom to isl."""
    for a in R.dom:
        for b in R.dom[a]:
            with when(a < 2):
                R[a, b] = c[a, b]
            with when(a >= 2):
                R[a, b] = -R[a - 2, b + 2]


@kernel
def one_cell(lim: Arr[Fin[1], Nat]) -> lim[0] == 1:
    """A statement without loops runs once, so it stores its one cell once."""
    lim[0] = 1


# }}}


# {{{ not decided


@kernel
def twice(y: Arr[Fin[n], Int]) -> all(y[i] == 1 for i in Fin[n]):
    """Every cell stored twice: the first store is not what the kernel leaves."""
    for i in y.dom:
        y[i] = 0
    for i in y.dom:
        y[i] = 1


@kernel
def from_the_right(y: Arr[Fin[n], Int]) -> all(y[i] == 0 for i in Fin[n] if i + 1 < n):
    """Reads ``y[i + 1]`` before the instance that stores it: true, and not decided."""
    for i in y.dom:
        with when(i + 1 < y.dom.size):
            y[i] = y[i + 1] - y[i + 1]


@kernel
def in_place(x: Arr[Fin[n], Int]) -> all(x[i] != 1000003 for i in Fin[n]):
    """Reads the cell it stores, in the instance that stores it, spelled apart.

    ``x[i + 1 - 1]`` is ``x[i]`` as it was before the store, so the store says
    nothing of what the kernel leaves. Read as an equation, ``x == 2x -
    1000005`` would make every cell 1000005 and decide the claim, which is
    false where ``x`` holds 1000004 on entry.
    """
    for i in x.dom:
        x[i] = 2 * x[i + 1 - 1] - 1000005


@kernel
def halves(y: Arr[Fin[n], Int]) -> all(y[i] != 1000003 for i in Fin[n]):
    """Reads an earlier cell, except at ``i = 0``, where it reads its own first."""
    for i in y.dom:
        y[i] = 2 * y[i // 2] - 1000005


@kernel
def first_three(y: Arr[Fin[n], Int]) -> all(y[i] == 1 for i in Fin[n]):
    """Stores only where ``i < 3``: a cell past them holds what it was given."""
    for i in y.dom:
        with when(i < 3):
            y[i] = 1


@kernel
def scan_backwards(cnt: Arr[Fin[n], Nat], off: Arr[Fin[n + 1], Int]) -> (
    off[0] == 0
) & all(off[r + 1] == off[r] + cnt[r] for r in Fin[n]):
    """The scan's statements the other way round: the loop reads ``off[0]`` first."""
    for r in cnt.dom:
        off[r + 1] = off[r] + cnt[r]
    off[0] = 0


@kernel
def lost_sign(
    c: Arr[Where[a : Fin[2], b : Fin[p + 1], a + b <= p], Real],
    R: Arr[Where[a : Fin[p + 1], b : Fin[p + 1], a + b <= p], Real],
) -> all(
    R[a + 2, b] == R[a, b + 2] for a in Fin[p + 1] for b in Fin[p + 1] if a + b + 2 <= p
):
    """Claims the recurrence without its sign: another atom than the one it stores."""
    for a in R.dom:
        for b in R.dom[a]:
            with when(a < 2):
                R[a, b] = c[a, b]
            with when(a >= 2):
                R[a, b] = -R[a - 2, b + 2]


# }}}


# {{{ cited theorems


@theorem
def offsets_increase(
    n: Nat,
    cnt: Fn[Fin[n], Nat],
    off: Fn[Fin[n + 1], Int],
    h0: off(0) == 0,
    hs: all(off(r + 1) == off(r) + cnt(r) for r in Fin[n]),
) -> all(off(a) <= off(b) for a in Fin[n + 1] for b in Fin[n + 1] if a <= b):
    """The offsets a scan computes from naturals are monotone, over ``Int``."""


@kernel(uses=[offsets_increase])
def scan_monotone(
    cnt: Arr[Fin[n], Nat], off: Arr[Fin[n + 1], Int]
) -> (
    (off[0] == 0)
    & all(off[r + 1] == off[r] + cnt[r] for r in Fin[n])
    & all(off[a] <= off[b] for a in Fin[n + 1] for b in Fin[n + 1] if a <= b)
):
    """Claims the monotonicity too, which the theorem gives from the recurrence."""
    off[0] = 0
    for r in cnt.dom:
        off[r + 1] = off[r] + cnt[r]


@kernel(uses=[offsets_increase])
def scan_cited(cnt: Arr[Fin[n], Nat], off: Arr[Fin[n + 1], Int]) -> (off[0] == 0) & all(
    off[r + 1] == off[r] + cnt[r] for r in Fin[n]
):
    """Claims the recurrence alone: the theorem gives the monotonicity as a fact."""
    off[0] = 0
    for r in cnt.dom:
        off[r + 1] = off[r] + cnt[r]


@theorem
def on_the_triangle(
    p: Nat,
    c: Fn[Nat, Fn[Nat, Real]],
    D: Fn[Nat, Fn[Nat, Real]],
    R: Fn[Nat, Fn[Nat, Real]],
    pde: all(D(a + 2)(b) + D(a)(b + 2) == 0 for a in Nat for b in Nat),
    stored: all(
        c(a)(b) == D(a)(b)
        for a in Fin[p + 1]
        for b in Fin[p + 1]
        if a < 2
        if a + b <= p
    ),
    copies: all(
        R(a)(b) == c(a)(b)
        for a in Fin[p + 1]
        for b in Fin[p + 1]
        if a < 2
        if a + b <= p
    ),
    recurrence: all(
        R(a + 2)(b) == -R(a)(b + 2)
        for a in Fin[p + 1]
        for b in Fin[p + 1]
        if a + b + 2 <= p
    ),
) -> all(R(a)(b) == D(a)(b) for a in Fin[p + 1] for b in Fin[p + 1] if a + b <= p):
    """What ``c`` stores of a harmonic ``D``, reconstructed, is ``D``."""


@kernel(uses=[on_the_triangle])
def reconstruct(
    c: Arr[Where[a : Fin[2], b : Fin[p + 1], a + b <= p], Real],
    R: Arr[Where[a : Fin[p + 1], b : Fin[p + 1], a + b <= p], Real],
) -> all(
    R[a, b] == c[a, b] for a in Fin[p + 1] for b in Fin[p + 1] if a < 2 if a + b <= p
) & all(
    R[a + 2, b] == -R[a, b + 2]
    for a in Fin[p + 1]
    for b in Fin[p + 1]
    if a + b + 2 <= p
):
    """``triangle``, citing the theorem whose two hypotheses its postcondition is."""
    for a in R.dom:
        for b in R.dom[a]:
            with when(a < 2):
                R[a, b] = c[a, b]
            with when(a >= 2):
                R[a, b] = -R[a - 2, b + 2]


@kernel(uses=[on_the_triangle])
def off_by_one(
    c: Arr[Where[a : Fin[2], b : Fin[p + 1], a + b <= p], Real],
    R: Arr[Where[a : Fin[p + 1], b : Fin[p + 1], a + b <= p], Real],
) -> all(
    R[a, b] == c[a, b] for a in Fin[p + 1] for b in Fin[p + 1] if a < 2 if a + b <= p
) & all(
    R[a + 2, b] == -R[a, b + 1]
    for a in Fin[p + 1]
    for b in Fin[p + 1]
    if a + b + 2 <= p
):
    """A recurrence off by one in ``b``, as the postcondition says, citing the theorem.

    The postcondition is what it stores, and is decided; the theorem's
    recurrence is not it, and the theorem gives nothing.
    """
    for a in R.dom:
        for b in R.dom[a]:
            with when(a < 2):
                R[a, b] = c[a, b]
            with when(a >= 2):
                R[a, b] = -R[a - 2, b + 1]


@kernel(uses=[on_the_triangle])
def wrong_coefficient(
    c: Arr[Where[a : Fin[2], b : Fin[p + 1], a + b <= p], Real],
    R: Arr[Where[a : Fin[p + 1], b : Fin[p + 1], a + b <= p], Real],
) -> all(
    R[a, b] == c[a, b] for a in Fin[p + 1] for b in Fin[p + 1] if a < 2 if a + b <= p
) & all(
    R[a + 2, b] == -R[a, b + 2]
    for a in Fin[p + 1]
    for b in Fin[p + 1]
    if a + b + 2 <= p
):
    """Claims the theorem's recurrence and computes another, off by one in ``b``."""
    for a in R.dom:
        for b in R.dom[a]:
            with when(a < 2):
                R[a, b] = c[a, b]
            with when(a >= 2):
                R[a, b] = -R[a - 2, b + 1]


@theorem
def tenfold(
    n: Nat,
    c: Fn[Nat, Real],
    R: Fn[Nat, Real],
    h: all(R(i) == c(i) * 0.1 for i in Fin[n]),
) -> all(R(i) * 10 == c(i) for i in Fin[n]):
    """As reals, ten tenths of ``c`` are ``c``."""


@kernel(uses=[tenfold])
def scaled(c: Arr[Fin[n], Real], R: Arr[Fin[n], Real]) -> all(
    R[i] == c[i] * 0.1 for i in Fin[n]
):
    """Stores a rounded tenth: as reals, ``R[i]`` is not ``c[i] * 0.1``.

    ``R[i] * 10 == c[i]`` is false of the floats at ``c[i] = 3.0``, and the
    theorem gives nothing.
    """
    for i in R.dom:
        R[i] = c[i] * 0.1


@theorem
def difference(
    n: Nat,
    a: Fn[Nat, Real],
    b: Fn[Nat, Real],
    s: Fn[Nat, Real],
    h: all(s(i) == a(i) + b(i) for i in Fin[n]),
) -> all(s(i) - a(i) == b(i) for i in Fin[n]):
    """As reals, ``s - a`` is ``b`` where ``s`` is ``a + b``."""


@kernel(uses=[difference])
def summed(a: Arr[Fin[n], Real], b: Arr[Fin[n], Real], s: Arr[Fin[n], Real]) -> all(
    s[i] == a[i] + b[i] for i in Fin[n]
):
    """Stores a rounded sum: ``s[i] - a[i] == b[i]`` is false of the floats."""
    for i in s.dom:
        s[i] = a[i] + b[i]


@theorem
def unrelated(n: Nat, f: Fn[Fin[n], Int], h: all(f(i) == 7 for i in Fin[n])) -> all(
    f(i) >= 7 for i in Fin[n]
):
    """A theorem whose hypothesis is no conjunct of ``scan_unrelated``'s claim."""


@kernel(uses=[unrelated])
def scan_unrelated(cnt: Arr[Fin[n], Nat], off: Arr[Fin[n + 1], Int]) -> (
    off[0] == 0
) & all(off[r + 1] == off[r] + cnt[r] for r in Fin[n]):
    """Cites a theorem that does not apply: it gives nothing."""
    off[0] = 0
    for r in cnt.dom:
        off[r + 1] = off[r] + cnt[r]


@theorem
def naturals_increase(
    n: Nat,
    cnt: Fn[Fin[n], Nat],
    off: Fn[Fin[n + 1], Nat],
    h0: off(0) == 0,
    hs: all(off(r + 1) == off(r) + cnt(r) for r in Fin[n]),
) -> all(off(a) <= off(b) for a in Fin[n + 1] for b in Fin[n + 1] if a <= b):
    """Over ``Nat``: the offsets the kernel writes are not known to be naturals."""


@kernel(uses=[naturals_increase])
def scan_naturals(cnt: Arr[Fin[n], Nat], off: Arr[Fin[n + 1], Int]) -> (
    off[0] == 0
) & all(off[r + 1] == off[r] + cnt[r] for r in Fin[n]):
    """Cites the theorem over ``Nat``, which the cells it writes cannot stand for."""
    off[0] = 0
    for r in cnt.dom:
        off[r + 1] = off[r] + cnt[r]


@theorem
def integral_fill(
    n: Nat, k: Int, R: Fn[Nat, Real], h: all(R(i) == k for i in Fin[n])
) -> all(R(i) * 2 != 1 for i in Fin[n]):
    """A table of an integer ``k`` is nowhere a half."""


@kernel(uses=[integral_fill])
def fill(s: Real, R: Arr[Fin[n], Real]) -> all(R[i] == s for i in Fin[n]):
    """Fills ``R`` with a real ``s``, which the theorem's integer ``k`` is not.

    At ``s = 0.5``, ``R[i] * 2 != 1`` is false: the theorem is not applied.
    """
    for i in R.dom:
        R[i] = s


@theorem
def copied(
    n: Nat, c: Fn[Nat, Real], R: Fn[Nat, Real], h: all(R(i) == c(i) for i in Fin[n])
) -> all(R(i) - c(i) == 0 for i in Fin[n]):
    """A copy differs from what it copies by nothing."""


@kernel(uses=[copied])
def copy64(c: Arr[Fin[n], np.float64], R: Arr[Fin[n], np.float64]) -> all(
    R[i] == c[i] for i in Fin[n]
):
    """Arrays of numpy's ``float64`` are arrays of reals to a theorem about reals."""
    for i in R.dom:
        R[i] = c[i]


# }}}


# {{{ module-level schedules, whose casts lanky check lists (#191)


#: Two schedules of the scan that share their first step: its facts are one claim.
split = Schedule(scan, target="c").split("r", 2)
split_twice = Schedule(scan, target="c").split("r", 2).split("r_outer", 2)


# }}}
