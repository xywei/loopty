"""Kernels whose postconditions are decided from what their statements store, or not.

``tests/test_dataflow.py`` checks each: what is decided, under what, and why
the rest is not.
"""

from __future__ import annotations

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


# }}}


# {{{ module-level schedules, whose casts lanky check lists (#191)


#: Two schedules of the scan that share their first step: its facts are one claim.
split = Schedule(scan, target="c").split("r", 2)
split_twice = Schedule(scan, target="c").split("r", 2).split("r_outer", 2)


# }}}
