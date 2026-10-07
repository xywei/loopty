"""Programs whose earlier call's postcondition does not imply the later requirement.

Each producer here says something about what it writes that falls short of
what the consumer's argument types require: a bound off by one, a claim about
another array, about some cells only, a disjunction, a non-affine value, a
recurrence with the wrong step, or two claims that contradict each other. None
of them may decide the requirement, so each program checks it between the
calls (``loopty.compose``).

The programs at the end are decided, or would be, on the strength of
something no run bears out: a postcondition its kernel's runs refute, one of
a kernel whose term is not its body, a theorem instantiated with a size its
own binder captures, a postcondition of a call whose contract nothing checks
in the program, and an axiom. Each is checked when the program runs.
"""

from __future__ import annotations

from lanky import axiom, theorem
from lanky.prelude import Fn, Int, Nat, Real

from loopty import Arr, Fin, kernel, program, reduce_sum, when

# {{{ an index array, and the reads through it


@kernel
def gather(perm: Arr[Fin[n], Fin[n]], x: Arr[Fin[n], Real], y: Arr[Fin[n], Real]):
    """``y[i] = x[perm[i]]``, in bounds by the element type of ``perm``."""
    for i in y.dom:
        y[i] = x[perm[i]]


@kernel
def past_the_end(perm: Arr[Fin[n], Fin[n]]) -> all(perm[i] == n - i for i in Fin[n]):
    """Off by one: ``n - i`` is ``n`` at ``i = 0``."""
    for i in perm.dom:
        perm[i] = perm.dom.size - 1 - i


@kernel
def the_other(
    perm: Arr[Fin[n], Fin[n]], q: Arr[Fin[n], Fin[n]]
) -> all(q[i] == n - 1 - i for i in Fin[n]):
    """Says what it writes into ``q``, and nothing of ``perm``."""
    for i in perm.dom:
        perm[i] = perm.dom.size - 1 - i
        q[i] = perm.dom.size - 1 - i


@kernel
def all_but_the_last(perm: Arr[Fin[n], Fin[n]]) -> all(
    perm[i] == n - 1 - i for i in Fin[n - 1]
):
    """Says nothing of ``perm[n - 1]``."""
    for i in perm.dom:
        perm[i] = perm.dom.size - 1 - i


@kernel
def all_but_the_first(perm: Arr[Fin[n], Fin[n]]) -> all(
    perm[i] == 0 for i in Fin[n] if i > 0
):
    """Says nothing of ``perm[0]``."""
    for i in perm.dom:
        perm[i] = 0


@kernel
def either(perm: Arr[Fin[n], Fin[n]]) -> all(
    (perm[i] == n - 1 - i) | (perm[i] == n) for i in Fin[n]
):
    """One of two values, and one of them is no point of ``Fin[n]``."""
    for i in perm.dom:
        perm[i] = perm.dom.size - 1 - i


@kernel
def doubled(perm: Arr[Fin[n], Fin[n]]) -> all(perm[i] == 2 * i for i in Fin[n]):
    """``2 * i`` leaves ``Fin[n]`` from ``i = n / 2`` on."""
    for i in perm.dom:
        perm[i] = 2 * i


@kernel
def squared(perm: Arr[Fin[n], Fin[n]]) -> all(perm[i] == i * i for i in Fin[n]):
    """A product of two unknowns, which isl cannot state."""
    for i in perm.dom:
        perm[i] = i * i


@kernel
def at_some(perm: Arr[Fin[n], Fin[n]]) -> any(perm[i] == 0 for i in Fin[n]):
    """An existential, which says nothing of any one cell."""
    for i in perm.dom:
        perm[i] = 0


@kernel
def both(perm: Arr[Fin[n], Fin[n]]) -> (perm[0] == 0) & (perm[0] == 1):
    """A postcondition no run can satisfy, which decides everything."""
    for i in perm.dom:
        perm[i] = 0


@program
def past_the_end_then_gather(perm, x, y):
    past_the_end(perm)
    gather(perm, x, y)


@program
def the_other_then_gather(perm, q, x, y):
    the_other(perm, q)
    gather(perm, x, y)


@program
def all_but_the_last_then_gather(perm, x, y):
    all_but_the_last(perm)
    gather(perm, x, y)


@program
def all_but_the_first_then_gather(perm, x, y):
    all_but_the_first(perm)
    gather(perm, x, y)


@program
def either_then_gather(perm, x, y):
    either(perm)
    gather(perm, x, y)


@program
def doubled_then_gather(perm, x, y):
    doubled(perm)
    gather(perm, x, y)


@program
def squared_then_gather(perm, x, y):
    squared(perm)
    gather(perm, x, y)


@program
def at_some_then_gather(perm, x, y):
    at_some(perm)
    gather(perm, x, y)


@program
def both_then_gather(perm, x, y):
    both(perm)
    gather(perm, x, y)


# }}}


# {{{ a claim about the cells another array counts


@kernel
def one_cell(lim: Arr[Fin[1], Nat]) -> lim[0] == 1:
    """Say how many cells the next call clears."""
    lim[0] = 1


@kernel
def clear_some(perm: Arr[Fin[n], Fin[n]], lim: Arr[Fin[1], Nat]) -> all(
    perm[i] == 0 for i in Fin[lim[0]]
):
    """Clear the first ``lim[0]`` cells, and put ``n`` in the others."""
    for i in perm.dom:
        perm[i] = perm.dom.size
        with when(i < lim[0]):
            perm[i] = 0


@kernel
def every_cell(lim: Arr[Fin[1], Nat], x: Arr[Fin[n], Real]) -> lim[0] == n:
    """Make ``lim[0]`` say every cell, after the cells were cleared."""
    lim[0] = x.dom.size


@program
def cleared(perm, lim, x, y):
    """clear_some's claim is about lim[0] cells, and every_cell changes lim."""
    one_cell(lim)
    clear_some(perm, lim)
    every_cell(lim, x)
    gather(perm, x, y)


# }}}


# {{{ a postcondition that names something it has no value for


@kernel
def touch(x: Arr[Fin[k], Real]):
    """Give the program a size called ``k``, the length of ``x``."""
    for i in x.dom:
        x[i] = x[i] + 0.0


@kernel
def capped(perm: Arr[Fin[n], Fin[n]]) -> all(
    (perm[i] >= 0) & (perm[i] < k) for i in Fin[n]
):
    """``k`` is no parameter and no size of this kernel: natively it has no value."""
    for i in perm.dom:
        perm[i] = perm.dom.size


@program
def capped_then_gather(perm, x, y):
    """The program's k is x's length, which capped's k never meant."""
    touch(x)
    capped(perm)
    gather(perm, x, y)


@kernel
def scan_at_q(
    cnt: Arr[Fin[n], Nat], off: Arr[Fin[n + 1], Nat]
) -> (off[0] == 0) & (off[q] == off[q - 1] + cnt[q - 1]):
    """``q`` is free: it says something of one row nobody named."""
    off[0] = 0
    for r in cnt.dom:
        off[r + 1] = off[r] + cnt[r] + 1


@program
def scanned_at_q(cnt, off, val, y):
    """The layout requirement is asked about every q, which scan_at_q's is not."""
    scan_at_q(cnt, off)
    rowsums(cnt, off, val, y)


# }}}


# {{{ offsets with the wrong step


@kernel
def scan_gapped(
    cnt: Arr[Fin[n], Nat], off: Arr[Fin[n + 1], Nat]
) -> (off[0] == 0) & all(off[r + 1] == off[r] + cnt[r] + 1 for r in Fin[n]):
    """Leaves a gap of one cell after every row, and says so."""
    off[0] = 0
    for r in cnt.dom:
        off[r + 1] = off[r] + cnt[r] + 1


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


@program
def gapped(cnt, off, val, y):
    scan_gapped(cnt, off)
    rowsums(cnt, off, val, y)


@kernel
def scan_gapped_flat(
    cnt: Arr[Fin[n], Nat], off: Arr[Fin[n + 1], Fin[nnz + 1]]
) -> (off[0] == 0) & all(off[r + 1] == off[r] + cnt[r] + 1 for r in Fin[n]):
    """The gapped scan, into offsets of a buffer of ``nnz`` cells."""
    off[0] = 0
    for r in cnt.dom:
        off[r + 1] = off[r] + cnt[r] + 1


@kernel
def weigh(
    cnt: Arr[Fin[n], Nat],
    off: Arr[Fin[n + 1], Fin[nnz + 1]],
    wt: Arr[Fin[n], Fin[cnt], Real],
    val: Arr[Fin[nnz], Real],
    y: Arr[Fin[n], Real],
):
    """Weigh the entries of each row, read from a flat buffer through ``off``."""
    for r in y.dom:
        y[r] = reduce_sum(wt[r, j] * val[off[r] + j] for j in wt.dom[r])


@program
def gapped_flat(cnt, off, wt, val, y):
    """The layout ``weigh`` reads ``wt`` through contradicts the gapped scan."""
    scan_gapped_flat(cnt, off)
    weigh(cnt, off, wt, val, y)


# }}}


# {{{ two buffers, each called nnz by the kernels that read it


@kernel
def scan_flat(
    cnt: Arr[Fin[n], Nat], off: Arr[Fin[n + 1], Fin[nnz + 1]]
) -> (off[0] == 0) & all(off[r + 1] == off[r] + cnt[r] for r in Fin[n]):
    """The scan into offsets of a buffer of ``nnz`` cells."""
    off[0] = 0
    for r in cnt.dom:
        off[r + 1] = off[r] + cnt[r]


@program
def two_buffers(cnt, off, wt, val, y, cnt2, off2, wt2, val2, y2):
    """Two matrices: ``val`` and ``val2`` need not be equally long."""
    scan_flat(cnt, off)
    weigh(cnt, off, wt, val, y)
    scan_flat(cnt2, off2)
    weigh(cnt2, off2, wt2, val2, y2)


# }}}


# {{{ decided on the strength of what no run bears out


@kernel
def liar(perm: Arr[Fin[n], Fin[n]]) -> all(perm[i] == n - 1 - i for i in Fin[n]):
    """Says it reverses the cells, and counts up to ``n`` instead (#115)."""
    for i in perm.dom:
        perm[i] = i + 1


@program
def lied_to(perm, x, y):
    """Decided under liar's postcondition, which its own runs refute."""
    liar(perm)
    gather(perm, x, y)


@kernel
def two_faced(perm: Arr[Fin[n], Fin[n]]) -> all(perm[i] == n - 1 - i for i in Fin[n]):
    """Reverses the cells natively, and counts up to ``n`` in its traced term.

    The postcondition is true of every native run, and the compiled program
    runs the term, which an ``isinstance`` the trace takes the other way makes
    another kernel: its ``trace-faithful`` fact is refuted.
    """
    for i in perm.dom:
        perm[i] = perm.dom.size - 1 - i
    if not isinstance(perm, Arr):
        for i in perm.dom:
            perm[i] = i + 1


@program
def faced(perm, x, y):
    """Decided under a postcondition of the body, which the term does not keep."""
    two_faced(perm)
    gather(perm, x, y)


@kernel
def up_to_a(perm: Arr[Fin[a], Fin[a]]) -> all(perm[j] == j + 1 for j in Fin[a]):
    """Counts up to ``a``, past the end, and says so; its size is called ``a``."""
    for i in perm.dom:
        perm[i] = i + 1


@theorem
def bounded(
    n: Nat,
    f: Fn[Fin[n], Int],
    h: all(f(a) == a + 1 for a in Fin[n]),
) -> all((f(a) <= n) & (f(a) >= 0) for a in Fin[n]):
    """True: ``f(a) = a + 1 <= n`` where ``a < n``. Its binder is called ``a``."""


@program(uses=[bounded])
def captured(perm, x, y):
    """bounded at n = a is about the size a, not about its own binder a."""
    up_to_a(perm)
    gather(perm, x, y)


@kernel
def below_zero(src: Arr[Fin[n], Nat]):
    """Leaves negative cells in an array of naturals, and says nothing."""
    for i in src.dom:
        src[i] = i - src.dom.size


@kernel
def clamp(
    src: Arr[Fin[n], Nat], perm: Arr[Fin[n], Fin[n]]
) -> all((perm[i] >= 0) & (perm[i] < n) for i in Fin[n]):
    """Copies the naturals below ``n``: a point of ``Fin[n]`` for a natural."""
    for i in perm.dom:
        perm[i] = perm.dom.size - 1
        with when(src[i] < perm.dom.size):
            perm[i] = src[i]


@program
def clamped(src, perm, x, y):
    """clamp's contract refuses src natively; nothing checks it compiled."""
    below_zero(src)
    clamp(src, perm)
    gather(perm, x, y)


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


@axiom(cite="any book on prefix sums")
def scan_monotone_cited(
    n: Nat,
    cnt: Fn[Fin[n], Nat],
    off: Fn[Fin[n + 1], Int],
    h0: off(0) == 0,
    hs: all(off(r + 1) == off(r) + cnt(r) for r in Fin[n]),
) -> all(off(a) <= off(b) for a in Fin[n + 1] for b in Fin[n + 1] if a <= b):
    """The offsets a scan produces are monotone, on a citation."""


@kernel
def pick(
    off: Arr[Fin[n + 1], Fin[n + 1]],
    x: Arr[Fin[n + 1], Real],
    y: Arr[Fin[n + 1], Real],
):
    """``y[i] = x[off[i]]``, in bounds by the element type of ``off``."""
    for i in y.dom:
        y[i] = x[off[i]]


@program(uses=[scan_monotone_cited])
def picked_on_a_citation(cnt, off, x, y):
    """Decided under the axiom, which is assumed: checked all the same."""
    scan_unit(cnt, off)
    pick(off, x, y)


# }}}
