"""Programs whose earlier call's postcondition does not imply the later requirement.

Each producer here says something about what it writes that falls short of
what the consumer's argument types require: a bound off by one, a claim about
another array, about some cells only, a disjunction, a non-affine value, a
recurrence with the wrong step, or two claims that contradict each other. None
of them may decide the requirement, so each program checks it between the
calls (``loopty.compose``).
"""

from __future__ import annotations

from lanky.prelude import Nat, Real

from loopty import Arr, Fin, kernel, program, reduce_sum

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
