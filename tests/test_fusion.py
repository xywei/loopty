"""Fusing a program's calls, and the storage of what passes between them (#13).

A program's term runs its calls one after the other (:mod:`loopty.compose`),
and lowers to one kernel whose loops follow one another. Three things are
built on that here:

* a ``definedness`` fact for each array the program makes and each call that
  reads it after another call wrote it, decided by isl: each cell it reads
  is one a call before it stored or one of the zeros the array was made
  with, and the fact says where it reads the zeros;
* :meth:`Schedule.fuse`, a map per statement that runs the consumer's loops
  in the producer's, checked as every cast is, and refused with the pair of
  instances that it would run backwards;
* :meth:`Schedule.substitute`, which computes such an array where it is
  read instead of storing it, through loopy's ``assignment_to_subst``, and
  checks every later step against the dependences of the program as it then
  runs.
"""

from __future__ import annotations

import islpy as isl
import numpy as np
import pytest
from lanky.ledger import Status
from lanky.prelude import Int, Nat, Real

from loopty import Arr, Fin, Schedule, kernel, program, reduce_sum, when
from loopty.executor import LoopyExecutor, emit_code
from loopty.oracle import IslOracle
from loopty.schedule import IllegalCast

# {{{ the kernels and programs


@kernel
def flux(u: Arr[Fin[n], Real], f: Arr[Fin[n], Real]):  # noqa: F821
    """Pointwise Burgers flux."""
    for j in u.dom:
        f[j] = 0.5 * u[j] * u[j]


@kernel
def divergence(f: Arr[Fin[n], Real], rhs: Arr[Fin[n], Real]):  # noqa: F821
    """Centred divergence of the flux, inside the boundary."""
    for i in rhs.dom:
        with when((i > 0) & (i + 1 < rhs.dom.size)):
            rhs[i] = -(f[i + 1] - f[i - 1]) / 2


@kernel
def interior_flux(u: Arr[Fin[n], Real], f: Arr[Fin[n], Real]):  # noqa: F821
    """The flux at the interior points only."""
    for j in u.dom:
        with when((j > 0) & (j + 1 < u.dom.size)):
            f[j] = 0.5 * u[j] * u[j]


@kernel
def shifted(f: Arr[Fin[n], Real], y: Arr[Fin[n], Real]):  # noqa: F821
    """Every cell, plus one."""
    for i in y.dom:
        y[i] = f[i] + 1.0


@kernel
def doubled(f: Arr[Fin[n], Real], g: Arr[Fin[n], Real]):  # noqa: F821
    """Every cell, twice."""
    for k in g.dom:
        g[k] = 2.0 * f[k]


@kernel
def front(f: Arr[Fin[n], Real], h: Arr[Fin[n], Real]):  # noqa: F821
    """The front half of ``f``."""
    for x in h.dom:
        with when(2 * x < h.dom.size):
            h[x] = f[x]


@kernel
def back(h: Arr[Fin[n], Real], rhs: Arr[Fin[n], Real]):  # noqa: F821
    """The back half of ``h``, plus one."""
    for i in rhs.dom:
        with when(2 * i >= rhs.dom.size):
            rhs[i] = h[i] + 1.0


@kernel
def bump(u: Arr[Fin[n], Real]):  # noqa: F821
    """Add one to every cell, in place."""
    for k in u.dom:
        u[k] = u[k] + 1.0


@kernel
def update(f: Arr[Fin[n], Real], u: Arr[Fin[n], Real]):  # noqa: F821
    """One explicit step of Burgers' equation, in place."""
    for i in u.dom:
        with when((i > 0) & (i + 1 < u.dom.size)):
            u[i] = u[i] - (f[i + 1] - f[i - 1]) / 2


@kernel
def halved_flux(u: Arr[Fin[n], Real], f: Arr[Fin[n], Real]):  # noqa: F821
    """The flux of each pair of cells, stored twice."""
    for j in u.dom:
        f[j // 2] = 0.5 * u[j] * u[j]


@kernel
def totals(u: Arr[Fin[n], Real], f: Arr[Fin[n], Real]):  # noqa: F821
    """The sum of every cell, in every cell."""
    for j in f.dom:
        f[j] = reduce_sum(u[k] for k in u.dom)


@kernel
def gather(
    idx: Arr[Fin[n], Fin[n]],  # noqa: F821
    f: Arr[Fin[n], Real],  # noqa: F821
    y: Arr[Fin[n], Real],  # noqa: F821
):
    """Each cell of ``f`` that ``idx`` names."""
    for i in y.dom:
        y[i] = f[idx[i]]


@kernel
def square2(
    a: Arr[Fin[n], Fin[m], Real],  # noqa: F821
    g: Arr[Fin[n], Fin[m], Real],  # noqa: F821
):
    """Every cell squared."""
    for i in a.dom:
        for j in a.dom[i]:
            g[i, j] = a[i, j] * a[i, j]


@kernel
def down(
    g: Arr[Fin[n], Fin[m], Real],  # noqa: F821
    b: Arr[Fin[n], Fin[m], Real],  # noqa: F821
):
    """Each cell less the one below it."""
    for p in b.dom:
        for q in b.dom[p]:
            with when(p + 1 < b.dom.size):
                b[p, q] = g[p + 1, q] - g[p, q]


@kernel
def row_total(
    g: Arr[Fin[n], Fin[m], Real],  # noqa: F821
    z: Arr[Fin[n], Real],  # noqa: F821
):
    """The sum of each row."""
    for w in z.dom:
        z[w] = reduce_sum(g[w, q] for q in g.dom[w])


@kernel
def rows_then_cells(
    a: Arr[Fin[n], Fin[m], Real],  # noqa: F821
    s: Arr[Fin[n], Real],  # noqa: F821
    g: Arr[Fin[n], Fin[m], Real],  # noqa: F821
):
    """A statement per row, and one per cell of the row."""
    for r in s.dom:
        s[r] = 1.0 + r
        for c in a.dom[r]:
            g[r, c] = a[r, c] + 1.0


@kernel
def row_sum(s: Arr[Fin[n], Real], z: Arr[Fin[n], Real]):  # noqa: F821
    """Each row's value, doubled."""
    for w in z.dom:
        z[w] = s[w] + s[w]


@program
def burgers(u, rhs):
    """The flux into an array the program makes, and its divergence."""
    f = Arr.zeros_like(u)
    flux(u, f)
    divergence(f, rhs)


@program
def padded(u, y):
    """A consumer that reads the boundary cells the producer leaves zero."""
    f = Arr.zeros_like(u)
    interior_flux(u, f)
    shifted(f, y)


@program
def early(u, rhs, y):
    """A read of the array before the producer stores it."""
    f = Arr.zeros_like(u)
    shifted(f, y)
    flux(u, f)
    divergence(f, rhs)


@program
def chained(u, rhs):
    """Two arrays the program makes, the second computed from the first."""
    f = Arr.zeros_like(u)
    g = Arr.zeros_like(u)
    flux(u, f)
    doubled(f, g)
    divergence(g, rhs)


@program
def between(u, rhs):
    """A call between the two fused that the second reads only by array."""
    f = Arr.zeros_like(u)
    h = Arr.zeros_like(u)
    flux(u, f)
    front(f, h)
    back(h, rhs)


@program
def bumped(u, rhs):
    """The producer's input written between the producer and the consumer."""
    f = Arr.zeros_like(u)
    flux(u, f)
    bump(u)
    divergence(f, rhs)


@program
def bumped_after(u, rhs):
    """The producer's input written after the consumer has read the array."""
    f = Arr.zeros_like(u)
    flux(u, f)
    divergence(f, rhs)
    bump(u)


@program
def stepped(u):
    """The consumer writes in place what the producer read."""
    f = Arr.zeros_like(u)
    flux(u, f)
    update(f, u)


@program
def twice(u, rhs):
    """Two calls store the same array."""
    f = Arr.zeros_like(u)
    flux(u, f)
    flux(u, f)
    divergence(f, rhs)


@program
def halved(u, rhs):
    """A producer that stores each cell of half the array twice."""
    f = Arr.zeros_like(u)
    halved_flux(u, f)
    divergence(f, rhs)


@program
def summed(u, rhs):
    """A producer that stores a sum."""
    f = Arr.zeros_like(u)
    totals(u, f)
    divergence(f, rhs)


@program
def unread(u, rhs):
    """An array the program makes and nothing reads."""
    f = Arr.zeros_like(u)
    flux(u, f)
    flux(u, rhs)


@program
def through(u, f, rhs):
    """The producer's array is a parameter of the program."""
    flux(u, f)
    divergence(f, rhs)


@program
def gathered(u, idx, y):
    """The array read through an index array."""
    f = Arr.zeros_like(u)
    flux(u, f)
    gather(idx, f, y)


@program
def gathered_inside(u, idx, y):
    """The array read through an index array, and stored inside only."""
    f = Arr.zeros_like(u)
    interior_flux(u, f)
    gather(idx, f, y)


@program
def squares(a, b):
    """Two nests of two loops, the second reading the row below."""
    g = Arr.zeros_like(a)
    square2(a, g)
    down(g, b)


@program
def squared_rows(a, z):
    """Every cell squared, then each row summed."""
    g = Arr.zeros_like(a)
    square2(a, g)
    row_total(g, z)


@program
def rows(a, z, g):
    """A producer with statements at two depths, then a consumer of one."""
    s = Arr.zeros_like(z)
    rows_then_cells(a, s, g)
    row_sum(s, z)


@kernel
def mark(u: Arr[Fin[n], Real], m: Arr[Fin[n], Real]):  # noqa: F821
    """One at the interior points."""
    for j in u.dom:
        with when((j > 0) & (j + 1 < u.dom.size)):
            m[j] = 1.0


@kernel
def masked(
    f: Arr[Fin[n], Real],  # noqa: F821
    m: Arr[Fin[n], Real],  # noqa: F821
    y: Arr[Fin[n], Real],  # noqa: F821
):
    """``f`` where ``m`` is set, a guard isl cannot state."""
    for i in y.dom:
        with when(m[i] > 0.5):
            y[i] = f[i]


@program
def marked(u, y):
    """masked reads f only where mark set m, which is where f is stored."""
    f = Arr.zeros_like(u)
    m = Arr.zeros_like(u)
    interior_flux(u, f)
    mark(u, m)
    masked(f, m, y)


@kernel
def bounded(f: Arr[Fin[n], Real], y: Arr[Fin[n], Real]):  # noqa: F821
    """Store the first and last cells of ``f``, then read every cell."""
    for k in f.dom:
        with when(k == 0):
            f[k] = 1.0
    for e in f.dom:
        with when(e + 1 == f.dom.size):
            f[e] = 2.0
    for i in y.dom:
        y[i] = f[i] + 1.0


@kernel
def first_edge(f: Arr[Fin[n], Real], y: Arr[Fin[n], Real]):  # noqa: F821
    """Store the first cell of ``f``, then read every cell."""
    for k in f.dom:
        with when(k == 0):
            f[k] = 1.0
    for i in y.dom:
        y[i] = f[i] + 1.0


@program
def edged(u, y):
    """The inside of f stored by one call, its edges by the one that reads it."""
    f = Arr.zeros_like(u)
    interior_flux(u, f)
    bounded(f, y)


@program
def half_edged(u, y):
    """The same, with the last cell stored by nobody."""
    f = Arr.zeros_like(u)
    interior_flux(u, f)
    first_edge(f, y)


@kernel
def plus_one(w: Arr[Fin[m], Real], y: Arr[Fin[m], Real]):  # noqa: F821
    """Every cell of ``w``, plus one."""
    for i in y.dom:
        y[i] = w[i] + 1.0


@program
def two_sizes(u, w, y, z):
    """A call over ``n`` and one over ``m`` between the two ends of an edge."""
    f = Arr.zeros_like(u)
    doubled(u, f)
    plus_one(w, y)
    shifted(f, z)


@kernel
def triple(u: Arr[Fin[n], Int], f: Arr[Fin[n], Int]):  # noqa: F821
    """Three times each cell, plus one."""
    for j in u.dom:
        f[j] = u[j] * 3 + 1


@kernel
def forward(f: Arr[Fin[n], Int], y: Arr[Fin[n], Int]):  # noqa: F821
    """Each cell less the one before it, but the last."""
    for i in y.dom:
        with when(i + 1 < y.dom.size):
            y[i] = f[i + 1] - f[i]


@program
def whole(u, y):
    """The same edge in integers, which every run has to agree on exactly."""
    f = Arr.zeros_like(u)
    triple(u, f)
    forward(f, y)


@kernel
def reverse_quiet(idx: Arr[Fin[n], Fin[n]]):  # noqa: F821
    """Reverse the cells, and say nothing about it."""
    for i in idx.dom:
        idx[i] = idx.dom.size - 1 - i


@kernel
def number_up(idx: Arr[Fin[n], Fin[n]]):  # noqa: F821
    """Off by one at the end: ``idx[n - 1]`` is ``n``, and nothing said."""
    for i in idx.dom:
        idx[i] = i + 1


@program
def reversed_gather(idx, x, y):
    """Nothing says what reverse_quiet writes, so gather's is checked."""
    reverse_quiet(idx)
    gather(idx, x, y)


@program
def wrong_gather(idx, x, y):
    """The same with an index past the end, which the check stops."""
    number_up(idx)
    gather(idx, x, y)


@program
def checked_flux(idx, x, y, rhs):
    """A checked point, then an edge through an array the program makes."""
    reverse_quiet(idx)
    gather(idx, x, y)
    f = Arr.zeros_like(y)
    flux(y, f)
    divergence(f, rhs)


@kernel
def tenth(u: Arr[Fin[n], Real], f: Arr[Fin[n], np.float32]):  # noqa: F821
    """A tenth of each cell, stored in single precision."""
    for j in u.dom:
        f[j] = u[j] * 0.1


@kernel
def tenfold(
    f: Arr[Fin[n], np.float32],  # noqa: F821
    t: Arr[Fin[n], np.float32],  # noqa: F821
    y: Arr[Fin[n], Real],  # noqa: F821
):
    """Ten times each cell, plus ``t``."""
    for i in y.dom:
        y[i] = f[i] * 10.0 + t[i]


@program
def single(u, t, y):
    """An intermediate in single precision, so each store rounds."""
    f = Arr.zeros_like(t)
    tenth(u, f)
    tenfold(f, t, y)


@kernel
def halfway(u: Arr[Fin[n], Real], f: Arr[Fin[n], Nat]):  # noqa: F821
    """Half of each cell, stored in a whole number, which truncates it."""
    for j in u.dom:
        f[j] = u[j] * 0.5


@kernel
def twice_whole(f: Arr[Fin[n], Nat], y: Arr[Fin[n], Nat]):  # noqa: F821
    """Twice each cell."""
    for i in y.dom:
        y[i] = f[i] * 2


@program
def truncated(u, y):
    """An intermediate of whole numbers, which the store truncates."""
    f = Arr.zeros_like(y)
    halfway(u, f)
    twice_whole(f, y)


@kernel
def copy_index(t: Arr[Fin[n], Fin[n]], p: Arr[Fin[n], Fin[n]]):  # noqa: F821
    """A copy of an index array."""
    for j in p.dom:
        p[j] = t[j]


@kernel
def scatter(
    p: Arr[Fin[n], Fin[n]],  # noqa: F821
    x: Arr[Fin[n], Real],  # noqa: F821
    y: Arr[Fin[n], Real],  # noqa: F821
):
    """``y[p[i]] = x[i]``."""
    for i in x.dom:
        y[p[i]] = x[i]


@program
def scattered(t, x, y):
    """An index array the program computes, then writes through."""
    p = Arr.zeros_like(t)
    copy_index(t, p)
    scatter(p, x, y)


@kernel
def narrow_index(t: Arr[Fin[n], Fin[n]], p: Arr[Fin[n], np.int16]):  # noqa: F821
    """A copy of an index array, in sixteen bits."""
    for j in p.dom:
        p[j] = t[j]


@kernel
def scatter_narrow(
    p: Arr[Fin[n], np.int16],  # noqa: F821
    w: Arr[Fin[n], np.int16],  # noqa: F821
    x: Arr[Fin[n], Real],  # noqa: F821
    y: Arr[Fin[n], Real],  # noqa: F821
):
    """``y[p[i]] = x[i] + w[i]``."""
    for i in x.dom:
        y[p[i]] = x[i] + w[i]


@program
def narrowed(t, w, x, y):
    """The index array stored in sixteen bits, which converts it."""
    p = Arr.zeros_like(w)
    narrow_index(t, p)
    scatter_narrow(p, w, x, y)


@kernel
def rotate(t: Arr[Fin[n], Fin[n]], p: Arr[Fin[n], Fin[n]]):  # noqa: F821
    """Each index of ``t``, one further round."""
    for j in p.dom:
        p[j] = (t[j] + 1) % p.dom.size


@program
def rotated(t, x, y):
    """An index array computed in 64 bits and stored in 32, then read through."""
    p = Arr.zeros_like(t)
    rotate(t, p)
    gather(p, x, y)


@kernel
def widen(t: Arr[Fin[n], Fin[n]], q: Arr[Fin[n], Nat]):  # noqa: F821
    """A copy of an index array, in whole numbers."""
    for j in q.dom:
        q[j] = t[j]


@kernel
def gather_whole(
    q: Arr[Fin[n], Nat],  # noqa: F821
    x: Arr[Fin[n], Real],  # noqa: F821
    y: Arr[Fin[n], Real],  # noqa: F821
):
    """Each cell of ``x`` that ``q`` names."""
    for i in y.dom:
        y[i] = x[q[i]]


@program
def widened(t, x, y):
    """An index array computed in 32 bits and stored in 64, then read through."""
    q = Arr.zeros_like(t)
    widen(t, q)
    gather_whole(q, x, y)


@kernel
def reversal(q: Arr[Fin[n], Nat]):  # noqa: F821
    """The last index first, in whole numbers."""
    for j in q.dom:
        q[j] = q.dom.size - 1 - j


@program
def reversed_whole(x, y):
    """An index computed in 32 bits from the sizes, stored in 64, then read."""
    q = Arr.zeros_like(x, dtype=np.int64)
    reversal(q)
    gather_whole(q, x, y)


@program
def scattered_into(p, x, y):
    """The array stored through an index array, then read at every cell."""
    f = Arr.zeros_like(x)
    scatter(p, x, f)
    shifted(f, y)


def velocity(size: int) -> np.ndarray:
    return np.sin(np.linspace(0.0, 2.0 * np.pi, size, endpoint=False)) + 0.3


def burgers_inputs(size: int = 8) -> dict:
    return {"u": Arr.from_numpy(velocity(size)), "rhs": Arr.zeros(size)}


def agrees(prog, schedule, inputs: dict) -> None:
    fact = LoopyExecutor().differential(prog, schedule, inputs)
    assert fact.status is Status.TESTED, fact.provenance


def statements(schedule: Schedule) -> list[tuple[str, str]]:
    return [(fact.kind, fact.status.value) for fact in schedule.facts()]


# }}}


# {{{ definedness


def definedness(prog) -> list:
    return [fact for fact in prog.facts() if fact.kind == "definedness"]


def test_the_consumer_reads_only_cells_the_producer_stored() -> None:
    (fact,) = definedness(burgers)
    assert fact.statement == (
        "every cell of f that divergence reads, flux stored before it"
    )
    assert fact.status is Status.ASSUMED
    decided = IslOracle().establish(fact)
    assert decided.status is Status.DECIDED
    assert decided.decided_by == "isl"
    assert fact.id.startswith("definedness:")
    assert fact.id.endswith(":f:divergence")


def test_a_consumer_that_reads_the_zeros_is_decided_with_the_cells() -> None:
    # shifted reads every cell, and interior_flux stores the inside ones: the
    # boundary cells hold the zeros f was made with when they are read, which
    # the zeroing stored when the program made f. Zero padding at a boundary
    # is meant, so the fact is decided, and says where the zeros are read.
    (fact,) = definedness(padded)
    assert fact.statement == (
        "every cell of f that shifted reads is one interior_flux stored before "
        "it or one of the zeros f was made with"
    )
    reason = "shifted.S0 reads the zeros at f[0] and f[n - 1]"
    assert fact.provenance["reason"] == reason
    decided = IslOracle().establish(fact)
    assert decided.status is Status.DECIDED
    assert decided.decided_by == "isl"
    assert decided.provenance["reason"] == reason
    # front stores the front half of h and back reads the back half, all
    # zeros; f, which flux stores in full, is read by front with no zero.
    front_half, back_half = definedness(between)
    assert front_half.statement == (
        "every cell of f that front reads, flux stored before it"
    )
    assert "reason" not in front_half.provenance
    assert back_half.provenance["reason"] == (
        "back.S0 reads the zeros at h[a0] for a0 <= n - 1, 2*a0 >= n"
    )
    # halved_flux stores the first half of f, and divergence reads the rest.
    (top,) = definedness(halved)
    assert top.provenance["reason"] == (
        "divergence.S0 reads the zeros at f[a0] for 2 <= a0 <= n - 1, 2*a0 >= n"
    )
    for fact in (front_half, back_half, top):
        assert IslOracle().establish(fact).status is Status.DECIDED


PADDED = """
from __future__ import annotations

from lanky.prelude import Real

from loopty import Arr, Fin, kernel, program, when


@kernel
def interior_flux(u: Arr[Fin[n], Real], f: Arr[Fin[n], Real]):
    for j in u.dom:
        with when((j > 0) & (j + 1 < u.dom.size)):
            f[j] = 0.5 * u[j] * u[j]


@kernel
def shifted(f: Arr[Fin[n], Real], y: Arr[Fin[n], Real]):
    for i in y.dom:
        y[i] = f[i] + 1.0


@program
def padded(u, y):
    f = Arr.zeros_like(u)
    interior_flux(u, f)
    shifted(f, y)
"""


def test_lanky_check_passes_a_program_that_reads_its_zero_padding(
    tmp_path, capsys
) -> None:
    from lanky.cli import main as lanky_main

    path = tmp_path / "padding.py"
    path.write_text(PADDED, encoding="utf-8")
    assert lanky_main(["check", str(path)]) == 0
    rows = capsys.readouterr().out.splitlines()
    (row,) = [line for line in rows if "every cell of f that shifted reads" in line]
    assert row.split()[:2] == ["decided", "isl"]
    assert not [line for line in rows if line.startswith("REFUTED")]


def test_the_cells_a_reason_lists_are_written_out() -> None:
    from loopty.flow import cells_text

    within = isl.Set("[n] -> { [a0] : 0 <= a0 < n }")
    # f[n - 1] is f[0] at n = 1, so the condition n >= 2 says nothing more.
    ends = isl.Set("[n] -> { [a0] : n >= 1 and (a0 = 0 or a0 = n - 1) }")
    assert cells_text(ends, "f", within) == "f[0] and f[n - 1]"
    late = isl.Set("[n] -> { [a0] : a0 = 0 and n >= 5 }")
    assert cells_text(late, "f", within) == "f[0] when n >= 5"
    row = isl.Set("[m, n] -> { [a0, a1] : a0 = n - 1 and 0 <= a1 < m }")
    assert cells_text(row, "g") == "g[n - 1, a1] for 0 <= a1 <= m - 1"
    # A piece isl writes with a remainder is given in isl's words.
    even = isl.Set("[n] -> { [a0] : 0 <= a0 < n and a0 mod 2 = 0 }")
    assert cells_text(even, "f", within).startswith("the cells [n] -> {")


def test_a_cell_the_reader_stores_itself_is_a_zero_or_its_own() -> None:
    # bounded stores f's first and last cells and then reads every cell:
    # interior_flux stored the others, and whether bounded stored its own
    # before it read them is not asked. Either way the read is of a zero or
    # of what bounded stored, and the reason says it is one or the other.
    (fact,) = definedness(edged)
    assert IslOracle().establish(fact).status is Status.DECIDED
    assert fact.provenance["reason"] == (
        "bounded.S2 reads f[0] and f[n - 1], the zeros there unless bounded.S0 "
        "or bounded.S1 stored them before the read, which is not asked"
    )
    # Storing only the first, the last cell is a zero nobody stored, at
    # every size it is not the first.
    (fact,) = definedness(half_edged)
    assert IslOracle().establish(fact).status is Status.DECIDED
    assert fact.provenance["reason"] == (
        "first_edge.S1 reads the zeros at f[n - 1] when n >= 2; first_edge.S1 "
        "reads f[0], the zeros there unless first_edge.S0 stored them before "
        "the read, which is not asked"
    )


def test_a_write_isl_cannot_list_may_have_stored_the_zeros_read() -> None:
    # scatter stores f[p[i]], which isl cannot list, and shifted reads every
    # cell: each holds a zero or what scatter stored there, both stored, so
    # the fact is decided, and the reason says the zeros may be gone.
    (fact,) = definedness(scattered_into)
    assert IslOracle().establish(fact).status is Status.DECIDED
    assert fact.provenance["reason"] == (
        "shifted.S0 reads the zeros at f[a0] for 0 <= a0 <= n - 1, unless "
        "scatter.S0 stored them, which isl cannot list"
    )


def test_a_read_under_a_guard_isl_cannot_state_is_not_shown_to_see_zeros() -> None:
    # masked reads f[i] only where m[i] > 0.5, which mark sets inside, where
    # interior_flux stores f: no zero reaches a write. Its read is listed
    # over every i, so a cell outside is only one it may read, not one it is
    # shown to read the zeros at, and the fact stays assumed. Its guard's
    # read of m is listed over every i as well, under that same guard.
    facts = {fact.provenance["array"]: fact for fact in definedness(marked)}
    for array in ("f", "m"):
        fact = facts[array]
        assert fact.status is Status.ASSUMED, fact.statement
        assert fact.term is None
        reason = fact.provenance["reason"]
        assert f"masked.S0 reads {array} under a guard isl cannot state (m[" in reason
    y = Arr.zeros(6)
    marked(Arr.from_numpy(velocity(6)), y)
    assert y.numpy()[0] == 0.0 and y.numpy()[-1] == 0.0


def test_the_guard_of_a_checked_point_leaves_an_edge_decided() -> None:
    # gather's requirement on idx is checked between the calls, and every
    # statement after the check runs only where its flag is clear: flux's
    # store and divergence's reads are guarded by it, and still decided,
    # since nothing runs where it is set.
    term = checked_flux.term
    assert term.checks
    assert term.stmt("flux.S0").unnarrowed
    (fact,) = definedness(checked_flux)
    assert fact.term is not None, fact.provenance
    assert IslOracle().establish(fact).status is Status.DECIDED


def test_a_read_through_an_index_array_is_decided_when_every_cell_is_stored() -> None:
    (fact,) = definedness(gathered)
    assert IslOracle().establish(fact).status is Status.DECIDED
    (unknown,) = definedness(gathered_inside)
    assert unknown.status is Status.ASSUMED
    assert unknown.term is None
    assert "not affine" in unknown.provenance["reason"]


def test_a_read_before_any_call_stored_the_array_is_no_edge() -> None:
    # shifted reads the zeros before flux runs: no call stored f before it,
    # so there is no edge to ask about. divergence's edge is decided.
    (fact,) = definedness(early)
    assert fact.provenance["call"] == "divergence"


def test_an_edge_through_a_parameter_has_no_definedness_fact() -> None:
    # The caller passes f and sees it: it is not the program's to store or
    # not, and what its cells hold before flux runs is the caller's.
    assert definedness(through) == []


# }}}


# {{{ fusion


def test_a_fusion_that_runs_a_dependence_backwards_is_refused_with_the_pair() -> None:
    schedule = Schedule(burgers, sizes={"n": 16})
    with pytest.raises(IllegalCast) as caught:
        schedule.fuse("flux", "divergence")
    assert caught.value.fact.kind == "monotone"
    assert caught.value.fact.status is Status.REFUTED
    (source, source_at), (sink, sink_at), sizes = caught.value.witness
    assert (source, sink) == ("flux.S0", "divergence.S0")
    assert source_at["j"] == sink_at["i"] + 1
    assert sizes == {"n": 16}
    message = str(caught.value)
    assert message.startswith("fuse(flux, divergence) illegal: instance flux.S0[")
    # The refusal names the least shift that is accepted, which it checked.
    assert message.endswith(
        "fuse('flux', 'divergence', shift=1) runs every dependence between "
        "them forward"
    )
    assert caught.value.fact.provenance["reason"] == message


def test_the_fused_loop_is_one_loop_and_agrees_with_the_native_run() -> None:
    fused = Schedule(burgers, sizes={"n": 16}).fuse("flux", "divergence", shift=1)
    assert fused.history == ("fuse(flux, divergence, shift=1)",)
    assert fused.key == "burgers[c].fuse('flux', 'divergence', shift=1)"
    assert statements(fused) == [("bijective", "decided"), ("monotone", "decided")]
    loop = fused.term.stmt("flux.S0").inames[0]
    assert fused._layout.coords["divergence.S0"] == (loop,)
    entry = fused.kernel.default_entrypoint
    insns = {insn.id: insn for insn in entry.instructions}
    assert insns["flux_S0"].within_inames == insns["divergence_S0"].within_inames
    owners = [d for d in entry.domains if loop in d.get_var_names(isl.dim_type.set)]
    assert len(owners) == 1
    for size in (1, 2, 3, 5, 16):
        agrees(burgers, fused, burgers_inputs(size))


def test_a_fused_and_substituted_edge_of_integers_agrees_exactly() -> None:
    # Integers are exact: each schedule's run is compared bit for bit.
    fused = Schedule(whole).fuse("triple", "forward", shift=1)
    for schedule in (fused, fused.substitute("f")):
        for size in (1, 2, 7):
            inputs = {
                "u": Arr.from_numpy(np.arange(size, dtype=np.int64) * 7 - 3),
                "y": Arr.from_numpy(np.zeros(size, dtype=np.int64)),
            }
            fact = LoopyExecutor().differential(whole, schedule, inputs)
            assert fact.status is Status.TESTED, fact.provenance
            (output,) = fact.provenance["outputs"].values()
            assert (output["exactness"], output["difference"]) == ("exact", 0)


def test_a_fusion_is_a_map_per_statement_and_affine_takes_it_as_well() -> None:
    term = burgers.term
    j = term.stmt("flux.S0").inames[0]
    i = term.stmt("divergence.S0").inames[0]
    by_hand = Schedule(burgers).affine(
        f"{{ flux_S0[{j}] -> [k] : k = {j}; divergence_S0[{i}] -> [k] : k = {i} + 1 }}"
    )
    assert statements(by_hand) == [("bijective", "decided"), ("monotone", "decided")]
    assert by_hand.order[-1] == "k"
    agrees(burgers, by_hand, burgers_inputs(9))
    with pytest.raises(ValueError, match="their maps have to make the same new ones"):
        Schedule(burgers).affine(
            f"{{ flux_S0[{j}] -> [k] : k = {j}; divergence_S0[{i}] -> [m] : m = {i} }}"
        )


def test_a_fusion_is_replayed_when_retargeted() -> None:
    fused = Schedule(burgers).fuse("flux", "divergence", shift=1)
    again = fused.retarget("c-source")
    assert again.history == fused.history
    assert statements(again) == statements(fused)


def test_a_fusion_in_place_runs_the_writes_after_the_reads() -> None:
    # update writes u[i] in place, which flux reads at j = i: run one step
    # behind, flux has read it by then.
    fused = Schedule(stepped).fuse("flux", "update", shift=1)
    assert statements(fused) == [("bijective", "decided"), ("monotone", "decided")]
    for size in (1, 3, 8):
        agrees(stepped, fused, {"u": Arr.from_numpy(velocity(size))})


def test_two_loops_fuse_with_a_shift_each() -> None:
    schedule = Schedule(squares, sizes={"n": 6, "m": 5})
    with pytest.raises(IllegalCast) as caught:
        schedule.fuse("square2", "down")
    assert str(caught.value).endswith(
        "fuse('square2', 'down', shift=(1, 0)) runs every dependence between "
        "them forward"
    )
    fused = schedule.fuse("square2", "down", shift=(1, 0))
    assert statements(fused) == [("bijective", "decided"), ("monotone", "decided")]
    for rows, cols in ((1, 1), (2, 3), (6, 5)):
        a = np.arange(rows * cols, dtype=float).reshape(rows, cols) / 7.0
        agrees(
            squares,
            fused,
            {"a": Arr.from_numpy(a), "b": Arr.from_numpy(np.zeros((rows, cols)))},
        )
    with pytest.raises(ValueError, match="one shift per loop"):
        schedule.fuse("square2", "down", shift=1)


def test_a_statement_deeper_in_the_producer_moves_with_its_row() -> None:
    # rows_then_cells has a statement in r and one in r and c; the domain of
    # c is nested in r's, and moves along the producer's map with it.
    fused = Schedule(rows).fuse("rows_then_cells", "row_sum")
    assert statements(fused) == [("bijective", "decided"), ("monotone", "decided")]
    assert fused.buildable == (True, "")
    for size, width in ((1, 1), (3, 2), (5, 4)):
        a = np.arange(size * width, dtype=float).reshape(size, width) + 1.0
        agrees(
            rows,
            fused,
            {
                "a": Arr.from_numpy(a),
                "z": Arr.zeros(size),
                "g": Arr.from_numpy(np.zeros((size, width))),
            },
        )


def test_the_outer_loop_of_a_nest_fuses_with_a_loop_of_one_level() -> None:
    # square2's two loops are one domain of the kernel, since no statement
    # leaves the nest; the fusion takes the outer one and cuts the domain
    # after it, as the lowering cuts a nest a statement leaves. Each row is
    # squared and then summed, in one loop over the rows.
    fused = Schedule(squared_rows).fuse("square2", "row_total")
    assert statements(fused) == [("bijective", "decided"), ("monotone", "decided")]
    assert fused.buildable == (True, "")
    loop = squared_rows.term.stmt("square2.S0").inames[0]
    entry = fused.kernel.default_entrypoint
    insns = {insn.id: insn for insn in entry.instructions}
    assert loop in insns["row_total_S0"].within_inames
    for rows, cols in ((1, 1), (3, 4), (5, 2)):
        a = np.arange(rows * cols, dtype=float).reshape(rows, cols) / 3.0
        agrees(squared_rows, fused, {"a": Arr.from_numpy(a), "z": Arr.zeros(rows)})
    # And with the squares computed where the sum reads them, no g at all.
    substituted = fused.substitute("g")
    assert "g" not in substituted.kernel.default_entrypoint.temporary_variables
    a = np.arange(12, dtype=float).reshape(3, 4)
    agrees(squared_rows, substituted, {"a": Arr.from_numpy(a), "z": Arr.zeros(3)})


def test_a_call_between_the_fused_runs_after_their_loop() -> None:
    # front reads the f flux stores, so it runs after the fused loop; back
    # reads h, which front writes, but at cells front does not write. The
    # lowering's instruction dependencies are by array, and loopy found no
    # order for the three (a CycleError) until those were cut back to the
    # dependences the casts were checked against.
    fused = Schedule(between).fuse("flux", "back")
    assert statements(fused) == [("bijective", "decided"), ("monotone", "decided")]
    entry = fused.kernel.default_entrypoint
    insns = {insn.id: insn for insn in entry.instructions}
    assert "front_S0" not in insns["back_S0"].depends_on
    assert "flux_S0" in insns["front_S0"].depends_on
    for size in (1, 2, 5, 8):
        inputs = {"u": Arr.from_numpy(velocity(size)), "rhs": Arr.zeros(size)}
        agrees(between, fused, inputs)


def gather_inputs(size: int) -> dict:
    return {
        "idx": Arr.zeros(size, dtype=np.int64),
        "x": Arr.from_numpy(np.arange(10.0, 10.0 + size)),
        "y": Arr.zeros(size),
    }


def test_a_fusion_past_a_checked_point_is_refused_for_its_flag() -> None:
    # gather's requirement on idx is checked between the two calls, and
    # gather runs only where the check's flag is clear. Fused, gather would
    # read the flag before the check over the cells after it is done: the
    # refusal names the check, its flag and the read. The label names the
    # call's own statement, not the check before it.
    term = reversed_gather.term
    ((flag, _message),) = term.checks
    (check,) = [stmt.id for stmt in term.stmts if stmt.assignee.array == flag]
    for shift in (0, 1, 3):
        with pytest.raises(IllegalCast) as caught:
            Schedule(reversed_gather, sizes={"n": 5}).fuse(
                "reverse_quiet", "gather", shift=shift
            )
        assert caught.value.fact.kind == "monotone"
        (source, _), (sink, _), _sizes = caught.value.witness
        assert (source, sink) == (check, "gather.S0")
        assert f"writes {flag}[0] read by gather.S0[" in str(caught.value)


def test_a_checked_point_fused_with_the_producer_stays_before_the_reads() -> None:
    # The check fused into the producer's loop checks each cell as it is
    # written, and gather still runs after the loop, where the flag is set
    # or not: the compiled program agrees, and stops where the native call
    # is refused.
    term = reversed_gather.term
    ((flag, _message),) = term.checks
    (check,) = [stmt.id for stmt in term.stmts if stmt.assignee.array == flag]
    fused = Schedule(reversed_gather).fuse("reverse_quiet", check)
    assert statements(fused) == [("bijective", "decided"), ("monotone", "decided")]
    for size in (1, 2, 5):
        agrees(reversed_gather, fused, gather_inputs(size))
    term = wrong_gather.term
    ((flag, _message),) = term.checks
    (check,) = [stmt.id for stmt in term.stmts if stmt.assignee.array == flag]
    fused = Schedule(wrong_gather).fuse("number_up", check)
    with pytest.raises(ValueError, match="is 4"):
        wrong_gather(**gather_inputs(4))
    with pytest.raises(ValueError, match="stops before gather"):
        LoopyExecutor().run(fused, **gather_inputs(4))


@pytest.mark.parametrize(
    ("producer", "consumer", "shift", "error", "message"),
    [
        ("nothing", "divergence", 0, ValueError, "names no statement of burgers"),
        ("divergence", "flux", 0, ValueError, "flux comes before divergence"),
        ("flux", "flux.S0", 0, ValueError, "is named on both sides"),
        ("flux", "divergence", True, TypeError, "a shift is a whole number"),
        ("flux", "divergence", (1, 0), ValueError, "takes 1 whole number"),
    ],
)
def test_a_fusion_names_two_sides_in_order(
    producer: str, consumer: str, shift, error: type, message: str
) -> None:
    with pytest.raises(error, match=message):
        Schedule(burgers).fuse(producer, consumer, shift=shift)


def test_a_statement_s_map_moves_every_loop_of_the_step_it_runs_in() -> None:
    # rows_then_cells.S1 runs in the row loop, which S0's map takes, and in
    # its cell loop: a map of the cell loop alone would leave it in a row
    # loop the step replaces.
    term = rows.term
    row = term.stmt("rows_then_cells.S0").inames[0]
    cell = term.stmt("rows_then_cells.S1").inames[1]
    with pytest.raises(ValueError, match=f"S1 runs in {row} as well"):
        Schedule(rows).affine(
            f"{{ rows_then_cells_S0[{row}] -> [q] : q = {row}; "
            f"rows_then_cells_S1[{cell}] -> [q] : q = {cell} }}"
        )


def test_loops_over_two_sizes_fuse_into_one_loop_bounded_by_both() -> None:
    # The union of { [j] : 0 <= j < n } and { [j] : 0 <= j < m } is no one
    # domain, and its hull is bounded only where the sizes are not negative,
    # which they never are: the fused loop runs to n + m, each statement on
    # its own points. Without that, the schedule was decided and loopy could
    # write no loop ("unbounded optimum").
    for shift in (0, 2, -2):
        fused = Schedule(two_sizes).fuse("doubled", "plus_one", shift=shift)
        assert fused.buildable == (True, ""), fused.buildable
        for n, m in ((1, 3), (3, 1), (2, 5), (5, 2), (1, 1)):
            agrees(
                two_sizes,
                fused,
                {
                    "u": Arr.from_numpy(np.arange(n) + 0.5),
                    "w": Arr.from_numpy(np.arange(m) * 0.25),
                    "y": Arr.zeros(m),
                    "z": Arr.zeros(n),
                },
            )


def test_statements_of_one_loop_cannot_be_fused_into_it() -> None:
    fused = Schedule(burgers).fuse("flux", "divergence", shift=1)
    with pytest.raises(ValueError, match="already run in"):
        fused.fuse("flux", "divergence")


def test_a_parallel_loop_after_fusion_is_refused_for_the_edge_it_carries() -> None:
    # In the fused loop, divergence at j reads f[j - 2], which flux stored two
    # steps earlier: on a hardware axis those are two work items. The C
    # target has no hardware axes, but the casts are asked before the target
    # is.
    fused = Schedule(burgers).fuse("flux", "divergence", shift=1)
    loop = fused.order[-1]
    with pytest.raises(IllegalCast, match="on another work item"):
        fused.tag(**{loop: "g.0"})


# }}}


# {{{ substitution


def test_a_substituted_array_is_computed_where_it_is_read() -> None:
    schedule = Schedule(burgers, sizes={"n": 16}).substitute("f")
    assert schedule.substituted == ("f",)
    assert statements(schedule) == [
        ("definedness", "decided"),
        ("bijective", "decided"),
        ("monotone", "decided"),
    ]
    entry = schedule.kernel.default_entrypoint
    assert "f" not in entry.temporary_variables
    assert {insn.id for insn in entry.instructions} == {"divergence_S0"}
    assert entry.substitutions
    for size in (1, 2, 3, 16):
        agrees(burgers, schedule, burgers_inputs(size))


def test_a_fused_array_is_substituted_into_the_fused_loop() -> None:
    schedule = (
        Schedule(burgers).fuse("flux", "divergence", shift=1).substitute("f")
    )
    assert schedule.history == (
        "fuse(flux, divergence, shift=1)",
        "substitute('f')",
    )
    assert "f" not in schedule.kernel.default_entrypoint.temporary_variables
    for size in (1, 4, 11):
        agrees(burgers, schedule, burgers_inputs(size))
    again = schedule.retarget("c-source")
    assert again.history == schedule.history


def test_a_substitution_drops_the_dependences_through_the_array() -> None:
    # Fused and substituted, nothing passes from one step of the loop to the
    # next, so it may go on a hardware axis; fused alone, it may not (above).
    schedule = Schedule(burgers).fuse("flux", "divergence", shift=1).substitute("f")
    tagged = schedule.tag(**{schedule.order[-1]: "g.0"})
    monotone = [fact for fact in tagged.facts() if fact.kind == "monotone"][-1]
    assert monotone.status is Status.DECIDED
    assert "within one work item" in monotone.statement
    # The target is the C target, which has no hardware axes.
    assert not tagged.buildable[0]


def test_a_read_of_a_cell_the_producer_does_not_store_is_refused() -> None:
    # The program reads the zeros at f's boundary, which its definedness fact
    # decides; computed where it is read, f would have no zeros to read, so
    # the substitution's own fact keeps the stricter claim and is refuted.
    with pytest.raises(IllegalCast) as caught:
        Schedule(padded).substitute("f")
    fact = caught.value.fact
    assert (fact.kind, fact.status) == ("definedness", Status.REFUTED)
    assert fact.statement == (
        "every cell of f that padded reads, interior_flux.S0 has stored by the "
        "time it is read"
    )
    message = str(caught.value)
    assert message.startswith("substitute('f') illegal: shifted.S0 reads f[")
    assert "which interior_flux.S0 does not store" in message
    assert "would see the zeros" in message
    assert len(fact.provenance["witness"]) == 1
    (program_fact,) = definedness(padded)
    assert IslOracle().establish(program_fact).status is Status.DECIDED
    # The same for h, whose back half back reads and front leaves zero.
    with pytest.raises(IllegalCast, match="back.S0 reads h\\["):
        Schedule(between).substitute("h")


def test_a_read_isl_cannot_list_refuses_the_substitution_undecided() -> None:
    # gather reads f[idx[i]], and interior_flux stores the inside of f: which
    # cells idx names is not known, so the substitution is refused, and its
    # definedness fact is assumed with the reason, not refuted.
    with pytest.raises(IllegalCast) as caught:
        Schedule(gathered_inside).substitute("f")
    fact = caught.value.fact
    assert (fact.kind, fact.status) == ("definedness", Status.ASSUMED)
    assert fact.decided_by is None
    assert "whose index is not affine" in fact.provenance["reason"]
    assert str(caught.value) == fact.provenance["reason"]


def test_a_read_before_the_producer_stores_the_cell_is_refused() -> None:
    with pytest.raises(IllegalCast) as caught:
        Schedule(early, sizes={"n": 4}).substitute("f")
    assert caught.value.fact.kind == "definedness"
    message = str(caught.value)
    assert "instance shifted.S0[" in message
    assert "before flux.S0[" in message
    assert "stores it (at n=4, as hinted)" in message


def test_a_write_between_the_producer_and_a_read_is_refused() -> None:
    # bump writes u after flux read it and before divergence reads f: the
    # value computed again at the read would be the bumped one.
    with pytest.raises(IllegalCast) as caught:
        Schedule(bumped, sizes={"n": 8}).substitute("f")
    assert caught.value.fact.kind == "monotone"
    message = str(caught.value)
    assert message.startswith("substitute('f') illegal: instance divergence.S0[")
    assert " reads u[" in message
    assert "overwritten by bump.S0[" in message


def test_a_later_write_of_what_the_producer_read_waits_for_the_reads() -> None:
    # bump writes u after divergence has read f; computed again at the
    # read, f reads u, so bump has to wait for divergence, which nothing in
    # the lowered kernel said: bump depended on flux, which is gone.
    schedule = Schedule(bumped_after).substitute("f")
    insns = {
        insn.id: insn for insn in schedule.kernel.default_entrypoint.instructions
    }
    assert "divergence_S0" in insns["bump_S0"].depends_on
    for size in (1, 3, 8):
        agrees(bumped_after, schedule, burgers_inputs(size))
    # Fused one step behind the divergence, bump writes u[k] in the step in
    # which the divergence reads it to compute f[k]: the divergence has to
    # come first in the step, and the kernel says so.
    fused = Schedule(bumped_after).fuse("divergence", "bump", shift=1)
    with pytest.raises(IllegalCast):
        Schedule(bumped_after).fuse("divergence", "bump").substitute("f")
    substituted = fused.substitute("f")
    insns = {
        insn.id: insn for insn in substituted.kernel.default_entrypoint.instructions
    }
    assert "divergence_S0" in insns["bump_S0"].depends_on
    for size in (1, 2, 6):
        agrees(bumped_after, substituted, burgers_inputs(size))


def test_an_in_place_consumer_cannot_have_the_array_computed_again() -> None:
    # update writes u[i - 1] one step before it reads f[i - 1], which is
    # 0.5 * u[i - 1]**2 of the u before the update.
    with pytest.raises(IllegalCast, match="overwritten by update.S0"):
        Schedule(stepped).substitute("f")
    # Stored, the same program fuses and runs.
    agrees(
        stepped,
        Schedule(stepped).fuse("flux", "update", shift=1),
        {"u": Arr.from_numpy(velocity(6))},
    )


def test_a_read_through_an_index_array_is_substituted() -> None:
    schedule = Schedule(gathered).substitute("f")
    for size in (1, 4, 7):
        idx = np.random.default_rng(size).permutation(size).astype(np.int64)
        agrees(
            gathered,
            schedule,
            {
                "u": Arr.from_numpy(velocity(size)),
                "idx": Arr.from_numpy(idx),
                "y": Arr.zeros(size),
            },
        )


@pytest.mark.parametrize(
    ("prog", "array", "message"),
    [
        (burgers, "u", "u is a parameter of burgers, which its caller passes"),
        (burgers, "g", "g is no array of burgers"),
        (twice, "f", "f is written by flux.S0, flux@2.S0"),
        (halved, "f", r"stores f\[j // 2\]|at its own loop variables"),
        (summed, "f", "stores a sum"),
        (unread, "f", "nothing in unread reads f"),
    ],
)
def test_what_a_substitution_cannot_take_is_named(
    prog, array: str, message: str
) -> None:
    with pytest.raises(ValueError, match=message):
        Schedule(prog).substitute(array)


def test_a_substitution_through_two_arrays_is_refused_either_way() -> None:
    # The dependences carried over are those of what the producer reads, and
    # through two arrays that is what the first producer reads, which the
    # second step does not see: refused, whichever comes first.
    with pytest.raises(ValueError, match="doubled.S0 reads f, which is computed"):
        Schedule(chained).substitute("f").substitute("g")
    with pytest.raises(ValueError, match="doubled.S0 reads f and runs no more"):
        Schedule(chained).substitute("g").substitute("f")
    for array in ("f", "g"):
        agrees(chained, Schedule(chained).substitute(array), burgers_inputs(7))


def test_a_statement_taken_out_leaves_no_loop_to_name() -> None:
    schedule = Schedule(burgers).substitute("f")
    term = burgers.term
    gone = {term.stmt("flux.S0").inames[0], term.stmt("f.zeros").inames[0]}
    assert not gone & set(schedule.order)
    assert schedule.order == term.stmt("divergence.S0").inames
    with pytest.raises(ValueError, match="is not an iname"):
        schedule.tag(**{min(gone): "g.0"})
    with pytest.raises(ValueError, match="runs no more"):
        schedule.fuse("flux", "divergence")
    loop = schedule.order[0]
    split = schedule.split(loop, 4)
    assert [fact.status.value for fact in split.facts()][-2:] == ["decided"] * 2
    agrees(burgers, split, burgers_inputs(10))


def test_an_array_is_substituted_once() -> None:
    schedule = Schedule(burgers).substitute("f")
    with pytest.raises(ValueError, match="substituted already"):
        schedule.substitute("f")


def test_a_substituted_value_is_converted_as_storing_it_converted_it() -> None:
    # tenth stores a tenth of u in single precision, which rounds it. The
    # value computed where it is read has to be rounded the same way, or
    # the substituted program computes something the stored one does not:
    # bit for bit the same as the kernel that stores f.
    rng = np.random.default_rng(3)
    u = rng.normal(size=9) * 1e3 + 1.0 / 3.0
    stored = LoopyExecutor().run(
        Schedule(single),
        u=Arr.from_numpy(u.copy()),
        t=Arr.from_numpy(np.zeros(9, dtype=np.float32)),
        y=Arr.zeros(9),
    )["y"]
    computed = LoopyExecutor().run(
        Schedule(single).substitute("f"),
        u=Arr.from_numpy(u.copy()),
        t=Arr.from_numpy(np.zeros(9, dtype=np.float32)),
        y=Arr.zeros(9),
    )["y"]
    assert np.array_equal(stored, computed)
    assert not np.array_equal(computed, u * 0.1 * 10.0)
    # A Real stored in a Nat cell is truncated, and the result is exact.
    schedule = Schedule(truncated).substitute("f")
    for size in (1, 4, 7):
        agrees(
            truncated,
            schedule,
            {
                "u": Arr.from_numpy(np.arange(size) * 1.5 + 3.0),
                "y": Arr.from_numpy(np.zeros(size, dtype=np.int64)),
            },
        )


def permuted_inputs(size: int, **arrays: np.ndarray) -> dict:
    """A permutation ``t`` of ``size`` cells, with ``x`` and ``y`` beside it."""
    t = np.random.default_rng(size).permutation(size).astype(np.int64)
    return {
        "t": t,
        "x": np.arange(size, dtype=np.float64) + 0.5,
        "y": np.zeros(size),
        **arrays,
    }


def substituted_agrees(prog, array: str, inputs: dict) -> str:
    """The substituted kernel's code, after its run agreed with two others.

    With the native program, by the differential fact, and bit for bit with
    the compiled program that stores ``array``.
    """
    schedule = Schedule(prog).substitute(array)
    assert [fact.status.value for fact in schedule.facts()][:3] == ["decided"] * 3
    assert schedule.buildable == (True, "")
    agrees(prog, schedule, {name: value.copy() for name, value in inputs.items()})
    stored = LoopyExecutor().run(
        Schedule(prog), **{name: value.copy() for name, value in inputs.items()}
    )
    computed = LoopyExecutor().run(
        schedule, **{name: value.copy() for name, value in inputs.items()}
    )
    for name, value in stored.items():
        assert np.array_equal(value, computed[name]), name
    return emit_code(schedule)


def test_an_index_array_computed_with_a_conversion_is_substituted() -> None:
    # rotate computes (t[j] + 1) % n in 64 bits, with the 32-bit t[j] cast
    # (#101), and p is stored in the 64 bits of an index array a checked
    # point reads (#128), so the store converts nothing. The cast is in the
    # value itself, and substituting p into x[p[i]] carries it into the
    # subscript, which loopy declined to read into isl with an error (#145);
    # loopy generates the subscript as written now. Leaving the cast out
    # would compute (t[j] + 1) in 32 bits, which is exact here and is not
    # for (t[j] * 3) % n, so it stays.
    for size in (1, 5, 8):
        code = substituted_agrees(rotated, "p", permuted_inputs(size))
    assert "y[i] = x[loopty_mod_int64((int64_t) (t[i]) + 1, (int64_t) (n))]" in code


def test_an_index_widened_by_its_store_is_read_unconverted_in_a_subscript() -> None:
    # A Fin[m] entry is read in 32 bits (#101), and an index array a checked
    # point reads between two calls is stored in 64 (#128), as a Nat array
    # is: t[j] is widened by p's store, and by q's. A widening keeps the
    # value, and is the last thing done to it, so a read in a subscript
    # reads the value inside it (#145); the checked point, which reads it as
    # a value, keeps the conversion.
    for prog, array, read in (
        (scattered, "p", "y[t[i]] = x[i];"),
        (widened, "q", "y[i] = x[t[i]];"),
    ):
        for size in (1, 4, 9):
            code = substituted_agrees(prog, array, permuted_inputs(size))
        assert read in code
        assert "(int64_t) (t[" in code
    # An index computed in 32 bits from the loop variable and a size, and
    # stored in 64, is affine with the widening left out: loopy reads it
    # into isl, checks its bounds and writes it as it simplifies it.
    for size in (1, 6):
        code = substituted_agrees(
            reversed_whole, "q", {"x": np.arange(size) + 0.5, "y": np.zeros(size)}
        )
    assert "y[i] = x[-1 + -1 * i + n];" in code


def test_a_narrowing_in_a_subscript_is_kept() -> None:
    # p is stored in sixteen bits, which converts the 32-bit value it is
    # computed from, and is read in the subscript of y: the substituted read
    # converts it as the store did, and loopy generates the subscript with
    # the cast (#145), as the kernel that stores p computes it.
    for size in (1, 3, 7):
        code = substituted_agrees(
            narrowed, "p", permuted_inputs(size, w=np.arange(size, dtype=np.int16))
        )
    assert "y[(int16_t) (t[i])] = x[i] + w[i];" in code


# }}}
