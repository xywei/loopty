"""Windows and substitutions: an array a program makes, stored less (#132, #187).

The Burgers right-hand side, the flux into an array of the program's own and
its centred divergence, fused one step behind, and then:

* kept in a window of the three cells live at once, with modulo addressing
  and with a shift register, and split after the window;
* computed where it is read, in a program that writes what the flux read
  after the divergence, so that the reads the substitution computes again
  are ordered against that write (the classes Lean states for #187).

The ``rejected*`` functions ask for what is refused: a window of two cells,
which a read sees overwritten, and a step after the window that runs the
divergence a step later, past the slot it reads.

And windows the review asked about: a second statement that reads further
back (a window of four), the fused loop run backwards (the window taken
along the reversed order), and the flux of every row of a 2-D array, the
window's loop inside the rows' (a shift register shifted in each row).

And a window read by an update: the divergence added into the node each
point names, ``y[node[i]] += ...``, a scatter-add, whose updates are made
atomic after the window, and the loop split after both.
"""

from __future__ import annotations

from lanky.prelude import Real

from loopty import Arr, Fin, Schedule, kernel, program, when


@kernel
def flux(u: Arr[Fin[n], Real], f: Arr[Fin[n], Real]):
    """The pointwise Burgers flux."""
    for j in u.dom:
        f[j] = 0.5 * u[j] * u[j]


@kernel
def divergence(f: Arr[Fin[n], Real], rhs: Arr[Fin[n], Real]):
    """The centred divergence of the flux, at the interior points."""
    for i in rhs.dom:
        with when((i > 0) & (i + 1 < rhs.dom.size)):
            rhs[i] = -(f[i + 1] - f[i - 1]) / 2


@kernel
def bump(u: Arr[Fin[n], Real]):
    """Add one to every cell, in place."""
    for k in u.dom:
        u[k] = u[k] + 1.0


@program
def burgers(u, rhs):
    """The flux into an array of the program's own, then its divergence."""
    f = Arr.zeros_like(u)
    flux(u, f)
    divergence(f, rhs)


@kernel
def divergence_and_lag(
    f: Arr[Fin[n], Real], rhs: Arr[Fin[n], Real], z: Arr[Fin[n], Real]
):
    """The divergence, and a second statement that reads two cells back."""
    for i in rhs.dom:
        with when((i > 0) & (i + 1 < rhs.dom.size)):
            rhs[i] = -(f[i + 1] - f[i - 1]) / 2
        with when(i >= 2):
            z[i] = f[i - 2] + 0.25 * f[i]


@kernel
def row_flux(u: Arr[Fin[m], Fin[n], Real], f: Arr[Fin[m], Fin[n], Real]):
    """The flux of every row."""
    for e in u.dom:
        for j in u.dom[e]:
            f[e, j] = 0.5 * u[e, j] * u[e, j]


@kernel
def row_divergence(f: Arr[Fin[m], Fin[n], Real], rhs: Arr[Fin[m], Fin[n], Real]):
    """The divergence of every row, at its interior points."""
    for e in rhs.dom:
        for i in rhs.dom[e]:
            with when((i > 0) & (i + 1 < rhs.dom[e].size)):
                rhs[e, i] = -(f[e, i + 1] - f[e, i - 1]) / 2


@kernel
def assembled_divergence(
    f: Arr[Fin[n], Real], node: Arr[Fin[n], Fin[m]], y: Arr[Fin[m], Real]
):
    """The divergence at the interior points, added into the node each names."""
    for i in f.dom:
        with when((i > 0) & (i + 1 < f.dom.size)):
            y[node[i]] += -(f[i + 1] - f[i - 1]) / 2


@program
def two_readers(u, rhs, z):
    """The flux, read by the divergence and by a statement two cells back."""
    f = Arr.zeros_like(u)
    flux(u, f)
    divergence_and_lag(f, rhs, z)


@program
def rows(u, rhs):
    """The Burgers right-hand side of every row."""
    f = Arr.zeros_like(u)
    row_flux(u, f)
    row_divergence(f, rhs)


@program
def bumped_after(u, rhs):
    """The flux's input written after the divergence has read the flux."""
    f = Arr.zeros_like(u)
    flux(u, f)
    divergence(f, rhs)
    bump(u)


@program
def burgers_assembled(u, node, y):
    """The flux, and its divergence added into the nodes: a scatter-add."""
    f = Arr.zeros_like(u)
    flux(u, f)
    assembled_divergence(f, node, y)


SIZES = {"n": 16}

#: The divergence one step behind the flux, in the flux's loop.
fused = Schedule(burgers, sizes=SIZES).fuse("flux", "divergence", shift=1)

#: ``f`` in the three cells live at once, ``f[j]`` in slot ``j % 3``.
windowed = fused.contract("f")

#: The same window, a shift register: every access at a constant index.
registers = fused.contract("f", registers=True)

#: A split of the loop after the window, which keeps every slot's order.
split_after = windowed.split("j", 4)

#: A second reader of ``f[j - 3]`` at step ``j``: a window of four cells.
deeper = (
    Schedule(two_readers, sizes=SIZES)
    .fuse("flux", "divergence_and_lag", shift=1)
    .contract("f")
)

#: The fused loop run backwards, and the window taken along that order.
REVERSED = "{ flux_S0[j] -> [k] : k = -j; divergence_S0[j] -> [k] : k = 2 - j }"
reversed_window = fused.affine(REVERSED).contract("f")

#: Each row's flux in a shift register, shifted at each step of each row.
row_registers = (
    Schedule(rows, sizes={"m": 3, "n": 8})
    .fuse("row_flux", "row_divergence", shift=(0, 1))
    .contract("f", registers=True)
)

#: The divergence added into the nodes, one step behind the flux: ``f`` in
#: the window of three cells, the updates of ``y`` atomic, and a split after
#: both, whose cast keeps every slot's order and no order of two updates.
assembled = (
    Schedule(burgers_assembled, sizes={"n": 16, "m": 4})
    .fuse("flux", "assembled_divergence", shift=1)
    .contract("f")
    .atomic("y")
    .split("j", 4)
)

#: The flux computed where it is read, before ``bump`` writes what it read.
substituted = (
    Schedule(bumped_after, sizes=SIZES)
    .fuse("flux", "divergence", shift=1)
    .substitute("f")
)


def rejected_window() -> str:
    """A window of two cells: the read of ``f[j - 2]`` sees its slot overwritten."""
    from loopty.schedule import IllegalCast

    try:
        fused.contract("f", lambda j: j % 2)
    except IllegalCast as refused:
        return str(refused)
    raise AssertionError("a window of two cells is too small")


def rejected_after_window() -> str:
    """The divergence a step later, past the slot it reads, refused by the window."""
    from loopty.schedule import IllegalCast

    try:
        windowed.affine(
            "{ flux_S0[j] -> [k] : k = j; divergence_S0[j] -> [k] : k = j + 1 }"
        )
    except IllegalCast as refused:
        return str(refused)
    raise AssertionError("the window of three cells is too small one step later")


def rejected_short_window() -> str:
    """A window of three cells, one short of the reader of ``f[j - 3]``."""
    from loopty.schedule import IllegalCast

    try:
        (
            Schedule(two_readers, sizes=SIZES)
            .fuse("flux", "divergence_and_lag", shift=1)
            .contract("f", lambda j: j % 3)
        )
    except IllegalCast as refused:
        return str(refused)
    raise AssertionError("the second reader reads a cell a window of three lost")


def rejected_reversal_after_window() -> str:
    """The loop run backwards after the window: each slot overwritten first."""
    from loopty.schedule import IllegalCast

    try:
        windowed.affine(REVERSED)
    except IllegalCast as refused:
        return str(refused)
    raise AssertionError("a window keeps the order it was taken in")
