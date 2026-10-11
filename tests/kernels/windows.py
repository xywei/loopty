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


@program
def bumped_after(u, rhs):
    """The flux's input written after the divergence has read the flux."""
    f = Arr.zeros_like(u)
    flux(u, f)
    divergence(f, rhs)
    bump(u)


SIZES = {"n": 16}

#: The divergence one step behind the flux, in the flux's loop.
fused = Schedule(burgers, sizes=SIZES).fuse("flux", "divergence", shift=1)

#: ``f`` in the three cells live at once, ``f[j]`` in slot ``j % 3``.
windowed = fused.contract("f")

#: The same window, a shift register: every access at a constant index.
registers = fused.contract("f", registers=True)

#: A split of the loop after the window, which keeps every slot's order.
split_after = windowed.split("j", 4)

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
