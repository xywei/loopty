"""Two kernels fused into one loop, and the array between them never stored.

The Burgers right-hand side of ``composition.py``, the flux ``f = u**2 / 2``
and then its centred divergence, run by a program that keeps the flux in an
array of its own::

    f = Arr.zeros_like(u)
    flux(u, f)
    divergence(f, rhs)

``composition.py`` lowers that program as one kernel with two loops, one after
the other. Here the loops become one, and then ``f`` is not stored at all:
two schedule steps, each checked before it is applied.

Run this file three ways.

``python examples/fusion.py``
    Runs the program natively and checks it against numpy slices. Then asks
    for the fusion with no shift, which isl refuses with a pair of instances,
    and prints the fused schedule and the substituted one: their facts, the
    code loopy generates for each, and the comparison of each compiled run
    with the native one.

``lanky check examples/fusion.py``
    The kernels' obligations and the program's, among them the definedness
    of ``f``: every cell ``divergence`` reads, ``flux`` stored before it.

``loopty run examples/fusion.py``
    Every schedule of the file compiled and compared with the native run,
    with its cast facts in the ledger, and each kernel on its own.

What is worth reading here
--------------------------

*A fusion is a map per statement.* ``fuse("flux", "divergence", shift=1)``
is ``affine`` with the map it builds, ``{ flux_S0[j] -> [j]; divergence_S0[i]
-> [j] : j = i + 1 }``: one loop, the flux first in each step, the divergence
one step behind. isl decides it as it decides every cast, over the
dependences between the two calls as well as within each. Without the shift,
``divergence`` at ``i`` would read ``f[i + 1]`` a step before ``flux`` stores
it, and the refusal names that pair, and the shift that is accepted.

*An intermediate is stored only if it has to be.* Once the flux is computed
in the loop that reads it, nothing needs it kept: ``substitute("f")`` computes
``0.5 * u[j] * u[j]`` at each read instead (loopy's ``assignment_to_subst``),
and the temporary goes, with the loop that zeroed it. That is legal because
every cell ``divergence`` reads, ``flux`` stored before the read (the
``definedness`` fact), and nothing writes ``u`` in between (the ``monotone``
fact over the program as it now runs).

*Each compiled run is compared with the program's own body.* The native run
is the reference both schedules are tested against, bit for bit where the
types ask for it.
"""

from __future__ import annotations

import numpy as np
from lanky.prelude import Real

from loopty import Arr, Fin, IllegalCast, Schedule, kernel, program, when

#: Sixteen points, as in ``composition.py``.
N = 16


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


@program
def burgers_rhs(u, rhs):
    """The flux into an array of the program's own, then its divergence."""
    f = Arr.zeros_like(u)
    flux(u, f)
    divergence(f, rhs)


#: The divergence one step behind the flux, in the flux's loop.
fused = Schedule(burgers_rhs, sizes={"n": N}).fuse("flux", "divergence", shift=1)

#: The same loop, with the flux computed where it is read and never stored.
substituted = fused.substitute("f")


def velocity(n: int = N) -> np.ndarray:
    """A smooth periodic velocity with some structure."""
    x = np.linspace(0.0, 2.0 * np.pi, n, endpoint=False)
    return np.sin(x) + 0.2 * np.sin(3.0 * x)


def reference(u: np.ndarray) -> np.ndarray:
    """The same right-hand side, written with numpy slices."""
    f = 0.5 * u * u
    rhs = np.zeros_like(u)
    rhs[1:-1] = -(f[2:] - f[:-2]) / 2
    return rhs


def example_inputs() -> dict:
    """The inputs ``loopty run`` gives each kernel, and the program."""
    u = velocity()
    return {
        "flux": {"u": Arr.from_numpy(u.copy()), "f": Arr.zeros(N)},
        "divergence": {"f": Arr.from_numpy(0.5 * u * u), "rhs": Arr.zeros(N)},
        "burgers_rhs": {"u": Arr.from_numpy(u.copy()), "rhs": Arr.zeros(N)},
    }


def main() -> int:
    """Run the program natively, then the refused fusion and the two accepted."""
    from loopty.executor import LoopyExecutor, emit_code

    u = velocity()
    rhs = Arr.zeros(N)
    burgers_rhs(Arr.from_numpy(u.copy()), rhs)
    native = np.allclose(rhs.numpy(), reference(u))
    print(f"native: the program agrees with the numpy slices: {native}")
    print()

    try:
        Schedule(burgers_rhs, sizes={"n": N}).fuse("flux", "divergence")
    except IllegalCast as exc:
        print(f"refused: {exc}")
    print()

    ok = native
    for schedule in (fused, substituted):
        print(f"accepted: {schedule!r}")
        for fact in schedule.facts():
            print(f"  {fact.status.value:8} {fact.decided_by:4} {fact.statement}")
        print()
        print(emit_code(schedule))
        print()
        inputs = example_inputs()["burgers_rhs"]
        fact = LoopyExecutor().differential(burgers_rhs, schedule, inputs)
        for name, detail in fact.provenance["outputs"].items():
            print(
                f"  {name}: difference {detail['difference']:.3g} within "
                f"{detail['tolerance']:.3g} ({detail['exactness']}) -> "
                f"{fact.status.value}"
            )
        print()
        ok = ok and fact.status.value == "tested"
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
