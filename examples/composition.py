"""Two kernels composed by a program, and lowered as one loopy kernel.

A Burgers right-hand side in two steps: the pointwise flux ``f = u**2 / 2``,
then its centred divergence ``rhs[i] = -(f[i + 1] - f[i - 1]) / 2`` inside the
boundary. Each step is a kernel, and the program that runs one after the other
is three lines::

    f = Arr.zeros_like(u)
    flux(u, f)
    divergence(f, rhs)

Run this file three ways.

``python examples/composition.py``
    Runs the program natively and checks it against the same arithmetic
    written with numpy slices. Then prints the program's term, one statement
    per line, the one kernel loopy generates for it, and the comparison of the
    compiled program with the native one.

``lanky check examples/composition.py``
    The two kernels' obligations. The program adds none: neither kernel states
    a postcondition for it to restate.

``loopty run examples/composition.py``
    Each kernel alone, and the program as one kernel, compiled and compared
    with the native run.

What is worth reading here
--------------------------

*The intermediate is the program's, not an argument.* Lowered one kernel at a
time, ``f`` is an output of ``flux`` and an input of ``divergence``, a public
array of both, and the two lowerings disagree about its role, which loopy's
``fuse_kernels`` will not accept. A program's term is one term: ``f`` is an
array the program made, so it is a temporary of the one kernel, declared inside
it, zeroed where the program made it, and passed by nobody.

*The edge between the kernels is in the footprints.* Nothing declares that
``divergence`` needs ``flux``. The ``f`` one writes and the other reads is one
array of the term, so the dependence is found as it is between two statements
of one kernel, and it orders the two loops of the generated code.

*This is the reference for fusion, not fusion.* The generated code runs the two
loops one after the other, as the program does. Fusing them into one loop is a
cast over this term, which a checker will have to accept against the
dependence above, and it is not done here.
"""

from __future__ import annotations

import numpy as np
from lanky.prelude import Real

from loopty import Arr, Fin, Schedule, kernel, program, when

#: Sixteen points: enough for the boundary to be visible, small enough that the
#: three commands each take about a second.
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
    """Run the program natively, then show its term, its kernel and its run."""
    from loopty.executor import LoopyExecutor, emit_code

    u = velocity()
    rhs = Arr.zeros(N)
    burgers_rhs(Arr.from_numpy(u.copy()), rhs)
    native = np.allclose(rhs.numpy(), reference(u))
    print(f"native: the program agrees with the numpy slices: {native}")

    term = burgers_rhs.term
    print()
    print(f"the term of {term.name}({', '.join(term.param_names)}):")
    for name, typ in term.temporaries:
        print(f"  temporary {name}: {len(typ.axes)} axis, element {typ.dtype}")
    for stmt in term.stmts:
        loops = ", ".join(stmt.inames)
        print(f"  {stmt.id:14} over {loops:3} from {stmt.where}")

    print()
    print(emit_code(burgers_rhs))
    print()

    inputs = example_inputs()["burgers_rhs"]
    fact = LoopyExecutor().differential(burgers_rhs, Schedule(burgers_rhs), inputs)
    for name, detail in fact.provenance["outputs"].items():
        print(
            f"  {name}: difference {detail['difference']:.3g} within "
            f"{detail['tolerance']:.3g} ({detail['exactness']}) -> "
            f"{fact.status.value}"
        )
    return 0 if native and fact.status.value == "tested" else 1


if __name__ == "__main__":
    raise SystemExit(main())
