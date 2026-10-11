"""Assembly: firedrake's P1 mass action, cell by cell, added into the nodes.

PyOP2 runs an element kernel over the cells of a mesh, and around it the
assembly loop: per cell, gather the field at the cell's three nodes, run the
kernel, and add the cell's three values into the nodes,
``y[cell[c, k]] += yl[c, k]``. This is that loop, with ``mass_ffc`` of
firedrake's PyOP2 tests as the element kernel (``examples/quadrature.py`` is
the kernel alone). Two cells that share a node add into one cell of ``y``,
which is what an assembly is, and what makes the loop over cells a race on a
device unless the additions are atomic.

Run this file three ways.

``python examples/assembly.py``
    Runs the kernel natively on the two triangles of PyOP2's test mesh, where
    ``y`` has to be PyOP2's expected matrix applied to the field, and on a mesh
    of the unit square. Then PyOP2's GPU schedule, the cells split by 32 onto
    groups and work items: refused, with the two cells of PyOP2's mesh that
    share a node; and accepted once the updates of ``y`` are atomic. Then Sun
    et al.'s, the cells four at a time, compiled and compared with the native
    run.

``lanky check examples/assembly.py``
    The kernel's obligations: the gathers and the scatter through the cell map
    in bounds by type, and the accumulation into ``y`` approx, which is what
    an update owes in place of distinct writes.

``loopty run examples/assembly.py``
    Sun et al.'s schedule compiled and compared with the native run;
    ``--target opencl`` runs it on a device, each update a compare-and-swap.

What is worth reading here
--------------------------

*An update is not a write.* ``y[cell[c, k]] += yl[c, k]`` adds to the cell
it writes a value that reads nothing else of ``y``: two such updates of one
cell give the same sum whichever runs first, to the class of the sum, which is
``approx`` for ``Real``. So the kernel owes no distinct writes for it, and a
schedule may reorder the updates; it only may not run two of one cell at the
same time, which loses one of them.

*The race is named by the mesh.* The cell of ``y`` an update writes is read
from the cell map, which isl cannot follow, so isl takes any two updates to
touch one cell. The refusal is not isl's pair: it is two updates the example
inputs make of one node, ``S1[c=0, k=2]`` and ``S1[c=1, k=1]`` on node 3,
which is what the two triangles share. loopy itself warns of no race here,
since the subscript names the cell loop; PyOP2 makes every update atomic.

*Atomic is a step.* ``atomic("y")`` makes the updates loopy's
``AtomicUpdate`` on OpenCL, a compare-and-swap loop, after which the split is
accepted and the sum is compared at ``reassoc`` or weaker. On the C target,
which runs in one thread, the code is unchanged.
"""

from __future__ import annotations

import numpy as np
from lanky.prelude import Real

from loopty import Arr, Fin, Schedule, kernel, reduce_sum, table

#: The 3-point rule on the reference triangle and the P1 basis at its points.
W = table([1 / 6, 1 / 6, 1 / 6])
PHI = table([[2 / 3, 1 / 6, 1 / 6], [1 / 6, 2 / 3, 1 / 6], [1 / 6, 1 / 6, 2 / 3]])


@kernel
def mass_action(
    cell: Arr[Fin[nc], Fin[3], Fin[nv]],
    X: Arr[Fin[nv], Real],
    Y: Arr[Fin[nv], Real],
    u: Arr[Fin[nv], Real],
    yl: Arr[Fin[nc], Fin[3], Real],
    y: Arr[Fin[nv], Real],
):
    """The P1 mass matrix applied to ``u``, assembled into ``y`` cell by cell."""
    for c in yl.dom:
        for j in yl.dom[c]:
            dx1 = X[cell[c, 1]] - X[cell[c, 0]]
            dy1 = Y[cell[c, 1]] - Y[cell[c, 0]]
            dx2 = X[cell[c, 2]] - X[cell[c, 0]]
            dy2 = Y[cell[c, 2]] - Y[cell[c, 0]]
            yl[c, j] = abs(dx1 * dy2 - dx2 * dy1) * reduce_sum(
                W[q] * PHI[q, j] * PHI[q, k] * u[cell[c, k]]
                for q in W.dom
                for k in PHI.dom[q]
            )
        for k in yl.dom[c]:
            y[cell[c, k]] += yl[c, k]


# {{{ meshes


def pyop2_mesh() -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """The two triangles of PyOP2's ``test_matrices.py``: nodes and cell map."""
    xs = np.array([0.0, 2.0, 1.0, 0.0])
    ys = np.array([0.0, 0.0, 1.0, 1.5])
    cells = np.array([[0, 1, 3], [2, 3, 1]])
    return xs, ys, cells


#: PyOP2's ``expected_matrix``: the assembled P1 mass matrix of its mesh, to
#: the six digits the test writes.
EXPECTED = np.array(
    [
        [0.25, 0.125, 0.0, 0.125],
        [0.125, 0.291667, 0.0208333, 0.145833],
        [0.0, 0.0208333, 0.0416667, 0.0208333],
        [0.125, 0.145833, 0.0208333, 0.291667],
    ]
)


def square_mesh(n: int = 4) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """The unit square in ``n`` by ``n`` squares, each cut into two triangles."""
    grid = np.linspace(0.0, 1.0, n + 1)
    xs, ys = (axis.ravel() for axis in np.meshgrid(grid, grid, indexing="ij"))
    cells = []
    for i in range(n):
        for j in range(n):
            a, b = i * (n + 1) + j, (i + 1) * (n + 1) + j
            cells += [[a, b, b + 1], [a, b + 1, a + 1]]
    return xs, ys, np.array(cells)


def inputs_on(mesh: tuple[np.ndarray, np.ndarray, np.ndarray]) -> dict:
    """The kernel's arguments on a mesh, with a smooth field on its nodes."""
    xs, ys, cells = mesh
    return {
        "cell": Arr.from_numpy(cells.astype(np.int64)),
        "X": Arr.from_numpy(xs),
        "Y": Arr.from_numpy(ys),
        "u": Arr.from_numpy(1.0 + xs * ys),
        "yl": Arr.zeros((len(cells), 3)),
        "y": Arr.zeros(len(xs)),
    }


def assembled(mesh: tuple[np.ndarray, np.ndarray, np.ndarray]) -> np.ndarray:
    """``y`` in closed form: each cell's ``|detJ| (u_j + sum_k u_k) / 24``, added."""
    xs, ys, cells = mesh
    u = 1.0 + xs * ys
    x, y, local = xs[cells], ys[cells], u[cells]
    det = (x[:, 1] - x[:, 0]) * (y[:, 2] - y[:, 0]) - (x[:, 2] - x[:, 0]) * (
        y[:, 1] - y[:, 0]
    )
    element = np.abs(det)[:, None] * (local + local.sum(axis=1)[:, None]) / 24
    out = np.zeros(len(xs))
    np.add.at(out, cells, element)
    return out


# }}}


def kulkarni(atomic: bool, target: str = "c") -> Schedule:
    """PyOP2's GPU schedule (Kulkarni): the cells split by 32 onto ``g.0``/``l.0``.

    Without ``atomic`` the split's tag is refused; with it, accepted. On the C
    target, which has no groups or work items, the accepted schedule cannot
    be built (its ``buildable`` fact says so); ``target="opencl"`` runs it.
    """
    schedule = Schedule(mass_action, target=target)
    if atomic:
        schedule = schedule.atomic("y")
    return schedule.split("c", 32).tag(c_outer="g.0", c_inner="l.0")


def rejected_split() -> tuple[str, tuple]:
    """PyOP2's GPU schedule without atomics: the message and the witness."""
    from loopty.schedule import IllegalCast

    try:
        kulkarni(atomic=False)
    except IllegalCast as refused:
        return str(refused), refused.witness
    raise AssertionError("the cells of a mesh sharing a node race without atomics")


#: Sun et al.'s schedule: the cells four at a time, which ``ilp`` leaves in no
#: order, so the updates of ``y`` are atomic first. What ``loopty run``
#: compiles and compares.
vectorized = (
    Schedule(mass_action, target="c")
    .atomic("y")
    .split("c", 4)
    .tag(c_inner="ilp")
    .example(**inputs_on(square_mesh()))
)


def example_inputs() -> dict:
    """The inputs ``loopty run`` gives the kernel: PyOP2's two triangles."""
    return inputs_on(pyop2_mesh())


def main() -> int:
    """Run natively on both meshes, then the refused and the accepted schedules."""
    from loopty.executor import LoopyExecutor

    ok = True
    for label, mesh in (
        ("PyOP2's two triangles", pyop2_mesh()),
        ("a square", square_mesh()),
    ):
        data = inputs_on(mesh)
        mass_action(**data)
        y = data["y"].numpy()
        agrees = bool(np.allclose(y, assembled(mesh)))
        if label.startswith("PyOP2"):
            u = data["u"].numpy()
            agrees = agrees and bool(np.allclose(y, EXPECTED @ u, atol=1e-5))
        ok = ok and agrees
        cells = len(mesh[2])
        print(f"{label}: {cells} cells, y agrees with the assembled matrix: {agrees}")

    message, _witness = rejected_split()
    print()
    print("Schedule(mass_action).split('c', 32).tag(c_outer='g.0', c_inner='l.0') ->")
    print(f"  IllegalCast: {message}")

    accepted = kulkarni(atomic=True)
    print()
    print(f"accepted: {accepted!r}")
    for fact in accepted.facts():
        if fact.kind != "buildable":
            status, by = fact.status.value, fact.decided_by or "-"
            print(f"  {status:8} {by:4} {fact.statement}")
    buildable, reason = accepted.buildable
    print(f"  buildable on c: {buildable}: {reason.split(':')[0]}")

    print()
    print(f"schedule: {vectorized!r}")
    fact = LoopyExecutor().differential(
        mass_action, vectorized, inputs_on(square_mesh())
    )
    for name, detail in fact.provenance["outputs"].items():
        print(
            f"  {name}: difference {detail['difference']:.3g} within "
            f"{detail['tolerance']:.3g} ({detail['exactness']}) -> "
            f"{fact.status.value}"
        )
    return 0 if ok and not buildable and fact.status.value == "tested" else 1


if __name__ == "__main__":
    raise SystemExit(main())
