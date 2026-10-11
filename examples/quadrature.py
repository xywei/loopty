"""Quadrature with constant tables: firedrake's P1 mass kernel, cell by cell.

The element kernel FFC and TSFC generate for the P1 mass matrix holds the
quadrature rule and the basis functions at its points as tables: read-only
arrays whose values are written into the generated code, not passed in. This
is that kernel, ``mass_ffc`` of firedrake's PyOP2 tests, applied to a field on
every cell of a mesh: per cell, gather the field at the cell's three nodes,
integrate the basis against it with the 3-point rule, and keep each cell's
three values. The tables are declared next to the kernel as the numpy arrays
they are::

    W = table([1 / 6, 1 / 6, 1 / 6])
    PHI = table([[2 / 3, 1 / 6, 1 / 6], [1 / 6, 2 / 3, 1 / 6], [1 / 6, 1 / 6, 2 / 3]])

Run this file three ways.

``python examples/quadrature.py``
    Runs the kernel natively on the two triangles of PyOP2's test mesh and on
    a mesh of the unit square, checks it against the element mass matrix in
    closed form, and prints the tables as the compiled code holds them and the
    comparison of the compiled run with the native one.

``lanky check examples/quadrature.py``
    The kernel's obligations: every read of a table in bounds over its shape,
    decided by isl, and every gather through the cell map in bounds by type.

``loopty run examples/quadrature.py``
    The kernel, its cells split by 32 as PyOP2's GPU schedule splits them,
    compiled and compared with the native run.

What is worth reading here
--------------------------

*A table is part of the kernel.* ``W[q]`` reads a value known when the kernel
is compiled, and the lowering writes it in, ``static double W[3] = {
0.16666666666666666, ... }``, with every digit, so the compiled run reads the
numbers the native one does. The table is a copy, and nobody can write it, the
kernel included.

*The gathers are in bounds by type.* ``cell: Arr[Fin[nc], Fin[3], Fin[nv]]``
says each entry of the cell map is a node, a point of ``Fin[nv]``, so ``u[cell[c,
k]]`` and ``X[cell[c, 0]]`` read cells the arrays have with no proof at all; the
contract checks the entries when the run starts.

*The answer is known.* The element mass matrix of a P1 triangle is ``|detJ| (1
+ delta_jk) / 24``, which the 3-point rule integrates exactly, so each cell's
values are ``|detJ| (u_j + sum_k u_k) / 24``.
"""

from __future__ import annotations

import numpy as np
from lanky.prelude import Real

from loopty import Arr, Fin, Schedule, kernel, reduce_sum, table

#: The 3-point rule on the reference triangle: a weight per point, and the P1
#: basis 1 - x - y, x, y at the points (1/6, 1/6), (2/3, 1/6), (1/6, 2/3).
W = table([1 / 6, 1 / 6, 1 / 6])
PHI = table([[2 / 3, 1 / 6, 1 / 6], [1 / 6, 2 / 3, 1 / 6], [1 / 6, 1 / 6, 2 / 3]])


@kernel
def mass_action(
    cell: Arr[Fin[nc], Fin[3], Fin[nv]],
    X: Arr[Fin[nv], Real],
    Y: Arr[Fin[nv], Real],
    u: Arr[Fin[nv], Real],
    yl: Arr[Fin[nc], Fin[3], Real],
):
    """Each cell's P1 mass matrix applied to the field at its nodes."""
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


# {{{ meshes


def pyop2_mesh() -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """The two triangles of PyOP2's ``test_matrices.py``: nodes and cell map."""
    xs = np.array([0.0, 2.0, 1.0, 0.0])
    ys = np.array([0.0, 0.0, 1.0, 1.5])
    cells = np.array([[0, 1, 3], [2, 3, 1]])
    return xs, ys, cells


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
    }


def closed_form(mesh: tuple[np.ndarray, np.ndarray, np.ndarray]) -> np.ndarray:
    """``|detJ| (u_j + sum_k u_k) / 24`` on every cell, the exact element mass."""
    xs, ys, cells = mesh
    u = 1.0 + xs * ys
    x, y, local = xs[cells], ys[cells], u[cells]
    det = (x[:, 1] - x[:, 0]) * (y[:, 2] - y[:, 0]) - (x[:, 2] - x[:, 0]) * (
        y[:, 1] - y[:, 0]
    )
    return np.abs(det)[:, None] * (local + local.sum(axis=1)[:, None]) / 24


# }}}


#: PyOP2's GPU schedule splits the cells by 32 (Kulkarni's ``snpt``); on the C
#: target the split stays in sequence.
split_cells = (
    Schedule(mass_action, target="c").split("c", 32).example(**inputs_on(square_mesh()))
)


def example_inputs() -> dict:
    """The inputs ``loopty run`` gives the kernel: the two triangles."""
    return inputs_on(pyop2_mesh())


def main() -> int:
    """Run on both meshes natively, then show the tables and the compiled run."""
    from loopty.executor import LoopyExecutor, emit_code

    ok = True
    for label, mesh in (
        ("PyOP2's two triangles", pyop2_mesh()),
        ("a square", square_mesh()),
    ):
        data = inputs_on(mesh)
        mass_action(**data)
        agrees = bool(np.allclose(data["yl"].numpy(), closed_form(mesh)))
        ok = ok and agrees
        cells = len(mesh[2])
        print(f"{label}: {cells} cells, agrees with the closed form: {agrees}")
    print()
    for line in emit_code(split_cells).splitlines():
        if line.startswith("static"):
            print(line)
    print()
    fact = LoopyExecutor().differential(
        mass_action, split_cells, inputs_on(square_mesh())
    )
    print(f"schedule: {split_cells!r}")
    for name, detail in fact.provenance["outputs"].items():
        print(
            f"  {name}: difference {detail['difference']:.3g} within "
            f"{detail['tolerance']:.3g} ({detail['exactness']}) -> "
            f"{fact.status.value}"
        )
    return 0 if ok and fact.status.value == "tested" else 1


if __name__ == "__main__":
    raise SystemExit(main())
