"""Assembly: firedrake's P1 mass matrix, cell by cell, applied and added into the nodes.

PyOP2 runs an element kernel over the cells of a mesh, and around it the
assembly loop: per cell, gather the coordinates of the cell's three nodes,
call the kernel, and add what it computed into the nodes. This file is that
loop, with ``mass_ffc``, the P1 mass kernel of firedrake's PyOP2 tests, as
the element kernel, called inside the loop over the cells as PyOP2 calls it,
and the cell's matrix applied to a field and added into ``y``,
``y[cell[c, a]] += ...``. Two cells that share a node add into one cell of
``y``, which is what an assembly is, and what makes the loop over the cells a
race on a device unless the additions are atomic.

Run this file three ways.

``python examples/assembly.py``
    Runs the kernel natively on the two triangles of PyOP2's test mesh, where
    the cells' matrices add up to the test's ``expected_matrix``, and on a
    mesh of the unit square, where ``y`` is the closed form. Then PyOP2's GPU
    schedule, the cells split by 32 onto groups and work items: refused, with
    the two cells of PyOP2's mesh that share a node; and accepted once the
    updates of ``y`` are atomic. Then Sun et al.'s, the cells four at a time,
    compiled and compared with the native run.

``lanky check examples/assembly.py``
    The ledger, three layers in one table. The mathematics: the rule
    integrates every quadratic exactly, the element's mass in closed form,
    and the scatter keeps the total on every mesh; proved by Lean where
    Mathlib is installed, tested where it is not. The kernels: the call's
    rows and gathers in bounds, ``mass_ffc``'s symmetric matrix restated of
    every cell after its call, and the accumulation into ``y`` approx, which
    is what an update owes in place of distinct writes. Then Sun et al.'s
    casts, under the assumption their schedule makes.

``loopty run examples/assembly.py``
    Both kernels and Sun et al.'s schedule compiled and compared with the
    native run; ``--target opencl`` runs them on a device, each update a
    compare-and-swap.

What it mirrors, at firedrake c68a775
-------------------------------------

- ``mass_ffc``, ``elem_node_map``, the coordinates and ``expected_matrix``:
  `tests/pyop2/test_matrices.py
  <https://github.com/firedrakeproject/firedrake/blob/c68a775711d59693f816b6c054226d4ea307d3b3/tests/pyop2/test_matrices.py#L291-L315>`__,
  with the mesh at `L57
  <https://github.com/firedrakeproject/firedrake/blob/c68a775711d59693f816b6c054226d4ea307d3b3/tests/pyop2/test_matrices.py#L57>`__
  and `L108-L113
  <https://github.com/firedrakeproject/firedrake/blob/c68a775711d59693f816b6c054226d4ea307d3b3/tests/pyop2/test_matrices.py#L108-L113>`__,
  the matrix at `L462-L468
  <https://github.com/firedrakeproject/firedrake/blob/c68a775711d59693f816b6c054226d4ea307d3b3/tests/pyop2/test_matrices.py#L462-L468>`__,
  and the tolerance of ``test_assemble_ffc`` at `L698-L706
  <https://github.com/firedrakeproject/firedrake/blob/c68a775711d59693f816b6c054226d4ea307d3b3/tests/pyop2/test_matrices.py#L698-L706>`__.
- The loop around it, PyOP2's wrapper: pack, call, and the ``INC`` unpack
  (`pyop2/codegen/builder.py
  <https://github.com/firedrakeproject/firedrake/blob/c68a775711d59693f816b6c054226d4ea307d3b3/pyop2/codegen/builder.py#L322-L431>`__),
  and TSFC's tables as read-only initialized temporaries (`tsfc/loopy.py
  <https://github.com/firedrakeproject/firedrake/blob/c68a775711d59693f816b6c054226d4ea307d3b3/tsfc/loopy.py#L215-L290>`__).
- The rule, FIAT's for degree 2 on a triangle (Strang and Fix), whose points
  are in the order of ``mass_ffc``'s rows: `FIAT/quadrature_schemes.py
  <https://github.com/firedrakeproject/fiat/blob/31ac887cca46a140bd0afd95147cfaf10acc0b7f/FIAT/quadrature_schemes.py#L356-L369>`__.
- Kulkarni's GPU split, the loop over the cells by 32 onto ``g.0``/``l.0``
  with every ``INC`` an ``AtomicUpdate`` (`OP2/PyOP2 gpu, snpt.py
  <https://github.com/OP2/PyOP2/blob/e105f5ec1de5894f6d8468a2e14c26626a981735/pyop2/transforms/snpt.py#L20-L50>`__
  and `gpu_utils.py
  <https://github.com/OP2/PyOP2/blob/e105f5ec1de5894f6d8468a2e14c26626a981735/pyop2/transforms/gpu_utils.py#L14-L95>`__),
  and Sun et al.'s vectorization, the loop split by 4 under ``end - start``
  a multiple of 4, with loopy's ``write_race`` warnings silenced
  (`OP2/PyOP2 tj/loopy-vectorise, sequential.py
  <https://github.com/OP2/PyOP2/blob/263ecb701dfad9abd3220509d3e7f4b525df5bb6/pyop2/sequential.py#L66-L111>`__
  and `rep2loopy.py
  <https://github.com/OP2/PyOP2/blob/263ecb701dfad9abd3220509d3e7f4b525df5bb6/pyop2/codegen/rep2loopy.py#L450-L470>`__).

What is worth reading here
--------------------------

*The element kernel is a kernel, called where PyOP2 calls it.*
``mass_ffc(coords[cell[c]], Ae[c])`` passes the cell's three corners, the
rows of ``coords`` the cell map picks, and the cell's row of ``Ae``; traced,
the call is inlined with loops of its own, and every fact of
``mass_action`` is about what it runs. ``mass_ffc`` promises a symmetric
matrix, and after each call ``mass_action`` has it of ``Ae[c]``.

*An update is not a write.* ``y[cell[c, a]] += ...`` adds to the cell it
writes a value that reads nothing else of ``y``: two such updates of one cell
give the same sum whichever runs first, to the class of the sum, which is
``approx`` for ``Real``. So the kernel owes no distinct writes for it, and a
schedule may reorder the updates; it only may not run two of one cell at the
same time, which loses one of them.

*The race is named by the mesh.* The cell of ``y`` an update writes is read
from the cell map, which isl cannot follow, so isl takes any two updates to
touch one cell. The refusal is not isl's pair: it is two updates the example
inputs make of one node, the node PyOP2's two triangles share. PyOP2 makes
every update atomic on a device, and silences loopy's ``write_race`` where it
vectorizes; here ``atomic("y")`` is a step, which makes the split legal.

*In bounds by type, and the type is checked.* ``cell`` is declared
``Arr[Fin[nc], Fin[3], Fin[nv]]``: every entry is a node. So ``y[cell[c, a]]``
is in bounds by type, with no check in the generated code, and a cell map
with an entry that is not a node is refused when the kernel is called,
natively or compiled: ``cell[1, 1] is 7, which is not a value of Fin(nv)``.
Sun et al.'s assumption is checked at the call the same way: their schedule
run on PyOP2's two triangles is refused, ``nc = 2``.

*The tables are the rule, to their 15 digits.* ``FE0`` and ``W3`` are
``mass_ffc``'s tables as the test writes them. The theorems are about the
rule they round, in exact rationals, and ``python`` checks that the tables
are that rule to within ``1e-15``.
"""

from __future__ import annotations

from fractions import Fraction

import numpy as np
from lanky import theorem
from lanky.prelude import Fn, Nat, Real

from loopty import Arr, Fin, Schedule, kernel, reduce_sum, table

# {{{ the mathematics

#: FIAT's rule for degree 2 on the reference triangle ``x, y >= 0, x + y <=
#: 1``: three points, each weighted 1/6, in the order of ``mass_ffc``'s rows.
POINTS = (
    (Fraction(1, 6), Fraction(1, 6)),
    (Fraction(1, 6), Fraction(2, 3)),
    (Fraction(2, 3), Fraction(1, 6)),
)


def rule(f):
    """The rule applied to ``f(x, y)``."""
    (x0, y0), (x1, y1), (x2, y2) = POINTS
    return (f(x0, y0) + f(x1, y1) + f(x2, y2)) / 6


#: The P1 basis on the reference triangle: ``1 - x - y``, ``x`` and ``y``.
P1 = (lambda x, y: 1 - x - y, lambda x, y: x, lambda x, y: y)


def field_at(u, x, y):
    """The P1 field with the values ``u`` at the corners, at ``(x, y)``."""
    return u[0] * P1[0](x, y) + u[1] * P1[1](x, y) + u[2] * P1[2](x, y)


@theorem
def exact_to_degree_two(
    a: Real, b: Real, c: Real, d: Real, e: Real, f: Real
) -> rule(lambda x, y: a + b * x + c * y + d * x * x + e * x * y + f * y * y) == (
    a / 2 + b / 6 + c / 6 + d / 12 + e / 24 + f / 12
):
    """The rule integrates every quadratic exactly.

    The right side is the quadratic's integral over the triangle, which has
    ``m! n! / (m + n + 2)!`` for ``x**m * y**n``.
    """


@theorem
def element_mass(J: Real, u: Fn[Fin[3], Real]) -> all(
    abs(J) * rule(lambda x, y: P1[j](x, y) * field_at(u, x, y))
    == abs(J) * (u[j] + u[0] + u[1] + u[2]) / 24
    for j in Fin[3]
):
    """A cell's mass matrix is ``|J| (1 + delta_jk) / 24``: here applied to ``u``.

    ``J`` is the Jacobian's determinant, twice the cell's area, and the left
    side is what ``mass_ffc`` computes, row ``j`` of its matrix times ``u``,
    in exact arithmetic.
    """


@theorem
def conservation(
    nc: Nat,
    nv: Nat,
    d: Nat,
    cell: Fn[Fin[nc], Fn[Fin[d], Fin[nv]]],
    yl: Fn[Fin[nc], Fn[Fin[d], Real]],
    y: Fn[Fin[nv], Real],
    scatter: all(
        y[i] == sum(yl[c][k] for c in Fin[nc] for k in Fin[d] if cell[c][k] == i)
        for i in Fin[nv]
    ),
) -> sum(y[i] for i in Fin[nv]) == sum(yl[c][k] for c in Fin[nc] for k in Fin[d]):
    """Adding each cell's values into its nodes keeps their total, on every mesh.

    ``cell[c][k]`` is the node of corner ``k`` of cell ``c``, ``yl[c][k]``
    the value the cell adds there, and ``y[i]`` what the cells add into node
    ``i``, from zero.
    """


# }}}


# {{{ the kernels

#: ``mass_ffc``'s tables as firedrake's test writes them: the rule's weights,
#: and the P1 basis at its points, a point to a row.
W3 = table([0.166666666666667, 0.166666666666667, 0.166666666666667])
FE0 = table(
    [
        [0.666666666666667, 0.166666666666667, 0.166666666666667],
        [0.166666666666667, 0.166666666666667, 0.666666666666667],
        [0.166666666666667, 0.666666666666667, 0.166666666666667],
    ]
)


@kernel
def mass_ffc(x: Arr[Fin[3], Fin[2], Real], A: Arr[Fin[3], Fin[3], Real]) -> all(
    A[j, k] == A[k, j] for j in Fin[3] for k in Fin[3]
):
    """FFC's P1 mass kernel: the mass matrix of the triangle with corners ``x``."""
    J_00 = x[1, 0] - x[0, 0]
    J_01 = x[2, 0] - x[0, 0]
    J_10 = x[1, 1] - x[0, 1]
    J_11 = x[2, 1] - x[0, 1]
    det = abs(J_00 * J_11 - J_01 * J_10)
    for j in A.dom:
        for k in A.dom[j]:
            A[j, k] = reduce_sum(
                FE0[ip, j] * FE0[ip, k] * W3[ip] * det for ip in W3.dom
            )


@kernel
def mass_action(
    cell: Arr[Fin[nc], Fin[3], Fin[nv]],
    coords: Arr[Fin[nv], Fin[2], Real],
    u: Arr[Fin[nv], Real],
    Ae: Arr[Fin[nc], Fin[3], Fin[3], Real],
    y: Arr[Fin[nv], Real],
):
    """PyOP2's loop over the cells: each cell's matrix times ``u``, added into ``y``."""
    for c in Ae.dom:
        mass_ffc(coords[cell[c]], Ae[c])
        for a in Ae.dom[c]:
            y[cell[c, a]] += reduce_sum(
                Ae[c, a, b] * u[cell[c, b]] for b in Ae.dom[c, a]
            )


# }}}


# {{{ meshes


def pyop2_mesh() -> tuple[np.ndarray, np.ndarray]:
    """The two triangles of PyOP2's ``test_matrices.py``: coordinates and cell map."""
    coords = np.array([[0.0, 0.0], [2.0, 0.0], [1.0, 1.0], [0.0, 1.5]])
    cells = np.array([[0, 1, 3], [2, 3, 1]])
    return coords, cells


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


def square_mesh(n: int = 4) -> tuple[np.ndarray, np.ndarray]:
    """The unit square in ``n`` by ``n`` squares, each cut into two triangles."""
    grid = np.linspace(0.0, 1.0, n + 1)
    xs, ys = (axis.ravel() for axis in np.meshgrid(grid, grid, indexing="ij"))
    i, j = (axis.ravel() for axis in np.meshgrid(range(n), range(n), indexing="ij"))
    a, b = i * (n + 1) + j, (i + 1) * (n + 1) + j
    cells = np.stack([a, b, b + 1, a, b + 1, a + 1], axis=1).reshape(-1, 3)
    return np.stack([xs, ys], axis=1), cells


def field(coords: np.ndarray) -> np.ndarray:
    """A smooth field on the nodes, ``1 + x y``."""
    return 1.0 + coords[:, 0] * coords[:, 1]


def inputs_on(mesh: tuple[np.ndarray, np.ndarray]) -> dict:
    """``mass_action``'s arguments on a mesh, with the field on its nodes."""
    coords, cells = mesh
    return {
        "cell": Arr.from_numpy(cells.astype(np.int64)),
        "coords": Arr.from_numpy(coords),
        "u": Arr.from_numpy(field(coords)),
        "Ae": Arr.zeros((len(cells), 3, 3)),
        "y": Arr.zeros(len(coords)),
    }


def assembled(mesh: tuple[np.ndarray, np.ndarray]) -> np.ndarray:
    """``y`` in closed form: each cell's ``|J| (u_j + u_0 + u_1 + u_2) / 24``, added."""
    coords, cells = mesh
    x, local = coords[cells], field(coords)[cells]
    J = (x[:, 1, 0] - x[:, 0, 0]) * (x[:, 2, 1] - x[:, 0, 1]) - (
        x[:, 2, 0] - x[:, 0, 0]
    ) * (x[:, 1, 1] - x[:, 0, 1])
    element = np.abs(J)[:, None] * (local + local.sum(axis=1)[:, None]) / 24
    out = np.zeros(len(coords))
    np.add.at(out, cells, element)
    return out


def matrix(cells: np.ndarray, Ae: np.ndarray, nv: int) -> np.ndarray:
    """The cells' matrices added into one, as PyOP2's ``INC`` into a ``Mat`` does."""
    out = np.zeros((nv, nv))
    np.add.at(out, (cells[:, :, None], cells[:, None, :]), Ae)
    return out


# }}}


# {{{ the schedules


def kulkarni(atomic: bool, target: str = "c") -> Schedule:
    """PyOP2's GPU schedule (Kulkarni): the cells split by 32 onto ``g.0``/``l.0``.

    Kulkarni's transform assumes ``start < end``, a loop of one cell or more.
    Without ``atomic`` the split's tag is refused; with it, accepted. On the C
    target, which has no groups or work items, the accepted schedule cannot
    be built (its ``buildable`` fact says so); ``target="opencl"`` runs it.
    """
    schedule = Schedule(mass_action, target=target).assume("nc >= 1")
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


#: Sun et al.'s schedule: the cells four at a time, under their assumption
#: that the cells are a multiple of four, ``exists zz: end = 4 zz + start``.
#: ``ilp`` leaves the four in no order, so the updates of ``y`` are atomic
#: first. Their branch tags the four ``c_vec``, a tag of their loopy fork
#: for C vector types; ``ilp`` is loopy's, and on the C target loopy writes
#: the four out one after another, which is not SIMD code. What ``loopty
#: run`` compiles and compares.
vectorized = (
    Schedule(mass_action, target="c")
    .assume("nc % 4 == 0")
    .atomic("y")
    .split("c", 4)
    .tag(c_inner="ilp")
    .example(**inputs_on(square_mesh()))
)


# }}}


def example_inputs() -> dict:
    """The inputs ``loopty run`` gives each kernel: PyOP2's two triangles."""
    coords, cells = pyop2_mesh()
    return {
        "mass_ffc": {"x": Arr.from_numpy(coords[cells[0]]), "A": Arr.zeros((3, 3))},
        "mass_action": inputs_on((coords, cells)),
    }


def main() -> int:
    """Run natively on both meshes, then the refused and the accepted schedules."""
    from loopty.executor import LoopyExecutor
    from loopty.obligations import prove

    coords, cells = pyop2_mesh()
    data = inputs_on((coords, cells))
    mass_action(**data)
    Ae, y = data["Ae"].numpy(), data["y"].numpy()
    M = matrix(cells, Ae, len(coords))
    test = bool(np.allclose(M, EXPECTED, 1e-5))
    action = bool(np.allclose(y, M @ field(coords)))
    print(f"PyOP2's two triangles: the cells' matrices sum to expected_matrix: {test}")
    print(f"  and y is that matrix times u: {action}")

    mesh = square_mesh()
    data = inputs_on(mesh)
    mass_action(**data)
    closed = bool(np.allclose(data["y"].numpy(), assembled(mesh)))
    print(f"a square, {len(mesh[1])} cells: y is the closed form: {closed}")

    exact = np.array([[float(phi(*p)) for phi in P1] for p in POINTS])
    gap = max(np.abs(W3.numpy() - 1 / 6).max(), np.abs(FE0.numpy() - exact).max())
    rounded = bool(gap <= 1e-15)
    print(f"W3 and FE0 are the rule to {gap:.1e}: {rounded}")

    message, _witness = rejected_split()
    print()
    print("Schedule(mass_action).split('c', 32).tag(c_outer='g.0', c_inner='l.0') ->")
    print(f"  IllegalCast: {message}")

    accepted = kulkarni(atomic=True)
    print()
    print(f"accepted: {accepted!r}")
    for fact in accepted.facts():
        if fact.kind != "buildable":
            # Lean, where it is installed, proves each cast, that no
            # dependence joins two work items included
            fact = prove(fact)
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
    ok = test and action and closed and rounded
    return 0 if ok and not buildable and fact.status.value == "tested" else 1


if __name__ == "__main__":
    raise SystemExit(main())
