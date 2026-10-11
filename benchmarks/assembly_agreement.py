"""``examples/assembly.py`` against firedrake and PyOP2 themselves, on their meshes.

Two halves, run in two environments, since firedrake and loopty pin different
islpy and loopy releases.

``python benchmarks/assembly_agreement.py firedrake OUT.npz [--n 64 256]``
    In a firedrake environment. Runs firedrake's own PyOP2 test of the mass
    kernel, ``test_assemble_ffc`` (``mass_ffc`` in a ``par_loop`` over the
    two triangles, ``INC`` into a PETSc matrix), and keeps the matrix PyOP2
    assembled. Then, on the same two triangles and on ``UnitSquareMesh(n,
    n)`` for each ``n``, assembles firedrake's P1 mass form, ``inner(u, v) *
    dx`` with ``u`` the P1 interpolant of ``1 + x y``, and keeps the mesh as
    firedrake numbers it (the node coordinates and the cell-node map), ``u``
    and the assembled vector, and on the two triangles the assembled matrix.

``python benchmarks/assembly_agreement.py compare IN.npz``
    In loopty's environment. Runs the example's ``mass_action`` on each mesh
    firedrake kept, natively where the mesh is small and compiled on the C
    target (the loop as it stands and Sun et al.'s schedule), and prints the
    largest difference from firedrake's vector, relative to its largest
    entry; on the two triangles also the matrix the cells' ``Ae`` add up to
    against PyOP2's and firedrake's matrices and against the test's
    ``expected_matrix``.
"""

from __future__ import annotations

import argparse
import importlib.util
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent

#: PyOP2's two triangles, as ``tests/pyop2/test_matrices.py`` writes them.
COORDS = np.array([[0.0, 0.0], [2.0, 0.0], [1.0, 1.0], [0.0, 1.5]])
CELLS = np.array([[0, 1, 3], [2, 3, 1]])

#: ``mass_ffc`` as the test writes it.
MASS_FFC = """
static void mass_ffc(PetscScalar A[9], PetscScalar x[6]) {
  double J_00 = x[2] - x[0];
  double J_01 = x[4] - x[0];
  double J_10 = x[3] - x[1];
  double J_11 = x[5] - x[1];

  double detJ = J_00*J_11 - J_01*J_10;
  double det = fabs(detJ);

  double W3[3] = {0.166666666666667, 0.166666666666667, 0.166666666666667};
  double FE0[3][3] = \\
  {{0.666666666666667, 0.166666666666667, 0.166666666666667},
  {0.166666666666667, 0.166666666666667, 0.666666666666667},
  {0.166666666666667, 0.666666666666667, 0.166666666666667}};

  for (unsigned int ip = 0; ip < 3; ip++)
    for (int j=0; j<3; ++j)
      for (int k=0; k<3; ++k)
        A[j*3+k] += FE0[ip][j]*FE0[ip][k]*W3[ip]*det;
}
"""


# {{{ the firedrake half


def pyop2_matrix() -> np.ndarray:
    """``test_assemble_ffc``: PyOP2 runs ``mass_ffc`` over the two triangles."""
    from pyop2 import op2
    from pyop2.datatypes import IntType, ScalarType

    nodes, elements = op2.Set(4, "nodes"), op2.Set(2, "elements")
    elem_node = op2.Map(
        elements, nodes, 3, CELLS.ravel().astype(IntType), "elem_node"
    )
    sparsity = op2.Sparsity(
        (op2.DataSet(nodes, 1), op2.DataSet(nodes, 1)),
        [(elem_node, elem_node, None)],
        name="sparsity",
    )
    mat = op2.Mat(sparsity, ScalarType, "mat")
    coords = op2.Dat(op2.DataSet(nodes, 2), COORDS.astype(ScalarType), ScalarType)
    op2.par_loop(
        op2.Kernel(MASS_FFC, "mass_ffc"),
        elements,
        mat(op2.INC, (elem_node, elem_node)),
        coords(op2.READ, elem_node),
    )
    mat.assemble()
    return np.asarray(mat.values)


def firedrake_mesh(mesh) -> dict:
    """firedrake's P1 mass form on ``mesh``: its numbering, ``u``, the vector."""
    import firedrake as fd

    V = fd.FunctionSpace(mesh, "CG", 1)
    W = fd.VectorFunctionSpace(mesh, "CG", 1)
    x, y = fd.SpatialCoordinate(mesh)
    coords = fd.Function(W).interpolate(fd.as_vector((x, y)))
    u = fd.Function(V).interpolate(1 + x * y)
    v = fd.TestFunction(V)
    assembled = fd.assemble(fd.inner(u, v) * fd.dx)
    out = {
        "coords": np.array(coords.dat.data_ro),
        "cells": np.array(V.cell_node_list),
        "u": np.array(u.dat.data_ro),
        "y": np.array(assembled.dat.data_ro),
    }
    if V.dof_count <= 16:
        trial = fd.TrialFunction(V)
        matrix = fd.assemble(fd.inner(trial, v) * fd.dx).M.handle
        out["M"] = np.array(matrix.convert("dense").getDenseArray())
    return out


def two_triangles():
    """PyOP2's two triangles as a firedrake mesh."""
    import firedrake as fd
    from firedrake.petsc import PETSc

    plex = PETSc.DMPlex().createFromCellList(
        2, CELLS.astype(PETSc.IntType), COORDS, comm=fd.COMM_WORLD
    )
    return fd.Mesh(plex)


def firedrake_half(out: str, sizes: list[int]) -> int:
    import firedrake as fd

    kept = {"pyop2_matrix": pyop2_matrix()}
    meshes = {"pyop2": two_triangles()}
    meshes.update({f"square{n}": fd.UnitSquareMesh(n, n) for n in sizes})
    for label, mesh in meshes.items():
        for key, value in firedrake_mesh(mesh).items():
            kept[f"{label}:{key}"] = value
        print(f"{label}: {len(kept[f'{label}:cells'])} cells", flush=True)
    import importlib.metadata as metadata

    kept["versions"] = np.array(
        [f"{name} {metadata.version(name)}" for name in ("firedrake", "loopy", "islpy")]
    )
    np.savez(out, **kept)
    return 0


# }}}


# {{{ the loopty half


def example():
    """``examples/assembly.py``, imported from the checkout."""
    path = ROOT / "examples" / "assembly.py"
    spec = importlib.util.spec_from_file_location("assembly", path)
    module = importlib.util.module_from_spec(spec)
    sys.modules["assembly"] = module
    spec.loader.exec_module(module)
    return module


def compare(path: str) -> int:
    import warnings

    warnings.simplefilter("ignore")
    from loopty import Arr, Schedule
    from loopty.executor import LoopyExecutor

    assembly = example()
    kept = np.load(path)
    print("firedrake:", ", ".join(str(v) for v in kept["versions"]))
    labels = sorted({key.split(":")[0] for key in kept if ":" in key})
    worst = 0.0
    for label in labels:
        coords, cells = kept[f"{label}:coords"], kept[f"{label}:cells"]
        u, want = kept[f"{label}:u"], kept[f"{label}:y"]

        def arguments():
            return {
                "cell": Arr.from_numpy(cells.astype(np.int64)),
                "coords": Arr.from_numpy(coords),
                "u": Arr.from_numpy(u),
                "Ae": Arr.zeros((len(cells), 3, 3)),
                "y": Arr.zeros(len(coords)),
            }

        runs = {}
        if len(cells) <= 10_000:
            data = arguments()
            assembly.mass_action(**data)
            runs["native"] = (data["y"].numpy(), data["Ae"].numpy())
        schedules = {"compiled": Schedule(assembly.mass_action, target="c")}
        if len(cells) % 4 == 0:
            schedules["sun"] = (
                Schedule(assembly.mass_action, target="c")
                .assume("nc % 4 == 0")
                .atomic("y")
                .split("c", 4)
                .tag(c_inner="ilp")
            )
        for name, schedule in schedules.items():
            out = LoopyExecutor().run(schedule, **arguments())
            runs[name] = (np.asarray(out["y"]), np.asarray(out["Ae"]))
        for name, (y, Ae) in runs.items():
            gap = float(np.abs(y - want).max() / np.abs(want).max())
            worst = max(worst, gap)
            print(f"{label}, {len(cells)} cells, {name}: y within {gap:.1e}")
            if f"{label}:M" in kept:
                M = assembly.matrix(cells, Ae, len(coords))
                gap = float(np.abs(M - kept[f"{label}:M"]).max())
                print(f"  the cells' Ae add up to firedrake's matrix within {gap:.1e}")
        if label == "pyop2":
            # firedrake numbers the nodes its own way; PyOP2's test does not
            data = assembly.inputs_on(assembly.pyop2_mesh())
            assembly.mass_action(**data)
            M = assembly.matrix(CELLS, data["Ae"].numpy(), 4)
            gap = float(np.abs(M - kept["pyop2_matrix"]).max())
            print(f"  on PyOP2's numbering, PyOP2's own matrix within {gap:.1e}")
            gap = float(np.abs(M - assembly.EXPECTED).max())
            print(f"  and the test's expected_matrix within {gap:.1e}, its 6 digits")
    return 0 if worst < 1e-12 else 1


# }}}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = parser.add_subparsers(dest="half", required=True)
    fd = sub.add_parser("firedrake")
    fd.add_argument("out")
    fd.add_argument("--n", type=int, nargs="*", default=[64, 256])
    cmp = sub.add_parser("compare")
    cmp.add_argument("path")
    args = parser.parse_args()
    if args.half == "firedrake":
        return firedrake_half(args.out, args.n)
    return compare(args.path)


if __name__ == "__main__":
    raise SystemExit(main())
