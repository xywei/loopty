"""Kernels that call kernels: a rule with a loop of its own, called where it is used.

loopy's substitution rules have no loops of their own yet: loopy #1044 asks
for a rule with a domain whose binder is fresh at every use, ``dot(a, b, n) :=
domain({[i]: 0<=i<n}, sum(i, a(i)*b(i)))``, and #1043 for rules passed to
rules. In loopty a kernel is such a rule, and a kernel calls it as Python
calls a function::

    @kernel
    def dot(a: Arr[Fin[n], Real], b: Arr[Fin[n], Real]) -> Real:
        return reduce_sum(a[k] * b[k] for k in a.dom)

    @kernel
    def matvec(A: Arr[Fin[m], Fin[n], Real], x: Arr[Fin[n], Real],
               y: Arr[Fin[m], Real]):
        for i in y.dom:
            y[i] = dot(A[i], x)

``-> Real`` makes ``dot`` a rule: it returns its value and writes nothing, so
a call of it is a value. ``A[i]`` is the row of ``A``, as in numpy, and
``x[col[r]]`` the cells of ``x`` that a row of column indices picks, so a
sparse product is ``y[r] = dot(val[r], x[col[r]])``.

The quadrature rule of firedrake's P1 mass kernel takes the integrand, a
function of the quadrature point, and a kernel that writes the element's mass
matrix calls it with a ``lambda``; the kernel over the mesh calls that one per
cell, with the cell's coordinates and its row of the output::

    @kernel
    def integrate(f: Fn[Fin[3], Real]) -> Real:
        return reduce_sum(W[q] * f(q) for q in W.dom)

    for c in M.dom:
        element_mass(X[cell[c]], Y[cell[c]], M[c])

Run this file three ways.

``python examples/calls.py``
    Runs the kernels natively against numpy and the closed form, then shows
    ``matvec``'s term, with ``dot`` inlined, the loop nest loopy generates
    for it, a call refused for the sizes of what it is given, and a schedule
    of ``matvec`` compiled and compared with the native run.

``lanky check examples/calls.py``
    Every kernel's ledger. The callers' facts are about the calls inlined:
    ``A[i, k]`` in bounds at every ``i``, a fresh ``k`` at each, and what the
    callee's contract checks of each call, at every point it is made: the row
    ``A[i]`` in bounds, the cells ``X[cell[c]]`` picks in bounds by type, and
    ``element_mass``'s postcondition restated after each call.

``loopty run examples/calls.py``
    Every kernel compiled on its own and compared with its native run, a
    rule writing its value into a one-cell result; ``integrate``, which takes
    a function, is compiled where it is called.

What is worth reading here
--------------------------

*A call is inlined, with binders of its own.* The callee's body runs again in
the caller's trace, against views of the caller's arrays, so ``dot``'s ``k``
is a sum of the caller's statement at every ``i``, and the compiled kernel is
one loop nest, as loopy would expand a rule. A name the caller already uses is
renamed (``k_0``).

*The call owes what the callee's contract checks.* Natively ``dot`` is called
``m`` times, and its contract checks each call. Compiled there is one kernel,
so the caller owes the same at every point of the call: the row in bounds,
an integral argument in its sort, and two rows of one array apart where the
callee writes one. isl decides them, and Lean states each from loopty's own
description of the call.

*The postcondition travels.* ``element_mass`` promises a symmetric matrix,
tested on its native runs; after each call the caller has it of ``M[c]``, a
fact decided by the call and resting on the callee's.
"""

from __future__ import annotations

import numpy as np
from lanky.prelude import Fn, Nat, Real

from loopty import Arr, Fin, Schedule, TraceError, kernel, reduce_sum, table

#: The 3-point rule on the reference triangle, and the P1 basis 1 - x - y, x, y
#: at its points (1/6, 1/6), (2/3, 1/6), (1/6, 2/3).
W = table([1 / 6, 1 / 6, 1 / 6])
PHI = table([[2 / 3, 1 / 6, 1 / 6], [1 / 6, 2 / 3, 1 / 6], [1 / 6, 1 / 6, 2 / 3]])


@kernel
def dot(a: Arr[Fin[n], Real], b: Arr[Fin[n], Real]) -> Real:
    """The dot product: a rule with a loop of its own."""
    return reduce_sum(a[k] * b[k] for k in a.dom)


@kernel
def matvec(A: Arr[Fin[m], Fin[n], Real], x: Arr[Fin[n], Real], y: Arr[Fin[m], Real]):
    """``y = A x``, one row at a time."""
    for i in y.dom:
        y[i] = dot(A[i], x)


@kernel
def csr(
    cnt: Arr[Fin[n], Nat],
    col: Arr[Fin[n], Fin[cnt], Fin[m]],
    val: Arr[Fin[n], Fin[cnt], Real],
    x: Arr[Fin[m], Real],
    y: Arr[Fin[n], Real],
):
    """A sparse product: each row's values dotted with the cells its columns pick."""
    for r in y.dom:
        y[r] = dot(val[r], x[col[r]])


@kernel
def integrate(f: Fn[Fin[3], Real]) -> Real:
    """The 3-point rule on the reference triangle, applied to ``f``."""
    return reduce_sum(W[q] * f(q) for q in W.dom)


@kernel
def element_mass(
    X: Arr[Fin[3], Real], Y: Arr[Fin[3], Real], Me: Arr[Fin[3], Fin[3], Real]
) -> all(Me[j, k] == Me[k, j] for j in Fin[3] for k in Fin[3]):
    """The P1 mass matrix of the triangle with corners ``(X, Y)``."""
    area = abs((X[1] - X[0]) * (Y[2] - Y[0]) - (X[2] - X[0]) * (Y[1] - Y[0]))
    for j, k in Me.dom:
        Me[j, k] = area * integrate(lambda q: PHI[q, j] * PHI[q, k])


@kernel
def mass_matrices(
    cell: Arr[Fin[nc], Fin[3], Fin[nv]],
    X: Arr[Fin[nv], Real],
    Y: Arr[Fin[nv], Real],
    M: Arr[Fin[nc], Fin[3], Fin[3], Real],
):
    """Each cell's mass matrix, from the coordinates of its corners."""
    for c in M.dom:
        element_mass(X[cell[c]], Y[cell[c]], M[c])


#: The rows of ``matvec`` split by 4, a schedule of the kernel with the call.
split_rows = Schedule(matvec).split(matvec.i, 4)


# {{{ inputs


def pyop2_mesh() -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """The two triangles of PyOP2's ``test_matrices.py``: nodes and cell map."""
    xs = np.array([0.0, 2.0, 1.0, 0.0])
    ys = np.array([0.0, 0.0, 1.0, 1.5])
    cells = np.array([[0, 1, 3], [2, 3, 1]])
    return xs, ys, cells


def sparse() -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """A small sparse matrix in CSR: the row lengths, the columns, the values."""
    counts = np.array([2, 0, 3, 1])
    columns = np.array([0, 4, 1, 2, 3, 0])
    values = np.linspace(-1.0, 1.5, 6)
    return counts, columns, values


def example_inputs() -> dict:
    """The inputs ``loopty run`` gives each kernel."""
    rng = np.random.default_rng(0)
    A, x = rng.standard_normal((6, 5)), rng.standard_normal(5)
    counts, columns, values = sparse()
    xs, ys, cells = pyop2_mesh()
    return {
        "dot": {"a": Arr.from_numpy(A[0].copy()), "b": Arr.from_numpy(x.copy())},
        # a rule that takes a function is compiled where it is called; its
        # term is checked against its body on a function of the point here
        "integrate": {"f": lambda q: 1.0 + q},
        "matvec": {"A": Arr.from_numpy(A), "x": Arr.from_numpy(x), "y": Arr.zeros(6)},
        "csr": {
            "cnt": Arr.from_numpy(counts),
            "col": Arr.ragged(counts, values=columns),
            "val": Arr.ragged(counts, values=values),
            "x": Arr.from_numpy(x),
            "y": Arr.zeros(4),
        },
        "element_mass": {
            "X": Arr.from_numpy(xs[:3].copy()),
            "Y": Arr.from_numpy(ys[:3].copy()),
            "Me": Arr.zeros((3, 3)),
        },
        "mass_matrices": {
            "cell": Arr.from_numpy(cells),
            "X": Arr.from_numpy(xs),
            "Y": Arr.from_numpy(ys),
            "M": Arr.zeros((2, 3, 3)),
        },
    }


# }}}


def rejected_size() -> None:
    """A call given a vector of a length nothing says is the row's: refused."""

    @kernel
    def matvec_by(
        A: Arr[Fin[m], Fin[n], Real], z: Arr[Fin[p], Real], y: Arr[Fin[m], Real]
    ):
        for i in y.dom:
            y[i] = dot(A[i], z)

    matvec_by.term


def main() -> int:
    """Run natively, then show the inlined term, its code, a refusal, a schedule."""
    from loopty.executor import LoopyExecutor, emit_code

    inputs = example_inputs()
    ok = True

    data = inputs["matvec"]
    want = data["A"].numpy() @ data["x"].numpy()
    matvec(**data)
    agrees = bool(np.allclose(data["y"].numpy(), want))
    ok &= agrees
    print(f"native: matvec agrees with A @ x: {agrees}")

    counts, columns, values = sparse()
    data = inputs["csr"]
    dense = np.zeros((4, 5))
    rows = np.repeat(np.arange(4), counts)
    dense[rows, columns] = values
    csr(**data)
    agrees = bool(np.allclose(data["y"].numpy(), dense @ data["x"].numpy()))
    ok &= agrees
    print(f"native: csr agrees with the dense product: {agrees}")

    xs, ys, cells = pyop2_mesh()
    data = inputs["mass_matrices"]
    mass_matrices(**data)
    x, y = xs[cells], ys[cells]
    det = (x[:, 1] - x[:, 0]) * (y[:, 2] - y[:, 0]) - (x[:, 2] - x[:, 0]) * (
        y[:, 1] - y[:, 0]
    )
    closed = np.abs(det)[:, None, None] * (np.ones((3, 3)) + np.eye(3)) / 24
    agrees = bool(np.allclose(data["M"].numpy(), closed))
    ok &= agrees
    print(f"native: each cell's matrix is |detJ| (1 + delta_jk) / 24: {agrees}")

    term = matvec.term
    (stmt,) = term.stmts
    (call,) = term.calls
    print()
    print(f"the term of matvec, with {call.callee} inlined at {call.where}:")
    print(
        f"  for every {', '.join(stmt.spelled)}: y[i] = sum over "
        f"{', '.join(stmt.expr.inames)} of {stmt.expr.body}"
    )
    for argument in call.arguments:
        print(f"  {call.callee}'s {argument.param} is {argument.spelled}")
    print()
    print("the loop nest loopy generates for it:")
    code = emit_code(matvec).splitlines()
    start = next(k for k, line in enumerate(code) if line.strip().startswith("for "))
    end = max(k for k, line in enumerate(code) if line.rstrip() == "}")
    for line in code[start:end]:
        print(line.rstrip())

    print()
    try:
        rejected_size()
    except TraceError as exc:
        print(f"refused: {exc}")

    print()
    fact = LoopyExecutor().differential(matvec, split_rows, example_inputs()["matvec"])
    print(f"schedule: {split_rows!r}")
    for name, detail in fact.provenance["outputs"].items():
        print(
            f"  {name}: difference {detail['difference']:.3g} within "
            f"{detail['tolerance']:.3g} ({detail['exactness']}) -> "
            f"{fact.status.value}"
        )
    return 0 if ok and fact.status.value == "tested" else 1


if __name__ == "__main__":
    raise SystemExit(main())
