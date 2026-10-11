"""Kernels that call kernels inside their loops (loopty.calls).

A rule (``dot``) returns its value, and its call is that value; a kernel that
writes its parameters (``symmetric``) is called as a statement and writes
through the row it is given. ``integrate`` takes a function, and ``p1_mass``
passes it one. Every call owes the caller facts at the points it is made at:
the rows it passes in bounds, its integral scalars in their sorts, two rows of
one array apart where the callee writes one, and the callee's postcondition
restated. The kernels whose
names start with ``off_`` break one of them, so that a refutation, and Lean's
proof of its negation, are exercised too.
"""

from __future__ import annotations

from lanky.prelude import Fn, Nat, Real

from loopty import Arr, Fin, kernel, reduce_sum, table, when

#: The 3-point rule on the reference triangle, and the P1 basis at its points.
W = table([1 / 6, 1 / 6, 1 / 6])
PHI = table([[2 / 3, 1 / 6, 1 / 6], [1 / 6, 2 / 3, 1 / 6], [1 / 6, 1 / 6, 2 / 3]])


@kernel
def dot(a: Arr[Fin[n], Real], b: Arr[Fin[n], Real]) -> Real:
    """The dot product: a rule with a loop of its own."""
    return reduce_sum(a[k] * b[k] for k in a.dom)


@kernel
def matvec(A: Arr[Fin[m], Fin[n], Real], x: Arr[Fin[n], Real], y: Arr[Fin[m], Real]):
    """``y = A x``, a row at a time."""
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
def gram(A: Arr[Fin[m], Fin[n], Real], G: Arr[Fin[m], Fin[m], Real]):
    """``A A^T``, two rows of ``A`` at a time, one row twice on the diagonal."""
    for i in G.dom:
        for j in G.dom[i]:
            G[i, j] = dot(A[i], A[j])


@kernel
def add_into(a: Arr[Fin[n], Real], b: Arr[Fin[n], Real]):
    """``a += b``: it writes ``a``, so ``b`` may not share its cells."""
    for k in a.dom:
        a[k] = a[k] + b[k]


@kernel
def running(A: Arr[Fin[m], Fin[n], Real]):
    """Each row of ``A`` plus the one before it, in order: a running sum."""
    for i in A.dom:
        with when(i > 0):
            add_into(A[i], A[i - 1])


@kernel
def weighted(
    c: Arr[Fin[p], Real],
    B: Arr[Fin[p], Fin[n], Real],
    x: Arr[Fin[n], Real],
    y: Arr[Fin[q], Real],
):
    """A rule called inside a sum: ``y[i] = sum_j c[j] (B[j] . x)``."""
    for i in y.dom:
        y[i] = reduce_sum(c[j] * dot(B[j], x) for j in c.dom)


@kernel
def symmetric(u: Arr[Fin[3], Real], M: Arr[Fin[3], Fin[3], Real]) -> all(
    M[j, k] == M[k, j] for j in Fin[3] for k in Fin[3]
):
    """The outer product of ``u`` with itself, which is symmetric."""
    for j in M.dom:
        for k in M.dom[j]:
            M[j, k] = u[j] * u[k]


@kernel
def outer(
    cell: Arr[Fin[nc], Fin[3], Fin[nv]],
    u: Arr[Fin[nv], Real],
    M: Arr[Fin[nc], Fin[3], Fin[3], Real],
):
    """Each cell's outer product of the field at its nodes."""
    for c in M.dom:
        symmetric(u[cell[c]], M[c])


@kernel
def integrate(f: Fn[Fin[3], Real]) -> Real:
    """The 3-point rule on the reference triangle, applied to ``f``."""
    return reduce_sum(W[q] * f(q) for q in W.dom)


@kernel
def p1_mass(u: Arr[Fin[3], Real], j: Fin[3]) -> Real:
    """Row ``j`` of the reference P1 mass matrix applied to ``u``."""
    return integrate(lambda q: PHI[q, j] * reduce_sum(PHI[q, k] * u[k] for k in u.dom))


@kernel
def mass_action(
    cell: Arr[Fin[nc], Fin[3], Fin[nv]],
    u: Arr[Fin[nv], Real],
    yl: Arr[Fin[nc], Fin[3], Real],
):
    """Each cell's reference mass matrix applied to the field at its nodes."""
    for c in yl.dom:
        for j in yl.dom[c]:
            yl[c, j] = p1_mass(u[cell[c]], j)


@kernel
def shifted(A: Arr[Fin[m], Fin[n], Real], x: Arr[Fin[n], Real], y: Arr[Fin[m], Real]):
    """``y[i] = A[i + 1] . x`` where there is a next row, kept so by its guard."""
    for i in y.dom:
        with when(i + 1 < y.dom.size):
            y[i] = dot(A[i + 1], x)


@kernel
def off_the_end(
    A: Arr[Fin[m], Fin[n], Real], x: Arr[Fin[n], Real], y: Arr[Fin[m], Real]
):
    """``A[i + 1]`` at the last ``i``: a row ``A`` does not have."""
    for i in y.dom:
        y[i] = dot(A[i + 1], x)


@kernel
def off_the_sort(
    cell: Arr[Fin[nc], Fin[3], Fin[nv]],
    u: Arr[Fin[nv], Real],
    yl: Arr[Fin[nc], Fin[3], Real],
):
    """``j + 1`` for a ``Fin[3]`` at ``j = 2``: no point of the sort."""
    for c in yl.dom:
        for j in yl.dom[c]:
            yl[c, j] = p1_mass(u[cell[c]], j + 1)


@kernel
def off_the_diagonal(A: Arr[Fin[m], Fin[n], Real]):
    """Row ``j`` added into row ``i``: one row, written and read, at ``i == j``."""
    for i in A.dom:
        for j in A.dom:
            add_into(A[i], A[j])
