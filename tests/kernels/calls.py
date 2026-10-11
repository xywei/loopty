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

from loopty import Arr, Fin, Schedule, kernel, layout, program, reduce_sum, table, when

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


# {{{ with updates, a window and a layout

# A callee whose statement is an update, y[cell[k]] += yl[k], whose updates
# the caller's schedule makes atomic; a program whose kernels call a kernel per
# row, its intermediate kept in a window over the inlined statements; and a
# callee that reads through a declared layout.


@kernel
def add_local(cell: Arr[Fin[3], Fin[nv]], yl: Arr[Fin[3], Real], y: Arr[Fin[nv], Real]):
    """A cell's three values added into its nodes: an update, ``y[...] += v``."""
    for k in yl.dom:
        y[cell[k]] += yl[k]


@kernel
def assemble(
    cell: Arr[Fin[nc], Fin[3], Fin[nv]],
    yl: Arr[Fin[nc], Fin[3], Real],
    y: Arr[Fin[nv], Real],
):
    """PyOP2's assembly loop, whose updates are a called kernel's."""
    for c in yl.dom:
        add_local(cell[c], yl[c], y)


@kernel
def half_square(u: Arr[Fin[n], Real], f: Arr[Fin[n], Real]):
    """The Burgers flux of one row."""
    for j in u.dom:
        f[j] = 0.5 * u[j] * u[j]


@kernel
def centred(f: Arr[Fin[n], Real], rhs: Arr[Fin[n], Real]):
    """The centred divergence of one row, at its interior points."""
    for i in rhs.dom:
        with when((i > 0) & (i + 1 < rhs.dom.size)):
            rhs[i] = -(f[i + 1] - f[i - 1]) / 2


@kernel
def row_fluxes(u: Arr[Fin[m], Fin[n], Real], f: Arr[Fin[m], Fin[n], Real]):
    """The flux of every row, a call per row."""
    for e in u.dom:
        half_square(u[e], f[e])


@kernel
def row_divergences(f: Arr[Fin[m], Fin[n], Real], rhs: Arr[Fin[m], Fin[n], Real]):
    """The divergence of every row, a call per row."""
    for e in rhs.dom:
        centred(f[e], rhs[e])


@program
def burgers_rows(u, rhs):
    """The flux of every row into an array of the program's own, then its divergence."""
    f = Arr.zeros_like(u)
    row_fluxes(u, f)
    row_divergences(f, rhs)


@layout
def by_columns(a: Fin[3], b: Fin[3]) -> Fin[9]:
    """A 3 x 3 block stored a column at a time."""
    return a + 3 * b


@kernel
def unflatten_block(flat: Arr[Fin[9], Real], u: Arr[Fin[3], Fin[3], Real]):
    """One block's cells, read through its layout."""
    for a in u.dom:
        for b in u.dom[a]:
            u[a, b] = flat[by_columns(a, b)]


@kernel
def unflatten(flat: Arr[Fin[ne], Fin[9], Real], u: Arr[Fin[ne], Fin[3], Fin[3], Real]):
    """Every element's block, a call per element."""
    for e in u.dom:
        unflatten_block(flat[e], u[e])


#: The called kernel's updates atomic, so that ``ilp`` may leave two in no order.
assembled_in_no_order = Schedule(assemble).atomic("y").split("c", 2).tag(c_inner="ilp")

#: Each row's divergence one step behind its flux, both inlined calls.
fused_rows = Schedule(burgers_rows, sizes={"m": 3, "n": 8}).fuse(
    "row_fluxes", "row_divergences", shift=(0, 1)
)

#: ``f`` in the three cells of a row live at once, ``f[e, j]`` in slot ``j % 3``.
windowed_rows = fused_rows.contract("f")


def rejected_window_of_two_rows() -> str:
    """A window of two cells, whose slot the inlined flux overwrites first."""
    from loopty.schedule import IllegalCast

    try:
        fused_rows.contract("f", lambda e, j: j % 2)
    except IllegalCast as refused:
        return str(refused)
    raise AssertionError("a window of two cells is too small")


# }}}
