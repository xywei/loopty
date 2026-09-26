"""Kernels over polyhedral domains, one or more of each kind.

The lower triangle as ``Where``, the triangle with its diagonal as ``Sigma``, a
band, a union of two pieces, and the edge cases the tests pin: an access that
is a cell of the box and not of the domain, a fiber taken at a point outside
the domain, a domain whose rows skip columns, and boxes whose extents are
negative at small sizes.
"""

from __future__ import annotations

from lanky.prelude import Int, Nat, Real

from loopty import Arr, Fin, Sigma, Where, kernel, reduce_sum


@kernel
def pairs(
    x: Arr[Fin[n], Real],
    f: Arr[Where[i: Fin[n], j: Fin[n], j < i], Real],
    e: Arr[Fin[n], Real],
):
    """A value per pair over the lower triangle, then each particle's share."""
    for i in f.dom:
        for j in f.dom[i]:
            f[i, j] = x[i] * x[j]
    for p in e.dom:
        e[p] = reduce_sum(f[p, j] for j in f.dom[p]) + reduce_sum(
            f[k, p] for k in e.dom if k > p
        )


@kernel
def diagonal(
    x: Arr[Fin[n], Real],
    f: Arr[Where[i: Fin[n], j: Fin[n], j < i], Real],
):
    """``f[i, i]``: a cell of the box around the triangle, and not of ``f``."""
    for i in x.dom:
        f[i, i] = x[i]


@kernel
def symmetric_product(
    a: Arr[Sigma[i: Fin[n], Fin[i + 1]], Real],
    x: Arr[Fin[n], Real],
    y: Arr[Fin[n], Real],
):
    """``y = A x`` for a symmetric ``A`` of which the lower half is stored."""
    for i in y.dom:
        y[i] = reduce_sum(a[i, j] * x[j] for j in a.dom[i]) + reduce_sum(
            a[k, i] * x[k] for k in y.dom if k > i
        )


@kernel
def band_product(
    b: Arr[Where[i: Fin[n], j: Fin[n], (i - j <= 1) & (j - i <= 1)], Real],
    x: Arr[Fin[n], Real],
    y: Arr[Fin[n], Real],
):
    """``y = B x`` for a tridiagonal ``B``, row by row over the band."""
    for i in b.dom:
        y[i] = reduce_sum(b[i, j] * x[j] for j in b.dom[i])


@kernel
def upper(
    x: Arr[Fin[n], Real],
    g: Arr[Where[i: Fin[n], j: Fin[n], j > i], Real],
):
    """The strict upper triangle, whose rows do not start at column 0."""
    for i in g.dom:
        for j in g.dom[i]:
            g[i, j] = x[j] - x[i]


@kernel
def two_pieces(
    u: Arr[Fin[n] + Fin[m], Real],
    x: Arr[Fin[n], Real],
    z: Arr[Fin[m], Real],
):
    """A union of two pieces, each written by a statement of its own."""
    for i in u.dom[0]:
        u[0, i] = 2.0 * x[i]
    for i in u.dom[1]:
        u[1, i] = z[i] + 1.0


@kernel
def every_piece(
    u: Arr[Fin[n] + Fin[m], Real],
    v: Arr[Fin[n] + Fin[m], Real],
):
    """The pieces walked by number: a Python loop in the trace too."""
    for p in u.dom:
        for i in u.dom[p]:
            v[p, i] = u[p, i] * (p + 1)


@kernel
def later_rows(
    x: Arr[Fin[n], Real],
    h: Arr[Where[i: Fin[n], j: Fin[n], (i >= 2) & (j < i)], Real],
    y: Arr[Fin[n], Real],
):
    """A fiber taken at every ``i``, of which the first two are not rows of ``h``."""
    for i in x.dom:
        y[i] = x[i]
        for j in h.dom[i]:
            h[i, j] = x[j]


@kernel
def even_columns(
    s: Arr[Where[i: Fin[n], j: Fin[n], j % 2 == 0], Real],
    x: Arr[Fin[n], Real],
):
    """Rows that skip every other column, which the packed layout cannot keep."""
    for i in s.dom:
        for j in s.dom[i]:
            s[i, j] = x[j]


@kernel
def offset_rows(
    k: Nat,
    x: Arr[Fin[n], Real],
    w: Arr[Where[i: Fin[n], i >= k], Real],
):
    """A domain cut by a scalar parameter, which the call gives."""
    for i in w.dom:
        w[i] = x[i] + 1.0


@kernel
def total(
    f: Arr[Where[i: Fin[n], j: Fin[n], j < i], Real],
    s: Arr[Fin[1], Real],
):
    """One reduction over both axes of the triangle, one a fiber of the other."""
    for r in s.dom:
        s[r] = reduce_sum(f[i, j] for i in f.dom for j in f.dom[i])


@kernel
def far_pairs(
    t: Arr[Sigma[i: Fin[n], Fin[i - 1]], Real],
    x: Arr[Fin[n], Real],
):
    """Pairs at least two apart, whose box ``n x (n - 2)`` has no shape at ``n = 1``."""
    for i in t.dom:
        for j in t.dom[i]:
            t[i, j] = x[i] - x[j]


@kernel
def after_a_short_piece(
    u: Arr[
        Where[i: Fin[n - 1], j: Fin[n - 1], j <= i] + Sigma[i: Fin[m], Fin[i + 1]],
        Real,
    ],
    z: Arr[Fin[m], Real],
):
    """A piece after one whose box, ``(n - 1) x (n - 1)``, is one cell at ``n = 0``.

    That piece is empty at ``n = 0``, and the second starts at the first cell.
    """
    for i in u.dom[1]:
        for j in u.dom[1, i]:
            u[1, i, j] = z[i] * 10.0 + z[j]
    for i in u.dom[0]:
        for j in u.dom[0, i]:
            u[0, i, j] = u[0, i, j] + 1.0


@kernel
def first_rows(
    k: Int,
    x: Arr[Fin[n], Real],
    w: Arr[Where[i: Fin[k], i < n], Real],
):
    """A domain whose bound is a scalar of the call, which could be negative."""
    for i in w.dom:
        w[i] = x[i]
