"""Programs whose postconditions are decided from what their calls store, or proved.

``tests/test_program_posts.py`` checks each: what the native runs find, what
isl decides, what is stated for Lean, and why the rest is neither.
"""

from __future__ import annotations

from lanky import theorem
from lanky.prelude import Fn, Int, Nat, Real

from loopty import Arr, Fin, kernel, program, reduce_sum

# {{{ integers: decided by isl


@kernel
def add_one(x: Arr[Fin[n], Int], y: Arr[Fin[n], Int]):
    """``y`` is ``x`` plus one."""
    for i in y.dom:
        y[i] = x[i] + 1


@kernel
def double(y: Arr[Fin[n], Int], z: Arr[Fin[n], Int]):
    """``z`` is twice ``y``."""
    for i in z.dom:
        z[i] = 2 * y[i]


@program
def steps(x, z) -> all(z[i] == 2 * x[i] + 2 for i in Fin[n]):
    """Two calls through an array the program makes: integers isl reads."""
    y = Arr.zeros_like(x)
    add_one(x, y)
    double(y, z)


# }}}


# {{{ reals: unfolded for Lean


@kernel
def twice(a: Arr[Fin[n], Real], b: Arr[Fin[n], Real]):
    """``b`` is twice ``a``."""
    for i in b.dom:
        b[i] = 2 * a[i]


@kernel
def plus(a: Arr[Fin[n], Real], b: Arr[Fin[n], Real], c: Arr[Fin[n], Real]):
    """``c`` is ``a`` plus ``b``."""
    for i in c.dom:
        c[i] = a[i] + b[i]


@program
def tripled(a, c) -> all(c[i] == 3 * a[i] for i in Fin[n]):
    """Reals, so each equation is an atom to isl, and ``ring`` adds them up."""
    b = Arr.zeros_like(a)
    twice(a, b)
    plus(a, b, c)


@program
def last_wins(a, b, c) -> all(c[i] == a[i] + b[i] for i in Fin[n]):
    """``c`` written twice: what the first call stored is not what is left."""
    twice(a, c)
    plus(a, b, c)


@program
def first_wins(a, b, c) -> all(c[i] == 2 * a[i] for i in Fin[n]):
    """The same calls, the claim about the first call's store: false."""
    twice(a, c)
    plus(a, b, c)


@program
def misnamed(a, c) -> all(c[i] == 3 * a[i] for i in Fin[m]):
    """A size no kernel it calls names."""
    b = Arr.zeros_like(a)
    twice(a, b)
    plus(a, b, c)


# }}}


# {{{ a cited theorem, used at the sections of the arrays


@theorem
def two_passes(
    n: Nat,
    A: Fn[Fin[n], Fn[Fin[n], Real]],
    B: Fn[Fin[n], Fn[Fin[n], Real]],
    h: Fn[Fin[n], Fn[Fin[n], Real]],
    k: Fin[n],
    j: Fin[n],
) -> sum(A(k)(b) * B(j)(a) * h(b)(a) for b in Fin[n] for a in Fin[n]) == sum(
    A(k)(b) * sum(B(j)(a) * h(b)(a) for a in Fin[n]) for b in Fin[n]
):
    """The operator ``A (x) B`` at a node is two passes, one along each axis.

    Its families are applied an index at a time, as the released lanky
    writes them; a family applied so stands for an array at a section, as one
    applied at several indices does.
    """


@kernel
def along_a(
    B: Arr[Fin[nq], Fin[n], Fin[n], Real],
    g: Arr[Fin[nq], Fin[n], Fin[n], Real],
    v: Arr[Fin[nq], Fin[n], Fin[n], Real],
):
    """Each field's own ``B`` applied along its last axis."""
    for q in v.dom:
        for b in v.dom[q]:
            for j in v.dom[q, b]:
                v[q, b, j] = reduce_sum(B[q, j, a] * g[q, b, a] for a in B.dom[q, j])


@kernel
def along_b(
    A: Arr[Fin[nq], Fin[n], Fin[n], Real],
    v: Arr[Fin[nq], Fin[n], Fin[n], Real],
    out: Arr[Fin[nq], Fin[n], Fin[n], Real],
):
    """Each field's own ``A`` applied along its first axis."""
    for q in out.dom:
        for k in out.dom[q]:
            for j in out.dom[q, k]:
                out[q, k, j] = reduce_sum(A[q, k, b] * v[q, b, j] for b in A.dom[q, k])


@program(uses=[two_passes])
def dense(A, B, g, out) -> all(
    out[q, k, j]
    == sum(A[q, k, b] * B[q, j, a] * g[q, b, a] for b in Fin[n] for a in Fin[n])
    for q in Fin[nq]
    for k in Fin[n]
    for j in Fin[n]
):
    """Two passes over each field ``q``: its dense operator, by ``two_passes``."""
    v = Arr.zeros_like(g)
    along_a(B, g, v)
    along_b(A, v, out)


@program(uses=[two_passes])
def transposed(A, B, g, out) -> all(
    out[q, k, j]
    == sum(A[q, b, k] * B[q, j, a] * g[q, b, a] for b in Fin[n] for a in Fin[n])
    for q in Fin[nq]
    for k in Fin[n]
    for j in Fin[n]
):
    """The same passes, claimed to apply ``A`` transposed: false, refuted by a run."""
    v = Arr.zeros_like(g)
    along_a(B, g, v)
    along_b(A, v, out)


# }}}
