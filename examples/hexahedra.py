"""grudge's weak gradient and divergence on hexahedra: three one-axis passes, checked.

A discontinuous Galerkin code spends much of its time in two operators on each
element: the weak gradient of a field and the weak divergence of a flux.
grudge PR #418 (Alvey-Blanco, "Operator construction, application rewrite",
as of a-alveyblanc/grudge 4d103bb) applies them on hexahedra by sum
factorization, which this file mirrors:

* each weak derivative is ``weak_local_d_dx``
  (`op.py#L711-L821
  <https://github.com/a-alveyblanc/grudge/blob/4d103bbbb31bfdb175f8997ae53a76a6cb8c14dc/grudge/op.py#L711-L821>`__):
  the field times the metric terms ``J dr_d/dx_x``, one einsum
  ``"rej,ej->rej"``, then, for each reference direction ``d``, a tensor-product
  operator applied by ``_apply_operator_to_group``
  (`op.py#L185-L267
  <https://github.com/a-alveyblanc/grudge/blob/4d103bbbb31bfdb175f8997ae53a76a6cb8c14dc/grudge/op.py#L185-L267>`__),
  one einsum per reference axis, and the three directions summed;
* the operator of direction ``d`` has a factor per axis, the weak stiffness
  ``S^T`` on axis ``d`` and the mass ``M`` on the other two, which
  ``make_stiffness_t_operator`` builds
  (`bilinear_forms.py#L809
  <https://github.com/a-alveyblanc/grudge/blob/4d103bbbb31bfdb175f8997ae53a76a6cb8c14dc/grudge/bilinear_forms.py#L809>`__);
  :func:`factors` computes the same matrices with numpy;
* the weak gradient is ``weak_local_d_dx`` along ``x``, ``y`` and ``z``, and
  the weak divergence their sum over the flux's components
  (`op.py#L592-L636
  <https://github.com/a-alveyblanc/grudge/blob/4d103bbbb31bfdb175f8997ae53a76a6cb8c14dc/grudge/op.py#L592-L636>`__).
  Here the divergence sums the metric terms first, ``w_d = sum_x J dr_d/dx_x
  f_x``, and then applies each direction's operator once, which by linearity
  is the same sum in three passes where grudge makes nine;
  ``benchmarks/hexahedra.py`` compares both operators with grudge's own.

Where a node is stored. grudge keeps a field as a DOFArray, one row of
``N**3`` nodes per element, each row reshaped by modepy in Fortran order
(`tools.py#L427-L489
<https://github.com/inducer/modepy/blob/01ac7955d30e248e460ccd751d40da4b0ff3e3e6/modepy/tools.py#L427-L489>`__),
so node ``(i, j, k)``, ``i`` along ``r``, sits at ``i + N j + N**2 k``; #418
tags its passes' outputs with loopy's ``dim_tags`` ``N3,N0,N1,N2`` to say so
(`array_context.py#L129-L151
<https://github.com/a-alveyblanc/grudge/blob/4d103bbbb31bfdb175f8997ae53a76a6cb8c14dc/grudge/array_context.py#L129-L151>`__).
loopty's arrays are row-major, so the kernels below read node ``(i, j, k)``
of element ``e`` as ``u[e, k, j, i]``: the same buffer, the reference axes
written slowest first. That the three are one layout is a fact for every
order (``dof_array``): isl, which cannot multiply an index by an unknown
order, decides it at ``n`` from 1 to 6, and Lean proves it for every ``n``.

Run this file three ways.

``python examples/hexahedra.py``
    Applies both operators natively, as the math and as passes, and compares
    them with the dense Kronecker operators grudge's escape hatch builds.
    Then derives the passes of each operator from its math, fuses the passes
    as deep as each pair allows, refuses one step deeper with its pair, keeps
    what passes between them in the cells live at once, and compares the
    compiled run with the native one; and lists the facts of the three
    launches a device runs.

``lanky check examples/hexahedra.py``
    The theorem, every kernel's facts, the programs', with what each program
    leaves, and the layout's.

``loopty run examples/hexahedra.py``
    Each kernel, both programs and the schedules, compiled and compared; the
    dense kernels factored, and the gradient's fused and contracted, among
    them.

What is worth reading here
--------------------------

*Every order at once.* The kernels are written for ``n`` nodes per axis, so
each fact about them and their schedules is decided for every order, the
Kronecker theorem is proved for every ``n``, and so are the layouts: isl tests
them at ``n`` from 1 to 6, as ``n * j`` is not linear, and Lean proves them.

*The math, and the passes derived from it.* ``weak_grad`` is the dense
kernel, as grudge's ``weak_local_grad`` says it: the field times the metric
term ``J dr_d/dx_x``, then for each direction ``d`` the Kronecker product of
its three tables applied to that, the three directions summed. ``factored``
rewrites ``grad`` by ``sum_factorization`` and keeps the pass along ``r``
and the pass along ``s`` in arrays of their own, ``r_pass`` and ``s_pass``;
``fused`` runs the two passes a slab of ``c`` at a time, and ``contracted``
keeps the ``r`` pass one slab at a time. The ledger says, proved by Lean
with Mathlib from what the derived passes store and the theorem at each
direction's tables, that they leave in ``grad`` what ``weak_grad``'s
statement says; the kept arrays' accesses in bounds, their writes distinct,
and each cell stored before it is read; and that the compiled ``factored``
agrees with the native dense kernel. ``weak_div`` is the divergence the same
way, grudge's ``weak_local_div``: the flux along each direction first,
``sum_x J dr_d/dx_x f_x``, then the Kronecker product of the direction's
tables applied to it, the three directions summed; ``factored_div`` derives
its passes by the same rewrite, and the ledger says the same of them.

*What the programs leave.* ``passes`` and ``div_passes`` are the same four
passes written by hand, as one call each, the gradient's and the
divergence's. ``passes`` claims that ``grad`` is the dense operator of each
direction applied to the field times its metric term, the three summed, and
``div_passes`` the same of the flux. Lean proves each, with Mathlib, from
what the four calls store and ``sum_factorization``, taken at each
direction's sections: ``Fr[3*x + 0]`` for the theorem's ``Fr`` and ``w[3*x
+ 0, e]`` for its ``u``. Without Mathlib the claims are tested on the
programs' runs, compared at the class of ``grad``, since the passes sum in
another order than the dense operator does. They are what the derived
passes are timed against, by ``benchmarks/hexahedra.py``.

*How deep each fusion goes.* In ``passes``, ``along_s`` reads a whole column
of what ``along_r`` stored in the same ``k``-slab, so the two fuse on ``(q,
e, k)`` and no deeper, ``fuse(passes.along_r.k, passes.along_s.k)``: a
fusion names the loop of each call it goes down to. The metric product
fuses with ``along_r`` on every loop but the row's, down to ``j``. The
refusal of the deeper fusion names the pair it would run backwards. The
passes derived from the math fuse the same way, on ``(x, d, e, c)``,
``fuse(factored.r_pass.c, factored.s_pass.c)``.

*What passes between the passes.* Fused so, ``passes`` reads ``w`` one row
at a time and ``v`` one slab at a time, so ``contract`` keeps them in ``n``
and ``n * n`` cells (``passes_contracted``), with the no-clobber fact for
each; ``contracted`` keeps the derived ``r_pass`` a slab at a time.

*On a device.* Every pass reads a column other work items stored, and
nothing inside one kernel orders two work items through global memory, so
each pass is a launch of its own, every node on a work item:
``passes_on_a_device``, which ``benchmarks/hexahedra.py`` times.
"""

from __future__ import annotations

from typing import Any

import numpy as np
from lanky import theorem
from lanky.prelude import Fn, Nat, Real

from loopty import Arr, Fin, IllegalCast, Schedule, kernel, layout, program, reduce_sum

#: The order of the demo: four Gauss-Lobatto-Legendre nodes per axis.
ORDER = 3
N = ORDER + 1

#: Two hexahedra; the timings take 48**3.
ELEMENTS = 2


# {{{ the theorem: three passes are the Kronecker operator, at every order


@theorem
def sum_factorization(
    n: Nat,
    Fr: Fn[Fin[n], Fin[n], Real],
    Fs: Fn[Fin[n], Fin[n], Real],
    Ft: Fn[Fin[n], Fin[n], Real],
    u: Fn[Fin[n], Fin[n], Fin[n], Real],
    k: Fin[n],
    j: Fin[n],
    i: Fin[n],
) -> sum(
    Ft[k, c] * Fs[j, b] * Fr[i, a] * u[c, b, a]
    for c in Fin[n]
    for b in Fin[n]
    for a in Fin[n]
) == sum(
    Ft[k, c]
    * sum(Fs[j, b] * sum(Fr[i, a] * u[c, b, a] for a in Fin[n]) for b in Fin[n])
    for c in Fin[n]
):
    """The Kronecker operator ``Ft (x) Fs (x) Fr`` at a node is three passes.

    The left side is the dense operator at the row ``(k, j, i)``, a sum over
    all ``n**3`` columns; the right side applies ``Fr`` along ``r``, then
    ``Fs`` along ``s``, then ``Ft`` along ``t``, ``n`` terms at a time, which
    is what ``along_r``, ``along_s`` and the last pass compute.
    """


# }}}


# {{{ where a node is stored


@layout
def dof_array(e: Fin[ne], i: Fin[n], j: Fin[n], k: Fin[n]) -> Fin[ne * n**3]:
    """grudge's DOFArray: a row per element, its nodes reshaped by modepy in F order."""
    return n**3 * e + (i + n * j + n**2 * k)


#: grudge #418 writes the same layout as loopy's dim_tags on (e, i, j, k).
as_dim_tags = dof_array.same_as("N3,N0,N1,N2")


@layout
def row_major(e: Fin[ne], i: Fin[n], j: Fin[n], k: Fin[n]) -> Fin[ne * n**3]:
    """loopty's ``u[e, k, j, i]``: row-major, the reference axes slowest first."""
    return ((e * n + k) * n + j) * n + i


#: The kernels' arrays are grudge's DOFArray.
as_row_major = dof_array.same_as(row_major)


# }}}


# {{{ the kernels: the metric, a pass per reference axis, and the sum


@kernel
def metric(
    G: Arr[Fin[9], Fin[ne], Fin[n], Fin[n], Fin[n], Real],
    u: Arr[Fin[ne], Fin[n], Fin[n], Fin[n], Real],
    w: Arr[Fin[9], Fin[ne], Fin[n], Fin[n], Fin[n], Real],
):
    """The field times each metric term ``J dr_d/dx_x``, field ``q = 3 x + d``."""
    for q, e, k, j, i in w.dom:
        w[q, e, k, j, i] = G[q, e, k, j, i] * u[e, k, j, i]


@kernel
def contravariant(
    G: Arr[Fin[9], Fin[ne], Fin[n], Fin[n], Fin[n], Real],
    f: Arr[Fin[3], Fin[ne], Fin[n], Fin[n], Fin[n], Real],
    w: Arr[Fin[3], Fin[ne], Fin[n], Fin[n], Fin[n], Real],
):
    """The flux along each reference direction ``d``: ``sum_x J dr_d/dx_x f_x``."""
    for d, e, k, j, i in w.dom:
        w[d, e, k, j, i] = reduce_sum(
            G[3 * x + d, e, k, j, i] * f[x, e, k, j, i] for x in Fin[3]
        )


@kernel
def along_r(
    Fr: Arr[Fin[nq], Fin[n], Fin[n], Real],
    w: Arr[Fin[nq], Fin[ne], Fin[n], Fin[n], Fin[n], Real],
    v: Arr[Fin[nq], Fin[ne], Fin[n], Fin[n], Fin[n], Real],
):
    """Each field's own matrix applied along ``r``, the first reference axis."""
    for q, e, k, j, i in v.dom:
        v[q, e, k, j, i] = reduce_sum(
            Fr[q, i, a] * w[q, e, k, j, a] for a in Fr.dom[q, i]
        )


@kernel
def along_s(
    Fs: Arr[Fin[nq], Fin[n], Fin[n], Real],
    v: Arr[Fin[nq], Fin[ne], Fin[n], Fin[n], Fin[n], Real],
    vs: Arr[Fin[nq], Fin[ne], Fin[n], Fin[n], Fin[n], Real],
):
    """Each field's own matrix applied along ``s``, the second reference axis."""
    for q, e, k, j, i in vs.dom:
        vs[q, e, k, j, i] = reduce_sum(
            Fs[q, j, b] * v[q, e, k, b, i] for b in Fs.dom[q, j]
        )


@kernel
def gradient_along_t(
    Ft: Arr[Fin[9], Fin[n], Fin[n], Real],
    vs: Arr[Fin[9], Fin[ne], Fin[n], Fin[n], Fin[n], Real],
    grad: Arr[Fin[3], Fin[ne], Fin[n], Fin[n], Fin[n], Real],
):
    """Along ``t``, the three directions of each component ``x`` summed."""
    for x, e, k, j, i in grad.dom:
        grad[x, e, k, j, i] = reduce_sum(
            reduce_sum(
                Ft[3 * x + d, k, c] * vs[3 * x + d, e, c, j, i]
                for c in Ft.dom[3 * x + d, k]
            )
            for d in Fin[3]
        )


@kernel
def divergence_along_t(
    Ft: Arr[Fin[3], Fin[n], Fin[n], Real],
    vs: Arr[Fin[3], Fin[ne], Fin[n], Fin[n], Fin[n], Real],
    div: Arr[Fin[ne], Fin[n], Fin[n], Fin[n], Real],
):
    """Along ``t``, the three directions summed."""
    for e, k, j, i in div.dom:
        div[e, k, j, i] = reduce_sum(
            reduce_sum(
                Ft[d, k, c] * vs[d, e, c, j, i] for c in Ft.dom[d, k]
            )
            for d in Fin[3]
        )


@program(uses=[sum_factorization])
def passes(G, u, Fr, Fs, Ft, grad) -> all(
    grad[x, e, k, j, i]
    == sum(
        Ft[3 * x + d, k, c]
        * Fs[3 * x + d, j, b]
        * Fr[3 * x + d, i, a]
        * G[3 * x + d, e, c, b, a]
        * u[e, c, b, a]
        for d in Fin[3]
        for c in Fin[n]
        for b in Fin[n]
        for a in Fin[n]
    )
    for x in Fin[3]
    for e in Fin[ne]
    for k in Fin[n]
    for j in Fin[n]
    for i in Fin[n]
):
    """grudge's ``weak_local_grad``: the metric, then a pass per reference axis.

    What it leaves in ``grad`` is the dense Kronecker operator of each
    direction ``d`` applied to ``u`` times the metric term ``J dr_d/dx_x``,
    the three directions summed, which is what grudge's escape hatch
    computes. The passes are that operator by ``sum_factorization``.
    """
    w = Arr.zeros_like(G)
    v = Arr.zeros_like(G)
    vs = Arr.zeros_like(G)
    metric(G, u, w)
    along_r(Fr, w, v)
    along_s(Fs, v, vs)
    gradient_along_t(Ft, vs, grad)


@program(uses=[sum_factorization])
def div_passes(G, f, Fr, Fs, Ft, div) -> all(
    div[e, k, j, i]
    == sum(
        Ft[d, k, c]
        * Fs[d, j, b]
        * Fr[d, i, a]
        * sum(G[3 * x + d, e, c, b, a] * f[x, e, c, b, a] for x in Fin[3])
        for d in Fin[3]
        for c in Fin[n]
        for b in Fin[n]
        for a in Fin[n]
    )
    for e in Fin[ne]
    for k in Fin[n]
    for j in Fin[n]
    for i in Fin[n]
):
    """grudge's ``weak_local_div``, the metric terms summed first.

    What it leaves in ``div`` is the dense operator of each direction ``d``
    applied to the flux along it, ``sum_x J dr_d/dx_x f_x``, the three
    directions summed.
    """
    w = Arr.zeros_like(f)
    v = Arr.zeros_like(f)
    vs = Arr.zeros_like(f)
    contravariant(G, f, w)
    along_r(Fr, w, v)
    along_s(Fs, v, vs)
    divergence_along_t(Ft, vs, div)


# }}}


# {{{ the math: the dense weak gradient and divergence, as grudge says them


@kernel
def weak_grad(
    G: Arr[Fin[3], Fin[3], Fin[ne], Fin[n], Fin[n], Fin[n], Real],
    u: Arr[Fin[ne], Fin[n], Fin[n], Fin[n], Real],
    Fr: Arr[Fin[3], Fin[n], Fin[n], Real],
    Fs: Arr[Fin[3], Fin[n], Fin[n], Real],
    Ft: Arr[Fin[3], Fin[n], Fin[n], Real],
    w: Arr[Fin[3], Fin[3], Fin[ne], Fin[n], Fin[n], Fin[n], Real],
    grad: Arr[Fin[3], Fin[ne], Fin[n], Fin[n], Fin[n], Real],
):
    """grudge's weak_local_grad, as the math says it.

    ``w[x, d]`` is the field times the metric term ``G[x, d] = J dr_d/dx_x``.
    Component ``x`` of the gradient is, for each direction ``d``, the
    Kronecker product of ``Ft[d]``, ``Fs[d]`` and ``Fr[d]`` applied to ``w[x,
    d]``; the three directions summed.
    """
    for x, d, e, k, j, i in w.dom:
        w[x, d, e, k, j, i] = G[x, d, e, k, j, i] * u[e, k, j, i]
    for x, e, k, j, i in grad.dom:
        grad[x, e, k, j, i] = reduce_sum(
            reduce_sum(
                Ft[d, k, c] * Fs[d, j, b] * Fr[d, i, a] * w[x, d, e, c, b, a]
                for c in Ft.dom[d, k]
                for b in Fs.dom[d, j]
                for a in Fr.dom[d, i]
            )
            for d in Fin[3]
        )


@kernel
def weak_div(
    G: Arr[Fin[3], Fin[3], Fin[ne], Fin[n], Fin[n], Fin[n], Real],
    f: Arr[Fin[3], Fin[ne], Fin[n], Fin[n], Fin[n], Real],
    Fr: Arr[Fin[3], Fin[n], Fin[n], Real],
    Fs: Arr[Fin[3], Fin[n], Fin[n], Real],
    Ft: Arr[Fin[3], Fin[n], Fin[n], Real],
    w: Arr[Fin[3], Fin[ne], Fin[n], Fin[n], Fin[n], Real],
    div: Arr[Fin[ne], Fin[n], Fin[n], Fin[n], Real],
):
    """grudge's weak_local_div, as the math says it.

    ``w[d]`` is the flux along the direction ``d``, ``sum_x J dr_d/dx_x
    f_x``, the metric terms ``G[x, d]`` summed first. The divergence is, for
    each direction ``d``, the Kronecker product of ``Ft[d]``, ``Fs[d]`` and
    ``Fr[d]`` applied to ``w[d]``; the three directions summed.
    """
    for d, e, k, j, i in w.dom:
        w[d, e, k, j, i] = reduce_sum(
            G[x, d, e, k, j, i] * f[x, e, k, j, i] for x in Fin[3]
        )
    for e, k, j, i in div.dom:
        div[e, k, j, i] = reduce_sum(
            reduce_sum(
                Ft[d, k, c] * Fs[d, j, b] * Fr[d, i, a] * w[d, e, c, b, a]
                for c in Ft.dom[d, k]
                for b in Fs.dom[d, j]
                for a in Fr.dom[d, i]
            )
            for d in Fin[3]
        )


# }}}


# {{{ the schedules

#: Sum factorization, as grudge applies it: the pass along r, then along s, each kept.
factored = Schedule(weak_grad).rewrite(
    weak_grad.grad, by=sum_factorization, keep=("r_pass", "s_pass")
)

#: The two passes together, a slab of c at a time.
fused = factored.fuse(factored.r_pass.c, factored.s_pass.c)

#: The r pass kept one slab at a time.
contracted = fused.contract(factored.r_pass)

#: The divergence's passes, derived as the gradient's are.
factored_div = Schedule(weak_div).rewrite(
    weak_div.div, by=sum_factorization, keep=("r_pass", "s_pass")
)

#: The passes written by hand: the metric with the pass along r, on every loop
#: but the row's, down to j; then the two passes on (q, e, k), a slab at a time.
passes_fused = (
    Schedule(passes)
    .fuse(passes.metric.j, passes.along_r.j)
    .fuse(passes.along_r.k, passes.along_s.k)
)

#: The same, with w kept one row at a time and v one slab at a time.
passes_contracted = passes_fused.contract(passes.w).contract(passes.v)


@program
def metric_along_r(G, u, Fr, v):
    """The first launch on a device: the metric product, then the pass along r."""
    w = Arr.zeros_like(G)
    metric(G, u, w)
    along_r(Fr, w, v)


@program
def contravariant_along_r(G, f, Fr, v):
    """The divergence's first launch: the flux along each direction, then along r."""
    w = Arr.zeros_like(v)
    contravariant(G, f, w)
    along_r(Fr, w, v)


def passes_on_a_device(operator: str = "grad", target: str = "c") -> list[Schedule]:
    """An operator as three launches, every node of an element on a work item.

    The element is the work group, ``e`` on ``g.0``, and its nodes the work
    items, ``k, j, i`` on ``l.2, l.1, l.0``; the fields run in sequence on
    each. The metric is computed where the pass along ``r`` reads it,
    ``substitute(first.w)``, so nothing is stored between them, and the
    loops of the first launch are the pass's, ``first.along_r.e`` and so
    on. Each launch finishes before the next starts, so a pass may read
    what another work item of the last one stored. ``benchmarks/hexahedra.py``
    times them, and other shapes of them, retargeted onto OpenCL.
    """
    first, last = {
        "grad": (metric_along_r, gradient_along_t),
        "div": (contravariant_along_r, divergence_along_t),
    }[operator]

    def on_nodes(schedule: Schedule, loops: Any) -> Schedule:
        """Each element a work group, each node of it a work item."""
        return schedule.tag(
            {loops.e: "g.0", loops.k: "l.2", loops.j: "l.1", loops.i: "l.0"}
        )

    return [
        on_nodes(Schedule(first, target=target).substitute(first.w), first.along_r),
        on_nodes(Schedule(along_s, target=target), along_s),
        on_nodes(Schedule(last, target=target), last),
    ]


# }}}


# {{{ the operators' matrices, the inputs, and the dense operators


def factors(order: int = ORDER) -> tuple[np.ndarray, np.ndarray]:
    """The weak stiffness ``S^T`` and the mass ``M`` on ``order + 1`` GLL nodes.

    ``M[i, j]`` is the integral of the Lagrange polynomials of nodes ``i`` and
    ``j`` on ``[-1, 1]``, exactly, and ``S^T[i, j]`` that of the derivative of
    the ``i``-th times the ``j``-th, as ``make_stiffness_t_operator`` builds
    them with modepy.
    """
    n = order + 1
    legendre = np.polynomial.legendre.Legendre
    interior = legendre.basis(n - 1).deriv().roots().real if n > 2 else []
    nodes = np.concatenate([[-1.0], np.sort(interior), [1.0]])
    # orthonormal Legendre polynomials and their derivatives at the nodes
    scale = np.sqrt((2 * np.arange(n) + 1) / 2)
    V = np.stack([legendre.basis(m)(nodes) for m in range(n)], axis=1) * scale
    Vr = np.stack([legendre.basis(m).deriv()(nodes) for m in range(n)], axis=1) * scale
    inverse = np.linalg.inv(V)
    mass = inverse.T @ inverse
    D = Vr @ inverse
    return D.T @ mass, mass


def tables(directions: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Each field's matrix along ``r``, ``s`` and ``t``: ``S^T`` on its direction.

    Field ``q`` is of direction ``q % 3``, as ``3 x + d`` is of ``d``, so the
    gradient takes nine fields and the divergence three.
    """
    stiffness_t, mass = factors()
    along = [
        np.stack([stiffness_t if q % 3 == axis else mass for q in range(directions)])
        for axis in range(3)
    ]
    return along[0], along[1], along[2]


def inputs(elements: int = ELEMENTS, seed: int = 0) -> dict:
    """Random metric terms and fields on ``elements`` hexahedra, and the tables.

    The dense kernel's metric is the passes' nine fields, field ``q = 3 x +
    d`` its ``G[x, d]``, and its tables the three directions' own, where the
    passes' are each field's.
    """
    rng = np.random.default_rng(seed)
    nodes = (elements, N, N, N)
    Fr, Fs, Ft = tables(9)
    Dr, Ds, Dt = tables(3)
    G = rng.normal(size=(9, *nodes))
    u = rng.normal(size=nodes)
    f = rng.normal(size=(3, *nodes))
    return {
        "weak_grad": {
            "G": Arr.from_numpy(G.reshape(3, 3, *nodes)),
            "u": Arr.from_numpy(u),
            "Fr": Arr.from_numpy(Dr),
            "Fs": Arr.from_numpy(Ds),
            "Ft": Arr.from_numpy(Dt),
            "w": Arr.zeros((3, 3, *nodes)),
            "grad": Arr.zeros((3, *nodes)),
        },
        "passes": {
            "G": Arr.from_numpy(G),
            "u": Arr.from_numpy(u),
            "Fr": Arr.from_numpy(Fr),
            "Fs": Arr.from_numpy(Fs),
            "Ft": Arr.from_numpy(Ft),
            "grad": Arr.zeros((3, *nodes)),
        },
        "weak_div": {
            "G": Arr.from_numpy(G.reshape(3, 3, *nodes)),
            "f": Arr.from_numpy(f),
            "Fr": Arr.from_numpy(Dr),
            "Fs": Arr.from_numpy(Ds),
            "Ft": Arr.from_numpy(Dt),
            "w": Arr.zeros((3, *nodes)),
            "div": Arr.zeros(nodes),
        },
        "div_passes": {
            "G": Arr.from_numpy(G),
            "f": Arr.from_numpy(f),
            "Fr": Arr.from_numpy(Dr),
            "Fs": Arr.from_numpy(Ds),
            "Ft": Arr.from_numpy(Dt),
            "div": Arr.zeros(nodes),
        },
    }


def dense(data: dict) -> tuple[np.ndarray, np.ndarray]:
    """Both operators as grudge's escape hatch applies them: ``"ij,ej->ei"``.

    The operator of direction ``d`` is ``kron(Ft, kron(Fs, Fr))`` with ``S^T``
    on axis ``d``, applied to each element's nodes in modepy's order, which
    is ``u[e, k, j, i]`` flattened row by row.
    """
    stiffness_t, mass = factors()

    def along(axis: int, d: int) -> np.ndarray:
        return stiffness_t if axis == d else mass

    ops = [np.kron(along(2, d), np.kron(along(1, d), along(0, d))) for d in range(3)]
    shape = data["passes"]["u"].numpy().shape
    G = data["passes"]["G"].numpy().reshape(3, 3, shape[0], -1)
    u = data["passes"]["u"].numpy().reshape(shape[0], -1)
    f = data["div_passes"]["f"].numpy().reshape(3, shape[0], -1)
    grad = np.stack([sum((G[x, d] * u) @ ops[d].T for d in range(3)) for x in range(3)])
    div = sum((G[x, d] * f[x]) @ ops[d].T for x in range(3) for d in range(3))
    return grad.reshape(3, *shape), div.reshape(shape)


def example_inputs() -> dict:
    """The inputs ``loopty run`` gives each kernel, and the programs."""
    data = inputs()
    grad_in, div_in = data["passes"], data["div_passes"]
    nodes = grad_in["u"].numpy().shape
    nine = np.random.default_rng(1).normal(size=(9, *nodes))
    three = np.random.default_rng(2).normal(size=(3, *nodes))
    return {
        "metric": {"G": grad_in["G"], "u": grad_in["u"], "w": Arr.zeros((9, *nodes))},
        "contravariant": {
            "G": div_in["G"],
            "f": div_in["f"],
            "w": Arr.zeros((3, *nodes)),
        },
        "along_r": {
            "Fr": grad_in["Fr"],
            "w": Arr.from_numpy(nine),
            "v": Arr.zeros((9, *nodes)),
        },
        "along_s": {
            "Fs": grad_in["Fs"],
            "v": Arr.from_numpy(nine),
            "vs": Arr.zeros((9, *nodes)),
        },
        "gradient_along_t": {
            "Ft": grad_in["Ft"],
            "vs": Arr.from_numpy(nine),
            "grad": Arr.zeros((3, *nodes)),
        },
        "divergence_along_t": {
            "Ft": div_in["Ft"],
            "vs": Arr.from_numpy(three),
            "div": Arr.zeros(nodes),
        },
        "weak_grad": inputs()["weak_grad"],
        "passes": inputs()["passes"],
        "weak_div": inputs()["weak_div"],
        "div_passes": inputs()["div_passes"],
        "metric_along_r": {
            "G": grad_in["G"],
            "u": grad_in["u"],
            "Fr": grad_in["Fr"],
            "v": Arr.zeros((9, *nodes)),
        },
        "contravariant_along_r": {
            "G": div_in["G"],
            "f": div_in["f"],
            "Fr": div_in["Fr"],
            "v": Arr.zeros((3, *nodes)),
        },
    }


# }}}


def _settled(facts: list) -> str:
    """``all decided by isl``, or ``4 proved by lean, 1 decided by isl``."""
    counts: dict[tuple[str, str], int] = {}
    for fact in facts:
        key = (fact.status.value, fact.decided_by)
        counts[key] = counts.get(key, 0) + 1
    if len(counts) == 1:
        ((status, by),) = counts
        return f"all {status} by {by}"
    return ", ".join(
        f"{count} {status} by {by}" for (status, by), count in counts.items()
    )


def main() -> int:
    """Natively against the dense operators, the fusions and windows, and the device."""
    from loopty.executor import LoopyExecutor, differences
    from loopty.obligations import prove

    data = inputs()
    passes(**data["passes"])
    weak_grad(**data["weak_grad"])
    weak_div(**data["weak_div"])
    div_passes(**data["div_passes"])
    grad, div = dense(data)
    scale = max(np.abs(grad).max(), np.abs(div).max())
    gap = max(
        np.abs(data["passes"]["grad"].numpy() - grad).max(),
        np.abs(data["weak_grad"]["grad"].numpy() - grad).max(),
        np.abs(data["weak_div"]["div"].numpy() - div).max(),
        np.abs(data["div_passes"]["div"].numpy() - div).max(),
    )
    # the bound, not the difference, whose digits are the BLAS's and so the
    # machine's
    native = bool(gap <= 1e-13 * scale)
    print(
        f"order {ORDER}, {ELEMENTS} hexahedra: the dense kernels and the passes are "
        f"the dense Kronecker operators, to 1e-13 of the largest value: {native}"
    )
    print()

    print(f"derived from the math: {contracted!r}")
    for name, window in contracted.windows.items():
        print(f"  {name} kept in {window.placement}")
    derived = True
    for fact in contracted.facts():
        # Lean, where Mathlib is, proves what the derived passes leave
        fact = prove(fact) if fact.kind == "rewrite" else fact
        derived = derived and fact.status.value != "refuted"
        print(f"  {fact.status.value:8} {fact.decided_by or '-':11} {fact.statement}")
    kept = LoopyExecutor().differential(weak_grad, contracted, inputs()["weak_grad"])
    for line in differences(kept):
        print(f"  {line}")
    print()

    print(f"the divergence, derived the same way: {factored_div!r}")
    for fact in factored_div.facts():
        if fact.kind != "rewrite":
            # its other facts are the gradient's, said above
            continue
        fact = prove(fact)
        derived = derived and fact.status.value != "refuted"
        print(f"  {fact.status.value:8} {fact.decided_by or '-':11} {fact.statement}")
    kept_div = LoopyExecutor().differential(
        weak_div, factored_div, inputs()["weak_div"]
    )
    for line in differences(kept_div):
        print(f"  {line}")
    print()

    try:
        Schedule(passes).fuse(passes.along_r.i, passes.along_s.i)
    except IllegalCast as exc:
        print(f"refused: {exc}")
    print()

    print(f"accepted, by hand: {passes_contracted!r}")
    for name, window in passes_contracted.windows.items():
        print(f"  {name} kept in {window.placement}")
    for fact in passes_contracted.facts():
        print(f"  {fact.status.value:8} {fact.decided_by:4} {fact.statement}")
    by_hand = LoopyExecutor().differential(
        passes, passes_contracted, inputs()["passes"]
    )
    for line in differences(by_hand):
        print(f"  {line}")
    print()

    print(
        "on a device, the weak gradient as three launches, every node on a work item:"
    )
    legal = True
    for schedule in passes_on_a_device():
        # built for C here, which has no work items; the timing driver
        # retargets them onto OpenCL, where they build. Where Lean is
        # installed it proves each cast, that no dependence joins two work
        # items included
        cast = [prove(fact) for fact in schedule.facts() if fact.kind != "buildable"]
        settled = ("decided", "proved")
        legal = legal and all(fact.status.value in settled for fact in cast)
        print(
            f"  {schedule.term.name}: {'.'.join(schedule.history)}: "
            f"{len(cast)} facts, {_settled(cast)}"
        )
    ok = kept.status.value == kept_div.status.value == by_hand.status.value == "tested"
    return 0 if native and derived and ok and legal else 1


if __name__ == "__main__":
    raise SystemExit(main())
