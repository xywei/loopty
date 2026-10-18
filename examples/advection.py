"""grudge's advection right-hand side on hexahedra: volume, faces and lift, checked.

The right-hand side of ``du/dt + v . grad u = 0`` on a hexahedral DG mesh is
what grudge's ``WeakAdvectionOperator`` computes (as of a-alveyblanc/grudge
4d103bb, grudge PR #418), and what this file states, kernel by kernel:

* ``WeakAdvectionOperator.operator``
  (`models/advection.py#L143-L184
  <https://github.com/a-alveyblanc/grudge/blob/4d103bbbb31bfdb175f8997ae53a76a6cb8c14dc/grudge/models/advection.py#L143-L184>`__)
  is ``inverse_mass(v . weak_local_grad(u) - face_mass(flux))``, the weak
  gradient through grudge's metric terms ``J dr_d/dx_x``
  (`metrics.py#L522
  <https://github.com/a-alveyblanc/grudge/blob/4d103bbbb31bfdb175f8997ae53a76a6cb8c14dc/grudge/geometry/metrics.py#L522>`__);
* the flux is the upwind flux, ``where(v.n > 0, u_inside, u_outside) * v.n``
  at every face node (``advection_weak_flux``,
  `models/advection.py#L42-L61
  <https://github.com/a-alveyblanc/grudge/blob/4d103bbbb31bfdb175f8997ae53a76a6cb8c14dc/grudge/models/advection.py#L42-L61>`__),
  the outside value read from the neighbor across the face;
* ``face_mass`` puts each face's ``sJ * flux`` back on the element's nodes,
  the face's own mass on its two axes and the face's end of the element on
  the third (`op.py#L1368-L1545
  <https://github.com/a-alveyblanc/grudge/blob/4d103bbbb31bfdb175f8997ae53a76a6cb8c14dc/grudge/op.py#L1368-L1545>`__),
  and ``inverse_mass`` applies the inverse of the mass on every axis and
  divides by the Jacobian ``J`` (`op.py#L1118-L1181
  <https://github.com/a-alveyblanc/grudge/blob/4d103bbbb31bfdb175f8997ae53a76a6cb8c14dc/grudge/op.py#L1118-L1181>`__).

The inverse mass is pushed inward, as grudge PR #362 rewrites it: on every
axis but one it meets a mass and cancels, so the volume term is one pass of
``D = M^-1 S^T`` along each axis, and a face's term is the column ``L =
M^-1 e`` of the face's end along its normal, times the face's flux. The two
theorems are that rearrangement, for every order; that ``M^-1`` is the
mass's inverse is a property of the tables, which ``main`` checks. The same
kernels take the tables of the GLL quadrature's mass, which Alvey-Blanco's
matched right-hand side and libParanumal use: there ``L`` is a face node's
own weight and nothing else, and the lift adds each face node's flux into
the node it sits on. A third theorem says that lift is the transpose of the
face gather the flux reads through, and so keeps the total.

What the right-hand side reads, per element ``e`` of a periodic mesh:

* ``u[e, k, j, i]``, node ``(i, j, k)`` with ``i`` along ``r``, the DOFArray
  of ``examples/hexahedra.py``;
* grudge's geometry: the metric terms ``G[3 x + d] = J dr_d/dx_x`` and
  ``1 / J`` at every node, and the unit normal and the area element ``sJ``
  at every face node, the faces in modepy's order (``t``, ``r`` and ``s``,
  each at ``-1`` and ``+1``, `shapes.py#L522-L571
  <https://github.com/inducer/modepy/blob/01ac7955d30e248e460ccd751d40da4b0ff3e3e6/modepy/shapes.py#L522-L571>`__);
* the face maps: ``nbr[e, f]``, the element across face ``f``, and
  ``inside[f, a, b]`` and ``outside[f, a, b]``, the node ``(k, j, i)`` of
  face node ``(a, b)`` on this element and on the neighbor.

The velocity enters through ``V[d] = J dr_d/dx . v`` at a node and ``vn =
sJ v.n`` at a face node, which the program computes first. A time stepper
computes them once per mesh (``folded_rhs`` reads them); a kernel may also
compute them where it reads them, from the nine metric terms, as grudge's
kernel does.

Run this file three ways.

``python examples/advection.py``
    Computes the right-hand side natively and compares it with grudge's
    operators applied as dense matrices, before the rearrangement. Then fuses
    the kernels on the element, refuses one loop deeper with its pair, keeps
    the fluxes one element at a time, and compares the compiled run with the
    native one; and lists the facts of every split's launches on a device,
    and the race libParanumal's surface kernel avoids with its barriers.

``lanky check examples/advection.py``
    The theorems, every kernel's facts, the programs', the schedules'.

``loopty run examples/advection.py``
    Each kernel, the programs and the fused schedule, compiled and compared.

What is worth reading here
--------------------------

*The face gather.* ``u[nbr[e, f], outside[f, a, b, 0], ...]`` reads the
neighbor's node through the face maps, and every subscript is in bounds by
type: ``nbr`` holds points of ``Fin[ne]``, ``outside`` points of ``Fin[n]``.
Its transpose is the lift onto the face nodes, a theorem for every order
(``lift_is_the_gather_transposed``) that ``main`` checks on these maps.

*How deep the fusion goes.* The lift at node ``(k, j, i)`` reads the fluxes
of the six faces through it, which the flux kernel stores at other ``(a,
b)``, so the two fuse on the element and no deeper; the volume term and the
lift fuse on every loop.

*Splits on a device.* libParanumal runs a volume kernel and a surface kernel
(`advectionVolumeHex3D.okl
<https://github.com/paranumal/libparanumal/blob/5f96a9a5e7f316ea3c030a6c5c32f1c4aff6a7ab/solvers/advection/okl/advectionVolumeHex3D.okl>`__,
`advectionSurfaceHex3D.okl
<https://github.com/paranumal/libparanumal/blob/5f96a9a5e7f316ea3c030a6c5c32f1c4aff6a7ab/solvers/advection/okl/advectionSurfaceHex3D.okl#L27-L80>`__),
whose flux is the average and half the jump, a form with no branch;
``upwind_flux`` writes grudge's ``where``, a conditional expression, which
the trace records as one value chosen by ``vn > 0`` and C computes with
``?:``. grudge runs one kernel for the whole right-hand side. Each split below is
launches whose facts are decided before they run: libParanumal's, the
fluxes first, three launches, one, and one from the metric.
``benchmarks/advection.py`` times them. libParanumal's surface kernel lifts
onto the face nodes alone, with a barrier between pairs of faces; without
the barriers, ``lift_on_faces`` on a device is refused with the edge node
two faces write from two work items.
"""

from __future__ import annotations

from typing import Any

import numpy as np
from lanky import theorem
from lanky.prelude import Fn, Nat, Real

from loopty import Arr, Fin, IllegalCast, Schedule, kernel, program, reduce_sum

#: The order of the demo: four Gauss-Lobatto-Legendre nodes per axis.
ORDER = 3
N = ORDER + 1

#: Elements per axis of the periodic box; the timings take 48.
ELEMENTS = 2

#: The constant velocity.
VELOCITY = (0.7, -0.4, 0.3)


# {{{ the theorems: the inverse mass, pushed inward, at every order


@theorem
def inverse_mass_inward(
    n: Nat,
    Mi: Fn[Fin[n], Fin[n], Real],
    S: Fn[Fin[n], Fin[n], Real],
    M: Fn[Fin[n], Fin[n], Real],
    w: Fn[Fin[n], Fin[n], Fin[n], Real],
    k: Fin[n],
    j: Fin[n],
    i: Fin[n],
    inverse: all(sum(Mi[p, c] * M[c, p] for c in Fin[n]) == 1 for p in Fin[n])
    & all(
        sum(Mi[p, c] * M[c, q] for c in Fin[n]) == 0
        for p in Fin[n]
        for q in Fin[n]
        if p != q
    ),
) -> sum(
    Mi[k, c]
    * Mi[j, b]
    * Mi[i, a]
    * sum(
        M[c, t] * M[b, s] * S[a, r] * w[t, s, r]
        for t in Fin[n]
        for s in Fin[n]
        for r in Fin[n]
    )
    for c in Fin[n]
    for b in Fin[n]
    for a in Fin[n]
) == sum(
    sum(Mi[i, a] * S[a, r] for a in Fin[n]) * w[k, j, r] for r in Fin[n]
):
    """The inverse mass after the weak derivative along ``r``: one pass of ``M^-1 S^T``.

    The left side is grudge's: ``S^T`` on ``r`` and the mass on ``s`` and
    ``t`` (``M (x) M (x) S``), then the inverse mass on every axis.
    ``inverse`` says that ``Mi`` is the mass's inverse: ``M^-1 M`` is one on
    its diagonal and zero off it. There the mass cancels on ``s`` and
    ``t``, and what is left is ``volume``'s pass of ``D = M^-1 S^T`` along
    ``r``. The other two axes are the same with the axes renamed.
    """


@theorem
def lift_inward(
    n: Nat,
    Mi: Fn[Fin[n], Fin[n], Real],
    M: Fn[Fin[n], Fin[n], Real],
    end: Fn[Fin[n], Real],
    g: Fn[Fin[n], Fin[n], Real],
    k: Fin[n],
    j: Fin[n],
    i: Fin[n],
    inverse: all(sum(Mi[p, c] * M[c, p] for c in Fin[n]) == 1 for p in Fin[n])
    & all(
        sum(Mi[p, c] * M[c, q] for c in Fin[n]) == 0
        for p in Fin[n]
        for q in Fin[n]
        if p != q
    ),
) -> sum(
    Mi[k, c]
    * Mi[j, b]
    * Mi[i, a]
    * (end[a] * sum(M[c, t] * M[b, s] * g[t, s] for t in Fin[n] for s in Fin[n]))
    for c in Fin[n]
    for b in Fin[n]
    for a in Fin[n]
) == sum(Mi[i, a] * end[a] for a in Fin[n]) * g[k, j]:
    """The inverse mass after a face's mass is the face's column, times its flux.

    The left side is grudge's face mass of the face ``r = -1`` (``end`` is
    the basis at ``r = -1``, the face's own mass on ``s`` and ``t``), then
    the inverse mass. Where ``Mi`` is the mass's inverse (``inverse``) the
    mass cancels on the face's two axes, and what is left is ``lift``'s term
    for that face: the column ``L = M^-1 end`` along ``r``, times the face's
    flux. The other five faces are the same with the axes renamed.
    """


@theorem
def lift_is_the_gather_transposed(
    faces: Nat,
    n: Nat,
    nodes: Nat,
    node: Fn[Fin[faces], Fin[n], Fin[n], Fin[nodes]],
    g: Fn[Fin[faces], Fin[n], Fin[n], Real],
    u: Fn[Fin[nodes], Real],
) -> sum(
    sum(
        g[f, a, b]
        for f in Fin[faces]
        for a in Fin[n]
        for b in Fin[n]
        if node[f, a, b] == x
    )
    * u[x]
    for x in Fin[nodes]
) == sum(
    g[f, a, b] * u[node[f, a, b]] for f in Fin[faces] for a in Fin[n] for b in Fin[n]
):
    """Lifting the faces' values onto the nodes is the transpose of reading them there.

    ``node[f, a, b]`` is the element's node at face node ``(a, b)`` of face
    ``f`` (``inside``, its three indices one node number, and ``faces`` six
    on a hexahedron), ``g`` a value at
    every face node and ``u`` a field. The inner sum is what the face nodes
    at node ``x`` add into it, as ``lift_on_faces`` adds each face's flux;
    the right side reads ``u`` at the face nodes, as ``upwind`` reads the
    element's own values. Paired with any ``u``, the two agree, on every
    element and at every order: the lift is the gather's transpose. With
    ``u = 1`` it keeps the total, so what the faces lift onto an element is
    what they hold.
    """


# }}}


# {{{ the kernels: the fluxes through the face maps, the volume, the lift


@kernel
def reference_velocity(
    G: Arr[Fin[9], Fin[ne], Fin[n], Fin[n], Fin[n], Real],
    v: Arr[Fin[3], Real],
    V: Arr[Fin[3], Fin[ne], Fin[n], Fin[n], Fin[n], Real],
):
    """The velocity along each reference axis ``d``, times ``J``: ``J dr_d/dx . v``.

    ``G[3 x + d]`` is grudge's metric term ``J dr_d/dx_x``, as
    ``inverse_surface_metric_derivative_mat(times_area_element=True)`` gives it.
    """
    for d, e, k, j, i in V.dom:
        V[d, e, k, j, i] = (
            G[d, e, k, j, i] * v[0]
            + G[3 + d, e, k, j, i] * v[1]
            + G[6 + d, e, k, j, i] * v[2]
        )


@kernel
def normal_velocity(
    normal: Arr[Fin[3], Fin[ne], Fin[6], Fin[n], Fin[n], Real],
    sJ: Arr[Fin[ne], Fin[6], Fin[n], Fin[n], Real],
    v: Arr[Fin[3], Real],
    vn: Arr[Fin[ne], Fin[6], Fin[n], Fin[n], Real],
):
    """``sJ v.n`` at every face node: the face's area element times ``v.n``."""
    for e, f, a, b in vn.dom:
        vn[e, f, a, b] = sJ[e, f, a, b] * (
            normal[0, e, f, a, b] * v[0]
            + normal[1, e, f, a, b] * v[1]
            + normal[2, e, f, a, b] * v[2]
        )


def upwind_flux(vn, inside, outside):
    """The upwind flux, ``inside * vn`` where ``vn > 0`` and ``outside * vn`` elsewhere.

    grudge's ``where(v.n > 0, u_inside, u_outside) * v.n``, as Python writes
    it: one statement, whose value the condition on ``vn`` chooses.
    """
    return vn * (inside if vn > 0 else outside)


@kernel
def upwind(
    u: Arr[Fin[ne], Fin[n], Fin[n], Fin[n], Real],
    nbr: Arr[Fin[ne], Fin[6], Fin[ne]],
    inside: Arr[Fin[6], Fin[n], Fin[n], Fin[3], Fin[n]],
    outside: Arr[Fin[6], Fin[n], Fin[n], Fin[3], Fin[n]],
    vn: Arr[Fin[ne], Fin[6], Fin[n], Fin[n], Real],
    flux: Arr[Fin[ne], Fin[6], Fin[n], Fin[n], Real],
):
    """The upwind flux at every face node, from the element and its neighbor."""
    for e, f, a, b in flux.dom:
        flux[e, f, a, b] = upwind_flux(
            vn[e, f, a, b],
            u[
                e,
                inside[f, a, b, 0],
                inside[f, a, b, 1],
                inside[f, a, b, 2],
            ],
            u[
                nbr[e, f],
                outside[f, a, b, 0],
                outside[f, a, b, 1],
                outside[f, a, b, 2],
            ],
        )


@kernel
def volume(
    D: Arr[Fin[n], Fin[n], Real],
    V: Arr[Fin[3], Fin[ne], Fin[n], Fin[n], Fin[n], Real],
    invJ: Arr[Fin[ne], Fin[n], Fin[n], Fin[n], Real],
    u: Arr[Fin[ne], Fin[n], Fin[n], Fin[n], Real],
    rhs: Arr[Fin[ne], Fin[n], Fin[n], Fin[n], Real],
):
    """The weak divergence of ``v u`` over ``J``, ``D = M^-1 S^T`` along each axis."""
    for e, k, j, i in rhs.dom:
        rhs[e, k, j, i] = invJ[e, k, j, i] * (
            reduce_sum(
                D[i, a] * V[0, e, k, j, a] * u[e, k, j, a]
                for a in D.dom[i]
            )
            + reduce_sum(
                D[j, b] * V[1, e, k, b, i] * u[e, k, b, i]
                for b in D.dom[j]
            )
            + reduce_sum(
                D[k, c] * V[2, e, c, j, i] * u[e, c, j, i]
                for c in D.dom[k]
            )
        )


@kernel
def lift(
    L: Arr[Fin[2], Fin[n], Real],
    invJ: Arr[Fin[ne], Fin[n], Fin[n], Fin[n], Real],
    flux: Arr[Fin[ne], Fin[6], Fin[n], Fin[n], Real],
    rhs: Arr[Fin[ne], Fin[n], Fin[n], Fin[n], Real],
):
    """Each face's column times its flux, over ``J``, taken from the volume term."""
    for e, k, j, i in rhs.dom:
        rhs[e, k, j, i] = rhs[e, k, j, i] - invJ[e, k, j, i] * (
            # the faces t = -1 and +1, across k; r across i; s across j
            L[0, k] * flux[e, 0, j, i]
            + L[1, k] * flux[e, 1, j, i]
            + L[0, i] * flux[e, 2, k, j]
            + L[1, i] * flux[e, 3, k, j]
            + L[0, j] * flux[e, 4, k, i]
            + L[1, j] * flux[e, 5, k, i]
        )


@kernel
def lift_on_faces(
    ends: Arr[Fin[2], Real],
    invJ: Arr[Fin[ne], Fin[n], Fin[n], Fin[n], Real],
    flux: Arr[Fin[ne], Fin[6], Fin[n], Fin[n], Real],
    rhs: Arr[Fin[ne], Fin[n], Fin[n], Fin[n], Real],
):
    """libParanumal's lift with the GLL mass: each face's flux onto its own nodes.

    ``ends`` is ``1 / w`` at the two ends, ``lift``'s ``L`` with the GLL
    quadrature's tables, where it is zero but on the face's nodes.
    """
    for e in flux.dom:
        last = rhs.dom[e].size - 1
        for a, b in flux.dom[e, 0]:
            # the faces t = -1 and t = +1, at k = 0 and k = n - 1
            rhs[e, 0, a, b] -= invJ[e, 0, a, b] * ends[0] * flux[e, 0, a, b]
            rhs[e, last, a, b] -= invJ[e, last, a, b] * ends[1] * flux[e, 1, a, b]
            # r = -1 and r = +1, at i = 0 and i = n - 1
            rhs[e, a, b, 0] -= invJ[e, a, b, 0] * ends[0] * flux[e, 2, a, b]
            rhs[e, a, b, last] -= invJ[e, a, b, last] * ends[1] * flux[e, 3, a, b]
            # s = -1 and s = +1, at j = 0 and j = n - 1
            rhs[e, a, 0, b] -= invJ[e, a, 0, b] * ends[0] * flux[e, 4, a, b]
            rhs[e, a, last, b] -= invJ[e, a, last, b] * ends[1] * flux[e, 5, a, b]


@kernel
def weak_rhs(
    Mi: Arr[Fin[n], Fin[n], Real],
    M: Arr[Fin[n], Fin[n], Real],
    S: Arr[Fin[n], Fin[n], Real],
    ends: Arr[Fin[2], Fin[n], Real],
    V: Arr[Fin[3], Fin[ne], Fin[n], Fin[n], Fin[n], Real],
    invJ: Arr[Fin[ne], Fin[n], Fin[n], Fin[n], Real],
    u: Arr[Fin[ne], Fin[n], Fin[n], Fin[n], Real],
    flux: Arr[Fin[ne], Fin[6], Fin[n], Fin[n], Real],
    w: Arr[Fin[3], Fin[ne], Fin[n], Fin[n], Fin[n], Real],
    rhs: Arr[Fin[ne], Fin[n], Fin[n], Fin[n], Real],
):
    """grudge's ``WeakAdvectionOperator.operator``, as the math says it.

    ``inverse_mass(v . weak_local_grad(u) - face_mass(flux))``. ``w[d]`` is
    the field times the velocity along ``d``, ``V[d] u``, stored with the
    axis of ``d`` last. The weak divergence applies ``M (x) M (x) S^T`` to
    each, ``S^T`` along its last axis; the face mass puts each face's flux
    on the face's own two axes, ``flux[e, f]``'s, and the basis at its end,
    ``ends``, on the third; then the inverse mass ``Mi`` on every axis, the
    axes of each term in its own order, and ``1 / J``. ``volume`` and
    ``lift`` are derived from it (``derived``).
    """
    for e, k, j, i in u.dom:
        w[0, e, k, j, i] = V[0, e, k, j, i] * u[e, k, j, i]
        w[1, e, k, i, j] = V[1, e, k, j, i] * u[e, k, j, i]
        w[2, e, j, i, k] = V[2, e, k, j, i] * u[e, k, j, i]
    for e, k, j, i in rhs.dom:
        rhs[e, k, j, i] = invJ[e, k, j, i] * (
            # the weak derivative along r
            reduce_sum(
                Mi[k, c]
                * Mi[j, b]
                * Mi[i, a]
                * reduce_sum(
                    M[c, t] * M[b, s] * S[a, r] * w[0, e, t, s, r]
                    for t in M.dom[c]
                    for s in M.dom[b]
                    for r in S.dom[a]
                )
                for c in Mi.dom[k]
                for b in Mi.dom[j]
                for a in Mi.dom[i]
            )
            # the weak derivative along s, which w[1] holds last
            + reduce_sum(
                Mi[k, c]
                * Mi[i, b]
                * Mi[j, a]
                * reduce_sum(
                    M[c, t] * M[b, s] * S[a, r] * w[1, e, t, s, r]
                    for t in M.dom[c]
                    for s in M.dom[b]
                    for r in S.dom[a]
                )
                for c in Mi.dom[k]
                for b in Mi.dom[i]
                for a in Mi.dom[j]
            )
            # the weak derivative along t, which w[2] holds last
            + reduce_sum(
                Mi[j, c]
                * Mi[i, b]
                * Mi[k, a]
                * reduce_sum(
                    M[c, t] * M[b, s] * S[a, r] * w[2, e, t, s, r]
                    for t in M.dom[c]
                    for s in M.dom[b]
                    for r in S.dom[a]
                )
                for c in Mi.dom[j]
                for b in Mi.dom[i]
                for a in Mi.dom[k]
            )
            - (
                # the face t = -1
                reduce_sum(
                    Mi[j, c]
                    * Mi[i, b]
                    * Mi[k, a]
                    * (
                        ends[0, a]
                        * reduce_sum(
                            M[c, t] * M[b, s] * flux[e, 0, t, s]
                            for t in M.dom[c]
                            for s in M.dom[b]
                        )
                    )
                    for c in Mi.dom[j]
                    for b in Mi.dom[i]
                    for a in Mi.dom[k]
                )
                # the face t = +1
                + reduce_sum(
                    Mi[j, c]
                    * Mi[i, b]
                    * Mi[k, a]
                    * (
                        ends[1, a]
                        * reduce_sum(
                            M[c, t] * M[b, s] * flux[e, 1, t, s]
                            for t in M.dom[c]
                            for s in M.dom[b]
                        )
                    )
                    for c in Mi.dom[j]
                    for b in Mi.dom[i]
                    for a in Mi.dom[k]
                )
                # the face r = -1
                + reduce_sum(
                    Mi[k, c]
                    * Mi[j, b]
                    * Mi[i, a]
                    * (
                        ends[0, a]
                        * reduce_sum(
                            M[c, t] * M[b, s] * flux[e, 2, t, s]
                            for t in M.dom[c]
                            for s in M.dom[b]
                        )
                    )
                    for c in Mi.dom[k]
                    for b in Mi.dom[j]
                    for a in Mi.dom[i]
                )
                # the face r = +1
                + reduce_sum(
                    Mi[k, c]
                    * Mi[j, b]
                    * Mi[i, a]
                    * (
                        ends[1, a]
                        * reduce_sum(
                            M[c, t] * M[b, s] * flux[e, 3, t, s]
                            for t in M.dom[c]
                            for s in M.dom[b]
                        )
                    )
                    for c in Mi.dom[k]
                    for b in Mi.dom[j]
                    for a in Mi.dom[i]
                )
                # the face s = -1
                + reduce_sum(
                    Mi[k, c]
                    * Mi[i, b]
                    * Mi[j, a]
                    * (
                        ends[0, a]
                        * reduce_sum(
                            M[c, t] * M[b, s] * flux[e, 4, t, s]
                            for t in M.dom[c]
                            for s in M.dom[b]
                        )
                    )
                    for c in Mi.dom[k]
                    for b in Mi.dom[i]
                    for a in Mi.dom[j]
                )
                # the face s = +1
                + reduce_sum(
                    Mi[k, c]
                    * Mi[i, b]
                    * Mi[j, a]
                    * (
                        ends[1, a]
                        * reduce_sum(
                            M[c, t] * M[b, s] * flux[e, 5, t, s]
                            for t in M.dom[c]
                            for s in M.dom[b]
                        )
                    )
                    for c in Mi.dom[k]
                    for b in Mi.dom[i]
                    for a in Mi.dom[j]
                )
            )
        )


@program
def advection_rhs(u, G, v, normal, sJ, invJ, nbr, inside, outside, D, L, rhs):
    """grudge's ``WeakAdvectionOperator.operator``, from grudge's own geometry."""
    nodes = D.dom.size
    V = Arr.zeros((3, u.dom.size, nodes, nodes, nodes))
    vn = Arr.zeros_like(sJ)
    flux = Arr.zeros_like(sJ)
    reference_velocity(G, v, V)
    normal_velocity(normal, sJ, v, vn)
    upwind(u, nbr, inside, outside, vn, flux)
    volume(D, V, invJ, u, rhs)
    lift(L, invJ, flux, rhs)


@program
def folded_rhs(u, V, invJ, nbr, inside, outside, vn, D, L, rhs):
    """The same, the velocity folded into ``V`` and ``vn`` once per mesh."""
    flux = Arr.zeros_like(vn)
    upwind(u, nbr, inside, outside, vn, flux)
    volume(D, V, invJ, u, rhs)
    lift(L, invJ, flux, rhs)


# }}}


# {{{ the schedules

#: The velocity computed where it is read; the volume term and the lift at
#: each node, on every loop; the fluxes with them on the element.
fused = (
    Schedule(advection_rhs)
    .substitute(advection_rhs.V)
    .substitute(advection_rhs.vn)
    .fuse(advection_rhs.volume.i, advection_rhs.lift.i)
    .fuse(advection_rhs.upwind.e, advection_rhs.volume.e)
)

#: The same, with the fluxes kept one element at a time.
contracted = fused.contract(advection_rhs.flux)

#: grudge PR #362's rearrangement, derived from grudge's operator: the inverse
#: mass pushed inward, which ``volume`` and ``lift`` compute, with ``D = M^-1
#: S^T`` and the lift's columns ``L = M^-1 e`` each computed once.
derived = Schedule(weak_rhs).rewrite(
    weak_rhs.rhs, by=(inverse_mass_inward, lift_inward), keep=("D", "L")
)


@program
def surface(u, nbr, inside, outside, vn, L, invJ, rhs):
    """libParanumal's second launch: the fluxes, and the lift into the volume term."""
    flux = Arr.zeros_like(vn)
    upwind(u, nbr, inside, outside, vn, flux)
    lift(L, invJ, flux, rhs)


@program
def volume_and_lift(D, V, invJ, u, L, flux, rhs):
    """The other split's second launch: the volume term and the lift."""
    volume(D, V, invJ, u, rhs)
    lift(L, invJ, flux, rhs)


def on_nodes(schedule: Schedule, loops: Any) -> Schedule:
    """``schedule`` with the element a work group, its nodes the work items.

    ``loops`` is the kernel, or the call of it, whose loops ``e, k, j, i``
    go on ``g.0, l.2, l.1, l.0``: ``on_nodes(whole, folded_rhs.volume)``
    puts the volume term's loops there, whatever the program calls them.
    """
    return schedule.tag(
        {loops.e: "g.0", loops.k: "l.2", loops.j: "l.1", loops.i: "l.0"}
    )


def on_face_nodes(schedule: Schedule, loops: Any) -> Schedule:
    """The fluxes' launch: the element a work group, a face's nodes the work items.

    ``loops`` is the kernel, or the call, whose loops ``e, f, a, b`` are
    the element, the face and a face's nodes; the six faces run in sequence
    on each work item.
    """
    return schedule.tag({loops.e: "g.0", loops.a: "l.1", loops.b: "l.0"})


def launches(split: str = "one", target: str = "c") -> list[Schedule]:
    """The right-hand side as launches on a device, each a checked schedule.

    The first four read the velocity folded into the geometry once per mesh,
    ``V`` and ``vn`` (``folded_rhs``). ``"volume, surface"`` is libParanumal's
    split: the volume term, then the fluxes computed where the lift reads
    them. ``"fluxes, volume"`` stores the fluxes first, then computes the
    volume term and the lift. ``"three"`` stores the fluxes and runs the
    volume term and the lift apart. ``"one"`` is one launch, as grudge runs
    it: a face's flux computed once per node of the line through it.
    ``"one, from the metric"`` is one launch from grudge's own geometry, the
    velocity computed where it is read, from the nine metric terms of a node
    and the normal of a face node.
    """
    if split == "three":
        return [
            on_face_nodes(Schedule(upwind, target=target), upwind),
            on_nodes(Schedule(volume, target=target), volume),
            on_nodes(Schedule(lift, target=target), lift),
        ]
    if split == "volume, surface":
        second = Schedule(surface, target=target).substitute(surface.flux)
        return [
            on_nodes(Schedule(volume, target=target), volume),
            on_nodes(second, surface.lift),
        ]
    if split == "fluxes, volume":
        both = Schedule(volume_and_lift, target=target).fuse(
            volume_and_lift.volume.i, volume_and_lift.lift.i
        )
        return [
            on_face_nodes(Schedule(upwind, target=target), upwind),
            on_nodes(both, volume_and_lift.volume),
        ]
    if split == "one":
        whole = Schedule(folded_rhs, target=target).substitute(folded_rhs.flux)
        fused_nodes = whole.fuse(folded_rhs.volume.i, folded_rhs.lift.i)
        return [on_nodes(fused_nodes, folded_rhs.volume)]
    if split == "one, from the metric":
        whole = (
            Schedule(advection_rhs, target=target)
            .substitute(advection_rhs.V)
            .substitute(advection_rhs.vn)
            .substitute(advection_rhs.flux)
        )
        fused_nodes = whole.fuse(advection_rhs.volume.i, advection_rhs.lift.i)
        return [on_nodes(fused_nodes, advection_rhs.volume)]
    raise ValueError(f"no split {split!r}")


#: Every split :func:`launches` knows.
SPLITS = ("volume, surface", "fluxes, volume", "three", "one", "one, from the metric")


# }}}


# {{{ the reference element's tables, the mesh, and grudge's dense operators


def gll(order: int) -> tuple[np.ndarray, np.ndarray]:
    """The Gauss-Lobatto-Legendre nodes on ``[-1, 1]`` and their weights."""
    n = order + 1
    legendre = np.polynomial.legendre.Legendre
    interior = legendre.basis(n - 1).deriv().roots().real if n > 2 else []
    nodes = np.concatenate([[-1.0], np.sort(interior), [1.0]])
    weights = 2.0 / (n * (n - 1) * legendre.basis(n - 1)(nodes) ** 2)
    return nodes, weights


def differentiation(order: int = ORDER) -> np.ndarray:
    """``Dr[m, i]``: the ``i``-th Lagrange polynomial's derivative at node ``m``."""
    n = order + 1
    legendre = np.polynomial.legendre.Legendre
    nodes, _ = gll(order)
    V = np.stack([legendre.basis(m)(nodes) for m in range(n)], axis=1)
    Vr = np.stack([legendre.basis(m).deriv()(nodes) for m in range(n)], axis=1)
    return Vr @ np.linalg.inv(V)


def factors(order: int = ORDER) -> tuple[np.ndarray, np.ndarray]:
    """The weak stiffness ``S^T`` and the exact mass ``M`` on the GLL nodes.

    As ``make_stiffness_t_operator`` and ``make_mass_operator`` build them
    with modepy: ``M[i, j]`` the integral of the Lagrange polynomials of
    nodes ``i`` and ``j``, ``S^T[i, j]`` that of the derivative of the
    ``i``-th times the ``j``-th.
    """
    n = order + 1
    legendre = np.polynomial.legendre.Legendre
    nodes, _ = gll(order)
    scale = np.sqrt((2 * np.arange(n) + 1) / 2)
    inverse = np.linalg.inv(
        np.stack([legendre.basis(m)(nodes) for m in range(n)], axis=1) * scale
    )
    mass = inverse.T @ inverse
    return differentiation(order).T @ mass, mass


def tables(order: int = ORDER, mass: str = "exact") -> tuple[np.ndarray, np.ndarray]:
    """``D = M^-1 S^T`` and the lift's two columns ``L[s] = M^-1 e_s``.

    ``mass`` is ``"exact"``, grudge #418's, or ``"gll"``, the GLL
    quadrature's, ``W = diag(w)``: then ``S^T = Dr^T W``, ``D = W^-1 Dr^T W``,
    and ``L[s]`` is the end node's weight's inverse there and zero elsewhere.
    """
    if mass == "gll":
        _, weights = gll(order)
        ends = np.eye(order + 1)[[0, -1]] / weights[[0, -1], None]
        operator = differentiation(order).T * weights / weights[:, None]
        return np.ascontiguousarray(operator), ends
    stiffness_t, exact = factors(order)
    inverse = np.linalg.inv(exact)
    return inverse @ stiffness_t, np.stack([inverse[:, 0], inverse[:, -1]])


#: Face ``f`` in modepy's order: its normal axis (0 for ``r``, 1 for ``s``,
#: 2 for ``t``) and its side, ``-1`` or ``+1``.
FACES = ((2, -1), (2, 1), (0, -1), (0, 1), (1, -1), (1, 1))


def face_maps(order: int = ORDER) -> tuple[np.ndarray, np.ndarray]:
    """``inside[f, a, b]`` and ``outside[f, a, b]``, a face node's node ``(k, j, i)``.

    Node ``(a, b)`` of a face is its two other axes, slowest first: ``(j,
    i)`` across ``t``, ``(k, j)`` across ``r``, ``(k, i)`` across ``s``.
    ``outside`` is the same node on the neighbor, at the other end of the
    normal, as on a mesh whose elements all have one orientation.
    """
    n = order + 1
    inside = np.zeros((6, n, n, 3), dtype=np.int32)
    outside = np.zeros((6, n, n, 3), dtype=np.int32)
    for f, (axis, side) in enumerate(FACES):
        here, there = (0, n - 1) if side < 0 else (n - 1, 0)
        position = 2 - axis  # (k, j, i) holds t, s, r
        for a in range(n):
            for b in range(n):
                others = [a, b]
                node = [others.pop(0) if p != position else here for p in range(3)]
                inside[f, a, b] = node
                node[position] = there
                outside[f, a, b] = node
    return inside, outside


def box(elements: int = ELEMENTS) -> np.ndarray:
    """``nbr[e, f]`` on a periodic box of ``elements**3`` hexahedra.

    Element ``(x, y, z)`` is ``e = (z * E + y) * E + x``, its axis ``r``
    along ``x``.
    """
    E = elements
    z, y, x = np.meshgrid(np.arange(E), np.arange(E), np.arange(E), indexing="ij")
    out = np.zeros((E**3, 6), dtype=np.int32)
    position = [x.ravel(), y.ravel(), z.ravel()]
    for f, (axis, side) in enumerate(FACES):
        moved = [p.copy() for p in position]
        moved[axis] = (moved[axis] + side) % E
        out[:, f] = (moved[2] * E + moved[1]) * E + moved[0]
    return out


def inputs(elements: int = ELEMENTS, order: int = ORDER, mass: str = "exact") -> dict:
    """A field on the periodic box ``[-1, 1]^3``, its geometry and the tables.

    The geometry is grudge's: the metric terms ``J dr_d/dx_x`` at every node,
    and the unit normal and the area element ``sJ`` at every face node.
    """
    n = order + 1
    E = elements
    h = 2.0 / E
    nodes, _ = gll(order)
    ne = E**3
    # node (i, j, k) of element (x, y, z), u[e, k, j, i]
    e = np.arange(ne)
    corner = np.stack([e % E, (e // E) % E, e // E**2], axis=1) * h - 1.0
    ref = (nodes + 1) * h / 2
    X = corner[:, 0, None, None, None] + ref[None, None, None, :]
    Y = corner[:, 1, None, None, None] + ref[None, None, :, None]
    Z = corner[:, 2, None, None, None] + ref[None, :, None, None]
    # a smooth field, shifted up and down on alternate elements: a DG field
    # jumps across faces, and the flux has to take the upwind side
    alternate = (-1.0) ** (e % E + (e // E) % E + e // E**2)
    u = (
        np.sin(np.pi * X) * np.cos(np.pi * Y)
        + 0.3 * np.sin(np.pi * Z)
        + 0.25 * alternate[:, None, None, None]
    )
    shape = (ne, n, n, n)
    # dr_d/dx_x is 2/h where d is x, and J is (h/2)**3
    G = np.zeros((9, *shape))
    for d in range(3):
        G[3 * d + d] = (h / 2) ** 2
    normal = np.zeros((3, ne, 6, n, n))
    for f, (axis, side) in enumerate(FACES):
        normal[axis, :, f] = side
    inside, outside = face_maps(order)
    D, L = tables(order, mass)
    return {
        "u": Arr.from_numpy(u),
        "G": Arr.from_numpy(G),
        "v": Arr.from_numpy(np.array(VELOCITY)),
        "normal": Arr.from_numpy(normal),
        "sJ": Arr.from_numpy(np.full((ne, 6, n, n), (h / 2) ** 2)),
        "invJ": Arr.from_numpy(np.full(shape, (2 / h) ** 3)),
        "nbr": Arr.from_numpy(box(E)),
        "inside": Arr.from_numpy(inside),
        "outside": Arr.from_numpy(outside),
        "D": Arr.from_numpy(D),
        "L": Arr.from_numpy(L),
        "rhs": Arr.zeros(shape),
    }


def folded(data: dict) -> dict:
    """``folded_rhs``'s inputs: the velocity folded into the geometry by its kernels."""
    G, v, normal, sJ = (data[name] for name in ("G", "v", "normal", "sJ"))
    V = Arr.zeros((3, *data["u"].numpy().shape))
    vn = Arr.zeros(sJ.numpy().shape)
    reference_velocity(G, v, V)
    normal_velocity(normal, sJ, v, vn)
    kept = ("u", "invJ", "nbr", "inside", "outside", "D", "L")
    return {
        **{name: data[name] for name in kept},
        "V": V,
        "vn": vn,
        "rhs": Arr.zeros(data["rhs"].numpy().shape),
    }


def operator_inputs(
    elements: int = ELEMENTS, order: int = ORDER, seed: int | None = None
) -> dict:
    """``weak_rhs``'s arguments: grudge's tables, the folded geometry and the fluxes.

    ``Mi`` is the inverse of the exact mass, as numpy computes it, so ``M^-1
    M`` is the identity to the tables' digits and not exactly; ``ends`` is
    the basis at ``-1`` and at ``+1``, which is one at the first and the last
    node. The field is :func:`inputs`' and the fluxes are ``upwind``'s; with
    a ``seed``, both are drawn at random instead, which on one periodic
    element leaves a right-hand side that is not zero.
    """
    stiffness_t, mass = factors(order)
    once = folded(inputs(elements, order))
    flux = Arr.zeros(once["vn"].numpy().shape)
    upwind(once["u"], once["nbr"], once["inside"], once["outside"], once["vn"], flux)
    u = once["u"]
    shape = u.numpy().shape
    if seed is not None:
        rng = np.random.default_rng(seed)
        u = Arr.from_numpy(rng.normal(size=shape))
        flux = Arr.from_numpy(rng.normal(size=flux.numpy().shape))
    return {
        "Mi": Arr.from_numpy(np.linalg.inv(mass)),
        "M": Arr.from_numpy(mass),
        "S": Arr.from_numpy(stiffness_t),
        "ends": Arr.from_numpy(np.eye(order + 1)[[0, -1]]),
        "V": once["V"],
        "invJ": once["invJ"],
        "u": u,
        "flux": flux,
        "w": Arr.zeros((3, *shape)),
        "rhs": Arr.zeros(shape),
    }


def dense(data: dict, order: int = ORDER) -> np.ndarray:
    """grudge's right-hand side with its dense operators, before the rearrangement.

    The weak divergence ``(S^T or M on each axis)`` of ``V_d u``, less each
    face's mass of ``vn * u_upwind``, then the inverse mass on every axis
    and ``1 / J``; ``kron`` with the fastest axis last, as modepy flattens.
    """
    stiffness_t, mass = factors(order)
    n = order + 1
    inverse = np.linalg.inv(mass)
    u = data["u"].numpy()
    v = data["v"].numpy()
    V = np.einsum("x,xdekji->dekji", v, data["G"].numpy().reshape(3, 3, *u.shape))
    vn = data["sJ"].numpy() * np.einsum("x,xefab->efab", v, data["normal"].numpy())
    nbr = data["nbr"].numpy()
    inside, outside = (data[name].numpy() for name in ("inside", "outside"))
    ne = u.shape[0]
    flat = u.reshape(ne, -1)

    def along(axis: int, d: int) -> np.ndarray:
        return stiffness_t if axis == d else mass

    weak = sum(
        (V[d].reshape(ne, -1) * flat)
        @ np.kron(along(2, d), np.kron(along(1, d), along(0, d))).T
        for d in range(3)
    )
    faces = np.zeros((ne, n, n, n))
    for f, (axis, side) in enumerate(FACES):
        k, j, i = (inside[f, ..., p] for p in range(3))
        ko, jo, io = (outside[f, ..., p] for p in range(3))
        here = u[:, k, j, i]
        there = u[nbr[:, f][:, None, None], ko, jo, io]
        upwind = np.where(vn[:, f] > 0, here, there) * vn[:, f]
        end = np.eye(n)[0 if side < 0 else -1]
        # the face's mass on its two axes, the end of the element on the third
        slab = np.einsum("ta,sb,eab->ets", mass, mass, upwind)
        if axis == 0:
            faces += np.einsum("ets,i->etsi", slab, end)
        elif axis == 1:
            faces += np.einsum("eti,j->etji", slab, end)
        else:
            faces += np.einsum("esi,k->eksi", slab, end)
    total = (weak - faces.reshape(ne, -1)) @ np.kron(
        inverse, np.kron(inverse, inverse)
    ).T
    return (total * data["invJ"].numpy().reshape(ne, -1)).reshape(u.shape)


def example_inputs() -> dict:
    """The inputs ``loopty run`` gives each kernel, and the programs."""
    data = inputs()
    once = folded(inputs())
    u, V, invJ, vn, D, L = (once[name] for name in ("u", "V", "invJ", "vn", "D", "L"))
    shape = u.numpy().shape
    rng = np.random.default_rng(1)
    volume_term = rng.normal(size=shape)
    flux = Arr.from_numpy(rng.normal(size=vn.numpy().shape))
    faces = {name: once[name] for name in ("u", "nbr", "inside", "outside", "vn")}
    return {
        "reference_velocity": {
            "G": data["G"],
            "v": data["v"],
            "V": Arr.zeros(V.numpy().shape),
        },
        "normal_velocity": {
            "normal": data["normal"],
            "sJ": data["sJ"],
            "v": data["v"],
            "vn": Arr.zeros(vn.numpy().shape),
        },
        "upwind": {**faces, "flux": Arr.zeros(vn.numpy().shape)},
        "volume": {"D": D, "V": V, "invJ": invJ, "u": u, "rhs": Arr.zeros(shape)},
        "lift": {
            "L": L,
            "invJ": invJ,
            "flux": flux,
            "rhs": Arr.from_numpy(volume_term),
        },
        "lift_on_faces": {
            "ends": Arr.from_numpy(1 / gll(ORDER)[1][[0, -1]]),
            "invJ": invJ,
            "flux": flux,
            "rhs": Arr.from_numpy(volume_term),
        },
        # one element of order 2: the native weak_rhs adds every term of the
        # dense operators
        "weak_rhs": operator_inputs(elements=1, order=2, seed=2),
        "advection_rhs": data,
        "folded_rhs": once,
        "surface": {**faces, "L": L, "invJ": invJ, "rhs": Arr.from_numpy(volume_term)},
        "volume_and_lift": {
            "D": D,
            "V": V,
            "invJ": invJ,
            "u": u,
            "L": L,
            "flux": flux,
            "rhs": Arr.zeros(shape),
        },
    }


# }}}


def _settled(facts: list) -> str:
    """``all decided``, or ``4 proved, 1 decided``: the statuses of a launch's casts."""
    counts: dict[str, int] = {}
    for fact in facts:
        counts[fact.status.value] = counts.get(fact.status.value, 0) + 1
    if len(counts) == 1:
        return f"all {next(iter(counts))}"
    return ", ".join(f"{count} {status}" for status, count in counts.items())


def main() -> int:
    """Natively against grudge's dense operators, the fusion and window, the device."""
    from loopty.executor import LoopyExecutor
    from loopty.obligations import prove

    _, mass = factors()
    inverse = bool(np.allclose(np.linalg.inv(mass) @ mass, np.eye(N), atol=1e-13))
    data = inputs()
    advection_rhs(**data)
    expected = dense(data)
    once = folded(inputs())
    folded_rhs(**once)
    scale = np.abs(expected).max()
    gap = max(
        np.abs(data["rhs"].numpy() - expected).max(),
        np.abs(once["rhs"].numpy() - expected).max(),
    )
    native = bool(gap <= 1e-13 * scale)
    # the bound, not the difference, which rounds differently on another BLAS
    print(
        f"order {ORDER}, {ELEMENTS}**3 periodic hexahedra: the right-hand side is "
        f"grudge's, inverse mass and all, to 1e-13 of the largest value: {native} "
        f"(M^-1 M is the identity: {inverse})"
    )
    # the third theorem on these face maps: with unit ends and no 1/J,
    # lift_on_faces takes each face node's value from the node it sits on
    rng = np.random.default_rng(0)
    shape = data["u"].numpy().shape
    values = rng.normal(size=data["sJ"].numpy().shape)
    lifted = Arr.zeros(shape)
    lift_on_faces(
        Arr.from_numpy(np.ones(2)),
        Arr.from_numpy(np.ones(shape)),
        Arr.from_numpy(values),
        lifted,
    )
    field = rng.normal(size=shape)
    inside, _ = face_maps()
    at_faces = field[:, inside[..., 0], inside[..., 1], inside[..., 2]]
    paired = -(lifted.numpy() * field).sum()
    transposed = bool(np.isclose(paired, (values * at_faces).sum(), rtol=1e-12))
    print(
        "the lift onto the face nodes is the transpose of the face gather on these "
        f"maps: {transposed}"
    )
    print()

    # volume and lift derived from grudge's operator by the two theorems, the
    # tables M^-1 S^T and M^-1 e computed once: compiled, against the kernels
    print(f"derived: {derived!r}")
    for name in ("D", "L"):
        print(f"  {name} computed once, before rhs")
    operator = operator_inputs()
    D, L = tables()
    by_kernels = Arr.zeros(operator["rhs"].numpy().shape)
    V, invJ, u = (operator[name] for name in ("V", "invJ", "u"))
    volume(Arr.from_numpy(D), V, invJ, u, by_kernels)
    lift(Arr.from_numpy(L), operator["invJ"], operator["flux"], by_kernels)
    want = by_kernels.numpy()
    got = LoopyExecutor().run(derived, **operator)["rhs"]
    agrees = bool(np.abs(got - want).max() <= 1e-13 * np.abs(want).max())
    print(
        "  the compiled derived rhs is volume and lift's, from the same fluxes, to "
        f"1e-13 of the largest value: {agrees}"
    )
    print()

    # one loop deeper than fused: the fluxes on the loops of the nodes
    sizes = {"n": N, "ne": ELEMENTS**3}
    nodes_fused = (
        Schedule(advection_rhs, sizes=sizes)
        .substitute(advection_rhs.V)
        .substitute(advection_rhs.vn)
        .fuse(advection_rhs.volume.i, advection_rhs.lift.i)
    )
    try:
        nodes_fused.fuse(advection_rhs.upwind.b, advection_rhs.volume.i)
    except IllegalCast as exc:
        print(f"refused: {exc}")
    print()

    print(f"accepted: {contracted!r}")
    for name, window in contracted.windows.items():
        print(f"  {name} kept in {window.placement}")
    kept = LoopyExecutor().differential(advection_rhs, contracted, inputs())
    for name, detail in kept.provenance["outputs"].items():
        print(
            f"  {name}: difference {detail['difference']:.3g} within "
            f"{detail['tolerance']:.3g} ({detail['exactness']}) -> {kept.status.value}"
        )
    print()

    print("libParanumal's lift onto the face nodes, a face's nodes on the work items:")
    try:
        on_face_nodes(Schedule(lift_on_faces, sizes=sizes), lift_on_faces)
    except IllegalCast as exc:
        print(f"  refused: {exc}")
    atomic = on_face_nodes(
        Schedule(lift_on_faces).atomic(lift_on_faces.rhs), lift_on_faces
    )
    # where Lean is installed it proves each cast, that no dependence joins
    # two work items included
    settled = ("decided", "proved")
    cast = [prove(fact) for fact in atomic.facts() if fact.kind != "buildable"]
    print(
        f"  accepted: {'.'.join(atomic.history)}: {len(cast)} facts, "
        f"{_settled(cast)}"
    )
    legal = all(fact.status.value in settled for fact in cast)
    print()

    print("on a device, each split's launches, every node on a work item:")
    for split in SPLITS:
        for schedule in launches(split):
            cast = [
                prove(fact) for fact in schedule.facts() if fact.kind != "buildable"
            ]
            legal = legal and all(fact.status.value in settled for fact in cast)
            steps = len(schedule.history)
            print(
                f"  {split}: {schedule.term.name}, "
                f"{steps} step{'s' if steps != 1 else ''}: "
                f"{len(cast)} facts, {_settled(cast)}"
            )
    good = native and inverse and transposed and agrees and legal
    return 0 if good and kept.status.value == "tested" else 1


if __name__ == "__main__":
    raise SystemExit(main())
