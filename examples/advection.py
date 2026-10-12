"""grudge's advection right-hand side on hexahedra: volume, faces and lift, checked.

The right-hand side of ``du/dt + v . grad u = 0`` on a hexahedral DG mesh is
what grudge's ``WeakAdvectionOperator`` computes (as of a-alveyblanc/grudge
4d103bb, grudge PR #418), and what this file states, kernel by kernel:

* ``WeakAdvectionOperator.operator``
  (`models/advection.py#L143-L183
  <https://github.com/a-alveyblanc/grudge/blob/4d103bbbb31bfdb175f8997ae53a76a6cb8c14dc/grudge/models/advection.py#L143-L183>`__)
  is ``inverse_mass(v . weak_local_grad(u) - face_mass(flux))``;
* the flux is the upwind flux, ``where(v.n > 0, u_inside, u_outside) * v.n``
  at every face node (``advection_weak_flux``,
  `models/advection.py#L42-L60
  <https://github.com/a-alveyblanc/grudge/blob/4d103bbbb31bfdb175f8997ae53a76a6cb8c14dc/grudge/models/advection.py#L42-L60>`__),
  the outside value read from the neighbor across the face;
* ``face_mass`` puts each face's ``sJ * flux`` back on the element's nodes,
  the face's own mass on its two axes and the face's end of the element on
  the third (`op.py#L1368-L1546
  <https://github.com/a-alveyblanc/grudge/blob/4d103bbbb31bfdb175f8997ae53a76a6cb8c14dc/grudge/op.py#L1368-L1546>`__),
  and ``inverse_mass`` applies the inverse of the mass on every axis and
  divides by the Jacobian ``J`` (`op.py#L1118-L1182
  <https://github.com/a-alveyblanc/grudge/blob/4d103bbbb31bfdb175f8997ae53a76a6cb8c14dc/grudge/op.py#L1118-L1182>`__).

The inverse mass is pushed inward, as grudge PR #362 rewrites it: on every
axis but one it meets a mass and cancels, so the volume term is one pass of
``D = M^-1 S^T`` along each axis, and a face's term is the column ``L =
M^-1 e`` of the face's end along its normal, times the face's flux. The two
theorems below are that rearrangement, proved for every order; that ``M^-1``
is the mass's inverse is a property of the tables, which ``main`` checks.
The same kernels take the tables of a GLL quadrature mass, the one
Alvey-Blanco's matched right-hand side and libParanumal use, where ``L`` is
a face node's own weight and nothing else.

What the kernels read, per element ``e`` of a periodic mesh:

* ``u[e, k, j, i]``, node ``(i, j, k)`` with ``i`` along ``r``, the DOFArray
  of ``examples/hexahedra.py``;
* ``V[d, e, k, j, i]``, the velocity in reference direction ``d`` times
  ``J``: grudge's metric terms ``J dr_d/dx_x`` summed against ``v``;
* ``vn[e, f, a, b]``, ``sJ v.n`` at node ``(a, b)`` of face ``f``, the faces
  in modepy's order (``t``, ``r`` and ``s`` at ``-1`` and ``+1``,
  `shapes.py#L522-L571
  <https://github.com/inducer/modepy/blob/01ac7955d30e248e460ccd751d40da4b0ff3e3e6/modepy/shapes.py#L522-L571>`__);
* the face maps: ``nbr[e, f]``, the element across face ``f``, and
  ``inside[f, a, b]`` and ``outside[f, a, b]``, the node ``(k, j, i)`` of the
  face node on this element and on the neighbor.

Run this file three ways.

``python examples/advection.py``
    Computes the right-hand side natively and compares it with grudge's
    operators applied as dense matrices, before the rearrangement. Then fuses
    the three kernels on the element, refuses one loop deeper with its pair,
    keeps the fluxes one element at a time, and compares the compiled run
    with the native one; and lists the facts of the launches a device runs.

``lanky check examples/advection.py``
    The theorems, every kernel's facts, the program's, the schedules'.

``loopty run examples/advection.py``
    Each kernel, the program and the fused schedule, compiled and compared.

What is worth reading here
--------------------------

*The face gather.* ``u[nbr[e, f], outside[f, a, b, 0], ...]`` reads the
neighbor's node through three tables. Every subscript is in bounds by type:
``nbr`` holds points of ``Fin[ne]``, ``outside`` points of ``Fin[n]``.

*How deep the fusion goes.* The lift at node ``(k, j, i)`` reads the fluxes
of the faces through it, which the flux kernel computes at other ``(a, b)``,
so the two fuse on the element and no deeper; the volume term and the lift
fuse on every loop.

*Splits on a device.* libParanumal runs a volume kernel and a surface
kernel; grudge one kernel for the whole right-hand side. Every launch below
is a schedule whose facts are decided before it runs: the split, the other
split, three launches, and one, the fluxes computed where the lift reads
them. ``benchmarks/advection.py`` times them.
"""

from __future__ import annotations

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
    sum(Mi[k, c] * M[c, t] for c in Fin[n])
    * sum(Mi[j, b] * M[b, s] for b in Fin[n])
    * sum(Mi[i, a] * S[a, r] for a in Fin[n])
    * w[t, s, r]
    for t in Fin[n]
    for s in Fin[n]
    for r in Fin[n]
):
    """The inverse mass after the weak derivative along ``r`` is one pass per axis.

    The left side is grudge's: ``S^T`` on ``r`` and the mass on ``s`` and
    ``t`` (``M (x) M (x) S``), then the inverse mass on every axis. The
    right side applies ``M^-1 M`` along ``t`` and ``s`` and ``M^-1 S`` along
    ``r``; where ``M^-1`` is the mass's inverse the first two are the
    identity, and what is left is ``volume``'s pass along ``r``.
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
) -> sum(
    Mi[k, c]
    * Mi[j, b]
    * Mi[i, a]
    * (end[a] * sum(M[c, t] * M[b, s] * g[t, s] for t in Fin[n] for s in Fin[n]))
    for c in Fin[n]
    for b in Fin[n]
    for a in Fin[n]
) == sum(Mi[i, a] * end[a] for a in Fin[n]) * sum(
    sum(Mi[k, c] * M[c, t] for c in Fin[n])
    * sum(Mi[j, b] * M[b, s] for b in Fin[n])
    * g[t, s]
    for t in Fin[n]
    for s in Fin[n]
):
    """The inverse mass after a face's mass is the face's column, times its flux.

    The left side is grudge's face mass of the face ``r = -1`` (``end`` is
    the basis at ``r = -1``, the face's own mass on ``s`` and ``t``), then
    the inverse mass; the right side is the column ``M^-1 end`` along ``r``
    times ``M^-1 M`` on the face's two axes, which is ``lift``'s term for
    that face where ``M^-1`` is the mass's inverse.
    """


# }}}


# {{{ the kernels: the fluxes through the face maps, the volume, the lift


def upwind_flux(vn, inside, outside):
    """The upwind flux, ``inside * vn`` where ``vn > 0`` and ``outside * vn`` elsewhere.

    Written as libParanumal's surface kernel writes it, the average and half
    the jump, with no branch.
    """
    return 0.5 * (vn * (inside + outside) + abs(vn) * (inside - outside))


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
    for e in flux.dom:
        for f in flux.dom[e]:
            for a in flux.dom[e, f]:
                for b in flux.dom[e, f, a]:
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
    for e in rhs.dom:
        for k in rhs.dom[e]:
            for j in rhs.dom[e, k]:
                for i in rhs.dom[e, k, j]:
                    rhs[e, k, j, i] = invJ[e, k, j, i] * (
                        reduce_sum(
                            D[i, a] * (V[0, e, k, j, a] * u[e, k, j, a])
                            for a in D.dom[i]
                        )
                        + reduce_sum(
                            D[j, b] * (V[1, e, k, b, i] * u[e, k, b, i])
                            for b in D.dom[j]
                        )
                        + reduce_sum(
                            D[k, c] * (V[2, e, c, j, i] * u[e, c, j, i])
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
    for e in rhs.dom:
        for k in rhs.dom[e]:
            for j in rhs.dom[e, k]:
                for i in rhs.dom[e, k, j]:
                    rhs[e, k, j, i] = rhs[e, k, j, i] - invJ[e, k, j, i] * (
                        # the faces t = -1 and t = +1, across k
                        (L[0, k] * flux[e, 0, j, i] + L[1, k] * flux[e, 1, j, i])
                        # r = -1 and r = +1, across i
                        + (L[0, i] * flux[e, 2, k, j] + L[1, i] * flux[e, 3, k, j])
                        # s = -1 and s = +1, across j
                        + (L[0, j] * flux[e, 4, k, i] + L[1, j] * flux[e, 5, k, i])
                    )


@program
def advection_rhs(u, V, invJ, nbr, inside, outside, vn, D, L, rhs):
    """grudge's ``WeakAdvectionOperator.operator``: the fluxes, the volume, the lift."""
    flux = Arr.zeros_like(vn)
    upwind(u, nbr, inside, outside, vn, flux)
    volume(D, V, invJ, u, rhs)
    lift(L, invJ, flux, rhs)


# }}}


# {{{ the schedules

#: The volume term and the lift at each node, on every loop; the fluxes with
#: them on the element.
fused = (
    Schedule(advection_rhs)
    .fuse("volume", "lift", loops=4)
    .fuse("upwind", "volume", loops=1)
)

#: The same, with the fluxes kept one element at a time.
contracted = fused.contract("flux")


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


#: The element a work group, its nodes the work items: ``e, k, j, i``.
NODES = ("g.0", "l.2", "l.1", "l.0")

#: The fluxes' launch: the element a work group, a face's nodes the work
#: items, its six faces in sequence on each: ``e, f, a, b``.
FACE_NODES = ("g.0", None, "l.1", "l.0")


def on(schedule: Schedule, call: str, axes: tuple) -> Schedule:
    """``schedule`` with the loops of ``call`` on ``axes``, outermost first.

    A program names each call's loops apart from the others' (the lift's
    ``e`` is ``e_1`` after the volume's ``e_0``); this reads the names off
    the statement ``call`` made.
    """
    stmt = next(s for s in schedule.term.stmts if s.id in ("S0", f"{call}.S0"))
    return schedule.tag(
        **{loop: axis for loop, axis in zip(stmt.inames, axes) if axis is not None}
    )


def launches(split: str = "one", target: str = "c") -> list[Schedule]:
    """The right-hand side as launches on a device, each a checked schedule.

    ``"volume, surface"`` is libParanumal's split: the volume term, then the
    fluxes computed where the lift reads them. ``"fluxes, volume"`` stores the
    fluxes first, then computes the volume term where the lift reads it.
    ``"three"`` stores both. ``"one"`` is one launch, as grudge runs it, both
    computed where the lift reads them: a face's flux once per node of the
    line through it.
    """
    if split == "three":
        return [
            on(Schedule(upwind, target=target), "upwind", FACE_NODES),
            on(Schedule(volume, target=target), "volume", NODES),
            on(Schedule(lift, target=target), "lift", NODES),
        ]
    if split == "volume, surface":
        return [
            on(Schedule(volume, target=target), "volume", NODES),
            on(Schedule(surface, target=target).substitute("flux"), "lift", NODES),
        ]
    if split == "fluxes, volume":
        both = Schedule(volume_and_lift, target=target).fuse("volume", "lift", loops=4)
        return [
            on(Schedule(upwind, target=target), "upwind", FACE_NODES),
            on(both, "volume", NODES),
        ]
    if split == "one":
        whole = (
            Schedule(advection_rhs, target=target)
            .substitute("flux")
            .fuse("volume", "lift", loops=4)
        )
        return [on(whole, "volume", NODES)]
    raise ValueError(f"no split {split!r}")


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
        return differentiation(order).T * weights / weights[:, None], ends
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
    """A smooth field on the periodic box ``[-1, 1]^3``, its geometry and the tables."""
    n = order + 1
    E = elements
    h = 2.0 / E
    nodes, _ = gll(order)
    v = np.array(VELOCITY)
    ne = E**3
    # node (i, j, k) of element (x, y, z), u[e, k, j, i]
    e = np.arange(ne)
    corner = np.stack([e % E, (e // E) % E, e // E**2], axis=1) * h - 1.0
    ref = (nodes + 1) * h / 2
    X = corner[:, 0, None, None, None] + ref[None, None, None, :]
    Y = corner[:, 1, None, None, None] + ref[None, None, :, None]
    Z = corner[:, 2, None, None, None] + ref[None, :, None, None]
    u = np.sin(np.pi * X) * np.cos(np.pi * Y) + 0.3 * np.sin(np.pi * Z)
    J = (h / 2) ** 3
    shape = (ne, n, n, n)
    V = np.stack([np.full(shape, J * (2 / h) * v[d]) for d in range(3)])
    vn = np.zeros((ne, 6, n, n))
    for f, (axis, side) in enumerate(FACES):
        vn[:, f] = (h / 2) ** 2 * side * v[axis]
    inside, outside = face_maps(order)
    D, L = tables(order, mass)
    return {
        "u": Arr.from_numpy(u),
        "V": Arr.from_numpy(V),
        "invJ": Arr.from_numpy(np.full(shape, 1 / J)),
        "nbr": Arr.from_numpy(box(E)),
        "inside": Arr.from_numpy(inside),
        "outside": Arr.from_numpy(outside),
        "vn": Arr.from_numpy(vn),
        "D": Arr.from_numpy(D),
        "L": Arr.from_numpy(L),
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
    V = data["V"].numpy()
    vn = data["vn"].numpy()
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
    u, V, invJ, nbr = (data[name] for name in ("u", "V", "invJ", "nbr"))
    inside, outside, vn, D, L = (
        data[name] for name in ("inside", "outside", "vn", "D", "L")
    )
    shape = u.numpy().shape
    rng = np.random.default_rng(1)
    volume_term = rng.normal(size=shape)
    flux = Arr.from_numpy(rng.normal(size=vn.numpy().shape))
    faces = {"u": u, "nbr": nbr, "inside": inside, "outside": outside, "vn": vn}
    return {
        "upwind": {**faces, "flux": Arr.zeros(vn.numpy().shape)},
        "volume": {"D": D, "V": V, "invJ": invJ, "u": u, "rhs": Arr.zeros(shape)},
        "lift": {
            "L": L,
            "invJ": invJ,
            "flux": flux,
            "rhs": Arr.from_numpy(volume_term),
        },
        "advection_rhs": inputs(),
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


def main() -> int:
    """Natively against grudge's dense operators, the fusion and window, the device."""
    from loopty.executor import LoopyExecutor

    stiffness_t, mass = factors()
    inverse = bool(np.allclose(np.linalg.inv(mass) @ mass, np.eye(N), atol=1e-13))
    data = inputs()
    advection_rhs(**data)
    expected = dense(data)
    got = data["rhs"].numpy()
    scale = np.abs(expected).max()
    gap = np.abs(got - expected).max() / scale
    native = bool(gap <= 1e-13)
    print(
        f"order {ORDER}, {ELEMENTS}**3 periodic hexahedra: the right-hand side is "
        f"grudge's, inverse mass and all: {native} (largest difference {gap:.0e} of "
        f"the largest; M^-1 M is the identity: {inverse})"
    )
    print()

    try:
        sizes = {"n": N, "ne": ELEMENTS**3}
        Schedule(advection_rhs, sizes=sizes).fuse("upwind", "lift", loops=2)
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

    print("on a device, every split's launches, every node on a work item:")
    legal = True
    for split in ("volume, surface", "fluxes, volume", "three", "one"):
        for schedule in launches(split):
            cast = [fact for fact in schedule.facts() if fact.kind != "buildable"]
            legal = legal and all(fact.status.value == "decided" for fact in cast)
            print(
                f"  {split}: {schedule.term.name}: {'.'.join(schedule.history)}: "
                f"{len(cast)} facts, all {cast[0].status.value}"
            )
    return 0 if native and inverse and kept.status.value == "tested" and legal else 1


if __name__ == "__main__":
    raise SystemExit(main())
