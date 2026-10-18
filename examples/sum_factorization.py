"""Sum factorization on quadrilaterals: two passes, fused on the outer loops only.

A tensor-product element's operator is a Kronecker product of one-dimensional
ones, and grudge applies it as a pass per reference axis (PR #418's
``_apply_operator_to_group``): for the weak derivative along ``r`` on a
quadrilateral, the differentiation matrix ``D`` along ``r`` and the quadrature
weights ``M`` along ``s``. Here each pass is a kernel over the elements and
the program runs one after the other::

    v = Arr.zeros_like(u)
    along_r(D, u, v)
    along_s(M, v, w)

Fusing the two passes is the obvious next step, and it is legal only so far:
``along_s`` at ``(e, a, b)`` reads ``v[e, a, m]`` for every ``m`` of the row,
which ``along_r`` stores at ``(e, a, m)``. Fused on the element and the row,
``(e, a)``, the first pass stores the whole row before the second reads it.
Fused on all three loops it would read ``v[e, a, m]`` for ``m > b`` before it
is stored, and isl refuses it with that pair.

Run this file three ways.

``python examples/sum_factorization.py``
    Runs the program natively and checks it against the Kronecker product of
    the two matrices applied to each element. Then asks for the fusion of
    every loop, which is refused with the pair and names the fusion that is
    accepted, and prints the fused schedule, its loops and the comparison of
    its compiled run with the native one.

``lanky check examples/sum_factorization.py``
    The two kernels' obligations, the program's (``v`` is defined where
    ``along_s`` reads it), and the casts of the fusion and of its window.

``loopty run examples/sum_factorization.py``
    Each kernel alone, the program as one kernel, and the fused schedule,
    compiled and compared with the native run.

What is worth reading here
--------------------------

*How deep a fusion goes is a choice, and each choice is checked.*
``fuse(weak_derivative.along_r.a, weak_derivative.along_s.a)`` fuses each
pass's loops down to the row, ``a``, the element and the row, and leaves the
inner loops one after the other; the refusal of the deeper fusion
names the pair it would run backwards and the shallower fusion it checked.
grudge's three-dimensional passes fuse the same way: on the element and the
row for ``r`` then ``s``, deeper for ``s`` then ``t``.

*What passes between the passes is one row.* Fused on the element and the
row, ``along_s`` reads only the row of ``v`` that ``along_r`` has just stored,
so ``contract(weak_derivative.v)`` keeps one row, ``lambda e, a, b: b``, in ``double
v_window[n]``, where the program stored ``ne * n * n`` cells; its
``no-clobber`` fact says no read sees its row overwritten, and putting the
elements on a group axis after it is refused, since every element would
share the row.

*The answer is a Kronecker product.* Flattened row by row, each element's
result is ``kron(D, M) @ u[e]``, the dense operator grudge's escape hatch
applies.
"""

from __future__ import annotations

import numpy as np
from lanky.prelude import Real

from loopty import Arr, Fin, IllegalCast, Schedule, kernel, program, reduce_sum

#: Four Gauss-Lobatto-Legendre points per axis (order 3), and eight elements.
NODES, ELEMENTS = 4, 8


@kernel
def along_r(
    F: Arr[Fin[n], Fin[n], Real],
    u: Arr[Fin[ne], Fin[n], Fin[n], Real],
    v: Arr[Fin[ne], Fin[n], Fin[n], Real],
):
    """``F`` applied along the first reference axis of every element."""
    for e, a, b in v.dom:
        v[e, a, b] = reduce_sum(F[a, m] * u[e, m, b] for m in F.dom[a])


@kernel
def along_s(
    G: Arr[Fin[n], Fin[n], Real],
    v: Arr[Fin[ne], Fin[n], Fin[n], Real],
    w: Arr[Fin[ne], Fin[n], Fin[n], Real],
):
    """``G`` applied along the second reference axis of every element."""
    for e, a, b in w.dom:
        w[e, a, b] = reduce_sum(G[b, m] * v[e, a, m] for m in G.dom[b])


@program
def weak_derivative(D, M, u, w):
    """The two passes, the first into an array of the program's own."""
    v = Arr.zeros_like(u)
    along_r(D, u, v)
    along_s(M, v, w)


#: The element and the row fused, the inner loops one after the other.
fused = Schedule(weak_derivative).fuse(
    weak_derivative.along_r.a, weak_derivative.along_s.a
)

#: The same, with v kept one row at a time.
contracted = fused.contract(weak_derivative.v)


def gll(nodes: int = NODES) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """The points, the weights and the differentiation matrix of the GLL rule."""
    interior = np.polynomial.legendre.Legendre.basis(nodes - 1).deriv().roots()
    x = np.concatenate(([-1.0], np.sort(interior), [1.0]))
    legendre = np.polynomial.legendre.Legendre.basis(nodes - 1)(x)
    weights = 2 / (nodes * (nodes - 1) * legendre**2)
    differences = x[:, None] - x[None, :]
    np.fill_diagonal(differences, 1.0)
    D = (legendre[:, None] / legendre[None, :]) / differences
    np.fill_diagonal(D, 0.0)
    D[0, 0] = -nodes * (nodes - 1) / 4
    D[-1, -1] = nodes * (nodes - 1) / 4
    return x, weights, D


def inputs(elements: int = ELEMENTS, seed: int = 0) -> dict:
    """``D`` along r, the weights along s, and a field on every element."""
    _x, weights, D = gll()
    rng = np.random.default_rng(seed)
    return {
        "D": Arr.from_numpy(D),
        "M": Arr.from_numpy(np.diag(weights)),
        "u": Arr.from_numpy(rng.normal(size=(elements, NODES, NODES))),
        "w": Arr.zeros((elements, NODES, NODES)),
    }


def kronecker(data: dict) -> np.ndarray:
    """``kron(D, M)`` applied to each element, flattened row by row."""
    D, M, u = data["D"].numpy(), data["M"].numpy(), data["u"].numpy()
    flat = u.reshape(len(u), -1) @ np.kron(D, M).T
    return flat.reshape(u.shape)


def example_inputs() -> dict:
    """The inputs ``loopty run`` gives each kernel, and the program."""
    data = inputs()
    middle = np.einsum("am,emb->eab", data["D"].numpy(), data["u"].numpy())
    return {
        "along_r": {
            "F": data["D"],
            "u": data["u"],
            "v": Arr.zeros((ELEMENTS, NODES, NODES)),
        },
        "along_s": {
            "G": data["M"],
            "v": Arr.from_numpy(middle),
            "w": Arr.zeros((ELEMENTS, NODES, NODES)),
        },
        "weak_derivative": inputs(),
    }


def main() -> int:
    """Run natively, the refused fusion, the accepted one, and its window."""
    from loopty.executor import LoopyExecutor, differences

    data = inputs()
    weak_derivative(**data)
    native = bool(np.allclose(data["w"].numpy(), kronecker(data)))
    print(f"native: each element is kron(D, M) applied to it: {native}")
    print()

    try:
        Schedule(weak_derivative).fuse(
            weak_derivative.along_r.b, weak_derivative.along_s.b
        )
    except IllegalCast as exc:
        print(f"refused: {exc}")
    print()

    print(f"accepted: {fused!r}")
    # no loop zeroes v: no read sees the zeros it is made with, so every
    # schedule of the program runs without them, the first fact below
    print(f"  the passes' loops, outermost first: {', '.join(fused.nest)}")
    for fact in fused.facts():
        print(f"  {fact.status.value:8} {fact.decided_by:4} {fact.statement}")
    fact = LoopyExecutor().differential(weak_derivative, fused, inputs())
    for line in differences(fact):
        print(f"  {line}")
    print()

    window = contracted.windows["v"]
    print(f"accepted: {contracted!r}")
    print(f"  v kept in {window.extents[0]} cells, {window.placement}")
    # the window's own fact; the definedness it rests on is fused's, above
    shown = {fact.id for fact in fused.facts()}
    for storage in contracted.facts():
        if storage.id not in shown:
            print(
                f"  {storage.status.value:8} {storage.decided_by:4} {storage.statement}"
            )
    kept = LoopyExecutor().differential(weak_derivative, contracted, inputs())
    print(f"  w: agrees with the native run -> {kept.status.value}")
    tested = fact.status.value == "tested" and kept.status.value == "tested"
    return 0 if native and tested else 1


if __name__ == "__main__":
    raise SystemExit(main())
