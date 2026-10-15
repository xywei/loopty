"""sumpy's L2P for the 2-D Laplace kernel: the theorem, the kernels and the schedule.

A fast multipole method ends by evaluating each leaf box's local expansion at
the targets the box owns, the L2P. sumpy does it in ``E2PFromSingleBox``, for
the 2-D Laplace kernel with the compressed Taylor expansion its
``DefaultExpansionFactory`` picks. That expansion stores ``2p + 1`` of the
``(p + 1)(p + 2) / 2`` derivatives through order ``p`` and reconstructs the
others from the PDE: the kernel is harmonic, ``G_xx = -G_yy``, so a derivative
with two more ``x``'s is minus the one with two more ``y``'s. This file is that
L2P, read top to bottom, in three layers:

1. **The algorithm, in lanky.** ``compressed_taylor`` is the claim sumpy's
   reconstruction rests on, for every order (lanky#86): a table that agrees
   with the derivatives where sumpy stores them and follows the recurrence
   everywhere else is the table of derivatives. ``on_every_triangle`` is its
   form for a kernel of order ``p``: box by box, on the triangle ``a + b <=
   p``, what sumpy stores taken as a table ``c[i, a, b]``. Lean proves both,
   for every order, with Mathlib; ``lanky check --show-lean`` prints the Lean.
2. **The kernels, in loopty.** ``unpack`` reads each box's ``2p + 1``
   coefficients, in sumpy's storage order, into that table: through sumpy's
   storage index, ``storage_index``, a layout whose three facts say it is one
   for one onto the ``2p + 1`` addresses, and its postcondition says which
   coefficient each cell holds. ``reconstruct`` fills each box's triangle
   from the table. Its postcondition is what it stores, decided from its
   statements, and is the theorem's hypotheses about the kernel, so the
   theorem applies: the triangle is the table of derivatives, for any
   harmonic table whose derivatives the coefficients hold. ``evaluate`` sums
   the Taylor polynomial at each target, and ``l2p`` runs the three in turn,
   the table and the triangles arrays it makes.
3. **The schedule.** Running ``b`` outside ``a`` is refused, with the pair of
   instances it would reorder: ``R[i, a, b]`` reads ``R[i, a - 2, b + 2]``,
   stored in an earlier row. ``fused`` runs each box's stored coefficients,
   its triangle and its targets before the next box's, as sumpy does, two
   casts isl decides and Lean proves. It keeps every box's triangle; keeping
   only the box's at hand, as sumpy's registers do, is a window
   (``contract``), which is refused for now, with the reason: a window of an
   array over a domain is loopty #244, and of a recurrence #213.

Run it three ways.

``python examples/l2p.py``
    Runs ``l2p`` natively against numpy on a quadtree's leaves, compares each
    target's potential with the sum over the charges, shows the refused
    interchange, and runs ``fused`` compiled.

``lanky check examples/l2p.py``
    The ledger: the theorems; the layout's facts; each kernel's accesses,
    writes and postcondition; the theorem applied to it; the program's
    definedness; the casts of the schedules; and the row that compares
    ``l2p`` with sumpy's own kernel.

``loopty run examples/l2p.py``
    Every kernel and the program, and each schedule, compiled for the C
    target and compared with the native run.

How faithful. The coefficients are sumpy's ``src_expansions`` row by row, in
its storage order, and ``rscale`` is sumpy's; the boxes are leaves owning
contiguous runs of targets, as boxtree's do, so ``box_target_starts`` is the
scan of the counts. sumpy reconstructs the triangle symbolically, inside the
straight-line code it generates for each target, and its Taylor sum is
sympy's; here both are loops. ``sumpy_l2p`` is sumpy's own kernel on this
file's inputs: the potentials agree to round-off (``benchmarks/l2p.py``
recorded them, at the commit below, on loopy main).

The code it mirrors, at sumpy 13a48a0:

* ``E2PFromSingleBox``: https://github.com/inducer/sumpy/blob/13a48a0260eec8cf7ac67f1faca701205d9096ae/sumpy/e2p.py#L123-L204
* the straight-line evaluation it inlines, ``make_e2p_loopy_kernel``:
  https://github.com/inducer/sumpy/blob/13a48a0260eec8cf7ac67f1faca701205d9096ae/sumpy/expansion/loopy.py#L51-L152
* ``evaluate``, the Taylor sum: https://github.com/inducer/sumpy/blob/13a48a0260eec8cf7ac67f1faca701205d9096ae/sumpy/expansion/local.py#L285-L310
* the reconstruction from the stored derivatives:
  https://github.com/inducer/sumpy/blob/13a48a0260eec8cf7ac67f1faca701205d9096ae/sumpy/expansion/__init__.py#L576-L586
* the storage index: https://github.com/inducer/sumpy/blob/13a48a0260eec8cf7ac67f1faca701205d9096ae/sumpy/expansion/__init__.py#L687-L753
* the expansion picked for 2-D Laplace: https://github.com/inducer/sumpy/blob/13a48a0260eec8cf7ac67f1faca701205d9096ae/sumpy/expansion/__init__.py#L1046-L1083

and loopy #970's L2P microkernel, the kernel-independent half of the same
L2P: https://github.com/inducer/loopy/blob/4ae406f29e084b67c60677d03b8a90c65c37d9f4/examples/python/compute-examples/l2p-tiled-basis-compute.py#L77-L170
"""

from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np
from lanky import theorem
from lanky.prelude import Fn, Nat, Real

from loopty import (
    Arr,
    Fin,
    Schedule,
    Where,
    kernel,
    layout,
    program,
    reduce_sum,
    table,
    upstream,
    when,
)
from loopty.math import ipow

#: The order of the expansion. sumpy generates a kernel per order, and so is
#: this one written at one; ``benchmarks/l2p.py`` writes it out at others.
ORDER = 10
P = ORDER
Q = ORDER + 1

#: 1 / k!, for the Taylor sum.
INV_FACT = table([1 / math.factorial(k) for k in range(Q)])

#: The constant factor of sumpy's LaplaceKernel(2), -1 / (2 pi).
SCALING = -1 / (2 * math.pi)


# {{{ 1. the algorithm


@theorem
def compressed_taylor(
    D: Fn[Nat, Nat, Real],
    R: Fn[Nat, Nat, Real],
    pde: all(D[a + 2, b] + D[a, b + 2] == 0 for a in Nat for b in Nat),
    stored: all(R[a, b] == D[a, b] for a in Nat for b in Nat if a < 2),
    recurrence: all(R[a + 2, b] == -R[a, b + 2] for a in Nat for b in Nat),
) -> all(R[a, b] == D[a, b] for a in Nat for b in Nat):
    """What sumpy stores of a harmonic table, reconstructed, is the table (lanky#86).

    ``D[a, b]`` is any table that satisfies Laplace's equation at every
    order, as the derivatives ``d^(a+b) G / dx^a dy^b`` of a harmonic ``G``
    do. ``R`` agrees with it where sumpy stores, ``a < 2``, and follows the
    recurrence everywhere else. By induction on ``a``, two at a time: the
    recurrence, the induction hypothesis at ``(a, b + 2)`` and the PDE.
    """


@theorem
def on_every_triangle(
    n: Nat,
    p: Nat,
    c: Fn[Nat, Nat, Nat, Real],
    D: Fn[Nat, Nat, Nat, Real],
    R: Fn[Nat, Nat, Nat, Real],
    pde: all(
        D[i, a + 2, b] + D[i, a, b + 2] == 0 for i in Fin[n] for a in Nat for b in Nat
    ),
    stored: all(
        c[i, a, b] == D[i, a, b]
        for i in Fin[n]
        for a in Nat
        for b in Nat
        if a < 2
        if a + b <= p
    ),
    copies: all(
        R[i, a, b] == c[i, a, b]
        for i in Fin[n]
        for a in Nat
        for b in Nat
        if a < 2
        if a + b <= p
    ),
    recurrence: all(
        R[i, a + 2, b] == -R[i, a, b + 2]
        for i in Fin[n]
        for a in Nat
        for b in Nat
        if a + b + 2 <= p
    ),
) -> all(
    R[i, a, b] == D[i, a, b] for i in Fin[n] for a in Nat for b in Nat if a + b <= p
):
    """Box by box, on the triangle ``a + b <= p``: what a kernel of order ``p`` cites.

    ``c[i, a, b]`` is what sumpy stores for box ``i``, the derivatives with
    ``a < 2``; ``R`` copies them and follows the recurrence wherever the
    derivative it gives is inside the order. Then ``R`` is ``D`` on every
    box's triangle.
    """


# }}}


# {{{ sumpy's storage index


def centre(a: Fin[1], b: Fin[1]) -> Fin[2 * P + 1]:
    """The value at the centre, stored first."""
    return 0


def derivatives(
    cell: Where[a: Fin[2], b: Fin[Q], (a + b >= 1) & (a + b < Q)],
) -> Fin[2 * P + 1]:
    """The two derivatives of each order a + b, the one with a = 0 first."""
    a, b = cell
    return 2 * (a + b) - 1 + a


#: sumpy's get_storage_index for the 2-D Laplace local expansion of order P.
storage_index = layout(centre, derivatives)


# }}}


# {{{ 2. the kernels


@kernel
def unpack(
    coeffs: Arr[Fin[nb], Fin[2 * P + 1], Real],
    C: Arr[Where[i: Fin[nb], a: Fin[2], b: Fin[Q], a + b <= P], Real],
) -> all(C[i, 0, 0] == coeffs[i, 0] for i in Fin[nb]) & all(
    C[i, a, b] == coeffs[i, 2 * (a + b) - 1 + a]
    for i in Fin[nb]
    for a in Nat
    for b in Nat
    if a < 2
    if a + b >= 1
    if a + b <= P
):
    """What sumpy stores for each box, read through its storage index as a table.

    The postcondition is sumpy's storage order, ``2 (a + b) - 1 + a`` after
    the value at the centre, decided from what the body stores.
    """
    for i in C.dom:
        for a in C.dom[i]:
            for b in C.dom[i, a]:
                with when(a + b == 0):
                    C[i, a, b] = coeffs[i, centre(a, b)]
                with when(a + b >= 1):
                    C[i, a, b] = coeffs[i, derivatives((a, b))]


@kernel(uses=[on_every_triangle])
def reconstruct(
    C: Arr[Where[i: Fin[nb], a: Fin[2], b: Fin[Q], a + b <= P], Real],
    R: Arr[Where[i: Fin[nb], a: Fin[Q], b: Fin[Q], a + b <= P], Real],
) -> all(
    R[i, a, b] == C[i, a, b]
    for i in Fin[nb]
    for a in Nat
    for b in Nat
    if a < 2
    if a + b <= P
) & all(
    R[i, a + 2, b] == -R[i, a, b + 2]
    for i in Fin[nb]
    for a in Nat
    for b in Nat
    if a + b + 2 <= P
):
    """Each box's triangle of coefficients, from the ``2p + 1`` sumpy stores.

    ``G_xx = -G_yy``: a coefficient with two more ``x``'s is minus the one
    with two more ``y``'s, which an earlier row of the triangle stored.
    """
    for i in R.dom:
        for a in R.dom[i]:
            for b in R.dom[i, a]:
                with when(a < 2):
                    R[i, a, b] = C[i, a, b]
                with when(a >= 2):
                    R[i, a, b] = -R[i, a - 2, b + 2]


#: What sumpy stores for every box, as a table: ``a < 2`` and ``a + b <= P``.
STORED = unpack.arg_types["C"].domain

#: Every box's triangle ``a + b <= P``: the domain of the coefficients.
TRIANGLES = reconstruct.arg_types["R"].domain


@kernel
def evaluate(
    R: Arr[Where[i: Fin[nb], a: Fin[Q], b: Fin[Q], a + b <= P], Real],
    cx: Arr[Fin[nb], Real],
    cy: Arr[Fin[nb], Real],
    cnt: Arr[Fin[nb], Nat],
    tx: Arr[Fin[nb], Fin[cnt], Real],
    ty: Arr[Fin[nb], Fin[cnt], Real],
    pot: Arr[Fin[nb], Fin[cnt], Real],
    rscale: Real,
):
    """Each target's Taylor polynomial: the sum over its box's triangle.

    ``ipow(x, a)`` is ``x`` to the ``a``, multiplied out as loopy multiplies
    an integer power, in both runs; ``x**a`` would be the C library's ``pow``
    in both, as numpy computes it.
    """
    for i in pot.dom:
        for t in pot.dom[i]:
            x = (tx[i, t] - cx[i]) / rscale
            y = (ty[i, t] - cy[i]) / rscale
            pot[i, t] = SCALING * reduce_sum(
                R[i, a, b] * ipow(x, a) * ipow(y, b) * INV_FACT[a] * INV_FACT[b]
                for a in R.dom[i]
                for b in R.dom[i, a]
            )


@program
def l2p(coeffs, cx, cy, cnt, tx, ty, pot, rscale):
    """sumpy's L2P: every box's stored coefficients, its triangle, its targets."""
    C = Arr.zeros(STORED, nb=cx.dom.size)
    R = Arr.zeros(TRIANGLES, nb=cx.dom.size)
    unpack(coeffs, C)
    reconstruct(C, R)
    evaluate(R, cx, cy, cnt, tx, ty, pot, rscale)


# }}}


# {{{ 3. the schedules

#: Box by box: its stored coefficients, its triangle, then its targets, the
#: order sumpy runs them in. Each fusion is of the loop over the boxes alone.
fused = (
    Schedule(l2p, target="c")
    .fuse("unpack", "reconstruct", loops=1)
    .fuse("reconstruct", "evaluate", loops=1)
)


def refused_interchange() -> Schedule:
    """Columns of the triangle outside its rows: refused, with the pair it reorders."""
    return Schedule(reconstruct, target="c").interchange("b", "a")


# }}}


# {{{ the inputs, and sumpy's own L2P on them


def derivative(a: int, b: int, z: complex) -> float:
    """``d^(a+b) log|z| / dx^a dy^b``: ``Re(i^b (-1)^(n-1) (n-1)! / z^n)``, n = a+b."""
    n = a + b
    if n == 0:
        return math.log(abs(z))
    return float(((1j) ** b * (-1) ** (n - 1) * math.factorial(n - 1) / z**n).real)


#: The charges whose potential the expansions hold: far from the unit square.
CHARGES = [
    (complex(2.5, 1.5), 1.0),
    (complex(2.4, 1.7), -0.5),
    (complex(2.7, 1.4), 0.75),
]


def tree(level: int = 2, per_box: int = 6, seed: int = 0) -> dict:
    """A uniform quadtree's leaves in Morton order, each with a few targets."""
    rng = np.random.default_rng(seed)
    side = 2**level
    h = 1.0 / side
    codes = np.arange(side * side)
    ix = sum(((codes >> (2 * k)) & 1) << k for k in range(level))
    iy = sum(((codes >> (2 * k + 1)) & 1) << k for k in range(level))
    cx, cy = (ix + 0.5) * h, (iy + 0.5) * h
    counts = rng.integers(1, 2 * per_box, size=codes.size)
    owner = np.repeat(codes, counts)
    tx = cx[owner] + h * (rng.random(owner.size) - 0.5)
    ty = cy[owner] + h * (rng.random(owner.size) - 0.5)
    rscale = 2 * h / P
    coeffs = np.zeros((codes.size, 2 * P + 1))
    for box in range(codes.size):
        for source, charge in CHARGES:
            z = complex(cx[box], cy[box]) - source
            coeffs[box, 0] += charge * derivative(0, 0, z)
            for n in range(1, Q):
                for a in (0, 1):
                    coeffs[box, 2 * n - 1 + a] += (
                        charge * derivative(a, n - a, z) * rscale**n
                    )
    return {"coeffs": coeffs, "cx": cx, "cy": cy, "counts": counts, "tx": tx, "ty": ty,
            "rscale": rscale}


def direct(tx: np.ndarray, ty: np.ndarray) -> np.ndarray:
    """The potential of the charges at the targets, summed directly."""
    z = tx + 1j * ty
    return SCALING * sum(charge * np.log(np.abs(z - s)) for s, charge in CHARGES)


def example_inputs() -> dict:
    """The arguments of ``l2p``, of each kernel, and of the schedules."""
    t = tree()
    counts = t["counts"]
    boxes = counts.size
    stored = {
        "coeffs": Arr.from_numpy(t["coeffs"]),
        "cx": Arr.from_numpy(t["cx"]),
        "cy": Arr.from_numpy(t["cy"]),
        "cnt": Arr.from_numpy(counts),
        "tx": Arr.ragged(counts, values=t["tx"]),
        "ty": Arr.ragged(counts, values=t["ty"]),
        "pot": Arr.ragged(counts),
        "rscale": t["rscale"],
    }
    triangles = Arr.zeros(TRIANGLES, nb=boxes)
    return {
        "l2p": stored,
        "unpack": {"coeffs": stored["coeffs"], "C": Arr.zeros(STORED, nb=boxes)},
        "reconstruct": {
            "C": Arr.from_cells(
                STORED, np.linspace(-1.0, 1.0, (2 * Q - 1) * boxes), nb=boxes
            ),
            "R": triangles,
        },
        "evaluate": {
            "R": Arr.from_cells(
                TRIANGLES, np.linspace(-1.0, 1.0, triangles.cells().size), nb=boxes
            ),
            **{k: v for k, v in stored.items() if k != "coeffs"},
        },
    }


#: What sumpy's own E2PFromSingleBox computes on ``example_inputs()``,
#: recorded by ``benchmarks/l2p.py record`` with sumpy and loopy main.
RECORDED = Path(__file__).with_name("l2p_sumpy.json")


@upstream(l2p)
def sumpy_l2p(coeffs, cx, cy, cnt, tx, ty, pot, rscale):
    """sumpy's own E2PFromSingleBox at 13a48a0, run on these inputs and recorded."""
    recorded = json.loads(RECORDED.read_text())
    if not np.array_equal(np.asarray(recorded["coeffs"]), coeffs.numpy()):
        raise ValueError(
            f"{RECORDED.name} is of other inputs; record it again with "
            "benchmarks/l2p.py record"
        )
    return {"pot": recorded["pot"]}


# }}}


def main() -> int:
    """Run natively, compare with the charges, show the refusal, run ``fused``."""
    from loopty.executor import LoopyExecutor
    from loopty.schedule import IllegalCast

    data = example_inputs()["l2p"]
    l2p(**data)
    got = data["pot"].numpy()
    want = direct(data["tx"].numpy(), data["ty"].numpy())
    error = float(np.max(np.abs(got - want)) / np.max(np.abs(want)))
    boxes, targets = data["cnt"].numpy().size, got.size
    print(
        f"order {P}: {boxes} boxes, {targets} targets, {2 * P + 1} coefficients "
        f"stored per box, {Q * (Q + 1) // 2} on its triangle"
    )
    print(f"largest error against the charges' potential: {error:.1e}")
    print()
    try:
        refused_interchange()
    except IllegalCast as exc:
        print(f"refused: {exc}")
    print()
    print(f"schedule: {fused!r}")
    for fact in fused.facts():
        print(f"  {fact.status.value:8} {fact.decided_by or '-':4} {fact.statement}")
    try:
        fused.contract("R")
    except ValueError as exc:
        print(f"  refused: {exc}")
    run = LoopyExecutor().differential(l2p, fused, example_inputs()["l2p"])
    for name, detail in run.provenance["outputs"].items():
        print(
            f"  {name}: difference {detail['difference']:.3g} within "
            f"{detail['tolerance']:.3g} ({detail['exactness']}) -> {run.status.value}"
        )
    return 0 if error < 1e-8 and run.status.value == "tested" else 1


if __name__ == "__main__":
    raise SystemExit(main())
