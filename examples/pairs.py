"""Symmetric pair interactions over the lower triangle: an array over a domain.

Between ``n`` particles there are ``n(n - 1)/2`` distinct pairs, the points of
``{(i, j) : 0 <= j < i < n}``. An array library offers two ways to store one
value per pair: an ``n x n`` matrix, half of it wasted and masked, or a flat
list of pairs behind two index arrays, which throws the triangle away and
leaves every access to be trusted. Here the triangle is the type:

.. code-block:: python

    f: Arr[Where[i: Fin[n], j: Fin[n], j < i], Real]

``Where`` takes binders and the constraints that cut their box (see
:mod:`loopty.domain`); ``f.dom`` runs over ``i`` and ``f.dom[i]`` over the
``j`` below it. One statement writes the interaction of every pair, once, over
the triangle, and a second sums for every particle the pairs it is in: its row
of the triangle, ``f[p, j]`` for ``j < p``, and its column, ``f[k, p]`` for
``k > p``, which is the symmetry the triangle stores only half of.

Three things are on show.

*The in-bounds obligations are decided over the exact triangle.* ``f[k, p]`` is
read under the reduction's condition ``k > p``, and isl decides that ``(k, p)``
is a point of the triangle there, which is a question about the domain and not
about the box around it: ``f[p, p]`` is a cell of the box and not of ``f``, and
it would be refused.

*One type, two layouts.* ``Schedule(pairs)`` keeps ``f`` in the box of its
binders, ``n x n`` cells of which the triangle uses fewer than half, at the
affine address loopy computes itself. ``Schedule(pairs).pack("f")`` keeps the
triangle's cells and no others, row after row, and reads ``f[i, j]`` as
``f[off_f[i] + j]`` through a table of row starts the executor computes from
the domain. Neither needs an index array, and the facts are the same for both,
since they are about the cells and not about where the cells are kept.

*The interaction is softened.* ``q[i] q[j] / (1 + r^2)`` rather than a Coulomb
``1 / r``, so the body is arithmetic that traces as it is, with no call to
write for a square root.

Run this file three ways.

``python examples/pairs.py``
    Runs the kernel natively on both layouts, checks it against a dense numpy
    reference, and compares both schedules' compiled runs with the native one.

``lanky check examples/pairs.py``
    Prints the ledger of the kernel's obligations.

``loopty run examples/pairs.py``
    Compiles both schedules for the C target and compares each with the native
    run.
"""

from __future__ import annotations

import numpy as np
from lanky.prelude import Real

from loopty import Arr, Fin, Schedule, Where, kernel, reduce_sum

#: How many particles. Tiny: the point is the ledger and the layouts.
PARTICLES = 6


@kernel
def pairs(
    x: Arr[Fin[n], Real],
    y: Arr[Fin[n], Real],
    q: Arr[Fin[n], Real],
    f: Arr[Where[i: Fin[n], j: Fin[n], j < i], Real],
    e: Arr[Fin[n], Real],
):
    """The interaction of every pair, then every particle's share of them."""
    for i in f.dom:
        for j in f.dom[i]:
            dx = x[i] - x[j]
            dy = y[i] - y[j]
            f[i, j] = q[i] * q[j] / (1.0 + dx * dx + dy * dy)
    for p in e.dom:
        e[p] = reduce_sum(f[p, j] for j in f.dom[p]) + reduce_sum(
            f[k, p] for k in e.dom if k > p
        )


#: The declared domain of ``f``, which is what an array for it is built over.
TRIANGLE = pairs.arg_types["f"].domain


def scene(particles: int = PARTICLES, storage: str = "box", seed: int = 0) -> dict:
    """Random particles with charges, and the arrays the kernel takes."""
    rng = np.random.default_rng(seed)
    return {
        "x": Arr.from_numpy(rng.random(particles)),
        "y": Arr.from_numpy(rng.random(particles)),
        "q": Arr.from_numpy(rng.normal(size=particles)),
        "f": Arr.zeros(TRIANGLE, n=particles, storage=storage),
        "e": Arr.zeros(Fin[particles]),
    }


def dense(data: dict) -> np.ndarray:
    """The same energies from the full ``n x n`` matrix, diagonal left out."""
    xs, ys, charges = data["x"].numpy(), data["y"].numpy(), data["q"].numpy()
    r2 = (xs[:, None] - xs[None, :]) ** 2 + (ys[:, None] - ys[None, :]) ** 2
    matrix = charges[:, None] * charges[None, :] / (1.0 + r2)
    np.fill_diagonal(matrix, 0.0)
    return matrix.sum(axis=1)


#: The triangle in the box of its binders, and the triangle packed.
boxed = Schedule(pairs, target="c").example(**scene())
packed = Schedule(pairs, target="c").pack("f").example(**scene(storage="packed"))


def example_inputs() -> dict:
    """The inputs ``loopty run`` gives a kernel no schedule names."""
    return scene()


def main() -> int:
    """Run natively on both layouts, then compiled on both, and compare."""
    from loopty.executor import LoopyExecutor

    count = PARTICLES * (PARTICLES - 1) // 2
    print(f"{PARTICLES} particles, {count} pairs")
    wanted = None
    agree = True
    for storage in ("box", "packed"):
        data = scene(storage=storage)
        pairs(**data)
        wanted = dense(data)
        energies = data["e"].numpy()
        agree = agree and bool(np.allclose(energies, wanted))
        f = data["f"]
        table = ""
        if storage == "packed":
            table = f", and a table of row starts {f.table().tolist()}"
        print(f"f {storage}: {f.numpy().size} cells for {count} pairs{table}")
    print("energies =", np.array2string(energies, precision=3))
    print("dense    =", np.array2string(wanted, precision=3))
    print(f"both layouts agree with the dense reference: {agree}")

    executor = LoopyExecutor()
    ok = agree
    for schedule, storage in ((boxed, "box"), (packed, "packed")):
        print()
        print(f"schedule: {schedule!r}")
        fact = executor.differential(pairs, schedule, scene(storage=storage))
        for name, detail in fact.provenance["outputs"].items():
            print(
                f"  {name}: difference {detail['difference']:.3g} within "
                f"{detail['tolerance']:.3g} ({detail['exactness']}) -> "
                f"{fact.status.value}"
            )
        ok = ok and fact.status.value == "tested"
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
