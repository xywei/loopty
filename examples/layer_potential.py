"""A layer potential by QBX: complex points, ``loopty.math``, and sumpy's CPU tile.

pytential evaluates the Laplace single layer of a density on a curve, at a
target on the curve, through a local expansion about a center just off the
curve (QBX). For a target ``z`` with center ``c``, the expansion of ``log |c
+ tau (z - c) - y|`` in ``tau`` to order ``P``, evaluated at ``tau = 1``, is
the line-Taylor form sumpy generates as straight-line code::

    Re( log(c - y) - sum_{k=1..P} ((c - z) / (c - y))**k / k )

Over complex points that is one line of a kernel, with ``real`` and ``log``
from ``loopty.math``:

    out[t] = SCALE * reduce_sum(
        w[s] * real(log(c[t] - y[s]) - reduce_sum(...))
        for s in y.dom
    )

The curve is the unit circle and the density ``cos(theta)``, whose single
layer on the circle is ``cos(theta) / 2``, so the answer is known.

Run this file three ways.

``python examples/layer_potential.py``
    Evaluates the layer potential natively at 32 targets from 512 sources,
    and checks it against the exact value. Then schedules it as sumpy's CPU
    tile, 16 targets at a time and 256 sources at a time, with the sources'
    loop outside the targets', and prints the facts of that schedule, its
    comparison with the native run, and the same order on a target loop that
    runs in sequence, which loopy cannot keep.

``lanky check examples/layer_potential.py``
    The kernel's obligations: every access in bounds, decided by isl, and
    the trace faithful to the body, with ``log`` and ``real`` interpreted by
    numpy.

``loopty run examples/layer_potential.py``
    The tiled schedule compiled on the C target and compared with the native
    run. The difference it prints is one of ``log``'s last bits, numpy's
    against the C library's, which can differ from machine to machine.

What is worth reading here
--------------------------

*Elementary functions are numpy's, in both runs.* ``log`` of a complex number
is numpy's natively and ``clog`` compiled (``cdouble_log`` on OpenCL), and
``real`` is ``creal``; each is computed in the type numpy computes it in.

*The tile is sumpy's.* ``LayerPotential.get_optimized_kernel`` splits the
targets by 16 onto a group and its work items, splits the sources by 256,
and runs the sources' outer loop outside the work items' loop
(``prioritize_loops([isrc_outer, itgt_inner])``). Here it is
``.prioritize("s_outer", "t_inner")``: each target's sum still adds its
sources in order, so nothing is reassociated, and the targets in a tile no
longer order one another, so a dependence between two of them would be
refused. On the C target the 16 targets are an ``ilp`` loop, so that loopy
keeps an accumulator for each of them, ``acc[16]``; on a loop in sequence
loopy keeps one and would drop the order, and the schedule says so. On a
device, the sumpy schedule is written as it is, ``.tag(t_outer="g.0",
t_inner="l.0")``.
"""

from __future__ import annotations

import numpy as np
from lanky.prelude import Real

from loopty import Arr, Fin, Schedule, kernel, reduce_sum
from loopty.math import log, real

#: The order of the expansion, the targets and the sources.
P, TARGETS, SOURCES = 4, 32, 512

#: The Laplace single layer's constant: G(x, y) = -log|x - y| / (2 pi).
SCALE = -1 / (2 * np.pi)


@kernel
def single_layer(
    z: Arr[Fin[nt], np.complex128],
    c: Arr[Fin[nt], np.complex128],
    y: Arr[Fin[ns], np.complex128],
    w: Arr[Fin[ns], Real],
    out: Arr[Fin[nt], Real],
):
    """The QBX single layer at each target, through its center's expansion."""
    for t in out.dom:
        out[t] = SCALE * reduce_sum(
            w[s]
            * real(
                log(c[t] - y[s])
                - reduce_sum(
                    ((c[t] - z[t]) / (c[t] - y[s])) ** k / k
                    for k in Fin[P + 1]
                    if k >= 1
                )
            )
            for s in y.dom
        )


def circle(targets: int = TARGETS, sources: int = SOURCES) -> dict:
    """Targets on the unit circle, centers inside it, and the density's sources.

    The sources are the trapezoid rule's points, each weighted by the rule's
    weight times the density ``cos(theta)``; the centers are four source
    spacings inside the circle, one per target.
    """
    theta_s = 2 * np.pi * (np.arange(sources) + 0.5) / sources
    theta_t = 2 * np.pi * np.arange(targets) / targets
    z = np.exp(1j * theta_t)
    radius = 4 * 2 * np.pi / sources
    return {
        "z": Arr.from_numpy(z),
        "c": Arr.from_numpy((1 - radius) * z),
        "y": Arr.from_numpy(np.exp(1j * theta_s)),
        "w": Arr.from_numpy(2 * np.pi / sources * np.cos(theta_s)),
        "out": Arr.zeros(targets),
    }


def exact(targets: int = TARGETS) -> np.ndarray:
    """The single layer of ``cos(theta)`` on the unit circle: ``cos(theta) / 2``."""
    return np.cos(2 * np.pi * np.arange(targets) / targets) / 2


#: sumpy's CPU tile, on the C target: 16 targets at a time, each with an
#: accumulator of its own, and 256 sources at a time outside them.
tiled = (
    Schedule(single_layer, target="c")
    .split("t", 16)
    .tag(t_inner="ilp")
    .split("s", 256)
    .prioritize("s_outer", "t_inner")
    .example(**circle())
)


def example_inputs() -> dict:
    """The inputs ``loopty run`` gives the kernel."""
    return circle()


def main() -> int:
    """Evaluate natively, then show the tile, its run, and the order loopy drops."""
    from loopty.executor import LoopyExecutor

    data = circle()
    single_layer(**data)
    error = float(np.abs(data["out"].numpy() - exact()).max())
    print(f"{TARGETS} targets, {SOURCES} sources, order {P}")
    print(f"largest error against cos(theta) / 2: {error:.1e}")
    ok = error < 1e-8

    print()
    print(f"schedule: {tiled!r}")
    for fact in tiled.facts():
        print(f"  {fact.status.value:8} {fact.decided_by or '-':4} {fact.statement}")
    # The difference is one of log's last bits, numpy's and the C library's,
    # which may differ from machine to machine; whether it is inside the
    # tolerance does not.
    fact = LoopyExecutor().differential(single_layer, tiled, circle())
    for name, detail in fact.provenance["outputs"].items():
        print(
            f"  {name}: the compiled run agrees with the native run to the "
            f"{detail['exactness']} tolerance: {detail['agree']} -> "
            f"{fact.status.value}"
        )
    ok = ok and fact.status.value == "tested"

    print()
    sequential = (
        Schedule(single_layer, target="c")
        .split("t", 16)
        .split("s", 256)
        .prioritize("s_outer", "t_inner")
    )
    buildable, reason = sequential.buildable
    print(f"schedule: {sequential!r}")
    print(f"  buildable: {buildable}")
    print(f"  reason: {reason}")
    return 0 if ok and not buildable else 1


if __name__ == "__main__":
    raise SystemExit(main())
