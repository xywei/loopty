"""A tiling that is illegal, the witness that says so, and the skew that fixes it.

One-dimensional Jacobi in time::

    u[t + 1, i] = (u[t, i - 1] + u[t, i + 1]) / 2

The dependences are the classic pair, ``(1, 1)`` and ``(1, -1)``: the next time
level at ``i`` needs the current level at ``i - 1`` and at ``i + 1``. Tiling the
``(t, i)`` nest rectangularly cuts both of them, because a tile that runs later
in space carries the earlier half of a dependence that the tile below it already
needed. Skewing the space axis by the time axis turns both vectors into
non-negative ones, and then the rectangular tiling is legal. This is the oldest
worked example in the polyhedral literature, and the point of putting it here is
not that loopty can tile a stencil; it is that loopty *refuses* to, and names the
two instances that make it wrong.

Run this file three ways.

``python examples/stencil_skew.py``
    Runs the kernel natively, prints the rejection and its witness, builds the
    skewed schedule, runs it on the C target and compares the two arrays.

``lanky check examples/stencil_skew.py``
    Prints the ledger of the kernel's own obligations: every access in bounds
    (the guard is what puts ``u[t + 1, i]`` in bounds), the writes disjoint,
    the source order monotone on the dependences.

``loopty run examples/stencil_skew.py``
    Compiles the skewed and tiled schedule and compares it with the native run.

The guard is a ``when`` block and not an ``if``. A Python ``if`` on a value the
kernel computes cannot be traced, because tracing would have to choose a branch;
``when`` records the condition, narrows the statement's isl domain by it, and
under plain ``python`` masks the writes of the block instead of skipping them.
Sizes come from the data on both paths: ``steps.size`` is an ``int`` on a real
array and a term while tracing, so the interior condition is written once.
"""

from __future__ import annotations

import numpy as np
from lanky.prelude import Real

from loopty import Arr, Fin, Schedule, kernel, when

#: Sixteen time levels over sixteen points. Large enough that a tile of eight by
#: eight is a real tiling with four tiles and a boundary between them, small
#: enough that the whole demo compiles and runs in about a second. The checks
#: themselves are symbolic in ``nt`` and ``nx``; these numbers are the size the
#: run uses and the size the printed witness is instantiated at.
NT, NX = 16, 16


@kernel
def jacobi(u: Arr[Fin[nt], Fin[nx], Real]):
    """Average each interior point's neighbours into the next time level."""
    steps = u.dom
    for t in steps:
        row = u.dom[t]
        for i in row:
            with when((t + 1 < steps.size) & (i > 0) & (i + 1 < row.size)):
                u[t + 1, i] = (u[t, i - 1] + u[t, i + 1]) / 2


def spike(nt: int = NT, nx: int = NX) -> np.ndarray:
    """A unit spike in the middle of the first time level, zeros elsewhere.

    A ramp would be a fixed point of this averaging, which makes for a dull
    demo; a spike spreads, so the printed corner shows the sweep doing work.
    """
    u = np.zeros((nt, nx))
    u[0, nx // 2] = 1.0
    return u


def initial(nt: int = NT, nx: int = NX) -> Arr:
    """The same data as a loopty runtime array.

    The kernel iterates ``u.dom``, so the array it is handed natively has to be
    an :class:`~loopty.arr.Arr` and not a bare numpy array: a domain is what an
    ``Arr`` has and ndarray does not. The compiled run takes either.
    """
    return Arr.from_numpy(spike(nt, nx))


def reference(u: np.ndarray) -> np.ndarray:
    """The same sweep written out by hand, as the thing to agree with."""
    out = u.copy()
    for t in range(len(out) - 1):
        for i in range(1, out.shape[1] - 1):
            out[t + 1, i] = (out[t, i - 1] + out[t, i + 1]) / 2
    return out


def rejected_tiling() -> tuple[str, tuple]:
    """Ask for the illegal tiling and return the message and the witness.

    The schedule the cast was asked of is untouched by the rejection, which is
    why this can be called from a demo, a test, and a docstring without leaving
    anything half-transformed behind.
    """
    from loopty.schedule import IllegalCast

    schedule = Schedule(jacobi, target="c", sizes={"nt": NT, "nx": NX})
    try:
        schedule.tile("t", "i", 8, 8)
    except IllegalCast as refused:
        return str(refused), refused.witness
    raise AssertionError("the rectangular tiling of a Jacobi stencil is illegal")


def skewed() -> Schedule:
    """The legal schedule: skew the space axis by time, then tile."""
    return (
        Schedule(jacobi, target="c", sizes={"nt": NT, "nx": NX})
        .skew("i", by="t")
        .tile("t", "i", 8, 8)
    )


#: What ``loopty run`` compiles and compares. Building it at module level is
#: safe because it is the schedule that passes; the rejected one lives in
#: :func:`rejected_tiling`, where the exception is the answer rather than a
#: broken import.
tiled = skewed().example(u=initial())


def example_inputs() -> dict:
    """The inputs ``loopty run`` gives the kernel when no schedule names any."""
    return {"u": initial()}


def main() -> int:
    """Run the demo: the rejection with its witness, then the skewed run."""
    from loopty.executor import LoopyExecutor

    u = initial()
    jacobi(u)
    middle = slice(NX // 2 - 3, NX // 2 + 4)
    print(f"native, the {NT} by {NX} array (first six levels around the spike):")
    print(np.array2string(u.numpy()[:6, middle], precision=3, suppress_small=True))

    message, witness = rejected_tiling()
    (source_id, source), (sink_id, sink), params = witness
    print()
    print("Schedule(jacobi).tile('t', 'i', 8, 8) ->")
    print(f"  IllegalCast: {message}")
    print(f"  witness: {source_id}{source} runs before {sink_id}{sink} at {params}")

    schedule = skewed()
    print()
    print(f"accepted: {schedule!r}")
    print(f"  loop nest: {' '.join(schedule.order)}")
    for fact in schedule.facts():
        print(f"  {fact.status.value:8} {fact.decided_by or '-':4} {fact.statement}")

    print()
    fact = LoopyExecutor().differential(jacobi, schedule, {"u": initial()})
    for name, detail in fact.provenance["outputs"].items():
        print(
            f"  {name}: difference {detail['difference']:.3g} within "
            f"{detail['tolerance']:.3g} ({detail['exactness']}) -> "
            f"{fact.status.value}"
        )
    agrees = np.allclose(u.numpy(), reference(spike()))
    print(f"  the native run matches the hand-written sweep: {agrees}")
    return 0 if agrees and fact.status.value == "tested" else 1


if __name__ == "__main__":
    raise SystemExit(main())
