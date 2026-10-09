"""Two coupled statements, a witness that crosses them, and the skew that fixes it.

A first-order acoustic update advances two fields in one space/time nest::

    v[t + 1, i] = v[t, i] + c * (p[t, i + 1] - p[t, i])
    p[t + 1, i] = p[t, i] + c * (v[t + 1, i] - v[t + 1, i - 1])

Where :mod:`stencil_skew` has one statement, this has two, and they feed each
other. The pressure update ``S1`` reads the velocity that ``S0`` wrote in the
same time step; the velocity update ``S0`` reads the pressure that ``S1`` wrote
in the step before. One of those cross-time dependences goes from ``S1`` at
``(t, i + 1)`` to ``S0`` at ``(t + 1, i)``: its distance is ``(1, -1)``, and it
is the only dependence with a negative space component. A rectangular ``(t, i)``
tile therefore runs a consumer before its producer at every boundary between
space tiles, and loopty refuses it; unlike the stencil's, the witness names two
different statements.

Skewing ``i`` by ``t`` changes that distance to ``(1, 0)`` and leaves every other
one non-negative, and the same tiling is then legal. The tiles are rectangles in
``(t, i + t)``, which are parallelograms in ``(t, i)``: a wavefront temporal
block. That is the building block behind diamond tiling, but it is not a
diamond.

A diamond tiles along ``t + i`` and ``t - i`` at once, and the map
``(t, i) -> (t + i, t - i)`` is not unimodular: its image is only the points
whose two coordinates have the same parity. No skew expresses it, and
``Schedule.affine`` takes it as it is. Four things happen, and the demo prints
all four:

* In the order ``(i + t, i - t)``, space first, the map is refused: the
  dependence of ``S0`` at ``(t + 1, i - 1)`` on ``S1`` at ``(t, i)`` keeps the
  first coordinate and lowers the second by two, so the new order runs it
  backwards, and the witness names those two instances.
* In the order ``(t + i, t - i)`` it is accepted, and loopy generates correct
  code over the image with its holes. loopy's own ``map_domain`` refuses this
  map (``t`` is ``(a + b) / 2``, which it cannot solve for), so loopty rewrites
  the kernel itself; see ``docs/loopy-notes.md``, note 13. Both fields come out
  of the compiled run bit for bit as they come out of the native one, and the
  loop over ``b`` counts its steps, ``b = 2*b_step - a``, rather than visiting
  every ``b`` and testing the parity, which is what loopy alone generates.
* Tiling that diamond, rectangles in ``(t + i, t - i)``, is refused: ``S1`` at
  ``(t, i + 1)`` reads the velocity ``S0`` wrote at ``(t, i)``, a distance of
  ``(0, 1)`` that the ``t - i`` direction runs backwards.
* A diamond tiling of this pair needs an offset between its two statements,
  and ``affine`` takes a map per statement: :data:`OFFSET_DIAMOND` puts ``S1``
  at ``(t, i)`` on ``(t + i, t - i + 1)``, half a step later in time and half a
  cell earlier in space than ``S0`` at the same ``(t, i)``, which is where a
  staggered scheme keeps its pressure. Every dependence of the pair is then
  ``(0, 1)``, ``(1, 0)`` or ``(1, 1)`` in ``(a, b)``, so the same four by four
  tiling is accepted, and it agrees with the native run bit for bit. The two
  statements share the loops over ``a`` and ``b``: ``S0`` owns the points where
  ``a + b`` is even and ``S1`` those where it is odd, so every point is one
  statement's.

Run this file three ways.

``python examples/wavefront_acoustic.py``
    Runs the kernel natively, prints the rejection and its witness, builds the
    skewed schedule, runs it on the C target and compares both fields; then
    does the same for the diamond, with its two refusals, and for the diamond
    tiling with its offset.

``lanky check examples/wavefront_acoustic.py``
    Prints the ledger of the kernel's own obligations: all eight accesses in
    bounds, the writes of each statement disjoint, the source order monotone on
    the dependences.

``loopty run examples/wavefront_acoustic.py``
    Compiles the skewed and tiled schedule, the diamond, and the diamond
    tiling, and compares each with the native run; each keeps its own facts in
    the ledger, since a fact's id names the schedule it is about, and not only
    the kernel.

A fourth, ``python examples/wavefront_acoustic.py --bench``, times the untiled
and the wavefront-blocked kernels at a larger size. It is a measurement, not a
test: generated-code quality, problem size and cache hierarchy decide whether
temporal blocking wins on a particular machine.
"""

from __future__ import annotations

import sys
from time import perf_counter

import numpy as np
from lanky.prelude import Real

from loopty import Arr, Fin, Schedule, kernel, when

#: Sixteen time levels over thirty-two points, tiled four by eight: enough tiles
#: that the skewed nest has boundaries in both directions, small enough that the
#: demo compiles and runs in about a second. The checks are symbolic in ``nt``
#: and ``nx``; these are the sizes the run uses and the printed witness is read
#: off at.
NT, NX = 16, 32
COURANT = 0.25


@kernel
def acoustic(
    pressure: Arr[Fin[nt], Fin[nx], Real],  # noqa: F821
    velocity: Arr[Fin[nt], Fin[nx], Real],  # noqa: F821
    courant: Real,
):
    """Advance a 1-D acoustic system with a staggered pair of differences."""
    steps = pressure.dom
    for t in steps:
        row = pressure.dom[t]
        for i in row:
            with when((t + 1 < steps.size) & (i > 0) & (i + 1 < row.size)):
                velocity[t + 1, i] = velocity[t, i] + courant * (
                    pressure[t, i + 1] - pressure[t, i]
                )
                pressure[t + 1, i] = pressure[t, i] + courant * (
                    velocity[t + 1, i] - velocity[t + 1, i - 1]
                )


def initial(nt: int = NT, nx: int = NX) -> dict:
    """A pressure impulse and zero velocity, with storage for every time level."""
    pressure = np.zeros((nt, nx))
    pressure[0, nx // 2] = 1.0
    velocity = np.zeros((nt, nx))
    return {
        "pressure": Arr.from_numpy(pressure),
        "velocity": Arr.from_numpy(velocity),
        "courant": COURANT,
    }


def reference(
    pressure: np.ndarray,
    velocity: np.ndarray,
    courant: float = COURANT,
) -> tuple[np.ndarray, np.ndarray]:
    """The same recurrence written out by hand, as the thing to agree with."""
    p = pressure.copy()
    v = velocity.copy()
    nt, nx = p.shape
    for t in range(nt - 1):
        for i in range(1, nx - 1):
            v[t + 1, i] = v[t, i] + courant * (p[t, i + 1] - p[t, i])
            p[t + 1, i] = p[t, i] + courant * (v[t + 1, i] - v[t + 1, i - 1])
    return p, v


def rejected_tiling(nt: int = NT, nx: int = NX) -> tuple[str, tuple]:
    """Ask for the rectangular tiling and return the message and the witness.

    As in :mod:`stencil_skew`, the schedule the cast was asked of is untouched
    by the rejection.
    """
    from loopty.schedule import IllegalCast

    schedule = Schedule(acoustic, target="c", sizes={"nt": nt, "nx": nx})
    try:
        schedule.tile("t", "i", 4, 8)
    except IllegalCast as refused:
        return str(refused), refused.witness
    raise AssertionError("the unskewed time/space tiling should be illegal")


def wavefront_schedule(nt: int = NT, nx: int = NX) -> Schedule:
    """The legal schedule: skew the space axis by time, then tile."""
    return (
        Schedule(acoustic, target="c", sizes={"nt": nt, "nx": nx})
        .skew("i", by="t")
        .tile("t", "i", 4, 8)
    )


#: The diamond coordinates, time first: ``a = t + i`` and ``b = t - i``. The
#: map has determinant -2, so its image is the points where ``a`` and ``b``
#: have the same parity.
DIAMOND = "{ [t, i] -> [a, b] : a = t + i and b = t - i }"

#: The same diamond with space first, ``a = i + t`` and ``b = i - t``.
DIAMOND_SPACE_FIRST = "{ [t, i] -> [a, b] : a = i + t and b = i - t }"


def rejected_diamond(nt: int = NT, nx: int = NX) -> tuple[str, tuple]:
    """Ask for the space-first diamond and return the message and the witness."""
    from loopty.schedule import IllegalCast

    schedule = Schedule(acoustic, target="c", sizes={"nt": nt, "nx": nx})
    try:
        schedule.affine(DIAMOND_SPACE_FIRST)
    except IllegalCast as refused:
        return str(refused), refused.witness
    raise AssertionError("the space-first diamond should be illegal")


def diamond_schedule(nt: int = NT, nx: int = NX) -> Schedule:
    """The loop nest in diamond coordinates, time first."""
    return Schedule(acoustic, target="c", sizes={"nt": nt, "nx": nx}).affine(
        DIAMOND
    )


def rejected_diamond_tiling(nt: int = NT, nx: int = NX) -> tuple[str, tuple]:
    """Ask to tile the diamond and return the message and the witness."""
    from loopty.schedule import IllegalCast

    try:
        diamond_schedule(nt, nx).tile("a", "b", 4, 4)
    except IllegalCast as refused:
        return str(refused), refused.witness
    raise AssertionError("tiling the diamond should be illegal for this pair")


#: The diamond with a map per statement: ``S0`` as in :data:`DIAMOND`, and
#: ``S1`` one further along ``b``, at ``a + b = 2t + 1``, half a time step after
#: the velocity update of the same ``(t, i)``. Each image is half the points of
#: the plane and together they are all of it, so the statements share their
#: loops and each point runs one of them.
OFFSET_DIAMOND = (
    "{ S0[t, i] -> [a, b] : a = t + i and b = t - i; "
    "S1[t, i] -> [a, b] : a = t + i and b = t - i + 1 }"
)


def offset_diamond_schedule(nt: int = NT, nx: int = NX) -> Schedule:
    """The diamond tiling: the offset diamond, then rectangles in ``(a, b)``."""
    return (
        Schedule(acoustic, target="c", sizes={"nt": nt, "nx": nx})
        .affine(OFFSET_DIAMOND)
        .tile("a", "b", 4, 4)
    )


#: What ``loopty run`` compiles and compares: the wavefront block, the
#: diamond, and the diamond tiling. The rejected casts live in
#: :func:`rejected_tiling`, :func:`rejected_diamond` and
#: :func:`rejected_diamond_tiling`, where the exception is the answer.
blocked = wavefront_schedule().example(**initial())
diamond = diamond_schedule().example(**initial())
diamond_tiled = offset_diamond_schedule().example(**initial())


def example_inputs() -> dict:
    """The inputs ``loopty run`` gives the kernel when no schedule names any."""
    return initial()


def benchmark(nt: int = 128, nx: int = 8192, repeats: int = 5) -> int:
    """Time the untiled and the wavefront-blocked compiled kernels.

    Only the compiled call is timed. :meth:`~loopty.executor.LoopyExecutor.run`
    also builds loopy's executor, finds the compiled code in loopy's cache and
    checks the arguments against the term, and at a size like this one that can
    cost as much as the kernel does, which would make the ratio a measurement of
    the overhead. So each schedule's executor is built once and warmed up, and
    fresh arrays are made outside the timed region. loopy is called directly
    here, so inside ``declining()``, where loopty's own runs call it: the
    kernel is written for how loopty has loopy read its subscripts and guards
    (:mod:`loopty.isl_reading`).
    """
    from loopty.isl_reading import declining

    sizes = {"nt": nt, "nx": nx}
    variants = {
        "plain": Schedule(acoustic, target="c", sizes=sizes),
        "wavefront": Schedule(acoustic, target="c", sizes=sizes)
        .skew("i", by="t")
        .tile("t", "i", 8, 128),
    }

    def arguments() -> dict:
        data = initial(nt, nx)
        return {
            "pressure": data["pressure"].numpy(),
            "velocity": data["velocity"].numpy(),
            "courant": data["courant"],
        }

    best = {}
    for label, schedule in variants.items():
        samples = []
        with declining():
            call = schedule.kernel.executor()
            call(**arguments())  # compile, or load from loopy's cache
            for _ in range(repeats):
                data = arguments()
                start = perf_counter()
                call(**data)
                samples.append(perf_counter() - start)
        best[label] = min(samples)

    print(f"problem: {nt} time levels by {nx} points, best of {repeats}")
    print(f"plain:     {best['plain']:.6f} s")
    print(f"wavefront: {best['wavefront']:.6f} s")
    print(f"plain / wavefront: {best['plain'] / best['wavefront']:.3f}")
    print("A measurement of this machine at this size, not a property of either")
    print("schedule and not a promised speedup.")
    return 0


def main() -> int:
    """Run the demo: each rejection with its witness, and each accepted run."""
    from loopty.executor import LoopyExecutor

    data = initial()
    pressure0 = data["pressure"].numpy().copy()
    velocity0 = data["velocity"].numpy().copy()
    acoustic(**data)
    middle = slice(NX // 2 - 3, NX // 2 + 4)
    print(
        f"native pressure, {NT} levels by {NX} points "
        "(first five levels around the impulse):"
    )
    pressure = data["pressure"].numpy()
    print(np.array2string(pressure[:5, middle], precision=3, suppress_small=True))
    writes = ", ".join(
        f"{stmt.id} writes {stmt.assignee.array}" for stmt in acoustic.term.stmts
    )
    print(f"statements: {writes}")

    message, witness = rejected_tiling()
    (source_id, source), (sink_id, sink), params = witness
    print()
    print("Schedule(acoustic).tile('t', 'i', 4, 8) ->")
    print(f"  IllegalCast: {message}")
    print(f"  witness: {source_id}{source} runs before {sink_id}{sink} at {params}")

    schedule = wavefront_schedule()
    print()
    print(f"accepted: {schedule!r}")
    print(f"  loop nest: {' '.join(schedule.order)}")
    for fact in schedule.facts():
        print(f"  {fact.status.value:8} {fact.decided_by or '-':4} {fact.statement}")

    print()
    fact = LoopyExecutor().differential(acoustic, schedule, initial())
    for name, detail in fact.provenance["outputs"].items():
        print(
            f"  {name}: difference {detail['difference']:.3g} within "
            f"{detail['tolerance']:.3g} ({detail['exactness']}) -> "
            f"{fact.status.value}"
        )
    want_pressure, want_velocity = reference(pressure0, velocity0)
    agrees = np.allclose(pressure, want_pressure) and np.allclose(
        data["velocity"].numpy(), want_velocity
    )
    print(f"  the native run matches the hand-written recurrence: {agrees}")

    message, _witness = rejected_diamond()
    print()
    print(f"Schedule(acoustic).affine({DIAMOND_SPACE_FIRST!r}) ->")
    print(f"  IllegalCast: {message}")

    schedule = diamond_schedule()
    (domain,) = schedule.kernel.default_entrypoint.domains
    print()
    print(f"accepted: {schedule!r}")
    print(f"  loop nest: {' '.join(schedule.order)}")
    print(f"  loopy domain: {domain}")
    for step in schedule.facts():
        print(f"  {step.status.value:8} {step.decided_by or '-':4} {step.statement}")
    print()
    diamond_fact = LoopyExecutor().differential(acoustic, schedule, initial())
    for name, detail in diamond_fact.provenance["outputs"].items():
        print(
            f"  {name}: difference {detail['difference']:.3g} within "
            f"{detail['tolerance']:.3g} ({detail['exactness']}) -> "
            f"{diamond_fact.status.value}"
        )

    message, _witness = rejected_diamond_tiling()
    print()
    print(f"{schedule!r}.tile('a', 'b', 4, 4) ->")
    print(f"  IllegalCast: {message}")

    schedule = offset_diamond_schedule()
    print()
    print(f"accepted: {schedule!r}")
    print(f"  loop nest: {' '.join(schedule.order)}")
    for step in schedule.facts():
        print(f"  {step.status.value:8} {step.decided_by or '-':4} {step.statement}")
    print()
    tiled_fact = LoopyExecutor().differential(acoustic, schedule, initial())
    for name, detail in tiled_fact.provenance["outputs"].items():
        print(
            f"  {name}: difference {detail['difference']:.3g} within "
            f"{detail['tolerance']:.3g} ({detail['exactness']}) -> "
            f"{tiled_fact.status.value}"
        )
    statuses = {fact.status.value, diamond_fact.status.value, tiled_fact.status.value}
    return 0 if agrees and statuses == {"tested"} else 1


if __name__ == "__main__":
    raise SystemExit(benchmark() if "--bench" in sys.argv else main())
