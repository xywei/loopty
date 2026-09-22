"""Two coupled wave-equation instructions, and the wavefront tile they require.

This is deliberately a stronger stencil example than :mod:`stencil_skew`.
A first-order acoustic update advances two fields in one space/time nest::

    v[t + 1, i] = v[t, i] + c * (p[t, i + 1] - p[t, i])
    p[t + 1, i] = p[t, i] + c * (v[t + 1, i] - v[t + 1, i - 1])

The second instruction consumes values produced by the first instruction in the
*same* time step.  The first instruction, in turn, consumes pressure written by
the second instruction in the *previous* time step.  One of those cross-time
dependences goes from `(t, i + 1)` to `(t + 1, i)`: its distance is
`(1, -1)`.  A rectangular `(t, i)` tile therefore runs a consumer before its
producer at an i-tile boundary and loopty must reject it.

Skewing `i` by `t` changes that distance to `(1, 0)`; rectangular tiles in
the skewed coordinates are then legal wavefront/parallelogram temporal blocks.
That is the useful building block behind diamond blocking, but it is not yet the
full diamond transform.  A true 1-D diamond uses both characteristic coordinates
`i + t` and `i - t`.  Expressing that cleanly is intentional application
pressure for a future multi-axis affine schedule primitive instead of calling a
single skew a diamond.

The optional benchmark is a measurement, not a test: generated-code quality,
problem size and cache hierarchy decide whether temporal blocking wins on a
particular machine::

    uv run python examples/wavefront_acoustic.py --bench
"""

from __future__ import annotations

import sys
from time import perf_counter

import numpy as np
from lanky.prelude import Real

from loopty import Arr, Fin, Schedule, kernel, when

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
    """The same recurrence written without loopty."""
    p = pressure.copy()
    v = velocity.copy()
    nt, nx = p.shape
    for t in range(nt - 1):
        for i in range(1, nx - 1):
            v[t + 1, i] = v[t, i] + courant * (p[t, i + 1] - p[t, i])
            p[t + 1, i] = p[t, i] + courant * (
                v[t + 1, i] - v[t + 1, i - 1]
            )
    return p, v


def rejected_tiling(
    nt: int = NT, nx: int = NX
) -> tuple[str, tuple]:
    """Return the cross-statement witness for a rectangular time/space tile."""
    from loopty.schedule import IllegalCast

    schedule = Schedule(acoustic, target="c", sizes={"nt": nt, "nx": nx})
    try:
        schedule.tile("t", "i", 4, 8)
    except IllegalCast as refused:
        return str(refused), refused.witness
    raise AssertionError("the unskewed time/space tiling should be illegal")


def wavefront_schedule(nt: int = NT, nx: int = NX) -> Schedule:
    """Skew the negative-distance dependence forward, then tile."""
    return (
        Schedule(acoustic, target="c", sizes={"nt": nt, "nx": nx})
        .skew("i", by="t")
        .tile("t", "i", 4, 8)
    )


blocked = wavefront_schedule().example(**initial())


def example_inputs() -> dict:
    """Inputs used by `loopty run`."""
    return initial()


def _arrays(data: dict) -> tuple[np.ndarray, np.ndarray]:
    return data["pressure"].numpy(), data["velocity"].numpy()


def benchmark(
    nt: int = 128,
    nx: int = 8192,
    repeats: int = 3,
) -> int:
    """Compare untiled and wavefront-blocked generated C after one warm-up."""
    from loopty.executor import LoopyExecutor

    executor = LoopyExecutor()
    plain = Schedule(acoustic, target="c", sizes={"nt": nt, "nx": nx})
    blocked_schedule = (
        Schedule(acoustic, target="c", sizes={"nt": nt, "nx": nx})
        .skew("i", by="t")
        .tile("t", "i", 8, 128)
    )

    # Compile and warm both variants before timing.  Fresh data keeps each run
    # semantically identical; allocation is outside the timed region.
    executor.run(plain, **initial(nt, nx))
    executor.run(blocked_schedule, **initial(nt, nx))

    def measured(schedule: Schedule) -> float:
        samples = []
        for _ in range(repeats):
            data = initial(nt, nx)
            start = perf_counter()
            executor.run(schedule, **data)
            samples.append(perf_counter() - start)
        return min(samples)

    plain_time = measured(plain)
    blocked_time = measured(blocked_schedule)
    print(f"problem: {nt} time levels x {nx} points")
    print(f"plain:     {plain_time:.6f} s")
    print(f"wavefront: {blocked_time:.6f} s")
    print(f"plain / wavefront: {plain_time / blocked_time:.3f}x")
    print(
        "This ratio is evidence for this machine/problem only; it is not a "
        "correctness property or a promised speedup."
    )
    return 0


def main() -> int:
    """Show the cross-instruction dependence witness and the legal schedule."""
    data = initial()
    pressure0 = data["pressure"].numpy().copy()
    velocity0 = data["velocity"].numpy().copy()
    acoustic(**data)
    expected_p, expected_v = reference(pressure0, velocity0)
    got_p, got_v = _arrays(data)

    print(f"traced instructions: {[stmt.id for stmt in acoustic.term.stmts]}")
    message, witness = rejected_tiling()
    (source_id, source), (sink_id, sink), params = witness
    print("rectangular tile ->")
    print(f"  IllegalCast: {message}")
    print(
        f"  witness: {source_id}{source} -> {sink_id}{sink} "
        f"at nt={params['nt']}, nx={params['nx']}"
    )

    schedule = wavefront_schedule()
    print()
    print(f"accepted: {schedule!r}")
    print(f"  loop nest: {' '.join(schedule.order)}")
    print(
        "  geometric meaning: rectangular in (t, i+t), "
        "a wavefront/parallelogram block in (t, i)"
    )
    print(
        "  next primitive for a true diamond: jointly map "
        "(t, i) -> (i+t, i-t), preserving the parity-constrained image"
    )

    from loopty.executor import LoopyExecutor

    fact = LoopyExecutor().differential(acoustic, schedule, initial())
    for name, detail in fact.provenance["outputs"].items():
        print(
            f"  {name}: difference {detail['difference']:.3g} within "
            f"{detail['tolerance']:.3g} ({detail['exactness']}) -> "
            f"{fact.status.value}"
        )

    native_ok = np.allclose(got_p, expected_p) and np.allclose(got_v, expected_v)
    print(f"  native run matches hand-written recurrence: {native_ok}")
    return 0 if native_ok and fact.status.value == "tested" else 1


if __name__ == "__main__":
    raise SystemExit(benchmark() if "--bench" in sys.argv else main())
