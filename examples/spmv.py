"""Sparse matrix-vector product: the design's main demo.

Run this file three ways and it answers three different questions.

``python examples/spmv.py``
    The kernels run natively on numpy and the theorem runs as a property test.
    Nothing but lanky, loopty and numpy is loaded; loopy is not involved.

``lanky check examples/spmv.py``
    Every obligation the typing rules state becomes a fact, the oracles are
    tried strongest first, and the ledger says who decided what.

``loopty run examples/spmv.py``
    The kernels are lowered through loopy, compiled for the C target, run on
    the example inputs, and compared with the native run at the tolerance the
    exactness class states.

What is worth reading here
--------------------------

*The ragged shape is a type.* ``val: Arr[Fin[n], Fin[cnt], Real]`` says: for
each of the ``n`` rows, ``cnt[r]`` entries. The second axis names the counts
array, which is what makes it a dependent sum rather than a rectangle, and the
inner loop iterates ``val.dom[r]``, the fiber over that row. Nothing in the
kernel mentions the flat storage or the offsets; those are the layout, and the
layout is the compiler's business.

*An indirection is in bounds by type.* ``col: Arr[Fin[n], Fin[cnt], Fin[m]]``
says the entries of ``col`` are points of ``Fin[m]``, and ``x`` has ``m`` cells,
so ``x[col[r, j]]`` needs no proof: the typing rule discharges it with
``decided_by="type"`` and never calls isl. That is the one obligation in a
sparse product that a polyhedral checker cannot decide on its own, and here it
is decided by the shape of the data.

*The offsets come with a claim.* ``scan`` writes them, and its return
annotation is not a result type (outputs are parameters) but a postcondition:
``off[0] == 0`` and the recurrence. ``scan_monotone`` is the theorem that turns
that recurrence into the monotonicity a flat CSR layout needs. lanky decides it
with the property tester, or proves it with Lean when the Lean extra is
installed.

*The schedule is a cast.* Each step is checked by isl before it is applied:
the reindexing must be a bijection on statement instances and the new order
must run every dependence forward in time. Splitting the row's entries and
summing the pieces separately is a reassociation, so ``realize("y", tree=True)``
lowers the accumulation to ``reassoc`` and the differential comparison widens
its tolerance accordingly.

*Legal is not buildable.* The device schedule the design note names,
:func:`device_schedule`, passes every cast and is still refused by loopy's code
generator, on a device as much as on C: a hardware axis cannot sit inside a
loop whose bound comes from an array, and a CSR row is such a loop. That was
measured (``docs/device-runs.md``), and the schedule reports it as a ``refuted``
fact of kind ``buildable``. :func:`rows_parallel` is the schedule for this shape
that does build, and did run on two device classes.
"""

from __future__ import annotations

import numpy as np
from lanky import theorem
from lanky.prelude import Fn, Nat, Real

from loopty import Arr, Fin, Schedule, kernel, program
from loopty import reduce_sum

# {{{ the kernels and the theorem


@kernel
def scan(
    cnt: Arr[Fin[n], Nat], off: Arr[Fin[n + 1], Nat]
) -> (off[0] == 0) & all(off[r + 1] == off[r] + cnt[r] for r in Fin[n]):
    """Exclusive prefix sum of the counts: where each row starts in storage.

    The loop is sequential and nobody said so: ``off[r + 1]`` reads ``off[r]``,
    so the dependence is in the footprints and any schedule that reordered the
    rows would be rejected by the checker rather than by a review comment.
    """
    off[0] = 0
    for r in cnt.dom:
        off[r + 1] = off[r] + cnt[r]


@theorem
def scan_monotone(
    n: Nat,
    cnt: Fn[Fin[n], Nat],
    off: Fn[Fin[n + 1], Nat],
    h0: off(0) == 0,
    hs: all(off(r + 1) == off(r) + cnt(r) for r in Fin[n]),
) -> all(off(a) <= off(b) for a in Fin[n + 1] for b in Fin[n + 1] if a <= b):
    """The offsets an exclusive scan produces are monotone.

    The hypotheses are exactly ``scan``'s postcondition, written over families
    rather than arrays because a theorem talks about the data and not about the
    storage. Counts are naturals, so nothing decreases; the interesting part is
    that this is the fact a flat CSR layout needs in order to be in bounds, and
    lanky establishes it once for every kernel that lays data out this way.
    """


@kernel
def spmv(
    cnt: Arr[Fin[n], Nat],
    col: Arr[Fin[n], Fin[cnt], Fin[m]],
    val: Arr[Fin[n], Fin[cnt], Real],
    x: Arr[Fin[m], Real],
    y: Arr[Fin[n], Real],
):
    """One row of ``y`` per row of the matrix, summed over that row's entries."""
    for r in y.dom:
        y[r] = reduce_sum(val[r, j] * x[col[r, j]] for j in val.dom[r])


@program
def solve(cnt, col, val, x, y, off):
    """Lay the rows out, then multiply: the composition the ledger records."""
    scan(cnt, off)
    spmv(cnt, col, val, x, y)


# }}}


# {{{ example data


def random_csr(rows: int = 6, cols: int = 5, seed: int = 0) -> dict:
    """A small random CSR matrix and a vector, as loopty runtime arrays.

    Tiny on purpose: every command in this file compiles and runs in about a
    second, and a sparse product is no more convincing at size ten thousand.
    """
    rng = np.random.default_rng(seed)
    counts = [int(k) for k in rng.integers(0, 4, size=rows)]
    columns: list[int] = []
    values: list[float] = []
    for count in counts:
        # Distinct columns per row, which is what a real matrix has; the kernel
        # never relies on it, because the product sums whatever is stored.
        picked = rng.choice(cols, size=count, replace=False)
        columns.extend(int(c) for c in sorted(picked))
        values.extend(float(v) for v in rng.normal(size=count))
    return {
        "cnt": Arr.from_numpy(np.array(counts, dtype=np.int64)),
        "col": Arr.ragged(counts, values=columns, dtype=np.int64),
        "val": Arr.ragged(counts, values=values),
        "x": Arr.from_numpy(rng.normal(size=cols)),
        "y": Arr.zeros(rows),
        "off": Arr.zeros(rows + 1, dtype=np.int64),
    }


def example_inputs() -> dict:
    """The inputs ``loopty run`` gives each kernel in this file."""
    data = random_csr()
    return {
        "scan": {"cnt": data["cnt"], "off": data["off"]},
        "spmv": {key: data[key] for key in ("cnt", "col", "val", "x", "y")},
    }


def dense(data: dict) -> np.ndarray:
    """The same matrix written out densely, to check the product by hand."""
    counts = data["cnt"].numpy()
    offsets = data["val"].offsets
    columns, values = data["col"].numpy(), data["val"].numpy()
    matrix = np.zeros((len(counts), len(data["x"].numpy())))
    for row in range(len(counts)):
        for a in range(offsets[row], offsets[row + 1]):
            matrix[row, columns[a]] += values[a]
    return matrix


# }}}


# {{{ schedules

#: The schedule ``loopty run`` exercises here. Splitting the entries of a row
#: and summing the pieces is a reassociation of a floating-point accumulation,
#: which is why ``realize`` has to be asked and why the fact it emits widens the
#: tolerance of the differential comparison.
rows_split = (
    Schedule(spmv, target="c")
    .split("j", 2, inner="j_in", outer="j_out")
    .realize("y", tree=True)
    .example(**example_inputs()["spmv"])
)


def device_schedule() -> Schedule:
    """The design note's schedule: rows across groups, entries across lanes.

    Legal, and not buildable, which is the interesting part. Every cast is
    ``DECIDED``: nothing is reordered that carries a dependence, and marking the
    accumulation ``reassoc`` is exactly the permission the split of ``j`` needs.
    What fails is code generation, and it fails on a device too, which was
    measured rather than assumed (see ``docs/device-runs.md``): loopy 2025.2
    will not put a hardware axis inside a loop whose bound comes from an array,
    and a CSR row is precisely such a loop. Applying loopy's own remedy,
    ``split_reduction_outward``, runs into the same wall from the other side.

    So ``.buildable`` is ``(False, reason)`` and the schedule carries a
    ``REFUTED`` fact of kind ``buildable`` decided by ``loopy-target`` beside
    its decided casts. Asking it to run raises
    :class:`~loopty.schedule.UnbuildableSchedule` naming the limit, rather than
    throwing from inside loopy several steps later.

    :func:`rows_parallel` is what does work on a device for this shape.
    """
    return (
        Schedule(spmv, target="c")
        .tag(r="g.0")
        .split("j", 32, inner="j_in", outer="j_out")
        .tag(j_in="l.0")
        .realize("y", tree=True)
    )


def rows_parallel(target: str = "opencl") -> Schedule:
    """One row per work group: the ragged product as a device actually runs it.

    ``r`` is bounded by a size known when the kernel is launched, so it may
    carry a hardware axis; the entries of a row stay a sequential reduction.
    This ran on both device classes and agreed with numpy; it is behind a
    function because building it for ``"opencl"`` at import time would make
    ``python examples/spmv.py`` need a device.
    """
    return Schedule(spmv, target=target).tag(r="g.0")


# }}}


def main() -> int:
    """Run the demo: natively, then compiled, and print what agreed."""
    from loopty.executor import LoopyExecutor

    data = random_csr()
    solve(data["cnt"], data["col"], data["val"], data["x"], data["y"], data["off"])
    print("counts  =", data["cnt"].numpy())
    print("offsets =", data["off"].numpy())
    print("y       =", np.array2string(data["y"].numpy(), precision=4))
    print("dense   =", np.array2string(dense(data) @ data["x"].numpy(), precision=4))

    report = scan_monotone.report(n=50)
    print()
    print(f"scan_monotone: {scan_monotone.statement}")
    print(
        f"  {'ok' if report.ok else 'REFUTED'} after {report.valid} valid draws "
        f"of {report.samples}"
    )

    print()
    print(f"schedule: {rows_split!r}")
    for fact in rows_split.facts():
        print(f"  {fact.status.value:8} {fact.decided_by or '-':4} {fact.statement}")
    device = device_schedule()
    print(f"device schedule: {device!r}")
    for fact in device.facts():
        print(f"  {fact.status.value:8} {fact.decided_by or '-':4} {fact.statement}")
    print(f"  reason: {device.buildable[1]}")

    print()
    # A fresh copy of the same matrix, so that the compiled run and the native
    # run inside ``differential`` both start from a zeroed ``y``.
    fact = LoopyExecutor().differential(spmv, rows_split, example_inputs()["spmv"])
    for name, detail in fact.provenance["outputs"].items():
        print(
            f"  {name}: difference {detail['difference']:.3g} within "
            f"{detail['tolerance']:.3g} ({detail['exactness']}) -> "
            f"{fact.status.value}"
        )
    return 0 if fact.status.value == "tested" else 1


if __name__ == "__main__":
    raise SystemExit(main())
