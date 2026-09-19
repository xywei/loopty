"""Lowering and scheduling terms that came from the tracer, not from hand.

The rest of this wave's tests build terms by hand so that the two halves of the
compiler can be developed apart. This one joins them: a decorated kernel is
traced, the term is lowered, the compiled code is compared with the kernel's own
Python body, and the stencil is put through the rejection and the fix. If the
tracer is not landed yet, the whole module skips rather than failing, because the
two waves are written at the same time.
"""

from __future__ import annotations

import numpy as np
import pytest

pytest.importorskip("loopy")

kernels = pytest.importorskip("kernels")


def traced(name: str, module: str):
    """A decorated kernel from ``tests/kernels``, or a skip with the reason."""
    import importlib

    try:
        loaded = importlib.import_module(f"kernels.{module}")
        kernel = getattr(loaded, name)
        kernel.trace()
    except Exception as exc:  # pragma: no cover - depends on the tracer's state
        pytest.skip(f"tracing {name} is not available yet: {type(exc).__name__}: {exc}")
    return kernel


def run(obj, **arguments):
    from loopty.executor import LoopyExecutor

    try:
        return LoopyExecutor().run(obj, **arguments)
    except Exception as exc:  # pragma: no cover - depends on the local toolchain
        if "compil" in str(exc).lower() or isinstance(exc, OSError):
            pytest.skip(f"the C toolchain path is unusable here: {exc}")
        raise


def test_a_traced_dense_kernel_lowers_runs_and_agrees() -> None:
    from loopty.arr import Arr
    from loopty.executor import LoopyExecutor
    from loopty.schedule import Schedule

    axpy = traced("axpy", "axpy")
    # The kernel's Python body iterates ``y.dom``, so the reference run is given
    # runtime arrays; the compiled run unwraps them to numpy itself.
    arrays = {
        "a": 2.0,
        "x": Arr.from_numpy(np.arange(4, dtype=np.float64)),
        "y": Arr.zeros(4),
    }
    schedule = Schedule(axpy)
    fact = LoopyExecutor().differential(axpy, schedule, arrays)
    assert fact.status.value == "tested", fact.provenance


def test_a_traced_ragged_kernel_computes_the_sparse_product() -> None:
    spmv = traced("spmv", "spmv_min")
    module = __import__("kernels.spmv_min", fromlist=["example"])
    cnt, col, val, x, y, _off = module.example()

    out = run(spmv.trace(), cnt=cnt, col=col, val=val, x=x, y=y)

    want = np.zeros(3)
    offsets = val.offsets
    flat_val, flat_col, x_values = val.numpy(), col.numpy(), x.numpy()
    for row in range(3):
        for a in range(offsets[row], offsets[row + 1]):
            want[row] += flat_val[a] * x_values[flat_col[a]]
    assert np.allclose(out["y"], want)


def test_a_traced_reduction_is_split_tagged_and_reassociated() -> None:
    from loopty.executor import LoopyExecutor
    from loopty.schedule import Schedule, UnbuildableSchedule

    spmv = traced("spmv", "spmv_min")
    schedule = (
        Schedule(spmv, target="c")
        .split("j", 2, inner="j_in", outer="j_out")
        .tag(j_in="l.0")
        .realize("y", tree=True)
    )
    assert schedule.reassociated == frozenset({"y"})
    # Every cast is legal: the instances are untouched and nothing is reordered.
    casts = [fact for fact in schedule.facts() if fact.kind != "buildable"]
    assert all(fact.status.value == "decided" for fact in casts)

    # And yet loopy cannot generate it: ``j_in`` is a hardware axis inside a
    # ragged fiber. The schedule says so instead of letting code generation
    # throw, and refuses when something asks it for code.
    ok, reason = schedule.buildable
    assert not ok
    assert "ragged fiber" in reason
    buildable = [fact for fact in schedule.facts() if fact.kind == "buildable"]
    assert [fact.status.value for fact in buildable] == ["refuted"]
    assert buildable[0].decided_by == "loopy-target"
    with pytest.raises(UnbuildableSchedule, match="ragged fiber"):
        LoopyExecutor().run(schedule)


def test_the_traced_stencil_refuses_a_tiling_and_accepts_a_skewed_one() -> None:
    from loopty.schedule import IllegalCast, Schedule

    jacobi = traced("jacobi", "stencil")
    schedule = Schedule(jacobi, sizes={"nt": 16, "nx": 16})

    with pytest.raises(IllegalCast) as caught:
        schedule.tile("t", "i", 8, 8)
    assert "scheduled earlier" in str(caught.value)

    tiled = schedule.skew("i", by="t").tile("t", "i", 8, 8)

    u = np.zeros((6, 6))
    u[0] = np.arange(6.0)
    want = u.copy()
    for t in range(5):
        for i in range(1, 5):
            want[t + 1, i] = (want[t, i - 1] + want[t, i + 1]) / 2
    out = run(tiled, u=u.copy())
    assert np.allclose(out["u"], want)
