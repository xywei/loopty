"""Run loopty's demos on an OpenCL device.

Usage::

    PYOPENCL_CTX=<platform>:<device> python device_demos.py

Select the device explicitly; ``PYOPENCL_CTX`` unset makes pyopencl choose, and
a run whose device nobody wrote down is not evidence of anything.

The demo files pin their schedules to ``target="c"`` because the C target is what
the local test suite can compile. This driver rebuilds the same schedules for
``target="opencl"`` and runs them on the device selected by ``PYOPENCL_CTX``, so
that the agreement between the compiled device run and the numpy reference is
measured rather than assumed. Each case prints the device, the schedule, the
difference and the tolerance the exactness class states.
"""

from __future__ import annotations

import os
import sys
import traceback

import numpy as np


def _repo_root() -> str:
    """The loopty checkout holding ``examples/``.

    Set ``LOOPTY_REPO`` to point at it, or run this from anywhere inside the
    checkout: the search walks upwards until it finds ``examples/spmv.py``, so
    the driver works both from the repository and from a copy rsynced next to
    it on a compute host.
    """
    named = os.environ.get("LOOPTY_REPO")
    if named:
        return os.path.abspath(named)
    here = os.path.dirname(os.path.abspath(__file__))
    for base in (here, os.getcwd()):
        path = base
        while True:
            if os.path.exists(os.path.join(path, "examples", "spmv.py")):
                return path
            parent = os.path.dirname(path)
            if parent == path:
                break
            path = parent
    raise SystemExit(
        "cannot find a loopty checkout with examples/; set LOOPTY_REPO"
    )


REPO = _repo_root()
sys.path.insert(0, REPO)


def banner(title: str) -> None:
    print()
    print("=" * 72)
    print(title)
    print("=" * 72)


def load(name: str, relpath: str):
    import importlib.util

    spec = importlib.util.spec_from_file_location(name, os.path.join(REPO, relpath))
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


def show_device() -> None:
    import pyopencl as cl

    ctx = cl.create_some_context(interactive=False)
    dev = ctx.devices[0]
    print("PYOPENCL_CTX  =", os.environ.get("PYOPENCL_CTX", "<unset>"))
    print("platform      =", dev.platform.name.strip())
    print("platform ver  =", dev.platform.version.strip())
    print("device        =", dev.name.strip())
    print("device type   =", cl.device_type.to_string(dev.type))
    print("driver        =", dev.driver_version)
    print("fp64          =", "cl_khr_fp64" in dev.extensions)
    print("compute units =", dev.max_compute_units)
    print("pyopencl      =", cl.VERSION_TEXT)
    import islpy
    import loopy

    print("loopy         =", getattr(loopy, "__version__", "?"))
    print("islpy         =", getattr(islpy, "__version__", "?"))


def report(fact) -> None:
    status = fact.status.value if hasattr(fact.status, "value") else str(fact.status)
    for name, detail in fact.provenance.get("outputs", {}).items():
        print(
            f"  {name}: difference {detail['difference']:.3g} within "
            f"{detail['tolerance']:.3g} ({detail['exactness']}) -> {status}"
        )
    print("  agreement fact:", status)


def facts_of(schedule) -> None:
    for fact in schedule.facts():
        status = (
            fact.status.value if hasattr(fact.status, "value") else str(fact.status)
        )
        print(f"  {status:8} {fact.decided_by or '-':4} {fact.statement}")


def case(title, fn) -> bool:
    banner(title)
    try:
        fn()
        return True
    except Exception as exc:  # noqa: BLE001 - the point is to record the failure
        print("FAILED:", type(exc).__name__)
        print(str(exc)[:900])
        print("--- traceback (last frames) ---")
        traceback.print_exc(limit=2)
        return False


def main() -> int:
    show_device()

    from loopty import Arr, IllegalCast
    from loopty.executor import LoopyExecutor
    from loopty.schedule import Schedule

    ex = LoopyExecutor(target="opencl")
    spmv_mod = load("demo_spmv", "examples/spmv.py")
    sten_mod = load("demo_stencil", "examples/stencil_skew.py")
    resh_mod = (
        load("demo_reshape", "examples/reshape_layouts.py")
        if os.path.exists(os.path.join(REPO, "examples/reshape_layouts.py"))
        else None
    )
    ok = {}

    # --- spmv: the split + realize schedule, on the device -----------------
    def spmv_split():
        data = spmv_mod.random_csr()
        inputs = {k: data[k] for k in ("cnt", "col", "val", "x", "y")}
        sched = (
            Schedule(spmv_mod.spmv, target="opencl")
            .split("j", 2, inner="j_in", outer="j_out")
            .realize("y", tree=True)
        )
        print("schedule:", repr(sched))
        facts_of(sched)
        fact = ex.differential(spmv_mod.spmv, sched, inputs)
        report(fact)
        dense = spmv_mod.dense(data) @ data["x"].numpy()
        got = ex.run(sched, **{k: spmv_mod._reset(k, v) for k, v in inputs.items()})
        print("  device y   =", np.array2string(np.asarray(got["y"]), precision=6))
        print("  dense ref  =", np.array2string(dense, precision=6))
        print("  max |diff| =", float(np.max(np.abs(np.asarray(got["y"]) - dense))))

    ok["spmv split+realize"] = case(
        "spmv: split(j,2).realize(y) on the device", spmv_split
    )

    # --- spmv: scan on the device -----------------------------------------
    def scan_device():
        data = spmv_mod.random_csr()
        inputs = {"cnt": data["cnt"], "off": data["off"]}
        sched = Schedule(spmv_mod.scan, target="opencl")
        print("schedule:", repr(sched))
        fact = ex.differential(spmv_mod.scan, sched, inputs)
        report(fact)

    ok["scan"] = case("spmv: scan on the device", scan_device)

    # --- spmv: the design's device schedule ------------------------------
    def spmv_device_schedule():
        data = spmv_mod.random_csr()
        inputs = {k: data[k] for k in ("cnt", "col", "val", "x", "y")}
        sched = (
            Schedule(spmv_mod.spmv, target="opencl")
            .tag(r="g.0")
            .split("j", 32, inner="j_in", outer="j_out")
            .tag(j_in="l.0")
            .realize("y", tree=True)
        )
        print("schedule:", repr(sched))
        facts_of(sched)
        fact = ex.differential(spmv_mod.spmv, sched, inputs)
        report(fact)

    ok["spmv g.0/l.0 (design note)"] = case(
        "spmv: tag(r=g.0).split(j,32).tag(j_in=l.0).realize(y) on the device",
        spmv_device_schedule,
    )

    # --- spmv: rows across groups only ------------------------------------
    def spmv_rows_parallel():
        data = spmv_mod.random_csr()
        inputs = {k: data[k] for k in ("cnt", "col", "val", "x", "y")}
        sched = Schedule(spmv_mod.spmv, target="opencl").tag(r="g.0")
        print("schedule:", repr(sched))
        facts_of(sched)
        fact = ex.differential(spmv_mod.spmv, sched, inputs)
        report(fact)

    ok["spmv rows parallel g.0"] = case(
        "spmv: tag(r=g.0), rows across work groups, on the device", spmv_rows_parallel
    )

    # --- stencil: the illegal tile is still refused on the device ----------
    def stencil_refused():
        try:
            Schedule(
                sten_mod.jacobi,
                target="opencl",
                sizes={"nt": sten_mod.NT, "nx": sten_mod.NX},
            ).tile("t", "i", 8, 8)
        except IllegalCast as exc:
            print("refused as expected:")
            print(" ", exc)
            print("  witness:", exc.witness)
            return
        raise AssertionError("the rectangular tiling was accepted on the device target")

    ok["stencil illegal tile refused"] = case(
        "stencil: tile(t,i,8,8) must be refused (device target)", stencil_refused
    )

    # --- stencil: skew then tile, run on the device ------------------------
    def stencil_skewed():
        sched = (
            Schedule(sten_mod.jacobi, target="opencl")
            .skew("i", by="t")
            .tile("t", "i", 8, 8)
        )
        print("schedule:", repr(sched))
        print("loop nest:", " ".join(sched.order))
        facts_of(sched)
        inputs = sten_mod.example_inputs()
        fact = ex.differential(sten_mod.jacobi, sched, inputs)
        report(fact)
        got = ex.run(sched, **{k: Arr(v.numpy().copy()) if hasattr(v, "numpy") else v
                               for k, v in inputs.items()})
        print("  device u[:3, :7] =")
        print(np.array2string(np.asarray(got["u"])[:3, :7], precision=4))

    ok["stencil skew+tile"] = case(
        "stencil: skew(i, by=t).tile(t,i,8,8) on the device", stencil_skewed
    )

    # --- reshape: the transpose, split and interchanged, on the device -----
    if resh_mod is not None:

        def reshape_transpose():
            sched = (
                Schedule(
                    resh_mod.transpose,
                    target="opencl",
                    sizes={"n": resh_mod.ROWS, "m": resh_mod.COLS},
                )
                .split("i", 2, inner="i_in", outer="i_out")
                .interchange("i_out", "j", "i_in")
            )
            print("schedule:", repr(sched))
            print("loop nest:", " ".join(sched.order))
            facts_of(sched)
            inputs = resh_mod.example_inputs()["transpose"]
            fact = ex.differential(resh_mod.transpose, sched, inputs)
            report(fact)
            got = ex.run(sched, a=resh_mod.matrix(),
                         b=Arr.zeros((resh_mod.COLS, resh_mod.ROWS)))
            expected = resh_mod.matrix().numpy().T
            print("  device b == a.T :",
                  bool(np.array_equal(np.asarray(got["b"]), expected)))

        ok["reshape transpose split+interchange"] = case(
            "reshape: transpose split(i,2).interchange(i_out,j,i_in) on the device",
            reshape_transpose,
        )

        def reshape_views():
            for name in ("rows_of", "cols_of"):
                fn = getattr(resh_mod, name)
                sched = Schedule(fn, target="opencl")
                print("schedule:", repr(sched))
                inputs = resh_mod.example_inputs()[name]
                fact = ex.differential(fn, sched, inputs)
                report(fact)

        ok["reshape row/column views"] = case(
            "reshape: rows_of and cols_of on the device", reshape_views
        )

    banner("summary")
    for name, good in ok.items():
        print(f"  {'OK    ' if good else 'FAILED'}  {name}")
    return 0 if all(ok.values()) else 1


if __name__ == "__main__":
    sys.exit(main())
