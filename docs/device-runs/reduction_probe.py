"""Why the design note's spmv device schedule does not reach a device.

The schedule ``tag(r="g.0").split("j", 32, inner="j_in").tag(j_in="l.0")`` asks
for one work group per row and the entries of a row across the lanes of that
group. loopty's own checker accepts it: the reindexing is a bijection, the order
is monotone on the dependences, and the reassociation is licensed. loopy then
refuses to generate code for it, twice over, and this script records both
refusals so that the next person does not have to rediscover them.

Run it the same way as ``device_demos.py``::

    PYOPENCL_CTX=<platform>:<device> python reduction_probe.py
"""

from __future__ import annotations

import importlib.util
import os
import sys

import numpy as np


def _repo_root() -> str:
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
    raise SystemExit("cannot find a loopty checkout with examples/; set LOOPTY_REPO")


REPO = _repo_root()
sys.path.insert(0, REPO)


def load(name: str, relpath: str):
    spec = importlib.util.spec_from_file_location(name, os.path.join(REPO, relpath))
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


def banner(title: str) -> None:
    print()
    print("-" * 72)
    print(title)
    print("-" * 72)


def main() -> int:
    import loopy as lp
    import pyopencl as cl

    from loopty.executor import LoopyExecutor, _call_arguments
    from loopty.schedule import Schedule

    ctx = cl.create_some_context(interactive=False)
    queue = cl.CommandQueue(ctx)
    device = ctx.devices[0]
    print("device:", device.name.strip(), "|", device.platform.name.strip())

    spmv_mod = load("probe_spmv", "examples/spmv.py")
    ex = LoopyExecutor(target="opencl")

    def inputs() -> dict:
        data = spmv_mod.random_csr()
        return {k: data[k] for k in ("cnt", "col", "val", "x", "y")}

    def attempt(label, build, fix=None) -> None:
        banner(label)
        try:
            sched = build()
            print("schedule:", repr(sched))
            if fix is None:
                fact = ex.differential(spmv_mod.spmv, sched, inputs())
                for name, detail in fact.provenance.get("outputs", {}).items():
                    print(
                        f"  {name}: difference {detail['difference']:.3g} within "
                        f"{detail['tolerance']:.3g} ({detail['exactness']})"
                    )
                print("  RESULT: ran, fact =", fact.status.value)
                return
            knl = fix(sched.kernel)
            args = inputs()
            call = _call_arguments(sched.term, sched.lowering, (), args)
            _evt, results = knl.executor(ctx)(queue, **call)
            out = ex._collect(sched.lowering, call, results)
            print("  RESULT: ran, y =", np.asarray(out["y"]))
        except Exception as exc:  # noqa: BLE001 - recording the refusal is the point
            print("  RESULT: refused by", type(exc).__name__)
            print("  " + str(exc).strip()[:500])

    base = lambda: Schedule(spmv_mod.spmv, target="opencl")  # noqa: E731

    attempt(
        "1. rows across work groups only: tag(r='g.0')",
        lambda: base().tag(r="g.0"),
    )
    attempt(
        "2. the whole ragged reduction across lanes: tag(r='g.0').tag(j='l.0')",
        lambda: base().tag(r="g.0").tag(j="l.0"),
    )
    attempt(
        "3. the design note's schedule, unmodified",
        lambda: base()
        .tag(r="g.0")
        .split("j", 32, inner="j_in", outer="j_out")
        .tag(j_in="l.0")
        .realize("y", tree=True),
    )
    attempt(
        "4. the same, with loopy's own remedy: split_reduction_outward('j_out')",
        lambda: base()
        .tag(r="g.0")
        .split("j", 32, inner="j_in", outer="j_out")
        .tag(j_in="l.0")
        .realize("y", tree=True),
        fix=lambda knl: lp.split_reduction_outward(knl, "j_out"),
    )
    attempt(
        "5. the same, with split_reduction_inward('j_in')",
        lambda: base()
        .tag(r="g.0")
        .split("j", 32, inner="j_in", outer="j_out")
        .tag(j_in="l.0")
        .realize("y", tree=True),
        fix=lambda knl: lp.split_reduction_inward(knl, "j_in"),
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
