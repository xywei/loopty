"""PyOP2's assembly loop on an OpenCL device: the updates of ``y`` atomic.

``examples/assembly.py`` adds each cell's three values into its nodes,
``y[cell[c, k]] += yl[c, k]``. This driver takes its schedules to a device:
PyOP2's GPU split (Kulkarni), the cells by 32 onto ``g.0``/``l.0``, refused
without atomics and accepted with them, and Sun et al.'s split by 4 with
``ilp``, retargeted from C; each run on PyOP2's two triangles and on meshes of
the unit square large enough that many work items add into one node at once,
and compared with the example's own body. The same split without atomics is
compiled and run too, past loopty's refusal, to show what the refusal is
about: with the updates racing, ``y`` comes out wrong. A driver, not a test:
it imports the example as it is.

Run with ``PYOPENCL_CTX`` naming the device, from anywhere in the checkout,
or with ``LOOPTY_REPO`` pointing at it.
"""

from __future__ import annotations

import importlib.util
import os
import sys
import warnings
from pathlib import Path

import numpy as np


def repository() -> Path:
    """The checkout: ``LOOPTY_REPO``, or the first parent with ``examples/``."""
    given = os.environ.get("LOOPTY_REPO")
    if given:
        return Path(given)
    here = Path(__file__).resolve()
    for parent in here.parents:
        if (parent / "examples").is_dir() and (parent / "src" / "loopty").is_dir():
            return parent
    raise SystemExit("cannot find the loopty checkout; set LOOPTY_REPO")


def load(name: str):
    """The example ``name``, imported from the checkout's ``examples/``."""
    path = repository() / "examples" / f"{name}.py"
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def racing(assembly, mesh) -> tuple[float, float]:
    """The largest error of ``y`` with the updates of PyOP2's split not atomic.

    And the largest value of ``y``, for the scale.

    The kernel loopty refuses, built with loopy directly from the lowering:
    the split and the tags applied to the kernel of the identity schedule.
    """
    import loopy as lp
    import pyopencl as cl

    from loopty import Schedule
    from loopty.isl_reading import declining

    kernel = Schedule(assembly.mass_action, target="opencl").kernel
    kernel = lp.split_iname(kernel, "c", 32, outer_tag="g.0", inner_tag="l.0")
    kernel = lp.set_options(kernel, return_dict=True)
    data = assembly.inputs_on(mesh)
    context = cl.create_some_context(interactive=False)
    queue = cl.CommandQueue(context)
    arrays = {
        name: value.numpy().astype(np.int32) if name == "cell" else value.numpy()
        for name, value in data.items()
    }
    with declining():
        _event, out = kernel.executor(context)(queue, **arrays)
    want = assembly.assembled(mesh)
    return float(np.abs(np.asarray(out["y"]) - want).max()), float(np.abs(want).max())


def main() -> int:
    warnings.simplefilter("ignore")
    from loopty.executor import LoopyExecutor, emit_code
    from loopty.schedule import IllegalCast

    assembly = load("assembly")
    executor = LoopyExecutor(target="opencl")
    ok = True

    print("--- PyOP2's split, the updates not atomic")
    try:
        assembly.kulkarni(atomic=False, target="opencl")
    except IllegalCast as refused:
        print(f"  IllegalCast: {refused}")
    else:
        ok = False
        print("  ACCEPTED, which it should not be")

    kulkarni = assembly.kulkarni(atomic=True, target="opencl")
    vectorized = assembly.vectorized.retarget("opencl")
    for label, schedule in (
        ("PyOP2's split, atomic updates", kulkarni),
        ("Sun et al.'s split, retargeted from C", vectorized),
    ):
        print(f"--- {label}")
        print(f"  {schedule!r}")
        print(f"  buildable: {schedule.buildable}")
        for name, mesh in (
            ("PyOP2's two triangles", assembly.pyop2_mesh()),
            ("a square of 2 x 64 x 64 cells", assembly.square_mesh(64)),
            ("a square of 2 x 256 x 256 cells", assembly.square_mesh(256)),
        ):
            fact = executor.differential(
                assembly.mass_action, schedule, assembly.inputs_on(mesh)
            )
            detail = fact.provenance["outputs"]["y"]
            print(
                f"  {name}: y differs by {detail['difference']:.3g} within "
                f"{detail['tolerance']:.3g} ({detail['exactness']}) -> "
                f"{fact.status.value}"
            )
            ok = ok and fact.status.value == "tested"
    for line in emit_code(kulkarni).splitlines():
        if "atom_cmpxchg" in line or "volatile" in line:
            print(f"  {line.strip()}")

    print("--- PyOP2's split compiled past the refusal, the updates racing")
    for name, mesh in (
        ("a square of 2 x 64 x 64 cells", assembly.square_mesh(64)),
        ("a square of 2 x 256 x 256 cells", assembly.square_mesh(256)),
    ):
        error, scale = racing(assembly, mesh)
        print(f"  {name}: largest error of y {error:.3g}, y at most {scale:.3g}")

    print("ALL OK" if ok else "SOMETHING FAILED")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
