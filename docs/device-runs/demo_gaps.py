"""Three examples of sumpy, firedrake and grudge kernels on an OpenCL device.

sumpy's QBX CPU tile with complex ``log`` and ``real``
(``examples/layer_potential.py``), the P1 mass kernel with constant tables on
a group and its work items (``examples/quadrature.py``), and the two
sum-factorization passes fused on the element and the row
(``examples/sum_factorization.py``), each compiled for ``target="opencl"``
and compared with the demo's own body. A driver, not
a test: it imports the examples as they are and changes only the schedule's
target, and the tags a device has.

Run with ``PYOPENCL_CTX`` naming the device, from anywhere in the checkout,
or with ``LOOPTY_REPO`` pointing at it.
"""

from __future__ import annotations

import importlib.util
import os
import sys
import warnings
from pathlib import Path


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


def main() -> int:
    warnings.simplefilter("ignore")
    from loopty import Schedule
    from loopty.executor import LoopyExecutor, emit_code

    executor = LoopyExecutor(target="opencl")
    ok = True

    def report(label, kern, schedule, inputs) -> None:
        nonlocal ok
        print(f"--- {label}")
        print(f"  {schedule!r}")
        print(f"  buildable: {schedule.buildable}")
        fact = executor.differential(kern, schedule, inputs)
        for name, detail in fact.provenance["outputs"].items():
            print(
                f"  {name}: difference {detail['difference']:.3g} within "
                f"{detail['tolerance']:.3g} ({detail['exactness']}) -> "
                f"{fact.status.value}"
            )
        ok = ok and fact.status.value == "tested"

    potential = load("layer_potential")
    report(
        "sumpy's QBX tile, complex points",
        potential.single_layer,
        potential.sumpy_tile("opencl"),
        potential.example_inputs()["single_layer"],
    )

    quadrature = load("quadrature")
    mass = (
        Schedule(quadrature.mass_action, target="opencl")
        .split(quadrature.mass_action.c, 32)
        .on_groups(quadrature.mass_action.c.outer)
        .on_items(quadrature.mass_action.c.inner)
    )
    report(
        "P1 mass with constant tables, cells on g.0/l.0",
        quadrature.mass_action,
        mass,
        quadrature.inputs_on(quadrature.square_mesh(8)),
    )
    for line in emit_code(mass).splitlines():
        if "__constant" in line:
            print(f"  {line}")

    factored = load("sum_factorization")
    report(
        "two passes fused on the element and the row",
        factored.weak_derivative,
        Schedule(factored.weak_derivative, target="opencl").fuse(
            factored.weak_derivative.along_r.a, factored.weak_derivative.along_s.a
        ),
        factored.inputs(),
    )
    print("ALL OK" if ok else "SOMETHING FAILED")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
