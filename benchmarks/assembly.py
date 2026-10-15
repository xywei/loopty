"""Time PyOP2's assembly loop: loopty's schedules against the same loops in plain loopy.

``examples/assembly.py`` is the loop: per cell, FFC's ``mass_ffc`` on the
cell's corners, and the cell's matrix applied to ``u`` and added into ``y``.
This driver times what loopty generates for it beside a plain-loopy
transcription of PyOP2's wrapper around the same kernel, each with the same
schedule written as PyOP2's branches write it in loopy:

``serial``
    the loop over the cells as it stands, on the C target;
``sun``
    Sun et al.'s vectorization, the cells split by 4 with the four in no
    order (``ilp``) under ``nc % 4 == 0``, on the C target;
``kulkarni``
    Kulkarni's GPU split, the cells by 32 onto ``g.0``/``l.0`` with every
    update of ``y`` atomic, on an OpenCL device (``PYOPENCL_CTX``).

The baseline (:func:`pyop2_transcription`) is what PyOP2's code generator
makes of the loop: a wrapper kernel over the cells that packs the corners,
calls the element kernel and adds into ``y``, merged with the element kernel
in TSFC's form (its tables read-only initialized temporaries), and the call
inlined with ``inline_callable_kernel``, as PyOP2's GPU path does; Kulkarni's
transform (:func:`kulkarni_transform`) is applied to it as his branch applies
it. Both sides run on the same mesh and arrays, through loopy's own invoker,
and are compiled with the same flags.

Each case is built, run once and checked against the closed form of ``y``,
and run once more to warm up; then the cases are timed in turn, ``--repeat``
rounds of one call each, so that a change in the machine's load falls on all
of them. The median, the minimum and the maximum are printed, and written as
CSV with ``--csv``, with the versions, the flags and the device. Run it on an
otherwise quiet machine::

    python benchmarks/assembly.py c --n 708 --flags native
    PYOPENCL_CTX=0:0 python benchmarks/assembly.py opencl --n 708

``--baseline-only`` times the plain-loopy side alone, without importing loopty,
for a loopy that loopty does not run on.

``--nested`` (C target) adds ``serial-nested`` and ``sun-nested`` on the
plain-loopy side: the transcription with the loops inside a cell nested under
the loop over the cells (:func:`nested`), as loopty's kernel has them. With
Sun et al.'s split that decides where loopy puts the ``ilp`` loop, and so
whether the four cells run interleaved or one after another; these two cases
say how much of a difference between the sides that is.
"""

from __future__ import annotations

import argparse
import contextlib
import csv
import importlib.util
import os
import platform
import statistics
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent

#: The compiler flags of each ``--flags`` set, after loopy's own ``-std=c99
#: -O3 -fPIC``. ``-fwrapv`` is loopty's on the C target, and both sides get it.
FLAGS = {
    "default": ["-fwrapv"],
    "native": ["-fwrapv", "-march=native", "-ffp-contract=off"],
    "native-fma": ["-fwrapv", "-march=native", "-ffp-contract=fast"],
}

#: ``mass_ffc``'s tables, as firedrake's PyOP2 test writes them.
W3 = np.array([0.166666666666667, 0.166666666666667, 0.166666666666667])
FE0 = np.array(
    [
        [0.666666666666667, 0.166666666666667, 0.166666666666667],
        [0.166666666666667, 0.166666666666667, 0.666666666666667],
        [0.166666666666667, 0.666666666666667, 0.166666666666667],
    ]
)


# {{{ the mesh


def square_mesh(n: int) -> tuple[np.ndarray, np.ndarray]:
    """The unit square in ``n`` by ``n`` squares, each cut into two triangles.

    The same mesh as ``examples/assembly.py``'s, in the same order.
    """
    grid = np.linspace(0.0, 1.0, n + 1)
    xs, ys = (axis.ravel() for axis in np.meshgrid(grid, grid, indexing="ij"))
    i, j = (axis.ravel() for axis in np.meshgrid(range(n), range(n), indexing="ij"))
    a, b = i * (n + 1) + j, (i + 1) * (n + 1) + j
    cells = np.stack([a, b, b + 1, a, b + 1, a + 1], axis=1).reshape(-1, 3)
    return np.stack([xs, ys], axis=1), cells


def closed_form(coords: np.ndarray, cells: np.ndarray, u: np.ndarray) -> np.ndarray:
    """``y`` in closed form: each cell's ``|J| (u_j + u_0 + u_1 + u_2) / 24``, added."""
    x, local = coords[cells], u[cells]
    J = (x[:, 1, 0] - x[:, 0, 0]) * (x[:, 2, 1] - x[:, 0, 1]) - (
        x[:, 2, 0] - x[:, 0, 0]
    ) * (x[:, 1, 1] - x[:, 0, 1])
    element = np.abs(J)[:, None] * (local + local.sum(axis=1)[:, None]) / 24
    out = np.zeros(len(coords))
    np.add.at(out, cells, element)
    return out


# }}}


# {{{ the plain-loopy side


def pyop2_transcription(target):
    """PyOP2's wrapper around ``mass_ffc``, in plain loopy, with the call inlined.

    The element kernel is TSFC's form of ``mass_ffc``: one domain per index,
    the tables read-only initialized temporaries, the Jacobian's determinant
    a scalar temporary. The wrapper is rep2loopy's: a loop over the cells,
    the corners packed into a temporary through the cell map, the kernel
    called on the packed corners and the cell's row of ``Ae``, and the
    ``INC`` unpack into ``y``. ``lp.merge`` joins the two and
    ``lp.inline_callable_kernel`` inlines the call, as PyOP2's GPU path does.
    """
    import loopy as lp

    tables = [
        lp.TemporaryVariable(
            name,
            dtype=np.float64,
            shape=values.shape,
            initializer=values,
            address_space=lp.AddressSpace.LOCAL,
            read_only=True,
        )
        for name, values in (("W3", W3), ("FE0", FE0))
    ]
    element = lp.make_function(
        ["{[j]: 0 <= j < 3}", "{[k]: 0 <= k < 3}", "{[ip]: 0 <= ip < 3}"],
        """
        <> J_00 = x[1, 0] - x[0, 0]
        <> J_01 = x[2, 0] - x[0, 0]
        <> J_10 = x[1, 1] - x[0, 1]
        <> J_11 = x[2, 1] - x[0, 1]
        <> det = fabs(J_00*J_11 - J_01*J_10)
        A[j, k] = sum(ip, FE0[ip, j]*FE0[ip, k]*W3[ip]*det)
        """,
        [
            lp.GlobalArg("A", np.float64, shape=(3, 3), is_input=False, is_output=True),
            lp.GlobalArg("x", np.float64, shape=(3, 2), is_input=True, is_output=False),
            *tables,
        ],
        name="mass_ffc",
        target=target,
        lang_version=(2018, 2),
    )
    wrapper = lp.make_kernel(
        [
            "{[c]: 0 <= c < nc}",
            "{[k, d, q, e]: 0 <= k, q < 3 and 0 <= d, e < 2}",
            "{[j, l, a, b]: 0 <= j, l, a, b < 3}",
        ],
        """
        t1[k, d] = coords[cell[c, k], d]  {id=pack}
        [j, l]: Ae[c, j, l] = mass_ffc([q, e]: t1[q, e])  {id=call, dep=pack}
        y[cell[c, a]] = y[cell[c, a]] + sum(b, Ae[c, a, b]*u[cell[c, b]])  {dep=call}
        """,
        [
            lp.GlobalArg("cell", np.int32, shape=("nc", 3)),
            lp.GlobalArg(
                "coords", np.float64, shape=None, dim_tags="stride:2,stride:1"
            ),
            lp.GlobalArg("u", np.float64, shape=None),
            lp.GlobalArg("Ae", np.float64, shape=("nc", 3, 3), is_output=True),
            lp.GlobalArg("y", np.float64, shape=None, is_output=True),
            lp.ValueArg("nc", np.int32),
            lp.TemporaryVariable(
                "t1", np.float64, shape=(3, 2), address_space=lp.AddressSpace.PRIVATE
            ),
        ],
        name="wrap_mass_action",
        target=target,
        lang_version=(2018, 2),
        silenced_warnings=["write_race*"],
    )
    merged = lp.merge([wrapper, element])
    merged = merged.copy(entrypoints=frozenset({"wrap_mass_action"}))
    return lp.inline_callable_kernel(merged, "mass_ffc")


def nested(t_unit, iname: str = "c"):
    """The transcription with each loop inside a cell nested under the cell loop.

    TSFC and rep2loopy write one rectangular domain per index, none of them
    depending on the loop over the cells. loopty's kernel nests the loops of
    a cell under it (their domains take ``c`` as a parameter), which leaves
    loopy no choice but to enter a cell before its loops. The two give the
    same loops and the same code without a schedule; with Sun et al.'s
    split, loopy's scheduler places an ``ilp`` loop innermost where it can,
    so the transcription runs the four cells interleaved inside the
    element's loops and loopty's kernel runs each cell whole, four in a row.
    This variant gives the transcription loopty's nesting, and nothing else:
    the domains an instruction uses together are joined (loopy's domain tree
    does not branch), and each is made a child of the cell loop's.

    loopy main keeps a kernel's domains as namedisl sets (loopy #1023,
    ``docs/loopy-main.md``): they are joined here as islpy sets and given
    back as namedisl ones. Not through :mod:`loopty.named`, which would
    import loopty, as ``--baseline-only`` does not.
    """
    from functools import reduce

    import islpy as isl

    kernel = t_unit.default_entrypoint
    named = any(hasattr(domain, "as_isl") for domain in kernel.domains)
    current = [domain.as_isl() if named else domain for domain in kernel.domains]

    def names(domain) -> set[str]:
        return set(domain.get_var_names(isl.dim_type.set))

    (cell,) = [domain for domain in current if iname in names(domain)]
    others = [domain for domain in current if iname not in names(domain)]
    groups = [{i} for i in range(len(others))]
    for insn in kernel.instructions:
        used = set(insn.within_inames) | set(insn.reduction_inames())
        hit = [g for g in groups if any(names(others[i]) & used for i in g)]
        if len(hit) > 1:
            groups = [g for g in groups if g not in hit] + [set().union(*hit)]
    domains = [cell]
    for group in groups:
        domain = reduce(
            lambda a, b: a.flat_product(b), [others[i] for i in sorted(group)]
        )
        domain = domain.insert_dims(isl.dim_type.param, 0, 1).set_dim_name(
            isl.dim_type.param, 0, iname
        )
        domains.append(domain)
    if named:
        import namedisl

        domains = [namedisl.to_named(domain) for domain in domains]
        domains = [
            domain.as_set() if isinstance(domain, namedisl.BasicSet) else domain
            for domain in domains
        ]
    return t_unit.with_kernel(kernel.copy(domains=domains))


def sun_transform(t_unit):
    """Sun et al.'s vectorization: the cells by 4, the four in no order.

    Their branch splits the loop by the vector width under the assumption
    that its extent is a multiple of it, and tags the inner loop with
    ``c_vec``, a tag of their loopy fork; ``ilp`` is loopy's.
    """
    import loopy as lp

    t_unit = lp.assume(t_unit, "nc mod 4 = 0")
    return lp.split_iname(t_unit, "c", 4, inner_tag="ilp")


def kulkarni_transform(t_unit):
    """Kulkarni's GPU transform, as his branch applies it to PyOP2's wrapper.

    Every assignment to an argument that reads it is an ``AtomicUpdate`` and
    the argument ``for_atomic``; the tables move to global memory; the loop
    over the cells is split by 32 onto ``g.0``/``l.0`` under ``nc > 0``.
    """
    import loopy as lp
    import pymbolic.primitives as prim

    kernel = t_unit.default_entrypoint
    temporaries = {
        name: (
            tv.copy(address_space=lp.AddressSpace.GLOBAL)
            if tv.read_only
            else tv.copy(address_space=lp.AddressSpace.PRIVATE)
        )
        for name, tv in kernel.temporary_variables.items()
    }
    atomic: set[str] = set()
    instructions = []
    for insn in kernel.instructions:
        assignee = getattr(insn, "assignee", None)
        if isinstance(insn, lp.Assignment) and isinstance(assignee, prim.Subscript):
            name = assignee.aggregate.name
            if name in kernel.arg_dict and name in insn.read_dependency_names():
                insn = insn.copy(atomicity=(lp.AtomicUpdate(name),))
                atomic.add(name)
        instructions.append(insn)
    args = [
        arg.copy(for_atomic=True) if arg.name in atomic else arg for arg in kernel.args
    ]
    kernel = kernel.copy(
        instructions=instructions, args=args, temporary_variables=temporaries
    )
    t_unit = t_unit.with_kernel(kernel)
    t_unit = lp.assume(t_unit, "nc > 0")
    return lp.split_iname(t_unit, "c", 32, outer_tag="g.0", inner_tag="l.0")


# }}}


# {{{ the loopty side


def example():
    """``examples/assembly.py``, imported from the checkout."""
    path = ROOT / "examples" / "assembly.py"
    spec = importlib.util.spec_from_file_location("assembly", path)
    module = importlib.util.module_from_spec(spec)
    sys.modules["assembly"] = module
    spec.loader.exec_module(module)
    return module


def loopty_kernels(target: str) -> dict:
    """loopty's loopy kernels for each schedule on ``target``: the example's own."""
    assembly = example()
    from loopty import Schedule

    if target == "c":
        return {
            "serial": Schedule(assembly.mass_action, target="c").kernel,
            "sun": assembly.vectorized.kernel,
        }
    return {"kulkarni": assembly.kulkarni(atomic=True, target="opencl").kernel}


# }}}


# {{{ timing


def environment(target: str, flags: str, device: str) -> dict:
    """What the numbers depend on, for the CSV: versions, flags and the device."""
    import importlib.metadata as metadata

    def version(name: str) -> str:
        try:
            return metadata.version(name)
        except metadata.PackageNotFoundError:
            return "-"

    try:
        compiler = subprocess.run(
            ["gcc", "--version"], capture_output=True, text=True, check=False
        ).stdout.splitlines()[0]
    except (OSError, IndexError):
        compiler = "-"
    cpu = "-"
    try:
        for line in Path("/proc/cpuinfo").read_text().splitlines():
            if line.startswith("model name"):
                cpu = line.split(":", 1)[1].strip()
                break
    except OSError:
        pass
    return {
        "target": target,
        "flags": (
            " ".join(["-std=c99 -O3 -fPIC", *FLAGS[flags]]) if target == "c" else "-"
        ),
        "device": device,
        "cpu": cpu,
        "load1": f"{os.getloadavg()[0]:.2f}",
        "python": platform.python_version(),
        "loopy": version("loopy"),
        "islpy": version("islpy"),
        "pyopencl": version("pyopencl"),
        "numpy": version("numpy"),
        "loopty": version("loopty"),
        "compiler": compiler,
    }


def readings(side: str):
    """loopty's readings of loopy's subscripts for its own kernels, nothing else."""
    if side != "loopty":
        return contextlib.nullcontext()
    from loopty.isl_reading import declining

    return declining()


def prepare_c(t_unit, flags: str, data: dict):
    """Compile ``t_unit`` for the C target; one checked run, and a timed call.

    The call returns the seconds it took. The C target writes the arrays it
    is passed, so the checked run gets copies and the timed calls another.
    """
    import loopy as lp

    t_unit = lp.set_options(t_unit, build_options=FLAGS[flags])
    executor = t_unit.executor()
    first = {name: value.copy() for name, value in data.items()}
    executor(**first)
    outputs = {"Ae": first["Ae"], "y": first["y"]}
    work = {name: value.copy() for name, value in data.items()}

    def call() -> tuple[float, float | None]:
        start = time.perf_counter()
        executor(**work)
        return time.perf_counter() - start, None

    return call, outputs, "C target, one thread"


def prepare_opencl(t_unit, data: dict):
    """Build ``t_unit`` for the ``PYOPENCL_CTX`` device, the arrays on the device.

    The call returns the wall time, from Python to the queue's end, and the
    kernel's own time from its event.
    """
    import pyopencl as cl
    import pyopencl.array as cla

    context = cl.create_some_context(interactive=False)
    queue = cl.CommandQueue(
        context, properties=cl.command_queue_properties.PROFILING_ENABLE
    )
    device = context.devices[0]
    described = (
        f"{device.name} | {device.platform.name} {device.platform.version} | "
        f"driver {device.driver_version}"
    )
    executor = t_unit.executor(context)

    def on_device():
        return {name: cla.to_device(queue, value) for name, value in data.items()}

    first = on_device()
    executor(queue, **first)
    queue.finish()
    outputs = {"Ae": first["Ae"].get(), "y": first["y"].get()}
    work = on_device()

    def call() -> tuple[float, float | None]:
        start = time.perf_counter()
        event, _out = executor(queue, **work)
        queue.finish()
        wall = time.perf_counter() - start
        return wall, 1e-9 * (event.profile.end - event.profile.start)

    return call, outputs, described


# }}}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("target", choices=("c", "opencl"))
    parser.add_argument("--n", type=int, default=708, help="squares per side")
    parser.add_argument("--repeat", type=int, default=20)
    parser.add_argument("--flags", choices=tuple(FLAGS), default="default")
    parser.add_argument("--baseline-only", action="store_true")
    parser.add_argument(
        "--nested",
        action="store_true",
        help="also time the transcription with loopty's nesting of its loops (C)",
    )
    parser.add_argument("--csv", help="append the rows to this CSV file")
    args = parser.parse_args()

    import warnings

    warnings.simplefilter("ignore")
    import loopy as lp

    coords, cells = square_mesh(args.n)
    u = 1.0 + coords[:, 0] * coords[:, 1]
    want = closed_form(coords, cells, u)
    data = {
        "cell": cells.astype(np.int32),
        "coords": coords,
        "u": u,
        "Ae": np.zeros((len(cells), 3, 3)),
        "y": np.zeros(len(coords)),
    }
    if args.target == "c":
        target = lp.ExecutableCTarget()
        baseline = pyop2_transcription(target)
        cases = {"serial": baseline, "sun": sun_transform(baseline)}
        if args.nested:
            inside = nested(baseline)
            cases["serial-nested"] = inside
            cases["sun-nested"] = sun_transform(inside)
    else:
        target = lp.PyOpenCLTarget()
        cases = {"kulkarni": kulkarni_transform(pyop2_transcription(target))}
    sides = {"loopy": cases}
    if not args.baseline_only:
        sides["loopty"] = loopty_kernels(args.target)
    if len(cells) % 4:
        # Sun et al.'s split assumes the cells a multiple of four, and these
        # kernels are run as loopy runs them, with no check of it at the call
        print(
            f"{len(cells)} cells (2 n^2 with n odd) are not a multiple of four, "
            "which Sun et al.'s split assumes: its cases are not timed",
            flush=True,
        )
        sides = {
            side: {name: k for name, k in kernels.items() if "sun" not in name}
            for side, kernels in sides.items()
        }

    prepared = {}
    device = "-"
    for side, kernels in sides.items():
        for schedule, t_unit in kernels.items():
            # loopty's kernels are checked and generated inside its readings
            # of loopy's subscripts (loopty.isl_reading), the baseline's not
            with readings(side):
                if args.target == "c":
                    call, outputs, device = prepare_c(t_unit, args.flags, data)
                else:
                    call, outputs, device = prepare_opencl(t_unit, data)
                call()  # warm up
            error = float(np.abs(outputs["y"] - want).max() / np.abs(want).max())
            prepared[side, schedule] = (call, error)

    # the cases take turns, so that a change in the machine's load falls on all
    times: dict = {key: ([], []) for key in prepared}
    for _ in range(args.repeat):
        for key, (call, _error) in prepared.items():
            side = key[0]
            with readings(side):
                wall, kernel = call()
            times[key][0].append(wall)
            if kernel is not None:
                times[key][1].append(kernel)

    rows = []
    for (side, schedule), (call, error) in prepared.items():
        wall, kernel_times = times[side, schedule]
        rows.append(
            {
                "side": side,
                "schedule": schedule,
                "cells": len(cells),
                "repeat": args.repeat,
                "median_ms": 1e3 * statistics.median(wall),
                "min_ms": 1e3 * min(wall),
                "max_ms": 1e3 * max(wall),
                "kernel_median_ms": (
                    1e3 * statistics.median(kernel_times) if kernel_times else ""
                ),
                "y_rel_error": error,
            }
        )
        row = rows[-1]
        on_device = (
            f", kernel {row['kernel_median_ms']:.3f} ms" if kernel_times else ""
        )
        print(
            f"{side:7} {schedule:9} {row['cells']} cells: median "
            f"{row['median_ms']:.3f} ms (min {row['min_ms']:.3f}, max "
            f"{row['max_ms']:.3f}){on_device}, y within {error:.1e} of the "
            "closed form",
            flush=True,
        )
    context = environment(args.target, args.flags, device)
    for key, value in context.items():
        print(f"  {key}: {value}")
    if args.csv:
        exists = Path(args.csv).exists()
        with open(args.csv, "a", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=[*rows[0], *context])
            if not exists:
                writer.writeheader()
            for row in rows:
                writer.writerow({**row, **context})
    return 0 if all(row["y_rel_error"] < 1e-12 for row in rows) else 1


if __name__ == "__main__":
    raise SystemExit(main())
