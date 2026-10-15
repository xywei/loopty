"""Time sumpy's 2-D Laplace L2P three ways, and check that the three agree.

``examples/l2p.py`` is the L2P sumpy evaluates a box's local expansion at its
targets with (``E2PFromSingleBox``, for ``LaplaceKernel(2)`` with the
compressed Taylor expansion sumpy picks for it). This driver puts it beside

* sumpy's own kernel, ``E2PFromSingleBox(...).get_optimized_kernel()``, run on
  an OpenCL device, which needs sumpy and the loopy it is written for (loopy
  main, on namedisl); and
* the same algorithm written directly in plain loopy, as a kernel author
  would: a triangle domain per box, the targets of a box through
  ``box_target_starts``, the reconstructed coefficients in a temporary of the
  box (the L2P microkernel of loopy #970, with the Laplace step put back);

on the same tree, coefficients and targets, at each order asked for. Each
subcommand runs in one environment and reads and writes plain files, so the
three can run where their dependencies are:

``inputs``
    numpy only. A uniform quadtree of the unit square, about 30 targets per
    leaf, sorted by leaf in Morton order, and the local coefficients of a
    cluster of charges well outside it, in sumpy's storage order, scaled as
    sumpy scales them (``rscale = 2 h / p``); and the potential summed
    directly. Written as ``.npz``.
``sumpy``
    sumpy's kernel on ``PYOPENCL_CTX``'s device.
``loopy``
    the plain loopy kernel on the C target or an OpenCL device; ``--variant
    unrolled`` unrolls the two triangle loops (loopy's ``unr``), which loopy
    does over the triangle's bounding box with a guard. On C it is built
    with the flags loopty builds its kernels with, ``-fwrapv`` beside loopy's
    own (loopty's integers wrap round as numpy's do), so that the two are
    compared at the same flags; ``--no-wrapv`` builds it with loopy's alone.
``loopty``
    the kernels of ``examples/l2p.py``, written out at the order of the
    inputs (the example fixes its order as sumpy fixes a kernel's), lowered
    and compiled by loopty, then called as the loopy kernel they are.
    ``--variant windowed`` is the example's ``contracted``, each box's
    triangle and table kept in windows of the box, as plain loopy keeps them
    in a temporary of the box; ``fused`` keeps every box's in the arrays the
    program makes.
``compare``
    the largest difference between the potentials of several runs, and
    against the direct sum.
``example`` and ``record``
    what ``examples/l2p.py`` compares ``l2p`` with, sumpy's own kernel on the
    example's inputs: ``example`` writes those inputs as ``inputs`` writes
    its own (loopty's environment), ``sumpy`` runs on them (sumpy's), and
    ``record`` writes its potential beside the example, with the commits and
    the device it came from, as ``examples/l2p_sumpy.json``.

Every run records the device (platform, device, driver), the versions of the
packages that compiled it, the compiler and the flags the kernel was built
with on the C target (loopy's and the kernel's own), the time of
the first call, in fresh code caches when ``--cold``, in two parts:
``prepare``, building the kernel (sumpy's symbolic code generation; loopy's
``make_kernel``; loopty's trace, its every fact and cast checked, and its
lowering), and ``compile``, loopy's code generation, the compiler and the
first run; and the median, minimum and maximum of ``--repeat`` timed calls
after ``--warmup`` untimed ones; on a device the times are the kernel's
events, on C the call's wall time. Results are one JSON object per line.

Example, at order 10 on a tree of 4**6 leaves::

    python benchmarks/l2p.py inputs --order 10 --level 6 --out in.npz
    python benchmarks/l2p.py loopty --inputs in.npz --target c --save pot-loopty.npz
    python benchmarks/l2p.py loopy --inputs in.npz --target c --save pot-loopy.npz
    PYOPENCL_CTX=0 python benchmarks/l2p.py sumpy --inputs in.npz --save pot-sumpy.npz
    python benchmarks/l2p.py compare in.npz pot-*.npz
"""

from __future__ import annotations

import argparse
import contextlib
import importlib.metadata
import importlib.util
import json
import math
import os
import platform
import re
import statistics
import sys
import tempfile
import time
from pathlib import Path
from typing import Any

import numpy as np

#: The constant factor of sumpy's LaplaceKernel(2), -1 / (2 pi) (Kress, Thm 6.2).
SCALING = -1 / (2 * math.pi)


# {{{ inputs


def morton(level: int) -> np.ndarray:
    """The leaves of a uniform quadtree at ``level``, ``(ix, iy)``, in Morton order."""
    count = 4**level
    codes = np.arange(count)
    ix = np.zeros(count, dtype=np.int64)
    iy = np.zeros(count, dtype=np.int64)
    for bit in range(level):
        ix |= ((codes >> (2 * bit)) & 1) << bit
        iy |= ((codes >> (2 * bit + 1)) & 1) << bit
    return np.stack([ix, iy])


def derivatives(order: int, z: np.ndarray) -> np.ndarray:
    """``d^(a+b) log|z| / dx^a dy^b`` at the points ``z``, in sumpy's storage order.

    Row ``0`` is ``log|z|``; row ``2 n - 1 + a`` is the derivative with ``a``
    derivatives in ``x`` and ``n - a`` in ``y``, ``a`` in ``{0, 1}``, which
    is ``Re(i^b (-1)^(n - 1) (n - 1)! / z^n)``: ``log|z|`` is the real part of
    ``log z``, and ``d/dy`` is ``i d/dz`` on an analytic function.
    """
    out = np.empty((2 * order + 1, *z.shape))
    out[0] = np.log(np.abs(z))
    for n in range(1, order + 1):
        base = (-1) ** (n - 1) * math.factorial(n - 1) / z**n
        for a in (0, 1):
            out[2 * n - 1 + a] = ((1j) ** (n - a) * base).real
    return out


def make_inputs(order: int, level: int, per_box: float, seed: int) -> dict:
    """The tree, targets, coefficients and direct potential of one run."""
    rng = np.random.default_rng(seed)
    side = 2**level
    h = 1.0 / side
    ix, iy = morton(level)
    centers = np.stack([(ix + 0.5) * h, (iy + 0.5) * h])
    nboxes = centers.shape[1]
    counts = np.maximum(rng.poisson(per_box, size=nboxes), 1)
    starts = np.concatenate([[0], np.cumsum(counts)[:-1]])
    owner = np.repeat(np.arange(nboxes), counts)
    targets = centers[:, owner] + h * (rng.random((2, owner.size)) - 0.5)
    # a cluster of charges well away from the unit square
    nsources = 32
    radius = 0.2 * np.sqrt(rng.random(nsources))
    angle = 2 * np.pi * rng.random(nsources)
    sources = np.stack(
        [2.5 + radius * np.cos(angle), 1.5 + radius * np.sin(angle)]
    )
    charges = rng.standard_normal(nsources)
    rscale = 2 * h / order
    coeffs = np.zeros((nboxes, 2 * order + 1))
    scale = np.array(
        [1.0] + [rscale ** n for n in range(1, order + 1) for _a in (0, 1)]
    )
    for k in range(nsources):
        z = (centers[0] - sources[0, k]) + 1j * (centers[1] - sources[1, k])
        coeffs += charges[k] * derivatives(order, z).T
    coeffs *= scale
    direct = np.zeros(owner.size)
    for k in range(nsources):
        direct += charges[k] * np.log(
            np.hypot(targets[0] - sources[0, k], targets[1] - sources[1, k])
        )
    direct *= SCALING
    return {
        "order": np.int64(order),
        "level": np.int64(level),
        "seed": np.int64(seed),
        "coeffs": coeffs,
        "centers": centers,
        "counts": counts.astype(np.int64),
        "starts": starts.astype(np.int64),
        "targets": targets,
        "rscale": np.float64(rscale),
        "direct": direct,
    }


def load_inputs(path: str) -> dict:
    """The inputs ``inputs`` wrote, as numpy arrays and Python numbers."""
    with np.load(path) as data:
        out = {key: data[key] for key in data.files}
    for key in ("order", "level", "seed"):
        out[key] = int(out[key])
    out["rscale"] = float(out["rscale"])
    return out


# }}}


# {{{ what a run records


def versions(*names: str) -> dict[str, str]:
    """The installed version of each package, and the commit of one from Git."""
    out = {"python": platform.python_version()}
    for name in names:
        try:
            dist = importlib.metadata.distribution(name)
        except importlib.metadata.PackageNotFoundError:
            continue
        version = dist.version
        direct = dist.read_text("direct_url.json")
        if direct:
            info = json.loads(direct)
            commit = info.get("vcs_info", {}).get("commit_id")
            if commit:
                version += f" ({info.get('url', '')}@{commit[:10]})"
        out[name] = version
    return out


def opencl_device(queue: Any) -> dict[str, str]:
    """The platform, device and driver of ``queue``, as a run's record names them."""
    import pyopencl as cl

    device = queue.device
    return {
        "platform": device.platform.name,
        "platform_version": device.platform.version,
        "device": device.name.strip(),
        "device_type": cl.device_type.to_string(device.type),
        "driver": device.driver_version,
        "compute_units": str(device.max_compute_units),
    }


def c_toolchain(kernel: Any) -> dict[str, str]:
    """The compiler, and the flags ``kernel`` is compiled with on the C target here.

    loopy's C target compiles with its toolchain's flags and then the
    kernel's own build options, which loopty sets on every kernel it lowers
    (``-fwrapv``, and ``-ffp-contract=off`` for an ``exact`` output); both
    are recorded, so that two runs are compared at the flags they had.
    """
    import loopy as lp

    target = lp.ExecutableCTarget()
    toolchain = target.compiler.toolchain
    options = list(kernel.default_entrypoint.options.build_options or [])
    return {
        "cc": " ".join(toolchain.cc) if isinstance(toolchain.cc, list) else str(
            toolchain.cc
        ),
        "cflags": " ".join([*toolchain.cflags, *options]),
        "build_options": " ".join(options),
        "cpu": cpu_model(),
    }


def device_kernels(kernel: Any) -> int | None:
    """How many device kernels loopy makes of ``kernel``, several at a barrier.

    ``None`` where loopy cannot generate the code before the call gives the
    arguments' types, as for sumpy's kernel, which is one loop over the boxes
    with no barrier, so one kernel.
    """
    import loopy as lp

    try:
        return len(lp.generate_code_v2(kernel).device_programs)
    except lp.LoopyError:
        return None


def cpu_model() -> str:
    """The CPU's model name, from ``/proc/cpuinfo`` where there is one."""
    with contextlib.suppress(OSError):
        for line in Path("/proc/cpuinfo").read_text().splitlines():
            if line.startswith("model name"):
                return line.split(":", 1)[1].strip()
    return platform.processor()


def summary(times: list[float]) -> dict[str, float]:
    """Median, minimum and maximum of the timed calls, in seconds."""
    return {
        "median": statistics.median(times),
        "min": min(times),
        "max": max(times),
        "runs": len(times),
    }


def emit(record: dict, out: str | None) -> None:
    """Print ``record`` as one JSON line, and append it to ``out``."""
    line = json.dumps(record, sort_keys=True)
    print(line, flush=True)
    if out:
        with open(out, "a") as handle:
            handle.write(line + "\n")


def cold_caches() -> None:
    """Fresh code caches, loopy's, PoCL's and CUDA's, for the rest of the process.

    Set before loopy or pyopencl is imported, which read where their caches
    are when they are imported, so that the first call generates and
    compiles everything; the directory is removed when the process exits.
    """
    import atexit
    import shutil

    fresh = tempfile.mkdtemp(prefix="l2p-cache-")
    atexit.register(shutil.rmtree, fresh, ignore_errors=True)
    os.environ["XDG_CACHE_HOME"] = os.path.join(fresh, "xdg")
    os.environ["POCL_CACHE_DIR"] = os.path.join(fresh, "pocl")
    os.environ["CUDA_CACHE_DISABLE"] = "1"


def opencl_queue() -> Any:
    """A profiling queue on ``PYOPENCL_CTX``'s device."""
    import pyopencl as cl

    context = cl.create_some_context(interactive=False)
    return cl.CommandQueue(
        context, properties=cl.command_queue_properties.PROFILING_ENABLE
    )


def time_opencl(
    call: Any, queue: Any, warmup: int, repeat: int, kernels: int | None = 1
) -> list[float]:
    """The device time of ``repeat`` calls of ``call()``, after ``warmup`` calls.

    ``call`` returns the event of the kernel it enqueues. Where loopy splits a
    kernel in several (``kernels`` > 1), the event is the last one's, and the
    wall time of the call to ``queue.finish()`` is taken instead.
    """
    for _ in range(warmup):
        call()
    queue.finish()
    times = []
    for _ in range(repeat):
        start = time.perf_counter()
        event = call()
        queue.finish()
        wall = time.perf_counter() - start
        device = None
        if kernels in (1, None):
            with contextlib.suppress(Exception):
                device = (event.profile.end - event.profile.start) * 1e-9
        times.append(wall if device is None else device)
    return times


def time_c(call: Any, warmup: int, repeat: int) -> list[float]:
    """The wall time of ``repeat`` calls of ``call()``, after ``warmup`` calls."""
    for _ in range(warmup):
        call()
    times = []
    for _ in range(repeat):
        start = time.perf_counter()
        call()
        times.append(time.perf_counter() - start)
    return times


# }}}


# {{{ sumpy's own kernel


def run_sumpy(inputs: dict, args: argparse.Namespace) -> tuple[dict, np.ndarray]:
    """sumpy's E2PFromSingleBox on the inputs, on ``PYOPENCL_CTX``'s device."""
    import pyopencl.array as cla
    from sumpy.e2p import E2PFromSingleBox
    from sumpy.expansion.local import LinearPDEConformingVolumeTaylorLocalExpansion
    from sumpy.kernel import LaplaceKernel

    order = inputs["order"]
    kernel = LaplaceKernel(2)
    expansion = LinearPDEConformingVolumeTaylorLocalExpansion(kernel, order)
    stored = [tuple(mi) for mi in expansion.get_coefficient_identifiers()]
    expected = [(0, 0)] + [
        (a, n - a) for n in range(1, order + 1) for a in (0, 1)
    ]
    if sorted(stored) != sorted(expected):
        raise SystemExit(f"sumpy stores {stored}, not the {expected} assumed")
    # the inputs are in the order 2 n - 1 + a; sumpy's own may differ
    place = [expected.index(mi) for mi in stored]
    coeffs = np.ascontiguousarray(inputs["coeffs"][:, place])

    queue = opencl_queue()
    nboxes = inputs["centers"].shape[1]
    ntargets = inputs["targets"].shape[1]
    start = time.perf_counter()
    program = E2PFromSingleBox(expansion, [kernel]).get_optimized_kernel()
    prepared = time.perf_counter() - start
    device_args = {
        "src_expansions": cla.to_device(queue, coeffs),
        "src_base_ibox": np.int32(0),
        "nsrc_level_boxes": np.int32(nboxes),
        "naligned_boxes": np.int32(nboxes),
        "target_boxes": cla.to_device(queue, np.arange(nboxes, dtype=np.int32)),
        "box_target_starts": cla.to_device(
            queue, inputs["starts"].astype(np.int32)
        ),
        "box_target_counts_nonchild": cla.to_device(
            queue, inputs["counts"].astype(np.int32)
        ),
        "centers": cla.to_device(queue, np.ascontiguousarray(inputs["centers"])),
        "targets": [
            cla.to_device(queue, np.ascontiguousarray(inputs["targets"][d]))
            for d in range(2)
        ],
        "rscale": np.float64(inputs["rscale"]),
        "ntargets": np.int32(ntargets),
        "result": [cla.zeros(queue, ntargets, np.float64)],
    }

    start = time.perf_counter()
    executor = program.executor(queue.context)

    def call() -> Any:
        event, _out = executor(queue, **device_args)
        return event

    call()
    queue.finish()
    compiled = time.perf_counter() - start
    kernels = device_kernels(program)
    times = time_opencl(call, queue, args.warmup, args.repeat, kernels)
    pot = device_args["result"][0].get()
    record = {
        "implementation": "sumpy",
        "variant": "E2PFromSingleBox.get_optimized_kernel",
        "target": "opencl",
        "order": order,
        "nboxes": nboxes,
        "ntargets": ntargets,
        "prepare": prepared,
        "compile": compiled,
        "first_call": prepared + compiled,
        "cold": args.cold,
        "device_kernels": kernels,
        "time": summary(times),
        "times": times,
        "device": opencl_device(queue),
        "versions": versions(
            "sumpy", "loopy", "islpy", "namedisl", "pymbolic", "pytools",
            "pyopencl", "numpy", "boxtree", "arraycontext", "sympy",
        ),
    }
    return record, pot


# }}}


# {{{ the same algorithm in plain loopy


#: The plain loopy variants: ``loop``, ``unrolled`` (the triangle's loops
#: unrolled), and those that make loopty's choices in plain loopy, to tell its
#: costs apart: ``pow`` computes ``x**a`` with the C library's ``pow``, as
#: loopty computes ``x**a`` to round as numpy does, where loopy multiplies
#: (loopty's ``ipow`` multiplies as loopy does); ``global``, ``unrolled_global``
#: and ``pow_global`` keep every box's triangle in one array of the kernel's,
#: zeroed first, as loopty's program keeps an array it makes, where loopy keeps
#: one box's in a temporary of the box; ``inline`` writes ``x`` and ``y`` into
#: the sum, computed again at every term, as loopty's code of the example has
#: them, where loopy computes them once per target.
LOOPY_VARIANTS = (
    "loop", "unrolled", "global", "unrolled_global", "pow", "pow_global",
    "inline",
)


def _c_headers(preamble_info: Any) -> Any:
    """``stdint.h`` and ``math.h``, which loopy 2025.2's C target leaves out.

    loopy writes ``x**a`` of an integer ``a`` as a call of a function it
    defines with ``int32_t`` in its signature, and a real exponent as a call
    of ``pow``, and includes neither header before them (loopty's
    docs/loopy-notes.md, note 2); a kernel author adds them, as here.
    """
    yield ("00_l2p_headers", "#include <stdint.h>\n#include <math.h>")


def plain_loopy_kernel(
    order: int, target: str, variant: str, wrap: bool = True
) -> Any:
    """The L2P written directly in loopy, as a kernel author would write it.

    One loop over the boxes; in it, the triangle ``a + b <= p`` of the box's
    coefficients reconstructed from the ``2 p + 1`` sumpy stores into a
    temporary (``G_xx = -G_yy``: a coefficient with two more ``x``'s is
    minus the one with two more ``y``'s), and then, for each target of the
    box, the Taylor sum over the triangle. On a device the boxes are work
    groups, as sumpy's own schedule has them (``itgt_box`` on ``g.0``). On
    C, ``wrap`` builds it with ``-fwrapv``, as loopty builds every kernel
    (``loopty.lower.WRAP_FLAG``), which changes the code GCC makes of the
    subscripts' arithmetic.
    """
    import loopy as lp

    if variant not in LOOPY_VARIANTS:
        raise SystemExit(f"--variant is one of {', '.join(LOOPY_VARIANTS)}")
    p = order
    power = "x**(1.0*ia) * y**(1.0*ib)" if variant.startswith("pow") else (
        "x**ia * y**ib"
    )
    if variant == "inline":
        power = (
            "((targets[0, itgt] - centers[0, ibox]) / rscale)**ia"
            " * ((targets[1, itgt] - centers[1, ibox]) / rscale)**ib"
        )
    whole = variant.endswith("global")
    keep = "R[ibox, {}]" if whole else "R[{}]"
    zero = (
        """
        for ibox_z, az, bz
            R[ibox_z, az, bz] = 0 {id=zero}
        end
        """
        if whole
        else ""
    )
    centre = "id=centre, dep=zero" if whole else "id=centre"
    scaled = (
        ""
        if variant == "inline"
        else """<> x = (targets[0, itgt] - centers[0, ibox]) / rscale
                <> y = (targets[1, itgt] - centers[1, ibox]) / rscale"""
    )
    knl = lp.make_kernel(
        [
            "{[ibox]: 0 <= ibox < nboxes}",
            f"{{[a, b]: 0 <= a and 0 <= b and a + b <= {p}}}",
            "{[itgt]: itgt_start <= itgt < itgt_end}",
            f"{{[ia, ib]: 0 <= ia and 0 <= ib and ia + ib <= {p}}}",
            *(
                [f"{{[ibox_z, az, bz]: 0 <= ibox_z < nboxes and 0 <= az, bz <= {p}}}"]
                if whole
                else []
            ),
        ],
        f"""
        {zero}
        for ibox
            <> itgt_start = box_target_starts[ibox]
            <> itgt_end = itgt_start + box_target_counts[ibox]
            for a, b
                if a + b == 0
                    {keep.format("a, b")} = coeffs[ibox, 0] {{{centre}}}
                end
                if a < 2 and a + b >= 1
                    {keep.format("a, b")} = coeffs[ibox, 2*(a + b) - 1 + a] \
                        {{id=stored, dep=centre}}
                end
                if a >= 2
                    {keep.format("a, b")} = -{keep.format("a - 2, b + 2")} \
                        {{id=recurrence, dep=stored}}
                end
            end
            for itgt
                {scaled}
                result[itgt] = {SCALING!r} * sum((ia, ib),
                    {keep.format("ia, ib")} * {power} * inv_fact[ia] * inv_fact[ib]
                    ) {{id=evaluate, dep=recurrence}}
            end
        end
        """,
        [
            lp.GlobalArg("coeffs", np.float64, shape=("nboxes", 2 * p + 1)),
            lp.GlobalArg("centers", np.float64, shape=(2, "nboxes")),
            lp.GlobalArg("box_target_starts,box_target_counts", np.int32,
                         shape=("nboxes",)),
            lp.GlobalArg("targets", np.float64, shape=(2, "ntargets")),
            lp.GlobalArg("result", np.float64, shape=("ntargets",)),
            lp.TemporaryVariable(
                "inv_fact",
                np.float64,
                shape=(p + 1,),
                initializer=np.array([1 / math.factorial(k) for k in range(p + 1)]),
                read_only=True,
                address_space=lp.AddressSpace.GLOBAL,
            ),
            # loopy's C executor allocates no global temporary (the kernel
            # writes through an unset pointer), so on C every box's triangle
            # is an argument, allocated by the caller, as loopty's executor
            # allocates the arrays a program makes (#62)
            lp.GlobalArg("R", np.float64, shape=("nboxes", p + 1, p + 1))
            if whole and target == "c"
            else lp.TemporaryVariable(
                "R",
                np.float64,
                shape=("nboxes", p + 1, p + 1) if whole else (p + 1, p + 1),
                address_space=lp.AddressSpace.GLOBAL if whole else lp.auto,
            ),
            lp.ValueArg("rscale", np.float64),
            lp.ValueArg("nboxes,ntargets", np.int32),
        ],
        name="l2p_plain",
        target=lp.ExecutableCTarget() if target == "c" else lp.PyOpenCLTarget(),
        lang_version=(2018, 2),
        silenced_warnings=["write_race(evaluate)", "write_race(zero)"],
    )
    knl = lp.prioritize_loops(knl, "a,b")
    knl = lp.prioritize_loops(knl, "ia,ib")
    if variant.startswith("unrolled"):
        knl = lp.tag_inames(knl, {"a": "unr", "b": "unr", "ia": "unr", "ib": "unr"})
    if target == "opencl":
        tags = {"ibox": "g.0"}
        if whole:
            tags["ibox_z"] = "g.0"
        knl = lp.tag_inames(knl, tags)
    else:
        knl = lp.register_preamble_generators(knl, [_c_headers])
        if wrap:
            from loopty.lower import WRAP_FLAG

            knl = lp.set_options(knl, build_options=[WRAP_FLAG])
    return knl


def run_loopy(inputs: dict, args: argparse.Namespace) -> tuple[dict, np.ndarray]:
    """The plain loopy kernel on the C target or ``PYOPENCL_CTX``'s device."""
    order = inputs["order"]
    nboxes = inputs["centers"].shape[1]
    ntargets = inputs["targets"].shape[1]
    call_args = {
        "coeffs": np.ascontiguousarray(inputs["coeffs"]),
        "centers": np.ascontiguousarray(inputs["centers"]),
        "box_target_starts": inputs["starts"].astype(np.int32),
        "box_target_counts": inputs["counts"].astype(np.int32),
        "targets": np.ascontiguousarray(inputs["targets"]),
        "result": np.zeros(ntargets),
        "rscale": np.float64(inputs["rscale"]),
        "nboxes": np.int32(nboxes),
        "ntargets": np.int32(ntargets),
    }
    if args.variant.endswith("global") and args.target == "c":
        call_args["R"] = np.zeros((nboxes, order + 1, order + 1))
    record = {
        "implementation": "loopy",
        "variant": args.variant,
        "target": args.target,
        "order": order,
        "nboxes": nboxes,
        "ntargets": ntargets,
        "cold": args.cold,
        "versions": versions("loopy", "islpy", "namedisl", "pymbolic", "pytools",
                             "pyopencl", "numpy", "codepy"),
    }
    import loopy  # noqa: F401 - imported before anything is timed

    queue = opencl_queue() if args.target == "opencl" else None
    start = time.perf_counter()
    knl = plain_loopy_kernel(order, args.target, args.variant, wrap=args.wrap)
    record["prepare"] = time.perf_counter() - start
    if queue is None:
        start = time.perf_counter()
        executor = knl.executor()

        def call() -> Any:
            return executor(**call_args)

        call()
        record["compile"] = time.perf_counter() - start
        times = time_c(call, args.warmup, args.repeat)
        record["toolchain"] = c_toolchain(knl)
        pot = call_args["result"]
    else:
        import pyopencl.array as cla

        device_args = {
            key: cla.to_device(queue, value) if isinstance(value, np.ndarray)
            else value
            for key, value in call_args.items()
        }
        start = time.perf_counter()
        executor = knl.executor(queue.context)

        def call() -> Any:
            event, _out = executor(queue, **device_args)
            return event

        call()
        queue.finish()
        record["compile"] = time.perf_counter() - start
        record["device_kernels"] = device_kernels(knl)
        times = time_opencl(
            call, queue, args.warmup, args.repeat, record["device_kernels"]
        )
        record["device"] = opencl_device(queue)
        pot = device_args["result"].get()
    record["first_call"] = record["prepare"] + record["compile"]
    record["time"] = summary(times)
    record["times"] = times
    return record, pot


# }}}


# {{{ loopty's kernels, from examples/l2p.py


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


def example_at(order: int | None) -> Any:
    """``examples/l2p.py`` written out at ``order``, imported as a module of its own.

    The example fixes its order, ``ORDER = ...``, as sumpy fixes the order of
    each kernel it generates; this is the same file with that one line changed,
    or as it is when ``order`` is ``None``.
    """
    path = repository() / "examples" / "l2p.py"
    source = path.read_text()
    changed, count = re.subn(
        r"^ORDER = \d+",
        "\\g<0>" if order is None else f"ORDER = {order}",
        source,
        count=1,
        flags=re.M,
    )
    if count != 1:
        raise SystemExit(f"{path} sets no ORDER = <number> line")
    name = f"l2p_order_{order or 'as_written'}"
    module = type(sys)(name)
    module.__file__ = str(path)
    sys.modules[name] = module
    exec(compile(changed, str(path), "exec"), module.__dict__)  # noqa: S102
    return module


def loopty_arguments(inputs: dict) -> dict:
    """The inputs as the arguments of the example's program ``l2p``."""
    from loopty import Arr

    counts = inputs["counts"]
    return {
        "coeffs": Arr.from_numpy(np.ascontiguousarray(inputs["coeffs"])),
        "cx": Arr.from_numpy(np.ascontiguousarray(inputs["centers"][0])),
        "cy": Arr.from_numpy(np.ascontiguousarray(inputs["centers"][1])),
        "cnt": Arr.from_numpy(counts),
        "tx": Arr.ragged(counts, values=np.ascontiguousarray(inputs["targets"][0])),
        "ty": Arr.ragged(counts, values=np.ascontiguousarray(inputs["targets"][1])),
        "pot": Arr.ragged(counts),
        "rscale": float(inputs["rscale"]),
    }


#: The schedules of ``examples/l2p.py`` that ``loopty --variant`` names.
VARIANTS = ("fused", "unrolled", "unfused", "windowed", "unrolled_windowed")


def unrolled(schedule: Any) -> Any:
    """``schedule`` with the triangle's loops unrolled: every loop over ``a`` and ``b``.

    Those of the calls, which loop over the box's triangle within the loop
    over the boxes, and the Taylor sum's, which runs inside the loop over a
    box's targets: every loop of a call's statements that is not its
    outermost or the targets', and every loop of a sum. loopy writes each
    out over the triangle's bounding box with the triangle as a guard
    (``unr``), as plain loopy's ``unrolled`` variant has it; the arrays the
    program zeroes keep their loops.
    """
    order = schedule.order
    rank = {name: k for k, name in enumerate(order)}
    entry = schedule.kernel.default_entrypoint
    loops: set[str] = set()
    for insn in entry.instructions:
        if insn.id.endswith("_zeros") or not insn.within_inames:
            continue
        outer = min(insn.within_inames, key=lambda name: rank.get(name, len(rank)))
        loops |= {name for name in insn.within_inames if name not in (outer, "t")}
        loops |= set(insn.reduction_inames())
    return schedule.tag(**dict.fromkeys(sorted(loops), "unr"))


def loopty_schedule(example: Any, variant: str, target: str) -> Any:
    """The example's schedule ``variant`` of ``l2p``, on ``target``.

    ``fused`` is the example's own: each box's triangle, then its targets.
    ``unfused`` is the program as written: every triangle, then every target.
    ``windowed`` is ``fused`` with each box's triangle and table kept in a
    window of the box, the example's ``contracted``. On a device each box is
    a work group (``g.0``), as in sumpy's ``get_optimized_kernel``, which
    keeps its windows in private memory.
    """
    from loopty import Schedule

    if variant == "fused":
        schedule = example.fused
    elif variant == "unrolled":
        schedule = unrolled(example.fused)
    elif variant == "windowed":
        schedule = example.contracted
    elif variant == "unrolled_windowed":
        schedule = unrolled(example.contracted)
    elif variant == "unfused":
        schedule = Schedule(example.l2p, target="c")
    else:
        raise SystemExit(f"--variant is one of {', '.join(VARIANTS)}, not {variant!r}")
    if target == "opencl":
        # every loop over the boxes: the outermost loop of each statement, the
        # calls' and those that zero the arrays the program makes (a window
        # has none)
        order = schedule.order
        entry = schedule.kernel.default_entrypoint
        rank = {name: k for k, name in enumerate(order)}
        boxes = {
            min(insn.within_inames, key=lambda name: rank.get(name, len(rank)))
            for insn in entry.instructions
            if insn.within_inames
        }
        schedule = schedule.retarget("opencl").tag(**dict.fromkeys(boxes, "g.0"))
    return schedule


def run_loopty(inputs: dict, args: argparse.Namespace) -> tuple[dict, np.ndarray]:
    """One of the example's schedules, lowered by loopty, called as its loopy kernel.

    The schedule is the example's own (``--variant`` names it), retargeted to
    ``--target``. The call is the one loopty's executor makes (the arguments
    as it lays them out), timed around loopy's executor alone, so that what
    is timed is the compiled code and not loopty's checks of the call.
    """
    from loopty import executor as ex
    from loopty.isl_reading import declining

    order = inputs["order"]
    nboxes = inputs["centers"].shape[1]
    ntargets = inputs["targets"].shape[1]
    record = {
        "implementation": "loopty",
        "variant": args.variant,
        "target": args.target,
        "order": order,
        "nboxes": nboxes,
        "ntargets": ntargets,
        "cold": args.cold,
        "versions": versions("loopty", "lanky", "loopy", "islpy", "pymbolic",
                             "pytools", "pyopencl", "numpy", "codepy"),
    }
    import loopy  # noqa: F401 - imported before anything is timed

    import loopty.kernel  # noqa: F401 - and loopty's tracer

    queue = opencl_queue() if args.target == "opencl" else None
    # loopy reads loopty's kernels with loopty's readings of their subscripts
    # and guards, which it has inside declining() (docs/loopy-notes.md, note 23)
    with declining():
        start = time.perf_counter()
        example = example_at(order)
        schedule = loopty_schedule(example, args.variant, args.target)
        term, kernel, lowering, target_name = ex._resolve(schedule)
        supplied = loopty_arguments(inputs)
        layouts = ex._declared_layouts(term, lowering, supplied)
        call = ex._call_arguments(term, lowering, (), dict(supplied), layouts)
        if target_name != "opencl":
            kernel, heap = ex._off_the_stack(kernel, term, supplied)
            for name, (shape, dtype) in heap.items():
                call[name] = np.zeros(shape, dtype=dtype)
        # tracing, checking every fact and cast of the file, and lowering
        record["prepare"] = time.perf_counter() - start
        if target_name == "c":
            start = time.perf_counter()
            executor = kernel.executor()

            def run() -> Any:
                return executor(**call)

            run()
            record["compile"] = time.perf_counter() - start
            times = time_c(run, args.warmup, args.repeat)
            record["toolchain"] = c_toolchain(kernel)
            pot = call["pot"]
        else:
            import pyopencl.array as cla

            device_call = {
                key: cla.to_device(queue, value) if isinstance(value, np.ndarray)
                and value.ndim else value
                for key, value in call.items()
            }
            start = time.perf_counter()
            executor = kernel.executor(queue.context)

            def run() -> Any:
                event, _out = executor(queue, **device_call)
                return event

            run()
            queue.finish()
            record["compile"] = time.perf_counter() - start
            record["device_kernels"] = device_kernels(kernel)
            times = time_opencl(
                run, queue, args.warmup, args.repeat, record["device_kernels"]
            )
            record["device"] = opencl_device(queue)
            pot = device_call["pot"].get()
        record["code"] = {
            "fp_contract_off": "FP_CONTRACT OFF" in ex.emit_code(schedule),
        }
    record["first_call"] = record["prepare"] + record["compile"]
    record["time"] = summary(times)
    record["times"] = times
    return record, np.asarray(pot).reshape(-1)


# }}}


# {{{ the example's inputs, and sumpy's potential on them, recorded


def example_inputs() -> dict:
    """``examples/l2p.py``'s inputs for ``l2p``, as :func:`make_inputs` writes them."""
    example = example_at(None)
    data = example.example_inputs()["l2p"]
    counts = data["cnt"].numpy().astype(np.int64)
    tx, ty = data["tx"].numpy(), data["ty"].numpy()
    return {
        "order": np.int64(example.P),
        "level": np.int64(-1),
        "seed": np.int64(-1),
        "coeffs": data["coeffs"].numpy(),
        "centers": np.stack([data["cx"].numpy(), data["cy"].numpy()]),
        "counts": counts,
        "starts": np.concatenate([[0], np.cumsum(counts)[:-1]]).astype(np.int64),
        "targets": np.stack([tx, ty]),
        "rscale": np.float64(data["rscale"]),
        "direct": example.direct(tx, ty),
    }


def record(inputs: dict, pot: str, run: str, out: str) -> None:
    """Write sumpy's potential on the example's inputs, and where it came from."""
    with np.load(pot) as data:
        potential = data["pot"]
    with open(run) as handle:
        made = [json.loads(line) for line in handle if line.strip()][-1]
    if made.get("implementation") != "sumpy":
        raise SystemExit(
            f"{run} records a run of {made.get('implementation')}, not sumpy"
        )
    # the platform and the kind of device; the device's own name stays with the
    # run's record, which is not kept in the repository
    device = made["device"]
    kind = "GPU" if "GPU" in device["device_type"] else "CPU"
    head = {
        "what": (
            "sumpy's E2PFromSingleBox(LinearPDEConformingVolumeTaylorLocalExpansion("
            "LaplaceKernel(2), order), [LaplaceKernel(2)]).get_optimized_kernel(), run "
            "on the inputs of examples/l2p.py by benchmarks/l2p.py"
        ),
        "order": inputs["order"],
        "versions": made["versions"],
        "device": f"{device['platform']}, a {kind}",
        "largest_difference_to_the_direct_sum": float(
            np.max(np.abs(potential - inputs["direct"]))
        ),
    }
    # one row of numbers per line: the file is data, and a change reads as one
    lines = json.dumps(head, indent=1)[:-2].splitlines()
    rows = ",\n".join(f"  {json.dumps(row)}" for row in inputs["coeffs"].tolist())
    text = "\n".join(
        [
            *lines[:-1],
            lines[-1] + ",",
            ' "coeffs": [',
            rows,
            " ],",
            f' "pot": {json.dumps(potential.tolist())}',
            "}",
        ]
    )
    Path(out).write_text(text + "\n")
    print(f"recorded sumpy's potential at {potential.size} targets -> {out}")


# }}}


# {{{ comparing


def compare(inputs: dict, files: list[str]) -> list[dict]:
    """The largest difference between each pair of potentials, and to the direct sum."""
    pots = {}
    for name in files:
        with np.load(name) as data:
            pots[Path(name).stem] = data["pot"]
    scale = float(np.max(np.abs(inputs["direct"])))
    out = []
    names = sorted(pots)
    for k, first in enumerate(names):
        out.append({
            "pair": [first, "direct sum"],
            "max_abs": float(np.max(np.abs(pots[first] - inputs["direct"]))),
            "relative_to_max": float(
                np.max(np.abs(pots[first] - inputs["direct"])) / scale
            ),
        })
        for second in names[k + 1:]:
            gap = np.abs(pots[first] - pots[second])
            out.append({
                "pair": [first, second],
                "max_abs": float(np.max(gap)),
                "relative_to_max": float(np.max(gap) / scale),
            })
    return out


# }}}


def main(argv: list[str] | None = None) -> int:
    """Parse the subcommand and run it."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    commands = parser.add_subparsers(dest="command", required=True)

    make = commands.add_parser("inputs", help="write a tree, targets and coefficients")
    make.add_argument("--order", type=int, required=True)
    make.add_argument("--level", type=int, required=True)
    make.add_argument("--per-box", type=float, default=30.0)
    make.add_argument("--seed", type=int, default=0)
    make.add_argument("--out", required=True)

    for name in ("sumpy", "loopy", "loopty"):
        run = commands.add_parser(name, help=f"time {name}'s L2P on the inputs")
        run.add_argument("--inputs", required=True)
        run.add_argument("--warmup", type=int, default=3)
        run.add_argument("--repeat", type=int, default=10)
        run.add_argument("--cold", action="store_true",
                         help="time the first call in fresh code caches")
        run.add_argument("--save", help="write the potential to this .npz")
        run.add_argument("--out", help="append the record to this JSON-lines file")
        run.add_argument("--label", default="", help="a free-form tag for the record")
        if name == "loopy":
            run.add_argument(
                "--no-wrapv", dest="wrap", action="store_false",
                help="on C, build with loopy's flags alone, without the -fwrapv "
                "loopty builds every kernel with",
            )
        if name != "sumpy":
            run.add_argument("--target", choices=("c", "opencl"), default="c")
            run.add_argument(
                "--variant",
                default="loop" if name == "loopy" else "fused",
                help=f"one of {', '.join(LOOPY_VARIANTS)} for loopy, of "
                f"{', '.join(VARIANTS)} for loopty",
            )

    sample = commands.add_parser("example", help="write examples/l2p.py's inputs")
    sample.add_argument("--out", required=True)

    keep = commands.add_parser("record", help="record sumpy's potential for it")
    keep.add_argument("--inputs", required=True, help="what `example` wrote")
    keep.add_argument("--pot", required=True, help="what `sumpy --save` wrote")
    keep.add_argument("--run", required=True, help="what `sumpy --out` wrote")
    keep.add_argument("--out", default=None, help="default: examples/l2p_sumpy.json")

    check = commands.add_parser("compare", help="compare the potentials of runs")
    check.add_argument("inputs")
    check.add_argument("pots", nargs="+")

    args = parser.parse_args(argv)
    if args.command == "inputs":
        data = make_inputs(args.order, args.level, args.per_box, args.seed)
        np.savez(args.out, **data)
        print(
            f"order {args.order}, {data['centers'].shape[1]} boxes, "
            f"{data['targets'].shape[1]} targets -> {args.out}"
        )
        return 0
    if args.command == "example":
        data = example_inputs()
        np.savez(args.out, **data)
        print(
            f"examples/l2p.py's inputs: order {int(data['order'])}, "
            f"{data['centers'].shape[1]} boxes, {data['targets'].shape[1]} targets "
            f"-> {args.out}"
        )
        return 0
    if args.command == "record":
        out = args.out or str(repository() / "examples" / "l2p_sumpy.json")
        record(load_inputs(args.inputs), args.pot, args.run, out)
        return 0
    if getattr(args, "cold", False):
        cold_caches()
    inputs = load_inputs(args.inputs)
    if args.command == "compare":
        for row in compare(inputs, args.pots):
            print(json.dumps(row))
        return 0
    runner = {"sumpy": run_sumpy, "loopy": run_loopy, "loopty": run_loopty}
    made, pot = runner[args.command](inputs, args)
    made["label"] = args.label
    made["load"] = os.getloadavg()[0]
    made["direct_max_abs"] = float(np.max(np.abs(pot - inputs["direct"])))
    if args.save:
        np.savez(args.save, pot=pot)
    emit(made, args.out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
