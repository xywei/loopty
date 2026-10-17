"""sumpy's QBX layer potential on one OpenCL device: loopty, plain loopy, sumpy.

The kernel is pytential's direct QBX self-interaction for the 2-D Laplace
single layer at QBX order 5 (``examples/layer_potential.py``):

- **loopty**: the example's ``single_layer`` under a checked schedule, the
  translation unit loopty lowers it to (``Schedule.kernel``);
- **loopy**: the same arithmetic transcribed by hand into plain loopy, with
  the same transforms, which is what loopty's lowering is compared against;
- **loopy-same**: the expression loopty lowers the kernel to, handed to plain
  loopy, so that only loopty's target and transforms differ;
- **sumpy**: sumpy's own ``LayerPotential`` with ``LineTaylorLocalExpansion``,
  the kernel pytential runs, from ``get_cached_kernel`` with the arguments
  pytential passes it (needs sumpy, which needs a loopy that reads namedisl
  sets, so this one runs in an environment of its own).

Each is called through loopy's executor on the device ``PYOPENCL_CTX`` names,
with the same inputs: the starfish of the example, at the targets and sources
asked for. The tiles:

- ``cpu``: sumpy's ``get_optimized_kernel(is_cpu=True)``, 16 targets a work
  group, a work item each, the sources 256 at a time, the sources' loop
  outside the targets'; this is what pytential runs on a CPU device;
- ``gpu``: the same with 128 targets a work group;
- ``sumpy-gpu`` (sumpy only): ``get_optimized_kernel(is_cpu=False)``, what
  pytential runs on any other device.

Usage::

    python benchmarks/qbx_layer_potential.py inputs --targets 10000 \\
        --sources 10000 --out inputs.npz
    python benchmarks/qbx_layer_potential.py time --impl loopty --tile cpu \\
        --inputs inputs.npz --runs 10 [--contraction off] \\
        [--save out.npy] [--json results.jsonl]
    python benchmarks/qbx_layer_potential.py compare a.npy b.npy
    python benchmarks/qbx_layer_potential.py capture --panels 30 --out calls.npz
    python benchmarks/qbx_layer_potential.py agree --reference calls.npz
    python benchmarks/qbx_layer_potential.py inputs --targets 36 --sources 132 \\
        --out example.npz
    python benchmarks/qbx_layer_potential.py record --inputs example.npz \\
        --out examples/layer_potential_sumpy.json
    python benchmarks/qbx_layer_potential.py write-race

``inputs`` writes the starfish's arrays once, in an environment where loopty
imports, so that every implementation, in whichever environment it runs, is
timed on the same numbers. ``time`` warms up twice and reports the median of
``--runs`` runs, with the fastest and the slowest, by the device's own event
clock and by the wall clock around the call. ``--contraction off`` adds ``#pragma OPENCL
FP_CONTRACT OFF`` to the kernel, whichever it is; ``on`` leaves OpenCL C's
default, which lets the compiler contract. ``--save`` keeps the result, so
that two environments' runs can be compared (``compare``).

``capture`` runs pytential itself, on its starfish identity case at
``--panels`` elements, and keeps what pytential handed sumpy's
``LayerPotential`` on each self-interaction call, and what it returned: S and
D, from inside and from outside. ``agree`` runs loopty's kernels on exactly
those arguments and reports the difference. ``write-race`` asks loopy whether
sumpy's kernel races once its ``write_race(write_lpot*)`` silence is lifted.
``record`` keeps what sumpy returns on saved inputs; at 36 targets and 132
sources those are the example's own, and the recording is the one its
``upstream`` rows, ``sumpy_single_layer`` and ``sumpy_double_layer``, compare
with.

The tables of the demonstration are, on one device and one host, for each
size (``inputs``) and each of ``--contraction on`` and ``off``: ``time`` of
``loopty``, ``loopy``, ``loopy-same`` and ``sumpy`` on one tile (``cpu`` on a
CPU device, ``gpu`` on a GPU), the same again with ``--keep-order`` for the
implementations whose source loops loopy reorders, ``sumpy --tile cpu`` and
``sumpy --tile sumpy-gpu`` for what pytential runs on each kind of device, and
``compare`` of the saved results. Two separate environments take part:
loopty's (the loopy it is pinned to) for ``loopty``, ``loopy`` and
``loopy-same``, and sumpy's (the loopy main sumpy needs) for ``sumpy``,
``capture``, ``record`` and ``write-race``; each record names its versions.

Nothing here names a machine: what the device and the versions are is
printed, and kept with the results.
"""

from __future__ import annotations

import argparse
import contextlib
import importlib.util
import json
import os
import sys
import time
import warnings
from pathlib import Path
from typing import Any

import numpy as np

#: The QBX order, pytential's in its identity test.
ORDER = 5

#: Targets to a work group, for each tile.
GROUP = {"cpu": 16, "gpu": 128}

#: The OpenCL C pragma that turns contraction off.
NO_CONTRACTION = "#pragma OPENCL FP_CONTRACT OFF"


# {{{ inputs


def example() -> Any:
    """``examples/layer_potential.py``, imported from the checkout."""
    path = Path(__file__).resolve().parent.parent / "examples" / "layer_potential.py"
    spec = importlib.util.spec_from_file_location("layer_potential", path)
    module = importlib.util.module_from_spec(spec)
    sys.modules["layer_potential"] = module
    spec.loader.exec_module(module)
    return module


def inputs(targets: int, sources: int) -> dict[str, np.ndarray]:
    """The starfish at about ``targets`` targets and ``sources`` sources.

    9 targets and 33 sources a panel, as the example discretizes it; the
    centers are inside, and the density is the example's ``cos(3 theta)``.
    """
    ex = example()
    curve = ex.starfish(max(1, round(targets / 9)), max(1, round(sources / 33)))
    return {
        "z": curve["z"],
        "c": curve["inside"],
        "y": curve["y"],
        "n": curve["ny"],
        "w": curve["wy"] * ex.density(curve["y"]),
        "radius": curve["size"] / 2,
    }


# }}}


# {{{ the three kernels


def _tiled(knl: Any, target: str, source: str, group: int) -> Any:
    """loopy's transforms for a tile: sumpy's, with ``group`` targets a group."""
    import loopy as lp

    knl = lp.split_iname(knl, target, group, outer_tag="g.0", inner_tag="l.0")
    knl = lp.split_iname(knl, source, 256)
    return lp.prioritize_loops(knl, f"{source}_outer,{target}_inner")


def arithmetic(ex: Any, layer: str, target: str) -> Any:
    """The example's kernel in sumpy's arithmetic: its math steps.

    Each division through the conjugate, and the single layer's ``log|alpha|``
    with no square root, as the example's schedules rewrite them. The kernel
    and its theorems are taken from ``ex``, one import of the example, by the
    layer's name: :func:`example` imports the file afresh at each call, so a
    kernel from one import is never the kernel of another.
    """
    from loopty import Schedule

    if layer == "single":
        kern, by = ex.single_layer, (ex.conjugate_form, ex.log_of_the_modulus)
    else:
        kern, by = ex.double_layer, ex.conjugate_form
    return Schedule(kern, target=target).rewrite(kern.out, by=by)


def loopty_schedule(tile: str, layer: str = "single") -> Any:
    """The example's kernel, in sumpy's arithmetic, under the tile, for OpenCL."""
    ex = example()
    kern = ex.single_layer if layer == "single" else ex.double_layer
    return (
        arithmetic(ex, layer, "opencl")
        .split(kern.t, GROUP[tile])
        .tag({kern.t.outer: "g.0", kern.t.inner: "l.0"})
        .split(kern.s, 256)
        .prioritize(kern.s.outer, kern.t.inner)
    )


def loopty_kernel(tile: str, layer: str = "single") -> Any:
    """loopty's lowering of the example's kernel under the tile, for OpenCL."""
    return loopty_schedule(tile, layer).kernel


#: The arithmetic of loopty's kernel, transcribed into plain loopy: the ratio
#: ``-beta / alpha`` multiplied out by the conjugate, ``log|alpha|`` as half
#: the logarithm of ``|alpha|**2``, and each power as the products loopty
#: writes it in.
PLAIN_LOOPY = """
a(tt, ss) := c[tt] - y[ss]
r2(tt, ss) := real(a(tt, ss) * conj(a(tt, ss)))
u(tt, ss) := -(z[tt] - c[tt]) * conj(a(tt, ss)) * (1 / r2(tt, ss))
u2(tt, ss) := u(tt, ss) * u(tt, ss)
out[t] = scale * sum(s, w[s] * (log(r2(t, s)) / 2 + real(\
    -u(t, s) - u2(t, s) / 2 - u(t, s) * u2(t, s) / 3 - u2(t, s) * u2(t, s) / 4 \
    - u(t, s) * (u2(t, s) * u2(t, s)) / 5)))
"""


def loopy_kernel(tile: str) -> Any:
    """The single layer written in plain loopy, under the same tile."""
    import loopy as lp

    knl = lp.make_kernel(
        "{[t, s]: 0 <= t < nt and 0 <= s < ns}",
        PLAIN_LOOPY,
        [
            lp.GlobalArg("z", np.complex128, shape=("nt",)),
            lp.GlobalArg("c", np.complex128, shape=("nt",)),
            lp.GlobalArg("y", np.complex128, shape=("ns",)),
            lp.GlobalArg("w", np.float64, shape=("ns",)),
            lp.GlobalArg("out", np.float64, shape=("nt",), is_output=True),
            lp.ValueArg("nt", np.int32),
            lp.ValueArg("ns", np.int32),
        ],
        fixed_parameters={"scale": -1 / (2 * np.pi)},
        target=lp.PyOpenCLTarget(),
        name="single_layer",
        lang_version=(2018, 2),
    )
    return _tiled(knl, "t", "s", GROUP[tile])


def loopy_same_kernel(tile: str) -> Any:
    """loopty's own expression, handed to plain loopy, under the same tile.

    The instruction loopty lowers the kernel to, with each complex power by
    a natural number written as the products loopty's OpenCL code writes it
    in, since plain loopy's ``cdouble_powr`` would time something else. What
    is left between this and loopty's kernel is loopty's target, its
    transforms and the order of the source loops they keep.
    """
    import loopy as lp
    from loopy.symbolic import IdentityMapper

    from loopty.isl_reading import declining
    from loopty.lower import _natural_power

    class Products(IdentityMapper):
        def map_power(self, expr: Any, *args: Any) -> Any:
            exponent = expr.exponent
            if isinstance(exponent, int) and 0 < exponent < 100:
                return _natural_power(self.rec(expr.base, *args), exponent)
            return super().map_power(expr, *args)

    with declining():
        lowered = arithmetic(example(), "single", "opencl").kernel
    (insn,) = lowered.default_entrypoint.instructions
    knl = lp.make_kernel(
        "{[t, s]: 0 <= t < nt and 0 <= s < ns}",
        [lp.Assignment(insn.assignee, Products()(insn.expression), id="S0")],
        [
            lp.ValueArg("ns", np.int32),
            lp.ValueArg("nt", np.int32),
            lp.GlobalArg("z", np.complex128, shape=("nt",)),
            lp.GlobalArg("c", np.complex128, shape=("nt",)),
            lp.GlobalArg("y", np.complex128, shape=("ns",)),
            lp.GlobalArg("w", np.float64, shape=("ns",)),
            lp.GlobalArg("out", np.float64, shape=("nt",), is_output=True),
        ],
        target=lp.PyOpenCLTarget(),
        name="single_layer",
        lang_version=(2018, 2),
    )
    return _tiled(knl, "t", "s", GROUP[tile])


def sumpy_layer_potential(layer: str = "single") -> Any:
    """sumpy's ``LayerPotential`` as pytential builds it for direct QBX.

    pytential's ``D`` is an ``IntG`` whose target kernel is the Laplace kernel
    with the source derivative removed and whose source kernel carries it
    (``int_g_vec``), and ``get_lpot_applier`` passes both on as they are.
    """
    from sumpy.expansion.local import LineTaylorLocalExpansion
    from sumpy.kernel import DirectionalSourceDerivative, LaplaceKernel
    from sumpy.qbx import LayerPotential

    laplace = LaplaceKernel(2)
    source = (
        laplace
        if layer == "single"
        else DirectionalSourceDerivative(laplace, "src_derivative_dir")
    )
    return LayerPotential(
        expansion=LineTaylorLocalExpansion(laplace, ORDER),
        target_kernels=(laplace,),
        source_kernels=(source,),
        value_dtypes=np.float64,
    )


def sumpy_kernel(tile: str) -> Any:
    """sumpy's kernel: as pytential runs it on a CPU or another device, or retiled.

    Its arguments are typed, as a call with double-precision arrays types them,
    so that its code can be generated without one.
    """
    import loopy as lp

    names = ("targets", "sources", "center", "expansion_radii", "strength_0")
    dtypes = dict.fromkeys((*names, "result_0"), np.float64)
    return lp.add_dtypes(_sumpy_kernel(tile), dtypes)


def _sumpy_kernel(tile: str) -> Any:
    lpot = sumpy_layer_potential()
    flags = {
        "targets_is_obj_array": True,
        "sources_is_obj_array": True,
        "centers_is_obj_array": True,
    }
    if tile == "cpu":
        return lpot.get_cached_kernel(is_cpu=True, **flags)
    if tile == "sumpy-gpu":
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            return lpot.get_cached_kernel(is_cpu=False, **flags)
    import loopy as lp

    knl = lpot.get_kernel()
    for name in ("targets", "sources", "center"):
        knl = lp.tag_array_axes(knl, name, "sep,C")
    knl = _tiled(knl, "itgt", "isrc", GROUP[tile])
    return lpot._allow_redundant_execution_of_knl_scaling(knl)


def keeping_order(knl: Any, impl: str) -> Any:
    """``knl`` with its source loops nested in the order the sources are summed.

    sumpy asks for the sources' outer loop outside the work items' loop,
    ``prioritize_loops([isrc_outer, itgt_inner])``, which loopy cannot keep
    with the work items on a hardware axis (it says so), and it then nests the
    inner source loop outside the outer one: each target adds its sources 256
    apart. This asks for the outer loop outside the inner one, as loopty's
    lowering does, so that the sum runs in the sources' order.
    """
    import loopy as lp

    if impl == "loopty":
        return knl
    source = "isrc" if impl == "sumpy" else "s"
    return lp.prioritize_loops(knl, f"{source}_outer,{source}_inner")


def without_contraction(knl: Any) -> Any:
    """``knl`` with OpenCL C's contraction turned off by its pragma."""
    entry = knl.default_entrypoint
    preambles = (*entry.preambles, ("00_no_contraction", NO_CONTRACTION))
    return knl.with_kernel(entry.copy(preambles=preambles))


# }}}


# {{{ running and timing


def _device_arrays(queue: Any, data: dict[str, np.ndarray], impl: str) -> dict:
    """The arguments of one implementation's kernel, on the device."""
    import pyopencl.array as cla

    def put(array: np.ndarray) -> Any:
        return cla.to_device(queue, np.ascontiguousarray(array))

    nt = data["z"].size
    if impl == "sumpy":

        def parts(points: np.ndarray) -> list:
            return [put(points.real.copy()), put(points.imag.copy())]

        return {
            "targets": parts(data["z"]),
            "sources": parts(data["y"]),
            "center": parts(data["c"]),
            "expansion_radii": put(data["radius"]),
            "strength_0": put(data["w"]),
            "result_0": cla.zeros(queue, nt, np.float64),
        }
    return {
        "z": put(data["z"]),
        "c": put(data["c"]),
        "y": put(data["y"]),
        "w": put(data["w"]),
        "out": cla.zeros(queue, nt, np.float64),
    }


def _result(args: dict, impl: str) -> np.ndarray:
    return (args["result_0"] if impl == "sumpy" else args["out"]).get()


def _queue() -> Any:
    import pyopencl as cl

    context = cl.create_some_context(interactive=False)
    return cl.CommandQueue(
        context, properties=cl.command_queue_properties.PROFILING_ENABLE
    )


def describe(queue: Any) -> dict[str, Any]:
    """The device, the OpenCL runtime and the Python packages, for the record."""
    import importlib.metadata as metadata

    device = queue.device
    versions = {}
    for name in ("loopy", "islpy", "pyopencl", "pymbolic", "numpy", "sumpy", "loopty"):
        try:
            versions[name] = metadata.version(name)
        except metadata.PackageNotFoundError:
            continue
    return {
        "device_name": device.name,
        "platform": device.platform.name,
        "platform_version": device.platform.version,
        "driver_version": device.driver_version,
        "compute_units": device.max_compute_units,
        "versions": versions,
        "load": os.getloadavg(),
    }


def time_kernel(
    queue: Any, knl: Any, args: dict, runs: int, warmup: int = 2
) -> dict[str, Any]:
    """Median, fastest and slowest of ``runs`` calls, by device events and by wall."""
    executor = knl.executor(queue.context)
    compile_start = time.perf_counter()
    executor(queue, **args)
    queue.finish()
    first = time.perf_counter() - compile_start
    for _ in range(warmup - 1):
        executor(queue, **args)
        queue.finish()
    walls, events = [], []
    for _ in range(runs):
        start = time.perf_counter()
        event, _ = executor(queue, **args)
        queue.finish()
        walls.append(time.perf_counter() - start)
        try:
            events.append((event.profile.end - event.profile.start) * 1e-9)
        except Exception:  # noqa: BLE001 - a device without profiling
            continue

    def summary(values: list[float]) -> dict[str, float] | None:
        if not values:
            return None
        return {
            "median": float(np.median(values)),
            "min": float(np.min(values)),
            "max": float(np.max(values)),
        }

    return {"first_call": first, "wall": summary(walls), "device": summary(events)}


def run_inputs(options: Any) -> int:
    """``inputs``: the starfish's arrays at the size asked for, saved."""
    data = inputs(options.targets, options.sources)
    np.savez(options.out, **data)
    print(f"{data['z'].size} targets, {data['y'].size} sources: {options.out}")
    return 0


def run_time(options: Any) -> int:
    """``time``: one implementation, one tile, on the saved inputs."""
    queue = _queue()
    data = dict(np.load(options.inputs))
    build = {
        "loopty": loopty_kernel,
        "loopy": loopy_kernel,
        "loopy-same": loopy_same_kernel,
        "sumpy": sumpy_kernel,
    }
    knl = build[options.impl](options.tile)
    if options.keep_order:
        knl = keeping_order(knl, options.impl)
    if options.contraction == "off":
        knl = without_contraction(knl)
    args = _device_arrays(queue, data, options.impl)
    with contextlib.ExitStack() as stack:
        if options.impl == "loopty":
            # loopty's targets generate their code with loopty's readings of
            # loopy's isl, which loopy has only inside this.
            from loopty.isl_reading import declining

            stack.enter_context(declining())
        timing = time_kernel(queue, knl, args, options.runs)
    result = _result(args, options.impl)
    record = {
        "impl": options.impl,
        "tile": options.tile,
        "contraction": options.contraction,
        "keep_order": options.keep_order,
        "targets": int(data["z"].size),
        "sources": int(data["y"].size),
        "runs": options.runs,
        **timing,
        **describe(queue),
    }
    pairs = record["targets"] * record["sources"]
    median = (timing["device"] or timing["wall"])["median"]
    record["pairs_per_second"] = pairs / median
    print(json.dumps(record))
    if options.json:
        with open(options.json, "a", encoding="utf-8") as stream:
            stream.write(json.dumps(record) + "\n")
    if options.save:
        np.save(options.save, result)
    return 0


def run_compare(options: Any) -> int:
    """``compare``: the largest difference of saved results, relative to the largest."""
    arrays = [np.load(path) for path in options.results]
    reference = arrays[0]
    scale = np.abs(reference).max()
    for path, array in zip(options.results[1:], arrays[1:], strict=True):
        difference = np.abs(array - reference).max() / scale
        print(f"{path}: {difference:.3e} relative to {options.results[0]}")
    return 0


# }}}


# {{{ agreement with pytential's own calls


def run_capture(options: Any) -> int:
    """``capture``: what pytential hands sumpy's LayerPotential, and what it returns."""
    import pyopencl as cl
    import sumpy.qbx
    from arraycontext import flatten, unflatten
    from meshmode.discretization import Discretization
    from meshmode.discretization.poly_element import (
        InterpolatoryQuadratureGroupFactory,
    )
    from meshmode.mesh.generation import NArmedStarfish, make_curve_mesh
    from pytential import GeometryCollection, bind, sym
    from pytential.array_context import PyOpenCLArrayContext
    from pytential.qbx import QBXLayerPotentialSource
    from pytential.qbx.refinement import refine_geometry_collection
    from sumpy.kernel import LaplaceKernel

    queue = cl.CommandQueue(cl.create_some_context(interactive=False))
    actx = PyOpenCLArrayContext(queue)
    calls: list[dict[str, np.ndarray]] = []
    original = sumpy.qbx.LayerPotential.__call__

    def host(value: Any) -> np.ndarray:
        if isinstance(value, np.ndarray) and value.dtype == object:
            return np.stack([actx.to_numpy(part) for part in value])
        return actx.to_numpy(value)

    def spy(self, actx_, targets, sources, centers, strengths, expansion_radii, **kw):
        out = original(
            self,
            actx_,
            targets=targets,
            sources=sources,
            centers=centers,
            strengths=strengths,
            expansion_radii=expansion_radii,
            **kw,
        )
        record = {
            "targets": host(targets),
            "sources": host(sources),
            "centers": host(centers),
            "strengths": host(strengths[0]),
            "radii": host(expansion_radii),
            "result": host(out[0]),
        }
        # the double layer's one kernel argument is the sources' normals, under
        # the name pytential gives its direction vector
        for value in kw.values():
            record["normals"] = host(value)
        calls.append(record)
        return out

    sumpy.qbx.LayerPotential.__call__ = spy
    target_order = 8
    mesh = make_curve_mesh(
        NArmedStarfish(5, 0.25),
        np.linspace(0, 1, options.panels + 1),
        target_order,
    )
    discr = Discretization(
        actx, mesh, InterpolatoryQuadratureGroupFactory(target_order)
    )
    qbx = QBXLayerPotentialSource(
        discr,
        4 * target_order,
        ORDER,
        fmm_order=False,
        target_association_tolerance=1.0e-1,
    )
    places = refine_geometry_collection(GeometryCollection(qbx))
    density_discr = places.get_discretization(places.auto_source.geometry)
    nodes = actx.to_numpy(flatten(density_discr.nodes(), actx)).reshape(2, -1)
    sigma_host = np.cos(3 * np.arctan2(nodes[1], nodes[0]))
    template = bind(places, sym.normal(2))(actx).as_vector(object)[0]
    sigma = unflatten(template, actx.from_numpy(sigma_host), actx, strict=False)
    laplace = LaplaceKernel(2)
    names = []
    for layer in ("S", "D"):
        for side in (-1, +1):
            op = getattr(sym, layer)(laplace, sym.var("sigma"), qbx_forced_limit=side)
            bind(places, op)(actx, sigma=sigma)
            names.append(f"{layer}{'-' if side < 0 else '+'}")
    sumpy.qbx.LayerPotential.__call__ = original
    saved = {}
    for name, call in zip(names, calls, strict=True):
        for key, value in call.items():
            saved[f"{name}/{key}"] = value
    np.savez(options.out, names=np.array(names), **saved)
    print(
        f"{len(names)} calls on {options.panels} panels: "
        f"{calls[0]['targets'].shape[1]} targets, {calls[0]['sources'].shape[1]} "
        f"sources; saved to {options.out}"
    )
    return 0


def run_record(options: Any) -> int:
    """``record``: sumpy's LayerPotential on saved inputs, called as pytential calls it.

    The example's ``upstream`` rows compare its kernels with this
    recording. sumpy is called through its own ``__call__`` with an array
    context of its own, so it picks its schedule as pytential's calls do
    (``is_cpu`` from the device), with the points as object arrays of their
    coordinates, the strengths as weight times density, and, for the double
    layer, the sources' normals as its ``src_derivative_dir``. What is kept
    besides the results is what they were computed with: sumpy's commit, the
    package versions and the kind of device, and nothing that names a machine.
    """
    import importlib.metadata as metadata

    import pyopencl as cl
    from pytools import obj_array
    from sumpy.array_context import PyOpenCLArrayContext, is_cl_cpu

    data = dict(np.load(options.inputs))
    queue = cl.CommandQueue(cl.create_some_context(interactive=False))
    actx = PyOpenCLArrayContext(queue)

    def points(values: np.ndarray) -> Any:
        return obj_array.new_1d(
            [actx.from_numpy(values.real.copy()), actx.from_numpy(values.imag.copy())]
        )

    recorded: dict[str, Any] = {}
    for name, layer in (("single_layer", "single"), ("double_layer", "double")):
        extra = {} if layer == "single" else {"src_derivative_dir": points(data["n"])}
        (result,) = sumpy_layer_potential(layer)(
            actx,
            targets=points(data["z"]),
            sources=points(data["y"]),
            centers=points(data["c"]),
            strengths=(actx.from_numpy(data["w"]),),
            expansion_radii=actx.from_numpy(data["radius"]),
            **extra,
        )
        recorded[name] = [float(value) for value in actx.to_numpy(result)]

    def commit(name: str) -> str:
        """The commit a package was installed from, else its version."""
        try:
            url = json.loads(metadata.distribution(name).read_text("direct_url.json"))
            return url["vcs_info"]["commit_id"]
        except Exception:  # noqa: BLE001 - not installed from a repository
            return metadata.version(name)

    record = {
        "what": (
            "sumpy's LayerPotential, LineTaylorLocalExpansion of the 2-D Laplace "
            f"kernel at order {ORDER}, on {data['z'].size} targets and "
            f"{data['y'].size} sources, called as pytential calls it"
        ),
        "sumpy": commit("sumpy"),
        "versions": {
            name: commit(name) for name in ("loopy", "islpy", "pyopencl", "numpy")
        },
        "device": "CPU" if is_cl_cpu(actx) else "not a CPU",
        **recorded,
    }
    Path(options.out).write_text(json.dumps(record, indent=1) + "\n", encoding="utf-8")
    print(
        f"sumpy {record['sumpy'][:7]}, {data['z'].size} targets, {data['y'].size} "
        f"sources: {options.out}"
    )
    return 0


def run_agree(options: Any) -> int:
    """``agree``: loopty's kernels on what pytential gave sumpy, against its result."""
    from loopty import Arr
    from loopty.executor import LoopyExecutor

    ex = example()
    data = np.load(options.reference)
    worst = 0.0
    for name in data["names"]:
        name = str(name)

        def get(key: str, name: str = name) -> np.ndarray:
            return data[f"{name}/{key}"]

        def complex_points(points: np.ndarray) -> np.ndarray:
            return points[0] + 1j * points[1]

        args = {
            "z": Arr.from_numpy(complex_points(get("targets"))),
            "c": Arr.from_numpy(complex_points(get("centers"))),
            "y": Arr.from_numpy(complex_points(get("sources"))),
            "w": Arr.from_numpy(get("strengths")),
            "out": Arr.zeros(get("targets").shape[1]),
        }
        layer = "double" if name.startswith("D") else "single"
        if layer == "double":
            args["n"] = Arr.from_numpy(complex_points(get("normals")))
        if options.target == "opencl":
            schedule = loopty_schedule("cpu", layer)
        else:
            schedule = ex.tiled_double if layer == "double" else ex.tiled
        mine = LoopyExecutor().run(schedule, **args)["out"]
        theirs = get("result")
        difference = float(np.abs(mine - theirs).max() / np.abs(theirs).max())
        worst = max(worst, difference)
        print(
            f"{name}: {theirs.size} targets, {get('sources').shape[1]} sources: "
            f"loopty ({options.target}) against sumpy, largest difference "
            f"{difference:.2e} of the largest value"
        )
    print(f"worst {worst:.2e}")
    return 0 if worst < options.tolerance else 1


# }}}


def _races(knl: Any) -> list[str]:
    """The ``write_race`` warnings loopy gives while generating ``knl``'s code."""
    import loopy as lp

    lp.set_caching_enabled(False)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        lp.generate_code_v2(knl)
    return [str(w.message) for w in caught if "write_race" in str(w.message)]


def run_write_race(options: Any) -> int:
    """``write-race``: does loopy warn of a race on sumpy's kernel, unsilenced?

    A control first: a kernel whose work items all write one cell, which
    loopy has to warn about, so that a quiet answer is loopy's and not the
    probe's.
    """
    import loopy as lp

    control = lp.make_kernel(
        "{[i]: 0 <= i < n}",
        "out[0] = x[i]",
        [
            lp.GlobalArg("x", np.float64, shape=("n",)),
            lp.GlobalArg("out", np.float64, shape=(1,)),
            ...,
        ],
        target=lp.PyOpenCLTarget(),
        lang_version=(2018, 2),
    )
    control = lp.split_iname(control, "i", 16, outer_tag="g.0", inner_tag="l.0")
    warned = len(_races(control))
    print(f"control, every work item writes out[0]: loopy warns {warned} time(s)")
    for tile in ("cpu", "sumpy-gpu"):
        knl = sumpy_kernel(tile)
        entry = knl.default_entrypoint
        kept = [
            warning
            for warning in entry.silenced_warnings
            if not str(warning).startswith("write_race")
        ]
        races = _races(knl.with_kernel(entry.copy(silenced_warnings=kept)))
        print(
            f"sumpy, {tile} tile: silenced {list(entry.silenced_warnings)}; lifted, "
            f"loopy warns {len(races)} time(s)"
        )
        for race in races:
            print(f"  {race}")
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    verbs = parser.add_subparsers(dest="verb", required=True)
    made = verbs.add_parser("inputs")
    made.add_argument("--targets", type=int, default=10_000)
    made.add_argument("--sources", type=int, default=10_000)
    made.add_argument("--out", required=True)
    timing = verbs.add_parser("time")
    timing.add_argument(
        "--impl", choices=("loopty", "loopy", "loopy-same", "sumpy"), required=True
    )
    timing.add_argument("--tile", choices=("cpu", "gpu", "sumpy-gpu"), default="cpu")
    timing.add_argument("--inputs", required=True)
    timing.add_argument("--runs", type=int, default=10)
    timing.add_argument("--contraction", choices=("on", "off"), default="on")
    timing.add_argument("--keep-order", action="store_true")
    timing.add_argument("--json")
    timing.add_argument("--save")
    compare = verbs.add_parser("compare")
    compare.add_argument("results", nargs="+")
    capture = verbs.add_parser("capture")
    capture.add_argument("--panels", type=int, default=30)
    capture.add_argument("--out", required=True)
    agree = verbs.add_parser("agree")
    agree.add_argument("--reference", required=True)
    agree.add_argument("--target", choices=("c", "opencl"), default="c")
    agree.add_argument("--tolerance", type=float, default=1e-12)
    record = verbs.add_parser("record")
    record.add_argument("--inputs", required=True)
    record.add_argument("--out", required=True)
    verbs.add_parser("write-race")
    options = parser.parse_args(argv)
    if options.verb == "time" and options.tile not in GROUP and options.impl != "sumpy":
        parser.error(
            f"--tile {options.tile} is sumpy's own schedule for a device that is no "
            f"CPU, which only --impl sumpy has; for {options.impl}, the tiles are "
            f"{', '.join(GROUP)}"
        )
    run = {
        "inputs": run_inputs,
        "time": run_time,
        "compare": run_compare,
        "capture": run_capture,
        "agree": run_agree,
        "record": run_record,
        "write-race": run_write_race,
    }
    return run[options.verb](options)


if __name__ == "__main__":
    raise SystemExit(main())
