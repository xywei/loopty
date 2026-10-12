"""Time the advection right-hand side on hexahedra: loopty, plain loopy, grudge.

The driver for ``examples/advection.py``: grudge's ``WeakAdvectionOperator``
with the upwind flux (grudge PR #418) on a periodic box of ``E**3``
hexahedra, double precision, orders 3 to 6. It reports throughput in the
metric of ``benchmarks/hexahedra.py``, whose clock, copy and launch helpers it
uses:

* **GDOFs/s**, ``E**3 (p + 1)**3 / t``;
* **% of the roof**, ``100 * required bytes / (copy bandwidth * t)``. The
  required bytes are what the right-hand side cannot avoid moving: ``u``
  read and ``rhs`` written, the three velocity terms ``V`` and ``1 / J``
  read at every node, and ``sJ v.n`` at every face node, in double
  precision; the neighbors' indices and the tables are left out.

A variant is a *split*, how the right-hand side is cut into launches, and a
*shape*, how each launch's loops go on the device:

* the splits are ``examples/advection.py``'s ``launches``: three launches;
  libParanumal's volume and surface; the fluxes, then the volume term and
  the lift; and one launch, as grudge runs it;
* the shapes put the element on a work group and its nodes on the work
  items, or a slab of them with ``k`` in sequence, or two elements on a
  group; the fluxes' launch puts a face's nodes on the work items, its six
  faces in sequence or on ``l.2``; and the compiler may contract, or not.

Each of loopty's launches is a schedule loopty checked; beside it is the
same launch in plain loopy, with the same loops on the same axes. The
tables are the exact mass's, grudge #418's: a GLL quadrature mass changes
their values and not the code.

Modes, each a separate process, since grudge needs loopy's main branch:

``python benchmarks/advection.py copy --elements 48 --csv copy.csv``
    The size-matched copy bandwidth, per order.
``python benchmarks/advection.py loopty --elements 48 --csv loopty.csv``
    Every variant, loopty's and plain loopy's, each checked against numpy
    first on elements at the start, the middle and the end of the mesh.
``python benchmarks/advection.py grudge --elements 16 --csv grudge.csv``
    grudge's own right-hand side: eager, and lazy as one kernel (needs grudge).
``python benchmarks/advection.py fields --elements 4 --order 3 --npz fields.npz``
    grudge's right-hand side of a random field on a warped periodic mesh,
    with its geometry and its face maps (needs grudge).
``python benchmarks/advection.py agree --npz fields.npz``
    loopty's program and launches on grudge's fields, against grudge's.
``python benchmarks/advection.py report copy.csv loopty.csv grudge.csv``
    The tables, GDOFs/s and % of the roof.

The device is the one ``PYOPENCL_CTX`` names; nothing chooses one.
"""

from __future__ import annotations

import argparse
import csv
import os
import runpy
import sys
import time
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

HERE = Path(__file__).resolve().parent
DEMO = HERE.parent / "examples" / "advection.py"

#: The clock, the copy, the launches and the CSV of ``benchmarks/hexahedra.py``.
_HEXAHEDRA = runpy.run_path(str(HERE / "hexahedra.py"), run_name="hexahedra_driver")
ITEMSIZE = _HEXAHEDRA["ITEMSIZE"]
FIELDS = _HEXAHEDRA["FIELDS"]
Launch = _HEXAHEDRA["Launch"]
clock = _HEXAHEDRA["clock"]
copy_seconds = _HEXAHEDRA["copy_seconds"]
describe = _HEXAHEDRA["describe"]
fixed = _HEXAHEDRA["fixed"]
from_schedule = _HEXAHEDRA["from_schedule"]
queue_and_context = _HEXAHEDRA["queue_and_context"]
sample = _HEXAHEDRA["sample"]
KernelClock = _HEXAHEDRA["KernelClock"]
_provenance = _HEXAHEDRA["_provenance"]
_rows_writer = _HEXAHEDRA["_rows_writer"]
_said = _HEXAHEDRA["_said"]
_without_contraction = _HEXAHEDRA["_without_contraction"]

#: The operator's name in the CSV rows.
OPERATOR = "advection_rhs"

#: How far a variant may be from numpy, relative to the largest value,
#: before it is refused rather than timed.
AGREEMENT = 1e-12


def load_demo() -> dict[str, Any]:
    """``examples/advection.py`` as a namespace, without running its ``main``."""
    return runpy.run_path(str(DEMO), run_name="advection_demo")


#: The variants that read grudge's geometry where they use it, and not the
#: velocity folded into it once per mesh.
FROM_THE_METRIC = "from the metric"


def required_bytes(order: int, elements: int, variant: str = "") -> int:
    """The bytes the right-hand side cannot avoid moving on ``elements`` hexahedra.

    At every node ``u`` read and ``rhs`` written, and ``1 / J`` read; with the
    velocity folded once per mesh, ``V`` (three) at every node and ``sJ v.n``
    at every face node; from grudge's geometry, the nine metric terms at
    every node and the normal (three) and ``sJ`` at every face node.
    """
    n = order + 1
    if FROM_THE_METRIC in variant:
        return ITEMSIZE * elements * (12 * n**3 + 24 * n**2)
    return ITEMSIZE * elements * (6 * n**3 + 6 * n**2)


# {{{ the variants


@dataclass(frozen=True)
class Shape:
    """How a variant puts each launch's loops on the device.

    A launch over the nodes has the loops ``e, k, j, i``: with ``per_group``
    1 the element is a work group and its nodes the work items, ``k, j, i``
    on ``l.2, l.1, l.0``, or with ``slab`` only ``j, i``, each work item
    running ``k`` in sequence; with more, ``e`` is split and its inner part
    is on ``l.2``, ``k`` in sequence. A launch over the face nodes has ``e,
    f, a, b``: ``a, b`` on ``l.1, l.0``, the faces in sequence or, with
    ``faces``, on ``l.2``. ``contraction`` says whether the compiler may fuse
    a multiply and an add, its default, or not.
    """

    per_group: int = 1
    slab: bool = False
    faces: bool = False
    contraction: bool = True

    @property
    def name(self) -> str:
        words = [
            "element per group"
            if self.per_group == 1
            else f"{self.per_group} elements per group"
        ]
        if self.slab:
            words.append("k in sequence")
        if self.faces:
            words.append("faces on l.2")
        if not self.contraction:
            words.append("no FMA")
        return ", ".join(words)

    def steps(
        self, role: str, loops: Sequence[str]
    ) -> tuple[list[tuple[str, int]], dict[str, str], list[str]]:
        """The splits, the tags and the sequential loops of a launch's ``loops``."""
        e, outer, middle, inner = loops
        splits: list[tuple[str, int]] = []
        if self.per_group > 1:
            splits.append((e, self.per_group))
            tags = {f"{e}_outer": "g.0", f"{e}_inner": "l.2", middle: "l.1"}
            return splits, {**tags, inner: "l.0"}, [outer]
        tags = {e: "g.0", middle: "l.1", inner: "l.0"}
        on_items = self.faces if role == "faces" else not self.slab
        if on_items:
            return splits, {**tags, outer: "l.2"}, []
        return splits, tags, [outer]


#: The shapes the variants try.
SHAPES = (
    Shape(),
    Shape(faces=True),
    Shape(slab=True),
    Shape(per_group=2),
    Shape(contraction=False),
)

#: Each split's launches: the demo's kernel or program, the steps that make it
#: one launch, the call whose loops are placed, and whether they run over the
#: nodes or the face nodes.
SPLITS = {
    "three launches": (
        ("upwind", (), "upwind", "faces"),
        ("volume", (), "volume", "nodes"),
        ("lift", (), "lift", "nodes"),
    ),
    "volume, surface": (
        ("volume", (), "volume", "nodes"),
        ("surface", (("substitute", ("flux",), {}),), "lift", "nodes"),
    ),
    "fluxes, volume": (
        ("upwind", (), "upwind", "faces"),
        (
            "volume_and_lift",
            (("fuse", ("volume", "lift"), {"loops": 4}),),
            "volume",
            "nodes",
        ),
    ),
    "one launch": (
        (
            "folded_rhs",
            (
                ("substitute", ("flux",), {}),
                ("fuse", ("volume", "lift"), {"loops": 4}),
            ),
            "volume",
            "nodes",
        ),
    ),
    f"one launch, {FROM_THE_METRIC}": (
        (
            "advection_rhs",
            (
                ("substitute", ("V",), {}),
                ("substitute", ("vn",), {}),
                ("substitute", ("flux",), {}),
                ("fuse", ("volume", "lift"), {"loops": 4}),
            ),
            "volume",
            "nodes",
        ),
    ),
}


def _loops_of(schedule: Any, call: str) -> tuple[str, ...]:
    """The loops of ``call``'s statement in ``schedule``, outermost first."""
    stmt = next(s for s in schedule.term.stmts if s.id in ("S0", f"{call}.S0"))
    return tuple(stmt.inames[:4])


def _arguments(kernel: Any) -> dict[str, str]:
    """A launch's arrays, each passed as the buffer of its own name."""
    import loopy as lp

    entry = kernel.default_entrypoint
    return {arg.name: arg.name for arg in entry.args if isinstance(arg, lp.ArrayArg)}


def loopty_launches(
    demo: dict[str, Any], split: str, shape: Shape, order: int, ne: int
) -> list[Any]:
    """loopty's launches of ``split`` in ``shape``: each a schedule loopty checked."""
    from loopty import Schedule

    sizes = {"n": order + 1, "ne": ne}
    out = []
    for name, steps, call, role in SPLITS[split]:
        schedule = Schedule(demo[name])
        for method, args, kwargs in steps:
            schedule = getattr(schedule, method)(*args, **kwargs)
        splits, tags, _sequential = shape.steps(role, _loops_of(schedule, call))
        for loop, factor in splits:
            schedule = schedule.split(loop, factor)
        launch = from_schedule(schedule.tag(**tags), {}, **sizes)
        launch.args = _arguments(launch.kernel)
        out.append(launch if shape.contraction else _without_contraction(launch))
    return out


#: The upwind flux at face node ``(a, b)`` of face ``f``, in plain loopy.
_RULES = """
uM(e, f, a, b) := u[e, inside[f, a, b, 0], inside[f, a, b, 1], inside[f, a, b, 2]]
uP(e, f, a, b) := u[nbr[e, f], outside[f, a, b, 0], outside[f, a, b, 1], \
outside[f, a, b, 2]]
upwind_flux(e, f, a, b) := 0.5 * (vn[e, f, a, b] * (uM(e, f, a, b) + uP(e, f, a, b)) \
+ abs(vn[e, f, a, b]) * (uM(e, f, a, b) - uP(e, f, a, b)))
"""

#: The velocity from grudge's geometry where it is read, in plain loopy.
_METRIC_RULES = """
Vm(d, e, k, j, i) := G[d, e, k, j, i] * v[0] + G[3 + d, e, k, j, i] * v[1] \
+ G[6 + d, e, k, j, i] * v[2]
vn_at(e, f, a, b) := sJ[e, f, a, b] * (normal[0, e, f, a, b] * v[0] \
+ normal[1, e, f, a, b] * v[1] + normal[2, e, f, a, b] * v[2])
"""


def _volume(V: str = "V") -> str:
    """The volume term, the velocity read from the array or the rule ``V``."""

    def at(d: int, k: str, j: str, i: str) -> str:
        if V == "V":
            return f"V[{d}, e, {k}, {j}, {i}]"
        return f"{V}({d}, e, {k}, {j}, {i})"

    return (
        "invJ[e, k, j, i] * ("
        f"sum(a, D[i, a] * ({at(0, 'k', 'j', 'a')} * u[e, k, j, a]))"
        f" + sum(b, D[j, b] * ({at(1, 'k', 'b', 'i')} * u[e, k, b, i]))"
        f" + sum(c, D[k, c] * ({at(2, 'c', 'j', 'i')} * u[e, c, j, i])))"
    )


_VOLUME = _volume()


def _lift(flux: str) -> str:
    """The lift of the six faces, each face's flux read from ``flux``.

    ``flux`` is the array the fluxes are stored in, or the rule ``upwind``
    that computes one where it is read.
    """

    def at(face: int, a: str, b: str) -> str:
        if flux == "upwind":
            return f"upwind_flux(e, {face}, {a}, {b})"
        return f"{flux}[e, {face}, {a}, {b}]"

    return (
        "rhs[e, k, j, i] - invJ[e, k, j, i] * ("
        f"(L[0, k] * {at(0, 'j', 'i')} + L[1, k] * {at(1, 'j', 'i')})"
        f" + (L[0, i] * {at(2, 'k', 'j')} + L[1, i] * {at(3, 'k', 'j')})"
        f" + (L[0, j] * {at(4, 'k', 'i')} + L[1, j] * {at(5, 'k', 'i')}))"
    )


_NODES = "0 <= e < ne and 0 <= k, j, i < n"
_FACES = "{[e, f, a, b]: 0 <= e < ne and 0 <= f < 6 and 0 <= a, b < n}"

#: Each launch in plain loopy: its domain, its instructions, and its loops.
PLAIN = {
    "upwind": (
        _FACES,
        _RULES + "flux[e, f, a, b] = upwind_flux(e, f, a, b)",
        ("e", "f", "a", "b"),
    ),
    "volume": (
        f"{{[e, k, j, i, a, b, c]: {_NODES} and 0 <= a, b, c < n}}",
        f"rhs[e, k, j, i] = {_VOLUME}",
        ("e", "k", "j", "i"),
    ),
    "lift": (
        f"{{[e, k, j, i]: {_NODES}}}",
        "rhs[e, k, j, i] = " + _lift("flux"),
        ("e", "k", "j", "i"),
    ),
    "surface": (
        f"{{[e, k, j, i]: {_NODES}}}",
        _RULES + "rhs[e, k, j, i] = " + _lift("upwind"),
        ("e", "k", "j", "i"),
    ),
    "volume_and_lift": (
        f"{{[e, k, j, i, a, b, c]: {_NODES} and 0 <= a, b, c < n}}",
        f"rhs[e, k, j, i] = {_VOLUME} {{id=volume}}\n"
        f"rhs[e, k, j, i] = {_lift('flux')} {{dep=volume}}",
        ("e", "k", "j", "i"),
    ),
    "folded_rhs": (
        f"{{[e, k, j, i, a, b, c]: {_NODES} and 0 <= a, b, c < n}}",
        _RULES + f"rhs[e, k, j, i] = {_VOLUME} {{id=volume}}\n"
        f"rhs[e, k, j, i] = {_lift('upwind')} {{dep=volume}}",
        ("e", "k", "j", "i"),
    ),
    "advection_rhs": (
        f"{{[e, k, j, i, a, b, c]: {_NODES} and 0 <= a, b, c < n}}",
        _METRIC_RULES
        + _RULES.replace("vn[e, f, a, b]", "vn_at(e, f, a, b)")
        + f"rhs[e, k, j, i] = {_volume('Vm')} {{id=volume}}\n"
        f"rhs[e, k, j, i] = {_lift('upwind')} {{dep=volume}}",
        ("e", "k", "j", "i"),
    ),
}

#: Every array a plain launch may take: its shape, and the integers' type.
#: The shapes are given, since an array read through an index read from
#: another has an access range loopy cannot find.
_ARRAYS = {
    "u": ("ne, n, n, n", np.float64),
    "V": ("3, ne, n, n, n", np.float64),
    "invJ": ("ne, n, n, n", np.float64),
    "nbr": ("ne, 6", np.int32),
    "inside": ("6, n, n, 3", np.int32),
    "outside": ("6, n, n, 3", np.int32),
    "vn": ("ne, 6, n, n", np.float64),
    "D": ("n, n", np.float64),
    "L": ("2, n", np.float64),
    "flux": ("ne, 6, n, n", np.float64),
    "rhs": ("ne, n, n, n", np.float64),
    "G": ("9, ne, n, n, n", np.float64),
    "v": ("3", np.float64),
    "normal": ("3, ne, 6, n, n", np.float64),
    "sJ": ("ne, 6, n, n", np.float64),
}


def plain_launches(split: str, shape: Shape, order: int, ne: int) -> list[Any]:
    """The same launches in plain loopy: the same splits, tags and loop order."""
    import loopy as lp

    sizes = {"n": order + 1, "ne": ne}
    out = []
    for name, _steps, _call, role in SPLITS[split]:
        domain, instructions, loops = PLAIN[name]
        reads = sorted(word for word in _ARRAYS if f"{word}[" in instructions)
        kernel = lp.make_kernel(
            "[n, ne] -> " + domain,
            instructions,
            [lp.GlobalArg(arg, _ARRAYS[arg][1], shape=_ARRAYS[arg][0]) for arg in reads]
            + [lp.ValueArg(size, np.int32) for size in ("n", "ne")],
            name=name,
            lang_version=(2018, 2),
        )
        kernel = fixed(kernel, False, **sizes)
        splits, tags, sequential = shape.steps(role, loops)
        for loop, factor in splits:
            kernel = lp.split_iname(kernel, loop, factor)
        kernel = lp.tag_inames(kernel, tags)
        if len(sequential) > 1:
            kernel = lp.prioritize_loops(kernel, ",".join(sequential))
        launch = Launch(kernel, {arg: arg for arg in reads}, loopty=False)
        out.append(launch if shape.contraction else _without_contraction(launch))
    return out


def variants() -> dict[str, tuple[Any, Shape]]:
    """Each variant, loopty's and plain loopy's, by name, with its shape."""
    out: dict[str, tuple[Any, Shape]] = {}
    for split in SPLITS:
        for shape in SHAPES:
            if shape.faces and "fluxes" not in split and "three" not in split:
                continue  # no launch of the split runs over the face nodes
            out[f"loopty, {split}, {shape.name}"] = (
                lambda demo, order, ne, split=split, shape=shape: loopty_launches(
                    demo, split, shape, order, ne
                ),
                shape,
            )
            out[f"loopy, {split}, {shape.name}"] = (
                lambda demo, order, ne, split=split, shape=shape: plain_launches(
                    split, shape, order, ne
                ),
                shape,
            )
    return out


# }}}


# {{{ the arrays, and the right-hand side in numpy


def folded_geometry(host: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
    """``V`` and ``vn`` from grudge's geometry, in numpy: the velocity folded in."""
    nodes = host["u"].shape
    G = host["G"].reshape(3, 3, *nodes)  # [x, d]: J dr_d/dx_x
    return {
        "V": np.einsum("x,xdekji->dekji", host["v"], G),
        "vn": host["sJ"] * np.einsum("x,xefab->efab", host["v"], host["normal"]),
    }


def host_arrays(demo: dict[str, Any], order: int, elements: int) -> dict[str, Any]:
    """The demo's inputs on a periodic box of ``elements**3``, as numpy arrays.

    grudge's geometry, and the velocity folded into it, ``V`` and ``vn``.
    """
    data = demo["inputs"](elements, order)
    out = {name: value.numpy() for name, value in data.items()}
    out.update(folded_geometry(out))
    out["flux"] = np.zeros_like(out["vn"])
    return out


def reference(
    host: dict[str, np.ndarray], elements: np.ndarray, downwind: bool = False
) -> np.ndarray:
    """The right-hand side on ``elements`` in numpy, grudge's ``where`` for the flux.

    ``downwind`` reads each face's flux from the side the flow leaves
    towards, a wrong reading the comparison with grudge has to tell apart.
    """
    u, V, invJ, nbr = (host[name] for name in ("u", "V", "invJ", "nbr"))
    inside, outside, vn, D, L = (
        host[name] for name in ("inside", "outside", "vn", "D", "L")
    )
    here = u[elements]
    weak = (
        np.einsum("ia,ekja->ekji", D, V[0][elements] * here)
        + np.einsum("jb,ekbi->ekji", D, V[1][elements] * here)
        + np.einsum("kc,ecji->ekji", D, V[2][elements] * here)
    )
    inner = here[:, inside[..., 0], inside[..., 1], inside[..., 2]]
    across = u[
        nbr[elements][:, :, None, None],
        outside[None, ..., 0],
        outside[None, ..., 1],
        outside[None, ..., 2],
    ]
    upwind = (vn[elements] > 0) != downwind
    flux = np.where(upwind, inner, across) * vn[elements]
    lifted = (
        np.einsum("k,eji->ekji", L[0], flux[:, 0])
        + np.einsum("k,eji->ekji", L[1], flux[:, 1])
        + np.einsum("i,ekj->ekji", L[0], flux[:, 2])
        + np.einsum("i,ekj->ekji", L[1], flux[:, 3])
        + np.einsum("j,eki->ekji", L[0], flux[:, 4])
        + np.einsum("j,eki->ekji", L[1], flux[:, 5])
    )
    return invJ[elements] * (weak - lifted)


# }}}


# {{{ the modes


def mode_copy(options: argparse.Namespace) -> None:
    """The size-matched copy bandwidth, per order."""
    _context, queue = queue_and_context()
    meta = describe(queue)
    handle, writer = _rows_writer(
        options.csv,
        (
            "operator",
            "order",
            "elements",
            "bytes",
            "median_s",
            "min_s",
            "max_s",
            "bandwidth",
            *meta,
        ),
    )
    for order in options.orders:
        for geometry in ("", FROM_THE_METRIC):
            _copy_row(options, queue, meta, writer, order, geometry)
            handle.flush()
    handle.close()


def _copy_row(
    options: argparse.Namespace,
    queue: Any,
    meta: dict[str, str],
    writer: Any,
    order: int,
    geometry: str,
) -> None:
    """The copy of one order's required bytes, with one geometry."""
    nbytes = required_bytes(order, options.elements**3, geometry)
    # a copy reads and writes, so half the bytes, copied, move them all
    timing = copy_seconds(queue, nbytes // 2, options.warmup, options.reps)
    writer.writerow(
        {
            "operator": OPERATOR + (f", {geometry}" if geometry else ""),
            "order": order,
            "elements": options.elements,
            "bytes": nbytes,
            "median_s": timing.median,
            "min_s": timing.low,
            "max_s": timing.high,
            "bandwidth": nbytes / timing.median,
            **meta,
        }
    )
    print(
        f"copy order {order} {geometry or 'folded'}: "
        f"{nbytes / timing.median / 1e12:.3f} TB/s "
        f"({nbytes / timing.high / 1e12:.3f} to {nbytes / timing.low / 1e12:.3f})",
        flush=True,
    )


def mode_loopty(options: argparse.Namespace) -> None:
    """Every variant, loopty's and plain loopy's, checked and timed."""
    context, queue = queue_and_context()
    meta = describe(queue)
    demo = load_demo()
    handle, writer = _rows_writer(options.csv, FIELDS)
    for order in options.orders:
        # one order's buffers at a time: they are freed when it returns
        _time_order(options, context, queue, demo, order, meta, writer, handle)
        queue.finish()
    handle.close()


def _time_order(
    options: argparse.Namespace,
    context: Any,
    queue: Any,
    demo: dict[str, Any],
    order: int,
    meta: dict[str, str],
    writer: Any,
    handle: Any,
) -> None:
    """Every variant at one order, on that order's buffers."""
    import pyopencl.array as cla

    ne = options.elements**3
    host = host_arrays(demo, order, options.elements)
    arrays = {name: cla.to_device(queue, value) for name, value in host.items()}
    checked = sample(ne)
    expected = reference(host, checked)
    del host
    ndofs = ne * (order + 1) ** 3
    for variant, (build, shape) in variants().items():
        if options.only and not any(word in variant for word in options.only):
            continue
        try:
            launches = build(demo, order, ne)
            calls = [launch.executor(context) for launch in launches]
        except Exception as exc:  # noqa: BLE001 - a variant that does not build
            print(
                f"order {order} {variant}: does not build: {type(exc).__name__}: {exc}",
                flush=True,
            )
            continue

        def call(calls: list[Any] = calls) -> None:
            for run in calls:
                run(queue, arrays)

        arrays["rhs"].fill(0)
        arrays["flux"].fill(0)
        try:
            call()
            queue.finish()
        except Exception as exc:  # noqa: BLE001 - a shape the device refuses
            print(
                f"order {order} {variant}: does not launch: "
                f"{type(exc).__name__}: {exc}",
                flush=True,
            )
            continue
        got = arrays["rhs"].get()[checked]
        gap = float(np.abs(got - expected).max() / np.abs(expected).max())
        if not gap <= AGREEMENT:
            print(
                f"order {order} {variant}: DISAGREES with numpy by {gap:.1e} of "
                "the largest value, so it is not timed",
                flush=True,
            )
            continue
        timing = clock(queue, call, options.warmup, options.reps, options.batch)
        writer.writerow(
            {
                "side": variant.split(",")[0],
                "operator": OPERATOR,
                "variant": variant,
                "order": order,
                "elements": options.elements,
                "ndofs": ndofs,
                "launches": len(launches),
                "median_s": timing.median,
                "min_s": timing.low,
                "max_s": timing.high,
                "kernel_s": timing.kernel,
                "agree": f"{gap:.1e}",
                "contraction": (
                    "compiler default"
                    if shape.contraction
                    else "off: #pragma OPENCL FP_CONTRACT OFF"
                ),
                **meta,
            }
        )
        handle.flush()
        print(
            f"order {order} {variant}: {_said(ndofs, timing)}, agrees to {gap:.1e}",
            flush=True,
        )


def _grudge_context(queue: Any, mode: str) -> Any:
    """grudge's array context: eager, lazy, or lazy as one kernel (MPMS off)."""
    from grudge.array_context import PyOpenCLArrayContext, PytatoPyOpenCLArrayContext
    from pyopencl.tools import ImmediateAllocator, MemoryPool

    allocator = MemoryPool(ImmediateAllocator(queue))
    if mode == "eager":
        return PyOpenCLArrayContext(queue, allocator=allocator)
    if mode == "lazy":
        return PytatoPyOpenCLArrayContext(queue, allocator=allocator)

    class OneKernel(PytatoPyOpenCLArrayContext):
        """The lazy context with nothing materialized: one kernel for the DAG."""

        def transform_dag(self, dag: Any) -> Any:
            import pytato as pt

            return pt.inline_calls(pt.tag_all_calls_to_be_inlined(dag))

    return OneKernel(queue, allocator=allocator)


def _grudge_mesh(elements: int, order: int, warped: bool) -> Any:
    """A periodic box ``[-1, 1]^3`` of hexahedra, warped where asked."""
    from dataclasses import replace

    from meshmode.mesh import TensorProductElementGroup
    from meshmode.mesh.generation import generate_regular_rect_mesh

    mesh = generate_regular_rect_mesh(
        a=(-1.0,) * 3,
        b=(1.0,) * 3,
        nelements_per_axis=(elements,) * 3,
        order=order,
        periodic=(True,) * 3,
        group_cls=TensorProductElementGroup,
    )
    if not warped:
        return mesh

    def warp(x: np.ndarray) -> np.ndarray:
        # vanishes nowhere but moves each face of the box as its opposite,
        # so the periodic faces still match
        s = np.sin(np.pi * x)
        return x + 0.1 * np.stack([s[1] * s[2], s[0] * s[2], s[0] * s[1]])

    return mesh.copy(
        vertices=warp(mesh.vertices),
        groups=[replace(group, nodes=warp(group.nodes)) for group in mesh.groups],
    )


VELOCITY = np.array([0.7, -0.4, 0.3])


def mode_grudge(options: argparse.Namespace) -> None:
    """grudge's own right-hand side, eager, lazy, and lazy as one kernel."""
    _context, queue = queue_and_context()
    meta = describe(queue)
    meta["loopy"] += " (grudge's environment)"
    handle, writer = _rows_writer(options.csv, FIELDS)
    for order in options.orders:
        for mode in options.grudge_modes:
            _time_grudge(options, queue, order, mode, meta, writer, handle)
            queue.finish()
    handle.close()


def _time_grudge(
    options: argparse.Namespace,
    queue: Any,
    order: int,
    mode: str,
    meta: dict[str, str],
    writer: Any,
    handle: Any,
) -> None:
    """grudge's right-hand side in one array context at one order."""
    from grudge.discretization import make_discretization_collection
    from grudge.models.advection import WeakAdvectionOperator

    try:
        actx = _grudge_context(queue, mode)
        mesh = _grudge_mesh(options.elements, order, warped=False)
        dcoll = make_discretization_collection(actx, mesh, order=order)
    except Exception as exc:  # noqa: BLE001 - said, and the next is tried
        print(f"grudge {mode} order {order}: setup failed: {exc}", flush=True)
        return
    operator = WeakAdvectionOperator(dcoll, VELOCITY, flux_type="upwind")
    x = actx.thaw(dcoll.nodes())
    u = actx.freeze(actx.np.sin(np.pi * x[0]) * actx.np.cos(np.pi * x[1]) + x[2])
    ndofs = dcoll.discr_from_dd("vol").ndofs

    def rhs(u: Any) -> Any:
        return operator.operator(0, u)

    run = rhs if mode == "eager" else actx.compile(rhs)

    def call() -> Any:
        return run(actx.thaw(u))

    try:
        call()
        queue.finish()
        with KernelClock() as one:
            call()
        kernels = sorted(one.by_kernel().items(), key=lambda kv: -kv[1][1])
        print(
            f"grudge {mode} order {order}: {sum(c for _, (c, _s) in kernels)} "
            f"launches of {len(kernels)} kernels",
            flush=True,
        )
        for name, (count, seconds) in kernels[:8]:
            print(f"    {name}: {count} launches, {seconds * 1e3:.3f} ms", flush=True)
        timing = clock(queue, call, options.warmup, options.reps, options.batch)
    except Exception as exc:  # noqa: BLE001 - said, and the next is tried
        print(
            f"grudge {mode} order {order}: failed: {type(exc).__name__}: {exc}",
            flush=True,
        )
        return
    variant = {
        "eager": "grudge eager",
        "lazy": "grudge lazy",
        "one": "grudge lazy, one kernel (MPMS off)",
    }[mode]
    writer.writerow(
        {
            "side": "grudge",
            "operator": OPERATOR,
            "variant": variant,
            "order": order,
            "elements": options.elements,
            "ndofs": ndofs,
            "launches": sum(c for _, (c, _s) in kernels),
            "median_s": timing.median,
            "min_s": timing.low,
            "max_s": timing.high,
            "kernel_s": timing.kernel,
            "agree": "",
            "contraction": "compiler default",
            **meta,
            "loopy": meta["loopy"] + f"; grudge {_provenance('grudge')}",
        }
    )
    handle.flush()
    print(f"order {order} {variant}: {_said(ndofs, timing)}", flush=True)


def mode_fields(options: argparse.Namespace) -> None:
    """grudge's right-hand side of a random field on a warped periodic mesh."""
    import grudge.geometry as geo
    from grudge import dof_desc
    from grudge.bilinear_forms import make_stiffness_t_operator
    from grudge.discretization import make_discretization_collection
    from grudge.models.advection import WeakAdvectionOperator

    _context, queue = queue_and_context()
    order = options.order
    actx = _grudge_context(queue, "eager")
    mesh = _grudge_mesh(options.elements, order, warped=True)
    dcoll = make_discretization_collection(actx, mesh, order=order)
    discr = dcoll.discr_from_dd("vol")
    group = discr.groups[0]
    # a random state: the jumps across faces are as large as the values, so
    # the flux's choice of side matters at every face
    rng = np.random.default_rng(options.seed)
    u = discr.zeros(actx)
    u = type(u)(actx, (actx.from_numpy(rng.standard_normal(u[0].shape)),))
    operator = WeakAdvectionOperator(dcoll, VELOCITY, flux_type="upwind")
    rhs = operator.operator(0, u)
    faces = dof_desc.as_dofdesc(dof_desc.FACE_RESTR_ALL)
    normal = geo.normal(actx, dcoll, faces)
    metric = geo.inverse_surface_metric_derivative_mat(
        actx, dcoll, times_area_element=True
    )
    stiffness_t, mass = make_stiffness_t_operator(actx, group, group)[0][:2]
    neighbors = -np.ones((mesh.nelements, 6), dtype=np.int64)
    for adjacency in mesh.facial_adjacency_groups[0]:
        if hasattr(adjacency, "neighbors"):
            neighbors[adjacency.elements, adjacency.element_faces] = adjacency.neighbors
    out = {
        "order": order,
        "velocity": VELOCITY,
        "u": actx.to_numpy(u[0]),
        "rhs": actx.to_numpy(rhs[0]),
        "G": actx.to_numpy(metric[0]),
        "J": actx.to_numpy(geo.area_element(actx, dcoll)[0]),
        "x": np.stack([actx.to_numpy(xi[0]) for xi in actx.thaw(discr.nodes())]),
        "face_x": np.stack(
            [actx.to_numpy(xi[0]) for xi in actx.thaw(dcoll.nodes(faces))]
        ),
        "normal": np.stack([actx.to_numpy(ni[0]) for ni in normal]),
        "sJ": actx.to_numpy(geo.area_element(actx, dcoll, dd=faces)[0]),
        "nbr": neighbors,
        "stiffness_t": actx.to_numpy(stiffness_t),
        "mass": actx.to_numpy(mass),
    }
    np.savez(options.npz, **out)
    shapes = ", ".join(f"{k} {np.shape(v)}" for k, v in out.items())
    print(f"wrote {options.npz}: {shapes}")


def grudge_arrays(data: dict[str, np.ndarray], demo: dict[str, Any]) -> dict:
    """The demo's arrays from grudge's: its geometry, the faces and their maps.

    grudge keeps node ``i + n j + n**2 k`` of an element at that position, so
    a row reshaped is ``[k, j, i]``, and its metric terms as ``[x, d]``, so
    ``3 x + d`` flat. Its faces keep their nodes in the face's own order,
    which is matched to ``inside[f, a, b]`` by the nodes' places.
    """
    order = int(data["order"])
    n = order + 1
    ne = data["u"].shape[0]
    nodes = (ne, n, n, n)
    inside, outside = demo["face_maps"](order)
    x = data["x"].reshape(3, *nodes)
    face_x = data["face_x"].reshape(3, 6, ne, n * n)
    normal = np.zeros((3, ne, 6, n, n))
    sJ = np.zeros((ne, 6, n, n))
    for f in range(6):
        k, j, i = (inside[f, ..., p] for p in range(3))
        here = x[:, :, k, j, i]  # (3, ne, n, n)
        # the face node at each volume node of the face
        gap = np.linalg.norm(
            here.reshape(3, ne, n * n, 1) - face_x[:, f, :, None, :], axis=0
        )
        which = gap.argmin(axis=-1)
        assert np.take_along_axis(gap, which[..., None], -1).max() < 1e-12
        for axis in range(3):
            row = data["normal"].reshape(3, 6, ne, n * n)[axis, f]
            normal[axis, :, f] = np.take_along_axis(row, which, 1).reshape(ne, n, n)
        row = data["sJ"].reshape(6, ne, n * n)[f]
        sJ[:, f] = np.take_along_axis(row, which, 1).reshape(ne, n, n)
    host = {
        "u": data["u"].reshape(nodes),
        "G": data["G"].reshape(9, *nodes),
        "v": np.asarray(data["velocity"], dtype=np.float64),
        "normal": normal,
        "sJ": sJ,
        "invJ": 1 / data["J"].reshape(nodes),
        "nbr": data["nbr"].astype(np.int32),
        "inside": inside,
        "outside": outside,
    }
    host.update(folded_geometry(host))
    return host


def matching_faces(data: dict[str, np.ndarray], host: dict[str, np.ndarray]) -> float:
    """How far ``outside`` lands from ``inside``'s node across each face.

    On a periodic box the two are a period apart along an axis, or at the
    same place; the largest distance left after taking the period out.
    """
    order = int(data["order"])
    n = order + 1
    ne = data["u"].shape[0]
    x = data["x"].reshape(3, ne, n, n, n)
    inside, outside, nbr = host["inside"], host["outside"], host["nbr"]
    here = x[:, :, inside[..., 0], inside[..., 1], inside[..., 2]]
    there = x[
        :,
        nbr[:, :, None, None],
        outside[None, ..., 0],
        outside[None, ..., 1],
        outside[None, ..., 2],
    ]
    apart = here - there
    return float(np.abs(apart - 2.0 * np.round(apart / 2.0)).max())


def mode_agree(options: argparse.Namespace) -> None:
    """loopty's programs and launches on grudge's fields, against grudge's."""
    from loopty import Arr
    from loopty.executor import LoopyExecutor

    data = dict(np.load(options.npz))
    order = int(data["order"])
    n = order + 1
    demo = load_demo()
    stiffness_t, mass = demo["factors"](order)
    print(
        f"order {order}: the demo's S^T and M against grudge's: "
        f"{np.abs(stiffness_t - data['stiffness_t']).max():.1e}, "
        f"{np.abs(mass - data['mass']).max():.1e}"
    )
    host = grudge_arrays(data, demo)
    print(
        "the face maps land on the neighbor's node across each face: "
        f"{matching_faces(data, host):.1e} away"
    )
    D, L = demo["tables"](order)
    host.update(D=D, L=L)
    ne = host["u"].shape[0]
    expected = data["rhs"].reshape(ne, n, n, n)
    scale = np.abs(expected).max()

    def gap(got: Any) -> float:
        return float(np.abs(np.asarray(got) - expected).max() / scale)

    def args(program: Any) -> dict[str, Any]:
        names = program.term.param_names
        out = {k: Arr.from_numpy(np.array(host[k])) for k in names if k != "rhs"}
        out["rhs"] = Arr.zeros((ne, n, n, n))
        return out

    every = np.arange(ne)
    print(f"numpy, grudge's where for the flux: {gap(reference(host, every)):.1e}")
    for name in ("advection_rhs", "folded_rhs"):
        program = demo[name]
        native = args(program)
        program(**native)
        compiled = LoopyExecutor().run(program, **args(program))
        print(
            f"loopty's {name}: native {gap(native['rhs'].numpy()):.1e}, "
            f"compiled on C {gap(compiled['rhs']):.1e}"
        )
    kept = LoopyExecutor().run(demo["contracted"], **args(demo["advection_rhs"]))
    print(
        f"the fused schedule, the fluxes kept per element, on C: {gap(kept['rhs']):.1e}"
    )
    # the comparison tells a wrong reading of grudge's apart: the neighbor's
    # node on this side of the face, the flux from the downwind side, and the
    # metric terms transposed (J dr_x/dx_d for J dr_d/dx_x)
    transposed = folded_geometry(
        {**host, "G": host["G"].reshape(3, 3, ne, n, n, n).swapaxes(0, 1)}
    )["V"]
    for what, result in (
        (
            "the neighbor's node on this side",
            reference({**host, "outside": host["inside"]}, every),
        ),
        ("the downwind side", reference(host, every, downwind=True)),
        ("the metric terms transposed", reference({**host, "V": transposed}, every)),
    ):
        print(f"a wrong reading, {what}, disagrees: {gap(result):.1e}")
    if os.environ.get("PYOPENCL_CTX"):
        import pyopencl.array as cla

        context, queue = queue_and_context()
        arrays = {
            name: cla.to_device(queue, np.ascontiguousarray(value))
            for name, value in host.items()
        }
        arrays["flux"] = cla.zeros(queue, host["vn"].shape, np.float64)
        arrays["rhs"] = cla.zeros(queue, expected.shape, np.float64)
        for split in SPLITS:
            arrays["rhs"].fill(0)
            for launch in loopty_launches(demo, split, Shape(), order, ne):
                launch.executor(context)(queue, arrays)
            print(
                f"loopty's launches, {split}, against grudge: "
                f"{gap(arrays['rhs'].get()):.1e} on {queue.device.name.strip()}"
            )


def mode_report(options: argparse.Namespace) -> None:
    """GDOFs/s and % of the copy roof, per order and variant."""
    rows: list[dict[str, str]] = []
    copies: dict[tuple[str, ...], float] = {}

    def matched(row: dict[str, str]) -> tuple[str, ...]:
        # size-matched: the bytes of the variant's geometry, at its order and
        # mesh, on its device
        named = row["operator"] if "bandwidth" in row else row["variant"]
        geometry = FROM_THE_METRIC if FROM_THE_METRIC in named else ""
        keys = ("order", "elements", "device", "driver")
        return (geometry, *(row[key] for key in keys))

    for path in options.files:
        with open(path, newline="") as handle:
            for row in csv.DictReader(handle):
                if "bandwidth" in row:
                    copies[matched(row)] = float(row["bandwidth"])
                elif row["agree"] and not float(row["agree"]) <= AGREEMENT:
                    print(f"left out, it disagrees with numpy: {row['variant']}")
                else:
                    rows.append(row)

    def setup(row: dict[str, str]) -> tuple[str, ...]:
        return (row["elements"], row["device"], row["driver"], row["loopy"])

    for this in dict.fromkeys(setup(r) for r in rows):
        mine = [r for r in rows if setup(r) == this]
        orders = sorted({r["order"] for r in mine}, key=int)
        print(
            f"\n{OPERATOR}: GDOFs/s (% of the copy roof), {this[0]}**3 hexahedra "
            f"on {this[1]}, loopy {this[3]}\n"
        )
        print("| variant | " + " | ".join(f"order {o}" for o in orders) + " |")
        print("|---|" + "---:|" * len(orders))
        for variant in dict.fromkeys(r["variant"] for r in mine):
            cells = []
            for order in orders:
                found = [
                    r for r in mine if r["variant"] == variant and r["order"] == order
                ]
                if not found:
                    cells.append("")
                    continue
                r = found[0]
                seconds = float(r["median_s"])
                rate = int(r["ndofs"]) / seconds / 1e9
                text = f"{rate:.2f}" if rate >= 0.1 else f"{rate:.2g}"
                bandwidth = copies.get(matched(r))
                if bandwidth:
                    needed = required_bytes(
                        int(order), int(r["elements"]) ** 3, r["variant"]
                    )
                    roof = needed / (bandwidth * seconds)
                    text += (
                        f" ({100 * roof:.0f}%)"
                        if roof >= 0.01
                        else (f" ({100 * roof:.1g}%)")
                    )
                cells.append(text)
            print(f"| {variant} | " + " | ".join(cells) + " |")


def main(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument(
        "mode", choices=["copy", "loopty", "grudge", "fields", "agree", "report"]
    )
    parser.add_argument("files", nargs="*", help="the CSV files to report")
    parser.add_argument("--elements", type=int, default=48, help="hexahedra per axis")
    parser.add_argument("--orders", type=int, nargs="+", default=[3, 4, 5, 6])
    parser.add_argument("--order", type=int, default=3, help="the order of fields")
    parser.add_argument("--seed", type=int, default=0, help="the random field's")
    parser.add_argument(
        "--only",
        nargs="*",
        default=[],
        help="variants whose names contain one of these",
    )
    parser.add_argument("--grudge-modes", nargs="+", default=["eager", "one", "lazy"])
    parser.add_argument("--warmup", type=int, default=3)
    parser.add_argument("--reps", type=int, default=7)
    parser.add_argument("--batch", type=int, default=5)
    parser.add_argument("--csv", default="advection.csv")
    parser.add_argument("--npz", default="fields.npz")
    options = parser.parse_args(argv)
    started = time.perf_counter()
    globals()[f"mode_{options.mode}"](options)
    print(f"({options.mode}: {time.perf_counter() - started:.0f} s)", file=sys.stderr)


if __name__ == "__main__":
    main()
