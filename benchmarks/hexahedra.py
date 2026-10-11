"""Time volume operators on hexahedra: loopty's checked launches, plain loopy, grudge.

The driver for ``examples/hexahedra.py``: grudge's weak gradient and weak
divergence (grudge PR #418) on a box of ``E**3`` hexahedra, double precision,
at orders 3 to 6. It reports throughput in the metric DG codes use:

* **GDOFs/s**, spatial degrees of freedom per second, ``E**3 (p + 1)**3 / t``;
* **% of the DG roof**, ``100 * required bytes / (copy bandwidth * t)``: the
  bytes an operator cannot avoid moving (each input and metric term read
  once, each output written once, in double precision) over the bandwidth of
  a device-to-device copy of the same number of bytes on the same device.
  100% is an empirical reference, not a bound: an operator that keeps some of
  its inputs in a cache can exceed it.

Each operator is in :data:`OPERATORS`, with what it reads and writes per
degree of freedom and its variants, so another operator on the same mesh (an
advection right-hand side, say) is one entry more.

Modes, each a separate process, since grudge and loopty may live in different
environments (grudge #418 needs loopy's main branch):

``python benchmarks/hexahedra.py copy --elements 48 --orders 3 4 5 6 --csv copy.csv``
    The size-matched copy bandwidth, per operator and order.
``python benchmarks/hexahedra.py loopty --elements 48 --csv loopty.csv``
    Every variant of every operator, loopty's launches beside the same
    kernels written in plain loopy, each checked against numpy first.
``python benchmarks/hexahedra.py grudge --elements 48 --csv grudge.csv``
    grudge's own operators, eager and lazy (needs grudge).
``python benchmarks/hexahedra.py fields --elements 4 --order 3 --npz fields.npz``
    grudge's metric terms, fields and operators on a warped mesh (needs grudge).
``python benchmarks/hexahedra.py agree --npz fields.npz``
    loopty's programs and launches on grudge's fields, against grudge's results.
``python benchmarks/hexahedra.py report copy.csv loopty.csv grudge.csv``
    The tables, GDOFs/s and % of the roof.

The device is the one ``PYOPENCL_CTX`` names; nothing chooses one. A time is
the median over ``--reps`` repetitions of a batch of ``--batch`` calls after
``--warmup`` calls, measured between two markers on the device's queue, and
the minimum and the maximum go beside it. Each row records the device, the
driver and the versions it ran with; nothing here names the machine.
"""

from __future__ import annotations

import argparse
import csv
import os
import statistics
import sys
import time
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

HERE = Path(__file__).resolve().parent
DEMO = HERE.parent / "examples" / "hexahedra.py"

#: Bytes per value: double precision throughout.
ITEMSIZE = 8


# {{{ the operators


@dataclass(frozen=True)
class Operator:
    """What an operator reads and writes per degree of freedom, and its variants.

    ``inputs`` and ``outputs`` count fields, ``metric`` the metric terms read
    per node; :func:`required_bytes` is their sum in bytes. ``variants`` maps
    a variant's name to a function ``(demo, order) -> list of launches``
    (see :class:`Launch`).
    """

    name: str
    inputs: int
    outputs: int
    metric: int
    variants: dict[str, Callable[..., list[Launch]]] = field(default_factory=dict)


def required_bytes(op: Operator, ndofs: int) -> int:
    """The bytes ``op`` cannot avoid moving on ``ndofs`` degrees of freedom."""
    return ITEMSIZE * ndofs * (op.inputs + op.outputs + op.metric)


# }}}


# {{{ the device and the clock


def queue_and_context() -> tuple[Any, Any]:
    """A profiling queue on the device ``PYOPENCL_CTX`` names."""
    import pyopencl as cl

    if "PYOPENCL_CTX" not in os.environ:
        raise SystemExit("set PYOPENCL_CTX to the platform:device to time on")
    context = cl.create_some_context(interactive=False)
    queue = cl.CommandQueue(
        context, properties=cl.command_queue_properties.PROFILING_ENABLE
    )
    return context, queue


def describe(queue: Any) -> dict[str, str]:
    """The device and the software a row was measured with."""
    import islpy
    import loopy
    import pyopencl as cl

    device = queue.device
    return {
        "device": device.name.strip(),
        "platform": device.platform.name.strip(),
        "platform_version": device.platform.version.strip(),
        "driver": device.driver_version.strip(),
        "loopy": ".".join(map(str, loopy.VERSION)),
        "islpy": islpy.__version__,
        "pyopencl": ".".join(map(str, cl.VERSION)),
        "numpy": np.__version__,
        "python": sys.version.split()[0],
    }


@dataclass
class Timing:
    """Seconds per call: the median of the repetitions, and the extremes."""

    median: float
    low: float
    high: float
    kernel: float | None = None

    @staticmethod
    def of(spans: Sequence[float], kernel: float | None = None) -> Timing:
        return Timing(statistics.median(spans), min(spans), max(spans), kernel)


class KernelClock:
    """The device time of every kernel launched while it is active.

    pyopencl's generated invokers, loopy's among them, enqueue through
    ``pyopencl._cl.enqueue_nd_range_kernel``; this wraps it and keeps each
    event, so that the time the device spent in kernels can be told apart
    from the time it waited for the host between them.
    """

    def __init__(self) -> None:
        self.events: list[Any] = []
        self.names: list[str] = []

    def __enter__(self) -> KernelClock:
        import pyopencl
        import pyopencl._cl as _cl

        # pyopencl's invokers call _cl's, loopy's host code pyopencl's
        self._own = _cl.enqueue_nd_range_kernel
        own, events, names = self._own, self.events, self.names

        def recorded(*args: Any, **kwargs: Any) -> Any:
            event = own(*args, **kwargs)
            events.append(event)
            names.append(getattr(args[1], "function_name", "?"))
            return event

        _cl.enqueue_nd_range_kernel = recorded
        pyopencl.enqueue_nd_range_kernel = recorded
        return self

    def __exit__(self, *exc: object) -> None:
        import pyopencl
        import pyopencl._cl as _cl

        _cl.enqueue_nd_range_kernel = self._own
        pyopencl.enqueue_nd_range_kernel = self._own

    def seconds(self) -> float:
        for event in self.events:
            event.wait()
        return 1e-9 * sum(e.profile.end - e.profile.start for e in self.events)

    def by_kernel(self) -> dict[str, tuple[int, float]]:
        """Launches and device seconds per kernel name."""
        out: dict[str, tuple[int, float]] = {}
        for name, event in zip(self.names, self.events):
            event.wait()
            count, seconds = out.get(name, (0, 0.0))
            out[name] = (
                count + 1,
                seconds + 1e-9 * (event.profile.end - event.profile.start),
            )
        return out


def clock(
    queue: Any, call: Callable[[], Any], warmup: int, reps: int, batch: int
) -> Timing:
    """Seconds per ``call()`` on the device, between two markers, and in kernels."""
    import pyopencl as cl

    for _ in range(warmup):
        call()
    queue.finish()
    spans, kernels = [], []
    for _ in range(reps):
        with KernelClock() as kernel_clock:
            start = cl.enqueue_marker(queue)
            for _ in range(batch):
                call()
            stop = cl.enqueue_marker(queue)
            stop.wait()
        spans.append(1e-9 * (stop.profile.end - start.profile.start) / batch)
        kernels.append(kernel_clock.seconds() / batch)
    return Timing.of(spans, statistics.median(kernels))


def copy_seconds(queue: Any, nbytes: int, warmup: int, reps: int) -> Timing:
    """A device-to-device copy of ``nbytes``: the time of one, by its event."""
    import pyopencl as cl

    count = nbytes // ITEMSIZE
    source = cl.Buffer(queue.context, cl.mem_flags.READ_WRITE, count * ITEMSIZE)
    target = cl.Buffer(queue.context, cl.mem_flags.READ_WRITE, count * ITEMSIZE)
    cl.enqueue_fill_buffer(queue, source, np.float64(1.0), 0, count * ITEMSIZE)
    for _ in range(warmup):
        cl.enqueue_copy(queue, target, source)
    queue.finish()
    spans = []
    for _ in range(reps):
        event = cl.enqueue_copy(queue, target, source)
        event.wait()
        spans.append(1e-9 * (event.profile.end - event.profile.start))
    return Timing.of(spans)


# }}}


# {{{ launches: loopty's schedules and plain loopy's kernels


@dataclass
class Launch:
    """One kernel launch of a variant: its loopy kernel and how to call it.

    ``kernel`` is a loopy translation unit with every size fixed; ``args``
    maps the kernel's argument names to the names of the arrays a variant
    passes (:func:`buffers`). ``loopty`` says whether loopy must generate it
    inside :func:`loopty.isl_reading.declining`, as loopty's kernels need.
    """

    kernel: Any
    args: dict[str, str]
    loopty: bool
    facts: tuple[Any, ...] = ()

    def executor(self, context: Any) -> Callable[..., Any]:
        """The kernel compiled for ``context``, called with the variant's arrays."""
        if not self.loopty:
            run = self.kernel.executor(context)
            return lambda queue, arrays: run(
                queue, **{a: arrays[b] for a, b in self.args.items()}
            )
        from loopty.isl_reading import declining

        with declining():
            run = self.kernel.executor(context)

        def call(queue: Any, arrays: dict[str, Any]) -> Any:
            with declining():
                return run(queue, **{a: arrays[b] for a, b in self.args.items()})

        return call


def fixed(kernel: Any, loopty: bool, **sizes: int) -> Any:
    """``kernel`` with ``sizes`` fixed, as loopy's ``fix_parameters`` does.

    Both sides of a comparison get the same: the sizes are numbers in the
    code, as in grudge's kernels, so loops have known lengths and work groups
    known shapes.
    """
    import loopy as lp

    names = set(kernel.default_entrypoint.all_params())
    given = {name: value for name, value in sizes.items() if name in names}
    if not loopty:
        return lp.fix_parameters(kernel, **given)
    from loopty.isl_reading import declining

    with declining():
        return lp.fix_parameters(kernel, **given)


def from_schedule(schedule: Any, args: dict[str, str], **sizes: int) -> Launch:
    """A loopty schedule, retargeted onto OpenCL, as a launch with ``sizes`` fixed."""
    on_device = schedule.retarget("opencl")
    built, reason = on_device.buildable
    if not built:
        raise RuntimeError(f"{on_device!r} is not buildable: {reason}")
    facts = tuple(on_device.facts())
    bad = [f for f in facts if f.status.value == "refuted"]
    if bad:
        raise RuntimeError(f"{on_device!r} has refuted facts: {bad}")
    kernel = fixed(on_device.kernel, True, **sizes)
    return Launch(kernel, args, loopty=True, facts=facts)


# }}}


# {{{ the weak gradient and divergence


def load_demo() -> dict[str, Any]:
    """``examples/hexahedra.py`` as a namespace, without running its ``main``."""
    import runpy

    return runpy.run_path(str(DEMO), run_name="hexahedra_demo")


@dataclass(frozen=True)
class Shape:
    """How a variant puts an operator's launches on the device.

    Every launch has the element's nodes ``k, j, i`` and its fields ``q``
    (the gradient's last launch: its components ``x``). With ``per_group``
    1, the element is a work group, ``e`` on ``g.0``, and its nodes the work
    items, ``k, j, i`` on ``l.2, l.1, l.0``; with more, ``e`` is split, its
    inner part on ``l.2``, and each work item runs its slabs ``k`` in
    sequence. ``fields`` runs the fields in sequence on each work item, or
    puts them on ``g.1``, or interleaves them (``ilp``). ``contraction``
    says whether the compiler may fuse a multiply and an add, its default,
    or not: ``#pragma OPENCL FP_CONTRACT OFF`` on both sides.
    """

    per_group: int = 1
    fields: str = "sequence"
    contraction: bool = True

    @property
    def name(self) -> str:
        words = [
            "element per group"
            if self.per_group == 1
            else f"{self.per_group} elements per group"
        ]
        if self.fields != "sequence":
            words.append(
                "fields on g.1"
                if self.fields == "groups"
                else "fields three at a time (ilp)"
            )
        if not self.contraction:
            words.append("no FMA")
        return ", ".join(words)

    def steps(
        self, loops: dict[str, str]
    ) -> tuple[list[tuple[str, int]], dict[str, str], list[str]]:
        """The splits, the tags and the order of the sequential loops, for ``loops``.

        ``loops`` names a launch's loops by role: ``e``, ``k``, ``j``, ``i``
        and, when it has one, ``q``.
        """
        e, k, j, i = (loops[x] for x in "ekji")
        q = loops.get("q")
        splits: list[tuple[str, int]] = []
        if self.per_group == 1:
            tags = {e: "g.0", k: "l.2", j: "l.1", i: "l.0"}
            sequential = []
        else:
            splits.append((e, self.per_group))
            tags = {f"{e}_outer": "g.0", f"{e}_inner": "l.2", j: "l.1", i: "l.0"}
            sequential = [k]
        if q is not None:
            if self.fields == "groups":
                tags[q] = "g.1"
            elif self.fields == "ilp":
                # three at a time: an ilp loop needs a length that is a number
                splits.append((q, 3))
                tags[f"{q}_inner"] = "ilp"
                sequential = [f"{q}_outer", *sequential]
            else:
                sequential = [q, *sequential]
        return splits, tags, sequential


#: The shapes the variants try.
SHAPES = (
    Shape(),
    Shape(per_group=2),
    Shape(per_group=4),
    Shape(fields="groups"),
    Shape(per_group=2, fields="groups"),
    Shape(fields="ilp"),
    Shape(contraction=False),
)


def _placed(schedule: Any, shape: Shape, loops: dict[str, str]) -> Any:
    """``schedule`` with ``shape``'s splits and tags, each step checked by loopty."""
    splits, tags, _sequential = shape.steps(loops)
    for loop, factor in splits:
        schedule = schedule.split(loop, factor)
    return schedule.tag(**tags)


def _without_contraction(launch: Launch) -> Launch:
    """``launch`` with ``#pragma OPENCL FP_CONTRACT OFF`` before its code."""
    entry = launch.kernel.default_entrypoint
    pragma = ("04_no_fp_contract", "#pragma OPENCL FP_CONTRACT OFF")
    kernel = launch.kernel.with_kernel(entry.copy(preambles=(*entry.preambles, pragma)))
    return Launch(kernel, launch.args, launch.loopty, launch.facts)


#: Each operator's launches: the demo's kernel or program, what is substituted
#: in it, its loops by role, and the arrays it takes, by the variants' names.
LAUNCHES = {
    "weak_grad": (
        (
            "metric_along_r",
            "w",
            {"e": "e_0", "k": "k_0", "j": "j_0", "i": "i_0", "q": "q_0"},
            {"G": "G", "u": "u", "Fr": "Fr9", "v": "v"},
        ),
        (
            "along_s",
            None,
            {"e": "e", "k": "k", "j": "j", "i": "i", "q": "q"},
            {"Fs": "Fs9", "v": "v", "vs": "vs"},
        ),
        (
            "gradient_along_t",
            None,
            {"e": "e", "k": "k", "j": "j", "i": "i", "q": "x"},
            {"Ft": "Ft9", "vs": "vs", "grad": "grad"},
        ),
    ),
    "weak_div": (
        (
            "contravariant_along_r",
            "w",
            {"e": "e_0", "k": "k_0", "j": "j_0", "i": "i_0", "q": "q"},
            {"G": "G", "f": "f", "Fr": "Fr3", "v": "v3"},
        ),
        (
            "along_s",
            None,
            {"e": "e", "k": "k", "j": "j", "i": "i", "q": "q"},
            {"Fs": "Fs3", "v": "v3", "vs": "vs3"},
        ),
        (
            "divergence_along_t",
            None,
            {"e": "e", "k": "k", "j": "j", "i": "i"},
            {"Ft": "Ft3", "vs": "vs3", "div": "div"},
        ),
    ),
}


def loopty_launches(
    demo: dict[str, Any], op: str, shape: Shape, order: int, ne: int
) -> list[Launch]:
    """loopty's launches of ``op`` in ``shape``: each a schedule loopty checked."""
    from loopty import Schedule

    sizes = {"n": order + 1, "ne": ne, "nq": 9 if op == "weak_grad" else 3}
    out = []
    for name, substituted, loops, args in LAUNCHES[op]:
        schedule = Schedule(demo[name])
        if substituted is not None:
            schedule = schedule.substitute(substituted)
        launch = from_schedule(_placed(schedule, shape, loops), args, **sizes)
        out.append(launch if shape.contraction else _without_contraction(launch))
    return out


_BOX = "0 <= e < ne and 0 <= k, j, i < n"

#: The same launches written in plain loopy, with the loops named by role:
#: their domains and instructions, as a loopy user writes them. A sum over
#: three directions is written out, three sums of three loops.
PLAIN = {
    "weak_grad": (
        (
            f"{{[q, e, k, j, i, a]: 0 <= q < nq and {_BOX} and 0 <= a < n}}",
            "v[q, e, k, j, i] = sum(a, Fr[q, i, a] * G[q, e, k, j, a] * u[e, k, j, a])",
        ),
        (
            f"{{[q, e, k, j, i, b]: 0 <= q < nq and {_BOX} and 0 <= b < n}}",
            "vs[q, e, k, j, i] = sum(b, Fs[q, j, b] * v[q, e, k, b, i])",
        ),
        (
            f"{{[q, e, k, j, i, c, c1, c2]: 0 <= q < 3 and {_BOX}"
            " and 0 <= c, c1, c2 < n}",
            "grad[q, e, k, j, i] = sum(c, Ft[3*q, k, c] * vs[3*q, e, c, j, i])"
            " + sum(c1, Ft[3*q + 1, k, c1] * vs[3*q + 1, e, c1, j, i])"
            " + sum(c2, Ft[3*q + 2, k, c2] * vs[3*q + 2, e, c2, j, i])",
        ),
    ),
    "weak_div": (
        (
            f"{{[q, e, k, j, i, a]: 0 <= q < nq and {_BOX} and 0 <= a < n}}",
            "v[q, e, k, j, i] = sum(a, Fr[q, i, a]"
            " * (G[q, e, k, j, a] * f[0, e, k, j, a]"
            " + G[3 + q, e, k, j, a] * f[1, e, k, j, a]"
            " + G[6 + q, e, k, j, a] * f[2, e, k, j, a]))",
        ),
        (
            f"{{[q, e, k, j, i, b]: 0 <= q < nq and {_BOX} and 0 <= b < n}}",
            "vs[q, e, k, j, i] = sum(b, Fs[q, j, b] * v[q, e, k, b, i])",
        ),
        (
            f"{{[e, k, j, i, c, c1, c2]: {_BOX} and 0 <= c, c1, c2 < n}}",
            "div[e, k, j, i] = sum(c, Ft[0, k, c] * vs[0, e, c, j, i])"
            " + sum(c1, Ft[1, k, c1] * vs[1, e, c1, j, i])"
            " + sum(c2, Ft[2, k, c2] * vs[2, e, c2, j, i])",
        ),
    ),
}


def _nested(target: str, iname: str, factor: str, first: str, second: str) -> list:
    """``target = sum(iname, factor * (first * second))``, the product kept nested.

    loopy's parser, and its substitution rules, flatten ``a * (b * c)`` into
    ``a * b * c``, which rounds differently and is other code; an assignment
    built from pymbolic's expressions keeps the product as written.
    """
    import loopy as lp
    from loopy.symbolic import Reduction
    from pymbolic import parse
    from pymbolic.primitives import Product

    body = Product((parse(factor), Product((parse(first), parse(second)))))
    return [lp.Assignment(parse(target), Reduction("sum", (iname,), body))]


#: The same launches in plain loopy as loopty lowers them: the metric product
#: a substitution rule (loopy's assignment_to_subst, which substitute uses),
#: and the three sums of a sum over Fin[3] in one loop.
MIRROR = {
    "weak_grad": (
        (
            f"{{[q, e, k, j, i, a]: 0 <= q < nq and {_BOX} and 0 <= a < n}}",
            # Fr * (G * u): a substitution rule's product would be flattened
            lambda: _nested(
                "v[q, e, k, j, i]",
                "a",
                "Fr[q, i, a]",
                "G[q, e, k, j, a]",
                "u[e, k, j, a]",
            ),
        ),
        PLAIN["weak_grad"][1],
        (
            f"{{[q, e, k, j, i, c]: 0 <= q < 3 and {_BOX} and 0 <= c < n}}",
            "grad[q, e, k, j, i] = 0.0 + sum(c, Ft[3*q, k, c] * vs[3*q, e, c, j, i])"
            " + sum(c, Ft[3*q + 1, k, c] * vs[3*q + 1, e, c, j, i])"
            " + sum(c, Ft[3*q + 2, k, c] * vs[3*q + 2, e, c, j, i])",
        ),
    ),
    "weak_div": (
        (
            f"{{[q, e, k, j, i, a]: 0 <= q < nq and {_BOX} and 0 <= a < n}}",
            "w(q, e, k, j, a) := 0.0 + G[q, e, k, j, a] * f[0, e, k, j, a]"
            " + G[3 + q, e, k, j, a] * f[1, e, k, j, a]"
            " + G[6 + q, e, k, j, a] * f[2, e, k, j, a]\n"
            "v[q, e, k, j, i] = sum(a, Fr[q, i, a] * w(q, e, k, j, a))",
        ),
        PLAIN["weak_div"][1],
        (
            f"{{[e, k, j, i, c]: {_BOX} and 0 <= c < n}}",
            "div[e, k, j, i] = 0.0 + sum(c, Ft[0, k, c] * vs[0, e, c, j, i])"
            " + sum(c, Ft[1, k, c] * vs[1, e, c, j, i])"
            " + sum(c, Ft[2, k, c] * vs[2, e, c, j, i])",
        ),
    ),
}


def plain_launches(
    op: str, shape: Shape, order: int, ne: int, written: dict = PLAIN
) -> list[Launch]:
    """The same launches in plain loopy: the same splits, tags and loop order.

    ``written`` is :data:`PLAIN`, as one writes them, or :data:`MIRROR`, as
    loopty lowers them.
    """
    import loopy as lp

    sizes = {"n": order + 1, "ne": ne, "nq": 9 if op == "weak_grad" else 3}
    roles = {"e": "e", "k": "k", "j": "j", "i": "i", "q": "q"}
    out = []
    for (domain, instruction), (name, _subst, loops, args) in zip(
        written[op], LAUNCHES[op]
    ):
        kernel = lp.make_kernel(
            "[n, ne, nq] -> " + domain,
            instruction() if callable(instruction) else instruction,
            [lp.GlobalArg(arg, np.float64, shape=lp.auto) for arg in args]
            + [lp.ValueArg(size, np.int32) for size in ("n", "ne", "nq")],
            name=name,
            lang_version=(2018, 2),
        )
        kernel = fixed(kernel, False, **sizes)
        mine = {role: roles[role] for role in loops}
        splits, tags, sequential = shape.steps(mine)
        for loop, factor in splits:
            kernel = lp.split_iname(kernel, loop, factor)
        kernel = lp.tag_inames(kernel, tags)
        if len(sequential) > 1:
            kernel = lp.prioritize_loops(kernel, ",".join(sequential))
        launch = Launch(kernel, args, loopty=False)
        out.append(launch if shape.contraction else _without_contraction(launch))
    return out


OPERATORS = {
    "weak_grad": Operator("weak_grad", inputs=1, outputs=3, metric=9),
    "weak_div": Operator("weak_div", inputs=3, outputs=1, metric=9),
}


def variants(op: str) -> dict[str, Callable[[dict[str, Any], int, int], list[Launch]]]:
    """Each variant of ``op``, loopty's and plain loopy's, by name."""
    out: dict[str, Callable[[dict[str, Any], int, int], list[Launch]]] = {}
    for shape in SHAPES:
        out[f"loopty, {shape.name}"] = lambda demo, order, ne, shape=shape: (
            loopty_launches(demo, op, shape, order, ne)
        )
        out[f"loopy, {shape.name}"] = lambda demo, order, ne, shape=shape: (
            plain_launches(op, shape, order, ne)
        )
        out[f"loopy as lowered, {shape.name}"] = lambda demo, order, ne, shape=shape: (
            plain_launches(op, shape, order, ne, MIRROR)
        )
    return out


def buffers(
    queue: Any, demo: dict[str, Any], order: int, ne: int, seed: int = 0
) -> dict[str, Any]:
    """Every array the variants pass, on the device: random metric and fields."""
    import pyopencl.array as cla

    n = order + 1
    rng = np.random.default_rng(seed)
    nodes = (ne, n, n, n)
    stiffness_t, mass = demo["factors"](order)

    def table(fields: int, axis: int) -> np.ndarray:
        return np.stack([stiffness_t if q % 3 == axis else mass for q in range(fields)])

    host = {
        "G": rng.standard_normal((9, *nodes)),
        "u": rng.standard_normal(nodes),
        "f": rng.standard_normal((3, *nodes)),
        **{f"F{a}{q}": table(q, axis) for q in (9, 3) for axis, a in enumerate("rst")},
    }
    arrays = {name: cla.to_device(queue, value) for name, value in host.items()}
    for name, shape in (
        ("v", (9, *nodes)),
        ("vs", (9, *nodes)),
        ("grad", (3, *nodes)),
        ("v3", (3, *nodes)),
        ("vs3", (3, *nodes)),
        ("div", nodes),
    ):
        arrays[name] = cla.zeros(queue, shape, np.float64)
    return arrays


def reference(arrays: dict[str, Any], op: str, elements: slice) -> np.ndarray:
    """The operator on a few elements, in numpy: the passes as einsums."""
    G = arrays["G"].get()[:, elements]
    if op == "weak_grad":
        u = arrays["u"].get()[elements]
        w = G * u
        Fr, Fs, Ft = (arrays[f"F{a}9"].get() for a in "rst")
        v = np.einsum("qia,qekja->qekji", Fr, w)
        vs = np.einsum("qjb,qekbi->qekji", Fs, v)
        t = np.einsum("qkc,qecji->qekji", Ft, vs)
        return t.reshape(3, 3, *t.shape[1:]).sum(axis=1)
    f = arrays["f"].get()[:, elements]
    w = np.einsum("xdekji,xekji->dekji", G.reshape(3, 3, *G.shape[1:]), f)
    Fr, Fs, Ft = (arrays[f"F{a}3"].get() for a in "rst")
    v = np.einsum("qia,qekja->qekji", Fr, w)
    vs = np.einsum("qjb,qekbi->qekji", Fs, v)
    return np.einsum("qkc,qecji->ekji", Ft, vs)


# }}}


# {{{ the modes


def _rows_writer(path: str, fields: Sequence[str]) -> tuple[Any, Any]:
    handle = open(path, "w", newline="")  # noqa: SIM115 - closed by the caller
    writer = csv.DictWriter(handle, fieldnames=list(fields), extrasaction="ignore")
    writer.writeheader()
    return handle, writer


FIELDS = (
    "side",
    "operator",
    "variant",
    "order",
    "elements",
    "ndofs",
    "launches",
    "median_s",
    "min_s",
    "max_s",
    "kernel_s",
    "agree",
    "contraction",
    "device",
    "platform",
    "platform_version",
    "driver",
    "loopy",
    "islpy",
    "pyopencl",
    "numpy",
    "python",
)


def _said(ndofs: int, timing: Timing) -> str:
    """A time in words: GDOFs/s, the median and its spread, and the kernels'."""
    rate = ndofs / timing.median / 1e9
    spread = f"{timing.low * 1e3:.3f} to {timing.high * 1e3:.3f}"
    kernels = "" if timing.kernel is None else f"; in kernels {timing.kernel * 1e3:.3f}"
    return f"{rate:7.2f} GDOFs/s ({timing.median * 1e3:.3f} ms, {spread}{kernels})"


def mode_copy(options: argparse.Namespace) -> None:
    """The size-matched copy bandwidth, per operator and order."""
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
    measured: dict[int, Timing] = {}
    for order in options.orders:
        ndofs = options.elements**3 * (order + 1) ** 3
        for op in OPERATORS.values():
            nbytes = required_bytes(op, ndofs)
            if nbytes not in measured:
                # a copy reads and writes, so half the bytes, copied, move them all
                measured[nbytes] = copy_seconds(
                    queue, nbytes // 2, options.warmup, options.reps
                )
            timing = measured[nbytes]
            row = {
                "operator": op.name,
                "order": order,
                "elements": options.elements,
                "bytes": nbytes,
                "median_s": timing.median,
                "min_s": timing.low,
                "max_s": timing.high,
                "bandwidth": nbytes / timing.median,
                **meta,
            }
            writer.writerow(row)
            handle.flush()
            print(
                f"copy {op.name} order {order}: "
                f"{nbytes / timing.median / 1e12:.3f} TB/s "
                f"({nbytes / timing.high / 1e12:.3f} "
                f"to {nbytes / timing.low / 1e12:.3f})",
                flush=True,
            )
    handle.close()


def mode_loopty(options: argparse.Namespace) -> None:
    """Every variant, loopty's and plain loopy's, checked and timed."""
    context, queue = queue_and_context()
    meta = describe(queue)
    demo = load_demo()
    handle, writer = _rows_writer(options.csv, FIELDS)
    ne = options.elements**3
    for order in options.orders:
        arrays = buffers(queue, demo, order, ne)
        sample = slice(0, min(ne, 7))
        for name in options.operators:
            expected = reference(arrays, name, sample)
            out = "grad" if name == "weak_grad" else "div"
            for variant, build in variants(name).items():
                if options.only and not any(word in variant for word in options.only):
                    continue
                try:
                    launches = build(demo, order, ne)
                    calls = [launch.executor(context) for launch in launches]
                except Exception as exc:  # noqa: BLE001 - a variant that does not build
                    print(
                        f"{name} order {order} {variant}: does not build: "
                        f"{type(exc).__name__}: {exc}",
                        flush=True,
                    )
                    continue

                def call(calls: list[Any] = calls, arrays: dict = arrays) -> None:
                    for run in calls:
                        run(queue, arrays)

                arrays[out].fill(0)
                call()
                got = (
                    arrays[out].get()[:, sample]
                    if name == "weak_grad"
                    else arrays[out].get()[sample]
                )
                gap = float(np.abs(got - expected).max() / np.abs(expected).max())
                timing = clock(queue, call, options.warmup, options.reps, options.batch)
                ndofs = ne * (order + 1) ** 3
                row = {
                    "side": variant.split(",")[0],
                    "operator": name,
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
                    "contraction": "compiler default",
                    **meta,
                }
                writer.writerow(row)
                handle.flush()
                print(
                    f"{name} order {order} {variant}: {_said(ndofs, timing)}, "
                    f"agrees to {gap:.1e}",
                    flush=True,
                )
        del arrays
    handle.close()


def _grudge_setup(
    queue: Any, elements: int, order: int, lazy: str | None, warped: bool
) -> tuple[Any, Any]:
    from grudge.array_context import PyOpenCLArrayContext, PytatoPyOpenCLArrayContext
    from grudge.discretization import make_discretization_collection
    from meshmode.mesh import TensorProductElementGroup
    from meshmode.mesh.generation import (
        generate_regular_rect_mesh,
        generate_warped_rect_mesh,
    )
    from pyopencl.tools import ImmediateAllocator, MemoryPool

    allocator = MemoryPool(ImmediateAllocator(queue))
    if lazy is None:
        actx = PyOpenCLArrayContext(queue, allocator=allocator)
    elif lazy == "pytato":
        actx = PytatoPyOpenCLArrayContext(queue, allocator=allocator)
    else:
        from meshmode.array_context import FusionContractorArrayContext

        actx = FusionContractorArrayContext(queue, allocator=allocator)
    if warped:
        mesh = generate_warped_rect_mesh(
            3, order=order, nelements_side=elements, group_cls=TensorProductElementGroup
        )
    else:
        mesh = generate_regular_rect_mesh(
            a=(-1.0,) * 3,
            b=(1.0,) * 3,
            nelements_per_axis=(elements,) * 3,
            group_cls=TensorProductElementGroup,
        )
    return actx, make_discretization_collection(actx, mesh, order=order)


def mode_grudge(options: argparse.Namespace) -> None:
    """grudge's own weak gradient and divergence, eager and lazy."""
    import grudge
    from grudge import op as gop
    from pytools.obj_array import new_1d

    _context, queue = queue_and_context()
    meta = describe(queue)
    meta["loopy"] += " (grudge's environment)"
    handle, writer = _rows_writer(options.csv, FIELDS)
    for order in options.orders:
        for lazy in options.grudge_modes:
            mode = None if lazy == "eager" else lazy
            try:
                actx, dcoll = _grudge_setup(
                    queue, options.elements, order, mode, warped=False
                )
            except Exception as exc:  # noqa: BLE001 - said, and the next is tried
                print(
                    f"grudge {lazy} order {order}: setup failed: "
                    f"{type(exc).__name__}: {exc}",
                    flush=True,
                )
                continue
            x = actx.thaw(dcoll.nodes())
            u = actx.freeze(actx.np.sin(x[0]) * actx.np.cos(x[1]) + x[2])
            f = actx.freeze(new_1d([actx.np.sin(x[0]), actx.np.cos(x[1]), x[2] * x[0]]))
            ndofs = dcoll.discr_from_dd("vol").ndofs
            for name in options.operators:
                for factorized in (kind == "factorized" for kind in options.kinds):

                    def apply(
                        u: Any,
                        f: Any,
                        name: str = name,
                        factorized: bool = factorized,
                        dcoll: Any = dcoll,
                    ) -> Any:
                        if name == "weak_grad":
                            return gop.weak_local_grad(
                                dcoll, u, enable_sum_factorization=factorized
                            )
                        return gop.weak_local_div(
                            dcoll, f, enable_sum_factorization=factorized
                        )

                    run = apply if mode is None else actx.compile(apply)

                    def call(run: Any = run, actx: Any = actx, u: Any = u, f: Any = f):
                        return run(actx.thaw(u), actx.thaw(f))

                    if options.profile:
                        call()
                        queue.finish()
                        with KernelClock() as one:
                            call()
                        for kname, (count, seconds) in sorted(
                            one.by_kernel().items(), key=lambda kv: -kv[1][1]
                        ):
                            print(
                                f"    {kname}: {count} launches, "
                                f"{seconds * 1e3:.3f} ms",
                                flush=True,
                            )
                    try:
                        timing = clock(
                            queue, call, options.warmup, options.reps, options.batch
                        )
                    except Exception as exc:  # noqa: BLE001 - said, and the next is tried
                        print(
                            f"grudge {lazy} {name} order {order}: failed: "
                            f"{type(exc).__name__}: {exc}",
                            flush=True,
                        )
                        continue
                    variant = (
                        f"grudge {lazy}, {'sum-factorized' if factorized else 'dense'}"
                    )
                    row = {
                        "side": "grudge",
                        "operator": name,
                        "variant": variant,
                        "order": order,
                        "elements": options.elements,
                        "ndofs": ndofs,
                        "launches": "",
                        "median_s": timing.median,
                        "min_s": timing.low,
                        "max_s": timing.high,
                        "kernel_s": timing.kernel,
                        "agree": "",
                        "contraction": "compiler default",
                        **meta,
                        "loopy": meta["loopy"] + f"; grudge {grudge.__file__}",
                    }
                    writer.writerow(row)
                    handle.flush()
                    print(
                        f"{name} order {order} {variant}: {_said(ndofs, timing)}",
                        flush=True,
                    )
            del actx, dcoll
    handle.close()


def mode_fields(options: argparse.Namespace) -> None:
    """grudge's metric terms, fields, factors and operators on a warped mesh."""
    from grudge import op as gop
    from grudge.bilinear_forms import make_stiffness_t_operator
    from grudge.geometry import inverse_surface_metric_derivative_mat
    from pytools.obj_array import new_1d

    _context, queue = queue_and_context()
    order = options.order
    actx, dcoll = _grudge_setup(queue, options.elements, order, None, warped=True)
    discr = dcoll.discr_from_dd("vol")
    group = discr.groups[0]
    x = actx.thaw(discr.nodes())
    u = actx.np.sin(x[0]) * actx.np.cos(2 * x[1]) + x[2] ** 2
    f = new_1d([actx.np.cos(x[1]) * x[2], actx.np.sin(x[0] * x[2]), x[0] * x[1]])
    ijm = inverse_surface_metric_derivative_mat(actx, dcoll, times_area_element=True)
    operators = make_stiffness_t_operator(actx, group, group)
    out = {
        "order": order,
        "G": actx.to_numpy(ijm[0]),
        "u": actx.to_numpy(u[0]),
        "f": np.stack([actx.to_numpy(fx[0]) for fx in f]),
        "stiffness_t": actx.to_numpy(operators[0][0]),
        "mass": actx.to_numpy(operators[0][1]),
    }
    for factorized in (True, False):
        grad = gop.weak_local_grad(dcoll, u, enable_sum_factorization=factorized)
        div = gop.weak_local_div(dcoll, f, enable_sum_factorization=factorized)
        key = "factorized" if factorized else "dense"
        out[f"grad_{key}"] = np.stack([actx.to_numpy(g[0]) for g in grad])
        out[f"div_{key}"] = actx.to_numpy(div[0])
    np.savez(options.npz, **out)
    shapes = ", ".join(f"{k} {np.shape(v)}" for k, v in out.items())
    print(f"wrote {options.npz}: {shapes}")


def mode_agree(options: argparse.Namespace) -> None:
    """loopty's programs and launches on grudge's fields, against grudge's results."""
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
    # grudge's (.., ne, n**3), node i + n j + n**2 k, is (.., ne, k, j, i) row-major
    ne = data["u"].shape[0]
    G = data["G"].reshape(9, ne, n, n, n)
    u = data["u"].reshape(ne, n, n, n)
    f = data["f"].reshape(3, ne, n, n, n)

    def tables(fields: int) -> list[np.ndarray]:
        return [
            np.stack([stiffness_t if q % 3 == axis else mass for q in range(fields)])
            for axis in range(3)
        ]

    Fr9, Fs9, Ft9 = tables(9)
    Fr3, Fs3, Ft3 = tables(3)
    expected = {
        "weak_grad": data["grad_factorized"].reshape(3, ne, n, n, n),
        "weak_div": data["div_factorized"].reshape(ne, n, n, n),
    }
    inputs = {
        "weak_grad": {
            "G": G,
            "u": u,
            "Fr": Fr9,
            "Fs": Fs9,
            "Ft": Ft9,
            "grad": np.zeros((3, ne, n, n, n)),
        },
        "weak_div": {
            "G": G,
            "f": f,
            "Fr": Fr3,
            "Fs": Fs3,
            "Ft": Ft3,
            "div": np.zeros((ne, n, n, n)),
        },
    }
    scale = {k: np.abs(v).max() for k, v in expected.items()}
    dense_gap = max(
        np.abs(data["grad_dense"] - data["grad_factorized"]).max() / scale["weak_grad"],
        np.abs(data["div_dense"] - data["div_factorized"]).max() / scale["weak_div"],
    )
    print(f"grudge's own dense against its sum-factorized: {dense_gap:.1e}")
    for name, program in (
        ("weak_grad", demo["weak_grad"]),
        ("weak_div", demo["weak_div"]),
    ):
        out = "grad" if name == "weak_grad" else "div"
        args = {k: Arr.from_numpy(np.array(v)) for k, v in inputs[name].items()}
        program(**args)
        native = np.abs(args[out].numpy() - expected[name]).max() / scale[name]
        compiled = LoopyExecutor().run(
            program, **{k: Arr.from_numpy(np.array(v)) for k, v in inputs[name].items()}
        )
        on_c = np.abs(np.asarray(compiled[out]) - expected[name]).max() / scale[name]
        print(
            f"{name}: loopty's program against grudge #418: native {native:.1e}, "
            f"compiled on C {on_c:.1e}"
        )
    if demo.get("contracted") is not None:
        kept = LoopyExecutor().run(
            demo["contracted"],
            **{k: Arr.from_numpy(np.array(v)) for k, v in inputs["weak_grad"].items()},
        )
        gap = (
            np.abs(np.asarray(kept["grad"]) - expected["weak_grad"]).max()
            / scale["weak_grad"]
        )
        print(
            "weak_grad: the fused and contracted schedule on C against grudge #418: "
            f"{gap:.1e}"
        )
    if os.environ.get("PYOPENCL_CTX"):
        import pyopencl.array as cla

        context, queue = queue_and_context()
        arrays = {
            "G": G,
            "u": u,
            "f": f,
            "Fr9": Fr9,
            "Fs9": Fs9,
            "Ft9": Ft9,
            "Fr3": Fr3,
            "Fs3": Fs3,
            "Ft3": Ft3,
        }
        arrays = {
            k: cla.to_device(queue, np.ascontiguousarray(v)) for k, v in arrays.items()
        }
        for name2, shape in (
            ("v", (9, ne, n, n, n)),
            ("vs", (9, ne, n, n, n)),
            ("grad", (3, ne, n, n, n)),
            ("v3", (3, ne, n, n, n)),
            ("vs3", (3, ne, n, n, n)),
            ("div", (ne, n, n, n)),
        ):
            arrays[name2] = cla.zeros(queue, shape, np.float64)
        for name in ("weak_grad", "weak_div"):
            build = variants(name)["loopty, element per group"]
            for launch in build(demo, order, ne):
                launch.executor(context)(queue, arrays)
            out = "grad" if name == "weak_grad" else "div"
            gap = np.abs(arrays[out].get() - expected[name]).max() / scale[name]
            print(
                f"{name}: loopty's device launches against grudge #418: {gap:.1e} "
                f"on {queue.device.name}"
            )


def mode_report(options: argparse.Namespace) -> None:
    """GDOFs/s and % of the copy roof, per operator, order and variant."""
    rows: list[dict[str, str]] = []
    copies: dict[tuple[str, str], float] = {}
    for path in options.files:
        with open(path, newline="") as handle:
            for row in csv.DictReader(handle):
                if "bandwidth" in row:
                    copies[(row["operator"], row["order"])] = float(row["bandwidth"])
                else:
                    rows.append(row)
    for name, op in OPERATORS.items():
        mine = [r for r in rows if r["operator"] == name]
        if not mine:
            continue
        orders = sorted({r["order"] for r in mine}, key=int)
        names = list(dict.fromkeys(r["variant"] for r in mine))
        print(f"\n{name}: GDOFs/s (% of the copy roof)\n")
        print("| variant | " + " | ".join(f"order {o}" for o in orders) + " |")
        print("|---|" + "---:|" * len(orders))
        for variant in names:
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
                ndofs = int(r["ndofs"])
                rate = ndofs / seconds / 1e9
                text = f"{rate:.2f}" if rate >= 0.1 else f"{rate:.2g}"
                bandwidth = copies.get((name, order))
                if bandwidth:
                    roof = required_bytes(op, ndofs) / (bandwidth * seconds)
                    text += (
                        f" ({100 * roof:.0f}%)"
                        if roof >= 0.01
                        else f" ({100 * roof:.1g}%)"
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
    parser.add_argument(
        "--order", type=int, default=3, help="the order of fields/agree"
    )
    parser.add_argument("--operators", nargs="+", default=list(OPERATORS))
    parser.add_argument(
        "--only",
        nargs="*",
        default=[],
        help="variants whose names contain one of these",
    )
    parser.add_argument("--grudge-modes", nargs="+", default=["eager", "pytato"])
    parser.add_argument(
        "--kinds",
        nargs="+",
        default=["factorized"],
        choices=["factorized", "dense"],
        help="grudge's operators: sum-factorized, dense (its escape hatch), or both",
    )
    parser.add_argument(
        "--profile",
        action="store_true",
        help="each kernel's time in one call of grudge's",
    )
    parser.add_argument("--warmup", type=int, default=3)
    parser.add_argument("--reps", type=int, default=7)
    parser.add_argument("--batch", type=int, default=5)
    parser.add_argument("--csv", default="hexahedra.csv")
    parser.add_argument("--npz", default="fields.npz")
    options = parser.parse_args(argv)
    started = time.perf_counter()
    globals()[f"mode_{options.mode}"](options)
    print(f"({options.mode}: {time.perf_counter() - started:.0f} s)", file=sys.stderr)


if __name__ == "__main__":
    main()
