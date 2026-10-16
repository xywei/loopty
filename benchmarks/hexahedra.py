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
``python benchmarks/hexahedra.py c --elements 16 --csv c.csv``
    What fusing and contracting the passes buys on a CPU: the weak gradient
    as the demo's programs on the C target, one thread, the passes one after
    another, ``fused`` and ``contracted``, built with the same flags.
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
the minimum and the maximum go beside it. On the C target it is the wall
time of the batch, and the programs take their batches in turn, so that a
change in the machine's load falls on all of them. Each row records the
device, the driver and the versions it ran with; nothing here names the
machine.
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
        "loopty": _checkout(),
        "loopy": _provenance("loopy", ".".join(map(str, loopy.VERSION))),
        "islpy": _provenance("islpy", islpy.__version__),
        "pyopencl": ".".join(map(str, cl.VERSION)),
        "numpy": np.__version__,
        "python": sys.version.split()[0],
    }


def _checkout() -> str:
    """loopty's version, and the commit of the checkout it was imported from.

    A checkout with changes to tracked files says so, since its commit alone
    does not say what ran.
    """
    import subprocess

    try:
        import loopty
    except ImportError:  # grudge's environment need not have it
        return "not installed"
    where = str(Path(loopty.__file__).resolve().parent)

    def git(*args: str) -> str:
        return subprocess.run(
            ["git", "-C", where, *args], capture_output=True, text=True, check=True
        ).stdout.strip()

    try:
        commit = git("rev-parse", "--short=7", "HEAD")
        changed = git("status", "--porcelain", "--untracked-files=no")
    except (OSError, subprocess.CalledProcessError):
        return loopty.__version__
    return f"{loopty.__version__} at {commit}" + (" with changes" if changed else "")


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
        self, loops: dict[str, Any]
    ) -> tuple[list[tuple[Any, int]], dict[Any, str], list[Any]]:
        """The splits, the tags and the order of the sequential loops, for ``loops``.

        ``loops`` names a launch's loops by role: ``e``, ``k``, ``j``, ``i``
        and, when it has one, ``q``; each is a loopty handle, or plain loopy's
        name, and a split's two loops the handle's children or loopy's
        ``e_outer`` and ``e_inner``.
        """
        e, k, j, i = (loops[x] for x in "ekji")
        q = loops.get("q")
        splits: list[tuple[Any, int]] = []
        if self.per_group == 1:
            tags = {e: "g.0", k: "l.2", j: "l.1", i: "l.0"}
            sequential = []
        else:
            splits.append((e, self.per_group))
            tags = {_part(e, "outer"): "g.0", _part(e, "inner"): "l.2", j: "l.1"}
            tags[i] = "l.0"
            sequential = [k]
        if q is not None:
            if self.fields == "groups":
                tags[q] = "g.1"
            elif self.fields == "ilp":
                # three at a time: an ilp loop needs a length that is a number
                splits.append((q, 3))
                tags[_part(q, "inner")] = "ilp"
                sequential = [_part(q, "outer"), *sequential]
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


def _part(loop: Any, side: str) -> Any:
    """The ``side`` half of a split of ``loop``: its handle's, or loopy's name."""
    return getattr(loop, side) if not isinstance(loop, str) else f"{loop}_{side}"


def _placed(schedule: Any, shape: Shape, loops: dict[str, Any]) -> Any:
    """``schedule`` with ``shape``'s splits and tags, each step checked by loopty."""
    splits, tags, _sequential = shape.steps(loops)
    for loop, factor in splits:
        schedule = schedule.split(loop, factor)
    return schedule.tag(tags)


def _without_contraction(launch: Launch) -> Launch:
    """``launch`` with ``#pragma OPENCL FP_CONTRACT OFF`` before its code."""
    entry = launch.kernel.default_entrypoint
    pragma = ("04_no_fp_contract", "#pragma OPENCL FP_CONTRACT OFF")
    kernel = launch.kernel.with_kernel(entry.copy(preambles=(*entry.preambles, pragma)))
    return Launch(kernel, launch.args, launch.loopty, launch.facts)


#: Each operator's launches: the demo's kernel or program, what is substituted
#: in it, its loops by role (the binders of the scope ``_scope`` gives), and
#: the arrays it takes, by the variants' names.
LAUNCHES = {
    "weak_grad": (
        (
            "metric_along_r",
            "w",
            {"e": "e", "k": "k", "j": "j", "i": "i", "q": "q"},
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
            {"e": "e", "k": "k", "j": "j", "i": "i", "q": "q"},
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


def _scope(launch: Any) -> Any:
    """Where a launch's loops are: a kernel's own, or the pass along r a program
    runs after what it substitutes (``metric_along_r.along_r``)."""
    return getattr(launch, "along_r", None) or launch


def loopty_launches(
    demo: dict[str, Any], op: str, shape: Shape, order: int, ne: int
) -> list[Launch]:
    """loopty's launches of ``op`` in ``shape``: each a schedule loopty checked."""
    from loopty import Schedule

    sizes = {"n": order + 1, "ne": ne, "nq": 9 if op == "weak_grad" else 3}
    out = []
    for name, substituted, loops, args in LAUNCHES[op]:
        program = demo[name]
        schedule = Schedule(program)
        if substituted is not None:
            schedule = schedule.substitute(getattr(program, substituted))
        scope = _scope(program)
        handles = {role: getattr(scope, binder) for role, binder in loops.items()}
        launch = from_schedule(_placed(schedule, shape, handles), args, **sizes)
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


#: How far a variant's result may be from numpy's, relative to its largest
#: value, before it is refused rather than timed: the sums are of at most
#: 3n terms, and every variant measured agrees to below 1e-15.
AGREEMENT = 1e-12


def variants(op: str) -> dict[str, tuple[Callable[..., list[Launch]], Shape]]:
    """Each variant of ``op``, loopty's and plain loopy's, by name, with its shape."""
    out: dict[str, tuple[Callable[..., list[Launch]], Shape]] = {}
    for shape in SHAPES:
        out[f"loopty, {shape.name}"] = (
            lambda demo, order, ne, shape=shape: loopty_launches(
                demo, op, shape, order, ne
            ),
            shape,
        )
        out[f"loopy, {shape.name}"] = (
            lambda demo, order, ne, shape=shape: plain_launches(op, shape, order, ne),
            shape,
        )
        out[f"loopy as lowered, {shape.name}"] = (
            lambda demo, order, ne, shape=shape: plain_launches(
                op, shape, order, ne, MIRROR
            ),
            shape,
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


def operator(
    op: str, G: np.ndarray, field: np.ndarray, Fr: Any, Fs: Any, Ft: Any
) -> np.ndarray:
    """``op`` in numpy, the passes as einsums: ``field`` is ``u`` or ``f``.

    ``G`` is ``(9, ne, n, n, n)``, field ``3 x + d`` the metric term ``J
    dr_d/dx_x``; ``Fr``, ``Fs`` and ``Ft`` are each field's matrix along an
    axis, nine of them for the gradient and three for the divergence.
    """
    if op == "weak_grad":
        w = G * field
    else:
        w = np.einsum("xdekji,xekji->dekji", G.reshape(3, 3, *G.shape[1:]), field)
    v = np.einsum("qia,qekja->qekji", Fr, w)
    vs = np.einsum("qjb,qekbi->qekji", Fs, v)
    t = np.einsum("qkc,qecji->qekji", Ft, vs)
    if op == "weak_grad":
        return t.reshape(3, 3, *t.shape[1:]).sum(axis=1)
    return t.sum(axis=0)


def sample(ne: int, each: int = 4) -> np.ndarray:
    """A few elements at the start, the middle and the end of ``ne``."""
    middle = ne // 2 - each // 2
    picked = np.r_[0:each, middle : middle + each, ne - each : ne]
    return np.unique(picked.clip(0, ne - 1))


def reference(arrays: dict[str, Any], op: str, elements: Any) -> np.ndarray:
    """The operator on a few elements, in numpy, from the device's arrays."""
    if op == "weak_grad":
        field, fields = arrays["u"].get()[elements], 9
    else:
        field, fields = arrays["f"].get()[:, elements], 3
    tables = (arrays[f"F{a}{fields}"].get() for a in "rst")
    return operator(op, arrays["G"].get()[:, elements], field, *tables)


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
    "flags",
    "load",
    "device",
    "platform",
    "platform_version",
    "driver",
    "loopty",
    "loopy",
    "islpy",
    "pyopencl",
    "numpy",
    "python",
)


def _provenance(package: str, version: str | None = None) -> str:
    """A package's version, and the commit it was installed from when it says.

    ``version`` is what the imported module says, used when the package is
    imported without being installed (from a source tree on the path).
    """
    import importlib.metadata as metadata
    import json

    try:
        distribution = metadata.distribution(package)
    except metadata.PackageNotFoundError:
        return version or "?"
    found = distribution.read_text("direct_url.json")
    commit = json.loads(found).get("vcs_info", {}).get("commit_id") if found else None
    return (version or distribution.version) + (f" at {commit[:7]}" if commit else "")


def _said(ndofs: int, timing: Timing) -> str:
    """A time in words: GDOFs/s, the median and its spread, and the kernels'."""
    rate = ndofs / timing.median / 1e9
    spread = f"{timing.low * 1e3:.3f} to {timing.high * 1e3:.3f}"
    kernels = "" if timing.kernel is None else f"; in kernels {timing.kernel * 1e3:.3f}"
    # three figures below one, where a core of a CPU is
    shown = f"{rate:7.2f}" if rate >= 1 else f"{rate:7.3g}"
    return f"{shown} GDOFs/s ({timing.median * 1e3:.3f} ms, {spread}{kernels})"


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
    ne = options.elements**3
    arrays = buffers(queue, demo, order, ne)
    # elements at the start, the middle and the end: a launch that maps a
    # later work group wrong is caught too
    checked = sample(ne)
    for name in options.operators:
        expected = reference(arrays, name, checked)
        out = "grad" if name == "weak_grad" else "div"
        for variant, (build, shape) in variants(name).items():
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

            def call(calls: list[Any] = calls) -> None:
                for run in calls:
                    run(queue, arrays)

            arrays[out].fill(0)
            try:
                call()
                queue.finish()
            except Exception as exc:  # noqa: BLE001 - a shape the device refuses
                print(
                    f"{name} order {order} {variant}: does not launch: "
                    f"{type(exc).__name__}: {exc}",
                    flush=True,
                )
                continue
            got = (
                arrays[out].get()[:, checked]
                if name == "weak_grad"
                else arrays[out].get()[checked]
            )
            gap = float(np.abs(got - expected).max() / np.abs(expected).max())
            if not gap <= AGREEMENT:
                # a wrong answer is not a throughput: said, and not timed
                print(
                    f"{name} order {order} {variant}: DISAGREES with numpy by "
                    f"{gap:.1e} of the largest value, so it is not timed",
                    flush=True,
                )
                continue
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
                "contraction": (
                    "compiler default"
                    if shape.contraction
                    else "off: #pragma OPENCL FP_CONTRACT OFF"
                ),
                **meta,
            }
            writer.writerow(row)
            handle.flush()
            print(
                f"{name} order {order} {variant}: {_said(ndofs, timing)}, "
                f"agrees to {gap:.1e}",
                flush=True,
            )


# {{{ on a CPU: the weak gradient's programs on the C target


def c_programs(demo: dict[str, Any]) -> dict[str, Any]:
    """The weak gradient as the demo's programs, by name, each a checked schedule.

    The passes one after another, each storing its field whole before the
    next reads it; the passes fused as deep as each pair allows, ``fused``,
    which still stores ``w`` and ``v`` whole; and ``contracted``, which keeps
    ``w`` a row at a time and ``v`` a slab at a time. So the first two differ
    by the order the passes run in, and the last two by the memory ``w`` and
    ``v`` take, and the zeroing a program gives each array it makes.
    """
    from loopty import Schedule

    return {
        "passes one after another": Schedule(demo["weak_grad"]),
        "fused": demo["fused"],
        "fused and contracted": demo["contracted"],
    }


def c_call(
    schedule: Any, order: int, ne: int, flags: Sequence[str]
) -> tuple[Callable[[dict[str, Any]], None], dict[str, tuple[Any, Any]], list[str]]:
    """``schedule`` compiled for the C target with every size fixed.

    Returns the call, which takes the arrays by name, the arrays the program
    makes, which a call on the C target is passed (see
    :func:`loopty.lower.heap_temporaries`) and which are allocated once, as a
    code that calls the program again and again keeps them, and the flags it
    is built with: loopty's for the program, then ``flags``, after loopy's
    own ``-std=c99 -O3 -fPIC``.
    """
    import loopy as lp

    from loopty.isl_reading import declining
    from loopty.lower import heap_temporaries

    sizes = {"n": order + 1, "ne": ne}
    kernel, made = heap_temporaries(schedule.kernel, schedule.term, sizes)
    built = [*(kernel.default_entrypoint.options.build_options or []), *flags]
    kernel = lp.set_options(fixed(kernel, True, **sizes), build_options=built)
    names = [arg.name for arg in kernel.default_entrypoint.args]
    with declining():
        run = kernel.executor()

    def call(arrays: dict[str, Any]) -> None:
        with declining():
            run(**{name: arrays[name] for name in names})

    return call, made, built


def describe_c() -> dict[str, str]:
    """The processor and the software a C row was measured with."""
    import subprocess

    import islpy
    import loopy

    cpu = "?"
    try:
        for line in Path("/proc/cpuinfo").read_text().splitlines():
            if line.startswith("model name"):
                cpu = line.split(":", 1)[1].strip()
                break
    except OSError:
        pass
    try:
        compiler = subprocess.run(
            ["gcc", "--version"], capture_output=True, text=True, check=True
        ).stdout.splitlines()[0]
    except (OSError, IndexError, subprocess.CalledProcessError):
        compiler = "?"
    return {
        "device": cpu,
        "platform": "C target, one thread",
        "platform_version": "",
        "driver": compiler,
        "loopty": _checkout(),
        "loopy": _provenance("loopy", ".".join(map(str, loopy.VERSION))),
        "islpy": _provenance("islpy", islpy.__version__),
        "pyopencl": "",
        "numpy": np.__version__,
        "python": sys.version.split()[0],
    }


def mode_c(options: argparse.Namespace) -> None:
    """The weak gradient's programs on the C target, each with the same flags."""
    meta = describe_c()
    demo = load_demo()
    handle, writer = _rows_writer(options.csv, FIELDS)
    for order in options.orders:
        # one order's arrays at a time: they are freed when it returns
        _time_order_c(options, demo, order, meta, writer, handle)
    handle.close()


def _time_order_c(
    options: argparse.Namespace,
    demo: dict[str, Any],
    order: int,
    meta: dict[str, str],
    writer: Any,
    handle: Any,
) -> None:
    """Every program at one order, each checked against numpy, then timed in turn."""
    ne = options.elements**3
    n = order + 1
    nodes = (ne, n, n, n)
    rng = np.random.default_rng(0)
    stiffness_t, mass = demo["factors"](order)
    Fr, Fs, Ft = (
        np.stack([stiffness_t if q % 3 == axis else mass for q in range(9)])
        for axis in range(3)
    )
    host = {
        "G": rng.standard_normal((9, *nodes)),
        "u": rng.standard_normal(nodes),
        "Fr": Fr,
        "Fs": Fs,
        "Ft": Ft,
    }
    checked = sample(ne)
    expected = operator(
        "weak_grad", host["G"][:, checked], host["u"][checked], Fr, Fs, Ft
    )
    ready: dict[str, tuple[Callable[[dict[str, Any]], None], dict[str, Any]]] = {}
    built: dict[str, list[str]] = {}
    for name, schedule in c_programs(demo).items():
        variant = f"loopty on C, {name}"
        if options.only and not any(word in variant for word in options.only):
            continue
        try:
            call, made, built[name] = c_call(
                schedule, order, ne, options.cflags.split()
            )
        except Exception as exc:  # noqa: BLE001 - a program that does not build
            print(
                f"weak_grad order {order} {variant}: does not build: "
                f"{type(exc).__name__}: {exc}",
                flush=True,
            )
            continue
        arrays = {
            **host,
            "grad": np.zeros((3, *nodes)),
            **{
                array: np.zeros(shape, dtype=dtype)
                for array, (shape, dtype) in made.items()
            },
        }
        call(arrays)
        got = arrays["grad"][:, checked]
        gap = float(np.abs(got - expected).max() / np.abs(expected).max())
        if not gap <= AGREEMENT:
            print(
                f"weak_grad order {order} {variant}: DISAGREES with numpy by "
                f"{gap:.1e} of the largest value, so it is not timed",
                flush=True,
            )
            continue
        ready[name] = (call, arrays)
    if len({tuple(flags) for flags in built.values()}) > 1:
        raise SystemExit(f"the programs are built with different flags: {built}")
    for _ in range(options.warmup):
        for call, arrays in ready.values():
            call(arrays)
    spans: dict[str, list[float]] = {name: [] for name in ready}
    load = f"{os.getloadavg()[0]:.1f}"
    for _ in range(options.reps):
        for name, (call, arrays) in ready.items():
            start = time.perf_counter()
            for _ in range(options.batch):
                call(arrays)
            spans[name].append((time.perf_counter() - start) / options.batch)
    load += f" to {os.getloadavg()[0]:.1f}"
    ndofs = ne * n**3
    for name, (call, arrays) in ready.items():
        timing = Timing.of(spans[name])
        gap = float(
            np.abs(arrays["grad"][:, checked] - expected).max()
            / np.abs(expected).max()
        )
        row = {
            "side": "loopty",
            "operator": "weak_grad",
            "variant": f"loopty on C, {name}",
            "order": order,
            "elements": options.elements,
            "ndofs": ndofs,
            "launches": 1,
            "median_s": timing.median,
            "min_s": timing.low,
            "max_s": timing.high,
            "kernel_s": "",
            "agree": f"{gap:.1e}",
            "contraction": "compiler default",
            "flags": " ".join(["-std=c99 -O3 -fPIC", *built[name]]),
            "load": load,
            **meta,
        }
        writer.writerow(row)
        handle.flush()
        print(
            f"weak_grad order {order} loopty on C, {name}: {_said(ndofs, timing)}, "
            f"agrees to {gap:.1e}",
            flush=True,
        )


# }}}


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
    from grudge import op as gop
    from pytools.obj_array import new_1d

    _context, queue = queue_and_context()
    meta = describe(queue)
    meta["loopy"] += " (grudge's environment)"
    handle, writer = _rows_writer(options.csv, FIELDS)
    for order in options.orders:
        for lazy in options.grudge_modes:
            # one setup at a time: its context, mesh and fields are freed
            # when the helper returns
            _time_grudge(options, queue, gop, new_1d, order, lazy, meta, writer, handle)
            queue.finish()
    handle.close()


def _time_grudge(
    options: argparse.Namespace,
    queue: Any,
    gop: Any,
    new_1d: Any,
    order: int,
    lazy: str,
    meta: dict[str, str],
    writer: Any,
    handle: Any,
) -> None:
    """grudge's operators in one array context at one order."""
    mode = None if lazy == "eager" else lazy
    try:
        actx, dcoll = _grudge_setup(queue, options.elements, order, mode, warped=False)
    except Exception as exc:  # noqa: BLE001 - said, and the next is tried
        print(
            f"grudge {lazy} order {order}: setup failed: {type(exc).__name__}: {exc}",
            flush=True,
        )
        return
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
                return gop.weak_local_div(dcoll, f, enable_sum_factorization=factorized)

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
                        f"    {kname}: {count} launches, {seconds * 1e3:.3f} ms",
                        flush=True,
                    )
            try:
                timing = clock(queue, call, options.warmup, options.reps, options.batch)
            except Exception as exc:  # noqa: BLE001 - said, and the next is tried
                print(
                    f"grudge {lazy} {name} order {order}: failed: "
                    f"{type(exc).__name__}: {exc}",
                    flush=True,
                )
                continue
            variant = f"grudge {lazy}, {'sum-factorized' if factorized else 'dense'}"
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
                "loopy": meta["loopy"] + f"; grudge {_provenance('grudge')}",
            }
            writer.writerow(row)
            handle.flush()
            print(
                f"{name} order {order} {variant}: {_said(ndofs, timing)}",
                flush=True,
            )


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
    # the comparison tells a wrong reading of grudge's arrays apart: the
    # metric terms transposed (J dr_x/dx_d for J dr_d/dx_x), or each factor
    # applied along the other end's axis
    transposed = G.reshape(3, 3, *G.shape[1:]).swapaxes(0, 1).reshape(G.shape)
    for what, metric, nine, three in (
        ("the metric terms transposed", transposed, tables(9), tables(3)),
        ("the factors along r and t swapped", G, tables(9)[::-1], tables(3)[::-1]),
    ):
        grad = operator("weak_grad", metric, u, *nine)
        div = operator("weak_div", metric, f, *three)
        print(
            f"a wrong reading, {what}, disagrees: "
            f"{np.abs(grad - expected['weak_grad']).max() / scale['weak_grad']:.1e}, "
            f"{np.abs(div - expected['weak_div']).max() / scale['weak_div']:.1e}"
        )
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
            build, _shape = variants(name)["loopty, element per group"]
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
    copies: dict[tuple[str, ...], float] = {}

    def matched(row: dict[str, str]) -> tuple[str, ...]:
        # size-matched: the same operator, order and mesh, on the same device
        keys = ("operator", "order", "elements", "device", "driver")
        return tuple(row[key] for key in keys)

    for path in options.files:
        with open(path, newline="") as handle:
            for row in csv.DictReader(handle):
                if "bandwidth" in row:
                    copies[matched(row)] = float(row["bandwidth"])
                elif row["agree"] and not float(row["agree"]) <= AGREEMENT:
                    print(f"left out, it disagrees with numpy: {row['variant']}")
                else:
                    rows.append(row)

    # a table per operator and setup: rows of another mesh, device, loopy or
    # set of C flags are other measurements, not repeats
    def setup(row: dict[str, str]) -> tuple[str, ...]:
        flags = row.get("flags") or ""  # empty on a device
        return (row["elements"], row["device"], row["driver"], row["loopy"], flags)

    setups = list(dict.fromkeys(setup(r) for r in rows))
    for (name, op), this in ((item, s) for item in OPERATORS.items() for s in setups):
        mine = [r for r in rows if r["operator"] == name and setup(r) == this]
        if not mine:
            continue
        orders = sorted({r["order"] for r in mine}, key=int)
        names = list(dict.fromkeys(r["variant"] for r in mine))
        where = (
            f", {this[0]}**3 hexahedra on {this[1]}, loopy {this[3]}"
            + (f", built with {this[4]}" if this[4] else "")
            if len(setups) > 1
            else ""
        )
        print(f"\n{name}: GDOFs/s (% of the copy roof){where}\n")
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
                bandwidth = copies.get(matched(r))
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
        "mode", choices=["copy", "loopty", "c", "grudge", "fields", "agree", "report"]
    )
    parser.add_argument("files", nargs="*", help="the CSV files to report")
    parser.add_argument(
        "--elements",
        type=int,
        default=None,
        help="hexahedra per axis: 48 on a device, 16 on the C target",
    )
    parser.add_argument(
        "--cflags",
        default="",
        help="C target: flags after loopty's, the same for every program, "
        "written with '=' (--cflags=-march=native)",
    )
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
    if options.elements is None:
        options.elements = 16 if options.mode == "c" else 48
    started = time.perf_counter()
    globals()[f"mode_{options.mode}"](options)
    print(f"({options.mode}: {time.perf_counter() - started:.0f} s)", file=sys.stderr)


if __name__ == "__main__":
    main()
