"""Running kernels: the loopy executor and the differential test.

The executor is an ordinary lanky executor: give it a kernel or a schedule and
numpy arguments and it runs them. Target ``c`` uses ``lp.ExecutableCTarget``,
which compiles and runs locally and is what the test suite and the demos use at
tiny sizes. Target ``opencl`` uses ``lp.PyOpenCLTarget`` and belongs on a machine
with a device; pyopencl is an optional extra for that reason, is imported inside
one branch of one function, and is never reached by importing loopty.

``differential`` is the point of having two ways to run the same body. It
compares the transformed, compiled run against the native numpy run of the
original Python, and decides agreement using the exactness class the types state:
bitwise for ``exact``, a reassociation tolerance for ``reassoc``, the stated
epsilon for ``approx``. The comparison returns a fact, so "the compiled code
agrees with Python" lands in the ledger with the tolerance it was judged by, and
a schedule that reassociated an accumulation is judged by the wider tolerance it
asked for rather than being quietly forgiven.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from loopty.lower import Lowering, lower_generic
from loopty.term import ArrType, Term

__all__ = [
    "TOLERANCE",
    "LoopyExecutor",
    "agreement",
    "emit_code",
    "exactness_of_output",
]

#: The relative tolerance each exactness class allows. ``exact`` means the bits:
#: no tolerance at all. ``reassoc`` is the room a different summation order
#: needs, and is scaled by the sum of the magnitudes involved, because that is
#: what bounds the error a reassociated sum can accumulate. ``approx`` is the
#: class of a type that never promised more than a few digits.
TOLERANCE = {"exact": 0.0, "reassoc": 1e-12, "approx": 1e-6}


def _resolve(obj: Any, target: str | None = None) -> tuple[Term, Any, Lowering, str]:
    """The term, the loopy kernel, the lowering and the target of ``obj``."""
    from loopty.schedule import Schedule

    if isinstance(obj, Schedule):
        if target is not None and target != obj.target:
            raise ValueError(
                f"schedule {obj!r} was built for target {obj.target!r}, "
                f"not {target!r}; use schedule.retarget({target!r}) to check "
                "every cast again against that target"
            )
        # Asked before loopy is, so that an accepted-but-unbuildable schedule
        # names the limit it hits instead of throwing from code generation.
        obj.require_buildable()
        return obj.term, obj.kernel, obj.lowering, obj.target
    term = obj if isinstance(obj, Term) else getattr(obj, "term", None)
    if term is None and hasattr(obj, "trace"):
        term = obj.trace()
    if not isinstance(term, Term):
        raise TypeError(f"{obj!r} is neither a kernel, a schedule, nor a term")
    lowering = lower_generic(term, target or "c")
    return term, lowering.kernel, lowering, target or "c"


def _as_numpy(value: Any, dtype: np.dtype | None = None) -> np.ndarray:
    """A numpy view of an argument, unwrapping a runtime :class:`~loopty.arr.Arr`."""
    from loopty.arr import Arr

    if isinstance(value, Arr):
        value = value.numpy()
    array = np.asarray(value)
    if dtype is not None and array.dtype != dtype:
        array = array.astype(dtype)
    return np.ascontiguousarray(array)


def _copy(value: Any) -> Any:
    """A private copy of an argument, so that a reference run cannot be seen.

    The differential test runs the same kernel twice, and a kernel writes into
    its arguments; without copies the second run would start from the first
    one's output. Runtime arrays are copied as what they are, values and offsets
    together.
    """
    from loopty.arr import Arr

    if isinstance(value, Arr):
        if value.is_ragged:
            return Arr(value.numpy().copy(), value.offsets.copy())
        return Arr(value.numpy().copy())
    if isinstance(value, np.ndarray):
        return value.copy()
    return value


def _call_arguments(
    term: Term, lowering: Lowering, args: tuple, kwargs: dict
) -> dict[str, np.ndarray]:
    """Match positional and keyword arguments to the term's parameters.

    A ragged :class:`~loopty.arr.Arr` supplies two arguments, its flat values and
    its offsets, which is the same splitting the lowering did to the type.
    """
    from loopty.arr import Arr

    names = [name for name, _ in term.params]
    supplied: dict[str, Any] = dict(zip(names, args, strict=False))
    for key, value in kwargs.items():
        supplied[key] = value

    dtypes = {
        arg.name: arg.dtype.numpy_dtype
        for arg in lowering.kernel.default_entrypoint.args
        if arg.dtype is not None
    }
    out: dict[str, np.ndarray] = {}
    for name, value in supplied.items():
        if isinstance(value, Arr) and value.is_ragged:
            offsets = lowering.ragged.get(name)
            if offsets is not None and offsets not in supplied:
                out[offsets] = _as_numpy(value.offsets, dtypes.get(offsets))
        if isinstance(value, np.ndarray | Arr):
            out[name] = _as_numpy(value, dtypes.get(name))
        else:
            out[name] = value
    return out


def _pad_empty_arrays(
    call: dict, lowering: Lowering
) -> tuple[dict, dict[str, np.ndarray]]:
    """Give a shapeless zero-length array one cell, and keep the original.

    A workaround for loopy 2025.2, kept local and narrow. Its C target passes an
    array argument as a pointer, and for an empty one it tries to pass a null
    pointer by calling the pointer type on ``0.0``
    (``loopy/target/c/c_execution.py``), which raises ``TypeError: expected
    c_double instead of float``. So a CSR matrix all of whose rows are empty
    cannot be run at all, which is the one case a ragged type most has to allow.

    Padding is sound precisely because the array is empty: no index into it is
    in bounds, so no generated loop can read or write the cell that was added.
    The originals are returned so that the caller can put them back in the
    results, where an empty output has to stay empty rather than become the
    one-cell pad; see ``docs/loopy-notes.md``.

    Only arguments loopy declares *without* a shape are padded, which for a
    lowered term means the flat buffer of a ragged axis and nothing else. An
    argument that does have a declared shape is how loopy infers the size
    parameters, and lengthening one would make it infer the wrong size. That
    leaves the fully degenerate case (a kernel run with zero rows, so that the
    shape-bearing arrays are empty too) still unrunnable on the C target, which
    is loopy's limit rather than something to paper over here.
    """
    shapeless = {
        arg.name
        for arg in lowering.kernel.default_entrypoint.args
        if getattr(arg, "shape", "missing") is None
    }
    originals: dict[str, np.ndarray] = {}
    out = dict(call)
    for name, value in call.items():
        if name in shapeless and isinstance(value, np.ndarray) and value.size == 0:
            originals[name] = value
            out[name] = np.zeros(1, dtype=value.dtype)
    return out, originals


class LoopyExecutor:
    """The executor loopty registers under ``lanky.executors``."""

    name = "loopy"

    def trust_class(self) -> str:
        """Running is evidence, not proof: an execution is a test."""
        return "test"

    def run(self, obj: Any, /, *args: Any, **kwargs: Any) -> dict[str, np.ndarray]:
        """Lower ``obj``, compile it for its target, and run it.

        Arguments are given positionally in the term's parameter order or by
        name. The result is a dictionary of the arrays the kernel writes; the
        arrays passed in are also updated in place, so that a kernel whose output
        is a parameter behaves the same way compiled as it does in Python.
        """
        target = kwargs.pop("target", None)
        term, kernel, lowering, target_name = _resolve(obj, target)
        call = _call_arguments(term, lowering, args, kwargs)
        call, empty = _pad_empty_arrays(call, lowering)
        names = [name for name, _ in term.params]
        supplied = {**dict(zip(names, args, strict=False)), **kwargs}
        if target_name == "opencl":
            out = self._run_opencl(kernel, lowering, call)
        else:
            # The executor object is where loopy keeps its compiled-code cache;
            # calling the translation unit directly recompiles, and says so.
            _event, results = kernel.executor()(**call)
            out = self._collect(lowering, call, results)
        for name, original in empty.items():
            # The pad is not a result: an empty output stays empty, so that a
            # caller comparing outputs sees the array it passed in.
            if name in out:
                out[name] = original
        self._write_back(supplied, out)
        return out

    def _collect(
        self, lowering: Lowering, call: dict, results: Any
    ) -> dict[str, np.ndarray]:
        """Name loopy's returned arrays, and write them back into the inputs."""
        if not isinstance(results, tuple):  # pragma: no cover - loopy returns tuples
            results = (results,)
        if len(results) != len(lowering.outputs):
            raise RuntimeError(
                f"{lowering.name} returned {len(results)} arrays for "
                f"{len(lowering.outputs)} outputs {lowering.outputs}"
            )
        out: dict[str, np.ndarray] = {}
        for name, value in zip(lowering.outputs, results, strict=True):
            out[name] = value
            given = call.get(name)
            if isinstance(given, np.ndarray) and given.shape == np.shape(value):
                given[...] = value
        return out

    @staticmethod
    def _write_back(supplied: dict, results: dict) -> None:
        """Copy the outputs into the arrays the caller handed in.

        Outputs are parameters, so a kernel run is expected to have changed what
        was passed to it. The compiled code writes into loopy's own buffers, and
        this puts the values where the caller is looking for them, including
        into a runtime :class:`~loopty.arr.Arr`.
        """
        from loopty.arr import Arr

        for name, value in results.items():
            given = supplied.get(name)
            if isinstance(given, Arr):
                given.numpy()[...] = np.asarray(value).reshape(given.numpy().shape)

    def _run_opencl(
        self, kernel: Any, lowering: Lowering, call: dict
    ) -> dict[str, np.ndarray]:
        """Run on a device. Imported here so that loopty never needs pyopencl."""
        import pyopencl as cl  # noqa: PLC0415 - deliberately local

        context = cl.create_some_context(interactive=False)
        queue = cl.CommandQueue(context)
        _event, results = kernel.executor(context)(queue, **call)
        return self._collect(lowering, call, results)

    def emit_code(self, obj: Any, target: str | None = None) -> str:
        """The code loopy generates, for ``--emit-code`` and for reading."""
        return emit_code(obj, target)

    def differential(
        self,
        kernel: Any,
        schedule: Any,
        args: dict[str, Any],
        /,
        reference: dict[str, Any] | None = None,
        **kwargs: Any,
    ) -> Any:
        """Compare a scheduled run with the native run; return the fact.

        ``kernel`` is the Python reference, called on copies of ``args``, unless
        ``reference`` gives the expected outputs outright. The tolerance is not a
        parameter: it is read from the exactness class of each output, widened by
        any accumulation the schedule marked reassociated, so the claim the fact
        records is "these agree to the accuracy the types promise".
        """
        term, _lowered, lowering, _target = _resolve(schedule)
        native = dict(reference or {})
        if not native:
            native_args = {name: _copy(value) for name, value in args.items()}
            if callable(kernel):
                kernel(**native_args)
                native_args = {
                    name: (
                        value.numpy()
                        if hasattr(value, "numpy") and not isinstance(value, np.ndarray)
                        else value
                    )
                    for name, value in native_args.items()
                }
            else:  # pragma: no cover - a reference is required in that case
                raise TypeError(
                    f"{kernel!r} is not callable and no reference was given"
                )
            native = {name: native_args[name] for name in lowering.outputs}
        scheduled_args = {name: _copy(value) for name, value in args.items()}
        got = self.run(schedule, **scheduled_args, **kwargs)
        return agreement(term, schedule, got, native)


def _element_class(dtype: Any) -> str:
    """The exactness class of an element type.

    A lanky sort states it (``Real`` is ``approx``, ``Nat`` and ``Int`` are
    ``exact``). A bare numpy dtype does not, so floating point is read as
    ``approx`` and everything else as ``exact``: a term whose element type is a
    plain ``float64`` has promised nothing about the last bits, and pretending
    otherwise would make a differential test that passes say more than it knows.
    """
    exactness = getattr(dtype, "exactness", None)
    if isinstance(exactness, str):
        return exactness
    try:
        kind = np.dtype(dtype).kind
    except TypeError:
        return "exact"
    return "approx" if kind in "fc" else "exact"


def exactness_of_output(term: Term, schedule: Any, name: str) -> str:
    """The exactness class the comparison of one output is judged by.

    The weakest of three: the class of the element sort, the class of the
    accumulation that writes it, and ``reassoc`` if the schedule realized that
    accumulation as a tree. Weakest wins because error does not cancel.
    """
    order = ["exact", "reassoc", "approx"]
    classes = ["exact"]
    reassociated = getattr(schedule, "reassociated", frozenset())
    if name in reassociated:
        classes.append("reassoc")
    for param, typ in term.params:
        if param != name or not isinstance(typ, ArrType):
            continue
        classes.append(_element_class(typ.dtype))
    from loopty.lower import reductions_of

    for stmt in term.stmts:
        if stmt.assignee.array != name:
            continue
        for reduction in reductions_of(stmt.expr):
            classes.append(reduction.exactness)
    return max(classes, key=order.index)


def _compare(
    got: np.ndarray, want: np.ndarray, exactness: str
) -> tuple[bool, float, float]:
    """``(agree, largest difference, tolerance)`` under one exactness class."""
    got = np.asarray(got)
    want = np.asarray(want)
    if got.shape != want.shape:
        return False, float("inf"), 0.0
    if exactness == "exact":
        return bool(np.array_equal(got, want)), float(
            np.max(np.abs(got - want)) if got.size else 0.0
        ), 0.0
    epsilon = TOLERANCE[exactness]
    magnitude = float(np.abs(want).sum()) if want.size else 0.0
    tolerance = epsilon * max(1.0, magnitude)
    difference = float(np.max(np.abs(got - want))) if got.size else 0.0
    return difference <= tolerance, difference, tolerance


def agreement(term: Term, schedule: Any, got: dict, want: dict) -> Any:
    """The fact recording whether two runs of a kernel agree."""
    from lanky.ledger import Fact, Status

    details: dict[str, Any] = {}
    ok = True
    for name, want_array in want.items():
        exactness = exactness_of_output(term, schedule, name)
        agree, difference, tolerance = _compare(got[name], want_array, exactness)
        ok = ok and agree
        details[name] = {
            "exactness": exactness,
            "difference": difference,
            "tolerance": tolerance,
            "agree": agree,
        }
    history = tuple(getattr(schedule, "history", ()))
    return Fact(
        id=f"agreement:{term.name}",
        kind="agreement",
        statement=(
            f"the scheduled run of {term.name} agrees with the native run "
            "to the accuracy its types state"
        ),
        term=None,
        status=Status.TESTED if ok else Status.REFUTED,
        decided_by="loopy",
        provenance={
            "outputs": details,
            "schedule": history,
            "target": getattr(schedule, "target", "c"),
        },
        where=term.stmts[0].where if term.stmts else "",
        owner=term.name,
    )


def emit_code(obj: Any, target: str | None = None) -> str:
    """The code loopy generates for a kernel or a schedule, as a string."""
    import loopy as lp

    _term, kernel, _lowering, _target = _resolve(obj, target)
    return lp.generate_code_v2(kernel).device_code()
