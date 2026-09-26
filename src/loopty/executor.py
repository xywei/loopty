"""Running kernels: the loopy executor and the differential test.

The executor is an ordinary lanky executor: give it a kernel or a schedule and
numpy arguments and it runs them. Target ``c`` uses ``lp.ExecutableCTarget``,
which compiles and runs locally and is what the test suite and the demos use at
tiny sizes. Target ``opencl`` uses ``lp.PyOpenCLTarget`` and belongs on a machine
with a device; pyopencl is an optional extra for that reason, is imported inside
one branch of one function, and is never reached by importing loopty. loopy
itself is imported the first time something is lowered, not with this module:
lanky loads the executor through its entry point for every command it runs,
including a check of a file that has no kernels in it.

Executor options and kernel arguments are kept apart. Every positional and
keyword argument of :meth:`LoopyExecutor.run` is an argument of the kernel, so a
kernel whose parameter is called ``target`` receives it like any other. The
target is chosen by the schedule, ``Schedule(kernel, target="opencl")``, or by
the executor, ``LoopyExecutor(target="opencl")``, and never by a keyword of the
call.

``differential`` is the point of having two ways to run the same body. It
compares the transformed, compiled run against the native numpy run of the
original Python, and decides agreement using the exactness class the types state:
bitwise for ``exact``, a reassociation tolerance for ``reassoc``, the stated
epsilon for ``approx``. The comparison returns a fact, so "the compiled code
agrees with Python" lands in the ledger with the tolerance it was judged by, and
a schedule that reassociated an accumulation is judged by the wider tolerance it
asked for rather than being quietly forgiven.

The tolerance is per element, and depends on nothing but that element:

    exact                      a_k == b_k, bit for bit
    reassoc, approx            |a_k - b_k| <= eps_class * (|b_k| + FLOOR)

with ``b`` the expected output, ``eps_class`` from :data:`TOLERANCE` and
``FLOOR`` the absolute floor that keeps a value near zero from demanding a
tolerance of zero. Both live in :mod:`loopty.tolerance`, which the faithfulness
fact reads as well. Every element has to pass. The scale is deliberately local:
an earlier version scaled one tolerance by the 1-norm of the whole expected
output, which made a big output easy to agree with (a million ones bought a
tolerance of 1.0) and tied the verdict for one cell to values it has nothing to
do with. The ``difference ... within ...`` line a run prints reports the element
that came closest to its own allowance, so the two numbers beside each other are
a claim about one cell rather than about an average.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import numpy as np

from loopty.contract import check_arguments
from loopty.term import Term
from loopty.tolerance import (
    TOLERANCE,
    TOLERANCE_FLOOR,
    disagreement,
    output_class,
)

if TYPE_CHECKING:
    from loopty.lower import Lowering

__all__ = [
    "TOLERANCE",
    "LoopyExecutor",
    "agreement",
    "emit_code",
    "exactness_of_output",
]

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
    from loopty.lower import lower_generic

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
    """The executor loopty registers under ``lanky.executors``.

    ``target`` is the loopy target a kernel or a term is compiled for, ``"c"``
    when it is ``None``. A :class:`~loopty.schedule.Schedule` carries its own
    target, and an executor given a different one refuses to run it rather
    than rebuild it silently; ``schedule.retarget(...)`` is how a schedule
    changes target, with every cast checked again.

    The target is an option of the executor and not of a call because every
    keyword of :meth:`run` belongs to the kernel. It used to be popped from the
    keywords as ``target=``, which made a kernel parameter of that name
    unreachable by keyword: its value was taken for the name of a backend.
    """

    name = "loopy"

    def __init__(self, target: str | None = None) -> None:
        self.target = target

    def trust_class(self) -> str:
        """Running is evidence, not proof: an execution is a test."""
        return "test"

    def run(self, obj: Any, /, *args: Any, **kwargs: Any) -> dict[str, np.ndarray]:
        """Lower ``obj``, compile it for its target, and run it.

        Arguments are given positionally in the term's parameter order or by
        name, and every one of them is an argument of the kernel: the target is
        the executor's or the schedule's, never a keyword here. The result is a
        dictionary of the arrays the kernel writes; the arrays passed in are
        also updated in place, so that a kernel whose output is a parameter
        behaves the same way compiled as it does in Python.

        The arguments are checked against the term before anything is compiled
        or run: distinct array parameters may not share storage, a ragged
        argument has to agree with its counts family and with any offsets given
        alongside it, and an element of a refined sort has to be one. All three
        are properties of the call rather than of the term, and all three are
        what a typing rule assumed when it decided something; see
        :mod:`loopty.contract`.
        """
        term, kernel, lowering, target_name = _resolve(obj, self.target)
        names = [name for name, _ in term.params]
        supplied = {**dict(zip(names, args, strict=False)), **kwargs}
        check_arguments(dict(term.params), supplied, lowering.ragged)
        call = _call_arguments(term, lowering, args, kwargs)
        call, empty = _pad_empty_arrays(call, lowering)
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

        A plain ``ndarray`` needs this as much as an ``Arr`` does. It reaches
        loopy through :func:`_as_numpy`, which hands over the caller's own array
        only when it is already contiguous and of the lowered dtype; a strided
        view (``z[:, 0]``) or a ``float32`` output for a ``Real`` parameter is
        copied on the way in, and the results used to stay in that copy. Such
        an output is written back here, cast to the caller's dtype the way any
        assignment into it would be.
        """
        from loopty.arr import Arr

        for name, value in results.items():
            given = supplied.get(name)
            if isinstance(given, Arr):
                given.numpy()[...] = np.asarray(value).reshape(given.numpy().shape)
            elif isinstance(given, np.ndarray) and given is not value:
                result = np.asarray(value)
                if result.size == given.size:
                    given[...] = result.reshape(given.shape)

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
        """The code loopy generates, for ``--emit-code`` and for reading.

        ``target`` defaults to the executor's own.
        """
        return emit_code(obj, self.target if target is None else target)

    def differential(
        self,
        kernel: Any,
        schedule: Any,
        args: dict[str, Any],
        /,
        reference: dict[str, Any] | None = None,
    ) -> Any:
        """Compare a scheduled run with the native run; return the fact.

        ``kernel`` is the Python reference, called on copies of ``args``, unless
        ``reference`` gives the expected outputs outright. The tolerance is not a
        parameter: it is read from the exactness class of each output, widened by
        any accumulation the schedule marked reassociated, so the claim the fact
        records is "these agree to the accuracy the types promise".

        ``args`` is checked against the term before either run, for the reasons
        :meth:`run` gives and for one more: the comparison copies each argument
        separately, so an alias between two of them would be destroyed here and
        the two runs would agree about a program that races.

        An explicit ``reference`` has to cover *every* output of the lowering,
        exactly. It replaces the native run, so an output it omits is compared
        against nothing at all and the resulting ``TESTED`` fact would claim
        more than was tested; an output it names that the kernel does not write
        is a caller error worth saying out loud rather than ignoring.

        The kernel's arguments are ``args`` and nothing else; the scheduled run
        is on the schedule's target, which has to be the executor's when the
        executor names one.

        A :class:`~loopty.trace.TraceError` from the native run is not raised
        but returned, as a ``refuted`` agreement fact with the refusal as its
        ``reason``, and the scheduled run is not made: the body is refused for
        its spelling (a ``when`` guard whose native value is an integer, say),
        whatever the input, so there is nothing to compare the compiled run
        with. The faithfulness fact counts the same refusal as a disagreement
        (:mod:`loopty.faithful`). Anything else the body raises is the input's
        or the body's, and is raised.
        """
        term, _lowered, lowering, _target = _resolve(schedule, self.target)
        # Before the copies: ``_copy`` gives every argument a buffer of its own,
        # which is exactly what hides an alias between two of them, and the
        # native run would otherwise be the first thing to meet a bad index.
        check_arguments(dict(term.params), args, lowering.ragged)
        native = dict(reference or {})
        if native:
            missing = [name for name in lowering.outputs if name not in native]
            extra = [name for name in native if name not in lowering.outputs]
            if missing or extra:
                parts = []
                if missing:
                    parts.append(f"does not cover {', '.join(missing)}")
                if extra:
                    parts.append(f"names {', '.join(extra)}, which is not an output")
                raise ValueError(
                    f"the reference given for {term.name} " + " and ".join(parts)
                    + f"; {term.name} writes {', '.join(lowering.outputs)}, and "
                    "every one of them has to be compared or the agreement fact "
                    "would claim more than was tested"
                )
        if not native:
            native_args = {name: _copy(value) for name, value in args.items()}
            if callable(kernel):
                from loopty.trace import TraceError

                try:
                    kernel(**native_args)
                except TraceError as exc:
                    return _refused_agreement(term, schedule, exc)
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
        got = self.run(schedule, **scheduled_args)
        return agreement(term, schedule, got, native)


def exactness_of_output(term: Term, schedule: Any, name: str) -> str:
    """The exactness class the comparison of one output is judged by.

    The weakest of three: the class of the element sort, the class of the
    accumulation that writes it, and ``reassoc`` if the schedule realized that
    accumulation as a tree. Weakest wins because error does not cancel. See
    :func:`loopty.tolerance.output_class`, which the faithfulness fact asks
    too.
    """
    return output_class(term, name, getattr(schedule, "reassociated", frozenset()))


def _compare(
    got: np.ndarray, want: np.ndarray, exactness: str
) -> tuple[bool, float, float]:
    """``(agree, difference, tolerance)`` under one exactness class.

    The verdict is :func:`loopty.tolerance.disagreement`, the comparison the
    faithfulness fact makes too, cell by cell: the bits for ``exact``, so
    ``-0.0`` against ``0.0`` is a difference and two NaNs are not, and for the
    other classes ``|a_k - b_k| <= eps_class * (|b_k| + FLOOR)``, with a cell
    whose two values match (are equal, or are both NaN) agreeing whatever its
    allowance. That last clause is what an infinity both runs computed needs:
    ``inf - inf`` is NaN, and a NaN is within no allowance. The two arrays are
    compared in the type both promote to, so an ``int32`` output that loopy
    wrote agrees with the ``int64`` array the native run filled, as it always
    did.

    The two numbers returned describe the finite cell that came closest to
    failing, ties going to the one with the smaller allowance, so a run that
    prints them names the tightest case rather than an average. A cell that
    disagrees with a value that is not finite on either side is reported first,
    as a difference of ``inf`` beside that cell's allowance, which is zero when
    the expected value is the one that is not finite: no finite cell describes
    it.
    """
    got = np.asarray(got)
    want = np.asarray(want)
    if got.shape != want.shape:
        return False, float("inf"), 0.0
    common = np.result_type(got, want)
    got = got.astype(common, copy=False)
    want = want.astype(common, copy=False)
    differs = np.asarray(disagreement(got, want, exactness)).reshape(-1)
    agree = not bool(differs.any())
    if not want.size:
        return True, 0.0, 0.0
    epsilon = TOLERANCE[exactness]
    flat_got, flat_want = got.reshape(-1), want.reshape(-1)
    finite = np.isfinite(flat_got) & np.isfinite(flat_want)
    with np.errstate(invalid="ignore", over="ignore"):
        if common.kind in "fc":
            difference = np.abs(flat_got - flat_want)
        else:
            # Integers and booleans: numpy refuses to subtract booleans, and an
            # unsigned difference wraps around.
            difference = np.abs(
                flat_got.astype(np.float64) - flat_want.astype(np.float64)
            )
        allowed = epsilon * (np.abs(flat_want) + TOLERANCE_FLOOR)
    unmatched = np.flatnonzero(differs & ~finite)
    if unmatched.size:
        # The allowance of that cell, which an expected value that is not
        # finite does not have.
        k = int(unmatched[0])
        expected = bool(np.isfinite(flat_want[k]))
        return False, float("inf"), float(allowed[k]) if expected else 0.0
    if not finite.any():
        return agree, 0.0, 0.0
    if exactness == "exact":
        return agree, float(np.max(difference[finite])), 0.0
    difference, allowed = difference[finite], allowed[finite]
    ratio = np.divide(
        difference,
        allowed,
        out=np.where(difference > 0, np.inf, 0.0).astype(float),
        where=allowed > 0,
    )
    worst = np.flatnonzero(ratio == ratio.max())
    k = int(worst[int(np.argmin(allowed[worst]))])
    return agree, float(difference[k]), float(allowed[k])


def agreement(term: Term, schedule: Any, got: dict, want: dict) -> Any:
    """The fact recording whether two runs of a kernel agree.

    A refuted one says which outputs disagreed as its ``reason``, one line per
    output, which is what lanky prints under its ``REFUTED`` line: the
    difference and what was allowed, or the two shapes when they are not the
    same, since a difference between arrays of two shapes is not a number
    anyone can read (``outputs`` records it as infinite). The numbers of every
    output, agreeing or not, are in ``outputs``.
    """
    details: dict[str, Any] = {}
    disagreements: list[str] = []
    for name, want_array in want.items():
        exactness = exactness_of_output(term, schedule, name)
        agree, difference, tolerance = _compare(got[name], want_array, exactness)
        details[name] = {
            "exactness": exactness,
            "difference": difference,
            "tolerance": tolerance,
            "agree": agree,
        }
        if agree:
            continue
        shape = np.asarray(got[name]).shape
        native_shape = np.asarray(want_array).shape
        disagreements.append(
            f"{name} has shape {shape}, and the native run's has shape "
            f"{native_shape}"
            if shape != native_shape
            else f"{name} differs from the native run: difference "
            f"{difference:.3g}, allowed {tolerance:.3g} ({exactness})"
        )
    ok = not disagreements
    provenance = _agreement_provenance(schedule, details)
    if not ok:
        provenance["reason"] = "\n".join(disagreements)
    return _agreement_fact(term, schedule, ok, provenance)


def _refused_agreement(term: Term, schedule: Any, error: Exception) -> Any:
    """The agreement fact of a run whose native half is refused.

    ``refuted``, with the refusal as its ``reason`` and ``error``, and no
    outputs, because nothing was compared: a
    :class:`~loopty.trace.TraceError` refuses the body for its spelling,
    whatever the input, which is a disagreement between the body and what
    the compiled run computes, and not an input to skip.
    """
    text = f"{type(error).__name__}: {error}"
    provenance = _agreement_provenance(schedule, {})
    provenance["error"] = text
    provenance["reason"] = (
        f"the body, run natively, is refused, so there is no native run for "
        f"the scheduled run to agree with. {text}"
    )
    return _agreement_fact(term, schedule, False, provenance)


def _agreement_provenance(schedule: Any, details: dict[str, Any]) -> dict[str, Any]:
    """What every agreement fact records beside its verdict."""
    return {
        "outputs": details,
        "schedule": tuple(getattr(schedule, "history", ())),
        "target": getattr(schedule, "target", "c"),
    }


def _agreement_fact(
    term: Term, schedule: Any, ok: bool, provenance: dict[str, Any]
) -> Any:
    """The agreement fact itself, named after the schedule that was run.

    Its id is ``agreement:`` and the schedule's
    :attr:`~loopty.schedule.Schedule.key`, so that two schedules of one kernel
    run from one file keep two facts in the ledger; a kernel or a term run
    without a schedule is named by its name and target alone.
    """
    from lanky.ledger import Fact, Status

    key = getattr(schedule, "key", None)
    if not isinstance(key, str):
        key = f"{term.name}[{provenance['target']}]"
    return Fact(
        id=f"agreement:{key}",
        kind="agreement",
        statement=(
            f"the scheduled run of {term.name} agrees with the native run "
            "to the accuracy its types state"
        ),
        term=None,
        status=Status.TESTED if ok else Status.REFUTED,
        decided_by="loopy",
        provenance=provenance,
        where=term.stmts[0].where if term.stmts else "",
        owner=term.name,
    )


def emit_code(obj: Any, target: str | None = None) -> str:
    """The code loopy generates for a kernel or a schedule, as a string."""
    import loopy as lp

    _term, kernel, _lowering, _target = _resolve(obj, target)
    return lp.generate_code_v2(kernel).device_code()
