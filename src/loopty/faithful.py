"""The faithfulness fact: the traced term computes what the body computes.

A kernel has two meanings. The native run is the body on real arrays, the
reference implementation. The traced term is what the body did once, at a
generic point, and it is what every other fact is about and what lowering
compiles. Tracing refuses the ways a body can hide state from it that it knows
how to see (loop-carried names, containers, globals, whole-array operations,
side effects; see :mod:`loopty.trace`), and each of those checks has an edge.
The fact here is the check that has none, because it looks at results rather
than at spellings: whatever the body keeps where the tracer does not look shows
up as the two meanings disagreeing.

For each kernel, :func:`faithfulness_fact` runs the native body and the
interpreted term (:func:`loopty.interpret.interpret`) on the same inputs, each
on copies of its own, and compares every array argument afterwards. The inputs
are the module's example inputs, when it has any (the ``example_inputs()`` that
``loopty run`` reads, see :mod:`loopty.cli`), and :data:`SAMPLES` inputs drawn
from the declared types with every size at least 2, from a fixed seed. An
output is compared at its exactness class (:func:`loopty.tolerance.output_class`):
bit for bit when it is ``exact``, within the class's tolerance otherwise.

The fact, of kind ``trace-faithful``, is

* ``tested`` when every input either agreed or could not be run natively and
  at least one agreed,
* ``refuted`` at the first input that disagrees, with the input and the first
  differing cell, or with the error the term raised where the body ran,
* ``assumed``, with the reason, when no input ran natively or when the term
  holds something the interpreter has no meaning for.

An input the body itself cannot run (a sampled input can break an assumption
the types do not state, such as offsets consistent with counts) says nothing
either way, and is listed in the provenance as skipped.
"""

from __future__ import annotations

import warnings
from collections.abc import Iterator, Mapping
from typing import Any

import numpy as np
from lanky.ledger import Fact, Status
from lanky.prelude import FinType, Refined
from lanky.terms import evaluate, free_variables

from loopty.arr import Arr
from loopty.interpret import InterpretError, TooLarge, interpret
from loopty.term import ArrType, Term
from loopty.tolerance import disagreement, output_class

__all__ = [
    "KIND",
    "MAX_INSTANCES",
    "SAMPLES",
    "SEED",
    "SIZES",
    "faithfulness_fact",
    "sample_arguments",
]

#: The kind of the fact, as the ledger and its JSON name it.
KIND = "trace-faithful"

#: How many inputs are drawn from the declared types, besides the examples.
SAMPLES = 3

#: The seed of the draws. Sample ``k`` is drawn from ``default_rng((SEED, k))``,
#: so each one can be drawn again on its own from what the fact records.
SEED = 0

#: The half-open range every size is drawn from. At least 2, so that a loop
#: runs more than one iteration, which is where state carried from one
#: iteration to the next shows.
SIZES = (2, 5)

#: How many statement instances and reduction terms, together, one interpreted
#: run may have. The interpreter runs in Python, one instance at a time, and an
#: example input written for a benchmark is skipped rather than interpreted, or
#: run natively, for minutes.
MAX_INSTANCES = 200_000

#: What the statement of the fact says, for every kernel.
STATEMENT = "the traced term computes what the body computes"


class _NoSample(Exception):
    """A declared type this module cannot draw a value of."""


def faithfulness_fact(kernel: Any, term: Term, owner: str, where: str) -> Fact:
    """The ``trace-faithful`` fact of one kernel, established by running it.

    ``kernel`` is the decorated kernel, which is called as the native run, and
    ``term`` its traced term. Nothing here raises: an input that cannot be run
    is skipped, and anything unexpected leaves the fact ``assumed`` with the
    error as its reason.
    """
    identifier = f"{owner}:{KIND}"
    inputs: list[dict[str, Any]] = []

    def fact(status: Status, **provenance: Any) -> Fact:
        return Fact(
            id=identifier,
            kind=KIND,
            statement=STATEMENT,
            term=None,
            status=status,
            decided_by=None if status is Status.ASSUMED else "interpreter",
            provenance={
                "inputs": inputs,
                "seed": SEED,
                "samples": SAMPLES,
                **provenance,
            },
            where=where,
            owner=owner,
        )

    try:
        for label, arguments, recorded in _inputs(kernel, term):
            if isinstance(arguments, str):
                inputs.append({"input": label, "outcome": f"skipped: {arguments}"})
                continue
            outcome = _compare(kernel, term, label, arguments)
            if outcome is None:
                inputs.append({"input": label, "outcome": "agreed"})
                continue
            kind, detail = outcome
            if kind == "skipped":
                inputs.append({"input": label, "outcome": f"skipped: {detail}"})
                continue
            if kind == "unknown":
                return fact(Status.ASSUMED, reason=detail)
            counterexample, reason = detail
            inputs.append({"input": label, "outcome": "differed"})
            extra = {"arguments": recorded} if recorded is not None else {}
            return fact(
                Status.REFUTED, counterexample=counterexample, reason=reason, **extra
            )
    except Exception as exc:  # noqa: BLE001 - a fact, never a crash of the check
        return fact(
            Status.ASSUMED,
            reason=f"the comparison could not be made: {type(exc).__name__}: {exc}",
        )
    agreed = sum(entry["outcome"] == "agreed" for entry in inputs)
    if not agreed:
        skipped = "; ".join(f"{e['input']}: {e['outcome']}" for e in inputs)
        return fact(
            Status.ASSUMED,
            reason=f"no input ran natively, so nothing was compared ({skipped})",
        )
    return fact(Status.TESTED, compared=agreed)


# {{{ inputs


def _inputs(
    kernel: Any, term: Term
) -> Iterator[tuple[str, dict[str, Any] | str, dict[str, Any] | None]]:
    """``(label, arguments, record)`` for every input, examples first.

    ``arguments`` is a string instead when the input could not be made, saying
    why; ``record`` is what the provenance keeps of a sampled input, which is
    small enough to write down, and ``None`` for an example.
    """
    from loopty.cli import EXAMPLE_FUNCTION, select_inputs

    factory = getattr(kernel.fn, "__globals__", {}).get(EXAMPLE_FUNCTION)
    label = f"{EXAMPLE_FUNCTION}()"
    if callable(factory):
        try:
            everything = factory()
        except Exception as exc:  # noqa: BLE001 - the module's code, reported
            yield label, f"{label} raised {type(exc).__name__}: {exc}", None
        else:
            if not isinstance(everything, dict):
                yield label, f"{label} did not return a dictionary", None
            else:
                chosen = select_inputs(everything, kernel.__name__)
                if chosen is not None:
                    yield label, chosen, None
    for index in range(SAMPLES):
        try:
            arguments, sizes = sample_arguments(term, (SEED, index))
        except _NoSample as exc:
            yield f"sample {index + 1}", str(exc), None
            continue
        shown = ", ".join(f"{name}={value}" for name, value in sorted(sizes.items()))
        label = f"sample {index + 1}" + (f" ({shown})" if shown else "")
        yield label, arguments, _record(arguments)


def _record(arguments: Mapping[str, Any]) -> dict[str, Any]:
    """A sampled input, written down for the provenance."""
    out: dict[str, Any] = {}
    for name, value in arguments.items():
        if isinstance(value, Arr):
            out[name] = value.numpy().tolist()
            if value.is_ragged:
                out[f"{name} offsets"] = value.offsets.tolist()
        else:
            out[name] = value
    return out


def sample_arguments(
    term: Term, seed: Any = SEED
) -> tuple[dict[str, Any], dict[str, int]]:
    """Arguments for ``term`` drawn from its declared types, with their sizes.

    Every size a shape or a sort names is drawn from :data:`SIZES`, and every
    array has the extents its axes then evaluate to, with a ragged axis taking
    its row lengths from the counts array it names. Elements are drawn by sort:
    a point of ``Fin[m]`` below ``m``, a ``Nat`` below 4 (which keeps ragged
    rows short, and allows empty ones), an ``Int`` between -3 and 3, a ``Real``
    from a standard normal. Arrays the kernel writes are drawn too, because a
    kernel may read what it writes before writing it, and both runs have to
    start from the same values. An integral scalar named like a size is that
    size.
    """
    rng = np.random.default_rng(seed)
    types = dict(term.params)
    names = sorted((_size_names(term) - set(types)) | _integral_scalars(term))
    sizes = {name: int(rng.integers(*SIZES)) for name in names}
    arguments: dict[str, Any] = {}
    ragged: list[tuple[str, ArrType]] = []
    for name, typ in term.params:
        if isinstance(typ, ArrType):
            if any(typ.ragged):
                ragged.append((name, typ))
                continue
            shape = tuple(_extent(axis, sizes) for axis in typ.axes)
            arguments[name] = Arr.from_numpy(_draw(typ.dtype, shape, sizes, rng))
        elif name in sizes:
            arguments[name] = sizes[name]
        else:
            arguments[name] = _draw(typ, (), sizes, rng).item()
    for name, typ in ragged:
        if typ.ragged != (False, True):
            raise _NoSample(f"{name} is ragged below its second axis")
        counts_name = getattr(typ.axes[1], "name", None)
        counts = arguments.get(counts_name)
        if not isinstance(counts, Arr) or counts.numpy().ndim != 1:
            raise _NoSample(f"the row lengths of {name} are not a drawn array")
        lengths = counts.numpy()
        if lengths.dtype.kind not in "iu" or np.any(lengths < 0):
            raise _NoSample(f"the row lengths {counts_name} of {name} are negative")
        values = _draw(typ.dtype, (int(lengths.sum()),), sizes, rng)
        arguments[name] = Arr.ragged(lengths, values=values)
    ordered = {name: arguments[name] for name, _typ in term.params}
    return ordered, sizes


def _size_names(term: Term) -> set[str]:
    """Every name a shape or a sort of ``term`` mentions."""
    out = set(term.sizes)
    for _name, typ in term.params:
        sort = typ.dtype if isinstance(typ, ArrType) else typ
        base = sort.base if isinstance(sort, Refined) else sort
        if isinstance(base, FinType):
            out |= set(free_variables(base.bound))
        if isinstance(typ, ArrType):
            for axis in typ.axes:
                out |= set(free_variables(axis))
    return out


def _integral_scalars(term: Term) -> set[str]:
    """The scalar parameters of an integral sort that a shape or a sort names."""
    from loopty.contract import integral_sort

    named = _size_names(term)
    return {
        name
        for name, typ in term.params
        if not isinstance(typ, ArrType) and name in named and integral_sort(typ)
    }


def _extent(axis: Any, sizes: Mapping[str, int]) -> int:
    """The number of cells of one dense axis at the drawn sizes."""
    try:
        value = evaluate(axis, dict(sizes))
    except Exception as exc:  # noqa: BLE001 - reported as a sample not drawn
        raise _NoSample(f"the axis {axis} has no value at {dict(sizes)}") from exc
    if isinstance(value, bool) or not isinstance(value, int | np.integer):
        raise _NoSample(f"the axis {axis} is {value!r} at {dict(sizes)}")
    if value < 0:
        raise _NoSample(f"the axis {axis} is negative at {dict(sizes)}")
    return int(value)


def _draw(
    sort: Any, shape: tuple[int, ...], sizes: Mapping[str, int], rng: Any
) -> np.ndarray:
    """Values of one sort, of one shape; see :func:`sample_arguments`."""
    base = sort.base if isinstance(sort, Refined) else sort
    if isinstance(base, FinType):
        bound = _extent(base.bound, sizes)
        if bound == 0:
            if int(np.prod(shape)) == 0:
                return np.zeros(shape, dtype=np.int64)
            raise _NoSample(f"{base} has no points at {dict(sizes)}")
        return rng.integers(0, bound, size=shape, dtype=np.int64)
    name = getattr(base, "name", None)
    if name == "Nat":
        return rng.integers(0, 4, size=shape, dtype=np.int64)
    if name == "Int":
        return rng.integers(-3, 4, size=shape, dtype=np.int64)
    if name == "Bool":
        return rng.integers(0, 2, size=shape).astype(bool)
    if name == "Real":
        return rng.standard_normal(shape)
    try:
        dtype = np.dtype(base)
    except TypeError as exc:
        raise _NoSample(f"no way to draw a value of the sort {base}") from exc
    if dtype.kind == "f":
        return rng.standard_normal(shape).astype(dtype)
    if dtype.kind == "c":
        pair = rng.standard_normal((*shape, 2))
        return (pair[..., 0] + 1j * pair[..., 1]).astype(dtype)
    if dtype.kind == "b":
        return rng.integers(0, 2, size=shape).astype(bool)
    if dtype.kind == "u":
        return rng.integers(0, 4, size=shape).astype(dtype)
    if dtype.kind == "i":
        return rng.integers(-3, 4, size=shape).astype(dtype)
    raise _NoSample(f"no way to draw a value of the sort {base}")


# }}}


# {{{ one comparison


def _copy(value: Any) -> Any:
    """A private copy of one argument, so that the two runs cannot see each other."""
    if isinstance(value, Arr):
        if value.is_ragged:
            return Arr(value.numpy().copy(), value.offsets.copy())
        return Arr(value.numpy().copy())
    if isinstance(value, np.ndarray):
        return value.copy()
    return value


def _buffer(value: Any) -> np.ndarray | None:
    """The numpy buffer of an array argument, or ``None`` for a scalar."""
    if isinstance(value, Arr):
        return value.numpy()
    if isinstance(value, np.ndarray):
        return value
    return None


def _python(value: Any) -> Any:
    """A numpy scalar as the Python number it is, for a message and for JSON."""
    return value.item() if isinstance(value, np.generic) else value


def _cell(name: str, value: Any, position: int) -> str:
    """``y[1, 0]``: the cell at a flat position of an argument."""
    from loopty.contract import _cell_label

    return _cell_label(name, value, position)


def _compare(
    kernel: Any, term: Term, label: str, arguments: Mapping[str, Any]
) -> tuple[str, Any] | None:
    """Run both meanings on one input; ``None`` when they agree.

    Otherwise ``("skipped", why)`` when the body cannot run the input or the
    input is too large to interpret, ``("unknown", why)`` when the term cannot
    be interpreted at all, and ``("differ", (counterexample, reason))`` when
    the two disagree.

    The interpreter runs first, because it is the one with a bound on its work
    (:data:`MAX_INSTANCES`): an example input written for a benchmark is
    skipped without running the body on it either. What it raised is judged
    only once the body has run, since an input the body refuses says nothing
    about the term.
    """
    native = {name: _copy(value) for name, value in arguments.items()}
    interpreted = {name: _copy(value) for name, value in arguments.items()}
    with np.errstate(all="ignore"), warnings.catch_warnings():
        warnings.simplefilter("ignore")
        failure: Exception | None = None
        try:
            interpret(term, interpreted, limit=MAX_INSTANCES)
        except TooLarge as exc:
            return "skipped", f"too large to interpret: {exc}"
        except Exception as exc:  # noqa: BLE001 - judged below, after the body
            failure = exc
        try:
            kernel(**native)
        except Exception as exc:  # noqa: BLE001 - an input the body refuses
            return "skipped", f"the body raised {type(exc).__name__}: {exc}"
    if isinstance(failure, InterpretError):
        return "unknown", f"the term cannot be interpreted: {failure}"
    if isinstance(failure, IndexError | ArithmeticError):
        # What a program raises: a cell that is not there, a division by zero.
        # The body ran the same input without either.
        error = f"{type(failure).__name__}: {failure}"
        return "differ", (
            {"input": label, "term raised": error},
            f"on {label} the body runs and the traced term, interpreted, "
            f"raises {error}, so the term is not what the body computes",
        )
    if failure is not None:
        # The interpreter's own failure, not the term's.
        return "unknown", (
            f"the interpreter failed on {label}: {type(failure).__name__}: {failure}"
        )
    for name, typ in term.params:
        if not isinstance(typ, ArrType):
            continue
        want, got = _buffer(native[name]), _buffer(interpreted[name])
        if want is None or got is None:
            continue
        exactness = output_class(term, name)
        mask = disagreement(got, want, exactness).reshape(-1)
        if not mask.any():
            continue
        position = int(np.flatnonzero(mask)[0])
        cell = _cell(name, native[name], position)
        body = _python(want.reshape(-1)[position])
        traced = _python(got.reshape(-1)[position])
        how = (
            "bit for bit"
            if exactness == "exact"
            else f"at the tolerance of the {exactness} class"
        )
        return "differ", (
            {"input": label, "cell": cell, "body": body, "term": traced},
            f"on {label}, {cell} is {body!r} after the native run and "
            f"{traced!r} after the traced term, interpreted, and the two are "
            f"compared {how}. The term does not compute what the body computes: "
            "tracing runs the body once, at one generic point, so state the "
            "body keeps outside the arrays, or reaches in a way the trace does "
            "not follow, is seen at one iteration only.",
        )
    return None


# }}}
