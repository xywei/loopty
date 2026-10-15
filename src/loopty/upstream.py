"""Agreement with the code a kernel mirrors: a row of the ledger, tested.

A kernel is often written after someone else's: ``examples/l2p.py`` is
sumpy's L2P, ``E2PFromSingleBox``. That the two compute the same thing is a
claim like any other, and it belongs in the ledger beside the kernel's facts,
with the status it has earned. ``@upstream(kernel)`` names a Python function
that computes, on the kernel's example inputs, what the code the kernel
mirrors computes there::

    @upstream(l2p)
    def sumpy_l2p(coeffs, cx, cy, cnt, tx, ty, pot, rscale):
        '''sumpy's own E2PFromSingleBox, run on these inputs and recorded.'''
        return {"pot": recorded_potential()}

The function takes the kernel's arguments, by name, as the file's
``example_inputs()`` gives them (copies, so it may not change what the kernel
reads), and returns a dictionary from the names of the arrays the kernel
writes to what the code it mirrors leaves in them: an array of the kernel's
cells, in the array's own shape for a dense array, flat for a ragged array
as :meth:`loopty.arr.Arr.numpy` gives it, or in the order of
:meth:`loopty.arr.Arr.cells` for an array over a domain. An array of another
shape is refuted, with both shapes, even where its cells would agree.

``lanky check`` runs the kernel natively on the same inputs and compares,
output by output, at the accuracy the kernel's types claim for that output
(:func:`loopty.tolerance.output_class`), cell by cell as the differential
test does. The outputs compared are the ones the function gives, and the
claim names them (``the native run of l2p agrees with sumpy_l2p in pot, on
its example inputs``); an array the kernel writes that the function leaves
out is not claimed, and the provenance lists it (``uncompared``).

Two codes that compute one thing in two orders agree to round-off, far
inside the allowance of ``approx``, a millionth of each value. The claim
can say so: ``@upstream(single_layer, within=1e-12)`` compares each cell of
an output computed in floating point within ``1e-12`` times the largest
value the function returned for it, and the claim names that allowance
(``the native run of single_layer agrees with sumpy_single_layer in out
within 1e-12 of the largest value, on its example inputs``), so the ledger
says how close the two are and not only that they agree to the class. An
output of an exact class is still compared bit for bit, which no allowance
loosens. The fact reads

* ``tested`` by ``upstream`` when every output agrees, with the largest
  difference and its allowance in the provenance;
* ``refuted`` when one does not, with the output, the cell and both values;
* ``assumed`` when the function could not run here: its package is not
  installed (``ImportError``), or what it reads is not there
  (``FileNotFoundError``), or the file has no example inputs for the kernel.
  The reason says which, so a ledger read in an environment without the
  upstream code says that the agreement was not checked there, not that it
  holds.

The comparison is the file's example inputs only: a tested row is evidence
on those inputs, and its provenance says which they were.
"""

from __future__ import annotations

import functools
import os
from collections.abc import Mapping, Sequence
from typing import Any

import numpy as np
from lanky.ledger import Fact, Status, fact_id
from lanky.plugins import registry

__all__ = ["KIND", "Upstream", "upstream"]

#: The kind of the fact an :class:`Upstream` claims.
KIND = "upstream"

#: What is not the function's fault: the code it runs or reads is not here.
_UNAVAILABLE = (ImportError, FileNotFoundError)


class Upstream:
    """The claim that a kernel computes what ``fn`` says the code it mirrors does.

    Made by :func:`upstream`. ``fn`` is the function, and ``kernel`` the
    decorated kernel or program it is compared with. Its one fact is
    :meth:`facts`; see the module.
    """

    def __init__(self, kernel: Any, fn: Any, within: float | None = None) -> None:
        from loopty.kernel import Kernel, Program

        if not isinstance(kernel, Kernel | Program):
            raise TypeError(
                f"@upstream takes the kernel the function is compared with, "
                f"@upstream(l2p), not {kernel!r}"
            )
        if not callable(fn):
            raise TypeError(f"@upstream({kernel.__name__}) decorates a function")
        self.kernel = kernel
        self.fn = fn
        #: The allowance the claim states, a fraction of the largest value of
        #: an output, or ``None`` for the class's (see the module).
        self.within = _allowance(kernel.__name__, within)
        functools.update_wrapper(self, fn)
        code = fn.__code__
        self.path = code.co_filename
        self.line = code.co_firstlineno
        self.where = f"{os.path.basename(self.path)}:{self.line}"
        from lanky.check import module_name

        self.module = module_name(self.path) or getattr(fn, "__module__", "") or ""
        doc = (fn.__doc__ or "").strip()
        self.description = doc.splitlines()[0] if doc else ""
        self._facts: tuple[Fact, ...] | None = None

    def __call__(self, *args: Any, **kwargs: Any) -> Any:
        """The function itself, as it was written."""
        return self.fn(*args, **kwargs)

    def __repr__(self) -> str:
        name, of = self.fn.__name__, self.kernel.__name__
        return f"<upstream {name} of {of} at {self.where}>"

    @property
    def statement(self) -> str:
        """``the native run of l2p agrees with sumpy_l2p on its example inputs``."""
        return self.compared_statement(())

    def compared_statement(self, outputs: Sequence[str]) -> str:
        """The claim, naming the outputs compared when they are known.

        ``the native run of l2p agrees with sumpy_l2p in pot, on its example
        inputs``: only the outputs the function gave are compared, and the
        claim says which, so that one it left out is not claimed.
        """
        compared = f" in {', '.join(outputs)}" if outputs else ""
        if self.within is not None:
            compared += f" within {self.within:g} of the largest value"
        if compared:
            compared += ","
        return (
            f"the native run of {self.kernel.__name__} agrees with "
            f"{self.fn.__name__}{compared} on its example inputs"
        )

    def facts(self) -> tuple[Fact, ...]:
        """The one fact: the native run and the function's outputs compared."""
        if self._facts is None:
            self._facts = (self._fact(),)
        return self._facts

    # {{{ the comparison

    def _fact(self) -> Fact:
        identifier = fact_id(
            KIND,
            self.kernel.__name__,
            module=self.module,
            line=self.line,
            detail=self.fn.__name__,
        )
        base = {
            "path": self.path,
            "line": self.line,
            "upstream": self.fn.__name__,
            "description": self.description,
        }

        def fact(
            status: Status, compared: Sequence[str] = (), **provenance: Any
        ) -> Fact:
            return Fact(
                id=identifier,
                kind=KIND,
                statement=self.compared_statement(compared),
                term=None,
                status=status,
                decided_by=None if status is Status.ASSUMED else KIND,
                provenance={**base, **provenance},
                where=self.where,
                owner=self.kernel.__name__,
            )

        arguments = self._example()
        if isinstance(arguments, str):
            return fact(Status.ASSUMED, reason=arguments)
        from loopty.executor import _copy

        try:
            given = {name: _copy(value) for name, value in arguments.items()}
            expected = self.fn(**given)
        except _UNAVAILABLE as exc:
            return fact(
                Status.ASSUMED,
                reason=(
                    f"{self.fn.__name__} could not run here, so the agreement was "
                    f"not checked: {type(exc).__name__}: {exc}"
                ),
            )
        except Exception as exc:  # noqa: BLE001 - the file's code, reported
            return fact(
                Status.ASSUMED,
                reason=f"{self.fn.__name__} raised {type(exc).__name__}: {exc}",
            )
        if not isinstance(expected, Mapping) or not expected:
            return fact(
                Status.ASSUMED,
                reason=(
                    f"{self.fn.__name__} returned {type(expected).__name__}, not a "
                    "dictionary from the names of the arrays the kernel writes to "
                    "their values"
                ),
            )
        try:
            native = {name: _copy(value) for name, value in arguments.items()}
            self.kernel(**native)
        except Exception as exc:  # noqa: BLE001 - the kernel's run, reported
            return fact(
                Status.ASSUMED,
                reason=(
                    f"the native run of {self.kernel.__name__} raised "
                    f"{type(exc).__name__}: {exc}"
                ),
            )
        return self._compared(native, expected, fact)

    def _example(self) -> dict[str, Any] | str:
        """The kernel's example inputs, or why there are none."""
        from loopty.cli import EXAMPLE_FUNCTION, select_inputs

        factory = getattr(self.kernel.fn, "__globals__", {}).get(EXAMPLE_FUNCTION)
        if not callable(factory):
            return (
                f"the file has no {EXAMPLE_FUNCTION}() to run {self.kernel.__name__} "
                f"and {self.fn.__name__} on"
            )
        try:
            everything = factory()
        except Exception as exc:  # noqa: BLE001 - the module's code, reported
            return f"{EXAMPLE_FUNCTION}() raised {type(exc).__name__}: {exc}"
        if not isinstance(everything, dict):
            return f"{EXAMPLE_FUNCTION}() did not return a dictionary"
        chosen = select_inputs(everything, self.kernel.__name__)
        if chosen is None:
            return f"{EXAMPLE_FUNCTION}() has no inputs for {self.kernel.__name__}"
        return dict(chosen)

    def _compared(
        self, native: Mapping[str, Any], expected: Mapping[str, Any], fact: Any
    ) -> Fact:
        from loopty.arr import Arr
        from loopty.executor import _compare
        from loopty.tolerance import output_class

        term = self.kernel.term
        written = {stmt.assignee.array for stmt in term.stmts}
        outputs: dict[str, Any] = {}
        for name, value in expected.items():
            if name not in native or name not in written:
                return fact(
                    Status.ASSUMED,
                    reason=(
                        f"{self.fn.__name__} returned {name!r}, which "
                        f"{self.kernel.__name__} does not write"
                    ),
                )
            got = native[name]
            if isinstance(got, Arr):
                got = got.cells() if got.domain is not None else got.numpy()
            got = np.asarray(got)
            want = np.asarray(value)
            # the shapes as given: an array of another shape is not the
            # kernel's output, even where its cells agree flattened, a (2, 3)
            # output returned as (6,) say
            shapes = [list(got.shape), list(want.shape)]
            got, want = got.reshape(-1), want.reshape(-1)
            exactness = output_class(term, name)
            within = self.within if exactness != "exact" else None
            if within is None:
                agree, difference, tolerance = _compare(got, want, exactness)
            else:
                agree, difference, tolerance = _within(got, want, within)
            agree = agree and shapes[0] == shapes[1]
            detail: dict[str, Any] = {
                "exactness": exactness,
                "difference": difference,
                "tolerance": tolerance,
                "largest": float(np.max(np.abs(want))) if want.size else 0.0,
                "agree": agree,
            }
            if within is not None:
                detail["within"] = within
            if shapes[0] != shapes[1]:
                detail["shapes"] = shapes
            elif not agree:
                from loopty.tolerance import disagreement

                differs = (
                    disagreement(got, want, exactness)
                    if within is None
                    else _beyond(got, want, tolerance)
                )
                k = int(np.flatnonzero(differs)[0])
                detail["cell"] = k
                detail["native"] = got[k].item()
                detail[self.fn.__name__] = want[k].item()
            outputs[name] = detail
        # what the kernel writes that its caller sees, and the function left
        # out: not compared, which the claim's words and provenance say
        params = set(term.param_names)
        uncompared = sorted((written & params) - set(outputs))
        compared = tuple(outputs)
        failed = [name for name, detail in outputs.items() if not detail["agree"]]
        if failed:
            lines = []
            for name in failed:
                detail = outputs[name]
                if "shapes" in detail:
                    lines.append(
                        f"{name}: {detail['shapes'][0]} cells natively, "
                        f"{detail['shapes'][1]} from {self.fn.__name__}"
                    )
                else:
                    allowance = (
                        f"{detail['within']:g} of the largest value"
                        if "within" in detail
                        else f"the {detail['exactness']} allowance"
                    )
                    lines.append(
                        f"{name}: cell {detail['cell']} is {detail['native']!r} "
                        f"natively and {detail[self.fn.__name__]!r} from "
                        f"{self.fn.__name__}, beyond {allowance}"
                    )
            return fact(
                Status.REFUTED,
                compared,
                outputs=outputs,
                uncompared=uncompared,
                reason="; ".join(lines),
            )
        return fact(Status.TESTED, compared, outputs=outputs, uncompared=uncompared)

    # }}}


def _allowance(name: str, within: Any) -> float | None:
    """``within`` as the allowance a claim states, or why it cannot be one.

    Raises:
        TypeError: If it is not a number.
        ValueError: If it is not a finite number above zero.
    """
    if within is None:
        return None
    if isinstance(within, bool) or not isinstance(within, int | float | np.floating):
        raise TypeError(
            f"@upstream({name}, within=...) takes a number, the allowance as a "
            f"fraction of the largest value, such as within=1e-12, not {within!r}"
        )
    if not np.isfinite(within) or within <= 0:
        raise ValueError(
            f"@upstream({name}, within={within!r}) allows nothing a run can meet; "
            "the allowance is a fraction of the largest value above zero, such as "
            "within=1e-12"
        )
    return float(within)


def _within(
    got: np.ndarray, want: np.ndarray, within: float
) -> tuple[bool, float, float]:
    """``(agree, difference, allowance)``: each cell within ``within`` of the largest.

    The allowance is ``within`` times the largest modulus of ``want``, one
    number for the whole output, which is what round-off between two orders
    of summation is measured against. A cell whose two values match (are
    equal, or are both NaN) agrees, and one that is not finite on either side
    otherwise does not, as :func:`loopty.tolerance.disagreement` has it.
    """
    common = np.result_type(got, want)
    got = got.astype(common, copy=False)
    want = want.astype(common, copy=False)
    if not want.size:
        return True, 0.0, 0.0
    finite = np.isfinite(want)
    largest = float(np.max(np.abs(want[finite]))) if finite.any() else 0.0
    allowance = within * largest
    differs = _beyond(got, want, allowance)
    with np.errstate(invalid="ignore", over="ignore"):
        gaps = np.abs(got - want)
    shown = gaps[np.isfinite(gaps)]
    difference = float(shown.max()) if shown.size else 0.0
    if differs.any() and not np.isfinite(gaps[differs]).all():
        difference = float("inf")
    return not bool(differs.any()), difference, allowance


def _beyond(got: np.ndarray, want: np.ndarray, allowance: float) -> np.ndarray:
    """The cells at which ``got`` is not within ``allowance`` of ``want``, as a mask.

    Two values match when they are equal or both NaN, a complex one part by
    part, as :func:`loopty.tolerance.disagreement` has it: ``nan + 1j`` and
    ``nan + 2j`` do not.
    """
    parts = [
        np.stack([side.real, side.imag], axis=-1)
        if side.dtype.kind == "c"
        else side[..., np.newaxis]
        for side in (got, want)
    ]
    with np.errstate(invalid="ignore", over="ignore"):
        matched = np.all(
            (parts[0] == parts[1]) | (np.isnan(parts[0]) & np.isnan(parts[1])),
            axis=-1,
        )
        near = np.isfinite(want) & (np.abs(got - want) <= allowance)
    return ~(matched | near)


def upstream(kernel: Any, *, within: float | None = None) -> Any:
    """``@upstream(kernel)``: the function computes what the code ``kernel`` mirrors.

    See the module. ``within=1e-12`` states the allowance in the claim, a
    fraction of the largest value of each output, where the class's is the
    default. The decorated name is the :class:`Upstream`, which calls the
    function when called, and whose fact ``lanky check`` lists under the
    kernel.
    """
    allowance = _allowance(getattr(kernel, "__name__", "kernel"), within)

    def decorate(fn: Any) -> Upstream:
        from loopty.kernel import ensure_registered

        ensure_registered()
        return registry.register_object(Upstream(kernel, fn, allowance))

    return decorate
