"""What "the same result" means, per exactness class, for any two runs of a kernel.

Two comparisons judge a kernel by what its types promise. The differential test
(:meth:`loopty.executor.LoopyExecutor.differential`) compares a compiled run
with the native run, and the faithfulness fact (:mod:`loopty.faithful`)
compares the native run with the traced term, interpreted. Both read the
tolerance of an output off the same place, which is here, so that neither
depends on the other and the faithfulness fact does not need loopy.

The tolerance is per element, and depends on nothing but that element:

    exact                      a_k == b_k, bit for bit
    reassoc, approx            |a_k - b_k| <= eps_class * (|b_k| + FLOOR)

with ``b`` the expected output, ``eps_class`` from :data:`TOLERANCE` and
``FLOOR`` the absolute floor :data:`TOLERANCE_FLOOR`.
"""

from __future__ import annotations

from collections.abc import Collection
from typing import Any

import numpy as np

from loopty.term import ArrType, Term
from loopty.trace import EXACTNESS_ORDER, reductions_in

__all__ = [
    "TOLERANCE",
    "TOLERANCE_FLOOR",
    "disagreement",
    "element_class",
    "output_class",
]

#: The relative tolerance each exactness class allows, per element. ``exact``
#: means the bits: no tolerance at all. ``reassoc`` is the room a different
#: summation order needs. ``approx`` is the class of a type that never promised
#: more than a few digits.
TOLERANCE = {"exact": 0.0, "reassoc": 1e-12, "approx": 1e-6}

#: The absolute floor added to an element's own magnitude before scaling by the
#: class epsilon. Without it an expected value of exactly zero would demand a
#: difference of exactly zero from a class that never promised one; with it, the
#: allowance for such a value is ``eps_class`` itself. It is not a free
#: parameter to tune away a failure: it sets the scale at which "near zero"
#: starts, and 1.0 is the scale of a normalized quantity.
TOLERANCE_FLOOR = 1.0


def element_class(dtype: Any) -> str:
    """The exactness class of an element type.

    A lanky sort states it (``Real`` is ``approx``, ``Nat`` and ``Int`` are
    ``exact``). A bare numpy dtype does not, so floating point is read as
    ``approx`` and everything else as ``exact``: a term whose element type is a
    plain ``float64`` has promised nothing about the last bits, and pretending
    otherwise would make a comparison that passes say more than it knows.
    """
    exactness = getattr(dtype, "exactness", None)
    if isinstance(exactness, str):
        return exactness
    try:
        kind = np.dtype(dtype).kind
    except TypeError:
        return "exact"
    return "approx" if kind in "fc" else "exact"


def output_class(
    term: Term, name: str, reassociated: Collection[str] = frozenset()
) -> str:
    """The exactness class the comparison of one array of ``term`` is judged by.

    The weakest of three: the class of the element sort, the class of every
    accumulation that writes the array, and ``reassoc`` if ``name`` is among
    the ``reassociated`` outputs, which is what a schedule's ``realize(...,
    tree=True)`` records. Weakest wins because error does not cancel.
    """
    classes = ["exact"]
    if name in reassociated:
        classes.append("reassoc")
    for param, typ in term.params:
        if param != name or not isinstance(typ, ArrType):
            continue
        classes.append(element_class(typ.dtype))
    for stmt in term.stmts:
        if stmt.assignee.array != name:
            continue
        for reduction in reductions_in(stmt.expr):
            classes.append(reduction.exactness)
    known = [c for c in classes if c in EXACTNESS_ORDER]
    return max(known, key=EXACTNESS_ORDER.index)


def disagreement(got: Any, want: Any, exactness: str) -> np.ndarray:
    """The cells at which ``got`` does not agree with ``want``, as a mask.

    ``exact`` compares the bits of each cell, so ``-0.0`` and ``0.0`` differ
    and a NaN agrees only with the same NaN. The other classes allow each cell
    ``eps_class * (|want| + FLOOR)``; a cell whose two values are equal, or are
    both NaN, agrees whatever its allowance, which is what keeps an infinity
    both runs computed from reading as a difference of NaN.

    The two arrays have one shape and one dtype, as two runs of a kernel on
    copies of one argument do; anything else is a disagreement at every cell.
    """
    got = np.asarray(got)
    want = np.asarray(want)
    if got.shape != want.shape or got.dtype != want.dtype:
        return np.ones(want.shape, dtype=bool)
    if want.dtype.kind not in "fc":
        return np.asarray(got != want, dtype=bool)
    if exactness == "exact":
        width = want.dtype.itemsize
        bits = [
            np.ascontiguousarray(side).view(np.uint8).reshape(*side.shape, width)
            for side in (got, want)
        ]
        return np.any(bits[0] != bits[1], axis=-1)
    epsilon = TOLERANCE[exactness]
    with np.errstate(invalid="ignore", over="ignore"):
        same = got == want
        both_nan = np.isnan(got) & np.isnan(want)
        near = np.abs(got - want) <= epsilon * (np.abs(want) + TOLERANCE_FLOOR)
    return ~(same | both_nan | near)
