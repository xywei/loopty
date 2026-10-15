"""Reduction frontends for Loopty kernels.

Loopty owns the computational reduction API. Lanky supplies the generator/binder
capture machinery used by the frontend, but the tracer converts the captured
node into :class:`loopty.term.Reduction`, where the reduced ISL domain and
exactness contract live.
"""

from __future__ import annotations

from collections.abc import Iterable
from typing import Any

import pymbolic.primitives as prim
from lanky import sum as _lanky_sum

__all__ = ["reduce_sum"]


def reduce_sum(gen: Iterable[Any]) -> Any:
    """Sum a generator over a concrete or symbolic Loopty domain.

    On concrete domains this behaves like Python's :func:`sum`. During
    tracing, Lanky captures the generator binders and Loopty immediately lowers
    that temporary node to :class:`loopty.term.Reduction` with its polyhedral
    iteration domain.

    A sum over a ``Fin`` of a number, ``reduce_sum(... for d in Fin[3])``, is
    traced as its terms added up, since lanky walks the ``Fin`` as a
    ``range``. The tracer is told the binder and the count
    (:meth:`loopty.trace.Tracer.unrolled_sum`), so that the ledger states
    the terms' accesses once, over ``d``, as the source writes them.
    """
    unrolled = _over_a_number(gen)
    value = _lanky_sum(gen)
    if unrolled is not None and isinstance(value, prim.ExpressionNode):
        from loopty.trace import current_tracer

        tracer = current_tracer()
        if tracer is not None:
            tracer.unrolled_sum(value, *unrolled)
    return value


def _over_a_number(gen: Any) -> tuple[str, int] | None:
    """``(binder, count)`` of a generator expression over a range, before it runs.

    ``(d, 3)`` for ``(... for d in Fin[3])``, whose iterable lanky made
    ``range(3)``. ``None`` for anything else: a generator with more than one
    ``for`` or a target that is not a name (whose terms are not one per
    point of the range), one already started, or one over a symbolic
    domain, which lanky binds a variable over.
    """
    code = getattr(gen, "gi_code", None)
    frame = getattr(gen, "gi_frame", None)
    if code is None or frame is None or getattr(gen, "gi_running", False):
        return None
    if code.co_varnames[:1] != (".0",):
        return None
    # The target is a cell variable where a sum inside the body reads it.
    names = [name for name in (*code.co_varnames, *code.co_cellvars) if name != ".0"]
    if len(names) != 1:
        return None
    try:
        iterator = frame.f_locals.get(".0")
    except Exception:  # noqa: BLE001 - a frame that will not show its locals
        return None
    if type(iterator).__name__ not in ("range_iterator", "longrange_iterator"):
        return None
    return names[0], iterator.__length_hint__()
