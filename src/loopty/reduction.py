"""Reduction frontends for Loopty kernels.

Loopty owns the computational reduction API. Lanky supplies the generator/binder
capture machinery used by the frontend, but the tracer converts the captured
node into :class:`loopty.term.Reduction`, where the reduced ISL domain and
exactness contract live.
"""

from __future__ import annotations

from collections.abc import Iterable
from typing import Any

from lanky import sum as _lanky_sum

__all__ = ["reduce_sum"]


def reduce_sum(gen: Iterable[Any]) -> Any:
    """Sum a generator over a concrete or symbolic Loopty domain.

    On concrete domains this behaves like Python's :func:`sum`. During
    tracing, Lanky captures the generator binders and Loopty immediately lowers
    that temporary node to :class:`loopty.term.Reduction` with its polyhedral
    iteration domain.
    """
    return _lanky_sum(gen)
