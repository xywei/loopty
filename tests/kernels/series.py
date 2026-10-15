"""Kernels that add a theorem's series term by term, and kernels that add another.

``halves`` is about the powers of ``x / 2``, and ``power_sum`` adds them as
``(x[i] * 0.5) ** k``, which the identity ``halved`` says is the same.
``quarter_sum`` adds the powers of ``x[i] * 0.25``, which no theorem here is
about, and ``tail_sum`` starts at the zeroth power, which ``from_one``
leaves out. tests/test_summands.py reads them.
"""

from __future__ import annotations

from lanky import theorem
from lanky.prelude import Fn, Nat, Real

from loopty import Arr, Fin, kernel, reduce_sum

#: The highest power each kernel adds.
ORDER = 3


@theorem
def halves(
    x: Real,
    c: Fn[Nat, Real],
    first: c(0) == 1,
    step: all(2 * c(n + 1) == x * c(n) for n in Nat),
) -> all(c(n) == (x / 2) ** n for n in Nat):
    """A table that starts at 1 and multiplies by ``x / 2`` holds its powers."""


@theorem
def halved(x: Real) -> x * 0.5 == x / 2:
    """Half of ``x``, by a product: what the kernel computes."""


@theorem
def from_one(
    x: Real,
    c: Fn[Nat, Real],
    first: c(1) == x,
    step: all(c(n + 1) == x * c(n) for n in Nat if n >= 1),
) -> all(c(n) == x**n for n in Nat if n >= 1):
    """From the first power on, a table that multiplies by ``x`` holds its powers."""


@kernel(uses=[halves, halved])
def power_sum(x: Arr[Fin[m], Real], y: Arr[Fin[m], Real]):
    """The powers of half of ``x[i]``, added through ``ORDER``."""
    for i in y.dom:
        y[i] = reduce_sum((x[i] * 0.5) ** k for k in Fin[ORDER + 1])


@kernel(uses=[halves, halved])
def quarter_sum(x: Arr[Fin[m], Real], y: Arr[Fin[m], Real]):
    """The powers of a quarter of ``x[i]``: no theorem here is about them."""
    for i in y.dom:
        y[i] = reduce_sum((x[i] * 0.25) ** k for k in Fin[ORDER + 1])


@kernel(uses=[from_one])
def tail_sum(x: Arr[Fin[m], Real], y: Arr[Fin[m], Real]):
    """The powers of ``x[i]`` from the zeroth, which ``from_one`` does not cover."""
    for i in y.dom:
        y[i] = reduce_sum(x[i] ** k for k in Fin[ORDER + 1])
