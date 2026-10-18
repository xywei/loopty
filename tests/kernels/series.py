"""Kernels that add a theorem's series term by term, and kernels that add another.

``halves`` is about the powers of ``x / 2``, and ``power_sum`` adds them as
``(x[i] * 0.5) ** k``, which the identity ``halved`` says is the same.
``quarter_sum`` adds the powers of ``x[i] * 0.25``, which no theorem here is
about, and ``tail_sum`` starts at the zeroth power, which ``from_one``
leaves out. ``inverse_square`` divides by a power ``halves`` is about,
which it does not add, and ``complex_sum`` adds the powers of half of a
complex number, which a theorem about reals says nothing of.
``premised_sum`` adds ``power_sum``'s powers through ``halved_from_zero``,
whose premise its row names. tests/test_summands.py reads them.
"""

from __future__ import annotations

import numpy as np
from lanky import claim
from lanky.prelude import Fn, Nat, Real

from loopty import Arr, Fin, kernel, reduce_sum

#: The highest power each kernel adds.
ORDER = 3


@claim
def halves(
    x: Real,
    c: Fn[Nat, Real],
    first: c(0) == 1,
    step: all(2 * c(n + 1) == x * c(n) for n in Nat),
) -> all(c(n) == (x / 2) ** n for n in Nat):
    """A table that starts at 1 and multiplies by ``x / 2`` holds its powers."""


@claim
def halved(x: Real) -> x * 0.5 == x / 2:
    """Half of ``x``, by a product: what the kernel computes."""


@claim
def halved_from_zero(x: Real, nonnegative: x >= 0) -> x * 0.5 == x / 2:
    """``halved``, for a nonnegative ``x`` only: a premise ``halves`` does not have."""


@claim
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


@kernel(uses=[halves, halved])
def inverse_square(x: Arr[Fin[m], Real], y: Arr[Fin[m], Real]):
    """One over the square of half of ``x[i]``: a power it divides by, not adds."""
    for i in y.dom:
        y[i] = 1 / ((x[i] * 0.5) ** 2)


@kernel(uses=[halves, halved])
def complex_sum(x: Arr[Fin[m], np.complex128], y: Arr[Fin[m], np.complex128]):
    """The powers of half of a complex ``x[i]``, which ``halves`` is not about."""
    for i in y.dom:
        y[i] = reduce_sum((x[i] * 0.5) ** k for k in Fin[ORDER + 1])


@kernel(uses=[halves, halved_from_zero])
def premised_sum(x: Arr[Fin[m], Real], y: Arr[Fin[m], Real]):
    """``power_sum``'s powers, through an identity with a premise of its own."""
    for i in y.dom:
        y[i] = reduce_sum((x[i] * 0.5) ** k for k in Fin[ORDER + 1])
