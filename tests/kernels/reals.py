"""Kernels whose postconditions about reals are compared at their class.

``thirds`` takes the mean of three values with a table of 15-digit thirds,
which is the mean to round-off only: its postcondition holds to the approx
tolerance and not bit for bit. ``thirds_exact`` claims the same of arrays
whose class is ``exact``, which no tolerance loosens, and ``off_mean`` a mean
it misses by a thousandth. ``copied`` claims what nothing in it rounds.
tests/test_reals.py reads them.
"""

from __future__ import annotations

from lanky.prelude import Real

from loopty import Arr, Fin, kernel, reduce_sum, table

#: A third, to the 15 digits a generated kernel writes.
W = table([0.333333333333333, 0.333333333333333, 0.333333333333333])


@kernel
def thirds(x: Arr[Fin[3], Real], y: Arr[Fin[1], Real]) -> y[0] == (
    x[0] + x[1] + x[2]
) / 3:
    """The mean of ``x``, by the table of thirds."""
    y[0] = reduce_sum(W[q] * x[q] for q in W.dom)


@kernel
def thirds_exact(
    x: Arr[Fin[3], Real.exact], y: Arr[Fin[1], Real.exact]
) -> y[0] == (x[0] + x[1] + x[2]) / 3:
    """The same, of arrays whose class allows no rounding at all."""
    y[0] = W[0] * x[0] + W[1] * x[1] + W[2] * x[2]


@kernel
def off_mean(x: Arr[Fin[3], Real], y: Arr[Fin[1], Real]) -> y[0] == (
    x[0] + x[1] + x[2]
) / 3 + 0.001:
    """A mean it misses by a thousandth, far beyond the approx tolerance."""
    y[0] = reduce_sum(W[q] * x[q] for q in W.dom)


@kernel
def copied(x: Arr[Fin[n], Real], y: Arr[Fin[n], Real]) -> all(
    y[i] == -x[i] for i in Fin[n]
):
    """Each value negated, which rounds nothing: compared bit for bit."""
    for i in y.dom:
        y[i] = -x[i]
