"""A kernel that is wrong, so that the ledger can say so.

``u[i + 1]`` over the whole of ``u.dom`` reads one cell past the end. Nothing
here is unusual: the indices are affine, the loop bound comes from the data, and
the body runs happily under plain ``python`` on a numpy array that happens to be
long enough. The typing rule states the obligation anyway, and isl refutes it
with the cell the access reaches and the array does not, which is the whole
point of recording obligations rather than trusting them.

``lanky check`` exits non-zero on this file.
"""

from __future__ import annotations

from lanky.prelude import Real

from loopty import Arr, Fin, kernel


@kernel
def shift(u: Arr[Fin[n], Real], v: Arr[Fin[n], Real]):
    """Shift ``u`` down by one, reading past the end at the last point."""
    for i in u.dom:
        v[i] = u[i + 1]
