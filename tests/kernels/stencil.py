"""A stencil kernel: one-dimensional Jacobi, in time.

Two things are on show. Sizes come from the data even inside a guard: a domain
knows its own extent (``row.size``), which is an ``int`` on real data and a term
while tracing, so the interior condition is written once and means the same
thing in both modes.

And the guard is a ``when`` block, not an ``if``. A Python ``if`` on a value the
kernel computes cannot be traced, because tracing would have to choose a branch;
``when`` records the condition instead, narrows the statement's domain by it
when it is affine, and under plain ``python`` masks the writes of the block. The
narrowed domain is what makes ``u[t + 1, i]`` provably in bounds: it is only
ever written where ``t + 1 < nt``.

The dependences of this kernel are the classic pair, ``(1, 1)`` and ``(1, -1)``,
which is why tiling it in ``t`` and ``i`` is illegal until the space axis is
skewed by the time axis.
"""

from __future__ import annotations

from lanky.prelude import Real

from loopty import Arr, Fin, kernel, when


@kernel
def jacobi(u: Arr[Fin[nt], Fin[nx], Real]):
    """Average each interior point's neighbours into the next time level."""
    steps = u.dom
    for t in steps:
        row = u.dom[t]
        for i in row:
            with when((t + 1 < steps.size) & (i > 0) & (i + 1 < row.size)):
                u[t + 1, i] = (u[t, i - 1] + u[t, i + 1]) / 2


def main() -> None:
    """Run three time levels over five points and print the result."""
    u = Arr.zeros((3, 5))
    u.numpy()[0] = [0.0, 1.0, 2.0, 3.0, 4.0]
    jacobi(u)
    print(u.numpy())


if __name__ == "__main__":
    main()
