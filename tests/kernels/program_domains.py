"""Programs whose arrays are over a polyhedral domain (#74).

``fill`` writes the lower triangle and ``rowsum`` reads it, each calling its
size its own way; the programs pass them an argument over the triangle, an
array made like it, and one of a declared shape whose size is read off the
program's vector, ``Arr.zeros(TRIANGLE, n=x.dom.size)``.
"""

from __future__ import annotations

from lanky.prelude import Fin, Real
from lanky.terms import Var

from loopty import Arr, Where, kernel, program, reduce_sum

i, j, n = Var("i"), Var("j"), Var("n")

#: The lower triangle, declared where a program makes an array over it.
TRIANGLE = Where[i : Fin[n], j : Fin[n], j < i]


@kernel
def fill(x: Arr[Fin[n], Real], f: Arr[Where[i : Fin[n], j : Fin[n], j < i], Real]):
    """``f[i, j] = x[i] x[j]`` on the lower triangle."""
    for i in f.dom:
        for j in f.dom[i]:
            f[i, j] = x[i] * x[j]


@kernel
def rowsum(f: Arr[Where[i : Fin[m], j : Fin[m], j < i], Real], e: Arr[Fin[m], Real]):
    """Each row of the triangle summed: the kernel calls its size ``m``."""
    for p in e.dom:
        e[p] = reduce_sum(f[p, q] for q in f.dom[p])


@kernel
def diagonal(f: Arr[Where[i : Fin[n], j : Fin[n], j <= i], Real], e: Arr[Fin[n], Real]):
    """A triangle with its diagonal: another set of points."""
    for p in e.dom:
        e[p] = f[p, p]


@program
def through(x, L, e):
    """The triangle an argument."""
    fill(x, L)
    rowsum(L, e)


@program
def like(x, L, e):
    """An array made like the argument over the triangle."""
    f = Arr.zeros_like(L)
    fill(x, f)
    rowsum(f, e)
    rowsum(L, e)


@program
def declared(x, e):
    """An array over the triangle, its size read off ``x``."""
    f = Arr.zeros(TRIANGLE, n=x.dom.size)
    fill(x, f)
    rowsum(f, e)
