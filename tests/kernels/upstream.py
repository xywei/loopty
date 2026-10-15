"""A kernel, and functions said to compute what the code it mirrors computes.

``double`` doubles a vector. ``numpy_double`` is numpy's own doubling, which
agrees with it; the others disagree at one cell, cannot run, or return what
the kernel does not write, one way each. ``rounded`` and ``off_by_a_billionth``
state the allowance of their claim, ``within=1e-12``: the first is off by
round-off, and the second by more, inside the class's allowance and beyond
the claim's. tests/test_upstream.py reads them.
"""

from __future__ import annotations

import numpy as np
from lanky.prelude import Fin, Real

from loopty import Arr, kernel, upstream


@kernel
def double(x: Arr[Fin[n], Real], y: Arr[Fin[n], Real]):
    """``y = 2 x``."""
    for i in y.dom:
        y[i] = 2.0 * x[i]


@kernel
def double_and_square(
    x: Arr[Fin[n], Real], y: Arr[Fin[n], Real], z: Arr[Fin[n], Real]
):
    """``y = 2 x`` and ``z = x * x``: two outputs."""
    for i in y.dom:
        y[i] = 2.0 * x[i]
        z[i] = x[i] * x[i]


def example_inputs() -> dict:
    """Five numbers, and vectors to double and square them into."""
    x = np.linspace(0.0, 1.0, 5)
    return {
        "double": {"x": Arr.from_numpy(x), "y": Arr.zeros(5)},
        "double_and_square": {
            "x": Arr.from_numpy(x.copy()),
            "y": Arr.zeros(5),
            "z": Arr.zeros(5),
        },
    }


@upstream(double)
def numpy_double(x, y):
    """numpy's doubling, which the kernel mirrors."""
    return {"y": 2.0 * x.numpy()}


@upstream(double)
def overwrites_its_input(x, y):
    """numpy's doubling, then the input zeroed: the kernel's run reads its own copy."""
    out = 2.0 * x.numpy()
    x.numpy()[...] = 0.0
    return {"y": out}


@upstream(double)
def off_at_the_end(x, y):
    """Doubling, one more at the last cell."""
    out = 2.0 * x.numpy()
    out[-1] += 1.0
    return {"y": out}


@upstream(double)
def not_installed(x, y):
    """Code from a package that is not installed here."""
    import a_package_nobody_installs  # noqa: F401

    return {"y": 2.0 * x.numpy()}


@upstream(double)
def not_recorded(x, y):
    """A recording that is not here."""
    return {"y": np.load("no/such/recording.npy")}


@upstream(double)
def about_its_input(x, y):
    """What the kernel only reads."""
    return {"x": x.numpy()}


@upstream(double_and_square)
def doubles_only(x, y, z):
    """Code that computes the doubles and not the squares."""
    return {"y": 2.0 * x.numpy()}



@upstream(double, within=1e-12)
def rounded(x, y):
    """Doubling, off at the last cell by a few units of the last place."""
    out = 2.0 * x.numpy()
    out[-1] += 4e-16
    return {"y": out}


@upstream(double, within=1e-12)
def off_by_a_billionth(x, y):
    """Doubling, off at the last cell by a billionth: inside approx, past the claim."""
    out = 2.0 * x.numpy()
    out[-1] += 1e-9
    return {"y": out}
