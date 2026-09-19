"""A dense kernel: ``y := a*x + y``.

The smallest complete example. Outputs are parameters, so ``y`` is an argument
and the kernel returns nothing; sizes come from the data, so the loop runs over
``y.dom`` and the name ``n`` appears only in the annotations, where lanky's
scope invents it. Every obligation here is affine, so isl decides all of them.
"""

from __future__ import annotations

import numpy as np
from lanky.prelude import Real

from loopty import Arr, Fin, kernel


@kernel
def axpy(a: Real, x: Arr[Fin[n], Real], y: Arr[Fin[n], Real]):
    """Scale ``x`` by ``a`` and add it into ``y``, elementwise."""
    for i in y.dom:
        y[i] = a * x[i] + y[i]


def main() -> None:
    """Run the kernel on tiny arrays and print the result."""
    x = Arr.from_numpy(np.arange(4, dtype=np.float64))
    y = Arr.zeros(4)
    axpy(2.0, x, y)
    print(y.numpy())


if __name__ == "__main__":
    main()
