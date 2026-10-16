"""Where each cell is stored: sumpy's storage index and grudge's dim_tags, checked.

A layout is the map from an array's cells to the addresses of its buffer.
loopy's shaped arrays (loopy #1053) take a layout's injectivity on trust;
here a layout is a Python function, and that it stores every cell inside its
buffer, no two cells at one address, and a cell at every address, are facts
isl decides and Lean proves. Two layouts from the codes the demos come from:

*sumpy's storage index.* The compressed Taylor expansion of a 2-D Laplace
potential keeps the derivatives ``D(a, b)`` with ``a`` in ``{0, 1}`` only, as
the others follow from the PDE: ``get_storage_index`` puts ``(0, 0)`` at 0,
and then the two derivatives of each total order ``a + b``, the one with
``a = 0`` first, at ``2 (a + b) - 1 + a``. No one formula gives both, so the
layout has two pieces, each a function over its own cells.

*grudge's output layout.* grudge #418 tags the output of its tensor-product
passes with loopy's ``dim_tags`` ``N3,N0,N1,N2`` on ``(e, a, b, c)``, and its
DOFArray is ``(ne, N**3)``, row-major, each element reshaped by modepy with
``order="F"``, ``a + N b + N**2 c``. That they are one layout is a fact too;
``N3,N2,N1,N0``, the reshape in C order, is refuted with a cell.

Run this file three ways.

``python examples/storage_layouts.py``
    Prints where the storage index puts each stored derivative, reads a box of
    coefficients through it, checks the DOFArray against numpy's reshape, and
    asks for the wrong nesting, which isl refutes.

``lanky check examples/storage_layouts.py``
    The layouts' facts, and the two kernels'. Each layout stores every cell
    inside its buffer, no two at one address, and a cell at every address; and
    grudge's dim_tags store every cell where the DOFArray does.

``loopty run examples/storage_layouts.py``
    The two kernels that read through the layouts, compiled and compared.
"""

from __future__ import annotations

import numpy as np
from lanky.prelude import Real

from loopty import Arr, Fin, Where, kernel, layout

#: The order of the expansion: P = 4 keeps 2P + 1 = 9 derivatives.
P = 4
Q = P + 1

#: Nodes per axis of a hexahedron (order 2), as in grudge's passes.
N = 3


# {{{ sumpy's storage index


def centre(a: Fin[1], b: Fin[1]) -> Fin[2 * P + 1]:
    """The value at the centre, stored first."""
    return 0


def derivatives(
    cell: Where[a: Fin[2], b: Fin[Q], (a + b >= 1) & (a + b < Q)],
) -> Fin[2 * P + 1]:
    """Two derivatives of each order a + b, the one with a = 0 first."""
    a, b = cell
    return 2 * (a + b) - 1 + a


#: sumpy's get_storage_index for the 2-D Laplace local expansion of order P.
storage_index = layout(centre, derivatives)


@kernel
def stored_table(
    coeffs: Arr[Fin[nb], Fin[2 * P + 1], Real],
    R: Arr[Fin[nb], Fin[2], Fin[Q], Real],
):
    """Each box's stored derivatives as a table R[i, a, b], read through the index."""
    for i, a, b in R.dom:
        if a + b == 0:
            R[i, a, b] = coeffs[i, centre(a, b)]
        elif a + b < Q:
            R[i, a, b] = coeffs[i, derivatives((a, b))]


# }}}


# {{{ grudge's output layout


@layout
def dof_array(e: Fin[ne], a: Fin[N], b: Fin[N], c: Fin[N]) -> Fin[ne * N**3]:
    """The DOFArray: row-major (ne, N**3), each element reshaped in F order."""
    return N**3 * e + (a + N * b + N**2 * c)


#: grudge #418 writes the same layout as loopy's dim_tags.
as_dim_tags = dof_array.same_as("N3,N0,N1,N2")


@kernel
def unflatten(
    flat: Arr[Fin[ne * N**3], Real],
    u: Arr[Fin[ne], Fin[N], Fin[N], Fin[N], Real],
):
    """Each element's nodes as u[e, a, b, c], read through the DOFArray's layout."""
    for e, a, b, c in u.dom:
        u[e, a, b, c] = flat[dof_array(e, a, b, c)]


# }}}


def coefficients(boxes: int = 3) -> np.ndarray:
    """Distinct numbers in sumpy's storage order, one row per box."""
    return np.arange(boxes * (2 * P + 1), dtype=float).reshape(boxes, 2 * P + 1)


def example_inputs() -> dict:
    """The inputs ``loopty run`` gives each kernel."""
    elements = 2
    return {
        "stored_table": {
            "coeffs": Arr.from_numpy(coefficients()),
            "R": Arr.zeros((3, 2, Q)),
        },
        "unflatten": {
            "flat": Arr.from_numpy(np.arange(elements * N**3, dtype=float)),
            "u": Arr.zeros((elements, N, N, N)),
        },
    }


def main() -> int:
    """The storage index's addresses, both kernels natively, and a wrong nesting."""
    print("sumpy's storage index, (a, b) -> address:")
    cells = [(0, 0)] + [
        (a, s - a) for s in range(1, Q) for a in (0, 1) if s - a >= 0
    ]
    print(
        "  "
        + ", ".join(
            f"({a}, {b}) -> {centre(a, b) if a + b == 0 else derivatives((a, b))}"
            for a, b in cells
        )
    )

    inputs = example_inputs()
    stored_table(**inputs["stored_table"])
    table = inputs["stored_table"]["R"].numpy()
    print(f"  box 0 read through it: R[0, 1, 3] = {table[0, 1, 3]:g} (address 8)")

    unflatten(**inputs["unflatten"])
    flat = inputs["unflatten"]["flat"].numpy()
    reshaped = flat.reshape(2, N**3).reshape(2, N, N, N, order="F")
    agrees = bool(np.array_equal(inputs["unflatten"]["u"].numpy(), reshaped))
    print(f"grudge's DOFArray through dof_array is modepy's reshape: {agrees}")

    from loopty import IslOracle

    (wrong,) = dof_array.same_as("N3,N2,N1,N0").facts()
    refuted = IslOracle().establish(wrong)
    print(f"{refuted.status.value}: {wrong.statement}")
    print(f"  {refuted.provenance['reason']}")
    return 0 if agrees and refuted.status.value == "refuted" else 1


if __name__ == "__main__":
    raise SystemExit(main())
