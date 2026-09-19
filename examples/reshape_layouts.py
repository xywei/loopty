"""Reshape is the identity on the type; the layout is a map beside it.

``Fin`` is a semiring homomorphism: ``Fin[n * m]`` and ``Fin[n] x Fin[m]`` are
the same index type, definitionally. That is the whole content of a reshape.
Nothing is moved, nothing is copied, and no nonlinear equation between sizes is
ever stated, which matters because a product of two unknown sizes is outside the
fragment isl decides. Where the data actually lives is a separate thing: a
*layout*, an affine map from the index type to a flat address. Two layouts over
the same index type are two views of one buffer.

Run this file three ways.

``python examples/reshape_layouts.py``
    Prints the normalization, the two layout maps, a small ledger of what isl
    decided about them, the two views of one flat vector, and a transpose run
    both natively and on the C target.

``lanky check examples/reshape_layouts.py``
    Prints the kernels' own obligations. The interesting ones are the in-bounds
    facts for ``flat[COLS * i + j]``: an access written through a linearization
    is affine as long as the stride is a literal, and isl decides it from the
    loop bounds alone.

``loopty run examples/reshape_layouts.py``
    Compiles the split-and-interchanged transpose and compares it with the
    native run.

Where the affine fragment ends
------------------------------

The row-major view needs only the *inner* size to be a literal: with ``COLS``
known, ``COLS * i + j`` is affine in ``i`` and ``j`` however many rows there are,
so ``rows_of`` keeps a symbolic ``n``. The column-major view multiplies the
column index by the *number of rows*, so a symbolic ``n`` would put ``n * j``,
a product of two unknowns, inside the index; ``cols_of`` therefore fixes both
sizes. This is not a limitation of the checker so much as a signpost: the way to
write the general case is to index the pair and leave the address to the layout,
which is what ``transpose`` does and why its obligations are decided for any
``n`` and ``m``.
"""

from __future__ import annotations

import numpy as np
from lanky.prelude import Real

from loopty import Arr, Fin, Schedule, kernel

#: A three by four matrix. Small enough to print in full, and the two layouts
#: of it are different enough to read off the numbers which one produced what.
ROWS, COLS = 3, 4

# {{{ the kernels


@kernel
def rows_of(flat: Arr[Fin[n * COLS], Real], mat: Arr[Fin[n], Fin[COLS], Real]):
    """The row-major view: ``mat[i, j]`` is the cell ``COLS * i + j``.

    The number of rows stays symbolic. The index is affine because the stride
    is a literal, so isl decides ``0 <= COLS * i + j < COLS * n`` from the loop
    bounds, with no size equation and no assumption about the storage.
    """
    for i in mat.dom:
        for j in mat.dom[i]:
            mat[i, j] = flat[COLS * i + j]


@kernel
def cols_of(flat: Arr[Fin[ROWS * COLS], Real], mat: Arr[Fin[ROWS], Fin[COLS], Real]):
    """The column-major view of the same buffer: the cell ``i + ROWS * j``.

    Both sizes are literals here, because the column stride is the number of
    rows: leaving that symbolic would multiply two unknowns inside an index.
    """
    for i in mat.dom:
        for j in mat.dom[i]:
            mat[i, j] = flat[i + ROWS * j]


@kernel
def transpose(a: Arr[Fin[n], Fin[m], Real], b: Arr[Fin[m], Fin[n], Real]):
    """The general case, written on the index pair: no address arithmetic.

    Every obligation is affine for any ``n`` and ``m``, and the loop nest is
    free to be reordered, which is what the schedule below asks for.
    """
    for i in a.dom:
        for j in a.dom[i]:
            b[j, i] = a[i, j]


# }}}


# {{{ the layouts, as isl objects


def layout_maps() -> tuple:
    """The two dense layouts of a ``ROWS`` by ``COLS`` matrix, as isl maps."""
    from loopty.idx import Layout

    shape = (Fin[ROWS], Fin[COLS])
    return Layout(shape, "C").to_map(), Layout(shape, "F").to_map()


def delinearization() -> object:
    """The isl map taking a flat address back to an index pair, row-major.

    It is built from :func:`loopty.idx.delinearize`'s own terms, floor division
    and remainder, so what isl is asked about is what the library computes.
    """
    import islpy as isl
    from pymbolic.primitives import Variable

    from loopty.idx import delinearize, isl_expr

    row, column = delinearize(Variable("a"), (Fin[ROWS], Fin[COLS]), "C")
    return isl.Map(
        f"{{ [a] -> [i0, i1] : i0 = {isl_expr(row)} and i1 = {isl_expr(column)} "
        f"and 0 <= a < {ROWS * COLS} }}"
    )


def layout_ledger() -> object:
    """What isl decides about the layouts, as a lanky ledger.

    Four questions, all of them in the Presburger fragment and all of them
    about types rather than about a run: each layout addresses every cell
    exactly once, delinearizing any address lands inside the matrix, and
    delinearization undoes the row-major layout.
    """
    import islpy as isl
    from lanky.ledger import Fact, Ledger, Status

    from loopty.idx import Layout
    from loopty.oracle import Bijective, IslOracle, Subset

    c_order, f_order = layout_maps()
    delin = delinearization()
    domain = Layout((Fin[ROWS], Fin[COLS]), "C").domain()
    identity = isl.Map("{ [i0, i1] -> [i0, i1] }")

    questions = [
        (
            "bijective",
            "the row-major layout addresses each cell of the matrix exactly once",
            Bijective(c_order),
        ),
        (
            "bijective",
            "the column-major layout addresses each cell exactly once",
            Bijective(f_order),
        ),
        (
            "subset",
            "delinearizing a flat address lands inside the matrix",
            Subset(delin.range(), domain),
        ),
        (
            "subset",
            "delinearization undoes the row-major layout",
            Subset(c_order.apply_range(delin), identity),
        ),
    ]

    oracle = IslOracle()
    ledger = Ledger()
    for index, (kind, statement, question) in enumerate(questions):
        fact = Fact(
            id=f"layout:{kind}:{index}",
            kind=kind,
            statement=statement,
            term=question,
            status=Status.ASSUMED,
            where=f"reshape_layouts.py:{ROWS}x{COLS}",
            owner="layouts",
        )
        ledger.add(oracle.establish(fact) or fact)
    return ledger


# }}}


# {{{ example data and schedules


def flat_vector() -> Arr:
    """``0, 1, ..., ROWS * COLS - 1`` in one flat buffer."""
    return Arr.from_numpy(np.arange(float(ROWS * COLS)))


def matrix() -> Arr:
    """A ``ROWS`` by ``COLS`` matrix of distinct numbers."""
    return Arr.from_numpy(np.arange(float(ROWS * COLS)).reshape(ROWS, COLS))


#: The transpose, split and interchanged: the cast is a permutation of the
#: instances, and the checker accepts it because the transpose carries no
#: dependence between instances at all. The same two steps over the stencil
#: would be rejected; the difference is in the footprints, not in the syntax.
blocked = (
    Schedule(transpose, target="c", sizes={"n": ROWS, "m": COLS})
    .split("i", 2, inner="i_in", outer="i_out")
    .interchange("i_out", "j", "i_in")
    .example(a=matrix(), b=Arr.zeros((COLS, ROWS)))
)


def example_inputs() -> dict:
    """The inputs ``loopty run`` gives each kernel in this file."""
    return {
        "rows_of": {"flat": flat_vector(), "mat": Arr.zeros((ROWS, COLS))},
        "cols_of": {"flat": flat_vector(), "mat": Arr.zeros((ROWS, COLS))},
        "transpose": {"a": matrix(), "b": Arr.zeros((COLS, ROWS))},
    }


# }}}


def main() -> int:
    """Print the normalization, the layouts, the views, and the transpose."""
    from pymbolic.primitives import Variable

    from loopty.executor import LoopyExecutor
    from loopty.idx import normalize

    product = Fin[Variable("n") * Variable("m")]
    print(f"normalize(Fin[n * m]) = {normalize((product,))}")
    c_order, f_order = layout_maps()
    print(f"row-major layout    {c_order}")
    print(f"column-major layout {f_order}")
    print()
    print(layout_ledger().render(width=64))

    flat = flat_vector()
    rows, cols = Arr.zeros((ROWS, COLS)), Arr.zeros((ROWS, COLS))
    rows_of(flat, rows)
    cols_of(flat, cols)
    print()
    print("one flat vector:", flat.numpy())
    print("read row-major:")
    print(rows.numpy())
    print("read column-major:")
    print(cols.numpy())

    print()
    print(f"schedule: {blocked!r}")
    print(f"  loop nest: {' '.join(blocked.order)}")
    for fact in blocked.facts():
        print(f"  {fact.status.value:8} {fact.decided_by or '-':4} {fact.statement}")

    a, b = matrix(), Arr.zeros((COLS, ROWS))
    transpose(a, b)
    fact = LoopyExecutor().differential(
        transpose, blocked, {"a": matrix(), "b": Arr.zeros((COLS, ROWS))}
    )
    print()
    print("transposed natively:")
    print(b.numpy())
    for name, detail in fact.provenance["outputs"].items():
        print(
            f"  {name}: difference {detail['difference']:.3g} within "
            f"{detail['tolerance']:.3g} ({detail['exactness']}) -> "
            f"{fact.status.value}"
        )
    return 0 if fact.status.value == "tested" else 1


if __name__ == "__main__":
    raise SystemExit(main())
