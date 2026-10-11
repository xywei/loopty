"""Constant tables: read-only arrays whose values are part of the kernel.

A table is declared next to the kernel as the numpy array it is (``W =
table([...])``), and the body reads it as it reads an array. Natively it is a
read-only numpy array; traced, each read is an access whose in-bounds fact
isl decides over the table's shape; lowered, it is a read-only loopy
temporary with the values as its initializer, which loopy writes once, with
every digit, outside the kernel's function. TSFC's element kernels hold their
quadrature weights and basis tables so (gap 18 of the demo kernels).
"""

from __future__ import annotations

import numpy as np
import pytest
from lanky.ledger import Status
from lanky.prelude import Int, Real

from loopty import Arr, Fin, Schedule, Table, kernel, program, reduce_sum
from loopty import table as make_table
from loopty.executor import LoopyExecutor, emit_code
from loopty.faithful import KIND
from loopty.oracle import IslOracle

pytest.importorskip("loopy")

#: The 3-point rule on the reference triangle, and the P1 basis at its points.
W = make_table([1 / 6, 1 / 6, 1 / 6])
PHI = make_table([[2 / 3, 1 / 6, 1 / 6], [1 / 6, 2 / 3, 1 / 6], [1 / 6, 1 / 6, 2 / 3]])
Z = make_table([1 + 2j, -0.5j, 3.0])
SINGLE = make_table([0.1, 0.2, 0.3], dtype=np.float32)
SHIFT = make_table([2, 0, 1])


# {{{ kernels


@kernel
def mass(u: Arr[Fin[nc], Fin[3], Real], y: Arr[Fin[nc], Fin[3], Real]):  # noqa: F821
    """The P1 mass matrix of each cell applied to its values, by quadrature."""
    for c in y.dom:
        for j in y.dom[c]:
            y[c, j] = reduce_sum(
                W[q] * PHI[q, j] * PHI[q, k] * u[c, k]
                for q in W.dom
                for k in PHI.dom[q]
            )


@kernel
def kinds(
    z: Arr[Fin[n], np.complex128],  # noqa: F821
    f: Arr[Fin[n], np.float32],  # noqa: F821
    k: Arr[Fin[n], Int],  # noqa: F821
):
    """A complex, a single precision and an integer table."""
    for i in z.dom:
        z[i] = reduce_sum(Z[q] for q in Z.dom) * z[i]
        f[i] = SINGLE[i % 3] * f[i]
        k[i] = SHIFT[i % 3] + k[i]


def weighted(q):
    """A helper of the body that reads a table."""
    return 2.0 * W[q]


@kernel
def through_a_helper(y: Arr[Fin[n], Real]):  # noqa: F821
    """A table read in a helper the body calls."""
    for i in y.dom:
        y[i] = reduce_sum(weighted(q) for q in W.dom) + y[i]


@kernel
def past_the_end(y: Arr[Fin[n], Real]):  # noqa: F821
    """A read one past the last weight."""
    for i in y.dom:
        y[i] = reduce_sum(W[q + 1] for q in W.dom)


@kernel
def scale(u: Arr[Fin[n], Real], v: Arr[Fin[n], Real]):  # noqa: F821
    """Each cell by a weight, round the table."""
    for i in v.dom:
        v[i] = W[i % 3] * u[i]


@program
def scaled_twice(u, w):
    """Two calls that read one table."""
    v = Arr.zeros_like(u)
    scale(u, v)
    scale(v, w)


@kernel
def writes_a_table(y: Arr[Fin[n], Real]):  # noqa: F821
    """A write into a table."""
    for i in y.dom:
        W[0] = 1.0
        y[i] = W[0]


def first_weight():
    """A helper that makes a table and holds it by the name of the kernel's array."""
    y = make_table([5.0, 6.0])
    return y[0]


@kernel
def clashing(y: Arr[Fin[n], Real]):  # noqa: F821
    """A table read by the name of a parameter."""
    for i in y.dom:
        y[i] = first_weight()


# }}}


def mass_inputs(cells: int = 4) -> dict:
    rng = np.random.default_rng(cells)
    return {"u": Arr.from_numpy(rng.random((cells, 3))), "y": Arr.zeros((cells, 3))}


def agrees(kern, schedule, inputs: dict) -> None:
    fact = LoopyExecutor().differential(kern, schedule, inputs)
    assert fact.status is Status.TESTED, fact.provenance
    faithful = [f for f in kern.facts() if f.kind == KIND]
    assert faithful and faithful[0].status is Status.TESTED, faithful


def test_a_table_is_a_read_only_copy_of_its_values() -> None:
    values = np.array([1.0, 2.0, 3.0])
    weights = make_table(values)
    values[0] = 7.0
    assert isinstance(weights, Table)
    assert weights[0] == 1.0 and isinstance(weights[0], np.float64)
    assert list(weights.dom) == [0, 1, 2] and list(PHI.dom[1]) == [0, 1, 2]
    assert weights.shape == (3,) and len(weights) == 3
    with pytest.raises(ValueError, match="read-only"):
        weights.numpy()[0] = 2.0
    with pytest.raises(TypeError, match="is a constant table"):
        weights[0] = 2.0
    with pytest.raises(IndexError, match="one value at a time, with 2 indices"):
        PHI[0]


@pytest.mark.parametrize(
    ("values", "error", "message"),
    [
        (2.0, ValueError, "at least one axis"),
        ([], ValueError, "at least one value"),
        (["a", "b"], TypeError, "holds numbers"),
    ],
)
def test_a_table_holds_numbers_along_an_axis(values, error: type, message: str) -> None:
    with pytest.raises(error, match=message):
        make_table(values)


def test_the_term_reads_each_table_by_the_name_the_body_gives_it() -> None:
    term = mass.term
    assert dict(term.tables) == {"W": W, "PHI": PHI}
    assert term.array_types["PHI"].axes == (3, 3)


def test_every_read_of_a_table_is_decided_in_bounds_over_its_shape() -> None:
    oracle = IslOracle()
    facts = [f for f in mass.facts() if f.kind == "in-bounds"]
    statements = {f.statement: oracle.establish(f).status for f in facts}
    assert statements["W[q] is in bounds for every instance of S0"] is Status.DECIDED
    assert statements["PHI[q, k] is in bounds for every instance of S0"] is (
        Status.DECIDED
    )
    (beyond,) = [
        f for f in past_the_end.facts() if f.kind == "in-bounds" and "W[" in f.statement
    ]
    refuted = oracle.establish(beyond)
    assert refuted.status is Status.REFUTED
    assert refuted.provenance["reason"].startswith(
        "cells W[q + 1] reaches are cells W has, except [a0=3]"
    )


def test_the_compiled_kernel_holds_the_values_with_every_digit() -> None:
    code = emit_code(Schedule(mass))
    assert (
        "static double W[3] = { 0.16666666666666666, 0.16666666666666666, "
        "0.16666666666666666 };" in code
    )
    assert "static double PHI[3 * 3] = { 0.6666666666666666," in code
    for cells in (1, 4, 7):
        agrees(mass, Schedule(mass).split("c", 2), mass_inputs(cells))


def test_tables_of_other_dtypes_are_read_in_their_own() -> None:
    code = emit_code(Schedule(kinds))
    assert "static double complex Z[3]" in code
    assert "static float SINGLE[3]" in code
    assert "static int64_t SHIFT[3] = { 2l, 0l, 1l };" in code
    inputs = {
        "z": Arr.from_numpy(np.arange(5) + 1j),
        "f": Arr.from_numpy(np.ones(5, np.float32)),
        "k": Arr.from_numpy(np.arange(5)),
    }
    native = {
        name: Arr.from_numpy(value.numpy().copy()) for name, value in inputs.items()
    }
    kinds(**native)
    out = LoopyExecutor().run(
        kinds, **{name: value.numpy().copy() for name, value in inputs.items()}
    )
    for name in ("z", "f", "k"):
        assert np.array_equal(out[name], native[name].numpy()), name
    agrees(kinds, Schedule(kinds), inputs)


def test_a_table_read_in_a_helper_is_the_kernels() -> None:
    assert dict(through_a_helper.term.tables) == {"W": W}
    agrees(through_a_helper, Schedule(through_a_helper), {"y": Arr.zeros(4)})


def test_a_program_holds_one_table_for_every_call_that_reads_it() -> None:
    term = scaled_twice.term
    assert dict(term.tables) == {"W": W}
    fact = LoopyExecutor().differential(
        scaled_twice,
        Schedule(scaled_twice),
        {"u": Arr.from_numpy(np.arange(5.0)), "w": Arr.zeros(5)},
    )
    assert fact.status is Status.TESTED, fact.provenance


def test_a_table_is_written_by_nobody() -> None:
    with pytest.raises(TypeError, match="is a constant table"):
        writes_a_table(Arr.zeros(2))
    with pytest.raises(TypeError, match="is a constant table"):
        writes_a_table.trace()


def test_a_table_named_like_a_parameter_is_named_apart() -> None:
    # The helper's frame names its table y, which is the kernel's array: the
    # table is y_table in the term and the generated code.
    tables = dict(clashing.term.tables)
    assert list(tables) == ["y_table"]
    assert list(tables["y_table"].values) == [5.0, 6.0]
    assert "static double y_table[2] = { 5.0, 6.0 };" in emit_code(Schedule(clashing))
    agrees(clashing, Schedule(clashing), {"y": Arr.zeros(3)})


def test_opencl_code_holds_a_table_in_constant_memory(plain_opencl) -> None:
    from loopty.isl_reading import declining

    lp = plain_opencl
    with lp.CacheMode(False), declining():
        code = lp.generate_code_v2(Schedule(mass, target="opencl").kernel).device_code()
    assert "__constant double W[3] = { 0.16666666666666666," in code
