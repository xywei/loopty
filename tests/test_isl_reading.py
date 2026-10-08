"""What loopy reads into isl: subscripts and guards on the loops.

loopy reads a subscript into isl for its bounds check and to simplify it in
code generation, and a guard that names only loop variables, sizes and
scalars for its bounds check. Its reader raised on a conversion, which kept
every subscript in 32 bits (#129) and refused a substitution that carried a
store's conversion into a subscript (#145, in ``test_fusion.py``), and it read
a constant by its integer part, which made its bounds check of a guarded
access wrong (#137). :mod:`loopty.isl_reading` makes it decline both, and
the lowering widens a subscript loopy does not read as affine as it widens
any other integer arithmetic.
"""

from __future__ import annotations

import islpy as isl
import numpy as np
import pymbolic.primitives as prim
import pytest
from lanky.prelude import Real
from loopy.diagnostic import ExpressionToAffineConversionError, LoopyIndexError
from loopy.symbolic import TypeCast, guarded_aff_from_expr

from loopty import Arr, Fin, Schedule, kernel, when
from loopty.executor import LoopyExecutor, emit_code
from loopty.isl_reading import read_as_affine


def agrees(kern, make) -> None:
    """The compiled run agrees with the native one, bit for bit and by its fact."""
    native = make()
    kern(**native)
    compiled = make()
    LoopyExecutor().run(kern, **compiled)
    for name, value in native.items():
        if isinstance(value, np.ndarray):
            assert np.array_equal(value, compiled[name]), name
    fact = LoopyExecutor().differential(kern, Schedule(kern), make())
    assert fact.status.value == "tested", fact.provenance


# {{{ the reader


def test_the_reader_declines_a_conversion_and_a_constant_not_an_integer() -> None:
    i = prim.Variable("i")
    space = isl.Space.create_from_names(isl.DEFAULT_CONTEXT, set=["i"])
    # loopy's reader raised UnsupportedExpressionError on a cast, which
    # nothing caught, and read 0.5 as 0.
    for declined in (
        TypeCast(np.dtype(np.int64), i),
        np.float64(0.5) * i,
        i * 0.5,
        complex(1, 0) * i,
        np.float64(np.inf) + i,
    ):
        with pytest.raises(ExpressionToAffineConversionError):
            guarded_aff_from_expr(space, declined)
        assert not read_as_affine(declined)
    # An integer, a float that is one, and a 64-bit integer literal are read
    # as the integers they are.
    for read, coefficient in (
        (2 * i, 2),
        (np.float64(2.0) * i, 2),
        (np.int64(100_000) * i, 100_000),
    ):
        aff = guarded_aff_from_expr(space, read)
        assert aff.get_coefficient_val(isl.dim_type.in_, 0).to_python() == coefficient
        assert read_as_affine(read)
    assert not read_as_affine(i * i)
    assert not read_as_affine(prim.Subscript(prim.Variable("col"), (i,)))


# }}}


# {{{ a subscript computed in 64 bits (#129)


@kernel
def index_square(x: Arr[Fin[n], Real], y: Arr[Fin[n], Real]):  # noqa: F821
    """A cell found by the square of the loop variable."""
    for i in x.dom:
        y[i] = x[(i * i) % x.dom.size]


@kernel
def index_scaled(x: Arr[Fin[n], Real], y: Arr[Fin[n], Real]):  # noqa: F821
    """A cell found by a multiple of the loop variable."""
    for i in x.dom:
        y[i] = x[(i * 7919) % x.dom.size]


@kernel
def col_scaled(
    col: Arr[Fin[n], Fin[m]],  # noqa: F821
    x: Arr[Fin[m], Real],  # noqa: F821
    y: Arr[Fin[n], Real],  # noqa: F821
):
    """A cell found by a multiple of an index array's entry."""
    for i in y.dom:
        y[i] = x[(col[i] * 7919) % x.dom.size]


def test_a_subscript_is_computed_in_64_bits() -> None:
    # i * i leaves 32 bits at i = 46341, and i * 7919 and col[i] * 7919 at
    # 271_183. Natively they are Python's and numpy's 64-bit integers; the
    # compiled run computed them in 32 bits, read wrong cells, and could die
    # of a read out of bounds. Each subscript is widened as any other integer
    # arithmetic is, and loopy, whose isl reader declines the cast, writes it
    # as it is written.
    rows = 46_342
    agrees(
        index_square,
        lambda: {"x": np.arange(rows, dtype=np.float64), "y": np.zeros(rows)},
    )
    assert "y[i] = x[loopty_mod_int64((int64_t) (i) * i, (int64_t) (n))]" in (
        emit_code(index_square)
    )
    rows = 271_184
    agrees(
        index_scaled,
        lambda: {"x": np.arange(rows, dtype=np.float64), "y": np.zeros(rows)},
    )
    assert "x[loopty_mod_int64((int64_t) (i) * 7919, (int64_t) (n))]" in (
        emit_code(index_scaled)
    )
    m = 300_000

    def entries() -> dict:
        return {
            "col": np.array([271_183, 3, 299_999]),
            "x": np.arange(m, dtype=np.float64),
            "y": np.zeros(3),
        }

    native = entries()
    col_scaled(**native)
    assert native["y"][0] == 271_183 * 7919 % m
    agrees(col_scaled, entries)
    assert "x[loopty_mod_int64((int64_t) (col[i]) * 7919, (int64_t) (m))]" in (
        emit_code(col_scaled)
    )


@kernel
def doubled_index(x: Arr[Fin[n], Real], y: Arr[Fin[n], Real]):  # noqa: F821
    """Every other cell, while there is one."""
    for i in y.dom:
        with when(2 * i < x.dom.size):
            y[i] = x[i * 2]


@kernel
def scaled_mod_seven(x: Arr[Fin[7], Real], y: Arr[Fin[n], Real]):  # noqa: F821
    """A cell of seven found by a multiple of the loop variable."""
    for i in y.dom:
        y[i] = x[(i * 7919) % 7]


def test_a_subscript_loopy_reads_as_affine_is_left_as_it_was() -> None:
    # loopy simplifies a subscript it reads as affine to the expression isl
    # gives back, in its 32-bit index type, so a widening there would only
    # make it unreadable, and its bounds unchecked: i * 2 is cast nowhere.
    # isl writes (i * 7919) % 7 with the multiple reduced, which C computes
    # in 32 bits past i = 271_183, where i * 7919 leaves them.
    code = emit_code(doubled_index)
    assert "y[i] = x[2 * i];" in code
    agrees(doubled_index, lambda: {"x": np.arange(9.0), "y": np.zeros(9)})
    assert "y[i] = x[2 * i + -7 * ((2 * i) / 7)];" in emit_code(scaled_mod_seven)
    rows = 271_190
    agrees(scaled_mod_seven, lambda: {"x": np.arange(7.0), "y": np.zeros(rows)})


# }}}


# {{{ a non-integer literal in a guard on the loops (#137)


@kernel
def half_guard(x: Arr[Fin[n], Real], y: Arr[Fin[n], Real]):  # noqa: F821
    """The cells four on, for the first four."""
    for i in y.dom:
        with when(i * 0.5 < 2):
            y[i] = x[i + 4]


@kernel
def half_guard_past(x: Arr[Fin[n], Real], y: Arr[Fin[n], Real]):  # noqa: F821
    """The cells four on, from the third on: past the end."""
    for i in y.dom:
        with when(i * 0.5 >= 1):
            y[i] = x[i + 4]


@kernel
def below_one_and_a_half(x: Arr[Fin[n], Real], y: Arr[Fin[n], Real]):  # noqa: F821
    """The last cell and the one past it."""
    for i in y.dom:
        with when(i < 1.5):
            y[i] = x[i + y.dom.size - 1]


@kernel
def twice_under(x: Arr[Fin[n], Real], y: Arr[Fin[n], Real]):  # noqa: F821
    """Every other cell, with the bound written in double."""
    for i in y.dom:
        with when(i * 2.0 < x.dom.size):
            y[i] = x[2 * i]


def inputs(size: int = 8):
    return lambda: {"x": np.arange(size, dtype=np.float64), "y": np.zeros(size)}


def test_a_guard_with_a_non_integer_literal_is_not_read_by_its_integer_part() -> None:
    # loopy read 0.5 as 0. when(i * 0.5 >= 1) read as false everywhere, so
    # its bounds check passed x[i + 4] without looking, and the compiled run
    # read past the end of x where the native one is refused. when(i < 1.5)
    # read as i < 1, which let x[i + n - 1] through at i = 1. Neither guard
    # is read into isl now, so loopy checks the access at every point of the
    # loop, and refuses it.
    for past in (half_guard_past, below_one_and_a_half):
        with pytest.raises(IndexError):
            past(**inputs()())
        with pytest.raises(LoopyIndexError, match="could not establish"):
            LoopyExecutor().run(past, **inputs()())
    # when(i * 0.5 < 2) read as true everywhere, which loopy refused x[i + 4]
    # under; it is refused for every i it checks now, as an access under any
    # guard loopy cannot read is (a product of loop variables, an array's
    # entry), though the native run is in bounds.
    native = inputs()()
    half_guard(**native)
    assert list(native["y"][:5]) == [4.0, 5.0, 6.0, 7.0, 0.0]
    with pytest.raises(LoopyIndexError, match=r"4 <= i0 <= 3 \+ n"):
        LoopyExecutor().run(half_guard, **inputs()())
    # A float that is an integer is read as the integer: loopy checks x[2 * i]
    # under 2 * i < n, and the kernel compiles and agrees.
    assert "if (i * 2.0 < n)" in emit_code(twice_under)
    agrees(twice_under, inputs(9))


# }}}
