"""What loopy reads into isl: subscripts and guards on the loops.

loopy reads a subscript into isl for its bounds check and to simplify it in
code generation, and a guard that names only loop variables, sizes and
scalars for its bounds check. Its reader raised on a conversion, which kept
every subscript in 32 bits (#129) and refused a substitution that carried a
store's conversion into a subscript (#145, in ``test_fusion.py``), and it read
a constant by its integer part, and a real scalar as an integer, which made
its bounds check of a guarded access wrong (#137). :mod:`loopty.isl_reading`
makes it decline all three, and the lowering widens a subscript loopy does
not read as affine with no division as it widens any other integer
arithmetic.
"""

from __future__ import annotations

import islpy as isl
import numpy as np
import pymbolic.primitives as prim
import pytest
from lanky.prelude import Int, Real
from loopy.diagnostic import ExpressionToAffineConversionError, LoopyIndexError
from loopy.symbolic import TypeCast, guarded_aff_from_expr

from loopty import Arr, Fin, Schedule, kernel, when
from loopty.executor import LoopyExecutor, emit_code
from loopty.isl_reading import affine_form, read_as_affine


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
    # nothing caught, read 0.5 as 0, and read a float that is an integer as
    # the integer, though an expression with one is computed in floating
    # point, which rounds: np.float32(1.0) * i is 2**24 at i = 2**24 + 1.
    for declined in (
        TypeCast(np.dtype(np.int64), i),
        np.float64(0.5) * i,
        i * 0.5,
        np.float64(2.0) * i,
        i * 2.0,
        np.float32(1.0) * i,
        complex(1, 0) * i,
        np.float64(np.inf) + i,
    ):
        with pytest.raises(ExpressionToAffineConversionError):
            guarded_aff_from_expr(space, declined)
        assert not read_as_affine(declined)
    # An integer and a 64-bit integer literal are read as the integers they
    # are.
    for read, coefficient in (
        (2 * i, 2),
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
def near_half(x: Arr[Fin[m], Real], y: Arr[Fin[n], Real]):  # noqa: F821
    """A cell found by a floor division of a multiple of the loop variable."""
    for i in y.dom:
        y[i] = x[(i * 499_999) // 1_000_000 + 2200]


@kernel
def near_half_mod(x: Arr[Fin[m], Real], y: Arr[Fin[n], Real]):  # noqa: F821
    """A cell found by a remainder of a multiple of the loop variable."""
    for i in y.dom:
        y[i] = x[(i * 499_999) % 1_000_000 + 1_000_000]


def test_an_affine_subscript_with_a_division_is_computed_in_64_bits() -> None:
    # loopy reads these as affine, and wrote isl's form in its 32-bit index
    # type, x[2200 + (499999 * i) / 1000000]: 499999 * i leaves 32 bits at
    # i = 4295, and the division of the wrapped value named cell 53 there,
    # where numpy reads cell 4347; without the 2200 it named a negative cell.
    # isl keeps a multiple below half the divisor as it is, so this happens
    # for small arrays. A division in isl's form makes the subscript one
    # computed in 64 bits, as any other.
    rows = 4400
    agrees(
        near_half,
        lambda: {"x": np.arange(rows, dtype=np.float64), "y": np.zeros(rows)},
    )
    assert (
        "x[loopty_floor_div_int64((int64_t) (i) * 499999, (int64_t) (1000000))"
        " + 2200]" in emit_code(near_half)
    )
    agrees(
        near_half_mod,
        lambda: {"x": np.arange(2_000_000, dtype=np.float64), "y": np.zeros(rows)},
    )
    assert (
        "x[loopty_mod_int64((int64_t) (i) * 499999, (int64_t) (1000000))"
        " + 1000000]" in emit_code(near_half_mod)
    )
    i = prim.Variable("i")
    form = affine_form(prim.FloorDiv(i * 499_999, 1_000_000))
    assert form is not None and form.dim(isl.dim_type.div) == 1


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


def test_an_affine_subscript_with_no_division_is_left_as_it_was() -> None:
    # loopy simplifies a subscript it reads as affine to the expression isl
    # gives back, in its 32-bit index type, so a widening there would only
    # make it unreadable, and its bounds unchecked: i * 2 is cast nowhere,
    # and C computes its sums and products modulo 2**32 under -fwrapv, which
    # gives the cell an in-bounds subscript names. isl writes (i * 7919) % 7
    # with a division, 2 * i + -7 * ((2 * i) / 7), which is computed in 64
    # bits instead, as the subscripts above are.
    code = emit_code(doubled_index)
    assert "y[i] = x[2 * i];" in code
    agrees(doubled_index, lambda: {"x": np.arange(9.0), "y": np.zeros(9)})
    assert "y[i] = x[loopty_mod_int64((int64_t) (i) * 7919, (int64_t) (7))];" in (
        emit_code(scaled_mod_seven)
    )
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


@kernel
def rounded_away(x: Arr[Fin[n], Real], y: Arr[Fin[n], Real]):  # noqa: F821
    """A hundred cells on, where a double sum past 2**53 rounds the one away."""
    for i in y.dom:
        with when(i * 4503599627370496.0 + 1.0 <= i * 4503599627370496.0):
            y[i] = x[i + 100]


@kernel
def between_reals(a: Real, x: Arr[Fin[n], Real], y: Arr[Fin[n], Real]):  # noqa: F821
    """A hundred cells on, at the loop variable strictly between a - 1 and a."""
    for i in y.dom:
        with when((i < a) & (i > a - 1)):
            y[i] = x[i + 100]


@kernel
def below_an_integer(
    k: Int,
    x: Arr[Fin[m], Real],  # noqa: F821
    y: Arr[Fin[n], Real],  # noqa: F821
):
    """The cells of x below k, where k is no more than its size."""
    for i in y.dom:
        with when((i < k) & (k <= x.dom.size)):
            y[i] = x[i]


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


def test_a_guard_computed_in_floating_point_is_not_read() -> None:
    # A float that is an integer was read as the integer, and a Real scalar
    # as an integer parameter, though the guard is computed in double. Read
    # so, the first guard held nowhere, and the second for no integer a, so
    # loopy's bounds check passed x[i + 100] under each without looking, and
    # the compiled run read past the end of x where the native one is
    # refused: i * 2**52 + 1.0 rounds to i * 2**52 from i = 2, and a = 1.5
    # puts i = 1 strictly between. Neither is read now.
    for past, arguments in (
        (rounded_away, {}),
        (between_reals, {"a": 1.5}),
    ):
        with pytest.raises(IndexError):
            past(**inputs()(), **arguments)
        with pytest.raises(LoopyIndexError, match="could not establish"):
            LoopyExecutor().run(past, **inputs()(), **arguments)
    # So under a guard in double, one in bounds is refused too (#148): the
    # same guard in integers, doubled_index's, compiles.
    native = inputs(9)()
    twice_under(**native)
    assert list(native["y"][:6]) == [0.0, 2.0, 4.0, 6.0, 8.0, 0.0]
    with pytest.raises(LoopyIndexError, match="could not establish"):
        LoopyExecutor().run(twice_under, **inputs(9)())
    # A guard on an integer scalar is read as it was, and narrows the check.
    agrees(
        below_an_integer,
        lambda: {"k": 3, "x": np.arange(4.0), "y": np.zeros(6)},
    )


# }}}
