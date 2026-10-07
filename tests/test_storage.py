"""What a cell is stored as, in both runs: literals, arguments and truth values.

The compiled run stores every element sort in the dtype the lowering gives it
(:func:`loopty.lower.numpy_dtype`) and the native run in the dtype numpy gives
it, and the two have to hold the same values or the contract has to refuse the
call. Three ways they did not: a real literal lowered in the integer type of
the cell it is stored into (#73), an argument whose dtype does not hold its
sort (#77), and a value that is not a truth value stored into an array of
``Bool`` (#78).
"""

from __future__ import annotations

import warnings

import numpy as np
import pytest
from lanky.prelude import Bool, Int, Nat, Real

from loopty import Arr, Fin, Schedule, TraceError, Where, kernel, reduce_sum, when
from loopty.executor import LoopyExecutor, emit_code
from loopty.interpret import interpret


def agrees(kern, make) -> None:
    """The scheduled run of ``kern`` agrees with its native run on ``make()``."""
    fact = LoopyExecutor().differential(kern, Schedule(kern), make())
    assert fact.status.value == "tested", fact.provenance


# {{{ a real literal in an integer cell (#73)


@kernel
def halve_nat(u: Arr[Fin[n], Real], c: Arr[Fin[n], Nat]):  # noqa: F821
    """Half of every entry, as a natural number: the fraction goes at the store."""
    for i in u.dom:
        c[i] = u[i] * 0.5


@kernel
def less_half(u: Arr[Fin[n], Real], k: Arr[Fin[n], Int]):  # noqa: F821
    """Every entry less a half, as an integer, which truncates towards zero."""
    for i in u.dom:
        k[i] = u[i] - 0.5


@kernel
def halve_int(k: Arr[Fin[n], Int], h: Arr[Fin[n], Int]):  # noqa: F821
    """Half of every integer, truncated: ``0.5`` beside an integer is a double."""
    for i in k.dom:
        h[i] = k[i] * 0.5


@kernel
def above_half(k: Arr[Fin[n], Int], b: Arr[Fin[n], Bool]):  # noqa: F821
    """Whether half of an integer is above ``2 ** 24``, compared in double."""
    for i in k.dom:
        b[i] = k[i] * 0.5 > 16777216.0


@kernel
def turn(u: Arr[Fin[n], Real], f: Arr[Fin[n], np.complex128]):  # noqa: F821
    """A turn by a complex number single precision does not hold."""
    for i in u.dom:
        f[i] = (0.1 + 0.2j) * u[i]


def test_a_real_literal_stored_into_naturals_is_not_lowered_as_its_integer_part():
    # loopy wrote the bare 0.5 in the type of the integer assignee, as 0.
    def make() -> dict:
        return {
            "u": np.array([1.0, 2.0, 3.0, 4.0]),
            "c": np.zeros(4, dtype=np.int64),
        }

    native = make()
    halve_nat(**native)
    assert list(native["c"]) == [0, 1, 1, 2]
    out = LoopyExecutor().run(halve_nat, **make())
    assert list(out["c"]) == [0, 1, 1, 2]
    assert "u[i] * 0.5" in emit_code(halve_nat)
    agrees(halve_nat, make)


def test_a_real_literal_stored_into_integers_truncates_as_the_native_store_does():
    def make() -> dict:
        return {
            "u": np.array([0.0, 1.0, -1.0, 2.75]),
            "k": np.zeros(4, dtype=np.int64),
        }

    out = LoopyExecutor().run(less_half, **make())
    assert list(out["k"]) == [0, 0, -1, 2]
    agrees(less_half, make)


def test_a_real_literal_beside_an_integer_is_double_precision():
    # loopy took 0.5 for a float32, which an integer joined into a float32, so
    # half of 2 ** 25 + 1 lost its half: 0 into an integer, and not above 2 **
    # 24 into a Bool. numpy computes both in double.
    def integers() -> dict:
        return {
            "k": np.array([33554433, 3], dtype=np.int64),
            "h": np.zeros(2, dtype=np.int64),
        }

    def truths() -> dict:
        return {
            "k": np.array([33554433, 3], dtype=np.int64),
            "b": np.zeros(2, dtype=bool),
        }

    assert list(LoopyExecutor().run(halve_int, **integers())["h"]) == [16777216, 1]
    assert list(LoopyExecutor().run(above_half, **truths())["b"]) == [1, 0]
    agrees(halve_int, integers)
    agrees(above_half, truths)


def test_a_complex_literal_is_double_precision():
    # loopy refuses to guess the precision of a complex constant that single
    # precision does not hold, so this did not lower at all.
    def make() -> dict:
        return {
            "u": np.array([1.0, -2.0]),
            "f": np.zeros(2, dtype=np.complex128),
        }

    out = LoopyExecutor().run(turn, **make())
    assert np.array_equal(out["f"], (0.1 + 0.2j) * np.array([1.0, -2.0]))
    agrees(turn, make)


# }}}


# {{{ an argument that does not hold its sort (#77)


@kernel
def halve_twice(x: Arr[Fin[n], Real]):  # noqa: F821
    """Halve every entry, then double it: the identity, on reals."""
    for i in x.dom:
        x[i] = x[i] / 2
    for i in x.dom:
        x[i] = x[i] * 2


@kernel
def double(x: Arr[Fin[n], Real], y: Arr[Fin[n], Real]):  # noqa: F821
    """Twice every entry."""
    for i in x.dom:
        y[i] = 2 * x[i]


@kernel
def square(x: Arr[Fin[n], Real], y: Arr[Fin[n], Real]):  # noqa: F821
    """The square of every entry."""
    for i in x.dom:
        y[i] = x[i] * x[i]


@kernel
def self_sum(x: Arr[Fin[n], Real], y: Arr[Fin[n], Real]):  # noqa: F821
    """Every entry added to itself."""
    for i in x.dom:
        y[i] = x[i] + x[i]


@kernel
def truncate(u: Arr[Fin[n], Real], c: Arr[Fin[n], Nat]):  # noqa: F821
    """Every entry as a natural number, which drops its fraction."""
    for i in u.dom:
        c[i] = u[i]


@kernel
def mark(u: Arr[Fin[n], Real], b: Arr[Fin[n], Bool]):  # noqa: F821
    """Which entries are above one."""
    for i in u.dom:
        b[i] = u[i] > 1.0


@kernel
def keep_unmarked(
    b: Arr[Fin[n], Bool],  # noqa: F821
    u: Arr[Fin[n], Real],  # noqa: F821
    y: Arr[Fin[n], Real],  # noqa: F821
):
    """Copy the entries ``b`` does not mark."""
    for i in u.dom:
        with when(~b[i]):
            y[i] = u[i]


def test_an_integer_array_written_as_reals_is_refused():
    # Natively every write was truncated, and compiled none was: [3, 5] came
    # back [2, 4] from the native run and [3., 5.] from the compiled one.
    def make() -> dict:
        return {"x": np.array([3, 5])}

    fix = "Pass x as float64"
    with pytest.raises(ValueError, match="x is stored as int64") as refused:
        halve_twice(**make())
    assert fix in str(refused.value)
    with pytest.raises(ValueError, match=fix):
        LoopyExecutor().run(halve_twice, **make())
    with pytest.raises(ValueError, match=fix):
        LoopyExecutor().differential(halve_twice, Schedule(halve_twice), make())
    agrees(halve_twice, lambda: {"x": np.array([3.0, 5.0])})


def test_a_complex_array_of_reals_is_refused_unless_it_is_real():
    # The cast into float64 dropped the imaginary part, which the native run
    # kept: y was [2+2j, 4-6j] natively and [2, 4] compiled.
    def make() -> dict:
        return {
            "x": np.array([1 + 1j, 2 - 3j]),
            "y": np.zeros(2, dtype=complex),
        }

    with pytest.raises(ValueError, match=r"x\[0\] is \(1\+1j\).*imaginary part"):
        double(**make())
    with pytest.raises(ValueError, match="imaginary part"):
        LoopyExecutor().run(double, **make())

    # A complex x with no imaginary part is a real one, read as one by both
    # runs, and the cast says nothing; a complex y is still refused, since it
    # is written.
    def real_parts() -> dict:
        return {"x": np.array([1 + 0j, -2 + 0j]), "y": np.zeros(2)}

    with warnings.catch_warnings():
        warnings.simplefilter("error", np.exceptions.ComplexWarning)
        native = real_parts()
        double(**native)
        assert list(native["y"]) == [2.0, -4.0]
        assert native["x"].dtype == np.complex128
        agrees(double, real_parts)
    with pytest.raises(ValueError, match="y is stored as complex128"):
        LoopyExecutor().run(double, x=np.ones(2), y=np.zeros(2, dtype=complex))


def test_an_integer_array_read_as_reals_is_read_as_reals_by_both_runs():
    # An integer x that is only read used to be computed in natively: x * x
    # overflowed int64 at 2 ** 32, where the compiled run squares a double.
    # The interpreter reads it the way the native run does.
    def make() -> dict:
        return {"x": np.array([2**32, 3], dtype=np.int64), "y": np.zeros(2)}

    native = make()
    with warnings.catch_warnings():
        warnings.simplefilter("error", RuntimeWarning)
        square(**native)
    assert list(native["y"]) == [2.0**64, 9.0]
    assert native["x"].dtype == np.int64 and list(native["x"]) == [2**32, 3]
    interpreted = make()
    interpret(square.term, interpreted)
    assert list(interpreted["y"]) == [2.0**64, 9.0]
    agrees(square, make)

    # A bool x is read as reals too: True + True was True natively.
    def truths() -> dict:
        return {"x": np.array([True, False]), "y": np.zeros(2)}

    native = truths()
    self_sum(**native)
    assert list(native["y"]) == [2.0, 0.0]
    agrees(self_sum, truths)


def test_an_array_of_naturals_written_as_floats_is_refused():
    # A float c kept the fraction natively that the compiled integer drops.
    def make() -> dict:
        return {"u": np.array([1.5, 2.0]), "c": np.zeros(2)}

    with pytest.raises(ValueError, match="c is stored as float64") as refused:
        truncate(**make())
    message = str(refused.value)
    assert "a signed integer of 32 bits or more" in message
    assert "Pass c as int64" in message
    with pytest.raises(ValueError, match="Pass c as int64"):
        LoopyExecutor().run(truncate, **make())
    for dtype in (np.int64, np.int32):
        agrees(
            truncate, lambda d=dtype: {"u": np.array([1.5, 2.0]), "c": np.zeros(2, d)}
        )


def test_an_array_of_truth_values_written_as_bytes_is_refused():
    with pytest.raises(ValueError, match="b is stored as int8") as refused:
        mark(u=np.array([0.5, 2.0]), b=np.zeros(2, dtype=np.int8))
    assert "Pass b as bool" in str(refused.value)


def test_an_array_of_reals_written_as_float32_is_refused():
    # Natively each write is rounded to single precision, and compiled none
    # is, so 1e8 + 1 written and 1e8 taken away again leaves 0 natively and 1
    # compiled.
    with pytest.raises(ValueError, match="x is stored as float32"):
        LoopyExecutor().run(halve_twice, x=np.ones(2, dtype=np.float32))


def test_an_array_of_truth_values_read_as_numbers_holds_zero_and_one():
    # Compiled, b is bytes, and a byte converted from 0.5 is 0 where a bool is
    # True. An int8 b of zeros and ones is read natively as the bools it
    # stands for, so ~ is logical on it there too; it used to be bitwise,
    # which when refuses.
    def make(b) -> dict:
        return {
            "b": b,
            "u": np.array([1.0, 2.0, 3.0]),
            "y": np.zeros(3),
        }

    with pytest.raises(ValueError, match=r"b\[0\] is 0.5.*a truth value"):
        keep_unmarked(**make(np.array([0.5, 0.0, 1.0])))
    with pytest.raises(ValueError, match=r"b\[1\] is 2"):
        LoopyExecutor().run(keep_unmarked, **make(np.array([0, 2, 1], dtype=np.int8)))

    native = make(np.array([0, 1, 0], dtype=np.int8))
    keep_unmarked(**native)
    assert list(native["y"]) == [1.0, 0.0, 3.0]
    agrees(keep_unmarked, lambda: make(np.array([0, 1, 0], dtype=np.int8)))
    agrees(keep_unmarked, lambda: make(np.array([0.0, 1.0, 0.0])))


@kernel
def plus_one(c: Arr[Fin[n], Nat], d: Arr[Fin[n], Nat]):  # noqa: F821
    """Every entry plus one."""
    for i in c.dom:
        d[i] = c[i] + 1


@kernel
def row_squares(
    cnt: Arr[Fin[n], Nat],  # noqa: F821
    val: Arr[Fin[n], Fin[cnt], Real],  # noqa: F821
    y: Arr[Fin[n], Real],  # noqa: F821
):
    """The sum of the squares of every row of a ragged array."""
    for r in val.dom:
        y[r] = reduce_sum(val[r, j] * val[r, j] for j in val.dom[r])


@kernel
def halve_rows(cnt: Arr[Fin[n], Nat], val: Arr[Fin[n], Fin[cnt], Real]):  # noqa: F821
    """Halve every entry of a ragged array."""
    for r in val.dom:
        for j in val.dom[r]:
            val[r, j] = val[r, j] / 2


@kernel
def lower_squares(
    L: Arr[Where[i: Fin[n], j: Fin[n], j < i], Real],  # noqa: F821
    y: Arr[Fin[n], Real],  # noqa: F821
):
    """The sum of the squares of every row of a strictly lower triangle."""
    for i in L.dom:
        y[i] = reduce_sum(L[i, j] * L[i, j] for j in L.dom[i])


def test_an_array_in_another_layout_is_read_and_refused_as_a_dense_one():
    # The copy keeps the layout: a ragged array its offsets, an array over a
    # domain its domain. A uint8 array of naturals is read as int64 too, where
    # natively 255 + 1 wrapped round to 0 and compiled it was 256.
    from lanky.terms import Var

    def narrow() -> dict:
        return {"c": np.array([255, 1], np.uint8), "d": np.zeros(2, np.int64)}

    native = narrow()
    plus_one(**native)
    assert list(native["d"]) == [256, 2]
    agrees(plus_one, narrow)

    cnt = np.array([2, 1], np.int64)

    def ragged(dtype) -> dict:
        values = np.array([2**32, 3, 5], dtype)
        return {"cnt": cnt, "val": Arr.ragged(cnt, values=values), "y": np.zeros(2)}

    native = ragged(np.int64)
    row_squares(**native)
    assert list(native["y"]) == [2.0**64, 25.0]
    agrees(row_squares, lambda: ragged(np.int64))
    with pytest.raises(ValueError, match="val is stored as int64.*Pass val as float64"):
        halve_rows(cnt=cnt, val=Arr.ragged(cnt, values=np.array([3, 5, 7])))

    i, j, size = Var("i"), Var("j"), Var("n")
    triangle = Where[i : Fin[size], j : Fin[size], j < i]

    def lower() -> dict:
        values = np.array([2**32, 3, 5], np.int64)
        return {"L": Arr.from_cells(triangle, values, n=3), "y": np.zeros(3)}

    native = lower()
    lower_squares(**native)
    assert list(native["y"]) == [0.0, 2.0**64, 34.0]
    agrees(lower_squares, lower)


# }}}


# {{{ a scalar that does not hold its sort (#77)


@kernel
def square_of(a: Real, y: Arr[Fin[n], Real]):  # noqa: F821
    """The square of a scalar, at every cell."""
    for i in y.dom:
        y[i] = a * a


@kernel
def twice(a: Nat, y: Arr[Fin[n], Nat]):  # noqa: F821
    """Twice a natural number, at every cell."""
    for i in y.dom:
        y[i] = a + a


@kernel
def scaled(a: Real, y: Arr[Fin[n], Real]):  # noqa: F821
    """Twice a scalar, at every cell."""
    for i in y.dom:
        y[i] = a * 2.0


@kernel
def negated(flag: Bool, b: Arr[Fin[n], Bool]):  # noqa: F821
    """The negation of a truth value, at every cell."""
    for i in b.dom:
        b[i] = ~flag


@kernel
def unless(flag: Bool, y: Arr[Fin[n], Real]):  # noqa: F821
    """Ones where a truth value does not hold."""
    for i in y.dom:
        with when(~flag):
            y[i] = 1.0


def test_a_scalar_is_computed_with_in_the_dtype_of_its_sort():
    # A scalar is passed by value, so the native run converts it as the
    # compiled run does: np.int64(2 ** 32) squared overflowed natively, and
    # np.int8(100) doubled wrapped round to -56.
    def wide() -> dict:
        return {"a": np.int64(2**32), "y": np.zeros(2)}

    native = wide()
    with warnings.catch_warnings():
        warnings.simplefilter("error", RuntimeWarning)
        square_of(**native)
    assert list(native["y"]) == [2.0**64, 2.0**64]
    interpreted = wide()
    interpret(square_of.term, interpreted)
    assert list(interpreted["y"]) == [2.0**64, 2.0**64]
    agrees(square_of, wide)

    def narrow() -> dict:
        return {"a": np.int8(100), "y": np.zeros(2, np.int64)}

    native = narrow()
    twice(**native)
    assert list(native["y"]) == [200, 200]
    agrees(twice, narrow)


def test_a_truth_value_scalar_is_a_numpy_bool_natively():
    # ~True is -2 on a Python bool, which a bool array stored as True and when
    # refused; compiled it is !flag. A numpy bool is negated logically, and
    # compiled it is passed as the byte Bool is lowered as.
    for flag in (True, False, np.True_, 1, 0.0):
        native = {"flag": flag, "b": np.zeros(2, dtype=bool)}
        negated(**native)
        assert list(native["b"]) == [not flag] * 2
        agrees(negated, lambda f=flag: {"flag": f, "b": np.zeros(2, dtype=bool)})
        # A guard that names no loop variable is kept compiled (#90).
        native = {"flag": flag, "y": np.zeros(2)}
        unless(**native)
        assert list(native["y"]) == [0.0 if flag else 1.0] * 2
        agrees(unless, lambda f=flag: {"flag": f, "y": np.zeros(2)})
    # The samples of the faithfulness fact draw Python bools.
    assert negated.facts()[-1].status.value == "tested"
    assert unless.facts()[-1].status.value == "tested"


def test_a_scalar_its_sort_does_not_hold_is_refused():
    # 2 for a Bool was stored as the byte 2 compiled and as True natively.
    for flag in (2, 0.5, np.int64(-1)):
        args = {"flag": flag, "b": np.zeros(2, dtype=bool)}
        with pytest.raises(ValueError, match="flag has to be a truth value"):
            negated(**args)
        with pytest.raises(ValueError, match="flag has to be a truth value"):
            LoopyExecutor().run(negated, **args)

    with pytest.raises(ValueError, match=r"a is \(1\+1j\).*imaginary part"):
        scaled(a=1 + 1j, y=np.zeros(2))
    with pytest.raises(ValueError, match="imaginary part"):
        LoopyExecutor().run(scaled, a=1 + 1j, y=np.zeros(2))
    # A complex scalar with no imaginary part is a real one in both runs, and
    # neither warns.
    with warnings.catch_warnings():
        warnings.simplefilter("error", np.exceptions.ComplexWarning)
        native = {"a": 1.5 + 0j, "y": np.zeros(2)}
        scaled(**native)
        assert list(native["y"]) == [3.0, 3.0]
        agrees(scaled, lambda: {"a": np.complex128(1.5), "y": np.zeros(2)})


# }}}


# {{{ a value that is not a truth value, stored into Bool (#78)


@kernel
def mark_raw(u: Arr[Fin[n], Real], b: Arr[Fin[n], Bool]):  # noqa: F821
    """A real stored into a Bool array: a truth value natively, a byte compiled."""
    for i in u.dom:
        b[i] = u[i]


@kernel
def low_bit(k: Arr[Fin[n], Int], b: Arr[Fin[n], Bool]):  # noqa: F821
    """``&`` of an integer, which is bitwise natively and ``and`` in the trace."""
    for i in k.dom:
        b[i] = k[i] & 1


@kernel
def set_all(u: Arr[Fin[n], Real], b: Arr[Fin[n], Bool]):  # noqa: F821
    """An integer constant stored into a Bool array."""
    for i in u.dom:
        b[i] = 1


@kernel
def not_first(b: Arr[Fin[n], Bool]):  # noqa: F821
    """``~`` of a comparison of a loop variable, which natively is ``-2`` or ``-1``."""
    for i in b.dom:
        b[i] = ~(i > 0)


@kernel
def in_band(
    u: Arr[Fin[n], Real],  # noqa: F821
    c: Arr[Fin[n], Bool],  # noqa: F821
    b: Arr[Fin[n], Bool],  # noqa: F821
):
    """Truth values every way they are written: comparisons, connectives, reads."""
    for i in u.dom:
        b[i] = ((u[i] > 0.5) & (u[i] < 2.5)) | ~c[i]


def test_a_real_stored_into_a_bool_array_is_refused_by_the_trace():
    # Natively 0.5 was stored as True, and compiled the byte was 0; 2.0 was a
    # byte of 2.
    with pytest.raises(TraceError, match=r"b\[i\] = u\[i\] != 0") as refused:
        mark_raw.trace()
    assert "0.5 becomes 0 and 2.0 becomes 2" in str(refused.value)
    (fact,) = mark_raw.facts()
    assert fact.kind == "trace" and fact.status.value == "refuted"
    assert "b[i] = u[i] != 0" in fact.provenance["reason"]

    with pytest.raises(TraceError, match="Store True, the truth value numpy"):
        set_all.trace()
    # The trace records k[i] & 1 as "and", and natively it is bitwise: an
    # integer operand makes a connective no truth value.
    with pytest.raises(TraceError, match=r"its operand k\[i\] is not a truth value"):
        low_bit.trace()


def test_truth_values_stored_into_a_bool_array_agree():
    def make() -> dict:
        return {
            "u": np.array([0.0, 1.0, 2.0, 3.0]),
            "c": np.array([True, True, False, True]),
            "b": np.zeros(4, dtype=bool),
        }

    native = make()
    in_band(**native)
    assert list(native["b"]) == [False, True, True, False]
    agrees(in_band, make)


@pytest.mark.filterwarnings("ignore:Bitwise inversion:DeprecationWarning")
def test_an_integer_stored_into_a_bool_array_natively_is_refused():
    # ~(i > 0) traces as "not", and natively is ~ of a Python bool, -2 or -1,
    # which a bool array stored as True at every point.
    def make() -> dict:
        return {"b": np.zeros(3, dtype=bool)}

    not_first.trace()
    with pytest.raises(TraceError, match="is the integer -1, not a truth value"):
        not_first(**make())
    fact = LoopyExecutor().differential(not_first, Schedule(not_first), make())
    assert fact.status.value == "refuted"
    assert "'i <= 0' for '~(i > 0)'" in fact.provenance["reason"]
    faithful = not_first.facts()[-1]
    assert faithful.kind == "trace-faithful"
    assert faithful.status.value == "refuted"


# }}}
