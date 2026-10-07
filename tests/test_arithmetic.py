"""Arithmetic in both runs: what C computes and what numpy computes.

The native run computes with numpy's arithmetic and the compiled run with C's,
and the two type an operation differently: numpy by NEP 50, where a Python
number takes the dtype of what stands beside it, C by its usual arithmetic
conversions (:mod:`loopty.promotion`). Five ways the runs disagreed: an
integer quotient (#82), a connective of integers (#83), a power (#84), a
literal beside a ``float32`` (#91), and an integral value outside 32 bits
(#92). Each kernel here either agrees with its native run bit for bit, or is
refused, by the trace or the contract, with the fix named.
"""

from __future__ import annotations

import numpy as np
import pytest
from lanky.prelude import Int, Nat, Real

from loopty import Arr, Fin, Schedule, TraceError, kernel, reduce_sum, when
from loopty.contract import INTEGRAL_RANGE
from loopty.executor import LoopyExecutor, emit_code
from loopty.lower import numpy_dtype


def agrees(kern, make) -> None:
    """The compiled run of ``kern`` agrees bit for bit with its native run."""
    native = make()
    kern(**native)
    compiled = make()
    LoopyExecutor().run(kern, **compiled)
    for name, value in native.items():
        if isinstance(value, np.ndarray):
            assert value.dtype == compiled[name].dtype
            assert np.array_equal(value, compiled[name]), (name, value, compiled[name])
    fact = LoopyExecutor().differential(kern, Schedule(kern), make())
    assert fact.status.value == "tested", fact.provenance


# {{{ an integer quotient (#82)


@kernel
def int_quot(k: Arr[Fin[n], Int], h: Arr[Fin[n], Int]):  # noqa: F821
    """Half of an integer, doubled: ``3 / 2 * 2`` is ``3.0`` natively."""
    for i in k.dom:
        h[i] = k[i] / 2 * 2


@kernel
def quotients(
    k: Arr[Fin[n], Int],  # noqa: F821
    m: Nat,
    h: Arr[Fin[n], Int],  # noqa: F821
    g: Arr[Fin[n], Int],  # noqa: F821
):
    """A quotient of loop variables, and one by a scalar, before the store."""
    for i in k.dom:
        h[i] = i / 2 * 2
        g[i] = k[i] / m * m


def test_an_integer_quotient_is_true_division_compiled_too():
    # C divided two integers as integers, so 3 / 2 * 2 was 2; numpy divides
    # them as doubles, and the store truncated 3.0 to 3.
    def make() -> dict:
        return {"k": np.array([3, -3, 4, 7]), "h": np.zeros(4, np.int64)}

    native = make()
    int_quot(**native)
    assert list(native["h"]) == [3, -3, 4, 7]
    assert "(double) (k[i]) / 2" in emit_code(int_quot)
    agrees(int_quot, make)

    def scaled() -> dict:
        return {
            "k": np.array([3, -3, 4, 7]),
            "m": 2,
            "h": np.zeros(4, np.int64),
            "g": np.zeros(4, np.int64),
        }

    native = scaled()
    quotients(**native)
    assert list(native["h"]) == [0, 1, 2, 3]
    assert list(native["g"]) == [3, -3, 4, 7]
    agrees(quotients, scaled)
    # A numpy scalar divides as a Python int does: the cast does not depend
    # on how the caller passed it.
    agrees(quotients, lambda: {**scaled(), "m": np.int64(2)})


# }}}


# {{{ a connective of integers (#83)


@kernel
def masked_scale(
    k: Arr[Fin[n], Int],  # noqa: F821
    x: Arr[Fin[n], Real],  # noqa: F821
    y: Arr[Fin[n], Real],  # noqa: F821
):
    """The low bit of an integer as a factor: bitwise natively, ``and`` traced."""
    for i in k.dom:
        y[i] = (k[i] & 1) * x[i]


@kernel
def odd_compared(k: Arr[Fin[n], Int], y: Arr[Fin[n], Real]):  # noqa: F821
    """A connective of an integer inside a comparison, in a guard."""
    for i in k.dom:
        with when((k[i] & 1) != 0):
            y[i] = 1.0


@kernel
def guarded_bits(k: Arr[Fin[n], Int], y: Arr[Fin[n], Real]):  # noqa: F821
    """A connective of an integer and a comparison, as a guard."""
    for i in k.dom:
        with when((k[i] > 0) | k[i]):
            y[i] = 1.0


@kernel
def summed_bits(k: Arr[Fin[n], Int], y: Arr[Fin[n], Int]):  # noqa: F821
    """``~`` of an integer, in a sum's body."""
    for i in k.dom:
        y[i] = reduce_sum(~k[j] for j in k.dom)


@kernel
def low_bits(
    k: Arr[Fin[n], Int],  # noqa: F821
    x: Arr[Fin[n], Real],  # noqa: F821
    y: Arr[Fin[n], Real],  # noqa: F821
):
    """The fix: the low bit as arithmetic, and as a comparison in a guard."""
    for i in k.dom:
        y[i] = (k[i] % 2) * x[i]
        with when((k[i] % 2 != 0) & (x[i] > 0.5)):
            y[i] = y[i] + 1.0


def test_a_connective_of_an_integer_is_refused_by_the_trace():
    # Natively 2 & 1 is 0, and the compiled kernel computed 2 and 1, true.
    def make() -> dict:
        return {"k": np.array([2, 3]), "x": np.ones(2), "y": np.zeros(2)}

    native = make()
    masked_scale(**native)
    assert list(native["y"]) == [0.0, 1.0]
    for kern, operand in (
        (masked_scale, r"k\[i\]"),
        (odd_compared, r"k\[i\]"),
        (guarded_bits, r"k\[i\]"),
        (summed_bits, r"k\[j\]"),
    ):
        with pytest.raises(TraceError) as refused:
            kern.trace()
        message = str(refused.value)
        assert "is a connective, and its operand" in message
        assert f"{operand.replace(chr(92), '')} != 0" in message
        assert "k % 2 for k & 1" in message
        (fact,) = kern.facts()
        assert fact.kind == "trace" and fact.status.value == "refuted"
        assert "is not a truth value" in fact.provenance["reason"]


def test_the_named_fix_agrees():
    def make() -> dict:
        return {
            "k": np.array([2, 3, -3, 4]),
            "x": np.array([1.0, 1.0, 0.25, 0.75]),
            "y": np.zeros(4),
        }

    native = make()
    low_bits(**native)
    assert list(native["y"]) == [0.0, 2.0, 0.25, 0.0]
    agrees(low_bits, make)


# }}}


# {{{ a power (#84)


@kernel
def inverse(x: Arr[Fin[n], Real], y: Arr[Fin[n], Real]):  # noqa: F821
    """A negative integer exponent: a call of loopy's integer power."""
    for i in x.dom:
        y[i] = x[i] ** -1


@kernel
def inverse_root(x: Arr[Fin[n], Real], y: Arr[Fin[n], Real]):  # noqa: F821
    """A floating exponent: a call of ``pow``, whose header loopy never included."""
    for i in x.dom:
        y[i] = x[i] ** -0.5


@kernel
def cubes(
    x: Arr[Fin[n], Real.exact],  # noqa: F821
    k: Arr[Fin[n], Int],  # noqa: F821
    y: Arr[Fin[n], Real.exact],  # noqa: F821
    h: Arr[Fin[n], Int],  # noqa: F821
):
    """A cube of a real, which numpy computes with ``pow``, and of an integer."""
    for i in x.dom:
        y[i] = x[i] ** 3
        h[i] = k[i] ** 3


def test_a_power_compiles_and_is_computed_with_pow():
    def make() -> dict:
        return {"x": np.array([0.5, 3.0, 1.4554425309821815]), "y": np.zeros(3)}

    for kern in (inverse, inverse_root):
        code = emit_code(kern)
        assert "#include <math.h>" in code
        assert "pow(" in code and "loopy_pow" not in code
        agrees(kern, make)

    # x ** 3 at this x is not (x * x) * x, which loopy's integer power
    # computed: numpy calls pow, and so does the lowered code now. An integer
    # cube stays an integer power, whose definition needs <stdint.h> first.
    x = 1.4554425309821815
    assert np.float64(x) ** 3 != (x * x) * x

    def cubed() -> dict:
        return {
            "x": np.array([x, 1.7199053588004087, -2.0]),
            "k": np.array([2, -3, 1290]),
            "y": np.zeros(3),
            "h": np.zeros(3, np.int64),
        }

    code = emit_code(cubes)
    assert "pow(x[i], 3.0)" in code
    assert "loopy_pow_int32_int32(k[i], 3)" in code
    assert code.index("#include <stdint.h>") < code.index("loopy_pow_int32_int32")
    agrees(cubes, cubed)


# }}}


# {{{ a literal beside a float32 (#91)


@kernel
def f32_scale(
    x: Arr[Fin[n], np.float32],  # noqa: F821
    y: Arr[Fin[n], np.float32],  # noqa: F821
):
    """A Python float beside a ``float32`` is a ``float32`` natively (NEP 50)."""
    for i in x.dom:
        y[i] = x[i] * 0.1 + 0.3


@kernel
def f32_mixed(
    x: Arr[Fin[n], np.float32],  # noqa: F821
    k: Arr[Fin[n], Int],  # noqa: F821
    y: Arr[Fin[n], Real],  # noqa: F821
    z: Arr[Fin[n], np.float32],  # noqa: F821
):
    """A ``float32`` beside an integer array, and beside a Python float sum."""
    for i in x.dom:
        y[i] = x[i] / k[i]
        z[i] = x[i] * (0.3 + i)


@kernel
def turn32(
    x: Arr[Fin[n], np.float32],  # noqa: F821
    z: Arr[Fin[n], np.complex64],  # noqa: F821
):
    """A Python complex beside a ``float32`` is a ``complex64`` natively."""
    for i in x.dom:
        z[i] = x[i] * (0.1 + 0.2j)


@kernel
def halve(k: Arr[Fin[n], Int], h: Arr[Fin[n], Real]):  # noqa: F821
    """A Python float beside an integer is a double, as before."""
    for i in k.dom:
        h[i] = k[i] * 0.5


def test_a_literal_beside_a_float32_is_single_precision():
    # The literals were doubles, so x * 0.1 + 0.3 was computed in double and
    # rounded at the store, a bit away from numpy's single precision.
    rng = np.random.default_rng(91)

    def make() -> dict:
        x = rng.uniform(-4.0, 4.0, 64).astype(np.float32)
        return {"x": x, "y": np.zeros(64, np.float32)}

    code = emit_code(f32_scale)
    assert "x[i] * 0.10000000149011612f + 0.30000001192092896f" in code
    data = make()
    agrees(f32_scale, lambda: {name: value.copy() for name, value in data.items()})
    assert "k[i] * 0.5;" in emit_code(halve)

    def turned() -> dict:
        return {
            "x": data["x"].copy(),
            "z": np.zeros(64, np.complex64),
        }

    code = emit_code(turn32)
    assert "x[i] * (0.10000000149011612f + 0.20000000298023224f * I)" in code
    agrees(turn32, turned)


def test_a_float32_beside_an_integer_or_a_python_float_is_what_numpy_makes():
    # x / k of a float32 x and an integer k is a double natively, and was a
    # float compiled; 0.3 + i is a Python float, which numpy rounds to single
    # precision beside x, and which was a double compiled.
    def make() -> dict:
        return {
            "x": np.array([1.0, 0.1, 3.5, -7.25], np.float32),
            "k": np.array([3, 7, -9, 11]),
            "y": np.zeros(4),
            "z": np.zeros(4, np.float32),
        }

    native = make()
    f32_mixed(**native)
    assert native["y"][0] == 1.0 / 3.0
    code = emit_code(f32_mixed)
    assert "(double) (x[i]) / k[i]" in code
    assert "(float) (0.3 + i)" in code
    agrees(f32_mixed, make)


# }}}


# {{{ an integral value outside 32 bits (#92)


@kernel
def plus_one(c: Arr[Fin[n], Nat], d: Arr[Fin[n], Nat]):  # noqa: F821
    """Every entry plus one."""
    for i in c.dom:
        d[i] = c[i] + 1


@kernel
def shifted(c: Arr[Fin[n], Int], s: Int, d: Arr[Fin[n], Int]):  # noqa: F821
    """Every entry plus a scalar."""
    for i in c.dom:
        d[i] = c[i] + s


def test_an_integral_value_outside_32_bits_is_refused():
    # 2**32 + 5 ran natively as it was and was 5 compiled.
    low, high = INTEGRAL_RANGE
    assert (low, high) == (-(2**31), 2**31)
    info = np.iinfo(numpy_dtype(Nat))
    assert (int(info.min), int(info.max) + 1) == INTEGRAL_RANGE

    def make(c) -> dict:
        return {"c": c, "d": np.zeros(len(c), np.int64)}

    message = r"c\[0\] is 4294967301, which is outside -2147483648 <= v < 2147483648"
    for run in (
        lambda data: plus_one(**data),
        lambda data: LoopyExecutor().run(plus_one, **data),
        lambda data: LoopyExecutor().differential(plus_one, Schedule(plus_one), data),
    ):
        with pytest.raises(ValueError, match=message):
            run(make(np.array([2**32 + 5, 1])))
    with pytest.raises(
        ValueError, match="declare the elements of c as a numpy integer"
    ):
        plus_one(**make(np.array([2**32 + 5, 1])))
    # A uint64 entry was read natively through an int64 copy, as a negative
    # number, and a float-stored one is a whole number outside the range.
    with pytest.raises(ValueError, match=r"c\[1\] is 9223372036854775813, which is"):
        plus_one(**make(np.array([1, 2**63 + 5], np.uint64)))
    with pytest.raises(
        ValueError, match=r"c\[0\] is 1099511627776.0, which is outside"
    ):
        plus_one(**make(np.array([2.0**40, 1.0])))
    agrees(plus_one, lambda: make(np.array([high - 2, 0])))

    def scalar(s) -> dict:
        return {"c": np.array([low, 5]), "s": s, "d": np.zeros(2, np.int64)}

    for s in (high, np.int64(2**40), low - 1):
        with pytest.raises(
            ValueError, match=f"the argument s is {s}, which is outside"
        ):
            shifted(**scalar(s))
        with pytest.raises(ValueError, match="declare s as a numpy integer"):
            LoopyExecutor().run(shifted, **scalar(s))
    agrees(shifted, lambda: scalar(7))


# }}}
