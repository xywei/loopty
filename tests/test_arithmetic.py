"""Arithmetic in both runs: what C computes and what numpy computes.

The native run computes with numpy's arithmetic and the compiled run with C's,
and the two type an operation differently: numpy by NEP 50, where a Python
number takes the dtype of what stands beside it, C by its usual arithmetic
conversions (:mod:`loopty.promotion`), and C leaves arithmetic undefined that
numpy defines (:mod:`loopty.operations`). The ways the runs disagreed: an
integer quotient (#82), a connective of integers (#83), a power (#84), a
literal beside a ``float32`` (#91), an integral value outside 32 bits (#92),
an integer result outside 32 bits (#101), a scalar weak or strong by the call
(#102), floating ``%`` and ``//`` (#104), integer ``//`` and ``%`` by zero
(#105), a sum of truth values (#106), ``^``, ``<<`` and ``>>`` (#107), a kernel
named like a library function (#108), and an integer to a negative power
(#109). Each kernel here either agrees with its native run bit for bit, or is
refused, by the trace or the contract, with the fix named.
"""

from __future__ import annotations

import types

import numpy as np
import pymbolic.primitives as prim
import pytest
from lanky.prelude import Bool, Int, Nat, Real

from loopty import Arr, Fin, Schedule, TraceError, kernel, program, reduce_sum, when
from loopty.contract import (
    INDEX_RANGE,
    INTEGRAL_RANGE,
    element_types,
    integral_range,
)
from loopty.executor import LoopyExecutor, emit_code
from loopty.lower import numpy_dtype
from loopty.term import ArrType


def agrees(kern, make) -> None:
    """The compiled run of ``kern`` agrees bit for bit with its native run."""
    native = make()
    kern(**native)
    compiled = make()
    LoopyExecutor().run(kern, **compiled)
    for name, value in native.items():
        if isinstance(value, np.ndarray):
            assert value.dtype == compiled[name].dtype
            assert same_bits(value, compiled[name]), (name, value, compiled[name])
    fact = LoopyExecutor().differential(kern, Schedule(kern), make())
    assert fact.status.value == "tested", fact.provenance


def same_bits(left: np.ndarray, right: np.ndarray) -> bool:
    """Equal values, a zero's sign included, and ``nan`` at the same cells."""
    if left.dtype.kind not in "fc":
        return np.array_equal(left, right)
    nan = np.isnan(left)
    if not np.array_equal(nan, np.isnan(right)):
        return False
    return np.array_equal(
        left[~nan].view(np.uint8), right[~nan].view(np.uint8)
    )


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


@program
def halved_twice(k, h):
    """``int_quot`` twice, through an integer array the program makes."""
    g = Arr.zeros_like(k)
    int_quot(k, g)
    int_quot(g, h)


def test_a_quotient_through_a_program_temporary_is_true_division():
    # The program's term types its temporary as the kernel's argument, and the
    # quotient of that temporary is a double compiled, as natively.
    def make() -> dict:
        return {"k": np.array([3, -3, 4, 7]), "h": np.zeros(4, np.int64)}

    native = make()
    halved_twice(**native)
    assert list(native["h"]) == [3, -3, 4, 7]
    assert "(double) (g[" in emit_code(halved_twice)
    agrees(halved_twice, make)


@kernel
def floors(
    k: Arr[Fin[n], Int],  # noqa: F821
    m: Arr[Fin[n], Int],  # noqa: F821
    s: Int,
    a: Arr[Fin[n], Int],  # noqa: F821
    b: Arr[Fin[n], Int],  # noqa: F821
    c: Arr[Fin[n], Int],  # noqa: F821
    d: Arr[Fin[n], Int],  # noqa: F821
    e: Arr[Fin[n], Int],  # noqa: F821
    f: Arr[Fin[n], Int],  # noqa: F821
):
    """Floor division and remainder by a constant of each sign, an array, a scalar."""
    for i in k.dom:
        a[i] = k[i] // 3 - k[i] % 3
        b[i] = k[i] // -3
        c[i] = k[i] % -3
        d[i] = k[i] // m[i]
        e[i] = k[i] % m[i]
        f[i] = k[i] // s + 100 * (k[i] % s)


def test_floor_division_and_remainder_follow_python_on_negatives():
    # loopy lowers // and % of integers to its floor division and modulo,
    # which round toward minus infinity as Python does whatever the signs:
    # the operation planning around them leaves an integer operation alone.
    k = [7, -7, 6, -6, 0, -1, 1, 5]
    m = [2, 2, -4, -4, 3, -3, -2, 5]

    def make(s) -> dict:
        outputs = {name: np.zeros(len(k), np.int64) for name in "abcdef"}
        return {"k": np.array(k), "m": np.array(m), "s": s, **outputs}

    for s in (3, -3, np.int64(-3)):
        native = make(s)
        floors(**native)
        assert list(native["a"]) == [p // 3 - p % 3 for p in k]
        assert list(native["b"]) == [p // -3 for p in k]
        assert list(native["c"]) == [p % -3 for p in k]
        assert list(native["d"]) == [p // q for p, q in zip(k, m, strict=True)]
        assert list(native["e"]) == [p % q for p, q in zip(k, m, strict=True)]
        assert list(native["f"]) == [p // int(s) + 100 * (p % int(s)) for p in k]
        agrees(floors, lambda s=s: make(s))


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


@kernel
def low_bit_cell(x: Arr[Fin[n], Real], y: Arr[Fin[n], Real]):  # noqa: F821
    """A connective of a loop variable in the index of the cell written."""
    for i in x.dom:
        y[i & 1] = x[i]


@kernel
def complement_scale(x: Arr[Fin[n], Real], y: Arr[Fin[n], Real]):  # noqa: F821
    """``~`` of a comparison of a loop variable, used as a number."""
    for i in x.dom:
        y[i] = x[i] * ~(i > 0)


@kernel
def complement_chain(x: Arr[Fin[n], Real], y: Arr[Fin[n], Real]):  # noqa: F821
    """``~`` of a connective of such comparisons, inside a sum's body."""
    for i in x.dom:
        y[i] = reduce_sum(x[j] * ~((j > 0) & (j < i)) for j in x.dom)


@kernel
def complement_read(
    x: Arr[Fin[n], Real],  # noqa: F821
    f: Bool,
    y: Arr[Fin[n], Real],  # noqa: F821
):
    """``~`` of a comparison of an array's element, and of a ``Bool`` scalar."""
    for i in x.dom:
        y[i] = x[i] * ~(x[i] > 0.5) + ~f


@pytest.mark.filterwarnings("ignore:Bitwise inversion:DeprecationWarning")
def test_a_complement_of_a_python_bool_is_refused_where_it_is_a_number():
    # i > 0 is a Python bool natively, and ~ of one is bitwise, -2 or -1, so y
    # was -2 * x[i] natively and 0 compiled. A guard or a cell of an array of
    # truth values refuses that integer natively; anywhere else the trace
    # refuses it, naming the complement.
    def make() -> dict:
        return {"x": np.array([0.25, 0.75, 1.0]), "y": np.zeros(3)}

    native = make()
    complement_scale(**native)
    assert list(native["y"]) == [-0.25, -1.5, -2.0]
    with pytest.raises(TraceError, match=r"'i <= 0' for '~\(i > 0\)'"):
        complement_scale.trace()
    with pytest.raises(TraceError, match="computes as a Python bool"):
        complement_chain.trace()
    (fact,) = complement_scale.facts()
    assert fact.kind == "trace" and fact.status.value == "refuted"
    # The index of the cell written is asked too: y[i & 1] wrote y[0] and
    # y[1] natively, and y[i and 1] does not even index compiled.
    with pytest.raises(TraceError, match=r"its operand i is not a truth value"):
        low_bit_cell.trace()
    # ~ of a numpy bool is logical natively, as it is compiled: an element
    # compared, and a Bool scalar, which the native run makes a numpy bool
    # however it is passed.
    for f in (True, np.True_, False):
        agrees(complement_read, lambda f=f: {**make(), "f": f})


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
    assert "loopy_pow_int64_int32(k[i], 3)" in code
    assert code.index("#include <stdint.h>") < code.index("loopy_pow_int64_int32")
    agrees(cubes, cubed)


@kernel
def complex_cube(
    z: Arr[Fin[n], np.complex128],  # noqa: F821
    w: Arr[Fin[n], np.complex128],  # noqa: F821
):
    """A complex cube: loopy's integer power over ``double complex``."""
    for i in z.dom:
        w[i] = z[i] ** 3


@kernel
def complex_roots(
    z: Arr[Fin[n], np.complex128],  # noqa: F821
    v: Arr[Fin[n], np.complex128],  # noqa: F821
    u: Arr[Fin[n], np.complex128],  # noqa: F821
):
    """A complex square root by ``cpow``, and a negative integer power."""
    for i in z.dom:
        v[i] = z[i] ** 0.5
        u[i] = z[i] ** -2


def test_a_complex_power_compiles():
    # loopy's integer power for a complex base names ``double complex`` in its
    # signature, above the complex.h it includes, so z ** 3 did not compile.
    # It multiplies as numpy's complex power does for a positive exponent, so
    # the cube agrees bit for bit; a negative one is inverted first, which
    # numpy does last, and is compared at the class of a bare complex dtype.
    rng = np.random.default_rng(84)
    z = rng.uniform(-2.0, 2.0, 32) + 1j * rng.uniform(-2.0, 2.0, 32)

    def make(*names: str) -> dict:
        return {"z": z.copy(), **{name: np.zeros(32, complex) for name in names}}

    code = emit_code(complex_cube)
    assert code.index("#include <complex.h>") < code.index("loopy_pow_complex128")
    agrees(complex_cube, lambda: make("w"))
    LoopyExecutor().run(complex_roots, **make("v", "u"))
    fact = LoopyExecutor().differential(
        complex_roots, Schedule(complex_roots), make("v", "u")
    )
    assert fact.status.value == "tested", fact.provenance


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


@kernel
def f32_compared(
    x: Arr[Fin[n], np.float32],  # noqa: F821
    k: Arr[Fin[n], Int],  # noqa: F821
    b: Arr[Fin[n], Bool],  # noqa: F821
    y: Arr[Fin[n], np.float32],  # noqa: F821
):
    """A ``float32`` compared with a Python float, and in a guard with an integer."""
    for i in x.dom:
        b[i] = x[i] > 0.1
        with when(x[i] < k[i]):
            y[i] = 1.0


def test_a_float32_comparison_is_what_numpy_compares():
    # numpy compares 0.1f with 0.1 in single precision, where they are equal,
    # and C compared them in double, where 0.1f is larger; and it compares a
    # float32 with a 64-bit integer in double, where C rounded 2**24 + 1 to a
    # float, equal to 2**24.
    def make() -> dict:
        return {
            "x": np.array([0.1, 2.0**24, 0.5], np.float32),
            "k": np.array([0, 2**24 + 1, 1]),
            "b": np.zeros(3, bool),
            "y": np.zeros(3, np.float32),
        }

    native = make()
    f32_compared(**native)
    assert list(native["b"]) == [False, True, True]
    assert list(native["y"]) == [0.0, 1.0, 1.0]
    code = emit_code(f32_compared)
    assert "x[i] > 0.10000000149011612f" in code
    assert "(double) (x[i]) < k[i]" in code
    agrees(f32_compared, make)


# }}}


# {{{ an integral value outside its compiled range (#92, #101)


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


def test_an_integral_value_outside_its_compiled_range_is_refused():
    # Nat and Int are 64 bits wide in both runs (#101), so 2**32 + 5, which
    # #92 refused as narrowed to 5 compiled, runs; a uint64 entry from 2**63
    # on is outside them, and was read natively as a negative number.
    low, high = INTEGRAL_RANGE
    assert (low, high) == (-(2**63), 2**63)
    info = np.iinfo(numpy_dtype(Nat))
    assert (int(info.min), int(info.max) + 1) == INTEGRAL_RANGE
    assert integral_range(Fin[8]) == INDEX_RANGE == (-(2**31), 2**31)

    def make(c) -> dict:
        return {"c": c, "d": np.zeros(len(c), np.int64)}

    agrees(plus_one, lambda: make(np.array([2**32 + 5, high - 2])))
    message = r"c\[1\] is 9223372036854775813, which is outside"
    for run in (
        lambda data: plus_one(**data),
        lambda data: LoopyExecutor().run(plus_one, **data),
        lambda data: LoopyExecutor().differential(plus_one, Schedule(plus_one), data),
    ):
        with pytest.raises(ValueError, match=message):
            run(make(np.array([1, 2**63 + 5], np.uint64)))
    with pytest.raises(ValueError, match="declare the elements of c as a numpy"):
        plus_one(**make(np.array([1, 2**63 + 5], np.uint64)))

    def scalar(s) -> dict:
        return {"c": np.array([low + 2**33, 5]), "s": s, "d": np.zeros(2, np.int64)}

    for s in (2**31, np.int64(2**40), -(2**31) - 1):
        agrees(shifted, lambda s=s: scalar(s))
    for s in (high, low - 1):
        refused = f"the argument s is {s}, which is outside"
        with pytest.raises(ValueError, match=refused):
            shifted(**scalar(s))
        with pytest.raises(ValueError, match=refused):
            LoopyExecutor().run(shifted, **scalar(s))

    # Fin[m] stays 32 bits wide compiled: its bound keeps an index array
    # compact. A point of a bound past 2**31 that leaves 32 bits is refused.
    n = prim.Variable("n")
    types = {"c": ArrType(axes=(n,), dtype=Fin[2**40], ragged=(False,))}
    element_types(types, {"c": np.array([2**31 - 1, 0])})
    with pytest.raises(ValueError, match=r"c\[0\] is 4294967301, which is outside"):
        element_types(types, {"c": np.array([2**32 + 5, 0])})


# }}}


# {{{ an integer result outside 32 bits (#101)


@kernel
def scaled_square(c: Arr[Fin[n], Int], d: Arr[Fin[n], Int]):  # noqa: F821
    """A square of an integer that leaves 32 bits, brought back by a division."""
    for i in c.dom:
        d[i] = c[i] * c[i] // 1024


@kernel
def index_squares(
    col: Arr[Fin[n], Fin[m]],  # noqa: F821
    x: Arr[Fin[m], Real],  # noqa: F821
    d: Arr[Fin[n], Int],  # noqa: F821
    e: Arr[Fin[n], Int],  # noqa: F821
    y: Arr[Fin[n], Real],  # noqa: F821
):
    """Squares of an index array's entries and of a loop variable."""
    for i in col.dom:
        d[i] = col[i] * col[i] + 1
        e[i] = i * i
        y[i] = x[col[i]]


@kernel
def counted(b: Arr[Fin[n], Bool], c: Arr[Fin[n], Int]):  # noqa: F821
    """The number of true entries: a sum of truth values, which is a count."""
    for i in b.dom:
        c[i] = reduce_sum(b[j] for j in b.dom)


def test_integer_arithmetic_is_64_bits_wide_compiled_too():
    # c[i] * c[i] at 2**20 is 2**40, which wrapped round to 0 in the 32 bits
    # the compiled run computed in; numpy computes it in 64.
    def make() -> dict:
        return {"c": np.array([2**20, 3, -(2**31)]), "d": np.zeros(3, np.int64)}

    native = make()
    scaled_square(**native)
    assert list(native["d"]) == [2**30, 0, 2**52]
    assert "int64_t" in emit_code(scaled_square)
    agrees(scaled_square, make)

    # A Fin[m] array stays 32 bits wide, and a product of its entries, or of
    # a loop variable, is computed in 64; a subscript is not widened.
    m = 50_000
    rows = 46_342

    def indices() -> dict:
        col = np.zeros(rows, np.int64)
        col[:3] = [m - 1, 3, 46_341]
        return {
            "col": col,
            "x": np.arange(m, dtype=np.float64),
            "d": np.zeros(rows, np.int64),
            "e": np.zeros(rows, np.int64),
            "y": np.zeros(rows),
        }

    native = indices()
    index_squares(**native)
    assert list(native["d"][:3]) == [(m - 1) ** 2 + 1, 10, 46_341**2 + 1]
    assert native["e"][-1] == (rows - 1) ** 2 > 2**31
    code = emit_code(index_squares)
    assert "int32_t const *__restrict__ col" in code
    assert "(int64_t) (col[i]) * col[i] + 1" in code
    assert "(int64_t) (i) * i" in code
    assert "y[i] = x[col[i]]" in code
    agrees(index_squares, indices)

    # A sum of truth values is a count natively, which a byte held to 127.
    def truths() -> dict:
        return {"b": np.ones(300, bool), "c": np.zeros(300, np.int64)}

    native = truths()
    counted(**native)
    assert native["c"][0] == 300
    agrees(counted, truths)


# }}}


# {{{ a scalar weak or strong by the call (#102)


@kernel
def scale_by(
    x: Arr[Fin[n], np.float32],  # noqa: F821
    a: Real,
    s: Int,
    y: Arr[Fin[n], np.float32],  # noqa: F821
    z: Arr[Fin[n], Real],  # noqa: F821
):
    """A float32 array beside a real scalar and an integral one."""
    for i in x.dom:
        y[i] = x[i] * a * 0.1
        z[i] = x[i] * s


def test_a_scalar_has_one_native_meaning_however_it_is_passed():
    # a=0.7 was a weak Python float natively, so x * a was single precision,
    # and a=np.float64(0.7) a strong one, double; the compiled run computes in
    # double. An integral s was weak or strong the same way.
    rng = np.random.default_rng(102)
    x = rng.uniform(-4.0, 4.0, 64).astype(np.float32)

    def make(a, s) -> dict:
        return {
            "x": x.copy(),
            "a": a,
            "s": s,
            "y": np.zeros(64, np.float32),
            "z": np.zeros(64),
        }

    results = []
    for a, s in ((0.7, 3), (np.float64(0.7), np.int64(3))):
        native = make(a, s)
        scale_by(**native)
        results.append(native)
        agrees(scale_by, lambda a=a, s=s: make(a, s))
    assert np.array_equal(results[0]["y"], results[1]["y"])
    assert np.array_equal(results[0]["z"], results[1]["z"])
    assert results[0]["z"][0] == np.float64(x[0]) * 3


# }}}


# {{{ floating % and // (#104)


@kernel
def wrapped(
    x: Arr[Fin[n], Real],  # noqa: F821
    q: Arr[Fin[n], Real],  # noqa: F821
    y: Arr[Fin[n], Real],  # noqa: F821
    z: Arr[Fin[n], Real],  # noqa: F821
    u: Arr[Fin[n], Real],  # noqa: F821
    v: Arr[Fin[n], Real],  # noqa: F821
):
    """``%`` and ``//`` of reals, by a constant and by an array."""
    for i in x.dom:
        y[i] = x[i] % 2.0
        z[i] = x[i] // -2.5
        u[i] = x[i] % q[i]
        v[i] = x[i] // q[i]


@kernel
def wrapped32(
    x: Arr[Fin[n], np.float32],  # noqa: F821
    y: Arr[Fin[n], np.float32],  # noqa: F821
    z: Arr[Fin[n], np.float32],  # noqa: F821
):
    """``%`` and ``//`` of a float32, in single precision as numpy computes."""
    for i in x.dom:
        y[i] = x[i] % 0.3
        z[i] = x[i] // 0.3


@pytest.mark.filterwarnings("ignore::RuntimeWarning")
def test_floating_remainder_and_floor_division_are_numpys():
    # loopy refused to generate either, from inside its code generator. numpy
    # computes them with fmod, corrected toward the divisor's sign.
    rng = np.random.default_rng(104)
    x = np.concatenate(
        [
            rng.uniform(-10.0, 10.0, 60),
            [0.0, -0.0, 7.5, -7.5, 1e300, np.inf, -np.inf, np.nan, 5.0, 1.0],
        ]
    )
    q = np.concatenate(
        [
            rng.uniform(-3.0, 3.0, 60),
            [-2.0, 2.0, 2.5, 2.5, 3.0, 2.0, 2.0, 2.0, 0.0, -0.0],
        ]
    )

    def make() -> dict:
        return {"x": x.copy(), "q": q.copy(), **{k: np.zeros(70) for k in "yzuv"}}

    native = make()
    wrapped(**native)
    assert native["y"][62] == 1.5 and native["z"][62] == -3.0
    assert np.signbit(native["u"][61]) == np.signbit(np.float64(-0.0) % 2.0)
    code = emit_code(wrapped)
    assert "loopty_mod_float64(x[i], 2.0)" in code
    assert "loopty_floor_div_float64(x[i], q[i])" in code
    agrees(wrapped, make)

    def single() -> dict:
        return {
            "x": rng.uniform(-10.0, 10.0, 64).astype(np.float32),
            "y": np.zeros(64, np.float32),
            "z": np.zeros(64, np.float32),
        }

    data = single()
    assert "loopty_mod_float32" in emit_code(wrapped32)
    agrees(wrapped32, lambda: {k: v.copy() for k, v in data.items()})


# }}}


# {{{ integer // and % by zero (#105)


@kernel
def divided(
    k: Arr[Fin[n], Int],  # noqa: F821
    m: Arr[Fin[n], Int],  # noqa: F821
    s: Int,
    a: Arr[Fin[n], Int],  # noqa: F821
    b: Arr[Fin[n], Int],  # noqa: F821
    c: Arr[Fin[n], Int],  # noqa: F821
):
    """Floor division and remainder by an array and by a scalar, of any value."""
    for i in k.dom:
        a[i] = k[i] // m[i]
        b[i] = k[i] % m[i]
        c[i] = k[i] // s + k[i] % s


@kernel
def divisible(
    k: Arr[Fin[n], Int],  # noqa: F821
    m: Arr[Fin[n], Int],  # noqa: F821
    y: Arr[Fin[n], Real],  # noqa: F821
):
    """A remainder in a guard."""
    for i in k.dom:
        with when(k[i] % m[i] == 0):
            y[i] = 1.0


@pytest.mark.filterwarnings("ignore::RuntimeWarning")
def test_integer_division_by_zero_and_by_minus_one_is_numpys():
    # C's / and % by zero killed the process with SIGFPE, and so did the
    # smallest integer by -1; numpy gives 0, and the smallest integer and 0.
    def make(s, k, m) -> dict:
        outputs = {name: np.zeros(len(k), np.int64) for name in "abc"}
        return {"k": np.array(k), "m": np.array(m), "s": s, **outputs}

    small = make(0, [7, -7, 0], [0, 0, 0])
    LoopyExecutor().run(divided, **small)
    assert list(small["a"]) == list(small["b"]) == list(small["c"]) == [0, 0, 0]

    low = -(2**63)
    k = [7, -7, 7, -7, 0, low, low, 2**63 - 1, low + 1, 5]
    m = [0, 0, -1, -1, 0, -1, 3, -2, -3, 2]
    for s in (0, -1, 3, np.int64(-3)):
        native = make(s, k, m)
        divided(**native)
        with np.errstate(all="ignore"):
            want = np.array(k) // np.array(m)
        assert list(native["a"]) == list(want)
        assert native["a"][0] == 0 and native["b"][0] == 0
        assert native["a"][5] == low and native["b"][5] == 0
        agrees(divided, lambda s=s: make(s, k, m))

    def guarded() -> dict:
        return {"k": np.array(k), "m": np.array(m), "y": np.zeros(len(k))}

    agrees(divisible, guarded)


# }}}


# {{{ a sum of truth values (#106)


@kernel
def doubled(b: Arr[Fin[n], Bool], y: Arr[Fin[n], Real]):  # noqa: F821
    """A sum of two truth values: ``or`` natively, ``2`` compiled."""
    for i in b.dom:
        y[i] = (b[i] + b[i]) * 1.0


@kernel
def either_or_both(
    b: Arr[Fin[n], Bool],  # noqa: F821
    c: Arr[Fin[n], Bool],  # noqa: F821
    y: Arr[Fin[n], Real],  # noqa: F821
    z: Arr[Fin[n], Real],  # noqa: F821
):
    """The two fixes: ``|`` for ``or``, and ``1 * b + c`` for a count."""
    for i in b.dom:
        y[i] = (b[i] | c[i]) * 1.0
        z[i] = (1 * b[i] + c[i]) * 1.0


@kernel
def guarded_sum(
    b: Arr[Fin[n], Bool],  # noqa: F821
    f: Bool,
    y: Arr[Fin[n], Real],  # noqa: F821
):
    """A sum of a truth value and a truth scalar, in a guard."""
    for i in b.dom:
        with when(b[i] + f > 1):
            y[i] = 1.0


def test_a_sum_of_truth_values_is_refused_by_the_trace():
    # numpy's + of two bools is or, so True + True is True natively; the
    # compiled kernel added the bytes and made 2.
    def make() -> dict:
        return {"b": np.array([True, False]), "y": np.zeros(2)}

    native = make()
    doubled(**native)
    assert list(native["y"]) == [1.0, 0.0]
    for kern in (doubled, guarded_sum):
        with pytest.raises(TraceError, match="adds truth values") as refused:
            kern.trace()
        assert "for 'or', or '1 * " in str(refused.value)
        (fact,) = kern.facts()
        assert fact.kind == "trace" and fact.status.value == "refuted"
    assert "'b[i] | b[i]' for 'or', or '1 * b[i] + b[i]' for a count" in str(
        pytest.raises(TraceError, doubled.trace).value
    )

    def both() -> dict:
        return {
            "b": np.array([True, True, False, False]),
            "c": np.array([True, False, True, False]),
            "y": np.zeros(4),
            "z": np.zeros(4),
        }

    native = both()
    either_or_both(**native)
    assert list(native["y"]) == [1.0, 1.0, 1.0, 0.0]
    assert list(native["z"]) == [2.0, 1.0, 1.0, 0.0]
    agrees(either_or_both, both)


# }}}


# {{{ ^, << and >> (#107)


@kernel
def flipped(
    k: Arr[Fin[n], Int],  # noqa: F821
    t: Arr[Fin[n], Int],  # noqa: F821
    x: Arr[Fin[n], Fin[n]],  # noqa: F821
    y: Arr[Fin[n], Int],  # noqa: F821
    z: Arr[Fin[n], Int],  # noqa: F821
    w: Arr[Fin[n], Int],  # noqa: F821
    v: Arr[Fin[n], Int],  # noqa: F821
):
    """``^``, ``<<`` and ``>>``, by constants, by an array, of an index array."""
    for i in k.dom:
        y[i] = k[i] ^ 3
        z[i] = (k[i] << 2) + (k[i] >> 1)
        w[i] = (k[i] << t[i]) ^ (k[i] >> t[i])
        v[i] = (x[i] << 40) + (i ^ x[i])


@kernel
def real_bits(x: Arr[Fin[n], Real], y: Arr[Fin[n], Real]):  # noqa: F821
    """``<<`` of a real, which numpy refuses."""
    for i in x.dom:
        y[i] = x[i] << 2


def test_bitwise_xor_and_shifts_are_numpys():
    # Each failed to lower with an empty NotImplementedError. A shift past the
    # width, or by a negative amount, is undefined in C and 0 (or -1) in numpy.
    k = [5, -5, 1, -1, 2**62, -(2**63), 7, 0]
    t = [0, 1, 63, 64, 70, -1, 2, 65]

    def make() -> dict:
        outputs = {name: np.zeros(len(k), np.int64) for name in "yzwv"}
        return {
            "k": np.array(k),
            "t": np.array(t),
            "x": np.arange(len(k))[::-1].copy(),
            **outputs,
        }

    code = emit_code(flipped)
    assert "k[i] ^ 3" in code
    assert "loopty_lshift_int64(k[i], t[i])" in code
    native = make()
    flipped(**native)
    assert list(native["y"]) == [p ^ 3 for p in k]
    assert native["w"][3] == np.int64(-1) >> 64 == -1
    assert native["v"][0] == (7 << 40) + 7
    agrees(flipped, make)
    with pytest.raises(TraceError, match="is not an integer") as refused:
        real_bits.trace()
    assert "k * 4 for k << 2" in str(refused.value)


# }}}


# {{{ a kernel named like a function loopy or the C library knows (#108)


def _named(name: str, body):
    """A copy of ``body`` as a kernel called ``name``."""
    copy = types.FunctionType(
        body.__code__, body.__globals__, name, body.__defaults__, body.__closure__
    )
    copy.__annotations__ = dict(body.__annotations__)
    copy.__qualname__ = name
    return kernel(copy)


def _doubled(x: Arr[Fin[n], Real], y: Arr[Fin[n], Real]):  # noqa: F821
    for i in x.dom:
        y[i] = x[i] * 2.0


def _cubed(x: Arr[Fin[n], Real], y: Arr[Fin[n], Real]):  # noqa: F821
    for i in x.dom:
        y[i] = x[i] ** 3


def _complex_cubed(
    z: Arr[Fin[n], np.complex128],  # noqa: F821
    w: Arr[Fin[n], np.complex128],  # noqa: F821
):
    for i in z.dom:
        w[i] = z[i] ** 3


def test_a_kernel_named_like_a_library_function_is_renamed():
    # floor failed inside loopy with KeyError: 'floor', and cpow over complex
    # arrays with a power clashed with complex.h's cpow in the compiler.
    def reals() -> dict:
        return {"x": np.array([0.5, -2.0, 3.0]), "y": np.zeros(3)}

    for name in ("floor", "conj", "make_tuple", "fabs"):
        kern = _named(name, _doubled)
        assert f"void {name}_knl(" in emit_code(kern)
        agrees(kern, reals)
    for name in ("pow", "exp", "cbrt"):
        kern = _named(name, _cubed)
        assert f"void {name}_knl(" in emit_code(kern)
        agrees(kern, reals)
    rng = np.random.default_rng(108)
    z = rng.uniform(-2.0, 2.0, 8) + 1j * rng.uniform(-2.0, 2.0, 8)
    kern = _named("cpow", _complex_cubed)
    assert "void cpow_knl(" in emit_code(kern)
    agrees(kern, lambda: {"z": z.copy(), "w": np.zeros(8, complex)})

    # Every function loopy resolves on a target, what the headers declare or
    # define, and the helpers of loopy and loopty, which a suffix ends.
    from loopy.target.c import CTarget

    from loopty.lower import _kernel_name, is_library_name

    known = sorted(CTarget().get_device_ast_builder().known_callables)
    assert "floor" in known and "make_tuple" not in known
    for name in [*known, "make_tuple", "fmod", "copysignf", "int64_t", "I", "NAN"]:
        assert is_library_name(name), name
        assert _kernel_name(name, []) == f"{name}_knl"
    assert _kernel_name("loopty_floor_div_int64", []) == "loopty_floor_div_int64_knl"
    assert _kernel_name("flooring", []) == "flooring"


# }}}


# {{{ an integer to a negative integer power (#109)


@kernel
def inverted(k: Arr[Fin[n], Int], a: Arr[Fin[n], Real]):  # noqa: F821
    """An integer element to a negative power, which numpy refuses."""
    for i in k.dom:
        a[i] = k[i] ** -1


@kernel
def inverted_scalar(s: Int, a: Arr[Fin[n], Real]):  # noqa: F821
    """An integral scalar to a negative power."""
    for i in a.dom:
        a[i] = (s + 1) ** -2


@kernel
def inverted_index(a: Arr[Fin[n], Real], b: Arr[Fin[n], Real]):  # noqa: F821
    """A loop variable to a negative power: a Python int, a real in both runs."""
    for i in a.dom:
        a[i] = (i + 1) ** -1
        b[i] = 1 / (i + 1) ** 2


def test_an_integer_to_a_negative_power_is_refused_by_the_trace():
    # numpy refuses it at every point; the compiled run stored 1, -1 or 0.
    with pytest.raises(ValueError, match="negative integer powers"):
        inverted(k=np.array([1, 2]), a=np.zeros(2))
    with pytest.raises(TraceError, match="Write '1 / k\\[i\\]', a real"):
        inverted.trace()
    with pytest.raises(TraceError, match=r"Write '1 / \(s \+ 1\) \*\* 2'"):
        inverted_scalar.trace()
    (fact,) = inverted.facts()
    assert fact.kind == "trace" and fact.status.value == "refuted"
    agrees(inverted_index, lambda: {"a": np.zeros(5), "b": np.zeros(5)})


# }}}


# {{{ a guard on the loops, which loopy reads into isl


@kernel
def on_the_loops(x: Arr[Fin[n], Real], y: Arr[Fin[n], Real]):  # noqa: F821
    """Guards on loop variables alone that numpy computes in other types."""
    for i in x.dom:
        with when((i ** 0.5 > 1.5) & (i * i < 40) & (i / 2 > 1)):
            y[i] = 2.0 * x[i]


def test_a_guard_on_the_loops_is_not_cast():
    # loopy reads a guard that names no array as an isl set, and its reader
    # failed on a cast: i ** 0.5 converted both operands to doubles.
    def make() -> dict:
        return {"x": np.arange(8.0), "y": np.zeros(8)}

    native = make()
    on_the_loops(**native)
    assert list(native["y"]) == [0.0, 0.0, 0.0, 6.0, 8.0, 10.0, 12.0, 0.0]
    agrees(on_the_loops, make)


# }}}
