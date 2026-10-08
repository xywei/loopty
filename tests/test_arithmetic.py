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
named like a library function (#108), an integer to a negative power
(#109), a written ``int32`` array of ``Fin[m]`` (#121), a narrow integer type
and an unsigned one beside a signed one (#122), ``abs`` of an integer (#123),
a name the generated code gives a meaning (#124), a negated truth value
(#130), a kernel named like an OpenCL C built-in (#131), an integer literal
past 64 bits (#140), and one beside an unsigned integer that does not hold it
(#141). Each kernel here either agrees with its native run bit for bit, or is
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
def past_the_index(
    y: Arr[Fin[n], Int],  # noqa: F821
    z: Arr[Fin[n], Int],  # noqa: F821
    w: Arr[Fin[n], Int],  # noqa: F821
):
    """A loop variable beside a literal near 2**31, a small one, and several."""
    for i in y.dom:
        y[i] = i + 2147483647
        z[i] = i + 1
        w[i] = i + 2**29 + 2**29 + 2**29 + 2**29


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

    # A sum of loop variables and literals is index arithmetic, left in 32
    # bits, but where its literals total 2**30 or more: i + 2**31 - 1 wrapped
    # round at i = 1 compiled, and so did i + 2**29 + ... + 2**29, which
    # lanky builds as sums of one literal each.
    code = emit_code(past_the_index)
    assert "y[i] = (int64_t) (i) + 2147483647" in code
    assert "z[i] = (int64_t) (i + 1)" in code
    assert "w[i] = (int64_t) (i + 536870912) + 536870912 + 536870912" in code
    native = {name: np.zeros(3, np.int64) for name in "yzw"}
    past_the_index(**native)
    assert native["w"][0] == 2**31
    agrees(past_the_index, lambda: {name: np.zeros(3, np.int64) for name in "yzw"})


@kernel
def past_the_top(
    x: Arr[Fin[n], Int],  # noqa: F821
    y: Arr[Fin[n], Real],  # noqa: F821
    z: Arr[Fin[n], Int],  # noqa: F821
):
    """Arithmetic that leaves 64 bits, which numpy wraps round."""
    for i in x.dom:
        y[i] = (x[i] + 1 > x[i]) * 1.0
        z[i] = x[i] * 3 // 7


@pytest.mark.filterwarnings("ignore::RuntimeWarning")
def test_a_signed_overflow_wraps_round_as_numpy_wraps_it():
    # numpy wraps 2**63 - 1 + 1 round to the smallest int64, so x + 1 > x is
    # false there. C leaves the overflow undefined, and GCC at -O3 folded the
    # comparison to true; the C target builds every kernel with -fwrapv.
    from loopty.lower import WRAP_FLAG, lower_generic

    def make() -> dict:
        return {
            "x": np.array([2**63 - 1, 5, -(2**63)]),
            "y": np.zeros(3),
            "z": np.zeros(3, np.int64),
        }

    native = make()
    past_the_top(**native)
    assert list(native["y"]) == [0.0, 1.0, 1.0]
    entry = lower_generic(past_the_top.trace(), "c").kernel.default_entrypoint
    assert WRAP_FLAG in entry.options.build_options
    agrees(past_the_top, make)


@kernel
def guarded_squares(
    x: Arr[Fin[n], Real],  # noqa: F821
    k: Fin[m],  # noqa: F821
    y: Arr[Fin[n], Real],  # noqa: F821
    z: Arr[Fin[n], Real],  # noqa: F821
    w: Arr[Fin[n], Real],  # noqa: F821
):
    """Guards on the loops and on a scalar whose products leave 32 bits."""
    for i in x.dom:
        with when(i * i < 2**31 + 5):
            y[i] = x[i]
        with when(k * k > i + 2**31):
            z[i] = x[i]
        with when(100_000 * i < 2**31):
            w[i] = x[i]


def test_a_guard_on_the_loops_computes_a_product_in_64_bits():
    # loopy reads such a guard into isl, so no cast goes there, and its
    # integers were left in 32 bits: i * i wrapped round at i = 46341, to a
    # negative number below the bound, and so did k * k of a Fin[m] scalar
    # and 100_000 * i at i = 21475. A product with a 64-bit 1, or a literal
    # written in 64 bits, is what isl reads as it reads the number, or
    # declines as it declines i * i.
    rows = 46_345

    def make() -> dict:
        return {
            "x": np.arange(rows, dtype=np.float64),
            "k": 46_341,
            **{name: np.zeros(rows) for name in "yzw"},
        }

    native = make()
    guarded_squares(**native)
    assert native["y"][46_340] == 46_340.0 and native["y"][46_341] == 0.0
    # k * k is 2**31 + 4633.
    assert native["z"][4632] == 4632.0 and native["z"][4633] == 0.0
    assert native["w"][21_474] == 21_474.0 and native["w"][21_475] == 0.0
    code = emit_code(guarded_squares)
    assert "if (1l * i * i < 2147483653)" in code
    assert "if (1l * k * k > " in code
    assert "if (100000l * i < 2147483648" in code
    agrees(guarded_squares, make)


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


@kernel
def complement_scalar(
    x: Arr[Fin[n], Real],  # noqa: F821
    a: Real,
    y: Arr[Fin[n], Real],  # noqa: F821
):
    """``~`` of a comparison of a real scalar, used as a number."""
    for i in x.dom:
        y[i] = x[i] * ~(a > 0.5)


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

    # So ~ of a comparison of a scalar is a numpy bool's, logical, however a
    # was passed, and the trace no longer refuses it as a Python bool's.
    def masked(a) -> dict:
        return {"x": np.array([0.25, 0.75]), "a": a, "y": np.zeros(2)}

    for a in (0.7, 0.2, np.float64(0.7)):
        agrees(complement_scalar, lambda a=a: masked(a))

    # A Python int past uint64 has no numpy dtype, and was passed on weak, so
    # x * a was single precision natively for a Real a = 2**64 + 2**40.
    for a in (2**64 + 2**40, 2**70 + 1):
        results = make(a, 3)
        scale_by(**results)
        assert results["y"][0] == np.float32(np.float64(x[0]) * a * 0.1)
        agrees(scale_by, lambda a=a: make(a, 3))


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


def test_the_opencl_definitions_call_one_overload():
    # OpenCL C overloads fmod, floor and copysign by type, and copysign(0, b)
    # of an int zero beside a float b is ambiguous between the float and the
    # double one. The device path is not run here, so its text is checked.
    from loopy.target.c import CTarget
    from loopy.target.opencl import OpenCLTarget

    from loopty.operations import definition

    code = definition("loopty_mod", np.dtype(np.float32), OpenCLTarget())
    assert "inline float loopty_mod_float32(float a, float b)" in code
    assert "fmod(a, b)" in code and "copysign((float) 0, b)" in code
    assert "#include" not in code
    code = definition("loopty_floor_div", np.dtype(np.float32), CTarget())
    assert "static inline float loopty_floor_div_float32" in code
    assert "floorf(div)" in code and "copysignf((float) 0, a / b)" in code


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


@kernel
def by_zero(
    k: Arr[Fin[n], Int],  # noqa: F821
    a: Arr[Fin[n], Int],  # noqa: F821
    y: Arr[Fin[n], Real],  # noqa: F821
):
    """A numpy integer by a literal zero, which numpy divides."""
    for i in k.dom:
        a[i] = k[i] // 0 + k[i] % 0
        y[i] = k[i] / 0


@kernel
def index_by_zero(y: Arr[Fin[n], Int]):  # noqa: F821
    """A loop variable by a literal zero, which Python refuses."""
    for i in y.dom:
        y[i] = i // 0


@kernel
def real_index_by_zero(y: Arr[Fin[n], Real]):  # noqa: F821
    """A loop variable modulo a real zero, which Python refuses."""
    for i in y.dom:
        y[i] = i % 0.0


@kernel
def complex_index_by_zero(y: Arr[Fin[n], np.complex128]):  # noqa: F821
    """A loop variable divided by a complex zero, which Python refuses."""
    for i in y.dom:
        y[i] = i / 0j


@kernel
def guard_by_zero(x: Arr[Fin[n], Real], y: Arr[Fin[n], Real]):  # noqa: F821
    """A loop variable divided by zero in a guard."""
    for i in x.dom:
        with when(i / 0 > 1):
            y[i] = x[i]


@pytest.mark.filterwarnings("ignore::RuntimeWarning")
def test_a_python_number_by_a_literal_zero_is_refused_by_the_trace():
    # Python refuses i // 0 of a loop variable, an int, with
    # ZeroDivisionError, where the compiled run computed numpy's 0 for it.
    # A numpy integer by zero is numpy's 0, inf or nan in both runs.
    agrees(
        by_zero,
        lambda: {
            "k": np.array([5, -5, 0]),
            "a": np.zeros(3, np.int64),
            "y": np.zeros(3),
        },
    )
    with pytest.raises(ZeroDivisionError):
        index_by_zero(y=np.zeros(3, np.int64))
    for kern, symbol in (
        (index_by_zero, "//"),
        (real_index_by_zero, "%"),
        (complex_index_by_zero, "/"),
        (guard_by_zero, "/"),
    ):
        with pytest.raises(TraceError, match="divides by zero") as refused:
            kern.trace()
        assert f"Python refuses '{symbol}' by zero" in str(refused.value)
        assert "Divide by a value that is not zero" in str(refused.value)


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


@kernel
def differenced(
    b: Arr[Fin[n], Bool],  # noqa: F821
    c: Arr[Fin[n], Bool],  # noqa: F821
    y: Arr[Fin[n], Real],  # noqa: F821
):
    """A difference of two truth values, which numpy refuses."""
    for i in b.dom:
        y[i] = (b[i] - c[i]) * 1.0


@kernel
def xor_or_difference(
    b: Arr[Fin[n], Bool],  # noqa: F821
    c: Arr[Fin[n], Bool],  # noqa: F821
    y: Arr[Fin[n], Real],  # noqa: F821
    z: Arr[Fin[n], Real],  # noqa: F821
):
    """The two fixes: ``^`` for ``xor``, and ``1 * b - c`` for a difference."""
    for i in b.dom:
        y[i] = (b[i] ^ c[i]) * 1.0
        z[i] = (1 * b[i] - c[i]) * 1.0


@kernel
def less_a_truth(
    x: Arr[Fin[n], Real],  # noqa: F821
    b: Arr[Fin[n], Bool],  # noqa: F821
    y: Arr[Fin[n], Real],  # noqa: F821
):
    """A truth value subtracted from a real, which numpy computes."""
    for i in x.dom:
        y[i] = x[i] - b[i] + (i > 1) - (i > 2)


def test_a_difference_of_truth_values_is_refused_by_the_trace():
    # numpy refuses b - c of two bools at every point, and the compiled
    # kernel subtracted the bytes.
    def make() -> dict:
        return {
            "b": np.array([True, True, False, False]),
            "c": np.array([True, False, True, False]),
            "y": np.zeros(4),
        }

    with pytest.raises(TypeError, match="numpy boolean subtract"):
        differenced(**make())
    with pytest.raises(TraceError, match="subtracts truth values") as refused:
        differenced.trace()
    assert "'b[i] ^ c[i]' for 'xor', or '1 * b[i] - c[i]'" in str(refused.value)

    def both() -> dict:
        return {**make(), "z": np.zeros(4)}

    native = both()
    xor_or_difference(**native)
    assert list(native["y"]) == [0.0, 1.0, 1.0, 0.0]
    assert list(native["z"]) == [0.0, 1.0, -1.0, 0.0]
    agrees(xor_or_difference, both)
    # A real less a truth value, and Python bools of the loops, are numbers.
    agrees(
        less_a_truth,
        lambda: {"x": np.arange(4.0), "b": make()["b"], "y": np.zeros(4)},
    )


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


@kernel
def shifted_back(k: Arr[Fin[n], Int], y: Arr[Fin[n], Int]):  # noqa: F821
    """A shift by a negative literal, of a numpy integer and of a loop variable."""
    for i in k.dom:
        y[i] = (k[i] << -1) + (i << -1)


@kernel
def shifted_numpy(k: Arr[Fin[n], Int], y: Arr[Fin[n], Int]):  # noqa: F821
    """A numpy integer shifted by a negative literal: ``0``, or ``-1``."""
    for i in k.dom:
        y[i] = (k[i] << -1) + (k[i] >> -1)


@kernel
def complex_remainder(
    z: Arr[Fin[n], np.complex128],  # noqa: F821
    w: Arr[Fin[n], np.complex128],  # noqa: F821
):
    """``%`` of a complex value, which numpy refuses."""
    for i in z.dom:
        w[i] = z[i] % 2.0


def test_a_shift_python_refuses_and_a_complex_remainder_are_refused():
    # Python refuses i << -1 of a loop variable, an int, where numpy shifts a
    # numpy integer to 0 or -1, which the compiled run computes too.
    with pytest.raises(TraceError, match="shifts by a negative amount") as refused:
        shifted_back.trace()
    assert "Write 'i >> 1'" in str(refused.value)
    agrees(
        shifted_numpy,
        lambda: {"k": np.array([5, -5, 0]), "y": np.zeros(3, np.int64)},
    )
    # numpy refuses % and // of a complex number; it failed inside loopy's
    # code generator.
    with pytest.raises(TraceError, match="which is complex"):
        complex_remainder.trace()


@kernel
def shifted_past(y: Arr[Fin[n], Real], z: Arr[Fin[n], Real]):  # noqa: F821
    """A loop variable shifted left past 64 bits, which Python shifts exactly."""
    for i in y.dom:
        y[i] = 1.0 * (i << 64)
        z[i] = 1.0 * (i >> 64)


@kernel
def scaled_past(y: Arr[Fin[n], Real]):  # noqa: F821
    """The named fix: the real a shift past 64 bits stands for."""
    for i in y.dom:
        y[i] = i * 2.0**64


def test_a_python_int_shifted_past_64_bits_is_refused():
    # Python shifts a loop variable exactly, to 2**64 at i = 1, and the
    # compiled kernel computes it in 64 bits, to 0 as numpy shifts every bit
    # out. A right shift is 0 or -1 in both.
    native = {"y": np.zeros(3), "z": np.zeros(3)}
    shifted_past(**native)
    assert list(native["y"]) == [0.0, 2.0**64, 2.0**65]
    with pytest.raises(TraceError, match="shifts past 64 bits") as refused:
        shifted_past.trace()
    assert "Write 'i * 2.0 ** 64' for the real it stands for" in str(refused.value)
    agrees(scaled_past, lambda: {"y": np.zeros(3)})


@kernel
def compared_bits(
    k: Arr[Fin[n], Int],  # noqa: F821
    x: Arr[Fin[n], Real],  # noqa: F821
    y: Arr[Fin[n], Real],  # noqa: F821
    z: Arr[Fin[n], Real],  # noqa: F821
    w: Arr[Fin[n], Real],  # noqa: F821
):
    """A comparison of an ``^`` and of a comparison, which C binds otherwise."""
    for i in k.dom:
        y[i] = (k[i] < (k[i] ^ 1)) * 1.0
        z[i] = (x[i] < (k[i] == 1)) * 1.0
        with when(i < (i ^ 1)):
            w[i] = 1.0


def test_a_comparison_of_an_xor_is_printed_with_cs_precedence():
    # loopy printed by Python's precedence, where ^ binds more tightly than a
    # comparison: k[i] < (k[i] ^ 1) was k[i] < k[i] ^ 1, which C reads as
    # (k[i] < k[i]) ^ 1, true at every k, and x[i] < (k[i] == 1) was
    # x[i] < k[i] == 1, (x[i] < k[i]) == 1.
    def make() -> dict:
        return {
            "k": np.array([0, 1, 2, 3, 2, 5]),
            "x": np.array([0.5, 0.5, 1.5, 0.5, -1.0, 0.5]),
            **{name: np.zeros(6) for name in "yzw"},
        }

    native = make()
    compared_bits(**native)
    assert list(native["y"]) == [1.0, 0.0, 1.0, 0.0, 1.0, 0.0]
    assert list(native["z"]) == [0.0, 1.0, 0.0, 0.0, 1.0, 0.0]
    assert list(native["w"]) == [1.0, 0.0, 1.0, 0.0, 1.0, 0.0]
    code = emit_code(compared_bits)
    assert "(k[i] < (k[i] ^ 1)) * 1.0" in code
    assert "(x[i] < (k[i] == 1)) * 1.0" in code
    assert "if (i < (i ^ 1))" in code
    agrees(compared_bits, make)


@kernel
def bits_compared(
    k: Arr[Fin[n], Int],  # noqa: F821
    y: Arr[Fin[n], Real],  # noqa: F821
    z: Arr[Fin[n], Real],  # noqa: F821
    w: Arr[Fin[n], Real],  # noqa: F821
    v: Arr[Fin[n], Real],  # noqa: F821
):
    """``^``, ``<<`` and ``>>`` compared, in a guard, as a number and under ``~``."""
    for i in k.dom:
        with when((k[i] ^ 1) == 0):
            y[i] = 1.0
        z[i] = ((k[i] << 1) != 4) * 1.0 + ((k[i] >> 1) > 1) * 2.0
        with when((i << 2) > 5):
            w[i] = 1.0
        v[i] = 1.0 * ~((k[i] ^ 1) == 0) + 1.0 * ~((k[i] >> 1) > 1)


def test_a_comparison_of_an_xor_or_a_shift_is_traced():
    # lanky left ^, << and >> as pymbolic's nodes, whose == compared them
    # structurally: when((k[i] ^ 1) == 0) was traced as when(False), and
    # (k[i] << 1) != 4 as True, which the trace-faithful fact found tested on
    # its draws; and whose > raised a TypeError. ~ of such a comparison of an
    # element is a numpy bool's, logical, and is not refused as a Python
    # bool's.
    def make() -> dict:
        return {"k": np.array([0, 1, 2, 3, 4, 5]), **{x: np.zeros(6) for x in "yzwv"}}

    native = make()
    bits_compared(**native)
    assert list(native["y"]) == [0.0, 1.0, 0.0, 0.0, 0.0, 0.0]
    assert list(native["z"]) == [1.0, 1.0, 0.0, 1.0, 3.0, 3.0]
    assert list(native["w"]) == [0.0, 0.0, 1.0, 1.0, 1.0, 1.0]
    assert list(native["v"]) == [2.0, 1.0, 2.0, 2.0, 1.0, 1.0]
    (statement, *_) = bits_compared.term.stmts
    assert isinstance(statement.guard, prim.Comparison)
    agrees(bits_compared, make)


# }}}


# {{{ an unsigned integer beside a Python int


@kernel
def unsigned_arithmetic(
    u: Arr[Fin[n], np.uint64],  # noqa: F821
    v: Arr[Fin[n], np.uint64],  # noqa: F821
    y: Arr[Fin[n], np.uint64],  # noqa: F821
    z: Arr[Fin[n], np.uint64],  # noqa: F821
    c: Arr[Fin[n], Bool],  # noqa: F821
):
    """``uint64`` arithmetic with literals and a loop variable beside it."""
    for i in u.dom:
        y[i] = u[i] // v[i] + u[i] % 3 + u[i] * i
        z[i] = (u[i] >> v[i]) + (u[i] << 3) + (u[i] ^ 5)
        c[i] = u[i] > 18446744073709551614


@pytest.mark.filterwarnings("ignore::RuntimeWarning")
def test_an_unsigned_integer_beside_a_python_int_stays_unsigned():
    # numpy keeps a Python int or a loop variable beside a uint64 weak, and
    # computes in uint64; loopy types a uint64 beside a signed integer as
    # numpy types two arrays, in double, so u % 3 lost the low bits and
    # u << 3 failed to lower. The literal is written as a uint64.
    def make() -> dict:
        return {
            "u": np.array([2**64 - 1, 5, 2**63], np.uint64),
            "v": np.array([0, 70, 63], np.uint64),
            "y": np.zeros(3, np.uint64),
            "z": np.zeros(3, np.uint64),
            "c": np.zeros(3, bool),
        }

    native = make()
    unsigned_arithmetic(**native)
    assert native["y"][0] == (2**64 - 1) % 3 == 0
    assert list(native["c"]) == [True, False, False]
    code = emit_code(unsigned_arithmetic)
    assert "loopty_mod_uint64(u[i], 3ul)" in code
    assert "loopty_lshift_uint64(u[i], 3ul)" in code
    agrees(unsigned_arithmetic, make)


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


@kernel
def shadowed(loopty_mod_int64: Arr[Fin[n], Int], y: Arr[Fin[n], Int]):  # noqa: F821
    """A parameter named like the helper its remainder calls."""
    for i in y.dom:
        y[i] = loopty_mod_int64[i] % 3


def test_a_name_of_a_helper_is_refused_in_a_kernel():
    # The parameter shadowed the helper in the kernel's body, and the call of
    # it failed to compile. A parameter is not renamed, since a caller passes
    # it by name, so it is refused, as a C keyword is.
    from loopty.lower import LoweringError

    with pytest.raises(LoweringError, match="parameters loopty_mod_int64") as refused:
        emit_code(shadowed)
    assert "helper function loopy or loopty defines" in str(refused.value)


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
        with when((i**0.5 > 1.5) & (i * i < 40) & (i / 2 > 1) & ((i + 1) ** -1 < 0.3)):
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


@kernel
def on_the_scalars(
    x: Arr[Fin[n], Real],  # noqa: F821
    a: np.float32,
    s: Int,
    y: Arr[Fin[n], Real],  # noqa: F821
):
    """A guard on scalars, which loopy reads into isl as it does one on the loops."""
    for i in x.dom:
        with when(s * a > 0.300000008):
            y[i] = x[i]


def test_a_guard_on_scalars_is_computed_as_numpy_computes_it():
    # numpy computes s * a of an int64 s and a float32 a in double, and C in
    # single precision. loopy reads a guard on scalars into isl, whose reader
    # raises on a cast, so the double is had by a product with 1.0 instead.
    # Without either, 3 * 0.1f was 0.3000000119 compiled and 0.3000000045
    # natively, either side of the bound.
    def make() -> dict:
        return {"x": np.ones(3), "a": np.float32(0.1), "s": 3, "y": np.zeros(3)}

    native = make()
    on_the_scalars(**native)
    assert list(native["y"]) == [0.0, 0.0, 0.0]
    assert "s * (1.0 * a) > 0.300000008" in emit_code(on_the_scalars)
    agrees(on_the_scalars, make)


@kernel
def single_on_the_loops(
    x: Arr[Fin[n], Real],  # noqa: F821
    a: np.float32,
    y: Arr[Fin[n], Real],  # noqa: F821
):
    """A guard on the loops that numpy computes in single precision."""
    for i in x.dom:
        with when(a + (i + 1) ** -1 > 0.5):
            y[i] = x[i]


def test_a_guard_on_the_loops_in_single_precision_is_refused():
    # numpy rounds the double (i + 1) ** -1 to single precision beside the
    # float32 a, which C does only by a cast, and loopy's isl reader raises
    # on a cast there; without it the guard was computed in double compiled.
    from loopty.lower import LoweringError

    native = {"x": np.ones(3), "a": np.float32(0.1), "y": np.zeros(3)}
    single_on_the_loops(**native)
    assert list(native["y"]) == [1.0, 1.0, 0.0]
    with pytest.raises(LoweringError, match="reads no array") as refused:
        emit_code(single_on_the_loops)
    assert "Declare the float32 scalars the guard names Real" in str(refused.value)


# }}}


# {{{ a written int32 array of Fin[m] (#121)


@kernel
def reversed_squares(
    p: Arr[Fin[n], Fin[n]],  # noqa: F821
    q: Arr[Fin[n], Int],  # noqa: F821
):
    """A permutation written into ``p``, and the squares of its entries."""
    for i in p.dom:
        p[i] = p.dom.size - 1 - i
    for i in p.dom:
        q[i] = p[i] * p[i]


def test_a_written_int32_index_array_is_computed_with_in_64_bits():
    # The native run read p[i] of a written int32 array as an np.int32, and
    # p[i] * p[i] wrapped round at 46341, where the compiled run computes in
    # 64 bits. Its elements are read as int64 now; the array is written as
    # the caller gave it.
    from loopty.interpret import interpret

    rows = 46_342

    def make() -> dict:
        return {"p": np.zeros(rows, np.int32), "q": np.zeros(rows, np.int64)}

    native = make()
    reversed_squares(**native)
    assert native["p"].dtype == np.int32 and native["p"][0] == rows - 1
    assert native["q"][0] == (rows - 1) ** 2 == 2_147_488_281
    agrees(reversed_squares, make)
    interpreted = make()
    interpret(reversed_squares.term, interpreted)
    assert interpreted["q"][0] == (rows - 1) ** 2


# }}}


# {{{ a narrow integer, and an unsigned one beside a signed one (#122)


@kernel
def narrow(
    a: Arr[Fin[n], np.int8],  # noqa: F821
    u: Arr[Fin[n], np.uint16],  # noqa: F821
    b: Arr[Fin[n], np.int8],  # noqa: F821
    c: Arr[Fin[n], Int],  # noqa: F821
    d: Arr[Fin[n], Real],  # noqa: F821
    e: Arr[Fin[n], Int],  # noqa: F821
):
    """Arithmetic numpy computes in ``int8`` and ``uint16``, and C in ``int``."""
    for i in a.dom:
        b[i] = a[i] * a[i] // 2
        c[i] = (a[i] + a[i]) * 1 + (a[i] << 3) + (a[i] // -1) + a[i] ** 3
        d[i] = (u[i] * u[i] > 3) * 1.0 + (u[i] + u[i]) * 0.5
        e[i] = (u[i] << i) + (a[i] * i)


@kernel
def mixed_signs(
    u: Arr[Fin[n], np.uint32],  # noqa: F821
    k: Arr[Fin[n], np.int32],  # noqa: F821
    col: Arr[Fin[n], Fin[n]],  # noqa: F821
    y: Arr[Fin[n], Int],  # noqa: F821
    z: Arr[Fin[n], Int],  # noqa: F821
    b: Arr[Fin[n], Bool],  # noqa: F821
    c: Arr[Fin[n], Bool],  # noqa: F821
    d: Arr[Fin[n], Bool],  # noqa: F821
):
    """A ``uint32`` beside signed integers, which numpy computes in ``int64``."""
    for i in u.dom:
        y[i] = (u[i] + k[i]) + (u[i] ^ k[i]) + (u[i] - col[i])
        z[i] = (u[i] - 1) + (u[i] << 3) + (u[i] + i) // 3
        b[i] = u[i] < k[i]
        c[i] = u[i] == -1
        d[i] = u[i] > col[i] - 5


@kernel
def wide_signs(
    u: Arr[Fin[n], np.uint64],  # noqa: F821
    k: Arr[Fin[n], Int],  # noqa: F821
    b: Arr[Fin[n], Bool],  # noqa: F821
    c: Arr[Fin[n], Bool],  # noqa: F821
    y: Arr[Fin[n], Real],  # noqa: F821
):
    """A ``uint64`` compared with an ``int64``, which numpy compares exactly."""
    for i in u.dom:
        b[i] = (u[i] > k[i]) | (u[i] <= -2)
        c[i] = (k[i] >= u[i]) & (u[i] != k[i])
        y[i] = u[i] + k[i]


@pytest.mark.filterwarnings("ignore::RuntimeWarning")
def test_a_narrow_integer_is_computed_in_its_own_type():
    # numpy computes int8 * int8 in int8, which wraps 100 * 100 to 16, and C
    # in int: b was [8, 4] natively and [-120, 4] compiled. The result of an
    # operation numpy computes in a narrow type is converted back into it.
    def make() -> dict:
        return {
            "a": np.array([100, 3, -128, 127, -1], np.int8),
            "u": np.array([65535, 2, 300, 0, 40000], np.uint16),
            "b": np.zeros(5, np.int8),
            "c": np.zeros(5, np.int64),
            "d": np.zeros(5),
            "e": np.zeros(5, np.int64),
        }

    native = make()
    narrow(**native)
    assert list(native["b"][:2]) == [8, 4]
    assert native["c"][2] == np.int8(-128) * 2 + 0 + (-128) + np.int8(-128) ** 3
    agrees(narrow, make)
    assert "(int8_t) (a[i] * a[i])" in emit_code(narrow)


@pytest.mark.filterwarnings("ignore::RuntimeWarning")
def test_an_unsigned_integer_beside_a_signed_one_is_numpys():
    # C computes a uint32 beside an int32 in uint32, where numpy computes in
    # int64: u[i] + k[i] at 0 + -1 was 4294967295 compiled, and u[i] < k[i]
    # at 0 < -1 true. A comparison with a negative value is decided by its
    # sign first, which numpy's exact comparison does too.
    def make() -> dict:
        return {
            "u": np.array([0, 5, 2**32 - 1, 7], np.uint32),
            "k": np.array([-1, -7, 3, 2**31 - 1], np.int32),
            "col": np.array([1, 0, 3, 2], np.int64),
            "y": np.zeros(4, np.int64),
            "z": np.zeros(4, np.int64),
            "b": np.zeros(4, bool),
            "c": np.zeros(4, bool),
            "d": np.zeros(4, bool),
        }

    native = make()
    mixed_signs(**native)
    assert native["y"][0] == -1 + (0 ^ -1) + -1
    # (u - 1) + (u << 3) + (u + i) // 3 at u = 2**32 - 1, i = 2, in uint32.
    assert native["z"][2] == (2**32 - 2 + 2**32 - 8 + 0) % 2**32
    assert list(native["b"]) == [False, False, False, True]
    assert list(native["c"]) == [False] * 4
    assert list(native["d"]) == [True] * 4
    agrees(mixed_signs, make)
    code = emit_code(mixed_signs)
    assert "(int64_t) (u[i]) + (int64_t) (k[i])" in code
    assert "k[i] >= 0 && u[i] < k[i]" in code

    # A uint64 and an int64 have no common integer type in C, and numpy
    # compares them exactly: 2**63 > 2**63 - 1, which doubles say is false.
    def wide() -> dict:
        return {
            "u": np.array([2**63, 0, 2**64 - 1, 5], np.uint64),
            "k": np.array([2**63 - 1, -1, -(2**63), 5], np.int64),
            "b": np.zeros(4, bool),
            "c": np.zeros(4, bool),
            "y": np.zeros(4),
        }

    native = wide()
    wide_signs(**native)
    assert list(native["b"]) == [True, True, True, False]
    assert list(native["c"]) == [False, False, False, False]
    agrees(wide_signs, wide)


@kernel
def unsigned_literal(
    u: Arr[Fin[n], np.uint32],  # noqa: F821
    y: Arr[Fin[n], Int],  # noqa: F821
):
    """A ``uint32`` literal, which numpy computes with in 32 bits."""
    for i in u.dom:
        y[i] = (u[i] * np.uint32(5)) * 1


@pytest.mark.filterwarnings("ignore::RuntimeWarning")
def test_a_uint32_literal_is_computed_with_in_32_bits():
    # loopy wrote an np.uint32 literal 5ul, an unsigned long, so u[i] * 5ul
    # was computed in 64 bits, where numpy wraps round at 2**32.
    import re

    def make() -> dict:
        return {
            "u": np.array([2**32 - 1, 2**31, 3], np.uint32),
            "y": np.zeros(3, np.int64),
        }

    native = make()
    unsigned_literal(**native)
    assert list(native["y"]) == [(5 * (2**32 - 1)) % 2**32, 2**31, 15]
    agrees(unsigned_literal, make)
    assert re.search(r"\b5u\b", emit_code(unsigned_literal))


@kernel
def narrow_unsigned_literals(
    a: Arr[Fin[n], np.int8],  # noqa: F821
    y: Arr[Fin[n], Int],  # noqa: F821
    b: Arr[Fin[n], Bool],  # noqa: F821
):
    """``np.uint16`` and ``np.uint8`` literals beside an ``int8``."""
    for i in a.dom:
        y[i] = a[i] + np.uint16(3)
        b[i] = a[i] < np.uint8(3)


@pytest.mark.filterwarnings("ignore::RuntimeWarning")
def test_a_narrow_unsigned_literal_is_an_int():
    # loopy wrote np.uint16(3) and np.uint8(3) as 3u, an unsigned int, which
    # took a[i] round into it: a[i] + 3u was 4294967294 at a[i] = -5, and
    # a[i] < 3u false. C's integer promotion makes an int of either type.
    import re

    def make() -> dict:
        return {
            "a": np.array([-5, 0, 3, -128, 127], np.int8),
            "y": np.zeros(5, np.int64),
            "b": np.zeros(5, bool),
        }

    native = make()
    narrow_unsigned_literals(**native)
    assert list(native["y"]) == [-2, 3, 6, -125, 130]
    assert list(native["b"]) == [True, True, False, True, False]
    agrees(narrow_unsigned_literals, make)
    assert not re.search(r"\b3u\b", emit_code(narrow_unsigned_literals))


@kernel
def literal_beside_integers(
    k: Arr[Fin[n], Int],  # noqa: F821
    y: Arr[Fin[n], Real],  # noqa: F821
    z: Arr[Fin[n], Real],  # noqa: F821
):
    """Integer arithmetic with literals, stored into reals."""
    for i in k.dom:
        y[i] = k[i] + 3
        z[i] = (k[i] ^ 3) + -1 * k[i]


@pytest.mark.filterwarnings("ignore::RuntimeWarning")
def test_an_integer_literal_in_integer_arithmetic_is_an_integer():
    # loopy wrote a literal in the type of the place its operation stands
    # in, a double for a store into a real: k[i] + 3 was k[i] + 3.0, which
    # never wraps round at 2**63 - 1 where numpy's int64 sum does, and
    # k[i] ^ 3 was k[i] ^ 3.0, which C refuses.
    def make() -> dict:
        return {
            "k": np.array([2**63 - 1, 2**53 + 1, -5]),
            "y": np.zeros(3),
            "z": np.zeros(3),
        }

    native = make()
    literal_beside_integers(**native)
    assert native["y"][0] == float(np.int64(-(2**63) + 2))
    agrees(literal_beside_integers, make)
    code = emit_code(literal_beside_integers)
    assert "(k[i] + 3)" in code and "(k[i] ^ 3)" in code


@kernel
def nested_right(
    k: Arr[Fin[n], Int],  # noqa: F821
    j: Arr[Fin[n], Int],  # noqa: F821
    x: Arr[Fin[n], Real],  # noqa: F821
    v: Arr[Fin[n], Real],  # noqa: F821
    y: Arr[Fin[n], Real],  # noqa: F821
    z: Arr[Fin[n], Real],  # noqa: F821
):
    """A product and a sum nested after the first operand of another."""
    for i in k.dom:
        y[i] = 1.0 * (k[i] * j[i])
        z[i] = x[i] + (v[i] + v[i])


@pytest.mark.filterwarnings("ignore::RuntimeWarning")
def test_a_sum_or_product_nested_to_the_right_is_computed_so():
    # pymbolic printed 1.0 * (k[i] * j[i]) flat, and C multiplied from the
    # left in double, 2**70, where numpy multiplies the integers first and
    # wraps round to 0; x[i] + (v[i] + v[i]) was added from the left, and
    # 1e16 + 1 + 1 rounds to 1e16 twice, where numpy adds 2.
    def make() -> dict:
        return {
            "k": np.array([2**40, 3]),
            "j": np.array([2**30, -5]),
            "x": np.array([1e16, 0.5]),
            "v": np.array([1.0, 0.25]),
            "y": np.zeros(2),
            "z": np.zeros(2),
        }

    native = make()
    nested_right(**native)
    assert list(native["y"]) == [0.0, -15.0]
    assert native["z"][0] == 1e16 + 2
    agrees(nested_right, make)
    code = emit_code(nested_right)
    assert "1.0 * (k[i] * j[i])" in code and "x[i] + (v[i] + v[i])" in code


@kernel
def differences(
    c: Arr[Fin[n], Int],  # noqa: F821
    a: Arr[Fin[n], np.int8],  # noqa: F821
    k: Arr[Fin[n], np.int32],  # noqa: F821
    u: Arr[Fin[n], np.uint32],  # noqa: F821
    y: Arr[Fin[n], Int],  # noqa: F821
    z: Arr[Fin[n], Int],  # noqa: F821
):
    """Differences pymbolic builds as a sum of a negation."""
    for i in c.dom:
        y[i] = c[i] - a[i]
        z[i] = k[i] - u[i]


@pytest.mark.filterwarnings("ignore::RuntimeWarning")
def test_a_difference_is_computed_in_the_type_of_the_two():
    # pymbolic builds c - a as c + -1 * a, which negates a in its own type
    # first: -1 * a of an int8 -128 is itself, and of a uint32 it is taken
    # round modulo 2**32, while numpy subtracts in the type of the two, int64
    # here. C computed k[i] - u[i] of an int32 k and a uint32 u in uint32.
    def make() -> dict:
        return {
            "c": np.array([5, -(2**63)]),
            "a": np.array([-128, 127], np.int8),
            "k": np.array([-1, 2**31 - 1], np.int32),
            "u": np.array([2**32 - 1, 0], np.uint32),
            "y": np.zeros(2, np.int64),
            "z": np.zeros(2, np.int64),
        }

    native = make()
    differences(**native)
    assert native["y"][0] == 133
    assert native["z"][0] == -1 - (2**32 - 1)
    agrees(differences, make)


# }}}


# {{{ abs of an integer (#123)


@kernel
def absolute(
    k: Arr[Fin[n], Int],  # noqa: F821
    a: Arr[Fin[n], np.int8],  # noqa: F821
    col: Arr[Fin[n], Fin[n]],  # noqa: F821
    u: Arr[Fin[n], np.uint32],  # noqa: F821
    b: Arr[Fin[n], Bool],  # noqa: F821
    y: Arr[Fin[n], Int],  # noqa: F821
    z: Arr[Fin[n], Int],  # noqa: F821
    x: Arr[Fin[n], Real],  # noqa: F821
):
    """``abs`` of integers of every kind, of a loop variable, and of a sum."""
    for i in k.dom:
        y[i] = abs(k[i]) + abs(a[i]) + abs(col[i] - i)
        z[i] = abs(u[i]) + abs(b[i]) + abs(i - 3) + abs(reduce_sum(k[j] for j in k.dom))
        x[i] = abs(-0.5 * k[i])


@pytest.mark.filterwarnings("ignore::RuntimeWarning")
def test_abs_of_an_integer_is_numpys():
    # loopy resolves abs as C's fabs, which it refuses for an integer (abs
    # does not support type float32). numpy's abs of the smallest int64 or
    # int8 is that value, which C's -k wraps round to under -fwrapv.
    def make() -> dict:
        return {
            "k": np.array([-3, 4, -(2**63), 0]),
            "a": np.array([-128, 5, -7, 0], np.int8),
            "col": np.array([3, 0, 1, 2]),
            "u": np.array([2**32 - 1, 0, 3, 1], np.uint32),
            "b": np.array([True, False, True, False]),
            "y": np.zeros(4, np.int64),
            "z": np.zeros(4, np.int64),
            "x": np.zeros(4),
        }

    native = make()
    absolute(**native)
    assert native["y"][2] == -(2**63) + 7 + 1
    assert native["y"][0] == 3 - 128 + 3
    assert "k[i] ^ loopty_rshift_int64(k[i], (int64_t) (63))" in emit_code(absolute)
    agrees(absolute, make)


@kernel
def absolute_compared(
    k: Arr[Fin[n], Int],  # noqa: F821
    c: Arr[Fin[n], np.int32],  # noqa: F821
    b: Arr[Fin[n], Bool],  # noqa: F821
    d: Arr[Fin[n], Bool],  # noqa: F821
    y: Arr[Fin[n], Int],  # noqa: F821
):
    """``abs`` of the smallest value compared with one, and divided."""
    for i in k.dom:
        b[i] = (k[i] ** 0) >= abs(k[i])
        d[i] = (c[i] ** 0) >= abs(c[i])
        y[i] = abs(k[i]) % 3 + abs(c[i]) // 2


@pytest.mark.filterwarnings("ignore::RuntimeWarning")
def test_abs_of_the_smallest_value_is_compared_as_numpys():
    # abs was written k < 0 ? -1 * k : k, which GCC reads as its own abs and
    # takes to be non-negative, -fwrapv or not: it folded 1 >= abs(c[i]) to
    # false at the smallest int32 and int64, even at -O0, where numpy's abs
    # is that value and 1 is greater.
    def make() -> dict:
        return {
            "k": np.array([-(2**63), -3, 4, 0]),
            "c": np.array([-(2**31), 5, -7, 2**31 - 1], np.int32),
            "b": np.zeros(4, bool),
            "d": np.zeros(4, bool),
            "y": np.zeros(4, np.int64),
        }

    native = make()
    absolute_compared(**native)
    assert list(native["b"]) == [True, False, False, True]
    assert list(native["d"]) == [True, False, False, False]
    assert native["y"][0] == (-(2**63)) % 3 + (-(2**31)) // 2
    agrees(absolute_compared, make)


@kernel
def absolute_on_the_loops(
    m: Int,
    s: np.int8,
    x: Arr[Fin[n], Real],  # noqa: F821
    y: Arr[Fin[n], Int],  # noqa: F821
    z: Arr[Fin[n], np.int8],  # noqa: F821
):
    """``abs`` of widened arithmetic of loop variables and scalars."""
    for i in x.dom:
        y[i] = abs(i * i - m) + abs((i << 3) - m * i)
        with when(x[i] < abs(i * i - m)):
            y[i] = 0
        z[i] = abs(s * s - i)


@pytest.mark.filterwarnings("ignore::RuntimeWarning")
def test_abs_of_arithmetic_on_the_loops_lowers():
    # loopy reads the condition of an If into isl in its bounds check, and
    # isl's reader raised on the cast that widens i * i: abs(i * i - m),
    # written with a branch, failed with UnsupportedExpressionError. abs is
    # written without one, of the int8 s * s converted back into int8 too.
    def make() -> dict:
        return {
            "m": 10,
            "s": np.int8(12),
            "x": np.array([0.5, 20.0, 3.0, -1.0, 9.0, 30.0]),
            "y": np.zeros(6, np.int64),
            "z": np.zeros(6, np.int8),
        }

    native = make()
    absolute_on_the_loops(**native)
    assert list(native["y"]) == [0, 11, 0, 0, 14, 25]
    assert list(native["z"]) == [112, 113, 114, 115, 116, 117]
    agrees(absolute_on_the_loops, make)
    code = emit_code(absolute_on_the_loops)
    assert "?" not in code[code.index("void absolute_on_the_loops(") :]


# }}}


# {{{ a name the generated code gives a meaning (#124) or OpenCL C has (#131)


def _cubed_complex(
    I: Arr[Fin[n], np.complex128],  # noqa: E741, F821, N803
    w: Arr[Fin[n], np.complex128],  # noqa: F821
):
    for i in I.dom:
        w[i] = I[i] * 2.0


def _powered(pow: Arr[Fin[n], Real], y: Arr[Fin[n], Real]):  # noqa: F821, A002
    for i in pow.dom:
        y[i] = pow[i] ** 3


def _floored(floor: Arr[Fin[n], Real], y: Arr[Fin[n], Real]):  # noqa: F821
    for i in floor.dom:
        y[i] = floor[i] * 2.0


def _copied(x: Arr[Fin[n], Real], y: Arr[Fin[n], Real]):  # noqa: F821
    for i in x.dom:
        y[i] = x[i]


def _with_parameter(name: str):
    """``_copied`` as a kernel whose first parameter is called ``name``."""
    code = _copied.__code__
    names = tuple(name if v == "x" else v for v in code.co_varnames)
    copy = types.FunctionType(code.replace(co_varnames=names), _copied.__globals__)
    copy.__annotations__ = {
        (name if key == "x" else key): value
        for key, value in _copied.__annotations__.items()
    }
    return kernel(copy)


def test_a_name_a_header_defines_or_a_called_function_is_refused():
    # complex.h defines I as a macro, which gcc expanded in the declaration
    # of the parameter I; a parameter pow hid the pow its power calls.
    from loopty.lower import LoweringError

    with pytest.raises(LoweringError, match=r"parameters I \(a macro or type complex"):
        emit_code(kernel(_cubed_complex))
    with pytest.raises(LoweringError, match="a C library function the kernel calls"):
        emit_code(kernel(_powered))
    # A function of the name the kernel never calls is no clash: floor here.
    agrees(kernel(_floored), lambda: {"floor": np.arange(3.0), "y": np.zeros(3)})
    for name, meaning in (
        ("NAN", "math.h"),
        ("INT32_MAX", "stdint.h"),
        ("M_PI", "OpenCL C"),
        ("get_local_id", "parallel loop"),
        # Built on an OpenCL device, each of these failed to compile:
        # OpenCL C defines NULL, SCHAR_MAX and a macro per extension.
        ("NULL", "OpenCL C"),
        ("SCHAR_MAX", "OpenCL C"),
        ("cl_khr_fp64", "OpenCL C"),
        ("pipe", "reserved words"),
        ("image2d_t", "reserved words"),
    ):
        with pytest.raises(LoweringError, match=meaning):
            emit_code(_with_parameter(name))
    agrees(_with_parameter("exp"), lambda: {"exp": np.arange(3.0), "y": np.zeros(3)})


def test_a_kernel_named_like_an_opencl_builtin_is_renamed(plain_opencl):
    # OpenCL C's built-ins are in neither loopy's C list nor C's headers: a
    # kernel named get_global_id was generated under that name.
    def reals() -> dict:
        return {"x": np.array([0.5, -2.0, 3.0]), "y": np.zeros(3)}

    for name in (
        "get_global_id",
        "clamp",
        "convert_int4_sat",
        "native_sin",
        "M_PI",
        # On an OpenCL device these failed to build: a macro, OpenCL C's
        # types, and a macro that takes arguments.
        "NULL",
        "size_t",
        "event_t",
        "ATOMIC_VAR_INIT",
        "main",
    ):
        kern = _named(name, _doubled)
        assert f"void {name}_knl(" in emit_code(kern)
        assert f" {name}_knl(" in emit_code(kern, target="opencl")
        agrees(kern, reals)
    # A family of built-ins takes in any suffix, and renaming one by a
    # suffix ran forever: it gets a prefix.
    for name in (
        "atomic_add",
        "work_group_reduce_add",
        "read_imagef",
        "memory_order_relaxed",
        "cl_khr_fp64",
    ):
        kern = _named(name, _doubled)
        assert f"void knl_{name}(" in emit_code(kern)
        assert f" knl_{name}(" in emit_code(kern, target="opencl")
        agrees(kern, reals)
    assert "void half_edged(" in emit_code(_named("half_edged", _doubled))


def _isl_size(x: Arr[Fin[max], Real], y: Arr[Fin[max], Real]):  # noqa: F821
    for i in x.dom:
        y[i] = x[i]


def _isl_scalar(x: Arr[Fin[n], Real], floor: Int, y: Arr[Fin[n], Real]):  # noqa: F821
    for i in x.dom:
        with when(i < floor):
            y[i] = x[i]


def _isl_loop(x: Arr[Fin[n], Real], y: Arr[Fin[n], Real]):  # noqa: F821
    for max in x.dom:  # noqa: A001
        y[max] = x[max] * 2.0


def _object_size(x: Arr[Fin[abs], Real], y: Arr[Fin[abs], Real]):  # noqa: F821
    for i in x.dom:
        y[i] = x[i]


def test_a_name_isl_reads_as_a_keyword():
    # isl's reader takes max, floor and its other keywords as its own
    # whatever their case: a size or a loop variable max failed inside the
    # trace with "isl_set_read_from_str failed: syntax error".
    with pytest.raises(TraceError, match="isl's keywords: max"):
        kernel(_isl_size).trace()
    with pytest.raises(TraceError, match="isl's keywords: floor"):
        kernel(_isl_scalar).trace()
    # A loop variable is renamed in the term.
    looped = kernel(_isl_loop)
    assert looped.term.stmts[0].assignee.indices[0].name == "max_0"
    agrees(looped, lambda: {"x": np.array([0.5, -2.0, 3.0]), "y": np.zeros(3)})
    # A size abs is lanky's abs, which lowering failed on as a foreign object.
    with pytest.raises(TraceError, match="neither a name nor a number"):
        kernel(_object_size).trace()


# }}}


# {{{ a negated truth value (#130)


@kernel
def negated(b: Arr[Fin[n], Bool], y: Arr[Fin[n], Int]):  # noqa: F821
    """``-b[i]``, which numpy refuses."""
    for i in b.dom:
        y[i] = -b[i]


@kernel
def negated_first(b: Arr[Fin[n], Bool], y: Arr[Fin[n], Int]):  # noqa: F821
    """``-b[i] + 1``, which numpy refuses too."""
    for i in b.dom:
        y[i] = -b[i] + 1


@kernel
def negations(
    b: Arr[Fin[n], Bool],  # noqa: F821
    y: Arr[Fin[n], Int],  # noqa: F821
    z: Arr[Fin[n], Int],  # noqa: F821
    w: Arr[Fin[n], Bool],  # noqa: F821
):
    """The named fixes, and a truth value subtracted from a number."""
    for i in b.dom:
        y[i] = -(1 * b[i])
        z[i] = 1 - b[i] + -(i > 1)
        w[i] = ~b[i]


def test_a_negated_truth_value_is_refused_by_the_trace():
    # numpy refuses -b[i] at every point, and the compiled kernel stored -1.
    def make() -> dict:
        return {"b": np.array([True, False, True]), "y": np.zeros(3, np.int64)}

    with pytest.raises(TypeError, match="boolean negative"):
        negated(**make())
    for kern in (negated, negated_first):
        with pytest.raises(TraceError, match="negates a truth value") as refused:
            kern.trace()
        assert "'~b[i]' for 'not', or '-(1 * b[i])'" in str(refused.value)

    def fixes() -> dict:
        return {
            **make(),
            "z": np.zeros(3, np.int64),
            "w": np.zeros(3, bool),
        }

    native = fixes()
    negations(**native)
    assert list(native["y"]) == [-1, 0, -1]
    assert list(native["z"]) == [0, 1, -1]
    agrees(negations, fixes)


# }}}


# {{{ an integer literal past 64 bits (#140), or beside a type that does not
# hold it (#141)


@kernel
def past_64_bits(
    x: Arr[Fin[n], Real],  # noqa: F821
    f: Arr[Fin[n], np.float32],  # noqa: F821
    u: Arr[Fin[n], np.uint64],  # noqa: F821
    k: Arr[Fin[n], Int],  # noqa: F821
    y: Arr[Fin[n], Real],  # noqa: F821
    z: Arr[Fin[n], np.uint64],  # noqa: F821
    b: Arr[Fin[n], Bool],  # noqa: F821
):
    """Literals past 64 bits where numpy computes in a type that holds them."""
    for i in x.dom:
        y[i] = x[i] * 2**70 + f[i] * -(2**80) + 1.0 * (i + 2.0**64)
        z[i] = u[i] + 2**63
        b[i] = (k[i] < 2**70) & (k[i] > -(2**70)) & (k[i] < 2**63)


@kernel
def exact_past_64(y: Arr[Fin[n], Real]):  # noqa: F821
    """A sum Python computes exactly past 64 bits."""
    for i in y.dom:
        y[i] = 1.0 * (i + 2**64)


@kernel
def stored_past_64(y: Arr[Fin[n], Real]):  # noqa: F821
    """A literal past 64 bits stored as it is."""
    for i in y.dom:
        y[i] = 2**70


@pytest.mark.filterwarnings("ignore::RuntimeWarning")
def test_an_integer_literal_past_64_bits():
    # Lowering failed inside loopy's type inference (integer constant too
    # large). It is written as numpy computes with it where that is a real
    # or a uint64, and refused elsewhere, naming the real.
    def make() -> dict:
        return {
            "x": np.array([1.0, -2.5]),
            "f": np.array([0.5, 2.0], np.float32),
            "u": np.array([5, 2**63 - 1], np.uint64),
            "k": np.array([-(2**63), 2**63 - 1]),
            "y": np.zeros(2),
            "z": np.zeros(2, np.uint64),
            "b": np.zeros(2, bool),
        }

    native = make()
    past_64_bits(**native)
    assert native["z"][1] == 2**64 - 1
    assert list(native["b"]) == [True, True]
    agrees(past_64_bits, make)
    with pytest.raises(TraceError, match=r"write it as a real, 2\.0 \*\* 64"):
        exact_past_64.trace()
    with pytest.raises(TraceError, match=r"2\.0 \*\* 70"):
        stored_past_64.trace()


@kernel
def compared_past_64(
    u: Arr[Fin[n], np.uint64],  # noqa: F821
    k: Arr[Fin[n], Int],  # noqa: F821
    b: Arr[Fin[n], Bool],  # noqa: F821
    c: Arr[Fin[n], Bool],  # noqa: F821
):
    """Literals just past the range of a ``uint64`` and of an ``int64``."""
    for i in u.dom:
        b[i] = (u[i] < 2**64) & (u[i] != 2**64 + 1)
        c[i] = (k[i] > -(2**63) - 1) & (k[i] >= -(2**63) - 7)


@pytest.mark.filterwarnings("ignore::RuntimeWarning")
def test_a_literal_just_past_64_bits_is_compared_exactly():
    # Such a literal was written as its own double, 2**64 or -2**63, onto
    # which the largest uint64 and the smallest int64 round: u[i] < 2**64 was
    # false at 2**64 - 1, where numpy compares exactly.
    def make() -> dict:
        return {
            "u": np.array([2**64 - 1, 0, 2**64 - 1024], np.uint64),
            "k": np.array([-(2**63), 0, 2**63 - 1]),
            "b": np.zeros(3, bool),
            "c": np.zeros(3, bool),
        }

    native = make()
    compared_past_64(**native)
    assert list(native["b"]) == list(native["c"]) == [True] * 3
    agrees(compared_past_64, make)


@kernel
def compared_past_64_in_reals(
    f: Arr[Fin[n], np.float32],  # noqa: F821
    z: Arr[Fin[n], np.complex64],  # noqa: F821
    k: Arr[Fin[n], Int],  # noqa: F821
    b: Arr[Fin[n], Bool],  # noqa: F821
    c: Arr[Fin[n], Bool],  # noqa: F821
    d: Arr[Fin[n], Bool],  # noqa: F821
):
    """A literal past 64 bits beside single precision, and past a double."""
    for i in f.dom:
        b[i] = (f[i] < 2**70 + 2**46) | (f[i] < -(2**130))
        c[i] = z[i] == 2**70 + 2**46
        d[i] = k[i] < 3**700


@kernel
def compared_past_a_double(x: Arr[Fin[n], Real], b: Arr[Fin[n], Bool]):  # noqa: F821
    """A literal no double holds, beside a real."""
    for i in x.dom:
        b[i] = x[i] < 3**700


@pytest.mark.filterwarnings("ignore::RuntimeWarning")
def test_a_literal_past_64_bits_is_compared_in_numpys_real():
    # numpy takes the literal into the real or complex type beside it, so
    # f[i] < 2**70 + 2**46 of a float32 compares with the float32 2**70. It
    # was written as a double, and differed at f[i] = 2**70. One no double
    # holds is refused beside a real, which numpy cannot convert it to, and
    # compared exactly with an integer, as numpy does.
    def make() -> dict:
        return {
            "f": np.array([2.0**70, 1.5, -np.inf], np.float32),
            "z": np.array([2.0**70, 1, 2.0**70 + 1j], np.complex64),
            "k": np.array([-(2**63), 0, 2**63 - 1]),
            "b": np.zeros(3, bool),
            "c": np.zeros(3, bool),
            "d": np.zeros(3, bool),
        }

    native = make()
    compared_past_64_in_reals(**native)
    assert list(native["b"]) == [False, True, True]
    assert list(native["c"]) == [True, False, False]
    assert list(native["d"]) == [True] * 3
    agrees(compared_past_64_in_reals, make)
    with pytest.raises(OverflowError, match="too large to convert to float"):
        compared_past_a_double(x=np.ones(2), b=np.zeros(2, bool))
    with pytest.raises(TraceError, match="no real holds it either"):
        compared_past_a_double.trace()


@kernel
def guarded_past_64(m: Int, y: Arr[Fin[n], Int], z: Arr[Fin[n], Int]):  # noqa: F821
    """Guards on the loops and on a scalar that compare past 64 bits."""
    for i in y.dom:
        with when((i < 2**70) & (i > -(2**70))):
            y[i] = 1
        with when((m < 2**63 + 5) & (m > -(2**63) - 5)):
            z[i] = 2


@pytest.mark.filterwarnings("ignore::RuntimeWarning")
def test_a_guard_on_the_loops_past_64_bits_is_left_to_the_predicate():
    # isl states i < 2**70 exactly, and generated a loop bound from it that
    # loopy failed to type (integer constant too large). Such a conjunct does
    # not narrow the domain, and the predicate compares it as numpy does.
    from loopty.trace import constraints_of

    def make(m: int) -> dict:
        return {"m": m, "y": np.zeros(3, np.int64), "z": np.zeros(3, np.int64)}

    for m in (-(2**63), 3, 2**63 - 1):
        agrees(guarded_past_64, lambda m=m: make(m))
    native = make(-1)
    guarded_past_64(**native)
    assert list(native["y"]) == [1] * 3 and list(native["z"]) == [2] * 3
    guard = guarded_past_64.term.stmts[0].guard
    assert constraints_of(guard) == ()


@kernel
def literal_contexts(
    x: Arr[Fin[n], np.float32],  # noqa: F821
    u: Arr[Fin[n], np.uint64],  # noqa: F821
    c: Arr[Fin[n], np.int16],  # noqa: F821
    a: Arr[Fin[n], np.int8],  # noqa: F821
    y: Arr[Fin[n], Real],  # noqa: F821
    b: Arr[Fin[n], Bool],  # noqa: F821
    d: Arr[Fin[n], Bool],  # noqa: F821
    z: Arr[Fin[n], Int],  # noqa: F821
):
    """Integer literals beside a ``float32``, a ``uint64`` and narrow integers."""
    for i in x.dom:
        y[i] = 1.0 * (x[i] * 2**62) + (i + 2.5)
        b[i] = u[i] == 9007199254740993
        d[i] = i < 1.5
        z[i] = (c[i] - -32768) + (a[i] + -128)


@pytest.mark.filterwarnings("ignore::RuntimeWarning")
def test_a_literal_is_written_in_the_type_numpy_computes_with_it():
    # 2**62 beside a float32 was written as a double, which loopy types an
    # int64 literal beside one as, so C multiplied in double where numpy's
    # float32 product is infinite. u[i] == 9007199254740993 of a uint64 was
    # written u[i] == 9007199254740992.0, true at 2**53. pymbolic builds
    # c - -32768 as c + 32768, which numpy refuses beside an int16, and was
    # computed in int; and the interpreter read a + -128 of an int8 as
    # a - 128, which numpy refuses.
    from loopty.interpret import interpret

    def make() -> dict:
        return {
            "x": np.array([3e38, -2.0, 0.5], np.float32),
            "u": np.array([2**53, 2**53 + 1, 3], np.uint64),
            "c": np.array([-3, 32767, 0], np.int16),
            "a": np.array([-1, 100, 0], np.int8),
            "y": np.zeros(3),
            "b": np.zeros(3, bool),
            "d": np.zeros(3, bool),
            "z": np.zeros(3, np.int64),
        }

    native = make()
    literal_contexts(**native)
    assert native["y"][0] == np.inf
    assert list(native["b"]) == [False, True, False]
    assert list(native["d"]) == [True, True, False]
    # int16 sums: 32765 + 127, -1 + -28 and -32768 + -128, wrapped round.
    assert list(native["z"]) == [-32644, -29, 32640]
    agrees(literal_contexts, make)
    interpreted = make()
    interpret(literal_contexts.term, interpreted)
    assert list(interpreted["z"]) == list(native["z"])


@kernel
def unsigned_negative(
    u: Arr[Fin[n], np.uint64],  # noqa: F821
    y: Arr[Fin[n], np.uint64],  # noqa: F821
):
    """A ``uint64`` divided by ``-1``, which numpy refuses."""
    for i in u.dom:
        y[i] = u[i] // -1


@kernel
def truths_scaled(
    b: Arr[Fin[n], Bool],  # noqa: F821
    c: Arr[Fin[n], Bool],  # noqa: F821
    y: Arr[Fin[n], Int],  # noqa: F821
):
    """``//`` of two truth values, an ``int8`` natively, times 200."""
    for i in b.dom:
        y[i] = (b[i] // c[i]) * 200


@kernel
def unsigned_fixed(
    u: Arr[Fin[n], np.uint64],  # noqa: F821
    v: Arr[Fin[n], np.uint32],  # noqa: F821
    y: Arr[Fin[n], Real],  # noqa: F821
    z: Arr[Fin[n], np.uint64],  # noqa: F821
):
    """The named fix, a difference, and a negation, which numpy computes."""
    for i in u.dom:
        y[i] = u[i] // np.int64(-1) + v[i] * np.int64(-2)
        z[i] = (u[i] - 1) + -u[i]


@kernel
def unsigned_times(u: Arr[Fin[n], np.uint64], y: Arr[Fin[n], np.uint64]):  # noqa: F821
    """A ``uint64`` times ``-2``."""
    for i in u.dom:
        y[i] = u[i] * -2


@kernel
def unsigned_mod(u: Arr[Fin[n], np.uint64], y: Arr[Fin[n], np.uint64]):  # noqa: F821
    """A ``uint64`` modulo ``-3``."""
    for i in u.dom:
        y[i] = u[i] % -3


@kernel
def unsigned_xor(u: Arr[Fin[n], np.uint64], y: Arr[Fin[n], np.uint64]):  # noqa: F821
    """``^`` of a ``uint64`` and ``-1``."""
    for i in u.dom:
        y[i] = u[i] ^ -1


@kernel
def unsigned_shift(u: Arr[Fin[n], np.uint32], y: Arr[Fin[n], np.uint32]):  # noqa: F821
    """A ``uint32`` shifted by ``-1``."""
    for i in u.dom:
        y[i] = u[i] << -1


@kernel
def literal_first(
    u: Arr[Fin[n], np.uint16],  # noqa: F821
    y: Arr[Fin[n], Int],  # noqa: F821
):
    """A literal before a numpy integer that does not hold it, in a sum."""
    for i in u.dom:
        y[i] = -7 + u[i]


@kernel
def wide_literal_first(
    k: Arr[Fin[n], np.int32],  # noqa: F821
    y: Arr[Fin[n], Int],  # noqa: F821
):
    """``2**31 + k[i]`` of an ``int32``, which numpy refuses."""
    for i in k.dom:
        y[i] = 2**31 + k[i]


@pytest.mark.filterwarnings("ignore::RuntimeWarning")
def test_a_literal_a_numpy_integer_does_not_hold_is_refused():
    # numpy refuses u[i] // -1 of a uint64 at every point, and the compiled
    # run computed it in double; (b // c) * 200 is int8 * 200 natively.
    with pytest.raises(OverflowError, match="-1 out of bounds for uint64"):
        unsigned_negative(u=np.array([5], np.uint64), y=np.zeros(1, np.uint64))
    with pytest.raises(TraceError, match="the integer -1 with a uint64") as refused:
        unsigned_negative.trace()
    assert "np.int64(-1)" in str(refused.value)
    with pytest.raises(TraceError, match="the integer 200 with a int8"):
        truths_scaled.trace()
    for kern in (unsigned_times, unsigned_mod, unsigned_xor, unsigned_shift):
        with pytest.raises(TraceError, match="the integer -[123] with a uint"):
            kern.trace()
    # A literal first in a sum was written so: pymbolic builds u[i] - 7 with
    # the literal after u[i], and -7 + u[i] as it stands, which numpy refuses.
    with pytest.raises(OverflowError, match="-7 out of bounds for uint16"):
        literal_first(u=np.array([5], np.uint16), y=np.zeros(1, np.int64))
    with pytest.raises(TraceError, match="the integer -7 with a uint16"):
        literal_first.trace()
    with pytest.raises(TraceError, match="the integer 2147483648 with a int32"):
        wide_literal_first.trace()

    def make() -> dict:
        return {
            "u": np.array([5, 2**64 - 1, 0], np.uint64),
            "v": np.array([3, 2**32 - 1, 0], np.uint32),
            "y": np.zeros(3),
            "z": np.zeros(3, np.uint64),
        }

    agrees(unsigned_fixed, make)


@kernel
def xor_signs(
    u: Arr[Fin[n], np.uint64],  # noqa: F821
    k: Arr[Fin[n], Int],  # noqa: F821
    y: Arr[Fin[n], np.uint64],  # noqa: F821
):
    """``^`` of a ``uint64`` and an ``int64``, which numpy refuses."""
    for i in u.dom:
        y[i] = u[i] ^ k[i]


def test_a_bitwise_operation_of_a_uint64_and_a_signed_integer_is_refused():
    # numpy has no ^ of the pair, whose common type is a double, and C
    # computed it in uint64.
    with pytest.raises(TypeError, match="bitwise_xor"):
        xor_signs(
            u=np.array([1], np.uint64), k=np.array([1]), y=np.zeros(1, np.uint64)
        )
    with pytest.raises(TraceError, match="of a uint64 and a signed integer"):
        xor_signs.trace()


# }}}
