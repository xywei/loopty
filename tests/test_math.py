"""``loopty.math``: elementary functions and complex parts, in both runs.

``sqrt``, ``exp``, ``log``, ``sin``, ``cos``, ``real``, ``imag`` and ``conj``
are numpy's natively and the target library's compiled, each computed in the
type numpy computes it in: of a real, a complex number, a ``float32`` and an
integer, which numpy takes to a double and loopy refused on the C target and
took to a ``float`` on OpenCL. The interpreter knows each, so a kernel that
calls them has a ``trace-faithful`` fact, and its compiled run is compared
with its native run as any other. A function numpy computes in half
precision, which no target has, is refused when the body is traced.
"""

from __future__ import annotations

import math as python_math

import numpy as np
import pymbolic.primitives as prim
import pytest
from lanky.ledger import Status
from lanky.prelude import Int, Real

from loopty import Arr, Fin, Schedule, TraceError, kernel, reduce_sum, when
from loopty.executor import LoopyExecutor, emit_code
from loopty.faithful import KIND
from loopty.math import conj, cos, exp, imag, log, real, sin, sqrt

pytest.importorskip("loopy")


# {{{ kernels


@kernel
def of_reals(x: Arr[Fin[n], Real], y: Arr[Fin[n], Fin[8], Real]):  # noqa: F821
    """Every function of every cell of a real array."""
    for i in x.dom:
        y[i, 0] = sqrt(x[i])
        y[i, 1] = exp(x[i])
        y[i, 2] = log(x[i])
        y[i, 3] = sin(x[i])
        y[i, 4] = cos(x[i])
        y[i, 5] = real(x[i])
        y[i, 6] = imag(x[i])
        y[i, 7] = conj(x[i])


@kernel
def of_complex(
    z: Arr[Fin[n], np.complex128],  # noqa: F821
    w: Arr[Fin[n], Fin[8], np.complex128],  # noqa: F821
):
    """Every function of every cell of a complex array."""
    for i in z.dom:
        w[i, 0] = sqrt(z[i])
        w[i, 1] = exp(z[i])
        w[i, 2] = log(z[i])
        w[i, 3] = sin(z[i])
        w[i, 4] = cos(z[i])
        w[i, 5] = real(z[i])
        w[i, 6] = imag(z[i])
        w[i, 7] = conj(z[i])


@kernel
def of_singles(
    x: Arr[Fin[n], np.float32],  # noqa: F821
    z: Arr[Fin[n], np.complex64],  # noqa: F821
    y: Arr[Fin[n], Fin[3], np.float32],  # noqa: F821
    w: Arr[Fin[n], Fin[2], np.complex64],  # noqa: F821
):
    """Single precision stays single precision, as numpy computes it."""
    for i in x.dom:
        y[i, 0] = sqrt(x[i])
        y[i, 1] = real(z[i])
        y[i, 2] = imag(z[i])
        w[i, 0] = sqrt(z[i])
        w[i, 1] = conj(z[i])


@kernel
def of_integers(k: Arr[Fin[n], Int], y: Arr[Fin[n], Fin[6], Real]):  # noqa: F821
    """Functions of integers and of the loop variable: doubles, numpy says."""
    for i in k.dom:
        y[i, 0] = sqrt(k[i])
        y[i, 1] = log(k[i] + 1)
        y[i, 2] = sqrt(i)
        y[i, 3] = real(k[i]) / 2
        y[i, 4] = imag(k[i]) + 0.5
        y[i, 5] = conj(i) * 0.25


@kernel
def single_layer(
    t: Arr[Fin[n], np.complex128],  # noqa: F821
    c: Arr[Fin[n], np.complex128],  # noqa: F821
    y: Arr[Fin[m], np.complex128],  # noqa: F821
    w: Arr[Fin[m], Real],  # noqa: F821
    out: Arr[Fin[n], Real],  # noqa: F821
):
    """The line-Taylor single layer of pytential's QBX, to order 4."""
    for p in out.dom:
        out[p] = reduce_sum(
            w[s]
            * real(
                log(c[p] - y[s])
                - reduce_sum(
                    ((y[s] - t[p]) / (c[p] - y[s])) ** k / k for k in Fin[5] if k >= 1
                )
            )
            for s in y.dom
        )


@kernel
def narrow(k: Arr[Fin[n], np.int8], y: Arr[Fin[n], Real]):  # noqa: F821
    """``sqrt`` of an ``int8``, which numpy computes in half precision."""
    for i in k.dom:
        y[i] = sqrt(k[i])


@kernel
def roots(
    x: Arr[Fin[n], Real],  # noqa: F821
    y: Arr[Fin[n], Real],  # noqa: F821
) -> all(y[i] == sqrt(x[i]) for i in Fin[n] if x[i] >= 0):  # noqa: F821
    """A postcondition that calls loopty.math as the body does.

    Where ``x[i]`` is negative both sides are ``nan``, which is equal to
    nothing, so the claim is made where it is a number.
    """
    for i in y.dom:
        y[i] = sqrt(x[i])


@kernel
def compared(x: Arr[Fin[n], Real], y: Arr[Fin[n], Real]):  # noqa: F821
    """A function's value in a guard, compared as any value is.

    The value calls ``sqrt`` too: loopy resolves a function by the values
    that call it, and a guard alone is refused when lowered (note 20).
    """
    for i in y.dom:
        with when(sqrt(x[i]) > 1.5):
            y[i] = sqrt(x[i]) + abs(sin(x[i]))


@kernel
def numpys_sqrt(x: Arr[Fin[n], Real], y: Arr[Fin[n], Real]):  # noqa: F821
    """numpy's sqrt, as a body that runs on numbers may call it."""
    for i in y.dom:
        y[i] = np.sqrt(x[i])


@kernel
def numpys_tanh(x: Arr[Fin[n], Real], y: Arr[Fin[n], Real]):  # noqa: F821
    """numpy's tanh, which loopty.math has not."""
    for i in y.dom:
        y[i] = np.tanh(x[i])


@kernel
def pythons_sqrt(x: Arr[Fin[n], Real], y: Arr[Fin[n], Real]):  # noqa: F821
    """Python's math.sqrt."""
    for i in y.dom:
        y[i] = python_math.sqrt(x[i])


@kernel
def powers(
    z: Arr[Fin[n], np.complex128],  # noqa: F821
    w: Arr[Fin[n], Fin[4], np.complex128],  # noqa: F821
):
    """Powers of a complex number by 1, 3, 5 and 7, as a written-out series has them."""
    for i in z.dom:
        w[i, 0] = z[i] ** 1
        w[i, 1] = z[i] ** 3
        w[i, 2] = z[i] ** 5
        w[i, 3] = z[i] ** 7


@kernel
def real_part(
    z: Arr[Fin[n], np.complex128],  # noqa: F821
    y: Arr[Fin[n], Real],  # noqa: F821
):
    """The real part, read as numpy's attribute."""
    for i in y.dom:
        y[i] = z[i].real


# }}}


def agrees(kern, inputs: dict) -> None:
    """The compiled run agrees with the native one, and the trace is faithful."""
    fact = LoopyExecutor().differential(kern, Schedule(kern), inputs)
    assert fact.status is Status.TESTED, fact.provenance
    (faithful,) = [f for f in kern.facts() if f.kind == KIND]
    assert faithful.status is Status.TESTED, faithful.provenance


def test_natively_each_function_is_numpys() -> None:
    assert sqrt(4.0) == 2.0 and isinstance(sqrt(np.float32(4.0)), np.float32)
    assert log(-1 + 0j) == np.log(-1 + 0j)
    assert real(3 - 4j) == 3.0 and imag(3 - 4j) == -4.0 and conj(3 - 4j) == 3 + 4j
    assert real(2.5) == 2.5 and imag(2.5) == 0.0 and conj(2.5) == 2.5
    with pytest.raises(TypeError, match=r"sqrt\(x\[i\]\), and not an array"):
        sqrt(Arr.from_numpy(np.ones(3)))


def test_traced_each_is_a_call_the_term_keeps() -> None:
    (stmt, *_rest) = of_reals.term.stmts
    assert isinstance(stmt.expr, prim.Call)
    assert stmt.expr.function.name == "sqrt"


def test_functions_of_reals_agree_compiled_and_native() -> None:
    x = np.array([0.25, 1.0, 2.0, 7.5, 1e-3])
    inputs = {"x": Arr.from_numpy(x), "y": Arr.zeros((5, 8))}
    agrees(of_reals, inputs)
    native = {"x": Arr.from_numpy(x), "y": Arr.zeros((5, 8))}
    of_reals(**native)
    compiled = LoopyExecutor().run(of_reals, x=x, y=np.zeros((5, 8)))["y"]
    # sqrt is correctly rounded in both libraries, and the parts are exact.
    for column in (0, 5, 6, 7):
        assert np.array_equal(compiled[:, column], native["y"].numpy()[:, column])
    code = emit_code(of_reals)
    assert "creal" not in code and "cimag" not in code and "conj" not in code


def test_functions_of_complex_numbers_agree_compiled_and_native() -> None:
    z = np.array([1 + 2j, -3 + 0.5j, 0.25 - 1j, -1 + 0j, 2j])
    agrees(of_complex, {"z": Arr.from_numpy(z), "w": Arr.zeros((5, 8), np.complex128)})
    code = emit_code(of_complex)
    for name in ("csqrt", "cexp", "clog", "csin", "ccos", "creal", "cimag", "conj"):
        assert f"{name}(" in code, name


def test_single_precision_stays_single() -> None:
    inputs = {
        "x": Arr.from_numpy(np.array([0.5, 2.0, 9.0], np.float32)),
        "z": Arr.from_numpy(np.array([1 + 1j, -2j, 3 + 0j], np.complex64)),
        "y": Arr.zeros((3, 3), np.float32),
        "w": Arr.zeros((3, 2), np.complex64),
    }
    agrees(of_singles, inputs)
    code = emit_code(of_singles)
    assert "sqrtf(" in code and "crealf(" in code and "csqrtf(" in code


def test_functions_of_integers_are_doubles_in_both_runs() -> None:
    # loopy refused sqrt of an integer on the C target ("sqrt does not
    # support type float32"), and took it for a float on OpenCL.
    k = np.array([0, 1, 2, 9, 1000], np.int64)
    native = {"k": Arr.from_numpy(k), "y": Arr.zeros((5, 6))}
    of_integers(**native)
    assert native["y"].numpy()[3, 0] == 3.0
    assert native["y"].numpy()[2, 3] == 1.0
    assert native["y"].numpy()[2, 4] == 0.5
    agrees(of_integers, {"k": Arr.from_numpy(k), "y": Arr.zeros((5, 6))})
    code = emit_code(of_integers)
    assert "sqrt((double) (k[i]))" in code
    assert "imag" not in code and "conj" not in code


def test_the_qbx_single_layer_of_complex_points_agrees() -> None:
    rng = np.random.default_rng(3)
    t = np.exp(2j * np.pi * np.arange(6) / 6)
    c = 0.9 * t
    y = 1.6 * np.exp(2j * np.pi * (np.arange(9) + 0.5) / 9)
    inputs = {
        "t": Arr.from_numpy(t),
        "c": Arr.from_numpy(c),
        "y": Arr.from_numpy(y),
        "w": Arr.from_numpy(rng.random(9)),
        "out": Arr.zeros(6),
    }
    agrees(single_layer, inputs)


@pytest.mark.filterwarnings("ignore:invalid value encountered:RuntimeWarning")
@pytest.mark.filterwarnings("ignore:divide by zero encountered:RuntimeWarning")
def test_outside_the_domain_both_runs_give_nan_and_infinity() -> None:
    x = np.array([-1.0, 0.0, 4.0])
    native = {"x": Arr.from_numpy(x), "y": Arr.zeros((3, 8))}
    of_reals(**native)
    compiled = LoopyExecutor().run(of_reals, x=x, y=np.zeros((3, 8)))["y"]
    assert np.isnan(compiled[0, 0]) and np.isnan(native["y"].numpy()[0, 0])
    assert compiled[1, 2] == native["y"].numpy()[1, 2] == -np.inf


def test_a_function_numpy_computes_in_half_precision_is_refused() -> None:
    with pytest.raises(TraceError) as caught:
        narrow.trace()
    message = str(caught.value)
    assert "sqrt(k[i]) at" in message
    assert "is a float16 natively, the type numpy computes it in for an int8" in message
    assert message.endswith("write sqrt(1.0 * k[i]) for a double")


def test_opencl_code_converts_an_integer_before_the_call(plain_opencl) -> None:
    # loopy's OpenCL target takes sqrt of an integer for a float.
    from loopty.isl_reading import declining

    lp = plain_opencl
    with lp.CacheMode(False), declining():
        code = lp.generate_code_v2(
            Schedule(of_integers, target="opencl").kernel
        ).device_code()
    assert "sqrt((double) (k[i]))" in code


def test_opencl_code_calls_pyopencls_complex_functions() -> None:
    # loopy's plain OpenCL target has no complex functions; the pyopencl one
    # calls those of pyopencl's complex header. Generating the code needs
    # pyopencl and no device.
    pytest.importorskip("pyopencl")
    import loopy as lp

    from loopty.isl_reading import declining

    with lp.CacheMode(False), declining():
        code = lp.generate_code_v2(
            Schedule(of_complex, target="opencl").kernel
        ).device_code()
    for name in ("cdouble_sqrt", "cdouble_log", "cdouble_real", "cdouble_conj"):
        assert f"{name}(" in code, name


@pytest.mark.parametrize("exponent", range(1, 13))
def test_a_complex_power_by_a_natural_number_is_numpys_products(exponent) -> None:
    # The products the OpenCL lowering writes, evaluated in numpy's complex
    # arithmetic, are numpy's own power by an integer, to the bit.
    from pymbolic import evaluate

    from loopty.lower import _natural_power

    rng = np.random.default_rng(exponent)
    for value in rng.normal(size=8) + 1j * rng.normal(size=8):
        z = np.complex128(value)
        product = evaluate(_natural_power(prim.Variable("z"), exponent), {"z": z})
        assert product == z**exponent, (z, exponent)


def test_opencl_code_writes_a_complex_power_as_products() -> None:
    # loopy writes a complex power by 1, 5 or 7 as cdouble_powr, exp(n log z),
    # and only 2 to 4 as products; loopty writes every natural one as products.
    pytest.importorskip("pyopencl")
    import loopy as lp

    from loopty.isl_reading import declining

    with lp.CacheMode(False), declining():
        code = lp.generate_code_v2(
            Schedule(powers, target="opencl").kernel
        ).device_code()
    body = code[code.index("__kernel") :]
    assert "powr" not in body and "cdouble_pow(" not in body
    assert "cdouble_mul(" in body


def test_on_an_opencl_device_complex_functions_agree_with_numpy() -> None:
    # Run by hand on a device (docs/device-runs.md); CI has none.
    cl = pytest.importorskip("pyopencl")
    try:
        cl.create_some_context(interactive=False)
    except Exception as exc:  # noqa: BLE001 - no platform, no device
        pytest.skip(f"no OpenCL device: {exc}")
    z = np.array([1 + 2j, -3 + 0.5j, 0.25 - 1j, -1 + 0j, 2j])
    inputs = {"z": Arr.from_numpy(z), "w": Arr.zeros((5, 8), np.complex128)}
    fact = LoopyExecutor(target="opencl").differential(
        of_complex, Schedule(of_complex, target="opencl"), inputs
    )
    assert fact.status is Status.TESTED, fact.provenance
    k = np.array([0, 1, 2, 9, 1000], np.int64)
    fact = LoopyExecutor(target="opencl").differential(
        of_integers,
        Schedule(of_integers, target="opencl"),
        {"k": Arr.from_numpy(k), "y": Arr.zeros((5, 6))},
    )
    assert fact.status is Status.TESTED, fact.provenance


def test_a_postcondition_calls_the_functions_the_body_calls() -> None:
    # The claim is evaluated after each native run with numpy's sqrt, where
    # it was "not evaluated: UnknownVariableError: sqrt" and stayed assumed.
    (post,) = [fact for fact in roots.facts() if fact.kind == "postcondition"]
    assert post.statement.endswith("y[i] == sqrt(x[i])")
    assert post.provenance["native"] == "tested", post.provenance
    # and decided from the term (#113): what y stores is the claim, an
    # equation of reals isl reads as an atom
    assert post.status is Status.DECIDED, post.provenance



def test_a_functions_value_compares_in_a_guard() -> None:
    # A plain pymbolic call refused the comparison: "expressions don't have
    # an order".
    x = np.array([1.0, 4.0, 2.25, 9.0])
    native = {"x": Arr.from_numpy(x), "y": Arr.zeros(4)}
    compared(**native)
    assert native["y"].numpy()[0] == 0.0 and native["y"].numpy()[1] > 0.0
    agrees(compared, {"x": Arr.from_numpy(x), "y": Arr.zeros(4)})


@pytest.mark.parametrize(
    ("kern", "message"),
    [
        (
            numpys_sqrt,
            r"numpy.sqrt of a traced value at test_math.py:\d+: numpy's "
            r"functions compute on numbers, .* Write sqrt\(...\) from "
            r"loopty.math \(from loopty.math import sqrt\)",
        ),
        (
            numpys_tanh,
            r"numpy.tanh of a traced value at test_math.py:\d+: .* loopty.math "
            r"has sqrt, exp, log, sin, cos, real, imag and conj, .* and no tanh",
        ),
        (
            pythons_sqrt,
            r"a traced value was made a Python number at test_math.py:\d+, as "
            r"float\(\) does, and math.sqrt .* Call loopty.math's functions",
        ),
        (
            real_part,
            r"z\[i\].real at test_math.py:\d+: while a kernel is traced z\[i\] "
            r"is a term, which has no attribute real. Write real\(z\[i\]\)",
        ),
    ],
)
def test_numpy_and_pythons_math_name_loopty_math_when_traced(
    kern, message: str
) -> None:
    # Natively each is fine; traced, each failed far from the body, with
    # numpy's "loop of ufunc does not support argument 0 of type Subscript",
    # pymbolic's "UnknownVariableError: x", or "'Subscript' object has no
    # attribute 'real'".
    with pytest.raises(TraceError, match=message):
        kern.trace()
