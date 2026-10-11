"""Elementary functions and complex parts, for kernel bodies.

::

    from loopty.math import log, real, sqrt

    @kernel
    def single_layer(z, c, y, w, out):
        for t in out.dom:
            out[t] = reduce_sum(w[s] * real(log(c[t] - y[s])) for s in y.dom)

Each function is numpy's when the body runs natively, on the numbers it is
given, and a call the trace records when the body is traced, which the term
interpreter evaluates with the same numpy function and the lowering writes as
the target's library function: ``sqrt`` and ``log`` of a double, ``csqrt``
and ``clog`` of a complex number on the C target, ``cdouble_sqrt`` and
``cdouble_log`` on OpenCL. So a kernel that uses them runs, traces, checks and
compiles as any other, and its compiled run is compared with its native run.

The type of each result is numpy's, as for every other operation
(:mod:`loopty.promotion`): ``sqrt`` of a ``float32`` is single precision,
``sqrt(k[i])`` of an integer ``k`` a double, which the lowering computes as
``sqrt((double) k[i])``, ``real`` and ``imag`` of a ``complex128`` doubles,
and ``real``, ``imag`` and ``conj`` of a number that is not complex the number
itself, ``0`` and the number, each in numpy's type. One numpy computes in a
type the compiled kernel has none of is refused when the body is traced:
``sqrt`` of an ``int8`` or of a truth value is a ``float16`` natively, and the
trace names ``sqrt(1.0 * k[i])``, a double.

A function of a value is what these take, ``sqrt(x[i])``; an array whole,
``sqrt(x)``, is refused natively as it is when traced.

``ipow(x, n)`` is ``x`` to the integer power ``n``, multiplied out as loopy
computes an integer power: by squaring, ``x * x`` and the odd factors taken
along, rounded at every product. ``x ** n`` of a real ``x`` is numpy's, the C
library's ``pow``, which the compiled kernel calls too, so that both runs
round alike; ``ipow`` is the same multiplications in both runs, so they still
agree bit for bit, and the compiled one costs a few products where ``pow``
costs a library call. It is not ``x ** n``: the two can differ in the last
bits, which is why it has a name of its own.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

import numpy as np
import pymbolic.primitives as prim

__all__ = ["conj", "cos", "exp", "imag", "integer_power", "ipow", "log", "real", "sin",
           "sqrt"]

def integer_power(x: Any, n: Any) -> Any:
    """``x`` to the integer power ``n``, by squaring, as loopy's C does it.

    loopy defines ``loopy_pow_<base>_<exponent>`` for a power with an
    integer exponent (``loopy.target.c``): ``1`` at ``n == 0``; for a
    negative ``n``, ``x`` replaced by ``1 / x`` and ``n`` by ``-n``; then,
    while ``n > 1``, the odd factor multiplied into ``y`` and ``x`` squared;
    and ``x * y`` at the end. This is that loop, in ``x``'s own type, so the
    native run and the compiled one multiply the same numbers in the same
    order.
    """
    x = np.float64(x) if isinstance(x, float) else x
    n = int(n)
    one = x.dtype.type(1) if isinstance(x, np.generic) else 1.0
    if n == 0:
        return one
    if n < 0:
        x = one / x
        n = -n
    y = one
    while n > 1:
        if n % 2:
            y = x * y
        x = x * x
        n = n // 2
    return x * y


#: The functions of this module, by the name a term's call gives each, and
#: the numpy function that computes it natively and in the interpreter.
FUNCTIONS: dict[str, Callable[..., Any]] = {
    "sqrt": np.sqrt,
    "exp": np.exp,
    "log": np.log,
    "sin": np.sin,
    "cos": np.cos,
    "real": np.real,
    "imag": np.imag,
    "conj": np.conj,
    "ipow": integer_power,
}


def _applied(name: str, value: Any) -> Any:
    """``name`` of ``value``: numpy's natively, a recorded call when traced.

    The call is lanky's, as every traced value is, so that it compares and
    combines like one: ``when(sqrt(x[i]) > 1.5)`` is a guard, where a plain
    pymbolic call refused the comparison ("expressions don't have an order").
    """
    from lanky.terms import Call, Var

    from loopty.arr import Arr

    if isinstance(value, Arr) or isinstance(value, np.ndarray) and value.ndim:
        raise TypeError(
            f"loopty.math.{name} takes one value, such as {name}(x[i]), and not "
            "an array: a kernel computes cell by cell"
        )
    if isinstance(value, prim.ExpressionNode):
        return Call(Var(name), (value,))
    return FUNCTIONS[name](value)


def sqrt(x: Any) -> Any:
    """The square root, ``numpy.sqrt``: of a negative real it is ``nan``."""
    return _applied("sqrt", x)


def exp(x: Any) -> Any:
    """The exponential, ``numpy.exp``."""
    return _applied("exp", x)


def log(x: Any) -> Any:
    """The natural logarithm, ``numpy.log``; of a complex number, the principal one."""
    return _applied("log", x)


def sin(x: Any) -> Any:
    """The sine, ``numpy.sin``."""
    return _applied("sin", x)


def cos(x: Any) -> Any:
    """The cosine, ``numpy.cos``."""
    return _applied("cos", x)


def real(x: Any) -> Any:
    """The real part, ``numpy.real``: of a number that is not complex, itself."""
    return _applied("real", x)


def imag(x: Any) -> Any:
    """The imaginary part, ``numpy.imag``: of a number that is not complex, ``0``."""
    return _applied("imag", x)


def conj(x: Any) -> Any:
    """The complex conjugate, ``numpy.conj``: of a real number, itself."""
    return _applied("conj", x)


def ipow(x: Any, n: Any) -> Any:
    """``x`` to the integer power ``n``, multiplied out as loopy does it.

    See the module: the same products natively and compiled, where ``x **
    n`` is the C library's ``pow`` in both. ``x`` is a real number, of a
    floating type, and ``n`` an integer, a loop variable, an integral cell or
    a literal; ``ipow(x[i], k)`` and ``ipow(x[i], 3)``.
    """
    from lanky.terms import Call, Var

    from loopty.arr import Arr

    for value, what in ((x, "x"), (n, "n")):
        if isinstance(value, Arr) or isinstance(value, np.ndarray) and value.ndim:
            raise TypeError(
                f"loopty.math.ipow takes one value for {what}, such as "
                "ipow(x[i], k), and not an array: a kernel computes cell by cell"
            )
    if isinstance(n, float | np.floating | complex | np.complexfloating) or (
        isinstance(n, bool | np.bool_)
    ):
        raise TypeError(
            f"loopty.math.ipow takes an integer power, not {n!r}; x ** n is the "
            "power of a real exponent"
        )
    if isinstance(x, bool | int | np.integer | np.bool_) or (
        isinstance(x, complex | np.complexfloating)
    ):
        raise TypeError(
            f"loopty.math.ipow takes a real base of a floating type, not {x!r}: "
            "for an integer or a complex number, write x ** n"
        )
    if isinstance(x, prim.ExpressionNode) or isinstance(n, prim.ExpressionNode):
        return Call(Var("ipow"), (x, n))
    return integer_power(x, n)

