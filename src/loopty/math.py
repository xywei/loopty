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
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

import numpy as np
import pymbolic.primitives as prim

__all__ = ["conj", "cos", "exp", "imag", "log", "real", "sin", "sqrt"]

#: The functions of this module, by the name a term's call gives each, and
#: the numpy function that computes it natively and in the interpreter.
FUNCTIONS: dict[str, Callable[[Any], Any]] = {
    "sqrt": np.sqrt,
    "exp": np.exp,
    "log": np.log,
    "sin": np.sin,
    "cos": np.cos,
    "real": np.real,
    "imag": np.imag,
    "conj": np.conj,
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
