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

A theorem may call them too, directly or through a helper the kernel calls:
``ratio(alpha, beta)``, which a kernel's body computes with ``conj`` and
``real``, is in a theorem's annotation the same expression over the
theorem's variables, in lanky's own functions (:class:`lanky.terms.Elementary`),
which every oracle reads: ``conj(alpha)`` there is the conjugate, and not a
family named ``conj``; where lanky runs the annotation again at numbers, the
numbers are Python's, as lanky computes them. A kernel's own annotations,
its postcondition, keep the kernel's spelling (:func:`as_traced`), which is
what its body's trace compares them with, and :func:`as_stated` puts a
kernel's term in a theorem's.

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

__all__ = ["as_stated", "as_traced", "conj", "cos", "exp", "imag", "integer_power",
           "ipow", "log", "real", "sin", "sqrt"]

def integer_power(x: Any, n: Any) -> Any:
    """``x`` to the integer power ``n``, by squaring, as loopy's C does it.

    loopy defines ``loopy_pow_<base>_<exponent>`` for a power with an
    integer exponent (``loopy.target.c``): ``1`` at ``n == 0``; for a
    negative ``n``, ``x`` replaced by ``1 / x`` and ``n`` by ``-n``; then,
    while ``n > 1``, the odd factor multiplied into ``y`` and ``x`` squared;
    and ``x * y`` at the end. This is that loop, in ``x``'s own type, so the
    native run and the compiled one multiply the same numbers in the same
    order.

    The one power it cannot match is the smallest of the integer type the
    compiled kernel holds ``n`` in, whose negation that type does not hold:
    there loopy's ``-n`` stays negative and its loop computes another power.
    So it is refused, ``-2**31`` and ``-2**63`` of a Python integer, and the
    smallest value of a numpy one.
    """
    smallest = (
        {int(np.iinfo(n.dtype).min)}
        if isinstance(n, np.signedinteger)
        else {-(2**31), -(2**63)}
        if isinstance(n, int) and not isinstance(n, bool)
        else set()
    )
    if int(n) in smallest:
        raise ValueError(
            f"loopty.math.ipow cannot take the power {int(n)}: the compiled kernel "
            "holds it in an integer type that has no -n, and loopy's integer "
            "power computes another power there"
        )
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
    stated = _statements_function(name, value)
    if stated is not None:
        return stated(value)
    if isinstance(value, prim.ExpressionNode):
        return Call(Var(name), (value,))
    return FUNCTIONS[name](value)


def _statements_function(name: str, value: Any) -> Any:
    """lanky's ``name`` where a statement's annotation is read or rerun, or ``None``.

    A theorem's annotation that calls ``conj(alpha)``, itself or through a
    helper, means the conjugate, which lanky's oracles read as
    :class:`lanky.terms.Elementary`; a kernel's body is traced outside any
    annotation, and keeps the call its lowering knows. Where lanky runs the
    annotation again at numbers, to check that its term computes what it
    computes (:mod:`lanky.faithful`), the numbers are Python's, as lanky's
    functions compute them, so that ``1 / real(0j)`` stops as Python stops
    where numpy's would be ``inf``. A lanky without the functions, or that
    cannot say whether an annotation is read, gets numpy's and the call.
    """
    try:
        from lanky import terms
    except ImportError:  # pragma: no cover - lanky is a dependency
        return None
    if name not in getattr(terms, "ELEMENTARY_FUNCTIONS", ()):
        return None
    if isinstance(value, prim.ExpressionNode):
        asked = getattr(terms, "reading_annotation", None)
    else:
        asked = getattr(terms, "rerunning", None)
    if asked is None or not asked():
        return None
    return getattr(terms, name, None)


def _mapped(expr: Any, leaf: Callable[[Any], Any]) -> Any:
    """``expr`` with each node ``leaf`` rewrites rewritten, and the rest as it was.

    A kernel's sum (:class:`loopty.term.Reduction`) is no pymbolic node, and
    its body is rewritten too.
    """
    import dataclasses

    from lanky.terms import init_args

    from loopty.term import Reduction

    found = leaf(expr)
    if found is not None:
        return found
    if isinstance(expr, Reduction):
        body = _mapped(expr.body, leaf)
        return expr if body is expr.body else dataclasses.replace(expr, body=body)
    if isinstance(expr, tuple):
        mapped = tuple(_mapped(item, leaf) for item in expr)
        same = all(a is b for a, b in zip(mapped, expr, strict=True))
        return expr if same else mapped
    if isinstance(expr, prim.ExpressionNode):
        args = init_args(expr)
        mapped = tuple(_mapped(arg, leaf) for arg in args)
        if all(a is b for a, b in zip(mapped, args, strict=True)):
            return expr
        return type(expr)(*mapped)
    return expr


def as_traced(expr: Any) -> Any:
    """``expr`` with lanky's functions of this module's names as a kernel's calls.

    A kernel's postcondition is read as an annotation, where ``sqrt(x[i])``
    is lanky's :class:`~lanky.terms.Elementary`; its body's trace records
    the call ``sqrt(x[i])``, and the two are compared, by isl's atoms and
    natively, so the postcondition is put in the trace's spelling. Anything
    else is left as it is.
    """
    from lanky.terms import Call, Elementary, Var

    def leaf(node: Any) -> Any:
        if isinstance(node, Elementary) and node.function in FUNCTIONS:
            return Call(Var(node.function), (as_traced(node.argument),))
        return None

    return _mapped(expr, leaf)


def as_stated(expr: Any) -> Any:
    """``expr`` with this module's calls as lanky's functions, as a theorem states them.

    The converse of :func:`as_traced`: a kernel's term records ``conj(alpha)``
    as a call, and a theorem the kernel cites has the conjugate, lanky's
    :class:`~lanky.terms.Elementary`, so a term is put in the theorem's
    spelling before the two are matched, and before Lean is shown it. A
    call of a function lanky has no node for, ``sin``, is left a call.
    """
    from lanky import terms

    stated = getattr(terms, "ELEMENTARY_FUNCTIONS", ())

    def leaf(node: Any) -> Any:
        if (
            isinstance(node, prim.Call)
            and isinstance(node.function, prim.Variable)
            and node.function.name in FUNCTIONS
            and node.function.name in stated
            and len(node.parameters) == 1
        ):
            return terms.Elementary(node.function.name, as_stated(node.parameters[0]))
        return None

    return _mapped(expr, leaf)


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

