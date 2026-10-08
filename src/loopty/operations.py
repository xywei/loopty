"""Operations numpy defines for every input, written in C as numpy computes them.

C leaves some arithmetic undefined that numpy defines (note 20 in
``docs/loopy-notes.md``):

* integer ``a // b`` and ``a % b`` with ``b == 0`` divide by zero, which kills
  the process with ``SIGFPE``, where numpy gives ``0`` (#105); so does the
  smallest integer by ``-1``, which numpy gives as the smallest integer and
  ``0``. loopy's own floor division (``loopy_floor_div_*``) also overflows
  near the ends of the range, at ``a - (b + 1)`` for a large ``a``;
* floating ``a // b`` and ``a % b`` are not lowered by loopy at all (#104);
  numpy computes them with ``fmod`` and a correction toward the sign of the
  divisor, which is Python's;
* ``a << b`` and ``a >> b`` are undefined in C for a shift past the width or a
  negative one, and ``a << b`` for a negative ``a``; numpy gives ``0`` (or
  ``-1`` for a negative ``a`` shifted right) past the width (#107).

Each is a function of two operands of one type, defined in a preamble as
numpy's loops define it (``floor_div``, ``npy_remainder``, ``npy_divmod``,
``npy_lshift`` and ``npy_rshift``). They are emitted where loopy generates
code for the node, by :class:`NumpyArithmetic`, which loopty's targets
(:func:`loopty.lower.target_for`) use for every expression, a guard's
included: loopy's type inference never visits a guard, so a function given to
loopy as a callable is not found there. A floor division or remainder of a
non-negative integer by a positive constant, which C's ``/`` and ``%`` compute
exactly, stays C's, as loopy writes it for an index ``i // 2``.

This module's source, and that of the modules that decide what a lowered
kernel holds, is hashed into the targets' persistent hash
(:data:`CODE_DIGEST`), so loopy's cache of generated code never serves code
generated from another definition or another plan.
"""

from __future__ import annotations

import hashlib
import os
from collections.abc import Iterator, Mapping
from typing import Any

import numpy as np
from loopy.codegen import SeenFunction
from loopy.expression import dtype_to_type_context
from loopy.types import NumpyType
from pymbolic import var

__all__ = [
    "CODE_DIGEST",
    "OPERATIONS",
    "NumpyArithmetic",
    "definition",
    "numpy_arithmetic_preambles",
    "operation_name",
]

#: The operations, by the name of the C functions that compute them.
OPERATIONS = ("loopty_floor_div", "loopty_mod", "loopty_lshift", "loopty_rshift")

#: numpy's integer floor division (``floor_div_@TYPE@``): ``0`` for a zero
#: divisor, and the smallest integer for the smallest integer by ``-1``, which
#: is ``0 - n`` in unsigned arithmetic, where C's ``/`` traps.
_SIGNED_FLOOR_DIV = """
{qualifier} {T} {name}({T} n, {T} d)
{{
  if (d == 0)
    return 0;
  if (d == -1)
    return ({T}) (({U}) 0 - ({U}) n);
  {T} q = n / d;
  if (((n > 0) != (d > 0)) && q * d != n)
    q -= 1;
  return q;
}}"""

#: numpy's integer remainder: ``0`` for a zero divisor and for ``-1``, and the
#: sign of the divisor otherwise.
_SIGNED_MOD = """
{qualifier} {T} {name}({T} n, {T} d)
{{
  if (d == 0 || d == -1)
    return 0;
  {T} r = n % d;
  if ((n > 0) == (d > 0) || r == 0)
    return r;
  return r + d;
}}"""

_UNSIGNED_FLOOR_DIV = """
{qualifier} {T} {name}({T} n, {T} d)
{{
  return d == 0 ? 0 : n / d;
}}"""

_UNSIGNED_MOD = """
{qualifier} {T} {name}({T} n, {T} d)
{{
  return d == 0 ? 0 : n % d;
}}"""

#: numpy's floating floor division (``npy_floor_divide``, ``npy_divmod``):
#: ``a / b`` for a zero divisor, and otherwise the quotient of ``a`` less its
#: ``fmod``, moved down by one where the remainder's sign is not the divisor's,
#: and snapped to the nearest integer.
_FLOATING_FLOOR_DIV = """
{qualifier} {T} {name}({T} a, {T} b)
{{
  if (!b)
    return a / b;
  {T} mod = fmod{f}(a, b);
  {T} div = (a - mod) / b;
  if (mod && ((b < 0) != (mod < 0)))
    div -= 1;
  if (div)
  {{
    {T} floordiv = floor{f}(div);
    if (div - floordiv > ({T}) 0.5)
      floordiv += 1;
    return floordiv;
  }}
  return copysign{f}(({T}) 0, a / b);
}}"""

#: numpy's floating remainder (``npy_remainder``): ``fmod``, moved toward the
#: divisor's sign, with a zero carrying the divisor's sign.
_FLOATING_MOD = """
{qualifier} {T} {name}({T} a, {T} b)
{{
  {T} mod = fmod{f}(a, b);
  if (!b)
    return mod;
  if (mod)
  {{
    if ((b < 0) != (mod < 0))
      mod += b;
  }}
  else
    mod = copysign{f}(({T}) 0, b);
  return mod;
}}"""

#: numpy's shifts (``npy_lshift``, ``npy_rshift``): a shift by the width or
#: more, or by a negative amount, which numpy reads as a huge unsigned one, is
#: ``0``, or ``-1`` for a negative value shifted right. A left shift is done in
#: unsigned arithmetic, which C defines for a negative value too.
_SIGNED_LSHIFT = """
{qualifier} {T} {name}({T} a, {T} b)
{{
  if (({U}) b < {bits})
    return ({T}) (({U}) a << b);
  return 0;
}}"""

_SIGNED_RSHIFT = """
{qualifier} {T} {name}({T} a, {T} b)
{{
  if (({U}) b < {bits})
    return a >> b;
  return a < 0 ? -1 : 0;
}}"""

_UNSIGNED_LSHIFT = """
{qualifier} {T} {name}({T} a, {T} b)
{{
  return b < {bits} ? a << b : 0;
}}"""

_UNSIGNED_RSHIFT = """
{qualifier} {T} {name}({T} a, {T} b)
{{
  return b < {bits} ? a >> b : 0;
}}"""

#: The definitions of each operation, by the kind of its type.
_DEFINITIONS: Mapping[str, Mapping[str, str]] = {
    "loopty_floor_div": {
        "i": _SIGNED_FLOOR_DIV,
        "u": _UNSIGNED_FLOOR_DIV,
        "f": _FLOATING_FLOOR_DIV,
    },
    "loopty_mod": {"i": _SIGNED_MOD, "u": _UNSIGNED_MOD, "f": _FLOATING_MOD},
    "loopty_lshift": {"i": _SIGNED_LSHIFT, "u": _UNSIGNED_LSHIFT},
    "loopty_rshift": {"i": _SIGNED_RSHIFT, "u": _UNSIGNED_RSHIFT},
}

#: What each operation is, for a message.
_SPELLED = {
    "loopty_floor_div": "//",
    "loopty_mod": "%",
    "loopty_lshift": "<<",
    "loopty_rshift": ">>",
}

#: Where the definitions sort among loopy's preambles: after the headers it
#: includes (``10_``), since a definition names their types.
_TAG = "11_loopty"

#: The modules whose source decides the code loopty's targets generate: the
#: definitions and the code generator that calls them, here, the plan of
#: conversions (:mod:`loopty.promotion`) and the lowering (:mod:`loopty.lower`).
_SOURCES = ("operations.py", "promotion.py", "lower.py")


def _digest() -> str:
    """A digest of the sources in :data:`_SOURCES`.

    A module's name stands for it if its source cannot be read, and the
    definitions for this one's.
    """
    digest = hashlib.sha256()
    for name in _SOURCES:
        try:
            with open(os.path.join(os.path.dirname(__file__), name), "rb") as source:
                digest.update(source.read())
        except OSError:  # pragma: no cover - a module with no source on disk
            digest.update(name.encode())
            if name == "operations.py":
                digest.update(repr(sorted(_DEFINITIONS.items())).encode())
    return digest.hexdigest()[:16]


#: A digest of every definition, of the code generator that calls them, and
#: of the modules that decide what the lowered kernel holds, which loopty's
#: targets hash in, so that loopy's persistent cache serves no code another
#: version of them generated. A cache key cannot tell that by the kernel
#: alone: pymbolic's persistent hash reads a numpy scalar as the Python number
#: it equals, so a kernel that writes ``u[i] % 3`` with a ``3`` and one that
#: writes it with an ``np.uint64(3)``, which loopy types and prints otherwise
#: (``3ul``), share an entry (note 20 in ``docs/loopy-notes.md``).
CODE_DIGEST = _digest()


def operation_name(operation: str, dtype: np.dtype) -> str:
    """The name of the C function computing ``operation`` in ``dtype``."""
    return f"{operation}_{np.dtype(dtype).name}"


def definition(operation: str, dtype: np.dtype, target: Any) -> str:
    """The C definition of ``operation`` on operands of ``dtype``, for ``target``.

    An OpenCL target gets ``inline`` functions and its overloaded ``fmod``,
    ``floor`` and ``copysign``; a C one ``static inline`` functions, the
    single-precision forms for a ``float``, and the headers they need.
    """
    from loopy.target.opencl import OpenCLTarget

    dtype = np.dtype(dtype)
    opencl = isinstance(target, OpenCLTarget)
    unsigned = np.dtype(f"u{dtype.itemsize}") if dtype.kind in "iu" else dtype
    code = _DEFINITIONS[operation][dtype.kind].format(
        qualifier="inline" if opencl else "static inline",
        T=target.dtype_to_typename(NumpyType(dtype)),
        U=target.dtype_to_typename(NumpyType(unsigned)),
        name=operation_name(operation, dtype),
        bits=8 * dtype.itemsize,
        f="f" if dtype == np.float32 and not opencl else "",
    )
    if opencl:
        return code
    return "#include <stdint.h>\n#include <math.h>" + code


def numpy_arithmetic_preambles(preamble_info: Any) -> Iterator[tuple[str, str]]:
    """The definitions of the operations a kernel's code calls.

    A preamble generator, as loopy's own ``_preamble_generator`` is for its
    floor division: :class:`NumpyArithmetic` records each function it calls
    among the code generator's seen functions.
    """
    target = preamble_info.kernel.target
    for function in sorted(preamble_info.seen_functions, key=lambda f: f.c_name):
        if function.name not in _DEFINITIONS:
            continue
        dtype = function.result_dtypes[0].numpy_dtype
        yield (
            f"{_TAG}_{function.c_name}",
            definition(function.name, dtype, target),
        )


class NumpyArithmetic:
    """loopy's C expression code generator, with ``//``, ``%``, ``<<``, ``>>`` numpy's.

    A mixin over ``ExpressionToCExpressionMapper`` or one of its subclasses.
    Each operation is a call of the function :func:`definition` writes, in the
    type loopy infers for the node, which the lowering has made numpy's
    (:mod:`loopty.promotion`); the call is recorded among the seen functions
    for :func:`numpy_arithmetic_preambles`.

    Index arithmetic is left to loopy, as it has always written it: a floor
    division or a remainder by a positive constant, in loopy's 32-bit index
    type, of a numerator that reads no array, which is a loop bound isl
    generates (``(1 + n) / 2``), a subscript ``x[i // 2]`` or a guard on the
    loops. loopy writes C's ``/`` for one it finds non-negative and its own
    floor division otherwise, which is exact there, since a loop variable
    and a size are far from the ends of the range.
    """

    def map_sum(self, expr: Any, type_context: Any) -> Any:
        return super().map_sum(expr, self._literal_context(expr, type_context))

    def map_product(self, expr: Any, type_context: Any) -> Any:
        return super().map_product(expr, self._literal_context(expr, type_context))

    def map_bitwise_xor(self, expr: Any, type_context: Any) -> Any:
        return super().map_bitwise_xor(
            expr, self._literal_context(expr, type_context)
        )

    def _literal_context(self, expr: Any, type_context: Any) -> Any:
        """The type context a sum, product or ``^`` writes its literals in.

        loopy writes a Python number in the type context it is handed, and
        hands an operation's operands the context of the place the operation
        stands in: the assignment's, for its right-hand side. So ``k[i] + 3``
        of an integer ``k`` stored into a real was written ``k[i] + 3.0``,
        which C computes in double, rounding past ``2**53`` and never wrapping
        round where numpy's integer sum does, ``-1 * k[i]`` was ``-1.0 *
        k[i]``, ``k[i] ^ 3`` was ``k[i] ^ 3.0``, which C refuses, and ``-1 *
        x[i]`` of a ``float32`` ``x`` was ``-1.0 * x[i]``, a double. An
        operation whose operands are all integers, literals or not, writes its
        literals as integers, and any other in its own type, as numpy computes
        a Python number beside it (note 23 in ``docs/loopy-notes.md``).
        """
        children = expr.children
        if all(
            isinstance(child, int | np.integer)
            or self.infer_type(child).is_integral()
            for child in children
        ):
            return "i"
        own = dtype_to_type_context(self.kernel.target, self.infer_type(expr))
        return own if own is not None else type_context

    def map_constant(self, expr: Any, type_context: Any) -> Any:
        """A ``uint32`` literal as C's ``unsigned int``, ``3u``.

        loopy gives every integer literal of a type wider than 31 bits an
        ``l``, so an ``np.uint32(3)`` was ``3ul``, an ``unsigned long``, and
        ``u[i] + 3ul`` of a ``uint32`` ``u`` was computed in 64 bits, where
        numpy wraps round at ``2**32`` (#122).
        """
        if isinstance(expr, np.uint32):
            from loopy.symbolic import Literal

            return Literal(f"{int(expr)}u")
        return super().map_constant(expr, type_context)

    def map_type_cast(self, expr: Any, type_context: Any) -> Any:
        """A conversion, written out where loopy would leave it out.

        loopy writes a cast only where the type it infers for the operand is
        not the one cast to. It infers ``int8`` for ``a[i] * a[i]`` of an
        ``int8`` ``a``, which C computes as an ``int`` (its integer
        promotion), so the conversion back into ``int8`` that numpy's
        arithmetic in that type stands for (:mod:`loopty.promotion`, #122) was
        left out. A conversion into an integer narrower than ``int`` is
        always written.
        """
        dtype = expr.type.numpy_dtype
        if dtype.kind in "iu" and dtype.itemsize < 4:
            registry = self.codegen_state.ast_builder.target.get_dtype_registry()
            cast = var(f"({registry.dtype_to_ctype(expr.type)}) ")
            return cast(self.rec(expr.child, type_context))
        return super().map_type_cast(expr, type_context)

    def map_floor_div(self, expr: Any, type_context: Any) -> Any:
        return self._numpy_operation(
            "loopty_floor_div", expr, type_context, expr.numerator, expr.denominator
        )

    def map_remainder(self, expr: Any, type_context: Any) -> Any:
        return self._numpy_operation(
            "loopty_mod", expr, type_context, expr.numerator, expr.denominator
        )

    def map_left_shift(self, expr: Any, type_context: Any) -> Any:
        return self._numpy_operation(
            "loopty_lshift", expr, type_context, expr.shiftee, expr.shift
        )

    def map_right_shift(self, expr: Any, type_context: Any) -> Any:
        return self._numpy_operation(
            "loopty_rshift", expr, type_context, expr.shiftee, expr.shift
        )

    def _numpy_operation(
        self, operation: str, expr: Any, type_context: Any, left: Any, right: Any
    ) -> Any:
        dtype = self.infer_type(expr).numpy_dtype
        if dtype.kind == "b":
            # C's integer promotion: a truth value is computed as an int.
            dtype = np.dtype(np.int32)
        if dtype.kind not in _DEFINITIONS[operation]:
            raise TypeError(
                f"{_SPELLED[operation]} of {dtype} operands has no definition "
                "here, and numpy refuses it natively"
            )
        if operation in _LOOPYS and self._index_arithmetic(dtype, left, right):
            return getattr(super(), _LOOPYS[operation])(expr, type_context)
        typ = NumpyType(dtype)
        name = operation_name(operation, dtype)
        self.codegen_state.seen_functions.add(
            SeenFunction(operation, name, (typ, typ), (typ,))
        )
        context = dtype_to_type_context(self.kernel.target, typ)
        return var(name)(self.rec(left, context, typ), self.rec(right, context, typ))

    def _index_arithmetic(
        self, dtype: np.dtype, numerator: Any, denominator: Any
    ) -> bool:
        """Whether a division is index arithmetic, which loopy writes exactly.

        A positive constant divisor, loopy's index type, and a numerator that
        reads no array, a scalar temporary such as a row's length included;
        see the class.
        """
        if dtype != self.kernel.index_dtype.numpy_dtype:
            return False
        if not isinstance(denominator, int | np.integer) or denominator <= 0:
            return False
        from loopy.kernel.array import ArrayBase
        from loopy.symbolic import get_dependencies

        kernel = self.kernel
        arrays = {arg.name for arg in kernel.args if isinstance(arg, ArrayBase)}
        arrays |= {
            name
            for name, temporary in kernel.temporary_variables.items()
            if temporary.shape
        }
        return not (get_dependencies(numerator) & arrays)


#: The method of loopy's code generator for an operation it writes itself,
#: as index arithmetic; see :class:`NumpyArithmetic`.
_LOOPYS = {"loopty_floor_div": "map_floor_div", "loopty_mod": "map_remainder"}
