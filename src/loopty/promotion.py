"""The type each operation of a term computes in, natively and compiled.

The native run computes with numpy's arithmetic, and the compiled one with C's,
and the two disagree about types in more places than a bare constant (note 17
in ``docs/loopy-notes.md``). numpy promotes by NEP 50: a numpy scalar is
*strong* and keeps its dtype, a Python number is *weak* and takes the dtype of
what stands beside it, and true division of two integers is a double. C
converts by its usual arithmetic conversions: a ``float`` beside any integer is
a ``float``, and ``/`` of two integers is integer division. So, among others:

* ``k[i] / 2 * 2`` of an integer ``k`` is ``3.0`` natively at ``k = 3`` and
  ``2`` compiled, where the quotient is C's integer division (#82);
* ``x[i] * 0.1`` of a ``float32`` ``x`` is single precision natively, since
  ``0.1`` is weak, and double compiled, where the literal is a double
  (#91);
* ``x[i] * k[i]`` of a ``float32`` ``x`` and an integer ``k`` is a double
  natively, since ``k`` is a strong 64-bit integer, and a ``float`` compiled;
* ``x[i] ** 3`` is the C library's ``pow`` natively and loopy's repeated
  multiplication compiled, which rounds at every step.

:class:`Promotion` reads both types off a term and says, for each operation,
which operand the lowering has to convert so that C computes it in the type
numpy computes it in (:class:`Step`). The native type is found by doing the
operation in numpy on a sample of each operand's type (the literal itself for a
literal), so the rules are numpy's own and not a table of them. The compiled
type is C's, with the operands typed as the lowering declares them
(:func:`loopty.contract.compiled_storage`), and an operation the plan converts
is then computed in numpy's type.

A leaf's native type is the one the native run gives it. An array element is a
numpy scalar of :func:`loopty.contract.native_storage`'s dtype (64 bits for an
integral sort). A loop variable, a size and a reduction binder are Python
ints. A scalar argument is a numpy scalar of that dtype too, however the
caller passed it (:func:`loopty.contract.native_scalar`, #102). Anything whose
type is not known here (a call of a function the interpreter does not know, a
``min``) is left as it is, and its type is unknown above it, and so is an
operation numpy refuses (an integer to a negative integer power, which the
trace refuses, #109), which the native run raises on whatever the compiled run
does.

Integer arithmetic is computed in 64 bits (#101), as numpy computes an
element's: ``Nat`` and ``Int`` are stored so compiled, and an operation that
can leave the range of its operands (a sum, a product, a power, a left shift)
whose operands C would compute in fewer bits, a ``Fin[m]`` element's (stored
in 32) or a loop variable's, has one converted. A floor division, a
remainder, an ``^`` and a right shift stay inside their operands' range and
are left as they are, and so is a sum of loop variables, sizes and literals
alone, the index arithmetic loopy computes every loop bound in, 32 bits wide;
a product of them is not, since ``i * i`` leaves 32 bits at ``i = 46341``,
and nor is a sum whose literals total ``2**30`` or more (``i + 2**31 - 1``).
The lowering widens an integer in a guard that reads no array as everywhere
else, with no cast (``when(i * i < m)`` wrapped round at ``i = 46341``), and
leaves a subscript's arithmetic in 32 bits, which loopy gives it no way to
widen, a limit (#129; :class:`loopty.lower.ExpressionLowerer`). A result
outside 64 bits wraps round compiled and is refused or wraps natively, which
is numpy's limit too.

The type C computes an operation in is C's, by its integer promotion and its
usual arithmetic conversions (:func:`_c_result`), and not numpy's, in two
more places (#122):

* an integer narrower than ``int`` (``np.int8``, ``np.uint16``, a truth
  value) is computed as an ``int``, where numpy computes ``int8 * int8`` in
  ``int8``: ``a[i] * a[i] // 2`` at ``a[i] = 100`` was ``8`` natively and
  ``-120`` compiled. Where numpy computes an operation in such a type, its
  result is converted back into it (:attr:`Step.result`), which wraps round
  as numpy's does;
* an unsigned integer beside a signed one of no more bits is unsigned in C,
  where numpy computes in a signed type that holds both: ``u[i] + k[i]`` of a
  ``uint32`` ``u`` and an ``int32`` ``k`` is ``int64`` natively, and was
  ``4294967295`` compiled at ``0 + -1``. The operands are converted then,
  and a comparison of integers, which numpy compares exactly whatever their
  types, is compared on the sign first where C would compare a negative
  value as an unsigned one (:attr:`Step.sign`).

An operation the lowering writes as a call of a function, ``//``, ``%``,
``<<``, ``>>`` and ``**``, is computed in the type loopy infers for it
(:func:`_loopy_result`), and is converted back the same way where that is not
numpy's. An integer literal past 64 bits has no type loopy can give it, and is
written in numpy's type where numpy computes in one that holds it, a real or
a ``uint64``, and refused by the trace elsewhere (#140).
"""

from __future__ import annotations

import dataclasses
import operator
import warnings
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from typing import Any

import numpy as np
import pymbolic.primitives as prim

from loopty.contract import array_storage, compiled_storage, native_storage
from loopty.term import Access, ArrType, Reduction, Term

__all__ = ["Promotion", "Step"]

#: What the native type of an expression is: one sample value of each type the
#: native run may give it, a numpy scalar for a strong type and a Python number
#: for a weak one, or ``None`` when it is not known.
Native = tuple[Any, ...] | None

#: The operator of each arithmetic node, applied to samples.
_ARITHMETIC: dict[type, Callable[[Any, Any], Any]] = {
    prim.Sum: operator.add,
    prim.Product: operator.mul,
    prim.Quotient: operator.truediv,
    prim.FloorDiv: operator.floordiv,
    prim.Remainder: operator.mod,
    prim.Power: operator.pow,
    prim.BitwiseXor: operator.xor,
    prim.LeftShift: operator.lshift,
    prim.RightShift: operator.rshift,
}


@dataclass(frozen=True)
class Step:
    """One binary operation of a term, as the lowering has to write it.

    ``left`` and ``right`` are the dtypes to convert the two operands into, or
    ``None`` to leave one as it is; a literal is written in that dtype, and
    anything else is cast. ``native`` and ``compiled`` are the types of the
    result, natively and as the lowered operation computes it.

    ``result`` is the dtype to convert the operation's result into, or
    ``None``: numpy computes it in that integer type, and C in a wider one or
    one of another sign, ``int`` for ``int8 * int8`` (#122). ``sign`` is, for
    a comparison of integers that C would compute in an unsigned type, the
    position (``0`` or ``1``) of the operand that may be negative, which the
    lowering compares with zero first; ``None`` otherwise.
    """

    left: np.dtype | None
    right: np.dtype | None
    native: Native
    compiled: np.dtype | None
    result: np.dtype | None = None
    sign: int | None = None

    @property
    def converts(self) -> bool:
        """Whether the step converts an operand or the result, or compares a sign."""
        return (
            self.left is not None
            or self.right is not None
            or self.result is not None
            or self.sign is not None
        )


def _kind(value: Any) -> tuple[np.dtype, bool] | None:
    """The dtype of a value and whether it is weak (a Python number)."""
    if isinstance(value, np.generic):
        return value.dtype, False
    if isinstance(value, bool):
        return np.dtype(np.bool_), True
    if isinstance(value, int):
        return np.dtype(np.int64), True
    if isinstance(value, float):
        return np.dtype(np.float64), True
    if isinstance(value, complex):
        return np.dtype(np.complex128), True
    return None


def _sample(dtype: np.dtype, weak: bool) -> Any:
    """A value of a type: one, as a Python number when weak."""
    if weak:
        return {"b": True, "i": 1, "u": 1, "f": 1.0, "c": 1.0 + 0j}[dtype.kind]
    return dtype.type(1)


def _apply(function: Callable[..., Any], *operands: Native) -> Native:
    """The native types of ``function`` of operands of the given native types.

    Every combination of the operands' samples is evaluated, with numpy's
    warnings off, since only the type of the result is wanted. Any evaluation
    that raises makes the whole result unknown: the native run would raise
    too, on some input.
    """
    if any(samples is None for samples in operands):
        return None
    combinations: list[tuple[Any, ...]] = [()]
    for samples in operands:
        assert samples is not None
        combinations = [(*done, sample) for done in combinations for sample in samples]
    out: list[Any] = []
    kinds: list[tuple[np.dtype, bool]] = []
    with warnings.catch_warnings(), np.errstate(all="ignore"):
        warnings.simplefilter("ignore")
        for arguments in combinations:
            try:
                value = function(*arguments)
            except Exception:  # noqa: BLE001 - numpy refuses the operation
                return None
            kind = _kind(value)
            if kind is None:
                return None
            if kind not in kinds:
                kinds.append(kind)
                out.append(_sample(*kind))
    return tuple(out)


def _operands(expr: Any) -> tuple[Any, ...]:
    """The operands of an arithmetic node, in the order both runs take them."""
    if isinstance(expr, prim.Sum | prim.Product | prim.BitwiseXor):
        return tuple(expr.children)
    if isinstance(expr, prim.Power):
        return (expr.base, expr.exponent)
    if isinstance(expr, prim.LeftShift | prim.RightShift):
        return (expr.shiftee, expr.shift)
    return (expr.numerator, expr.denominator)


def _one_dtype(native: Native) -> np.dtype | None:
    """The dtype of a native type, when every sample of it has the same one."""
    if not native:
        return None
    dtypes = {_kind(sample)[0] for sample in native}  # type: ignore[index]
    return dtypes.pop() if len(dtypes) == 1 else None


def _precision(dtype: np.dtype) -> int:
    """The width of a floating dtype's real part in bytes, ``0`` for any other."""
    if dtype.kind == "f":
        return dtype.itemsize
    if dtype.kind == "c":
        return dtype.itemsize // 2
    return 0


#: The magnitude from which the integer literals of a sum are not index
#: arithmetic: beside a loop variable they can leave 32 bits, ``i + 2**31 - 1``
#: at ``i = 1``, and so can several smaller ones, ``i + 2**29 + ... + 2**29``.
_INDEX_LITERAL = 2**30


def _integer_literal(expr: Any) -> bool:
    """Whether ``expr`` is an integer literal, a truth value not counted."""
    return isinstance(expr, int | np.integer) and not isinstance(expr, bool | np.bool_)


def _literal_total(expr: Any) -> int:
    """The magnitude of the integer literals of a sum, the sums in it included.

    lanky builds ``i + a + b`` as ``(i + a) + b``, so the literals of a sum
    written in one line are spread over the sums inside it.
    """
    if _integer_literal(expr):
        return abs(int(expr))
    if isinstance(expr, prim.Sum):
        return sum(_literal_total(child) for child in expr.children)
    return 0


def _large_literals(expr: Any) -> bool:
    """Whether the integer literals of a sum total :data:`_INDEX_LITERAL` or more."""
    return _literal_total(expr) >= _INDEX_LITERAL


def _negation(expr: Any) -> bool:
    """Whether a product is a negation, ``-i``, which pymbolic builds as ``-1 * i``.

    It is planned as a sum is (:func:`_plan`): ``n - 1 - i`` is index
    arithmetic, and a negated loop variable stays inside 32 bits.
    """
    return (
        isinstance(expr, prim.Product)
        and len(expr.children) == 2
        and any(_integer_literal(child) and child == -1 for child in expr.children)
    )


def _weak_integers(*natives: Native) -> bool:
    """Whether every sample of every native type is a Python int (or bool).

    Loop variables, sizes, reduction binders and integer literals are, and
    arithmetic of them alone is index arithmetic (see :func:`_plan`).
    """
    return all(
        native is not None
        and all(
            isinstance(sample, int) and not isinstance(sample, np.generic)
            for sample in native
        )
        for native in natives
    )


def _c_integer(dtype: np.dtype) -> np.dtype:
    """C's integer promotion: a truth value or an integer narrower than ``int``.

    Either is computed as an ``int``, which holds every value of it.
    """
    if dtype.kind in "biu" and dtype.itemsize < 4:
        return np.dtype(np.int32)
    return dtype


def _c_result(left: np.dtype | None, right: np.dtype | None) -> np.dtype | None:
    """The type C computes an operation of two operands of these types in.

    Its usual arithmetic conversions: the widest floating type of the two when
    either is floating, complex when either is, and otherwise an integer
    after the integer promotion (:func:`_c_integer`): the wider of two of one
    sign, the unsigned one beside a signed one of no more bits (``uint32``
    beside ``int32``, where numpy computes ``int64``), and the signed one
    beside an unsigned one of fewer bits, which it holds.
    """
    if left is None or right is None:
        return None
    kinds = {left.kind, right.kind}
    width = max(_precision(left), _precision(right))
    if "c" in kinds:
        return np.dtype(np.complex64 if width <= 4 else np.complex128)
    if "f" in kinds:
        return np.dtype(np.float32 if width <= 4 else np.float64)
    left, right = _c_integer(left), _c_integer(right)
    if left == right:
        return left
    if (left.kind == "u") == (right.kind == "u"):
        return left if left.itemsize >= right.itemsize else right
    unsigned, signed = (left, right) if left.kind == "u" else (right, left)
    return unsigned if unsigned.itemsize >= signed.itemsize else signed


def _loopy_result(left: np.dtype | None, right: np.dtype | None) -> np.dtype | None:
    """The type loopy infers for an operation of two operands of these types.

    loopy promotes as numpy promotes two arrays, but for a ``float32`` beside
    an ``int32``, which it keeps in single precision (``combine`` in
    ``loopy.type_inference``). It is the type of the function a ``//``, a
    ``%``, a ``<<``, a ``>>`` or an integer ``**`` is computed by
    (:mod:`loopty.operations`, loopy's ``loopy_pow``), and the type loopy
    takes the result of any operation to be, which is what it casts an operand
    of such a function by and writes a literal beside it in.
    """
    if left is None or right is None:
        return None
    if {left, right} == {np.dtype(np.int32), np.dtype(np.float32)}:
        return np.dtype(np.float32)
    return np.promote_types(left, right)


#: The integer literals loopy types, as ``int32`` or ``int64``; any other is
#: refused by its type inference ("integer constant too large", #140).
_LOOPY_INTEGERS = (-(2**63), 2**63)


def _untyped(literal: Any) -> bool:
    """Whether ``literal`` is a Python int loopy cannot type, past 64 bits."""
    return (
        isinstance(literal, int)
        and not isinstance(literal, bool)
        and not _LOOPY_INTEGERS[0] <= literal < _LOOPY_INTEGERS[1]
    )


def _literal_of(expr: Any) -> Any:
    """``expr`` when it is an integer literal (a truth value is not), else ``None``."""
    return expr if _integer_literal(expr) else None


def _holds(dtype: np.dtype, value: int) -> bool:
    """Whether the numpy dtype ``dtype`` holds the integer ``value`` exactly."""
    if dtype.kind in "iu":
        info = np.iinfo(dtype)
        return int(info.min) <= value <= int(info.max)
    if dtype.kind in "fc":
        return True
    return False


def _strong_integer(native: Native) -> bool:
    """Whether every sample of a native type is a numpy integer (not a bool)."""
    return bool(native) and all(
        isinstance(sample, np.integer) for sample in native
    )


def _maybe_negative(literal: Any, compiled: np.dtype | None) -> bool:
    """Whether an integer operand may hold a negative value.

    ``literal`` is the operand when it is an integer literal, which says, and
    ``None`` otherwise; anything else may when its compiled type is signed: an
    element or a scalar of a signed type, a loop variable, or arithmetic of
    them.
    """
    if literal is not None:
        return int(literal) < 0
    return compiled is not None and compiled.kind == "i"


def _plan(
    left: tuple[Native, np.dtype | None],
    right: tuple[Native, np.dtype | None],
    common: Native,
    native: Native,
    kind: str,
    helper: bool = False,
    literals: tuple[Any, Any] = (None, None),
) -> Step:
    """Which operands to convert so that C computes in numpy's type ``common``.

    ``common`` is the type numpy computes the operation in, and ``native`` the
    type of its result, which differ for a comparison. ``kind`` is
    ``"comparison"``, ``"power"``, ``"sum"``, ``"growing"`` (a product or a
    left shift, which can leave the range of its operands) or ``"bounded"``
    (a quotient, a floor division, a remainder, an ``^`` or a right shift,
    which cannot). ``helper`` says that the lowering writes the operation as
    a call of a function, computed in the type loopy infers for it
    (:func:`_loopy_result`), and not as C's operator, computed in C's type
    (:func:`_c_result`). ``literals`` holds each operand that is an integer
    literal, and ``None`` for one that is not.

    A comparison is planned by :func:`_compared`, and an operation with an
    integer literal loopy cannot type by :func:`_beyond`.

    Where ``common`` is one floating dtype and C would compute in another,
    each floating operand of another precision is converted to it, and if C
    would still compute in another type (two integers), the left one is.

    A power with a floating result is computed by numpy with the C library's
    ``pow``, its exponent converted too, where loopy computes an integer
    exponent by repeated multiplication, which rounds at every step and
    differs from ``pow`` in the last bit. So both of its operands are
    converted, which makes loopy call ``pow`` (``powf`` in single precision).

    Where ``common`` is one integer dtype, :func:`_widened` says what to
    convert.
    """
    (_, lc), (_, rc) = left, right
    if kind == "comparison":
        return _compared(left, right, common, native, literals)
    result_of = _loopy_result if helper else _c_result
    target = _one_dtype(common)
    big = tuple(_untyped(literal) for literal in literals)
    if any(big):
        return _beyond(left, right, native, target, literals, big, result_of)
    compiled = result_of(lc, rc)
    if target is None or compiled is None:
        return Step(None, None, native, compiled)
    if target.kind in "iu":
        return _widened(left, right, native, compiled, target, kind, helper, literals)
    if target.kind not in "fc":
        return Step(None, None, native, compiled)
    if kind == "power":
        if target.kind != "f":
            return Step(None, None, native, compiled)
        to_left = target if lc != target else None
        to_right = target if rc != target else None
        result = target if to_left or to_right else compiled
        return Step(to_left, to_right, native, result)
    if compiled == target:
        return Step(None, None, native, compiled)
    assert lc is not None and rc is not None
    precision = _precision(target)
    to_left = target if lc.kind in "fc" and _precision(lc) != precision else None
    to_right = target if rc.kind in "fc" and _precision(rc) != precision else None
    after = result_of(to_left or lc, to_right or rc)
    if after != target:
        to_left = target
    return Step(to_left, to_right, native, target)


def _beyond(
    left: tuple[Native, np.dtype | None],
    right: tuple[Native, np.dtype | None],
    native: Native,
    target: np.dtype | None,
    literals: tuple[Any, Any],
    big: tuple[bool, ...],
    result_of: Callable[[Any, Any], np.dtype | None],
) -> Step:
    """The plan for an operation with an integer literal loopy cannot type (#140).

    loopy types an integer literal as ``int32`` or ``int64`` and refuses any
    other ("integer constant too large"). Where numpy computes the operation
    in a type that holds the literal, a real (``x[i] * 2**70``) or a
    ``uint64`` (``u[i] + 2**63``), the literal is written in that type, and
    the other operand is converted too where C would compute in another.
    Anywhere else (``i + 2**64``, which Python computes exactly) the literal
    is left, its type unknown, and the trace refuses it, naming a real.
    """
    if target is None or not all(
        _holds(target, int(literal))
        for literal, beyond in zip(literals, big, strict=True)
        if beyond
    ):
        return Step(None, None, native, None)
    (_, lc), (_, rc) = left, right
    to_left = target if big[0] else None
    to_right = target if big[1] else None
    if result_of(to_left or lc, to_right or rc) != target:
        to_left = target if lc != target else to_left
        to_right = target if rc != target else to_right
    return Step(to_left, to_right, native, target)


def _widened(
    left: tuple[Native, np.dtype | None],
    right: tuple[Native, np.dtype | None],
    native: Native,
    compiled: np.dtype,
    target: np.dtype,
    kind: str,
    helper: bool,
    literals: tuple[Any, Any],
) -> Step:
    """The plan for an operation numpy computes in the integer dtype ``target``.

    Where the compiled type is no integer at all, every operand of another
    type is converted, whatever the operation: loopy types an ``np.uint64``
    beside a signed integer as numpy types two arrays, in double, where numpy
    keeps a Python int or a loop variable beside it weak, and computes ``u[i]
    % 3`` in ``uint64``.

    Where numpy computes in a numpy integer type (:func:`_strong`), the
    compiled run computes in it too, by :func:`_strong`.

    Otherwise, where ``target`` is wider than the integer C would compute in,
    which is 64 bits against a ``Fin[m]`` element's 32 or a loop variable's,
    and the operation can leave the range of its operands, the narrower
    operands are converted to it, so that the compiled run computes integer
    arithmetic in 64 bits as numpy does (#101); of a power, the base. A
    ``"bounded"`` operation is not converted, since its result is inside 32
    bits when its operands are, and neither is a sum of Python ints alone
    (loop variables, sizes and literals): that is index arithmetic, which
    loopy computes in 32 bits as it does every loop bound and subscript, and
    a sum of a few of them stays inside 32 bits while the sizes do; the
    caller plans a negation, ``-1 * i``, as a sum too. A product, a power or a
    left shift of them does not, and is converted, and so is a sum whose
    literals total ``2**30`` or more, which the caller plans as
    ``"growing"``: ``i + 2**31 - 1`` left 32 bits at ``i = 1``.
    """
    (left_native, lc), (right_native, rc) = left, right
    if compiled.kind not in "biu":
        to_left = target if lc != target else None
        to_right = target if rc != target else None
        return Step(to_left, to_right, native, target)
    assert lc is not None and rc is not None
    if _strong_integer(native):
        step = _strong(left, right, native, compiled, target, kind, helper, literals)
        if step is not None:
            return step
    narrower = compiled.itemsize < target.itemsize
    if not narrower or kind == "bounded":
        return Step(None, None, native, compiled)
    if kind == "sum" and _weak_integers(left_native, right_native):
        return Step(None, None, native, compiled)
    to_left = target if lc != target else None
    if kind == "power":
        return Step(to_left, None, native, target)
    to_right = None
    if _c_result(to_left or lc, rc) != target:
        to_right = target
    return Step(to_left, to_right, native, target)


def _strong(
    left: tuple[Native, np.dtype],
    right: tuple[Native, np.dtype],
    native: Native,
    compiled: np.dtype,
    target: np.dtype,
    kind: str,
    helper: bool,
    literals: tuple[Any, Any],
) -> Step | None:
    """The plan for an operation numpy computes in the numpy integer ``target``.

    ``None`` where :func:`_widened`'s rules for a wider ``target`` apply: the
    compiled run computes in an integer of ``target``'s sign and no more bits
    than it, as C computes it and loopy types it. Otherwise numpy's type is
    had one of two ways (#122):

    * an operand that C would take round, a signed one beside an unsigned
      one of as many bits (``uint32 + int32``, which numpy computes in
      ``int64``), and an operation loopy types otherwise than C computes it
      (``u[i] + i`` of a ``uint32`` ``u``, ``int64`` to loopy) or computes in
      a wider type (``u[i] << 3``, by loopy's ``int64`` function), have the
      operands converted into ``target``, which holds them, ``target`` being
      at least 32 bits wide;
    * where that does not do, the result is converted into ``target``
      (:attr:`Step.result`), which wraps round as numpy's arithmetic in it
      does. C computes an integer narrower than ``int`` as an ``int``, so
      ``a[i] * a[i]`` of an ``np.int8`` ``a`` is ``(int8_t) (a[i] * a[i])``,
      and a negative literal beside an unsigned integer has no conversion
      into it, so ``u[i] - 1``, which pymbolic builds as ``u[i] + -1``, is
      ``(uint32_t) (u[i] + -1)``. A power is converted so too, its exponent
      left to loopy.
    """
    (_, lc), (_, rc) = left, right
    loopy = compiled if helper else _loopy_result(lc, rc)
    if compiled == target and loopy == target:
        return None
    signed_round = (
        not helper
        and compiled.kind == "u"
        and any(
            dtype.kind == "i" and _strong_integer(operand)
            for operand, dtype in (left, right)
        )
    )
    if compiled.kind == target.kind and compiled.itemsize <= target.itemsize:
        if loopy == compiled and not signed_round:
            return None
    narrow = target.itemsize < 4
    if narrow or kind == "power":
        return Step(None, None, native, target, result=target)
    convertible = all(
        literal is None or _holds(target, int(literal)) for literal in literals
    )
    if not convertible:
        return Step(None, None, native, target, result=target)
    to_left = target if lc != target else None
    to_right = target if rc != target else None
    return Step(to_left, to_right, native, target)


def _compared(
    left: tuple[Native, np.dtype | None],
    right: tuple[Native, np.dtype | None],
    common: Native,
    native: Native,
    literals: tuple[Any, Any],
) -> Step:
    """The plan for a comparison, which numpy computes exactly on integers.

    numpy compares two integers of any types by their values, and an integer
    with a Python int outside its type's range too (``u[i] > -1`` of a
    ``uint64`` is true, NEP 50). C compares two integers exactly but where its
    usual arithmetic conversions take a negative value round into an unsigned
    type: ``u[i] < k[i]`` of a ``uint32`` ``u`` and an ``int32`` ``k``, and
    ``u[i] == -1``. There the operand that may be negative is compared with
    zero first (:attr:`Step.sign`), and C compares the two only where it is
    not, which it does exactly. An integer literal loopy cannot type is
    written as a ``uint64`` where one holds it and the other operand is an
    integer, and as a double elsewhere, beyond which every integer of 64 bits
    compares alike.

    A comparison with a real or a complex operand is computed in numpy's
    ``common`` type, as :func:`_plan` converts any operation.
    """
    (_, lc), (_, rc) = left, right
    big = [_untyped(literal) for literal in literals]
    types = [lc, rc]
    to: list[np.dtype | None] = [None, None]
    integral = [dtype is not None and dtype.kind in "biu" for dtype in types]
    for k in range(2):
        if not big[k]:
            continue
        other = types[1 - k]
        wide = np.dtype(np.uint64)
        value = int(literals[k])
        if other is not None and other.kind in "biu" and _holds(wide, value):
            to[k] = wide
        else:
            to[k] = np.dtype(np.float64)
        types[k] = to[k]
        integral[k] = to[k].kind in "biu"
    if all(integral) and types[0] is not None and types[1] is not None:
        compiled = _c_result(types[0], types[1])
        sign = None
        if compiled is not None and compiled.kind == "u":
            for k in range(2):
                literal = literals[k] if not big[k] else None
                if types[k].kind == "i" and _maybe_negative(literal, types[k]):
                    sign = k
                    break
        return Step(to[0], to[1], native, np.dtype(np.bool_), sign=sign)
    if any(big):
        return Step(to[0], to[1], native, np.dtype(np.bool_))
    step = _plan(left, right, common, native, "bounded")
    return dataclasses.replace(step, compiled=np.dtype(np.bool_))


class Promotion:
    """The native and compiled types of a term's expressions, and the plan.

    ``types(expr)`` is the pair of the native type (:data:`Native`) and the
    dtype the lowered code computes ``expr`` in, given that every operation
    inside it is lowered as :meth:`steps` says. ``steps(expr)`` is the plan for
    one arithmetic node or comparison: one :class:`Step` for a binary one, and
    one per operand after the first for a sum or a product of several, which
    numpy and C both evaluate from the left.

    The types are memoized by node, since the lowering asks for the steps of
    every node it rebuilds and for the types of each one's operands.
    """

    def __init__(self, term: Term) -> None:
        self.arrays: dict[str, ArrType] = term.array_types
        #: The term, whose checked points widen what they read compiled
        #: (:func:`loopty.contract.array_storage`); ``None`` before there is one.
        self.term: Term | None = term
        self.scalars: dict[str, Any] = {
            name: sort for name, sort in term.params if not isinstance(sort, ArrType)
        }
        self._types: dict[int, tuple[Native, np.dtype | None]] = {}
        self._steps: dict[int, tuple[Step, ...]] = {}
        #: Every node a memo is keyed by, kept alive so that its id stays its.
        self._seen: list[Any] = []

    @classmethod
    def of_sorts(cls, sorts: Mapping[str, Any]) -> Promotion:
        """The promotion of expressions over named sorts, before there is a term.

        ``sorts`` maps an array's name to its :class:`~loopty.term.ArrType`
        and a scalar's to its sort, as the tracer holds them; the trace asks
        it of an expression it is about to record (:mod:`loopty.trace`).
        """
        promotion = cls.__new__(cls)
        # A term being traced has no checked points yet: those come of
        # composing calls, so each array is stored as its sort is.
        promotion.term = None
        promotion.arrays = {
            name: typ for name, typ in sorts.items() if isinstance(typ, ArrType)
        }
        promotion.scalars = {
            name: sort for name, sort in sorts.items() if not isinstance(sort, ArrType)
        }
        promotion._types = {}
        promotion._steps = {}
        promotion._seen = []
        return promotion

    def steps(self, expr: Any) -> tuple[Step, ...]:
        """The plan for one node: empty for a node that is no operation."""
        self.types(expr)
        return self._steps.get(id(expr), ())

    def types(self, expr: Any) -> tuple[Native, np.dtype | None]:
        """The native type of ``expr`` and the dtype it is computed in compiled."""
        if isinstance(expr, bool | int | float | complex | np.generic):
            return self._literal(expr)
        key = id(expr)
        if key not in self._types:
            self._seen.append(expr)
            self._types[key] = self._compute(expr)
        return self._types[key]

    # {{{ leaves

    @staticmethod
    def _literal(value: Any) -> tuple[Native, np.dtype | None]:
        """A literal is its own sample; compiled, loopy types it by its value.

        An integer as ``int32`` or ``int64``, and one past 64 bits not at all
        (#140): its compiled type is unknown until a step writes it in one.
        """
        if isinstance(value, np.generic):
            return (value,), value.dtype
        if isinstance(value, bool):
            return (value,), np.dtype(np.bool_)
        if isinstance(value, int):
            if _untyped(value):
                return (value,), None
            fits = -(2**31) <= value < 2**31
            return (value,), np.dtype(np.int32 if fits else np.int64)
        if isinstance(value, float):
            return (value,), np.dtype(np.float64)
        return (value,), np.dtype(np.complex128)

    def _element(self, name: str) -> tuple[Native, np.dtype | None]:
        """An element of an array, read natively as its sort is stored."""
        typ = self.arrays.get(name)
        if typ is None:
            return None, None
        stored = native_storage(typ.dtype)
        native = None if stored is None else (_sample(stored, False),)
        return native, array_storage(self.term, name, typ.dtype)

    def _variable(self, name: str) -> tuple[Native, np.dtype | None]:
        """A scalar argument by its sort; a loop variable or a size is an int.

        :func:`loopty.contract.native_scalar` converts a scalar into a numpy
        scalar of its sort's native dtype, however the caller passed it, so
        it is strong (#102).
        """
        if name in self.arrays:
            return self._element(name)
        if name not in self.scalars:
            return (1,), np.dtype(np.int32)
        sort = self.scalars[name]
        compiled = compiled_storage(sort)
        stored = native_storage(sort)
        if stored is None:
            return None, compiled
        return (_sample(stored, False),), compiled

    # }}}

    def _compute(self, expr: Any) -> tuple[Native, np.dtype | None]:
        if isinstance(expr, Access):
            return self._element(expr.array)
        if isinstance(expr, prim.Subscript):
            if isinstance(expr.aggregate, prim.Variable):
                return self._element(expr.aggregate.name)
            return None, None
        if isinstance(expr, prim.Variable):
            return self._variable(expr.name)
        if isinstance(expr, Reduction):
            return self._summed(expr, expr.body)
        from lanky.terms import Abs
        from lanky.terms import Sum as LankySum

        if isinstance(expr, LankySum):
            return self._summed(expr, expr.body)
        if isinstance(expr, Abs):
            return self._absolute(expr, expr.operand, abs)
        for kind, function in _ARITHMETIC.items():
            if isinstance(expr, kind):
                return self._arithmetic(expr, function)
        if isinstance(expr, prim.Comparison):
            return self._comparison(expr)
        if isinstance(expr, prim.LogicalAnd | prim.LogicalOr):
            function = (
                operator.and_ if isinstance(expr, prim.LogicalAnd) else operator.or_
            )
            natives = [self.types(child)[0] for child in expr.children]
            native = natives[0]
            for other in natives[1:]:
                native = _apply(function, native, other)
            return native, np.dtype(np.bool_)
        if isinstance(expr, prim.LogicalNot):
            native = _apply(operator.invert, self.types(expr.child)[0])
            return native, np.dtype(np.bool_)
        if isinstance(expr, prim.If):
            then, other = self.types(expr.then), self.types(expr.else_)
            if then[0] is None or other[0] is None:
                return None, _c_result(then[1], other[1])
            union = list(then[0])
            union += [s for s in other[0] if _kind(s) not in map(_kind, union)]
            return tuple(union), _c_result(then[1], other[1])
        if isinstance(expr, prim.Call):
            return self._call(expr)
        return None, None

    def _summed(self, expr: Any, body: Any) -> tuple[Native, np.dtype | None]:
        """A sum of ``body``: natively ``reduce_sum`` adds the terms to ``0``.

        loopy accumulates in the type of the body. Where numpy adds the terms
        in a wider integer, the body is converted to it, a :class:`Step`
        whose ``right`` is the body: the sum of a ``Bool`` array's elements is
        a count natively, which a byte holds up to 127, and the sum of a
        ``Fin[m]`` array's or of a binder leaves 32 bits as readily.
        """
        body_native, compiled = self.types(body)
        native = _apply(operator.add, (0,), body_native)
        target = _one_dtype(native)
        if (
            target is not None
            and compiled is not None
            and target.kind in "iu"
            and compiled.kind in "biu"
            and compiled.itemsize < target.itemsize
        ):
            self._steps[id(expr)] = (Step(None, target, native, target),)
            return native, target
        return native, compiled

    def _arithmetic(
        self, expr: Any, function: Callable[[Any, Any], Any]
    ) -> tuple[Native, np.dtype | None]:
        operands = _operands(expr)
        if isinstance(expr, prim.Power):
            kind = "power"
        elif isinstance(expr, prim.Sum):
            kind = "growing" if _large_literals(expr) else "sum"
        elif _negation(expr):
            kind = "sum"
        elif isinstance(expr, prim.Product | prim.LeftShift):
            kind = "growing"
        else:
            kind = "bounded"
        helper = isinstance(
            expr,
            prim.FloorDiv | prim.Remainder | prim.LeftShift | prim.RightShift
            | prim.Power,
        )
        accumulated = self.types(operands[0])
        literal = _literal_of(operands[0])
        steps: list[Step] = []
        for operand in operands[1:]:
            right = self.types(operand)
            literals = (literal, _literal_of(operand))
            native = _apply(function, accumulated[0], right[0])
            subtrahend = literals[1]
            if native is None and isinstance(expr, prim.Sum) and subtrahend is not None:
                # pymbolic builds u - 1 as u + -1, which numpy refuses beside
                # an unsigned integer, and the difference, which it computes.
                if subtrahend < 0:
                    native = _apply(operator.sub, accumulated[0], (-subtrahend,))
            step = _plan(accumulated, right, native, native, kind, helper, literals)
            steps.append(step)
            accumulated = (step.native, step.compiled)
            literal = None
        self._steps[id(expr)] = tuple(steps)
        return accumulated

    def _comparison(self, expr: prim.Comparison) -> tuple[Native, np.dtype | None]:
        """A comparison is computed in the type of its operands' sum."""
        left, right = self.types(expr.left), self.types(expr.right)
        common = _apply(operator.add, left[0], right[0])
        native = _apply(operator.eq, left[0], right[0])
        literals = (_literal_of(expr.left), _literal_of(expr.right))
        step = _plan(left, right, common, native, "comparison", literals=literals)
        self._steps[id(expr)] = (step,)
        return native, np.dtype(np.bool_)

    def _absolute(
        self, expr: Any, operand: Any, function: Callable[[Any], Any]
    ) -> tuple[Native, np.dtype | None]:
        """``abs`` of ``operand``, by numpy's ``function``; of an integer, #123.

        The lowering writes ``abs`` of an integer as ``k < 0 ? -1 * k : k``
        in its own type (:meth:`loopty.lower.ExpressionLowerer.map_call`), and
        ``abs`` of a truth value or an unsigned integer as the operand itself,
        as numpy computes them. C computes ``-1 * k`` of an integer narrower
        than ``int`` as an ``int``, where numpy keeps it in its type, in which
        ``abs`` of the smallest value is that value: the result is converted
        back into it (:attr:`Step.result`).
        """
        native, compiled = self.types(operand)
        result = _apply(function, native)
        if compiled is not None and compiled.kind == "c":
            width = _precision(compiled)
            return result, np.dtype(np.float32 if width <= 4 else np.float64)
        if compiled is None or compiled.kind not in "biu":
            return result, compiled
        target = _one_dtype(result)
        if (
            compiled.kind == "i"
            and _strong_integer(result)
            and target is not None
            and _c_integer(target) != target
        ):
            self._steps[id(expr)] = (Step(None, None, result, target, result=target),)
            return result, target
        return result, compiled

    def _call(self, expr: prim.Call) -> tuple[Native, np.dtype | None]:
        """A library function, by the numpy function the interpreter calls."""
        from loopty.interpret import _FUNCTIONS

        function = expr.function
        name = function.name if isinstance(function, prim.Variable) else None
        numpy_function = _FUNCTIONS.get(name) if name is not None else None
        if name == "abs" and len(expr.parameters) == 1:
            return self._absolute(expr, expr.parameters[0], np.abs)
        operands = [self.types(parameter) for parameter in expr.parameters]
        native = (
            None
            if numpy_function is None
            else _apply(numpy_function, *(native for native, _ in operands))
        )
        widths = [
            _precision(compiled)
            for _, compiled in operands
            if compiled is not None and compiled.kind in "fc"
        ]
        if not widths or any(compiled is None for _, compiled in operands):
            return native, None
        return native, np.dtype(np.float32 if max(widths) <= 4 else np.float64)
