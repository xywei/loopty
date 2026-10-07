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
a product of them is not, since ``i * i`` leaves 32 bits at ``i = 46341``.
The lowering leaves a subscript's arithmetic as it is too, and a guard's that
reads no array (:class:`loopty.lower.ExpressionLowerer`). A result outside 64
bits wraps round compiled and is refused or wraps natively, which is numpy's
limit too.
"""

from __future__ import annotations

import operator
import warnings
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from typing import Any

import numpy as np
import pymbolic.primitives as prim

from loopty.contract import compiled_storage, native_storage
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
    """

    left: np.dtype | None
    right: np.dtype | None
    native: Native
    compiled: np.dtype | None

    @property
    def converts(self) -> bool:
        """Whether the step converts either operand."""
        return self.left is not None or self.right is not None


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


def _c_result(left: np.dtype | None, right: np.dtype | None) -> np.dtype | None:
    """The type C computes an operation of two operands of these types in.

    Its usual arithmetic conversions: the widest floating type of the two when
    either is floating, complex when either is, and an integer otherwise.
    """
    if left is None or right is None:
        return None
    kinds = {left.kind, right.kind}
    width = max(_precision(left), _precision(right))
    if "c" in kinds:
        return np.dtype(np.complex64 if width <= 4 else np.complex128)
    if "f" in kinds:
        return np.dtype(np.float32 if width <= 4 else np.float64)
    return np.promote_types(left, right)


def _plan(
    left: tuple[Native, np.dtype | None],
    right: tuple[Native, np.dtype | None],
    common: Native,
    native: Native,
    kind: str,
) -> Step:
    """Which operands to convert so that C computes in numpy's type ``common``.

    ``common`` is the type numpy computes the operation in, and ``native`` the
    type of its result, which differ for a comparison. ``kind`` is
    ``"comparison"``, ``"power"``, ``"sum"``, ``"growing"`` (a product or a
    left shift, which can leave the range of its operands) or ``"bounded"``
    (a quotient, a floor division, a remainder, an ``^`` or a right shift,
    which cannot).

    Where ``common`` is one floating dtype and C would compute in another,
    each floating operand of another precision is converted to it, and if C
    would still compute in another type (two integers), the left one is.

    A power with a floating result is computed by numpy with the C library's
    ``pow``, its exponent converted too, where loopy computes an integer
    exponent by repeated multiplication, which rounds at every step and
    differs from ``pow`` in the last bit. So both of its operands are
    converted, which makes loopy call ``pow`` (``powf`` in single precision).

    Where ``common`` is one integer dtype wider than the integer C would
    compute in, which is 64 bits against a ``Fin[m]`` element's 32 or a loop
    variable's, and the operation can leave the range of its operands, the
    narrower operands are converted to it, so that the compiled run computes
    integer arithmetic in 64 bits as numpy does (#101); of a power, the base.
    A comparison and a ``"bounded"`` operation are not converted, since their
    result is inside 32 bits when their operands are, and neither is a sum of
    Python ints alone (loop variables, sizes and literals): that is index
    arithmetic, which loopy computes in 32 bits as it does every loop bound
    and subscript, and a sum of a few of them stays inside 32 bits while the
    sizes do. A product, a power or a left shift of them does not, and is
    converted.
    """
    (_, lc), (_, rc) = left, right
    compiled = _c_result(lc, rc)
    target = _one_dtype(common)
    if target is None or compiled is None:
        return Step(None, None, native, compiled)
    if target.kind in "iu":
        return _widened(left, right, native, compiled, target, kind)
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
    after = _c_result(to_left or lc, to_right or rc)
    if after != target:
        to_left = target
    return Step(to_left, to_right, native, target)


def _widened(
    left: tuple[Native, np.dtype | None],
    right: tuple[Native, np.dtype | None],
    native: Native,
    compiled: np.dtype,
    target: np.dtype,
    kind: str,
) -> Step:
    """The plan for an operation numpy computes in the integer dtype ``target``.

    See :func:`_plan`: the narrower operands are converted when C computes in
    fewer bits, but for a comparison, a bounded operation and a sum of Python
    ints alone.
    """
    (left_native, lc), (right_native, rc) = left, right
    narrower = compiled.kind in "biu" and compiled.itemsize < target.itemsize
    if not narrower or kind in ("comparison", "bounded"):
        return Step(None, None, native, compiled)
    if kind == "sum" and _weak_integers(left_native, right_native):
        return Step(None, None, native, compiled)
    assert lc is not None and rc is not None
    to_left = target if lc != target else None
    if kind == "power":
        return Step(to_left, None, native, target)
    to_right = None
    if _c_result(to_left or lc, rc) != target:
        to_right = target
    return Step(to_left, to_right, native, target)


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
        """A literal is its own sample; compiled, loopy types it by its value."""
        if isinstance(value, np.generic):
            return (value,), value.dtype
        if isinstance(value, bool):
            return (value,), np.dtype(np.bool_)
        if isinstance(value, int):
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
        return native, compiled_storage(typ.dtype)

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
            native, compiled = self.types(expr.operand)
            return _apply(abs, native), compiled
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
            kind = "sum"
        elif isinstance(expr, prim.Product | prim.LeftShift):
            kind = "growing"
        else:
            kind = "bounded"
        accumulated = self.types(operands[0])
        steps: list[Step] = []
        for operand in operands[1:]:
            right = self.types(operand)
            native = _apply(function, accumulated[0], right[0])
            step = _plan(accumulated, right, native, native, kind)
            steps.append(step)
            accumulated = (step.native, step.compiled)
        self._steps[id(expr)] = tuple(steps)
        return accumulated

    def _comparison(self, expr: prim.Comparison) -> tuple[Native, np.dtype | None]:
        """A comparison is computed in the type of its operands' sum."""
        left, right = self.types(expr.left), self.types(expr.right)
        common = _apply(operator.add, left[0], right[0])
        native = _apply(operator.eq, left[0], right[0])
        step = _plan(left, right, common, native, "comparison")
        self._steps[id(expr)] = (step,)
        return native, np.dtype(np.bool_)

    def _call(self, expr: prim.Call) -> tuple[Native, np.dtype | None]:
        """A library function, by the numpy function the interpreter calls."""
        from loopty.interpret import _FUNCTIONS

        function = expr.function
        name = function.name if isinstance(function, prim.Variable) else None
        numpy_function = _FUNCTIONS.get(name) if name is not None else None
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
