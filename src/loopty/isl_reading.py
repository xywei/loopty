"""What loopy reads into isl, and what it has to decline.

loopy reads an expression into isl in several places: a subscript for its
bounds check (``loopy.symbolic.get_access_map``) and for code generation
(``simplify_using_aff``), and a guard that names only loop variables, sizes
and scalars for its bounds check (``condition_to_set``, over the domain
``loopy.kernel.instruction.get_insn_domain`` builds). Its reader,
``loopy.symbolic.PwAffEvaluationMapper``, builds an affine expression, and an
expression it declines (with a ``TypeError``, which
``with_aff_conversion_guard`` catches) is one loopy treats as not affine: a
subscript is then generated as it is written, and a guard is not used to
narrow what the bounds check covers. Three of its readings were wrong for
what loopty writes (note 23 in ``docs/loopy-notes.md``):

* a conversion (loopy's ``TypeCast``) is no case of the reader's, so it
  raises ``UnsupportedExpressionError``, which nothing catches. A subscript
  computed in 64 bits, ``x[(i * i) % n]``, with ``i`` cast as everywhere else
  (:mod:`loopty.promotion`), failed in loopy's bounds check, and so did a
  substitution that carried a store's conversion into a subscript, ``x[p[i]]``
  (#129, #145);
* a constant is read by ``int()``, so the ``0.5`` of ``when(i * 0.5 < 2)``
  read as ``0``: the guard read as true everywhere, and the bounds check
  refused an access that was in bounds, and ``when(i * 0.5 >= 1)`` read as
  false everywhere, so the check passed an access past the end of the array
  that it never looked at (#137). A float that is an integer was read as the
  integer, though the guard is computed in floating point, which rounds:
  ``when(i * 2.0**52 + 1.0 <= i * 2.0**52)`` read as false everywhere, and
  holds from ``i = 2`` in double, and ``np.float32(1.0) * i <= 2**24`` holds
  at ``i = 2**24 + 1`` in single precision;
* a value argument is a parameter of the domain the guards are read over,
  an integer, whatever its dtype: a ``Real`` scalar ``a`` in ``when((i < a) &
  (i > a - 1))`` was an integer, for which the guard holds nowhere, and the
  guard holds at ``i = 1`` for ``a = 1.5``.

:func:`install` makes loopy decline all three: a conversion, a constant
whose type is not an integer's, and a guard that names a value argument
whose dtype is not an integer's. Declining is the safe direction: an
expression loopy does not read into isl is generated as written, and a guard
it does not read narrows nothing, so its bounds check covers more points,
not fewer. loopty installs it when :mod:`loopty.lower` is imported, before
any kernel is built.

:func:`affine_form` asks the reader for the affine expression loopy would
read an expression as, which the lowering asks of a subscript
(:class:`loopty.lower.ExpressionLowerer`), and :func:`read_as_affine` whether
there is one.
"""

from __future__ import annotations

from typing import Any

import islpy as isl
import numpy as np

__all__ = ["affine_form", "install", "read_as_affine"]


def _integral(value: Any) -> bool:
    """Whether a constant is an integer, the only constant an affine form has.

    A float is not, even one whose value is an integer: an expression with
    one is computed in floating point, which rounds where integers do not.
    """
    return isinstance(value, bool | np.bool_ | int | np.integer)


def _integral_dtype(dtype: Any) -> bool:
    """Whether a value argument of loopy's dtype ``dtype`` holds an integer."""
    numpy_dtype = getattr(dtype, "numpy_dtype", None)
    return numpy_dtype is not None and numpy_dtype.kind in "biu"


def _decline_conversion(self: Any, expr: Any, *args: Any, **kwargs: Any) -> Any:
    """A conversion is no affine expression: declined, as a call is."""
    raise TypeError(
        f"the conversion in '{expr}' is not read as affine: an expression "
        "computed in another type is generated as it is written"
    )


def _insn_domain(insn: Any, kernel: Any) -> isl.Set:
    """loopy's ``get_insn_domain``, with only an integer value argument a parameter.

    The domain of the loops around ``insn``, with every value argument it
    reads that the kernel does not write as a parameter, narrowed by the
    guards loopy can read over it, which is the domain loopy's bounds check
    checks the accesses of ``insn`` over. loopy adds every such argument, an
    integer to isl whatever its dtype; here one whose dtype is not an
    integer's is left out, so a guard that names it names a variable the
    space does not have, which ``condition_to_set`` declines as it declines
    a guard that reads an array, and which narrows nothing.
    """
    from loopy.kernel.data import ValueArg
    from loopy.symbolic import condition_to_set

    domain = kernel.get_inames_domain(insn.within_inames)
    present = set(domain.get_var_names(isl.dim_type.param))
    written = kernel.get_written_variables()
    reads = insn.read_dependency_names()
    for arg in kernel.args:
        if (
            not isinstance(arg, ValueArg)
            or arg.name in written
            or arg.name in present
            or arg.name not in reads
            or not _integral_dtype(arg.dtype)
        ):
            continue
        position = domain.dim(isl.dim_type.param)
        domain = domain.add_dims(isl.dim_type.param, 1)
        domain = domain.set_dim_name(isl.dim_type.param, position, arg.name)
        present.add(arg.name)
    guards = isl.Set.universe(domain.space)
    for predicate in insn.predicates:
        read = condition_to_set(domain.space, predicate)
        if read is not None:
            guards = guards & read
    return domain & guards


def install() -> None:
    """Make loopy decline a conversion, a non-integer constant and a real scalar.

    Idempotent. A reading loopy may add of its own is replaced too, since
    declining is always the safe direction (see the module). loopy's bounds
    check imports ``get_insn_domain`` where it runs, so the replacement is
    the one it calls.
    """
    import loopy.kernel.instruction as instruction
    from loopy.symbolic import PwAffEvaluationMapper

    if getattr(PwAffEvaluationMapper, "_loopty_declines", False):
        return
    read_constant = PwAffEvaluationMapper.map_constant

    def map_constant(self: Any, expr: Any) -> Any:
        if not _integral(expr):
            raise TypeError(
                f"the constant {expr!r} is not an integer, and an affine "
                "expression is made of integers"
            )
        return read_constant(self, expr)

    PwAffEvaluationMapper.map_constant = map_constant
    PwAffEvaluationMapper.map_type_cast = _decline_conversion
    instruction.get_insn_domain = _insn_domain
    PwAffEvaluationMapper._loopty_declines = True


def affine_form(expr: Any) -> isl.Aff | None:
    """The affine expression loopy reads ``expr`` as, or ``None`` if it reads none.

    Over a space of the names ``expr`` uses, as ``simplify_via_aff`` reads
    one. A subscript loopy reads so is simplified to the affine expression
    isl gives back, in loopy's 32-bit index type, whatever it was written as;
    one it does not is generated as it is written.
    """
    from loopy.diagnostic import ExpressionToAffineConversionError
    from loopy.symbolic import get_dependencies, guarded_aff_from_expr

    install()
    try:
        names = sorted(get_dependencies(expr))
        space = isl.Space.create_from_names(isl.DEFAULT_CONTEXT, set=names)
        return guarded_aff_from_expr(space, expr)
    except ExpressionToAffineConversionError:
        return None
    except Exception:  # noqa: BLE001 - what loopy cannot read is not affine
        return None


def read_as_affine(expr: Any) -> bool:
    """Whether loopy reads ``expr`` as one affine expression (:func:`affine_form`)."""
    return affine_form(expr) is not None
