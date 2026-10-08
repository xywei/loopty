"""What loopy reads into isl, and what it has to decline.

loopy reads an expression into isl in several places: a subscript for its
bounds check (``loopy.symbolic.get_access_map``) and for code generation
(``simplify_using_aff``), and a guard that names only loop variables, sizes
and scalars for its bounds check (``condition_to_set``). Its reader,
``loopy.symbolic.PwAffEvaluationMapper``, builds an affine expression, and an
expression it declines (with a ``TypeError``, which
``with_aff_conversion_guard`` catches) is one loopy treats as not affine: a
subscript is then generated as it is written, and a guard is not used to
narrow what the bounds check covers. Two of its readings were wrong for what
loopty writes (note 23 in ``docs/loopy-notes.md``):

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
  that it never looked at (#137).

:func:`install` makes the reader decline both: a conversion, and a constant
that is not an integer. A float whose value is an integer of fewer than 53
bits is the integer it is in double, and is still read as one. Declining is
the safe direction of the two: an expression loopy does not read into isl is
generated as written, and a guard it does not read narrows nothing, so its
bounds check covers more points, not fewer. loopty installs it when
:mod:`loopty.lower` is imported, before any kernel is built.

:func:`read_as_affine` asks the reader whether loopy would read an
expression as affine, which the lowering asks of a subscript
(:class:`loopty.lower.ExpressionLowerer`).
"""

from __future__ import annotations

from typing import Any

import islpy as isl
import numpy as np

__all__ = ["install", "read_as_affine"]

#: Below this magnitude a float that is an integer is that integer in double,
#: and the reader takes it as the integer.
_EXACT = 2**53


def _integral(value: Any) -> bool:
    """Whether a constant is an integer the reader may take as one."""
    if isinstance(value, bool | np.bool_ | int | np.integer):
        return True
    if isinstance(value, float | np.floating):
        return bool(np.isfinite(value)) and float(value).is_integer() and (
            abs(float(value)) < _EXACT
        )
    return False


def _decline_conversion(self: Any, expr: Any, *args: Any, **kwargs: Any) -> Any:
    """A conversion is no affine expression: declined, as a call is."""
    raise TypeError(
        f"the conversion in '{expr}' is not read as affine: an expression "
        "computed in another type is generated as it is written"
    )


def install() -> None:
    """Make loopy's reader decline a conversion and a constant not an integer.

    Idempotent. A reading loopy may add of its own is replaced too, since
    declining is always the safe direction (see the module).
    """
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
    PwAffEvaluationMapper._loopty_declines = True


def read_as_affine(expr: Any) -> bool:
    """Whether loopy reads ``expr`` as one affine expression.

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
        guarded_aff_from_expr(space, expr)
    except ExpressionToAffineConversionError:
        return False
    except Exception:  # noqa: BLE001 - what loopy cannot read is not affine
        return False
    return True
