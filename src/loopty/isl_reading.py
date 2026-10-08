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

loopty's readings decline all three: a conversion, a constant whose type is
not an integer's, and a guard that names a value argument whose dtype is not
an integer's. Declining is the safe direction: an expression loopy does not
read into isl is generated as written, and a guard it does not read narrows
nothing, so its bounds check covers more points, not fewer.

**Where they apply.** Inside :func:`declining`, and only in the thread that
entered it. Nothing is installed when loopty, or any module of it, is
imported, and nothing stays installed once the last :func:`declining` in the
process exits: loopy's own ``PwAffEvaluationMapper.map_constant``, its
``map_type_cast`` (it has none) and ``loopy.kernel.instruction.get_insn_domain``
are put back. Another user of loopy in the process (sumpy, pytential, a
Volumential kernel) reads its kernels with loopy's own readings before, after,
and in another thread while loopty is inside one, so loopy checks and
generates them as it does without loopty. loopty enters it at these
chokepoints, which every path of its into loopy goes through:

* :func:`loopty.lower.lower_generic`, which builds every kernel loopty makes
  (``lp.make_kernel``, ``lp.assume``, ``lp.prioritize_loops``) and asks
  :func:`affine_form` of each subscript, for :func:`loopty.lower.lower`, for
  a schedule, and for the executor given a kernel or a term;
* :class:`loopty.schedule.Schedule`: building one, and every public method
  (:func:`declining_methods`), each step and ``retarget`` among them, which
  transform the kernel with loopy and read some of its expressions with
  loopy's reader; :attr:`~loopty.schedule.Schedule.kernel` and
  :attr:`~loopty.schedule.Schedule.buildable` are made there;
* :meth:`loopty.executor.LoopyExecutor.run`, in which loopy preprocesses the
  kernel, checks its bounds, generates and compiles its code and runs it, and
  through which :meth:`~loopty.executor.LoopyExecutor.differential` and
  ``loopty run`` run it; and :func:`loopty.executor.emit_code`, which
  ``LoopyExecutor.emit_code`` and ``loopty run --emit-code`` call;
* :func:`affine_form` and :func:`read_as_affine`, which enter it themselves.

The native run and the trace-faithful one (:mod:`loopty.interpret`,
:mod:`loopty.faithful`) do not call loopy.

A kernel loopty builds is written for these readings: its subscripts may hold
a cast, which loopy's own reader raises on, and its guards are checked with
them. And loopy's code cache, whose key does not say which readings made an
entry, would serve code loopy made outside to a run inside, without the
bounds check that refuses it. So loopty's targets refuse to be preprocessed or
to have code generated outside :func:`declining`
(:class:`ReadingsInactive`, :func:`require_active`). A caller who hands
loopty's kernel to loopy directly, ``lp.generate_code_v2(schedule.kernel)``
or ``schedule.kernel.executor()``, does it inside ``with declining():``.

:func:`affine_form` asks the reader for the affine expression loopy would
read an expression as, which the lowering asks of a subscript
(:class:`loopty.lower.ExpressionLowerer`), and :func:`read_as_affine` whether
there is one.
"""

from __future__ import annotations

import inspect
import threading
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from typing import Any

import islpy as isl
import numpy as np

__all__ = [
    "ReadingsInactive",
    "active",
    "affine_form",
    "declining",
    "declining_methods",
    "read_as_affine",
    "require_active",
]

#: What a name of loopy's had in its class's or module's own namespace when
#: it was absent from it, as ``map_type_cast`` is from the mapper's.
_ABSENT = object()

#: Guards :data:`_OPEN` and the installing and restoring of the readings.
_LOCK = threading.Lock()

#: How many :func:`declining` contexts are open, in every thread. The
#: readings are installed while it is not zero.
_OPEN = 0

#: Each name the readings replace, as ``(owner, name)``, with what loopy had
#: there in the owner's own namespace (or :data:`_ABSENT`), to put back.
_SAVED: dict[tuple[Any, str], Any] = {}

#: loopy's own reading of each name, as attribute lookup on the owner found
#: it when the readings were installed: what a thread outside
#: :func:`declining` calls. Kept after the readings are restored, since a
#: thread may be inside one of them as they are.
_OWN: dict[str, Any] = {}

#: How deep the current thread is in :func:`declining`.
_DEPTH = threading.local()


class ReadingsInactive(RuntimeError):
    """loopy was asked to check or generate loopty's kernel outside the readings.

    Raised by :func:`require_active`, which loopty's targets call.
    """


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


def active() -> bool:
    """Whether loopy reads with loopty's readings in this thread.

    That is, whether the thread is inside :func:`declining`.
    """
    return getattr(_DEPTH, "value", 0) > 0


def require_active(what: str) -> None:
    """Refuse, unless this thread is inside :func:`declining`.

    loopty's targets call it where loopy preprocesses a kernel and before it
    generates code, so that a kernel loopty built is never checked or
    generated with loopy's own readings (see the module). ``what`` names the
    kernel.
    """
    if active():
        return
    raise ReadingsInactive(
        f"loopy was asked to check or generate code for {what}, a kernel on "
        "one of loopty's targets, outside loopty.isl_reading.declining(). "
        "Its subscripts and guards are written for loopty's readings of them "
        "(note 23 in docs/loopy-notes.md), which loopy has only inside it: "
        "use loopty.executor.emit_code or LoopyExecutor, which enter it, or "
        "call loopy inside 'with declining():'"
    )


# {{{ the readings


def _map_constant(self: Any, expr: Any) -> Any:
    """loopy's reading of a constant, which declines one not an integer inside."""
    if active() and not _integral(expr):
        raise TypeError(
            f"the constant {expr!r} is not an integer, and an affine "
            "expression is made of integers"
        )
    return _OWN["map_constant"](self, expr)


def _map_type_cast(self: Any, expr: Any, *args: Any, **kwargs: Any) -> Any:
    """A conversion is no affine expression inside: declined, as a call is.

    Outside, loopy's own: the mapper has no case for a conversion, so the
    one its dispatch falls back to.
    """
    if active():
        raise TypeError(
            f"the conversion in '{expr}' is not read as affine: an expression "
            "computed in another type is generated as it is written"
        )
    own = _OWN["map_type_cast"]
    if own is not None:
        return own(self, expr, *args, **kwargs)
    return self.rec_fallback(expr, *args, **kwargs)


def _get_insn_domain(insn: Any, kernel: Any) -> isl.Set:
    """loopy's ``get_insn_domain``; inside, :func:`_insn_domain`."""
    if active():
        return _insn_domain(insn, kernel)
    return _OWN["get_insn_domain"](insn, kernel)


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


def _readings() -> tuple[tuple[Any, str, Any], ...]:
    """Each place a reading goes, as ``(owner, name, reading)``.

    loopy's bounds check imports ``get_insn_domain`` where it runs, so the
    module's attribute is the one it calls.
    """
    import loopy.kernel.instruction as instruction
    from loopy.symbolic import PwAffEvaluationMapper

    return (
        (PwAffEvaluationMapper, "map_constant", _map_constant),
        (PwAffEvaluationMapper, "map_type_cast", _map_type_cast),
        (instruction, "get_insn_domain", _get_insn_domain),
    )


def _install() -> None:
    """Put the readings in loopy's place, keeping loopy's own. Under :data:`_LOCK`."""
    for owner, name, reading in _readings():
        _SAVED[owner, name] = vars(owner).get(name, _ABSENT)
        own = getattr(owner, name, None)
        if own is not reading:
            _OWN[name] = own
        setattr(owner, name, reading)


def _restore() -> None:
    """Put loopy's own readings back. Under :data:`_LOCK`.

    A name something else replaced while the readings were in is left as it
    replaced it.
    """
    for owner, name, reading in _readings():
        saved = _SAVED.pop((owner, name), _ABSENT)
        if vars(owner).get(name) is not reading:
            continue
        if saved is _ABSENT:
            delattr(owner, name)
        else:
            setattr(owner, name, saved)


@contextmanager
def declining() -> Iterator[None]:
    """Have loopy read with loopty's readings in this thread, until it exits.

    A conversion, a constant whose type is not an integer's and a guard on a
    value argument whose dtype is not an integer's are declined (see the
    module). Re-entrant: an inner one leaves the readings to the outer. The
    readings are installed in loopy when the first one in the process opens
    and restored when the last one exits, an exception included, under a
    lock; while they are installed, a thread outside every one reads with
    loopy's own. Also a decorator, ``@declining()``, which runs the function
    inside one.
    """
    global _OPEN
    with _LOCK:
        if _OPEN == 0:
            _install()
        _OPEN += 1
    _DEPTH.value = getattr(_DEPTH, "value", 0) + 1
    try:
        yield
    finally:
        _DEPTH.value -= 1
        with _LOCK:
            _OPEN -= 1
            if _OPEN == 0:
                _restore()


def declining_methods[T: type](cls: T) -> T:
    """``cls``, its constructor and every public method run inside :func:`declining`.

    What :class:`loopty.schedule.Schedule` is built and transformed under:
    one chokepoint for every step, where the step's own calls into loopy are
    many.
    """
    for name, member in list(vars(cls).items()):
        if inspect.isfunction(member) and (
            name == "__init__" or not name.startswith("_")
        ):
            setattr(cls, name, _inside(member))
    return cls


def _inside(function: Callable[..., Any]) -> Callable[..., Any]:
    """``function``, run inside :func:`declining`."""
    return declining()(function)


# }}}


def affine_form(expr: Any) -> isl.Aff | None:
    """The affine expression loopy reads ``expr`` as, or ``None`` if it reads none.

    Over a space of the names ``expr`` uses, as ``simplify_via_aff`` reads
    one, with loopty's readings (:func:`declining`). A subscript loopy reads
    so is simplified to the affine expression isl gives back, in loopy's
    32-bit index type, whatever it was written as; one it does not is
    generated as it is written.
    """
    from loopy.diagnostic import ExpressionToAffineConversionError
    from loopy.symbolic import get_dependencies, guarded_aff_from_expr

    with declining():
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
