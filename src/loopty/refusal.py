"""loopy's refusals of a kernel, said in the kernel's own terms.

loopy checks and generates a kernel loopty lowered, and two of its refusals
name an instruction of its own and no fix:

* its bounds check reads a guard into isl only where the guard is an affine
  comparison of integers, and under any other guard it checks an access at
  every point of the loop, so it refuses one the guard keeps in bounds:
  ``x[i + 4]`` under ``when(i * 0.5 < 2)``, ``when(i * i < 16)`` or
  ``when(m[i] > 0.5)`` fails with ``LoopyIndexError: 'x[i + 4]' in
  instruction 'S0' ... could not establish ...`` (#148, note 23 in
  ``docs/loopy-notes.md``);
* its type inference visits an instruction's value and never its guard, so a
  function only a guard calls is never resolved, and loopy fails with
  ``KeyError`` and the function's name (#154, note 20).

:func:`naming_the_fix` runs loopy, and where it refuses so, raises instead an
error that names the statement, where the body wrote it, the guard, and what
to write: :class:`GuardNotRead`, a ``LoopyIndexError`` still, with loopy's
message at its end, and a :class:`~loopty.lower.LoweringError` for a call.
:meth:`loopty.executor.LoopyExecutor.run` and :func:`loopty.executor.emit_code`
run loopy inside it. Any other refusal is loopy's own, raised as it is.

The compiled run is also refused where the native run is, at a point the
call's sizes reach: numpy converts a Python int beside a numpy integer into
the integer's type, and refuses one the type does not hold, so ``u[i] + (i -
3)`` of a ``uint32`` ``u`` is refused natively at ``i = 0`` and ``a[i] * (i *
100)`` of an ``int8`` ``a`` from ``i = 2``, where C computes both (#153).
:func:`refuse_where_native_does` finds such a point with isl before the
compiled run starts, and raises :class:`RefusedNatively`, an
``OverflowError`` as numpy's is.
"""

from __future__ import annotations

import re
from collections.abc import Collection, Iterator, Mapping, Sequence
from contextlib import contextmanager
from typing import TYPE_CHECKING, Any

import islpy as isl
import numpy as np
import pymbolic.primitives as prim
from loopy.diagnostic import LoopyIndexError

from loopty.lower import LoweringError, walk
from loopty.names import NamingError, check_names
from loopty.term import Access, ArrType, Reduction, Stmt, Term, subtracted

if TYPE_CHECKING:
    from loopty.lower import Lowering

__all__ = [
    "GuardNotRead",
    "RefusedNatively",
    "naming_the_fix",
    "refuse_where_native_does",
]

#: How loopy's bounds check names the instruction it refuses an access of.
_INSTRUCTION = re.compile(r"in instruction '([^']+)'")


class GuardNotRead(LoopyIndexError):
    """loopy's bounds check refused an access under a guard it does not read.

    The message names the statement, the guard and the fix, and ends with
    loopy's own; a caller catching loopy's ``LoopyIndexError`` catches this
    too (#148).
    """


@contextmanager
def naming_the_fix(lowering: Lowering) -> Iterator[None]:
    """Run loopy on ``lowering``'s kernel, naming the fix where it refuses a guard.

    See the module. loopy prints the translation unit it refuses before it
    raises, which is left as it is.
    """
    try:
        yield
    except LoopyIndexError as exc:
        if isinstance(exc, GuardNotRead):
            raise
        named = _unread_guard(exc, lowering)
        if named is None:
            raise
        raise named from exc
    except KeyError as exc:
        named = _called_in_a_guard(exc, lowering)
        if named is None:
            raise
        raise named from exc


def _statement(lowering: Lowering, instruction: str) -> Stmt | None:
    """The statement of the term that became the loopy instruction ``instruction``."""
    for stmt_id, insn_id in lowering.insn_ids.items():
        if insn_id == instruction:
            return next(
                (stmt for stmt in lowering.term.stmts if stmt.id == stmt_id), None
            )
    return None


def _said(term: Term, stmt_id: str) -> str:
    """A statement as a message names it: ``y[i] = x[i] (f.py:3)``."""
    from loopty.handles import statement_names

    return statement_names(term)(stmt_id)


def _unread_guard(error: LoopyIndexError, lowering: Lowering) -> GuardNotRead | None:
    """:class:`GuardNotRead` for loopy's refusal where a guard explains it, or ``None``.

    The statement has to have a guard conjunct its domain does not state
    (:attr:`loopty.term.Stmt.unnarrowed`): those are the guards loopy reads
    no more than loopty does, a product of loop variables, a guard computed in
    floating point, a ``Real`` scalar's, or an array's entry. Under a guard
    it does read, the refusal is loopy's to make.
    """
    text = str(error)
    found = _INSTRUCTION.search(text)
    if found is None:
        return None
    stmt = _statement(lowering, found.group(1))
    if stmt is None or not stmt.unnarrowed:
        return None
    term = lowering.term
    conjuncts = [conjunct for conjunct, _why in stmt.unnarrowed]
    guard = " and ".join(f"'{conjunct}'" for conjunct in conjuncts)
    suggested = _in_bounds_guards(stmt, lowering)
    if suggested:
        fix = (
            "Write the guard as a comparison of integers, which loopy reads, "
            "or add one that keeps the access in bounds, which it reads too: "
            + " and ".join(f"'with when({condition}):'" for condition in suggested)
        )
    else:
        fix = (
            "Write the guard as a comparison of integers, which loopy reads "
            "('when(i < 4)' for 'when(i * 0.5 < 2)'), or add one that keeps "
            "the access in bounds"
        )
    plural = "s" if len(conjuncts) > 1 else ""
    return GuardNotRead(
        f"loopy's bounds check refuses an access of {_said(term, stmt.id)} in "
        f"{term.name}: it reads into isl only a guard that is an affine "
        f"comparison of integers, so under the guard{plural} {guard} it checks "
        "the access at every point of the loops around it, and cannot show it "
        f"in bounds there. {fix}. loopy said: {text}"
    )


def _in_bounds_guards(stmt: Stmt, lowering: Lowering) -> list[str]:
    """Guards over the loops under which every access loopy refused is in bounds.

    For each access of the statement's value and its assignee that is not in
    bounds at every point of the statement's domain, the points where it is,
    less what the domain says already, written as the body would write it:
    ``i <= x.dom.size - 5`` for ``x[i + 4]`` of an ``x`` of ``n`` cells, a
    size as an array's domain reads it (:func:`_size_spellings`), since the
    body has no name ``n``. Only one isl writes with no quotient and over the
    loop variables, the scalars and the sizes an axis of an array is is
    given; anything else, or an error, gives none.
    """
    from loopty import flow

    term = lowering.term
    types = dict(term.params)
    names = dict.fromkeys(stmt.inames, "")
    names.update(_size_spellings(term, stmt))
    names.update(
        (name, "") for name, sort in term.params if not isinstance(sort, ArrType)
    )
    names = {name: spelled or name for name, spelled in names.items()}
    out: list[str] = []
    try:
        for array, indices, _kind, inames, domain in flow.statement_accesses(
            stmt, term
        ):
            arrtype = types.get(array)
            if not isinstance(arrtype, ArrType):
                continue
            if tuple(inames) != tuple(stmt.inames) or domain is not stmt.domain:
                continue
            relation = flow.access_relation(inames, domain, indices)
            cells = flow.cell_set(arrtype, indices, reflections=term.reflections)
            cells = cells.align_params(relation.get_space().params())
            relation = relation.align_params(cells.get_space().params())
            inside = relation.intersect_range(cells).domain()
            whole = relation.domain()
            if inside.is_equal(whole):
                continue
            guard = _guard_text(inside.gist(whole), names)
            if guard is None:
                return []
            if guard not in out:
                out.append(guard)
    except NamingError:
        # Not the kernel's: an isl object loopty built is misnamed.
        raise
    except Exception:  # noqa: BLE001 - a suggestion is extra; the message stands
        return []
    return out


def _size_spellings(term: Term, stmt: Stmt) -> dict[str, str]:
    """How the body reads each size that is an axis of an array parameter.

    ``x.dom.size`` for the ``n`` of ``x: Arr[Fin[n], Real]``, and
    ``m.dom[0].size`` for the ``k`` of ``m: Arr[Fin[n], Fin[k], Real]``: a
    body asks an array's domain for a size, and has no name for it. The
    statement's own array is asked first, then the parameters in order. A
    ragged axis, an extent that is not one size, and an array over a
    polyhedral domain spell none.
    """
    types = dict(term.params)
    order = [stmt.assignee.array, *(name for name, _ in term.params)]
    out: dict[str, str] = {}
    for array in order:
        typ = types.get(array)
        if not isinstance(typ, ArrType) or typ.domain is not None:
            continue
        for axis, (extent, ragged) in enumerate(zip(typ.axes, typ.ragged, strict=True)):
            if ragged or not isinstance(extent, prim.Variable) or extent.name in out:
                continue
            fiber = f"[{', '.join(['0'] * axis)}]" if axis else ""
            out[extent.name] = f"{array}.dom{fiber}.size"
    return out


def _guard_text(points: isl.Set, names: Mapping[str, str]) -> str | None:
    """``points`` as a Python guard, ``(i <= x.dom.size - 5) & (j >= 1)``, or ``None``.

    ``names`` holds the names a guard may use, each with how the body writes
    it.
    """
    from loopty.flow import _constraint_text

    pieces = points.coalesce().get_basic_sets()
    if len(pieces) != 1 or pieces[0].dim(isl.dim_type.div):
        return None
    piece = pieces[0]
    dims = [
        piece.get_dim_name(isl.dim_type.set, k)
        for k in range(piece.dim(isl.dim_type.set))
    ]
    params = [
        piece.get_dim_name(isl.dim_type.param, k)
        for k in range(piece.dim(isl.dim_type.param))
    ]
    texts: list[str] = []
    for constraint in piece.get_constraints():
        on = _coefficients(constraint, isl.dim_type.set, dims)
        by = _coefficients(constraint, isl.dim_type.param, params)
        if not on and not by:
            continue
        if any(name not in names for name, _ in (*on, *by)):
            return None
        on = [(names[name], value) for name, value in on]
        by = [(names[name], value) for name, value in by]
        constant = int(constraint.get_constant_val().to_python())
        text = _constraint_text(on, by, constant, bool(constraint.is_equality()))
        texts.append(text.replace(" = ", " == ").replace("*", " * "))
    if not texts:
        return None
    if len(texts) == 1:
        return texts[0]
    return " & ".join(f"({text})" for text in texts)


def _coefficients(
    constraint: Any, kind: Any, names: Sequence[str]
) -> list[tuple[str, int]]:
    """The names of one kind a constraint has, with their nonzero coefficients."""
    out = []
    for k, name in enumerate(names):
        value = int(constraint.get_coefficient_val(kind, k).to_python())
        if value:
            out.append((name, value))
    return out


def _called_in_a_guard(error: KeyError, lowering: Lowering) -> LoweringError | None:
    """A :class:`~loopty.lower.LoweringError` for a function only a guard calls.

    loopy's ``KeyError`` names the function. It is explained only where a
    statement's guard calls a function of that name and no statement's value
    does; any other ``KeyError`` is loopy's, or loopty's own, and stands.
    """
    if len(error.args) != 1 or not isinstance(error.args[0], str):
        return None
    name = error.args[0]
    term = lowering.term

    def calls(expr: Any) -> bool:
        return expr is not None and any(
            isinstance(node, prim.Call)
            and isinstance(node.function, prim.Variable)
            and node.function.name == name
            for node in walk(expr)
        )

    if any(calls(stmt.expr) for stmt in term.stmts):
        return None
    guarded = [stmt for stmt in term.stmts if calls(stmt.guard)]
    if not guarded:
        return None
    stmt = guarded[0]
    return LoweringError(
        f"the guard of {_said(term, stmt.id)} in {term.name} calls {name}, "
        "which loopy cannot call there: its type inference visits a statement's "
        "value and never its guard, so a function only a guard calls is never "
        f"resolved, and loopy failed with KeyError: {name!r} (note 20 in "
        "docs/loopy-notes.md). Compute the value the guard calls it on in a "
        "statement first, into an array the guard then reads. abs of a real "
        "is written without a call, and needs none of this"
    )


# {{{ loop arithmetic beside a numpy integer that does not hold it (#153)


class RefusedNatively(OverflowError):
    """The compiled run refused a call at a point where the native run is refused.

    numpy refuses there, with an ``OverflowError``, a Python int beside a
    numpy integer whose type does not hold it, and the compiled kernel would
    compute a value in C's type instead (#153). The message names the point
    and the fix. ``checks`` names the flags of a program's checked points the
    refused statement runs after, empty for one every run reaches, and
    ``statement`` the refused statement's id, for the executor to stop the
    program at the call it is in (#176).
    """

    checks: frozenset[str] = frozenset()
    statement: str | None = None


#: The operations numpy converts a Python int beside a numpy integer for, into
#: the integer's type, refusing one it does not hold. A comparison compares
#: the two exactly, and ``/`` divides them in double.
_CONVERTING = (
    prim.Sum,
    prim.Product,
    prim.FloorDiv,
    prim.Remainder,
    prim.Power,
    prim.BitwiseXor,
    prim.LeftShift,
    prim.RightShift,
)


def refuse_where_native_does(
    term: Term, supplied: Mapping[str, Any]
) -> RefusedNatively | None:
    """Refuse the compiled run of ``term`` on ``supplied`` where the native run is.

    Arithmetic of loop variables and sizes is a Python int natively, and
    beside a numpy integer numpy converts it into the integer's type and
    refuses a value the type does not hold (NEP 50). The compiled kernel
    computes it in C's type, so it computed a value at a point where the
    native run has none: ``u[i] + (i - 3)`` of a ``uint32`` ``u``, and ``y[i]
    = i`` of an ``int8`` ``y`` from ``i = 128`` (#153). The value is known
    only once the sizes are, which the call gives, and isl finds whether the
    loops reach a point where the type does not hold it:

    * an operation, ``+``, ``*``, ``//``, ``%``, ``**``, ``^``, ``<<`` or
      ``>>``, of such arithmetic beside a numpy integer narrower than 64 bits
      or an unsigned one, over the loops around the statement with no guard
      applied, since a ``when`` masks the write and the body computes at
      every point of its loops; inside a sum, over the sum's own, narrowed by
      the generator's ``if`` clause, which skips the body
      (:func:`_summed_points`). A point
      where one of the integer's reads is out of bounds is left out: under a
      false guard the native run reads such a cell as the Python int ``0``;
    * a store of such arithmetic into an array stored so natively, over the
      points the statement stores at, when its guard narrows its domain
      whole, a program's checked points apart (:func:`_store`); and of a
      literal the array's type does not hold under such a
      guard, which the trace leaves here (:func:`loopty.trace.trace`), since
      the guard may hold nowhere: ``with when(i < 0): y[i] = 300``;
    * such arithmetic a kernel called in a loop is given for a scalar of a
      numpy integer sort, which its contract converts into that type, over
      the points the call is made at (:func:`_converted`, #232).

    An integer of 64 bits that is signed is not asked about, since loop
    arithmetic leaves it only past 64 bits, which is #139's limit; nor is
    arithmetic isl cannot read, a product of two loop variables say, nor
    loops whose bounds the call's sizes do not fix, a ragged row's, nor a
    store under a guard isl does not read, which stores where an array's
    entries say: those are left to the native refusal (#166).

    A program's statement after a checked point runs only where the point
    passed (:meth:`loopty.compose._Composer.guarded`), and natively the call
    after a point that fails is refused before it starts, with the point's
    error. So its refusal is returned, not raised, with the flags of the
    points before it (:attr:`RefusedNatively.checks`) and the statement
    (:attr:`RefusedNatively.statement`), for the executor to run the program
    up to the statement's call and raise it once the run has shown that
    those points passed, before anything a later point says; a statement
    every run reaches is refused here.
    """
    from loopty.contract import resolve_sizes
    from loopty.promotion import Promotion

    sizes = resolve_sizes(dict(term.params), supplied)
    promotion = Promotion(term)
    flags = {flag for flag, _message in term.checks}
    deferred: RefusedNatively | None = None
    for call in term.calls:
        _converted(term, call, promotion, sizes)
    for stmt in term.stmts:
        loops = stmt.loop_domain if stmt.loop_domain is not None else stmt.domain
        try:
            for part in (stmt.assignee.indices, stmt.expr, stmt.guard):
                _operations(part, term, stmt, promotion, loops, sizes)
            _store(term, stmt, promotion, sizes, flags)
        except RefusedNatively as refusal:
            before = _checks_before(stmt, flags)
            if not before:
                raise
            if deferred is None:
                deferred = RefusedNatively(
                    str(refusal).replace(
                        "stops before it starts",
                        "stops at the call, since the program's checked points "
                        "before the call passed, which only the run shows, and "
                        "keeps what the calls before it wrote, as natively",
                        1,
                    )
                )
                deferred.checks = before
                deferred.statement = stmt.id
    return deferred


def _checks_before(stmt: Stmt, flags: set[str]) -> frozenset[str]:
    """The flags of a program's checked points ``stmt`` runs only after."""
    if not flags or stmt.guard is None:
        return frozenset()
    out: set[str] = set()
    for node in walk(stmt.guard):
        if isinstance(node, Access) and node.array in flags:
            out.add(node.array)
        elif (
            isinstance(node, prim.Subscript)
            and isinstance(node.aggregate, prim.Variable)
            and node.aggregate.name in flags
        ):
            out.add(node.aggregate.name)
    return frozenset(out)


def _operations(
    expr: Any,
    term: Term,
    stmt: Stmt,
    promotion: Any,
    loops: isl.Set,
    sizes: Mapping[str, int],
) -> None:
    """:func:`refuse_where_native_does` of the operations in ``expr``."""
    from lanky.terms import init_args

    if expr is None:
        return
    if isinstance(expr, Reduction):
        inner = _summed_points(expr, stmt)
        _operations(expr.body, term, stmt, promotion, inner, sizes)
        return
    if isinstance(expr, Access):
        _operations(expr.indices, term, stmt, promotion, loops, sizes)
        return
    if isinstance(expr, tuple | list):
        for item in expr:
            _operations(item, term, stmt, promotion, loops, sizes)
        return
    if not isinstance(expr, prim.ExpressionNode):
        return
    if isinstance(expr, _CONVERTING):
        _operation(expr, term, stmt, promotion, loops, sizes)
    for arg in init_args(expr):
        _operations(arg, term, stmt, promotion, loops, sizes)


def _summed_points(reduction: Reduction, stmt: Stmt) -> isl.Set:
    """The points at which the native run computes a sum's body.

    The sum's loops, narrowed by the generator's own ``if`` clause, which
    skips the body where it is false, and not by the statement's guard,
    which masks only the write: ``reduce_sum(a[j] + j for j in a.dom if j <
    100)`` computes ``a[j] + j`` for ``j < 100`` alone. The sum's
    ``loop_domain`` is narrowed by neither and its ``domain`` by both. Under a
    guard the clause cannot be told apart from the guard once both narrow the
    domain, and then the sum's ``domain`` is asked, a part of those points;
    the points the guard leaves out are left to the native refusal (#166).
    """
    if reduction.loop_domain is None:
        return reduction.domain
    if stmt.guard is None:
        # Nothing but the clause narrowed the domain.
        return reduction.domain
    try:
        loops = reduction.loop_domain
        names = loops.get_var_names(isl.dim_type.set)
        outer = stmt.domain.get_var_names(isl.dim_type.set)
        if names[: len(outer)] != outer:
            return reduction.domain
        guarded = stmt.domain.add_dims(isl.dim_type.set, len(names) - len(outer))
        for k in range(len(outer), len(names)):
            guarded = guarded.set_dim_name(isl.dim_type.set, k, names[k])
        guarded = check_names(guarded, f"the domain of {stmt.id} over its sum's loops")
        guarded, loops = isl.align_two(guarded, loops)
        narrowed, domain = isl.align_two(loops.intersect(guarded), reduction.domain)
        if narrowed.is_equal(domain):
            # The guard alone narrowed it: there is no clause.
            return reduction.loop_domain
    except isl.Error:
        pass
    return reduction.domain


def _operands(node: Any) -> tuple[Any, ...]:
    """The operands of an operation of :data:`_CONVERTING`, in numpy's order."""
    if isinstance(node, prim.Sum | prim.Product | prim.BitwiseXor):
        return tuple(node.children)
    if isinstance(node, prim.Power):
        return (node.base, node.exponent)
    if isinstance(node, prim.LeftShift | prim.RightShift):
        return (node.shiftee, node.shift)
    return (node.numerator, node.denominator)


def _operation(
    node: Any,
    term: Term,
    stmt: Stmt,
    promotion: Any,
    loops: isl.Set,
    sizes: Mapping[str, int],
) -> None:
    """:func:`refuse_where_native_does` of one operation, operand by operand.

    numpy computes a sum or a product of several from the left, so an
    operand after the first stands beside what the ones before it make, and
    the first beside the second. A difference subtracts the operand its last
    term negates (:func:`loopty.term.subtracted`).
    """
    from loopty.promotion import _subtrahend

    operands = _operands(node)
    steps = promotion.steps(node)
    accumulated = promotion.types(operands[0])[0]
    for position, operand in enumerate(operands[1:], start=1):
        native = promotion.types(operand)[0]
        pairs = [(operand, native, accumulated, operands[:position])]
        if position == 1:
            pairs.append((operands[0], accumulated, native, (operand,)))
        for value, value_native, beside, others in pairs:
            if value is operand and subtracted(node, position):
                inner = _subtrahend(value)
                if inner is not None:
                    value, value_native = inner, promotion.types(inner)[0]
            dtype = _narrow_integer(beside)
            if dtype is None or not _loop_arithmetic(value, value_native):
                continue
            leaves = _leaves(value, dtype, loops, sizes, term, others)
            if leaves is None:
                continue
            raise RefusedNatively(_operation_message(node, value, dtype, leaves, stmt))
        if position - 1 < len(steps):
            accumulated = steps[position - 1].native


def _store(
    term: Term,
    stmt: Stmt,
    promotion: Any,
    sizes: Mapping[str, int],
    flags: Collection[str] = (),
) -> None:
    """:func:`refuse_where_native_does` of the store a statement makes.

    A guard isl does not read leaves points in the statement's domain at
    which nothing is stored, and the store is not asked about; but for the
    flags of a program's checked points (``flags``), which say whether the
    statement runs at all, and are asked after the run
    (:attr:`RefusedNatively.checks`).
    """
    from lanky.terms import Comparison, Subscript, Var, render

    from loopty.contract import native_storage

    checks = {render(Comparison(Subscript(Var(flag), 0), "==", 0)) for flag in flags}
    if any(conjunct not in checks for conjunct, _why in stmt.unnarrowed):
        # Its domain holds points its guard masks, at which nothing is stored.
        return
    typ = dict(term.params).get(stmt.assignee.array)
    if typ is None:
        typ = dict(term.temporaries).get(stmt.assignee.array)
    if not isinstance(typ, ArrType):
        return
    stored = native_storage(typ.dtype)
    if stored is None or stored.kind not in "iu":
        return
    dtype = _narrow_integer((stored.type(0),))
    value = stmt.expr
    # A literal under a guard, which the trace leaves to the call's sizes: the
    # guard masks the store, and numpy converts only what is stored.
    literal = isinstance(value, int) and not isinstance(value, bool)
    if dtype is None or not (
        literal or _loop_arithmetic(value, promotion.types(value)[0])
    ):
        return
    leaves = _leaves(value, dtype, stmt.domain, sizes, term, ())
    if leaves is None:
        return
    raise RefusedNatively(
        f"the compiled run of {term.name} stops before it starts: the store of "
        f"{_shown(value)} into {stmt.assignee.array} at {stmt.where} stores a "
        f"Python int into an array of {dtype}, which does not hold it, "
        f"{leaves}. numpy refuses it there ('Python integer out of bounds for "
        f"{dtype}'), so the native run stops there, and the compiled kernel "
        "would convert it as C does and store a value where the native one "
        f"has none. Give {stmt.assignee.array} a type that holds it, as an "
        f"array of Int, or keep the value inside {dtype}"
    )


def _converted(
    term: Term, call: Any, promotion: Any, sizes: Mapping[str, int]
) -> None:
    """:func:`refuse_where_native_does` of the scalars a call converts.

    Natively a kernel called in a loop converts each scalar it is given into
    the numpy type its sort is stored in, and numpy refuses a Python int that
    type does not hold: ``i - 1`` for an ``np.uint32`` parameter at ``i = 0``,
    ``i + 120`` for an ``np.int8`` one from ``i = 8``. The compiled kernel
    converts it as C does, and wraps round (:mod:`loopty.calls`, #232). The
    points are the call's, narrowed by the guards around it, under which the
    native run makes no call; a call under a guard isl does not read is left
    to the native refusal, as a store is.
    """
    from loopty.contract import native_storage

    if call.unknown or call.unnarrowed:
        return
    for argument in call.arguments:
        if argument.kind != "scalar":
            continue
        stored = native_storage(argument.sort)
        if stored is None or stored.kind not in "iu":
            continue
        dtype = _narrow_integer((stored.type(0),))
        value = argument.value
        if dtype is None or not _loop_arithmetic(value, promotion.types(value)[0]):
            continue
        leaves = _leaves(value, dtype, call.domain, sizes, term, ())
        if leaves is None:
            continue
        raise RefusedNatively(
            f"the compiled run of {term.name} stops before it starts: the call "
            f"of {call.callee} at {call.where} passes {_shown(value)}, a Python "
            f"int, for its {argument.param}, which it declares {dtype}, and a "
            f"{dtype} does not hold it, {leaves}. Natively the contract of "
            f"{call.callee} converts it into a {dtype}, which numpy refuses there "
            f"('Python integer out of bounds for {dtype}'), so the native run "
            "stops there, and the compiled kernel would convert it as C does and "
            f"wrap round. Keep the value inside {dtype}, or declare "
            f"{argument.param} Int"
        )


def _narrow_integer(native: Any) -> np.dtype | None:
    """The dtype of a native type that is one numpy integer narrower than 64
    bits, or an unsigned one, of any width: what loop arithmetic can leave."""
    dtype = None
    for sample in native or ():
        if not isinstance(sample, np.integer):
            return None
        if dtype is not None and sample.dtype != dtype:
            return None
        dtype = sample.dtype
    if dtype is None or (dtype.itemsize == 8 and dtype.kind == "i"):
        return None
    return dtype


def _loop_arithmetic(expr: Any, native: Any) -> bool:
    """Whether ``expr`` is arithmetic of loop variables and sizes.

    A Python int natively, and no literal: the trace refuses a literal a
    numpy integer does not hold, at every point (#141).
    """
    from loopty.promotion import _weak_integers

    if isinstance(expr, bool | int | np.generic):
        return False
    return bool(native) and _weak_integers(native)


def _leaves(
    expr: Any,
    dtype: np.dtype,
    loops: isl.Set,
    sizes: Mapping[str, int],
    term: Term,
    others: Sequence[Any],
) -> str | None:
    """Where the loop arithmetic ``expr`` leaves ``dtype``, at the call's sizes.

    ``loops`` with its parameters fixed to ``sizes``, less the points where a
    read in ``others`` is out of bounds, and the least point there at which
    the value of ``expr`` is outside ``dtype``, as ``-3 at i = 0``. ``None``
    where there is none, and where isl cannot say: ``expr`` is not
    quasi-affine, or a parameter of the loops or of ``expr`` is no size the
    call fixes.
    """
    from lanky.terms import evaluate

    from loopty import flow
    from loopty.trace import accesses_in

    info = np.iinfo(dtype)
    try:
        text = flow.expr_text(expr, None, None)
    except (flow.NonAffine, TypeError, ValueError):
        return None
    dims = list(loops.get_var_names(isl.dim_type.set))
    named = flow.free_names(text)
    params = sorted((named | set(loops.get_var_names(isl.dim_type.param))) - set(dims))
    if any(name not in sizes for name in params):
        return None
    fixed = " and ".join(f"{name} = {sizes[name]}" for name in params) or "true"
    try:
        # A size the text reads is a parameter, which a loop of its name
        # would take for itself (loopty.names).
        outside = check_names(
            isl.Set(
                f"[{', '.join(params)}] -> {{ [{', '.join(dims)}] : "
                f"({text} < {int(info.min)} or {text} > {int(info.max)}) "
                f"and {fixed} }}"
            ),
            f"the points where {text} leaves {dtype}",
            meant=named & set(sizes),
        )
        loops = loops.align_params(outside.get_space().params())
        outside = outside.align_params(loops.get_space().params()).intersect(loops)
        types = dict(term.params)
        for access in accesses_in(tuple(others)):
            arrtype = types.get(access.array)
            if not isinstance(arrtype, ArrType):
                continue
            relation = flow.access_relation(dims, outside, access.indices)
            cells = flow.cell_set(arrtype, access.indices, reflections=term.reflections)
            cells = cells.align_params(relation.get_space().params())
            relation = relation.align_params(cells.get_space().params())
            inside = relation.intersect_range(cells).domain()
            outside, inside = isl.align_two(outside, inside)
            outside = outside.intersect(inside)
        if outside.is_empty():
            return None
        point = outside.lexmin().sample_point()
        values = {
            name: int(point.get_coordinate_val(isl.dim_type.set, k).to_python())
            for k, name in enumerate(dims)
        }
    except NamingError:
        raise
    except (isl.Error, flow.NonAffine, ValueError):
        return None
    values.update({name: sizes[name] for name in params})
    try:
        value = str(evaluate(expr, values))
    except Exception:  # noqa: BLE001 - the point says where without it
        value = "a value outside it"
    at = ", ".join(f"{name} = {values[name]}" for name in dims)
    return f"{value} at {at}" if at else value


def _operation_message(
    node: Any, value: Any, dtype: np.dtype, leaves: str, stmt: Stmt
) -> str:
    """Why the compiled run refuses loop arithmetic beside ``dtype``, and the fix."""
    shown = _shown(value)
    bracketed = shown if isinstance(value, prim.Variable) else f"({shown})"
    article = "an" if dtype.name.startswith("int") else "a"
    double = " (in double beside a uint64)" if dtype == np.uint64 else ""
    return (
        f"the compiled run stops before it starts: {_written(node)} at "
        f"{stmt.where} computes the loop arithmetic {shown} with {article} "
        f"{dtype} value, whose type does not hold it, {leaves}. numpy refuses "
        f"it there ('Python integer out of bounds for {dtype}', NEP 50), so "
        "the native run stops there, and the compiled kernel would compute a "
        "value in C's type where the native one has none. Compute it in int64, "
        f"'{bracketed} * np.int64(1)', with which numpy computes in a type that "
        f"holds both{double}, as the compiled run does too, or keep it inside "
        f"{dtype}"
    )


def _shown(value: Any) -> str:
    """A term as the trace's messages print it."""
    from loopty.trace import _shown as shown

    return shown(value)


def _written(value: Any) -> str:
    """A sum as the body wrote it (:func:`loopty.trace._written`)."""
    from loopty.trace import _written as written

    return written(value)


# }}}
