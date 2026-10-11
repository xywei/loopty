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
"""

from __future__ import annotations

import re
from collections.abc import Iterator, Sequence
from contextlib import contextmanager
from typing import TYPE_CHECKING, Any

import islpy as isl
import pymbolic.primitives as prim
from loopy.diagnostic import LoopyIndexError

from loopty.lower import LoweringError, walk
from loopty.term import ArrType, Stmt

if TYPE_CHECKING:
    from loopty.lower import Lowering

__all__ = ["GuardNotRead", "naming_the_fix"]

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
        f"loopy's bounds check refuses an access of {stmt.id} of {term.name} at "
        f"{stmt.where}: it reads into isl only a guard that is an affine "
        f"comparison of integers, so under the guard{plural} {guard} it checks "
        "the access at every point of the loops around it, and cannot show it "
        "in bounds there. loopty's in-bounds fact of the access asks the same "
        "of the same points, and 'lanky check' reports it refuted, with the "
        f"guard left out. {fix}. loopy said: {text}"
    )


def _in_bounds_guards(stmt: Stmt, lowering: Lowering) -> list[str]:
    """Guards over the loops under which every access loopy refused is in bounds.

    For each access of the statement's value and its assignee that is not in
    bounds at every point of the statement's domain, the points where it is,
    less what the domain says already, written as Python: ``i <= n - 5`` for
    ``x[i + 4]`` of an ``x`` of ``n`` cells. Only one isl writes with no
    quotient and over the loop variables and the sizes is given; anything
    else, or an error, gives none.
    """
    from loopty import flow

    term = lowering.term
    types = dict(term.params)
    names = set(stmt.inames) | set(flow.size_names(term))
    names |= {
        name for name, sort in term.params if not isinstance(sort, ArrType)
    }
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
    except Exception:  # noqa: BLE001 - a suggestion is extra; the message stands
        return []
    return out


def _guard_text(points: isl.Set, names: set[str]) -> str | None:
    """``points`` as a Python guard, ``(i <= n - 5) & (j >= 1)``, or ``None``."""
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
        f"the guard of {stmt.id} of {term.name} at {stmt.where} calls {name}, "
        "which loopy cannot call there: its type inference visits a statement's "
        "value and never its guard, so a function only a guard calls is never "
        f"resolved, and loopy failed with KeyError: {name!r} (note 20 in "
        "docs/loopy-notes.md). Compute the value the guard calls it on in a "
        "statement first, into an array the guard then reads. abs of a real "
        "is written without a call, and needs none of this"
    )
