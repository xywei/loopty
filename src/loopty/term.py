"""The Term IR: what tracing produces and what lowering consumes.

A term is a kernel in one form: its parameters with their array types, its free
size parameters, and a tuple of statements. Every statement carries its iteration
domain as an isl set, so a statement's type is an isl object and nothing has to
be re-derived from source. Index expressions are pymbolic, in the array's own
index-type axes rather than flattened, so a layout change is a change of map and
not a rewrite of the indices. The ``where`` string is ``file:line`` taken from
the frame that executed the assignment, which is how diagnostics point back at
Python source without an AST pass.

The dataclasses are frozen: a term is a value, and a transformation produces a
new one rather than mutating the old, which is what lets a cast be checked by
comparing the two.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import islpy as isl

__all__ = ["Access", "ArrType", "Expression", "Reduction", "Stmt", "Term"]

#: A pymbolic expression. Kept loose on purpose: lanky's term classes (``Sum``,
#: ``Forall``, ``Abs``) and pymbolic's primitives both appear here.
Expression = Any


@dataclass(frozen=True)
class Access:
    """One array reference.

    ``indices`` are in the array's index-type axes, not in flat storage: a
    ragged array is indexed ``(r, j)``, and the offsets enter only at lowering.
    """

    array: str
    indices: tuple[Expression, ...]


@dataclass(frozen=True)
class Reduction:
    """A reduction over ``inames``, the term ``lanky.sum`` and friends lower to.

    ``domain`` is the reduced iteration space, with the enclosing inames present
    as parameters or dimensions so that a ragged reduction bound can depend on
    the row. ``exactness`` is the floating-point contract of the accumulation:
    ``exact`` forbids reassociation (no trees, no atomics), ``reassoc`` permits
    it and marks the result, ``approx`` carries a tolerance.
    """

    op: str
    inames: tuple[str, ...]
    domain: isl.Set
    body: Expression
    exactness: str


@dataclass(frozen=True)
class Stmt:
    """One statement instance family: a domain plus what it computes.

    ``inames`` are the enclosing loop variables, outer to inner, and ``domain``
    is the isl set of their values. ``guard`` is the condition of an enclosing
    ``when`` block, which masks the write rather than skipping it.

    ``kind`` is ``"assign"`` or ``"accumulate"``, and the difference is a claim
    about ``expr``, not a licence to guess:

    * ``"assign"`` means ``expr`` does not read the cell ``assignee`` names.
    * ``"accumulate"`` means it does. ``expr`` is always the **complete**
      right-hand side, so ``y[r] += t`` is recorded as
      ``assignee=y[r], expr=y[r] + t``, never as the increment ``t`` alone.

    That convention is fixed here because it cannot be recovered afterwards:
    ``y[r] = y[r] + t`` and ``y[r] += t`` are the same Python, an increment and
    a full right-hand side are both well-formed expressions, and a lowering that
    guessed wrong would silently double the term. :mod:`loopty.lower` therefore
    checks the invariant and refuses a term that breaks it, rather than
    repairing it; :mod:`loopty.flow` reads the same convention when it decides
    that the accumulated cell's read is already covered by the ``acc``
    footprint.

    ``order`` is the statement's position in the loop tree, one integer per
    level from the outermost block down to the statement itself, so that
    ``order`` interleaved with ``inames`` is the 2d+1 time vector of the source
    schedule. It is what tells two statements in the same loop nest apart from
    two statements in sequence; see :mod:`loopty.flow`. Defaulted, because a
    term written by hand in a test does not have to care.
    """

    id: str
    inames: tuple[str, ...]
    domain: isl.Set
    assignee: Access
    expr: Expression
    kind: str
    guard: Expression | None
    where: str
    order: tuple[int, ...] = ()


@dataclass(frozen=True)
class ArrType:
    """The type of an array parameter: axis sizes, element sort, raggedness.

    An axis whose size names another array parameter is ragged: for
    ``val: Arr[Fin[n], Fin[cnt], Real]`` the axes are ``(n, cnt)`` and
    ``ragged == (False, True)``, meaning the bound of the second axis at row
    ``r`` is ``cnt[r]``.
    """

    axes: tuple[Expression, ...]
    dtype: object
    ragged: tuple[bool, ...]

    def __post_init__(self) -> None:
        if len(self.axes) != len(self.ragged):
            raise ValueError(
                f"{len(self.axes)} axes but {len(self.ragged)} raggedness flags"
            )

    @property
    def ndim(self) -> int:
        """Number of index axes."""
        return len(self.axes)


@dataclass(frozen=True)
class Term:
    """A traced kernel.

    ``params`` are in signature order, each with its :class:`ArrType` (arrays) or
    a scalar sort. ``sizes`` are the free size parameters, which are the isl
    parameters of every statement domain. ``post`` is the return annotation, a
    proposition about the parameters, which becomes a fact to establish rather
    than an assertion to trust.

    ``reflected`` names the isl parameters that stand for the term's non-affine
    subexpressions, each with the subexpression it stands for: ``nl_cnt_r`` and
    ``cnt[r]``. It travels with the term because the allocation is a table
    rather than a rule (see :class:`loopty.idx.Reflections`), so anything that
    builds another isl set about this term, or that has to read ``cnt[r]`` back
    out of a domain parameter, has to be told what was allocated instead of
    guessing from the spelling. A term written by hand leaves it empty and is
    read by spelling, which is what :data:`loopty.lower.COUNT_PARAM_REFLECTED`
    is for.
    """

    name: str
    params: tuple[tuple[str, ArrType | object], ...]
    sizes: tuple[str, ...]
    stmts: tuple[Stmt, ...]
    post: Expression | None
    reflected: tuple[tuple[str, Expression], ...] = ()

    @property
    def param_names(self) -> tuple[str, ...]:
        """Parameter names, in signature order."""
        return tuple(name for name, _ in self.params)

    @property
    def reflections(self) -> Any:
        """This term's :class:`loopty.idx.Reflections`, rebuilt from the record.

        Every parameter it already allocated is adopted, and every name the term
        uses is reserved, so a set built later (the cell set of an in-bounds
        obligation, say) reuses the parameter for a term it has already seen and
        cannot collide with one it has not.
        """
        from loopty.idx import Reflections

        table = Reflections()
        table.reserve(self.param_names)
        table.reserve(self.sizes)
        for stmt in self.stmts:
            table.reserve(stmt.inames)
        for name, expr in self.reflected:
            table.adopt(name, expr)
        return table

    def stmt(self, stmt_id: str) -> Stmt:
        """The statement with the given id."""
        for stmt in self.stmts:
            if stmt.id == stmt_id:
                return stmt
        raise KeyError(stmt_id)
