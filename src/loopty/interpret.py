"""The term interpreter: what a traced term computes, run with numpy semantics.

Tracing runs a body once, at one generic point, and what it records is the
meaning every later step works from: the typing rules state obligations about
it, and lowering compiles it. The native run is the other meaning, the body
itself on real arrays. Nothing forces the two to agree. A body can keep state
where the tracer does not look, and then the term is one iteration of it, not
what it computes. This module gives the term a meaning of its own, without
loopy, so that the two can be compared (:mod:`loopty.faithful`).

:func:`interpret` runs a term on concrete arguments, in place, as the native run
does. Its reading of the term is literal:

* A statement's instances are the points of its isl domain at the sizes the
  arguments determine, enumerated with isl, and they run in the order of the
  source schedule, the 2d+1 time vector that :func:`loopty.flow.schedule_of`
  states: statement by statement in source order within each iteration of the
  loops around them.
* A loop is enumerated when the run reaches it, as the body's ``for`` is. A
  ragged bound such as ``cnt[r]`` is a reflected parameter of the domain (see
  :class:`loopty.idx.Reflections`), and it is read from the arrays where the
  loop it bounds starts, each time it starts: ``for j in val.dom[r]`` reads the
  length of row ``r`` then, as the statements before have left it, and runs to
  that length whatever the loop's own statements write. So a kernel that
  writes its counts runs here as it runs natively (:meth:`_Run.loop`). A bound
  of two loops of one statement, or of a loop and a sum inside it, is read
  where each of them starts, as the body reads it (:meth:`_Run.read_apart`),
  each loop's and each sum's own bound read off the loop nest before a guard
  or a clause narrowed it, where isl has not merged it into another's.
* A guard is evaluated at each instance, and an instance whose guard is false
  writes nothing. Guards and connectives short-circuit, so a condition to the
  right of a false one is not read.
* An expression is evaluated node by node with Python's operators on the numpy
  scalars read from the arrays, which is the arithmetic the native run does, in
  the order the tree was built. A call is looked up by the name loopy resolves
  against the target's library (``sqrt`` is :func:`numpy.sqrt`).
* A reduction adds its terms in the order the native ``reduce_sum`` does:
  lexicographically over its binders, the order of a generator's nested
  ``for`` clauses, with Python's :func:`sum`, starting from ``0``. A bound of
  its own binders is read where it is summed, and one of the loops around it
  keeps the value it was read at where its loop started, as the body's loop
  does.
* A ragged array is read and written through the counts and offsets the
  kernel declares, as the statements before have left them
  (:meth:`loopty.arr.Arr.through`), which is how the native run and the
  lowered kernel index it too.

What the interpreter does not do is guess. A construct it has no numpy meaning
for, or a domain it cannot enumerate at the given sizes, is an
:class:`InterpretError` that says so, never a best effort.
"""

from __future__ import annotations

import builtins
import operator
from collections.abc import Iterator, Mapping, Sequence
from dataclasses import dataclass, field, replace
from typing import Any

import islpy as isl
import numpy as np
import pymbolic.primitives as prim
from lanky.terms import Abs, evaluate, init_args, render

from loopty.arr import Arr
from loopty.contract import (
    native_copy,
    native_scalar,
    native_storage,
    read_storage,
    resolve_sizes,
)
from loopty.flow import bounds_dimension
from loopty.term import Access, ArrType, Reduction, Stmt, Term, declared_layout
from loopty.trace import accesses_in, reductions_in

__all__ = ["CheckFailed", "InterpretError", "TooLarge", "interpret"]


class InterpretError(RuntimeError):
    """A term the interpreter cannot give a meaning to, with the reason."""


class TooLarge(InterpretError):
    """More statement instances than the caller allowed."""


class CheckFailed(ValueError):
    """A checked point of a program found a cell its requirement excludes.

    A program checks what a call's contract checks of an array an earlier
    call wrote, where nothing decided it (:class:`loopty.term.Requirement`).
    The message is the requirement's, which the compiled run raises too; it
    is a :class:`ValueError`, as the native refusal of the call is.
    """


class _Unknown(Exception):
    """A name with no value yet: a loop variable of a level not reached."""


#: The functions a term may call, by the name loopy resolves against the
#: target's math library, and the numpy function that computes the same thing.
_FUNCTIONS = {
    "sqrt": np.sqrt,
    "exp": np.exp,
    "log": np.log,
    "log2": np.log2,
    "log10": np.log10,
    "sin": np.sin,
    "cos": np.cos,
    "tan": np.tan,
    "asin": np.arcsin,
    "acos": np.arccos,
    "atan": np.arctan,
    "atan2": np.arctan2,
    "sinh": np.sinh,
    "cosh": np.cosh,
    "tanh": np.tanh,
    "fabs": np.fabs,
    "abs": np.abs,
    "floor": np.floor,
    "ceil": np.ceil,
    "pow": np.power,
    "fmin": np.fmin,
    "fmax": np.fmax,
}

_COMPARISONS = {
    "==": operator.eq,
    "!=": operator.ne,
    "<": operator.lt,
    "<=": operator.le,
    ">": operator.gt,
    ">=": operator.ge,
}

_BINARY = (
    (prim.Quotient, operator.truediv),
    (prim.FloorDiv, operator.floordiv),
    (prim.Remainder, operator.mod),
)


def interpret(
    term: Term, arguments: Mapping[str, Any], limit: int | None = None
) -> dict[str, np.ndarray]:
    """Run ``term`` on ``arguments``, in place, and return what it wrote.

    ``arguments`` maps every parameter to its value: an :class:`~loopty.arr.Arr`
    or a numpy array for an array parameter (a ragged one has to be an ``Arr``,
    which knows where its rows end), a number for a scalar. Arrays are written
    in place, as the native run writes them, and the result maps each array the
    term writes to its buffer.

    ``limit`` bounds the work: the statement instances and the terms of the
    reductions they evaluate, counted together, since one instance can sum a
    whole row. More is a :class:`TooLarge`. It is raised before anything runs
    when the domains that no array bounds are over between them, by their
    bounding boxes, and otherwise at the loop, the instance or the reduction
    that goes over, with the arrays partly written. A domain is checked by its
    bounding box before its points are collected (see :meth:`_Run.enumerate`),
    so a domain far past the limit costs nothing to refuse. An array the term
    does not write is read in the dtype its sort is stored in when it is
    given in another (:func:`loopty.contract.read_storage`): an integral one
    stored as floats as integers, a ``Real`` one stored as integers as
    ``float64``, and a scalar is read in that dtype whatever the term does
    (:func:`loopty.contract.native_scalar`). That is what the native run does
    (see :meth:`loopty.kernel.Kernel.__call__`).
    """
    return _Run(term, arguments).run(limit)


def _storage(value: Any, typ: ArrType, written: bool) -> Arr:
    """The runtime array the interpreter reads and writes for one argument."""
    if isinstance(value, np.ndarray) and not isinstance(value, Arr):
        value = Arr(value)
    if not isinstance(value, Arr):
        raise InterpretError(f"an array parameter was given {value!r}")
    want = None if written else read_storage(typ.dtype, value.numpy().dtype)
    return value if want is None else native_copy(value, want)


def _key(indices: Sequence[Any]) -> Any:
    """A subscript key the way the body writes it: one index, or a tuple."""
    return indices[0] if len(indices) == 1 else tuple(indices)


@dataclass
class _Member:
    """One statement as the walk of the loop tree has reached it.

    ``env`` gives its loop variables down to the walk's level, and the rest of
    its domain is ``space``, with those fixed, until every reflected bound in
    ``pending`` has been read; from then on it is ``points``, the coordinates
    of its instances below the walk's level, and ``space`` is ``None``.
    ``levels`` gives the loop each bound of the domain bounds, by its depth, or
    the statement's own depth for a bound of none. ``read`` gives the bounds
    read so far, by parameter, which a reduction's domain repeats for the
    loops around it (see :meth:`_Run.reduction`).
    """

    index: int
    stmt: Stmt
    order: tuple[int, ...]
    env: dict[str, int]
    space: isl.Set | None
    pending: dict[str, Any]
    levels: dict[str, int]
    points: list[tuple[int, ...]] | None = None
    read: dict[str, int] = field(default_factory=dict)

    def at(
        self,
        env: dict[str, int],
        *,
        space: isl.Set | None = None,
        points: list[tuple[int, ...]] | None = None,
    ) -> _Member:
        """This statement one iteration further down, at ``env``."""
        return replace(
            self,
            env=env,
            space=space,
            pending=dict(self.pending),
            points=points,
            read=dict(self.read),
        )


class _Run:
    """One interpretation: the arguments, the sizes, and the values of names."""

    def __init__(self, term: Term, arguments: Mapping[str, Any]) -> None:
        self.term = term
        written = {stmt.assignee.array for stmt in term.stmts}
        self.arrays: dict[str, Arr] = {}
        self.scalars: dict[str, Any] = {}
        #: The dtype each written array's elements are read in where it is
        #: stored otherwise: an ``int32`` array of ``Fin[m]`` is read as
        #: ``int64``, as the native run reads it (#121).
        self.read_as: dict[str, np.dtype] = {}
        for name, typ in term.params:
            if name not in arguments:
                raise InterpretError(f"no argument was given for {name}")
            value = arguments[name]
            if isinstance(typ, ArrType):
                self.arrays[name] = _storage(value, typ, name in written)
                if name in written:
                    want = read_storage(typ.dtype, self.arrays[name].numpy().dtype)
                    if want is not None:
                        self.read_as[name] = want
            else:
                self.scalars[name] = native_scalar(typ, value)
        for name, (counts, offsets) in declared_layout(
            term.params, term.offsets
        ).items():
            array = self.arrays[name]
            if array.is_ragged:
                self.arrays[name] = array.through(
                    self.arrays.get(counts) if counts else None,
                    self.arrays.get(offsets) if offsets else None,
                )
        self.sizes = resolve_sizes(dict(term.params), arguments)
        for name, typ in term.temporaries:
            self.arrays[name] = self._temporary(name, typ)
        self.reflected = dict(term.reflected)
        #: The arrays the term writes, which a bound can read.
        self.written = written
        #: The work allowed (see :func:`interpret`) and the work done so far.
        self.limit: int | None = None
        self.spent = 0

    def _temporary(self, name: str, typ: ArrType) -> Arr:
        """A buffer for one of a program's own arrays, at the arguments' sizes.

        Zeros, which is what the statement that begins its life writes anyway
        (:mod:`loopty.compose`); the interpreter runs that statement like any
        other. It is stored as the native ``Arr.zeros_like`` has to store it
        to hold what the compiled one holds
        (:func:`loopty.contract.native_storage`): ``float32`` for
        ``np.float32``, so that what is written into it is rounded here as it
        is in both runs, ``complex128`` for ``np.complex128``, ``bool`` for
        ``Bool``, and ``int64`` for an integral sort.
        """
        if any(typ.ragged):
            raise InterpretError(f"the temporary {name} is ragged")
        try:
            shape = tuple(int(evaluate(axis, dict(self.sizes))) for axis in typ.axes)
        except Exception as exc:  # noqa: BLE001 - said as the reason
            raise InterpretError(
                f"the temporary {name} has no shape at the sizes "
                f"{dict(self.sizes)}: {exc}"
            ) from exc
        dtype = native_storage(typ.dtype) or np.dtype(np.float64)
        return Arr(np.zeros(shape, dtype=dtype))

    # {{{ running

    def spend(self) -> None:
        """Count one statement instance or one reduction term against the limit."""
        self.spent += 1
        if self.limit is not None and self.spent > self.limit:
            raise TooLarge(
                f"more than {self.limit} statement instances and reduction terms"
            )

    def enumerate(
        self, space: isl.Set, positions: Sequence[int]
    ) -> list[tuple[int, ...]]:
        """:func:`_enumerate` under the limit, checked before a point is visited.

        Collecting a domain's points is itself the work the limit bounds, so a
        domain with more points than the limit has left is refused first. The
        check is by the domain's bounding box, the bound isl gives without
        visiting points. A box can hold more points than its domain (a triangle
        fills half of one), so an input near the limit may be refused that
        would have fit; one far past it is refused at once, not after its
        points have been collected.
        """
        if self.limit is not None and not space.is_empty() and space.is_bounded():
            if _box_volume(space) > self.limit - self.spent:
                raise TooLarge(
                    f"more than {self.limit} statement instances and reduction terms"
                )
        return _enumerate(space, positions)

    def run(self, limit: int | None) -> dict[str, np.ndarray]:
        """Every instance of every statement, in the order of the source schedule.

        The loop tree is walked as the body runs it (:meth:`block`): the
        statements of one level in source order, and a loop's iterations in
        order, each with what the loop holds at that iteration.
        """
        self.limit = limit
        known = self._known()
        members = [
            self.member(index, stmt, known)
            for index, stmt in enumerate(self.term.stmts)
        ]
        if limit is not None:
            # The domains no array bounds have their points fixed by the sizes,
            # so their boxes bound the work before anything runs, as they
            # always did; the others are counted as their loops are reached.
            boxes = sum(
                _box_volume(member.space)
                for member in members
                if not member.pending
                and not member.space.is_empty()
                and member.space.is_bounded()
            )
            if boxes > limit:
                raise TooLarge(
                    f"more than {limit} statement instances and reduction terms"
                )
        self.block(members, 0)
        for flag, message in self.term.checks:
            # A checked point of a program (loopty.compose): its statement set
            # the flag where a cell failed, every later statement was guarded
            # by it, and the run stops here as the compiled one does.
            if self.arrays[flag].numpy().reshape(-1)[0]:
                raise CheckFailed(message)
        return {
            name: self.arrays[name].numpy()
            for name in dict.fromkeys(stmt.assignee.array for stmt in self.term.stmts)
        }

    def member(self, index: int, stmt: Stmt, known: Mapping[str, Any]) -> _Member:
        """One statement at the start of the run: its domain at the known sizes.

        Its dimensions are its loop variables, outermost first. Every
        parameter of the domain is a size or a scalar, which is fixed here, or
        a reflected one, which is left until the loop it bounds starts (see
        :meth:`loop`); one that is neither has no value anywhere. A reflected
        bound that reads an array the term writes is read once for each loop
        and each sum it bounds, where that loop or sum starts, as the body
        reads it (:meth:`read_apart`).
        """
        names = tuple(stmt.domain.get_var_names(isl.dim_type.set))
        if names != tuple(stmt.inames):
            raise InterpretError(
                f"the domain of {stmt.id} has the dimensions {', '.join(names)}, "
                f"and the statement runs in the loops {', '.join(stmt.inames)}"
            )
        stmt, given = self.read_apart(stmt)
        domain = stmt.domain
        space = domain
        for position, name in enumerate(names):
            if name in known:
                space = space.fix_val(isl.dim_type.set, position, int(known[name]))
        pending: dict[str, Any] = {}
        levels: dict[str, int] = {}
        for position, name in enumerate(domain.get_var_names(isl.dim_type.param)):
            if name in known:
                space = _fix_param(space, name, known[name])
            elif name in self.reflected:
                expr = self.reflected[name]
                bounded = [
                    level
                    for level in range(len(names))
                    if bounds_dimension(domain, level, position)
                ]
                pending[name] = expr
                levels[name] = given.get(
                    name, bounded[0] if bounded else len(names)
                )
            else:
                raise InterpretError(
                    f"the domain {domain} has a parameter {name} that no argument "
                    "gives a value"
                )
        return _Member(
            index=index,
            stmt=stmt,
            order=stmt.order or (0,) * (len(stmt.inames) + 1),
            env={},
            space=space,
            pending=pending,
            levels=levels,
        )

    def read_apart(self, stmt: Stmt) -> tuple[Stmt, dict[str, int]]:
        """``stmt`` with a written bound read where each loop and sum starts.

        A reflected bound is one parameter of a domain, read once. The body
        reads it where each loop over it starts, ``for k in val.dom[r]``
        inside ``for j in val.dom[r]`` as well as the loop over ``j``, and
        where each sum over it starts, ``reduce_sum(val[r, k] for k in
        val.dom[r])`` inside the loop over ``j``. Those readings agree unless
        the term writes what the bound reads in between, so a bound that
        reads an array the term writes gets a parameter of its own for each
        of them (#87). The outermost loop it bounds keeps the parameter, and
        each loop inside that one gets a copy, read where that loop starts
        (:meth:`read_bounds`). The sums of the statement get one more copy,
        which no loop reads, so :meth:`reduction` reads it where the sum
        starts; a sum's domain repeats the bounds of the loops around it, and
        those keep the copy of their loop, the value the loop was read at.

        A constraint goes with the innermost loop or binder it mentions. One
        on the parameters alone stays with the outermost reading, unless it
        names the binder of an enclosing sum, which is that sum's.

        The loops and the sums a bound bounds, and their constraints, are read
        off the sets before a guard or a clause narrowed them, the
        statement's ``loop_domain`` and each sum's: isl simplifies the
        narrowed set, and ``when(k == j)`` inside the two loops above leaves
        no constraint of the bound of ``k`` in it, so ``k`` had no reading of
        its own and ran past its row's new length (#111). What the guard or
        the clause adds is the narrowed set's gist in the other, which is
        intersected back once the bounds are read apart. A gist that still
        names a bound read apart is refused (:func:`_narrowing`): which loop's
        reading it is cannot be told.

        Returns the statement, and the loop each reading of a written bound
        is read where it starts, by depth, which the narrowed set may no
        longer show: ``j < nl_cnt_r__k`` is about ``k`` once ``k = j``.
        """
        domain = stmt.loop_domain if stmt.loop_domain is not None else stmt.domain
        names = tuple(stmt.inames)
        sums = reductions_in((stmt.expr, stmt.guard))
        # The parameter each loop reads a bound as, by depth, and the sums'.
        loops: dict[str, dict[int, str]] = {}
        summed: dict[str, str] = {}
        for position, name in enumerate(domain.get_var_names(isl.dim_type.param)):
            expr = self.reflected.get(name)
            if expr is None:
                continue
            if not {access.array for access in accesses_in(expr)} & self.written:
                continue
            bounded = [
                level
                for level in range(len(names))
                if bounds_dimension(domain, level, position)
            ]
            if not bounded:
                continue
            loops[name] = {bounded[0]: name}
            for level in bounded[1:]:
                loops[name][level] = self.copy_of(name, f"{name}__{names[level]}")
            if sums:
                summed[name] = self.copy_of(name, f"{name}__sum")
        if not loops:
            return stmt, {}
        levels = {
            reading: level
            for copies in loops.values()
            for level, reading in copies.items()
        }

        def of_loops(domain: isl.Set) -> isl.Set:
            for name, copies in loops.items():

                def of_loop(
                    constraint: isl.Constraint, name: str = name, copies: Any = copies
                ) -> str:
                    level = _innermost(constraint)
                    return name if level is None else copies.get(level, name)

                domain = _read_apart(domain, name, of_loop)
            return domain

        domain = of_loops(domain)
        if stmt.loop_domain is not None:
            domain = _intersected(
                domain, _narrowing(stmt.domain, stmt.loop_domain, loops, stmt.id)
            )
        if not summed:
            return replace(stmt, domain=domain), levels

        known = {*self.sizes, *self.scalars, *self.reflected}

        def rewrite(node: Reduction) -> isl.Set:
            """The domain of one sum of the statement, its readings apart."""
            out = node.domain if node.loop_domain is None else node.loop_domain
            own = out.dim(isl.dim_type.set) - len(node.inames)
            binders = [
                position
                for position, param in enumerate(out.get_var_names(isl.dim_type.param))
                if param not in known
            ]
            for name in summed:
                if out.find_dim_by_name(isl.dim_type.param, name) < 0:
                    continue
                copies = loops[name]

                def of_sum(
                    constraint: isl.Constraint, name: str = name, copies: Any = copies
                ) -> str:
                    level = _innermost(constraint)
                    if level is not None:
                        return summed[name] if level >= own else copies.get(level, name)
                    # A binder of an enclosing sum is a parameter here, and the
                    # bound of it is that sum's reading.
                    if any(
                        not constraint.get_coefficient_val(
                            isl.dim_type.param, k
                        ).is_zero()
                        for k in binders
                    ):
                        return summed[name]
                    return name

                out = _read_apart(out, name, of_sum)
            if node.loop_domain is not None:
                out = _intersected(
                    out,
                    _narrowing(
                        node.domain,
                        node.loop_domain,
                        {name: {} for name in summed},
                        f"the sum over {', '.join(node.inames)} in {stmt.id}",
                    ),
                )
            return out

        return (
            replace(
                stmt,
                domain=domain,
                expr=_with_domains(stmt.expr, rewrite),
                guard=_with_domains(stmt.guard, rewrite),
            ),
            levels,
        )

    def copy_of(self, name: str, spelled: str) -> str:
        """A fresh parameter that reads what the reflected bound ``name`` reads."""
        taken = {*self.sizes, *self.scalars, *self.reflected}
        while spelled in taken:
            spelled += "_"
        self.reflected[spelled] = self.reflected[name]
        return spelled

    def block(self, members: Sequence[_Member], level: int) -> None:
        """The statements and loops of one level of the loop tree, in order.

        ``members`` share the loops above ``level`` and are at one iteration
        of each. They are taken in the order their positions at this level
        give, the statements that end here as they come and the ones in a
        loop of this level together, as that loop.
        """
        items: dict[tuple[int, bool], list[_Member]] = {}
        for member in members:
            nested = len(member.stmt.inames) > level
            position = member.order[level] if level < len(member.order) else 0
            items.setdefault((position, nested), []).append(member)
        for (_position, nested), group in sorted(items.items(), key=lambda i: i[0]):
            if nested:
                self.loop(group, level)
            else:
                for member in group:
                    self.instance(member)

    def loop(self, members: Sequence[_Member], level: int) -> None:
        """One loop, from where it starts: its bounds, then its iterations.

        A reflected bound of a statement in the loop is read now, when the
        loop is entered, if it bounds this loop or one around it and its term
        has every value it needs; once a statement has no bound left to read,
        its points below here are enumerated in one go. A statement that still
        has one contributes the values this loop can take with those bounds
        projected out, which can only give it more candidates, and the exact
        question is asked where its instance would run. Then every iteration
        runs, in order, with the statements that have a point there.
        """
        values: set[int] = set()
        prepared: list[tuple[_Member, Any]] = []
        for member in members:
            if member.points is None:
                self.read_bounds(member, level)
            if member.points is None and not member.pending:
                count = member.space.dim(isl.dim_type.set)
                member.points = self.enumerate(member.space, range(level, count))
                member.space = None
            if member.points is not None:
                tails: dict[int, list[tuple[int, ...]]] = {}
                for point in member.points:
                    tails.setdefault(point[0], []).append(point[1:])
                values.update(tails)
                prepared.append((member, tails))
                continue
            assert member.space is not None
            shadow = member.space.project_out(
                isl.dim_type.set,
                level + 1,
                member.space.dim(isl.dim_type.set) - level - 1,
            )
            for name in member.pending:
                shadow = shadow.project_out(
                    isl.dim_type.param,
                    shadow.find_dim_by_name(isl.dim_type.param, name),
                    1,
                )
            candidates = {value for (value,) in self.enumerate(shadow, [level])}
            values.update(candidates)
            prepared.append((member, candidates))
        for value in sorted(values):
            inside: list[_Member] = []
            for member, at in prepared:
                if value not in at:
                    continue
                env = {**member.env, member.stmt.inames[level]: value}
                if isinstance(at, dict):
                    inside.append(member.at(env, points=at[value]))
                else:
                    assert member.space is not None
                    space = member.space.fix_val(isl.dim_type.set, level, value)
                    inside.append(member.at(env, space=space))
            self.block(inside, level + 1)

    def read_bounds(self, member: _Member, level: int) -> None:
        """Read the reflected bounds of ``member`` that are due at ``level``.

        A bound is due at the loop it bounds, the outermost one if several,
        and at any loop inside that one; it is read once its term has every
        value it needs. A bound of no loop is read where the statement runs.
        """
        assert member.space is not None
        for name, expr in list(member.pending.items()):
            if member.levels[name] > level:
                continue
            try:
                value = self.value(expr, member.env)
            except _Unknown:
                continue
            member.space = _fix_param(member.space, name, value)
            member.read[name] = value
            del member.pending[name]

    def instance(self, member: _Member) -> None:
        """Run one statement at the point the walk has reached, if it has one.

        Its expressions see the bounds its loops were read at as well as its
        loop variables, so that a sum inside the loops, whose domain repeats
        their bounds, takes them as the loops did (:meth:`reduction`).
        """
        if member.points is not None:
            if not member.points:
                return
        else:
            assert member.space is not None
            self.read_bounds(member, len(member.stmt.inames))
            if member.pending:
                raise InterpretError(
                    "the domain parameters "
                    + ", ".join(member.pending)
                    + " are still unknown once every loop variable has a value"
                )
            if member.space.is_empty():
                return
        self.spend()
        stmt, point = member.stmt, {**member.read, **member.env}
        if stmt.guard is not None and not self.truth(stmt.guard, point):
            return
        value = self.value(stmt.expr, point)
        indices = [self.index(i, point) for i in stmt.assignee.indices]
        self.arrays[stmt.assignee.array][_key(indices)] = value

    def _known(self) -> dict[str, Any]:
        """The sizes and the numeric scalars, which a domain may name.

        A domain parameter is an integer to isl. A scalar that is not a whole
        number is kept as it is, so that fixing it refuses with its value
        (:func:`_fix_param`) rather than reporting the parameter unknown.
        """
        out: dict[str, Any] = dict(self.sizes)
        for name, value in self.scalars.items():
            if isinstance(value, bool | np.bool_):
                continue
            if isinstance(value, int | np.integer):
                out.setdefault(name, int(value))
            elif isinstance(value, float | np.floating):
                out.setdefault(name, value)
        return out

    # }}}

    # {{{ the points of a domain

    def points(
        self, domain: isl.Set, known: Mapping[str, int]
    ) -> Iterator[dict[str, int]]:
        """The points of ``domain`` in lexicographic order, as name-value maps.

        A set dimension ``known`` names is fixed there, and so is a parameter;
        every other parameter has to be a reflected one, which is fixed as soon
        as the loop variables its term mentions have values. Until then the
        dimension being walked is enumerated with the unknown parameters
        projected out, which can only give it more candidates, and the leaf
        asks the exact question with every parameter fixed. Once every
        parameter is fixed the rest is enumerated in one go.
        """
        names = domain.get_var_names(isl.dim_type.set)
        space = domain
        env = dict(known)
        for position, name in enumerate(names):
            if name in env:
                space = space.fix_val(isl.dim_type.set, position, int(env[name]))
        pending: dict[str, Any] = {}
        for name in domain.get_var_names(isl.dim_type.param):
            if name in env:
                space = _fix_param(space, name, env[name])
            elif name in self.reflected:
                pending[name] = self.reflected[name]
            else:
                raise InterpretError(
                    f"the domain {domain} has a parameter {name} that no argument "
                    "gives a value"
                )
        free = [k for k, name in enumerate(names) if name not in env]
        yield from self._walk(space, names, free, 0, env, pending)

    def _walk(
        self,
        space: isl.Set,
        names: Sequence[str],
        free: Sequence[int],
        level: int,
        env: dict[str, int],
        pending: dict[str, Any],
    ) -> Iterator[dict[str, int]]:
        """The points below one level of the walk; see :meth:`points`."""
        still: dict[str, Any] = {}
        for name, expr in pending.items():
            try:
                value = self.value(expr, env)
            except _Unknown:
                still[name] = expr
                continue
            space = _fix_param(space, name, value)
        if level == len(free):
            if still:
                raise InterpretError(
                    "the domain parameters "
                    + ", ".join(still)
                    + " are still unknown once every loop variable has a value"
                )
            if not space.is_empty():
                yield {name: env[name] for name in names}
            return
        if not still:
            for coordinates in self.enumerate(space, free[level:]):
                point = dict(env)
                point.update(
                    (names[k], value)
                    for k, value in zip(free[level:], coordinates, strict=True)
                )
                yield {name: point[name] for name in names}
            return
        # The dimensions after this one, and the parameters not known yet, are
        # projected out: what is left bounds this dimension alone, and bounds
        # it loosely, because a candidate the exact set does not have is
        # dropped at the leaf.
        position = free[level]
        shadow = space.project_out(
            isl.dim_type.set, position + 1, len(names) - position - 1
        )
        for name in still:
            shadow = shadow.project_out(
                isl.dim_type.param,
                shadow.find_dim_by_name(isl.dim_type.param, name),
                1,
            )
        for (value,) in self.enumerate(shadow, [position]):
            yield from self._walk(
                space.fix_val(isl.dim_type.set, position, value),
                names,
                free,
                level + 1,
                {**env, names[position]: value},
                still,
            )

    # }}}

    # {{{ expressions

    def truth(self, condition: Any, env: Mapping[str, Any]) -> bool:
        """Whether a guard holds at one instance."""
        return bool(self.value(condition, env))

    def index(self, expr: Any, env: Mapping[str, Any]) -> Any:
        """One subscript, as the body would compute it."""
        return self.value(expr, env)

    def read(self, array: str, indices: Sequence[Any], env: Mapping[str, Any]) -> Any:
        """One array element, read the way the native run reads it."""
        if array not in self.arrays:
            raise InterpretError(f"{array} is subscripted and is not an array")
        key = _key([self.index(i, env) for i in indices])
        value = self.arrays[array][key]
        read_in = self.read_as.get(array)
        if read_in is not None and isinstance(value, np.generic):
            return read_in.type(value)
        return value

    def value(self, node: Any, env: Mapping[str, Any]) -> Any:
        """The value of one expression at one point, by numpy's arithmetic."""
        if isinstance(node, bool | int | float | complex | np.generic):
            return node
        if isinstance(node, Reduction):
            return self.reduction(node, env)
        if isinstance(node, Access):
            return self.read(node.array, node.indices, env)
        if isinstance(node, prim.Subscript):
            aggregate = node.aggregate
            if not isinstance(aggregate, prim.Variable):
                raise InterpretError(f"cannot subscript {render(aggregate)}")
            index = node.index
            indices = index if isinstance(index, tuple) else (index,)
            return self.read(aggregate.name, indices, env)
        if isinstance(node, prim.Variable):
            return self.variable(node.name, env)
        if isinstance(node, prim.Sum):
            return _fold(operator.add, [self.value(c, env) for c in node.children])
        if isinstance(node, prim.Product):
            return _fold(operator.mul, [self.value(c, env) for c in node.children])
        if isinstance(node, prim.BitwiseXor):
            return _fold(operator.xor, [self.value(c, env) for c in node.children])
        if isinstance(node, prim.LeftShift | prim.RightShift):
            shift = (
                operator.lshift if isinstance(node, prim.LeftShift) else operator.rshift
            )
            return shift(self.value(node.shiftee, env), self.value(node.shift, env))
        for kind, apply in _BINARY:
            if isinstance(node, kind):
                return apply(
                    self.value(node.numerator, env), self.value(node.denominator, env)
                )
        if isinstance(node, prim.Power):
            return self.value(node.base, env) ** self.value(node.exponent, env)
        if isinstance(node, prim.Comparison):
            compare = _COMPARISONS.get(node.operator)
            if compare is None:
                raise InterpretError(f"no meaning for the comparison {node.operator}")
            return compare(self.value(node.left, env), self.value(node.right, env))
        if isinstance(node, prim.LogicalAnd):
            return all(self.truth(child, env) for child in node.children)
        if isinstance(node, prim.LogicalOr):
            return any(self.truth(child, env) for child in node.children)
        if isinstance(node, prim.LogicalNot):
            return not self.truth(node.child, env)
        if isinstance(node, prim.If):
            branch = node.then if self.truth(node.condition, env) else node.else_
            return self.value(branch, env)
        if isinstance(node, prim.Min | prim.Max):
            pick = builtins.min if isinstance(node, prim.Min) else builtins.max
            return pick(self.value(child, env) for child in node.children)
        if isinstance(node, Abs):
            return builtins.abs(self.value(node.operand, env))
        if isinstance(node, prim.Call):
            return self.call(node, env)
        if isinstance(node, tuple):
            return tuple(self.value(item, env) for item in node)
        raise InterpretError(
            f"no numpy meaning for {type(node).__name__} in the term: "
            f"{_text(node)}"
        )

    def variable(self, name: str, env: Mapping[str, Any]) -> Any:
        """A loop or binder variable, a scalar argument, or a size."""
        if name in env:
            return env[name]
        if name in self.scalars:
            return self.scalars[name]
        if name in self.sizes:
            return self.sizes[name]
        raise _Unknown(name)

    def call(self, node: prim.Call, env: Mapping[str, Any]) -> Any:
        """A call of a library function, by the name the term gives it."""
        function = node.function
        name = function.name if isinstance(function, prim.Variable) else None
        numpy_function = _FUNCTIONS.get(name) if name is not None else None
        if numpy_function is None:
            raise InterpretError(
                f"no numpy counterpart for the call {_text(node)}; the "
                f"interpreter knows {', '.join(sorted(_FUNCTIONS))}"
            )
        return numpy_function(*(self.value(arg, env) for arg in node.parameters))

    def reduction(self, node: Reduction, env: Mapping[str, Any]) -> Any:
        """A reduction, summed in the order the native ``reduce_sum`` sums it.

        The domain's dimensions are the statement's inames, fixed by ``env``,
        followed by the reduction's own. An enclosing reduction's binders are
        parameters of the domain, and ``env`` gives them too. So does a bound
        of the statement's loops that the domain repeats, at the value it had
        where its loop started: ``0 <= j < nl_cnt_r`` for a sum inside ``for
        j in val.dom[r]`` holds at the ``j`` the loop reached, whatever the
        loop has written into ``cnt[r]`` since. A bound of the sum's own
        binders is read now, where the native ``reduce_sum`` reads it, the
        row's length in ``reduce_sum(val[r, k] for k in val.dom[r])`` inside
        ``for j in val.dom[r]`` included, which is a parameter of its own
        (:meth:`read_apart`).
        """
        if node.op != "sum":
            raise InterpretError(f"no meaning for a reduction of kind {node.op!r}")
        known = {**self._known(), **env}
        terms = []
        for point in self.points(node.domain, known):
            self.spend()
            terms.append(self.value(node.body, {**env, **point}))
        return builtins.sum(terms)

    # }}}


def _innermost(constraint: isl.Constraint) -> int | None:
    """The innermost set dimension a constraint mentions, or ``None``."""
    mentioned = [
        level
        for level in range(constraint.get_space().dim(isl.dim_type.set))
        if not constraint.get_coefficient_val(isl.dim_type.set, level).is_zero()
    ]
    return max(mentioned, default=None)


def _narrowing(
    narrowed: isl.Set,
    loops: isl.Set,
    apart: Mapping[str, Any],
    where: str,
) -> isl.Set:
    """What a guard or a clause adds to ``loops``: ``narrowed``'s gist in it.

    ``narrowed`` is ``loops`` intersected with what the guard states, and
    the gist is a set that gives ``narrowed`` back when it is intersected
    with ``loops``, with what ``loops`` already says left out: the loops'
    bounds, which :meth:`_Run.read_apart` reads apart. One that still names
    a bound in ``apart`` is one isl could not leave out, and which reading
    of the bound it is about cannot be told, so it is refused.
    """
    narrowed = narrowed.align_params(loops.get_space())
    loops = loops.align_params(narrowed.get_space())
    gist = narrowed.gist(loops)
    for name in apart:
        position = gist.find_dim_by_name(isl.dim_type.param, name)
        if position >= 0 and gist.involves_dims(isl.dim_type.param, position, 1):
            raise InterpretError(
                f"the guard of {where} narrows it by {gist}, which names the "
                f"bound {name}: the term writes what {name} reads, the loops "
                "it bounds read it apart, and the interpreter cannot tell which "
                "reading the guard is about"
            )
    return gist


def _intersected(first: isl.Set, second: isl.Set) -> isl.Set:
    """``first`` intersected with ``second``, their parameters aligned."""
    first = first.align_params(second.get_space())
    return first.intersect(second.align_params(first.get_space()))


def _read_apart(domain: isl.Set, name: str, reading: Any) -> isl.Set:
    """``domain`` with the parameter ``name`` read as ``reading(constraint)``.

    Each constraint that mentions ``name`` is stated of the parameter
    ``reading`` names for it instead, ``name`` itself or a parameter added
    here; the other constraints are kept as they are. A domain with local
    variables (a stride, say) is refused: its constraints are not independent
    of one another, and restating them one by one could lose what ties them.
    """
    position = domain.find_dim_by_name(isl.dim_type.param, name)
    targets: dict[str, None] = {}
    local = False

    def survey(basic: isl.BasicSet) -> None:
        nonlocal local
        local = local or basic.dim(isl.dim_type.div) > 0
        for constraint in basic.get_constraints():
            if constraint.get_coefficient_val(isl.dim_type.param, position).is_zero():
                continue
            target = reading(constraint)
            if target != name:
                targets[target] = None

    domain.foreach_basic_set(survey)
    if not targets:
        return domain
    if local:
        raise InterpretError(
            f"the domain {domain} reads the bound {name} where more than one "
            "loop or sum starts, and its constraints have local variables, "
            "which the interpreter cannot state of each reading apart"
        )
    count = domain.dim(isl.dim_type.param)
    widened = domain.add_dims(isl.dim_type.param, len(targets))
    index: dict[str, int] = {}
    for offset, target in enumerate(targets):
        widened = widened.set_dim_name(isl.dim_type.param, count + offset, target)
        index[target] = count + offset
    pieces: list[isl.BasicSet] = []

    def rebuild(basic: isl.BasicSet) -> None:
        piece = isl.BasicSet.universe(basic.get_space())
        for constraint in basic.get_constraints():
            coefficient = constraint.get_coefficient_val(isl.dim_type.param, position)
            if not coefficient.is_zero():
                target = reading(constraint)
                if target != name:
                    constraint = constraint.set_coefficient_val(
                        isl.dim_type.param, position, 0
                    )
                    constraint = constraint.set_coefficient_val(
                        isl.dim_type.param, index[target], coefficient
                    )
            piece = piece.add_constraint(constraint)
        pieces.append(piece)

    widened.foreach_basic_set(rebuild)
    out = isl.Set.empty(widened.get_space())
    for piece in pieces:
        out = out.union(piece)
    return out


def _with_domains(node: Any, rewrite: Any) -> Any:
    """``node`` with each reduction's domain replaced by ``rewrite(reduction)``."""
    if isinstance(node, Reduction):
        return replace(
            node, domain=rewrite(node), body=_with_domains(node.body, rewrite)
        )
    if isinstance(node, prim.ExpressionNode):
        args = init_args(node)
        rebuilt = tuple(_with_domains(arg, rewrite) for arg in args)
        if all(new is old for new, old in zip(rebuilt, args, strict=True)):
            return node
        return type(node)(*rebuilt)
    if isinstance(node, tuple):
        return tuple(_with_domains(item, rewrite) for item in node)
    return node


def _fold(apply: Any, values: Sequence[Any]) -> Any:
    """``((a op b) op c) ...``: Python's order for a chain of one operator."""
    out = values[0]
    for value in values[1:]:
        out = apply(out, value)
    return out


def _fix_param(space: isl.Set, name: str, value: Any) -> isl.Set:
    """``space`` with the parameter ``name`` fixed to an integer value."""
    if isinstance(value, bool) or not isinstance(value, int | np.integer):
        if isinstance(value, float | np.floating) and float(value).is_integer():
            value = int(value)
        else:
            shown = value.item() if isinstance(value, np.generic) else value
            raise InterpretError(
                f"the domain parameter {name} is {shown!r}, which is not an integer"
            )
    position = space.find_dim_by_name(isl.dim_type.param, name)
    return space.fix_val(isl.dim_type.param, position, int(value))


def _enumerate(space: isl.Set, positions: Sequence[int]) -> list[tuple[int, ...]]:
    """The distinct values of the dimensions at ``positions``, sorted."""
    if space.is_empty():
        return []
    if not space.is_bounded():
        raise InterpretError(
            f"the domain {space} is unbounded at these sizes, so its points "
            "cannot be enumerated"
        )
    found: set[tuple[int, ...]] = set()

    def visit(point: Any) -> None:
        found.add(
            tuple(
                point.get_coordinate_val(isl.dim_type.set, k).to_python()
                for k in positions
            )
        )

    space.foreach_point(visit)
    return sorted(found)


def _box_volume(space: isl.Set) -> int:
    """How many points the bounding box of a bounded, non-empty set holds."""
    volume = 1
    for position in range(space.dim(isl.dim_type.set)):
        low = space.dim_min_val(position).to_python()
        high = space.dim_max_val(position).to_python()
        volume *= high - low + 1
    return volume


def _text(node: Any) -> str:
    """A node rendered for a message, or its ``repr`` when lanky cannot."""
    try:
        return render(node)
    except Exception:  # noqa: BLE001 - a message, not a result
        return repr(node)
