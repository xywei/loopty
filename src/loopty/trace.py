"""Tracing: run the body once, record what it did.

Kernel bodies are not parsed. They are executed once against symbolic arrays:
indexing a proxy builds a pymbolic subscript, assigning to one records a
:class:`~loopty.term.Stmt` tagged with the calling frame's file and line,
iterating a proxy's ``.dom`` yields a single fresh iname and pushes its bound
onto the enclosing domain, and ``loopty.reduce_sum`` over such a domain becomes
a :class:`~loopty.term.Reduction`. Source maps are frame line numbers, so there is
no AST pass and no span bookkeeping. The body that traces is the body that runs:
the same source, given real arrays, is the reference implementation.

Execution-based tracing needs two explicit symbolic constructs where ordinary
Python syntax would otherwise force a concrete decision.

``with when(cond):`` is the guard, because a Python ``if`` on a symbolic value
cannot be traced: tracing would have to pick a branch, and the value is not
known until the kernel runs. A symbolic condition asked for its truth value
raises :class:`TraceError` naming ``when`` as the fix. Under tracing ``when``
pushes the condition onto the guard stack, and the statements recorded inside
carry it and have it intersected into their domain when it is affine, so a
guarded access is in bounds exactly where it is executed. Under plain
``python`` it masks the writes of the block rather than skipping them, which is
what keeps one body serving as both the specification and the reference run.

``loopty.reduce_sum`` marks the reduction explicitly, so that the reduced domain
is known rather than guessed from an accumulator loop. It takes an ordinary
generator expression; internally Lanky's binder tracing supplies the bound
variable and its name, and this module only has to say what a symbolic ``.dom``
yields when the binder tracer asks it for a point.

What the tracer produces is a :class:`~loopty.term.Term`: parameters with their
array types, the free size parameters, the statements with their isl domains,
and the postcondition. Nothing here decides anything; the obligations are read
off the term by :mod:`loopty.typing`.
"""

from __future__ import annotations

import dis
import sys
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np
import pymbolic.primitives as prim
from lanky.prelude import FinType, Refined
from lanky.terms import (
    Subscript,
    Sum,
    SymbolicBoolError,
    Var,
    init_args,
    structurally_equal,
)

from loopty.arr import Arr, ArrSpec
from loopty.flow import domain_set, expr_text
from loopty.idx import Reflections
from loopty.term import Access, ArrType, Reduction, Stmt, Term

__all__ = [
    "EXACTNESS_ORDER",
    "TraceError",
    "Tracer",
    "SymArr",
    "SymDom",
    "accesses_in",
    "current_tracer",
    "join_exactness",
    "mask_writes",
    "reduction_exactness",
    "reductions_in",
    "trace",
    "when",
]


def _abandoned_message(inames: Sequence[str]) -> str:
    """What to say about a loop the body left early."""
    names = ", ".join(repr(name) for name in inames)
    plural = "loops" if len(inames) > 1 else "loop"
    return (
        f"the {plural} over {names} was left early. Tracing runs the body once "
        "with every loop taking one generic point, and a 'break' or a 'return' "
        "inside a traced loop skips the point at which the loop level is "
        "closed, so every statement after it is recorded under a loop variable "
        "the body has already left. Write 'with when(condition):' around the "
        "part that should not run instead: it records the condition, narrows "
        "the statement's domain, and masks the writes under plain python."
    )


class TraceError(RuntimeError):
    """A body did something tracing cannot follow, with the fix in the message.

    The archetype is a Python ``if`` on a symbolic value: the message names the
    ``with when(...)`` replacement. Data-dependent ``while``, ``break``,
    ``return`` out of a loop, and Python's builtin ``sum`` over a symbolic
    domain are the other cases.
    """


# {{{ the tracer


@dataclass
class _Loop:
    """One open loop level: its iname, the variable, and its exclusive bound.

    ``owner`` is the iterator that opened the level. A level is closed when
    *that* iterator is asked for its second point, so an iterator closing a
    level it did not open means a loop was abandoned; see
    :meth:`Tracer.leave_loop`.
    """

    iname: str
    var: Var
    bound: Any
    owner: Any = None


class Tracer:
    """The state of one trace: open loops, open guards, recorded statements.

    There is one tracer per :func:`trace` call, pushed on a module-level stack
    so that a proxy reached from anywhere in the body can find it. Statements are
    appended in execution order, which is source order, because the body runs
    once with every loop taking exactly one generic point.
    """

    def __init__(self, name: str, params: Mapping[str, Any]) -> None:
        self.name = name
        self.params = dict(params)
        #: The array parameters with their resolved types, filled in by
        #: :func:`trace` as the proxies are built. A reduction needs them to
        #: read off the exactness class of what it accumulates.
        self.array_types: dict[str, ArrType] = {}
        #: The isl parameters this trace reflects its non-affine bounds into,
        #: allocated once for the whole term so that ``cnt[r]`` is one parameter
        #: in every domain and cannot collide with a size the kernel declares.
        self.reflections = Reflections(params)
        self.loops: list[_Loop] = []
        self.guards: list[Any] = []
        self.stmts: list[Stmt] = []
        self._inames: set[str] = set()
        #: Position of each open block in its parent, outermost first.
        self._path: list[int] = []
        #: How many children each open block has produced so far.
        self._counters: list[int] = [0]

    # {{{ loops

    def fresh_iname(self, hint: str | None) -> str:
        """A loop variable name, from the ``for`` target when one can be read."""
        stem = hint or f"i{len(self._inames)}"
        name = stem
        suffix = 0
        while name in self._inames or name in self.reflections:
            name = f"{stem}_{suffix}"
            suffix += 1
        self._inames.add(name)
        # An iname and a reflected parameter share one isl space, so neither may
        # take a name the other has.
        self.reflections.reserve((name,))
        return name

    def enter_loop(self, bound: Any, hint: str | None, owner: Any = None) -> Var:
        """Open a loop level over ``0 <= i < bound`` and return its variable."""
        name = self.fresh_iname(hint)
        var = Var(name)
        self.loops.append(_Loop(name, var, bound, owner))
        self._path.append(self._position())
        self._counters.append(0)
        return var

    def leave_loop(self, owner: Any = None) -> None:
        """Close the innermost loop level, refusing to close somebody else's.

        A level is popped when its iterator is asked for a second point and
        raises ``StopIteration``. A ``break`` or a ``return`` inside the loop
        means that never happens, and the level stays open: the next statement
        would then be recorded under a loop variable the body has left, and the
        enclosing loop's own ``StopIteration`` would pop the abandoned level
        instead of its own. That mismatch is what is caught here, at the point
        where it happens, so the message can name the loop.
        """
        if owner is not None and self.loops and self.loops[-1].owner is not owner:
            raise TraceError(_abandoned_message([self.loops[-1].iname]))
        self.loops.pop()
        self._path.pop()
        self._counters.pop()

    def _position(self) -> int:
        """The position of the next child of the innermost open block."""
        position = self._counters[-1]
        self._counters[-1] = position + 1
        return position

    @property
    def inames(self) -> tuple[str, ...]:
        """The enclosing inames, outermost first."""
        return tuple(loop.iname for loop in self.loops)

    @property
    def bounds(self) -> tuple[Any, ...]:
        """The exclusive bounds of the enclosing loops, outermost first."""
        return tuple(loop.bound for loop in self.loops)

    # }}}

    # {{{ guards

    def push_guard(self, condition: Any) -> None:
        """Open a ``when`` block."""
        self.guards.append(condition)

    def pop_guard(self) -> None:
        """Close the innermost ``when`` block."""
        self.guards.pop()

    def guard(self) -> Any:
        """The conjunction of the open guards, or ``None``."""
        if not self.guards:
            return None
        if len(self.guards) == 1:
            return self.guards[0]
        return prim.LogicalAnd(tuple(self.guards))

    # }}}

    def domain(self) -> Any:
        """The isl set of the enclosing loop nest, narrowed by affine guards."""
        return domain_set(
            self.inames,
            self.bounds,
            constraints=constraints_of(self.guard()),
            reflections=self.reflections,
        )

    def record(self, assignee: Access, expr: Any, kind: str, where: str) -> Stmt:
        """Append one statement instance family to the term being built."""
        stmt = Stmt(
            id=f"S{len(self.stmts)}",
            inames=self.inames,
            domain=self.domain(),
            assignee=assignee,
            expr=expr,
            kind=kind,
            guard=self.guard(),
            where=where,
            order=(*self._path, self._position()),
        )
        self.stmts.append(stmt)
        return stmt


_TRACERS: list[Tracer] = []


def current_tracer() -> Tracer | None:
    """The tracer of the innermost :func:`trace` call, or ``None``."""
    return _TRACERS[-1] if _TRACERS else None


# }}}


# {{{ guards as isl constraints


def constraints_of(condition: Any) -> tuple[str, ...]:
    """Render a guard as isl constraints, dropping what isl cannot express.

    A guard narrows the statement's domain, which is what makes
    ``with when(i + 1 < n): u[i + 1] = ...`` provably in bounds. A guard that is
    not quasi-affine (a data-dependent test) is dropped, which widens the domain
    and can only make an obligation harder, never falsely discharge one.
    """
    if condition is None:
        return ()
    if isinstance(condition, prim.LogicalAnd):
        out: tuple[str, ...] = ()
        for child in condition.children:
            out += constraints_of(child)
        return out
    if isinstance(condition, prim.Comparison):
        try:
            left = expr_text(condition.left, None, None)
            right = expr_text(condition.right, None, None)
        except ValueError:
            return ()
        operator = condition.operator
        if operator == "!=":
            return ()
        return (f"{left} {operator} {right}",)
    return ()


# }}}


# {{{ symbolic arrays


def _index_tuple(key: Any) -> tuple[Any, ...]:
    """Normalize a subscript key to a tuple of index expressions."""
    return tuple(key) if isinstance(key, tuple) else (key,)


def _subscript(name: str, indices: Sequence[Any]) -> Subscript:
    """Build ``name[i]`` or ``name[i, j]`` as a pymbolic subscript."""
    index = indices[0] if len(indices) == 1 else tuple(indices)
    return Subscript(Var(name), index)


class SymDom:
    """The symbolic iteration domain of one axis of a symbolic array.

    ``arr.dom`` is the outer axis and ``arr.dom[r]`` the fiber over ``r``, whose
    extent for a ragged array is ``cnt[r]``: the bound of the fiber is the bound
    of that row, which is where the dependent sum enters the type. Iterating in
    a ``for`` loop opens a loop level; inside a ``loopty.reduce_sum`` generator,
    iteration binds a reduction variable instead, because Lanky is driving it
    and asks for the point itself.
    """

    __slots__ = ("array", "prefix")

    def __init__(self, array: SymArr, prefix: tuple[Any, ...] = ()) -> None:
        self.array = array
        self.prefix = prefix

    @property
    def axis(self) -> int:
        """Which axis of the array this domain runs over."""
        return len(self.prefix)

    @property
    def bound(self) -> Any:
        """The exclusive upper bound of this axis, as a term."""
        return self.array.axis_bound(self.axis, self.prefix)

    @property
    def size(self) -> Any:
        """The extent of this axis: a term here, an ``int`` on real data.

        Sizes come from the data, so a body that needs one (the interior guard
        of a stencil, say) asks the domain rather than naming a type parameter,
        and the same source works in both modes.
        """
        return self.bound

    def __getitem__(self, index: Any) -> SymDom:
        """The fiber over ``index``: the domain of the next axis."""
        if self.axis + 1 >= self.array.ndim:
            raise TraceError(
                f"{self.array.name} has {self.array.ndim} axes; there is no "
                f"axis {self.axis + 1} to take a fiber of"
            )
        return SymDom(self.array, (*self.prefix, index))

    def __iter__(self) -> _DomIterator:
        """One generic point, bound on the first ``next`` and not before."""
        return _DomIterator(self)

    def __len__(self) -> int:
        """Refuse: the extent is symbolic."""
        raise TraceError(
            f"the size of {self.array.name}.dom is symbolic while tracing; "
            "iterate it instead of asking for its length"
        )

    def __bool__(self) -> bool:
        """Refuse: emptiness of a symbolic domain is not a Python truth value."""
        raise TraceError(
            f"the truth value of {self.array.name}.dom is symbolic; "
            "use 'with when(...)' for a data-dependent condition"
        )

    def __repr__(self) -> str:
        return f"SymDom({self.array.name}, axis={self.axis})"


def _loop_target_name() -> str | None:
    """The name of the ``for`` target in the frame that asked for a point.

    A generic point is handed out from inside ``FOR_ITER``, so the calling
    frame's next store instruction is the loop variable. Reading it keeps the
    inames of the term equal to the names in the source, which is what makes a
    witness or a generated loop nest recognizable. Any surprise (a comprehension,
    a future bytecode layout) gives ``None`` and the tracer invents a name.
    """
    frame = sys._getframe(2)
    try:
        for instruction in dis.get_instructions(frame.f_code):
            if instruction.offset > frame.f_lasti and instruction.opname.startswith(
                "STORE_"
            ):
                name = instruction.argval
                return name if isinstance(name, str) and name.isidentifier() else None
    except Exception:  # pragma: no cover - bytecode reading is best effort
        return None
    return None


class _DomIterator:
    """The iterator of a symbolic domain: exactly one generic point.

    Binding happens on the first ``__next__`` and not in ``__iter__``, because
    Python evaluates and calls ``iter`` on a generator expression's outermost
    iterable before ``reduce_sum`` enters binder tracing; binding early would
    put the binder outside the trace that wants it.
    """

    __slots__ = ("dom", "done", "loop")

    def __init__(self, dom: SymDom) -> None:
        self.dom = dom
        self.done = False
        self.loop = False

    def __iter__(self) -> _DomIterator:
        return self

    def __next__(self) -> Any:
        from lanky.terms import current_trace

        if self.done:
            if self.loop:
                tracer = current_tracer()
                if tracer is not None:
                    tracer.leave_loop(self)
            raise StopIteration
        self.done = True
        binder_trace = current_trace()
        if binder_trace is not None:
            # lanky is driving a generator expression: this is a reduction
            # binder, and lanky names it after the source's loop target.
            return binder_trace.bind(self.dom)
        tracer = current_tracer()
        if tracer is None:
            raise TraceError(
                f"cannot iterate {self.dom!r} outside a trace; under plain "
                "python a kernel iterates a real array's .dom"
            )
        caller = sys._getframe(1).f_code.co_name
        if caller in ("<genexpr>", "<listcomp>", "<setcomp>", "<dictcomp>"):
            raise TraceError(
                f"a comprehension over {self.dom!r} is being driven by Python "
                "itself, which cannot see the reduced domain; write "
                "loopty.reduce_sum(... for j in arr.dom[r]) so that the "
                "reduction and its domain are recorded"
            )
        self.loop = True
        return tracer.enter_loop(self.dom.bound, _loop_target_name(), self)


class SymArr:
    """A symbolic array: the proxy a kernel body sees while it is traced.

    It has the surface of :class:`loopty.arr.Arr` that a kernel is allowed to
    use: ``.dom`` and ``.dom[i]`` for iteration, subscripting for reads, item
    assignment for writes. Everything else is refused, because the body must be
    the same source that runs on real data and anything a symbolic array cannot
    answer is a question the kernel should not be asking.
    """

    def __init__(self, name: str, arrtype: ArrType, tracer: Tracer) -> None:
        self.name = name
        self.type = arrtype
        self.tracer = tracer

    @property
    def ndim(self) -> int:
        """Number of index axes."""
        return self.type.ndim

    def axis_bound(self, axis: int, prefix: Sequence[Any] = ()) -> Any:
        """The exclusive bound of ``axis``, given the indices of the axes before it.

        A ragged axis names a counts array, and its bound at row ``r`` is
        ``cnt[r]``: the type says where the size comes from, and the index
        expression of the enclosing axis says which row.
        """
        size = self.type.axes[axis]
        if not self.type.ragged[axis]:
            return size
        counts = size.name if isinstance(size, prim.Variable) else str(size)
        row = prefix[axis - 1] if axis >= 1 and len(prefix) >= axis else 0
        return prim.Subscript(Var(counts), row)

    @property
    def dom(self) -> SymDom:
        """The iteration domain of the outer axis."""
        return SymDom(self, ())

    def __getitem__(self, key: Any) -> Subscript:
        """Read: build the index expression ``name[...]``."""
        return _subscript(self.name, _index_tuple(key))

    def __setitem__(self, key: Any, value: Any) -> None:
        """Write: record a statement at the caller's file and line."""
        indices = _index_tuple(key)
        tracer = self.tracer
        frame = sys._getframe(1)
        where = f"{frame.f_code.co_filename.rsplit('/', 1)[-1]}:{frame.f_lineno}"
        expr = lower_reductions(value, tracer)
        assignee = Access(self.name, indices)
        kind = "accumulate" if _reads_assignee(expr, assignee) else "assign"
        tracer.record(assignee, expr, kind, where)

    def __len__(self) -> int:
        """Refuse: the size is symbolic; iterate ``.dom``."""
        raise TraceError(
            f"len({self.name}) is symbolic while tracing; iterate {self.name}.dom"
        )

    def __bool__(self) -> bool:
        """Refuse: an array is not a truth value."""
        raise TraceError(
            f"the truth value of the symbolic array {self.name} is undefined"
        )

    def __repr__(self) -> str:
        return f"SymArr({self.name}: {self.type})"


def _reads_assignee(expr: Any, assignee: Access) -> bool:
    """Does the right-hand side read the very cell it writes?

    ``y[r] = y[r] + t`` and ``y[r] += t`` (which Python compiles to the same
    subscript assignment) are read-modify-writes, and the dependence analysis
    wants them recorded as accumulations rather than as an unrelated read and
    write of the same cell.
    """
    for access in accesses_in(expr):
        if access.array == assignee.array and structurally_equal(
            access.indices, assignee.indices
        ):
            return True
    return False


# }}}


# {{{ reductions


def _binder_bound(domain: Any) -> Any:
    """The exclusive bound of one reduction binder's domain."""
    if isinstance(domain, SymDom):
        return domain.bound
    if isinstance(domain, FinType):
        return domain.bound
    raise TraceError(
        f"cannot reduce over {domain!r}: iterate an array's .dom, or Fin[n]"
    )


#: The exactness classes, weakest last. Joining two classes keeps the weaker
#: one, which is what an accumulation of mixed operands is entitled to.
EXACTNESS_ORDER = ("exact", "reassoc", "approx")


def join_exactness(*classes: str) -> str:
    """The weakest of the exactness classes given; ``exact`` if there are none."""
    return max(
        (c for c in classes if c in EXACTNESS_ORDER),
        key=EXACTNESS_ORDER.index,
        default="exact",
    )


def reduction_exactness(body: Any, tracer: Tracer) -> str:
    """The exactness class of a reduction, read off what it sums.

    The class comes from the element sorts of the arrays the body reads, joined
    so that the weakest wins: a sum of ``Nat`` is ``exact``, because adding
    integers in any order gives the same integer and a caller asking for
    ``exact`` is asking for exactly those bits; a sum of ``Real`` is ``approx``,
    because floating-point addition is not associative and the result is only
    ever right to a tolerance. ``reassoc`` sits between them and is what a
    schedule *lowers* an accumulation to when it reorders one (see
    :meth:`loopty.schedule.Schedule.realize`), which is why it is not a class a
    trace can invent: the source did not ask for it.

    A body that reads no array at all (a sum of literals, or of an index
    expression) is ``exact``.
    """
    classes: list[str] = []
    for access in accesses_in(body):
        arrtype = tracer.array_types.get(access.array)
        if arrtype is None:
            continue
        classes.append(_sort_exactness(arrtype.dtype))
    return join_exactness(*classes)


def _sort_exactness(dtype: Any) -> str:
    """The exactness class of one element sort, defaulting to ``approx``."""
    from lanky.prelude import exactness_of

    if isinstance(dtype, np.dtype):
        return "exact" if dtype.kind in "biu" else "approx"
    try:
        return exactness_of(dtype)
    except Exception:  # pragma: no cover - an exotic element type
        return "approx"


def lower_reductions(expr: Any, tracer: Tracer) -> Any:
    """Replace every lanky ``Sum`` in ``expr`` by a :class:`~loopty.term.Reduction`.

    Lanky builds the temporary binder node used by ``reduce_sum``; Loopty gives
    it a domain. The domain's dimensions are the enclosing inames followed by the
    reduction's own, so a ragged reduction bound may mention the row it belongs
    to, and the set is directly comparable with the statement's domain.

    The accumulation's exactness class is derived from what it sums rather than
    fixed; see :func:`reduction_exactness`. It is the tolerance a later
    differential test judges the compiled run by, and the permission a schedule
    needs before it may build a reduction tree.
    """
    if isinstance(expr, Sum):
        inames: list[str] = []
        bounds: list[Any] = []
        for var, domain in expr.binders:
            if var.name in tracer.inames:
                raise TraceError(
                    f"the reduction binder {var.name!r} shadows the enclosing "
                    f"loop variable {var.name!r}; rename one of them"
                )
            inames.append(var.name)
            bounds.append(_binder_bound(domain))
        body = lower_reductions(expr.body, tracer)
        domain = domain_set(
            (*tracer.inames, *inames),
            (*tracer.bounds, *bounds),
            constraints=(
                *constraints_of(tracer.guard()),
                *constraints_of(expr.guard),
            ),
            reflections=tracer.reflections,
        )
        return Reduction(
            "sum", tuple(inames), domain, body, reduction_exactness(body, tracer)
        )
    if isinstance(expr, prim.ExpressionNode):
        return type(expr)(*(lower_reductions(arg, tracer) for arg in init_args(expr)))
    if isinstance(expr, tuple):
        return tuple(lower_reductions(item, tracer) for item in expr)
    return expr


def accesses_in(expr: Any, into_reductions: bool = True) -> tuple[Access, ...]:
    """Every array reference in an expression, as :class:`~loopty.term.Access`.

    Reads are ordinary pymbolic subscripts while the body runs, so that the
    arithmetic around them is pymbolic's and needs no translation; this is the
    reading of them as footprints. An indirection contributes twice, once for
    the outer array and once for the index array, because both cells are read.

    ``into_reductions`` says whether to descend into a reduction's body. A
    footprint or an in-bounds obligation about an access inside a reduction is
    stated over the *reduction's* domain, so the caller that needs that domain
    walks the reduction itself; see :func:`loopty.flow.statement_accesses`.
    """
    out: list[Access] = []

    def walk(node: Any) -> None:
        if isinstance(node, prim.Subscript):
            if isinstance(node.aggregate, prim.Variable):
                out.append(Access(node.aggregate.name, _index_tuple(node.index)))
            walk(node.index)
            return
        if isinstance(node, Reduction):
            if into_reductions:
                walk(node.body)
            return
        if isinstance(node, prim.ExpressionNode):
            for arg in init_args(node):
                walk(arg)
            return
        if isinstance(node, tuple | list):
            for item in node:
                walk(item)

    walk(expr)
    return tuple(out)


def reductions_in(expr: Any) -> tuple[Reduction, ...]:
    """Every reduction in an expression, outermost first.

    A reduction carries its own domain, so the rules that need one (in-bounds
    inside the reduction, the exactness class of the accumulation) find it here
    rather than re-deriving the reduced iteration space.
    """
    out: list[Reduction] = []

    def walk(node: Any) -> None:
        if isinstance(node, Reduction):
            out.append(node)
            walk(node.body)
            return
        if isinstance(node, prim.ExpressionNode):
            for arg in init_args(node):
                walk(arg)
            return
        if isinstance(node, tuple | list):
            for item in node:
                walk(item)

    walk(expr)
    return tuple(out)


# }}}


# {{{ when


_MASKS: list[bool] = []


def _writes_are_masked() -> bool:
    """Whether an open ``when`` block is false, so writes must be dropped."""
    return not all(_MASKS)


class _MaskedArr(Arr):
    """An :class:`~loopty.arr.Arr` sharing its buffers, whose writes obey ``when``."""

    def __setitem__(self, key: Any, value: Any) -> None:
        """Write, unless an open ``when`` block is false."""
        if _writes_are_masked():
            return
        super().__setitem__(key, value)

    def __getitem__(self, key: Any) -> Any:
        """Read, answering zero for an out-of-range read under a false guard."""
        try:
            return super().__getitem__(key)
        except IndexError:
            if _writes_are_masked():
                return 0
            raise


class _MaskedArray(np.ndarray):
    """A numpy view whose writes obey ``when``."""

    def __setitem__(self, key: Any, value: Any) -> None:
        """Write, unless an open ``when`` block is false."""
        if _writes_are_masked():
            return
        super().__setitem__(key, value)

    def __getitem__(self, key: Any) -> Any:
        """Read, answering zero for an out-of-range read under a false guard."""
        try:
            return super().__getitem__(key)
        except IndexError:
            if _writes_are_masked():
                return 0
            raise


def mask_writes(value: Any) -> Any:
    """Wrap an argument so that writes inside a false ``when`` are dropped.

    The wrapper shares the buffers of what it wraps, so a masked run still
    writes through to the caller's array everywhere the guard holds. Only
    arrays are wrapped, and only when the body opens a guard at all:
    :func:`loopty.kernel.opens_a_guard` decides that by looking for this very
    object in the code, whatever name it was imported under, so an unguarded
    kernel is called with exactly the objects it was given.

    Masking covers reads too, in one direction only: inside a false guard, a
    read that would go out of range answers zero rather than raising, because
    the block's value cannot be used. That is what lets the boundary guard of a
    stencil be written once, as the condition it is, instead of as an ``if``
    around the loop bounds.
    """
    if isinstance(value, Arr):
        offsets = value.offsets if value.is_ragged else None
        return _MaskedArr(value.numpy(), offsets)
    if isinstance(value, np.ndarray):
        return value.view(_MaskedArray)
    return value


class when:  # noqa: N801 - a context manager written like a statement
    """Guard the writes of a block by ``condition``.

    Under tracing the condition is pushed onto the guard stack: the statements
    recorded inside carry it, and it narrows their domain when it is affine, so
    a guarded access is proved in bounds exactly where it runs. Under plain
    ``python`` the block still executes and the writes are masked, which is why
    the guard has to be a condition on data and not a Python ``if``: masking
    keeps the traced term and the native run agreeing statement for statement.
    """

    def __init__(self, condition: Any) -> None:
        self.condition = condition
        self.tracer = current_tracer()

    def __enter__(self) -> when:
        """Open the guard."""
        if self.tracer is not None:
            self.tracer.push_guard(self.condition)
        else:
            _MASKS.append(bool(self.condition))
        return self

    def __exit__(self, *exc_info: Any) -> None:
        """Close the guard."""
        if self.tracer is not None:
            self.tracer.pop_guard()
        else:
            _MASKS.pop()


# }}}


# {{{ building the term


def _axis_size(axis: Any) -> Any:
    """The size term of one written axis (``Fin[n]``, an int, or a term)."""
    if isinstance(axis, FinType):
        return axis.bound
    if isinstance(axis, int | prim.ExpressionNode):
        return axis
    bound = getattr(axis, "bound", None)
    if bound is not None:
        return bound
    raise TraceError(f"not an index type: {axis!r}")


def array_type(
    spec: ArrSpec, parameters: Mapping[str, Any], name: str = ""
) -> ArrType:
    """Read an ``Arr[...]`` annotation as an :class:`~loopty.term.ArrType`.

    An axis after the first whose size is a bare name that is *also* an array
    parameter of this kernel is ragged: writing ``val: Arr[Fin[n], Fin[cnt],
    Real]`` next to ``cnt: Arr[Fin[n], Nat]`` says that row ``r`` of ``val`` has
    ``cnt[r]`` entries, which is the dependent sum spelled point-free. Any other
    symbolic size is an ordinary size parameter, uniform across rows.
    """
    axes = tuple(_axis_size(axis) for axis in spec.axes)
    ragged = []
    for position, size in enumerate(axes):
        named = isinstance(size, prim.Variable) and size.name in parameters
        counts = parameters.get(size.name) if named else None
        ragged.append(
            bool(position >= 1 and named and isinstance(counts, ArrSpec))
        )
    return ArrType(axes=axes, dtype=spec.dtype, ragged=tuple(ragged))


def _free_size_names(axes: Sequence[Any]) -> set[str]:
    """Names occurring in axis sizes."""
    out: set[str] = set()
    for axis in axes:
        if isinstance(axis, prim.Variable):
            out.add(axis.name)
        elif isinstance(axis, prim.ExpressionNode):
            for arg in init_args(axis):
                out |= _free_size_names([arg])
        elif isinstance(axis, tuple | list):
            out |= _free_size_names(axis)
    return out


def _proposition_of(annotation: Any) -> Any:
    """The postcondition carried by a return annotation, or ``None``.

    Outputs are parameters, so the return annotation of a kernel is not a result
    type but a claim about the parameters. A bare proposition is that claim; a
    refined type ``T & p & q`` contributes its propositions.
    """
    if annotation is None:
        return None
    if isinstance(annotation, Refined):
        props = tuple(annotation.props)
        if not props:
            return None
        return props[0] if len(props) == 1 else prim.LogicalAnd(props)
    if isinstance(annotation, prim.ExpressionNode):
        return annotation
    return None


def trace(kernel: Any, arg_types: Any) -> Term:
    """Run ``kernel``'s body against symbolic arguments and return its term.

    ``arg_types`` gives one type per parameter, in signature order: an
    ``Arr[...]`` specification for an array, a lanky sort for a scalar, and
    optionally ``"return"`` for the postcondition. It is what the proxies are
    built from, and it is the only thing tracing needs to know that the body
    does not already say, because sizes come from ``.dom``.
    """
    function = getattr(kernel, "fn", kernel)
    name = getattr(kernel, "__name__", getattr(function, "__name__", "kernel"))
    types = dict(arg_types)
    post_annotation = types.pop("return", None)

    tracer = Tracer(name, types)
    arguments: list[Any] = []
    params: list[tuple[str, Any]] = []
    for parameter, annotation in types.items():
        if isinstance(annotation, ArrSpec):
            arrtype = array_type(annotation, types, parameter)
            params.append((parameter, arrtype))
            tracer.array_types[parameter] = arrtype
            # A size a shape mentions is a name the isl spaces already use, so
            # no reflected parameter may be called that; see Reflections.
            tracer.reflections.reserve(_free_size_names(arrtype.axes))
            arguments.append(SymArr(parameter, arrtype, tracer))
        else:
            params.append((parameter, annotation))
            arguments.append(Var(parameter))

    _TRACERS.append(tracer)
    try:
        function(*arguments)
    except SymbolicBoolError as exc:
        raise TraceError(
            f"{exc}\nA kernel body may not branch on a value it computes: "
            "write 'with when(condition):' instead of 'if condition:', which "
            "masks the writes of the block rather than choosing a branch."
        ) from exc
    finally:
        _TRACERS.pop()

    if tracer.loops:
        # Every loop level is closed by its own iterator raising StopIteration.
        # A level still open once the body has returned means that iterator was
        # abandoned, which only a 'break' or a 'return' inside the loop does.
        raise TraceError(_abandoned_message([loop.iname for loop in tracer.loops]))

    sizes: set[str] = set()
    for _, arrtype in params:
        if isinstance(arrtype, ArrType):
            sizes |= _free_size_names(arrtype.axes)
    return Term(
        name=name,
        params=tuple(params),
        sizes=tuple(sorted(sizes - set(types))),
        stmts=tuple(tracer.stmts),
        post=_proposition_of(post_annotation),
        reflected=tracer.reflections.items(),
    )


# }}}
