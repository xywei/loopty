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

One generic point per loop has a blind spot: state a Python name carries from
one iteration to the next. ``s = s + x[i]`` in a loop runs once, so the trace
sees ``s = 0.0 + x[i]`` and never the sum, and the polyhedral model has no cell
for such a name anyway. Two checks refuse the idiom, and the message names the
fix, which is to give the state an index. :meth:`Tracer.record` refuses a
statement that mentions the variable of a loop it is not inside, which is how
a value carried *out* of a loop shows up. :meth:`Tracer.leave_loop` compares
what the frame running the ``for`` holds with what it held when the loop
opened, which catches a carried value that mentions no loop variable at all,
such as ``s = s + 1.0``: its locals, the globals its code rebinds, and the
contents of the lists, dicts and sets it can reach, since ``acc[0] += 1.0``
leaves ``acc`` bound to the same list.
"""

from __future__ import annotations

import builtins
import dis
import itertools
import sys
from collections.abc import Collection, Iterator, Mapping, Sequence
from dataclasses import dataclass
from types import CodeType
from typing import Any

import islpy as isl
import numpy as np
import pymbolic.primitives as prim
from lanky.prelude import FinType, Refined
from lanky.terms import (
    Exists,
    Forall,
    Subscript,
    Sum,
    SymbolicBoolError,
    Var,
    init_args,
    render,
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


#: The two ways to give loop-carried state an index, which is what both
#: loop-carried refusals end with.
_INDEX_THE_STATE = (
    "Give the state an index: loopty.reduce_sum({reduction}) when it is an "
    "accumulation, or an indexed cell that each iteration writes ({cell}, the "
    "way a prefix scan is written) when it is not."
)


#: How many items of a container a message prints before it writes ``...``.
_SHOWN_ITEMS = 6


def _shown(value: Any) -> str:
    """A value the way a message prints it: as the body spells it, where it can.

    A term is rendered, a symbolic array is its name and a symbolic domain is
    ``x.dom`` (a ping-pong ``a, b = b, a`` swaps two arrays). A list, tuple,
    dict or set is written the way Python writes it, with its items shown the
    same way and cut off after the first few; anything else is its ``repr``.
    """
    if isinstance(value, SymArr):
        return value.name
    if isinstance(value, SymDom):
        return _domain_text(value)
    if isinstance(value, prim.ExpressionNode):
        try:
            return render(value)
        except Exception:  # pragma: no cover - a node lanky cannot print
            return str(value)
    if isinstance(value, dict):
        items = itertools.islice(value.items(), _SHOWN_ITEMS)
        shown = [f"{_shown(key)}: {_shown(item)}" for key, item in items]
        return "{" + _items_text(shown, len(value)) + "}"
    if isinstance(value, list | tuple | set):
        shown = [_shown(item) for item in itertools.islice(value, _SHOWN_ITEMS)]
        text = _items_text(shown, len(value))
        if isinstance(value, list):
            return f"[{text}]"
        if isinstance(value, tuple):
            return f"({text},)" if len(value) == 1 else f"({text})"
        return f"{{{text}}}" if value else "set()"
    return repr(value)


def _items_text(shown: Sequence[str], total: int) -> str:
    """The items a message shows of a container of ``total``, comma separated."""
    return ", ".join(shown) + (", ..." if total > len(shown) else "")


def _loop_variable(loop: _Loop) -> str:
    """The name the body gives a loop's variable: its ``for`` target, if read."""
    return loop.target or loop.iname


def _loop_domain(loop: _Loop) -> str:
    """The domain a loop runs over, the way the body spells it."""
    dom = getattr(loop.owner, "dom", None)
    return _domain_text(dom) if isinstance(dom, SymDom) else "its domain"


def _escaped_message(loops: Sequence[_Loop], cell: str, where: str) -> str:
    """What to say about a statement that mentions loops it is not inside.

    Each loop is named by its ``for`` target and line rather than by its iname
    alone, which differs from the target when the target is reused: after
    ``for i in v.dom: for i in u.dom[i]: ...`` the ``i`` the outer body reads
    is the inner loop's, which the trace calls ``i_0``.
    """
    many = len(loops) > 1
    listed = " and ".join(
        f"{_loop_variable(loop)!r} of the loop at {loop.where}"
        + (
            ""
            if _loop_variable(loop) == loop.iname
            else f" ({loop.iname} in the trace)"
        )
        for loop in loops
    )
    variable = _loop_variable(loops[0])
    return (
        f"the write to {cell} at {where} mentions the loop "
        f"variable{'s' if many else ''} {listed}, outside "
        f"{'those loops' if many else 'that loop'}. Tracing runs the body once "
        "with every loop taking one generic point, so a value that a Python "
        "name carries out of a loop, or from one iteration into the next, is "
        "that point and not what the loop computes. "
        + _INDEX_THE_STATE.format(
            reduction=f"... for {variable} in {_loop_domain(loops[0])}",
            cell=f"s[{variable} + 1] = s[{variable}] + ...",
        )
    )


class _Unbound:
    """What a name deleted inside a loop holds after one iteration: nothing."""

    def __repr__(self) -> str:
        return "unbound"


#: The value :meth:`Tracer._carried` compares a name the iteration deleted by.
_UNBOUND = _Unbound()


@dataclass(frozen=True)
class _Carried:
    """One piece of state a loop carries from one iteration into the next.

    ``holder`` is how a message names it (``'s'``, ``the list 'acc'``, ``the
    global 'G'``) and ``name`` the Python name the body reaches it by, which
    the fixes are spelled with. ``cell`` is the name or the element whose value
    changed (``acc[0]``, or ``acc`` itself when its length did), and ``before``
    and ``after`` are that value when the loop opened and after one iteration.
    ``kind`` is ``"name"``, ``"global"`` or ``"container"``, which decides
    what the message suggests for a temporary.
    """

    holder: str
    name: str
    cell: str
    before: Any
    after: Any
    kind: str


#: What each kind of carried state should be when it is only per-iteration
#: scratch, which is how a false positive is answered.
_SCRATCH = {
    "name": (
        "If {name!r} is only a temporary that each iteration assigns before it "
        "reads it, give it a name that is not bound before the loop."
    ),
    "global": (
        "If {name!r} is only a temporary that each iteration assigns before it "
        "reads it, make it a local name that is first bound inside the loop."
    ),
    "container": (
        "If {name!r} is only scratch that each iteration fills before it reads "
        "it, create it inside the loop instead."
    ),
}


def _probe_message(loop: _Loop, probes: Sequence[str]) -> str:
    """What to say about a loop whose code can ask whether a name is bound."""
    variable = _loop_variable(loop)
    listed = " and ".join(f"{name}()" if name.islower() else name for name in probes)
    return (
        f"the code running the loop over {variable!r} at {loop.where} uses "
        f"{listed}, and a kernel body may not inspect which names are bound. "
        "Tracing runs the body once with the loop taking one generic point, so "
        "a name that one iteration binds or deletes for the next is seen only "
        "as the first iteration finds it, and a test on it takes one branch in "
        "the trace and another natively. "
        + _INDEX_THE_STATE.format(
            reduction=f"... for {variable} in {_loop_domain(loop)}",
            cell=f"s[{variable} + 1] = s[{variable}] + ...",
        )
        + " If the name is only a temporary, bind it in every iteration instead "
        "of testing for it."
    )


def _carried_message(loop: _Loop, carried: Sequence[_Carried]) -> str:
    """What to say about the state a loop carries from one iteration to the next."""
    variable = _loop_variable(loop)
    holders = ", ".join(entry.holder for entry in carried)
    values = "; ".join(
        f"{entry.cell!r} is {_shown(entry.before)} before the loop and "
        f"{_shown(entry.after)} after one iteration"
        for entry in carried
    )
    first = carried[0]
    return (
        f"the loop over {variable!r} at {loop.where} carries {holders} from one "
        f"iteration to the next ({values}). Tracing runs the body once with the "
        "loop taking one generic point, so it sees one iteration and not what "
        "the loop computes, and the polyhedral model has no cell for state that "
        "Python keeps outside the arrays. "
        + _INDEX_THE_STATE.format(
            reduction=f"... for {variable} in {_loop_domain(loop)}",
            cell=f"{first.name}[{variable} + 1] = {first.name}[{variable}] + ...",
        )
        + " "
        + _SCRATCH[first.kind].format(name=first.name)
    )


class TraceError(RuntimeError):
    """A body did something tracing cannot follow, with the fix in the message.

    The archetype is a Python ``if`` on a symbolic value: the message names the
    ``with when(...)`` replacement. Data-dependent ``while``, ``break``,
    ``return`` out of a loop, Python's builtin ``sum`` over a symbolic domain,
    state that a Python name, a global, or a list, dict or set carries from one
    loop iteration to the next, and a loop whose code can ask which names are
    bound are the other cases.
    """


# {{{ the tracer


@dataclass
class _Loop:
    """One open loop level: its iname, the variable, and its exclusive bound.

    ``owner`` is the iterator that opened the level. A level is closed when
    *that* iterator is asked for its second point, so an iterator closing a
    level it did not open means a loop was abandoned; see
    :meth:`Tracer.leave_loop`.

    ``target`` is the name the ``for`` statement binds, when it could be read,
    and ``where`` the line of the ``for``. ``state`` is what the frame running
    the ``for`` held when the level opened, and so before the ``for`` bound its
    target: what the loop's first iteration started from, which the frame at
    the close is compared with.
    """

    iname: str
    var: Var
    bound: Any
    owner: Any = None
    target: str | None = None
    where: str = ""
    state: _Snapshot | None = None


@dataclass
class _Held:
    """A list, dict or set a loop could carry state in, copied at the loop's open.

    ``label`` is how the body reaches it (``acc``, or ``pair[0]`` for a list
    kept in a tuple), ``root`` the name the label starts with, and ``scope``
    ``"local"`` or ``"global"``. ``copy`` is shallow: its elements are the very
    objects the container held, so a change is seen one level deep.
    """

    label: str
    root: str
    scope: str
    container: Any
    copy: Any


@dataclass
class _Snapshot:
    """What the frame running a ``for`` held when the loop level opened.

    ``names`` is a copy of its locals, and ``local`` every name its code keeps
    as a local, cell or free variable rather than as a global. ``namespace``
    is its module's globals, the live dictionary, and ``globals`` the values
    then of the names its code rebinds there with ``global G``. ``held`` is
    every list, dict and set
    reachable from a local or from a global the code names: a container
    mutated in place is the same object after the iteration, so comparing the
    names cannot see what changed in it. See :func:`_snapshot`.
    """

    names: dict[str, Any]
    local: frozenset[str]
    namespace: Mapping[str, Any]
    globals: dict[str, Any]
    held: list[_Held]


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
        #: The loop levels that have closed, by iname, so that a statement
        #: mentioning one of their variables can be told which loop it is.
        self._closed: dict[str, _Loop] = {}
        #: Position of each open block in its parent, outermost first.
        self._path: list[int] = []
        #: How many children each open block has produced so far.
        self._counters: list[int] = [0]

    # {{{ loops

    def fresh_iname(self, hint: str | None) -> str:
        """A loop variable name, from the ``for`` target when one can be read.

        The name avoids every name the term already uses: earlier inames,
        reflected parameters, and the sizes and parameters the signature
        declares. A ``for k in x.dom`` over ``x: Arr[Fin[k], Real]`` would
        otherwise make one isl dimension of the size and the iname, and the
        loop's domain ``0 <= k < k`` would be empty.
        """
        stem = hint or f"i{len(self._inames)}"
        name = stem
        suffix = 0
        while name in self._inames or self.reflections.taken(name):
            name = f"{stem}_{suffix}"
            suffix += 1
        self._inames.add(name)
        # An iname and a reflected parameter share one isl space, so neither may
        # take a name the other has.
        self.reflections.reserve((name,))
        return name

    def enter_loop(
        self,
        bound: Any,
        hint: str | None,
        owner: Any = None,
        frame: Any = None,
    ) -> Var:
        """Open a loop level over ``0 <= i < bound`` and return its variable.

        ``hint`` is the ``for`` target, and ``frame`` the frame running the
        ``for``, as it is now, before it binds that target. What it holds is
        copied (see :func:`_snapshot`), because ``frame.f_locals`` is a live,
        write-through view from Python 3.13 on (PEP 667), and before that a
        dictionary the frame refreshes in place on the next access, so either
        way the level would otherwise see the locals at its close twice.

        A frame whose code can ask whether a name is bound is refused here,
        before anything is copied; see :func:`_binding_probes`.
        """
        name = self.fresh_iname(hint)
        var = Var(name)
        loop = _Loop(name, var, bound, owner, target=hint)
        if frame is not None:
            loop.where = _location(frame)
            probes = _binding_probes(frame)
            if probes:
                raise TraceError(_probe_message(loop, probes))
            loop.state = _snapshot(frame, self._owned())
        self.loops.append(loop)
        self._path.append(self._position())
        self._counters.append(0)
        return var

    def _owned(self) -> set[int]:
        """The containers the tracer keeps for itself, by identity.

        No body can carry state in them, and a helper frame that reaches one
        would otherwise see it change as statements are recorded. The symbolic
        arrays and domains need no entry: they are not containers.
        """
        return {id(value) for value in (*vars(self).values(), _TRACERS, _MASKS)}

    def leave_loop(self, owner: Any = None, frame: Any = None) -> None:
        """Close the innermost loop level, refusing to close somebody else's.

        A level is popped when its iterator is asked for a second point and
        raises ``StopIteration``. A ``break`` or a ``return`` inside the loop
        means that never happens, and the level stays open: the next statement
        would then be recorded under a loop variable the body has left, and the
        enclosing loop's own ``StopIteration`` would pop the abandoned level
        instead of its own. That mismatch is what is caught here, at the point
        where it happens, so the message can name the loop.

        ``frame`` is the frame running the ``for``, now that the body has run
        once. A name bound to a different value than it had when the loop
        opened, or a container whose contents changed, is state the second
        iteration would start from, which the trace never runs; see
        :meth:`_carried`.
        """
        if owner is not None and self.loops and self.loops[-1].owner is not owner:
            raise TraceError(_abandoned_message([self.loops[-1].iname]))
        loop = self.loops.pop()
        self._closed[loop.iname] = loop
        self._path.pop()
        self._counters.pop()
        if loop.state is None or frame is None:
            return
        carried = self._carried(loop, frame.f_locals)
        if carried:
            raise TraceError(_carried_message(loop, carried))

    def _carried(self, loop: _Loop, after: Mapping[str, Any]) -> list[_Carried]:
        """The state ``loop`` carries into its next iteration, with both values.

        A name counts when it was bound before the loop opened and is bound to
        a different value after one iteration, whatever the value is: a term,
        or a plain Python number such as a counter ``k = k + 1`` that ends up
        in an index. A name the iteration deleted counts too, since the next
        iteration starts without it. So does a global the code of the frame
        rebinds with ``global G``. Rebinding to the identical object or to an
        equal value carries nothing (see :func:`_same_value`), and a name
        first bound inside the loop is a per-iteration temporary.

        A list, dict or set counts when it was reachable before the loop opened
        and its contents are different after one iteration (see
        :func:`_changed`), which is how ``acc[0] = acc[0] + x[i]`` and
        ``seen.add(i)`` show up; a container first created inside the loop is
        per-iteration scratch, like a temporary name. A change nested deeper
        than the container's own elements, such as ``acc[0][0] += 1``, is not
        seen, and neither is an attribute an object holds.

        Three things are left alone on purpose:

        * the loop's own target, which the ``for`` rebinds before every
          iteration, so no iteration can read the value the previous one left;
        * a name bound to a :class:`when`, the object ``with when(...) as g``
          binds, which carries no data: its condition is recorded on the
          statements it guards;
        * a name whose value before the loop already mentions the variable of
          a loop that has closed, such as a ``for`` target reused by a later
          loop, and likewise a global, or an element of a container, whose
          value does. Anything that reads such a value is refused as an escaped
          loop variable by :meth:`record`, wherever the read ends up, so there
          is nothing the comparison here could add.
        """
        state = loop.state
        if state is None:
            return []
        closed = self._inames.difference(self.inames)
        out: list[_Carried] = []
        # A global is a name like a local, down to the loop's own target:
        # ``global i`` before ``for i in x.dom`` stores the target there. The
        # ``for`` stores it in one scope only, so a global that a helper
        # defined in the body rebinds under the name of a local target is
        # still state.
        stored = "name" if loop.target in state.local else "global"
        scopes = (
            (state.names, after, "name"),
            (state.globals, state.namespace, "global"),
        )
        for values, now, kind in scopes:
            for name, before in values.items():
                if name == loop.target and kind == stored:
                    continue
                value = now[name] if name in now else _UNBOUND
                if loop.target is None and value is loop.var:
                    # The target could not be read off the bytecode; the name
                    # still holding the loop's own variable is that target.
                    continue
                if _same_value(before, value) or isinstance(value, when):
                    continue
                if _loop_variables(before, closed):
                    continue
                holder = repr(name) if kind == "name" else f"the global {name!r}"
                out.append(_Carried(holder, name, name, before, value, kind))
        # A container whose name was rebound is reported as that rebinding.
        reported = {entry.name for entry in out}
        for held in state.held:
            change = None if held.root in reported else _changed(held, closed)
            if change is None:
                continue
            cell, before, value = change
            scope = "global " if held.scope == "global" else ""
            kind = type(held.container).__name__
            out.append(
                _Carried(
                    f"the {scope}{kind} {held.label!r}",
                    held.root,
                    cell,
                    before,
                    value,
                    "container",
                )
            )
        return out

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

    def loop_domain(self) -> Any:
        """The isl set of the enclosing loop nest with no guard applied.

        A ``when`` masks the write; it does not skip the block, and Python
        evaluates the whole condition at every point of the nest before the
        mask exists (``&`` is eager). So the reads a guard performs happen over
        this set and not over :meth:`domain`, which the guard has narrowed: a
        guard ``(i + 1 < n) & (flag[i + 1] != 0)`` still reads ``flag[n]`` at
        ``i = n - 1``.
        """
        return domain_set(self.inames, self.bounds, reflections=self.reflections)

    def record(
        self,
        assignee: Access,
        expr: Any,
        kind: str,
        where: str,
        source: Any = None,
    ) -> Stmt:
        """Append one statement instance family to the term being built.

        A statement that mentions the variable of a loop it is not inside is
        refused. Every statement is a family over its own inames, so such a
        variable would be free in the term, and the only way one gets there is
        a Python name carrying the loop's generic point past the loop: out of
        it, as in ``for i in x.dom: s = s + x[i]`` followed by ``y[0] = s``, or
        into a later loop through a bound or a guard.

        ``source`` is the right-hand side as the body built it, before
        :func:`lower_reductions` gave its reductions an isl domain. It is
        looked at too, because each reduction binder's bound is still its own
        there: ``reduce_sum(... for r in val.dom[r])`` after a loop over ``r``
        reads that loop's ``r`` in its bound, which the lowered domain can no
        longer tell from the binder.
        """
        escaped = self._escaped(assignee, expr, source)
        if escaped:
            cell = (
                _shown(_subscript(assignee.array, assignee.indices))
                if assignee.indices
                else assignee.array
            )
            loops = [self._closed[name] for name in sorted(escaped)]
            raise TraceError(_escaped_message(loops, cell, where))
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
            loop_domain=self.loop_domain() if self.guards else None,
        )
        self.stmts.append(stmt)
        return stmt

    def _escaped(self, assignee: Access, expr: Any, source: Any = None) -> set[str]:
        """Loop variables a statement mentions that no loop around it binds.

        Everything the statement's instances depend on is looked at: the
        assignee's indices, the right-hand side with its reductions (lowered,
        and as the body built it), the guard, and the bounds of the enclosing
        loops, since ``for j in val.dom[r]`` after the loop over ``r`` has
        closed puts ``r`` in the domain alone.
        """
        return _loop_variables(
            (assignee, expr, source, self.guard(), self.bounds),
            self._inames,
            bound=self.inames,
            reflections=self.reflections,
        )


_TRACERS: list[Tracer] = []


def current_tracer() -> Tracer | None:
    """The tracer of the innermost :func:`trace` call, or ``None``."""
    return _TRACERS[-1] if _TRACERS else None


# }}}


# {{{ loop-carried state


def _loop_variables(
    node: Any,
    names: Collection[str],
    bound: Collection[str] = (),
    reflections: Reflections | None = None,
) -> set[str]:
    """The loop variables among ``names`` that ``node`` mentions free.

    A lanky binder (``Sum``, ``Forall``, ``Exists``) that has not been lowered
    yet, as in a guard or a right-hand side as the body built it, binds each of
    its variables in the binders after it and in its body, but not in its own
    bound: Python evaluates the iterable of a generator's ``for`` before it
    binds that ``for``'s target, so the ``r`` in ``for r in val.dom[r]`` is the
    ``r`` from before, such as a closed loop's variable.

    A lowered reduction binds its own inames in its body, and keeps its bounds
    only in its isl domain, where a loop variable shows up as a parameter: by
    name when the bound is affine, inside the term a reflected parameter stands
    for when it is not, which is what ``reflections`` is asked for. Which
    binder a bound belonged to is gone by then, so a bound that mentions a
    closed loop's variable under the name of its own binder reads as bound
    here; :meth:`Tracer.record` also walks the right-hand side before lowering
    for that reason.
    """
    out: set[str] = set()

    def walk(node: Any, bound: frozenset[str]) -> None:
        if isinstance(node, prim.Variable):
            if node.name in names and node.name not in bound:
                out.add(node.name)
        elif isinstance(node, Reduction):
            inner = bound | frozenset(node.inames)
            walk(node.body, inner)
            for param in node.domain.get_var_names(isl.dim_type.param):
                if param in names and param not in inner:
                    out.add(param)
                if reflections is not None:
                    walk(reflections.get(param), inner)
        elif isinstance(node, Sum | Forall | Exists):
            inner = bound
            for var, domain in node.binders:
                walk(getattr(domain, "bound", None), inner)
                inner = inner | {var.name}
            walk(node.body, inner)
            walk(node.guard, inner)
        elif isinstance(node, Access):
            walk(node.indices, bound)
        elif isinstance(node, prim.ExpressionNode):
            for arg in init_args(node):
                walk(arg, bound)
        elif isinstance(node, tuple | list):
            for item in node:
                walk(item, bound)

    walk(node, frozenset(bound))
    return out


def _same_value(before: Any, after: Any) -> bool:
    """Whether rebinding a name from ``before`` to ``after`` changed nothing.

    Identity first. Terms are compared as syntax trees, because ``==`` on a
    lanky term builds a proposition rather than answering; tuples item by item;
    a symbolic domain by its array and the indices of its fiber; numbers and
    strings of one type by value, so ``f = f * 1.0`` is not state. Anything
    else is the same value only when it is the same object.
    """
    if before is after:
        return True
    if isinstance(before, prim.ExpressionNode) or isinstance(
        after, prim.ExpressionNode
    ):
        return structurally_equal(before, after)
    if isinstance(before, tuple) and isinstance(after, tuple):
        return len(before) == len(after) and all(map(_same_value, before, after))
    if isinstance(before, SymDom) and isinstance(after, SymDom):
        return before.array is after.array and _same_value(
            before.prefix, after.prefix
        )
    if type(before) is type(after) and isinstance(
        before, int | float | complex | str | np.generic
    ):
        return bool(before == after)
    return False


def _snapshot(frame: Any, owned: Collection[int]) -> _Snapshot:
    """What ``frame``, which is running a ``for``, holds as the loop opens.

    Its locals are copied, and so are the values of the globals its code
    rebinds. Every list, dict and set that a local holds, or that a global its
    code names holds, is copied shallowly, looking through tuples, which cannot
    change themselves but can hold something that does. A container reached
    twice is copied once, under the first name; one the tracer keeps for
    itself (``owned``, by identity) is not copied at all.
    """
    names = dict(frame.f_locals)
    namespace = frame.f_globals
    rebound, read = _global_names(frame.f_code)
    held: list[_Held] = []
    seen = set(owned)

    def visit(label: str, root: str, scope: str, value: Any) -> None:
        if id(value) in seen:
            return
        if isinstance(value, tuple):
            for position, item in enumerate(value):
                visit(f"{label}[{position}]", root, scope, item)
            return
        for kind in (list, dict, set):
            if isinstance(value, kind):
                seen.add(id(value))
                held.append(_Held(label, root, scope, value, kind(value)))
                return

    for name, value in names.items():
        visit(name, name, "local", value)
    for name in sorted(rebound | read):
        if name in namespace:
            visit(name, name, "global", namespace[name])
    code = frame.f_code
    return _Snapshot(
        names=names,
        local=frozenset((*code.co_varnames, *code.co_cellvars, *code.co_freevars)),
        namespace=namespace,
        globals={name: namespace[name] for name in rebound if name in namespace},
        held=held,
    )


def _global_names(code: CodeType) -> tuple[set[str], set[str]]:
    """The globals ``code`` rebinds and the globals it reads, by name.

    Functions defined inside ``code`` count too: they share its module's
    globals, so a helper written in the body with ``global G`` rebinds the
    same ``G`` when the loop calls it. A builtin read by name is in neither
    set's namespace and is skipped by the caller.
    """
    rebound: set[str] = set()
    read: set[str] = set()
    for current in _codes(code):
        for instruction in dis.get_instructions(current):
            if instruction.opname in ("STORE_GLOBAL", "DELETE_GLOBAL"):
                rebound.add(instruction.argval)
            elif instruction.opname == "LOAD_GLOBAL":
                read.add(instruction.argval)
    return rebound, read


def _codes(code: CodeType) -> Iterator[CodeType]:
    """``code`` and every code object defined inside it, at any depth."""
    pending = [code]
    while pending:
        current = pending.pop()
        yield current
        pending.extend(c for c in current.co_consts if isinstance(c, CodeType))


#: The builtins a body can ask whether a name is bound with: the namespaces
#: themselves, and the exceptions that reading an unbound name raises.
_BINDING_PROBES = ("locals", "globals", "vars", "NameError", "UnboundLocalError")


def _binding_probes(frame: Any) -> list[str]:
    """The builtins in ``frame``'s code that can ask whether a name is bound.

    The snapshot compares the names bound before a loop, and a name first bound
    inside it is taken for a per-iteration temporary. That holds only while
    each iteration binds the name before reading it, and ``if "s" not in
    locals(): s = 0`` or ``try: s except NameError: s = 0`` binds it in the
    first iteration only: the one the trace runs. A body that can see which
    names are bound can always choose a branch by what an earlier iteration
    left, so it is refused rather than analysed.

    The code object's names are read first, with those of the functions
    defined inside it: a name the code reads as a global or a builtin is in
    ``co_names`` however a CPython version compiles the call or the ``except``
    clause. A local never is, so a local ``locals`` is left alone, and so is a
    name the frame's locals or its module's globals bind, which is somebody
    else's function by the time the call runs. An attribute is in
    ``co_names`` too, and pytest's assertion rewriting puts one there in every
    body with an ``assert`` (``@py_builtins.locals()``), so a name that every
    instruction mentioning it reads as an attribute is left alone as well.
    Any other instruction, one a later CPython may add included, counts as a
    probe, so an instruction this does not know makes it refuse, not miss.
    """
    uses: dict[str, set[str]] = {}
    for code in _codes(frame.f_code):
        named = [name for name in _BINDING_PROBES if name in code.co_names]
        if not named:
            continue
        for name in named:
            uses.setdefault(name, set())
        for instruction in dis.get_instructions(code):
            if isinstance(instruction.argval, str) and instruction.argval in named:
                uses[instruction.argval].add(instruction.opname)
    shadows = (frame.f_locals, frame.f_globals)
    return [
        name
        for name, opnames in uses.items()
        if not (opnames and all(_reads_an_attribute(op) for op in opnames))
        and not any(name in scope for scope in shadows)
        and frame.f_builtins.get(name) is getattr(builtins, name)
    ]


def _reads_an_attribute(opname: str) -> bool:
    """Whether an instruction named ``opname`` looks its name up on an object."""
    return "ATTR" in opname or "METHOD" in opname


def _changed(held: _Held, closed: Collection[str]) -> tuple[str, Any, Any] | None:
    """The first change to a container since it was copied, or ``None``.

    The change is named by the cell the body would read it back from, with that
    cell's two values: ``acc[0]`` for a list of one length, ``d['k']`` for a
    dict of the same keys in the same order. A container whose length, keys or
    elements otherwise changed is named whole, with both of its contents. An
    element is compared the way a name's value is (see :func:`_same_value`),
    and one whose old value mentions the variable of a closed loop is left
    alone, as a name holding such a value is; see :meth:`Tracer._carried`.
    """
    old, new, label = held.copy, held.container, held.label
    if isinstance(old, list):
        if len(old) != len(new):
            return label, old, list(new)
        cells = zip(range(len(old)), old, new)
    elif isinstance(old, dict):
        if len(old) != len(new) or not all(map(_same_value, old, new)):
            return label, old, dict(new)
        cells = zip(old, old.values(), new.values())
    else:
        # A set has no cells to name, and no order to compare in.
        ids = {id(item) for item in new}
        if len(old) == len(new) and all(
            id(item) in ids or any(_same_value(item, other) for other in new)
            for item in old
        ):
            return None
        return label, old, set(new)
    for key, before, after in cells:
        if not _same_value(before, after) and not _loop_variables(before, closed):
            return f"{label}[{_shown(key)}]", before, after
    return None


def _location(frame: Any) -> str:
    """``file:line`` of what ``frame`` is executing, the tracer's source map."""
    return f"{frame.f_code.co_filename.rsplit('/', 1)[-1]}:{frame.f_lineno}"


def _domain_text(dom: SymDom) -> str:
    """A symbolic domain the way the body spells it: ``x.dom`` or ``val.dom[r]``."""
    return f"{dom.array.name}.dom" + "".join(
        f"[{_shown(index)}]" for index in dom.prefix
    )


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
        # isl spells equality with one '='; its parser refuses Python's '=='.
        if operator == "==":
            operator = "="
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

    From Python 3.13 on the compiler fuses a store with the instruction after
    it when both are on one line, so ``for i in x.dom: s = s + x[i]`` stores
    its target with ``STORE_FAST_LOAD_FAST ('i', 's')``. The target is what is
    stored first, which is the first name of the pair.
    """
    frame = sys._getframe(2)
    try:
        for instruction in dis.get_instructions(frame.f_code):
            if instruction.offset > frame.f_lasti and instruction.opname.startswith(
                "STORE_"
            ):
                name = instruction.argval
                if isinstance(name, tuple) and name:
                    name = name[0]
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

    The frame that calls ``__next__`` is the one running the ``for``, whether
    that is the kernel body or a helper it calls, and both calls come from it:
    the first before the target is bound, the second once the body has run.
    What it holds at those two moments is what the tracer compares to find
    state carried across iterations; see :meth:`Tracer.leave_loop`.
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
                    tracer.leave_loop(self, sys._getframe(1))
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
        frame = sys._getframe(1)
        caller = frame.f_code.co_name
        if caller in ("<genexpr>", "<listcomp>", "<setcomp>", "<dictcomp>"):
            raise TraceError(
                f"a comprehension over {self.dom!r} is being driven by Python "
                "itself, which cannot see the reduced domain; write "
                "loopty.reduce_sum(... for j in arr.dom[r]) so that the "
                "reduction and its domain are recorded"
            )
        self.loop = True
        return tracer.enter_loop(self.dom.bound, _loop_target_name(), self, frame)


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
        where = _location(sys._getframe(1))
        expr = lower_reductions(value, tracer, where=where)
        assignee = Access(self.name, indices)
        kind = "accumulate" if _reads_assignee(expr, assignee) else "assign"
        tracer.record(assignee, expr, kind, where, source=value)

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

    The class is that of the *value* the body computes, joined so that the
    weakest wins: a sum of ``Nat`` is ``exact``, because adding integers in any
    order gives the same integer and a caller asking for ``exact`` is asking
    for exactly those bits; a sum of ``Real`` is ``approx``, because
    floating-point addition is not associative and the result is only ever
    right to a tolerance. ``reassoc`` sits between them and is what a schedule
    *lowers* an accumulation to when it reorders one (see
    :meth:`loopty.schedule.Schedule.realize`), which is why it is not a class a
    trace can invent: the source did not ask for it.

    Arrays are not the only thing a body reads. ``a * x[j]`` with ``a: Real``
    sums floats over an integer ``x``, and so does ``0.1 * j``, which reads no
    array at all; both used to be called ``exact`` because only the element
    sorts of the arrays were joined, and a schedule then refused to reassociate
    a sum that was never exact. :func:`_value_exactness` walks the whole
    expression instead. A body built from integer literals, loop variables and
    integral arrays with ``+``, ``*``, ``//`` and ``%`` is ``exact``.
    """
    return _value_exactness(body, tracer)


def _value_exactness(node: Any, tracer: Tracer) -> str:
    """The exactness class of the value of one expression.

    Integers stay integers under ``+``, ``*``, ``//``, ``%`` and a power with a
    non-negative literal exponent; anything else that can produce a float does
    produce one. So a float or complex literal is ``approx``, and so is true
    division, a power whose exponent could be negative, and a call, whose
    result nothing here knows the sort of. A comparison or a logical connective
    is a boolean and ``exact``. A name is a scalar parameter, which has its
    sort, or a loop variable or a size, which is an integer. An array read is
    its element sort; its index is an integer by construction and is not a
    contribution. An expression this walk does not know is ``approx``, which
    is the class that claims nothing.
    """
    if isinstance(node, bool | np.bool_ | int | np.integer):
        return "exact"
    if isinstance(node, float | complex | np.floating | np.complexfloating):
        return "approx"
    if isinstance(node, Reduction):
        return node.exactness
    if isinstance(node, Access):
        arrtype = tracer.array_types.get(node.array)
        return "approx" if arrtype is None else _sort_exactness(arrtype.dtype)
    if isinstance(node, prim.Subscript):
        name = getattr(node.aggregate, "name", None)
        arrtype = tracer.array_types.get(name) if name is not None else None
        return "approx" if arrtype is None else _sort_exactness(arrtype.dtype)
    if isinstance(node, prim.Variable):
        if node.name in tracer.array_types:
            return _sort_exactness(tracer.array_types[node.name].dtype)
        if node.name in tracer.params:
            return _sort_exactness(tracer.params[node.name])
        return "exact"
    if isinstance(node, prim.Quotient | prim.Call):
        return "approx"
    if isinstance(node, prim.Power):
        exponent = node.exponent
        if not (isinstance(exponent, int | np.integer) and exponent >= 0):
            return "approx"
        return _value_exactness(node.base, tracer)
    if isinstance(
        node, prim.Comparison | prim.LogicalAnd | prim.LogicalOr | prim.LogicalNot
    ):
        return "exact"
    if isinstance(node, prim.If):
        return join_exactness(
            _value_exactness(node.then, tracer), _value_exactness(node.else_, tracer)
        )
    if isinstance(node, prim.Sum | prim.Product | prim.Min | prim.Max):
        return join_exactness(
            *(_value_exactness(child, tracer) for child in node.children)
        )
    if isinstance(node, prim.FloorDiv | prim.Remainder):
        return join_exactness(
            _value_exactness(node.numerator, tracer),
            _value_exactness(node.denominator, tracer),
        )
    return "approx"


def _sort_exactness(dtype: Any) -> str:
    """The exactness class of one element sort, defaulting to ``approx``.

    A Python or numpy scalar type is read by its kind, before lanky is asked:
    lanky calls anything that is not one of its sorts an index type and so
    ``exact``, which is the wrong answer for ``float``. A builtin type arrives
    here from a hand-built term; a kernel annotated ``a: float`` does not pass
    one, because lanky evaluates a postponed annotation's builtin names as free
    names, and such a kernel is refused before it is traced
    (:func:`loopty.term.free_name_sorts`).
    """
    from lanky.prelude import exactness_of

    if isinstance(dtype, np.dtype):
        return "exact" if dtype.kind in "biu" else "approx"
    # By identity: a lanky sort may answer ``==`` with a proposition.
    if dtype is bool or dtype is int:
        return "exact"
    if dtype is float or dtype is complex:
        return "approx"
    if isinstance(dtype, type) and issubclass(dtype, np.generic):
        return "exact" if np.dtype(dtype).kind in "biu" else "approx"
    try:
        return exactness_of(dtype)
    except Exception:  # pragma: no cover - an exotic element type
        return "approx"


def lower_reductions(
    expr: Any,
    tracer: Tracer,
    enclosing: Sequence[tuple[str, Any]] = (),
    outer_guards: Sequence[Any] = (),
    where: str = "",
) -> Any:
    """Replace every lanky ``Sum`` in ``expr`` by a :class:`~loopty.term.Reduction`.

    Lanky builds the temporary binder node used by ``reduce_sum``; Loopty gives
    it a domain. The domain's dimensions are the enclosing inames followed by the
    reduction's own, so a ragged reduction bound may mention the row it belongs
    to, and the set is directly comparable with the statement's domain.

    A reduction nested inside another one also runs inside the outer one's
    binders, and the tracer's loop stack does not hold those: they are not
    loops of the statement. ``enclosing`` carries each outer binder with its
    bound, outermost first, and ``outer_guards`` the outer generators' ``if``
    clauses, and both are stated as constraints of the inner domain. The outer
    binders are parameters of that domain rather than dimensions, which keeps
    its dimensions the statement's inames followed by the reduction's own, the
    shape every collector expects. Without them ``i`` in
    ``reduce_sum(reduce_sum(a[i, j] for j in a.dom[i]) for i in a.dom)`` was an
    unconstrained parameter, and ``a[i, j]`` was refuted at ``i = -1``.

    The accumulation's exactness class is derived from what it sums rather than
    fixed; see :func:`reduction_exactness`. It is the tolerance a later
    differential test judges the compiled run by, and the permission a schedule
    needs before it may build a reduction tree.

    A generator's ``if`` clause is a constraint of the domain, and a reduction
    has nowhere else to keep it. So a condition isl cannot state (one that
    reads an array, or compares with ``!=``) is refused here, with ``where``
    the statement's location, rather than dropped: the term would sum over
    every point while the body skips the ones the condition excludes.
    """
    if isinstance(expr, Sum):
        unstated = _unstated(expr.guard)
        if unstated:
            names = [var.name for var, _domain in expr.binders]
            raise TraceError(_reduction_condition_message(unstated, names, where))
        inames: list[str] = []
        bounds: list[Any] = []
        for var, domain in expr.binders:
            if var.name in tracer.inames:
                raise TraceError(
                    f"the reduction binder {var.name!r} shadows the enclosing "
                    f"loop variable {var.name!r}; rename one of them"
                )
            if any(var.name == name for name, _bound in enclosing):
                raise TraceError(
                    f"the reduction binder {var.name!r} shadows the binder of "
                    "the reduction it is nested in; rename one of them"
                )
            inames.append(var.name)
            bounds.append(_binder_bound(domain))
        inner = (*enclosing, *zip(inames, bounds, strict=True))
        guards = (*outer_guards, expr.guard)
        body = lower_reductions(expr.body, tracer, inner, guards, where)
        domain = domain_set(
            (*tracer.inames, *inames),
            (*tracer.bounds, *bounds),
            constraints=(
                *constraints_of(tracer.guard()),
                *_binder_constraints(enclosing, tracer),
                *(piece for guard in outer_guards for piece in constraints_of(guard)),
                *constraints_of(expr.guard),
            ),
            reflections=tracer.reflections,
        )
        return Reduction(
            "sum", tuple(inames), domain, body, reduction_exactness(body, tracer)
        )
    if isinstance(expr, prim.ExpressionNode):
        return type(expr)(
            *(
                lower_reductions(arg, tracer, enclosing, outer_guards, where)
                for arg in init_args(expr)
            )
        )
    if isinstance(expr, tuple):
        return tuple(
            lower_reductions(item, tracer, enclosing, outer_guards, where)
            for item in expr
        )
    return expr


def _unstated(condition: Any) -> list[Any]:
    """The conjuncts of a condition that :func:`constraints_of` has to drop."""
    if condition is None:
        return []
    if isinstance(condition, prim.LogicalAnd):
        return [part for child in condition.children for part in _unstated(child)]
    if constraints_of(condition):
        return []
    return [condition]


def _reduction_condition_message(
    parts: Sequence[Any], binders: Sequence[str], where: str
) -> str:
    """What to say about a reduction condition its domain cannot state."""
    listed = " and ".join(repr(_shown(part)) for part in parts)
    at = f" at {where}" if where else ""
    unequal = any(
        isinstance(part, prim.Comparison) and part.operator == "!=" for part in parts
    )
    why = (
        "compares with '!=', which is not a convex set of points"
        if unequal
        else "reads an array or is not affine"
    )
    split = (
        " For '!=', split the sum in two, one over '<' and one over '>'."
        if unequal
        else ""
    )
    return (
        f"the condition {listed} of the reduction over {', '.join(binders)}{at} "
        "cannot be a constraint of the reduction's domain, which is the only "
        "place a reduction keeps its condition: the term would sum over every "
        "point, while the body skips the points the condition excludes. isl "
        "states an affine comparison of loop variables and sizes, and this "
        f"condition {why}.{split} Otherwise write each term to an indexed cell, "
        "0.0 where the condition is false and the term under "
        "'with when(condition):', and sum the cells."
    )


def _binder_constraints(
    enclosing: Sequence[tuple[str, Any]], tracer: Tracer
) -> tuple[str, ...]:
    """``0 <= i < bound`` for each enclosing reduction binder, as isl text.

    Rendered through the tracer's reflection table, so that a ragged outer bound
    such as ``cnt[r]`` is the same parameter here as in every other set about
    the term.
    """
    reflected: dict[str, Any] = {}
    return tuple(
        f"0 <= {name} < {expr_text(bound, None, reflected, tracer.reflections)}"
        for name, bound in enclosing
    )


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

    An explicit :class:`~loopty.term.Access` inside an expression is recognized
    too. The tracer never builds one there, but a term written by hand may, and
    this is the collector every rule now goes through.
    """
    out: list[Access] = []

    def walk(node: Any) -> None:
        if isinstance(node, prim.Subscript):
            if isinstance(node.aggregate, prim.Variable):
                out.append(Access(node.aggregate.name, _index_tuple(node.index)))
            walk(node.index)
            return
        if isinstance(node, Access):
            out.append(node)
            walk(node.indices)
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
