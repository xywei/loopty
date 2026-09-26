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
carry it and have it intersected into their domain where isl can state it, an
affine comparison of integers (loop variables, sizes, scalars of an integral
sort), so a guarded access is in bounds exactly where it is executed. Any other
conjunct is left to the statement's guard, evaluated at run time, and the
statement records it as leaving its domain wider than the instances that write
(:attr:`loopty.term.Stmt.unnarrowed`). Under plain
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

A body also has effects that are not array writes, and one trace records none
of them. So a symbolic array refuses to be used whole (``y[:] = ...``,
``x * 2``, ``for v in x``, a numpy function of it), with the loop nest that
does the same one cell at a time as the fix. And :func:`trace` copies the state
the body's code reaches by name outside itself (module globals, closure cells,
defaults, one level into the containers, arrays and objects they hold, and the
same for the helpers it calls) and compares it once the body has run; a change
is refused, and so is a call that prints, reads input, opens a file or draws a
random number, which :class:`_CallWatch` sees through ``sys.monitoring``. Code
is the kernel author's or a library's by its module (:func:`_library`), so a
kernel installed into site-packages is watched like one in a source tree.
State hidden deeper than that is what the faithfulness fact is for; see
:mod:`loopty.faithful`.
"""

from __future__ import annotations

import builtins
import dis
import functools
import importlib.util
import inspect
import itertools
import os
import random
import sys
import sysconfig
import threading
from collections.abc import Callable, Collection, Iterator, Mapping, Sequence
from dataclasses import dataclass
from types import (
    BuiltinFunctionType,
    CodeType,
    FunctionType,
    MemberDescriptorType,
    MethodType,
    ModuleType,
    SimpleNamespace,
)
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
from loopty.flow import domain_set, expr_text, free_names
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
    loop iteration to the next, a loop whose code can ask which names are
    bound, an operation on a whole array, a reduction condition its domain
    cannot state, a change to Python state outside the arrays, and a call that
    prints, reads, opens a file or draws a random number are the other cases.

    One case is raised by a native run as well as by a trace: a ``when`` guard
    whose value is an integer rather than a truth value, which is what ``~``
    makes of a Python bool.
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
        #: The calls with an effect outside the arrays that the body made, as
        #: ``(call, file:line)``, recorded by :class:`_CallWatch` and refused
        #: once the body has run.
        self.effects: list[tuple[str, str]] = []
        #: The thread running the body; calls on any other are not its effects.
        self.thread = threading.get_ident()
        #: The names a guard may hand isl as integers: the sizes, the scalar
        #: parameters of an integral sort (filled in by :func:`trace` from the
        #: signature), and every loop variable, added as it is made. isl holds
        #: every name of a constraint as an integer, so a comparison naming
        #: anything else is not stated as one; see :func:`constraints_of`.
        self.integers: set[str] = set()
        #: The kernel's own code, which is watched and followed wherever it is
        #: installed; see :class:`_Own`. :func:`trace` sets it.
        self.own: _Own | None = None

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
        self.integers.add(name)
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
        """The isl set of the enclosing loop nest, narrowed by the guards.

        Only a comparison isl can state narrows it: an affine one, over loop
        variables, sizes and scalars of an integral sort (see
        :func:`constraints_of`). Every other conjunct of the guard is left to
        the statement's guard predicate, evaluated at run time, and the domain
        is wider than the instances that write; :meth:`unnarrowed` says which
        conjuncts, and why.
        """
        return domain_set(
            self.inames,
            self.bounds,
            constraints=constraints_of(self.guard(), self),
            reflections=self.reflections,
        )

    def unnarrowed(self) -> tuple[tuple[str, str], ...]:
        """The conjuncts of the open guards that do not narrow :meth:`domain`.

        Each is ``(conjunct, why)``, the conjunct as the body spells it. A
        statement records them (:attr:`loopty.term.Stmt.unnarrowed`), so that
        the facts stated over its domain can say that the domain is an
        over-approximation, and of what.
        """
        return tuple(
            (_shown(part), why) for part, why in _unstated(self.guard(), self)
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
            unnarrowed=self.unnarrowed(),
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


# {{{ state and effects outside the trace


#: The top-level packages that are the machinery rather than the kernel. Their
#: objects are not snapshotted, their functions are not followed, and a call
#: made from their code is not the body's effect.
_LIBRARIES = ("loopty", "lanky", "numpy", "pymbolic", "islpy", "loopy", "pytools")

#: How many levels of helper call the snapshot follows, as guard detection does
#: (:data:`loopty.kernel._GUARD_SEARCH_DEPTH`).
_HELPER_DEPTH = 8


@functools.cache
def _library_roots() -> tuple[str, ...]:
    """The directories whose code belongs to a library and not to a kernel.

    The packages of :data:`_LIBRARIES`, found without importing them, and the
    standard library and site-packages directories. A kernel in a source tree,
    and a helper next to it, are in none of them; a kernel installed into
    site-packages is in one, and :func:`_library` exempts it by its module.
    """
    roots: list[str] = []
    for name in _LIBRARIES:
        try:
            spec = importlib.util.find_spec(name)
        except (ImportError, ValueError):  # pragma: no cover - a broken install
            spec = None
        for location in getattr(spec, "submodule_search_locations", None) or ():
            roots.append(os.path.realpath(location))
    paths = sysconfig.get_paths()
    for key in ("stdlib", "platstdlib", "purelib", "platlib"):
        if key in paths:
            roots.append(os.path.realpath(paths[key]))
    return tuple(dict.fromkeys(roots))


@functools.lru_cache(maxsize=4096)
def _library_file(filename: str) -> bool:
    """Whether code from ``filename`` is in a library's directory.

    That is a question about the path alone. A kernel installed into
    site-packages is in one, and is still the kernel's code; :func:`_library`
    is the question every check asks, and it asks this one last.
    """
    if filename.startswith("<frozen"):
        return True
    if filename.startswith("<"):
        return False
    path = os.path.realpath(filename)
    return any(
        path == root or path.startswith(root + os.sep) for root in _library_roots()
    )


def _machinery_module(module: str | None, filename: str | None = None) -> bool:
    """Whether ``module``, with its code in ``filename``, is loopty's machinery.

    A module of :data:`_LIBRARIES` is, by the name of its top-level package,
    whatever directory it was imported from: an editable install of one is in
    a source tree. A module with a standard library name is when its code is
    where the standard library is (:func:`_library_file`), or has no file: a
    module of the author's that takes such a name, ``wave.py`` next to the
    script that imports it, is the author's code and not the standard
    library's. No kernel is written in the machinery, so its code is machinery
    for every trace.
    """
    if not module:
        return False
    top = module.partition(".")[0]
    if top in _LIBRARIES:
        return True
    if top not in sys.stdlib_module_names:
        return False
    return filename is None or _library_file(filename)


@dataclass(frozen=True)
class _Own:
    """The kernel's own code, which is never a library's, wherever it is installed.

    ``module`` is the name of the module the body is defined in, and
    ``package`` the top-level package holding it, or ``None`` when that package
    is machinery (:func:`_machinery_module`), whose other modules stay
    machinery. A kernel installed into site-packages by a non-editable install
    lives in a library's directory, and it is still the kernel's code: its
    calls are watched, the helpers of its package followed and the objects of
    its classes copied, as they are for a kernel in a source tree.
    """

    module: str
    package: str | None

    def holds(self, module: str | None) -> bool:
        """Whether code of ``module`` is the kernel's own."""
        if not module:
            return False
        if module == self.module:
            return True
        package = self.package
        return package is not None and (
            module == package or module.startswith(package + ".")
        )


def _own_of(function: Any) -> _Own | None:
    """The kernel code of ``function``: its module, and the package holding it.

    The package is the top-level one, or, when that is a namespace package
    (a directory several distributions install into, with no ``__init__``),
    the first regular package below it, which is the kernel author's alone.
    It is read off the module's ``__package__`` as well as its name, because
    a module can sit in a package without being named after it: ``lanky
    check`` imports a file under a name of its own (``lanky_checked_kernels``)
    and gives it the package the file is in, and ``python -m`` runs a module
    of a package as ``__main__``.
    """
    if isinstance(function, MethodType):
        function = function.__func__
    module = _module_of_function(function)
    if not module:
        return None
    code = getattr(function, "__code__", None)
    filename = getattr(code, "co_filename", None)
    if _machinery_module(module, filename):
        return _Own(module, None)
    dotted = module
    parent = function.__globals__.get("__package__")
    if (
        isinstance(parent, str)
        and parent
        and module != parent
        and not module.startswith(parent + ".")
    ):
        dotted = f"{parent}.{module.rpartition('.')[2]}"
    parts = dotted.split(".")
    if _machinery_module(parts[0], filename):
        # A file checked from inside one of loopty's dependencies is its
        # own module there, and the rest of the package stays machinery.
        return _Own(module, None)
    package = parts[0]
    for depth in range(1, len(parts) + 1):
        package = ".".join(parts[:depth])
        found = sys.modules.get(package)
        namespace = (
            found is not None
            and getattr(found, "__file__", None) is None
            and hasattr(found, "__path__")
        )
        if not namespace:
            break
    return _Own(module, package)


def _library(module: str | None, filename: str | None, own: _Own | None) -> bool:
    """Whether code of ``module``, in ``filename``, is a library's for this trace.

    Decided by module. The kernel's own module and the package holding it
    (``own``) are never a library's; :data:`_LIBRARIES` and the standard
    library always are (:func:`_machinery_module`); anything else is a
    library's when its file is in a library's directory (:func:`_library_file`),
    which is where a third-party package installed next to the kernel lives.
    Code whose module is not known is judged by its file alone.
    """
    if own is not None and own.holds(module):
        return False
    if _machinery_module(module, filename):
        return True
    return filename is not None and _library_file(filename)


def _module_of_function(function: Any) -> str | None:
    """The module a function's code runs in, by its globals."""
    namespace = getattr(function, "__globals__", None)
    module = namespace.get("__name__") if isinstance(namespace, dict) else None
    return module if isinstance(module, str) else None


def _user_object(value: Any, own: _Own | None = None) -> bool:
    """Whether ``value`` is an object of the kernel author's, with attributes.

    Modules, classes, functions and code are not, and neither is an object of
    a library's type (a lanky sort, a numpy array, a ``logging.Logger``, an
    object from a third-party package): its attributes are the library's
    business, and may change while a body is traced without the body having
    done anything, as a logger's level cache does on its first ``debug`` call.
    A type is a library's when the module defining it is (:func:`_library`),
    so a class of the kernel's own package (``own``) is the author's wherever
    the package is installed. A :class:`types.SimpleNamespace` is the
    exception: a bag of attributes with no machinery of its own, so what it
    holds is exactly what the kernel author put there.

    Attributes are what :func:`_attributes` reads, the ``__dict__`` and the
    slots, so an object of a class with ``__slots__`` counts.
    """
    if isinstance(
        value, ModuleType | type | FunctionType | MethodType | BuiltinFunctionType
    ):
        return False
    if type(value) is SimpleNamespace:
        return True
    module = getattr(type(value), "__module__", None) or ""
    if module == "builtins":
        return False
    defined_in = getattr(sys.modules.get(module), "__file__", None)
    if _library(module, defined_in if isinstance(defined_in, str) else None, own):
        return False
    return _attributes(value) is not None


def _attributes(value: Any) -> dict[str, Any] | None:
    """The attributes ``value`` keeps, in its slots and in its ``__dict__``.

    Every slot named along the class's MRO is read through the member
    descriptor its class defines, under its mangled name when it is private
    (``__slots__ = ("__n",)`` in ``class Box`` keeps ``_Box__n``), and one that
    is not set is left out. Each descriptor is a cell of its own, so a slot a
    subclass declares again does not hide its base's: the base's is kept as
    ``Base.x``, and an entry of the ``__dict__`` that a slot's name hides as
    ``__dict__['x']``. Which name a cell gets depends on the classes alone, not
    on which cells are set, so two snapshots name the same cells alike.
    ``None`` when the object keeps attributes in neither place, as an
    ``object()`` or a number does.
    """
    out: dict[str, Any] = {}
    found = False
    declared: set[str] = set()
    for cls in type(value).__mro__:
        slots = cls.__dict__.get("__slots__")
        if slots is None:
            continue
        found = True
        for slot in (slots,) if isinstance(slots, str) else slots:
            if slot in ("__dict__", "__weakref__"):
                continue
            name = _mangled(cls, slot)
            descriptor = cls.__dict__.get(name)
            if not isinstance(descriptor, MemberDescriptorType):
                continue
            key = name if name not in declared else f"{cls.__qualname__}.{name}"
            declared.add(name)
            try:
                out.setdefault(key, descriptor.__get__(value, cls))
            except AttributeError:
                continue
    try:
        entries = vars(value)
    except TypeError:
        pass
    else:
        found = True
        for name, item in entries.items():
            key = name if name not in declared else f"__dict__[{name!r}]"
            out.setdefault(key, item)
    return out if found else None


def _mangled(cls: type, name: str) -> str:
    """``name`` as Python stores it in ``cls``: ``__n`` in ``Box`` is ``_Box__n``."""
    stem = cls.__name__.lstrip("_")
    if name.startswith("__") and not name.endswith("__") and stem:
        return f"_{stem}{name}"
    return name


@dataclass
class _Name:
    """A name the traced code reaches outside itself, and its value then.

    ``holder`` is how a message names it (``the global 'G'``), ``scope`` is
    ``"global"``, ``"closure"`` or ``"default"``, and ``read`` gives its value
    now.
    """

    holder: str
    name: str
    scope: str
    read: Callable[[], Any]
    before: Any


@dataclass
class _Attributes:
    """An object a name holds, with a copy of its attributes then."""

    label: str
    root: str
    scope: str
    obj: Any
    copy: dict[str, Any]


@dataclass
class _Outside:
    """What the body's code reached outside the trace when it started.

    ``names`` are the globals and closure cells it names, ``held`` the lists,
    dicts and sets those hold (directly, through tuples, or as an attribute of
    an object they hold), ``buffers`` the numpy arrays they hold in the same
    places, each with a copy of its cells, and ``objects`` the kernel author's
    objects they hold, each with its attributes. See :func:`_outside`.
    """

    names: list[_Name]
    held: list[_Held]
    buffers: list[_Held]
    objects: list[_Attributes]


@dataclass(frozen=True)
class _Change:
    """One piece of outside state the trace changed, with both values."""

    holder: str
    cell: str
    before: Any
    after: Any


def _read_global(namespace: Mapping[str, Any], name: str) -> Any:
    """A module global's value now, or :data:`_UNBOUND`."""
    return namespace[name] if name in namespace else _UNBOUND


def _read_cell(cell: Any) -> Any:
    """A closure cell's value now, or :data:`_UNBOUND`."""
    try:
        return cell.cell_contents
    except ValueError:
        return _UNBOUND


def _outside(
    function: Any, owned: Collection[int], own: _Own | None = None
) -> _Outside:
    """Copy the state the body's code reaches by name outside the trace.

    The roots are the globals the code of ``function`` names (with the code of
    the functions defined inside it, and whether the module binds them yet or
    not), its closure cells and its default values, and the same for every
    function of the kernel author's that those hold, :data:`_HELPER_DEPTH`
    levels deep: a helper keeping a counter in its module is state too. A
    function is the author's when its module is not a library's for this trace
    (:func:`_library`, with ``own`` the kernel's own code). Each value is
    copied one level deep, as the loop snapshot is: a list, dict or set
    shallowly, looking through tuples, a numpy array cell by cell (an
    :class:`~loopty.arr.Arr` by its buffer, and a ragged one by its offsets as
    well), and an object of the kernel author's (:func:`_user_object`) by its
    attributes, slots included, with a list, dict, set or array an attribute
    holds copied as well. A container or object reached twice is copied once,
    and one the tracer keeps for itself (``owned``) not at all.

    A numpy array is copied whole, because a write into it is an effect the
    compiled kernel never makes even when no output reads it back, which the
    faithfulness fact, comparing outputs, cannot see. A body that only reads
    a global array pays for one copy per trace.
    """
    names: list[_Name] = []
    held: list[_Held] = []
    buffers: list[_Held] = []
    objects: list[_Attributes] = []
    seen: set[int] = set(owned)
    recorded: set[Any] = set()
    walked: set[int] = set()
    if isinstance(function, MethodType):
        function = function.__func__
    pending: list[tuple[Any, int]] = (
        [(function, 0)] if isinstance(function, FunctionType) else []
    )

    def visit(
        value: Any, label: str, root: str, scope: str, level: int, depth: int
    ) -> None:
        if id(value) in seen:
            return
        if isinstance(value, tuple):
            for position, item in enumerate(value):
                visit(item, f"{label}[{position}]", root, scope, level, depth)
            return
        for kind in (list, dict, set):
            if isinstance(value, kind):
                seen.add(id(value))
                held.append(_Held(label, root, scope, value, kind(value)))
                return
        if isinstance(value, Arr):
            # A ragged array keeps two buffers, and a write into its offsets
            # moves where every row starts.
            parts = [(label, value.numpy())]
            if value.is_ragged:
                parts.append((f"{label}.offsets", value.offsets))
        elif isinstance(value, np.ndarray):
            parts = [(label, value)]
        else:
            parts = []
        if parts:
            for part_label, buffer in parts:
                if id(buffer) not in seen:
                    seen.add(id(buffer))
                    buffers.append(
                        _Held(part_label, root, scope, buffer, buffer.copy())
                    )
            return
        if isinstance(value, MethodType):
            value = value.__func__
        if isinstance(value, FunctionType):
            if depth < _HELPER_DEPTH and not _library(
                _module_of_function(value), value.__code__.co_filename, own
            ):
                pending.append((value, depth + 1))
            return
        if level == 0 and _user_object(value, own):
            seen.add(id(value))
            attributes = _attributes(value) or {}
            objects.append(_Attributes(label, root, scope, value, attributes))
            for attribute, item in attributes.items():
                visit(item, f"{label}.{attribute}", root, scope, 1, depth)

    while pending:
        current, depth = pending.pop()
        if id(current) in walked:
            continue
        walked.add(id(current))
        code = current.__code__
        namespace = current.__globals__
        used: set[str] = set()
        for nested in _codes(code):
            used.update(nested.co_names)
        for name in sorted(used):
            # A name the module does not bind yet is recorded as unbound: a
            # body that creates a global with ``global G`` changes it too.
            if (id(namespace), name) in recorded:
                continue
            recorded.add((id(namespace), name))
            value = _read_global(namespace, name)
            names.append(
                _Name(
                    f"the global {name!r}",
                    name,
                    "global",
                    functools.partial(_read_global, namespace, name),
                    value,
                )
            )
            if value is not _UNBOUND:
                visit(value, name, name, "global", 0, depth)
        cells = current.__closure__ or ()
        for name, cell in zip(code.co_freevars, cells, strict=False):
            if id(cell) in recorded:
                continue
            recorded.add(id(cell))
            value = _read_cell(cell)
            names.append(
                _Name(
                    f"the closure variable {name!r}",
                    name,
                    "closure",
                    functools.partial(_read_cell, cell),
                    value,
                )
            )
            if value is not _UNBOUND:
                visit(value, name, name, "closure", 0, depth)
        positional = code.co_varnames[: code.co_argcount]
        defaults = current.__defaults__ or ()
        pairs = [
            *zip(positional[len(positional) - len(defaults) :], defaults, strict=True),
            *(current.__kwdefaults__ or {}).items(),
        ]
        for name, value in pairs:
            visit(value, name, name, "default", 0, depth)
    return _Outside(names, held, buffers, objects)


def _outside_changes(outside: _Outside, tracer: Tracer) -> list[_Change]:
    """What the trace changed of the state :func:`_outside` copied.

    Values are compared as the loop snapshot compares them
    (:func:`_same_value`, never ``==``). Two things are left alone, as they are
    across a loop iteration: a name now bound to a :class:`when`, and a name now
    bound to a loop's own variable, which is what a ``for`` whose target is a
    global stores there; anything that reads it is refused as an escaped loop
    variable. So is an attribute a :class:`functools.cached_property` stored
    the first time the body read it: the property computes it from the object,
    once, whoever asks first, and a native call reads the same value. A
    container whose name was rebound is reported as that rebinding.
    """
    loops = tracer._inames

    def exempt(after: Any) -> bool:
        if isinstance(after, when):
            return True
        return isinstance(after, prim.Variable) and after.name in loops

    out: list[_Change] = []
    rebound: set[tuple[str, str]] = set()
    for entry in outside.names:
        after = entry.read()
        if _same_value(entry.before, after) or exempt(after):
            continue
        rebound.add((entry.scope, entry.name))
        out.append(_Change(entry.holder, entry.name, entry.before, after))
    for held in outside.held:
        if (held.scope, held.root) in rebound:
            continue
        change = _changed(held, ())
        if change is None:
            continue
        cell, before, after = change
        kind = type(held.container).__name__
        if held.label == held.root:
            where = "closure " if held.scope == "closure" else f"{held.scope} "
            holder = f"the {where}{kind} {held.label!r}"
        else:
            holder = f"the {kind} {held.label!r}"
        out.append(_Change(holder, cell, before, after))
    for buffer in outside.buffers:
        if (buffer.scope, buffer.root) in rebound:
            continue
        change = _buffer_change(buffer)
        if change is not None:
            where = f"{buffer.scope} " if buffer.label == buffer.root else ""
            out.append(_Change(f"the {where}array {buffer.label!r}", *change))
    for entry in outside.objects:
        if (entry.scope, entry.root) in rebound:
            continue
        now = _attributes(entry.obj) or {}
        for attribute in dict.fromkeys([*entry.copy, *now]):
            before = entry.copy.get(attribute, _UNBOUND)
            after = now.get(attribute, _UNBOUND)
            if _same_value(before, after) or exempt(after):
                continue
            if before is _UNBOUND and _cached_property(entry.obj, attribute):
                continue
            out.append(
                _Change(
                    f"the attribute {attribute!r} of {entry.label!r}",
                    f"{entry.label}.{attribute}",
                    before,
                    after,
                )
            )
            break
    return out


def _buffer_change(buffer: _Held) -> tuple[str, Any, Any] | None:
    """The first cell of a numpy array that changed since it was copied, if any.

    Cells are compared by their bits, so ``-0.0`` over ``0.0`` is a change,
    and the cells of an object array as the loop snapshot compares values
    (:func:`_same_value`). An array resized in place is named whole.
    """
    old, new, label = buffer.copy, buffer.container, buffer.label
    if old.shape != new.shape or old.dtype != new.dtype:
        return label, old.tolist(), new.tolist()
    flat_old = np.ascontiguousarray(old).reshape(-1)
    flat_new = np.ascontiguousarray(new).reshape(-1)
    if old.dtype.hasobject:
        differ = np.array(
            [not _same_value(a, b) for a, b in zip(flat_old, flat_new, strict=True)],
            dtype=bool,
        )
    elif flat_old.tobytes() == flat_new.tobytes():
        return None
    else:
        width = old.dtype.itemsize
        differ = np.any(
            flat_old.view(np.uint8).reshape(-1, width)
            != flat_new.view(np.uint8).reshape(-1, width),
            axis=-1,
        )
    if not differ.any():
        return None
    position = int(np.flatnonzero(differ)[0])
    index = np.unravel_index(position, old.shape)
    cell = f"{label}[{', '.join(str(int(k)) for k in index)}]"
    return cell, _python_cell(flat_old[position]), _python_cell(flat_new[position])


def _python_cell(value: Any) -> Any:
    """One cell of a numpy array as the Python value a message shows."""
    return value.item() if isinstance(value, np.generic) else value


def _cached_property(obj: Any, attribute: str) -> bool:
    """Whether ``attribute`` of ``obj`` holds the value of a ``cached_property``."""
    try:
        descriptor = inspect.getattr_static(type(obj), attribute)
    except AttributeError:
        return False
    return isinstance(descriptor, functools.cached_property)


def _outside_message(name: str, changes: Sequence[_Change]) -> str:
    """What to say about Python state outside the arrays that a trace changed."""
    holders = ", ".join(change.holder for change in changes)
    values = "; ".join(
        f"{change.cell!r} is {_shown(change.before)} before the trace and "
        f"{_shown(change.after)} after it"
        for change in changes
    )
    return (
        f"tracing {name} changed {holders} ({values}), Python state outside the "
        "kernel's array parameters. Tracing runs the body once, at one generic "
        "point, so such a change happens once in the trace and once per call "
        "natively, and the compiled kernel never makes it. Keep the state in an "
        "array parameter and write it at an index, or change it outside the "
        "kernel."
    )


def _effect_of(function: Any, first: Any) -> str | None:
    """The effect outside the arrays that calling ``function`` has, if known.

    Printing, reading input and opening a file, and drawing a random number
    from :mod:`random` or from numpy's generators, which changes a generator's
    hidden state and bakes one draw into the term as a constant. ``first`` is
    the call's first argument, which is the object an unbound method is
    called on.
    """
    if function is builtins.print or function is builtins.input:
        return f"{function.__name__}()"
    if function is builtins.open:
        return "open()"
    name = getattr(function, "__name__", None)
    if not isinstance(name, str):
        return None
    owner = getattr(function, "__self__", None)
    if (owner is None or isinstance(owner, ModuleType)) and not isinstance(
        function, FunctionType
    ):
        # ``rng.normal()`` calls the unbound method with the generator first.
        prefix = type(first).__qualname__ + "."
        if getattr(function, "__qualname__", "").startswith(prefix):
            owner = first
    if isinstance(owner, random.Random):
        return f"random.{name}()"
    generators = sys.modules.get("numpy.random")
    kinds = tuple(
        getattr(generators, kind, None)
        for kind in ("RandomState", "Generator", "BitGenerator")
    )
    kinds = tuple(kind for kind in kinds if isinstance(kind, type))
    if kinds and isinstance(owner, kinds):
        return f"numpy.random.{type(owner).__name__}.{name}()"
    return None


def _call_location(code: CodeType, offset: int) -> str:
    """``file:line`` of the instruction at ``offset``, from ``co_positions``."""
    line = None
    for index, position in enumerate(code.co_positions()):
        if index == offset // 2:
            line = position[0]
            break
    return f"{os.path.basename(code.co_filename)}:{line or code.co_firstlineno}"


class _CallWatch:
    """The body's calls, seen through ``sys.monitoring`` while it is traced.

    One tool id is taken, the first of :data:`IDS` that nothing holds, the
    first time a body is traced, and kept. Its ``CALL`` events are on only
    while a trace runs. A call made from a library's code is never the body's.
    Which code is a library's is decided by the module the calling frame runs
    in (:func:`_library`), so the kernel's own module and package are watched
    even when they are installed into site-packages. A call location in
    :data:`_LIBRARIES` or the standard library is disabled for good, so that
    the tracer's own calls cost nothing after the first trace; one in another
    installed package is only passed over, because that package may hold the
    kernel of a later trace, and a disabled location stays disabled. A call
    from the kernel author's code that has an effect (:func:`_effect_of`) is
    recorded on the tracer, and :func:`trace` refuses it once the body has run.

    Without a free tool id nothing is watched, and the faithfulness fact is
    what remains.
    """

    NAME = "loopty-trace"
    IDS = (4, 3)
    tool: int | None = None
    tried = False
    depth = 0

    @classmethod
    def start(cls) -> bool:
        """Turn the events on; whether they are on."""
        monitoring = getattr(sys, "monitoring", None)
        if monitoring is None:  # pragma: no cover - Python 3.11 and older
            return False
        if not cls.tried:
            cls.tried = True
            for tool in cls.IDS:
                if monitoring.get_tool(tool) is None:
                    monitoring.use_tool_id(tool, cls.NAME)
                    monitoring.register_callback(
                        tool, monitoring.events.CALL, cls.on_call
                    )
                    cls.tool = tool
                    break
        if cls.tool is None:
            return False
        if cls.depth == 0:
            monitoring.set_events(cls.tool, monitoring.events.CALL)
        cls.depth += 1
        return True

    @classmethod
    def stop(cls) -> None:
        """Turn the events off again when the outermost trace ends."""
        cls.depth -= 1
        if cls.depth == 0 and cls.tool is not None:
            sys.monitoring.set_events(cls.tool, sys.monitoring.events.NO_EVENTS)

    @staticmethod
    def on_call(code: CodeType, offset: int, function: Any, first: Any) -> Any:
        """The ``CALL`` callback: record an effect of the body's own code."""
        try:
            tracer = current_tracer()
            if tracer is None or tracer.thread != threading.get_ident():
                return None
            # The frame one below this callback is the one making the call.
            frame = sys._getframe(1)
            module = (
                frame.f_globals.get("__name__") if frame.f_code is code else None
            )
            if not isinstance(module, str):
                module = None
            if _library(module, code.co_filename, tracer.own):
                if _machinery_module(
                    module, code.co_filename
                ) or code.co_filename.startswith("<frozen"):
                    return sys.monitoring.DISABLE
                return None
            effect = _effect_of(function, first)
            if effect is not None:
                tracer.effects.append((effect, _call_location(code, offset)))
        except Exception:  # noqa: BLE001 - a watcher never breaks the body
            return None
        return None


def _effects_message(name: str, effects: Sequence[tuple[str, str]]) -> str:
    """What to say about calls with an effect outside the arrays."""
    listed = ", ".join(f"{call} at {where}" for call, where in dict.fromkeys(effects))
    return (
        f"tracing {name} calls {listed}, an effect outside the kernel's array "
        "parameters. Tracing runs the body once, at one generic point, so the "
        "call happens once in the trace and once per iteration natively, and "
        "the compiled kernel never makes it: a print shows one symbolic value, "
        "and a random draw becomes a constant of the term. Print from the code "
        "that calls the kernel, read and write files there, and draw random "
        "numbers there and pass them in an array parameter."
    )


# }}}


# {{{ guards as isl constraints


#: Why a conjunct that reads an array, or is not an affine comparison, is not
#: a constraint.
_NOT_AFFINE = "reads an array or is not affine"

#: Why a conjunct comparing with ``!=`` is not a constraint.
_UNEQUAL = "compares with '!=', which is not a convex set of points"

#: Why a guard that is already false is not a constraint.
_FALSE = "is the constant False, which is not stated to isl"


def constraints_of(
    condition: Any, tracer: Tracer | None = None, bound: Collection[str] = ()
) -> tuple[str, ...]:
    """Render a guard as isl constraints, dropping what isl cannot express.

    A guard narrows the statement's domain, which is what makes
    ``with when(i + 1 < n): u[i + 1] = ...`` provably in bounds. A guard that is
    not quasi-affine (a data-dependent test) is dropped, which widens the domain
    and can only make an obligation harder, never falsely discharge one.

    isl holds every name of a constraint as an integer. With a ``tracer``, a
    comparison is kept only when every name in it is one the tracer knows to
    be an integer (:attr:`Tracer.integers`: a loop variable, a size, a scalar
    of sort ``Nat``, ``Int`` or ``Fin[...]``) or one of the reduction binders
    in ``bound``. ``i < a`` with ``a : Real`` is dropped: stated to isl it
    reads ``a`` as an integer parameter, and at ``a = 2.5`` the domain, and
    the compiled kernel built from it, would disagree with the native run.
    Without a tracer every name is taken for an integer, which is right for a
    hand-built guard over sizes and loop variables only.
    """
    return tuple(
        text for _part, text, _why in _conjuncts(condition, tracer, bound) if text
    )


def _unstated(
    condition: Any, tracer: Tracer | None = None, bound: Collection[str] = ()
) -> list[tuple[Any, str]]:
    """The conjuncts :func:`constraints_of` drops, each with the reason."""
    return [
        (part, why)
        for part, text, why in _conjuncts(condition, tracer, bound)
        if text is None
    ]


def _conjuncts(
    condition: Any, tracer: Tracer | None, bound: Collection[str]
) -> list[tuple[Any, str | None, str]]:
    """``(conjunct, constraint, why)`` for each conjunct of a guard.

    ``constraint`` is the conjunct's isl text, or ``None`` when it cannot be
    one, and then ``why`` says why.
    """
    if condition is None:
        return []
    if isinstance(condition, bool | np.bool_):
        # A guard the trace computed, from values it knows: true leaves every
        # instance of the loop nest writing, which is what the domain says.
        return [] if condition else [(condition, None, _FALSE)]
    if isinstance(condition, prim.LogicalAnd):
        return [
            piece
            for child in condition.children
            for piece in _conjuncts(child, tracer, bound)
        ]
    if not isinstance(condition, prim.Comparison):
        return [(condition, None, _NOT_AFFINE)]
    try:
        left = expr_text(condition.left, None, None)
        right = expr_text(condition.right, None, None)
    except ValueError:
        return [(condition, None, _NOT_AFFINE)]
    operator = condition.operator
    if operator == "!=":
        return [(condition, None, _UNEQUAL)]
    if tracer is not None:
        known = tracer.integers.union(bound)
        others = sorted((free_names(left) | free_names(right)) - known)
        if others:
            return [(condition, None, _not_integers_text(others, tracer))]
    # isl spells equality with one '='; its parser refuses Python's '=='.
    if operator == "==":
        operator = "="
    return [(condition, f"{left} {operator} {right}", "")]


def _not_integers_text(names: Sequence[str], tracer: Tracer) -> str:
    """Why a comparison naming ``names`` is not a constraint: isl's are integers."""
    described = []
    for name in names:
        sort = tracer.params.get(name)
        if name in tracer.params and not isinstance(sort, ArrSpec):
            described.append(f"the scalar {name} of sort {_sort_text(sort)}")
        else:
            described.append(name)
    listed = " and ".join(described)
    which = "which is" if len(names) == 1 else "which are"
    return (
        f"compares with {listed}, {which} not a loop variable, a size or a "
        "scalar of an integral sort (Nat, Int, Fin[...]), and isl would read "
        "every name of a constraint as an integer"
    )


def _sort_text(sort: Any) -> str:
    """A sort the way a message names it: ``Real``, ``float64``."""
    if isinstance(sort, type):
        return sort.__name__
    return str(sort)


def _integer_names(params: Sequence[tuple[str, Any]]) -> set[str]:
    """The names of a signature that are integers, other than loop variables.

    The sizes an array's axes name, the sizes a ``Fin`` sort names (an element
    sort ``Fin[m]``, or a scalar's ``i : Fin[n]``), and the scalar parameters
    of an integral sort (:func:`loopty.contract.integral_sort`). An array
    parameter is none of them, even where a ragged axis names it as its counts.
    """
    from loopty.contract import integral_sort

    arrays = {name for name, typ in params if isinstance(typ, ArrType)}
    out: set[str] = set()
    for name, typ in params:
        if isinstance(typ, ArrType):
            out |= _free_size_names(typ.axes)
            sort = typ.dtype
        else:
            sort = typ
            if integral_sort(sort):
                out.add(name)
        base = sort.base if isinstance(sort, Refined) else sort
        if isinstance(base, FinType):
            out |= _free_size_names([base.bound])
    return out - arrays


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
    of that row, which is where the dependent sum enters the type.
    ``arr.dom[r, i]`` is ``arr.dom[r][i]``. Iterating in a ``for`` loop opens a
    loop level; inside a ``loopty.reduce_sum`` generator, iteration binds a
    reduction variable instead, because Lanky is driving it and asks for the
    point itself.
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
        """The fiber over ``index``: the domain of the next axis.

        A tuple fixes one axis per entry, so ``a.dom[r, i]`` is
        ``a.dom[r][i]``, as it is on a runtime array. It used to be taken for
        one index, the tuple, and gave the domain of axis 1 whatever the length
        of the tuple.
        """
        if _whole_key(index):
            raise TraceError(
                f"{_domain_text(self)}[{_key_text(index)}] at "
                f"{_location(sys._getframe(1))} takes a fiber over more than "
                "one point. A fiber is taken at one index, "
                f"{_domain_text(self)}[i]; to leave points of a domain out, "
                "iterate all of it and put the statements under "
                "'with when(condition):'"
            )
        if isinstance(index, tuple):
            if not index:
                raise TraceError(
                    f"{self.array.name}.dom[()] fixes no axis; a domain index "
                    "needs at least one entry"
                )
            fiber = self
            for part in index:
                fiber = fiber[part]
            return fiber
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
        self._refuse_whole(key, _location(sys._getframe(1)), write=False)
        return _subscript(self.name, _index_tuple(key))

    def __setitem__(self, key: Any, value: Any) -> None:
        """Write: record a statement at the caller's file and line."""
        where = _location(sys._getframe(1))
        self._refuse_whole(key, where, write=True)
        indices = _index_tuple(key)
        tracer = self.tracer
        expr = lower_reductions(value, tracer, where=where)
        assignee = Access(self.name, indices)
        kind = "accumulate" if _reads_assignee(expr, assignee) else "assign"
        tracer.record(assignee, expr, kind, where, source=value)

    def _refuse_whole(self, key: Any, where: str, write: bool) -> None:
        """Refuse a subscript that names more than one cell.

        A slice, an ``...``, a list or an array of indices, or fewer indices
        than the array has axes (``u[t]`` of a two-axis ``u`` is a row) name
        many cells at once, and a statement is one cell per instance. Natively
        numpy would do the operation on all of them; the trace would record one
        statement with a slice for an index, which nothing downstream reads as
        a loop.
        """
        spelled = f"{self.name}[{_key_text(key)}]" + (" = ..." if write else "")
        if _whole_key(key):
            raise TraceError(_whole_array_message(self, spelled, where, write))
        given = len(_index_tuple(key))
        if given < self.ndim:
            raise TraceError(
                _whole_array_message(
                    self,
                    spelled,
                    where,
                    write,
                    why=(
                        f"gives {given} of the {self.ndim} indices of "
                        f"{self.name}, so it names every cell that has them"
                    ),
                )
            )

    def __iter__(self) -> Any:
        """Refuse: iterating an array walks its cells, a whole-array operation."""
        raise TraceError(
            _whole_array_message(
                self,
                f"iterating {self.name} itself",
                _location(sys._getframe(1)),
                write=False,
            )
        )

    def numpy(self) -> Any:
        """Refuse: a symbolic array has no storage to hand over."""
        raise TraceError(
            _whole_array_message(
                self,
                f"{self.name}.numpy()",
                _location(sys._getframe(1)),
                write=False,
                why="asks for the storage of the array, and a traced array has none",
            )
        )

    def __array__(self, *args: Any, **kwargs: Any) -> Any:
        """Refuse: numpy cannot hold a symbolic array."""
        raise TraceError(
            _whole_array_message(self, f"converting {self.name} to numpy", "", False)
        )

    def __array_ufunc__(self, ufunc: Any, method: str, *inputs: Any, **kwargs: Any):
        """Refuse: a numpy ufunc of an array is an operation on all its cells."""
        name = getattr(ufunc, "__name__", "a ufunc")
        raise TraceError(
            _whole_array_message(self, f"numpy.{name} of {self.name}", "", False)
        )

    def __array_function__(self, func: Any, types: Any, args: Any, kwargs: Any):
        """Refuse: a numpy function of an array is an operation on all its cells."""
        name = getattr(func, "__name__", "a function")
        raise TraceError(
            _whole_array_message(self, f"numpy.{name} of {self.name}", "", False)
        )

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


#: The index names a message spells a loop nest with, one per axis.
_INDEX_NAMES = ("i", "j", "k", "l")


def _whole_key(key: Any) -> bool:
    """Whether a subscript key names more than one cell per index."""
    parts = key if isinstance(key, tuple) else (key,)
    many = slice | list | np.ndarray | SymArr | SymDom
    return any(part is Ellipsis or isinstance(part, many) for part in parts)


def _key_text(key: Any) -> str:
    """A subscript key the way the body spells it: ``:``, ``1:``, ``i, ...``."""

    def text(part: Any) -> str:
        if part is Ellipsis:
            return "..."
        if isinstance(part, slice):
            ends = [
                "" if end is None else _shown(end) for end in (part.start, part.stop)
            ]
            step = "" if part.step is None else f":{_shown(part.step)}"
            return f"{ends[0]}:{ends[1]}{step}"
        if isinstance(part, np.ndarray):
            return "<array>"
        return _shown(part)

    parts = key if isinstance(key, tuple) else (key,)
    return ", ".join(text(part) for part in parts)


def _whole_array_message(
    array: SymArr, spelled: str, where: str, write: bool, why: str = ""
) -> str:
    """What to say about an operation on a whole array, with the loop nest."""
    indices: list[str] = []
    loops: list[str] = []
    for axis in range(array.ndim):
        index = _INDEX_NAMES[axis] if axis < len(_INDEX_NAMES) else f"i{axis}"
        domain = f"{array.name}.dom" + "".join(f"[{name}]" for name in indices)
        loops.append(f"for {index} in {domain}:")
        indices.append(index)
    cell = f"{array.name}[{', '.join(indices)}]"
    use = f"{cell} = ..." if write else f"... {cell} ..."
    at = f" at {where}" if where else ""
    because = f", and {why}" if why else ""
    return (
        f"{spelled}{at} is an operation on the whole array {array.name}{because}. "
        "A traced kernel records one statement per family of cells, each cell "
        "named by its indices, so an operation on many cells at once has no "
        "statement to be. Write the loop nest and index the array: "
        f"{' '.join(loops)} {use}"
    )


def _refused_operator(symbol: str, side: str) -> Any:
    """A dunder of :class:`SymArr` that refuses the operator ``symbol``."""

    def operation(self: SymArr, *_other: Any) -> Any:
        spelled = {
            "left": f"{self.name} {symbol} ...",
            "right": f"... {symbol} {self.name}",
            "unary": f"{symbol}{self.name}",
            "call": f"{symbol}({self.name})",
        }[side]
        raise TraceError(
            _whole_array_message(
                self, spelled, _location(sys._getframe(1)), write=False
            )
        )

    return operation


for _name, _symbol in (
    ("add", "+"), ("sub", "-"), ("mul", "*"), ("truediv", "/"),
    ("floordiv", "//"), ("mod", "%"), ("pow", "**"), ("matmul", "@"),
    ("and", "&"), ("or", "|"), ("xor", "^"), ("lshift", "<<"), ("rshift", ">>"),
):  # fmt: skip
    setattr(SymArr, f"__{_name}__", _refused_operator(_symbol, "left"))
    setattr(SymArr, f"__r{_name}__", _refused_operator(_symbol, "right"))
for _name, _symbol in (("lt", "<"), ("le", "<="), ("gt", ">"), ("ge", ">=")):
    setattr(SymArr, f"__{_name}__", _refused_operator(_symbol, "left"))
for _name, _symbol in (("neg", "-"), ("pos", "+"), ("invert", "~")):
    setattr(SymArr, f"__{_name}__", _refused_operator(_symbol, "unary"))
SymArr.__abs__ = _refused_operator("abs", "call")  # type: ignore[method-assign]
del _name, _symbol


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
        binders = [var.name for var, _domain in expr.binders]
        bound = {*(name for name, _bound in enclosing), *binders}
        unstated = _unstated(expr.guard, tracer, bound)
        if unstated:
            raise TraceError(_reduction_condition_message(unstated, binders, where))
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
                *constraints_of(tracer.guard(), tracer),
                *_binder_constraints(enclosing, tracer),
                *(
                    piece
                    for guard in outer_guards
                    for piece in constraints_of(guard, tracer, bound)
                ),
                *constraints_of(expr.guard, tracer, bound),
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


def _reduction_condition_message(
    parts: Sequence[tuple[Any, str]], binders: Sequence[str], where: str
) -> str:
    """What to say about a reduction condition its domain cannot state.

    ``parts`` are the conjuncts its domain cannot state, each with the reason
    :func:`constraints_of` gives for dropping it.
    """
    listed = " and ".join(repr(_shown(part)) for part, _why in parts)
    at = f" at {where}" if where else ""
    why = "; ".join(dict.fromkeys(why for _part, why in parts))
    unequal = any(why == _UNEQUAL for _part, why in parts)
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
        "states an affine comparison of loop variables, sizes and integral "
        f"scalars, and this condition {why}.{split} Otherwise write each term "
        "to an indexed cell, 0.0 where the condition is false and the term "
        "under 'with when(condition):', and sum the cells."
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
        masked = _MaskedArr(value.numpy(), offsets)
        # A view already reading through a kernel's declared layout keeps it.
        layout = value.layout
        return masked if layout is None else masked.through(*layout)
    if isinstance(value, np.ndarray):
        return value.view(_MaskedArray)
    return value


def _integer_guard_message(value: Any, where: str) -> str:
    """What to say about a guard whose value is an integer and not a bool."""
    at = f" at {where}" if where else ""
    return (
        f"the guard of 'with when(...)'{at} is the integer {int(value)}, not a "
        "truth value. Python's '~' is bitwise on an int and on a bool: ~True is "
        "-2 and ~False is -1, and both are true, so a guard such as "
        "'~(i > 0)' on a loop variable holds at every point when the body runs "
        "natively, while the traced term reads it as 'not' and the compiled "
        "kernel skips the points where i > 0. '&' and '|' with an integer "
        "operand are bitwise in the same way. Write the complement as a "
        "comparison ('i <= 0' for '~(i > 0)', and "
        "'(i <= 0) | (i >= n)' for '~((i > 0) & (i < n))'), and compare an "
        "integer explicitly ('k != 0') rather than guarding on it."
    )


def _integer(value: Any) -> bool:
    """Whether ``value`` is a Python or numpy integer that is not a bool."""
    return isinstance(value, int | np.integer) and not isinstance(
        value, bool | np.bool_
    )


class when:  # noqa: N801 - a context manager written like a statement
    """Guard the writes of a block by ``condition``.

    Under tracing the condition is pushed onto the guard stack: the statements
    recorded inside carry it, and it narrows their domain where isl can state
    it (an affine comparison of loop variables, sizes and integral scalars; see
    :func:`constraints_of`), so a guarded access is proved in bounds exactly
    where it runs. Under plain ``python`` the block still executes and the
    writes are masked, which is why the guard has to be a condition on data and
    not a Python ``if``: masking keeps the traced term and the native run
    agreeing statement for statement.

    The condition has to be a truth value, and an integer that is not a bool
    is refused, under tracing and natively, with a :class:`TraceError`. The
    native value of ``~(i > 0)`` is where one comes from: ``i`` is a Python
    ``int``, ``i > 0`` a Python ``bool``, and ``~`` on a bool is bitwise, so
    the guard is ``-2`` or ``-1`` and always true, while the trace records
    ``not (i > 0)``. A data comparison is a numpy ``bool_``, on which ``~`` is
    logical, and is not affected. Natively the guard is asked wherever the
    guards around it hold; under a false one nothing is written whatever it
    says.
    """

    def __init__(self, condition: Any) -> None:
        self.condition = condition
        self.tracer = current_tracer()

    def __enter__(self) -> when:
        """Open the guard."""
        # Natively, a guard inside a block whose guard is false is not asked:
        # nothing under it is written, and a read out of range there answers
        # the integer 0 (see _MaskedArr), whatever the array holds.
        asked = self.tracer is not None or not _writes_are_masked()
        if asked and _integer(self.condition):
            raise TraceError(
                _integer_guard_message(self.condition, _location(sys._getframe(1)))
            )
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

    The body's effects outside its array parameters are refused once it has
    run: a call :class:`_CallWatch` recognizes, and a change to the state
    :func:`_outside` copied before it started.
    """
    function = getattr(kernel, "fn", kernel)
    name = getattr(kernel, "__name__", getattr(function, "__name__", "kernel"))
    types = dict(arg_types)
    post_annotation = types.pop("return", None)

    tracer = Tracer(name, types)
    tracer.own = _own_of(function)
    outside = _outside(function, tracer._owned(), tracer.own)
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
    tracer.integers |= _integer_names(params)

    _TRACERS.append(tracer)
    watching = _CallWatch.start()
    try:
        function(*arguments)
    except SymbolicBoolError as exc:
        raise TraceError(
            f"{exc}\nA kernel body may not branch on a value it computes: "
            "write 'with when(condition):' instead of 'if condition:', which "
            "masks the writes of the block rather than choosing a branch."
        ) from exc
    finally:
        if watching:
            _CallWatch.stop()
        _TRACERS.pop()

    if tracer.loops:
        # Every loop level is closed by its own iterator raising StopIteration.
        # A level still open once the body has returned means that iterator was
        # abandoned, which only a 'break' or a 'return' inside the loop does.
        raise TraceError(_abandoned_message([loop.iname for loop in tracer.loops]))
    if tracer.effects:
        raise TraceError(_effects_message(name, tracer.effects))
    changes = _outside_changes(outside, tracer)
    if changes:
        raise TraceError(_outside_message(name, changes))

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
