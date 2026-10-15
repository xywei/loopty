"""Handles: what a schedule names a loop, an array or a call by.

A schedule step acts on loops and arrays, and a loop's name is a poor way to
say which. The binders of two kernels collide once a program composes them
(``unpack``'s ``i`` and ``reconstruct``'s ``i`` are ``i`` and ``i_0`` in the
program's term), a fusion makes two loops one, a split makes one loop two,
and the loops a program zeroes its arrays in are named by loopty and not by
the source. So a schedule names what it acts on by where it is in the source,
a path from the kernel or the program it schedules::

    l2p.reconstruct.i       # the loop i of the call of reconstruct in l2p
    l2p.R                   # the array R that l2p makes
    reconstruct.b           # the loop b of the kernel reconstruct
    mass_action.mass_ffc.j  # the loop j of the kernel mass_action calls

That path is a *handle*. It is made from the traced kernel or program, it
carries the path and the source line of what it names (:attr:`Handle.path`,
:attr:`Handle.where`), and it is resolved against a schedule only when a step
is taken, through what each step before it did to the loops.

What a path can name
--------------------

A kernel's names are its loops, the binders of its ``for`` loops and of its
sums (:class:`loopty.term.Binder`), its array parameters, its sizes, and the
kernels its body calls, each inlined where it is called
(:mod:`loopty.calls`). A program's are its calls, its arrays, the parameters
and the arrays it makes with ``Arr.zeros_like`` or ``Arr.zeros``, and its
sizes; it has no loops of its own. A call is a scope too: it has the callee's
loops and arrays, each scoped to that call, and the calls the callee makes.
``l2p.reconstruct.R`` is the array the program passed to ``reconstruct`` for
its parameter ``R``, which is ``l2p.R``, and ``l2p.reconstruct.i`` is the
loop ``i`` of that call, which no other call has.

Every name is reached two ways. The short form is an attribute of the
kernel, the program or the call, ``reconstruct.i``. The long form says what
kind of name it is, and never meets an attribute of the API:
``reconstruct.loops.i``, ``reconstruct.arrays.R``, ``l2p.calls.unpack``,
``l2p.arrays.R`` and ``l2p.sizes.nb``. A name that is also an attribute of
:class:`~loopty.kernel.Kernel` or :class:`~loopty.kernel.Program` (a loop
called ``term``, say) keeps the attribute: its short form is refused when the
kernel is defined, with a :class:`ShadowedName` warning that names the long
form, and a short form that names two kinds of thing, a loop and an array
of one name, is refused where it is asked for, naming both long forms.

One name can name several things of one kind. A kernel called twice in a
program is two calls, ``l2p.unpack[0]`` and ``l2p.unpack[1]`` in call
order, and ``l2p.unpack`` alone is refused with a message that lists both;
a second ``for i`` in one body is ``k.i[1]``. Indexing a name that names one
thing with ``[0]`` is the thing itself. Handles complete in an interactive
session (``dir`` lists the names), and a name that is not there is refused
with the scope and what it has: ``reconstruct has no loop, array, call or
size named t; its loops are i, a, b, its arrays C, R and its sizes nb``.

Resolution
----------

A schedule keeps, for every loop of the source, the loops of its kernel
that loop is now (:meth:`loopty.schedule.Schedule._loop_names`): at first
the loop the lowering made of it, and after each step what the step made of
that. A fusion makes the consumer's loops the producer's, so after
``fuse(l2p.unpack.i, l2p.reconstruct.i, l2p.evaluate.i)`` the three handles
are one loop. A split replaces a loop by two, which are the handle's
children, ``h.outer`` and ``h.inner`` (``tile`` makes the same children of
both its loops), and a split of a child makes grandchildren,
``h.inner.outer``. A map of :meth:`~loopty.schedule.Schedule.affine` keeps
the loops it does not rename and replaces the others by the loops it names,
which the long form reaches, ``acoustic.loops.a`` (or the schedule's own,
``diamond.loops.a``). A handle
to a loop no step left, one a split replaced or one only statements a
substitution took out ran in, is refused with the step that did it.

Arrays are not moved by steps, so an array handle resolves to the array of
the term it names; a call's array resolves to the array the call was passed.

What is printed
---------------

Everything a schedule prints names loops and arrays by these paths: its
steps (:attr:`~loopty.schedule.Schedule.history` and its repr), the
statements of its facts, its refusals and the fixes they name, and ``loopty
run``. A loop is printed by the first handle that reaches it, the producer's
after a fusion, and a loop the program made to zero an array it makes,
which the source never wrote, by the array: ``l2p.R.zeros``. No name loopy
made (``i_2``) is printed. The steps' arguments are printed as the handles
were written, so ``contract(l2p.R)`` and ``contract(l2p.reconstruct.R)``
read as they were written, and keep the facts of the two apart, as two
spellings of a step always have.

A size is a handle too, with arithmetic: ``mass_action.nc % 4 == 0`` is a
:class:`Condition` of the sizes, which is what
:meth:`~loopty.schedule.Schedule.assume` takes. ``==`` on a size builds a
condition rather than comparing, as lanky's terms do, so a size is hashed by
identity, and a condition refuses to be a ``bool``: Python reads ``1 <= n <=
4`` as ``(1 <= n) and (n <= 4)``, which would drop half of it; write ``(1 <=
n) & (n <= 4)``.
"""

from __future__ import annotations

import ast
import inspect
import textwrap
import warnings
from collections.abc import Iterable, Iterator, Sequence
from typing import Any

import pymbolic.primitives as prim

from loopty.term import ArrType, Binder, Callsite, Term

__all__ = [
    "Array",
    "Call",
    "Condition",
    "Handle",
    "Loop",
    "Names",
    "ShadowedName",
    "Size",
    "SizeExpression",
    "binders_of",
    "callsites_of",
    "scope",
]


class ShadowedName(UserWarning):
    """A loop, an array or a size named like an attribute of the API.

    Issued when the kernel or the program is defined: its short form,
    ``k.term``, is the attribute, and the message names the long form that
    reaches the loop, ``k.loops.term``.
    """


# {{{ what a term holds


def binders_of(term: Term) -> tuple[Binder, ...]:
    """The loops of ``term``'s source: its records, or its statements' loops.

    A traced term records its binders (:attr:`loopty.term.Term.binders`).
    A term written by hand records none, and its loops are then the loops
    of its statements and of their sums, each named as the term names it,
    at the line of the first statement that runs in it.
    """
    if term.binders:
        return term.binders
    from loopty.trace import reductions_in

    out: dict[str, Binder] = {}
    for stmt in term.stmts:
        for name in stmt.inames:
            out.setdefault(name, Binder(name, name, stmt.where))
        for reduction in reductions_in(stmt.expr):
            for name in reduction.inames:
                out.setdefault(name, Binder(name, name, stmt.where, sum=True))
    return tuple(out.values())


def callsites_of(term: Term) -> tuple[Callsite, ...]:
    """The calls ``term`` holds: its records, or its program scopes and calls.

    A term written by hand that has a program's scopes or a kernel's calls
    has its calls read off them, with no arrays.
    """
    if term.callsites:
        return term.callsites
    return (
        *(Callsite((scope.call,), scope.kernel, scope.where) for scope in term.scopes),
        *(
            Callsite((*call.within, call.label), call.callee, call.where)
            for call in term.calls
        ),
    )


def _count(label: str) -> int:
    """Which call of its kernel a label is: ``dot`` the first, ``dot@3`` the third."""
    _name, _at, count = label.partition("@")
    return int(count) if count.isdigit() else 1


def _arrays_of(term: Term) -> dict[str, str]:
    """The arrays of a term a schedule names, by name: parameters and its own.

    The flags of a program's checked points are no array of the source, and
    a table is read and never stored, so neither is among them.
    """
    flags = {flag for flag, _message in term.checks}
    out = {name: name for name, typ in term.params if isinstance(typ, ArrType)}
    out.update((name, name) for name, _typ in term.temporaries if name not in flags)
    return out


class _Index:
    """What a term holds, scope by scope, and how each thing is printed.

    ``root`` is the kernel, the program or the term the paths start from,
    and ``name`` the first segment of every path.
    """

    def __init__(self, term: Term, root: Any) -> None:
        self.term = term
        self.root = root
        self.name = getattr(root, "__name__", None) or term.name
        self.binders = binders_of(term)
        self.sites = {site.labels: site for site in callsites_of(term)}
        self.loops: dict[tuple[str, ...], dict[str, list[Binder]]] = {}
        for binder in self.binders:
            self.loops.setdefault(binder.scope, {}).setdefault(
                binder.binder, []
            ).append(binder)
        self.calls: dict[tuple[str, ...], dict[str, list[tuple[str, ...]]]] = {}
        for labels, site in self.sites.items():
            self.calls.setdefault(labels[:-1], {}).setdefault(site.kernel, []).append(
                labels
            )
        for kernels in self.calls.values():
            for labels in kernels.values():
                labels.sort(key=lambda path: _count(path[-1]))
        self.root_arrays = _arrays_of(term)

    # {{{ names in a scope

    def arrays(self, labels: tuple[str, ...]) -> dict[str, str]:
        """The arrays of a scope, each with the term's array it is."""
        if not labels:
            return dict(self.root_arrays)
        site = self.sites.get(labels)
        return {} if site is None else dict(site.arrays)

    def sizes(self, labels: tuple[str, ...]) -> tuple[str, ...]:
        """The sizes of a scope: the term's, at its root, and none in a call."""
        return tuple(self.term.sizes) if not labels else ()

    def kinds(self, labels: tuple[str, ...], name: str) -> list[str]:
        """What ``name`` names in a scope: ``loop``, ``array``, ``call``, ``size``."""
        out = []
        if name in self.loops.get(labels, {}):
            out.append("loop")
        if name in self.arrays(labels):
            out.append("array")
        if name in self.calls.get(labels, {}):
            out.append("call")
        if name in self.sizes(labels):
            out.append("size")
        return out

    def names(self, labels: tuple[str, ...]) -> list[str]:
        """Every name of a scope, for ``dir``."""
        return list(
            dict.fromkeys(
                [
                    *self.loops.get(labels, {}),
                    *self.arrays(labels),
                    *self.calls.get(labels, {}),
                    *self.sizes(labels),
                ]
            )
        )

    def contents(self, labels: tuple[str, ...]) -> str:
        """What a scope has, in words: ``its loops are i, a, b and its arrays C, R``."""
        parts = []
        for word, names in (
            ("loops", list(self.loops.get(labels, {}))),
            ("arrays", list(self.arrays(labels))),
            ("calls", list(self.calls.get(labels, {}))),
            ("sizes", list(self.sizes(labels))),
        ):
            if names:
                parts.append(f"{word} {', '.join(names)}")
        if not parts:
            return "it has no loops, arrays or calls"
        first, *rest = parts
        first = first.replace(" ", " are ", 1)
        if not rest:
            return f"its {first}"
        return f"its {first}, " + ", ".join(rest[:-1]) + (
            f" and its {rest[-1]}" if len(rest) == 1 else f" and {rest[-1]}"
        )

    # }}}

    # {{{ paths

    def segments(self, labels: Sequence[str]) -> list[str]:
        """The printed segments of a scope: ``unpack``, or ``unpack[1]``."""
        out = []
        for depth in range(1, len(labels) + 1):
            path = tuple(labels[:depth])
            site = self.sites.get(path)
            if site is None:
                out.append(path[-1])
                continue
            siblings = self.calls.get(path[:-1], {}).get(site.kernel, [path])
            if len(siblings) > 1:
                out.append(f"{site.kernel}[{siblings.index(path)}]")
            else:
                out.append(site.kernel)
        return out

    def scope_path(self, labels: Sequence[str]) -> str:
        """``l2p.unpack``: the path of a scope."""
        return ".".join([self.name, *self.segments(labels)])

    def binder_path(self, binder: Binder) -> str:
        """``l2p.reconstruct.i``, or ``k.i[1]`` for a second loop of one name."""
        same = self.loops.get(binder.scope, {}).get(binder.binder, [binder])
        name = binder.binder
        if len(same) > 1:
            name = f"{name}[{same.index(binder)}]"
        if self.kinds(binder.scope, binder.binder) != ["loop"] or _shadowed(
            self, binder.scope, binder.binder
        ):
            name = f"loops.{name}"
        return f"{self.scope_path(binder.scope)}.{name}"

    def array_path(self, labels: tuple[str, ...], name: str) -> str:
        """``l2p.R``, or ``l2p.arrays.R`` where the short form is not the array."""
        if self.kinds(labels, name) != ["array"] or _shadowed(self, labels, name):
            name = f"arrays.{name}"
        return f"{self.scope_path(labels)}.{name}"

    def size_path(self, name: str) -> str:
        """``mass_action.nc``, or ``mass_action.sizes.nc``."""
        if self.kinds((), name) != ["size"] or _shadowed(self, (), name):
            name = f"sizes.{name}"
        return f"{self.name}.{name}"

    # }}}


def _shadowed(index: _Index, labels: tuple[str, ...], name: str) -> bool:
    """Whether the short form of ``name`` in a scope is an attribute instead."""
    if labels:
        return name in _CALL_ATTRIBUTES
    return name in _api_names(index.root)


def _api_names(root: Any) -> frozenset[str]:
    """The attributes of a kernel's or a program's API, which short forms yield to."""
    from loopty.kernel import Kernel, Program

    if isinstance(root, Kernel | Program):
        return frozenset(
            name for name in {*dir(type(root)), *vars(root)} if not name.startswith("_")
        )
    return frozenset()


def _index(root: Any) -> _Index:
    """The index of a kernel's, a program's or a term's names, kept with it."""
    term = root if isinstance(root, Term) else root.term
    if isinstance(root, Term):
        return _Index(term, root)
    cached = root.__dict__.get("_handles_index")
    if cached is not None and cached.term is term:
        return cached
    index = _Index(term, root)
    root.__dict__["_handles_index"] = index
    return index


# }}}


# {{{ handles


class Handle:
    """A path from a kernel, a program or a term to something in it.

    ``path`` is the path as it is printed, ``l2p.reconstruct.i``, and
    ``where`` the line of the source that wrote what it names: the ``for``
    of a loop, the call of a call, the definition of an array or a size.
    A handle is resolved against a schedule of its root when a step is
    taken (see the module docstring).
    """


    def __init__(self, index: _Index) -> None:
        self._index = index

    @property
    def root(self) -> Any:
        """The kernel, the program or the term the path starts from."""
        return self._index.root

    @property
    def path(self) -> str:
        raise NotImplementedError

    @property
    def where(self) -> str:
        raise NotImplementedError

    def _key(self) -> tuple:
        raise NotImplementedError

    def __eq__(self, other: object) -> bool:
        if type(other) is not type(self):
            return NotImplemented
        return self._index.root is other._index.root and self._key() == other._key()

    def __hash__(self) -> int:
        return hash((type(self).__name__, id(self._index.root), self._key()))

    def __repr__(self) -> str:
        return self.path

    __str__ = __repr__


class Loop(Handle):
    """A loop of the source, or a loop a step made of one.

    ``binder`` is the loop's record in the term (:class:`loopty.term.Binder`),
    and ``part`` the steps' children from there, outermost first:
    ``("outer",)`` for ``h.outer`` after a split of ``h``. A loop a map of
    :meth:`~loopty.schedule.Schedule.affine` named has no binder, and is
    ``made`` there instead.
    """


    def __init__(
        self,
        index: _Index,
        binder: Binder | None,
        part: tuple[str, ...] = (),
        made: str | None = None,
    ) -> None:
        super().__init__(index)
        self._binder = binder
        self._part = part
        self._made = made

    def _key(self) -> tuple:
        return (self._binder, self._made, self._part)

    @property
    def binder(self) -> Binder | None:
        """The loop of the source the handle starts from, or ``None``."""
        return self._binder

    @property
    def part(self) -> tuple[str, ...]:
        """The children of splits from :attr:`binder`, outermost first."""
        return self._part

    @property
    def made(self) -> str | None:
        """The name an affine map gave a loop it made, or ``None``."""
        return self._made

    @property
    def outer(self) -> Loop:
        """The outer of the two loops a split of this one makes."""
        return Loop(self._index, self._binder, (*self._part, "outer"), self._made)

    @property
    def inner(self) -> Loop:
        """The inner of the two loops a split of this one makes."""
        return Loop(self._index, self._binder, (*self._part, "inner"), self._made)

    @property
    def path(self) -> str:
        if self._binder is not None:
            head = self._index.binder_path(self._binder)
        else:
            head = f"{self._index.name}.loops.{self._made}"
        return ".".join([head, *self._part])

    @property
    def where(self) -> str:
        return "" if self._binder is None else self._binder.where


class Array(Handle):
    """An array: a parameter, an array a program makes, or a call's argument.

    ``name`` is the term's array the handle is, which a call's array is
    through what the call was passed: ``l2p.reconstruct.R`` is ``R``.
    """


    def __init__(
        self, index: _Index, labels: tuple[str, ...], param: str, name: str
    ) -> None:
        super().__init__(index)
        self._labels = labels
        self._param = param
        self._name = name

    def _key(self) -> tuple:
        return (self._labels, self._param)

    @property
    def name(self) -> str:
        """The term's array this is."""
        return self._name

    @property
    def path(self) -> str:
        return self._index.array_path(self._labels, self._param)

    @property
    def where(self) -> str:
        if self._labels:
            site = self._index.sites.get(self._labels)
            return "" if site is None else site.where
        return getattr(self._index.root, "where", "") or self._index.term.where


class _Scoped(Handle):
    """A scope a path goes through: a kernel's, a program's, or a call's.

    Its short names are attributes, ``scope.i``; its long forms are
    :attr:`loops`, :attr:`arrays`, :attr:`calls` and :attr:`sizes`.
    """


    def __init__(self, index: _Index, labels: tuple[str, ...]) -> None:
        super().__init__(index)
        self._labels = labels

    def _key(self) -> tuple:
        return (self._labels,)

    @property
    def path(self) -> str:
        return self._index.scope_path(self._labels)

    @property
    def loops(self) -> Names:
        """The loops of this scope by name: ``scope.loops.i``."""
        return Names(self._index, self._labels, "loop")

    @property
    def arrays(self) -> Names:
        """The arrays of this scope by name: ``scope.arrays.R``."""
        return Names(self._index, self._labels, "array")

    @property
    def calls(self) -> Names:
        """The calls made in this scope, by the callee's name: ``scope.calls.dot``."""
        return Names(self._index, self._labels, "call")

    @property
    def sizes(self) -> Names:
        """The sizes of this scope by name: ``scope.sizes.n``."""
        return Names(self._index, self._labels, "size")

    def __getattr__(self, name: str) -> Any:
        if name.startswith("_"):
            raise AttributeError(name)
        return _short(self._index, self._labels, name)

    def __dir__(self) -> list[str]:
        return sorted({*object.__dir__(self), *self._index.names(self._labels)})


class Root(_Scoped):
    """The scope of a term written by hand, or of a kernel seen as a scope.

    What :func:`scope` gives: a kernel and a program have their names as
    attributes of their own, and a term, whose attributes are its fields,
    has them here.
    """


    @property
    def where(self) -> str:
        return getattr(self._index.root, "where", "") or self._index.term.where


class Call(_Scoped):
    """A call: a program's call of a kernel, or a kernel's inlined call.

    Its loops and arrays are the callee's, scoped to the call:
    ``l2p.reconstruct.i`` is the loop ``i`` of this call of ``reconstruct``.
    """


    @property
    def where(self) -> str:
        site = self._index.sites.get(self._labels)
        return "" if site is None else site.where

    @property
    def kernel(self) -> str:
        """The name of the kernel called."""
        return self._index.sites[self._labels].kernel


#: What a call's short form yields to: its own attributes.
_CALL_ATTRIBUTES = frozenset(
    {"loops", "arrays", "calls", "sizes", "path", "where", "kernel", "root"}
)


class _Several:
    """One name that names several things of a kind, until it is indexed.

    ``l2p.unpack`` where the program calls ``unpack`` twice, or ``k.i``
    where the body has two loops ``i``: ``[0]`` and ``[1]`` are each one,
    in the order of the source, and anything else is refused with both.
    """


    def __init__(self, index: _Index, path: str, kind: str, items: Sequence[Any]):
        self._index = index
        self._path = path
        self._kind = kind
        self._items = tuple(items)

    def __getitem__(self, position: int) -> Any:
        if isinstance(position, bool) or not isinstance(position, int):
            raise TypeError(f"{self._path}[...] takes a whole number, not {position!r}")
        if not -len(self._items) <= position < len(self._items):
            raise IndexError(
                f"{self._path}[{position}]: {self._path} names "
                f"{len(self._items)} {self._kind}s, {self._path}[0] to "
                f"{self._path}[{len(self._items) - 1}]"
            )
        return self._items[position]

    def __len__(self) -> int:
        return len(self._items)

    def __iter__(self) -> Iterator[Any]:
        return iter(self._items)

    @property
    def message(self) -> str:
        """Why the name alone is refused, naming each of the things."""
        listed = [
            f"{self._path}[{position}] at {item.where}"
            if item.where
            else f"{self._path}[{position}]"
            for position, item in enumerate(self._items)
        ]
        return (
            f"{self._path} names {len(self._items)} {self._kind}s, "
            f"{_joined(listed)}; name one"
        )

    def __getattr__(self, name: str) -> Any:
        if name.startswith("_"):
            raise AttributeError(name)
        raise AttributeError(self.message)

    def __repr__(self) -> str:
        return self._path

    def __dir__(self) -> list[str]:
        return []


class Names:
    """The names of one kind in a scope: the long forms, ``k.loops.i``."""


    def __init__(self, index: _Index, labels: tuple[str, ...], kind: str) -> None:
        self._index = index
        self._labels = labels
        self._kind = kind

    def __getattr__(self, name: str) -> Any:
        if name.startswith("_"):
            raise AttributeError(name)
        return _named(self._index, self._labels, name, self._kind)

    def __dir__(self) -> list[str]:
        return list(self._names())

    def __iter__(self) -> Iterator[str]:
        return iter(self._names())

    def _names(self) -> list[str]:
        index, labels = self._index, self._labels
        if self._kind == "loop":
            return list(index.loops.get(labels, {}))
        if self._kind == "array":
            return list(index.arrays(labels))
        if self._kind == "call":
            return list(index.calls.get(labels, {}))
        return list(index.sizes(labels))

    def __repr__(self) -> str:
        word = {"loop": "loops", "array": "arrays", "call": "calls", "size": "sizes"}
        return f"{self._index.scope_path(self._labels)}.{word[self._kind]}"


def _named(index: _Index, labels: tuple[str, ...], name: str, kind: str) -> Any:
    """The ``kind`` named ``name`` in a scope, or an ``AttributeError`` saying why."""
    path = f"{index.scope_path(labels)}.{name}"
    if kind == "loop":
        found = index.loops.get(labels, {}).get(name)
        if found:
            handles = [Loop(index, binder) for binder in found]
            if len(handles) == 1:
                return handles[0]
            return _Several(index, path, "loop", handles)
        if not labels and name.isidentifier():
            # no loop of the source: a loop a map of affine names, which a
            # schedule resolves where one of its steps made it
            return Loop(index, None, (), made=name)
    elif kind == "array":
        arrays = index.arrays(labels)
        if name in arrays:
            return Array(index, labels, name, arrays[name])
    elif kind == "call":
        found = index.calls.get(labels, {}).get(name)
        if found:
            calls = [Call(index, site) for site in found]
            if len(calls) == 1:
                return calls[0]
            return _Several(index, path, "call", calls)
    elif name in index.sizes(labels):
        return Size(index, name)
    raise AttributeError(_missing(index, labels, name, kind))


def _short(index: _Index, labels: tuple[str, ...], name: str) -> Any:
    """What the short form ``name`` names in a scope, refusing two kinds of it."""
    kinds = index.kinds(labels, name)
    if not kinds:
        raise AttributeError(_missing(index, labels, name, None))
    if len(kinds) > 1:
        word = {"loop": "loops", "array": "arrays", "call": "calls", "size": "sizes"}
        path = index.scope_path(labels)
        forms = [f"{path}.{word[kind]}.{name}" for kind in kinds]
        what = " and ".join(
            f"{'an' if kind[0] in 'aeiou' else 'a'} {kind}" for kind in kinds
        )
        raise AttributeError(
            f"{path}.{name} names {what}; write {_joined(forms, 'or')}"
        )
    return _named(index, labels, name, kinds[0])


def _missing(
    index: _Index, labels: tuple[str, ...], name: str, kind: str | None
) -> str:
    """``reconstruct has no loop named t; its loops are i, a, b and ...``."""
    what = kind or "loop, array, call or size"
    path = index.scope_path(labels)
    message = f"{path} has no {what} named {name}; {index.contents(labels)}"
    if not labels and kind in (None, "loop") and not index.loops.get((), {}):
        # a program's loops are its calls'
        scoped = [
            index.binder_path(binder)
            for binder in index.binders
            if binder.binder == name
        ]
        if scoped:
            message += f". Its calls' loops of that name are {_joined(scoped)}"
    return message


def scope(obj: Any) -> Root:
    """The names of a kernel, a program or a term, as a scope of handles.

    A kernel and a program have them as attributes of their own,
    ``reconstruct.i`` and ``l2p.R``; a term, whose attributes are its
    fields, has them here: ``scope(term).i``.
    """
    from loopty.schedule import Schedule

    if isinstance(obj, Schedule):
        obj = obj.source
    if not isinstance(obj, Term) and not hasattr(obj, "term"):
        raise TypeError(f"{obj!r} is not a kernel, a program or a term")
    return Root(_index(obj), ())


def short_name(root: Any, name: str) -> Any:
    """``root.name`` for a kernel or a program: what :meth:`Kernel.__getattr__` gives.

    Only a name that is no attribute comes here. The kernel is traced, once,
    to know what it names; a body that cannot be traced has no names, and
    the ``AttributeError`` says why.
    """
    try:
        index = _index(root)
    except Exception as exc:  # noqa: BLE001 - a body that cannot be traced
        raise AttributeError(
            f"{getattr(root, '__name__', root)}.{name}: its body cannot be traced, "
            f"so what it names is not known: {type(exc).__name__}: {exc}"
        ) from exc
    return _short(index, (), name)


# }}}


# {{{ sizes and conditions


class SizeExpression:
    """An expression of sizes: ``2 * n + 1``, written with size handles.

    Arithmetic builds a larger one, a comparison a :class:`Condition`.
    ``expression`` is it in the term's names, which loopty reads as it reads
    an assumption written out (:meth:`loopty.schedule.Schedule.assume`).
    """


    def __init__(self, index: _Index, expression: Any) -> None:
        self._index = index
        self._expression = expression

    @property
    def expression(self) -> Any:
        """The expression, a pymbolic one over the term's sizes."""
        return self._expression

    def _other(self, other: Any) -> Any:
        if isinstance(other, SizeExpression):
            if other._index.root is not self._index.root:
                raise ValueError(
                    f"{self!r} and {other!r} are sizes of {self._index.name} and "
                    f"{other._index.name}; a condition is of one kernel's sizes"
                )
            return other._expression
        if isinstance(other, int) and not isinstance(other, bool):
            return other
        return NotImplemented

    def _binary(self, other: Any, build: Any) -> Any:
        value = self._other(other)
        if value is NotImplemented:
            return NotImplemented
        return SizeExpression(self._index, build(self._expression, value))

    def __add__(self, other: Any) -> Any:
        return self._binary(other, lambda a, b: prim.Sum((a, b)))

    def __radd__(self, other: Any) -> Any:
        return self._binary(other, lambda a, b: prim.Sum((b, a)))

    def __sub__(self, other: Any) -> Any:
        return self._binary(other, lambda a, b: prim.Sum((a, prim.Product((-1, b)))))

    def __rsub__(self, other: Any) -> Any:
        return self._binary(other, lambda a, b: prim.Sum((b, prim.Product((-1, a)))))

    def __mul__(self, other: Any) -> Any:
        return self._binary(other, lambda a, b: prim.Product((a, b)))

    def __rmul__(self, other: Any) -> Any:
        return self._binary(other, lambda a, b: prim.Product((b, a)))

    def __floordiv__(self, other: Any) -> Any:
        return self._binary(other, prim.FloorDiv)

    def __mod__(self, other: Any) -> Any:
        return self._binary(other, prim.Remainder)

    def __neg__(self) -> SizeExpression:
        return SizeExpression(self._index, prim.Product((-1, self._expression)))

    def _compare(self, other: Any, operator: str) -> Any:
        value = self._other(other)
        if value is NotImplemented:
            return NotImplemented
        return Condition(
            self._index, (prim.Comparison(self._expression, operator, value),)
        )

    def __eq__(self, other: Any) -> Any:  # type: ignore[override]
        return self._compare(other, "==")

    def __ne__(self, other: Any) -> Any:  # type: ignore[override]
        return self._compare(other, "!=")

    def __lt__(self, other: Any) -> Any:
        return self._compare(other, "<")

    def __le__(self, other: Any) -> Any:
        return self._compare(other, "<=")

    def __gt__(self, other: Any) -> Any:
        return self._compare(other, ">")

    def __ge__(self, other: Any) -> Any:
        return self._compare(other, ">=")

    __hash__ = object.__hash__

    def __bool__(self) -> bool:
        raise TypeError(
            f"{self!r} is an expression of sizes, which has no truth value until "
            "a run gives the sizes; compare it, as in assume(n % 4 == 0)"
        )

    def __repr__(self) -> str:
        return _shown(self._expression, self._index)

    __str__ = __repr__


class Size(SizeExpression, Handle):
    """A size of a kernel or a program: ``mass_action.nc``."""


    def __init__(self, index: _Index, name: str) -> None:
        SizeExpression.__init__(self, index, prim.Variable(name))
        self._name = name

    def _key(self) -> tuple:
        return (self._name,)

    @property
    def name(self) -> str:
        """The term's name for the size."""
        return self._name

    @property
    def path(self) -> str:
        return self._index.size_path(self._name)

    @property
    def where(self) -> str:
        return getattr(self._index.root, "where", "") or self._index.term.where

    __hash__ = object.__hash__

    def __repr__(self) -> str:
        return self.path

    __str__ = __repr__


class Condition:
    """A condition of sizes: comparisons joined by ``&``, ``nc % 4 == 0``.

    What :meth:`loopty.schedule.Schedule.assume` takes. ``text`` is it in
    the term's names, as Python, and its repr in the handles' paths.
    """


    def __init__(self, index: _Index, parts: tuple[Any, ...]) -> None:
        self._index = index
        self._parts = parts

    @property
    def root(self) -> Any:
        """The kernel or the program whose sizes these are."""
        return self._index.root

    @property
    def text(self) -> str:
        """The condition in the term's names, ``nc % 4 == 0 and nc >= 1``."""
        return " and ".join(_python(part) for part in self._parts)

    def __and__(self, other: Any) -> Condition:
        if not isinstance(other, Condition):
            return NotImplemented
        if other._index.root is not self._index.root:
            raise ValueError(
                f"{self!r} and {other!r} are conditions of {self._index.name} and "
                f"{other._index.name}; a condition is of one kernel's sizes"
            )
        return Condition(self._index, (*self._parts, *other._parts))

    def __bool__(self) -> bool:
        raise TypeError(
            f"{self!r} is a condition of sizes, which has no truth value until a "
            "run gives the sizes. Join two with &, not and: (1 <= n) & (n <= 4), "
            "since Python reads 1 <= n <= 4 as (1 <= n) and (n <= 4)"
        )

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, Condition):
            return NotImplemented
        return self._index.root is other._index.root and self.text == other.text

    def __hash__(self) -> int:
        return hash((id(self._index.root), self.text))

    def __repr__(self) -> str:
        return " & ".join(
            _shown(part, self._index, len(self._parts) > 1) for part in self._parts
        )

    __str__ = __repr__


def _python(expression: Any, bracketed: bool = False) -> str:
    """An expression of sizes as Python, in the term's names."""
    from pymbolic.mapper.stringifier import StringifyMapper

    text = StringifyMapper()(expression)
    return f"({text})" if bracketed else text


def _shown(expression: Any, index: _Index, bracketed: bool = False) -> str:
    """An expression of sizes as Python, each size by its path."""
    from pymbolic.mapper.substitutor import substitute

    paths = {
        name: prim.Variable(index.size_path(name))
        for name in {var.name for var in _variables(expression)}
    }
    return _python(substitute(expression, paths), bracketed)


def _variables(expression: Any) -> list[Any]:
    """The variables of a pymbolic expression."""
    from pymbolic.mapper.dependency import DependencyMapper

    return list(DependencyMapper()(expression))


# }}}


# {{{ the names a definition gives, read without tracing it


def definition_names(root: Any) -> list[tuple[str, str]]:
    """The binders and parameters of a definition, each with its kind.

    What :func:`warn_shadowed` checks when a kernel or a program is
    defined: the names that are certainly the source's, not the sizes its
    annotations might name.
    """
    from loopty.kernel import Program

    fn = root.fn
    code = fn.__code__
    count = code.co_argcount + code.co_kwonlyargcount
    annotations = getattr(fn, "__annotations__", {}) or {}
    out = []
    for name in code.co_varnames[:count]:
        # a kernel's array parameter is annotated Arr[...]; a scalar is no
        # handle, and a program's parameters are mostly arrays
        text = str(annotations.get(name, ""))
        if isinstance(root, Program) or "Arr[" in text:
            out.append((name, "array"))
    tree = _function_tree(fn)
    if tree is not None:
        out += [(name, "loop") for name in _targets(tree)]
    return out


def warn_shadowed(root: Any) -> None:
    """Warn of a binder or a parameter named like an attribute of the API.

    Its short form is the attribute (see :class:`ShadowedName`), and the
    warning, issued where the kernel or the program is defined, names the
    long form that reaches it.
    """
    api = _api_names(root)
    owner = getattr(root, "__name__", "kernel")
    kind = type(root).__name__
    seen = set()
    for name, what in definition_names(root):
        if name not in api or name in seen or name.startswith("_"):
            continue
        seen.add(name)
        long_form = "loops" if what == "loop" else "arrays"
        warnings.warn(
            ShadowedName(
                f"the {what} {name} of {owner} ({root.where}) has the name of "
                f"{kind}.{name}, so {owner}.{name} is not the {what}; a schedule "
                f"names it {owner}.{long_form}.{name}"
            ),
            stacklevel=4,
        )


def _function_tree(fn: Any) -> ast.AST | None:
    """The function's definition, parsed, or ``None`` where it cannot be read."""
    try:
        source = textwrap.dedent(inspect.getsource(fn))
        return ast.parse(source)
    except (OSError, TypeError, SyntaxError, IndentationError):
        return None


def _targets(tree: ast.AST) -> list[str]:
    """The names the ``for`` loops and generators of a body bind, in order."""
    out: list[str] = []

    def names(target: ast.AST) -> Iterable[str]:
        if isinstance(target, ast.Name):
            yield target.id
        elif isinstance(target, ast.Tuple | ast.List):
            for element in target.elts:
                yield from names(element)

    for node in ast.walk(tree):
        if isinstance(node, ast.For | ast.AsyncFor | ast.comprehension):
            out.extend(names(node.target))
    return list(dict.fromkeys(out))


# }}}


def _joined(items: Sequence[str], word: str = "and") -> str:
    """``a``, ``a and b``, ``a, b and c``."""
    items = list(items)
    if len(items) <= 1:
        return "".join(items)
    return f"{', '.join(items[:-1])} {word} {items[-1]}"

