"""The ``@kernel`` and ``@program`` decorators, and the theory they belong to.

A decorated kernel stays callable: the decorator is inert in the mypy sense, so
``python file.py`` runs the body on numpy as the reference implementation. What
the decorator adds is registration (the object joins ``lanky.registry`` in
import order, so ``lanky check FILE`` finds it) and a lazily traced
:class:`~loopty.term.Term`.

Outputs are parameters, loopy style: a kernel takes its output arrays as
arguments, and the return annotation is not a result type but a proposition
about the parameters, the postcondition. Sizes are not parameters at all; they
come from the arrays, which is why the body iterates ``y.dom`` and never a name
out of the annotations, and why the very same body runs on real data.

Annotations are evaluated rather than parsed, so a kernel file wants
``from __future__ import annotations``: the sizes in ``Arr[Fin[n], Real]`` have
no binding anywhere, and lanky's :class:`~lanky.terms.Scope` invents them while
evaluating the annotation string. (ruff will want ``F821`` ignored for such a
file, the same as for a file carrying theorem statements.)

``KernelTheory`` is what lanky asks for the facts of such an object; it runs the
typing rules in :mod:`loopty.typing` over the traced term. Tracing is done once
and cached, because every fact about a kernel is a fact about the same term.
"""

from __future__ import annotations

import functools
import os
from types import ModuleType
from typing import Any

import numpy as np
from lanky.ledger import Fact, Status
from lanky.plugins import registry
from lanky.terms import evaluate_annotations

from loopty import typing as rules
from loopty.arr import Arr
from loopty.term import Term
from loopty.trace import mask_writes, trace, when

__all__ = [
    "Kernel",
    "KernelTheory",
    "Program",
    "ensure_registered",
    "kernel",
    "opens_a_guard",
    "program",
]


class _Decorated:
    """What ``@kernel`` and ``@program`` have in common: a wrapped function.

    Both keep the original function callable and remember where it was written,
    because a fact's value depends on being able to point at the source line
    that owes it.
    """

    def __init__(self, fn: Any) -> None:
        self.fn = fn
        functools.update_wrapper(self, fn)
        code = fn.__code__
        self.path = code.co_filename
        self.line = code.co_firstlineno
        self.where = f"{os.path.basename(code.co_filename)}:{code.co_firstlineno}"
        self.qualname = getattr(fn, "__qualname__", fn.__name__)

    @property
    def annotations(self) -> dict[str, Any]:
        """The annotations, evaluated in a scope that invents unknown names."""
        return evaluate_annotations(self.fn)


def _code_objects(code: Any) -> list[Any]:
    """``code`` and every code object nested in it.

    A ``with when(...)`` inside a comprehension, a nested ``def`` or a lambda
    lives in its own code object, and the outer one mentions only the constant
    that holds it. Guard detection has to look at all of them.
    """
    out = [code]
    seen = {id(code)}
    index = 0
    while index < len(out):
        for const in out[index].co_consts:
            if isinstance(const, type(code)) and id(const) not in seen:
                seen.add(id(const))
                out.append(const)
        index += 1
    return out


def opens_a_guard(fn: Any) -> bool:
    """Does ``fn``'s body open a :class:`loopty.trace.when` block?

    The question decides whether the native run wraps its arrays in the masking
    views, so getting it wrong makes ``python file.py`` compute something the
    lowered kernel does not: a write under a false guard is performed instead of
    dropped.

    It used to be asked as ``"when" in fn.__code__.co_names``, which is a
    question about spelling rather than about the object. ``from loopty import
    when as guard`` and ``loopty.when(...)`` both open a guard and neither
    mentions the bare name in the frame that uses it. So the guard object is
    looked for *by identity*: in the constants, the globals and the closure of
    every code object of the body, and as an attribute of any module those
    reach. The old name test is kept as well, because a body that gets hold of
    ``when`` in a way no static walk can follow still says ``when`` somewhere,
    and over-reporting only costs a wrapper.
    """
    codes = _code_objects(fn.__code__)
    names: set[str] = set()
    for code in codes:
        names |= set(code.co_names)
        for const in code.co_consts:
            if const is when:
                return True
    if "when" in names:
        return True
    reachable: list[Any] = []
    globals_ = getattr(fn, "__globals__", None) or {}
    for name in names:
        if name in globals_:
            reachable.append(globals_[name])
    for cell in getattr(fn, "__closure__", None) or ():
        try:
            reachable.append(cell.cell_contents)
        except ValueError:  # pragma: no cover - a cell not yet filled in
            continue
    for value in reachable:
        if value is when:
            return True
        # ``loopty.when(...)`` reaches the guard through a module. Attributes
        # are read off modules only: asking an arbitrary object for an
        # attribute can run a property, and this is a question about the body,
        # not an invitation to execute part of it.
        if isinstance(value, ModuleType):
            for name in names:
                if getattr(value, name, None) is when:
                    return True
    return False


class Kernel(_Decorated):
    """A decorated kernel: callable natively, traceable, registered.

    Attributes:
        term: The traced :class:`~loopty.term.Term`, built on first use.
        fn: The original function, which is the reference implementation.
    """

    def __init__(self, fn: Any) -> None:
        super().__init__(fn)
        self._term: Term | None = None
        self._facts: tuple[Fact, ...] | None = None
        #: Whether the body opens a ``when`` block, and so needs masked writes.
        self.guards_writes = opens_a_guard(fn)
        #: Whether the body iterates an array's ``.dom``, and so needs its
        #: arguments to be :class:`~loopty.arr.Arr` rather than bare ndarrays.
        self.iterates_domains = "dom" in fn.__code__.co_names

    # {{{ running

    def __call__(self, *args: Any, **kwargs: Any) -> Any:
        """Run the body on real arrays: the reference implementation.

        Arrays are handed through as they are, with two exceptions, both of
        which share the caller's buffers so that every write still lands where
        the caller is looking.

        A plain ``ndarray`` is wrapped in :class:`~loopty.arr.Arr` when the body
        iterates a ``.dom``. Sizes come from the arrays, so such a body asks its
        arguments for their domains, and a bare ndarray has none; without this
        the reference run failed with ``AttributeError: 'ndarray' object has no
        attribute 'dom'``, which names neither the cause nor the fix. Wrapping
        reads a bare ndarray as the dense array it is; a ragged argument still
        has to be built with :meth:`~loopty.arr.Arr.ragged`, because nothing in
        an ndarray says where its rows end.

        A body that uses ``when`` has its arrays wrapped again, so that a write
        under a false guard is dropped rather than performed.
        """
        return self.fn(
            *(self._prepare(arg) for arg in args),
            **{name: self._prepare(value) for name, value in kwargs.items()},
        )

    def _prepare(self, value: Any) -> Any:
        """One argument, as the body needs to see it."""
        if (
            self.iterates_domains
            and isinstance(value, np.ndarray)
            and not isinstance(value, Arr)
        ):
            value = Arr(value)
        return mask_writes(value) if self.guards_writes else value

    # }}}

    # {{{ the term

    def trace(self) -> Term:
        """Trace the body against a generic point and return its term."""
        self._term = trace(self, self.annotations)
        self._facts = None
        return self._term

    @property
    def term(self) -> Term:
        """The traced term, built once and kept."""
        if self._term is None:
            self.trace()
        assert self._term is not None
        return self._term

    # }}}

    def facts(self) -> tuple[Fact, ...]:
        """The obligations this kernel owes, from the typing rules.

        A body that cannot be traced is itself reported as a fact rather than as
        a crash, so that ``lanky check`` on a file with one broken kernel still
        prints the ledger of the others.
        """
        if self._facts is not None:
            return self._facts
        try:
            term = self.term
        except Exception as exc:  # noqa: BLE001 - reported as a fact, not raised
            self._facts = (
                Fact(
                    id=f"kernel:{self.qualname}:traced",
                    kind="trace",
                    statement=f"{self.qualname} can be traced",
                    term=None,
                    status=Status.REFUTED,
                    decided_by="trace",
                    provenance={"error": f"{type(exc).__name__}: {exc}"},
                    where=self.where,
                    owner=self.qualname,
                ),
            )
            return self._facts
        self._facts = tuple(
            rules.facts_for(term, owner=self.qualname, where=self.where)
        )
        return self._facts

    def __repr__(self) -> str:
        return f"<kernel {self.__name__} at {self.where}>"


class Program(_Decorated):
    """A sequence of kernel calls, run natively and recorded.

    A program is deliberately thin in this release. It runs its body, which
    calls kernels, which run natively, so ``python file.py`` works end to end.
    What it adds to the ledger is bookkeeping rather than reasoning: the
    postcondition of every kernel it calls is restated as a fact *in the scope
    of the program*, pointing at the callee's own fact, so that a reader can see
    which claims the program depends on.

    What it does not do yet is use those postconditions as hypotheses. Carrying
    the scan's recurrence into the in-bounds proof of the product is the
    interesting case, and it needs the isl oracle to accept a hypothesis, which
    the offsets formulation in :mod:`loopty.flow` does not yet provide. Until
    then the facts are recorded and left ``ASSUMED``, which is visible in the
    ledger rather than quietly assumed to be handled.
    """

    def __call__(self, *args: Any, **kwargs: Any) -> Any:
        """Run the body natively; its kernel calls run natively too."""
        return self.fn(*args, **kwargs)

    def callees(self) -> tuple[Kernel, ...]:
        """The kernels this program's body names.

        Read off the code object's global references, which is enough for the
        straight-line composition a program is, and needs no AST pass.
        """
        globals_ = getattr(self.fn, "__globals__", {})
        out = []
        for name in self.fn.__code__.co_names:
            value = globals_.get(name)
            if isinstance(value, Kernel) and value not in out:
                out.append(value)
        return tuple(out)

    def facts(self) -> tuple[Fact, ...]:
        """One fact per callee postcondition, as a claim in this program's scope."""
        out: list[Fact] = []
        for callee in self.callees():
            try:
                post = callee.term.post
            except Exception:  # noqa: BLE001 - the callee reports its own failure
                continue
            if post is None:
                continue
            from lanky.terms import render

            out.append(
                Fact(
                    id=f"program:{self.qualname}:{callee.qualname}:postcondition",
                    kind="postcondition-in-scope",
                    statement=(
                        f"after {callee.__name__}(...) in {self.__name__}: "
                        f"{render(post)}"
                    ),
                    term=post,
                    status=Status.ASSUMED,
                    provenance={
                        "callee": callee.qualname,
                        "from": f"{callee.qualname}:postcondition",
                    },
                    where=self.where,
                    owner=self.qualname,
                )
            )
        return tuple(out)

    def __repr__(self) -> str:
        return f"<program {self.__name__} at {self.where}>"


class KernelTheory:
    """The theory loopty registers with lanky, under ``lanky.theories``.

    ``facts(obj)`` traces a registered kernel and runs :mod:`loopty.typing` over
    the resulting term, returning the obligations: in-bounds per access, write
    disjointness, the ordering the dependences impose, the exactness class of
    each reduction, and the postcondition. An object this theory does not own
    gives an empty tuple, which is how several theories share one ledger.
    """

    name = "kernel"

    _instance: KernelTheory | None = None

    def __new__(cls) -> KernelTheory:
        """One theory per process, so that registering twice is detectable."""
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    def __call__(self, obj: Any, /) -> Kernel:
        """Decorate ``obj``; the same thing :func:`kernel` does."""
        return registry.register_object(Kernel(obj))

    def facts(self, obj: Any, /) -> tuple:
        """Return the facts claimed by a decorated kernel or program."""
        if isinstance(obj, Kernel | Program):
            return obj.facts()
        return ()


#: The theory itself, also usable as the ``@kernel`` decorator. There is exactly
#: one, because :meth:`KernelTheory.__new__` hands the same object back.
kernel_theory = KernelTheory()


def ensure_registered() -> KernelTheory:
    """Put the kernel theory in lanky's registry, once.

    Registration cannot happen at import time. ``lanky check`` loads plugins
    through the entry points, which imports :mod:`loopty.plugin`, which imports
    this module; if importing also registered, the theory would be in the
    registry twice, once from the import and once from the entry point, and
    every kernel's facts would be asked for twice. So the import is inert and
    the theory is registered on demand: by the entry point when lanky loads
    plugins, and by the first ``@kernel`` in a file when nothing loaded them.
    Either way the registry holds it once, and the check is by identity rather
    than by name so that a different ``Theory`` also called ``kernel`` is not
    silently swallowed.
    """
    if not any(theory is kernel_theory for theory in registry.theories):
        registry.register_theory(kernel_theory)
    return kernel_theory


def kernel(fn: Any) -> Kernel:
    """Decorate ``fn`` as a loopty kernel and register it with lanky."""
    return ensure_registered()(fn)


def program(fn: Any) -> Program:
    """Decorate ``fn`` as a sequence of kernel calls and register it."""
    ensure_registered()
    return registry.register_object(Program(fn))
