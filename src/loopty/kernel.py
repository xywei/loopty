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
from types import FunctionType, MethodType, ModuleType
from typing import Any

import numpy as np
from lanky.ledger import Fact, Status
from lanky.plugins import registry
from lanky.terms import evaluate_annotations

from loopty import typing as rules
from loopty.arr import Arr, ArrSpec
from loopty.contract import check_arguments, integral_sort
from loopty.term import (
    ArrType,
    Term,
    declared_layout,
    free_name_sorts,
    free_name_sorts_message,
)
from loopty.trace import TraceError, array_type, mask_writes, trace, when

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


#: How many levels of helper call :func:`opens_a_guard` follows before it gives
#: up. Deep enough for the helpers a kernel body actually calls, and finite
#: because the walk is over a graph that may well have a cycle.
_GUARD_SEARCH_DEPTH = 8


def _reachable_functions(fn: Any, names: set[str]) -> list[Any]:
    """Functions the body could be calling: its globals and its closure cells.

    Only plain functions and bound methods are followed. An arbitrary callable
    is not: reading attributes off one can run a property, and this is a
    question about what the body says rather than an invitation to execute part
    of it.
    """
    out: list[Any] = []
    globals_ = getattr(fn, "__globals__", None) or {}
    for name in names:
        value = globals_.get(name)
        if isinstance(value, FunctionType | MethodType):
            out.append(value)
    for cell in getattr(fn, "__closure__", None) or ():
        try:
            value = cell.cell_contents
        except ValueError:  # pragma: no cover - a cell not yet filled in
            continue
        if isinstance(value, FunctionType | MethodType):
            out.append(value)
    return out


def opens_a_guard(fn: Any) -> bool:
    """Does ``fn``, or a helper it calls, open a :class:`loopty.trace.when` block?

    Reported rather than relied on. The native run wraps its arrays in the
    masking views unconditionally (see :meth:`Kernel.__call__`), so an answer of
    ``False`` here no longer means a body's writes go unmasked; this says
    whether the guard is visible from the source, which is what a reader and a
    diagnostic want to know.

    It used to be asked as ``"when" in fn.__code__.co_names``, which is a
    question about spelling rather than about the object. ``from loopty import
    when as guard`` and ``loopty.when(...)`` both open a guard and neither
    mentions the bare name in the frame that uses it. So the guard object is
    looked for *by identity*: in the constants, the globals and the closure of
    every code object of the body, and as an attribute of any module those
    reach. The old name test is kept as well, because a body that gets hold of
    ``when`` in a way no static walk can follow still says ``when`` somewhere.

    The search follows function-valued globals and closure cells, to
    :data:`_GUARD_SEARCH_DEPTH` levels, because a body that calls a helper which
    opens the guard opens it too. Bounded and cycle-safe, since two mutually
    recursive helpers are an ordinary thing to write. Plain functions and bound
    methods only; anything else reached through a name is left alone.
    """
    pending = [(fn, 0)]
    seen: set[int] = set()
    while pending:
        current, depth = pending.pop()
        code = getattr(current, "__code__", None)
        if code is None or id(current) in seen:
            continue
        seen.add(id(current))
        names: set[str] = set()
        for nested in _code_objects(code):
            names |= set(nested.co_names)
            for const in nested.co_consts:
                if const is when:
                    return True
        if "when" in names:
            return True
        globals_ = getattr(current, "__globals__", None) or {}
        for name in names:
            if globals_.get(name) is when:
                return True
        for cell in getattr(current, "__closure__", None) or ():
            try:
                if cell.cell_contents is when:
                    return True
            except ValueError:  # pragma: no cover - a cell not yet filled in
                continue
        # ``loopty.when(...)`` reaches the guard through a module. Attributes
        # are read off modules only: asking an arbitrary object for an
        # attribute can run a property, and this is a question about the body,
        # not an invitation to execute part of it.
        for name in names:
            value = globals_.get(name)
            if isinstance(value, ModuleType) and any(
                getattr(value, attribute, None) is when for attribute in names
            ):
                return True
        if depth < _GUARD_SEARCH_DEPTH:
            pending.extend(
                (helper, depth + 1) for helper in _reachable_functions(current, names)
            )
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
        self._arg_types: dict[str, Any] | None = None
        #: Whether a ``when`` block is visible from the body, following the
        #: helpers it calls. Reported, not relied on: :meth:`__call__` masks
        #: every run. See :func:`opens_a_guard`.
        self.guards_writes = opens_a_guard(fn)

    # {{{ running

    @property
    def arg_types(self) -> dict[str, Any]:
        """Each parameter's type, read off the annotations without tracing.

        The same types :func:`loopty.trace.trace` gives the term, built here so
        that a native call can check its arguments against them without paying
        for a trace, and without loopty's native path depending on loopy.
        """
        if self._arg_types is None:
            annotations = dict(self.annotations)
            annotations.pop("return", None)
            out: dict[str, Any] = {}
            for name, annotation in annotations.items():
                if isinstance(annotation, ArrSpec):
                    out[name] = array_type(annotation, annotations, name)
                else:
                    out[name] = annotation
            self._arg_types = out
        return self._arg_types

    def _bound(self, args: tuple, kwargs: dict) -> dict[str, Any]:
        """The arguments of one call, by parameter name."""
        code = self.fn.__code__
        names = code.co_varnames[: code.co_argcount]
        bound = dict(zip(names, args, strict=False))
        bound.update(kwargs)
        return bound

    def __call__(self, *args: Any, **kwargs: Any) -> Any:
        """Run the body on real arrays: the reference implementation.

        The arguments are checked against the kernel's declared types first, by
        :mod:`loopty.contract`: two array parameters may not share storage, a
        ragged argument has to agree with the counts array its type names and
        with the offsets the kernel declares beside it, if it declares them,
        and an element of a refined sort such as ``Fin[m]`` has to be one.
        These are the assumptions the typing rules make about a *call*, and the
        native run is a call, so it makes the same promise the compiled run
        does rather than a weaker one.

        Every array argument is then wrapped in the masking view of
        :func:`loopty.trace.mask_writes`, which shares the caller's buffer, so
        the writes still land where the caller is looking. Two things follow
        from doing it unconditionally.

        *A write under a false ``when`` is dropped, whoever opened the block.*
        It used to be wrapped only when the guard was visible in the body's own
        code, so a body calling a helper that opens ``with when(cond):``
        performed the guarded write and ``python file.py`` computed something
        the lowered kernel does not. No inspection of the body can decide that
        question in general; wrapping always removes it.

        *A bare ``ndarray`` has a ``.dom``.* Sizes come from the arrays, so a
        body asks its arguments for their domains, and an ndarray has none; the
        view is an :class:`~loopty.arr.Arr` over the same buffer, so a caller
        may pass either. A ragged argument still has to be built with
        :meth:`~loopty.arr.Arr.ragged`, because nothing in an ndarray says where
        its rows end.

        The cost is one Python-level ``__getitem__`` per element access on the
        reference run, which is the slow path by construction; the demos are
        unchanged to the tenth of a second.

        *A ragged array is read through the layout the kernel declares.* The
        counts array its type names bounds a row, and a declared offsets
        parameter (:func:`loopty.term.declared_offsets`) says where a row
        starts, read as the body has left them (:meth:`loopty.arr.Arr.through`,
        :func:`loopty.term.declared_layout`). Those are the arrays the lowered
        kernel is handed and indexes through, so a kernel that writes them
        means the same thing natively as compiled, where following the
        array's own offsets gave it a second meaning. On entry the two layouts
        agree, which is what the contract checked above.

        *An index array stored as floats is read as integers.* The contract
        accepts ``col = [1.0, 0.0]`` for ``col: Arr[..., Fin[m]]``, because being
        a point of ``Fin[m]`` is a property of the value, and the compiled run
        casts it to an integer on the way in. numpy refuses a float as an index,
        so the native run used to raise on the very input the contract had just
        accepted, and the differential test could not compare the two. See
        :meth:`_integer_copies` for which arrays are copied and why only those.
        """
        bound = self._bound(args, kwargs)
        layout = declared_layout(tuple(self.arg_types.items()))
        check_arguments(
            self.arg_types,
            bound,
            {name: offsets for name, (_, offsets) in layout.items() if offsets},
        )
        copies = self._integer_copies(bound)
        code = self.fn.__code__
        names = code.co_varnames[: code.co_argcount]
        positional = [
            self._prepare(copies.get(names[k], arg) if k < len(names) else arg)
            for k, arg in enumerate(args)
        ]
        keywords = {
            name: self._prepare(copies.get(name, value))
            for name, value in kwargs.items()
        }
        prepared = {**dict(zip(names, positional, strict=False)), **keywords}
        for name, (counts, offsets) in layout.items():
            value = prepared.get(name)
            if not (isinstance(value, Arr) and value.is_ragged):
                continue
            view = value.through(
                prepared.get(counts) if counts else None,
                prepared.get(offsets) if offsets else None,
            )
            if name in keywords:
                keywords[name] = view
            else:
                positional[names.index(name)] = view
        return self.fn(*positional, **keywords)

    def _integer_copies(self, bound: dict[str, Any]) -> dict[str, Any]:
        """Integer copies of the float-stored arrays of an integral sort.

        Only arrays whose declared element sort is integral (``Fin``, ``Nat``,
        ``Int``) and whose storage is floating or complex are candidates, and
        :func:`loopty.contract.check_arguments` has already required every
        entry of those to be a finite whole number inside
        :data:`loopty.contract.INT64_RANGE`, so the copy is exact. The copy is
        ``int64`` while the lowering stores an integral sort as ``int32``; a
        value between the two ranges runs natively and is narrowed by the
        compiled run's cast, as the same value stored as ``int64`` is.

        An array the body *writes* is left as it is. A copy is a new buffer, and
        the native run's promise is that writes land in the caller's array; an
        array that is only read can be copied without anybody being able to
        tell. Which arrays are written is a fact about the term, so it is asked
        of the term, and only when there is a candidate at all: a call with no
        float-stored index array never traces. A body that cannot be traced
        still runs natively, with its arguments as given.
        """
        candidates: dict[str, Any] = {}
        for name, typ in self.arg_types.items():
            if not isinstance(typ, ArrType) or not integral_sort(typ.dtype):
                continue
            value = bound.get(name)
            buffer = value.numpy() if isinstance(value, Arr) else value
            if isinstance(buffer, np.ndarray) and buffer.dtype.kind in "fc":
                candidates[name] = value
        if not candidates:
            return {}
        try:
            written = {stmt.assignee.array for stmt in self.term.stmts}
        except Exception:  # noqa: BLE001 - reported by facts(), not by a native run
            return {}
        out: dict[str, Any] = {}
        for name, value in candidates.items():
            if name in written:
                continue
            if isinstance(value, Arr):
                whole = np.real(value.numpy()).astype(np.int64)
                out[name] = Arr(whole, value.offsets) if value.is_ragged else Arr(whole)
            else:
                out[name] = np.real(value).astype(np.int64)
        return out

    def _prepare(self, value: Any) -> Any:
        """One argument, as the body needs to see it: a masking view of it."""
        if isinstance(value, np.ndarray) and not isinstance(value, Arr):
            value = Arr(value)
        return mask_writes(value)

    # }}}

    # {{{ the term

    def trace(self) -> Term:
        """Trace the body against a generic point and return its term.

        A signature with a sort that is a free name (``a: float`` under
        postponed annotations gives ``Var("float")``) is refused first, with
        a :class:`~loopty.trace.TraceError` naming the sort to write instead;
        see :func:`loopty.term.free_name_sorts`. The native run does not need
        a sort and is not refused.
        """
        params = tuple(self.arg_types.items())
        found = free_name_sorts(params)
        if found:
            raise TraceError(free_name_sorts_message(self.qualname, params, found))
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

        The last one is the ``trace-faithful`` fact: the traced term,
        interpreted, against the native run of the body, on the module's
        example inputs and on inputs drawn from the declared types. Every other
        fact is about the term, and this one is about whether the term is the
        body; see :mod:`loopty.faithful`.

        A body that cannot be traced is itself reported as a fact rather than as
        a crash, so that ``lanky check`` on a file with one broken kernel still
        prints the ledger of the others.

        The error is also the fact's ``reason``, which lanky prints under the
        fact's ``REFUTED`` line (``lanky.cli.refutation_lines``, which ``loopty
        run`` prints too): the error of a :class:`~loopty.trace.TraceError`
        names the fix, and it belongs on the screen rather than only in the
        JSON ledger. There is no ``counterexample``, because the claim is not
        refuted at any assignment in particular, and lanky needs none to print
        the reason.
        """
        if self._facts is not None:
            return self._facts
        try:
            term = self.term
        except Exception as exc:  # noqa: BLE001 - reported as a fact, not raised
            error = f"{type(exc).__name__}: {exc}"
            self._facts = (
                Fact(
                    id=f"kernel:{self.qualname}:traced",
                    kind="trace",
                    statement=f"{self.qualname} can be traced",
                    term=None,
                    status=Status.REFUTED,
                    decided_by="trace",
                    provenance={"error": error, "reason": error},
                    where=self.where,
                    owner=self.qualname,
                ),
            )
            return self._facts
        from loopty.faithful import faithfulness_fact

        self._facts = (
            *rules.facts_for(term, owner=self.qualname, where=self.where),
            faithfulness_fact(self, term, owner=self.qualname, where=self.where),
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
    of the program*, which rests on the callee's own fact. That is lanky's
    ``rests_on``, so the ledger names the callee's postcondition beside the
    restatement (``assumed under scan:postcondition``) and counts it in what
    the restatement is worth, and a reader can see which claims the program
    depends on.

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
        """One fact per callee postcondition, as a claim in this program's scope.

        Each rests on the callee's postcondition fact, named by the id the
        callee's own facts give it (:func:`loopty.typing.postcondition_id`).
        When the callee is checked in the same file, that fact is in the same
        ledger, and the restatement is worth no more than it; when it is not,
        lanky counts the id it cannot find as an assumption.
        """
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
                    provenance={"callee": callee.qualname},
                    where=self.where,
                    owner=self.qualname,
                    rests_on=(rules.postcondition_id(callee.qualname),),
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
    each reduction, and the postcondition; then the ``trace-faithful`` fact,
    that the term computes what the body computes (:mod:`loopty.faithful`). An
    object this theory does not own gives an empty tuple, which is how several
    theories share one ledger.
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
