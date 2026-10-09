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

Every fact a kernel or a program claims is keyed by its definition, through
:func:`lanky.ledger.fact_id`: the kind, then the module the file's path gives
it under its source root (:func:`lanky.check.module_name`), the qualified name
and the line, then what the fact is about. ``scan``'s postcondition in
``spmv.py`` is ``postcondition:spmv.scan@69`` in that file's ledger and in the
``rests_on`` of any program that calls it, from that file or from another, and
a kernel of the same name defined somewhere else has an id of its own.
"""

from __future__ import annotations

import functools
import os
import sys
from types import FunctionType, MethodType, ModuleType
from typing import Any

import numpy as np
from lanky.check import module_name
from lanky.ledger import Fact, Status, fact_id
from lanky.plugins import registry
from lanky.terms import evaluate_annotations

from loopty import typing as rules
from loopty.arr import Arr, ArrSpec
from loopty.compose import current_recorder, trace_program
from loopty.contract import (
    check_arguments,
    native_copy,
    native_scalar,
    read_storage,
    written_storage,
)
from loopty.term import (
    ArrType,
    Term,
    declared_layout,
    free_name_sorts,
    free_name_sorts_message,
)
from loopty.trace import (
    TraceError,
    array_type,
    mask_writes,
    read_elements_as,
    trace,
    when,
)

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

    ``module`` is the module name the file's path gives it under its source
    root (:func:`lanky.check.module_name`), and ``__module__`` only for a
    function with no file behind it. It is what the facts' ids are keyed by,
    with ``qualname`` and ``line``, rather than the name the module was
    imported under, which differs between ``lanky check`` of the file and an
    import of it from another.
    """

    def __init__(self, fn: Any) -> None:
        self.fn = fn
        functools.update_wrapper(self, fn)
        code = fn.__code__
        self.path = code.co_filename
        self.line = code.co_firstlineno
        self.where = f"{os.path.basename(code.co_filename)}:{code.co_firstlineno}"
        self.qualname = getattr(fn, "__qualname__", fn.__name__)
        self.module = (
            module_name(code.co_filename) or getattr(fn, "__module__", "") or ""
        )

    @property
    def definition(self) -> str:
        """``spmv.scan@69``: the module, the qualified name and the line.

        It is the part of every fact id that names this object, as
        :func:`lanky.ledger.fact_id` writes it.
        """
        name = f"{self.module}.{self.qualname}" if self.module else self.qualname
        return f"{name}@{self.line}"

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

        *An array is read as its sort is stored.* The compiled run converts
        every array into the dtype its element sort is stored in on the way in,
        so an array the body only reads is read natively through a copy in
        :func:`loopty.contract.native_storage`'s dtype too. The contract
        accepts ``col = [1.0, 0.0]`` for ``col: Arr[..., Fin[m]]``, because
        being a point of ``Fin[m]`` is a property of the value, and numpy
        refuses a float as an index, so the native run used to raise on the
        very input the contract had just accepted; an integer ``x`` of a
        ``Real`` parameter used to overflow at ``x[i] * x[i]``, where the
        compiled run squares a double. An array the body writes cannot be read
        through a copy, since its writes have to land in the caller's array,
        and is refused unless it is stored as its sort is. See
        :meth:`_storage_copies` for which arrays are copied and why only those.
        The one written array that may be stored otherwise is an ``int32``
        array of ``Fin[m]``, which the compiled run stores in 32 bits too: its
        elements are read as ``int64``, which the compiled run computes
        integer arithmetic in, through the masking view and not through a
        copy (#121). ``p[i] * p[i]`` wrapped round at ``p[i] = 46341``
        natively before, where the compiled run computes it in 64 bits.
        A scalar is passed by value and is converted whatever the body does
        (:func:`loopty.contract.native_scalar`): ``np.int64(2**32)`` for a
        ``Real`` is a double, and Python's ``True`` for a ``Bool`` a numpy
        bool, on which ``~`` is ``not`` as it is compiled.

        *An array over a domain is run over the declared domain.* The contract
        asks an argument for the declared points, which another spelling can
        have, and a spelling is more than its points: ``L.dom`` runs binder by
        binder, and ``L.dom[i].size`` is the binder's bound. Over
        ``Sigma[a: Fin[n], Fin[a]]`` that size is ``i``, and over the declared
        ``Where[i: Fin[n], j: Fin[n], j < i]`` it is ``n``, which is what the
        trace and the compiled run say. Such an argument is copied into an
        array over the declared domain for the call and copied back after it
        (:meth:`_over_declared_domains`).

        *Inside a program whose term is being built, nothing runs.* The call is
        recorded, with what it was given, and the program's term is composed
        from the calls afterwards; see :mod:`loopty.compose`.
        """
        recorder = current_recorder()
        if recorder is not None:
            return recorder.call(self, args, kwargs, sys._getframe(1))
        bound = self._bound(args, kwargs)
        layout = declared_layout(tuple(self.arg_types.items()))
        check_arguments(
            self.arg_types,
            bound,
            {name: offsets for name, (_, offsets) in layout.items() if offsets},
        )
        stored, widened = self._storage_copies(bound)
        declared = self._over_declared_domains(bound, stored)
        copies = {**stored, **declared, **self._scalar_copies(bound)}
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
                name,
            )
            if name in keywords:
                keywords[name] = view
            else:
                positional[names.index(name)] = view
        for name, dtype in widened.items():
            if name in keywords:
                read_elements_as(keywords[name], dtype)
            elif name in names[: len(positional)]:
                read_elements_as(positional[names.index(name)], dtype)
        try:
            return self.fn(*positional, **keywords)
        finally:
            for name, copy in declared.items():
                # A storage copy is of an array the body only reads.
                if name not in stored:
                    bound[name].load(copy.storage, copy.numpy(), copy.domain)

    def _over_declared_domains(
        self, bound: dict[str, Any], stored: dict[str, Any]
    ) -> dict[str, Arr]:
        """Copies over the declared domain of the arrays built over another one.

        One for every argument over a domain that is not the declared domain
        at the call's sizes, holding the same values at the same points (the
        contract checked the points) in the argument's own storage, and taken
        from its storage copy when it has one (``stored``, see
        :meth:`_storage_copies`). The body runs on the copy, so its loops and
        sizes are the declared domain's, and :meth:`__call__` writes the copy
        back into the argument.
        """
        from loopty.contract import resolve_sizes

        out: dict[str, Arr] = {}
        sizes: dict[str, int] | None = None
        for name, typ in self.arg_types.items():
            if not isinstance(typ, ArrType) or typ.domain is None:
                continue
            value = stored.get(name, bound.get(name))
            if not (isinstance(value, Arr) and value.domain is not None):
                continue
            if sizes is None:
                sizes = resolve_sizes(self.arg_types, bound)
            needed = {size: sizes[size] for size in typ.domain.size_names()}
            given = value.domain
            if given.domain == typ.domain and given.sizes == needed:
                continue
            out[name] = Arr.from_cells(
                typ.domain, value.cells(), storage=value.storage or "box", **needed
            )
        return out

    def _storage_copies(
        self, bound: dict[str, Any]
    ) -> tuple[dict[str, Any], dict[str, np.dtype]]:
        """Copies of the arrays the body only reads, in the dtype of their sort.

        Returned with the dtype each written array that is not copied has its
        elements read in, which is ``int64`` for an ``int32`` array of
        ``Fin[m]`` (see below).

        An array is a candidate when its dtype does not hold its declared
        element sort as the compiled run holds it
        (:func:`loopty.contract.read_storage`): an integer or ``float32`` array
        of ``Real``, a float or ``int32`` one of ``Fin``, ``Nat`` or ``Int``, an
        ``int8`` one of ``Bool``. :func:`loopty.contract.check_arguments` has
        already required every entry of those to be a value of the sort (a
        finite whole number inside :func:`loopty.contract.integral_range`, no
        imaginary part, ``0`` or ``1``), so the copy changes no value, except by
        rounding to a narrower sort (a ``float64`` array of ``np.float32``),
        which the compiled run's conversion rounds the same way. An integral
        sort is copied as ``int64``, which the lowering stores ``Nat`` and
        ``Int`` in too, and in which both runs compute integer arithmetic; it
        stores ``Fin[m]`` in ``int32``, and the contract keeps a ``Fin[m]``
        value inside that range, so both hold it.

        An array the body *writes* is not copied. A copy is a new buffer, and
        the native run's promise is that writes land in the caller's array; an
        array that is only read can be copied without anybody being able to
        tell. So a written candidate is refused
        (:func:`loopty.contract.written_storage`), naming the dtype to pass.
        Which arrays are written is a fact about the term, so it is asked of
        the term, and only when there is a candidate at all: a call whose
        arrays are all stored as their sorts are never traces. A body that
        cannot be traced still runs natively, with its arguments as given.

        A written candidate that holds its sort as the compiled run does is
        accepted: an ``int32`` array of ``Fin[m]``, whose values ``m`` bounds,
        which the compiled run stores in 32 bits as well. The compiled run
        computes integer arithmetic on its elements in 64 bits, as on every
        integer's (:mod:`loopty.promotion`), so the native run reads each of
        them as an ``int64``, through the masking view
        (:func:`loopty.trace.read_elements_as`), and writes into the array as
        it is (#121).
        """
        candidates: dict[str, tuple[Any, np.dtype]] = {}
        for name, typ in self.arg_types.items():
            if not isinstance(typ, ArrType):
                continue
            value = bound.get(name)
            buffer = value.numpy() if isinstance(value, Arr) else value
            if not isinstance(buffer, np.ndarray):
                continue
            want = read_storage(typ.dtype, buffer.dtype)
            if want is not None:
                candidates[name] = (value, want)
        if not candidates:
            return {}, {}
        try:
            written = {stmt.assignee.array for stmt in self.term.stmts}
        except Exception:  # noqa: BLE001 - reported by facts(), not by a native run
            return {}, {}
        written_storage(self.arg_types, bound, written & candidates.keys())
        copies = {
            name: native_copy(value, want)
            for name, (value, want) in candidates.items()
            if name not in written
        }
        widened = {
            name: want
            for name, (_value, want) in candidates.items()
            if name in written
        }
        return copies, widened

    def _scalar_copies(self, bound: dict[str, Any]) -> dict[str, Any]:
        """The scalar arguments in the dtype of their sort, where they are not.

        :func:`loopty.contract.native_scalar` says which and why. A scalar is
        passed by value, so unlike an array it is converted whether or not
        the body assigns to its name, and without a trace.
        """
        out: dict[str, Any] = {}
        for name, sort in self.arg_types.items():
            if isinstance(sort, ArrType) or name not in bound:
                continue
            value = native_scalar(sort, bound[name])
            if value is not bound[name]:
                out[name] = value
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
        body; see :mod:`loopty.faithful`. The postcondition is tested on the
        same native runs: it is evaluated at what each of them left in the
        arguments, and is ``tested`` when it held after every one
        (:func:`loopty.faithful.postcondition_fact`), which is what a program
        that calls the kernel counts it as. So is a ``layout`` fact no isl
        question decides, which a run that moves a row off its buffer or onto
        another refutes (:func:`loopty.faithful.layout_fact`).

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
                    id=fact_id(
                        "trace", self.qualname, module=self.module, line=self.line
                    ),
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
        from loopty.faithful import faithfulness_fact, layout_fact, postcondition_fact

        key = {"module": self.module, "line": self.line}
        observed: list[Any] = []
        faithful = faithfulness_fact(
            self, term, owner=self.qualname, where=self.where, observed=observed, **key
        )
        stated = rules.facts_for(term, owner=self.qualname, where=self.where, **key)
        tested = [
            postcondition_fact(
                self, term, observed, owner=self.qualname, where=self.where, **key
            )
            if fact.kind == "postcondition"
            else layout_fact(term, fact, observed)
            if fact.kind == "layout"
            else fact
            for fact in stated
        ]
        self._facts = (*tested, faithful)
        return self._facts

    def __repr__(self) -> str:
        return f"<kernel {self.__name__} at {self.where}>"


class Program(_Decorated):
    """A sequence of kernel calls: run natively, traced, lowered as one kernel.

    ``python file.py`` runs the body, which calls kernels, which run natively,
    so a program works end to end without loopy.

    Its :attr:`term` is the kernels it calls, in call order, composed into one
    term in the program's names (:mod:`loopty.compose`): an array one call
    writes and the next reads is one array of that term, an array the body
    makes with :meth:`~loopty.arr.Arr.zeros_like` is a temporary, and the
    dependences between the calls are in the footprints, as between two
    statements of one kernel. That term lowers into one loopy kernel whose
    statements run in call order, so a program can be run compiled
    (``LoopyExecutor().run(solve, ...)``), scheduled (``Schedule(solve)``)
    and compared with its native run, which is what ``loopty run`` does with
    every program in a file. Fusing two calls' loops is a cast over this
    term (:meth:`~loopty.schedule.Schedule.fuse`), and computing an array
    the program makes where it is read, instead of storing it, is a step of
    a schedule too (:meth:`~loopty.schedule.Schedule.substitute`).

    The postcondition of every kernel it calls is restated as a fact *in the
    scope of the program*: after the call, the callee's claim holds of what
    the call passed it. The restatement is ``decided`` by the call, and rests
    on the callee's own fact (lanky's ``rests_on``), so the ledger counts that
    fact in what the restatement is worth: ``tested`` when the callee's
    postcondition was tested against its native runs. The id names the
    callee's definition, so a callee imported from another module under
    another name is named by its own fact there, and never by a kernel of the
    same name in the program's file.

    The postconditions are hypotheses as well. A call whose contract checks
    the cells of an array an earlier call wrote (a ``Fin[m]`` element sort,
    the offsets a ragged family is read through) has that check as a
    requirement of the program, decided by isl under the postconditions that
    held at the call and the theorems the program cites
    (``@program(uses=[scan_monotone])``), or checked by the compiled program
    between the two calls where it is not, or where what decided it is not at
    least ``tested`` (:mod:`loopty.compose`). Each is a ``requirement`` fact,
    resting on the facts it used. A callee's in-bounds
    fact its own term leaves ``assumed``, a flat ``val[off[r] + j]``, is
    decided under the same hypotheses where they decide it
    (:func:`loopty.typing.scoped_in_bounds_facts`).

    The last fact is the program's ``trace-faithful`` fact, as a kernel's is:
    its term, interpreted, against its body run natively, on the module's
    example inputs for it and on inputs drawn from the term's parameters
    (:mod:`loopty.faithful`). The term is built from what the body does with
    placeholders, and a body can look at what it was given in ways no
    placeholder sees (``isinstance(x, Arr)``, say), so this is the fact that
    catches a term that is not what the body computes.

    ``uses`` are the theorems the program cites, lanky's
    :class:`~lanky.theory.Theorem` objects (an axiom is one): a theorem is
    offered as a hypothesis by its statement, so an id or a fact, which name
    a claim without stating it, is refused.
    """

    def __init__(self, fn: Any, *, uses: Any = ()) -> None:
        super().__init__(fn)
        self._term: Term | None = None
        self._facts: tuple[Fact, ...] | None = None
        self.uses = _theorems(uses, self.qualname)

    def __call__(self, *args: Any, **kwargs: Any) -> Any:
        """Run the body natively; its kernel calls run natively too.

        Inside another program whose term is being built, the body runs
        against that program's placeholders, so its calls are recorded there,
        in place.
        """
        return self.fn(*args, **kwargs)

    def trace(self) -> Term:
        """Run the body against placeholders and compose the calls it makes.

        See :func:`loopty.compose.trace_program`. A body the composition
        refuses raises :class:`~loopty.trace.TraceError` naming the fix; the
        native run does not need a term and is not refused.
        """
        self._term = trace_program(self)
        self._facts = None
        return self._term

    @property
    def term(self) -> Term:
        """The program's term, built once and kept."""
        if self._term is None:
            self.trace()
        assert self._term is not None
        return self._term

    def callees(self) -> tuple[Kernel, ...]:
        """The kernels this program's body names, and those of the programs it names.

        Read off the code object's global references, which is enough for the
        straight-line composition a program is, and needs no AST pass. A
        program the body calls has its calls recorded in place in this
        program's term (:mod:`loopty.compose`), so its kernels are this
        program's callees too, and their postconditions are restated here.
        """
        out: list[Kernel] = []
        seen = {id(self)}

        def visit(program: Program) -> None:
            globals_ = getattr(program.fn, "__globals__", {})
            for name in program.fn.__code__.co_names:
                value = globals_.get(name)
                if isinstance(value, Kernel) and value not in out:
                    out.append(value)
                elif isinstance(value, Program) and id(value) not in seen:
                    seen.add(id(value))
                    visit(value)

        visit(self)
        return tuple(out)

    def facts(self) -> tuple[Fact, ...]:
        """The restatements, the requirements, then the ``trace-faithful`` fact.

        One restatement per callee postcondition, ``decided`` by the call
        (see :class:`Program`). Each rests on the callee's postcondition fact,
        named by the id the callee's own facts give it
        (:func:`loopty.typing.postcondition_id`),
        which is keyed by the callee's definition: the module its file's path
        gives it, its qualified name and its line. When the callee is checked
        in the same file, that fact is in the same ledger, and the
        restatement is worth no more than it; when it is not, lanky counts the
        id it cannot find as an assumption and names it under the table, and
        it is the id the callee's own file's ledger holds. A kernel of the
        program's file that has the callee's name, as in ``from helpers import
        scan as helper_scan`` beside a ``scan`` of its own, has an id of its
        own, and cannot stand in for the callee.

        The restatement's own id names the program and then the callee's
        definition, so two callees of one name get a restatement each.

        Then one ``requirement`` fact per requirement of the term
        (:func:`loopty.typing.requirement_facts`), each resting on the
        restatements and theorems it was decided by, or ``assumed`` and
        checked when the program runs; and the in-bounds facts its calls'
        hypotheses decide (:func:`loopty.typing.scoped_in_bounds_facts`).

        The ``trace-faithful`` fact compares the program's term, interpreted,
        with its body, run natively, as a kernel's does
        (:func:`loopty.faithful.faithfulness_fact`), keyed by the program's
        definition. A program whose term cannot be built has no term to
        compare, and the fact is ``assumed`` with the reason; ``loopty run``
        reports the same program as one it cannot schedule. Before it come
        the ``layout`` facts of the term (:func:`loopty.typing.layout_facts`),
        when a call rewrites the counts or the offsets a ragged array of the
        term is read through: a ``Schedule`` of the program rests its
        ``monotone`` casts on them, as one of a kernel does. And before it
        too, a ``definedness`` fact for each array the program makes and
        each call that reads it after another call wrote it
        (:func:`loopty.typing.definedness_facts`): whether each cell it
        reads is one a call before it stored or one of the zeros the array
        was made with, and, where it reads zeros no call stored, which.
        """
        if self._facts is not None:
            return self._facts
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
                    id=rules.restatement_id(
                        self.qualname,
                        callee.definition,
                        module=self.module,
                        line=self.line,
                    ),
                    kind="postcondition-in-scope",
                    statement=(
                        f"after {callee.__name__}(...) in {self.__name__}: "
                        f"{render(post)}"
                    ),
                    term=rules.AfterCall(callee.__name__, post),
                    status=Status.DECIDED,
                    decided_by="call",
                    provenance={
                        "callee": callee.qualname,
                        "rule": (
                            f"every call of {callee.__name__} leaves its "
                            "postcondition true of what it was passed, so it "
                            "holds after each call the program makes"
                        ),
                    },
                    where=self.where,
                    owner=self.qualname,
                    rests_on=(
                        rules.postcondition_id(
                            callee.qualname, module=callee.module, line=callee.line
                        ),
                    ),
                )
            )
        from loopty.faithful import faithfulness_fact, no_term_fact

        key = {"module": self.module, "line": self.line}
        try:
            term = self.term
        except Exception as exc:  # noqa: BLE001 - reported as a fact, not raised
            out.append(
                no_term_fact(
                    self.qualname,
                    self.where,
                    f"the term of {self.__name__} cannot be built: "
                    f"{type(exc).__name__}: {exc}",
                    **key,
                )
            )
        else:
            out.extend(rules.requirement_facts(term, self.qualname, **key))
            out.extend(rules.scoped_in_bounds_facts(term, self.qualname, **key))
            out.extend(rules.layout_facts(term, self.qualname, **key))
            out.extend(rules.definedness_facts(term, self.qualname, **key))
            out.append(
                faithfulness_fact(
                    self, term, owner=self.qualname, where=self.where, **key
                )
            )
        self._facts = tuple(out)
        return self._facts

    def __repr__(self) -> str:
        return f"<program {self.__name__} at {self.where}>"


class KernelTheory:
    """The theory loopty registers with lanky, under ``lanky.theories``.

    ``facts(obj)`` traces a registered kernel and runs :mod:`loopty.typing` over
    the resulting term, returning the obligations: in-bounds per access, write
    disjointness, the ordering the dependences impose, the exactness class of
    each reduction, and the postcondition; then the ``trace-faithful`` fact,
    that the term computes what the body computes (:mod:`loopty.faithful`). A
    program's are its callees' postconditions restated, and its own
    ``trace-faithful`` fact. An object this theory does not own gives an empty
    tuple, which is how several theories share one ledger.
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


def program(fn: Any = None, /, *, uses: Any = ()) -> Any:
    """Decorate ``fn`` as a sequence of kernel calls and register it.

    ``@program`` alone, or ``@program(uses=[scan_monotone])`` to cite
    theorems, which are offered as hypotheses wherever the program's
    requirements are decided (see :class:`Program`).
    """
    ensure_registered()
    if fn is None:
        theorems = _theorems(uses, "the program")

        def decorate(function: Any) -> Program:
            return registry.register_object(Program(function, uses=theorems))

        return decorate
    if not callable(fn) or getattr(fn, "__code__", None) is None:
        raise TypeError(
            f"@program decorates a function, and was given {fn!r}; the theorems "
            "a program cites go in uses=[...]"
        )
    return registry.register_object(Program(fn, uses=uses))


def _theorems(uses: Any, owner: str) -> tuple[Any, ...]:
    """The theorems a ``uses=`` argument names, checked to be statements.

    A theorem is offered to a program's requirements as a hypothesis, which
    takes its statement: its variables, its hypotheses and its goal. An id
    or a :class:`~lanky.ledger.Fact` names a claim without stating it in
    that form, so it is refused, naming the fix.
    """
    if uses is None:
        raise TypeError(
            f"uses=None names no theorem for {owner}; a program that cites none "
            "leaves uses= out"
        )
    entries = [uses] if _is_statement(uses) or isinstance(uses, str) else list(uses)
    out: list[Any] = []
    for entry in entries:
        if not _is_statement(entry):
            raise TypeError(
                f"uses= of {owner} names theorems, which are offered as "
                f"hypotheses by their statements, and {entry!r} states nothing a "
                "program can instantiate; pass the theorem itself, as "
                "uses=[scan_monotone]"
            )
        if not any(entry is seen for seen in out):
            out.append(entry)
    return tuple(out)


def _is_statement(entry: Any) -> bool:
    """Whether ``entry`` is a theorem a program can instantiate."""
    return (
        hasattr(entry, "variables")
        and hasattr(entry, "hypotheses")
        and hasattr(entry, "goal")
        and isinstance(getattr(entry, "fact_id", None), str)
    )
