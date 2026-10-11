"""A kernel called inside another kernel's loops: inlined, with fresh binders.

A kernel is a function, and a kernel body calls one as it calls any function:

.. code-block:: python

    @kernel
    def dot(a: Arr[Fin[n], Real], b: Arr[Fin[n], Real]) -> Real:
        return reduce_sum(a[k] * b[k] for k in a.dom)

    @kernel
    def matvec(A: Arr[Fin[m], Fin[n], Real], x: Arr[Fin[n], Real],
               y: Arr[Fin[m], Real]):
        for i in y.dom:
            y[i] = dot(A[i], x)

``dot`` is a *rule*: its return annotation is the sort of the value it
returns, and it writes none of its parameters, so a call of it is a value. A
kernel that writes its parameters is called as a statement,
``element(X[cell[c]], Y[cell[c]], M[c])``, and writes through what it was
given. Natively both are what they say: ``A[i]`` is numpy's row of ``A``,
``u[cell[c]]`` the cells ``cell[c]`` picks, and the callee runs on them,
returning its value or writing into the row.

*Traced, a call is inlined, with binders of its own.* The callee's body runs
once more, inside the caller's trace, against views of the caller's arrays
(:class:`loopty.trace._RowView`, :class:`loopty.trace._GatherView`): ``a[k]``
reads ``A[i, k]``, ``a.dom`` runs over the row, and a write lands in the
caller's array. Its loops are loops of the caller nested inside the loops
around the call, and its sums sums of the caller's statement, so the callee's
domain is instantiated at every point the call is made at, and a name the
callee's loops or sums share with the caller's is renamed, ``k_0`` (see
:meth:`loopty.trace.Tracer.bind`). That is loopy's substitution rule with a
domain of its own (loopy #1044), expanded where it is used: the caller's term
holds the expansion, and every fact of the caller, its in-bounds and disjoint
writes and dependences, is about it, decided over the points it runs at, as
for code the caller wrote itself. Lowering and the interpreter see a term
like any other.

*What is in scope in the callee is its parameters and its domain* (loopy
#1045's question). A rule reads only what it is passed, and the tables of
its module, which are part of its code.

*A rule may take a rule* (loopy #1043). A parameter of a function sort,
``f: Fn[Fin[3], Real]``, is passed a function, a ``lambda`` or another rule,
and the callee's body calls it where it uses it: the function runs in the
caller's trace too, so a quadrature rule applied to an integrand is
``integrate(lambda q: u[q] * PHI[q, j])``.

*The call owes what the callee's contract checks.* Natively each call runs
the callee's contract: the arguments have the shapes the callee declares, a
row is a row of its array, a scalar of a ``Fin[m]`` sort a point of it, and
two arguments do not share storage where the callee writes either. Traced,
the shapes are unified when the call is made, size by size, and a
disagreement is refused, naming the axis. The rest are facts of the caller,
about every point the call is made at (:func:`loopty.typing.call_facts`):
each row in bounds, each integral scalar in its sort, the cells a gather
picks in bounds, two rows of one array apart where the callee writes one.
isl decides them, and Lean states each from loopty's own description of the
call (:mod:`loopty.obligations`).

*An argument is evaluated once, when the call is made*, as Python does: the
value a scalar has then, the row its indices name then, the cells a gather
copies then. Inlined, it is read where the callee uses it, so a call whose
callee writes an array one of its arguments reads is refused, naming both
(:meth:`_Inlining.refuse_by_name`).

*The callee's postcondition travels into the caller.* After every call the
callee's claim holds of what the call passed it: ``all(Me[j, k] == Me[k, j]
...)`` of ``element`` is ``all(M[c, j, k] == M[c, k, j] ...)`` at every ``c``
the caller calls it at, a fact of the caller decided by the call and resting
on the callee's own postcondition fact, as a program restates its callees'.

The call is recorded in the caller's term as a :class:`~loopty.term.Call`,
with what was passed for each parameter (:class:`~loopty.term.Argument`).
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any, NoReturn

import numpy as np
import pymbolic.primitives as prim
from lanky.prelude import FinType, FnType, Refined
from lanky.terms import SymbolicMixin, Var, current_trace, init_args, render

from loopty.compose import _names_in, _plus, _same, _shifted, rename_expr
from loopty.flow import domain_set
from loopty.term import Argument, ArrType, Call
from loopty.trace import (
    SymArr,
    SymDom,
    TraceError,
    Tracer,
    _binder_bound,
    _conjuncts,
    _conversion_dtype,
    _domain_texts,
    _GatherView,
    _location,
    _RowView,
    _shown,
    constraints_of,
)

__all__ = ["Guarded", "inline", "takes_a_function"]

#: How deep calls inside calls are inlined before the nest is refused as one
#: that would not end (see :meth:`_Inlining.refuse_recursion`).
_NESTING = 64


def takes_a_function(kernel: Any) -> str | None:
    """The first parameter of ``kernel`` of a function sort, or ``None``.

    Such a kernel is compiled where it is called, with the function it is
    passed there; on its own it has nothing to call.
    """
    for name, sort in kernel.arg_types.items():
        if _function_sort(sort):
            return name
    return None


def _function_sort(sort: Any) -> bool:
    """Whether ``sort`` is a function sort, ``Fn[A, B]``, refined or not."""
    base = sort.base if isinstance(sort, Refined) else sort
    return isinstance(base, FnType)


class _Refused(Exception):
    """A name of the callee's claim that the caller's names cannot say."""


def inline(kernel: Any, tracer: Tracer, args: tuple, kwargs: dict, frame: Any) -> Any:
    """Trace a call of ``kernel`` inside the body ``tracer`` traces, in place.

    Returns what the call is in the caller's body: the value of a rule, an
    expression in the caller's names, or ``None``. See the module docstring.
    """
    return _Inlining(kernel, tracer, frame).run(args, kwargs)


class _Inlining:
    """One call being inlined: the callee, the caller's trace, and where."""

    def __init__(self, kernel: Any, tracer: Tracer, frame: Any) -> None:
        self.kernel = kernel
        self.tracer = tracer
        self.where = _location(frame)
        self.callee = kernel.__name__
        self.caller = tracer.name
        #: The arrays over a domain passed, as ``(param, callee's domain,
        #: caller's domain, the array)``, compared once the sizes are settled.
        self.domains: list[tuple[str, Any, Any, str]] = []

    def refuse(self, message: str) -> NoReturn:
        raise TraceError(
            f"{self.caller} calls {self.callee} at {self.where}: {message}"
        )

    # {{{ the call

    def run(self, args: tuple, kwargs: dict) -> Any:
        kernel, tracer = self.kernel, self.tracer
        bound = self.bind(args, kwargs)
        key = self.reentry_key(bound)
        self.refuse_recursion(key)
        try:
            term = kernel.term
        except TraceError as exc:
            self.refuse(f"the body of {self.callee} cannot be traced: {exc}")
        declared = kernel.arg_types
        label = tracer.call_label(self.callee)
        inames, domain, loop_domain, unnarrowed = self.points()
        unknown = self.unseen_binders()
        values: dict[str, Any] = {}
        proxies: dict[str, Any] = {}
        arguments: dict[str, Argument] = {}
        written = {stmt.assignee.array for stmt in term.stmts}
        if term.result is not None:
            written.discard(term.result)
        for param, sort in declared.items():
            if not isinstance(sort, ArrType):
                proxies[param] = self.scalar(param, sort, bound[param], values)
        pending: list[tuple[str, Any, Any, str]] = []
        for param, typ in declared.items():
            if isinstance(typ, ArrType):
                proxy = self.array(param, typ, bound[param], bound, values, pending)
                proxies[param] = proxy
        self.settle_all(pending, values)
        for param, sort in declared.items():
            value = proxies[param]
            if isinstance(sort, ArrType):
                arguments[param] = self.array_argument(param, value, param in written)
            elif _function_sort(sort):
                arguments[param] = Argument(
                    param,
                    "function",
                    getattr(value, "__name__", repr(value)),
                    value=_captured(value),
                )
            else:
                # What the caller passed, which the call's facts are about; the
                # body is given it converted into the sort (converted).
                given = bound[param]
                spelled = _shown(given)
                if isinstance(given, np.integer):
                    # A number, which isl reads as a Python int.
                    given = given.item()
                unfixed = _unknown(_fin_bound(sort) or 0, values, self.sizes())
                arguments[param] = Argument(
                    param,
                    "scalar",
                    spelled,
                    value=given,
                    sort=self.in_caller(sort, values),
                    unfixed=tuple(sorted(unfixed)),
                )
        self.refuse_shared(arguments, written)
        self.refuse_by_name(arguments, written)
        before = len(tracer.stmts)
        tracer.inlining.append((label, key))
        try:
            # In the order of its parameters, as trace calls it: a parameter
            # may be positional-only (#233).
            value = kernel.fn(*(proxies[name] for name in self.parameters()))
        finally:
            tracer.inlining.pop()
        statements = tuple(stmt.id for stmt in tracer.stmts[before:])
        if term.result is not None:
            if value is None:
                self.refuse(f"{self.callee} returned nothing where it was called")
        elif value is not None:
            self.refuse(
                f"{self.callee} returned {_shown(value)}, and declares no sort for "
                "it; a kernel that returns its value declares the sort after ->"
            )
        post = self.postcondition(term, proxies, values)
        if term.result is not None and tracer.guards:
            # Natively a call under a false guard is not made, and its value
            # there is 0: the value outside the when block is that.
            spelled = ", ".join(argument.spelled for argument in arguments.values())
            value = Guarded(tracer.guard(), value, 0)
            tracer.guarded[id(value)] = (value, f"{self.callee}({spelled})")
        tracer.calls.append(
            Call(
                callee=self.callee,
                definition=kernel.definition,
                label=label,
                where=self.where,
                inames=inames,
                domain=domain,
                arguments=tuple(arguments.values()),
                statements=statements,
                post=post,
                postcondition=self.postcondition_id(term),
                returns=term.result is not None,
                loop_domain=loop_domain,
                unnarrowed=unnarrowed,
                unknown=unknown,
            )
        )
        return value

    def parameters(self) -> tuple[str, ...]:
        """The callee's parameters, in the order of its signature."""
        code = self.kernel.fn.__code__
        return code.co_varnames[: code.co_argcount]

    def bind(self, args: tuple, kwargs: dict) -> dict[str, Any]:
        """The call's arguments by the callee's parameter names.

        A parameter before the ``/`` of the signature is positional-only, and
        Python refuses it by name natively, so the call is refused so too.
        """
        names = self.parameters()
        positional = names[: self.kernel.fn.__code__.co_posonlyargcount]
        if len(args) > len(names):
            self.refuse(f"it is given {len(args)} arguments; it takes {len(names)}")
        bound = dict(zip(names, args, strict=False))
        for key, value in kwargs.items():
            if key not in names or key in bound:
                self.refuse(
                    f"the argument {key} is given "
                    + ("twice" if key in bound else "and is no parameter of it")
                )
            if key in positional:
                self.refuse(
                    f"its parameter {key} is positional-only (before the / of its "
                    f"signature), and it is given by name, which Python refuses "
                    f"natively; pass it by position"
                )
            bound[key] = value
        missing = [name for name in names if name not in bound]
        if missing:
            self.refuse(f"it is not given {', '.join(missing)}")
        return bound

    def reentry_key(self, bound: Mapping[str, Any]) -> tuple[Any, tuple[int, ...]]:
        """The callee and the code of each function it is passed, by identity.

        A call made while the same kernel is being inlined is a nested use,
        not a recursion, when it is passed other functions: the quadrature
        rule applied to an integrand that applies it again,
        ``integrate(lambda q: integrate(lambda p: g(p, q)))``, inlines the
        rule twice, once per lambda. A chain of calls that never ends repeats
        a kernel with the same code for its functions, since a program has
        finitely many functions; that is what :meth:`refuse_recursion` looks
        for.
        """
        codes = []
        for name, sort in self.kernel.arg_types.items():
            if _function_sort(sort):
                value = bound.get(name)
                code = getattr(value, "__code__", None)
                if code is None:
                    code = getattr(getattr(value, "fn", None), "__code__", value)
                codes.append(id(code))
        return self.kernel, tuple(codes)

    def refuse_recursion(self, key: tuple[Any, tuple[int, ...]]) -> None:
        """Refuse a call that would inline its kernel into itself without end."""
        kernel, codes = key
        repeated = any(
            entry[0] is kernel and entry[1] == codes
            for _label, entry in self.tracer.inlining
        )
        if getattr(kernel, "_tracing", False) or repeated:
            given = " with the same functions" if codes else ""
            self.refuse(
                f"{self.callee} is already being called there{given}, so the call "
                "would inline it into itself without end; a kernel does not call "
                "itself, directly or through the functions it is passed"
            )
        if len(self.tracer.inlining) >= _NESTING:
            self.refuse(
                f"calls are nested {_NESTING} deep there, "
                f"{' in '.join(label for label, _key in self.tracer.inlining)}; "
                "a kernel's calls are inlined, and a nest that deep is taken for "
                "one that does not end"
            )

    def postcondition_id(self, term: Any) -> str | None:
        """The id of the callee's postcondition fact, when it has one."""
        if term.post is None:
            return None
        from loopty.typing import postcondition_id

        kernel = self.kernel
        return postcondition_id(kernel.qualname, module=kernel.module, line=kernel.line)

    # }}}

    # {{{ where the call is made

    def points(self) -> tuple[tuple[str, ...], Any, Any, tuple[tuple[str, str], ...]]:
        """The caller's loops and binders around the call, and the points of them.

        The loops of the caller's trace and the binders of the sums the call is
        made inside (:attr:`loopty.trace.Tracer.open_binders`), with the guards
        of the ``when`` blocks around it and the ``if`` clauses of the sums'
        generators, where isl can state them; the conjuncts it cannot are
        returned with the reason, as a statement lists them. Also the points
        before the guards narrowed them, ``None`` when nothing did.
        """
        tracer = self.tracer
        names = list(tracer.inames)
        bounds = list(tracer.bounds)
        loops = list(tracer.loop_constraints())
        binders: list[str] = []
        fibers: list[str] = []
        clauses: list[str] = []
        unstated: list[tuple[str, str]] = []
        for _iterator, var, dom, binder_trace in tracer.open_binders:
            binders.append(var.name)
            names.append(var.name)
            bounds.append(_binder_bound(dom))
            if isinstance(dom, SymDom):
                fibers.extend(
                    _domain_texts(
                        dom.constraints_for(var),
                        tracer,
                        binders,
                        f"the domain of the sum over {var.name}",
                    )
                )
        seen: set[int] = set()
        for _iterator, _var, _dom, binder_trace in tracer.open_binders:
            if id(binder_trace) in seen:
                continue
            seen.add(id(binder_trace))
            for clause in binder_trace.guards:
                for part, text, why in _conjuncts(clause, tracer, binders):
                    if text is None:
                        unstated.append((_shown(part), why))
                    else:
                        clauses.append(text)
        guards = constraints_of(tracer.guard(), tracer, binders)
        unstated.extend(tracer.unnarrowed())
        domain = domain_set(
            names,
            bounds,
            constraints=(*loops, *fibers, *clauses, *guards),
            reflections=tracer.reflections,
        )
        loop_domain = None
        if guards or clauses:
            loop_domain = domain_set(
                names,
                bounds,
                constraints=(*loops, *fibers),
                reflections=tracer.reflections,
            )
        return tuple(names), domain, loop_domain, tuple(unstated)

    def unseen_binders(self) -> str:
        """Why the points of the call cannot be stated, or ``""`` when they can.

        A sum binds its binders through loopty's domains, which the tracer
        sees (:meth:`loopty.trace.Tracer.bind`); one over a ``Fin`` of a
        symbolic size is bound by lanky alone, and a call inside it is made
        at points of a binder the call's domain would not have.
        """
        innermost = current_trace()
        if innermost is None:
            return ""
        seen = {id(var) for _it, var, _dom, _trace in self.tracer.open_binders}
        unseen = [var.name for var, _domain in innermost.binders if id(var) not in seen]
        if not unseen:
            return ""
        return (
            f"the call is made inside a sum over {', '.join(unseen)}, a binder "
            "the trace does not bind itself, so the points it is made at are not "
            "known; sum over an array's .dom"
        )

    # }}}

    # {{{ the arguments

    def scalar(self, param: str, sort: Any, value: Any, values: dict[str, Any]) -> Any:
        """What the callee's body is given for a scalar or a function parameter."""
        from loopty.contract import integral_sort

        value = unguarded(value, self.tracer, self.where)
        if _function_sort(sort):
            if isinstance(value, prim.ExpressionNode | SymArr | SymDom) or not callable(
                value
            ):
                self.refuse(
                    f"its {param} is a function, {sort}, and it is given "
                    f"{_shown(value)}; pass a function of the point, as "
                    "lambda q: ..., or a rule"
                )
            return value
        if isinstance(value, SymArr | SymDom | np.ndarray) or value is None:
            self.refuse(
                f"its scalar parameter {param} is given {_shown(value)}, which is "
                "not a value: pass a number, a loop variable, or an expression of "
                "them"
            )
        if not isinstance(
            value,
            bool | int | float | complex | np.number | np.bool_ | prim.ExpressionNode,
        ):
            self.refuse(
                f"its scalar parameter {param} is given {value!r}, which is neither "
                "a number nor a value the body computed"
            )
        if integral_sort(sort):
            if isinstance(value, bool | np.bool_) or (
                isinstance(value, float | complex | np.floating | np.complexfloating)
            ):
                self.refuse(
                    f"its scalar parameter {param} is declared {sort}, and it is "
                    f"given {value!r}, which is not an integer; a value of an "
                    "integral sort is passed as one"
                )
            values[param] = value.item() if isinstance(value, np.generic) else value
        return self.converted(param, sort, value)

    def converted(self, param: str, sort: Any, value: Any) -> Any:
        """``value`` as the callee's body computes with it: in its sort's storage.

        Natively the callee's contract converts every scalar it is given into
        the numpy type its sort is stored in (:func:`loopty.contract.native_scalar`):
        a double given for ``np.float32`` is rounded to single precision, an
        integer given for ``Real`` is a double, and a loop variable given for
        ``Int`` is an ``np.int64``, which numpy computes with as a numpy type
        and not as a Python int beside a ``float32`` (NEP 50). Inlined, the
        value is put where the callee uses the parameter, so it is put there
        converted, ``np.float32(x[i])``: a call the interpreter computes as the
        contract converts (:data:`loopty.contract.CONVERSIONS`) and the
        lowering writes as C's cast (#232). A value the caller computes in
        that type already is put there as it is, and where an index or a
        comparison of integers reads a conversion into ``int64`` it is left out
        (:func:`exact`), so that ``A[i, j]`` reads as the caller's own access.

        What the contract refuses natively is refused, with the fix: a value
        that is not an integer, or is a truth value, for an integral sort; a
        number for ``Bool``, which the contract takes only as ``0`` or ``1``; a
        complex value for a sort that is not complex, which it takes only with
        no imaginary part; and a literal it refuses or numpy does not convert,
        ``300`` for ``np.int8``. The compiled kernel cannot ask those of a value
        it computes, so a value of the caller's is refused where its type
        allows one at all. A Python int a narrow integer does not hold where
        the call is made is refused when the compiled run is asked for, at the
        call's sizes (:func:`loopty.refusal.refuse_where_native_does`).
        """
        from lanky.terms import Call

        from loopty.contract import CONVERSIONS, conversion_name, native_storage

        want = native_storage(sort)
        if want is None:
            return value
        name = conversion_name(want)
        if name not in CONVERSIONS:
            self.refuse(
                f"its scalar parameter {param} is declared {_sort_text(sort)}, which "
                f"natively is a {want}, and the compiled kernel has no {want}; "
                f"declare {param} np.float32 or Real"
            )
        if not isinstance(value, prim.ExpressionNode):
            self.literal(param, sort, value, want)
            # Put there as a call, so that the callee computes with it as
            # with a numpy scalar of the type: a numpy scalar left of a term
            # in the callee's body would be made a Python number by numpy.
            return Call(Var(name), (value,))
        kinds = self.kinds(value)
        if kinds is None:
            return Call(Var(name), (value,))
        if all(dtype == want and not weak for dtype, weak in kinds):
            return value
        self.refuse_unconverted(param, sort, value, want, kinds)
        return Call(Var(name), (value,))

    def kinds(self, value: Any) -> set[tuple[np.dtype, bool]] | None:
        """The types the native run may give ``value``, each with whether it is weak.

        ``None`` when one is not known (:mod:`loopty.promotion`).
        """
        from loopty.promotion import Promotion, _kind

        tracer = self.tracer
        promotion = Promotion.of_sorts({**tracer.params, **tracer.array_types})
        native = promotion.types(value)[0]
        if not native:
            return None
        kinds = {_kind(sample) for sample in native}
        return None if None in kinds else kinds  # type: ignore[return-value]

    def literal(self, param: str, sort: Any, value: Any, want: np.dtype) -> None:
        """Refuse a literal the callee's contract refuses natively, or numpy does.

        An integral one outside the integers the sort is stored in, a complex
        one with an imaginary part for a sort that is not complex, a number
        other than ``0`` and ``1`` for ``Bool``
        (:func:`loopty.contract.scalar_parameters`), and one numpy does not
        convert: ``300`` for ``np.int8``. Whether an integral one is a point
        of its sort is a fact of the call (:func:`loopty.typing.call_facts`).
        """
        import warnings

        from loopty.contract import (
            _integral_range_message,
            _scalar_value,
            integral_range,
            integral_sort,
            native_scalar,
        )

        if integral_sort(sort):
            low, high = integral_range(sort)
            if not low <= int(value) < high:
                self.refuse(
                    _integral_range_message(
                        f"its scalar parameter {param}", value, sort, param
                    )
                )
            return
        try:
            _scalar_value(param, sort, value)
        except ValueError as exc:
            self.refuse(f"natively its contract refuses the call: {exc}")
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("error")
                native_scalar(sort, value)
        except (ArithmeticError, ValueError, TypeError, RuntimeWarning) as exc:
            self.refuse(
                f"its scalar parameter {param} is declared {_sort_text(sort)}, and "
                f"it is given {value!r}, which natively its contract converts into "
                f"{_a(want.name)}, and numpy refuses that ({exc}). Pass a value "
                f"{_a(want.name)} holds, or declare {param} a sort that holds "
                f"{value!r}"
            )

    def refuse_unconverted(
        self,
        param: str,
        sort: Any,
        value: Any,
        want: np.dtype,
        kinds: set[tuple[np.dtype, bool]],
    ) -> None:
        """Refuse a value of the caller's that the callee's contract may refuse.

        See :meth:`converted`. ``kinds`` are the types the native run may give
        ``value``.
        """
        from loopty.contract import integral_sort, truth_sort

        letters = {dtype.kind for dtype, _weak in kinds}
        shown = _shown(value)
        given = (
            f"its scalar parameter {param} is declared {_sort_text(sort)}, and it "
            f"is given {shown}, {_typed(kinds)}"
        )
        cannot = "which the compiled kernel cannot ask of a value it computes"
        if integral_sort(sort):
            if letters & set("fc"):
                self.refuse(
                    f"{given}, which is not an integer: natively the contract of "
                    f"{self.callee} refuses it at every call, since a value of an "
                    f"integral sort is passed as an integer. Pass an integer, or "
                    f"declare {param} Real"
                )
            if "b" in letters:
                self.refuse(
                    f"{given}, a truth value, which natively the contract of "
                    f"{self.callee} refuses for an integral sort at every call. "
                    f"Pass an integer, or declare {param} Bool"
                )
            if any(dtype == np.uint64 for dtype, _weak in kinds):
                self.refuse(
                    f"{given}: natively the contract of {self.callee} refuses one "
                    f"past the int64 a value of {sort} is stored in, {cannot}, "
                    "and would wrap round. Pass a value of a signed integer sort, "
                    "such as an element of an array of Int or Nat"
                )
        elif truth_sort(sort) and letters != {"b"}:
            self.refuse(
                f"{given}: natively the contract of {self.callee} takes a number "
                f"for {_sort_text(sort)} only when it is 0 or 1, {cannot}. Pass a "
                f"truth value, such as {shown} != 0"
            )
        if "c" in letters and want.kind != "c":
            self.refuse(
                f"{given}: natively the contract of {self.callee} takes a complex "
                "value for it only when its imaginary part is zero, "
                f"{cannot}. Pass real({shown}) (loopty.math.real), or declare "
                f"{param} complex"
            )

    def array(
        self,
        param: str,
        typ: ArrType,
        value: Any,
        bound: Mapping[str, Any],
        values: dict[str, Any],
        pending: list[tuple[str, Any, Any, str]],
    ) -> SymArr:
        """The array the callee's body is given for ``param``, its shape unified."""
        from loopty.table import Table, passed_message

        if isinstance(value, Table):
            self.refuse(passed_message(param, value))
        if not isinstance(value, SymArr):
            self.refuse(
                f"its {param} is an array, and it is given {_shown(value)}; pass "
                f"an array of {self.caller}, a row of one (A[i]), or the cells an "
                "array of indices picks (u[cell[c]])"
            )
        given = value.type
        if typ.domain is not None or given.domain is not None:
            return self.over_a_domain(param, typ, value, pending)
        if len(typ.axes) != len(given.axes):
            self.refuse(
                f"its {param} has {len(typ.axes)} axes, and {value.name}, which it "
                f"is given there, has {len(given.axes)}"
            )
        if any(typ.ragged):
            self.ragged(param, typ, value, bound)
        elif any(given.ragged):
            self.refuse(
                f"its {param} is dense, and it is given {value.name}, a ragged "
                f"array; pass a row of it, {value.name}[r], or declare {param} "
                "ragged"
            )
        for k, (mine, theirs, ragged) in enumerate(
            zip(typ.axes, given.axes, typ.ragged, strict=True)
        ):
            if not ragged:
                pending.append((f"axis {k} of its {param}", mine, theirs, value.name))
        mine_bound = _fin_bound(typ.dtype)
        theirs_bound = _fin_bound(given.dtype)
        if mine_bound is not None and theirs_bound is not None:
            pending.append(
                (
                    f"the sort of the elements of its {param}",
                    mine_bound,
                    theirs_bound,
                    value.name,
                )
            )
        elif not _same_sort(typ.dtype, given.dtype):
            self.refuse(
                f"it declares the elements of its {param} {_sort_text(typ.dtype)}, "
                f"and the elements of {value.name}, which it is given there, are "
                f"{_sort_text(given.dtype)}; the two have to be one sort"
            )
        return value

    def over_a_domain(
        self,
        param: str,
        typ: ArrType,
        value: SymArr,
        pending: list[tuple[str, Any, Any, str]],
    ) -> SymArr:
        """An array over a domain, passed whole: its sizes unified, its points one.

        The callee's binders are read as the caller's, axis by axis, so that
        each bound settles a size of the callee; whether the two domains are
        one set of points is asked once every size is settled
        (:meth:`settle_all`).
        """
        from loopty.domain import Union

        mine, theirs = typ.domain, value.type.domain
        if (
            mine is None
            or theirs is None
            or isinstance(value, _RowView | _GatherView)
            or type(mine) is not type(theirs)
            or mine.ndim != theirs.ndim
            or (isinstance(mine, Union) and len(mine.pieces) != len(theirs.pieces))
        ):
            self.refuse(
                f"its {param} is {_shape_text(typ)}, and it is given "
                f"{value.name}, {_shape_text(value.type)}; an array over a "
                "domain is passed whole, to a parameter over the same domain"
            )
        pairs = (
            list(zip(mine.pieces, theirs.pieces, strict=True))
            if isinstance(mine, Union)
            else [(mine, theirs)]
        )
        for first, second in pairs:
            renamed = dict(zip(first.names, second.names, strict=True))
            for k, ((_binder, bound), (_other, other)) in enumerate(
                zip(first.binders, second.binders, strict=True)
            ):
                pending.append(
                    (
                        f"the bound of axis {k} of the domain of its {param}",
                        rename_expr(bound, renamed, {}),
                        other,
                        value.name,
                    )
                )
        self.domains.append((param, mine, theirs, value.name))
        mine_bound = _fin_bound(typ.dtype)
        theirs_bound = _fin_bound(value.type.dtype)
        if mine_bound is not None and theirs_bound is not None:
            # Fin[n] against Fin[m]: equal once the domain's sizes are settled
            pending.append(
                (
                    f"the sort of the elements of its {param}",
                    mine_bound,
                    theirs_bound,
                    value.name,
                )
            )
        elif not _same_sort(typ.dtype, value.type.dtype):
            self.refuse(
                f"it declares the elements of its {param} {_sort_text(typ.dtype)}, "
                f"and the elements of {value.name}, which it is given there, are "
                f"{_sort_text(value.type.dtype)}; the two have to be one sort"
            )
        return value

    def ragged(
        self, param: str, typ: ArrType, value: SymArr, bound: Mapping[str, Any]
    ) -> None:
        """Check a ragged argument: the caller's ragged array, counted alike."""
        given = value.type
        if isinstance(value, _RowView | _GatherView) or given.ragged != typ.ragged:
            self.refuse(
                f"its {param} is a ragged array, and it is given {value.name}; pass "
                "the caller's ragged array itself, and its counts for the counts "
                "it names"
            )
        counts = typ.axes[1].name
        passed = bound.get(counts)
        mine = getattr(passed, "name", None)
        theirs = getattr(given.axes[1], "name", None)
        if not isinstance(passed, SymArr) or isinstance(passed, _RowView | _GatherView):
            self.refuse(
                f"the rows of its {param} are counted by its {counts}, and it is "
                f"given {_shown(passed)} for {counts}; pass the array that counts "
                f"the rows of {value.name}, {theirs}"
            )
        if mine != theirs:
            self.refuse(
                f"the rows of its {param} are counted by its {counts}, which it is "
                f"given {mine} for, and the rows of {value.name} are counted by "
                f"{theirs}"
            )
        callee_offsets = self.kernel.term.offsets_of(counts)
        caller_offsets = self.caller_offsets(theirs)
        given_offsets = (
            getattr(bound.get(callee_offsets), "name", None) if callee_offsets else None
        )
        if given_offsets != caller_offsets:
            self.refuse(
                f"it reads the rows of its {param} through "
                f"{given_offsets or 'their own offsets'}, and {self.caller} reads "
                f"the rows of {value.name} through "
                f"{caller_offsets or 'their own offsets'}; the compiled kernel "
                "reads them one way, so pass the offsets the caller declares, or "
                "declare none in either"
            )

    def caller_offsets(self, counts: str) -> str | None:
        """The offsets the caller's rows counted by ``counts`` are read through."""
        from loopty.term import declared_offsets

        return declared_offsets(self.tracer.params.items(), counts)

    def settle_all(
        self, pending: list[tuple[str, Any, Any, str]], values: dict[str, Any]
    ) -> None:
        """Solve the callee's sizes from the axes of what it was given.

        Each item is an equation between an axis of the callee's declared type,
        in its sizes, and the length of that axis of the argument, in the
        caller's names. A bare size is solved by it, ``n + 1`` too (``n`` is
        the length less one), and an axis whose sizes are all known has to be
        equal to it, as isl sees it, or the call is refused naming both.
        """
        sizes = self.sizes()
        while pending:
            waiting = []
            for item in pending:
                outcome = self.settle(item, values, sizes)
                if outcome == "wait":
                    waiting.append(item)
            if len(waiting) == len(pending):
                what, mine, _theirs, name = waiting[0]
                self.refuse(
                    f"{what} is {render(mine)} long, and nothing it is given says "
                    f"what {', '.join(sorted(_unknown(mine, values, sizes)))} is "
                    f"there; {name} has to fix it"
                )
            pending[:] = waiting
        from loopty.compose import _rename_domain, _same_domain

        for param, mine, theirs, name in self.domains:
            if not _same_domain(_rename_domain(mine, {}, values), theirs):
                self.refuse(
                    f"its {param} is over {mine}, and {name}, which it is given "
                    f"there, is over {theirs}; nothing says the two have the same "
                    "points"
                )

    def sizes(self) -> set[str]:
        """The callee's own sizes, which a call solves."""
        term = self.kernel.term
        out = set(term.sizes)
        for name, typ in term.params:
            sort = typ.dtype if isinstance(typ, ArrType) else typ
            bound = _fin_bound(sort)
            if bound is not None:
                out |= set(_names_in(bound))
            if isinstance(typ, ArrType):
                for axis, ragged in zip(typ.axes, typ.ragged, strict=True):
                    if not ragged:
                        out |= set(_names_in(axis))
        return out - set(self.kernel.arg_types)

    def settle(
        self,
        item: tuple[str, Any, Any, str],
        values: dict[str, Any],
        sizes: set[str],
    ) -> str:
        """``"ok"`` when an axis equation is settled, ``"wait"`` when it has to."""
        what, mine, theirs, name = item
        expr = _tidy(rename_expr(mine, {}, values))
        loose = _unknown(expr, values, sizes)
        if not loose:
            if _same(expr, theirs):
                return "ok"
            self.refuse(
                f"{what} is {render(expr)} long there, and {name}, which it is "
                f"given, has {_shown(theirs)}; nothing says the two agree"
            )
        if isinstance(expr, prim.Variable):
            values[expr.name] = theirs
            return "ok"
        shifted = _shifted(expr)
        if shifted is not None:
            size, constant = shifted
            values[size] = _plus(theirs, -constant)
            return "ok"
        return "wait"

    def in_caller(self, sort: Any, values: Mapping[str, Any]) -> Any:
        """A sort whose bound names the callee's sizes, in the caller's names."""
        import dataclasses

        if isinstance(sort, FinType):
            return dataclasses.replace(
                sort, bound=_tidy(rename_expr(sort.bound, {}, values))
            )
        if isinstance(sort, Refined):
            return dataclasses.replace(
                sort,
                base=self.in_caller(sort.base, values),
                props=tuple(rename_expr(p, {}, values) for p in sort.props),
            )
        return sort

    def array_argument(self, param: str, value: SymArr, written: bool) -> Argument:
        """What the call passed for an array parameter, as the facts read it."""
        if isinstance(value, _RowView) and isinstance(value.base, _GatherView):
            gather = value.base
            depth = gather.index.ndim
            picked = gather.index._read(tuple(value.prefix[:depth]), self.where)
            rest = ", ".join(_shown(part) for part in value.prefix[depth:])
            row = f"{gather.source.name}[{_shown(picked)}{', ' + rest if rest else ''}]"
            self.refuse(
                f"its {param} is given {value.name}, a part of the cells "
                f"{gather.name} picks, which natively is a copy of them; pass "
                f"{row}, the row of {gather.source.name} the indices pick"
            )
        if isinstance(value, _RowView):
            return Argument(
                param,
                "row",
                value.name,
                array=value.base.name,
                prefix=value.prefix,
                written=written,
            )
        if isinstance(value, _GatherView):
            source, index = value.source, value.index
            if isinstance(source, _RowView | _GatherView):
                self.refuse(
                    f"its {param} is given {value.name}, the cells a row of "
                    f"{source.name} picks; pass the array and the indices"
                )
            if isinstance(index, _GatherView):
                self.refuse(
                    f"its {param} is given {value.name}, whose indices are "
                    "gathered in turn; gather once, or pass the arrays"
                )
            return Argument(
                param,
                "gather",
                value.name,
                array=source.name,
                index=self.array_argument("", index, False),
                written=written,
            )
        return Argument(param, "array", value.name, array=value.name, written=written)

    def refuse_shared(
        self, arguments: Mapping[str, Argument], written: set[str]
    ) -> None:
        """Refuse two arguments that share storage the way no fact can tell apart.

        Natively the callee's contract refuses two array parameters that share
        storage where it writes either; two it only reads may, ``dot(x, x)``.
        Two rows of one array share none where their indices differ, which is
        a fact of the call where one is written
        (:func:`loopty.typing.call_facts`); a whole array and any part of it
        always do, and are refused here when the callee writes either.
        """
        rows: dict[str, list[Argument]] = {}
        for argument in arguments.values():
            if argument.kind in ("array", "row"):
                rows.setdefault(argument.array, []).append(argument)
        for array, sharing in rows.items():
            whole = [a for a in sharing if a.kind == "array"]
            writing = [a for a in sharing if a.param in written]
            if not whole or not writing or len(sharing) < 2:
                continue
            first = writing[0]
            second = next(a for a in sharing if a is not first)
            self.refuse(
                f"it is given {first.spelled} for {first.param}, which it writes, "
                f"and {second.spelled} for {second.param}, which share the cells "
                f"of {array}: two array parameters of a kernel may not share "
                "storage where it writes either, which its contract checks at "
                "every call"
            )

    def refuse_by_name(
        self, arguments: Mapping[str, Argument], written: set[str]
    ) -> None:
        """Refuse an argument whose value reads an array the callee writes.

        Natively an argument is evaluated once, when the call is made: a
        scalar is the value ``A[i, 0]`` has then, a row ``A[p[i]]`` is the
        row ``p[i]`` names then, and the cells a gather picks are copied
        then. Inlined, the expression is put where the callee uses the
        parameter and read again there, after whatever the callee wrote
        before it. Where the callee writes an array the expression reads,
        the two runs part, so the call is refused, naming both.
        """
        writes = {
            argument.array: argument
            for argument in arguments.values()
            if argument.param in written and argument.kind in ("array", "row")
        }
        if not writes:
            return
        for argument in arguments.values():
            for array in sorted(_read_when_passed(argument)):
                other = writes.get(array)
                if other is None:
                    continue
                if argument.kind == "function":
                    held = _shown(
                        next(v for v in argument.value if array in _held_reads(v))
                    )
                    self.refuse(
                        f"it is given {argument.spelled} for {argument.param}, which "
                        f"holds {held}, a value it captured that reads {array}, and "
                        f"it writes {array} through its {other.param} "
                        f"({other.spelled}): natively the function holds the value "
                        f"{held} had when it was made, and inlined {held} is read "
                        f"again wherever {self.callee} calls {argument.param}, after "
                        f"its own writes into {array}. Capture something that does "
                        f"not read {array}, or read {array} inside the function, "
                        f"where {self.callee} calls it"
                    )
                if argument.kind == "scalar":
                    natively = (
                        f"natively {argument.param} is the value "
                        f"{argument.spelled} has when the call is made"
                    )
                elif argument.kind == "gather":
                    natively = (
                        f"natively {argument.spelled} is a copy of the cells, "
                        "taken when the call is made"
                    )
                else:
                    natively = (
                        f"natively {argument.spelled} is the row its indices "
                        "name when the call is made"
                    )
                self.refuse(
                    f"it is given {argument.spelled} for {argument.param}, which "
                    f"reads {array}, and it writes {array} through its "
                    f"{other.param} ({other.spelled}): {natively}, and inlined "
                    f"{argument.spelled} is read again wherever {self.callee} "
                    f"uses {argument.param}, after its own writes into {array}. "
                    f"For {argument.param}, pass something that does not read "
                    f"{array}, or have {self.callee} read {array} itself"
                )

    # }}}

    # {{{ the postcondition, after the call

    def postcondition(
        self, term: Any, proxies: Mapping[str, Any], values: Mapping[str, Any]
    ) -> Any:
        """The callee's postcondition in the caller's names, or ``None``.

        Its parameters are what the call passed (a row's cells are read
        through the row, ``Me[j, k]`` as ``M[c, j, k]``), and its sizes what
        they are where the call is made. A claim that names an array whole,
        or a function the call passed, says nothing the caller's names can,
        and is not restated.
        """
        if term.post is None:
            return None
        arrays = {
            name: proxy for name, proxy in proxies.items() if isinstance(proxy, SymArr)
        }
        functions = {
            name
            for name, sort in self.kernel.arg_types.items()
            if not isinstance(sort, ArrType) and _function_sort(sort)
        }
        scalars = {
            name: proxy
            for name, proxy in proxies.items()
            if name not in arrays and name not in functions
        }
        mapping = {**values, **scalars}
        # Every name of the caller's a restated claim may mention: its loops
        # and binders, its parameters and sizes, and the indices of the views
        # passed, M[j] for a scalar j, which a binder j of the claim would
        # capture.
        tracer = self.tracer
        avoid = (
            set(tracer._inames)
            | set(tracer.binders)
            | set(tracer.params)
            | set(tracer.integers)
            | set(tracer.array_types)
            | set(tracer.tables)
        )
        for value in mapping.values():
            avoid |= set(_names_in(value))
        for proxy in arrays.values():
            avoid |= _view_names(proxy)
        try:
            restated = _restated(
                term.post, arrays, mapping, functions, avoid, self.where
            )
        except _Refused:
            return None
        return exact(restated, tracer, truth=True)

    # }}}


# {{{ a rule's value under a guard


class Guarded(SymbolicMixin, prim.If):
    """The value of a rule called inside ``with when(...)``.

    Natively a kernel called under a false guard is not made, and a rule's
    value there is ``0`` (:meth:`loopty.kernel.Kernel.__call__`), which
    nothing inside the block keeps, since a write there is masked. Traced,
    the call is inlined wherever the guard holds or not, so its value is
    this: ``condition`` the guards open at the call, ``then`` the call's
    value and ``else_`` that ``0``. Inside the block the value is the call's
    (:func:`unguarded`); a ``with`` block is no scope in Python, though, and
    a use after it would be the call's value compiled and ``0`` natively
    where the guard is false, so it is refused.
    """


def unguarded(expr: Any, tracer: Any, where: str) -> Any:
    """``expr`` with each rule value made in an open ``when`` block as the call's.

    A rule's value made in a block that has closed is refused, naming the
    call and the fix (:class:`Guarded`).
    """
    if tracer is None or not tracer.guarded or not _has_guarded(expr):
        return expr
    return _unguarded(expr, tracer, where)


def _has_guarded(expr: Any) -> bool:
    """Whether ``expr`` holds a :class:`Guarded` value anywhere."""
    if isinstance(expr, Guarded):
        return True
    if isinstance(expr, tuple | list):
        return any(_has_guarded(item) for item in expr)
    if hasattr(expr, "binders") and isinstance(expr, prim.ExpressionNode):
        return _has_guarded(expr.body) or _has_guarded(getattr(expr, "guard", None))
    if isinstance(expr, prim.ExpressionNode):
        return any(_has_guarded(arg) for arg in init_args(expr))
    return False


def _unguarded(node: Any, tracer: Any, where: str) -> Any:
    """See :func:`unguarded`."""
    if isinstance(node, Guarded):
        if not _open(node.condition, tracer.guards):
            entry = tracer.guarded.get(id(node))
            call = entry[1] if entry is not None and entry[0] is node else "a rule"
            raise TraceError(
                f"the value of {call}, called inside 'with when("
                f"{_shown(node.condition)}):', is used at {where}, after that "
                "block: natively a call under a false guard is not made, and "
                "its value is 0 there, while the compiled kernel computes the "
                "call wherever its value is used, so the two runs would part. "
                "Use the value inside the block, or make the call outside it"
            )
        return _unguarded(node.then, tracer, where)
    if not _has_guarded(node):
        return node
    if isinstance(node, tuple | list):
        return type(node)(_unguarded(item, tracer, where) for item in node)
    if hasattr(node, "binders") and isinstance(node, prim.ExpressionNode):
        guard = getattr(node, "guard", None)
        return type(node)(
            node.binders,
            _unguarded(node.body, tracer, where),
            None if guard is None else _unguarded(guard, tracer, where),
        )
    return type(node)(*(_unguarded(arg, tracer, where) for arg in init_args(node)))


def _open(condition: Any, guards: list[Any]) -> bool:
    """Whether the guards ``condition`` conjoins are all open, by identity."""
    if any(condition is guard for guard in guards):
        return True
    return isinstance(condition, prim.LogicalAnd) and all(
        any(part is guard for guard in guards) for part in condition.children
    )


# }}}


# {{{ the conversions an index or a comparison does not need


#: The integer arithmetic through which :func:`exact` leaves out a conversion
#: into int64.
_INTEGER_ARITHMETIC = (prim.Sum, prim.Product, prim.FloorDiv, prim.Remainder)

_INT64 = np.dtype(np.int64)


def _conversion(node: Any, name: str | None = None) -> bool:
    """Whether ``node`` is a scalar's conversion into its sort, ``np.int64(j)``."""
    return _conversion_dtype(node) is not None and (
        name is None or node.function.name == name
    )


def _has_conversion(expr: Any) -> bool:
    """Whether ``expr`` holds a conversion anywhere, the indices of its reads too."""
    if _conversion(expr):
        return True
    if isinstance(expr, tuple | list):
        return any(_has_conversion(item) for item in expr)
    if hasattr(expr, "binders") and isinstance(expr, prim.ExpressionNode):
        return _has_conversion(expr.body) or _has_conversion(
            getattr(expr, "guard", None)
        )
    if isinstance(expr, prim.ExpressionNode):
        return any(_has_conversion(arg) for arg in init_args(expr))
    return False


def exact(expr: Any, tracer: Any, *, truth: bool = False) -> Any:
    """``expr`` without the conversions an index or a comparison of it does not need.

    A scalar a called kernel is given is converted into its sort's storage
    (:meth:`_Inlining.converted`), so a loop variable ``j`` given for
    ``Fin[m]`` is ``np.int64(j)`` in the callee's body. Where an index reads
    it, or a comparison, and every other value there is an integer numpy
    computes in 64 bits or a Python int, the two hold the same number, and
    the conversion is left out: ``A[i, j]`` and ``j < n`` are read by isl as
    the caller's own index and guard are. So is a truth value's conversion
    where a guard reads it (``truth``), ``np.bool(i < 2)``. Beside a value of
    another type, ``x8[k]`` of an ``int8`` array, a Python int is computed in
    eight bits, and the conversion stays; so it does in a value.
    """
    if tracer is None or not _has_conversion(expr):
        return expr
    from loopty.promotion import Promotion

    promotion = Promotion.of_sorts({**tracer.params, **tracer.array_types})
    return _exact(expr, promotion, truth)


def exact_key(key: Any, tracer: Any) -> Any:
    """A subscript key, or a fiber's, with :func:`exact` of each index in it."""
    if tracer is None or not _has_conversion(key):
        return key
    from loopty.promotion import Promotion

    promotion = Promotion.of_sorts({**tracer.params, **tracer.array_types})
    if isinstance(key, tuple):
        return tuple(_exact_index(part, promotion) for part in key)
    return _exact_index(key, promotion)


def _exact_index(part: Any, promotion: Any) -> Any:
    """One index of a subscript, with the conversions it does not need left out."""
    if not isinstance(part, prim.ExpressionNode):
        return part
    return _exact_unit(_exact(part, promotion, False), promotion)


def _exact(node: Any, promotion: Any, truth: bool) -> Any:
    """See :func:`exact`; ``truth`` says whether a guard reads ``node``."""
    if not _has_conversion(node):
        return node
    if isinstance(node, tuple | list):
        return type(node)(_exact(item, promotion, truth) for item in node)
    if truth and _conversion(node, "np.bool"):
        (operand,) = node.parameters
        kinds = promotion.types(operand)[0]
        if kinds and all(isinstance(k, bool | np.bool_) for k in kinds):
            return _exact(operand, promotion, truth)
        return node
    if isinstance(node, prim.Subscript):
        index = node.index if isinstance(node.index, tuple) else (node.index,)
        parts = tuple(_exact_index(part, promotion) for part in index)
        return type(node)(node.aggregate, parts[0] if len(parts) == 1 else parts)
    if isinstance(node, prim.Comparison):
        compared = type(node)(
            _exact(node.left, promotion, False),
            node.operator,
            _exact(node.right, promotion, False),
        )
        return _exact_unit(compared, promotion)
    if isinstance(node, prim.LogicalAnd | prim.LogicalOr):
        return type(node)(tuple(_exact(c, promotion, truth) for c in node.children))
    if isinstance(node, prim.LogicalNot):
        return type(node)(_exact(node.child, promotion, truth))
    if hasattr(node, "binders") and isinstance(node, prim.ExpressionNode):
        guard = getattr(node, "guard", None)
        return type(node)(
            node.binders,
            _exact(node.body, promotion, truth),
            None if guard is None else _exact(guard, promotion, True),
        )
    if isinstance(node, prim.ExpressionNode):
        return type(node)(*(_exact(arg, promotion, False) for arg in init_args(node)))
    return node


def _exact_unit(node: Any, promotion: Any) -> Any:
    """``node``, an index or a comparison, without its conversions into int64.

    Only where every value left in it is an integer the native run computes in
    64 bits, or a Python int: then both hold one number, and the conversion
    changes nothing an index or a comparison reads.
    """
    if not _has_conversion(node):
        return node
    stripped = _without_int64(node)
    leaves = _integer_leaves(stripped)
    for leaf in leaves:
        native = promotion.types(leaf)[0]
        if not native or not all(_int64(sample) for sample in native):
            return node
    return stripped


def _int64(sample: Any) -> bool:
    """Whether a native sample is an int64 or a Python int (not a truth value)."""
    from loopty.promotion import _kind

    if isinstance(sample, bool | np.bool_):
        return False
    kind = _kind(sample)
    return kind is not None and kind[0] == _INT64


def _without_int64(node: Any) -> Any:
    """``node`` with each conversion into int64 its arithmetic reaches left out."""
    if _conversion(node, "np.int64"):
        return _without_int64(node.parameters[0])
    if isinstance(node, prim.Comparison):
        return type(node)(
            _without_int64(node.left), node.operator, _without_int64(node.right)
        )
    if isinstance(node, _INTEGER_ARITHMETIC):
        return type(node)(*(_without_int64(arg) for arg in init_args(node)))
    if isinstance(node, tuple):
        return tuple(_without_int64(item) for item in node)
    return node


def _integer_leaves(node: Any) -> list[Any]:
    """The operands of the integer arithmetic of ``node``, which :func:`exact` asks."""
    if isinstance(node, prim.Comparison):
        return _integer_leaves(node.left) + _integer_leaves(node.right)
    if isinstance(node, _INTEGER_ARITHMETIC):
        out: list[Any] = []
        for arg in init_args(node):
            out.extend(_integer_leaves(arg))
        return out
    if isinstance(node, tuple):
        return [leaf for item in node for leaf in _integer_leaves(item)]
    return [node]


def _typed(kinds: set[tuple[np.dtype, bool]]) -> str:
    """``a float64``, ``a Python int``: the types a native value may have, in words."""
    python = {"b": "bool", "i": "int", "u": "int", "f": "float", "c": "complex"}
    names = sorted(
        {
            f"Python {python[dtype.kind]}" if weak else dtype.name
            for dtype, weak in kinds
        }
    )
    return " or ".join(_a(name) for name in names)


def _a(name: str) -> str:
    """``an int8``, ``a uint64``, ``a Python int``: a type's name with its article."""
    return f"an {name}" if name.startswith(("int", "a", "e", "i", "o")) else f"a {name}"


# }}}


def _restated(
    expr: Any,
    arrays: Mapping[str, SymArr],
    values: Mapping[str, Any],
    functions: set[str],
    avoid: set[str],
    where: str,
) -> Any:
    """``expr`` with the callee's names replaced by the caller's.

    A binder of the claim that is named like a name of the caller's is
    renamed, so that a value put in for a parameter is not captured by it.
    """
    if isinstance(expr, prim.Subscript) and isinstance(expr.aggregate, prim.Variable):
        name = expr.aggregate.name
        index = expr.index if isinstance(expr.index, tuple) else (expr.index,)
        indices = tuple(
            _restated(part, arrays, values, functions, avoid, where) for part in index
        )
        if name in arrays:
            return arrays[name]._read(indices, where)
        if name in functions or name in values:
            raise _Refused(name)
        return type(expr)(expr.aggregate, indices[0] if len(indices) == 1 else indices)
    if isinstance(expr, prim.Variable):
        if expr.name in arrays or expr.name in functions:
            raise _Refused(expr.name)
        return values.get(expr.name, expr)
    if hasattr(expr, "binders") and isinstance(expr, prim.ExpressionNode):
        inner = dict(values)
        binders = []
        for var, domain in expr.binders:
            new_domain = _restated_domain(
                domain, arrays, inner, functions, avoid, where
            )
            name = var.name
            if name in avoid:
                stem, suffix = name, 0
                while f"{stem}_{suffix}" in avoid:
                    suffix += 1
                name = f"{stem}_{suffix}"
            avoid = avoid | {name}
            new_var = Var(name)
            inner[var.name] = new_var
            binders.append((new_var, new_domain))
        body = _restated(expr.body, arrays, inner, functions, avoid, where)
        guard = (
            None
            if expr.guard is None
            else _restated(expr.guard, arrays, inner, functions, avoid, where)
        )
        return type(expr)(tuple(binders), body, guard)
    if isinstance(expr, prim.ExpressionNode):
        args = init_args(expr)
        return type(expr)(
            *(_restated(arg, arrays, values, functions, avoid, where) for arg in args)
        )
    if isinstance(expr, tuple):
        return tuple(
            _restated(item, arrays, values, functions, avoid, where) for item in expr
        )
    return expr


def _restated_domain(
    domain: Any,
    arrays: Mapping[str, SymArr],
    values: Mapping[str, Any],
    functions: set[str],
    avoid: set[str],
    where: str,
) -> Any:
    """A binder's domain in the caller's names: a ``Fin`` with its bound replaced."""
    import dataclasses

    if isinstance(domain, FinType):
        return dataclasses.replace(
            domain,
            bound=_tidy(
                _restated(domain.bound, arrays, values, functions, avoid, where)
            ),
        )
    if isinstance(domain, Refined):
        return dataclasses.replace(
            domain,
            base=_restated_domain(domain.base, arrays, values, functions, avoid, where),
            props=tuple(
                _restated(p, arrays, values, functions, avoid, where)
                for p in domain.props
            ),
        )
    return domain


def _read_when_passed(argument: Argument) -> set[str]:
    """The arrays a native call reads to evaluate ``argument`` as it is made.

    A scalar's value, the indices of a row, for a gather the cells it
    copies and the indices that pick them, and the values a function
    captured when it was made (:func:`_captured`); a whole array, and what a
    function's body reads, are read only where the callee reads them, as they
    are inlined.
    """
    if argument.kind == "scalar":
        return _arrays_read(argument.value)
    if argument.kind == "row":
        return _arrays_read(argument.prefix)
    if argument.kind == "function":
        return set().union(*(_held_reads(value) for value in argument.value))
    if argument.kind == "gather":
        out = {argument.array} if argument.array is not None else set()
        index = argument.index
        if index is not None:
            if index.array is not None:
                out.add(index.array)
            out |= _read_when_passed(index)
        return out
    return set()


def _captured(function: Any, seen: set[int] | None = None) -> tuple[Any, ...]:
    """The values ``function`` captured when it was made, which a call holds.

    A default, ``lambda q, saved=x[0]: ...``, a name of the caller's it
    closes over, ``v = x[0]`` before ``lambda q: v``, and what a
    ``functools.partial`` binds: natively each is the value it had when the
    function was made, and inlined it is the expression that read it, read
    again wherever the callee calls the function. A function captured in
    turn is looked into; a kernel passed as a function captures nothing.
    """
    import functools
    import types

    seen = set() if seen is None else seen
    if id(function) in seen:
        return ()
    seen.add(id(function))
    if isinstance(function, functools.partial):
        values = [*function.args, *function.keywords.values()]
        inner = function.func
    elif isinstance(function, types.FunctionType):
        values = [
            *(function.__defaults__ or ()),
            *(function.__kwdefaults__ or {}).values(),
        ]
        for cell in function.__closure__ or ():
            try:
                values.append(cell.cell_contents)
            except ValueError:  # a name not bound yet
                continue
        inner = None
    else:
        return ()
    out: list[Any] = []
    for value in values:
        if isinstance(value, functools.partial | types.FunctionType):
            out.extend(_captured(value, seen))
        else:
            out.append(value)
    if inner is not None:
        out.extend(_captured(inner, seen))
    return tuple(out)


def _held_reads(value: Any) -> set[str]:
    """The arrays a value a function captured was read from when it was made.

    An expression reads the arrays it reads a cell of; a row, ``A[p[i]]``,
    what its indices read, its cells being read through it later; the cells
    a gather picks, ``u[cell[c]]``, which natively are a copy, the array they
    are copied from and the indices that pick them.
    """
    if isinstance(value, _RowView):
        return _arrays_read(value.prefix) | _held_reads(value.base)
    if isinstance(value, _GatherView):
        return (
            {_root(value.source), _root(value.index)}
            | _held_reads(value.source)
            | _held_reads(value.index)
        )
    if isinstance(value, SymArr | SymDom):
        return set()
    return _arrays_read(value)


def _root(view: SymArr) -> str:
    """The array of the trace a view's cells are in: ``A`` of ``A[i]``."""
    while isinstance(view, _RowView | _GatherView):
        view = view.base if isinstance(view, _RowView) else view.source
    return view.name


def _arrays_read(expr: Any) -> set[str]:
    """The arrays ``expr`` reads a cell of, ``A`` for ``A[i, 0]``, sums included."""
    out: set[str] = set()

    def walk(node: Any) -> None:
        aggregate = getattr(node, "aggregate", None)
        if isinstance(node, prim.Subscript) and isinstance(aggregate, prim.Variable):
            out.add(aggregate.name)
            walk(node.index)
        elif hasattr(node, "binders") and isinstance(node, prim.ExpressionNode):
            walk(node.body)
            walk(getattr(node, "guard", None))
        elif isinstance(node, prim.ExpressionNode):
            for arg in init_args(node):
                walk(arg)
        elif isinstance(node, tuple | list):
            for item in node:
                walk(item)

    walk(expr)
    return out


def _view_names(proxy: Any) -> set[str]:
    """The caller's names the indices of a view mention: ``i`` of ``A[i]``."""
    if isinstance(proxy, _RowView):
        return set(_names_in(proxy.prefix)) | _view_names(proxy.base)
    if isinstance(proxy, _GatherView):
        return _view_names(proxy.index) | _view_names(proxy.source)
    return set()


def _fin_bound(sort: Any) -> Any:
    """The bound of a ``Fin`` sort, a refinement's included, or ``None``."""
    base = sort.base if isinstance(sort, Refined) else sort
    return base.bound if isinstance(base, FinType) else None


def _unknown(expr: Any, values: Mapping[str, Any], sizes: set[str]) -> set[str]:
    """The callee's sizes ``expr`` names that no argument has fixed yet."""
    return {name for name in _names_in(expr) if name in sizes and name not in values}


def _tidy(expr: Any) -> Any:
    """``expr`` with the integers of a sum of names added up (see compose)."""
    from loopty.compose import _tidy as tidy

    return tidy(expr)


def _same_sort(mine: Any, theirs: Any) -> bool:
    """Whether an array of ``theirs`` may stand for a parameter of ``mine``.

    The same sort, or two sorts that are not integral and that both runs
    store alike, ``Real`` and ``np.float64`` say. An integral sort says more
    than its storage (``Nat`` that no cell is negative), which the callee's
    facts rest on, so it has to be the same.
    """
    from loopty.contract import integral_sort, native_storage

    if str(mine) == str(theirs):
        return True
    if integral_sort(mine) or integral_sort(theirs):
        return False
    stored = native_storage(mine)
    return stored is not None and stored == native_storage(theirs)


def _sort_text(sort: Any) -> str:
    """A sort as a message writes it, a numpy scalar type by its dtype's name."""
    if isinstance(sort, type) and issubclass(sort, np.generic):
        return np.dtype(sort).name
    return str(sort)


def _shape_text(typ: ArrType) -> str:
    """``an array over Where[...]``, or ``a dense or ragged array``."""
    if typ.domain is not None:
        return f"an array over {typ.domain}"
    return "a dense or ragged array"
