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
(:class:`loopy.trace._RowView`, :class:`loopty.trace._GatherView`): ``a[k]``
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
two arguments do not share storage. Traced, the shapes are unified when the
call is made, size by size, and a disagreement is refused, naming the axis.
The rest are facts of the caller, about every point the call is made at
(:func:`loopty.typing.call_facts`): each row in bounds, each integral scalar
in its sort, the cells a gather picks in bounds, two rows of one array apart.
isl decides them, and Lean states each from loopty's own description of the
call (:mod:`loopty.obligations`).

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
from lanky.terms import Var, current_trace, init_args, render

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
    _domain_texts,
    _GatherView,
    _location,
    _RowView,
    _shown,
    constraints_of,
)

__all__ = ["inline", "takes_a_function"]


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

    def refuse(self, message: str) -> NoReturn:
        raise TraceError(
            f"{self.caller} calls {self.callee} at {self.where}: {message}"
        )

    # {{{ the call

    def run(self, args: tuple, kwargs: dict) -> Any:
        kernel, tracer = self.kernel, self.tracer
        if any(entry is kernel for _label, entry in tracer.inlining):
            self.refuse(
                f"{self.callee} is already being called there, so the call would "
                "inline it into itself without end; a kernel does not call itself"
            )
        bound = self.bind(args, kwargs)
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
                    param, "function", getattr(value, "__name__", repr(value))
                )
            else:
                unfixed = _unknown(_fin_bound(sort) or 0, values, self.sizes())
                arguments[param] = Argument(
                    param,
                    "scalar",
                    _shown(value),
                    value=value,
                    sort=self.in_caller(sort, values),
                    unfixed=tuple(sorted(unfixed)),
                )
        self.refuse_shared(arguments, written)
        before = len(tracer.stmts)
        tracer.inlining.append((label, kernel))
        try:
            value = kernel.fn(**proxies)
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

    def bind(self, args: tuple, kwargs: dict) -> dict[str, Any]:
        """The call's arguments by the callee's parameter names."""
        code = self.kernel.fn.__code__
        names = code.co_varnames[: code.co_argcount]
        if len(args) > len(names):
            self.refuse(f"it is given {len(args)} arguments; it takes {len(names)}")
        bound = dict(zip(names, args, strict=False))
        for key, value in kwargs.items():
            if key not in names or key in bound:
                self.refuse(
                    f"the argument {key} is given "
                    + ("twice" if key in bound else "and is no parameter of it")
                )
            bound[key] = value
        missing = [name for name in names if name not in bound]
        if missing:
            self.refuse(f"it is not given {', '.join(missing)}")
        return bound

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

        if _function_sort(sort):
            if isinstance(value, prim.Expression | SymArr | SymDom) or not callable(
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
            value, bool | int | float | complex | np.number | np.bool_ | prim.Expression
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
        return value

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
            self.refuse(
                f"its {param} is {_shape_text(typ)}, and it is given "
                f"{value.name}, {_shape_text(given)}; an array over a Where, "
                "Sigma or union domain is not passed to a kernel called inside "
                "another yet, so call the kernel from a program, or write the "
                "loops in the caller"
            )
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
        elif str(typ.dtype) != str(given.dtype):
            self.refuse(
                f"it declares the elements of its {param} {_sort_text(typ.dtype)}, "
                f"and the elements of {value.name}, which it is given there, are "
                f"{_sort_text(given.dtype)}; the two have to be one sort"
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
        storage. Two rows of one array share none where their indices differ,
        which is a fact of the call (:func:`loopty.typing.call_facts`); a
        whole array and any part of it always do, and are refused here. A
        gather is a copy natively, read once when the call is made, so the
        callee may not write the array it gathers from through another
        parameter: the compiled call would read what the write left.
        """
        rows: dict[str, list[Argument]] = {}
        for argument in arguments.values():
            if argument.kind == "array":
                rows.setdefault(argument.array, []).append(argument)
            elif argument.kind == "row":
                rows.setdefault(argument.array, []).append(argument)
        for array, sharing in rows.items():
            whole = [a for a in sharing if a.kind == "array"]
            if whole and len(sharing) > 1:
                first, second = sharing[0], sharing[1]
                self.refuse(
                    f"it is given {first.spelled} for {first.param} and "
                    f"{second.spelled} for {second.param}, which share the cells "
                    f"of {array}, and two array parameters of a kernel may not "
                    "share storage, which its contract checks at every call"
                )
        for argument in arguments.values():
            if argument.kind != "gather":
                continue
            for other in arguments.values():
                if (
                    other.array == argument.array
                    and other.kind != "gather"
                    and (other.param in written)
                ):
                    self.refuse(
                        f"it is given {argument.spelled} for {argument.param} and "
                        f"{other.spelled} for {other.param}, which it writes. "
                        f"Natively {argument.spelled} is a copy, taken when the "
                        "call is made, and compiled it is read where the callee "
                        "reads it, after the writes; pass a copy of the cells "
                        "in an array of the caller's"
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
        avoid = set(self.tracer._inames) | set(self.tracer.binders)
        for value in mapping.values():
            avoid |= set(_names_in(value))
        try:
            return _restated(term.post, arrays, mapping, functions, avoid, self.where)
        except _Refused:
            return None

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
