"""A program's term: its kernel calls, composed in call order.

A :class:`~loopty.kernel.Program` runs natively: its body calls kernels, and
they run natively inside it. Its term is what the body does when it runs once
more against placeholders, as a kernel's term is what its body does against
symbolic arrays. Every kernel call is recorded instead of run, with what was
passed to it, and the callees' terms follow one another in call order, each in
the program's names. Nothing is parsed.

That is the sequential program: one term, which lowers into one loopy kernel
whose statements run in call order. An edge between two kernels is not
declared. An array one call writes and a later call reads is one array of the
program's term, so the dependence is in the footprints, exactly as between two
statements of one kernel: the lowering orders the instructions by it, and a
:class:`~loopty.schedule.Schedule` of the program has it among the dependences
every cast is checked against. Fusion is not here. It is a cast over this
term, and it waits for facts that travel from one kernel to the next.

Three things are decided in composing.

*Names are the program's.* A callee's array and scalar parameters become what
the program passed: a parameter of the program, an array it made, or, for a
scalar, a Python number, which is substituted. Its sizes are unified through
the arrays. A program array has one type, taken from the first call that
passes it, and every later call has to agree with it, a size of the callee
standing for whatever the program's size is there: ``scale``'s ``n`` is
``scan``'s ``n + 1`` when both are handed ``off``. Two sizes that cannot be
shown to agree are refused. The callee's loop variables, reduction binders and
reflected parameters get names no earlier call has, the way the tracer names a
second ``for r`` ``r_0``, so that agreeing on a name is agreeing on a loop. Its
statements are named after the call, ``scan.S1`` and ``spmv.S0``, and
``step@2.S0`` in the second call of ``step``, and keep their own ``file:line``.

*An array the program makes is a temporary.* ``f = Arr.zeros_like(u)`` in the
body makes an array natively and a placeholder under tracing. It becomes one
of the term's :attr:`~loopty.term.Term.temporaries`, typed by the kernels it is
passed to and shaped like ``u``, and zeroed by a statement of its own where the
body made it, because ``zeros_like`` is zeros. The lowering declares it as a
loopy temporary rather than an argument, so nobody passes it and nothing
outside the kernel sees it.

*A parameter has one role.* It is an output of the program's term when any
call writes it, which the lowering reads off the statements, and an input
otherwise. The two declarations that independent lowerings disagree about
(``f`` an output of the producer and input-only in the consumer) cannot arise,
because there is one lowering. The layout of a ragged parameter is stated on
the term (:attr:`~loopty.term.Term.offsets`): the offsets each call's kernel
reads its rows through, which the calls have to agree on.

What a program's body may do is pass things to kernels, make arrays with
``Arr.zeros_like``, and run Python that touches neither. Reading or writing a
placeholder, asking it for its domain, computing with it or branching on it is
refused with a :class:`~loopty.trace.TraceError` naming the fix, and so is a
loop whose trip count is a parameter: that is a host loop, which a term does
not have. A ``for`` over ``range(3)`` runs three times and records three calls,
which is what it does natively.

The term has no postcondition. Each callee's stays its own, and
:meth:`loopty.kernel.Program.facts` restates it in the program's scope.
"""

from __future__ import annotations

import dataclasses
import dis
import inspect
import os
import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any, NoReturn

import islpy as isl
import numpy as np
import pymbolic.primitives as prim
from lanky.prelude import FinType, Refined
from lanky.terms import init_args, render, structurally_equal

from loopty.flow import NonAffine, domain_set, expr_text, free_names
from loopty.term import Access, ArrType, Reduction, Stmt, Term
from loopty.trace import TraceError

__all__ = [
    "ProgramValue",
    "current_recorder",
    "rename_expr",
    "rename_set",
    "trace_program",
]


# {{{ renaming a term's names


def rename_expr(expr: Any, names: Mapping[str, str], exprs: Mapping[str, Any]) -> Any:
    """``expr`` with every name in ``names`` renamed and every one in ``exprs``
    replaced by the expression it maps to, all at once.

    Arrays are renamed where they are read or written (an
    :class:`~loopty.term.Access`, the aggregate of a subscript), a reduction's
    binders and its domain follow (:func:`rename_set`), and a node that changes
    nothing is returned as it is. Nodes are rebuilt from their constructor
    arguments, so lanky's subclasses stay lanky's.
    """
    if isinstance(expr, Access):
        return Access(
            names.get(expr.array, expr.array),
            tuple(rename_expr(index, names, exprs) for index in expr.indices),
        )
    if isinstance(expr, Reduction):
        return Reduction(
            op=expr.op,
            inames=tuple(names.get(name, name) for name in expr.inames),
            domain=rename_set(expr.domain, names, exprs),
            body=rename_expr(expr.body, names, exprs),
            exactness=expr.exactness,
        )
    if isinstance(expr, prim.Variable):
        if expr.name in exprs:
            return exprs[expr.name]
        if expr.name in names:
            return type(expr)(names[expr.name])
        return expr
    if isinstance(expr, prim.ExpressionNode):
        args = init_args(expr)
        new = tuple(rename_expr(arg, names, exprs) for arg in args)
        if all(old is changed for old, changed in zip(args, new, strict=True)):
            return expr
        return type(expr)(*new)
    if isinstance(expr, tuple):
        return tuple(rename_expr(item, names, exprs) for item in expr)
    return expr


def rename_set(obj: Any, names: Mapping[str, str], exprs: Mapping[str, Any]) -> Any:
    """An isl set with its dimensions renamed and its parameters replaced.

    The same map as :func:`rename_expr`, over a statement's or a reduction's
    domain. A name is renamed in place. A parameter mapped to an expression, or
    to a name the set already has as a parameter, is replaced: the set is
    intersected with ``p = expression`` and ``p`` is projected out, which is
    exact, since an equality determines it. Every name is first moved out of
    the way, so that ``a -> b`` beside ``b -> c`` does not give the set two
    dimensions called ``b``.
    """
    moved: list[tuple[str, Any, str]] = []
    for kind in (isl.dim_type.param, isl.dim_type.set):
        for position in range(obj.dim(kind)):
            original = obj.get_dim_name(kind, position)
            if original in names or original in exprs:
                placeholder = f"__loopty_moved_{len(moved)}"
                obj = obj.set_dim_name(kind, position, placeholder)
                moved.append((placeholder, kind, original))
    replaced: list[tuple[str, Any]] = []
    for placeholder, kind, original in moved:
        if original not in names:
            replaced.append((placeholder, exprs[original]))
            continue
        target = names[original]
        present = set(obj.get_var_names(isl.dim_type.param))
        present |= set(obj.get_var_names(isl.dim_type.set))
        if target in present and kind == isl.dim_type.param:
            replaced.append((placeholder, prim.Variable(target)))
            continue
        position = obj.find_dim_by_name(kind, placeholder)
        obj = obj.set_dim_name(kind, position, target)
    for placeholder, value in replaced:
        obj = _replace_param(obj, placeholder, value)
    return obj


def _isl_text(value: Any) -> str:
    """``value`` in isl's syntax, or a refusal: a domain holds integers only."""
    if isinstance(value, bool | np.bool_):
        return "1" if value else "0"
    if isinstance(value, int | np.integer):
        return str(int(value))
    if isinstance(value, float | np.floating):
        if float(value).is_integer():
            return str(int(value))
        raise TraceError(
            f"the number {value} would have to stand in an iteration domain, "
            "which holds integers only; pass an integer, or pass it as a "
            "parameter of the program"
        )
    try:
        return expr_text(value)
    except NonAffine as exc:
        raise TraceError(
            f"{render(value)} would have to stand in an iteration domain, and "
            "isl can state only an affine expression of integers there"
        ) from exc


def _replace_param(obj: Any, name: str, value: Any) -> Any:
    """``obj`` with its parameter ``name`` replaced by ``value``."""
    text = _isl_text(value)
    params = [name, *sorted(free_names(text) - {name})]
    kind = isl.BasicSet if isinstance(obj, isl.BasicSet) else isl.Set
    constraint = kind(f"[{', '.join(params)}] -> {{ : {name} = {text} }}")
    obj = obj.align_params(constraint.get_space())
    constraint = constraint.align_params(obj.get_space())
    obj = obj.intersect_params(constraint)
    position = obj.find_dim_by_name(isl.dim_type.param, name)
    return obj.project_out(isl.dim_type.param, position, 1)


def _rename_sort(sort: Any, names: Mapping[str, str], exprs: Mapping[str, Any]) -> Any:
    """A sort whose bound or refinement names sizes, in the new names.

    ``Fin[m]`` is the element sort of a column-index array and the sort of an
    index argument, and its bound is a size of the kernel. Any other sort is
    returned as it is.
    """
    if isinstance(sort, FinType):
        return dataclasses.replace(sort, bound=rename_expr(sort.bound, names, exprs))
    if isinstance(sort, Refined) and dataclasses.is_dataclass(sort):
        return dataclasses.replace(
            sort,
            base=_rename_sort(sort.base, names, exprs),
            props=tuple(rename_expr(prop, names, exprs) for prop in sort.props),
        )
    return sort


def _rename_type(typ: Any, names: Mapping[str, str], exprs: Mapping[str, Any]) -> Any:
    """An array type or a sort, in the new names."""
    if isinstance(typ, ArrType):
        return ArrType(
            axes=tuple(rename_expr(axis, names, exprs) for axis in typ.axes),
            dtype=_rename_sort(typ.dtype, names, exprs),
            ragged=typ.ragged,
        )
    return _rename_sort(typ, names, exprs)


def _names_in(expr: Any) -> list[str]:
    """The variable names in an expression, in the order they first occur."""
    out: list[str] = []

    def visit(node: Any) -> None:
        if isinstance(node, prim.Variable):
            if node.name not in out:
                out.append(node.name)
        elif isinstance(node, prim.ExpressionNode):
            for arg in init_args(node):
                visit(arg)
        elif isinstance(node, tuple | list):
            for item in node:
                visit(item)

    visit(expr)
    return out


def _same(left: Any, right: Any) -> bool:
    """Are two size expressions equal for every value of their sizes?

    Structurally equal, or affine and equal as isl sees it, so that ``n + 1``
    and ``1 + n`` agree. Anything else is not known to be equal.
    """
    if structurally_equal(left, right):
        return True
    try:
        first, second = expr_text(left), expr_text(right)
    except NonAffine:
        return False
    params = sorted(free_names(first) | free_names(second))
    head = f"[{', '.join(params)}] -> " if params else ""
    differ = isl.Set(f"{head}{{ : {first} < {second} or {first} > {second} }}")
    return bool(differ.is_empty())


def _shown(value: Any) -> str:
    """A size or a sort as a message prints it."""
    if isinstance(value, prim.ExpressionNode):
        return render(value)
    return str(value)


# }}}


# {{{ the placeholders and the recorder


def _where(frame: Any) -> str:
    """``file:line`` of a frame."""
    return f"{os.path.basename(frame.f_code.co_filename)}:{frame.f_lineno}"


def _stored_name(frame: Any) -> str | None:
    """The name the value a call returns is stored to, when it is stored at once.

    ``f = Arr.zeros_like(u)`` stores into ``f`` with the instruction right after
    the call, and that is the name the temporary gets, which keeps the
    generated code readable. Anything else, a call inside another call or a
    bytecode layout nobody foresaw, gives ``None``.
    """
    try:
        for instruction in dis.get_instructions(frame.f_code):
            if instruction.offset <= frame.f_lasti:
                continue
            if not instruction.opname.startswith("STORE_"):
                return None
            name = instruction.argval
            if isinstance(name, tuple) and name:
                name = name[0]
            return name if isinstance(name, str) and name.isidentifier() else None
    except Exception:  # noqa: BLE001 - reading bytecode is best effort
        return None
    return None


class ProgramValue:
    """What a program's body holds while its term is built.

    A parameter of the program, or an array the body made with
    :meth:`loopty.arr.Arr.zeros_like`. It can be passed to a kernel and given
    to ``Arr.zeros_like``, and nothing else. The term is the kernels the body
    calls and the arrays it makes, so anything else the body did to an array
    would be missing from it, and the placeholder refuses it with a
    :class:`~loopty.trace.TraceError` rather than answer something the native
    run would not. Equality and hashing are by identity, which is what a
    placeholder is.
    """

    __slots__ = ("_recorder", "like", "name")

    def __init__(
        self, recorder: _Recorder, name: str, like: ProgramValue | None = None
    ) -> None:
        self._recorder = recorder
        self.name = name
        #: What ``Arr.zeros_like`` made this like, for an array the body made;
        #: ``None`` for a parameter.
        self.like = like

    def _loopty_zeros_like(self, frame: Any) -> ProgramValue:
        """``Arr.zeros_like(self)``, under tracing: a new array of the program."""
        return self._recorder.make(self, frame)

    def _refuse(self, what: str) -> NoReturn:
        program = self._recorder.name
        which = (
            f"the array {self.name} it made"
            if self.like is not None
            else f"its parameter {self.name}"
        )
        raise TraceError(
            f"the body of the program {program} {what} {which}. A program's "
            "term is the kernels it calls, in the order it calls them, and the "
            "arrays it makes with Arr.zeros_like(...), so its body may pass its "
            "arguments to kernels and nothing else: whatever else it did to "
            "them would not be in the term, and the compiled program would "
            "compute something the native one does not. Do the work in a "
            "kernel and pass it the arrays; a loop whose trip count is an "
            "argument is a host loop, which a program's term does not have."
        )

    def __getattr__(self, attribute: str) -> Any:
        if attribute.startswith("__") and attribute.endswith("__"):
            raise AttributeError(attribute)
        self._refuse(f"asks for .{attribute} of")

    def __getitem__(self, key: Any) -> Any:
        self._refuse("reads a cell of")

    def __setitem__(self, key: Any, value: Any) -> None:
        self._refuse("writes a cell of")

    def __delitem__(self, key: Any) -> None:
        self._refuse("deletes a cell of")

    def __iter__(self) -> Any:
        self._refuse("iterates over")

    def __contains__(self, item: Any) -> bool:
        self._refuse("searches")

    def __len__(self) -> int:
        self._refuse("asks for the length of")

    def __bool__(self) -> bool:
        self._refuse("branches on")

    def __index__(self) -> int:
        self._refuse("uses as an integer (a loop count, say)")

    def __int__(self) -> int:
        self._refuse("uses as an integer")

    def __float__(self) -> float:
        self._refuse("uses as a number")

    def __complex__(self) -> complex:
        self._refuse("uses as a number")

    def __array__(self, *args: Any, **kwargs: Any) -> Any:
        self._refuse("hands numpy")

    def __array_ufunc__(self, *args: Any, **kwargs: Any) -> Any:
        self._refuse("hands numpy")

    def __array_function__(self, *args: Any, **kwargs: Any) -> Any:
        self._refuse("hands numpy")

    def __repr__(self) -> str:
        return f"<{self.name} of the program {self._recorder.name}>"


def _refusing(what: str) -> Any:
    def method(self: ProgramValue, *_args: Any) -> Any:
        self._refuse(what)

    return method


#: The operators a placeholder refuses: arithmetic in both directions, then
#: the unary ones and the orderings, which have no reflected spelling.
_BINARY = "add sub mul truediv floordiv mod pow matmul and or xor lshift rshift"
_UNREFLECTED = "neg pos abs invert lt le gt ge"

for _operator in _BINARY.split():
    setattr(ProgramValue, f"__{_operator}__", _refusing("computes with"))
    setattr(ProgramValue, f"__r{_operator}__", _refusing("computes with"))
for _operator in _UNREFLECTED.split():
    setattr(ProgramValue, f"__{_operator}__", _refusing("computes with"))


@dataclass
class _Call:
    """One kernel call of the body: the kernel, its arguments by name, where."""

    kernel: Any
    bound: dict[str, Any]
    where: str


@dataclass
class _Made:
    """One ``Arr.zeros_like`` of the body: the array it made, and where."""

    value: ProgramValue
    where: str


class _Recorder:
    """What one run of a program's body against placeholders did, in order."""

    def __init__(self, program: Any, names: Sequence[str]) -> None:
        self.name = program.__name__
        self.where = program.where
        self.params = tuple(ProgramValue(self, name) for name in names)
        self.events: list[_Call | _Made] = []
        self.taken: set[str] = set(names)

    def call(self, kernel: Any, args: tuple, kwargs: dict, frame: Any) -> None:
        """Record a kernel call, by parameter name, instead of running it."""
        code = kernel.fn.__code__
        names = code.co_varnames[: code.co_argcount]
        where = _where(frame)
        if len(args) > len(names):
            raise TraceError(
                f"{self.name} calls {kernel.__name__} at {where} with "
                f"{len(args)} arguments; it takes {len(names)}"
            )
        bound = dict(zip(names, args, strict=False))
        for key, value in kwargs.items():
            if key not in names or key in bound:
                raise TraceError(
                    f"{self.name} calls {kernel.__name__} at {where} with the "
                    f"argument {key} {'twice' if key in bound else 'it does not take'}"
                )
            bound[key] = value
        missing = [name for name in names if name not in bound]
        if missing:
            raise TraceError(
                f"{self.name} calls {kernel.__name__} at {where} without "
                f"{', '.join(missing)}"
            )
        self.events.append(_Call(kernel, bound, where))

    def make(self, like: ProgramValue, frame: Any) -> ProgramValue:
        """Record an array the body makes, named after what it is stored to."""
        stem = _stored_name(frame) or "tmp"
        name = stem
        suffix = 0
        while name in self.taken:
            name = f"{stem}_{suffix}"
            suffix += 1
        self.taken.add(name)
        value = ProgramValue(self, name, like=like)
        self.events.append(_Made(value, _where(frame)))
        return value


_RECORDERS: list[_Recorder] = []


def current_recorder() -> _Recorder | None:
    """The recorder of the program being traced, or ``None``.

    :meth:`loopty.kernel.Kernel.__call__` asks, and records its call here
    rather than running when there is one.
    """
    return _RECORDERS[-1] if _RECORDERS else None


def trace_program(program: Any) -> Term:
    """Run ``program``'s body against placeholders and compose its calls.

    The body gets one :class:`ProgramValue` per parameter. The kernels it calls
    record their calls rather than run, and a program it calls runs its own
    body with the same placeholders, so its calls are recorded in place. The
    recorder is closed before anything is composed, because composing traces
    the callees, and a callee's body calling a kernel is not the program's
    call.
    """
    function = program.fn
    names: list[str] = []
    for parameter in inspect.signature(function).parameters.values():
        if parameter.kind not in (
            inspect.Parameter.POSITIONAL_ONLY,
            inspect.Parameter.POSITIONAL_OR_KEYWORD,
        ):
            raise TraceError(
                f"the program {program.__name__} takes {parameter}, and a "
                "program's term has one argument per named parameter; name "
                "each one"
            )
        names.append(parameter.name)
    recorder = _Recorder(program, names)
    _RECORDERS.append(recorder)
    try:
        function(*recorder.params)
    finally:
        _RECORDERS.pop()
    return _Composer(recorder).term()


# }}}


# {{{ composing


#: What a callee's own size is called while it is unified with the program's.
#: No identifier starts with it, so a callee's ``n`` can never be taken for
#: the program's ``n``.
_CALLEE = "@"


def _is_number(value: Any) -> bool:
    """Whether ``value`` is a Python or numpy number, which a scalar may be."""
    return isinstance(value, int | float | complex | np.number | np.bool_)


def _order(stmt: Stmt) -> tuple[int, ...]:
    """A statement's position in its loop tree, or the one a hand term implies."""
    return tuple(stmt.order) if stmt.order else (0,) * (len(stmt.inames) + 1)


def _own_names(term: Term) -> list[str]:
    """The names a callee's term binds itself, in the order they first occur.

    Loop variables, reduction binders, and the parameters of its sets that are
    neither a parameter of the kernel nor a size nor a reflected parameter
    (which are renamed apart). Each gets a name of its own in the program.
    """
    out: list[str] = []

    def add(name: str) -> None:
        if name not in out:
            out.append(name)

    def reductions(expr: Any) -> list[Reduction]:
        found: list[Reduction] = []

        def visit(node: Any) -> None:
            if isinstance(node, Reduction):
                found.append(node)
                visit(node.body)
            elif isinstance(node, Access):
                visit(node.indices)
            elif isinstance(node, prim.ExpressionNode):
                for arg in init_args(node):
                    visit(arg)
            elif isinstance(node, tuple | list):
                for item in node:
                    visit(item)

        visit(expr)
        return found

    sets: list[Any] = []
    for stmt in term.stmts:
        for iname in stmt.inames:
            add(iname)
        sets.append(stmt.domain)
        if stmt.loop_domain is not None:
            sets.append(stmt.loop_domain)
        for source in (stmt.expr, stmt.guard, stmt.assignee):
            for reduction in reductions(source):
                for iname in reduction.inames:
                    add(iname)
                sets.append(reduction.domain)
    known = set(term.param_names) | set(term.sizes)
    known |= {symbol for symbol, _ in term.reflected}
    for obj in sets:
        for kind in (isl.dim_type.set, isl.dim_type.param):
            for name in obj.get_var_names(kind):
                if name not in known:
                    add(name)
    return out


def _through(offsets: str | None) -> str:
    """How a message says which offsets rows are read through."""
    return "through its own offsets" if offsets is None else f"through {offsets}"


def _reflected_spelling(expr: Any) -> str | None:
    """``nl_cnt_r`` for ``cnt[r]``: the name the tracer would give that bound."""
    if not isinstance(expr, prim.Subscript):
        return None
    if not isinstance(expr.aggregate, prim.Variable):
        return None
    index = expr.index
    if isinstance(index, tuple):
        if len(index) != 1:
            return None
        index = index[0]
    if not isinstance(index, prim.Variable):
        return None
    return f"nl_{expr.aggregate.name}_{index.name}"


class _Composer:
    """The program's term, built one recorded event at a time."""

    def __init__(self, recorder: _Recorder) -> None:
        self.recorder = recorder
        self.program = recorder.name
        self.taken: set[str] = set(recorder.taken)
        #: What each program name is, ``"array"`` or ``"scalar"``, and its
        #: type in program names, with the call that first gave it one.
        self.kinds: dict[str, str] = {}
        self.types: dict[str, Any] = {}
        self.origin: dict[str, str] = {}
        #: The program's own sizes, and the ones unification eliminated, each
        #: with the expression it turned out to be, in the standing sizes.
        self.sizes: list[str] = []
        self.resolved: dict[str, Any] = {}
        #: The offsets each counts family is read through, and who said so.
        self.layout: dict[str, str | None] = {}
        self.layout_origin: dict[str, str] = {}
        self.slots: list[list[Stmt] | tuple[_Made, int]] = []
        self.reflected: list[tuple[str, Any]] = []
        self.made: dict[str, _Made] = {}
        #: How many top-level blocks the statements so far occupy.
        self.blocks = 0
        self.labels: set[str] = set()
        self.calls: dict[str, int] = {}

    # {{{ names

    def fresh(self, stem: str) -> str:
        """A name nothing in the program uses yet, ``stem`` when that is free."""
        name = stem
        suffix = 0
        while name in self.taken:
            name = f"{stem}_{suffix}"
            suffix += 1
        self.taken.add(name)
        return name

    def label(self, kernel: str) -> str:
        """The prefix of one call's statements: ``scan``, then ``scan@2``."""
        count = self.calls.get(kernel, 0)
        while True:
            count += 1
            label = kernel if count == 1 else f"{kernel}@{count}"
            spelled = re.sub(r"\W", "_", label)
            if spelled not in self.labels:
                break
        self.calls[kernel] = count
        self.labels.add(spelled)
        return label

    def resolve(self, expr: Any) -> Any:
        """``expr`` in the sizes still standing."""
        return rename_expr(expr, {}, self.resolved) if self.resolved else expr

    def equate(self, first: Any, second: Any) -> bool:
        """Make two size expressions equal by eliminating a size, if one can go.

        A size of the program standing alone on one side, and not on the
        other, is replaced by the other side everywhere. When both sides are
        such sizes, the younger one goes, so that the name the first call gave
        a size is the name the term keeps. Returns whether a size went.
        """
        candidates = [
            (self.sizes.index(alone.name), alone.name, other)
            for alone, other in ((first, second), (second, first))
            if isinstance(alone, prim.Variable)
            and alone.name in self.sizes
            and alone.name not in _names_in(other)
        ]
        if not candidates:
            return False
        _, size, value = max(candidates, key=lambda candidate: candidate[0])
        self.eliminate(size, value)
        return True

    def eliminate(self, size: str, value: Any) -> None:
        """Replace the program size ``size`` by ``value`` everywhere."""
        value = self.resolve(value)
        self.resolved = {
            name: rename_expr(expr, {}, {size: value})
            for name, expr in self.resolved.items()
        }
        self.resolved[size] = value
        self.sizes.remove(size)

    # }}}

    def term(self) -> Term:
        """Compose every recorded event, in order, into the program's term."""
        for event in self.recorder.events:
            if isinstance(event, _Made):
                self.made[event.value.name] = event
                self.slots.append((event, self.blocks))
                self.blocks += 1
            else:
                self.slots.append(self.call(event))
        for name, made in self.made.items():
            like = made.value.like
            if name in self.types and like is not None and like.name in self.types:
                self.shaped_like(name, like.name, made.where)
        return self.finish()

    # {{{ one call

    def refuse(self, call: _Call, message: str) -> NoReturn:
        raise TraceError(
            f"{self.program} calls {call.kernel.__name__} at {call.where}: {message}"
        )

    def call(self, call: _Call) -> list[Stmt]:
        """One call's statements, in the program's names."""
        kernel = call.kernel
        try:
            term = kernel.term
        except Exception as exc:
            raise TraceError(
                f"{self.program} calls {kernel.__name__} at {call.where}, whose "
                f"body cannot be traced: {type(exc).__name__}: {exc}"
            ) from exc
        label = self.label(kernel.__name__)
        names: dict[str, str] = {}
        exprs: dict[str, Any] = {}
        arrays: dict[str, str] = {}
        for param, typ in term.params:
            value = call.bound[param]
            mine = isinstance(value, ProgramValue) and value._recorder is self.recorder
            if isinstance(typ, ArrType):
                if not mine:
                    self.refuse(
                        call,
                        f"its argument {param} is {type(value).__name__} "
                        f"{value!r}, which is neither a parameter of "
                        f"{self.program} nor an array it made with "
                        "Arr.zeros_like(...). A program's term has no other "
                        "arrays; pass it in, or make it with Arr.zeros_like",
                    )
                if value.name in arrays.values():
                    other = next(p for p, v in arrays.items() if v == value.name)
                    self.refuse(
                        call,
                        f"it is given {value.name} as both {other} and {param}, "
                        "and two array parameters of a kernel may not share "
                        "storage",
                    )
                self.claim(value.name, "array", call, param)
                arrays[param] = value.name
                names[param] = value.name
            elif mine:
                if value.like is not None:
                    self.refuse(
                        call,
                        f"it is given the array {value.name} for its scalar "
                        f"parameter {param}",
                    )
                self.claim(value.name, "scalar", call, param)
                names[param] = value.name
            elif _is_number(value):
                exprs[param] = value.item() if isinstance(value, np.generic) else value
            else:
                self.refuse(
                    call,
                    f"its scalar argument {param} is {value!r}, which is "
                    f"neither a parameter of {self.program} nor a number",
                )
        exprs.update(self.unify(call, term, arrays))
        for param, typ in term.params:
            if not isinstance(typ, ArrType) and param in names:
                renamed = _rename_type(typ, names, exprs)
                self.settle_type(names[param], renamed, call, param)
        self.settle_layout(call, term, arrays)

        for name in _own_names(term):
            if name not in names and name not in exprs:
                names[name] = self.fresh(name)
        for symbol, expr in term.reflected:
            spelled = _reflected_spelling(rename_expr(expr, names, exprs))
            names[symbol] = self.fresh(spelled or symbol)
            self.reflected.append((names[symbol], rename_expr(expr, names, exprs)))

        stmts: list[Stmt] = []
        top = 0
        for stmt in term.stmts:
            order = _order(stmt)
            top = max(top, order[0] + 1)
            stmts.append(
                Stmt(
                    id=f"{label}.{stmt.id}",
                    inames=tuple(names[iname] for iname in stmt.inames),
                    domain=rename_set(stmt.domain, names, exprs),
                    assignee=rename_expr(stmt.assignee, names, exprs),
                    expr=rename_expr(stmt.expr, names, exprs),
                    kind=stmt.kind,
                    guard=(
                        None
                        if stmt.guard is None
                        else rename_expr(stmt.guard, names, exprs)
                    ),
                    where=stmt.where,
                    order=(order[0] + self.blocks, *order[1:]),
                    loop_domain=(
                        None
                        if stmt.loop_domain is None
                        else rename_set(stmt.loop_domain, names, exprs)
                    ),
                    unnarrowed=stmt.unnarrowed,
                )
            )
        self.blocks += top
        return stmts

    def claim(self, name: str, kind: str, call: _Call, param: str) -> None:
        """Say that program name ``name`` is an array, or a scalar."""
        known = self.kinds.setdefault(name, kind)
        if known != kind:
            self.refuse(
                call,
                f"it is given {name} for its {kind} parameter {param}, and "
                f"{name} is {'an' if known == 'array' else 'a'} {known} where "
                f"{self.origin.get(name, 'another parameter of this call')} is "
                "given it",
            )

    def settle_type(self, name: str, typ: Any, call: _Call, param: str) -> None:
        """Give ``name`` its type, or check the type it already has."""
        here = f"{call.kernel.__name__}'s {param}"
        known = self.types.get(name)
        if known is None:
            self.types[name] = typ
            self.origin[name] = here
            return
        mine = _rename_type(typ, {}, self.resolved)
        theirs = _rename_type(known, {}, self.resolved)
        first = mine.dtype if isinstance(mine, ArrType) else mine
        second = theirs.dtype if isinstance(theirs, ArrType) else theirs
        if str(first) != str(second):
            what = "the elements of " if isinstance(mine, ArrType) else ""
            self.refuse(
                call,
                f"it declares {what}{param} as {first}, and {what}{name}, which "
                f"it is given there, {'are' if what else 'is'} {second} as "
                f"{self.origin[name]}. The lowered program declares {name} once, "
                "so the kernels it is passed to have to declare it alike",
            )

    def unify(
        self, call: _Call, term: Term, arrays: Mapping[str, str]
    ) -> dict[str, Any]:
        """What each size of the callee is, in the program's sizes.

        Every axis of an array the program has already typed is an equation
        between the callee's size expression and the program's. A bare size of
        the callee is solved by what the program has; a size of the program
        standing alone on one side is replaced by the other side everywhere
        (:meth:`equate`); two expressions have to be equal as isl sees them,
        or are refused. A size of the callee nothing determines becomes a size
        of the program, under its own name where that is free.
        """
        params = dict(term.params)
        sizes = list(term.sizes)
        for _, typ in term.params:
            if not isinstance(typ, ArrType):
                continue
            for axis, ragged in zip(typ.axes, typ.ragged, strict=True):
                if ragged:
                    continue
                for name in _names_in(axis):
                    if name not in params and name not in sizes:
                        sizes.append(name)
        marks = {name: _CALLEE + name for name in sizes}
        marked = {**marks, **arrays}
        bound: dict[str, Any] = {}
        pending: list[tuple[str, int, Any, bool, Any, bool]] = []
        for param, typ in term.params:
            if not isinstance(typ, ArrType) or arrays[param] not in self.types:
                continue
            known = self.types[arrays[param]]
            if len(known.axes) != len(typ.axes):
                self.refuse(
                    call,
                    f"its {param} has {len(typ.axes)} axes, and "
                    f"{arrays[param]}, which it is given there, has "
                    f"{len(known.axes)} as {self.origin[arrays[param]]}",
                )
            for k, (axis, ragged) in enumerate(zip(typ.axes, typ.ragged, strict=True)):
                pending.append(
                    (
                        param,
                        k,
                        rename_expr(axis, marked, {}),
                        ragged,
                        known.axes[k],
                        known.ragged[k],
                    )
                )
        while pending:
            # Filtered by position, never by ``in``: an item holds terms, and
            # lanky's ``==`` builds a proposition instead of answering.
            waiting = [
                item for item in pending if not self.settle(call, arrays, item, bound)
            ]
            progressed = len(waiting) < len(pending)
            pending = waiting
            if progressed or not pending:
                continue
            # Every equation left names a size of the callee that no equation
            # solves on its own: it becomes a size of the program.
            loose = next(
                name
                for item in pending
                for name in _names_in(rename_expr(item[2], {}, bound))
                if name.startswith(_CALLEE)
            )
            bound[loose] = prim.Variable(self.new_size(loose[len(_CALLEE) :]))
        for name in sizes:
            if marks[name] not in bound:
                bound[marks[name]] = prim.Variable(self.new_size(name))
        out = {name: bound[marks[name]] for name in sizes}
        for param, typ in term.params:
            if isinstance(typ, ArrType):
                self.settle_type(
                    arrays[param], _rename_type(typ, arrays, out), call, param
                )
        return out

    def new_size(self, stem: str) -> str:
        """A size of the program, under ``stem`` when that is free."""
        name = self.fresh(stem)
        self.sizes.append(name)
        return name

    def settle(
        self,
        call: _Call,
        arrays: Mapping[str, str],
        item: tuple[str, int, Any, bool, Any, bool],
        bound: dict[str, Any],
    ) -> bool:
        """Settle one axis equation, or say that it has to wait."""
        param, k, mine, ragged, theirs, known_ragged = item
        name = arrays[param]
        theirs = self.resolve(theirs)
        if ragged != known_ragged:
            self.refuse(
                call,
                f"axis {k} of its {param} is {'ragged' if ragged else 'dense'}, "
                f"and that axis of {name}, which it is given there, is "
                f"{'ragged' if known_ragged else 'dense'} as {self.origin[name]}",
            )
        if ragged:
            if structurally_equal(mine, theirs):
                return True
            self.refuse(
                call,
                f"the rows of its {param} are counted by {_shown(mine)}, and "
                f"those of {name}, which it is given there, by "
                f"{_shown(theirs)} as {self.origin[name]}",
            )
        mine = self.resolve(rename_expr(mine, {}, bound))
        loose = [found for found in _names_in(mine) if found.startswith(_CALLEE)]
        if isinstance(mine, prim.Variable) and loose:
            bound[mine.name] = theirs
            return True
        if loose:
            return False
        if _same(mine, theirs) or self.equate(mine, theirs):
            return True
        self.refuse(
            call,
            f"axis {k} of its {param} is {_shown(mine)} long, and that axis of "
            f"{name}, which it is given there, is {_shown(theirs)} long as "
            f"{self.origin[name]}; nothing says the two agree",
        )

    def settle_layout(self, call: _Call, term: Term, arrays: Mapping[str, str]) -> None:
        """Record the offsets each ragged argument of the call is read through.

        The kernel's own choice (:meth:`loopty.term.Term.offsets_of`), in the
        program's names: an array it is handed, or ``None`` for the array's
        own offsets. Every call has to make the same choice for one counts
        family, because the lowered program indexes a family's rows through
        one array, and its row lengths have to be a parameter of the program,
        which the call's contract can check them against.
        """
        for param, typ in term.params:
            if not isinstance(typ, ArrType):
                continue
            for size, ragged in zip(typ.axes, typ.ragged, strict=True):
                if not ragged or not isinstance(size, prim.Variable):
                    continue
                counts = arrays.get(size.name)
                if counts is None:
                    continue
                if counts not in {value.name for value in self.recorder.params}:
                    self.refuse(
                        call,
                        f"the rows of its {param} are counted by {counts}, an "
                        f"array {self.program} makes; the row lengths of a "
                        "ragged array have to be a parameter of the program, "
                        "so that the call can check the array against them",
                    )
                offsets = term.offsets_of(size.name)
                mine = None if offsets is None else arrays.get(offsets)
                here = f"{call.kernel.__name__} at {call.where}"
                if counts in self.layout and self.layout[counts] != mine:
                    known = self.layout[counts]
                    self.refuse(
                        call,
                        f"it reads the rows of {arrays[param]} "
                        f"{_through(mine)}, and {self.layout_origin[counts]} "
                        f"reads the rows counted by {counts} {_through(known)}. "
                        "The lowered program reads them one way, so pass every "
                        "kernel that declares the offsets the same array, and "
                        "declare them in all of them or in none",
                    )
                self.layout[counts] = mine
                self.layout_origin.setdefault(counts, here)

    def shaped_like(self, name: str, like: str, where: str) -> None:
        """Unify an array the program made with the one it was made like."""
        mine = self.types[name]
        theirs = self.types[like]
        if not isinstance(theirs, ArrType):
            raise TraceError(
                f"{self.program} makes {name} like {like} at {where}, and "
                f"{like} is a scalar, which has no shape to copy"
            )
        if len(mine.axes) != len(theirs.axes) or mine.ragged != theirs.ragged:
            raise TraceError(
                f"{self.program} makes {name} like {like} at {where}, and "
                f"{self.origin[name]} gives it another shape than {like} has"
            )
        for k, (first, second) in enumerate(zip(mine.axes, theirs.axes, strict=True)):
            first, second = self.resolve(first), self.resolve(second)
            if _same(first, second) or self.equate(first, second):
                continue
            raise TraceError(
                f"{self.program} makes {name} like {like} at {where}, so its "
                f"axis {k} is {_shown(second)} long, and {self.origin[name]} "
                f"wants it {_shown(first)} long; nothing says the two agree"
            )

    # }}}

    def finish(self) -> Term:
        """The term, with every eliminated size replaced by what it is."""
        params: list[tuple[str, Any]] = []
        for value in self.recorder.params:
            if value.name not in self.types:
                raise TraceError(
                    f"the program {self.program} never passes its parameter "
                    f"{value.name} to a kernel, so its term has nothing to say "
                    "about it and no type to give it. Pass it to the kernel "
                    "that uses it, or drop it"
                )
            params.append((value.name, self.types[value.name]))
        temporaries = [
            (name, self.types[name]) for name in self.made if name in self.types
        ]
        stmts: list[Stmt] = []
        for slot in self.slots:
            if isinstance(slot, list):
                stmts.extend(slot)
                continue
            made, block = slot
            name = made.value.name
            if name in self.types:
                stmts.append(self.zeros(name, self.types[name], made, block))
        resolved = self.resolved
        return Term(
            name=self.program,
            params=tuple((name, _rename_type(t, {}, resolved)) for name, t in params),
            sizes=tuple(sorted(self.sizes)),
            stmts=tuple(self.resolved_stmt(stmt) for stmt in stmts),
            post=None,
            reflected=tuple(
                (name, self.resolve(expr)) for name, expr in self.reflected
            ),
            temporaries=tuple(
                (name, _rename_type(t, {}, resolved)) for name, t in temporaries
            ),
            offsets=tuple(self.layout.items()),
            where=self.recorder.where,
        )

    def zeros(self, name: str, typ: ArrType, made: _Made, block: int) -> Stmt:
        """The statement that zeroes an array the program made, where it made it.

        A ragged one is refused here: a temporary has no offsets its rows could
        be found through, and nothing to check them against.
        """
        typ = _rename_type(typ, {}, self.resolved)
        if any(typ.ragged):
            raise TraceError(
                f"{self.program} makes {name} at {made.where}, and the kernels "
                f"it is passed to read it as a ragged array. An array a "
                "program makes is a temporary of the lowered kernel, which has "
                "no offsets for its rows; make it a parameter of the program"
            )
        inames = tuple(self.fresh("i") for _ in typ.axes)
        return Stmt(
            id=f"{name}.zeros",
            inames=inames,
            domain=domain_set(inames, typ.axes),
            assignee=Access(name, tuple(prim.Variable(iname) for iname in inames)),
            expr=0,
            kind="assign",
            guard=None,
            where=made.where,
            order=(block, *(0,) * len(inames)),
        )

    def resolved_stmt(self, stmt: Stmt) -> Stmt:
        """A statement with every eliminated size replaced."""
        if not self.resolved:
            return stmt
        exprs = self.resolved
        return dataclasses.replace(
            stmt,
            domain=rename_set(stmt.domain, {}, exprs),
            assignee=rename_expr(stmt.assignee, {}, exprs),
            expr=rename_expr(stmt.expr, {}, exprs),
            guard=None if stmt.guard is None else rename_expr(stmt.guard, {}, exprs),
            loop_domain=(
                None
                if stmt.loop_domain is None
                else rename_set(stmt.loop_domain, {}, exprs)
            ),
        )


# }}}
