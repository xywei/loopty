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
every cast is checked against. Fusing two calls is a cast over this term
(:meth:`loopty.schedule.Schedule.fuse`), and computing an array the program
makes where it is read, instead of storing it, is a step of a schedule too
(:meth:`loopty.schedule.Schedule.substitute`).

Three things are decided in composing.

*Names are the program's.* A callee's array and scalar parameters become what
the program passed: a parameter of the program, an array it made, or, for a
scalar, a Python number, which is substituted. The number is no argument of
the compiled program, so it is checked against the scalar's sort when the term
is built, as the callee's contract checks it at the call natively; a ``Fin``
bounded by a size cannot be checked before the call, and a number for it is
refused. Its sizes are unified through the arrays, and a scalar that sizes an
array (``m: Nat`` beside ``x: Arr[Fin[m], Real]``) sizes it in the program's
names too. A program array has one type, taken from the first call that
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
outside the kernel sees it. Natively it has ``u``'s dtype, which has to hold
what the compiled temporary holds, whatever its sort
(:func:`loopty.contract.native_storage`): a ``dtype`` given to
``zeros_like`` is checked here, and ``u``'s is checked when the compiled
program is run (:attr:`~loopty.term.Term.temporaries_like`).

*A parameter has one role.* It is an output of the program's term when any
call writes it, which the lowering reads off the statements, and an input
otherwise. The two declarations that independent lowerings disagree about
(``f`` an output of the producer and input-only in the consumer) cannot arise,
because there is one lowering. The layout of a ragged parameter is stated on
the term (:attr:`~loopty.term.Term.offsets`): the offsets each call's kernel
reads its rows through, which the calls have to agree on.

The compiled program is one call, so its contract checks the program's
arguments once, when it starts, where natively every kernel's contract checks
its own when it is called. What a kernel's contract checks of the cells of an
array (that a ``Fin[m]`` element is a point of ``Fin[m]``, that a ``Nat``
element is not negative, that the offsets a ragged family is read through are
the ones its counts give) is what its in-bounds facts, and the runs its
postcondition is tested on, rest on. A kernel's requirements on its inputs are its
argument types, so where an earlier call wrote such an array, or the program
made it, the requirement is an obligation of the program
(:class:`~loopty.term.Requirement`), and facts travel to discharge it.

*A requirement is decided under what held at the call.* The postconditions of
the earlier calls whose arrays nothing has written over since, the zeros an
``Arr.zeros_like`` starts an array with, the types the program's contract
checks of the arrays nothing has written yet, and the theorems the program
cites (``@program(uses=[scan_monotone])``), instantiated at those, are its
hypotheses, and isl decides it under them (:mod:`loopty.hypotheses`): ``gather``
reads ``x[perm[i]]`` in bounds by the element type of ``perm``, and
``number``'s postcondition says what ``number`` left in ``perm``. The fact
rests on the facts it used.

*Where it is not decided, the program checks it.* The lowered program gets
the contract's cell checks for that array between the two calls: a statement
of its own that sets a one-cell flag when a cell fails, with every later
statement guarded by the flag, so the compiled program runs no further where
the native one is refused, and the executor raises the requirement's message.
The requirement stays ``assumed``, and its fact says why. A count an earlier
call wrote is still refused: the rows of the ragged array it bounds are laid
out in a buffer the program was given, which nothing in the term can state.

*The compiled program trusts no more than a run has borne out.* A decided
requirement whose fact rests on something not at least ``tested`` (a
postcondition its kernel's runs refute, a cited theorem the property tester
does not pass, an axiom) keeps its checked point, and stays ``decided`` in
the ledger, worth what it rests on (#115). And a postcondition is tested on
the runs its kernel's contract lets in, so it is a hypothesis after a call at
which that contract held, which every requirement of the call, decided or
checked, makes so: the ``Nat`` element sort of an array an earlier call wrote
is one of them (#119), so that the compiled program stops where the native
callee is refused for a negative cell, and does not run it.

The hypotheses that held where each call ran are kept with the term too
(:class:`~loopty.term.Scope`), so that a callee's fact its own term leaves
``assumed`` can be decided in the program's scope (see
:func:`loopty.typing.scoped_in_bounds_facts`).

What a program's body may do is pass things to kernels, make arrays with
``Arr.zeros_like``, and run Python that touches neither. Reading or writing a
placeholder, asking it for its domain, computing with it or branching on it is
refused with a :class:`~loopty.trace.TraceError` naming the fix, and so is a
loop whose trip count is a parameter: that is a host loop, which a term does
not have. A ``for`` over ``range(3)`` runs three times and records three calls,
which is what it does natively. Three more things are refused: a parameter
with a default, which the compiled program would be called without; a body
that returns an array it made, which the compiled program keeps to itself (a
program called by a program may return one, which the caller then passes on);
and a program that writes none of its parameters, which would compute nothing
anybody sees. A callee with an array over a polyhedral domain
(:mod:`loopty.domain`) is refused too, since a domain's sizes are not unified
across calls.

Each callee's postcondition stays its own, and
:meth:`loopty.kernel.Program.facts` restates it in the program's scope. The
term's own postcondition is the program's return annotation, a claim about
what the program leaves in its parameters (``-> all(grad[x, e, k, j, i] ==
... for x in Fin[3] ...)``), written in the sizes the kernels it calls name;
:meth:`loopty.kernel.Program.facts` tests it on the program's native runs
and decides it from what the calls store (:mod:`loopty.unfolding`). What
the body does with a placeholder is all the term can see, and a body can tell
one from an array without touching it (``isinstance(x, Arr)``), so the term
can leave out calls the native run makes. That is what the program's
``trace-faithful`` fact is for, as a kernel's is (:mod:`loopty.faithful`): the
term, interpreted, against the body, run natively.
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
from lanky.prelude import FinType, Nat, Refined
from lanky.terms import (
    Comparison,
    Forall,
    LogicalAnd,
    LogicalOr,
    Subscript,
    Var,
    init_args,
    render,
    structurally_equal,
)

from loopty.contract import (
    CHECKED_STORAGE,
    holds_natively,
    integral_sort,
    native_storage,
    sort_bound,
    storage_wanted,
)
from loopty.description import describe, description_of
from loopty.flow import NonAffine, domain_set, expr_text, free_names
from loopty.hypotheses import (
    discharge,
    free_in,
    linear,
    mentioned,
    substitute,
    theorem_instances,
)
from loopty.idx import Reflections
from loopty.names import rename
from loopty.table import Table, passed_message
from loopty.term import (
    Access,
    ArrType,
    Binder,
    Callsite,
    Fiber,
    Hypothesis,
    Reduction,
    Requirement,
    Scope,
    Stmt,
    Term,
)
from loopty.trace import TraceError

__all__ = [
    "ProgramValue",
    "current_recorder",
    "rename_expr",
    "rename_set",
    "stopped_at",
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
            loop_domain=(
                None
                if expr.loop_domain is None
                else rename_set(expr.loop_domain, names, exprs)
            ),
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
    domain: :func:`loopty.names.rename`, which renames every name at once, so
    that ``a -> b`` beside ``b -> c`` does not give the set two dimensions
    called ``b``, replaces a parameter mapped to an expression, or to a name
    the set already has as a parameter, and checks the names of what it
    returns.
    """
    out = rename(obj, names, exprs, what="a callee's domain in its caller's names")
    described = description_of(obj)
    if described is not None:
        # the text the callee's set was written in, read in the caller's names
        # (loopty.description)
        describe(out, described.renamed(names, exprs))
    return out


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


def _rename_sort(sort: Any, names: Mapping[str, str], exprs: Mapping[str, Any]) -> Any:
    """A sort whose bound or refinement names sizes, in the new names.

    ``Fin[m]`` is the element sort of a column-index array and the sort of an
    index argument, and its bound is a size of the kernel. Any other sort is
    returned as it is.
    """
    if isinstance(sort, FinType):
        return dataclasses.replace(
            sort, bound=_tidy(rename_expr(sort.bound, names, exprs))
        )
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
            axes=tuple(_tidy(rename_expr(axis, names, exprs)) for axis in typ.axes),
            dtype=_rename_sort(typ.dtype, names, exprs),
            ragged=typ.ragged,
            domain=(
                None
                if typ.domain is None
                else _rename_domain(typ.domain, names, exprs)
            ),
        )
    return _rename_sort(typ, names, exprs)


def _rename_domain(
    domain: Any, names: Mapping[str, str], exprs: Mapping[str, Any]
) -> Any:
    """A polyhedral domain with its sizes in the new names; its binders are kept.

    Only the names the domain does not bind are renamed: a binder ``i`` is
    the domain's own, whatever a parameter of the callee is called.
    """
    from loopty.domain import Polyhedron, Union

    if isinstance(domain, Union):
        pieces = domain.pieces
        return Union(tuple(_rename_domain(piece, names, exprs) for piece in pieces))
    sizes = domain.size_names()
    own_names = {key: value for key, value in names.items() if key in sizes}
    own_exprs = {key: value for key, value in exprs.items() if key in sizes}
    if not own_names and not own_exprs:
        return domain
    meant = set(own_names.values())
    for value in own_exprs.values():
        meant |= set(_names_in(value))
    if meant & set(domain.names):
        raise TraceError(
            f"the domain {domain} would be read in names one of its binders has "
            f"({', '.join(sorted(meant & set(domain.names)))}); rename the binder "
            "in the kernel's annotation"
        )
    assert isinstance(domain, Polyhedron)
    return Polyhedron(
        binders=tuple(
            (name, _tidy(rename_expr(bound, own_names, own_exprs)))
            for name, bound in domain.binders
        ),
        constraints=tuple(
            rename_expr(prop, own_names, own_exprs) for prop in domain.constraints
        ),
        spelling=domain.spelling,
        shown=domain.shown,
    )


def _shape_text(typ: Any) -> str:
    """``an array over Where[...]``, or ``a dense or ragged array``."""
    if isinstance(typ, ArrType) and typ.domain is not None:
        return f"an array over {typ.domain}"
    return "a dense or ragged array"


def _ragged_axes(typ: ArrType) -> str:
    """Which axes of a dense or ragged type are ragged, in words."""
    axes = [str(k) for k, ragged in enumerate(typ.ragged) if ragged]
    if not axes:
        return "no axis"
    return f"axis {axes[0]}" if len(axes) == 1 else f"axes {', '.join(axes)}"


def _same_domain(first: Any, second: Any) -> bool:
    """Whether two domains, in one set of sizes, have the same points at every size."""
    from loopty.domain import dimension_names

    if first.ndim != second.ndim:
        return False
    avoid = first.size_names() | second.size_names()
    names = dimension_names(first.ndim, avoid)
    left, right = first.isl_set(names), second.isl_set(names)
    left = left.align_params(right.get_space())
    right = right.align_params(left.get_space())
    return bool(left.is_equal(right))


def _tidy(expr: Any) -> Any:
    """A sum of sums of names and integers as one sum, its integers added up.

    Substituting ``n_0 + 1`` for ``n`` in ``n + 1`` gives ``(n_0 + 1) + 1``,
    which is ``n_0 + 2``; anything that is not such a sum is left as it is.
    """
    if not isinstance(expr, prim.Sum):
        return expr
    leaves: list[Any] = []

    def flatten(node: Any) -> None:
        if isinstance(node, prim.Sum):
            for child in node.children:
                flatten(child)
        else:
            leaves.append(node)

    flatten(expr)

    def integer(node: Any) -> bool:
        return isinstance(node, int | np.integer) and not isinstance(node, bool)

    others = [leaf for leaf in leaves if not integer(leaf)]
    constant = sum(int(leaf) for leaf in leaves if integer(leaf))
    if not all(isinstance(leaf, prim.Variable) for leaf in others):
        return expr
    if not others:
        return constant
    out = others[0]
    for leaf in others[1:]:
        out = out + leaf
    return out + constant if constant else out


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


def _shifted(expr: Any) -> tuple[str, int] | None:
    """``(n, c)`` when ``expr`` is a name plus an integer, ``n + c``, or ``None``.

    ``n`` alone is ``(n, 0)``, and ``n - 1`` is ``(n, -1)``.
    """
    if isinstance(expr, prim.Variable):
        return expr.name, 0
    if not isinstance(expr, prim.Sum):
        return None
    names = [child for child in expr.children if isinstance(child, prim.Variable)]
    constants = [
        child
        for child in expr.children
        if isinstance(child, int | np.integer) and not isinstance(child, bool)
    ]
    if len(names) != 1 or len(names) + len(constants) != len(expr.children):
        return None
    return names[0].name, int(sum(constants))


def _plus(expr: Any, constant: int) -> Any:
    """``expr + constant``, with the integers of a sum added up."""
    if not constant:
        return expr
    shifted = _shifted(expr)
    if shifted is None:
        return expr + constant
    name, known = shifted
    total = known + constant
    variable = next(node for node in _variables(expr) if node.name == name)
    return variable + total if total else variable


def _variables(expr: Any) -> list[Any]:
    """The variable nodes of an expression, in the order they occur."""
    out: list[Any] = []

    def visit(node: Any) -> None:
        if isinstance(node, prim.Variable):
            out.append(node)
        elif isinstance(node, prim.ExpressionNode):
            for arg in init_args(node):
                visit(arg)
        elif isinstance(node, tuple | list):
            for item in node:
                visit(item)

    visit(expr)
    return out


def _shown(value: Any) -> str:
    """A size or a sort as a message prints it, a numpy scalar type by name."""
    if isinstance(value, prim.ExpressionNode):
        return render(value)
    if isinstance(value, type) and issubclass(value, np.generic):
        return np.dtype(value).name
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

    __slots__ = ("_recorder", "like", "made", "name")

    def __init__(
        self,
        recorder: _Recorder,
        name: str,
        like: ProgramValue | None = None,
        made: bool = False,
    ) -> None:
        self._recorder = recorder
        self.name = name
        #: What ``Arr.zeros_like`` made this like, for an array the body made
        #: like another; ``None`` for a parameter and for one ``Arr.zeros``
        #: made of a declared shape.
        self.like = like
        #: Whether the body made it, with ``Arr.zeros_like`` or ``Arr.zeros``.
        self.made = made

    def _loopty_zeros_like(self, frame: Any, dtype: Any = None) -> ProgramValue:
        """``Arr.zeros_like(self)``, under tracing: a new array of the program."""
        return self._recorder.make(self, frame, dtype)

    def _refuse(self, what: str) -> NoReturn:
        program = self._recorder.name
        which = (
            f"the array {self.name} it made"
            if self.made
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
        if attribute == "dom":
            # Its sizes, and nothing else: x.dom.size is the size of an
            # array the program makes, Arr.zeros(TRIANGLE, n=x.dom.size).
            return _ProgramDom(self, ())
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


#: How a size of a program's array is named while the term is built: the
#: array and the indices of the fiber, ``@size:x:`` for ``x.dom.size`` and
#: ``@size:x:0,2`` for ``x.dom[0][2].size``. No identifier starts with it, so
#: it is no size's name.
_SIZE_OF = "@size"


def size_of(name: str, index: tuple[int, ...] = ()) -> Var:
    """The name ``x.dom[...].size`` stands for, in a program's body traced."""
    return Var(f"{_SIZE_OF}:{name}:{','.join(str(k) for k in index)}")


class _ProgramDom:
    """``x.dom`` of a program's placeholder: its sizes, for an array the body makes.

    ``x.dom.size`` is the length of ``x``'s first axis and ``x.dom[0].size``
    of its second, as natively, which the program's term reads off ``x``'s
    type once a kernel has given it one. The fiber's indices are kept with
    the size: natively ``x.dom[0]`` raises ``IndexError`` where ``x`` has no
    row 0, and so the compiled program is refused at such sizes (#250, see
    :class:`loopty.term.Fiber`). Iterating it is a loop of the body's whose
    trip count is an argument, a host loop, and is refused.
    """

    __slots__ = ("prefix", "value")

    def __init__(self, value: ProgramValue, prefix: tuple[Any, ...]) -> None:
        self.value = value
        self.prefix = prefix

    @property
    def size(self) -> Any:
        """The length of the axis, a size of the program."""
        return size_of(self.value.name, self.prefix)

    def __getitem__(self, index: Any) -> _ProgramDom:
        if not isinstance(index, int | np.integer) or isinstance(index, bool):
            self.value._refuse("takes a fiber at a value that is no number of")
        if index < 0:
            spelled = f"{self.value.name}.dom" + "".join(
                f"[{k}]" for k in (*self.prefix, int(index))
            )
            raise TraceError(
                f"the body of the program {self.value._recorder.name} takes "
                f"{spelled}, a fiber at a negative index, which no axis has: "
                f"natively {spelled} raises IndexError at every size. Index "
                "a fiber from 0"
            )
        return _ProgramDom(self.value, (*self.prefix, int(index)))

    def __iter__(self) -> Any:
        self.value._refuse("iterates over the domain of")

    def __len__(self) -> int:
        self.value._refuse("asks for the length of the domain of")

    def __bool__(self) -> bool:
        self.value._refuse("branches on the domain of")


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
    """One ``Arr.zeros_like`` of the body: the array it made, where, and the
    ``dtype`` it was given, which is the native array's alone.

    ``declared`` is the shape an ``Arr.zeros`` gave it instead, a domain or
    a tuple of axes, with the sizes it was given (``n=x.dom.size``), in the
    names :func:`size_of` writes.
    """

    value: ProgramValue
    where: str
    dtype: Any = None
    declared: Any = None
    sizes: dict[str, Any] | None = None


@dataclass
class _Want:
    """A requirement of one call on an array an earlier call wrote.

    ``kind`` is ``"element"`` (the callee declares the array's elements of a
    ``Fin[m]`` sort), ``"nat"`` (of the ``Nat`` sort) or ``"layout"`` (the
    callee reads the rows of ``owner``, counted by ``counts``, through the
    array as offsets). ``writer`` is the call, or the ``Arr.zeros_like``,
    that wrote it last.
    """

    kind: str
    param: str
    array: str
    writer: str
    owner: str = ""
    counts: str = ""


@dataclass
class _Done:
    """One composed call: its statements and what it means to the calls after it.

    ``stmts`` are in the program's names, their blocks counted from
    ``start``, which the checked points before the call shift
    (:meth:`_Composer.travel`). ``post`` is the callee's postcondition in the
    program's names, ``types`` its parameters' types and ``arrays`` the
    program array passed for each, ``written`` the program arrays it writes,
    and ``wants`` its requirements on arrays written before it.
    ``binders`` and ``callsites`` are the callee's loops and calls, in the
    program's names and inside this call (:class:`loopty.term.Binder`).
    """

    label: str
    kernel: Any
    where: str
    start: int
    stmts: list[Stmt]
    post: Any
    types: dict[str, Any]
    arrays: dict[str, str]
    written: set[str]
    wants: list[_Want]
    binders: tuple[Binder, ...] = ()
    callsites: tuple[Callsite, ...] = ()


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

    def declare(
        self, shape: Any, frame: Any, dtype: Any, storage: str, sizes: dict[str, Any]
    ) -> ProgramValue:
        """Record ``Arr.zeros(shape, ...)`` of the body: an array of a declared shape.

        ``shape`` is a domain (``Where[...]``) with a value for each of its
        sizes, or a tuple of axes; a size is a number, or a size of an array
        of the program, ``x.dom.size``. It is stored boxed.
        """
        from loopty.domain import index_domain

        where = _where(frame)
        if storage != "box":
            raise TraceError(
                f"the program {self.name} makes an array at {where} stored "
                f"{storage!r}, and an array a program makes is stored boxed: a "
                "packed one would need its table of row starts, which nothing "
                "computes for it. Leave storage out"
            )
        domain = index_domain(shape)
        if domain is not None:
            declared: Any = domain
            missing = sorted(domain.size_names() - set(sizes))
            if missing:
                raise TraceError(
                    f"the program {self.name} makes an array over {domain} at "
                    f"{where} and gives no {', '.join(missing)}: give each size, "
                    f"{missing[0]}=x.dom.size for an array x of that length"
                )
        else:
            parts = shape if isinstance(shape, tuple) else (shape,)
            declared = tuple(
                part.bound if isinstance(part, FinType) else part for part in parts
            )
            if sizes:
                raise TraceError(
                    f"the program {self.name} gives sizes to Arr.zeros at {where}, "
                    "whose shape is no domain; write them in the shape, "
                    "Arr.zeros((x.dom.size, 3))"
                )
        value = self.make(None, frame, dtype)
        made = self.events[-1]
        assert isinstance(made, _Made) and made.value is value
        made.declared = declared
        made.sizes = dict(sizes)
        return value

    def make(
        self, like: ProgramValue | None, frame: Any, dtype: Any = None
    ) -> ProgramValue:
        """Record an array the body makes, named after what it is stored to."""
        stem = _stored_name(frame) or "tmp"
        name = stem
        suffix = 0
        while name in self.taken:
            name = f"{stem}_{suffix}"
            suffix += 1
        self.taken.add(name)
        value = ProgramValue(self, name, like=like, made=True)
        self.events.append(_Made(value, _where(frame), dtype))
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

    The theorems the program cites (``program.uses``) are offered as
    hypotheses wherever a requirement is decided, and the facts the program's
    requirements rest on are named by the ids its own facts give them, which
    are keyed by its definition (``qualname``, ``module`` and ``line``).
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
        if parameter.default is not inspect.Parameter.empty:
            # The body runs against a placeholder for every parameter, so the
            # term has an argument where the native call may use the default,
            # and the compiled program called without it has nothing to pass.
            raise TraceError(
                f"the program {program.__name__} gives its parameter "
                f"{parameter.name} the default {parameter.default!r}. Its term "
                "is one kernel, which is passed every argument, so a call that "
                "leaves it out natively has no value for it compiled; drop the "
                "default, or pass the value to the kernel inside the body"
            )
        names.append(parameter.name)
    recorder = _Recorder(program, names)
    _RECORDERS.append(recorder)
    try:
        returned = function(*recorder.params)
    finally:
        _RECORDERS.pop()
    made = _made_in(returned)
    if made is not None:
        # A program called by this one may return an array it made, which
        # this body then passes on; that is recorded above. But the program
        # whose term this is lowers to one kernel, whose results are what it
        # writes into its parameters: an array it made is a temporary of that
        # kernel, and nobody would see it, or compare it with the native one.
        raise TraceError(
            f"the program {program.__name__} returns {made!r}. Its term "
            "lowers to one kernel, whose results are what it writes into its "
            f"parameters, and {made.name} is a temporary of that kernel, which "
            "the compiled program keeps to itself; write the result into a "
            "parameter instead"
        )
    return _Composer(recorder, program).term()


def _made_in(value: Any) -> ProgramValue | None:
    """An array the program made, in what its body returns, or ``None``.

    Looked for in tuples, lists and the values of a dictionary, the containers
    a body returns several results in.
    """
    if isinstance(value, ProgramValue):
        return value if value.made else None
    if isinstance(value, dict):
        value = tuple(value.values())
    if isinstance(value, tuple | list):
        for item in value:
            found = _made_in(item)
            if found is not None:
                return found
    return None


# }}}


# {{{ composing


#: What a callee's own size is called while it is unified with the program's.
#: No identifier starts with it, so a callee's ``n`` can never be taken for
#: the program's ``n``.
_CALLEE = "@"

#: Why a guard conjunct that reads an array leaves a domain as it is, in the
#: words :mod:`loopty.trace` uses for an ``if`` of the body.
_NOT_AFFINE = "reads an array or is not affine"


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


def _binders(term: Term) -> tuple[Binder, ...]:
    """A callee's loops: its term's records, or its statements' loops.

    A term written by hand records none (:attr:`loopty.term.Term.binders`),
    and its loops are then the loops of its statements and of their sums,
    each named as the term names it.
    """
    from loopty.handles import binders_of

    return binders_of(term)


def _fin_bound(sort: Any) -> Any:
    """The bound of a ``Fin`` sort, a refinement's included, or ``None``."""
    base = sort.base if isinstance(sort, Refined) else sort
    return base.bound if isinstance(base, FinType) else None


def _through(offsets: str | None) -> str:
    """How a message says which offsets rows are read through."""
    return "through its own offsets" if offsets is None else f"through {offsets}"


#: What :func:`_theorem_standing` found of each theorem, by identity.
_THEOREM_STANDINGS: dict[int, tuple[Any, str | None]] = {}


def _theorem_standing(theorem: Any) -> str | None:
    """``None`` when lanky's property tester passes ``theorem``, else why not.

    That is the least a ``lanky check`` of the theorem gives it, ``tested``,
    and what a program that skips a check on the strength of the theorem
    asks of it. An axiom is ``assumed`` on its citation, and a theorem that
    rests on facts of its own (``uses=``) is worth what they are, which the
    program does not see; neither is enough. A stronger oracle that would
    prove what the tester cannot sample is not asked, so the check stays.
    """
    found = _THEOREM_STANDINGS.get(id(theorem))
    if found is not None and found[0] is theorem:
        return found[1]
    name = getattr(theorem, "__name__", None) or getattr(theorem, "qualname", "theorem")
    try:
        from lanky.ledger import STATUS_STRENGTH
        from lanky.oracles.test import TestOracle

        fact = theorem.fact()
        if fact.is_axiom:
            why: str | None = f"the cited claim {name} is assumed on its source"
        elif fact.rests_on:
            why = (
                f"{name} rests on {', '.join(fact.rests_on)}, whose status the "
                "program does not know"
            )
        else:
            tested = TestOracle().establish(fact) or fact
            if STATUS_STRENGTH[tested.status] < 1:
                reason = tested.provenance.get("untested") or tested.provenance.get(
                    "reason"
                )
                why = f"{name} is {tested.status.value}" + (
                    f" ({reason})" if reason else ""
                )
            elif tested.provenance.get("goal_reached") == 0:
                why = f"{name} passed no draw at which its goal's guard held"
            else:
                why = None
    except Exception as exc:  # noqa: BLE001 - then it stands on nothing
        why = f"{name} could not be tested: {type(exc).__name__}: {exc}"
    _THEOREM_STANDINGS[id(theorem)] = (theorem, why)
    return why


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


def _laid_out(term: Term, typ: ArrType, read_through: Any) -> bool:
    """Whether a layout requirement makes every cell of ``typ`` a natural.

    ``read_through`` is ``(owner, counts)`` when the callee reads the rows
    of ``owner``, counted by ``counts``, through the array, and ``None``
    otherwise. The layout says ``off[0] == 0`` and ``off[q + 1] == off[q] +
    cnt[q]`` for every row ``q``, which covers every cell of offsets of one
    axis, one longer than the rows, and makes each a sum of counts; those
    are naturals where their sort is ``Nat`` or ``Fin``.
    """
    if read_through is None or len(typ.axes) != 1:
        return False
    owner, counts = read_through
    rows = term.array_types[owner].axes[0]
    if not _same(typ.axes[0], _plus(rows, 1)):
        return False
    counts_type = term.array_types.get(counts)
    if not isinstance(counts_type, ArrType):
        return False
    sort = counts_type.dtype
    sort = sort.base if isinstance(sort, Refined) else sort
    return isinstance(sort, FinType) or getattr(sort, "name", None) == "Nat"


class _Composer:
    """The program's term, built one recorded event at a time."""

    def __init__(self, recorder: _Recorder, program: Any = None) -> None:
        self.recorder = recorder
        self.program = recorder.name
        #: The decorated program, whose return annotation is the term's
        #: postcondition.
        self.declared = program
        #: What keys the program's facts: its qualified name, module and line.
        self.owner = {
            "owner": getattr(program, "qualname", recorder.name),
            "module": getattr(program, "module", None),
            "line": getattr(program, "line", None),
        }
        #: The theorems the program cites, offered as hypotheses.
        self.uses: tuple[Any, ...] = tuple(getattr(program, "uses", ()))
        #: The callee each restated postcondition is the postcondition of, by
        #: the restatement's id, and why each fact a decision rests on is not
        #: at least ``tested`` (``None`` when it is), by id.
        self.postconditions: dict[str, Any] = {}
        #: The callee each ``trace-faithful`` fact a postcondition hypothesis
        #: rests on is the fact of, by its id.
        self.faithfuls: dict[str, Any] = {}
        self.standings: dict[str, str | None] = {}
        #: What last wrote each program array, by name.
        self.last_writer: dict[str, str] = {}
        #: The offsets some call has read rows through so far, and those an
        #: earlier call wrote before the first of them did, which the
        #: program's contract does not compare with the rows' own offsets on
        #: entry: a requirement does, where they are first read.
        self.read_through: set[str] = set()
        self.deferred: list[str] = []
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
        self.slots: list[_Done | tuple[_Made, int]] = []
        self.reflected: list[tuple[str, Any]] = []
        self.made: dict[str, _Made] = {}
        #: Every program array something has written so far, with what wrote
        #: it first: a call, or the ``Arr.zeros_like`` that made it.
        self.writers: dict[str, str] = {}
        #: Each temporary whose native dtype is a parameter's, with it.
        self.inherits: list[tuple[str, str]] = []
        #: How many top-level blocks the statements so far occupy.
        self.blocks = 0
        #: The fibers the body read sizes off for the arrays it made.
        self.fibers: list[Fiber] = []
        self.labels: set[str] = set()
        self.calls: dict[str, int] = {}
        #: The constant tables the callees read, by the program's name for
        #: each: one name for one table, whichever callees read it.
        self.tables: dict[str, Any] = {}

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
            return self.equate_shifted(first, second)
        _, size, value = max(candidates, key=lambda candidate: candidate[0])
        self.eliminate(size, value)
        return True

    def equate_shifted(self, first: Any, second: Any) -> bool:
        """Make ``a + c`` equal to ``b + d``, two sizes of the program shifted.

        The size on the side with the smaller shift goes, and becomes the
        other plus the difference: ``a + 1 = b + 2`` makes ``a`` into ``b + 1``,
        never ``b`` into ``a - 1``, which a size ``a`` of 0 would make
        negative. Equal shifts make the sizes equal, and the younger goes.
        """
        one, other = _shifted(first), _shifted(second)
        if one is None or other is None:
            return False
        (a, c), (b, d) = one, other
        if a == b or a not in self.sizes or b not in self.sizes:
            return False
        if c == d:
            younger, older = sorted((a, b), key=self.sizes.index, reverse=True)
            self.eliminate(younger, Var(older))
        elif c < d:
            self.eliminate(a, _plus(Var(b), d - c))
        else:
            self.eliminate(b, _plus(Var(a), c - d))
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
                maker = "Arr.zeros" if event.declared is not None else "Arr.zeros_like"
                self.writers.setdefault(
                    event.value.name, f"the {maker} at {event.where}"
                )
                self.last_writer[event.value.name] = f"the {maker} at {event.where}"
                self.slots.append((event, self.blocks))
                self.blocks += 1
            else:
                self.slots.append(self.call(event))
        for name, made in self.made.items():
            if name not in self.types:
                continue
            # Made like an array no kernel was given, it is laid out as that
            # one is, which is laid out as what it was made like, and so on.
            like = made.value.like
            while like is not None and like.name not in self.types:
                like = like.like
            if like is not None:
                self.shaped_like(name, like.name, made.where)
            if made.declared is not None:
                self.shaped_declared(name, made)
            self.stored_like(name, made)
        return self.finish()

    def shaped_declared(self, name: str, made: _Made) -> None:
        """Unify an array the program made with ``Arr.zeros`` with its declared shape.

        The shape's sizes are numbers or sizes of the program's arrays
        (``x.dom.size``), read off their types now that every call has
        given them one.
        """
        mine = self.types[name]
        sizes = {
            key: self.size_value(value, made) for key, value in made.sizes.items()
        }
        if isinstance(made.declared, tuple):
            axes = tuple(self.size_value(axis, made) for axis in made.declared)
            if mine.domain is not None or any(mine.ragged) or len(mine.axes) != len(
                axes
            ):
                raise TraceError(
                    f"{self.program} makes {name} at {made.where} of "
                    f"{len(axes)} axes, and {self.origin[name]} wants it "
                    f"{_shape_text(mine)} of {len(mine.axes)}"
                )
            for k, (first, second) in enumerate(zip(mine.axes, axes, strict=True)):
                first, second = self.resolve(first), self.resolve(second)
                if _same(first, second) or self.equate(first, second):
                    continue
                raise TraceError(
                    f"{self.program} makes {name} at {made.where} with an axis {k} "
                    f"of length {_shown(second)}, and {self.origin[name]} wants "
                    f"it of length {_shown(first)}; nothing says the two agree"
                )
            return
        theirs = ArrType(
            axes=(),
            dtype=mine.dtype,
            ragged=(),
            domain=_rename_domain(made.declared, {}, sizes),
        )
        self.shaped_over(name, mine, theirs, "with Arr.zeros", made.where)

    def size_value(self, value: Any, made: _Made) -> Any:
        """A size given to ``Arr.zeros``, in the program's sizes.

        A number is itself, and ``x.dom.size`` is the length of ``x``'s axis
        as its type says.
        """
        if isinstance(value, int | np.integer) and not isinstance(value, bool):
            return int(value)
        if not isinstance(value, prim.ExpressionNode):
            raise TraceError(
                f"{self.program} gives Arr.zeros at {made.where} the size "
                f"{value!r}, which is neither a number nor a size of its arrays "
                "(x.dom.size)"
            )
        exprs = {}
        for found in _names_in(value):
            if not found.startswith(_SIZE_OF + ":"):
                raise TraceError(
                    f"{self.program} gives Arr.zeros at {made.where} a size "
                    f"that names {found}, which is no size of its arrays; write "
                    "x.dom.size for the length of an array x"
                )
            _tag, array, fixed = found.split(":")
            index = tuple(int(k) for k in fixed.split(",") if k)
            exprs[found] = self.axis_size(array, index, made)
        return self.resolve(_tidy(rename_expr(value, {}, exprs)))

    def axis_size(self, array: str, index: tuple[int, ...], made: _Made) -> Any:
        """The length of the axis after the fiber at ``index`` of ``array``.

        ``x.dom.size`` is the length of ``x``'s first axis, and
        ``x.dom[0].size`` of its second, where ``x`` has a row 0. A fiber
        at an index past an axis of a numeric length is refused here, as it
        is natively at every size; one past an axis whose length is a size
        is a :class:`~loopty.term.Fiber` of the term, which the compiled
        run checks (:func:`loopty.contract.made_fibers`).
        """
        axis = len(index)
        typ = self.types.get(array)
        if not isinstance(typ, ArrType):
            raise TraceError(
                f"{self.program} sizes the array it makes at {made.where} by "
                f"{array}.dom, and {array} is passed to no kernel, which would "
                "give it a type to read the size off"
            )
        typ = _rename_type(typ, {}, self.resolved)
        if typ.domain is not None:
            if axis != 0 or not hasattr(typ.domain, "binders"):
                raise TraceError(
                    f"{self.program} reads the size of axis {axis} of {array}, an "
                    f"array over {typ.domain}, at {made.where}; read the first "
                    "axis of a domain, or the axes of a dense array"
                )
            return typ.domain.binders[0][1]
        if axis >= len(typ.axes) or typ.ragged[axis]:
            raise TraceError(
                f"{self.program} reads the size of axis {axis} of {array} at "
                f"{made.where}, which "
                + (
                    "it does not have"
                    if axis >= len(typ.axes)
                    else "is ragged: its length is a row's"
                )
            )
        if index:
            fiber = Fiber(
                made=made.value.name,
                where=made.where,
                array=array,
                index=index,
                extents=typ.axes[:axis],
            )
            for position, (k, extent) in enumerate(
                zip(index, fiber.extents, strict=True)
            ):
                if isinstance(extent, int | np.integer) and k >= extent:
                    raise TraceError(
                        f"{self.program} makes {made.value.name} at "
                        f"{made.where} with {fiber.spelled}.size, and axis "
                        f"{position} of {array} has {int(extent)} cells as "
                        f"{self.origin[array]}, so {fiber.spelled} is no fiber "
                        "of it: natively it raises IndexError at every size"
                    )
            self.fibers.append(fiber)
        return typ.axes[axis]

    # {{{ one call

    def refuse(self, call: _Call, message: str) -> NoReturn:
        raise TraceError(
            f"{self.program} calls {call.kernel.__name__} at {call.where}: {message}"
        )

    def call(self, call: _Call) -> _Done:
        """One call's statements, in the program's names, and what it means.

        Its requirements on the arrays earlier calls wrote are recorded here
        and decided once every call is composed, when the sizes are final
        (:meth:`travel`).
        """
        kernel = call.kernel
        try:
            term = kernel.term
        except Exception as exc:
            raise TraceError(
                f"{self.program} calls {kernel.__name__} at {call.where}, whose "
                f"body cannot be traced: {type(exc).__name__}: {exc}"
            ) from exc
        if term.result is not None:
            self.refuse(
                call,
                f"{kernel.__name__} is a rule, which returns its value "
                f"({term.result} of its term) and writes none of its "
                "arguments, and a program's term is what its kernels write into "
                "the arrays it passes them. Call it inside a kernel, where its "
                f"call is a value: y[i] = {kernel.__name__}(...)",
            )
        label = self.label(kernel.__name__)
        names: dict[str, str] = {}
        exprs: dict[str, Any] = {}
        arrays: dict[str, str] = {}
        literals: list[tuple[str, Any]] = []
        # What the callee's contract lets share storage: two array parameters
        # it only reads, dot(x, x) (#221).
        written = {stmt.assignee.array for stmt in term.stmts}
        for param, typ in term.params:
            value = call.bound[param]
            mine = isinstance(value, ProgramValue) and value._recorder is self.recorder
            if isinstance(typ, ArrType):
                if isinstance(value, Table):
                    self.refuse(call, passed_message(param, value))
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
                    writes = [p for p in (other, param) if p in written]
                    if writes:
                        self.refuse(
                            call,
                            f"it is given {value.name} as both {other} and "
                            f"{param}, and it writes {' and '.join(writes)}: two "
                            "array parameters of a kernel may not share storage "
                            "where it writes either, which its contract checks "
                            "at every call",
                        )
                self.claim(value.name, "array", call, param)
                arrays[param] = value.name
                names[param] = value.name
            elif mine:
                if value.made:
                    self.refuse(
                        call,
                        f"it is given the array {value.name} for its scalar "
                        f"parameter {param}",
                    )
                self.claim(value.name, "scalar", call, param)
                names[param] = value.name
            elif _is_number(value):
                number = value.item() if isinstance(value, np.generic) else value
                self.integral_literal(call, param, typ, number)
                literals.append((param, typ))
                exprs[param] = number
            else:
                self.refuse(
                    call,
                    f"its scalar argument {param} is {value!r}, which is "
                    f"neither a parameter of {self.program} nor a number",
                )
        exprs.update(self.unify(call, term, names, exprs))
        for param, typ in term.params:
            if not isinstance(typ, ArrType) and param in names:
                renamed = _rename_type(typ, names, exprs)
                self.settle_type(names[param], renamed, call, param)
        for param, typ in literals:
            renamed = _rename_type(_rename_type(typ, names, exprs), {}, self.resolved)
            self.literal(call, term, param, renamed, exprs[param])
        self.settle_layout(call, term, arrays)
        wants = self.requirements_of(call, term, arrays)
        # The postcondition and the types in the program's names: its
        # parameters are what the call passed, its sizes what they unified to.
        claimed = {
            param: Var(names[param]) for param, _typ in term.params if param in names
        }
        claimed.update(exprs)
        if term.post is not None:
            # A name the postcondition leaves free that is neither a
            # parameter nor a size of the callee has no value natively, and
            # nothing of the program's is meant by it: it is spelled so that
            # no name of the program is, and isl cannot state it.
            for name in sorted(free_in(term.post) - set(claimed)):
                claimed[name] = Var(f"{name}@{label}")
        post = None if term.post is None else substitute(term.post, claimed)
        types = {
            param: _rename_type(typ, names, exprs) for param, typ in term.params
        }

        for name, values in term.tables:
            # A table is the callee's own, and the program's under a name of
            # its own: the same one for the same table read by two calls, and
            # another for another table its callee happens to name alike.
            mine = next(
                (known for known, held in self.tables.items() if held is values),
                None,
            )
            if mine is None:
                mine = self.fresh(name)
                self.tables[mine] = values
            names[name] = mine
        for name in _own_names(term):
            if name not in names and name not in exprs:
                names[name] = self.fresh(name)
        for symbol, expr in term.reflected:
            spelled = _reflected_spelling(rename_expr(expr, names, exprs))
            names[symbol] = self.fresh(spelled or symbol)
            self.reflected.append((names[symbol], rename_expr(expr, names, exprs)))

        start = self.blocks
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
                    exactness=stmt.exactness,
                    text=stmt.text,
                    spelled=stmt.spelled or stmt.inames,
                    unrolled=stmt.unrolled,
                    path=tuple(
                        (
                            rename_expr(condition, names, exprs),
                            rename_set(points, names, exprs),
                        )
                        for condition, points in stmt.path
                    ),
                )
            )
        self.blocks += top
        for stmt in stmts:
            self.writers.setdefault(
                stmt.assignee.array, f"{kernel.__name__} at {call.where}"
            )
            self.last_writer[stmt.assignee.array] = f"{kernel.__name__} at {call.where}"
        # The callee's loops and calls, in the program's names and inside
        # this call, for the handles a schedule names them by
        # (loopty.handles); a loop no statement runs in has no name here.
        binders = tuple(
            dataclasses.replace(
                binder, name=names[binder.name], scope=(label, *binder.scope)
            )
            for binder in _binders(term)
            if binder.name in names
        )
        callsites = (
            Callsite((label,), kernel.__name__, call.where, tuple(arrays.items())),
            *(
                dataclasses.replace(
                    site,
                    labels=(label, *site.labels),
                    arrays=tuple(
                        (param, names.get(array, array)) for param, array in site.arrays
                    ),
                )
                for site in term.callsites
            ),
        )
        return _Done(
            label=label,
            kernel=kernel,
            where=call.where,
            start=start,
            stmts=stmts,
            post=post,
            types=types,
            arrays=dict(arrays),
            written={stmt.assignee.array for stmt in stmts},
            wants=wants,
            binders=binders,
            callsites=callsites,
        )

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

    def integral_literal(
        self, call: _Call, param: str, sort: Any, value: Any
    ) -> None:
        """Refuse a number for a scalar of an integral sort that is no integer.

        The first half of what :meth:`literal` asks, asked before the number is
        substituted into the callee's sizes, where ``1.5`` has no meaning.
        """
        if not integral_sort(sort):
            return
        if isinstance(value, bool) or not isinstance(value, int):
            self.refuse(
                call,
                f"its scalar parameter {param} is declared {sort}, and it is "
                f"given {value!r}, which is not an integer; a value of an "
                "integral sort has to be passed as one"
                + (
                    f", so pass int({value!r}) if that is what is meant"
                    if isinstance(value, float) and value.is_integer()
                    else ""
                ),
            )

    def literal(
        self, call: _Call, term: Term, param: str, sort: Any, value: Any
    ) -> None:
        """Check a number passed for a scalar against the sort that declares it.

        A number is substituted into the callee's statements, so it is no
        argument of the program and its contract never sees it. Natively the
        callee's own contract checks it at the call, and a typing rule
        discharged the accesses it indexes on the strength of that check: a
        ``7`` passed for ``i: Fin[n]`` beside ``x[i]`` would reach C as
        ``x[7]``. So what the contract asks of a scalar is asked here, when the
        term is built: a value of an integral sort is an integer stored as one
        (:meth:`integral_literal`), and it lies in the range its sort states.

        ``sort`` is in the program's names, so ``Fin[n]`` is bounded by
        whatever ``n`` came to in the program: a number when an array of a
        fixed length was passed, and a size or a scalar of the program
        otherwise. That has no value until the program is called, and the
        number is refused, with the fix: pass it as a parameter of the
        program, which the contract of the compiled program checks.
        """
        if not integral_sort(sort):
            return
        base = sort.base if isinstance(sort, Refined) else sort
        if isinstance(base, FinType) and not isinstance(base.bound, int | np.integer):
            self.refuse(
                call,
                f"its scalar parameter {param} is declared {sort}, and it is "
                f"given the number {value}. Whether {value} is a point of "
                f"{sort} depends on {_shown(base.bound)}, which is known only "
                "when the call is made, and the compiled program is one call, "
                "whose contract checks its own parameters and nothing else. "
                f"Pass {value} as a parameter of {self.program}",
            )
        low, high = sort_bound(sort, {}) or (None, None)
        if low is not None and (value < low or (high is not None and value >= high)):
            allowed = f"{low} <= {param}" + ("" if high is None else f" < {high}")
            self.refuse(
                call,
                f"its scalar parameter {param} is given {value}, which is not "
                f"a value of {sort}: {param} has to satisfy {allowed}, as the "
                f"contract of {term.name} would say at the call",
            )

    def requirements_of(
        self, call: _Call, term: Term, arrays: Mapping[str, str]
    ) -> list[_Want]:
        """The call's requirements on the arrays earlier calls wrote.

        Natively a kernel's contract checks its arguments when it is called,
        and two of the checks are about what the cells hold, which the
        kernel's facts rest on. The cells of an array of a ``Fin[m]`` element
        sort are points of ``Fin[m]``, so an access indexed by one is in bounds
        by type, with no test in the generated code. The cells of one of the
        ``Nat`` sort are not negative, which the kernel's postcondition was
        tested on runs of. And the offsets a ragged family is read through
        are the ones its counts give, so an access inside a row is inside the
        buffer. The compiled program is one call,
        whose contract checks the program's arguments once, when it starts,
        so an array that an earlier call wrote, or that the program made,
        owes the check here: ``perm[i] = i + 1`` in one kernel is an address
        past the end of ``x`` in the next one's ``x[perm[i]]``, where the
        native run is refused by the second kernel's contract. Each such
        array is a requirement, decided or checked once the program is
        composed (:meth:`travel`).

        Offsets of the ``Nat`` sort with a layout requirement owe no ``nat``
        one: the layout says that they start at 0 and grow by the counts,
        which are naturals where the call is made (the program's contract
        checks them on entry, and a count an earlier call wrote is refused
        below), so every cell the layout covers is a natural, by induction
        over the rows, which isl cannot do; it covers every cell when the
        offsets have one more than the rows (#119).

        An array that is the row lengths of a ragged array the call reads is
        refused instead, when an earlier call wrote it: the rows are laid out
        in a buffer the program is given, and whether a count still says how
        long its row is there is a question about that buffer, which no
        hypothesis about the program's arrays can answer.
        """
        counts: dict[str, str] = {}
        offsets_of: dict[str, tuple[str, str]] = {}
        for owner, typ in term.params:
            if not isinstance(typ, ArrType):
                continue
            for size, ragged in zip(typ.axes, typ.ragged, strict=True):
                if not ragged or not isinstance(size, prim.Variable):
                    continue
                if size.name in arrays:
                    counts.setdefault(size.name, owner)
                offsets = term.offsets_of(size.name)
                if offsets is not None and offsets in arrays:
                    offsets_of.setdefault(offsets, (owner, size.name))
        wants: list[_Want] = []
        for param, typ in term.params:
            if not isinstance(typ, ArrType):
                continue
            name = arrays[param]
            writer = self.last_writer.get(name)
            if writer is None:
                if param in offsets_of:
                    self.read_through.add(name)
                continue
            element = typ.dtype.base if isinstance(typ.dtype, Refined) else typ.dtype
            if typ.domain is not None and (
                isinstance(element, FinType) or getattr(element, "name", None) == "Nat"
            ):
                self.refuse(
                    call,
                    f"it is given {name}, which {writer} writes, an array over "
                    f"{typ.domain} whose elements are {element}: what its "
                    f"contract checks of them, natively, at the call, is not "
                    "yet checked between the calls of a program. Call "
                    f"{term.name} on its own, outside the program",
                )
            if param in counts:
                owner = counts[param]
                self.refuse(
                    call,
                    f"it is given {name}, which {writer} writes, and {param} is "
                    f"the row lengths of {owner}. Natively the contract of "
                    f"{term.name} checks them against the rows {owner} is "
                    "stored in when it is called, and its facts rest on that "
                    "check: an access inside a row is inside the buffer. The "
                    "compiled program is one call, whose contract checks its "
                    "arguments when it starts, and whether a count still says "
                    "how long its row is stored is a question about the "
                    "buffer, which nothing in the program's term can check "
                    f"before {term.name} reads it. Call {term.name} on its "
                    "own, outside the program, where its contract checks it",
                )
            if isinstance(element, FinType):
                wants.append(_Want("element", param, name, writer))
            elif getattr(element, "name", None) == "Nat" and not _laid_out(
                term, typ, offsets_of.get(param)
            ):
                wants.append(_Want("nat", param, name, writer))
            if param in offsets_of:
                owner, family = offsets_of[param]
                if name not in self.read_through and name not in self.deferred:
                    self.deferred.append(name)
                wants.append(
                    _Want(
                        "layout",
                        param,
                        name,
                        writer,
                        owner=arrays[owner],
                        counts=arrays[family],
                    )
                )
                self.read_through.add(name)
        # One array given for two parameters the call only reads (#221) owes
        # its check once: both declare it alike (same_shape, settle_type).
        once: dict[tuple[str, str, str, str], _Want] = {}
        for want in wants:
            once.setdefault((want.kind, want.array, want.owner, want.counts), want)
        # A layout is checked before an element read through it, which the
        # check of the element reads its cells through.
        return sorted(once.values(), key=lambda want: want.kind != "layout")

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
        if isinstance(mine, ArrType) and isinstance(theirs, ArrType) and (
            mine.domain is not None or theirs.domain is not None
        ):
            if (
                mine.domain is None
                or theirs.domain is None
                or not _same_domain(mine.domain, theirs.domain)
            ):
                self.refuse(
                    call,
                    f"its {param} is {_shape_text(mine)}, and {name}, which it "
                    f"is given there, is {_shape_text(theirs)} as "
                    f"{self.origin[name]}; nothing says the two have the same "
                    "points",
                )
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
        self,
        call: _Call,
        term: Term,
        names: Mapping[str, str],
        exprs: Mapping[str, Any],
    ) -> dict[str, Any]:
        """What each size of the callee is, in the program's sizes.

        Every axis of an array the program has already typed is an equation
        between the callee's size expression and the program's. A bare size of
        the callee is solved by what the program has; a size of the program
        standing alone on one side is replaced by the other side everywhere
        (:meth:`equate`); two expressions have to be equal as isl sees them,
        or are refused. A size of the callee nothing determines becomes a size
        of the program, under its own name where that is free.

        ``names`` and ``exprs`` are what the call's parameters became: the
        program's array or scalar for each, or the number passed for a scalar.
        A scalar parameter can size an array (``m: Nat`` beside
        ``x: Arr[Fin[m], Real]``), and that axis is the program's scalar, or
        the number, as it is in the statements' domains.
        """
        params = dict(term.params)
        arrays = {
            param: names[param]
            for param, typ in term.params
            if isinstance(typ, ArrType)
        }
        fresh = {name for name in arrays.values() if name not in self.types}
        sizes = [name for name in term.sizes if name not in params]
        for _, typ in term.params:
            # The bound of a Fin sort names a size too, an element sort's
            # (``nnz`` in ``off: Arr[Fin[n + 1], Fin[nnz + 1]]``) or a
            # scalar's, even where no axis of the callee is that long: it is
            # renamed apart like any other, or two calls that each name a
            # buffer ``nnz`` would be given one size.
            bound = _fin_bound(typ.dtype if isinstance(typ, ArrType) else typ)
            for name in _names_in(bound):
                if name not in params and name not in sizes:
                    sizes.append(name)
            if not isinstance(typ, ArrType):
                continue
            if typ.domain is not None:
                for name in sorted(typ.domain.size_names()):
                    if name not in params and name not in sizes:
                        sizes.append(name)
                continue
            for axis, ragged in zip(typ.axes, typ.ragged, strict=True):
                if ragged:
                    continue
                for name in _names_in(axis):
                    if name not in params and name not in sizes:
                        sizes.append(name)
        marks = {name: _CALLEE + name for name in sizes}
        marked = {**marks, **names}
        bound: dict[str, Any] = {}
        pending: list[tuple[str, int, Any, bool, Any, bool]] = []
        for param, typ in term.params:
            if not isinstance(typ, ArrType) or arrays[param] not in self.types:
                continue
            known = self.types[arrays[param]]
            if typ.domain is not None or known.domain is not None:
                pending.extend(
                    self.domain_equations(
                        call, param, arrays[param], typ, known, marked, exprs
                    )
                )
                continue
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
                        rename_expr(axis, marked, exprs),
                        ragged,
                        known.axes[k],
                        known.ragged[k],
                    )
                )
            # The bound of a Fin element sort is a size too: an offsets array
            # of Fin[nnz + 1] is the same array in every call it is passed to.
            mine_bound, theirs_bound = _fin_bound(typ.dtype), _fin_bound(known.dtype)
            if mine_bound is not None and theirs_bound is not None:
                pending.append(
                    (
                        param,
                        None,
                        rename_expr(mine_bound, marked, exprs),
                        False,
                        theirs_bound,
                        False,
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
            bound[loose] = Var(self.new_size(loose[len(_CALLEE) :]))
        for name in sizes:
            if marks[name] not in bound:
                bound[marks[name]] = Var(self.new_size(name))
        out = {name: bound[marks[name]] for name in sizes}
        first: dict[str, str] = {}
        for param, typ in term.params:
            if isinstance(typ, ArrType):
                renamed = _rename_type(typ, names, {**exprs, **out})
                given = first.setdefault(arrays[param], param)
                if given != param and arrays[param] in fresh:
                    # The array was typed by an earlier parameter of this
                    # call, against which no equation was written.
                    self.same_shape(call, given, param, arrays[param], renamed)
                self.settle_type(arrays[param], renamed, call, param)
        return out

    def same_shape(
        self, call: _Call, first: str, param: str, name: str, typ: ArrType
    ) -> None:
        """Make two parameters of one call that are given one array agree on it.

        A kernel that only reads two array parameters may be given one array
        for both, ``dot(u, u)`` (#221), and natively both are its shape:
        ``a: Arr[Fin[n], Real]`` and ``b: Arr[Fin[m], Real]`` given ``u`` make
        ``n`` and ``m`` its length. So the second parameter's axes, binder
        bounds and ``Fin`` element bound are equated with the first's, which
        typed the array, eliminating a size where one stands alone, and two
        that cannot be shown to agree are refused.
        """
        from loopty.domain import Union

        known = _rename_type(self.types[name], {}, self.resolved)
        mine = _rename_type(typ, {}, self.resolved)
        refused = f"it is given {name} for both {first} and {param}"
        pairs: list[tuple[Any, Any]] = []
        if mine.domain is not None or known.domain is not None:
            if (
                mine.domain is None
                or known.domain is None
                or type(mine.domain) is not type(known.domain)
                or mine.domain.ndim != known.domain.ndim
                or (
                    isinstance(mine.domain, Union)
                    and len(mine.domain.pieces) != len(known.domain.pieces)
                )
            ):
                self.refuse(
                    call,
                    f"{refused}, which it declares {_shape_text(known)} and "
                    f"{_shape_text(mine)}; the two are not one domain",
                )
            pieces = (
                list(zip(mine.domain.pieces, known.domain.pieces, strict=True))
                if isinstance(mine.domain, Union)
                else [(mine.domain, known.domain)]
            )
            for one, other in pieces:
                renamed = dict(zip(one.names, other.names, strict=True))
                pairs.extend(
                    (rename_expr(bound, renamed, {}), theirs)
                    for (_a, bound), (_b, theirs) in zip(
                        one.binders, other.binders, strict=True
                    )
                )
        else:
            if len(mine.axes) != len(known.axes):
                self.refuse(
                    call,
                    f"{refused}, which it declares of {len(known.axes)} and "
                    f"{len(mine.axes)} axes: one array has one shape",
                )
            if mine.ragged != known.ragged:
                self.refuse(
                    call,
                    f"{refused}, which it declares ragged at {_ragged_axes(known)} "
                    f"and at {_ragged_axes(mine)}: one array has one shape",
                )
            pairs.extend(zip(mine.axes, known.axes, strict=True))
        bounds = _fin_bound(mine.dtype), _fin_bound(known.dtype)
        if bounds[0] is not None and bounds[1] is not None:
            pairs.append(bounds)
        for left, right in pairs:
            left, right = self.resolve(left), self.resolve(right)
            if _same(left, right) or self.equate(left, right):
                continue
            self.refuse(
                call,
                f"{refused}, which it declares {_shown(right)} and "
                f"{_shown(left)} long where one array has one length, and "
                "nothing says the two agree",
            )

    def domain_equations(
        self,
        call: _Call,
        param: str,
        name: str,
        typ: ArrType,
        known: ArrType,
        marked: Mapping[str, str],
        exprs: Mapping[str, Any],
    ) -> list[tuple[str, int | None, Any, bool, Any, bool]]:
        """The equations between two domains' binder bounds, one per axis.

        The callee's binders are read as the ones the program's type has, axis
        by axis, so that a bound naming an earlier binder (``Fin[i + 1]``) is
        compared with the program's; whether the two are the same set of
        points, constraints included, is asked once the sizes are settled
        (:meth:`settle_type`).
        """
        from loopty.domain import Union

        mine, theirs = typ.domain, known.domain
        if mine is None or theirs is None:
            self.refuse(
                call,
                f"its {param} is {_shape_text(typ)}, and {name}, which it is "
                f"given there, is {_shape_text(known)} as {self.origin[name]}",
            )
        if type(mine) is not type(theirs) or mine.ndim != theirs.ndim or (
            isinstance(mine, Union) and len(mine.pieces) != len(theirs.pieces)
        ):
            self.refuse(
                call,
                f"its {param} is over {mine}, and {name}, which it is given "
                f"there, is over {theirs} as {self.origin[name]}; the two are "
                "not one domain",
            )
        pairs = (
            list(zip(mine.pieces, theirs.pieces, strict=True))
            if isinstance(mine, Union)
            else [(mine, theirs)]
        )
        out = []
        for first, second in pairs:
            renamed = {**marked, **dict(zip(first.names, second.names, strict=True))}
            for k, ((_binder, bound), (_other, other)) in enumerate(
                zip(first.binders, second.binders, strict=True)
            ):
                out.append(
                    (param, k, rename_expr(bound, renamed, exprs), False, other, False)
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
        item: tuple[str, int | None, Any, bool, Any, bool],
        bound: dict[str, Any],
    ) -> bool:
        """Settle one axis equation, or say that it has to wait.

        ``k`` is the axis, or ``None`` for the bound of the array's ``Fin``
        element sort, which is settled the way an axis is.
        """
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
        if k is None:
            self.refuse(
                call,
                f"it declares the elements of its {param} points of "
                f"Fin({_shown(mine)}), and the elements of {name}, which it is "
                f"given there, are points of Fin({_shown(theirs)}) as "
                f"{self.origin[name]}; nothing says the two agree",
            )
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
        if mine.domain is not None or theirs.domain is not None:
            self.shaped_over(name, mine, _rename_type(theirs, {}, self.resolved),
                             f"like {like}", where)
            return
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

    def shaped_over(
        self, name: str, mine: ArrType, theirs: ArrType, how: str, where: str
    ) -> None:
        """Unify an array the program made over a domain with the one it is made as.

        The binders' bounds are equated axis by axis, which may settle a size
        of the program, and then the two have to be one set of points.
        """
        mine = _rename_type(mine, {}, self.resolved)
        if mine.domain is None or theirs.domain is None or (
            mine.domain.ndim != theirs.domain.ndim
        ):
            raise TraceError(
                f"{self.program} makes {name} {how} at {where}, "
                f"{_shape_text(theirs)}, and {self.origin[name]} wants it "
                f"{_shape_text(mine)}"
            )
        from loopty.domain import Union

        if not isinstance(mine.domain, Union) and not isinstance(theirs.domain, Union):
            for (_a, first), (_b, second) in zip(
                mine.domain.binders, theirs.domain.binders, strict=True
            ):
                renamed = dict(zip(mine.domain.names, theirs.domain.names, strict=True))
                first = self.resolve(rename_expr(first, renamed, {}))
                second = self.resolve(second)
                if not _same(first, second):
                    self.equate(first, second)
            mine = _rename_type(mine, {}, self.resolved)
            theirs = _rename_type(theirs, {}, self.resolved)
        if not _same_domain(mine.domain, theirs.domain):
            raise TraceError(
                f"{self.program} makes {name} {how} at {where}, over "
                f"{theirs.domain}, and {self.origin[name]} wants it over "
                f"{mine.domain}; nothing says the two have the same points"
            )

    def stored_like(self, name: str, made: _Made) -> None:
        """Say what dtype an array the program made has natively, or refuse it.

        ``Arr.zeros_like(u)`` is natively an array of ``u``'s dtype, or of the
        ``dtype`` it is given, and the program's temporary has the element
        sort the kernels it is passed to declare, stored as the lowering
        stores that sort. The native array has to hold what the compiled one
        holds (:func:`loopty.contract.native_storage`), or the native run
        truncates (into integers), rounds (into ``float32``) or drops an
        imaginary part (into reals) where the compiled run does not. A
        ``dtype`` given, here or to the array it was made like, is checked
        now. One left to a parameter is whatever the parameter is called
        with: the pair goes into :attr:`~loopty.term.Term.temporaries_like`,
        and is checked when the compiled program is run
        (:func:`loopty.contract.inherited_storage`).
        """
        sort = self.types[name].dtype
        want = native_storage(sort)
        if want is None:
            return
        value: ProgramValue | None = made.value
        while value is not None and value.made:
            here = self.made.get(value.name)
            if here is not None and here.dtype is not None:
                try:
                    got = np.dtype(here.dtype)
                except TypeError:
                    got = None
                if got is not None and holds_natively(sort, got):
                    return
                raise TraceError(
                    f"{self.program} makes {name} at {made.where} as an array of "
                    f"{got if got is not None else here.dtype!r}, and "
                    f"{self.origin[name]} declares its elements "
                    f"{_shown(sort)}, which the native run has to store as "
                    f"{storage_wanted(sort)} to hold them as the compiled "
                    f"program does: the native run would compute {name} "
                    f"otherwise than the compiled one. "
                    f"Pass Arr.zeros_like dtype={want}"
                )
            value = value.like
        if value is not None and value.name in self.types:
            self.inherits.append((name, value.name))

    def stored_wide(self, name: str, made: _Made) -> None:
        """Refuse a ``dtype`` narrower than 64 bits for an array a check reads.

        The compiled program stores an integral array a checked point reads
        in 64 bits (:func:`loopty.contract.array_storage`), so that the check
        reads what the earlier call wrote; a native array of fewer bits
        narrows it first, and the native run would check another value
        (#128). A ``dtype`` given to the ``Arr.zeros_like``, or to the array
        it was made like, is checked here; one left to a parameter is checked
        when the compiled program is run
        (:func:`loopty.contract.checked_storage`).
        """
        if not integral_sort(self.types[name].dtype):
            return
        value: ProgramValue | None = made.value
        while value is not None and value.made:
            here = self.made.get(value.name)
            if here is not None and here.dtype is not None:
                try:
                    got = np.dtype(here.dtype)
                except TypeError:
                    return
                if got.kind == "i" and got.itemsize >= CHECKED_STORAGE.itemsize:
                    return
                raise TraceError(
                    f"{self.program} makes {name} at {made.where} as an array of "
                    f"{got}, and a checked point of the program reads {name} "
                    "between two calls: the compiled program stores it in 64 "
                    "bits, so that the check reads what the earlier call wrote, "
                    "and the native array would narrow what is written into it, "
                    "so the two runs would check two values. Pass "
                    f"Arr.zeros_like dtype={CHECKED_STORAGE}"
                )
            value = value.like

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
        if not any(value.name in self.writers for value in self.recorder.params):
            raise TraceError(
                f"the program {self.program} writes none of its parameters. Its "
                "term lowers to one kernel, whose results are what it writes "
                "into its parameters, so the compiled program would compute "
                "nothing anybody sees, and comparing it with the native one "
                "would compare nothing; write the result into a parameter"
            )
        stmts, requirements, scopes, flags = self.travel()
        for requirement in requirements:
            if requirement.flag is not None and requirement.array in self.made:
                self.stored_wide(requirement.array, self.made[requirement.array])
        resolved = self.resolved
        return Term(
            name=self.program,
            params=tuple((name, _rename_type(t, {}, resolved)) for name, t in params),
            sizes=tuple(sorted(self.sizes)),
            stmts=tuple(self.resolved_stmt(stmt) for stmt in stmts),
            post=self.postcondition(),
            reflected=tuple(
                (name, self.resolve(expr)) for name, expr in self.reflected
            ),
            temporaries=(
                *((name, _rename_type(t, {}, resolved)) for name, t in temporaries),
                *flags,
            ),
            offsets=tuple(self.layout.items()),
            where=self.recorder.where,
            temporaries_like=tuple(self.inherits),
            requirements=tuple(requirements),
            scopes=tuple(scopes),
            deferred_offsets=tuple(self.deferred),
            fibers=tuple(
                dataclasses.replace(
                    fiber,
                    extents=tuple(self.resolve(extent) for extent in fiber.extents),
                )
                for fiber in self.fibers
            ),
            tables=tuple(self.tables.items()),
            binders=tuple(
                binder
                for slot in self.slots
                if isinstance(slot, _Done)
                for binder in slot.binders
            ),
            callsites=tuple(
                site
                for slot in self.slots
                if isinstance(slot, _Done)
                for site in slot.callsites
            ),
        )

    def postcondition(self) -> Any:
        """The program's return annotation, in the sizes still standing, or ``None``.

        It is a claim about the program's parameters once the body has run,
        written in their names and in the sizes the kernels it calls name,
        ``n`` and ``ne``; a size unification eliminated is put in as what it
        turned out to be. :meth:`loopty.kernel.Program.facts` states it, and
        says so when it names anything else.
        """
        annotations = getattr(self.declared, "annotations", None) or {}
        claim = annotations.get("return")
        if claim is None:
            return None
        return self.resolve_claim(claim)

    # {{{ facts that travel

    def travel(
        self,
    ) -> tuple[list[Stmt], list[Requirement], list[Scope], list[tuple[str, ArrType]]]:
        """The statements in order, each call's requirements decided or checked.

        The events are walked in order, keeping the hypotheses that hold
        between them: a call's postcondition holds after it until a later
        call writes an array it mentions, and so do the zeros an
        ``Arr.zeros_like`` makes. At each call, those, the types the
        program's contract checks of the arrays nothing has written yet, and
        the theorems the program cites, instantiated at them, are what its
        requirements are decided under (:meth:`requirement`). A requirement
        isl does not decide is checked: its check statements come before the
        call's, each in a block of its own, and every statement after them is
        guarded by its flag (:meth:`guarded`). The hypotheses at each call,
        with the call's own argument types, are its :class:`Scope`.

        Returns the statements, the requirements, the scopes and the flag
        temporaries.
        """
        valid: list[Hypothesis] = []
        written: set[str] = set()
        flags: list[str] = []
        flag_types: list[tuple[str, ArrType]] = []
        shift = 0
        stmts: list[Stmt] = []
        requirements: list[Requirement] = []
        scopes: list[Scope] = []
        for slot in self.slots:
            if not isinstance(slot, _Done):
                made, block = slot
                name = made.value.name
                if name in self.types:
                    zeros = self.zeros(name, self.types[name], made, block + shift)
                    stmts.append(self.guarded(zeros, flags))
                    valid = [h for h in valid if name not in h.mentions]
                    zeros_held = self.zeros_hypothesis(name, made)
                    if zeros_held is not None:
                        valid.append(zeros_held)
                    written.add(name)
                continue
            offered, notes = self.offered(valid, written)
            types_here: list[Hypothesis] = []
            for want in slot.wants:
                requirement, checks = self.requirement(slot, want, offered, notes)
                requirements.append(requirement)
                for check in checks:
                    order = (slot.start + shift, *(0,) * len(check.inames))
                    stmts.append(
                        self.guarded(dataclasses.replace(check, order=order), flags)
                    )
                    shift += 1
                if requirement.flag is not None:
                    flags.append(requirement.flag)
                    flag_types.append(
                        (
                            requirement.flag,
                            ArrType(axes=(1,), dtype=Nat, ragged=(False,)),
                        )
                    )
                types_here.append(
                    Hypothesis(
                        claim=requirement.claim,
                        source=f"the requirement on {want.array} where {slot.label} "
                        "is called",
                        rests_on=(
                            self.requirement_id(slot.label, want.kind, want.param),
                        ),
                        mentions=frozenset(
                            {
                                want.array,
                                *([want.counts] if want.counts else []),
                                *self.arrays_in(requirement.claim),
                            }
                        ),
                    )
                )
            ids = []
            for stmt in slot.stmts:
                moved = dataclasses.replace(
                    stmt, order=(stmt.order[0] + shift, *stmt.order[1:])
                )
                stmts.append(self.guarded(moved, flags))
                ids.append(stmt.id)
            # What held when the call started holds of the arrays it reads
            # throughout, and of those it writes only until it writes them.
            scopes.append(
                Scope(
                    call=slot.label,
                    kernel=slot.kernel.__name__,
                    where=slot.where,
                    statements=tuple(ids),
                    hypotheses=tuple(
                        hypothesis
                        for hypothesis in (*offered, *types_here)
                        if not (hypothesis.mentions & slot.written)
                    ),
                )
            )
            written |= slot.written
            valid = [h for h in valid if not (h.mentions & slot.written)]
            if slot.post is not None:
                identifier = self.restatement_id(slot.kernel)
                self.postconditions[identifier] = slot.kernel
                # The postcondition is tested on the callee's body, and the
                # program runs the callee's term: it says what the term leaves
                # only where the two compute alike.
                faithful = self.faithful_id(slot.kernel)
                self.faithfuls[faithful] = slot.kernel
                claim = self.resolve_claim(slot.post)
                # The postcondition was tested on the runs its contract let
                # in. Here the contract held at the call: what nothing wrote
                # before it, the program's contract checked on entry, and
                # what an earlier call wrote is a requirement of the call,
                # decided or checked (#119).
                valid.append(
                    Hypothesis(
                        claim=claim,
                        source=f"the postcondition of {slot.kernel.__name__}, "
                        f"after {slot.label} at {slot.where}",
                        rests_on=(identifier, faithful),
                        mentions=frozenset(self.arrays_in(claim)),
                    )
                )
        return stmts, requirements, scopes, flag_types

    def weakly_supported(self, used: Sequence[Hypothesis]) -> list[str]:
        """Why the facts ``used`` rest on are not all at least ``tested``.

        A requirement isl decided under hypotheses is skipped by the compiled
        program, which trusts what the hypotheses rest on. A postcondition its
        kernel's native runs refuted, or that nothing tested, is no ground for
        that, nor one of a kernel whose term is not shown to compute what its
        body does, nor a cited theorem that the property tester does not pass,
        or an axiom: the requirement keeps its checked point (#115).
        """
        out: list[str] = []
        for hypothesis in used:
            for identifier in hypothesis.rests_on:
                why = self.standing(identifier)
                if why is not None and why not in out:
                    out.append(why)
        return out

    def standing(self, identifier: str) -> str | None:
        """``None`` when the fact ``identifier`` is at least ``tested``, else why."""
        if identifier in self.standings:
            return self.standings[identifier]
        why = self.look_up(identifier)
        self.standings[identifier] = why
        return why

    def look_up(self, identifier: str) -> str | None:
        """The standing of one fact a hypothesis rests on (see :meth:`standing`)."""
        from lanky.ledger import STATUS_STRENGTH

        from loopty.faithful import KIND as FAITHFUL

        tables = ((self.postconditions, "postcondition"), (self.faithfuls, FAITHFUL))
        for table, kind in tables:
            kernel = table.get(identifier)
            if kernel is None:
                continue
            try:
                facts = kernel.facts()
            except Exception as exc:  # noqa: BLE001 - then it stands on nothing
                return (
                    f"the facts of {kernel.__name__} could not be established: "
                    f"{type(exc).__name__}: {exc}"
                )
            fact = next((f for f in facts if f.kind == kind), None)
            if fact is not None and STATUS_STRENGTH[fact.status] >= 1:
                return None
            status = fact.status.value if fact is not None else "assumed"
            if kind == FAITHFUL:
                return (
                    f"that the term of {kernel.__name__} computes what its body "
                    f"computes, on which its postcondition was tested, is {status}"
                )
            return f"the postcondition of {kernel.__name__} is {status}"
        for theorem in self.uses:
            if theorem.fact_id == identifier:
                return _theorem_standing(theorem)
        return f"{identifier} is a fact whose status the program does not know"

    @staticmethod
    def faithful_id(kernel: Any) -> str:
        """The id of ``kernel``'s ``trace-faithful`` fact, as its own facts give it."""
        from lanky.ledger import fact_id

        from loopty.faithful import KIND as FAITHFUL

        return fact_id(
            FAITHFUL,
            getattr(kernel, "qualname", kernel.__name__),
            module=getattr(kernel, "module", None),
            line=getattr(kernel, "line", None),
        )

    def restatement_id(self, kernel: Any) -> str:
        """The id of the program's restatement of ``kernel``'s postcondition."""
        from loopty.typing import restatement_id

        return restatement_id(
            self.owner["owner"],
            kernel.definition,
            module=self.owner["module"],
            line=self.owner["line"],
        )

    def requirement_id(self, label: str, kind: str, param: str) -> str:
        """The id of the requirement of the call ``label`` on its ``param``."""
        from loopty.typing import requirement_id

        return requirement_id(
            self.owner["owner"],
            label,
            kind,
            param,
            module=self.owner["module"],
            line=self.owner["line"],
        )

    def resolve_claim(self, claim: Any) -> Any:
        """A proposition in the sizes still standing."""
        return substitute(claim, self.resolved) if self.resolved else claim

    def program_type(self, name: str) -> Any:
        """The type of a program array or scalar, in the sizes still standing."""
        return _rename_type(self.types[name], {}, self.resolved)

    def arrays_in(self, claim: Any) -> set[str]:
        """The program arrays a proposition names, in a binder's sort too.

        ``all(perm[i] == 0 for i in Fin[lim[0]])`` is about ``lim`` as much
        as about ``perm``, and a call that writes either retires it.
        """
        return {name for name in mentioned(claim) if self.kinds.get(name) == "array"}

    def integral_arrays(self) -> set[str]:
        """The program arrays whose cells are integers, which isl may read."""
        out = set()
        for name, typ in self.types.items():
            if isinstance(typ, ArrType) and integral_sort(typ.dtype):
                out.add(name)
        return out

    def known_names(self) -> set[str]:
        """The names a hypothesis may compare: the sizes and the integral scalars."""
        out = set(self.sizes)
        for name, typ in self.types.items():
            if not isinstance(typ, ArrType) and integral_sort(typ):
                out.add(name)
        return out

    def cells_hypothesis(
        self, array: str, body: Any, source: str, mentions: set[str]
    ) -> Hypothesis | None:
        """``body`` of every cell of ``array``, as a universal over its cells.

        ``body`` maps the cell, ``array[c0, c1]``, to the proposition about it.
        """
        typ = self.program_type(array)
        if not isinstance(typ, ArrType) or typ.domain is not None:
            return None
        names = [f"c{k}" for k in range(len(typ.axes))]
        binders = []
        for k, (axis, ragged) in enumerate(zip(typ.axes, typ.ragged, strict=True)):
            if ragged:
                if k == 0 or not isinstance(axis, prim.Variable):
                    return None
                bound = Subscript(Var(axis.name), Var(names[k - 1]))
                mentions = {*mentions, axis.name}
            else:
                bound = axis
            binders.append((Var(names[k]), FinType(bound)))
        index = tuple(Var(name) for name in names)
        cell = Subscript(Var(array), index[0] if len(index) == 1 else index)
        return Hypothesis(
            claim=Forall(tuple(binders), body(cell)),
            source=source,
            mentions=frozenset({array, *mentions}),
        )

    def entry_types(self, written: set[str]) -> list[Hypothesis]:
        """What the program's contract checks of what nothing has written yet.

        Every element of an argument of a ``Fin[m]`` sort is a point of it,
        and one of ``Nat`` is not negative, until a call writes it; so is a
        scalar argument of those sorts, which nothing writes.
        """
        out: list[Hypothesis] = []
        for value in self.recorder.params:
            name = value.name
            if name not in self.types or name in written:
                continue
            typ = self.program_type(name)
            sort = typ.dtype if isinstance(typ, ArrType) else typ
            bound = _fin_bound(sort)
            base = sort.base if isinstance(sort, Refined) else sort
            if bound is not None:

                def body(cell: Any, bound: Any = bound) -> Any:
                    return LogicalAnd(
                        (Comparison(0, "<=", cell), Comparison(cell, "<", bound))
                    )

            elif getattr(base, "name", None) == "Nat":

                def body(cell: Any) -> Any:
                    return Comparison(cell, ">=", 0)

            else:
                continue
            source = (
                f"the type of {name}, which the contract of {self.program} "
                "checks when it starts"
            )
            if isinstance(typ, ArrType):
                found = self.cells_hypothesis(name, body, source, set())
                if found is not None:
                    out.append(found)
            else:
                out.append(Hypothesis(body(Var(name)), source))
        return out

    def zeros_hypothesis(self, name: str, made: _Made) -> Hypothesis | None:
        """Every cell of an array the program made is zero, until a call writes it.

        ``None`` for one over a domain, whose cells no hypothesis isl reads
        ranges over; the zeros its statement stores are what a definedness
        fact reads (:func:`loopty.typing.definedness_facts`).
        """
        return self.cells_hypothesis(
            name,
            lambda cell: Comparison(cell, "==", 0),
            f"the zeros the Arr.zeros_like at {made.where} starts {name} with",
            set(),
        )

    def offered(
        self, valid: Sequence[Hypothesis], written: set[str]
    ) -> tuple[list[Hypothesis], list[str]]:
        """The hypotheses at a call, and why a cited theorem gave none.

        ``valid`` and the types of what nothing has written yet, then each
        theorem the program cites, instantiated at them
        (:func:`loopty.hypotheses.theorem_instances`). A family of a theorem
        stands for a program array only where the array's cells are points of
        the family's sort when the call is made: an array of one axis as long
        as the family's domain, whose element sort the program's contract
        checks on entry and nothing has written since, or a family of
        ``Int``.
        """
        base = [*valid, *self.entry_types(written)]
        out = list(base)
        notes: list[str] = []

        def applicable(array: str, domain: Any, codomain: Any) -> str | None:
            typ = self.program_type(array) if array in self.types else None
            if (
                not isinstance(typ, ArrType)
                or typ.domain is not None
                or len(typ.axes) != 1
                or typ.ragged[0]
            ):
                return f"{array} is no array of one dense axis"
            if not isinstance(domain, FinType) or not _same(
                self.resolve(domain.bound), typ.axes[0]
            ):
                return (
                    f"{array} has {_shown(typ.axes[0])} cells, and the family it "
                    f"would stand for ranges over {domain}"
                )
            element = typ.dtype.base if isinstance(typ.dtype, Refined) else typ.dtype
            if not integral_sort(element):
                return f"the elements of {array} are not integers"
            wanted = getattr(codomain, "name", None)
            if wanted == "Int":
                return None
            if array in written:
                return (
                    f"{array} is written before the call, and nothing establishes "
                    f"that its cells are points of {codomain}, which the family "
                    "it would stand for ranges over"
                )
            if wanted == "Nat" and (
                isinstance(element, FinType) or getattr(element, "name", None) == "Nat"
            ):
                return None
            if isinstance(codomain, FinType) and isinstance(element, FinType):
                if _same(self.resolve(element.bound), codomain.bound):
                    return None
            return (
                f"the elements of {array} are points of {typ.dtype}, and the family "
                f"it would stand for ranges over {codomain}"
            )

        def nonnegative(expr: Any) -> bool:
            form = linear(expr)
            if form is None:
                return False
            names, constant = form
            return constant >= 0 and all(
                coefficient >= 0 and name in self.sizes
                for name, coefficient in names.items()
            )

        for theorem in self.uses:
            instances, reasons = theorem_instances(
                theorem, base, applicable, nonnegative
            )
            out.extend(instances)
            if not instances:
                notes.extend(reasons)
        return out, notes

    def requirement(
        self,
        slot: _Done,
        want: _Want,
        offered: Sequence[Hypothesis],
        notes: Sequence[str],
    ) -> tuple[Requirement, list[Stmt]]:
        """One requirement, decided under ``offered``, or its check statements.

        An element requirement is about every cell of the array: it is a
        point of the sort the callee declares, ``0 <= perm[i] < n``. So is a
        ``nat`` one, ``src[i] >= 0``. A layout
        requirement is what the contract compares the offsets with, the
        offsets the counts give the rows: ``off[0] == 0``, and ``off[q] ==
        off[q - 1] + cnt[q - 1]`` for ``1 <= q <= n``. Its check is two
        statements, one for the start and one for the rows.
        """
        kernel = slot.kernel.__name__
        array = want.array
        typ = self.program_type(array)
        where = slot.where
        checks: list[tuple[str, tuple[str, ...], isl.Set, Any]] = []
        if want.kind in ("element", "nat"):
            dims = tuple(self.fresh("i") for _ in typ.axes)
            bounds = [
                Subscript(Var(axis.name), Var(dims[k - 1])) if ragged else axis
                for k, (axis, ragged) in enumerate(
                    zip(typ.axes, typ.ragged, strict=True)
                )
            ]
            table = self.reflections(dims)
            domain = domain_set(dims, bounds, reflections=table)
            index = tuple(Var(d) for d in dims)
            cell = Subscript(Var(array), index[0] if len(index) == 1 else index)
            if want.kind == "element":
                sort = _rename_type(slot.types[want.param].dtype, {}, self.resolved)
                bound = _fin_bound(sort)

                def holds(cell: Any, bound: Any = bound) -> Any:
                    return LogicalAnd(
                        (Comparison(0, "<=", cell), Comparison(cell, "<", bound))
                    )

                breaks = LogicalOr(
                    (Comparison(cell, "<", 0), Comparison(cell, ">=", bound))
                )
                what = f"the elements of {array} are points of {sort}"
                failure = f"an element of {array} is not a point of {sort}"
            else:

                def holds(cell: Any) -> Any:
                    return Comparison(cell, ">=", 0)

                breaks = Comparison(cell, "<", 0)
                what = f"the elements of {array} are naturals"
                failure = f"an element of {array} is not a natural"
            goal = holds(cell)
            checks.append((f"{slot.label}.check.{want.param}", dims, domain, breaks))
            found = self.cells_hypothesis(array, holds, "", set())
            claimed = found.claim if found is not None else goal
            used = set(domain.get_var_names(isl.dim_type.param))
            reflected = {name: expr for name, expr in table.items() if name in used}
        else:
            counts = want.counts
            owner = self.program_type(want.owner)
            rows = owner.axes[0]
            q = self.fresh("q")
            domain = domain_set((q,), (_plus(rows, 1),))
            at = Var(q)
            before = _plus(at, -1)
            goal = LogicalAnd(
                (
                    LogicalOr(
                        (
                            Comparison(at, ">", 0),
                            Comparison(Subscript(Var(array), 0), "==", 0),
                        )
                    ),
                    LogicalOr(
                        (
                            Comparison(at, "==", 0),
                            Comparison(
                                Subscript(Var(array), at),
                                "==",
                                Subscript(Var(array), before)
                                + Subscript(Var(counts), before),
                            ),
                        )
                    ),
                )
            )
            r = self.fresh("r")
            rv = Var(r)
            stem = f"{slot.label}.check.{want.param}"
            checks.append(
                (
                    f"{stem}.start",
                    (),
                    isl.Set("{ [] }"),
                    Comparison(Subscript(Var(array), 0), "!=", 0),
                )
            )
            checks.append(
                (
                    f"{stem}.rows",
                    (r,),
                    domain_set((r,), (rows,)),
                    Comparison(
                        Subscript(Var(array), _plus(rv, 1)),
                        "!=",
                        Subscript(Var(array), rv) + Subscript(Var(counts), rv),
                    ),
                )
            )
            what = (
                f"{array} holds the offsets the counts in {counts} give the rows "
                f"of {want.owner} ({array}[0] == 0 and {array}[r + 1] == "
                f"{array}[r] + {counts}[r])"
            )
            rr = Var("r")
            claimed = LogicalAnd(
                (
                    Comparison(Subscript(Var(array), 0), "==", 0),
                    Forall(
                        ((rr, FinType(rows)),),
                        Comparison(
                            Subscript(Var(array), _plus(rr, 1)),
                            "==",
                            Subscript(Var(array), rr) + Subscript(Var(counts), rr),
                        ),
                    ),
                )
            )
            failure = (
                f"{array} does not hold the offsets the counts in {counts} give "
                f"the rows of {want.owner}"
            )
            reflected = {}
        statement = (
            f"{what} where {slot.label} is called at {where}, after {want.writer} "
            f"wrote {array}"
        )
        outcome = discharge(
            domain,
            goal,
            offered,
            integral=self.integral_arrays(),
            known=self.known_names(),
            nonneg=set(self.sizes),
            reflected=reflected,
            description=(
                f"points at which a cell of {array} breaks the requirement of "
                f"{slot.label} where the hypotheses used hold"
            ),
        )
        common = {
            "call": slot.label,
            "kernel": kernel,
            "definition": slot.kernel.definition,
            "where": where,
            "param": want.param,
            "array": array,
            "kind": want.kind,
            "writer": want.writer,
            "statement": statement,
            "claim": claimed,
            "offered": tuple(offered),
        }
        weak = self.weakly_supported(outcome.used) if outcome.decided else []
        if outcome.decided and not weak:
            return (
                Requirement(**common, used=outcome.used, question=outcome.question),
                [],
            )
        if weak:
            # Decided, and the fact rests on something no run has borne out:
            # the compiled program does not skip the check on its strength.
            reason = (
                "decided under "
                + "; ".join(hypothesis.source for hypothesis in outcome.used)
                + ", and checked all the same, since "
                + "; ".join(weak)
            )
            common["used"] = outcome.used
            common["question"] = outcome.question
        elif outcome.contradicting:
            # A false hypothesis decides everything; the check decides nothing
            # it does not see.
            reason = outcome.reason
        elif offered:
            reason = (
                "the hypotheses that held at the call leave room for a cell that "
                f"breaks it: {outcome.reason}"
            )
        else:
            reason = (
                f"nothing that held at the call says what {want.writer} left "
                f"in {array}"
            )
        if outcome.unstated:
            shown = list(outcome.unstated[:3])
            more = len(outcome.unstated) - len(shown)
            reason += (
                "; isl cannot state "
                + "; ".join(shown)
                + (f"; and {more} more" if more else "")
                + ", which is read as saying nothing"
            )
        if notes and not weak:
            reason += "; " + "; ".join(notes)
        spelled = re.sub(r"\W", "_", slot.label)
        flag = self.fresh(f"{spelled}_{want.param}_ok")
        why = (
            f"what decided it rests on facts no run bore out ({'; '.join(weak)})"
            if weak
            else "nothing decided what it left there"
        )
        message = (
            f"the compiled program {self.program} stops before {slot.label} at "
            f"{where}: {failure}, which the contract of {kernel} checks when it "
            f"is called. {want.writer} wrote {array} before the call, and {why}, "
            f"so the program checks it where the native {kernel} is refused"
        )
        statements = []
        for check_id, inames, check_domain, breaks in checks:
            self.register(check_domain)
            statements.append(
                Stmt(
                    id=check_id,
                    inames=tuple(inames),
                    domain=check_domain,
                    assignee=Access(flag, (0,)),
                    expr=1,
                    kind="assign",
                    guard=breaks,
                    where=where,
                    order=(0, *(0,) * len(inames)),
                    unnarrowed=((render(breaks), _NOT_AFFINE),),
                )
            )
        if want.kind in ("element", "nat"):
            self.adopt(table)
        return (
            Requirement(**common, reason=reason, flag=flag, message=message),
            statements,
        )

    def reflections(self, dims: Sequence[str]) -> Reflections:
        """A table of reflected parameters that knows every name of the program."""
        table = Reflections()
        table.reserve(self.taken)
        table.reserve(dims)
        table.reserve(self.sizes)
        for name, expr in self.reflected:
            table.adopt(name, expr)
        return table

    def adopt(self, table: Reflections) -> None:
        """Make the parameters ``table`` allocated parameters of the term."""
        known = {name for name, _ in self.reflected}
        for name, expr in table.items():
            if name not in known:
                self.reflected.append((name, expr))
                self.taken.add(name)

    def register(self, domain: isl.Set) -> None:
        """Keep the names of a check statement's loops from later statements."""
        for name in domain.get_var_names(isl.dim_type.set):
            self.taken.add(name)

    @staticmethod
    def guarded(stmt: Stmt, flags: Sequence[str]) -> Stmt:
        """``stmt``, run only where no checked point before it has failed.

        The flags come first in the guard, which short-circuits in the
        interpreter and in the generated C, so that a condition of the
        statement's own, which may read the array that failed, is not
        evaluated either. A guard that reads an array is no constraint isl
        can state, so the domain is unchanged and the conjuncts are listed
        as ``unnarrowed``.
        """
        if not flags:
            return stmt
        oks = [Comparison(Subscript(Var(flag), 0), "==", 0) for flag in flags]
        parts = [*oks, *([stmt.guard] if stmt.guard is not None else [])]
        guard = parts[0] if len(parts) == 1 else LogicalAnd(tuple(parts))
        loops = stmt.loop_domain if stmt.loop_domain is not None else stmt.domain
        return dataclasses.replace(
            stmt,
            guard=guard,
            unnarrowed=(
                *((render(ok), _NOT_AFFINE) for ok in oks),
                *stmt.unnarrowed,
            ),
            # A flag is read at every point of the loops, before the
            # statement's own conditions are asked (Stmt.path).
            path=(
                ((*((ok, loops) for ok in oks), *stmt.path)) if stmt.path else ()
            ),
        )

    # }}}

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
        if typ.domain is not None:
            return self.domain_zeros(name, typ, made, block)
        inames = tuple(self.fresh("i") for _ in typ.axes)
        return Stmt(
            id=f"{name}.zeros",
            inames=inames,
            domain=domain_set(inames, typ.axes),
            assignee=Access(name, tuple(Var(iname) for iname in inames)),
            expr=0,
            kind="assign",
            guard=None,
            where=made.where,
            order=(block, *(0,) * len(inames)),
        )

    def domain_zeros(self, name: str, typ: ArrType, made: _Made, block: int) -> Stmt:
        """The statement that zeroes an array the program made over a domain.

        Over the points of the domain, which is the array's every cell. A
        union of pieces is refused: a temporary is stored boxed, as one array
        of the box's shape, which a union's pieces are not.
        """
        from loopty.domain import Union

        if isinstance(typ.domain, Union):
            raise TraceError(
                f"{self.program} makes {name} at {made.where} over {typ.domain}, "
                "a union of pieces, and an array a program makes over a domain "
                "is stored as one box; make it a parameter of the program, or "
                "make one array for each piece"
            )
        inames = tuple(self.fresh("i") for _ in range(typ.domain.ndim))
        domain = typ.domain.isl_set(inames)
        return Stmt(
            id=f"{name}.zeros",
            inames=inames,
            domain=domain,
            assignee=Access(name, tuple(Var(iname) for iname in inames)),
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
            path=tuple(
                (rename_expr(condition, {}, exprs), rename_set(points, {}, exprs))
                for condition, points in stmt.path
            ),
        )


# }}}


# {{{ a program stopped at a call


def stopped_at(term: Term, statement: str) -> tuple[Term, str]:
    """``term``, stopped where it reaches the call ``statement`` is in.

    A program's call whose loop arithmetic the native run refuses, after a
    checked point, is refused only once the run shows that the point passed
    (:func:`loopty.refusal.refuse_where_native_does`), and natively the
    program stops in that call, with what the calls before it wrote (#176).
    This is the program with one parameter more, a one-cell flag, which
    guards the call's statements and every one after it, as a checked
    point's flag guards those after the point (:meth:`_Composer.guarded`):
    run with the flag set, it runs the calls before and stops there, its
    checked points before the call included. Returns the term and the
    flag's name.
    """
    scope = next((s for s in term.scopes if statement in s.statements), None)
    ids = set(scope.statements) if scope is not None else {statement}
    start = min(stmt.order[0] for stmt in term.stmts if stmt.id in ids)
    taken = {
        *term.param_names,
        *term.sizes,
        *(name for name, _ in term.temporaries),
        *(name for name, _ in term.tables),
        *(name for name, _ in term.reflected),
        *(iname for stmt in term.stmts for iname in stmt.inames),
    }
    label = scope.call if scope is not None else term.name
    flag = re.sub(r"\W", "_", label) + "_refused"
    while flag in taken:
        flag = f"{flag}_"
    stmts = tuple(
        _Composer.guarded(stmt, [flag]) if stmt.order[0] >= start else stmt
        for stmt in term.stmts
    )
    typ = ArrType(axes=(1,), dtype=Nat, ragged=(False,))
    return (
        dataclasses.replace(term, params=(*term.params, (flag, typ)), stmts=stmts),
        flag,
    )


# }}}
