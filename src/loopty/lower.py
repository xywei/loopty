"""Lowering: from a checked term to a loopy kernel.

There is no translation of the expression language. loopty and lanky both use
pymbolic, and loopy's instructions are pymbolic too, so a term's index and
scalar expressions are already in the target language. The work is structural:
statement domains become the ``lp.make_kernel`` domains, each
:class:`~loopty.term.Stmt` becomes one instruction with its own iname set,
:class:`~loopty.term.Reduction` becomes ``lp.Reduction``, and argument shapes and
dtypes come from the :class:`~loopty.term.ArrType` of each parameter.

Two things are nevertheless rebuilt rather than passed through.

*lanky's nodes are rebuilt as plain pymbolic nodes.* lanky's expression classes
subclass pymbolic's, but they redefine ``==`` to build a proposition instead of
answering a bool, and loopy compares expressions for equality everywhere. So
:class:`ExpressionLowerer` walks the term and reconstructs it out of
``pymbolic.primitives`` nodes. The mapper dispatch is the ordinary one, which is
exactly why lanky's decision to subclass pays off here.

*A ragged axis becomes what it already is in memory.* For
``val: Arr[Fin[n], Fin[cnt], Real]`` the term indexes ``val[r, j]`` in the array's
own index-type axes; storage is a flat buffer plus an offsets array, so the
lowered access is ``val[off[r] + j]``. The loop bound ``cnt[r]`` is not
expressible in isl, and does not have to be: loopy takes a domain parameter whose
value is assigned to a scalar temporary inside the enclosing loop, so
``{ [j] : 0 <= j < cnt_r }`` with ``cnt_r = off[r+1] - off[r]`` generates exactly
the C loop a CSR product wants. Naming that parameter is the one piece of
vocabulary shared between the tracer, which builds the domains, and this module,
which reads them: :data:`COUNT_PARAM` is the direct spelling and
:data:`COUNT_PARAM_REFLECTED` the one the tracer produces when it reflects the
non-affine term ``cnt[r]`` into a fresh isl parameter. Both are recognized.
"""

from __future__ import annotations

import dataclasses
import re
from collections.abc import Iterator, Sequence
from dataclasses import dataclass
from typing import Any

import islpy as isl
import loopy as lp
import numpy as np
import pymbolic.primitives as prim
from loopy.symbolic import Reduction as LoopyReduction
from pymbolic.mapper import Mapper

from loopty.term import Access, ArrType, Reduction, Stmt, Term

__all__ = [
    "COUNT_PARAM",
    "COUNT_PARAM_REFLECTED",
    "RESERVED_WORDS",
    "ExpressionLowerer",
    "LoweringError",
    "Lowering",
    "count_param_name",
    "count_param_names",
    "lower",
    "lower_generic",
    "numpy_dtype",
    "target_for",
]

#: How a ragged loop bound appears as a parameter of a statement domain: the
#: counts name, an underscore, and the enclosing iname. For a loop over
#: ``val.dom[r]`` of ``val: Arr[Fin[n], Fin[cnt], Real]`` that is ``cnt_r``, and
#: lowering assigns it ``off[r+1] - off[r]`` (or ``cnt[r]`` when the counts array
#: itself is a parameter) in a scalar temporary inside the ``r`` loop.
COUNT_PARAM = "{counts}_{iname}"

#: The same bound as the tracer spells it when it reflects the non-affine term
#: ``cnt[r]`` into a fresh isl parameter (see ``loopty.idx``). Both spellings are
#: recognized, so that a hand-written term and a traced one lower the same way.
COUNT_PARAM_REFLECTED = "nl_{counts}_{iname}"

#: Candidate names for the offsets array of a ragged axis, most specific first.
#: The first one that is a parameter of the term wins; if none is, an argument
#: named ``off_<counts>`` is added to the lowered kernel.
OFFSETS_CANDIDATES = ("off_{counts}", "{counts}_off", "off")

_LANG_VERSION = (2018, 2)


class LoweringError(TypeError):
    """A term that cannot be handed to loopy as it stands.

    Raised rather than guessed at: a silently wrong lowering would be checked
    against nothing, since the isl obligations are stated about the term and not
    about the generated code.
    """


def count_param_name(counts: str, iname: str) -> str:
    """The domain parameter standing for a ragged bound; see :data:`COUNT_PARAM`."""
    return COUNT_PARAM.format(counts=counts, iname=iname)


def count_param_names(counts: str, iname: str) -> tuple[str, ...]:
    """Every spelling of one ragged bound parameter, most direct first."""
    return (
        COUNT_PARAM.format(counts=counts, iname=iname),
        COUNT_PARAM_REFLECTED.format(counts=counts, iname=iname),
    )


# {{{ dtypes and targets


def numpy_dtype(sort: Any) -> np.dtype:
    """The numpy dtype a lanky sort or an index type is stored in.

    ``Real`` is double precision, ``Nat`` and ``Int`` are 32-bit (which is what
    an index into an array is on every target loopy generates for), ``Bool`` is a
    byte, and an index type such as ``Fin[m]``, which is the element type of a
    column-index array, is stored as an integer like any other index.
    """
    if isinstance(sort, np.dtype):
        return sort
    if isinstance(sort, type) and issubclass(sort, np.generic):
        return np.dtype(sort)
    if sort is float:
        return np.dtype(np.float64)
    if sort is int:
        return np.dtype(np.int32)
    if sort is bool:
        return np.dtype(np.int8)
    base = getattr(sort, "base", None)  # a lanky refinement T & prop
    if base is not None and base is not sort:
        return numpy_dtype(base)
    name = getattr(sort, "name", None)
    if name in ("Real",):
        return np.dtype(np.float64)
    if name in ("Nat", "Int"):
        return np.dtype(np.int32)
    if name in ("Bool",):
        return np.dtype(np.int8)
    if hasattr(sort, "bound") or hasattr(sort, "size"):  # an index type Fin[m]
        return np.dtype(np.int32)
    raise LoweringError(f"no numpy dtype for {sort!r}")


def target_for(target: str = "c") -> Any:
    """The loopy target named by ``target``.

    ``"c"`` is ``lp.ExecutableCTarget``, which compiles with the system toolchain
    and runs in process; it is the only target a laptop or CI ever uses.
    ``"opencl"`` is ``lp.PyOpenCLTarget``, and pyopencl is imported here and
    nowhere else, inside the branch, so that importing loopty on a machine
    without a device costs nothing and can never fail.
    """
    if target in ("c", None):
        return lp.ExecutableCTarget()
    if target == "c-source":
        return lp.CTarget()
    if target == "opencl":
        from loopy.target.pyopencl import PyOpenCLTarget

        return PyOpenCLTarget()
    raise LoweringError(f"unknown target {target!r}; expected 'c' or 'opencl'")


# }}}


# {{{ walking terms


def _children(expr: Any) -> tuple[Any, ...]:
    """The subexpressions of a node, for the generic walks below."""
    if isinstance(expr, Access):
        return tuple(expr.indices)
    if isinstance(expr, Reduction):
        return (expr.body,)
    if isinstance(expr, prim.Subscript):
        index = expr.index
        return (expr.aggregate, *(index if isinstance(index, tuple) else (index,)))
    if isinstance(expr, prim.ExpressionNode):
        if dataclasses.is_dataclass(expr):
            return tuple(getattr(expr, f.name) for f in dataclasses.fields(expr))
        return tuple(expr.__getinitargs__())
    if isinstance(expr, tuple | list):
        return tuple(expr)
    return ()


def walk(expr: Any) -> Iterator[Any]:
    """Every node of an expression tree, parents before children."""
    yield expr
    for child in _children(expr):
        yield from walk(child)


def arrays_of(expr: Any) -> tuple[str, ...]:
    """Names of the arrays referenced anywhere in ``expr``, in first-seen order."""
    names: list[str] = []
    for node in walk(expr):
        name = None
        if isinstance(node, Access):
            name = node.array
        elif isinstance(node, prim.Subscript) and isinstance(
            node.aggregate, prim.Variable
        ):
            name = node.aggregate.name
        if name is not None and name not in names:
            names.append(name)
    return tuple(names)


def reductions_of(expr: Any) -> tuple[Reduction, ...]:
    """Every :class:`~loopty.term.Reduction` in ``expr``, outermost first."""
    return tuple(node for node in walk(expr) if isinstance(node, Reduction))


# }}}


# {{{ expressions


class ExpressionLowerer(Mapper):
    """Rebuild a term's expression out of plain pymbolic nodes.

    Three jobs in one walk. lanky's subclasses are replaced by pymbolic's, so
    that loopy's structural comparisons work (lanky's ``==`` builds a
    proposition). :class:`~loopty.term.Access` and bare subscripts are turned
    into flat storage accesses, which is where a ragged layout's ``off[r] + j``
    enters. :class:`~loopty.term.Reduction` becomes ``lp.Reduction`` over its
    inames, and the reduction's domain is collected on the side so the caller can
    add it to the kernel.
    """

    def __init__(self, lowering: _Builder) -> None:
        super().__init__()
        self.lowering = lowering

    # The dispatcher: two of our node types are not pymbolic nodes at all, so
    # they are recognized before the mapper method lookup.
    def rec(self, expr: Any, *args: Any, **kwargs: Any) -> Any:
        if isinstance(expr, Access):
            return self.map_access(expr)
        if isinstance(expr, Reduction):
            return self.map_term_reduction(expr)
        return super().rec(expr, *args, **kwargs)

    __call__ = rec

    def map_access(self, expr: Access) -> prim.Expression:
        """An array reference in index-type axes becomes one in flat storage."""
        return self.lowering.access(
            expr.array, tuple(self.rec(i) for i in expr.indices)
        )

    def map_term_reduction(self, expr: Reduction) -> prim.Expression:
        """``Reduction`` becomes ``lp.Reduction``; its domain is collected."""
        self.lowering.add_reduction_domain(expr)
        return LoopyReduction(expr.op, tuple(expr.inames), self.rec(expr.body))

    def map_lanky_sum(self, expr: Any) -> prim.Expression:
        """A lanky ``Sum`` the tracer left in place becomes a loopy reduction.

        The tracer normally replaces it with :class:`~loopty.term.Reduction`,
        which carries an isl domain. When it does not, the binders' index types
        are read for their sizes, which covers ``lanky.sum(... for j in Fin[n])``.
        """
        inames = tuple(binder.name for binder, _domain in expr.binders)
        self.lowering.add_binder_domains(expr.binders)
        return LoopyReduction("sum", inames, self.rec(expr.body))

    def map_constant(self, expr: Any) -> Any:
        return expr

    def map_variable(self, expr: Any) -> prim.Variable:
        return prim.Variable(expr.name)

    def map_subscript(self, expr: Any) -> prim.Expression:
        if not isinstance(expr.aggregate, prim.Variable):
            raise LoweringError(f"cannot lower a subscript of {expr.aggregate!r}")
        index = expr.index if isinstance(expr.index, tuple) else (expr.index,)
        return self.lowering.access(
            expr.aggregate.name, tuple(self.rec(i) for i in index)
        )

    def map_sum(self, expr: Any) -> prim.Expression:
        return prim.Sum(tuple(self.rec(child) for child in expr.children))

    def map_product(self, expr: Any) -> prim.Expression:
        return prim.Product(tuple(self.rec(child) for child in expr.children))

    def map_quotient(self, expr: Any) -> prim.Expression:
        return prim.Quotient(self.rec(expr.numerator), self.rec(expr.denominator))

    def map_floor_div(self, expr: Any) -> prim.Expression:
        return prim.FloorDiv(self.rec(expr.numerator), self.rec(expr.denominator))

    def map_remainder(self, expr: Any) -> prim.Expression:
        return prim.Remainder(self.rec(expr.numerator), self.rec(expr.denominator))

    def map_power(self, expr: Any) -> prim.Expression:
        return prim.Power(self.rec(expr.base), self.rec(expr.exponent))

    def map_call(self, expr: Any) -> prim.Expression:
        return prim.Call(
            self.rec(expr.function), tuple(self.rec(p) for p in expr.parameters)
        )

    def map_comparison(self, expr: Any) -> prim.Expression:
        return prim.Comparison(self.rec(expr.left), expr.operator, self.rec(expr.right))

    def map_logical_and(self, expr: Any) -> prim.Expression:
        return prim.LogicalAnd(tuple(self.rec(c) for c in expr.children))

    def map_logical_or(self, expr: Any) -> prim.Expression:
        return prim.LogicalOr(tuple(self.rec(c) for c in expr.children))

    def map_logical_not(self, expr: Any) -> prim.Expression:
        return prim.LogicalNot(self.rec(expr.child))

    def map_lanky_abs(self, expr: Any) -> prim.Expression:
        """lanky's ``Abs`` is the C library's ``abs``, which loopy knows."""
        return prim.Call(prim.Variable("abs"), (self.rec(expr.operand),))

    # lanky's Abs may dispatch under either name depending on its mapper method.
    map_abs = map_lanky_abs

    def map_reduction(self, expr: Any) -> prim.Expression:
        """A loopy reduction that is already lowered passes through."""
        return LoopyReduction(expr.operation, tuple(expr.inames), self.rec(expr.expr))

    def handle_unsupported_expression(
        self, expr: Any, *args: Any, **kwargs: Any
    ) -> Any:
        raise LoweringError(
            f"cannot lower {type(expr).__name__} into a loopy expression: {expr!r}"
        )


# }}}


@dataclass(frozen=True)
class Lowering:
    """A lowered term: the loopy kernel, and how to find one's way back.

    ``insn_ids`` maps a :class:`~loopty.term.Stmt` id to the id of the loopy
    instruction it became, which is what lets a rejected schedule name a term
    statement rather than a generated instruction. ``value_args`` and
    ``array_args`` name the arguments in call order, ``outputs`` those the kernel
    writes, and ``ragged`` records, per array, the offsets argument its flat
    storage is indexed through.
    """

    term: Term
    kernel: Any
    insn_ids: dict[str, str]
    array_args: tuple[str, ...]
    value_args: tuple[str, ...]
    outputs: tuple[str, ...]
    ragged: dict[str, str]
    target: str = "c"

    @property
    def name(self) -> str:
        """The kernel's name."""
        return self.term.name


def _reads_assignee(stmt: Stmt) -> bool:
    """Does the statement's expression read the very cell the statement writes?

    This is the invariant :class:`~loopty.term.Stmt` states for
    ``kind == "accumulate"``, not a guess at what the expression means: an
    accumulating statement records the *complete* right-hand side, so the cell
    it adds into occurs in ``expr``. Lowering asks the question in order to
    refuse a term that breaks the invariant, because the two readings
    (``expr`` as the increment, ``expr`` as the whole right-hand side) differ by
    one addition and nothing downstream could tell them apart.
    """
    target = _render(stmt.assignee)
    for node in walk(stmt.expr):
        if isinstance(node, Access | prim.Subscript) and _render(node) == target:
            return True
    return False


def _render(reference: Any) -> str:
    """A plain-pymbolic rendering of an array reference, for comparing two."""
    if isinstance(reference, Access):
        indices = tuple(_plain(index) for index in reference.indices)
        return f"{reference.array}{indices!r}"
    if isinstance(reference, prim.Subscript) and isinstance(
        reference.aggregate, prim.Variable
    ):
        index = reference.index
        indices = tuple(
            _plain(i) for i in (index if isinstance(index, tuple) else (index,))
        )
        return f"{reference.aggregate.name}{indices!r}"
    return repr(reference)


#: Words the generated code cannot use as an identifier. C's keywords (C23),
#: plus the type names OpenCL C adds, because the same term is lowered for both
#: targets and a name that compiles on one has to compile on the other.
RESERVED_WORDS = frozenset(
    """
    alignas alignof auto bool break case char const constexpr continue default
    do double else enum extern false float for goto if inline int long nullptr
    register restrict return short signed sizeof static static_assert struct
    switch thread_local true typedef typeof union unsigned void volatile while
    complex imaginary generic noreturn
    half quad uchar ushort uint ulong char2 char3 char4 char8 char16 uchar2
    uchar3 uchar4 uchar8 uchar16 short2 short3 short4 short8 short16 ushort2
    ushort3 ushort4 ushort8 ushort16 int2 int3 int4 int8 int16 uint2 uint3
    uint4 uint8 uint16 long2 long3 long4 long8 long16 ulong2 ulong3 ulong4
    ulong8 ulong16 float2 float3 float4 float8 float16 double2 double3 double4
    double8 double16 kernel global local constant private read_only write_only
    read_write
    """.split()
)


def _sanitize(name: str) -> str:
    """A loopy-safe identifier: every non-word character becomes an underscore."""
    return re.sub(r"\W", "_", name)


def _kernel_name(name: str, taken: Sequence[str]) -> str:
    """A C-safe function name for a lowered term, renamed where it has to be.

    loopy passes the kernel's name straight through to the generated source, so
    a kernel written ``def double(...)`` produces ``void double(...)``, which no
    C compiler accepts, and a kernel whose name is also an argument's name
    produces a function whose own name is shadowed by a parameter. Neither is
    diagnosed anywhere downstream: the first is a compiler error about generated
    code the user never wrote, and the second is undefined behaviour. The name
    is therefore renamed here, deterministically, with an ``_knl`` suffix.
    Renaming rather than refusing keeps a legal Python name legal: nothing
    outside the generated source refers to the kernel by this name, because
    callers hold the :class:`Lowering` and address arguments by name.
    """
    base = _sanitize(name)
    if not base or base[0].isdigit():
        base = f"k_{base}"
    reserved = set(taken) | RESERVED_WORDS
    if base not in reserved:
        return base
    candidate = f"{base}_knl"
    while candidate in reserved:
        candidate = f"{candidate}_"
    return candidate


def _written_arrays(term: Term) -> tuple[str, ...]:
    """Arrays the term assigns to, in first-seen order."""
    names: list[str] = []
    for stmt in term.stmts:
        if stmt.assignee.array not in names:
            names.append(stmt.assignee.array)
    return tuple(names)


class _Builder:
    """Mutable scratch space for one lowering; see :func:`lower_generic`."""

    def __init__(self, term: Term, target: str) -> None:
        self.term = term
        self.target = target
        self.arr_types: dict[str, ArrType] = {
            name: typ for name, typ in term.params if isinstance(typ, ArrType)
        }
        self.scalar_types: dict[str, Any] = {
            name: typ for name, typ in term.params if not isinstance(typ, ArrType)
        }
        self.written = _written_arrays(term)
        self.ragged: dict[str, str] = {}
        self.extra_domains: list[isl.Set] = []
        self.extra_args: list[Any] = []
        self.value_args: list[str] = []
        self.expr = ExpressionLowerer(self)

    # {{{ ragged storage

    def ragged_axis(self, name: str) -> int | None:
        """Index of the ragged axis of array ``name``, or ``None`` if dense."""
        typ = self.arr_types.get(name)
        if typ is None:
            return None
        for k, flag in enumerate(typ.ragged):
            if flag:
                return k
        return None

    @property
    def ragged_params(self) -> tuple[str, ...]:
        """Every name a ragged bound could go by; see :data:`COUNT_PARAM`."""
        names: list[str] = []
        for name in self.arr_types:
            if self.ragged_axis(name) is None:
                continue
            counts = self.counts_name(name)
            for stmt in self.term.stmts:
                for iname in stmt.inames:
                    for param in count_param_names(counts, iname):
                        if param not in names:
                            names.append(param)
        return tuple(names)

    def counts_name(self, name: str) -> str:
        """The name of the counts family of a ragged array's ragged axis."""
        typ = self.arr_types[name]
        axis = self.ragged_axis(name)
        assert axis is not None
        size = typ.axes[axis]
        if isinstance(size, prim.Variable):
            return size.name
        raise LoweringError(
            f"the ragged axis of {name} is bounded by {size!r}; a ragged bound "
            "must name the counts array so that its offsets can be found"
        )

    def offsets_for(self, name: str) -> str:
        """The offsets argument that flattens array ``name``.

        The first of :data:`OFFSETS_CANDIDATES` that is a parameter of the term
        wins, which makes ``spmv(off, col, val, x, y)`` work with no
        configuration; when none is, an ``int32`` argument is added.
        """
        if name in self.ragged:
            return self.ragged[name]
        counts = self.counts_name(name)
        params = dict(self.term.params)
        for pattern in OFFSETS_CANDIDATES:
            candidate = pattern.format(counts=counts)
            if candidate in params:
                self.ragged[name] = candidate
                return candidate
        candidate = f"off_{counts}"
        typ = self.arr_types[name]
        outer = typ.axes[0]
        self.extra_args.append(
            lp.GlobalArg(
                candidate,
                np.dtype(np.int32),
                shape=(_plus_one(outer),),
                is_input=True,
                is_output=False,
            )
        )
        self.ragged[name] = candidate
        return candidate

    def access(self, name: str, indices: tuple[Any, ...]) -> prim.Expression:
        """The flat-storage reference for an index tuple in index-type axes."""
        variable = prim.Variable(name)
        axis = self.ragged_axis(name)
        if axis is None:
            if not indices:
                return variable
            return prim.Subscript(variable, indices)
        if axis != 1 or len(indices) != 2:
            raise LoweringError(
                f"{name} is ragged in axis {axis} with {len(indices)} indices; "
                "only a two-axis ragged array (row, fiber) is lowered today"
            )
        offsets = self.offsets_for(name)
        flat = prim.Subscript(prim.Variable(offsets), (indices[0],)) + indices[1]
        return prim.Subscript(variable, (flat,))

    # }}}

    def add_reduction_domain(self, reduction: Reduction) -> None:
        """Record a reduction's iteration domain as a domain of the kernel."""
        self.extra_domains.append(_domain_over(reduction.domain, reduction.inames))

    def add_binder_domains(self, binders: Sequence[Any]) -> None:
        """Record domains for a lanky ``Sum`` the tracer did not convert."""
        for binder, domain in binders:
            size = getattr(domain, "bound", getattr(domain, "size", None))
            if size is None:
                raise LoweringError(
                    f"the binder {binder} of a lanky sum has no size to build a "
                    "domain from; trace the kernel so the reduction carries one"
                )
            from loopty.idx import to_set

            self.extra_domains.append(to_set((size,), names=(binder.name,)))


def _plus_one(size: Any) -> Any:
    """``size + 1``, folded when the size is an integer literal."""
    if isinstance(size, int):
        return size + 1
    return prim.Sum((_plain(size), 1))


def _plain(expr: Any) -> Any:
    """A size expression rebuilt out of plain pymbolic nodes."""
    if isinstance(expr, int):
        return expr
    return ExpressionLowerer(_NullBuilder())(expr)


class _NullBuilder:
    """A builder for expressions that cannot contain array references."""

    def access(self, name: str, indices: tuple[Any, ...]) -> prim.Expression:
        variable = prim.Variable(name)
        return prim.Subscript(variable, indices) if indices else variable

    def add_reduction_domain(self, reduction: Reduction) -> None:
        raise LoweringError("a size expression may not contain a reduction")

    def add_binder_domains(self, binders: Sequence[Any]) -> None:
        raise LoweringError("a size expression may not contain a reduction")


def _domain_over(domain: isl.Set, inames: Sequence[str]) -> isl.Set:
    """``domain`` with its set dimensions named exactly ``inames``.

    A statement domain may arrive with more dimensions than the inames it is
    being used for: a reduction's domain carries the enclosing inames so that a
    ragged bound can depend on the row. The extra leading dimensions become
    parameters, which is how they reach loopy (as the scalar temporaries that
    hold ``off[r+1] - off[r]``).
    """
    n_dim = domain.dim(isl.dim_type.set)
    n_inames = len(inames)
    if n_dim < n_inames:
        raise LoweringError(
            f"domain {domain} has {n_dim} dimensions but {n_inames} inames"
        )
    if n_dim > n_inames:
        extra = n_dim - n_inames
        domain = domain.move_dims(isl.dim_type.param, 0, isl.dim_type.set, 0, extra)
    for k, iname in enumerate(inames):
        domain = domain.set_dim_name(isl.dim_type.set, k, iname)
    return domain


def _domain_params(domain: isl.Set) -> tuple[str, ...]:
    """Parameter names of an isl set."""
    return tuple(domain.get_var_names(isl.dim_type.param))


def _forget_params(domain: isl.Set, params: Sequence[str]) -> isl.Set:
    """Drop ``params`` from a set, existentially, then remove the dimensions.

    Eliminating first and projecting second is the point: projecting a parameter
    out of ``0 <= j < cnt_r`` alone would leave ``cnt_r > 0`` behind as a
    constraint on the remaining dimensions, and a row loop that skipped empty
    rows is not the loop the term describes.
    """
    for param in params:
        position = domain.find_dim_by_name(isl.dim_type.param, param)
        if position < 0:
            continue
        domain = domain.eliminate(isl.dim_type.param, position, 1)
        domain = domain.project_out(isl.dim_type.param, position, 1)
    return domain


def _statement_domains(stmt: Stmt, ragged_params: Sequence[str]) -> list[isl.Set]:
    """The domains a statement contributes, split at a ragged bound.

    loopy refuses a domain whose parameter is written inside a loop the same
    domain provides ("domain parameter may not be written inside a domain
    dependent on it"), and a ragged bound is exactly that: ``cnt_r`` is assigned
    inside the ``r`` loop, and it bounds ``j``. The cure is the shape loopy wants
    anyway, a nest of domains: ``{ [r] : 0 <= r < n }`` outside, and
    ``[r, cnt_r] -> { [j] : 0 <= j < cnt_r }`` inside it. A dense statement has
    no such parameter and keeps its single domain.
    """
    present = [p for p in ragged_params if p in _domain_params(stmt.domain)]
    if not present:
        return [_domain_over(stmt.domain, stmt.inames)]

    cut = -1
    for position, iname in enumerate(stmt.inames):
        if any(param.endswith(f"_{iname}") for param in present):
            cut = max(cut, position)
    if cut < 0 or cut + 1 >= len(stmt.inames):
        return [_domain_over(stmt.domain, stmt.inames)]

    outer = stmt.domain.project_out(
        isl.dim_type.set, cut + 1, stmt.domain.dim(isl.dim_type.set) - (cut + 1)
    )
    outer = _forget_params(outer, present)
    return [
        _domain_over(outer, stmt.inames[: cut + 1]),
        _domain_over(stmt.domain, stmt.inames[cut + 1 :]),
    ]


def _count_inits(
    term: Term, builder: _Builder, domains: Sequence[isl.Set]
) -> tuple[list[Any], dict[str, str]]:
    """Instructions assigning the ragged bound parameters, and their ids.

    A domain parameter named ``cnt_r`` (see :data:`COUNT_PARAM`) is a ragged
    bound: the length of row ``r`` of whichever array has ``cnt`` as its counts
    family. It is emitted as a scalar temporary inside the ``r`` loop, computed
    from the offsets when the offsets are a parameter and from the counts array
    when it is one. loopy then generates ``for (j = 0; j < cnt_r; ++j)``.
    """
    wanted: dict[str, str] = {}
    for domain in domains:
        for param in _domain_params(domain):
            wanted.setdefault(param, param)

    params = dict(term.params)
    insns: list[Any] = []
    ids: dict[str, str] = {}
    for name in builder.arr_types:
        axis = builder.ragged_axis(name)
        if axis is None:
            continue
        counts = builder.counts_name(name)
        for stmt in term.stmts:
            for iname in stmt.inames:
                candidates = [
                    name for name in count_param_names(counts, iname) if name in wanted
                ]
                if not candidates or candidates[0] in ids:
                    continue
                param = candidates[0]
                row = prim.Variable(iname)
                if counts in params:
                    value: Any = prim.Subscript(prim.Variable(counts), (row,))
                else:
                    offsets = builder.offsets_for(name)
                    value = prim.Subscript(
                        prim.Variable(offsets), (row + 1,)
                    ) - prim.Subscript(prim.Variable(offsets), (row,))
                insn_id = f"{param}_init"
                enclosing = stmt.inames[: stmt.inames.index(iname) + 1]
                insns.append(
                    lp.Assignment(
                        assignee=prim.Variable(param),
                        expression=value,
                        id=insn_id,
                        within_inames=frozenset(enclosing),
                        temp_var_type=lp.Optional(np.dtype(np.int32)),
                    )
                )
                ids[param] = insn_id
    return insns, ids


def lower_generic(term: Term, target: str = "c") -> Lowering:
    """Lower ``term``, keeping the map from term statements to instructions.

    This is :func:`lower` plus bookkeeping. A schedule needs to say *which term
    statement* it would reorder, and an executor needs to know which arguments
    the kernel writes; both are recorded here rather than recovered by matching
    names against generated code.
    """
    reserved = sorted(
        name for name, _ in term.params if _sanitize(name) in RESERVED_WORDS
    )
    if reserved:
        # An argument cannot be renamed the way the kernel can: the caller
        # passes it by name, so a rename here would silently break every call.
        raise LoweringError(
            f"{term.name} has parameters the generated code cannot name: "
            f"{', '.join(reserved)}. These are reserved words in C or OpenCL C; "
            "rename them in the kernel's signature."
        )
    builder = _Builder(term, target)
    expr = builder.expr
    ragged_params = builder.ragged_params

    domains: list[isl.Set] = []
    insns: list[Any] = []
    insn_ids: dict[str, str] = {}
    writes_before: dict[str, list[str]] = {}
    reads_before: dict[str, list[str]] = {}

    for stmt in term.stmts:
        if not isinstance(stmt, Stmt):  # pragma: no cover - defensive
            raise LoweringError(f"not a statement: {stmt!r}")
        domains.extend(_statement_domains(stmt, ragged_params))

        if stmt.kind not in ("assign", "accumulate"):
            raise LoweringError(
                f"statement {stmt.id} has kind {stmt.kind!r}; "
                "expected 'assign' or 'accumulate'"
            )
        if stmt.kind == "accumulate" and not _reads_assignee(stmt):
            raise LoweringError(
                f"statement {stmt.id} has kind 'accumulate' but its expression "
                f"does not read {_render(stmt.assignee)}. An accumulating "
                "statement records the complete right-hand side, so "
                "'y[r] += t' is Stmt(assignee=y[r], expr=y[r] + t); see "
                "loopty.term.Stmt. Write the whole right-hand side, or use "
                "kind='assign'."
            )
        assignee = expr(stmt.assignee)
        body = expr(stmt.expr)

        read_arrays = set(arrays_of(stmt.expr)) | set(
            arrays_of(tuple(stmt.assignee.indices))
        )
        if stmt.kind == "accumulate":
            read_arrays.add(stmt.assignee.array)
        written = stmt.assignee.array

        # Order the statements by their data: a statement runs after every
        # earlier one it could read from, write over, or overwrite the input of.
        depends: set[str] = set()
        for array in read_arrays:
            depends.update(writes_before.get(array, ()))
        depends.update(writes_before.get(written, ()))
        depends.update(reads_before.get(written, ()))

        insn_id = _sanitize(stmt.id)
        insn_ids[stmt.id] = insn_id
        predicates = frozenset()
        if stmt.guard is not None:
            predicates = frozenset([expr(stmt.guard)])
        insns.append(
            lp.Assignment(
                assignee=assignee,
                expression=body,
                id=insn_id,
                within_inames=frozenset(stmt.inames),
                depends_on=frozenset(depends),
                predicates=predicates,
            )
        )
        for array in read_arrays:
            reads_before.setdefault(array, []).append(insn_id)
        writes_before.setdefault(written, []).append(insn_id)

    domains.extend(builder.extra_domains)

    count_insns, count_ids = _count_inits(term, builder, domains)
    if count_insns:
        by_id = {insn.id: insn for insn in insns}
        for stmt in term.stmts:
            insn = by_id[insn_ids[stmt.id]]
            needed = {
                count_ids[param]
                for param in _domain_params(_domain_over(stmt.domain, stmt.inames))
                if param in count_ids
            }
            for reduction in reductions_of(stmt.expr):
                needed |= {
                    count_ids[param]
                    for param in _domain_params(reduction.domain)
                    if param in count_ids
                }
            if needed:
                by_id[insn.id] = insn.copy(depends_on=insn.depends_on | needed)
        insns = [by_id[insn.id] for insn in insns]
        insns = count_insns + insns

    args, array_args, value_args, outputs = _arguments(
        term, builder, domains, count_ids, insns
    )
    domains = _merge_domains(domains)

    kernel = lp.make_kernel(
        domains,
        insns,
        args,
        target=target_for(target),
        lang_version=_LANG_VERSION,
        name=_kernel_name(term.name, [arg.name for arg in args]),
    )
    # Pin the loop nest to the order the body was written in. loopy is free to
    # choose an order otherwise, and its choice is not checked against the term:
    # for the Jacobi stencil it puts the space loop outside the time loop, which
    # reverses a dependence. The traced order is the one the term means, and any
    # departure from it is a schedule, hence a cast, hence checked.
    for stmt in term.stmts:
        if len(stmt.inames) > 1:
            kernel = lp.prioritize_loops(kernel, ",".join(stmt.inames))
    return Lowering(
        term=term,
        kernel=kernel,
        insn_ids=insn_ids,
        array_args=array_args,
        value_args=value_args,
        outputs=outputs,
        ragged=dict(builder.ragged),
        target=target,
    )


def _merge_domains(domains: Sequence[isl.Set]) -> list[isl.Set]:
    """One domain per tuple of inames, because loopy defines each iname once.

    Every statement contributes its own domain, so two statements in the same
    loop contribute the same domain twice and loopy refuses the second one:
    "redefines iname 't' that is part of a previous domain". A reduction over a
    fiber contributes the same inner domain as the statements inside that fiber,
    for the same reason. Identical domains are therefore dropped.

    Two domains over the same inames that are genuinely different sets are
    merged by union, which is what a ``when`` guard produces: the guarded
    statement's domain is narrower, it keeps its predicate, and the loop has to
    run over the wider of the two. A union that is not convex is a nest loopy
    cannot express with one iname, and saying so here names the domains rather
    than letting loopy fail later about a generated instruction.
    """
    merged: list[isl.Set] = []
    for domain in domains:
        names = tuple(domain.get_var_names(isl.dim_type.set))
        for position, seen in enumerate(merged):
            if tuple(seen.get_var_names(isl.dim_type.set)) != names:
                continue
            try:
                left = seen.align_params(domain.get_space())
                right = domain.align_params(seen.get_space())
            except Exception:  # pragma: no cover - isl declines to align
                merged.append(domain)
                break
            if left.is_equal(right):
                break
            union = left.union(right).coalesce()
            if union.n_basic_set() != 1:
                raise LoweringError(
                    f"two domains over {names} that do not merge into one: "
                    f"{seen} and {domain}. loopy gives an iname a single "
                    "domain, so the two loops need different inames."
                )
            merged[position] = union
            break
        else:
            merged.append(domain)
    return merged


def _used_names(domains: Sequence[isl.Set], insns: Sequence[Any]) -> set[str]:
    """Names the generated code itself mentions: domain parameters and variables.

    This is not a nicety. loopy builds the device function's signature from the
    names the kernel body needs, and the host wrapper's call from the names the
    kernel *has*; a value argument that occurs only in another argument's shape
    is in the second list and not in the first, and the two lists then disagree
    about what is being passed. Declaring only the names the code uses keeps
    them the same list.
    """
    names: set[str] = set()
    # The callee of a call is the name of a function the target provides, such
    # as ``sqrt``, and loopy resolves it through its callable registry. It is
    # not a value the kernel is passed, so it must not become a value argument;
    # it is collected separately and removed at the end.
    functions: set[str] = set()
    for domain in domains:
        names.update(_domain_params(domain))
    for insn in insns:
        for expr in (
            insn.assignee,
            insn.expression,
            *getattr(insn, "predicates", ()),
        ):
            for node in walk(expr):
                if isinstance(node, prim.Call) and isinstance(
                    node.function, prim.Variable
                ):
                    functions.add(node.function.name)
                elif isinstance(node, prim.Variable):
                    names.add(node.name)
    return names - functions


def _arguments(
    term: Term,
    builder: _Builder,
    domains: Sequence[isl.Set],
    count_ids: dict[str, str],
    insns: Sequence[Any],
) -> tuple[list[Any], tuple[str, ...], tuple[str, ...], tuple[str, ...]]:
    """The loopy arguments of a term, in signature order.

    Sizes the code uses become value arguments, and an array keeps its symbolic
    shape when every name in it is one of those, which is how loopy deduces ``n``
    from the array that was passed for ``y``. A size that appears in no loop
    bound and no expression is deliberately not declared, and the arrays whose
    shape mentions it are declared without one: see :func:`_used_names`. A ragged
    array is a flat buffer of a length no loop bound knows, so it is always in
    that case, and its offsets argument is what gives its rows back.
    """
    used = _used_names(domains, insns)
    provided = {arg.name for arg in builder.extra_args} | set(builder.ragged.values())
    known_inames = {iname for stmt in term.stmts for iname in stmt.inames}
    for stmt in term.stmts:
        for reduction in reductions_of(stmt.expr):
            known_inames.update(reduction.inames)
    sizes = [
        name
        for name in used
        if name not in count_ids
        and name not in known_inames
        and name not in provided
        and name not in dict(term.params)
    ]
    scalars = {name for name, typ in term.params if not isinstance(typ, ArrType)}

    args: list[Any] = []
    array_args: list[str] = []
    value_args: list[str] = []
    outputs: list[str] = []
    declared: set[str] = set()

    def declare_value(name: str, dtype: np.dtype) -> None:
        if name in declared:
            return
        args.append(lp.ValueArg(name, dtype))
        value_args.append(name)
        declared.add(name)

    # Sizes first: an array's shape may only mention names that are arguments.
    for name in sorted(sizes):
        declare_value(name, np.dtype(np.int32))

    def shape_of(typ: ArrType, ragged: bool) -> tuple[Any, ...] | None:
        if ragged:
            return None
        shape = tuple(_plain(size) for size in typ.axes)
        free: set[str] = set()
        for size in shape:
            free |= {name for name in _names_in(size)}
        return shape if free <= declared | scalars else None

    for name, typ in term.params:
        if not isinstance(typ, ArrType):
            declare_value(name, numpy_dtype(typ))
            continue
        is_output = name in builder.written
        ragged = builder.ragged_axis(name) is not None
        if ragged:
            builder.offsets_for(name)  # make sure the offsets argument exists
        args.append(
            lp.GlobalArg(
                name,
                numpy_dtype(typ.dtype),
                shape=shape_of(typ, ragged),
                is_input=True,
                is_output=is_output,
            )
        )
        array_args.append(name)
        declared.add(name)
        if is_output:
            outputs.append(name)

    for extra in builder.extra_args:
        if extra.name in declared:
            continue
        args.append(extra)
        array_args.append(extra.name)
        declared.add(extra.name)

    return args, tuple(array_args), tuple(value_args), tuple(outputs)


def _names_in(expr: Any) -> set[str]:
    """Free variable names of a size expression."""
    return {node.name for node in walk(expr) if isinstance(node, prim.Variable)}


def lower(term: Term, target: str = "c") -> Any:
    """Build the loopy kernel for ``term`` on ``target``."""
    return lower_generic(term, target).kernel


def _register_term_lowering() -> None:
    """Tell lanky how loopty turns its ``Sum`` term into a loopy reduction.

    lanky owns the reduction term and knows nothing about loopy; the lowering is
    a plugin hook, and this is loopty filling it in. Registered at import so that
    anything holding only lanky's registry can lower a sum.
    """
    try:
        from lanky.plugins import registry
        from lanky.terms import Sum
    except ImportError:  # pragma: no cover - lanky is a hard dependency
        return

    def lower_sum(expr: Any, rec: Any = None) -> Any:
        inames = tuple(binder.name for binder, _domain in expr.binders)
        body = expr.body if rec is None else rec(expr.body)
        return LoopyReduction("sum", inames, body)

    registry.term_lowerings.setdefault(Sum, lower_sum)


_register_term_lowering()
