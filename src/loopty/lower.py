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
non-affine term ``cnt[r]`` into a fresh isl parameter. Both are recognized, by
:func:`loopty.flow.ragged_bound_params`, which is also how the access collector
lists the read the assignment makes; the spellings live in :mod:`loopty.term`.

*An array over a polyhedral domain is stored in the layout the lowering is
given* (:mod:`loopty.domain`). The domain itself needs nothing: a statement's
domain already carries the constraints of the fibers it runs over, so the
triangle is a triangular loop nest. Only the address is the layout's. Boxed, a
single domain is an array of the box's shape and ``L[i, j]`` is left to loopy;
a union's pieces follow one another in a flat buffer, each boxed, at a base that
is a sum of the boxes before it. Packed, ``L[i, j]`` is ``L[off_L[i] + j]``
through a table of row starts that the executor computes from the domain and
passes in, as it passes a ragged array's offsets.
"""

from __future__ import annotations

import dataclasses
import re
from collections.abc import Iterator, Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any

import islpy as isl
import loopy as lp
import numpy as np
import pymbolic.primitives as prim
from loopy.symbolic import Reduction as LoopyReduction
from loopy.symbolic import set_to_cond_expr
from pymbolic.mapper import Mapper

from loopty.domain import STORAGES, Union
from loopty.flow import (
    access_relation,
    bounds_dimension,
    counts_families,
    ragged_bound_params,
    statement_accesses,
)
from loopty.idx import linearize
from loopty.term import (
    COUNT_PARAM,
    COUNT_PARAM_REFLECTED,
    Access,
    ArrType,
    Reduction,
    Stmt,
    Term,
    count_param_names,
    declared_offsets,
    free_name_sorts,
    free_name_sorts_message,
)

__all__ = [
    "COUNT_PARAM",
    "COUNT_PARAM_REFLECTED",
    "GCC_NO_CONTRACTION_PRAGMA",
    "NO_CONTRACTION_FLAG",
    "NO_CONTRACTION_PRAGMAS",
    "RESERVED_PREFIX",
    "RESERVED_WORDS",
    "ExpressionLowerer",
    "LoweringError",
    "Lowering",
    "allows_contraction",
    "count_param_name",
    "count_param_names",
    "is_reserved",
    "lower",
    "lower_generic",
    "numpy_dtype",
    "target_for",
]

_LANG_VERSION = (2018, 2)

#: The C compiler flag that keeps ``a * b + c`` two roundings, set on a kernel
#: with an ``exact`` output (see :func:`allows_contraction`). GCC and clang both
#: take it, and GCC ignores the standard pragma below. loopy compiles with
#: ``-std=c99``, in which GCC does not contract anyway, but clang does, and the
#: flag says so rather than leaving it to the compiler.
NO_CONTRACTION_FLAG = "-ffp-contract=off"

#: GCC's own spelling of the flag in the source, behind a guard that keeps it
#: from any other compiler. The flag pins the build loopty runs; this pins the
#: source that ``loopty run --emit-code`` prints, which someone may compile by
#: hand with GCC in a GNU dialect, where GCC contracts by default whenever
#: ``-march`` gives it an FMA instruction. GCC documents the ``optimize``
#: pragma as meant for debugging; here it asks for less optimization, not
#: more, and for exactly the one thing the flag asks for.
GCC_NO_CONTRACTION_PRAGMA = (
    "#if defined(__GNUC__) && !defined(__clang__)\n"
    '#pragma GCC optimize ("fp-contract=off")\n'
    "#endif"
)

#: The pragmas that ask the same in the source, per target. C99's is honoured
#: by clang and ignored by GCC, which :data:`GCC_NO_CONTRACTION_PRAGMA` and the
#: flag cover. OpenCL C may contract by default and has no build option to stop
#: it, so there the pragma is the way.
NO_CONTRACTION_PRAGMAS = {
    "c": f"#pragma STDC FP_CONTRACT OFF\n{GCC_NO_CONTRACTION_PRAGMA}",
    "c-source": f"#pragma STDC FP_CONTRACT OFF\n{GCC_NO_CONTRACTION_PRAGMA}",
    "opencl": "#pragma OPENCL FP_CONTRACT OFF",
}

#: Where the pragma sorts among loopy's own preambles: after OpenCL's extension
#: pragmas (``00_``) and before the includes (``10_``).
_NO_CONTRACTION_TAG = "05_loopty_fp_contract"


class LoweringError(TypeError):
    """A term that cannot be handed to loopy as it stands.

    Raised rather than guessed at: a silently wrong lowering would be checked
    against nothing, since the isl obligations are stated about the term and not
    about the generated code.
    """


def count_param_name(counts: str, iname: str) -> str:
    """The domain parameter standing for a ragged bound; see :data:`COUNT_PARAM`."""
    return COUNT_PARAM.format(counts=counts, iname=iname)


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


# ``arrays_of`` used to live here, walking one expression for the array names in
# it. It was a second answer to "what does this statement touch?", and it was
# the wrong one: it was given ``stmt.expr`` and the assignee's subscripts and
# never the guard. :func:`loopty.flow.statement_accesses` is the only answer
# now, and the instruction dependencies below read it.


def reductions_of(expr: Any) -> tuple[Reduction, ...]:
    """Every :class:`~loopty.term.Reduction` in ``expr``, outermost first."""
    return tuple(node for node in walk(expr) if isinstance(node, Reduction))


def _reduction_nesting(expr: Any) -> list[tuple[Reduction, tuple[int, ...]]]:
    """Every reduction in ``expr``, with the reductions it is nested in.

    In the order of :func:`reductions_of`, so that a position names the same
    reduction in both, and each paired with the positions of the reductions
    enclosing it, outermost first.
    """
    out: list[tuple[Reduction, tuple[int, ...]]] = []

    def visit(node: Any, enclosing: tuple[int, ...]) -> None:
        if isinstance(node, Reduction):
            position = len(out)
            out.append((node, enclosing))
            visit(node.body, (*enclosing, position))
            return
        for child in _children(node):
            visit(child, enclosing)

    visit(expr, ())
    return out


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
        """``Reduction`` becomes ``lp.Reduction``; its domain is collected.

        The binders may have been renamed so that this reduction keeps its own
        domain; see :meth:`_Builder.plan_reductions`. The renaming is pushed
        while the body is walked, so every occurrence of the binder inside it
        follows, and popped afterwards, so a sibling reduction with the same
        written name is unaffected.
        """
        renaming = self.lowering.reduction_rename(expr)
        inames = tuple(renaming.get(name, name) for name in expr.inames)
        self.lowering.add_reduction_domain(expr, inames)
        if not renaming:
            return LoopyReduction(expr.op, inames, self.rec(expr.body))
        self.lowering.push_renaming(renaming)
        try:
            body = self.rec(expr.body)
        finally:
            self.lowering.pop_renaming()
        return LoopyReduction(expr.op, inames, body)

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
        return prim.Variable(self.lowering.rename(expr.name))

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

    ``reduction_inames`` gives the inames each reduction has in the generated
    kernel, keyed ``"S0:0"`` by its statement and its position in
    :func:`reductions_of`. They are its binders unless
    :meth:`_Builder.plan_reductions` had to rename them, and a schedule names a
    reduction's loops by them. ``contraction`` says whether the compiler may
    fuse ``a * b + c`` into one multiply-add; it is ``False`` when an output is
    compared bit for bit, see :func:`allows_contraction`.

    ``storage`` says, for every array over a polyhedral domain, which layout
    it is stored in, ``"box"`` or ``"packed"`` (:mod:`loopty.domain`), and
    ``tables`` names the argument holding the table of row starts of each
    packed one, which the executor computes and passes.
    """

    term: Term
    kernel: Any
    insn_ids: dict[str, str]
    array_args: tuple[str, ...]
    value_args: tuple[str, ...]
    outputs: tuple[str, ...]
    ragged: dict[str, str]
    target: str = "c"
    reduction_inames: dict[str, tuple[str, ...]] = field(default_factory=dict)
    contraction: bool = True
    storage: dict[str, str] = field(default_factory=dict)
    tables: dict[str, str] = field(default_factory=dict)

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

#: The identifiers C reserves by their spelling rather than by a list: every
#: name that starts with an underscore and a capital letter, or with two
#: underscores. That is where C puts its own later keywords (``_Bool``,
#: ``_Complex``, ``_Generic``, ``_Static_assert``, ``_Thread_local`` and the
#: rest, which C23 still accepts beside the new spellings), and where OpenCL C
#: puts its address-space and access qualifiers (``__global``, ``__kernel``,
#: ``__read_only``). A name of either shape is a keyword or a name the
#: implementation may define, so none of them can be declared by generated
#: code.
RESERVED_PREFIX = re.compile(r"_[A-Z_]")


def is_reserved(name: str) -> bool:
    """Whether generated C or OpenCL C code cannot declare ``name``.

    A word of :data:`RESERVED_WORDS`, or a name that starts the way
    :data:`RESERVED_PREFIX` says C reserves.
    """
    return name in RESERVED_WORDS or RESERVED_PREFIX.match(name) is not None


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
    is therefore renamed here, deterministically, with an ``_knl`` suffix, or
    with a ``k`` prefix for a name C reserves by its first two characters
    (:data:`RESERVED_PREFIX`), which no suffix can make legal. Renaming rather
    than refusing keeps a legal Python name legal: nothing
    outside the generated source refers to the kernel by this name, because
    callers hold the :class:`Lowering` and address arguments by name.
    """
    base = _sanitize(name)
    if not base or base[0].isdigit():
        base = f"k_{base}"
    elif RESERVED_PREFIX.match(base):
        # ``_Generic`` stays reserved with any suffix, so it gets a prefix.
        base = f"k{base}"
    taken = set(taken)
    if base not in taken and not is_reserved(base):
        return base
    candidate = f"{base}_knl"
    while candidate in taken or is_reserved(candidate):
        candidate = f"{candidate}_"
    return candidate


def _refuse_reserved_names(term: Term) -> None:
    """Refuse a term that would make the generated code use a keyword as a name.

    Every name the term chooses reaches the generated source verbatim: a
    parameter as an argument, a size as the value argument loopy infers from a
    shape (``Arr[Fin[long], Real]`` gives ``int32_t const long``), and a loop or
    reduction variable as the counter of a ``for`` (``for double in x.dom``
    gives ``for (int32_t double = 0; ...)``). Each is a compiler error about
    code the user never wrote, so all of them are checked, not only the
    parameters.

    None of them is renamed the way the kernel is (see :func:`_kernel_name`).
    A caller passes a parameter by name, and may pass a size the same way; a
    schedule names inames (``split("j", 2)``) and so do the ledger's messages.
    A rename would break each of those silently, where a refusal says what to
    change. The names loopty generates itself (``off_cnt``, ``nl_cnt_r``, a
    suffixed reduction binder) carry a prefix or a suffix and cannot be a
    keyword.

    A keyword is a word of :data:`RESERVED_WORDS` or a name of the shape C
    reserves, :data:`RESERVED_PREFIX`: ``for _Bool in x.dom`` fails in the
    compiler exactly as ``for double in x.dom`` does.
    """
    roles: dict[str, list[str]] = {
        "parameters": [name for name, _ in term.params],
        "sizes": list(term.sizes),
        "loop variables": [],
        "reduction variables": [],
    }
    for stmt in term.stmts:
        roles["loop variables"].extend(stmt.inames)
        for reduction in reductions_of(stmt.expr):
            roles["reduction variables"].extend(reduction.inames)
    found = []
    for role, names in roles.items():
        refused = sorted({name for name in names if is_reserved(_sanitize(name))})
        if refused:
            found.append(f"{role} {', '.join(refused)}")
    if found:
        raise LoweringError(
            f"{term.name} has names the generated code cannot use: "
            f"{'; '.join(found)}. These are reserved words in C or OpenCL C, "
            "or start with an underscore and a capital letter or with two "
            "underscores, which C reserves; rename them in the kernel (a "
            "parameter in its signature, a size in its annotations, a loop or "
            "reduction variable where it is bound)."
        )


def _refuse_free_name_sorts(term: Term) -> None:
    """Refuse a parameter whose sort is a free name, such as ``Var("float")``.

    A traced kernel is refused before it gets here (see
    :meth:`loopty.kernel.Kernel.trace`); a hand-built term is refused here,
    rather than by :func:`numpy_dtype` with "no numpy dtype for float", which
    does not say where the name came from or what to write instead.
    """
    found = free_name_sorts(term.params)
    if found:
        raise LoweringError(free_name_sorts_message(term.name, term.params, found))


def _refuse_bounds_over_reduction_binders(term: Term, builder: _Builder) -> None:
    """Refuse a nested reduction whose bound is read off an outer reduction's binder.

    ``reduce_sum(reduce_sum(val[q, j] for j in val.dom[q]) for q in val.dom)``
    is a term the analysis decides, and one loopy cannot be given. The inner
    bound ``cnt[q]`` is not affine, so it reaches isl as a parameter
    (``nl_cnt_q``), and the lowering computes such a parameter in a scalar
    temporary assigned inside the loop over its row (see :func:`_count_inits`).
    When the row is a statement's loop variable there is such a loop. When it is
    the binder of an enclosing reduction there is none: a reduction is one
    instruction's expression, and no other instruction can run inside its
    loop. Nothing assigned the parameter, loopy declared it a value argument,
    and the run failed with "value argument 'nl_cnt_q' was not given", which
    names neither the reduction nor a way out.

    An inner bound that is affine in the outer binder (``Fin[i + 1]``, the
    lower triangle) needs no temporary and still lowers; only a bound that had
    to be reflected, or that a hand-built term spells as a row length
    (:data:`COUNT_PARAM`), over an outer binder is refused.
    """
    reflected = dict(term.reflected)
    families = builder.counts_families
    for stmt in term.stmts:
        for outer in reductions_of(stmt.expr):
            binders = set(outer.inames)
            spelled = {
                spelling: (f"{counts}[{binder}]", {binder})
                for counts in families
                for binder in binders
                for spelling in count_param_names(counts, binder)
            }
            for inner in reductions_of(outer.body):
                for param in _domain_params(inner.domain):
                    if param in reflected:
                        depends = _names_in(reflected[param]) & binders
                        if not depends:
                            continue
                        bound = str(_plain(reflected[param]))
                    elif param in spelled:
                        bound, depends = spelled[param]
                    else:
                        continue
                    over = ", ".join(sorted(depends))
                    where = f" ({stmt.where})" if stmt.where else ""
                    raise LoweringError(
                        f"statement {stmt.id} of {term.name}{where} has a "
                        f"reduction over {', '.join(inner.inames)} bounded by "
                        f"{bound}, which depends on {over}, the binder of the "
                        "reduction it is nested in. A bound that is not affine "
                        "is computed inside the loop over the row it depends "
                        "on, and a reduction's binder has no loop another "
                        "instruction can run in, so this nesting cannot be "
                        f"lowered. Write the reduction over {over} as a for "
                        "loop that accumulates into the output "
                        f"('for {over} in ...: out[...] += reduce_sum(...)'), "
                        f"or keep each inner sum in a cell indexed by {over} "
                        f"('rows[{over}] = reduce_sum(...)' in that loop) and "
                        "reduce over those cells."
                    )


def _written_arrays(term: Term) -> tuple[str, ...]:
    """Arrays the term assigns to, in first-seen order."""
    names: list[str] = []
    for stmt in term.stmts:
        if stmt.assignee.array not in names:
            names.append(stmt.assignee.array)
    return tuple(names)


class _Builder:
    """Mutable scratch space for one lowering; see :func:`lower_generic`."""

    def __init__(
        self, term: Term, target: str, layouts: Mapping[str, str] | None = None
    ) -> None:
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
        #: The layout of every array over a domain, and the argument holding
        #: the table of row starts of each packed one; see Lowering.
        self.storage = _storage_plan(term, layouts)
        self.tables: dict[str, str] = {}
        self.extra_domains: list[isl.Set] = []
        self.extra_args: list[Any] = []
        self.value_args: list[str] = []
        self._ragged_bounds: dict[str, tuple[str, str]] | None = None
        #: Per reduction, the binders it had to rename, its domain over the
        #: names it ends up with, and the inames those renames introduce; see
        #: :meth:`plan_reductions`. A reduction is keyed by the statement it is
        #: planned in and its identity, because a term built by hand may hold
        #: one ``Reduction`` object in two statements, and each of them has to
        #: reduce over inames of its own.
        self.reduction_renames: dict[tuple[str, int], dict[str, str]] = {}
        self.reduction_domains: dict[tuple[str, int], isl.Set] = {}
        self.extra_inames: set[str] = set()
        self._renames: list[dict[str, str]] = []
        #: The statement whose expressions are being lowered, which is the
        #: other half of a reduction's key.
        self.statement: str | None = None
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
    def counts_families(self) -> tuple[str, ...]:
        """The counts array of every ragged parameter, in signature order."""
        names: list[str] = []
        for name in self.arr_types:
            if self.ragged_axis(name) is None:
                continue
            counts = self.counts_name(name)
            if counts not in names:
                names.append(counts)
        return tuple(names)

    @property
    def ragged_bound_params(self) -> dict[str, tuple[str, str]]:
        """Domain parameters standing for a ragged bound: name -> counts, iname.

        :func:`loopty.flow.ragged_bound_params`, the recognition the access
        collector lists the bound's read by, so that the parameter lowering
        assigns and the read the rules check are recognized alike. It keeps
        the bounds whose row is a loop variable of a statement, which are the
        ones a scalar temporary inside that loop can hold.
        """
        if self._ragged_bounds is None:
            # The counts families first: a ragged axis that names no counts
            # array is refused here, with the reason, as it always was.
            self.counts_families  # noqa: B018 - raises for an unnamed bound
            self._ragged_bounds = ragged_bound_params(self.term)
        return self._ragged_bounds

    def count_param_spellings(self, counts: str, iname: str) -> tuple[str, ...]:
        """Every name this one ragged bound could go by, most direct first."""
        names = list(count_param_names(counts, iname))
        for symbol, pair in self.ragged_bound_params.items():
            if pair == (counts, iname) and symbol not in names:
                names.append(symbol)
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

        The first of :data:`loopty.term.OFFSETS_CANDIDATES` that is a parameter
        of the term wins, which makes ``spmv(off, col, val, x, y)`` work with no
        configuration; when none is, an ``int32`` argument is added. The choice
        is :func:`loopty.term.declared_offsets`, the same one the access
        collector makes when it lists the read of the offsets.
        """
        if name in self.ragged:
            return self.ragged[name]
        counts = self.counts_name(name)
        declared = declared_offsets(self.term.params, counts)
        if declared is not None:
            self.ragged[name] = declared
            return declared
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
        typ = self.arr_types.get(name)
        if typ is not None and typ.domain is not None:
            return self.domain_access(name, typ, indices)
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

    def domain_access(
        self, name: str, typ: ArrType, indices: tuple[Any, ...]
    ) -> prim.Expression:
        """The reference into an array over a domain, through its layout.

        See the module docstring: boxed, a single domain is indexed as it is
        and a union's piece at its base in a flat buffer; packed, through the
        table of row starts, ``L[off_L[i] + j]``. The piece of a union is a
        Python integer in every access (the tracer refuses anything else), so
        its base is a term of the sizes alone.
        """
        domain = typ.domain
        union = isinstance(domain, Union)
        pieces = domain.pieces if union else (domain,)
        if union:
            position = indices[0]
            if not isinstance(position, int | np.integer):
                raise LoweringError(
                    f"{name} is indexed by {position!r} in its piece, and the "
                    f"piece of the union {domain} is chosen by an integer"
                )
            position = int(position)
            rest = tuple(indices[1:])
        else:
            position, rest = 0, tuple(indices)
        variable = prim.Variable(name)
        boxes = [tuple(_plain(extent) for extent in piece.box()) for piece in pieces]
        if self.storage[name] == "box":
            if not union:
                return prim.Subscript(variable, rest)
            base = _total(_volume(box) for box in boxes[:position])
            flat = _plus(base, linearize(rest, boxes[position]))
            return prim.Subscript(variable, (flat,))
        rows = [box[:-1] for box in boxes]
        base = _total(_volume(box) for box in rows[:position])
        entry = _plus(base, linearize(rest[:-1], rows[position]) if rest[:-1] else 0)
        start = prim.Subscript(prim.Variable(self.table_for(name)), (entry,))
        return prim.Subscript(variable, (_plus(start, rest[-1]),))

    def table_for(self, name: str) -> str:
        """The argument holding the table of row starts of a packed array.

        ``off_<name>``, suffixed while it is a name the kernel already uses, and
        an ``int32`` argument of no declared shape, which the executor fills
        from the array's domain (:meth:`loopty.domain.Fixed.table`).
        """
        if name in self.tables:
            return self.tables[name]
        taken = set(dict(self.term.params)) | set(self.term.sizes)
        taken |= {iname for stmt in self.term.stmts for iname in stmt.inames}
        taken |= {symbol for symbol, _ in self.term.reflected}
        taken |= {arg.name for arg in self.extra_args}
        candidate = f"off_{name}"
        while candidate in taken:
            candidate = f"{candidate}_"
        self.extra_args.append(
            lp.GlobalArg(
                candidate,
                np.dtype(np.int32),
                shape=None,
                is_input=True,
                is_output=False,
            )
        )
        self.tables[name] = candidate
        return candidate

    # }}}

    # {{{ reduction binders

    def plan_reductions(self) -> None:
        """Give every reduction binders that are its own in the generated kernel.

        loopy defines an iname once, with one domain, and :func:`_merge_domains`
        unions two domains over the same iname. For two *statements* that is
        right and :func:`_restore_narrower_domains` cuts each back with a
        predicate. For two reductions it is not: a reduction is one instruction's
        expression and cannot carry a predicate of its own, so a reduction over
        ``0 <= j < 2`` beside one over ``0 <= j < 4`` would silently become a sum
        over four points, reading two cells past the end of its input.

        Nor can two *instructions* share a reduction iname, even over one
        domain. loopy realizes a reduction as a loop inside the instruction, and
        a loop over an iname two instructions reduce over is one loop for both:
        when the second statement depends on the first, as ``y[1]`` after
        ``y[0]`` does, it has to run inside a loop that must finish before it
        starts, and loopy stops with a ``CycleError``. The same goes for a name
        another statement uses as a loop variable. So a reduction keeps its
        binders only when no other statement has them, as loop variables
        anywhere in the kernel or as the binders of an earlier reduction, and
        gets fresh inames otherwise; see note 8 in ``docs/loopy-notes.md``.

        Within one statement a reduction keeps the name the kernel wrote
        whenever the domain under that name is the one it already has, and gets
        a fresh iname when it is not. Keeping the name in the common case
        matters: ``Schedule.split("j", ...)``, the demos and the messages all
        name reduction inames as the source does, and renaming unconditionally
        would rename them all.

        A nested reduction's domain names its enclosing binders as parameters,
        so when an outer binder is renamed, the parameter follows it: the domain
        compared here and handed to loopy is stated over the names the kernel
        will actually have (see :meth:`add_reduction_domain`).

        A plan belongs to a reduction *in a statement*. The tracer builds a
        fresh :class:`~loopty.term.Reduction` for every statement, but a term
        built by hand may hold one object in two, and a plan keyed by the
        object alone was one plan for both: the second statement's overwrote
        the first's, both instructions reduced over one iname, and loopy
        stopped with the ``CycleError`` above. Keyed by statement as well, the
        shared object is planned twice, once in each, like two equal objects.
        """
        taken = set(self.term.sizes) | set(dict(self.term.params))
        taken |= {iname for stmt in self.term.stmts for iname in stmt.inames}
        taken |= {symbol for symbol, _ in self.term.reflected}
        for stmt in self.term.stmts:
            for reduction in reductions_of(stmt.expr):
                taken |= set(reduction.inames)
        owner: dict[str, str] = {}
        for stmt in self.term.stmts:
            for iname in stmt.inames:
                owner.setdefault(iname, stmt.id)
        for stmt in self.term.stmts:
            seen: dict[tuple[str, ...], list[tuple[isl.Set, tuple[str, ...]]]] = {}
            self._plan_in(stmt.id, stmt.expr, {}, owner, seen, taken)

    def _plan_in(
        self,
        stmt_id: str,
        expr: Any,
        renaming: Mapping[str, str],
        owner: dict[str, str],
        seen: dict[tuple[str, ...], list[tuple[isl.Set, tuple[str, ...]]]],
        taken: set[str],
    ) -> None:
        """Plan the reductions of ``expr``, outermost first.

        See :meth:`plan_reductions` for the rules.

        ``renaming`` is what the enclosing reductions' binders became, ``owner``
        the statement each name already belongs to, and ``seen`` the domains
        the names of this statement's reductions stand for so far.
        """
        for reduction in _outermost_reductions(expr):
            names = tuple(reduction.inames)
            domain = _rename_params(_domain_over(reduction.domain, names), renaming)
            known = seen.setdefault(names, [])
            chosen: tuple[str, ...] | None = None
            if all(owner.get(name, stmt_id) == stmt_id for name in names):
                for other, allocated in known:
                    if _same_set(domain, other):
                        chosen = allocated
                        break
                if chosen is None and not any(name in owner for name in names):
                    chosen = names
            if chosen is None:
                chosen = self._fresh_inames(names, taken)
            if not any(allocated == chosen for _other, allocated in known):
                known.append((domain, chosen))
            for name in chosen:
                owner.setdefault(name, stmt_id)
            rename = {
                old: new for old, new in zip(names, chosen, strict=True) if old != new
            }
            self.reduction_renames[stmt_id, id(reduction)] = rename
            self.reduction_domains[stmt_id, id(reduction)] = _domain_over(
                domain, chosen
            )
            self.extra_inames.update(chosen)
            self._plan_in(
                stmt_id, reduction.body, {**renaming, **rename}, owner, seen, taken
            )

    @staticmethod
    def _fresh_inames(names: Sequence[str], taken: set[str]) -> tuple[str, ...]:
        """Fresh loopy inames for a reduction whose binders are already spoken for."""
        out: list[str] = []
        for stem in names:
            suffix = 0
            candidate = f"{stem}_{suffix}"
            while candidate in taken:
                suffix += 1
                candidate = f"{stem}_{suffix}"
            taken.add(candidate)
            out.append(candidate)
        return tuple(out)

    def reduction_rename(
        self, reduction: Reduction, statement: str | None = None
    ) -> dict[str, str]:
        """How this reduction's binders were renamed, if they were.

        ``statement`` is the statement the reduction is read in, the one being
        lowered when it is not given.
        """
        key = (self.statement if statement is None else statement, id(reduction))
        return self.reduction_renames.get(key, {})

    def push_renaming(self, renaming: Mapping[str, str]) -> None:
        """Rename these variables while the reduction's body is lowered."""
        self._renames.append(dict(renaming))

    def pop_renaming(self) -> None:
        """Close the innermost renaming."""
        self._renames.pop()

    def rename(self, name: str) -> str:
        """The loopy name of a variable, innermost renaming first."""
        for renaming in reversed(self._renames):
            if name in renaming:
                return renaming[name]
        return name

    # }}}

    def add_reduction_domain(
        self, reduction: Reduction, inames: Sequence[str] | None = None
    ) -> None:
        """Record a reduction's iteration domain as a domain of the kernel.

        A planned reduction contributes the domain :meth:`plan_reductions`
        stated over the kernel's names, in which a renamed enclosing binder is
        renamed too; the domain under the written names would tie the inner
        loop to the outer reduction of a different statement.
        """
        planned = self.reduction_domains.get((self.statement, id(reduction)))
        if planned is not None:
            self.extra_domains.append(planned)
            return
        names = tuple(reduction.inames) if inames is None else tuple(inames)
        self.extra_domains.append(_domain_over(reduction.domain, names))

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


def _storage_plan(
    term: Term, layouts: Mapping[str, str] | None
) -> dict[str, str]:
    """The layout of every array over a domain: ``"box"`` unless ``layouts`` says.

    ``layouts`` may name only arrays over a domain, and only a layout of
    :data:`loopty.domain.STORAGES`; anything else is refused, since a dense or
    ragged array has one layout and no choice to make.
    """
    arrays = {name: typ for name, typ in term.params if isinstance(typ, ArrType)}
    plan = {name: "box" for name, typ in arrays.items() if typ.domain is not None}
    for name, storage in (layouts or {}).items():
        if name not in plan:
            raise LoweringError(
                f"{name} is not an array over a Where, Sigma or union domain of "
                f"{term.name}, so it has no layout to choose; a dense or ragged "
                "array is stored one way"
            )
        if storage not in STORAGES:
            raise LoweringError(
                f"an array over a domain is stored "
                f"{' or '.join(map(repr, STORAGES))}, not {storage!r}"
            )
        plan[name] = storage
    return plan


def _refuse_packed_without_interval_rows(term: Term, builder: _Builder) -> None:
    """Refuse a packed array whose domain has a row that is not an interval.

    The packed layout stores a row as the run from its first column to its
    last, and addresses ``(r, j)`` as ``table[r] + j``, so a column the row
    skips (``j % 2 == 0``) would take a cell it does not have. Asked of isl
    for every size (:meth:`loopty.domain.Polyhedron.rows_are_intervals`).
    """
    for name, storage in builder.storage.items():
        if storage != "packed":
            continue
        domain = builder.arr_types[name].domain
        pieces = domain.pieces if isinstance(domain, Union) else (domain,)
        for piece in pieces:
            if piece.rows_are_intervals():
                continue
            raise LoweringError(
                f"{name} of {term.name} is stored packed, a row at a time from "
                f"its first column, and a row of {piece} is not an interval "
                "for every size: a constraint with a remainder or a floor "
                "division skips columns inside a row. Store it boxed."
            )


def _volume(extents: Sequence[Any]) -> Any:
    """The number of cells of a box, folded where the extents are integers."""
    total: Any = 1
    for extent in extents:
        if isinstance(total, int) and isinstance(extent, int):
            total *= extent
        elif isinstance(total, int) and total == 1:
            total = extent
        else:
            total = prim.Product((total, extent))
    return total


def _plus(left: Any, right: Any) -> Any:
    """``left + right``, folded where either is the integer ``0``."""
    if isinstance(left, int) and left == 0:
        return right
    if isinstance(right, int) and right == 0:
        return left
    if isinstance(left, int) and isinstance(right, int):
        return left + right
    return prim.Sum((left, right))


def _total(terms: Iterator[Any] | Sequence[Any]) -> Any:
    """The sum of ``terms``, folded as :func:`_plus` folds."""
    out: Any = 0
    for term in terms:
        out = _plus(out, term)
    return out


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

    def rename(self, name: str) -> str:
        return name

    def reduction_rename(
        self, reduction: Reduction, statement: str | None = None
    ) -> dict[str, str]:
        return {}

    def add_reduction_domain(
        self, reduction: Reduction, inames: Sequence[str] | None = None
    ) -> None:
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


def _rename_params(domain: isl.Set, renaming: Mapping[str, str]) -> isl.Set:
    """``domain`` with each parameter ``renaming`` names renamed.

    How a nested reduction's domain follows an enclosing binder that
    :meth:`_Builder.plan_reductions` renamed: the binder is a parameter of the
    inner domain, and it has to name the loop the renamed binder became.
    """
    for position, name in enumerate(_domain_params(domain)):
        if name in renaming:
            domain = domain.set_dim_name(isl.dim_type.param, position, renaming[name])
    return domain


def _outermost_reductions(expr: Any) -> tuple[Reduction, ...]:
    """The reductions of ``expr`` that no other reduction in it encloses."""
    if isinstance(expr, Reduction):
        return (expr,)
    return tuple(
        reduction
        for child in _children(expr)
        for reduction in _outermost_reductions(child)
    )


def _domain_params(domain: isl.Set) -> tuple[str, ...]:
    """Parameter names of an isl set."""
    return tuple(domain.get_var_names(isl.dim_type.param))


def _same_set(left: isl.Set, right: isl.Set) -> bool:
    """Are two domains the same set, once their parameters are aligned?"""
    try:
        first = left.align_params(right.get_space())
        second = right.align_params(first.get_space())
    except Exception:  # pragma: no cover - isl declines to align
        return False
    return bool(first.is_equal(second))


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


def _depth_cuts(term: Term) -> dict[str, frozenset[int]]:
    """Where each statement's domain is cut so that every loop is defined once.

    loopy defines an iname in exactly one domain, and a statement's domain is
    over every loop around it. Two statements at different depths of one loop,
    ``y[r] = y[r] + a[r, j]`` inside the loop over ``j`` and ``z[r] = 1.0``
    after it, contributed ``{ [r, j] }`` and ``{ [r] }``, and loopy refused the
    second for redefining ``r`` with a bare ``RuntimeError``. So a statement's
    domain is cut after every loop at which another statement leaves its nest:
    after position ``k`` when the other's loops agree with its first ``k + 1``
    and then stop or go on into a different loop. ``{ [r, j] }`` becomes
    ``{ [r] }`` and ``[r] -> { [j] }``, the first merges with the other
    statement's domain, and the nest is the one the source wrote.

    A traced term gives every loop an iname of its own (a second ``for r`` is
    ``r_0``), so agreeing on a name is agreeing on a loop. A statement whose
    loops need no cut keeps its single domain, as every statement did before.
    """
    cuts: dict[str, set[int]] = {stmt.id: set() for stmt in term.stmts}
    for stmt in term.stmts:
        for other in term.stmts:
            if other is stmt:
                continue
            shared = 0
            for mine, theirs in zip(stmt.inames, other.inames, strict=False):
                if mine != theirs:
                    break
                shared += 1
            if 0 < shared < len(stmt.inames):
                cuts[stmt.id].add(shared - 1)
    return {key: frozenset(value) for key, value in cuts.items()}


def _outer_part(domain: isl.Set, keep: int) -> isl.Set:
    """``domain`` over its first ``keep`` dimensions: the loops outside the rest.

    The constraints that mention an inner dimension are dropped, not projected
    out. Projecting ``j`` out of ``0 <= j < m`` leaves ``m >= 1`` behind, which
    says when the inner loop has an iteration and bounds nothing about ``r``.
    Two inner loops side by side, over ``m`` and over ``p``, would then give the
    loop over ``r`` the union of ``m >= 1`` and ``p >= 1``, which is not convex
    and so cannot be one loop: :func:`_merge_domains` would refuse the kernel.
    Beside a statement at the outer depth the union is the whole range again,
    and the constraint would only come back as a predicate on the inner
    statement. What is dropped is not lost: the innermost domain of the
    statement keeps every constraint, so the statement's instances are its
    domain, exactly. So are the constraints on the sizes alone (``m >= 0``),
    which bound no loop: kept, they would make this range differ from the same
    loop's range in a statement that has no ``m``, and the merged loop would
    carry them as a predicate.

    Dropping can leave a loop without a bound when its bound was written
    through an inner loop (``0 <= r <= j < n``), which no loop a person writes
    does. The projection is used then, which is a loop range too.
    """
    total = domain.dim(isl.dim_type.set)
    if keep >= total:
        return domain
    inner = total - keep
    dropped = (
        domain.drop_constraints_involving_dims(isl.dim_type.set, keep, inner)
        .project_out(isl.dim_type.set, keep, inner)
        .drop_constraints_not_involving_dims(isl.dim_type.set, 0, keep)
    )
    if dropped.is_bounded():
        return dropped
    return domain.project_out(isl.dim_type.set, keep, inner)


def _statement_domains(
    stmt: Stmt,
    ragged_bounds: Mapping[str, tuple[str, str]],
    cuts: frozenset[int] = frozenset(),
) -> list[isl.Set]:
    """The domains a statement contributes, one per stretch of its loop nest.

    loopy refuses a domain whose parameter is written inside a loop the same
    domain provides ("domain parameter may not be written inside a domain
    dependent on it"), and a ragged bound is exactly that: ``cnt_r`` is assigned
    inside the ``r`` loop, and it bounds ``j``. The cure is the shape loopy wants
    anyway, a nest of domains: ``{ [r] : 0 <= r < n }`` outside, and
    ``[r, cnt_r] -> { [j] : 0 <= j < cnt_r }`` inside it.

    The nest is also cut at each position of ``cuts``, where another statement
    leaves the loop (see :func:`_depth_cuts`). Each stretch of loops is a domain
    over those loops, with the loops outside it as parameters and the loops
    inside it dropped (see :func:`_outer_part`); a row length assigned inside
    the stretch, or deeper, is forgotten from every stretch but the last. A
    dense statement no other one leaves keeps its single domain.
    """
    inames = tuple(stmt.inames)
    full = _domain_over(stmt.domain, inames)
    params = set(_domain_params(full))
    present = [p for p in ragged_bounds if p in params]
    rows = {
        param: inames.index(ragged_bounds[param][1])
        for param in present
        if ragged_bounds[param][1] in inames
    }
    positions = set(cuts)
    if rows:
        positions.add(max(rows.values()))
    ends = sorted(p for p in positions if 0 <= p < len(inames) - 1)
    if not ends:
        return [full]

    out: list[isl.Set] = []
    start = 0
    for end in ends:
        stretch = _outer_part(full, end + 1)
        # A row length whose row is not one of these loops at all is forgotten
        # too: no loop around this statement assigns it.
        stretch = _forget_params(
            stretch, [param for param in present if rows.get(param, start) >= start]
        )
        out.append(_domain_over(stretch, inames[start : end + 1]))
        start = end + 1
    out.append(_domain_over(full, inames[start:]))
    return out


def _refuse_redefined_inames(domains: Sequence[isl.Set], term: Term) -> None:
    """Refuse a kernel in which two domains still define one loop.

    What is left once :func:`_merge_domains` has merged the domains over the
    same loops and :func:`_depth_cuts` has cut every nest another statement
    leaves: a term built by hand that uses one name for two different loops,
    such as ``j`` inside ``r`` in one statement and on its own in another.
    loopy would refuse it with a bare ``RuntimeError`` about a generated
    domain; this names the loop and the two domains.
    """
    defined: dict[str, isl.Set] = {}
    for domain in domains:
        for iname in domain.get_var_names(isl.dim_type.set):
            earlier = defined.setdefault(iname, domain)
            if earlier is domain:
                continue
            raise LoweringError(
                f"{term.name} uses the loop variable {iname} for two different "
                f"loops, whose domains {earlier} and {domain} cannot be one "
                "domain: loopy defines each loop variable once. Give one of "
                "the loops a name of its own."
            )


def _count_inits(
    term: Term, builder: _Builder, domains: Sequence[isl.Set]
) -> tuple[list[Any], dict[str, str], dict[str, str]]:
    """Instructions assigning the ragged bound parameters, their ids, and reads.

    A domain parameter named ``cnt_r`` (see :data:`COUNT_PARAM`) is a ragged
    bound: the length of row ``r`` of whichever array has ``cnt`` as its counts
    family. It is emitted as a scalar temporary inside the ``r`` loop, computed
    from the counts array when that is a parameter and from the offsets when it
    is not. loopy then generates ``for (j = 0; j < cnt_r; ++j)``.

    The three results are the instructions, the id of each parameter's
    instruction, and the array each instruction reads, which is what
    :func:`lower_generic` orders it by.
    """
    wanted: dict[str, str] = {}
    for domain in domains:
        for param in _domain_params(domain):
            wanted.setdefault(param, param)

    params = dict(term.params)
    insns: list[Any] = []
    ids: dict[str, str] = {}
    reads: dict[str, str] = {}
    for name in builder.arr_types:
        axis = builder.ragged_axis(name)
        if axis is None:
            continue
        counts = builder.counts_name(name)
        for stmt in term.stmts:
            for iname in stmt.inames:
                candidates = [
                    name
                    for name in builder.count_param_spellings(counts, iname)
                    if name in wanted
                ]
                if not candidates or candidates[0] in ids:
                    continue
                param = candidates[0]
                row = prim.Variable(iname)
                insn_id = f"{param}_init"
                if counts in params:
                    value: Any = prim.Subscript(prim.Variable(counts), (row,))
                    reads[insn_id] = counts
                else:
                    offsets = builder.offsets_for(name)
                    value = prim.Subscript(
                        prim.Variable(offsets), (row + 1,)
                    ) - prim.Subscript(prim.Variable(offsets), (row,))
                    reads[insn_id] = offsets
                enclosing = stmt.inames[: stmt.inames.index(iname) + 1]
                insns.append(
                    lp.Assignment(
                        assignee=prim.Variable(param),
                        expression=value,
                        id=insn_id,
                        within_inames=frozenset(enclosing),
                        # The row loop and the loops around it, and none that
                        # loopy would infer from a writer of the counts; see
                        # the statements' instructions in lower_generic.
                        within_inames_is_final=True,
                        temp_var_type=lp.Optional(np.dtype(np.int32)),
                    )
                )
                ids[param] = insn_id
    return insns, ids, reads


def _loops_before_fiber(loops: isl.Set, param: str) -> int:
    """How many loops of a statement enclose the first loop ``param`` bounds.

    ``loops`` is the statement's domain over its loop variables. A loop over a
    ragged fiber reads its bound where it starts, once per iteration of the
    loops around it, so those are the loops across whose iterations the body
    sees a row length change. When ``param`` bounds none of them, all of them
    count.
    """
    index = loops.find_dim_by_name(isl.dim_type.param, param)
    total = loops.dim(isl.dim_type.set)
    if index < 0:
        return total
    for position in range(total):
        if bounds_dimension(loops, position, index):
            return position
    return total


def _refuse_bounds_rewritten_in_a_loop(
    term: Term,
    count_insns: Sequence[Any],
    count_params: Mapping[str, str],
    count_reads: Mapping[str, str],
    uses: Mapping[str, Sequence[tuple[Stmt, int]]],
) -> None:
    """Refuse a row length that a loop inside its row would read stale.

    The lowered kernel computes the length of row ``r`` once per row, in the
    loop over ``r`` (:func:`_count_inits`). The body reads it where the loop
    over the row's fiber starts, once per iteration of every loop around that
    one. The two agree unless a statement rewrites the cell the length is read
    from inside a loop that sits between the two and also encloses a statement
    bounded by it::

        for r in y.dom:
            for i in x.dom:
                y[r] = y[r] + x[i] + reduce_sum(val[r, j] for j in val.dom[r])
                cnt[r] = 1

    From the second ``i`` on, the body sums a row of the new length and the
    lowered kernel one of the old. :func:`lower_generic` refuses the rewrite
    that comes between two statements in the body's order; this is the same
    stale length across the iterations of a loop the two share, whichever of
    them comes first in the body, and it is refused the same way. Two rewrites
    are not refused. One inside the loop over the fiber itself: that loop reads
    its bound once, when it starts, in the body as in the lowered kernel. And
    one of another row's cell (:func:`_writes_its_row`): ``cnt[r + 1] = 0``
    inside row ``r`` changes a length that both runs read when row ``r + 1``
    starts.

    ``uses`` maps a bound's instruction to each statement bounded by it, with
    how many of the statement's loops enclose the start of the loop the bound
    bounds (:func:`_loops_before_fiber`, or every loop for a sum).
    """
    rows = {insn.id: len(insn.within_inames) for insn in count_insns}
    families = set(counts_families(term))
    for count_id, users in uses.items():
        source = count_reads[count_id]
        row = rows[count_id]
        # ``cnt[r]``, or ``off[r]`` and ``off[r + 1]`` when the counts are not
        # a parameter (see _count_inits).
        shifts = (0,) if source in families else (0, 1)
        for writer in term.stmts:
            if writer.assignee.array != source:
                continue
            for user, fiber in users:
                shared = 0
                for mine, theirs in zip(user.inames, writer.inames, strict=False):
                    if mine != theirs:
                        break
                    shared += 1
                if min(shared, fiber) <= row:
                    continue
                if not _writes_its_row(writer, row, shifts):
                    continue
                loop = user.inames[row]
                raise LoweringError(
                    f"statement {user.id} is bounded by the row length "
                    f"{count_params[count_id]}, which is computed from {source} "
                    f"once per row, before the loop over {loop}; {writer.id} "
                    f"rewrites that row's {source} inside that loop, so from its "
                    f"second iteration on {user.id} would see the old length "
                    "where the body reads the new one. Rewrite it outside the "
                    f"loop over {loop}, or in a kernel of its own."
                )


def _writes_its_row(writer: Stmt, row: int, shifts: Sequence[int]) -> bool:
    """Can ``writer`` write a cell its own row's length is read from?

    The row is ``writer``'s loop variable at depth ``row - 1``, and the cells
    are that variable plus each of ``shifts``: ``cnt[r]``, or ``off[r]`` and
    ``off[r + 1]``. Asked of the statement's domain, so a write a guard masks
    where isl can state the guard is not counted. An index isl cannot state
    reaches every cell (:func:`loopty.flow.access_relation`), so the answer
    errs towards a refusal.
    """
    indices = tuple(writer.assignee.indices)
    if len(indices) != 1:
        return True
    written = access_relation(writer.inames, writer.domain, indices)
    source = ", ".join(writer.inames)
    iname = writer.inames[row - 1]
    for shift in shifts:
        cell = isl.Map(f"{{ [{source}] -> [{iname} + {shift}] }}")
        cell = cell.align_params(written.get_space())
        if not written.align_params(cell.get_space()).intersect(cell).is_empty():
            return True
    return False


def lower_generic(
    term: Term, target: str = "c", layouts: Mapping[str, str] | None = None
) -> Lowering:
    """Lower ``term``, keeping the map from term statements to instructions.

    This is :func:`lower` plus bookkeeping. A schedule needs to say *which term
    statement* it would reorder, and an executor needs to know which arguments
    the kernel writes; both are recorded here rather than recovered by matching
    names against generated code.

    ``layouts`` chooses the layout of an array over a polyhedral domain,
    ``{"L": "packed"}``; every such array is boxed otherwise.
    """
    _refuse_reserved_names(term)
    _refuse_free_name_sorts(term)
    builder = _Builder(term, target, layouts)
    _refuse_packed_without_interval_rows(term, builder)
    _refuse_bounds_over_reduction_binders(term, builder)
    builder.plan_reductions()
    expr = builder.expr
    ragged_bounds = builder.ragged_bound_params

    domains: list[isl.Set] = []
    insns: list[Any] = []
    insn_ids: dict[str, str] = {}
    writes_before: dict[str, list[str]] = {}
    reads_before: dict[str, list[str]] = {}

    #: The domains each statement contributed, kept so that a statement whose
    #: domain is widened by :func:`_merge_domains` can get it back as a
    #: predicate; see :func:`_restore_narrower_domains`.
    own_domains: dict[str, list[isl.Set]] = {}
    cuts = _depth_cuts(term)

    for stmt in term.stmts:
        if not isinstance(stmt, Stmt):  # pragma: no cover - defensive
            raise LoweringError(f"not a statement: {stmt!r}")
        mine = _statement_domains(stmt, ragged_bounds, cuts[stmt.id])
        own_domains[_sanitize(stmt.id)] = list(mine)
        domains.extend(mine)

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
        # A reduction's plan is looked up by the statement it is lowered in;
        # see _Builder.plan_reductions.
        builder.statement = stmt.id
        assignee = expr(stmt.assignee)
        body = expr(stmt.expr)

        # What this statement reads, from the one collector every rule uses:
        # the right-hand side, the subscripts of the assignee, the guard, the
        # accumulated cell, and the offsets a ragged access, read or written,
        # indexes through. A name missing here is a dependence edge that is
        # never drawn, so the list is not written out a second time. The last
        # of those is the edge loopy's single-writer heuristic used to supply
        # when one statement wrote the offsets; the dependences below are
        # final, so it has to come from the collector, and it follows the body.
        read_arrays = {
            array
            for array, _indices, kind, _inames, _domain in statement_accesses(
                stmt, term
            )
            if kind in ("read", "acc")
        }
        written = stmt.assignee.array

        # Order the statements by their data: a statement runs after every
        # earlier one it could read from, write over, or overwrite the input of.
        # This is the whole of the order within one iteration, which is why the
        # instruction's dependences can be final.
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
                # Final too: these are every loop around the statement, and no
                # others. Left open, loopy adds to an instruction the loops of
                # every instruction that writes what it reads, less the loops
                # the writer's subscripts name, so ``z[r] = z[r] + y[r]`` after
                # the loop over ``j`` that accumulates ``y[r]`` was put inside
                # that loop, ran once per ``j``, and never ran for a row whose
                # loop over ``j`` is empty. See note 12 in docs/loopy-notes.md.
                within_inames_is_final=True,
                depends_on=frozenset(depends),
                # Final, so that loopy adds nothing to it. Its single-writer
                # heuristic makes an instruction depend on the only writer of
                # anything it reads, wherever that writer is in the body. When
                # the writer comes later and feeds this statement only across
                # an iteration of an enclosing loop, as the pressure a wave
                # update reads from the previous time level does, the edge
                # points against the one drawn here and loopy refuses the
                # cycle. An instruction dependence orders two statements within
                # one iteration; the order across iterations is the loop's.
                depends_on_is_final=True,
                predicates=predicates,
            )
        )
        for array in read_arrays:
            reads_before.setdefault(array, []).append(insn_id)
        writes_before.setdefault(written, []).append(insn_id)

    domains.extend(builder.extra_domains)

    count_insns, count_ids, count_reads = _count_inits(term, builder, domains)
    if count_insns:
        # A bound's instruction is ordered by the same rule as the statements,
        # at the place of the first statement that needs it: after every
        # earlier writer of the array it reads, and before every later one.
        # Left to loopy's single-writer heuristic, it waited for that writer
        # wherever it was in the body, and when the writer came later (a
        # statement that rewrites the offsets after a ragged loop has read
        # through them) the three instructions made a cycle.
        #
        # A bound is computed once, so a statement that needs it after its array
        # has been rewritten would see the old row length, and through the new
        # offsets when those are what was rewritten. That order is refused.
        by_id = {insn.id: insn for insn in insns}
        count_params = {count_id: param for param, count_id in count_ids.items()}
        count_depends: dict[str, frozenset[str]] = {}
        first_use: dict[str, str] = {}
        rewritten: dict[str, str] = {}
        writers: dict[str, list[str]] = {}
        uses: dict[str, list[tuple[Stmt, int]]] = {}
        for stmt in term.stmts:
            insn = by_id[insn_ids[stmt.id]]
            loops = _domain_over(stmt.domain, stmt.inames)
            needed = {
                count_ids[param]
                for param in _domain_params(loops)
                if param in count_ids
            }
            for count_id in needed:
                uses.setdefault(count_id, []).append(
                    (stmt, _loops_before_fiber(loops, count_params[count_id]))
                )
            for reduction in reductions_of(stmt.expr):
                for param in _domain_params(reduction.domain):
                    if param in count_ids:
                        needed.add(count_ids[param])
                        # A sum is evaluated inside every loop of its statement.
                        uses.setdefault(count_ids[param], []).append(
                            (stmt, len(stmt.inames))
                        )
            stale = sorted(needed & rewritten.keys())
            if stale:
                count_id = stale[0]
                source = count_reads[count_id]
                raise LoweringError(
                    f"statement {stmt.id} is bounded by the row length "
                    f"{count_params[count_id]}, which is computed from {source} "
                    f"once, where {first_use[count_id]} first needs it; "
                    f"{rewritten[count_id]} rewrites {source} before {stmt.id} "
                    f"runs, so {stmt.id} would see the old length. Rewrite "
                    f"{source} after the last statement bounded by it, or in a "
                    "kernel of its own."
                )
            for count_id in needed - count_depends.keys():
                count_depends[count_id] = frozenset(
                    writers.get(count_reads[count_id], ())
                )
                first_use[count_id] = stmt.id
            written = stmt.assignee.array
            overwrites = {
                count_id
                for count_id in count_depends
                if count_reads[count_id] == written
            }
            for count_id in overwrites:
                rewritten.setdefault(count_id, stmt.id)
            needed |= overwrites
            if needed:
                by_id[insn.id] = insn.copy(depends_on=insn.depends_on | needed)
            writers.setdefault(written, []).append(insn.id)
        _refuse_bounds_rewritten_in_a_loop(
            term, count_insns, count_params, count_reads, uses
        )
        insns = [by_id[insn.id] for insn in insns]
        # A bound no statement needs keeps the heuristic, as before.
        count_insns = [
            insn.copy(depends_on=count_depends[insn.id], depends_on_is_final=True)
            if insn.id in count_depends
            else insn
            for insn in count_insns
        ]
        insns = count_insns + insns

    merged = _nest_domains(_merge_domains(domains))
    _refuse_redefined_inames(merged, term)
    insns = _restore_narrower_domains(insns, own_domains, merged)

    args, array_args, value_args, outputs = _arguments(
        term, builder, domains, count_ids, insns
    )

    contraction = allows_contraction(term)
    kernel = lp.make_kernel(
        merged,
        insns,
        args,
        target=target_for(target),
        lang_version=_LANG_VERSION,
        name=_kernel_name(term.name, [arg.name for arg in args]),
        preambles=() if contraction else _no_contraction_preambles(target),
    )
    if not contraction and target in ("c", None):
        kernel = lp.set_options(kernel, build_options=[NO_CONTRACTION_FLAG])
    assumptions = _scalar_assumptions(term, {arg.name for arg in args})
    if assumptions is not None:
        # Not ``if assumptions:``: truthiness on an isl set is ``__len__``,
        # which islpy deprecates for a BasicSet.
        kernel = lp.assume(kernel, assumptions)
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
        reduction_inames=_reduction_inames(term, builder),
        contraction=contraction,
        storage=dict(builder.storage),
        tables=dict(builder.tables),
    )


def _reduction_inames(term: Term, builder: _Builder) -> dict[str, tuple[str, ...]]:
    """Each reduction's inames in the generated kernel; see :class:`Lowering`."""
    out: dict[str, tuple[str, ...]] = {}
    for stmt in term.stmts:
        for position, reduction in enumerate(reductions_of(stmt.expr)):
            renaming = builder.reduction_rename(reduction, stmt.id)
            out[f"{stmt.id}:{position}"] = tuple(
                renaming.get(name, name) for name in reduction.inames
            )
    return out


def allows_contraction(term: Term) -> bool:
    """Whether the compiled code may fuse ``a * b + c`` into one multiply-add.

    Not when any output is ``exact``. A fused multiply-add rounds once where
    the native run, which is Python and numpy arithmetic, rounds after the
    multiplication and again after the addition, so the two can differ in the
    last bit, and an ``exact`` output is compared bit for bit. The class is the
    one the differential test judges the output by
    (:func:`loopty.executor.exactness_of_output`): the element sort joined with
    the accumulations that write it, so ``Real.exact`` pins contraction off and
    ``Real`` leaves it to the compiler. The pin covers the whole kernel, since
    a compiler flag and a file-scope pragma cannot pick out one output. Which
    compilers contract when is note 9 in ``docs/loopy-notes.md``.
    """
    from loopty.executor import exactness_of_output

    return not any(
        exactness_of_output(term, None, name) == "exact"
        for name in _written_arrays(term)
    )


def _no_contraction_preambles(target: str | None) -> tuple[tuple[str, str], ...]:
    """The preamble that asks the target's compiler not to contract, if any."""
    pragma = NO_CONTRACTION_PRAGMAS.get(target or "c")
    return () if pragma is None else ((_NO_CONTRACTION_TAG, pragma),)


def _scalar_assumptions(term: Term, declared: set[str]) -> isl.BasicSet | None:
    """What the sorts of the scalar parameters say about them, for loopy.

    ``i: Fin[n]`` is a *declaration* that ``0 <= i < n``, and loopy has no other
    way to learn it: a scalar is a plain value argument, so a kernel writing
    ``x[i]`` fails loopy's own bounds check ("could not establish ... is a
    subset of ...") for the legal call as readily as for the illegal one, which
    is a blanket refusal rather than a safety net.

    Telling loopy is sound because the declaration is enforced where it can be:
    :func:`loopty.contract.scalar_parameters` refuses an argument outside its
    sort at both entry points that run a kernel, so no run reaches here with an
    ``i`` the assumption is false of. ``Nat`` contributes non-negativity and
    ``Int`` nothing. A constraint naming something loopy does not have as a
    parameter is dropped rather than guessed at.
    """
    from loopty.contract import sort_bound

    pieces: list[str] = []
    names: set[str] = set()
    for name, typ in term.params:
        if isinstance(typ, ArrType) or name not in declared:
            continue
        sort = typ
        base = getattr(sort, "base", None)  # a lanky refinement T & prop
        if base is not None and base is not sort:
            sort = base
        bound = getattr(sort, "bound", None)
        if bound is not None and not isinstance(bound, int | np.integer):
            # ``Fin[n]`` or ``Fin[n + 1]`` with a symbolic bound: sort_bound
            # cannot resolve the sizes without the call's arguments, but loopy
            # has every size the bound names as a parameter, so the bound is
            # stated as an affine constraint over them.
            from lanky.terms import free_variables

            from loopty import idx

            free = set(free_variables(bound))
            if not idx.is_affine(bound) or not free <= declared:
                continue
            pieces.append(f"{name} >= 0")
            pieces.append(f"{name} < {idx.isl_expr(bound)}")
            names |= {name, *free}
            continue
        limits = sort_bound(typ, {})
        if limits is not None:
            low, high = limits
            pieces.append(f"{name} >= {low}")
            if high is not None:
                pieces.append(f"{name} < {high}")
            names.add(name)
    if not pieces:
        return None
    # A set rather than the text ``lp.assume`` also accepts: that path wraps the
    # constraint in the kernel's own outer parameters, and a scalar argument is
    # not one of them until this assumption introduces it.
    params = ", ".join(sorted(names))
    return isl.BasicSet(f"[{params}] -> {{ : {' and '.join(pieces)} }}")


def _merge_domains(domains: Sequence[isl.Set]) -> list[isl.Set]:
    """One domain per tuple of inames, because loopy defines each iname once.

    Every statement contributes its own domain, so two statements in the same
    loop contribute the same domain twice and loopy refuses the second one:
    "redefines iname 't' that is part of a previous domain". A reduction over a
    fiber contributes the same inner domain as the statements inside that fiber,
    for the same reason. Identical domains are therefore dropped.

    Two domains over the same inames that are genuinely different sets are
    merged by union, which is what a ``when`` guard produces: the guarded
    statement's domain is narrower and the loop has to run over the wider of the
    two. The narrower statement does not simply inherit the union;
    :func:`_restore_narrower_domains` gives it back its own domain as an
    instruction predicate, so the loop is the union and the statement is not. A
    union that is not convex is a nest loopy cannot express with one iname, and
    saying so here names the domains rather than letting loopy fail later about
    a generated instruction.
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


def _nest_domains(domains: Sequence[isl.Set]) -> list[isl.Set]:
    """The domains in the order loopy reads their nesting from.

    loopy has no explicit tree of domains. It walks the list and nests a domain
    inside the one before it when its parameters name that domain's inames,
    and otherwise climbs out until it finds a domain it depends on or reaches
    the top (``LoopKernel.parents_per_domain``). So the row-length domain
    ``[q, nl_cnt_q] -> { [j] : ... }`` of a ragged loop has to come after the
    domain of ``q`` with nothing unrelated in between. The domains are
    collected statement by statement, with the reductions' domains after all
    of them, and a kernel whose ragged loop is followed by a second loop gave
    ``[{q}, {p}, {j over q}]``: loopy made the ``j`` domain a root, and only
    got its loop right by moving ``q`` out of the parameters again inside
    ``combine_domains``, through a call islpy deprecates.

    Each domain is put right after the domain whose inames it names as
    parameters, the deepest one when it names several, and the order is
    otherwise the order the domains came in. A list that was already nested
    is returned in the same order.
    """
    inames = [set(domain.get_var_names(isl.dim_type.set)) for domain in domains]
    params = [set(domain.get_var_names(isl.dim_type.param)) for domain in domains]

    def owners(k: int) -> list[int]:
        return [i for i in range(len(domains)) if i != k and inames[i] & params[k]]

    depth: dict[int, int] = {}

    def depth_of(k: int, visiting: frozenset[int] = frozenset()) -> int:
        if k not in depth:
            above = [i for i in owners(k) if i not in visiting]
            depth[k] = 1 + max(
                (depth_of(i, visiting | {k}) for i in above), default=-1
            )
        return depth[k]

    children: dict[int | None, list[int]] = {}
    for k in range(len(domains)):
        above = owners(k)
        parent = max(above, key=lambda i: (depth_of(i), i)) if above else None
        children.setdefault(parent, []).append(k)

    order: list[int] = []

    def place(k: int) -> None:
        order.append(k)
        for child in children.get(k, ()):
            if child not in order:
                place(child)

    for root in children.get(None, ()):
        place(root)
    # A domain caught in a cycle of parameters has no root to hang from; it
    # keeps its place at the end, and loopy says what it makes of it.
    order.extend(k for k in range(len(domains)) if k not in order)
    return [domains[k] for k in order]


def _restore_narrower_domains(
    insns: Sequence[Any],
    own_domains: Mapping[str, Sequence[isl.Set]],
    merged: Sequence[isl.Set],
) -> list[Any]:
    """Predicate every instruction whose domain :func:`_merge_domains` widened.

    loopy gives an iname one domain, so two statements over the same iname with
    different bounds have to share the union of the two. Sharing it silently is
    wrong: a statement written over ``0 <= i < 2`` would then execute over the
    four points of ``0 <= i < 4``, writing cells the term says it does not
    write. The union is still the loop, and each statement gets back its own
    domain as an instruction predicate, which loopy emits as an ``if`` around
    the statement inside the wider loop.

    The predicate is the *gist* of the statement's domain relative to the merged
    one, so a statement that was not widened gets nothing and the generated code
    is unchanged. A gist isl cannot render as a condition (an existentially
    quantified constraint, say) is refused rather than dropped: dropping it is
    exactly the silent widening this exists to prevent.
    """
    by_names: dict[tuple[str, ...], isl.Set] = {}
    for domain in merged:
        by_names[tuple(domain.get_var_names(isl.dim_type.set))] = domain
    out: list[Any] = []
    for insn in insns:
        own = own_domains.get(insn.id)
        if not own:
            out.append(insn)
            continue
        extra = _narrowing_predicates(insn.id, own, by_names)
        out.append(insn.copy(predicates=insn.predicates | extra) if extra else insn)
    return out


def _narrowing_predicates(
    insn_id: str,
    own: Sequence[isl.Set],
    by_names: Mapping[tuple[str, ...], isl.Set],
) -> frozenset[Any]:
    """The conditions that cut a merged domain back to ``own``."""
    out: list[Any] = []
    for domain in own:
        names = tuple(domain.get_var_names(isl.dim_type.set))
        wider = by_names.get(names)
        if wider is None:  # pragma: no cover - every domain was merged
            continue
        narrow = domain.align_params(wider.get_space())
        wide = wider.align_params(narrow.get_space())
        if narrow.is_equal(wide):
            continue
        if narrow.is_empty():
            # The term says this statement has no instances at all (a guard
            # that is affine and contradictory). It still has to be an
            # instruction, because loopy builds the loop from the merged
            # domain, so it gets a condition nothing satisfies rather than
            # running everywhere the wider domain does.
            out.append(prim.Comparison(0, ">", 0))
            continue
        extra = narrow.gist(wide)
        if extra.plain_is_universe():  # pragma: no cover - is_equal caught this
            continue
        try:
            out.append(set_to_cond_expr(extra))
        except Exception as exc:
            raise LoweringError(
                f"the domain of {insn_id}, {domain}, is narrower than the "
                f"domain its inames {names} end up with, {wider}, and the "
                f"difference cannot be written as a condition on the loop "
                f"variables ({exc}). Running the statement over the wider "
                "domain would execute instances the term does not contain, so "
                "the two loops need different inames."
            ) from exc
    return frozenset(out)


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
    # An array the generated code never mentions cannot be passed: loopy's C
    # target lists only the arrays the body touches in the device function's
    # signature and passes every argument from the host wrapper, so such a
    # parameter shifts every later argument into the wrong register (observed
    # as zeros and a corrupted heap). Refuse the term instead; see
    # docs/loopy-notes.md, note 1.
    untouched = [
        name
        for name, typ in term.params
        if isinstance(typ, ArrType) and name not in used
    ]
    if untouched:
        plural = "s" if len(untouched) > 1 else ""
        raise LoweringError(
            f"the array parameter{plural} {', '.join(untouched)} of {term.name} "
            f"{'are' if plural else 'is'} never read or written by the body, and "
            "loopy's C target cannot pass such an argument: the device function's "
            "signature lists only the arrays the body touches while the host "
            "wrapper passes every argument, so every later argument would land in "
            "the wrong register. Read the array somewhere, or drop the parameter "
            "and take the size it determines from an array that is used "
            "(docs/loopy-notes.md, note 1)"
        )
    provided = {arg.name for arg in builder.extra_args} | set(builder.ragged.values())
    known_inames = {iname for stmt in term.stmts for iname in stmt.inames}
    for stmt in term.stmts:
        for reduction in reductions_of(stmt.expr):
            known_inames.update(reduction.inames)
    # A reduction binder that had to be renamed is an iname of the generated
    # kernel and not a size the caller passes; see _Builder.plan_reductions.
    known_inames |= builder.extra_inames
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

    def shape_of(typ: ArrType, ragged: bool, name: str) -> tuple[Any, ...] | None:
        if ragged:
            return None
        if typ.domain is not None:
            # Only a single domain in a box has a shape; a union, or packed
            # rows, is a flat buffer of a length no loop bound states.
            if isinstance(typ.domain, Union) or builder.storage[name] != "box":
                return None
            shape = tuple(_plain(extent) for extent in typ.domain.box())
        else:
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
                shape=shape_of(typ, ragged, name),
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


def lower(
    term: Term, target: str = "c", layouts: Mapping[str, str] | None = None
) -> Any:
    """Build the loopy kernel for ``term`` on ``target``."""
    return lower_generic(term, target, layouts).kernel


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
