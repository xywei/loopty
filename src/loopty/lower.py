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

An extent of a box can be negative at some sizes, ``n - 1`` at ``n = 0`` for
``Sigma[i: Fin[n], Fin[i]]``, where the domain is empty and its box has no
cells. Such an extent is no shape and no part of a base: a single domain whose
box has one is a flat buffer addressed row-major (:func:`box_is_shape`), and a
piece after one starts at a value argument the executor computes
(:meth:`_Builder.piece_base`), since a term would put it a cell or more away.
"""

from __future__ import annotations

import dataclasses
import functools
import re
from collections.abc import Iterator, Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any

import islpy as isl
import loopy as lp
import numpy as np
import pymbolic.primitives as prim
from loopy.symbolic import Reduction as LoopyReduction
from loopy.symbolic import TypeCast, set_to_cond_expr
from loopy.target.c import CASTBuilder, CFamilyASTBuilder
from loopy.target.c.codegen.expression import (
    CExpressionToCodeMapper,
    ExpressionToCExpressionMapper,
)
from loopy.target.pyopencl import (
    ExpressionToPyOpenCLCExpressionMapper,
    PyOpenCLCASTBuilder,
    PyOpenCLPythonASTBuilder,
)
from pymbolic.mapper import Mapper
from pymbolic.mapper.stringifier import PREC_COMPARISON, PREC_PRODUCT, PREC_SUM

from loopty.contract import array_storage, compiled_storage
from loopty.domain import STORAGES, Union
from loopty.flow import (
    access_relation,
    bounds_dimension,
    counts_families,
    ragged_bound_params,
    statement_accesses,
)
from loopty.idx import linearize
from loopty.operations import (
    CODE_DIGEST,
    OPERATIONS,
    NumpyArithmetic,
    numpy_arithmetic_preambles,
)
from loopty.promotion import Promotion
from loopty.term import (
    COUNT_PARAM,
    COUNT_PARAM_REFLECTED,
    Access,
    ArrType,
    Reduction,
    Stmt,
    Term,
    count_param_names,
    free_name_sorts,
    free_name_sorts_message,
)

__all__ = [
    "COUNT_PARAM",
    "COUNT_PARAM_REFLECTED",
    "GCC_NO_CONTRACTION_PRAGMA",
    "NO_CONTRACTION_FLAG",
    "NO_CONTRACTION_PRAGMAS",
    "POWER_INCLUDES",
    "RESERVED_PREFIX",
    "RESERVED_WORDS",
    "WRAP_FLAG",
    "ExpressionLowerer",
    "InKernelOpenCLTarget",
    "InProcessCTarget",
    "LoweringError",
    "Lowering",
    "SourceCTarget",
    "allows_contraction",
    "array_dtype",
    "count_param_name",
    "count_param_names",
    "is_library_name",
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

#: The C compiler flag that makes signed integer overflow wrap round, as numpy's
#: integer arithmetic does, set on every kernel the C target builds. C leaves
#: the overflow undefined, and GCC at ``-O3`` folds ``x[i] + 1 > x[i]`` to true
#: where numpy wraps ``2**63 - 1`` round to the smallest ``int64`` and finds it
#: false; ``Nat`` and ``Int`` hold the whole ``int64`` range since #101. GCC and
#: clang both take it; OpenCL C has no such build option (note 20).
WRAP_FLAG = "-fwrapv"

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

#: The headers a power needs on the C targets (:func:`_power_preambles`).
POWER_INCLUDES = "#include <stdint.h>\n#include <math.h>"

#: Where they sort among loopy's preambles: after the pragma above, and before
#: loopy's definition of an integer power (``07_``), which needs ``int32_t``.
_POWER_TAG = "06_loopty_power"

#: The header a power of a complex base needs there too, for ``double complex``
#: in the signature of loopy's integer power; only a term with complex values
#: gets it, since it defines ``I``.
_COMPLEX_INCLUDE = "#include <complex.h>"


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

    ``Real`` is double precision, ``Nat`` and ``Int`` are 64-bit, as numpy's
    integers are (#101), ``Bool`` is a byte, and an index type such as
    ``Fin[m]``, which is the element type of a column-index array, is 32-bit,
    which is what an index into an array is on every target loopy generates
    for. A numpy dtype or scalar type is itself, and Python's ``float`` and
    ``complex`` are double precision, which is how
    :func:`loopty.contract.native_storage` has them stored natively too. The
    rule is :func:`loopty.contract.compiled_storage`, which the contract reads
    too: a value of an integral sort outside the range it is stored in is
    refused there (:func:`loopty.contract.integral_range`).
    """
    dtype = compiled_storage(sort)
    if dtype is None:
        raise LoweringError(f"no numpy dtype for {sort!r}")
    return dtype


def array_dtype(term: Term, name: str, sort: Any) -> np.dtype:
    """The numpy dtype the array ``name`` of ``term`` is stored in.

    :func:`numpy_dtype` of its element sort, except for an integral array a
    checked point of a program reads, which is 64 bits wide
    (:func:`loopty.contract.array_storage`, #128).
    """
    dtype = array_storage(term, name, sort)
    if dtype is None:
        raise LoweringError(f"no numpy dtype for {sort!r}")
    return dtype


class _HostCodeWithoutConditionals(CFamilyASTBuilder):
    """The host code of :class:`InProcessCTarget`, which holds no ``if``."""

    @property
    def can_implement_conditionals(self) -> bool:
        return False


class _CExpressions(NumpyArithmetic, ExpressionToCExpressionMapper):
    """loopy's C expressions, with ``//``, ``%``, ``<<`` and ``>>`` numpy's."""


#: The operations C binds more loosely than a comparison, or than ``<`` where
#: the comparison is ``==``, and Python more tightly or as tightly.
_LOOSER_IN_C = (prim.BitwiseAnd, prim.BitwiseXor, prim.BitwiseOr, prim.Comparison)


class _CText(CExpressionToCodeMapper):
    """loopy's printer of C expressions, with C's precedence and the term's nesting.

    loopy prints by pymbolic's precedences, which are Python's: there ``&``,
    ``^`` and ``|`` bind more tightly than a comparison, and every comparison
    as tightly as any other. C binds a comparison more tightly than ``&``,
    ``^`` and ``|``, and ``<`` more tightly than ``==``. So ``(k[i] ^ 1) == 0``
    was printed ``k[i] ^ 1 == 0``, which C reads as ``k[i] ^ (1 == 0)``, and
    ``k[i] < (k[i] ^ 1)`` as ``k[i] < k[i] ^ 1``, ``(k[i] < k[i]) ^ 1``, true
    at every ``k``. An operand of a comparison that is one of these is
    bracketed (note 20). A sum or a product nested after the first operand of
    another is bracketed too, so that C adds and multiplies in the order the
    term does (note 23).
    """

    def map_comparison(self, expr: Any, enclosing_prec: int) -> str:
        left, right = (
            self.rec_with_force_parens_around(
                operand, PREC_COMPARISON, force_parens_around=_LOOSER_IN_C
            )
            for operand in (expr.left, expr.right)
        )
        return self.parenthesize_if_needed(
            f"{left} {expr.operator} {right}", enclosing_prec, PREC_COMPARISON
        )

    def map_sum(self, expr: Any, enclosing_prec: int) -> str:
        """A sum, with a sum after its first operand in brackets.

        pymbolic prints a sum of sums flat, since addition is associative, and
        C adds from the left: ``x[i] + (y[i] + z[i])`` was ``x[i] + y[i] +
        z[i]``, rounded otherwise than numpy rounds it, and ``1.0 + (k[i] +
        j[i])`` was added in double, where numpy adds the integers first and
        wraps round. A sum the term nests to the left prints as before.
        """
        first, *rest = expr.children
        parts = [self.rec(first, PREC_SUM)] + [
            self.rec_with_force_parens_around(
                child, PREC_SUM, force_parens_around=(prim.Sum,)
            )
            for child in rest
        ]
        return self.parenthesize_if_needed(" + ".join(parts), enclosing_prec, PREC_SUM)

    def map_product(self, expr: Any, enclosing_prec: int) -> str:
        """A product, with a product after its first operand in brackets.

        As :meth:`map_sum`: ``1.0 * (k[i] * j[i])`` was ``1.0 * k[i] * j[i]``,
        which C multiplies in double from the left, where numpy multiplies the
        integers first. A quotient, a floor division or a remainder is in
        brackets anywhere, as loopy puts it.
        """
        around = (prim.Quotient, prim.FloorDiv, prim.Remainder)
        first, *rest = expr.children
        parts = [
            self.rec_with_force_parens_around(
                first, PREC_PRODUCT, force_parens_around=around
            )
        ] + [
            self.rec_with_force_parens_around(
                child, PREC_PRODUCT, force_parens_around=(*around, prim.Product)
            )
            for child in rest
        ]
        # Spaces keep ``* *z`` from reading as a dereference, as loopy's do.
        return self.parenthesize_if_needed(
            " * ".join(parts), enclosing_prec, PREC_PRODUCT
        )


class _CCode(CASTBuilder):
    """The device code of loopty's C targets: loopy's, with numpy's arithmetic.

    ``//``, ``%``, ``<<`` and ``>>`` are calls of functions defined as numpy
    defines them (:mod:`loopty.operations`), whose definitions are preambles,
    and expressions are printed with C's precedence (:class:`_CText`).
    """

    def preamble_generators(self) -> Any:
        return [*super().preamble_generators(), numpy_arithmetic_preambles]

    def get_expression_to_c_expression_mapper(self, codegen_state: Any) -> Any:
        return _CExpressions(codegen_state)

    def get_c_expression_to_code_mapper(self) -> Any:
        return _CText()


class InProcessCTarget(lp.ExecutableCTarget):
    """``lp.ExecutableCTarget``, with every condition in the code that runs.

    loopy hoists a condition shared by every instruction of a kernel as far out
    as the iname it names allows, and one that names no iname, such as a
    ``when(flag)`` or ``when(a > 0.5)`` guarding the whole body, goes out of the
    device function into the host code around its call. ``lp.ExecutableCTarget``
    generates that host code and never runs it: its executor calls the device
    function directly. So the guard was gone and the body ran unconditionally
    (#90). With host code that cannot hold a condition, loopy hoists the guard
    no further than the device function's body, and it is emitted there.
    Note 18 in ``docs/loopy-notes.md`` has the details.

    Its device code writes ``//``, ``%``, ``<<`` and ``>>`` as numpy computes
    them (:class:`_CCode`, note 20), and the definitions are hashed in, with
    the plan and the lowering, so that loopy's cache keeps no code of other
    ones (:data:`loopty.operations.CODE_DIGEST`).
    """

    hash_fields = (*lp.ExecutableCTarget.hash_fields, "arithmetic")
    arithmetic = CODE_DIGEST

    def get_host_ast_builder(self) -> Any:
        return _HostCodeWithoutConditionals(self)

    def get_device_ast_builder(self) -> Any:
        return _CCode(self)


class SourceCTarget(lp.CTarget):
    """``lp.CTarget``, whose code is only printed, with numpy's arithmetic.

    The source ``loopty run --emit-code`` prints for target ``c-source``: the
    device code of :class:`InProcessCTarget`, with no host code.
    """

    hash_fields = (*lp.CTarget.hash_fields, "arithmetic")
    arithmetic = CODE_DIGEST

    def get_device_ast_builder(self) -> Any:
        return _CCode(self)


class _LaunchWithoutConditionals(PyOpenCLPythonASTBuilder):
    """The host code of :class:`InKernelOpenCLTarget`, which holds no ``if``."""

    @property
    def can_implement_conditionals(self) -> bool:
        return False


class _OpenCLExpressions(NumpyArithmetic, ExpressionToPyOpenCLCExpressionMapper):
    """loopy's OpenCL C expressions, with numpy's arithmetic."""


class _OpenCLCode(PyOpenCLCASTBuilder):
    """The kernel code of :class:`InKernelOpenCLTarget`; see :class:`_CCode`."""

    def preamble_generators(self) -> Any:
        return [*super().preamble_generators(), numpy_arithmetic_preambles]

    def get_expression_to_c_expression_mapper(self, codegen_state: Any) -> Any:
        return _OpenCLExpressions(codegen_state)

    def get_c_expression_to_code_mapper(self) -> Any:
        return _CText()


class InKernelOpenCLTarget(lp.PyOpenCLTarget):
    """``lp.PyOpenCLTarget``, with every condition in the kernel.

    The PyOpenCL target's host code is Python that runs, and a condition
    hoisted into it, as on the C target (#90), wraps the kernel's launch and
    the line that names the launch's event. The host code returns that event
    either way, so a run whose guard is false raised ``UnboundLocalError``
    instead of writing nothing. With host code that cannot hold a condition,
    the guard is emitted in the kernel, as it is for ``lp.OpenCLTarget``,
    which generates no host code. Building one imports pyopencl, as
    ``lp.PyOpenCLTarget`` does; defining the class does not. Its kernel
    writes numpy's arithmetic, as :class:`InProcessCTarget` does.
    """

    hash_fields = (*lp.PyOpenCLTarget.hash_fields, "arithmetic")
    arithmetic = CODE_DIGEST

    def get_host_ast_builder(self) -> Any:
        return _LaunchWithoutConditionals(self)

    def get_device_ast_builder(self) -> Any:
        return _OpenCLCode(self)


def target_for(target: str = "c") -> Any:
    """The loopy target named by ``target``.

    ``"c"`` is :class:`InProcessCTarget`, ``lp.ExecutableCTarget`` with every
    condition in the device function, which compiles with the system toolchain
    and runs in process; it is the only target a laptop or CI ever uses.
    ``"c-source"`` is :class:`SourceCTarget`, whose code is only printed.
    ``"opencl"`` is :class:`InKernelOpenCLTarget`, ``lp.PyOpenCLTarget`` with
    every condition in the kernel, and pyopencl is imported when it is built,
    here and nowhere else, inside the branch, so that importing loopty on a
    machine without a device costs nothing and can never fail. All three
    write ``//``, ``%``, ``<<`` and ``>>`` as numpy computes them
    (:mod:`loopty.operations`).
    """
    if target in ("c", None):
        return InProcessCTarget()
    if target == "c-source":
        return SourceCTarget()
    if target == "opencl":
        return InKernelOpenCLTarget()
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

    Four jobs in one walk. lanky's subclasses are replaced by pymbolic's, so
    that loopy's structural comparisons work (lanky's ``==`` builds a
    proposition). :class:`~loopty.term.Access` and bare subscripts are turned
    into flat storage accesses, which is where a ragged layout's ``off[r] + j``
    enters. :class:`~loopty.term.Reduction` becomes ``lp.Reduction`` over its
    inames, and the reduction's domain is collected on the side so the caller can
    add it to the kernel. An operation whose operands C would compute in
    another type than numpy does natively has them converted
    (:mod:`loopty.promotion`): ``k[i] / 2`` of an integer ``k`` is
    ``(double) (k[i]) / 2``, ``0.1`` beside a ``float32`` is ``0.1f``, and
    ``col[i] * col[i]`` of a ``Fin[m]`` array is computed in 64 bits, but in a
    subscript: an index is index arithmetic, which loopy computes in 32 bits
    and reads into isl, whose reader raises on a cast, and which loopy
    simplifies, dropping a product with a 64-bit ``1``, so an integer is not
    widened inside one, a limit (#129). A guard that reads no array is a
    condition on the loops, which loopy reads into isl too (note 20 in
    ``docs/loopy-notes.md``), so nothing in one is cast at all: a literal is
    still written in numpy's dtype, an operand is converted by a product with
    ``1.0`` or a 64-bit ``1``, and loopy computes a quotient of integers in
    double in a comparison of its own accord (:meth:`condition`). An
    operation C leaves undefined where numpy does not, ``//``, ``%``, ``<<``
    and ``>>``, is rebuilt as it is, and loopty's targets write it as numpy
    computes it (:mod:`loopty.operations`, :func:`target_for`).
    """

    def __init__(self, lowering: _Builder | _NullBuilder) -> None:
        super().__init__()
        self.lowering = lowering
        #: How many subscripts deep the walk is.
        self._subscripts = 0
        #: The guard the walk is in, when it reads no array; see
        #: :meth:`condition`.
        self._on_loops: Any = None

    def condition(self, guard: Any) -> Any:
        """Lower a statement's guard.

        A guard that reads no array names loop variables, sizes and scalars
        alone, and loopy reads such a predicate as an isl set
        (``loopy.symbolic.condition_to_set``), whose evaluator raises on a
        cast instead of declining it: ``when(i ** 0.5 > 1.5)`` failed in
        loopy's bounds check. loopy reads a guard on scalars so too, since it
        counts each scalar argument an instruction reads as a parameter
        (``loopy.kernel.instruction.get_insn_domain``). So no operand in one
        is cast. A literal is written in the dtype, and an operand numpy
        computes in double is multiplied by ``1.0``: C computes the product in
        double exactly as it would the cast, and the reader reads the ``1.0``
        as the ``1`` it is. ``s * a`` of an ``Int`` ``s`` and a ``float32``
        ``a`` is computed in double so, as numpy computes it, where C computed
        it in single precision when the cast was left out. An integer is
        widened so too, by a product with a 64-bit ``1``: ``when(i * i <
        m)`` is ``1l * i * i < m``, which wrapped round at ``i = 46341`` when
        its integers were left in 32 bits as a subscript's are. An operand
        numpy rounds to single precision, ``(i + 1) ** -1`` beside a
        ``float32`` scalar ``a``, has no such product, and the guard is
        refused (:meth:`_convert`).
        """
        reads = any(isinstance(node, Access | prim.Subscript) for node in walk(guard))
        if reads:
            return self.rec(guard)
        self._on_loops = guard
        try:
            return self.rec(guard)
        finally:
            self._on_loops = None

    @property
    def in_subscript(self) -> bool:
        """Whether the expression being lowered is index arithmetic."""
        return self._subscripts > 0

    def _indices(self, indices: Sequence[Any]) -> tuple[Any, ...]:
        """Lower the indices of a subscript, as index arithmetic."""
        self._subscripts += 1
        try:
            return tuple(self.rec(index) for index in indices)
        finally:
            self._subscripts -= 1

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
        return self.lowering.access(expr.array, self._indices(expr.indices))

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
            return LoopyReduction(expr.op, inames, self._summed(expr))
        self.lowering.push_renaming(renaming)
        try:
            body = self._summed(expr)
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
        return LoopyReduction("sum", inames, self._summed(expr))

    def _summed(self, expr: Any) -> Any:
        """The body of a sum, converted where numpy adds its terms in more bits.

        loopy accumulates in the type of the body, and numpy adds the terms to
        ``0``, so the count a sum of truth values is, or a sum of indices, is
        64 bits wide natively (:meth:`loopty.promotion.Promotion._summed`).
        """
        body = self.rec(expr.body)
        promotion = self.lowering.promotion
        steps = () if promotion is None else promotion.steps(expr)
        for step in steps:
            if step.right is not None:
                body = self._convert(body, step.right)
        return body

    def map_constant(self, expr: Any) -> Any:
        """A Python ``float`` becomes ``np.float64``, a ``complex`` ``np.complex128``.

        loopy gives a bare Python number the type of the place it stands in,
        and does not say so. Its C code generator writes one in the type
        context of the expression around it, which on the right-hand side of
        an assignment is the assignee's: ``c[i] = u[i] * 0.5`` into an integer
        ``c`` came out as ``c[i] = (int32_t) (u[i] * 0)``. Its type inference
        takes a float that single precision holds for a ``float32``, so
        ``k[i] * 0.5`` of an integer ``k`` was computed in single precision
        while numpy computes it in double. A numpy scalar is typed explicitly,
        and loopy writes it as it is. Double precision is what numpy gives a
        Python float beside an integer or a double, which is what the native
        run computes with; see note 17 in ``docs/loopy-notes.md``. Beside a
        ``float32`` numpy gives it single precision, and the operation it
        stands in writes it so (:meth:`_operation`, note 19).

        Integers and booleans are left alone. A numpy scalar already says its
        type, and ``np.float64`` is a subclass of ``float``, so it is asked
        about first. An integer past 64 bits, which loopy types as neither
        ``int32`` nor ``int64`` ("integer constant too large"), is refused
        here; an operation writes one in numpy's type where that holds it
        (:meth:`_operation`, #140), and the trace refuses it elsewhere.
        """
        if isinstance(expr, int) and _untyped(expr):
            raise LoweringError(_untyped_message(expr))
        if isinstance(expr, np.generic | bool | int):
            return expr
        if isinstance(expr, float):
            return np.float64(expr)
        if isinstance(expr, complex):
            return np.complex128(expr)
        return expr

    def map_variable(self, expr: Any) -> prim.Variable:
        return prim.Variable(self.lowering.rename(expr.name))

    def map_subscript(self, expr: Any) -> prim.Expression:
        if not isinstance(expr.aggregate, prim.Variable):
            raise LoweringError(f"cannot lower a subscript of {expr.aggregate!r}")
        index = expr.index if isinstance(expr.index, tuple) else (expr.index,)
        return self.lowering.access(expr.aggregate.name, self._indices(index))

    def _operation(
        self, expr: Any, operands: Sequence[Any], build: Any
    ) -> prim.Expression:
        """An operation, with its operands converted where numpy's type is not C's.

        ``build`` makes the node from a list of lowered operands. The plan is
        :meth:`loopty.promotion.Promotion.steps`, one step per operand after
        the first, evaluated from the left as numpy and C both evaluate a sum
        or a product of several: a step that converts its left operand
        converts everything to the left of it, as one cast around the
        operands before, and a step that converts its result converts
        everything up to its operand (``(int8_t) (a[i] * b[i])``, #122). An
        operation the plan leaves alone is rebuilt as it was, so a kernel
        whose arithmetic C and numpy type alike lowers to the code it always
        did. See note 19 in ``docs/loopy-notes.md``.

        Inside a subscript an integer is not widened or narrowed (see the
        class): a step converting to an integer dtype is left out there. In a
        guard on the loops no operand is cast at all (:meth:`_convert`). An
        integer literal past 64 bits is written in the dtype its step gives
        it, and refused where none does (#140).
        """
        promotion = self.lowering.promotion
        # The plan of the whole operation first: a sum plans the negations
        # it adds, ``a - b``, before they are lowered (Promotion._subtracted).
        steps = () if promotion is None else promotion.steps(expr)
        lowered = [
            operand if _untyped(operand) else self.rec(operand)
            for operand in operands
        ]
        if self.in_subscript:
            steps = tuple(_without_widening(step) for step in steps)
        if not any(step.converts for step in steps):
            _refuse_untyped(lowered)
            return build(lowered)
        head = [lowered[0]]
        for operand, step in zip(lowered[1:], steps, strict=True):
            if step.left is not None:
                before = head[0] if len(head) == 1 else build(head)
                head = [self._convert(before, step.left)]
            if step.right is not None:
                operand = self._convert(operand, step.right)
            head.append(operand)
            if step.result is not None:
                head = [self._convert(build(head), step.result, narrowing=True)]
        _refuse_untyped(head)
        return head[0] if len(head) == 1 else build(head)

    def _convert(self, operand: Any, dtype: np.dtype, narrowing: bool = False) -> Any:
        """``operand`` computed in ``dtype``, as a guard on the loops allows.

        A literal is written in the dtype, and anything else is cast, but in a
        guard on the loops, whose cast loopy's isl reader raises on (see
        :meth:`condition`): there a double is had by a product with ``1.0``,
        and an integer by a product with ``1`` in its dtype, which C computes
        from the left in that type exactly as it would the cast, and which the
        reader takes as the number it is, or declines with any product of two
        variables. A conversion to another type has no such product, and the
        guard is refused: numpy rounds ``(i + 1) ** -1`` to single precision
        beside a ``float32`` scalar, and C would keep it in double. Nor has a
        conversion of a result back into the integer type numpy computes it
        in (``narrowing``, #122), which a product cannot narrow. The ``1.0``
        comes first, beside the operand it converts: ``s * (1.0 * a)``, which
        loopty's printer brackets (:class:`_CText`), is computed in double,
        where ``s * a * 1.0`` would multiply ``s * a`` in single precision,
        and ``1l * i * i`` is a product of longs.
        """
        if self._on_loops is None or (_is_literal(operand) and not narrowing):
            return _converted(operand, dtype)
        if not narrowing and (dtype == np.float64 or dtype.kind in "iu"):
            return prim.Product((dtype.type(1), operand))
        declare = "Real" if dtype.kind in "fc" else "Int"
        raise LoweringError(
            f"the guard {self._on_loops} reads no array, so loopy reads it into "
            f"isl, whose reader raises on a cast, and numpy computes {operand} "
            f"in it in {dtype}, which C computes in another type without one. "
            f"Declare the {dtype} scalars the guard names {declare}, so that "
            "both runs compute it in "
            f"{'double' if declare == 'Real' else '64 bits'}"
        )

    def map_sum(self, expr: Any) -> prim.Expression:
        return self._operation(expr, expr.children, lambda ops: prim.Sum(tuple(ops)))

    def map_product(self, expr: Any) -> prim.Expression:
        return self._operation(
            expr, expr.children, lambda ops: prim.Product(tuple(ops))
        )

    def map_quotient(self, expr: Any) -> prim.Expression:
        return self._operation(
            expr, (expr.numerator, expr.denominator), lambda ops: prim.Quotient(*ops)
        )

    def map_floor_div(self, expr: Any) -> prim.Expression:
        return self._operation(
            expr, (expr.numerator, expr.denominator), lambda ops: prim.FloorDiv(*ops)
        )

    def map_remainder(self, expr: Any) -> prim.Expression:
        return self._operation(
            expr, (expr.numerator, expr.denominator), lambda ops: prim.Remainder(*ops)
        )

    def map_power(self, expr: Any) -> prim.Expression:
        return self._operation(
            expr, (expr.base, expr.exponent), lambda ops: prim.Power(*ops)
        )

    def map_bitwise_xor(self, expr: Any) -> prim.Expression:
        """``a ^ b``, which C computes as numpy does in the type numpy does (#107)."""
        return self._operation(
            expr, expr.children, lambda ops: prim.BitwiseXor(tuple(ops))
        )

    def map_left_shift(self, expr: Any) -> prim.Expression:
        return self._operation(
            expr, (expr.shiftee, expr.shift), lambda ops: prim.LeftShift(*ops)
        )

    def map_right_shift(self, expr: Any) -> prim.Expression:
        return self._operation(
            expr, (expr.shiftee, expr.shift), lambda ops: prim.RightShift(*ops)
        )

    def map_call(self, expr: Any) -> prim.Expression:
        function = expr.function
        if (
            isinstance(function, prim.Variable)
            and function.name == "abs"
            and len(expr.parameters) == 1
        ):
            return self._absolute(expr, expr.parameters[0])
        return prim.Call(
            self.rec(expr.function), tuple(self.rec(p) for p in expr.parameters)
        )

    def _absolute(self, expr: Any, operand: Any) -> prim.Expression:
        """``abs(operand)``: C's ``abs`` of a number, and numpy's of an integer.

        loopy resolves ``abs`` as the C library's, ``fabs`` or ``cabs``, which
        it refuses for an integer (``abs does not support type float32``,
        #123). numpy's ``abs`` of an integer is ``-k`` where ``k`` is negative
        and ``k`` elsewhere, the smallest value of its type included, which
        it returns as it is, as ``-k`` wraps round to it. A truth value or an
        unsigned integer, of which ``abs`` is itself, is written as itself.
        An integer is written without a branch, ``(k ^ s) - s``, where ``s =
        k >> 63`` (``>> 31`` in C's ``int``) is ``-1`` for a negative ``k``
        and ``0`` otherwise, which C computes so with ``-fwrapv``. Not as ``k
        < 0 ? -1 * k : k``: GCC reads that as its ``abs``, which it takes to
        be non-negative even under ``-fwrapv``, and folded ``1 >= abs(k)`` to
        false at the smallest ``int32``, where numpy's is true, at ``-O0``
        too; and loopy realizes a sum in a branch of an ``If`` under the
        ``If``'s condition, so ``acc < 0 ? -acc : acc`` added a term only
        where the sum so far was negative (note 23). Of an integer narrower
        than ``int`` the result is converted back into its type, which C's
        ``int`` leaves (:meth:`loopty.promotion.Promotion._absolute`). The
        operand is written three times; a sum in it is computed for each.
        """
        promotion = self.lowering.promotion
        native, compiled = (
            (None, None) if promotion is None else promotion.types(operand)
        )
        lowered = self.rec(operand)
        if compiled is None or compiled.kind not in "biu":
            return prim.Call(prim.Variable("abs"), (lowered,))
        truths = bool(native) and all(
            isinstance(sample, bool | np.bool_) for sample in native
        )
        if compiled.kind in "bu" or truths:
            return lowered
        bits = 8 * max(compiled.itemsize, 4) - 1
        sign = prim.RightShift(lowered, bits)
        value: Any = prim.Sum(
            (prim.BitwiseXor((lowered, sign)), prim.Product((-1, sign)))
        )
        assert promotion is not None
        for step in promotion.steps(expr):
            if step.result is not None and not self.in_subscript:
                value = self._convert(value, step.result, narrowing=True)
        return value

    def map_comparison(self, expr: Any) -> prim.Expression:
        """A comparison, its sign compared first where C would lose it (#122).

        Where C compares two integers in an unsigned type and one of them may
        be negative (:attr:`loopty.promotion.Step.sign`), ``u[i] < k[i]`` of a
        ``uint32`` ``u`` and an ``int32`` ``k`` is ``k[i] >= 0 && u[i] <
        k[i]``, and ``u[i] != -1`` is ``-1 < 0 || u[i] != -1``: where the
        signed operand is negative the comparison is decided by its sign, as
        numpy decides it, and C compares the two only where it is not.

        An integer literal past 64 bits that a ``uint64`` does not hold either
        is compared with an integer as a double (``_compared`` in
        :mod:`loopty.promotion`), and is written as ``2.0 ** 65`` with its
        sign (:func:`_beyond_integers`): every integer of 64 bits is on the
        same side of that as of the literal, which numpy compares exactly,
        where the literal's own double rounds ``2**64`` and ``-2**63 - 1``
        onto values a ``uint64`` and an ``int64`` reach: ``u[i] < 2**64`` was
        false at ``2**64 - 1`` (#140).
        """
        promotion = self.lowering.promotion
        steps = () if promotion is None else promotion.steps(expr)
        sign = steps[0].sign if steps else None
        operands = (expr.left, expr.right)
        if promotion is not None:
            operands = tuple(
                _beyond_integers(operand, promotion.types(other)[1])
                for operand, other in zip(operands, operands[::-1], strict=True)
            )

        def build(ops: Sequence[Any]) -> prim.Expression:
            compared = prim.Comparison(ops[0], expr.operator, ops[1])
            if sign is None:
                return compared
            signed = ops[sign]
            # Whether the comparison holds where the signed operand is
            # negative and the other, unsigned, is not.
            holds = ("<", "<=", "!=") if sign == 0 else (">", ">=", "!=")
            if expr.operator in holds:
                return prim.LogicalOr((prim.Comparison(signed, "<", 0), compared))
            return prim.LogicalAnd((prim.Comparison(signed, ">=", 0), compared))

        return self._operation(expr, operands, build)

    def map_logical_and(self, expr: Any) -> prim.Expression:
        return prim.LogicalAnd(tuple(self.rec(c) for c in expr.children))

    def map_logical_or(self, expr: Any) -> prim.Expression:
        return prim.LogicalOr(tuple(self.rec(c) for c in expr.children))

    def map_logical_not(self, expr: Any) -> prim.Expression:
        return prim.LogicalNot(self.rec(expr.child))

    def map_lanky_abs(self, expr: Any) -> prim.Expression:
        """lanky's ``Abs``, as a call of ``abs`` is (:meth:`_absolute`)."""
        return self._absolute(expr, expr.operand)

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


def _without_widening(step: Any) -> Any:
    """``step`` without a conversion to an integer dtype; see ExpressionLowerer.

    Of an operand or of the result: a subscript's arithmetic is left in the
    type loopy computes it in, a limit (#129).
    """

    def kept(dtype: np.dtype | None) -> np.dtype | None:
        return dtype if dtype is None or dtype.kind not in "biu" else None

    left, right, result = kept(step.left), kept(step.right), kept(step.result)
    if left is step.left and right is step.right and result is step.result:
        return step
    return dataclasses.replace(step, left=left, right=right, result=result)


def _untyped(expr: Any) -> bool:
    """Whether ``expr`` is a Python int loopy cannot type, past 64 bits (#140)."""
    return (
        isinstance(expr, int)
        and not isinstance(expr, bool)
        and not -(2**63) <= expr < 2**63
    )


def _beyond_integers(operand: Any, other: np.dtype | None) -> Any:
    """``operand`` as :meth:`ExpressionLowerer.map_comparison` compares it.

    An integer literal that neither an ``int64`` nor a ``uint64`` holds,
    beside an integer, is ``2.0 ** 65`` with its sign, a double past every
    integer of 64 bits; any other operand is itself.
    """
    if (
        not _untyped(operand)
        or other is None
        or other.kind not in "biu"
        or 0 <= operand < 2**64
    ):
        return operand
    return np.float64(2.0**65 if operand > 0 else -(2.0**65))


def _untyped_message(value: int, where: str = "") -> str:
    """Why an integer literal past 64 bits is refused, and what to write."""
    exponent = abs(value).bit_length() - 1
    if abs(value) == 2**exponent:
        spelled = f"{'-' if value < 0 else ''}2.0 ** {exponent}"
    else:
        spelled = repr(float(value))
    at = f" at {where}" if where else ""
    return (
        f"the integer {value}{at} is past 64 bits, and the compiled kernel has "
        "no integer type that holds it: loopy types an integer literal as "
        "int32 or int64. Such a literal is written in numpy's type where numpy "
        "computes the operation it stands in in a real or a uint64 that holds "
        f"it, x[i] * 2**70, and nowhere else; write it as a real, {spelled}, "
        "which both runs compute with alike"
    )


def _refuse_untyped(operands: Sequence[Any]) -> None:
    """Refuse an integer literal past 64 bits that no step wrote in a type."""
    for operand in operands:
        if _untyped(operand):
            raise LoweringError(_untyped_message(operand))


def _is_literal(expr: Any) -> bool:
    """Whether ``expr`` is a number, which :func:`_converted` writes in a dtype."""
    return isinstance(expr, bool | int | float | complex | np.number | np.bool_)


def _converted(expr: Any, dtype: np.dtype) -> Any:
    """``expr`` computed in ``dtype``: a literal written in it, anything else cast.

    A literal is a constant of the dtype, which loopy writes as it is
    (``0.1f`` for a ``float32``), so a literal beside a ``float32`` operand
    is what numpy makes of it there; a cast would round the double instead,
    which is the same value and more to read.
    """
    if _is_literal(expr):
        return dtype.type(expr)
    return TypeCast(dtype, expr)


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
    compared bit for bit, see :func:`allows_contraction`. ``temporaries``
    names the arrays declared as loopy temporaries, a program's own arrays
    (:attr:`loopty.term.Term.temporaries`), which are nobody's argument.

    ``storage`` says, for every array over a polyhedral domain, which layout
    it is stored in, ``"box"`` or ``"packed"`` (:mod:`loopty.domain`), and
    ``tables`` names the argument holding the table of row starts of each
    packed one, which the executor computes and passes. ``bases`` maps a
    value argument to the ``(array, piece)`` whose start it holds, for a piece
    of a union whose start is not a term of the sizes (see
    :meth:`_Builder.piece_base`); the executor computes that too, in the
    array's layout.

    ``checks`` maps the flag of each checked point of a program to what a
    failure raises (:attr:`loopty.term.Term.checks`). A flag is a one-cell
    argument the kernel writes, so it is among ``outputs``; the executor
    passes it zeroed, reads it after the run, and keeps it out of the
    results, which :attr:`results` lists.
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
    temporaries: tuple[str, ...] = ()
    storage: dict[str, str] = field(default_factory=dict)
    tables: dict[str, str] = field(default_factory=dict)
    bases: dict[str, tuple[str, int]] = field(default_factory=dict)
    checks: dict[str, str] = field(default_factory=dict)

    @property
    def name(self) -> str:
        """The kernel's name."""
        return self.term.name

    @property
    def results(self) -> tuple[str, ...]:
        """The outputs that are parameters of the term: every one but the flags."""
        return tuple(name for name in self.outputs if name not in self.checks)


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
#: plus the type names and keywords OpenCL C adds (its vector types, ``pipe``
#: and the image types, which clang reads as keywords there, #124), because the
#: same term is lowered for both targets and a name that compiles on one has to
#: compile on the other.
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
    read_write half2 half3 half4 half8 half16 pipe image1d_t image1d_array_t
    image1d_buffer_t image2d_t image2d_array_t image2d_depth_t
    image2d_array_depth_t image2d_msaa_t image2d_array_msaa_t
    image2d_msaa_depth_t image2d_array_msaa_depth_t image3d_t
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


#: The functions C99's ``math.h`` declares, each also with an ``f`` and an
#: ``l`` suffix, and ``complex.h``'s, which the generated code includes for a
#: power (:func:`_power_preambles`) and for complex values, and for
#: :mod:`loopty.operations`.
_C_LIBRARY_FUNCTIONS = frozenset(
    form
    for function in """
    acos asin atan atan2 cos sin tan acosh asinh atanh cosh sinh tanh exp exp2
    expm1 frexp ilogb ldexp log log10 log1p log2 logb modf scalbn scalbln cbrt
    fabs hypot pow sqrt erf erfc lgamma tgamma ceil floor nearbyint rint lrint
    llrint round lround llround trunc fmod remainder remquo copysign nan
    nextafter nexttoward fdim fmax fmin fma
    cabs cacos cacosh carg casin casinh catan catanh ccos ccosh cexp cimag clog
    conj cpow cproj creal csin csinh csqrt ctan ctanh
""".split()
    for form in (function, f"{function}f", f"{function}l")
)

#: What those headers, and ``stdint.h``, define as macros or types: a
#: function of that name is expanded, or declared over a type. A ``uint8_t``
#: and every other exact, least and fast width is matched by
#: :data:`_STDINT_TYPE`.
_C_LIBRARY_MACROS = frozenset(
    """
    fpclassify isfinite isinf isnan isnormal signbit isgreater isgreaterequal
    isless islessequal islessgreater isunordered NAN INFINITY HUGE_VAL
    HUGE_VALF HUGE_VALL FP_INFINITE FP_NAN FP_NORMAL FP_SUBNORMAL FP_ZERO
    FP_FAST_FMA FP_FAST_FMAF FP_FAST_FMAL FP_ILOGB0 FP_ILOGBNAN MATH_ERRNO
    MATH_ERREXCEPT math_errhandling float_t double_t I CMPLX CMPLXF CMPLXL
    intmax_t uintmax_t intptr_t uintptr_t
""".split()
)

_STDINT_TYPE = re.compile(
    r"u?int(_least|_fast)?(8|16|32|64)_t$"
    r"|U?INT\w*_(MIN|MAX|C)$|(SIZE|PTRDIFF|SIG_ATOMIC|WCHAR|WINT)_(MIN|MAX)$"
)

#: The macros of :data:`_C_LIBRARY_MACROS` that a name alone expands, with the
#: header that defines each: a variable of such a name is the macro's body in
#: the generated code, so ``I`` declared as a parameter of a kernel with
#: complex values was ``complex.h``'s imaginary unit, and gcc read
#: ``double complex const *I`` as ``... *(__extension__ 1.0iF)`` (#124). A
#: macro that expands only before a ``(`` (``isnan``) and a function do not,
#: and a variable may take their names. The types are declared over as the
#: ``stdint.h`` ones are.
_OBJECT_MACROS = {
    **{
        name: "math.h"
        for name in """
        NAN INFINITY HUGE_VAL HUGE_VALF HUGE_VALL FP_INFINITE FP_NAN FP_NORMAL
        FP_SUBNORMAL FP_ZERO FP_FAST_FMA FP_FAST_FMAF FP_FAST_FMAL FP_ILOGB0
        FP_ILOGBNAN MATH_ERRNO MATH_ERREXCEPT math_errhandling float_t double_t
        """.split()
    },
    "I": "complex.h",
    **{
        name: "stdint.h"
        for name in "intmax_t uintmax_t intptr_t uintptr_t".split()
    },
}

#: What OpenCL C predefines as a macro: ``NULL``, its limits, its constants,
#: the arguments of its fences and image functions, the status of an event,
#: the initializers of its atomics, and a macro for each extension the device
#: has (``cl_khr_fp64``); and what PoCL, which loopy's OpenCL code is run on in
#: tests, defines beside them (``MAX_WORK_DIM``, ``IMG_RO_AQ``). The same term
#: is lowered for both targets, so a name of one of these is refused as a C
#: header's is.
_OPENCL_MACROS = re.compile(
    r"(NULL|MAXFLOAT|CHAR_BIT|[US]?CHAR_MAX|U?(SHRT|INT|LONG)_MAX"
    r"|(S?CHAR|SHRT|INT|LONG)_MIN|FP_FAST_FMA_HALF|ATOMIC_FLAG_INIT"
    r"|MAX_WORK_DIM|IMG_(RO|WO|RW)_AQ)$"
    r"|(FLT|DBL|HALF)_(DIG|MANT_DIG|MAX_10_EXP|MAX_EXP|MIN_10_EXP|MIN_EXP"
    r"|RADIX|MAX|MIN|EPSILON)$"
    r"|M_(E|LOG2E|LOG10E|LN2|LN10|PI|PI_2|PI_4|1_PI|2_PI|2_SQRTPI|SQRT2"
    r"|SQRT1_2)(_F|_H)?$"
    r"|CLK_\w+$|CL_VERSION_\d+_\d+$|CL_(COMPLETE|RUNNING|SUBMITTED|QUEUED)$"
    r"|cl_(khr|ext|intel|amd|nv|arm|img|qcom|apple|APPLE|clang|pocl)_\w+$|POCL_\w+$"
)

#: The types OpenCL C declares, and the constants of its enumerations: a
#: kernel of one of these names redeclares it at file scope on the OpenCL
#: target (``redefinition of 'size_t' as different kind of symbol``, #131). A
#: variable may take one of these names, which hides the type in its block.
_OPENCL_TYPES = re.compile(
    r"(size_t|ptrdiff_t|intptr_t|uintptr_t|event_t|sampler_t|queue_t"
    r"|clk_event_t|reserve_id_t|ndrange_t|kernel_enqueue_flags_t"
    r"|clk_profiling_info|cl_mem_fence_flags)$"
    r"|memory_(order|scope)(_\w+)?$"
)

#: OpenCL C's built-in functions (its specification's section 6.13 and 6.15),
#: and the macros it defines that take arguments, which expand only before a
#: ``(``, as a kernel's declaration has (``ATOMIC_VAR_INIT``, and PoCL's
#: ``kernel_exec``): a kernel of one of these names shares it with the
#: built-in on the OpenCL target (#131). Those C also has are in
#: :data:`_C_LIBRARY_FUNCTIONS`, and the families of many names in
#: :data:`_OPENCL_FAMILIES`.
_OPENCL_FUNCTIONS = frozenset(
    """
    get_work_dim get_global_size get_global_id get_local_size
    get_enqueued_local_size get_local_id get_num_groups get_group_id
    get_global_offset get_global_linear_id get_local_linear_id
    get_sub_group_size get_max_sub_group_size get_num_sub_groups
    get_enqueued_num_sub_groups get_sub_group_id get_sub_group_local_id
    acospi asinpi atanpi atan2pi cospi sinpi tanpi exp10 fract mad maxmag minmag
    pown powr rootn rsqrt sincos lgamma_r
    abs abs_diff add_sat hadd rhadd clamp clz ctz mad_hi mad_sat max min mul_hi
    rotate sub_sat upsample popcount mad24 mul24
    degrees mix radians step smoothstep sign
    cross dot distance length normalize fast_distance fast_length
    fast_normalize
    isequal isnotequal isgreater isgreaterequal isless islessequal
    islessgreater isfinite isinf isnan isnormal isordered isunordered signbit
    any all bitselect select
    barrier mem_fence read_mem_fence write_mem_fence atomic_work_item_fence
    to_global to_local to_private get_fence
    async_work_group_copy async_work_group_strided_copy wait_group_events
    prefetch vec_step shuffle shuffle2 printf
    read_pipe write_pipe reserve_read_pipe reserve_write_pipe commit_read_pipe
    commit_write_pipe is_valid_reserve_id get_pipe_num_packets
    get_pipe_max_packets
    enqueue_kernel enqueue_marker retain_event release_event create_user_event
    is_valid_event set_user_event_status capture_event_profiling_info
    get_default_queue ndrange_1D ndrange_2D ndrange_3D
    get_kernel_work_group_size get_kernel_preferred_work_group_size_multiple
    get_kernel_sub_group_count_for_ndrange
    get_kernel_max_sub_group_size_for_ndrange ATOMIC_VAR_INIT kernel_exec
    """.split()
)

#: Names no kernel may take: OpenCL C refuses a kernel called ``main``
#: ("kernel cannot be called 'main'"), and C has the program's entry point by
#: that name.
_KERNEL_NAMES_TAKEN = frozenset({"main"})

#: The families of OpenCL C built-ins named by a pattern: conversions
#: (``convert_int4_sat_rte``), reinterpretations (``as_float``), vector loads
#: and stores, atomics, work-group and sub-group functions, images, and the
#: ``half_`` and ``native_`` forms of the math functions.
_OPENCL_FAMILIES = re.compile(
    r"(convert|as)_(u?(char|short|int|long)|half|float|double)\d*(_sat)?"
    r"(_rt[ezpn])?$"
    r"|v(load|store)a?(_half)?\d*(_rt[ezpn])?$"
    r"|(atomic|atom)_\w+$|(work_group|sub_group)_\w+$"
    r"|(read|write)_image\w*$|get_image_\w+$"
    r"|(half|native)_(cos|divide|exp|exp2|exp10|log|log2|log10|powr|recip|rsqrt"
    r"|sin|sqrt|tan)$"
)

#: The functions the OpenCL target's code calls on a loop it runs in parallel,
#: whatever the kernel computes: a variable of such a name would shadow them.
_OPENCL_WORK_ITEM = frozenset(
    name for name in _OPENCL_FUNCTIONS if name.startswith("get_") and "_id" in name
) | frozenset({"get_local_size", "get_global_size", "get_num_groups"})

#: The helper functions loopy and loopty define in a preamble:
#: ``loopy_floor_div_pos_b_int32``, ``loopy_pow_int32_int32``,
#: ``loopty_mod_int64``. A name with anything after the types is not one, so
#: the ``_knl`` a clash adds ends the clash.
_HELPER_FUNCTION = re.compile(
    r"loopt?y_(floor_div(_pos_b)?|mod(_pos_b)?|pow|lshift|rshift)"
    r"(_(u?int|float|complex)\d+)+$"
)


@functools.cache
def _known_functions() -> frozenset[str]:
    """Every function name loopy resolves on a target loopty lowers for.

    loopy looks a call up among its own functions (``make_tuple``), the
    target's (``floor``, ``sqrt``, ``conj`` on C; ``dot`` and ``make_float2``
    on OpenCL) and a translation unit's, the kernel itself included, and the
    target's win: a kernel named ``floor`` is looked up as the function. The
    targets are asked for their lists, so a loopy that knows more names is
    followed; :mod:`loopty.operations` adds its own.
    """
    from loopy.library.function import get_loopy_callables
    from loopy.target.pyopencl import get_pyopencl_callables

    names = set(get_loopy_callables())
    for target in (lp.CTarget(), lp.OpenCLTarget()):
        names |= set(target.get_device_ast_builder().known_callables)
    names |= set(get_pyopencl_callables())
    names |= set(OPERATIONS)
    return frozenset(names)


def is_library_name(name: str) -> bool:
    """Whether a function of this name collides with one the generated code sees.

    A function loopy resolves a call by (:func:`_known_functions`), one the C
    headers the generated code includes declare, with its ``f`` and ``l``
    forms, or define as a macro or a type, or a helper loopy or loopty
    defines in a preamble. A kernel of such a name failed inside loopy
    (``KeyError: 'floor'``) or in the C compiler (``conflicting types for
    'cpow'``), #108. So does a built-in function, a macro or a type of OpenCL
    C, on the OpenCL target (``get_global_id``, ``clamp``, ``convert_int``,
    ``NULL``, ``size_t``, #131), whatever target the kernel is lowered for, as
    a keyword of either is, and so does ``main``.
    """
    if name in _known_functions() or name in _C_LIBRARY_FUNCTIONS:
        return True
    if name in _C_LIBRARY_MACROS or name in _OPENCL_FUNCTIONS:
        return True
    if name in _KERNEL_NAMES_TAKEN:
        return True
    return bool(
        _STDINT_TYPE.match(name)
        or _HELPER_FUNCTION.match(name)
        or _OPENCL_FAMILIES.match(name)
        or _OPENCL_MACROS.match(name)
        or _OPENCL_TYPES.match(name)
    )


def is_reserved(name: str) -> bool:
    """Whether generated C or OpenCL C code cannot declare ``name``.

    A word of :data:`RESERVED_WORDS`, or a name that starts the way
    :data:`RESERVED_PREFIX` says C reserves.
    """
    return name in RESERVED_WORDS or RESERVED_PREFIX.match(name) is not None


def _sanitize(name: str) -> str:
    """A loopy-safe identifier: every non-word character becomes an underscore."""
    return re.sub(r"\W", "_", name)


def _refuse_colliding_ids(term: Term) -> None:
    """Refuse two statements whose ids spell one instruction id (#88).

    A statement's instruction is named by its id with every character other
    than a letter, a digit or an underscore written ``_`` (:func:`_sanitize`),
    and loopy refuses two instructions of one id from inside ``lp.make_kernel``
    ("duplicate instruction id"), which names neither statement. A traced
    kernel's ids (``S0``, ``S1``, ...) and a program's (its call labels, unique
    once spelled) cannot collide; a term built by hand can, and
    ``Schedule.affine`` reads an instruction's id as the one statement it
    names.
    """
    seen: dict[str, str] = {}
    for stmt in term.stmts:
        insn_id = _sanitize(stmt.id)
        first = seen.get(insn_id)
        if first is None:
            seen[insn_id] = stmt.id
            continue
        both = (
            f"two statements have the id {stmt.id!r}"
            if first == stmt.id
            else f"statements {first!r} and {stmt.id!r} both spell the "
            f"instruction id {insn_id!r}"
        )
        raise LoweringError(
            f"{both} in {term.name}. A statement is lowered as the instruction "
            "named by its id, with every character other than a letter, a "
            "digit or an underscore written '_', and loopy needs one "
            "instruction per id: give the statements ids that stay apart"
        )


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

    A name a function the generated code sees already has is renamed the same
    way (:func:`is_library_name`): a kernel named ``floor`` was looked up by
    loopy as C's ``floor`` and failed with ``KeyError: 'floor'``, and one
    named ``cpow`` over complex arrays clashed in the C compiler with the
    ``cpow`` that ``complex.h`` declares (#108). So is a name of OpenCL C's
    built-ins (#131), but for one of a family of them that takes in any
    ending (``atomic_add``, ``work_group_reduce_add``), which gets a ``knl_``
    prefix instead: ``knl_atomic_add``.
    """
    base = _sanitize(name)
    if not base or base[0].isdigit():
        base = f"k_{base}"
    elif RESERVED_PREFIX.match(base):
        # ``_Generic`` stays reserved with any suffix, so it gets a prefix.
        base = f"k{base}"
    taken = set(taken)

    def clashes(candidate: str) -> bool:
        return (
            candidate in taken
            or is_reserved(candidate)
            or is_library_name(candidate)
        )

    if not clashes(base):
        return base
    candidate = f"{base}_knl"
    if clashes(candidate):
        # A family of OpenCL C's built-ins takes in any ending (``atomic_add``
        # and ``atomic_add_knl`` are both ``atomic_\w+``), so no suffix leaves
        # it, and appending one ran forever. Every family and pattern is
        # matched from a name's start, which a prefix leaves.
        candidate = f"knl_{base}"
    while clashes(candidate):
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
    compiler exactly as ``for double in x.dom`` does. So is the name of a
    helper loopy or loopty defines in the generated code
    (``loopty_mod_int64``, ``loopy_pow_int64_int32``): a parameter of that
    name shadows the helper in the kernel's body, and a call of it fails to
    compile.

    So is a name the generated code already gives a meaning to (#124,
    :func:`_shadowing`): a macro a header defines, ``I`` with complex values,
    ``NAN`` or ``INT32_MAX``, and a function the kernel calls, ``pow`` with a
    power or ``floor`` beside a call of ``floor``. The message names what the
    code means by each.
    """
    roles: dict[str, list[str]] = {
        "parameters": [name for name, _ in term.params],
        "program-local arrays": [name for name, _ in term.temporaries],
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
        refused = sorted(
            {
                name
                for name in names
                if is_reserved(_sanitize(name))
                or _HELPER_FUNCTION.match(_sanitize(name))
            }
        )
        if refused:
            found.append(f"{role} {', '.join(refused)}")
    if found:
        raise LoweringError(
            f"{term.name} has names the generated code cannot use: "
            f"{'; '.join(found)}. These are reserved words in C or OpenCL C, "
            "start with an underscore and a capital letter or with two "
            "underscores, which C reserves, or name a helper function loopy or "
            "loopty defines in the generated code; rename them in the kernel (a "
            "parameter in its signature, a size in its annotations, a loop or "
            "reduction variable where it is bound), or in the program that "
            "makes a program-local array."
        )
    called = _called_functions(term)
    shadowed = []
    for role, names in roles.items():
        reasons = {
            name: reason
            for name in sorted(set(names))
            if (reason := _shadowing(_sanitize(name), called)) is not None
        }
        if reasons:
            listed = ", ".join(f"{name} ({reason})" for name, reason in reasons.items())
            shadowed.append(f"{role} {listed}")
    if shadowed:
        raise LoweringError(
            f"{term.name} has names the generated code gives another meaning: "
            f"{'; '.join(shadowed)}. A macro is expanded wherever its name "
            "stands, a declaration of the name included, and a variable of a "
            "function's name hides the function from the code the kernel "
            "calls it in, so the C would not compile (#124); rename them in the "
            "kernel (a parameter in its signature, a size in its annotations, "
            "a loop or reduction variable where it is bound), or in the program "
            "that makes a program-local array."
        )


def _shadowing(name: str, called: frozenset[str]) -> str | None:
    """What the generated code means by ``name`` already, if anything.

    A macro that a header the code includes defines and that its name alone
    expands, a type of ``stdint.h``, a macro OpenCL C predefines
    (:data:`_OBJECT_MACROS`, :data:`_STDINT_TYPE`, :data:`_OPENCL_MACROS`), a
    function the kernel's code calls (:func:`_called_functions`), or a
    function OpenCL code calls on a parallel loop. A variable may have the
    name of any other function: a parameter ``exp`` of a kernel that never
    calls ``exp`` is fine.
    """
    header = _OBJECT_MACROS.get(name)
    if header is not None:
        return f"a macro or type {header} defines"
    if _STDINT_TYPE.match(name):
        return "a type or macro stdint.h defines"
    if _OPENCL_MACROS.match(name):
        return "a macro OpenCL C defines"
    if name in called:
        return "a C library function the kernel calls"
    if name in _OPENCL_WORK_ITEM:
        return "a function OpenCL code calls on a parallel loop"
    return None


def _called_functions(term: Term) -> frozenset[str]:
    """The C library functions the code generated for ``term`` may call.

    Each function a statement calls, ``sqrt``, in every form loopy may write
    it in by the type (``sqrtf``, ``sqrtl``, ``csqrt``, ...), ``abs`` as
    ``fabs`` and ``cabs`` too, and the forms of ``pow`` where a statement has
    a power. The helpers of loopy and loopty are refused by their pattern
    (:data:`_HELPER_FUNCTION`) whether called or not.
    """
    names: set[str] = set()

    def forms(function: str) -> set[str]:
        return {
            spelled
            for base in (function, f"c{function}")
            for spelled in (base, f"{base}f", f"{base}l")
        }

    from lanky.terms import Abs

    for stmt in term.stmts:
        sources = (stmt.expr, stmt.guard, tuple(stmt.assignee.indices))
        for node in walk(sources):
            if isinstance(node, prim.Call) and isinstance(node.function, prim.Variable):
                names |= forms(node.function.name)
                if node.function.name == "abs":
                    names |= forms("fabs")
            elif isinstance(node, Abs):
                names |= forms("abs") | forms("fabs")
            elif isinstance(node, prim.Power):
                names |= forms("pow")
    return frozenset(names)


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
        #: Parameters and a program's temporaries alike: an access is lowered
        #: the same way whoever allocates the array.
        self.arr_types: dict[str, ArrType] = term.array_types
        self.scalar_types: dict[str, Any] = {
            name: typ for name, typ in term.params if not isinstance(typ, ArrType)
        }
        self.written = _written_arrays(term)
        self.ragged: dict[str, str] = {}
        #: The layout of every array over a domain, and the argument holding
        #: the table of row starts of each packed one; see Lowering.
        self.storage = _storage_plan(term, layouts)
        self.tables: dict[str, str] = {}
        #: The value arguments holding where a piece of a union starts, where
        #: that is not a term of the sizes; see :meth:`piece_base`.
        self.bases: dict[str, tuple[str, int]] = {}
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
        #: The type numpy computes each operation in, and where the lowered
        #: code has to convert an operand to compute it in that type too.
        self.promotion = Promotion(term)
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
        is :meth:`loopty.term.Term.offsets_of`, the same one the access
        collector makes when it lists the read of the offsets, and a term that
        states its offsets (a program's) has them taken as stated.

        The added argument is ``off_<counts>``, suffixed with underscores when
        the term already uses that name. Only a term that states its offsets
        can: a kernel with a parameter of that name has it found as declared.
        """
        if name in self.ragged:
            return self.ragged[name]
        counts = self.counts_name(name)
        declared = self.term.offsets_of(counts)
        if declared is not None:
            self.ragged[name] = declared
            return declared
        candidate = f"off_{counts}"
        used = set(self.term.param_names) | set(self.term.sizes)
        used |= {temporary for temporary, _ in self.term.temporaries}
        used |= {symbol for symbol, _ in self.term.reflected}
        for stmt in self.term.stmts:
            used |= set(stmt.inames)
            for reduction in reductions_of(stmt.expr):
                used |= set(reduction.inames)
        while candidate in used:
            candidate = f"{candidate}_"
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
                if box_is_shape(domain):
                    return prim.Subscript(variable, rest)
                # A flat buffer, addressed row-major in the box: the address
                # of a point is the same term, and no extent is a shape.
                return prim.Subscript(variable, (linearize(rest, boxes[0]),))
            base = self.piece_base(name, position, boxes, "box")
            flat = _plus(base, linearize(rest, boxes[position]))
            return prim.Subscript(variable, (flat,))
        rows = [box[:-1] for box in boxes]
        base = self.piece_base(name, position, rows, "packed")
        entry = _plus(base, linearize(rest[:-1], rows[position]) if rest[:-1] else 0)
        start = prim.Subscript(prim.Variable(self.table_for(name)), (entry,))
        return prim.Subscript(variable, (_plus(start, rest[-1]),))

    def piece_base(
        self, name: str, position: int, boxes: Sequence[Sequence[Any]], storage: str
    ) -> Any:
        """Where piece ``position`` of the union ``name`` starts in its buffer.

        ``boxes`` are the boxes the pieces take up in it: the whole box of each
        for ``"box"``, and the box of its rows for the table of ``"packed"``.
        The start is the sum of the volumes of the boxes before it, which is a
        term of the sizes when every extent in them is non-negative at every
        size (:meth:`loopty.domain.Polyhedron.extents_never_negative`). One that
        can be negative belongs to a piece that is empty at those sizes and
        takes up no cells, which the term would not say (``(n - 1) * (n - 1)``
        is ``1`` at ``n = 0``), so the start is then a value argument the
        executor computes from the domain at the call's sizes, as it computes
        the table (:attr:`loopty.domain.Fixed.box_bases` and
        :attr:`~loopty.domain.Fixed.table_bases`).
        """
        if position == 0:
            return 0
        domain = self.arr_types[name].domain
        known = True
        for piece, box in zip(domain.pieces[:position], boxes, strict=False):
            known = known and all(piece.extents_never_negative()[: len(box)])
        if known:
            return _total(_volume(box) for box in boxes[:position])
        for argument, key in self.bases.items():
            if key == (name, position):
                return prim.Variable(argument)
        argument = self.fresh_argument(f"base_{name}_{position}")
        self.bases[argument] = (name, position)
        return prim.Variable(argument)

    def fresh_argument(self, base: str) -> str:
        """``base``, suffixed while it is a name the kernel already uses."""
        taken = set(dict(self.term.params)) | set(self.term.sizes)
        taken |= {iname for stmt in self.term.stmts for iname in stmt.inames}
        taken |= {symbol for symbol, _ in self.term.reflected}
        taken |= {arg.name for arg in self.extra_args}
        taken |= set(self.bases)
        candidate = base
        while candidate in taken:
            candidate = f"{candidate}_"
        return candidate

    def table_for(self, name: str) -> str:
        """The argument holding the table of row starts of a packed array.

        ``off_<name>``, suffixed while it is a name the kernel already uses, and
        an ``int32`` argument of no declared shape, which the executor fills
        from the array's domain (:meth:`loopty.domain.Fixed.table`).
        """
        if name in self.tables:
            return self.tables[name]
        candidate = self.fresh_argument(f"off_{name}")
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
        taken |= set(dict(self.term.temporaries))
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


def box_is_shape(domain: Any) -> bool:
    """Whether a single domain stored boxed is an array of the box's shape.

    It is when every extent of its box is non-negative at every size
    (:meth:`loopty.domain.Polyhedron.extents_never_negative`). Otherwise a
    size at which an extent is negative, ``n - 1`` at ``n = 0``, would give
    the array a negative shape, where the argument, whose domain is empty
    there, has a box of no cells; such a domain is a flat buffer instead,
    addressed row-major through the same extents, which differ from the
    argument's only where there is no point to address.
    """
    return not isinstance(domain, Union) and all(domain.extents_never_negative())


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

    #: A size expression is an integer, which numpy and C compute alike.
    promotion = None

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


def _shared_loops(stmt: Stmt, other: Stmt) -> int:
    """How many loops, outermost first, two statements have in common."""
    shared = 0
    for mine, theirs in zip(stmt.inames, other.inames, strict=False):
        if mine != theirs:
            break
        shared += 1
    return shared


def _ragged_rows(
    stmt: Stmt, ragged_bounds: Mapping[str, tuple[str, str]]
) -> dict[str, int]:
    """The ragged row lengths bounding ``stmt``, each with its row's position.

    A row length is assigned inside its row loop, so the statement's domain
    has to be cut after that loop: see :func:`_statement_domains`. Only the
    lengths whose row is one of the statement's loops are listed.
    """
    inames = tuple(stmt.inames)
    params = set(_domain_params(_domain_over(stmt.domain, inames)))
    return {
        param: inames.index(row)
        for param, (_counts, row) in ragged_bounds.items()
        if param in params and row in inames
    }


def _depth_cuts(
    term: Term, ragged_bounds: Mapping[str, tuple[str, str]] | None = None
) -> dict[str, frozenset[int]]:
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

    A statement bounded by a ragged row length is cut after the row loop as
    well, because the length is assigned inside it (``ragged_bounds``, see
    :func:`_statement_domains`). Every cut is then passed on to each statement
    that has the loops up to it and more loops beyond: two statements that
    share the loops ``r`` and ``i``, one of which goes on into a fiber of row
    ``r``, have to be cut alike after ``r``, or the one cut there defines ``r``
    in ``{ [r] }`` and ``i`` in ``[r] -> { [i] }`` while the other defines both
    in ``{ [r, i] }``, and no two of those domains merge. Passing a cut on can
    call for another, so it is repeated until nothing changes.

    A traced term gives every loop an iname of its own (a second ``for r`` is
    ``r_0``), so agreeing on a name is agreeing on a loop. A statement whose
    loops need no cut keeps its single domain, as every statement did before.
    """
    cuts: dict[str, set[int]] = {stmt.id: set() for stmt in term.stmts}
    for stmt in term.stmts:
        for other in term.stmts:
            if other is stmt:
                continue
            shared = _shared_loops(stmt, other)
            if 0 < shared < len(stmt.inames):
                cuts[stmt.id].add(shared - 1)
        rows = _ragged_rows(stmt, ragged_bounds or {})
        if rows and max(rows.values()) < len(stmt.inames) - 1:
            cuts[stmt.id].add(max(rows.values()))
    changed = True
    while changed:
        changed = False
        for stmt in term.stmts:
            for other in term.stmts:
                if other is stmt:
                    continue
                shared = _shared_loops(stmt, other)
                for cut in sorted(cuts[stmt.id]):
                    if cut < shared and cut < len(other.inames) - 1:
                        if cut not in cuts[other.id]:
                            cuts[other.id].add(cut)
                            changed = True
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
    rows = _ragged_rows(stmt, ragged_bounds)
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

    Which of the two it is is read off the term. Every statement that has the
    loop has the same loops around it when the name is one loop, as it always
    is in a traced term, and then the name is not the problem, and asking for a
    rename would send the reader after one that does not exist: the refusal
    says instead that the lowering could not give the loop one domain, and
    which statements share it.
    """
    defined: dict[str, isl.Set] = {}
    for domain in domains:
        for iname in domain.get_var_names(isl.dim_type.set):
            earlier = defined.setdefault(iname, domain)
            if earlier is domain:
                continue
            nests = {
                tuple(stmt.inames[: list(stmt.inames).index(iname) + 1])
                for stmt in term.stmts
                if iname in stmt.inames
            }
            if len(nests) == 1:
                users = ", ".join(
                    stmt.id for stmt in term.stmts if iname in stmt.inames
                )
                raise LoweringError(
                    f"{term.name} could not be lowered: the loop {iname}, which "
                    f"{users} share, came out of the lowering in two domains, "
                    f"{earlier} and {domain}, and loopy defines each loop in "
                    "one. This is a limit of loopty's lowering, which could not "
                    "cut the statements' domains alike, and not of the kernel; "
                    "moving the statements that differ into loops of their "
                    "own avoids it."
                )
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
    """How many loops of a statement enclose the innermost loop ``param`` bounds.

    ``loops`` is the statement's loop nest over its loop variables, before a
    guard narrowed it (its ``loop_domain``). A loop over a ragged fiber reads
    its bound where it starts, once per iteration of the loops around it, so
    those are the loops across whose iterations the body sees a row length
    change, and the innermost loop the length bounds has the most of them:
    in ``for j in val.dom[r]: for k in val.dom[r]:`` the loop over ``k``
    reads the length once per ``j``, and only counting the loop over ``j``
    missed a rewrite inside it (#111). A guard can make that loop's bound
    follow from another's, ``when(k == j)``, which isl then leaves out of the
    narrowed domain, and the body still reads it. When ``param`` bounds none
    of the loops, all of them count.
    """
    index = loops.find_dim_by_name(isl.dim_type.param, param)
    total = loops.dim(isl.dim_type.set)
    if index < 0:
        return total
    bounded = [
        position
        for position in range(total)
        if bounds_dimension(loops, position, index)
    ]
    return bounded[-1] if bounded else total


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
    how many of the statement's loops enclose the start of the innermost loop
    the bound bounds (:func:`_loops_before_fiber`, or every loop for a sum).
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
    _refuse_colliding_ids(term)
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
    cuts = _depth_cuts(term, ragged_bounds)

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
            predicates = frozenset([expr.condition(stmt.guard)])
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
    statement_of = {insn_id: stmt_id for stmt_id, insn_id in insn_ids.items()}
    for param, count_id in count_ids.items():
        if count_id in statement_of:
            raise LoweringError(
                f"statement {statement_of[count_id]!r} of {term.name} is "
                f"lowered as the instruction {count_id!r}, which is the id of "
                f"the instruction that computes the row length {param}, and "
                "loopy needs one instruction per id: give the statement "
                "another id"
            )
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
            nest = (
                loops
                if stmt.loop_domain is None
                else _domain_over(stmt.loop_domain, stmt.inames)
            )
            needed = {
                count_ids[param]
                for param in _domain_params(loops)
                if param in count_ids
            }
            for count_id in needed:
                uses.setdefault(count_id, []).append(
                    (stmt, _loops_before_fiber(nest, count_params[count_id]))
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
    preambles = () if contraction else _no_contraction_preambles(target)
    kernel = lp.make_kernel(
        merged,
        insns,
        args,
        target=target_for(target),
        lang_version=_LANG_VERSION,
        name=_kernel_name(term.name, [arg.name for arg in args]),
        preambles=(*preambles, *_power_preambles(term, target)),
    )
    if target in ("c", None):
        flags = [WRAP_FLAG] if contraction else [WRAP_FLAG, NO_CONTRACTION_FLAG]
        kernel = lp.set_options(kernel, build_options=flags)
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
        temporaries=tuple(
            name for name, _ in term.temporaries if name not in dict(term.checks)
        ),
        storage=dict(builder.storage),
        tables=dict(builder.tables),
        bases=dict(builder.bases),
        checks=dict(term.checks),
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


def _power_preambles(term: Term, target: str | None) -> tuple[tuple[str, str], ...]:
    """The headers a power needs on the C targets, when the term has one.

    loopy writes ``x ** 2`` as ``x * x``, and any other power as a call: of
    ``pow`` (``powf``) for a floating exponent, whose header ``math.h`` it
    never includes (note 2 in ``docs/loopy-notes.md``), and of a
    ``loopy_pow_<base>_<exponent>`` it defines for an integer one, whose
    signature names ``int32_t`` above the ``stdint.h`` it includes. Both
    failed to compile, ``x ** -1`` and ``x ** 0.5`` alike (#84). The lowering
    gives a floating power a floating exponent (:mod:`loopty.promotion`),
    which leaves the integer definition to integer powers. A complex base
    keeps an integer exponent, and the definition's signature names
    ``double complex`` above the ``complex.h`` loopy includes, so a term with
    complex values gets that header here too. See note 19.
    """
    if (target or "c") not in ("c", "c-source"):
        return ()
    nodes = [
        node
        for stmt in term.stmts
        for node in walk((stmt.assignee, stmt.expr, stmt.guard))
    ]
    if not any(
        isinstance(node, prim.Power) and not _written_as_product(node)
        for node in nodes
    ):
        return ()
    sorts = [typ.dtype for typ in term.array_types.values()]
    sorts += [sort for _, sort in term.params if not isinstance(sort, ArrType)]
    stored = [compiled_storage(sort) for sort in sorts]
    complex_term = any(dtype is not None and dtype.kind == "c" for dtype in stored)
    complex_term = complex_term or any(
        isinstance(node, complex | np.complexfloating) for node in nodes
    )
    if complex_term:
        return ((_POWER_TAG, f"{POWER_INCLUDES}\n{_COMPLEX_INCLUDE}"),)
    return ((_POWER_TAG, POWER_INCLUDES),)


def _written_as_product(power: prim.Power) -> bool:
    """Whether loopy writes a power without a call: exponent ``0``, ``1`` or ``2``."""
    exponent = power.exponent
    return isinstance(exponent, int | float | np.number) and exponent in (0, 1, 2)


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

    The sizes the kernel is passed are said to be non-negative too, whenever
    anything is said. An assumption brings every parameter of the kernel into
    the domain loopy checks an access over, and loopy checks an access only
    when that domain names everything the array's shape does. So ``off[0]``
    of ``off: Arr[Fin[n + 1], Nat]``, written outside any loop, went unchecked
    until a ``Nat`` scalar made ``n`` a parameter of the assumption, and was
    then refused for ``n = -1``. The contract refuses the one argument that
    would make it so, an ``off`` of no cells
    (:func:`loopty.contract.sizes_not_negative`).
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
    for name in term.sizes:
        if name in declared:
            pieces.append(f"{name} >= 0")
            names.add(name)
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

    A program's temporaries (:attr:`loopty.term.Term.temporaries`) come last,
    as loopy temporaries: see :func:`_temporary`.
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
    provided |= set(builder.bases)
    known_inames = {iname for stmt in term.stmts for iname in stmt.inames}
    for stmt in term.stmts:
        for reduction in reductions_of(stmt.expr):
            known_inames.update(reduction.inames)
    # A reduction binder that had to be renamed is an iname of the generated
    # kernel and not a size the caller passes; see _Builder.plan_reductions.
    known_inames |= builder.extra_inames
    temporaries = dict(term.temporaries)
    sizes = [
        name
        for name in used
        if name not in count_ids
        and name not in known_inames
        and name not in provided
        and name not in dict(term.params)
        and name not in temporaries
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
    # Then where a piece of a union starts, where that is not a term of them.
    for name in sorted(builder.bases):
        declare_value(name, np.dtype(np.int32))

    def shape_of(typ: ArrType, ragged: bool, name: str) -> tuple[Any, ...] | None:
        if ragged:
            return None
        if typ.domain is not None:
            # Only a single domain in a box has a shape, and only when no
            # extent of the box can be negative (box_is_shape); a union, or
            # packed rows, is a flat buffer of a length no loop bound states.
            if builder.storage[name] != "box" or not box_is_shape(typ.domain):
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
                array_dtype(term, name, typ.dtype),
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

    flags = dict(term.checks)
    for name, typ in term.temporaries:
        if name in flags:
            # A checked point's flag is the program's own, and an argument all
            # the same: the executor passes it zeroed and reads it after the
            # run, which a temporary would not let it do (loopty.compose).
            args.append(
                lp.GlobalArg(
                    name,
                    numpy_dtype(typ.dtype),
                    shape=(1,),
                    is_input=True,
                    is_output=True,
                )
            )
            array_args.append(name)
            outputs.append(name)
            declared.add(name)
            continue
        args.append(_temporary(term, name, typ, builder.target, declared | scalars))
        declared.add(name)

    return args, tuple(array_args), tuple(value_args), tuple(outputs)


def _temporary(
    term: Term, name: str, typ: ArrType, target: str | None, declared: set[str]
) -> Any:
    """The loopy temporary for one of a program's own arrays.

    Where it lives depends on the target, because loopy 2025.2 allocates a
    temporary in global memory on one target and not on the other (note 16 in
    ``docs/loopy-notes.md``). On OpenCL it is a global temporary, which the
    PyOpenCL host code allocates for each call. On C it is private, which the
    C target declares as a variable-length array on the stack of the call
    (``double f[n];``): the C host code never allocates a global temporary and
    passes the device function a pointer it never set, so a global one there
    crashes the process. The stack bounds how big such an array can be.

    A ragged temporary is refused, since nothing would give it its offsets, and
    so is one whose shape names a size nothing else in the kernel uses, which
    loopy would have no value for.
    """
    if any(typ.ragged):
        raise LoweringError(
            f"{name} is an array {term.name} makes for itself, and it is ragged; "
            "a temporary has no offsets for its rows to be found by, so make "
            "it a parameter of the program instead"
        )
    shape = tuple(_plain(size) for size in typ.axes)
    missing = sorted(
        {found for size in shape for found in _names_in(size)} - declared
    )
    if missing:
        raise LoweringError(
            f"the temporary {name} of {term.name} is sized by "
            f"{', '.join(missing)}, which nothing else in the kernel names, so "
            "loopy would have no value for it"
        )
    space = lp.AddressSpace.GLOBAL if target == "opencl" else lp.AddressSpace.PRIVATE
    return lp.TemporaryVariable(
        name, array_dtype(term, name, typ.dtype), shape=shape, address_space=space
    )


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
