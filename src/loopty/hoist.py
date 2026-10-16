"""A sum's operands that read none of its binders, computed once before it.

A sum is evaluated term by term, and so is every operand of its body. In
``examples/l2p.py``'s ``evaluate``, ``x`` and ``y`` are Python locals, which
the trace reads into the summand, so the code loopty lowered divided twice
and loaded four times at every term of the triangle, where plain loopy, as a
kernel author writes it, computes them once per target (``<> x = ...``
before the sum). The compilers do not move them out of the triangle's loops,
and on the H200 that was 1.37x plain loopy at order 10 (#270).

:func:`hoist_invariants` lowers such an operand into a temporary of the work
item, assigned before the sum in every iteration of the loops around it::

    for (int32_t target = 0; target <= -1 + nl_cnt_box_1; ++target)
    {
      acc_a_1_b_1 = (double) (0.0);
      invariant_0 = (tx[off_cnt[box] + target] + -1.0 * cx[box]) / rscale;
      invariant_1 = (ty[off_cnt[box] + target] + -1.0 * cy[box]) / rscale;
      for (int32_t a_1 = 0; a_1 <= 10; ++a_1)
        for (int32_t b_1 = 0; b_1 <= 10 + -1 * a_1; ++b_1)
          acc_a_1_b_1 = acc_a_1_b_1 + R_window[11 * a_1 + b_1]
            * loopy_pow_float64_int32(invariant_0, a_1)
            * loopy_pow_float64_int32(invariant_1, b_1) * INV_FACT[a_1] * ...;

*The same bits.* The temporary holds the value the operand has where it
stands, so the sum adds the same terms in the same order:

- The operand is a whole subexpression of the body, or the leading operands
  of a sum or a product, which C computes first: ``c * w[i] * f[j]`` is
  ``(c * w[i]) * f[j]``. Nothing is reassociated; ``f[j] * c * w[i]`` keeps
  both of its products in the sum.
- Its type is a floating or a complex one, the type loopy infers for it,
  which C computes it in, and every operation from the sum down to it is
  written in the same type context, so its literals are written as they
  were (note 25 in ``docs/loopy-notes.md``). A ``float32`` operand of a
  double sum stays where it is.
- Where an output is ``exact``, the kernel does not contract
  (:func:`loopty.lower.allows_contraction`), so a product stored in a
  temporary rounds as it did in the sum. Where it may, the output is
  ``approx``, and a multiply-add the compiler fused across the operand's
  edge may now round twice, within the tolerance the class states.

*The same reads.* The temporary is computed at every instance of the
statement, under its predicates and after what it waits for, where the sum
computed it at each of its terms, of which there may be none:

- An operand that reads arrays is moved where isl finds each cell it reads
  inside its array at every instance of the statement, as loopy checks every
  instruction's reads (``x[i]`` in a loop over ``i in x.dom``), or else
  where the sum has a term at every instance, so that it read the same cells
  there before: ``x[i - 1]`` in a sum over ``j in Fin[i]`` stays, since
  ``i = 0`` has no term and no ``x[-1]``. One that reads only scalars, sizes
  and loops is moved anywhere: loopty's arithmetic is defined everywhere,
  a division by zero included (:mod:`loopty.operations`).
- Nothing is moved out of a branch that is not always taken, of a
  conditional, an ``and`` or an ``or``, out of a subscript, which loopy reads
  as affine, or out of a substitution.

An operand of a sum nested in another that reads the outer sum's binders and
not the inner's stays: a loopy instruction cannot run inside another's sum.
Note 33 in ``docs/loopy-notes.md`` has the account.

This is done on the kernel code is generated from, after every step:
:attr:`loopty.schedule.Schedule.kernel`, and the kernel of a kernel or a
program run without a schedule (:func:`loopty.executor._resolve`). The steps
transform and check the kernel as it was lowered, and no fact is about the
temporaries: the ledger is the same, and so is every native run.
"""

from __future__ import annotations

import itertools
from collections.abc import Callable
from typing import Any

import islpy as isl
import loopy as lp
import pymbolic.primitives as prim
from loopy.symbolic import Reduction as LoopyReduction
from loopy.symbolic import TypeCast, get_dependencies

from loopty.lower import walk

__all__ = ["hoist_invariants"]

#: The nodes whose operands are always evaluated where they are, and are
#: written in the type context the node is written in (a call's arguments in
#: the types of its parameters). Every other node is not looked into: a
#: conditional, ``and`` and ``or`` evaluate a part only sometimes, a
#: subscript is index arithmetic, and a comparison writes its operands in the
#: type of their difference.
_OPERATIONS = (prim.Sum, prim.Product, prim.Quotient, prim.Power, TypeCast, prim.Call)

#: The nodes that compute something, and are worth a temporary: not a load,
#: a scalar, a conversion of one or its negation, which the lowering writes
#: ``-1 * x`` for ``-x`` and ``a - b``.
_COMPUTING = (
    prim.Sum,
    prim.Product,
    prim.Quotient,
    prim.FloorDiv,
    prim.Remainder,
    prim.Power,
    prim.Call,
    prim.If,
)


def hoist_invariants(kernel: Any) -> Any:
    """``kernel`` with each sum's operands that read none of its binders before it.

    See the module. ``None``, a kernel a step could not write, stays
    ``None``; a kernel loopy cannot type is left as it is, for code
    generation to refuse with its own message.
    """
    if kernel is None:
        return None
    entry = kernel.default_entrypoint
    candidates = [
        insn
        for insn in entry.instructions
        if isinstance(insn, lp.Assignment) and _has_reduction(insn.expression)
    ]
    if not candidates:
        return kernel
    try:
        typed = lp.infer_unknown_types(kernel, expect_completion=False)
    except Exception:  # noqa: BLE001 - code generation refuses it, with its message
        return kernel
    typed_entry = typed.default_entrypoint

    taken = entry.get_var_name_generator()
    numbers = itertools.count()

    def fresh() -> str:
        """``invariant_0``, ``invariant_1``, ...: the next name the kernel has not."""
        while True:
            name = f"invariant_{next(numbers)}"
            if not taken.is_name_conflicting(name):
                taken.add_name(name)
                return name

    ids = entry.get_instruction_id_generator()
    temporaries = dict(entry.temporary_variables)
    rewritten: dict[str, list[Any]] = {}
    for insn in candidates:
        typed_insn = typed_entry.id_to_insn.get(insn.id)
        if typed_insn is None:  # pragma: no cover - type inference keeps the ids
            continue
        operands = _Operands(entry, typed, insn, fresh)
        expression = operands.outside(insn.expression, typed_insn.expression)
        if not operands.found:
            continue
        before = []
        for name, (operand, dtype) in operands.found.items():
            temporaries[name] = lp.TemporaryVariable(
                name, dtype, shape=(), address_space=lp.AddressSpace.PRIVATE
            )
            before.append(
                insn.copy(
                    id=ids(f"{insn.id}_{name}"),
                    assignee=prim.Variable(name),
                    expression=operand,
                    atomicity=(),
                    temp_var_type=lp.Optional(),
                )
            )
        rewritten[insn.id] = [
            *before,
            insn.copy(
                expression=expression,
                depends_on=insn.depends_on | {each.id for each in before},
            ),
        ]
    if not rewritten:
        return kernel
    instructions = [
        new for insn in entry.instructions for new in rewritten.get(insn.id, [insn])
    ]
    return kernel.with_kernel(
        entry.copy(instructions=instructions, temporary_variables=temporaries)
    )


def _has_reduction(expr: Any) -> bool:
    """Whether a sum is anywhere in ``expr``."""
    return any(isinstance(node, LoopyReduction) for node in walk(expr))


class _Operands:
    """The operands of one instruction's sums that read none of their binders.

    The instruction's expression is walked beside the same expression in the
    typed kernel, which says the type of each node. :attr:`found` names each
    operand moved, by its temporary, with its type; an operand met twice
    shares one temporary, which holds the same value.
    """

    def __init__(
        self, entry: Any, typed: Any, insn: Any, fresh: Callable[[], str]
    ) -> None:
        from loopy.type_inference import TypeReader

        self.entry = entry
        self.insn = insn
        self.fresh = fresh
        self.typed_entry = typed.default_entrypoint
        self.callables = typed.callables_table
        self.reader = TypeReader(self.typed_entry, self.callables)
        self.found: dict[str, tuple[Any, Any]] = {}
        self._by_operand: dict[Any, str] = {}
        self._always: dict[frozenset[str], bool] = {}
        #: The instruction's instances, once :meth:`_domain` has asked.
        self._instances: tuple[Any] | None = None
        self._memory = set(entry.temporary_variables) | {
            arg.name for arg in entry.args if isinstance(arg, lp.ArrayArg)
        }

    # {{{ walking

    def outside(self, expr: Any, typed: Any) -> Any:
        """``expr``, outside every sum, with the sums in it rewritten."""
        if isinstance(expr, LoopyReduction):
            context = self._context(typed)
            if context is None:
                return expr
            return self._reduction(expr, typed, frozenset(), context)
        return self._children(
            expr, typed, lambda child, other, _ctx: self.outside(child, other), None
        )

    def _reduction(
        self, expr: Any, typed: Any, bound: frozenset[str], context: str
    ) -> Any:
        """A sum, its body rewritten with its binders and the enclosing ones bound."""
        if not isinstance(typed, LoopyReduction) or typed.inames != expr.inames:
            return expr
        body = self._inside(expr.expr, typed.expr, bound | set(expr.inames), context)
        if body is expr.expr:
            return expr
        return LoopyReduction(
            expr.operation, expr.inames, body, expr.allow_simultaneous
        )

    def _inside(
        self, expr: Any, typed: Any, bound: frozenset[str], context: str
    ) -> Any:
        """``expr`` in a sum over ``bound``, written in ``context``."""
        if self._context(typed) != context:
            return expr
        if isinstance(expr, LoopyReduction):
            return self._reduction(expr, typed, bound, context)
        if self._invariant(expr, bound):
            if _computes(expr):
                moved = self._move(expr, typed, bound)
                if moved is not None:
                    return moved
            # A leaf, or an operand that cannot move as a whole: a part of it
            # may.
        if isinstance(expr, prim.Sum | prim.Product) and self._real(typed):
            led = self._leading(expr, typed, bound, context)
            if led is not None:
                return led
        return self._children(
            expr,
            typed,
            lambda child, other, ctx: self._inside(child, other, bound, ctx),
            context,
        )

    def _leading(
        self, expr: Any, typed: Any, bound: frozenset[str], context: str
    ) -> Any:
        """A real sum or product whose leading operands read no binder, them moved.

        C computes ``a * b * c`` as ``(a * b) * c``, so the leading ``a * b``
        is an operand the product already has, in a temporary of the
        product's own type; any later ones are not. ``None`` when fewer than
        two lead, or all do (the whole moves then, or nothing does).
        """
        children = expr.children
        if len(typed.children) != len(children):
            return None
        count = 0
        while count < len(children) and self._invariant(children[count], bound):
            count += 1
        if count < 2 or count == len(children):
            return None
        kind = type(expr)
        lead, typed_lead = kind(children[:count]), kind(typed.children[:count])
        if self._dtype(typed_lead) != self._dtype(typed):
            return None
        moved = self._move(lead, typed_lead, bound)
        if moved is None:
            return None
        rest = [
            self._inside(child, other, bound, context)
            for child, other in zip(
                children[count:], typed.children[count:], strict=True
            )
        ]
        return kind((moved, *rest))

    def _children(
        self,
        expr: Any,
        typed: Any,
        rec: Callable[[Any, Any, str | None], Any],
        context: str | None,
    ) -> Any:
        """``expr`` with ``rec`` applied to each operand :data:`_OPERATIONS` evaluates.

        ``rec`` is given the operand, its typed twin, and the type context it
        is written in: the node's, or a call's parameter's.
        """
        if not isinstance(expr, _OPERATIONS) or type(typed) is not type(expr):
            return expr
        if isinstance(expr, prim.Sum | prim.Product):
            if len(typed.children) != len(expr.children):
                return expr
            children = tuple(
                rec(child, other, context)
                for child, other in zip(expr.children, typed.children, strict=True)
            )
            if all(
                new is old for new, old in zip(children, expr.children, strict=True)
            ):
                return expr
            return type(expr)(children)
        if isinstance(expr, prim.Quotient):
            numerator = rec(expr.numerator, typed.numerator, context)
            denominator = rec(expr.denominator, typed.denominator, context)
            if numerator is expr.numerator and denominator is expr.denominator:
                return expr
            return prim.Quotient(numerator, denominator)
        if isinstance(expr, prim.Power):
            base = rec(expr.base, typed.base, context)
            exponent = rec(expr.exponent, typed.exponent, context)
            if base is expr.base and exponent is expr.exponent:
                return expr
            return prim.Power(base, exponent)
        if isinstance(expr, TypeCast):
            child = rec(expr.child, typed.child, context)
            return expr if child is expr.child else TypeCast(expr.type, child)
        return self._call(expr, typed, rec, context)

    def _call(
        self,
        expr: Any,
        typed: Any,
        rec: Callable[[Any, Any, str | None], Any],
        context: str | None,
    ) -> Any:
        """A call, its arguments each in the type context of its parameter."""
        name = getattr(typed.function, "name", None)
        if (
            name is None
            or name in self.entry.substitutions
            or name not in self.callables
            or len(typed.parameters) != len(expr.parameters)
        ):
            return expr
        dtypes = self.callables[name].arg_id_to_dtype or {}
        parameters = []
        for position, (parameter, other) in enumerate(
            zip(expr.parameters, typed.parameters, strict=True)
        ):
            dtype = dtypes.get(position)
            if dtype is None:
                parameters.append(parameter)
                continue
            inner = _type_context(self.entry, dtype)
            if context is not None and inner != context:
                parameters.append(parameter)
                continue
            parameters.append(rec(parameter, other, inner))
        if all(
            new is old for new, old in zip(parameters, expr.parameters, strict=True)
        ):
            return expr
        return type(expr)(expr.function, tuple(parameters))

    # }}}

    # {{{ moving

    def _move(self, expr: Any, typed: Any, bound: frozenset[str]) -> Any:
        """The temporary ``expr`` is moved into, or ``None`` where it may not move."""
        name = self._by_operand.get(expr)
        if name is not None:
            return prim.Variable(name)
        dtype = self._dtype(typed)
        if dtype is None or dtype.numpy_dtype.kind not in "fc":
            return None
        if (
            get_dependencies(expr) & self._memory
            and not self._reads_in_bounds(expr)
            and not self._always_a_term(bound)
        ):
            return None
        name = self.fresh()
        self._by_operand[expr] = name
        self.found[name] = (expr, dtype)
        return prim.Variable(name)

    def _invariant(self, expr: Any, bound: frozenset[str]) -> bool:
        """Whether ``expr`` reads no binder, and holds no sum and no substitution."""
        if get_dependencies(expr) & bound:
            return False
        return not any(
            isinstance(node, LoopyReduction)
            or (
                isinstance(node, prim.Call)
                and getattr(node.function, "name", None) in self.entry.substitutions
            )
            for node in walk(expr)
        )

    def _reads_in_bounds(self, expr: Any) -> bool:
        """Whether every cell ``expr`` reads is in its array at every instance.

        As loopy's own check of an instruction's reads asks it
        (``loopy.check._AccessCheckMapper``): the range of each subscript
        over the instruction's loops, its predicates isl reads and the
        kernel's assumptions, inside the box of the array's shape, for every
        value of the sizes. A subscript isl cannot read, one that reads an
        array (``tx[off[r] + j]``), and a scalar temporary, whose value may
        be stored under a condition, answer no.
        """
        domain = self._domain()
        if domain is None:
            return False
        for name in get_dependencies(expr) & set(self.entry.temporary_variables):
            if not self.entry.temporary_variables[name].shape:
                return False
        return all(
            self._in_bounds(node, domain)
            for node in walk(expr)
            if isinstance(node, prim.Subscript)
        )

    def _domain(self) -> Any:
        """The instruction's instances, as loopy checks its reads over them.

        ``None`` where isl declines them. Never asked whether it is empty:
        truth on an isl set is its length, which islpy deprecates.
        """
        if self._instances is None:
            from loopy.kernel.instruction import get_insn_domain

            try:
                domain = get_insn_domain(self.insn, self.entry)
                if not self.entry.assumptions.is_universe():
                    domain, assumptions = isl.align_two(domain, self.entry.assumptions)
                    domain = domain & assumptions
                self._instances = (domain,)
            except isl.Error:  # pragma: no cover - isl declines: none in bounds
                self._instances = (None,)
        return self._instances[0]

    def _in_bounds(self, node: Any, domain: Any) -> bool:
        """Whether the cells subscript ``node`` reads are inside its array's shape."""
        from loopy.diagnostic import LoopyError
        from loopy.isl_helpers import make_slab
        from loopy.symbolic import get_access_map

        if not isinstance(node.aggregate, prim.Variable):
            return False
        name = node.aggregate.name
        array = self.entry.arg_dict.get(name) or self.entry.temporary_variables.get(
            name
        )
        shape = getattr(array, "shape", None)
        index = node.index if isinstance(node.index, tuple) else (node.index,)
        if (
            not isinstance(shape, tuple)
            or len(shape) != len(index)
            or any(axis is None for axis in shape)
        ):
            return False
        sizes = set().union(*(get_dependencies(axis) for axis in shape))
        missing = sizes - set(domain.get_var_dict())
        values = {arg.name for arg in self.entry.args if isinstance(arg, lp.ValueArg)}
        if not missing <= values:
            return False
        for size in sorted(missing):
            position = domain.dim(isl.dim_type.param)
            domain = domain.add_dims(isl.dim_type.param, 1).set_dim_name(
                isl.dim_type.param, position, size
            )
        if not get_dependencies(index) <= set(domain.get_var_dict()):
            return False
        try:
            cells = get_access_map(domain, index).range()
            box = isl.BasicSet.universe(cells.get_space())
            for axis, extent in enumerate(shape):
                box = box.intersect(
                    make_slab(box.get_space(), (isl.dim_type.in_, axis), 0, extent)
                )
            return cells.is_subset(box)
        except (LoopyError, isl.Error):
            return False

    def _always_a_term(self, bound: frozenset[str]) -> bool:
        """Whether the sums over ``bound`` have a term at every instance.

        isl compares the loops of the instruction with what is left of the
        loops and the sums' binders once the binders are projected out; the
        sizes and the lengths of ragged rows are free, so it holds for every
        one of them.
        """
        known = self._always.get(bound)
        if known is not None:
            return known
        loops = sorted(self.insn.within_inames)
        try:
            whole = self.entry.get_inames_domain(
                frozenset(self.insn.within_inames) | bound
            )
            nest = self.entry.get_inames_domain(frozenset(self.insn.within_inames))
            whole = whole.project_out_except(loops, [isl.dim_type.set])
            nest = nest.project_out_except(loops, [isl.dim_type.set])
            nest, whole = isl.align_two(nest, whole)
            whole = isl.align_spaces(whole, nest)
            always = nest.is_subset(whole)
        except isl.Error:  # pragma: no cover - isl declines: the reads stay
            always = False
        self._always[bound] = always
        return always

    # }}}

    # {{{ types

    def _dtype(self, typed: Any) -> Any:
        """The type loopy infers for a node of the typed kernel, or ``None``."""
        try:
            found = self.reader(typed)
        except Exception:  # noqa: BLE001 - a node loopy cannot type stays put
            return None
        if not isinstance(found, list | tuple):
            # A sum's type comes back on its own, any other node's in a list.
            return found
        if len(found) != 1:
            return None
        return found[0]

    def _context(self, typed: Any) -> str | None:
        """The type context loopy writes a node of the typed kernel in, if any."""
        dtype = self._dtype(typed)
        return None if dtype is None else _type_context(self.entry, dtype)

    def _real(self, typed: Any) -> bool:
        """Whether a node is real-valued: loopy groups a complex product's factors."""
        dtype = self._dtype(typed)
        return dtype is not None and dtype.numpy_dtype.kind == "f"

    # }}}


def _type_context(entry: Any, dtype: Any) -> str | None:
    """loopy's type context for ``dtype``: ``"d"``, ``"f"``, ``"i"`` or ``"b"``."""
    from loopy.expression import dtype_to_type_context

    return dtype_to_type_context(entry.target, dtype)


def _computes(expr: Any) -> bool:
    """Whether ``expr`` computes something, outside its subscripts.

    A load, a scalar, or a conversion or a negation of one, is left where
    it is: a temporary would hold what a register already does.
    """
    if isinstance(expr, TypeCast):
        return _computes(expr.child)
    if (
        isinstance(expr, prim.Product)
        and len(expr.children) == 2
        and _is_minus_one(expr.children[0])
    ):
        return _computes(expr.children[1])
    return isinstance(expr, _COMPUTING)


def _is_minus_one(value: Any) -> bool:
    """Whether ``value`` is the literal ``-1``, of any numeric type."""
    import numpy as np

    return (
        isinstance(value, int | float | np.number)
        and not isinstance(value, bool)
        and value == -1
    )
