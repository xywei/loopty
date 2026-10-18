"""A program's postcondition, proved by unfolding what its calls store.

The design idea. A program's postcondition is a claim about what it leaves
in its parameters once its calls have run. ``weak_grad`` in
``examples/hexahedra.py`` claims that ``grad`` is grudge's weak gradient of
``u``: for each component, the dense Kronecker operator of each reference
direction applied to the field times that direction's metric term, summed::

    grad[x, e, k, j, i] == sum(Ft[3*x + d, k, c] * Fs[3*x + d, j, b]
                               * Fr[3*x + d, i, a] * G[3*x + d, e, c, b, a]
                               * u[e, c, b, a] for d, c, b, a ...)

Its calls compute that in three passes, which sum in another order. isl
decides a claim whose equations of reals are atoms
(:func:`loopty.dataflow.decide_postcondition`), and this one is not such a
claim: it holds because three passes are the dense operator, which is a
theorem the program cites (``@program(uses=[sum_factorization])``), and
because addition and multiplication of reals are what they are, which
Lean's ``ring`` knows.

So the claim is stated for Lean, a part of Mathlib's for each conjunct, from
what each call stores (:func:`loopty.dataflow.program_stores`), a universal
equation about the program's arrays per statement, and from the theorems the
program cites. Python finds the steps, and Lean checks them, as it checks
every fact loopty states (:mod:`lanky.obligations`):

1. The arrays the claim reads are unfolded into what the calls stored in
   them, a pass at a time: ``grad`` into the sum over ``c`` of ``Ft`` times
   ``vs``, ``vs`` into its sum, ``v`` into its, until one side of a cited
   theorem is there. Lean's ``simp only`` unfolds them, each premise (the
   cell is one the call stored) shown by ``omega``, inside a sum from the
   range of the sum.
2. Each instance of that side is rewritten into the other, ``rw [← h ...]``,
   with the theorem's variables given: its families at the sections of the
   arrays they stand for, ``Fr (3 * x + 0)`` for ``Fr`` and ``w (3 * x + 0)
   e`` for ``u``, found by matching the theorem's side against the unfolded
   claim (:class:`_Sections`).
3. What is left is unfolded too, each sum over a literal range is written
   out term by term, and ``ring_nf`` compares the two sides.

A step Python got wrong leaves a goal Lean does not close, and the fact
stays as the native runs left it, with Lean's reason. Each equation and
each theorem is a hypothesis of the part, so Lean proves the claim of any
arrays that hold what the calls store, and the claim rests on the
``single-assignment`` facts that make those equations true of the program,
and on the theorems.

The claim Lean proves is about the reals: each cell a call stores is the
real its expression computes, where the float a run stores is that real
rounded, step by step, in the order the call sums. That is what an array of
``Real`` cells promises (:mod:`loopty.tolerance`), and so the native runs
compare a program's postcondition at the class of the outputs it reads,
``approx`` for reals, as the compiled run is compared with the native one
(:func:`loopty.faithful.postcondition_fact`).

What is not done: a claim that needs an equation whose stored cell is read
back by the same statement (a recurrence, ``off[r + 1] == off[r] +
cnt[r]``) is not unfolded, since rewriting with it would not end; a sum
whose domain is not a ``Fin`` bound loopty wrote is not stated; and a
theorem is used only where all its variables are given by the instance
found, and none of them is a variable a sum around the instance binds.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

import islpy as isl
import pymbolic.primitives as prim
from lanky.prelude import FinType, FnType, Real
from lanky.terms import (
    Comparison,
    Forall,
    Subscript,
    Var,
    conjuncts,
    init_args,
    render,
    structurally_equal,
)
from lanky.terms import Sum as LankySum

from loopty.description import Unstatable, conjunction, description_of
from loopty.hypotheses import (
    _BINDING,
    _call_spine,
    _fresh,
    _indices,
    _Matcher,
    _names,
    structural_key,
    substitute,
)
from loopty.term import ArrType, Hypothesis, Reduction, Term

__all__ = ["Plan", "Step", "Unfolded", "as_sums", "plan", "unfolded"]

#: How many steps a plan takes at most, unfoldings and rewrites together.
_STEPS = 64


# {{{ loopty's sums as lanky's


def as_sums(expr: Any, outer: frozenset[str] = frozenset()) -> Any:
    """``expr`` with each of loopty's sums a lanky sum, which Mathlib states.

    A :class:`~loopty.term.Reduction` over ``a`` is ``sum(... for a in
    Fin[n])``: its domain, as loopty wrote it (:mod:`loopty.description`),
    bounds ``a`` below by ``0`` and above by ``n``, and what else it says of
    ``a`` is the sum's guard. What it says of the loops around it is where
    the statement runs, which an equation is claimed under already: the
    conjuncts in ``outer``, rendered, are left out, and any other is a guard
    of the sum too.

    Raises:
        Unstatable: For a reduction that is not a sum, a domain isl computed,
            or a binder with no lower bound ``0`` and no upper bound loopty
            wrote.
    """
    if isinstance(expr, Reduction):
        return _as_sum(expr, outer)
    if isinstance(expr, prim.ExpressionNode):
        args = init_args(expr)
        new = tuple(as_sums(arg, outer) for arg in args)
        if all(old is changed for old, changed in zip(args, new, strict=True)):
            return expr
        return type(expr)(*new)
    if isinstance(expr, tuple):
        return tuple(as_sums(item, outer) for item in expr)
    return expr


def _as_sum(reduction: Reduction, outer: frozenset[str]) -> Any:
    """One reduction as a lanky sum (see :func:`as_sums`)."""
    if reduction.op != "sum":
        raise Unstatable(
            f"a {reduction.op} over {', '.join(reduction.inames)} has no form of "
            "Mathlib's here"
        )
    described = description_of(reduction.domain)
    if described is None:
        raise Unstatable(
            f"the sum over {', '.join(reduction.inames)} is over a set isl computed"
        )
    dims = reduction.domain.get_var_names(isl.dim_type.set)
    parts = list(conjuncts(described.predicate([Var(name) for name in dims])))
    inames = set(reduction.inames)
    used: set[int] = set()
    binders = []
    for iname in reduction.inames:
        lower = upper = None
        for k, part in enumerate(parts):
            if k in used:
                continue
            found = _bound(part, iname)
            if found is None:
                continue
            side, value = found
            if side == "lower" and lower is None:
                lower = value
                used.add(k)
            elif side == "upper" and upper is None:
                upper = value
                used.add(k)
        if lower is None or upper is None:
            raise Unstatable(
                f"the sum over {iname} is not over 0 <= {iname} < n as loopty "
                "wrote its domain"
            )
        binders.append((Var(iname), FinType(upper)))
    guard = [
        part
        for k, part in enumerate(parts)
        if k not in used and (_names(part) & inames or render(part) not in outer)
    ]
    body = as_sums(reduction.body, outer | {render(part) for part in parts})
    return LankySum(tuple(binders), body, None if not guard else conjunction(guard))


def _bound(part: Any, iname: str) -> tuple[str, Any] | None:
    """``("lower", 0)`` for ``0 <= a``, ``("upper", n)`` for ``a < n``, else ``None``.

    ``a <= m`` is the upper bound ``m + 1``, and either may be written the
    other way round. A bound that names ``a`` itself is no bound.
    """
    if not isinstance(part, prim.Comparison):
        return None
    left, operator, right = part.left, part.operator, part.right
    mirrored = {"<": ">", "<=": ">=", ">": "<", ">=": "<="}
    if isinstance(right, prim.Variable) and right.name == iname:
        if not (isinstance(left, prim.Variable) and left.name == iname):
            left, right = right, left
            operator = mirrored.get(operator, operator)
    if not (isinstance(left, prim.Variable) and left.name == iname):
        return None
    if iname in _names(right):
        return None
    if operator == ">=" and _is_zero(right):
        return "lower", 0
    if operator == ">" and right == -1:
        return "lower", 0
    if operator == "<":
        return "upper", right
    if operator == "<=":
        return "upper", right + 1
    return None


def _is_zero(expr: Any) -> bool:
    return isinstance(expr, int) and not isinstance(expr, bool) and expr == 0


def statable_equation(equation: Any) -> Any:
    """A value equation (:func:`loopty.dataflow._equation`) with its sums lanky's.

    The sums are read in the statement's own domain, the equation's guard.

    Raises:
        Unstatable: As :func:`as_sums` does.
    """
    if not isinstance(equation, Forall) or not isinstance(equation.body, Comparison):
        return as_sums(equation)
    outer = frozenset(render(part) for part in conjuncts(equation.guard))
    body = equation.body
    value = as_sums(body.right, outer)
    if value is body.right:
        return equation
    return Forall(
        equation.binders,
        Comparison(body.left, body.operator, value),
        equation.guard,
    )


# }}}


# {{{ the steps: unfold, rewrite with a theorem


@dataclass(frozen=True, eq=False)
class Step:
    """One step of a plan: a hypothesis the goal is rewritten with.

    ``hypothesis`` is the proposition: an equation a call stores, which
    unfolds its array wherever the goal reads it, or a cited theorem's
    statement, taken at ``arguments``, one per variable of the theorem in
    the order it binds them, and rewritten from its right side into its
    left with ``backwards``.
    """

    hypothesis: Any
    arguments: tuple[Any, ...] = ()
    backwards: bool = False
    #: The theorem the step takes, for the fact it rests on; ``None`` for
    #: an equation.
    theorem: Any = None


@dataclass(frozen=True, eq=False)
class Plan:
    """The steps that take one conjunct of a postcondition to ``ring``.

    ``conjunct`` is the claim, with its sums lanky's; ``steps`` are in
    order. The hypotheses of the part are the theorems and the equations the
    steps take, each once.
    """

    conjunct: Any
    steps: tuple[Step, ...]

    @property
    def hypotheses(self) -> tuple[Any, ...]:
        """The theorems the steps take, then the equations, each once, in order."""
        out: list[Any] = []
        seen: set[Any] = set()
        for step in sorted(self.steps, key=lambda step: step.theorem is None):
            key = structural_key(step.hypothesis)
            if key not in seen:
                seen.add(key)
                out.append(step.hypothesis)
        return tuple(out)

    @property
    def theorems(self) -> tuple[Any, ...]:
        """The theorems the steps take, each once, in order."""
        out: list[Any] = []
        for step in self.steps:
            if step.theorem is not None and all(
                step.theorem is not seen for seen in out
            ):
                out.append(step.theorem)
        return tuple(out)


def _unfoldable(equation: Any) -> tuple[str, tuple[str, ...], Any] | None:
    """The array, the binders and the value of an equation ``simp`` unfolds.

    ``all(v[q, e, k, j, i] == ... for ...)``: the cell is the array at its
    binders, in order, so that every cell of the array the goal reads is an
    instance, and the value does not read the array, so that unfolding ends.
    ``None`` for any other.
    """
    if not isinstance(equation, Forall) or not isinstance(equation.body, Comparison):
        return None
    body = equation.body
    if body.operator != "==":
        return None
    cell = body.left
    if not (
        isinstance(cell, prim.Subscript) and isinstance(cell.aggregate, prim.Variable)
    ):
        return None
    array = cell.aggregate.name
    names = tuple(var.name for var, _ in equation.binders)
    indices = _indices(cell)
    if len(indices) != len(names) or not all(
        isinstance(index, prim.Variable) and index.name == name
        for index, name in zip(indices, names, strict=True)
    ):
        return None
    if len(set(names)) != len(names) or array in _names(body.right):
        return None
    return array, names, body.right


def _unfold(expr: Any, array: str, names: tuple[str, ...], value: Any) -> Any:
    """``expr`` with every cell of ``array`` it reads replaced by ``value`` there."""
    if (
        isinstance(expr, prim.Subscript)
        and isinstance(expr.aggregate, prim.Variable)
        and expr.aggregate.name == array
    ):
        indices = _indices(expr)
        if len(indices) == len(names):
            indices = tuple(_unfold(index, array, names, value) for index in indices)
            return substitute(value, dict(zip(names, indices, strict=True)))
    if isinstance(expr, prim.ExpressionNode):
        args = init_args(expr)
        new = tuple(_unfold(arg, array, names, value) for arg in args)
        if all(old is changed for old, changed in zip(args, new, strict=True)):
            return expr
        return type(expr)(*new)
    if isinstance(expr, tuple):
        return tuple(_unfold(item, array, names, value) for item in expr)
    return expr


def _read(expr: Any) -> set[str]:
    """The arrays ``expr`` reads a cell of."""
    out: set[str] = set()

    def visit(node: Any) -> None:
        if isinstance(node, prim.Subscript) and isinstance(
            node.aggregate, prim.Variable
        ):
            out.add(node.aggregate.name)
        if isinstance(node, prim.ExpressionNode):
            for arg in init_args(node):
                visit(arg)
        elif isinstance(node, tuple | list):
            for item in node:
                visit(item)

    visit(expr)
    return out


class _Sections(_Matcher):
    """A theorem's side matched against a goal, a family at a section of an array.

    As :class:`loopty.hypotheses._Matcher` matches, but a family the theorem
    applies, ``Fr[i, a]``, matches an array read at more indices,
    ``Fr[3*x + 0, i, a]``: the family is the array with its first indices
    fixed, its section ``Fr[3*x + 0]``, and the indices left are its
    arguments. A section's fixed indices name nothing a sum binds, in the
    theorem's side or around it in the goal (``outside``), since the family
    is one function for the whole side.
    """

    def __init__(self, functions: Any, scalars: Any, outside: frozenset[str]) -> None:
        super().__init__(functions, scalars)
        self.outside = outside

    def match(
        self, pattern: Any, target: Any, sigma: dict[str, Any], bound: dict[str, str]
    ) -> bool:
        if (
            isinstance(pattern, prim.Variable)
            and pattern.name in self.functions
            and pattern.name not in bound
        ):
            section = _section(target, 0)
            if section is None or _names(section) & self._bound(bound):
                return False
            return _bind(pattern.name, section, sigma)
        if isinstance(pattern, prim.Call | prim.Subscript):
            spine = _call_spine(pattern)
            if spine is not None and spine[0] in self.functions:
                name, parameters = spine
                if not (
                    isinstance(target, prim.Subscript)
                    and isinstance(target.aggregate, prim.Variable)
                ):
                    return False
                indices = _indices(target)
                cut = len(indices) - len(parameters)
                if cut < 0:
                    return False
                section = _section(target, len(parameters))
                if section is None or _names(section) & self._bound(bound):
                    return False
                if not _bind(name, section, sigma):
                    return False
                return all(
                    self.match(p, t, sigma, bound)
                    for p, t in zip(parameters, indices[cut:], strict=True)
                )
        return super().match(pattern, target, sigma, bound)

    def _bound(self, bound: dict[str, str]) -> set[str]:
        return set(bound.values()) | set(self.outside)


def _section(target: Any, arguments: int) -> Any:
    """``target``, an array read, with its last ``arguments`` indices taken off.

    ``Fr`` for ``Fr`` at no index, ``Fr[3*x + 0]`` for ``Fr[3*x + 0, i, a]``
    and two arguments; ``None`` for anything else.
    """
    if isinstance(target, prim.Variable):
        return target if arguments == 0 else None
    if not (
        isinstance(target, prim.Subscript)
        and isinstance(target.aggregate, prim.Variable)
    ):
        return None
    indices = _indices(target)
    fixed = indices[: len(indices) - arguments]
    if not fixed:
        return Var(target.aggregate.name)
    return Subscript(Var(target.aggregate.name), fixed[0] if len(fixed) == 1 else fixed)


def _bind(name: str, section: Any, sigma: dict[str, Any]) -> bool:
    if name in sigma:
        return structurally_equal(sigma[name], section)
    sigma[name] = section
    return True


def _applied(section: Any, arguments: tuple[Any, ...]) -> Any:
    """A section applied: ``Fr[3*x + 0]`` at ``(i, a)`` is ``Fr[3*x + 0, i, a]``.

    A section is an array, or an array read at its first indices
    (:func:`_section`), and the arguments follow those.
    """
    if isinstance(section, prim.Subscript):
        fixed = _indices(section)
        aggregate = section.aggregate
    else:
        fixed, aggregate = (), section
    indices = (*fixed, *arguments)
    if not indices:
        return aggregate
    return Subscript(aggregate, indices[0] if len(indices) == 1 else indices)


def _instantiate(expr: Any, sigma: dict[str, Any], functions: set[str]) -> Any:
    """A theorem's side at ``sigma``: its families the sections, its scalars the terms.

    The side's own binders are named apart from every name the values
    mention first, so that none of them is captured.
    """
    taken: set[str] = set()
    for value in sigma.values():
        taken |= _names(value)
    expr = _apart(expr, taken)
    scalars = {name: value for name, value in sigma.items() if name not in functions}

    def visit(node: Any) -> Any:
        if isinstance(node, prim.Call | prim.Subscript):
            spine = _call_spine(node)
            if spine is not None and spine[0] in functions and spine[0] in sigma:
                return _applied(sigma[spine[0]], tuple(visit(arg) for arg in spine[1]))
        if isinstance(node, prim.Variable) and node.name in functions:
            return sigma.get(node.name, node)
        if isinstance(node, prim.ExpressionNode):
            args = init_args(node)
            return type(node)(*(visit(arg) for arg in args))
        if isinstance(node, tuple):
            return tuple(visit(item) for item in node)
        return node

    return substitute(visit(expr), scalars)


def _apart(expr: Any, taken: set[str]) -> Any:
    """``expr`` with each binder whose name is in ``taken`` renamed to a fresh one."""
    if isinstance(expr, _BINDING):
        renames: dict[str, Any] = {}
        binders = []
        for var, domain in expr.binders:
            if var.name in taken:
                fresh = _fresh(var.name, taken | _names(expr))
                taken = taken | {fresh}
                renames[var.name] = Var(fresh)
                binders.append((Var(fresh), domain))
            else:
                binders.append((var, domain))
        body = _apart(substitute(expr.body, renames), taken)
        guard = expr.guard
        if guard is not None:
            guard = _apart(substitute(guard, renames), taken)
        return type(expr)(tuple(binders), body, guard)
    if isinstance(expr, prim.ExpressionNode):
        args = init_args(expr)
        return type(expr)(*(_apart(arg, taken) for arg in args))
    if isinstance(expr, tuple):
        return tuple(_apart(item, taken) for item in expr)
    return expr


def _replace(expr: Any, old: Any, new: Any) -> Any:
    """``expr`` with every subterm that is ``old``, as a term, replaced by ``new``."""
    key = structural_key(old)

    def visit(node: Any) -> Any:
        if isinstance(node, prim.ExpressionNode):
            if structural_key(node) == key:
                return new
            args = init_args(node)
            changed = tuple(visit(arg) for arg in args)
            if all(a is b for a, b in zip(args, changed, strict=True)):
                return node
            return type(node)(*changed)
        if isinstance(node, tuple):
            return tuple(visit(item) for item in node)
        return node

    return visit(expr)


def _instances(
    goal: Any, theorem: Any, backwards: bool
) -> list[tuple[Any, tuple, Any]]:
    """Each instance of one side of ``theorem`` in ``goal``, once.

    Returns the subterm, the theorem's arguments (one per variable, in the
    order its statement binds them) and the other side there. An instance
    some variable of the theorem is not given at, or whose arguments name a
    variable a sum around it binds, is left out: ``rw`` takes a closed
    instance, and one inside a binder is no term of the goal.
    """
    variables = dict(getattr(theorem, "variables", ()))
    statement = getattr(theorem, "term", None)
    goal_of = getattr(theorem, "goal", None)
    if not variables or not isinstance(statement, Forall) or goal_of is None:
        return []
    if not (isinstance(goal_of, prim.Comparison) and goal_of.operator == "=="):
        return []
    functions = {name for name, sort in variables.items() if isinstance(sort, FnType)}
    scalars = set(variables) - functions
    pattern, other = (
        (goal_of.right, goal_of.left) if backwards else (goal_of.left, goal_of.right)
    )
    order = [var.name for var, _ in statement.binders]
    if set(order) != set(variables):
        return []
    found: list[tuple[Any, tuple, Any]] = []
    seen: set[Any] = set()

    def visit(node: Any, outside: frozenset[str]) -> None:
        if isinstance(node, prim.ExpressionNode):
            matcher = _Sections(functions, scalars, outside)
            sigma: dict[str, Any] = {}
            if matcher.match(pattern, node, sigma, {}) and set(sigma) == set(variables):
                if not any(
                    _names(value) & outside
                    for name, value in sigma.items()
                    if name in scalars
                ):
                    key = structural_key(node)
                    if key not in seen:
                        seen.add(key)
                        arguments = tuple(sigma[name] for name in order)
                        found.append(
                            (node, arguments, _instantiate(other, sigma, functions))
                        )
                    return
            inner = outside
            if isinstance(node, _BINDING):
                inner = outside | {var.name for var, _ in node.binders}
            for arg in init_args(node):
                visit(arg, inner)
        elif isinstance(node, tuple | list):
            for item in node:
                visit(item, outside)

    visit(goal, frozenset())
    return found


def plan(
    conjunct: Any, equations: Sequence[Any], theorems: Sequence[Any]
) -> Plan | None:
    """The steps that take ``conjunct`` to a goal ``ring`` closes, or ``None``.

    ``equations`` are what the calls store, their sums lanky's; ``theorems``
    the cited ones. Each round, an instance of a theorem's right side in the
    goal is rewritten into its left (or of its left into its right, a
    theorem taken one way only), and when there is none, every array the
    goal reads that an equation unfolds is unfolded. The rounds end when
    nothing is left to unfold. ``None`` when nothing was unfolded: the calls
    store nothing the claim reads.
    """
    body = conjunct.body if isinstance(conjunct, Forall) else conjunct
    if not (isinstance(body, prim.Comparison) and body.operator == "=="):
        return None
    unfolds: dict[str, tuple[Any, tuple[str, ...], Any]] = {}
    for equation in equations:
        found = _unfoldable(equation)
        if found is not None and found[0] not in unfolds:
            unfolds[found[0]] = (equation, found[1], found[2])
    goal: Any = body
    steps: list[Step] = []
    direction: dict[int, bool] = {}
    unfolded = False
    for _round in range(_STEPS):
        rewritten = False
        for theorem in theorems:
            ways = (
                [direction[id(theorem)]] if id(theorem) in direction else [True, False]
            )
            for backwards in ways:
                found = _instances(goal, theorem, backwards)
                if not found:
                    continue
                direction[id(theorem)] = backwards
                for subterm, arguments, other in found:
                    steps.append(Step(theorem.term, arguments, backwards, theorem))
                    goal = _replace(goal, subterm, other)
                rewritten = True
                break
        if rewritten:
            continue
        arrays = sorted(_read(goal) & set(unfolds))
        if not arrays:
            break
        for array in arrays:
            equation, names, value = unfolds[array]
            steps.append(Step(equation))
            goal = _unfold(goal, array, names, value)
        unfolded = True
    else:
        return None
    if not unfolded:
        return None
    return Plan(conjunct, tuple(steps))


# }}}


# {{{ the claim, stated for Lean


def _real_sort(typ: Any) -> bool:
    from loopty.dataflow import _real

    return _real(typ.dtype if isinstance(typ, ArrType) else typ)


@dataclass(frozen=True, eq=False)
class Unfolded:
    """A program's postcondition as the term of its fact, stated for Lean.

    One part per conjunct (:class:`Plan`): the theorems and the equations its
    steps take imply the conjunct, a part of Mathlib's since the arrays hold
    reals, with the steps as :class:`lanky.obligations.Rewrite` hints. It
    prints as the postcondition, and is offered to Lean alone: no other
    oracle takes it, so without Lean the fact keeps what the native runs
    found. Each theorem is a citation (:func:`loopty.obligations.cited`),
    which lanky restates from the text the theorem's own command proved.
    """

    term: Term
    plans: tuple[Plan, ...]
    rests_on: tuple[str, ...] = ()

    def lean_obligation(self) -> Any:
        """The parts, one per conjunct, stated from the equations and the theorems."""
        from loopty.obligations import (
            Claim,
            _declined,
            _lanky_obligations,
            _obligation,
            cited,
        )

        lanky = _lanky_obligations()
        fields = getattr(lanky.Part, "__dataclass_fields__", {})
        if not hasattr(lanky, "Rewrite") or "mathlib" not in fields:
            return _declined(
                "this lanky's Lean oracle rewrites no part's goal with its "
                "hypotheses; lanky 0.3 does"
            )
        sorts = {
            name: Real
            for name, typ in (*self.term.params, *self.term.temporaries)
            if _real_sort(typ)
        }
        parts = []
        for k, plan_ in enumerate(self.plans):
            claim = Claim(self.term, sorts=sorts)
            claim.hypotheses.extend(plan_.hypotheses)
            try:
                closed = claim.close(plan_.conjunct)
            except Unstatable as exc:
                return _declined(str(exc))
            guards = [render(part) for part in conjuncts(closed.guard)]
            rewrites = []
            for step in plan_.steps:
                try:
                    position = guards.index(render(step.hypothesis))
                except ValueError:
                    return _declined(
                        "a hypothesis the steps take is no hypothesis of the part"
                    )
                rewrites.append(lanky.Rewrite(position, step.arguments, step.backwards))
            parts.append(
                lanky.Part(
                    f"conjunct_{k}" if len(self.plans) > 1 else "claim",
                    render(plan_.conjunct),
                    closed,
                    lanky.Hints(rewrites=tuple(rewrites)),
                    mathlib=True,
                    **cited(guards, plan_.theorems),
                )
            )
        return _obligation(parts, rests_on=self.rests_on, decider="")

    def __str__(self) -> str:
        return render(self.term.post)


def unfolded(
    term: Term, equations: Sequence[Hypothesis], theorems: Sequence[Any]
) -> tuple[Unfolded, tuple[Hypothesis, ...], str] | str:
    """The postcondition of a program's ``term`` stated for Lean, or why it is not.

    ``equations`` are what its calls store (:func:`loopty.dataflow.program_stores`)
    and ``theorems`` those it cites. Every conjunct has to have a
    :func:`plan`. Returns the term of the fact, the equations the plans take,
    and the plans' words for the provenance.
    """
    if term.post is None:
        return "the program claims nothing"
    statable: list[tuple[Hypothesis, Any]] = []
    for hypothesis in equations:
        try:
            statable.append((hypothesis, statable_equation(hypothesis.claim)))
        except Unstatable:
            continue
    claims = [claim for _hypothesis, claim in statable]
    plans = []
    for conjunct in conjuncts(term.post):
        try:
            found = plan(conjunct, claims, theorems)
        except Unstatable as exc:
            return str(exc)
        if found is None:
            return (
                f"what the calls store does not unfold {render(conjunct)}: it reads "
                "no array a call stores once, or reads it through a recurrence"
            )
        plans.append(found)
    keys = {
        structural_key(hypothesis) for found in plans for hypothesis in found.hypotheses
    }
    used = tuple(
        hypothesis for hypothesis, claim in statable if structural_key(claim) in keys
    )
    theorem_ids = tuple(
        dict.fromkeys(
            getattr(theorem, "fact_id", "")
            for found in plans
            for theorem in found.theorems
        )
    )
    rests = tuple(
        dict.fromkeys(
            [
                *(fact for hypothesis in used for fact in hypothesis.rests_on),
                *theorem_ids,
            ]
        )
    )
    cited = sorted(
        {
            getattr(theorem, "__name__", "a claim")
            for found in plans
            for theorem in found.theorems
        }
    )
    words = "stated for Lean from what the calls store" + (
        f" and {' and '.join(cited)}" if cited else ""
    )
    return Unfolded(term, tuple(plans), rests), used, words


# }}}
