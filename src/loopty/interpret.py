"""The term interpreter: what a traced term computes, run with numpy semantics.

Tracing runs a body once, at one generic point, and what it records is the
meaning every later step works from: the typing rules state obligations about
it, and lowering compiles it. The native run is the other meaning, the body
itself on real arrays. Nothing forces the two to agree. A body can keep state
where the tracer does not look, and then the term is one iteration of it, not
what it computes. This module gives the term a meaning of its own, without
loopy, so that the two can be compared (:mod:`loopty.faithful`).

:func:`interpret` runs a term on concrete arguments, in place, as the native run
does. Its reading of the term is literal:

* A statement's instances are the points of its isl domain at the sizes the
  arguments determine, enumerated with isl. A ragged bound such as ``cnt[r]``
  is a reflected parameter of the domain (see :class:`loopty.idx.Reflections`),
  and is read from the arrays once the loop variables it mentions have values.
* All instances of all statements run in the order of the source schedule, the
  2d+1 time vector that :func:`loopty.flow.schedule_of` states: statement by
  statement in source order within each iteration of the loops around them.
* A guard is evaluated at each instance, and an instance whose guard is false
  writes nothing. Guards and connectives short-circuit, so a condition to the
  right of a false one is not read.
* An expression is evaluated node by node with Python's operators on the numpy
  scalars read from the arrays, which is the arithmetic the native run does, in
  the order the tree was built. A call is looked up by the name loopy resolves
  against the target's library (``sqrt`` is :func:`numpy.sqrt`).
* A reduction adds its terms in the order the native ``reduce_sum`` does:
  lexicographically over its binders, the order of a generator's nested
  ``for`` clauses, with Python's :func:`sum`, starting from ``0``.
* A ragged array is read and written through the counts and offsets the
  kernel declares, as the statements before have left them
  (:meth:`loopty.arr.Arr.through`), which is how the native run and the
  lowered kernel index it too.

What the interpreter does not do is guess. A construct it has no numpy meaning
for, a domain it cannot enumerate at the given sizes, or a loop bound that reads
an array the same kernel writes (whose instances would depend on when the bound
is read) is an :class:`InterpretError` that says so, never a best effort.
"""

from __future__ import annotations

import builtins
import operator
from collections.abc import Iterator, Mapping, Sequence
from typing import Any

import islpy as isl
import numpy as np
import pymbolic.primitives as prim
from lanky.terms import Abs, render

from loopty.arr import Arr
from loopty.contract import integral_sort, resolve_sizes
from loopty.term import Access, ArrType, Reduction, Term, declared_layout
from loopty.trace import accesses_in

__all__ = ["InterpretError", "TooLarge", "interpret"]


class InterpretError(RuntimeError):
    """A term the interpreter cannot give a meaning to, with the reason."""


class TooLarge(InterpretError):
    """More statement instances than the caller allowed."""


class _Unknown(Exception):
    """A name with no value yet: a loop variable of a level not reached."""


#: The functions a term may call, by the name loopy resolves against the
#: target's math library, and the numpy function that computes the same thing.
_FUNCTIONS = {
    "sqrt": np.sqrt,
    "exp": np.exp,
    "log": np.log,
    "log2": np.log2,
    "log10": np.log10,
    "sin": np.sin,
    "cos": np.cos,
    "tan": np.tan,
    "asin": np.arcsin,
    "acos": np.arccos,
    "atan": np.arctan,
    "atan2": np.arctan2,
    "sinh": np.sinh,
    "cosh": np.cosh,
    "tanh": np.tanh,
    "fabs": np.fabs,
    "abs": np.abs,
    "floor": np.floor,
    "ceil": np.ceil,
    "pow": np.power,
    "fmin": np.fmin,
    "fmax": np.fmax,
}

_COMPARISONS = {
    "==": operator.eq,
    "!=": operator.ne,
    "<": operator.lt,
    "<=": operator.le,
    ">": operator.gt,
    ">=": operator.ge,
}

_BINARY = (
    (prim.Quotient, operator.truediv),
    (prim.FloorDiv, operator.floordiv),
    (prim.Remainder, operator.mod),
)


def interpret(
    term: Term, arguments: Mapping[str, Any], limit: int | None = None
) -> dict[str, np.ndarray]:
    """Run ``term`` on ``arguments``, in place, and return what it wrote.

    ``arguments`` maps every parameter to its value: an :class:`~loopty.arr.Arr`
    or a numpy array for an array parameter (a ragged one has to be an ``Arr``,
    which knows where its rows end), a number for a scalar. Arrays are written
    in place, as the native run writes them, and the result maps each array the
    term writes to its buffer.

    ``limit`` bounds the work: the statement instances and the terms of the
    reductions they evaluate, counted together, since one instance can sum a
    whole row. More is a :class:`TooLarge`, raised before anything runs when
    the statement instances alone are over, and otherwise at the reduction that
    goes over, with the arrays partly written. A domain is checked by its
    bounding box before its points are collected (see :meth:`_Run.enumerate`),
    so a domain far past the limit costs nothing to refuse. An array of an
    integral sort stored as floats is read as integers when the term does not
    write it, which is what the native run does (see
    :meth:`loopty.kernel.Kernel.__call__`).
    """
    return _Run(term, arguments).run(limit)


def _storage(value: Any, typ: ArrType, written: bool) -> Arr:
    """The runtime array the interpreter reads and writes for one argument."""
    if isinstance(value, np.ndarray) and not isinstance(value, Arr):
        value = Arr(value)
    if not isinstance(value, Arr):
        raise InterpretError(f"an array parameter was given {value!r}")
    buffer = value.numpy()
    if not written and integral_sort(typ.dtype) and buffer.dtype.kind in "fc":
        return value._replaced(np.real(buffer).astype(np.int64))
    return value


def _key(indices: Sequence[Any]) -> Any:
    """A subscript key the way the body writes it: one index, or a tuple."""
    return indices[0] if len(indices) == 1 else tuple(indices)


class _Run:
    """One interpretation: the arguments, the sizes, and the values of names."""

    def __init__(self, term: Term, arguments: Mapping[str, Any]) -> None:
        self.term = term
        written = {stmt.assignee.array for stmt in term.stmts}
        self.arrays: dict[str, Arr] = {}
        self.scalars: dict[str, Any] = {}
        for name, typ in term.params:
            if name not in arguments:
                raise InterpretError(f"no argument was given for {name}")
            value = arguments[name]
            if isinstance(typ, ArrType):
                self.arrays[name] = _storage(value, typ, name in written)
            else:
                self.scalars[name] = value
        for name, (counts, offsets) in declared_layout(term.params).items():
            array = self.arrays[name]
            if array.is_ragged:
                self.arrays[name] = array.through(
                    self.arrays.get(counts) if counts else None,
                    self.arrays.get(offsets) if offsets else None,
                )
        self.sizes = resolve_sizes(dict(term.params), arguments)
        self.reflected = dict(term.reflected)
        for parameter, expr in term.reflected:
            touched = sorted({a.array for a in accesses_in(expr)} & written)
            if touched:
                raise InterpretError(
                    f"the bound {render(expr)} (the domain parameter {parameter}) "
                    f"reads {', '.join(touched)}, which {term.name} also writes, "
                    "so which instances run depends on when the bound is read"
                )
        #: The work allowed (see :func:`interpret`) and the work done so far.
        self.limit: int | None = None
        self.spent = 0

    # {{{ running

    def spend(self) -> None:
        """Count one statement instance or one reduction term against the limit."""
        self.spent += 1
        if self.limit is not None and self.spent > self.limit:
            raise TooLarge(
                f"more than {self.limit} statement instances and reduction terms"
            )

    def enumerate(
        self, space: isl.Set, positions: Sequence[int]
    ) -> list[tuple[int, ...]]:
        """:func:`_enumerate` under the limit, checked before a point is visited.

        Collecting a domain's points is itself the work the limit bounds, so a
        domain with more points than the limit has left is refused first. The
        check is by the domain's bounding box, the bound isl gives without
        visiting points. A box can hold more points than its domain (a triangle
        fills half of one), so an input near the limit may be refused that
        would have fit; one far past it is refused at once, not after its
        points have been collected.
        """
        if self.limit is not None and not space.is_empty() and space.is_bounded():
            if _box_volume(space) > self.limit - self.spent:
                raise TooLarge(
                    f"more than {self.limit} statement instances and reduction terms"
                )
        return _enumerate(space, positions)

    def run(self, limit: int | None) -> dict[str, np.ndarray]:
        """Every instance of every statement, in the order of the source schedule."""
        self.limit = limit
        stmts = self.term.stmts
        depth = max((len(stmt.inames) for stmt in stmts), default=0)
        known = self._known()
        instances: list[tuple[tuple[int, ...], int, dict[str, int]]] = []
        for index, stmt in enumerate(stmts):
            order = stmt.order or (0,) * (len(stmt.inames) + 1)
            for point in self.points(stmt.domain, known):
                time: list[int] = []
                for level in range(depth):
                    time.append(order[level] if level < len(order) else 0)
                    time.append(
                        point[stmt.inames[level]] if level < len(stmt.inames) else 0
                    )
                time.append(order[depth] if depth < len(order) else 0)
                instances.append((tuple(time), index, point))
                self.spend()
        instances.sort(key=lambda item: item[0])
        for _time, index, point in instances:
            stmt = stmts[index]
            if stmt.guard is not None and not self.truth(stmt.guard, point):
                continue
            value = self.value(stmt.expr, point)
            indices = [self.index(i, point) for i in stmt.assignee.indices]
            self.arrays[stmt.assignee.array][_key(indices)] = value
        return {
            name: self.arrays[name].numpy()
            for name in dict.fromkeys(stmt.assignee.array for stmt in stmts)
        }

    def _known(self) -> dict[str, Any]:
        """The sizes and the numeric scalars, which a domain may name.

        A domain parameter is an integer to isl. A scalar that is not a whole
        number is kept as it is, so that fixing it refuses with its value
        (:func:`_fix_param`) rather than reporting the parameter unknown.
        """
        out: dict[str, Any] = dict(self.sizes)
        for name, value in self.scalars.items():
            if isinstance(value, bool | np.bool_):
                continue
            if isinstance(value, int | np.integer):
                out.setdefault(name, int(value))
            elif isinstance(value, float | np.floating):
                out.setdefault(name, value)
        return out

    # }}}

    # {{{ the points of a domain

    def points(
        self, domain: isl.Set, known: Mapping[str, int]
    ) -> Iterator[dict[str, int]]:
        """The points of ``domain`` in lexicographic order, as name-value maps.

        A set dimension ``known`` names is fixed there, and so is a parameter;
        every other parameter has to be a reflected one, which is fixed as soon
        as the loop variables its term mentions have values. Until then the
        dimension being walked is enumerated with the unknown parameters
        projected out, which can only give it more candidates, and the leaf
        asks the exact question with every parameter fixed. Once every
        parameter is fixed the rest is enumerated in one go.
        """
        names = domain.get_var_names(isl.dim_type.set)
        space = domain
        env = dict(known)
        for position, name in enumerate(names):
            if name in env:
                space = space.fix_val(isl.dim_type.set, position, int(env[name]))
        pending: dict[str, Any] = {}
        for name in domain.get_var_names(isl.dim_type.param):
            if name in env:
                space = _fix_param(space, name, env[name])
            elif name in self.reflected:
                pending[name] = self.reflected[name]
            else:
                raise InterpretError(
                    f"the domain {domain} has a parameter {name} that no argument "
                    "gives a value"
                )
        free = [k for k, name in enumerate(names) if name not in env]
        yield from self._walk(space, names, free, 0, env, pending)

    def _walk(
        self,
        space: isl.Set,
        names: Sequence[str],
        free: Sequence[int],
        level: int,
        env: dict[str, int],
        pending: dict[str, Any],
    ) -> Iterator[dict[str, int]]:
        """The points below one level of the walk; see :meth:`points`."""
        still: dict[str, Any] = {}
        for name, expr in pending.items():
            try:
                value = self.value(expr, env)
            except _Unknown:
                still[name] = expr
                continue
            space = _fix_param(space, name, value)
        if level == len(free):
            if still:
                raise InterpretError(
                    "the domain parameters "
                    + ", ".join(still)
                    + " are still unknown once every loop variable has a value"
                )
            if not space.is_empty():
                yield {name: env[name] for name in names}
            return
        if not still:
            for coordinates in self.enumerate(space, free[level:]):
                point = dict(env)
                point.update(
                    (names[k], value)
                    for k, value in zip(free[level:], coordinates, strict=True)
                )
                yield {name: point[name] for name in names}
            return
        # The dimensions after this one, and the parameters not known yet, are
        # projected out: what is left bounds this dimension alone, and bounds
        # it loosely, because a candidate the exact set does not have is
        # dropped at the leaf.
        position = free[level]
        shadow = space.project_out(
            isl.dim_type.set, position + 1, len(names) - position - 1
        )
        for name in still:
            shadow = shadow.project_out(
                isl.dim_type.param,
                shadow.find_dim_by_name(isl.dim_type.param, name),
                1,
            )
        for (value,) in self.enumerate(shadow, [position]):
            yield from self._walk(
                space.fix_val(isl.dim_type.set, position, value),
                names,
                free,
                level + 1,
                {**env, names[position]: value},
                still,
            )

    # }}}

    # {{{ expressions

    def truth(self, condition: Any, env: Mapping[str, Any]) -> bool:
        """Whether a guard holds at one instance."""
        return bool(self.value(condition, env))

    def index(self, expr: Any, env: Mapping[str, Any]) -> Any:
        """One subscript, as the body would compute it."""
        return self.value(expr, env)

    def read(self, array: str, indices: Sequence[Any], env: Mapping[str, Any]) -> Any:
        """One array element, read the way the native run reads it."""
        if array not in self.arrays:
            raise InterpretError(f"{array} is subscripted and is not an array")
        key = _key([self.index(i, env) for i in indices])
        return self.arrays[array][key]

    def value(self, node: Any, env: Mapping[str, Any]) -> Any:
        """The value of one expression at one point, by numpy's arithmetic."""
        if isinstance(node, bool | int | float | complex | np.generic):
            return node
        if isinstance(node, Reduction):
            return self.reduction(node, env)
        if isinstance(node, Access):
            return self.read(node.array, node.indices, env)
        if isinstance(node, prim.Subscript):
            aggregate = node.aggregate
            if not isinstance(aggregate, prim.Variable):
                raise InterpretError(f"cannot subscript {render(aggregate)}")
            index = node.index
            indices = index if isinstance(index, tuple) else (index,)
            return self.read(aggregate.name, indices, env)
        if isinstance(node, prim.Variable):
            return self.variable(node.name, env)
        if isinstance(node, prim.Sum):
            return _fold(operator.add, [self.value(c, env) for c in node.children])
        if isinstance(node, prim.Product):
            return _fold(operator.mul, [self.value(c, env) for c in node.children])
        for kind, apply in _BINARY:
            if isinstance(node, kind):
                return apply(
                    self.value(node.numerator, env), self.value(node.denominator, env)
                )
        if isinstance(node, prim.Power):
            return self.value(node.base, env) ** self.value(node.exponent, env)
        if isinstance(node, prim.Comparison):
            compare = _COMPARISONS.get(node.operator)
            if compare is None:
                raise InterpretError(f"no meaning for the comparison {node.operator}")
            return compare(self.value(node.left, env), self.value(node.right, env))
        if isinstance(node, prim.LogicalAnd):
            return all(self.truth(child, env) for child in node.children)
        if isinstance(node, prim.LogicalOr):
            return any(self.truth(child, env) for child in node.children)
        if isinstance(node, prim.LogicalNot):
            return not self.truth(node.child, env)
        if isinstance(node, prim.If):
            branch = node.then if self.truth(node.condition, env) else node.else_
            return self.value(branch, env)
        if isinstance(node, prim.Min | prim.Max):
            pick = builtins.min if isinstance(node, prim.Min) else builtins.max
            return pick(self.value(child, env) for child in node.children)
        if isinstance(node, Abs):
            return builtins.abs(self.value(node.operand, env))
        if isinstance(node, prim.Call):
            return self.call(node, env)
        if isinstance(node, tuple):
            return tuple(self.value(item, env) for item in node)
        raise InterpretError(
            f"no numpy meaning for {type(node).__name__} in the term: "
            f"{_text(node)}"
        )

    def variable(self, name: str, env: Mapping[str, Any]) -> Any:
        """A loop or binder variable, a scalar argument, or a size."""
        if name in env:
            return env[name]
        if name in self.scalars:
            return self.scalars[name]
        if name in self.sizes:
            return self.sizes[name]
        raise _Unknown(name)

    def call(self, node: prim.Call, env: Mapping[str, Any]) -> Any:
        """A call of a library function, by the name the term gives it."""
        function = node.function
        name = function.name if isinstance(function, prim.Variable) else None
        numpy_function = _FUNCTIONS.get(name) if name is not None else None
        if numpy_function is None:
            raise InterpretError(
                f"no numpy counterpart for the call {_text(node)}; the "
                f"interpreter knows {', '.join(sorted(_FUNCTIONS))}"
            )
        return numpy_function(*(self.value(arg, env) for arg in node.parameters))

    def reduction(self, node: Reduction, env: Mapping[str, Any]) -> Any:
        """A reduction, summed in the order the native ``reduce_sum`` sums it.

        The domain's dimensions are the statement's inames, fixed by ``env``,
        followed by the reduction's own. An enclosing reduction's binders are
        parameters of the domain, and ``env`` gives them too.
        """
        if node.op != "sum":
            raise InterpretError(f"no meaning for a reduction of kind {node.op!r}")
        known = {**self._known(), **env}
        terms = []
        for point in self.points(node.domain, known):
            self.spend()
            terms.append(self.value(node.body, {**env, **point}))
        return builtins.sum(terms)

    # }}}


def _fold(apply: Any, values: Sequence[Any]) -> Any:
    """``((a op b) op c) ...``: Python's order for a chain of one operator."""
    out = values[0]
    for value in values[1:]:
        out = apply(out, value)
    return out


def _fix_param(space: isl.Set, name: str, value: Any) -> isl.Set:
    """``space`` with the parameter ``name`` fixed to an integer value."""
    if isinstance(value, bool) or not isinstance(value, int | np.integer):
        if isinstance(value, float | np.floating) and float(value).is_integer():
            value = int(value)
        else:
            raise InterpretError(
                f"the domain parameter {name} is {value!r}, which is not an integer"
            )
    position = space.find_dim_by_name(isl.dim_type.param, name)
    return space.fix_val(isl.dim_type.param, position, int(value))


def _enumerate(space: isl.Set, positions: Sequence[int]) -> list[tuple[int, ...]]:
    """The distinct values of the dimensions at ``positions``, sorted."""
    if space.is_empty():
        return []
    if not space.is_bounded():
        raise InterpretError(
            f"the domain {space} is unbounded at these sizes, so its points "
            "cannot be enumerated"
        )
    found: set[tuple[int, ...]] = set()

    def visit(point: Any) -> None:
        found.add(
            tuple(
                point.get_coordinate_val(isl.dim_type.set, k).to_python()
                for k in positions
            )
        )

    space.foreach_point(visit)
    return sorted(found)


def _box_volume(space: isl.Set) -> int:
    """How many points the bounding box of a bounded, non-empty set holds."""
    volume = 1
    for position in range(space.dim(isl.dim_type.set)):
        low = space.dim_min_val(position).to_python()
        high = space.dim_max_val(position).to_python()
        volume *= high - low + 1
    return volume


def _text(node: Any) -> str:
    """A node rendered for a message, or its ``repr`` when lanky cannot."""
    try:
        return render(node)
    except Exception:  # noqa: BLE001 - a message, not a result
        return repr(node)
