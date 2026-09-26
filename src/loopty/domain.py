"""Polyhedral domains: array arguments over a set of index tuples that is not a box.

loopty's index types are isl sets, and inside a kernel much of that universe is
already used: ``when`` restricts a statement's domain, a ragged loop iterates a
dependent sum, and dependences are exact per instance. This module brings the
same sets to the *interface*, as the domain of an array argument:

.. code-block:: python

    L: Arr[Where[i: Fin[n], j: Fin[n], j < i], Real]  # a box cut by constraints
    T: Arr[Sigma[i: Fin[n], Fin[i + 1]], Real]  # a sum with affine fibers
    B: Arr[Where[i: Fin[n], j: Fin[n], (i - j <= 1) & (j - i <= 1)], Real]
    U: Arr[Fin[n] + Fin[m], Real]  # a union of pieces

(``B`` is a band.)
Python's grammar reads ``i: Fin[n]`` inside a subscript as a slice, and lanky's
annotation scope turns the unknown names ``i`` and ``j`` into symbols, so a
binder is written where it is used and nothing is parsed. ``Where[...]`` takes
binders and then constraints, ``Sigma[...]`` binders and an unnamed last fiber,
and ``Fin[n] + Fin[m]`` is lanky's :class:`~lanky.prelude.SumType`, whose pieces
are read here. All three become one of two objects: a :class:`Polyhedron`, the
set of tuples whose ``k``-th entry lies below the ``k``-th binder's bound (which
may name the binders before it) and which satisfies every constraint, or a
:class:`Union`, whose points are ``(p, x)`` with ``x`` a point of piece ``p``.

*The type is the exact set.* :meth:`Polyhedron.isl_set` is what an in-bounds
obligation compares an access with, so ``L[i, i]`` is refused although it is a
cell of the ``n x n`` box around the triangle. A constraint is therefore
restricted to what isl states exactly: comparisons of quasi-affine terms,
joined by ``&``. ``!=`` and ``|`` are refused rather than widened, since a
domain wider than it was written would let a read outside it be decided in
bounds; a union of pieces is how a non-convex domain is written.

*Iteration is binder by binder.* ``L.dom`` runs over the points of the first
binder's domain that the constraints on it alone allow, and ``L.dom[i]`` over
the points of the second at ``i`` that every constraint on the first two
allows, and so on: a loop over ``L.dom[i]`` inside a loop over ``L.dom`` is
the domain, and a statement inside only the first runs over every ``i`` below
``n``. :meth:`Polyhedron.prefix_constraints` is that reading for the tracer and
:meth:`Fixed.fiber` for a native run, and the two agree by construction. A
fiber taken over a point outside the domain is empty, in both. The pieces of a
union are iterated by number, ``for p in U.dom``, which is a Python loop over
``range`` in a trace as well: a piece is chosen by a Python integer, and each
piece traces as a statement of its own.

*Storage is a layout, not a type.* Two layouts store the same array
(:class:`Fixed` computes both, and :mod:`loopty.lower` generates both):

* ``"box"``, the bounding-box embedding: the box of the binders' own bounds,
  row-major, with the cells outside the domain wasted. Its map is affine.
* ``"packed"``, row by row through a table: the cells of the domain in
  lexicographic order, and a table giving, for every row (every value of the
  axes but the last), where the row starts less its first column, so that the
  cell ``(r, j)`` lives at ``table[r] + j``. A packed triangle's address,
  ``i(i - 1)/2 + j``, is not quasi-affine, and the table is what makes it one
  read. It needs every row to be an interval, which a domain whose constraints
  involve no remainder always has, and which is checked.

A union stores its pieces one after another, each in the array's layout.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from typing import Any

import islpy as isl
import numpy as np
import pymbolic.primitives as prim
from lanky.prelude import FinType, SumType
from lanky.terms import Comparison, init_args, render, structurally_equal

__all__ = [
    "STORAGES",
    "Fixed",
    "Polyhedron",
    "Sigma",
    "Union",
    "Where",
    "fixed_set",
    "index_domain",
    "is_domain",
    "linear",
    "same_points",
    "substitute",
]

#: The layouts an array over a domain can be stored in; see the module docstring.
STORAGES = ("box", "packed")

#: The comparisons a constraint may make, and how isl spells each.
_OPERATORS = {"<": "<", "<=": "<=", ">": ">", ">=": ">=", "==": "="}


# {{{ terms


def _names(expr: Any) -> set[str]:
    """Every variable name a term mentions."""
    if isinstance(expr, prim.Variable):
        return {expr.name}
    if isinstance(expr, prim.ExpressionNode):
        out: set[str] = set()
        for arg in init_args(expr):
            out |= _names(arg)
        return out
    if isinstance(expr, tuple | list):
        out = set()
        for item in expr:
            out |= _names(item)
        return out
    return set()


def substitute(expr: Any, mapping: Mapping[str, Any]) -> Any:
    """``expr`` with every variable ``mapping`` names replaced, node types kept.

    lanky's nodes are rebuilt as lanky's nodes, so a constraint with a binder
    replaced by a loop variable is still a proposition the tracer can render.
    """
    if isinstance(expr, prim.Variable):
        return mapping.get(expr.name, expr)
    if isinstance(expr, prim.ExpressionNode):
        return type(expr)(*(substitute(arg, mapping) for arg in init_args(expr)))
    if isinstance(expr, tuple):
        return tuple(substitute(item, mapping) for item in expr)
    return expr


def linear(expr: Any) -> bool:
    """Whether ``expr`` is an integer combination of names plus a constant."""
    if isinstance(expr, bool):
        return False
    if isinstance(expr, int | np.integer | prim.Variable):
        return True
    if isinstance(expr, prim.Sum):
        return all(linear(child) for child in expr.children)
    if isinstance(expr, prim.Product):
        unknown = [c for c in expr.children if not isinstance(c, int | np.integer)]
        return len(unknown) <= 1 and all(linear(child) for child in expr.children)
    return False


def _isl_text(expr: Any, rename: Mapping[str, str]) -> str:
    """``expr`` in isl's syntax, or :class:`ValueError` when isl cannot state it."""
    from loopty.flow import NonAffine, expr_text

    try:
        return expr_text(expr, rename, None)
    except NonAffine as exc:
        raise ValueError(str(exc)) from exc


def conjuncts(prop: Any) -> list[Any]:
    """The comparisons a constraint is the conjunction of, or :class:`TypeError`."""
    if isinstance(prop, prim.LogicalAnd):
        return [piece for child in prop.children for piece in conjuncts(child)]
    if isinstance(prop, prim.Comparison):
        if prop.operator not in _OPERATORS:
            raise TypeError(
                f"the constraint {_shown(prop)} compares with {prop.operator!r}, and "
                "the points it leaves are not one convex set, so isl could not "
                "state the domain exactly. Write the domain as a sum of pieces, "
                "one on each side: Where[..., a < b] + Where[..., a > b]"
            )
        return [prop]
    if isinstance(prop, prim.LogicalOr):
        raise TypeError(
            f"the constraint {_shown(prop)} is a disjunction, and a domain is one "
            "conjunction of comparisons, which is what isl states exactly. Write "
            "the domain as a sum of pieces, one for each side of the '|'"
        )
    raise TypeError(
        f"{_shown(prop)} is not a constraint: a domain is cut by comparisons of "
        "its binders and sizes (<, <=, >, >=, ==), joined by '&'"
    )


def _shown(value: Any) -> str:
    """A term the way a message prints it."""
    if isinstance(value, prim.ExpressionNode):
        try:
            return render(value)
        except Exception:  # pragma: no cover - a node lanky cannot print
            return str(value)
    return repr(value)


# }}}


# {{{ the written forms


def _binder(part: slice, family: str) -> tuple[str, Any]:
    """``i: Fin[n]`` read as ``(name, bound)``; see :class:`Polyhedron`."""
    name, domain = part.start, part.stop
    if part.step is not None:
        raise TypeError(
            f"a binder of {family}[...] is written 'name: Fin[bound]', and "
            f"{_shown(name)}: {_shown(domain)}: {_shown(part.step)} has a third part"
        )
    if not isinstance(name, prim.Variable):
        raise TypeError(
            f"a binder of {family}[...] is written 'name: Fin[bound]' with a "
            f"fresh name, and {name!r} is not a name. A name the module defines "
            "shadows the binder of the same spelling; rename one of them"
        )
    if not isinstance(domain, FinType):
        raise TypeError(
            f"the binder {name.name} of {family}[...] ranges over {_shown(domain)}, "
            f"and a binder ranges over Fin[bound]: write {name.name}: Fin[...]"
        )
    return name.name, domain.bound


class _WhereFamily:
    """The ``Where`` name itself: ``Where[i: Fin[n], ..., constraint, ...]``."""

    def __getitem__(self, item: Any) -> Polyhedron:
        """Build the domain; binders first, then the constraints."""
        parts = item if isinstance(item, tuple) else (item,)
        binders: list[tuple[str, Any]] = []
        constraints: list[Any] = []
        for part in parts:
            if isinstance(part, slice):
                if constraints:
                    raise TypeError(
                        "Where[...] takes its binders first and its constraints "
                        f"after them, and the binder {_shown(part.start)} comes "
                        "after a constraint"
                    )
                binders.append(_binder(part, "Where"))
            else:
                constraints.append(part)
        return Polyhedron.build(binders, constraints, "Where")

    def __repr__(self) -> str:
        return "Where"


class _SigmaFamily:
    """The ``Sigma`` name itself: ``Sigma[i: Fin[n], ..., Fin[bound]]``."""

    def __getitem__(self, item: Any) -> Polyhedron:
        """Build the dependent sum; every part but the last is a named binder."""
        parts = item if isinstance(item, tuple) else (item,)
        binders: list[tuple[str, Any]] = []
        shown: list[bool] = []
        for position, part in enumerate(parts):
            last = position == len(parts) - 1
            if isinstance(part, slice):
                binders.append(_binder(part, "Sigma"))
                shown.append(True)
            elif last and isinstance(part, FinType):
                binders.append(("", part.bound))
                shown.append(False)
            elif isinstance(part, FinType):
                raise TypeError(
                    "only the last part of Sigma[...] may be an unnamed fiber, "
                    f"and {part} comes before another part: name it, as in "
                    f"i: {part}"
                )
            else:
                raise TypeError(
                    f"Sigma[...] takes binders and a last fiber, and {_shown(part)} "
                    "is neither; a domain cut by constraints is written "
                    "Where[i: Fin[n], ..., constraint]"
                )
        return Polyhedron.build(binders, (), "Sigma", tuple(shown))

    def __repr__(self) -> str:
        return "Sigma"


#: ``Where[i: Fin[n], j: Fin[n], j < i]``: binders, then the constraints that cut
#: their box. The lower triangle, a band, a restricted box.
Where = _WhereFamily()

#: ``Sigma[i: Fin[n], Fin[i + 1]]``: a dependent sum whose fibers are affine in
#: the binders before them, the lower triangle with its diagonal.
Sigma = _SigmaFamily()


# }}}


# {{{ the domains


@dataclass(frozen=True, eq=False)
class Polyhedron:
    """A set of index tuples: binders with affine bounds, cut by constraints.

    ``binders`` holds one ``(name, bound)`` per axis: the ``k``-th entry of a
    point lies in ``0 <= x_k < bound_k``, and ``bound_k`` may name the binders
    before it (``Fin[i + 1]``) and the sizes. ``constraints`` are the
    comparisons that cut the box, over the binders and the sizes. ``spelling``
    is how it was written, ``"Where"``, ``"Sigma"`` or ``"Fin"`` (a piece of a
    union written ``Fin[n]``), and ``shown`` says which binder names were
    written, since a ``Sigma``'s last fiber is not named and gets a name of
    its own here.

    A bound has to be linear: an integer combination of names plus a constant,
    so that its largest value over the binders before it, which is the box
    the ``"box"`` layout stores, is a term again. A constraint may be
    quasi-affine (``i % 2 == 0``), which only the ``"packed"`` layout refuses.
    """

    binders: tuple[tuple[str, Any], ...]
    constraints: tuple[Any, ...]
    spelling: str = "Where"
    shown: tuple[bool, ...] = ()

    @classmethod
    def build(
        cls,
        binders: Sequence[tuple[str, Any]],
        constraints: Iterable[Any],
        spelling: str,
        shown: tuple[bool, ...] | None = None,
    ) -> Polyhedron:
        """Check a written domain and build it; see :class:`Polyhedron`."""
        written = f"{spelling}[...]"
        if not binders:
            raise TypeError(f"{written} needs at least one binder")
        shown = shown if shown is not None else (True,) * len(binders)
        props: list[Any] = []
        for constraint in constraints:
            props.extend(conjuncts(constraint))
        # An unnamed fiber gets a name nothing else in the domain uses.
        taken = {name for name, _bound in binders if name}
        taken |= _names([bound for _name, bound in binders]) | _names(props)
        named: list[tuple[str, Any]] = []
        for position, (name, bound) in enumerate(binders):
            if not name:
                name = f"_{position}"
                while name in taken:
                    name = f"_{name}"
                taken.add(name)
            named.append((name, bound))
        names = [name for name, _bound in named]
        for position, (name, bound) in enumerate(named):
            if names.count(name) > 1:
                raise TypeError(f"{written} binds {name} twice")
            later = _names(bound) & set(names[position:])
            if later:
                raise TypeError(
                    f"the bound {_shown(bound)} of the binder {name} names "
                    f"{', '.join(sorted(later))}, which is bound at or after it; a "
                    "bound may name the binders before it"
                )
            if not linear(bound):
                raise TypeError(
                    f"the bound {_shown(bound)} of the binder {name} is not linear: "
                    "a bound is an integer combination of sizes and earlier "
                    "binders plus a constant, so that the box it spans is a size "
                    "again. Cut the box with a constraint instead"
                )
        rename = {name: name for name in names}
        for prop in props:
            for side in (prop.left, prop.right):
                try:
                    _isl_text(side, rename)
                except ValueError:
                    raise TypeError(
                        f"the constraint {_shown(prop)} of {written} is not "
                        "quasi-affine, so isl could not state the domain "
                        "exactly: a constraint compares sums of sizes and "
                        "binders times integers, with // and % by an integer"
                    ) from None
        return cls(tuple(named), tuple(props), spelling, shown)

    # {{{ shape

    @property
    def ndim(self) -> int:
        """How many axes the domain has: one per binder."""
        return len(self.binders)

    @property
    def names(self) -> tuple[str, ...]:
        """The binder names, one per axis."""
        return tuple(name for name, _bound in self.binders)

    def size_names(self) -> frozenset[str]:
        """The names the domain mentions that it does not bind: its sizes."""
        mentioned = _names([bound for _name, bound in self.binders])
        mentioned |= _names(self.constraints)
        return frozenset(mentioned - set(self.names))

    def _axis_of(self, prop: Any) -> int:
        """The last axis a constraint names, or ``-1`` when it names none."""
        mentioned = _names(prop)
        axes = [k for k, name in enumerate(self.names) if name in mentioned]
        return max(axes, default=-1)

    def box(self) -> tuple[Any, ...]:
        """The extents of the box the binders span, outermost first, as terms.

        The ``k``-th extent is the largest value of the ``k``-th bound over
        the box of the axes before it, a corner of that box since the bound is
        linear: an earlier binder with a positive coefficient is taken at the
        top of its range and one with a negative coefficient at ``0``. So
        ``Sigma[i: Fin[n], Fin[i + 1]]`` spans ``n x n``. The constraints are
        not consulted: the box is the binders', which is what the ``"box"``
        layout stores and what makes its map affine.
        """
        from lanky.terms import evaluate

        extents: list[Any] = []
        for position, (_name, bound) in enumerate(self.binders):
            corner: dict[str, Any] = {}
            for earlier in range(position):
                name = self.names[earlier]
                if name not in _names(bound):
                    continue
                zero = {n: 0 for n in _names(bound)}
                one = {**zero, name: 1}
                slope = evaluate(bound, one) - evaluate(bound, zero)
                corner[name] = extents[earlier] - 1 if slope > 0 else 0
            extents.append(substitute(bound, corner) if corner else bound)
        return tuple(extents)

    # }}}

    # {{{ isl

    def constraint_texts(
        self, names: Sequence[str], upto: int | None = None
    ) -> tuple[list[str], set[str], set[str]]:
        """The isl text of the domain over dimensions ``names``.

        ``(constraints, params, sizes)``: the constraints of axes ``0`` to
        ``upto`` (every axis by default), the binder bounds and every
        constraint whose last axis is among them, with the binders renamed to
        ``names``; the parameters they mention; and those of the parameters
        that a bound mentions, which are extents and may be assumed
        non-negative (see :func:`loopty.flow.domain_set`).
        """
        last = self.ndim - 1 if upto is None else upto
        rename = dict(zip(self.names, names, strict=False))
        pieces: list[str] = []
        sizes: set[str] = set()
        for k in range(last + 1):
            name, bound = self.binders[k]
            text = _isl_text(bound, rename)
            pieces.append(f"0 <= {names[k]} < {text}")
            sizes |= _names(bound) - set(self.names)
        for prop in self.constraints:
            if self._axis_of(prop) > last:
                continue
            left = _isl_text(prop.left, rename)
            right = _isl_text(prop.right, rename)
            pieces.append(f"{left} {_OPERATORS[prop.operator]} {right}")
        params = set()
        from loopty.flow import free_names

        for piece in pieces:
            params |= free_names(piece)
        params -= set(names)
        return pieces, params, sizes

    def isl_set(
        self, names: Sequence[str] | None = None, upto: int | None = None
    ) -> isl.Set:
        """The exact set of points, over dimensions ``names`` (``a0``, ...).

        ``upto`` keeps the first ``upto + 1`` axes and the constraints on them
        alone, which is the set a loop over ``L.dom[...]`` at that depth runs
        over (see the module docstring).
        """
        count = self.ndim if upto is None else upto + 1
        names = tuple(names) if names is not None else _default_names(count)
        pieces, params, sizes = self.constraint_texts(names, upto)
        return isl.Set(_assembled(names[:count], [(pieces, sizes)], params))

    def prefix_constraints(self, prefix: Sequence[Any], var: Any) -> tuple[Any, ...]:
        """What a loop over the fiber at ``prefix`` is constrained by, as terms.

        The fiber of axis ``len(prefix)``, with ``var`` its loop variable: the
        bounds of the axes before it at ``prefix`` and every constraint on the
        axes up to it, with the binders replaced by the prefix's entries and
        by ``var``. The loop's own bound, :meth:`axis_bound`, is the tracer's.
        A fiber at a prefix outside the domain is therefore empty.
        """
        axis = len(prefix)
        mapping: dict[str, Any] = dict(zip(self.names, prefix, strict=False))
        mapping[self.names[axis]] = var
        out: list[Any] = []
        for k in range(axis):
            bound = substitute(self.binders[k][1], mapping)
            out.append(Comparison(prefix[k], ">=", 0))
            out.append(Comparison(prefix[k], "<", bound))
        for prop in self.constraints:
            if self._axis_of(prop) <= axis:
                out.append(substitute(prop, mapping))
        return tuple(out)

    def axis_bound(self, prefix: Sequence[Any]) -> Any:
        """The bound of axis ``len(prefix)`` at ``prefix``, as a term."""
        axis = len(prefix)
        mapping = dict(zip(self.names, prefix, strict=False))
        return substitute(self.binders[axis][1], mapping)

    def rows_are_intervals(self) -> bool:
        """Whether every row, for every value of the sizes, is an interval.

        A row is the points with given values of every axis but the last; the
        ``"packed"`` layout stores it as a run from its first column, so a
        column the domain skips inside a row (``j % 2 == 0``) cannot be
        stored that way. Asked of isl for every size at once: the points
        between two points of a row have to be points too.
        """
        names = _default_names(self.ndim)
        domain = self.isl_set(names)
        prefix = ", ".join(names[:-1])
        head = f"{prefix}, " if prefix else ""
        last = names[-1]
        params = domain.get_var_names(isl.dim_type.param)
        space = f"[{', '.join(params)}] -> " if params else ""
        up = isl.Map(f"{space}{{ [{head}{last}] -> [{head}x] : x >= {last} }}")
        down = isl.Map(f"{space}{{ [{head}{last}] -> [{head}x] : x <= {last} }}")
        up = up.align_params(domain.get_space())
        down = down.align_params(domain.get_space())
        below = up.intersect_range(domain).domain()
        above = down.intersect_range(domain).domain()
        between = below.intersect(above)
        return bool(between.is_subset(domain))

    # }}}

    def fixed(self, sizes: Mapping[str, int]) -> Fixed:
        """This domain at concrete sizes; see :class:`Fixed`."""
        return Fixed(self, sizes)

    def __add__(self, other: Any) -> SumType:
        """``Where[...] + Fin[m]``: a sum of pieces; see :class:`Union`."""
        if not is_domain(other):
            return NotImplemented
        return SumType.of(self, other)

    def __radd__(self, other: Any) -> SumType:
        """``Fin[m] + Where[...]``: a sum of pieces; see :class:`Union`."""
        if not is_domain(other):
            return NotImplemented
        return SumType.of(other, self)

    def __str__(self) -> str:
        """Print as it was written: ``Where[i: Fin(n), j: Fin(n), j < i]``."""
        if self.spelling == "Fin":
            return f"Fin({render(self.binders[0][1])})"
        parts = []
        for (name, bound), shown in zip(
            self.binders, self.shown or (True,) * self.ndim, strict=True
        ):
            fin = f"Fin({render(bound)})"
            parts.append(f"{name}: {fin}" if shown else fin)
        parts.extend(render(prop) for prop in self.constraints)
        return f"{self.spelling}[{', '.join(parts)}]"

    def __repr__(self) -> str:
        return str(self)

    def __eq__(self, other: Any) -> bool:
        """Compare binders and constraints structurally."""
        return (
            isinstance(other, Polyhedron)
            and self.names == other.names
            and structurally_equal(
                tuple(bound for _name, bound in self.binders),
                tuple(bound for _name, bound in other.binders),
            )
            and structurally_equal(self.constraints, other.constraints)
        )

    def __hash__(self) -> int:
        return hash(("Polyhedron", self.names, len(self.constraints)))


@dataclass(frozen=True, eq=False)
class Union:
    """A disjoint union of pieces, ``Fin[n] + Fin[m]``: points ``(p, x)``.

    The first axis is the piece, a number below ``len(pieces)``, and the
    others are a point of that piece, so every piece has as many axes as the
    others. A body chooses a piece with a Python integer, ``U[0, i]``, and
    iterates the pieces with ``for p in U.dom``, which a trace runs as the
    Python loop it is (see the module docstring).
    """

    pieces: tuple[Polyhedron, ...]

    @property
    def ndim(self) -> int:
        """The piece's axis first, then the axes of a piece."""
        return 1 + self.pieces[0].ndim

    def size_names(self) -> frozenset[str]:
        """The sizes every piece mentions."""
        out: frozenset[str] = frozenset()
        for piece in self.pieces:
            out |= piece.size_names()
        return out

    def isl_set(self, names: Sequence[str] | None = None) -> isl.Set:
        """The exact set of points ``(p, x)``, over dimensions ``names``."""
        names = tuple(names) if names is not None else _default_names(self.ndim)
        disjuncts: list[tuple[list[str], set[str]]] = []
        params: set[str] = set()
        for position, piece in enumerate(self.pieces):
            pieces, mentioned, extents = piece.constraint_texts(names[1:])
            disjuncts.append(([f"{names[0]} = {position}", *pieces], extents))
            params |= mentioned
        return isl.Set(_assembled(names, disjuncts, params))

    def fixed(self, sizes: Mapping[str, int]) -> Fixed:
        """This union at concrete sizes; see :class:`Fixed`."""
        return Fixed(self, sizes)

    def __str__(self) -> str:
        """Print as it was written: ``Fin(n) + Fin(m)``."""
        return " + ".join(str(piece) for piece in self.pieces)

    def __repr__(self) -> str:
        return str(self)

    def __eq__(self, other: Any) -> bool:
        """Compare the pieces in order."""
        return isinstance(other, Union) and len(self.pieces) == len(
            other.pieces
        ) and all(a == b for a, b in zip(self.pieces, other.pieces, strict=True))

    def __hash__(self) -> int:
        return hash(("Union", self.pieces))


def _default_names(count: int) -> tuple[str, ...]:
    """``a0``, ``a1``, ...: the dimension names a cell set uses."""
    return tuple(f"a{k}" for k in range(count))


def _assembled(
    names: Sequence[str],
    disjuncts: Sequence[tuple[Sequence[str], set[str]]],
    params: set[str],
) -> str:
    """The isl text of a union of conjunctions over ``names``.

    Each disjunct comes with the sizes its bounds mention, which are extents
    and are assumed non-negative in it, as :func:`loopty.flow.domain_set`
    assumes them for a loop nest; nothing else is assumed about a parameter.
    """
    head = f"[{', '.join(sorted(params))}] -> " if params else ""
    dims = ", ".join(names)
    parts = []
    for pieces, sizes in disjuncts:
        nonneg = [f"{name} >= 0" for name in sorted(sizes)]
        constraints = " and ".join([*pieces, *nonneg])
        parts.append(f"[{dims}] : {constraints}")
    return f"{head}{{ {'; '.join(parts)} }}"


def is_domain(obj: Any) -> bool:
    """Whether ``obj`` can be read as a domain: see :func:`index_domain`."""
    return isinstance(obj, Polyhedron | Union | FinType | SumType)


def index_domain(obj: Any) -> Polyhedron | Union | None:
    """``obj`` as a domain, or ``None`` when it is not one of the written forms.

    ``Where[...]`` and ``Sigma[...]`` are already :class:`Polyhedron`s; a
    lanky :class:`~lanky.prelude.SumType` becomes a :class:`Union` of its
    pieces, each a ``Where``, a ``Sigma`` or a ``Fin[n]`` (a one-axis piece).
    A bare ``Fin[n]`` is an ordinary axis, not a domain, and gives ``None``.
    """
    if isinstance(obj, Polyhedron):
        return obj
    if isinstance(obj, Union):
        return obj
    if not isinstance(obj, SumType):
        return None
    pieces: list[Polyhedron] = []
    for piece in obj.pieces:
        if isinstance(piece, FinType):
            pieces.append(Polyhedron.build([("", piece.bound)], (), "Fin", (False,)))
        elif isinstance(piece, Polyhedron):
            pieces.append(piece)
        else:
            raise TypeError(
                f"the piece {piece} of {obj} is not a domain: a piece is Fin[n], "
                "Where[...] or Sigma[...]"
            )
    counts = {piece.ndim for piece in pieces}
    if len(counts) > 1:
        raise TypeError(
            f"the pieces of {obj} have {', '.join(str(p.ndim) for p in pieces)} "
            "axes, and a union's pieces have the same number of axes, so that "
            "every point of it is (piece, one index per axis)"
        )
    return Union(tuple(pieces))


# }}}


# {{{ at concrete sizes


class Fixed:
    """A domain at concrete sizes: its points, its fibers, and both layouts.

    What a runtime :class:`~loopty.arr.Arr` over a domain holds. The points of
    every piece are enumerated once, by isl, in lexicographic order, together
    with the fiber of every axis at every prefix (the binder-by-binder reading
    of the module docstring), and the two layouts are computed from them:

    * ``box``: per piece, the binders' box at these sizes
      (:meth:`Polyhedron.box`), and where the piece starts in a flat buffer of
      the pieces one after another.
    * ``packed``: the cells in lexicographic order, piece after piece, and the
      table of row starts, indexed by the piece's position in a table of all
      the pieces' rows, then by the row in the box of the axes but the last.
      An entry is where its row starts less the row's first column, so the
      cell ``(r, j)`` is at ``table[r] + j``; an empty row's entry is where it
      would start.
    """

    def __init__(self, domain: Polyhedron | Union, sizes: Mapping[str, int]) -> None:
        missing = sorted(domain.size_names() - set(sizes))
        if missing:
            raise ValueError(
                f"the domain {domain} names {', '.join(missing)}, and no size "
                "was given for it: pass it by name, as in "
                f"Arr.zeros(domain, {missing[0]}=...)"
            )
        self.domain = domain
        self.sizes = {name: int(sizes[name]) for name in sorted(domain.size_names())}
        self.union = isinstance(domain, Union)
        self.pieces: tuple[Polyhedron, ...] = (
            domain.pieces if isinstance(domain, Union) else (domain,)
        )
        self._points: list[list[tuple[int, ...]]] = []
        self._members: list[set[tuple[int, ...]]] = []
        self._fibers: list[list[dict[tuple[int, ...], list[int]]]] = []
        self.boxes: list[tuple[int, ...]] = []
        for piece in self.pieces:
            points = _points(piece.isl_set(), self.sizes)
            self._points.append(points)
            self._members.append(set(points))
            self._fibers.append(
                [
                    _grouped(_points(piece.isl_set(upto=k), self.sizes))
                    for k in range(piece.ndim)
                ]
            )
            self.boxes.append(
                tuple(max(_integer(extent, self.sizes), 0) for extent in piece.box())
            )
        # Where each piece starts in the flat buffer of either layout.
        volumes = [int(np.prod(box, dtype=np.int64)) for box in self.boxes]
        self.box_bases = _starts(volumes)
        self.counts = [len(points) for points in self._points]
        self.packed_bases = _starts(self.counts)
        self.row_boxes = [box[:-1] for box in self.boxes]
        self.table_bases = _starts(
            [int(np.prod(box, dtype=np.int64)) for box in self.row_boxes]
        )
        self._table: np.ndarray | None = None

    # {{{ points

    @property
    def ndim(self) -> int:
        """The number of axes, the piece's included for a union."""
        return self.domain.ndim

    @property
    def count(self) -> int:
        """How many points the domain has."""
        return sum(self.counts)

    def points(self) -> list[tuple[int, ...]]:
        """Every point in lexicographic order; ``(p, ...)`` for a union."""
        if not self.union:
            return list(self._points[0])
        return [
            (position, *point)
            for position, points in enumerate(self._points)
            for point in points
        ]

    def split(self, key: Sequence[int]) -> tuple[int, tuple[int, ...]]:
        """``(piece, point)`` of a full index tuple, or :class:`IndexError`."""
        if not self.union:
            return 0, tuple(key)
        position = key[0]
        if not 0 <= position < len(self.pieces):
            raise IndexError(
                f"piece {position} out of range for the {len(self.pieces)} pieces "
                f"of {self.domain}"
            )
        return int(position), tuple(key[1:])

    def contains(self, key: Sequence[int]) -> bool:
        """Whether a full index tuple is a point of the domain."""
        try:
            position, point = self.split(key)
        except IndexError:
            return False
        return point in self._members[position]

    def fiber(self, prefix: Sequence[int]) -> Sequence[int]:
        """The values axis ``len(prefix)`` runs over at ``prefix``.

        The pieces' numbers for the first axis of a union; otherwise the
        binder-by-binder fiber (see the module docstring), which is empty at a
        prefix outside the domain.
        """
        if self.union:
            if not prefix:
                return range(len(self.pieces))
            position, rest = self.split(prefix)
        else:
            position, rest = 0, tuple(prefix)
        fibers = self._fibers[position]
        if len(rest) >= len(fibers):
            raise IndexError(
                f"{self.domain} has {self.ndim} axes; there is no axis "
                f"{len(prefix)} to take a fiber of"
            )
        return fibers[len(rest)].get(tuple(rest), [])

    def bound(self, prefix: Sequence[int]) -> int:
        """The bound of axis ``len(prefix)`` at ``prefix``: what ``.size`` is."""
        if self.union:
            if not prefix:
                return len(self.pieces)
            position, rest = self.split(prefix)
        else:
            position, rest = 0, tuple(prefix)
        piece = self.pieces[position]
        if len(rest) >= piece.ndim:
            raise IndexError(
                f"{self.domain} has {self.ndim} axes, not {len(prefix) + 1}"
            )
        return _integer(piece.axis_bound(rest), self.sizes)

    # }}}

    # {{{ layouts

    def address(self, key: Sequence[int], storage: str) -> int | tuple[int, ...]:
        """Where a point is stored: an index of the box, or a flat position.

        A point of a single ``box`` domain is its own index into the array of
        the box's shape; every other address is a position in a flat buffer.
        The point has to be one: :meth:`contains` first.
        """
        position, point = self.split(key)
        if storage == "box":
            if not self.union:
                return point
            box = self.boxes[position]
            return self.box_bases[position] + int(np.ravel_multi_index(point, box))
        table = self.table()
        return int(table[self.table_index(position, point[:-1])]) + point[-1]

    def table_index(self, position: int, row: Sequence[int]) -> int:
        """The entry of the table of row starts for ``row`` of piece ``position``."""
        box = self.row_boxes[position]
        offset = int(np.ravel_multi_index(tuple(row), box)) if box else 0
        return self.table_bases[position] + offset

    def table(self) -> np.ndarray:
        """The table of row starts of the ``packed`` layout; see :class:`Fixed`.

        Raises :class:`ValueError` when a row is not an interval, which only a
        constraint with a remainder or a floor division can make it.
        """
        if self._table is not None:
            return self._table
        total = self.table_bases[-1] + (
            int(np.prod(self.row_boxes[-1], dtype=np.int64)) if self.pieces else 0
        )
        table = np.zeros(max(total, 0), dtype=np.int64)
        for position, points in enumerate(self._points):
            base = self.packed_bases[position]
            rows: dict[tuple[int, ...], list[int]] = {}
            for point in points:
                rows.setdefault(point[:-1], []).append(point[-1])
            cursor = base
            box = self.row_boxes[position]
            for row in (np.ndindex(*box) if box else [()]):
                row = tuple(int(k) for k in row)
                columns = rows.get(row, [])
                entry = self.table_index(position, row)
                if not columns:
                    table[entry] = cursor
                    continue
                first, last = columns[0], columns[-1]
                if last - first + 1 != len(columns):
                    raise ValueError(
                        f"row {row} of {self.pieces[position]} at "
                        f"{_sizes_text(self.sizes)} is {columns}, which is not an "
                        "interval, and the packed layout stores a row as the run "
                        "from its first column to its last; store it boxed"
                    )
                table[entry] = cursor - first
                cursor += len(columns)
        self._table = table
        return table

    def storage_shape(self, storage: str) -> tuple[int, ...]:
        """The shape of the buffer a layout keeps: the box, or a flat length."""
        if storage == "box":
            if not self.union:
                return self.boxes[0]
            return (self.box_bases[-1] + int(np.prod(self.boxes[-1], dtype=np.int64)),)
        return (self.count,)

    def gather(self, buffer: np.ndarray, storage: str) -> np.ndarray:
        """The values at the points, in :meth:`points` order, from a buffer."""
        buffer = np.asarray(buffer)
        if storage == "packed":
            return buffer.reshape(-1)[: self.count]
        out = []
        for key in self.points():
            out.append(buffer[self.address(key, "box")])
        return np.array(out, dtype=buffer.dtype).reshape(-1)

    def scatter(
        self, values: np.ndarray, storage: str, into: np.ndarray | None = None
    ) -> np.ndarray:
        """A buffer of a layout holding ``values`` at the points, in order.

        ``into`` is written in place when it is given, and a new buffer of
        zeros is made when it is not; the cells a layout keeps outside the
        domain are left as they were.
        """
        values = np.asarray(values).reshape(-1)
        if into is None:
            into = np.zeros(self.storage_shape(storage), dtype=values.dtype)
        if storage == "packed":
            into.reshape(-1)[: self.count] = values
            return into
        for key, value in zip(self.points(), values, strict=True):
            into[self.address(key, "box")] = value
        return into

    # }}}

    def isl_points(self) -> isl.Set:
        """The points as an isl set with no parameters: the sizes are fixed."""
        return fixed_set(self.domain, self.sizes)

    def __repr__(self) -> str:
        return f"{self.domain} at {_sizes_text(self.sizes)}"


def fixed_set(domain: Polyhedron | Union, sizes: Mapping[str, int]) -> isl.Set:
    """The points of ``domain`` at ``sizes``, as an isl set with no parameters.

    Two domains written with different names for their sizes or their binders
    are compared this way, point for point.
    """
    points = domain.isl_set()
    count = points.dim(isl.dim_type.param)
    for position, name in enumerate(points.get_var_names(isl.dim_type.param)):
        points = points.fix_val(isl.dim_type.param, position, int(sizes[name]))
    return points.project_out(isl.dim_type.param, 0, count)


def same_points(
    left: isl.Set, right: isl.Set
) -> tuple[bool, tuple[int, ...] | None]:
    """Whether two sets of :func:`fixed_set` are equal, and a point of one only."""
    if left.dim(isl.dim_type.set) != right.dim(isl.dim_type.set):
        return False, None
    if left.is_equal(right):
        return True, None
    point = left.subtract(right).union(right.subtract(left)).sample_point()
    count = left.dim(isl.dim_type.set)
    return False, tuple(
        point.get_coordinate_val(isl.dim_type.set, k).to_python() for k in range(count)
    )


def _sizes_text(sizes: Mapping[str, int]) -> str:
    """``n=4, m=3``, or ``no sizes``."""
    return ", ".join(f"{name}={value}" for name, value in sizes.items()) or "no sizes"


def _integer(expr: Any, sizes: Mapping[str, int]) -> int:
    """A term's value at concrete sizes, as an ``int``."""
    from lanky.terms import evaluate

    value = evaluate(expr, dict(sizes))
    return int(value)


def _starts(lengths: Sequence[int]) -> list[int]:
    """Where each of consecutive runs of these lengths starts."""
    out, cursor = [], 0
    for length in lengths:
        out.append(cursor)
        cursor += int(length)
    return out


def _points(domain: isl.Set, sizes: Mapping[str, int]) -> list[tuple[int, ...]]:
    """The points of a set with its parameters fixed, in lexicographic order."""
    for position, name in enumerate(domain.get_var_names(isl.dim_type.param)):
        domain = domain.fix_val(isl.dim_type.param, position, int(sizes[name]))
    if domain.is_empty():
        return []
    count = domain.dim(isl.dim_type.set)
    found: list[tuple[int, ...]] = []

    def visit(point: Any) -> None:
        found.append(
            tuple(
                point.get_coordinate_val(isl.dim_type.set, k).to_python()
                for k in range(count)
            )
        )

    domain.foreach_point(visit)
    return sorted(found)


def _grouped(points: Sequence[tuple[int, ...]]) -> dict[tuple[int, ...], list[int]]:
    """The last coordinates of points grouped by the others, each list sorted."""
    out: dict[tuple[int, ...], list[int]] = {}
    for point in points:
        out.setdefault(point[:-1], []).append(point[-1])
    return out


# }}}
