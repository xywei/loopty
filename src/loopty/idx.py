"""Index types as isl objects.

The type of an iteration space is an isl set, and the type of a layout is an isl
map. This module is the translation: it turns a tuple of axis sizes into the isl
set of index tuples, normalizes ``Fin[a*b]`` into ``Fin[a] x Fin[b]`` (``Fin`` is
a semiring homomorphism, so a reshape is the identity on the type), and builds
the affine map from an index tuple to a flat address for dense row-major and
column-major storage, or through an offsets array for ragged (dependent-sum)
storage.

Two facts about isl shape everything here. Its constraints are quasi-affine: a
product of two unknowns, such as an axis of size ``n*m``, cannot be written down.
Normalization removes the common cause by splitting the axis; whatever is left is
*reflected* as a fresh parameter, which is sound (it widens the set) and is the
rule fixed at the start of the design, types are isl objects: non-affine terms
never enter the type language. Second, a stride that is not an integer literal makes
``i*stride`` non-affine, so a dense layout map over symbolic sizes has no isl
representation at all; :meth:`Layout.to_map` says so with
:class:`NonAffineLayout` rather than producing a lie, and the delinearization
helpers give the inverse direction symbolically for the term language.

``Fin`` belongs to ``lanky.prelude`` and is imported from there: an index type
is a lanky object, because the same ``Fin[n]`` appears in a theorem's statement
and in a kernel's annotation and the two have to be the same thing.
"""

from __future__ import annotations

import re
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from typing import Any

import islpy as isl
import pymbolic.primitives as prim
from lanky.prelude import Fin
from pymbolic.mapper.dependency import DependencyMapper

__all__ = [
    "Fin",
    "Layout",
    "NonAffineLayout",
    "RaggedLayout",
    "Reflections",
    "axis_size",
    "delinearize",
    "is_affine",
    "isl_expr",
    "linearize",
    "normalize",
    "size_params",
    "strides_of",
    "to_set",
]


class NonAffineLayout(ValueError):
    """Raised when a layout has no quasi-affine isl representation.

    A dense layout's flat address is ``sum_k i_k * stride_k``. isl can express
    that only when every stride is an integer literal, because ``i_k * stride_k``
    with a symbolic stride is a product of two unknowns. The caller's options are
    to instantiate the sizes, to reason on the index tuple rather than on the
    address, or to reflect the address as an opaque parameter.
    """


Expression = Any
Axis = Any


def axis_size(axis: Axis) -> Any:
    """Return the size of one axis, given ``Fin[n]``, an int, or a term.

    Several attribute names are tried for the index type's size so that this
    works with ``lanky.prelude.Fin`` as well as with the local fallback.
    """
    if isinstance(axis, int | prim.ExpressionNode):
        return axis
    for attr in ("size", "n", "bound"):
        size = getattr(axis, attr, None)
        if size is not None:
            return size
    raise TypeError(f"not an axis: {axis!r}")


def _product_factors(expr: Any) -> tuple[Any, ...]:
    """Flatten nested products into a tuple of factors."""
    if isinstance(expr, prim.Product):
        out: list[Any] = []
        for child in expr.children:
            out.extend(_product_factors(child))
        return tuple(out)
    return (expr,)


def normalize(shape: Sequence[Axis]) -> tuple[Fin, ...]:
    """Split every product axis, so ``Fin[a*b]`` becomes ``Fin[a] x Fin[b]``.

    ``Fin`` is a semiring homomorphism: ``Fin (a*b)`` and ``Fin a x Fin b`` are
    definitionally the same index type. Splitting is what makes a reshape the
    identity on the type, and it is also what keeps sizes inside isl's affine
    fragment, since an axis bound of ``n*m`` is not expressible while two axes
    bounded by ``n`` and by ``m`` are.

    Integer sizes are left alone: ``Fin[6]`` is one axis, not a factorization.
    """
    out: list[Fin] = []
    for axis in shape:
        size = axis_size(axis)
        factors = _product_factors(size)
        if len(factors) == 1:
            out.append(Fin[size])
            continue
        # Fold the numeric factors together; each symbolic factor gets an axis.
        constant = 1
        symbolic: list[Any] = []
        for factor in factors:
            if isinstance(factor, int):
                constant *= factor
            else:
                symbolic.append(factor)
        if constant != 1:
            out.append(Fin[constant])
        out.extend(Fin[f] for f in symbolic)
    return tuple(out)


def is_affine(expr: Any) -> bool:
    """Is ``expr`` quasi-affine, hence writable as an isl constraint?

    Quasi-affine means: integer literals and variables, sums, products in which
    at most one factor is not an integer literal, and floor division or remainder
    by a positive integer literal. Anything else (a product of two unknowns, a
    subscript such as ``cnt[r]``, a call) is not, and must be reflected as a
    fresh parameter before it can appear in a set.
    """
    if isinstance(expr, int):
        return True
    if isinstance(expr, prim.Variable):
        return True
    if isinstance(expr, prim.Sum):
        return all(is_affine(child) for child in expr.children)
    if isinstance(expr, prim.Product):
        non_constant = [c for c in expr.children if not isinstance(c, int)]
        if len(non_constant) > 1:
            return False
        return all(is_affine(child) for child in expr.children)
    if isinstance(expr, prim.FloorDiv | prim.Remainder):
        denominator = expr.denominator
        return (
            isinstance(denominator, int)
            and denominator > 0
            and is_affine(expr.numerator)
        )
    return False


def isl_expr(expr: Any) -> str:
    """Render a quasi-affine term in isl's input syntax.

    pymbolic's own printing is already isl-compatible for sums and products; the
    two cases that differ are floor division (``floord(a, b)``) and remainder,
    which isl writes with explicit parentheses.
    """
    if isinstance(expr, int):
        return str(expr)
    if isinstance(expr, prim.Variable):
        return expr.name
    if isinstance(expr, prim.FloorDiv):
        return f"floord({isl_expr(expr.numerator)}, {isl_expr(expr.denominator)})"
    if isinstance(expr, prim.Remainder):
        return f"(({isl_expr(expr.numerator)}) % {isl_expr(expr.denominator)})"
    if isinstance(expr, prim.Sum):
        return "(" + " + ".join(isl_expr(c) for c in expr.children) + ")"
    if isinstance(expr, prim.Product):
        return "(" + " * ".join(isl_expr(c) for c in expr.children) + ")"
    raise NonAffineLayout(f"cannot render {expr!r} in isl syntax")


def _reflected_name(expr: Any) -> str:
    """The *preferred spelling* of the parameter standing for a non-affine term.

    Readable, and deliberately not unique: every non-word run becomes one
    underscore, so ``cnt[r]`` and ``cnt*r`` both spell ``nl_cnt_r``, and a user
    size may already be called that. Allocation is :class:`Reflections`'s job;
    this only says what the parameter would like to be called.
    """
    stem = re.sub(r"\W+", "_", str(expr)).strip("_") or "x"
    return f"nl_{stem}"


def _structural_key(expr: Any) -> str:
    """The identity of a reflected term: two terms share a parameter iff equal.

    ``str`` is the structural form pymbolic prints, so ``cnt[r]`` and
    ``cnt[r + 1]`` are different keys and the same ``cnt[r]`` built in two
    places is one key. It is *not* the name: the name is derived from this and
    then made unique, which is the difference this table exists to keep.
    """
    return str(expr)


class Reflections:
    """The non-affine terms of one term, and the isl parameter each stands for.

    isl's constraints are quasi-affine, so an array read such as ``cnt[r]`` or a
    product of two unknowns has to become a fresh parameter before it can appear
    in a set. Two properties make that sound and readable, and both need a table
    rather than a function of the term alone:

    *The same term is the same parameter, everywhere.* A statement's domain, a
    reduction's domain and the cell set an in-bounds obligation compares it
    against are three separately built isl objects, and the obligation is
    meaningless unless ``cnt[r]`` is one parameter across all three. Keyed by
    :func:`_structural_key`, which is the term and not its spelling.

    *Different terms are different parameters.* The readable spelling is not
    injective (``cnt[r]`` and ``cnt*r`` both read ``nl_cnt_r``) and may already
    be a size the kernel declares. A name that is taken gets a numeric suffix,
    so two distinct terms are never silently asserted equal and a user's
    ``nl_cnt_r`` keeps its meaning.

    ``reserved`` is every name the parameter must avoid: sizes, parameters,
    inames, and anything else already in the space.
    """

    __slots__ = ("_by_key", "_exprs", "_reserved")

    def __init__(self, reserved: Iterable[str] = ()) -> None:
        self._reserved: set[str] = set(reserved)
        self._by_key: dict[str, str] = {}
        self._exprs: dict[str, Any] = {}

    def reserve(self, names: Iterable[str]) -> None:
        """Forbid these names to future allocations."""
        self._reserved.update(names)

    def adopt(self, name: str, expr: Any) -> str:
        """Record a name that was already allocated for ``expr``."""
        self._by_key[_structural_key(expr)] = name
        self._exprs[name] = expr
        return name

    def symbol(self, expr: Any) -> str:
        """The parameter standing for ``expr``, allocating one if it has none."""
        key = _structural_key(expr)
        existing = self._by_key.get(key)
        if existing is not None:
            return existing
        base = _reflected_name(expr)
        name = base
        suffix = 2
        while name in self._reserved or name in self._exprs:
            name = f"{base}_{suffix}"
            suffix += 1
        return self.adopt(name, expr)

    def items(self) -> tuple[tuple[str, Any], ...]:
        """Every allocated parameter with the term it stands for."""
        return tuple(self._exprs.items())

    @property
    def names(self) -> tuple[str, ...]:
        """Every allocated parameter name, in allocation order."""
        return tuple(self._exprs)

    def get(self, name: str) -> Any:
        """The term parameter ``name`` stands for, or ``None``."""
        return self._exprs.get(name)

    def __contains__(self, name: object) -> bool:
        return name in self._exprs

    def __repr__(self) -> str:
        inner = ", ".join(f"{name}={expr}" for name, expr in self._exprs.items())
        return f"Reflections({inner})"


def size_params(sizes: Iterable[Any]) -> tuple[str, ...]:
    """Free variable names occurring in a collection of sizes, sorted."""
    names: set[str] = set()
    mapper = DependencyMapper()
    for size in sizes:
        if isinstance(size, int):
            continue
        for dep in mapper(size):
            if isinstance(dep, prim.Variable):
                names.add(dep.name)
    return tuple(sorted(names))


def _bound_string(
    size: Any, reflected: dict[str, Any], table: Reflections
) -> str:
    """isl text for an axis bound, reflecting it as a parameter if non-affine."""
    if is_affine(size):
        return isl_expr(size)
    name = table.symbol(size)
    reflected[name] = size
    return name


def to_set(
    shape: Sequence[Axis],
    params: Sequence[str] = (),
    names: Sequence[str] | None = None,
    reflections: Reflections | None = None,
) -> isl.Set:
    """The isl set of index tuples of ``shape``.

    ``shape`` is a tuple of axes (``Fin[n]``, an int, or a term). The result is
    ``[params] -> { [i0, ...] : 0 <= i0 < s0 and ... }``. Parameters are the free
    names in the sizes, plus any extra ``params`` the caller wants in the space
    (a ragged bound, say), plus one fresh parameter per non-affine size.

    A non-affine size is widened, not rejected: the set of index tuples over
    ``Fin[n*m]`` becomes the set over ``Fin[nl_n_m]``, which contains it for
    every instantiation, so an in-bounds proof against the widened set still
    proves in-bounds. Call :func:`normalize` first when the product is a genuine
    reshape and the two factors should become two axes.

    ``reflections`` is the term's :class:`Reflections` table, which is what
    makes the fresh parameter the *same* one another set built for the same
    term uses, and what keeps it clear of the names already in play. Omit it and
    the set gets a table of its own, reserved against the names it can see.
    """
    sizes = [axis_size(axis) for axis in shape]
    if names is None:
        names = tuple(f"i{k}" for k in range(len(sizes)))
    if len(names) != len(sizes):
        raise ValueError(f"{len(names)} names for {len(sizes)} axes")

    table = reflections
    if table is None:
        table = Reflections([*size_params(sizes), *params, *names])
    reflected: dict[str, Any] = {}
    bounds = [_bound_string(size, reflected, table) for size in sizes]
    # Only the sizes that survived as affine text contribute their free names;
    # a reflected size hides its own names behind the fresh parameter.
    affine_sizes = [size for size in sizes if is_affine(size)]

    param_names = list(dict.fromkeys([*size_params(affine_sizes), *params, *reflected]))
    head = f"[{', '.join(param_names)}] -> " if param_names else ""
    if not sizes:
        return isl.Set(f"{head}{{ [] }}")
    dims = ", ".join(names)
    constraints = " and ".join(
        f"0 <= {name} < {bound}" for name, bound in zip(names, bounds, strict=True)
    )
    return isl.Set(f"{head}{{ [{dims}] : {constraints} }}")


def _mul(a: Any, b: Any) -> Any:
    """Multiply, folding the integer cases so strides stay literals when they can."""
    if isinstance(a, int) and isinstance(b, int):
        return a * b
    if a == 1:
        return b
    if b == 1:
        return a
    return a * b


def _floordiv(a: Any, b: Any) -> Any:
    """Floor division, folded when both sides are integer literals."""
    if isinstance(a, int) and isinstance(b, int):
        return a // b
    return prim.FloorDiv(a, b)


def _mod(a: Any, b: Any) -> Any:
    """Remainder, folded when both sides are integer literals."""
    if isinstance(a, int) and isinstance(b, int):
        return a % b
    return prim.Remainder(a, b)


def strides_of(shape: Sequence[Axis], order: str = "C") -> tuple[Any, ...]:
    """Strides of a dense layout, in units of elements.

    ``order`` is ``"C"`` for row-major (last axis contiguous) or ``"F"`` for
    column-major. Strides of a symbolic shape are symbolic terms.
    """
    sizes = [axis_size(axis) for axis in shape]
    strides: list[Any] = [1] * len(sizes)
    if order == "C":
        for k in range(len(sizes) - 2, -1, -1):
            strides[k] = _mul(strides[k + 1], sizes[k + 1])
    elif order == "F":
        for k in range(1, len(sizes)):
            strides[k] = _mul(strides[k - 1], sizes[k - 1])
    else:
        raise ValueError(f"order must be 'C' or 'F', not {order!r}")
    return tuple(strides)


def linearize(indices: Sequence[Any], shape: Sequence[Axis], order: str = "C") -> Any:
    """The flat address of an index tuple under a dense layout."""
    strides = strides_of(shape, order)
    if len(indices) != len(strides):
        raise ValueError(f"{len(indices)} indices for {len(strides)} axes")
    total: Any = 0
    for index, stride in zip(indices, strides, strict=True):
        term = _mul(index, stride)
        total = term if total == 0 else total + term
    return total


def delinearize(flat: Any, shape: Sequence[Axis], order: str = "C") -> tuple[Any, ...]:
    """Recover the index tuple from a flat address.

    The inverse of :func:`linearize`, written with floor division and remainder
    so that it is a term in the same language as the forward direction. For an
    axis whose stride and size are integer literals the result stays inside
    isl's quasi-affine fragment, which is how a flat array viewed as a matrix
    gets its in-bounds facts decided.
    """
    strides = strides_of(shape, order)
    sizes = [axis_size(axis) for axis in shape]
    # The axis with the largest stride needs no remainder: the flat address is
    # already below its extent, and leaving the modulo out keeps the term affine
    # when the other sizes are literals.
    outermost = 0 if order == "C" else len(sizes) - 1
    out: list[Any] = []
    for k, (stride, size) in enumerate(zip(strides, sizes, strict=True)):
        quotient = _floordiv(flat, stride) if stride != 1 else flat
        out.append(quotient if k == outermost else _mod(quotient, size))
    return tuple(out)


@dataclass(frozen=True)
class Layout:
    """A dense layout: the affine map from an index tuple to a flat address.

    ``order`` is ``"C"`` (row-major) or ``"F"`` (column-major). The map is a
    type, not a computation: it says where an index tuple lives, so that two
    views of the same buffer can be compared as maps and a reshape can be seen
    to be the identity.
    """

    shape: tuple[Axis, ...]
    order: str = "C"

    @property
    def strides(self) -> tuple[Any, ...]:
        """Element strides, outermost axis first."""
        return strides_of(self.shape, self.order)

    @property
    def sizes(self) -> tuple[Any, ...]:
        """Axis sizes, outermost first."""
        return tuple(axis_size(axis) for axis in self.shape)

    def flat_index(self, indices: Sequence[Any]) -> Any:
        """The flat address of ``indices`` as a term."""
        return linearize(indices, self.shape, self.order)

    def delinearize(self, flat: Any) -> tuple[Any, ...]:
        """The index tuple of a flat address, as terms."""
        return delinearize(flat, self.shape, self.order)

    def domain(self, params: Sequence[str] = ()) -> isl.Set:
        """The isl set of index tuples this layout is defined on."""
        return to_set(self.shape, params)

    def to_map(self, params: Sequence[str] = ()) -> isl.Map:
        """The isl map from index tuple to flat address.

        Raises :class:`NonAffineLayout` unless every stride is an integer
        literal: ``i * stride`` with a symbolic stride is a product of two
        unknowns and has no isl representation.
        """
        strides = self.strides
        if not all(isinstance(stride, int) for stride in strides):
            raise NonAffineLayout(
                f"layout {self} has symbolic strides {strides}; isl cannot "
                "multiply an index by an unknown. Instantiate the sizes, or "
                "reason on the index tuple instead of the address."
            )
        domain = self.domain(params)
        names = [f"i{k}" for k in range(len(strides))]
        terms = " + ".join(
            f"{stride}*{name}" if stride != 1 else name
            for name, stride in zip(names, strides, strict=True)
        )
        param_names = domain.get_var_names(isl.dim_type.param)
        head = f"[{', '.join(param_names)}] -> " if param_names else ""
        dims = ", ".join(names)
        flat = terms if terms else "0"
        universe = isl.Map(f"{head}{{ [{dims}] -> [{flat}] }}")
        return universe.intersect_domain(domain)


@dataclass(frozen=True)
class RaggedLayout:
    """A ragged layout: row ``r`` starts at ``offsets[r]`` in a flat buffer.

    This is the storage side of a dependent sum: the index type of the second
    axis depends on the first, and the counts are ``offsets[r+1] - offsets[r]``.
    The offsets are concrete here (a tuple of ints of length ``nrows + 1``),
    which is what the runtime :class:`loopty.arr.Arr` has; the parametric
    version, where ``cnt[r]`` is reflected per row, is the tracing side and is
    documented in ``loopty.flow``.
    """

    offsets: tuple[int, ...]
    name: str = "off"

    def __post_init__(self) -> None:
        if len(self.offsets) < 1:
            raise ValueError("offsets must have at least one entry")
        if any(b < a for a, b in zip(self.offsets, self.offsets[1:], strict=False)):
            raise ValueError(f"offsets must be non-decreasing: {self.offsets}")

    @property
    def nrows(self) -> int:
        """Number of rows (the extent of the outer axis)."""
        return len(self.offsets) - 1

    @property
    def counts(self) -> tuple[int, ...]:
        """Per-row counts, the differences of the offsets."""
        return tuple(
            b - a for a, b in zip(self.offsets, self.offsets[1:], strict=False)
        )

    @property
    def total(self) -> int:
        """Number of stored elements."""
        return self.offsets[-1]

    def flat_index(self, row: Any, column: Any) -> Any:
        """The flat address of ``(row, column)`` as a term: ``off[r] + j``."""
        return prim.Subscript(prim.Variable(self.name), row) + column

    def to_set(self) -> isl.Set:
        """The isl set ``{ [r, j] : 0 <= r < nrows and 0 <= j < cnt[r] }``.

        With concrete counts this is a finite union of boxes, which is exactly
        representable; no widening is needed.
        """
        pieces = [
            f"[{r}, j] : 0 <= j < {count}"
            for r, count in enumerate(self.counts)
            if count > 0
        ]
        if not pieces:
            return isl.Set("{ [r, j] : 1 = 0 }")
        return isl.Set("{ " + "; ".join(pieces) + " }")

    def to_map(self) -> isl.Map:
        """The isl map ``{ [r, j] -> [off[r] + j] }`` over the ragged domain."""
        pieces = [
            f"[{r}, j] -> [{self.offsets[r]} + j] : 0 <= j < {count}"
            for r, count in enumerate(self.counts)
            if count > 0
        ]
        if not pieces:
            return isl.Map("{ [r, j] -> [a] : 1 = 0 }")
        return isl.Map("{ " + "; ".join(pieces) + " }")
