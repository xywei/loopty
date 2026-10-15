"""Checked layout maps: where each cell of an array is stored, and what holds of it.

A layout is a map from an array's cells to the addresses of its buffer, and
one that sends two cells to one address, or one that leaves a cell outside
the buffer, is a bug that no run of a kernel through it need show. loopy's
shaped arrays (loopy #1053) take a layout's injectivity as a trusted
contract. Here it is a checked fact, decided by isl and proved by Lean.

A layout is written as what it is, a Python function from a cell's indices
to its address, whose annotations say which cells and which addresses::

    @layout
    def dof_array(e: Fin[ne], a: Fin[N], b: Fin[N], c: Fin[N]) -> Fin[ne * N**3]:
        return N**3 * e + (a + N * b + N**2 * c)

The parameters are the cell's indices, one ``Fin`` per axis, or one
parameter over a domain whose points are the cells, ``cell: Where[a: Fin[2],
b: Fin[Q], a + b < Q]``, unpacked in the body. The return annotation is the
buffer, ``Fin[k]``. The annotations are read as a kernel's are, so the
sizes need no definition; the body is run once on symbolic indices, so the
address is quasi-affine: a sum of the indices times numbers, with ``//`` and
``%`` by numbers. The numbers it computes with are module constants, as
``N`` is here, and ``ne``, which only the annotations name, is not read there.

The body may compute with a size the annotations name, too, and then the
layout is one for every value of it::

    @layout
    def dof_array(e: Fin[ne], i: Fin[n], j: Fin[n], k: Fin[n]) -> Fin[ne * n**3]:
        return n**3 * e + (i + n * j + n**2 * k)

is grudge's DOFArray at every order. Its address multiplies an index by a
size, which isl cannot state: isl decides each fact at a few sizes, ``n``
from 1 to 6, so the fact is ``tested`` there, and Lean proves it for every
``n``. An address that is a sum of the indices times polynomials in the
sizes is read so; a call gives the sizes it computes with by name,
``dof_array(e, i, j, k, n=4)``.

A layout of several pieces, each over its own cells, is ``layout(centre,
derivatives)``: sumpy keeps the value at the centre at address 0 and the two
derivatives of each order after it, which no one formula gives.

Each layout claims three facts, and ``lanky check`` asks them:

* every cell is stored at an address of the buffer (``layout-in-bounds``);
* no two cells are stored at one address (``layout-one-for-one``);
* every address of the buffer stores a cell (``layout-onto``): together, the
  layout is a bijection onto its buffer.

And two layouts of the same cells may be claimed to be one,
``dof_array.same_as("N3,N0,N1,N2")``: every cell is stored at the same address
by both (``layout-same``). A string is loopy's ``dim_tags``, a nesting
level per axis, ``N0`` the fastest, which is how grudge #418 writes the
layout of its outputs; a wrong nesting is refuted with the cell the two put
apart.

Lean states each fact from the functions' own terms and the cells as the
annotations write them; isl's inverse of a layout, in pieces with ``//`` and
``%``, is the witness that every address has its cell. A layout at every
size is stated the same way, its address as the function writes it, and two
layouts the same is then a polynomial identity, which Lean multiplies out.
Where its strides are each the product of the extents nested inside, as
loopy's dim_tags make them (a mixed radix), loopty gives Lean the steps
``omega`` lacks: the products that bound each digit's term, ``0 ≤ n * (n - 1
- j)``, the case split on the first digit two cells differ in, and, for onto,
the digits of an address by division, ``address % n`` and ``address // n %
n``, which is the inverse isl gives at a number.
"""

from __future__ import annotations

import inspect
import os
import re
from collections.abc import Sequence
from dataclasses import dataclass
from functools import partial
from typing import Any

import islpy as isl
import pymbolic.primitives as prim
from lanky.ledger import Fact, Status, fact_id
from lanky.prelude import FinType

from loopty import flow, idx

__all__ = ["Layout", "LayoutError", "SameLayout", "dim_tags", "layout"]


class LayoutError(TypeError):
    """A layout written in a way that cannot be read, with how to write it."""


@dataclass(frozen=True, eq=False)
class Piece:
    """One function of a layout: its cells, its address, its buffer.

    ``names`` name the cell's indices as the function names them, ``cells``
    is the isl set of them, described (:mod:`loopty.description`), and
    ``address`` the term the function returns. ``buffer`` is the bound of the
    return annotation's ``Fin``. ``packed`` says the function takes the cell
    as one tuple, ``cell: Where[...]``. ``sizes`` are the sizes the function
    computes with, which it is run with as themselves (see the module), and
    ``extents`` the bound of each index's ``Fin``, for a box.
    """

    fn: Any
    names: tuple[str, ...]
    cells: isl.Set
    address: Any
    buffer: Any
    packed: bool = False
    sizes: tuple[str, ...] = ()
    extents: tuple[Any, ...] = ()

    @property
    def polynomial(self) -> bool:
        """Whether the address multiplies an index by a size, which isl cannot state."""
        return not _affine(self.address)

    def run(self, args: Sequence[Any], sizes: dict[str, Any] | None = None) -> Any:
        """The function called with ``args``, ``sizes`` the sizes it computes with."""
        fn = _with_sizes(self.fn, sizes or {}) if self.sizes else self.fn
        return fn(*args)

    def stated(
        self, indices: Sequence[Any], got: Any, sizes: dict[str, Any] | None = None
    ) -> str | None:
        """Why ``got``, the function's address at ``indices``, is not the stated one.

        The facts are about :attr:`address`, the term the function returned
        when the layout was declared, and a call runs the function again, so
        anything it computes with besides the indices (a module constant, a
        helper) may have changed since. At a cell of numbers the address is
        compared with the term's value there; at symbolic indices, in a
        traced kernel, the function's term is computed again and compared
        with the stated one, at the ``sizes`` the call gave. ``None`` when
        they agree.
        """
        import numbers

        from lanky.terms import Var, render, structurally_equal
        from pymbolic import evaluate

        from loopty.hypotheses import substitute
        from loopty.lower import _plain

        sizes = dict(sizes or {})
        values = tuple(indices[0]) if self.packed else tuple(indices)
        if len(values) == len(self.names) and all(
            isinstance(value, numbers.Integral)
            for value in (*values, *sizes.values())
        ):
            stated = evaluate(
                _plain(self.address), {**sizes, **dict(zip(self.names, values))}
            )
            if got == stated:
                return None
            cell = ", ".join(f"{n}={v}" for n, v in zip(self.names, values))
            return f"{got} for ({cell}), where its facts say {stated}"
        fresh = tuple(Var(name) for name in self.names)
        now = self.run((fresh,) if self.packed else fresh, sizes)
        address = substitute(self.address, sizes) if sizes else self.address
        if structurally_equal(_plain(now), _plain(address)):
            return None
        return f"{render(now)}, where its facts say {render(address)}"

    def at(self, values: dict[str, int]) -> Piece:
        """This piece at the sizes ``values``: its address, cells and buffer there."""
        return Piece(
            self.fn,
            self.names,
            _without(self.cells, values),
            _folded(self.address, values),
            _folded(self.buffer, values),
            self.packed,
            tuple(name for name in self.sizes if name not in values),
            tuple(_folded(extent, values) for extent in self.extents),
        )

    @property
    def label(self) -> str:
        return getattr(self.fn, "__name__", "the layout")

    def relation(self) -> isl.Map:
        """``{ [c0, c1] -> [address] }`` over the cells, cells positional."""
        relation = flow.access_relation(self.names, self.cells, (self.address,))
        for k in range(len(self.names)):
            relation = relation.set_dim_name(isl.dim_type.in_, k, f"c{k}'")
        return relation


def _sizes(expr: Any, names: Sequence[str]) -> set[str]:
    """The names ``expr`` reads that are not the cell's indices: sizes."""
    from loopty.window import _walk

    return {
        node.name
        for node in _walk(expr)
        if isinstance(node, prim.Variable) and node.name not in names
    }


def _with_sizes(fn: Any, sizes: dict[str, Any]) -> Any:
    """``fn`` with each name of ``sizes`` bound in its globals, a size it computes with.

    A layout's function reads a size as a module would read a constant, so
    it is run in a copy of its module's namespace that has the size: as
    itself, a symbol, when the layout is declared, and as a number when a
    call gives one.
    """
    import types

    bound = types.FunctionType(
        fn.__code__,
        {**fn.__globals__, **sizes},
        fn.__name__,
        fn.__defaults__,
        fn.__closure__,
    )
    bound.__kwdefaults__ = fn.__kwdefaults__
    return bound


def _unit(term: Any) -> bool:
    """Whether ``term`` is the number 1, by type: a term's ``==`` is a claim."""
    return isinstance(term, int) and not isinstance(term, bool) and term == 1


def _affine(expr: Any) -> bool:
    """Whether isl reads ``expr`` as it is: a number, or quasi-affine in its names."""
    from loopty.lower import _plain

    if isinstance(expr, bool):
        return False
    return isinstance(expr, int) or idx.is_affine(_plain(expr))


#: A polynomial: each monomial, its names and their powers in order, to its
#: integer coefficient.
Polynomial = dict[tuple[tuple[str, int], ...], int]


def _times(one: Polynomial, other: Polynomial) -> Polynomial:
    out: Polynomial = {}
    for first, a in one.items():
        for second, b in other.items():
            powers: dict[str, int] = dict(first)
            for name, power in second:
                powers[name] = powers.get(name, 0) + power
            key = tuple(sorted(powers.items()))
            out[key] = out.get(key, 0) + a * b
    return {key: value for key, value in out.items() if value}


def polynomial(expr: Any) -> Polynomial | None:
    """``expr`` multiplied out, a polynomial with integer coefficients, or ``None``.

    Sums, products, natural powers, names and integers; anything else (a
    division, a remainder, a read of an array) is no polynomial.
    """
    from loopty.lower import _plain

    expr = _plain(expr)
    if isinstance(expr, bool):
        return None
    if isinstance(expr, int):
        return {(): expr} if expr else {}
    if isinstance(expr, prim.Variable):
        return {((expr.name, 1),): 1}
    if isinstance(expr, prim.Sum):
        out: Polynomial = {}
        for child in expr.children:
            found = polynomial(child)
            if found is None:
                return None
            for key, value in found.items():
                out[key] = out.get(key, 0) + value
        return {key: value for key, value in out.items() if value}
    if isinstance(expr, prim.Product):
        out = {(): 1}
        for child in expr.children:
            found = polynomial(child)
            if found is None:
                return None
            out = _times(out, found)
        return out
    if isinstance(expr, prim.Power):
        exponent = _plain(expr.exponent)
        if not isinstance(exponent, int) or isinstance(exponent, bool) or exponent < 0:
            return None
        base = polynomial(expr.base)
        if base is None:
            return None
        out = {(): 1}
        for _ in range(exponent):
            out = _times(out, base)
        return out
    return None


def _term(found: Polynomial) -> Any:
    """A polynomial as a term: a sum of its monomials, each a number times its names."""
    from lanky.terms import Var

    terms: list[Any] = []
    for key, value in sorted(found.items()):
        factors: list[Any] = [] if value == 1 else [value]
        for name, power in key:
            factors.extend([Var(name)] * power)
        if not factors:
            terms.append(value)
        elif len(factors) == 1:
            terms.append(factors[0])
        else:
            terms.append(prim.Product(tuple(factors)))
    if not terms:
        return 0
    return terms[0] if len(terms) == 1 else prim.Sum(tuple(terms))


def _folded(expr: Any, values: dict[str, int]) -> Any:
    """``expr`` at the sizes ``values``, multiplied out where it is a polynomial."""
    from loopty.hypotheses import substitute

    if not values or isinstance(expr, int):
        return expr
    found = polynomial(substitute(expr, dict(values)))
    if found is None:
        return substitute(expr, dict(values))
    return _term(found)


def _linear_in(found: Polynomial, names: Sequence[str]) -> bool:
    """Whether each monomial of ``found`` has at most one of ``names``, once."""
    for key in found:
        powers = [power for name, power in key if name in names]
        if len(powers) > 1 or any(power > 1 for power in powers):
            return False
    return True


def _piece(fn: Any) -> Piece:
    """Read one function of a layout: see the module docstring."""
    from lanky.terms import Var, evaluate_annotations

    from loopty.domain import Polyhedron, index_domain

    name = getattr(fn, "__name__", repr(fn))
    if not callable(fn) or not hasattr(fn, "__code__"):
        raise LayoutError(
            f"a layout is a function from a cell's indices to its address, "
            f"def dof_array(e: Fin[ne], a: Fin[N]) -> Fin[N * ne], not {fn!r}"
        )
    annotations = evaluate_annotations(fn)
    params = list(inspect.signature(fn).parameters)
    buffer = annotations.get("return")
    if not isinstance(buffer, FinType):
        raise LayoutError(
            f"{name}: the return annotation says which addresses the layout "
            f"stores its cells at, -> Fin[k], and {name} has "
            f"{'none' if buffer is None else buffer}"
        )
    if len(params) == 1 and index_domain(annotations.get(params[0])) is not None:
        domain = index_domain(annotations[params[0]])
        if not isinstance(domain, Polyhedron):
            raise LayoutError(
                f"{name}: a union of pieces is written as a layout of several "
                "functions, layout(first, second), one over each piece"
            )
        names = tuple(domain.names)
        cells = domain.isl_set(names)
        args: tuple[Any, ...] = (tuple(Var(n) for n in names),)
    else:
        missing = [p for p in params if not isinstance(annotations.get(p), FinType)]
        if missing or not params:
            raise LayoutError(
                f"{name}: each parameter is an index of the cell, annotated with "
                f"its Fin, or the one parameter is over a domain, cell: Where[...]; "
                f"{', '.join(missing) or 'there is none'} is neither"
            )
        names = tuple(params)
        bounds = [annotations[p].bound for p in params]
        cells = flow.domain_set(names, bounds, names=names)
        args = tuple(Var(n) for n in names)
    # the sizes the annotations name, which the body may compute with
    known = set(cells.get_var_names(isl.dim_type.param)) | _sizes(buffer.bound, names)
    sizes: dict[str, Any] = {}
    run = fn
    while True:
        try:
            address = run(*args)
            break
        except NameError as exc:
            if exc.name in known and exc.name not in sizes:
                # a size the body computes with: run it again with the size
                # as itself, so the layout is one for every value of it
                sizes[exc.name] = Var(exc.name)
                run = _with_sizes(fn, sizes)
                continue
            raise LayoutError(
                f"{name}: the address could not be computed on symbolic indices: "
                f"NameError: {exc}"
            ) from None
        except Exception as exc:  # noqa: BLE001 - said in the author's words
            raise LayoutError(
                f"{name}: the address could not be computed on symbolic indices: "
                f"{type(exc).__name__}: {exc}; write it with +, -, * by a number "
                "or a size, // and % by a number"
            ) from None
    found = None
    if not isinstance(address, bool) and not _affine(address):
        found = polynomial(address)
    if isinstance(address, bool) or not (
        _affine(address) or (found is not None and _linear_in(found, names))
    ):
        raise LayoutError(
            f"{name}: the address {address} is not a sum of the indices times "
            "numbers or sizes, with // and % by numbers, which isl and Lean can "
            "state"
        )
    packed = len(params) == 1 and index_domain(annotations.get(params[0])) is not None
    extents = () if packed else tuple(annotations[p].bound for p in params)
    used = tuple(sorted(name for name in sizes if name in _sizes(address, names)))
    return Piece(fn, names, cells, address, buffer.bound, packed, used, extents)


def _owner(obj: Any, fn: Any) -> str:
    """The name ``obj`` has in the module ``fn`` was written in, or ``fn``'s."""
    for key, value in getattr(fn, "__globals__", {}).items():
        if value is obj:
            return key
    return getattr(fn, "__name__", "layout")


def _register(obj: Any) -> None:
    """Put ``obj`` among the claims ``lanky check`` collects from its file."""
    from lanky.plugins import registry

    from loopty.kernel import ensure_registered

    ensure_registered()
    registry.register_object(obj)


@dataclass(frozen=True)
class Level:
    """One digit of a mixed-radix layout (:meth:`Layout.radix`).

    ``name`` is the index, ``extent`` the bound of its ``Fin`` and ``stride``
    the product of the extents nested inside it, as a term: ``1`` for the
    innermost, then ``n``, then ``n * n``.
    """

    name: str
    extent: Any
    stride: Any


class Layout:
    """A layout: functions from cells to addresses, the pieces of one map.

    Made by :func:`layout`. Its facts are in :meth:`facts`; see the module.
    """

    def __init__(self, pieces: Sequence[Piece]) -> None:
        self.pieces = tuple(pieces)
        first = self.pieces[0]
        ranks = {len(piece.names) for piece in self.pieces}
        if len(ranks) != 1:
            raise LayoutError(
                f"the pieces {', '.join(p.label for p in self.pieces)} have cells "
                "of different ranks"
            )
        buffers = {str(piece.buffer) for piece in self.pieces}
        if len(buffers) != 1:
            raise LayoutError(
                f"the pieces {', '.join(p.label for p in self.pieces)} store into "
                f"different buffers, {', '.join(sorted(buffers))}"
            )
        self.rank = len(first.names)
        self.buffer = first.buffer
        self.__wrapped__ = first.fn
        code = first.fn.__code__
        self.line = code.co_firstlineno
        self.where = f"{os.path.basename(code.co_filename)}:{self.line}"
        from lanky.check import module_name

        self.module = (
            module_name(code.co_filename) or getattr(first.fn, "__module__", "") or ""
        )
        if len(self.pieces) > 1 and self.polynomial:
            raise LayoutError(
                f"the pieces {', '.join(p.label for p in self.pieces)} compute "
                "with a size that multiplies an index, which a layout of pieces "
                "cannot; write each address with numbers"
            )
        if not self.polynomial and not self.relation().is_single_valued():
            raise LayoutError(
                f"the pieces {', '.join(p.label for p in self.pieces)} store one "
                "cell at two addresses; pieces that share a cell agree on it"
            )
        self._facts: tuple[Fact, ...] | None = None
        _register(self)

    #: How a layout made of loopy's dim_tags is named (see :func:`dim_tags`).
    _shown: str | None = None

    @property
    def name(self) -> str:
        """The name the layout has in its module, or the dim_tags it is."""
        if self._shown is not None:
            return self._shown
        return _owner(self, self.pieces[0].fn)

    @property
    def polynomial(self) -> bool:
        """Whether a size multiplies an index, in an address or the buffer's bound.

        isl cannot state such a layout at every size, and asks its facts at a
        few (see the module).
        """
        return any(piece.polynomial for piece in self.pieces) or not _affine(
            self.buffer
        )

    @property
    def sizes(self) -> tuple[str, ...]:
        """The sizes the layout's functions compute with, which a call gives by name."""
        return tuple(sorted({name for piece in self.pieces for name in piece.sizes}))

    def __call__(self, *indices: Any, **sizes: Any) -> Any:
        """The address of a cell, as the layout's one function computes it.

        So a kernel reads through the layout it declares, ``flat[dof_array(e,
        a, b, c)]``, natively and traced. A layout of pieces has a function
        per piece, and each is called as itself. A layout that computes with a
        size is given it by name, ``dof_array(e, i, j, k, n=4)``.
        """
        if len(self.pieces) != 1:
            raise LayoutError(
                f"{self.name} has pieces, "
                f"{', '.join(piece.label for piece in self.pieces)}; call the "
                "one whose cells you are at"
            )
        (piece,) = self.pieces
        if set(sizes) != set(piece.sizes):
            wanted = ", ".join(f"{name}=..." for name in piece.sizes)
            if piece.sizes:
                said = (
                    f"{self.name} computes with the size "
                    f"{', '.join(piece.sizes)}, which a call gives by name: "
                    f"{self.name}({', '.join(piece.names)}, {wanted})"
                )
            else:
                said = (
                    f"{self.name} computes with no size, and is called with the "
                    f"cell alone: {self.name}({', '.join(piece.names)})"
                )
            raise LayoutError(said)
        got = piece.run(indices, sizes)
        differs = piece.stated(indices, got, sizes)
        if differs is not None:
            # the facts are about the address as it was computed when the
            # layout was declared, and something it computes with changed
            raise LayoutError(
                f"{self.name} now computes the address {differs}: something it "
                "computes with besides the indices changed after it was "
                "declared; declare the layout after the numbers it reads are set, "
                "and do not change them"
            )
        return got

    def relation(self) -> isl.Map:
        """The layout as one isl map, ``{ [c0, ...] -> [address] }``.

        Raises:
            LayoutError: For a layout whose address multiplies an index by a
                size, which isl cannot state; it has a map at each size,
                :meth:`at`.
        """
        if self.polynomial:
            sizes = ", ".join(f"{name}=4" for name in self.sampled())
            raise LayoutError(
                f"{self.name} multiplies an index by a size, which isl cannot "
                f"state; it has an isl map at each size, {self.name}.at({sizes})"
            )
        out: isl.Map | None = None
        for piece in self.pieces:
            relation = piece.relation()
            if out is None:
                out = relation
                continue
            out = out.align_params(relation.get_space())
            relation = relation.align_params(out.get_space())
            out = out.union(relation)
        assert out is not None
        return out.coalesce()

    def addresses(self) -> isl.Set:
        """The buffer's addresses, ``{ [a] : 0 <= a < k }``, described."""
        return flow.domain_set(("address",), [self.buffer], names=("address",))

    def size_names(self) -> frozenset[str]:
        """Every size the layout's cells, addresses and buffer name."""
        names: set[str] = set(_sizes(self.buffer, ()))
        for piece in self.pieces:
            names |= set(piece.cells.get_var_names(isl.dim_type.param))
            names |= _sizes(piece.address, piece.names)
        return frozenset(names)

    def sampled(self) -> tuple[str, ...]:
        """The sizes isl asks a layout at every size at, a few values each.

        Every size a monomial multiplies an index by, and of a monomial of
        sizes alone, every one but its last: fixed, they leave every address
        and the buffer's bound a sum of the indices and one size times
        numbers, which isl states. ``ne * n**3`` with ``n`` fixed is ``64 *
        ne`` at ``n = 4``. Empty for a layout isl states as it is.
        """
        if not self.polynomial:
            return ()
        indices = set(self.pieces[0].names)
        monomials: list[tuple[str, ...]] = []
        for expr in (self.buffer, *(piece.address for piece in self.pieces)):
            found = polynomial(expr)
            if found is None:
                continue
            for key in found:
                names = tuple(name for name, power in key for _ in range(power))
                if len(names) > 1:
                    monomials.append(names)
        fixed: set[str] = set()
        for names in monomials:
            if indices & set(names):
                fixed |= set(names) - indices
        for names in monomials:
            free = sorted(set(names) - fixed - indices)
            if len(free) > 1 or (free and names.count(free[0]) > 1):
                fixed |= set(free[:-1]) if len(free) > 1 else set(free)
        return tuple(sorted(fixed))

    def radix(self) -> tuple[Level, ...] | None:
        """The layout's digits, innermost first, where it is a mixed radix.

        A box whose address is each index times the product of the extents
        nested inside it, and whose buffer is the product of every extent:
        ``n**3 * e + (i + n * j + n**2 * k)`` over ``ne x n x n x n``, whose
        digits are ``i``, ``j``, ``k`` and ``e``, of strides ``1``, ``n``,
        ``n * n`` and ``n * n * n``. Each is in bounds, one for one and onto
        by the same few steps, which Lean is given (see the module). ``None``
        for any other layout, and for one of pieces.
        """
        if len(self.pieces) != 1:
            return None
        (piece,) = self.pieces
        if piece.packed or len(piece.extents) != len(piece.names):
            return None
        found = polynomial(piece.address)
        if found is None or not _linear_in(found, piece.names):
            return None
        names = piece.names
        coefficients: dict[str, Polynomial] = {name: {} for name in names}
        for key, value in found.items():
            mine = [name for name, _ in key if name in names]
            if len(mine) != 1:
                # a term with no index is an offset, and no mixed radix has one
                return None
            rest = tuple((name, power) for name, power in key if name not in names)
            coefficients[mine[0]][rest] = value
        extents = {
            name: (extent, polynomial(extent))
            for name, extent in zip(names, piece.extents, strict=True)
        }
        if any(found is None for _, found in extents.values()):
            return None
        levels: list[Level] = []
        stride: Polynomial = {(): 1}
        term: Any = 1
        remaining = list(names)
        while remaining:
            matched = [name for name in remaining if coefficients[name] == stride]
            if len(matched) != 1:
                return None
            (name,) = matched
            extent, size = extents[name]
            levels.append(Level(name, extent, term))
            stride = _times(stride, size)
            term = extent if _unit(term) else prim.Product((term, extent))
            remaining.remove(name)
        if polynomial(self.buffer) != stride:
            return None
        return tuple(levels)

    def at(self, **values: int) -> Layout:
        """The layout at the sizes ``values``, which isl states: ``dof_array.at(n=4)``.

        A layout of its own, which ``lanky check`` does not collect: it is how
        isl asks the facts of a layout at every size, at each of a few.
        """
        out = object.__new__(Layout)
        out.pieces = tuple(piece.at(values) for piece in self.pieces)
        out.rank = self.rank
        out.buffer = _folded(self.buffer, values)
        out.__wrapped__ = self.__wrapped__
        out.line = self.line
        out.where = self.where
        out.module = self.module
        out._facts = None
        out._shown = self.name
        return out

    def same_as(self, other: Any) -> SameLayout:
        """The claim that ``other`` stores every cell where this layout does.

        ``other`` is a layout of the same cells, or a string of loopy's
        ``dim_tags``, ``"N3,N0,N1,N2"``: a nesting level per axis, ``N0`` the
        fastest (see :func:`dim_tags`).
        """
        import sys

        if isinstance(other, str):
            other = dim_tags(other, self)
        if not isinstance(other, Layout):
            raise LayoutError(
                f"same_as takes a layout or loopy's dim_tags as a string, "
                f"'N1,N0', not {other!r}"
            )
        # the claim is written where same_as is called, and named by that line
        caller = sys._getframe(1)
        return SameLayout(self, other, line=caller.f_lineno)

    def facts(self) -> tuple[Fact, ...]:
        """The layout's three facts: in bounds, one for one, onto its buffer."""
        if self._facts is None:
            self._facts = tuple(self._make_facts())
        return self._facts

    def _statements(self) -> list[tuple[str, str]]:
        """Each fact's kind and statement, in the order :meth:`facts` has them."""
        owner = self.name
        buffer = f"Fin({self.buffer})"
        return [
            (
                "layout-in-bounds",
                f"{owner} stores every cell at an address of {buffer}",
            ),
            ("layout-one-for-one", f"{owner} stores no two cells at one address"),
            ("layout-onto", f"{owner} stores a cell at every address of {buffer}"),
        ]

    def _questions(self, lean: bool = True) -> list[Any]:
        """isl's question of each fact, in the order :meth:`facts` has them.

        ``lean`` gives each the statement Lean is asked; a layout at a size
        (:meth:`at`) has none of its own, since Lean is asked about the layout
        at every size.
        """
        from loopty.obligations import (
            Stated,
            layout_bounds_obligation,
            layout_injective_obligation,
            layout_onto_obligation,
        )
        from loopty.oracle import Empty, Subset

        owner = self.name
        relation = self.relation()
        addresses = self.addresses()
        image = relation.range()
        image, addresses = (
            image.align_params(addresses.get_space()),
            addresses.align_params(image.get_space()),
        )
        identity = isl.Map.identity(relation.get_space().domain().map_from_set())
        collisions = relation.apply_range(relation.reverse())
        identity = identity.align_params(collisions.get_space())
        collisions = collisions.subtract(identity)
        sizes = set()
        for piece in self.pieces:
            sizes |= set(piece.cells.get_var_names(isl.dim_type.param))
        sizes |= set(addresses.get_var_names(isl.dim_type.param))
        buffer = f"Fin({self.buffer})"
        labels = _labels(self)
        cells = (
            f"the cells of {self.pieces[0].label}"
            if len(self.pieces) == 1
            else "the cells of "
            + " and ".join(piece.label for piece in self.pieces)
        )
        explain = partial(_explained, self)

        def stated(obligation: Any) -> Any:
            return Stated(partial(obligation, self)) if lean else None

        return [
            Subset(
                flow.assume_sizes(image, sizes),
                addresses,
                description=f"the addresses {owner} stores {cells} at are "
                f"addresses of {buffer}",
                lean=stated(layout_bounds_obligation),
                explain=partial(explain, "in-bounds"),
            ),
            Empty(
                flow.assume_sizes(collisions.wrap(), sizes),
                description=f"pairs of {cells} that {owner} stores at one address",
                labels=(*labels, *(f"{label}'" for label in labels)),
                lean=stated(layout_injective_obligation),
                explain=partial(explain, "one-for-one"),
            ),
            Subset(
                flow.assume_sizes(addresses, sizes),
                image,
                description=f"every address of {buffer} stores one of {cells}",
                lean=stated(layout_onto_obligation),
                explain=partial(explain, "onto"),
            ),
        ]

    def _sampled_questions(self) -> list[Any]:
        """Each fact's question for a layout isl cannot state at every size.

        isl asks it at each of a few sizes (:meth:`sampled`), of the layout
        there (:meth:`at`), and Lean about the layout at every size.
        """
        from loopty.obligations import (
            Stated,
            layout_bounds_obligation,
            layout_injective_obligation,
            layout_onto_obligation,
        )
        from loopty.oracle import AtSizes

        sampled = self.sampled()
        obligations = (
            layout_bounds_obligation,
            layout_injective_obligation,
            layout_onto_obligation,
        )
        return [
            AtSizes(
                sizes=sampled,
                samples=_samples(sampled),
                ask=partial(_sampled_question, self, position),
                why=_why_sampled(self, sampled),
                lean=Stated(partial(obligation, self)),
            )
            for position, obligation in enumerate(obligations)
        ]

    def _make_facts(self) -> list[Fact]:
        owner = self.name
        key = {"module": self.module, "line": self.line}
        questions = self._sampled_questions() if self.polynomial else self._questions()
        return [
            Fact(
                id=fact_id(kind, owner, **key),
                kind=kind,
                statement=statement,
                term=question,
                status=Status.ASSUMED,
                provenance={"layout": owner, "pieces": [p.label for p in self.pieces]},
                where=self.where,
                owner=owner,
            )
            for (kind, statement), question in zip(
                self._statements(), questions, strict=True
            )
        ]

    def __repr__(self) -> str:
        return f"layout({', '.join(piece.label for piece in self.pieces)})"


class SameLayout:
    """The claim that two layouts store every cell at the same address."""

    def __init__(
        self, first: Layout, second: Layout, line: int | None = None
    ) -> None:
        if first.rank != second.rank:
            raise LayoutError(
                f"{first.name} has cells of rank {first.rank} and {second.name} of "
                f"rank {second.rank}, so they are no layouts of the same cells"
            )
        self.first = first
        self.second = second
        self.__wrapped__ = first.pieces[0].fn
        self.module = first.module
        self.line = first.line if line is None else line
        self.where = f"{first.where.rsplit(':', 1)[0]}:{self.line}"
        self._facts: tuple[Fact, ...] | None = None
        _register(self)

    def facts(self) -> tuple[Fact, ...]:
        if self._facts is not None:
            return self._facts
        from loopty.obligations import Stated, layout_same_obligation
        from loopty.oracle import AtSizes

        owner = _owner(self, self.first.pieces[0].fn)
        second = self.second.name
        lean = Stated(partial(layout_same_obligation, self.first, self.second))
        if self.first.polynomial or self.second.polynomial:
            sampled = tuple(
                sorted(set(self.first.sampled()) | set(self.second.sampled()))
            )
            question: Any = AtSizes(
                sizes=sampled,
                samples=_samples(sampled),
                ask=partial(_apart_at, self.first, self.second),
                why=_why_sampled(
                    self.first if self.first.polynomial else self.second, sampled
                ),
                lean=lean,
            )
        else:
            question = _apart(self.first, self.second, lean)
        fact = Fact(
            id=fact_id(
                "layout-same", owner, module=self.module, line=self.line, detail=second
            ),
            kind="layout-same",
            statement=(
                f"{self.first.name} stores every cell where {second} does"
            ),
            term=question,
            status=Status.ASSUMED,
            provenance={"layouts": [self.first.name, second]},
            where=self.where,
            owner=owner,
        )
        self._facts = (fact,)
        return self._facts


def _apart(first: Layout, second: Layout, lean: Any = None) -> Any:
    """isl's question whether two layouts store a cell at different addresses."""
    from loopty.oracle import Empty

    one = first.relation()
    other = second.relation()
    one, other = one.align_params(other.get_space()), other.align_params(
        one.get_space()
    )
    apart = one.subtract(other).union(other.subtract(one))
    return Empty(
        apart.wrap(),
        description=(
            f"cells {first.name} and {second.name} store at different "
            "addresses, given with the address one of the two has for it"
        ),
        labels=(*_labels(first), "address"),
        lean=lean,
        explain=partial(_explained_apart, first, second),
    )


def _apart_at(first: Layout, second: Layout, values: dict[str, int]) -> Any:
    """:func:`_apart` of two layouts at the sizes ``values``."""
    return _apart(first.at(**values), second.at(**values))


def _sampled_question(layout: Layout, position: int, values: dict[str, int]) -> Any:
    """isl's question of the layout's fact at ``position``, at the sizes ``values``."""
    return layout.at(**values)._questions(lean=False)[position]


#: The values isl asks a layout at every size at, by how many sizes it fixes:
#: one size from 1 to 6, two from 1 to 3 each, more from 1 to 2.
_SAMPLED = {1: 6, 2: 3}


def _samples(names: Sequence[str]) -> tuple[dict[str, int], ...]:
    """Every point of the sizes ``names`` isl asks a layout at (:data:`_SAMPLED`)."""
    import itertools

    top = _SAMPLED.get(len(names), 2)
    return tuple(
        dict(zip(names, values, strict=True))
        for values in itertools.product(range(1, top + 1), repeat=len(names))
    )


def _why_sampled(layout: Layout, sampled: Sequence[str]) -> str:
    """Why isl asks ``layout`` at a few sizes: what multiplies an index, in words.

    ``n multiplies an index in the address of dof_array, which isl cannot
    state for every n; Lean proves it for every n``.
    """
    said = " and ".join(sampled)
    verb = "multiplies" if len(sampled) == 1 else "multiply"
    return (
        f"{said} {verb} an index in the address of {layout.name}, which isl "
        f"cannot state for every {said}; Lean proves it for every {said}"
    )


def _cell_words(layout: Layout, values: Sequence[int]) -> str:
    """``(a=1, b=3)``: a cell, by the names its first piece gives its indices."""
    names = _labels(layout)
    return "(" + ", ".join(f"{n}={v}" for n, v in zip(names, values, strict=True)) + ")"


def _sizes_words(params: Any) -> str:
    """`` (at n=1)``, the sizes a witness was read off at, or nothing."""
    if not params:
        return ""
    return " (at " + ", ".join(f"{k}={v}" for k, v in sorted(params.items())) + ")"


def _fixed(obj: Any, params: Any) -> Any:
    """``obj`` at the sizes ``params``: its parameters fixed to them."""
    for name, value in (params or {}).items():
        at = obj.find_dim_by_name(isl.dim_type.param, name)
        if at >= 0:
            obj = obj.fix_val(isl.dim_type.param, at, int(value))
    return obj


def _without(obj: Any, values: Any) -> Any:
    """``obj`` at the sizes ``values``, which are then no parameters of it.

    A size left a parameter, fixed, would leave the buffer's addresses at
    every other value of it uncovered by the cells.
    """
    for name, value in values.items():
        at = obj.find_dim_by_name(isl.dim_type.param, name)
        if at >= 0:
            obj = obj.fix_val(isl.dim_type.param, at, int(value))
            obj = obj.project_out(isl.dim_type.param, at, 1)
    return obj


def _address_of(layout: Layout, cell: Sequence[int], params: Any) -> int | None:
    """The address ``layout`` stores ``cell`` at, at the sizes, or ``None``."""
    from loopty.oracle import sample_point

    relation = _fixed(layout.relation(), params)
    for k, value in enumerate(cell):
        relation = relation.fix_val(isl.dim_type.in_, k, int(value))
    point = sample_point(relation.range())
    return None if point is None else int(point[0])


def _cell_at(layout: Layout, address: int, params: Any) -> tuple[int, ...] | None:
    """A cell ``layout`` stores at ``address``, at the sizes, or ``None``."""
    from loopty.oracle import sample_point

    relation = _fixed(layout.relation(), params)
    relation = relation.fix_val(isl.dim_type.out, 0, int(address))
    point = sample_point(relation.domain())
    return None if point is None else tuple(int(v) for v in point)


def _explained(layout: Layout, kind: str, verdict: Any) -> str | None:
    """A refuted fact of ``layout``, in words: the cells and the address.

    ``fold stores (a=1, b=3) and (a=3, b=2) both at 7``; ``wide stores
    (a=3, b=4) at 16, outside Fin(16)``; ``fold stores no cell at 15 of
    Fin(16)``. ``None`` where the witness cannot be read so.
    """
    witness = verdict.witness
    params = dict(verdict.parameters or {})
    at = _sizes_words(params)
    buffer = f"Fin({layout.buffer})"
    name = layout.name
    try:
        if kind == "one-for-one":
            rank = layout.rank
            one, other = witness[:rank], witness[rank : 2 * rank]
            address = _address_of(layout, one, params)
            where = "one address" if address is None else address
            return (
                f"{name} stores {_cell_words(layout, one)} and "
                f"{_cell_words(layout, other)} both at {where}{at}"
            )
        (address,) = witness
        if kind == "onto":
            return f"{name} stores no cell at {address} of {buffer}{at}"
        cell = _cell_at(layout, address, params)
        which = "a cell" if cell is None else _cell_words(layout, cell)
        return f"{name} stores {which} at {address}, outside {buffer}{at}"
    except (TypeError, ValueError, isl.Error):
        return None


def _explained_apart(first: Layout, second: Layout, verdict: Any) -> str | None:
    """A refuted ``same_as``, in words: the cell, and the address each gives it.

    ``dof_array stores (e=0, a=1, b=0, c=2) at 19, and dim_tags N3,N2,N1,N0
    at 11``.
    """
    params = dict(verdict.parameters or {})
    try:
        cell = tuple(verdict.witness[: first.rank])
        mine = _address_of(first, cell, params)
        theirs = _address_of(second, cell, params)
    except (TypeError, ValueError, isl.Error):
        return None
    if mine is None and theirs is None:
        return None
    cell_text = _cell_words(first, cell)
    if mine is None:
        said = f"{second.name} stores {cell_text} at {theirs}, and {first.name} none"
    elif theirs is None:
        said = f"{first.name} stores {cell_text} at {mine}, and {second.name} none"
    else:
        said = (
            f"{first.name} stores {cell_text} at {mine}, and {second.name} at "
            f"{theirs}"
        )
    return said + _sizes_words(params)


def _labels(layout: Layout) -> tuple[str, ...]:
    """The names a witness's cell is read with: the first piece's indices."""
    return tuple(layout.pieces[0].names)


def layout(*fns: Any) -> Layout:
    """A layout of one function, or of several, each over its own cells.

    As a decorator, ``@layout`` over one function; as a call,
    ``layout(centre, derivatives)``, a layout of pieces. See the module.
    """
    if not fns:
        raise LayoutError("layout() takes the functions of the layout")
    return Layout([_piece(fn) for fn in fns])


_TAG = re.compile(r"^N(\d+)$")


def dim_tags(tags: str, of: Layout) -> Layout:
    """The layout loopy's ``dim_tags`` give the cells of ``of``.

    ``tags`` has one nesting level per axis, ``"N3,N0,N1,N2"``: the axis of
    ``N0`` is contiguous, and each next level's stride is the product of the
    extents of the levels below it. So ``(e, a, b, c)`` with ``N3,N0,N1,N2``
    over ``ne x N x N x N`` is stored at ``N**3 e + N**2 c + N b + a``. An
    extent that is a size gives the strides as loopy's rule writes them, the
    product of the extents nested inside: ``n * n * n * e + i + n * j + n * n
    * k`` over ``ne x n x n x n``, the layout at every ``n``. The layout is of
    the cells of ``of``'s one piece, a box of one ``Fin`` per axis.
    """
    from lanky.terms import Var

    parts = [part.strip() for part in tags.split(",")]
    levels = []
    for part in parts:
        found = _TAG.match(part)
        if found is None:
            raise LayoutError(
                f"dim_tags {tags!r}: {part!r} is no nesting level; loopy writes "
                "one per axis, N0 the fastest, as 'N3,N0,N1,N2'"
            )
        levels.append(int(found.group(1)))
    if len(of.pieces) != 1:
        raise LayoutError(
            f"dim_tags {tags!r} lay out a box, and {of.name} has pieces"
        )
    (piece,) = of.pieces
    if len(levels) != len(piece.names) or sorted(levels) != list(range(len(levels))):
        raise LayoutError(
            f"dim_tags {tags!r} give {len(levels)} levels for the {len(piece.names)} "
            f"axes of {of.name}, and loopy wants one per axis, N0 to "
            f"N{len(piece.names) - 1}, each once"
        )
    from lanky.terms import evaluate_annotations

    evaluated = evaluate_annotations(piece.fn)
    if set(evaluated) - {"return"} != set(piece.names):
        raise LayoutError(
            f"dim_tags {tags!r} lay out a box, one Fin per axis, and the cells "
            f"of {of.name} are a domain"
        )
    extents = [evaluated[name].bound for name in piece.names]
    stride: Any = 1
    strides: dict[int, Any] = {}
    for level in range(len(levels)):
        axis = levels.index(level)
        strides[axis] = stride
        extent = extents[axis]
        if level < len(levels) - 1:
            # the next stride is the product of the extents nested inside it
            stride = extent if _unit(stride) else stride * extent
    terms = [
        Var(name) if _unit(strides[axis]) else strides[axis] * Var(name)
        for axis, name in enumerate(piece.names)
    ]
    address: Any = terms[0]
    for term in terms[1:]:
        address = address + term
    shown = f"dim_tags {tags}"
    tagged = Piece(
        _Named(piece.fn, shown),
        piece.names,
        piece.cells,
        address,
        piece.buffer,
        sizes=tuple(sorted(_sizes(address, piece.names))),
        extents=tuple(extents),
    )
    out = object.__new__(Layout)
    out.pieces = (tagged,)
    out.rank = len(piece.names)
    out.buffer = piece.buffer
    out.__wrapped__ = piece.fn
    out.line = of.line
    out.where = of.where
    out.module = of.module
    out._facts = ()
    out._shown = shown
    return out


class _Named:
    """A layout's function under another name: ``dim_tags N3,N0,N1,N2``."""

    def __init__(self, fn: Any, name: str) -> None:
        self.fn = fn
        self.__name__ = name
        self.__code__ = fn.__code__
        self.__globals__ = {}

