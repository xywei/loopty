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
``%`` by numbers. The body sees a cell's indices and nothing else, natively
in a kernel too, so the numbers it computes with are module constants, as
``N`` is here, and ``ne``, which only the annotations name, is not read there.

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
``%``, is the witness that every address has its cell.
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
    return annotation's ``Fin``.
    """

    fn: Any
    names: tuple[str, ...]
    cells: isl.Set
    address: Any
    buffer: Any

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


def _piece(fn: Any) -> Piece:
    """Read one function of a layout: see the module docstring."""
    from lanky.terms import Var, evaluate_annotations

    from loopty.domain import Polyhedron, index_domain
    from loopty.lower import _plain

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
    try:
        address = fn(*args)
    except NameError as exc:
        sizes = set(cells.get_var_names(isl.dim_type.param)) | _sizes(
            buffer.bound, names
        )
        if exc.name in sizes:
            raise LayoutError(
                f"{name}: the address reads the size {exc.name}, which a layout "
                "cannot know: it is called with a cell's indices alone, in a "
                "kernel too; make the sizes it computes with module constants, "
                f"N = 8 and Fin[N] for Fin[{exc.name}]"
            ) from None
        raise LayoutError(
            f"{name}: the address could not be computed on symbolic indices: "
            f"NameError: {exc}"
        ) from None
    except Exception as exc:  # noqa: BLE001 - said in the author's words
        raise LayoutError(
            f"{name}: the address could not be computed on symbolic indices: "
            f"{type(exc).__name__}: {exc}; write it with +, -, * by a number, "
            "// and % by a number"
        ) from None
    if isinstance(address, bool) or not (
        isinstance(address, int) or idx.is_affine(_plain(address))
    ):
        raise LayoutError(
            f"{name}: the address {address} is not a sum of the indices times "
            "numbers, with // and % by numbers, which isl and Lean can state"
        )
    return Piece(fn, names, cells, address, buffer.bound)


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
        relation = self.relation()
        if not relation.is_single_valued():
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

    def __call__(self, *indices: Any) -> Any:
        """The address of a cell, as the layout's one function computes it.

        So a kernel reads through the layout it declares, ``flat[dof_array(e,
        a, b, c)]``, natively and traced. A layout of pieces has a function
        per piece, and each is called as itself.
        """
        if len(self.pieces) != 1:
            raise LayoutError(
                f"{self.name} has pieces, "
                f"{', '.join(piece.label for piece in self.pieces)}; call the "
                "one whose cells you are at"
            )
        return self.pieces[0].fn(*indices)

    def relation(self) -> isl.Map:
        """The layout as one isl map, ``{ [c0, ...] -> [address] }``."""
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

    def _make_facts(self) -> list[Fact]:
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
        key = {"module": self.module, "line": self.line}
        explain = partial(_explained, self)
        questions = [
            (
                "layout-in-bounds",
                f"{owner} stores every cell at an address of {buffer}",
                Subset(
                    flow.assume_sizes(image, sizes),
                    addresses,
                    description=f"the addresses {owner} stores {cells} at are "
                    f"addresses of {buffer}",
                    lean=Stated(partial(layout_bounds_obligation, self)),
                    explain=partial(explain, "in-bounds"),
                ),
            ),
            (
                "layout-one-for-one",
                f"{owner} stores no two cells at one address",
                Empty(
                    flow.assume_sizes(collisions.wrap(), sizes),
                    description=f"pairs of {cells} that {owner} stores at one "
                    "address",
                    labels=(*labels, *(f"{label}'" for label in labels)),
                    lean=Stated(partial(layout_injective_obligation, self)),
                    explain=partial(explain, "one-for-one"),
                ),
            ),
            (
                "layout-onto",
                f"{owner} stores a cell at every address of {buffer}",
                Subset(
                    flow.assume_sizes(addresses, sizes),
                    image,
                    description=f"every address of {buffer} stores one of {cells}",
                    lean=Stated(partial(layout_onto_obligation, self)),
                    explain=partial(explain, "onto"),
                ),
            ),
        ]
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
            for kind, statement, question in questions
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
        from loopty.oracle import Empty

        one = self.first.relation()
        other = self.second.relation()
        one, other = one.align_params(other.get_space()), other.align_params(
            one.get_space()
        )
        apart = one.subtract(other).union(other.subtract(one))
        owner = _owner(self, self.first.pieces[0].fn)
        second = self.second.name
        labels = (*_labels(self.first), "address")
        fact = Fact(
            id=fact_id(
                "layout-same", owner, module=self.module, line=self.line, detail=second
            ),
            kind="layout-same",
            statement=(
                f"{self.first.name} stores every cell where {second} does"
            ),
            term=Empty(
                apart.wrap(),
                description=(
                    f"cells {self.first.name} and {second} store at different "
                    "addresses, given with the address one of the two has for it"
                ),
                labels=labels,
                lean=Stated(partial(layout_same_obligation, self.first, self.second)),
                explain=partial(_explained_apart, self.first, self.second),
            ),
            status=Status.ASSUMED,
            provenance={"layouts": [self.first.name, second]},
            where=self.where,
            owner=owner,
        )
        self._facts = (fact,)
        return self._facts


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
    over ``ne x N x N x N`` is stored at ``N**3 e + N**2 c + N b + a``. The
    extents of every axis but the outermost have to be numbers, as a stride
    that is a size would multiply two unknowns. The layout is of the cells of
    ``of``'s one piece, a box of one ``Fin`` per axis.
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
        if level < len(levels) - 1 and not isinstance(extent, int):
            raise LayoutError(
                f"dim_tags {tags!r}: the axis {piece.names[axis]} of {of.name} has "
                f"the extent {extent}, which is not a number, and the axes nested "
                "outside it would have a stride that multiplies two unknowns"
            )
        stride = stride * extent if level < len(levels) - 1 else stride
    terms = [
        strides[axis] * Var(name) if strides[axis] != 1 else Var(name)
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

