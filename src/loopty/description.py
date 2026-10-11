"""loopty's own description of a set it hands isl, kept on the set.

The design idea. Every isl set loopty builds about a kernel is parsed from text
loopty wrote: the names of the loops, the bound of each, the conjuncts of a
guard, the constraints of a polyhedral domain, and which sizes are taken to be
non-negative. isl keeps what it parsed in a form of its own, simplified and
coalesced, and a set isl computes from others (an image, a projection, a
dependence relation) has no text behind it at all.

A fact Lean checks (:mod:`loopty.obligations`) is stated from the first and
never from the second. That is what makes Lean's proof independent of isl: a
set isl simplified wrong, or a relation it composed wrong, would otherwise be
the very statement Lean proves, and the proof would confirm the mistake. So
loopty keeps its text, and the names the text uses, on the set isl made of it
(an isl object holds Python attributes): :func:`describe` puts it there and
:func:`description_of` reads it back. A set with none was computed, and a
fact over it is not stated for Lean, which says so.

A :class:`Description` is a union of conjunctions, the shape of the text
loopty writes (:func:`loopty.flow.domain_set`,
:meth:`loopty.domain.Polyhedron.isl_set`):

- ``dims`` name the set's dimensions in the text, and ``names`` are what the
  term that made the set calls the same positions, which the parameters a
  bound reflects are written in (``cnt[r]`` for a loop ``r`` over ``d0``);
- each :class:`Disjunct` holds its constraints, a ragged bound
  ``0 <= j < nl_cnt_r`` among them, and the parameters it takes to be
  non-negative;
- ``reflected`` gives the term each reflected parameter stands for
  (``nl_cnt_r`` is ``cnt[r]``), which Lean reads as the array cell it is,
  an atom for ``omega`` as the parameter is one for isl;
- ``renamings`` are the substitutions a program's calls made of the names
  (:func:`loopty.compose.rename_set`), applied in order.

:meth:`Description.predicate` reads it back as a lanky proposition over the
coordinates it is given: the constraint text is parsed, here, by a reader of
the isl syntax loopty writes, and a person writes for
:meth:`loopty.schedule.Schedule.affine` (chained comparisons, ``and``, ``or``,
sums, integer multiples, ``2 i`` as well as ``2 * i``, ``floord(e, k)``,
``floor(e / k)`` and ``e % k``), and anything else is refused, which leaves the
fact to isl with the reason.
"""

from __future__ import annotations

import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field, replace
from typing import Any

import pymbolic.primitives as prim
from lanky.terms import (
    Add,
    Comparison,
    FloorDiv,
    LogicalAnd,
    LogicalOr,
    Product,
    Remainder,
    Var,
    init_args,
)

__all__ = [
    "Description",
    "Disjunct",
    "MapText",
    "Unstatable",
    "conjunction",
    "describe",
    "description_of",
    "disjunction",
    "map_text",
    "map_text_of",
    "parse",
    "substitute",
]

#: The attribute an isl set carries its description under.
_ATTRIBUTE = "_loopty_description"


class Unstatable(ValueError):
    """A fact loopty cannot state for Lean from its own description, and why."""


@dataclass(frozen=True)
class Disjunct:
    """One conjunction of a set's text: its constraints and non-negative parameters."""

    constraints: tuple[str, ...]
    nonneg: tuple[str, ...] = ()


@dataclass(frozen=True)
class Description:
    """A set as loopty wrote it for isl; see the module docstring."""

    dims: tuple[str, ...]
    disjuncts: tuple[Disjunct, ...]
    names: tuple[str, ...] = ()
    reflected: tuple[tuple[str, Any], ...] = ()
    renamings: tuple[tuple[tuple[str, Any], ...], ...] = field(default=())

    def __post_init__(self) -> None:
        if not self.names:
            object.__setattr__(self, "names", tuple(self.dims))
        if len(self.names) != len(self.dims):
            raise ValueError(f"{len(self.names)} names for {len(self.dims)} dimensions")

    # {{{ derived descriptions

    def renamed(
        self, names: Mapping[str, str], exprs: Mapping[str, Any]
    ) -> Description:
        """This set after a call's renaming: names to names, parameters to expressions.

        All at once, as :func:`loopty.names.rename` renames the set. The text
        keeps the names it was written with, and the renaming is applied
        when the description is read (:meth:`predicate`), after the
        dimensions, which are read by position.
        """
        mapping: dict[str, Any] = {old: Var(new) for old, new in names.items()}
        mapping.update(exprs)
        keyed = tuple(sorted(mapping.items(), key=lambda item: item[0]))
        return replace(self, renamings=(*self.renamings, keyed))

    def outside(self, keep: int) -> Description:
        """The set over its first ``keep`` dimensions, whatever the loops inside do.

        :func:`loopty.flow.loops_outside`, on the text: a constraint that
        mentions a later dimension is dropped, and the others kept. A
        non-negative parameter is kept with the constraints that mention it,
        since the assumption is the set's whatever its dimensions.
        """
        if keep >= len(self.dims):
            return self
        inner = set(self.dims[keep:])
        disjuncts = tuple(
            Disjunct(
                tuple(
                    text
                    for text in disjunct.constraints
                    if not (_names(text) & inner)
                ),
                disjunct.nonneg,
            )
            for disjunct in self.disjuncts
        )
        return replace(
            self,
            dims=self.dims[:keep],
            names=self.names[:keep],
            disjuncts=disjuncts,
        )

    # }}}

    # {{{ reading it back

    def predicate(
        self, coords: Sequence[Any], others: Mapping[str, Any] | None = None
    ) -> Any:
        """The set as a lanky proposition: that ``coords`` is one of its points.

        ``coords`` are terms, one per dimension, put in as they are. ``others``
        gives a term for a name the set mentions that is no dimension of its
        own: a ragged cell set's ``cnt[r]`` is written in the loops of the
        access that reaches it, and ``r`` there is that access's ``r``. A name
        in neither stays a variable of its own name, a size or a scalar. A
        call's renamings (:meth:`renamed`) are applied before either.

        Raises:
            Unstatable: If a constraint is not text this reader takes.
        """
        if len(coords) != len(self.dims):
            raise Unstatable(
                f"{len(coords)} coordinates for a set of {len(self.dims)} dimensions"
            )
        # placeholders called like nothing the set, its renamings or what is
        # put in for it mention, so that none stands for a name of the kernel
        taken: set[str] = set(self.dims) | set(self.names)
        for disjunct in self.disjuncts:
            for text in disjunct.constraints:
                taken |= _names(text)
            taken |= set(disjunct.nonneg)
        taken |= _variables(tuple(expr for _, expr in self.reflected))
        for keyed in self.renamings:
            taken |= {old for old, _ in keyed}
            taken |= _variables(tuple(new for _, new in keyed))
        if others:
            taken |= set(others) | _variables(tuple(others.values()))
        taken |= _variables(tuple(coords))
        stem = "__loopty_dim"
        while any(name.startswith(stem) for name in taken):
            stem += "_"
        placeholders = {dim: Var(f"{stem}{k}") for k, dim in enumerate(self.dims)}
        by_name = {
            name: placeholders[dim]
            for name, dim in zip(self.names, self.dims, strict=True)
        }
        reflected = dict(self.reflected)

        def resolve(name: str) -> Any:
            if name in placeholders:
                return placeholders[name]
            if name in reflected:
                return substitute(reflected[name], by_name)
            return Var(name)

        pieces = []
        for disjunct in self.disjuncts:
            parts = [parse(text, resolve) for text in disjunct.constraints]
            parts += [Comparison(resolve(name), ">=", 0) for name in disjunct.nonneg]
            pieces.append(conjunction(parts))
        out = disjunction(pieces)
        for keyed in self.renamings:
            out = substitute(out, dict(keyed))
        if others:
            out = substitute(out, dict(others))
        values = {
            placeholder.name: coord
            for placeholder, coord in zip(placeholders.values(), coords, strict=True)
        }
        return substitute(out, values)

    # }}}


def substitute(expr: Any, mapping: Mapping[str, Any]) -> Any:
    """``expr`` with every variable named in ``mapping`` replaced, by name.

    By name and not by the variable: a lanky variable's ``==`` builds a
    proposition, so a table keyed by variables, as pymbolic's substitution
    keeps one, would ask a term for its truth value. Nodes are rebuilt from
    their constructor arguments, so lanky's classes stay lanky's; a
    description holds no quantifier, so no name is bound below one.
    """
    if not mapping:
        return expr
    if isinstance(expr, prim.Variable):
        return mapping.get(expr.name, expr)
    if isinstance(expr, prim.ExpressionNode):
        args = init_args(expr)
        new = tuple(substitute(arg, mapping) for arg in args)
        if all(old is changed for old, changed in zip(args, new, strict=True)):
            return expr
        return type(expr)(*new)
    if isinstance(expr, tuple):
        return tuple(substitute(item, mapping) for item in expr)
    return expr


@dataclass(frozen=True)
class MapText:
    """A step's map as loopy's checker was given it: loops in, loops out, constraints.

    The constraints name the loops it takes ``a0, a1, ...`` and the loops it
    makes ``b0, b1, ...``, in the order of ``inputs`` and ``outputs``, as
    :func:`loopty.schedule._reindexing` writes them, so that a new loop may
    keep the name of the one it replaces; any other name is a size. ``stems``
    are the two prefixes, ``a`` and ``b`` unless a size of the text is called
    like one of those names (``a0``), when :func:`map_text` picks others.
    """

    inputs: tuple[str, ...]
    outputs: tuple[str, ...]
    constraints: tuple[str, ...]
    stems: tuple[str, str] = ("a", "b")

    def relation(self, before: Mapping[str, Any], after: Mapping[str, Any]) -> Any:
        """The map as a proposition: the loops ``before`` go to the loops ``after``.

        ``before`` and ``after`` give a term for each loop by name.
        """
        into, out_of = self.stems
        table = {f"{into}{k}": before[name] for k, name in enumerate(self.inputs)}
        table |= {f"{out_of}{k}": after[name] for k, name in enumerate(self.outputs)}

        def resolve(name: str) -> Any:
            return table.get(name, Var(name))

        return conjunction([parse(text, resolve) for text in self.constraints])


def map_text(text: str) -> dict[str | None, MapText]:
    """The maps of an isl map's text, by the statement each names (``None`` for none).

    For :meth:`loopty.schedule.Schedule.affine` given text:
    ``{ S0[t, i] -> [a, b] : a = t + i and b = t - i; ... }``, with the sizes
    declared before it, ``[n] -> { ... }``. A loop named on both sides keeps
    its value, as isl reads a name written twice. A piece whose output is not
    a list of names, or that this reader cannot split, is refused.

    Several pieces for one statement are one piecewise map, as isl reads them,
    and keep every piece: their constraints are one disjunction, a piece's
    conjunction each. Pieces of one statement that name its loops
    differently are refused, since a step matches loops by name.

    Raises:
        Unstatable: For text of another shape.
    """
    body = text.strip()
    head = re.match(r"^\[[^\]]*\]\s*->\s*", body)
    if head:
        body = body[head.end() :]
    if not (body.startswith("{") and body.endswith("}")):
        raise Unstatable(f"not the text of a map: {text}")
    # the names the coordinates are written with, apart from every name of
    # the text: a size called a0 stays the size (Codex)
    names = set(re.findall(r"[A-Za-z_][A-Za-z_0-9']*", text))
    into = _fresh_stem("a", names)
    out_of = _fresh_stem("b", names | {into})
    found_pieces: dict[str | None, list[MapText]] = {}
    for piece in body[1:-1].split(";"):
        found = re.match(
            r"^\s*([A-Za-z_][A-Za-z_0-9]*)?\s*\[([^\]]*)\]\s*->\s*\[([^\]]*)\]\s*"
            r"(?::(.*))?$",
            piece.strip(),
            re.S,
        )
        if not found:
            raise Unstatable(f"cannot read the map piece {piece.strip()!r}")
        name = found.group(1)
        ins = tuple(x.strip() for x in found.group(2).split(",") if x.strip())
        outs = tuple(x.strip() for x in found.group(3).split(",") if x.strip())
        pattern = r"[A-Za-z_][A-Za-z_0-9']*"
        if not all(re.fullmatch(pattern, x) for x in (*ins, *outs)):
            raise Unstatable(f"a map piece whose loops are not all names: {piece!r}")
        table = {x: f"{into}{k}" for k, x in enumerate(ins)}
        written = (found.group(4) or "").strip()
        constraints: list[str] = []
        for k, x in enumerate(outs):
            if x in table:
                constraints.append(f"{out_of}{k} = {table[x]}")
        renamed = {**table, **{x: f"{out_of}{k}" for k, x in enumerate(outs)}}
        if written:
            words = []
            for token in _tokens(written):
                words.append(renamed.get(token, token))
            constraints.append(" ".join(words))
        found_pieces.setdefault(name, []).append(
            MapText(ins, outs, tuple(constraints), (into, out_of))
        )
    out: dict[str | None, MapText] = {}
    for name, pieces in found_pieces.items():
        first = pieces[0]
        if len(pieces) == 1:
            out[name] = first
            continue
        if any((p.inputs, p.outputs) != (first.inputs, first.outputs) for p in pieces):
            raise Unstatable(
                f"the pieces of the map of {name or 'every statement'} name its loops "
                "differently"
            )
        either = " or ".join(
            "(" + (" and ".join(f"({c})" for c in p.constraints) or "0 = 0") + ")"
            for p in pieces
        )
        out[name] = MapText(first.inputs, first.outputs, (either,), first.stems)
    return out


def _fresh_stem(stem: str, names: set[str]) -> str:
    """``stem``, or ``stem_``, ``stem__``, ...: one no ``stem<k>`` of ``names`` uses."""
    while any(re.fullmatch(re.escape(stem) + r"\d+", name) for name in names):
        stem += "_"
    return stem


def _variables(expr: Any) -> set[str]:
    """The names of the variables of a term."""
    if isinstance(expr, prim.Variable):
        return {expr.name}
    if isinstance(expr, prim.ExpressionNode):
        out: set[str] = set()
        for arg in init_args(expr):
            out |= _variables(arg)
        return out
    if isinstance(expr, tuple | list):
        out = set()
        for item in expr:
            out |= _variables(item)
        return out
    return set()


def map_text_of(obj: Any) -> MapText | None:
    """The text loopty kept on the isl map ``obj``, or ``None`` for one isl computed."""
    found = getattr(obj, _ATTRIBUTE, None)
    return found if isinstance(found, MapText) else None


def describe(obj: Any, description: Description | MapText) -> Any:
    """Keep ``description`` on the isl set or map ``obj``, and return ``obj``."""
    try:
        setattr(obj, _ATTRIBUTE, description)
    except (AttributeError, TypeError):  # pragma: no cover - an isl without __dict__
        pass
    return obj


def description_of(obj: Any) -> Description | None:
    """The description loopty kept on ``obj``, or ``None`` for a set isl computed."""
    found = getattr(obj, _ATTRIBUTE, None)
    return found if isinstance(found, Description) else None


def conjunction(parts: Sequence[Any]) -> Any:
    """``a & b & ...`` as one term; ``True`` for none."""
    parts = [part for part in parts if part is not True]
    if not parts:
        return True
    if len(parts) == 1:
        return parts[0]
    return LogicalAnd(tuple(parts))


def disjunction(parts: Sequence[Any]) -> Any:
    """``a | b | ...`` as one term; ``False`` for none."""
    parts = [part for part in parts if part is not False]
    if not parts:
        return False
    if len(parts) == 1:
        return parts[0]
    return LogicalOr(tuple(parts))


# {{{ reading isl's syntax, as loopty writes it

_TOKEN = re.compile(
    r"\s*(<=|>=|==|!=|=|<|>|\+|-|\*|%|/|\(|\)|,|[A-Za-z_][A-Za-z_0-9']*|\d+)"
)
_RELATIONS = {"<": "<", "<=": "<=", ">": ">", ">=": ">=", "=": "==", "==": "=="}
_WORDS = frozenset(
    {"and", "or", "not", "mod", "floord", "floor", "exists", "min", "max"}
)


def _tokens(text: str) -> list[str]:
    out: list[str] = []
    position = 0
    text = text.strip()
    while position < len(text):
        found = _TOKEN.match(text, position)
        if not found or found.end() == position:
            raise Unstatable(f"cannot read {text[position:]!r} in {text!r}")
        out.append(found.group(1))
        position = found.end()
    return out


def _names(text: str) -> set[str]:
    """The names a constraint mentions."""
    return {
        token
        for token in _tokens(text)
        if re.fullmatch(r"[A-Za-z_][A-Za-z_0-9']*", token) and token not in _WORDS
    }


class _Reader:
    """A recursive descent over one constraint, building lanky terms."""

    def __init__(self, text: str, resolve: Any) -> None:
        self.text = text
        self.tokens = _tokens(text)
        self.position = 0
        self.resolve = resolve

    def peek(self) -> str | None:
        return self.tokens[self.position] if self.position < len(self.tokens) else None

    def take(self, wanted: str | None = None) -> str:
        token = self.peek()
        if token is None or (wanted is not None and token != wanted):
            raise Unstatable(f"expected {wanted or 'more'} in {self.text!r}")
        self.position += 1
        return token

    def done(self) -> None:
        if self.peek() is not None:
            rest = " ".join(self.tokens[self.position :])
            raise Unstatable(f"cannot read {rest!r} in {self.text!r}")

    # propositions

    def disjunction(self) -> Any:
        parts = [self.conjunction()]
        while self.peek() == "or":
            self.take()
            parts.append(self.conjunction())
        return disjunction(parts)

    def conjunction(self) -> Any:
        parts = [self.atom()]
        while self.peek() == "and":
            self.take()
            parts.append(self.atom())
        return conjunction(parts)

    def atom(self) -> Any:
        if self.peek() in ("not", "exists"):
            raise Unstatable(f"{self.peek()!r} in {self.text!r}")
        if self.peek() == "(":
            # a bracketed proposition, or the bracketed expression of a comparison
            saved = self.position
            self.take("(")
            try:
                inner = self.disjunction()
                self.take(")")
                if self.peek() not in _RELATIONS:
                    return inner
            except Unstatable:
                pass
            self.position = saved
        return self.comparison()

    def comparison(self) -> Any:
        left = self.expression()
        parts = []
        while self.peek() in _RELATIONS:
            operator = _RELATIONS[self.take()]
            right = self.expression()
            parts.append(Comparison(left, operator, right))
            left = right
        if not parts:
            raise Unstatable(f"{self.text!r} is not a comparison")
        return conjunction(parts)

    # expressions

    def expression(self) -> Any:
        terms = [self.term()]
        while self.peek() in ("+", "-"):
            sign = self.take()
            term = self.term()
            terms.append(term if sign == "+" else _negated(term))
        return terms[0] if len(terms) == 1 else Add(tuple(terms))

    def term(self) -> Any:
        out = self.factor()
        while self.peek() in ("*", "%", "mod") or self._implicit(out):
            # isl writes a multiple as 2 i as well as 2 * i
            operator = "*" if self._implicit(out) else self.take()
            right = self.factor()
            if operator == "*":
                if not (isinstance(out, int) or isinstance(right, int)):
                    raise Unstatable(f"a product of two unknowns in {self.text!r}")
                out = Product((out, right))
            else:
                if not (isinstance(right, int) and right > 0):
                    raise Unstatable(f"a remainder by {right} in {self.text!r}")
                out = Remainder(out, right)
        return out

    def _implicit(self, left: Any) -> bool:
        """Whether a number is followed by what it multiplies: ``2 i``, ``2 (i)``."""
        token = self.peek()
        if not isinstance(left, int) or token is None:
            return False
        if token in ("(", "floord", "floor"):
            return True
        named = re.fullmatch(r"[A-Za-z_][A-Za-z_0-9']*", token) is not None
        return named and token not in _WORDS

    def factor(self) -> Any:
        token = self.peek()
        if token is None:
            raise Unstatable(f"{self.text!r} ends early")
        if token == "floor":
            # isl's floor(e / k), the same as floord(e, k)
            self.take()
            self.take("(")
            numerator = self.expression()
            self.take("/")
            denominator = self.factor()
            self.take(")")
            if not (isinstance(denominator, int) and denominator > 0):
                raise Unstatable(f"floor by {denominator} in {self.text!r}")
            return FloorDiv(numerator, denominator)
        if token == "(":
            self.take()
            inner = self.expression()
            self.take(")")
            return inner
        if token == "-":
            self.take()
            inner = self.factor()
            return -inner if isinstance(inner, int) else _negated(inner)
        if token == "floord":
            self.take()
            self.take("(")
            numerator = self.expression()
            self.take(",")
            denominator = self.expression()
            self.take(")")
            if not (isinstance(denominator, int) and denominator > 0):
                raise Unstatable(f"floord by {denominator} in {self.text!r}")
            return FloorDiv(numerator, denominator)
        if token.isdigit():
            self.take()
            return int(token)
        if re.fullmatch(r"[A-Za-z_][A-Za-z_0-9']*", token) and token not in _WORDS:
            self.take()
            return self.resolve(token)
        raise Unstatable(f"cannot read {token!r} in {self.text!r}")


def _negated(term: Any) -> Any:
    """``-term``, as a product by ``-1`` (what pymbolic builds for a difference)."""
    if isinstance(term, int):
        return -term
    return Product((-1, term))


def parse(text: str, resolve: Any = Var) -> Any:
    """One constraint of loopty's isl text as a lanky proposition.

    ``resolve`` maps a name to the term it stands for, and is a variable of
    that name by default.

    Raises:
        Unstatable: If the text is not of the shapes loopty writes.
    """
    reader = _Reader(text, resolve)
    out = reader.disjunction()
    reader.done()
    return out


# }}}
