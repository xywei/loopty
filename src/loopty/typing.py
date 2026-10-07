"""Typing rules: from a term to a list of facts.

Nothing here decides anything. The rules read a :class:`~loopty.term.Term` and
*state* the obligations it generates, as lanky facts, which the oracles then try
from the strongest trust class down. That separation is the point of the design:
a rule is a short, readable statement of what has to hold, and the only thing
trusted to answer it is an oracle that says how much it should be trusted.

The rules are these.

*In bounds, one per access.* When the index expression is quasi-affine the
obligation is an isl subset: the cells the access reaches, over the domain it
runs on, are cells the array has. When the index is an array read whose element
*type* is the index type of the axis it indexes, there is nothing to decide:
``col: Arr[Fin[n], Fin[cnt], Fin[m]]`` says every entry of ``col`` is a point of
``Fin[m]``, so ``x[col[r, j]]`` with ``x: Arr[Fin[m], Real]`` is in bounds by
type, recorded ``DECIDED`` with ``decided_by="type"`` and no oracle call. An
access that is neither is left ``ASSUMED`` with its reason, because widening a
non-affine index into a parameter would let isl *refute* an obligation that is
merely unknown.

*Element sorts, one per write.* The by-type rule rests on every cell of
``col`` being a point of ``Fin[m]``, which the contract checks when the kernel
is called. A kernel that writes into such an array has to keep it so, or
``perm[i] = i + 1`` followed by ``x[perm[j]]`` reads past the end of ``x``
under an in-bounds fact decided by type. So every write into an array whose
element sort is ``Fin[m]`` owes the fact that the value written is a point of
``Fin[m]`` (kind ``element-sort``): isl decides it where the value is
quasi-affine, refuting it with the instance that writes outside; the type
decides it where the value is itself read from an array of that element sort;
and it is ``assumed`` otherwise. An in-bounds fact decided by type through an
array the kernel writes rests on the element-sort facts of every write into
that array (lanky's ``rests_on``), so the ledger counts them in what it is
worth.

*A layout the kernel rewrites.* The in-bounds fact of a ragged access
``val[r, j]`` is decided against the length of row ``r``, and the disjoint
writes and the dependences of ``val`` are computed over ``[r, j]``. Both hold
of the flat buffer only while the offsets lay every row out inside it and
apart from the others, which the contract checks when the call starts. A
kernel that writes the counts or the offsets its ragged arrays are read
through can break that during the run: a row moved past the end of the buffer,
or two rows moved onto the same cells. Such a kernel gets one ``layout`` fact
per counts family it rewrites (:func:`layout_facts`). isl decides it where the
kernel writes only the offsets, each as the counts lay it out
(``off[r + 1] = off[r] + cnt[r]``) or as a value of its loop variables and the
sizes alone, of which only ``off[0] = 0`` is right whatever the counts, and
refutes it with an instance that writes another; any other write leaves it
``assumed`` with the reason. The in-bounds and disjoint-writes facts of the
family's ragged arrays rest on it, as does a fact decided by type through an
index read from one of them (``x[col[r, j]]``), and the ``monotone`` casts of
a schedule of the kernel (:mod:`loopty.schedule`). The ledger then shows them
decided under the layout, and worth no more than it.

*Write disjointness.* Distinct instances of a statement must write distinct
cells, or the loop cannot be run in parallel and the order of the writes is
part of the result. The obligation is the emptiness of the set of pairs of
distinct instances whose writes collide, which isl decides and, when it fails,
witnesses with the two instances.

*Ordering.* The dependence relation is defined from the footprints
(:mod:`loopty.flow`), and the source order has to run every dependence forward
in time. That is the same monotonicity question a schedule has to answer, asked
of the order the body was written in, so the ledger records the baseline that
every later cast is compared against.

*Reduction exactness.* Every reduction states the exactness class of its
accumulation. ``reassoc`` says the sum may be reassociated, which is what
licenses a reduction tree or atomics, and which fixes the tolerance a
differential test is allowed to use. The class is read off the term, so the fact
is ``DECIDED`` by the type rather than by an oracle.

*The postcondition.* The return annotation of a kernel is a claim about its
parameters. It is emitted as a fact with its term, for whatever oracle can take
it, and stays ``ASSUMED`` when none can, which is the honest outcome and is
visible in the ledger rather than lost.

The in-bounds, disjointness and ordering facts are stated over statement
domains, and a guard narrows a domain only where isl can state it. A conjunct
it cannot (one that reads an array, compares with ``!=``, or compares with a
``Real`` scalar, which isl would read as an integer) leaves the domain wider
than the instances that write, and each such fact lists those conjuncts, with
the reason, under ``unnarrowed`` in its provenance: proved, it holds for the
instances that write too; refuted, its witness may be an instance the guard
masks.

Every id is built by :func:`lanky.ledger.fact_id`: the rule's kind, the
kernel's definition (``module`` and ``line`` as well as ``owner``, which
:class:`loopty.kernel.Kernel` passes), and what the fact is about, as in
``in-bounds:spmv.spmv@102:S0:read:x[col[r, j]]``. A term checked with no
kernel behind it leaves the module and the line out.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

import islpy as isl
import pymbolic.primitives as prim
from lanky.ledger import Fact, Status, fact_id
from lanky.prelude import FinType
from lanky.terms import render, structurally_equal

from loopty import flow
from loopty.oracle import Empty, Monotone, Subset
from loopty.term import ArrType, Term
from loopty.trace import reductions_in

__all__ = [
    "element_sort_facts",
    "facts_for",
    "in_bounds_facts",
    "instance_labels",
    "layout_fact_ids",
    "layout_facts",
    "ordering_facts",
    "postcondition_facts",
    "postcondition_id",
    "reduction_facts",
    "render_instance",
    "write_disjointness_facts",
]


def _align_both(first: Any, second: Any) -> tuple[Any, Any]:
    """Give two isl objects the union of their parameters, so they can be compared."""
    second = second.align_params(first.get_space())
    first = first.align_params(second.get_space())
    return first, second


def instance_labels(term: Term) -> tuple[str, ...]:
    """Names of the coordinates of the padded instance space, for a witness."""
    depth = flow.instance_space_depth(term)
    return ("s", *[f"d{k}" for k in range(depth)])


def render_instance(term: Term, coordinates: Sequence[int]) -> str:
    """Render one padded instance tuple as ``S1[r=2, j=0]``.

    A witness comes back from isl as a tuple of integers in the padded instance
    space, whose first coordinate is the statement's index. Turning it back into
    the statement's own inames is what makes a rejected schedule's explanation
    about the source rather than about the encoding.
    """
    if not coordinates:
        return "[]"
    index = int(coordinates[0])
    if not 0 <= index < len(term.stmts):
        return "(" + ", ".join(str(c) for c in coordinates) + ")"
    stmt = term.stmts[index]
    pairs = [
        f"{iname}={coordinates[k + 1]}"
        for k, iname in enumerate(stmt.inames)
        if k + 1 < len(coordinates)
    ]
    return f"{stmt.id}[{', '.join(pairs)}]"


# {{{ in bounds


def _index_type(expr: Any, types: dict[str, Any]) -> Any:
    """The index type of a value used as a subscript, when its type says one.

    ``col[r, j]`` has the element type of ``col``; a scalar parameter has its
    own annotation. Anything else has no type-level bound, and the obligation
    goes to isl instead.
    """
    if isinstance(expr, prim.Subscript) and isinstance(expr.aggregate, prim.Variable):
        declared = types.get(expr.aggregate.name)
        if isinstance(declared, ArrType):
            return declared.dtype
        return None
    if isinstance(expr, prim.Variable):
        declared = types.get(expr.name)
        return declared if isinstance(declared, FinType) else None
    return None


def _typed_by(indices: Sequence[Any], types: dict[str, Any]) -> list[str]:
    """The arrays whose element sorts the by-type rule reads for ``indices``.

    ``col`` for ``x[col[r, j]]``; a scalar parameter of a ``Fin`` sort is a
    value of the call, which no statement writes, and is not listed.
    """
    out: list[str] = []
    for index in indices:
        if isinstance(index, prim.Subscript) and isinstance(
            index.aggregate, prim.Variable
        ):
            name = index.aggregate.name
            if isinstance(types.get(name), ArrType) and name not in out:
                out.append(name)
    return out


def _by_type_rests_on(
    places: Sequence[Sequence[Any]],
    types: dict[str, Any],
    sorts: Mapping[str, tuple[str, ...]],
    layouts: Mapping[str, tuple[str, ...]],
) -> tuple[str, ...]:
    """What a fact decided by type through the indices of ``places`` rests on.

    For every array an index is read from (:func:`_typed_by`), the
    element-sort facts of the writes into it, since the type says what the
    contract checked when the call started and those facts say the term keeps
    it so, and then the layout fact of a ragged one whose layout the term
    rewrites, since the element is read from a cell of a row that the layout
    is to keep inside its buffer (:func:`layout_facts`).
    """
    names = list(
        dict.fromkeys(name for indices in places for name in _typed_by(indices, types))
    )
    return tuple(
        dict.fromkeys(
            [
                *(sort_id for name in names for sort_id in sorts.get(name, ())),
                *(layout_id for name in names for layout_id in layouts.get(name, ())),
            ]
        )
    )


def _justified_by_type(
    indices: Sequence[Any], arrtype: ArrType, types: dict[str, Any]
) -> str | None:
    """Explain why every index is in bounds by type, or return ``None``.

    The rule is the index-type homomorphism doing its job: a value of type
    ``Fin[m]`` is a point of ``Fin[m]``, so indexing an axis of size ``m`` with
    it needs no proof. Every axis has to be justified this way for the access to
    skip the oracle, and a ragged axis never is, because its bound depends on the
    row and no element type can state that.
    """
    reasons = []
    for axis, index in enumerate(indices):
        if axis >= len(arrtype.axes) or arrtype.ragged[axis]:
            return None
        declared = _index_type(index, types)
        if not isinstance(declared, FinType):
            return None
        if not structurally_equal(declared.bound, arrtype.axes[axis]):
            return None
        reasons.append(f"{render(index)} : {declared}")
    return ", ".join(reasons) if reasons else None


def in_bounds_facts(
    term: Term, owner: str, *, module: str | None = None, line: int | None = None
) -> list[Fact]:
    """One fact per array access: the cells it reaches are cells the array has.

    The fact's id names the access, ``S1:read:x[r - 1]``, and not the domain it
    runs over, while the collector can list one access over several: the body's
    and the guard's, a reduction's, and the reads a ragged layout makes (the
    offsets a ragged access reads through, the bound of a ragged loop). The
    ledger keeps one fact per id, so two facts for one access would leave the
    later in place of the earlier, whatever the earlier said: ``x[r - 1]``
    read directly and again in a sum over ``q < r`` would be reported in
    bounds from the sum, where ``r >= 1``, with the direct read of ``x[-1]``
    gone from the ledger. The domains of one access are therefore gathered
    into one obligation, about the union of the cells they reach.

    A read a ragged layout makes is an access the source never writes, so its
    fact says what it serves: ``off[r], the start of row r that val[r, j] is
    flattened through, is in bounds ...``, or ``cnt[r], the length of row r
    that bounds the loop over j, is in bounds ...``, and ``layout`` in its
    provenance says the same. Without that a refuted one names a read nobody
    can find in the kernel.

    A fact decided by type through an array the term writes rests on the
    element-sort facts of the writes into it (:func:`element_sort_facts`):
    the type says what the contract checked when the call started, and those
    facts say the term keeps it so. The fact of an access to a ragged array
    whose layout the term rewrites rests on that layout's fact
    (:func:`layout_facts`), since it is decided against the row, and so does
    a fact decided by type through an index read from such an array
    (``x[col[r, j]]``), since the element is read from a cell of the row.
    """
    types = dict(term.params)
    sizes = flow.size_names(term)
    sorts = _element_sort_ids(term, owner, module=module, line=line)
    layouts = layout_fact_ids(term, owner, module=module, line=line)
    # One table for the whole term: the cells an array has and the cells an
    # access reaches are two isl sets that get compared, so both have to call
    # ``cnt[r]`` by the parameter the statement domains already use.
    reflections = term.reflections
    facts: list[Fact] = []
    for stmt in term.stmts:
        listed: dict[
            tuple[str, str, str],
            list[tuple[tuple[Any, ...], tuple[str, ...], isl.Set]],
        ] = {}
        for array, indices, kind, inames, domain in flow.statement_accesses(
            stmt, term
        ):
            if not isinstance(types.get(array), ArrType):
                continue
            text = _access_text(array, indices)
            listed.setdefault((array, kind, text), []).append(
                (tuple(indices), inames, domain)
            )
        roles = _layout_roles(stmt, term)
        for (array, kind, text), places in listed.items():
            arrtype = types[array]
            indices = places[0][0]
            identifier = fact_id(
                "in-bounds",
                owner,
                module=module,
                line=line,
                detail=f"{stmt.id}:{kind}:{text}",
            )
            role = roles.get((array, kind, text))
            subject = text if role is None else f"{text}, {role},"
            layout = {} if role is None else {"layout": role}
            reasons = [
                _justified_by_type(place[0], arrtype, types) for place in places
            ]
            reason = reasons[0] if None not in reasons else None
            if reason is not None:
                rests_on = _by_type_rests_on(
                    [place[0] for place in places], types, sorts, layouts
                )
                rests_on += layouts.get(array, ())
                facts.append(
                    Fact(
                        id=identifier,
                        kind="in-bounds",
                        statement=f"{subject} is in bounds by type ({reason})",
                        term=None,
                        status=Status.DECIDED,
                        decided_by="type",
                        provenance={
                            "rule": "index type",
                            "reason": reason,
                            **layout,
                        },
                        where=stmt.where,
                        owner=owner,
                        rests_on=rests_on,
                    )
                )
                continue
            try:
                widened = False
                reached = None
                for place_indices, inames, domain in places:
                    relation = flow.access_relation(inames, domain, place_indices)
                    widened = widened or _is_widened(relation, place_indices)
                    part = relation.range()
                    if reached is not None:
                        reached, part = _align_both(reached, part)
                        part = reached.union(part).coalesce()
                    reached = part
                cells = flow.cell_set(arrtype, indices, reflections=reflections)
                reached, cells = _align_both(reached, cells)
                reached = flow.assume_sizes(reached, sizes)
                cells = flow.assume_sizes(cells, sizes)
            except Exception as exc:  # noqa: BLE001 - an unstatable rule is ASSUMED
                facts.append(
                    Fact(
                        id=identifier,
                        kind="in-bounds",
                        statement=f"{subject} is in bounds",
                        term=None,
                        status=Status.ASSUMED,
                        provenance={"reason": f"no isl form: {exc}", **layout},
                        where=stmt.where,
                        owner=owner,
                    )
                )
                continue
            if widened:
                facts.append(
                    Fact(
                        id=identifier,
                        kind="in-bounds",
                        statement=f"{subject} is in bounds",
                        term=None,
                        status=Status.ASSUMED,
                        provenance={
                            "reason": (
                                "the index is not quasi-affine and its type does "
                                "not bound it, so isl is not asked"
                            ),
                            **layout,
                        },
                        where=stmt.where,
                        owner=owner,
                    )
                )
                continue
            wide = any(place[2] is not stmt.loop_domain for place in places)
            facts.append(
                Fact(
                    id=identifier,
                    kind="in-bounds",
                    statement=(
                        f"{subject} is in bounds for every instance of {stmt.id}"
                    ),
                    term=Subset(
                        reached,
                        cells,
                        description=f"cells {text} reaches are cells {array} has",
                        labels=tuple(f"a{k}" for k in range(len(indices))),
                    ),
                    status=Status.ASSUMED,
                    provenance={
                        "access": text,
                        "statement": stmt.id,
                        **layout,
                        **(_unnarrowed(stmt) if wide else {}),
                    },
                    where=stmt.where,
                    owner=owner,
                    rests_on=layouts.get(array, ()),
                )
            )
    return facts


def _access_text(array: str, indices: Sequence[Any]) -> str:
    """``val[r, j]``: an access as the ledger writes it."""
    return f"{array}[{', '.join(render(i) for i in indices)}]"


def _layout_roles(stmt: Any, term: Term) -> dict[tuple[str, str, str], str]:
    """What each read a ragged layout makes is to the accesses and loops it serves.

    Keyed as :func:`in_bounds_facts` keys an access, ``(array, kind, text)``.
    The value reads ``the start of row r that val[r, j] is flattened through``
    for a read the flat index makes, ``the length of row r that bounds the loop
    over j`` for a ragged loop's bound read from the counts, and ``the end of
    row r whose length bounds the loop over j`` for one computed from the
    offsets (:func:`loopty.flow.layout_reads`). A read that is several of those
    gets one clause each, joined, and opens with ``read directly and as`` when
    the statement also spells the read itself.
    """
    spelled = {
        (array, kind, _access_text(array, indices))
        for array, indices, kind, _inames, _domain in flow.source_accesses(
            stmt, term
        )
    }
    served: dict[tuple[str, str, str], dict[tuple[str, str, str], list[str]]] = {}
    for layout in flow.layout_reads(stmt, term):
        array, indices, kind, _inames, _domain = layout.read
        key = (array, kind, _access_text(array, indices))
        if layout.access is not None:
            how = "index"
            users = [_access_text(layout.access[0], layout.access[1])]
        else:
            how = "bound"
            users = list(layout.loops)
        clause = served.setdefault(key, {}).setdefault(
            (how, layout.part, render(layout.row)), []
        )
        clause.extend(user for user in users if user not in clause)
    out: dict[tuple[str, str, str], str] = {}
    for key, clauses in served.items():
        role = _listed(
            [
                _layout_clause(how, part, row, users)
                for (how, part, row), users in clauses.items()
            ]
        )
        out[key] = f"read directly and as {role}" if key in spelled else role
    return out


def _layout_clause(how: str, part: str, row: str, users: Sequence[str]) -> str:
    """One clause of :func:`_layout_roles`: what a layout read is to its users."""
    if how == "index":
        verb = "is" if len(users) == 1 else "are"
        spelled = _listed(users)
        return f"the {part} of row {row} that {spelled} {verb} flattened through"
    loops = (
        "a loop"
        if not users
        else f"the loop over {users[0]}"
        if len(users) == 1
        else f"the loops over {_listed(users)}"
    )
    if part == "length":
        return f"the length of row {row} that bounds {loops}"
    if part == "row":
        return f"the index of the row whose length bounds {loops}"
    return f"the {part} of row {row} whose length bounds {loops}"


def _listed(items: Sequence[str]) -> str:
    """``a``, ``a and b``, ``a, b and c``."""
    if len(items) < 3:
        return " and ".join(items)
    return f"{', '.join(items[:-1])} and {items[-1]}"


def _unnarrowed(stmt: Any) -> dict[str, Any]:
    """The provenance of a fact stated over a domain its guard left wide.

    A guard conjunct isl cannot state (:attr:`loopty.term.Stmt.unnarrowed`)
    does not narrow the statement's domain, so the domain over-approximates
    the instances that write, and a fact about it is about instances the guard
    masks as well. Proved, it holds for the instances that write all the same;
    refuted, its witness may be one of the masked ones. The fact records which
    conjuncts were left out, and why, under ``unnarrowed``.
    """
    if not stmt.unnarrowed:
        return {}
    return {
        "unnarrowed": [
            {"conjunct": conjunct, "why": why} for conjunct, why in stmt.unnarrowed
        ]
    }


def _is_widened(relation: isl.Map, indices: Sequence[Any]) -> bool:
    """Whether the access map was widened to every cell rather than computed.

    :func:`loopty.flow.access_relation` widens a non-affine index, which is the
    right answer for a dependence and the wrong one for an obligation: isl would
    happily refute a claim about a cell the access never reaches. The widened map
    is recognizable because its range is unbounded.
    """
    try:
        return not relation.range().is_bounded()
    except Exception:  # pragma: no cover - isl always answers this
        return False


# }}}


# {{{ element sorts


def _fin_sort(typ: Any) -> FinType | None:
    """The ``Fin[m]`` element sort of an array type, or ``None``."""
    if isinstance(typ, ArrType) and isinstance(typ.dtype, FinType):
        return typ.dtype
    return None


def _element_sort_id(
    stmt: Any, owner: str, *, module: str | None, line: int | None
) -> str:
    """The id of the element-sort fact of one write, by statement and cell."""
    written = stmt.assignee
    return fact_id(
        "element-sort",
        owner,
        module=module,
        line=line,
        detail=f"{stmt.id}:{_access_text(written.array, written.indices)}",
    )


def _element_sort_ids(
    term: Term, owner: str, *, module: str | None, line: int | None
) -> dict[str, tuple[str, ...]]:
    """The element-sort fact ids of the writes into each array, by array."""
    types = term.array_types
    out: dict[str, list[str]] = {}
    for stmt in term.stmts:
        array = stmt.assignee.array
        if _fin_sort(types.get(array)) is None:
            continue
        out.setdefault(array, []).append(
            _element_sort_id(stmt, owner, module=module, line=line)
        )
    return {array: tuple(ids) for array, ids in out.items()}


def element_sort_facts(
    term: Term, owner: str, *, module: str | None = None, line: int | None = None
) -> list[Fact]:
    """One fact per write into an array of a ``Fin`` element sort.

    The value written is a point of the sort, which is what keeps the by-type
    rule for an index read from the array sound once the kernel has written
    it (see the module docstring). isl decides it when the value is
    quasi-affine: the instances that write a value outside the sort are none,
    and a refutation names one, at the sizes it was read off at. A value read
    from an array of the same element sort, or a scalar of that sort, is a
    point of it by type, and the fact rests on the element-sort facts of the
    writes into that array, and on its layout fact when the term rewrites the
    layout of its rows (:func:`layout_facts`). Anything else is ``assumed``,
    with the reason.
    """
    types = dict(term.params)
    types.update(term.temporaries)
    sizes = flow.size_names(term)
    reflections = term.reflections
    sorts = _element_sort_ids(term, owner, module=module, line=line)
    layouts = layout_fact_ids(term, owner, module=module, line=line)
    facts: list[Fact] = []
    for stmt in term.stmts:
        written = stmt.assignee
        sort = _fin_sort(types.get(written.array))
        if sort is None:
            continue
        cell = _access_text(written.array, written.indices)
        value = stmt.expr
        shown = render(value)
        identifier = _element_sort_id(stmt, owner, module=module, line=line)
        statement = (
            f"the value {stmt.id} writes into {cell}, {shown}, is a point of {sort}"
        )
        common = {
            "statement": stmt.id,
            "array": written.array,
            "value": shown,
            "sort": str(sort),
        }
        virtual = ArrType(axes=(sort.bound,), dtype=sort, ragged=(False,))
        reason = _justified_by_type((value,), virtual, types)
        if reason is not None:
            facts.append(
                Fact(
                    id=identifier,
                    kind="element-sort",
                    statement=f"{statement} by type ({reason})",
                    term=None,
                    status=Status.DECIDED,
                    decided_by="type",
                    provenance={**common, "rule": "index type", "reason": reason},
                    where=stmt.where,
                    owner=owner,
                    rests_on=_by_type_rests_on([(value,)], types, sorts, layouts),
                )
            )
            continue
        try:
            relation = flow.access_relation(stmt.inames, stmt.domain, (value,))
            if _is_widened(relation, (value,)):
                raise flow.NonAffine("widened")
            points = flow.cell_set(virtual, (value,), reflections=reflections)
            relation, points = _align_both(relation, points)
            outside = relation.intersect_range(points.complement()).domain()
            outside = flow.assume_sizes(outside, sizes)
        except Exception as exc:  # noqa: BLE001 - an unstatable rule is ASSUMED
            why = (
                "the value is not quasi-affine and its type does not say it is "
                f"a point of {sort}, so isl is not asked"
                if isinstance(exc, flow.NonAffine)
                else f"no isl form: {exc}"
            )
            facts.append(
                Fact(
                    id=identifier,
                    kind="element-sort",
                    statement=statement,
                    term=None,
                    status=Status.ASSUMED,
                    provenance={**common, "reason": why},
                    where=stmt.where,
                    owner=owner,
                )
            )
            continue
        facts.append(
            Fact(
                id=identifier,
                kind="element-sort",
                statement=statement,
                term=Empty(
                    outside,
                    description=(
                        f"instances of {stmt.id} writing a value outside {sort} "
                        f"into {written.array}"
                    ),
                    labels=tuple(stmt.inames),
                ),
                status=Status.ASSUMED,
                provenance={**common, **_unnarrowed(stmt)},
                where=stmt.where,
                owner=owner,
            )
        )
    return facts


# }}}


# {{{ a layout the term rewrites


def _rewritten_layouts(term: Term) -> dict[str, dict[str, Any]]:
    """Every counts family whose layout the term writes, with what writes it.

    For each, by counts name: ``arrays``, the ragged arrays laid out over the
    family; ``layout``, the arrays their rows are read through, the counts
    when they are an array of the term and the offsets it declares
    (:meth:`loopty.term.Term.offsets_of`); ``written``, those of them the
    term writes; and ``statements``, the statements that write them.
    """
    types = term.array_types
    families: dict[str, list[str]] = {}
    for name, typ in types.items():
        if not isinstance(typ, ArrType) or typ.domain is not None:
            continue
        for size, ragged in zip(typ.axes, typ.ragged, strict=True):
            if ragged and isinstance(size, prim.Variable):
                families.setdefault(size.name, []).append(name)
    out: dict[str, dict[str, Any]] = {}
    for counts, arrays in families.items():
        layout = [counts] if isinstance(types.get(counts), ArrType) else []
        offsets = term.offsets_of(counts)
        if offsets is not None and offsets not in layout:
            layout.append(offsets)
        writers = [stmt for stmt in term.stmts if stmt.assignee.array in layout]
        if not writers:
            continue
        assigned = {stmt.assignee.array for stmt in writers}
        written = [name for name in layout if name in assigned]
        out[counts] = {
            "arrays": arrays,
            "layout": layout,
            "written": written,
            "statements": writers,
        }
    return out


def layout_fact_ids(
    term: Term, owner: str, *, module: str | None = None, line: int | None = None
) -> dict[str, tuple[str, ...]]:
    """The ids of the layout facts the facts about each ragged array rest on.

    By array, for every ragged array laid out over a family the term
    rewrites (see :func:`layout_facts`); an array not listed rests on none.
    """
    out: dict[str, tuple[str, ...]] = {}
    for counts, family in _rewritten_layouts(term).items():
        identifier = fact_id("layout", owner, module=module, line=line, detail=counts)
        for array in family["arrays"]:
            out[array] = (*out.get(array, ()), identifier)
    return out


def layout_facts(
    term: Term, owner: str, *, module: str | None = None, line: int | None = None
) -> list[Fact]:
    """One fact per counts family whose layout the term rewrites.

    It states what the facts about the family's ragged arrays take for
    granted once the term has written the counts or the offsets they are read
    through: that every row stays inside the buffer and apart from the
    others, as the contract checked when the call started. A term that only
    reads its layout has none.

    Where the family's rows are as long as a counts array the term reads and
    does not write, and the term writes only the offsets, the fact is decided
    (#86). The contract checks on entry that the offsets start at 0 and have
    the counts as their differences, so ``off[q] = off[q - 1] + cnt[q - 1]``
    stores the value the cell already holds, and so does ``off[0] = 0``: if
    every write is one of those, every write leaves the offsets as they were,
    by induction over the run, and the rows stay where the contract found
    them. A start written without reading anything, a value of the loop
    variables and the sizes, is the start the counts give its row only at row
    0, where it is 0: the counts are data, and the start of row ``q`` is the
    sum of the counts before it, which some counts make another value. So the
    fact is the isl question whether any instance writes such a start
    elsewhere, and a witness is an instance that moves a row: a negative
    start is a row before the buffer; a positive one, with a count of 1 in
    that row alone, a row past the end of a buffer of one cell; and 0 at
    another row, with a count of 1 in it and in row 0, two rows on cell 0. Any
    other
    write (the counts themselves, a start read from another array, a value
    under a guard isl cannot state) leaves the fact ``assumed``, with the
    reason, and the facts that rest on it say so in the ledger.
    """
    facts: list[Fact] = []
    for counts, family in _rewritten_layouts(term).items():
        arrays = _listed(family["arrays"])
        written = _listed(family["written"])
        writers = [stmt.id for stmt in family["statements"]]
        provenance: dict[str, Any] = {
            "counts": counts,
            "arrays": list(family["arrays"]),
            "layout": list(family["layout"]),
            "written": list(family["written"]),
            "statements": writers,
        }
        question = _row_starts_question(term, counts, family)
        if isinstance(question, str):
            provenance["reason"] = (
                f"{_listed(writers)} write {written}, which the rows of "
                f"{arrays} are read through. Their accesses are in bounds "
                "against the length of their row, and their cells are told "
                "apart as [r, j], which holds of the flat buffer while every "
                "row lies inside it and apart from the others, as the contract "
                "checks when the call starts; nothing states what the kernel "
                f"writes there during the run: {question}"
            )
        else:
            question, restated, fixed = question
            provenance["rule"] = _row_starts_rule(
                term.offsets_of(counts) or "", counts, restated, fixed
            )
            provenance["restated"] = restated
            provenance["fixed"] = fixed
        facts.append(
            Fact(
                id=fact_id("layout", owner, module=module, line=line, detail=counts),
                kind="layout",
                statement=(
                    f"the rows of {arrays} stay inside their buffers and apart "
                    f"while {_listed(writers)} write {written}"
                ),
                term=None if isinstance(question, str) else question,
                status=Status.ASSUMED,
                provenance=provenance,
                where=family["statements"][0].where,
                owner=owner,
            )
        )
    return facts


def _row_starts_rule(
    offsets: str, counts: str, restated: Sequence[str], fixed: Sequence[str]
) -> str:
    """What a decided layout fact rests on, in words."""
    def verb(ids: Sequence[str]) -> str:
        return "writes" if len(ids) == 1 else "write"

    parts = []
    if restated:
        parts.append(
            f"{_listed(restated)} {verb(restated)} {offsets}[q] = "
            f"{offsets}[q - 1] + {counts}[q - 1] at 1 <= q <= the number of "
            "rows, the value the contract checked there, which leaves the "
            "offsets as they were"
        )
    if fixed:
        parts.append(
            f"{_listed(fixed)} {verb(fixed)} a start that reads no array, "
            f"which is the start the counts give its row only for "
            f"{offsets}[0] = 0"
        )
    return (
        "; ".join(parts)
        + f", and nothing writes {counts}: the rows stay as the contract "
        "checked them if no instance writes another start"
    )


def _row_starts_question(
    term: Term, counts: str, family: Mapping[str, Any]
) -> tuple[Empty, list[str], list[str]] | str:
    """The question that decides a family's layout fact, or why there is none.

    See :func:`layout_facts`. The question is an :class:`Empty` over the
    padded instance space, of the instances that write a start of a row
    without reading anything, other than 0 at row 0. A write that restates
    the start from the counts adds nothing to it. With the restated and the
    fixed statements, by id.
    """
    types = term.array_types
    counts_type = types.get(counts)
    if not isinstance(counts_type, ArrType):
        return (
            f"the rows are as long as the offsets say, {counts} being no "
            "array of the kernel, and no rule follows their differences"
        )
    if counts in family["written"]:
        writers = [s.id for s in family["statements"] if s.assignee.array == counts]
        return (
            f"{_listed(writers)} write {counts}, the lengths of the rows, and "
            "no rule follows what a row's new length reaches"
        )
    offsets = term.offsets_of(counts)
    try:
        rows = flow.expr_text(counts_type.axes[0], None, None)
    except Exception:  # noqa: BLE001 - said as the reason
        return f"the number of rows, {render(counts_type.axes[0])}, is not affine"
    sizes = flow.size_names(term)
    allowed = set(sizes) - set(dict(term.reflected))
    depth = flow.instance_space_depth(term)
    restated: list[str] = []
    fixed: list[tuple[int, Any, isl.Set]] = []
    for index, stmt in enumerate(term.stmts):
        if not any(stmt is writer for writer in family["statements"]):
            continue
        if _restates_start(stmt, offsets, counts, rows, sizes):
            restated.append(stmt.id)
            continue
        instances = _fixed_starts(stmt, rows, allowed)
        if instances is None:
            cell = _access_text(stmt.assignee.array, stmt.assignee.indices)
            return (
                f"{stmt.id} writes {cell} = {render(stmt.expr)}, which is "
                f"neither the start the counts give the row, {offsets}[q - 1] "
                f"+ {counts}[q - 1] at q >= 1, nor a value of the loop "
                "variables and the sizes alone"
            )
        fixed.append((index, stmt, instances))
    if len(fixed) == 1 and fixed[0][1].inames:
        # One statement: its instances in its own loop variables, which is
        # how a witness reads best, ``[r=1]``.
        ((_index, stmt, found),) = fixed
        labels: tuple[str, ...] = tuple(stmt.inames)
    else:
        labels = instance_labels(term)
        found = isl.Set(f"{{ [{', '.join(labels)}] : 1 = 0 }}")
        for index, stmt, instances in fixed:
            pad = flow.pad_map(len(stmt.inames), index, depth)
            piece = instances.apply(pad.align_params(instances.get_space()))
            found, piece = _align_both(found, piece)
            found = found.union(piece)
    ids = [stmt.id for _index, stmt, _instances in fixed]
    question = Empty(
        flow.assume_sizes(found, sizes),
        description=(
            f"instances of {_listed(ids) or 'no statement'} that write into "
            f"{offsets} the start of a row which the counts in {counts} can "
            "put elsewhere (the start of row q is the sum of the counts before "
            "it, and only row 0 starts at 0 whatever they are)"
        ),
        labels=labels,
    )
    return question, restated, ids


def _restates_start(
    stmt: Any, offsets: str | None, counts: str, rows: str, sizes: Any
) -> bool:
    """Whether ``stmt`` writes ``off[q] = off[q - 1] + cnt[q - 1]``, ``1 <= q <= n``.

    The two reads are told by their arrays and their indices are compared
    with ``q - 1`` by isl, over the statement's domain, so ``off[r + 1] =
    cnt[r] + off[r]`` is one. ``rows`` is the number of rows, as isl text.
    """
    if offsets is None or stmt.kind != "assign" or len(stmt.assignee.indices) != 1:
        return False
    (cell,) = stmt.assignee.indices
    value = stmt.expr
    if not isinstance(value, prim.Sum) or len(value.children) != 2:
        return False
    reads: dict[str, Any] = {}
    for child in value.children:
        if not (
            isinstance(child, prim.Subscript)
            and isinstance(child.aggregate, prim.Variable)
        ):
            return False
        index = child.index
        indices = index if isinstance(index, tuple) else (index,)
        if len(indices) != 1:
            return False
        reads[child.aggregate.name] = indices[0]
    if set(reads) != {offsets, counts}:
        return False
    try:
        before = flow.access_relation(stmt.inames, stmt.domain, (cell - 1,))
        if _is_widened(before, (cell,)):
            return False
        for name in (offsets, counts):
            read = flow.access_relation(stmt.inames, stmt.domain, (reads[name],))
            left, right = _align_both(before, read)
            if _is_widened(read, (reads[name],)) or not left.is_equal(right):
                return False
        cells = flow.access_relation(stmt.inames, stmt.domain, (cell,))
        names = sorted(flow.free_names(rows))
        inside = isl.Set(
            f"[{', '.join(names)}] -> {{ [q] : 1 <= q <= {rows} }}"
        )
        cells, inside = _align_both(cells, inside)
        outside = flow.assume_sizes(
            cells.intersect_range(inside.complement()).domain(), sizes
        )
        return bool(outside.is_empty())
    except Exception:  # noqa: BLE001 - not the shape, then
        return False


def _fixed_starts(stmt: Any, rows: str, allowed: set[str]) -> isl.Set | None:
    """The instances of ``stmt`` that write a start of a row other than 0 at 0.

    Over the statement's own iteration space, or ``None`` when ``stmt`` is
    not a write of a value that reads nothing: its domain, its cell and its
    value have to be affine in its loop variables and the sizes, with no
    reflected bound and no scalar, and its guard has to be stated whole, so
    that every instance in the set does write.
    """
    from loopty.trace import accesses_in

    if stmt.kind != "assign" or stmt.unnarrowed or len(stmt.assignee.indices) != 1:
        return None
    (cell,) = stmt.assignee.indices
    if accesses_in((cell, stmt.expr)):
        return None
    if not set(stmt.domain.get_var_names(isl.dim_type.param)) <= allowed:
        return None
    try:
        relation = flow.access_relation(stmt.inames, stmt.domain, (cell, stmt.expr))
    except Exception:  # noqa: BLE001 - not affine
        return None
    if _is_widened(relation, (cell, stmt.expr)):
        return None
    if not set(relation.get_var_names(isl.dim_type.param)) <= allowed:
        return None
    names = sorted(flow.free_names(rows))
    elsewhere = isl.Set(
        f"[{', '.join(names)}] -> {{ [q, v] : 0 <= q < {rows} and "
        "(q > 0 or v < 0 or v > 0) }"
    )
    relation, elsewhere = _align_both(relation, elsewhere)
    return relation.intersect_range(elsewhere).domain()


# }}}


# {{{ write disjointness, ordering, exactness, postcondition


def write_disjointness_facts(
    term: Term, owner: str, *, module: str | None = None, line: int | None = None
) -> list[Fact]:
    """One fact per writing statement: distinct instances write distinct cells.

    The cells of a ragged array are ``[r, j]``, which are distinct cells of
    its buffer while its layout keeps the rows apart; the fact of a ragged
    array whose layout the term rewrites rests on that layout's fact.
    """
    labels = instance_labels(term)
    sizes = flow.size_names(term)
    layouts = layout_fact_ids(term, owner, module=module, line=line)
    by_statement = {stmt.id: stmt for stmt in term.stmts}
    facts: list[Fact] = []
    for footprint in flow.footprints(term):
        if not footprint.touches_memory:
            continue
        stmt = by_statement[footprint.stmt]
        relation = footprint.relation
        identity = isl.Map.identity(relation.get_space().domain().map_from_set())
        collisions = flow.assume_sizes(
            relation.apply_range(relation.reverse()).subtract(
                identity.align_params(relation.get_space())
            ),
            sizes,
        )
        facts.append(
            Fact(
                id=fact_id(
                    "disjoint-writes",
                    owner,
                    module=module,
                    line=line,
                    detail=f"{stmt.id}:{footprint.array}",
                ),
                kind="disjoint-writes",
                statement=(
                    f"distinct instances of {stmt.id} write distinct cells of "
                    f"{footprint.array}"
                ),
                term=Empty(
                    collisions,
                    description=f"pairs of {stmt.id} instances writing the same cell",
                    labels=labels,
                ),
                status=Status.ASSUMED,
                provenance={
                    "statement": stmt.id,
                    "array": footprint.array,
                    "kind": footprint.kind,
                    **_unnarrowed(stmt),
                },
                where=stmt.where,
                owner=owner,
                rests_on=layouts.get(footprint.array, ()),
            )
        )
    return facts


def ordering_facts(
    term: Term,
    owner: str,
    where: str,
    *,
    module: str | None = None,
    line: int | None = None,
) -> list[Fact]:
    """One fact: the order the body was written in respects its own dependences.

    The dependence relation is *defined* from the footprints, so this is not a
    tautology about a cached analysis: it is the statement that the schedule the
    source implies is legal for the relation the source generates, which is the
    baseline a transformation is later checked against.
    """
    if not term.stmts:
        return []
    identifier = fact_id("ordering", owner, module=module, line=line)
    try:
        sizes = flow.size_names(term)
        schedule = flow.assume_sizes(flow.schedule_of(term), sizes)
        deps = flow.assume_sizes(flow.dependences(term), sizes)
    except Exception as exc:  # noqa: BLE001 - a term we cannot analyse is ASSUMED
        return [
            Fact(
                id=identifier,
                kind="ordering",
                statement="the source order runs every dependence forward in time",
                term=None,
                status=Status.ASSUMED,
                provenance={"reason": f"no dependence relation: {exc}"},
                where=where,
                owner=owner,
            )
        ]
    # A statement whose guard is only partly stated contributes dependences
    # from instances that write nothing; see _unnarrowed.
    provenance: dict[str, Any] = {"dependences": str(deps)}
    wide = {
        stmt.id: _unnarrowed(stmt)["unnarrowed"]
        for stmt in term.stmts
        if stmt.unnarrowed
    }
    if wide:
        provenance["unnarrowed"] = wide
    return [
        Fact(
            id=identifier,
            kind="ordering",
            statement="the source order runs every dependence forward in time",
            term=Monotone(
                schedule,
                deps,
                description="the source schedule is monotone on the dependences",
                labels=instance_labels(term),
            ),
            status=Status.ASSUMED,
            provenance=provenance,
            where=where,
            owner=owner,
        )
    ]


def reduction_facts(
    term: Term, owner: str, *, module: str | None = None, line: int | None = None
) -> list[Fact]:
    """One fact per reduction: the exactness class its accumulation is allowed."""
    facts: list[Fact] = []
    for stmt in term.stmts:
        for position, reduction in enumerate(reductions_in(stmt.expr)):
            written = stmt.assignee
            indices = ", ".join(render(i) for i in written.indices)
            target = f"{written.array}[{indices}]"
            facts.append(
                Fact(
                    id=fact_id(
                        "exactness",
                        owner,
                        module=module,
                        line=line,
                        detail=f"{stmt.id}:{position}",
                    ),
                    kind="exactness",
                    statement=(
                        f"the accumulation into {target} over "
                        f"{', '.join(reduction.inames)} is {reduction.exactness}"
                    ),
                    term=None,
                    status=Status.DECIDED,
                    decided_by="type",
                    provenance={
                        "exactness": reduction.exactness,
                        "op": reduction.op,
                        "inames": list(reduction.inames),
                    },
                    where=stmt.where,
                    owner=owner,
                )
            )
    return facts


def postcondition_id(
    owner: str, *, module: str | None = None, line: int | None = None
) -> str:
    """The id of the fact a kernel's return annotation becomes.

    One builder for it, because a program names the fact of each kernel it
    calls by this id (see :meth:`loopty.kernel.Program.facts`), and an id that
    drifted from the kernel's own would name a fact the ledger does not hold.
    It is keyed by the kernel's definition, ``owner`` with ``module`` and
    ``line`` (see :func:`lanky.ledger.fact_id`), so two kernels of one name
    in two modules have two ids: ``postcondition:spmv.scan@69``.
    """
    return fact_id("postcondition", owner, module=module, line=line)


def postcondition_facts(
    term: Term,
    owner: str,
    where: str,
    *,
    module: str | None = None,
    line: int | None = None,
) -> list[Fact]:
    """The return annotation as a fact, for whatever oracle can take it."""
    if term.post is None:
        return []
    return [
        Fact(
            id=postcondition_id(owner, module=module, line=line),
            kind="postcondition",
            statement=render(term.post),
            term=term.post,
            status=Status.ASSUMED,
            provenance={"kernel": term.name},
            where=where,
            owner=owner,
        )
    ]


# }}}


def facts_for(
    term: Term,
    owner: str = "",
    where: str = "",
    *,
    module: str | None = None,
    line: int | None = None,
) -> list[Fact]:
    """Every obligation ``term`` owes, in the order the rules generate them.

    ``owner`` is the decorated object's qualified name, which the ledger prints;
    ``where`` is the kernel's own ``file:line``, used by the facts that belong
    to the kernel as a whole rather than to one statement. ``module`` and
    ``line`` complete the kernel's definition, the module its file's path
    gives it and the line it is defined at, and every id is keyed by all
    three (:func:`lanky.ledger.fact_id`), so the ids are stable across runs
    and two kernels of one name in two modules do not share them.
    """
    owner = owner or term.name
    key = {"module": module, "line": line}
    return [
        *in_bounds_facts(term, owner, **key),
        *element_sort_facts(term, owner, **key),
        *layout_facts(term, owner, **key),
        *write_disjointness_facts(term, owner, **key),
        *ordering_facts(term, owner, where, **key),
        *reduction_facts(term, owner, **key),
        *postcondition_facts(term, owner, where, **key),
    ]
