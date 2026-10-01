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

from collections.abc import Sequence
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
    facts say the term keeps it so.
    """
    types = dict(term.params)
    sizes = flow.size_names(term)
    sorts = _element_sort_ids(term, owner, module=module, line=line)
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
                rests_on = tuple(
                    dict.fromkeys(
                        sort_id
                        for place in places
                        for name in _typed_by(place[0], types)
                        for sort_id in sorts.get(name, ())
                    )
                )
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
    writes into that array. Anything else is ``assumed``, with the reason.
    """
    types = dict(term.params)
    types.update(term.temporaries)
    sizes = flow.size_names(term)
    reflections = term.reflections
    sorts = _element_sort_ids(term, owner, module=module, line=line)
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
                    rests_on=tuple(
                        dict.fromkeys(
                            sort_id
                            for name in _typed_by((value,), types)
                            for sort_id in sorts.get(name, ())
                        )
                    ),
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


# {{{ write disjointness, ordering, exactness, postcondition


def write_disjointness_facts(
    term: Term, owner: str, *, module: str | None = None, line: int | None = None
) -> list[Fact]:
    """One fact per writing statement: distinct instances write distinct cells."""
    labels = instance_labels(term)
    sizes = flow.size_names(term)
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
        *write_disjointness_facts(term, owner, **key),
        *ordering_facts(term, owner, where, **key),
        *reduction_facts(term, owner, **key),
        *postcondition_facts(term, owner, where, **key),
    ]
