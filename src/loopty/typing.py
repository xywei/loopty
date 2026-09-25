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
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import islpy as isl
import pymbolic.primitives as prim
from lanky.ledger import Fact, Status
from lanky.prelude import FinType
from lanky.terms import render, structurally_equal

from loopty import flow
from loopty.oracle import Empty, Monotone, Subset
from loopty.term import ArrType, Term
from loopty.trace import reductions_in

__all__ = [
    "facts_for",
    "in_bounds_facts",
    "instance_labels",
    "ordering_facts",
    "postcondition_facts",
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


def in_bounds_facts(term: Term, owner: str) -> list[Fact]:
    """One fact per array access: the cells it reaches are cells the array has.

    The fact's id names the access, ``S1:read:x[r - 1]``, and not the domain it
    runs over, while the collector can list one access over several: the body's
    and the guard's, a reduction's, and the offsets every ragged access reads
    through. The ledger keeps one fact per id, so two facts for one access
    would leave the later in place of the earlier, whatever the earlier said:
    ``x[r - 1]`` read directly and again in a sum over ``q < r`` would be
    reported in bounds from the sum, where ``r >= 1``, with the direct read of
    ``x[-1]`` gone from the ledger. The domains of one access are therefore
    gathered into one obligation, about the union of the cells they reach.
    """
    types = dict(term.params)
    sizes = flow.size_names(term)
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
            text = f"{array}[{', '.join(render(i) for i in indices)}]"
            listed.setdefault((array, kind, text), []).append(
                (tuple(indices), inames, domain)
            )
        for (array, kind, text), places in listed.items():
            arrtype = types[array]
            indices = places[0][0]
            identifier = f"{owner}:in-bounds:{stmt.id}:{kind}:{text}"
            reasons = [
                _justified_by_type(place[0], arrtype, types) for place in places
            ]
            reason = reasons[0] if None not in reasons else None
            if reason is not None:
                facts.append(
                    Fact(
                        id=identifier,
                        kind="in-bounds",
                        statement=f"{text} is in bounds by type ({reason})",
                        term=None,
                        status=Status.DECIDED,
                        decided_by="type",
                        provenance={"rule": "index type", "reason": reason},
                        where=stmt.where,
                        owner=owner,
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
                        statement=f"{text} is in bounds",
                        term=None,
                        status=Status.ASSUMED,
                        provenance={"reason": f"no isl form: {exc}"},
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
                        statement=f"{text} is in bounds",
                        term=None,
                        status=Status.ASSUMED,
                        provenance={
                            "reason": (
                                "the index is not quasi-affine and its type does "
                                "not bound it, so isl is not asked"
                            )
                        },
                        where=stmt.where,
                        owner=owner,
                    )
                )
                continue
            facts.append(
                Fact(
                    id=identifier,
                    kind="in-bounds",
                    statement=(
                        f"{text} is in bounds for every instance of {stmt.id}"
                    ),
                    term=Subset(
                        reached,
                        cells,
                        description=f"cells {text} reaches are cells {array} has",
                        labels=tuple(f"a{k}" for k in range(len(indices))),
                    ),
                    status=Status.ASSUMED,
                    provenance={"access": text, "statement": stmt.id},
                    where=stmt.where,
                    owner=owner,
                )
            )
    return facts


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


# {{{ write disjointness, ordering, exactness, postcondition


def write_disjointness_facts(term: Term, owner: str) -> list[Fact]:
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
                id=f"{owner}:disjoint-writes:{stmt.id}:{footprint.array}",
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
                },
                where=stmt.where,
                owner=owner,
            )
        )
    return facts


def ordering_facts(term: Term, owner: str, where: str) -> list[Fact]:
    """One fact: the order the body was written in respects its own dependences.

    The dependence relation is *defined* from the footprints, so this is not a
    tautology about a cached analysis: it is the statement that the schedule the
    source implies is legal for the relation the source generates, which is the
    baseline a transformation is later checked against.
    """
    if not term.stmts:
        return []
    try:
        sizes = flow.size_names(term)
        schedule = flow.assume_sizes(flow.schedule_of(term), sizes)
        deps = flow.assume_sizes(flow.dependences(term), sizes)
    except Exception as exc:  # noqa: BLE001 - a term we cannot analyse is ASSUMED
        return [
            Fact(
                id=f"{owner}:ordering",
                kind="ordering",
                statement="the source order runs every dependence forward in time",
                term=None,
                status=Status.ASSUMED,
                provenance={"reason": f"no dependence relation: {exc}"},
                where=where,
                owner=owner,
            )
        ]
    return [
        Fact(
            id=f"{owner}:ordering",
            kind="ordering",
            statement="the source order runs every dependence forward in time",
            term=Monotone(
                schedule,
                deps,
                description="the source schedule is monotone on the dependences",
                labels=instance_labels(term),
            ),
            status=Status.ASSUMED,
            provenance={"dependences": str(deps)},
            where=where,
            owner=owner,
        )
    ]


def reduction_facts(term: Term, owner: str) -> list[Fact]:
    """One fact per reduction: the exactness class its accumulation is allowed."""
    facts: list[Fact] = []
    for stmt in term.stmts:
        for position, reduction in enumerate(reductions_in(stmt.expr)):
            written = stmt.assignee
            indices = ", ".join(render(i) for i in written.indices)
            target = f"{written.array}[{indices}]"
            facts.append(
                Fact(
                    id=f"{owner}:exactness:{stmt.id}:{position}",
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


def postcondition_facts(term: Term, owner: str, where: str) -> list[Fact]:
    """The return annotation as a fact, for whatever oracle can take it."""
    if term.post is None:
        return []
    return [
        Fact(
            id=f"{owner}:postcondition",
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


def facts_for(term: Term, owner: str = "", where: str = "") -> list[Fact]:
    """Every obligation ``term`` owes, in the order the rules generate them.

    ``owner`` is the decorated object's qualified name, which the ledger prints
    and which makes the fact ids stable across runs; ``where`` is the kernel's
    own ``file:line``, used by the facts that belong to the kernel as a whole
    rather than to one statement.
    """
    owner = owner or term.name
    return [
        *in_bounds_facts(term, owner),
        *write_disjointness_facts(term, owner),
        *ordering_facts(term, owner, where),
        *reduction_facts(term, owner),
        *postcondition_facts(term, owner, where),
    ]
