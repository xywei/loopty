"""Schedules: loop transformations as casts, checked one at a time.

A transformation is untrusted. ``.split``, ``.tile``, ``.interchange``,
``.skew``, ``.tag`` and ``.realize`` apply the corresponding loopy transform and
then hand the result to a small checker, which asks isl two questions: is the
reindexing a bijection on statement instances, and is the new execution order
monotone on the dependence relation? Parallel inames (``g.*``, ``l.*``) carry no
order, so they are dropped from the order before the second question is asked.
This is the de Bruijn criterion applied to scheduling: any Python transformation
is admissible because its output is checked, not its code.

Three things have to be written down for those two questions to be askable.

*One instance space.* A statement instance is encoded as ``[s, x0, ..., x_{W-1}]``
with ``s`` the statement's position in the term and ``x_j`` the value of its
``j``-th coordinate, padded with zeros. All statements then live in one isl space,
so the dependence relation is a single map and not a union over pairs of spaces,
and ``loopty.oracle``'s primitives apply directly.

*A reindexing map per transformation.* Splitting replaces a coordinate by two
related ones, skewing shifts one by another, and interchanging and tagging change
no coordinate at all: they change only the order. Each transformation states its
map, and isl decides whether it is a bijection on the instances that exist.

*An order as a map into logical time.* The time of an instance is
``[c_0, i_0, c_1, i_1, ..., c_k]``: the loop values interleaved with constants
that place the statement among its siblings at each nesting level, which is the
standard way of making "the order the source is written in" a lexicographic
comparison. Dropping the parallel inames from that vector is what makes tagging
checkable: two instances that differ only in a parallel iname get the same time,
so a dependence between them is no longer ordered forward, and the cast is
rejected.

The order checked is also the order imposed: every accepted step sets loopy's
loop priority to the nest it just checked, so the generated code runs the nest
the checker approved rather than one loopy chose for itself.

A rejected cast raises :class:`IllegalCast` carrying ``witness``, the pair of
statement instances the transformation would reorder. That is the difference
between "tiling is illegal here" and "instance (0, 8) writes what instance
(1, 7) reads, and your tiling runs them the other way round".

Legal is not the same as buildable
----------------------------------

The two isl questions are about meaning, and meaning is all isl can see. A
transformation can preserve it and still be one the backend cannot generate
code for. loopy 2025.2 has two such limits, both of which the design's own
spmv device schedule walks into: it will not put a hardware axis (``g.*``,
``l.*``) inside a loop whose bound comes from an array, which is exactly what a
CSR inner loop is, and it will not generate a reduction whose inames are partly
parallel and partly sequential. Neither is a wrong verdict about the cast, and
neither used to be reported: the casts were all ``DECIDED`` and loopy then threw
during code generation, several steps away from the line that caused it.

So every accepted step is asked a third question, this one about the target
rather than about meaning, and its answer is a fact of kind ``buildable``
decided by ``loopy-target``. A schedule that fails it still exists, still
carries its ``DECIDED`` cast facts, and still reports what it is: the refusal
happens when something asks for code (see :class:`UnbuildableSchedule`), which
is the moment the claim actually matters.
"""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from dataclasses import dataclass, field
from typing import Any

import islpy as isl
import loopy as lp
import pymbolic.primitives as prim
from pymbolic.mapper.substitutor import substitute

from loopty import idx
from loopty import oracle as isl_oracle
from loopty.lower import (
    Lowering,
    _plain,
    lower_generic,
    reductions_of,
    walk,
)
from loopty.term import Access, Stmt, Term

__all__ = [
    "IllegalCast",
    "Schedule",
    "UnbuildableSchedule",
    "parallel_tag",
]

#: Iname tags that impose no order: two instances differing only in such an
#: iname may run in either order, or at the same time.
PARALLEL_TAG_PREFIXES = ("g.", "l.", "ilp")


def parallel_tag(tag: str) -> bool:
    """Does ``tag`` mark an iname as carrying no order?"""
    return any(tag.startswith(prefix) for prefix in PARALLEL_TAG_PREFIXES)


class IllegalCast(TypeError):
    """A transformation that would change the meaning of the program.

    ``witness`` is a ``(source_instance, sink_instance)`` pair from the isl
    oracle, and ``str()`` renders the explanation: which dependence, which two
    instances, and which way round the new order would run them. ``fact`` is the
    ``REFUTED`` ledger entry, carried on the exception because the schedule that
    would have held it was never built.
    """

    def __init__(self, message: str, witness: Any = None, fact: Any = None) -> None:
        super().__init__(message)
        self.witness = witness
        self.fact = fact


class UnbuildableSchedule(TypeError):
    """A schedule the checker accepts and the backend cannot generate code for.

    This is not an :class:`IllegalCast`: nothing about the meaning of the
    program is wrong, and the cast facts stay ``DECIDED``. What is wrong is the
    combination of the schedule and the target, so the exception carries
    ``reason`` in the words of the limit it hits, and ``fact``, the ``REFUTED``
    ``buildable`` entry the schedule has been carrying since the step that
    caused it.
    """

    def __init__(self, message: str, reason: str = "", fact: Any = None) -> None:
        super().__init__(message)
        self.reason = reason
        self.fact = fact


# {{{ the uniform instance space


@dataclass(frozen=True)
class _Layout:
    """Which coordinate each statement's instances are indexed by.

    ``coords[stmt_id]`` is the tuple of iname names that are the coordinates of
    that statement's instances, in a canonical order that only splitting changes.
    The loop *order* is kept separately, because interchanging loops renames no
    instance.
    """

    stmt_ids: tuple[str, ...]
    coords: dict[str, tuple[str, ...]]

    @property
    def width(self) -> int:
        """Number of coordinate dimensions, the deepest statement's nesting."""
        return max((len(c) for c in self.coords.values()), default=0)

    def dims(self, letter: str, suffix: str = "") -> str:
        """isl dimension names for this space, such as ``s, x0, x1``."""
        names = [f"s{suffix}"] + [f"{letter}{k}" for k in range(self.width)]
        return ", ".join(names)

    def index(self, stmt_id: str) -> int:
        """The position of a statement in the term."""
        return self.stmt_ids.index(stmt_id)


def _embed(domain: isl.Set, index: int, layout: _Layout) -> isl.Set:
    """Lift one statement's domain into the uniform instance space."""
    k = domain.dim(isl.dim_type.set)
    source = ", ".join(f"i{j}" for j in range(k))
    target = layout.dims("x")
    constraints = [f"s = {index}"]
    constraints += [f"x{j} = i{j}" for j in range(k)]
    constraints += [f"x{j} = 0" for j in range(k, layout.width)]
    lift = isl.Map(f"{{ [{source}] -> [{target}] : {' and '.join(constraints)} }}")
    return domain.apply(lift)


def _instances(term: Term, layout: _Layout, domains: dict[str, isl.Set]) -> isl.Set:
    """The set of all statement instances of a term, in one space."""
    out: isl.Set | None = None
    for stmt_id, domain in domains.items():
        lifted = _embed(domain, layout.index(stmt_id), layout)
        out = lifted if out is None else out.union(lifted)
    if out is None:  # pragma: no cover - a term always has a statement
        raise ValueError("a term with no statements has no instances")
    return out.coalesce()


def _coefficients(
    stmt_ids: Sequence[str], nests: dict[str, tuple[str, ...]]
) -> dict[str, tuple[int, ...]]:
    """Per-level sibling positions, the constants of the time vector.

    Two statements in the same loop interleave; two statements in different
    loops do not. The difference is recorded by walking the statements in
    program order and numbering, at each level, the distinct loop (or statement)
    that follows the prefix seen so far. That is the classical scattering
    construction, and it is what makes lexicographic comparison of time vectors
    reproduce the order of the source.
    """
    counters: dict[tuple, dict[tuple, int]] = {}
    out: dict[str, tuple[int, ...]] = {}
    for stmt_id in stmt_ids:
        prefix: tuple = ()
        coefficients: list[int] = []
        for iname in nests[stmt_id]:
            key = ("loop", iname)
            table = counters.setdefault(prefix, {})
            coefficients.append(table.setdefault(key, len(table)))
            prefix = (*prefix, key)
        table = counters.setdefault(prefix, {})
        coefficients.append(table.setdefault(("stmt", stmt_id), len(table)))
        out[stmt_id] = tuple(coefficients)
    return out


def _nests(
    layout: _Layout, order: Sequence[str], tags: dict[str, str]
) -> dict[str, tuple[str, ...]]:
    """Each statement's ordered loops, with the unordered ones left out."""
    return {
        stmt_id: tuple(
            iname
            for iname in order
            if iname in coords and not parallel_tag(tags.get(iname, ""))
        )
        for stmt_id, coords in layout.coords.items()
    }


def _time_map(
    layout: _Layout, order: Sequence[str], tags: dict[str, str]
) -> isl.Map:
    """The map from statement instances to logical time."""
    nests = _nests(layout, order, tags)
    coefficients = _coefficients(layout.stmt_ids, nests)
    depth = max((len(nest) for nest in nests.values()), default=0)
    length = 2 * depth + 1
    source = layout.dims("x")
    target = ", ".join(f"t{k}" for k in range(length))
    out: isl.Map | None = None
    for stmt_id in layout.stmt_ids:
        nest = nests[stmt_id]
        coords = layout.coords[stmt_id]
        constraints = [f"s = {layout.index(stmt_id)}"]
        used = 0
        for level, iname in enumerate(nest):
            constraints.append(f"t{2 * level} = {coefficients[stmt_id][level]}")
            constraints.append(f"t{2 * level + 1} = x{coords.index(iname)}")
            used = 2 * level + 2
        constraints.append(f"t{used} = {coefficients[stmt_id][len(nest)]}")
        constraints += [f"t{k} = 0" for k in range(used + 1, length)]
        piece = isl.Map(
            f"{{ [{source}] -> [{target}] : {' and '.join(constraints)} }}"
        )
        out = piece if out is None else out.union(piece)
    assert out is not None
    return out.coalesce()


def _step_map(
    layout_old: _Layout,
    layout_new: _Layout,
    constraints: dict[str, list[str]] | None = None,
    overridden: dict[str, set[str]] | None = None,
) -> isl.Map:
    """The reindexing map of one transformation.

    Coordinates that keep their name are equated automatically, which makes an
    interchange or a tag the identity and keeps the interesting part of a split
    or a skew down to the one or two constraints the caller states.
    """
    constraints = constraints or {}
    overridden = overridden or {}
    source = layout_old.dims("x")
    target = layout_new.dims("y", suffix="_")
    out: isl.Map | None = None
    for stmt_id in layout_old.stmt_ids:
        old = layout_old.coords[stmt_id]
        new = layout_new.coords[stmt_id]
        index = layout_old.index(stmt_id)
        pieces = [f"s = {index}", f"s_ = {index}"]
        for position, name in enumerate(new):
            if name in overridden.get(stmt_id, set()):
                continue
            if name in old:
                pieces.append(f"y{position} = x{old.index(name)}")
        pieces += [f"y{k} = 0" for k in range(len(new), layout_new.width)]
        pieces += [f"x{k} = 0" for k in range(len(old), layout_old.width)]
        pieces += constraints.get(stmt_id, [])
        piece = isl.Map(
            f"{{ [{source}] -> [{target}] : {' and '.join(pieces)} }}"
        )
        out = piece if out is None else out.union(piece)
    assert out is not None
    return out.coalesce()


# }}}


# {{{ dependences


@dataclass(frozen=True)
class _Dep:
    """One dependence: which instances, through which cell of which array."""

    kind: str  # "raw", "war", "waw"
    array: str
    source: str  # statement id
    sink: str
    source_indices: tuple[Any, ...]
    sink_indices: tuple[Any, ...]
    relation: isl.Map

    def verbs(self) -> tuple[str, str]:
        """How to say, in the rejection message, what each end did."""
        return {
            "raw": ("writes", "read"),
            "war": ("reads", "overwritten"),
            "waw": ("writes", "also written"),
        }[self.kind]


def _accesses(stmt: Stmt) -> list[tuple[str, str, tuple[Any, ...]]]:
    """Every array reference of a statement, as ``(kind, array, indices)``.

    The assignee is a write, and also a read when the statement accumulates.
    References inside the assignee's own subscripts are reads, as is everything
    in the right-hand side, including a reduction body: a reduction is part of
    one statement instance, so its accesses belong to that instance.
    """
    out: list[tuple[str, str, tuple[Any, ...]]] = []
    out.append(("write", stmt.assignee.array, tuple(stmt.assignee.indices)))
    if stmt.kind == "accumulate":
        out.append(("read", stmt.assignee.array, tuple(stmt.assignee.indices)))
    sources: list[Any] = [stmt.expr, *stmt.assignee.indices]
    for reduction in reductions_of(stmt.expr):
        sources.append(reduction.body)
    for source in sources:
        for node in walk(source):
            if isinstance(node, Access):
                out.append(("read", node.array, tuple(node.indices)))
            elif isinstance(node, prim.Subscript) and isinstance(
                node.aggregate, prim.Variable
            ):
                index = node.index
                indices = index if isinstance(index, tuple) else (index,)
                out.append(("read", node.aggregate.name, tuple(indices)))
    return out


def _index_text(expr: Any, renaming: dict[str, Any], allowed: set[str]) -> str | None:
    """An index expression as isl text, or ``None`` when it has to be widened.

    A subscript inside a subscript (``x[col[r, j]]``), a reduction iname that is
    not a coordinate of the statement, or any other non-affine term has no isl
    form. Returning ``None`` means "this access reaches an unknown cell of that
    axis", and the caller then leaves the axis unconstrained, which widens the
    footprint and so can only add dependences, never drop one.
    """
    try:
        plain = substitute(_plain(expr), renaming)
    except Exception:  # pragma: no cover - defensive against exotic terms
        return None
    if not idx.is_affine(plain):
        return None
    free = set(idx.size_params([plain]))
    if not free <= allowed:
        return None
    try:
        return idx.isl_expr(plain)
    except Exception:  # pragma: no cover - is_affine already ruled this out
        return None


def _dependences(
    term: Term,
    layout: _Layout,
    instances: isl.Set,
    before: isl.Map,
    params: set[str],
) -> tuple[_Dep, ...]:
    """The dependence relation, defined from the term's footprints.

    Two instances depend on each other when they touch the same cell of the same
    array, at least one of them writes it, and the original order runs one before
    the other. Nothing is declared: this is the definition, evaluated by isl.
    """
    source_dims = layout.dims("x")
    target_dims = layout.dims("y", suffix="_")
    out: list[_Dep] = []
    for a in term.stmts:
        renaming_a = {
            iname: prim.Variable(f"x{k}")
            for k, iname in enumerate(layout.coords[a.id])
        }
        allowed_a = {f"x{k}" for k in range(len(layout.coords[a.id]))} | params
        for b in term.stmts:
            renaming_b = {
                iname: prim.Variable(f"y{k}")
                for k, iname in enumerate(layout.coords[b.id])
            }
            allowed_b = {f"y{k}" for k in range(len(layout.coords[b.id]))} | params
            for a_kind, a_array, a_indices in _accesses(a):
                for b_kind, b_array, b_indices in _accesses(b):
                    if a_array != b_array:
                        continue
                    if a_kind == "read" and b_kind == "read":
                        continue
                    if len(a_indices) != len(b_indices):
                        continue
                    constraints = [
                        f"s = {layout.index(a.id)}",
                        f"s_ = {layout.index(b.id)}",
                    ]
                    for a_index, b_index in zip(a_indices, b_indices, strict=True):
                        left = _index_text(a_index, renaming_a, allowed_a)
                        right = _index_text(b_index, renaming_b, allowed_b)
                        if left is None or right is None:
                            continue
                        constraints.append(f"{left} = {right}")
                    relation = isl.Map(
                        f"{{ [{source_dims}] -> [{target_dims}] : "
                        f"{' and '.join(constraints)} }}"
                    )
                    relation = (
                        relation.intersect_domain(instances)
                        .intersect_range(instances)
                        .intersect(before)
                    )
                    if relation.is_empty():
                        continue
                    kind = {
                        ("write", "read"): "raw",
                        ("read", "write"): "war",
                        ("write", "write"): "waw",
                    }[a_kind, b_kind]
                    out.append(
                        _Dep(
                            kind=kind,
                            array=a_array,
                            source=a.id,
                            sink=b.id,
                            source_indices=a_indices,
                            sink_indices=b_indices,
                            relation=relation.coalesce(),
                        )
                    )
    return tuple(out)


def _union(relations: Iterable[isl.Map]) -> isl.Map | None:
    """Union of dependence relations, or ``None`` when there are none."""
    out: isl.Map | None = None
    for relation in relations:
        out = relation if out is None else out.union(relation)
    return None if out is None else out.coalesce()


def _cross_check(term: Term, mine: isl.Map | None) -> tuple[isl.Map | None, str]:
    """Reconcile the dependences computed here with ``loopty.flow``'s.

    The definition of the dependence relation belongs in ``flow.py``, and this
    module computes its own only because it needs each dependence separately, to
    be able to say *which* array cell a rejected schedule would reorder; the
    union is what the verdict is actually about. The two are compared, and the
    schedule is checked against the union of both, so that a dependence either of
    them finds is one the schedule has to respect. A disagreement is recorded in
    the provenance of every cast fact rather than left for a reader to notice.
    """
    try:
        from loopty.flow import dependences
    except ImportError:  # pragma: no cover - flow.py is always present
        return mine, "loopty.flow is not importable"
    try:
        theirs = dependences(term)
    except NotImplementedError:
        return mine, "loopty.flow.dependences is not implemented yet"
    except Exception as exc:  # pragma: no cover - depends on the other wave
        return mine, f"loopty.flow.dependences raised {type(exc).__name__}: {exc}"
    if theirs is None:  # pragma: no cover - defensive
        return mine, "loopty.flow.dependences returned nothing"
    if mine is None:
        return (
            None if theirs.is_empty() else theirs,
            "no dependences here; loopty.flow found "
            + ("none either" if theirs.is_empty() else "some"),
        )
    if theirs.dim(isl.dim_type.in_) != mine.dim(isl.dim_type.in_):
        return mine, (
            "loopty.flow uses an instance space of "
            f"{theirs.dim(isl.dim_type.in_)} dimensions, this one uses "
            f"{mine.dim(isl.dim_type.in_)}; not compared"
        )
    if mine.is_equal(theirs):
        return mine, "agree with loopty.flow"
    return mine.union(theirs), "differ from loopty.flow; checked against the union"


# }}}


# {{{ what the target can build


def _constrains(domain: isl.Set, position: int, param: int) -> bool:
    """Does parameter ``param`` bound set dimension ``position`` of ``domain``?

    Read off the constraints rather than inferred by projecting: a constraint
    that mentions both the dimension and the parameter is one in which the
    parameter helps bound it, and nothing else counts. Projecting the other
    dimensions out instead would answer yes far too often, because eliminating
    a dimension leaves behind what its own existence implied. For the spmv
    reduction over ``[r, j]`` with ``0 <= j < nl_cnt_r``, projecting ``j`` out
    leaves ``nl_cnt_r >= 1``, which would make the *row* loop look as if its
    bound came from data.
    """
    found = False

    def visit_basic_set(basic_set: isl.BasicSet) -> None:
        nonlocal found
        for constraint in basic_set.get_constraints():
            if constraint.get_coefficient_val(isl.dim_type.set, position).is_zero():
                continue
            if not constraint.get_coefficient_val(
                isl.dim_type.param, param
            ).is_zero():
                found = True
                return

    domain.foreach_basic_set(visit_basic_set)
    return found


def _data_dependent_in(
    domain: isl.Set, inames: Sequence[str], sizes: set[str]
) -> set[str]:
    """The inames of one domain whose bound comes from data rather than a size."""
    params = list(domain.get_var_names(isl.dim_type.param))
    positions = [k for k, name in enumerate(params) if name not in sizes]
    if not positions:
        return set()
    out: set[str] = set()
    for position in range(min(domain.dim(isl.dim_type.set), len(inames))):
        if any(_constrains(domain, position, param) for param in positions):
            out.add(inames[position])
    return out


def data_dependent_inames(term: Term) -> frozenset[str]:
    """Loop variables whose extent is read out of an array.

    A ragged fiber is the case that matters: the bound of ``j`` in
    ``val.dom[r]`` is ``cnt[r]``, which the tracer reflects into an isl
    parameter that is not one of the term's sizes. Such a loop is a perfectly
    ordinary loop to isl and to C, and an impossible one to put on a hardware
    axis, because the number of work items is not known when the kernel is
    launched.
    """
    sizes = set(term.sizes)
    out: set[str] = set()
    for stmt in term.stmts:
        out |= _data_dependent_in(stmt.domain, stmt.inames, sizes)
        for reduction in reductions_of(stmt.expr):
            out |= _data_dependent_in(
                reduction.domain, (*stmt.inames, *reduction.inames), sizes
            )
    return frozenset(out)


def _unbuildable_reason(draft: _Draft) -> str | None:
    """Why loopy could not generate code for this draft, or ``None``.

    Two limits of loopy 2025.2, both measured rather than guessed: a device run
    of the design's spmv schedule fails on the first, and applying loopy's own
    ``split_reduction_outward`` remedy to it then fails on the second.
    """
    parallel = {name for name, tag in draft.tags.items() if parallel_tag(tag)}
    inside = sorted(parallel & draft.data_dependent)
    if inside:
        names = ", ".join(inside)
        return (
            f"the parallel tag on {names} sits inside a loop whose bound comes "
            "from an array (a ragged fiber), and loopy will not put a hardware "
            "axis in a domain with a data-dependent parameter. Parallelize an "
            "enclosing loop with a size known at launch instead, such as the "
            "rows of a CSR product"
        )
    by_accumulation: dict[str, list[str]] = {}
    for iname, accumulated in draft.reductions.items():
        by_accumulation.setdefault(accumulated, []).append(iname)
    for accumulated, inames in sorted(by_accumulation.items()):
        tagged = sorted(name for name in inames if name in parallel)
        untagged = sorted(name for name in inames if name not in parallel)
        if tagged and untagged:
            return (
                f"the reduction into {accumulated} runs over {', '.join(tagged)} "
                f"in parallel and {', '.join(untagged)} in sequence, and loopy "
                "generates code only for a reduction whose inames are all one or "
                "all the other. loopty has no split_reduction transform to offer "
                "as the remedy"
            )
    return None


# }}}


def _term_of(obj: Any) -> Term:
    """The term of a kernel, a schedule, or a term."""
    if isinstance(obj, Term):
        return obj
    term = getattr(obj, "term", None)
    if term is None and hasattr(obj, "trace"):
        term = obj.trace()
    if isinstance(term, Term):
        return term
    raise TypeError(f"{obj!r} is not a kernel and has no term to schedule")


def _with_priority(kernel: Any, orders: Iterable[Sequence[str]]) -> Any:
    """Set loopy's loop priority to exactly these nests.

    Replacing rather than adding is the point: ``lp.prioritize_loops``
    accumulates, and after an interchange the old priority contradicts the new
    one, which loopy can only report as an unschedulable kernel.
    """
    entry = kernel.default_entrypoint
    priorities = frozenset(tuple(order) for order in orders if len(order) > 1)
    return kernel.with_kernel(entry.copy(loop_priority=priorities))


@dataclass
class _Draft:
    """The state one transformation builds, before the checker sees it."""

    coords: dict[str, tuple[str, ...]]
    order: list[str]
    tags: dict[str, str]
    kernel: Any
    reductions: dict[str, str] = field(default_factory=dict)
    constraints: dict[str, list[str]] = field(default_factory=dict)
    overridden: dict[str, set[str]] = field(default_factory=dict)
    reassoc: set[str] = field(default_factory=set)
    #: Loop variables whose extent comes from an array; see
    #: :func:`data_dependent_inames`. A split passes the property to both halves.
    data_dependent: set[str] = field(default_factory=set)


class Schedule:
    """A kernel plus the transformations applied to it, each one checked.

    ``Schedule(kernel, target="c")`` starts from the identity schedule. Every
    method returns a new schedule, so a rejected step leaves the previous one
    intact. ``.facts()`` yields the cast facts, ``DECIDED`` for the steps isl
    accepted and ``REFUTED`` with a witness for one it did not.

    ``sizes`` is a hint, not a constraint: the checks are made with the size
    parameters free, and the hint is used only to print a witness with concrete
    numbers in it, and as the default example inputs for ``loopty run``.
    """

    def __init__(
        self,
        kernel: Any,
        target: str = "c",
        sizes: dict[str, int] | None = None,
    ) -> None:
        self._source = kernel
        self._term = _term_of(kernel)
        self._target = target
        self._sizes = dict(sizes or {})
        self._lowering: Lowering = lower_generic(self._term, target)
        self._kernel = self._lowering.kernel

        stmt_ids = tuple(stmt.id for stmt in self._term.stmts)
        self._layout = _Layout(
            stmt_ids=stmt_ids,
            coords={stmt.id: tuple(stmt.inames) for stmt in self._term.stmts},
        )
        self._domains = {stmt.id: _set_over(stmt) for stmt in self._term.stmts}
        self._instances = _instances(self._term, self._layout, self._domains)
        self._origin = self._instances
        self._origin_layout = self._layout

        order: list[str] = []
        for stmt in self._term.stmts:
            for iname in stmt.inames:
                if iname not in order:
                    order.append(iname)
        self._order = order
        self._tags: dict[str, str] = {}
        # A reduction iname is not a coordinate of the instance space: a whole
        # reduction happens inside one statement instance. Splitting or tagging
        # one therefore renames no instance and reorders no dependence; what it
        # changes is the order of the accumulation, which is a question about
        # exactness rather than about dependences.
        self._reductions: dict[str, str] = {}
        for stmt in self._term.stmts:
            for reduction in reductions_of(stmt.expr):
                for iname in reduction.inames:
                    self._reductions[iname] = stmt.assignee.array
        self._reassoc: frozenset[str] = frozenset()
        self._data_dependent = data_dependent_inames(self._term)
        self._history: tuple[str, ...] = ()
        #: Each step as ``(method, args, kwargs)``, so that :meth:`retarget` can
        #: replay it against another target and have every cast checked again.
        self._steps: tuple[tuple[str, tuple, dict], ...] = ()
        self._facts: tuple[Any, ...] = ()
        self._unbuildable: str | None = None
        self._examples: dict[str, Any] | None = None

        params = set()
        for domain in self._domains.values():
            params |= set(domain.get_var_names(isl.dim_type.param))
        self._params = params

        time = _time_map(self._layout, self._order, {})
        lex = isl.Map.lex_lt(time.get_space().range())
        before = (
            time.apply_range(lex)
            .apply_range(time.reverse())
            .intersect_domain(self._instances)
            .intersect_range(self._instances)
        )
        self._deps = _dependences(
            self._term, self._layout, self._instances, before, params
        )
        self._deps_total, self._flow_note = _cross_check(
            self._term, _union(dep.relation for dep in self._deps)
        )
        self._reindex = isl.Map.identity(
            self._instances.get_space().map_from_set()
        ).intersect_domain(self._instances)

    # {{{ plumbing

    def _clone(self) -> Schedule:
        """A shallow copy; every public method builds one rather than mutating."""
        other = object.__new__(Schedule)
        other.__dict__.update(self.__dict__)
        return other

    def _draft(self) -> _Draft:
        return _Draft(
            coords=dict(self._layout.coords),
            order=list(self._order),
            tags=dict(self._tags),
            kernel=self._kernel,
            reductions=dict(self._reductions),
            reassoc=set(self._reassoc),
            data_dependent=set(self._data_dependent),
        )

    @property
    def term(self) -> Term:
        """The term being scheduled."""
        return self._term

    @property
    def source(self) -> Any:
        """The object the schedule was built from (a kernel, or a term)."""
        return self._source

    @property
    def target(self) -> str:
        """The name of the loopy target, ``"c"`` or ``"opencl"``."""
        return self._target

    @property
    def sizes(self) -> dict[str, int]:
        """The size hint given at construction."""
        return dict(self._sizes)

    @property
    def kernel(self) -> Any:
        """The loopy kernel as transformed so far."""
        return self._kernel

    @property
    def lowering(self) -> Lowering:
        """The lowering the schedule started from."""
        return self._lowering

    @property
    def history(self) -> tuple[str, ...]:
        """The transformations applied, as they would be written in Python."""
        return self._history

    @property
    def order(self) -> tuple[str, ...]:
        """The loop nest, outermost first."""
        return tuple(self._order)

    @property
    def tags(self) -> dict[str, str]:
        """Iname tags applied so far."""
        return dict(self._tags)

    @property
    def reassociated(self) -> frozenset[str]:
        """Arrays whose accumulation has been marked reassociated."""
        return self._reassoc

    @property
    def buildable(self) -> tuple[bool, str]:
        """Can this target generate code for this schedule, and if not, why not?

        A pair, ``(ok, reason)``, with ``reason`` empty when it is buildable.
        The question is asked of every accepted step; see
        :func:`_unbuildable_reason` for the two limits it knows about.
        """
        return (self._unbuildable is None, self._unbuildable or "")

    def require_buildable(self) -> None:
        """Raise :class:`UnbuildableSchedule` unless code can be generated.

        Called by everything that is about to ask loopy for code, so that the
        refusal names the schedule and the limit rather than arriving as a
        ``LoopyError`` from inside code generation.
        """
        if self._unbuildable is None:
            return
        raise UnbuildableSchedule(
            f"{self!r} cannot be built for the {self._target} target: "
            f"{self._unbuildable}",
            reason=self._unbuildable,
            fact=next(
                (fact for fact in self._facts if fact.kind == "buildable"), None
            ),
        )

    def retarget(self, target: str) -> Schedule:
        """The same transformations, checked again against another target.

        A schedule is written against a target: the lowering it starts from has
        that target's dtypes and its code generator, and the question of what
        can be built is a question about that target. So retargeting is not a
        relabelling; it lowers the term again and replays every step, which
        re-checks every cast and re-asks the buildability question. The casts
        will answer the same way, because they are about meaning; the third
        question need not.

        This is what ``loopty run --target opencl`` uses on a file whose
        schedules are written for ``"c"``, rather than running them on C and
        reporting a device run that never happened.
        """
        if target == self._target:
            return self
        out = Schedule(self._source, target=target, sizes=dict(self._sizes))
        for method, args, kwargs in self._steps:
            out = getattr(out, method)(*args, **kwargs)
        if self._examples is not None:
            out = out.example(**self._examples)
        return out

    def example(self, **arrays: Any) -> Schedule:
        """Record example inputs for ``loopty run``; returns a new schedule."""
        other = self._clone()
        other._examples = dict(arrays)
        return other

    @property
    def examples(self) -> dict[str, Any] | None:
        """The example inputs recorded with :meth:`example`, if any."""
        return None if self._examples is None else dict(self._examples)

    def __repr__(self) -> str:
        steps = "".join(f".{step}" for step in self._history)
        return f"Schedule({self._term.name}, target={self._target!r}){steps}"

    # }}}

    # {{{ transformations

    def tag(self, **inames: str) -> Schedule:
        """Tag inames with target coordinates, such as ``r="g.0"``.

        A tag is a coordinate in the target's execution type, and the parallel
        ones are the interesting case: they remove the iname from the order, so
        tagging a loop that carries a dependence is rejected here rather than
        producing a race at run time.
        """
        draft = self._draft()
        for name, tag in inames.items():
            if name not in draft.reductions or not parallel_tag(tag):
                continue
            # Running the pieces of a reduction at the same time sums them in an
            # order the source did not write, which is a reassociation and needs
            # the accumulation's permission.
            accumulated = draft.reductions[name]
            if self._exactness_of(accumulated) == "exact":
                raise IllegalCast(
                    f"tag({name}={tag!r}) illegal: it would run the pieces of "
                    f"the accumulation into {accumulated} at the same time, "
                    "which reassociates an exact reduction",
                    witness=None,
                    fact=self._fact(
                        "exactness",
                        f"the accumulation into {accumulated} may be reassociated",
                        status="refuted",
                        witness=None,
                        detail=f"the accumulation into {accumulated} is exact",
                        position=len(self._history),
                    ),
                )
            draft.reassoc.add(accumulated)
        draft.tags.update(inames)
        draft.kernel = lp.tag_inames(draft.kernel, dict(inames))
        text = "tag(" + ", ".join(f"{k}={v!r}" for k, v in inames.items()) + ")"
        return self._commit(draft, text, ("tag", (), dict(inames)))

    def split(
        self,
        iname: str,
        factor: int,
        inner: str | None = None,
        outer: str | None = None,
    ) -> Schedule:
        """Split ``iname`` by ``factor`` into an outer and an inner iname."""
        inner = inner or f"{iname}_inner"
        outer = outer or f"{iname}_outer"
        if iname in self._reductions:
            return self._split_reduction(iname, factor, inner, outer)
        if iname not in self._order:
            raise ValueError(f"{iname!r} is not an iname of {self._term.name}")
        draft = self._draft()
        self._split_into(draft, iname, factor, inner, outer)
        draft.kernel = lp.split_iname(
            draft.kernel, iname, factor, inner_iname=inner, outer_iname=outer
        )
        text = f"split({iname}, {factor})"
        return self._commit(
            draft, text, ("split", (iname, factor), {"inner": inner, "outer": outer})
        )

    def _split_into(
        self, draft: _Draft, iname: str, factor: int, inner: str, outer: str
    ) -> None:
        """The bookkeeping of a split, shared with :meth:`tile`.

        The constraint relates the *new* coordinates, at their positions in the
        draft, to the *old* one at its position before this schedule step began.
        Tiling splits twice in one step, so the two sides are numbered in
        different layouts and reading both positions off the same tuple would
        silently mean something else.
        """
        if factor < 1:
            raise ValueError(f"a split factor must be positive, not {factor}")
        for stmt_id, coords in list(draft.coords.items()):
            if iname not in coords:
                continue
            was = self._layout.coords[stmt_id]
            if iname not in was:  # pragma: no cover - defensive
                raise ValueError(f"{iname!r} is not a coordinate of {stmt_id}")
            source = was.index(iname)
            position = coords.index(iname)
            draft.coords[stmt_id] = (
                *coords[:position],
                outer,
                inner,
                *coords[position + 1 :],
            )
            draft.constraints.setdefault(stmt_id, []).extend(
                [
                    f"y{position} * {factor} + y{position + 1} = x{source}",
                    f"0 <= y{position + 1} < {factor}",
                ]
            )
        position = draft.order.index(iname)
        draft.order[position : position + 1] = [outer, inner]
        if iname in draft.data_dependent:
            draft.data_dependent.discard(iname)
            draft.data_dependent.update((outer, inner))

    def _split_reduction(
        self, iname: str, factor: int, inner: str, outer: str
    ) -> Schedule:
        """Split a reduction iname: the instances are untouched, the sum is not.

        Splitting the reduced domain is the first half of realizing a reduction
        as a tree; on its own it changes nothing about exactness, because the
        pieces still run in order. Tagging the inner piece parallel is what
        reassociates, and that is checked in :meth:`tag`.
        """
        draft = self._draft()
        accumulated = draft.reductions.pop(iname)
        draft.reductions[inner] = accumulated
        draft.reductions[outer] = accumulated
        if iname in draft.data_dependent:
            draft.data_dependent.discard(iname)
            draft.data_dependent.update((inner, outer))
        draft.kernel = lp.split_iname(
            draft.kernel, iname, factor, inner_iname=inner, outer_iname=outer
        )
        return self._commit(
            draft,
            f"split({iname}, {factor})",
            ("split", (iname, factor), {"inner": inner, "outer": outer}),
        )

    def interchange(self, *inames: str) -> Schedule:
        """Reorder the named loops into the order given.

        With two names this is the familiar interchange; with more it is a
        permutation. The instances are untouched, so the bijection is trivial and
        the whole question is whether the dependences still run forward.
        """
        return self._reorder(
            inames, f"interchange({', '.join(inames)})", "interchange"
        )

    def prioritize(self, *inames: str) -> Schedule:
        """Alias of :meth:`interchange`, in loopy's vocabulary."""
        return self._reorder(
            inames, f"prioritize({', '.join(inames)})", "prioritize"
        )

    def _reorder(self, inames: Sequence[str], text: str, method: str) -> Schedule:
        unknown = [iname for iname in inames if iname not in self._order]
        if unknown:
            raise ValueError(f"not inames of {self._term.name}: {unknown}")
        draft = self._draft()
        positions = sorted(draft.order.index(iname) for iname in inames)
        for position, iname in zip(positions, inames, strict=True):
            draft.order[position] = iname
        return self._commit(draft, text, (method, tuple(inames), {}))

    def tile(
        self,
        first: str,
        second: str,
        first_factor: int,
        second_factor: int,
    ) -> Schedule:
        """Split two loops and interchange the pieces, tiling the nest.

        One cast, not three: the tiles are what the user asked for, so the
        witness of a rejection names the tiling and not the interchange inside
        it.
        """
        for iname in (first, second):
            if iname not in self._order:
                raise ValueError(f"{iname!r} is not an iname of {self._term.name}")
        draft = self._draft()
        names = {}
        for iname, factor in ((first, first_factor), (second, second_factor)):
            inner, outer = f"{iname}_inner", f"{iname}_outer"
            names[iname] = (outer, inner)
            self._split_into(draft, iname, factor, inner, outer)
            draft.kernel = lp.split_iname(
                draft.kernel, iname, factor, inner_iname=inner, outer_iname=outer
            )
        outer_first, inner_first = names[first]
        outer_second, inner_second = names[second]
        wanted = [outer_first, outer_second, inner_first, inner_second]
        positions = sorted(draft.order.index(iname) for iname in wanted)
        for position, iname in zip(positions, wanted, strict=True):
            draft.order[position] = iname
        text = f"tile({first},{second},{first_factor},{second_factor})"
        return self._commit(
            draft,
            text,
            ("tile", (first, second, first_factor, second_factor), {}),
        )

    def skew(self, iname: str, by: str, factor: int = 1) -> Schedule:
        """Skew ``iname`` by ``factor`` times ``by``, making tiling legal.

        The instances are renamed, so this is the one transformation whose
        bijectivity is a real question, and the reason a skew makes a rectangular
        tiling legal is visible in the map: the dependence vectors it adds to
        every instance are exactly what stops a tile boundary from running a sink
        before its source.
        """
        if iname not in self._order or by not in self._order:
            raise ValueError(f"not inames of {self._term.name}: {iname}, {by}")
        draft = self._draft()
        for stmt_id, coords in draft.coords.items():
            if iname not in coords or by not in coords:
                continue
            position, other = coords.index(iname), coords.index(by)
            draft.constraints.setdefault(stmt_id, []).append(
                f"y{position} = x{position} + {factor} * x{other}"
            )
            draft.overridden.setdefault(stmt_id, set()).add(iname)
        draft.kernel = _skew_kernel(draft.kernel, iname, by, factor)
        text = f"skew({iname}, by={by!r}" + (
            f", factor={factor})" if factor != 1 else ")"
        )
        return self._commit(
            draft, text, ("skew", (iname,), {"by": by, "factor": factor})
        )

    def realize(self, var: str, tree: bool = True) -> Schedule:
        """Realize an accumulation, optionally as a reduction tree.

        A tree reassociates, so the result's exactness class drops to
        ``reassoc`` and the fact records it; over an ``exact`` accumulation the
        cast is rejected instead, because ``exact`` is a request for the bits and
        a tree does not give them.
        """
        exactness = self._exactness_of(var)
        if exactness is None:
            raise ValueError(f"{var!r} is not accumulated by {self._term.name}")
        text = f"realize({var!r}, tree={tree})"
        if tree and exactness == "exact":
            fact = self._fact(
                "exactness",
                f"the accumulation into {var} may be reassociated",
                status="refuted",
                witness=None,
                detail=f"the accumulation into {var} is exact",
                position=len(self._history),
            )
            raise IllegalCast(
                f"{text} illegal: the accumulation into {var} is exact, and a "
                "reduction tree reassociates it; ask for the accumulation at "
                "'reassoc' if the bits may change",
                witness=None,
                fact=fact,
            )
        draft = self._draft()
        if tree:
            draft.reassoc.add(var)
        return self._commit(draft, text, ("realize", (var,), {"tree": tree}))

    def _exactness_of(self, var: str) -> str | None:
        """The exactness class of the accumulation into ``var``, if there is one."""
        for stmt in self._term.stmts:
            if stmt.assignee.array != var:
                continue
            reductions = reductions_of(stmt.expr)
            if reductions:
                return reductions[0].exactness
            if stmt.kind == "accumulate":
                return _element_exactness(self._term, var)
        return None

    # }}}

    # {{{ the checker

    def _commit(
        self,
        draft: _Draft,
        text: str,
        recipe: tuple[str, tuple, dict] | None = None,
    ) -> Schedule:
        """Check one transformation and return the schedule it produces.

        ``recipe`` is how the transformation would be written in Python, as
        ``(method, args, kwargs)``, kept so that :meth:`retarget` can replay it.
        """
        layout = _Layout(
            stmt_ids=self._layout.stmt_ids, coords=dict(draft.coords)
        )
        step = _step_map(
            self._layout, layout, draft.constraints, draft.overridden
        ).intersect_domain(self._instances)

        position = len(self._history)
        facts: list[Any] = []

        verdict = isl_oracle.is_bijective(step)
        facts.append(
            self._fact(
                "bijective",
                f"{text} renames the instances of {self._term.name} one for one",
                status="decided" if verdict.ok else "refuted",
                witness=verdict.witness,
                detail=verdict.detail,
                position=position,
            )
        )
        if not verdict.ok:
            raise IllegalCast(
                f"{text} illegal: the reindexing is not a bijection on the "
                f"instances of {self._term.name}; {verdict.detail}",
                witness=verdict.witness,
                fact=facts[-1],
            )

        reindex = self._reindex.apply_range(step)
        instances = step.range()
        time = _time_map(layout, draft.order, draft.tags)
        schedule = reindex.apply_range(time).intersect_domain(self._origin)

        total = self._deps_total
        overall = (
            isl_oracle.is_monotone(schedule, total)
            if total is not None
            else isl_oracle.Verdict(True, None, "no dependences to violate")
        )
        # Attribution costs one isl question per dependence, and is only needed
        # to explain a refusal, so it is asked only when there is one to explain.
        bad = None if overall.ok else self._first_violation(schedule)
        facts.append(
            self._fact(
                "monotone",
                f"the order after {text} runs every dependence of "
                f"{self._term.name} forward",
                status="decided" if bad is None and overall.ok else "refuted",
                witness=overall.witness if bad is None else bad[1],
                detail=overall.detail if bad is None else bad[2],
                position=position,
            )
        )
        if bad is not None or not overall.ok:
            witness = overall.witness if bad is None else bad[1]
            message = (
                self._render_violation(text, *bad)
                if bad is not None
                else f"{text} illegal: {overall.detail}"
            )
            raise IllegalCast(message, witness=witness, fact=facts[-1])

        other = self._clone()
        other._layout = layout
        other._instances = instances
        other._reindex = reindex
        other._order = list(draft.order)
        other._tags = dict(draft.tags)
        other._kernel = _with_priority(
            draft.kernel, _nests(layout, draft.order, draft.tags).values()
        )
        other._reassoc = frozenset(draft.reassoc)
        other._reductions = dict(draft.reductions)
        other._data_dependent = frozenset(draft.data_dependent)
        # The cast is legal; whether the target can build it is a separate
        # question, asked once per step and recorded either way.
        reason = self._unbuildable or _unbuildable_reason(draft)
        other._unbuildable = reason
        if reason is not None and self._unbuildable is None:
            facts.append(
                self._fact(
                    "buildable",
                    f"{self._target} code can be generated for "
                    f"{self._term.name} after {text}",
                    status="refuted",
                    witness=None,
                    detail=reason,
                    position=position,
                    oracle="loopy-target",
                )
            )
        for accumulated in sorted(set(draft.reassoc) - set(self._reassoc)):
            facts.append(
                self._fact(
                    "exactness",
                    f"the accumulation into {accumulated} is reassociated by "
                    f"{text}, so its result is compared at 'reassoc'",
                    status="decided",
                    witness=None,
                    detail=f"exactness of {accumulated} lowered to reassoc",
                    position=position,
                )
            )
        other._history = (*self._history, text)
        other._steps = (
            (*self._steps, recipe) if recipe is not None else self._steps
        )
        other._facts = (*self._facts, *facts)
        return other

    def _first_violation(
        self, schedule: isl.Map
    ) -> tuple[_Dep, tuple, str] | None:
        """The first dependence the new order runs backwards, with its witness.

        Checking dependence by dependence rather than on the union is what lets
        the message name the array cell: the pair of instances alone does not say
        which access made them dependent.
        """
        for dep in self._deps:
            verdict = isl_oracle.is_monotone(schedule, dep.relation)
            if verdict.ok:
                continue
            witness = self._witness(schedule, dep)
            return dep, witness, verdict.detail
        return None

    def _witness(self, schedule: isl.Map, dep: _Dep) -> tuple:
        """A concrete violating pair, with the size parameters instantiated.

        The verdict is decided with the sizes free; the witness is printed with
        them fixed, because "instance ``S[t=0, i=8]``" is a sentence and
        "instance ``S[t=0, i=n-8]``" is a puzzle. The hint given to the schedule
        is preferred, and isl chooses when there is none.
        """
        relation = dep.relation.subtract(
            isl.Map.identity(dep.relation.get_space().domain().map_from_set())
        )
        timed = relation.apply_domain(schedule).apply_range(schedule)
        lex = isl.Map.lex_lt(timed.get_space().domain())
        violating_times = timed.subtract(lex)
        inverse = schedule.reverse()
        bad = (
            violating_times.apply_domain(inverse)
            .apply_range(inverse)
            .intersect(relation)
        )
        if self._sizes:
            names = bad.get_var_names(isl.dim_type.param)
            fixed = bad
            for name, value in self._sizes.items():
                if name in names:
                    fixed = fixed.fix_val(
                        isl.dim_type.param,
                        names.index(name),
                        isl.Val.int_from_si(bad.get_ctx(), value),
                    )
            if not fixed.is_empty():
                bad = fixed
        n_params = bad.dim(isl.dim_type.param)
        n_in = bad.dim(isl.dim_type.in_)
        wrapped = bad.wrap()
        if n_params:
            wrapped = wrapped.move_dims(
                isl.dim_type.set, 0, isl.dim_type.param, 0, n_params
            )
        point = isl_oracle.sample_point(wrapped)
        if point is None:  # pragma: no cover - the map is known to be non-empty
            return ()
        params = dict(
            zip(
                bad.get_var_names(isl.dim_type.param),
                point[:n_params],
                strict=False,
            )
        )
        source = point[n_params : n_params + n_in]
        sink = point[n_params + n_in :]
        return (self._name_instance(source), self._name_instance(sink), params)

    def _name_instance(self, point: Sequence[int]) -> tuple[str, dict[str, int]]:
        """A uniform-space point read back as ``(statement id, coordinates)``."""
        stmt_id = self._origin_layout.stmt_ids[int(point[0])]
        coords = self._origin_layout.coords[stmt_id]
        return stmt_id, {
            iname: int(value)
            for iname, value in zip(coords, point[1:], strict=False)
        }

    def _render_violation(self, text: str, dep: _Dep, witness: tuple, _: str) -> str:
        """The rejection message: which instance, which cell, which way round.

        The sizes are part of the message, not decoration. The verdict is
        decided with them free, so the witness is one violating pair out of
        many, and which one isl picks depends on the ``sizes`` hint (and, when
        the hint makes the violating set empty, is isl's own choice instead).
        A reader comparing two runs of the same demo needs to see at which sizes
        the pair was read off, or the numbers look unstable.
        """
        if not witness:  # pragma: no cover - a violation always has a witness
            return f"{text} illegal: {dep.kind} on {dep.array} runs backwards"
        (source_id, source_coords), (sink_id, sink_coords), params = witness
        source_verb, sink_verb = dep.verbs()
        cell = _cell_text(dep.source_indices, source_coords, params)
        at = _sizes_text(params, self._sizes)
        return (
            f"{text} illegal: instance {_instance_text(source_id, source_coords)} "
            f"{source_verb} {dep.array}[{cell}] {sink_verb} by "
            f"{_instance_text(sink_id, sink_coords)} scheduled earlier{at}"
        )

    # }}}

    def _fact(
        self,
        kind: str,
        statement: str,
        status: str,
        witness: Any,
        detail: str,
        position: int,
        oracle: str = "isl",
    ) -> Any:
        """One ledger entry for one question about one step.

        ``oracle`` is who answered: ``isl`` for the two questions about meaning,
        ``loopy-target`` for the one about what the backend can generate.
        """
        from lanky.ledger import Fact, Status

        provenance: dict[str, Any] = {
            "oracle": oracle,
            "detail": detail,
            "dependences": self._flow_note,
            "target": self._target,
        }
        if witness:
            provenance["witness"] = witness
        return Fact(
            id=f"cast:{self._term.name}:{position}:{kind}",
            kind=kind,
            statement=statement,
            term=None,
            status=Status.REFUTED if status == "refuted" else Status.DECIDED,
            decided_by=oracle,
            provenance=provenance,
            where=self._term.stmts[0].where if self._term.stmts else "",
            owner=self._term.name,
        )

    def facts(self) -> tuple:
        """The cast facts accumulated by the transformations applied so far."""
        return self._facts


def _sizes_text(params: dict[str, int], hint: dict[str, int]) -> str:
    """`` at n=16, nx=16`` (`` hinted``), or nothing when there are no sizes."""
    if not params:
        return ""
    inner = ", ".join(f"{name}={value}" for name, value in sorted(params.items()))
    honoured = all(hint.get(name, value) == value for name, value in params.items())
    how = "as hinted" if hint and honoured else "isl's choice"
    return f" (at {inner}, {how})"


def _instance_text(stmt_id: str, coords: dict[str, int]) -> str:
    """``S[t=0, i=8]``."""
    inner = ", ".join(f"{name}={value}" for name, value in coords.items())
    return f"{stmt_id}[{inner}]"


def _cell_text(
    indices: Sequence[Any], coords: dict[str, int], params: dict[str, int]
) -> str:
    """The array cell an instance touches, evaluated where it can be."""
    from pymbolic import evaluate

    context = {**params, **coords}
    out = []
    for index in indices:
        try:
            out.append(str(evaluate(_plain(index), context)))
        except Exception:
            out.append(str(index))
    return ", ".join(out)


def _set_over(stmt: Stmt) -> isl.Set:
    """A statement's domain with its set dimensions named after its inames."""
    domain = stmt.domain
    for k, iname in enumerate(stmt.inames):
        domain = domain.set_dim_name(isl.dim_type.set, k, iname)
    return domain


def _element_exactness(term: Term, name: str) -> str:
    """The exactness class of an array's element type."""
    for param, typ in term.params:
        if param != name:
            continue
        dtype = getattr(typ, "dtype", typ)
        try:
            from lanky.prelude import exactness_of

            return exactness_of(dtype)
        except Exception:
            break
    return "approx"


def _skew_kernel(kernel: Any, iname: str, by: str, factor: int) -> Any:
    """Apply a skew to the loopy kernel with ``lp.map_domain``.

    loopy takes the reindexing as an isl map, which is the same object the
    checker reasons about, so the two cannot drift apart. isl will not let one
    map mention a name twice, so every iname of the domain is renamed on the way
    through and renamed back afterwards.
    """
    entry = kernel.default_entrypoint
    domain = entry.get_inames_domain(frozenset([iname, by]))
    names = list(domain.get_var_names(isl.dim_type.set))
    params = list(domain.get_var_names(isl.dim_type.param))
    fresh = [f"{name}__skew" for name in names]
    constraints = []
    for old, new in zip(names, fresh, strict=True):
        if old == iname:
            constraints.append(f"{new} = {old} + {factor} * {by}")
        else:
            constraints.append(f"{new} = {old}")
    head = f"[{', '.join(params)}] -> " if params else ""
    text = (
        f"{head}{{ [{', '.join(names)}] -> [{', '.join(fresh)}] : "
        f"{' and '.join(constraints)} }}"
    )
    # loopy refuses to remap an iname a loop priority mentions. The priority is
    # reinstated by the caller once the new nest has been checked, so dropping it
    # here loses nothing.
    kernel = kernel.with_kernel(entry.copy(loop_priority=frozenset()))
    # A ``BasicMap``, and it has to be one: loopy's ``_find_aff_subst_from_map``
    # refuses anything else. loopy then asks it whether it is bijective, which
    # islpy 2025 answers by converting it to a Map and warns that it will stop
    # doing so in 2026. There is no spelling of this call that avoids the
    # warning from here; it is filtered in pyproject and recorded in
    # ``docs/loopy-notes.md``, and it is the second reason for the islpy pin.
    kernel = lp.map_domain(kernel, isl.BasicMap(text))
    for old, new in zip(names, fresh, strict=True):
        kernel = lp.rename_iname(kernel, new, old)
    return kernel
