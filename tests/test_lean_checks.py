"""isl decides, Lean checks: loopty's facts stated for Lean from its own description.

Every set loopty builds keeps the text it was written from
(:mod:`loopty.description`), and every schedule step a record of its maps as
written; the statements Lean proves are built from those
(:mod:`loopty.obligations`). The first half of this module needs no Lean:
the text is kept and read back as isl reads it (a differential against isl's
membership at points), the statements say what they should, the facts this
track adds are there (mixed subscripts, definedness inside one kernel, a
schedule's assumptions). The second half runs Lean, and skips without it (or
fails where ``LANKY_LEAN_TEST_REQUIRED`` promises it): every fact of the
examples isl decides is proved, the counterexample of every refused cast is
proved false, and the printer is held to three safeguards, non-vacuity,
mutations Lean has to refuse exactly where isl does, and the printed
predicates against isl at points.
"""

from __future__ import annotations

import importlib.util
import itertools
import os
import sys
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import islpy as isl
import numpy as np
import pytest

pytest.importorskip("loopy")
lanky_obligations = pytest.importorskip(
    "lanky.obligations", reason="this lanky states no obligations for Lean"
)

#: Lean is left on for these, where it is installed (tests/conftest.py).
pytestmark = pytest.mark.lean

from lanky.lean import statement_of  # noqa: E402
from lanky.ledger import Status  # noqa: E402
from lanky.prelude import Int, Real  # noqa: E402
from lanky.terms import Var, evaluate, render  # noqa: E402

from loopty import Arr, Fin, Schedule, kernel  # noqa: E402
from loopty.description import (  # noqa: E402
    Unstatable,
    description_of,
    map_text,
    map_text_of,
    parse,
)
from loopty.obligations import (  # noqa: E402
    cast_obligation,
    lexicographically_before,
)
from loopty.oracle import decide  # noqa: E402
from loopty.schedule import IllegalCast  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
EXAMPLES = ROOT / "examples"
KERNELS = ROOT / "tests" / "kernels"


def load(path: Path) -> Any:
    """Import an example or a test kernel's file as a module of its own."""
    name = f"lean_checks_{path.parent.name}_{path.stem}"
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    sys.path.insert(0, str(path.parent))
    try:
        spec.loader.exec_module(module)
    finally:
        sys.path.remove(str(path.parent))
    return module


def kernels_of(module: Any) -> list[Any]:
    """The kernels a module defines, in the order it defines them."""
    from loopty.kernel import Kernel

    return [value for value in vars(module).values() if isinstance(value, Kernel)]


def stencil() -> Any:
    return load(EXAMPLES / "stencil_skew.py")


# {{{ the description, kept and read back


def test_a_statements_domain_keeps_the_text_isl_was_given() -> None:
    (stmt,) = stencil().jacobi.term.stmts
    described = description_of(stmt.domain)
    assert described is not None
    assert described.dims == ("t", "i")
    (disjunct,) = described.disjuncts
    # the guard as the trace wrote it, and the loops' bounds
    assert "(t + 1) < nt" in disjunct.constraints
    assert "0 <= i < nx" in disjunct.constraints
    assert set(disjunct.nonneg) == {"nt", "nx"}
    # a set isl computed from it has none: nothing is stated from one
    assert description_of(stmt.domain.coalesce()) is None
    assert description_of(stmt.domain.project_out(isl.dim_type.set, 1, 1)) is None


def test_a_call_renames_the_description_with_the_set() -> None:
    module = load(EXAMPLES / "composition.py")
    term = module.burgers_rhs.term
    for stmt in term.stmts:
        if stmt.id == "f.zeros":
            continue
        described = description_of(stmt.domain)
        assert described is not None, stmt.id
        assert described.renamings, stmt.id


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("0 <= i < n", "0 <= i and i < n"),
        ("(t + 1) < nt", "t + 1 < nt"),
        ("floord((a + b), 2) >= 0", "(a + b) // 2 >= 0"),
        ("((i) % 4) = 0", "i % 4 == 0"),
        ("i - j > -3", "i - j > -3"),
        ("(0 <= i) or i >= n", "0 <= i or i >= n"),
        ("2 * i + 1 <= n", "2*i + 1 <= n"),
        # isl's own spellings, which a map given to affine() uses
        ("b0 = 2 a0 + a1", "b0 == 2*a0 + a1"),
        ("2 b1 <= a1 < 2 b1 + 2", "2*b1 <= a1 and a1 < 2*b1 + 2"),
        ("b1 = floor((a1) / 4)", "b1 == a1 // 4"),
        ("-2 (i + 1) >= n", "-2*(i + 1) >= n"),
    ],
)
def test_the_reader_reads_the_text_loopty_writes(text: str, expected: str) -> None:
    assert render(parse(text)) == expected


@pytest.mark.parametrize(
    "text",
    [
        "exists e: i = 2e",
        "i * j < n",
        "i % n = 0",
        "i <",
        "i 2 < n",
        "floor(i / n) = 0",
    ],
)
def test_the_reader_refuses_what_loopty_does_not_write(text: str) -> None:
    with pytest.raises(Unstatable):
        parse(text)


def _member(obj: Any, point: dict[str, int], params: dict[str, int]) -> bool:
    """Whether isl's set holds the point at the parameters."""
    fixed = obj
    for k, name in enumerate(obj.get_var_names(isl.dim_type.param)):
        fixed = fixed.fix_val(
            isl.dim_type.param, k, isl.Val.int_from_si(obj.get_ctx(), params[name])
        )
    for k, name in enumerate(obj.get_var_names(isl.dim_type.set)):
        fixed = fixed.fix_val(
            isl.dim_type.set, k, isl.Val.int_from_si(obj.get_ctx(), point[name])
        )
    return not fixed.is_empty()


#: The most points of a box a domain is compared with isl at, per size.
_POINTS = 64


def _domains(path: Path) -> Iterator[tuple[str, Any, Any]]:
    """Every domain of a file's kernels and programs, a program's in its calls' names.

    Each with the term it belongs to, whose reflections say what a parameter
    a call renamed stands for.
    """
    from loopty.kernel import Program
    from loopty.trace import reductions_in

    module = load(path)
    # a program's, for the examples: tracing the test kernels' many programs
    # takes half a minute, and the examples' have the calls' renamings
    programs = [
        value
        for value in vars(module).values()
        if isinstance(value, Program) and path.parent == EXAMPLES
    ]
    for found in [*kernels_of(module), *programs]:
        term = found.term
        for stmt in term.stmts:
            yield f"{found.__name__}.{stmt.id}", stmt.domain, term
            if stmt.loop_domain is not None:
                yield f"{found.__name__}.{stmt.id} loops", stmt.loop_domain, term
            for reduction in reductions_in(stmt.expr):
                yield f"{found.__name__}.{stmt.id} sum", reduction.domain, term
        for call in term.calls:
            # the points a call of another kernel is made at, which its
            # arguments' facts are stated over (loopty.calls)
            yield f"{found.__name__} call {call.label}", call.domain, term
            if call.loop_domain is not None:
                label = f"{found.__name__} call {call.label} loops"
                yield label, call.loop_domain, term


@pytest.mark.parametrize(
    "path",
    [*sorted(EXAMPLES.glob("*.py")), *sorted(KERNELS.glob("*.py"))],
    ids=lambda path: f"{path.parent.name}/{path.name}",
)
def test_the_description_reads_back_as_isl_reads_it(path: Path) -> None:
    """The predicate built from loopty's text holds exactly where isl's set does.

    The differential of the printed predicates against isl's membership:
    every domain of every kernel and program, a program's after its calls'
    renamings, at every point of a box around it, at a few sizes, with each
    array cell a bound reads given a value of its own. The points are read
    by position, since a call renames isl's dimensions and the description
    keeps the callee's names until it is read.
    """
    rng = np.random.default_rng(0)
    checked = 0
    renamed = 0
    for label, domain, term in _domains(path):
        described = description_of(domain)
        if described is None and term.checks:
            # the check a program puts before a call, a statement of its own
            # built from isl's sets, which no fact is stated over for Lean
            continue
        assert described is not None, label
        dims = list(domain.get_var_names(isl.dim_type.set))
        names = list(domain.get_var_names(isl.dim_type.param))
        reflected = {**dict(described.reflected), **dict(term.reflected)}
        arrays: dict[str, Any] = {}
        for expr in reflected.values():
            aggregate = getattr(expr, "aggregate", None)
            if aggregate is not None and aggregate.name not in arrays:
                rank = len(expr.index) if isinstance(expr.index, tuple) else 1
                arrays[aggregate.name] = rng.integers(0, 4, size=(8,) * rank)
        coords = [Var(f"__x{k}") for k in range(len(dims))]
        predicate = described.predicate(coords)
        renamed += bool(described.renamings)
        box = list(itertools.product(range(-1, 4), repeat=len(dims)))
        if len(box) > _POINTS:
            # a seeded sample of a large box, which keeps the test quick
            box = [box[k] for k in sorted(rng.choice(len(box), _POINTS, replace=False))]
        for size in (0, 1, 3):
            sizes = {name: size + 2 for name in names if name not in reflected}
            for point in box:
                where = dict(zip(dims, point, strict=True))
                at = {**dict(zip(described.dims, point, strict=True)), **where, **sizes}
                params = dict(sizes)
                for name in names:
                    if name in reflected:
                        params[name] = int(evaluate(reflected[name], {**at, **arrays}))
                values = {f"__x{k}": v for k, v in enumerate(point)}
                expected = _member(domain, where, params)
                got = bool(evaluate(predicate, {**values, **sizes, **arrays}))
                assert got == expected, (label, point, params, render(predicate))
                checked += 1
    assert checked
    if path.parent == EXAMPLES and path.name in ("composition.py", "travel.py"):
        assert renamed, "a program's domains, after its calls' renamings"


# {{{ the statements


def test_an_in_bounds_fact_is_stated_from_the_kernels_text() -> None:
    from loopty.typing import in_bounds_facts

    term = stencil().jacobi.term
    facts = in_bounds_facts(term, "jacobi")
    fact = next(f for f in facts if "u[t + 1, i]" in f.statement)
    obligation = fact.term.lean_obligation()
    (part,) = obligation.parts
    statement = statement_of(part.term, "jacobi_S0", mathlib=False)
    assert [name for name, _ in statement.binders] == ["nt", "nx", "t", "i"]
    hypotheses = [text for _, text in statement.hypotheses]
    assert hypotheses == [
        "nt ≥ 0",
        "nx ≥ 0",
        "t + 1 < nt",
        "i > 0",
        "i + 1 < nx",
        "0 ≤ t",
        "t < nt",
        "0 ≤ i",
        "i < nx",
    ]
    assert statement.goal == "0 ≤ t + 1 ∧ t + 1 < nt ∧ 0 ≤ i ∧ i < nx"
    assert obligation.decider == "isl"


def test_a_question_isl_alone_is_given_says_so() -> None:
    from loopty.oracle import Empty

    obligation = Empty(isl.Set("{ [i] : 0 <= i < 3 }")).lean_obligation()
    said = "loopty states this kind of fact for isl only"
    assert obligation.declined.startswith(said)


def test_a_set_isl_computed_is_not_stated() -> None:
    from loopty.obligations import in_bounds_obligation

    (stmt,) = stencil().jacobi.term.stmts
    from loopty import flow

    term = stencil().jacobi.term
    cells = flow.cell_set(term.array_types["u"], stmt.assignee.indices)
    places = ((tuple(stmt.assignee.indices), stmt.inames, stmt.domain.coalesce()),)
    verdict = decide(_trivial())
    obligation = in_bounds_obligation(term, places, cells, "u", "u[t + 1, i]", verdict)
    assert "a set isl computed" in obligation.declined


def _trivial() -> Any:
    from loopty.oracle import Empty

    return Empty(isl.Set("{ [i] : 1 = 0 }"))


def test_the_time_vectors_are_compared_with_their_constants_settled() -> None:
    a, b = Var("a"), Var("b")
    assert lexicographically_before([0, a, 1], [1, b, 0]) is True
    assert lexicographically_before([1, a, 0], [0, b, 1]) is False
    assert render(lexicographically_before([0, a, 0], [0, b, 1])) == "a < b or a == b"


def test_a_step_keeps_its_map_as_written() -> None:
    schedule = stencil().skewed()
    skew, tile = schedule._records
    assert skew.text == "skew(i, by='t')"
    written = skew.maps["S0"]
    assert written.inputs == ("t", "i") and written.outputs == ("t", "i")
    assert written.constraints == ("b0 = a0", "b1 = a1 + 1 * a0")
    assert tile.old["S0"] == ("t", "i")
    assert tile.new["S0"] == ("t_outer", "t_inner", "i_outer", "i_inner")
    assert map_text_of(tile.isl_maps["S0"]) is tile.maps["S0"]


def test_an_affine_map_given_as_text_keeps_its_pieces() -> None:
    found = map_text(
        "{ S0[t, i] -> [a, b] : a = t + i and b = t - i; "
        "S1[t, i] -> [a, b] : a = t + i and b = t - i + 1 }"
    )
    assert set(found) == {"S0", "S1"}
    assert found["S1"].constraints == ("b0 = a0 + a1 and b1 = a0 - a1 + 1",)
    kept = map_text("{ [t, i] -> [t, b] : b = i + t }")[None]
    assert kept.constraints == ("b0 = a0", "b1 = a1 + b0")
    # Codex: pieces of one statement are one piecewise map, every piece kept
    piecewise = map_text(
        "{ [t, i] -> [a, b] : a = t and b = i and i < 4; "
        "[t, i] -> [a, b] : a = t + 1 and b = i and i >= 4 }"
    )[None]
    assert piecewise.constraints == (
        "((b0 = a0 and b1 = a1 and a1 < 4)) or ((b0 = a0 + 1 and b1 = a1 and a1 >= 4))",
    )
    with pytest.raises(Unstatable, match="name its loops differently"):
        map_text(
            "{ [t, i] -> [a, b] : a = t and b = i; [t, j] -> [a, b] : a = t and b = j }"
        )


def test_a_size_called_like_a_coordinate_stays_a_size() -> None:
    # Codex: a size a0 read as the map's first input, and a size called like
    # the reader's placeholder read as the coordinate it stands for
    from loopty.description import Description, Disjunct

    found = map_text("[a0] -> { [i] -> [j] : j = a0 - i }")[None]
    assert found.stems == ("a_", "b")
    relation = render(found.relation({"i": Var("x")}, {"j": Var("y")}))
    assert "a0" in relation and "x" in relation and "y" in relation, relation
    described = Description(
        dims=("i",), disjuncts=(Disjunct(("0 <= i < __loopty_dim0",)),)
    )
    assert render(described.predicate([Var("x")])) == "0 <= x and x < __loopty_dim0"


def test_a_cast_is_stated_one_part_per_dependence_class() -> None:
    schedule = stencil().skewed()
    monotone = schedule.facts()[-1]
    obligation = monotone.term.lean_obligation()
    labels = [part.label for part in obligation.parts]
    # jacobi writes u[t + 1, i] and reads u[t, i - 1] and u[t, i + 1]
    assert len(labels) == 5
    assert "u[t, i - 1] read by S0 after u[t + 1, i] is written by S0" in labels
    bijective = schedule.facts()[-2]
    names = [part.name for part in bijective.term.lean_obligation().parts]
    assert names == ["S0_total", "S0_single_valued", "S0_injective"]
    # isl's functional form of the tile is the witness of its totality
    (piece,) = bijective.term.lean_obligation().parts[0].hints.witness
    assert [render(value) for value in piece.values] == [
        "t_a1 // 8",
        "t_a1 + -8*(t_a1 // 8)",
        "i_a1 // 8",
        "i_a1 + -8*(i_a1 // 8)",
    ]


def test_a_refused_cast_is_stated_with_isls_pair() -> None:
    module = stencil()
    with pytest.raises(IllegalCast) as refused:
        Schedule(module.jacobi, target="c", sizes={"nt": 16, "nx": 16}).tile(
            "t", "i", 8, 8
        )
    obligation = refused.value.fact.term.lean_obligation()
    counter = obligation.counterexample
    assert counter is not None
    assert counter.text == "S0[t=0, i=8] and S0[t=1, i=7] at nt=16, nx=16"
    assert counter.point["t_a"] == 0 and counter.point["i_b"] == 7
    assert counter.point["t_outer_a1"] == 0 and counter.point["i_outer_a1"] == 1


def test_a_refused_cast_with_no_pair_is_not_offered_to_lean_as_a_claim() -> None:
    # A refused cast is checked at isl's pair; with no pair to check (a
    # refused bijection, or a reversal isl did not attribute), its parts are
    # not offered as a claim, which would read proved where isl refused.
    from dataclasses import replace

    from loopty.obligations import REFUTED_UNSTATED

    with pytest.raises(IllegalCast) as refused:
        Schedule(stencil().jacobi, sizes={"nt": 8, "nx": 8}).affine(
            "{ [t, i] -> [a, b] : a = t + i and b = 0 }"
        )
    fact = refused.value.fact
    assert fact.kind == "bijective" and fact.term.refuted
    assert fact.term.lean_obligation().declined == REFUTED_UNSTATED
    with pytest.raises(IllegalCast) as reversed_:
        Schedule(stencil().jacobi, sizes={"nt": 8, "nx": 8}).tile("t", "i", 4, 4)
    cast = reversed_.value.fact.term
    assert cast.refuted and cast.lean_obligation().counterexample is not None
    unattributed = replace(cast, refusal=None)
    assert unattributed.lean_obligation().declined == REFUTED_UNSTATED


def test_a_cast_after_a_substitution_is_left_to_isl() -> None:
    module = load(EXAMPLES / "fusion.py")
    schedule = next(
        value
        for value in vars(module).values()
        if isinstance(value, Schedule) and "substitute" in repr(value)
    )
    monotone = schedule.facts()[-1]
    assert "a substitution before this step" in cast_obligation(monotone.term).declined


def test_a_substitutions_definedness_is_isls_and_says_why() -> None:
    # Decided over the dependences isl computed of what the substitution
    # leaves: its term says it is isl's, and why Lean is not asked, where it
    # had none and read as a fact isl did not decide.
    module = load(EXAMPLES / "fusion.py")
    schedule = next(
        value
        for value in vars(module).values()
        if isinstance(value, Schedule) and "substitute" in repr(value)
    )
    (fact,) = [fact for fact in schedule.facts() if fact.kind == "definedness"]
    assert (fact.status, fact.decided_by) == (Status.DECIDED, "isl")
    assert "for isl only" in fact.term.lean_obligation().declined


def _pugh() -> Any:
    from lanky.prelude import Int
    from lanky.terms import Forall, LogicalAnd

    x, y = Var("x"), Var("y")
    hypotheses = (
        11 * x + 13 * y >= 27,
        11 * x + 13 * y <= 45,
        7 * x - 9 * y >= -10,
        7 * x - 9 * y <= 4,
    )
    return Forall(((x, Int), (y, Int)), False, LogicalAnd(hypotheses))


def test_isl_splits_a_part_by_the_rational_bounds_of_a_variable() -> None:
    """Pugh's example: rational points, no integer ones, which omega cannot tell.

    The split is on the variable the rational relaxation confines to the
    fewest integers, read off isl's set scaled by 64 and rounded outward.
    """
    from loopty.obligations import shadow_hints

    (split,) = shadow_hints(_pugh()).splits
    text = render(split)
    assert text in ("x == 0 or x == 1 or x == 2 or x == 3", "x == 1 or x == 2"), text


def test_isl_splits_a_disjunctive_hypothesis_on_the_disjuncts_it_allows() -> None:
    """The level that carries a dependence: the source order's disjuncts isl allows."""
    from lanky.prelude import Int
    from lanky.terms import Forall, LogicalAnd, LogicalOr

    from loopty.obligations import shadow_hints

    a, b = Var("a"), Var("b")
    source = LogicalOr((a < b, LogicalAnd((a == b, a + 1 < b))))
    term = Forall(((a, Int), (b, Int)), a <= b, LogicalAnd((a >= 0, source)))
    splits = shadow_hints(term).splits
    assert [render(split) for split in splits] == ["a < b"]


# }}}


# {{{ what this track adds to the ledger


@kernel
def gather(
    cell: Arr[Fin[nc], Fin[3], Fin[nv]],  # noqa: F821
    X: Arr[Fin[nv], Fin[2], Real],  # noqa: F821
    g: Arr[Fin[nc], Fin[3], Fin[2], Real],  # noqa: F821
):
    """Gather each cell's vertex coordinates, PyOP2's pack."""
    for c in g.dom:
        for k in g.dom[c]:
            for d in g.dom[c, k]:
                g[c, k, d] = X[cell[c, k], d]


def test_a_mixed_subscript_is_in_bounds_by_type_and_isl() -> None:
    """``X[cell[c, k], d]``: the vertex by its sort, the coordinate by isl (#17).

    It used to be ``assumed``: the by-type rule wanted every axis typed, and
    isl was not asked of an index that is not affine.
    """
    from loopty.oracle import IslOracle

    fact = next(f for f in gather.facts() if "X[cell[c, k], d]" in f.statement)
    assert fact.statement == (
        "X[cell[c, k], d] is in bounds for every instance of S0 "
        "(cell[c, k] : Fin(nv), by type)"
    )
    decided = IslOracle().establish(fact)
    assert decided.status is Status.DECIDED and decided.decided_by == "isl"
    obligation = fact.term.lean_obligation()
    (part,) = obligation.parts
    text = render(part.term)
    assert "0 <= cell[c, k]" in text and "cell[c, k] < nv" in text


def test_an_array_the_kernel_reads_where_it_wrote_it_has_a_definedness_fact() -> None:
    spmv = load(EXAMPLES / "spmv.py")
    facts = [f for f in spmv.scan.facts() if f.kind == "definedness"]
    assert [f.statement for f in facts] == [
        "every cell of off that S1 reads, S0 or S1 wrote before the read"
    ]
    assert decide(facts[0].term).ok
    obligation = facts[0].term.lean_obligation()
    (part,) = obligation.parts
    pieces = part.hints.witness
    assert [piece.disjunct for piece in pieces] == [0, 1]
    # jacobi reads what u held at its first time level: no fact
    assert not [f for f in stencil().jacobi.facts() if f.kind == "definedness"]


GUARDED = """
from __future__ import annotations

from lanky.prelude import Real
from loopty import Arr, Fin, kernel, when


@kernel
def guarded(x: Arr[Fin[n], Real], y: Arr[Fin[n], Real], z: Arr[Fin[n], Real]):
    for i in y.dom:
        with when(x[i] > 0):
            y[i] = 1.0
    for j in z.dom:
        z[j] = y[j]
"""


def test_a_write_under_a_guard_isl_cannot_state_shows_nothing_defined(
    tmp_path: Path,
) -> None:
    # S0 stores y[i] only where x[i] > 0, which isl cannot state, so its
    # domain is every i: an instance there is no witness that y[i] was
    # stored, and S1's read of y is not claimed defined (it was, wrongly,
    # decided by isl and proved by Lean from the wide domain).
    path = tmp_path / "guarded.py"
    path.write_text(GUARDED)
    module = load(path)
    first, _second = module.guarded.term.stmts
    assert first.unnarrowed
    assert not [f for f in module.guarded.facts() if f.kind == "definedness"]


def test_a_ragged_rows_bound_reads_back_whatever_the_loops_are_called() -> None:
    # The bound of row r + 1 is cnt[r + 1], in the access's loops; a loop
    # called like a dimension of the cell set (a0) was read as that
    # dimension, which stated cnt[a0 + 2] for Lean where isl had cnt[a0 + 1].
    import pymbolic.primitives as prim

    from loopty import flow
    from loopty.idx import Reflections
    from loopty.term import ArrType

    arrtype = ArrType(
        axes=(prim.Variable("n"), prim.Variable("cnt")),
        dtype=Real,
        ragged=(False, True),
    )
    for row in ("r", "a0", "a1"):
        indices = (Var(row) + 1, Var("j"))
        cells = flow.cell_set(arrtype, indices, reflections=Reflections())
        described = description_of(cells)
        predicate = described.predicate([Var("c0"), Var("c1")], others={row: Var(row)})
        assert f"c1 < cnt[{row} + 1]" in render(predicate), row


def _cell_sets(path: Path) -> Iterator[tuple[str, Any]]:
    """Every set of cells an in-bounds fact of a file states its goal over."""
    module = load(path)
    for found in kernels_of(module):
        for fact in found.facts():
            state = getattr(getattr(fact.term, "lean", None), "state", None)
            if fact.kind != "in-bounds" or state is None:
                continue
            for arg in getattr(state, "args", ()):
                if isinstance(arg, isl.Set) and description_of(arg) is not None:
                    yield f"{found.__name__}: {fact.statement}", arg


@pytest.mark.parametrize(
    "path",
    [*sorted(EXAMPLES.glob("*.py")), *sorted(KERNELS.glob("*.py"))],
    ids=lambda path: f"{path.parent.name}/{path.name}",
)
def test_the_cells_an_access_is_checked_against_read_back_as_isl_reads_them(
    path: Path,
) -> None:
    """The differential of the goals: an in-bounds fact's cells against isl's.

    The goal of an in-bounds part is the array's cells, read back from
    loopty's text; one weaker than isl's set would let Lean prove an access
    in bounds that is not. At every point of a box, at a few sizes, at a few
    values of the loops a ragged row's bound is written in, and with each
    array a bound reads given values of its own, the predicate holds exactly
    where isl's set does.
    """
    from loopty.obligations import _names

    rng = np.random.default_rng(0)
    checked = 0
    for label, cells in _cell_sets(path):
        described = description_of(cells)
        reflected = dict(described.reflected)
        params = list(cells.get_var_names(isl.dim_type.param))
        dims = list(cells.get_var_names(isl.dim_type.set))
        found: dict[str, int] = {}
        _names(tuple(reflected.values()), found)
        arrays = {
            name: rng.integers(0, 4, size=(8,) * rank)
            for name, rank in found.items()
            if rank
        }
        loops = [
            name
            for name, rank in found.items()
            if not rank and name not in params and name not in described.dims
        ]
        coords = [Var(f"__x{k}") for k in range(len(dims))]
        for size in (0, 1, 3):
            sizes = {name: size + 2 for name in params if name not in reflected}
            for values in itertools.product(range(3), repeat=len(loops)):
                at = dict(zip(loops, values, strict=True))
                predicate = described.predicate(coords, others=at)
                known = dict(sizes)
                for name, expr in reflected.items():
                    known[name] = int(evaluate(expr, {**at, **sizes, **arrays}))
                for point in itertools.product(range(-1, 4), repeat=len(dims)):
                    where = dict(zip(dims, point, strict=True))
                    expected = _member(cells, where, known)
                    values_at = {f"__x{k}": v for k, v in enumerate(point)}
                    got = bool(evaluate(predicate, {**values_at, **sizes, **arrays}))
                    assert got == expected, (label, point, known, render(predicate))
                    checked += 1
    assert checked


@kernel
def axpy(a: Real, x: Arr[Fin[n], Real], y: Arr[Fin[n], Real]):  # noqa: F821
    """``y := a x + y``."""
    for i in y.dom:
        y[i] = a * x[i] + y[i]


def test_an_assumption_is_a_fact_the_casts_rest_on() -> None:
    schedule = Schedule(axpy).assume("n % 4 == 0").split("i", 4)
    assumption, bijective, monotone = schedule.facts()
    assert assumption.kind == "assumption"
    assert assumption.status is Status.ASSUMED
    assert assumption.statement == "n % 4 == 0, which axpy is scheduled assuming"
    assert bijective.rests_on == (assumption.id,)
    assert monotone.rests_on == (assumption.id,)
    (part, *_rest) = monotone.term.lean_obligation().parts
    assert "n % 4 == 0" in render(part.term)
    assert schedule.history == ("assume('n % 4 == 0')", "split(i, 4)")


@pytest.mark.parametrize(
    ("condition", "why"),
    [
        ("m > 2", "m is not a size of the kernel"),
        ("n * n > 4", "isl states affine expressions"),
        ("n != 4", "an assumption is a comparison of sizes"),
    ],
)
def test_an_assumption_isl_cannot_state_is_refused(condition: str, why: str) -> None:
    with pytest.raises(ValueError, match=why):
        Schedule(axpy).assume(condition)


def test_a_call_that_breaks_the_assumption_is_refused() -> None:
    from loopty.executor import LoopyExecutor

    schedule = Schedule(axpy).assume("n % 4 == 0").split("i", 4)
    run = LoopyExecutor().run
    out = run(schedule, a=2.0, x=np.arange(8.0), y=np.ones(8))
    assert np.allclose(out["y"], 2.0 * np.arange(8.0) + 1)
    refusal = "assumes n % 4 == 0 of its sizes, and here n = 10"
    with pytest.raises(ValueError, match=refusal):
        run(schedule, a=2.0, x=np.arange(10.0), y=np.ones(10))


# }}}


# {{{ with Lean


def _without_lean(reason: str) -> None:
    if os.environ.get("LANKY_LEAN_TEST_REQUIRED"):
        pytest.fail(f"LANKY_LEAN_TEST_REQUIRED is set, but {reason}", pytrace=False)
    pytest.skip(reason)


@pytest.fixture(scope="module")
def lean(tmp_path_factory: pytest.TempPathFactory) -> Iterator[Any]:
    """A Lean oracle with an open session and a proof cache of its own."""
    from lanky.oracles.lean import LeanOracle, ProofCache

    if os.environ.get("LANKY_LEAN_DISABLE"):
        _without_lean("the Lean oracle is disabled by LANKY_LEAN_DISABLE")
    oracle = LeanOracle(timeout=float(os.environ.get("LANKY_LEAN_TEST_TIMEOUT", "120")))
    available, reason = oracle.availability()
    if not available:
        _without_lean(f"no Lean oracle here: {reason}")
    if not oracle.session.start():
        _without_lean(f"the Lean REPL could not be built: {oracle.session.error}")
    oracle.proofs = ProofCache(str(tmp_path_factory.mktemp("proofs")))
    yield oracle
    for session in oracle._pool:
        session.close()
    oracle.session.close()


def _isl_facts(path: Path) -> list[Any]:
    """Every fact of a file with a statement for Lean: kernels', programs', casts'."""
    from loopty.cli import collect
    from loopty.kernel import Program

    module = load(path)
    facts = []
    for found in kernels_of(module):
        facts.extend(found.facts())
    for value in vars(module).values():
        if isinstance(value, Program):
            facts.extend(value.facts())
    schedules, _ = collect(module, [])
    for schedule in schedules:
        facts.extend(schedule.facts())
    seen: dict[str, Any] = {}
    for fact in facts:
        seen.setdefault(fact.id, fact)
    return [
        fact
        for fact in seen.values()
        if callable(getattr(fact.term, "lean_obligation", None))
        and fact.status is not Status.REFUTED
    ]


#: What loopty states for isl alone so far, by the words of the statement: a
#: layout fact, an induction over the run; and a substitution's definedness
#: and the casts after it, whose dependences are isl's. The coverage report
#: lists them with the reason (docs/lean-checks.md), and an issue each.
NOT_STATED = (
    "stay inside their buffers",
    "substitute(",
    "has stored by the time",
)


@pytest.mark.parametrize("path", sorted(EXAMPLES.glob("*.py")), ids=lambda p: p.name)
def test_lean_proves_what_isl_decides_in_every_example(lean: Any, path: Path) -> None:
    unproved = []
    for fact in _isl_facts(path):
        result = lean.establish(fact)
        if result.status is not Status.PROVED:
            why = result.provenance.get("lean_reason") or result.provenance.get(
                "lean_declined"
            )
            unproved.append((fact.statement, why))
    unexpected = [
        (statement, why)
        for statement, why in unproved
        if not any(text in statement for text in NOT_STATED)
    ]
    assert not unexpected


def _refused_casts() -> list[tuple[str, IllegalCast]]:
    """Every cast the examples refuse, with isl's pair."""
    stencil_module = stencil()
    wave = load(EXAMPLES / "wavefront_acoustic.py")
    fusion = load(EXAMPLES / "fusion.py")
    sizes = {"nt": wave.NT, "nx": wave.NX}
    asks = [
        (
            "the stencil's rectangular tile",
            lambda: Schedule(
                stencil_module.jacobi, target="c", sizes={"nt": 16, "nx": 16}
            ).tile("t", "i", 8, 8),
        ),
        (
            "the wavefront's rectangular tile",
            lambda: Schedule(wave.acoustic, target="c", sizes=sizes).tile(
                "t", "i", 4, 8
            ),
        ),
        (
            "the space-first diamond",
            lambda: Schedule(wave.acoustic, target="c", sizes=sizes).affine(
                wave.DIAMOND_SPACE_FIRST
            ),
        ),
        (
            "the diamond's tile",
            lambda: wave.diamond_schedule().tile("a", "b", 4, 4),
        ),
        (
            "a fusion one step early",
            lambda: Schedule(fusion.burgers_rhs, sizes={"n": fusion.N}).fuse(
                "flux", "divergence"
            ),
        ),
    ]
    out = []
    for label, ask in asks:
        with pytest.raises(IllegalCast) as refused:
            ask()
        out.append((label, refused.value))
    return out


def test_run_proves_the_casts_and_shows_their_lean(
    lean: Any, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: Any
) -> None:
    # loopty run asks Lean about every cast isl decided as the steps were
    # taken: they read proved by lean, a LEAN line says what that means, and
    # --show-lean prints the theorems of the rows whose statement has the text.
    from lanky.oracles.lean import LeanOracle, ProofCache
    from lanky.plugins import registry

    from loopty.cli import main

    for oracle in registry.sorted_oracles():
        if isinstance(oracle, LeanOracle):
            monkeypatch.setattr(oracle, "proofs", ProofCache(str(tmp_path)))
    path = EXAMPLES / "stencil_skew.py"
    assert main(["run", str(path), "--show-lean", "tile("]) == 0
    out = capsys.readouterr().out
    rows = [line for line in out.splitlines() if "tile(t,i,8,8)" in line]
    assert any(line.startswith("proved") and " lean " in line for line in rows)
    assert not any(line.startswith("decided") for line in rows)
    assert "LEAN proved" in out
    shown = [line for line in out.splitlines() if line.startswith("LEAN jacobi at")]
    assert shown and all("tile(" in line for line in shown)
    assert "theorem Lanky." in out


def test_run_names_a_cast_lean_was_not_asked_about_when_verbose(
    lean: Any, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: Any
) -> None:
    # The casts after fusion's substitution are isl's alone: decided, and
    # with --verbose a NOT PROVED line says so, and why.
    from lanky.oracles.lean import LeanOracle, ProofCache
    from lanky.plugins import registry

    from loopty.cli import main

    for oracle in registry.sorted_oracles():
        if isinstance(oracle, LeanOracle):
            monkeypatch.setattr(oracle, "proofs", ProofCache(str(tmp_path)))
    path = EXAMPLES / "fusion.py"
    assert main(["run", str(path)]) == 0
    assert "Lean was not asked" not in capsys.readouterr().out
    assert main(["run", str(path), "--verbose"]) == 0
    out = capsys.readouterr().out
    assert "NOT PROVED burgers_rhs" in out
    assert "isl decided it; Lean was not asked: a substitution before this step" in out


def test_every_refused_cast_is_proved_false_at_isls_pair(lean: Any) -> None:
    for label, exc in _refused_casts():
        fact = lean.establish(exc.fact)
        said = fact.provenance.get("lean_counterexample", "")
        assert said.startswith("Lean confirmed the counterexample"), (label, said)
        assert fact.provenance["lean_refuted_by"] in ("decide", "omega"), label


class _Planted:
    """A fact's term with the statement it had and the counterexample taken away."""

    def __init__(self, parts: Any) -> None:
        self.parts = parts

    def lean_obligation(self) -> Any:
        from lanky.obligations import Obligation

        return Obligation(parts=self.parts, decider="isl")


def test_lean_fails_on_exactly_the_class_isl_refuses(lean: Any) -> None:
    """A planted wrong verdict: every refused cast stated as if isl had found it legal.

    Lean proves every dependence class but the ones isl's verdict breaks,
    and fails on the class isl's pair belongs to: a fake "legal" is caught,
    and caught where it is wrong.
    """
    from dataclasses import replace

    for label, exc in _refused_casts():
        stated = exc.fact.term.lean_obligation()
        planted = replace(exc.fact, term=_Planted(stated.parts), status=Status.DECIDED)
        result = lean.establish(planted)
        assert result.status is Status.DECIDED, label
        failed = [r["name"] for r in result.provenance["lean_parts"] if not r["proved"]]
        assert stated.counterexample.part in failed, (label, failed)


VARIANT = """
from __future__ import annotations

from lanky.prelude import Real
from loopty import Arr, Fin, kernel, when


@kernel
def variant(u: Arr[Fin[nt], Fin[nx], Real]):
    steps = u.dom
    for t in steps:
        row = u.dom[t]
        for i in row:
            with when({guard}):
                u[{write}] = (u[{first}] + u[{second}]) / 2
"""

#: The stencil with its subscripts and guard perturbed: (guard, write, reads).
VARIANTS = {
    "jacobi": (
        "(t + 1 < steps.size) & (i > 0) & (i + 1 < row.size)",
        "t + 1, i", "t, i - 1", "t, i + 1",
    ),
    "wider": (
        "(t + 1 < steps.size) & (i > 0) & (i + 2 < row.size)",
        "t + 1, i", "t, i - 1", "t, i + 2",
    ),
    "in_place": ("(i > 0) & (i + 1 < row.size)", "t, i", "t, i - 1", "t, i + 1"),
    "same_level": (
        "(t + 1 < steps.size) & (i > 0) & (i + 1 < row.size)",
        "t + 1, i", "t + 1, i - 1", "t, i + 1",
    ),
    "two_levels": (
        "(t + 2 < steps.size) & (i > 0) & (i + 1 < row.size)",
        "t + 2, i", "t, i - 1", "t + 1, i + 1",
    ),
    "repeated": ("t + 1 < steps.size", "t + 1, i", "t, i", "t, i"),
}

#: The right half a time step later: a reversal at the seam for the stencil.
PIECEWISE_SHIFTED = (
    "{ [t, i] -> [a, b] : a = t and b = i and i < 4; "
    "[t, i] -> [a, b] : a = t + 1 and b = i and i >= 4 }"
)

#: The right half moved a hundred places on: the order kept, in two pieces.
PIECEWISE_SPREAD = (
    "{ [t, i] -> [a, b] : a = t and b = i and i < 4; "
    "[t, i] -> [a, b] : a = t and b = i + 100 and i >= 4 }"
)

#: Schedules, legal for some variants and refused for others.
SCHEDULES = {
    "tile": lambda s: s.tile("t", "i", 4, 4),
    "skew_tile": lambda s: s.skew("i", by="t").tile("t", "i", 4, 4),
    "interchange": lambda s: s.interchange("i", "t"),
    "split": lambda s: s.split("t", 2),
    "reverse_i": lambda s: s.affine("{ [t, i] -> [t2, i2] : t2 = t and i2 = -i }"),
    "reverse_t": lambda s: s.affine("{ [t, i] -> [t2, i2] : t2 = -t and i2 = i }"),
    "wavefront": lambda s: s.affine("{ [t, i] -> [a, b] : a = 2 t + i and b = t }"),
    # piecewise maps, every piece of which Lean has to see (Codex)
    "piecewise": lambda s: s.affine(PIECEWISE_SHIFTED),
    "piecewise_legal": lambda s: s.affine(PIECEWISE_SPREAD),
}


def test_lean_fails_on_exactly_the_classes_isl_finds_reversed(
    lean: Any, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Mutations of subscripts, guards and schedules, class by class, both ways.

    For each variant and schedule, the last step's legality is stated for
    Lean as if isl had found it legal, and the classes Lean does not prove
    are exactly those isl finds the order reverses: none for a legal cast.
    A class isl finds reversed that Lean proves would be an unsound printer;
    one isl finds kept that Lean does not prove, an incomplete one.
    """
    from dataclasses import replace

    from loopty import oracle as isl_oracle
    from loopty.obligations import _classes, cast_obligation
    from loopty.schedule import Schedule as Scheduled

    reversed_by_isl: dict[str, list] = {}
    first_violation = Scheduled._first_violation

    def recording(self: Any, schedule: Any) -> Any:
        reversed_by_isl["deps"] = [
            dep
            for dep in self._deps
            if not isl_oracle.is_monotone(schedule, dep.relation).ok
        ]
        return first_violation(self, schedule)

    monkeypatch.setattr(Scheduled, "_first_violation", recording)
    kinds = {
        "raw": ("write", "read"),
        "war": ("read", "write"),
        "waw": ("write", "write"),
    }

    def classes_of(origin: Any, dep: Any) -> set[str]:
        from lanky.terms import structurally_equal

        return {
            f"dependence_{n}"
            for n, (a, b, first, second) in enumerate(_classes(origin))
            if (a.id, b.id) == (dep.source, dep.sink)
            and (first[0], second[0]) == kinds[dep.kind]
            and first[1] == dep.array
            and structurally_equal(tuple(first[2]), tuple(dep.source_indices))
            and structurally_equal(tuple(second[2]), tuple(dep.sink_indices))
        }

    outcomes = {"legal": 0, "refused": 0}
    for name, (guard, write, first, second) in VARIANTS.items():
        path = tmp_path / f"variant_{name}.py"
        path.write_text(
            VARIANT.format(guard=guard, write=write, first=first, second=second)
        )
        module = load(path)
        for label, build in SCHEDULES.items():
            reversed_by_isl.clear()
            start = Schedule(module.variant, target="c", sizes={"nt": 12, "nx": 12})
            try:
                fact = [f for f in build(start).facts() if f.kind == "monotone"][-1]
                wanted: set[str] = set()
                outcomes["legal"] += 1
            except IllegalCast as refused:
                fact = refused.fact
                if fact.kind != "monotone":
                    continue
                origin = fact.term.schedule
                wanted = set().union(
                    *(classes_of(origin, dep) for dep in reversed_by_isl["deps"])
                )
                outcomes["refused"] += 1
            stated = cast_obligation(replace(fact.term, refusal=None, refuted=False))
            assert not stated.declined, (name, label, stated.declined)
            planted = replace(fact, term=_Planted(stated.parts), status=Status.DECIDED)
            result = lean.establish(planted)
            failed = {
                record["name"]
                for record in result.provenance["lean_parts"]
                if not record["proved"]
            }
            assert failed == wanted, (name, label, sorted(failed), sorted(wanted))
    assert outcomes["legal"] >= 10 and outcomes["refused"] >= 10, outcomes


def test_a_piecewise_map_is_proved_piece_by_piece(lean: Any) -> None:
    # Every piece of the map is in the statement, and isl's functional form
    # of it, in two pieces, is the witness that the step is total.
    schedule = Schedule(stencil().jacobi, target="c", sizes={"nt": 8, "nx": 8})
    bijective, monotone = [
        f for f in schedule.affine(PIECEWISE_SPREAD).facts()
        if f.kind in ("bijective", "monotone")
    ][-2:]
    total = bijective.term.lean_obligation().parts[0]
    assert len(total.hints.witness) == 2
    for fact in (bijective, monotone):
        assert lean.establish(fact).status is Status.PROVED, fact.statement


MUTANT = """
from __future__ import annotations

from lanky.prelude import Real
from loopty import Arr, Fin, kernel, when


@kernel
def mutant(u: Arr[Fin[nt], Fin[nx], Real]):
    steps = u.dom
    for t in steps:
        row = u.dom[t]
        for i in row:
            {body}
"""


@pytest.mark.parametrize(
    "body",
    [
        # the stencil as written: every access in bounds
        "with when((t + 1 < steps.size) & (i > 0) & (i + 1 < row.size)):\n"
        "                u[t + 1, i] = (u[t, i - 1] + u[t, i + 1]) / 2",
        # its guard dropped: u[t + 1, i] and u[t, i - 1] leave the array
        "u[t + 1, i] = (u[t, i - 1] + u[t, i + 1]) / 2",
        # one subscript moved a cell further than the guard allows
        "with when((t + 1 < steps.size) & (i > 0) & (i + 1 < row.size)):\n"
        "                u[t + 1, i] = (u[t, i - 1] + u[t, i + 2]) / 2",
    ],
    ids=["as written", "guard dropped", "subscript moved"],
)
def test_lean_refuses_a_perturbed_subscript_exactly_where_isl_does(
    lean: Any, body: str, tmp_path: Path
) -> None:
    """Mutations of the stencil's subscripts: isl's verdict and Lean's agree on each.

    An access isl decides in bounds Lean proves; one isl refutes Lean
    cannot prove, stated as if isl had decided it, and Lean proves it false
    at isl's counterexample.
    """
    from dataclasses import replace

    path = tmp_path / "mutant.py"
    path.write_text(MUTANT.format(body=body))
    module = load(path)
    for fact in module.mutant.facts():
        if fact.kind != "in-bounds":
            continue
        verdict = decide(fact.term)
        stated = fact.term.lean_obligation()
        if verdict.ok:
            assert lean.establish(fact).status is Status.PROVED, fact.statement
            continue
        planted = replace(fact, term=_Planted(stated.parts))
        assert lean.establish(planted).status is Status.ASSUMED, fact.statement
        refuted = lean.establish(fact.with_status(Status.REFUTED, "isl"))
        said = refuted.provenance["lean_counterexample"]
        confirmed = said.startswith("Lean confirmed the counterexample")
        assert confirmed, (fact.statement, said)


def test_isls_split_closes_what_omega_cannot(lean: Any) -> None:
    """The fallback of every part loopty states: asked once omega has failed."""
    from lanky.obligations import Obligation

    from loopty.obligations import _obligation

    (part,) = _obligation(
        [lanky_obligations.Part("pugh", "Pugh's example", _pugh())]
    ).parts
    assert part.fallback is not None

    class Stated:
        def lean_obligation(self) -> Any:
            return Obligation(parts=(part,), decider="isl")

    from lanky.ledger import Fact

    handed = Fact(id="pugh", kind="in-bounds", statement="", term=Stated())
    fact = lean.establish(handed)
    assert fact.status is Status.PROVED
    assert fact.provenance["lean_parts"][0]["how"] == "omega, with a case split"


def _atomized(expr: Any, atoms: dict[str, Any]) -> Any:
    """``expr`` with each application of an array an integer of its own.

    The same application, by its text, is the same integer; ``atoms`` maps
    each integer's name to the application, its arguments atomized too.
    """
    import pymbolic.primitives as prim
    from lanky.terms import init_args

    if isinstance(expr, prim.Subscript) and isinstance(expr.aggregate, prim.Variable):
        index = expr.index if isinstance(expr.index, tuple) else (expr.index,)
        inner = prim.Subscript(
            expr.aggregate, tuple(_atomized(item, atoms) for item in index)
        )
        for name, cell in atoms.items():
            if render(cell) == render(inner):
                return Var(name)
        name = f"__cell{len(atoms)}"
        atoms[name] = inner
        return Var(name)
    if isinstance(expr, tuple | list):
        return type(expr)(_atomized(item, atoms) for item in expr)
    if isinstance(expr, prim.ExpressionNode):
        return type(expr)(*(_atomized(arg, atoms) for arg in init_args(expr)))
    return expr


def _sample(part: Any) -> tuple[dict[str, int], dict[str, dict]] | None:
    """A point where every hypothesis of a part holds, from isl: a non-vacuity witness.

    The hypotheses are stated to isl as text again, each application of an
    array an integer of its own, with one value for one array at equal
    arguments (two applications of ``cnt`` at equal arguments are equal), and
    isl's sample gives every variable a value, and every array a table of
    the values it takes there. ``None`` for a part whose hypotheses isl
    finds empty, which a dependence class two statements never form is, or
    cannot read.
    """
    import pymbolic.primitives as prim
    from lanky.terms import conjuncts

    term = part.term
    scalars = [var.name for var, sort in term.binders if sort is Int]
    atoms: dict[str, Any] = {}
    pieces = []
    for hypothesis in conjuncts(term.guard):
        text = _isl_text(_atomized(hypothesis, atoms))
        if text is None:
            return None
        pieces.append(text)
    cells = list(atoms.items())
    for (one, first), (other, second) in itertools.combinations(cells, 2):
        if first.aggregate.name != second.aggregate.name:
            continue
        apart = [
            _isl_text(prim.Comparison(a, "!=", b))
            for a, b in zip(first.index, second.index, strict=True)
        ]
        if None in apart:
            return None
        pieces.append("(" + " or ".join([*apart, f"{one} = {other}"]) + ")")
    names = [*scalars, *atoms]
    body = " and ".join(pieces) or "1 = 1"
    try:
        points = isl.Set(f"{{ [{', '.join(names)}] : {body} }}")
    except isl.Error:
        return None
    if points.is_empty():
        return None
    point = points.sample_point()
    values = {
        name: point.get_coordinate_val(isl.dim_type.set, k).to_python()
        for k, name in enumerate(names)
    }
    tables: dict[str, dict] = {}
    for name, cell in cells:
        at = tuple(int(evaluate(item, values)) for item in cell.index)
        tables.setdefault(cell.aggregate.name, {})[at] = values[name]
    return {name: values[name] for name in scalars}, tables


def _lean_value(name: str, kind: str, scalars: dict, tables: dict) -> str:
    """A binder's value as Lean source: a number, or an array as a finite table."""
    if kind == "Int":
        value = scalars[name]
        return str(value) if value >= 0 else f"({value})"
    table = tables.get(name, {})
    rank = kind.count("→")
    params = " ".join(f"x{k}" for k in range(rank))
    out = "0"
    for at, value in table.items():
        test = " ∧ ".join(
            f"x{k} = {v}" if v >= 0 else f"x{k} = ({v})" for k, v in enumerate(at)
        )
        shown = str(value) if value >= 0 else f"({value})"
        out = f"if {test} then {shown} else {out}"
    return f"(fun {params} => {out} : {kind})"


def _isl_text(prop: Any) -> str | None:
    """A lanky proposition over integers, as isl text; ``None`` for one it is not."""
    import pymbolic.primitives as prim

    from loopty.flow import NonAffine, expr_text

    if prop is True:
        return "1 = 1"
    if prop is False:
        return "1 = 0"
    if isinstance(prop, prim.LogicalAnd | prim.LogicalOr):
        parts = [_isl_text(child) for child in prop.children]
        if None in parts:
            return None
        joiner = " and " if isinstance(prop, prim.LogicalAnd) else " or "
        return "(" + joiner.join(parts) + ")"
    if isinstance(prop, prim.LogicalNot):
        inner = _isl_text(prop.child)
        return None if inner is None else f"(not {inner})"
    if isinstance(prop, prim.Comparison):
        try:
            left = expr_text(prop.left, None, None)
            right = expr_text(prop.right, None, None)
        except (NonAffine, ValueError):
            return None
        operator = {"==": "=", "!=": "!="}.get(prop.operator, prop.operator)
        if operator == "!=":
            return f"({left} < {right} or {left} > {right})"
        return f"{left} {operator} {right}"
    return None


def test_every_hypothesis_set_isl_can_sample_is_not_vacuous_in_lean(lean: Any) -> None:
    """Non-vacuity: at isl's sample, Lean proves every hypothesis of each part.

    A printer bug that empties a domain makes its theorem hold vacuously.
    For every part of every fact and cast of the examples and the test
    kernels whose hypotheses isl finds a point of, Lean proves ``∃
    variables, hypotheses`` there by ``decide``, an array given as the table
    of the values isl sampled: the hypotheses as Lean reads them hold
    somewhere. The parts that read arrays (a ragged row's length, a flat
    layout's offsets) are among them.
    """
    from lanky.obligations import part_statement

    checked = 0
    arrays = 0
    for path in [*sorted(EXAMPLES.glob("*.py")), *sorted(KERNELS.glob("*.py"))]:
        # one Lean command per file, each theorem again alone to name a failure
        sources: list[tuple[str, str]] = []
        for fact in _isl_facts(path):
            obligation = fact.term.lean_obligation()
            for part in obligation.parts:
                found = _sample(part)
                if found is None:
                    continue
                scalars, tables = found
                statement = part_statement(part, f"nonvacuous_{checked}")
                if not statement.binders:
                    # a part with no variable, a step that moves nothing
                    continue
                kinds = dict(statement.binders)
                hypotheses = " ∧ ".join(f"({text})" for _, text in statement.hypotheses)
                binders = " ".join(
                    f"({name} : {kind})" for name, kind in statement.binders
                )
                values = ", ".join(
                    _lean_value(variable, kinds[name], scalars, tables)
                    for (name, _), variable in zip(
                        statement.binders, statement.variables, strict=True
                    )
                )
                source = (
                    "set_option synthInstance.maxSize 4096 in\n"
                    f"theorem Lanky.nonvacuous_{checked} : ∃ {binders}, "
                    f"{hypotheses or 'True'} := ⟨{values}, by decide⟩\n"
                )
                sources.append((f"{fact.statement}: {part.name}", source))
                checked += 1
                arrays += bool(tables)
        if not sources:
            continue
        closed, _detail = lean.session.run("\n".join(source for _, source in sources))
        if closed:
            continue
        for label, source in sources:
            closed, detail = lean.session.run(source)
            assert closed, (path.name, label, source, detail)
    assert checked > 300 and arrays > 20, (checked, arrays)


def test_the_printed_predicates_hold_where_isl_says_in_lean(lean: Any) -> None:
    """The differential, through the printer: Lean's ``decide`` against isl at points.

    The stencil's domain, printed by lanky as the theorem prints it, is
    decided by Lean at a handful of points, in and out of the set, and
    agrees with isl's membership at each.
    """
    from lanky.lean import print_lean

    (stmt,) = stencil().jacobi.term.stmts
    described = description_of(stmt.domain)
    for t, i in ((0, 1), (0, 0), (2, 3), (3, 1), (-1, 1), (1, 4)):
        coords = [t, i]
        predicate = described.predicate(coords)
        sizes = {"nt": 4, "nx": 5}
        from loopty.description import substitute

        closed = substitute(predicate, sizes)
        expected = _member(stmt.domain, {"t": t, "i": i}, sizes)
        prop = print_lean(closed)
        claim = prop if expected else f"¬({prop})"
        ok, detail = lean.session.run(f"example : {claim} := by decide\n")
        assert ok, (t, i, prop, detail)


# }}}
