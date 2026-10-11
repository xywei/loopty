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

from lanky.lean import statement_of  # noqa: E402
from lanky.ledger import Status  # noqa: E402
from lanky.prelude import Real  # noqa: E402
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
    ],
)
def test_the_reader_reads_the_text_loopty_writes(text: str, expected: str) -> None:
    assert render(parse(text)) == expected


@pytest.mark.parametrize("text", ["exists e: i = 2e", "i * j < n", "i % n = 0", "i <"])
def test_the_reader_refuses_what_loopty_does_not_write(text: str) -> None:
    with pytest.raises(Unstatable):
        parse(text)


def _assignments(described: Any, sizes: dict[str, int], box: int) -> Iterator[dict]:
    """Points of a box around the set, and a value for each array cell a bound reads."""
    rng = np.random.default_rng(0)
    arrays = {}
    for _name, expr in described.reflected:
        aggregate = getattr(expr, "aggregate", None)
        if aggregate is not None and aggregate.name not in arrays:
            rank = len(expr.index) if isinstance(expr.index, tuple) else 1
            arrays[aggregate.name] = rng.integers(0, 4, size=(box + 4,) * rank)
    for point in itertools.product(range(-1, box), repeat=len(described.dims)):
        yield dict(zip(described.dims, point, strict=True)), arrays


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


def _domains(path: Path) -> Iterator[tuple[str, Any]]:
    from loopty.trace import reductions_in

    module = load(path)
    for found in kernels_of(module):
        for stmt in found.term.stmts:
            yield f"{found.__name__}.{stmt.id}", stmt.domain
            if stmt.loop_domain is not None:
                yield f"{found.__name__}.{stmt.id} loops", stmt.loop_domain
            for reduction in reductions_in(stmt.expr):
                yield f"{found.__name__}.{stmt.id} sum", reduction.domain


@pytest.mark.parametrize(
    "path",
    [*sorted(EXAMPLES.glob("*.py")), *sorted(KERNELS.glob("*.py"))],
    ids=lambda path: f"{path.parent.name}/{path.name}",
)
def test_the_description_reads_back_as_isl_reads_it(path: Path) -> None:
    """The predicate built from loopty's text holds exactly where isl's set does.

    The differential of the printed predicates against isl's membership:
    every domain of every kernel, at every point of a box around it, at a few
    sizes, with each array cell a bound reads given a value of its own.
    """
    checked = 0
    for label, domain in _domains(path):
        described = description_of(domain)
        assert described is not None, label
        names = [name for name in domain.get_var_names(isl.dim_type.param)]
        reflected = dict(described.reflected)
        for size in (0, 1, 3):
            sizes = {name: size + 2 for name in names if name not in reflected}
            coords = [Var(f"__x{k}") for k in range(len(described.dims))]
            predicate = described.predicate(coords)
            for point, arrays in _assignments(described, sizes, 4):
                values = {f"__x{k}": point[d] for k, d in enumerate(described.dims)}
                params = dict(sizes)
                at = {**point, **sizes}
                for name, expr in reflected.items():
                    params[name] = int(evaluate(expr, {**at, **arrays}))
                expected = _member(domain, point, params)
                got = bool(evaluate(predicate, {**values, **sizes, **arrays}))
                assert got == expected, (label, point, params, render(predicate))
                checked += 1
    assert checked


# }}}


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


def test_a_cast_after_a_substitution_is_left_to_isl() -> None:
    module = load(EXAMPLES / "fusion.py")
    schedule = next(
        value
        for value in vars(module).values()
        if isinstance(value, Schedule) and "substitute" in repr(value)
    )
    monotone = schedule.facts()[-1]
    assert "a substitution before this step" in cast_obligation(monotone.term).declined


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
#: layout fact, an induction over the run; the casts after a substitution,
#: whose dependences are isl's; and a program's definedness, over the union
#: of the writes isl computed. The coverage report lists them with the reason
#: (docs/lean-checks.md), and an issue each.
NOT_STATED = (
    "stay inside their buffers",
    "substitute(",
    "stored before it",
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


def test_every_refused_cast_is_proved_false_at_isls_pair(lean: Any) -> None:
    for label, exc in _refused_casts():
        fact = lean.establish(exc.fact)
        said = fact.provenance.get("lean_counterexample", "")
        assert said.startswith("Lean proved the claim false at"), (label, said)


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
        assert said.startswith("Lean proved the claim false"), (fact.statement, said)


def _sample(part: Any) -> dict[str, int] | None:
    """A point where every hypothesis of a part holds, from isl: a non-vacuity witness.

    The hypotheses are stated to isl as text again, each application of an
    array an integer of its own, and isl's sample gives every variable a
    value. ``None`` for a part whose hypotheses isl finds empty, which a
    dependence class two statements never form is.
    """
    from lanky.terms import conjuncts

    from loopty.obligations import _names

    term = part.term
    names = [var.name for var, _ in term.binders]
    found: dict[str, int] = {}
    _names(term.guard, found)
    if any(arity for arity in found.values()):
        return None
    pieces = []
    for hypothesis in conjuncts(term.guard):
        text = _isl_text(hypothesis)
        if text is None:
            return None
        pieces.append(text)
    body = " and ".join(pieces) or "1 = 1"
    try:
        points = isl.Set(f"{{ [{', '.join(names)}] : {body} }}")
    except isl.Error:
        return None
    if points.is_empty():
        return None
    point = points.sample_point()
    return {
        name: point.get_coordinate_val(isl.dim_type.set, k).to_python()
        for k, name in enumerate(names)
    }


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
    For every part of the stencil's and the wavefront's facts and casts
    whose hypotheses isl finds a point of, Lean proves ``∃ variables,
    hypotheses`` at that point by ``decide``: the hypotheses as Lean reads
    them hold somewhere.
    """
    from lanky.obligations import part_statement

    checked = 0
    for path in (EXAMPLES / "stencil_skew.py", EXAMPLES / "wavefront_acoustic.py"):
        for fact in _isl_facts(path):
            obligation = fact.term.lean_obligation()
            for part in obligation.parts:
                point = _sample(part)
                if point is None:
                    continue
                statement = part_statement(part, f"nonvacuous_{checked}")
                hypotheses = " ∧ ".join(f"({text})" for _, text in statement.hypotheses)
                binders = " ".join(f"({name} : Int)" for name, _ in statement.binders)
                values = ", ".join(str(point[name]) for name, _ in statement.binders)
                source = (
                    "set_option synthInstance.maxSize 4096 in\n"
                    f"theorem Lanky.nonvacuous_{checked} : ∃ {binders}, "
                    f"{hypotheses or 'True'} := ⟨{values}, by decide⟩\n"
                )
                closed, detail = lean.session.run(source)
                assert closed, (fact.statement, part.name, detail)
                checked += 1
    assert checked > 10


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
