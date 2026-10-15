"""A program's postcondition: tested at its outputs' class, decided, or unfolded.

A ``@program``'s return annotation is a claim about what it leaves in its
parameters (:meth:`loopty.kernel.Program.facts`). Its native runs test it, an
equation of an output compared at that output's class; isl decides it under
what each call stores (:func:`loopty.dataflow.program_stores`) where it can;
and otherwise it is stated for Lean from those equations and the theorems the
program cites (:mod:`loopty.unfolding`), which Lean with Mathlib proves. The
first half needs no Lean; the second runs it, and skips without it.
"""

from __future__ import annotations

import dataclasses
import importlib.util
import os
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pytest
from lanky.ledger import Status
from lanky.plugins import registry
from lanky.prelude import FinType
from lanky.terms import Comparison, Forall, Subscript, Var, render

from loopty.dataflow import KIND, program_stores
from loopty.unfolding import Unfolded, as_sums, unfolded

pytest.importorskip("loopy")

ROOT = Path(__file__).resolve().parent.parent
KERNELS = ROOT / "tests" / "kernels"
EXAMPLES = ROOT / "examples"


def load(path: Path) -> Any:
    """Import a file of kernels as a module of its own."""
    name = f"program_posts_{path.parent.name}_{path.stem}"
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    sys.path.insert(0, str(path.parent))
    try:
        spec.loader.exec_module(module)
    finally:
        sys.path.remove(str(path.parent))
    return module


# imported with what they register released again, so that the registry other
# test modules read is the one they made
with registry.collecting():
    programs = load(KERNELS / "program_posts.py")
    hexahedra = load(EXAMPLES / "hexahedra.py")


def facts(program: Any, kind: str) -> list[Any]:
    return [fact for fact in program.facts() if fact.kind == kind]


def steps(fact: Any) -> list[tuple[str, list[str], bool]]:
    """Each step of each plan: a theorem's name or an equation's cell, its arguments."""
    out = []
    for plan in fact.term.plans:
        for step in plan.steps:
            if step.theorem is not None:
                name = step.theorem.__name__
            else:
                name = render(step.hypothesis.body.left)
            out.append((name, [render(a) for a in step.arguments], step.backwards))
    return out


# {{{ decided by isl


def test_an_integer_programs_postcondition_is_decided_from_what_its_calls_store() -> (
    None
):
    """``z[i] == 2 * x[i] + 2``: what ``add_one`` and ``double`` store, read by isl.

    Each call stores each cell once, and reads none before it stores it, a
    ``single-assignment`` fact of the program's per call, which come before
    the postcondition, and which it rests on.
    """
    single = facts(programs.steps, KIND)
    assert [fact.statement for fact in single] == [
        "add_one stores no cell of y twice, and reads none before its store",
        "double stores no cell of z twice, and reads none before its store",
    ]
    assert [fact.id for fact in single] == [
        f"{KIND}:program_posts.steps@31:add_one:y",
        f"{KIND}:program_posts.steps@31:double:z",
    ]
    (post,) = facts(programs.steps, "postcondition")
    assert (post.status, post.decided_by) == (Status.DECIDED, "isl")
    assert post.statement == "forall i in Fin(n). z[i] == 2*x[i] + 2"
    assert post.provenance["native"] == "tested"
    assert post.rests_on == tuple(fact.id for fact in single)
    ids = [fact.id for fact in programs.steps.facts()]
    assert all(ids.index(fact.id) < ids.index(post.id) for fact in single)


def test_what_a_later_call_writes_over_is_no_equation() -> None:
    """``c`` is written by ``twice``, then by ``plus``: only ``plus``'s store is left.

    So ``c[i] == a[i] + b[i]`` is decided, and ``c[i] == 2 * a[i]``, what the
    first call stored, is refuted by a run, which names the cell and the two
    values, compared at the class of ``c``.
    """
    stored = program_stores(programs.last_wins.term, "last_wins")
    assert [hypothesis.source.split(" at ")[0] for hypothesis in stored.equations] == [
        "what plus.S0 stores in c"
    ]
    (post,) = facts(programs.last_wins, "postcondition")
    assert (post.status, post.decided_by) == (Status.DECIDED, "isl")
    (wrong,) = facts(programs.first_wins, "postcondition")
    assert (wrong.status, wrong.decided_by) == (Status.REFUTED, "native")
    reason = wrong.provenance["reason"]
    assert "(at i=0, c[i] is " in reason
    assert "further apart than the approx class allows" in reason
    assert "np.float64" not in reason


# }}}


# {{{ tested, and stated for Lean


def test_a_claim_about_reals_is_tested_at_its_class_and_unfolded_for_lean() -> None:
    """``c[i] == 3 * a[i]``: two equations isl reads as atoms, which ``ring`` adds up.

    The native runs compare ``c`` at ``approx``, the class of an array of
    reals. isl does not decide it, so it is stated for Lean: ``c`` unfolded
    into ``a + b``, ``b`` into ``2 * a``, and ``ring``.
    """
    (post,) = facts(programs.tripled, "postcondition")
    assert (post.status, post.decided_by) == (Status.TESTED, "native")
    assert post.provenance["classes"] == {"c[i] == 3*a[i]": "approx"}
    assert isinstance(post.term, Unfolded)
    assert str(post.term) == "forall i in Fin(n). c[i] == 3*a[i]"
    assert steps(post) == [("c[i_0]", [], False), ("b[i]", [], False)]
    assert post.provenance["lean"] == "stated for Lean from what the calls store"
    # the native runs rest on nothing; the proof will rest on what it uses
    assert post.rests_on == ()
    assert set(post.term.rests_on) == {
        fact.id for fact in facts(programs.tripled, KIND)
    }


def test_a_cited_theorem_is_used_at_the_sections_of_the_arrays() -> None:
    """``two_passes`` is about one field; ``dense`` passes ``nq`` of them.

    The theorem's families are the arrays at the field ``q``, ``A[q]`` for
    ``A`` and ``g[q]`` for ``h``: taken forwards, onto the dense sum the
    claim states, before the passes are unfolded.
    """
    (post,) = facts(programs.dense, "postcondition")
    assert (post.status, post.decided_by) == (Status.TESTED, "native")
    assert steps(post)[0] == (
        "two_passes",
        ["n", "A[q]", "B[q]", "g[q]", "k", "j"],
        False,
    )
    assert [name for name, _, _ in steps(post)[1:]] == [
        "out[q_0, k, j_0]",
        "v[q, b, j]",
    ]
    assert post.provenance["lean"] == (
        "stated for Lean from what the calls store and two_passes"
    )
    assert programs.two_passes.fact_id in post.term.rests_on


def test_the_hexahedra_operators_cite_sum_factorization_backwards() -> None:
    """Three passes unfolded, then the theorem at each direction's sections, then ``w``.

    ``grad``'s store is the sum over the three directions written out, so
    each direction is an instance of its own: ``Fr[3*x + 0]`` for ``Fr``,
    and ``w[3*x + 0, e]`` for the field ``u``.
    """
    (post,) = facts(hexahedra.weak_grad, "postcondition")
    assert (post.status, post.decided_by) == (Status.TESTED, "native")
    found = steps(post)
    assert [name for name, _, _ in found[:3]] == [
        "grad[x, e_2, k_2, j_2, i_2]",
        "vs[q_1, e_1, k_1, j_1, i_1]",
        "v[q_0, e_0, k_0, j_0, i_0]",
    ]
    assert found[3:6] == [
        (
            "sum_factorization",
            [
                "n",
                f"Fr[3*x + {d}]",
                f"Fs[3*x + {d}]",
                f"Ft[3*x + {d}]",
                f"w[3*x + {d}, e]",
                "k",
                "j",
                "i",
            ],
            True,
        )
        for d in range(3)
    ]
    assert found[6][0] == "w[q, e, k, j, i]"
    (div,) = facts(hexahedra.weak_div, "postcondition")
    assert (div.status, div.decided_by) == (Status.TESTED, "native")
    assert [arguments[4] for name, arguments, _ in steps(div) if arguments] == [
        "w[0, e]",
        "w[1, e]",
        "w[2, e]",
    ]


def test_the_hexahedra_claims_read_a_literal_domain_as_a_binder() -> None:
    """``for x in Fin[3] for e in Fin[ne]``: one universal, ``x`` over ``Fin(3)``."""
    post = hexahedra.weak_grad.term.post
    assert isinstance(post, Forall)
    assert [(var.name, str(sort)) for var, sort in post.binders] == [
        ("x", "Fin(3)"),
        ("e", "Fin(ne)"),
        ("k", "Fin(n)"),
        ("j", "Fin(n)"),
        ("i", "Fin(n)"),
    ]
    assert [str(sort) for _, sort in post.body.right.binders] == [
        "Fin(3)",
        "Fin(n)",
        "Fin(n)",
        "Fin(n)",
    ]


def test_a_false_claim_is_refuted_by_a_run_naming_the_cell() -> None:
    """``A`` transposed: the dense sum is not what the passes compute."""
    (post,) = facts(programs.transposed, "postcondition")
    assert (post.status, post.decided_by) == (Status.REFUTED, "native")
    assert "(at q=0, k=0, j=0, out[q, k, j] is " in post.provenance["reason"]


def test_a_claim_naming_a_size_no_kernel_names_is_assumed_saying_so() -> None:
    (post,) = facts(programs.misnamed, "postcondition")
    assert post.status is Status.ASSUMED
    assert post.provenance["reason"] == (
        "the postcondition of misnamed names m, which is none of its parameters "
        "and none of its sizes (n, as the kernels it calls name them)"
    )


def test_a_loopty_sum_is_a_lanky_sum_over_the_bound_loopty_wrote() -> None:
    """``reduce_sum(... for a in B.dom[q, j])`` is ``sum(... for a in Fin[n])``."""
    stmt = next(s for s in programs.along_a.term.stmts if s.assignee.array == "v")
    total = as_sums(stmt.expr)
    assert [(var.name, str(sort)) for var, sort in total.binders] == [("a", "Fin(n)")]
    assert total.guard is not None  # the loops around it, which no equation says
    from loopty.dataflow import _equation
    from loopty.unfolding import statable_equation

    equation = statable_equation(_equation(stmt))
    value = equation.body.right
    assert value.guard is None
    assert render(value) == "sum(B[q, j, a]*g[q, b, a] for a in Fin(n))"


def test_the_native_comparison_walks_points_where_the_binders_make_no_box() -> None:
    """A bound that names another binder is evaluated point by point, at the class."""
    from loopty.faithful import _at_class

    term = programs.tripled.term
    i, j, n = Var("i"), Var("j"), Var("n")
    claim = Forall(
        ((i, FinType(n)), (j, FinType(i + 1))),
        Comparison(Subscript(Var("c"), i), "==", 3 * Subscript(Var("a"), i)),
    )
    a = np.linspace(-1.0, 1.0, 4)
    near = {"a": a, "c": 3 * a * (1 + 1e-9), "n": 4}
    assert _at_class(claim, near, term) is None
    far = {"a": a, "c": 3 * a + 1, "n": 4}
    assert "beyond the approx class" in _at_class(claim, far, term)


# }}}


# {{{ with Lean


def _without_lean(reason: str) -> None:
    if os.environ.get("LANKY_LEAN_TEST_REQUIRED"):
        pytest.fail(f"LANKY_LEAN_TEST_REQUIRED is set, but {reason}", pytrace=False)
    pytest.skip(reason)


@pytest.fixture(scope="module")
def lean(tmp_path_factory: pytest.TempPathFactory) -> Any:
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


def _mathlib(lean: Any) -> None:
    if not lean._mathlib():
        pytest.skip("Mathlib is not imported here (LANKY_LEAN_MATHLIB)")


@pytest.mark.lean
def test_lean_proves_a_decided_program_postcondition_and_what_it_rests_on(
    lean: Any,
) -> None:
    for fact in [*facts(programs.steps, KIND), *facts(programs.steps, "postcondition")]:
        proved = lean.establish(fact)
        assert proved.status is Status.PROVED, proved.provenance.get("lean_reason")


@pytest.mark.lean
def test_without_mathlib_an_unfolded_claim_waits_for_it_saying_why(lean: Any) -> None:
    if lean._mathlib():
        pytest.skip("Mathlib is imported here")
    (post,) = facts(programs.tripled, "postcondition")
    result = lean.establish(post)
    assert (result.status, result.decided_by) == (Status.TESTED, "native")
    assert "real numbers" in result.provenance["lean_declined"]


@pytest.mark.lean
def test_with_mathlib_an_unfolded_claim_is_proved_resting_on_what_it_used(
    lean: Any,
) -> None:
    """``tripled`` by ``ring`` alone; ``dense`` with ``two_passes`` at ``A q``."""
    _mathlib(lean)
    for program in (programs.tripled, programs.dense):
        (post,) = facts(program, "postcondition")
        proved = lean.establish(post)
        assert proved.status is Status.PROVED, proved.provenance.get("lean_reason")
        assert proved.decided_by == "lean"
        assert set(post.term.rests_on) <= set(proved.rests_on)
    assert "rw [h2 n (by omega) (A q) (B q) (g q)" in proved.provenance["lean_source"]
    assert programs.two_passes.fact_id in proved.rests_on


@pytest.mark.lean
def test_with_mathlib_the_hexahedra_operators_are_proved(lean: Any) -> None:
    """``weak_grad`` and ``weak_div``: the dense operators, by ``sum_factorization``."""
    _mathlib(lean)
    for program in (hexahedra.weak_grad, hexahedra.weak_div):
        (post,) = facts(program, "postcondition")
        proved = lean.establish(post)
        assert proved.status is Status.PROVED, proved.provenance.get("lean_reason")
        assert hexahedra.sum_factorization.fact_id in proved.rests_on
        assert "rw [← h" in proved.provenance["lean_source"]


@pytest.mark.lean
def test_a_wrong_hint_or_a_false_claim_is_no_proof(lean: Any) -> None:
    """Lean checks the steps Python found, and the claim, and trusts neither.

    The theorem taken with ``A`` and ``B`` swapped rewrites nothing the goal
    has; the transposed claim, stated from the same equations, is not what
    they give.
    """
    _mathlib(lean)
    (post,) = facts(programs.dense, "postcondition")
    plan = post.term.plans[0]
    first = plan.steps[0]
    swapped = dataclasses.replace(
        first,
        arguments=(
            first.arguments[0],
            first.arguments[2],
            first.arguments[1],
            *first.arguments[3:],
        ),
    )
    wrong = dataclasses.replace(
        post.term,
        plans=(dataclasses.replace(plan, steps=(swapped, *plan.steps[1:])),),
    )
    result = lean.establish(dataclasses.replace(post, term=wrong))
    assert result.status is not Status.PROVED
    term = dataclasses.replace(programs.dense.term, post=programs.transposed.term.post)
    stored = program_stores(term, "dense")
    found = unfolded(term, stored.equations, programs.dense.uses)
    assert not isinstance(found, str), found
    result = lean.establish(dataclasses.replace(post, term=found[0]))
    assert result.status is not Status.PROVED


# }}}
