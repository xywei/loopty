"""A postcondition decided from what the statements store (#113), and cited theorems.

Where no cell of an array is stored twice, or read before its store, what each
statement stores is what the kernel leaves (:mod:`loopty.dataflow`): its value
equation. The postcondition is decided under them, a claim about reals with
each equation an atom, and a theorem the kernel cites is applied to it. The
first half needs no Lean; the second runs it, and skips without it.
"""

from __future__ import annotations

import importlib.util
import os
import sys
from pathlib import Path
from typing import Any

import pytest
from lanky.ledger import Ledger, Status
from lanky.plugins import registry
from lanky.terms import render

from loopty.dataflow import APPLIED, KIND, stores
from loopty.schedule import IllegalCast

pytest.importorskip("loopy")

ROOT = Path(__file__).resolve().parent.parent
KERNELS = ROOT / "tests" / "kernels"
EXAMPLES = ROOT / "examples"


def load(path: Path) -> Any:
    """Import a file of kernels as a module of its own."""
    name = f"dataflow_{path.parent.name}_{path.stem}"
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    sys.path.insert(0, str(path.parent))
    try:
        spec.loader.exec_module(module)
    finally:
        sys.path.remove(str(path.parent))
    return module


# imported with what it registers released again, so that the registry other
# test modules read is the one they made
with registry.collecting():
    kernels = load(KERNELS / "dataflow.py")


def facts(kernel: Any, kind: str) -> list[Any]:
    return [fact for fact in kernel.facts() if fact.kind == kind]


# {{{ decided


def test_the_scans_postcondition_is_decided_from_what_it_stores() -> None:
    """``off[r + 1] == off[r] + cnt[r]`` is what ``S1`` stores at ``r``: decided by isl.

    It used to be ``tested`` on the native runs, which it still is, as
    ``native`` in its provenance; it rests on the fact that no cell of
    ``off`` is stored twice or read before its store, which comes before it.
    """
    (post,) = facts(kernels.scan, "postcondition")
    assert (post.status, post.decided_by) == (Status.DECIDED, "isl")
    assert post.provenance["native"] == "tested"
    (single,) = facts(kernels.scan, KIND)
    assert (single.status, single.decided_by) == (Status.DECIDED, "isl")
    assert (
        single.statement == "no cell of off is stored twice, or read before its store"
    )
    assert post.rests_on == (single.id,)
    assert single.id == f"{KIND}:dataflow.scan@17:off"
    ids = [fact.id for fact in kernels.scan.facts()]
    assert ids.index(single.id) == ids.index(post.id) - 1
    assert post.provenance["used"] == [
        f"what {stmt.id} stores in off at {stmt.where}"
        for stmt in kernels.scan.term.stmts
    ]


def test_the_value_equations_are_what_each_statement_stores_where_it_runs() -> None:
    found = stores(kernels.scan.term, "scan")
    texts = [render(equation.claim) for equation in found.equations]
    assert texts[0] == "off[0] == 0"
    assert texts[1].startswith("forall r in Int where ")
    assert texts[1].endswith(". off[r + 1] == off[r] + cnt[r]")
    assert "0 <= r and r < n" in texts[1]


def test_the_isl_oracle_decides_the_postcondition_again() -> None:
    from loopty.oracle import IslOracle

    (post,) = facts(kernels.scan, "postcondition")
    (single,) = facts(kernels.scan, KIND)
    oracle = IslOracle()
    for fact in (post, single):
        handed = fact.with_status(Status.ASSUMED)
        assert oracle.establish(handed).status is Status.DECIDED


def test_a_postcondition_about_reals_is_decided_with_each_equation_an_atom() -> None:
    """``R[a + 2, b] == -R[a, b + 2]`` is what ``S1`` stores at ``(a + 2, b)``."""
    (post,) = facts(kernels.triangle, "postcondition")
    assert (post.status, post.decided_by) == (Status.DECIDED, "isl")
    assert len(post.provenance["used"]) == 2


def test_a_statement_without_loops_is_stated_for_lean_as_storing_once() -> None:
    """One instance, so no two that differ: a part, its hypothesis ``False``."""
    (fact,) = facts(kernels.one_cell, KIND)
    assert fact.status is Status.DECIDED
    obligation = fact.term.lean_obligation()
    assert not obligation.declined
    (part,) = obligation.parts
    assert "two instances of S0 storing into lim" in part.label


# }}}


# {{{ not decided, and why


def test_a_cell_stored_twice_leaves_the_postcondition_tested() -> None:
    (post,) = facts(kernels.twice, "postcondition")
    assert post.status is Status.TESTED
    assert "S0 and S1 store into one cell of y" in post.provenance["undecided"]
    assert not facts(kernels.twice, KIND)


def test_a_read_before_its_store_leaves_the_postcondition_tested() -> None:
    """True at every input, and not decided: what ``S0`` read was stored after it."""
    (post,) = facts(kernels.from_the_right, "postcondition")
    assert post.status is Status.TESTED
    assert (
        "S0 stores a cell of y that S0 reads before it" in post.provenance["undecided"]
    )


def test_a_recurrence_with_its_sign_lost_is_no_atom_it_stores() -> None:
    """Refuted by the native runs, never decided: another equation than the stored."""
    (post,) = facts(kernels.lost_sign, "postcondition")
    assert post.status is Status.REFUTED
    from loopty.dataflow import decide_postcondition

    term = kernels.lost_sign.term
    found = stores(term, "lost_sign")
    assert isinstance(decide_postcondition(term, found.equations), str)


# }}}


# {{{ a cited theorem


def test_a_conjunct_the_cited_theorem_gives_is_decided_under_it() -> None:
    """The monotonicity is ``offsets_increase`` at ``off`` and ``cnt``.

    Its hypotheses are the conjuncts decided before.
    """
    (post,) = facts(kernels.scan_monotone, "postcondition")
    assert (post.status, post.decided_by) == (Status.DECIDED, "isl")
    assert kernels.offsets_increase.fact_id in post.rests_on
    assert any(
        source.startswith("offsets_increase at ") for source in post.provenance["used"]
    )
    # its conclusion is a conjunct already: no fact of its own
    assert not facts(kernels.scan_monotone, APPLIED)


def test_a_cited_theorem_gives_its_conclusion_from_the_postcondition() -> None:
    (applied,) = facts(kernels.scan_cited, APPLIED)
    assert (applied.status, applied.decided_by) == (Status.DECIDED, "theorem")
    assert applied.statement == (
        "by offsets_increase: forall a in Fin(n + 1), b in Fin(n + 1) where a <= b. "
        "off[a] <= off[b]"
    )
    (post,) = facts(kernels.scan_cited, "postcondition")
    assert applied.rests_on == (kernels.offsets_increase.fact_id, post.id)
    assert applied.provenance["bindings"] == "n = n, cnt = cnt, off = off"
    ids = [fact.id for fact in kernels.scan_cited.facts()]
    assert ids.index(applied.id) == ids.index(post.id) + 1


def test_the_flagships_theorem_gives_r_is_d_for_every_d_c_holds() -> None:
    """``on_the_triangle``'s copies and recurrence are ``reconstruct``'s postcondition.

    The PDE and what ``c`` stores of ``D`` are no conjunct of it, so the
    claim is for every ``D`` they hold of: ``R`` is ``D`` on the triangle.
    """
    (applied,) = facts(kernels.reconstruct, APPLIED)
    claim = applied.term.application.claim
    assert [var.name for var, _ in claim.binders] == ["D"]
    assert render(claim.guard).startswith(
        "(forall a in Nat, b in Nat. D(a + 2)(b) + D(a)(b + 2) == 0) and "
    )
    assert "c[a, b] == D(a)(b)" in render(claim.guard)
    assert render(claim.body) == (
        "forall a in Fin(p + 1), b in Fin(p + 1) where a + b <= p. R[a, b] == D(a)(b)"
    )
    assert applied.provenance["bindings"] == "p = p, c = c, R = R"
    assert applied.provenance["matched"] == [
        render(part) for part in kernels.reconstruct.term.post.children
    ]


def test_a_theorem_that_does_not_apply_gives_nothing() -> None:
    assert not facts(kernels.scan_unrelated, APPLIED)
    # over Nat: the cells the scan writes are not known to be naturals
    assert not facts(kernels.scan_naturals, APPLIED)
    (post,) = facts(kernels.scan_naturals, "postcondition")
    assert kernels.naturals_increase.fact_id not in post.rests_on


def test_uses_takes_theorems_and_refuses_anything_else() -> None:
    from loopty import kernel

    with pytest.raises(TypeError, match="uses= of the kernel names theorems"):
        kernel(uses=["postcondition:spmv.scan@69"])
    with pytest.raises(TypeError, match="the theorems a kernel cites go in uses="):
        kernel(kernels.offsets_increase)


def test_the_ledger_counts_the_theorem_in_what_the_application_is_worth() -> None:
    applied = facts(kernels.scan_cited, APPLIED)[0]
    ledger = Ledger([*kernels.scan_cited.facts()])
    # the theorem is not in this ledger: it is what the application is under
    assert ledger.support(applied).under == (kernels.offsets_increase.fact_id,)


# }}}


# {{{ lanky check lists the casts of a file's schedules (#191)


def test_lanky_check_lists_the_casts_of_the_files_schedules_once(capsys: Any) -> None:
    """Two schedules that share their first step share its facts: one claim each."""
    from lanky import cli
    from lanky.check import check_path

    ledger = check_path(KERNELS / "dataflow.py")
    casts = [fact for fact in ledger if fact.id.startswith("cast:")]
    assert casts
    assert all(
        fact.status is Status.DECIDED and fact.decided_by == "isl" for fact in casts
    )
    first = [
        fact
        for fact in casts
        if ".split('r', 2" in fact.id and "split('r_outer'" not in fact.id
    ]
    assert len(first) == 2  # bijective and monotone, once
    assert not [fact for fact in ledger if fact.provenance.get("duplicate_claims")]
    assert (
        cli.main(["check", str(KERNELS / "dataflow.py")]) == 1
    )  # the refuted postcondition
    out = capsys.readouterr().out
    assert "DUPLICATE" not in out
    assert "split('r', 2" in out or "split(r, 2" in out


# }}}


# {{{ with Lean


def _without_lean(reason: str) -> None:
    if os.environ.get("LANKY_LEAN_TEST_REQUIRED"):
        pytest.fail(f"LANKY_LEAN_TEST_REQUIRED is set, but {reason}", pytrace=False)
    pytest.skip(reason)


@pytest.fixture(scope="module")
def lean(tmp_path_factory: pytest.TempPathFactory) -> Any:
    """A Lean oracle with an open session and a proof cache of its own."""
    pytest.importorskip("lanky.obligations", reason="this lanky states no obligations")
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


@pytest.mark.lean
def test_lean_proves_the_scans_postcondition_and_what_it_rests_on(lean: Any) -> None:
    for kind in (KIND, "postcondition"):
        (fact,) = facts(kernels.scan, kind)
        proved = lean.establish(fact)
        assert proved.status is Status.PROVED, proved.provenance.get("lean_reason")
        assert proved.decided_by == "lean"


@pytest.mark.lean
def test_lean_proves_a_statement_without_loops_stores_its_cell_once(lean: Any) -> None:
    for kind in (KIND, "postcondition"):
        (fact,) = facts(kernels.one_cell, kind)
        proved = lean.establish(fact)
        assert proved.status is Status.PROVED, proved.provenance.get("lean_reason")


@pytest.mark.lean
def test_lean_checks_that_the_cited_theorem_applies(lean: Any) -> None:
    (applied,) = facts(kernels.scan_cited, APPLIED)
    proved = lean.establish(applied)
    assert proved.status is Status.PROVED, proved.provenance.get("lean_reason")
    assert "apply h" in proved.provenance["lean_source"]
    assert kernels.offsets_increase.fact_id in proved.rests_on


@pytest.mark.lean
def test_without_mathlib_a_claim_about_reals_is_left_to_isl_saying_why(
    lean: Any,
) -> None:
    if lean._mathlib():
        pytest.skip("Mathlib is imported here")
    (post,) = facts(kernels.triangle, "postcondition")
    result = lean.establish(post)
    assert result.status is Status.DECIDED
    assert "real numbers" in result.provenance["lean_declined"]


@pytest.mark.lean
def test_with_mathlib_the_reconstruction_and_the_theorem_it_cites_are_proved(
    lean: Any,
) -> None:
    """Over reals: the postcondition by ``linarith``, the application by ``apply``."""
    if not lean._mathlib():
        pytest.skip("Mathlib is not imported here (LANKY_LEAN_MATHLIB)")
    for kind in (KIND, "postcondition"):
        (fact,) = facts(kernels.triangle, kind)
        proved = lean.establish(fact)
        assert proved.status is Status.PROVED, proved.provenance.get("lean_reason")
    (applied,) = facts(kernels.reconstruct, APPLIED)
    proved = lean.establish(applied)
    assert proved.status is Status.PROVED, proved.provenance.get("lean_reason")
    assert "apply h" in proved.provenance["lean_source"]
    assert kernels.on_the_triangle.fact_id in proved.rests_on


@pytest.mark.lean
def test_a_wrong_application_is_no_proof(lean: Any) -> None:
    """The claim turned round is no consequence of the theorem, and is not proved."""
    import dataclasses

    from loopty.dataflow import Applied

    (applied,) = facts(kernels.scan_cited, APPLIED)
    # the claim with its order turned round, which the theorem does not give
    flipped = applied.term.application.claim
    from lanky.terms import Comparison, Forall

    body = Comparison(flipped.body.right, "<=", flipped.body.left)
    wrong = dataclasses.replace(
        applied.term.application, claim=Forall(flipped.binders, body, flipped.guard)
    )
    term = Applied(applied.term.term, wrong, applied.term.given, applied.term.owner)
    result = lean.establish(dataclasses.replace(applied, term=term))
    assert result.status is Status.DECIDED


# }}}


def test_the_example_reads_top_to_bottom_and_refuses_the_interchange() -> None:
    example = load(EXAMPLES / "taylor_reconstruction.py")
    with pytest.raises(IllegalCast, match="interchange"):
        example.rejected_interchange()
    (applied,) = facts(example.reconstruct, APPLIED)
    assert applied.statement.startswith("by on_the_triangle: forall D in ")
