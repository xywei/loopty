"""A cited theorem is restated from the text its own command proved, ``as_cited``.

loopty states three kinds of fact for Lean with a cited theorem as a
hypothesis of the part: a theorem applied to a kernel's postcondition
(:class:`loopty.dataflow.Applied`), the series a kernel adds
(:class:`loopty.summands.SeriesTerms`), and a program's postcondition
unfolded through a theorem (:class:`loopty.unfolding.Unfolded`). Each part
names those hypotheses as citations (:func:`loopty.obligations.cited`), and
lanky's Lean oracle restates each from the text the theorem's command proved,
never by printing the theorem again. The design's T4 is the reason: a
citation restated without one of its conditions proves anything, here that
offsets which merely start at zero are monotone.
"""

from __future__ import annotations

import importlib.util
import os
import sys
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest
from lanky.ledger import Status
from lanky.plugins import registry
from lanky.terms import Forall, conjuncts, structurally_equal

from loopty.dataflow import APPLIED
from loopty.summands import KIND as SERIES

pytest.importorskip("loopy")
lanky_citations = pytest.importorskip("lanky.citations")
if "hypothesis" not in getattr(lanky_citations.Citation, "__dataclass_fields__", {}):
    pytest.skip(
        "this lanky has no one citation for a part's hypothesis",
        allow_module_level=True,
    )

import lanky.obligations as lanky_obligations  # noqa: E402
from lanky.citations import Citation, Proved, ProvedTexts, restatements  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
KERNELS = ROOT / "tests" / "kernels"


def load(path: Path) -> Any:
    """Import a file of kernels as a module of its own."""
    name = f"citations_{path.parent.name}_{path.stem}"
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
    dataflow = load(KERNELS / "dataflow.py")
    series = load(KERNELS / "series.py")
    programs = load(KERNELS / "program_posts.py")


def _applied() -> Any:
    (fact,) = [
        fact
        for fact in dataflow.scan_cited.facts()
        if fact.kind == APPLIED and fact.status is not Status.ASSUMED
    ]
    return fact


def _cited_terms(part: Any) -> dict[str, Any]:
    """The conjunct each citation of ``part`` names, by the theorem's name."""
    guards = conjuncts(part.term.guard)
    return {citation.name: guards[citation.hypothesis] for citation in part.citations}


def test_an_application_cites_its_theorem() -> None:
    (part,) = _applied().term.lean_obligation().parts
    (citation,) = part.citations
    # lanky's one citation: the theorem, at the hypothesis that restates it
    assert isinstance(citation, Citation)
    assert citation.source is dataflow.offsets_increase
    assert citation.fact == dataflow.offsets_increase.fact_id
    assert citation.name == "offsets_increase"
    cited = _cited_terms(part)["offsets_increase"]
    assert structurally_equal(cited, dataflow.offsets_increase.term)


def test_a_series_cites_its_theorem_and_the_identities_it_rewrites_by() -> None:
    (row,) = [
        fact
        for fact in series.power_sum.facts()
        if fact.kind == SERIES and fact.status is not Status.ASSUMED
    ]
    parts = row.term.lean_obligation().parts
    assert parts
    for part in parts:
        cited = _cited_terms(part)
        assert set(cited) == {"halves", "halved"}
        assert structurally_equal(cited["halves"], series.halves.term)
        assert structurally_equal(cited["halved"], series.halved.term)


def test_an_unfolded_postcondition_cites_the_theorems_its_steps_take() -> None:
    (post,) = [fact for fact in programs.dense.facts() if fact.kind == "postcondition"]
    parts = post.term.lean_obligation().parts
    assert parts
    for part in parts:
        cited = _cited_terms(part)
        assert set(cited) == {"two_passes"}
        assert structurally_equal(cited["two_passes"], programs.two_passes.term)


def test_a_lanky_without_citations_is_handed_no_keyword(monkeypatch) -> None:
    """lanky 0.4.0 prints the statement as it always did, and loopty runs against it."""
    import loopty.obligations as obligations

    class Part:
        __dataclass_fields__ = {"name": None, "mathlib": None}

    monkeypatch.setattr(
        obligations, "_lanky_obligations", lambda: SimpleNamespace(Part=Part)
    )
    assert obligations.cited(["x"], [dataflow.offsets_increase]) == {}


def _proved(theorem: Any) -> ProvedTexts:
    """The text ``theorem``'s command proves, kept as lanky's oracle keeps it."""
    from lanky.oracles.lean import claim_statement

    statement = claim_statement(theorem.fact(), False)
    texts = ProvedTexts()
    texts.record(
        Proved(
            theorem.fact_id,
            statement.declared_name,
            statement.as_cited,
            False,
            theorem.term,
        )
    )
    return texts


def _misrestated(part: Any) -> Any:
    """``part`` with its cited theorem restated without its recurrence (T4)."""
    (citation,) = part.citations
    guards = list(conjuncts(part.term.guard))
    theorem = guards[citation.hypothesis]
    first, _recurrence = conjuncts(theorem.guard)
    guards[citation.hypothesis] = Forall(theorem.binders, theorem.body, first)
    from loopty.description import conjunction

    return replace(
        part, term=Forall(part.term.binders, part.term.body, conjunction(guards))
    )


def test_t4_lanky_restates_the_application_from_the_proved_text() -> None:
    (part,) = _applied().term.lean_obligation().parts
    texts = _proved(dataflow.offsets_increase)
    restated = restatements(part, texts)
    assert not restated.refused
    (citation,) = part.citations
    expected = texts.get(dataflow.offsets_increase.fact_id).text
    assert restated.texts == {citation.hypothesis: expected}
    # the theorem without its recurrence is no statement Lean proved of it
    refused = restatements(_misrestated(part), texts)
    assert refused.refused.startswith(
        "the part restates offsets_increase as something other than the statement "
        "Lean proved of it"
    )


# {{{ with Lean


def _without_lean(reason: str) -> None:
    if os.environ.get("LANKY_LEAN_TEST_REQUIRED"):
        pytest.fail(f"LANKY_LEAN_TEST_REQUIRED is set, but {reason}", pytrace=False)
    pytest.skip(reason)


@pytest.fixture(scope="module")
def lean(tmp_path_factory: pytest.TempPathFactory) -> Any:
    """A core Lean oracle with an open session and a proof cache of its own."""
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
def test_t4_against_lean(lean: Any) -> None:
    """Proved from the proved text; misrestated, it proves if trusted, not if cited."""
    theorem = lean.establish(dataflow.offsets_increase.fact())
    assert theorem.status is Status.PROVED, theorem.provenance.get("lean_reason")
    text = theorem.provenance["lean_as_cited"]
    applied = _applied()
    proved = lean.establish(applied)
    assert proved.status is Status.PROVED, proved.provenance.get("lean_reason")
    # the theorem is restated as its command proved it, as the first parameter
    first_line = proved.provenance["lean_source"].splitlines()[1]
    parameters = first_line.split(" (", 1)[1]
    assert parameters.split(" : ", 1)[1].startswith(f"{text}) ("), first_line
    assert "lean_restated_by_printer" not in proved.provenance

    (part,) = applied.term.lean_obligation().parts
    wrong = _misrestated(part)
    lanky = lanky_obligations

    def stated(*parts: Any) -> Any:
        return replace(
            applied,
            term=SimpleNamespace(lean_obligation=lambda: lanky.Obligation(parts=parts)),
            status=Status.DECIDED,
        )

    # trusted, the misrestated theorem proves the application (T4)...
    trusted = lean.establish(stated(replace(wrong, citations=())))
    assert trusted.status is Status.PROVED, "T4 no longer reproduces"
    # ...and cited, it is refused before Lean is asked
    refused = lean.establish(stated(wrong))
    assert refused.status is Status.DECIDED
    (record,) = refused.provenance["lean_parts"]
    assert "something other than the statement Lean proved" in record["reason"]


@pytest.mark.lean
def test_with_mathlib_the_rewrites_restate_both_theorems_from_proved_text() -> None:
    """The single layer in sumpy's arithmetic, as the design's 1.5 checks it.

    ``conjugate_form`` is proved by lanky alone, and ``log_of_the_modulus`` by
    its written proof, a chain citing two facts of ``lanky.known``, which is
    checked after the kernels' facts were established: their parts restated
    it by the printer then, and are asked of Lean again from the text its
    command proved.
    """
    if importlib.util.find_spec("lanky.known") is None:
        pytest.skip("this lanky has no library of known facts")
    project = os.environ.get("LANKY_LEAN_MATHLIB")
    if os.environ.get("LANKY_LEAN_DISABLE"):
        _without_lean("the Lean oracle is disabled by LANKY_LEAN_DISABLE")
    if not project:
        if os.environ.get("LANKY_LEAN_MATHLIB_TEST_REQUIRED"):
            pytest.fail(
                "LANKY_LEAN_MATHLIB_TEST_REQUIRED is set, but no project is named"
            )
        pytest.skip("LANKY_LEAN_MATHLIB names no Lake project with Mathlib")
    from lanky.check import check_path

    ledger = check_path(ROOT / "examples" / "layer_potential.py")
    by_owner = {fact.owner: fact for fact in ledger if fact.kind == "claim"}
    modulus = by_owner["log_of_the_modulus"]
    assert modulus.status is Status.PROVED, modulus.provenance
    assert modulus.provenance["proof"]["closed"] is True
    assert {"log_of_a_power", "modulus_squared"} <= set(by_owner)
    rewrites = [
        fact
        for fact in ledger
        if fact.statement.startswith("rewrite(out, by=")
        and fact.provenance.get("lean_parts")
    ]
    assert len(rewrites) == 3
    for fact in rewrites:
        assert fact.status is Status.PROVED, fact.statement
        assert "lean_restated_by_printer" not in fact.provenance, fact.statement
        notes = [
            note for record in fact.provenance["lean_parts"] for note in record["cited"]
        ]
        assert notes and all(
            "which its command proved" in note["restated"] for note in notes
        )


# }}}
