"""Retained phase checks cannot certify another snapshot or hide prerequisites."""

from dataclasses import replace

import pytest
from lanky.ledger import Fact, Ledger, Status
from lanky.plugins import registry

from loopty.storage_evidence import StorageOracle
from test_window import WINDOWS


@pytest.fixture(autouse=True)
def storage_oracle(monkeypatch):
    registry.load_entry_points()
    monkeypatch.setattr(registry, "oracles", [StorageOracle()])


def windowed():
    return WINDOWS.fused.contract(WINDOWS.burgers.f, registers=False)


def test_phase_facts_recheck_actual_objects_and_preserve_source_prerequisites():
    schedule = windowed()
    (fact,) = schedule.storage_facts()
    assert fact.status is Status.DECIDED
    assert fact.decided_by == "loopty-storage"
    assert fact.provenance["scope"] == "window rewrite before dependency ordering"
    assert fact.provenance["trust_class"] == "decision-procedure"
    assert fact.term.checkpoint.source is schedule.term
    assert len(fact.rests_on) >= 4
    reading = next(key for key in fact.rests_on if key.startswith("trace-faithful:"))
    lowering = next(key for key in fact.rests_on if key.startswith("storage-input:"))
    ledger = Ledger([*schedule.facts(), fact])
    assert ledger.support(fact).effective is Status.ASSUMED
    assert lowering in ledger.support(fact).under
    ledger.add(Fact(reading, "trace-faithful", "reading", status=Status.TESTED))
    assert ledger.support(fact).effective is Status.ASSUMED  # missing lowering
    ledger.add(
        Fact(lowering, "correspondence", "input lowering", status=Status.DECIDED)
    )
    assert ledger.support(fact).effective is Status.TESTED
    assert ledger.to_dicts()[-1]["status"] == "decided"
    assert "object at 0x" not in ledger.to_json()
    ledger.add(
        Fact(lowering, "correspondence", "bad input lowering", status=Status.REFUTED)
    )
    assert ledger.support(fact).effective is Status.REFUTED


def test_later_cast_invalidates_inherited_evidence_but_does_not_change_old_snapshot():
    before = windowed()
    original_facts = before.facts()
    after = before.split(WINDOWS.burgers.flux.j, 4)
    (stale,) = after.storage_facts()
    assert stale.status is Status.ASSUMED
    assert stale.provenance["stale"] is True
    assert before.storage_facts()[0].status is Status.DECIDED
    assert before.facts() == original_facts


def test_retargeting_replays_and_binds_new_evidence():
    before = windowed()
    after = before.retarget("opencl")
    (fresh,) = after.storage_facts()
    assert fresh.status is Status.DECIDED
    assert fresh.provenance["target"] == "opencl"
    assert fresh.id != before.storage_facts()[0].id
    assert fresh.term.checkpoint.binding is not before._storage_binding


def test_register_shift_limit_is_visible():
    schedule = WINDOWS.fused.contract(WINDOWS.burgers.f, registers=True)
    (fact,) = schedule.storage_facts()
    assert fact.status is Status.ASSUMED
    assert "register-shift" in fact.provenance["declined"]


def test_tampered_retained_output_and_snapshot_cannot_inherit_success():
    schedule = windowed()
    (checkpoint,) = schedule._storage_checkpoints
    entry = checkpoint.result.kernel.default_entrypoint
    insn = next(i for i in entry.instructions if hasattr(i, "expression"))
    bad = checkpoint.result.kernel.with_kernel(
        entry.copy(
            instructions=[
                i.copy(expression=i.expression + 1) if i.id == insn.id else i
                for i in entry.instructions
            ]
        )
    )
    schedule._storage_checkpoints = (
        replace(checkpoint, result=replace(checkpoint.result, kernel=bad)),
    )
    assert schedule.storage_facts()[0].status is Status.REFUTED
    schedule._storage_checkpoints = (checkpoint,)
    schedule._kernel = bad
    assert schedule.storage_facts()[0].status is Status.ASSUMED


def test_unexpected_checker_failure_is_an_error_not_a_refutation(monkeypatch):
    from loopty.backend_evidence import WindowEvidence

    def broken(*args):
        raise TimeoutError("checker budget exhausted")

    schedule = windowed()
    monkeypatch.setattr(WindowEvidence, "verify", broken)
    fact = schedule.storage_facts()[0]
    assert fact.status is Status.ASSUMED
    assert "TimeoutError" in str(fact.provenance["raised"])


def test_multiple_contractions_do_not_rebind_historical_checkpoints():
    from test_window import ROOT, load

    schedule = load(ROOT / "examples" / "l2p.py").contracted
    older, newest = schedule.storage_facts()
    assert older.status is Status.ASSUMED
    assert older.provenance["stale"]
    assert newest.status is Status.DECIDED
    assert older.id != newest.id


def test_installed_oracle_discovery_rechecks_a_real_checkpoint(monkeypatch):
    # Undo this module's isolated dispatcher fixture: exercise installed entry
    # points and the other real oracles as an application sees them.
    monkeypatch.undo()
    registry.load_entry_points()
    assert any(isinstance(oracle, StorageOracle) for oracle in registry.oracles)
    (fact,) = windowed().storage_facts()
    assert fact.status is Status.DECIDED
    assert fact.provenance["scope"] == "window rewrite before dependency ordering"
