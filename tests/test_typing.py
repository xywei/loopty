"""The typing rules and the isl oracle: which obligation is settled by whom."""

from __future__ import annotations

from pathlib import Path

from lanky.check import check_path
from lanky.ledger import Status
from lanky.prelude import Nat, Real
from lanky.terms import evaluate_annotations

from loopty import Arr, Fin, when
from loopty import typing as rules
from loopty.oracle import IslOracle, Monotone, Subset
from loopty.trace import trace

KERNELS = Path(__file__).parent / "kernels"


def facts_of(fn):
    """Trace a plain function and state its obligations."""
    term = trace(fn, evaluate_annotations(fn))
    return term, rules.facts_for(term, owner=term.name, where="test.py:1")


def settled(facts):
    """Run the isl oracle over the facts it is willing to take."""
    oracle = IslOracle()
    out = []
    for fact in facts:
        if oracle.can_establish(fact):
            fact = oracle.establish(fact) or fact
        out.append(fact)
    return out


def shift(u: Arr[Fin[n], Real], v: Arr[Fin[n], Real]):  # noqa: F821
    for i in u.dom:
        v[i] = u[i + 1]


def copy(u: Arr[Fin[n], Real], v: Arr[Fin[n], Real]):  # noqa: F821
    for i in u.dom:
        v[i] = u[i]


def test_an_affine_access_states_a_subset_obligation() -> None:
    _, facts = facts_of(copy)
    in_bounds = [f for f in facts if f.kind == "in-bounds"]
    assert len(in_bounds) == 2
    assert all(isinstance(f.term, Subset) for f in in_bounds)
    assert all(f.status is Status.ASSUMED for f in in_bounds)


def test_isl_decides_the_affine_obligations() -> None:
    _, facts = facts_of(copy)
    decided = settled(facts)
    assert all(f.status is Status.DECIDED for f in decided)
    assert {f.decided_by for f in decided} == {"isl"}


def test_an_out_of_bounds_read_is_refuted_with_a_witness() -> None:
    _, facts = facts_of(shift)
    bad = [
        f
        for f in settled(facts)
        if f.kind == "in-bounds" and f.status is Status.REFUTED
    ]
    assert len(bad) == 1
    fact = bad[0]
    assert fact.decided_by == "isl"
    assert fact.provenance["access"] == "u[i + 1]"
    witness = fact.provenance["witness"]
    assert witness is not None and len(witness) == 1
    assert fact.provenance["witness_text"].startswith("[a0=")


# {{{ the reads that are not in the right-hand side


def scatter_past_end(
    col: Arr[Fin[n], Fin[m]],  # noqa: F821
    x: Arr[Fin[n], Real],  # noqa: F821
    y: Arr[Fin[m], Real],  # noqa: F821
):
    for i in x.dom:
        y[col[i + 1]] = x[i]


def scatter(
    col: Arr[Fin[n], Fin[m]],  # noqa: F821
    x: Arr[Fin[n], Real],  # noqa: F821
    y: Arr[Fin[m], Real],  # noqa: F821
):
    for i in x.dom:
        y[col[i]] = x[i]


def gated_past_the_end(
    flag: Arr[Fin[n], Nat],  # noqa: F821
    y: Arr[Fin[n], Real],  # noqa: F821
):
    for i in y.dom:
        with when(flag[i + 1] != 0):
            y[i] = 2.0


def in_bounds_of(fn) -> dict[str, object]:
    """The in-bounds facts of a traced function, by the access they are about."""
    _, facts = facts_of(fn)
    return {
        fact.statement.split(" is ")[0]: fact
        for fact in settled(facts)
        if fact.kind == "in-bounds"
    }


def test_a_read_in_an_assignee_index_is_an_obligation_of_its_own() -> None:
    # The write is in bounds *by type*, from ``col``'s element sort, and that
    # is exactly why the read of ``col`` had to be stated: it is the premise of
    # the write's own discharge, and ``col[i + 1]`` runs off the end.
    facts = in_bounds_of(scatter_past_end)
    assert facts["y[col[i + 1]]"].status is Status.DECIDED
    assert facts["y[col[i + 1]]"].decided_by == "type"

    refuted = facts["col[i + 1]"]
    assert refuted.status is Status.REFUTED
    assert refuted.decided_by == "isl"
    assert refuted.provenance["witness"] is not None
    assert refuted.provenance["witness_text"].startswith("[a0=")


def test_the_same_scatter_with_an_index_in_range_is_decided() -> None:
    facts = in_bounds_of(scatter)
    assert facts["col[i]"].status is Status.DECIDED
    assert {fact.status for fact in facts.values()} == {Status.DECIDED}


def test_an_array_read_in_a_guard_is_refuted_with_a_witness() -> None:
    # The guard is evaluated for every instance, so its accesses are in bounds
    # obligations like any other; ``flag[i + 1]`` at ``i = n - 1`` is not.
    refuted = in_bounds_of(gated_past_the_end)["flag[i + 1]"]
    assert refuted.status is Status.REFUTED
    assert refuted.decided_by == "isl"
    assert refuted.provenance["witness"] is not None
    assert refuted.provenance["witness_text"].startswith("[a0=")


# }}}


def test_the_ordering_obligation_is_a_monotonicity_question() -> None:
    _, facts = facts_of(copy)
    ordering = [f for f in facts if f.kind == "ordering"]
    assert len(ordering) == 1
    assert isinstance(ordering[0].term, Monotone)
    assert settled(ordering)[0].status is Status.DECIDED


def test_the_ledger_of_the_ragged_kernel_names_isl_and_the_type() -> None:
    ledger = check_path(KERNELS / "spmv_min.py")
    deciders = {fact.decided_by for fact in ledger}
    assert "isl" in deciders
    assert "type" in deciders
    assert not ledger.by_status(Status.REFUTED)

    by_type = [f for f in ledger if f.decided_by == "type"]
    statements = " ".join(f.statement for f in by_type)
    assert "x[col[r, j]] is in bounds by type" in statements
    # The exactness class of a ``Real`` reduction is ``approx``; ``reassoc`` is
    # what a schedule lowers it to when it reorders the accumulation.
    assert "is approx" in statements

    # The postcondition is nobody's yet, and says so rather than disappearing.
    posts = [f for f in ledger if f.kind == "postcondition"]
    assert len(posts) == 1
    assert posts[0].status is Status.ASSUMED


def test_the_ragged_in_bounds_fact_is_decided_by_isl() -> None:
    ledger = check_path(KERNELS / "spmv_min.py")
    ragged = [f for f in ledger if f.statement.startswith("val[r, j]")]
    assert len(ragged) == 1
    assert ragged[0].status is Status.DECIDED
    assert ragged[0].decided_by == "isl"


def test_checking_the_wrong_kernel_refutes_one_fact() -> None:
    ledger = check_path(KERNELS / "out_of_bounds.py")
    refuted = ledger.by_status(Status.REFUTED)
    assert len(refuted) == 1
    assert refuted[0].statement.startswith("u[i + 1]")
    assert refuted[0].provenance["witness"] is not None


def test_every_stencil_obligation_is_decided() -> None:
    ledger = check_path(KERNELS / "stencil.py")
    assert len(ledger) == 5
    assert {f.status for f in ledger} == {Status.DECIDED}


def test_the_dense_kernel_file_checks_clean() -> None:
    ledger = check_path(KERNELS / "axpy.py")
    assert len(ledger) >= 3
    assert not ledger.by_status(Status.REFUTED)


def test_the_command_prints_the_ledger_and_exits_zero(capsys) -> None:
    from lanky.cli import main

    assert main(["check", str(KERNELS / "spmv_min.py")]) == 0
    printed = capsys.readouterr().out
    assert "STATUS" in printed
    assert "isl" in printed
    assert "type" in printed


def test_the_command_exits_one_when_a_fact_is_refuted(capsys) -> None:
    from lanky.cli import main

    assert main(["check", str(KERNELS / "out_of_bounds.py")]) == 1
    assert "refuted" in capsys.readouterr().out


# {{{ a guard's reads happen over the loop nest, not the narrowed domain


def gated_by_a_conjunction(
    flag: Arr[Fin[n], Nat],  # noqa: F821
    y: Arr[Fin[n], Real],  # noqa: F821
):
    row = y.dom
    for i in row:
        with when((i + 1 < row.size) & (flag[i + 1] != 0)):
            y[i] = 2.0


def test_a_guard_read_is_stated_over_the_loop_nest_not_the_narrowed_domain() -> None:
    # ``when`` evaluates its whole condition at every point and only masks the
    # write, so ``flag[i + 1]`` is read at ``i = n - 1`` however the affine
    # conjunct narrows the write's domain. Stating the read over the narrowed
    # domain would prove it in bounds by the very condition that does not
    # protect it.
    facts = in_bounds_of(gated_by_a_conjunction)
    assert facts["flag[i + 1]"].status is Status.REFUTED
    assert facts["flag[i + 1]"].provenance["witness"] is not None
    assert facts["y[i]"].status is Status.DECIDED


def guarded_shift(u: Arr[Fin[n], Real], v: Arr[Fin[n], Real]):  # noqa: F821
    row = u.dom
    for i in row:
        with when(i + 1 < row.size):
            v[i] = u[i + 1]


def test_the_body_of_a_guarded_statement_still_enjoys_the_narrowing() -> None:
    # The write and the right-hand side happen only where the guard holds, so
    # the narrowed domain is the right one for them; only the guard's own reads
    # are different.
    facts = in_bounds_of(guarded_shift)
    assert facts["u[i + 1]"].status is Status.DECIDED
    assert {fact.status for fact in facts.values()} == {Status.DECIDED}


# }}}
