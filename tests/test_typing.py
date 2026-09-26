"""The typing rules and the isl oracle: which obligation is settled by whom."""

from __future__ import annotations

from pathlib import Path

from lanky.check import check_path
from lanky.ledger import Status
from lanky.prelude import Nat, Real
from lanky.terms import evaluate_annotations

from loopty import Arr, Fin, reduce_sum, when
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
    """The in-bounds facts of a traced function, by the access they are about.

    The access is the last part of the fact's id, ``...:read:off[r]``: the
    statement of an offsets read says more than the access (see
    :func:`test_an_offsets_fact_names_the_ragged_access_it_serves`).
    """
    _, facts = facts_of(fn)
    return {
        fact.id.rsplit(":", 1)[-1]: fact
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
    assert len(ledger) == 6
    # Five obligations about the term, and the faithfulness fact about whether
    # the term is the body, which running both can only test.
    *obligations, faithful = ledger
    assert {f.status for f in obligations} == {Status.DECIDED}
    assert faithful.kind == "trace-faithful"
    assert faithful.status is Status.TESTED


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


def accumulates_where_the_guard_reads(
    y: Arr[Fin[n], Real],  # noqa: F821
):
    row = y.dom
    for i in row:
        with when((i + 1 < row.size) & (y[i + 1] != 0)):
            y[i + 1] = y[i + 1] + 1.0


def test_a_guard_read_of_the_accumulated_cell_is_still_an_obligation() -> None:
    # The right-hand side's read of ``y[i + 1]`` is covered by the ``acc``
    # footprint over the narrowed domain; the guard's read of the same cell is
    # not, because the guard is evaluated at ``i = n - 1`` too, so it keeps its
    # own in-bounds fact and that fact is refuted.
    _, facts = facts_of(accumulates_where_the_guard_reads)
    refuted = [
        f
        for f in settled(facts)
        if f.kind == "in-bounds" and f.status is Status.REFUTED
    ]
    assert refuted
    assert {f.provenance["access"] for f in refuted} == {"y[i + 1]"}


# {{{ a reduction nested in another one


def nested_total(
    a: Arr[Fin[n], Fin[m], Real],  # noqa: F821
    s: Arr[Fin[1], Real],
):
    s[0] = reduce_sum(reduce_sum(a[i, j] for j in a.dom[i]) for i in a.dom)


def lower_triangle(
    a: Arr[Fin[n], Fin[n], Real],  # noqa: F821
    s: Arr[Fin[1], Real],
):
    s[0] = reduce_sum(reduce_sum(a[i, j] for j in Fin[i + 1]) for i in a.dom)


def ragged_total(
    cnt: Arr[Fin[n], Nat],  # noqa: F821
    val: Arr[Fin[n], Fin[cnt], Real],  # noqa: F821
    s: Arr[Fin[1], Real],
):
    s[0] = reduce_sum(reduce_sum(val[r, j] for j in val.dom[r]) for r in val.dom)


def past_the_diagonal(
    a: Arr[Fin[n], Fin[n], Real],  # noqa: F821
    s: Arr[Fin[1], Real],
):
    s[0] = reduce_sum(reduce_sum(a[i, j] for j in Fin[i + 2]) for i in a.dom)


def test_a_read_in_a_nested_reduction_is_bounded_by_the_outer_binder() -> None:
    # The inner reduction runs inside the outer one's binder, and its domain did
    # not say so: ``i`` was an unconstrained parameter there, and ``a[i, j]``
    # was refuted at ``i = -1`` for a sum that never leaves the array. A ragged
    # inner bound, ``cnt[r]`` of the outer binder, is the same case.
    for fn, access in (
        (nested_total, "a[i, j]"),
        (lower_triangle, "a[i, j]"),
        (ragged_total, "val[r, j]"),
    ):
        fact = in_bounds_of(fn)[access]
        assert fact.status is Status.DECIDED, (fn.__name__, fact.provenance)


def test_a_nested_reduction_that_leaves_the_array_is_still_refuted() -> None:
    # ``j < i + 2`` reaches one column past the diagonal of the last row.
    fact = in_bounds_of(past_the_diagonal)["a[i, j]"]
    assert fact.status is Status.REFUTED
    assert fact.provenance["witness"] is not None


# }}}


# {{{ the offsets a ragged access reads


def spmv_through_offsets(
    cnt: Arr[Fin[n], Nat],  # noqa: F821
    off: Arr[Fin[n + 1], Nat],  # noqa: F821
    col: Arr[Fin[n], Fin[cnt], Fin[m]],  # noqa: F821
    val: Arr[Fin[n], Fin[cnt], Real],  # noqa: F821
    x: Arr[Fin[m], Real],  # noqa: F821
    y: Arr[Fin[n], Real],  # noqa: F821
):
    for r in y.dom:
        y[r] = reduce_sum(val[r, j] * x[col[r, j]] for j in val.dom[r])


def spmv_through_short_offsets(
    cnt: Arr[Fin[n], Nat],  # noqa: F821
    off: Arr[Fin[n], Nat],  # noqa: F821
    col: Arr[Fin[n], Fin[cnt], Fin[m]],  # noqa: F821
    val: Arr[Fin[n], Fin[cnt], Real],  # noqa: F821
    x: Arr[Fin[m], Real],  # noqa: F821
    y: Arr[Fin[n], Real],  # noqa: F821
):
    for r in y.dom:
        y[r] = reduce_sum(val[r, j] * x[col[r, j]] for j in val.dom[r])


def test_the_offsets_read_through_are_decided_in_bounds() -> None:
    # ``val[r, j]`` is ``val[off[r] + j]`` once lowered, a read that is not in
    # the source, and a true fact nobody stated: ``off`` has a cell for every
    # row start. Where row ``r`` ends, ``off[r + 1]``, is not read at all,
    # since the row's length is read from ``cnt``, so it is not an obligation.
    facts = in_bounds_of(spmv_through_offsets)
    assert facts["off[r]"].status is Status.DECIDED, facts["off[r]"].provenance
    assert facts["off[r]"].decided_by == "isl"
    assert facts["off[r]"].statement.endswith("for every instance of S0")
    assert "off[r + 1]" not in facts


def test_offsets_a_cell_short_are_in_bounds_where_only_row_starts_are_read() -> None:
    # The lowered code reads ``off[r]`` for ``r < n`` and nothing past it, so
    # ``n`` offsets are enough for it. Its ledger used to refute ``off[r + 1]``,
    # a read the code does not make. A call is still refused by the contract,
    # because offsets of ``n`` cells are not the layout of ``n`` rows.
    facts = in_bounds_of(spmv_through_short_offsets)
    assert facts["off[r]"].status is Status.DECIDED
    assert "off[r + 1]" not in facts
    assert not [name for name, fact in facts.items() if fact.status is Status.REFUTED]


def short_offsets_hand_term():
    """:func:`hand_terms.spmv_term` with ``off`` declared a cell short.

    Its counts are not a parameter, so lowering computes a row's length as
    ``off[r + 1] - off[r]``, and the end of the last row is a cell ``off``
    does not have.
    """
    import dataclasses

    import hand_terms as ht

    term = ht.spmv_term()
    params = tuple(
        (name, ht.dense(ht.V("n"), dtype=ht.INT) if name == "off" else typ)
        for name, typ in term.params
    )
    return dataclasses.replace(term, params=params)


def test_offsets_a_row_length_is_read_from_are_refuted_a_cell_short() -> None:
    # Nothing else in the ledger notices, because the source never names
    # ``off``.
    term = short_offsets_hand_term()
    facts = {
        fact.id.rsplit(":", 1)[-1]: fact
        for fact in settled(rules.facts_for(term, owner=term.name, where="t:1"))
        if fact.kind == "in-bounds"
    }
    assert facts["off[r]"].status is Status.DECIDED
    refuted = facts["off[r + 1]"]
    assert refuted.status is Status.REFUTED
    assert refuted.decided_by == "isl"
    assert refuted.provenance["witness"] is not None
    assert refuted.provenance["witness_text"].startswith("[a0=")
    # The source never writes ``off[r + 1]``, so the fact says what it serves.
    assert refuted.statement == (
        "off[r + 1], the end of row r whose length bounds the loop over j, is "
        "in bounds for every instance of S0"
    )
    assert refuted.provenance["layout"] == (
        "the end of row r whose length bounds the loop over j"
    )
    assert facts["off[r]"].provenance["layout"] == (
        "the start of row r that val[r, j] and col[r, j] are flattened through "
        "and the start of row r whose length bounds the loop over j"
    )


def test_an_offsets_fact_names_the_ragged_access_it_serves() -> None:
    # The source never writes ``off[r]``, so a fact about it used to name a
    # read nobody could find in the kernel.
    facts = in_bounds_of(spmv_through_short_offsets)
    served = "val[r, j] and col[r, j] are flattened through"
    assert facts["off[r]"].statement == (
        f"off[r], the start of row r that {served}, is in bounds for every "
        "instance of S0"
    )
    assert facts["off[r]"].provenance["layout"] == f"the start of row r that {served}"
    # An access the source spells keeps its statement, and its provenance.
    assert facts["val[r, j]"].statement == (
        "val[r, j] is in bounds for every instance of S0"
    )
    assert "layout" not in facts["val[r, j]"].provenance


def spmv_short_counts(
    cnt: Arr[Fin[n - 1], Nat],  # noqa: F821
    val: Arr[Fin[n], Fin[cnt], Real],  # noqa: F821
    y: Arr[Fin[n], Real],  # noqa: F821
):
    for r in y.dom:
        y[r] = reduce_sum(val[r, j] for j in val.dom[r])


def test_a_ragged_loop_bound_is_an_obligation_of_its_own() -> None:
    # The loop over ``val.dom[r]`` runs to ``cnt[r]``, which the lowered code
    # reads once per row. That read is in no expression of the body.
    facts = in_bounds_of(spmv_through_offsets)
    fact = facts["cnt[r]"]
    assert fact.status is Status.DECIDED
    assert fact.decided_by == "isl"
    assert fact.statement == (
        "cnt[r], the length of row r that bounds the loop over j, is in bounds "
        "for every instance of S0"
    )
    assert fact.provenance["layout"] == (
        "the length of row r that bounds the loop over j"
    )


def previous_row_sums(
    cnt: Arr[Fin[n], Nat],  # noqa: F821
    val: Arr[Fin[n], Fin[cnt], Real],  # noqa: F821
    y: Arr[Fin[n], Real],  # noqa: F821
):
    for r in y.dom:
        y[r] = reduce_sum(val[r - 1, j] for j in val.dom[r - 1])


def indirect_row_sums(
    cnt: Arr[Fin[n], Nat],  # noqa: F821
    p: Arr[Fin[k], Fin[n]],  # noqa: F821
    val: Arr[Fin[n], Fin[cnt], Real],  # noqa: F821
    y: Arr[Fin[k], Real],  # noqa: F821
):
    for i in y.dom:
        y[i] = reduce_sum(val[p[i], j] for j in val.dom[p[i]])


def test_the_length_of_another_row_is_read_where_its_loop_starts() -> None:
    # ``val.dom[r - 1]`` runs to ``cnt[r - 1]``, read at every ``r``, and at
    # ``r = 0`` that is a cell ``cnt`` does not have.
    fact = in_bounds_of(previous_row_sums)["cnt[r - 1]"]
    assert fact.status is Status.REFUTED
    assert fact.statement == (
        "cnt[r - 1], the length of row r - 1 that bounds the loop over j, is in "
        "bounds for every instance of S0"
    )
    assert fact.provenance["witness_text"].startswith("[a0=-1]")


def test_a_bound_inside_a_sum_is_read_only_where_that_sum_runs() -> None:
    # The inner sum runs for ``q < r``, so only for ``r >= 1``, and reads
    # ``cnt[r - 1]`` there alone.
    fact = in_bounds_of(offsets_before_and_through_a_sum)["cnt[r - 1]"]
    assert fact.status is Status.DECIDED, fact.provenance


def test_the_length_of_an_indirect_row_is_in_bounds_by_type() -> None:
    # ``cnt[p[i]]`` is in bounds because ``p``'s entries are points of
    # ``Fin[n]``, and ``p[i]`` is read to find the row, as well as directly.
    facts = in_bounds_of(indirect_row_sums)
    length = facts["cnt[p[i]]"]
    assert length.status is Status.DECIDED
    assert length.decided_by == "type"
    assert length.statement.startswith(
        "cnt[p[i]], the length of row p[i] that bounds the loop over j, is in "
        "bounds by type"
    )
    row = facts["p[i]"]
    assert row.status is Status.DECIDED
    assert row.statement.startswith(
        "p[i], read directly and as the index of the row whose length bounds "
        "the loop over j, is in bounds"
    )


def test_counts_declared_a_cell_short_are_refuted_at_the_last_row() -> None:
    # Nothing else noticed: ``val[r, j]`` is decided against the row's length,
    # whatever that length is, and nothing in the body names ``cnt``.
    facts = in_bounds_of(spmv_short_counts)
    refuted = facts["cnt[r]"]
    assert refuted.status is Status.REFUTED
    assert refuted.provenance["witness_text"].startswith("[a0=")
    assert [
        name for name, fact in facts.items() if fact.status is Status.REFUTED
    ] == ["cnt[r]"]


def indirect_rows(
    cnt: Arr[Fin[n], Nat],  # noqa: F821
    off: Arr[Fin[n + 1], Nat],  # noqa: F821
    p: Arr[Fin[k], Fin[n]],  # noqa: F821
    val: Arr[Fin[n], Fin[cnt], Real],  # noqa: F821
    y: Arr[Fin[k], Real],  # noqa: F821
):
    for i in y.dom:
        y[i] = val[p[i], 0]


def test_an_assumed_offsets_fact_names_its_access_too() -> None:
    facts = in_bounds_of(indirect_rows)
    fact = facts["off[p[i]]"]
    assert fact.status is Status.ASSUMED
    assert fact.statement == (
        "off[p[i]], the start of row p[i] that val[p[i], 0] is flattened "
        "through, is in bounds"
    )
    # No loop runs over a row here, so nothing reads a row's end or length.
    assert "off[p[i] + 1]" not in facts
    assert not [access for access in facts if access.startswith("cnt[")]


def test_an_offsets_read_the_source_also_spells_says_both() -> None:
    fact = only_fact(offsets_before_and_through_a_sum, "off[r - 1]")
    assert fact.statement.startswith(
        "off[r - 1], read directly and as the start of row r - 1 that "
        "val[r - 1, j] is flattened through, is in bounds"
    )


# }}}


# {{{ one access listed over several domains


def before_and_inside_a_sum(x: Arr[Fin[n], Real], y: Arr[Fin[n], Real]):  # noqa: F821
    for r in y.dom:
        y[r] = x[r - 1] + reduce_sum(x[r - 1] for q in Fin[r])


def offsets_before_and_through_a_sum(
    cnt: Arr[Fin[n], Nat],  # noqa: F821
    off: Arr[Fin[n + 1], Nat],  # noqa: F821
    val: Arr[Fin[n], Fin[cnt], Real],  # noqa: F821
    y: Arr[Fin[n], Real],  # noqa: F821
):
    for r in y.dom:
        y[r] = off[r - 1] + reduce_sum(
            reduce_sum(val[r - 1, j] for j in val.dom[r - 1]) for q in Fin[r]
        )


def only_fact(fn, access: str):
    """The one in-bounds fact about ``access``, checked to be the only one."""
    _, facts = facts_of(fn)
    (fact,) = [
        fact
        for fact in settled(facts)
        if fact.kind == "in-bounds" and fact.id.endswith(f":read:{access}")
    ]
    return fact


def test_an_access_read_over_two_domains_is_one_obligation_over_both() -> None:
    # The direct read reaches ``x[-1]`` at ``r = 0``; the read inside the sum
    # only runs for ``r >= 1``. Both have the id ``S0:read:x[r - 1]``, and the
    # ledger keeps one fact per id: stated separately, the second, which is in
    # bounds, would take the place of the first.
    fact = only_fact(before_and_inside_a_sum, "x[r - 1]")
    assert fact.status is Status.REFUTED
    assert fact.provenance["witness_text"].startswith("[a0=-1] ")


def test_the_offsets_a_sum_reads_through_do_not_hide_a_direct_read() -> None:
    # The same collision, through the layout: ``val[r - 1, j]`` in the sum
    # reads ``off[r - 1]`` for ``r >= 1`` only, and the direct read of
    # ``off[r - 1]`` reaches ``off[-1]`` at ``r = 0``.
    fact = only_fact(offsets_before_and_through_a_sum, "off[r - 1]")
    assert fact.status is Status.REFUTED
    assert fact.provenance["witness_text"].startswith("[a0=-1] ")


# }}}


# {{{ a guard isl cannot state


def test_a_fact_over_a_domain_a_guard_left_wide_says_so() -> None:
    # i < a with a : Real is not a constraint isl can state, so the domain is
    # the whole loop nest and the facts about it are about masked instances too.
    def below(a: Real, y: Arr[Fin[n], Real]):  # noqa: F821
        for i in y.dom:
            with when(i < a):
                y[i] = 1.0

    _term, facts = facts_of(below)
    kinds = {fact.kind: fact for fact in facts}
    (entry,) = kinds["in-bounds"].provenance["unnarrowed"]
    assert entry["conjunct"] == "i < a"
    assert "the scalar a of sort Real" in entry["why"]
    assert kinds["disjoint-writes"].provenance["unnarrowed"] == [entry]
    assert kinds["ordering"].provenance["unnarrowed"] == {"S0": [entry]}
    # Wider is harder, never easier: y[i] is still in bounds everywhere.
    (write,) = [f for f in settled(facts) if f.kind == "in-bounds"]
    assert write.status is Status.DECIDED


def test_a_real_guard_no_longer_discharges_an_obligation() -> None:
    # Only the guard keeps x[i] in bounds: i < a < m. Read with a as an
    # integer parameter, that was decided; left out of the domain, the read is
    # refuted at an instance the guard masks, and the fact says the domain is
    # wide. Sound, if not sharp: a Real a no longer proves anything about i.
    def clipped(a: Real, x: Arr[Fin[m], Real], y: Arr[Fin[n], Real]):  # noqa: F821
        for i in y.dom:
            with when((i < a) & (a < x.dom.size)):
                y[i] = x[i]

    _term, facts = facts_of(clipped)
    bounds = {
        fact.provenance["access"]: fact
        for fact in settled(facts)
        if fact.kind == "in-bounds"
    }
    read = bounds["x[i]"]
    assert read.status is Status.REFUTED, read.provenance
    assert [entry["conjunct"] for entry in read.provenance["unnarrowed"]] == [
        "i < a",
        "a < m",
    ]
    assert bounds["y[i]"].status is Status.DECIDED


def test_a_guard_isl_states_whole_leaves_nothing_to_say() -> None:
    def below(a: Nat, y: Arr[Fin[n], Real]):  # noqa: F821
        for i in y.dom:
            with when(i < a):
                y[i] = 1.0

    _term, facts = facts_of(below)
    assert not any("unnarrowed" in fact.provenance for fact in facts)


def test_a_guards_own_read_is_not_over_the_wide_domain() -> None:
    # The guard's read happens at every point of the loop nest, which is the
    # domain it is stated over, so nothing about it is over-approximated.
    def flagged(flag: Arr[Fin[n], Real], y: Arr[Fin[n], Real]):  # noqa: F821
        for i in y.dom:
            with when(flag[i] > 0.0):
                y[i] = 1.0

    _term, facts = facts_of(flagged)
    bounds = {
        fact.provenance["access"]: fact
        for fact in facts
        if fact.kind == "in-bounds"
    }
    assert "unnarrowed" not in bounds["flag[i]"].provenance
    assert bounds["y[i]"].provenance["unnarrowed"] == [
        {"conjunct": "flag[i] > 0.0", "why": "reads an array or is not affine"}
    ]


# }}}
