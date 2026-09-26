"""Footprints, the padded instance space, and dependences defined from them."""

from __future__ import annotations

import islpy as isl
from lanky.prelude import Int, Nat, Real
from lanky.terms import evaluate_annotations

from loopty import Arr, Fin, flow, reduce_sum, when
from loopty.trace import trace
from loopty.typing import facts_for


def term_of(fn):
    """Trace a plain function the way ``@kernel`` would."""
    return trace(fn, evaluate_annotations(fn))


def scan(cnt: Arr[Fin[n], Nat], off: Arr[Fin[n + 1], Nat]):  # noqa: F821
    off[0] = 0
    for r in cnt.dom:
        off[r + 1] = off[r] + cnt[r]


def jacobi(u: Arr[Fin[nt], Fin[nx], Real]):  # noqa: F821
    steps = u.dom
    for t in steps:
        row = u.dom[t]
        for i in row:
            with when((t + 1 < steps.size) & (i > 0) & (i + 1 < row.size)):
                u[t + 1, i] = (u[t, i - 1] + u[t, i + 1]) / 2


def test_footprints_name_the_array_and_the_kind() -> None:
    prints = flow.footprints(term_of(scan))
    assert {(f.stmt, f.array, f.kind) for f in prints} == {
        ("S0", "off", "write"),
        ("S1", "off", "write"),
        ("S1", "off", "read"),
        ("S1", "cnt", "read"),
    }


def test_every_instance_lives_in_one_padded_space() -> None:
    term = term_of(scan)
    assert flow.instance_space_depth(term) == 1
    instances = flow.instance_domain(term.stmts[0], 0, 1)
    # The statement outside the loop is one point, tagged with its index.
    assert instances.is_equal(isl.Set("{ [s = 0, d0 = 0] }"))


def test_the_scan_carries_a_dependence_from_each_row_to_the_next() -> None:
    deps = flow.dependences(term_of(scan))
    expected = isl.Map(
        "[n] -> { [s = 1, d0] -> [s' = 1, d0' = d0 + 1] : 0 <= d0 < n - 1;"
        "         [s = 0, d0 = 0] -> [s' = 1, d0' = 0] : n > 0 }"
    )
    assert deps.is_equal(expected.align_params(deps.get_space()))


def test_the_jacobi_dependences_are_the_two_diagonals() -> None:
    deps = flow.dependences(term_of(jacobi))
    # Every dependence advances time by exactly one level.
    times = deps.deltas()
    assert times.is_subset(isl.Set("{ [ds = 0, dt = 1, di] : di = 1 or di = -1 }"))
    assert not times.is_empty()


def test_a_reads_only_array_carries_no_dependence() -> None:
    term = term_of(scan)
    prints = [f for f in flow.footprints(term) if f.array == "cnt"]
    assert prints and all(not f.touches_memory for f in prints)


def test_the_source_schedule_is_the_2d_plus_1_vector() -> None:
    schedule = flow.schedule_of(term_of(scan))
    assert schedule.dim(isl.dim_type.out) == 3
    # The statement before the loop runs first, at time [0, 0, 0].
    point = schedule.intersect_domain(isl.Set("{ [s = 0, d0 = 0] }")).range()
    assert point.is_equal(isl.Set("{ [0, 0, 0] }"))


def rolling(u: Arr[Fin[n], Real]):  # noqa: F821
    for i in u.dom:
        with when(i + 1 < u.dom.size):
            u[i] = u[i] + u[i + 1]


def signed_guard(a: Int, u: Arr[Fin[n], Real]):  # noqa: F821
    for i in u.dom:
        with when(a < 0):
            u[i] = 1.0


def test_a_guard_on_a_signed_scalar_does_not_empty_the_domain() -> None:
    # Every parameter of a domain used to be assumed non-negative, which is
    # right for an extent and wrong for a signed scalar a guard mentions:
    # ``a < 0`` and ``a >= 0`` together make the domain empty, and an empty
    # domain discharges every obligation over it vacuously.
    term = term_of(signed_guard)
    domain = term.stmts[0].domain
    assert not domain.is_empty()
    assert "a >= 0" not in str(domain)
    # The extent is still assumed non-negative; only the scalar is not.
    assert domain.is_subset(
        isl.Set("[a, n] -> { [i] : a < 0 and n >= 0 and 0 <= i < n }")
    )


def test_the_facts_of_a_kernel_guarded_on_a_signed_scalar_are_not_vacuous() -> None:
    term = term_of(signed_guard)
    facts = facts_for(term, owner="signed_guard")
    (in_bounds,) = [fact for fact in facts if fact.kind == "in-bounds"]
    # The cells the write reaches is the thing isl is asked about. Empty would
    # mean the obligation holds because the statement never runs.
    assert not flow.assume_sizes(
        in_bounds.term.small, flow.size_names(term)
    ).is_empty()
    (disjoint,) = [fact for fact in facts if fact.kind == "disjoint-writes"]
    assert not flow.footprints(term)[0].relation.is_empty()
    assert disjoint.term is not None


def test_the_schedule_checker_sees_the_same_non_empty_instances() -> None:
    # ``loopty.schedule`` computes its own dependence relation, over the same
    # statement domains, so the assumption has to be the same one there. It is,
    # because the domains are the term's: nothing in schedule.py adds a
    # non-negativity constraint of its own.
    from loopty.schedule import Schedule

    schedule = Schedule(term_of(signed_guard))
    assert not schedule._instances.is_empty()  # noqa: SLF001 - the point of the test


def test_size_names_keeps_extents_and_drops_scalars() -> None:
    term = term_of(signed_guard)
    names = flow.size_names(term)
    assert "n" in names
    assert "a" not in names


# {{{ the one collector: assignee indices and guards are reads too


def scatter_past_end(
    col: Arr[Fin[n], Fin[m]],  # noqa: F821
    x: Arr[Fin[n], Real],  # noqa: F821
    y: Arr[Fin[m], Real],  # noqa: F821
):
    """``y[col[i + 1]] = x[i]``: the write's own index is a read of ``col``."""
    for i in x.dom:
        y[col[i + 1]] = x[i]


def gated(flag: Arr[Fin[n], Nat], y: Arr[Fin[n], Real]):  # noqa: F821
    """S0 writes the flag S1 is guarded on, so the guard read is a dependence."""
    for i in y.dom:
        flag[i] = 1
        with when(flag[i] != 0):
            y[i] = 2.0


def rendered(accesses) -> set[tuple[str, str, str]]:
    """``(kind, array, "i + 1")`` for the schedule checker's own access list.

    Rendered rather than compared as terms: ``==`` on a lanky variable builds a
    proposition instead of answering a bool.
    """
    from lanky.terms import render

    return {
        (kind, array, ", ".join(render(i) for i in indices))
        for kind, array, indices in accesses
    }


def touched(stmt, term) -> set[tuple[str, str]]:
    """``(kind, "array[index, ...]")`` for every access the collector reports."""
    from lanky.terms import render

    return {
        (kind, f"{array}[{', '.join(render(i) for i in indices)}]")
        for array, indices, kind, _inames, _domain in flow.statement_accesses(
            stmt, term
        )
    }


def test_an_array_read_inside_an_assignee_index_is_collected_as_a_read() -> None:
    # The write to ``y`` is discharged in bounds *by type* from ``col``'s
    # element sort, so with the read of ``col`` itself missing nothing ever
    # asked whether the kernel reads past the end of ``col``.
    term = term_of(scatter_past_end)
    (stmt,) = term.stmts
    assert ("write", "y[col[i + 1]]") in touched(stmt, term)
    assert ("read", "col[i + 1]") in touched(stmt, term)


def test_an_array_read_inside_a_guard_is_collected_as_a_read() -> None:
    term = term_of(gated)
    assert touched(term.stmts[0], term) == {("write", "flag[i]")}
    assert touched(term.stmts[1], term) == {("write", "y[i]"), ("read", "flag[i]")}


def test_a_guard_read_carries_a_dependence_from_the_statement_that_writes_it() -> None:
    # Without the guard among the accesses there is no dependence at all here,
    # and a reordering could let the predicate observe the old value.
    deps = flow.dependences(term_of(gated))
    expected = isl.Map("[n] -> { [s = 0, d0] -> [s' = 1, d0' = d0] : 0 <= d0 < n }")
    assert deps.is_equal(expected.align_params(deps.get_space()))


def test_the_schedule_and_the_lowering_see_the_guard_read_as_well() -> None:
    # The four collectors are one function now; this is the check that the
    # other three really route through it.
    from loopty.lower import lower_generic
    from loopty.schedule import _accesses

    term = term_of(gated)
    assert ("read", "flag", "i") in rendered(_accesses(term.stmts[1], term))

    lowering = lower_generic(term)
    insns = {
        insn.id: insn for insn in lowering.kernel.default_entrypoint.instructions
    }
    assert insns["S1"].depends_on == frozenset({"S0"})


def test_an_assignee_index_read_reaches_the_schedule_checker_too() -> None:
    from loopty.schedule import _accesses

    term = term_of(scatter_past_end)
    (stmt,) = term.stmts
    kinds = rendered(_accesses(stmt, term))
    assert ("write", "y", "col[i + 1]") in kinds
    assert ("read", "col", "i + 1") in kinds


# }}}


def test_an_accumulation_still_records_reads_of_its_other_cells() -> None:
    term = term_of(rolling)
    assert term.stmts[0].kind == "accumulate"
    prints = flow.footprints(term)
    assert sorted(f.kind for f in prints) == ["acc", "read"]
    # Instance i reads the cell instance i + 1 overwrites: a real dependence.
    deps = flow.dependences(term)
    assert not deps.is_empty()
    assert deps.deltas().is_subset(isl.Set("{ [ds = 0, di = 1] }"))


# {{{ the reads a ragged access makes through its offsets


def scan_then_spmv(
    cnt: Arr[Fin[n], Nat],  # noqa: F821
    off: Arr[Fin[n + 1], Nat],  # noqa: F821
    col: Arr[Fin[n], Fin[cnt], Fin[m]],  # noqa: F821
    val: Arr[Fin[n], Fin[cnt], Real],  # noqa: F821
    x: Arr[Fin[m], Real],  # noqa: F821
    y: Arr[Fin[n], Real],  # noqa: F821
):
    """``scan`` and ``spmv`` of ``examples/spmv.py``, fused into one kernel.

    Lowered, ``val[i, j]`` and ``col[i, j]`` are ``val[off[i] + j]`` and
    ``col[off[i] + j]``: ``S2`` reads the offsets ``S1`` writes, and nothing in
    its source says so.
    """
    off[0] = 0
    for r in cnt.dom:
        off[r + 1] = off[r] + cnt[r]
    for i in y.dom:
        y[i] = reduce_sum(val[i, j] * x[col[i, j]] for j in val.dom[i])


def scale_rows(
    cnt: Arr[Fin[n], Nat],  # noqa: F821
    off: Arr[Fin[n + 1], Nat],  # noqa: F821
    val: Arr[Fin[n], Fin[cnt], Real],  # noqa: F821
):
    """A ragged array written in place, through the offsets it declares."""
    for r in cnt.dom:
        for j in val.dom[r]:
            val[r, j] = 2.0 * val[r, j]


def spmv_without_offsets(
    cnt: Arr[Fin[n], Nat],  # noqa: F821
    col: Arr[Fin[n], Fin[cnt], Fin[m]],  # noqa: F821
    val: Arr[Fin[n], Fin[cnt], Real],  # noqa: F821
    x: Arr[Fin[m], Real],  # noqa: F821
    y: Arr[Fin[n], Real],  # noqa: F821
):
    """``spmv`` as the example writes it: the offsets are lowering's to add."""
    for r in y.dom:
        y[r] = reduce_sum(val[r, j] * x[col[r, j]] for j in val.dom[r])


def offsets_entries(stmt, term) -> list[tuple[str, str]]:
    """``(kind, "off[...]")`` for each offsets entry, in order, repeats kept."""
    from lanky.terms import render

    return [
        (kind, f"{array}[{', '.join(render(i) for i in indices)}]")
        for array, indices, kind, _inames, _domain in flow.statement_accesses(
            stmt, term
        )
        if array == "off"
    ]


def test_a_ragged_access_reads_the_start_of_its_row_in_the_offsets() -> None:
    # ``off[i]`` is what the flat index reads. ``val`` and ``col`` share their
    # offsets, so it is listed once for the two of them. ``off[i + 1]``, where
    # the row ends, is not read by anything here: the counts are a parameter,
    # and the loop over the row runs to ``cnt[i]``.
    term = term_of(scan_then_spmv)
    product = term.stmts[2]
    assert offsets_entries(product, term) == [("read", "off[i]")]
    assert {
        ("write", "y[i]"),
        ("read", "val[i, j]"),
        ("read", "x[col[i, j]]"),
        ("read", "col[i, j]"),
    } <= touched(product, term)
    # The scan names ``off`` itself, and touches no ragged array.
    assert offsets_entries(term.stmts[1], term) == [
        ("write", "off[r + 1]"),
        ("read", "off[r]"),
    ]


def test_a_ragged_write_reads_the_offsets_too() -> None:
    # ``val[r, j] = ...`` stores to ``val[off[r] + j]``: the write goes through
    # the offsets as the read does, and the two share one read.
    term = term_of(scale_rows)
    (stmt,) = term.stmts
    assert offsets_entries(stmt, term) == [("read", "off[r]")]


def test_offsets_the_kernel_does_not_declare_are_not_listed() -> None:
    # Lowering adds ``off_cnt`` itself; nothing in the body can write it, and
    # ``r`` in bounds of ``val``'s rows already keeps the read of it in
    # bounds. Listing it would add a footprint on an array the term lacks.
    # The row's length is read from ``cnt``, which the kernel does declare.
    term = term_of(spmv_without_offsets)
    (stmt,) = term.stmts
    arrays = {array for array, *_ in flow.statement_accesses(stmt, term)}
    assert arrays == {"y", "val", "x", "col", "cnt"}


def test_the_scan_carries_a_dependence_to_the_rows_read_through_it() -> None:
    # ``S1[r]`` writes ``off[r + 1]``, which is where row ``r + 1`` starts, so
    # that row of ``S2`` depends on it. Without the offsets among ``S2``'s
    # accesses there was no dependence at all between the scan and the
    # product.
    term = term_of(scan_then_spmv)
    deps = flow.dependences(term)
    assert not deps.intersect(
        isl.Map("{ [1, 0] -> [2, 1] }").align_params(deps.get_space())
    ).is_empty()
    # Row ``0`` ends at ``off[1]``, but nothing reads where a row ends when the
    # counts are a parameter, and row ``2`` does not start there.
    for pair in ("{ [1, 0] -> [2, 0] }", "{ [1, 0] -> [2, 2] }"):
        assert deps.intersect(
            isl.Map(pair).align_params(deps.get_space())
        ).is_empty(), pair


def test_the_schedule_checker_draws_the_same_dependence_through_the_offsets() -> None:
    # The checker computes its own relation, to be able to name the cell of a
    # refused cast; it reads the same collector, so it names ``off`` too.
    from loopty.schedule import Schedule

    schedule = Schedule(term_of(scan_then_spmv))
    through_offsets = [
        dep
        for dep in schedule._deps  # noqa: SLF001 - the point of the test
        if dep.array == "off" and (dep.source, dep.sink) == ("S1", "S2")
    ]
    assert [dep.kind for dep in through_offsets] == ["raw"]


# }}}


# {{{ the read a ragged loop's bound makes


def counts_rewritten_ahead(
    cnt: Arr[Fin[n], Nat],  # noqa: F821
    val: Arr[Fin[n], Fin[cnt], Real],  # noqa: F821
    y: Arr[Fin[n], Real],  # noqa: F821
):
    """``S0`` sums row ``r``, whose length is ``cnt[r]``; ``S1`` then clears
    the length of the next row."""
    for r in y.dom:
        y[r] = reduce_sum(val[r, j] for j in val.dom[r])
        with when(r + 1 < y.dom.size):
            cnt[r + 1] = 0


def guarded_rows(
    cnt: Arr[Fin[n], Nat],  # noqa: F821
    val: Arr[Fin[n], Fin[cnt], Real],  # noqa: F821
    z: Arr[Fin[n], Fin[cnt], Real],  # noqa: F821
):
    """A loop over the fiber of every row, with a guard on the row."""
    for r in cnt.dom:
        for j in val.dom[r]:
            with when(r > 0):
                z[r, j] = val[r, j]


def same_loops(domain: str, expected: str) -> bool:
    """Whether ``domain`` is ``expected`` wherever its parameters may be.

    The constraints a set keeps on its parameters alone (``n >= 0``) are not
    what is being compared.
    """
    got = isl.Set(domain)
    want = isl.Set(expected).align_params(got.get_space())
    got = got.align_params(want.get_space())
    return got.is_equal(want.intersect_params(got.params()))


def bound_entries(stmt, term) -> list[tuple[tuple[str, ...], str, str]]:
    """``(inames, "cnt[...]", domain)`` for each read of ``cnt``, in order."""
    from lanky.terms import render

    return [
        (inames, f"{array}[{', '.join(render(i) for i in indices)}]", str(domain))
        for array, indices, kind, inames, domain in flow.statement_accesses(
            stmt, term
        )
        if array == "cnt" and kind == "read"
    ]


def test_a_ragged_loop_reads_its_bound_from_the_counts() -> None:
    # Lowering assigns the reduction's bound ``cnt[r]`` inside the loop over
    # ``r``; that read is in no expression of the statement, and it used to
    # be in no list either.
    term = term_of(counts_rewritten_ahead)
    ((inames, text, domain),) = bound_entries(term.stmts[0], term)
    assert (inames, text) == (("r",), "cnt[r]")
    assert same_loops(domain, "[n] -> { [r] : 0 <= r < n }")
    (layout,) = flow.layout_reads(term.stmts[0], term)
    assert (layout.part, layout.loops, layout.access) == ("length", ("j",), None)


def test_the_bound_is_read_over_the_loops_up_to_its_row() -> None:
    # Once per row, whether or not the row has entries and whatever the guard
    # inside the loop says: the statement's domain is over ``j`` and narrowed
    # to ``r > 0``, and the read of ``cnt[r]`` is over every ``r``.
    term = term_of(guarded_rows)
    (stmt,) = term.stmts
    ((inames, text, domain),) = bound_entries(stmt, term)
    assert (inames, text) == (("r",), "cnt[r]")
    assert same_loops(domain, "[n] -> { [r] : 0 <= r < n }")


def test_a_writer_of_the_counts_is_ordered_against_the_rows_they_bound() -> None:
    # ``S1[r]`` writes ``cnt[r + 1]``, the length of the row ``S0[r + 1]``
    # sums. The dependence is there only because the bound's read is listed.
    term = term_of(counts_rewritten_ahead)
    deps = flow.dependences(term)
    assert not deps.intersect(
        isl.Map("{ [1, 0] -> [0, 1] }").align_params(deps.get_space())
    ).is_empty()


def test_a_bound_computed_from_the_offsets_reads_both_ends_of_the_row() -> None:
    # A term built by hand whose counts are not a parameter: lowering computes
    # the row's length as ``off[r + 1] - off[r]``, so both are read, once per
    # row, and ``off[r]`` a second time by the flat index, over the reduction.
    import hand_terms as ht

    term = ht.spmv_term()
    (stmt,) = term.stmts
    parts = [
        (layout.part, layout.read[3], layout.loops, layout.access is None)
        for layout in flow.layout_reads(stmt, term)
    ]
    assert parts == [
        ("start", ("r", "j"), (), False),
        ("start", ("r", "j"), (), False),
        ("start", ("r",), ("j",), True),
        ("end", ("r",), ("j",), True),
    ]
    assert offsets_entries(stmt, term) == [
        ("read", "off[r]"),
        ("read", "off[r]"),
        ("read", "off[r + 1]"),
    ]


# }}}
