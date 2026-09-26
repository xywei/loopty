"""Transformations as casts: what is accepted, what is rejected, and why.

The stencil is the case worth reading. Tiling ``(t, i)`` is rejected with a
concrete pair of instances, the same pair a person would find by hand; skewing
the space axis by the time axis and then tiling is accepted, and the tiled code
computes what the untiled code computes. Everything else here is the surrounding
contract: bijectivity is a real question for a reindexing, tagging a loop that
carries a dependence is not allowed, and reassociating an exact accumulation is
not allowed either.
"""

from __future__ import annotations

import dataclasses

import islpy as isl
import numpy as np
import pytest
from lanky.prelude import Nat, Real

import hand_terms as ht
from loopty import Arr, Fin, reduce_sum, when
from loopty.lower import reductions_of
from loopty.schedule import IllegalCast, Schedule, parallel_tag

pytest.importorskip("loopy")


def run(schedule, **arguments):
    """Run a schedule on the C target, or skip if the toolchain is unusable."""
    from loopty.executor import LoopyExecutor

    try:
        return LoopyExecutor().run(schedule, **arguments)
    except Exception as exc:  # pragma: no cover - depends on the local toolchain
        if "compil" in str(exc).lower() or isinstance(exc, OSError):
            pytest.skip(f"the C toolchain path is unusable here: {exc}")
        raise


# {{{ the stencil


def test_tiling_the_stencil_is_rejected_with_a_witness() -> None:
    schedule = Schedule(ht.jacobi_term(), sizes={"nt": 16, "nx": 16})
    with pytest.raises(IllegalCast) as caught:
        schedule.tile("t", "i", 8, 8)

    message = str(caught.value)
    assert message.startswith("tile(t,i,8,8) illegal: instance S0[")
    assert "writes u[" in message
    assert "read by S0[" in message
    # The sizes the witness was read off at are part of the message: the pair
    # isl picks depends on them, and on whether the hint could be honoured.
    assert message.endswith("scheduled earlier (at nt=16, nx=16, as hinted)")

    # The witness is a pair of real instances of the term, not a set difference.
    (source_id, source), (sink_id, sink), params = caught.value.witness
    assert source_id == sink_id == "S0"
    assert sink["t"] == source["t"] + 1
    assert abs(sink["i"] - source["i"]) == 1
    assert params["nt"] > 0

    # The refuted fact is on the exception, since no schedule was built.
    assert caught.value.fact.status.value == "refuted"
    assert caught.value.fact.kind == "monotone"


def test_skewing_first_makes_the_tiling_legal_and_it_still_computes() -> None:
    schedule = Schedule(ht.jacobi_term(), sizes={"nt": 6, "nx": 6})
    skewed = schedule.skew("i", by="t")
    tiled = skewed.tile("t", "i", 8, 8)

    assert tiled.history == ("skew(i, by='t')", "tile(t,i,8,8)")
    assert tiled.order == ("t_outer", "i_outer", "t_inner", "i_inner")
    assert [fact.status.value for fact in tiled.facts()] == ["decided"] * 4

    u = np.zeros((6, 6))
    u[0] = np.arange(6.0)
    out = run(tiled, u=u.copy())
    assert np.allclose(out["u"], ht.jacobi_reference(u))


def test_a_rejected_step_leaves_the_schedule_it_came_from_intact() -> None:
    schedule = Schedule(ht.jacobi_term(), sizes={"nt": 16, "nx": 16})
    with pytest.raises(IllegalCast):
        schedule.tile("t", "i", 8, 8)
    assert schedule.history == ()
    assert schedule.order == ("t", "i")


def test_running_a_loop_that_carries_a_dependence_in_parallel_is_rejected() -> None:
    schedule = Schedule(ht.jacobi_term(), sizes={"nt": 16, "nx": 16})
    with pytest.raises(IllegalCast) as caught:
        schedule.tag(t="g.0")
    assert "illegal" in str(caught.value)
    assert caught.value.fact.kind == "monotone"


def test_a_parallel_tag_is_what_makes_an_iname_unordered() -> None:
    assert parallel_tag("g.0")
    assert parallel_tag("l.1")
    assert not parallel_tag("unr")


def gated_by_the_next(
    flag: Arr[Fin[n + 1], Nat],  # noqa: F821
    y: Arr[Fin[n], Real],  # noqa: F821
):
    """``S0`` writes ``flag[i]``; ``S1`` is guarded on ``flag[i + 1]``.

    The only reference to ``flag`` in ``S1`` is in its guard, so the whole
    dependence between the two statements rests on the guard being collected as
    a read.
    """
    for i in y.dom:
        flag[i] = 1
        with when(flag[i + 1] != 0):
            y[i] = 2.0


def test_a_parallel_tag_that_would_reorder_a_guard_read_is_rejected() -> None:
    # ``S1[i]`` reads the cell ``S0[i + 1]`` overwrites, which the sequential
    # loop runs in that order and a parallel loop does not. The access lives
    # only in ``stmt.guard``, so before the collectors were unified the schedule
    # checker saw no dependence here at all and accepted the tag.
    from lanky.terms import evaluate_annotations

    from loopty.trace import trace

    term = trace(gated_by_the_next, evaluate_annotations(gated_by_the_next))
    schedule = Schedule(term)
    with pytest.raises(IllegalCast) as caught:
        schedule.tag(i="g.0")
    message = str(caught.value)
    assert "reads flag[" in message
    assert caught.value.fact.kind == "monotone"
    assert caught.value.fact.status.value == "refuted"


def row_sums_then_next_offset(
    ends: Arr[Fin[n], Nat],  # noqa: F821
    cnt: Arr[Fin[n], Nat],  # noqa: F821
    off: Arr[Fin[n + 1], Nat],  # noqa: F821
    val: Arr[Fin[n], Fin[cnt], Real],  # noqa: F821
    y: Arr[Fin[n], Real],  # noqa: F821
):
    """``S0`` sums row ``r`` through ``off[r]``; ``S1`` then stores ``off[r + 1]``.

    ``S1[r]`` writes the offset ``S0[r + 1]`` indexes through, and the loop runs
    them in that order. ``S0`` never names ``off``: its only reference to it is
    the flat index lowering gives ``val[r, j]``, so the whole dependence rests on
    the collector listing the layout's reads.
    """
    for r in y.dom:
        y[r] = reduce_sum(val[r, j] for j in val.dom[r])
        off[r + 1] = ends[r]


def test_a_parallel_tag_that_would_read_through_stale_offsets_is_rejected() -> None:
    # Run in parallel, row ``r + 1`` is summed before ``off[r + 1]``, where it
    # starts, is stored. The schedule checker used to see no dependence between
    # the two statements and accepted the tag.
    from lanky.terms import evaluate_annotations

    from loopty.trace import trace

    term = trace(
        row_sums_then_next_offset, evaluate_annotations(row_sums_then_next_offset)
    )
    schedule = Schedule(term, sizes={"n": 4})
    with pytest.raises(IllegalCast) as caught:
        schedule.tag(r="l.0")
    message = str(caught.value)
    assert message.startswith("tag(r='l.0') illegal: instance S1[r=")
    assert "writes off[" in message
    assert "read by S0[" in message
    (source_id, source), (sink_id, sink), params = caught.value.witness
    assert (source_id, sink_id) == ("S1", "S0")
    assert sink["r"] == source["r"] + 1
    assert params["n"] == 4
    assert caught.value.fact.kind == "monotone"
    assert caught.value.fact.status.value == "refuted"


def scan_then_row_sums(
    cnt: Arr[Fin[n], Nat],  # noqa: F821
    off: Arr[Fin[n + 1], Nat],  # noqa: F821
    val: Arr[Fin[n], Fin[cnt], Real],  # noqa: F821
    y: Arr[Fin[n], Real],  # noqa: F821
):
    """The scan of the counts, then row sums through the offsets it stored."""
    off[0] = 0
    for r in cnt.dom:
        off[r + 1] = off[r] + cnt[r]
    for i in y.dom:
        y[i] = reduce_sum(val[i, j] for j in val.dom[i])


def test_the_rows_read_after_the_scan_can_still_run_in_parallel() -> None:
    # The offsets reads add dependences from the scan to the rows, and every
    # one of them runs from the first loop to the second: none is carried by
    # the row loop, so tagging it stays legal. The scan's own loop carries its
    # recurrence and is refused, as it always was.
    from lanky.terms import evaluate_annotations

    from loopty.trace import trace

    term = trace(scan_then_row_sums, evaluate_annotations(scan_then_row_sums))
    schedule = Schedule(term, sizes={"n": 6})
    schedule.tag(i="l.0")
    with pytest.raises(IllegalCast, match=r"writes off\[.*read by S1\["):
        schedule.tag(r="l.0")


# }}}


# {{{ reindexings that are not bijections


def test_a_reindexing_that_collapses_instances_is_rejected() -> None:
    # loopty ships no transformation that collapses an index space, which is
    # exactly why the checker has to be asked: any Python function may produce a
    # draft, and the bijection is decided about its output, not its code.
    schedule = Schedule(ht.jacobi_term())
    draft = schedule._draft()
    draft.constraints["S0"] = ["y1 = 0"]
    draft.overridden["S0"] = {"i"}
    with pytest.raises(IllegalCast) as caught:
        schedule._commit(draft, "flatten(i)")
    assert "not a bijection" in str(caught.value)
    assert caught.value.fact.kind == "bijective"
    assert caught.value.fact.status.value == "refuted"


# }}}


# {{{ splitting, tagging and reassociating


def test_split_then_tag_is_accepted_when_nothing_depends_on_the_order() -> None:
    schedule = Schedule(ht.transpose_term())
    split = schedule.split("i", 4, inner="i_in", outer="i_out")
    assert split.order == ("i_out", "i_in", "j")
    tagged = split.tag(i_out="g.0")
    assert tagged.tags == {"i_out": "g.0"}
    assert [fact.status.value for fact in tagged.facts()] == ["decided"] * 4

    a = np.arange(8, dtype=np.float64).reshape(2, 4)
    b = np.zeros((4, 2))
    out = run(split, a=a, b=b)
    assert np.array_equal(out["b"], a.T)


def test_interchange_is_checked_even_though_it_renames_nothing() -> None:
    schedule = Schedule(ht.transpose_term()).interchange("j", "i")
    assert schedule.order == ("j", "i")
    kinds = [fact.kind for fact in schedule.facts()]
    assert kinds == ["bijective", "monotone"]


def test_realizing_a_reduction_as_a_tree_marks_it_reassociated() -> None:
    schedule = Schedule(ht.spmv_term(exactness="reassoc")).realize("y", tree=True)
    assert schedule.reassociated == frozenset({"y"})
    exactness = [fact for fact in schedule.facts() if fact.kind == "exactness"]
    assert len(exactness) == 1
    assert exactness[0].status.value == "decided"


def test_a_tree_over_an_exact_accumulation_is_rejected() -> None:
    schedule = Schedule(ht.spmv_term(exactness="exact"))
    with pytest.raises(IllegalCast) as caught:
        schedule.realize("y", tree=True)
    assert "exact" in str(caught.value)
    assert caught.value.fact.kind == "exactness"


def test_a_reduction_iname_splits_without_touching_the_instances() -> None:
    schedule = Schedule(ht.spmv_term(exactness="reassoc"))
    split = schedule.split("j", 2, inner="j_in", outer="j_out")
    # The statement instances are the rows, before and after.
    assert split.order == ("r",)
    tagged = split.tag(j_in="l.0")
    assert tagged.reassociated == frozenset({"y"})


def test_tagging_a_piece_of_an_exact_reduction_in_parallel_is_rejected() -> None:
    schedule = Schedule(ht.spmv_term(exactness="exact")).split(
        "j", 2, inner="j_in", outer="j_out"
    )
    with pytest.raises(IllegalCast) as caught:
        schedule.tag(j_in="l.0")
    assert "reassociates an exact reduction" in str(caught.value)


def test_exactness_is_read_off_the_reduction_being_transformed() -> None:
    # Two reductions write ``y``: one exact, one approx. Asking the array what
    # its exactness is has two answers, and the one that matters is the one
    # belonging to the iname being tagged.
    schedule = Schedule(ht.two_reductions_term())

    with pytest.raises(IllegalCast) as caught:
        schedule.tag(j="l.0")
    assert "reassociates an exact reduction" in str(caught.value)
    assert "into y" in str(caught.value)

    # ``k`` belongs to the approx reduction, so tagging it is a reassociation
    # the type permits. It used to be refused, because the first reduction
    # found for ``y`` was the exact one.
    tagged = schedule.tag(k="l.0")
    assert tagged.reassociated == frozenset({"y"})


def test_realize_answers_for_every_reduction_writing_the_array() -> None:
    # ``realize`` is a claim about the whole accumulation into ``y``, and one
    # exact reduction among them forbids a tree.
    schedule = Schedule(ht.two_reductions_term())
    with pytest.raises(IllegalCast) as caught:
        schedule.realize("y", tree=True)
    assert "is exact" in caught.value.fact.provenance["detail"]


# }}}


def test_the_facts_name_the_term_the_oracle_and_the_dependence_source() -> None:
    schedule = Schedule(ht.transpose_term()).split("i", 4)
    facts = schedule.facts()
    assert [fact.id for fact in facts] == [
        "cast:transpose:0:bijective",
        "cast:transpose:0:monotone",
    ]
    for fact in facts:
        assert fact.decided_by == "isl"
        assert fact.owner == "transpose"
        assert "dependences" in fact.provenance


def test_a_schedule_carries_its_example_inputs() -> None:
    arrays = ht.csr_example()
    schedule = Schedule(ht.spmv_term()).example(**arrays)
    assert set(schedule.examples) == set(arrays)
    # example() returns a new schedule, like every other method.
    assert Schedule(ht.spmv_term()).examples is None


def test_unknown_inames_are_refused_before_loopy_sees_them() -> None:
    schedule = Schedule(ht.transpose_term())
    with pytest.raises(ValueError, match="not an iname"):
        schedule.split("k", 4)
    with pytest.raises(ValueError, match="not inames"):
        schedule.interchange("i", "k")


def test_the_dependences_are_reconciled_with_the_flow_module() -> None:
    # Two independent computations of the same relation: this module's, which
    # keeps each dependence separately so that a rejection can name the array
    # cell, and ``loopty.flow``'s, which is the definition. A schedule is checked
    # against the union, and every cast fact carries the comparison.
    schedule = Schedule(ht.jacobi_term()).skew("i", by="t")
    notes = {fact.provenance["dependences"] for fact in schedule.facts()}
    assert notes == {"agree with loopty.flow"}


def test_a_term_with_no_dependences_says_so_on_both_sides() -> None:
    schedule = Schedule(ht.transpose_term()).split("i", 4)
    note = schedule.facts()[0].provenance["dependences"]
    assert "none either" in note


# {{{ retargeting


def test_retargeting_to_the_same_target_is_the_same_schedule() -> None:
    schedule = Schedule(ht.transpose_term()).split("i", 4)
    assert schedule.retarget("c") is schedule


def test_retargeting_replays_every_step_and_checks_it_again() -> None:
    # The casts are about meaning, so they answer the same way; what changes is
    # the lowering they are checked against, and the buildability question.
    schedule = (
        Schedule(ht.transpose_term(), sizes={"n": 4, "m": 4})
        .split("i", 4, inner="i_in", outer="i_out")
        .interchange("j", "i_out")
    )
    other = schedule.retarget("c-source")

    assert other is not schedule
    assert other.target == "c-source"
    assert other.history == schedule.history
    assert other.order == schedule.order
    assert other.sizes == schedule.sizes
    assert [fact.kind for fact in other.facts()] == [
        fact.kind for fact in schedule.facts()
    ]
    assert all(fact.status.value == "decided" for fact in other.facts())
    assert all(fact.provenance["target"] == "c-source" for fact in other.facts())


def test_retargeting_carries_the_example_inputs() -> None:
    arrays = ht.csr_example()
    schedule = Schedule(ht.spmv_term()).example(**arrays)
    assert set(schedule.retarget("c-source").examples) == set(arrays)


# }}}


# {{{ what the target can build


def test_a_hardware_axis_inside_a_ragged_fiber_is_reported_not_thrown() -> None:
    from loopty.executor import LoopyExecutor
    from loopty.schedule import UnbuildableSchedule

    schedule = (
        Schedule(ht.spmv_term())
        .split("j", 32, inner="j_in", outer="j_out")
        .tag(j_in="l.0")
    )
    ok, reason = schedule.buildable
    assert not ok
    assert "ragged fiber" in reason and "hardware axis" in reason

    (fact,) = [f for f in schedule.facts() if f.kind == "buildable"]
    assert fact.status.value == "refuted"
    assert fact.decided_by == "loopy-target"
    assert fact.provenance["detail"] == reason

    with pytest.raises(UnbuildableSchedule, match="ragged fiber"):
        LoopyExecutor().run(schedule, **ht.csr_example())


def test_a_reduction_split_across_parallel_and_sequential_inames_is_reported():
    # A dense reduction, so the ragged rule cannot be what refuses it: what is
    # wrong is that half the reduction is parallel and half is not, which loopy
    # refuses in code generation.
    term = ht.spmv_term()
    dense_domain = isl.Set("[n, k] -> { [r, j] : 0 <= r < n and 0 <= j < k }")
    reduction = dataclasses.replace(
        reductions_of(term.stmts[0].expr)[0], domain=dense_domain
    )
    stmt = dataclasses.replace(term.stmts[0], expr=reduction)
    # ``k`` is a size of the term, so the fiber is *not* data dependent and the
    # ragged rule cannot fire.
    dense = dataclasses.replace(
        term, name="spmv_dense", stmts=(stmt,), sizes=("n", "m", "k")
    )

    schedule = (
        Schedule(dense)
        .split("j", 8, inner="j_in", outer="j_out")
        .tag(j_in="l.0")
    )
    ok, reason = schedule.buildable
    assert not ok
    assert "in parallel" in reason and "in sequence" in reason


def test_a_schedule_the_target_can_build_says_so_and_carries_no_extra_fact():
    schedule = Schedule(ht.spmv_term()).tag(r="g.0")
    assert schedule.buildable == (True, "")
    assert not [f for f in schedule.facts() if f.kind == "buildable"]
    schedule.require_buildable()


# }}}


# {{{ what a refuted cast fact says


def refused(step) -> IllegalCast:
    """The refusal ``step()`` raises."""
    with pytest.raises(IllegalCast) as caught:
        step()
    return caught.value


def collapsed() -> IllegalCast:
    """A draft that sends every ``i`` to 0, which no shipped cast does."""
    schedule = Schedule(ht.jacobi_term())
    draft = schedule._draft()
    draft.constraints["S0"] = ["y1 = 0"]
    draft.overridden["S0"] = {"i"}
    return refused(lambda: schedule._commit(draft, "flatten(i)"))


def test_a_refused_cast_carries_its_message_as_the_reason() -> None:
    # lanky prints a refuted fact's ``reason`` under its REFUTED line, and a
    # cast fact used to carry its explanation as ``detail`` alone, which only
    # the JSON ledger shows. The reason is the message of the IllegalCast, so
    # the fact and the exception say the same thing.
    stencil = Schedule(ht.jacobi_term(), sizes={"nt": 16, "nx": 16})
    exact = Schedule(ht.spmv_term(exactness="exact"))
    refusals = {
        "monotone": refused(lambda: stencil.tile("t", "i", 8, 8)),
        "bijective": collapsed(),
        "realize": refused(lambda: exact.realize("y", tree=True)),
        "tag": refused(
            lambda: exact.split("j", 2, inner="j_in", outer="j_out").tag(
                j_in="l.0"
            )
        ),
    }
    for exc in refusals.values():
        assert exc.fact.status.value == "refuted"
        assert exc.fact.provenance["reason"] == str(exc)
        # ``detail`` stays, in the oracle's own words.
        assert exc.fact.provenance["detail"]
    assert refusals["monotone"].fact.provenance["reason"].endswith(
        "scheduled earlier (at nt=16, nx=16, as hinted)"
    )


def test_a_refused_cast_carries_a_witness_exactly_when_isl_gave_one() -> None:
    stencil = Schedule(ht.jacobi_term(), sizes={"nt": 16, "nx": 16})
    backwards = refused(lambda: stencil.tile("t", "i", 8, 8))
    assert backwards.fact.provenance["witness"] == backwards.witness
    (source_id, _), (sink_id, _), params = backwards.fact.provenance["witness"]
    assert source_id == sink_id == "S0"
    assert params == {"nt": 16, "nx": 16}

    flattened = collapsed()
    first, second = flattened.fact.provenance["witness"]
    assert first != second
    assert flattened.fact.provenance["witness"] == flattened.witness

    # Exactness is not a question for isl, so there is no point to show.
    exact = Schedule(ht.spmv_term(exactness="exact"))
    tree = refused(lambda: exact.realize("y", tree=True))
    assert tree.witness is None
    assert "witness" not in tree.fact.provenance


def test_a_schedule_the_target_cannot_build_gives_the_limit_as_the_reason() -> None:
    schedule = (
        Schedule(ht.spmv_term())
        .split("j", 32, inner="j_in", outer="j_out")
        .tag(j_in="l.0")
    )
    (fact,) = [f for f in schedule.facts() if f.kind == "buildable"]
    assert fact.provenance["reason"] == schedule.buildable[1]
    assert "witness" not in fact.provenance
    # A decided fact is explained by nothing, because nothing needs explaining.
    assert all(
        "reason" not in f.provenance for f in schedule.facts() if f is not fact
    )


def test_lanky_prints_the_reason_of_every_refuted_cast_fact() -> None:
    # What ``lanky check`` and ``loopty run`` print under a REFUTED line. The
    # exactness and buildable facts used to come out as ``no witness
    # recorded``, and the others with nothing under them at all.
    from lanky.cli import refutation_lines

    stencil = Schedule(ht.jacobi_term(), sizes={"nt": 16, "nx": 16})
    exact = Schedule(ht.spmv_term(exactness="exact"))
    unbuildable = (
        Schedule(ht.spmv_term())
        .split("j", 32, inner="j_in", outer="j_out")
        .tag(j_in="l.0")
    )
    facts = [
        refused(lambda: stencil.tile("t", "i", 8, 8)).fact,
        collapsed().fact,
        refused(lambda: exact.realize("y", tree=True)).fact,
        *[f for f in unbuildable.facts() if f.kind == "buildable"],
    ]
    assert len(facts) == 4
    for fact in facts:
        lines = refutation_lines(fact)
        assert fact.provenance["reason"] in lines
        assert "no witness recorded" not in lines


# }}}
