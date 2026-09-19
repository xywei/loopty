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

import hand_terms as ht
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
