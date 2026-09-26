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


def test_tiling_two_loops_named_in_the_other_order_is_a_tiling_too() -> None:
    # The second loop named first puts its tiles outside. Both splits used to
    # be written against the positions the coordinates had before either of
    # them, and the second one read the first one's position: the tiling came
    # out "not single-valued" and was refused although it is a bijection.
    tiled = Schedule(ht.jacobi_term(), sizes={"nt": 8, "nx": 8}).skew(
        "i", by="t"
    ).tile("i", "t", 4, 4)
    assert tiled.order == ("i_outer", "t_outer", "i_inner", "t_inner")
    assert [fact.status.value for fact in tiled.facts()] == ["decided"] * 4
    for nt, nx in [(3, 3), (7, 9), (13, 4), (16, 16)]:
        u = np.zeros((nt, nx))
        u[0] = np.random.default_rng(nt * nx).standard_normal(nx)
        out = run(tiled, u=u.copy())
        assert np.array_equal(out["u"], ht.jacobi_reference(u)), (nt, nx)

    transposed = Schedule(ht.transpose_term()).tile("j", "i", 2, 2)
    assert transposed.order == ("j_outer", "i_outer", "j_inner", "i_inner")
    a = np.arange(15, dtype=np.float64).reshape(3, 5)
    out = run(transposed, a=a, b=np.zeros((5, 3)))
    assert np.array_equal(out["b"], a.T)


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
    # exactly why the checker has to be asked: ``affine`` takes any map, and
    # the bijection is decided about the map, not about who wrote it.
    schedule = Schedule(ht.jacobi_term())
    with pytest.raises(IllegalCast) as caught:
        schedule.affine("{ [t, i] -> [t2, i2] : t2 = t and i2 = 0 }")
    assert "not a bijection" in str(caught.value)
    assert "not injective" in str(caught.value)
    assert caught.value.fact.kind == "bijective"
    assert caught.value.fact.status.value == "refuted"


def test_a_reindexing_that_misses_instances_is_rejected() -> None:
    # One for one on the instances it reaches, and none at all for the rest:
    # the map would drop every step after the third from the program, which a
    # bijectivity check on the map's own domain cannot see.
    schedule = Schedule(ht.jacobi_term(), sizes={"nt": 8, "nx": 8})
    with pytest.raises(IllegalCast) as caught:
        schedule.affine("{ [t, i] -> [t2, i2] : t2 = t and i2 = i and t <= 2 }")
    assert "not total" in str(caught.value)
    fact = caught.value.fact
    assert (fact.kind, fact.status.value) == ("bijective", "refuted")
    # The witness is an instance of the statement the map does not reach.
    statement, t, _i = fact.provenance["witness"]
    assert statement == 0 and t > 2


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


def test_a_new_loop_cannot_take_the_name_of_another_one() -> None:
    # Splitting i with its inner half called j would make two loops of the
    # transpose one: the checker's coordinates used to get j twice, and loopy
    # then refused the split from inside.
    with pytest.raises(ValueError, match="share its name"):
        Schedule(ht.transpose_term()).split("i", 4, inner="j")
    with pytest.raises(ValueError, match="share its name"):
        Schedule(ht.transpose_term()).split("i", 4, outer="n")


def test_a_split_reduction_loop_cannot_take_another_name_either() -> None:
    # A reduction loop is split without a reindexing map, and used to reach
    # isl's "non-unique var name" from inside loopy's split_iname.
    with pytest.raises(ValueError, match="share its name"):
        Schedule(ht.spmv_term()).split("j", 4, inner="r")
    with pytest.raises(ValueError, match="share its name"):
        Schedule(ht.spmv_term()).split("j", 4, outer="n")
    with pytest.raises(ValueError, match="named twice"):
        Schedule(ht.spmv_term()).split("j", 4, inner="k", outer="k")
    with pytest.raises(ValueError, match="must be positive"):
        Schedule(ht.spmv_term()).split("j", 0)


def test_skewing_or_tiling_a_loop_by_itself_is_refused() -> None:
    schedule = Schedule(ht.jacobi_term())
    with pytest.raises(ValueError, match="two different loops"):
        schedule.skew("i", by="i")
    with pytest.raises(ValueError, match="two different loops"):
        schedule.tile("t", "t", 4, 4)


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


# {{{ affine maps

#: The diamond coordinates, ``a = t + i`` and ``b = t - i``. The map has
#: determinant -2, so its image is only the points of equal parity.
DIAMOND = "{ [t, i] -> [a, b] : a = t + i and b = t - i }"

#: Sizes that include a single interior point, an odd and an even extent, and
#: more time levels than points and the other way round.
STENCIL_SIZES = [(2, 3), (3, 3), (6, 6), (7, 9), (13, 4), (16, 16)]


def _jacobi_input(nt: int, nx: int) -> np.ndarray:
    u = np.zeros((nt, nx))
    u[0] = np.random.default_rng(100 * nt + nx).standard_normal(nx)
    return u


def test_the_diamond_is_accepted_and_the_stencil_still_computes() -> None:
    schedule = Schedule(ht.jacobi_term(), sizes={"nt": 16, "nx": 16}).affine(
        DIAMOND
    )
    assert schedule.history == ("affine({ [t, i] -> [a = t + i, b = t - i] })",)
    assert schedule.order == ("a", "b")
    assert [(f.kind, f.status.value) for f in schedule.facts()] == [
        ("bijective", "decided"),
        ("monotone", "decided"),
    ]
    assert schedule.buildable == (True, "")

    # The kernel loops over the image, which has holes: isl states it with the
    # parity as an existentially quantified constraint, and the old loops are
    # floor divisions of the new ones.
    (domain,) = schedule.kernel.default_entrypoint.domains
    assert domain.get_var_names(isl.dim_type.set) == ["a", "b"]
    assert "mod 2" in str(domain)

    # loopy generates correct code for it, bit for bit, at every size.
    for nt, nx in STENCIL_SIZES:
        u = _jacobi_input(nt, nx)
        out = run(schedule, u=u.copy())
        assert np.array_equal(out["u"], ht.jacobi_reference(u)), (nt, nx)


def test_loopy_itself_refuses_the_diamond() -> None:
    # Why the rewrite is loopty's: both of loopy's own affine transforms solve
    # for each old iname with a unit coefficient, and t = (a + b) / 2 has none.
    # A loopy that learns to do this makes this test fail, and the rewrite can
    # then be dropped for lp.map_domain.
    import warnings

    import loopy as lp
    from loopy.diagnostic import LoopyError

    kernel = Schedule(ht.jacobi_term()).kernel
    kernel = kernel.with_kernel(
        kernel.default_entrypoint.copy(loop_priority=frozenset())
    )
    with (
        pytest.raises(LoopyError, match="No suitable equation"),
        warnings.catch_warnings(),
    ):
        # map_domain asks the BasicMap whether it is bijective, which islpy
        # 2025 answers with a deprecation warning; loopty no longer calls it,
        # so the exemption lives here and not in the suite's filters.
        warnings.filterwarnings(
            "ignore",
            message="BasicMap.is_bijective with implicit conversion",
            category=DeprecationWarning,
        )
        lp.map_domain(kernel, isl.BasicMap(DIAMOND))
    with pytest.raises(RuntimeError, match="division with remainder"):
        lp.affine_map_inames(kernel, "t, i", "a, b", ["a = t + i", "b = t - i"])


def test_tiling_the_diamond_is_legal_for_the_stencil_and_computes() -> None:
    # The stencil's two dependences, (1, 1) and (1, -1), become (2, 0) and
    # (0, 2): both non-negative, so rectangles in (a, b), which are diamonds
    # in (t, i), are legal. loopy splits the image's loops as it would any.
    schedule = (
        Schedule(ht.jacobi_term(), sizes={"nt": 16, "nx": 16})
        .affine(DIAMOND)
        .tile("a", "b", 4, 4)
    )
    assert schedule.order == ("a_outer", "b_outer", "a_inner", "b_inner")
    assert [fact.status.value for fact in schedule.facts()] == ["decided"] * 4
    for nt, nx in [*STENCIL_SIZES, (17, 23), (32, 32)]:
        u = _jacobi_input(nt, nx)
        out = run(schedule, u=u.copy())
        assert np.array_equal(out["u"], ht.jacobi_reference(u)), (nt, nx)


def test_a_diamond_that_runs_a_dependence_backwards_is_refused() -> None:
    # With the space axis first, a = i + t and b = i - t: the dependence
    # S0[t, i] -> S0[t + 1, i - 1] keeps a and lowers b by two, so the new
    # order runs it the wrong way round, and the witness says so.
    schedule = Schedule(ht.jacobi_term(), sizes={"nt": 16, "nx": 16})
    with pytest.raises(IllegalCast) as caught:
        schedule.affine("{ [t, i] -> [a, b] : a = i + t and b = i - t }")
    (source_id, source), (sink_id, sink), _params = caught.value.witness
    assert source_id == sink_id == "S0"
    assert (sink["t"] - source["t"], sink["i"] - source["i"]) == (1, -1)
    assert caught.value.fact.kind == "monotone"
    assert "scheduled earlier" in str(caught.value)


def test_skew_is_the_affine_map_that_keeps_the_loop_names(monkeypatch) -> None:
    import loopy as lp

    # A skew goes through the same rewrite as any affine map, not through
    # lp.map_domain, which it used to call.
    def refuse(*args, **kwargs):  # pragma: no cover - reached only on regression
        raise AssertionError("skew called lp.map_domain")

    monkeypatch.setattr(lp, "map_domain", refuse)
    skewed = Schedule(ht.jacobi_term()).skew("i", by="t")

    # The same map written out, with the names put back on the outputs.
    mapping = isl.Map("{ [t, i] -> [t2, i2] : t2 = t and i2 = i + t }")
    mapping = mapping.set_dim_name(isl.dim_type.out, 0, "t")
    mapping = mapping.set_dim_name(isl.dim_type.out, 1, "i")
    affine = Schedule(ht.jacobi_term()).affine(mapping)

    assert skewed.order == affine.order == ("t", "i")
    assert [f.status.value for f in skewed.facts()] == [
        f.status.value for f in affine.facts()
    ]
    (mine,) = skewed.kernel.default_entrypoint.domains
    (theirs,) = affine.kernel.default_entrypoint.domains
    assert mine.is_equal(theirs)
    assert "t < i" in str(mine)


def test_a_skewed_loop_keeps_its_tag_in_the_kernel() -> None:
    # The skew keeps the loop's name, so it keeps the loop's tag: the schedule
    # says i is a local axis, and so does the kernel it builds. Through
    # lp.map_domain and back the tag was lost, and the kernel ran the loop
    # the schedule had checked as parallel one iteration at a time.
    schedule = Schedule(ht.jacobi_term()).tag(i="l.0").skew("i", by="t")
    assert schedule.tags == {"i": "l.0"}
    (tag,) = schedule.kernel.default_entrypoint.inames["i"].tags
    assert type(tag).__name__ == "LocalInameTag"


def test_a_permutation_that_keeps_the_names_is_an_interchange() -> None:
    mapping = isl.Map("{ [i, j] -> [j2, i2] : j2 = j and i2 = i }")
    mapping = mapping.set_dim_name(isl.dim_type.out, 0, "j")
    mapping = mapping.set_dim_name(isl.dim_type.out, 1, "i")
    schedule = Schedule(ht.transpose_term()).affine(mapping)
    assert schedule.order == Schedule(ht.transpose_term()).interchange("j", "i").order
    assert [fact.status.value for fact in schedule.facts()] == ["decided"] * 2

    a = np.arange(8, dtype=np.float64).reshape(2, 4)
    out = run(schedule, a=a, b=np.zeros((4, 2)))
    assert np.array_equal(out["b"], a.T)


def test_a_map_over_a_ragged_row_loop_moves_the_fiber_with_it() -> None:
    # Reversing the rows of a CSR product. The fiber's domain names the row as
    # a parameter, which is how loopy nests it, so it is rewritten too, with
    # the new row loop in its place.
    arrays = ht.csr_example()
    schedule = Schedule(ht.spmv_term(), sizes={"n": 4, "m": 5}).affine(
        "[n] -> { [r] -> [q] : q = n - 1 - r }"
    )
    assert [fact.status.value for fact in schedule.facts()] == ["decided"] * 2
    domains = [str(d) for d in schedule.kernel.default_entrypoint.domains]
    assert any("[j]" in text and "q" in text.split("->")[0] for text in domains)

    def fresh() -> dict:
        return {k: (v.copy() if hasattr(v, "copy") else v) for k, v in arrays.items()}

    reversed_rows = run(schedule, **fresh())
    in_order = run(Schedule(ht.spmv_term()), **fresh())
    assert np.array_equal(reversed_rows["y"], in_order["y"])


def ragged_row_sums(
    cnt: Arr[Fin[n], Nat],  # noqa: F821
    val: Arr[Fin[n], Fin[cnt], Real],  # noqa: F821
    y: Arr[Fin[n], Real],  # noqa: F821
):
    """Row sums written as a loop over the fiber, not as a reduction."""
    for r in y.dom:
        for j in val.dom[r]:
            y[r] = y[r] + val[r, j]


def test_a_map_the_kernel_rewrite_cannot_write_is_reported_not_thrown() -> None:
    # The row and the fiber are two loopy domains, one nested in the other,
    # and a map that mixes them has no one domain to be the image of. The map
    # is legal, so the casts are decided; what is refuted is buildability, and
    # the schedule has no kernel from then on.
    from lanky.terms import evaluate_annotations

    from loopty.executor import LoopyExecutor
    from loopty.schedule import UnbuildableSchedule
    from loopty.trace import trace

    term = trace(ragged_row_sums, evaluate_annotations(ragged_row_sums))
    schedule = Schedule(term, sizes={"n": 4}).affine(
        "{ [r, j] -> [q, k] : q = r and k = j + r }"
    )
    assert [f.status.value for f in schedule.facts() if f.kind != "buildable"] == [
        "decided",
        "decided",
    ]
    ok, reason = schedule.buildable
    assert not ok
    assert "not all defined by one loopy domain" in reason
    (fact,) = [f for f in schedule.facts() if f.kind == "buildable"]
    assert (fact.status.value, fact.decided_by) == ("refuted", "loopy-target")
    assert schedule.kernel is None

    # Later steps are still checked, and still unbuildable.
    later = schedule.split("q", 2)
    assert later.kernel is None
    assert later.buildable == (False, reason)
    assert [f.status.value for f in later.facts()][-2:] == ["decided", "decided"]
    with pytest.raises(UnbuildableSchedule, match="one loopy domain"):
        LoopyExecutor().run(later, cnt=np.array([1, 2]), val=np.ones(3), y=np.zeros(2))


def test_only_the_loops_a_ragged_loop_becomes_keep_its_extent_from_data() -> None:
    # What a hardware axis may not sit inside is a loop whose extent is read
    # from an array. Tiling the ragged fiber with the dense row loop makes the
    # fiber's halves such loops and leaves the row's halves alone; skewing the
    # fiber by the row leaves the row alone.
    from lanky.terms import evaluate_annotations

    from loopty.trace import trace

    term = trace(ragged_row_sums, evaluate_annotations(ragged_row_sums))
    schedule = Schedule(term, sizes={"n": 4})
    assert schedule._data_dependent == {"j"}
    tiled = schedule.tile("r", "j", 2, 2)
    assert tiled._data_dependent == {"j_outer", "j_inner"}
    assert schedule.skew("j", by="r")._data_dependent == {"j"}
    diamond = schedule.affine("{ [r, j] -> [a, b] : a = r + j and b = r - j }")
    assert diamond._data_dependent == {"a", "b"}


def ragged_row_totals(
    cnt: Arr[Fin[n], Nat],  # noqa: F821
    val: Arr[Fin[n], Fin[cnt], Real],  # noqa: F821
    y: Arr[Fin[n], Real],  # noqa: F821
):
    """Row sums with the row cleared first: one statement outside the fiber."""
    for r in y.dom:
        y[r] = 0.0
        for j in val.dom[r]:
            y[r] = y[r] + val[r, j]


def _row_totals_run(schedule: Schedule) -> np.ndarray:
    from loopty.arr import Arr as RuntimeArr

    counts = [2, 0, 3, 1, 4]
    out = run(
        schedule,
        cnt=RuntimeArr.from_numpy(np.array(counts, dtype=np.int64)),
        val=RuntimeArr.ragged(counts, values=np.arange(1.0, 11.0)),
        y=np.full(5, 99.0),
    )
    return out["y"]


def test_a_statement_in_one_of_two_tiled_loops_is_split_by_its_own_loop() -> None:
    # S0 clears the row and runs in r only; S1 runs in r and in the fiber. A
    # tile is two splits side by side, so S0 takes the split of r and S1 the
    # whole tile, which is what loopy's split_iname does to the two
    # instructions.
    from lanky.terms import evaluate_annotations

    from loopty.trace import trace

    term = trace(ragged_row_totals, evaluate_annotations(ragged_row_totals))
    want = np.array([3.0, 0.0, 3.0 + 4.0 + 5.0, 6.0, 7.0 + 8.0 + 9.0 + 10.0])
    tiled = Schedule(term, sizes={"n": 5}).tile("r", "j", 2, 2)
    assert tiled.order == ("r_outer", "j_outer", "r_inner", "j_inner")
    assert [fact.status.value for fact in tiled.facts()] == ["decided"] * 2
    assert tiled._layout.coords["S0"] == ("r_outer", "r_inner")
    # The fiber's bound is read in the row loop, so loopy cannot run j_outer
    # outside r_inner and would pick a nest of its own (issue #41). Run the
    # tiles with both halves of the row outside, an order it can keep.
    rows_first = tiled.interchange("r_inner", "j_outer")
    assert rows_first.order == ("r_outer", "r_inner", "j_outer", "j_inner")
    assert np.array_equal(_row_totals_run(rows_first), want)

    # An affine map that is two maps side by side is taken apart the same way;
    # its kernel is out of reach, because the row and the fiber are two loopy
    # domains, and says so.
    apart = Schedule(term, sizes={"n": 5}).affine(
        "[n] -> { [r, j] -> [q, k] : q = n - 1 - r and k = j }"
    )
    assert apart._layout.coords["S0"] == ("q",)
    assert [f.status.value for f in apart.facts()] == ["decided", "decided", "refuted"]
    assert "one loopy domain" in apart.buildable[1]

    # A skew mixes the two loops: S0 has no j to shift, and the skew has no
    # part over r alone that means anything for it.
    with pytest.raises(ValueError, match="not in j, and the map mixes them"):
        Schedule(term, sizes={"n": 5}).skew("j", by="r")
    with pytest.raises(ValueError, match="S0 runs in r but not in j"):
        Schedule(term, sizes={"n": 5}).affine(
            "{ [r, j] -> [a, b] : a = r + j and b = r - j }"
        )


def test_a_new_loop_cannot_take_a_name_the_lowering_added() -> None:
    # The offsets of a ragged array are an argument the lowering adds, not one
    # of the term's names; a loop called that would be one name for two things.
    from lanky.terms import evaluate_annotations

    from loopty.trace import trace

    term = trace(ragged_row_totals, evaluate_annotations(ragged_row_totals))
    schedule = Schedule(term, sizes={"n": 5})
    (offsets,) = schedule.lowering.ragged.values()
    with pytest.raises(ValueError, match="share its name"):
        schedule.affine(f"{{ [r] -> [{offsets}] : {offsets} = r }}")
    with pytest.raises(ValueError, match="share its name"):
        schedule.split("r", 2, inner=offsets)


@pytest.mark.parametrize(
    ("mapping", "message"),
    [
        ("{ [t, k] -> [a, b] : a = t and b = k }", "not loops of jacobi: k"),
        ("{ [t] -> [i] : i = t }", "share its name"),
        ("{ [t] -> [nx] : nx = t }", "share its name"),
        ("{ [t] -> [u] : u = t }", "share its name"),
        ("{ [t] -> [int] : int = t }", "cannot name a loop"),
        ("[p] -> { [t] -> [a] : a = t + p }", "not sizes of jacobi"),
        ("{ S0[t, i] -> [a, b] : a = t and b = i }", "named tuple"),
        ("{ [t, i] -> [a, b] : a = t and b = i", "is not an isl map"),
    ],
)
def test_affine_refuses_a_map_it_cannot_give_a_meaning(
    mapping: str, message: str
) -> None:
    with pytest.raises(ValueError, match=message):
        Schedule(ht.jacobi_term()).affine(mapping)


def test_affine_refuses_reduction_loops_tagged_loops_and_unions() -> None:
    with pytest.raises(ValueError, match="j is the loop of a reduction"):
        Schedule(ht.spmv_term()).affine("{ [j] -> [k] : k = j }")
    doubled = isl.Map("{ [t, i] -> [a, b] : a = t and b = i }")
    doubled = doubled.set_dim_name(isl.dim_type.out, 1, "a")
    with pytest.raises(ValueError, match="a is named twice"):
        Schedule(ht.jacobi_term()).affine(doubled)
    with pytest.raises(ValueError, match="carries a tag"):
        Schedule(ht.transpose_term()).tag(i="g.0").affine("{ [i] -> [k] : k = i }")
    with pytest.raises(TypeError, match="one map for every statement"):
        Schedule(ht.jacobi_term()).affine(isl.UnionMap(DIAMOND))
    with pytest.raises(ValueError, match="has to be named"):
        Schedule(ht.jacobi_term()).affine("{ [t, i] -> [t + i, t - i] }")


def test_retargeting_replays_an_affine_step() -> None:
    schedule = Schedule(ht.jacobi_term(), sizes={"nt": 6, "nx": 6}).affine(DIAMOND)
    other = schedule.retarget("c-source")
    assert other.history == schedule.history
    assert other.order == ("a", "b")
    assert all(fact.provenance["target"] == "c-source" for fact in other.facts())


# }}}


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
    """The refusal of a map that sends every ``i`` to 0."""
    schedule = Schedule(ht.jacobi_term())
    return refused(
        lambda: schedule.affine("{ [t, i] -> [t2, i2] : t2 = t and i2 = 0 }")
    )


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
