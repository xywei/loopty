"""The isl oracle's decision primitives, and the witnesses they return."""

from __future__ import annotations

import islpy as isl

from loopty.oracle import (
    TRUST_CLASS,
    IslOracle,
    is_bijective,
    is_empty,
    is_monotone,
    is_subset,
    sample_pair,
    sample_point,
)


def test_trust_class_is_decision_procedure() -> None:
    assert IslOracle().trust_class() == TRUST_CLASS == "decision-procedure"


def test_is_empty() -> None:
    assert is_empty(isl.Set("{ [i] : 0 <= i < 0 }"))
    assert not is_empty(isl.Set("{ [i] : 0 <= i < 3 }"))


def test_sample_point_returns_a_concrete_point() -> None:
    point = sample_point(isl.Set("{ [i, j] : i = 2 and j = 5 }"))
    assert point == (2, 5)
    assert sample_point(isl.Set("{ [i] : 1 = 0 }")) is None


def test_subset_holds_for_an_in_bounds_access() -> None:
    accessed = isl.Set("{ [a] : 0 <= a < 8 }")
    stored = isl.Set("{ [a] : 0 <= a < 10 }")
    verdict = is_subset(accessed, stored)
    assert verdict
    assert verdict.witness is None


def test_subset_fails_with_the_offending_address() -> None:
    accessed = isl.Set("{ [a] : 0 <= a < 12 }")
    stored = isl.Set("{ [a] : 0 <= a < 10 }")
    verdict = is_subset(accessed, stored)
    assert not verdict
    (address,) = verdict.witness
    assert address >= 10
    assert "not in the second" in verdict.detail


def test_a_reindexing_that_renames_instances_is_bijective() -> None:
    # Splitting i into (i // 4, i % 4) is the archetypal cast.
    split = isl.Map(
        "{ [i] -> [o, k] : o = floord(i, 4) and k = i - 4o and 0 <= i < 16 }"
    )
    verdict = is_bijective(split)
    assert verdict
    assert verdict.detail == "bijective"


def test_a_collapsing_map_is_refuted_with_two_colliding_instances() -> None:
    # Dropping an iname merges instances: the witness names two of them.
    collapse = isl.Map("{ [i, j] -> [i] : 0 <= i < 4 and 0 <= j < 4 }")
    verdict = is_bijective(collapse)
    assert not verdict
    source, target = verdict.witness
    assert source != target
    assert source[0] == target[0]
    assert "not injective" in verdict.detail


def test_a_one_to_many_map_is_refuted_as_not_single_valued() -> None:
    spread = isl.Map("{ [i] -> [i, j] : 0 <= i < 4 and 0 <= j < 2 }")
    verdict = is_bijective(spread)
    assert not verdict
    first, second = verdict.witness
    assert first != second
    assert "not single-valued" in verdict.detail


def test_sample_pair_splits_at_the_domain_dimension() -> None:
    pair = sample_pair(isl.Map("{ [i, j] -> [k] : i = 1 and j = 2 and k = 3 }"))
    assert pair == ((1, 2), (3,))
    assert sample_pair(isl.Map("{ [i] -> [j] : 1 = 0 }")) is None


# A 1-D Jacobi-shaped dependence: instance [t, i] feeds [t+1, i+1].
DEPS = isl.Map(
    "{ [t, i] -> [t', i'] : t' = t + 1 and i' = i + 1 "
    "and 0 <= t < 4 and 0 <= i < 4 and t' < 5 and i' < 5 }"
)


def test_the_sequential_order_is_monotone() -> None:
    identity = isl.Map("{ [t, i] -> [t, i] }")
    verdict = is_monotone(identity, DEPS)
    assert verdict
    assert verdict.witness is None


def test_interchanging_the_loops_is_still_monotone_here() -> None:
    # Both components of this dependence move forward, so swapping them is legal.
    swap = isl.Map("{ [t, i] -> [i, t] }")
    assert is_monotone(swap, DEPS)


def test_running_time_backwards_is_refuted_with_an_instance_pair() -> None:
    reversed_time = isl.Map("{ [t, i] -> [-t, i] }")
    verdict = is_monotone(reversed_time, DEPS)
    assert not verdict
    source, sink = verdict.witness
    # The witness is a real dependence of the program, in instance space.
    assert sink == (source[0] + 1, source[1] + 1)
    assert "not ordered forward" in verdict.detail


def test_an_empty_dependence_relation_is_vacuously_monotone() -> None:
    nothing = isl.Map("{ [t, i] -> [t', i'] : 1 = 0 }")
    verdict = is_monotone(isl.Map("{ [t, i] -> [t, i] }"), nothing)
    assert verdict
    assert "no dependences" in verdict.detail


def test_can_establish_recognizes_the_decidable_kinds() -> None:
    class Fact:
        def __init__(self, kind: str) -> None:
            self.kind = kind

    oracle = IslOracle()
    assert oracle.can_establish(Fact("subset"))
    assert oracle.can_establish(Fact("monotone"))
    assert not oracle.can_establish(Fact("postcondition"))
    assert not oracle.can_establish(object())


def test_establish_declines_a_fact_that_is_not_an_isl_question() -> None:
    assert IslOracle().establish(object()) is None


def test_the_primitives_are_reachable_from_the_oracle_object() -> None:
    oracle = IslOracle()
    assert oracle.is_empty(isl.Set("{ [i] : 1 = 0 }"))
    assert oracle.is_subset(
        isl.Set("{ [a] : 0 <= a < 2 }"), isl.Set("{ [a] : 0 <= a < 3 }")
    )


def test_the_oracle_answers_the_four_questions() -> None:
    from loopty.oracle import Bijective, Empty, Monotone, Subset, decide

    assert decide(Empty(isl.Set("{ [i] : 1 = 0 }")))
    assert not decide(Empty(isl.Set("{ [i] : i = 3 }")))
    small = isl.Set("{ [a] : 0 <= a < 2 }")
    assert decide(Subset(small, isl.Set("{ [a] : 0 <= a < 3 }")))
    assert decide(Bijective(isl.Map("{ [i] -> [i + 1] : 0 <= i < 8 }")))
    assert decide(
        Monotone(isl.Map("{ [i] -> [i] }"), isl.Map("{ [i] -> [i + 1] : 0 <= i < 8 }"))
    )


def test_a_refuted_fact_carries_a_labelled_witness() -> None:
    from lanky.ledger import Fact, Status

    from loopty.oracle import Empty

    fact = Fact(
        id="x",
        kind="disjoint-writes",
        statement="the set is empty",
        term=Empty(isl.Set("{ [s, d0] : s = 0 and d0 = 4 }"), labels=("s", "d0")),
    )
    decided = IslOracle().establish(fact)
    assert decided.status is Status.REFUTED
    assert decided.provenance["witness"] == (0, 4)
    assert decided.provenance["witness_text"] == "[s=0, d0=4]"


def test_a_refuted_fact_says_what_refutes_it_as_its_reason() -> None:
    # lanky prints a refuted fact's ``reason`` under its REFUTED line, and
    # prints no plugin's labelled witness. Without a reason, an out-of-bounds
    # access or two colliding writes came out as a bare REFUTED line.
    from lanky.cli import refutation_lines
    from lanky.ledger import Fact, Status

    from loopty.oracle import Empty, Monotone, Subset

    def refuted(term):
        fact = Fact(id="x", kind="k", statement="s", term=term)
        decided = IslOracle().establish(fact)
        assert decided.status is Status.REFUTED
        assert decided.provenance["reason"] in refutation_lines(decided)
        return decided.provenance["reason"]

    # The claim, and the witness that is the exception to it, with the sizes.
    escapes = Subset(
        isl.Set("[n] -> { [a] : a = n and n = 3 }"),
        isl.Set("[n] -> { [a] : 0 <= a < n }"),
        description="cells u[i + 1] reaches are cells u has",
        labels=("a0",),
    )
    assert refuted(escapes) == (
        "cells u[i + 1] reaches are cells u has, except [a0=3] at [n=3]"
    )
    # An emptiness question names the bad case, and the witness is one of it.
    collisions = Empty(
        isl.Map("{ [s, d0] -> [s, d1] : s = 0 and d0 = 1 and d1 = 0 }"),
        description="pairs of S0 instances writing the same cell",
        labels=("s", "d"),
    )
    assert refuted(collisions) == (
        "[s=0, d=1] -> [s=0, d=0] is one of the pairs of S0 instances writing "
        "the same cell"
    )
    backwards = Monotone(
        isl.Map("{ [i] -> [-i] }"),
        isl.Map("{ [i] -> [i + 1] : i = 0 }"),
        description="the source schedule is monotone on the dependences",
        labels=("i",),
    )
    assert refuted(backwards) == (
        "the source schedule is monotone on the dependences, except [i=0] -> [i=1]"
    )
    # A question with nothing to say what it asked falls back on the verdict.
    bare = Subset(isl.Set("{ [a] : a = 5 }"), isl.Set("{ [a] : a < 5 }"))
    assert refuted(bare) == "point (5,) is in the first set but not in the second"


def test_a_fact_isl_decides_has_no_reason() -> None:
    from lanky.ledger import Fact, Status

    from loopty.oracle import Empty

    empty = Empty(isl.Set("{ [i] : 1 = 0 }"))
    fact = Fact(id="x", kind="k", statement="s", term=empty)
    decided = IslOracle().establish(fact)
    assert decided.status is Status.DECIDED
    assert "reason" not in decided.provenance


def test_a_witness_and_its_sizes_come_from_one_sample(monkeypatch) -> None:
    # The point and the parameter valuation of a refutation used to be read off
    # two separate samples, which leaves isl free to answer with two different
    # points: the reported cell need not be outside the array at the reported
    # size. isl happens to answer the same way twice, so a stand-in that answers
    # differently every time it is asked is what makes the difference visible.
    import loopty.oracle as oracle
    from loopty.oracle import Empty, decide

    calls: list[int] = []

    def drifting(a_set):
        calls.append(1)
        k = len(calls)
        n_params = a_set.dim(isl.dim_type.param)
        names = [a_set.get_dim_name(isl.dim_type.param, p) for p in range(n_params)]
        return (k,) * a_set.dim(isl.dim_type.set), {name: k for name in names}

    monkeypatch.setattr(oracle, "_sample_full", drifting)

    def one_sample(verdict) -> bool:
        witness = verdict.witness
        if isinstance(witness[0], tuple):  # a pair of instances
            witness = (*witness[0], *witness[1])
        return set(witness) == set(verdict.parameters.values())

    small = isl.Set("[n] -> { [i] : 0 <= i <= n }")
    large = isl.Set("[n] -> { [i] : 0 <= i < n }")
    assert one_sample(is_subset(small, large))
    assert one_sample(is_bijective(isl.Map("[n] -> { [i] -> [0] : 0 <= i < n }")))
    assert one_sample(is_bijective(isl.Map("[n] -> { [0] -> [i] : 0 <= i < n }")))
    assert one_sample(
        is_monotone(
            isl.Map("[n] -> { [i] -> [-i] }"),
            isl.Map("[n] -> { [i] -> [i + 1] : 0 <= i < n }"),
        )
    )
    assert one_sample(decide(Empty(isl.Set("[n] -> { [i] : 0 <= i < n }"))))
