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
