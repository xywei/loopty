"""Footprints, the padded instance space, and dependences defined from them."""

from __future__ import annotations

import islpy as isl
from lanky.prelude import Int, Nat, Real
from lanky.terms import evaluate_annotations

from loopty import Arr, Fin, flow, when
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


def test_an_accumulation_still_records_reads_of_its_other_cells() -> None:
    term = term_of(rolling)
    assert term.stmts[0].kind == "accumulate"
    prints = flow.footprints(term)
    assert sorted(f.kind for f in prints) == ["acc", "read"]
    # Instance i reads the cell instance i + 1 overwrites: a real dependence.
    deps = flow.dependences(term)
    assert not deps.is_empty()
    assert deps.deltas().is_subset(isl.Set("{ [ds = 0, di = 1] }"))
