"""A math step under a condition on a table, and sums of tables computed once.

The inverse mass cancels the mass only where the two tables are each other's
inverse, which is a property of the tables a run is given and not of every
pair of tables. A theorem says so with a hypothesis about table cells; at the
kernel that is a condition on tables, tested at the tables' digits, and the
rewritten statement reads ``M^-1 S^T``, a sum of tables, which ``keep=``
computes once. These tests take a two-axis form of ``examples/advection.py``'s
volume term.
"""

from __future__ import annotations

from fractions import Fraction

import numpy as np
import pytest
from lanky import theorem
from lanky.prelude import Fn, Nat, Real

from loopty import Arr, Fin, Schedule, kernel, reduce_sum, table

pytest.importorskip("loopy")


@theorem
def inverse_mass_inward(
    n: Nat,
    Mi: Fn[Fin[n], Fin[n], Real],  # noqa: F821
    M: Fn[Fin[n], Fin[n], Real],  # noqa: F821
    S: Fn[Fin[n], Fin[n], Real],  # noqa: F821
    w: Fn[Fin[n], Fin[n], Real],  # noqa: F821
    k: Fin[n],  # noqa: F821
    j: Fin[n],  # noqa: F821
    inverse: all(  # noqa: F821
        sum(Mi[p, c] * M[c, p] for c in Fin[n]) == 1 for p in Fin[n]  # noqa: F821
    )
    & all(
        sum(Mi[p, c] * M[c, q] for c in Fin[n]) == 0  # noqa: F821
        for p in Fin[n]  # noqa: F821
        for q in Fin[n]  # noqa: F821
        if p != q  # noqa: F821
    ),
) -> sum(  # noqa: F821
    Mi[k, c]  # noqa: F821
    * Mi[j, b]  # noqa: F821
    * sum(M[c, t] * S[b, s] * w[t, s] for t in Fin[n] for s in Fin[n])  # noqa: F821
    for c in Fin[n]  # noqa: F821
    for b in Fin[n]  # noqa: F821
) == sum(  # noqa: F821
    sum(Mi[j, b] * S[b, s] for b in Fin[n]) * w[k, s] for s in Fin[n]  # noqa: F821
):
    """The inverse mass after the weak derivative along the second axis, in two axes."""


@kernel
def weak(
    Mi: Arr[Fin[n], Fin[n], Real],  # noqa: F821
    M: Arr[Fin[n], Fin[n], Real],  # noqa: F821
    S: Arr[Fin[n], Fin[n], Real],  # noqa: F821
    u: Arr[Fin[ne], Fin[n], Fin[n], Real],  # noqa: F821
    out: Arr[Fin[ne], Fin[n], Fin[n], Real],  # noqa: F821
):
    """The inverse mass of the weak derivative, as the math says it, per element."""
    for e, k, j in out.dom:
        out[e, k, j] = reduce_sum(
            Mi[k, c]
            * Mi[j, b]
            * reduce_sum(
                M[c, t] * S[b, s] * u[e, t, s] for t in M.dom[c] for s in S.dom[b]
            )
            for c in Mi.dom[k]
            for b in Mi.dom[j]
        )


def mass(n: int = 4) -> np.ndarray:
    """A symmetric positive definite table, not diagonal: an exact mass's shape."""
    rng = np.random.default_rng(3)
    a = rng.normal(size=(n, n))
    return a @ a.T + n * np.eye(n)


def weak_inputs(n: int = 4, ne: int = 3, off: float = 0.0) -> dict:
    rng = np.random.default_rng(4)
    M = mass(n)
    Mi = np.linalg.inv(M)
    Mi[0, 1] += off
    return {
        "Mi": Arr.from_numpy(Mi),
        "M": Arr.from_numpy(M),
        "S": Arr.from_numpy(rng.normal(size=(n, n))),
        "u": Arr.from_numpy(rng.normal(size=(ne, n, n))),
        "out": Arr.zeros((ne, n, n)),
    }


def rows(schedule: Schedule) -> dict[str, list]:
    out: dict[str, list] = {}
    for fact in schedule.facts():
        out.setdefault(fact.kind, []).append(fact)
    return out


def conditions(schedule: Schedule, hypothesis: str) -> list:
    """The condition rows of one of the theorems' hypotheses."""
    return [
        fact
        for fact in rows(schedule)["rewrite-condition"]
        if fact.provenance["hypothesis"] == hypothesis
    ]


# {{{ a condition on tables


def test_a_condition_on_whole_tables_is_on_tables() -> None:
    """``M^-1 M == I`` reads ``Mi`` and ``M`` at no loop: a property of the tables."""
    derived = Schedule(weak).rewrite(weak.out, by=inverse_mass_inward)
    (rewritten,) = derived.rewrites
    kinds = {condition.hypothesis: condition for condition in rewritten.conditions}
    inverse = kinds["inverse"]
    assert inverse.kind == "table"
    assert inverse.over == ()


def test_the_tables_condition_holds_at_their_digits() -> None:
    """``np.linalg.inv`` is the inverse to the last digits, not exactly: tested."""
    derived = Schedule(weak).rewrite(weak.out, by=inverse_mass_inward)
    derived = derived.example(**weak_inputs())
    found = rows(derived)
    (inverse,) = conditions(derived, "inverse")
    assert inverse.status.value == "tested", inverse.provenance
    assert inverse.statement.endswith(
        "which inverse_mass_inward needs, at the tables' digits, on the example inputs"
    )
    assert "at the tables' 15 digits" in inverse.provenance["detail"]
    # Lean is told the condition as a hypothesis of the part
    (row,) = found["rewrite"]
    assert inverse.id in row.rests_on


def test_a_table_off_beyond_its_digits_is_refuted_naming_them() -> None:
    derived = (
        Schedule(weak)
        .rewrite(weak.out, by=inverse_mass_inward)
        .example(**weak_inputs(off=1e-9))
    )
    (inverse,) = conditions(derived, "inverse")
    assert inverse.status.value == "refuted"
    assert "fails at the tables' 15 digits" in inverse.provenance["reason"]
    assert "beyond" in inverse.provenance["reason"]


def test_draws_keep_the_tables_at_hand() -> None:
    """Tables drawn at random are never inverses; the draws keep the example's."""
    derived = Schedule(weak).rewrite(weak.out, by=inverse_mass_inward)
    found = rows(derived.example(**weak_inputs()))
    (row,) = found["rewrite"]
    assert row.status.value in ("tested", "proved"), row.provenance
    assert "drawn inputs agree" in row.provenance["detail"]
    (compared,) = found["rewrite-compared"]
    assert compared.status.value == "tested", compared.provenance
    assert "drawn ones" in compared.statement


def test_sample_arguments_takes_fixed_arrays_and_their_sizes() -> None:
    from loopty.faithful import sample_arguments

    fixed = {name: weak_inputs()[name] for name in ("Mi", "M")}
    drawn, sizes = sample_arguments(weak.term, seed=0, fixed=fixed)
    assert sizes["n"] == 4
    assert np.array_equal(drawn["Mi"].numpy(), fixed["Mi"].numpy())
    assert drawn["Mi"] is not fixed["Mi"]
    assert drawn["S"].numpy().shape == (4, 4)


#: A rule's weight as a test writes it, to 15 digits.
W3 = table([0.166666666666667, 0.166666666666667, 0.166666666666667])

#: The same, to 4 digits.
W4 = table([0.1667, 0.1667, 0.1667])


@theorem
def weighted(
    n: Nat,
    W: Fn[Fin[n], Real],  # noqa: F821
    f: Fn[Fin[n], Real],  # noqa: F821
    sixth: all(W[q] == Fraction(1, 6) for q in Fin[n]),  # noqa: F821
) -> sum(W[q] * f[q] for q in Fin[n]) == sum(f[q] for q in Fin[n]) / 6:  # noqa: F821
    """A rule of weights 1/6 is a sixth of the sum."""


@kernel
def rule3(x: Arr[Fin[nc], Fin[3], Real], y: Arr[Fin[nc], Real]):  # noqa: F821
    for c in y.dom:
        y[c] = reduce_sum(W3[q] * x[c, q] for q in W3.dom)


@kernel
def rule4(x: Arr[Fin[nc], Fin[3], Real], y: Arr[Fin[nc], Real]):  # noqa: F821
    for c in y.dom:
        y[c] = reduce_sum(W4[q] * x[c, q] for q in W4.dom)


def test_a_constant_table_is_the_rule_to_its_fifteen_digits() -> None:
    rng = np.random.default_rng(5)
    example = {"x": Arr.from_numpy(rng.normal(size=(4, 3))), "y": Arr.zeros(4)}
    exact = Schedule(rule3).rewrite(rule3.y, by=weighted).example(**example)
    (sixth,) = conditions(exact, "sixth")
    assert sixth.status.value == "tested", sixth.provenance
    assert sixth.statement.endswith("which weighted needs, at the table's digits")
    rounded = Schedule(rule4).rewrite(rule4.y, by=weighted).example(**example)
    (sixth,) = conditions(rounded, "sixth")
    assert sixth.status.value == "refuted"


# }}}


# {{{ sums of tables, computed once


def derived() -> Schedule:
    return Schedule(weak).rewrite(weak.out, by=inverse_mass_inward, keep=("D",))


def test_keep_computes_a_sum_of_tables_once_outside_the_elements() -> None:
    from loopty.executor import LoopyExecutor

    schedule = derived()
    (rewritten,) = schedule.rewrites
    assert rewritten.tables
    (kept,) = rewritten.kept
    assert kept.name == "D"
    assert kept.spelled == ("j", "s")
    stored = schedule.term.stmt(kept.stmt)
    # over the table's two indices, and no element loop
    assert len(stored.inames) == 2
    assert stored.text == "D[j, s] = reduce_sum(...)"
    fact = LoopyExecutor().differential(weak, schedule, weak_inputs())
    assert fact.status.value == "tested", fact.provenance


def test_the_mass_the_rewrite_cancels_is_set_aside() -> None:
    """``M`` is read nowhere once it cancels: taken, and not passed to the code."""
    from loopty.executor import LoopyExecutor

    schedule = derived()
    assert schedule.term.set_aside == ("M",)
    names = [arg.name for arg in schedule.lowering.kernel.default_entrypoint.args]
    assert "M" not in names
    got = LoopyExecutor().run(schedule, **weak_inputs())
    want = weak_inputs()
    weak(**want)
    assert np.allclose(got["out"], want["out"].numpy(), rtol=1e-12, atol=1e-12)


def test_keep_of_tables_rows_each_theorem_and_reading_the_kept_table() -> None:
    found = rows(derived().example(**weak_inputs()))
    statements = [fact.statement for fact in found["rewrite"]]
    assert statements[0].startswith(
        "rewrite(out, by=inverse_mass_inward): over the reals, what out stores is "
        "unchanged wherever"
    )
    assert statements[1] == (
        "keep=(D): D is computed once, before out, over the indices it reads, and "
        "out reads it where its sum was (D in 1 place): what out stores is unchanged"
    )
    for fact in found["rewrite"]:
        assert fact.status.value in ("tested", "proved"), fact.provenance
    kept_row = found["rewrite"][1]
    assert not kept_row.term.lean_obligation().declined
    (defined,) = [
        fact for fact in found["definedness"] if fact.provenance.get("array") == "D"
    ]
    assert defined.status.value == "decided"


def test_a_kept_table_has_a_handle() -> None:
    schedule = derived()
    assert repr(schedule.D) == "weak.D"


# }}}
