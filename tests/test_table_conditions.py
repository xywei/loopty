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

    given = {name: weak_inputs()[name] for name in ("Mi", "M")}
    drawn, sizes = sample_arguments(weak.term, seed=0, given=given)
    assert sizes["n"] == 4
    assert np.array_equal(drawn["Mi"].numpy(), given["Mi"].numpy())
    assert drawn["Mi"] is not given["Mi"]
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


def test_an_array_set_aside_stays_aside_through_the_next_rewrite() -> None:
    """A later rewrite reads ``M`` no more than the one that set it aside (Codex, #348).

    The arrays set aside were worked out again from what the term before
    the step read, which no longer names ``M``, and the next lowering took
    ``M`` for an argument the code needs.
    """
    from loopty.rewriting import set_aside

    term = derived().term
    assert set_aside(term, term).set_aside == ("M",)


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


@kernel
def two_extents(
    A: Arr[Fin[n], Fin[m], Real],  # noqa: F821
    B: Arr[Fin[k], Fin[m], Real],  # noqa: F821
    out: Arr[Fin[k], Real],  # noqa: F821
):
    """A sum of two tables at one row index, over the rows ``B`` has."""
    for i in out.dom:
        out[i] = reduce_sum(A[i, q] * B[i, q] for q in B.dom[i])


@kernel
def shifted_rows(
    T: Arr[Fin[n + 1], Real],  # noqa: F821
    x: Arr[Fin[n], Real],  # noqa: F821
    y: Arr[Fin[n], Real],  # noqa: F821
):
    """A table one longer than the data, read one on."""
    for i in y.dom:
        y[i] = T[i + 1] * x[i]


def test_a_table_taken_as_it_is_fixes_the_size_its_axis_solves_for() -> None:
    """``T: Fin[n + 1]`` of 6 cells makes ``n`` 5 in a draw (Codex, #348).

    Only a bare ``Fin[n]`` axis fixed a size, so the other arrays were drawn
    at a size of their own, and the draws compared the rewrite on inputs of
    two shapes, or none.
    """
    from loopty.faithful import sample_arguments

    table = Arr.from_numpy(np.arange(6.0))
    arguments, sizes = sample_arguments(shifted_rows.term, given={"T": table})
    assert sizes["n"] == 5
    assert arguments["x"].numpy().shape == (5,)


def test_a_kept_sum_of_complex_tables_is_complex_in_a_real_output() -> None:
    """Stored at the output's type, its imaginary part was gone before ``real`` read it.

    Codex on #348: the kept table took the destination's element type, and
    then the first complex one, where it is the type the tables promote to.
    """
    from lanky.prelude import Complex
    from lanky.terms import Subscript, Var

    from loopty.rewriting import _kept_dtype
    from loopty.term import ArrType

    types = {
        "C": ArrType(axes=(3,), dtype=Complex, ragged=(False,)),
        "W": ArrType(axes=(3,), dtype=Real, ragged=(False,)),
        "Z": ArrType(axes=(3,), dtype=np.dtype(np.complex128), ragged=(False,)),
    }
    template = Subscript(Var("C"), Var("_b0")) * Subscript(Var("W"), Var("_b0"))
    assert _kept_dtype(template, types, Real) is Complex
    numpy = Subscript(Var("Z"), Var("_b0")) * Subscript(Var("W"), Var("_b0"))
    assert _kept_dtype(numpy, types, np.dtype(np.float64)) == np.complex128
    real_only = Subscript(Var("W"), Var("_b0")) * 2
    assert _kept_dtype(real_only, types, Real) is Real
    # promoted as numpy promotes, among real and complex types only
    types["S"] = ArrType(axes=(3,), dtype=np.dtype(np.float32), ragged=(False,))
    types["Y"] = ArrType(axes=(3,), dtype=np.dtype(np.complex64), ragged=(False,))
    types["N"] = ArrType(axes=(3,), dtype=np.dtype(np.int64), ragged=(False,))
    wide = Subscript(Var("W"), Var("_b0")) * Subscript(Var("S"), Var("_b0"))
    assert _kept_dtype(wide, types, np.dtype(np.float32)) is Real
    both = Subscript(Var("Y"), Var("_b0")) * Subscript(Var("Z"), Var("_b0"))
    assert _kept_dtype(both, types, np.dtype(np.float32)) == np.complex128
    index = Subscript(Var("N"), Var("_b0")) * Subscript(Var("S"), Var("_b0"))
    assert _kept_dtype(index, types, np.dtype(np.float32)) == np.float32


def test_a_sum_reading_one_index_on_axes_of_two_extents_is_not_a_table() -> None:
    """Kept, it would run over ``A``'s rows and read ``B`` past its own (Codex, #348).

    The index ``i`` reads an axis of ``n`` in ``A`` and of ``k`` in ``B``;
    the sum was a table over ``A``'s axis alone. It is none now, and a
    rewrite keeps no such sum.
    """
    from loopty.rewriting import _templated
    from loopty.trace import reductions_in

    term = two_extents.term
    (stmt,) = term.stmts
    (found,) = reductions_in(stmt.expr)
    assert _templated(found, {"out"}, term.array_types) is None


# }}}


# {{{ a postcondition by a cited theorem, under conditions on the tables


# not collected as a property test by lanky's pytest plugin (a leading
# underscore): what these tests ask is what the kernel citing it owes
@theorem
def _sixth_of_the_sum(
    W: Fn[Fin[3], Real],  # noqa: F821
    f: Fn[Fin[3], Real],  # noqa: F821
    sixth: all(W[q] == Fraction(1, 6) for q in Fin[3]),  # noqa: F821
) -> sum(W[q] * f[q] for q in Fin[3]) == (f[0] + f[1] + f[2]) / 6:  # noqa: F821
    """Weights of a sixth make a sixth of the sum."""


@kernel(uses=[_sixth_of_the_sum])
def sixth(x: Arr[Fin[3], Real], y: Arr[Fin[1], Real]) -> (  # noqa: F821
    y[0] == (x[0] + x[1] + x[2]) / 6  # noqa: F821
):
    """A sixth of the sum, from a table of weights written to 15 digits."""
    y[0] = reduce_sum(W3[q] * x[q] for q in W3.dom)


#: A sixth to 11 digits: within the approx tolerance, and not the 15 digits.
W11 = table([0.16666666666, 0.16666666666, 0.16666666666])


@kernel(uses=[_sixth_of_the_sum])
def sixth_rounded(x: Arr[Fin[3], Real], y: Arr[Fin[1], Real]) -> (  # noqa: F821
    y[0] == (x[0] + x[1] + x[2]) / 6  # noqa: F821
):
    """The same, from weights written to 11 digits."""
    y[0] = reduce_sum(W11[q] * x[q] for q in W11.dom)


@kernel(uses=[_sixth_of_the_sum])
def sixth_claimed_otherwise(x: Arr[Fin[3], Real], y: Arr[Fin[1], Real]) -> (  # noqa: F821
    y[0] == (x[0] + x[1] + x[2]) / 6 * (1 + 1e-8)  # noqa: F821
):
    """The same sum, claimed a sixth of it give or take 1e-8: true at approx only."""
    y[0] = reduce_sum(W3[q] * x[q] for q in W3.dom)


def example_inputs() -> dict:
    rng = np.random.default_rng(6)
    x = Arr.from_numpy(rng.normal(size=3))
    return {
        "sixth": {"x": x, "y": Arr.zeros(1)},
        "sixth_rounded": {"x": x, "y": Arr.zeros(1)},
        "sixth_claimed_otherwise": {"x": x, "y": Arr.zeros(1)},
    }


def test_a_postcondition_by_the_theorem_it_cites_under_the_tables_condition() -> None:
    facts = {fact.kind: fact for fact in sixth.facts() if fact.kind != "in-bounds"}
    post = facts["postcondition"]
    assert post.statement.endswith(", by _sixth_of_the_sum")
    condition = facts["rewrite-condition"]
    assert condition.statement == (
        "W3[0] == 1/6 and W3[1] == 1/6 and W3[2] == 1/6, which _sixth_of_the_sum "
        "needs, at the table's digits"
    )
    assert condition.status.value == "tested", condition.provenance
    assert condition.id in post.rests_on
    assert _sixth_of_the_sum.fact_id in post.rests_on
    obligation = post.term.lean_obligation()
    assert not getattr(obligation, "declined", None), obligation


def test_a_theorem_that_concludes_another_right_side_is_not_cited_for_it() -> None:
    """Its left side is what the kernel stores, its right side not the claim's.

    Codex on #348: ``_sixth_of_the_sum`` concludes ``(x[0] + x[1] + x[2]) / 6`` of what
    ``sixth_claimed_otherwise`` stores, which claims that times ``1 + 1e-8``:
    the same to the approx tolerance, which the runs test, and not over the
    reals. Matched by its left side alone, the row read "by
    _sixth_of_the_sum" and rested on the theorem and its condition on the
    tables; the claim is now the kernel's own, tested on the runs.
    """
    facts = {
        fact.kind: fact
        for fact in sixth_claimed_otherwise.facts()
        if fact.kind != "in-bounds"
    }
    post = facts["postcondition"]
    assert post.status.value == "tested", post.provenance
    assert "_sixth_of_the_sum" not in post.statement
    assert _sixth_of_the_sum.fact_id not in post.rests_on
    assert "rewrite-condition" not in facts


def test_a_right_side_is_compared_at_draws_of_both_signs() -> None:
    """``abs(x)`` is not ``x``, though they agree at every positive draw (Codex, #348).

    Drawn only between a half and two, the comparison took a theorem
    concluding ``abs(x)`` for a claim of ``x``.
    """
    from lanky.terms import Abs, Subscript, Var

    from loopty.reals import _one_value

    x = Subscript(Var("x"), 0)
    J = Var("J")
    assert not _one_value(Abs(x), x)
    assert not _one_value(Abs(x - Subscript(Var("x"), 1)), x - Subscript(Var("x"), 1))
    assert _one_value(Abs(J) * 2 / 24, Abs(J) / 12)
    assert _one_value(Abs(-x), Abs(x))
    # a complex cell is drawn off the real axis (Codex, #348)
    z = Subscript(Var("z"), 0)
    assert _one_value(Abs(z) ** 2, z * z)  # equal at every real draw
    assert not _one_value(Abs(z) ** 2, z * z, frozenset({"z"}))
    assert _one_value(Abs(z) * 2 / 24, Abs(z) / 12, frozenset({"z"}))


def test_weights_off_beyond_their_digits_refute_the_condition() -> None:
    """Close enough for the runs at approx, and not the rule to 15 digits."""
    facts = {fact.kind: fact for fact in sixth_rounded.facts()}
    assert facts["postcondition"].statement.endswith(", by _sixth_of_the_sum")
    condition = facts["rewrite-condition"]
    assert condition.status.value == "refuted"
    assert "at the tables' 15 digits" in condition.provenance["reason"]


# }}}
