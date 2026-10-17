"""Math steps of a schedule: ``Schedule(k).rewrite(k.out, by=theorem)``.

A math step changes what a statement computes by an identity: wherever what an
array stores contains the theorem's left side, the schedule computes its right
side. These tests take the single layer of ``examples/layer_potential.py`` in a
small form, the math as written, and rewrite it as sumpy computes it: each
division through the conjugate, and ``log|alpha|`` with no square root.
"""

from __future__ import annotations

import numpy as np
import pytest
from lanky import theorem
from lanky.prelude import Complex, Fn, Nat, Real
from lanky.terms import render

from loopty import Arr, Fin, IllegalRewrite, Schedule, kernel, program, reduce_sum
from loopty.math import conj, log, real
from loopty.rewriting import cited, identity_of

pytest.importorskip("loopy")

ORDER = 3


def coefficient(ratio, k):
    """The ``k``-th Taylor coefficient of ``log(alpha + tau * beta)``."""
    return -(ratio**k) / k


@theorem
def conjugate_form(x: Complex, alpha: Complex, nonzero: alpha != 0) -> (  # noqa: F821
    x / alpha == x * conj(alpha) * (1 / real(alpha * conj(alpha)))  # noqa: F821
):
    """Division by alpha through its conjugate."""


@theorem
def log_of_the_modulus(alpha: Complex, nonzero: alpha != 0) -> (  # noqa: F821
    log(abs(alpha)) == log(real(alpha * conj(alpha))) / 2  # noqa: F821
):
    """``log|alpha|`` with no square root."""


@theorem
def quotient_of_the_conjugate(x: Complex, alpha: Complex, nonzero: alpha != 0) -> (  # noqa: F821
    x * conj(alpha) * (1 / real(alpha * conj(alpha))) == x / alpha  # noqa: F821
):
    """The conjugate form read backwards: what a kernel should not be rewritten by."""


@theorem
def doubling(x: Real) -> x + x == 2 * x:  # noqa: F821
    """A theorem neither side of which the kernel computes."""


@theorem
def anything(x: Complex) -> x == x + 0:  # noqa: F821
    """A left side that is a bare variable matches every term."""


def single(alpha, beta):
    return log(abs(alpha)) + real(
        reduce_sum(coefficient(-beta / alpha, k) for k in Fin[ORDER + 1] if k >= 1)
    )


@kernel
def layer(
    z: Arr[Fin[nt], Complex],  # noqa: F821
    c: Arr[Fin[nt], Complex],  # noqa: F821
    y: Arr[Fin[ns], Complex],  # noqa: F821
    w: Arr[Fin[ns], Real],  # noqa: F821
    out: Arr[Fin[nt], Real],  # noqa: F821
):
    """The single layer, the math as written."""
    for t in out.dom:
        out[t] = reduce_sum(w[s] * single(c[t] - y[s], z[t] - c[t]) for s in y.dom)


@kernel
def counts(a: Arr[Fin[n], Fin[n]], b: Arr[Fin[n], Fin[n]]):  # noqa: F821
    """An integral output, compared bit for bit."""
    for i in b.dom:
        b[i] = a[i]


def inputs(nt: int = 5, ns: int = 7, seed: int = 0, apart: bool = True) -> dict:
    rng = np.random.default_rng(seed)
    z = rng.normal(size=nt) + 1j * rng.normal(size=nt)
    c = z + 0.1 * (rng.normal(size=nt) + 1j * rng.normal(size=nt))
    y = 3 + rng.normal(size=ns) + 1j * rng.normal(size=ns)
    if not apart:
        y[2] = c[1]
    return {
        "z": Arr.from_numpy(z),
        "c": Arr.from_numpy(c),
        "y": Arr.from_numpy(y),
        "w": Arr.from_numpy(rng.normal(size=ns)),
        "out": Arr.zeros(nt),
    }


def as_sumpy_computes() -> Schedule:
    return Schedule(layer).rewrite(layer.out, by=(conjugate_form, log_of_the_modulus))


def rows(schedule: Schedule) -> dict[str, list]:
    out: dict[str, list] = {}
    for fact in schedule.facts():
        out.setdefault(fact.kind, []).append(fact)
    return out


# {{{ lanky's Complex as an element type


def test_lankys_complex_is_an_element_type_stored_as_complex128() -> None:
    from loopty.contract import compiled_storage, native_storage

    assert compiled_storage(Complex) == np.dtype(np.complex128)
    assert native_storage(Complex) == np.dtype(np.complex128)
    args = inputs()
    layer(**args)
    from loopty.executor import LoopyExecutor

    fact = LoopyExecutor().differential(layer, Schedule(layer), inputs())
    assert fact.status.value == "tested", fact.provenance


# }}}


# {{{ matching, by type, and what the schedule compiles


def test_the_conjugate_form_binds_the_complex_alpha_and_never_the_order() -> None:
    """``x / alpha`` matches ``-beta / alpha`` in each term, not ``-(r**k) / k``.

    The order ``k`` the trace writes out is an integer, and ``alpha`` is a
    ``Complex``; the five-term series of the design has three terms here.
    """
    schedule = Schedule(layer).rewrite(layer.out, by=conjugate_form)
    (rewritten,) = schedule.rewrites
    (step,) = rewritten.steps
    assert len(step.matches) == ORDER
    for match in step.matches:
        sigma = dict(match.sigma)
        assert render(sigma["alpha"]) == "c[t] - y[s]"
        assert [name for name, _ in match.within] == ["s"]
    assert schedule.traced is layer.term
    assert schedule.term is not layer.term
    assert schedule.rewritten == frozenset({"out"})


def test_theorems_apply_in_order_each_to_what_the_one_before_left() -> None:
    schedule = as_sumpy_computes()
    (rewritten,) = schedule.rewrites
    assert [len(step.matches) for step in rewritten.steps] == [ORDER, 1]
    assert schedule.history == (
        "rewrite(layer.out, by=(conjugate_form, log_of_the_modulus))",
    )
    assert schedule.key == (
        "layer[c].rewrite(layer.out, by=(conjugate_form, log_of_the_modulus))"
    )


def test_the_compiled_rewrite_computes_what_the_native_kernel_does() -> None:
    from loopty.executor import LoopyExecutor

    fact = LoopyExecutor().differential(layer, as_sumpy_computes(), inputs())
    assert fact.status.value == "tested", fact.provenance
    assert fact.provenance["outputs"]["out"]["exactness"] == "approx"


def test_a_cast_after_the_rewrite_is_checked_on_the_rewritten_term() -> None:
    tiled = as_sumpy_computes().split(layer.t, 2)
    assert tiled.history[0].startswith("rewrite(layer.out")
    casts = [fact for fact in tiled.facts() if fact.kind in ("bijective", "monotone")]
    assert casts and all(fact.status.value == "decided" for fact in casts)


def test_retargeting_replays_the_rewrite() -> None:
    again = Schedule(as_sumpy_computes())
    assert again.history == as_sumpy_computes().history
    assert again.rewritten == frozenset({"out"})


# }}}


# {{{ refusals


def test_a_rewrite_after_a_cast_is_refused_with_where_to_put_it() -> None:
    with pytest.raises(IllegalRewrite) as caught:
        Schedule(layer).split(layer.t, 2).rewrite(layer.out, by=conjugate_form)
    assert str(caught.value) == (
        "rewrite(layer.out, by=conjugate_form) comes after split(layer.t, 2): a "
        "math step changes what the instances compute, and the steps after it "
        "are checked on what it leaves, so it comes before the loops; put "
        "rewrite(layer.out, ...) before split(layer.t, 2)"
    )


def test_a_theorem_stated_the_other_way_round_is_refused_naming_the_fix() -> None:
    with pytest.raises(IllegalRewrite) as caught:
        Schedule(layer).rewrite(layer.out, by=quotient_of_the_conjugate)
    assert str(caught.value).startswith(
        "rewrite(layer.out, by=quotient_of_the_conjugate): x / alpha, the right "
        "side of quotient_of_the_conjugate, is in what layer.out stores, and its "
        "left side is not: state it the other way round, as 'what the kernel "
        "says == what it should compute' (out[t] = reduce_sum(...) "
    )


def test_a_rewrite_by_the_converse_undoes_the_one_before() -> None:
    """Each theorem rewrites what the step before left: the converse puts it back."""
    back = as_sumpy_computes().rewrite(layer.out, by=quotient_of_the_conjugate)
    assert len(back.rewrites) == 2
    assert [len(step.matches) for step in back.rewrites[1].steps] == [ORDER]


def test_a_theorem_whose_sides_are_nowhere_is_refused_naming_the_statement() -> None:
    with pytest.raises(IllegalRewrite) as caught:
        Schedule(layer).rewrite(layer.out, by=doubling)
    assert str(caught.value).startswith(
        "rewrite(layer.out, by=doubling): x + x, the left side of doubling, is "
        "nowhere in what layer.out stores: out[t] = reduce_sum(...) "
    )


def test_a_bare_variable_on_the_left_is_refused() -> None:
    with pytest.raises(IllegalRewrite, match="matches every term"):
        identity_of(anything, "rewrite(layer.out, by=anything)")


def test_an_exact_output_is_refused() -> None:
    with pytest.raises(IllegalRewrite, match="compared exact, bit for bit"):
        Schedule(counts).rewrite(counts.b, by=conjugate_form)


def test_a_rewrite_names_an_array_by_its_handle() -> None:
    with pytest.raises(TypeError, match="rewrite\\(\\) takes an array handle"):
        Schedule(layer).rewrite("out", by=conjugate_form)


# }}}


# {{{ the rows


def test_the_rows_say_what_was_rewritten_over_the_complex_numbers() -> None:
    found = rows(as_sumpy_computes().example(**inputs()))
    first, second = found["rewrite"]
    assert first.statement == (
        "rewrite(out, by=conjugate_form): over the complex numbers, what out "
        "stores is unchanged wherever c[t] - y[s] != 0: -(z[t] - c[t]) / (c[t] - "
        "y[s]) becomes its conjugate form, in 3 places"
    )
    assert second.statement == (
        "rewrite(out, by=log_of_the_modulus): over the complex numbers, what out "
        "stores is unchanged wherever c[t] - y[s] != 0: log(abs(c[t] - y[s])) "
        "becomes log(real((c[t] - y[s])*conj(c[t] - y[s]))) / 2, in 1 place"
    )
    # without Lean, each step is tested by draws of T == T' on the interpreter
    for row in (first, second):
        assert row.status.value in ("tested", "proved"), row.provenance
        assert conjugate_form.fact_id in row.rests_on or (
            log_of_the_modulus.fact_id in row.rests_on
        )
    (condition,) = found["rewrite-condition"]
    assert condition.statement == (
        "c[t] - y[s] != 0 for every t and every s, which conjugate_form and "
        "log_of_the_modulus need, on the example inputs"
    )
    assert (condition.status.value, condition.decided_by) == ("tested", "run")
    assert condition.id in first.rests_on
    (exactness,) = found["exactness"]
    assert exactness.statement == (
        "after rewrite(out, ...), out is compared at approx (it was approx)"
    )
    assert (exactness.status.value, exactness.decided_by) == ("decided", "type")


def test_a_data_condition_that_fails_on_the_example_inputs_is_refuted() -> None:
    found = rows(as_sumpy_computes().example(**inputs(apart=False)))
    (condition,) = found["rewrite-condition"]
    assert condition.status.value == "refuted"
    assert condition.provenance["reason"] == (
        "c[t] - y[s] != 0 fails at t=1, s=2"
    )


@theorem
def through_the_reciprocal(x: Real, a: Real, nonzero: a != 0) -> (  # noqa: F821
    x / a == x * (1 / a)  # noqa: F821
):
    """Division as multiplication by the reciprocal."""


@kernel
def two_branches(
    x: Arr[Fin[n], Real],  # noqa: F821
    a: Arr[Fin[n], Real],  # noqa: F821
    out: Arr[Fin[n], Real],  # noqa: F821
):
    """Two statements store out, each dividing by a on its own cells."""
    for i in out.dom:
        if i < 2:
            out[i] = x[i] / a[i]
        else:
            out[i] = 2.0 * x[i] / a[i]


def test_a_condition_two_statements_need_is_tested_over_each() -> None:
    """``a[i] != 0`` on the first two cells and on the rest: a row each (Codex, #341).

    One row for both tested the claim over the first statement's cells alone,
    so the zero at i = 3, which only the second statement divides by, passed.
    """
    a = np.ones(5)
    a[3] = 0.0
    schedule = (
        Schedule(two_branches)
        .rewrite(two_branches.out, by=through_the_reciprocal)
        .example(
            x=Arr.from_numpy(np.arange(5.0)), a=Arr.from_numpy(a), out=Arr.zeros(5)
        )
    )
    first, second = rows(schedule)["rewrite-condition"]
    assert (first.status.value, second.status.value) == ("tested", "refuted")
    assert second.provenance["reason"] == "a[i] != 0 fails at i=3"
    assert first.statement != second.statement
    assert first.statement.startswith("a[i] != 0 for every i, in out[i] = ")
    (equality,) = rows(schedule)["rewrite"]
    assert {first.id, second.id} <= set(equality.rests_on)


def test_the_compared_row_is_measured_and_notes_the_extremes() -> None:
    """The float range moves, as sumpy's does: a note at the extremes, not a failure.

    ``alpha * conj(alpha)`` underflows at ``|alpha|`` about ``1e-170`` and
    overflows at ``1e170``, where ``log|alpha|`` does not.
    """
    found = rows(as_sumpy_computes().example(**inputs()))
    (compared,) = found["rewrite-compared"]
    assert compared.status.value == "tested", compared.provenance
    assert compared.statement.startswith(
        "the compiled layer after rewrite(out, ...) agrees with the native layer "
        "in out at approx, within 1e-06 of each value (measured "
    )
    assert (
        "on its example inputs and 3 drawn ones; note: with the inputs scaled by "
        "1e-170"
    ) in compared.statement
    notes = compared.provenance["notes"]
    assert any("1e-170" in note and "nan" in note for note in notes), notes
    assert compared.provenance["outputs"]["out"]["measured"] < 1e-12


def test_a_condition_is_needed_only_where_the_statement_runs() -> None:
    """``x[i] / a[i]`` under ``if a[i] != 0``: a zero where it does not run is fine.

    The condition was tested at every point of the domain, which a guard
    that reads data does not narrow, and refuted at a zero the guard skips
    (Codex, #341).
    """

    @kernel
    def guarded(
        x: Arr[Fin[n], Real],  # noqa: F821
        a: Arr[Fin[n], Real],  # noqa: F821
        out: Arr[Fin[n], Real],  # noqa: F821
    ):
        for i in out.dom:
            if a[i] != 0.0:
                out[i] = x[i] / a[i]

    a = np.ones(5)
    a[3] = 0.0
    inputs_ = {"x": Arr.from_numpy(np.arange(5.0)), "a": Arr.from_numpy(a)}
    schedule = (
        Schedule(guarded)
        .rewrite(guarded.out, by=through_the_reciprocal)
        .example(**inputs_, out=Arr.zeros(5))
    )
    (condition,) = rows(schedule)["rewrite-condition"]
    assert condition.status.value == "tested", condition.provenance
    assert condition.statement.startswith(
        "a[i] != 0 for every i where a[i] != 0.0, which through_the_reciprocal needs"
    )
    # and Lean is asked T == T' where the statement runs, which the row tests
    (equality,) = rows(schedule)["rewrite"]
    (part,) = equality.term.lean_obligation().parts
    guards = [render(guard) for guard in part.term.guard.children]
    assert "a[i] != 0.0" in guards, guards


def test_a_rewrite_finite_before_and_nan_after_at_a_draw_is_refuted() -> None:
    """``log(x * x)`` as ``2 * log(x)``: NaN where x < 0, and the row says so.

    A cell finite on one side and NaN on the other was left out of the
    comparison of drawn inputs, so the step read tested (Codex, #341).
    """

    @theorem
    def log_of_a_square(x: Real) -> log(x * x) == 2 * log(x):  # noqa: F821
        """False for a negative x, where the right side is NaN."""

    @kernel
    def logs(x: Arr[Fin[n], Real], out: Arr[Fin[n], Real]):  # noqa: F821
        for i in out.dom:
            out[i] = log(x[i] * x[i])

    (row,) = rows(Schedule(logs).rewrite(logs.out, by=log_of_a_square))["rewrite"]
    assert row.status.value == "refuted", row.provenance
    assert "nan after" in row.provenance["reason"], row.provenance


def test_draws_that_break_the_theorems_condition_are_not_compared() -> None:
    """``log(x * x) == 2 * log(x)`` where ``x > 0``: draws with x <= 0 say nothing.

    The theorem says nothing where its condition fails, so a draw there is
    no evidence either way (Codex, #341): with every draw outside it, the
    row says why it is not tested.
    """

    @theorem
    def log_of_a_square(x: Real, positive: x > 0) -> log(x * x) == 2 * log(x):  # noqa: F821
        """True where x > 0."""

    @kernel
    def logs(x: Arr[Fin[n], Real], out: Arr[Fin[n], Real]):  # noqa: F821
        for i in out.dom:
            out[i] = log(x[i] * x[i])

    (row,) = rows(Schedule(logs).rewrite(logs.out, by=log_of_a_square))["rewrite"]
    assert row.status.value in ("tested", "assumed"), row.provenance
    if row.status.value == "assumed":
        reason = row.provenance["reason"]
        assert "broke a condition the theorems need, x[i] > 0" in reason, reason


def test_a_later_step_leaves_an_earlier_steps_compared_row_as_it_was() -> None:
    """A rewrite compares the schedule it made, asked of a later one (Codex, #341)."""

    @theorem
    def wrong_form(x: Complex, alpha: Complex, nonzero: alpha != 0) -> (  # noqa: F821
        x / alpha == x * alpha  # noqa: F821
    ):
        """False: what a later step should not refute the earlier row by."""

    first = Schedule(layer).rewrite(layer.out, by=log_of_the_modulus)
    first = first.example(**inputs())
    later = first.rewrite(layer.out, by=wrong_form)
    alone = rows(first)["rewrite-compared"][0]
    earlier, wrong = rows(later)["rewrite-compared"]
    assert alone.status.value == earlier.status.value == "tested", earlier.provenance
    assert earlier.statement == alone.statement
    assert wrong.status.value == "refuted"


def test_without_examples_the_data_rows_say_what_they_need() -> None:
    found = rows(as_sumpy_computes())
    (condition,) = found["rewrite-condition"]
    assert condition.status.value == "assumed"
    assert "no example inputs" in condition.provenance["reason"]


def test_the_lean_part_restates_the_theorem_and_the_condition_under_the_sum() -> None:
    """One part per statement: the theorem, the condition for every s, T == T'."""
    (first, _second) = rows(as_sumpy_computes())["rewrite"]
    obligation = first.term.lean_obligation()
    assert not obligation.declined
    (part,) = obligation.parts
    assert part.mathlib
    guards = [render(guard) for guard in part.term.guard.children]
    assert render(cited(conjugate_form)) in guards
    assert "forall s in Fin(ns). c[t] - y[s] != 0" in guards
    (rewrite,) = part.hints.rewrites
    assert guards[rewrite.hypothesis] == render(cited(conjugate_form))
    assert obligation.rests_on == (conjugate_form.fact_id,)


def test_the_part_names_the_theorem_it_restates_as_a_citation() -> None:
    """lanky restates the theorem from the text its command proved, by the citation."""
    assert render(cited(conjugate_form)) == render(conjugate_form.term)
    (first, _second) = rows(as_sumpy_computes())["rewrite"]
    (part,) = first.term.lean_obligation().parts
    citations = getattr(part, "citations", None)
    if citations is None:  # a lanky without citations restates by the printer
        return
    (citation,) = citations
    guards = [render(guard) for guard in part.term.guard.children]
    assert guards[citation.hypothesis] == render(conjugate_form.term)
    assert citation.fact == conjugate_form.fact_id


# }}}


# {{{ structural rewrites: keep= and into=


@theorem
def two_passes(
    n: Nat,
    F: Fn[Fin[n], Fin[n], Real],  # noqa: F821
    G: Fn[Fin[n], Fin[n], Real],  # noqa: F821
    u: Fn[Fin[n], Fin[n], Real],  # noqa: F821
    k: Fin[n],  # noqa: F821
    j: Fin[n],  # noqa: F821
) -> sum(  # noqa: F821
    F[k, b] * G[j, a] * u[b, a] for b in Fin[n] for a in Fin[n]  # noqa: F821
) == sum(  # noqa: F821
    F[k, b] * sum(G[j, a] * u[b, a] for a in Fin[n]) for b in Fin[n]  # noqa: F821
):
    """The Kronecker product of two tables applied to a matrix is two passes."""


@kernel
def kron(
    F: Arr[Fin[n], Fin[n], Real],  # noqa: F821
    G: Arr[Fin[n], Fin[n], Real],  # noqa: F821
    u: Arr[Fin[n], Fin[n], Real],  # noqa: F821
    out: Arr[Fin[n], Fin[n], Real],  # noqa: F821
):
    """The dense operator, as the math says it."""
    for k, j in out.dom:
        out[k, j] = reduce_sum(
            F[k, b] * G[j, a] * u[b, a] for b in F.dom[k] for a in G.dom[j]
        )


@kernel
def along_a(
    G: Arr[Fin[n], Fin[n], Real],  # noqa: F821
    u: Arr[Fin[n], Fin[n], Real],  # noqa: F821
    v: Arr[Fin[n], Fin[n], Real],  # noqa: F821
):
    """``G`` along the second axis."""
    for b, j in v.dom:
        v[b, j] = reduce_sum(G[j, a] * u[b, a] for a in G.dom[j])


@kernel
def along_b(
    F: Arr[Fin[n], Fin[n], Real],  # noqa: F821
    v: Arr[Fin[n], Fin[n], Real],  # noqa: F821
    out: Arr[Fin[n], Fin[n], Real],  # noqa: F821
):
    """``F`` along the first axis."""
    for k, j in out.dom:
        out[k, j] = reduce_sum(F[k, b] * v[b, j] for b in F.dom[k])


@program(uses=[two_passes])
def kron_passes(F, G, u, out):
    """The two passes, written by hand."""
    v = Arr.zeros_like(u)
    along_a(G, u, v)
    along_b(F, v, out)


@program
def wrong_arguments(G, u, v):
    """A program that takes other arguments than the kernel."""
    along_a(G, u, v)


def kron_inputs(n: int = 4, seed: int = 1) -> dict:
    rng = np.random.default_rng(seed)
    return {
        "F": Arr.from_numpy(rng.normal(size=(n, n))),
        "G": Arr.from_numpy(rng.normal(size=(n, n))),
        "u": Arr.from_numpy(rng.normal(size=(n, n))),
        "out": Arr.zeros((n, n)),
    }


def test_keep_and_into_refuse_a_guarded_statement() -> None:
    """A kept pass would run where the guard keeps the statement from (Codex, #341)."""

    @kernel
    def corner(
        F: Arr[Fin[n], Fin[n], Real],  # noqa: F821
        G: Arr[Fin[n], Fin[n], Real],  # noqa: F821
        u: Arr[Fin[n], Fin[n], Real],  # noqa: F821
        out: Arr[Fin[n], Fin[n], Real],  # noqa: F821
    ):
        for k, j in out.dom:
            if k <= j:
                out[k, j] = reduce_sum(
                    F[k, b] * G[j, a] * u[b, a] for b in F.dom[k] for a in G.dom[j]
                )

    with pytest.raises(IllegalRewrite, match="runs only where k <= j"):
        Schedule(corner).rewrite(corner.out, by=two_passes, keep=("v",))
    with pytest.raises(IllegalRewrite, match="the program into= names runs at every"):
        Schedule(corner).rewrite(corner.out, by=two_passes, into=kron_passes)
    # without either, the rewrite stays inside the guarded statement
    Schedule(corner).rewrite(corner.out, by=two_passes)


def test_keep_stores_the_inner_sum_and_the_statement_reads_it() -> None:
    from loopty.executor import LoopyExecutor

    factored = Schedule(kron).rewrite(kron.out, by=two_passes, keep=("v",))
    (rewritten,) = factored.rewrites
    (kept,) = rewritten.kept
    assert kept.name == "v"
    # the pass along a stores G applied to u's row b, at j: u[b, a] with a as j
    assert kept.spelled == ("b", "j")
    texts = [stmt.text for stmt in factored.term.stmts]
    assert texts[0] == "v[b, j] = reduce_sum(...)"
    assert ("v", ) == tuple(name for name, _ in factored.term.temporaries)
    fact = LoopyExecutor().differential(kron, factored, kron_inputs())
    assert fact.status.value == "tested", fact.provenance


def test_a_kept_array_and_its_loops_have_handles() -> None:
    factored = Schedule(kron).rewrite(kron.out, by=two_passes, keep=("v",))
    assert repr(factored.v) == "kron.v"
    assert repr(factored.v.b) == "kron.v.b"
    with pytest.raises(AttributeError, match="the arrays its math steps keep are v"):
        factored.w  # noqa: B018


def test_keep_rows_say_what_the_passes_leave_and_their_facts() -> None:
    factored = Schedule(kron).rewrite(
        kron.out, by=two_passes, keep=("v",)
    ).example(**kron_inputs())
    found = rows(factored)
    (row,) = found["rewrite"]
    assert row.statement == (
        "rewrite(out, by=two_passes, keep=(v)): over the reals, the passes leave in "
        "out what kron's statement says, unfolded through v; two_passes at F := F, "
        "G := G, u := u"
    )
    assert row.status.value in ("tested", "proved"), row.provenance
    obligation = row.term.lean_obligation()
    assert not obligation.declined, obligation.declined
    assert two_passes.fact_id in row.rests_on
    kinds = {fact.kind for fact in found.get("in-bounds", ())}
    assert kinds == {"in-bounds"}
    (writes,) = found["disjoint-writes"]
    assert writes.status.value == "decided"
    (defined,) = found["definedness"]
    assert defined.status.value == "decided"
    assigned = {fact.provenance["array"] for fact in found["single-assignment"]}
    assert assigned == {"v", "out"}


def test_keep_refuses_a_name_the_kernel_has() -> None:
    with pytest.raises(IllegalRewrite, match="keep= names a new array, and kron has"):
        Schedule(kron).rewrite(kron.out, by=two_passes, keep=("u",))


def test_keep_refuses_more_names_than_sums_to_store() -> None:
    with pytest.raises(IllegalRewrite, match="nests fewer sums than keep= stores"):
        Schedule(kron).rewrite(kron.out, by=two_passes, keep=("v", "w"))


def test_into_compiles_the_program_and_proves_it_leaves_the_kernels_statement() -> None:
    from loopty.executor import LoopyExecutor

    into = Schedule(kron).rewrite(kron.out, by=two_passes, into=kron_passes)
    assert into.term is kron_passes.term
    assert into.traced is kron.term
    fact = LoopyExecutor().differential(kron, into, kron_inputs())
    assert fact.status.value == "tested", fact.provenance
    (row,) = rows(into.example(**kron_inputs()))["rewrite"]
    assert row.statement.startswith(
        "rewrite(out, by=two_passes, into=kron_passes): over the reals, kron_passes "
        "leaves in out what kron's statement says, unfolded through its calls"
    )
    assert not row.term.lean_obligation().declined


def test_into_refuses_a_program_of_other_arguments() -> None:
    with pytest.raises(IllegalRewrite, match="wrong_arguments takes \\(G, u, v\\)"):
        Schedule(kron).rewrite(kron.out, by=two_passes, into=wrong_arguments)


def test_into_and_keep_are_one_or_the_other() -> None:
    with pytest.raises(IllegalRewrite, match="give one"):
        Schedule(kron).rewrite(
            kron.out, by=two_passes, keep=("v",), into=kron_passes
        )


# }}}
