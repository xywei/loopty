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
from lanky.prelude import Complex, Real
from lanky.terms import render

from loopty import Arr, Fin, IllegalRewrite, Schedule, kernel, reduce_sum
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


def test_cited_is_the_seam_for_the_text_a_proof_restates() -> None:
    assert render(cited(conjugate_form)) == render(conjugate_form.term)

    class Proved:
        term = "the term"
        as_cited = "the text its own command proved"

    assert cited(Proved()) == "the text its own command proved"


# }}}
