"""A cited theorem about a series, applied to the terms a kernel adds (#252).

``tests/kernels/series.py``'s kernels cite ``halves``, the powers of ``x / 2``,
and ``halved``, the identity that ``x * 0.5`` is ``x / 2``. The first half of
this file needs no Lean; the last test proves a row with Mathlib, and skips
without it.
"""

from __future__ import annotations

import importlib.util
import os
import sys
from pathlib import Path
from typing import Any

import pytest
from lanky.ledger import Status
from lanky.plugins import registry
from lanky.terms import render

from lines import said
from loopty.summands import KIND, series_applications

ROOT = Path(__file__).resolve().parent.parent



def load(path: Path) -> Any:
    """Import a file of kernels as a module of its own."""
    name = f"summands_{path.parent.name}_{path.stem}"
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    sys.path.insert(0, str(path.parent))
    try:
        spec.loader.exec_module(module)
    finally:
        sys.path.remove(str(path.parent))
    return module


with registry.collecting():
    kernels = load(ROOT / "tests" / "kernels" / "series.py")


def rows(kernel: Any) -> list[Any]:
    return [
        fact
        for fact in kernel.facts()
        if fact.kind == KIND and fact.status is not Status.ASSUMED
    ]


def nothing(kernel: Any) -> dict[str, str]:
    """Why each theorem ``kernel`` cites gives it nothing, as its row says (#234)."""
    return {
        fact.statement.split(",")[0]: fact.provenance["unapplied"]
        for fact in kernel.facts()
        if fact.kind == KIND and fact.status is Status.ASSUMED
    }


#: What a kernel that adds no series of ``halves`` is told of the two it cites.
NO_SERIES = {
    "halves": "no sum of {k} adds the terms of its series c(n)",
    "halved": (
        "it is an identity, which puts a series theorem's terms in the kernel's "
        "spelling, and no series {k} adds is read with it"
    ),
}


def test_the_terms_a_kernel_adds_are_the_theorems_through_the_identity() -> None:
    """``(x[i] * 0.5) ** k`` is ``halves``'s ``c(k)``, by ``halved`` of ``x / 2``."""
    (row,) = rows(kernels.power_sum)
    assert row.statement == (
        "power_sum adds halves's c(n) for n in 0..3, for every c with first and "
        "step, read over the reals"
    )
    assert (row.status, row.decided_by) == (Status.DECIDED, "theorem")
    assert row.provenance["bindings"] == "x = x[i]"
    assert row.provenance["orders"] == [0, 1, 2, 3]
    assert row.provenance["terms"][2] == "(x[i]*0.5)**2"
    assert row.rests_on == (kernels.halves.fact_id, kernels.halved.fact_id)
    assert row.where == f"series.py:{row.where.rsplit(':', 1)[1]}"
    # the statement by its text and line, as a message names it (#307)
    stmt = said(
        kernels.power_sum,
        "y[i] = reduce_sum((x[i] * 0.5) ** k for k in Fin[ORDER + 1])",
    )
    assert row.provenance["rule"].startswith(f"{stmt} adds (x[i]*0.5)**0 and ")


def test_a_kernel_that_adds_other_terms_gets_no_row() -> None:
    assert rows(kernels.quarter_sum) == []
    # but each theorem it cites says that it gives nothing, and why (#234)
    assert nothing(kernels.quarter_sum) == {
        name: words.format(k="quarter_sum") for name, words in NO_SERIES.items()
    }


def test_a_term_inside_another_is_not_one_the_kernel_adds() -> None:
    """``1 / (x[i] * 0.5) ** 2`` divides by ``c(2)`` and adds nothing (Codex)."""
    assert rows(kernels.inverse_square) == []
    assert set(nothing(kernels.inverse_square)) == {"halves", "halved"}


def test_a_theorem_about_reals_is_not_applied_to_complex_numbers() -> None:
    """The same powers of a complex ``x[i]`` match the term, not the sort (Codex)."""
    assert rows(kernels.complex_sum) == []
    assert set(nothing(kernels.complex_sum)) == {"halves", "halved"}


def test_an_identitys_premise_is_named_in_the_row() -> None:
    """Lean is handed ``halved_from_zero``'s premise, so the row says it (Codex)."""
    (row,) = rows(kernels.premised_sum)
    assert row.statement == (
        "premised_sum adds halves's c(n) for n in 0..3, for every c with first, "
        "step and halved_from_zero's nonnegative, read over the reals"
    )
    assert row.rests_on == (kernels.halves.fact_id, kernels.halved_from_zero.fact_id)


def test_a_term_the_theorems_guard_leaves_out_is_not_claimed() -> None:
    """``from_one`` is about ``n >= 1``, so ``x[i] ** 0`` is no term of its."""
    (row,) = rows(kernels.tail_sum)
    assert row.provenance["orders"] == [1, 2, 3]
    assert row.statement.startswith("tail_sum adds from_one's c(n) for n in 1..3, ")
    # no identity was needed, and none is named
    assert row.rests_on == (kernels.from_one.fact_id,)


def test_each_term_is_a_part_lean_rewrites_and_applies_the_theorem_to() -> None:
    """One part per order: the identity rewritten in the goal, the theorem applied."""
    from lanky.obligations import attempts, obligation_of, part_statement, theorem_name

    (row,) = rows(kernels.power_sum)
    obligation = obligation_of(row.term)
    assert not obligation.declined
    assert [part.name for part in obligation.parts] == [f"term_{k}" for k in range(4)]
    part = obligation.parts[2]
    assert part.mathlib
    assert part.label == "halves's c(2) is what power_sum adds"
    statement = part_statement(part, theorem_name("power_sum", part))
    assert statement.goal_term is not None
    # the theorem's c is no name of the kernel's, and keeps its own
    assert render(statement.goal_term) == "c(2) == (x[i]*0.5)**2"
    tried = dict(attempts(part, statement))
    script = tried["rewritten by the identity, the theorem applied"].splitlines()
    assert script[1].startswith("rw [h1 (x := x i)]")
    assert script[2] == "apply h0 (x := x i) (c := c)"


def test_series_are_found_in_the_term_and_not_in_its_spelling() -> None:
    (found,) = series_applications(kernels.power_sum.term, kernels.power_sum.uses)
    assert found.family == "c" and found.binder == "n"
    assert [order for order, _ in found.terms] == [0, 1, 2, 3]
    ((identity, values),) = found.rewrites
    assert identity.theorem is kernels.halved
    assert [(name, render(value)) for name, value in values] == [("x", "x[i]")]


@pytest.mark.lean
def test_with_mathlib_each_term_is_proved_to_be_the_theorems() -> None:
    from lanky.oracles.lean import LeanOracle

    if os.environ.get("LANKY_LEAN_DISABLE"):
        pytest.skip("the Lean oracle is disabled by LANKY_LEAN_DISABLE")
    oracle = LeanOracle(timeout=float(os.environ.get("LANKY_LEAN_TEST_TIMEOUT", "120")))
    available, reason = oracle.availability()
    if not available:
        if os.environ.get("LANKY_LEAN_TEST_REQUIRED"):
            pytest.fail(f"LANKY_LEAN_TEST_REQUIRED is set, but {reason}")
        pytest.skip(reason)
    if not oracle._mathlib():
        pytest.skip("Mathlib is not imported here (LANKY_LEAN_MATHLIB)")
    (row,) = rows(kernels.power_sum)
    proved = oracle.establish(row)
    assert proved.status is Status.PROVED, proved.provenance.get("lean_reason")
    assert "rw [h1" in proved.provenance["lean_source"]
