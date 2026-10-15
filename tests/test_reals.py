"""A postcondition's equations of reals, compared at their class (#254).

``tests/kernels/reals.py``'s ``thirds`` takes a mean with a table of
15-digit thirds: its postcondition holds to the approx tolerance, as its
compiled run is compared with its native one, and not bit for bit. The first
half of this file needs no Lean; the last test proves it with Mathlib, and
skips without it.
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

from loopty.reals import AtTheClass, at_the_class, comparison_class

ROOT = Path(__file__).resolve().parent.parent


def load(path: Path) -> Any:
    """Import a file of kernels as a module of its own."""
    name = f"reals_{path.parent.name}_{path.stem}"
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
    kernels = load(ROOT / "tests" / "kernels" / "reals.py")


def post(kernel: Any) -> Any:
    (fact,) = [fact for fact in kernel.facts() if fact.kind == "postcondition"]
    return fact


def test_a_mean_by_rounded_thirds_holds_to_the_approx_tolerance() -> None:
    """Bit for bit it is false; at the class its runs are compared at, it holds."""
    fact = post(kernels.thirds)
    assert fact.status is Status.TESTED, fact.provenance
    assert fact.decided_by == "native"
    assert fact.statement == "y[0] == (x[0] + x[1] + x[2]) / 3, to the approx tolerance"
    term = kernels.thirds.term
    assert comparison_class(term, term.post) == "approx"
    assert render(at_the_class(term, term.post)) == (
        "abs(y[0] - (x[0] + x[1] + x[2]) / 3) <= "
        "1e-06*(abs((x[0] + x[1] + x[2]) / 3) + 1.0)"
    )


def test_it_rests_on_what_the_kernel_stores_and_on_its_trace() -> None:
    """Lean proves it of the term over the reals: y[0] holds what S0 stores there."""
    facts = kernels.thirds.facts()
    fact = post(kernels.thirds)
    assert isinstance(fact.term, AtTheClass)
    (stored,) = [f for f in facts if f.kind == "single-assignment"]
    assert stored.provenance["array"] == "y"
    assert stored.id in fact.rests_on
    (faithful,) = [f for f in facts if f.kind == "trace-faithful"]
    assert faithful.id in fact.rests_on
    # the single-assignment fact it rests on comes before it
    assert facts.index(stored) < facts.index(fact)


def test_an_exact_class_is_compared_bit_for_bit_and_refutes() -> None:
    fact = post(kernels.thirds_exact)
    assert fact.status is Status.REFUTED
    assert fact.statement == "y[0] == (x[0] + x[1] + x[2]) / 3"
    assert "tolerance" not in fact.provenance["reason"]


def test_a_miss_beyond_the_tolerance_refutes_and_says_the_tolerance() -> None:
    fact = post(kernels.off_mean)
    assert fact.status is Status.REFUTED
    assert fact.provenance["reason"].endswith(
        "false, to the approx tolerance, so off_mean does not establish its "
        "postcondition"
    )


def test_an_equation_nothing_in_which_rounds_is_compared_bit_for_bit() -> None:
    term = kernels.copied.term
    assert at_the_class(term, term.post) is term.post
    fact = post(kernels.copied)
    assert fact.statement == "forall i in Fin(n). y[i] == -x[i]"
    assert fact.status is Status.DECIDED


def test_each_conjunct_is_stated_for_lean_over_the_inputs() -> None:
    """The cell is what its statement stores, the sum written out, the table read."""
    obligations = pytest.importorskip("lanky.obligations")
    if "mathlib" not in getattr(obligations.Part, "__dataclass_fields__", {}):
        pytest.skip("this lanky states no part of Mathlib's")
    obligation = obligations.obligation_of(post(kernels.thirds).term)
    assert not obligation.declined
    (part,) = obligation.parts
    assert part.mathlib
    statement = obligations.part_statement(part, "thirds_conjunct_0")
    (hypothesis,) = [h for h in statement.hypothesis_terms if h is not None]
    third = 0.333333333333333
    assert render(hypothesis) == (
        f"y[0] == {third}*x[0] + {third}*x[1] + {third}*x[2]"
    )


@pytest.mark.lean
def test_with_mathlib_the_mean_is_proved_to_the_tolerance() -> None:
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
    proved = oracle.establish(post(kernels.thirds))
    assert proved.status is Status.PROVED, proved.provenance.get("lean_reason")
    assert "abs_le" in proved.provenance["lean_source"]
    # and a miss beyond the tolerance is no proof
    missed = oracle.establish(post(kernels.off_mean))
    assert missed.status is Status.REFUTED
