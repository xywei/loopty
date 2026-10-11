"""Claims settled by running them: ``@tested`` rows in the ledger.

A claim no oracle reads, an accuracy that converges at its order, is a
function that says whether it held (:mod:`loopty.tested`). Its row is
``tested`` by ``run`` when it did and ``refuted`` with the reason when it did
not, raised, or answered with something that is not a verdict; and nothing
here ever reads more than ``tested``.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
from lanky.ledger import Status

from loopty import tested
from loopty.tested import Tested


def test_a_claim_that_holds_is_a_tested_row_with_what_it_measured() -> None:
    @tested
    def converges():
        """The error falls at order 4."""
        return True, "errors 1e-4, 6e-6: order 4.01"

    (fact,) = converges.facts()
    assert fact.status is Status.TESTED
    assert fact.decided_by == "run"
    assert fact.kind == "tested"
    assert fact.owner.endswith("converges")
    # the claim is the docstring's first line, without its full stop
    assert fact.statement == "The error falls at order 4"
    assert fact.provenance["measured"] == "errors 1e-4, 6e-6: order 4.01"
    # a pass carries no reason, which lanky would print as thin evidence
    assert "reason" not in fact.provenance
    assert fact.term is None


def test_a_bare_true_is_a_verdict_with_nothing_measured() -> None:
    @tested
    def holds():
        """It holds."""
        return True

    (fact,) = holds.facts()
    assert fact.status is Status.TESTED
    assert fact.provenance["measured"] == ""


def test_a_claim_that_fails_is_refuted_with_what_it_measured() -> None:
    @tested
    def too_slow():
        """The error falls at order 4."""
        return False, "order 2.10"

    (fact,) = too_slow.facts()
    assert fact.status is Status.REFUTED
    assert fact.decided_by == "run"
    assert fact.provenance["reason"] == "order 2.10"


def test_a_bare_false_is_refuted_and_says_so() -> None:
    @tested
    def fails():
        """It holds."""
        return False

    (fact,) = fails.facts()
    assert fact.status is Status.REFUTED
    assert fact.provenance["reason"] == "the run returned False"


def test_a_claim_that_raises_is_refuted_with_the_error() -> None:
    @tested
    def broken():
        """It holds."""
        raise ZeroDivisionError("division by zero")

    (fact,) = broken.facts()
    assert fact.status is Status.REFUTED
    assert fact.provenance["reason"] == (
        "the run raised ZeroDivisionError: division by zero"
    )


@pytest.mark.parametrize("answer", [1, np.float64(3.5), "yes", None, (1, "x")])
def test_an_answer_that_is_not_a_verdict_is_refuted_not_read_as_true(answer) -> None:
    # A number or a string returned by mistake would read as true.
    def answers():
        """It holds."""
        return answer

    (fact,) = Tested(answers).facts()
    assert fact.status is Status.REFUTED
    assert "not a bool" in fact.provenance["reason"]


def test_a_numpy_truth_value_is_a_verdict() -> None:
    # ``order > 3.4`` of a numpy float is one.
    def compared():
        """It holds."""
        return np.float64(3.5) > 3.4, "order 3.50"

    (fact,) = Tested(compared).facts()
    assert fact.status is Status.TESTED


def test_the_claim_runs_once_for_its_row_and_again_when_called() -> None:
    runs = []

    @tested
    def counted():
        """It holds."""
        runs.append(1)
        return True, "once"

    counted.facts()
    counted.facts()
    assert len(runs) == 1
    assert counted() == (True, "once")
    assert len(runs) == 2


def test_a_claim_needs_a_docstring_and_a_function() -> None:
    def bare():
        return True

    with pytest.raises(TypeError, match="its first line is the claim"):
        tested(bare)
    with pytest.raises(TypeError, match="decorates a function"):
        tested(3)


def test_lanky_check_lists_the_claims_of_a_file(tmp_path: Path) -> None:
    from lanky.check import check_path

    path = tmp_path / "claims.py"
    path.write_text(
        "from loopty import tested\n"
        "\n"
        "\n"
        "@tested\n"
        "def converges():\n"
        '    """The error falls at order 4."""\n'
        '    return True, "order 4.01"\n'
        "\n"
        "\n"
        "@tested\n"
        "def diverges():\n"
        '    """The error falls at order 9."""\n'
        '    return False, "order 4.01"\n'
    )
    ledger = check_path(path)
    rows = {fact.owner: fact for fact in ledger if fact.kind == "tested"}
    assert set(rows) == {"converges", "diverges"}
    assert rows["converges"].status is Status.TESTED
    assert rows["converges"].where == "claims.py:4"
    assert rows["diverges"].status is Status.REFUTED
    assert rows["diverges"].provenance["reason"] == "order 4.01"
