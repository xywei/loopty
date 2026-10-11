"""Agreement with the code a kernel mirrors: ``@upstream(kernel)``, one row each.

The fact compares the kernel's native run on its file's example inputs with
what the decorated function says the code it mirrors computes there, at the
accuracy the kernel's types claim. It is ``tested`` on agreement, ``refuted``
with the cell that differs, and ``assumed``, with why, where the function
cannot run: its package or its recording is not there.
"""

from __future__ import annotations

import importlib
from pathlib import Path

import pytest
from lanky.check import check_path
from lanky.ledger import Status

from loopty import kernel, upstream
from loopty.upstream import KIND, Upstream

K = importlib.import_module("kernels.upstream")


def _fact(reference: Upstream):
    (fact,) = reference.facts()
    return fact


def test_a_function_that_computes_what_the_kernel_does_is_tested() -> None:
    fact = _fact(K.numpy_double)
    assert (fact.kind, fact.status, fact.decided_by) == (
        KIND,
        Status.TESTED,
        "upstream",
    )
    assert fact.owner == "double"
    assert fact.statement == (
        "the native run of double agrees with numpy_double in y, on its example "
        "inputs"
    )
    assert fact.provenance["uncompared"] == []
    detail = fact.provenance["outputs"]["y"]
    assert detail["agree"] and detail["exactness"] == "approx"
    assert detail["difference"] == 0.0
    assert fact.provenance["description"] == (
        "numpy's doubling, which the kernel mirrors."
    )


def test_the_function_is_given_copies_of_the_inputs() -> None:
    # it zeroes x after computing from it; the kernel's run reads its own x
    assert _fact(K.overwrites_its_input).status is Status.TESTED


def test_a_function_that_differs_refutes_it_with_the_cell() -> None:
    fact = _fact(K.off_at_the_end)
    assert fact.status is Status.REFUTED
    assert fact.decided_by == "upstream"
    assert fact.provenance["reason"] == (
        "y: cell 4 is 2.0 natively and 3.0 from off_at_the_end, beyond the approx "
        "allowance"
    )


def test_a_function_that_cannot_run_here_leaves_it_assumed() -> None:
    missing = _fact(K.not_installed)
    assert missing.status is Status.ASSUMED
    assert missing.decided_by is None
    assert missing.provenance["reason"].startswith(
        "not_installed could not run here, so the agreement was not checked: "
        "ModuleNotFoundError: No module named 'a_package_nobody_installs'"
    )
    unrecorded = _fact(K.not_recorded)
    assert unrecorded.status is Status.ASSUMED
    assert "FileNotFoundError" in unrecorded.provenance["reason"]


def test_the_claim_names_the_outputs_compared_and_no_others() -> None:
    # the squares are the kernel's too, and the function gives none: the row
    # does not claim them, and says they were not compared
    fact = _fact(K.doubles_only)
    assert fact.status is Status.TESTED
    assert fact.statement == (
        "the native run of double_and_square agrees with doubles_only in y, on its "
        "example inputs"
    )
    assert list(fact.provenance["outputs"]) == ["y"]
    assert fact.provenance["uncompared"] == ["z"]


def test_an_output_the_kernel_does_not_write_is_no_agreement() -> None:
    fact = _fact(K.about_its_input)
    assert fact.status is Status.ASSUMED
    assert fact.provenance["reason"] == (
        "about_its_input returned 'x', which double does not write"
    )


def test_the_rows_are_in_the_ledger_under_the_kernel() -> None:
    ledger = check_path(Path(K.__file__))
    rows = {
        fact.provenance["upstream"]: fact
        for fact in ledger
        if fact.kind == KIND
    }
    assert {name: fact.status for name, fact in rows.items()} == {
        "numpy_double": Status.TESTED,
        "overwrites_its_input": Status.TESTED,
        "off_at_the_end": Status.REFUTED,
        "not_installed": Status.ASSUMED,
        "not_recorded": Status.ASSUMED,
        "about_its_input": Status.ASSUMED,
        "doubles_only": Status.TESTED,
    }
    assert {fact.owner for fact in rows.values()} == {"double", "double_and_square"}


def test_a_file_without_example_inputs_leaves_it_assumed(tmp_path) -> None:
    source = tmp_path / "bare.py"
    source.write_text(
        "from __future__ import annotations\n"
        "from lanky.prelude import Fin, Real\n"
        "from loopty import Arr, kernel, upstream\n"
        "@kernel\n"
        "def copy(x: Arr[Fin[n], Real], y: Arr[Fin[n], Real]):\n"
        "    for i in y.dom:\n"
        "        y[i] = x[i]\n"
        "@upstream(copy)\n"
        "def numpy_copy(x, y):\n"
        "    return {'y': x.numpy()}\n"
    )
    ledger = check_path(source)
    (row,) = [fact for fact in ledger if fact.kind == KIND]
    assert row.status is Status.ASSUMED
    assert row.provenance["reason"] == (
        "the file has no example_inputs() to run copy and numpy_copy on"
    )


def test_upstream_takes_a_kernel() -> None:
    with pytest.raises(TypeError, match=r"@upstream\(l2p\)"):
        upstream(len)(lambda: None)

    @kernel
    def fill(y):
        pass

    assert isinstance(upstream(fill)(lambda y: {"y": 0}), Upstream)
