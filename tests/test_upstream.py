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


def test_a_claim_states_the_allowance_it_was_checked_to() -> None:
    # round-off, which the claim's 1e-12 of the largest value allows
    fact = _fact(K.rounded)
    assert fact.status is Status.TESTED
    assert fact.statement == (
        "the native run of double agrees with rounded in y within 1e-12 of the "
        "largest value, on its example inputs"
    )
    detail = fact.provenance["outputs"]["y"]
    assert detail["within"] == 1e-12
    assert detail["exactness"] == "approx"
    assert detail["tolerance"] == 1e-12 * detail["largest"]
    assert 0 < detail["difference"] <= detail["tolerance"]


def test_a_difference_inside_the_class_and_beyond_the_claim_refutes_it() -> None:
    # a billionth is inside approx's millionth, and beyond what the row says
    fact = _fact(K.off_by_a_billionth)
    assert fact.status is Status.REFUTED
    assert fact.provenance["reason"].startswith("y: cell 4 is 2.0 natively and ")
    assert fact.provenance["reason"].endswith(
        "from off_by_a_billionth, beyond 1e-12 of the largest value"
    )


def test_a_complex_nan_matches_only_part_by_part() -> None:
    """``nan + 1j`` and ``nan + 2j`` differ in a part both have (Codex)."""
    import numpy as np

    from loopty.upstream import _beyond

    got = np.array([complex(np.nan, 1.0), complex(np.nan, 1.0), 1.0 + 1j])
    want = np.array([complex(np.nan, 2.0), complex(np.nan, 1.0), 1.0 + 1j])
    assert _beyond(got, want, 1e-12).tolist() == [True, False, False]


@pytest.mark.parametrize(
    ("within", "error", "message"),
    [
        ("tight", TypeError, r"within=\.\.\.\) takes a number"),
        (True, TypeError, r"takes a number"),
        (0.0, ValueError, r"allows nothing a run can meet"),
        (-1e-12, ValueError, r"allows nothing a run can meet"),
        (float("nan"), ValueError, r"allows nothing a run can meet"),
    ],
)
def test_an_allowance_is_a_positive_number(within, error, message) -> None:
    with pytest.raises(error, match=message):
        upstream(K.double, within=within)


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
        "rounded": Status.TESTED,
        "off_by_a_billionth": Status.REFUTED,
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


def test_an_output_of_another_shape_is_refuted_with_both_shapes(tmp_path) -> None:
    # (2, 3) returned as (6,): the cells agree flattened, the array is not
    # the kernel's output (Codex on #248)
    source = tmp_path / "shaped.py"
    source.write_text(
        "from __future__ import annotations\n"
        "import numpy as np\n"
        "from lanky.prelude import Fin, Real\n"
        "from loopty import Arr, kernel, upstream\n"
        "@kernel\n"
        "def copy(x: Arr[Fin[m], Fin[n], Real], y: Arr[Fin[m], Fin[n], Real]):\n"
        "    for i in y.dom:\n"
        "        for j in y.dom[i]:\n"
        "            y[i, j] = x[i, j]\n"
        "def example_inputs():\n"
        "    x = np.arange(6.0).reshape(2, 3)\n"
        "    return {'copy': {'x': Arr.from_numpy(x), 'y': Arr.zeros((2, 3))}}\n"
        "@upstream(copy)\n"
        "def flattened(x, y):\n"
        "    return {'y': x.numpy().reshape(-1)}\n"
        "@upstream(copy)\n"
        "def shaped(x, y):\n"
        "    return {'y': x.numpy().copy()}\n"
    )
    ledger = check_path(source)
    rows = {fact.provenance["upstream"]: fact for fact in ledger if fact.kind == KIND}
    assert rows["shaped"].status is Status.TESTED
    flat = rows["flattened"]
    assert flat.status is Status.REFUTED
    assert flat.provenance["outputs"]["y"]["shapes"] == [[2, 3], [6]]
    assert flat.provenance["reason"] == (
        "y: [2, 3] cells natively, [6] from flattened"
    )


def test_an_output_of_another_length_is_refuted_within_an_allowance(tmp_path) -> None:
    # within= compares cell by cell, and 5 cells against 4 is no comparison:
    # refuted with both shapes, not a broadcasting error (Codex)
    source = tmp_path / "short.py"
    source.write_text(
        "from __future__ import annotations\n"
        "import numpy as np\n"
        "from lanky.prelude import Fin, Real\n"
        "from loopty import Arr, kernel, upstream\n"
        "@kernel\n"
        "def copy(x: Arr[Fin[m], Real], y: Arr[Fin[m], Real]):\n"
        "    for i in y.dom:\n"
        "        y[i] = x[i]\n"
        "def example_inputs():\n"
        "    x = Arr.from_numpy(np.arange(5.0))\n"
        "    return {'copy': {'x': x, 'y': Arr.zeros((5,))}}\n"
        "@upstream(copy, within=1e-12)\n"
        "def short(x, y):\n"
        "    return {'y': x.numpy()[:4]}\n"
    )
    ledger = check_path(source)
    (row,) = [fact for fact in ledger if fact.kind == KIND]
    assert row.status is Status.REFUTED
    assert row.provenance["outputs"]["y"]["shapes"] == [[5], [4]]
    assert row.provenance["reason"] == "y: [5] cells natively, [4] from short"


def test_upstream_takes_a_kernel() -> None:
    with pytest.raises(TypeError, match=r"@upstream\(l2p\)"):
        upstream(len)(lambda: None)

    @kernel
    def fill(y):
        pass

    assert isinstance(upstream(fill)(lambda y: {"y": 0}), Upstream)


#: A kernel in a file of its own, whose example inputs are ones.
DOUBLING = (
    "from __future__ import annotations\n"
    "import numpy as np\n"
    "from lanky.prelude import Fin, Real\n"
    "from loopty import Arr, kernel\n"
    "@kernel\n"
    "def double(x: Arr[Fin[n], Real], y: Arr[Fin[n], Real]):\n"
    "    for i in y.dom:\n"
    "        y[i] = 2 * x[i]\n"
    "def example_inputs():\n"
    "    return {'double': {'x': Arr.from_numpy(np.ones(4)), 'y': Arr.zeros((4,))}}\n"
)

#: A comparison in a file of its own: what was recorded on 0, 1, 2, 3.
COMPARED = (
    "from __future__ import annotations\n"
    "import numpy as np\n"
    "from loopty import Arr, upstream\n"
    "from {kernel} import double\n"
    "{inputs}"
    "@upstream(double)\n"
    "def recorded(x, y):\n"
    "    return {{'y': [0.0, 2.0, 4.0, 6.0]}}\n"
)

#: The comparison file's own inputs, which the recording is of.
COUNTING = (
    "def example_inputs():\n"
    "    x = Arr.from_numpy(np.arange(4.0))\n"
    "    return {'double': {'x': x, 'y': Arr.zeros((4,))}}\n"
)


def test_a_comparison_in_a_file_of_its_own_runs_on_that_files_inputs(
    tmp_path, monkeypatch
) -> None:
    # #262: the inputs were the kernel's module's, so a comparison declared
    # beside an import of the kernel compared on inputs it did not mean, or
    # read assumed where the kernel's file has none
    monkeypatch.syspath_prepend(str(tmp_path))
    (tmp_path / "doubling_262.py").write_text(DOUBLING)
    (tmp_path / "bare_262.py").write_text(
        DOUBLING.split("def example_inputs")[0].replace("double", "twice")
    )
    own = tmp_path / "doubling_vs_262.py"
    own.write_text(COMPARED.format(kernel="doubling_262", inputs=COUNTING))
    (row,) = [fact for fact in check_path(own) if fact.kind == KIND]
    assert row.status is Status.TESTED, row.provenance
    assert row.provenance["inputs"] == "example_inputs() of doubling_vs_262.py"
    # a kernel's file without inputs: the comparison's own are enough
    alone = tmp_path / "bare_vs_262.py"
    alone.write_text(
        COMPARED.format(kernel="bare_262", inputs=COUNTING).replace("double", "twice")
    )
    (row,) = [fact for fact in check_path(alone) if fact.kind == KIND]
    assert row.status is Status.TESTED, row.provenance
    # without its own, the comparison runs on the kernel's file's, and says so
    theirs = tmp_path / "doubling_theirs_262.py"
    theirs.write_text(COMPARED.format(kernel="doubling_262", inputs=""))
    (row,) = [fact for fact in check_path(theirs) if fact.kind == KIND]
    assert row.status is Status.REFUTED
    assert row.provenance["inputs"] == "example_inputs() of doubling_262.py"
    assert row.provenance["reason"].startswith("y: cell 0 is 2.0 natively and 0.0 ")
    # neither has any
    neither = tmp_path / "bare_theirs_262.py"
    neither.write_text(
        COMPARED.format(kernel="bare_262", inputs="").replace("double", "twice")
    )
    (row,) = [fact for fact in check_path(neither) if fact.kind == KIND]
    assert row.status is Status.ASSUMED
    assert row.provenance["reason"] == (
        "neither bare_theirs_262.py nor bare_262.py has an example_inputs() to run "
        "twice and recorded on"
    )
