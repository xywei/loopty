"""A ragged array is read through the layout its kernel declares, however it runs.

The lowered kernel is handed a ragged array's counts and offsets as arguments,
runs a loop over ``val.dom[r]`` to the ``cnt[r]`` it reads once per row, and
indexes ``val[off[r] + j]`` with ``off`` as the kernel has left it. The native
run used to follow the runtime array's own offsets instead, so a kernel that
writes its counts or its offsets had two meanings, and only the differential
run noticed. Now the native run and the term interpreter read the declared
arrays too, and the contract checks, on entry, that they are the array's own
layout.
"""

from __future__ import annotations

import numpy as np
import pytest
from lanky.prelude import Nat, Real
from lanky.terms import evaluate_annotations

from loopty import Arr, Fin, reduce_sum, when
from loopty.faithful import KIND, SAMPLES, sample_arguments
from loopty.interpret import interpret
from loopty.kernel import Kernel
from loopty.trace import mask_writes, trace


def row_sums_then_next_offset(
    ends: Arr[Fin[n], Nat],  # noqa: F821
    cnt: Arr[Fin[n], Nat],  # noqa: F821
    off: Arr[Fin[n + 1], Nat],  # noqa: F821
    val: Arr[Fin[n], Fin[cnt], Real],  # noqa: F821
    y: Arr[Fin[n], Real],  # noqa: F821
):
    """Sum row ``r``, then store where row ``r + 1`` starts."""
    for r in y.dom:
        y[r] = reduce_sum(val[r, j] for j in val.dom[r])
        off[r + 1] = ends[r]


def row_sums_then_next_count(
    cnt: Arr[Fin[n], Nat],  # noqa: F821
    val: Arr[Fin[n], Fin[cnt], Real],  # noqa: F821
    y: Arr[Fin[n], Real],  # noqa: F821
):
    """Sum row ``r``, then clear the length of row ``r + 1``."""
    for r in y.dom:
        y[r] = reduce_sum(val[r, j] for j in val.dom[r])
        with when(r + 1 < y.dom.size):
            cnt[r + 1] = 0


def row_sums_through_offsets(
    cnt: Arr[Fin[n], Nat],  # noqa: F821
    off: Arr[Fin[n + 1], Nat],  # noqa: F821
    val: Arr[Fin[n], Fin[cnt], Real],  # noqa: F821
    y: Arr[Fin[n], Real],  # noqa: F821
):
    for r in y.dom:
        y[r] = reduce_sum(val[r, j] for j in val.dom[r])


COUNTS = [2, 1, 3]


def issue_input(ends: list[int]) -> dict:
    """The input of the issue: counts [2, 1, 3], values 1 to 6, CSR offsets."""
    return {
        "ends": Arr.from_numpy(np.array(ends, dtype=np.int64)),
        "cnt": Arr.from_numpy(np.array(COUNTS, dtype=np.int64)),
        "off": Arr.from_numpy(np.array([0, 2, 3, 6], dtype=np.int64)),
        "val": Arr.ragged(COUNTS, values=np.arange(1.0, 7.0)),
        "y": Arr.zeros(3),
    }


def term_of(fn):
    return trace(fn, evaluate_annotations(fn))


# {{{ the view


def test_a_view_reads_rows_through_the_arrays_it_is_given() -> None:
    val = Arr.ragged(COUNTS, values=np.arange(1.0, 7.0))
    cnt = np.array(COUNTS)
    off = np.array([0, 2, 3, 6])
    view = val.through(cnt, off)
    assert view.numpy() is val.numpy()
    assert val.layout is None
    counts, offsets = view.layout
    assert counts is cnt and offsets is off
    assert [view[1, j] for j in view.dom[1]] == [3.0]

    # Read as the arrays are now, not as they were when the view was made.
    off[1] = 1
    cnt[1] = 2
    assert list(view.dom[1]) == [0, 1]
    assert [view[1, j] for j in view.dom[1]] == [2.0, 3.0]
    assert list(view[1]) == [2.0, 3.0]
    view[1, 1] = 30.0
    assert val.numpy()[2] == 30.0
    # The array's own layout is untouched.
    assert list(val.dom[1]) == [0]


def test_a_view_without_offsets_starts_rows_where_the_array_does() -> None:
    val = Arr.ragged(COUNTS, values=np.arange(1.0, 7.0))
    cnt = np.array([2, 0, 3])
    view = val.through(cnt, None)
    assert list(view.dom[1]) == []
    assert [view[2, j] for j in view.dom[2]] == [4.0, 5.0, 6.0]
    # A negative count is a loop with no iterations, as the compiled one is.
    cnt[2] = -1
    assert len(view.dom[2]) == 0


def test_a_view_refuses_a_cell_its_layout_puts_outside_the_buffer() -> None:
    val = Arr.ragged(COUNTS, values=np.arange(1.0, 7.0))
    off = np.array([0, 2, 5, 6])
    view = val.through(np.array(COUNTS), off)
    assert view[2, 0] == 6.0
    with pytest.raises(IndexError, match="cell 6 of the flat buffer"):
        view[2, 1]
    with pytest.raises(IndexError, match="column 1 out of range for row 1"):
        view[1, 1]
    with pytest.raises(IndexError, match="row 2 occupies cells 5 to 8"):
        view[2]


def test_a_dense_array_has_no_layout_to_read_through() -> None:
    with pytest.raises(TypeError, match="no layout"):
        Arr.zeros(3).through(np.zeros(3), None)


def test_the_masking_view_keeps_the_layout() -> None:
    val = Arr.ragged(COUNTS, values=np.arange(1.0, 7.0))
    off = np.array([0, 2, 3, 6])
    masked = mask_writes(val.through(None, off))
    assert masked.layout[1] is off
    off[1] = 1
    assert masked[1, 0] == 2.0


def test_the_declared_layout_names_the_counts_and_the_offsets() -> None:
    from loopty.term import declared_layout

    term = term_of(row_sums_then_next_offset)
    assert declared_layout(term.params) == {"val": ("cnt", "off")}
    assert declared_layout(term_of(row_sums_then_next_count).params) == {
        "val": ("cnt", None)
    }


# }}}


# {{{ the native run and the interpreter


def test_the_native_run_indexes_through_the_offsets_the_kernel_writes() -> None:
    # The issue's numbers. With ``ends`` the offsets the array already has,
    # nothing changes; with ``ends = [1, 1, 1]`` row ``r + 1`` starts at 1
    # once ``r`` is summed, which the lowered kernel always read and the
    # native run used to ignore, answering [3, 3, 15].
    arguments = issue_input([2, 3, 6])
    Kernel(row_sums_then_next_offset)(**arguments)
    assert arguments["y"].numpy().tolist() == [3.0, 3.0, 15.0]

    arguments = issue_input([1, 1, 1])
    Kernel(row_sums_then_next_offset)(**arguments)
    assert arguments["y"].numpy().tolist() == [3.0, 2.0, 9.0]
    assert arguments["off"].numpy().tolist() == [0, 1, 1, 1]


def test_the_native_run_bounds_a_row_by_the_counts_the_kernel_writes() -> None:
    arguments = issue_input([2, 3, 6])
    del arguments["ends"], arguments["off"]
    Kernel(row_sums_then_next_count)(**arguments)
    assert arguments["y"].numpy().tolist() == [3.0, 0.0, 0.0]


def test_the_interpreter_indexes_through_the_offsets_the_term_writes() -> None:
    arguments = issue_input([1, 1, 1])
    out = interpret(term_of(row_sums_then_next_offset), arguments)
    assert out["y"].tolist() == [3.0, 2.0, 9.0]


def test_a_native_call_with_offsets_that_are_not_the_array_s_own_is_refused() -> None:
    # The two runs read through ``off``, and they start from the same layout
    # only if ``off`` is ``val``'s. The compiled run always refused this.
    arguments = issue_input([2, 3, 6])
    del arguments["ends"]
    arguments["off"] = Arr.from_numpy(np.array([0, 1, 3, 6]))
    with pytest.raises(ValueError, match="offsets argument off"):
        Kernel(row_sums_through_offsets)(**arguments)


# }}}


# {{{ the two runs agree


@pytest.mark.parametrize(
    ("fn", "ends"),
    [
        (row_sums_then_next_offset, [1, 1, 1]),
        (row_sums_then_next_offset, [2, 3, 6]),
        (row_sums_then_next_count, None),
    ],
)
def test_the_compiled_and_the_native_run_give_one_meaning(fn, ends) -> None:
    pytest.importorskip("loopy")
    from lanky.ledger import Status

    from loopty.executor import LoopyExecutor

    arguments = issue_input(ends or [2, 3, 6])
    if fn is row_sums_then_next_count:
        del arguments["ends"], arguments["off"]
    kernel = Kernel(fn)
    try:
        fact = LoopyExecutor().differential(kernel, kernel, arguments)
    except Exception as exc:  # pragma: no cover - depends on the local toolchain
        if "compil" in str(exc).lower() or isinstance(exc, OSError):
            pytest.skip(f"the C toolchain path is unusable here: {exc}")
        raise
    assert fact.status is Status.TESTED, fact.provenance


def sums_then_count_in_a_loop_of_the_row(
    x: Arr[Fin[m], Real],  # noqa: F821
    cnt: Arr[Fin[n], Nat],  # noqa: F821
    val: Arr[Fin[n], Fin[cnt], Real],  # noqa: F821
    y: Arr[Fin[n], Real],  # noqa: F821
):
    """Sum row ``r`` once per ``i``, and set its length to one after each sum."""
    for r in y.dom:
        for i in x.dom:
            y[r] = y[r] + x[i] + reduce_sum(val[r, j] for j in val.dom[r])
            cnt[r] = 1


def count_then_sums_in_a_loop_of_the_row(
    x: Arr[Fin[m], Real],  # noqa: F821
    cnt: Arr[Fin[n], Nat],  # noqa: F821
    val: Arr[Fin[n], Fin[cnt], Real],  # noqa: F821
    y: Arr[Fin[n], Real],  # noqa: F821
):
    """Set the length of row ``r`` to one, then sum it, once per ``i``."""
    for r in y.dom:
        for i in x.dom:
            cnt[r] = 1
            y[r] = y[r] + x[i] + reduce_sum(val[r, j] for j in val.dom[r])


def fiber_then_count_in_a_loop_of_the_row(
    x: Arr[Fin[m], Real],  # noqa: F821
    cnt: Arr[Fin[n], Nat],  # noqa: F821
    val: Arr[Fin[n], Fin[cnt], Real],  # noqa: F821
    y: Arr[Fin[n], Real],  # noqa: F821
):
    """A loop over the fiber of row ``r`` once per ``i``, then its length set."""
    for r in y.dom:
        for i in x.dom:
            for j in val.dom[r]:
                y[r] = y[r] + x[i] * val[r, j]
            cnt[r] = 1


def count_grown_inside_its_own_fiber(
    cnt: Arr[Fin[n], Nat],  # noqa: F821
    val: Arr[Fin[n], Fin[cnt], Real],  # noqa: F821
    y: Arr[Fin[n], Real],  # noqa: F821
):
    """The length of row ``r`` grown by one inside the loop that it bounds."""
    for r in y.dom:
        for j in val.dom[r]:
            y[r] = y[r] + val[r, j]
            cnt[r] = cnt[r] + 1


def row_loop_input() -> dict:
    return {
        "x": Arr.from_numpy(np.ones(2)),
        "cnt": Arr.from_numpy(np.array(COUNTS, dtype=np.int64)),
        "val": Arr.ragged(COUNTS, values=np.arange(1.0, 7.0)),
        "y": Arr.zeros(3),
    }


@pytest.mark.parametrize(
    ("fn", "user", "writer", "native"),
    [
        (sums_then_count_in_a_loop_of_the_row, "S0", "S1", [6.0, 8.0, 21.0]),
        (count_then_sums_in_a_loop_of_the_row, "S1", "S0", [4.0, 8.0, 10.0]),
        (fiber_then_count_in_a_loop_of_the_row, "S0", "S1", [4.0, 6.0, 19.0]),
    ],
)
def test_a_length_rewritten_in_a_loop_of_its_row_is_not_lowered(
    fn, user, writer, native
) -> None:
    # The body reads the length of row ``r`` where the loop over its fiber
    # starts, once per ``i``; the lowered kernel computed it once per row. So
    # from the second ``i`` on the native run summed one entry and the
    # compiled one the whole row: the first kernel's ``y`` came out 11 apart,
    # and only the differential run said so. The writer first in the body
    # left loopy with an order it could not schedule, and the loop over the
    # fiber inside ``i`` was refused for an unrelated reason (#53).
    pytest.importorskip("loopy")
    from loopty.lower import LoweringError, lower_generic

    arguments = row_loop_input()
    Kernel(fn)(**arguments)
    assert arguments["y"].numpy().tolist() == native
    assert arguments["cnt"].numpy().tolist() == [1, 1, 1]
    with pytest.raises(LoweringError) as caught:
        lower_generic(term_of(fn))
    message = str(caught.value)
    assert message.startswith(f"statement {user} is bounded by the row length ")
    assert "once per row, before the loop over i" in message
    assert f"{writer} rewrites cnt inside that loop" in message


def test_a_length_rewritten_inside_the_loop_it_bounds_is_read_once() -> None:
    # The loop over ``val.dom[r]`` reads its bound when it starts, natively as
    # compiled, so growing ``cnt[r]`` inside it adds no iteration to it.
    pytest.importorskip("loopy")
    from lanky.ledger import Status

    from loopty.executor import LoopyExecutor

    arguments = row_loop_input()
    del arguments["x"]
    kernel = Kernel(count_grown_inside_its_own_fiber)
    kernel(**arguments)
    assert arguments["y"].numpy().tolist() == [3.0, 3.0, 15.0]
    assert arguments["cnt"].numpy().tolist() == [4, 2, 6]
    arguments = row_loop_input()
    del arguments["x"]
    try:
        fact = LoopyExecutor().differential(kernel, kernel, arguments)
    except Exception as exc:  # pragma: no cover - depends on the local toolchain
        if "compil" in str(exc).lower() or isinstance(exc, OSError):
            pytest.skip(f"the C toolchain path is unusable here: {exc}")
        raise
    assert fact.status is Status.TESTED, fact.provenance


def test_samples_pass_the_offsets_of_the_ragged_array_they_draw() -> None:
    # A call has to pass them so, and a native run that refuses a sample says
    # nothing about the term.
    term = term_of(row_sums_then_next_offset)
    for seed in range(4):
        arguments, _sizes = sample_arguments(term, seed)
        assert arguments["off"].numpy().tolist() == arguments["val"].offsets.tolist()


def test_the_faithfulness_fact_runs_the_samples_through_the_layout() -> None:
    # No sample is refused by the contract. One whose ``ends`` send a row past
    # the end of the buffer is skipped, as any input the body cannot run is.
    from lanky.ledger import Status

    fact = Kernel(row_sums_then_next_offset).facts()[-1]
    assert fact.kind == KIND
    assert fact.status is Status.TESTED, fact.provenance
    outcomes = [entry["outcome"] for entry in fact.provenance["inputs"]]
    assert len(outcomes) == SAMPLES
    assert not [outcome for outcome in outcomes if "ValueError" in outcome]
    assert fact.provenance["compared"] >= 1


# }}}
