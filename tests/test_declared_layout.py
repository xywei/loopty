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

import dataclasses

import islpy as isl
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


def test_a_row_off_the_buffer_names_the_array_it_is_read_as() -> None:
    # A read the layout puts off the buffer is a LayoutError, an IndexError
    # that says which parameter was read, so that a native run that raises it
    # refutes that family's layout fact (#103); a column past its row is not.
    from loopty.arr import LayoutError

    val = Arr.ragged(COUNTS, values=np.arange(1.0, 7.0))
    view = val.through(np.array(COUNTS), np.array([0, 2, 5, 6]), "val")
    with pytest.raises(LayoutError, match="cell 6 of the flat buffer") as caught:
        view[2, 1]
    assert caught.value.array == "val"
    with pytest.raises(LayoutError, match="row 2 occupies cells 5 to 8") as caught:
        view[2]
    assert caught.value.array == "val"
    with pytest.raises(IndexError) as caught:
        view[1, 1]
    assert not isinstance(caught.value, LayoutError)


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


def sums_then_next_count_in_a_loop_of_the_row(
    x: Arr[Fin[m], Real],  # noqa: F821
    cnt: Arr[Fin[n], Nat],  # noqa: F821
    val: Arr[Fin[n], Fin[cnt], Real],  # noqa: F821
    y: Arr[Fin[n], Real],  # noqa: F821
):
    """Sum row ``r`` once per ``i``, and clear the length of the next row."""
    for r in y.dom:
        for i in x.dom:
            y[r] = y[r] + x[i] + reduce_sum(val[r, j] for j in val.dom[r])
            with when(r + 1 < y.dom.size):
                cnt[r + 1] = 0


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
    assert f"{writer} rewrites that row's cnt inside that loop" in message


def test_the_next_row_s_length_rewritten_in_a_loop_of_the_row_is_lowered() -> None:
    # ``cnt[r + 1]`` is not the length of row ``r``, which stays what it was
    # for every ``i``, and both runs read the new one when row ``r + 1``
    # starts. Only a write that can reach the row's own cell is refused.
    pytest.importorskip("loopy")
    from lanky.ledger import Status

    from loopty.executor import LoopyExecutor

    kernel = Kernel(sums_then_next_count_in_a_loop_of_the_row)
    arguments = row_loop_input()
    kernel(**arguments)
    assert arguments["y"].numpy().tolist() == [8.0, 2.0, 2.0]
    try:
        fact = LoopyExecutor().differential(kernel, kernel, row_loop_input())
    except Exception as exc:  # pragma: no cover - depends on the local toolchain
        if "compil" in str(exc).lower() or isinstance(exc, OSError):
            pytest.skip(f"the C toolchain path is unusable here: {exc}")
        raise
    assert fact.status is Status.TESTED, fact.provenance


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


@pytest.mark.parametrize(
    ("fn", "native"),
    [
        (sums_then_count_in_a_loop_of_the_row, [6.0, 8.0, 21.0]),
        (count_then_sums_in_a_loop_of_the_row, [4.0, 8.0, 10.0]),
        (fiber_then_count_in_a_loop_of_the_row, [4.0, 6.0, 19.0]),
        (sums_then_next_count_in_a_loop_of_the_row, [8.0, 2.0, 2.0]),
        (count_grown_inside_its_own_fiber, [3.0, 3.0, 15.0]),
    ],
)
def test_the_interpreter_reads_a_written_bound_where_its_loop_starts(
    fn, native
) -> None:
    # The interpreter used to enumerate every statement's domain before it ran
    # anything, and refused these kernels (#52). The bound of a loop over a
    # fiber is now read when that loop starts, each time it starts, and a
    # reduction's when it is summed, which is what the body does.
    arguments = row_loop_input()
    if "x" not in term_of(fn).param_names:
        del arguments["x"]
    copies = {name: value.copy() for name, value in arguments.items()}
    Kernel(fn)(**arguments)
    assert arguments["y"].numpy().tolist() == native
    interpret(term_of(fn), copies)
    for name, value in arguments.items():
        assert copies[name].numpy().tobytes() == value.numpy().tobytes(), name


def pairs_in_a_row_recounted(
    cnt: Arr[Fin[n], Nat],  # noqa: F821
    val: Arr[Fin[n], Fin[cnt], Real],  # noqa: F821
    y: Arr[Fin[n], Real],  # noqa: F821
):
    """A fiber loop inside another over the same row, the row's length reset."""
    for r in y.dom:
        for j in val.dom[r]:
            for k in val.dom[r]:
                y[r] = y[r] + val[r, k]
            cnt[r] = 1


def test_a_written_bound_of_two_loops_of_one_statement_is_read_at_each() -> None:
    # Natively the loop over ``k`` reads the row's length each time it starts,
    # 2 and then 1 for row 0. ``nl_cnt_r`` was one parameter of the domain of
    # the statement, read once, so the interpreter refused the kernel and the
    # fact was left assumed (#87). The loop over ``k`` reads a copy of its own.
    from lanky.ledger import Status

    arguments = row_loop_input()
    del arguments["x"]
    copies = {name: value.copy() for name, value in arguments.items()}
    Kernel(pairs_in_a_row_recounted)(**arguments)
    assert arguments["y"].numpy().tolist() == [4.0, 3.0, 23.0]
    interpret(term_of(pairs_in_a_row_recounted), copies)
    for name, value in arguments.items():
        assert copies[name].numpy().tobytes() == value.numpy().tobytes(), name
    fact = Kernel(pairs_in_a_row_recounted).facts()[-1]
    assert fact.status is Status.TESTED, fact.provenance


def sum_of_x_in_a_fiber_recounted(
    x: Arr[Fin[m], Real],  # noqa: F821
    cnt: Arr[Fin[n], Nat],  # noqa: F821
    val: Arr[Fin[n], Fin[cnt], Real],  # noqa: F821
    y: Arr[Fin[n], Real],  # noqa: F821
):
    """A sum over ``x`` once per entry of row ``r``, the row's length reset."""
    for r in y.dom:
        for j in val.dom[r]:
            y[r] = y[r] + reduce_sum(x[i] for i in x.dom)
            cnt[r] = 1


def test_a_sum_in_a_fiber_keeps_the_length_its_loop_was_read_at() -> None:
    # The sum's domain repeats ``0 <= j < nl_cnt_r`` from the loop around it.
    # Read again where the sum starts, ``cnt[r]`` is 1 from ``j = 1`` on, so
    # the sum was empty there and the interpreter disagreed with the body
    # (y = [2, 2, 2]), which refuted a faithful trace.
    from lanky.ledger import Status

    arguments = row_loop_input()
    copies = {name: value.copy() for name, value in arguments.items()}
    Kernel(sum_of_x_in_a_fiber_recounted)(**arguments)
    assert arguments["y"].numpy().tolist() == [4.0, 2.0, 6.0]
    interpret(term_of(sum_of_x_in_a_fiber_recounted), copies)
    for name, value in arguments.items():
        assert copies[name].numpy().tobytes() == value.numpy().tobytes(), name
    fact = Kernel(sum_of_x_in_a_fiber_recounted).facts()[-1]
    assert fact.status is Status.TESTED, fact.provenance


def row_sum_in_its_own_fiber_recounted(
    cnt: Arr[Fin[n], Nat],  # noqa: F821
    val: Arr[Fin[n], Fin[cnt], Real],  # noqa: F821
    y: Arr[Fin[n], Real],  # noqa: F821
):
    """The row summed once per entry of the row, the row's length reset."""
    for r in y.dom:
        for j in val.dom[r]:
            y[r] = y[r] + reduce_sum(val[r, k] for k in val.dom[r])
            cnt[r] = 1


def nested_sums_in_a_fiber_recounted(
    cnt: Arr[Fin[n], Nat],  # noqa: F821
    val: Arr[Fin[n], Fin[cnt], Real],  # noqa: F821
    y: Arr[Fin[n], Real],  # noqa: F821
):
    """The row summed once per entry, once per entry, the row's length reset."""
    for r in y.dom:
        for j in val.dom[r]:
            y[r] = y[r] + reduce_sum(
                reduce_sum(val[r, k] for k in val.dom[r]) for i in val.dom[r]
            )
            cnt[r] = 1


@pytest.mark.parametrize(
    ("fn", "native"),
    [
        (row_sum_in_its_own_fiber_recounted, [4.0, 3.0, 23.0]),
        (nested_sums_in_a_fiber_recounted, [7.0, 3.0, 53.0]),
    ],
)
def test_a_written_bound_of_a_loop_and_a_sum_inside_it_is_read_at_each(
    fn, native
) -> None:
    # Natively the loop over ``j`` reads the row's length once, 2 for row 0,
    # and the sum over ``k`` reads it each time it starts, 2 and then 1. The
    # two shared the one parameter ``nl_cnt_r``, read once, so the interpreter
    # refused the kernel and the fact was left assumed (#87). The sums read a
    # copy of their own where each starts, and a sum inside a sum bounds the
    # outer binder by the same copy, read where the outer sum started.
    from lanky.ledger import Status

    arguments = row_loop_input()
    del arguments["x"]
    copies = {name: value.copy() for name, value in arguments.items()}
    Kernel(fn)(**arguments)
    assert arguments["y"].numpy().tolist() == native
    interpret(term_of(fn), copies)
    for name, value in arguments.items():
        assert copies[name].numpy().tobytes() == value.numpy().tobytes(), name
    fact = Kernel(fn).facts()[-1]
    assert fact.status is Status.TESTED, fact.provenance


def diagonal_recounted(
    cnt: Arr[Fin[n], Nat],  # noqa: F821
    val: Arr[Fin[n], Fin[cnt], Real],  # noqa: F821
    y: Arr[Fin[n], Real],  # noqa: F821
):
    """The kernel of #111: the guard makes the bound of ``k`` follow from ``j``'s."""
    for r in y.dom:
        for j in val.dom[r]:
            for k in val.dom[r]:
                with when(k == j):
                    y[r] = y[r] + val[r, k]
            cnt[r] = 1


def lower_triangle_recounted(
    cnt: Arr[Fin[n], Nat],  # noqa: F821
    val: Arr[Fin[n], Fin[cnt], Real],  # noqa: F821
    y: Arr[Fin[n], Real],  # noqa: F821
):
    """``k <= j``, which leaves the bound of ``k`` in the domain."""
    for r in y.dom:
        for j in val.dom[r]:
            for k in val.dom[r]:
                with when(k <= j):
                    y[r] = y[r] + val[r, k]
            cnt[r] = 1


def next_entry_recounted(
    cnt: Arr[Fin[n], Nat],  # noqa: F821
    val: Arr[Fin[n], Fin[cnt], Real],  # noqa: F821
    y: Arr[Fin[n], Real],  # noqa: F821
):
    """The bound of ``k`` read after it grew, the domain stated by ``j``'s."""
    for r in y.dom:
        for j in val.dom[r]:
            cnt[r] = 3
            for k in val.dom[r]:
                with when(k == j + 1):
                    y[r] = y[r] + val[r, k]


def diagonal_sum_recounted(
    cnt: Arr[Fin[n], Nat],  # noqa: F821
    val: Arr[Fin[n], Fin[cnt], Real],  # noqa: F821
    y: Arr[Fin[n], Real],  # noqa: F821
):
    """The sum's clause makes its bound follow from the loop's."""
    for r in y.dom:
        for j in val.dom[r]:
            y[r] = y[r] + reduce_sum(val[r, t] for t in val.dom[r] if t == j)
            cnt[r] = 1


@pytest.mark.parametrize(
    ("fn", "native", "tested"),
    [
        (diagonal_recounted, [1.0, 3.0, 4.0], True),
        (lower_triangle_recounted, [2.0, 3.0, 12.0], True),
        (next_entry_recounted, [5.0, 4.0, 11.0], False),
        (diagonal_sum_recounted, [1.0, 3.0, 4.0], True),
    ],
)
def test_a_written_bound_a_guard_makes_redundant_is_read_at_its_loop(
    fn, native, tested
) -> None:
    # isl states the domain of the first kernel as k = j, 0 <= j < nl_cnt_r:
    # the bound of the loop over ``k`` follows from the one over ``j`` and is
    # gone, so the loop over ``k`` had no reading of its own. The interpreter
    # ran ``k = 1`` past row 0's new length and raised, which refuted the
    # faithful trace by an exception rather than a comparison; with ``k <=
    # j`` isl kept the bound, and the same kernel was tested (#111). The
    # bounds are read apart over the loop nest before the guard narrowed it,
    # and the sum's over its domain before its clause did. The third kernel's
    # drawn inputs make its body read past a row (its count grows), so its
    # fact compares nothing, and only the run here says the two agree.
    from lanky.ledger import Status

    arguments = row_loop_input()
    del arguments["x"]
    copies = {name: value.copy() for name, value in arguments.items()}
    Kernel(fn)(**arguments)
    assert arguments["y"].numpy().tolist() == native
    interpret(term_of(fn), copies)
    for name, value in arguments.items():
        assert copies[name].numpy().tobytes() == value.numpy().tobytes(), name
    fact = Kernel(fn).facts()[-1]
    assert fact.kind == KIND
    if tested:
        assert fact.status is Status.TESTED, fact.provenance
    else:
        assert fact.status is Status.ASSUMED, fact.provenance
        assert "no input ran natively" in fact.provenance["reason"]


@pytest.mark.parametrize(
    "fn",
    [
        diagonal_recounted,
        lower_triangle_recounted,
        diagonal_sum_recounted,
        pairs_in_a_row_recounted,
    ],
)
def test_a_length_rewritten_around_an_inner_loop_it_bounds_is_not_lowered(fn) -> None:
    # The lowered kernel computes the length of row ``r`` once per row; the
    # body reads it where each loop over the row starts, and the loop over
    # ``k`` starts once per ``j``, after ``cnt[r] = 1``. Only the first loop
    # the length bounds was counted, so these were lowered and their compiled
    # runs disagreed with the native ones ([3, 3, 15] for the first, against
    # [1, 3, 4]); and for the first, the loop over ``k`` was not in the
    # narrowed domain at all (#111).
    pytest.importorskip("loopy")
    from loopty.lower import LoweringError, lower_generic

    with pytest.raises(LoweringError) as caught:
        lower_generic(term_of(fn))
    message = str(caught.value)
    assert message.startswith("statement S0 is bounded by the row length ")
    assert "once per row, before the loop over j" in message
    assert "S1 rewrites that row's cnt inside that loop" in message


def test_the_faithfulness_fact_of_a_kernel_writing_its_counts_is_tested() -> None:
    # The issue's kernel: the next row's length cleared after each row is
    # summed. Its differential run was tested and its faithfulness fact was
    # left assumed, because the interpreter refused the bound (#52).
    from lanky.ledger import Status

    fact = Kernel(row_sums_then_next_count).facts()[-1]
    assert fact.kind == KIND
    assert fact.status is Status.TESTED, fact.provenance
    assert fact.provenance["compared"] == SAMPLES
    arguments = issue_input([2, 3, 6])
    del arguments["ends"], arguments["off"]
    out = interpret(term_of(row_sums_then_next_count), arguments)
    assert out["y"].tolist() == [3.0, 0.0, 0.0]
    assert out["cnt"].tolist() == [2, 0, 0]


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


# {{{ the layout fact (#51)


def overlap_then_write(
    cnt: Arr[Fin[n], Nat],  # noqa: F821
    off: Arr[Fin[n + 1], Nat],  # noqa: F821
    val: Arr[Fin[n], Fin[cnt], Real],  # noqa: F821
):
    """Every row moved to start at cell 0, then written: the issue's kernel."""
    for r in cnt.dom:
        off[r] = 0
        for j in val.dom[r]:
            val[r, j] = r + 1.0


def shortened_in_its_own_loop(
    cnt: Arr[Fin[n], Nat],  # noqa: F821
    val: Arr[Fin[n], Fin[cnt], Real],  # noqa: F821
    y: Arr[Fin[n], Real],  # noqa: F821
):
    """A row's length set to zero inside the loop it bounds."""
    for r in y.dom:
        for j in val.dom[r]:
            y[r] = y[r] + val[r, j]
            cnt[r] = 0


def layout_of(fn):
    """The term of ``fn``, its facts settled by isl, and its layout facts."""
    from lanky.ledger import Ledger

    from loopty import typing as rules
    from loopty.oracle import IslOracle

    term = term_of(fn)
    oracle = IslOracle()
    facts = [
        (oracle.establish(fact) or fact) if oracle.can_establish(fact) else fact
        for fact in rules.facts_for(term, owner=term.name, where="test.py:1")
    ]
    layouts = [fact for fact in facts if fact.kind == "layout"]
    return term, Ledger(facts), layouts


def test_rows_moved_onto_each_other_refute_the_layout_fact() -> None:
    # The disjoint writes of ``val`` were decided over ``[r, j]``, and a
    # parallel ``r`` accepted, while every row starts at cell 0 once the
    # kernel has run its first statement (#51). The layout fact they rest on
    # was assumed whatever the kernel wrote; a start that reads nothing is
    # now asked of isl, and 0 at row 1 is one some counts contradict (#86).
    from lanky.ledger import Status

    from loopty.schedule import Schedule

    term, ledger, (layout,) = layout_of(overlap_then_write)
    assert layout.id == "layout:overlap_then_write:cnt"
    assert layout.status is Status.REFUTED, layout.provenance
    assert layout.decided_by == "isl"
    assert layout.statement == (
        "the rows of val stay inside their buffers and apart while S0 write off"
    )
    assert layout.provenance["written"] == ["off"]
    assert layout.provenance["fixed"] == ["S0"]
    assert layout.provenance["reason"].startswith(
        "[r=1] is one of the instances of S0 that write into off the start of "
        "a row which the counts in cnt can put elsewhere"
    )
    assert layout.provenance["reason"].endswith("at [n=2]")
    (disjoint,) = [
        fact
        for fact in ledger
        if fact.kind == "disjoint-writes" and fact.provenance["array"] == "val"
    ]
    assert disjoint.status is Status.DECIDED
    assert disjoint.rests_on == (layout.id,)
    assert ledger.support(disjoint).effective is Status.REFUTED
    (access,) = [
        fact for fact in ledger if fact.id.endswith(":S1:write:val[r, j]")
    ]
    assert access.status is Status.DECIDED
    assert ledger.support(access).under == (layout.id,)
    # The facts about the dense arrays, the layout's own reads among them,
    # rest on nothing.
    assert {fact.id for fact in ledger if fact.rests_on} == {disjoint.id, access.id}
    # A cast is decided against the dependences of ``val`` over ``[r, j]``.
    monotone = [
        fact
        for fact in Schedule(term).split("r", 2).facts()
        if fact.kind == "monotone"
    ]
    assert [fact.rests_on for fact in monotone] == [(layout.id,)]
    # The witness, run: with one entry in rows 0 and 1, both rows are cell 0,
    # and row 1 overwrites what row 0 wrote there.
    arguments = {
        "cnt": Arr.from_numpy(np.array([1, 1], dtype=np.int64)),
        "off": Arr.from_numpy(np.array([0, 1, 2], dtype=np.int64)),
        "val": Arr.ragged([1, 1], values=np.zeros(2)),
    }
    Kernel(overlap_then_write)(**arguments)
    assert arguments["val"].numpy().tolist() == [2.0, 0.0]


def rescanned_then_summed(
    cnt: Arr[Fin[n], Nat],  # noqa: F821
    off: Arr[Fin[n + 1], Nat],  # noqa: F821
    val: Arr[Fin[n], Fin[cnt], Real],  # noqa: F821
    y: Arr[Fin[n], Real],  # noqa: F821
):
    """The offsets computed again from the counts, row by row, then read."""
    off[0] = 0
    for r in y.dom:
        off[r + 1] = cnt[r] + off[r]
        y[r] = reduce_sum(val[r, j] for j in val.dom[r])


def first_row_moved(
    cnt: Arr[Fin[n], Nat],  # noqa: F821
    off: Arr[Fin[n + 1], Nat],  # noqa: F821
    val: Arr[Fin[n], Fin[cnt], Real],  # noqa: F821
    y: Arr[Fin[n], Real],  # noqa: F821
):
    """Row 0 moved one cell on, then every row summed."""
    off[0] = 1
    for r in y.dom:
        y[r] = reduce_sum(val[r, j] for j in val.dom[r])


def test_offsets_written_as_the_counts_lay_them_out_decide_the_layout() -> None:
    # The contract checks that the offsets start at 0 and have the counts as
    # their differences, so writing either again leaves every cell as it was,
    # and the facts that rest on the layout are worth what they are (#86).
    from lanky.ledger import Status

    _term, ledger, (layout,) = layout_of(rescanned_then_summed)
    assert layout.status is Status.DECIDED, layout.provenance
    assert layout.decided_by == "isl"
    assert layout.provenance["restated"] == ["S1"]
    assert layout.provenance["fixed"] == ["S0"]
    assert "S1 writes off[q] = off[q - 1] + cnt[q - 1]" in layout.provenance["rule"]
    (read,) = [fact for fact in ledger if fact.id.endswith(":S2:read:val[r, j]")]
    assert read.rests_on == (layout.id,)
    assert ledger.support(read).effective is Status.DECIDED
    assert ledger.support(read).under == ()


def test_a_start_written_past_the_end_refutes_the_layout() -> None:
    # Row 0 starts at 0 whatever the counts; at 1, a buffer of one cell, the
    # counts 1 in row 0 and 0 after it, has the row past its end.
    from lanky.ledger import Status

    _term, _ledger, (layout,) = layout_of(first_row_moved)
    assert layout.status is Status.REFUTED, layout.provenance
    assert layout.provenance["reason"].endswith("at [n=1]")
    arguments = {
        "cnt": Arr.from_numpy(np.array([1], dtype=np.int64)),
        "off": Arr.from_numpy(np.array([0, 1], dtype=np.int64)),
        "val": Arr.ragged([1], values=np.ones(1)),
        "y": Arr.zeros(1),
    }
    with pytest.raises(IndexError, match="of the flat buffer"):
        Kernel(first_row_moved)(**arguments)


def test_a_start_read_from_another_array_is_refuted_by_a_run() -> None:
    # isl has no question about the write alone, nor about it in order: the
    # rows in order before it leave room for s[r] before the row ahead of it.
    # So the typing rules leave the fact assumed, with both reasons, and a
    # native run of the kernel's own, on a drawn input, leaves two rows on one
    # cell, which refutes it (#103).
    from lanky.ledger import Status

    _term, _ledger, (layout,) = layout_of(gather_through_moved_rows)
    assert layout.status is Status.ASSUMED
    assert layout.term is None
    reason = layout.provenance["reason"]
    assert (
        "S0 writes off[r] = s[r], which is neither the start the counts give "
        "the row, off[q - 1] + cnt[q - 1] at q >= 1, nor a value of the loop "
        "variables and the sizes alone; and by induction over the run, S0 "
        "writes off[r] = s[r], and the rows in order before it leave room for "
        "a write out of order: [r="
    ) in reason
    (ran,) = [
        fact
        for fact in Kernel(gather_through_moved_rows).facts()
        if fact.kind == "layout"
    ]
    assert ran.status is Status.REFUTED, ran.provenance
    assert ran.decided_by == "native"
    assert ran.provenance["counterexample"]["input"].startswith("sample ")
    # A row moved onto another, or off the buffer and read there.
    assert (
        "the body run natively leaves rows " in ran.provenance["reason"]
        or "the body run natively reads col through the layout it has written"
        in ran.provenance["reason"]
    )
    assert ran.provenance["reason"].endswith(
        "so the rows do not stay inside the buffer and apart"
    )


def one_further_on(
    cnt: Arr[Fin[n], Nat],  # noqa: F821
    off: Arr[Fin[n + 1], Nat],  # noqa: F821
    val: Arr[Fin[n], Fin[cnt], Real],  # noqa: F821
    y: Arr[Fin[n], Real],  # noqa: F821
):
    """The issue's second kernel: each row a cell further on than its counts say."""
    for r in y.dom:
        off[r + 1] = off[r] + cnt[r] + 1
        y[r] = reduce_sum(val[r, j] for j in val.dom[r])


def counts_grown(
    cnt: Arr[Fin[n], Nat],  # noqa: F821
    val: Arr[Fin[n], Fin[cnt], Real],  # noqa: F821
    y: Arr[Fin[n], Real],  # noqa: F821
):
    """Every row one entry longer, onto the next row or past the buffer."""
    for r in y.dom:
        cnt[r] = cnt[r] + 1
        y[r] = reduce_sum(val[r, j] for j in val.dom[r])


@pytest.mark.parametrize("fn", [one_further_on, counts_grown])
def test_a_row_moved_past_the_buffer_is_refuted_by_the_run_that_reads_it(fn) -> None:
    # Each row a cell further on, or a cell longer: the last one past the end
    # of the buffer. The native run reads it there and raises, which skipped
    # the input and said nothing of the layout (#103).
    from lanky.ledger import Status

    _term, _ledger, (layout,) = layout_of(fn)
    assert layout.status is Status.ASSUMED
    (ran,) = [fact for fact in Kernel(fn).facts() if fact.kind == "layout"]
    assert ran.status is Status.REFUTED, ran.provenance
    assert ran.decided_by == "native"
    reason = ran.provenance["reason"]
    assert "the body run natively reads val through the layout it has written" in (
        reason
    )
    assert "of the flat buffer by the counts and offsets the kernel declares" in (
        reason
    )


def counts_cleared(
    cnt: Arr[Fin[n], Nat],  # noqa: F821
    off: Arr[Fin[n + 1], Nat],  # noqa: F821
    val: Arr[Fin[n], Fin[cnt], Real],  # noqa: F821
    y: Arr[Fin[n], Real],  # noqa: F821
):
    """The issue's third kernel: each row's length cleared once it is summed."""
    for r in y.dom:
        y[r] = reduce_sum(val[r, j] for j in val.dom[r])
        cnt[r] = 0


def counts_halved(
    cnt: Arr[Fin[n], Nat],  # noqa: F821
    val: Arr[Fin[n], Fin[cnt], Real],  # noqa: F821
    y: Arr[Fin[n], Real],  # noqa: F821
):
    """Every row half as long, from where it starts, by its own offsets."""
    for r in y.dom:
        cnt[r] = cnt[r] // 2
        y[r] = reduce_sum(val[r, j] for j in val.dom[r])


def counts_cleared_where_flagged(
    cnt: Arr[Fin[n], Nat],  # noqa: F821
    flag: Arr[Fin[n], Nat],  # noqa: F821
    val: Arr[Fin[n], Fin[cnt], Real],  # noqa: F821
    y: Arr[Fin[n], Real],  # noqa: F821
):
    """A row's length cleared under a guard isl cannot state."""
    for r in y.dom:
        with when(flag[r] != 0):
            cnt[r] = 0
        y[r] = reduce_sum(val[r, j] for j in val.dom[r])


def start_within_its_slack(
    cnt: Arr[Fin[n], Nat],  # noqa: F821
    off: Arr[Fin[n + 1], Nat],  # noqa: F821
    s: Arr[Fin[n], Nat],  # noqa: F821
    val: Arr[Fin[n], Fin[cnt], Real],  # noqa: F821
    y: Arr[Fin[n], Real],  # noqa: F821
):
    """A start moved on by ``s[r]`` where the moved row ends before the next."""
    for r in y.dom:
        with when(
            (r + 1 < y.dom.size) & (off[r] + cnt[r] + s[r] <= off[r + 1] - cnt[r])
        ):
            off[r] = off[r] + s[r]
        y[r] = reduce_sum(val[r, j] for j in val.dom[r])


@pytest.mark.parametrize(
    ("fn", "writers"),
    [
        (counts_cleared, ["S1"]),
        (counts_halved, ["S0"]),
        (counts_cleared_where_flagged, ["S0"]),
        (start_within_its_slack, ["S0"]),
    ],
)
def test_a_write_that_keeps_the_rows_in_order_decides_the_layout(fn, writers) -> None:
    # A count written, a start read from another array, a write under a guard
    # isl cannot state: each left the fact assumed (#103). The rows in order,
    # each starting no earlier than the row before it ends, inside the buffer,
    # hold when the call starts, and isl shows that each of these writes keeps
    # them so wherever they held before it, so they hold throughout the run.
    from lanky.ledger import Status

    _term, ledger, (layout,) = layout_of(fn)
    assert layout.status is Status.DECIDED, layout.provenance
    assert layout.decided_by == "isl"
    assert layout.provenance["ordered"] == writers
    assert layout.provenance["rule"].startswith(
        "every write keeps the rows of val in order inside the buffer: "
    )
    assert layout.provenance["rule"].endswith(
        "rows so laid out are inside the buffer and apart"
    )
    reads = [fact for fact in ledger if layout.id in fact.rests_on]
    assert reads
    for fact in reads:
        assert ledger.support(fact).effective is Status.DECIDED, fact.id
    (ran,) = [fact for fact in Kernel(fn).facts() if fact.kind == "layout"]
    assert ran.status is Status.ASSUMED and ran.term is not None


def start_moved_by_a_bit(
    cnt: Arr[Fin[n], Nat],  # noqa: F821
    off: Arr[Fin[n + 1], Nat],  # noqa: F821
    bit: Arr[Fin[n], Fin[2]],  # noqa: F821
    val: Arr[Fin[n], Fin[cnt], Real],  # noqa: F821
    y: Arr[Fin[n], Real],  # noqa: F821
):
    """A start moved on by 0 or 1 where its row leaves a cell of room."""
    for r in y.dom:
        with when((r + 1 < y.dom.size) & (off[r] + cnt[r] + 1 <= off[r + 1])):
            off[r] = off[r] + bit[r]
        y[r] = reduce_sum(val[r, j] for j in val.dom[r])


def test_only_a_parameters_type_is_a_hypothesis_of_the_induction() -> None:
    # bit's type, Fin(2), which the contract checks on entry, is what keeps
    # the moved row inside its room. A program's temporary has no contract
    # on entry, so the same array as one is no hypothesis, and the fact is
    # left assumed.
    from lanky.ledger import Status

    from loopty import typing as rules
    from loopty.oracle import IslOracle

    term, _ledger, (layout,) = layout_of(start_moved_by_a_bit)
    assert layout.status is Status.DECIDED, layout.provenance
    typ = dict(term.params)["bit"]
    made = dataclasses.replace(
        term,
        params=tuple((name, sort) for name, sort in term.params if name != "bit"),
        temporaries=(*term.temporaries, ("bit", typ)),
    )
    (fact,) = rules.layout_facts(made, "start_moved_by_a_bit")
    oracle = IslOracle()
    if oracle.can_establish(fact):
        fact = oracle.establish(fact) or fact
    assert fact.status is Status.ASSUMED, fact.provenance
    assert "by induction over the run, S0 writes off[r] = off[r] + bit[r]" in (
        fact.provenance["reason"]
    )


def test_offsets_with_no_counts_are_kept_in_order_by_induction() -> None:
    # A family whose rows are as long as the differences of their offsets
    # (a term written by hand; tracing names counts): starting row 0 at 0
    # keeps the offsets in order, and moving it past row 1's start does not.
    from lanky.ledger import Status

    import hand_terms as ht
    from loopty import typing as rules
    from loopty.oracle import IslOracle
    from loopty.term import Access, Stmt

    base = ht.spmv_term()
    oracle = IslOracle()

    def layout_with(value) -> object:
        stmt = Stmt(
            id="S1",
            inames=(),
            domain=isl.Set("{ [] }"),
            assignee=Access("off", (0,)),
            expr=value,
            kind="assign",
            guard=None,
            where="hand.py:2",
        )
        term = dataclasses.replace(base, stmts=(*base.stmts, stmt))
        (fact,) = rules.layout_facts(term, "spmv")
        return oracle.establish(fact) if oracle.can_establish(fact) else fact

    kept = layout_with(0)
    assert kept.status is Status.DECIDED, kept.provenance
    assert "off[q] <= off[q + 1] for every row q" in kept.provenance["rule"]
    moved = layout_with(ht.S("off", 1) + 1)
    assert moved.status is Status.ASSUMED
    assert "by induction over the run, S1 writes off[0] = off[1] + 1" in (
        moved.provenance["reason"]
    )


def second_block_moved(
    cnt: Arr[Fin[n], Fin[m], Nat],  # noqa: F821
    off: Arr[Fin[n * m + 1], Nat],  # noqa: F821
    val: Arr[Fin[n], Fin[m], Fin[cnt], Real],  # noqa: F821
):
    """``off[n]``, the start of a row when there are ``n * m``, moved to 0."""
    off[cnt.dom.size] = 0
    for r in cnt.dom:
        for s in cnt.dom[r]:
            for j in val.dom[r, s]:
                val[r, s, j] = 1.0


def test_a_two_axis_family_is_refused_before_any_layout_fact() -> None:
    # The rows would be the n * m cells of the counts, and off[n] starts one
    # of them when m > 1. Counted by the first axis alone, n rows, it was past
    # the last start and the fact was decided; then left assumed. Nothing
    # builds such an array, and lowering indexes none, so tracing refuses the
    # type before any fact is stated about its layout (#112).
    from loopty.trace import TraceError

    with pytest.raises(TraceError, match="is ragged in axis 2 of 3"):
        term_of(second_block_moved)


def test_a_row_shortened_in_its_own_loop_leaves_its_reads_assumed() -> None:
    # A row's length set to zero keeps the rows in order, and the layout was
    # decided by induction, its reads worth decided. The loop over the row
    # read its length when it started, and the read of val[r, 1] after the
    # write is past the row's end: the native run refuses it, and the read's
    # in-bounds fact, stated against the loop's reading, is false. So the
    # layout is not asked by induction where a length is written inside a
    # loop it bounds, and the reads stay worth assumed (#103).
    from lanky.ledger import Status

    _term, ledger, (layout,) = layout_of(shortened_in_its_own_loop)
    assert layout.provenance["written"] == ["cnt"]
    assert layout.status is Status.ASSUMED
    assert layout.provenance["reason"].endswith(
        "S1 writes cnt inside a loop over a row, which runs to the length it "
        "read as nl_cnt_r when it started: the in-bounds facts of the row's "
        "entries are stated against that reading, which a write that shortens "
        "the row leaves behind, and which the rows in order do not keep"
    )
    (read,) = [fact for fact in ledger if fact.id.endswith(":S0:read:val[r, j]")]
    assert read.status is Status.DECIDED
    assert ledger.support(read).effective is Status.ASSUMED
    assert ledger.support(read).under == (layout.id,)
    counts = [2, 1, 3]
    arguments = {
        "cnt": Arr.from_numpy(np.array(counts, dtype=np.int64)),
        "val": Arr.ragged(counts, values=np.arange(1.0, 7.0)),
        "y": Arr.zeros(3),
    }
    with pytest.raises(IndexError, match="column 1 out of range for row 0 of length 0"):
        Kernel(shortened_in_its_own_loop)(**arguments)


def emptied_then_moved_onto(
    cnt: Arr[Fin[n], Nat],  # noqa: F821
    off: Arr[Fin[n + 1], Nat],  # noqa: F821
    val: Arr[Fin[n], Fin[cnt], Real],  # noqa: F821
):
    """Each row written, emptied, and the next row moved onto its start."""
    for r in cnt.dom:
        for j in val.dom[r]:
            val[r, j] = val[r, j] + 1.0
        cnt[r] = 0
        with when(r + 1 < cnt.dom.size):
            off[r + 1] = off[r] + cnt[r]


def test_rows_moved_onto_cells_written_before_leave_the_layout_assumed() -> None:
    # Every write keeps the rows in order, and the layout was decided by
    # induction, and with it the disjoint writes of S0, which tell the cells
    # of val apart as [r, j]. Each row is moved onto the start of the one
    # before it once that one is emptied, so S0 writes val[0, 0], val[1, 0]
    # and val[2, 0] on one cell. A layout whose rows move is not asked by
    # induction in a kernel that writes the family's arrays (#103).
    from lanky.ledger import Status

    _term, ledger, (layout,) = layout_of(emptied_then_moved_onto)
    assert layout.status is Status.ASSUMED
    assert layout.term is None
    assert layout.provenance["reason"].endswith(
        "S2 writes off, which can move a row of val, and S0 writes val, whose "
        "disjoint writes and dependences tell its cells apart as [r, j]: one "
        "cell of the buffer each only while every row keeps its start, which "
        "the rows in order do not say"
    )
    (disjoint,) = [
        fact
        for fact in ledger
        if fact.kind == "disjoint-writes" and fact.provenance["array"] == "val"
    ]
    assert disjoint.status is Status.DECIDED
    assert ledger.support(disjoint).effective is Status.ASSUMED
    counts = [2, 2, 2]
    arguments = {
        "cnt": Arr.from_numpy(np.array(counts, dtype=np.int64)),
        "off": Arr.from_numpy(np.array([0, 2, 4, 6], dtype=np.int64)),
        "val": Arr.ragged(counts, values=np.zeros(6)),
    }
    Kernel(emptied_then_moved_onto)(**arguments)
    # Three instances of S0 wrote cell 0, and three cell 1.
    assert arguments["val"].numpy().tolist() == [3.0, 3.0, 0.0, 0.0, 0.0, 0.0]


@pytest.mark.parametrize("fn", [row_sums_then_next_offset, row_sums_then_next_count])
def test_a_kernel_that_writes_its_layout_has_a_layout_fact(fn) -> None:
    _term, ledger, (layout,) = layout_of(fn)
    (read,) = [fact for fact in ledger if fact.id.endswith(":S0:read:val[r, j]")]
    assert read.rests_on == (layout.id,)


def gather_through_moved_rows(
    cnt: Arr[Fin[n], Nat],  # noqa: F821
    off: Arr[Fin[n + 1], Nat],  # noqa: F821
    col: Arr[Fin[n], Fin[cnt], Fin[m]],  # noqa: F821
    s: Arr[Fin[n], Nat],  # noqa: F821
    x: Arr[Fin[m], Real],  # noqa: F821
    first: Arr[Fin[n], Fin[m]],  # noqa: F821
    y: Arr[Fin[n], Real],  # noqa: F821
):
    """The rows of ``col`` moved to start at ``s[r]``, then read as indices."""
    for r in y.dom:
        off[r] = s[r]
        y[r] = reduce_sum(x[col[r, j]] for j in col.dom[r])
        for j in col.dom[r]:
            first[r] = col[r, j]


def test_an_index_read_from_a_moved_row_rests_on_the_layout() -> None:
    # ``x[col[r, j]]`` is in bounds by the element sort of ``col``, which holds
    # of a cell of ``col``, and ``col[r, j]`` is a cell of ``col`` only while
    # the rows stay inside its buffer: with ``s[r]`` past its end the compiled
    # run reads an index from outside ``col`` and then a cell outside ``x``.
    from lanky.ledger import Status

    _term, ledger, (layout,) = layout_of(gather_through_moved_rows)
    (gather,) = [fact for fact in ledger if fact.id.endswith(":read:x[col[r, j]]")]
    assert gather.decided_by == "type"
    assert gather.rests_on == (layout.id,)
    assert ledger.support(gather).effective is Status.ASSUMED
    # So does the write of such an index into an array of the sort.
    (sort,) = [fact for fact in ledger if fact.kind == "element-sort"]
    assert sort.decided_by == "type"
    assert sort.rests_on == (layout.id,)


def test_a_kernel_that_only_reads_its_layout_has_none() -> None:
    from loopty.schedule import Schedule

    term, ledger, layouts = layout_of(row_sums_through_offsets)
    assert layouts == []
    assert not [fact for fact in ledger if fact.rests_on]
    assert not [fact for fact in Schedule(term).facts() if fact.rests_on]


OVERLAP = """
from __future__ import annotations

from lanky.prelude import Nat, Real

from loopty import Arr, Fin, kernel


@kernel
def overlap_then_write(
    cnt: Arr[Fin[n], Nat],
    off: Arr[Fin[n + 1], Nat],
    val: Arr[Fin[n], Fin[cnt], Real],
):
    for r in cnt.dom:
        off[r] = 0
        for j in val.dom[r]:
            val[r, j] = r + 1.0
"""


def test_check_shows_the_writes_decided_under_the_layout(tmp_path, capsys) -> None:
    from lanky.cli import main as lanky_main

    path = tmp_path / "overlap.py"
    path.write_text(OVERLAP, encoding="utf-8")
    assert lanky_main(["check", str(path)]) == 1
    out = capsys.readouterr().out
    layout = "layout:overlap.overlap_then_write@9:cnt"
    rows = out.splitlines()
    (row,) = [line for line in rows if "write distinct cells of val" in line]
    assert row.startswith(f"decided under {layout}")
    assert row.split()[3] == "refuted"
    (fact,) = [line for line in rows if line.startswith("refuted ")]
    assert "stay inside their buffers" in fact
    assert "  [r=1] is one of the instances of S0 that write into off" in out


# }}}
