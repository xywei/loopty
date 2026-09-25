"""The runtime array: dense, ragged, domains, and the type it reports."""

from __future__ import annotations

import numpy as np
import pytest

from loopty.arr import Arr, ArrSpec
from loopty.idx import Fin


def test_zeros_from_index_types() -> None:
    a = Arr.zeros((Fin[2], Fin[3]))
    assert a.shape == (2, 3)
    assert a.dtype == np.float64
    assert not a.is_ragged
    assert a.numpy().sum() == 0.0


def test_zeros_accepts_a_bare_axis_and_a_dtype() -> None:
    a = Arr.zeros(Fin[4], dtype=np.int64)
    assert a.shape == (4,)
    assert a.dtype == np.int64


def test_zeros_refuses_a_symbolic_size() -> None:
    import pymbolic.primitives as prim

    with pytest.raises(TypeError, match="symbolic axis"):
        Arr.zeros(Fin[prim.Variable("n")])


def test_from_numpy_shares_the_buffer() -> None:
    values = np.arange(6, dtype=np.float64).reshape(2, 3)
    a = Arr.from_numpy(values)
    a[1, 2] = 42.0
    assert values[1, 2] == 42.0
    assert a[0, 1] == 1.0


def test_dense_dom_iterates_the_outer_axis_and_fibers_the_inner() -> None:
    a = Arr.zeros((Fin[2], Fin[3]))
    assert list(a.dom) == [0, 1]
    assert len(a.dom) == 2
    assert list(a.dom[1]) == [0, 1, 2]
    with pytest.raises(IndexError, match="no axis"):
        a.dom[0][0]


def test_dense_dom_rejects_an_out_of_range_fiber() -> None:
    a = Arr.zeros((Fin[2], Fin[3]))
    with pytest.raises(IndexError, match="out of range"):
        a.dom[5]


def test_ragged_offsets_counts_and_values() -> None:
    a = Arr.ragged([2, 0, 3], values=[1.0, 2.0, 3.0, 4.0, 5.0])
    assert a.is_ragged
    assert list(a.offsets) == [0, 2, 2, 5]
    assert list(a.counts) == [2, 0, 3]
    assert a.ndim == 2
    assert a[0, 1] == 2.0
    assert a[2, 0] == 3.0
    assert list(a[2]) == [3.0, 4.0, 5.0]


def test_ragged_dom_gives_the_per_row_extent() -> None:
    a = Arr.ragged([2, 0, 3])
    assert list(a.dom) == [0, 1, 2]
    assert list(a.dom[0]) == [0, 1]
    assert list(a.dom[1]) == []
    assert list(a.dom[2]) == [0, 1, 2]


def test_ragged_indexing_is_bounds_checked_per_row() -> None:
    a = Arr.ragged([2, 3])
    with pytest.raises(IndexError, match="column 2 out of range"):
        a[0, 2]
    with pytest.raises(IndexError, match="row 2 out of range"):
        a[2, 0]


def test_ragged_write_lands_in_the_flat_buffer() -> None:
    a = Arr.ragged([2, 3])
    a[1, 2] = 7.0
    assert list(a.numpy()) == [0.0, 0.0, 0.0, 0.0, 7.0]
    a[0] = 1.0
    assert list(a.numpy()) == [1.0, 1.0, 0.0, 0.0, 7.0]


def test_ragged_rejects_a_mismatched_value_buffer() -> None:
    with pytest.raises(ValueError, match="ragged slots"):
        Arr.ragged([2, 3], values=[1.0, 2.0])


def test_offsets_that_do_not_start_at_zero_are_refused() -> None:
    # Monotone and ending at the right place was not enough: [-1, 2] passed
    # both of those and gave row 0 the flat slice values[-1:2], which numpy
    # wraps round to the end of the buffer and generated C reads in front of it.
    with pytest.raises(ValueError, match="must start at 0"):
        Arr(np.zeros(3), offsets=np.array([-1, 2]))
    with pytest.raises(ValueError, match="must start at 0"):
        Arr(np.zeros(3), offsets=np.array([1, 1, 3]))
    # The CSR spelling, which is what Arr.ragged builds, is still accepted.
    assert Arr(np.zeros(3), offsets=np.array([0, 1, 3])).counts.tolist() == [1, 2]


def test_a_negative_index_into_a_dense_array_is_refused() -> None:
    # numpy would read x[-1] as the last cell; Fin[n] has no negative points and
    # generated C reads in front of the buffer, so the native run may not wrap.
    a = Arr.from_numpy(np.arange(6.0).reshape(2, 3))
    with pytest.raises(IndexError, match="index -1 in -1 is negative"):
        a[-1]
    with pytest.raises(IndexError, match="is negative"):
        a[0, -1]
    with pytest.raises(IndexError, match="is negative"):
        a[np.int64(-2), 0]
    with pytest.raises(IndexError, match="is negative"):
        a[0, -1] = 7.0
    assert a.numpy()[0, 2] == 2.0
    # Every non-negative index, and a slice, still reaches numpy unchanged.
    assert a[1, 2] == 5.0
    assert a[np.int64(1), 0] == 3.0
    assert a[0].tolist() == [0.0, 1.0, 2.0]
    assert a[0, 1:].tolist() == [1.0, 2.0]
    with pytest.raises(IndexError):
        a[2, 0]


def test_dense_type_is_an_arrtype_with_concrete_axes() -> None:
    a = Arr.zeros((Fin[2], Fin[3]), dtype=np.int64)
    t = a.type
    assert t.axes == (2, 3)
    assert t.ragged == (False, False)
    assert t.ndim == 2
    assert t.dtype == np.int64


def test_ragged_type_marks_the_second_axis() -> None:
    t = Arr.ragged([2, 0, 3]).type
    assert t.axes == (3, (2, 0, 3))
    assert t.ragged == (False, True)


def test_arr_subscript_is_the_written_type() -> None:
    n = 5
    spec = Arr[Fin[n], Fin[3], float]
    assert isinstance(spec, ArrSpec)
    assert spec.axes == (Fin[5], Fin[3])
    assert spec.dtype is float
    # The axes print as the index type prints them, whichever Fin is in play.
    assert repr(spec).startswith("Arr[")
    assert "float" in repr(spec)


def test_a_native_kernel_body_runs_on_arrs() -> None:
    # This is the shape of a traced body, executed natively: sizes come from
    # the data through .dom, never from a free size name.
    off = Arr.from_numpy(np.array([0, 2, 2, 5], dtype=np.int64))
    val = Arr.ragged([2, 0, 3], values=[1.0, 2.0, 3.0, 4.0, 5.0])
    y = Arr.zeros(Fin[3])
    for r in val.dom:
        total = 0.0
        for j in val.dom[r]:
            total += val[r, j]
        y[r] = total
    assert list(y.numpy()) == [3.0, 0.0, 12.0]
    assert off[3] == 5
