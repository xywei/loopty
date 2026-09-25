"""The term interpreter: a traced term, run with numpy semantics.

The interpreter is what the faithfulness fact compares with the native run, so
the property that matters most is that on a faithful trace the two agree *bit
for bit*: the same operations on the same numpy scalars in the same order. Each
test runs the native body and the interpreter on copies of one input and asks
for identical buffers, so a wrong order, a skipped guard or a reduction summed
another way shows up as a difference in the last bit rather than hiding inside
a tolerance.
"""

from __future__ import annotations

import islpy as isl
import numpy as np
import pymbolic.primitives as prim
import pytest
from lanky.prelude import Nat, Real

from loopty import Arr, Fin, kernel, reduce_sum, when
from loopty.interpret import InterpretError, TooLarge, interpret
from loopty.term import Access, ArrType, Stmt, Term


def _copies(arguments: dict) -> dict:
    out = {}
    for name, value in arguments.items():
        if isinstance(value, Arr):
            out[name] = (
                Arr(value.numpy().copy(), value.offsets.copy())
                if value.is_ragged
                else Arr(value.numpy().copy())
            )
        else:
            out[name] = value
    return out


def both(k, arguments: dict) -> tuple[dict, dict]:
    """The native run and the interpreted term, each on copies of ``arguments``."""
    native = _copies(arguments)
    interpreted = _copies(arguments)
    with np.errstate(divide="ignore"):
        # A masked write still computes its right-hand side natively.
        k(**native)
    interpret(k.term, interpreted)
    return native, interpreted


def assert_identical(native: dict, interpreted: dict) -> None:
    for name, value in native.items():
        if isinstance(value, Arr):
            got = interpreted[name].numpy()
            assert got.dtype == value.numpy().dtype, name
            assert got.tobytes() == value.numpy().tobytes(), (name, got, value)


@kernel
def spmv(
    cnt: Arr[Fin[n], Nat],  # noqa: F821
    col: Arr[Fin[n], Fin[cnt], Fin[m]],  # noqa: F821
    val: Arr[Fin[n], Fin[cnt], Real],  # noqa: F821
    x: Arr[Fin[m], Real],  # noqa: F821
    y: Arr[Fin[n], Real],  # noqa: F821
):
    for r in y.dom:
        y[r] = reduce_sum(val[r, j] * x[col[r, j]] for j in val.dom[r])


def csr(counts: list[int], columns: int = 4, seed: int = 0) -> dict:
    rng = np.random.default_rng(seed)
    total = sum(counts)
    return {
        "cnt": Arr.from_numpy(np.array(counts, dtype=np.int64)),
        "col": Arr.ragged(counts, values=rng.integers(0, columns, total)),
        "val": Arr.ragged(counts, values=rng.standard_normal(total)),
        "x": Arr.from_numpy(rng.standard_normal(columns)),
        "y": Arr.zeros(len(counts)),
    }


def test_a_ragged_product_interprets_to_the_native_bits() -> None:
    # Rows of every length, including empty ones: the ragged bound is a
    # reflected parameter read from ``cnt`` once ``r`` has a value.
    native, interpreted = both(spmv, csr([3, 0, 1, 4, 2]))
    assert_identical(native, interpreted)


def test_the_result_names_the_arrays_the_term_writes() -> None:
    arguments = csr([2, 1])
    out = interpret(spmv.term, arguments)
    assert list(out) == ["y"]
    assert out["y"] is arguments["y"].numpy()


@kernel
def interleaved(a: Arr[Fin[n], Real], b: Arr[Fin[n], Real]):  # noqa: F821
    for i in a.dom:
        with when(i > 0):
            a[i] = b[i - 1] + 1.0
        b[i] = a[i] * 2.0


def test_statements_run_in_the_order_of_the_loop_nest() -> None:
    # ``a[i]`` reads the ``b`` the previous iteration's second statement wrote.
    # Running every instance of one statement before the next would read the
    # initial ``b`` instead, and differ from the second iteration on.
    arguments = {"a": Arr.from_numpy(np.zeros(5)), "b": Arr.from_numpy(np.zeros(5))}
    native, interpreted = both(interleaved, arguments)
    assert list(native["b"].numpy()) == [0.0, 2.0, 6.0, 14.0, 30.0]
    assert_identical(native, interpreted)


@kernel
def ordered(x: Arr[Fin[n], Real], y: Arr[Fin[1], Real]):  # noqa: F821
    y[0] = reduce_sum(x[j] for j in x.dom)


def test_a_reduction_adds_in_the_order_the_native_sum_does() -> None:
    # Left to right this is 1.0; right to left the 1.0s are absorbed by 1e16
    # and it is 0.0.
    arguments = {
        "x": Arr.from_numpy(np.array([1e16, 1.0, -1e16, 1.0])),
        "y": Arr.zeros(1),
    }
    native, interpreted = both(ordered, arguments)
    assert native["y"].numpy()[0] == 1.0
    assert_identical(native, interpreted)


@kernel
def masked(
    cnt: Arr[Fin[n], Nat],  # noqa: F821
    lst: Arr[Fin[n], Fin[cnt], Fin[n]],  # noqa: F821
    x: Arr[Fin[n], Real],  # noqa: F821
    term: Arr[Fin[n], Fin[cnt], Real],  # noqa: F821
    out: Arr[Fin[n], Real],  # noqa: F821
):
    for t in out.dom:
        for j in lst.dom[t]:
            term[t, j] = 0.0
            with when(lst[t, j] != t):
                term[t, j] = 1.0 / (x[t] - x[lst[t, j]])
        out[t] = reduce_sum(term[t, j] for j in lst.dom[t])


def test_a_guard_that_reads_data_masks_the_write() -> None:
    # The self pair would divide by zero; the guard is evaluated at each
    # instance and the write, with its right-hand side, is skipped.
    counts = [3, 2, 3]
    arguments = {
        "cnt": Arr.from_numpy(np.array(counts, dtype=np.int64)),
        "lst": Arr.ragged(counts, values=[0, 1, 2, 1, 0, 2, 1, 0]),
        "x": Arr.from_numpy(np.array([0.0, 0.5, 2.0])),
        "term": Arr.ragged(counts),
        "out": Arr.zeros(3),
    }
    native, interpreted = both(masked, arguments)
    assert_identical(native, interpreted)


@kernel
def lookup(
    col: Arr[Fin[n], Fin[m]],  # noqa: F821
    x: Arr[Fin[m], Real],  # noqa: F821
    y: Arr[Fin[n], Real],  # noqa: F821
):
    for i in y.dom:
        y[i] = x[col[i]]


def test_an_index_array_stored_as_floats_is_read_as_integers() -> None:
    # The native run reads such an array through an integer copy (see
    # ``Kernel.__call__``), and so does the interpreter.
    arguments = {
        "col": Arr.from_numpy(np.array([2.0, 0.0, 1.0])),
        "x": Arr.from_numpy(np.array([10.0, 20.0, 30.0])),
        "y": Arr.zeros(3),
    }
    native, interpreted = both(lookup, arguments)
    assert list(interpreted["y"].numpy()) == [30.0, 10.0, 20.0]
    assert_identical(native, interpreted)


def _mystery(value):
    """A call the interpreter has no numpy counterpart for."""
    if isinstance(value, prim.ExpressionNode):
        return prim.Call(prim.Variable("mystery"), (value,))
    return value


@kernel
def mysterious(x: Arr[Fin[n], Real], y: Arr[Fin[n], Real]):  # noqa: F821
    for i in y.dom:
        y[i] = _mystery(x[i])


def test_a_call_with_no_numpy_counterpart_is_an_error_not_a_guess() -> None:
    arguments = {"x": Arr.from_numpy(np.ones(2)), "y": Arr.zeros(2)}
    with pytest.raises(InterpretError, match="no numpy counterpart for the call"):
        interpret(mysterious.term, arguments)


def test_more_instances_than_allowed_is_too_large() -> None:
    arguments = {"x": Arr.from_numpy(np.ones(10)), "y": Arr.zeros(10)}
    with pytest.raises(TooLarge, match="more than 5 statement instances"):
        interpret(mysterious.term, arguments, limit=5)


def test_the_terms_of_a_reduction_count_against_the_limit() -> None:
    # One statement instance, which sums ten terms: a matrix-vector product
    # with a benchmark's sizes is few instances and a great deal of work.
    arguments = {"x": Arr.from_numpy(np.ones(10)), "y": Arr.zeros(1)}
    with pytest.raises(
        TooLarge, match="more than 5 statement instances and reduction terms"
    ):
        interpret(ordered.term, arguments, limit=5)
    interpret(ordered.term, arguments, limit=11)
    assert arguments["y"].numpy()[0] == 10.0


def test_a_domain_past_the_limit_is_refused_before_a_point_is_visited(
    monkeypatch,
) -> None:
    # Collecting a domain's points is itself the work: a million by a million
    # is refused by its bounding box, not after a trillion points were visited.
    import loopty.interpret as interpret_module

    def visited(*_args):
        raise AssertionError("a point of the domain was visited")

    monkeypatch.setattr(interpret_module, "_enumerate", visited)
    n = prim.Variable("n")
    stmt = Stmt(
        id="S0",
        inames=("i", "j"),
        domain=isl.Set("[a, n] -> { [i, j] : 0 <= i < a and 0 <= j < a }"),
        assignee=Access("y", (0,)),
        expr=1.0,
        kind="assign",
        guard=None,
        where="hand.py:1",
    )
    real = np.dtype(np.float64)
    term = Term(
        name="square",
        params=(("a", Nat), ("y", ArrType(axes=(n,), dtype=real, ragged=(False,)))),
        sizes=("n",),
        stmts=(stmt,),
        post=None,
    )
    with pytest.raises(TooLarge, match="more than 1000 statement instances"):
        interpret(term, {"a": 10**6, "y": Arr.zeros(1)}, limit=1000)


@kernel
def recount(cnt: Arr[Fin[n], Nat], val: Arr[Fin[n], Fin[cnt], Real]):  # noqa: F821
    for r in cnt.dom:
        cnt[r] = cnt[r] + 0
    for r in val.dom:
        for j in val.dom[r]:
            val[r, j] = 1.0


def test_a_loop_bound_the_kernel_writes_is_refused() -> None:
    # Which instances of the second loop run depends on when ``cnt[r]`` is
    # read, and the interpreter reads every domain before it runs anything.
    arguments = {
        "cnt": Arr.from_numpy(np.array([1, 2], dtype=np.int64)),
        "val": Arr.ragged([1, 2]),
    }
    with pytest.raises(InterpretError, match="reads cnt, which recount also writes"):
        interpret(recount.term, arguments)


def test_a_domain_parameter_that_is_not_a_whole_number_is_named() -> None:
    # isl's parameters are integers, so a domain bounded by a real scalar has
    # no points to enumerate at a = 2.5; the error says which value it got.
    real = np.dtype(np.float64)
    n, i, a = prim.Variable("n"), prim.Variable("i"), prim.Variable("a")
    stmt = Stmt(
        id="S0",
        inames=("i",),
        domain=isl.Set("[a, n] -> { [i] : 0 <= i < n and i < a }"),
        assignee=Access("y", (i,)),
        expr=1.0,
        kind="assign",
        guard=prim.Comparison(i, "<", a),
        where="hand.py:1",
    )
    term = Term(
        name="below",
        params=(("a", Real), ("y", ArrType(axes=(n,), dtype=real, ragged=(False,)))),
        sizes=("n",),
        stmts=(stmt,),
        post=None,
    )
    with pytest.raises(InterpretError, match="parameter a is 2.5, which is not"):
        interpret(term, {"a": 2.5, "y": Arr.zeros(4)})
    y = Arr.zeros(4)
    interpret(term, {"a": 2.0, "y": y})
    assert list(y.numpy()) == [1.0, 1.0, 0.0, 0.0]
