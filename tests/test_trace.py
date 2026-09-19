"""Tracing: what running a body once against a generic point records."""

from __future__ import annotations

import islpy as isl
import pytest
from lanky.prelude import Nat, Real
from lanky.terms import evaluate_annotations, render

from loopty import Arr, Fin, when
from loopty import sum as reduce_sum
from loopty.term import Reduction
from loopty.trace import TraceError, accesses_in, reductions_in, trace


def term_of(fn):
    """Trace a plain function the way ``@kernel`` would."""
    return trace(fn, evaluate_annotations(fn))


def axpy(a: Real, x: Arr[Fin[n], Real], y: Arr[Fin[n], Real]):  # noqa: F821
    for i in y.dom:
        y[i] = a * x[i] + y[i]


def spmv(
    cnt: Arr[Fin[n], Nat],  # noqa: F821
    col: Arr[Fin[n], Fin[cnt], Fin[m]],  # noqa: F821
    val: Arr[Fin[n], Fin[cnt], Real],  # noqa: F821
    x: Arr[Fin[m], Real],  # noqa: F821
    y: Arr[Fin[n], Real],  # noqa: F821
):
    for r in y.dom:
        y[r] = reduce_sum(val[r, j] * x[col[r, j]] for j in val.dom[r])


def guarded(u: Arr[Fin[n], Real], v: Arr[Fin[n], Real]):  # noqa: F821
    for i in u.dom:
        with when(i + 1 < u.dom.size):
            v[i] = u[i + 1]


def test_the_iname_is_the_name_in_the_source() -> None:
    term = term_of(axpy)
    assert term.stmts[0].inames == ("i",)
    assert term.name == "axpy"


def test_sizes_come_from_the_array_types_and_exclude_parameters() -> None:
    assert term_of(axpy).sizes == ("n",)
    # cnt is a parameter, not a size, even though it bounds an axis.
    assert term_of(spmv).sizes == ("m", "n")


def test_a_second_axis_naming_a_counts_parameter_is_ragged() -> None:
    params = dict(term_of(spmv).params)
    assert params["val"].ragged == (False, True)
    assert params["x"].ragged == (False,)
    assert render(params["val"].axes[1]) == "cnt"


def test_reading_the_assignee_at_the_same_cell_is_an_accumulation() -> None:
    assert term_of(axpy).stmts[0].kind == "accumulate"
    assert term_of(spmv).stmts[0].kind == "assign"


def test_the_domain_is_the_loop_nest_as_an_isl_set() -> None:
    domain = term_of(axpy).stmts[0].domain
    assert domain.is_subset(type(domain)("[n] -> { [i] : 0 <= i < n }"))


def test_a_reduction_carries_its_own_domain_and_exactness() -> None:
    stmt = term_of(spmv).stmts[0]
    reductions = reductions_in(stmt.expr)
    assert len(reductions) == 1
    reduction = reductions[0]
    assert isinstance(reduction, Reduction)
    assert reduction.op == "sum"
    assert reduction.inames == ("j",)
    # The class comes from what is summed: ``val`` and ``x`` are ``Real``.
    assert reduction.exactness == "approx"
    # The enclosing iname is a dimension of the reduced space, and the ragged
    # bound of the row has become a parameter.
    assert reduction.domain.dim(isl.dim_type.set) == 2
    assert "nl_cnt_r" in str(reduction.domain)


def test_accesses_are_read_off_the_expression() -> None:
    stmt = term_of(spmv).stmts[0]
    outer = {access.array for access in accesses_in(stmt.expr, into_reductions=False)}
    inner = {access.array for access in accesses_in(stmt.expr)}
    assert outer == set()
    assert inner == {"val", "x", "col"}


def test_a_when_block_narrows_the_domain_and_is_kept() -> None:
    stmt = term_of(guarded).stmts[0]
    assert stmt.guard is not None
    # 0 <= i < n and i + 1 < n, so the last point is gone.
    assert stmt.domain.is_equal(type(stmt.domain)("[n] -> { [i] : 0 <= i < n - 1 }"))


def test_an_if_on_a_symbolic_value_names_when_as_the_fix() -> None:
    def branchy(u: Arr[Fin[n], Real]):  # noqa: F821
        for i in u.dom:
            if i > 0:
                u[i] = 1.0

    with pytest.raises(TraceError, match="when"):
        term_of(branchy)


def test_a_break_out_of_a_traced_loop_is_refused() -> None:
    # The loop level is popped when the iterator raises StopIteration, which a
    # break skips. Every statement after the loop would then be recorded under a
    # loop variable the body has left.
    def early(u: Arr[Fin[n], Real], v: Arr[Fin[n], Real]):  # noqa: F821
        for i in u.dom:
            v[i] = u[i]
            break

    with pytest.raises(TraceError, match="left early"):
        term_of(early)


def test_a_return_from_inside_a_traced_loop_is_refused() -> None:
    def returns(u: Arr[Fin[n], Real], v: Arr[Fin[n], Real]):  # noqa: F821
        for i in u.dom:
            v[i] = u[i]
            return

    with pytest.raises(TraceError, match="left early"):
        term_of(returns)


def test_a_break_out_of_an_inner_loop_is_refused_by_name() -> None:
    # The nested case is the one that used to go quietly wrong: the outer
    # loop's own StopIteration popped the abandoned inner level, so the
    # statement after the inner loop carried a stale iname.
    def nested(u: Arr[Fin[n], Fin[m], Real], v: Arr[Fin[n], Real]):  # noqa: F821
        for i in v.dom:
            for j in u.dom[i]:
                v[i] = u[i, j]
                break
            v[i] = v[i] + 1.0

    with pytest.raises(TraceError, match="left early"):
        term_of(nested)


def test_the_message_names_when_as_the_fix() -> None:
    def early(u: Arr[Fin[n], Real], v: Arr[Fin[n], Real]):  # noqa: F821
        for i in u.dom:
            v[i] = u[i]
            break

    with pytest.raises(TraceError, match="with when"):
        term_of(early)


def test_a_loop_that_runs_to_the_end_still_traces() -> None:
    # The guard against early exits must not fire on an ordinary nest.
    term = term_of(spmv)
    assert [stmt.inames for stmt in term.stmts] == [("r",)]


def test_asking_a_symbolic_domain_for_its_length_is_an_error() -> None:
    def sized(u: Arr[Fin[n], Real]):  # noqa: F821
        for i in u.dom:
            u[i] = len(u.dom)

    with pytest.raises(TraceError, match="symbolic"):
        term_of(sized)


def test_statements_are_recorded_in_source_order_with_their_position() -> None:
    def two(cnt: Arr[Fin[n], Nat], off: Arr[Fin[n + 1], Nat]):  # noqa: F821
        off[0] = 0
        for r in cnt.dom:
            off[r + 1] = off[r] + cnt[r]

    term = term_of(two)
    assert [stmt.id for stmt in term.stmts] == ["S0", "S1"]
    assert term.stmts[0].order == (0,)
    assert term.stmts[1].order == (1, 0)
    assert term.stmts[0].inames == ()
    assert term.stmts[1].inames == ("r",)


def test_when_masks_writes_and_out_of_range_reads_under_plain_python() -> None:
    from loopty.kernel import Kernel

    values = Arr.from_numpy([1.0, 2.0, 3.0])
    out = Arr.zeros(3)
    Kernel(guarded)(values, out)
    assert list(out.numpy()) == [2.0, 3.0, 0.0]


def test_the_builtin_sum_over_a_symbolic_domain_names_the_replacement() -> None:
    def naive(x: Arr[Fin[n], Real], y: Arr[Fin[n], Real]):  # noqa: F821
        for i in y.dom:
            y[i] = sum(x[j] for j in x.dom)

    with pytest.raises(TraceError, match="loopty.sum"):
        term_of(naive)


def test_an_augmented_assignment_is_an_accumulation() -> None:
    def add_into(x: Arr[Fin[n], Real], y: Arr[Fin[n], Real]):  # noqa: F821
        for i in y.dom:
            y[i] += x[i]

    stmt = term_of(add_into).stmts[0]
    assert stmt.kind == "accumulate"
    assert render(stmt.expr) == "y[i] + x[i]"


# {{{ the parameters a trace reflects its non-affine bounds into


def test_the_reflected_bound_is_recorded_on_the_term() -> None:
    term = term_of(spmv)
    reflected = dict(term.reflected)
    assert list(reflected) == ["nl_cnt_r"]
    assert render(reflected["nl_cnt_r"]) == "cnt[r]"


def test_the_same_bound_in_two_statements_is_one_parameter() -> None:
    # Both statements run over the same fiber, so both domains are bounded by
    # ``cnt[r]``. One term, one parameter: two would be two unrelated unknowns
    # and nothing would relate the loop the statements share.
    def two_writes(
        cnt: Arr[Fin[n], Nat],  # noqa: F821
        val: Arr[Fin[n], Fin[cnt], Real],  # noqa: F821
        y: Arr[Fin[n], Real],  # noqa: F821
        z: Arr[Fin[n], Real],  # noqa: F821
    ):
        for r in y.dom:
            for j in val.dom[r]:
                y[r] = y[r] + val[r, j]
                z[r] = z[r] + 2.0 * val[r, j]

    term = term_of(two_writes)
    assert [name for name, _ in term.reflected] == ["nl_cnt_r"]
    assert len(term.stmts) == 2
    for stmt in term.stmts:
        assert "nl_cnt_r" in stmt.domain.get_var_names(isl.dim_type.param)


def test_a_size_spelled_like_a_reflected_bound_keeps_its_own_parameter() -> None:
    # ``nl_cnt_r`` is the name the bound ``cnt[r]`` would like. Here a size is
    # already called that, and giving the bound the same name would assert that
    # the length of a row equals the length of ``w``.
    def shadowed(
        cnt: Arr[Fin[n], Nat],  # noqa: F821
        val: Arr[Fin[n], Fin[cnt], Real],  # noqa: F821
        w: Arr[Fin[nl_cnt_r], Real],  # noqa: F821
        y: Arr[Fin[n], Real],  # noqa: F821
    ):
        for r in y.dom:
            y[r] = reduce_sum(val[r, j] * w[0] for j in val.dom[r])

    term = term_of(shadowed)
    assert "nl_cnt_r" in term.sizes
    reflected = dict(term.reflected)
    assert "nl_cnt_r" not in reflected
    (name,) = reflected
    assert name.startswith("nl_cnt_r")
    assert render(reflected[name]) == "cnt[r]"

    (stmt,) = term.stmts
    (reduction,) = reductions_in(stmt.expr)
    params = reduction.domain.get_var_names(isl.dim_type.param)
    assert name in params
    assert "nl_cnt_r" not in params


# }}}
