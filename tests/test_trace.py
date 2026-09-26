"""Tracing: what running a body once against a generic point records."""

from __future__ import annotations

import islpy as isl
import numpy as np
import pytest
from lanky.prelude import Nat, Real
from lanky.terms import evaluate_annotations, render

from loopty import Arr, Fin, reduce_sum, when
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


def scaled_counts(
    a: Real,
    x: Arr[Fin[n], Fin[m], Nat],  # noqa: F821
    y: Arr[Fin[n], Real],  # noqa: F821
):
    for r in y.dom:
        y[r] = reduce_sum(a * x[r, j] for j in x.dom[r])


def tenths(x: Arr[Fin[n], Fin[m], Nat], y: Arr[Fin[n], Real]):  # noqa: F821
    for r in y.dom:
        y[r] = reduce_sum(0.1 * j for j in x.dom[r])


def halves(x: Arr[Fin[n], Fin[m], Nat], y: Arr[Fin[n], Real]):  # noqa: F821
    for r in y.dom:
        y[r] = reduce_sum(x[r, j] / 2 for j in x.dom[r])


def weighted_counts(x: Arr[Fin[n], Fin[m], Nat], y: Arr[Fin[n], Nat]):  # noqa: F821
    for r in y.dom:
        y[r] = reduce_sum(2 * x[r, j] + j // 2 for j in x.dom[r])


def test_what_a_reduction_sums_decides_its_class_and_not_only_its_arrays() -> None:
    # Only the element sorts of the arrays read used to be joined, so each of
    # the first three was ``exact``: a ``Real`` scalar times an integer array, a
    # float literal times the binder, and a true division of an integer array.
    # A schedule then refused to reassociate sums that were never exact.
    def exactness(fn) -> str:
        (reduction,) = reductions_in(term_of(fn).stmts[0].expr)
        return reduction.exactness

    assert exactness(scaled_counts) == "approx"
    assert exactness(tenths) == "approx"
    assert exactness(halves) == "approx"
    # Integers under ``+``, ``*`` and ``//`` stay integers.
    assert exactness(weighted_counts) == "exact"


def shadowing_binders(a: Arr[Fin[n], Fin[n], Real], s: Arr[Fin[1], Real]):  # noqa: F821
    s[0] = reduce_sum(reduce_sum(a[i, i] for i in a.dom[i]) for i in a.dom)


def test_a_nested_reduction_may_not_reuse_the_outer_binder() -> None:
    with pytest.raises(TraceError, match="shadows the binder of the reduction"):
        term_of(shadowing_binders)


def fibers_by_tuple(u: Arr[Fin[nx], Fin[ny], Fin[nz], Real]):  # noqa: F821
    for t in u.dom:
        for i in u.dom[t]:
            for k in u.dom[t, i]:
                u[t, i, k] = 0.0


def fibers_by_chain(u: Arr[Fin[nx], Fin[ny], Fin[nz], Real]):  # noqa: F821
    for t in u.dom:
        for i in u.dom[t]:
            for k in u.dom[t][i]:
                u[t, i, k] = 0.0


def test_a_tuple_domain_index_is_one_index_per_axis() -> None:
    # ``u.dom[t, i]`` used to be read as one index, the tuple, and gave the
    # domain of axis 1 whatever its length: the loop over ``k`` ran over
    # ``Fin[ny]`` rather than ``Fin[nz]``. It is ``u.dom[t][i]``, as it is on a
    # runtime array.
    (stmt,) = term_of(fibers_by_tuple).stmts
    (want,) = term_of(fibers_by_chain).stmts
    assert stmt.domain.is_subset(want.domain)
    assert want.domain.is_subset(stmt.domain)
    box = isl.Set(
        "[nx, ny, nz] -> { [t, i, k] : 0 <= t < nx and 0 <= i < ny and 0 <= k < nz }"
    )
    assert stmt.domain.is_subset(box) and box.is_subset(stmt.domain)


def fiber_past_the_last_axis(
    cnt: Arr[Fin[n], Nat],  # noqa: F821
    val: Arr[Fin[n], Fin[cnt], Real],  # noqa: F821
):
    for r in val.dom:
        for j in val.dom[r, 0]:
            val[r, j] = 0.0


def test_a_tuple_domain_index_past_the_last_axis_is_refused() -> None:
    with pytest.raises(TraceError, match="no axis 2"):
        term_of(fiber_past_the_last_axis)


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

    with pytest.raises(TraceError, match="loopty.reduce_sum"):
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


# {{{ state a Python name carries from one iteration to the next


def running_sum(x: Arr[Fin[n], Real], y: Arr[Fin[1], Real]):  # noqa: F821
    s = 0.0
    for i in x.dom:
        s = s + x[i]
    y[0] = s


def test_a_running_sum_in_a_python_name_is_refused_with_both_fixes() -> None:
    # This used to trace to one statement, ``y[0] = 0.0 + x[i]``, with no loop
    # around it and ``i`` free, while the native run summed the array.
    with pytest.raises(TraceError) as caught:
        term_of(running_sum)
    message = str(caught.value)
    assert "'s'" in message
    assert "reduce_sum(... for i in x.dom)" in message
    assert "indexed cell" in message
    assert "s[i + 1] = s[i]" in message
    # The values the name had are part of the explanation.
    assert "0.0 + x[i]" in message


def test_a_carried_value_that_mentions_no_loop_variable_is_refused() -> None:
    # Nothing escapes here, so only comparing the locals can see it.
    def counts(x: Arr[Fin[n], Real], y: Arr[Fin[1], Real]):  # noqa: F821
        s = 0.0
        for i in x.dom:
            s = s + 1.0
        y[0] = s

    with pytest.raises(TraceError, match="carries 's'"):
        term_of(counts)


def test_an_augmented_assignment_to_a_name_is_a_rebinding() -> None:
    def augmented(x: Arr[Fin[n], Real], y: Arr[Fin[1], Real]):  # noqa: F821
        s = 0.0
        for i in x.dom:
            s += x[i]
        y[0] = s

    with pytest.raises(TraceError, match="carries 's'"):
        term_of(augmented)


def test_tuple_unpacking_rebinds_every_name_it_assigns() -> None:
    def fibonacci(x: Arr[Fin[n], Real], y: Arr[Fin[n], Real]):  # noqa: F821
        a, b = x[0], x[1]
        for i in y.dom:
            y[i] = a
            a, b = b, a + b

    with pytest.raises(TraceError, match="carries 'a', 'b'"):
        term_of(fibonacci)


def test_a_plain_python_counter_is_carried_state_too() -> None:
    # ``k`` is an int, not a term, and the statement inside the loop is
    # recorded at ``k = 1`` as if every iteration wrote ``y[1]``.
    def counter(x: Arr[Fin[n], Real], y: Arr[Fin[n + 1], Real]):  # noqa: F821
        k = 0
        for i in x.dom:
            k = k + 1
            y[k] = x[i]

    with pytest.raises(TraceError, match="carries 'k'.*'k' is 0 before the loop"):
        term_of(counter)


def test_a_loop_in_a_helper_is_checked_in_the_helper_frame() -> None:
    def total(x, y) -> None:
        acc = 0.0
        for i in x.dom:
            acc = acc + x[i]
        y[0] = acc

    def calls_a_helper(x: Arr[Fin[n], Real], y: Arr[Fin[1], Real]):  # noqa: F821
        total(x, y)

    with pytest.raises(TraceError, match="carries 'acc'"):
        term_of(calls_a_helper)


def test_a_name_a_closure_rebinds_through_nonlocal_is_carried_state() -> None:
    # ``s`` lives in a cell of the frame running the ``for``, and that frame's
    # locals show a cell's contents, so the closure's rebinding is seen there.
    def through_a_closure(x: Arr[Fin[n], Real], y: Arr[Fin[1], Real]):  # noqa: F821
        s = 0.0

        def add(value):
            nonlocal s
            s = s + value

        for i in x.dom:
            add(x[i])
        y[0] = s

    with pytest.raises(TraceError, match="carries 's'"):
        term_of(through_a_closure)


def test_swapping_two_arrays_between_time_steps_is_carried_state() -> None:
    # The trace would record every step as a copy from ``u`` into ``v``. The
    # arrays are named in the message as the body names them.
    def ping_pong(
        u: Arr[Fin[n], Real],  # noqa: F821
        v: Arr[Fin[n], Real],  # noqa: F821
        steps: Arr[Fin[k], Real],  # noqa: F821
    ):
        a, b = u, v
        for t in steps.dom:
            for i in a.dom:
                b[i] = a[i]
            a, b = b, a

    with pytest.raises(
        TraceError, match="'a' is u before the loop and v after one iteration"
    ):
        term_of(ping_pong)


def test_state_carried_by_an_inner_loop_names_that_loop() -> None:
    def rows(u: Arr[Fin[n], Fin[m], Real], v: Arr[Fin[n], Real]):  # noqa: F821
        for i in v.dom:
            acc = 0.0
            for j in u.dom[i]:
                acc = acc + u[i, j]
            v[i] = acc

    with pytest.raises(TraceError, match="loop over 'j'.*carries 'acc'") as caught:
        term_of(rows)
    assert "reduce_sum(... for j in u.dom[i])" in str(caught.value)


def test_a_value_carried_out_of_a_loop_is_an_escaped_loop_variable() -> None:
    # ``t`` is first bound inside the loop, so it is no rebinding; natively
    # ``y[0]`` is the last ``x[i]``, and the trace would write it about a free
    # ``i``. The statement is refused where it is recorded.
    def last(x: Arr[Fin[n], Real], y: Arr[Fin[1], Real]):  # noqa: F821
        for i in x.dom:
            t = x[i]
        y[0] = t

    with pytest.raises(
        TraceError, match="write to y\\[0\\].*loop variable 'i' of the loop at"
    ) as caught:
        term_of(last)
    assert "reduce_sum(... for i in x.dom)" in str(caught.value)


def test_an_escape_names_the_loop_as_the_body_wrote_it() -> None:
    # The inner ``for`` rebinds the outer target, so the ``i`` the outer body
    # reads after it is the inner loop's variable, which the trace calls
    # ``i_0``. The message names the loop by its target and its line.
    def shadowed(u: Arr[Fin[n], Fin[m], Real], v: Arr[Fin[n], Real]):  # noqa: F821
        for i in v.dom:
            for i in u.dom[i]:
                u[0, i] = 1.0
            v[i] = 2.0

    with pytest.raises(TraceError) as caught:
        term_of(shadowed)
    message = str(caught.value)
    assert "loop variable 'i' of the loop at test_trace.py:" in message
    assert "(i_0 in the trace)" in message
    assert "reduce_sum(... for i in u.dom[i])" in message


def test_a_loop_variable_escaping_into_a_guard_is_refused() -> None:
    def guarded_after(x: Arr[Fin[n], Real], y: Arr[Fin[n], Real]):  # noqa: F821
        for i in x.dom:
            y[i] = x[i]
        with when(i > 0):
            y[0] = 1.0

    with pytest.raises(TraceError, match="loop variable 'i'"):
        term_of(guarded_after)


def test_a_loop_variable_escaping_into_a_loop_bound_is_refused() -> None:
    # Nothing the statement computes mentions ``r``; its domain does, through
    # the extent ``cnt[r]`` of the fiber the second loop runs over.
    def fiber_after(
        cnt: Arr[Fin[n], Nat],  # noqa: F821
        val: Arr[Fin[n], Fin[cnt], Real],  # noqa: F821
        y: Arr[Fin[n], Real],  # noqa: F821
    ):
        for r in y.dom:
            y[r] = 0.0
        for j in val.dom[r]:
            y[j] = 1.0

    with pytest.raises(TraceError, match="loop variable 'r'"):
        term_of(fiber_after)


def test_a_loop_variable_escaping_into_a_reduction_bound_is_refused() -> None:
    # The reduction's body is a constant; ``r`` survives only in the parameter
    # its bound ``cnt[r]`` is reflected into.
    def reduced_after(
        cnt: Arr[Fin[n], Nat],  # noqa: F821
        val: Arr[Fin[n], Fin[cnt], Real],  # noqa: F821
        y: Arr[Fin[n], Real],  # noqa: F821
    ):
        for r in y.dom:
            y[r] = 0.0
        y[0] = reduce_sum(1.0 for j in val.dom[r])

    with pytest.raises(TraceError, match="loop variable 'r'"):
        term_of(reduced_after)


def test_a_reduction_binder_does_not_bind_its_own_bound() -> None:
    # Python evaluates ``val.dom[r]`` before the generator binds its own ``r``,
    # so the ``r`` in the bound is the closed loop's variable. The lowered
    # domain has one dimension ``r`` for the two, which reads as bound; the
    # reduction as the body built it still tells them apart.
    def rebound_after(
        cnt: Arr[Fin[n], Nat],  # noqa: F821
        val: Arr[Fin[n], Fin[cnt], Real],  # noqa: F821
        y: Arr[Fin[n], Real],  # noqa: F821
    ):
        for r in y.dom:
            y[r] = 0.0
        y[0] = reduce_sum(1.0 for r in val.dom[r])

    with pytest.raises(TraceError, match="loop variable 'r' of the loop at"):
        term_of(rebound_after)


def test_a_reduction_binder_binds_the_bounds_of_the_binders_after_it() -> None:
    # ``k``'s bound ``cnt[j]`` is the binder before it, not the closed loop's
    # ``j`` that shares its name.
    def nested_sum(
        cnt: Arr[Fin[n], Nat],  # noqa: F821
        val: Arr[Fin[n], Fin[cnt], Real],  # noqa: F821
        y: Arr[Fin[n], Real],  # noqa: F821
    ):
        for j in y.dom:
            y[j] = 0.0
        y[0] = reduce_sum(val[j, k] for j in val.dom for k in val.dom[j])

    _, total = term_of(nested_sum).stmts
    assert [r.inames for r in reductions_in(total.expr)] == [("j", "k")]


def test_a_per_iteration_temporary_still_traces() -> None:
    def temporary(x: Arr[Fin[n], Real], y: Arr[Fin[n], Real]):  # noqa: F821
        for i in x.dom:
            t = x[i] * 2
            y[i] = t + 1

    (stmt,) = term_of(temporary).stmts
    assert stmt.inames == ("i",)
    assert render(stmt.expr) == "x[i]*2 + 1"


def test_names_the_loop_reads_or_rebinds_to_the_same_value_still_trace() -> None:
    # ``c`` is read and never rebound; ``out`` is rebound to the very object it
    # held; ``two`` to a new term equal to the old one, which only a structural
    # comparison can see (``==`` on a term builds a proposition, and asking it
    # for a truth value would be refused as an ``if``); ``f`` to a new float
    # equal to the old one.
    def rebinds(a: Real, x: Arr[Fin[n], Real], y: Arr[Fin[n], Real]):  # noqa: F821
        c = 2.0
        out = y
        two = a * 2
        f = 1.5
        for i in x.dom:
            out = y
            two = a * 2
            f = f * 1.0
            out[i] = c * two * f * x[i]

    (stmt,) = term_of(rebinds).stmts
    assert stmt.inames == ("i",)


def test_a_scan_written_with_an_indexed_cell_still_traces() -> None:
    # The fix the message names for state that is not an accumulation.
    def scan(x: Arr[Fin[n], Real], s: Arr[Fin[n + 1], Real]):  # noqa: F821
        s[0] = 0.0
        for i in x.dom:
            s[i + 1] = s[i] + x[i]

    first, step = term_of(scan).stmts
    assert first.inames == ()
    assert step.inames == ("i",)
    assert render(step.expr) == "s[i] + x[i]"


def test_a_running_sum_written_as_a_reduction_still_traces() -> None:
    # The fix the message names for an accumulation.
    def total(x: Arr[Fin[n], Real], y: Arr[Fin[1], Real]):  # noqa: F821
        y[0] = reduce_sum(x[i] for i in x.dom)

    (stmt,) = term_of(total).stmts
    assert stmt.inames == ()
    assert [r.inames for r in reductions_in(stmt.expr)] == [("i",)]


def test_a_target_reused_by_a_sibling_loop_is_not_state() -> None:
    # After the first loop ``i`` is still bound, to that loop's variable; the
    # second ``for`` rebinds it, which is what a ``for`` does.
    def siblings(x: Arr[Fin[n], Real], y: Arr[Fin[n], Real]):  # noqa: F821
        for i in x.dom:
            y[i] = x[i]
        for i in y.dom:
            x[i] = 2.0 * y[i]

    term = term_of(siblings)
    assert [stmt.inames for stmt in term.stmts] == [("i",), ("i_0",)]


def test_a_target_reused_by_a_nested_loop_is_not_state() -> None:
    # ``j`` is bound before the loop over ``i`` (to the first loop's variable)
    # and rebound inside it by the inner ``for``; so is the inner target of
    # two sibling loops nested in one outer loop.
    def nested(
        u: Arr[Fin[n], Fin[m], Real],  # noqa: F821
        v: Arr[Fin[n], Real],  # noqa: F821
        w: Arr[Fin[m], Real],  # noqa: F821
    ):
        for j in w.dom:
            w[j] = 0.0
        for i in v.dom:
            for j in u.dom[i]:
                u[i, j] = v[i]
            for j in u.dom[i]:
                u[i, j] = u[i, j] + w[j]

    term = term_of(nested)
    assert [stmt.inames for stmt in term.stmts] == [
        ("j",),
        ("i", "j_0"),
        ("i", "j_1"),
    ]


def test_a_loop_target_named_like_a_size_gets_its_own_iname() -> None:
    # ``k`` is also the size of both arrays. The iname used to be ``k`` too,
    # one isl dimension for the two, so the first loop's domain was
    # ``0 <= k < k``, which is empty, and the second loop's bound ``k`` read as
    # the first loop's variable escaping.
    def sized(x: Arr[Fin[k], Real], y: Arr[Fin[k], Real]):  # noqa: F821
        for k in x.dom:
            x[k] = 0.0
        for i in y.dom:
            y[i] = 1.0

    first, second = term_of(sized).stmts
    assert first.inames == ("k_0",)
    assert not first.domain.is_empty()
    assert second.inames == ("i",)


def test_a_loop_written_on_one_line_keeps_its_source_name() -> None:
    # From Python 3.13 on the store of the ``for`` target is fused with the load
    # after it when both are on one line, and the target used to go unread: the
    # iname became ``i0``, and the message told the author to write
    # ``for i0 in x.dom``.
    def one_line(x: Arr[Fin[n], Real], y: Arr[Fin[n], Real]):  # noqa: F821
        for i in x.dom: y[i] = x[i]  # noqa: E701

    def carried(x: Arr[Fin[n], Real], y: Arr[Fin[1], Real]):  # noqa: F821
        s = 0.0
        for i in x.dom: s = s + x[i]  # noqa: E701
        y[0] = s

    assert term_of(one_line).stmts[0].inames == ("i",)
    with pytest.raises(TraceError) as caught:
        term_of(carried)
    assert "reduce_sum(... for i in x.dom)" in str(caught.value)


def test_what_guards_and_reductions_bind_is_not_state() -> None:
    # ``g`` is rebound to a new guard object each time (and read at the end, so
    # it is live across the loops); the reduction's ``j`` lives in the
    # generator's own frame, although it shares a name with an earlier loop's
    # target; ``dx`` is a temporary bound inside a guard.
    def binders(
        cnt: Arr[Fin[n], Nat],  # noqa: F821
        val: Arr[Fin[n], Fin[cnt], Real],  # noqa: F821
        y: Arr[Fin[n], Real],  # noqa: F821
        z: Arr[Fin[n], Real],  # noqa: F821
    ):
        with when(y.dom.size > 1) as g:
            z[0] = 0.0
        for j in z.dom:
            with when(j > 0) as g:
                dx = y[j] - y[j - 1]
                z[j] = dx
        for r in y.dom:
            y[r] = reduce_sum(val[r, j] for j in val.dom[r])
        assert g is not None

    term = term_of(binders)
    assert [stmt.inames for stmt in term.stmts] == [(), ("j",), ("r",)]


def test_plain_python_runs_the_carried_sum_as_written() -> None:
    # Both checks are trace-time only; the native run is the reference.
    from loopty.kernel import Kernel

    x = Arr.from_numpy([0.0, 1.0, 2.0, 3.0])
    y = Arr.zeros(1)
    Kernel(running_sum)(x, y)
    assert list(y.numpy()) == [6.0]


# }}}


# {{{ state a list, a dict, a set or a global carries


#: Module globals for the kernels below to carry state in, or to bind as a loop
#: target. Each test sets its own through ``monkeypatch``, so a trace that
#: changes one leaves nothing behind for the next.
_COUNT = 0
_TOTALS = [0.0]
point = 0


def test_a_counter_kept_in_a_list_is_refused() -> None:
    # ``state`` is bound to the same list before and after the iteration, so
    # comparing the names saw nothing, and the trace recorded ``y[i] = 1`` for
    # every ``i`` while the native run writes 1, 2, 3, ...
    def counted(x: Arr[Fin[n], Real], y: Arr[Fin[n], Real]):  # noqa: F821
        state = [0]
        for i in x.dom:
            state[0] += 1
            y[i] = state[0]

    with pytest.raises(TraceError) as caught:
        term_of(counted)
    message = str(caught.value)
    assert "loop over 'i' at test_trace.py:" in message
    assert "carries the list 'state'" in message
    assert "'state[0]' is 0 before the loop and 1 after one iteration" in message
    assert "reduce_sum(... for i in x.dom)" in message
    assert "state[i + 1] = state[i] + ..." in message
    assert "If 'state' is only scratch" in message
    assert "create it inside the loop" in message


def test_an_accumulator_kept_in_a_dict_is_refused() -> None:
    # A running sum through a dict entry: the trace saw ``y[i] = 0.0 + x[i]``.
    def running(x: Arr[Fin[n], Real], y: Arr[Fin[n], Real]):  # noqa: F821
        acc = {"total": 0.0}
        for i in x.dom:
            acc["total"] = acc["total"] + x[i]
            y[i] = acc["total"]

    with pytest.raises(TraceError) as caught:
        term_of(running)
    message = str(caught.value)
    assert "carries the dict 'acc'" in message
    assert "\"acc['total']\" is 0.0 before the loop and 0.0 + x[i]" in message


def test_a_set_that_grows_across_iterations_is_refused() -> None:
    # Natively only the first iteration doubles; the one point the trace runs
    # is a first iteration, so every ``y[i]`` was recorded as ``2.0*x[i]``.
    def first_doubled(x: Arr[Fin[n], Real], y: Arr[Fin[n], Real]):  # noqa: F821
        done = set()
        for i in x.dom:
            y[i] = x[i] * (1.0 if "doubled" in done else 2.0)
            done.add("doubled")

    with pytest.raises(TraceError) as caught:
        term_of(first_doubled)
    message = str(caught.value)
    assert "carries the set 'done'" in message
    assert "'done' is set() before the loop and {'doubled'} after one" in message


def test_a_list_kept_in_a_tuple_is_reached_through_it() -> None:
    def through_a_tuple(x: Arr[Fin[n], Real], y: Arr[Fin[n], Real]):  # noqa: F821
        counters = ([0], [0])
        for i in x.dom:
            counters[1][0] += 1
            y[i] = counters[1][0] * x[i]

    with pytest.raises(TraceError, match=r"carries the list 'counters\[1\]'"):
        term_of(through_a_tuple)


def test_a_global_the_body_rebinds_is_refused(monkeypatch) -> None:
    monkeypatch.setitem(globals(), "_COUNT", 0)

    def numbered(x: Arr[Fin[n], Real], y: Arr[Fin[n], Real]):  # noqa: F821
        global _COUNT
        for i in x.dom:
            _COUNT = _COUNT + 1
            y[i] = _COUNT * x[i]

    with pytest.raises(TraceError) as caught:
        term_of(numbered)
    message = str(caught.value)
    assert "carries the global '_COUNT'" in message
    assert "'_COUNT' is 0 before the loop and 1 after one iteration" in message
    assert "make it a local name that is first bound inside the loop" in message


def test_a_global_list_the_body_mutates_is_refused(monkeypatch) -> None:
    # No ``global`` statement: the list is read by name and changed in place.
    monkeypatch.setitem(globals(), "_TOTALS", [0.0])

    def totals(x: Arr[Fin[n], Real], y: Arr[Fin[n], Real]):  # noqa: F821
        for i in x.dom:
            _TOTALS[0] = _TOTALS[0] + 1.0
            y[i] = _TOTALS[0] * x[i]

    with pytest.raises(TraceError) as caught:
        term_of(totals)
    message = str(caught.value)
    assert "carries the global list '_TOTALS'" in message
    assert "'_TOTALS[0]' is 0.0 before the loop and 1.0 after" in message


def test_a_global_loop_target_is_not_state(monkeypatch) -> None:
    # ``global point`` makes the ``for`` store its target with STORE_GLOBAL,
    # and it rebinds that target before every iteration, as it does a local
    # one; the module already binding ``point`` used to make this refused.
    monkeypatch.setitem(globals(), "point", 0)

    def global_target(x: Arr[Fin[n], Real], y: Arr[Fin[n], Real]):  # noqa: F821
        global point
        for point in x.dom:
            y[point] = x[point]

    (stmt,) = term_of(global_target).stmts
    assert stmt.inames == ("point",)


def test_a_global_named_like_a_local_loop_target_is_still_state(
    monkeypatch,
) -> None:
    # The ``for`` binds a local ``point``; the helper rebinds the module's
    # ``point``, a different name that only shares the spelling. The trace
    # recorded ``y[point] = 1*x[point]`` while the native run scales by 1, 2, 3.
    monkeypatch.setitem(globals(), "point", 0)

    def shadowing(x: Arr[Fin[n], Real], y: Arr[Fin[n], Real]):  # noqa: F821
        def bump():
            global point
            point += 1
            return point

        for point in x.dom:
            y[point] = bump() * x[point]

    with pytest.raises(
        TraceError, match="carries the global 'point'.*'point' is 0 before the loop"
    ):
        term_of(shadowing)


def test_a_name_or_global_deleted_inside_the_loop_is_carried_state(
    monkeypatch,
) -> None:
    # Natively the second iteration fails on the name the first one deleted;
    # the one point the trace runs is a first iteration, which finds it bound,
    # so the trace recorded ``y[i] = 2.0*x[i]`` for every ``i``.
    monkeypatch.setitem(globals(), "_COUNT", 2)

    def scaled_once(x: Arr[Fin[n], Real], y: Arr[Fin[n], Real]):  # noqa: F821
        scale = 2.0
        for i in x.dom:
            y[i] = scale * x[i]
            del scale

    def counted_once(x: Arr[Fin[n], Real], y: Arr[Fin[n], Real]):  # noqa: F821
        global _COUNT
        for i in x.dom:
            y[i] = _COUNT * x[i]
            del _COUNT

    with pytest.raises(
        TraceError, match="'scale' is 2.0 before the loop and unbound after"
    ):
        term_of(scaled_once)
    with pytest.raises(
        TraceError,
        match="carries the global '_COUNT'.*'_COUNT' is 2 before the loop and unbound",
    ):
        term_of(counted_once)


def test_a_name_first_bound_by_a_test_on_locals_is_refused() -> None:
    # ``s`` is first bound inside the loop, which reads as a temporary, but only
    # the first iteration binds it: the trace recorded ``y[i] = 1`` while the
    # native run writes 1, 2, 3, ...
    def counted(x: Arr[Fin[n], Real], y: Arr[Fin[n], Real]):  # noqa: F821
        for i in x.dom:
            if "s" not in locals():
                s = 0
            s += 1
            y[i] = s

    with pytest.raises(TraceError) as caught:
        term_of(counted)
    message = str(caught.value)
    assert "the code running the loop over 'i' at test_trace.py:" in message
    assert "uses locals()" in message
    assert "a kernel body may not inspect which names are bound" in message
    assert "reduce_sum(... for i in x.dom)" in message
    assert "s[i + 1] = s[i] + ..." in message
    assert "bind it in every iteration instead of testing for it" in message


def test_a_name_first_bound_under_except_name_error_is_refused() -> None:
    def counted(x: Arr[Fin[n], Real], y: Arr[Fin[n], Real]):  # noqa: F821
        for i in x.dom:
            try:
                s += 1
            except (NameError, UnboundLocalError):
                s = 1
            y[i] = s

    with pytest.raises(
        TraceError, match="uses NameError and UnboundLocalError, and a kernel body"
    ):
        term_of(counted)


def test_a_name_first_bound_by_a_test_on_vars_is_refused() -> None:
    # ``vars()`` with no argument is ``locals()`` by another name.
    def counted(x: Arr[Fin[n], Real], y: Arr[Fin[n], Real]):  # noqa: F821
        for i in x.dom:
            if "s" not in vars():
                s = 0
            s += 1
            y[i] = s

    with pytest.raises(TraceError, match=r"uses vars\(\), and a kernel body"):
        term_of(counted)


def test_a_local_named_like_a_probe_is_not_a_probe() -> None:
    # ``locals`` here is the body's own function; nothing asks what is bound.
    def shadowed(x: Arr[Fin[n], Real], y: Arr[Fin[n], Real]):  # noqa: F821
        def locals():
            return (2.0,)

        for i in x.dom:
            y[i] = locals()[0] * x[i]

    (stmt,) = term_of(shadowed).stmts
    assert stmt.inames == ("i",)
    assert render(stmt.expr) == "2.0*x[i]"


def test_an_attribute_named_like_a_probe_is_not_a_probe() -> None:
    # pytest rewrites the ``assert`` into code that calls
    # ``@py_builtins.locals()``, which puts ``locals`` among the body's names
    # as an attribute; ``cfg.vars`` is an attribute too.
    from types import SimpleNamespace

    cfg = SimpleNamespace(vars=2.0)

    def configured(x: Arr[Fin[n], Real], y: Arr[Fin[n], Real]):  # noqa: F821
        for i in x.dom:
            y[i] = cfg.vars * x[i]
        assert cfg.vars > 0

    (stmt,) = term_of(configured).stmts
    assert render(stmt.expr) == "2.0*x[i]"


def test_a_list_created_inside_the_loop_is_scratch() -> None:
    # First bound inside the loop, like a per-iteration temporary name.
    def paired(x: Arr[Fin[n], Real], y: Arr[Fin[n], Real]):  # noqa: F821
        for i in x.dom:
            pair = [x[i], 2.0 * x[i]]
            pair[0] = pair[0] + pair[1]
            y[i] = pair[0]

    (stmt,) = term_of(paired).stmts
    assert stmt.inames == ("i",)
    assert [access.array for access in accesses_in(stmt.expr)] == ["x", "x"]


def test_a_list_the_loop_only_reads_still_traces() -> None:
    def weighted(x: Arr[Fin[n], Real], y: Arr[Fin[n], Real]):  # noqa: F821
        coeffs = [0.5, 2.0]
        for i in x.dom:
            y[i] = coeffs[0] * x[i] + coeffs[1]

    (stmt,) = term_of(weighted).stmts
    assert stmt.inames == ("i",)


def test_a_list_holding_a_closed_loops_values_is_not_state() -> None:
    # ``buf`` holds ``x[i]`` from the first loop, which the second overwrites
    # as scratch; reading the old value would be refused as an escaped ``i``.
    def reused(
        x: Arr[Fin[n], Real],  # noqa: F821
        y: Arr[Fin[n], Real],  # noqa: F821
        z: Arr[Fin[n], Real],  # noqa: F821
    ):
        for i in x.dom:
            buf = [x[i]]
            y[i] = buf[0]
        for j in x.dom:
            buf[0] = 2.0 * x[j]
            z[j] = buf[0]

    term = term_of(reused)
    assert [stmt.inames for stmt in term.stmts] == [("i",), ("j",)]


def test_plain_python_runs_the_list_counter_as_written() -> None:
    # The refusal is trace-time only; the native run is the reference.
    from loopty.kernel import Kernel

    def counted(x: Arr[Fin[n], Real], y: Arr[Fin[n], Real]):  # noqa: F821
        state = [0]
        for i in x.dom:
            state[0] += 1
            y[i] = state[0]

    x = Arr.from_numpy([0.0, 0.0, 0.0])
    y = Arr.zeros(3)
    Kernel(counted)(x, y)
    assert list(y.numpy()) == [1.0, 2.0, 3.0]


# }}}


# {{{ operations on a whole array


def test_a_slice_assignment_is_refused_with_the_loop_that_does_it() -> None:
    # ``y[:] = 0.0`` used to be recorded as one statement whose index was a
    # slice, which nothing downstream reads as a loop.
    def zeroed(y: Arr[Fin[n], Real]):  # noqa: F821
        y[:] = 0.0

    with pytest.raises(TraceError) as caught:
        term_of(zeroed)
    message = str(caught.value)
    assert "y[:] = ... at test_trace.py:" in message
    assert "an operation on the whole array y" in message
    assert "for i in y.dom: y[i] = ..." in message


def test_arithmetic_on_a_whole_array_is_refused() -> None:
    # ``x * 2`` used to fail with numpy-free Python's "unsupported operand".
    def doubled(x: Arr[Fin[n], Real], y: Arr[Fin[n], Real]):  # noqa: F821
        z = x * 2
        for i in y.dom:
            y[i] = z[i]

    with pytest.raises(TraceError, match=r"x \* \.\.\. at test_trace.py:\d+ is an"):
        term_of(doubled)


def test_fewer_indices_than_axes_name_a_row_and_are_refused() -> None:
    # ``u[t]`` of a two-axis array is a whole row, natively as in numpy.
    def rows(u: Arr[Fin[nt], Fin[nx], Real]):  # noqa: F821
        for t in u.dom:
            u[t] = 0.0

    with pytest.raises(TraceError) as caught:
        term_of(rows)
    message = str(caught.value)
    assert "u[t] = ... at test_trace.py:" in message
    assert "gives 1 of the 2 indices of u" in message
    assert "for i in u.dom: for j in u.dom[i]: u[i, j] = ..." in message


def test_iterating_an_array_itself_is_refused() -> None:
    # A symbolic array answers any index, so ``for v in x`` used to walk it
    # forever through the old sequence protocol; the ``break`` keeps this test
    # finite on a tracer without the refusal.
    def first(x: Arr[Fin[n], Real], y: Arr[Fin[1], Real]):  # noqa: F821
        for v in x:
            y[0] = v
            break

    with pytest.raises(TraceError, match="iterating x itself at test_trace.py"):
        term_of(first)


def test_numpy_functions_of_a_whole_array_are_refused() -> None:
    import numpy as np

    def total(x: Arr[Fin[n], Real], y: Arr[Fin[1], Real]):  # noqa: F821
        y[0] = np.sum(x)

    def roots(x: Arr[Fin[n], Real], y: Arr[Fin[n], Real]):  # noqa: F821
        z = np.sqrt(x)
        for i in y.dom:
            y[i] = z[i]

    def storage(x: Arr[Fin[n], Real], y: Arr[Fin[n], Real]):  # noqa: F821
        z = x.numpy()
        for i in y.dom:
            y[i] = z[i]

    with pytest.raises(TraceError, match="numpy.sum of x is an operation"):
        term_of(total)
    with pytest.raises(TraceError, match="numpy.sqrt of x is an operation"):
        term_of(roots)
    with pytest.raises(TraceError, match="asks for the storage of the array"):
        term_of(storage)


def test_a_fiber_over_a_slice_is_refused() -> None:
    def tail(u: Arr[Fin[nt], Fin[nx], Real]):  # noqa: F821
        for t in u.dom:
            for i in u.dom[1:]:
                u[t, i] = 0.0

    with pytest.raises(TraceError, match=r"u.dom\[1:\] at .* over more than one"):
        term_of(tail)


# }}}


# {{{ a reduction's condition


def test_a_reduction_condition_that_reads_data_is_refused() -> None:
    # The condition used to be dropped from the domain without a word, and the
    # term summed every x[j] where the body sums the positive ones.
    def positive(x: Arr[Fin[n], Real], y: Arr[Fin[1], Real]):  # noqa: F821
        y[0] = reduce_sum(x[j] for j in x.dom if x[j] > 0)

    with pytest.raises(TraceError) as caught:
        term_of(positive)
    message = str(caught.value)
    assert "the condition 'x[j] > 0' of the reduction over j at test_trace.py:" in (
        message
    )
    assert "reads an array or is not affine" in message
    assert "'with when(condition):'" in message


def test_a_reduction_condition_with_not_equal_is_refused() -> None:
    def off_diagonal(a: Arr[Fin[n], Fin[n], Real], y: Arr[Fin[n], Real]):  # noqa: F821
        for i in y.dom:
            y[i] = reduce_sum(a[i, j] for j in a.dom[i] if j != i)

    with pytest.raises(TraceError) as caught:
        term_of(off_diagonal)
    message = str(caught.value)
    assert "compares with '!='" in message
    assert "split the sum in two, one over '<' and one over '>'" in message


def test_an_affine_reduction_condition_is_a_constraint_of_the_domain() -> None:
    def lower(a: Arr[Fin[n], Fin[n], Real], y: Arr[Fin[n], Real]):  # noqa: F821
        for i in y.dom:
            y[i] = reduce_sum(a[i, j] for j in a.dom[i] if j < i)

    (reduction,) = reductions_in(term_of(lower).stmts[0].expr)
    assert reduction.domain.is_equal(
        isl.Set("[n] -> { [i, j] : 0 <= j < i < n }")
    )


def test_an_equality_condition_is_spelled_the_way_isl_reads_it() -> None:
    # isl's equality is '='; handed '==', it stopped the trace on a syntax error.
    def diagonal(a: Arr[Fin[n], Fin[n], Real], y: Arr[Fin[n], Real]):  # noqa: F821
        for i in y.dom:
            y[i] = reduce_sum(a[i, j] for j in a.dom[i] if j == i)

    (reduction,) = reductions_in(term_of(diagonal).stmts[0].expr)
    assert reduction.domain.is_equal(isl.Set("[n] -> { [i, i] : 0 <= i < n }"))


# }}}


# {{{ state and effects outside the arrays


class _Counter:
    """An object of the kernel author's, with an attribute to keep state in."""

    def __init__(self) -> None:
        self.count = 0.0
        self.cache: dict[str, object] = {}


#: Module state for the kernels below. Each test replaces what it uses through
#: ``monkeypatch``, so a trace that changes one leaves nothing for the next.
_BOX = _Counter()
_SEEN: list[object] = []
_LAST = 0.0
_COUNTS = np.zeros(2)
_WEIGHTS = np.array([0.5, 2.0])


def _note(value: object) -> None:
    """A helper defined outside the body that keeps what it is given."""
    _SEEN.append(value)


def test_an_attribute_the_body_stores_is_refused(monkeypatch) -> None:
    # Natively the count is 1 after a call; the trace recorded y[0] = 1.0, and
    # a second call adds 1 again, which no term says.
    monkeypatch.setitem(globals(), "_BOX", _Counter())

    def counted(y: Arr[Fin[1], Real]):  # noqa: F821
        _BOX.count = _BOX.count + 1.0
        y[0] = _BOX.count

    with pytest.raises(TraceError) as caught:
        term_of(counted)
    message = str(caught.value)
    assert "tracing counted changed the attribute 'count' of '_BOX'" in message
    assert "'_BOX.count' is 0.0 before the trace and 1.0 after it" in message
    assert "Keep the state in an array parameter" in message


def test_an_attribute_carried_across_a_loop_is_refused() -> None:
    # The loop snapshot does not look at attributes; the trace-wide one does.
    box = _Counter()

    def running(x: Arr[Fin[n], Real], y: Arr[Fin[n], Real]):  # noqa: F821
        for i in x.dom:
            box.count = box.count + x[i]
            y[i] = x[i]

    with pytest.raises(TraceError, match="changed the attribute 'count' of 'box'"):
        term_of(running)


def test_a_dict_an_attribute_holds_is_compared_too() -> None:
    box = _Counter()

    def cached(x: Arr[Fin[n], Real], y: Arr[Fin[n], Real]):  # noqa: F821
        box.cache["x"] = x[0]
        for i in y.dom:
            y[i] = x[i]

    with pytest.raises(TraceError, match=r"changed the dict 'box.cache'"):
        term_of(cached)


def test_a_global_list_the_body_appends_to_is_refused(monkeypatch) -> None:
    monkeypatch.setitem(globals(), "_SEEN", [])

    def logged(x: Arr[Fin[n], Real], y: Arr[Fin[n], Real]):  # noqa: F821
        _SEEN.append(x[0])
        for i in y.dom:
            y[i] = x[i]

    with pytest.raises(TraceError) as caught:
        term_of(logged)
    message = str(caught.value)
    assert "changed the global list '_SEEN'" in message
    assert "'_SEEN' is [] before the trace and [x[0]] after it" in message


def test_a_global_only_a_helper_changes_is_refused(monkeypatch) -> None:
    # The helper is defined outside the body, so the body's code never names
    # ``_SEEN``; the loop snapshot missed it, inside a loop or not.
    monkeypatch.setitem(globals(), "_SEEN", [])

    def noted(x: Arr[Fin[n], Real], y: Arr[Fin[n], Real]):  # noqa: F821
        for i in x.dom:
            _note(1)
            y[i] = x[i]

    with pytest.raises(TraceError, match="changed the global list '_SEEN'"):
        term_of(noted)


def test_a_write_into_a_global_numpy_array_is_refused(monkeypatch) -> None:
    # The write changes no output, so comparing outputs could never see it;
    # the compiled kernel never makes it.
    monkeypatch.setitem(globals(), "_COUNTS", np.zeros(2))

    def counting(x: Arr[Fin[n], Real], y: Arr[Fin[n], Real]):  # noqa: F821
        _COUNTS[1] += 1.0
        for i in y.dom:
            y[i] = x[i]

    with pytest.raises(TraceError) as caught:
        term_of(counting)
    message = str(caught.value)
    assert "changed the global array '_COUNTS'" in message
    assert "'_COUNTS[1]' is 0.0 before the trace and 1.0 after it" in message


def test_a_write_into_an_array_an_attribute_holds_is_refused() -> None:
    box = _Counter()
    box.buffer = Arr.zeros(3)

    def stashing(x: Arr[Fin[n], Real], y: Arr[Fin[n], Real]):  # noqa: F821
        box.buffer[2] = 7.0
        for i in y.dom:
            y[i] = x[i]

    with pytest.raises(TraceError, match="changed the array 'box.buffer'"):
        term_of(stashing)


def test_a_global_numpy_array_the_body_only_reads_is_not_state() -> None:
    def weighted(x: Arr[Fin[n], Real], y: Arr[Fin[n], Real]):  # noqa: F821
        for i in y.dom:
            y[i] = _WEIGHTS[1] * x[i]

    (stmt,) = term_of(weighted).stmts
    assert render(stmt.expr) == "2.0*x[i]"


def test_a_global_the_body_rebinds_outside_any_loop_is_refused(monkeypatch) -> None:
    monkeypatch.setitem(globals(), "_LAST", 0.0)

    def remembered(x: Arr[Fin[n], Real], y: Arr[Fin[n], Real]):  # noqa: F821
        global _LAST
        _LAST = x[0]
        for i in y.dom:
            y[i] = x[i]

    with pytest.raises(TraceError, match="changed the global '_LAST'"):
        term_of(remembered)


def test_state_created_by_the_body_is_scratch() -> None:
    # A list or an object the body makes itself is gone when it returns.
    def scratch(x: Arr[Fin[n], Real], y: Arr[Fin[n], Real]):  # noqa: F821
        weights = []
        weights.append(0.5)
        holder = _Counter()
        holder.count = 2.0
        for i in y.dom:
            y[i] = weights[0] * holder.count * x[i]

    (stmt,) = term_of(scratch).stmts
    assert render(stmt.expr) == "1.0*x[i]"


def test_a_cached_property_the_body_reads_first_is_not_state() -> None:
    # The first read stores the value in the object's __dict__, which the
    # trace-wide snapshot sees as a new attribute. It is what every later read,
    # native or traced, gets, so it is not state; an assignment to the same
    # attribute once it is there still is.
    import functools

    class Mesh:
        @functools.cached_property
        def h(self) -> float:
            return 1.0 / 16

    mesh = Mesh()

    def scaled(x: Arr[Fin[n], Real], y: Arr[Fin[n], Real]):  # noqa: F821
        for i in y.dom:
            y[i] = x[i] * mesh.h

    (stmt,) = term_of(scaled).stmts
    assert render(stmt.expr) == "x[i]*0.0625"

    def rescaled(x: Arr[Fin[n], Real], y: Arr[Fin[n], Real]):  # noqa: F821
        mesh.h = mesh.h / 2
        for i in y.dom:
            y[i] = x[i] * mesh.h

    with pytest.raises(TraceError, match="changed the attribute 'h' of 'mesh'"):
        term_of(rescaled)


def test_the_attributes_of_a_library_object_are_the_librarys() -> None:
    # A logger fills a level cache, one of its attributes, on its first debug
    # call. That is the logging module's bookkeeping, not the body's state.
    import logging

    logger = logging.getLogger("loopty.tests.trace")
    logger._cache.clear()

    def logged(x: Arr[Fin[n], Real], y: Arr[Fin[n], Real]):  # noqa: F821
        for i in y.dom:
            logger.debug("step")
            y[i] = x[i]

    (stmt,) = term_of(logged).stmts
    assert render(stmt.expr) == "x[i]"
    assert logger._cache


def test_a_simple_namespace_is_the_kernel_authors_object() -> None:
    # A bag of attributes and nothing else: a store into it is the body's.
    from types import SimpleNamespace

    state = SimpleNamespace(count=0.0)

    def counted(y: Arr[Fin[1], Real]):  # noqa: F821
        state.count = state.count + 1.0
        y[0] = state.count

    with pytest.raises(TraceError, match="changed the attribute 'count' of 'state'"):
        term_of(counted)


def test_a_print_in_the_body_is_refused() -> None:
    def chatty(x: Arr[Fin[n], Real], y: Arr[Fin[n], Real]):  # noqa: F821
        for i in y.dom:
            print("at", i)
            y[i] = x[i]

    with pytest.raises(TraceError) as caught:
        term_of(chatty)
    message = str(caught.value)
    assert "tracing chatty calls print() at test_trace.py:" in message
    assert "Print from the code that calls the kernel" in message


def test_a_print_in_a_helper_is_refused() -> None:
    def shout(value):
        print(value)
        return value

    def chatty(x: Arr[Fin[n], Real], y: Arr[Fin[n], Real]):  # noqa: F821
        for i in y.dom:
            y[i] = shout(x[i])

    with pytest.raises(TraceError, match="calls print"):
        term_of(chatty)


def test_a_random_draw_is_refused() -> None:
    import random

    import numpy as np

    generator = np.random.default_rng(0)

    def jittered(x: Arr[Fin[n], Real], y: Arr[Fin[n], Real]):  # noqa: F821
        for i in y.dom:
            y[i] = x[i] + random.random()

    def noisy(x: Arr[Fin[n], Real], y: Arr[Fin[n], Real]):  # noqa: F821
        for i in y.dom:
            y[i] = x[i] + generator.normal()

    with pytest.raises(TraceError, match=r"calls random\.random\(\) at"):
        term_of(jittered)
    with pytest.raises(TraceError, match=r"numpy\.random\.Generator\.normal\(\)"):
        term_of(noisy)


def test_the_call_watch_is_off_once_a_trace_ends() -> None:
    import sys

    from loopty.trace import _CallWatch

    term_of(axpy)
    if _CallWatch.tool is not None:
        assert sys.monitoring.get_events(_CallWatch.tool) == 0
    assert _CallWatch.depth == 0


def test_a_global_the_body_creates_is_refused(monkeypatch) -> None:
    # Absent before the trace, bound to a term after it.
    monkeypatch.delitem(globals(), "_CREATED", raising=False)

    def creating(x: Arr[Fin[n], Real], y: Arr[Fin[n], Real]):  # noqa: F821
        global _CREATED
        _CREATED = x[0]
        for i in y.dom:
            y[i] = x[i]

    try:
        with pytest.raises(
            TraceError, match="'_CREATED' is unbound before the trace and x"
        ):
            term_of(creating)
    finally:
        globals().pop("_CREATED", None)


# }}}



# {{{ guards isl cannot state as integers


def _loop_nest(domain) -> isl.Set:
    """The one-deep loop nest ``0 <= i < n``, in ``domain``'s parameter space."""
    return isl.Set("[n] -> { [i] : 0 <= i < n }").align_params(domain.get_space())


def test_a_guard_against_a_real_scalar_leaves_the_domain_unnarrowed() -> None:
    # isl reads every name of a constraint as an integer, so stating i < a
    # would take a = 2.5 for an integer parameter; the domain and the compiled
    # kernel then disagreed with the native run.
    def below(a: Real, y: Arr[Fin[n], Real]):  # noqa: F821
        for i in y.dom:
            with when(i < a):
                y[i] = 1.0

    (stmt,) = term_of(below).stmts
    assert "a" not in stmt.domain.get_var_names(isl.dim_type.param)
    assert stmt.domain.is_equal(_loop_nest(stmt.domain))
    # The guard itself is kept, and evaluated at run time.
    assert render(stmt.guard) == "i < a"
    ((conjunct, why),) = stmt.unnarrowed
    assert conjunct == "i < a"
    assert "the scalar a of sort Real" in why
    assert "isl would read every name of a constraint as an integer" in why


def test_the_same_guard_against_an_integral_scalar_still_narrows() -> None:
    def below(a: Nat, y: Arr[Fin[n], Real]):  # noqa: F821
        for i in y.dom:
            with when(i < a):
                y[i] = 1.0

    (stmt,) = term_of(below).stmts
    assert "a" in stmt.domain.get_var_names(isl.dim_type.param)
    nest = _loop_nest(stmt.domain)
    assert stmt.domain.is_subset(nest) and not stmt.domain.is_equal(nest)
    assert stmt.unnarrowed == ()


def test_only_the_conjuncts_isl_cannot_state_are_left_out() -> None:
    def clipped(a: Real, y: Arr[Fin[n], Real]):  # noqa: F821
        for i in y.dom:
            with when((i < a) & (i + 1 < y.dom.size)):
                y[i] = 1.0

    (stmt,) = term_of(clipped).stmts
    assert stmt.domain.is_equal(
        isl.Set("[n] -> { [i] : 0 <= i < n - 1 }").align_params(stmt.domain.get_space())
    )
    assert [conjunct for conjunct, _why in stmt.unnarrowed] == ["i < a"]


def test_a_data_guard_is_recorded_as_unnarrowed_too() -> None:
    def positive(x: Arr[Fin[n], Real], y: Arr[Fin[n], Real]):  # noqa: F821
        for i in y.dom:
            with when(x[i] > 0.0):
                y[i] = x[i]

    (stmt,) = term_of(positive).stmts
    assert stmt.unnarrowed == (("x[i] > 0.0", "reads an array or is not affine"),)


def test_a_reduction_condition_against_a_real_scalar_is_refused() -> None:
    # A reduction keeps its condition only in its domain, so a condition the
    # domain cannot state is refused rather than dropped.
    def partial(a: Real, x: Arr[Fin[n], Real], y: Arr[Fin[1], Real]):  # noqa: F821
        y[0] = reduce_sum(x[j] for j in x.dom if j < a)

    with pytest.raises(TraceError) as caught:
        term_of(partial)
    message = str(caught.value)
    assert "the condition 'j < a' of the reduction over j" in message
    assert "compares with the scalar a of sort Real" in message
    assert "loop variables, sizes and integral scalars" in message


def test_a_reduction_condition_against_an_integral_scalar_is_its_domain() -> None:
    def partial(a: Nat, x: Arr[Fin[n], Real], y: Arr[Fin[1], Real]):  # noqa: F821
        y[0] = reduce_sum(x[j] for j in x.dom if j < a)

    (reduction,) = reductions_in(term_of(partial).stmts[0].expr)
    assert "a" in reduction.domain.get_var_names(isl.dim_type.param)


# }}}


# {{{ a guard is a truth value


def flipped(y: Arr[Fin[n], Real]):  # noqa: F821
    """``~`` on a Python bool is bitwise: the guard is -2 or -1, always true."""
    for i in y.dom:
        with when(~(i > 0)):
            y[i] = 1.0


def test_natively_a_guard_that_is_an_integer_is_refused() -> None:
    # ~False is -1 at i = 0; the trace records not (i > 0) and writes y[0]
    # only, while the native run used to write every cell. Python warns about
    # ~ on a bool, which this suite makes an error, and a user's run does not.
    import warnings

    from loopty.kernel import Kernel

    y = Arr.zeros(3)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", DeprecationWarning)
        with pytest.raises(TraceError) as caught:
            Kernel(flipped)(y)
    message = str(caught.value)
    assert "is the integer -1, not a truth value" in message
    assert "test_trace.py:" in message
    assert "'i <= 0' for '~(i > 0)'" in message
    assert list(y.numpy()) == [0.0, 0.0, 0.0]


def test_a_numpy_integer_guard_is_refused_natively_as_well() -> None:
    # & with an integer operand is bitwise too: True & 2 is 0.
    def masked(m: Arr[Fin[n], Nat], y: Arr[Fin[n], Real]):  # noqa: F821
        for i in y.dom:
            with when((i >= 0) & m[i]):
                y[i] = 1.0

    from loopty.kernel import Kernel

    with pytest.raises(TraceError, match="is the integer 0, not a truth value"):
        Kernel(masked)(Arr.from_numpy(np.array([2, 1])), Arr.zeros(2))


def test_a_guard_that_is_a_bool_runs_natively() -> None:
    # A data comparison is a numpy bool, on which ~ is logical; a comparison of
    # loop variables is a Python bool.
    def complement(x: Arr[Fin[n], Real], y: Arr[Fin[n], Real]):  # noqa: F821
        for i in y.dom:
            with when(~(x[i] > 0.0) & (i + 1 < y.dom.size)):
                y[i] = 1.0

    from loopty.kernel import Kernel

    y = Arr.zeros(4)
    Kernel(complement)(Arr.from_numpy(np.array([1.0, -1.0, 0.0, -2.0])), y)
    assert list(y.numpy()) == [0.0, 1.0, 1.0, 0.0]


def test_a_guard_under_a_false_guard_is_not_asked_natively() -> None:
    # At i = n - 1 the outer guard is false and the read of w[i + 1] is out of
    # range, which a masked read answers with the integer 0: nothing under the
    # outer guard is written there, whatever the inner one says.
    from loopty.kernel import Kernel

    def weighted(w: Arr[Fin[n], Real], y: Arr[Fin[n], Real]):  # noqa: F821
        for i in y.dom:
            with when(i + 1 < y.dom.size):
                with when(w[i + 1]):
                    y[i] = 1.0

    y = Arr.zeros(3)
    Kernel(weighted)(Arr.from_numpy(np.array([0.0, 2.0, 0.0])), y)
    assert list(y.numpy()) == [1.0, 0.0, 0.0]


def test_a_concrete_integer_guard_is_refused_while_tracing() -> None:
    flag = 1

    def constant(y: Arr[Fin[n], Real]):  # noqa: F821
        for i in y.dom:
            with when(flag):
                y[i] = 1.0

    with pytest.raises(TraceError, match="is the integer 1, not a truth value"):
        term_of(constant)


# }}}


# {{{ the slots of an object, and the offsets of a ragged array


class _Slotted:
    """An object of the kernel author's that keeps its state in slots."""

    __slots__ = ("count", "__secret", "unset")

    def __init__(self) -> None:
        self.count = 0.0
        self.__secret = 1.0

    def secret(self) -> float:
        return self.__secret


class _Mixed(_Slotted):
    """Slots from a base class, and a ``__dict__`` of its own."""

    def __init__(self) -> None:
        super().__init__()
        self.label = "a"


def test_an_attribute_kept_in_a_slot_is_state() -> None:
    state = _Slotted()

    def counted(y: Arr[Fin[1], Real]):  # noqa: F821
        state.count = state.count + 1.0
        y[0] = state.count

    with pytest.raises(TraceError) as caught:
        term_of(counted)
    message = str(caught.value)
    assert "changed the attribute 'count' of 'state'" in message
    assert "'state.count' is 0.0 before the trace and 1.0 after it" in message


def test_a_private_slot_is_read_by_its_mangled_name() -> None:
    state = _Slotted()

    def peeking(y: Arr[Fin[1], Real]):  # noqa: F821
        state._Slotted__secret = state.secret() + 1.0
        y[0] = 1.0

    with pytest.raises(TraceError, match="the attribute '_Slotted__secret'"):
        term_of(peeking)


def test_a_slot_set_for_the_first_time_is_a_change() -> None:
    state = _Slotted()

    def filling(y: Arr[Fin[1], Real]):  # noqa: F821
        state.unset = 1.0
        y[0] = 1.0

    with pytest.raises(
        TraceError, match="'state.unset' is unbound before the trace and 1.0"
    ):
        term_of(filling)


def test_slots_and_a_dict_are_both_read() -> None:
    state = _Mixed()

    def relabelled(y: Arr[Fin[1], Real]):  # noqa: F821
        state.label = "b"
        y[0] = state.count

    def recounted(y: Arr[Fin[1], Real]):  # noqa: F821
        state.count = 2.0
        y[0] = 1.0

    with pytest.raises(TraceError, match="the attribute 'label' of 'state'"):
        term_of(relabelled)
    with pytest.raises(TraceError, match="the attribute 'count' of 'state'"):
        term_of(recounted)


def test_a_slotted_object_the_body_only_reads_is_not_state() -> None:
    state = _Slotted()

    def scaled(x: Arr[Fin[n], Real], y: Arr[Fin[n], Real]):  # noqa: F821
        for i in y.dom:
            y[i] = x[i] * state.secret()

    (stmt,) = term_of(scaled).stmts
    assert render(stmt.expr) == "x[i]*1.0"


def test_a_write_into_the_offsets_of_a_ragged_array_is_refused() -> None:
    rows = Arr.ragged([1, 2], values=np.zeros(3))

    def shifting(x: Arr[Fin[n], Real], y: Arr[Fin[n], Real]):  # noqa: F821
        rows.offsets[1] = 2
        for i in y.dom:
            y[i] = x[i]

    with pytest.raises(TraceError) as caught:
        term_of(shifting)
    message = str(caught.value)
    assert "changed the array 'rows.offsets'" in message
    assert "'rows.offsets[1]' is 1 before the trace and 2 after it" in message


def test_a_write_into_the_values_of_a_ragged_array_is_still_refused() -> None:
    rows = Arr.ragged([1, 2], values=np.zeros(3))

    def stashing(x: Arr[Fin[n], Real], y: Arr[Fin[n], Real]):  # noqa: F821
        rows[1, 1] = 5.0
        for i in y.dom:
            y[i] = x[i]

    with pytest.raises(TraceError, match=r"'rows\[2\]' is 0.0 before the trace"):
        term_of(stashing)


# }}}


# {{{ a kernel installed into site-packages


@pytest.fixture
def site_packages(tmp_path, monkeypatch, request):
    """A directory that counts as site-packages, with a writer of packages into it.

    Its path is added to the library roots, which is where a non-editable
    install puts a kernel, and to ``sys.path``. The packages written into it
    are imported by name and forgotten again at the end.
    """
    import importlib
    import os
    import sys
    import textwrap

    # The module, not the function loopty exports under the same name.
    tracing = importlib.import_module("loopty.trace")
    site = tmp_path / "site-packages"
    site.mkdir()
    roots = tracing._library_roots()
    monkeypatch.setattr(
        tracing, "_library_roots", lambda: (*roots, os.path.realpath(site))
    )
    tracing._library_file.cache_clear()
    request.addfinalizer(tracing._library_file.cache_clear)
    monkeypatch.syspath_prepend(str(site))
    written: list[str] = []

    def forget() -> None:
        for name in list(sys.modules):
            if any(name == top or name.startswith(top + ".") for top in written):
                del sys.modules[name]

    request.addfinalizer(forget)

    def install(files: dict[str, str]):
        for relative, text in files.items():
            path = site / relative
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(textwrap.dedent(text))
            written.append(relative.split("/")[0].removesuffix(".py"))
        importlib.invalidate_caches()
        return importlib.import_module

    return install


_HEADER = """\
    from __future__ import annotations

    from lanky.prelude import Real

    from loopty import Arr, Fin
"""


def test_a_print_in_an_installed_kernel_is_refused(site_packages) -> None:
    load = site_packages(
        {
            "installed_chatty/__init__.py": "",
            "installed_chatty/kernels.py": _HEADER
            + """
    def chatty(x: Arr[Fin[n], Real], y: Arr[Fin[n], Real]):
        for i in y.dom:
            print("at", i)
            y[i] = x[i]
    """,
        }
    )
    from loopty.trace import _library_file

    module = load("installed_chatty.kernels")
    assert _library_file(module.__file__)
    with pytest.raises(TraceError, match=r"calls print\(\) at kernels.py:"):
        term_of(module.chatty)


def test_the_helpers_of_an_installed_kernel_are_followed(site_packages) -> None:
    load = site_packages(
        {
            "installed_noted/__init__.py": "",
            "installed_noted/helpers.py": """\
    SEEN = []


    class Box:
        __slots__ = ("count",)

        def __init__(self):
            self.count = 0.0


    BOX = Box()


    def note(value):
        SEEN.append(value)
        return value
    """,
            "installed_noted/kernels.py": _HEADER
            + """
    from installed_noted.helpers import BOX, note


    def noted(x: Arr[Fin[n], Real], y: Arr[Fin[n], Real]):
        for i in y.dom:
            y[i] = note(x[i])


    def counted(y: Arr[Fin[1], Real]):
        BOX.count = BOX.count + 1.0
        y[0] = BOX.count
    """,
        }
    )
    module = load("installed_noted.kernels")
    with pytest.raises(TraceError, match="changed the global list 'SEEN'"):
        term_of(module.noted)
    with pytest.raises(TraceError, match="changed the attribute 'count' of 'BOX'"):
        term_of(module.counted)


def test_another_installed_package_is_still_a_library(site_packages) -> None:
    # Its print is not the body's while another package's kernel is traced,
    # and is when a kernel of its own is: the location was only passed over,
    # not disabled for good.
    load = site_packages(
        {
            "installed_loud/__init__.py": """\
    def shout(value):
        print(value)
        return value
    """,
            "installed_loud/kernels.py": _HEADER
            + """
    from installed_loud import shout


    def loud(x: Arr[Fin[n], Real], y: Arr[Fin[n], Real]):
        for i in y.dom:
            y[i] = shout(x[i])
    """,
            "installed_quiet/__init__.py": "",
            "installed_quiet/kernels.py": _HEADER
            + """
    from installed_loud import shout


    def quiet(x: Arr[Fin[n], Real], y: Arr[Fin[n], Real]):
        for i in y.dom:
            y[i] = shout(x[i])
    """,
        }
    )
    quiet = load("installed_quiet.kernels").quiet
    loud = load("installed_loud.kernels").loud
    (stmt,) = term_of(quiet).stmts
    assert render(stmt.expr) == "x[i]"
    with pytest.raises(TraceError, match=r"calls print\(\) at __init__.py:"):
        term_of(loud)


def test_a_namespace_package_is_not_all_the_kernels(site_packages) -> None:
    # Two distributions install into one namespace directory; the kernel's own
    # package is the regular package below it, and the other is a library.
    load = site_packages(
        {
            "installed_ns/alpha/__init__.py": "",
            "installed_ns/alpha/kernels.py": _HEADER
            + """
    from installed_ns.beta import shout


    def relayed(x: Arr[Fin[n], Real], y: Arr[Fin[n], Real]):
        for i in y.dom:
            y[i] = shout(x[i])


    def chatty(x: Arr[Fin[n], Real], y: Arr[Fin[n], Real]):
        for i in y.dom:
            print(i)
            y[i] = x[i]
    """,
            "installed_ns/beta/__init__.py": """\
    def shout(value):
        print(value)
        return value
    """,
        }
    )
    from loopty.trace import _own_of

    module = load("installed_ns.alpha.kernels")
    assert _own_of(module.relayed).package == "installed_ns.alpha"
    (stmt,) = term_of(module.relayed).stmts
    assert render(stmt.expr) == "x[i]"
    with pytest.raises(TraceError, match=r"calls print\(\) at kernels.py:"):
        term_of(module.chatty)


def test_library_code_is_decided_by_module() -> None:
    import logging

    from loopty.trace import _library, _Own

    own = _Own("mykernels.stencil", "mykernels")
    site = "/opt/site-packages"
    assert not _library("mykernels.helpers", f"{site}/mykernels/helpers.py", own)
    assert _library("numpy.linalg", "/src/numpy/linalg.py", own)
    assert _library("logging", logging.__file__, own)
    # A standard library name is the standard library's where its code is.
    assert not _library("logging", "/src/logging/__init__.py", own)
    # A kernel defined in a module of loopty's own is only that module.
    inside = _Own("loopty.examples", None)
    assert not _library("loopty.examples", "/src/loopty/examples.py", inside)
    assert _library("loopty.trace", "/src/loopty/trace.py", inside)


# }}}


def test_a_module_named_like_the_standard_library_is_the_authors(
    tmp_path, monkeypatch
) -> None:
    # colorsys.py next to the kernel's script is the author's module, whatever
    # the standard library has under the same name: its helper's module state
    # is copied and its print is watched, as they are for any other name.
    import importlib
    import os
    import sys
    import textwrap

    name = "colorsys"
    assert name in sys.stdlib_module_names
    monkeypatch.syspath_prepend(str(tmp_path))
    monkeypatch.delitem(sys.modules, name, raising=False)
    (tmp_path / f"{name}.py").write_text(
        textwrap.dedent(
            """\
            SEEN = []


            def note(value):
                SEEN.append(value)
                return value


            def shout(value):
                print(value)
                return value
            """
        )
    )
    importlib.invalidate_caches()
    helpers = importlib.import_module(name)
    assert os.path.dirname(os.path.realpath(helpers.__file__)) == os.path.realpath(
        tmp_path
    )
    note, shout = helpers.note, helpers.shout

    def noted(x: Arr[Fin[n], Real], y: Arr[Fin[n], Real]):  # noqa: F821
        for i in y.dom:
            y[i] = note(x[i])

    def shouted(x: Arr[Fin[n], Real], y: Arr[Fin[n], Real]):  # noqa: F821
        for i in y.dom:
            y[i] = shout(x[i])

    with pytest.raises(TraceError, match="changed the global list 'SEEN'"):
        term_of(noted)
    with pytest.raises(TraceError, match=r"calls print\(\) at colorsys.py:"):
        term_of(shouted)


# }}}
