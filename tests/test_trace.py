"""Tracing: what running a body once against a generic point records."""

from __future__ import annotations

import islpy as isl
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


#: Module globals for the kernels below to carry state in. Each test sets its
#: own through ``monkeypatch``, so a trace that changes one leaves nothing
#: behind for the next.
_COUNT = 0
_TOTALS = [0.0]


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
