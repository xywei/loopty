"""A kernel body branches with Python's own if, and loops over a nest in one line.

The trace runs the body once per path through its branches, answering each
traced condition's truth value from the path, and joins the runs into one
term (loopty 0.4; ``kb/decisions/2026-10-15``). These tests hold the joined
term to what the decision asks of it: two branches are two statements under
complementary guards, a statement after an ``if``/``else`` is one statement,
a value two paths choose between on data is a select, a path isl shows to be
empty records nothing, one run answers a condition once, the runs are capped,
a ``while`` on a traced condition is refused, and ``for box, a, b in R.dom:``
is the three loops it stands for. Every kernel that runs is run three ways,
natively, interpreted and compiled, which have to agree.
"""

from __future__ import annotations

import numpy as np
import pytest
from lanky.prelude import Real
from lanky.terms import evaluate_annotations, render

from loopty import Arr, Fin, Where, kernel
from loopty.trace import MAX_RUNS, Select, TraceError, trace

pytest.importorskip("loopy")

P = 4
Q = P + 1


def term_of(fn):
    """Trace a plain function the way ``@kernel`` would."""
    return trace(fn, evaluate_annotations(fn))


def three_runs(caller, inputs, output):
    """``output`` after the native, interpreted and compiled runs of ``caller``."""
    from loopty.executor import LoopyExecutor
    from loopty.interpret import interpret

    runs = []
    for run in (
        lambda args: caller(**args),
        lambda args: interpret(caller.term, args),
        lambda args: LoopyExecutor().run(caller, **args),
    ):
        args = {name: Arr.from_numpy(np.array(value)) for name, value in inputs.items()}
        run(args)
        runs.append(args[output].numpy())
    return runs


def agrees(caller, inputs, output, expected):
    """The three runs of ``caller`` agree with each other and with ``expected``."""
    native, interpreted, compiled = three_runs(caller, inputs, output)
    np.testing.assert_allclose(native, expected)
    np.testing.assert_allclose(interpreted, native)
    np.testing.assert_allclose(compiled, native)


# {{{ two branches, and what follows them


@kernel
def reconstruct(
    C: Arr[Where[box: Fin[nb], a: Fin[2], b: Fin[Q], a + b <= P], Real],  # noqa: F821
    R: Arr[Where[box: Fin[nb], a: Fin[Q], b: Fin[Q], a + b <= P], Real],  # noqa: F821
):
    for box, a, b in R.dom:
        if a < 2:
            R[box, a, b] = C[box, a, b]
        else:
            R[box, a, b] = -R[box, a - 2, b + 2]


def test_an_if_and_its_else_are_two_statements_under_complementary_guards() -> None:
    copy, recurrence = reconstruct.term.stmts
    assert render(copy.guard) == "a < 2"
    assert render(recurrence.guard) == "a >= 2"
    assert copy.domain.intersect(recurrence.domain).is_empty()
    assert copy.domain.union(recurrence.domain).is_equal(copy.loop_domain)
    # The else's read of R[box, a - 2, b + 2] is in bounds where it runs.
    assert not copy.unnarrowed and not recurrence.unnarrowed


@kernel
def recurrence(c: Arr[Fin[2], Real], r: Arr[Fin[n], Real]):  # noqa: F821
    for a in r.dom:
        if a < 2:
            r[a] = c[a]
        else:
            r[a] = -r[a - 2]


def test_the_branch_not_taken_is_not_run_natively() -> None:
    # The masked branch of 'with when' read r[a - 2] at a < 2, a negative
    # index numpy wraps; the native if never runs it there, and c[a] is read
    # only where a < 2, in all three runs.
    c = np.array([1.0, 2.0])
    expected = [1.0, 2.0, -1.0, -2.0, 1.0, 2.0]
    agrees(recurrence, {"c": c, "r": np.zeros(6)}, "r", expected)


@kernel
def after_the_join(x: Arr[Fin[n], Real], y: Arr[Fin[n], Real]):  # noqa: F821
    for i in y.dom:
        if x[i] > 0.0:
            y[i] = x[i]
        elif i > 2:
            y[i] = 2.0 * x[i]
        else:
            y[i] = 0.0
        x[i] = 1.0


def test_a_statement_after_an_if_else_is_recorded_once_with_no_guard() -> None:
    stmts = after_the_join.term.stmts
    guards = [None if stmt.guard is None else render(stmt.guard) for stmt in stmts]
    assert guards == [
        "x[i] > 0.0",
        "not (x[i] > 0.0) and i > 2",
        "not (x[i] > 0.0) and i <= 2",
        None,
    ]
    assert stmts[-1].text == "x[i] = 1.0"


def test_an_elif_chain_runs_alike_three_ways() -> None:
    x = np.array([1.0, -2.0, 0.5, -1.0, 3.0, -4.0])
    expected = np.where(x > 0.0, x, np.where(np.arange(6) > 2, 2.0 * x, 0.0))
    agrees(after_the_join, {"x": x, "y": np.zeros(6)}, "y", expected)


@kernel
def nested(x: Arr[Fin[n], Real], y: Arr[Fin[n], Real]):  # noqa: F821
    for i in y.dom:
        if i > 0:
            if x[i] > 0.0:
                y[i] = x[i]
            else:
                y[i] = -x[i]
            y[i] = y[i] + 1.0


def test_a_join_inside_a_branch_keeps_the_outer_branchs_guard() -> None:
    first, second, after = nested.term.stmts
    assert render(after.guard) == "i > 0"
    assert render(first.guard) == "i > 0 and x[i] > 0.0"
    x = np.array([1.0, -2.0, 3.0, -4.0])
    expected = np.where(np.arange(4) > 0, np.abs(x) + 1.0, 0.0)
    agrees(nested, {"x": x, "y": np.zeros(4)}, "y", expected)


@kernel
def between_two(
    x: Arr[Fin[n], Real],  # noqa: F821
    y: Arr[Fin[n], Real],  # noqa: F821
    z: Arr[Fin[n], Real],  # noqa: F821
    w: Arr[Fin[n], Real],  # noqa: F821
):
    for i in x.dom:
        if x[i] > 0.0:
            y[i] = 1.0
        z[i] = z[i] + 1.0
        if x[i] > 1.0:
            w[i] = 1.0


def test_a_statement_between_two_branches_is_one_statement() -> None:
    # Four runs record the update, two under each answer of the first if;
    # it is one statement with no guard, run once (it ran twice where
    # x[i] <= 0 when the records of one path were not joined first).
    stmts = between_two.term.stmts
    assert [stmt.text for stmt in stmts] == [
        "y[i] = 1.0",
        "z[i] = z[i] + 1.0",
        "w[i] = 1.0",
    ]
    assert stmts[1].guard is None
    x = np.array([2.0, 0.5, -1.0])
    inputs = {"x": x, "y": np.zeros(3), "z": np.zeros(3), "w": np.zeros(3)}
    agrees(between_two, inputs, "z", np.ones(3))


# }}}


# {{{ connectives and conditional expressions


@kernel
def inside(x: Arr[Fin[n], Real], y: Arr[Fin[n], Real]):  # noqa: F821
    for i in y.dom:
        if i > 0 and i < y.dom.size - 1 and not (x[i] < 0.0):
            y[i] = x[i - 1] + x[i + 1]


def test_and_or_not_branch_and_guard_the_reads_they_protect() -> None:
    (stmt,) = inside.term.stmts
    assert stmt.domain.is_equal(type(stmt.domain)("[n] -> { [i] : 0 < i < n - 1 }"))
    x = np.array([1.0, 2.0, -3.0, 4.0, 5.0])
    expected = [0.0, 1.0 + -3.0, 0.0, -3.0 + 5.0, 0.0]
    agrees(inside, {"x": x, "y": np.zeros(5)}, "y", expected)


@kernel
def either_end(y: Arr[Fin[n], Real]):  # noqa: F821
    for i in y.dom:
        if i == 0 or i == y.dom.size - 1:
            y[i] = 1.0


def test_an_or_runs_alike_three_ways() -> None:
    agrees(either_end, {"y": np.zeros(5)}, "y", [1.0, 0.0, 0.0, 0.0, 1.0])


@kernel
def upwind(
    vn: Arr[Fin[n], Real],  # noqa: F821
    a: Arr[Fin[n], Real],  # noqa: F821
    b: Arr[Fin[n], Real],  # noqa: F821
    f: Arr[Fin[n], Real],  # noqa: F821
):
    for i in f.dom:
        f[i] = vn[i] * (a[i] if vn[i] > 0.0 else b[i])


def test_a_conditional_expression_on_data_is_one_statement_with_a_select() -> None:
    # The flux grudge computes with where (#266): one statement, whose value
    # chooses between the two reads, as C's ?: does.
    (stmt,) = upwind.term.stmts
    assert stmt.guard is None
    assert any(isinstance(child, Select) for child in stmt.expr.children)
    assert stmt.text == "f[i] = vn[i] * (a[i] if vn[i] > 0.0 else b[i])"
    rng = np.random.default_rng(1)
    vn, a, b = (rng.standard_normal(7) for _ in range(3))
    inputs = {"vn": vn, "a": a, "b": b, "f": np.zeros(7)}
    agrees(upwind, inputs, "f", vn * np.where(vn > 0.0, a, b))


@kernel
def shifted(x: Arr[Fin[n], Real], y: Arr[Fin[n], Real]):  # noqa: F821
    for i in y.dom:
        y[i] = x[i - 1] if i > 0 else 0.0


def test_a_conditional_expression_on_a_loop_is_two_statements() -> None:
    # A condition isl states narrows each value's domain, so x[i - 1] is read
    # only where i > 0.
    first, second = shifted.term.stmts
    assert render(first.guard) == "i > 0"
    assert render(second.guard) == "i <= 0"
    agrees(shifted, {"x": np.arange(4.0), "y": np.ones(4)}, "y", [0.0, 0.0, 1.0, 2.0])


@kernel
def asked_twice(x: Arr[Fin[n], Real], y: Arr[Fin[n], Real]):  # noqa: F821
    for i in y.dom:
        both = x[i] > 0.0 and i > 1
        if both:
            y[i] = 1.0


def test_one_run_answers_a_condition_once() -> None:
    # Where x[i] > 0.0 is false, 'and' gives it back, and the if asks it
    # again: the run answers as before, and no path takes it both ways.
    (stmt,) = asked_twice.term.stmts
    assert render(stmt.guard) == "x[i] > 0.0 and i > 1"
    x = np.array([1.0, 1.0, -1.0, 2.0])
    agrees(asked_twice, {"x": x, "y": np.zeros(4)}, "y", [0.0, 0.0, 0.0, 1.0])


# }}}


# {{{ paths that hold nowhere, and paths without end


def test_a_path_isl_shows_empty_records_nothing() -> None:
    def impossible(y: Arr[Fin[n], Real]):  # noqa: F821
        for i in y.dom:
            if i < 2:
                if i == 3:
                    y[i] = 1.0
                else:
                    y[i] = 2.0

    (stmt,) = term_of(impossible).stmts
    assert render(stmt.guard) == "i < 2"


def test_more_paths_than_the_cap_are_refused() -> None:
    def counted(x: Arr[Fin[n], Real], y: Arr[Fin[n], Real]):  # noqa: F821
        for i in y.dom:
            if x[i] > 0.0:
                y[i] = y[i] + 1.0
            if x[i] > 1.0:
                y[i] = y[i] + 1.0
            if x[i] > 2.0:
                y[i] = y[i] + 1.0
            if x[i] > 3.0:
                y[i] = y[i] + 1.0
            if x[i] > 4.0:
                y[i] = y[i] + 1.0
            if x[i] > 5.0:
                y[i] = y[i] + 1.0
            if x[i] > 6.0:
                y[i] = y[i] + 1.0
            if x[i] > 7.0:
                y[i] = y[i] + 1.0
            if x[i] > 8.0:
                y[i] = y[i] + 1.0

    assert 2**9 > MAX_RUNS
    with pytest.raises(TraceError) as caught:
        term_of(counted)
    message = str(caught.value)
    assert f"has more than {MAX_RUNS} paths through its branches" in message
    assert "x[i] > 0.0, x[i] > 1.0" in message


def test_a_while_on_a_traced_condition_is_refused() -> None:
    def looping(u: Arr[Fin[n], Real]):  # noqa: F821
        for i in u.dom:
            while u[i] < 1.0:
                u[i] = u[i] + 1.0

    with pytest.raises(TraceError, match=r"the while loop at .* while u\[i\] < 1.0"):
        term_of(looping)


def test_a_branch_that_raises_is_refused_with_its_path() -> None:
    def checked(x: Arr[Fin[n], Real], y: Arr[Fin[n], Real]):  # noqa: F821
        for i in y.dom:
            if i > 0:
                raise ValueError("only the first")
            y[i] = x[i]

    with pytest.raises(TraceError, match="on the branch where i > 0"):
        term_of(checked)


def test_a_condition_on_a_closed_loops_variable_is_refused() -> None:
    def after(x: Arr[Fin[n], Real], y: Arr[Fin[n], Real]):  # noqa: F821
        for i in x.dom:
            y[i] = x[i]
        if i > 0:
            y[0] = 1.0

    with pytest.raises(TraceError, match="the condition 'i > 0' at .* loop variable"):
        term_of(after)


# }}}


# {{{ a condition its branch writes, asked again, or chosen inside a closed loop


def test_a_branch_that_writes_what_its_condition_reads_is_refused() -> None:
    # Run as the term says, the else would ask x[i] > 0.0 again after the
    # then-branch wrote x[i], and run too where x[i] > 0.0 (Codex, #320).
    def flipped(x: Arr[Fin[n], Real]):  # noqa: F821
        for i in x.dom:
            if x[i] > 0.0:
                x[i] = -1.0
            else:
                x[i] = 1.0

    with pytest.raises(TraceError) as caught:
        term_of(flipped)
    message = str(caught.value)
    assert "x[i] = 1.0 (" in message
    assert "is under the condition 'x[i] > 0.0'" in message
    assert "x[i] = -1.0 (" in message
    assert "writes x after the if and before it" in message
    assert "'... if x[i] > 0.0 else ...'" in message


def test_a_write_before_a_later_statement_of_its_branch_is_refused() -> None:
    # Run as the term says, y[i] = 1.0 would ask x[i] > 0.0 again after the
    # negation and be skipped (Codex, #319).
    def negated(x: Arr[Fin[n], Real], y: Arr[Fin[n], Real]):  # noqa: F821
        for i in x.dom:
            if x[i] > 0.0:
                x[i] = -x[i]
                y[i] = 1.0

    with pytest.raises(TraceError, match=r"y\[i\] = 1.0 .* writes x after the if"):
        term_of(negated)


def test_a_write_between_an_if_and_a_statement_it_decides_is_refused() -> None:
    # flag is a Python int the if chose; y[i] = 1.0 is under x[i] > 0.0,
    # which the term would ask after x[i] = 0.0.
    def flagged_before(x: Arr[Fin[n], Real], y: Arr[Fin[n], Real]):  # noqa: F821
        for i in x.dom:
            if x[i] > 0.0:
                flag = 1
            else:
                flag = 0
            x[i] = 0.0
            if flag:
                y[i] = 1.0

    with pytest.raises(TraceError) as caught:
        term_of(flagged_before)
    message = str(caught.value)
    assert "y[i] = 1.0 (" in message
    assert "is under the condition 'x[i] > 0.0'" in message
    assert "x[i] = 0.0 (" in message
    assert "writes x after the if and before it" in message


def test_a_write_in_a_loop_under_the_branch_is_refused() -> None:
    # The next pass of the loop over j would ask x[i] > 0.0 after the write.
    def drained(x: Arr[Fin[n], Real], z: Arr[Fin[m], Real]):  # noqa: F821
        for i in x.dom:
            if x[i] > 0.0:
                for j in z.dom:
                    x[i] = x[i] - 1.0

    with pytest.raises(TraceError, match="writes x itself, in a loop opened after"):
        term_of(drained)


def test_a_select_on_a_condition_written_before_it_is_refused() -> None:
    def chosen_before(x: Arr[Fin[n], Real], y: Arr[Fin[n], Real]):  # noqa: F821
        for i in x.dom:
            v = 1.0 if x[i] > 0.0 else 2.0
            x[i] = 0.0
            y[i] = v

    with pytest.raises(TraceError, match=r"y\[i\] = v .* writes x after the if"):
        term_of(chosen_before)


def test_a_branch_whose_one_statement_writes_its_condition_is_one_statement() -> None:
    # Its guard is asked as it runs, before it writes: nothing under
    # x[i] > 1.0 runs after the write.
    def clamped(x: Arr[Fin[n], Real]):  # noqa: F821
        for i in x.dom:
            if x[i] > 1.0:
                x[i] = 1.0

    (stmt,) = term_of(clamped).stmts
    assert render(stmt.guard) == "x[i] > 1.0"


@kernel
def clamped_in_one(x: Arr[Fin[n], Real], y: Arr[Fin[n], Real]):  # noqa: F821
    for i in x.dom:
        x[i] = 1.0 if x[i] > 1.0 else x[i]
        y[i] = 2.0 * x[i]


def test_a_value_chosen_by_what_its_statement_writes_runs() -> None:
    # A select reads x[i] once, before the statement writes it, and compiles,
    # where a guard that reads what its statement writes does not (#321).
    x = np.array([0.5, 2.0, -1.0, 1.5])
    agrees(clamped_in_one, {"x": x, "y": np.zeros(4)}, "y", 2.0 * np.minimum(x, 1.0))


@kernel
def asked_after_a_write(
    x: Arr[Fin[n], Real],  # noqa: F821
    y: Arr[Fin[n], Real],  # noqa: F821
    z: Arr[Fin[n], Real],  # noqa: F821
):
    for i in x.dom:
        if x[i] > 0.0:
            y[i] = 1.0
        x[i] = -x[i]
        if x[i] > 0.0:
            z[i] = 1.0


def test_a_condition_asked_again_after_a_write_is_another_condition() -> None:
    # The second if is not answered as the first: where x[i] < 0, it holds
    # after the negation, and z[i] is written (Codex, #320).
    first, negation, second = asked_after_a_write.term.stmts
    assert render(first.guard) == "x[i] > 0.0"
    assert negation.guard is None
    assert render(second.guard) == "x[i] > 0.0"
    x = np.array([1.0, -2.0, 0.0, -0.5])
    inputs = {"x": x, "y": np.zeros(4), "z": np.zeros(4)}
    agrees(asked_after_a_write, inputs, "y", np.where(x > 0.0, 1.0, 0.0))
    agrees(asked_after_a_write, inputs, "z", np.where(-x > 0.0, 1.0, 0.0))


def test_a_value_chosen_inside_a_closed_loop_is_refused() -> None:
    # Natively a is what the last iteration chose; the trace saw both
    # choices at a generic i (Codex, #320).
    def chosen(x: Arr[Fin[n], Real], y: Arr[Fin[n], Real]):  # noqa: F821
        for i in x.dom:
            a = 1.0 if i == 0 else 2.0
            x[i] = a
        y[0] = a

    with pytest.raises(TraceError) as caught:
        term_of(chosen)
    message = str(caught.value)
    assert "y[0] = a (" in message
    assert "rests on which way the branch on 'i == 0'" in message
    assert "inside a loop that has closed before it" in message


def test_a_write_made_or_not_inside_a_closed_loop_is_refused() -> None:
    # flag is asked twice, before and after the write, and is the value it
    # was: seen is 1 on one path and 0 on the other, both at a generic i.
    def flagged(x: Arr[Fin[n], Real], y: Arr[Fin[n], Real]):  # noqa: F821
        for i in x.dom:
            flag = x[i] > 0.0
            if flag:
                x[i] = 0.0
            seen = 1 if flag else 0
        if seen:
            y[0] = 1.0

    with pytest.raises(TraceError, match=r"y\[0\] = 1.0 .* rests on which way"):
        term_of(flagged)


# }}}


# {{{ a loop nest on one line


@kernel
def reconstruct_nested(
    C: Arr[Where[box: Fin[nb], a: Fin[2], b: Fin[Q], a + b <= P], Real],  # noqa: F821
    R: Arr[Where[box: Fin[nb], a: Fin[Q], b: Fin[Q], a + b <= P], Real],  # noqa: F821
):
    for box in R.dom:
        for a in R.dom[box]:
            for b in R.dom[box, a]:
                if a < 2:
                    R[box, a, b] = C[box, a, b]
                else:
                    R[box, a, b] = -R[box, a - 2, b + 2]


def test_a_loop_nest_on_one_line_is_the_nest_it_stands_for() -> None:
    one_line = reconstruct.term.stmts
    nested_ = reconstruct_nested.term.stmts
    assert [stmt.inames for stmt in one_line] == [("box", "a", "b")] * 2
    for mine, theirs in zip(one_line, nested_, strict=True):
        assert mine.inames == theirs.inames
        assert mine.domain.is_equal(theirs.domain)
    # The scoped handles are the loops' names.
    handles = (reconstruct.box, reconstruct.a, reconstruct.b)
    assert [str(handle) for handle in handles] == [
        "reconstruct.box",
        "reconstruct.a",
        "reconstruct.b",
    ]


@kernel
def transpose_lower(a: Arr[Fin[n], Fin[n], Real], t: Arr[Fin[n], Fin[n], Real]):  # noqa: F821
    for i, j in t.dom:
        if j <= i:
            t[j, i] = a[i, j]


def test_a_loop_nest_on_one_line_runs_alike_three_ways() -> None:
    (stmt,) = transpose_lower.term.stmts
    assert stmt.inames == ("i", "j")
    a = np.arange(16.0).reshape(4, 4)
    agrees(transpose_lower, {"a": a, "t": np.zeros((4, 4))}, "t", np.tril(a).T)


# }}}
