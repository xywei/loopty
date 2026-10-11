"""A kernel called inside another kernel's loop: inlined, with binders of its own.

A kernel body calls a kernel as it calls a function (:mod:`loopty.calls`). A
rule, whose return annotation is a sort, returns a value and its call is that
value; a kernel that writes its parameters is called as a statement and writes
through what it was given. Traced, the callee's body runs again in the
caller's trace against views of the caller's arrays, so its loops and sums
are the caller's, at every point the call is made, and every fact of the
caller is about them. The call owes what the callee's contract checks
natively, at every point it is made, and leaves its postcondition behind.

The kernels shared with the Lean checks and the coverage report are in
``kernels/calls.py``.
"""

from __future__ import annotations

import importlib

import islpy as isl
import numpy as np
import pytest
from lanky.ledger import Status
from lanky.prelude import Nat, Real
from lanky.terms import render

from loopty import (
    Arr,
    Fin,
    Schedule,
    TraceError,
    Where,
    kernel,
    program,
    reduce_sum,
    table,
    when,
)
from loopty.executor import LoopyExecutor, emit_code
from loopty.faithful import KIND
from loopty.term import Reduction

calls = importlib.import_module("kernels.calls")

RNG = np.random.default_rng(7)


def _facts(found, kind=None):
    facts = found.facts()
    return [fact for fact in facts if kind is None or fact.kind == kind]


def _decided(fact):
    from loopty.oracle import IslOracle

    return IslOracle().establish(fact) or fact


def _faithful(found):
    (fact,) = [fact for fact in found.facts() if fact.kind == KIND]
    return fact


def _cells():
    """PyOP2's two triangles: nodes 0 to 3, and the cell map."""
    return np.array([[0, 1, 3], [2, 3, 1]])


# {{{ a rule: a value with a loop of its own


def test_a_rule_called_in_a_loop_is_its_value_with_a_loop_of_its_own() -> None:
    term = calls.matvec.term
    (stmt,) = term.stmts
    assert stmt.inames == ("i",)
    assert render(stmt.assignee.indices[0]) == "i"
    (sum_,) = [stmt.expr] if isinstance(stmt.expr, Reduction) else []
    # dot's k, a binder of the caller's statement over the row at each i
    assert sum_.inames == ("k",)
    assert sum_.domain.get_var_names(isl.dim_type.set) == ["i", "k"]
    assert render(sum_.body) == "A[i, k]*x[k]"
    (call,) = term.calls
    assert (call.callee, call.label, call.returns) == ("dot", "dot", True)
    assert call.inames == ("i",)
    kinds = [(a.param, a.kind, a.spelled) for a in call.arguments]
    assert kinds == [("a", "row", "A[i]"), ("b", "array", "x")]


def test_the_rule_runs_natively_compiled_and_interpreted_alike() -> None:
    A, x = RNG.standard_normal((5, 4)), RNG.standard_normal(4)
    y = Arr.zeros(5)
    calls.matvec(Arr.from_numpy(A), Arr.from_numpy(x), y)
    np.testing.assert_allclose(y.numpy(), A @ x)
    inputs = {"A": Arr.from_numpy(A), "x": Arr.from_numpy(x), "y": Arr.zeros(5)}
    fact = LoopyExecutor().differential(calls.matvec, Schedule(calls.matvec), inputs)
    assert fact.status is Status.TESTED
    assert _faithful(calls.matvec).status is Status.TESTED


def test_the_generated_code_has_the_rules_loop_inside_the_callers() -> None:
    code = emit_code(calls.matvec)
    assert "for (int32_t k = 0; k <= -1 + n; ++k)" in code
    assert "A[n * i + k] * x[k]" in code


def test_a_rule_on_its_own_writes_its_value_into_its_result() -> None:
    term = calls.dot.term
    assert term.result == "result"
    assert term.params[-1][0] == "result"
    (stmt,) = term.stmts
    assert stmt.assignee.array == "result" and stmt.inames == ()
    a, b = RNG.standard_normal(6), RNG.standard_normal(6)
    value = calls.dot(Arr.from_numpy(a), Arr.from_numpy(b))
    assert value == pytest.approx(a @ b)
    out = LoopyExecutor().run(calls.dot, a=Arr.from_numpy(a), b=Arr.from_numpy(b))
    assert out["result"][0] == pytest.approx(a @ b)
    # the result array is the term's, and no fact is about it
    assert not [f for f in calls.dot.facts() if "result[" in f.statement]
    (exactness,) = _facts(calls.dot, "exactness")
    assert exactness.statement == (
        "the accumulation into the value dot returns over k is approx"
    )
    assert _faithful(calls.dot).status is Status.TESTED


def test_a_binder_named_like_a_loop_of_the_caller_gets_a_name_of_its_own() -> None:
    @kernel
    def by_k(
        A: Arr[Fin[m], Fin[n], Real],  # noqa: F821
        x: Arr[Fin[n], Real],  # noqa: F821
        y: Arr[Fin[m], Real],  # noqa: F821
    ):
        for k in y.dom:
            y[k] = calls.dot(A[k], x)

    (stmt,) = by_k.term.stmts
    assert stmt.inames == ("k",)
    assert stmt.expr.inames == ("k_0",)
    assert render(stmt.expr.body) == "A[k, k_0]*x[k_0]"
    A, x = RNG.standard_normal((3, 2)), RNG.standard_normal(2)
    inputs = {"A": Arr.from_numpy(A), "x": Arr.from_numpy(x), "y": Arr.zeros(3)}
    fact = LoopyExecutor().differential(by_k, Schedule(by_k), inputs)
    assert fact.status is Status.TESTED


def test_a_rule_called_inside_a_sum_is_made_at_the_sums_points() -> None:
    term = calls.weighted.term
    (call,) = term.calls
    assert call.inames == ("i", "j")
    (row,) = [
        f for f in _facts(calls.weighted, "in-bounds") if "row passed" in f.statement
    ]
    assert _decided(row).status is Status.DECIDED
    c, B, x = (
        RNG.standard_normal(3),
        RNG.standard_normal((3, 4)),
        RNG.standard_normal(4),
    )
    y = Arr.zeros(2)
    calls.weighted(Arr.from_numpy(c), Arr.from_numpy(B), Arr.from_numpy(x), y)
    np.testing.assert_allclose(y.numpy(), np.full(2, c @ (B @ x)))
    assert _faithful(calls.weighted).status is Status.TESTED


def test_the_row_of_a_ragged_array_and_a_gather_through_it() -> None:
    term = calls.csr.term
    (stmt,) = term.stmts
    assert render(stmt.expr.body) == "val[r, k]*x[col[r, k]]"
    kinds = [(a.param, a.kind, a.spelled) for a in term.calls[0].arguments]
    assert kinds == [("a", "row", "val[r]"), ("b", "gather", "x[col[r]]")]
    counts = np.array([2, 0, 3])
    col = Arr.ragged(counts, values=np.array([0, 2, 1, 0, 2]))
    val = Arr.ragged(counts, values=RNG.standard_normal(5))
    x = RNG.standard_normal(3)
    inputs = {
        "cnt": Arr.from_numpy(counts),
        "col": col,
        "val": val,
        "x": Arr.from_numpy(x),
        "y": Arr.zeros(3),
    }
    fact = LoopyExecutor().differential(calls.csr, Schedule(calls.csr), inputs)
    assert fact.status is Status.TESTED
    gather = next(f for f in _facts(calls.csr) if "x[col[r]] picks" in f.statement)
    assert gather.status is Status.DECIDED and gather.decided_by == "type"


# }}}


# {{{ a kernel that writes, called as a statement


def test_a_kernel_that_writes_is_called_as_a_statement_and_writes_through_the_row() -> (
    None
):
    term = calls.outer.term
    (stmt,) = term.stmts
    assert stmt.inames == ("c", "j", "k")
    assert render(stmt.assignee.indices[0]) == "c"
    assert render(stmt.expr) == "u[cell[c, j]]*u[cell[c, k]]"
    (call,) = term.calls
    assert call.statements == ("S0",) and not call.returns
    u = RNG.standard_normal(4)
    M = Arr.zeros((2, 3, 3))
    calls.outer(Arr.from_numpy(_cells()), Arr.from_numpy(u), M)
    local = u[_cells()]
    np.testing.assert_array_equal(M.numpy(), local[:, :, None] * local[:, None, :])
    assert _faithful(calls.outer).status is Status.TESTED


def test_the_postcondition_of_a_callee_is_restated_at_every_call() -> None:
    (fact,) = _facts(calls.outer, "postcondition-in-scope")
    assert fact.status is Status.DECIDED and fact.decided_by == "call"
    (post,) = _facts(calls.symmetric, "postcondition")
    assert fact.rests_on == (post.id,)
    assert "M[c, 0, 1] == M[c, 1, 0]" in fact.statement
    assert fact.statement.startswith(
        "after each call symmetric(u[cell[c]], M[c]) at calls.py:104:"
    )


def test_a_gather_is_read_only() -> None:
    @kernel
    def zero(u: Arr[Fin[3], Real]):  # noqa: F821
        for k in u.dom:
            u[k] = 0.0

    @kernel
    def zeroing(
        cell: Arr[Fin[nc], Fin[3], Fin[nv]],  # noqa: F821
        u: Arr[Fin[nv], Real],  # noqa: F821
    ):
        for c in cell.dom:
            zero(u[cell[c]])

    with pytest.raises(TraceError, match="natively u\\[cell\\[c\\]\\] is a copy"):
        zeroing.term


def test_a_row_of_the_cells_a_gather_picks_is_the_gather_by_the_row() -> None:
    # numpy reads X[cell][c] as X[cell[c]]: so does the trace, and a part that
    # fixes an axis of the source is refused, naming the row to pass
    @kernel
    def weighted_corners(X: Arr[Fin[3], Real]) -> Real:  # noqa: F821
        return X[0] + 2.0 * X[1] + 3.0 * X[2]

    @kernel
    def by_rows(
        cell: Arr[Fin[nc], Fin[3], Fin[nv]],  # noqa: F821
        X: Arr[Fin[nv], Real],  # noqa: F821
        a: Arr[Fin[nc], Real],  # noqa: F821
    ):
        for c in a.dom:
            a[c] = weighted_corners(X[cell][c])

    (call,) = by_rows.term.calls
    (argument,) = call.arguments
    assert (argument.kind, argument.spelled) == ("gather", "X[cell[c]]")
    picks = [f for f in _facts(by_rows) if "picks" in f.statement]
    assert [f.decided_by for f in picks] == ["type"]
    X = RNG.standard_normal(4)
    a = Arr.zeros(2)
    by_rows(Arr.from_numpy(_cells()), Arr.from_numpy(X), a)
    np.testing.assert_allclose(a.numpy(), X[_cells()] @ [1.0, 2.0, 3.0])
    inputs = {
        "cell": Arr.from_numpy(_cells()),
        "X": Arr.from_numpy(X),
        "a": Arr.zeros(2),
    }
    fact = LoopyExecutor().differential(by_rows, Schedule(by_rows), inputs)
    assert fact.status is Status.TESTED

    @kernel
    def apart(P: Arr[Fin[2], Real]) -> Real:  # noqa: F821
        return P[0] - P[1]

    @kernel
    def by_corner(
        cell: Arr[Fin[nc], Fin[3], Fin[nv]],  # noqa: F821
        xy: Arr[Fin[nv], Fin[2], Real],  # noqa: F821
        a: Arr[Fin[nc], Real],  # noqa: F821
    ):
        for c in a.dom:
            a[c] = apart(xy[cell][c, 0])

    with pytest.raises(TraceError, match=r"pass xy\[cell\[c, 0\]\], the row of xy"):
        by_corner.term


# }}}


# {{{ what a call owes, at every point it is made


def test_a_row_is_in_bounds_wherever_its_guard_keeps_it() -> None:
    (row,) = [f for f in _facts(calls.shifted) if "row passed" in f.statement]
    assert _decided(row).status is Status.DECIDED
    # natively the call under the false guard is not made: its contract
    # would refuse A[m], where the compiled kernel skips the instance
    A, x = RNG.standard_normal((3, 2)), RNG.standard_normal(2)
    y = Arr.zeros(3)
    calls.shifted(Arr.from_numpy(A), Arr.from_numpy(x), y)
    np.testing.assert_allclose(y.numpy(), [A[1] @ x, A[2] @ x, 0.0])
    assert _faithful(calls.shifted).status is Status.TESTED


def test_a_row_past_the_end_is_refuted_with_the_point() -> None:
    (row,) = [f for f in _facts(calls.off_the_end) if "row passed" in f.statement]
    refuted = _decided(row)
    assert refuted.status is Status.REFUTED
    assert refuted.provenance["witness"] == (1,)


def test_an_integral_argument_is_a_point_of_its_sort() -> None:
    (fact,) = _facts(calls.mass_action, "argument")
    assert fact.statement.startswith(
        "j, passed to p1_mass as j, is a point of Fin(3) wherever p1_mass is called"
    )
    assert _decided(fact).status is Status.DECIDED
    (wrong,) = _facts(calls.off_the_sort, "argument")
    refuted = _decided(wrong)
    assert refuted.status is Status.REFUTED
    assert refuted.provenance["witness"] == (0, 2)


def test_two_rows_of_one_array_are_two_rows_where_one_is_written() -> None:
    (apart,) = _facts(calls.running, "distinct-arguments")
    assert _decided(apart).status is Status.DECIDED
    assert apart.statement.startswith(
        "A[i] and A[i - 1], passed to add_into as a and b, share no cell of A"
    )
    A = RNG.standard_normal((4, 3))
    out = Arr.from_numpy(A.copy())
    calls.running(out)
    np.testing.assert_allclose(out.numpy(), np.cumsum(A, axis=0))
    (together,) = _facts(calls.off_the_diagonal, "distinct-arguments")
    refuted = _decided(together)
    assert refuted.status is Status.REFUTED
    assert refuted.provenance["witness"] == (0, 0)


def test_a_callee_that_only_reads_may_be_given_one_array_twice() -> None:
    # The diagonal of a Gram matrix dots a row with itself: reads do not race,
    # so neither the callee's contract nor the caller's facts ask for apart.
    assert not _facts(calls.gram, "distinct-arguments")
    A = RNG.standard_normal((3, 4))
    G = Arr.zeros((3, 3))
    calls.gram(Arr.from_numpy(A), G)
    np.testing.assert_allclose(G.numpy(), A @ A.T)
    inputs = {"A": Arr.from_numpy(A), "G": Arr.zeros((3, 3))}
    fact = LoopyExecutor().differential(calls.gram, Schedule(calls.gram), inputs)
    assert fact.status is Status.TESTED
    assert _faithful(calls.gram).status is Status.TESTED
    # natively, one array for both parameters of a rule; and compiled
    x = RNG.standard_normal(5)
    assert calls.dot(Arr.from_numpy(x), Arr.from_numpy(x)) == pytest.approx(x @ x)
    shared = Arr.from_numpy(x.copy())
    out = LoopyExecutor().run(calls.dot, a=shared, b=shared)
    assert out["result"][0] == pytest.approx(x @ x)

    @kernel
    def norms(A: Arr[Fin[m], Fin[n], Real], y: Arr[Fin[m], Real]):  # noqa: F821
        for i in y.dom:
            y[i] = calls.dot(A[i], A[i])

    y = Arr.zeros(3)
    norms(Arr.from_numpy(A), y)
    np.testing.assert_allclose(y.numpy(), (A * A).sum(axis=1))
    # a callee that writes one of the two is still refused, natively too
    with pytest.raises(ValueError, match="share storage"):
        calls.add_into(shared, shared)


def test_a_call_inside_a_sum_over_a_fin_says_its_points_are_unknown() -> None:
    @kernel
    def summed(
        A: Arr[Fin[m], Fin[n], Real],  # noqa: F821
        x: Arr[Fin[n], Real],  # noqa: F821
        y: Arr[Fin[m], Real],  # noqa: F821
    ):
        for i in y.dom:
            y[i] = reduce_sum(calls.dot(A[j], x) for j in Fin[y.dom.size])

    (row,) = [f for f in _facts(summed) if "row passed" in f.statement]
    assert row.status is Status.ASSUMED
    assert "a binder the trace does not bind itself" in row.provenance["reason"]


def test_the_argument_facts_are_stated_for_lean_from_the_call() -> None:
    # lanky's obligations are newer than its 0.2.0 release, which CI tests too
    pytest.importorskip(
        "lanky.obligations", reason="the released lanky states no Lean obligations"
    )
    from lanky.lean import statement_of

    (fact,) = _facts(calls.mass_action, "argument")
    obligation = fact.term.lean_obligation()
    (part,) = obligation.parts
    statement = statement_of(part.term, "mass_action_argument", mathlib=False)
    assert [name for name, _ in statement.binders] == ["nc", "c", "j"]
    assert statement.goal == "0 ≤ j ∧ j < 3"
    assert obligation.decider == "isl"


# }}}


# {{{ rules taking rules


def test_a_rule_takes_a_function_and_calls_it_where_it_uses_it() -> None:
    term = calls.mass_action.term
    labels = [call.label for call in term.calls]
    assert labels == ["integrate", "p1_mass"]
    u = RNG.standard_normal(4)
    yl = Arr.zeros((2, 3))
    calls.mass_action(Arr.from_numpy(_cells()), Arr.from_numpy(u), yl)
    local = u[_cells()]
    np.testing.assert_allclose(yl.numpy(), (local + local.sum(1)[:, None]) / 24)
    inputs = {
        "cell": Arr.from_numpy(_cells()),
        "u": Arr.from_numpy(u),
        "yl": Arr.zeros((2, 3)),
    }
    fact = LoopyExecutor().differential(
        calls.mass_action, Schedule(calls.mass_action), inputs
    )
    assert fact.status is Status.TESTED


def test_a_rule_that_takes_a_function_is_checked_on_drawn_tables() -> None:
    fact = _faithful(calls.integrate)
    assert fact.status is Status.TESTED
    assert calls.integrate(lambda q: q + 1.0) == pytest.approx(1.0)


def test_loopty_run_compiles_a_rule_that_takes_a_function_where_it_is_called() -> None:
    from loopty.calls import takes_a_function

    assert takes_a_function(calls.integrate) == "f"
    assert takes_a_function(calls.dot) is None


# }}}


# {{{ the calls refused, each naming the fix


def test_sizes_that_cannot_be_shown_to_agree_are_refused() -> None:
    @kernel
    def other_size(
        A: Arr[Fin[m], Fin[n], Real],  # noqa: F821
        z: Arr[Fin[p], Real],  # noqa: F821
        y: Arr[Fin[m], Real],  # noqa: F821
    ):
        for i in y.dom:
            y[i] = calls.dot(A[i], z)

    with pytest.raises(TraceError, match="axis 0 of its b is n long there, and z"):
        other_size.term


def test_an_array_and_a_part_of_it_passed_together_are_refused() -> None:
    @kernel
    def into(u: Arr[Fin[n], Real], M: Arr[Fin[n], Fin[n], Real]):  # noqa: F821
        for j in M.dom:
            for k in M.dom[j]:
                M[j, k] = u[j]

    @kernel
    def overlap(A: Arr[Fin[m], Fin[m], Real]):  # noqa: F821
        for i in A.dom:
            into(A[i], A)

    with pytest.raises(TraceError, match="share the cells of A"):
        overlap.term


def test_a_rule_writes_nothing_and_returns_its_value() -> None:
    @kernel
    def writes(x: Arr[Fin[n], Real]) -> Real:  # noqa: F821
        x[0] = 1.0
        return x[0]

    with pytest.raises(TraceError, match="writes none of its parameters"):
        writes.term

    @kernel
    def nothing(x: Arr[Fin[n], Real]) -> Real:  # noqa: F821
        x.dom

    with pytest.raises(TraceError, match="its body returns nothing"):
        nothing.term

    @kernel
    def undeclared(x: Arr[Fin[n], Real]):  # noqa: F821
        return reduce_sum(x[k] for k in x.dom)

    with pytest.raises(TraceError, match="declares the sort of the value"):
        undeclared.term


def test_a_claim_about_a_rules_value_is_refused_naming_the_fix() -> None:
    @kernel
    def positive(x: Arr[Fin[n], Real]) -> Real & (x[0] > 0):  # noqa: F821
        return x[0]

    with pytest.raises(TraceError, match="State it as a theorem"):
        positive.term


def test_a_kernel_does_not_call_itself() -> None:
    @kernel
    def again(x: Arr[Fin[n], Real]) -> Real:  # noqa: F821
        return again(x)

    with pytest.raises(TraceError, match="a kernel does not call itself"):
        again.term

    @kernel
    def ping(x: Arr[Fin[n], Real]) -> Real:  # noqa: F821
        return pong(x)

    @kernel
    def pong(x: Arr[Fin[n], Real]) -> Real:  # noqa: F821
        return ping(x) + 1.0

    with pytest.raises(TraceError, match="a kernel does not call itself"):
        ping.term

    def forever(q):
        return calls.integrate(forever)

    @kernel
    def through(y: Arr[Fin[1], Real]):  # noqa: F821
        for z in y.dom:
            y[z] = calls.integrate(forever)

    with pytest.raises(TraceError, match="with the same functions"):
        through.term


def test_a_rule_applied_to_a_function_that_applies_it_again() -> None:
    # Nested quadrature: the rule inlined once per lambda, its binder named
    # apart the second time; not a kernel calling itself.
    @kernel
    def double(y: Arr[Fin[1], Real]):  # noqa: F821
        for z in y.dom:
            y[z] = calls.integrate(
                lambda q: calls.integrate(lambda p: calls.PHI[p, q] * calls.PHI[q, p])
            )

    assert [call.label for call in double.term.calls] == ["integrate@2", "integrate"]
    phi, w = np.array(calls.PHI.values), np.array(calls.W.values)
    want = w @ (phi * phi.T) @ w
    y = Arr.zeros(1)
    double(y)
    assert y.numpy()[0] == pytest.approx(want)
    fact = LoopyExecutor().differential(double, Schedule(double), {"y": Arr.zeros(1)})
    assert fact.status is Status.TESTED
    assert _faithful(double).status is Status.TESTED


def test_a_program_calls_kernels_and_not_rules() -> None:
    @program
    def ruled(a, b):
        calls.dot(a, b)

    with pytest.raises(TraceError, match="is a rule"):
        ruled.term


def test_a_kernel_calls_kernels_and_not_programs() -> None:
    @kernel
    def scaled(x: Arr[Fin[n], Real]):  # noqa: F821
        for i in x.dom:
            x[i] = 2.0 * x[i]

    @program
    def twice(x):
        scaled(x)
        scaled(x)

    @kernel
    def calls_a_program(x: Arr[Fin[n], Real]):  # noqa: F821
        twice(x)

    with pytest.raises(TraceError, match="calls the program twice"):
        calls_a_program.term


def test_a_scalar_and_a_function_are_given_what_they_are() -> None:
    @kernel
    def given_an_array(
        u: Arr[Fin[nv], Real],  # noqa: F821
        yl: Arr[Fin[3], Real],  # noqa: F821
    ):
        for j in yl.dom:
            yl[j] = calls.p1_mass(u, u)

    with pytest.raises(TraceError, match="scalar parameter j is given"):
        given_an_array.term

    @kernel
    def given_a_number(y: Arr[Fin[1], Real]):  # noqa: F821
        y[0] = calls.integrate(2.0)

    with pytest.raises(TraceError, match="pass a function of the point"):
        given_a_number.term


def test_an_argument_is_read_once_when_the_call_is_made() -> None:
    # Natively s is A[i, 0] when the call is made; inlined, A[i, 0] would be
    # read again at every k, after the callee wrote A[i, 0] / s into it.
    @kernel
    def scale(a: Arr[Fin[n], Real], s: Real):  # noqa: F821
        for k in a.dom:
            a[k] = a[k] / s

    @kernel
    def normalize(A: Arr[Fin[m], Fin[n], Real]):  # noqa: F821
        for i in A.dom:
            scale(A[i], A[i, 0])

    with pytest.raises(TraceError, match="natively s is the value A\\[i, 0\\] has"):
        normalize.term

    # the cells a gather copies are picked by indices the callee writes
    @kernel
    def renumber(
        a: Arr[Fin[3], Real],  # noqa: F821
        idx: Arr[Fin[3], Fin[nv]],  # noqa: F821
        out: Arr[Fin[3], Real],  # noqa: F821
    ):
        for k in idx.dom:
            idx[k] = 0
        for k in out.dom:
            out[k] = a[k]

    @kernel
    def gathered(
        cell: Arr[Fin[nc], Fin[3], Fin[nv]],  # noqa: F821
        u: Arr[Fin[nv], Real],  # noqa: F821
        o: Arr[Fin[nc], Fin[3], Real],  # noqa: F821
    ):
        for c in o.dom:
            renumber(u[cell[c]], cell[c], o[c])

    with pytest.raises(TraceError, match="which reads cell, and it writes cell"):
        gathered.term

    # a value the callee does not write may be passed by its expression
    @kernel
    def by_first(A: Arr[Fin[m], Fin[n], Real], B: Arr[Fin[m], Fin[n], Real]):  # noqa: F821
        for i in A.dom:
            scale(A[i], B[i, 0])

    A, B = RNG.standard_normal((2, 3)), RNG.uniform(1.0, 2.0, (2, 3))
    inputs = {"A": Arr.from_numpy(A), "B": Arr.from_numpy(B)}
    fact = LoopyExecutor().differential(by_first, Schedule(by_first), inputs)
    assert fact.status is Status.TESTED


def test_a_kernel_that_writes_is_not_called_inside_a_sum() -> None:
    @kernel
    def writer(u: Arr[Fin[n], Real]):  # noqa: F821
        for k in u.dom:
            u[k] = 0.0

    @kernel
    def inside(A: Arr[Fin[m], Fin[n], Real], y: Arr[Fin[m], Real]):  # noqa: F821
        for i in y.dom:
            y[i] = reduce_sum(writer(A[j]) for j in A.dom)

    with pytest.raises(TraceError, match="writes an array inside a sum"):
        inside.term


# }}}


# {{{ with the rest of loopty


def test_a_program_calls_a_kernel_that_calls_a_rule() -> None:
    @kernel
    def double(x: Arr[Fin[n], Real], y: Arr[Fin[n], Real]):  # noqa: F821
        for i in y.dom:
            y[i] = 2.0 * x[i]

    @program
    def solve(A, x, y):
        z = Arr.zeros_like(x)
        double(x, z)
        calls.matvec(A, z, y)

    A, x = RNG.standard_normal((3, 2)), RNG.standard_normal(2)
    inputs = {"A": Arr.from_numpy(A), "x": Arr.from_numpy(x), "y": Arr.zeros(3)}
    fact = LoopyExecutor().differential(solve, Schedule(solve), inputs)
    assert fact.status is Status.TESTED


def test_a_callees_names_never_capture_the_callers() -> None:
    # The claim's binder j would capture the scalar j the row M[j] is at.
    @kernel
    def summed(
        Me: Arr[Fin[p], Fin[p], Real],  # noqa: F821
    ) -> all(Me[j, k] == Me[k, j] for j in Fin[p] for k in Fin[p]):  # noqa: F821
        for j in Me.dom:
            for k in Me.dom[j]:
                Me[j, k] = 1.0 * (j + k)

    @kernel
    def at(M: Arr[Fin[q], Fin[r], Fin[r], Real], j: Fin[q]):  # noqa: F821
        summed(M[j])

    (call,) = at.term.calls
    assert render(call.post) == (
        "forall j_1 in Fin(r), k_0 in Fin(r). M[j, j_1, k_0] == M[j, k_0, j_1]"
    )
    # the callee's sum binder k, a loop k of the caller's, and a size k
    @kernel
    def by_k(
        A: Arr[Fin[m], Fin[n], Real],  # noqa: F821
        x: Arr[Fin[n], Real],  # noqa: F821
        y: Arr[Fin[m], Real],  # noqa: F821
    ):
        for k in y.dom:
            y[k] = calls.dot(A[k], x)

    (stmt,) = by_k.term.stmts
    assert render(stmt.expr.body) == "A[k, k_0]*x[k_0]"
    A, x = RNG.standard_normal((4, 3)), RNG.standard_normal(3)
    inputs = {"A": Arr.from_numpy(A), "x": Arr.from_numpy(x), "y": Arr.zeros(4)}
    fact = LoopyExecutor().differential(by_k, Schedule(by_k), inputs)
    assert fact.status is Status.TESTED

    @kernel
    def fill(a: Arr[Fin[p], Real], v: Real):  # noqa: F821
        for n in a.dom:
            a[n] = v

    @kernel
    def rows(M: Arr[Fin[n], Fin[q], Real]):  # noqa: F821
        for i in M.dom:
            fill(M[i], 1.0 * i)

    assert rows.term.stmts[0].inames == ("i", "n_0")
    with pytest.raises(ValueError, match="fill's loop n, inlined at .*, is n_0 here"):
        Schedule(rows).split("n", 2)
    schedule = Schedule(rows).split("n_0", 2)
    fact = LoopyExecutor().differential(rows, schedule, {"M": Arr.zeros((3, 5))})
    assert fact.status is Status.TESTED


def test_a_schedule_of_a_caller_is_checked_and_runs() -> None:
    schedule = Schedule(calls.matvec).split("i", 2)
    assert all(f.status is Status.DECIDED for f in schedule.facts())
    A, x = RNG.standard_normal((5, 3)), RNG.standard_normal(3)
    inputs = {"A": Arr.from_numpy(A), "x": Arr.from_numpy(x), "y": Arr.zeros(5)}
    fact = LoopyExecutor().differential(calls.matvec, schedule, inputs)
    assert fact.status is Status.TESTED


def test_a_row_of_a_table_is_passed_as_an_array() -> None:
    rows = table([[1.0, 2.0], [3.0, 4.0]])

    @kernel
    def by_rows(x: Arr[Fin[2], Real], y: Arr[Fin[2], Real]):  # noqa: F821
        for i in y.dom:
            y[i] = calls.dot(rows[i], x)

    x = np.array([1.0, -1.0])
    y = Arr.zeros(2)
    by_rows(Arr.from_numpy(x), y)
    np.testing.assert_allclose(y.numpy(), [-1.0, -1.0])
    inputs = {"x": Arr.from_numpy(x), "y": Arr.zeros(2)}
    fact = LoopyExecutor().differential(by_rows, Schedule(by_rows), inputs)
    assert fact.status is Status.TESTED


def test_a_row_of_a_table_under_a_false_guard_is_not_read() -> None:
    rows = table([[1.0, 2.0], [3.0, 4.0]])

    @kernel
    def next_rows(x: Arr[Fin[2], Real], y: Arr[Fin[2], Real]):  # noqa: F821
        for i in y.dom:
            with when(i + 1 < 2):
                y[i] = calls.dot(rows[i + 1], x)

    x = np.array([1.0, -1.0])
    y = Arr.zeros(2)
    next_rows(Arr.from_numpy(x), y)
    np.testing.assert_allclose(y.numpy(), [-1.0, 0.0])
    (row,) = [f for f in _facts(next_rows) if "row passed" in f.statement]
    assert _decided(row).status is Status.DECIDED
    inputs = {"x": Arr.from_numpy(x), "y": Arr.zeros(2)}
    fact = LoopyExecutor().differential(next_rows, Schedule(next_rows), inputs)
    assert fact.status is Status.TESTED
    with pytest.raises(IndexError):
        rows[2]


def test_a_scalar_sizes_the_callees_arrays() -> None:
    @kernel
    def first(x: Arr[Fin[m], Real], m: Nat) -> Real:  # noqa: F821
        return reduce_sum(x[k] for k in x.dom)

    @kernel
    def rows(A: Arr[Fin[r], Fin[c], Real], y: Arr[Fin[r], Real]):  # noqa: F821
        for i in y.dom:
            y[i] = first(A[i], A.dom[i].size)

    A = RNG.standard_normal((3, 4))
    y = Arr.zeros(3)
    rows(Arr.from_numpy(A), y)
    np.testing.assert_allclose(y.numpy(), A.sum(axis=1))
    assert _faithful(rows).status is Status.TESTED


def test_a_rule_under_a_false_guard_is_not_called_natively() -> None:
    @kernel
    def picky(a: Arr[Fin[n], Real], i: Fin[n]) -> Real:  # noqa: F821
        return a[i]

    @kernel
    def guarded(A: Arr[Fin[m], Fin[n], Real], y: Arr[Fin[m], Real]):  # noqa: F821
        for i in y.dom:
            with when(i < A.dom[i].size):
                y[i] = picky(A[i], i)

    A = RNG.standard_normal((4, 2))
    y = Arr.zeros(4)
    guarded(Arr.from_numpy(A), y)
    np.testing.assert_allclose(y.numpy(), [A[0, 0], A[1, 1], 0.0, 0.0])
    (fact,) = _facts(guarded, "argument")
    assert _decided(fact).status is Status.DECIDED


# }}}


def test_one_index_row_two_gathers_read_through_is_one_fact() -> None:
    @kernel
    def corners(
        X: Arr[Fin[3], Real],  # noqa: F821
        Y: Arr[Fin[3], Real],  # noqa: F821
    ) -> Real:
        return X[0] * Y[1] - X[1] * Y[0]

    @kernel
    def twice(
        cell: Arr[Fin[nc], Fin[3], Fin[nv]],  # noqa: F821
        X: Arr[Fin[nv], Real],  # noqa: F821
        Y: Arr[Fin[nv], Real],  # noqa: F821
        a: Arr[Fin[nc], Real],  # noqa: F821
    ):
        for c in a.dom:
            a[c] = corners(X[cell[c]], Y[cell[c]])

    facts = _facts(twice)
    assert len({fact.id for fact in facts}) == len(facts)
    (index,) = [f for f in facts if "the indices of the cells" in f.statement]
    assert index.statement.startswith(
        "cell[c], the indices of the cells passed as X and Y, is in bounds"
    )


def test_an_array_over_a_domain_is_passed_whole() -> None:
    @kernel
    def row_sum(
        f: Arr[Where[i : Fin[n], j : Fin[n], j < i], Real],  # noqa: F821
        r: Fin[n],  # noqa: F821
    ) -> Real:
        return reduce_sum(f[r, q] for q in f.dom[r])

    @kernel
    def rows(
        f: Arr[Where[a : Fin[m], b : Fin[m], b < a], Real],  # noqa: F821
        e: Arr[Fin[m], Real],  # noqa: F821
    ):
        for p in e.dom:
            e[p] = row_sum(f, p)

    (stmt,) = rows.term.stmts
    assert render(stmt.expr.body) == "f[p, q]"
    (argument,) = _facts(rows, "argument")
    assert argument.statement.startswith(
        "p, passed to row_sum as r, is a point of Fin(m)"
    )
    assert _decided(argument).status is Status.DECIDED
    domain = rows.arg_types["f"].domain
    f = Arr.from_cells(domain, np.arange(6.0), m=4)
    e = Arr.zeros(4)
    rows(f, e)
    np.testing.assert_allclose(e.numpy(), [0.0, 0.0, 3.0, 12.0])
    inputs = {"f": Arr.from_cells(domain, np.arange(6.0), m=4), "e": Arr.zeros(4)}
    fact = LoopyExecutor().differential(rows, Schedule(rows), inputs)
    assert fact.status is Status.TESTED

    @kernel
    def other_points(
        f: Arr[Where[a : Fin[m], b : Fin[m], b <= a], Real],  # noqa: F821
        e: Arr[Fin[m], Real],  # noqa: F821
    ):
        for p in e.dom:
            e[p] = row_sum(f, p)

    with pytest.raises(TraceError, match="nothing says the two have the same points"):
        other_points.term
