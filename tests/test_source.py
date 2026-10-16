"""A statement's source text, which a refusal names its instances by (#260)."""

from __future__ import annotations

import textwrap

import pytest
from lanky.prelude import Real
from lanky.terms import evaluate_annotations

from lines import where
from loopty import Arr, Fin, Schedule, reduce_sum
from loopty.handles import scope
from loopty.schedule import IllegalCast
from loopty.source import LIMIT, instances_text
from loopty.trace import trace


def term_of(fn):
    """Trace a plain function the way ``@kernel`` would."""
    return trace(fn, evaluate_annotations(fn))


def two_on_a_line(
    x: Arr[Fin[n], Real],  # noqa: F821
    a: Arr[Fin[n], Real],  # noqa: F821
    b: Arr[Fin[n], Real],  # noqa: F821
):
    for i in x.dom:
        a[i] = x[i]; b[i] = 2.0 * x[i]  # noqa: E702


def long_ones(
    x: Arr[Fin[n], Fin[n], Real],  # noqa: F821
    y: Arr[Fin[n], Real],  # noqa: F821
    z: Arr[Fin[n], Real],  # noqa: F821
):
    for i in y.dom:
        y[i] = reduce_sum(
            x[i, j] * x[j, i] * x[i, i] * x[j, j] + x[i, j] for j in x.dom[i]
        )
        z[i] = (
            x[i, i] * x[i, i] * x[i, i] * x[i, i] * x[i, i] * x[i, i] * x[i, i]
            + y[i]
        )


def test_each_statement_keeps_its_text_and_the_names_of_its_loops() -> None:
    first, second = term_of(two_on_a_line).stmts
    # two statements on one line are told apart by where each one is
    assert first.text == "a[i] = x[i]"
    assert second.text == "b[i] = 2.0 * x[i]"
    assert first.spelled == second.spelled == ("i",)
    assert first.where == where(two_on_a_line, "a[i] = x[i]")


def test_a_long_statement_keeps_what_it_writes_into() -> None:
    summed, scaled = term_of(long_ones).stmts
    # the calls on the right are shortened first, written over three lines
    assert summed.text == "y[i] = reduce_sum(...)"
    # and a right side with no call to shorten is left out whole
    assert scaled.text == "z[i] = ..."
    assert all(len(stmt.text) <= LIMIT for stmt in (summed, scaled))


def test_a_body_with_no_file_is_named_by_its_ids() -> None:
    # A body made by exec has no source to read: its statements have no text,
    # and a refusal names their instances by id, as a term written by hand.
    namespace: dict = {}
    exec(  # noqa: S102 - the point of the test
        textwrap.dedent(
            """
            from __future__ import annotations

            from lanky.prelude import Real
            from loopty import Arr, Fin

            def chain(x: Arr[Fin[n], Real]):
                for i in x.dom:
                    x[i] = x[i] + x[i - 1]
            """
        ),
        namespace,
    )
    chain = namespace["chain"]
    term = term_of(chain)
    (stmt,) = term.stmts
    assert stmt.text == ""
    with pytest.raises(IllegalCast) as caught:
        Schedule(term, sizes={"n": 4}).tag(scope(term).i, "g.0")
    assert str(caught.value).startswith("tag(chain.i, 'g.0') illegal: S0[i=")
    assert ", read by S0[i=" in str(caught.value)


def test_two_instances_of_one_statement_name_it_once() -> None:
    term = term_of(two_on_a_line)
    stmts = {stmt.id: stmt for stmt in term.stmts}
    at = where(two_on_a_line, "a[i] = x[i]")
    assert instances_text(stmts, ("S0", {"i": 1}), ("S0", {"i": 2})) == [
        f"a[i] = x[i] ({at}) at i=1",
        "the same statement at i=2",
    ]
    assert instances_text(stmts, ("S1", {"i": 0}), ("S0", {"i": 3})) == [
        f"b[i] = 2.0 * x[i] ({at}) at i=0",
        f"a[i] = x[i] ({at}) at i=3",
    ]
