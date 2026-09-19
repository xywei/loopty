"""The Term IR: the contract between tracing and lowering."""

from __future__ import annotations

import dataclasses

import islpy as isl
import pymbolic.primitives as prim
import pytest

from loopty.term import Access, ArrType, Reduction, Stmt, Term

r = prim.Variable("r")
j = prim.Variable("j")


def a_stmt() -> Stmt:
    """The spmv row statement, as tracing would record it."""
    return Stmt(
        id="s0",
        inames=("r",),
        domain=isl.Set("[n] -> { [r] : 0 <= r < n }"),
        assignee=Access("y", (r,)),
        expr=Reduction(
            op="sum",
            inames=("j",),
            domain=isl.Set("[cnt_r] -> { [j] : 0 <= j < cnt_r }"),
            body=prim.Variable("val")[r, j] * prim.Variable("x")[r],
            exactness="reassoc",
        ),
        kind="assign",
        guard=None,
        where="spmv.py:17",
    )


def test_access_indices_are_in_index_type_axes() -> None:
    access = Access("val", (r, j))
    assert access.array == "val"
    assert len(access.indices) == 2


def test_stmt_carries_its_domain_and_source_line() -> None:
    stmt = a_stmt()
    assert stmt.inames == ("r",)
    assert stmt.domain.get_var_names(isl.dim_type.param) == ["n"]
    assert stmt.where.endswith(":17")
    assert stmt.assignee.array == "y"


def test_reduction_records_its_exactness_class() -> None:
    stmt = a_stmt()
    assert isinstance(stmt.expr, Reduction)
    assert stmt.expr.op == "sum"
    assert stmt.expr.exactness == "reassoc"
    assert stmt.expr.inames == ("j",)


def test_terms_are_frozen_values() -> None:
    stmt = a_stmt()
    with pytest.raises(dataclasses.FrozenInstanceError):
        stmt.kind = "accumulate"  # type: ignore[misc]
    # A transformation builds a new statement instead.
    accumulating = dataclasses.replace(stmt, kind="accumulate")
    assert accumulating.kind == "accumulate"
    assert stmt.kind == "assign"


def test_arrtype_axes_and_raggedness_must_agree() -> None:
    dense = ArrType(axes=(4, 3), dtype="Real", ragged=(False, False))
    assert dense.ndim == 2
    with pytest.raises(ValueError, match="raggedness flags"):
        ArrType(axes=(4, 3), dtype="Real", ragged=(False,))


def test_ragged_arrtype_names_the_counts_array() -> None:
    val = ArrType(
        axes=(prim.Variable("n"), prim.Variable("cnt")),
        dtype="Real",
        ragged=(False, True),
    )
    assert val.ragged == (False, True)
    assert str(val.axes[1]) == "cnt"


def test_term_lists_params_in_signature_order() -> None:
    term = Term(
        name="spmv",
        params=(
            ("val", ArrType(axes=(prim.Variable("n"),), dtype="Real", ragged=(False,))),
            ("y", ArrType(axes=(prim.Variable("n"),), dtype="Real", ragged=(False,))),
        ),
        sizes=("n",),
        stmts=(a_stmt(),),
        post=None,
    )
    assert term.param_names == ("val", "y")
    assert term.stmt("s0").where.endswith(":17")
    with pytest.raises(KeyError):
        term.stmt("nope")
