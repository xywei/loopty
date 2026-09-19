"""Lowering a term to loopy, and running what comes out.

The terms are built by hand (see ``hand_terms``) so that this half of the
compiler can be tested without the tracer. Two questions are asked of each one:
does loopy generate code for it, and does that code compute what numpy computes.
The second question is the only one that catches a lowering that is plausible and
wrong, which is why every shape here is run and not merely generated.
"""

from __future__ import annotations

import numpy as np
import pytest

import hand_terms as ht
from loopty.lower import (
    LoweringError,
    count_param_names,
    lower,
    lower_generic,
    numpy_dtype,
    target_for,
)

lp = pytest.importorskip("loopy")


def code_for(term) -> str:
    """The C loopy generates for a term."""
    return lp.generate_code_v2(lower(term)).device_code()


def run(term, **arguments):
    """Run a term on the C target, or skip if the toolchain is unusable."""
    from loopty.executor import LoopyExecutor

    try:
        return LoopyExecutor().run(term, **arguments)
    except Exception as exc:  # pragma: no cover - depends on the local toolchain
        if "compil" in str(exc).lower() or isinstance(exc, OSError):
            pytest.skip(f"the C toolchain path is unusable here: {exc}")
        raise


def test_every_hand_built_term_generates_code() -> None:
    for term in (
        ht.axpy_term(),
        ht.transpose_term(),
        ht.jacobi_term(),
        ht.spmv_term(),
        ht.spmv_accumulate_term(),
    ):
        code = code_for(term)
        assert term.name in code


def test_axpy_runs_and_matches_numpy() -> None:
    x = np.arange(8, dtype=np.float64)
    y = np.ones(8)
    z = np.zeros(8)
    out = run(ht.axpy_term(), a=2.0, x=x, y=y, z=z)
    assert np.allclose(out["z"], 2.0 * x + y)
    # Outputs are parameters: the array that was passed in is the one written.
    assert np.allclose(z, 2.0 * x + y)


def test_transpose_runs_and_matches_numpy() -> None:
    a = np.arange(6, dtype=np.float64).reshape(2, 3)
    b = np.zeros((3, 2))
    out = run(ht.transpose_term(), a=a, b=b)
    assert np.array_equal(out["b"], a.T)


def test_the_stencil_runs_in_the_order_the_term_was_written() -> None:
    # loopy is free to choose a nest, and for this term its choice reverses a
    # dependence; lowering pins the nest to the traced order instead.
    u = np.zeros((6, 6))
    u[0] = np.arange(6.0)
    out = run(ht.jacobi_term(), u=u.copy())
    assert np.allclose(out["u"], ht.jacobi_reference(u))


def test_a_ragged_reduction_becomes_a_csr_loop() -> None:
    code = code_for(ht.spmv_term())
    assert "off[r] + j" in code.replace("  ", " ")
    assert "cnt_r" in code

    arrays = ht.csr_example()
    out = run(ht.spmv_term(), **arrays)
    want = ht.csr_reference(
        arrays["off"], arrays["col"], arrays["val"], arrays["x"]
    )
    assert np.allclose(out["y"], want)


def test_an_accumulation_runs_over_the_ragged_nest() -> None:
    arrays = ht.csr_example()
    out = run(ht.spmv_accumulate_term(), **arrays)
    want = ht.csr_reference(
        arrays["off"], arrays["col"], arrays["val"], arrays["x"]
    )
    assert np.allclose(out["y"], want)


def test_an_accumulation_whose_expression_is_complete_is_not_doubled() -> None:
    # A traced accumulation records the whole right-hand side, ``y[i] + ...``,
    # and a hand-written one records only the increment. Both must lower to one
    # addition, not two.
    import islpy as isl
    import pymbolic.primitives as prim

    from loopty.term import Access, Stmt, Term

    domain = isl.Set("[n] -> { [i] : 0 <= i < n }")
    written = Stmt(
        id="S0",
        inames=("i",),
        domain=domain,
        assignee=Access("y", (prim.Variable("i"),)),
        expr=ht.S("y", prim.Variable("i")) + ht.S("x", prim.Variable("i")),
        kind="accumulate",
        guard=None,
        where="test:1",
    )
    term = Term(
        name="acc_written_out",
        params=(
            ("x", ht.dense(prim.Variable("n"))),
            ("y", ht.dense(prim.Variable("n"))),
        ),
        sizes=("n",),
        stmts=(written,),
        post=None,
    )
    x = np.ones(4)
    y = np.zeros(4)
    out = run(term, x=x, y=y)
    assert np.allclose(out["y"], 1.0)


def test_lower_generic_records_the_statement_to_instruction_map() -> None:
    lowering = lower_generic(ht.spmv_term())
    assert lowering.insn_ids == {"S0": "S0"}
    assert lowering.outputs == ("y",)
    # Ragged arrays are flat, and the offsets argument is the one the term has.
    assert lowering.ragged == {"col": "off", "val": "off"}


def test_the_offsets_argument_is_invented_when_the_term_has_none() -> None:
    term = ht.spmv_term()
    without = type(term)(
        name="spmv_no_offsets",
        params=tuple(p for p in term.params if p[0] != "off"),
        sizes=term.sizes,
        stmts=term.stmts,
        post=None,
    )
    lowering = lower_generic(without)
    assert lowering.ragged == {"col": "off_cnt", "val": "off_cnt"}
    names = [arg.name for arg in lowering.kernel.default_entrypoint.args]
    assert "off_cnt" in names
    assert isinstance(
        lowering.kernel.default_entrypoint.arg_dict["off_cnt"], lp.ArrayArg
    )


def test_count_parameters_are_recognized_in_both_spellings() -> None:
    assert count_param_names("cnt", "r") == ("cnt_r", "nl_cnt_r")


def test_dtypes_come_from_the_sorts() -> None:
    from lanky.prelude import Fin, Int, Nat, Real

    assert numpy_dtype(Real) == np.dtype(np.float64)
    assert numpy_dtype(Nat) == np.dtype(np.int32)
    assert numpy_dtype(Int) == np.dtype(np.int32)
    assert numpy_dtype(Fin[8]) == np.dtype(np.int32)
    with pytest.raises(LoweringError):
        numpy_dtype(object())


def test_the_opencl_target_is_named_but_not_imported() -> None:
    import sys

    assert isinstance(target_for("c"), lp.ExecutableCTarget)
    with pytest.raises(LoweringError):
        target_for("cuda")
    # Importing loopty, lowering, and running must never pull in pyopencl.
    assert "pyopencl" not in sys.modules
