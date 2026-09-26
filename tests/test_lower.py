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


def test_statements_that_feed_each_other_across_the_loop_are_not_a_cycle() -> None:
    # S0 reads what S1 wrote one iteration earlier. That order is the loop's;
    # loopy's single-writer heuristic used to state it as an instruction
    # dependence of S0 on S1, against S1's own on S0, and refused the cycle.
    term = ht.coupled_pair_term()
    insns = {insn.id: insn for insn in lower(term).default_entrypoint.instructions}
    assert insns["S0"].depends_on == frozenset()
    assert insns["S1"].depends_on == frozenset({"S0"})

    x = np.zeros(8)
    x[0] = 1.0
    v = np.zeros(8)
    out = run(term, x=x.copy(), v=v.copy())
    want_x, want_v = ht.coupled_pair_reference(x, v)
    assert np.allclose(out["x"], want_x)
    assert np.allclose(out["v"], want_v)


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


@pytest.mark.parametrize("order", ["ps", "sp"])
def test_a_ragged_bound_is_ordered_against_the_offsets_it_reads(order: str) -> None:
    # The row length cnt_r is computed from off, which the other statement
    # rewrites. Its instruction used to be left to loopy's single-writer
    # heuristic, which made it wait for the shift wherever the shift was. With
    # the product first, the shift waits for the product, which reads through
    # off, the product waits for cnt_r, and cnt_r waited for the shift: a cycle.
    shift_first = order == "sp"
    term = ht.spmv_and_shift_term(order)
    insns = {insn.id: insn for insn in lower(term).default_entrypoint.instructions}
    shift, product = ("S0", "S1") if shift_first else ("S1", "S0")
    before = {shift} if shift_first else set()
    assert insns["cnt_r_init"].depends_on == frozenset(before)
    assert "cnt_r_init" in insns[product].depends_on

    # Row 0 starts at 1; the entry at 0 is read only through shifted offsets.
    off = np.array([1, 3, 3, 6], dtype=np.int32)
    col = np.array([0, 0, 1, 0, 2, 3], dtype=np.int32)
    val = np.array([7.0, 1.0, 2.0, 3.0, 4.0, 5.0])
    x = np.array([1.0, 10.0, 100.0, 1000.0])
    out = run(term, off=off.copy(), col=col, val=val, x=x, y=np.zeros(3))
    seen = off - 1 if shift_first else off
    assert np.allclose(out["y"], ht.csr_reference(seen, col, val, x))
    assert np.array_equal(out["off"], off - 1)


@pytest.mark.parametrize("order", ["psp", "prq"])
def test_a_ragged_bound_is_not_reused_after_its_offsets_are_rewritten(
    order: str,
) -> None:
    # cnt_r is computed once, where the first product needs it. The second
    # product would read through the shifted offsets with the old row lengths.
    # With the shift in a loop of its own, loopy could not schedule the two
    # products around it. With the shift inside the row loop and the second
    # product over an inner loop of its own, the kernel ran and misread rows.
    with pytest.raises(LoweringError, match=r"S2 is bounded by the row length cnt_r"):
        lower(ht.spmv_and_shift_term(order))


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


def test_two_statements_over_one_iname_keep_their_own_domains() -> None:
    # loopy gives an iname one domain, so the two statements share the union of
    # theirs. Sharing it silently ran the narrower statement over the wider
    # domain: every cell was scaled, including the one the term leaves alone.
    term = ht.narrowed_second_statement_term()
    code = code_for(term)
    assert "if (" in code
    a = np.arange(1.0, 9.0)
    out = run(term, a=a, b=np.zeros(8))
    assert np.array_equal(out["b"], ht.narrowed_reference(a))


def test_a_statement_that_was_not_widened_gets_no_predicate() -> None:
    # The predicate is the gist of the statement's own domain against the
    # merged one, so an unwidened statement is generated exactly as before.
    assert "if (" not in code_for(ht.axpy_term())


def test_a_statement_with_an_empty_domain_never_runs() -> None:
    # A domain that is affinely contradictory has no instances at all. The
    # instruction still exists, because loopy builds the loop from the merged
    # domain, so it gets a condition nothing satisfies.
    import dataclasses

    import islpy as isl

    term = ht.narrowed_second_statement_term()
    nowhere = dataclasses.replace(
        term.stmts[1], domain=isl.Set("[n] -> { [i] : 0 <= i < n and i < 0 }")
    )
    term = dataclasses.replace(term, stmts=(term.stmts[0], nowhere))
    a = np.arange(1.0, 5.0)
    out = run(term, a=a, b=np.zeros(4))
    assert np.array_equal(out["b"], a)


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


# {{{ the loops outside an inner loop


def test_the_outer_part_of_a_domain_keeps_nothing_of_the_inner_loop() -> None:
    # Projected, ``0 <= j < m`` would leave ``m >= 1`` on the loop over ``r``,
    # and two inner loops side by side over ``m`` and ``p`` would give it a
    # union that is not convex (see the traced test of two inner loops).
    import islpy as isl

    from loopty.lower import _outer_part

    dense = isl.Set("[n, m] -> { [r, j] : m >= 0 and 0 <= r < n and 0 <= j < m }")
    rows = isl.Set("[n, m] -> { [r] : 0 <= r < n }")
    assert _outer_part(dense, 1).is_equal(rows)
    assert not dense.project_out(isl.dim_type.set, 1, 1).is_equal(rows)
    # Nothing to drop when every loop is kept.
    assert _outer_part(dense, 2) is dense


def test_a_loop_bounded_only_through_an_inner_one_keeps_its_projection() -> None:
    # Dropping ``r <= j < n`` would leave ``r`` unbounded, which no loop is.
    import islpy as isl

    from loopty.lower import _outer_part

    through = isl.Set("[n] -> { [r, j] : 0 <= r <= j < n }")
    assert _outer_part(through, 1).is_equal(isl.Set("[n] -> { [r] : 0 <= r < n }"))


# }}}


# {{{ temporaries, and offsets a term states


def _through_temporary(temporaries, params=None):
    """``t[i] = x[i]`` and then, in a loop of its own, ``y[j] = t[j]``."""
    import islpy as isl
    import pymbolic.primitives as prim

    from loopty.term import Access, ArrType, Stmt, Term

    V = prim.Variable
    vector = ArrType(axes=(V("n"),), dtype=np.dtype(np.float64), ragged=(False,))
    stmts = tuple(
        Stmt(
            id=f"S{k}",
            inames=(iname,),
            domain=isl.Set(f"[n] -> {{ [{iname}] : 0 <= {iname} < n }}"),
            assignee=Access(target, (V(iname),)),
            expr=prim.Subscript(V(source), (V(iname),)),
            kind="assign",
            guard=None,
            where=f"hand.py:{k + 1}",
            order=(k, 0),
        )
        for k, (source, target, iname) in enumerate(
            (("x", "t", "i"), ("t", "y", "j"))
        )
    )
    return Term(
        name="through",
        params=params or (("x", vector), ("y", vector)),
        sizes=("n",),
        stmts=stmts,
        post=None,
        temporaries=temporaries,
    )


def _vector(size="n"):
    import pymbolic.primitives as prim

    from loopty.term import ArrType

    return ArrType(
        axes=(prim.Variable(size),), dtype=np.dtype(np.float64), ragged=(False,)
    )


def test_a_temporary_is_declared_in_the_kernel_and_passed_by_nobody() -> None:
    term = _through_temporary((("t", _vector()),))
    lowering = lower_generic(term)
    entry = lowering.kernel.default_entrypoint
    assert lowering.temporaries == ("t",)
    assert lowering.array_args == ("x", "y")
    assert lowering.outputs == ("y",)
    assert "t" not in {arg.name for arg in entry.args}
    assert entry.temporary_variables["t"].address_space == lp.AddressSpace.PRIVATE
    # A Real temporary counts in the contraction pin as a Real argument does.
    assert lowering.contraction
    x = np.arange(5.0)
    y = np.zeros(5)
    run(term, x=x, y=y)
    assert list(y) == list(x)


def test_a_temporary_lives_in_global_memory_on_opencl() -> None:
    # Built without lowering for the device, which would import pyopencl: the
    # C target's host code never allocates a global temporary, and the
    # PyOpenCL host code does (docs/loopy-notes.md, note 14).
    from loopty.lower import _temporary

    term = _through_temporary((("t", _vector()),))
    temporary = _temporary(term, "t", _vector(), "opencl", {"n"})
    assert temporary.address_space == lp.AddressSpace.GLOBAL
    assert _temporary(term, "t", _vector(), "c", {"n"}).address_space == (
        lp.AddressSpace.PRIVATE
    )


def test_a_ragged_temporary_is_refused() -> None:
    import pymbolic.primitives as prim

    from loopty.lower import _temporary
    from loopty.term import ArrType

    ragged = ArrType(
        axes=(prim.Variable("n"), prim.Variable("x")),
        dtype=np.dtype(np.float64),
        ragged=(False, True),
    )
    term = _through_temporary(())
    with pytest.raises(LoweringError, match="no offsets"):
        _temporary(term, "t", ragged, "c", {"n", "x"})


def test_a_temporary_sized_by_nothing_else_is_refused() -> None:
    term = _through_temporary((("t", _vector("p")),))
    with pytest.raises(LoweringError, match="sized by p"):
        lower_generic(term)


def test_an_exact_temporary_pins_contraction_off() -> None:
    from lanky.prelude import Real

    from loopty.term import ArrType
    from loopty.tolerance import output_class

    exact = ArrType(axes=_vector().axes, dtype=Real.exact, ragged=(False,))
    term = _through_temporary((("t", exact),))
    assert output_class(term, "t") == "exact"
    assert not lower_generic(term).contraction


def test_offsets_stated_as_none_are_an_argument_of_a_fresh_name() -> None:
    import islpy as isl
    import pymbolic.primitives as prim

    from loopty.term import Access, ArrType, Stmt, Term

    V = prim.Variable
    real = np.dtype(np.float64)
    counts = ArrType(axes=(V("n"),), dtype=np.dtype(np.int64), ragged=(False,))
    ragged = ArrType(axes=(V("n"), V("cnt")), dtype=real, ragged=(False, True))
    term = Term(
        name="rowsum",
        params=(("cnt", counts), ("val", ragged), ("off_cnt", _vector())),
        sizes=("n",),
        stmts=(
            Stmt(
                id="S0",
                inames=("r",),
                domain=isl.Set("[n] -> { [r] : 0 <= r < n }"),
                assignee=Access("off_cnt", (V("r"),)),
                expr=prim.Subscript(V("val"), (V("r"), 0))
                + prim.Subscript(V("cnt"), (V("r"),)),
                kind="assign",
                guard=None,
                where="hand.py:1",
            ),
        ),
        post=None,
        offsets=(("cnt", None),),
    )
    # Left to the names, ``off_cnt`` would be the offsets; stated as None, it
    # is an ordinary array, and the added offsets argument avoids its name.
    assert term.offsets_of("cnt") is None
    assert lower_generic(term).ragged == {"val": "off_cnt_"}


def test_a_kernels_term_still_finds_its_offsets_by_name() -> None:
    from loopty.term import Term

    counts = _vector()
    term = Term(
        name="k",
        params=(("cnt", counts), ("cnt_off", counts)),
        sizes=("n",),
        stmts=(),
        post=None,
    )
    assert term.offsets_of("cnt") == "cnt_off"
    assert term.offsets_of("other") is None
    stated = Term(
        name="k",
        params=term.params,
        sizes=term.sizes,
        stmts=(),
        post=None,
        offsets=(("cnt", "elsewhere"),),
    )
    assert stated.offsets_of("cnt") == "elsewhere"


# }}}
