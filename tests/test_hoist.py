"""A sum's operands that read none of its binders, computed once before it (#270).

``x = (t[i] - centre) / scale`` is a Python local, which the trace reads into
the summand, so the sum computed it at every term, where a kernel author
writes ``<> x = ...`` before the sum in loopy. The code generated now computes
it once per point, in a temporary of the work item, and computes the same
bits: the temporary holds what the operand held in the sum, which adds the
same terms in the same order. The tests read the code, run it with and
without the temporaries, and check where an operand must stay.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import numpy as np
import pytest
from lanky.prelude import Nat, Real

from loopty import Arr, Fin, Schedule, kernel, reduce_sum
from loopty.executor import LoopyExecutor, emit_code
from loopty.isl_reading import declining
from loopty.math import ipow

lp = pytest.importorskip("loopy")


# {{{ kernels


@kernel
def taylor(
    c: Arr[Fin[m], Real],  # noqa: F821
    t: Arr[Fin[n], Real],  # noqa: F821
    p: Arr[Fin[n], Real],  # noqa: F821
    centre: Real,
    scale: Real,
):
    """A Taylor polynomial at each point, its scaled offset a Python local."""
    for i in p.dom:
        x = (t[i] - centre) / scale
        p[i] = reduce_sum(c[k] * ipow(x, k) for k in c.dom)


@kernel
def squares(
    c: Arr[Fin[m], Real.exact],  # noqa: F821
    t: Arr[Fin[n], Real.exact],  # noqa: F821
    p: Arr[Fin[n], Real.exact],  # noqa: F821
    centre: Real.exact,
    scale: Real.exact,
):
    """The same offset, multiplied rather than divided, so that it is ``exact``."""
    for i in p.dom:
        x = (t[i] - centre) * scale
        p[i] = reduce_sum(c[k] * x * x for k in c.dom)


@kernel
def rows(
    cnt: Arr[Fin[n], Nat],  # noqa: F821
    val: Arr[Fin[n], Fin[cnt], Real],  # noqa: F821
    s: Arr[Fin[n], Real],  # noqa: F821
    y: Arr[Fin[n], Real],  # noqa: F821
    h: Real,
):
    """Rows that may be empty, scaled by a value of the row and by a scalar."""
    for r in y.dom:
        y[r] = reduce_sum(val[r, j] * (s[r] * 2.0) * (h * 3.0) for j in val.dom[r])


@kernel
def before(
    x: Arr[Fin[n], Real],  # noqa: F821
    a: Arr[Fin[n], Real],  # noqa: F821
    y: Arr[Fin[n], Real],  # noqa: F821
    h: Real,
):
    """A sum with no term at ``i = 0``, where ``x[i - 1]`` is no cell."""
    for i in y.dom:
        y[i] = reduce_sum(a[j] * (x[i - 1] * 2.0) * (h * 3.0) for j in Fin[i])


@kernel
def mixed(
    a: Arr[Fin[m], Real],  # noqa: F821
    x: Arr[Fin[n], np.float32],  # noqa: F821
    y: Arr[Fin[n], Real],  # noqa: F821
):
    """A single-precision operand of a double sum."""
    for i in y.dom:
        y[i] = reduce_sum(a[j] * (x[i] * x[i]) for j in a.dom)


@kernel
def leading(
    w: Arr[Fin[n], Real],  # noqa: F821
    f: Arr[Fin[m], Real],  # noqa: F821
    y: Arr[Fin[n], Real],  # noqa: F821
    z: Arr[Fin[n], Real],  # noqa: F821
    h: Real,
):
    """The same product, its binder-free factors first and then last."""
    for i in y.dom:
        y[i] = reduce_sum(h * w[i] * f[j] for j in f.dom)
        z[i] = reduce_sum(f[j] * h * w[i] for j in f.dom)


@kernel
def guarded(
    x: Arr[Fin[n], Real],  # noqa: F821
    a: Arr[Fin[m], Real],  # noqa: F821
    y: Arr[Fin[n], Real],  # noqa: F821
):
    """A sum under a guard that reads an array."""
    for i in y.dom:
        if x[i] > 0:
            y[i] = reduce_sum(a[j] * (x[i] * 2.0) for j in a.dom)


@kernel
def spread(
    a: Arr[Fin[32], Real],
    x: Arr[Fin[n], Real],  # noqa: F821
    y: Arr[Fin[n], Real],  # noqa: F821
):
    """A sum a group's work items can share, one term each."""
    for r in y.dom:
        y[r] = reduce_sum(a[j] * (x[r] * 2.0) for j in a.dom)


@kernel
def lanes(
    a: Arr[Fin[8], Fin[8], Real],
    x: Arr[Fin[8], Real],
    y: Arr[Fin[8], Real],
):
    """Row sums scaled by a value of the row, a few rows to a work item."""
    for i in y.dom:
        y[i] = reduce_sum(a[i, j] * (x[i] * 2.0) for j in a.dom[i])


@kernel
def nested(
    a: Arr[Fin[m], Real],  # noqa: F821
    b: Arr[Fin[q], Real],  # noqa: F821
    x: Arr[Fin[n], Real],  # noqa: F821
    y: Arr[Fin[n], Real],  # noqa: F821
):
    """A sum in a sum: one operand reads neither binder, one the outer one."""
    for i in y.dom:
        y[i] = reduce_sum(
            a[j] * reduce_sum(b[k] * (x[i] * 3.0) * (a[j] * 2.0) for k in b.dom)
            for j in a.dom
        )


# }}}


ROOT = Path(__file__).resolve().parent.parent


def load(path: Path):
    """Import an example as a module of its own."""
    name = f"hoist_{path.parent.name}_{path.stem}"
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    sys.path.insert(0, str(path.parent))
    try:
        spec.loader.exec_module(module)
    finally:
        sys.path.remove(str(path.parent))
    return module


def body(code: str, name: str) -> str:
    """The kernel function of ``code``, without the helpers before it."""
    return code[code.index(f"void {name}(") :]


def agrees(kern, schedule, arrays) -> None:
    """The scheduled run agrees with the native one, as its types state."""
    fact = LoopyExecutor().differential(kern, schedule, arrays)
    assert fact.status.value == "tested", fact.provenance


# {{{ what moves


def test_a_local_the_sum_reads_is_computed_once_per_point() -> None:
    code = body(emit_code(taylor), "taylor")
    once = "invariant_0 = (t[i] + -1.0 * centre) / scale;"
    assert once in code
    assert code.index(once) < code.index("for (int32_t k = 0;")
    assert "acc_k = acc_k + c[k] * loopy_pow_float64_int32(invariant_0, k);" in code
    assert "double invariant_0;" in code


def test_the_compiled_sum_adds_the_same_bits_as_before_and_as_numpy() -> None:
    from loopty.hoist import hoist_invariants
    from loopty.lower import lower_generic

    # x is read twice in each term, and one temporary holds it for both
    code = body(emit_code(squares), "squares")
    assert "invariant_0 = (t[i] + -1.0 * centre) * scale;" in code
    assert "acc_k = acc_k + c[k] * invariant_0 * invariant_0;" in code
    assert "#pragma STDC FP_CONTRACT OFF" in emit_code(squares)

    rng = np.random.default_rng(0)
    for kern in (taylor, squares):
        arrays = {
            "c": rng.standard_normal(9),
            "t": rng.standard_normal(40),
            "p": np.zeros(40),
            "centre": 0.1,
            "scale": 0.3,
        }
        inline = lower_generic(kern.trace(), "c").kernel
        moved = hoist_invariants(inline)
        assert "invariant_0" in moved.default_entrypoint.temporary_variables
        assert "invariant_0" not in inline.default_entrypoint.temporary_variables
        with declining():
            _event, (before,) = inline.executor()(**arrays)
            _event, (after,) = moved.executor()(**arrays)
        assert np.array_equal(after, before), kern
        agrees(kern, Schedule(kern), arrays)

    # an exact output is compared with numpy's bit for bit
    native = {**arrays, "p": np.zeros(40)}
    squares(**native)
    assert np.array_equal(after, native["p"])
    fact = LoopyExecutor().differential(squares, Schedule(squares), arrays)
    assert fact.provenance["outputs"]["p"]["exactness"] == "exact"


def test_the_leading_operands_of_a_product_move_and_trailing_ones_stay() -> None:
    # C multiplies h * w[i] * f[j] as (h * w[i]) * f[j], so the first product
    # is an operand of the sum; f[j] * h * w[i] has none that reads no binder.
    code = body(emit_code(leading), "leading")
    assert "invariant_0 = h * w[i];" in code
    assert "acc_j = acc_j + invariant_0 * f[j];" in code
    assert "acc_j_0 = acc_j_0 + f[j_0] * h * w[i];" in code
    assert "invariant_1" not in code
    rng = np.random.default_rng(1)
    agrees(
        leading,
        Schedule(leading),
        {
            "w": rng.standard_normal(5),
            "f": rng.standard_normal(7),
            "y": np.zeros(5),
            "z": np.zeros(5),
            "h": 0.7,
        },
    )


def test_an_operand_moves_out_of_a_row_that_may_be_empty_where_its_cells_are() -> None:
    # s[r] is a cell of s for every row, empty or not, as loopy checks the
    # instruction's reads; h is a scalar.
    code = body(emit_code(rows), "rows")
    assert "invariant_0 = s[r] * 2.0;" in code
    assert "invariant_1 = h * 3.0;" in code
    assert "val[off_cnt[r] + j] * invariant_0 * invariant_1;" in code
    counts = [0, 3, 0, 1, 2]
    agrees(
        rows,
        Schedule(rows),
        {
            "cnt": Arr.from_numpy(np.array(counts, dtype=np.int64)),
            "val": Arr.ragged(counts, values=[1.0, 2.0, 4.0, 8.0, 16.0, 32.0]),
            "s": np.array([0.5, 1.5, 2.5, 3.5, 4.5]),
            "y": np.zeros(5),
            "h": 0.25,
        },
    )


def test_an_operand_on_a_device_is_each_work_items_own(plain_opencl) -> None:
    schedule = Schedule(taylor, target="opencl").on_groups(taylor.i)
    with declining():
        code = lp.generate_code_v2(schedule.kernel).device_code()
    kernel_code = code[code.index("__kernel") :]
    assert "double invariant_0;" in kernel_code
    assert "invariant_0 = (t[gid(0)] + -1.0 * centre) / scale;" in kernel_code
    assert "loopy_pow_float64_int32(invariant_0, k)" in kernel_code


def test_an_operand_in_an_ilp_loop_is_each_lanes_own() -> None:
    # loopy gives a temporary written in an ilp loop one copy per iteration,
    # as it does the sum's accumulator, so each lane reads its own row's.
    schedule = Schedule(lanes).split(lanes.i, 4).tag(lanes.i.inner, "ilp")
    assert schedule.buildable == (True, "")
    code = body(emit_code(schedule), "lanes")
    assert "double invariant_0[4];" in code
    for lane in range(4):
        offset = f"{lane} + " if lane else ""
        assert f"invariant_0[{lane}] = x[{offset}4 * i_outer] * 2.0;" in code
        assert f"* invariant_0[{lane}];" in code
    rng = np.random.default_rng(3)
    arrays = {
        "a": rng.standard_normal((8, 8)),
        "x": rng.standard_normal(8),
        "y": np.zeros(8),
    }
    with declining():
        _event, (before,) = schedule._code.executor()(**arrays)
        _event, (after,) = schedule.kernel.executor()(**arrays)
    assert np.array_equal(after, before)
    agrees(lanes, schedule, arrays)


def test_the_l2p_demo_computes_x_and_y_once_per_target() -> None:
    l2p = load(ROOT / "examples" / "l2p.py")
    code = body(emit_code(l2p.contracted), "l2p")
    x = "invariant_0 = (tx[off_cnt[box] + target] + -1.0 * cx[box]) / rscale;"
    y = "invariant_1 = (ty[off_cnt[box] + target] + -1.0 * cy[box]) / rscale;"
    assert x in code and y in code
    term = next(line for line in code.splitlines() if "acc_a_1_b_1 + " in line)
    assert "loopy_pow_float64_int32(invariant_0, a_1)" in term
    assert "loopy_pow_float64_int32(invariant_1, b_1)" in term
    assert "rscale" not in term


# }}}


# {{{ what stays


def test_an_operand_whose_cell_only_a_term_reads_stays_in_the_sum() -> None:
    # x[i - 1] is read where the sum has a term, i >= 1; at i = 0 it is no
    # cell, and the sum has none. h * 3.0 reads no array, and moves.
    code = body(emit_code(before), "before")
    assert "invariant_0 = h * 3.0;" in code
    assert "acc_j = acc_j + a[j] * (x[-1 + i] * 2.0) * invariant_0;" in code
    assert "invariant_1" not in code
    agrees(
        before,
        Schedule(before),
        {"x": np.arange(1.0, 5.0), "a": np.ones(4), "y": np.zeros(4), "h": 0.5},
    )


def test_a_single_precision_operand_of_a_double_sum_stays() -> None:
    # C computes x[i] * x[i] in float either way, but the sum writes its
    # literals in double, so only an operand of the sum's own precision moves.
    code = body(emit_code(mixed), "mixed")
    assert "invariant" not in code
    assert "acc_j = acc_j + a[j] * (x[i] * x[i]);" in code


def test_a_sum_on_a_local_axis_keeps_its_operands(plain_opencl) -> None:
    # loopy runs the statement on l.0 with its sum, and refuses an
    # instruction that runs on no local axis in a kernel that uses one.
    schedule = Schedule(spread, target="opencl").tag({spread.j: "l.0", spread.r: "g.0"})
    assert schedule.buildable == (True, "")
    assert "invariant_0" not in schedule.kernel.default_entrypoint.temporary_variables
    with declining():
        code = lp.generate_code_v2(schedule.kernel).device_code()
    assert "x[gid(0)] * 2.0" in code


def test_an_operand_under_a_guard_is_computed_under_it() -> None:
    entry = Schedule(guarded).kernel.default_entrypoint
    insns = {insn.id: insn for insn in entry.instructions}
    assert set(insns) == {"S0", "S0_invariant_0"}
    assert insns["S0_invariant_0"].predicates == insns["S0"].predicates
    assert insns["S0_invariant_0"].predicates
    assert "S0_invariant_0" in insns["S0"].depends_on
    agrees(
        guarded,
        Schedule(guarded),
        {
            "x": np.array([1.0, -2.0, 3.0, 0.0]),
            "a": np.array([0.5, 0.25]),
            "y": np.zeros(4),
        },
    )


def test_an_operand_of_an_inner_sum_moves_only_if_it_reads_no_outer_binder() -> None:
    code = body(emit_code(nested), "nested")
    assert "invariant_0 = x[i] * 3.0;" in code
    assert code.index("invariant_0 = x[i] * 3.0;") < code.index("for (int32_t j")
    assert "acc_k = acc_k + b[k] * invariant_0 * (a[j] * 2.0);" in code
    rng = np.random.default_rng(2)
    agrees(
        nested,
        Schedule(nested),
        {
            "a": rng.standard_normal(3),
            "b": rng.standard_normal(4),
            "x": rng.standard_normal(5),
            "y": np.zeros(5),
        },
    )


# }}}


def test_the_steps_transform_and_check_the_kernel_as_lowered() -> None:
    # The temporaries are in the code a schedule generates, after its steps,
    # and nowhere a step or a fact looks.
    schedule = Schedule(taylor).split(taylor.i, 4)
    assert "invariant_0" not in schedule._kernel.default_entrypoint.temporary_variables
    entry = schedule.kernel.default_entrypoint
    assert "invariant_0" in entry.temporary_variables
    hoisted = entry.id_to_insn["S0_invariant_0"]
    assert hoisted.within_inames == entry.id_to_insn["S0"].within_inames
    assert hoisted.within_inames == {"i_outer", "i_inner"}
    assert not any("invariant" in fact.statement for fact in schedule.facts())
    assert schedule.kernel is schedule.kernel
