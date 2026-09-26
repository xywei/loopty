"""Running a term, and saying how well two runs agree.

The differential test is the executor's reason for existing: a kernel has two
implementations, the Python body and the compiled code, and the claim that they
agree is only meaningful with a tolerance attached. The tolerance is not a
parameter of the comparison; it is read off the exactness class of the output,
widened by any accumulation the schedule reassociated.
"""

from __future__ import annotations

import sys

import numpy as np
import pytest
from lanky.prelude import Nat, Real

import hand_terms as ht
from loopty import Arr, Fin, kernel, when
from loopty import sum as reduce_sum
from loopty.executor import (
    TOLERANCE,
    TOLERANCE_FLOOR,
    LoopyExecutor,
    agreement,
    emit_code,
    exactness_of_output,
)
from loopty.schedule import Schedule

pytest.importorskip("loopy")


def executor() -> LoopyExecutor:
    return LoopyExecutor()


def axpy_reference(a, x, y, z):
    """The Python body of the axpy term, as a kernel would have written it."""
    z[...] = a * x + y


def test_importing_and_running_never_imports_pyopencl() -> None:
    # The opencl target belongs on a machine with a device. Nothing on the
    # default path may import pyopencl, so that loopty installs and runs without
    # it.
    x = np.arange(4, dtype=np.float64)
    executor().run(ht.axpy_term(), a=1.0, x=x, y=np.zeros(4), z=np.zeros(4))
    assert "pyopencl" not in sys.modules


def test_the_executor_names_itself_and_its_trust_class() -> None:
    assert executor().name == "loopy"
    assert executor().trust_class() == "test"


def test_positional_arguments_follow_the_term_signature() -> None:
    x = np.arange(4, dtype=np.float64)
    y = np.ones(4)
    z = np.zeros(4)
    out = executor().run(ht.axpy_term(), 3.0, x, y, z)
    assert np.allclose(out["z"], 3.0 * x + y)


def test_a_strided_or_differently_typed_output_is_updated_in_place() -> None:
    # ``_as_numpy`` copies an output that is not contiguous, or not of the
    # lowered dtype, on the way into loopy. The results have to come back out
    # of that copy, or ``run`` returns the right values and leaves the caller's
    # array as it was.
    x = np.arange(4, dtype=np.float64)
    y = np.ones(4)
    storage = np.zeros((4, 2))
    z = storage[:, 0]
    out = executor().run(ht.axpy_term(), a=3.0, x=x, y=y, z=z)
    assert np.allclose(out["z"], 3.0 * x + y)
    assert np.allclose(z, 3.0 * x + y)
    assert np.all(storage[:, 1] == 0.0)

    single = np.zeros(4, dtype=np.float32)
    executor().run(ht.axpy_term(), a=3.0, x=x, y=y, z=single)
    assert single.dtype == np.float32
    assert np.allclose(single, 3.0 * x + y)


def test_a_ragged_array_supplies_its_own_offsets() -> None:
    from loopty.arr import Arr

    counts = [2, 0, 3]
    col = Arr.ragged(counts, values=[0, 1, 0, 2, 3], dtype=np.int64)
    val = Arr.ragged(counts, values=[1.0, 2.0, 3.0, 4.0, 5.0])
    x = np.array([1.0, 10.0, 100.0, 1000.0])
    y = np.zeros(3)
    off = np.array([0, 2, 2, 5], dtype=np.int32)
    out = executor().run(ht.spmv_term(), off=off, col=col, val=val, x=x, y=y)
    want = ht.csr_reference(off, col.numpy(), val.numpy(), x)
    assert np.allclose(out["y"], want)


def test_the_differential_fact_records_the_tolerance_it_judged_by() -> None:
    schedule = Schedule(ht.axpy_term())
    arrays = {
        "a": 2.0,
        "x": np.arange(4, dtype=np.float64),
        "y": np.ones(4),
        "z": np.zeros(4),
    }
    fact = executor().differential(axpy_reference, schedule, arrays)
    assert fact.status.value == "tested"
    assert fact.kind == "agreement"
    detail = fact.provenance["outputs"]["z"]
    assert detail["agree"]
    assert detail["difference"] == 0.0
    assert detail["exactness"] == "approx"
    assert fact.provenance["target"] == "c"


def test_a_disagreement_is_refuted_rather_than_raised() -> None:
    def wrong(a, x, y, z):
        z[...] = a * x + y + 1.0

    schedule = Schedule(ht.axpy_term())
    arrays = {
        "a": 2.0,
        "x": np.arange(4, dtype=np.float64),
        "y": np.ones(4),
        "z": np.zeros(4),
    }
    fact = executor().differential(wrong, schedule, arrays)
    assert fact.status.value == "refuted"
    assert not fact.provenance["outputs"]["z"]["agree"]


@kernel
def flipped(x: Arr[Fin[n], Real], y: Arr[Fin[n], Real]):  # noqa: F821
    """``~`` on a Python bool is bitwise, so natively this guard is an integer."""
    for i in y.dom:
        with when(~(i > 0)):
            y[i] = x[i]


@pytest.mark.filterwarnings("ignore:Bitwise inversion:DeprecationWarning")
def test_a_native_refusal_is_a_refuted_agreement_rather_than_raised() -> None:
    # The native TraceError came out of ``differential``, so the run recorded
    # nothing, while the faithfulness fact counts the same refusal as a
    # disagreement (#44). Python warns about ~ on a bool, which this suite
    # makes an error, and a user's run does not.
    arrays = {"x": Arr.from_numpy(np.arange(4.0)), "y": Arr.zeros(4)}
    schedule = Schedule(flipped)
    fact = executor().differential(flipped, schedule, arrays)
    assert fact.kind == "agreement"
    assert fact.status.value == "refuted"
    assert fact.id == f"agreement:{schedule.key}"
    assert fact.provenance["error"].startswith("TraceError: the guard of")
    reason = fact.provenance["reason"]
    assert reason.startswith("the body, run natively, is refused")
    assert "'i <= 0' for '~(i > 0)'" in reason
    assert fact.provenance["outputs"] == {}
    # The caller's arrays are untouched: neither run wrote into them.
    assert np.array_equal(arrays["y"].numpy(), np.zeros(4))


def test_an_agreement_fact_is_named_after_the_schedule_that_ran() -> None:
    arrays = {
        "a": 2.0,
        "x": np.arange(4, dtype=np.float64),
        "y": np.ones(4),
        "z": np.zeros(4),
    }
    split = Schedule(ht.axpy_term()).split("i", 2)
    fact = executor().differential(axpy_reference, split, arrays)
    assert fact.id == f"agreement:{split.key}"
    assert fact.id != executor().differential(
        axpy_reference, Schedule(ht.axpy_term()), arrays
    ).id


def test_the_exactness_class_of_an_output_is_the_weakest_of_three() -> None:
    term = ht.spmv_term(exactness="exact")
    plain = Schedule(term)
    assert exactness_of_output(term, plain, "y") == "approx"

    integer = ht.spmv_term(exactness="exact")
    integer = type(integer)(
        name=integer.name,
        params=tuple(
            (name, typ if name != "y" else ht.dense(typ.axes[0], dtype=ht.INT))
            for name, typ in integer.params
        ),
        sizes=integer.sizes,
        stmts=integer.stmts,
        post=None,
    )
    assert exactness_of_output(integer, Schedule(integer), "y") == "exact"

    reassociated = Schedule(ht.spmv_term(exactness="reassoc")).realize("y", tree=True)
    assert exactness_of_output(integer, reassociated, "y") == "reassoc"


def test_tolerances_are_stated_once_and_ordered() -> None:
    assert TOLERANCE["exact"] == 0.0
    assert TOLERANCE["exact"] < TOLERANCE["reassoc"] < TOLERANCE["approx"]


def test_one_wrong_element_of_a_large_output_is_a_refutation() -> None:
    # The tolerance used to be scaled by the 1-norm of the whole expected
    # output, so a big output bought a big allowance for every one of its
    # cells: 2000 entries of 1000.0 gave a tolerance of 2.0, and an error of
    # 0.5 in one cell "agreed". The scale is now that cell's own magnitude.
    term = ht.axpy_term()
    want = np.full(2000, 1000.0)
    got = want.copy()
    got[7] += 0.5
    fact = agreement(term, Schedule(term), {"z": got}, {"z": want})
    assert fact.status.value == "refuted"
    detail = fact.provenance["outputs"]["z"]
    assert detail["difference"] == pytest.approx(0.5)
    assert detail["tolerance"] == pytest.approx(
        TOLERANCE["approx"] * (1000.0 + TOLERANCE_FLOOR)
    )


def test_the_tolerance_of_an_output_does_not_grow_with_its_size() -> None:
    term = ht.axpy_term()

    def tolerance_for(size: int) -> float:
        want = np.full(size, 3.0)
        fact = agreement(term, Schedule(term), {"z": want.copy()}, {"z": want})
        return fact.provenance["outputs"]["z"]["tolerance"]

    assert tolerance_for(4) == pytest.approx(tolerance_for(4000))
    assert tolerance_for(4) == pytest.approx(
        TOLERANCE["approx"] * (3.0 + TOLERANCE_FLOOR)
    )


def test_an_element_sized_error_is_still_forgiven_at_its_own_scale() -> None:
    # The relative half of the formula: a rounding-sized difference on a large
    # value agrees, which is what a reassociated sum needs.
    term = ht.axpy_term()
    want = np.full(16, 1e6)
    got = want.copy()
    got[3] += 0.5
    fact = agreement(term, Schedule(term), {"z": got}, {"z": want})
    assert fact.status.value == "tested"


def test_a_reference_has_to_cover_every_output() -> None:
    term = ht.two_output_term()
    schedule = Schedule(term)
    arrays = {
        "x": np.arange(4, dtype=np.float64),
        "y": np.zeros(4),
        "z": np.zeros(4),
    }
    with pytest.raises(ValueError, match="does not cover z"):
        executor().differential(None, schedule, arrays, reference={"y": np.zeros(4)})
    with pytest.raises(ValueError, match="not an output"):
        executor().differential(
            None,
            schedule,
            arrays,
            reference={"y": np.zeros(4), "z": np.zeros(4), "x": np.zeros(4)},
        )


def test_a_reference_covering_every_output_is_accepted() -> None:
    term = ht.two_output_term()
    x = np.arange(4, dtype=np.float64)
    fact = executor().differential(
        None,
        Schedule(term),
        {"x": x, "y": np.zeros(4), "z": np.zeros(4)},
        reference={"y": 2 * x, "z": x + 1},
    )
    assert fact.status.value == "tested"


def test_a_refuted_agreement_names_the_outputs_that_disagree_as_its_reason() -> None:
    # lanky prints a refuted fact's ``reason`` under its REFUTED line, and so
    # does ``loopty run``; without one, the line under a refuted agreement said
    # ``no witness recorded``. Only the outputs that disagree are named.
    term = ht.two_output_term()
    want = {"y": np.full(4, 2.0), "z": np.full(4, 3.0)}
    got = {"y": want["y"].copy(), "z": want["z"] + 0.5}
    fact = agreement(term, Schedule(term), got, want)
    assert fact.status.value == "refuted"
    assert fact.provenance["reason"] == (
        "z differs from the native run: difference 0.5, allowed 4e-06 (approx)"
    )

    both = {"y": want["y"] - 1.0, "z": want["z"] + 0.5}
    lines = agreement(term, Schedule(term), both, want).provenance["reason"]
    assert [line.split(" ", 1)[0] for line in lines.splitlines()] == ["y", "z"]

    same = {name: value.copy() for name, value in want.items()}
    agreed = agreement(term, Schedule(term), same, want)
    assert agreed.status.value == "tested"
    assert "reason" not in agreed.provenance


def test_agreement_on_arrays_of_different_shapes_is_a_refutation() -> None:
    term = ht.axpy_term()
    fact = agreement(
        term, Schedule(term), {"z": np.zeros(3)}, {"z": np.zeros(4)}
    )
    assert fact.status.value == "refuted"
    # The difference is recorded as infinite, which says nothing on a screen;
    # the reason lanky prints names the two shapes instead.
    assert fact.provenance["outputs"]["z"]["difference"] == float("inf")
    assert fact.provenance["reason"] == (
        "z has shape (3,), and the native run's has shape (4,)"
    )


# {{{ values that are not finite, and the bits of an exact output


@kernel
def doubled(x: Arr[Fin[n], Real], y: Arr[Fin[n], Real]):  # noqa: F821
    """``y = 2 x``, over ``Real``, which is ``approx``."""
    for i in y.dom:
        y[i] = 2.0 * x[i]


@kernel
def doubled_exact(
    x: Arr[Fin[n], Real.exact],  # noqa: F821
    y: Arr[Fin[n], Real.exact],  # noqa: F821
):
    """``y = 2 x``, compared bit for bit."""
    for i in y.dom:
        y[i] = 2.0 * x[i]


def test_an_infinity_both_runs_compute_is_agreement_not_a_crash() -> None:
    # ``|inf - inf|`` is NaN and a NaN is within no allowance, so the cell used
    # to disagree, and the search for the worst cell then took the argmin of an
    # empty selection and raised ValueError out of ``differential``.
    arrays = {"x": Arr.from_numpy(np.array([1.0, np.inf])), "y": Arr.zeros(2)}
    fact = executor().differential(doubled, Schedule(doubled), arrays)
    assert fact.status.value == "tested", fact.provenance
    detail = fact.provenance["outputs"]["y"]
    # The report is about the finite cell: 2.0 against 2.0.
    assert detail["difference"] == 0.0
    assert detail["tolerance"] == pytest.approx(
        TOLERANCE["approx"] * (2.0 + TOLERANCE_FLOOR)
    )


def test_a_nan_both_runs_compute_in_an_exact_output_is_agreement() -> None:
    # ``np.array_equal`` without ``equal_nan`` refuted this with "difference
    # nan": the kernel is right, and so is the comparison of every other bit.
    arrays = {
        "x": Arr.from_numpy(np.array([np.nan, 1.0, -0.0])),
        "y": Arr.zeros(3),
    }
    fact = executor().differential(doubled_exact, Schedule(doubled_exact), arrays)
    assert fact.status.value == "tested", fact.provenance
    assert fact.provenance["outputs"]["y"]["exactness"] == "exact"


def test_an_exact_output_is_compared_by_its_bits() -> None:
    # The docstring always said bits; the comparison was value equality, which
    # takes -0.0 for 0.0.
    term = doubled_exact.trace()
    got, want = np.array([1.0, 0.0]), np.array([1.0, -0.0])
    fact = agreement(term, Schedule(term), {"y": got}, {"y": want})
    assert fact.status.value == "refuted"
    assert fact.provenance["outputs"]["y"]["difference"] == 0.0


def test_a_finite_value_where_an_infinity_was_expected_is_a_refutation() -> None:
    # The allowance of an infinite expected value is ``eps * inf``, which used
    # to excuse any value at all, the infinity of the other sign included.
    term = doubled.trace()
    for got in (1.0, -np.inf, np.nan):
        fact = agreement(
            term,
            Schedule(term),
            {"y": np.array([2.0, got])},
            {"y": np.array([2.0, np.inf])},
        )
        assert fact.status.value == "refuted", got
        detail = fact.provenance["outputs"]["y"]
        assert detail["difference"] == np.inf and detail["tolerance"] == 0.0


def test_an_infinity_where_a_finite_value_was_expected_is_reported_as_one() -> None:
    term = doubled.trace()
    got, want = np.array([2.0, np.inf]), np.array([2.0, 4.0])
    fact = agreement(term, Schedule(term), {"y": got}, {"y": want})
    assert fact.status.value == "refuted"
    detail = fact.provenance["outputs"]["y"]
    assert detail["difference"] == np.inf
    assert detail["tolerance"] == pytest.approx(
        TOLERANCE["approx"] * (4.0 + TOLERANCE_FLOOR)
    )


def test_nans_of_either_sign_and_complex_infinities_are_judged_by_the_cell() -> None:
    # A NaN's sign is not the kernel's to keep (see disagreement), and a
    # complex cell with an infinite part used to crash the report the way a
    # real infinity did.
    negative_nan = -np.array([np.nan])[0]
    assert np.signbit(negative_nan)
    term = doubled.trace()
    for exactness_term in (term, doubled_exact.trace()):
        fact = agreement(
            exactness_term,
            Schedule(exactness_term),
            {"y": np.array([np.nan, 1.0])},
            {"y": np.array([negative_nan, 1.0])},
        )
        assert fact.status.value == "tested", exactness_term.name
    inf = complex(np.inf, 0.0)
    for got, want, verdict in (
        ([inf, 1 + 1j], [inf, 1 + 1j], "tested"),
        ([complex(np.nan, 1.0)], [complex(np.nan, 1.0)], "tested"),
        ([complex(np.nan, 1.0)], [complex(np.nan, 2.0)], "refuted"),
        ([complex(1.0, np.inf)], [1 + 0j], "refuted"),
        ([inf, 1 + 1j], [inf, 1 + 2j], "refuted"),
    ):
        fact = agreement(
            term, Schedule(term), {"y": np.array(got)}, {"y": np.array(want)}
        )
        assert fact.status.value == verdict, (got, want)
        detail = fact.provenance["outputs"]["y"]
        assert not np.isnan(detail["difference"]), (got, want)
        assert np.isfinite(detail["tolerance"]), (got, want)
    # The real mismatch is the one reported, beside an infinity that agrees.
    got, want = np.array([inf, 1 + 1j]), np.array([inf, 1 + 2j])
    fact = agreement(term, Schedule(term), {"y": got}, {"y": want})
    assert fact.provenance["outputs"]["y"]["difference"] == 1.0


def test_outputs_of_different_integer_widths_still_agree_by_value() -> None:
    # loopy writes a Nat output as int32, and the native run fills whatever the
    # caller passed; the comparison is in the type both promote to.
    term = doubled_exact.trace()
    fact = agreement(
        term,
        Schedule(term),
        {"y": np.array([1, 2], dtype=np.int32)},
        {"y": np.array([1, 2], dtype=np.int64)},
    )
    assert fact.status.value == "tested"


# }}}


def test_emit_code_returns_something_a_person_can_read() -> None:
    code = emit_code(Schedule(ht.jacobi_term()).skew("i", by="t"))
    assert "void jacobi" in code
    assert "for (int32_t t" in code


def test_a_schedule_is_run_on_the_target_it_was_built_for() -> None:
    # The opencl target belongs on a machine with a device, and a schedule built
    # for one target is not silently run on another: the transformations were
    # checked against the kernel that target produced.
    schedule = Schedule(ht.axpy_term())
    with pytest.raises(ValueError, match="was built for target"):
        LoopyExecutor(target="opencl").run(schedule)
    with pytest.raises(ValueError, match="was built for target"):
        LoopyExecutor(target="opencl").differential(None, schedule, {}, reference={})


def test_every_keyword_of_a_run_is_a_kernel_argument() -> None:
    # The backend used to be popped from the keywords as ``target=``, so a
    # kernel parameter of that name could not be passed by keyword: its array
    # was taken for the name of a target, and the run failed on numpy's "truth
    # value of an array is ambiguous". The target is the executor's option now.
    @kernel
    def shift(source: Arr[Fin[n], Real], target: Arr[Fin[n], Real]):  # noqa: F821
        for i in source.dom:
            target[i] = source[i] + 1.0

    source = np.array([1.0, 2.0, 3.0])
    target = np.zeros(3)
    out = executor().run(shift.trace(), source=source, target=target)
    assert np.array_equal(out["target"], source + 1.0)
    assert np.array_equal(target, source + 1.0)

    target = np.zeros(3)
    out = LoopyExecutor(target="c").run(shift.trace(), source=source, target=target)
    assert np.array_equal(target, source + 1.0)

    target = np.zeros(3)
    fact = LoopyExecutor(target="c").differential(
        shift, Schedule(shift), {"source": source, "target": target}
    )
    assert fact.status.value == "tested"


# {{{ the contract an argument list has to satisfy


@kernel
def csr_product(
    cnt: Arr[Fin[n], Nat],  # noqa: F821
    col: Arr[Fin[n], Fin[cnt], Fin[m]],  # noqa: F821
    val: Arr[Fin[n], Fin[cnt], Real],  # noqa: F821
    x: Arr[Fin[m], Real],  # noqa: F821
    y: Arr[Fin[n], Real],  # noqa: F821
):
    """The demo's product, so that a ragged type and a refined element are here."""
    for r in y.dom:
        y[r] = reduce_sum(val[r, j] * x[col[r, j]] for j in val.dom[r])


#: Three rows of two, zero and three entries into a vector of four.
CSR_COUNTS = [2, 0, 3]
CSR_COLUMNS = [0, 1, 0, 2, 3]
CSR_VALUES = [1.0, 2.0, 3.0, 4.0, 5.0]


def csr_arguments(counts=None, columns=None, val_counts=None, dtype=np.int64) -> dict:
    """A consistent CSR call, with one piece of it replaceable per test.

    ``dtype`` is the storage of the column array, which a test about what an
    element *is* has to be able to vary independently of its value.
    """
    counts = CSR_COUNTS if counts is None else counts
    return {
        "cnt": Arr.from_numpy(np.array(counts, dtype=np.int64)),
        "col": Arr.ragged(
            counts,
            values=CSR_COLUMNS if columns is None else columns,
            dtype=dtype,
        ),
        "val": Arr.ragged(
            counts if val_counts is None else val_counts, values=CSR_VALUES
        ),
        "x": Arr.from_numpy(np.array([1.0, 10.0, 100.0, 1000.0])),
        "y": Arr.zeros(len(counts)),
    }


def csr_want(arguments: dict) -> np.ndarray:
    """The product, by hand, from the arguments a test built."""
    offsets = arguments["val"].offsets
    columns = arguments["col"].numpy()
    values = arguments["val"].numpy()
    vector = arguments["x"].numpy()
    out = np.zeros(len(offsets) - 1)
    for row in range(len(offsets) - 1):
        for a in range(offsets[row], offsets[row + 1]):
            out[row] += values[a] * vector[columns[a]]
    return out


def test_a_consistent_ragged_call_runs_and_matches_numpy() -> None:
    arguments = csr_arguments()
    out = executor().run(csr_product.trace(), **arguments)
    assert np.allclose(out["y"], csr_want(arguments))


def test_a_ragged_argument_must_agree_with_its_counts_family() -> None:
    # ``val`` is laid out over [2, 1, 2] while its type says its rows are
    # ``cnt[r]`` long, and ``cnt`` is [2, 0, 3]. The generated loop is bounded
    # by ``cnt[r]`` and the access is flattened through the offsets, so row 1
    # would run twice into cells row 0 does not own.
    arguments = csr_arguments(val_counts=[2, 1, 2])
    with pytest.raises(ValueError, match=r"ragged argument val"):
        executor().run(csr_product.trace(), **arguments)


def test_two_ragged_arguments_over_one_family_must_share_their_offsets() -> None:
    # The term has no ``cnt`` parameter, so nothing decides between the two
    # layouts; they are flattened through one offsets argument and have to be
    # the same layout.
    col = Arr.ragged([2, 0, 3], values=[0, 1, 0, 2, 3], dtype=np.int64)
    val = Arr.ragged([3, 0, 2], values=[1.0, 2.0, 3.0, 4.0, 5.0])
    with pytest.raises(ValueError, match="col and val"):
        executor().run(
            ht.spmv_term(),
            off=np.array([0, 2, 2, 5], dtype=np.int32),
            col=col,
            val=val,
            x=np.array([1.0, 10.0, 100.0, 1000.0]),
            y=np.zeros(3),
        )


def test_explicit_offsets_must_agree_with_the_ragged_arguments_they_flatten() -> None:
    arrays = {
        "off": np.array([0, 1, 2, 5], dtype=np.int32),
        "col": Arr.ragged([2, 0, 3], values=[0, 1, 0, 2, 3], dtype=np.int64),
        "val": Arr.ragged([2, 0, 3], values=[1.0, 2.0, 3.0, 4.0, 5.0]),
        "x": np.array([1.0, 10.0, 100.0, 1000.0]),
        "y": np.zeros(3),
    }
    with pytest.raises(ValueError, match="offsets argument off"):
        executor().run(ht.spmv_term(), **arrays)


def test_an_element_outside_its_index_type_is_refused_at_the_upper_end() -> None:
    # ``col: Arr[..., Fin[m]]`` is what discharges ``x[col[r, j]]`` in bounds
    # *by type*: no isl call, no check in the generated code. A column equal to
    # ``m`` reaches the compiled code as an address past the end of ``x``.
    arguments = csr_arguments(columns=[0, 4, 0, 2, 3])
    with pytest.raises(ValueError, match=r"col\[0, 1\] is 4"):
        executor().run(csr_product.trace(), **arguments)


def test_an_element_outside_its_index_type_is_refused_at_the_lower_end() -> None:
    arguments = csr_arguments(columns=[0, 1, -1, 2, 3])
    with pytest.raises(ValueError, match=r"col\[2, 0\] is -1"):
        executor().run(csr_product.trace(), **arguments)


def test_a_valid_index_array_is_not_refused() -> None:
    # The boundary values, 0 and m - 1, are points of Fin[m] and have to pass.
    arguments = csr_arguments(columns=[0, 3, 0, 2, 3])
    out = executor().run(csr_product.trace(), **arguments)
    assert np.allclose(out["y"], csr_want(arguments))


def test_two_array_arguments_that_are_the_same_array_are_refused() -> None:
    # Dependences are computed per array name, so nothing is ever reported
    # between ``x`` and ``z`` and a schedule may run them in parallel. Called
    # with one array for both, that schedule is a race.
    shared = np.arange(4, dtype=np.float64)
    with pytest.raises(ValueError, match="share storage"):
        executor().run(
            ht.axpy_term(), a=2.0, x=shared, y=np.ones(4), z=shared
        )


def test_two_array_arguments_that_overlap_are_refused() -> None:
    buffer = np.arange(8, dtype=np.float64)
    with pytest.raises(ValueError, match="share storage"):
        executor().run(
            ht.axpy_term(), a=2.0, x=buffer[:4], y=np.ones(4), z=buffer[2:6]
        )


def test_the_differential_test_refuses_an_alias_instead_of_copying_it_away() -> None:
    # ``_copy`` gives every argument a buffer of its own, which destroys the
    # alias, so without this check the two runs agree about a program that
    # races. The check has to happen before the copies.
    shared = np.arange(4, dtype=np.float64)
    with pytest.raises(ValueError, match="share storage"):
        executor().differential(
            axpy_reference,
            Schedule(ht.axpy_term()),
            {"a": 2.0, "x": shared, "y": np.ones(4), "z": shared},
        )


def test_the_native_run_holds_its_arguments_to_the_same_contract() -> None:
    # The native run is a call too, and the typing rules' assumptions about a
    # call do not weaken because loopy is not involved.
    with pytest.raises(ValueError, match=r"col\[0, 1\] is 4"):
        csr_product(**csr_arguments(columns=[0, 4, 0, 2, 3]))
    with pytest.raises(ValueError, match=r"ragged argument val"):
        csr_product(**csr_arguments(val_counts=[2, 1, 2]))

    shared = Arr.from_numpy(np.zeros(3))
    with pytest.raises(ValueError, match="share storage"):
        shift_into(shared, shared)

    arguments = csr_arguments()
    csr_product(**arguments)
    assert np.allclose(arguments["y"].numpy(), csr_want(arguments))


@kernel
def shift_into(u: Arr[Fin[n], Real], v: Arr[Fin[n], Real]):  # noqa: F821
    """``v[i] = u[i]``: legal for disjoint arrays, a race for one array twice."""
    for i in v.dom:
        v[i] = u[i]


@kernel
def scale_counts(c: Arr[Fin[n], Nat], y: Arr[Fin[n], Real]):  # noqa: F821
    """A ``Nat`` array that is not a counts family, so the Nat rule is alone."""
    for i in y.dom:
        y[i] = 2.0 * c[i]


def test_a_negative_entry_of_a_nat_array_is_refused() -> None:
    # ``Nat`` is the other element sort that constrains a value, and a negative
    # count is what a ragged layout is least able to survive.
    with pytest.raises(ValueError, match=r"c\[1\] is -1"):
        executor().run(
            scale_counts.trace(),
            c=np.array([2, -1, 3], dtype=np.int64),
            y=np.zeros(3),
        )


def test_a_fractional_entry_of_an_index_array_is_refused() -> None:
    # A range test is two comparisons, and 1.5 passes both of them while being
    # no point of ``Fin[m]`` at all: the cast into the compiled kernel's integer
    # dtype would make it the point 1, and the native run would keep the float.
    arguments = csr_arguments(columns=[0.0, 1.5, 0.0, 2.0, 3.0], dtype=np.float64)
    with pytest.raises(ValueError, match=r"col\[0, 1\] is 1.5"):
        executor().run(csr_product.trace(), **arguments)


def test_a_nan_entry_of_an_index_array_is_refused() -> None:
    # The one a range test cannot catch: ``nan`` is neither ``< 0`` nor
    # ``>= m``, so both comparisons are false and the old check passed it.
    arguments = csr_arguments(
        columns=[0.0, float("nan"), 0.0, 2.0, 3.0], dtype=np.float64
    )
    with pytest.raises(ValueError, match=r"col\[0, 1\] is nan"):
        executor().run(csr_product.trace(), **arguments)


def test_an_integer_valued_float_index_array_is_accepted() -> None:
    # The documented rule: being a point of ``Fin[m]`` is a property of the
    # value and not of the storage. 1.0 is the point 1, the cast the executor
    # makes on the way in is exact on it, and the product is the same product.
    arguments = csr_arguments(columns=[0.0, 1.0, 0.0, 2.0, 3.0], dtype=np.float64)
    out = executor().run(csr_product.trace(), **arguments)
    assert np.allclose(out["y"], csr_want(csr_arguments()))


def test_the_native_run_reads_an_integer_valued_float_index_array_too() -> None:
    # The compiled run casts 1.0 to the index 1. The native run used to hand
    # numpy the float, which refuses it as an index, so an input the contract
    # accepts could not be run natively and so not tested differentially.
    arguments = csr_arguments(columns=[0.0, 1.0, 0.0, 2.0, 3.0], dtype=np.float64)
    csr_product(**arguments)
    assert np.allclose(arguments["y"].numpy(), csr_want(csr_arguments()))
    # The copy is the body's; the caller's array keeps its storage.
    assert arguments["col"].numpy().dtype == np.float64

    fact = executor().differential(
        csr_product,
        Schedule(csr_product),
        csr_arguments(columns=[0.0, 1.0, 0.0, 2.0, 3.0], dtype=np.float64),
    )
    assert fact.status.value == "tested"


def test_a_float_stored_integral_array_the_body_writes_is_not_copied() -> None:
    # A write has to land in the caller's buffer, so an array the body writes is
    # never swapped for a copy, whatever its storage.
    @kernel
    def count_up(c: Arr[Fin[n], Nat]):  # noqa: F821
        for i in c.dom:
            c[i] = c[i] + 1

    counts = np.array([0.0, 2.0, 5.0])
    count_up(counts)
    assert counts.tolist() == [1.0, 3.0, 6.0]


@kernel
def broadcast_at(i: Fin[n], x: Arr[Fin[n], Real], y: Arr[Fin[n], Real]):  # noqa: F821
    """``y[k] = x[i]``: a *scalar* whose index type is what puts ``x[i]`` in bounds."""
    for k in y.dom:
        y[k] = x[i]


def broadcast_arguments(index) -> dict:
    return {
        "i": index,
        "x": np.array([1.0, 10.0, 100.0, 1000.0]),
        "y": np.zeros(4),
    }


def test_a_scalar_parameter_outside_its_index_type_is_refused() -> None:
    # ``i: Fin[n]`` makes ``x[i]`` in bounds *by type*, exactly as a column
    # array's element sort does for an indirection: there is no test in the
    # generated code, so ``i = -1`` was an address in front of ``x``.
    term = broadcast_at.trace()
    with pytest.raises(ValueError, match=r"the argument i is -1"):
        executor().run(term, **broadcast_arguments(-1))
    with pytest.raises(ValueError, match=r"the argument i is 4"):
        executor().run(term, **broadcast_arguments(4))


def test_a_scalar_parameter_at_either_end_of_its_index_type_is_accepted() -> None:
    # 0 and n - 1 are points of Fin[n] and have to run.
    term = broadcast_at.trace()
    assert np.allclose(executor().run(term, **broadcast_arguments(0))["y"], 1.0)
    assert np.allclose(executor().run(term, **broadcast_arguments(3))["y"], 1000.0)


def test_a_fractional_scalar_parameter_is_refused() -> None:
    term = broadcast_at.trace()
    with pytest.raises(ValueError, match=r"the argument i is 1.5"):
        executor().run(term, **broadcast_arguments(1.5))
    with pytest.raises(ValueError, match=r"the argument i is nan"):
        executor().run(term, **broadcast_arguments(float("nan")))


def test_a_whole_float_for_an_integral_scalar_is_refused_on_both_runs() -> None:
    # ``i = 1.0`` passed the contract as a finite whole number, and neither run
    # could use it: numpy refuses a float index, and the compiled run cannot
    # pass a float to a C integer argument.
    term = broadcast_at.trace()
    x = np.array([1.0, 10.0, 100.0, 1000.0])
    with pytest.raises(ValueError, match=r"the argument i is 1\.0, a float.*int\(i\)"):
        executor().run(term, **broadcast_arguments(1.0))
    with pytest.raises(ValueError, match=r"the argument i is .*1\.0.*, a float"):
        executor().run(term, **broadcast_arguments(np.float64(1.0)))
    with pytest.raises(ValueError, match=r"the argument i is 1\.0, a float"):
        broadcast_at(1.0, x, np.zeros(4))
    # An integer runs, in either storage.
    for index in (1, np.int64(1)):
        out = executor().run(term, **broadcast_arguments(index))
        assert np.allclose(out["y"], 10.0)
        out = np.zeros(4)
        broadcast_at(index, x, out)
        assert np.allclose(out, 10.0)


def test_a_float_stored_count_outside_the_int64_range_is_refused() -> None:
    # The native run reads a float-stored array of an integral sort as int64.
    # 1e20 is a whole float that no int64 holds, and ``Nat`` has no upper end
    # to refuse it, so the conversion used to hand the body an unrelated
    # integer and the reference run computed with it.
    @kernel
    def scale_counts(c: Arr[Fin[n], Nat], y: Arr[Fin[n], Real]):  # noqa: F821
        for i in y.dom:
            y[i] = 2.0 * c[i]

    counts = np.array([1.0, 1e20, 0.0])
    with pytest.raises(ValueError, match=r"c\[1\] is 1e\+20.*64-bit integer"):
        scale_counts(counts, np.zeros(3))
    with pytest.raises(ValueError, match=r"c\[1\] is 1e\+20.*64-bit integer"):
        executor().run(scale_counts.trace(), c=counts, y=np.zeros(3))
    # Whole floats inside the range still convert, exactly, at both ends.
    from loopty.contract import element_types

    types = scale_counts.arg_types
    edge = np.nextafter(2.0**63, 0.0)
    element_types(types, {"c": np.array([edge, 0.0]), "y": np.zeros(2)})
    with pytest.raises(ValueError, match=r"64-bit integer"):
        element_types(types, {"c": np.array([2.0**63, 0.0]), "y": np.zeros(2)})
    y = np.zeros(2)
    scale_counts(np.array([edge, 3.0]), y)
    assert y.tolist() == [2.0 * edge, 6.0]


def test_the_native_run_checks_scalar_parameters_too() -> None:
    # The native run is a call, and the same claim about the call is made there.
    with pytest.raises(ValueError, match=r"the argument i is -1"):
        broadcast_at(-1, np.array([1.0, 10.0, 100.0, 1000.0]), np.zeros(4))
    out = np.zeros(4)
    broadcast_at(3, np.array([1.0, 10.0, 100.0, 1000.0]), out)
    assert np.allclose(out, 1000.0)


# }}}
