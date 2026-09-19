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

import hand_terms as ht
from loopty.executor import (
    TOLERANCE,
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


def test_agreement_on_arrays_of_different_shapes_is_a_refutation() -> None:
    term = ht.axpy_term()
    fact = agreement(
        term, Schedule(term), {"z": np.zeros(3)}, {"z": np.zeros(4)}
    )
    assert fact.status.value == "refuted"


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
        executor().run(schedule, target="opencl")
